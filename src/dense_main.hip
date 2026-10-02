#include "hip/dense.cuh"
#include "dense.hpp"
#include "model.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Frozen before GPU execution: unchanged values, ascending raw-FP32 CPU oracle,
// full comparisons, and completed events around ONLY the resident linear call.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "dense_main.hip requires strict floating-point arithmetic"
#endif

namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
constexpr double absolute_gate = 2e-4, relative_gate = 2e-4;
constexpr int repetitions = 20;
constexpr std::array tokens{1, 2, 3, 128};
constexpr int matrix_cases = 28, synthetic_cases = 20, actual_cases = 8;
constexpr int host_rejects = 26, prefix_cases = 21;
constexpr std::size_t matrix_elements = 2220648, prefix_elements = 66288;
constexpr float canary = -12345.0f;
bool cleanup_failed = false;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void check(rocblas_status status) {
    if (status != rocblas_status_success)
        throw std::runtime_error("rocBLAS status " + std::to_string(int(status)));
}
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) {
        cleanup_failed = true;
        std::fprintf(stderr, "dense cleanup: %s\n", hipGetErrorString(status));
    }
}
void cleanup(rocblas_status status) noexcept {
    if (status != rocblas_status_success) {
        cleanup_failed = true;
        std::fprintf(stderr, "dense cleanup: rocBLAS status %d\n", int(status));
    }
}
// Each constructor acquires just one resource; later failures unwind completed
// members. Destructors check every release and prohibit a successful footer.
struct Buffer {
    void* value = nullptr;
    explicit Buffer(std::size_t bytes) { check(hipMalloc(&value, bytes)); }
    ~Buffer() { if (value) cleanup(hipFree(value)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    float* data(std::size_t byte_offset = 0) const {
        return reinterpret_cast<float*>(static_cast<std::byte*>(value) + byte_offset);
    }
};
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { if (value) cleanup(hipStreamDestroy(value)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Event {
    hipEvent_t value = nullptr;
    Event() { check(hipEventCreate(&value)); }
    ~Event() { if (value) cleanup(hipEventDestroy(value)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct Handle {
    rocblas_handle value = nullptr;
    Handle() { check(rocblas_create_handle(&value)); }
    ~Handle() { if (value) cleanup(rocblas_destroy_handle(value)); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
struct Completion {
    hipStream_t stream;
    ~Completion() { cleanup(hipStreamSynchronize(stream)); }
};

void json_string(std::string_view text) {
    std::cout << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32) {
            constexpr char hex[] = "0123456789abcdef";
            std::cout << "\\u00" << hex[c / 16] << hex[c % 16];
        } else std::cout << char(c);
    }
    std::cout << '"';
}
struct Error {
    double max_abs = 0, max_bound_ratio = 0;
    std::size_t elements = 0;
    void compare(std::span<const float> actual, std::span<const float> expected) {
        require(actual.size() == expected.size(), "comparison size mismatch");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite dense result");
            const double error = std::abs(double(actual[i]) - double(expected[i]));
            const double ratio = error / (absolute_gate + relative_gate * std::abs(double(expected[i])));
            if (ratio > 1) std::cerr << "dense element=" << i << " got=" << actual[i]
                << " reference=" << expected[i] << " bound_ratio=" << ratio << '\n';
            require(ratio <= 1, "dense frozen FP32 gate");
            max_abs = std::max(max_abs, error);
            max_bound_ratio = std::max(max_bound_ratio, ratio);
            ++elements;
        }
    }
    void merge(const Error& other) {
        max_abs = std::max(max_abs, other.max_abs);
        max_bound_ratio = std::max(max_bound_ratio, other.max_bound_ratio);
        elements += other.elements;
    }
    void emit() const {
        std::cout << "{\"max_abs\":" << max_abs << ",\"max_bound_ratio\":" << max_bound_ratio
                  << ",\"elements\":" << elements << '}';
    }
};
struct Matrix {
    std::string name;
    qwen::TensorType original_type;
    int width, rows;
    std::vector<float> weights, input, reference;
    qwen::TensorView tensor{};
    void prepare_reference() {
        input.resize(std::size_t(width) * 128);
        for (int column = 0; column < 128; ++column)
            for (int k = 0; k < width; ++k)
                input[std::size_t(column) * width + k] = float((k * 17 + column * 23 + 3) % 67 - 33) / 64;
        reference.resize(std::size_t(rows) * 128);
        // F32 view of the ONCE converted values: no original-type decoder in
        // repeated projections, no activation quantization, no FMA oracle.
        qwen::matmul_f32({qwen::TensorType::F32, width, rows,
                         std::as_bytes(std::span<const float>(weights))}, input, 128, reference);
    }
};
Matrix load(const qwen::Model& model, std::string_view name, qwen::TensorType type,
            int width, int rows) {
    const auto& t = model.tensor(name);
    const std::uint64_t element_bytes = type == qwen::TensorType::F32 ? 4 : 2;
    require(t.type == type && t.rank == 2 && t.dimensions[0] == std::uint64_t(width) &&
            t.dimensions[1] == std::uint64_t(rows), "actual dense tensor type/rank/dimensions");
    require(t.strides[0] == element_bytes && t.strides[1] == std::uint64_t(width) * element_bytes &&
            t.elements == std::uint64_t(width) * rows && t.byte_size == t.elements * element_bytes,
            "actual dense tensor strides/elements/bytes");
    require(t.relative_offset % model.alignment() == 0 &&
            t.file_offset >= model.data_offset() &&
            t.file_offset - model.data_offset() == t.relative_offset &&
            t.file_offset <= model.file_size() && t.byte_size <= model.file_size() - t.file_offset,
            "actual dense tensor offsets/range");
    Matrix result{std::string(name), type, width, rows, {}, {}, {}, t};
    std::vector<std::byte> bytes(static_cast<std::size_t>(t.byte_size));
    model.read_tensor(name, bytes);
    result.weights.resize(static_cast<std::size_t>(t.elements));
    for (std::size_t i = 0; i < result.weights.size(); ++i) {
        result.weights[i] = qwen::tensor_element(type, bytes, i);
        require(std::isfinite(result.weights[i]), "nonfinite actual weight");
    }
    result.prepare_reference();
    return result;
}
Matrix synthetic(int width, int rows) {
    Matrix result{"synthetic", qwen::TensorType::F32, width, rows, {}, {}, {}, {}};
    result.weights.resize(std::size_t(width) * rows);
    for (int row = 0; row < rows; ++row)
        for (int k = 0; k < width; ++k)
            result.weights[std::size_t(row) * width + k] = float((row * 29 + k * 11 + 7) % 41 - 20) / 256;
    result.prepare_reference();
    return result;
}
struct Run {
    std::vector<float> output;
    Error error;
    double resident_ms = 0;
};
struct Fixture {
    Stream stream;
    Handle handle;
    Event begin, end;
    Fixture() { check(rocblas_set_pointer_mode(handle.value, rocblas_pointer_mode_host)); }
    ~Fixture() { cleanup(hipStreamSynchronize(stream.value)); }
    Run run(const Matrix& m, int columns, int repeats = 0) {
        const auto input = std::span<const float>(m.input).first(std::size_t(m.width) * columns);
        const auto expected = std::span<const float>(m.reference).first(std::size_t(m.rows) * columns);
        std::vector<float> guarded(expected.size() + 2);
        std::vector<float> weight_check(m.weights.size()), input_check(input.size());
        Run result;
        Buffer weights(m.weights.size() * sizeof(float)), x(input.size_bytes());
        Buffer y((expected.size() + 2) * sizeof(float));
        // Drain accepted copies on exception BEFORE their host destinations or
        // device buffers are destroyed, including failure of a later enqueue.
        Completion completion{stream.value};
        check(hipMemcpyAsync(weights.value, m.weights.data(), m.weights.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream.value));
        check(hipMemcpyAsync(x.value, input.data(), input.size_bytes(), hipMemcpyHostToDevice, stream.value));
        const qwen::DenseDeviceMatrix matrix{weights.data(), m.width, m.rows};
        auto reset = [&] {
            std::fill(guarded.begin(), guarded.end(), std::numeric_limits<float>::quiet_NaN());
            guarded.front() = guarded.back() = canary;
            check(hipMemcpyAsync(y.value, guarded.data(), guarded.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream.value));
            check(hipStreamSynchronize(stream.value));
        };
        auto validate = [&] {
            check(hipMemcpyAsync(guarded.data(), y.value, guarded.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream.value));
            check(hipStreamSynchronize(stream.value));
            require(guarded.front() == canary && guarded.back() == canary, "dense output canaries");
            Error error;
            error.compare(std::span<const float>(guarded).subspan(1, expected.size()), expected);
            return error;
        };
        reset();
        check(qwen::launch_dense_linear(handle.value, matrix, x.data(), columns, y.data(4), stream.value));
        const auto warmup = validate();
        if (repeats == 0) result.error = warmup;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            reset(); // Poison beta's old output; upload is outside the interval.
            check(hipEventRecord(begin.value, stream.value));
            check(qwen::launch_dense_linear(handle.value, matrix, x.data(), columns, y.data(4), stream.value));
            check(hipEventRecord(end.value, stream.value));
            check(hipEventSynchronize(end.value));
            float ms = 0;
            check(hipEventElapsedTime(&ms, begin.value, end.value));
            require(std::isfinite(ms) && ms > 0, "invalid completed dense event time");
            result.resident_ms += ms;
            result.error.merge(validate()); // EVERY completed interval, every element.
        }
        if (repeats) result.resident_ms /= repeats;
        result.output.assign(guarded.begin() + 1, guarded.end() - 1);
        check(hipMemcpyAsync(weight_check.data(), weights.value, weight_check.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream.value));
        check(hipMemcpyAsync(input_check.data(), x.value, input_check.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream.value));
        check(hipStreamSynchronize(stream.value));
        require(std::memcmp(weight_check.data(), m.weights.data(), weight_check.size() * sizeof(float)) == 0 &&
                std::memcmp(input_check.data(), input.data(), input.size_bytes()) == 0,
                "dense modified read-only weights/input");
        return result;
    }
    int reject_cases() {
        const std::vector<float> weights(35, .25f), input(21, .5f), output(17, canary);
        Buffer w(weights.size() * 4), x(input.size() * 4), y(output.size() * 4);
        std::vector<float> wc(weights.size()), xc(input.size()), yc(output.size());
        Completion completion{stream.value};
        check(hipMemcpyAsync(w.value, weights.data(), weights.size() * 4, hipMemcpyHostToDevice, stream.value));
        check(hipMemcpyAsync(x.value, input.data(), input.size() * 4, hipMemcpyHostToDevice, stream.value));
        check(hipMemcpyAsync(y.value, output.data(), output.size() * 4, hipMemcpyHostToDevice, stream.value));
        check(hipStreamSynchronize(stream.value));
        qwen::DenseDeviceMatrix matrix{w.data(), 7, 5};
        int rejected = 0;
        auto reject = [&](qwen::DenseDeviceMatrix a, const float* b, int n, float* c,
                          rocblas_handle h) {
            require(qwen::launch_dense_linear(h, a, b, n, c, stream.value) != rocblas_status_success,
                    "dense host argument was not rejected");
            check(hipGetLastError());
            // Read all three allocations after EVERY reject, including aliased
            // output requests. A rejection must not touch even output interiors.
            check(hipMemcpyAsync(wc.data(), w.value, wc.size() * 4, hipMemcpyDeviceToHost, stream.value));
            check(hipMemcpyAsync(xc.data(), x.value, xc.size() * 4, hipMemcpyDeviceToHost, stream.value));
            check(hipMemcpyAsync(yc.data(), y.value, yc.size() * 4, hipMemcpyDeviceToHost, stream.value));
            check(hipStreamSynchronize(stream.value));
            require(wc == weights && xc == input && yc == output, "dense rejected call wrote memory/canaries");
            ++rejected;
        };
        auto call = [&](qwen::DenseDeviceMatrix a, const float* b, int n, float* c) {
            reject(a, b, n, c, handle.value);
        };
        for (int n : {0, -1, 129, std::numeric_limits<int>::max()}) call(matrix, x.data(), n, y.data(4));
        for (int d : {0, -1, 16385, std::numeric_limits<int>::max()}) {
            auto a = matrix; a.input = d; call(a, x.data(), 3, y.data(4));
            a = matrix; a.output = d; call(a, x.data(), 3, y.data(4));
        }
        auto a = matrix; a.weights = nullptr; call(a, x.data(), 3, y.data(4));
        call(matrix, nullptr, 3, y.data(4)); call(matrix, x.data(), 3, nullptr);
        a = matrix; a.weights = w.data(1); call(a, x.data(), 3, y.data(4));
        call(matrix, x.data(1), 3, y.data(4)); call(matrix, x.data(), 3, y.data(1));
        auto* wrap = reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max() - 3);
        a = matrix; a.weights = wrap; call(a, x.data(), 3, y.data(4));
        call(matrix, wrap, 3, y.data(4)); call(matrix, x.data(), 3, wrap);
        call(matrix, x.data(), 3, x.data()); call(matrix, x.data(), 3, x.data(4));
        call(matrix, x.data(), 3, w.data()); call(matrix, x.data(), 3, w.data(4));
        reject(matrix, x.data(), 3, y.data(4), nullptr);
        require(rejected == host_rejects, "dense host rejection coverage");
        return rejected;
    }
};
struct Correctness {
    int matrices = 0, prefixes = 0, rejected = 0;
    Error error, prefix_error;
    void matrix(Fixture& fixture, const Matrix& m) {
        std::array<std::vector<float>, 3> prefix;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            auto result = fixture.run(m, tokens[i]);
            error.merge(result.error); ++matrices;
            if (i < 3) prefix[i] = std::move(result.output);
        }
        // Bounded parity accommodates different valid GEMM reduction orders.
        for (const auto pair : {std::array{1, 0}, std::array{2, 0}, std::array{2, 1}}) {
            prefix_error.compare(std::span<const float>(prefix[pair[0]]).first(prefix[pair[1]].size()),
                                 prefix[pair[1]]);
            ++prefixes;
        }
    }
    void emit(int device) const {
        require(matrices == matrix_cases && prefixes == prefix_cases && rejected == host_rejects &&
                error.elements == matrix_elements && prefix_error.elements == prefix_elements,
                "dense frozen correctness coverage");
        std::cout << "{\"kind\":\"dense_correctness\",\"device\":" << device
                  << ",\"matrix_cases\":28,\"synthetic_cases\":20,\"actual_cases\":8"
                  << ",\"host_rejects\":26,\"prefix_cases\":21,\"error\":";
        error.emit(); std::cout << ",\"prefix_error\":"; prefix_error.emit();
        std::cout << ",\"all_finite\":true,\"canaries\":true,\"immutable\":true,\"passed\":true}\n";
    }
};
void source(const char* path, const std::array<Matrix, 2>& actual) {
    std::cout << std::setprecision(17) << "{\"kind\":\"dense_source\",\"revision\":";
    json_string(CORE_REVISION);
    std::cout << ",\"dirty\":" << CORE_DIRTY << ",\"model\":"; json_string(path);
    std::cout << ",\"absolute_gate\":0.0002,\"relative_gate\":0.0002"
              << ",\"scope\":\"full unchanged F32 and BF16 tensors converted once to F32; deterministic raw-FP32 inputs; ascending FP32 CPU oracle\""
              << ",\"timing_scope\":\"completed HIP events around resident dense linear only; excludes allocation, conversion, upload, CPU reference, quantization, readback and warmup\""
              << ",\"tokens\":[1,2,3,128],\"repeats\":20,\"measurements\":16,\"protocol_rows\":20"
              << ",\"correctness_per_device\":{\"matrix_cases\":" << matrix_cases
              << ",\"synthetic_cases\":" << synthetic_cases << ",\"actual_cases\":" << actual_cases
              << ",\"host_rejects\":" << host_rejects << ",\"prefix_cases\":" << prefix_cases
              << ",\"matrix_elements\":" << matrix_elements << ",\"prefix_elements\":" << prefix_elements
              << "},\"synthetic_shapes\":[[1,3],[7,5],[3,1],[16384,3],[3,16384]],\"actual\":[";
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (i) std::cout << ',';
        const auto& m = actual[i]; const auto& t = m.tensor;
        std::cout << "{\"type\":"; json_string(qwen::type_name(m.original_type));
        std::cout << ",\"tensor\":"; json_string(m.name);
        std::cout << ",\"rank\":2,\"dimensions\":[" << m.width << ',' << m.rows
                  << "],\"strides_bytes\":[" << t.strides[0] << ',' << t.strides[1]
                  << "],\"elements\":" << t.elements << ",\"bytes\":" << t.byte_size
                  << ",\"file_offset\":" << t.file_offset << ",\"relative_offset\":" << t.relative_offset
                  << ",\"converted_type\":\"F32\",\"converted_bytes\":" << m.weights.size() * sizeof(float) << '}';
    }
    std::cout << "]}\n";
}
void execute(const char* path) {
    qwen::Model model(path);
    require(model.metadata_value("general.architecture").get<std::string>() == "qwen4exp",
            "dense requires qwen4exp model");
    const std::array actual{
        load(model, "blk.0.ssm_alpha.weight", qwen::TensorType::F32, 2560, 48),
        load(model, "blk.3.indexer.k_proj.weight", qwen::TensorType::BF16, 2560, 128)};
    const std::array synthetic_matrices{synthetic(1, 3), synthetic(7, 5), synthetic(3, 1),
                                        synthetic(16384, 3), synthetic(3, 16384)};
    int count = 0; check(hipGetDeviceCount(&count));
    require(count == 2, "dense requires exactly two GPUs");
    for (int device = 0; device < 2; ++device) {
        hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p, device));
        const std::string_view arch(p.gcnArchName);
        require(p.warpSize == 64 && (arch == "gfx906" || arch.starts_with("gfx906:")),
                "dense requires gfx906 wave64");
    }
    source(path, actual);
    // Fixed protocol order: source, correctness0/1, then device0/1 measurements,
    // F32 before BF16 and N1/2/3/128 within each type, footer after ALL cleanup.
    for (int device = 0; device < 2; ++device) {
        check(hipSetDevice(device));
        Correctness correctness;
        {
            Fixture fixture; correctness.rejected = fixture.reject_cases();
            for (const auto& m : synthetic_matrices) correctness.matrix(fixture, m);
            for (const auto& m : actual) correctness.matrix(fixture, m);
        }
        require(!cleanup_failed, "dense correctness cleanup failed");
        correctness.emit(device);
    }
    for (int device = 0; device < 2; ++device) {
        check(hipSetDevice(device));
        {
            Fixture fixture;
            for (const auto& m : actual) for (int n : tokens) {
                const auto result = fixture.run(m, n, repetitions);
                require(result.error.elements == std::size_t(m.rows) * n * repetitions,
                        "dense measured full-comparison coverage");
                require(!cleanup_failed, "dense measurement buffer cleanup failed");
                std::cout << "{\"kind\":\"dense_measurement\",\"device\":" << device << ",\"type\":";
                json_string(qwen::type_name(m.original_type));
                std::cout << ",\"tensor\":"; json_string(m.name);
                std::cout << ",\"width\":" << m.width << ",\"rows\":" << m.rows
                          << ",\"tokens\":" << n << ",\"repeats\":20,\"resident_ms\":" << result.resident_ms
                          << ",\"error\":"; result.error.emit();
                std::cout << ",\"all_finite\":true,\"canaries\":true,\"immutable\":true}\n";
            }
        }
        require(!cleanup_failed, "dense measurement cleanup failed");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::cout.exceptions(std::ios::badbit | std::ios::failbit);
        if (argc != 2) throw std::invalid_argument("usage: core-dense target.gguf");
        execute(argv[1]); // Model, host payloads and every HIP/rocBLAS owner released.
        require(!cleanup_failed, "dense cleanup failed");
        std::cout << "{\"kind\":\"dense_complete\",\"measurements\":16,\"rows\":20,\"passed\":true}\n";
        std::cout.flush();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dense: " << error.what() << '\n';
        return 1;
    }
}
