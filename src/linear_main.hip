#include "hip/linear.cuh"
#include "linear_reference.hpp"
#include "model.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

// Common-Q8 correctness, not dequantized-F32 parity or full-model performance.
// Frozen gates precede GPU execution. Timing includes only the resident linear
// launch, each interval completed and checked; no quantization/transfers/oracle.
namespace {
constexpr double absolute_gate = 2e-4, relative_gate = 2e-5;
constexpr int repetitions = 20;
constexpr std::array types{qwen::TensorType::Q4_0, qwen::TensorType::Q4_1,
    qwen::TensorType::Q5_0, qwen::TensorType::Q8_0, qwen::TensorType::Q6_K};
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) { std::cerr << hipGetErrorString(status) << '\n'; std::abort(); }
}
struct Buffer {
    void* pointer = nullptr;
    explicit Buffer(std::size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Buffer() { if (pointer) cleanup(hipFree(pointer)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as(std::size_t offset = 0) {
        return reinterpret_cast<T*>(static_cast<std::byte*>(pointer) + offset);
    }
};
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { if (value) cleanup(hipStreamDestroy(value)); }
};
struct Event {
    hipEvent_t value = nullptr;
    Event() { check(hipEventCreate(&value)); }
    ~Event() { if (value) cleanup(hipEventDestroy(value)); }
};
void json_string(std::string_view s) {
    std::cout << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32) {
            constexpr char h[] = "0123456789abcdef";
            std::cout << "\\u00" << h[c / 16] << h[c % 16];
        } else std::cout << char(c);
    }
    std::cout << '"';
}
struct Error {
    double max_abs = 0, max_bound_ratio = 0;
    void compare(std::span<const float> actual, std::span<const float> expected) {
        require(actual.size() == expected.size(), "linear comparison size");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "linear nonfinite result");
            const double error = std::abs(double(actual[i]) - double(expected[i]));
            const double ratio = error / (absolute_gate + relative_gate * std::abs(double(expected[i])));
            max_abs = std::max(max_abs, error); max_bound_ratio = std::max(max_bound_ratio, ratio);
            if (ratio > 1) std::cerr << "linear element=" << i << " got=" << actual[i]
                                   << " expected=" << expected[i] << " ratio=" << ratio << '\n';
            require(ratio <= 1, "linear common-Q8 gate");
        }
    }
};
struct Matrix {
    qwen::TensorType type;
    int width, rows;
    std::string name;
    std::vector<std::byte> bytes;
    qwen::QMatrix view() const { return {type, width, rows, bytes}; }
};
void half(std::byte* p, float x) {
    const auto bits = qwen::float_to_half(x);
    p[0] = std::byte(bits & 255); p[1] = std::byte(bits >> 8);
}
Matrix synthetic(qwen::TensorType type, int width, int rows) {
    const auto layout = qwen::type_layout(type);
    Matrix m{type, width, rows, "synthetic", std::vector<std::byte>(
        std::size_t(rows) * width / layout.block_elements * layout.block_bytes)};
    std::uint32_t random = 0x9e3779b9;
    for (auto& b : m.bytes) { random = 1664525 * random + 1013904223; b = std::byte(random >> 24); }
    for (std::size_t b = 0; b < m.bytes.size() / layout.block_bytes; ++b) {
        auto* p = m.bytes.data() + b * layout.block_bytes;
        const float scale = std::ldexp(float((b % 4) + 1), type == qwen::TensorType::Q6_K ? -19 : -12);
        half(p + (type == qwen::TensorType::Q6_K ? 208 : 0), b % 2 ? -scale : scale);
        if (type == qwen::TensorType::Q4_1) half(p + 2, float(int(b % 9) - 4) / 512);
    }
    return m;
}
std::array<Matrix, 5> actual_rows(const qwen::Model& model) {
    std::array<Matrix, 5> result{};
    for (std::size_t i = 0; i < types.size(); ++i) {
        const qwen::TensorView* selected = nullptr;
        for (const auto& t : model.tensors()) {
            if (t.type == types[i] && t.rank >= 2 && t.dimensions[0] <= 16384) {
                selected = &t; break;
            }
        }
        require(selected != nullptr, "missing actual quantized type");
        const auto& t = *selected;
        const auto layout = qwen::type_layout(t.type);
        auto& m = result[i]; m.type = t.type; m.width = int(t.dimensions[0]);
        m.rows = int(std::min<std::uint64_t>(9, t.dimensions[1])); m.name = t.name;
        m.bytes.resize(std::size_t(m.rows) * m.width / layout.block_elements * layout.block_bytes);
        model.read_slice(t.name, 0, m.bytes);
    }
    return result;
}
struct Fixture {
    Stream stream; Event begin, end;
    Buffer weights{200004}, raw{196608}, packed{55300}, output{116}, error{4};
    std::size_t quantization_cases = 0, numeric_rejects = 0, host_rejects = 0, corners = 0;
    void clear() { check(hipMemsetAsync(error.pointer, 0, 4, stream.value)); }
    int flag() {
        int value = 0;
        check(hipMemcpyAsync(&value, error.pointer, 4, hipMemcpyDeviceToHost, stream.value));
        check(hipStreamSynchronize(stream.value)); return value;
    }
    std::vector<qwen::Q8_1> quantize(const std::vector<float>& x, int width, int columns) {
        std::vector<qwen::Q8_1> expected(x.size() / 32), got(expected.size());
        qwen::quantize_q8(x, expected); clear();
        check(hipMemcpyAsync(raw.pointer, x.data(), x.size() * 4, hipMemcpyHostToDevice, stream.value));
        check(qwen::launch_quantize_q8_1(raw.as<float>(), packed.as<qwen::Q8_1>(4),
              width, columns, error.as<int>(), stream.value));
        check(hipMemcpyAsync(got.data(), packed.as<qwen::Q8_1>(4), got.size() * sizeof(got[0]),
                            hipMemcpyDeviceToHost, stream.value));
        require(flag() == 0, "Q8 producer flag");
        require(std::memcmp(got.data(), expected.data(), got.size() * sizeof(got[0])) == 0,
                "Q8 producer byte parity"); ++quantization_cases; return expected;
    }
    void edge_cases() {
        for (int test = 0; test < 8; ++test) {
            std::vector<float> x(256);
            for (std::size_t j = 0; j < x.size(); ++j) {
                if (test == 0) x[j] = 0;
                else if (test == 1) x[j] = -0.0f;
                else if (test == 2) x[j] = j % 32 == 0 ? 15.875f : float(int(j % 13) - 6) * .0625f;
                else if (test == 3) x[j] = std::ldexp(j % 32 == 0 ? 127.0f : float(int(j % 63) - 31), -24);
                else if (test == 4) x[j] = j % 2 ? -1.0003f : 1.0003f;
                else if (test == 5) x[j] = std::sin(float(j)) * .013f;
                else if (test == 6) x[j] = j % 32 == 0 ? 127.0381f : float(int(j % 11) - 5) * 1.0003f;
                else x[j] = j % 2 ? -500.0f : 500.0f;
            }
            quantize(x, 256, 1);
        }
        for (int test = 0; test < 5; ++test) {
            std::vector<float> x(256, 0);
            x[0] = test == 0 ? std::numeric_limits<float>::quiet_NaN() : test == 1 ?
                std::numeric_limits<float>::infinity() : test == 2 ?
                std::numeric_limits<float>::denorm_min() : std::numeric_limits<float>::max();
            if (test == 4) std::fill_n(x.begin(), 32, 3000.0f);
            clear(); check(hipMemcpyAsync(raw.pointer, x.data(), x.size() * 4, hipMemcpyHostToDevice, stream.value));
            check(qwen::launch_quantize_q8_1(raw.as<float>(), packed.as<qwen::Q8_1>(4),
                256, 1, error.as<int>(), stream.value));
            std::vector<qwen::Q8_1> got(8);
            check(hipMemcpyAsync(got.data(), packed.as<qwen::Q8_1>(4), got.size() * sizeof(got[0]),
                                hipMemcpyDeviceToHost, stream.value));
            require(flag() == (test < 2 ? 1 : 2), "Q8 invalid sticky bits");
            const qwen::Q8_1 zero{};
            for (const auto& b : got) require(std::memcmp(&b, &zero, sizeof(b)) == 0, "Q8 failed/zero blocks");
            ++numeric_rejects;
            quantize(std::vector<float>(256, .125f), 256, 1);
        }
        const int sticky = 1;
        check(hipMemcpyAsync(error.pointer, &sticky, 4, hipMemcpyHostToDevice, stream.value));
        check(qwen::launch_quantize_q8_1(raw.as<float>(), packed.as<qwen::Q8_1>(4),
                256, 1, error.as<int>(), stream.value));
        require(flag() == 1, "quantizer cleared sticky flag");
        auto reject = [&](hipError_t status, hipError_t expected = hipErrorInvalidValue) {
            require(status == expected, "linear pre-enqueue rejection"); ++host_rejects;
        };
        qwen::QuantizedDeviceMatrix m{qwen::TensorType::Q4_0, weights.as<std::byte>(2), 256, 9};
        auto run = [&](qwen::QuantizedDeviceMatrix a, const qwen::Q8_1* x, int n, float* y) {
            return qwen::launch_quantized_linear(a, x, n, y, stream.value);
        };
        auto* q = packed.as<qwen::Q8_1>(4); auto* y = output.as<float>(4);
        for (int n : {0, -1, 4}) reject(run(m, q, n, y));
        for (int k : {0, -32, 33, 16416}) { auto a = m; a.input = k; reject(run(a, q, 1, y)); }
        for (int r : {0, -1, 16385}) { auto a = m; a.output = r; reject(run(a, q, 1, y)); }
        for (auto t : {qwen::TensorType::F32, qwen::TensorType::F16, qwen::TensorType::BF16}) {
            auto a = m; a.type = t; reject(run(a, q, 1, y), hipErrorNotSupported);
        }
        auto a = m; a.weights = nullptr; reject(run(a, q, 1, y));
        a = m; a.weights = weights.as<std::byte>(1); reject(run(a, q, 1, y));
        a = m; a.weights = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 1);
        reject(run(a, q, 1, y));
        a = m; a.type = qwen::TensorType::Q6_K; a.input = 32; reject(run(a, q, 1, y));
        reject(run(m, nullptr, 1, y)); reject(run(m, q, 1, nullptr));
        reject(run(m, packed.as<qwen::Q8_1>(2), 1, y));
        reject(run(m, q, 1, output.as<float>(2)));
        reject(run(m, q, 1, reinterpret_cast<float*>(q)));
        reject(run(m, q, 1, weights.as<float>(4)));
        for (int width : {0, -32, 31, 16416}) reject(qwen::launch_quantize_q8_1(
            raw.as<float>(), q, width, 1, error.as<int>(), stream.value));
        reject(qwen::launch_quantize_q8_1(nullptr, q, 256, 1, error.as<int>(), stream.value));
        reject(qwen::launch_quantize_q8_1(raw.as<float>(), nullptr, 256, 1, error.as<int>(), stream.value));
        reject(qwen::launch_quantize_q8_1(raw.as<float>(), q, 256, 1, nullptr, stream.value));
        reject(qwen::launch_quantize_q8_1(raw.as<float>(), raw.as<qwen::Q8_1>(), 256, 1, error.as<int>(), stream.value));
        reject(qwen::launch_quantize_q8_1(raw.as<float>(), q, 256, 1, packed.as<int>(4), stream.value));
    }
    Error matrix(const Matrix& m, int columns, bool corner, int device, bool emit) {
        std::vector<float> x(std::size_t(m.width) * columns);
        for (std::size_t j = 0; j < x.size(); ++j) x[j] = .7f * std::sin(float(j) * .137f) + .2f * std::cos(float(j) * .071f);
        auto q = quantize(x, m.width, columns);
        if (corner) {
            for (std::size_t b = 0; b < q.size(); ++b) {
                q[b].d = qwen::float_to_half(float(b % 7 + 1) / 512);
                q[b].s = qwen::float_to_half(float(int(b % 5) - 2) / 8);
                for (int j = 0; j < 32; ++j) q[b].qs[j] = std::int8_t(int((b * 37 + j * 17) % 256) - 128);
            }
        }
        std::vector<float> expected(std::size_t(m.rows) * columns), got(expected.size() + 2, -12345.0f);
        qwen::matmul_q8_reference(m.view(), q, columns, expected);
        check(hipMemcpyAsync(weights.as<std::byte>(2), m.bytes.data(), m.bytes.size(), hipMemcpyHostToDevice, stream.value));
        check(hipMemcpyAsync(packed.as<qwen::Q8_1>(4), q.data(), q.size() * sizeof(q[0]), hipMemcpyHostToDevice, stream.value));
        check(hipMemcpyAsync(output.pointer, got.data(), got.size() * 4, hipMemcpyHostToDevice, stream.value));
        qwen::QuantizedDeviceMatrix view{m.type, weights.as<std::byte>(2), m.width, m.rows};
        auto validate = [&] {
            check(hipMemcpyAsync(got.data(), output.pointer, got.size() * 4, hipMemcpyDeviceToHost, stream.value));
            check(hipStreamSynchronize(stream.value));
            require(got.front() == -12345.0f && got.back() == -12345.0f, "linear output canary");
            Error result; result.compare(std::span<const float>(got).subspan(1, expected.size()), expected); return result;
        };
        check(qwen::launch_quantized_linear(view, packed.as<qwen::Q8_1>(4), columns, output.as<float>(4), stream.value));
        Error result = validate(); double time = 0;
        for (int repeat = 0; repeat < (emit ? repetitions : 0); ++repeat) {
            check(hipEventRecord(begin.value, stream.value));
            check(qwen::launch_quantized_linear(view, packed.as<qwen::Q8_1>(4), columns, output.as<float>(4), stream.value));
            check(hipEventRecord(end.value, stream.value)); check(hipEventSynchronize(end.value));
            float ms = 0; check(hipEventElapsedTime(&ms, begin.value, end.value));
            require(std::isfinite(ms) && ms > 0, "linear completed event time"); time += ms;
            const auto current = validate(); result.max_abs = std::max(result.max_abs, current.max_abs);
            result.max_bound_ratio = std::max(result.max_bound_ratio, current.max_bound_ratio);
        }
        std::vector<std::byte> weight_check(m.bytes.size()); std::vector<qwen::Q8_1> q_check(q.size());
        check(hipMemcpyAsync(weight_check.data(), weights.as<std::byte>(2), weight_check.size(), hipMemcpyDeviceToHost, stream.value));
        check(hipMemcpyAsync(q_check.data(), packed.as<qwen::Q8_1>(4), q_check.size() * sizeof(q[0]), hipMemcpyDeviceToHost, stream.value));
        check(hipStreamSynchronize(stream.value));
        require(weight_check == m.bytes && std::memcmp(q_check.data(), q.data(), q.size() * sizeof(q[0])) == 0, "linear immutable inputs");
        if (corner) ++corners;
        if (emit) {
            std::cout << "{\"kind\":\"linear_measurement\",\"device\":" << device << ",\"type\":"; json_string(qwen::type_name(m.type));
            std::cout << ",\"tensor\":"; json_string(m.name);
            std::cout << ",\"width\":" << m.width << ",\"rows\":" << m.rows << ",\"columns\":" << columns
                << ",\"repeats\":20,\"resident_ms\":" << time / repetitions
                << ",\"max_abs\":" << result.max_abs << ",\"max_bound_ratio\":" << result.max_bound_ratio
                << ",\"q8_byte_parity\":true,\"canary\":true,\"immutable\":true}\n";
        }
        return result;
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: core-linear target.gguf");
        qwen::Model model(argv[1]);
        require(model.metadata_value("general.architecture").get<std::string>() == "qwen4exp", "linear model architecture");
        const auto actual = actual_rows(model);
        int count = 0; check(hipGetDeviceCount(&count)); require(count == 2, "linear requires exactly two GPUs");
        for (int device = 0; device < 2; ++device) {
            hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, device));
            require(properties.warpSize == 64 && std::string_view(properties.gcnArchName).starts_with("gfx906"), "linear gfx906 wave64");
        }
        std::cout << std::setprecision(17) << "{\"kind\":\"linear_source\",\"revision\":"; json_string(CORE_REVISION);
        std::cout << ",\"dirty\":" << CORE_DIRTY << ",\"model\":"; json_string(argv[1]);
        std::cout << ",\"absolute_gate\":0.0002,\"relative_gate\":0.00002,\"scope\":\"resident common-Q8 linear; synthetic matrices and first up-to-nine unchanged actual weight rows; not full projection/inference\",\"actual\":[";
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << "{\"type\":"; json_string(qwen::type_name(actual[i].type)); std::cout << ",\"tensor\":"; json_string(actual[i].name);
            std::cout << ",\"width\":" << actual[i].width << ",\"rows\":" << actual[i].rows << '}';
        }
        std::cout << "]}\n";
        for (int device = 0; device < 2; ++device) {
            check(hipSetDevice(device));
            { Fixture f; f.edge_cases();
                for (auto type : types) {
                    const std::array<int, 6> widths = type == qwen::TensorType::Q6_K ?
                        std::array{256, 512, 1024, 1280, 2560, 16384} : std::array{32, 64, 2048, 2080, 2560, 16384};
                    constexpr std::array rows{1, 3, 9, 7, 9, 9};
                    for (std::size_t i = 0; i < widths.size(); ++i) {
                        const auto m = synthetic(type, widths[i], rows[i]);
                        for (int n : {1, 2, 3}) f.matrix(m, n, false, device, true);
                    }
                    f.matrix(synthetic(type, 256, 9), 3, true, device, false);
                }
                for (const auto& m : actual) for (int n : {1, 2, 3}) f.matrix(m, n, false, device, true);
                require(f.quantization_cases == 123 && f.numeric_rejects == 5 && f.host_rejects == 32 && f.corners == 5, "linear fixture coverage");
                std::cout << "{\"kind\":\"linear_correctness\",\"device\":" << device
                    << ",\"matrix_cases\":110,\"quantization_cases\":" << f.quantization_cases
                    << ",\"numeric_rejects\":5,\"host_rejects\":32,\"corner_cases\":5,\"passed\":true}\n";
            }
        }
        std::cout << "{\"kind\":\"linear_complete\",\"measurements\":210,\"rows\":214,\"passed\":true}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "linear: " << e.what() << '\n'; return 1; }
}
