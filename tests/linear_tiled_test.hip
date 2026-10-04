#include "hip/linear.cuh"
#include "hip/dense.cuh"
#include "linear_reference.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#ifndef CORE_REVISION
#error "compiled source revision required"
#endif
#ifndef CORE_DIRTY
#error "compiled dirty flag required"
#endif
namespace {
constexpr float sentinel = -12345.0f;
constexpr double absolute_gate = 2e-4, relative_gate = 2e-5;
void require(bool b, const char* s) { if (!b) throw std::runtime_error(s); }
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void check(rocblas_status s) { if (s != rocblas_status_success) throw std::runtime_error("rocBLAS status"); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t bytes) { check(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p && hipFree(p) != hipSuccess) std::abort(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* at(std::size_t bytes = 0) {
        return reinterpret_cast<T*>(static_cast<std::byte*>(p) + bytes);
    }
};
struct Gpu {
    hipStream_t stream = nullptr;
    rocblas_handle handle = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    int device;
    std::uint64_t cases = 0, exact_values = 0, cpu_values = 0, rejects = 0, graph_checks = 0;
    double max_abs = 0, max_ratio = 0;
    explicit Gpu(int d) : device(d) {
        check(hipSetDevice(d));
        hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, d));
        require(std::string_view(prop.gcnArchName).starts_with("gfx906") && prop.warpSize == 64, "gfx906 required");
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        try {
            check(rocblas_create_handle(&handle));
            check(hipEventCreate(&begin)); check(hipEventCreate(&end));
        } catch (...) {
            if (end) (void)hipEventDestroy(end);
            if (begin) (void)hipEventDestroy(begin);
            if (handle) (void)rocblas_destroy_handle(handle);
            (void)hipStreamDestroy(stream);
            throw;
        }
    }
    ~Gpu() {
        if (hipStreamSynchronize(stream) != hipSuccess || hipEventDestroy(end) != hipSuccess ||
            hipEventDestroy(begin) != hipSuccess || rocblas_destroy_handle(handle) != rocblas_status_success ||
            hipStreamDestroy(stream) != hipSuccess) std::abort();
    }
    template<class F> double time(F f) {
        constexpr int reps = 10;
        check(hipEventRecord(begin, stream));
        for (int i = 0; i < reps; ++i) f();
        check(hipEventRecord(end, stream)); check(hipEventSynchronize(end));
        float ms = 0; check(hipEventElapsedTime(&ms, begin, end));
        require(std::isfinite(ms) && ms > 0, "completed event interval");
        return ms / reps;
    }
    template<class F> void launch_count(F f, int columns) {
        hipGraph_t graph = nullptr;
        bool active = false;
        try {
            check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal)); active = true;
            f();
            const auto status = hipStreamEndCapture(stream, &graph); active = false; check(status);
            std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes));
            const std::size_t expected = columns <= 8 || columns % 8 == 0 ? 1 : 2;
            require(nodes == expected, "tiled logical call kernel-node count");
            check(hipGraphDestroy(graph)); graph = nullptr; ++graph_checks;
        } catch (...) {
            if (active) (void)hipStreamEndCapture(stream, &graph);
            if (graph) (void)hipGraphDestroy(graph);
            throw;
        }
    }
    void gate(float a, float b) {
        require(std::isfinite(a) && std::isfinite(b), "nonfinite oracle");
        const double error = std::abs(double(a) - b);
        const double ratio = error / (absolute_gate + relative_gate * std::abs(double(b)));
        max_abs = std::max(max_abs, error); max_ratio = std::max(max_ratio, ratio);
        require(ratio <= 1, "frozen common-Q8 gate"); ++cpu_values;
    }
    void compare(const std::vector<float>& a, const std::vector<float>& b) {
        require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                "tiled versus chronological N8 bits differ");
        require(a.front() == sentinel && a.back() == sentinel, "output canary");
        for (std::size_t i = 1; i + 1 < a.size(); ++i) require(std::isfinite(a[i]), "nonfinite tiled output");
        exact_values += a.size() - 2;
    }
    void timing(const char* type, int k, int m, int n, double a, double b, double z) {
        std::cout << "{\"kind\":\"linear_tiled_timing\",\"device\":" << device
                  << ",\"type\":\"" << type << "\",\"input\":" << k << ",\"output\":" << m
                  << ",\"columns\":" << n << ",\"old_a_ms\":" << a << ",\"tiled_b_ms\":" << b
                  << ",\"old_a2_ms\":" << z
                  << ",\"scope\":\"resident completed stream events; no quantization, transfers or model; no speed promotion\"}\n";
    }
    void quant(qwen::TensorType type, int k, int m, int n, bool benchmark) {
        const auto layout = qwen::type_layout(type);
        const std::size_t row_bytes = std::size_t(k) / layout.block_elements * layout.block_bytes;
        std::vector<std::byte> weights(row_bytes * m + 4, std::byte{0xa5});
        std::uint32_t random = 0x9e3779b9U;
        for (std::size_t i = 2; i + 2 < weights.size(); ++i) {
            random = 1664525U * random + 1013904223U; weights[i] = std::byte(random >> 24);
        }
        for (std::size_t b = 0; b < row_bytes * m / layout.block_bytes; ++b) {
            auto half = [&](int offset, float value) {
                const auto bits = qwen::float_to_half(value);
                auto* p = weights.data() + 2 + b * layout.block_bytes + offset;
                p[0] = std::byte(bits & 255); p[1] = std::byte(bits >> 8);
            };
            const float scale = std::ldexp(float(b % 4 + 1), type == qwen::TensorType::Q6_K ? -19 : -12);
            half(type == qwen::TensorType::Q6_K ? 208 : 0, b % 2 ? -scale : scale);
            if (type == qwen::TensorType::Q4_1) half(2, float(int(b % 9) - 4) / 512);
        }
        std::vector<float> raw(std::size_t(k) * n);
        for (std::size_t i = 0; i < raw.size(); ++i)
            raw[i] = (.7f * std::sin(float(i) * .137f) + .2f * std::cos(float(i) * .071f)) *
                     (i / 32 % 11 == 0 ? 1e-7f : 1.0f);
        std::vector<qwen::Q8_1> input(raw.size() / 32), produced(input.size());
        qwen::quantize_q8(raw, input);
        std::vector<float> a(std::size_t(m) * n + 2, sentinel), b(a);
        Buffer dw(weights.size()), dx(raw.size() * sizeof(float)), dq(input.size() * sizeof(input[0]) + 8);
        Buffer da(a.size() * sizeof(float)), db(b.size() * sizeof(float)), flag(sizeof(int));
        check(hipMemcpyAsync(dw.p, weights.data(), weights.size(), hipMemcpyHostToDevice, stream));
        check(hipMemcpyAsync(dx.p, raw.data(), raw.size() * sizeof(float), hipMemcpyHostToDevice, stream));
        check(hipMemsetAsync(dq.p, 0xa5, input.size() * sizeof(input[0]) + 8, stream));
        check(hipMemsetAsync(flag.p, 0, sizeof(int), stream));
        check(qwen::launch_quantize_q8_1(dx.at<float>(), dq.at<qwen::Q8_1>(4), k, n, flag.at<int>(), stream));
        int error = -1;
        check(hipMemcpyAsync(produced.data(), dq.at<qwen::Q8_1>(4), input.size() * sizeof(input[0]), hipMemcpyDeviceToHost, stream));
        check(hipMemcpyAsync(&error, flag.p, sizeof(int), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        require(error == 0 && std::memcmp(input.data(), produced.data(), input.size() * sizeof(input[0])) == 0,
                "logical-wide quantizer differs from CPU canonical Q8");
        const qwen::QuantizedDeviceMatrix matrix{type, dw.at<std::byte>(2), k, m};
        auto old = [&] {
            for (int first = 0; first < n; first += 8)
                check(qwen::launch_quantized_linear_short(matrix, dq.at<qwen::Q8_1>(4) + std::size_t(first) * (k / 32),
                    std::min(8, n - first), da.at<float>(4) + std::size_t(first) * m, stream));
        };
        auto tiled = [&] {
            check(qwen::launch_quantized_linear_tiled(matrix, dq.at<qwen::Q8_1>(4), n, db.at<float>(4), stream));
        };
        for (int pattern = 0; pattern < 2; ++pattern) {
            if (pattern) {
                for (std::size_t i = 0; i < input.size(); ++i) {
                    input[i].d = qwen::float_to_half(float(i % 7 + 1) / 512);
                    input[i].s = qwen::float_to_half(float(int(i % 5) - 2) / 8);
                    for (int j = 0; j < 32; ++j) input[i].qs[j] = std::int8_t(int((i * 37 + j * 17) % 256) - 128);
                }
                check(hipMemcpyAsync(dq.at<qwen::Q8_1>(4), input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice, stream));
            }
            std::fill(a.begin(), a.end(), sentinel); std::fill(b.begin(), b.end(), sentinel);
            check(hipMemcpyAsync(da.p, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice, stream));
            check(hipMemcpyAsync(db.p, b.data(), b.size() * sizeof(float), hipMemcpyHostToDevice, stream));
            old(); tiled();
            check(hipMemcpyAsync(a.data(), da.p, a.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
            check(hipMemcpyAsync(b.data(), db.p, b.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
            check(hipStreamSynchronize(stream)); compare(a, b);
            if (pattern == 0) launch_count(tiled, n);
            // Five spread CPU rows (all rows when m<=5), across every column.
            const int samples = std::min(m, 5);
            for (int j = 0; j < samples; ++j) {
                const int row = samples == 1 ? 0 : int(std::int64_t(j) * (m - 1) / (samples - 1));
                for (int c = 0; c < n; ++c) {
                    float ref = 0;
                    qwen::matmul_q8_reference({type, k, 1, std::span<const std::byte>(weights).subspan(2 + row * row_bytes, row_bytes)},
                        std::span<const qwen::Q8_1>(input).subspan(std::size_t(c) * (k / 32), k / 32), 1, {&ref, 1});
                    gate(b[1 + std::size_t(c) * m + row], ref);
                }
            }
            if (benchmark && pattern == 0) {
                const auto expected = b;
                const double ta = time(old), tb = time(tiled), tz = time(old);
                const char* label = type == qwen::TensorType::Q4_0 ? "Q4_0" : type == qwen::TensorType::Q4_1 ? "Q4_1" :
                    type == qwen::TensorType::Q5_0 ? "Q5_0" : type == qwen::TensorType::Q8_0 ? "Q8_0" : "Q6_K";
                timing(label, k, m, n, ta, tb, tz);
                check(hipMemcpyAsync(a.data(), da.p, a.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
                check(hipMemcpyAsync(b.data(), db.p, b.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
                check(hipStreamSynchronize(stream)); compare(a, expected); compare(b, expected);
            }
        }
        std::vector<std::byte> qbytes(input.size() * sizeof(input[0]) + 8), unchanged(weights.size());
        check(hipMemcpyAsync(qbytes.data(), dq.p, qbytes.size(), hipMemcpyDeviceToHost, stream));
        check(hipMemcpyAsync(unchanged.data(), dw.p, unchanged.size(), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream)); require(unchanged == weights, "weights/canaries modified");
        require(std::memcmp(qbytes.data() + 4, input.data(), input.size() * sizeof(input[0])) == 0, "Q8 modified");
        for (std::size_t i = 0; i < 4; ++i)
            require(qbytes[i] == std::byte{0xa5} && qbytes[qbytes.size() - 1 - i] == std::byte{0xa5}, "Q8 canary");
        ++cases;
    }
    void dense(int k, int m, int n, bool benchmark) {
        std::vector<float> weights(std::size_t(k) * m + 4, sentinel), input(std::size_t(k) * n + 4, sentinel);
        std::vector<float> a(std::size_t(m) * n + 2, sentinel), b(a);
        Buffer dw(weights.size() * 4), dx(input.size() * 4), da(a.size() * 4), db(b.size() * 4);
        for (int pattern = 0; pattern < 2; ++pattern) {
            for (std::size_t i = 2; i + 2 < weights.size(); ++i)
                weights[i] = pattern ? std::sin(float(i) * .13f) / 8 : float(int((i * 7 + 3) % 15) - 7) / 64;
            for (std::size_t i = 2; i + 2 < input.size(); ++i)
                input[i] = pattern ? std::cos(float(i) * .07f) / 8 : float(int((i * 11 + 5) % 15) - 7) / 64;
            check(hipMemcpyAsync(dw.p, weights.data(), weights.size() * 4, hipMemcpyHostToDevice, stream));
            check(hipMemcpyAsync(dx.p, input.data(), input.size() * 4, hipMemcpyHostToDevice, stream));
            std::fill(a.begin(), a.end(), sentinel); std::fill(b.begin(), b.end(), sentinel);
            check(hipMemcpyAsync(da.p, a.data(), a.size() * 4, hipMemcpyHostToDevice, stream));
            check(hipMemcpyAsync(db.p, b.data(), b.size() * 4, hipMemcpyHostToDevice, stream));
            const qwen::DenseDeviceMatrix matrix{dw.at<float>(8), k, m};
            auto old = [&] {
                for (int first = 0; first < n; first += 8)
                    check(qwen::launch_dense_linear(handle, matrix, dx.at<float>(8) + std::size_t(first) * k,
                        std::min(8, n - first), da.at<float>(4) + std::size_t(first) * m, stream));
            };
            auto tiled = [&] { check(qwen::launch_dense_linear_tiled(handle, matrix, dx.at<float>(8), n, db.at<float>(4), stream)); };
            old(); tiled();
            check(hipMemcpyAsync(a.data(), da.p, a.size() * 4, hipMemcpyDeviceToHost, stream));
            check(hipMemcpyAsync(b.data(), db.p, b.size() * 4, hipMemcpyDeviceToHost, stream));
            check(hipStreamSynchronize(stream)); compare(a, b);
            if (pattern == 0) {
                launch_count(tiled, n);
                const int samples = std::min(m, 5);
                for (int j = 0; j < samples; ++j) {
                    const int row = samples == 1 ? 0 : int(std::int64_t(j) * (m - 1) / (samples - 1));
                    for (int c = 0; c < n; ++c) {
                        float ref = 0;
                        for (int i = 0; i < k; ++i) ref += weights[2 + std::size_t(row) * k + i] * input[2 + std::size_t(c) * k + i];
                        require(b[1 + std::size_t(c) * m + row] == ref, "dyadic CPU dense oracle"); ++cpu_values;
                    }
                }
            }
            if (benchmark && pattern == 0) {
                const auto expected = b;
                const double ta = time(old), tb = time(tiled), tz = time(old);
                timing("F32", k, m, n, ta, tb, tz);
                check(hipMemcpyAsync(a.data(), da.p, a.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
                check(hipMemcpyAsync(b.data(), db.p, b.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
                check(hipStreamSynchronize(stream)); compare(a, expected); compare(b, expected);
            }
            std::vector<float> wcopy(weights.size()), xcopy(input.size());
            check(hipMemcpyAsync(wcopy.data(), dw.p, wcopy.size() * 4, hipMemcpyDeviceToHost, stream));
            check(hipMemcpyAsync(xcopy.data(), dx.p, xcopy.size() * 4, hipMemcpyDeviceToHost, stream));
            check(hipStreamSynchronize(stream));
            require(weights == wcopy && input == xcopy, "dense input/canaries modified");
        }
        ++cases;
    }
    void invalid() {
        Buffer weights(4096), input(131104), output(4096);
        check(hipMemsetAsync(output.p, 0x5a, 4096, stream));
        const auto fake_stream = reinterpret_cast<hipStream_t>(1);
        qwen::QuantizedDeviceMatrix base{qwen::TensorType::Q4_0, weights.at<std::byte>(2), 256, 3};
        auto reject = [&](hipError_t s, hipError_t expected = hipErrorInvalidValue) {
            require(s == expected && hipPeekAtLastError() == hipSuccess, "quant host rejection enqueued work"); ++rejects;
        };
        for (int n : {-1, 0, 129, std::numeric_limits<int>::max()})
            reject(qwen::launch_quantized_linear_tiled(base, input.at<qwen::Q8_1>(4), n, output.at<float>(4), fake_stream));
        reject(qwen::launch_quantized_linear_tiled(base, input.at<qwen::Q8_1>(2), 128, output.at<float>(4), fake_stream));
        reject(qwen::launch_quantized_linear_tiled(base, input.at<qwen::Q8_1>(4), 128,
            input.at<float>(4 + 128 * 8 * sizeof(qwen::Q8_1) - 4), fake_stream));
        auto bad = base; bad.weights = nullptr;
        reject(qwen::launch_quantized_linear_tiled(bad, input.at<qwen::Q8_1>(4), 128, output.at<float>(4), fake_stream));
        bad = base; bad.type = qwen::TensorType::F32;
        reject(qwen::launch_quantized_linear_tiled(bad, input.at<qwen::Q8_1>(4), 128, output.at<float>(4), fake_stream), hipErrorNotSupported);
        bad = base; bad.input = 33;
        reject(qwen::launch_quantized_linear_tiled(bad, input.at<qwen::Q8_1>(4), 128, output.at<float>(4), fake_stream));
        reject(qwen::launch_quantized_linear_tiled(base, input.at<qwen::Q8_1>(4), 128,
            reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max() - 3), fake_stream));
        // Original short and N1..3 limits still reject wider requests.
        reject(qwen::launch_quantized_linear_short(base, input.at<qwen::Q8_1>(4), 9, output.at<float>(4), fake_stream));
        reject(qwen::launch_quantized_linear(base, input.at<qwen::Q8_1>(4), 4, output.at<float>(4), fake_stream));
        check(rocblas_set_stream(handle, stream)); check(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        auto dense_reject = [&](qwen::DenseDeviceMatrix matrix, const float* x, int n, float* y, rocblas_status expected) {
            require(qwen::launch_dense_linear_tiled(handle, matrix, x, n, y, fake_stream) == expected, "dense host rejection");
            hipStream_t observed = nullptr; rocblas_pointer_mode mode{};
            check(rocblas_get_stream(handle, &observed)); check(rocblas_get_pointer_mode(handle, &mode));
            require(observed == stream && mode == rocblas_pointer_mode_device, "rejection mutated handle"); ++rejects;
        };
        const qwen::DenseDeviceMatrix dense_matrix{weights.at<float>(8), 256, 3};
        require(qwen::launch_dense_linear_tiled(nullptr, dense_matrix, input.at<float>(8), 128,
            output.at<float>(4), fake_stream) == rocblas_status_invalid_handle, "null dense handle"); ++rejects;
        for (int n : {-1, 0, 129, std::numeric_limits<int>::max()})
            dense_reject(dense_matrix, input.at<float>(8), n, output.at<float>(4), rocblas_status_invalid_size);
        dense_reject({weights.at<float>(8), 255, 3}, input.at<float>(8), 128, output.at<float>(4), rocblas_status_invalid_size);
        dense_reject(dense_matrix, input.at<float>(4), 128, output.at<float>(4), rocblas_status_invalid_pointer);
        dense_reject({weights.at<float>(4), 256, 3}, input.at<float>(8), 128, output.at<float>(4), rocblas_status_invalid_pointer);
        dense_reject(dense_matrix, input.at<float>(8), 128, input.at<float>(8 + 128 * 256 * 4 - 4), rocblas_status_invalid_pointer);
        std::vector<std::byte> unchanged(4096);
        check(hipMemcpyAsync(unchanged.data(), output.p, unchanged.size(), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        require(std::all_of(unchanged.begin(), unchanged.end(), [](auto b) { return b == std::byte{0x5a}; }), "host rejection wrote output");
        // Read-only overlap remains legal across the entire logical N128 range.
        check(hipMemsetAsync(input.p, 0, 131104, stream));
        auto verify_zero = [&] {
            check(hipMemcpyAsync(unchanged.data(), output.p, unchanged.size(), hipMemcpyDeviceToHost, stream));
            check(hipStreamSynchronize(stream));
            for (std::size_t i = 0; i < unchanged.size(); ++i)
                require(unchanged[i] == (i >= 4 && i < 4 + 128 * 3 * 4 ? std::byte{0} : std::byte{0x5a}),
                        "read-only alias output/canary");
            ++cases;
        };
        check(qwen::launch_quantized_linear_tiled({qwen::TensorType::Q4_0, input.at<std::byte>(4), 256, 3},
            input.at<qwen::Q8_1>(4), 128, output.at<float>(4), stream));
        verify_zero();
        check(hipMemsetAsync(output.p, 0x5a, 4096, stream));
        check(qwen::launch_dense_linear_tiled(handle, {input.at<float>(8), 256, 3},
            input.at<float>(8), 128, output.at<float>(4), stream));
        verify_zero();
    }
};
} // namespace
int main() {
    try {
        std::cout << std::setprecision(17);
        int count = 0; check(hipGetDeviceCount(&count)); require(count == 2, "two GPUs required");
        std::cout << "{\"kind\":\"linear_tiled_source\",\"revision\":\"" << CORE_REVISION
                  << "\",\"dirty\":" << (CORE_DIRTY ? "true" : "false")
                  << ",\"model_free\":true,\"max_logical_columns\":128,\"physical_tile_columns\":8"
                     ",\"cpu_reference_scope\":\"up to five spread rows, all columns; Q8 both patterns, dense dyadic pattern\""
                     ",\"common_q8_absolute_gate\":0.0002,\"common_q8_relative_gate\":0.00002,\"performance_qualified\":false}\n";
        for (int device = 0; device < 2; ++device) {
            Gpu f(device); f.invalid();
            for (auto type : {qwen::TensorType::Q4_0, qwen::TensorType::Q4_1, qwen::TensorType::Q5_0,
                              qwen::TensorType::Q8_0, qwen::TensorType::Q6_K}) {
                for (int n : {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128})
                    f.quant(type, 256, 7, n, false);
                for (int k : {32, 64, 640, 2560, 16384}) {
                    if (k % qwen::type_layout(type).block_elements) continue;
                    for (int n : {9, 17, 128}) f.quant(type, k, 3, n, false);
                }
            }
            for (int n : {9, 16, 32, 128}) {
                f.quant(qwen::TensorType::Q4_0, 2560, 640, n, true);
                f.quant(qwen::TensorType::Q4_0, 640, 2560, n, true);
                f.quant(qwen::TensorType::Q4_1, 640, 2560, n, true);
                f.quant(qwen::TensorType::Q6_K, 2560, 248320, n, true);
                if (n >= 16) {
                    f.quant(qwen::TensorType::Q5_0, 2560, 10240, n, true);
                    f.quant(qwen::TensorType::Q8_0, 2560, 10240, n, true);
                }
            }
            for (int k : {2, 64, 128, 130, 320, 2560, 10240, 16384})
                for (int n : {1, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128}) f.dense(k, 3, n, false);
            for (int n : {9, 16, 32, 128}) {
                f.dense(10240, 4, n, true); f.dense(2560, 512, n, true); f.dense(2560, 2560, n, true);
                if (n >= 16) {
                    f.dense(10240, 16, n, true); f.dense(2560, 48, n, true); f.dense(2560, 128, n, true);
                }
            }
            std::cout << "{\"kind\":\"linear_tiled_device\",\"device\":" << device << ",\"cases\":" << f.cases
                      << ",\"exact_values\":" << f.exact_values << ",\"cpu_values\":" << f.cpu_values
                      << ",\"host_rejects\":" << f.rejects << ",\"max_abs\":" << f.max_abs
                      << ",\"graph_launch_checks\":" << f.graph_checks
                      << ",\"max_bound_ratio\":" << f.max_ratio << ",\"passed\":true}\n";
        }
        std::cout << "{\"kind\":\"linear_tiled_complete\",\"devices\":2,\"cleanup_complete\":true,\"passed\":true}\n";
        std::cout.flush(); require(bool(std::cout), "stdout failed");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "linear_tiled: " << e.what() << '\n';
        return 1;
    }
}
