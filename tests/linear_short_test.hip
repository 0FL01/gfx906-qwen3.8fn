#include "hip/linear.cuh"
#include "linear_reference.hpp"
#include "model.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

// Frozen common-Q8 CPU gate plus exact old-N1 arithmetic parity. Slicing is
// diagnostic-only here; the short API executes one multi-column kernel.
namespace {
constexpr double abs_gate = 2e-4, rel_gate = 2e-5;
constexpr float canary = -12345.0f;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void cleanup(hipError_t s) { if (s != hipSuccess) std::abort(); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t bytes) { check(hipMalloc(&p, bytes)); }
    ~Buffer() { cleanup(hipFree(p)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* at(std::size_t offset = 0) {
        return reinterpret_cast<T*>(static_cast<std::byte*>(p) + offset);
    }
};
struct Fixture {
    hipStream_t stream{};
    std::size_t cases = 0, rejects = 0, elements = 0, oracle_elements = 0;
    double max_abs = 0, max_ratio = 0;
    Fixture() { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); }
    ~Fixture() { cleanup(hipStreamSynchronize(stream)); cleanup(hipStreamDestroy(stream)); }
    void compare(float got, float ref) {
        require(std::isfinite(got) && std::isfinite(ref), "nonfinite common-Q8 result");
        const double error = std::abs(double(got) - ref);
        const double ratio = error / (abs_gate + rel_gate * std::abs(double(ref)));
        max_abs = std::max(max_abs, error); max_ratio = std::max(max_ratio, ratio);
        require(ratio <= 1, "common-Q8 gate 2e-4 + 2e-5*abs(ref)"); ++oracle_elements;
    }
    void invalid() {
        Buffer w(1024), q(1024), y(1024);
        check(hipMemsetAsync(y.p, 0x5a, 1024, stream));
        qwen::QuantizedDeviceMatrix base{qwen::TensorType::Q4_0, w.at<std::byte>(2), 256, 3};
        // An invalid stream must never be touched by host-side rejection.
        auto run = [&](qwen::QuantizedDeviceMatrix m, const qwen::Q8_1* x, int n, float* out) {
            return qwen::launch_quantized_linear_short(m, x, n, out, reinterpret_cast<hipStream_t>(1));
        };
        auto reject = [&](hipError_t got, hipError_t expected = hipErrorInvalidValue) {
            require(got == expected, "checked short API rejection");
            require(hipPeekAtLastError() == hipSuccess, "reject touched HIP error state"); ++rejects;
        };
        auto* x = q.at<qwen::Q8_1>(4); auto* out = y.at<float>(4);
        for (int n : {-1, 0, 9, 128, std::numeric_limits<int>::max()}) reject(run(base, x, n, out));
        for (int n : {4, 5, 6, 7, 8}) reject(qwen::launch_quantized_linear(base, x, n, out, reinterpret_cast<hipStream_t>(1)));
        for (int k : {-32, 0, 33, 16416, std::numeric_limits<int>::max()}) {
            auto m = base; m.input = k; reject(run(m, x, 8, out));
        }
        for (int r : {-1, 0, 16385, 248319, 248320, 248321}) {
            auto m = base; m.output = r; reject(run(m, x, 8, out));
        }
        for (auto type : {qwen::TensorType::F32, qwen::TensorType::F16, qwen::TensorType::BF16}) {
            auto m = base; m.type = type; reject(run(m, x, 8, out), hipErrorNotSupported);
        }
        auto m = base; m.type = qwen::TensorType::Q6_K; m.input = 32; reject(run(m, x, 8, out));
        for (auto shape : {std::array{256, 248320}, std::array{2560, 248319}, std::array{2560, 248321}}) {
            m = {qwen::TensorType::Q6_K, base.weights, shape[0], shape[1]}; reject(run(m, x, 8, out));
        }
        m = {qwen::TensorType::Q8_0, base.weights, 2560, 248320}; reject(run(m, x, 8, out));
        m = base; m.weights = nullptr; reject(run(m, x, 8, out));
        m.weights = w.at<std::byte>(1); reject(run(m, x, 8, out));
        m.weights = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 1); reject(run(m, x, 8, out));
        reject(run(base, nullptr, 8, out)); reject(run(base, x, 8, nullptr));
        reject(run(base, q.at<qwen::Q8_1>(2), 8, out)); reject(run(base, x, 8, y.at<float>(2)));
        reject(run(base, reinterpret_cast<qwen::Q8_1*>(std::numeric_limits<std::uintptr_t>::max() - 3), 8, out));
        reject(run(base, x, 8, reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max() - 3)));
        reject(run(base, x, 8, q.at<float>(4))); reject(run(base, x, 8, w.at<float>(4)));
        // Overlap at the last bytes, not just identical starts; ranges include N8.
        reject(run(base, x, 8, q.at<float>(4 + 8 * 8 * sizeof(qwen::Q8_1) - 4)));
        reject(run(base, x, 8, w.at<float>(2 + 3 * 8 * 18 - 2)));
        std::array<std::byte, 1024> bytes{};
        check(hipMemcpyAsync(bytes.data(), y.p, bytes.size(), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        require(std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == std::byte{0x5a}; }), "reject wrote output");
        // Read-only ranges are permitted to alias. Zero Q4/Q8 gives finite zero.
        check(hipMemsetAsync(q.p, 0, 1024, stream));
        check(qwen::launch_quantized_linear_short({qwen::TensorType::Q4_0, x, 32, 1}, x, 8, out, stream));
        std::array<float, 8> zeros{};
        check(hipMemcpyAsync(zeros.data(), out, sizeof(zeros), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        require(std::all_of(zeros.begin(), zeros.end(), [](float v) { return v == 0 && std::isfinite(v); }), "read-only alias");
    }
    void run(qwen::TensorType type, int k, int m, int n, bool head = false) {
        const auto layout = qwen::type_layout(type);
        const std::size_t weight_bytes = std::size_t(m) * (k / layout.block_elements) * layout.block_bytes;
        std::vector<std::byte> w(weight_bytes + 4, std::byte{0xa5});
        std::uint32_t random = 0x9e3779b9;
        for (std::size_t j = 2; j < weight_bytes + 2; ++j) {
            random = 1664525 * random + 1013904223; w[j] = std::byte(random >> 24);
        }
        for (std::size_t b = 0; b < weight_bytes / layout.block_bytes; ++b) {
            const float scale = std::ldexp(float(b % 4 + 1), type == qwen::TensorType::Q6_K ? -19 : -12);
            auto half = [&](int offset, float v) {
                const auto bits = qwen::float_to_half(v);
                auto* p = w.data() + 2 + b * layout.block_bytes + offset;
                p[0] = std::byte(bits & 255); p[1] = std::byte(bits >> 8);
            };
            half(type == qwen::TensorType::Q6_K ? 208 : 0, b % 2 ? -scale : scale);
            if (type == qwen::TensorType::Q4_1) half(2, float(int(b % 9) - 4) / 512);
        }
        std::vector<float> raw(std::size_t(k) * n);
        for (std::size_t j = 0; j < raw.size(); ++j)
            raw[j] = .7f * std::sin(float(j) * .137f) + .2f * std::cos(float(j) * .071f);
        std::vector<qwen::Q8_1> q(raw.size() / 32), produced(q.size()); qwen::quantize_q8(raw, q);
        std::vector<float> y(std::size_t(m) * n + 2, canary), single(y.size(), canary), previous;
        Buffer dw(w.size()), dx(raw.size() * 4), dq(q.size() * sizeof(q[0]) + 8), dy(y.size() * 4), ds(y.size() * 4), flag(4);
        check(hipMemcpyAsync(dw.p, w.data(), w.size(), hipMemcpyHostToDevice, stream));
        check(hipMemcpyAsync(dx.p, raw.data(), raw.size() * 4, hipMemcpyHostToDevice, stream));
        check(hipMemsetAsync(dq.p, 0xa5, q.size() * sizeof(q[0]) + 8, stream));
        check(hipMemsetAsync(flag.p, 0, 4, stream));
        check(qwen::launch_quantize_q8_1(dx.at<float>(), dq.at<qwen::Q8_1>(4), k, n, flag.at<int>(), stream));
        int error = -1;
        check(hipMemcpyAsync(produced.data(), dq.at<qwen::Q8_1>(4), q.size() * sizeof(q[0]), hipMemcpyDeviceToHost, stream));
        check(hipMemcpyAsync(&error, flag.p, 4, hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        require(error == 0 && std::memcmp(q.data(), produced.data(), q.size() * sizeof(q[0])) == 0, "36-byte raw-sum Q8 producer parity");
        const qwen::QuantizedDeviceMatrix matrix{type, dw.at<std::byte>(2), k, m};
        for (int generation = 0; generation < 2; ++generation) {
            if (generation) {
                // Deliberately independent raw s plus -128 codes stresses offset
                // corrections, half products and in-place workspace reuse.
                for (std::size_t b = 0; b < q.size(); ++b) {
                    q[b].d = qwen::float_to_half(float(b % 7 + 1) / 512);
                    q[b].s = qwen::float_to_half(float(int(b % 5) - 2) / 8);
                    for (int j = 0; j < 32; ++j) q[b].qs[j] = std::int8_t(int((b * 37 + j * 17) % 256) - 128);
                }
                check(hipMemcpyAsync(dq.at<qwen::Q8_1>(4), q.data(), q.size() * sizeof(q[0]), hipMemcpyHostToDevice, stream));
            }
            for (int repeat = 0; repeat < 2; ++repeat) {
                std::fill(y.begin(), y.end(), canary);
                check(hipMemcpyAsync(dy.p, y.data(), y.size() * 4, hipMemcpyHostToDevice, stream));
                check(qwen::launch_quantized_linear_short(matrix, dq.at<qwen::Q8_1>(4), n, dy.at<float>(4), stream));
                check(hipMemcpyAsync(y.data(), dy.p, y.size() * 4, hipMemcpyDeviceToHost, stream));
                check(hipStreamSynchronize(stream));
                require(y.front() == canary && y.back() == canary, "short output canaries");
                if (repeat) require(std::memcmp(y.data(), previous.data(), y.size() * 4) == 0, "repeat/reuse drift");
                else previous = y;
            }
            check(hipMemcpyAsync(ds.p, single.data(), single.size() * 4, hipMemcpyHostToDevice, stream));
            for (int c = 0; c < n; ++c)
                check(qwen::launch_quantized_linear(matrix, dq.at<qwen::Q8_1>(4) + c * (k / 32), 1, ds.at<float>(4) + c * m, stream));
            check(hipMemcpyAsync(single.data(), ds.p, single.size() * 4, hipMemcpyDeviceToHost, stream));
            check(hipStreamSynchronize(stream));
            require(single.front() == canary && single.back() == canary, "old N1 output canaries");
            require(std::memcmp(y.data(), single.data(), y.size() * 4) == 0, "short versus qualified old N1 arithmetic");
            for (std::size_t j = 1; j + 1 < y.size(); ++j) { require(std::isfinite(y[j]), "nonfinite output"); ++elements; }
            if (!head) {
                std::vector<float> ref(std::size_t(m) * n);
                for (int c = 0; c < n; ++c)
                    qwen::matmul_q8_reference({type, k, m, std::span<const std::byte>(w).subspan(2, weight_bytes)},
                        std::span<const qwen::Q8_1>(q).subspan(std::size_t(c) * (k / 32), k / 32), 1,
                        std::span<float>(ref).subspan(std::size_t(c) * m, m));
                for (std::size_t j = 0; j < ref.size(); ++j) compare(y[1 + j], ref[j]);
            } else {
                // Sole real head shape: full-vocabulary finite/exact-N1 coverage;
                // CPU common-Q8 oracle covers 24 spread rows, not full vocabulary.
                const std::size_t row_bytes = weight_bytes / m;
                for (int s = 0; s < 24; ++s) {
                    const int r = int(std::int64_t(s) * (m - 1) / 23);
                    for (int c = 0; c < n; ++c) {
                        float ref = 0;
                        qwen::matmul_q8_reference({type, k, 1, std::span<const std::byte>(w).subspan(2 + r * row_bytes, row_bytes)},
                            std::span<const qwen::Q8_1>(q).subspan(std::size_t(c) * (k / 32), k / 32), 1, {&ref, 1});
                        compare(y[1 + std::size_t(c) * m + r], ref);
                    }
                }
            }
            std::vector<std::byte> packed(q.size() * sizeof(q[0]) + 8);
            check(hipMemcpyAsync(packed.data(), dq.p, packed.size(), hipMemcpyDeviceToHost, stream));
            check(hipStreamSynchronize(stream));
            require(std::memcmp(packed.data() + 4, q.data(), q.size() * sizeof(q[0])) == 0, "short modified Q8 input");
            for (std::size_t j : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(3), packed.size()-4, packed.size()-3, packed.size()-2, packed.size()-1})
                require(packed[j] == std::byte{0xa5}, "Q8 canary");
        }
        std::vector<std::byte> immutable(w.size());
        check(hipMemcpyAsync(immutable.data(), dw.p, immutable.size(), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream)); require(w == immutable, "weight bytes/canaries changed"); ++cases;
    }
};
}
int main() {
    try {
        int count = 0; check(hipGetDeviceCount(&count)); require(count == 2, "requires both gfx906 GPUs");
        std::printf("{\"kind\":\"linear_short_source\",\"revision\":\"%s\",\"dirty\":%d,\"absolute_gate\":0.0002,\"relative_gate\":0.00002,\"scope\":\"synthetic common-Q8 and exact old-N1, not PP acceptance or performance\"}\n", CORE_REVISION, CORE_DIRTY);
        for (int d = 0; d < 2; ++d) {
            check(hipSetDevice(d)); hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p, d));
            require(p.warpSize == 64 && std::string_view(p.gcnArchName).starts_with("gfx906"), "requires gfx906 wave64");
            Fixture f; f.invalid();
            for (auto type : {qwen::TensorType::Q4_0, qwen::TensorType::Q4_1, qwen::TensorType::Q5_0, qwen::TensorType::Q8_0, qwen::TensorType::Q6_K}) {
                const auto widths = type == qwen::TensorType::Q6_K ? std::array{256, 768, 1024, 1280, 2560, 16384} :
                    type == qwen::TensorType::Q8_0 ? std::array{32, 992, 1024, 1056, 2560, 16384} : std::array{32, 2016, 2048, 2080, 2560, 16384};
                for (int k : widths) for (int m : {1, 3, 9}) for (int n = 1; n <= 8; ++n) f.run(type, k, m, n);
                f.run(type, type == qwen::TensorType::Q6_K ? 256 : 32, 16384, 8);
            }
            for (int n = 4; n <= 8; ++n) {
                f.run(qwen::TensorType::Q5_0, 10240, 24, n);
                f.run(qwen::TensorType::Q4_0, 2560, 640, n);
                f.run(qwen::TensorType::Q4_1, 640, 2560, n);
                f.run(qwen::TensorType::Q8_0, 2560, 640, n);
            }
            for (int n : {4, 8}) f.run(qwen::TensorType::Q6_K, 2560, 248320, n, true);
            require(f.cases == 747 && f.rejects == 42, "fixture coverage");
            std::printf("{\"kind\":\"linear_short_boundaries\",\"device\":%d,\"cases\":%zu,\"host_rejects\":%zu,\"finite_exact_N1_elements\":%zu,\"cpu_oracle_elements\":%zu,\"max_abs\":%.17g,\"max_bound_ratio\":%.17g,\"q8_byte_parity\":true,\"immutable\":true,\"canaries\":true,\"reuse_generations\":2,\"old_api_rejects_N4\":true,\"passed\":true}\n", d, f.cases, f.rejects, f.elements, f.oracle_elements, f.max_abs, f.max_ratio);
        }
        std::printf("{\"kind\":\"linear_short_complete\",\"devices\":2,\"cases\":1494,\"passed\":true}\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "linear-short: %s\n", e.what()); return 1; }
}
