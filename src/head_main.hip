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

// Full actual LM-head computation; sampled independent common-Q8 CPU oracle,
// full-output finite/prefix/canary checks. Not a full-model logits fixture.
namespace {
constexpr int width = 2560, rows = 248320, repeats = 20;
constexpr std::size_t row_bytes = 2100;
constexpr double absolute_gate = 2e-4, relative_gate = 2e-5;
constexpr float canary = -12345.0f;
constexpr std::array sample_rows{0,1,7,31,63,127,255,511,1023,4095,8191,16383,
    32767,65535,131071,200003,248043,248044,248045,248046,248047,248127,248318,248319};
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void cleanup(hipError_t s) noexcept {
    if (s != hipSuccess) { std::cerr << hipGetErrorString(s) << '\n'; std::abort(); }
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t bytes) { check(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p) cleanup(hipFree(p)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as(std::size_t offset = 0) {
        return reinterpret_cast<T*>(static_cast<std::byte*>(p) + offset);
    }
};
struct Stream {
    hipStream_t p = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&p, hipStreamNonBlocking)); }
    ~Stream() { if (p) cleanup(hipStreamDestroy(p)); }
};
struct Event {
    hipEvent_t p = nullptr;
    Event() { check(hipEventCreate(&p)); }
    ~Event() { if (p) cleanup(hipEventDestroy(p)); }
};
struct Completion {
    hipStream_t stream;
    ~Completion() { cleanup(hipStreamSynchronize(stream)); }
};
struct Error {
    double max_abs = 0, max_bound_ratio = 0;
    std::size_t elements = 0;
    void compare(float got, float expected) {
        require(std::isfinite(got) && std::isfinite(expected), "nonfinite head comparison");
        const double e = std::abs(double(got) - double(expected));
        const double ratio = e / (absolute_gate + relative_gate * std::abs(double(expected)));
        max_abs = std::max(max_abs, e); max_bound_ratio = std::max(max_bound_ratio, ratio);
        ++elements;
        require(ratio <= 1, "LM head common-Q8 gate");
    }
    void emit() const {
        std::cout << "{\"max_abs\":" << max_abs << ",\"max_bound_ratio\":"
                  << max_bound_ratio << ",\"elements\":" << elements << '}';
    }
};
struct Fixture {
    Stream stream;
    Event begin, end;
    Buffer weights, x{width * 3 * sizeof(float)}, q{width / 32 * 3 * sizeof(qwen::Q8_1)},
        y{(std::size_t(rows) * 3 + 2) * sizeof(float)}, flag{sizeof(int)};
    explicit Fixture(std::span<const std::byte> bytes) : weights(bytes.size()) {
        Completion completion{stream.p};
        check(hipMemcpyAsync(weights.p, bytes.data(), bytes.size(), hipMemcpyHostToDevice, stream.p));
    }
    void rejects() {
        Completion completion{stream.p};
        const qwen::QuantizedDeviceMatrix good{qwen::TensorType::Q6_K, weights.p, width, rows};
        int count = 0;
        auto bad = [&](qwen::QuantizedDeviceMatrix m, int n) {
            require(qwen::launch_quantized_linear(m, q.as<qwen::Q8_1>(), n,
                y.as<float>(4), stream.p) == hipErrorInvalidValue, "head shape not rejected");
            check(hipGetLastError()); ++count;
        };
        for (int r : {rows - 1, rows + 1, std::numeric_limits<int>::max()}) {
            auto m = good; m.output = r; bad(m, 1);
        }
        for (int k : {width - 256, width + 256}) { auto m = good; m.input = k; bad(m, 1); }
        for (auto t : {qwen::TensorType::Q4_0, qwen::TensorType::Q4_1,
                       qwen::TensorType::Q5_0, qwen::TensorType::Q8_0}) {
            auto m = good; m.type = t; bad(m, 1);
        }
        bad(good, 0); bad(good, 4);
        require(count == 11, "head reject coverage");
    }
    std::vector<float> run(int device, int n, std::span<const std::byte> bytes) {
        Completion completion{stream.p};
        std::vector<float> input(std::size_t(width) * n), output(std::size_t(rows) * n + 2);
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = .5f * std::sin(float(i) * .137f) + .125f * std::cos(float(i) * .071f);
        std::vector<qwen::Q8_1> expected_q(input.size() / 32), got_q(expected_q.size());
        qwen::quantize_q8(input, expected_q);
        check(hipMemcpyAsync(x.p, input.data(), input.size() * 4, hipMemcpyHostToDevice, stream.p));
        check(hipMemsetAsync(flag.p, 0, sizeof(int), stream.p));
        check(qwen::launch_quantize_q8_1(x.as<float>(), q.as<qwen::Q8_1>(), width, n,
            flag.as<int>(), stream.p));
        int error = -1;
        check(hipMemcpyAsync(&error, flag.p, sizeof(error), hipMemcpyDeviceToHost, stream.p));
        check(hipMemcpyAsync(got_q.data(), q.p, got_q.size() * sizeof(qwen::Q8_1), hipMemcpyDeviceToHost, stream.p));
        check(hipStreamSynchronize(stream.p));
        require(error == 0 && std::memcmp(expected_q.data(), got_q.data(),
            got_q.size() * sizeof(qwen::Q8_1)) == 0, "head Q8 producer parity");
        std::array<std::array<float, 3>, sample_rows.size()> reference{};
        for (std::size_t s = 0; s < sample_rows.size(); ++s) {
            const qwen::QMatrix m{qwen::TensorType::Q6_K, width, 1,
                bytes.subspan(std::size_t(sample_rows[s]) * row_bytes, row_bytes)};
            qwen::matmul_q8_reference(m, expected_q, n, std::span<float>(reference[s]).first(n));
        }
        const qwen::QuantizedDeviceMatrix matrix{qwen::TensorType::Q6_K, weights.p, width, rows};
        auto reset = [&] {
            std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
            output.front() = output.back() = canary;
            check(hipMemcpyAsync(y.p, output.data(), output.size() * 4, hipMemcpyHostToDevice, stream.p));
            check(hipStreamSynchronize(stream.p));
        };
        auto validate = [&](Error& e) {
            check(hipMemcpyAsync(output.data(), y.p, output.size() * 4, hipMemcpyDeviceToHost, stream.p));
            check(hipStreamSynchronize(stream.p));
            require(output.front() == canary && output.back() == canary, "head output canaries");
            require(std::all_of(output.begin() + 1, output.end() - 1,
                [](float v) { return std::isfinite(v); }), "head full-output finite");
            for (int c = 0; c < n; ++c) for (std::size_t s = 0; s < sample_rows.size(); ++s)
                e.compare(output[1 + std::size_t(c) * rows + sample_rows[s]], reference[s][c]);
        };
        Error warmup;
        reset(); check(qwen::launch_quantized_linear(matrix, q.as<qwen::Q8_1>(), n,
            y.as<float>(4), stream.p)); validate(warmup);
        Error measured; double ms_sum = 0;
        for (int rep = 0; rep < repeats; ++rep) {
            reset(); check(hipEventRecord(begin.p, stream.p));
            check(qwen::launch_quantized_linear(matrix, q.as<qwen::Q8_1>(), n, y.as<float>(4), stream.p));
            check(hipEventRecord(end.p, stream.p)); check(hipEventSynchronize(end.p));
            float ms = 0; check(hipEventElapsedTime(&ms, begin.p, end.p));
            require(std::isfinite(ms) && ms > 0, "head completed timing");
            ms_sum += ms; validate(measured);
        }
        check(hipMemcpyAsync(got_q.data(), q.p, got_q.size() * sizeof(qwen::Q8_1), hipMemcpyDeviceToHost, stream.p));
        check(hipStreamSynchronize(stream.p));
        require(std::memcmp(expected_q.data(), got_q.data(), got_q.size() * sizeof(qwen::Q8_1)) == 0,
            "head changed input");
        std::cout << "{\"kind\":\"head_measurement\",\"device\":" << device
                  << ",\"tokens\":" << n << ",\"rows\":248320,\"repeats\":20,\"resident_ms\":"
                  << ms_sum / repeats << ",\"error\":"; measured.emit();
        std::cout << ",\"all_finite\":true,\"canaries\":true,\"q8_bytes_exact\":true}\n";
        return {output.begin() + 1, output.end() - 1};
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: core-head target.gguf");
        qwen::Model model(argv[1]);
        require(model.metadata_value("general.architecture").get<std::string>() == "qwen4exp",
            "head model architecture");
        const auto& t = model.tensor("output.weight");
        require(t.type == qwen::TensorType::Q6_K && t.rank == 2 && t.dimensions[0] == width &&
            t.dimensions[1] == rows && t.byte_size == std::size_t(rows) * row_bytes,
            "actual LM head geometry/type/bytes");
        std::vector<std::byte> bytes(t.byte_size); model.read_tensor(t.name, bytes);
        for (std::size_t b = 0; b < bytes.size() / 210; ++b) {
            const auto* p = bytes.data() + b * 210 + 208;
            const auto bits = std::uint16_t(std::to_integer<unsigned>(p[0]) |
                (std::to_integer<unsigned>(p[1]) << 8));
            require(std::isfinite(qwen::half_to_float(bits)), "nonfinite LM head scale");
        }
        int devices = 0; check(hipGetDeviceCount(&devices)); require(devices == 2, "requires two gfx906");
        for (int d = 0; d < devices; ++d) {
            hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p, d));
            require(p.warpSize == 64 && std::string_view(p.gcnArchName).starts_with("gfx906"), "head ISA");
        }
        std::cout << std::setprecision(17) << "{\"kind\":\"head_source\",\"revision\":\""
                  << CORE_REVISION << "\",\"dirty\":" << CORE_DIRTY
                  << ",\"model_variant\":\"qwen38-keep1-Q4_0\",\"type\":\"Q6_K\",\"input\":2560,\"output\":248320,\"bytes\":" << bytes.size()
                  << ",\"absolute_gate\":0.0002,\"relative_gate\":0.00002,\"sample_rows\":[";
        for (std::size_t i = 0; i < sample_rows.size(); ++i) { if (i) std::cout << ','; std::cout << sample_rows[i]; }
        std::cout << "],\"scope\":\"full actual LM head with deterministic common-Q8 input; sampled CPU oracle, full finite and prefix checks\""
                  << ",\"timing_scope\":\"completed resident linear events; excludes upload, quantization, reset, CPU oracle and readback\"}\n";
        for (int d = 0; d < devices; ++d) {
            check(hipSetDevice(d));
            {
                Fixture fixture(bytes); fixture.rejects();
                const auto one = fixture.run(d, 1, bytes), two = fixture.run(d, 2, bytes), three = fixture.run(d, 3, bytes);
                require(std::memcmp(one.data(), two.data(), one.size() * 4) == 0 &&
                    std::memcmp(two.data(), three.data(), two.size() * 4) == 0, "head full-prefix parity");
            }
            std::cout << "{\"kind\":\"head_correctness\",\"device\":" << d
                      << ",\"host_rejects\":11,\"prefix_elements\":744960,\"prefix_exact\":true,\"passed\":true}\n";
        }
        std::cout << "{\"kind\":\"head_complete\",\"measurements\":6,\"rows\":10,\"passed\":true}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "core-head: " << e.what() << '\n'; return 1; }
}
