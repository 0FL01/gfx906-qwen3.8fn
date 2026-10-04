#include "hip/session_ops.cuh"
#include "hip/linear.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

// Parent wires C++20/gfx906 + strict FP, qwen-session/session_ops.hip and
// qwen-linear-gpu/linear.hip. No model required. BOTH real gfx906 devices.
// Exact CURRENT canonical SiLU + Q8 byte parity only, not a CPU-exp emulator,
// full-model gate or performance proof. All records after both devices cleanup.
namespace {
constexpr int width = 640, capacity = 30;
constexpr std::size_t floats = width * capacity, blocks = floats / 32, guard = 64;
constexpr std::array<int, 7> lengths{1, 2, 3, 7, 10, 29, 30};
constexpr std::byte poison{0xa5};
static_assert(qwen::silu_pairs_width == width && qwen::silu_pairs_max_columns == capacity);
static_assert(sizeof(float) == 4 && sizeof(qwen::Q8_1) == 36);
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
void hip(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) { std::cerr << "paired-middle cleanup: " << hipGetErrorString(status) << '\n'; std::abort(); }
}
int live = 0;
struct Stream {
    hipStream_t value = nullptr;
    Stream() { hip(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); ++live; }
    ~Stream() { cleanup(hipStreamSynchronize(value)); cleanup(hipStreamDestroy(value)); --live; }
};
struct Buffer {
    hipStream_t stream;
    void* allocation = nullptr;
    std::size_t bytes;
    explicit Buffer(hipStream_t s, std::size_t n) : stream(s), bytes(n) {
        hip(hipMalloc(&allocation, bytes + 2 * guard)); ++live;
    }
    ~Buffer() { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation)); --live; }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* get() const { return reinterpret_cast<T*>(static_cast<std::byte*>(allocation) + guard); }
    void fill() const { hip(hipMemsetAsync(allocation, 0xa5, bytes + 2 * guard, stream)); }
    void zero() const { hip(hipMemsetAsync(get<void>(), 0, bytes, stream)); }
    void upload(const void* source) const { hip(hipMemcpyAsync(get<void>(), source, bytes, hipMemcpyHostToDevice, stream)); }
    void read(void* destination) const {
        hip(hipMemcpyAsync(destination, get<void>(), bytes, hipMemcpyDeviceToHost, stream)); hip(hipStreamSynchronize(stream));
    }
    void guards(std::size_t used) const {
        require(used <= bytes, "used prefix capacity");
        std::array<std::byte, 153728> raw{}; // Largest buffer: 153600 + two guards.
        require(bytes + 2 * guard <= raw.size(), "bounded guard readback");
        hip(hipMemcpyAsync(raw.data(), allocation, bytes + 2 * guard, hipMemcpyDeviceToHost, stream));
        hip(hipStreamSynchronize(stream));
        for (std::size_t i = 0; i < guard; ++i) require(raw[i] == poison, "prefix guard");
        for (std::size_t i = guard + used; i < bytes + 2 * guard; ++i) require(raw[i] == poison, "tail/suffix guard");
    }
};
struct Result {
    int device = 0;
    std::size_t cases = 0, middle_values = 0, q8_bytes = 0, host_rejects = 0, invalid_sources = 0, tiny_packed_blocks = 0;
};
void provenance() {
    const std::string revision(CORE_REVISION);
    require(revision.size() == 40 && std::all_of(revision.begin(), revision.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }), "40-hex build revision");
    require(CORE_DIRTY == 0 || CORE_DIRTY == 1, "typed dirty provenance");
}
Result device(int id) {
    hip(hipSetDevice(id)); hipDeviceProp_t prop{}; hip(hipGetDeviceProperties(&prop, id));
    const std::string arch(prop.gcnArchName);
    require(prop.warpSize == 64 && (arch == "gfx906" || arch.starts_with("gfx906:")), "gfx906 wave64");
    Result result; result.device = id;
    Stream stream;
    Buffer pair(stream.value, 2 * floats * sizeof(float)), gate(stream.value, floats * sizeof(float)),
        up(stream.value, floats * sizeof(float)), middle(stream.value, floats * sizeof(float)),
        reference(stream.value, floats * sizeof(float)), quant(stream.value, blocks * sizeof(qwen::Q8_1)),
        reference_quant(stream.value, blocks * sizeof(qwen::Q8_1)), flag(stream.value, sizeof(int)), reference_flag(stream.value, sizeof(int));
    std::array<float, 2 * floats> paired{};
    std::array<float, floats> gates{}, ups{}, actual{}, expected{};
    std::array<qwen::Q8_1, blocks> q{}, ref_q{}, repeated{};
    const auto launch = [&](int n, bool quantize) {
        const int count = n * width;
        middle.fill(); reference.fill(); quant.fill(); reference_quant.fill(); flag.fill(); reference_flag.fill();
        flag.zero(); reference_flag.zero();
        hip(qwen::launch_silu_up_pairs(pair.get<float>(), n, middle.get<float>(), flag.get<int>(), stream.value));
        hip(qwen::launch_silu_up(gate.get<float>(), up.get<float>(), count, reference.get<float>(), reference_flag.get<int>(), stream.value));
        if (quantize) {
            hip(qwen::launch_quantize_q8_1(middle.get<float>(), quant.get<qwen::Q8_1>(), width, n, flag.get<int>(), stream.value));
            hip(qwen::launch_quantize_q8_1(reference.get<float>(), reference_quant.get<qwen::Q8_1>(), width, n, reference_flag.get<int>(), stream.value));
        }
        middle.read(actual.data()); reference.read(expected.data());
        int a = 0, b = 0; flag.read(&a); reference_flag.read(&b);
        require(a == b, "paired/canonical sticky flags differ");
        require(std::memcmp(actual.data(), expected.data(), std::size_t(count) * sizeof(float)) == 0, "paired/canonical FP32 bit parity");
        middle.guards(std::size_t(count) * sizeof(float)); reference.guards(std::size_t(count) * sizeof(float));
        flag.guards(sizeof(int)); reference_flag.guards(sizeof(int));
        if (quantize) {
            require(a == 0, "finite case numeric flag");
            quant.read(q.data()); reference_quant.read(ref_q.data());
            const auto bytes = std::size_t(n) * 20 * sizeof(qwen::Q8_1);
            require(std::memcmp(q.data(), ref_q.data(), bytes) == 0, "paired/canonical middle Q8 bytes including RAW sum");
            for (int b = 0; b < n * 20; ++b)
                if (q[b].d == 0 && q[b].s != 0 && std::any_of(std::begin(q[b].qs), std::end(q[b].qs), [](auto code) { return code != 0; }))
                    ++result.tiny_packed_blocks;
            quant.guards(bytes); reference_quant.guards(bytes);
            // Reuse the SAME canonical producer on the exact same floats; its
            // bytes must be stable with no stale tail / unreset-flag influence.
            hip(qwen::launch_quantize_q8_1(middle.get<float>(), quant.get<qwen::Q8_1>(), width, n, flag.get<int>(), stream.value));
            quant.read(repeated.data()); require(std::memcmp(q.data(), repeated.data(), bytes) == 0, "repeat quantization bytes");
            ++result.cases; result.middle_values += count; result.q8_bytes += bytes;
        } else { require(a != 0, "invalid source/arithmetic not reported"); ++result.invalid_sources; }
    };
    for (int variant = 0; variant < 3; ++variant) {
        for (std::size_t i = 0; i < floats; ++i) {
            // Deterministic near-rounding, tiny and negative-exp-overflow rows.
            const float g = variant == 0 ? float(int((i * 37) % 4093) - 2046) / 256.0f :
                variant == 1 ? std::ldexp(float(int(i % 31) - 15), -22) : (i % 2 ? -1000.0f : 1000.0f);
            const float u = variant == 2 ? 0.00001f * float(int(i % 13) - 6) : float(int((i * 53) % 2039) - 1019) / 1024.0f;
            gates[i] = g; ups[i] = u;
            paired[(i / width) * 2 * width + i % width] = g;
            paired[(i / width) * 2 * width + width + i % width] = u;
        }
        pair.fill(); gate.fill(); up.fill(); pair.upload(paired.data()); gate.upload(gates.data()); up.upload(ups.data());
        // Ascending then descending capacities exercise former larger live spans.
        for (int n : lengths) launch(n, true);
        for (auto it = lengths.rbegin(); it != lengths.rend(); ++it) launch(*it, true);
        std::array<float, 2 * floats> got{}; pair.read(got.data());
        require(std::memcmp(got.data(), paired.data(), sizeof(got)) == 0, "paired source modified");
        gate.read(actual.data()); up.read(expected.data());
        require(std::memcmp(actual.data(), gates.data(), sizeof(gates)) == 0 &&
            std::memcmp(expected.data(), ups.data(), sizeof(ups)) == 0, "canonical source modified");
        pair.guards(pair.bytes); gate.guards(gate.bytes); up.guards(up.bytes);
    }
    // Early/last live source, both halves, all nonfinite types; same canonical
    // sanitization/flags/zero result. Arithmetic overflow is a separate case.
    for (float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        for (bool is_up : {false, true}) for (std::size_t i : {std::size_t{0}, floats - 1}) {
            gates.fill(0.25f); ups.fill(0.5f);
            (is_up ? ups : gates)[i] = bad;
            for (std::size_t j = 0; j < floats; ++j) {
                paired[(j / width) * 2 * width + j % width] = gates[j];
                paired[(j / width) * 2 * width + width + j % width] = ups[j];
            }
            pair.upload(paired.data()); gate.upload(gates.data()); up.upload(ups.data()); launch(capacity, false);
        }
    gates.fill(std::numeric_limits<float>::max()); ups.fill(2);
    for (std::size_t i = 0; i < floats; ++i) {
        paired[(i / width) * 2 * width + i % width] = gates[i];
        paired[(i / width) * 2 * width + width + i % width] = ups[i];
    }
    pair.upload(paired.data()); gate.upload(gates.data()); up.upload(ups.data()); launch(capacity, false);
    // Host preflight: all failures leave output AND sticky flag untouched.
    middle.fill(); flag.fill(); hip(hipStreamSynchronize(stream.value));
    const auto reject = [&](const float* source, int n, float* output, int* error) {
        require(qwen::launch_silu_up_pairs(source, n, output, error, stream.value) == hipErrorInvalidValue, "host invalid paired call accepted");
        middle.guards(0); flag.guards(0); ++result.host_rejects;
    };
    for (int n : {-1, 0, 31, std::numeric_limits<int>::max()}) reject(pair.get<float>(), n, middle.get<float>(), flag.get<int>());
    reject(nullptr, 1, middle.get<float>(), flag.get<int>());
    reject(pair.get<float>(), 1, nullptr, flag.get<int>());
    reject(pair.get<float>(), 1, middle.get<float>(), nullptr);
    reject(pair.get<float>(), 1, pair.get<float>(), flag.get<int>());
    reject(pair.get<float>(), 1, middle.get<float>(), reinterpret_cast<int*>(pair.get<float>()));
    reject(pair.get<float>(), 1, middle.get<float>(), reinterpret_cast<int*>(middle.get<float>()));
    reject(reinterpret_cast<float*>(reinterpret_cast<std::byte*>(pair.get<float>()) + 1), 1, middle.get<float>(), flag.get<int>());
    reject(pair.get<float>(), 1, reinterpret_cast<float*>(reinterpret_cast<std::byte*>(middle.get<float>()) + 1), flag.get<int>());
    reject(pair.get<float>(), 1, middle.get<float>(), reinterpret_cast<int*>(reinterpret_cast<std::byte*>(flag.get<int>()) + 1));
    reject(reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3), 1, middle.get<float>(), flag.get<int>());
    reject(pair.get<float>(), 1, reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max() - 3), flag.get<int>());
    reject(pair.get<float>(), 1, middle.get<float>(), reinterpret_cast<int*>(std::numeric_limits<std::uintptr_t>::max() - 3));
    reject(pair.get<float>(), 1, pair.get<float>() + 1, flag.get<int>());
    // Recovery after actual numeric failures and host rejections, with SAME owners.
    paired.fill(0); gates.fill(0); ups.fill(0);
    pair.upload(paired.data()); gate.upload(gates.data()); up.upload(ups.data()); launch(3, true);
    // Valid work does not clear or gate on a previously set sticky bit. Match
    // the canonical kernel on the same retained sources/owners without reset.
    const int sticky = qwen::session_ops_error_arithmetic;
    flag.upload(&sticky); reference_flag.upload(&sticky);
    hip(qwen::launch_silu_up_pairs(pair.get<float>(), 3, middle.get<float>(), flag.get<int>(), stream.value));
    hip(qwen::launch_silu_up(gate.get<float>(), up.get<float>(), 3 * width, reference.get<float>(), reference_flag.get<int>(), stream.value));
    middle.read(actual.data()); reference.read(expected.data());
    int a = 0, b = 0; flag.read(&a); reference_flag.read(&b);
    require(a == sticky && b == sticky && std::memcmp(actual.data(), expected.data(), 3 * width * sizeof(float)) == 0, "sticky flag cleared/gated payload");
    middle.guards(3 * width * sizeof(float)); reference.guards(3 * width * sizeof(float));
    return result;
}
} // namespace
int main() {
    try {
        provenance(); int devices = 0; hip(hipGetDeviceCount(&devices)); require(devices == 2, "two actual gfx906 devices required");
        const std::array<Result, 2> results{device(0), device(1)};
        require(live == 0, "HIP ownership not released");
        std::cout << "{\"kind\":\"silu_pairs_header\",\"protocol\":1,\"revision\":\"" << CORE_REVISION << "\",\"dirty\":"
            << (CORE_DIRTY ? "true" : "false") << ",\"reference\":\"unchanged_canonical_GPU_SiLU_and_Q8\",\"scope\":\"CPUlinear_GPUmiddle_component\"}\n";
        for (const auto& r : results) {
            require(r.cases == 43 && r.invalid_sources == 13 && r.host_rejects == 17 && r.tiny_packed_blocks > 0, "exact component case counts/tiny ABI exercised");
            std::cout << "{\"kind\":\"silu_pairs_device\",\"device\":" << r.device << ",\"finite_cases\":" << r.cases
                << ",\"middle_values\":" << r.middle_values << ",\"middle_Q8_bytes\":" << r.q8_bytes
                << ",\"numeric_failures\":" << r.invalid_sources << ",\"host_rejects\":" << r.host_rejects
                << ",\"valid_zero_half_nonzero_codes_raw_sum_blocks\":" << r.tiny_packed_blocks
                << ",\"FP32_bit_differences\":0,\"Q8_byte_differences\":0,\"sticky_flag_checks\":1,\"canaries_and_reuse\":true,\"passed\":true}\n";
        }
        std::cout << "{\"kind\":\"silu_pairs_footer\",\"devices\":2,\"owners_released\":true,\"passed\":true,"
            "\"full_model_claim\":false,\"R5_complete_claim\":false,\"performance_claim\":false}\n";
        require(static_cast<bool>(std::cout), "JSONL write failure");
    } catch (const std::exception& e) { std::cerr << "silu-pairs-test: " << e.what() << '\n'; return 1; }
}
