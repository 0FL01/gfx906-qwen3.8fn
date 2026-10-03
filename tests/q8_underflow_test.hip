#include "hip/linear.cuh"
#include "linear_reference.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Link qwen-linear-gpu and qwen-core. C++20, strict FP, -ffp-contract=off,
// gfx906; no kernels or HIP runtime are required on the controller for syntax
// checking. Run manually on BOTH real devices: core-q8-underflow [ROW.f32].
// Component correctness only. No model weights, inference/performance claims,
// or attestation of a caller-supplied capture's provenance.
namespace {
constexpr double absolute_gate = 2e-4, relative_gate = 2e-5;
constexpr std::size_t capture_elements = 6144, capture_bytes = 24576;
constexpr std::string_view donor_revision = "dcd685463d597d31f5ca759d32c94592a2740fa4";
constexpr std::array<std::uint32_t, 32> observed_bits{
    0xb593b090u,0xb355b0eeu,0x3617cb5eu,0xb447656eu,0x36533193u,0xb4c04717u,0xb387e398u,0x3530214au,
    0x35492904u,0x35f67f2eu,0x35144cebu,0x361decd5u,0x35c4b6f4u,0x348fc19bu,0xb425b09cu,0xb4c60eecu,
    0x350bdd37u,0x364588d3u,0xb64161aeu,0x350eb3cdu,0x363893c7u,0x344e17f4u,0x3362558bu,0xb6687cc8u,
    0x363eb07fu,0xb53ceb08u,0xb653c440u,0x3607fa11u,0x356d09eeu,0x350f2bbdu,0xb60ae7f1u,0xb6016f2bu};
using Block = std::array<float, 32>;
constexpr std::array<std::string_view, 12> valid_names{
    "observed_block117", "positive_3.4e-6", "negative_3.4e-6", "mixed_sign",
    "negative_zero", "below_half_zero_tie", "at_half_zero_tie", "above_half_zero_tie",
    "valid_fp32_subnormal_scale_positive", "valid_fp32_subnormal_scale_negative", "zero", "normal"};
constexpr std::array<std::string_view, 11> invalid_names{
    "source_nan", "source_positive_inf", "source_negative_inf", "raw_sum_positive_half_overflow",
    "raw_sum_negative_half_overflow", "scale_positive_half_overflow", "scale_negative_half_overflow",
    "original_fp32_scale_zero_positive", "original_fp32_scale_zero_negative", "unsafe_code_positive128", "unsafe_code_negative128"};
static_assert(sizeof(qwen::Q8_1) == 36 && offsetof(qwen::Q8_1, d) == 0 &&
    offsetof(qwen::Q8_1, s) == 2 && offsetof(qwen::Q8_1, qs) == 4);
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little && capture_elements * sizeof(float) == capture_bytes);
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "q8_underflow_test.hip requires strict floating-point arithmetic"
#endif

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) { std::cerr << "Q8 cleanup: " << hipGetErrorString(status) << '\n'; std::abort(); }
}
struct Owners {
    std::size_t buffers = 0, streams = 0;
    void empty() const { require(buffers == 0 && streams == 0, "live HIP owners after cleanup"); }
} owners;
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); ++owners.streams; }
    ~Stream() { cleanup(hipStreamSynchronize(value)); cleanup(hipStreamDestroy(value)); --owners.streams; }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct CopyCompletion {
    hipStream_t stream;
    bool finished = false;
    ~CopyCompletion() { if (!finished) cleanup(hipStreamSynchronize(stream)); }
    void wait() { check(hipStreamSynchronize(stream)); finished = true; }
};
struct Buffer {
    hipStream_t stream;
    std::size_t bytes, prefix;
    void* allocation = nullptr;
    std::vector<std::byte> expected;
    Buffer(hipStream_t s, std::size_t n, std::size_t guard = 4)
        : stream(s), bytes(n), prefix(guard), expected(n + 2 * guard, std::byte{0xa5}) {
        require(n > 0 && n <= capture_bytes && (guard == 2 || guard == 4), "bounded buffer geometry");
        check(hipMalloc(&allocation, expected.size())); ++owners.buffers;
        try { copy(allocation, expected.data(), expected.size(), hipMemcpyHostToDevice); }
        catch (...) {
            cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation));
            allocation = nullptr; --owners.buffers; throw;
        }
    }
    ~Buffer() {
        if (allocation) { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation)); --owners.buffers; }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() const { return reinterpret_cast<T*>(static_cast<std::byte*>(allocation) + prefix); }
    void copy(void* destination, const void* source, std::size_t n, hipMemcpyKind kind) const {
        // Drain BEFORE temporary host storage unwinds, including copy errors.
        CopyCompletion completion{stream};
        check(hipMemcpyAsync(destination, source, n, kind, stream)); completion.wait();
    }
    void expect(std::span<const std::byte> source) {
        require(source.size() == bytes, "exact buffer expectation");
        std::copy(source.begin(), source.end(), expected.begin() + prefix);
    }
    void store(std::span<const std::byte> source) {
        expect(source); copy(as<void>(), source.data(), bytes, hipMemcpyHostToDevice);
    }
    void poison() {
        std::fill(expected.begin() + prefix, expected.begin() + prefix + bytes, std::byte{0xa5});
        copy(allocation, expected.data(), expected.size(), hipMemcpyHostToDevice);
    }
    std::vector<std::byte> read() const {
        std::vector<std::byte> result(expected.size());
        copy(result.data(), allocation, result.size(), hipMemcpyDeviceToHost); return result;
    }
    void readonly() const { require(read() == expected, "payload parity or readonly/redzone failure"); }
    void canaries(std::span<const std::byte> got) const {
        require(got.size() == expected.size() &&
            std::equal(got.begin(), got.begin() + prefix, expected.begin()) &&
            std::equal(got.begin() + prefix + bytes, got.end(), expected.begin() + prefix + bytes), "output canary failure");
    }
};

// Independent finite-FP32 -> FP16 RNE encoder: no production half helper is
// shared with the producer oracle. IEEE integer rounding also covers the tie
// to zero and all FP32 subnormals without a host F16C/FTZ dependency.
std::uint32_t round_right(std::uint32_t value, unsigned shift) {
    const auto base = value >> shift, remainder = value & ((1u << shift) - 1u);
    const auto tie = 1u << (shift - 1);
    return base + (remainder > tie || (remainder == tie && (base & 1u)));
}
std::uint16_t half_rne(float value) {
    require(std::isfinite(value), "test half encoder requires finite input");
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = std::uint16_t((bits >> 16) & 0x8000u);
    const int exponent = int((bits >> 23) & 255u) - 127;
    if (exponent < -25) return sign;
    if (exponent > 15) return std::uint16_t(sign | 0x7c00u);
    const auto fraction = bits & 0x7fffffu;
    const auto magnitude = exponent < -14 ? round_right(fraction | 0x800000u, unsigned(-exponent - 1)) :
        (std::uint32_t(exponent + 15) << 10) + round_right(fraction, 13);
    return std::uint16_t(sign | magnitude);
}
struct DonorInput {
    float amax = 0, d = 0, sum = 0;
    std::array<float, 32> codes{};
};
DonorInput donor_preflight(std::span<const float> x) {
    require(x.size() == 32, "donor block size"); DonorInput result;
    for (float v : x) {
        require(std::isfinite(v), "invalid source must not enter donor");
        result.amax = std::max(result.amax, std::abs(v));
    }
    if (result.amax == 0) return result;
    result.d = result.amax / 127.0f;
    require(std::isfinite(result.d) && result.d > 0, "invalid original FP32 scale must not enter donor");
    std::array<float, 32> sums{}; std::copy(x.begin(), x.end(), sums.begin());
    // Whole-stage snapshots independently emulate mx warp XOR reductions.
    for (unsigned mask = 1; mask < 32; mask *= 2) {
        const auto previous = sums;
        for (unsigned lane = 0; lane < 32; ++lane) sums[lane] = previous[lane] + previous[lane ^ mask];
    }
    result.sum = sums[0];
    require(std::isfinite(result.sum), "invalid sum must not enter donor");
    require((half_rne(result.d) & 0x7c00u) != 0x7c00u &&
        (half_rne(result.sum) & 0x7c00u) != 0x7c00u, "invalid half header must not enter donor");
    for (std::size_t j = 0; j < 32; ++j) {
        result.codes[j] = std::roundf(x[j] / result.d);
        require(std::isfinite(result.codes[j]) && result.codes[j] >= -127 && result.codes[j] <= 127,
            "unsafe code must not enter donor cast");
    }
    return result;
}
qwen::Q8_1 pinned_donor_valid(std::span<const float> x) {
    // Test-only adaptation of mx ggml/src/ggml-cuda/quantize.cu:89..101,
    // dcd685463d597d31f5ca759d32c94592a2740fa4, MIT (third_party/mx-LICENSE).
    // d=amax/127; q=amax==0?0:roundf(x/d); store half(d,RAW sum).
    // Preflight ALWAYS precedes the donor casts; invalid fixtures never call it.
    // Canonical zero blocks normalize d/s/codes to +0, including negative zero.
    const auto valid = donor_preflight(x); qwen::Q8_1 result{};
    if (valid.amax == 0) return result;
    result.d = half_rne(valid.d); result.s = half_rne(valid.sum);
    for (std::size_t j = 0; j < 32; ++j) result.qs[j] = static_cast<std::int8_t>(valid.codes[j]);
    return result;
}
struct Counters {
    std::size_t calls = 0, blocks = 0, valid_only_calls = 0, valid_compared_blocks = 0;
    std::size_t failure_calls = 0, failed_blocks = 0, cpu_rejects = 0, source_blocks = 0, arithmetic_blocks = 0;
    std::size_t sticky_reuses = 0, cleared_reuses = 0, matrices = 0, elements = 0, capture_blocks = 0;
    double max_abs = 0, max_bound_ratio = 0;
    void verify(bool capture) const {
        const std::size_t extra = capture ? 1 : 0, extra_blocks = capture ? 192 : 0;
        require(calls == 84 + extra && blocks == 324 + extra_blocks && valid_only_calls == 61 + extra &&
            valid_compared_blocks == 300 + extra_blocks && failure_calls == 23 && failed_blocks == 24 &&
            cpu_rejects == 24 && source_blocks == 7 && arithmetic_blocks == 17 &&
            sticky_reuses == 23 && cleared_reuses == 23 && matrices == 6 && elements == 70 &&
            capture_blocks == extra_blocks, "bounded fixture coverage mismatch");
    }
};
std::vector<qwen::Q8_1> expected_valid(std::span<const float> x, Counters& counters) {
    require(!x.empty() && x.size() % 32 == 0, "valid input shape");
    const auto before = std::vector<std::byte>(std::as_bytes(x).begin(), std::as_bytes(x).end());
    std::vector<qwen::Q8_1> expected(x.size() / 32), cpu(expected.size() + 2);
    for (std::size_t b = 0; b < expected.size(); ++b) expected[b] = pinned_donor_valid(x.subspan(b * 32, 32));
    std::memset(cpu.data(), 0xa5, cpu.size() * sizeof(cpu[0]));
    qwen::quantize_q8(x, std::span<qwen::Q8_1>(cpu).subspan(1, expected.size()));
    const auto cpu_bytes = std::as_bytes(std::span<const qwen::Q8_1>(cpu));
    require(std::all_of(cpu_bytes.begin(), cpu_bytes.begin() + 36, [](auto b) { return b == std::byte{0xa5}; }) &&
        std::all_of(cpu_bytes.end() - 36, cpu_bytes.end(), [](auto b) { return b == std::byte{0xa5}; }), "CPU Q8 redzone");
    require(std::memcmp(cpu.data() + 1, expected.data(), expected.size() * sizeof(expected[0])) == 0,
        "CPU versus pinned donor exact Q8 bytes");
    require(std::equal(before.begin(), before.end(), std::as_bytes(x).begin()), "CPU changed input bytes");
    counters.valid_compared_blocks += expected.size(); return expected;
}
void cpu_invalid(const Block& x, Counters& counters) {
    std::array<qwen::Q8_1, 3> output{}; std::memset(output.data(), 0xa5, sizeof(output));
    const auto before = output; const auto input_before = x; bool rejected = false;
    try { qwen::quantize_q8(x, std::span<qwen::Q8_1>(output).subspan(1, 1)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "CPU accepted genuinely invalid block");
    require(std::memcmp(output.data(), before.data(), sizeof(output)) == 0, "CPU failed-block publication/redzone");
    require(std::memcmp(x.data(), input_before.data(), sizeof(x)) == 0, "CPU invalid source bytes changed");
    ++counters.cpu_rejects;
}
struct QuantWorkspace {
    hipStream_t stream;
    int width, columns;
    Buffer raw, packed, flag;
    QuantWorkspace(hipStream_t s, int k, int n) : stream(s), width(k), columns(n),
        raw(s, std::size_t(k) * n * 4), packed(s, std::size_t(k / 32) * n * 36), flag(s, 4) {}
    void launch(std::span<const float> x, std::span<const qwen::Q8_1> expected,
                int initial_flag, int expected_flag, Counters& counters) {
        raw.store(std::as_bytes(x)); packed.poison();
        if (initial_flag >= 0) flag.store(std::as_bytes(std::span(&initial_flag, 1)));
        check(qwen::launch_quantize_q8_1(raw.as<float>(), packed.as<qwen::Q8_1>(), width, columns, flag.as<int>(), stream));
        const auto bytes = flag.read(); flag.canaries(bytes); int error = -1;
        std::memcpy(&error, bytes.data() + flag.prefix, sizeof(error));
        if (error != expected_flag)
            throw std::runtime_error("GPU quantizer sticky bits: call=" + std::to_string(counters.calls) +
                " width=" + std::to_string(width) + " columns=" + std::to_string(columns) +
                " expected=" + std::to_string(expected_flag) + " actual=" + std::to_string(error));
        flag.expect(std::as_bytes(std::span(&expected_flag, 1)));
        packed.expect(std::as_bytes(expected)); readonly();
        ++counters.calls; counters.blocks += expected.size();
    }
    void readonly() const { raw.readonly(); packed.readonly(); flag.readonly(); }
};

std::array<Block, 12> valid_blocks() {
    std::array<Block, 12> blocks{};
    for (std::size_t j = 0; j < 32; ++j) blocks[0][j] = std::bit_cast<float>(observed_bits[j]);
    blocks[1].fill(3.4e-6f); blocks[2].fill(-3.4e-6f);
    for (std::size_t j = 0; j < 32; ++j) blocks[3][j] = (float(j) - 15.5f) * (3.4e-6f / 16);
    blocks[4].fill(-0.0f);
    const float tie = std::ldexp(127.0f, -25);
    blocks[5][0] = std::nextafter(tie, 0.0f); blocks[6][0] = tie;
    blocks[7][0] = std::nextafter(tie, std::numeric_limits<float>::infinity());
    blocks[8][0] = std::bit_cast<float>(std::uint32_t{127}); blocks[9][0] = -blocks[8][0];
    for (std::size_t j = 0; j < 32; ++j) blocks[11][j] = float(int(j) - 12) / 64;
    const auto observed = donor_preflight(blocks[0]); const auto packed = pinned_donor_valid(blocks[0]);
    require(std::bit_cast<std::uint32_t>(observed.amax) == 0x36687cc8u &&
        std::bit_cast<std::uint32_t>(observed.d) == 0x32ea516bu &&
        std::bit_cast<std::uint32_t>(observed.sum) == 0x372921eau && packed.d == 0 && packed.s == 0x00a9u &&
        std::any_of(std::begin(packed.qs), std::end(packed.qs), [](auto q) { return q != 0; }), "exact observed block proof");
    for (int i : {1, 2, 3}) {
        const auto q = pinned_donor_valid(blocks[i]);
        require(q.d == 0 && std::any_of(std::begin(q.qs), std::end(q.qs), [](auto v) { return v != 0; }), "tiny codes preserved");
        if (i != 3) require((q.s & 0x7fffu) != 0, "tiny raw sum preserved");
    }
    require(pinned_donor_valid(blocks[5]).d == 0 && pinned_donor_valid(blocks[6]).d == 0 &&
        pinned_donor_valid(blocks[7]).d == 1, "FP16 RNE below/at/above zero tie proof");
    for (int i : {8, 9}) {
        const auto info = donor_preflight(blocks[i]);
        require(std::fpclassify(info.d) == FP_SUBNORMAL && info.d > 0 && std::abs(info.codes[0]) == 127,
            "valid original FP32 subnormal scale proof");
    }
    for (int i : {4, 10}) {
        const auto q = pinned_donor_valid(blocks[i]); const qwen::Q8_1 zero{};
        require(std::memcmp(&q, &zero, sizeof(q)) == 0, "canonical positive zero proof");
    }
    return blocks;
}
struct Invalid { Block x{}; int bit = 2; };
std::array<Invalid, 11> invalid_blocks() {
    std::array<Invalid, 11> bad{};
    bad[0].x[0] = std::numeric_limits<float>::quiet_NaN();
    bad[1].x[0] = std::numeric_limits<float>::infinity(); bad[2].x[0] = -bad[1].x[0];
    for (int i : {0, 1, 2}) bad[i].bit = 1;
    bad[3].x.fill(3000.0f); bad[4].x.fill(-3000.0f);
    // +/- cancellation makes the sum zero: this isolates HALF SCALE overflow.
    for (std::size_t j = 0; j < 32; ++j) { bad[5].x[j] = j % 2 ? -1e8f : 1e8f; bad[6].x[j] = -bad[5].x[j]; }
    bad[7].x[0] = std::numeric_limits<float>::denorm_min(); bad[8].x[0] = -bad[7].x[0];
    bad[9].x[0] = std::bit_cast<float>(std::uint32_t{128}); bad[10].x[0] = -bad[9].x[0];
    require(bad[7].x[0] != 0 && bad[7].x[0] / 127.0f == 0, "original FP32 zero scale proof");
    for (int i : {9, 10}) {
        const float d = std::abs(bad[i].x[0]) / 127.0f;
        require(d > 0 && std::abs(std::roundf(bad[i].x[0] / d)) == 128, "unsafe signed code proof before cast");
    }
    return bad;
}
std::vector<float> flatten(std::span<const Block> blocks) {
    std::vector<float> result; result.reserve(blocks.size() * 32);
    for (const auto& b : blocks) result.insert(result.end(), b.begin(), b.end());
    return result;
}
void linear(QuantWorkspace& input, std::span<const qwen::Q8_1> q, Counters& counters) {
    constexpr int rows = 7;
    for (auto type : {qwen::TensorType::Q4_0, qwen::TensorType::Q4_1}) {
        const std::size_t bb = type == qwen::TensorType::Q4_0 ? 18 : 20;
        std::vector<std::byte> weights(std::size_t(input.width / 32) * rows * bb);
        for (std::size_t b = 0; b < weights.size() / bb; ++b) {
            auto* p = weights.data() + b * bb;
            const auto d = half_rne(float(b % 7 + 1) / 8);
            std::memcpy(p, &d, 2); const std::size_t header = type == qwen::TensorType::Q4_0 ? 2 : 4;
            if (header == 4) { const auto m = half_rne(float(b % 7 + 1) / 4); std::memcpy(p + 2, &m, 2); }
            for (std::size_t j = 0; j < 16; ++j) p[header + j] = std::byte(((b + j) % 16) | (((3 * b + j + 5) % 16) << 4));
        }
        std::vector<float> reference(std::size_t(rows) * input.columns);
        qwen::matmul_q8_reference({type, input.width, rows, weights}, q, input.columns, reference);
        Buffer device_weights(input.stream, weights.size(), 2), output(input.stream, reference.size() * 4);
        device_weights.store(weights); std::vector<float> unwritten(reference.size(), std::numeric_limits<float>::quiet_NaN());
        output.store(std::as_bytes(std::span<const float>(unwritten)));
        check(qwen::launch_quantized_linear({type, device_weights.as<void>(), input.width, rows},
            input.packed.as<qwen::Q8_1>(), input.columns, output.as<float>(), input.stream));
        const auto got = output.read(); output.canaries(got);
        for (std::size_t j = 0; j < reference.size(); ++j) {
            float value = 0; std::memcpy(&value, got.data() + output.prefix + j * 4, 4);
            require(std::isfinite(value) && std::isfinite(reference[j]), "nonfinite common-Q8 linear result");
            const double error = std::abs(double(value) - reference[j]);
            const double ratio = error / (absolute_gate + relative_gate * std::abs(double(reference[j])));
            require(ratio <= 1, "frozen common-Q8 2e-4 + 2e-5*abs(ref) gate");
            // Single observed d=0 block MUST retain a nonzero raw-sum correction.
            if (input.width == 32) require(reference[j] != 0 && value != 0, "linear discarded valid d=0 raw sum");
            counters.max_abs = std::max(counters.max_abs, error);
            counters.max_bound_ratio = std::max(counters.max_bound_ratio, ratio); ++counters.elements;
        }
        device_weights.readonly(); input.readonly(); ++counters.matrices;
    }
}
void failure_cycle(hipStream_t stream, const std::array<Block, 12>& valid,
                   const std::array<Invalid, 11>& bad, int test, int failed_half, Counters& counters) {
    std::array<Block, 4> good{valid[0], valid[11], valid[1], valid[2]}, mixed = good;
    std::array<int, 4> bits{};
    if (test < 0) { mixed[0] = bad[0].x; mixed[1] = bad[3].x; bits[0] = 1; bits[1] = 2; }
    else {
        mixed[failed_half] = bad[test].x; bits[failed_half] = bad[test].bit;
        // Exercise the opposite end lane in the second half of the wave too.
        if (failed_half == 1) std::reverse(mixed[failed_half].begin(), mixed[failed_half].end());
    }
    std::vector<qwen::Q8_1> expected(4); int combined = 0;
    for (std::size_t b = 0; b < 4; ++b) {
        if (bits[b]) {
            cpu_invalid(mixed[b], counters); combined |= bits[b]; ++counters.failed_blocks;
            if (bits[b] == 1) ++counters.source_blocks; else ++counters.arithmetic_blocks;
        } else expected[b] = expected_valid(mixed[b], counters)[0];
    }
    QuantWorkspace workspace(stream, 128, 1);
    workspace.launch(flatten(mixed), expected, 0, combined, counters); ++counters.failure_calls;
    const auto clean = flatten(good);
    const auto wanted = expected_valid(clean, counters);
    workspace.launch(clean, wanted, -1, combined, counters); ++counters.sticky_reuses; ++counters.valid_only_calls;
    const auto wanted_again = expected_valid(clean, counters);
    workspace.launch(clean, wanted_again, 0, 0, counters); ++counters.cleared_reuses; ++counters.valid_only_calls;
}
Counters qualify(int device, std::span<const float> capture) {
    check(hipSetDevice(device)); Stream stream; Counters counters;
    const auto valid = valid_blocks(); const auto bad = invalid_blocks();
    for (std::size_t b = 0; b < valid.size(); ++b) {
        const auto expected = expected_valid(valid[b], counters); QuantWorkspace workspace(stream.value, 32, 1);
        workspace.launch(valid[b], expected, 0, 0, counters); ++counters.valid_only_calls;
        if (b == 0) linear(workspace, expected, counters);
    }
    const auto all = flatten(valid);
    for (auto shape : {std::array{384, 1}, std::array{128, 3}, std::array{32, 12}}) {
        const auto expected = expected_valid(all, counters); QuantWorkspace workspace(stream.value, shape[0], shape[1]);
        workspace.launch(all, expected, 0, 0, counters); ++counters.valid_only_calls;
        if (shape[1] <= 3) linear(workspace, expected, counters);
    }
    for (int test = 0; test < int(bad.size()); ++test)
        for (int failed_half : {0, 1}) failure_cycle(stream.value, valid, bad, test, failed_half, counters);
    failure_cycle(stream.value, valid, bad, -1, 0, counters); // sticky OR: source 1 plus arithmetic 2 -> 3
    if (!capture.empty()) {
        const auto expected = expected_valid(capture, counters); QuantWorkspace workspace(stream.value, 6144, 1);
        workspace.launch(capture, expected, 0, 0, counters); ++counters.valid_only_calls;
        counters.capture_blocks = expected.size();
    }
    counters.verify(!capture.empty()); return counters; // Stream and all buffers die before caller emits success.
}
std::vector<float> load_capture(const char* path) {
    require(std::string_view(path).size() <= 4096, "capture path exceeds bounded JSON input");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file) && file.tellg() == std::streamoff(capture_bytes), "capture must be exactly 24576 bytes");
    file.seekg(0); std::vector<float> row(capture_elements);
    file.read(reinterpret_cast<char*>(row.data()), std::streamsize(capture_bytes));
    require(bool(file) && file.gcount() == std::streamsize(capture_bytes) && file.peek() == std::char_traits<char>::eof() &&
        !file.bad(), "capture bounded read failed or file length changed");
    for (float v : row) require(std::isfinite(v), "capture contains nonfinite F32");
    for (std::size_t j = 0; j < 32; ++j)
        require(std::bit_cast<std::uint32_t>(row[117 * 32 + j]) == observed_bits[j], "capture block117 differs from embedded observation");
    return row;
}
void json_string(std::string_view value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32 || c >= 127) {
            constexpr char hex[] = "0123456789abcdef";
            std::cout << "\\u00" << hex[c / 16] << hex[c % 16];
        } else std::cout << char(c);
    }
    std::cout << '"';
}
template<std::size_t N> void names(const std::array<std::string_view, N>& values) {
    std::cout << '[';
    for (std::size_t j = 0; j < N; ++j) { if (j) std::cout << ','; json_string(values[j]); }
    std::cout << ']';
}
void run(const char* capture_path) {
    require(std::fegetround() == FE_TONEAREST, "requires host FP round-to-nearest");
    require(CORE_DIRTY == 0 || CORE_DIRTY == 1, "CORE_DIRTY must be numeric 0/1");
    const std::string_view revision = CORE_REVISION;
    require(revision.size() == 40 && std::all_of(revision.begin(), revision.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }), "requires quoted 40-lowercase-hex CORE_REVISION");
    const auto capture = capture_path ? load_capture(capture_path) : std::vector<float>{};
    int count = 0; check(hipGetDeviceCount(&count)); require(count == 2, "requires exactly two real gfx906 devices");
    for (int d = 0; d < 2; ++d) {
        hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p, d)); const std::string_view arch = p.gcnArchName;
        require(p.warpSize == 64 && (arch == "gfx906" || arch.starts_with("gfx906:")), "requires exact gfx906 ISA wave64");
    }
    std::cout << std::setprecision(17) << "{\"kind\":\"q8_underflow_source\",\"protocol\":1,\"revision\":"; json_string(revision);
    std::cout << ",\"dirty\":" << CORE_DIRTY << ",\"donor_revision\":"; json_string(donor_revision);
    std::cout << ",\"donor_source\":\"ggml/src/ggml-cuda/quantize.cu:89..101\",\"scope\":\"component Q8 byte parity and bounded synthetic common-Q8 linear correctness\""
        ",\"abi\":{\"bytes\":36,\"d_offset\":0,\"s_offset\":2,\"qs_offset\":4},\"absolute_gate\":0.0002,\"relative_gate\":0.00002"
        ",\"embedded_observation\":{\"block\":117,\"elements\":32,\"amax_bits\":912817352,\"fp32_d_bits\":854217067,\"xor_sum_bits\":925442538,\"stored_d\":0,\"stored_s\":169}"
        ",\"valid_cases\":"; names(valid_names); std::cout << ",\"invalid_cases\":"; names(invalid_names);
    std::cout << ",\"capture\":{\"supplied\":" << (capture_path ? "true" : "false") << ",\"path\":";
    if (capture_path) json_string(capture_path); else std::cout << "null";
    std::cout << ",\"expected_bytes\":24576,\"expected_blocks\":192,\"provenance_attested\":false},\"jsonl_records\":4}\n";
    for (int d = 0; d < 2; ++d) {
        const auto c = qualify(d, capture); owners.empty();
        std::cout << "{\"kind\":\"q8_underflow_correctness\",\"protocol\":1,\"device\":" << d
            << ",\"quantizer_calls\":" << c.calls << ",\"output_blocks\":" << c.blocks << ",\"valid_only_calls\":" << c.valid_only_calls
            << ",\"donor_cpu_gpu_valid_blocks\":" << c.valid_compared_blocks << ",\"failure_calls\":" << c.failure_calls
            << ",\"failed_zero_blocks\":" << c.failed_blocks << ",\"cpu_rejects\":" << c.cpu_rejects
            << ",\"source_error_blocks\":" << c.source_blocks << ",\"arithmetic_error_blocks\":" << c.arithmetic_blocks
            << ",\"sticky_valid_reuses\":" << c.sticky_reuses << ",\"cleared_valid_reuses\":" << c.cleared_reuses
            << ",\"linear_matrices\":" << c.matrices << ",\"linear_elements\":" << c.elements
            << ",\"capture_blocks\":" << c.capture_blocks << ",\"max_abs\":" << c.max_abs << ",\"max_bound_ratio\":" << c.max_bound_ratio
            << ",\"q8_exact_bytes\":true,\"failed_blocks_zero\":true,\"wave64_halves_isolated\":true,\"sticky_or_3\":true"
               ",\"finite\":true,\"readonly\":true,\"canaries\":true,\"cleanup\":true,\"passed\":true}\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 1 || argc == 2, "usage: core-q8-underflow [captured-6144.f32]");
        run(argc == 2 ? argv[1] : nullptr); // All host payloads and HIP RAII owners are now destroyed.
        owners.empty();
        std::cout << "{\"kind\":\"q8_underflow_complete\",\"protocol\":1,\"devices\":2,\"correctness_records\":2,\"jsonl_records\":4,\"live_owners\":0,\"cleanup\":true,\"passed\":true}\n";
        std::cout.flush(); require(bool(std::cout), "Q8 JSONL output failure"); return 0;
    } catch (const std::exception& e) { std::cerr << "q8-underflow: " << e.what() << '\n'; return 1; }
}
