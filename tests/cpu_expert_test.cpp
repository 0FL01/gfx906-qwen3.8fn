#include "cpu_expert.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CPU_EXPERT_TEST_X86 1
#else
#define CPU_EXPERT_TEST_X86 0
#endif
#if defined(__linux__)
#include <sched.h>
#endif

// Counts C++ allocations on ALL threads, including persistent workers. This is
// not a malloc/syscall/DDR/HIP measurement. Startup and numeric-error exception
// machinery are outside the normal-job allocation contract.
namespace heap_monitor {
std::atomic<bool> enabled{false};
std::atomic<std::size_t> calls{0};
void count() noexcept { if (enabled.load(std::memory_order_relaxed)) calls.fetch_add(1, std::memory_order_relaxed); }
void* allocate(std::size_t n) {
    count();
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
void* aligned(std::size_t n, std::size_t alignment) {
    count();
    void* p = nullptr;
    if (posix_memalign(&p, alignment, n == 0 ? 1 : n) == 0) return p;
    throw std::bad_alloc();
}
}
void* operator new(std::size_t n) { return heap_monitor::allocate(n); }
void* operator new[](std::size_t n) { return heap_monitor::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return heap_monitor::aligned(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return heap_monitor::aligned(n, std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {
using namespace qwen;
std::size_t checks = 0, cases = 0, byte_mismatches = 0;
std::size_t fp_environment_rejects = 0, ticket_alias_rejects = 0, synthetic_numeric_rejects = 0;
std::size_t staged_cases = 0, staged_rejects = 0, staged_fp_rejects = 0;
double linear_error = 0, middle_error = 0, output_error = 0;
double ordered_error = 0, common_down_error = 0;
int max_quant_code_delta = 0;
constexpr float canary = -917.25f;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void status(CpuExpertStatus got, CpuExpertStatus expected, const char* message) {
    if (got != expected) std::cerr << message << ": " << cpu_expert_status_name(got) << '\n';
    check(got == expected, message);
}
template<class F> void throws(F f, const char* message) {
    bool did = false;
    try { f(); } catch (const std::exception&) { did = true; }
    check(did, message);
}
void parity(std::span<const float> actual, std::span<const float> reference,
            double absolute, double relative, double& maximum, const char* message) {
    check(actual.size() == reference.size(), "parity span");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::fabs(double(actual[i]) - double(reference[i]));
        maximum = std::max(maximum, error);
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i]) ||
            error > absolute + relative * std::fabs(double(reference[i]))) {
            std::cerr << message << " index=" << i << " actual=" << actual[i] << " ref=" << reference[i] << " abs=" << error << '\n';
            check(false, message);
        }
        ++checks;
    }
}
template<class T> bool bytes_equal(std::span<const T> a, std::span<const T> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// Independent mathematical binary16 oracle, deliberately not production bit
// manipulation or F16C. Encoding searches the finite nearest-neighbor lattice.
float from_half(std::uint16_t h) {
    const int exponent = (h / 1024) % 32, fraction = h % 1024;
    float value = exponent == 0 ? std::ldexp(float(fraction), -24) :
                                 std::ldexp(float(1024 + fraction), exponent - 25);
    if (exponent == 31) value = fraction == 0 ? std::numeric_limits<float>::infinity() :
                                               std::numeric_limits<float>::quiet_NaN();
    return (h & 0x8000U) != 0 ? -value : value;
}
std::uint16_t to_half(float x) {
    const auto sign = std::uint16_t(std::signbit(x) ? 0x8000U : 0);
    const double magnitude = std::fabs(double(x));
    if (std::isnan(x)) return std::uint16_t(sign | 0x7e00U);
    if (magnitude >= 65520) return std::uint16_t(sign | 0x7c00U);
    static const auto lattice = [] {
        std::array<float, 0x7c00> values{};
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = from_half(std::uint16_t(i));
        return values;
    }();
    const auto upper = std::lower_bound(lattice.begin(), lattice.end(), magnitude);
    if (upper == lattice.begin()) return sign;
    if (upper == lattice.end()) return std::uint16_t(sign | 0x7bffU);
    const auto hi = std::size_t(upper - lattice.begin());
    const double below = magnitude - double(lattice[hi - 1]), above = double(lattice[hi]) - magnitude;
    const auto i = below < above || (below == above && (hi & 1U) != 0) ? hi - 1 : hi;
    return std::uint16_t(sign | std::uint16_t(i));
}
std::uint16_t load_half(const std::byte* b) {
    std::uint16_t h;
    std::memcpy(&h, b, sizeof(h));
    return h;
}
// Independent FULL-block dot32, separate multiply/subtract/add, serial K sum.
// This remains an oracle; it is NOT replaced with the candidate's GPU ordering.
void scalar_matrix(QMatrix m, std::span<const Q8_1> input, int columns, std::span<float> out) {
    const auto k = std::size_t(m.input / 32);
    const auto stride = m.type == TensorType::Q4_0 ? 18U : 20U;
    const auto offset = m.type == TensorType::Q4_0 ? 2U : 4U;
    for (int c = 0; c < columns; ++c) for (int row = 0; row < m.output; ++row) {
        float total = 0;
        for (std::size_t b = 0; b < k; ++b) {
            const auto* w = m.weights.data() + (std::size_t(row) * k + b) * stride;
            const auto& x = input[std::size_t(c) * k + b];
            int dot = 0;
            for (std::size_t j = 0; j < 32; ++j) {
                const auto packed = std::to_integer<unsigned>(w[offset + j % 16]);
                dot += int(j < 16 ? packed & 15U : packed >> 4) * int(x.qs[j]);
            }
            const float d4 = from_half(load_half(w)), dx = from_half(x.d), sx = from_half(x.s);
            float value;
            if (m.type == TensorType::Q4_0) value = d4 * (float(dot) * dx - 8.0f * sx);
            else value = float(dot) * from_half(to_half(d4 * dx)) +
                         from_half(to_half(from_half(load_half(w + 2)) * sx));
            total = total + value;
        }
        out[std::size_t(c) * std::size_t(m.output) + std::size_t(row)] = total;
    }
}
// Additional SOURCE-DERIVED scalar MMVQ emulator. Full-block oracle above is
// still independent. This checks fragment/FMA/reduction ordering locally, not
// actual HIP code generation, device denorm modes, libm, or GPU execution.
void ordered_matrix(QMatrix m, std::span<const Q8_1> input, int columns, std::span<float> out) {
    const int blocks = m.input / 32;
    const int stride = m.type == TensorType::Q4_0 ? 18 : 20;
    const int offset = m.type == TensorType::Q4_0 ? 2 : 4;
    for (int c = 0; c < columns; ++c) for (int row = 0; row < m.output; ++row) {
        std::array<float, 128> lane{};
        for (int tid = 0; tid < 128; ++tid) {
            const int first = (tid % 2) * 8;
            for (int k = tid / 2; k < blocks; k += 64) {
                const auto* b = m.weights.data() + std::size_t(row * blocks + k) * std::size_t(stride);
                const auto& x = input[std::size_t(c * blocks + k)];
                int dot = 0;
                for (int j = first; j < first + 8; ++j) {
                    const auto packed = std::to_integer<unsigned>(b[offset + j]);
                    dot += int(packed & 15U) * int(x.qs[j]);
                    dot += int(packed >> 4) * int(x.qs[j + 16]);
                }
                const float d4 = from_half(load_half(b)), dx = from_half(x.d), sx = from_half(x.s);
                auto& acc = lane[std::size_t(tid)];
                if (m.type == TensorType::Q4_0) acc = std::fma(d4, std::fma(float(dot), dx, -4.0f * sx), acc);
                else acc = acc + std::fma(float(dot), from_half(to_half(d4 * dx)),
                                  from_half(to_half(from_half(load_half(b + 2)) * sx)) * 0.5f);
            }
        }
        for (std::size_t i = 0; i < 64; ++i) lane[i] += lane[i + 64];
        for (std::size_t mask = 1; mask < 64; mask *= 2) {
            const auto old = lane;
            for (std::size_t i = 0; i < 64; ++i) lane[i] = old[i] + old[i ^ mask];
        }
        out[std::size_t(c * m.output + row)] = lane[0];
    }
}
// Independent same-float Q8 oracle: no production quantizer/conversions. Raw
// sum uses staged arrays (not candidate in-place updates), RNE half storage,
// original FP32 amax/127 division for round-away-from-zero int8 decisions.
void oracle_quant(std::span<const float> in, std::span<Q8_1> out) {
    for (std::size_t b = 0; b < out.size(); ++b) {
        std::array<float, 32> sum{};
        float maximum = 0;
        for (std::size_t i = 0; i < 32; ++i) {
            sum[i] = in[b * 32 + i];
            maximum = std::max(maximum, std::fabs(sum[i]));
        }
        Q8_1 q{};
        if (maximum != 0) {
            const float d = maximum / 127.0f;
            q.d = to_half(d);
            for (std::size_t mask = 1; mask < 32; mask *= 2) {
                const auto old = sum;
                for (std::size_t i = 0; i < 32; ++i) sum[i] = old[i] + old[i ^ mask];
            }
            q.s = to_half(sum[0]);
            for (std::size_t i = 0; i < 32; ++i) {
                const float scaled = in[b * 32 + i] / d;
                q.qs[i] = std::int8_t(scaled >= 0 ? std::floor(double(scaled) + 0.5) : std::ceil(double(scaled) - 0.5));
            }
        }
        out[b] = q;
    }
}
struct Matrix {
    TensorType type;
    int input, output;
    std::vector<std::byte> bytes;
    Matrix(TensorType t, int k, int m, unsigned seed) : type(t), input(k), output(m),
        bytes(std::size_t(k / 32 * m) * (t == TensorType::Q4_0 ? 18U : 20U)) {
        const auto stride = t == TensorType::Q4_0 ? 18U : 20U;
        for (std::size_t i = 0; i < bytes.size() / stride; ++i) {
            const float scale = float(2 + (i * 7 + seed) % 11) / 2048.0f;
            const auto d = to_half((i + seed) % 3 == 0 ? -scale : scale);
            std::memcpy(bytes.data() + i * stride, &d, 2);
            const auto offset = t == TensorType::Q4_0 ? 2U : 4U;
            if (t == TensorType::Q4_1) {
                const auto minimum = to_half(-7.5f * scale);
                std::memcpy(bytes.data() + i * stride + 2, &minimum, 2);
            }
            for (std::size_t j = 0; j < 16; ++j) {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                bytes[i * stride + offset + j] = std::byte(seed & 255U);
            }
        }
    }
    QMatrix view() const { return {type, input, output, bytes}; }
};
struct Fixture {
    Matrix gate{TensorType::Q4_0, 2560, 640, 0x70af451U};
    Matrix up{TensorType::Q4_0, 2560, 640, 0xa5f726eU};
    Matrix down;
    CpuExpert expert;
    alignas(4) std::array<Q8_1, 320> input{}; // Column0/1/2 at 0/160/80: nonadjacent gather.
    explicit Fixture(TensorType type) : down(type, 640, 2560, 0x525a31U), expert(gate.view(), up.view(), down.view()) {
        std::array<float, 10240> floats{};
        for (std::size_t i = 0; i < floats.size(); ++i)
            floats[i] = 0.7f * std::sin(float(i) * 0.137f) + 0.2f * std::cos(float(i) * 0.071f);
        quantize_q8(floats, input); // ABI fixture, not a claim of actual GPU-produced bytes.
    }
    CpuExpertJob job(int n, std::span<float> output) const {
        CpuExpertJob j;
        j.expert = &expert; j.columns = n; j.output = output;
        constexpr std::size_t starts[3] = {0, 160, 80};
        for (int c = 0; c < n && c < 3; ++c) j.input[std::size_t(c)] = {input.data() + starts[c], 80};
        return j;
    }
};
struct alignas(64) GuardedScratch {
    std::array<std::uint64_t, 8> before{};
    CpuExpertScratch scratch{};
    std::array<std::uint64_t, 8> after{};
    GuardedScratch() { before.fill(0x12345678abcdeff0ULL); after = before; }
    void guard() const { for (auto x : before) check(x == 0x12345678abcdeff0ULL, "scratch prefix guard");
                         for (auto x : after) check(x == 0x12345678abcdeff0ULL, "scratch suffix guard"); }
};

void correctness(Fixture& f) {
    GuardedScratch workspace;
    std::array<float, 7682> output{};
    std::array<float, 1920> sg{}, su{}, sm{}, ordered{};
    std::array<float, 7680> sd{}, common{}, expected{};
    std::array<Q8_1, 60> qm{}, same{}, library{};
    std::array<Q8_1, 240> x{};
    for (int n : {1, 2, 3}) {
        ++cases;
        output.fill(canary);
        std::memset(&workspace.scratch, 0x5a, sizeof(workspace.scratch));
        auto job = f.job(n, {output.data() + 1, std::size_t(n) * 2560});
        status(run_cpu_expert(job, workspace.scratch), CpuExpertStatus::ok, "expert run");
        auto& scratch = workspace.scratch;
        for (int c = 0; c < n; ++c) {
            std::copy(job.input[std::size_t(c)].begin(), job.input[std::size_t(c)].end(), x.begin() + c * 80);
            check(std::memcmp(scratch.gathered_x.data() + c * 80, job.input[std::size_t(c)].data(), 80 * sizeof(Q8_1)) == 0,
                  "borrowed Q8 gather changed bytes");
        }
        const auto mid = std::size_t(n) * 640, out = std::size_t(n) * 2560;
        const std::span<const Q8_1> packed{x.data(), std::size_t(n) * 80};
        scalar_matrix(f.gate.view(), packed, n, {sg.data(), mid});
        scalar_matrix(f.up.view(), packed, n, {su.data(), mid});
        // Original core-expert gates, fixed before candidate execution.
        parity({scratch.gate.data(), mid}, {sg.data(), mid}, 2e-4, 2e-5, linear_error, "gate scalar");
        parity({scratch.up.data(), mid}, {su.data(), mid}, 2e-4, 2e-5, linear_error, "up scalar");
        ordered_matrix(f.gate.view(), packed, n, {ordered.data(), mid});
        parity({scratch.gate.data(), mid}, {ordered.data(), mid}, 0, 0, ordered_error, "gate source order");
        check(bytes_equal<float>({scratch.gate.data(), mid}, {ordered.data(), mid}), "gate source order bits");
        ordered_matrix(f.up.view(), packed, n, {ordered.data(), mid});
        check(bytes_equal<float>({scratch.up.data(), mid}, {ordered.data(), mid}), "up source order bits");
        for (std::size_t i = 0; i < mid; ++i) {
            sm[i] = (sg[i] / (1.0f + std::exp(-sg[i]))) * su[i];
            const float same_middle = (scratch.gate[i] / (1.0f + std::exp(-scratch.gate[i]))) * scratch.up[i];
            check(std::bit_cast<std::uint32_t>(same_middle) == std::bit_cast<std::uint32_t>(scratch.middle[i]), "separate SiLU math");
        }
        parity({scratch.middle.data(), mid}, {sm.data(), mid}, 2e-3, 2e-4, middle_error, "middle scalar");
        oracle_quant({scratch.middle.data(), mid}, {same.data(), mid / 32});
        quantize_q8({scratch.middle.data(), mid}, {library.data(), mid / 32});
        check(bytes_equal<Q8_1>({scratch.middle_q8.data(), mid / 32}, {same.data(), mid / 32}), "independent same-float Q8 bytes");
        check(bytes_equal<Q8_1>({scratch.middle_q8.data(), mid / 32}, {library.data(), mid / 32}), "library same-float Q8 bytes");
        oracle_quant({sm.data(), mid}, {qm.data(), mid / 32});
        for (std::size_t b = 0; b < mid / 32; ++b) {
            const auto* a = reinterpret_cast<const unsigned char*>(&qm[b]);
            const auto* z = reinterpret_cast<const unsigned char*>(&scratch.middle_q8[b]);
            for (std::size_t i = 0; i < sizeof(Q8_1); ++i) byte_mismatches += std::size_t(a[i] != z[i]);
            for (std::size_t i = 0; i < 32; ++i)
                max_quant_code_delta = std::max(max_quant_code_delta, std::abs(int(qm[b].qs[i]) - int(scratch.middle_q8[b].qs[i])));
        }
        scalar_matrix(f.down.view(), {qm.data(), mid / 32}, n, {sd.data(), out});
        parity(job.output, {sd.data(), out}, 2e-3, 2e-4, output_error, "full scalar pipeline");
        scalar_matrix(f.down.view(), {scratch.middle_q8.data(), mid / 32}, n, {common.data(), out});
        parity(job.output, {common.data(), out}, 2e-4, 2e-5, common_down_error, "common-Q8 down");
        ordered_matrix(f.down.view(), {scratch.middle_q8.data(), mid / 32}, n, {expected.data(), out});
        check(bytes_equal<float>(job.output, {expected.data(), out}), "down source order bits");
        check(output.front() == canary && output[1 + out] == canary && output.back() == canary, "output canaries");
        auto untouched = [](const void* p, std::size_t bytes) {
            const auto* data = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < bytes; ++i) check(data[i] == 0x5a, "scratch unused tail mutated");
        };
        untouched(scratch.gathered_x.data() + n * 80, std::size_t(240 - n * 80) * sizeof(Q8_1));
        untouched(scratch.middle_q8.data() + mid / 32, (60 - mid / 32) * sizeof(Q8_1));
        untouched(scratch.gate.data() + mid, (1920 - mid) * sizeof(float));
        untouched(scratch.up.data() + mid, (1920 - mid) * sizeof(float));
        untouched(scratch.middle.data() + mid, (1920 - mid) * sizeof(float));
        untouched(scratch.down.data() + out, (7680 - out) * sizeof(float));
        workspace.guard();
        // Individual jobs must produce exact grouped results, including middle
        // decisions, with no inter-column/cross-rank accumulation.
        for (int c = 0; c < n; ++c) {
            CpuExpertScratch individual{};
            std::array<float, 2560> one{};
            auto j = f.job(1, one); j.input[0] = job.input[std::size_t(c)];
            status(run_cpu_expert(j, individual), CpuExpertStatus::ok, "individual run");
            check(bytes_equal<float>(one, job.output.subspan(std::size_t(c) * 2560, 2560)), "group vs individual output");
            check(bytes_equal<Q8_1>({individual.middle_q8.data(), 20}, {scratch.middle_q8.data() + c * 20, 20}), "group middle quant bytes");
        }
        // Repeat the same borrowed views on the same caller scratch.
        std::copy(job.output.begin(), job.output.end(), expected.begin());
        status(run_cpu_expert(job, scratch), CpuExpertStatus::ok, "scratch reuse");
        check(bytes_equal<float>(job.output, {expected.data(), out}), "reuse bits");
    }
    // Full signed byte extrema are valid opaque dot inputs, despite not being
    // produced by roundf(x/d). Keep raw sums independent from code sums.
    const auto original = f.input;
    for (auto& b : f.input) {
        b.d = to_half(1.0f / 512); b.s = to_half(-0.1875f);
        for (int i = 0; i < 32; ++i) b.qs[i] = std::int8_t(i % 2 == 0 ? -128 : 127);
    }
    auto j = f.job(3, {output.data() + 1, 7680});
    status(run_cpu_expert(j, workspace.scratch), CpuExpertStatus::ok, "signed extrema");
    for (int c = 0; c < 3; ++c) std::copy(j.input[std::size_t(c)].begin(), j.input[std::size_t(c)].end(), x.begin() + c * 80);
    ordered_matrix(f.gate.view(), x, 3, ordered);
    check(bytes_equal<float>(workspace.scratch.gate, ordered), "signed dot fragment order");
    f.input = original;
}

// Component composition, NOT a host-libm/GPU-middle parity claim. Borrow the
// whole oracle's EXACT middle Q8 bytes to verify both narrow CPU projections.
// Real canonical GPU middle bit parity is a separate two-device HIP fixture.
void staged_projections(Fixture& f) {
    GuardedScratch whole, stage;
    std::array<float, 7682> full{}, output{};
    alignas(4) std::array<Q8_1, 60> middle{};
    const auto original_input = f.input;
    const auto gate_bytes = f.gate.bytes, up_bytes = f.up.bytes, down_bytes = f.down.bytes;
    CpuExpertPool pool({3, false, {}, cpu_expert_max_workers});
    constexpr std::array<std::size_t, 3> starts{0, 40, 20};
    for (int n : {1, 2, 3}) {
        ++staged_cases;
        const auto down_size = std::size_t(n) * 2560, pair_size = std::size_t(n) * 1280;
        full.fill(canary); output.fill(canary);
        status(run_cpu_expert(f.job(n, {full.data() + 1, down_size}), whole.scratch), CpuExpertStatus::ok, "whole stage reference");
        auto gate = f.job(n, {output.data() + 1, pair_size}); gate.stage = CpuExpertStage::gate_up;
        status(run_cpu_expert(gate, stage.scratch), CpuExpertStatus::ok, "gate/up stage");
        const std::vector<float> paired(gate.output.begin(), gate.output.end());
        for (int c = 0; c < n; ++c) {
            check(bytes_equal<float>(gate.output.subspan(std::size_t(c) * 1280, 640),
                {whole.scratch.gate.data() + c * 640, 640}), "interleaved gate bits");
            check(bytes_equal<float>(gate.output.subspan(std::size_t(c) * 1280 + 640, 640),
                {whole.scratch.up.data() + c * 640, 640}), "interleaved up bits");
            check(std::memcmp(stage.scratch.gathered_x.data() + c * 80, gate.input[std::size_t(c)].data(), 2880) == 0,
                "gate stage exact original Q8 bytes");
            std::memcpy(middle.data() + starts[std::size_t(c)], whole.scratch.middle_q8.data() + c * 20, 720);
        }
        check(output.front() == canary && output[pair_size + 1] == canary && output.back() == canary, "paired output canaries");
        CpuExpertJob down; down.expert = &f.expert; down.columns = n; down.stage = CpuExpertStage::down;
        down.output = {output.data() + 1, down_size};
        for (int c = 0; c < n; ++c) down.input[std::size_t(c)] = {middle.data() + starts[std::size_t(c)], 20};
        const auto original_middle = middle;
        output.fill(canary);
        status(run_cpu_expert(down, stage.scratch), CpuExpertStatus::ok, "down stage opaque middle Q8");
        check(bytes_equal<float>(down.output, {full.data() + 1, down_size}), "down stage vs whole exact bytes");
        for (int c = 0; c < n; ++c)
            check(std::memcmp(stage.scratch.middle_q8.data() + c * 20, down.input[std::size_t(c)].data(), 720) == 0,
                "down input gathered without requantization");
        check(bytes_equal<Q8_1>(middle, original_middle), "down mutated borrowed middle Q8");
        check(output.front() == canary && output[down_size + 1] == canary && output.back() == canary, "down canaries");
        whole.guard(); stage.guard();

        std::size_t ticket = 999;
        output.fill(canary);
        status(pool.submit(gate, ticket), CpuExpertStatus::ok, "pooled gate stage"); check(ticket == 0, "gate phase ticket");
        status(pool.drain(), CpuExpertStatus::ok, "gate phase drain");
        check(bytes_equal<float>(gate.output, paired), "pooled paired bits");
        const auto paired_snapshot = output;
        ticket = 999;
        status(pool.submit(down, ticket), CpuExpertStatus::invalid_argument, "stage output reuse before reset rejected");
        ++staged_rejects;
        check(ticket == 999 && output == paired_snapshot && pool.result(0).status == CpuExpertStatus::ok,
            "stage alias changed output/ticket/result");
        status(pool.reset(), CpuExpertStatus::ok, "retire gate phase ranges");
        output.fill(canary);
        // Startup already complete. ALL these narrow normal jobs must allocate
        // zero C++ heap across caller/workers, as does the original whole job.
        heap_monitor::calls.store(0); heap_monitor::enabled.store(true);
        const auto submitted = pool.submit(down, ticket), drained = pool.drain();
        heap_monitor::enabled.store(false);
        status(submitted, CpuExpertStatus::ok, "pooled down stage"); status(drained, CpuExpertStatus::ok, "down phase drain");
        check(heap_monitor::calls.load() == 0 && ticket == 0, "hot down heap/ticket");
        check(bytes_equal<float>(down.output, {full.data() + 1, down_size}), "pooled down exact bits");
        status(pool.reset(), CpuExpertStatus::ok, "down phase reset");

        for (auto good : {gate, down}) {
            output.fill(canary);
            const auto scratch_before = stage.scratch;
            const auto rejected = [&](CpuExpertJob bad) {
                status(run_cpu_expert(bad, stage.scratch), CpuExpertStatus::invalid_argument, "staged argument rejection");
                check(std::memcmp(&scratch_before, &stage.scratch, sizeof(scratch_before)) == 0, "staged rejection changed scratch");
                ticket = 999;
                status(pool.submit(bad, ticket), CpuExpertStatus::invalid_argument, "staged pool argument rejection");
                check(ticket == 999 && pool.result(0).state == CpuExpertJobState::empty &&
                    std::all_of(output.begin(), output.end(), [](float x) { return x == canary; }), "staged rejection changed output/ticket");
                ++staged_rejects;
            };
            auto bad = good; bad.stage = static_cast<CpuExpertStage>(99); rejected(bad);
            bad = good; bad.input[std::size_t(n - 1)] = bad.input[std::size_t(n - 1)].first(bad.input[std::size_t(n - 1)].size() - 1); rejected(bad);
            bad = good; bad.output = bad.output.first(bad.output.size() - 1); rejected(bad);
            bad = good; bad.output = {reinterpret_cast<float*>(const_cast<Q8_1*>(good.input[0].data())), good.output.size()}; rejected(bad);
            // Header of LAST block in LAST live column: no prefix publication.
            auto* last = const_cast<Q8_1*>(good.input[std::size_t(n - 1)].data()) + good.input[std::size_t(n - 1)].size() - 1;
            const auto saved = *last;
            last->s = 0x7e00; rejected(good); *last = saved;
            last->d = 0xbc00; rejected(good); *last = saved;
            const int round = std::fegetround();
            check(std::fesetround(FE_DOWNWARD) == 0, "stage set bad rounding");
            const auto direct = run_cpu_expert(good, stage.scratch);
            ticket = 999; const auto queued = pool.submit(good, ticket);
            check(std::fesetround(round) == 0, "stage restore rounding");
            status(direct, CpuExpertStatus::numeric_error, "stage direct bad fenv");
            status(queued, CpuExpertStatus::numeric_error, "stage submit bad fenv");
            ++staged_fp_rejects;
            check(ticket == 999 && std::memcmp(&scratch_before, &stage.scratch, sizeof(scratch_before)) == 0 &&
                std::all_of(output.begin(), output.end(), [](float x) { return x == canary; }), "stage fenv changed scratch/output/ticket");
        }
        // Two phase warm/reuse hot loop; no per-stage threads or row-range jobs.
        heap_monitor::calls.store(0); heap_monitor::enabled.store(true);
        std::size_t t = 999;
        bool hot = pool.submit(gate, t) == CpuExpertStatus::ok && pool.drain() == CpuExpertStatus::ok &&
            pool.reset() == CpuExpertStatus::ok && pool.submit(down, t) == CpuExpertStatus::ok &&
            pool.drain() == CpuExpertStatus::ok && pool.reset() == CpuExpertStatus::ok;
        hot &= run_cpu_expert(gate, stage.scratch) == CpuExpertStatus::ok && run_cpu_expert(down, stage.scratch) == CpuExpertStatus::ok;
        heap_monitor::enabled.store(false);
        check(hot && heap_monitor::calls.load() == 0, "hot staged jobs allocate/failed");
    }
    check(bytes_equal<Q8_1>(f.input, original_input) && f.gate.bytes == gate_bytes && f.up.bytes == up_bytes && f.down.bytes == down_bytes,
        "staged projections mutated canonical inputs/weights");
}

void invalid_arguments(Fixture& f) {
    CpuExpertScratch scratch{};
    std::array<float, 7680> output{}; output.fill(canary);
    std::array<std::byte, sizeof(scratch)> before{};
    std::memcpy(before.data(), &scratch, sizeof(scratch));
    auto rejected = [&](const CpuExpertJob& j) {
        status(run_cpu_expert(j, scratch), CpuExpertStatus::invalid_argument, "invalid job");
        check(std::memcmp(before.data(), &scratch, sizeof(scratch)) == 0, "invalid job changed scratch");
        for (auto x : output) check(x == canary, "invalid job changed output");
    };
    auto good = f.job(3, output);
    auto j = good; j.expert = nullptr; rejected(j);
    for (int n : {-1, 0, 4, std::numeric_limits<int>::max()}) { j = good; j.columns = n; rejected(j); }
    j = good; j.output = {output.data(), 7679}; rejected(j);
    j = good; j.input[2] = j.input[2].first(79); rejected(j);
    j = f.job(1, {output.data(), 2560}); j.input[2] = good.input[2]; rejected(j);
    j = good; j.input[2] = {reinterpret_cast<const Q8_1*>(reinterpret_cast<const char*>(f.input.data()) + 1), 80}; rejected(j);
    j = good; j.output = {reinterpret_cast<float*>(reinterpret_cast<char*>(output.data()) + 1), 7680}; rejected(j);
    j = good; j.output = {reinterpret_cast<float*>(f.input.data()), 7680}; rejected(j);
    j = good; j.output = {reinterpret_cast<float*>(f.gate.bytes.data()), 7680}; rejected(j);
    j = good; j.output = scratch.down; rejected(j);
    j = good; j.input[0] = {scratch.gathered_x.data(), 80}; rejected(j);
    const auto late = f.input[159]; // Last block of the LAST active column.
    for (auto h : {std::uint16_t(0x7c00), std::uint16_t(0xfc00), std::uint16_t(0x7e00),
                   std::uint16_t(0xfe00), std::uint16_t(0xbc00)}) {
        f.input[159].d = h; rejected(good); f.input[159] = late;
    }
    for (auto h : {std::uint16_t(0x7c00), std::uint16_t(0xfc00), std::uint16_t(0x7e00), std::uint16_t(0xfe00)}) {
        f.input[159].s = h; rejected(good); f.input[159] = late;
    }
    // Constructor geometry/dtype/size/alignment and late weight-scale rejects.
    auto g = f.gate.view(), u = f.up.view(), d = f.down.view();
    auto bad = g; bad.type = TensorType::Q4_1;
    throws([&] { CpuExpert e(bad, u, d); }, "gate dtype accepted");
    bad = g; bad.input = 32;
    throws([&] { CpuExpert e(bad, u, d); }, "shape accepted");
    bad = d; bad.type = TensorType::F16;
    throws([&] { CpuExpert e(g, u, bad); }, "down dtype accepted");
    bad = g; bad.weights = bad.weights.first(bad.weights.size() - 1);
    throws([&] { CpuExpert e(bad, u, d); }, "short weight span accepted");
    bad = g; bad.weights = {g.weights.data() + 1, g.weights.size()};
    throws([&] { CpuExpert e(bad, u, d); }, "misaligned weights accepted");
    Matrix bad_down = f.down;
    const auto stride = bad_down.type == TensorType::Q4_0 ? 18U : 20U;
    const auto position = bad_down.bytes.size() - stride;
    const std::uint16_t nan = 0x7e00;
    std::memcpy(bad_down.bytes.data() + position, &nan, 2);
    throws([&] { CpuExpert e(g, u, bad_down.view()); }, "late nonfinite weight accepted");
    if (bad_down.type == TensorType::Q4_1) {
        bad_down = f.down;
        std::memcpy(bad_down.bytes.data() + position + 2, &nan, 2);
        throws([&] { CpuExpert e(g, u, bad_down.view()); }, "late nonfinite minimum accepted");
    }
    status(run_cpu_expert(good, scratch), CpuExpertStatus::ok, "reuse after invalid");
    // Numeric failure: finite input/weights but half products/intermediates not
    // representable. NO partial output, and next normal job remains usable.
    output.fill(canary);
    const auto saved = f.input;
    for (auto& b : f.input) { b.d = to_half(65504); b.s = 0; }
    status(run_cpu_expert(good, scratch), CpuExpertStatus::numeric_error, "numeric failure");
    for (auto x : output) check(x == canary, "numeric failure published output");
    f.input = saved;
    status(run_cpu_expert(good, scratch), CpuExpertStatus::ok, "numeric error reuse");
    const int old_round = std::fegetround();
    check(std::fesetround(FE_DOWNWARD) == 0, "set rounding");
    output.fill(canary);
    const auto bad_fp = run_cpu_expert(good, scratch);
    check(std::fesetround(old_round) == 0, "restore rounding");
    status(bad_fp, CpuExpertStatus::numeric_error, "non-nearest rounding");
    for (auto x : output) check(x == canary, "bad FP environment published output");
}

void independent_fp_environment(Fixture& f) {
#if CPU_EXPERT_TEST_X86
    struct Restore {
        std::fenv_t saved{};
        unsigned mxcsr = _mm_getcsr();
        Restore() { check(std::fegetenv(&saved) == 0, "save full FP environment"); }
        ~Restore() {
            if (std::fesetenv(&saved) != 0) std::terminate();
            _mm_setcsr(mxcsr); // Restore even independently changed MXCSR state.
        }
    };
    GuardedScratch workspace;
    std::array<std::byte, sizeof(CpuExpertScratch)> before{};
    std::memcpy(before.data(), &workspace.scratch, before.size());
    std::array<float, 7680> output{}; output.fill(canary);
    const auto good = f.job(3, output);
    const auto reject = [&] {
        status(run_cpu_expert(good, workspace.scratch), CpuExpertStatus::numeric_error, "independent FP environment rejection");
        check(std::memcmp(before.data(), &workspace.scratch, before.size()) == 0, "FP rejection changed scratch");
        check(std::all_of(output.begin(), output.end(), [](float x) { return x == canary; }), "FP rejection published output");
        throws([] { CpuExpertPool p({1, false, {}, cpu_expert_max_workers}); }, "bad independent FP environment pool startup");
        workspace.guard(); ++fp_environment_rejects;
    };
    const unsigned original_mxcsr = _mm_getcsr();
    const int original_round = std::fegetround(), original_exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    check(original_round == FE_TONEAREST && (original_mxcsr & 0xe040U) == 0, "FP test requires canonical environment");
    for (unsigned bits : {0x2000U, 0x4000U, 0x6000U, 0x8000U, 0x0040U}) {
        {
            Restore restore;
            _mm_setcsr((restore.mxcsr & ~0xe040U) | bits);
            check(std::fegetround() == FE_TONEAREST, "MXCSR-only change affected x87 rounding");
            check((_mm_getcsr() & 0xe040U) == bits, "MXCSR-only perturbation not installed");
            reject();
        }
        check(_mm_getcsr() == original_mxcsr && std::fegetround() == original_round &&
              std::fetestexcept(FE_ALL_EXCEPT) == original_exceptions, "MXCSR test failed to restore full fenv");
    }
    {
        Restore restore;
        check(std::fesetround(FE_DOWNWARD) == 0, "set independent x87 rounding");
        _mm_setcsr(restore.mxcsr & ~0xe040U);
        check(std::fegetround() == FE_DOWNWARD && (_mm_getcsr() & 0xe040U) == 0, "independent x87 perturbation");
        reject();
    }
    check(_mm_getcsr() == original_mxcsr && std::fegetround() == original_round &&
          std::fetestexcept(FE_ALL_EXCEPT) == original_exceptions, "x87 test failed to restore full fenv");
    status(run_cpu_expert(good, workspace.scratch), CpuExpertStatus::ok, "FP rejection recovery");
#else
    (void)f;
#endif
}

void ticket_aliases(Fixture& f) {
    // Actual live size_t objects back canonical byte-addressed weights. The
    // ticket references below are legal size_t references, not type-punned Q8,
    // float, or expert members. A buggy acceptance would write shared read RAM.
    const auto words = [](const Matrix& matrix) {
        check(matrix.bytes.size() % sizeof(std::size_t) == 0, "word-backed weight capacity");
        std::vector<std::size_t> result(matrix.bytes.size() / sizeof(std::size_t));
        std::memcpy(result.data(), matrix.bytes.data(), matrix.bytes.size());
        return result;
    };
    auto gate_words = words(f.gate), up_words = words(f.up), down_words = words(f.down);
    const auto original_gate = gate_words, original_up = up_words, original_down = down_words;
    const auto original_input = f.input;
    auto gate = f.gate.view(), up = f.up.view(), down = f.down.view();
    gate.weights = std::as_bytes(std::span(gate_words));
    up.weights = std::as_bytes(std::span(up_words));
    down.weights = std::as_bytes(std::span(down_words));
    CpuExpert shared(gate, up, down);
    CpuExpertPool pool({3, false, {}, cpu_expert_max_workers});
    std::array<float, 7680> rejected_output{}, expected{}; rejected_output.fill(canary);
    CpuExpertScratch scratch{};
    status(run_cpu_expert(f.job(3, expected), scratch), CpuExpertStatus::ok, "ticket alias reference");
    auto current = f.job(3, rejected_output); current.expert = &shared;
    const auto unchanged = [&] {
        check(gate_words == original_gate && up_words == original_up && down_words == original_down,
              "ticket alias modified borrowed weight bytes");
        check(bytes_equal<Q8_1>(f.input, original_input), "ticket alias modified borrowed Q8 bytes");
        check(std::all_of(rejected_output.begin(), rejected_output.end(), [](float x) { return x == canary; }),
              "ticket alias modified rejected output");
    };
    const auto reject = [&](const CpuExpertJob& j, std::size_t& alias) {
        const auto saved = alias;
        status(pool.submit(j, alias), CpuExpertStatus::invalid_argument, "ticket writer alias rejection");
        check(alias == saved, "rejected aliased ticket changed");
        unchanged(); ++ticket_alias_rejects;
    };
    for (auto* storage : {&gate_words, &up_words, &down_words}) {
        reject(current, storage->front()); reject(current, storage->back());
    }
    for (auto stage : {CpuExpertStage::gate_up, CpuExpertStage::down}) {
        auto phased = current; phased.stage = stage;
        if (stage == CpuExpertStage::gate_up) phased.output = phased.output.first(3 * 1280);
        else for (auto& x : phased.input) x = x.first(20);
        for (auto* storage : {&gate_words, &up_words, &down_words}) {
            reject(phased, storage->front()); reject(phased, storage->back());
        }
    }
    check(pool.result(0).state == CpuExpertJobState::empty, "current ticket alias enqueued a job");
    // Range-only invalid input/output descriptors also use real size_t ticket
    // storage. They must reject BEFORE interpreting that storage as Q8/float.
    std::vector<std::size_t> range_words(7680 * sizeof(float) / sizeof(std::size_t), 999);
    auto invalid = f.job(3, rejected_output);
    invalid.input[2] = {reinterpret_cast<const Q8_1*>(range_words.data()), 80};
    reject(invalid, range_words.front());
    invalid = f.job(3, {reinterpret_cast<float*>(range_words.data()), 7680});
    reject(invalid, range_words.back());
    check(std::all_of(range_words.begin(), range_words.end(), [](auto x) { return x == 999; }), "range-only ticket storage changed");

    std::array<std::array<float, 7682>, 6> accepted{};
    for (std::size_t i = 0; i < accepted.size(); ++i) {
        accepted[i].fill(canary);
        auto j = f.job(3, {accepted[i].data() + 1, 7680}); j.expert = &shared;
        std::size_t ticket = 999;
        status(pool.submit(j, ticket), CpuExpertStatus::ok, "shared reader submit");
        check(ticket == i, "shared reader ticket");
    }
    // The new job uses DIFFERENT weight storage: only retained readers protect
    // these tickets. Those readers can be queued/running/complete; all states
    // retain their ranges until reset, so the test has no scheduling dependency.
    const auto other = f.job(3, rejected_output);
    for (auto* storage : {&gate_words, &up_words, &down_words}) {
        reject(other, storage->front()); reject(other, storage->back());
    }
    status(pool.drain(), CpuExpertStatus::ok, "ticket rejection preserved prior batch");
    unchanged();
    for (std::size_t i = 0; i < accepted.size(); ++i) {
        const auto r = pool.result(i);
        check(r.state == CpuExpertJobState::complete && r.status == CpuExpertStatus::ok, "ticket rejection changed prior result");
        check(std::memcmp(accepted[i].data() + 1, expected.data(), sizeof(expected)) == 0, "ticket rejection changed prior output");
        check(accepted[i].front() == canary && accepted[i].back() == canary, "shared reader output guards");
    }
    // Completed jobs retain protection too, without reusing their tickets.
    reject(other, gate_words.front());
    std::size_t ticket = 999;
    status(pool.submit(other, ticket), CpuExpertStatus::ok, "healthy job after ticket rejection");
    check(ticket == accepted.size(), "ticket rejection advanced queue");
    status(pool.drain(), CpuExpertStatus::ok, "ticket recovery drain");
    check(bytes_equal<float>(rejected_output, expected), "ticket recovery output");
    status(pool.reset(), CpuExpertStatus::ok, "ticket alias reset");
}

void pool_tests(Fixture& f) {
    throws([] { CpuExpertPool p({0, false, {}, cpu_expert_max_workers}); }, "zero workers");
    throws([] { CpuExpertPool p({16, false, {}, cpu_expert_max_workers}); }, "too many workers");
    throws([] { CpuExpertPool p({1, true, {}, cpu_expert_max_workers}); }, "missing affinity");
    throws([] { CpuExpertPool p({1, true, {static_cast<const int*>(nullptr), 1}, cpu_expert_max_workers}); }, "null affinity span");
    // Worker0 has already started (or is starting) when worker1 fails; repeated
    // partial constructors must safely join, with no terminate/leaked threads.
    for (std::size_t failed_worker : {0U, 1U, 2U}) for (int i = 0; i < 4; ++i)
        throws([&] { CpuExpertPool p({3, false, {}, failed_worker}); }, "partial startup failure");
    std::array<int, 1024> physical{};
    std::size_t count = 987;
#if defined(__linux__)
    status(cpu_expert_allowed_physical_cpus(physical, count), CpuExpertStatus::ok, "physical topology");
    check(count > 0, "no physical CPUs");
    cpu_set_t allowed; CPU_ZERO(&allowed);
    check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "allowed CPU mask");
    for (std::size_t i = 0; i < count; ++i) check(CPU_ISSET(physical[i], &allowed), "fabricated CPU");
    CpuExpertPool pinned({1, true, {physical.data(), 1}, cpu_expert_max_workers});
    status(pinned.drain(), CpuExpertStatus::ok, "empty pinned drain");
    std::array<int, 2> duplicate{physical[0], physical[0]};
    throws([&] { CpuExpertPool p({2, true, duplicate, cpu_expert_max_workers}); }, "duplicate physical CPU");
    const int impossible = -1;
    throws([&] { CpuExpertPool p({1, true, {&impossible, 1}, cpu_expert_max_workers}); }, "invalid affinity CPU");
    std::size_t untouched_count = 771;
    status(cpu_expert_allowed_physical_cpus({}, untouched_count), CpuExpertStatus::invalid_argument, "short CPU destination");
    check(untouched_count == 771, "topology failure mutated count");
    alignas(int) std::array<std::byte, sizeof(physical) + 4> misaligned{};
    status(cpu_expert_allowed_physical_cpus({reinterpret_cast<int*>(misaligned.data() + 1), 1024}, untouched_count),
           CpuExpertStatus::invalid_argument, "misaligned CPU destination");
    check(untouched_count == 771, "misaligned topology output changed count");
    throws([&] { CpuExpertPool p({1, true, {reinterpret_cast<const int*>(misaligned.data() + 1), 1}, cpu_expert_max_workers}); }, "misaligned affinity span");
#endif
    throws([&] { CpuExpertPool p({1, false, {physical.data(), 1}, cpu_expert_max_workers}); }, "no-affinity nonempty cpus");
    CpuExpertPool pool({3, false, {}, cpu_expert_max_workers});
    check(pool.workers() == 3, "pool worker count");
    status(pool.drain(), CpuExpertStatus::ok, "empty drain");
    std::array<std::array<float, 7682>, 30> outputs{};
    std::array<float, 7680> expected{};
    CpuExpertScratch scratch{};
    status(run_cpu_expert(f.job(3, expected), scratch), CpuExpertStatus::ok, "pool reference");
    // Warm up all fixed workers and libm before monitoring the steady loop.
    for (std::size_t i = 0; i < 30; ++i) {
        std::size_t ticket = 999;
        status(pool.submit(f.job(3, {outputs[i].data() + 1, 7680}), ticket), CpuExpertStatus::ok, "warm submit");
    }
    status(pool.drain(), CpuExpertStatus::ok, "warm drain");
    status(pool.reset(), CpuExpertStatus::ok, "warm reset");
    heap_monitor::calls.store(0);
    heap_monitor::enabled.store(true);
    bool normal_ok = true;
    for (int batch = 0; batch < 3; ++batch) {
        for (std::size_t i = 0; i < 30; ++i) {
            outputs[i].fill(canary);
            const int n = int(i % 3) + 1;
            std::size_t ticket = 999;
            normal_ok &= pool.submit(f.job(n, {outputs[i].data() + 1, std::size_t(n) * 2560}), ticket) == CpuExpertStatus::ok;
            normal_ok &= ticket == i;
        }
        std::size_t untouched_ticket = 999;
        normal_ok &= pool.submit(f.job(3, expected), untouched_ticket) == CpuExpertStatus::queue_full;
        normal_ok &= untouched_ticket == 999;
        normal_ok &= pool.drain() == CpuExpertStatus::ok;
        normal_ok &= pool.drain() == CpuExpertStatus::ok;
        for (std::size_t i = 0; i < 30; ++i) {
            const auto r = pool.result(i);
            normal_ok &= r.state == CpuExpertJobState::complete && r.status == CpuExpertStatus::ok;
            const auto n = i % 3 + 1;
            normal_ok &= std::memcmp(outputs[i].data() + 1, expected.data(), n * 2560 * sizeof(float)) == 0;
            normal_ok &= outputs[i].front() == canary && outputs[i][n * 2560 + 1] == canary && outputs[i].back() == canary;
        }
        normal_ok &= pool.reset() == CpuExpertStatus::ok;
    }
    // Also synchronous caller scratch has zero hot C++ heap allocations.
    for (int i = 0; i < 3; ++i) normal_ok &= run_cpu_expert(f.job(3, expected), scratch) == CpuExpertStatus::ok;
    heap_monitor::enabled.store(false);
    check(normal_ok, "steady pool jobs/status/bits/canaries");
    check(heap_monitor::calls.load() == 0, "hot C++ heap allocation");
    check(pool.result(0).state == CpuExpertJobState::empty, "reset left old ticket");
    // Cross-job write/read and write/write alias rejection; completed tickets
    // retain lifetimes until reset, so this is deterministic even on fast CPUs.
    std::size_t ticket = 999;
    status(pool.submit(f.job(3, {outputs[0].data() + 1, 7680}), ticket), CpuExpertStatus::ok, "alias setup");
    const auto reset_while_running = pool.reset();
    check(reset_while_running == CpuExpertStatus::busy || reset_while_running == CpuExpertStatus::ok, "reset in flight status");
    if (reset_while_running == CpuExpertStatus::ok) {
        // The main thread may have been descheduled until after completion.
        // Do not make the fixture depend on winning a race against a worker.
        check(pool.result(0).state == CpuExpertJobState::empty, "completed reset status");
        status(pool.submit(f.job(3, {outputs[0].data() + 1, 7680}), ticket), CpuExpertStatus::ok, "alias resubmit");
    }
    ticket = 999;
    status(pool.submit(f.job(3, {outputs[0].data() + 1, 7680}), ticket), CpuExpertStatus::invalid_argument, "cross-job write alias");
    check(ticket == 999, "rejected submit mutated ticket");
    auto late = f.job(3, {outputs[1].data() + 1, 7680});
    late.input[2] = {reinterpret_cast<const Q8_1*>(outputs[0].data() + 1), 80};
    status(pool.submit(late, ticket), CpuExpertStatus::invalid_argument, "cross-job read alias");
    late = f.job(3, {outputs[1].data() + 1, 7680});
    late.expert = reinterpret_cast<const CpuExpert*>(outputs[0].data() + 2);
    status(pool.submit(late, ticket), CpuExpertStatus::invalid_argument, "cross-job descriptor alias before dereference");
    alignas(4) const auto other_input = f.input;
    late = f.job(3, {reinterpret_cast<float*>(f.input.data()), 7680});
    for (int c = 0; c < 3; ++c) late.input[std::size_t(c)] = {other_input.data() + c * 80, 80};
    status(pool.submit(late, ticket), CpuExpertStatus::invalid_argument, "cross-job output aliases prior reader");
    status(pool.drain(), CpuExpertStatus::ok, "alias drain");
    status(pool.reset(), CpuExpertStatus::ok, "alias reset");
    // An invalid late input must neither enqueue nor affect the next ticket.
    late = f.job(3, {outputs[0].data() + 1, 7680}); late.input[2] = late.input[2].first(79);
    status(pool.submit(late, ticket), CpuExpertStatus::invalid_argument, "pool late invalid input");
    const auto saved = f.input;
    for (auto& b : f.input) { b.d = to_half(65504); b.s = 0; }
    outputs[0].fill(canary); outputs[1].fill(canary);
    status(pool.submit(f.job(3, {outputs[0].data() + 1, 7680}), ticket), CpuExpertStatus::ok, "numeric submit");
    check(ticket == 0, "rejection advanced ticket");
    status(pool.drain(), CpuExpertStatus::numeric_error, "pool error drain");
    check(pool.result(0).state == CpuExpertJobState::complete && pool.result(0).status == CpuExpertStatus::numeric_error, "pool error status");
    for (auto x : outputs[0]) check(x == canary, "pool numeric error published");
    f.input = saved; // Restore only after drain: borrowed inputs were immutable.
    status(pool.submit(f.job(3, {outputs[1].data() + 1, 7680}), ticket), CpuExpertStatus::ok, "success after failed job");
    status(pool.drain(), CpuExpertStatus::numeric_error, "batch retains first error");
    check(pool.result(1).status == CpuExpertStatus::ok, "healthy subsequent job");
    status(pool.reset(), CpuExpertStatus::ok, "error reset");
    // Destructor joins/drains WITHOUT explicit drain. Outputs/weights stay alive
    // outside the scope. Exercise maximum worker bound with no affinity in CI.
    for (auto& out : outputs) out.fill(canary);
    {
        CpuExpertPool in_flight({15, false, {}, cpu_expert_max_workers});
        for (std::size_t i = 0; i < 30; ++i)
            status(in_flight.submit(f.job(3, {outputs[i].data() + 1, 7680}), ticket), CpuExpertStatus::ok, "destructor submit");
    }
    for (const auto& out : outputs) check(std::memcmp(out.data() + 1, expected.data(), sizeof(expected)) == 0, "destructor failed to drain");
}

void zero_and_silu_overflow() {
    Matrix gate(TensorType::Q4_0, 2560, 640, 19), up(TensorType::Q4_0, 2560, 640, 23),
           down(TensorType::Q4_1, 640, 2560, 27);
    // All-zero nibbles with d=1/16, constant positive input => g=-1280.
    // expf(1280) overflows, but the DIVISION and middle remain finite signed
    // zero. This matches the checked Session formula rather than rejecting an
    // otherwise valid expert merely because expf's intermediate is infinite.
    for (auto* m : {&gate, &up}) {
        for (std::size_t i = 0; i < m->bytes.size(); i += sizeof(Q4_0)) {
            const auto d = to_half(1.0f / 16);
            std::memcpy(m->bytes.data() + i, &d, 2);
            std::fill_n(m->bytes.data() + i + 2, 16, std::byte{0});
        }
    }
    CpuExpert e(gate.view(), up.view(), down.view());
    alignas(4) std::array<Q8_1, 80> input{};
    std::array<float, 2560> values{}; values.fill(1);
    quantize_q8(values, input);
    std::array<float, 2560> output{};
    CpuExpertJob j{&e, 1, {std::span<const Q8_1>(input), {}, {}}, output};
    CpuExpertScratch scratch{};
    status(run_cpu_expert(j, scratch), CpuExpertStatus::ok, "finite negative SiLU with exp overflow");
    for (std::size_t i = 0; i < 640; ++i) check(scratch.gate[i] < -1000 && scratch.middle[i] == 0, "negative SiLU division");
    const Q8_1 zero{};
    for (std::size_t i = 0; i < 20; ++i) check(std::memcmp(&scratch.middle_q8[i], &zero, sizeof(zero)) == 0, "zero middle positive-zero ABI");
    for (auto x : output) check(x == 0 && std::isfinite(x), "zero middle output");
    input.fill({});
    status(run_cpu_expert(j, scratch), CpuExpertStatus::ok, "all-zero borrowed input");
    for (auto x : output) check(x == 0, "zero input output");
    // Independent ABI edge fixtures on the EXACT same float inputs, including
    // original-scale half rounding, subnormal d and away-from-zero ties.
    std::array<float, 128> edges{};
    for (std::size_t b = 0; b < 4; ++b) {
        const float scale = b == 3 ? std::ldexp(1.0f, -24) : float(128 + b) / 1024;
        edges[b * 32] = 127.0f * scale;
        for (std::size_t i = 1; i < 32; ++i) edges[b * 32 + i] = (float(i) - 15.5f) * scale;
    }
    std::array<Q8_1, 4> actual{}, expected{};
    quantize_q8(edges, actual); oracle_quant(edges, expected);
    check(bytes_equal<Q8_1>(actual, expected), "same-float quant ties/subnormal/original scale");
}

void deterministic_numeric_rejection() {
    // SEPARATE synthetic contract fixture, never a model/captured-row parity
    // test. Fully canonical Q8: +/-7,620,000 -> half(d)=60,000, raw sum=0,
    // codes +/-127. Finite positive Q4 gate/up overflow HALF middle headers,
    // not FP32 gate/up/SiLU/middle; this works independently of caller rows.
    Q4_0 large{}; large.d = 0x7b53;
    Q4_0 ordinary{}; ordinary.d = 0x3c00;
    for (std::size_t j = 0; j < 16; ++j) {
        large.qs[j] = j % 2 == 0 ? 0xff : 0;
        ordinary.qs[j] = 0x88;
    }
    check(from_half(large.d) == 60000 && to_half(60000) == large.d, "synthetic scale independent half oracle");
    const std::vector<Q4_0> gate(51200, large), up(51200, large), down(51200, ordinary);
    CpuExpert expert({TensorType::Q4_0, 2560, 640, std::as_bytes(std::span(gate))},
                     {TensorType::Q4_0, 2560, 640, std::as_bytes(std::span(up))},
                     {TensorType::Q4_0, 640, 2560, std::as_bytes(std::span(down))});
    std::array<float, 7680> values{};
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = i % 2 == 0 ? 7620000.0f : -7620000.0f;
    alignas(4) std::array<Q8_1, 240> input{}, oracle{};
    quantize_q8(values, input); oracle_quant(values, oracle);
    check(bytes_equal<Q8_1>(input, oracle), "synthetic producer independent exact Q8 bytes");
    for (const auto& b : input) {
        check(b.d == large.d && b.s == 0, "synthetic finite scale/raw sum headers");
        for (std::size_t i = 0; i < 32; ++i) check(b.qs[i] == (i % 2 == 0 ? 127 : -127), "synthetic canonical codes");
    }
    const auto original_input = input;
    // Independent full-block scalar math establishes a FINITE middle whose
    // scale and raw sum overflow binary16. No numeric-error status is assumed
    // solely from deliberately large headers or the candidate's own result.
    std::array<float, 1920> scalar_gate{}, scalar_up{}, scalar_middle{};
    scalar_matrix(expert.gate(), input, 3, scalar_gate);
    scalar_matrix(expert.up(), input, 3, scalar_up);
    for (std::size_t i = 0; i < scalar_middle.size(); ++i) {
        scalar_middle[i] = (scalar_gate[i] / (1.0f + std::exp(-scalar_gate[i]))) * scalar_up[i];
        check(std::isfinite(scalar_gate[i]) && scalar_gate[i] > 0 && std::isfinite(scalar_up[i]) && scalar_up[i] > 0 &&
              std::isfinite(scalar_middle[i]), "synthetic scalar intermediates finite positive");
        check(to_half(scalar_middle[i] / 127.0f) == 0x7c00 && to_half(32.0f * scalar_middle[i]) == 0x7c00,
              "synthetic independent middle headers must overflow half");
    }
    GuardedScratch workspace;
    std::array<float, 7682> output{}, failed_pool{}, recovered{};
    output.fill(canary); failed_pool.fill(canary); recovered.fill(canary);
    const auto make = [&](std::span<const Q8_1> x, int n, std::span<float> out) {
        CpuExpertJob j; j.expert = &expert; j.columns = n; j.output = out;
        for (int c = 0; c < n; ++c) j.input[std::size_t(c)] = x.subspan(std::size_t(c) * 80, 80);
        return j;
    };
    const auto numeric = make(input, 3, {output.data() + 1, 7680});
    status(run_cpu_expert(numeric, workspace.scratch), CpuExpertStatus::numeric_error, "deterministic synthetic arithmetic rejection");
    ++synthetic_numeric_rejects;
    double error = 0;
    parity(workspace.scratch.gate, scalar_gate, 2e-4, 2e-5, error, "synthetic gate independent full-block scalar");
    parity(workspace.scratch.up, scalar_up, 2e-4, 2e-5, error, "synthetic up independent full-block scalar");
    for (float x : workspace.scratch.middle)
        check(std::isfinite(x) && x > 0 && to_half(x / 127.0f) == 0x7c00, "synthetic rejection reached finite middle/half overflow");
    check(std::all_of(output.begin(), output.end(), [](float x) { return x == canary; }), "synthetic error published output");
    workspace.guard();

    // Three valid all-zero stand-in caller rows must stay healthy, even while
    // the same pool retains the SEPARATE synthetic job's arithmetic error.
    values.fill(0);
    alignas(4) std::array<Q8_1, 240> zero_input{};
    quantize_q8(values, zero_input);
    CpuExpertPool pool({2, false, {}, cpu_expert_max_workers});
    std::size_t ticket = 999;
    status(pool.submit(make(input, 3, {failed_pool.data() + 1, 7680}), ticket), CpuExpertStatus::ok, "synthetic numeric pool submit");
    check(ticket == 0, "synthetic numeric pool ticket");
    status(pool.drain(), CpuExpertStatus::numeric_error, "synthetic numeric pool rejection");
    ++synthetic_numeric_rejects;
    check(pool.result(0).state == CpuExpertJobState::complete && pool.result(0).status == CpuExpertStatus::numeric_error,
          "synthetic numeric pool result");
    check(std::all_of(failed_pool.begin(), failed_pool.end(), [](float x) { return x == canary; }), "synthetic pool error published output");
    status(pool.submit(make(zero_input, 3, {recovered.data() + 1, 7680}), ticket), CpuExpertStatus::ok, "zero rows recovery pool submit");
    check(ticket == 1, "zero rows recovery did not preserve failed ticket");
    status(pool.drain(), CpuExpertStatus::numeric_error, "zero rows recovery retains first batch error");
    check(pool.result(1).state == CpuExpertJobState::complete && pool.result(1).status == CpuExpertStatus::ok, "zero rows recovery status");
    for (std::size_t i = 1; i <= 7680; ++i) check(recovered[i] == 0 && std::isfinite(recovered[i]), "zero rows recovery output");
    check(recovered.front() == canary && recovered.back() == canary, "zero rows recovery guards");
    check(bytes_equal<Q8_1>(input, original_input), "synthetic pool changed immutable input");
    status(pool.reset(), CpuExpertStatus::ok, "synthetic failed batch reset");
    for (int n : {1, 2, 3}) {
        output.fill(canary);
        status(run_cpu_expert(make(zero_input, n, {output.data() + 1, std::size_t(n) * 2560}), workspace.scratch),
               CpuExpertStatus::ok, "synthetic expert accepts zero caller rows");
        for (std::size_t i = 1; i <= std::size_t(n) * 2560; ++i) check(output[i] == 0, "zero caller-row direct output");
        check(output.front() == canary && output[std::size_t(n) * 2560 + 1] == canary && output.back() == canary,
              "zero caller-row direct guards");
        workspace.guard();
    }
}

void tiny_packed_input(TensorType down_type) {
    // mx dcd685463d597d31f5ca759d32c94592a2740fa4 quantize.cu:89..101
    // permits half(d)=0 with nonzero RAW sum and codes. Both gate/up AND middle
    // quantization exercise this valid contract; no artificial tiny-block zeros.
    Matrix gate(TensorType::Q4_0, 2560, 640, 19), up(TensorType::Q4_0, 2560, 640, 23),
           down(down_type, 640, 2560, 27);
    for (auto* m : {&gate, &up, &down}) {
        const auto stride = m->type == TensorType::Q4_0 ? sizeof(Q4_0) : sizeof(Q4_1);
        const std::uint16_t d = to_half(m == &down ? 1.0f / 32 : -1.0f / 32);
        const std::uint16_t minimum = to_half(0.5f);
        for (std::size_t i = 0; i < m->bytes.size(); i += stride) {
            std::memcpy(m->bytes.data() + i, &d, 2);
            if (m->type == TensorType::Q4_1) std::memcpy(m->bytes.data() + i + 2, &minimum, 2);
            std::fill_n(m->bytes.data() + i + (m->type == TensorType::Q4_0 ? 2 : 4), 16, std::byte{0});
        }
    }
    CpuExpert expert(gate.view(), up.view(), down.view());
    std::array<float, 7680> values{};
    for (std::size_t i = 0; i < 2560; ++i) {
        values[i] = 3.4e-6f;
        values[2560 + i] = -3.4e-6f;
        values[5120 + i] = i % 32 < 16 ? 3.4e-6f : i % 32 < 24 ? -3.4e-6f : -0.0f;
    }
    alignas(4) std::array<Q8_1, 240> input{}, oracle_input{};
    quantize_q8(values, input);
    oracle_quant(values, oracle_input);
    check(bytes_equal<Q8_1>(input, oracle_input), "tiny expert input independent Q8 bytes");
    for (const auto& b : input) {
        check(b.d == 0 && (b.s & 0x7fffU) != 0, "tiny expert input zero scale/nonzero raw sum");
        check(b.qs[0] != 0, "tiny expert input must retain nonzero codes");
    }
    GuardedScratch workspace;
    std::array<float, 7682> output{}, pooled{};
    output.fill(canary); pooled.fill(canary);
    CpuExpertJob job{&expert, 3, {std::span<const Q8_1>(input).first(80),
                                std::span<const Q8_1>(input).subspan(80, 80),
                                std::span<const Q8_1>(input).subspan(160, 80)},
                     {output.data() + 1, 7680}};
    ++cases;
    status(run_cpu_expert(job, workspace.scratch), CpuExpertStatus::ok, "tiny packed input direct run");
    const auto& scratch = workspace.scratch;
    check(bytes_equal<Q8_1>(input, scratch.gathered_x), "tiny borrowed input gather preserves bytes");
    std::array<float, 1920> expected_gate{}, expected_up{}, expected_middle{};
    std::array<Q8_1, 60> expected_q8{};
    std::array<float, 7680> expected_output{};
    scalar_matrix(gate.view(), input, 3, expected_gate);
    scalar_matrix(up.view(), input, 3, expected_up);
    check(bytes_equal<float>(scratch.gate, expected_gate), "tiny gate exact independent full-block scalar output");
    check(bytes_equal<float>(scratch.up, expected_up), "tiny up exact independent full-block scalar output");
    for (std::size_t i = 0; i < expected_middle.size(); ++i)
        expected_middle[i] = (expected_gate[i] / (1.0f + std::exp(-expected_gate[i]))) * expected_up[i];
    check(bytes_equal<float>(scratch.middle, expected_middle), "tiny middle exact independent scalar SiLU");
    oracle_quant(expected_middle, expected_q8);
    check(bytes_equal<Q8_1>(scratch.middle_q8, expected_q8), "tiny middle independent Q8 bytes");
    for (const auto& b : scratch.middle_q8) {
        check(b.d == 0 && b.s != 0, "tiny middle zero stored scale with nonzero raw sum");
        for (auto q : b.qs) check(q == 127, "tiny middle retains all original-scale codes");
    }
    scalar_matrix(down.view(), expected_q8, 3, expected_output);
    check(bytes_equal<float>(job.output, expected_output), "tiny expert unchanged UNWEIGHTED full scalar output");
    // The narrow down stage must ALSO preserve valid stored-half zero/nonzero
    // raw-sum/codes, not accidentally restore the obsolete all-zero assumption.
    auto projected = job; projected.stage = CpuExpertStage::down;
    for (std::size_t c = 0; c < 3; ++c) projected.input[c] = {expected_q8.data() + c * 20, 20};
    projected.output = {pooled.data() + 1, 7680};
    CpuExpertScratch staged{};
    status(run_cpu_expert(projected, staged), CpuExpertStatus::ok, "tiny opaque down stage");
    check(bytes_equal<float>(projected.output, expected_output) && bytes_equal<Q8_1>(staged.middle_q8, expected_q8),
        "tiny down stage changed packed bytes/output");
    for (auto x : job.output) check(std::isfinite(x) && x != 0 && std::fabs(x) < 0.01f, "tiny expert finite bounded nonzero output");
    check(output.front() == canary && output.back() == canary, "tiny output guards");
    workspace.guard();

    CpuExpertPool pool({1, false, {}, cpu_expert_max_workers});
    auto pool_job = job; pool_job.output = {pooled.data() + 1, 7680};
    std::size_t ticket = 999;
    status(pool.submit(pool_job, ticket), CpuExpertStatus::ok, "tiny packed input pool submit");
    check(ticket == 0, "tiny pool first ticket");
    status(pool.drain(), CpuExpertStatus::ok, "tiny packed input pool drain");
    check(pool.result(ticket).state == CpuExpertJobState::complete && pool.result(ticket).status == CpuExpertStatus::ok,
          "tiny packed input pool result");
    check(bytes_equal<float>(pool_job.output, job.output), "tiny pooled output remains same unweighted output");
    check(pooled.front() == canary && pooled.back() == canary, "tiny pool output guards");
    status(pool.reset(), CpuExpertStatus::ok, "tiny pool reset");

    // Zero-scale headers are now valid, but NaN/Inf in EITHER lane still reject
    // atomically in direct and pooled validation, including the final input block.
    const auto saved = input.back();
    std::array<std::byte, sizeof(CpuExpertScratch)> before{};
    std::memcpy(before.data(), &workspace.scratch, before.size());
    for (const bool scale : {false, true}) {
        for (const std::uint16_t bad : {0x7c00, 0xfc00, 0x7e00, 0xfe00}) {
            if (scale) input.back().d = bad; else input.back().s = bad;
            output.fill(canary); pooled.fill(canary); ticket = 999;
            status(run_cpu_expert(job, workspace.scratch), CpuExpertStatus::invalid_argument, "tiny malformed header direct rejection");
            check(std::memcmp(before.data(), &workspace.scratch, before.size()) == 0, "tiny malformed header changed scratch");
            status(pool.submit(pool_job, ticket), CpuExpertStatus::invalid_argument, "tiny malformed header pool rejection");
            check(ticket == 999, "tiny malformed header advanced pool ticket");
            check(std::all_of(output.begin(), output.end(), [](float x) { return x == canary; }) &&
                  std::all_of(pooled.begin(), pooled.end(), [](float x) { return x == canary; }),
                  "tiny malformed header published output");
            input.back() = saved;
        }
    }
    status(run_cpu_expert(job, workspace.scratch), CpuExpertStatus::ok, "tiny reuse after malformed header");
    check(bytes_equal<float>(job.output, expected_output), "tiny error reuse unweighted output");
}
} // namespace

int main() {
    try {
        if (!qwen::cpu_has_avx2()) {
            throws([] { qwen::CpuExpertPool p({1, false, {}, qwen::cpu_expert_max_workers}); }, "unsupported ISA pool");
            Fixture f(TensorType::Q4_0);
            CpuExpertScratch scratch{};
            std::array<float, 2560> output{}; output.fill(canary);
            status(run_cpu_expert(f.job(1, output), scratch), CpuExpertStatus::unsupported_cpu, "unsupported ISA direct");
            for (auto x : output) check(x == canary, "unsupported ISA output changed");
            std::cout << "{\"kind\":\"cpu_expert_test\",\"avx2\":false,\"execution_skipped\":true,\"gpu_shadow_pending\":true}\n";
            return 0;
        }
        Fixture q40(TensorType::Q4_0), q41(TensorType::Q4_1);
        correctness(q40); correctness(q41);
        staged_projections(q40); staged_projections(q41);
        invalid_arguments(q40); invalid_arguments(q41);
        independent_fp_environment(q40); independent_fp_environment(q41);
        ticket_aliases(q40); ticket_aliases(q41);
        pool_tests(q40); pool_tests(q41);
        zero_and_silu_overflow();
        deterministic_numeric_rejection();
        tiny_packed_input(TensorType::Q4_0); tiny_packed_input(TensorType::Q4_1);
        std::cout.precision(9);
        std::cout << "{\"kind\":\"cpu_expert_test\",\"avx2\":true,\"cases\":" << cases
                  << ",\"checks\":" << checks << ",\"linear_max_abs\":" << linear_error
                  << ",\"middle_max_abs\":" << middle_error << ",\"output_max_abs\":" << output_error
                  << ",\"common_down_max_abs\":" << common_down_error << ",\"source_order_max_abs\":" << ordered_error
                  << ",\"same_float_quant_byte_mismatches\":0,\"scalar_middle_quant_byte_mismatches\":" << byte_mismatches
                   << ",\"scalar_middle_max_code_delta\":" << max_quant_code_delta << ",\"hot_cpp_allocations\":0"
                   << ",\"fp_environment_rejects\":" << fp_environment_rejects << ",\"ticket_alias_rejects\":" << ticket_alias_rejects
                    << ",\"synthetic_numeric_rejects\":" << synthetic_numeric_rejects
                    << ",\"staged_cases\":" << staged_cases << ",\"staged_argument_or_alias_rejects\":" << staged_rejects
                    << ",\"staged_fp_rejects\":" << staged_fp_rejects << ",\"staged_component_bitparity\":true"
                   << ",\"gpu_shadow_pending\":true,\"full_logits_pending\":true,\"device_expf_bitparity_claimed\":false,\"passed\":true}\n";
    } catch (const std::exception& e) {
        heap_monitor::enabled.store(false);
        std::cerr << "cpu-expert-test: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
