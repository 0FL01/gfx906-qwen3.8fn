#include "cpu_expert.hpp"

#include <algorithm>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CPU_EXPERT_X86 1
#define CPU_EXPERT_TARGET __attribute__((target("avx2,fma,f16c")))
#else
#define CPU_EXPERT_X86 0
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "cpu_expert requires strict floating point"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

namespace qwen {
namespace {
struct Range { std::uintptr_t begin, end; };
bool range(const void* p, std::size_t bytes, std::size_t alignment, Range& r) noexcept {
    const auto start = reinterpret_cast<std::uintptr_t>(p);
    if (!p || start % alignment != 0 ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - start) return false;
    r = {start, start + bytes};
    return true;
}
bool overlaps(Range a, Range b) noexcept { return a.begin < b.end && b.begin < a.end; }
std::uint16_t half(const std::byte* p) noexcept {
    std::uint16_t h;
    std::memcpy(&h, p, sizeof(h));
    return h;
}
bool finite_half(std::uint16_t h) noexcept { return (h & 0x7c00U) != 0x7c00U; }
bool fp_environment() noexcept {
    if (std::fegetround() != FE_TONEAREST) return false;
#if CPU_EXPERT_X86
    // fegetround() can report x87 nearest while MXCSR was changed separately.
    if ((_mm_getcsr() & (0x6000U | 0x8040U)) != 0) return false; // RC + FTZ/DAZ.
#endif
    return true;
}
bool matrix_valid(QMatrix m, bool down) noexcept {
    if (m.input != (down ? 640 : 2560) || m.output != (down ? 2560 : 640) ||
        (m.type != TensorType::Q4_0 && (!down || m.type != TensorType::Q4_1))) return false;
    const auto block_bytes = m.type == TensorType::Q4_0 ? sizeof(Q4_0) : sizeof(Q4_1);
    constexpr std::size_t blocks = 51200; // Both canonical shapes, no unchecked products.
    Range r{};
    if (m.weights.size() != blocks * block_bytes || !range(m.weights.data(),
            m.weights.size(), alignof(std::uint16_t), r)) return false;
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto* p = m.weights.data() + b * block_bytes;
        if (!finite_half(half(p)) || (m.type == TensorType::Q4_1 &&
                                    !finite_half(half(p + 2)))) return false;
    }
    return true;
}
struct JobRanges { Range output; std::array<Range, 7> reads; std::size_t count = 0; };
// Range-only preflight MUST NOT dereference Q8 or the borrowed expert before
// the pool checks cross-job writers. Rejecting an alias after reading it would
// itself race a running worker's final output copy.
CpuExpertStatus descriptor_ranges(const CpuExpertJob& job, JobRanges& ranges) noexcept {
    switch (job.stage) {
    case CpuExpertStage::whole: case CpuExpertStage::gate_up: case CpuExpertStage::down: break;
    default: return CpuExpertStatus::invalid_argument;
    }
    if (!job.expert || job.columns < 1 || job.columns > 3) return CpuExpertStatus::invalid_argument;
    const std::size_t width = job.stage == CpuExpertStage::gate_up ? 1280 : 2560;
    const std::size_t blocks = job.stage == CpuExpertStage::down ? 20 : 80;
    if (job.output.size() != std::size_t(job.columns) * width ||
        !range(job.output.data(), job.output.size_bytes(), alignof(float), ranges.output))
        return CpuExpertStatus::invalid_argument;
    Range descriptor{};
    if (!range(job.expert, sizeof(CpuExpert), alignof(CpuExpert), descriptor) || overlaps(descriptor, ranges.output))
        return CpuExpertStatus::invalid_argument;
    ranges.reads[ranges.count++] = descriptor;
    for (std::size_t c = 0; c < 3; ++c) {
        const auto x = job.input[c];
        if (c >= std::size_t(job.columns)) {
            if (!x.empty()) return CpuExpertStatus::invalid_argument;
            continue;
        }
        Range r{};
        if (x.size() != blocks || !range(x.data(), x.size_bytes(), 4, r) || overlaps(r, ranges.output))
            return CpuExpertStatus::invalid_argument;
        ranges.reads[ranges.count++] = r;
    }
    return CpuExpertStatus::ok;
}
CpuExpertStatus weight_ranges(const CpuExpertJob& job, JobRanges& ranges) noexcept {
    for (auto m : {job.expert->gate(), job.expert->up(), job.expert->down()}) {
        Range r{};
        if (!range(m.weights.data(), m.weights.size(), 2, r) || overlaps(r, ranges.output))
            return CpuExpertStatus::invalid_argument;
        ranges.reads[ranges.count++] = r;
    }
    return CpuExpertStatus::ok;
}
CpuExpertStatus input_values(const CpuExpertJob& job) noexcept {
    for (int c = 0; c < job.columns; ++c) {
        for (const auto& b : job.input[std::size_t(c)]) {
            if (!finite_half(b.d) || !finite_half(b.s) || (b.d & 0x8000U) != 0)
                return CpuExpertStatus::invalid_argument;
            // Canonical mx Q8_1 stores half(d,sum) but selects codes with the
            // original FP32 d (quant.hpp): stored d=0 does not imply s/qs=0.
        }
    }
    return CpuExpertStatus::ok;
}
bool conflicts(const JobRanges& a, const JobRanges& b) noexcept {
    if (overlaps(a.output, b.output)) return true;
    for (std::size_t i = 0; i < a.count; ++i) if (overlaps(a.reads[i], b.output)) return true;
    for (std::size_t i = 0; i < b.count; ++i) if (overlaps(b.reads[i], a.output)) return true;
    return false;
}
bool writer_conflicts(Range writer, const JobRanges& job) noexcept {
    if (overlaps(writer, job.output)) return true;
    for (std::size_t i = 0; i < job.count; ++i)
        if (overlaps(writer, job.reads[i])) return true;
    return false;
}

#if CPU_EXPERT_X86
// KEEP canonical Q4 and the 36-byte raw-sum Q8 ABI (quant.hpp/quant.cpp).
// ADAPT arithmetic, not donor source: mx dcd685463d597d31f5ca759d32c94592a2740fa4
// vecdotq.cuh/mmvq.cu, as qualified in our hip/linear.hip::matrix32/accumulate32:
// two dot16 fragments, k stride64, wave1+wave0 THEN ascending XOR 1..32.
// Q4_0 inner+acc explicit FMA; Q4_1 half-RNE products, inner FMA, separate add.
// WRITE OURS: AVX2 nibble dot fragments and borrowed expert/pool. Donor order
// RECON16: furnace 905021dbad71c5056ef51f9fd45d545403fc989c lacks canonical Q4;
// reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8 multi-N reuse is a pattern,
// NOT its incompatible 40-byte activation ABI. No GPU layout/core changes.
template<TensorType Type>
CPU_EXPERT_TARGET void matrix_gpu_order(QMatrix m, const Q8_1* input, int columns,
                                        float* output) noexcept {
    constexpr std::size_t stride = Type == TensorType::Q4_0 ? 18 : 20;
    constexpr std::size_t offset = Type == TensorType::Q4_0 ? 2 : 4;
    const auto mask = _mm_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    const auto blocks = std::size_t(m.input / 32);
    for (int row = 0; row < m.output; ++row) {
        alignas(32) float partial[3][128]{};
        for (std::size_t k = 0; k < blocks; ++k) {
            const auto* b = m.weights.data() + (std::size_t(row) * blocks + k) * stride;
            const float d4 = _cvtsh_ss(half(b));
            float m4 = 0;
            if constexpr (Type == TensorType::Q4_1) m4 = _cvtsh_ss(half(b + 2));
            // Load/unpack each weight block ONCE for all 1..3 columns.
            const auto packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset));
            const auto low = _mm_and_si128(packed, mask);
            const auto high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            const auto codes = _mm256_set_m128i(high, low);
            const auto tid = 2 * (k % 64);
            for (int c = 0; c < columns; ++c) {
                const auto& x = input[std::size_t(c) * blocks + k];
                const auto q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs));
                // Unsigned nibble*signed int8 including -128: max pair3840,
                // so maddubs cannot saturate. Preserve the TWO dot16 results.
                const auto quads = _mm256_madd_epi16(_mm256_maddubs_epi16(codes, q8), ones);
                const auto both = _mm_add_epi32(_mm256_castsi256_si128(quads),
                                               _mm256_extracti128_si256(quads, 1));
                const auto dots = _mm_hadd_epi32(both, both);
                const int dot[2] = {_mm_cvtsi128_si32(dots), _mm_extract_epi32(dots, 1)};
                const float dx = _cvtsh_ss(x.d), sx = _cvtsh_ss(x.s);
                float dd = 0, ms = 0;
                if constexpr (Type == TensorType::Q4_1) {
                    dd = _cvtsh_ss(_cvtss_sh(d4 * dx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    ms = _cvtsh_ss(_cvtss_sh(m4 * sx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                }
                for (std::size_t f = 0; f < 2; ++f) {
                    auto& acc = partial[c][tid + f];
                    if constexpr (Type == TensorType::Q4_0)
                        acc = std::fma(d4, std::fma(float(dot[f]), dx, -4.0f * sx), acc);
                    else acc = acc + std::fma(float(dot[f]), dd, ms * 0.5f);
                }
            }
        }
        for (int c = 0; c < columns; ++c) {
            for (std::size_t lane = 0; lane < 64; ++lane)
                partial[c][lane] = partial[c][lane] + partial[c][lane + 64];
            for (std::size_t xor_mask = 1; xor_mask < 64; xor_mask *= 2)
                for (std::size_t lane = 0; lane < 64; ++lane) if ((lane & xor_mask) == 0) {
                    const float sum = partial[c][lane] + partial[c][lane ^ xor_mask];
                    partial[c][lane] = sum;
                    partial[c][lane ^ xor_mask] = sum;
                }
            output[std::size_t(c) * std::size_t(m.output) + std::size_t(row)] = partial[c][0];
        }
    }
}
#endif

CpuExpertStatus execute(const CpuExpertJob& job, CpuExpertScratch& scratch) noexcept {
    // Called only after structural/input/scratch checks; still check the actual
    // worker's ISA/FP environment before entering a target-attributed function.
    if (!cpu_has_avx2()) return CpuExpertStatus::unsupported_cpu;
    if (!fp_environment()) return CpuExpertStatus::numeric_error;
#if CPU_EXPERT_X86
    if (job.stage == CpuExpertStage::down) {
        for (int c = 0; c < job.columns; ++c)
            std::memcpy(scratch.middle_q8.data() + std::size_t(c) * 20, job.input[std::size_t(c)].data(), 20 * sizeof(Q8_1));
        if (job.expert->down().type == TensorType::Q4_0)
            matrix_gpu_order<TensorType::Q4_0>(job.expert->down(), scratch.middle_q8.data(), job.columns, scratch.down.data());
        else matrix_gpu_order<TensorType::Q4_1>(job.expert->down(), scratch.middle_q8.data(), job.columns, scratch.down.data());
        for (std::size_t i = 0; i < job.output.size(); ++i)
            if (!std::isfinite(scratch.down[i])) return CpuExpertStatus::numeric_error;
        std::memcpy(job.output.data(), scratch.down.data(), job.output.size_bytes());
        return CpuExpertStatus::ok;
    }
    for (int c = 0; c < job.columns; ++c)
        std::memcpy(scratch.gathered_x.data() + std::size_t(c) * 80, job.input[std::size_t(c)].data(), 80 * sizeof(Q8_1));
    matrix_gpu_order<TensorType::Q4_0>(job.expert->gate(), scratch.gathered_x.data(), job.columns, scratch.gate.data());
    matrix_gpu_order<TensorType::Q4_0>(job.expert->up(), scratch.gathered_x.data(), job.columns, scratch.up.data());
    const auto middle_size = std::size_t(job.columns) * 640;
    if (job.stage == CpuExpertStage::gate_up) {
        for (std::size_t i = 0; i < middle_size; ++i)
            if (!std::isfinite(scratch.gate[i]) || !std::isfinite(scratch.up[i])) return CpuExpertStatus::numeric_error;
        // Every column's projections are finite BEFORE the first output write.
        for (int c = 0; c < job.columns; ++c) {
            std::memcpy(job.output.data() + std::size_t(c) * 1280, scratch.gate.data() + std::size_t(c) * 640, 640 * sizeof(float));
            std::memcpy(job.output.data() + std::size_t(c) * 1280 + 640, scratch.up.data() + std::size_t(c) * 640, 640 * sizeof(float));
        }
        return CpuExpertStatus::ok;
    }
    for (std::size_t i = 0; i < middle_size; ++i) {
        const float g = scratch.gate[i], u = scratch.up[i];
        if (!std::isfinite(g) || !std::isfinite(u)) return CpuExpertStatus::numeric_error;
        const float activated = g / (1.0f + std::exp(-g));
        const float middle = activated * u;
        if (!std::isfinite(activated) || !std::isfinite(middle)) return CpuExpertStatus::numeric_error;
        scratch.middle[i] = middle;
    }
    try {
        // Reuse the EXISTING quantizer on exactly these FP32 middle values.
        quantize_q8({scratch.middle.data(), middle_size}, {scratch.middle_q8.data(), middle_size / 32});
    } catch (const std::invalid_argument&) { return CpuExpertStatus::numeric_error; }
      catch (...) { return CpuExpertStatus::worker_error; }
    if (job.expert->down().type == TensorType::Q4_0)
        matrix_gpu_order<TensorType::Q4_0>(job.expert->down(), scratch.middle_q8.data(), job.columns, scratch.down.data());
    else matrix_gpu_order<TensorType::Q4_1>(job.expert->down(), scratch.middle_q8.data(), job.columns, scratch.down.data());
    for (std::size_t i = 0; i < job.output.size(); ++i)
        if (!std::isfinite(scratch.down[i])) return CpuExpertStatus::numeric_error;
    std::memcpy(job.output.data(), scratch.down.data(), job.output.size_bytes());
    return CpuExpertStatus::ok;
#else
    (void)job; (void)scratch;
    return CpuExpertStatus::unsupported_cpu;
#endif
}

#if defined(__linux__)
bool topology_integer(int cpu, const char* name, int& value) noexcept {
    char path[160], contents[64];
    const int n = std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, name);
    if (n <= 0 || std::size_t(n) >= sizeof(path)) return false;
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    ssize_t bytes;
    do { bytes = read(fd, contents, sizeof(contents) - 1); } while (bytes < 0 && errno == EINTR);
    close(fd);
    if (bytes <= 0 || std::size_t(bytes) >= sizeof(contents) - 1) return false;
    contents[bytes] = '\0';
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(contents, &end, 10);
    if (errno || end == contents || parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
    while (*end == '\n' || *end == ' ' || *end == '\t') ++end;
    if (*end != '\0') return false;
    value = int(parsed);
    return true;
}
#endif
} // namespace

std::string_view cpu_expert_status_name(CpuExpertStatus status) noexcept {
    switch (status) {
    case CpuExpertStatus::ok: return "ok";
    case CpuExpertStatus::invalid_argument: return "invalid_argument";
    case CpuExpertStatus::unsupported_cpu: return "unsupported_cpu";
    case CpuExpertStatus::numeric_error: return "numeric_error";
    case CpuExpertStatus::busy: return "busy";
    case CpuExpertStatus::queue_full: return "queue_full";
    case CpuExpertStatus::affinity_error: return "affinity_error";
    case CpuExpertStatus::worker_error: return "worker_error";
    case CpuExpertStatus::stopped: return "stopped";
    }
    return "unknown";
}
CpuExpert::CpuExpert(QMatrix gate, QMatrix up, QMatrix down) : gate_(gate), up_(up), down_(down) {
    if (!matrix_valid(gate, false) || !matrix_valid(up, false) || !matrix_valid(down, true))
        throw std::invalid_argument("cpu expert: expected finite canonical Q4_0 gate/up [2560,640], Q4_0/Q4_1 down [640,2560]");
}
CpuExpertStatus run_cpu_expert(const CpuExpertJob& job, CpuExpertScratch& scratch) noexcept {
    JobRanges ranges{};
    const auto descriptor = descriptor_ranges(job, ranges);
    if (descriptor != CpuExpertStatus::ok) return descriptor;
    const auto weights = weight_ranges(job, ranges);
    if (weights != CpuExpertStatus::ok) return weights;
    Range workspace{};
    if (!range(&scratch, sizeof(scratch), alignof(CpuExpertScratch), workspace) || overlaps(workspace, ranges.output))
        return CpuExpertStatus::invalid_argument;
    Range job_descriptor{};
    if (!range(&job, sizeof(job), alignof(CpuExpertJob), job_descriptor) || overlaps(workspace, job_descriptor) ||
        overlaps(ranges.output, job_descriptor)) return CpuExpertStatus::invalid_argument;
    for (std::size_t i = 0; i < ranges.count; ++i)
        if (overlaps(workspace, ranges.reads[i])) return CpuExpertStatus::invalid_argument;
    const auto values = input_values(job);
    if (values != CpuExpertStatus::ok) return values;
    return execute(job, scratch);
}
CpuExpertStatus cpu_expert_allowed_physical_cpus(std::span<int> destination, std::size_t& count) noexcept {
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) return CpuExpertStatus::affinity_error;
    std::array<int, CPU_SETSIZE> packages{}, cores{}, representatives{};
    std::size_t n = 0;
    for (int id = 0; id < CPU_SETSIZE; ++id) if (CPU_ISSET(id, &allowed)) {
        int package = -1, core = -1;
        if (!topology_integer(id, "physical_package_id", package) || !topology_integer(id, "core_id", core))
            return CpuExpertStatus::affinity_error;
        bool seen = false;
        for (std::size_t i = 0; i < n; ++i) if (packages[i] == package && cores[i] == core) seen = true;
        if (!seen) { packages[n] = package; cores[n] = core; representatives[n++] = id; }
    }
    if (n == 0) return CpuExpertStatus::affinity_error;
    if (destination.size() < n || !destination.data()) return CpuExpertStatus::invalid_argument;
    Range output{}, counter{};
    if (!range(destination.data(), n * sizeof(int), alignof(int), output) ||
        !range(&count, sizeof(count), alignof(std::size_t), counter) || overlaps(output, counter))
        return CpuExpertStatus::invalid_argument;
    std::copy_n(representatives.begin(), n, destination.begin());
    count = n;
    return CpuExpertStatus::ok;
#else
    (void)destination; (void)count;
    return CpuExpertStatus::affinity_error;
#endif
}

struct CpuExpertPool::Impl {
    struct Worker {
        std::thread thread;
        CpuExpertScratch* scratch = nullptr;
        std::size_t scratch_bytes = 0;
        int cpu = -1;
        // Called only after shutdown has joined all started threads, including
        // constructor unwinding. Metadata never shares these scratch pages.
        ~Worker() {
            if (!scratch) return;
            std::destroy_at(scratch);
#if defined(__linux__)
            (void)munmap(scratch, scratch_bytes);
#else
            std::free(scratch);
#endif
        }
        bool initialize_scratch() noexcept {
#if defined(__linux__)
            const long page = sysconf(_SC_PAGESIZE);
            if (page <= 0 || std::size_t(page) % alignof(CpuExpertScratch) != 0) return false;
            const auto page_bytes = std::size_t(page);
            const auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
            if (page_bytes > limit || sizeof(CpuExpertScratch) > limit - (page_bytes - 1)) return false;
            scratch_bytes = ((sizeof(CpuExpertScratch) + page_bytes - 1) / page_bytes) * page_bytes;
            void* memory = mmap(nullptr, scratch_bytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (memory == MAP_FAILED) { scratch_bytes = 0; return false; }
#else
            // Preserve no-affinity support off Linux; private anonymous page
            // placement is a Linux target contract, not a portability claim.
            scratch_bytes = sizeof(CpuExpertScratch);
            void* memory = std::aligned_alloc(alignof(CpuExpertScratch), scratch_bytes);
            if (!memory) { scratch_bytes = 0; return false; }
#endif
            scratch = ::new (memory) CpuExpertScratch; // Default-init: no implicit zero fill.
            // Allocation AND first writes occur here, after worker pinning.
            // Touch the full mapped capacity, including the final page's tail.
            std::memset(memory, 0, scratch_bytes);
            return true;
        }
    };
    struct Record { CpuExpertJob job{}; JobRanges ranges{}; CpuExpertJobResult result{}; };
    std::mutex mutex;
    std::condition_variable work, completed;
    std::unique_ptr<Worker[]> worker;
    std::array<Record, cpu_expert_max_jobs> records{};
    std::size_t worker_count, submitted = 0, next = 0, outstanding = 0, started = 0;
    std::size_t fail_worker;
    bool stop = false;
    CpuExpertStatus startup = CpuExpertStatus::ok;

    explicit Impl(CpuExpertPoolConfig config) : worker_count(config.workers), fail_worker(config.test_fail_worker) {
        if (worker_count < 1 || worker_count > cpu_expert_max_workers ||
            (config.pin_workers ? config.cpus.size() != worker_count : !config.cpus.empty()) ||
            (fail_worker != cpu_expert_max_workers && fail_worker >= worker_count))
            throw std::invalid_argument("cpu expert pool: invalid worker/affinity/test configuration");
        if (!cpu_has_avx2()) throw std::runtime_error("cpu expert pool: AVX2/FMA/F16C unavailable");
        if (!fp_environment()) throw std::runtime_error("cpu expert pool: requires nearest FP rounding without FTZ/DAZ");
        if (config.pin_workers) {
            Range cpu_list{};
            if (!range(config.cpus.data(), worker_count * sizeof(int), alignof(int), cpu_list))
                throw std::invalid_argument("cpu expert pool: invalid affinity span range/alignment");
#if defined(__linux__)
            std::array<int, CPU_SETSIZE> allowed{};
            std::size_t count = 0;
            if (cpu_expert_allowed_physical_cpus(allowed, count) != CpuExpertStatus::ok)
                throw std::runtime_error("cpu expert pool: allowed physical topology unavailable");
            for (std::size_t i = 0; i < worker_count; ++i) {
                if (std::find(allowed.begin(), allowed.begin() + std::ptrdiff_t(count), config.cpus[i]) == allowed.begin() + std::ptrdiff_t(count))
                    throw std::invalid_argument("cpu expert pool: CPU is not an allowed physical representative");
                for (std::size_t j = 0; j < i; ++j) if (config.cpus[i] == config.cpus[j])
                    throw std::invalid_argument("cpu expert pool: duplicate physical CPU");
            }
#else
            throw std::runtime_error("cpu expert pool: pinning requires Linux");
#endif
        }
        worker.reset(new Worker[worker_count]); // Controller touches metadata only.
        for (std::size_t i = 0; i < worker_count; ++i) worker[i].cpu = config.pin_workers ? config.cpus[i] : -1;
        try {
            for (std::size_t i = 0; i < worker_count; ++i)
                worker[i].thread = std::thread([this, i] { loop(i); });
            std::unique_lock lock(mutex);
            completed.wait(lock, [this] { return started == worker_count; });
            if (startup != CpuExpertStatus::ok) {
                lock.unlock();
                throw std::runtime_error("cpu expert pool: worker startup failed");
            }
        } catch (...) { shutdown(); throw; }
    }
    void shutdown() noexcept {
        { std::lock_guard lock(mutex); stop = true; }
        work.notify_all();
        for (std::size_t i = 0; i < worker_count; ++i)
            if (worker[i].thread.joinable()) worker[i].thread.join();
    }
    void loop(std::size_t id) noexcept {
        CpuExpertStatus status = CpuExpertStatus::ok;
        if (worker[id].cpu >= 0) {
#if defined(__linux__)
            cpu_set_t mask;
            CPU_ZERO(&mask); CPU_SET(worker[id].cpu, &mask);
            if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask)) status = CpuExpertStatus::affinity_error;
#endif
        }
        if (!cpu_has_avx2()) status = CpuExpertStatus::unsupported_cpu;
        if (!fp_environment()) status = CpuExpertStatus::numeric_error;
        if (id == fail_worker) status = CpuExpertStatus::worker_error;
        if (status == CpuExpertStatus::ok && !worker[id].initialize_scratch()) status = CpuExpertStatus::worker_error;
        {
            std::lock_guard lock(mutex);
            if (status != CpuExpertStatus::ok && startup == CpuExpertStatus::ok) startup = status;
            ++started;
        }
        completed.notify_all();
        if (status != CpuExpertStatus::ok) return;
        for (;;) {
            std::size_t ticket;
            CpuExpertJob job;
            {
                std::unique_lock lock(mutex);
                work.wait(lock, [this] { return stop || next < submitted; });
                if (next == submitted) { if (stop) return; else continue; }
                ticket = next++;
                records[ticket].result.state = CpuExpertJobState::running;
                job = records[ticket].job;
            }
            const auto result_status = execute(job, *worker[id].scratch);
            {
                std::lock_guard lock(mutex);
                records[ticket].result = {CpuExpertJobState::complete, result_status};
                --outstanding;
            }
            completed.notify_all();
        }
    }
};

CpuExpertPool::CpuExpertPool(CpuExpertPoolConfig config) : impl_(std::make_unique<Impl>(config)) {}
CpuExpertPool::~CpuExpertPool() { impl_->shutdown(); }
std::size_t CpuExpertPool::workers() const noexcept { return impl_->worker_count; }
std::size_t CpuExpertPool::metadata_bytes() const noexcept {
    return sizeof(*impl_) + impl_->worker_count * sizeof(Impl::Worker);
}
std::size_t CpuExpertPool::scratch_bytes() const noexcept {
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < impl_->worker_count; ++i) bytes += impl_->worker[i].scratch_bytes;
    return bytes;
}
CpuExpertStatus CpuExpertPool::submit(const CpuExpertJob& job, std::size_t& ticket) noexcept {
    JobRanges ranges{};
    // Worker scratch is private, unreachable by the owner through this API.
    std::lock_guard lock(impl_->mutex);
    if (impl_->stop) return CpuExpertStatus::stopped;
    Range ticket_range{}, job_descriptor{};
    if (!range(&ticket, sizeof(ticket), alignof(std::size_t), ticket_range) ||
        !range(&job, sizeof(job), alignof(CpuExpertJob), job_descriptor) ||
        overlaps(ticket_range, job_descriptor)) return CpuExpertStatus::invalid_argument;
    // The ticket is a writer too. Check retained jobs before dereferencing ANY
    // borrowed expert/Q8: a ticket may live in their size_t-backed weight RAM.
    for (std::size_t i = 0; i < impl_->submitted; ++i)
        if (writer_conflicts(ticket_range, impl_->records[i].ranges)) return CpuExpertStatus::invalid_argument;
    const auto descriptor = descriptor_ranges(job, ranges);
    if (descriptor != CpuExpertStatus::ok) return descriptor;
    if (writer_conflicts(ticket_range, ranges)) return CpuExpertStatus::invalid_argument;
    for (std::size_t i = 0; i < impl_->submitted; ++i)
        if (conflicts(ranges, impl_->records[i].ranges)) return CpuExpertStatus::invalid_argument;
    const auto weights = weight_ranges(job, ranges);
    if (weights != CpuExpertStatus::ok) return weights;
    if (writer_conflicts(ticket_range, ranges)) return CpuExpertStatus::invalid_argument;
    for (std::size_t i = 0; i < impl_->submitted; ++i)
        if (conflicts(ranges, impl_->records[i].ranges)) return CpuExpertStatus::invalid_argument;
    const auto values = input_values(job);
    if (values != CpuExpertStatus::ok) return values;
    // Reject the owner's bad FP environment before accepting a ticket. Workers
    // independently recheck their own environment in execute() for every stage.
    if (!fp_environment()) return CpuExpertStatus::numeric_error;
    if (impl_->submitted == cpu_expert_max_jobs) return CpuExpertStatus::queue_full;
    const auto t = impl_->submitted++;
    impl_->records[t] = {job, ranges, {CpuExpertJobState::queued, CpuExpertStatus::ok}};
    ++impl_->outstanding;
    ticket = t;
    impl_->work.notify_one();
    return CpuExpertStatus::ok;
}
CpuExpertStatus CpuExpertPool::drain() noexcept {
    std::unique_lock lock(impl_->mutex);
    impl_->completed.wait(lock, [this] { return impl_->outstanding == 0; });
    for (std::size_t i = 0; i < impl_->submitted; ++i)
        if (impl_->records[i].result.status != CpuExpertStatus::ok) return impl_->records[i].result.status;
    return CpuExpertStatus::ok;
}
CpuExpertJobResult CpuExpertPool::result(std::size_t ticket) const noexcept {
    std::lock_guard lock(impl_->mutex);
    if (ticket >= impl_->submitted) return {CpuExpertJobState::empty, CpuExpertStatus::invalid_argument};
    return impl_->records[ticket].result;
}
CpuExpertStatus CpuExpertPool::reset() noexcept {
    std::lock_guard lock(impl_->mutex);
    if (impl_->outstanding != 0) return CpuExpertStatus::busy;
    for (auto& record : impl_->records) record = {};
    impl_->submitted = impl_->next = 0;
    return CpuExpertStatus::ok;
}
} // namespace qwen
