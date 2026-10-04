#pragma once

#include "quant.hpp"
#include <array>
#include <memory>
#include <string_view>

namespace qwen {

inline constexpr std::size_t cpu_expert_max_columns = 3;
inline constexpr std::size_t cpu_expert_max_workers = 15;
inline constexpr std::size_t cpu_expert_max_jobs = 30;

enum class CpuExpertStatus {
    ok, invalid_argument, unsupported_cpu, numeric_error, busy, queue_full,
    affinity_error, worker_error, stopped
};
[[nodiscard]] std::string_view cpu_expert_status_name(CpuExpertStatus) noexcept;

// Immutable borrowed canonical RAM. Construction checks geometry, byte ranges,
// alignment and EVERY weight scale. The owner must keep both this object and its
// payload alive and immutable until run_cpu_expert()/drain()/destruction returns.
// No repack, dequantized weight copy, HIP, route weight or rank accumulation.
class CpuExpert {
public:
    CpuExpert(QMatrix gate, QMatrix up, QMatrix down);
    [[nodiscard]] QMatrix gate() const noexcept { return gate_; }
    [[nodiscard]] QMatrix up() const noexcept { return up_; }
    [[nodiscard]] QMatrix down() const noexcept { return down_; }
private:
    QMatrix gate_, up_, down_;
};

// Whole remains the local oracle/shadow component (host-libm middle), not the
// qualified hybrid path. Hybrid uses two CPU projection phases with the CURRENT
// canonical GPU SiLU + Q8 producer between them. No CPU middle requantization.
enum class CpuExpertStage { whole, gate_up, down };

// GPU-produced Q8_1: 36 bytes, half(d), half(RAW sum), codes selected with the
// original FP32 d. Each active span is 80 blocks (whole/gate_up) or 20 (down),
// 4-byte aligned. Down borrows opaque middle Q8 from the GPU producer.
// Columns may be nonadjacent in the caller's staging buffer. Unused spans empty.
// Input is copied byte-for-byte, NEVER reconstructed or requantized. Encoded
// finite scales/raw sums are checked; their original floats are unavailable.
struct CpuExpertJob {
    const CpuExpert* expert = nullptr;
    int columns = 0; // Only 1/2/3.
    std::array<std::span<const Q8_1>, cpu_expert_max_columns> input{};
    // whole/down: [columns][2560] UNWEIGHTED. gate_up: interleaved
    // [columns][2][640], gate first then up. Exact extents, no partial output.
    std::span<float> output{};
    CpuExpertStage stage = CpuExpertStage::whole; // Appended; old aggregates unchanged.
};

// One instance per synchronous caller or pool worker. Only live prefixes are
// used. All arrays are fixed; no allocation or thread creation on a normal job.
// Scratch must be disjoint from all input/weight/output ranges and exclusively
// owned during run. Inspect live gate/up/middle/Q8/down prefixes after success
// for a WHOLE shadow fixture; gate_up only publishes gathered_x/gate/up prefixes,
// down only middle_q8/down. Other arrays/tails are untouched by those stages.
// Their content after an execution error is unspecified.
struct alignas(64) CpuExpertScratch {
    std::array<Q8_1, 240> gathered_x;
    std::array<Q8_1, 60> middle_q8;
    std::array<float, 1920> gate, up, middle;
    std::array<float, 7680> down;
};

// Strict FP compilation (-ffp-contract=off, no fast-math), round-to-nearest,
// no FTZ/DAZ. Both x87 and MXCSR rounding are checked. Runtime AVX2/FMA/F16C
// check precedes all target-attributed code.
// Linear arithmetic adapts Session's Q4 two-fragment/FMA/64-XOR order, NOT
// quant.cpp's full-dot/serial oracle. SiLU uses host expf, separate float add,
// division and multiply. Device expf parity and full logits remain unqualified.
// All argument checks precede scratch/output mutation. Output is copied ONLY
// after the entire requested stage is finite/successful; errors leave output intact.
[[nodiscard]] CpuExpertStatus run_cpu_expert(const CpuExpertJob&,
                                           CpuExpertScratch&) noexcept;

enum class CpuExpertJobState { empty, queued, running, complete };
struct CpuExpertJobResult {
    CpuExpertJobState state = CpuExpertJobState::empty;
    CpuExpertStatus status = CpuExpertStatus::ok;
};

struct CpuExpertPoolConfig {
    std::size_t workers = 1; // Explicit, 1..15; no hardware_concurrency guess.
    bool pin_workers = true;
    // When pinning, exactly workers distinct allowed physical representatives.
    // Copied in the constructor, not borrowed afterward. No-affinity CI mode
    // requires this span empty. It does not establish target performance.
    std::span<const int> cpus{};
    // Constructor-only diagnostic injection: max_workers disables; otherwise
    // that worker reports startup failure (after thread creation) for tests.
    std::size_t test_fail_worker = cpu_expert_max_workers;
};

// Linux: actual sched_getaffinity mask + sysfs (package,core), ascending first
// allowed sibling per physical core. No NUMA fabrication. All-or-nothing writes;
// insufficient destination returns invalid_argument. No pinning side effect.
[[nodiscard]] CpuExpertStatus cpu_expert_allowed_physical_cpus(
    std::span<int> destination, std::size_t& count) noexcept;

// Session-dedicated persistent pool, minimal fixed queue/status/mutex/CVs.
// Startup allocates fixed state, checks ISA/config/topology, and waits for workers.
// Linux scratch has private anonymous, page-rounded mappings allocated/first-
// touched after worker pinning, disjoint from controller-written metadata.
// Any partial failure joins every started thread before freeing its scratch
// and throwing. Destructor drains accepted jobs then joins (not cancellation).
// A single owner serializes submit/drain/result/reset/destruction; workers never
// call HIP. No external mutation of borrowed inputs/weights/output during jobs.
// Result/input spans across ALL accepted jobs must be disjoint if either writes;
// read/read sharing is allowed. Keep descriptors' owners alive through drain.
// submit treats its ticket reference as a writer: overlap with its job descriptor,
// current input/expert/weights/output or any retained job range rejects without
// changing the ticket or borrowed bytes, even for already-completed tickets.
// Tickets 0..29 are stable until reset, including completed jobs; submit never
// overwrites old tickets. drain waits for all and returns the first error in
// ticket order. A failed batch can contain successful jobs: the GPU consumer
// must gate publication on drain()==ok, not just one completed job. reset only
// succeeds when no job is outstanding and clears statuses, not borrowed output.
class CpuExpertPool {
public:
    explicit CpuExpertPool(CpuExpertPoolConfig);
    ~CpuExpertPool();
    CpuExpertPool(const CpuExpertPool&) = delete;
    CpuExpertPool& operator=(const CpuExpertPool&) = delete;
    CpuExpertPool(CpuExpertPool&&) = delete;
    CpuExpertPool& operator=(CpuExpertPool&&) = delete;
    [[nodiscard]] CpuExpertStatus submit(const CpuExpertJob&, std::size_t& ticket) noexcept;
    [[nodiscard]] CpuExpertStatus drain() noexcept;
    [[nodiscard]] CpuExpertJobResult result(std::size_t ticket) const noexcept;
    [[nodiscard]] CpuExpertStatus reset() noexcept;
    [[nodiscard]] std::size_t workers() const noexcept;
    // Immutable allocation capacities after successful startup. Excludes stacks,
    // allocator/thread-runtime bookkeeping and any borrowed Session payloads.
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t scratch_bytes() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace qwen
