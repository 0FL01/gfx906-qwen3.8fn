#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace qwen {
struct Q8_1;

struct SessionConfig {
    int capacity = 4096;
    int expert_slots = 112;
    // Empty disables all synchronous intermediate capture. Diagnostics only.
    std::string trace_directory;
    // Preallocated logical chunk capacity, 1..1024 (canonical projection tiles <=8).
    // Bounded prefill; real 4K/16K and longer-history qualification is separate.
    // Append after the existing fields
    // so positional aggregate initialization keeps its source contract.
    int max_batch_tokens = 1;
    // Diagnostic capture starts at this absolute consumed-token position, 0..capacity.
    // capacity captures nothing; an empty trace_directory still disables tracing.
    // Appended so existing aggregate initializers retain their meaning.
    int trace_first_token = 0;
    // Constructor-only resources. Zero preserves the historical allocations/path.
    // Explicit 1..15 persistent pinned workers; one allowed physical core reserved.
    int cpu_workers = 0;
    // Bounded correctness fixture capture, independent of text tracing. Extra
    // pageable host frames only when explicitly requested with cpu_workers>0.
    bool hybrid_probe = false;
};

// force_cpu is a source-compatible diagnostic name for CPU LINEAR projections
// on every routed assignment; all such work still uses canonical GPU SiLU/Q8.
// No mode dispatches the host-libm whole CpuExpert as a qualified Session path.
enum class SessionHybridMode { disabled, mixed, force_cpu, force_gpu_misses };
struct SessionHybridPolicy {
    SessionHybridMode mode = SessionHybridMode::disabled;
    // Experimental bounded admission, not a measured dispatch threshold. Mixed
    // requires 1..2; forced diagnostics permit 0..2. No resource reallocation.
    int gpu_miss_groups = 2;
    // Synthetic correctness-only one-shot: throw after accepted CPU jobs AND
    // queued GPU admission, before whole gate/up join, GPU middle and CPU down.
    // No CPU-row H2D has run. Requires constructor probe
    // and mixed mode. A false->true policy transition rearms it; no new buffers.
    bool diagnostic_fail_after_admission = false;
};
// Physical READY residency and dispatch are deliberately distinct. Cumulative
// since reset, published only after a successful full call (including head).
struct SessionHybridStats {
    std::uint64_t short_layers = 0, gpu_only_wide_layers = 0;
    std::uint64_t ready_hit_assignments = 0, physical_miss_assignments = 0;
    // Sum(C-1) across hybrid short-window groups, irrespective of residency/dispatch.
    // This is within-window reuse, not an additional physical READY hit.
    std::uint64_t group_reuse_assignments = 0;
    std::uint64_t cpu_groups = 0, cpu_assignments = 0;
    std::uint64_t gpu_hit_groups = 0, gpu_hit_assignments = 0;
    std::uint64_t gpu_miss_groups = 0, gpu_miss_assignments = 0;
    std::uint64_t admitted_groups = 0, evicted_ready_slots = 0;
    std::uint64_t input_extractions = 0, input_bytes = 0, cpu_return_bytes = 0;
    // Constructor probe only: CPU descriptor bytes checked against an independent
    // D2H of original GPU f21. Mismatch fails BEFORE that descriptor's submission.
    std::uint64_t cpu_input_bytes_checked = 0;
    std::uint64_t all_hit_layers = 0, forced_cpu_layers = 0, forced_gpu_layers = 0;
    // Logical cpu_groups counts each expert only ONCE, despite two pool phases.
    // CPUlinear_GPUmiddle: actual accepted projection jobs and middle transfers.
    // Appended; private storage for these fields lives only in the opt-in owner.
    std::uint64_t cpu_gate_up_jobs = 0, cpu_down_jobs = 0, gpu_middle_columns = 0;
    std::uint64_t gpu_middle_batches = 0, paired_gate_up_bytes = 0, middle_q8_bytes = 0;
};
struct SessionHybridIntermediates {
    // Last completed SHORT call only, token-major [N][48][10][2560] unweighted
    // routed down and [N][48][2560] FFN output (including shared). Empty if the
    // constructor probe is off or last success was N>3. No partial publication.
    std::span<const float> routed_down, ffn_output;
};
struct SessionHybridInputs {
    // Constructor probe + last successful short call only. Token-major
    // original_q8 [N][48][80], expert_ids/route_weights/cpu_assignment [N][48][10].
    // input_available [N][48] is 1 ONLY where the existing CPU-input D2H completed.
    // Unavailable rows are unspecified and MUST NOT be read. All-hit does no new
    // extraction. cpu_assignment is actual dispatch, not physical residency.
    // Opaque original GPU bytes, never requantized. Disabled has route metadata
    // but all input_available/cpu_assignment flags zero; wide/reset => empty.
    std::span<const Q8_1> original_q8;
    std::span<const std::int32_t> expert_ids;
    std::span<const float> route_weights;
    std::span<const std::uint8_t> input_available, cpu_assignment;
};
// Exclusive constructor-probe snapshot, separate from published successful-call
// counters. Failure evidence survives reset until a false->true rearm; cache ID
// counts are current host maps. Accepted/queued does NOT claim work was in flight
// when the synthetic throw occurred. Drained means completed ownership chains.
struct SessionHybridProbeDiagnostics {
    std::uint64_t accepted_cpu_jobs = 0, queued_admission_copies = 0;
    // Actual successful CPU-row H2D submission bytes in the failing call, not
    // published statistics (which retain the previous successful call).
    std::uint64_t pending_slots_at_failure = 0, cpu_return_bytes_before_failure = 0;
    std::uint64_t ready_cache_ids = 0, pending_cache_ids = 0;
    int failure_layer = -1;
    bool synthetic_failure = false, cpu_pool_drained = false, gpu_streams_drained = false;
};

// Historical group accounting: count C contributes C-1 within-window reuse hits,
// plus one hit for an actual READY first acquisition or one miss if non-READY.
// Forced CPU/GPU dispatch does not change that physical first-acquisition test.
// Hybrid upload_bytes counts actual RAM->GPU uploads, with no miss*payload identity;
// physical assignment residency is separately exposed in SessionHybridStats.
struct SessionStats {
    std::uint64_t consumed_tokens = 0;
    std::uint64_t expert_hits = 0, expert_misses = 0, expert_upload_bytes = 0;
    double last_completed_ms = 0;
};

// Completed-call routing diagnostics, separate from the historical SessionStats.
// Count logical assignments to a single expert in a layer, before physical tiling.
struct SessionRouteStats {
    // Maximum across all 48 layers of the last successfully completed call.
    std::uint64_t last_max_expert_group_assignments = 0;
    // Cumulative number of call/layer/expert groups with >128 assignments since reset.
    std::uint64_t expert_groups_gt128 = 0;
};

struct SessionDeviceMemory {
    int device = 0, first_layer = 0, last_layer = 0;
    int gdn_layers = 0, qsa_layers = 0;
    std::uint64_t weights = 0, expert_slots = 0, qsa_kv = 0, qsa_index = 0;
    std::uint64_t gdn_state = 0, ple_state = 0, workspace = 0;
    std::uint64_t owned_bytes = 0, owned_peak_bytes = 0, owned_buffers = 0;
    std::uint64_t total_vram = 0, free_vram = 0;
};

// Diagnostic snapshot, not a hot-path allocator or occupied-context test.
// GPU categories cover every live Session Buffer; workspace includes staged
// recurrent/conv state. HIP/context/rocBLAS allocations are visible only in
// total/free VRAM. Legacy host fields report known payload capacities; opt-in
// CPU/plans metadata is reported separately. Allocator overhead and thread stacks
// are excluded; these capacities are not full RSS.
struct SessionMemory {
    int capacity = 0, expert_slots = 0;
    std::array<SessionDeviceMemory, 2> devices{};
    std::uint64_t ram_expert_capacity = 0, ram_expert_payload = 0;
    std::uint64_t host_embedding_capacity = 0, host_logit_capacity = 0;
    std::uint64_t pinned_handoff = 0;
    std::uint64_t expert_payload_reads = 0, expert_payload_bytes_read = 0;
    bool ownership_verified = false;
    // Two pinned payloads per device: band16 for max>3, single triplets for
    // prepared hybrid max<=3. Zero on the historical unprepared short path.
    std::uint64_t pinned_expert_staging = 0;
    // Known payload and explicit metadata capacities, NOT RSS/stacks/HIP bookkeeping.
    std::uint64_t pinned_hybrid_input = 0, pinned_hybrid_output = 0, pinned_hybrid_error = 0;
    std::uint64_t host_hybrid_plans = 0, host_cpu_expert_views = 0;
    std::uint64_t cpu_pool_metadata = 0, cpu_pool_scratch = 0;
    std::array<std::uint64_t, 2> hybrid_contribution_bytes{}, expert_stage_capacity_bytes{};
    int cpu_workers = 0;
    std::uint64_t host_hybrid_probe = 0;
    std::uint64_t host_routing_capacity = 0;
    // RouteGroups object plus its requested fixed assignment payload. Private
    // allocator slack (if any) is excluded, as are other allocator overheads.
    std::uint64_t host_route_group_payload = 0;
    // Explicit 8-byte {token,rank} upload DTOs, bmax*10 per prepared device.
    // GPU buffers are also counted in workspace/the independent owned ledger.
    // Zero for the historical max_batch_tokens=1, cpu_workers=0 allocation path.
    std::uint64_t pinned_route_metadata = 0;
    std::array<std::uint64_t, 2> route_metadata_bytes{};
    // Included in host_hybrid_probe: double-buffered bounded original-input,
    // route ID/weight and availability/dispatch payloads (constructor probe only).
    std::uint64_t host_hybrid_input_probe = 0;
    // Opt-in only: pinned [30][20] middle Q8 per device, plus four dedicated GPU
    // buffers (paired gate/up, middle FP32, middle Q8, independent sticky flag).
    // GPU capacities are included in workspace/the independent Buffer ledger.
    std::uint64_t pinned_hybrid_middle = 0;
    std::array<std::uint64_t, 2> hybrid_gate_up_bytes{}, hybrid_middle_float_bytes{},
        hybrid_middle_q8_bytes{}, hybrid_middle_error_bytes{};
};

// One exclusive interactive session, static 24/24 layer split. Canonical expert
// weights stay in RAM; each layer owns an immutable-weight LRU on its device.
// A failed numeric window invalidates the session until reset(), rather than
// exposing partially advanced cross-layer state. Argument rejection is atomic.
// Returned logits live until the next accepted call.
// No tokenizer/sampler and no linkage to llama/ggml execution.
class Session {
public:
    explicit Session(const std::string& model_path, SessionConfig config = {});
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    std::span<const float> step(std::int32_t token);
    // N=1..max_batch_tokens, token-major [N][248320] completed logits. Validate
    // the entire window's IDs/length/capacity before any state/cache mutation.
    std::span<const float> step_batch(std::span<const std::int32_t> tokens);
    // Marks unusable at entry. Only complete success clears published counters/
    // probe extents and re-enables execution; successful reset retains warm cache.
    void reset();
    // Exclusive, nonconcurrent snapshots; neither accessor synchronizes or allocates.
    SessionStats stats() const;
    // Failed execution/argument rejection retains the last successful diagnostics.
    // Construction and a successfully completed reset() clear both fields.
    SessionRouteStats route_stats() const;
    // Exclusive API. Validates before any drain/mutation; enabling requires
    // constructor cpu_workers>0. Changing a bounded policy allocates nothing.
    void set_hybrid_policy(SessionHybridPolicy);
    SessionHybridPolicy hybrid_policy() const;
    SessionHybridStats hybrid_stats() const;
    // Borrowed last-success views until the next accepted call/destruction.
    // Successful reset publishes empty extents; inspection/rejection preserves.
    SessionHybridIntermediates hybrid_intermediates() const;
    // Same full-call atomic publication/lifetime as intermediates. A fixture's
    // subsequent numerical comparison failure does not invalidate these views.
    SessionHybridInputs hybrid_inputs() const;
    // Empty evidence/counts when constructor hybrid_probe is off. No drain/alloc.
    SessionHybridProbeDiagnostics hybrid_probe_diagnostics() const;
    // Drains owned compute/copy streams and restores the caller's current device.
    // Does not modify logits, state, cache maps, counters or logical visibility.
    SessionMemory memory() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
