#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include "model.hpp"

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
    // Constructor-only bounded query-private QSA capacity, 1..128. Does not
    // enable batching; tile=1 preserves the historical GPU allocations.
    int attention_query_tile = 1;
    // Constructor-only target prefix snapshots/tap. Requires max_batch_tokens>=3.
    // Off adds no GPU buffers, transfers or arithmetic; host sizeof does change.
    bool speculative_checkpoints = false;
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

// Experimental batch API invocations in successfully completed full calls only,
// cumulative since reset. Disabled/N1 calls contribute nothing. These are NOT
// physical kernel counts or selected-ID/visibility observations. Rejection and
// failed execution retain the last published counters; reset clears them.
struct SessionAttentionStats {
    std::uint64_t batch_calls = 0, query_rows = 0;
    std::uint64_t multiquery_calls = 0, multiquery_rows = 0;
    std::uint64_t singleton_tail_calls = 0, max_query_rows = 0;
};

// Logical pending-window ownership. Slot k is after k consumed inputs, including
// slot 0 PRE-window. Only a completed verify publishes N+1 valid slots (max 4).
// A restore consumes the window even when k==N. No persistent KV snapshot.
struct SessionCheckpointState {
    bool enabled = false, pending = false;
    std::uint64_t start_position = 0;
    int inputs = 0, valid_slots = 0;
    bool operator==(const SessionCheckpointState&) const = default;
};

// Typed BORROWED GPU view, never a CPU-dereferenceable span. Device1 FP32
// [rows][10240], the layer47 final widened combine BEFORE root HC mixing.
// Latest successful ordinary/verify forward; restore never changes these rows.
// Copy/read on device before the next SUCCESSFUL forward/reset/destruction.
// Rejection and failed execution preserve this view AND its physical bytes;
// execution failure still requires reset before further forward/restore.
// k>0 selects row k-1 as next carry; restoring k==0 provides no carry (the
// caller must retain its earlier carry). No concurrent access during a call.
struct SessionTargetTap {
    int device = 1;
    const float* pointer = nullptr;
    int rows = 0, width = 10240;
    std::uint64_t first_position = 0;
    bool operator==(const SessionTargetTap&) const = default;
};

// Immutable payload borrowing only. Session must outlive its MtpSession borrower;
// both owners and all views are exclusive/nonconcurrent, including destruction.
// No model reopen, embedding duplication, Q6 repack or root-HC sharing.
struct SessionQ6Head {
    static constexpr TensorType type = TensorType::Q6_K;
    int device = 1, input = 2560, output = 248320;
    const void* pointer = nullptr;
    std::uint64_t bytes = 0;
};
struct SessionEmbedding {
    static constexpr TensorType type = TensorType::Q4_0;
    std::span<const std::byte> cpu_bytes;
    std::uint64_t row_bytes = 0;
    int width = 2560, rows = 248320;
};
struct SessionSharedWeights {
    const Model* descriptor = nullptr; // already-open actual target inventory
    SessionQ6Head output;
    SessionEmbedding embedding;
};

// Separate opt-in diagnostics, cumulative until successful reset. Restore ONLY
// changes the logical consumed cursor, never these completed-work counters or
// historical expert/upload/route/hybrid/attention diagnostics. Prefix calls are
// invocations of the existing prefix-enabled launch APIs, NOT profiler kernel
// counts: GDN has two prefix-writing kernels, PLE and QSA tail have one each.
struct SessionSpeculativeStats {
    std::uint64_t target_forward_rows = 0, verify_windows = 0, verify_rows = 0;
    std::uint64_t restore_calls = 0, retained_inputs = 0;
    std::uint64_t gdn_prefix_calls = 0, ple_prefix_calls = 0, qsa_tail_prefix_calls = 0;
    std::array<std::uint64_t, 4> restored_prefixes{};
    bool operator==(const SessionSpeculativeStats&) const = default;
};

// Actual backing Buffer bytes, already included in workspace/owned_bytes.
// staged/output reuse the existing f(15)/f(17) scratch buffers; private_workspace
// describes only the fixed tile prefix of staged, plus the four private buffers.
// Selection stride/width is 2051 IDs, 512 blocks, 2 counts (10260 bytes/query).
struct SessionAttentionMemory {
    std::uint64_t gathered_key_bytes = 0, gathered_value_bytes = 0;
    std::uint64_t partial_output_bytes = 0, partial_max_sum_bytes = 0;
    std::uint64_t staged_buffer_bytes = 0, output_buffer_bytes = 0;
    std::uint64_t selected_id_bytes = 0, selected_block_bytes = 0, selected_count_bytes = 0;
    std::uint64_t private_workspace_bytes = 0;
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
    int attention_query_tile = 1;
    std::array<SessionAttentionMemory, 2> attention{};
    // Snapshot/tap GPU capacities also included in workspace and independent
    // live-Buffer traversal. Exactly FOUR prefix slots per mutable state.
    bool speculative_checkpoints = false;
    // Each tap capacity is max_batch_tokens*10240*sizeof(float), on device1.
    // The second buffer protects published physical bytes on execution failure.
    std::array<std::uint64_t, 2> checkpoint_recurrent_bytes{}, checkpoint_history_bytes{},
        checkpoint_qsa_tail_bytes{}, checkpoint_ple_history_bytes{}, target_tap_bytes{}, target_tap_staging_bytes{};
    // Actual object sizeof; config is INCLUDED in Impl (do not add twice).
    // Owner only when enabled; hash config vector capacities plus requested
    // fixed history payload are counted. Private history-vector allocator slack,
    // allocator overhead, RSS and thread stacks are excluded.
    std::uint64_t host_session_impl_bytes = 0, host_session_config_bytes = 0;
    std::uint64_t host_speculative_owner_bytes = 0, host_speculative_hash_payload_bytes = 0;
};

// One exclusive interactive session, static 24/24 layer split. Canonical expert
// weights stay in RAM; each layer owns an immutable-weight LRU on its device.
// A failed numeric window invalidates the session until reset(), rather than
// exposing partially advanced cross-layer state. Argument rejection is atomic.
// Returned logits live until the next accepted call; with constructor snapshots
// enabled, last-success views instead survive failures until the next successful
// forward/reset/destruction. No concurrent calls or view access during execution.
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
    // Exclusive N=1..3 target forward. Requires constructor snapshots and no
    // pending verify. Full argument preflight precedes any mutation. Publishes
    // PRE-window + all chronological prefix states only after network/head
    // success. Returns token-major full logits. With snapshots enabled, borrowed
    // logits/tap survive rejection, restore and execution failure, until the
    // next successful forward/reset/destruction. No concurrent view access.
    // Ordinary successful forward invalidates a previous pending verification;
    // invalid ordinary arguments do not. Failed execution requires reset; its
    // slots are unusable and checkpoint_state() rejects inspection until reset.
    std::span<const float> verify_window(std::span<const std::int32_t> tokens);
    // Exclusive one-shot restore of the latest pending verify, k=0..N. Drains
    // pool and both device streams before copying. Restores GDN/history, PLE
    // hash/conv and QSA raw tail; cursor=start+k hides main-Q4/pooled suffixes,
    // which are overwritten on reuse. No logits/tap writes or counter rewind.
    // Successful restore consumes pending ownership. Rejection preserves ALL
    // published spans/diagnostics. A restore execution failure requires reset.
    void restore_prefix(int retained_inputs);
    // Exclusive allocation-free getters. Only checkpoint_state requires usable
    // execution; last-success tap/stats remain inspectable after failure.
    SessionCheckpointState checkpoint_state() const;
    SessionTargetTap target_tap() const;
    // Drains existing streams, restores current device; immutable constructor seam.
    SessionSharedWeights borrow_mtp_weights() const;
    SessionSpeculativeStats speculative_stats() const;
    // Marks unusable at entry. Only complete success clears published counters/
    // probe extents and re-enables execution; successful reset retains warm cache.
    void reset();
    // Exclusive, nonconcurrent snapshots; neither accessor synchronizes or allocates.
    SessionStats stats() const;
    // Failed execution/argument rejection retains the last successful diagnostics.
    // Construction and a successfully completed reset() clear both fields.
    SessionRouteStats route_stats() const;
    // Exclusive allocation-free mode toggle. Validate before draining owned
    // CPU/GPU work; explicit enable with a nonempty trace directory is rejected
    // even outside its capture range. N1 always uses the historical API/body.
    // Neither toggle changes state, logits, cache budget, counters or probes.
    // Reset retains the mode. Getter is the explicit policy, not a kernel claim.
    void set_attention_batch(bool enabled);
    bool attention_batch() const;
    SessionAttentionStats attention_stats() const;
    // Exclusive API. Validates before any drain/mutation; enabling requires
    // constructor cpu_workers>0. Changing a bounded policy allocates nothing.
    void set_hybrid_policy(SessionHybridPolicy);
    SessionHybridPolicy hybrid_policy() const;
    SessionHybridStats hybrid_stats() const;
    // Borrowed last-success views until the next accepted call/destruction;
    // with snapshots enabled, until next successful forward/reset/destruction.
    // Successful reset publishes empty extents; rejection/failure preserves bytes.
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
