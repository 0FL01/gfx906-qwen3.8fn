#pragma once

#include "session.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace qwen {

struct MtpSessionConfig {
    int capacity = 4096; // 1..131072; N1..3 resources always constructor-owned
    bool diagnostic_probe = false; // bounded fixture capture/fault/poison only
    bool diagnostic_first_row = false; // requires diagnostic_probe; no normal-path copies
    int diagnostic_probe_position = 0; // absolute row, 0..capacity-1; nonzero requires flag above
};

// Immutable device1 F32 widened row, BEFORE either target or own HC head.
// Caller guarantees actual residency/readable allocation and exclusive lifetime.
// row pointers need not be contiguous: teacher rebuild uses saved carry,V0,V1.
// null/zero-elements is an explicit ZERO row, allowed ONLY at absolute position0.
struct MtpHiddenRow {
    int device = 1;
    const float* pointer = nullptr;
    std::size_t elements = 0; // exactly 10240 when nonnull
    static MtpHiddenRow zero() noexcept { return {}; }
};
struct MtpTap {
    int device = 1;
    const float* pointer = nullptr;
    int rows = 0, width = 10240;
    std::uint64_t first_position = 0;
    MtpHiddenRow row(int index) const;
    bool operator==(const MtpTap&) const = default;
};
enum class MtpTeacherHead { skip, publish };

struct MtpSessionStats {
    std::uint64_t consumed_rows = 0, full_rows = 0, kv_only_rows = 0;
    std::uint64_t full_calls = 0, teacher_calls = 0, restore_calls = 0;
    std::uint64_t routed_groups = 0, routed_assignments = 0, grouped_reuses = 0;
    std::uint64_t head_rows = 0;
    double last_completed_ms = 0;
    bool operator==(const MtpSessionStats&) const = default;
};
struct MtpSessionMemory {
    // Every owned GPU Buffer is covered; HIP/context/rocBLAS are visible through
    // free/total VRAM only. Host fields are known payload capacities, not RSS,
    // parser inventories, allocator overhead or stack accounting.
    std::uint64_t ram_expert_payload = 0, ram_expert_capacity = 0;
    std::uint64_t gpu_experts = 0, gpu_nonexperts = 0, gpu_kv = 0;
    std::uint64_t gpu_dense_scratch = 0, gpu_workspace = 0;
    std::uint64_t owned_bytes = 0, owned_buffers = 0, peak_owned_bytes = 0;
    std::uint64_t host_logits = 0, host_probe = 0, host_staging = 0;
    std::uint64_t model_payload_reads = 0, model_payload_bytes = 0;
    std::uint64_t expert_payload_reads = 0, expert_payload_bytes = 0;
    std::uint64_t shared_target_payload_reads = 0; // MUST remain zero
    std::uint64_t borrowed_head_bytes = 0, borrowed_embedding_bytes = 0;
    std::uint64_t free_vram = 0, total_vram = 0;
    int immutable_ready_slots = 0;
    bool ownership_verified = false;
};
struct MtpSessionProbe {
    // Last successful FULL call only, token-major. CPU views, diagnostics only.
    std::span<const float> raw_attention_gates; // [N][6144], no sigmoid conversion
    std::span<const float> head_input;          // [N][2560], distinct from D[N]
    std::span<const std::int32_t> expert_ids;    // [N][10], original rank order
    std::span<const float> route_weights;       // [N][10], original raw bits
};
// Optional exact graph-site snapshots of ONE configured absolute row, CPU-owned by this
// session. Flat token/branch-major F32; gamma/embedding repetitions are retained.
// Default position0 preserves the original first-row diagnostic. Published only
// by a successful FULL window containing diagnostic_probe_position; reset clears
// validity, retaining configuration. KV-only/restore/invalid/failure and FULL
// windows excluding that position preserve these views/bytes. They expire on
// reset, next successful FULL window containing that position,
// or destruction. No model payload read or extra GPU allocation is involved.
struct MtpFirstRowProbe {
    bool published = false;
    std::span<const float> token_embedding;         // 2560
    std::span<const float> hidden_norm;             // 4*2560, distinct gamma slices
    std::span<const float> embedding_norm_repeated; // 4*2560
    std::span<const float> fusion_concat;           // 4*5120, embedding FIRST
    std::span<const float> fusion_output;           // 4*2560
    std::span<const float> attention_mixed;         // 2560, before Q/K/V
    std::span<const float> attention_pregate;       // 6144, after inverse V Hadamard
    std::span<const float> attention_output;        // 2560, after output projection
    std::span<const float> attention_combined;      // 4*2560
    std::span<const float> ffn_mixed;               // 2560, before router/experts
    std::span<const float> ffn_output;              // 2560, routed+shared
    std::span<const float> D;                       // 4*2560, before own HC head
    std::span<const float> head_input;              // 2560, own HC head output
    std::uint64_t first_position = 0;               // valid when published; selected absolute row
};

// Trained ONE blk48, compression0. Two proposals invoke this SAME block twice.
// Borrowed Session outlives this owner; no concurrent target/MTP calls/view reads.
// Fresh teacher row0=(ZERO,x0,0), rowp=(T[p-1],x[p],p). This is the production
// history convention, NOT a claim about HF training. Proposals at prefix N:
// (T[N-1],c,N)->D[N]/q0; (D[N],a,N+1)->D[N+1]/q1.
// Discard proposal KV with restore_prefix(N) BEFORE target verify[c,a,b].
// save_target_carry() copies T[N-1] BEFORE successful target verify invalidates
// its BorrowedTap; rebuild rows (saved,c),(V0,a),(V1,b), then restore_prefix(N+r),
// where r=SpeculativeTerminalDecision::restore_prefix is the retained input count.
// Carry V[r-1] belongs
// to the coordinator; the last pending token is excluded. No sampling here.
//
// Invalid arguments preserve all state/publications. Execution/numeric failure
// drains, requires reset, preserves prior physical tap/logits/probe bytes.
// Successful KV-only teacher append preserves those publications as well.
// restore masks a suffix; every newly visible slot is overwritten before read.
// No recurrent checkpoints are needed for this dense feed-forward block.
//
// nextn.hnorm source pin: mx dcd685463d597d31f5ca759d32c94592a2740fa4,
// src/models/qwen4exp.cpp:438-450. Layout [N][4][2560] (ggml [2560,4,N]):
// hidden -> RMS(axis0=2560, epsilon=geometry.rms_epsilon) -> gamma[4][2560].
// norm[n,b,j] = (hidden[n,b,j]/sqrt(mean_j(hidden[n,b,j]^2)+epsilon))
//                * gamma[b*2560+j]; all four gamma slices remain distinct.
// embedding[N][2560] -> RMS2560 -> repeat per branch -> embedding-FIRST
// concat[N][4][5120] -> original Q8_0 eh_proj -> widened[N][4][2560].
// Zero-first-row is the production convention above, not HF training proof.
// Trained forward parity still requires the independent teacher oracle.
class MtpSession {
public:
    MtpSession(Session& target, const std::string& sidecar_path, MtpSessionConfig = {});
    ~MtpSession();
    MtpSession(const MtpSession&) = delete;
    MtpSession& operator=(const MtpSession&) = delete;
    MtpSession(MtpSession&&) = delete;
    MtpSession& operator=(MtpSession&&) = delete;

    std::span<const float> step(std::span<const MtpHiddenRow> hidden,
                               std::span<const std::int32_t> tokens);
    // Explicit KV-only skips AFTER canonical K/V append (no dense attention,
    // FFN or head). publish executes full block, own HC head and borrowed Q6.
    std::span<const float> teacher_append(std::span<const MtpHiddenRow> hidden,
        std::span<const std::int32_t> tokens, MtpTeacherHead head);
    void restore_prefix(std::uint64_t absolute_rows); // only 0..current logical prefix
    void reset(); // retains immutable all-512 cache; clears visibility/publications

    void save_target_carry(int target_tap_row); // synchronous own FP32 40960B copy
    MtpHiddenRow saved_target_carry() const;
    MtpTap own_tap() const;
    std::span<const float> logits() const;
    std::uint64_t prefix() const;
    bool requires_reset() const;
    MtpSessionStats stats() const;
    MtpSessionMemory memory() const; // drains; independent live-buffer traversal
    MtpSessionProbe probe() const;
    MtpFirstRowProbe first_row_probe() const;

    // Explicit opt-in fixture hooks, no normal-path corruption/failure dispatch.
    void diagnostic_poison_future(); // fills ONLY invisible suffix with NaN scales
    void diagnostic_fail_after_kv(bool enabled);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwen
