#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qwen {

// Fixed target indexer geometry, not the main attention's Q24/KV2/D256.
// RoPE parameters are deliberately explicit: use the ATTENTION rotary dimension
// and its inverse frequencies, not frequencies derived from index_head_dim.
struct QsaIndexConfig {
    std::size_t head_dim = 128;
    std::size_t query_heads = 4;
    std::size_t key_heads = 1;
    std::size_t compress_ratio = 4;
    std::size_t capacity = 131072; // positive, <= 131072; smaller fixtures allowed
    std::size_t rotary_dim = 0;   // positive even prefix, <= 128, required
    float rms_epsilon = 0.0f;     // positive finite, required
    float rope_attention_scale = 1.0f; // multiplier of BOTH cos and sin
};

struct QsaIndexParameters {
    // Direct GGUF multipliers: mx conversion/qwen4exp.py already folds HF +1.
    std::span<const float> query_norm_weight; // [128], shared by all four heads
    std::span<const float> key_norm_weight;   // [128], applied AFTER mean pooling
    // [rotary_dim/2], copied at construction. Text-only split-half RoPE on the
    // prefix: pair i with i+rotary_dim/2; the remaining dimensions stay unchanged.
    // Default HF frequencies are 1 / theta^(2*i/ATTENTION_rotary_dim). Providing
    // this array also avoids guessing theta/scaling from indexer geometry.
    std::span<const float> inverse_frequencies;
};

// Small, preallocated snapshot. No pooled-history copy. Default construction is
// unbound; save() binds to one owner's current reset epoch and retained prefix.
// Copies are independent snapshots; moved-from snapshots cannot be used.
class QsaIndexCheckpoint {
public:
    QsaIndexCheckpoint();
    std::size_t consumed_tokens() const noexcept { return consumed_; }
    std::span<const float> raw_tail() const noexcept;

private:
    friend class QsaIndexCpu;
    std::vector<float> tail_;
    std::size_t consumed_ = 0;
    std::uint64_t owner_ = 0, epoch_ = 0, prefix_tag_ = 0;
};

// Independent numeric FP32 CPU semantic prerequisite, not a production GPU path.
// Inputs are already projected raw index Q/K. No index V or full raw history.
// Raw K -> canonical kv.hpp Q4_0 roundtrip -> chronological mean4 -> RMS/direct
// gamma -> RoPE at block START (4*b). Q -> RMS/direct gamma -> RoPE at consumed-1.
// score[b] = sum_h ReLU(dot(Q_h, K_b)) / sqrt(128). Contiguous text positions only.
// Compile with -ffp-contract=off, without fast-math: ascending FP32 reductions.
//
// Constructor copies parameters and allocates cache/workspaces once. Successful
// append/score/save/restore/reset calls allocate nothing and create no threads.
// One owner; not concurrent. Public views are read-only, with stable storage but
// contents/logical extent can change on append/restore/reset; do not cast away const.
// Pass score()'s exactly completed_blocks() values and consumed_tokens() to the
// existing qsa_select for whole blocks + actual tail. That DIAGNOSTIC selector
// has its own per-call allocation; it is outside this allocation-free numeric API.
class QsaIndexCpu {
public:
    QsaIndexCpu(QsaIndexConfig config, QsaIndexParameters parameters);
    QsaIndexCpu(const QsaIndexCpu&) = delete;
    QsaIndexCpu& operator=(const QsaIndexCpu&) = delete;
    QsaIndexCpu(QsaIndexCpu&&) = delete;
    QsaIndexCpu& operator=(QsaIndexCpu&&) = delete;

    const QsaIndexConfig& config() const noexcept { return config_; }
    std::size_t consumed_tokens() const noexcept { return consumed_; }
    std::size_t completed_blocks() const noexcept { return consumed_ / 4; }
    std::span<const float> completed_keys() const noexcept;
    std::span<const float> raw_tail() const noexcept;

    void reset(); // invalidates all saved checkpoints, preserves allocations
    void save(QsaIndexCheckpoint& checkpoint) const;
    // Restores/truncates logical length and tail. Even a newer snapshot may be
    // restored while its completed prefix remains retained and unoverwritten.
    // Reset, different owner, or an overwritten required prefix is rejected.
    void restore(const QsaIndexCheckpoint& checkpoint);

    // Exact token-major [tokens][128]. Optional tokens+1 preallocated snapshots:
    // prefixes[0] before chunk; prefixes[i+1] after input i. For verify3 the
    // session's consumed-input contract restores slot 1+accepted_drafts.
    // Invalid sizes/geometry/overflow/alias/nonfinite input/unrepresentable Q4
    // scales are rejected before ANY state/prefix mutation (whole chunk check).
    // Arithmetic failure throws runtime_error atomically for the failing token;
    // earlier successful tokens remain consumed, later prefix slots untouched.
    void append(std::span<const float> raw_keys, std::size_t tokens,
                std::span<QsaIndexCheckpoint> prefixes = {});

    // Exact raw Q [4][128], scores [completed_blocks()]. Requires consumed>0;
    // visibility includes the current token. No future or partial block scores.
    // Invalid input throws invalid_argument; FP32 arithmetic failure throws
    // runtime_error. Neither case writes scores or changes semantic state.
    // All borrowed input/output spans must be disjoint from owned memory;
    // output must also be disjoint from query, even for an empty block prefix.
    void score(std::span<const float> raw_query, std::span<float> scores);

private:
    bool aliases_owned(const void* data, std::size_t bytes) const;
    void validate_destination(const QsaIndexCheckpoint& checkpoint) const;
    void norm_rope(std::span<float> values, std::span<const float> weight,
                   std::size_t position) const;
    void append_one(std::span<const float> raw_key);
    QsaIndexConfig config_;
    std::vector<float> query_weight_, key_weight_, inverse_frequencies_;
    std::vector<float> keys_, score_scratch_;
    std::vector<std::uint64_t> block_tags_;
    std::array<float, 3 * 128> tail_{};
    std::array<float, 128> raw_scratch_{}, pooled_scratch_{};
    std::array<float, 4 * 128> query_scratch_{};
    std::size_t consumed_ = 0, retained_blocks_ = 0;
    std::uint64_t owner_ = 0, epoch_ = 1, next_tag_ = 0;
};

} // namespace qwen
