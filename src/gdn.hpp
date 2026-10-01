#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qwen {

struct GdnConfig {
    std::size_t key_heads = 16;
    std::size_t value_heads = 48;
    std::size_t key_head_dim = 128;
    std::size_t value_head_dim = 128;
    std::size_t conv_width = 4;
    float rms_epsilon = 1.0e-6f;

    // Validate positive dimensions, integral V/K head ratio and checked sizes.
    std::size_t key_elements() const;
    std::size_t value_elements() const;
    std::size_t qkv_elements() const;
    std::size_t recurrent_elements() const;
    std::size_t conv_elements() const;
};

// Existing GGUF ABI, not raw HF grouped V-head order. The mx converter's
// _LinearAttentionVReorderBase maps HF head (k * ratio + r) to (r * K + k).
// Consequently V head h uses Q/K head h % K. ALL V-side arrays (including the
// V part of qkv/conv weights, z, alpha, beta, dt_bias and ssm_a) share this order.
// Norm weight is a direct multiplier shared by all heads, NOT (1 + weight).
struct GdnParameters {
    std::span<const float> conv_weights; // [qkv_feature][conv_width], oldest first
    std::span<const float> dt_bias;      // [value_head]
    std::span<const float> ssm_a;        // [value_head], GGUF -exp(HF A_log), <= 0
    std::span<const float> norm_weight;  // [value_head_dim]
};

struct GdnInput {
    // Token-major contiguous rows. qkv row = Q[K*Dk], K[K*Dk], V[H*Dv].
    // These are projections BEFORE convolution/activation; z/alpha/beta are
    // raw projections (no sigmoid/softplus applied by the caller).
    std::span<const float> qkv;
    std::span<const float> z;
    std::span<const float> alpha;
    std::span<const float> beta;
};

// Frozen external state ABI for decode/chunk/verify/checkpoint:
// FP32 recurrent[(h * Dv + v) * Dk + k] = S_h[k,v], i.e. K contiguous.
// FP32 conv_history[feature * (W-1) + age], oldest -> most recent RAW projection.
// No padding, ring index, swizzle or backend-specific representation in snapshots.
// consumed_tokens counts inputs actually forwarded, never pending sampled tokens.
// Allocate checkpoints once; save/restore copies into existing buffers.
class GdnCheckpoint {
public:
    explicit GdnCheckpoint(GdnConfig config = {});
    const GdnConfig& config() const noexcept { return config_; }
    std::span<const float> recurrent() const noexcept { return recurrent_; }
    std::span<const float> conv_history() const noexcept { return conv_history_; }
    std::uint64_t consumed_tokens() const noexcept { return consumed_tokens_; }

private:
    friend class GdnCpu;
    GdnConfig config_;
    std::vector<float> recurrent_;
    std::vector<float> conv_history_;
    std::uint64_t consumed_tokens_ = 0;
};

// Own scalar FP32 correctness path, not a production GPU prefill implementation.
// Parameters, state and scratch allocate only in the constructor; no per-call
// allocations or threads. Helpers start after the four input projections and
// finish before out_proj. One owner; calls on the same instance are not concurrent.
class GdnCpu {
public:
    GdnCpu(GdnConfig config, GdnParameters parameters);
    GdnCpu(const GdnCpu&) = delete;
    GdnCpu& operator=(const GdnCpu&) = delete;

    const GdnCheckpoint& state() const noexcept { return state_; }
    void reset() noexcept;
    void save(GdnCheckpoint& checkpoint) const;
    void restore(const GdnCheckpoint& checkpoint);

    void step(GdnInput input, std::span<float> output);

    // Sequential CPU chunk reference, output [tokens][H*Dv]. Optional prefixes
    // must have EXACTLY tokens+1 preallocated, geometry-matched checkpoints:
    // prefixes[0] = pre-chunk state, prefixes[i+1] = after consuming token i.
    // restore(prefixes[n]) therefore selects n consumed inputs, not n emitted
    // draft tokens. Integration owns the verify-window consumed/pending contract.
    // Empty chunks are allowed (prefixes may contain the pre-chunk checkpoint).
    // Spans must have exact sizes; output may not overlap inputs/owned buffers or
    // checkpoints. Inputs may not alias state/checkpoints, which mutate mid-chunk.
    // Invalid dimensions/sizes/nonfinites/overlap/count overflow throw
    // invalid_argument before mutation. FP32 arithmetic overflow throws
    // runtime_error: the failing token changes neither state nor its output;
    // earlier successfully consumed tokens in that chunk remain committed.
    void run(GdnInput input, std::size_t tokens, std::span<float> output,
             std::span<GdnCheckpoint> prefixes = {});

private:
    void validate_checkpoint(const GdnCheckpoint& checkpoint) const;
    void step_unchecked(GdnInput input, std::span<float> output);
    GdnCheckpoint state_;
    std::vector<float> conv_weights_, dt_bias_, ssm_a_, norm_weight_;
    std::vector<float> qkv_scratch_, output_scratch_, next_recurrent_;
};

}
