#pragma once

#include "quant.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace qwen {

struct HcConfig {
    std::size_t hidden_size = 2560;
    std::size_t branches = 4;
    std::size_t low_rank = 320;
    float rms_epsilon = 1.0e-6f;

    // Positive dimensions, branches > 1 (HF invariant), finite positive epsilon,
    // checked sizes and dimensions representable by the QMatrix int ABI.
    [[nodiscard]] std::size_t widened_elements() const;
};

struct HcParameters {
    QMatrix down; // GGUF [branches * hidden_size, low_rank]
    QMatrix up;   // GGUF [low_rank, branches * hidden_size]
    std::span<const float> norm_weight; // [branch][feature], direct GGUF gamma
    std::optional<QMatrix> inject = std::nullopt; // GGUF [widened, branches]
};

// Original scalar FP32 numerical oracle, not a donor runtime or GPU hot path.
// Borrows matrix byte spans and norm_weight; their backing storage must outlive
// this instance and remain immutable, including during calls. Descriptors are
// copied. Owns scratch allocated once at construction for capacity tokens.
// One owner: calls on the same instance must not run concurrently.
//
// Layouts are token-major: widened[token][branch][feature], block[token][feature],
// injection_weights[token][branch]. No branch is omitted or averaged in the
// residual. Initial embeddings repeat to every branch via expand().
//
// All spans must be exactly sized, 1 <= tokens <= capacity. Writable spans may
// overlap neither any readable span nor weights/norm nor another writable span;
// read-only overlap is allowed. Existing output contents need not be finite.
// Bad geometry, sizes, aliases and nonfinite arguments throw invalid_argument;
// FP32 arithmetic overflow throws runtime_error. Both leave ALL caller outputs
// unchanged, including when a later token fails. There is no retained gate or
// residual state: combine() consumes explicit caller-owned pre-mix residual and
// gates. Scratch is private, disposable and reused after either success/failure.
// No allocations or threads on successful calls. Compile -ffp-contract=off,
// without fast-math; FP32 ascending-index sums, stable sigmoid and SiLU.
class HcCpu {
public:
    HcCpu(HcConfig config, HcParameters parameters, std::size_t capacity = 1);
    HcCpu(const HcCpu&) = delete;
    HcCpu& operator=(const HcCpu&) = delete;

    [[nodiscard]] const HcConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool has_injection() const noexcept { return parameters_.inject.has_value(); }

    void expand(std::span<const float> embedding, std::size_t tokens,
                std::span<float> widened);

    // Attn/FFN pre: return the hidden_size block input and already-activated
    // 2*sigmoid injection weights. With inject absent this is the final head
    // collapse; injection_weights must be empty. No separate final output norm.
    // The caller's widened residual BEFORE this head call is the MTP Tap;
    // neither block_input nor normalized scratch is the Tap (mx t_h_nextn).
    void mix(std::span<const float> residual, std::size_t tokens,
             std::span<float> block_input, std::span<float> injection_weights = {});

    // Attn/FFN post: output[t,c,j] = residual[t,c,j] +
    // injection_weights[t,c] * block_output[t,j]. Gates must be finite in [0,2],
    // already activated as returned by mix(), not raw projection logits.
    // A head-only instance rejects combine().
    void combine(std::span<const float> residual, std::span<const float> block_output,
                 std::span<const float> injection_weights, std::size_t tokens,
                 std::span<float> widened);

private:
    void validate_tokens(std::size_t tokens) const;
    void validate_output(std::span<float> output) const;
    HcConfig config_;
    HcParameters parameters_;
    std::size_t capacity_;
    std::vector<float> normalized_, low_, gates_, mixed_, injection_;
};

} // namespace qwen
