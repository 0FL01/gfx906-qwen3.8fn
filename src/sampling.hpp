#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace qwen {

struct SamplingConfig {
    // A seed is mandatory, including when using the baseline defaults.
    explicit SamplingConfig(std::uint64_t seed_value) noexcept : seed(seed_value) {}
    double temperature = 1.0; // Finite, >=0; zero selects the separate greedy path.
    double top_p = 0.95;      // Finite, in (0,1].
    int top_k = 20;           // >=0; zero disables, otherwise capped at vocabulary.
    std::uint64_t seed;
};

// FP32 probabilities must individually be finite and in [0,1], with an FP64
// compensated sum within this absolute distance of 1. Negative values are never
// clamped (signed zero is valid). Accepted roundoff is renormalized in FP64 by
// draw/acceptance/residual, so all three operate on the same distributions.
inline constexpr double sampling_probability_sum_tolerance =
    8.0 * std::numeric_limits<float>::epsilon();
// distribution/residual measure the FP64 sum of their rounded FP32 result before
// publishing it and require this bound. No correction that invents support.
inline constexpr double sampling_emitted_sum_tolerance =
    2.0 * std::numeric_limits<float>::epsilon();

enum class ResidualStatus { ready, no_residual };

// Independent stochastic speculative math, not an MTP state/runtime interface.
// p is target, q is proposal; token must have been sampled from q, and q[token]
// must be >0. Return min(1, normalized_p[token]/normalized_q[token]).
double acceptance_probability(std::span<const float> p, std::span<const float> q,
                              std::size_t token);

// Normalize max(normalized_p-normalized_q,0). Exact p=q has no residual and leaves
// output unchanged. The caller must honor no_residual: rejection is impossible,
// so requesting a replacement in that case is a caller error. Arbitrarily small
// positive residuals are retained; there is no epsilon-based equality shortcut.
ResidualStatus residual_distribution(std::span<const float> p, std::span<const float> q,
                                     std::span<float> probabilities);

// One owner, non-concurrent; all workspace is allocated once in the constructor.
// Vocabulary is 1..INT_MAX. Success calls allocate nothing/create no threads.
// All float spans must be exactly vocabulary-sized (free primitives: equal sizes
// in the same range) and designate live accessible float objects. Null/wrapping
// ranges, invalid values/config/shapes and any writable/input overlap throw
// invalid_argument before caller-output or RNG mutation. Read-only p/q overlap,
// including exact identity, is valid. Sampler calls also reject owner overlap.
class Sampler {
public:
    Sampler(std::size_t vocabulary, SamplingConfig config);
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) = delete;
    Sampler& operator=(Sampler&&) = delete;

    // Temperature -> top-k -> top-p -> final normalization. Top-p uses the
    // top-k-renormalized mass, retaining the smallest descending prefix reaching
    // top_p, minimum one. At top_p=1 retain all positive mass, even tiny tails.
    // Ranking uses original logits (positive temperature preserves order), with
    // lower token ID winning ties; no min-p/repetition penalties. All logits must
    // be finite, even those filtered out. temperature=0 emits greedy one-hot.
    // Defaults at temperature=1 match the project's baseline filter settings,
    // not its RNG stream. This operation never consumes randomness.
    void distribution(std::span<const float> logits, std::span<float> probabilities);
    // Separate deterministic lower-ID argmax, independent of filtering/config.
    std::size_t greedy(std::span<const float> logits) const;
    // Validated categorical draw from the normalized FP32 input. Ascending-ID
    // CDF; uses mt19937_64's high 53 bits for an explicit uniform in [0,1).
    std::size_t draw(std::span<const float> probabilities);
    // One uniform is consumed on every success, including acceptance 0 and 1.
    bool accept(std::span<const float> p, std::span<const float> q, std::size_t token);
    void reset_seed(std::uint64_t seed) noexcept;
    std::uint64_t random_draws() const noexcept { return random_draws_; }
    std::size_t vocabulary() const noexcept { return ids_.size(); }

private:
    static std::size_t checked_vocabulary(std::size_t vocabulary, const SamplingConfig& config);
    void validate_span(std::span<const float> values) const;
    std::size_t validate_logits(std::span<const float> logits) const;
    double uniform() noexcept;

    SamplingConfig config_;
    std::vector<std::size_t> ids_;
    std::vector<double> weights_;
    std::mt19937_64 rng_;
    std::uint64_t random_draws_ = 0;
};

} // namespace qwen
