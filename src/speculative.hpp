#pragma once

#include "sampling.hpp"

#include <array>

namespace qwen {

inline constexpr std::size_t speculative_empty_token = std::numeric_limits<std::size_t>::max();

// Full-window inputs, already filtered by the existing Sampler.
// For target verify input [old_pending, a, b], p[0] predicts a, p[1] predicts b,
// and p[2] predicts the bonus. q[i] is the actual distribution that proposed
// draft_ids[i], not a reconstruction from the target logits.
struct SpeculativeWindow {
    std::size_t horizon = 0; // Exactly 0, 1 or 2.
    std::array<std::span<const float>, 3> p{};
    std::array<std::span<const float>, 2> q{};
    std::array<std::size_t, 2> draft_ids{speculative_empty_token, speculative_empty_token};
    // p[0..horizon] and q[0..horizon-1] must have constructor vocabulary size.
    // All unused spans must be empty; unused IDs must be speculative_empty_token.
};

struct SpeculativeDecision {
    std::size_t accepted_drafts = 0;
    // Number of target verify inputs actually consumed: old_pending + accepted
    // drafts. A surrounding runtime must restore/retain this prefix itself.
    std::size_t restore_prefix = 1;
    std::array<std::size_t, 3> emitted_ids{
        speculative_empty_token, speculative_empty_token, speculative_empty_token};
    std::size_t emitted_count = 0; // accepted_drafts + one replacement/bonus.
    std::size_t pending_carry = speculative_empty_token; // Last emitted ID.
    // old_pending was already emitted and is never inserted in emitted_ids.
};

struct SpeculativeStopConfig {
    std::size_t remaining_output_tokens = 0; // Required >=1; includes emitted EOS.
    std::size_t eos_id = speculative_empty_token; // Required in vocabulary, even if ignored.
    bool ignore_eos = false;
};

enum class SpeculativeStopReason { window_complete, eos, output_budget };

struct SpeculativeTerminalDecision {
    std::size_t accepted_drafts = 0; // Includes an accepted final EOS.
    std::size_t forwarded_accepted_drafts = 0; // Excludes that pending EOS.
    // old_pending + accepted drafts preceding the last emitted (pending) token.
    // The future target tap row is restore_prefix-1, even for accepted EOS.
    std::size_t restore_prefix = 1;
    std::array<std::size_t, 3> emitted_ids{
        speculative_empty_token, speculative_empty_token, speculative_empty_token};
    std::size_t emitted_count = 0;
    std::size_t pending_carry = speculative_empty_token; // Always last emitted, including EOS.
    SpeculativeStopReason stop_reason = SpeculativeStopReason::window_complete;
};

// Mathematical R6 prerequisite only: no trained MTP, model state, checkpoints or
// GPU rollback. Decisions report the required target prefix; the caller restores
// it. The original overload is strictly nonterminal. The stop-config overload
// stops on token IDs/budget, never on text, and reserves one replacement/bonus:
// horizon <= min(2, remaining_output_tokens-1). It preflights even rows behind EOS.
// An accepted EOS remains pending and is not part of the retained target inputs.
// Therefore restore_prefix == emitted_count for the terminal-aware overload:
// consumed_before=P+prior_outputs-1 -> consumed_after=P+new_outputs-1. The last
// emitted token is never committed, including when the output budget is exhausted.
// One owner, non-concurrent. All storage is allocated at construction. Successful
// calls allocate nothing/create no threads. Uses the existing explicit seed and
// baseline filter defaults; statistical equivalence is to supplied target p,
// not to the non-speculative baseline's random stream.
//
// Every used distribution/ID (even behind a first rejection) is preflighted before
// RNG mutation. Spans must designate live accessible float objects. Null,
// unaligned, wrapping, wrong-shaped and owner/workspace aliases reject before
// reading the offending range. Read-only input overlap, including p=q, is valid.
// Invalid arguments leave caller storage and the RNG stream unchanged.
class SpeculativeSampler {
public:
    SpeculativeSampler(std::size_t vocabulary, SamplingConfig config);
    SpeculativeSampler(const SpeculativeSampler&) = delete;
    SpeculativeSampler& operator=(const SpeculativeSampler&) = delete;
    SpeculativeSampler(SpeculativeSampler&&) = delete;
    SpeculativeSampler& operator=(SpeculativeSampler&&) = delete;

    SpeculativeDecision decide(const SpeculativeWindow& window);
    // ignore_eos=true produces exactly the original overload's output/RNG stream
    // when the budget fits. EOS takes precedence when it also exhausts the budget.
    // Each attempted accept consumes one uniform, even at probability 0/1. An
    // accepted EOS stops without a replacement/bonus draw or any further accept.
    SpeculativeTerminalDecision decide(const SpeculativeWindow& window,
                                       const SpeculativeStopConfig& stop);
    void reset_seed(std::uint64_t seed) noexcept { sampler_.reset_seed(seed); }
    std::uint64_t random_draws() const noexcept { return sampler_.random_draws(); }
    std::size_t vocabulary() const noexcept { return sampler_.vocabulary(); }

private:
    void validate_range(std::span<const float> probabilities) const;
    void preflight(const SpeculativeWindow& window);

    Sampler sampler_;
    std::vector<float> residual_;
};

} // namespace qwen
