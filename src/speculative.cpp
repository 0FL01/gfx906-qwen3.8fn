#include "speculative.hpp"

#include <stdexcept>

namespace qwen {
namespace {

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(message);
}

// Flat address ranges on the supported Linux hosts, as in sampling.cpp.
bool overlaps(const void* a_data, std::size_t a_bytes,
              const void* b_data, std::size_t b_bytes) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(a_data);
    const auto b = reinterpret_cast<std::uintptr_t>(b_data);
    return a < b ? b - a < a_bytes : a - b < b_bytes;
}

} // namespace

SpeculativeSampler::SpeculativeSampler(std::size_t vocabulary, SamplingConfig config)
    : sampler_(vocabulary, config), residual_(sampler_.vocabulary()) {}

void SpeculativeSampler::validate_range(std::span<const float> probabilities) const {
    if (probabilities.size() != vocabulary())
        invalid("speculative: expected constructor vocabulary shape");
    const auto start = reinterpret_cast<std::uintptr_t>(probabilities.data());
    if (probabilities.data() == nullptr || start % alignof(float) != 0 ||
        probabilities.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        probabilities.size_bytes() > std::numeric_limits<std::uintptr_t>::max() - start)
        invalid("speculative: null, unaligned or wrapping range");
    if (overlaps(probabilities.data(), probabilities.size_bytes(), this, sizeof(*this)) ||
        overlaps(probabilities.data(), probabilities.size_bytes(),
                 residual_.data(), residual_.capacity() * sizeof(float)))
        invalid("speculative: caller span overlaps owner storage");
}

void SpeculativeSampler::preflight(const SpeculativeWindow& window) {
    if (window.horizon > 2) invalid("speculative: horizon must be 0, 1 or 2");
    // Range/owner checks precede probability access and scratch writes.
    for (std::size_t i = 0; i < window.p.size(); ++i) {
        if (i <= window.horizon) validate_range(window.p[i]);
        else if (!window.p[i].empty()) invalid("speculative: unused target slot must be empty");
    }
    for (std::size_t i = 0; i < window.q.size(); ++i) {
        if (i < window.horizon) {
            validate_range(window.q[i]);
            if (window.draft_ids[i] >= vocabulary())
                invalid("speculative: draft token outside vocabulary");
        } else if (!window.q[i].empty() || window.draft_ids[i] != speculative_empty_token) {
            invalid("speculative: unused proposal slot must be empty");
        }
    }
    // greedy is a pure, no-RNG Sampler operation. It additionally rejects aliases
    // of Sampler's private heap storage before accessing each span. Probability
    // semantics are validated by the shared primitives below, not by greedy.
    for (std::size_t i = 0; i <= window.horizon; ++i) (void)sampler_.greedy(window.p[i]);
    for (std::size_t i = 0; i < window.horizon; ++i) (void)sampler_.greedy(window.q[i]);

    std::array<double, 2> acceptance{};
    for (std::size_t i = 0; i < window.horizon; ++i)
        acceptance[i] = acceptance_probability(window.p[i], window.q[i], window.draft_ids[i]);
    // Identical inputs validate the bonus (including horizon=0) without changing
    // workspace. Never substitute an arbitrary positive-mass token as its ID.
    (void)residual_distribution(window.p[window.horizon], window.p[window.horizon], residual_);
    // Precompute/check every potential replacement before any randomness. Only
    // one residual buffer is retained; on rejection that row is recomputed.
    for (std::size_t i = 0; i < window.horizon; ++i) {
        const auto status = residual_distribution(window.p[i], window.q[i], residual_);
        if (status == ResidualStatus::no_residual && acceptance[i] < 1.0)
            throw std::runtime_error("speculative: possible rejection has no residual");
    }
}

SpeculativeDecision SpeculativeSampler::decide(const SpeculativeWindow& window) {
    preflight(window);
    SpeculativeDecision result;
    for (std::size_t i = 0; i < window.horizon; ++i) {
        // Includes one uniform for acceptance probability 0 and 1.
        if (!sampler_.accept(window.p[i], window.q[i], window.draft_ids[i])) {
            if (residual_distribution(window.p[i], window.q[i], residual_) != ResidualStatus::ready)
                throw std::runtime_error("speculative: rejected proposal has no residual");
            result.pending_carry = sampler_.draw(residual_);
            result.emitted_ids[result.emitted_count++] = result.pending_carry;
            return result;
        }
        ++result.accepted_drafts;
        ++result.restore_prefix;
        result.emitted_ids[result.emitted_count++] = window.draft_ids[i];
    }
    result.pending_carry = sampler_.draw(window.p[window.horizon]);
    result.emitted_ids[result.emitted_count++] = result.pending_carry;
    return result;
}

SpeculativeTerminalDecision SpeculativeSampler::decide(const SpeculativeWindow& window,
                                                       const SpeculativeStopConfig& stop) {
    if (stop.remaining_output_tokens == 0)
        invalid("speculative: remaining output budget must be positive");
    if (stop.eos_id >= vocabulary()) invalid("speculative: EOS token outside vocabulary");
    if (window.horizon > 2 || window.horizon > stop.remaining_output_tokens - 1)
        invalid("speculative: horizon exceeds reserved-bonus output budget");
    preflight(window);
    SpeculativeTerminalDecision result;
    const auto emit = [&](std::size_t token) {
        result.emitted_ids[result.emitted_count++] = token;
        result.pending_carry = token;
        if (!stop.ignore_eos && token == stop.eos_id)
            result.stop_reason = SpeculativeStopReason::eos;
        else if (result.emitted_count == stop.remaining_output_tokens)
            result.stop_reason = SpeculativeStopReason::output_budget;
    };
    for (std::size_t i = 0; i < window.horizon; ++i) {
        if (!sampler_.accept(window.p[i], window.q[i], window.draft_ids[i])) {
            if (residual_distribution(window.p[i], window.q[i], residual_) != ResidualStatus::ready)
                throw std::runtime_error("speculative: rejected proposal has no residual");
            emit(sampler_.draw(residual_));
            return result;
        }
        ++result.accepted_drafts;
        emit(window.draft_ids[i]);
        if (result.stop_reason == SpeculativeStopReason::eos) return result;
        ++result.forwarded_accepted_drafts;
        ++result.restore_prefix;
    }
    emit(sampler_.draw(window.p[window.horizon]));
    return result;
}

} // namespace qwen
