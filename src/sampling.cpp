#include "sampling.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qwen {
namespace {

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(message);
}

// Neumaier summation: in particular preserve small residual/tail mass. FP64 is
// used throughout; FP32 rounding happens only after the final normalization.
struct Sum {
    double main = 0.0;
    double correction = 0.0;
    void add(double value) noexcept {
        const double next = main + value;
        correction += std::abs(main) >= std::abs(value) ? (main - next) + value
                                                       : (value - next) + main;
        main = next;
    }
    double value() const noexcept { return main + correction; }
};

void validate_range(std::span<const float> values) {
    if (values.empty() || values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        values.size() > std::numeric_limits<std::size_t>::max() / sizeof(float))
        invalid("sampling: vocabulary must be 1..INT_MAX");
    const auto start = reinterpret_cast<std::uintptr_t>(values.data());
    if (values.data() == nullptr || start % alignof(float) != 0 ||
        values.size_bytes() > std::numeric_limits<std::uintptr_t>::max() - start)
        invalid("sampling: null, unaligned or wrapping range");
}

// Flat address ranges on supported Linux hosts; never compare unrelated pointers.
bool overlaps(const void* a_data, std::size_t a_bytes,
              const void* b_data, std::size_t b_bytes) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(a_data);
    const auto b = reinterpret_cast<std::uintptr_t>(b_data);
    return a < b ? b - a < a_bytes : a - b < b_bytes;
}

void validate_pair(std::span<const float> p, std::span<const float> q) {
    if (p.size() != q.size()) invalid("sampling: distribution shapes differ");
    validate_range(p);
    validate_range(q);
}

double probability_sum(std::span<const float> probabilities) {
    Sum sum;
    for (const float probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f)
            invalid("sampling: probability must be finite and in [0,1]");
        sum.add(static_cast<double>(probability));
    }
    const double total = sum.value();
    if (std::abs(total - 1.0) > sampling_probability_sum_tolerance)
        invalid("sampling: probability sum is not normalized");
    return total;
}

double positive_difference(float p, float q, double p_sum, double q_sum) noexcept {
    return std::max(static_cast<double>(p) / p_sum - static_cast<double>(q) / q_sum, 0.0);
}

void check_emitted_sum(double sum) {
    // An internal arithmetic failure, not a relaxed input-validation rule. The
    // rounded candidate has not yet been written to caller storage.
    if (!std::isfinite(sum) || std::abs(sum - 1.0) > sampling_emitted_sum_tolerance)
        throw std::runtime_error("sampling: FP32 normalization exceeded its error bound");
}

} // namespace

double acceptance_probability(std::span<const float> p, std::span<const float> q,
                              std::size_t token) {
    validate_pair(p, q);
    if (token >= p.size()) invalid("sampling: proposal token outside vocabulary");
    const double p_sum = probability_sum(p);
    const double q_sum = probability_sum(q);
    if (!(q[token] > 0.0f)) invalid("sampling: sampled proposal token must have positive mass");
    return std::min(1.0, (static_cast<double>(p[token]) / p_sum) /
                         (static_cast<double>(q[token]) / q_sum));
}

ResidualStatus residual_distribution(std::span<const float> p, std::span<const float> q,
                                     std::span<float> probabilities) {
    validate_pair(p, q);
    if (probabilities.size() != p.size()) invalid("sampling: residual output shape differs");
    validate_range(probabilities);
    if (overlaps(p.data(), p.size_bytes(), probabilities.data(), probabilities.size_bytes()) ||
        overlaps(q.data(), q.size_bytes(), probabilities.data(), probabilities.size_bytes()))
        invalid("sampling: residual output overlaps input");
    const double p_sum = probability_sum(p);
    const double q_sum = probability_sum(q);
    Sum mass;
    for (std::size_t i = 0; i < p.size(); ++i)
        mass.add(positive_difference(p[i], q[i], p_sum, q_sum));
    const double total = mass.value();
    if (total == 0.0) return ResidualStatus::no_residual;

    Sum emitted;
    for (std::size_t i = 0; i < p.size(); ++i)
        emitted.add(static_cast<float>(positive_difference(p[i], q[i], p_sum, q_sum) / total));
    check_emitted_sum(emitted.value());
    for (std::size_t i = 0; i < p.size(); ++i)
        probabilities[i] = static_cast<float>(positive_difference(p[i], q[i], p_sum, q_sum) / total);
    return ResidualStatus::ready;
}

std::size_t Sampler::checked_vocabulary(std::size_t vocabulary, const SamplingConfig& config) {
    if (vocabulary == 0 || vocabulary > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        vocabulary > std::numeric_limits<std::size_t>::max() / sizeof(std::size_t) ||
        vocabulary > std::numeric_limits<std::size_t>::max() / sizeof(double))
        invalid("sampling: vocabulary must be 1..INT_MAX");
    if (!std::isfinite(config.temperature) || config.temperature < 0.0 ||
        !std::isfinite(config.top_p) || config.top_p <= 0.0 || config.top_p > 1.0 || config.top_k < 0)
        invalid("sampling: invalid filter configuration");
    return vocabulary;
}

Sampler::Sampler(std::size_t vocabulary, SamplingConfig config)
    : config_(config), ids_(checked_vocabulary(vocabulary, config)), weights_(ids_.size()),
      rng_(config.seed) {}

void Sampler::validate_span(std::span<const float> values) const {
    if (values.size() != vocabulary()) invalid("sampling: expected constructor vocabulary shape");
    validate_range(values);
    if (overlaps(values.data(), values.size_bytes(), this, sizeof(*this)) ||
        overlaps(values.data(), values.size_bytes(), ids_.data(), ids_.capacity() * sizeof(std::size_t)) ||
        overlaps(values.data(), values.size_bytes(), weights_.data(), weights_.capacity() * sizeof(double)))
        invalid("sampling: caller span overlaps sampler storage");
}

std::size_t Sampler::validate_logits(std::span<const float> logits) const {
    validate_span(logits);
    std::size_t best = 0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) invalid("sampling: all logits must be finite");
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

std::size_t Sampler::greedy(std::span<const float> logits) const {
    return validate_logits(logits);
}

void Sampler::distribution(std::span<const float> logits, std::span<float> probabilities) {
    // All validation precedes scratch/output mutation. In particular check alias
    // before reading either span, so fake owner/output aliases cannot be read.
    validate_span(logits);
    validate_span(probabilities);
    if (overlaps(logits.data(), logits.size_bytes(), probabilities.data(), probabilities.size_bytes()))
        invalid("sampling: output overlaps logits");
    const auto best = validate_logits(logits);
    if (config_.temperature == 0.0) {
        std::fill(probabilities.begin(), probabilities.end(), 0.0f);
        probabilities[best] = 1.0f;
        return;
    }

    for (std::size_t i = 0; i < vocabulary(); ++i) ids_[i] = i;
    const auto count = config_.top_k == 0 ? vocabulary()
        : std::min(vocabulary(), static_cast<std::size_t>(config_.top_k));
    const auto order = [&](std::size_t a, std::size_t b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    };
    std::partial_sort(ids_.begin(), ids_.begin() + static_cast<std::ptrdiff_t>(count), ids_.end(), order);
    Sum mass;
    const double maximum = logits[best];
    for (std::size_t i = 0; i < count; ++i) {
        // Subtract in double before division: finite FP32 extremes cannot cause
        // inf-inf. Negative overflow at tiny temperature correctly yields exp=0.
        weights_[i] = std::exp((static_cast<double>(logits[ids_[i]]) - maximum) / config_.temperature);
        mass.add(weights_[i]);
    }
    std::size_t kept = count;
    if (config_.top_p < 1.0) {
        Sum prefix;
        const double threshold = config_.top_p * mass.value();
        for (std::size_t i = 0; i < count; ++i) {
            prefix.add(weights_[i]);
            if (prefix.value() >= threshold) {
                kept = i + 1;
                break;
            }
        }
    }
    Sum retained;
    for (std::size_t i = 0; i < kept; ++i) retained.add(weights_[i]);
    Sum emitted;
    for (std::size_t i = 0; i < kept; ++i) {
        weights_[i] = static_cast<float>(weights_[i] / retained.value());
        emitted.add(weights_[i]);
    }
    check_emitted_sum(emitted.value());
    std::fill(probabilities.begin(), probabilities.end(), 0.0f);
    for (std::size_t i = 0; i < kept; ++i)
        probabilities[ids_[i]] = static_cast<float>(weights_[i]);
}

double Sampler::uniform() noexcept {
    ++random_draws_;
    return static_cast<double>(rng_() >> 11) * 0x1.0p-53;
}

std::size_t Sampler::draw(std::span<const float> probabilities) {
    validate_span(probabilities);
    const double total = probability_sum(probabilities);
    const double threshold = uniform() * total;
    Sum prefix;
    std::size_t last_positive = 0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (probabilities[i] == 0.0f) continue;
        last_positive = i;
        prefix.add(probabilities[i]);
        if (threshold < prefix.value()) return i;
    }
    // Endpoint rounding can put threshold at total, never on a zero-mass token.
    return last_positive;
}

bool Sampler::accept(std::span<const float> p, std::span<const float> q, std::size_t token) {
    validate_span(p);
    validate_span(q);
    const double probability = acceptance_probability(p, q, token);
    return uniform() < probability;
}

void Sampler::reset_seed(std::uint64_t seed) noexcept {
    rng_.seed(seed);
    random_draws_ = 0;
}

} // namespace qwen
