#include "sampling.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
std::size_t allocations = 0;

[[gnu::noinline]] void* allocate(std::size_t bytes) {
    ++allocations;
    if (void* p = std::malloc(bytes == 0 ? 1 : bytes)) return p;
    throw std::bad_alloc();
}

[[gnu::noinline]] void* allocate_aligned(std::size_t bytes, std::size_t alignment) {
    ++allocations;
    if (bytes == 0) bytes = 1;
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1))
        throw std::bad_alloc();
    bytes = ((bytes + alignment - 1) / alignment) * alignment;
    if (void* p = std::aligned_alloc(alignment, bytes)) return p;
    throw std::bad_alloc();
}
} // namespace

// Observe all scalar/array/aligned/nothrow C++ allocation forms, including first
// successful calls. Construction, test fixtures and exceptions are outside hot
// intervals. The production code has no direct malloc or threading calls.
[[gnu::noinline]] void* operator new(std::size_t n) { return allocate(n); }
[[gnu::noinline]] void* operator new[](std::size_t n) { return allocate(n); }
[[gnu::noinline]] void* operator new(std::size_t n, std::align_val_t a) {
    return allocate_aligned(n, static_cast<std::size_t>(a));
}
[[gnu::noinline]] void* operator new[](std::size_t n, std::align_val_t a) {
    return allocate_aligned(n, static_cast<std::size_t>(a));
}
[[gnu::noinline]] void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocate(n); } catch (...) { return nullptr; }
}
[[gnu::noinline]] void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocate(n); } catch (...) { return nullptr; }
}
[[gnu::noinline]] void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocate_aligned(n, static_cast<std::size_t>(a)); } catch (...) { return nullptr; }
}
[[gnu::noinline]] void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocate_aligned(n, static_cast<std::size_t>(a)); } catch (...) { return nullptr; }
}
[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace {

std::size_t checks = 0, rejections = 0, hot_calls = 0, hot_allocations = 0;
double max_sum_error = 0.0, max_empirical_error = 0.0;
static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == sizeof(std::uint32_t));
static_assert(!std::is_default_constructible_v<qwen::SamplingConfig>);
static_assert(!std::is_copy_constructible_v<qwen::Sampler>);
static_assert(!std::is_move_constructible_v<qwen::Sampler>);

void check(bool condition, const char* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

template<class Function> decltype(auto) hot(Function function) {
    const auto before = allocations;
    if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
        function();
        ++hot_calls;
        hot_allocations += allocations - before;
        check(allocations == before, "successful hot call allocated");
    } else {
        const auto result = function();
        ++hot_calls;
        hot_allocations += allocations - before;
        check(allocations == before, "successful hot call allocated");
        return result;
    }
}

template<class Function> void rejected(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        ++rejections;
        return;
    }
    throw std::runtime_error("invalid sampling arguments accepted");
}

bool same_bytes(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

long double sum(std::span<const float> values) {
    long double result = 0.0L;
    for (const float value : values) result += value;
    return result;
}

void normalized(std::span<const float> values) {
    for (const float value : values)
        check(std::isfinite(value) && value >= 0.0f && value <= 1.0f, "invalid emitted probability");
    const double error = static_cast<double>(std::abs(sum(values) - 1.0L));
    max_sum_error = std::max(max_sum_error, error);
    check(error <= qwen::sampling_emitted_sum_tolerance, "emitted probability sum outside bound");
}

void near(std::span<const float> actual, std::span<const double> expected, double tolerance = 1.0e-7) {
    check(actual.size() == expected.size(), "fixture shape mismatch");
    normalized(actual);
    for (std::size_t i = 0; i < actual.size(); ++i)
        check(std::abs(actual[i] - expected[i]) <= tolerance, "analytic probability fixture mismatch");
}

qwen::SamplingConfig config(double temperature = 1.0, double top_p = 1.0, int top_k = 0,
                            std::uint64_t seed = 20261003) {
    qwen::SamplingConfig result(seed);
    result.temperature = temperature;
    result.top_p = top_p;
    result.top_k = top_k;
    return result;
}

void allocation_probe_test() {
    const auto before = allocations;
    void* a = ::operator new(4);
    void* b = ::operator new[](16);
    void* c = ::operator new(64, std::align_val_t{64});
    void* d = ::operator new[](128, std::align_val_t{64});
    void* e = ::operator new(4, std::nothrow);
    void* f = ::operator new[](16, std::nothrow);
    void* g = ::operator new(64, std::align_val_t{64}, std::nothrow);
    void* h = ::operator new[](128, std::align_val_t{64}, std::nothrow);
    check(allocations - before == 8, "allocation probe missed new form");
    ::operator delete(a);
    ::operator delete[](b);
    ::operator delete(c, std::align_val_t{64});
    ::operator delete[](d, std::align_val_t{64});
    ::operator delete(e, std::nothrow);
    ::operator delete[](f, std::nothrow);
    ::operator delete(g, std::align_val_t{64}, std::nothrow);
    ::operator delete[](h, std::align_val_t{64}, std::nothrow);
}

void filter_tests() {
    const qwen::SamplingConfig defaults(17);
    check(defaults.temperature == 1.0 && defaults.top_p == 0.95 && defaults.top_k == 20 &&
          defaults.seed == 17, "baseline sampling defaults");
    // Analytic ratios 1:2:4:8, without reimplementing the production algorithm.
    const float log_two = static_cast<float>(std::log(2.0));
    const std::array<float, 4> logits{0.0f, log_two, 2.0f * log_two, 3.0f * log_two};
    std::array<float, 4> out{};
    const auto run = [&](qwen::SamplingConfig cfg, const std::array<double, 4>& expected) {
        qwen::Sampler sampler(4, cfg);
        const auto input_before = logits;
        hot([&] { sampler.distribution(logits, out); });
        near(out, expected);
        check(same_bytes(logits, input_before) && sampler.random_draws() == 0,
              "distribution mutated logits or RNG");
        const auto before = out;
        hot([&] { sampler.distribution(logits, out); });
        check(same_bytes(out, before), "distribution replay is not bitwise deterministic");
    };
    run(config(), {1.0 / 15, 2.0 / 15, 4.0 / 15, 8.0 / 15});
    run(config(0.5), {1.0 / 85, 4.0 / 85, 16.0 / 85, 64.0 / 85});
    const double root_two = std::sqrt(2.0);
    const double total = 3.0 + 3.0 * root_two;
    run(config(2.0), {1.0 / total, root_two / total, 2.0 / total, 2.0 * root_two / total});
    run(config(1.0, 1.0, 2), {0, 0, 1.0 / 3, 2.0 / 3});
    // Top-k renormalization comes BEFORE top-p: .65 keeps just the maximum;
    // top-p before top-k would keep both 8 and 4 from the full softmax.
    run(config(1.0, 0.65, 2), {0, 0, 0, 1});
    run(config(1.0, 0.7, 2), {0, 0, 1.0 / 3, 2.0 / 3});
    run(config(1.0, 1.0, 1), {0, 0, 0, 1});
    run(config(1.0, 1.0, std::numeric_limits<int>::max()), {1.0 / 15, 2.0 / 15, 4.0 / 15, 8.0 / 15});
    run(config(0.0, 0.001, 1), {0, 0, 0, 1});

    constexpr std::array<float, 8> tied{};
    std::array<float, 8> tie_out{};
    for (const auto cfg : {config(1.0, 0.5), config(1.0, std::nextafter(0.5, 0.0)),
                           config(1.0, std::nextafter(0.5, 1.0)), config(1.0, 1.0, 3),
                           config(1.0, 0.5, 4), config(1.0, 0.95, 20),
                           config(1.0, std::numeric_limits<double>::denorm_min())}) {
        qwen::Sampler sampler(8, cfg);
        hot([&] { sampler.distribution(tied, tie_out); });
        const std::size_t retained = cfg.top_p == 0.95 ? 8 : cfg.top_k == 3 ? 3 : cfg.top_k == 4 ? 2
            : cfg.top_p > 0.5 ? 5 : cfg.top_p < 0.1 ? 1 : 4;
        for (std::size_t i = 0; i < 8; ++i)
            check(tie_out[i] == (i < retained ? 1.0f / static_cast<float>(retained) : 0.0f),
                  "top-p boundary/minimum-one/lower-ID tie");
        normalized(tie_out);
    }
    // Non-contiguous maxima and signed zeros exercise ties at a top-k cutoff.
    constexpr std::array<float, 8> scattered{-1, 2, -0.0f, 2, 2, 1, 2, 0};
    qwen::Sampler two(8, config(1.0, 1.0, 2));
    hot([&] { two.distribution(scattered, tie_out); });
    for (std::size_t i = 0; i < 8; ++i)
        check(tie_out[i] == (i == 1 || i == 3 ? 0.5f : 0.0f), "scattered top-k tie convention");
    check(hot([&] { return two.greedy(scattered); }) == 1, "greedy tie convention");
    check(hot([&] { return two.greedy(tied); }) == 0, "greedy equal logits");

    constexpr float maximum = std::numeric_limits<float>::max();
    const std::array<float, 4> extreme{-maximum, maximum, maximum, -maximum};
    for (const double temperature : {std::numeric_limits<double>::denorm_min(), 1.0,
                                     std::numeric_limits<double>::max(), 0.0}) {
        qwen::Sampler sampler(4, config(temperature));
        hot([&] { sampler.distribution(extreme, out); });
        if (temperature == 0.0) near(out, std::array<double, 4>{0, 1, 0, 0});
        else if (temperature > 1.0) near(out, std::array<double, 4>{0.25, 0.25, 0.25, 0.25});
        else near(out, std::array<double, 4>{0, 0.5, 0.5, 0});
    }
    // At top-p=1, a positive FP32-representable tail cannot disappear simply
    // because adding it to the leading probability rounds to 1 in FP64.
    qwen::Sampler tail(4, config());
    const std::array<float, 4> tiny_tail{0, -90, -100, -1000};
    hot([&] { tail.distribution(tiny_tail, out); });
    normalized(out);
    check(out[0] == 1.0f && out[1] > 0.0f && out[2] > 0.0f && out[3] == 0.0f,
          "top-p=1 lost representable tail or failed extreme underflow");
    for (const auto cfg : {config(), config(0.0), defaults}) {
        qwen::Sampler single(1, cfg);
        const std::array<float, 1> logit{-maximum};
        std::array<float, 1> probability{};
        hot([&] { single.distribution(logit, probability); });
        check(probability[0] == 1 && hot([&] { return single.draw(probability); }) == 0,
              "singleton vocabulary");
    }
}

void large_workspace_tests() {
    // Vocabulary-sized workspace qualification without any model/weight access.
    constexpr std::size_t vocabulary = 248320;
    std::vector<float> logits(vocabulary, -1.0f), out(vocabulary);
    for (std::size_t i = 0; i < vocabulary; ++i)
        logits[i] = static_cast<float>((37 * i) % 257) / 32.0f;
    for (const int top_k : {0, 1, 20, 257}) {
        qwen::Sampler sampler(vocabulary, config(1.0, 0.95, top_k));
        hot([&] { sampler.distribution(logits, out); });
        normalized(out);
        check(sampler.random_draws() == 0, "large distribution consumed RNG");
        for (int iteration = 0; iteration < 32; ++iteration) {
            const auto token = hot([&] { return sampler.draw(out); });
            check(out[token] > 0.0f, "categorical draw selected zero support");
        }
        hot([&] { sampler.distribution(logits, out); });
    }
    // Exact tie selection at a large cutoff, and measured uniform sum error.
    std::fill(logits.begin(), logits.end(), -std::numeric_limits<float>::max());
    qwen::Sampler all(vocabulary, config());
    hot([&] { all.distribution(logits, out); });
    normalized(out);
    for (const float probability : out)
        check(probability == out[0] && probability > 0, "large uniform logits");
    qwen::Sampler twenty(vocabulary, qwen::SamplingConfig(1));
    hot([&] { twenty.distribution(logits, out); });
    for (std::size_t i = 0; i < vocabulary; ++i)
        check(out[i] == (i < 19 ? 1.0f / 19.0f : 0.0f), "default top20 then top-p .95 boundary");
    normalized(out);
}

void primitive_tests() {
    constexpr std::array<float, 4> p{0.5f, 0.25f, 0.125f, 0.125f};
    constexpr std::array<float, 4> q{0.25f, 0.5f, 0.25f, 0};
    std::array<float, 4> residual{-1, -2, -3, -4};
    for (std::size_t i = 0; i < 3; ++i) {
        constexpr std::array<double, 3> expected{1, 0.5, 0.5};
        check(hot([&] { return qwen::acceptance_probability(p, q, i); }) == expected[i],
              "analytic acceptance ratio");
    }
    check(hot([&] { return qwen::residual_distribution(p, q, residual); }) == qwen::ResidualStatus::ready,
          "overlapping supports should have a residual");
    near(residual, std::array<double, 4>{2.0 / 3, 0, 0, 1.0 / 3});

    const auto before = residual;
    check(hot([&] { return qwen::residual_distribution(p, p, residual); }) == qwen::ResidualStatus::no_residual,
          "identical inputs must have no residual");
    check(same_bytes(residual, before), "no-residual modified caller output");
    const auto distinct_p = p;
    check(hot([&] { return qwen::residual_distribution(p, distinct_p, residual); }) ==
          qwen::ResidualStatus::no_residual, "equal separate storage must have no residual");
    for (std::size_t i = 0; i < 4; ++i)
        check(hot([&] { return qwen::acceptance_probability(p, p, i); }) == 1.0,
              "equal distributions do not accept exactly");

    constexpr std::array<float, 4> left{0.5f, 0.5f, 0, -0.0f};
    constexpr std::array<float, 4> right{0, 0, 0.25f, 0.75f};
    check(hot([&] { return qwen::residual_distribution(left, right, residual); }) == qwen::ResidualStatus::ready,
          "disjoint supports must have a residual");
    check(same_bytes(residual, std::array<float, 4>{0.5f, 0.5f, 0, 0}), "disjoint residual != target");
    check(hot([&] { return qwen::acceptance_probability(left, right, 3); }) == 0.0,
          "disjoint acceptance must be zero");

    // Smallest FP32 positive residual must not be mistaken for no-residual.
    constexpr float denorm = std::numeric_limits<float>::denorm_min();
    constexpr std::array<float, 4> tiny_p{1, denorm, 0, 0}, tiny_q{1, 0, denorm, 0};
    check(hot([&] { return qwen::residual_distribution(tiny_p, tiny_q, residual); }) == qwen::ResidualStatus::ready,
          "subnormal positive residual discarded");
    near(residual, std::array<double, 4>{0, 1, 0, 0});

    // Equal distributions under different allowed normalization roundoff are
    // canonicalized separately; raw p/q or raw max(p-q,0) would be incorrect.
    constexpr std::array<float, 4> scaled_p{0.25000011920928955078125f, 0.25000011920928955078125f,
                                         0.25000011920928955078125f, 0.25000011920928955078125f};
    constexpr std::array<float, 4> scaled_q{0.25f, 0.25f, 0.25f, 0.25f};
    check(hot([&] { return qwen::residual_distribution(scaled_p, scaled_q, residual); }) ==
          qwen::ResidualStatus::no_residual, "roundoff-scaled identical distributions");
    check(hot([&] { return qwen::acceptance_probability(scaled_p, scaled_q, 1); }) == 1,
          "acceptance did not canonicalize allowed roundoff");

    // Unequal sums near the admitted bound; independent long-double identity
    // verifies accepted + rejected/replacement mass for every token.
    auto rounded_p = p;
    auto rounded_q = q;
    rounded_p[0] += 4.0f * std::numeric_limits<float>::epsilon();
    rounded_q[1] -= 4.0f * std::numeric_limits<float>::epsilon();
    check(hot([&] { return qwen::residual_distribution(rounded_p, rounded_q, residual); }) ==
          qwen::ResidualStatus::ready, "allowed FP32 roundoff rejected");
    normalized(residual);
    const long double p_sum = sum(rounded_p), q_sum = sum(rounded_q);
    long double accepted_mass = 0;
    std::array<long double, 4> accepted{};
    for (std::size_t i = 0; i < 4; ++i) {
        accepted[i] = std::min(rounded_p[i] / p_sum, rounded_q[i] / q_sum);
        accepted_mass += accepted[i];
        if (rounded_q[i] > 0) {
            const double a = hot([&] { return qwen::acceptance_probability(rounded_p, rounded_q, i); });
            check(std::abs(a - accepted[i] / (rounded_q[i] / q_sum)) < 1e-14L,
                  "acceptance roundoff normalization differs from independent oracle");
        }
    }
    for (std::size_t i = 0; i < 4; ++i)
        check(std::abs(accepted[i] + (1 - accepted_mass) * residual[i] - rounded_p[i] / p_sum) < 1e-7L,
              "accepted plus residual mass does not equal target");
}

void seed_tests() {
    constexpr std::array<float, 5> probabilities{0, 0.125f, 0, 0.375f, 0.5f};
    constexpr std::uint64_t seed = 0xfedcba9876543210ULL;
    qwen::Sampler first(5, config(1, 1, 0, seed)), second(5, config(1, 1, 0, seed));
    std::mt19937_64 reference(seed);
    std::array<std::size_t, 1024> replay{};
    std::array<float, 5> residual{}, distribution{};
    constexpr std::array<float, 5> logits{1, 2, 3, 4, 5};
    for (std::size_t i = 0; i < replay.size(); ++i) {
        const long double u = static_cast<long double>(reference() >> 11) / 9007199254740992.0L;
        const auto expected = u < 0.125L ? std::size_t{1} : u < 0.5L ? std::size_t{3} : std::size_t{4};
        replay[i] = hot([&] { return first.draw(probabilities); });
        check(replay[i] == expected, "seeded categorical differs from independent quantile oracle");
        check(hot([&] { return second.draw(probabilities); }) == replay[i], "seed replay mismatch");
        hot([&] { first.distribution(logits, distribution); });
        hot([&] { return first.greedy(logits); });
        hot([&] { return qwen::residual_distribution(probabilities, probabilities, residual); });
    }
    check(first.random_draws() == replay.size(), "math-only calls consumed RNG");
    hot([&] { first.reset_seed(seed); });
    check(first.random_draws() == 0, "seed reset did not reset counter");
    for (const auto expected : replay)
        check(hot([&] { return first.draw(probabilities); }) == expected, "reset seed did not replay sequence");
    hot([&] { first.reset_seed(seed + 1); });
    std::size_t differences = 0;
    for (const auto expected : replay)
        differences += hot([&] { return first.draw(probabilities); }) != expected;
    check(differences > 0, "different seed did not affect stream");

    qwen::Sampler acceptor(5, config(1, 1, 0, seed));
    std::mt19937_64 accept_reference(seed);
    constexpr std::array<float, 5> target{0, 0.0625f, 0, 0.4375f, 0.5f};
    for (int i = 0; i < 1024; ++i) {
        const bool expected = static_cast<double>(accept_reference() >> 11) * 0x1.0p-53 < 0.5;
        check(hot([&] { return acceptor.accept(target, probabilities, 1); }) == expected,
              "acceptance stream differs from explicit uniform mapping");
    }
    const auto count = acceptor.random_draws();
    check(hot([&] { return acceptor.accept(probabilities, probabilities, 1); }), "acceptance one rejected");
    constexpr std::array<float, 5> zero_target{1, 0, 0, 0, 0};
    check(!hot([&] { return acceptor.accept(zero_target, probabilities, 1); }), "acceptance zero accepted");
    check(acceptor.random_draws() == count + 2, "certain acceptance/rejection did not consume one uniform");
}

void invalid_tests() {
    for (const auto n : {std::size_t{0}, static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1,
                         std::numeric_limits<std::size_t>::max()})
        rejected([&] { qwen::Sampler sampler(n, config()); });
    for (const double value : {-1.0, -std::numeric_limits<double>::denorm_min(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        rejected([&] { qwen::Sampler sampler(4, config(value)); });
    for (const double value : {0.0, -0.0, -1.0, std::nextafter(1.0, 2.0),
                               std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        rejected([&] { qwen::Sampler sampler(4, config(1, value)); });
    for (const int value : {-1, std::numeric_limits<int>::min()})
        rejected([&] { qwen::Sampler sampler(4, config(1, 1, value)); });

    qwen::Sampler sampler(4, config()), twin(4, config());
    constexpr std::array<float, 4> logits{0, 1, 2, 3}, p{0.5f, 0.25f, 0.125f, 0.125f};
    constexpr std::array<float, 4> q{0.25f, 0.5f, 0.25f, 0};
    // Warm both RNGs equally; rejection must preserve the actual subsequent
    // stream, not merely its public counter. Preserve arbitrary output bit values.
    check(sampler.draw(p) == twin.draw(p), "initial twin stream mismatch");
    std::array<float, 4> out{std::bit_cast<float>(0x7fc12345U), -0.0f, -7, 42};
    const auto unchanged = out;
    const auto fail = [&](auto function) {
        const auto before = sampler.random_draws();
        rejected(function);
        check(same_bytes(out, unchanged), "invalid call modified caller output bytes");
        check(sampler.random_draws() == before, "invalid call advanced RNG counter");
    };
    fail([&] { sampler.distribution({}, out); });
    fail([&] { sampler.distribution(std::span(logits).first(3), out); });
    fail([&] { sampler.distribution(logits, std::span(out).first(3)); });
    fail([&] { sampler.draw({}); });
    fail([&] { sampler.greedy({}); });
    fail([&] { sampler.accept(p, {}, 0); });
    fail([&] { qwen::acceptance_probability(p, {}, 0); });
    fail([&] { qwen::residual_distribution(p, {}, out); });
    fail([&] { qwen::residual_distribution(p, q, std::span(out).first(3)); });
    fail([&] { qwen::residual_distribution({}, {}, {}); });
    for (const std::size_t token : {std::size_t{4}, std::numeric_limits<std::size_t>::max()}) {
        fail([&] { sampler.accept(p, q, token); });
        fail([&] { qwen::acceptance_probability(p, q, token); });
    }
    fail([&] { sampler.accept(p, q, 3); }); // q[token] == 0, even if p[token] > 0
    fail([&] { qwen::acceptance_probability(p, q, 3); });

    for (std::size_t position = 0; position < 4; ++position) {
        for (const std::uint32_t bits : {0x7fc00001U, 0x7f800001U, 0xffc12345U, 0x7f800000U, 0xff800000U}) {
            auto bad = logits;
            bad[position] = std::bit_cast<float>(bits);
            fail([&] { sampler.distribution(bad, out); });
            fail([&] { sampler.greedy(bad); });
            qwen::Sampler zero(4, config(0.0));
            fail([&] { zero.distribution(bad, out); });
        }
        for (const float bad_value : {-std::numeric_limits<float>::denorm_min(), -0.5f,
                                       std::nextafter(1.0f, 2.0f), std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            auto bad = p;
            bad[position] = bad_value;
            fail([&] { sampler.draw(bad); });
            fail([&] { sampler.accept(bad, q, 0); });
            fail([&] { sampler.accept(p, bad, 0); });
            fail([&] { qwen::acceptance_probability(bad, q, 0); });
            fail([&] { qwen::acceptance_probability(p, bad, 0); });
            fail([&] { qwen::residual_distribution(bad, q, out); });
            fail([&] { qwen::residual_distribution(p, bad, out); });
        }
    }
    for (const auto bad : {std::array<float, 4>{0, 0, 0, 0}, std::array<float, 4>{1, 1, 1, 1},
                           std::array<float, 4>{0.49f, 0.25f, 0.125f, 0.125f},
                           std::array<float, 4>{0.51f, 0.25f, 0.125f, 0.125f},
                           std::array<float, 4>{0.500002f, 0.25f, 0.125f, 0.125f}}) {
        fail([&] { sampler.draw(bad); });
        fail([&] { sampler.accept(bad, q, 0); });
        fail([&] { sampler.accept(p, bad, 0); });
        fail([&] { qwen::residual_distribution(bad, q, out); });
        fail([&] { qwen::residual_distribution(p, bad, out); });
    }
    // The exact documented sum boundary is admitted; one FP32 ULP outside is not.
    auto boundary = p;
    boundary[0] += static_cast<float>(qwen::sampling_probability_sum_tolerance);
    hot([&] { return qwen::acceptance_probability(boundary, q, 0); });
    boundary[0] = std::nextafter(boundary[0], 1.0f);
    fail([&] { sampler.draw(boundary); });

    // Null/wrapping ranges and owner aliases reject without dereferencing them.
    const std::span<const float> null_input(static_cast<const float*>(nullptr), 4);
    const std::span<float> null_output(static_cast<float*>(nullptr), 4);
    constexpr auto address = std::numeric_limits<std::uintptr_t>::max() -
                             std::numeric_limits<std::uintptr_t>::max() % alignof(float);
    const std::span<const float> wrapping(reinterpret_cast<const float*>(address), 4);
    for (const auto bad : {null_input, wrapping}) {
        fail([&] { sampler.distribution(bad, out); });
        fail([&] { sampler.draw(bad); });
        fail([&] { sampler.greedy(bad); });
        fail([&] { sampler.accept(bad, q, 0); });
        fail([&] { qwen::residual_distribution(bad, q, out); });
    }
    fail([&] { sampler.distribution(logits, null_output); });
    fail([&] { qwen::residual_distribution(p, q, null_output); });
    const auto* owner_data = reinterpret_cast<const float*>(&sampler);
    const std::span<const float> owner_input(owner_data, 4);
    fail([&] { sampler.distribution(owner_input, out); });
    fail([&] { sampler.draw(owner_input); });
    fail([&] { sampler.greedy(owner_input); });
    fail([&] { sampler.accept(owner_input, q, 0); });
    fail([&] { sampler.distribution(logits, {const_cast<float*>(owner_data), 4}); });

    // Both directions of partial overlap, exact overlap and touching but
    // non-overlapping ranges. Aliases are intentionally valid live float arrays.
    std::array<float, 12> storage{0.5f, 0.25f, 0.125f, 0.125f, 0.25f, 0.5f, 0.25f, 0};
    const auto storage_before = storage;
    const auto a = std::span(storage).first(4), b = std::span(storage).subspan(4, 4);
    for (const std::size_t offset : {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
        const auto overlapping = std::span(storage).subspan(offset, 4);
        fail([&] { sampler.distribution(a, overlapping); });
        fail([&] { sampler.distribution(overlapping, a); });
        fail([&] { qwen::residual_distribution(a, q, overlapping); });
        fail([&] { qwen::residual_distribution(p, a, overlapping); });
        check(same_bytes(storage, storage_before), "overlap rejection modified input/output storage");
    }
    hot([&] { sampler.distribution(a, b); }); // endpoints touch, do not overlap
    normalized(b);
    // Invalid calls must leave the entire RNG stream unchanged, including calls
    // involving fake owner spans. Also establish recovery after all rejections.
    for (int i = 0; i < 256; ++i)
        check(hot([&] { return sampler.draw(p); }) == hot([&] { return twin.draw(p); }),
              "invalid call mutated RNG state despite unchanged counter");
    hot([&] { sampler.distribution(logits, out); });
    normalized(out);
}

// Fixed BEFORE running any empirical checks. Two-sided Bernstein bounds for
// independent Bernoulli indicators, union-bounded over <=100 coordinates/rates,
// family-wise alpha=1e-9. An additional 4*FP32 epsilon covers measured rounding
// of residual output (not statistical error); no tolerance is fitted to a seed.
constexpr std::size_t empirical_trials = 200000;
constexpr double empirical_alpha = 1e-9;
constexpr double empirical_comparisons = 100;

void empirical_check(std::size_t count, long double expected) {
    const double probability = static_cast<double>(expected);
    const double t = std::log(2.0 * empirical_comparisons / empirical_alpha);
    const double bound = std::sqrt(2.0 * probability * (1.0 - probability) * t / empirical_trials) +
                         2.0 * t / (3.0 * empirical_trials) +
                         4.0 * std::numeric_limits<float>::epsilon();
    const double error = std::abs(static_cast<double>(count) / empirical_trials - probability);
    max_empirical_error = std::max(max_empirical_error, error);
    check(error <= bound, "empirical distribution outside predeclared Bernstein bound");
    if (expected == 0.0L) check(count == 0, "empirical output invented zero-probability support");
    if (expected == 1.0L) check(count == empirical_trials, "empirical deterministic output not exact");
}

void speculative_empirical_tests() {
    using Probabilities = std::array<float, 5>;
    struct Fixture { Probabilities p, q; };
    constexpr Probabilities target{0.5f, 0.25f, 0.125f, 0.125f, 0};
    constexpr Probabilities skew{1 - 4.0f / 65536, 1.0f / 65536, 1.0f / 65536, 1.0f / 65536, 1.0f / 65536};
    constexpr std::array<Fixture, 6> fixtures{{
        {target, target},                                  // p=q, no replacement possible
        {target, {0, 0, 0, 0, 1}},                         // disjoint supports
        {target, {0.125f, 0.5f, 0.25f, 0.0625f, 0.0625f}}, // overlapping, target-only and q-only mass
        {target, skew},                                   // very skewed q
        {skew, {0, 0.25f, 0.25f, 0.25f, 0.25f}},           // very skewed p, mostly replacements
        {skew, {0.25f, 0.25f, 0.25f, 0.125f, 0.125f}}     // skewed target with overlap
    }};
    for (std::size_t f = 0; f < fixtures.size(); ++f) {
        const auto& p = fixtures[f].p;
        const auto& q = fixtures[f].q;
        const auto p_before = p, q_before = q;
        qwen::Sampler sampler(5, config(1, 1, 0, 1009 + f * 9176));
        Probabilities residual{-1, -2, -3, -4, -5};
        const auto sentinel = residual;
        const auto status = hot([&] { return qwen::residual_distribution(p, q, residual); });
        const long double p_sum = sum(p), q_sum = sum(q);
        long double acceptance = 0;
        std::array<long double, 5> common{};
        for (std::size_t i = 0; i < p.size(); ++i) {
            common[i] = std::min(p[i] / p_sum, q[i] / q_sum);
            acceptance += common[i];
        }
        if (status == qwen::ResidualStatus::ready) {
            normalized(residual);
            for (std::size_t i = 0; i < p.size(); ++i)
                check(std::abs(common[i] + (1 - acceptance) * residual[i] - p[i] / p_sum) <= 1e-7L,
                      "independent analytic speculative-mixture identity failed");
        } else {
            check(acceptance == 1 && same_bytes(residual, sentinel), "no-residual condition not exact/atomic");
        }
        std::array<std::size_t, 5> proposal_counts{}, output_counts{}, replacement_counts{};
        std::size_t accepted = 0, replacements = 0;
        for (std::size_t trial = 0; trial < empirical_trials; ++trial) {
            const auto proposal = hot([&] { return sampler.draw(q); });
            ++proposal_counts[proposal];
            if (hot([&] { return sampler.accept(p, q, proposal); })) {
                ++accepted;
                ++output_counts[proposal];
            } else {
                check(status == qwen::ResidualStatus::ready, "rejection requested impossible replacement");
                const auto replacement = hot([&] { return sampler.draw(residual); });
                ++replacements;
                ++replacement_counts[replacement];
                ++output_counts[replacement];
            }
        }
        check(accepted + replacements == empirical_trials &&
              sampler.random_draws() == 2 * empirical_trials + replacements, "speculative draw accounting");
        empirical_check(accepted, acceptance);
        for (std::size_t i = 0; i < p.size(); ++i) {
            empirical_check(proposal_counts[i], q[i] / q_sum);
            empirical_check(output_counts[i], p[i] / p_sum);
            empirical_check(replacement_counts[i], p[i] / p_sum - common[i]);
        }
        check(same_bytes(p, p_before) && same_bytes(q, q_before), "stochastic math mutated inputs");
        std::cout << "{\"case\":\"speculative\",\"fixture\":" << f << ",\"trials\":" << empirical_trials
                  << ",\"accepted\":" << accepted << ",\"replacements\":" << replacements << "}\n";
    }
}

} // namespace

void sparse_zero_bits_test() {
    const std::array<float,6> p{0.0f,-0.0f,0.0f,-0.0f,0.75f,0.25f};
    const std::array<float,6> q{0.0f,0.0f,-0.0f,-0.0f,0.25f,0.75f};
    std::array<float,6> out{};
    check(qwen::residual_distribution(p,q,out)==qwen::ResidualStatus::ready,"signed-zero residual status");
    const std::array<std::uint32_t,6> expected{0,0x80000000U,0,0,0x3f800000U,0};
    for(std::size_t i=0;i<out.size();++i)
        check(std::bit_cast<std::uint32_t>(out[i])==expected[i],"signed-zero residual bits");
    const auto tiny=std::numeric_limits<float>::denorm_min();
    const std::array<float,3> a{1.0f,tiny,0.0f},b{1.0f,0.0f,tiny};
    std::array<float,3> residual{};
    check(qwen::residual_distribution(a,b,residual)==qwen::ResidualStatus::ready,"subnormal residual retained");
    check(residual[1]==1.0f && residual[0]==0.0f && residual[2]==0.0f,"no epsilon cutoff in sparse residual");
}

int main() {
    try {
        allocation_probe_test();
        filter_tests();
        large_workspace_tests();
        primitive_tests();
        sparse_zero_bits_test();
        seed_tests();
        invalid_tests();
        speculative_empirical_tests();
        std::cout << "{\"test\":\"sampling\",\"checks\":" << checks << ",\"rejections\":" << rejections
                  << ",\"hot_calls\":" << hot_calls << ",\"hot_allocations\":" << hot_allocations
                  << ",\"max_sum_error\":" << max_sum_error << ",\"max_empirical_error\":" << max_empirical_error
                  << ",\"empirical_family_alpha\":" << empirical_alpha << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sampling test: " << error.what() << '\n';
        return 1;
    }
}
