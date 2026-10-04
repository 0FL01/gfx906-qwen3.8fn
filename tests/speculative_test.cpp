#include "speculative.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace {
std::size_t allocations = 0;
bool capture_allocations = false;
struct Allocation { void* data = nullptr; std::size_t bytes = 0; };
std::array<Allocation, 8> captured{};
std::size_t captured_count = 0;

void record(void* data, std::size_t bytes) noexcept {
    ++allocations;
    if (capture_allocations && captured_count < captured.size())
        captured[captured_count++] = {data, bytes};
}

[[gnu::noinline]] void* allocate(std::size_t bytes) {
    if (void* p = std::malloc(bytes == 0 ? 1 : bytes)) { record(p, bytes); return p; }
    throw std::bad_alloc();
}

[[gnu::noinline]] void* allocate_aligned(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) bytes = 1;
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc();
    bytes = ((bytes + alignment - 1) / alignment) * alignment;
    if (void* p = std::aligned_alloc(alignment, bytes)) { record(p, bytes); return p; }
    throw std::bad_alloc();
}
} // namespace

// All scalar/array/aligned/nothrow C++ allocation forms. Exceptions/construction
// are outside successful-call intervals. No production malloc/thread calls.
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
std::size_t checks = 0, rejections = 0, hot_calls = 0, hot_allocations = 0, windows = 0;
double max_empirical_error = 0.0, max_terminal_empirical_error = 0.0;
using Probabilities = std::array<float, 5>;
constexpr std::uint64_t guard = 0xdeadbeefabcdef09ULL;

static_assert(!std::is_copy_constructible_v<qwen::SpeculativeSampler>);
static_assert(!std::is_move_constructible_v<qwen::SpeculativeSampler>);
static_assert(std::is_trivially_copyable_v<qwen::SpeculativeDecision>);
static_assert(std::is_trivially_copyable_v<qwen::SpeculativeTerminalDecision>);

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template<class Function> decltype(auto) hot(Function function) {
    const auto before = allocations;
    if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
        function();
        ++hot_calls;
        hot_allocations += allocations - before;
        check(allocations == before, "successful call allocated");
    } else {
        const auto result = function();
        ++hot_calls;
        hot_allocations += allocations - before;
        check(allocations == before, "successful call allocated");
        return result;
    }
}

template<class Function> void rejected(Function function) {
    try { function(); }
    catch (const std::invalid_argument&) { ++rejections; return; }
    throw std::runtime_error("invalid window accepted");
}

template<std::size_t N> struct GuardedProbabilities {
    std::array<std::uint64_t, 8> before{guard, guard, guard, guard, guard, guard, guard, guard};
    std::array<float, N> values{};
    std::array<std::uint64_t, 8> after{guard, guard, guard, guard, guard, guard, guard, guard};
    void verify() const {
        for (const auto value : before) check(value == guard, "input leading canary changed");
        for (const auto value : after) check(value == guard, "input trailing canary changed");
    }
};

struct GuardedOwner {
    std::array<std::uint64_t, 8> before{guard, guard, guard, guard, guard, guard, guard, guard};
    qwen::SpeculativeSampler sampler;
    std::array<std::uint64_t, 8> after{guard, guard, guard, guard, guard, guard, guard, guard};
    explicit GuardedOwner(std::uint64_t seed) : sampler(5, qwen::SamplingConfig(seed)) {}
    void verify() const {
        for (const auto value : before) check(value == guard, "owner leading canary changed");
        for (const auto value : after) check(value == guard, "owner trailing canary changed");
    }
};

qwen::SpeculativeDecision decide(qwen::SpeculativeSampler& sampler, const qwen::SpeculativeWindow& input) {
    const auto before = sampler.random_draws();
    const auto result = hot([&] { return sampler.decide(input); });
    ++windows;
    check(result.accepted_drafts <= input.horizon, "acceptance outside horizon");
    check(result.restore_prefix == 1 + result.accepted_drafts, "wrong consumed target prefix");
    check(result.emitted_count == 1 + result.accepted_drafts, "wrong output count");
    check(result.pending_carry == result.emitted_ids[result.emitted_count - 1], "wrong pending carry");
    for (std::size_t i = 0; i < result.accepted_drafts; ++i)
        check(result.emitted_ids[i] == input.draft_ids[i], "accepted prefix changed draft ID");
    for (std::size_t i = result.emitted_count; i < result.emitted_ids.size(); ++i)
        check(result.emitted_ids[i] == qwen::speculative_empty_token, "unused result slot not empty");
    const auto attempts = result.accepted_drafts + (result.accepted_drafts < input.horizon ? 1 : 0);
    check(sampler.random_draws() == before + attempts + 1, "accept/terminal draw RNG accounting");
    if (result.accepted_drafts == input.horizon) {
        check(input.p[input.horizon][result.pending_carry] > 0, "bonus outside target support");
    } else {
        const auto row = result.accepted_drafts;
        long double p_sum = 0, q_sum = 0;
        for (const auto value : input.p[row]) p_sum += value;
        for (const auto value : input.q[row]) q_sum += value;
        check(input.p[row][result.pending_carry] / p_sum > input.q[row][result.pending_carry] / q_sum,
              "replacement outside positive normalized residual support");
    }
    return result;
}

bool same(const qwen::SpeculativeDecision& a, const qwen::SpeculativeDecision& b) {
    return a.accepted_drafts == b.accepted_drafts && a.restore_prefix == b.restore_prefix &&
           a.emitted_ids == b.emitted_ids && a.emitted_count == b.emitted_count && a.pending_carry == b.pending_carry;
}

qwen::SpeculativeTerminalDecision decide(qwen::SpeculativeSampler& sampler,
                                        const qwen::SpeculativeWindow& input,
                                        const qwen::SpeculativeStopConfig& stop) {
    const auto before = sampler.random_draws();
    const auto result = hot([&] { return sampler.decide(input, stop); });
    ++windows;
    const bool accepted_eos = result.accepted_drafts > 0 && !stop.ignore_eos &&
                             input.draft_ids[result.accepted_drafts - 1] == stop.eos_id;
    check(result.accepted_drafts <= input.horizon, "terminal acceptance outside horizon");
    check(result.forwarded_accepted_drafts == result.accepted_drafts - (accepted_eos ? 1 : 0),
          "pending accepted EOS was forwarded");
    check(result.restore_prefix == 1 + result.forwarded_accepted_drafts, "terminal target prefix");
    check(result.emitted_count == result.restore_prefix, "consumed/produced pending invariant");
    check(result.emitted_count >= 1 && result.emitted_count <= stop.remaining_output_tokens,
          "terminal output budget exceeded");
    check(result.pending_carry == result.emitted_ids[result.emitted_count - 1], "terminal pending carry");
    for (std::size_t i = 0; i < result.accepted_drafts; ++i)
        check(result.emitted_ids[i] == input.draft_ids[i], "terminal accepted draft source ID");
    for (std::size_t i = 0; i < result.emitted_count; ++i) {
        check(result.emitted_ids[i] < sampler.vocabulary(), "terminal ID outside vocabulary");
        if (!stop.ignore_eos && result.emitted_ids[i] == stop.eos_id)
            check(i + 1 == result.emitted_count, "emitted after EOS");
    }
    for (std::size_t i = result.emitted_count; i < result.emitted_ids.size(); ++i)
        check(result.emitted_ids[i] == qwen::speculative_empty_token, "terminal unused result slot");
    const auto expected_reason = !stop.ignore_eos && result.pending_carry == stop.eos_id
        ? qwen::SpeculativeStopReason::eos : result.emitted_count == stop.remaining_output_tokens
        ? qwen::SpeculativeStopReason::output_budget : qwen::SpeculativeStopReason::window_complete;
    check(result.stop_reason == expected_reason, "terminal stop reason/precedence");
    if (accepted_eos) {
        check(result.emitted_count == result.accepted_drafts, "accepted EOS drew an extra token");
        check(sampler.random_draws() == before + result.accepted_drafts, "extra RNG after accepted EOS");
    } else {
        check(result.emitted_count == 1 + result.accepted_drafts, "terminal replacement/bonus count");
        const auto attempts = result.accepted_drafts + (result.accepted_drafts < input.horizon ? 1 : 0);
        check(sampler.random_draws() == before + attempts + 1, "terminal accept/draw RNG accounting");
        if (result.accepted_drafts == input.horizon) {
            check(input.p[input.horizon][result.pending_carry] > 0, "terminal bonus outside support");
        } else {
            const auto row = result.accepted_drafts;
            long double p_sum = 0, q_sum = 0;
            for (const auto value : input.p[row]) p_sum += value;
            for (const auto value : input.q[row]) q_sum += value;
            check(input.p[row][result.pending_carry] / p_sum > input.q[row][result.pending_carry] / q_sum,
                  "terminal replacement outside positive normalized residual support");
        }
    }
    // A pure decision report models the absolute target token count; no model
    // state is touched. All emitted tokens except the final pending are consumed.
    constexpr std::size_t prompt = 17, prior_outputs = 13;
    const auto consumed_before = prompt + prior_outputs - 1;
    check(consumed_before + result.restore_prefix == prompt + prior_outputs + result.emitted_count - 1,
          "absolute target state count invariant");
    return result;
}

bool same(const qwen::SpeculativeTerminalDecision& a, const qwen::SpeculativeTerminalDecision& b) {
    return a.accepted_drafts == b.accepted_drafts && a.restore_prefix == b.restore_prefix &&
           a.emitted_ids == b.emitted_ids && a.emitted_count == b.emitted_count &&
           a.pending_carry == b.pending_carry && a.forwarded_accepted_drafts == b.forwarded_accepted_drafts &&
           a.stop_reason == b.stop_reason;
}

bool same_outputs(const qwen::SpeculativeDecision& a, const qwen::SpeculativeTerminalDecision& b) {
    return a.accepted_drafts == b.accepted_drafts && a.restore_prefix == b.restore_prefix &&
           a.emitted_ids == b.emitted_ids && a.emitted_count == b.emitted_count && a.pending_carry == b.pending_carry;
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
    check(allocations - before == 8, "allocation probe missed a new form");
    ::operator delete(a); ::operator delete[](b);
    ::operator delete(c, std::align_val_t{64}); ::operator delete[](d, std::align_val_t{64});
    ::operator delete(e, std::nothrow); ::operator delete[](f, std::nothrow);
    ::operator delete(g, std::align_val_t{64}, std::nothrow);
    ::operator delete[](h, std::align_val_t{64}, std::nothrow);
}

void deterministic_tests() {
    const qwen::SamplingConfig defaults(19);
    check(defaults.seed == 19 && defaults.temperature == 1 && defaults.top_p == 0.95 && defaults.top_k == 20,
          "existing explicit-seed/default sampling contract changed");
    GuardedOwner owner(19);
    std::array<GuardedProbabilities<5>, 4> probs{};
    for (std::size_t i = 0; i < probs.size(); ++i) probs[i].values[i] = 1;
    const auto saved = probs;
    const auto run = [&](std::size_t horizon, std::span<const float> p0, std::span<const float> p1,
                         std::span<const float> p2, std::size_t accepted, std::size_t last) {
        qwen::SpeculativeWindow input;
        input.horizon = horizon;
        input.p = {p0, p1, p2};
        if (horizon > 0) { input.q[0] = probs[1].values; input.draft_ids[0] = 1; }
        if (horizon > 1) { input.q[1] = probs[2].values; input.draft_ids[1] = 2; }
        const auto result = decide(owner.sampler, input);
        check(result.accepted_drafts == accepted && result.pending_carry == last, "deterministic window result");
        // Conceptual old_pending=4; it is already emitted, never an output here.
        for (std::size_t i = 0; i < result.emitted_count; ++i)
            check(result.emitted_ids[i] != 4, "old pending was repeated");
        for (std::size_t i = 0; i < probs.size(); ++i) {
            probs[i].verify();
            check(probs[i].values == saved[i].values, "caller probabilities mutated");
        }
        owner.verify();
    };
    run(2, probs[1].values, probs[2].values, probs[3].values, 2, 3);
    run(2, probs[0].values, probs[2].values, probs[3].values, 0, 0);
    run(2, probs[1].values, probs[0].values, probs[3].values, 1, 0);
    run(1, probs[1].values, probs[2].values, {}, 1, 2);
    run(1, probs[0].values, probs[2].values, {}, 0, 0);
    run(0, probs[3].values, {}, {}, 0, 3);

    // Exact equality, also separate storage, cannot reject and needs no residual.
    constexpr Probabilities equal{0.5f, 0.25f, 0.125f, 0.125f, -0.0f};
    const auto separate = equal;
    qwen::SpeculativeWindow exact;
    exact.horizon = 2;
    exact.p = {equal, separate, probs[3].values};
    exact.q = {equal, equal};
    exact.draft_ids = {0, 1};
    for (int i = 0; i < 128; ++i) check(decide(owner.sampler, exact).accepted_drafts == 2, "p=q rejected");

    constexpr auto tiny = std::numeric_limits<float>::denorm_min();
    constexpr Probabilities small_p{1, tiny, 0, 0, 0}, small_q{1, 0, tiny, 0, 0};
    qwen::SpeculativeWindow small;
    small.horizon = 1;
    small.p = {small_p, equal, {}};
    small.q[0] = small_q;
    small.draft_ids[0] = 2;
    const auto result = decide(owner.sampler, small);
    check(result.accepted_drafts == 0 && result.pending_carry == 1, "smallest FP32 residual discarded");

    // Unequal allowed FP32 sums are independently normalized in FP64. These
    // proportional rows have exactly the same distribution and no residual.
    constexpr Probabilities scaled_p{0.25000011920928955078125f, 0.25000011920928955078125f,
                                     0.25000011920928955078125f, 0.25000011920928955078125f, 0};
    constexpr Probabilities scaled_q{0.25f, 0.25f, 0.25f, 0.25f, 0};
    exact.p = {scaled_p, scaled_q, probs[3].values};
    exact.q = {scaled_q, scaled_p};
    check(decide(owner.sampler, exact).accepted_drafts == 2, "FP64 normalization of proportional rows failed");

    qwen::SpeculativeSampler singleton(1, defaults);
    constexpr std::array<float, 1> one{1};
    for (std::size_t h = 0; h <= 2; ++h) {
        qwen::SpeculativeWindow input;
        input.horizon = h;
        for (std::size_t i = 0; i <= h; ++i) input.p[i] = one;
        for (std::size_t i = 0; i < h; ++i) { input.q[i] = one; input.draft_ids[i] = 0; }
        check(decide(singleton, input).accepted_drafts == h, "singleton window failed");
    }
}

void seed_tests() {
    constexpr std::uint64_t seed = 0x123456789abcdef0ULL;
    qwen::SpeculativeSampler sampler(5, qwen::SamplingConfig(seed));
    std::mt19937_64 reference(seed);
    constexpr Probabilities p{0.25f, 0.25f, 0.5f, 0, 0}, q{0.5f, 0.125f, 0.375f, 0, 0};
    constexpr Probabilities bonus{0, 0.25f, 0, 0.75f, 0};
    qwen::SpeculativeWindow input;
    input.horizon = 2;
    input.p = {p, p, bonus};
    input.q = {q, q};
    input.draft_ids = {0, 0};
    std::array<qwen::SpeculativeDecision, 1024> replay{};
    const auto uniform = [&] { return static_cast<double>(reference() >> 11) * 0x1.0p-53; };
    for (auto& saved : replay) {
        const bool first = uniform() < 0.5;
        const bool second = first && uniform() < 0.5;
        const auto accepted = first ? second ? std::size_t{2} : std::size_t{1} : std::size_t{0};
        const auto u = uniform();
        // Residual is [0,.5,.5,0,0]; bonus is [0,.25,0,.75,0].
        const auto last = accepted == 2 ? u < 0.25 ? std::size_t{1} : std::size_t{3}
                                        : u < 0.5 ? std::size_t{1} : std::size_t{2};
        saved = decide(sampler, input);
        check(saved.accepted_drafts == accepted && saved.pending_carry == last,
              "window differs from explicit independent uniform/quantile oracle");
    }
    hot([&] { sampler.reset_seed(seed); });
    check(sampler.random_draws() == 0, "reset failed to clear RNG count");
    for (const auto& saved : replay) check(same(saved, decide(sampler, input)), "window seed replay differs");
}

void terminal_deterministic_tests() {
    struct Fixture {
        std::size_t horizon;
        std::array<std::size_t, 3> target_ids;
        std::size_t accepted;
        std::array<std::size_t, 3> outputs;
        std::size_t count;
    };
    // Fixed exact decisions before any EOS truncation, with both rejection rows,
    // each possible bonus row, and both accepted-draft EOS positions.
    constexpr std::array<Fixture, 6> fixtures{{
        {0, {3, 0, 0}, 0, {3, 0, 0}, 1},
        {1, {0, 2, 0}, 0, {0, 0, 0}, 1},
        {1, {1, 2, 0}, 1, {1, 2, 0}, 2},
        {2, {0, 2, 3}, 0, {0, 0, 0}, 1},
        {2, {1, 0, 3}, 1, {1, 0, 0}, 2},
        {2, {1, 2, 3}, 2, {1, 2, 3}, 3}
    }};
    std::array<GuardedProbabilities<5>, 4> rows{};
    for (std::size_t i = 0; i < rows.size(); ++i) rows[i].values[i] = 1;
    const auto saved = rows;
    constexpr Probabilities probe{0.125f, 0.25f, 0.5f, 0.125f, 0};
    constexpr std::uint64_t seed = 0xabc123def456ULL;
    std::size_t cases = 0;
    for (const auto& fixture : fixtures) {
        qwen::SpeculativeWindow input;
        input.horizon = fixture.horizon;
        for (std::size_t i = 0; i <= input.horizon; ++i) input.p[i] = rows[fixture.target_ids[i]].values;
        for (std::size_t i = 0; i < input.horizon; ++i) { input.q[i] = rows[i + 1].values; input.draft_ids[i] = i + 1; }
        for (const auto budget : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4},
                                  std::numeric_limits<std::size_t>::max()}) {
            if (input.horizon > budget - 1) continue;
            for (std::size_t eos = 0; eos < 5; ++eos) {
                for (const bool ignore : {false, true}) {
                    GuardedOwner owner(seed), original(seed);
                    const qwen::SpeculativeStopConfig stop{budget, eos, ignore};
                    const auto result = decide(owner.sampler, input, stop);
                    const auto full = decide(original.sampler, input);
                    std::size_t count = fixture.count;
                    bool has_eos = false;
                    for (std::size_t i = 0; i < fixture.count; ++i) {
                        if (!ignore && fixture.outputs[i] == eos) { count = i + 1; has_eos = true; break; }
                    }
                    const auto accepted = std::min(fixture.accepted, count);
                    check(result.accepted_drafts == accepted && result.emitted_count == count,
                          "fixed terminal accept/output counts");
                    for (std::size_t i = 0; i < count; ++i)
                        check(result.emitted_ids[i] == fixture.outputs[i], "fixed terminal emitted IDs");
                    const auto expected_draws = has_eos && count <= fixture.accepted ? count
                        : fixture.accepted + (fixture.accepted < fixture.horizon ? 1 : 0) + 1;
                    check(owner.sampler.random_draws() == expected_draws, "fixed terminal RNG count");
                    if (ignore || !has_eos) {
                        check(same_outputs(full, result), "ignored/absent EOS changed original decision");
                        check(owner.sampler.random_draws() == original.sampler.random_draws(),
                              "ignored/absent EOS changed original RNG count");
                    }
                    // Check the next actual uniform against an independent engine,
                    // detecting even a clipped implementation with a false counter.
                    std::mt19937_64 reference(seed);
                    reference.discard(expected_draws);
                    const auto u = static_cast<double>(reference() >> 11) * 0x1.0p-53;
                    const auto expected = u < 0.125 ? std::size_t{0} : u < 0.375 ? std::size_t{1}
                        : u < 0.875 ? std::size_t{2} : std::size_t{3};
                    qwen::SpeculativeWindow next; next.p[0] = probe;
                    const auto next_result = decide(owner.sampler, next, {1, 4, false});
                    check(next_result.pending_carry == expected, "terminal next uniform stream");
                    if (ignore || !has_eos)
                        check(same_outputs(decide(original.sampler, next), next_result),
                              "ignored/absent EOS changed subsequent original RNG stream");
                    owner.verify(); original.verify();
                    ++cases;
                }
            }
        }
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        rows[i].verify(); check(rows[i].values == saved[i].values, "terminal caller distribution mutated");
    }
    constexpr float tiny = std::numeric_limits<float>::denorm_min();
    constexpr Probabilities small_p{1, tiny, 0, 0, 0}, small_q{1, 0, tiny, 0, 0};
    GuardedOwner owner(seed);
    qwen::SpeculativeWindow small;
    small.horizon = 1; small.p = {small_p, rows[3].values, {}};
    small.q[0] = small_q; small.draft_ids[0] = 2;
    auto result = decide(owner.sampler, small, {2, 1, false});
    check(result.pending_carry == 1 && result.accepted_drafts == 0 && result.restore_prefix == 1 &&
          result.stop_reason == qwen::SpeculativeStopReason::eos, "minimum FP32 EOS residual dropped");
    small.horizon = 2; small.p = {rows[3].values, small_p, rows[0].values};
    small.q = {rows[3].values, small_q}; small.draft_ids = {3, 2};
    result = decide(owner.sampler, small, {3, 1, false});
    check(result.emitted_ids[0] == 3 && result.pending_carry == 1 && result.accepted_drafts == 1 &&
          result.restore_prefix == 2, "second minimum FP32 EOS residual dropped");
    owner.verify();

    qwen::SpeculativeSampler singleton(1, qwen::SamplingConfig(seed));
    constexpr std::array<float, 1> one{1};
    for (std::size_t horizon = 0; horizon <= 2; ++horizon) {
        qwen::SpeculativeWindow input; input.horizon = horizon;
        for (std::size_t i = 0; i <= horizon; ++i) input.p[i] = one;
        for (std::size_t i = 0; i < horizon; ++i) { input.q[i] = one; input.draft_ids[i] = 0; }
        const auto stopped = decide(singleton, input, {horizon + 1, 0, false});
        check(stopped.emitted_count == 1 && stopped.restore_prefix == 1 &&
              stopped.accepted_drafts == (horizon > 0 ? 1 : 0), "singleton accepted EOS forwarding");
        const auto ignored = decide(singleton, input, {horizon + 1, 0, true});
        check(ignored.accepted_drafts == horizon && ignored.emitted_count == horizon + 1 &&
              ignored.stop_reason == qwen::SpeculativeStopReason::output_budget, "singleton ignored EOS budget");
    }
    std::cout << "{\"case\":\"terminal_deterministic\",\"table_cases\":" << cases << ",\"passed\":true}\n";
}

void terminal_seed_tests() {
    constexpr std::uint64_t seed = 0x8877665544332211ULL;
    GuardedOwner owner(seed), original(seed);
    constexpr Probabilities p{0.25f, 0.25f, 0.5f, 0, 0}, q{0.5f, 0.125f, 0.375f, 0, 0};
    constexpr Probabilities bonus{0, 0.25f, 0, 0.75f, 0};
    std::mt19937_64 reference(seed);
    const auto uniform = [&] { return static_cast<double>(reference() >> 11) * 0x1.0p-53; };
    std::array<qwen::SpeculativeTerminalDecision, 2048> replay{};
    const auto make_input = [&](std::size_t trial) {
        qwen::SpeculativeWindow input; input.horizon = trial % 3;
        for (std::size_t i = 0; i < input.horizon; ++i) {
            input.p[i] = p; input.q[i] = q; input.draft_ids[i] = (trial / 3 + i) % 2;
        }
        input.p[input.horizon] = bonus;
        return input;
    };
    const auto make_stop = [](std::size_t trial) {
        return qwen::SpeculativeStopConfig{trial % 3 + 1 + trial % 2, (trial / 7) % 5, trial % 11 == 0};
    };
    for (std::size_t trial = 0; trial < replay.size(); ++trial) {
        const auto input = make_input(trial);
        const auto stop = make_stop(trial);
        qwen::SpeculativeTerminalDecision expected;
        bool complete = false;
        for (std::size_t i = 0; i < input.horizon; ++i) {
            const auto id = input.draft_ids[i];
            if (uniform() >= (id == 0 ? 0.5 : 1.0)) {
                const auto replacement = uniform() < 0.5 ? std::size_t{1} : std::size_t{2};
                expected.emitted_ids[expected.emitted_count++] = replacement;
                complete = true; break;
            }
            ++expected.accepted_drafts;
            expected.emitted_ids[expected.emitted_count++] = id;
            if (!stop.ignore_eos && id == stop.eos_id) { complete = true; break; }
            ++expected.forwarded_accepted_drafts; ++expected.restore_prefix;
        }
        if (!complete) expected.emitted_ids[expected.emitted_count++] = uniform() < 0.25 ? 1 : 3;
        expected.pending_carry = expected.emitted_ids[expected.emitted_count - 1];
        if (!stop.ignore_eos && expected.pending_carry == stop.eos_id)
            expected.stop_reason = qwen::SpeculativeStopReason::eos;
        else if (expected.emitted_count == stop.remaining_output_tokens)
            expected.stop_reason = qwen::SpeculativeStopReason::output_budget;
        replay[trial] = decide(owner.sampler, input, stop);
        check(same(expected, replay[trial]), "terminal independent uniform/quantile oracle");
    }
    hot([&] { owner.sampler.reset_seed(seed); });
    for (std::size_t trial = 0; trial < replay.size(); ++trial)
        check(same(replay[trial], decide(owner.sampler, make_input(trial), make_stop(trial))),
              "terminal reset-seed replay");
    hot([&] { owner.sampler.reset_seed(seed); });
    // Varied probability-0/1/interior accept outcomes, ignored EOS IDs, and budgets
    // must match the untouched nonterminal overload bit-for-bit and draw-for-draw.
    for (std::size_t trial = 0; trial < replay.size(); ++trial) {
        const auto input = make_input(trial);
        auto stop = make_stop(trial); stop.ignore_eos = true;
        check(same_outputs(decide(original.sampler, input), decide(owner.sampler, input, stop)),
              "ignored EOS changed old stochastic outputs");
        check(original.sampler.random_draws() == owner.sampler.random_draws(), "ignored EOS changed old RNG stream");
    }
    owner.verify(); original.verify();
}

void invalid_tests() {
    for (const auto n : {std::size_t{0}, static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1,
                         std::numeric_limits<std::size_t>::max()})
        rejected([&] { qwen::SpeculativeSampler sampler(n, qwen::SamplingConfig(1)); });
    for (int i = 0; i < 3; ++i) {
        auto config = qwen::SamplingConfig(1);
        if (i == 0) config.temperature = -1;
        if (i == 1) config.top_p = 0;
        if (i == 2) config.top_k = -1;
        rejected([&] { qwen::SpeculativeSampler sampler(5, config); });
    }
    captured_count = 0;
    capture_allocations = true;
    GuardedOwner owner(20261003);
    capture_allocations = false;
    check(captured_count == 3, "expected one Sampler and one residual workspace");
    const auto owned_allocations = captured;
    GuardedOwner twin(20261003);
    constexpr Probabilities target{0.5f, 0.25f, 0.125f, 0.125f, 0}, proposal{0, 0, 0, 0, 1};
    qwen::SpeculativeWindow valid;
    valid.horizon = 2;
    valid.p = {target, target, target};
    valid.q = {proposal, proposal};
    valid.draft_ids = {4, 4}; // Certain first rejection; all later inputs still must validate.
    check(same(decide(owner.sampler, valid), decide(twin.sampler, valid)), "initial twin stream mismatch");
    const auto fail = [&](const qwen::SpeculativeWindow& input) {
        const auto before = owner.sampler.random_draws();
        std::array<std::byte, sizeof(qwen::SpeculativeWindow)> bytes{};
        std::memcpy(bytes.data(), &input, bytes.size());
        rejected([&] { (void)owner.sampler.decide(input); });
        check(owner.sampler.random_draws() == before, "invalid input advanced RNG count");
        check(std::memcmp(bytes.data(), &input, bytes.size()) == 0, "invalid call changed descriptor");
        owner.verify();
        // Check the actual subsequent stream after EACH failure, not only count.
        for (int i = 0; i < 8; ++i)
            check(same(decide(owner.sampler, valid), decide(twin.sampler, valid)), "validation changed RNG stream");
        const qwen::SpeculativeStopConfig stop{3, 4, false};
        const auto terminal_before = owner.sampler.random_draws();
        rejected([&] { (void)owner.sampler.decide(input, stop); });
        check(owner.sampler.random_draws() == terminal_before, "terminal invalid input advanced RNG");
        check(std::memcmp(bytes.data(), &input, bytes.size()) == 0, "terminal invalid call changed descriptor");
        owner.verify();
        for (int i = 0; i < 8; ++i)
            check(same(decide(owner.sampler, valid, stop), decide(twin.sampler, valid, stop)),
                  "terminal validation changed RNG stream");
    };
    for (const auto h : {std::size_t{3}, std::numeric_limits<std::size_t>::max()}) {
        auto bad = valid; bad.horizon = h; fail(bad);
    }
    for (std::size_t slot = 0; slot < 5; ++slot) {
        const auto with_span = [&](std::span<const float> span) {
            auto bad = valid;
            if (slot < 3) bad.p[slot] = span; else bad.q[slot - 3] = span;
            fail(bad);
        };
        with_span({});
        with_span(std::span(target).first(4));
        with_span({static_cast<const float*>(nullptr), 5});
        constexpr auto last = std::numeric_limits<std::uintptr_t>::max() -
                              std::numeric_limits<std::uintptr_t>::max() % alignof(float);
        with_span({reinterpret_cast<const float*>(last), 5});
        with_span({reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(target.data()) + 1), 5});
        with_span({reinterpret_cast<const float*>(&owner.sampler), 5});
        const auto owner_end = reinterpret_cast<std::uintptr_t>(&owner.sampler) + sizeof(owner.sampler);
        with_span({reinterpret_cast<const float*>(owner_end - sizeof(float)), 5});
        for (std::size_t i = 0; i < 3; ++i) {
            const auto begin = reinterpret_cast<std::uintptr_t>(owned_allocations[i].data);
            for (const auto address : {begin, begin - sizeof(float), begin + owned_allocations[i].bytes - sizeof(float)})
                with_span({reinterpret_cast<const float*>(address), 5});
        }
        for (const float value : {-std::numeric_limits<float>::denorm_min(), std::nextafter(1.0f, 2.0f),
                                  std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                  std::bit_cast<float>(0x7fc12345U), std::bit_cast<float>(0x7f800001U)}) {
            for (std::size_t coordinate = 0; coordinate < 5; ++coordinate) {
                auto values = target; values[coordinate] = value;
                const auto saved = values;
                with_span(values);
                check(std::memcmp(values.data(), saved.data(), sizeof(values)) == 0, "bad distribution mutated");
            }
        }
        for (const auto values : {Probabilities{}, Probabilities{1, 1, 1, 1, 1},
                                  Probabilities{0.49f, 0.25f, 0.125f, 0.125f, 0},
                                  Probabilities{0.500002f, 0.25f, 0.125f, 0.125f, 0}})
            with_span(values);
    }
    for (std::size_t slot = 0; slot < 2; ++slot) {
        for (const auto id : {std::size_t{5}, qwen::speculative_empty_token, std::size_t{0}}) {
            auto bad = valid; bad.draft_ids[slot] = id; fail(bad); // q[0] == 0 for id0.
        }
    }
    for (std::size_t h = 0; h <= 1; ++h) {
        qwen::SpeculativeWindow short_window;
        short_window.horizon = h;
        for (std::size_t i = 0; i <= h; ++i) short_window.p[i] = target;
        for (std::size_t i = 0; i < h; ++i) { short_window.q[i] = proposal; short_window.draft_ids[i] = 4; }
        for (std::size_t i = h + 1; i < 3; ++i) { auto bad = short_window; bad.p[i] = target; fail(bad); }
        for (std::size_t i = h; i < 2; ++i) {
            auto bad = short_window; bad.q[i] = proposal; fail(bad);
            bad = short_window; bad.draft_ids[i] = 0; fail(bad);
        }
    }
    // Frozen normalization boundary accepted, the next FP32 ULP rejected.
    auto boundary = target;
    boundary[0] += static_cast<float>(qwen::sampling_probability_sum_tolerance);
    auto input = valid; input.p[2] = boundary;
    decide(owner.sampler, input); decide(twin.sampler, input);
    boundary[0] = std::nextafter(boundary[0], 1.0f);
    fail(input);
}

void terminal_invalid_tests() {
    captured_count = 0; capture_allocations = true;
    GuardedOwner owner(123456789);
    capture_allocations = false;
    check(captured_count == 3, "terminal overload added constructor workspace");
    const auto owned_allocations = captured;
    GuardedOwner twin(123456789);
    std::array<GuardedProbabilities<5>, 3> rows{};
    rows[0].values = {0, 0, 0, 0, 1};
    rows[1].values = {0.5f, 0.25f, 0.125f, 0.125f, 0};
    rows[2].values = {0.125f, 0.25f, 0.5f, 0.125f, 0};
    const auto saved = rows;
    qwen::SpeculativeWindow valid;
    valid.horizon = 2; valid.p = {rows[0].values, rows[1].values, rows[1].values};
    valid.q = {rows[0].values, rows[2].values}; valid.draft_ids = {4, 2};
    const qwen::SpeculativeStopConfig valid_stop{3, 4, false};
    qwen::SpeculativeWindow probe; probe.p[0] = rows[1].values;
    const auto fail = [&](const qwen::SpeculativeWindow& input, const qwen::SpeculativeStopConfig& stop) {
        const auto before = owner.sampler.random_draws();
        std::array<std::byte, sizeof(input)> input_bytes{};
        std::array<std::byte, sizeof(stop)> stop_bytes{};
        std::memcpy(input_bytes.data(), &input, input_bytes.size());
        std::memcpy(stop_bytes.data(), &stop, stop_bytes.size());
        rejected([&] { (void)owner.sampler.decide(input, stop); });
        check(owner.sampler.random_draws() == before, "invalid suffix behind EOS mutated RNG count");
        check(std::memcmp(input_bytes.data(), &input, input_bytes.size()) == 0, "invalid EOS window mutated descriptor");
        check(std::memcmp(stop_bytes.data(), &stop, stop_bytes.size()) == 0, "invalid EOS config mutated descriptor");
        for (int i = 0; i < 8; ++i) {
            check(same(decide(owner.sampler, valid, valid_stop), decide(twin.sampler, valid, valid_stop)),
                  "invalid EOS suffix changed subsequent accept stream");
            check(same(decide(owner.sampler, probe, {1, 4, false}), decide(twin.sampler, probe, {1, 4, false})),
                  "invalid EOS suffix changed subsequent categorical stream");
        }
        owner.verify(); twin.verify();
        for (std::size_t i = 0; i < rows.size(); ++i) {
            rows[i].verify(); check(rows[i].values == saved[i].values, "invalid terminal call mutated probabilities");
        }
    };
    for (const auto budget : {std::size_t{0}, std::size_t{1}, std::size_t{2}})
        for (const bool ignore : {false, true}) fail(valid, {budget, 4, ignore});
    for (const auto eos : {std::size_t{5}, qwen::speculative_empty_token})
        for (const bool ignore : {false, true}) fail(valid, {3, eos, ignore});
    for (const auto horizon : {std::size_t{3}, std::numeric_limits<std::size_t>::max()}) {
        auto bad = valid; bad.horizon = horizon;
        fail(bad, {std::numeric_limits<std::size_t>::max(), 4, false});
    }
    qwen::SpeculativeWindow short_window;
    short_window.horizon = 1; short_window.p = {rows[0].values, rows[1].values, {}};
    short_window.q[0] = rows[0].values; short_window.draft_ids[0] = 4;
    fail(short_window, {1, 4, false});
    auto bad = short_window; bad.p[2] = rows[1].values; fail(bad, {2, 4, false});
    bad = short_window; bad.q[1] = rows[2].values; fail(bad, {2, 4, false});
    bad = short_window; bad.draft_ids[1] = 0; fail(bad, {2, 4, false});
    for (const std::size_t slot : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        const auto with_span = [&](std::span<const float> span) {
            auto input = valid;
            if (slot < 3) input.p[slot] = span; else input.q[slot - 3] = span;
            fail(input, valid_stop);
        };
        with_span({}); with_span(std::span(rows[1].values).first(4));
        with_span({static_cast<const float*>(nullptr), 5});
        constexpr auto last = std::numeric_limits<std::uintptr_t>::max() -
                              std::numeric_limits<std::uintptr_t>::max() % alignof(float);
        with_span({reinterpret_cast<const float*>(last), 5});
        with_span({reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(rows[1].values.data()) + 1), 5});
        const auto owner_address = reinterpret_cast<std::uintptr_t>(&owner.sampler);
        for (const auto address : {owner_address, owner_address - sizeof(float),
                                  owner_address + sizeof(owner.sampler) - sizeof(float)})
            with_span({reinterpret_cast<const float*>(address), 5});
        for (std::size_t i = 0; i < 3; ++i) {
            const auto begin = reinterpret_cast<std::uintptr_t>(owned_allocations[i].data);
            for (const auto address : {begin, begin - sizeof(float), begin + owned_allocations[i].bytes - sizeof(float)})
                with_span({reinterpret_cast<const float*>(address), 5});
        }
        for (const float value : {-std::numeric_limits<float>::denorm_min(), std::nextafter(1.0f, 2.0f),
                                  std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                  std::bit_cast<float>(0x7fc12345U), std::bit_cast<float>(0x7f800001U)}) {
            for (std::size_t coordinate = 0; coordinate < 5; ++coordinate) {
                auto values = rows[1].values; values[coordinate] = value;
                const auto before = values; with_span(values);
                check(std::memcmp(values.data(), before.data(), sizeof(values)) == 0, "bad EOS suffix probability mutated");
            }
        }
        for (const auto values : {Probabilities{}, Probabilities{1, 1, 1, 1, 1},
                                  Probabilities{0.49f, 0.25f, 0.125f, 0.125f, 0},
                                  Probabilities{0.500002f, 0.25f, 0.125f, 0.125f, 0}})
            with_span(values);
    }
    for (const auto id : {std::size_t{5}, qwen::speculative_empty_token, std::size_t{4}}) {
        auto input = valid; input.draft_ids[1] = id; fail(input, valid_stop);
    }
    auto boundary = rows[1].values;
    boundary[0] += static_cast<float>(qwen::sampling_probability_sum_tolerance);
    auto input = valid; input.p[2] = boundary;
    check(same(decide(owner.sampler, input, valid_stop), decide(twin.sampler, input, valid_stop)),
          "EOS suffix frozen probability boundary rejected");
    boundary[0] = std::nextafter(boundary[0], 1.0f); fail(input, valid_stop);
}

void readonly_overlap_and_large_tests() {
    GuardedProbabilities<9> storage;
    storage.values.fill(0.2f);
    const auto saved = storage.values;
    qwen::SpeculativeSampler sampler(5, qwen::SamplingConfig(41));
    qwen::SpeculativeWindow input;
    input.horizon = 2;
    input.p = {std::span(storage.values).first(5), std::span(storage.values).subspan(1, 5),
               std::span(storage.values).subspan(4, 5)};
    input.q = {std::span(storage.values).subspan(2, 5), std::span(storage.values).first(5)};
    input.draft_ids = {4, 3};
    check(decide(sampler, input).accepted_drafts == 2, "read-only partial overlap rejected");
    check(decide(sampler, input, {3, 4, true}).accepted_drafts == 2, "terminal read-only partial overlap rejected");
    check(storage.values == saved, "read-only overlap mutated input");
    storage.verify();

    constexpr std::size_t vocabulary = 248320;
    std::vector<float> large(vocabulary + 2, 1.0f / static_cast<float>(vocabulary));
    large.front() = -17; large.back() = -19;
    const auto probabilities = std::span(large).subspan(1, vocabulary);
    qwen::SpeculativeSampler wide(vocabulary, qwen::SamplingConfig(42));
    input.p = {probabilities, probabilities, probabilities};
    input.q = {probabilities, probabilities};
    input.draft_ids = {15, vocabulary - 1};
    for (int i = 0; i < 8; ++i) check(decide(wide, input).accepted_drafts == 2, "full vocabulary window rejected");
    for (int i = 0; i < 8; ++i)
        check(decide(wide, input, {3, vocabulary - 1, false}).restore_prefix == 2,
              "full vocabulary accepted EOS was committed");
    check(large.front() == -17 && large.back() == -19, "large probability canary changed");
    for (const auto value : probabilities)
        check(value == 1.0f / static_cast<float>(vocabulary), "large probability input changed");
}

// Declared BEFORE execution: Bernoulli Bernstein bounds union-bounded over at
// most 300 coordinates/rates, family-wise alpha=1e-9, 200000 independent windows
// per fixture. Same 4*FP32-epsilon rounding allowance as sampling_test.cpp.
constexpr std::size_t trials = 200000;
constexpr double empirical_alpha = 1e-9, empirical_comparisons = 300;

void empirical_check(std::size_t count, long double expected) {
    const auto p = static_cast<double>(expected);
    const auto t = std::log(2 * empirical_comparisons / empirical_alpha);
    const auto bound = std::sqrt(2 * p * (1 - p) * t / trials) + 2 * t / (3 * trials) +
                       4 * std::numeric_limits<float>::epsilon();
    const auto error = std::abs(static_cast<double>(count) / trials - p);
    max_empirical_error = std::max(max_empirical_error, error);
    check(error <= bound, "empirical window distribution outside predeclared bound");
    if (expected == 0) check(count == 0, "empirical zero support invented");
    if (expected == 1) check(count == trials, "empirical certain event not exact");
}

std::array<long double, 5> normalized(const Probabilities& values) {
    long double sum = 0;
    for (const auto value : values) sum += value;
    std::array<long double, 5> result{};
    for (std::size_t i = 0; i < 5; ++i) result[i] = values[i] / sum;
    return result;
}

void stochastic_tests() {
    struct Fixture { Probabilities p, q; };
    constexpr Probabilities target{0.5f, 0.25f, 0.125f, 0.125f, 0};
    constexpr Probabilities skew{1 - 4.0f / 65536, 1.0f / 65536, 1.0f / 65536, 1.0f / 65536, 1.0f / 65536};
    constexpr std::array<Fixture, 5> fixtures{{
        {target, target}, {target, {0, 0, 0, 0, 1}},
        {target, {0.125f, 0.5f, 0.25f, 0.0625f, 0.0625f}},
        {target, skew}, {skew, {0.25f, 0.25f, 0.25f, 0.125f, 0.125f}}
    }};
    constexpr Probabilities second_base{0.125f, 0.5f, 0.25f, 0.125f, 0};
    constexpr Probabilities draft_base{0.375f, 0.125f, 0.125f, 0.125f, 0.25f};
    constexpr Probabilities bonus{0.125f, 0.25f, 0.5f, 0.125f, 0};
    std::array<GuardedProbabilities<5>, 5> second{}, draft{};
    for (std::size_t x = 0; x < 5; ++x)
        for (std::size_t y = 0; y < 5; ++y) {
            second[x].values[y] = second_base[(y + x) % 5];
            draft[x].values[y] = draft_base[(y + 2 * x) % 5];
        }
    const auto saved_second = second, saved_draft = draft;
    for (std::size_t f = 0; f < fixtures.size(); ++f) {
        GuardedOwner owner(7919 + 65537 * f);
        qwen::Sampler proposer(5, qwen::SamplingConfig(9176 + 131071 * f));
        std::array<std::size_t, 5> first_counts{}, second_counts{}, bonus_counts{};
        std::array<std::array<std::size_t, 5>, 5> joint{};
        std::array<std::size_t, 3> histogram{};
        std::size_t continuations = 0;
        for (std::size_t trial = 0; trial < trials; ++trial) {
            const auto a = hot([&] { return proposer.draw(fixtures[f].q); });
            const auto b = hot([&] { return proposer.draw(draft[a].values); });
            qwen::SpeculativeWindow input;
            input.horizon = 2;
            input.p = {fixtures[f].p, second[a].values, bonus};
            input.q = {fixtures[f].q, draft[a].values};
            input.draft_ids = {a, b};
            const auto result = decide(owner.sampler, input);
            ++histogram[result.accepted_drafts];
            const auto first = result.emitted_ids[0];
            std::size_t next;
            if (result.emitted_count == 1) {
                // A first rejection ends this window. To observe the actual next
                // output, a fresh horizon0 call uses target p(. | replacement).
                // This is a probability fixture, not model/state integration.
                qwen::SpeculativeWindow continuation;
                continuation.p[0] = second[first].values;
                next = decide(owner.sampler, continuation).pending_carry;
                ++continuations;
            } else {
                next = result.emitted_ids[1];
            }
            check(fixtures[f].p[first] > 0 && second[first].values[next] > 0, "sequence outside target support");
            ++first_counts[first]; ++second_counts[next]; ++joint[first][next];
            if (result.emitted_count == 3) ++bonus_counts[result.pending_carry];
        }
        const auto p0 = normalized(fixtures[f].p), q0 = normalized(fixtures[f].q);
        long double first_acceptance = 0, full_acceptance = 0;
        std::array<long double, 5> expected_second{};
        for (std::size_t x = 0; x < 5; ++x) {
            const auto p1 = normalized(second[x].values), q1 = normalized(draft[x].values);
            long double second_acceptance = 0;
            first_acceptance += std::min(p0[x], q0[x]);
            empirical_check(first_counts[x], p0[x]);
            for (std::size_t y = 0; y < 5; ++y) {
                second_acceptance += std::min(p1[y], q1[y]);
                empirical_check(joint[x][y], p0[x] * p1[y]);
                expected_second[y] += p0[x] * p1[y];
            }
            full_acceptance += std::min(p0[x], q0[x]) * second_acceptance;
        }
        const auto p_bonus = normalized(bonus);
        for (std::size_t y = 0; y < 5; ++y) {
            empirical_check(second_counts[y], expected_second[y]);
            empirical_check(bonus_counts[y], full_acceptance * p_bonus[y]);
        }
        empirical_check(histogram[0], 1 - first_acceptance);
        empirical_check(histogram[1], first_acceptance - full_acceptance);
        empirical_check(histogram[2], full_acceptance);
        check(continuations == histogram[0] && owner.sampler.random_draws() == 3 * trials &&
              proposer.random_draws() == 2 * trials, "two-output sequence RNG/count accounting");
        owner.verify();
        for (std::size_t x = 0; x < 5; ++x) {
            second[x].verify(); draft[x].verify();
            check(second[x].values == saved_second[x].values && draft[x].values == saved_draft[x].values,
                  "stochastic caller probabilities mutated");
        }
        std::cout << "{\"case\":\"nonterminal_window\",\"fixture\":" << f << ",\"trials\":" << trials
                  << ",\"accept0\":" << histogram[0] << ",\"accept1\":" << histogram[1]
                  << ",\"accept2\":" << histogram[2] << ",\"horizon0_continuations\":" << continuations << "}\n";
    }
}

void default_filtered_stochastic_test() {
    constexpr std::size_t vocabulary = 32; // Exercises default top20 and top-p .95.
    std::array<GuardedProbabilities<vocabulary>, 4> logits{}, probabilities{};
    qwen::Sampler filter(vocabulary, qwen::SamplingConfig(42));
    for (std::size_t row = 0; row < logits.size(); ++row) {
        for (std::size_t i = 0; i < vocabulary; ++i)
            logits[row].values[i] = static_cast<float>((13 * i + 7 * row) % vocabulary) / 16.0f;
        hot([&] { filter.distribution(logits[row].values, probabilities[row].values); });
        const auto support = std::count_if(probabilities[row].values.begin(), probabilities[row].values.end(),
                                          [](float p) { return p > 0; });
        check(support > 0 && support < 20, "fixture did not exercise default top-k then top-p");
    }
    check(filter.random_draws() == 0, "default filter consumed randomness");
    const auto saved_logits = logits, saved_probabilities = probabilities;
    qwen::Sampler proposer(vocabulary, qwen::SamplingConfig(20261003));
    qwen::SpeculativeSampler sampler(vocabulary, qwen::SamplingConfig(20261004));
    std::array<std::size_t, vocabulary> first_counts{}, second_counts{};
    qwen::SpeculativeWindow input;
    input.horizon = 2;
    input.p = {probabilities[0].values, probabilities[1].values, probabilities[2].values};
    input.q = {probabilities[0].values, probabilities[3].values};
    for (std::size_t trial = 0; trial < trials; ++trial) {
        input.draft_ids[0] = hot([&] { return proposer.draw(input.q[0]); });
        input.draft_ids[1] = hot([&] { return proposer.draw(input.q[1]); });
        const auto result = decide(sampler, input);
        check(result.accepted_drafts >= 1, "identical filtered first row rejected");
        ++first_counts[result.emitted_ids[0]];
        ++second_counts[result.emitted_ids[1]];
        check(input.p[0][result.emitted_ids[0]] > 0 && input.p[1][result.emitted_ids[1]] > 0,
              "speculative output escaped default target filters");
    }
    for (std::size_t row = 0; row < 2; ++row) {
        long double sum = 0;
        for (const auto p : probabilities[row].values) sum += p;
        for (std::size_t i = 0; i < vocabulary; ++i)
            empirical_check(row == 0 ? first_counts[i] : second_counts[i], probabilities[row].values[i] / sum);
    }
    check(sampler.random_draws() == 3 * trials && proposer.random_draws() == 2 * trials,
          "default-filtered fixture RNG accounting");
    for (std::size_t row = 0; row < logits.size(); ++row) {
        logits[row].verify(); probabilities[row].verify();
        check(logits[row].values == saved_logits[row].values &&
              probabilities[row].values == saved_probabilities[row].values, "default-filtered inputs mutated");
    }
    std::cout << "{\"case\":\"default_filtered_window\",\"trials\":" << trials
               << ",\"temperature\":1,\"top_p\":0.95,\"top_k\":20,\"passed\":true}\n";
}

// A separately declared terminal family: <=300 Bernoulli coordinates/rates,
// alpha=1e-9, 200000 independent truncated chains per fixture. Original tests
// retain their original family and all numerical gates. These fixtures describe
// a small CPU probability chain, not a complete trained MTP/state integration.
constexpr double terminal_empirical_comparisons = 300;

void terminal_empirical_check(std::size_t count, long double expected) {
    const auto p = static_cast<double>(expected);
    const auto t = std::log(2 * terminal_empirical_comparisons / empirical_alpha);
    const auto bound = std::sqrt(2 * p * (1 - p) * t / trials) + 2 * t / (3 * trials) +
                       4 * std::numeric_limits<float>::epsilon();
    const auto error = std::abs(static_cast<double>(count) / trials - p);
    max_terminal_empirical_error = std::max(max_terminal_empirical_error, error);
    check(error <= bound, "truncated terminal distribution outside predeclared bound");
    if (expected == 0) check(count == 0, "terminal statistical zero support invented");
    if (expected == 1) check(count == trials, "terminal statistical certain event not exact");
}

void terminal_stochastic_tests() {
    constexpr std::size_t eos = 4;
    struct Fixture { Probabilities p, q; };
    constexpr std::array<Fixture, 3> fixtures{{
        {{0.375f, 0.25f, 0.125f, 0.125f, 0.125f}, {0.375f, 0.25f, 0.125f, 0.125f, 0.125f}},
        {{0.375f, 0.25f, 0.125f, 0.125f, 0.125f}, {0.125f, 0.375f, 0.25f, 0.125f, 0.125f}},
        {{0.25f, 0.125f, 0.25f, 0.125f, 0.25f}, {0.125f, 0.375f, 0.125f, 0.25f, 0.125f}}
    }};
    constexpr Probabilities second_base{0.125f, 0.375f, 0.25f, 0.125f, 0.125f};
    constexpr Probabilities draft_base{0.375f, 0.125f, 0.125f, 0.125f, 0.25f};
    constexpr Probabilities bonus{0.125f, 0.25f, 0.375f, 0.125f, 0.125f};
    std::array<GuardedProbabilities<5>, 5> second{}, draft{};
    for (std::size_t x = 0; x < 5; ++x)
        for (std::size_t y = 0; y < 5; ++y) {
            second[x].values[y] = second_base[(y + x) % 5];
            draft[x].values[y] = draft_base[(y + 2 * x) % 5];
        }
    const auto saved_second = second, saved_draft = draft;
    for (std::size_t f = 0; f < fixtures.size(); ++f) {
        const auto budget = f + 1;
        GuardedOwner owner(65537 + 104729 * f);
        qwen::Sampler proposer(5, qwen::SamplingConfig(131071 + 65537 * f));
        std::array<std::array<std::size_t, 5>, 3> token_counts{};
        std::array<std::array<std::size_t, 5>, 5> joint{};
        std::array<std::size_t, 3> lengths{}, eos_positions{};
        std::size_t budget_stops = 0, continuations = 0;
        for (std::size_t trial = 0; trial < trials; ++trial) {
            qwen::SpeculativeWindow input;
            input.horizon = budget - 1; input.p[0] = fixtures[f].p;
            if (input.horizon > 0) {
                input.q[0] = fixtures[f].q;
                input.draft_ids[0] = hot([&] { return proposer.draw(input.q[0]); });
                input.p[1] = second[input.draft_ids[0]].values;
            }
            if (input.horizon > 1) {
                input.q[1] = draft[input.draft_ids[0]].values;
                input.draft_ids[1] = hot([&] { return proposer.draw(input.q[1]); });
                input.p[2] = bonus;
            }
            auto result = decide(owner.sampler, input, {budget, eos, false});
            std::array<std::size_t, 3> outputs{};
            std::size_t count = 0;
            while (true) {
                for (std::size_t i = 0; i < result.emitted_count; ++i) outputs[count++] = result.emitted_ids[i];
                if (result.stop_reason != qwen::SpeculativeStopReason::window_complete) break;
                check(count < budget && outputs[count - 1] != eos, "continued a stopped probability chain");
                // Following a rejection, the next target row is conditioned on
                // the actual replacement, not the rejected proposal's suffix.
                qwen::SpeculativeWindow continuation;
                continuation.p[0] = count == 1 ? std::span<const float>(second[outputs[0]].values)
                                                : std::span<const float>(bonus);
                result = decide(owner.sampler, continuation, {budget - count, eos, false});
                ++continuations;
            }
            check(count >= 1 && count <= budget, "truncated chain length outside budget");
            for (std::size_t i = 0; i < count; ++i) {
                const auto probabilities = i == 0 ? std::span<const float>(fixtures[f].p)
                    : i == 1 ? std::span<const float>(second[outputs[0]].values) : std::span<const float>(bonus);
                check(probabilities[outputs[i]] > 0, "truncated output outside conditional target support");
                check(outputs[i] != eos || i + 1 == count, "truncated chain emitted after EOS");
                ++token_counts[i][outputs[i]];
            }
            if (count >= 2) ++joint[outputs[0]][outputs[1]];
            ++lengths[count - 1];
            if (outputs[count - 1] == eos) {
                ++eos_positions[count - 1];
                check(result.stop_reason == qwen::SpeculativeStopReason::eos, "truncated EOS final reason");
            } else {
                ++budget_stops;
                check(count == budget && result.stop_reason == qwen::SpeculativeStopReason::output_budget,
                      "truncated budget final reason");
            }
        }
        const auto p0 = normalized(fixtures[f].p), p_bonus = normalized(bonus);
        std::array<long double, 5> expected_second{}, expected_third{};
        long double survives_second = 0;
        for (std::size_t x = 0; x < 5; ++x) {
            terminal_empirical_check(token_counts[0][x], p0[x]);
            const auto p1 = normalized(second[x].values);
            for (std::size_t y = 0; y < 5; ++y) {
                const auto expected_joint = budget >= 2 && x != eos ? p0[x] * p1[y] : 0;
                terminal_empirical_check(joint[x][y], expected_joint);
                expected_second[y] += expected_joint;
                if (y != eos) survives_second += expected_joint;
            }
        }
        for (std::size_t y = 0; y < 5; ++y) {
            expected_third[y] = budget >= 3 ? survives_second * p_bonus[y] : 0;
            terminal_empirical_check(token_counts[1][y], expected_second[y]);
            terminal_empirical_check(token_counts[2][y], expected_third[y]);
        }
        const std::array<long double, 3> expected_eos{p0[eos], expected_second[eos], expected_third[eos]};
        long double stops_eos = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            terminal_empirical_check(eos_positions[i], expected_eos[i]);
            const auto expected_length = i + 1 > budget ? 0 : i + 1 < budget ? expected_eos[i] : 1 - stops_eos;
            terminal_empirical_check(lengths[i], expected_length);
            stops_eos += expected_eos[i];
        }
        terminal_empirical_check(budget_stops, 1 - stops_eos);
        check(proposer.random_draws() == (budget - 1) * trials, "truncated draft RNG count");
        owner.verify();
        for (std::size_t x = 0; x < 5; ++x) {
            second[x].verify(); draft[x].verify();
            check(second[x].values == saved_second[x].values && draft[x].values == saved_draft[x].values,
                  "truncated chain probabilities mutated");
        }
        std::cout << "{\"case\":\"terminal_truncated_chain\",\"budget\":" << budget << ",\"trials\":" << trials
                  << ",\"eos_first\":" << eos_positions[0] << ",\"eos_second\":" << eos_positions[1]
                  << ",\"eos_third\":" << eos_positions[2] << ",\"budget_stops\":" << budget_stops
                  << ",\"horizon0_continuations\":" << continuations << "}\n";
    }
}

} // namespace

int main() {
    try {
        allocation_probe_test();
        deterministic_tests();
        seed_tests();
        terminal_deterministic_tests();
        terminal_seed_tests();
        invalid_tests();
        terminal_invalid_tests();
        readonly_overlap_and_large_tests();
        stochastic_tests();
        default_filtered_stochastic_test();
        terminal_stochastic_tests();
        std::cout << "{\"test\":\"speculative_decisions\",\"checks\":" << checks << ",\"rejections\":" << rejections
                  << ",\"windows\":" << windows << ",\"hot_calls\":" << hot_calls
                   << ",\"hot_allocations\":" << hot_allocations << ",\"max_empirical_error\":" << max_empirical_error
                   << ",\"max_terminal_empirical_error\":" << max_terminal_empirical_error
                  << ",\"empirical_family_alpha\":" << empirical_alpha << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "speculative test: " << error.what() << '\n';
        return 1;
    }
}
