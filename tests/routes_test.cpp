#include "routes.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
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

// Cover scalar/array/aligned C++ allocations, including the first successful
// prepare. Exceptions and the independent test oracle are outside hot intervals.
[[gnu::noinline]] void* operator new(std::size_t n) { return allocate(n); }
[[gnu::noinline]] void* operator new[](std::size_t n) { return allocate(n); }
[[gnu::noinline]] void* operator new(std::size_t n, std::align_val_t a) {
    return allocate_aligned(n, static_cast<std::size_t>(a));
}
[[gnu::noinline]] void* operator new[](std::size_t n, std::align_val_t a) {
    return allocate_aligned(n, static_cast<std::size_t>(a));
}
[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {

std::size_t checks = 0, rejections = 0, cases = 0, hot_calls = 0, hot_allocations = 0;
constexpr std::size_t experts = 512;
constexpr std::size_t top_k = 10;

static_assert(std::is_same_v<std::int32_t, int>); // Supported Linux host ABI.
static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559);
static_assert(!std::is_copy_constructible_v<qwen::RouteGroups>);
static_assert(!std::is_move_constructible_v<qwen::RouteGroups>);

void check(bool condition, const char* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

template<class Function> void rejected(Function function, const char* guard = nullptr) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        ++rejections;
        if (guard != nullptr)
            check(std::string_view(error.what()).find(guard) != std::string_view::npos,
                  "wrong rejection guard");
        return;
    }
    throw std::runtime_error("invalid route arguments unexpectedly accepted");
}

bool equal(const qwen::RouteAssignment& a, const qwen::RouteAssignment& b) {
    return a.token == b.token && a.rank == b.rank && a.expert == b.expert &&
           std::bit_cast<std::uint32_t>(a.weight) == std::bit_cast<std::uint32_t>(b.weight);
}

struct Fixture {
    std::vector<std::int32_t> ids;
    std::vector<float> weights;

    // Varied IDs cover all experts; repeated IDs occur in different original
    // ranks across tokens. All-zero still carries all ten (signed-zero) weights.
    Fixture(int tokens, int pattern)
        : ids(static_cast<std::size_t>(tokens) * top_k), weights(ids.size()) {
        constexpr std::array<std::int32_t, top_k> repeated{511, 0, 510, 1, 127,
                                                         255, 256, 257, 400, 41};
        constexpr std::array<std::uint32_t, top_k> bits{0, 0x80000000U, 1, 0x007fffffU,
            0x00800000U, 0x3dcccccdU, 0x3f800000U, 0x7f7fffffU, 0x3f000001U, 0x4f123456U};
        for (std::size_t token = 0; token < static_cast<std::size_t>(tokens); ++token) {
            for (std::size_t rank = 0; rank < top_k; ++rank) {
                const auto index = token * top_k + rank;
                ids[index] = pattern == 0 ? static_cast<std::int32_t>((511 + 37 * token + 73 * rank) % experts)
                                         : repeated[(rank + 3 * token) % top_k];
                const auto raw = pattern == 2 ? ((token + rank) % 2 == 0 ? 0U : 0x80000000U)
                                             : bits[(11 * token + 3 * rank) % top_k];
                weights[index] = std::bit_cast<float>(raw);
            }
        }
    }
};

struct Expected {
    std::vector<qwen::RouteAssignment> assignments;
    std::array<int, experts> counts{};
};

// Independent expert-by-expert filter oracle: deliberately no histogram,
// exclusive scan, scatter, sorting, or call to the production implementation.
Expected oracle(const Fixture& input, int tokens) {
    Expected result;
    result.assignments.reserve(input.ids.size());
    for (std::size_t expert = 0; expert < experts; ++expert) {
        for (int token = 0; token < tokens; ++token) {
            for (std::size_t rank = 0; rank < top_k; ++rank) {
                const auto index = static_cast<std::size_t>(token) * top_k + rank;
                if (input.ids[index] != static_cast<std::int32_t>(expert)) continue;
                result.assignments.push_back({token, static_cast<int>(rank),
                                              static_cast<int>(expert), input.weights[index]});
                ++result.counts[expert];
            }
        }
    }
    return result;
}

struct Snapshot {
    std::span<const qwen::RouteAssignment> assignments;
    std::span<const int> counts;
    std::array<std::span<const qwen::RouteAssignment>, experts> groups;
    std::vector<std::byte> bytes;
    std::array<int, experts> count_values;

    explicit Snapshot(const qwen::RouteGroups& owner)
        : assignments(owner.assignments()), counts(owner.counts()),
          bytes(std::as_bytes(assignments).begin(), std::as_bytes(assignments).end()) {
        std::copy(counts.begin(), counts.end(), count_values.begin());
        for (std::size_t e = 0; e < experts; ++e) groups[e] = owner.group(static_cast<int>(e));
    }

    void unchanged(const qwen::RouteGroups& owner) const {
        const auto current = owner.assignments();
        check(current.data() == assignments.data() && current.size() == assignments.size(),
              "rejection changed assignment pointer/extent");
        check(std::equal(bytes.begin(), bytes.end(), std::as_bytes(current).begin()),
              "rejection changed output bytes");
        check(std::equal(bytes.begin(), bytes.end(), std::as_bytes(assignments).begin()),
              "rejection invalidated a borrowed assignment view");
        check(owner.counts().data() == counts.data() && owner.counts().size() == experts,
              "rejection changed count pointer/extent");
        check(std::equal(counts.begin(), counts.end(), count_values.begin()),
              "rejection changed borrowed counts");
        for (std::size_t e = 0; e < experts; ++e) {
            const auto group = owner.group(static_cast<int>(e));
            check(group.data() == groups[e].data() && group.size() == groups[e].size(),
                  "rejection changed a group pointer/extent");
        }
    }
};

template<class Function>
void rejected_unchanged(qwen::RouteGroups& owner, Function function, const char* guard = nullptr) {
    const Snapshot snapshot(owner);
    rejected(function, guard);
    snapshot.unchanged(owner);
}

void prepare(qwen::RouteGroups& owner, const Fixture& input, int tokens) {
    const auto before = allocations;
    owner.prepare(input.ids, input.weights, tokens);
    check(owner.assignments().size() == static_cast<std::size_t>(tokens) * top_k,
          "successful prepare output extent");
    check(owner.counts().size() == experts, "counts not exactly 512");
    for (int e = 0; e < static_cast<int>(experts); ++e)
        check(owner.group(e).data() != nullptr, "null empty group span");
    hot_allocations += allocations - before;
    ++hot_calls;
    check(allocations == before, "successful prepare/accessor hot allocation");
}

void same(const qwen::RouteGroups& owner, const Expected& expected, int tokens) {
    const auto assignments = owner.assignments();
    check(assignments.size() == expected.assignments.size(), "lost/extra contributions");
    for (std::size_t i = 0; i < assignments.size(); ++i)
        check(equal(assignments[i], expected.assignments[i]), "assignment/order/raw weight differs from oracle");
    check(std::equal(owner.counts().begin(), owner.counts().end(), expected.counts.begin()),
          "histogram differs from oracle");
    std::size_t offset = 0;
    for (std::size_t e = 0; e < experts; ++e) {
        const auto group = owner.group(static_cast<int>(e));
        check(group.size() == static_cast<std::size_t>(expected.counts[e]), "wrong group extent");
        check(group.data() == assignments.data() + offset, "group not a contiguous ascending slice");
        check(group.size() <= static_cast<std::size_t>(tokens), "per-token duplicate escaped validation");
        for (std::size_t i = 0; i < group.size(); ++i)
            check(equal(group[i], expected.assignments[offset + i]), "group contents/order differ");
        offset += group.size();
    }
    check(offset == static_cast<std::size_t>(tokens) * top_k, "group count conservation");
}

void correctness_tests() {
    for (const int capacity : {1, 2, 3, 128, 1024}) {
        qwen::RouteGroups owner(capacity);
        const auto* assignment_storage = owner.assignments().data();
        const auto* count_storage = owner.counts().data();
        check(owner.assignments().empty() && assignment_storage != nullptr, "initial assignment view");
        check(owner.counts().size() == experts, "initial counts not 512");
        for (int e = 0; e < static_cast<int>(experts); ++e) {
            check(owner.counts()[static_cast<std::size_t>(e)] == 0, "initial count not zero");
            check(owner.group(e).empty() && owner.group(e).data() == assignment_storage,
                  "initial empty group invalid");
        }
        // Reverse N on a second pass to exercise shrinking/growing reuse as well
        // as exact constructor capacity; no fresh hot allocations are permitted.
        constexpr std::array<int, 10> lengths{1, 2, 3, 128, 1024, 1024, 128, 3, 2, 1};
        for (const int tokens : lengths) {
            if (tokens > capacity) continue;
            for (int pattern = 0; pattern < 3; ++pattern) {
                const Fixture input(tokens, pattern);
                const auto ids_before = input.ids;
                const auto weights_before = input.weights;
                const auto expected = oracle(input, tokens);
                prepare(owner, input, tokens);
                same(owner, expected, tokens);
                check(owner.assignments().data() == assignment_storage &&
                      owner.counts().data() == count_storage, "successful prepare reallocated storage");
                check(input.ids == ids_before &&
                      std::memcmp(input.weights.data(), weights_before.data(),
                                  input.weights.size() * sizeof(float)) == 0, "prepare changed inputs");
                if (pattern != 0)
                    check(owner.group(511).size() == static_cast<std::size_t>(tokens),
                          "cutoff expert 511 or repeated/zero contribution lost");
                const Snapshot snapshot(owner);
                prepare(owner, input, tokens);
                snapshot.unchanged(owner); // deterministic replay, including raw bits
                ++cases;
            }
        }
    }
}

void invalid_tests() {
    for (const int capacity : {std::numeric_limits<int>::min(), -1, 0, 1025,
                               std::numeric_limits<int>::max()})
        rejected([&] { qwen::RouteGroups invalid_owner(capacity); }, "token capacity");

    qwen::RouteGroups owner(128);
    const Fixture baseline(128, 0);
    prepare(owner, baseline, 128);
    const auto fail = [&](std::span<const std::int32_t> ids, std::span<const float> weights,
                          int tokens, const char* guard = nullptr) {
        rejected_unchanged(owner, [&] { owner.prepare(ids, weights, tokens); }, guard);
    };
    const Fixture input(3, 1);
    for (const int tokens : {std::numeric_limits<int>::min(), -1, 0, 129,
                             std::numeric_limits<int>::max()})
        fail({}, {}, tokens, "tokens outside");
    fail({}, input.weights, 3, "exactly");
    fail(input.ids, {}, 3, "exactly");
    fail(std::span(input.ids).first(29), input.weights, 3, "exactly");
    fail(input.ids, std::span(input.weights).first(29), 3, "exactly");
    fail(std::span(baseline.ids).first(31), input.weights, 3, "exactly");
    fail(input.ids, std::span(baseline.weights).first(31), 3, "exactly");
    fail(baseline.ids, baseline.weights, 3, "exactly");
    for (const std::size_t position : {std::size_t{0}, std::size_t{14}, std::size_t{29}}) {
        for (const std::int32_t id : {std::numeric_limits<std::int32_t>::min(), -1, 512,
                                      std::numeric_limits<std::int32_t>::max()}) {
            auto bad = input;
            bad.ids[position] = id;
            fail(bad.ids, bad.weights, 3, "expert ID");
        }
        for (const std::uint32_t raw : {0x80000001U, 0xbf800000U, 0xff7fffffU,
                                        0x7fc00001U, 0x7f800001U, 0xffc12345U,
                                        0x7f800000U, 0xff800000U}) {
            auto bad = input;
            bad.weights[position] = std::bit_cast<float>(raw);
            fail(bad.ids, bad.weights, 3, "weight");
        }
    }
    for (const std::size_t position : {std::size_t{1}, std::size_t{14}, std::size_t{29}}) {
        auto bad = input;
        bad.ids[position] = bad.ids[position / top_k * top_k];
        fail(bad.ids, bad.weights, 3, "duplicate");
        std::fill(bad.weights.begin(), bad.weights.end(), 0.0f);
        fail(bad.ids, bad.weights, 3, "duplicate"); // zero does not excuse duplicates
    }
    for (const int expert : {std::numeric_limits<int>::min(), -1, 512,
                             std::numeric_limits<int>::max()})
        rejected_unchanged(owner, [&] { (void)owner.group(expert); }, "group expert");

    // Synthetic null/wrapping addresses probe the numerical guards only. These
    // ranges are never dereferenced; malformed geometry must reject them first.
    fail({static_cast<const std::int32_t*>(nullptr), 30}, input.weights, 3, "null or wrapping");
    fail(input.ids, {static_cast<const float*>(nullptr), 30}, 3, "null or wrapping");
    constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
    const auto id_address = max_address - max_address % alignof(std::int32_t);
    const auto weight_address = max_address - max_address % alignof(float);
    fail({reinterpret_cast<const std::int32_t*>(id_address), 30}, input.weights, 3, "null or wrapping");
    fail(input.ids, {reinterpret_cast<const float*>(weight_address), 30}, 3, "null or wrapping");

    // Both inputs: direct and interior aliases of counts, assignments and owner
    // metadata. Demand the alias guard so invalid aliased values cannot mask a
    // missing overlap check. No test writes through a returned const view.
    const auto alias_ids = [&](const void* data) {
        fail({static_cast<const std::int32_t*>(data), 10}, std::span(input.weights).first(10),
             1, "overlaps owned");
    };
    const auto alias_weights = [&](const void* data) {
        fail(std::span(input.ids).first(10), {static_cast<const float*>(data), 10},
             1, "overlaps owned");
    };
    for (const std::size_t offset : {std::size_t{0}, std::size_t{3}, experts - top_k}) {
        alias_ids(const_cast<int*>(owner.counts().data()) + offset);
        alias_weights(owner.counts().data() + offset);
    }
    const auto large_view = owner.assignments();
    for (const std::size_t offset : {std::size_t{0}, std::size_t{1}, std::size_t{1260}}) {
        alias_ids(large_view.data() + offset);
        alias_weights(large_view.data() + offset);
    }
    alias_ids(&owner);
    alias_weights(&owner);
    alias_ids(owner.group(511).data());
    alias_weights(owner.group(511).data());

    // A stale view into unused capacity is still internal writable storage.
    prepare(owner, input, 3);
    same(owner, oracle(input, 3), 3);
    alias_ids(large_view.data() + 1260);
    alias_weights(large_view.data() + 1260);
    prepare(owner, baseline, 128); // recovery after all rejected prepares
    same(owner, oracle(baseline, 128), 128);

    qwen::RouteGroups fresh(1);
    auto bad = Fixture(1, 0);
    bad.weights.back() = -1.0f;
    rejected_unchanged(fresh, [&] { fresh.prepare(bad.ids, bad.weights, 1); });
    const Fixture good(1, 2);
    prepare(fresh, good, 1);
    same(fresh, oracle(good, 1), 1);
    rejected_unchanged(fresh, [&] { fresh.prepare(input.ids, input.weights, 3); }, "tokens outside");
}

void allocation_probe_test() {
    const auto before = allocations;
    void* scalar = ::operator new(4);
    void* array = ::operator new[](16);
    void* aligned = ::operator new(64, std::align_val_t{64});
    void* aligned_array = ::operator new[](128, std::align_val_t{64});
    check(allocations - before == 4, "allocation probe not observing all new forms");
    ::operator delete(scalar);
    ::operator delete[](array);
    ::operator delete(aligned, std::align_val_t{64});
    ::operator delete[](aligned_array, std::align_val_t{64});
}

} // namespace

int main() {
    try {
        allocation_probe_test();
        correctness_tests();
        invalid_tests();
        std::cout << "{\"test\":\"routes\",\"checks\":" << checks << ",\"cases\":" << cases
                  << ",\"rejections\":" << rejections << ",\"hot_calls\":" << hot_calls
                  << ",\"hot_allocations\":" << hot_allocations << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "routes test: " << error.what() << '\n';
        return 1;
    }
}
