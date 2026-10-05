#include "routes.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace qwen {
namespace {

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(message);
}

void validate_range(const void* data, std::size_t bytes) {
    const auto start = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || bytes > std::numeric_limits<std::uintptr_t>::max() - start)
        invalid("routes: null or wrapping input range");
}

// Flat address ranges on the supported Linux hosts; no relational comparison or
// pointer arithmetic between unrelated allocations. Input endpoints are checked
// before this call, and owned ranges come from bounded constructor allocations.
bool overlaps(const void* data, std::size_t bytes, const void* owned,
              std::size_t owned_bytes) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(data);
    const auto b = reinterpret_cast<std::uintptr_t>(owned);
    return a < b ? b - a < bytes : a - b < owned_bytes;
}

} // namespace

std::size_t RouteGroups::checked_capacity(int token_capacity, Bound bound) {
    static_assert(max_tokens <= std::numeric_limits<int>::max() / top_k);
    static_assert(static_cast<std::size_t>(max_tokens) <=
                  std::numeric_limits<std::size_t>::max() / top_k / sizeof(RouteAssignment));
    int limit = 0;
    switch (bound) {
    case Bound::chunk1024: limit = 1024; break;
    case Bound::layerwise16384: limit = max_tokens; break;
    default: invalid("routes: unknown constructor bound");
    }
    if (token_capacity < 1 || token_capacity > limit)
        invalid("routes: token capacity outside explicit constructor bound");
    return static_cast<std::size_t>(token_capacity) * top_k;
}

RouteGroups::RouteGroups(int token_capacity, Bound bound)
    : token_capacity_(token_capacity), assignments_(checked_capacity(token_capacity, bound)) {}

bool RouteGroups::aliases_owned(const void* data, std::size_t bytes) const noexcept {
    // Include all inline scratch/metadata and the full allocated vector storage,
    // not only the current logical output. This also rejects stale output views.
    return overlaps(data, bytes, this, sizeof(*this)) ||
           overlaps(data, bytes, assignments_.data(),
                    assignments_.capacity() * sizeof(RouteAssignment));
}

void RouteGroups::prepare(std::span<const std::int32_t> ids,
                          std::span<const float> weights, int tokens) {
    // Check signed bounds before conversion/multiplication, and exact shapes
    // before bytes/endpoints or reads. expected <= 163840, so all scan positions,
    // cursors and byte counts are representable and fit constructor storage.
    if (tokens < 1 || tokens > token_capacity_)
        invalid("routes: tokens outside constructor capacity");
    const auto expected = static_cast<std::size_t>(tokens) * top_k;
    if (ids.size() != expected || weights.size() != expected)
        invalid("routes: expected exactly tokens*10 IDs and weights");
    const auto id_bytes = expected * sizeof(std::int32_t);
    const auto weight_bytes = expected * sizeof(float);
    validate_range(ids.data(), id_bytes);
    validate_range(weights.data(), weight_bytes);
    if (aliases_owned(ids.data(), id_bytes) || aliases_owned(weights.data(), weight_bytes))
        invalid("routes: input overlaps owned storage");

    // Validate the entire chunk before touching even scratch. top10 makes a
    // bounded pairwise uniqueness check simpler than another clearing/tag array.
    for (int token = 0; token < tokens; ++token) {
        const auto base = static_cast<std::size_t>(token) * top_k;
        for (int rank = 0; rank < top_k; ++rank) {
            const auto index = base + static_cast<std::size_t>(rank);
            if (ids[index] < 0 || ids[index] >= expert_count)
                invalid("routes: expert ID must be 0..511");
            if (!std::isfinite(weights[index]) || weights[index] < 0.0f)
                invalid("routes: weight must be finite and nonnegative");
            for (int earlier = 0; earlier < rank; ++earlier)
                if (ids[base + static_cast<std::size_t>(earlier)] == ids[index])
                    invalid("routes: duplicate expert ID within token");
        }
    }

    // No fallible operations below: stable histogram / exclusive scan / scatter.
    counts_.fill(0);
    for (const auto expert : ids) ++counts_[static_cast<std::size_t>(expert)];
    offsets_[0] = 0;
    for (int expert = 0; expert < expert_count; ++expert) {
        const auto e = static_cast<std::size_t>(expert);
        offsets_[e + 1] = offsets_[e] + counts_[e];
        cursors_[e] = offsets_[e];
    }
    for (int token = 0; token < tokens; ++token) {
        const auto base = static_cast<std::size_t>(token) * top_k;
        for (int rank = 0; rank < top_k; ++rank) {
            const auto index = base + static_cast<std::size_t>(rank);
            const auto expert = ids[index];
            auto& out = assignments_[static_cast<std::size_t>(
                cursors_[static_cast<std::size_t>(expert)]++)];
            out.token = token;
            out.rank = rank;
            out.expert = expert;
            std::memcpy(&out.weight, &weights[index], sizeof(float));
        }
    }
    assignment_count_ = expected;
}

std::span<const RouteAssignment> RouteGroups::assignments() const noexcept {
    return {assignments_.data(), assignment_count_};
}

std::span<const RouteAssignment> RouteGroups::group(int expert) const {
    if (expert < 0 || expert >= expert_count)
        invalid("routes: group expert must be 0..511");
    const auto e = static_cast<std::size_t>(expert);
    return {assignments_.data() + offsets_[e], static_cast<std::size_t>(counts_[e])};
}

std::span<const int> RouteGroups::counts() const noexcept {
    return counts_;
}

} // namespace qwen
