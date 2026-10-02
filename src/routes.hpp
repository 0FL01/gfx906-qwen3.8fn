#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qwen {

struct RouteAssignment {
    int token;
    int rank;
    int expert;
    float weight;
};

// Fixed 512-expert/top10 CPU routing map, not an expert/activation scheduler.
// Constructor capacity is 1..1024 tokens; all storage is allocated once there.
// Successful prepare/accessor calls allocate nothing and create no threads.
// One owner, not concurrent; copying/moving is disabled to keep views owner-bound.
// Initially assignments/groups are empty and all 512 counts are zero.
// Views are read-only and remain backed by the same storage until destruction;
// reacquire their extents/contents after successful prepare. Rejected calls leave
// all previously published views and their contents unchanged.
class RouteGroups {
public:
    explicit RouteGroups(int token_capacity);
    RouteGroups(const RouteGroups&) = delete;
    RouteGroups& operator=(const RouteGroups&) = delete;
    RouteGroups(RouteGroups&&) = delete;
    RouteGroups& operator=(RouteGroups&&) = delete;

    // Exact token-major [tokens][10] inputs, tokens in 1..constructor capacity.
    // IDs must be 0..511 and unique within each token (reuse across tokens is
    // valid). Weights must be finite/nonnegative, including either signed zero.
    // Preserve every contribution, original rank and raw weight bits. Groups
    // ascend by expert ID; within each group order is token, then original rank.
    // Bad geometry/values or ANY input byte overlap with this owner's storage
    // throws invalid_argument before output mutation. Do not cast away view const.
    // Input spans must otherwise designate live, readable objects for this call.
    void prepare(std::span<const std::int32_t> ids, std::span<const float> weights,
                 int tokens);
    std::span<const RouteAssignment> assignments() const noexcept;
    // expert in 0..511, otherwise invalid_argument; empty groups have valid spans.
    std::span<const RouteAssignment> group(int expert) const;
    std::span<const int> counts() const noexcept;

private:
    static constexpr int expert_count = 512;
    static constexpr int top_k = 10;
    static constexpr int max_tokens = 1024;
    static std::size_t checked_capacity(int token_capacity);
    bool aliases_owned(const void* data, std::size_t bytes) const noexcept;

    int token_capacity_;
    std::vector<RouteAssignment> assignments_;
    std::array<int, expert_count> counts_{};
    std::array<int, expert_count + 1> offsets_{};
    std::array<int, expert_count> cursors_{};
    std::size_t assignment_count_ = 0;
};

} // namespace qwen
