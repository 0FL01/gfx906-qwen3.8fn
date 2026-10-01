#include "qsa.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qwen {
namespace {

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(message);
}

}

// WRITE OURS: semantic reference (no Transformers implementation copied):
// Qwen4ExpTextQSAIndexer.forward, The Qwen Team / HuggingFace, Apache-2.0,
// transformers revision a005fc82babfe8871d87746decad2dbee100a125:
// https://github.com/huggingface/transformers/blob/a005fc82babfe8871d87746decad2dbee100a125/src/transformers/models/qwen4_exp/modular_qwen4_exp.py
// Largest-first/sorted defaults and unspecified tied IDs:
// https://docs.pytorch.org/docs/2.14/generated/torch.topk.html
// mx dcd685463d597d31f5ca759d32c94592a2740fa4 is a diagnostic baseline,
// not the block/count oracle. Its expanded-position path exists only in tests.
QsaSelectionCounts qsa_select(std::span<const float> completed_block_scores,
                              std::size_t visible_tokens,
                              std::span<std::int32_t> token_ids,
                              std::span<std::int32_t> block_ids,
                              QsaConfig config) {
    const auto ratio = config.compress_ratio;
    if (ratio == 0 || config.token_budget == 0 || config.token_budget % ratio != 0)
        invalid("qsa: positive budget must be divisible by positive compress ratio");
    if (config.token_budget > std::numeric_limits<std::size_t>::max() - (ratio - 1))
        invalid("qsa: selection capacity overflow");
    // A count of INT32_MAX + 1 still has representable IDs [0, INT32_MAX].
    if (visible_tokens > std::size_t(std::numeric_limits<std::int32_t>::max()) + 1)
        invalid("qsa: visible token IDs exceed int32 range");

    const auto complete_blocks = visible_tokens / ratio;
    if (completed_block_scores.size() != complete_blocks)
        invalid("qsa: expected exactly the fully visible block scores");
    const auto selected_blocks = std::min(complete_blocks, config.token_budget / ratio);
    const auto tail_count = visible_tokens % ratio;
    // selected_blocks * ratio + tail_count <= visible_tokens, so no overflow.
    const auto selected_tokens = selected_blocks * ratio + tail_count;
    if (token_ids.size() < selected_tokens || block_ids.size() < selected_blocks)
        invalid("qsa: insufficient output capacity");
    if (selected_tokens != 0 && selected_blocks != 0) {
        const std::less<const std::int32_t*> before;
        if (before(token_ids.data(), block_ids.data() + selected_blocks) &&
            before(block_ids.data(), token_ids.data() + selected_tokens))
            invalid("qsa: overlapping output regions");
    }
    for (const float score : completed_block_scores)
        if (!std::isfinite(score)) invalid("qsa: nonfinite block score");

    std::vector<std::int32_t> order(complete_blocks);
    // The largest block ID is <= the largest token ID, including ratio == 1.
    // Do not use int32 iota: incrementing after INT32_MAX would overflow.
    for (std::size_t block = 0; block < complete_blocks; ++block)
        order[block] = static_cast<std::int32_t>(block);
    const auto better = [&](std::int32_t left, std::int32_t right) {
        const float a = completed_block_scores[static_cast<std::size_t>(left)];
        const float b = completed_block_scores[static_cast<std::size_t>(right)];
        return a > b || (a == b && left < right);
    };
    std::partial_sort(order.begin(), order.begin() + selected_blocks, order.end(), better);

    std::size_t output = 0;
    for (std::size_t rank = 0; rank < selected_blocks; ++rank) {
        const auto block = order[rank];
        block_ids[rank] = block;
        const auto start = static_cast<std::size_t>(block) * ratio;
        for (std::size_t member = 0; member < ratio; ++member)
            token_ids[output++] = static_cast<std::int32_t>(start + member);
    }
    const auto tail_start = complete_blocks * ratio;
    for (std::size_t member = 0; member < tail_count; ++member)
        token_ids[output++] = static_cast<std::int32_t>(tail_start + member);
    return {selected_tokens, selected_blocks};
}

}
