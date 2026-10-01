#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace qwen {

// Pass the loaded model's indexer_budget/indexer_compress_ratio at integration.
// The starting GGUF uses 2048 tokens / 4 = 512 complete blocks; capacity 2051.
struct QsaConfig {
    std::size_t token_budget = 2048;
    std::size_t compress_ratio = 4;
};

struct QsaSelectionCounts {
    std::size_t token_count;
    std::size_t block_count;
};

// CPU correctness selector for ONE query with contiguous causal visibility
// [0, visible_tokens), including the query's own token. Not a padding/holey-mask
// frontend. completed_block_scores must contain EXACTLY visible_tokens / ratio
// finite scores, in chronological block order; no partial/future block scores.
// The reference score is sum_h ReLU(dot(q_h, pooled_key)) / sqrt(index_head_dim).
// Score calculation, raw-key pooling, quantization, norm and RoPE are upstream.
//
// Select min(budget / ratio, visible_tokens / ratio) whole blocks, ordered by
// descending score. PyTorch topk does not specify tied IDs: our explicit tie
// convention is lower block ID first (including +0/-0). Expand each selected
// block in ascending token order, then append ONLY the actual chronological
// tail. Finite negative scores are also rankable; no implicit ReLU is applied.
//
// Writes only token_ids[0:token_count] and block_ids[0:block_count], leaving all
// capacity beyond those counts untouched; it is NOT additional valid selection.
// Counts are actual counts, not budget + ratio - 1. Both spans must have enough
// room for this query and their written regions must not overlap. IDs are int32
// (nonnegative, zero-based). Empty visibility is allowed. Invalid config, ID
// range, score size/nonfinites, output capacity or overlap throws invalid_argument
// before either output is changed. This CPU oracle allocates temporary indices.
QsaSelectionCounts qsa_select(std::span<const float> completed_block_scores,
                              std::size_t visible_tokens,
                              std::span<std::int32_t> token_ids,
                              std::span<std::int32_t> block_ids,
                              QsaConfig config = {});

}
