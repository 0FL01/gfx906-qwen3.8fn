#pragma once

#include "quant.hpp"
#include <cstddef>
#include <cstdint>
#include <span>

namespace qwen {

inline constexpr int attention_head_dim = 256;
inline constexpr int attention_query_heads = 24;
inline constexpr int attention_kv_heads = 2;
inline constexpr int attention_gqa = 12;
inline constexpr int attention_q4_blocks_per_head = 8;
inline constexpr int attention_max_selected = 2051;
inline constexpr int attention_max_capacity = 131072;
inline constexpr int attention_chunk_keys = 64;
inline constexpr int attention_max_chunks = 33;
inline constexpr std::size_t attention_query_elements = 6144;
inline constexpr std::size_t attention_gather_elements = 2051 * 2 * 256;
inline constexpr std::size_t attention_partial_elements = 33 * 24 * 256;
inline constexpr std::size_t attention_partial_ml_elements = 33 * 24;
inline constexpr float attention_query_coefficient = 0.0625f; // 1/sqrt(256)

enum class AttentionQ4Mode {
    gathered_fp16, // GPU contract: Q4 decode -> FP16 RNE -> FP32 arithmetic
    direct_fp32   // diagnostic: Q4 decode directly into FP32, no half roundtrip
};

struct AttentionQ4Cache {
    // EXACT capacity*2*8 blocks each, canonical [token][KV head][8 Q4_0].
    std::span<const Q4_0> keys;
    std::span<const Q4_0> values;
    std::size_t capacity;
};

struct AttentionQ4Scratch {
    std::span<float> scores; // EXACT selected_ids.size() FP32 scores/weights
    std::span<float> output; // EXACT 6144 staged FP32 output values
};

// Independent scalar CPU oracle, ONE query [24][256], result [24][256]. Head h
// uses KV head h/12. IDs are int32 token positions in [0,visible_tokens), in the
// supplied order; arbitrary counts 1..2051 and repeated IDs are supported (a
// repeated ID is a repeated softmax entry). visible_tokens=1..capacity<=131072.
// No implied selection budget, mask, Hadamard, RoPE, gate or projection here.
//
// Exact shapes, mode, aligned/nonwrapping ranges and every writable overlap are
// checked before dereference. Read-only ranges may overlap. All IDs and query
// values, then ALL selected scale bit patterns are validated BEFORE any scale
// decode; unused cache rows are never read. gathered_fp16 additionally rejects
// overflow in any selected K/V FP16 roundtrip, even a zero-weight V operand.
// invalid_argument leaves output AND scratch untouched. runtime_error indicates
// nonfinite FP32 arithmetic; scratch is provisional, output is still untouched.
// Caches/query/IDs are always read-only. No successful call allocates memory.
// Compile with -ffp-contract=off and without fast/finite-only math. Products,
// ascending dots, ascending softmax sums and ascending weighted-value sums are
// FP32; scores=dot(Q,K)/16, output=numerator*(1/sum(exp(score-max))).
void attention_q4(AttentionQ4Cache cache, std::span<const float> query,
                  std::span<const std::int32_t> selected_ids,
                  std::size_t visible_tokens, AttentionQ4Mode mode,
                  AttentionQ4Scratch scratch, std::span<float> output);

} // namespace qwen
