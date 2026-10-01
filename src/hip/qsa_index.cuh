#pragma once

#include "qsa_index.hpp"
#include "quant.hpp"
#include <hip/hip_runtime.h>

namespace qwen {

// Borrowed immutable device buffers: gamma[128], inverse frequencies[rotary/2].
struct QsaIndexDeviceParameters {
    QsaIndexConfig config;
    const float* query_norm;
    const float* key_norm;
    const float* inverse_frequencies;
};

// Append scratch capacities: packed[tokens*4], roundtrip[tokens*128],
// new_keys[ceil((start%4+tokens)/4)*128], staged_tail[384]. Optional tail
// prefixes[(tokens+1)*384], slot n after n consumed inputs, are speculative.
struct QsaIndexAppendWorkspace {
    Q4_0* packed;
    float* roundtrip;
    float* new_keys;
    float* staged_tail;
    float* tail_prefixes;
    int* error;
};

// Caller owns cache[capacity/4*128], tail[384], logical length and checkpoint
// validity. Success publishes ONLY newly completed keys and final tail; numeric
// failure leaves both active buffers unchanged. No allocations or hidden sync.
// Supports tokens=1..128, explicit stream, sticky flag 1 invalid / 2 arithmetic
// or unrepresentable Q4. Caller resets/checks before accepting the new length.
hipError_t launch_qsa_index_append(QsaIndexDeviceParameters parameters,
    const float* raw_keys, int start_consumed, int tokens, float* completed_keys,
    float* tail, QsaIndexAppendWorkspace workspace, hipStream_t stream) noexcept;

// One causal query, raw Q[512] -> normalized/RoPE scratch[512] -> scores
// [visible_tokens/4]. Only that completed prefix is read, even if future keys
// have been prepared. Query position is visible_tokens-1. Outputs provisional
// until the caller-owned sticky error flag is checked. No state publication.
hipError_t launch_qsa_index_score(QsaIndexDeviceParameters parameters,
    const float* raw_query, const float* completed_keys, int visible_tokens,
    float* normalized_query, float* scores, int* error, hipStream_t stream) noexcept;

} // namespace qwen
