#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace qwen {

// Scratch is constructor/caller-owned, reused in stream order. Candidate keys
// encode numeric descending score then lower block ID. At most 512 candidates.
// histogram: 32*256 ints; state: 16 uint64 words; candidates: 512 uint64 words.
struct QsaSelectWorkspace {
    int* histogram;
    std::uint64_t* state;
    std::uint64_t* candidates;
    int* error;
};

// scores[visible/4]; block_ids[min(512,visible/4)]; token_ids[2051]; counts[2]
// is {actual tokens, actual blocks}. Whole blocks in descending score order,
// lower-ID ties, ascending members then actual tail; unused capacity untouched.
// visible=1..131072. Numeric nonfinite scores set sticky flag 1, internal numeric
// failure flag 2. Caller resets/checks flag before consuming IDs/counts.
// Explicit stream, borrowed buffers, no allocation, query or synchronization.
hipError_t launch_qsa_select(const float* scores, int visible_tokens,
    std::int32_t* token_ids, std::int32_t* block_ids, int* counts,
    QsaSelectWorkspace workspace, hipStream_t stream) noexcept;

} // namespace qwen
