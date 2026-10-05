#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>

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


// Bounded chronological query batch, first_visible+q, q=0..queries-1.
// Strides/elements are in units of T. Every declared view is disjoint from
// every writable view, including padding. Output widths: tokens2051,
// blocks512, counts2. Score width is max(1,(first_visible+queries-1)/4);
// only each row's actually completed block prefix is read.
template<class T> struct QsaSelectView {
    T* data = nullptr;
    std::size_t elements = 0;
    std::size_t stride = 0;
};
struct QsaSelectBatchWorkspace {
    QsaSelectView<int> histogram;             // 8192 integers per query
    QsaSelectView<std::uint64_t> state;       // 16 words per query
    QsaSelectView<std::uint64_t> candidates;  // 512 keys per query
    int* error = nullptr;                    // shared caller-owned sticky flag
};
// queries1..8, visibility1..131072. No allocation/query/synchronization.
// Host rejection enqueues nothing. Device failures leave ALL output IDs/counts
// unchanged: scratch sorting/validation completes before a separate publication.
hipError_t launch_qsa_select_batch(QsaSelectView<const float> scores,
    int queries, int first_visible, QsaSelectView<std::int32_t> token_ids,
    QsaSelectView<std::int32_t> block_ids, QsaSelectView<int> counts,
    QsaSelectBatchWorkspace workspace, hipStream_t stream) noexcept;

} // namespace qwen
