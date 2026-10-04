#pragma once

#include "attention.hpp"
#include <hip/hip_runtime.h>

namespace qwen {

inline constexpr int attention_error_invalid = 1;
inline constexpr int attention_error_arithmetic = 2;

struct AttentionQ4DeviceWorkspace {
    std::uint16_t* gathered_keys;   // 1,050,112 half bits [selected][2][256]
    std::uint16_t* gathered_values; // 1,050,112 half bits [selected][2][256]
    float* partial_output;         // 202,752 [chunk][24][256]; first 4096 words are temporary IDs before gather
    float2* partial_max_sum;       // 792 [chunk][24], {max, sum(exp(score-max))}
    float* staged_output;          // 6,144 [24][256]
    int* error;                    // one caller-initialized sticky device int
};

// One query, borrowed device buffers on the EXPLICIT stream; no allocation,
// synchronization, copy, device/pointer query or hidden pool. Required element
// capacities: Q/output=6144 floats, K/V=capacity*16 Q4_0 each, IDs=2051 int32,
// counts=2 ints {actual tokens, actual complete blocks}, workspace as above.
// capacity=1..131072, visible_tokens=1..capacity. Host address validation checks
// aligned nonwrapping ranges and rejects EVERY writable overlap, including all
// scratch/error; read-only ranges may overlap. Q requires 16-byte alignment,
// Q4/gathers 2, float buffers 4, float2 8, IDs/counts/error 4. Actual allocation
// sizes, residency, lifetimes, stream dependencies and exclusive ownership are
// caller obligations. Bad host arguments return hipErrorInvalidValue before
// enqueue. Otherwise returns the first HIP launch error; hipSuccess is enqueue,
// not completion. No exceptions.
//
// Device token count must be 1..2051. Complete-block count must be 0..512,
// <=visible_tokens/4 and <=tokens/4; it is selector metadata, not a constraint
// that all tokens must be whole blocks plus the current tail. Every selected ID
// must be in [0,visible_tokens). Repetitions are retained; the selected multiset
// is gathered chronologically for masked-attention reduction order. Caller IDs
// and counts remain unchanged. This can differ in FP32 rounding from the CPU
// oracle's supplied-order reduction, so parity uses the fixed bounded gate.
// Count/ID/query/all-selected-scale validation precedes ANY Q4 decode/cache
// gather. Invalid scale bits are never decoded. Only selected rows are gathered
// with FP16 RNE; no full-history FP16 conversion or mask. FP16 overflow is an
// arithmetic failure. Sticky atomic OR: 1 invalid count/ID/query/scale; 2 nonfinite
// arithmetic/FP16 gather overflow; 3 both. NEVER clears the flag. Error==0 alone
// authorizes final output publication in a separate stream-ordered kernel;
// failures leave output unchanged. Scratch is PROVISIONAL until completion and
// flag check. Caches are read-only. Caller must check flag after completion and
// discard the failed chain before resetting/reusing it. Publication guarantees
// concern detected semantic/numeric failures, not HIP faults/concurrent misuse.
//
// ADAPT furnace fattn_dec_chunk<256,12> / fattn_dec_combine<256>: 256 threads,
// four wave64, 64 selected keys/chunk, <=33 chunks, FP32 partials and softmax
// merge. Compile attention.hip with -ffp-contract=off, NO fast/finite-only math;
// Dot/value reductions are ascending within each 64-key chunk. Exponential
// evaluation widens internally to FP64 then rounds to FP32; all stored operands,
// probabilities, partials and output remain FP32. Split-K merge ordering still
// differs from a full serial CPU reduction; bit identity is not an API promise.
[[nodiscard]] hipError_t launch_attention_q4(const float* query,
    const Q4_0* keys, const Q4_0* values, int capacity,
    const std::int32_t* selected_ids, const int* counts, int visible_tokens,
    float* output, AttentionQ4DeviceWorkspace workspace, hipStream_t stream) noexcept;

inline constexpr int attention_max_batch = 128;
static_assert(sizeof(int) == sizeof(std::int32_t) && sizeof(int) == 4);
inline constexpr std::size_t attention_batch_workspace_bytes_per_query =
    2 * attention_gather_elements * sizeof(std::uint16_t) +
    attention_partial_elements * sizeof(float) +
    attention_partial_ml_elements * sizeof(float2) + attention_query_elements * sizeof(float);
static_assert(attention_batch_workspace_bytes_per_query == 5042368);
static_assert(attention_batch_workspace_bytes_per_query * 128 == 645423104);

// Closed, typed borrowed view. elements is the ENTIRE declared buffer capacity;
// stride is in T elements between query rows, including any caller padding.
template<class T> struct AttentionQ4BatchView {
    T* data;
    std::size_t elements;
    std::size_t stride;
};

struct AttentionQ4BatchCache {
    const Q4_0* keys;
    std::size_t key_elements;
    const Q4_0* values;
    std::size_t value_elements;
    int capacity;   // shared canonical [capacity][2][8], 1..131072
    int valid_rows; // loaded immutable prefix, 1..capacity
};

struct AttentionQ4BatchWorkspace {
    AttentionQ4BatchView<std::uint16_t> gathered_keys;   // row >=1050112
    AttentionQ4BatchView<std::uint16_t> gathered_values; // row >=1050112
    AttentionQ4BatchView<float> partial_output;         // row >=202752
    AttentionQ4BatchView<float2> partial_max_sum;        // row >=792
    AttentionQ4BatchView<float> staged_output;          // row >=6144
    int* error; // ONE shared sticky int, same bit classes as N1; never cleared
};

// B=1..128 independent QSA queries, NOT dense attention to 128K rows. Query q
// sees [0,first_visible+q); first_visible>=1 and last visibility<=valid_rows.
// Required row widths: Q/output=6144, IDs=2051 int32, counts=2 ints. All strides
// must be >=row width and <=elements even for B1, and preserve N1 alignment
// at every row.
// Full declared buffer ranges (including gaps/tails) must be nonwrapping,
// aligned and disjoint whenever either range is writable, including error.
// Read-only overlaps are permitted. Validation precedes ANY HIP call; bad host
// metadata returns hipErrorInvalidValue with all device buffers/flag unchanged.
// No allocation, pointer/device query, copy, synchronization or implicit stream.
// Residency, actual allocations, lifetimes and stream dependencies are borrowed.
//
// SAME N1 count/ID/query/scale preflight, repeated-ID semantics, chronological
// bitonic sort, Q4->FP32->FP16 RNE gather, chunk arithmetic, mx sum64 and merge.
// Each query owns a scalar-sized workspace row: 5042368 bytes, excluding padding
// and the ONE 4-byte flag. Private ordered IDs reuse its first 4096 partial words;
// no extra selection/count scratch. Inputs remain immutable until consumed.
// Six stream-ordered batched kernels: complete ALL-query input preflight, then
// ALL-query scale preflight, gather, chunks, merge, and publication. The global
// sticky flag is also every query's invalid marker: any failed query suppresses
// every subsequent stage/publication. Provisional scratch may change on failure.
// With no detected failure all B rows publish only after all merges complete;
// otherwise ALL original output rows remain untouched. This is failure-atomic
// publication, not cross-stream transactional visibility or HIP-fault recovery.
// Same strict -ffp-contract=off / no fast/finite-only math build contract as N1.
[[nodiscard]] hipError_t launch_attention_q4_batch(AttentionQ4BatchView<const float> query,
    AttentionQ4BatchCache cache, int batch, int first_visible,
    AttentionQ4BatchView<const std::int32_t> selected_ids,
    AttentionQ4BatchView<const int> counts, AttentionQ4BatchView<float> output,
    AttentionQ4BatchWorkspace workspace, hipStream_t stream) noexcept;

} // namespace qwen
