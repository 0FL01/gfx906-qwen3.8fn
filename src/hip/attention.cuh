#pragma once

#include "attention.hpp"
#include <hip/hip_runtime.h>

namespace qwen {

inline constexpr int attention_error_invalid = 1;
inline constexpr int attention_error_arithmetic = 2;

struct AttentionQ4DeviceWorkspace {
    std::uint16_t* gathered_keys;   // 1,050,112 half bits [selected][2][256]
    std::uint16_t* gathered_values; // 1,050,112 half bits [selected][2][256]
    float* partial_output;         // 202,752 [chunk][24][256]
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
// must be in [0,visible_tokens). Order/repetition semantics match the CPU oracle.
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
// normal expf. GPU reduction ordering differs from ascending CPU, not bit exact.
[[nodiscard]] hipError_t launch_attention_q4(const float* query,
    const Q4_0* keys, const Q4_0* values, int capacity,
    const std::int32_t* selected_ids, const int* counts, int visible_tokens,
    float* output, AttentionQ4DeviceWorkspace workspace, hipStream_t stream) noexcept;

} // namespace qwen
