#pragma once

#include "quant.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace qwen {

inline constexpr int mtp_attention_max_batch = 3;
inline constexpr int mtp_attention_max_capacity = 131072;
inline constexpr int mtp_attention_chunk_keys = 64;
inline constexpr int mtp_attention_max_chunks = 2048;
inline constexpr std::size_t mtp_attention_query_elements = 24 * 256;
inline constexpr int mtp_attention_error_source = 1;
inline constexpr int mtp_attention_error_arithmetic = 2;

// All counts/strides are ELEMENTS, including padding and the entire declared
// tail. There is no ID/count/selector ABI: every row [0,first_visible+q) is used.
template<class T> struct MtpAttentionQ4View {
    T* data;
    std::size_t elements;
    std::size_t stride;
};
struct MtpAttentionQ4Cache {
    const Q4_0* keys;
    std::size_t key_elements;
    const Q4_0* values;
    std::size_t value_elements;
    int capacity;   // canonical [capacity][2][8], 1..131072
    int valid_rows; // immutable loaded prefix, 1..capacity
};
struct MtpAttentionQ4Workspace {
    MtpAttentionQ4View<float> partial_output;   // [query][ceil(capacity/64)][24][256]
    MtpAttentionQ4View<float2> partial_max_sum;  // [query][ceil(capacity/64)][24]
    MtpAttentionQ4View<float> staged_output;    // [query][24][256]
    int* error; // one caller-initialized sticky int, NEVER cleared by this API
};
struct MtpAttentionQ4Layout {
    std::size_t chunks, cache_elements, partial_elements, max_sum_elements;
    std::size_t workspace_bytes_per_query;
};
// A pure host layout calculation; invalid capacity returns an all-zero layout.
[[nodiscard]] constexpr MtpAttentionQ4Layout mtp_attention_q4_layout(int capacity) noexcept {
    if (capacity < 1 || capacity > mtp_attention_max_capacity) return {};
    const auto chunks = (static_cast<std::size_t>(capacity) + 63) / 64;
    return {chunks, static_cast<std::size_t>(capacity) * 16,
            chunks * mtp_attention_query_elements, chunks * 24,
            chunks * mtp_attention_query_elements * sizeof(float) +
            chunks * 24 * sizeof(float2) + mtp_attention_query_elements * sizeof(float)};
}
static_assert(sizeof(Q4_0) == 18 && sizeof(float2) == 8 && sizeof(int) == 4);
static_assert(mtp_attention_q4_layout(131072).workspace_bytes_per_query == 50749440);
static_assert(3 * mtp_attention_q4_layout(131072).workspace_bytes_per_query + sizeof(int) == 152248324);

namespace mtp_attention_detail {
struct Range { std::uintptr_t begin, end; bool writable, valid; };
template<class T> inline Range range(const T* data, std::size_t elements, bool writable = false,
                                    std::size_t alignment = alignof(T)) noexcept {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (!data || elements == 0 || begin % alignment != 0 ||
        elements > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T) ||
        elements > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(T))
        return {0, 0, writable, false};
    return {begin, begin + elements * sizeof(T), writable, true};
}
template<class T> inline Range view_range(MtpAttentionQ4View<T> v, int batch, std::size_t width,
                                        bool writable = false, std::size_t alignment = alignof(T)) noexcept {
    // Subtractions/divisions before products: even adversarial SIZE_MAX strides
    // cannot wrap. Alignment of every query row is checked even for B1.
    if (v.stride < width || v.stride > v.elements || v.elements < width ||
        v.stride % (alignment / alignof(T)) != 0 ||
        (batch > 1 && v.stride > (v.elements - width) / static_cast<std::size_t>(batch - 1)))
        return {0, 0, writable, false};
    return range(v.data, v.elements, writable, alignment);
}
// Pure host preflight, shared verbatim with the CPU-only metadata fixture.
// Actual allocation/residency/lifetime/exclusive access are caller obligations.
[[nodiscard]] inline bool valid_arguments(MtpAttentionQ4View<const float> query,
    MtpAttentionQ4Cache cache, int batch, int first_visible, MtpAttentionQ4View<float> output,
    MtpAttentionQ4Workspace w) noexcept {
    if (batch < 1 || batch > mtp_attention_max_batch || cache.capacity < 1 ||
        cache.capacity > mtp_attention_max_capacity || cache.valid_rows < 1 ||
        cache.valid_rows > cache.capacity || first_visible < 1 || first_visible > cache.valid_rows ||
        batch - 1 > cache.valid_rows - first_visible) return false;
    const auto l = mtp_attention_q4_layout(cache.capacity);
    if (cache.key_elements < l.cache_elements || cache.value_elements < l.cache_elements) return false;
    const std::array buffers{
        view_range(query, batch, mtp_attention_query_elements, false, 16),
        range(cache.keys, cache.key_elements), range(cache.values, cache.value_elements),
        view_range(output, batch, mtp_attention_query_elements, true),
        view_range(w.partial_output, batch, l.partial_elements, true),
        view_range(w.partial_max_sum, batch, l.max_sum_elements, true),
        view_range(w.staged_output, batch, mtp_attention_query_elements, true), range(w.error, 1, true)};
    for (const auto& r : buffers) if (!r.valid) return false;
    for (std::size_t i = 0; i < buffers.size(); ++i)
        for (std::size_t j = i + 1; j < buffers.size(); ++j) {
            const auto& a = buffers[i]; const auto& b = buffers[j];
            if ((a.writable || b.writable) && a.begin < b.end && b.begin < a.end) return false;
        }
    return true;
}
} // namespace mtp_attention_detail

// Specialized dense causal blk48/compress0 primitive: B1..3 contiguous queries,
// Q24/KV2/GQA12/D256, unchanged unscaled Q/RMS/RoPE producers outside this API.
// Query q sees exactly positions 0..first_visible+q-1, WITHOUT the QSA2051 budget.
// No gather/history FP16 copy. EACH K/V operand is Q4->FP32->FP16 RNE->FP32,
// including signed/tiny scales. Ascending dot256/value64, old .0625 coefficient,
// existing widened chunk exp / expf merge, mx sum64, ascending chunk merge.
// ADAPT current furnace-derived attention.hip; donor details in mtp_attention.hip.
// Compile HIP20/gfx906 with -ffp-contract=off, NO fast/finite-only math.
//
// Workspace widths derive from actual CACHE capacity (not visibility): layout()
// describes unpadded private rows. At128K/B3: 152248324 bytes INCLUDING the flag,
// excluding caller padding/Q/output/cache. Alignment: Q16, Q4_0=2, float/int4,
// float2=8. Every full declared range, padding/tail, and flag is validated for
// alignment, PTRDIFF bounds, wrap and writable overlap BEFORE ANY HIP call.
// Invalid host metadata returns hipErrorInvalidValue; ALL buffers/flag unchanged.
// Read-only overlaps allowed. No allocation/copy/sync/pointer or device query,
// implicit stream or exception. Supplied stream is explicit (must be nonnull).
// Return is the first launch error, hipSuccess means enqueue, not completion.
//
// Three stream-ordered ALL-query preflights: finite Q/visibility, finite scales
// throughout EACH full causal prefix, then finite decoded FP16 operands (finite
// scale can still decode to +/-524032 and overflow half). Future loaded rows
// beyond each query visibility are never inspected. Flag atomic OR: bit1 invalid
// source, bit2 nonfinite arithmetic/half overflow; never cleared. Any query's
// failure suppresses ALL B publication. Scratch is provisional after arithmetic
// begins; source/decode preflight failures leave all scratch/output untouched.
// Only after all chunks AND all merges finish does a separate kernel publish
// if flag==0. Caller checks flag after completion and resets explicitly after
// discarding a failed chain. This guarantee excludes HIP faults/concurrent misuse.
[[nodiscard]] hipError_t launch_mtp_attention_q4(MtpAttentionQ4View<const float> query,
    MtpAttentionQ4Cache cache, int batch, int first_visible, MtpAttentionQ4View<float> output,
    MtpAttentionQ4Workspace workspace, hipStream_t stream) noexcept;

} // namespace qwen
