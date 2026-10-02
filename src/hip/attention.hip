#include "attention.cuh"
#include "dpp.cuh"

#include <hip/hip_fp16.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "attention.hip is specialized for gfx906 wave64"
#endif

// Donor-first: mx dcd685463d597d31f5ca759d32c94592a2740fa4,
// ggml/src/ggml-cuda/common.cuh::ggml_gfx906_warp_reduce_sum_f32: KEEP the
// project's extracted dpp.cuh::sum64 (including hazard waits). mx's expanded
// position mask is a diagnostic baseline, NOT our selected-ID specification.
// ADAPT furnace 905021dbad71c5056ef51f9fd45d545403fc989c,
// ggml/src/ggml-cuda/fattn.cu::fattn_dec_chunk<256,12> / fattn_dec_combine<256>.
// Entire related gather/dispatch/chunk/merge section, lines 744..1154, read from
// runs/donors/furnace-fattn.cu before port. Retain: 256 threads/four wave64,
// lane=key, LDS float4 Q reuse across 12 GQA heads, FP32 chunk-local
// max/sum/unnormalized output, 16 V loads in flight per dim, then numerically
// stable split-K merge. WRITE OURS: direct selected-ID/count
// frontend, canonical unchanged Q4_0 cache, bounded selected-only RNE gather,
// all-input preflight, borrowed ranges, sticky errors, conditional publication.
// Changes: one query Q24/KV2/D256 only; 64 keys and <=33 chunks; no masks,
// ggml stride/dispatch/pool/runtime, environment switches, full FP16 history,
// allocations or synchronization. Packed half operands are loaded as their
// declared uint16 type (no alignment/type-punning assumptions on gather buffers).
// PORT max wave64 shuffle fallback from mx common.cuh::warp_reduce_max;
// max/merge reductions use ascending XOR distances, normal expf and contraction OFF.
// Checked reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8,
// kernels/attn_partial_q8.cpp / attn_merge.cpp: KEEP split-K algebra reference;
// do NOT port int8 Q/KV quantization or fast __expf (different precision/ABI).
// Frozen parity gate belongs to the fixture: common-gather output error
// <=2e-4 + 2e-4*abs(reference); no tolerance is encoded/relaxed here.
// Disposition: ADAPT arithmetic-control candidate; qualification pending real
// HIP build, exact-same-input attention witness and R2d core-qsa before Session;
// A/B timing only if accepted. Ascending grouping is intentional: each wave
// owns three heads and completes each D256 dot before applying 0.0625f; chunk
// denominator and weighted-V additions follow selected-member order, with
// separately rounded FP32 products/additions. src/attention.cpp normalizes AFTER
// the unnormalized weighted-V sum, not each probability before its V product.
// Keep that partial-output convention for every count and the stable split-K
// merge; a single chunk reduces to numerator * (1 / ascending denominator).
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace qwen {
namespace {

constexpr int threads = 256;
constexpr int D = attention_head_dim;
constexpr int G = attention_gqa;
constexpr int C = attention_chunk_keys;
constexpr int row_values = attention_kv_heads * D;
constexpr int row_blocks = attention_kv_heads * attention_q4_blocks_per_head;
static_assert(D == 256 && G == 12 && C == 64 && attention_query_heads == 24);
static_assert(attention_max_chunks == (attention_max_selected + C - 1) / C);
static_assert(sizeof(int) == 4 && sizeof(float2) == 8 && sizeof(Q4_0) == 18);

struct Range { std::uintptr_t begin, end; bool writable, valid; };

template<class T>
Range range(const T* pointer, std::size_t count, bool writable = false,
            std::size_t alignment = alignof(T)) noexcept {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (!pointer || count == 0 || begin % alignment != 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(T))
        return {0, 0, writable, false};
    return {begin, begin + count * sizeof(T), writable, true};
}

template<std::size_t N>
bool valid_ranges(const std::array<Range, N>& buffers) noexcept {
    for (const auto& r : buffers) if (!r.valid) return false;
    for (std::size_t i = 0; i < N; ++i) for (std::size_t j = i + 1; j < N; ++j) {
        const auto& a = buffers[i]; const auto& b = buffers[j];
        if ((a.writable || b.writable) && a.begin < b.end && b.begin < a.end) return false;
    }
    return true;
}

__device__ __forceinline__ void report(int flags, int* error) {
    if (flags) atomicOr(error, flags);
}

__device__ __forceinline__ bool valid_counts(const int* counts, int visible) {
    // Signed comparisons BEFORE multiplication, loop bounds or ID/cache loads.
    const int n = counts[0], b = counts[1];
    return n > 0 && n <= attention_max_selected && b >= 0 && b <= 512 &&
           b <= visible / 4 && b <= n / 4;
}

__device__ __forceinline__ bool ready(const int* counts, int visible, int* error) {
    if (!valid_counts(counts, visible)) { report(attention_error_invalid, error); return false; }
    return __atomic_load_n(error, __ATOMIC_RELAXED) == 0;
}

__device__ __forceinline__ float checked(float value, int& flags) {
    if (!isfinite(value)) { flags |= attention_error_arithmetic; return 0.0f; }
    return value;
}

__device__ __forceinline__ float half_value(std::uint16_t value) {
    return __half2float(__ushort_as_half(value));
}

__device__ __forceinline__ float max64(float value) {
#pragma unroll
    for (int mask = 1; mask < 64; mask *= 2) value = fmaxf(value, __shfl_xor(value, mask, 64));
    return value;
}

// No Q4/cache load here. Complete validation before any cache row is touched.
// ADAPT mx argsort.cu's bounded LDS bitonic network: chronological selected-ID
// order reproduces masked attention's cache traversal without changing the
// selected multiset, its ranked public IDs/counts, or repeated-ID semantics.
// Reuse the first 4096 words of partial-output scratch only until gather ends;
// the later partial-output kernel overwrites them in stream order.
__global__ __launch_bounds__(threads)
void validate_inputs(const float* query, const std::int32_t* ids, const int* counts,
                     int visible, std::int32_t* ordered_ids, int* error) {
    __shared__ int active;
    __shared__ std::int32_t ordered[4096];
    if (threadIdx.x == 0) active = ready(counts, visible, error);
    __syncthreads();
    if (!active) return;
    const int n = counts[0];
    int flags = 0;
    for (int j = threadIdx.x; j < n; j += threads)
        if (ids[j] < 0 || ids[j] >= visible) flags |= attention_error_invalid;
    for (int i = threadIdx.x; i < static_cast<int>(attention_query_elements); i += threads)
        if (!isfinite(query[i])) flags |= attention_error_invalid;
    report(flags, error);
    __syncthreads();
    if (threadIdx.x == 0) active = __atomic_load_n(error, __ATOMIC_RELAXED) == 0;
    __syncthreads();
    if (!active) return;
    int width = 1;
    while (width < n) width *= 2;
    for (int i = threadIdx.x; i < width; i += threads)
        ordered[i] = i < n ? ids[i] : INT32_MAX;
    __syncthreads();
    for (int span = 2; span <= width; span *= 2) {
        for (int stride = span / 2; stride > 0; stride /= 2) {
            for (int i = threadIdx.x; i < width; i += threads) {
                const int other = i ^ stride;
                if (i < other) {
                    const auto a = ordered[i], b = ordered[other];
                    if (((i & span) == 0 && a > b) || ((i & span) != 0 && a < b)) {
                        ordered[i] = b; ordered[other] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < n; i += threads) ordered_ids[i] = ordered[i];
}

// Check bits, not half arithmetic. A late invalid V scale prevents ALL scale
// decode/gather in the following kernel, including otherwise valid early rows.
__global__ __launch_bounds__(threads)
void validate_scales(const Q4_0* keys, const Q4_0* values, const std::int32_t* ids,
                     const int* counts, int visible, int* error) {
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(counts, visible, error);
    __syncthreads();
    if (!active) return;
    const int n = counts[0];
    int flags = 0;
    for (int i = threadIdx.x; i < n * row_blocks; i += threads) {
        const int id = ids[i / row_blocks];
        if (id < 0 || id >= visible) { flags |= attention_error_invalid; continue; }
        const std::size_t b = static_cast<std::size_t>(id) * row_blocks + i % row_blocks;
        if ((keys[b].d & 0x7c00) == 0x7c00 || (values[b].d & 0x7c00) == 0x7c00)
            flags |= attention_error_invalid;
    }
    report(flags, error);
}

__device__ __forceinline__ std::uint16_t decode_half(const Q4_0& block, int i, int& flags) {
    // Defensive check also enforces no invalid scale decode if caller violates
    // the required immutable-input lifetime between validation and gather.
    if ((block.d & 0x7c00) == 0x7c00) { flags |= attention_error_invalid; return 0; }
    const int code = i < 16 ? block.qs[i] & 15 : block.qs[i - 16] >> 4;
    const float value = half_value(block.d) * static_cast<float>(code - 8);
    const auto bits = __half_as_ushort(__float2half_rn(value));
    if ((bits & 0x7c00) == 0x7c00) { flags |= attention_error_arithmetic; return 0; }
    return bits;
}

// One wave per selected token, eight dims/lane across both KV heads. Only the
// actual n rows are touched; unused 2051 capacity remains untouched.
__global__ __launch_bounds__(threads)
void gather_selected(const Q4_0* keys, const Q4_0* values, const std::int32_t* ids,
                     const int* counts, int visible, AttentionQ4DeviceWorkspace w) {
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(counts, visible, w.error);
    __syncthreads();
    if (!active) return;
    const int j = blockIdx.x * 4 + threadIdx.x / 64;
    if (j >= counts[0]) return;
    const int id = ids[j];
    if (id < 0 || id >= visible) { report(attention_error_invalid, w.error); return; }
    const std::size_t base = static_cast<std::size_t>(id) * row_blocks;
    int flags = 0;
    for (int d = threadIdx.x % 64; d < row_values; d += 64) {
        const std::size_t b = base + d / 32;
        const std::size_t target = static_cast<std::size_t>(j) * row_values + d;
        w.gathered_keys[target] = decode_half(keys[b], d % 32, flags);
        w.gathered_values[target] = decode_half(values[b], d % 32, flags);
    }
    report(flags, w.error);
}

// ADAPT fattn_dec_chunk<D=256,G=12>. The grid is fixed at 33 chunks x 2 KV
// heads, but chunks outside actual count exit before ANY operand/scratch access.
__global__ __launch_bounds__(threads)
void fattn_dec_chunk(const float* query, const int* counts, int visible,
                     AttentionQ4DeviceWorkspace w) {
    constexpr int H = G / 4;
    __shared__ float4 qs[G][D / 4];
    __shared__ float ps[G][C];
    __shared__ float maxima[G];
    __shared__ int cells[C];
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(counts, visible, w.error);
    __syncthreads();
    if (!active) return;
    const int n = counts[0];
    const int chunk = blockIdx.x, g = blockIdx.y;
    if (chunk >= (n + C - 1) / C) return;
    const int chunk_count = n - chunk * C < C ? n - chunk * C : C;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    int flags = 0;
    for (int i = tid; i < G * D / 4; i += threads) {
        const int h = i / (D / 4), d4 = i % (D / 4);
        qs[h][d4] = reinterpret_cast<const float4*>(query + (g * G + h) * D)[d4];
    }
    if (tid < C) {
        const int j = chunk * C + tid;
        cells[tid] = j < n ? j : -1; // Positions in the validated bounded gather.
    }
    __syncthreads();

    // Lane=key, each wave owns h=wave+4*i for i=0..2. Eight K components in
    // flight are reused across those three heads, but EVERY product is added to
    // the full dot separately in ascending d order; no eight-term/quarter tree.
    {
        const int cell = cells[lane];
        float s[H];
#pragma unroll
        for (int h = 0; h < H; ++h) s[h] = 0.0f;
        const auto* kr = w.gathered_keys + static_cast<std::size_t>(cell >= 0 ? cell : 0) * row_values + g * D;
#pragma unroll
        for (int c = 0; c < D / 8; ++c) {
            const float k0 = cell >= 0 ? half_value(kr[8 * c + 0]) : 0.0f;
            const float k1 = cell >= 0 ? half_value(kr[8 * c + 1]) : 0.0f;
            const float k2 = cell >= 0 ? half_value(kr[8 * c + 2]) : 0.0f;
            const float k3 = cell >= 0 ? half_value(kr[8 * c + 3]) : 0.0f;
            const float k4 = cell >= 0 ? half_value(kr[8 * c + 4]) : 0.0f;
            const float k5 = cell >= 0 ? half_value(kr[8 * c + 5]) : 0.0f;
            const float k6 = cell >= 0 ? half_value(kr[8 * c + 6]) : 0.0f;
            const float k7 = cell >= 0 ? half_value(kr[8 * c + 7]) : 0.0f;
#pragma unroll
            for (int h = 0; h < H; ++h) {
                const float4 qa = qs[wave + 4 * h][2 * c];
                const float4 qb = qs[wave + 4 * h][2 * c + 1];
                s[h] = checked(s[h] + checked(qa.x * k0, flags), flags);
                s[h] = checked(s[h] + checked(qa.y * k1, flags), flags);
                s[h] = checked(s[h] + checked(qa.z * k2, flags), flags);
                s[h] = checked(s[h] + checked(qa.w * k3, flags), flags);
                s[h] = checked(s[h] + checked(qb.x * k4, flags), flags);
                s[h] = checked(s[h] + checked(qb.y * k5, flags), flags);
                s[h] = checked(s[h] + checked(qb.z * k6, flags), flags);
                s[h] = checked(s[h] + checked(qb.w * k7, flags), flags);
            }
        }
#pragma unroll
        for (int i = 0; i < H; ++i) {
            const int h = wave + 4 * i;
            const float score = cell >= 0 ? checked(s[i] * attention_query_coefficient, flags) : -INFINITY;
            const float m = max64(score);
            // Accuracy experiment: round the transcendental result back to FP32.
            // One-ULP output differences remain on the same-input witness;
            // widened evaluation is explicit, not an unchanged-precision speedup.
            const float difference = cell >= 0 ? checked(score - m, flags) : 0.0f;
            ps[h][lane] = cell >= 0 ? checked(static_cast<float>(exp(static_cast<double>(difference))), flags) : 0.0f;
            if (lane == 0) maxima[h] = m;
        }
    }
    __syncthreads();

    // LDS weights are complete before leaders sum ONLY actual members, in
    // ascending selected-position order. Keep exp weights unnormalized, matching
    // the CPU numerator-then-inverse equations and the existing split-K ABI.
    if (lane == 0) for (int h = wave; h < G; h += 4) {
        float l = 0.0f;
        for (int j = 0; j < chunk_count; ++j) l = checked(l + ps[h][j], flags);
        if (!(l > 0.0f)) flags |= attention_error_arithmetic;
        w.partial_max_sum[(chunk * attention_kv_heads + g) * G + h] = make_float2(maxima[h], l);
    }

    // Thread=dim, sixteen V loads in flight, all twelve heads share those loads.
    float o[G];
#pragma unroll
    for (int h = 0; h < G; ++h) o[h] = 0.0f;
    for (int j0 = 0; j0 < chunk_count; j0 += 16) {
        float v[16];
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int cell = cells[j0 + jj];
            v[jj] = cell >= 0 ? half_value(w.gathered_values[static_cast<std::size_t>(cell) * row_values + g * D + tid]) : 0.0f;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
#pragma unroll
            for (int jj = 0; jj < 16; ++jj)
                if (j0 + jj < chunk_count)
                    o[h] = checked(o[h] + checked(ps[h][j0 + jj] * v[jj], flags), flags);
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h)
        w.partial_output[((chunk * attention_kv_heads + g) * G + h) * D + tid] = o[h];
    report(flags, w.error);
}

// ADAPT fattn_dec_combine<256>. Stream order guarantees every chunk has
// finished reporting errors before any combine group reads its first flag.
__global__ __launch_bounds__(threads)
void fattn_dec_combine(const int* counts, int visible, AttentionQ4DeviceWorkspace w) {
    __shared__ float factor[attention_max_chunks];
    __shared__ float red[4];
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(counts, visible, w.error);
    __syncthreads();
    if (!active) return;
    const int n_chunks = (counts[0] + C - 1) / C;
    const int hq = blockIdx.x, tid = threadIdx.x;
    int flags = 0;
    float2 ml = make_float2(-INFINITY, 0.0f);
    if (tid < n_chunks) {
        ml = w.partial_max_sum[tid * attention_query_heads + hq];
        if (!isfinite(ml.x) || !isfinite(ml.y) || !(ml.y > 0.0f)) {
            flags |= attention_error_arithmetic;
            ml = make_float2(-INFINITY, 0.0f);
        }
    }
    float m = max64(ml.y > 0.0f ? ml.x : -INFINITY);
    if (tid % 64 == 0) red[tid / 64] = m;
    __syncthreads();
    const float m_all = checked(fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])), flags);
    __syncthreads();
    const float f = ml.y > 0.0f ? checked(expf(checked(ml.x - m_all, flags)), flags) : 0.0f;
    if (tid < n_chunks) factor[tid] = f;
    const float l = sum64(checked(ml.y * f, flags));
    if (tid % 64 == 0) red[tid / 64] = l;
    __syncthreads();
    const float l_all = checked((red[0] + red[1]) + (red[2] + red[3]), flags);
    if (!(l_all > 0.0f)) flags |= attention_error_arithmetic;
    const float inv_l = l_all > 0.0f ? checked(1.0f / l_all, flags) : 0.0f;
    float acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c)
        acc = checked(acc + checked(w.partial_output[(c * attention_query_heads + hq) * D + tid] * factor[c], flags), flags);
    w.staged_output[hq * D + tid] = checked(acc * inv_l, flags);
    report(flags, w.error);
}

// No error-producing kernel can overlap this publication in the supplied stream.
__global__ __launch_bounds__(threads)
void publish_output(const float* staged, float* output, const int* error) {
    if (__atomic_load_n(error, __ATOMIC_RELAXED) != 0) return;
    const int i = blockIdx.x * threads + threadIdx.x;
    if (i < static_cast<int>(attention_query_elements)) output[i] = staged[i];
}

} // namespace

hipError_t launch_attention_q4(const float* query, const Q4_0* keys, const Q4_0* values,
    int capacity, const std::int32_t* ids, const int* counts, int visible,
    float* output, AttentionQ4DeviceWorkspace w, hipStream_t stream) noexcept {
    if (capacity <= 0 || capacity > attention_max_capacity || visible <= 0 || visible > capacity)
        return hipErrorInvalidValue;
    const auto blocks = static_cast<std::size_t>(capacity) * row_blocks;
    if (!valid_ranges(std::array{
            range(query, attention_query_elements, false, 16), range(keys, blocks), range(values, blocks),
            range(ids, attention_max_selected), range(counts, 2), range(output, attention_query_elements, true),
            range(w.gathered_keys, attention_gather_elements, true), range(w.gathered_values, attention_gather_elements, true),
            range(w.partial_output, attention_partial_elements, true), range(w.partial_max_sum, attention_partial_ml_elements, true),
            range(w.staged_output, attention_query_elements, true), range(w.error, 1, true)}))
        return hipErrorInvalidValue;
    auto* ordered_ids = reinterpret_cast<std::int32_t*>(w.partial_output);
    validate_inputs<<<1, threads, 0, stream>>>(query, ids, counts, visible, ordered_ids, w.error);
    auto status = hipGetLastError(); if (status != hipSuccess) return status;
    validate_scales<<<1, threads, 0, stream>>>(keys, values, ids, counts, visible, w.error);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    gather_selected<<<(attention_max_selected + 3) / 4, threads, 0, stream>>>(keys, values, ordered_ids, counts, visible, w);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    fattn_dec_chunk<<<dim3(attention_max_chunks, attention_kv_heads), threads, 0, stream>>>(query, counts, visible, w);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    fattn_dec_combine<<<attention_query_heads, threads, 0, stream>>>(counts, visible, w);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    publish_output<<<attention_query_heads, threads, 0, stream>>>(w.staged_output, output, w.error);
    return hipGetLastError();
}

} // namespace qwen
