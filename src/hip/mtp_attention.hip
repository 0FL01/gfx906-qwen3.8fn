#include "mtp_attention.cuh"
#include "dpp.cuh"
#include <hip/hip_fp16.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "mtp_attention.hip requires gfx906 wave64"
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "mtp_attention.hip requires strict finite checks and -ffp-contract=off"
#endif

// Donor-first mx -> furnace -> reinstinct, RECON16/PLAN selection rule:
// KEEP mx dcd685463d597d31f5ca759d32c94592a2740fa4 common.cuh sum64,
// already extracted in dpp.cuh (including VALU/DPP hazard waits).
// ADAPT furnace 905021dbad71c5056ef51f9fd45d545403fc989c fattn.cu
// fattn_dec_chunk<256,12>/fattn_dec_combine<256> through the CURRENT qualified
// src/hip/attention.hip numerical seam: wave=three heads/lane=key, LDS Q,
// ascending dot256/value64, existing exp(double)->float chunk and expf merge.
// Inspect RECON16 documented gather/dispatch seam and actual attention.hip,
// NOT the donor's masked/fast-exp arithmetic as a replacement for this seam.
// KEEP reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8
// attn_partial_q8.cpp/attn_merge.cpp as split-K algebra reference ONLY; Q8/fast
// exp/layout do not satisfy unchanged canonical Q4->half RNE arithmetic.
// WRITE OURS typed full-prefix host preflight/device preflight/publication API.
// ADAPT addresses to immutable canonical [capacity][2][8], B1..3, no IDs/sort,
// no 2051 budget/gather/runtime imports. <=256 chunks use the EXACT old merge;
// >256 chunks extend each thread's max/denominator over ascending 256-stripes,
// retaining mx sum64/four-wave fold and ascending final chunk numerator fold.
// Long merge is bounded, not claimed bit-identical to a serial CPU oracle.
// MIT attribution/license for the adapted ggml/furnace/mx kernel follows.
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace qwen {
namespace {
constexpr int T = 256, D = 256, G = 12, C = 64;
template<class U> __device__ __forceinline__ U* row(MtpAttentionQ4View<U> v, unsigned q) {
    return v.data + static_cast<std::size_t>(q) * v.stride;
}
__device__ __forceinline__ void report(int flags, int* error) {
    if (flags) atomicOr(error, flags);
}
__device__ __forceinline__ bool ready(int first, unsigned q, MtpAttentionQ4Cache cache, int* error) {
    if (cache.capacity < 1 || cache.capacity > mtp_attention_max_capacity || cache.valid_rows < 1 ||
        cache.valid_rows > cache.capacity || first < 1 || first > cache.valid_rows ||
        q > static_cast<unsigned>(cache.valid_rows - first)) {
        report(mtp_attention_error_source, error); return false;
    }
    return __atomic_load_n(error, __ATOMIC_RELAXED) == 0;
}
__device__ __forceinline__ float checked(float x, int& flags) {
    if (!isfinite(x)) { flags |= mtp_attention_error_arithmetic; return 0.0f; }
    return x;
}
__device__ __forceinline__ float half_value(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}
__device__ __forceinline__ float decode(const Q4_0& b, int d, int& flags) {
    if ((b.d & 0x7c00) == 0x7c00) { flags |= mtp_attention_error_source; return 0.0f; }
    const int code = d < 16 ? b.qs[d] & 15 : b.qs[d - 16] >> 4;
    const float x = half_value(b.d) * static_cast<float>(code - 8);
    const auto bits = __half_as_ushort(__float2half_rn(x));
    if ((bits & 0x7c00) == 0x7c00) { flags |= mtp_attention_error_arithmetic; return 0.0f; }
    return half_value(bits);
}
__device__ __forceinline__ float operand(const Q4_0* cache, int cell, int g, int d, int& flags) {
    return decode(cache[static_cast<std::size_t>(cell) * 16 + g * 8 + d / 32], d % 32, flags);
}
__device__ __forceinline__ float max64(float x) {
#pragma unroll
    for (int mask = 1; mask < 64; mask *= 2) x = fmaxf(x, __shfl_xor(x, mask, 64));
    return x;
}
__global__ __launch_bounds__(T)
void validate_queries(MtpAttentionQ4View<const float> queries, MtpAttentionQ4Cache cache,
                      int first, int* error) {
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(first, blockIdx.x, cache, error);
    __syncthreads();
    if (!active) return;
    const auto* query = row(queries, blockIdx.x);
    int flags = 0;
    for (int i = threadIdx.x; i < 6144; i += T)
        if (!isfinite(query[i])) flags |= mtp_attention_error_source;
    report(flags, error);
}
// Only actual causal rows, not valid_rows, determine these loops. Independent
// query grids may share a read-only prefix; no decoded/staged history allocation.
__global__ __launch_bounds__(T)
void validate_scales(MtpAttentionQ4Cache cache, int first, int* error) {
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(first, blockIdx.y, cache, error);
    __syncthreads();
    if (!active) return;
    const int visible = first + static_cast<int>(blockIdx.y);
    const int begin = static_cast<int>(blockIdx.x) * C * 16;
    const int end = (begin + C * 16 < visible * 16) ? begin + C * 16 : visible * 16;
    int flags = 0;
    for (int b = begin + threadIdx.x; b < end; b += T)
        if ((cache.keys[b].d & 0x7c00) == 0x7c00 || (cache.values[b].d & 0x7c00) == 0x7c00)
            flags |= mtp_attention_error_source;
    report(flags, error);
}
__global__ __launch_bounds__(T)
void validate_decoded(MtpAttentionQ4Cache cache, int first, int* error) {
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(first, blockIdx.y, cache, error);
    __syncthreads();
    if (!active) return;
    const int visible = first + static_cast<int>(blockIdx.y);
    const int begin = static_cast<int>(blockIdx.x) * C * 512;
    const int end = (begin + C * 512 < visible * 512) ? begin + C * 512 : visible * 512;
    int flags = 0;
    for (int i = begin + threadIdx.x; i < end; i += T) {
        (void)decode(cache.keys[i / 32], i % 32, flags);
        (void)decode(cache.values[i / 32], i % 32, flags);
    }
    report(flags, error);
}
__global__ __launch_bounds__(T)
void dense_chunk(MtpAttentionQ4View<const float> queries, MtpAttentionQ4Cache cache,
                 int first, MtpAttentionQ4Workspace workspace) {
    constexpr int H = G / 4;
    const auto q = blockIdx.z;
    __shared__ float4 qs[G][D / 4];
    __shared__ float ps[G][C];
    __shared__ float maxima[G];
    __shared__ int cells[C];
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(first, q, cache, workspace.error);
    __syncthreads();
    if (!active) return;
    const int n = first + static_cast<int>(q);
    const int chunk = blockIdx.x, g = blockIdx.y;
    if (chunk >= (n + C - 1) / C) return;
    const int chunk_count = n - chunk * C < C ? n - chunk * C : C;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const auto* query = row(queries, q);
    auto* partial = row(workspace.partial_output, q);
    auto* ml = row(workspace.partial_max_sum, q);
    int flags = 0;
    for (int i = tid; i < G * D / 4; i += T) {
        const int h = i / (D / 4), d4 = i % (D / 4);
        qs[h][d4] = reinterpret_cast<const float4*>(query + (g * G + h) * D)[d4];
    }
    if (tid < C) { const int j = chunk * C + tid; cells[tid] = j < n ? j : -1; }
    __syncthreads();
    {
        const int cell = cells[lane];
        float s[H];
#pragma unroll
        for (int h = 0; h < H; ++h) s[h] = 0.0f;
#pragma unroll
        for (int c = 0; c < D / 8; ++c) {
            const float k0 = cell >= 0 ? operand(cache.keys, cell, g, 8 * c + 0, flags) : 0.0f;
            const float k1 = cell >= 0 ? operand(cache.keys, cell, g, 8 * c + 1, flags) : 0.0f;
            const float k2 = cell >= 0 ? operand(cache.keys, cell, g, 8 * c + 2, flags) : 0.0f;
            const float k3 = cell >= 0 ? operand(cache.keys, cell, g, 8 * c + 3, flags) : 0.0f;
            const float k4 = cell >= 0 ? operand(cache.keys, cell, g, 8 * c + 4, flags) : 0.0f;
            const float k5 = cell >= 0 ? operand(cache.keys, cell, g, 8 * c + 5, flags) : 0.0f;
            const float k6 = cell >= 0 ? operand(cache.keys, cell, g, 8 * c + 6, flags) : 0.0f;
            const float k7 = cell >= 0 ? operand(cache.keys, cell, g, 8 * c + 7, flags) : 0.0f;
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
            const float score = cell >= 0 ? checked(s[i] * 0.0625f, flags) : -INFINITY;
            const float m = max64(score);
            const float difference = cell >= 0 ? checked(score - m, flags) : 0.0f;
            ps[h][lane] = cell >= 0 ? checked(static_cast<float>(exp(static_cast<double>(difference))), flags) : 0.0f;
            if (lane == 0) maxima[h] = m;
        }
    }
    __syncthreads();
    if (lane == 0) for (int h = wave; h < G; h += 4) {
        float l = 0.0f;
        for (int j = 0; j < chunk_count; ++j) l = checked(l + ps[h][j], flags);
        if (!(l > 0.0f)) flags |= mtp_attention_error_arithmetic;
        ml[(chunk * 2 + g) * G + h] = make_float2(maxima[h], l);
    }
    float o[G];
#pragma unroll
    for (int h = 0; h < G; ++h) o[h] = 0.0f;
    for (int j0 = 0; j0 < chunk_count; j0 += 16) {
        float v[16];
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int cell = cells[j0 + jj];
            v[jj] = cell >= 0 ? operand(cache.values, cell, g, tid, flags) : 0.0f;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
#pragma unroll
            for (int jj = 0; jj < 16; ++jj)
                if (j0 + jj < chunk_count) o[h] = checked(o[h] + checked(ps[h][j0 + jj] * v[jj], flags), flags);
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) partial[((chunk * 2 + g) * G + h) * D + tid] = o[h];
    report(flags, workspace.error);
}
__device__ __forceinline__ float2 checked_ml(const float2* source, int c, int h, int& flags) {
    auto ml = source[c * 24 + h];
    if (!isfinite(ml.x) || !isfinite(ml.y) || !(ml.y > 0.0f)) {
        flags |= mtp_attention_error_arithmetic; ml = make_float2(-INFINITY, 0.0f);
    }
    return ml;
}
__global__ __launch_bounds__(T)
void dense_combine(MtpAttentionQ4Cache cache, int first, MtpAttentionQ4Workspace workspace) {
    const auto q = blockIdx.y;
    __shared__ float factor[mtp_attention_max_chunks];
    __shared__ float red[4];
    __shared__ int active;
    if (threadIdx.x == 0) active = ready(first, q, cache, workspace.error);
    __syncthreads();
    if (!active) return;
    const int n_chunks = (first + static_cast<int>(q) + C - 1) / C;
    const int hq = blockIdx.x, tid = threadIdx.x;
    const auto* partial = row(workspace.partial_output, q);
    const auto* max_sum = row(workspace.partial_max_sum, q);
    auto* staged = row(workspace.staged_output, q);
    int flags = 0;
    float2 ml = make_float2(-INFINITY, 0.0f);
    if (tid < n_chunks) ml = checked_ml(max_sum, tid, hq, flags);
    float local_m = ml.y > 0.0f ? ml.x : -INFINITY;
    for (int c = tid + T; c < n_chunks; c += T) {
        const auto next = checked_ml(max_sum, c, hq, flags);
        local_m = fmaxf(local_m, next.y > 0.0f ? next.x : -INFINITY);
    }
    const float m = max64(local_m);
    if (tid % 64 == 0) red[tid / 64] = m;
    __syncthreads();
    const float m_all = checked(fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])), flags);
    __syncthreads();
    const float f = ml.y > 0.0f ? checked(expf(checked(ml.x - m_all, flags)), flags) : 0.0f;
    if (tid < n_chunks) factor[tid] = f;
    float local_l = checked(ml.y * f, flags); // EXACT old operand for <=256 chunks
    for (int c = tid + T; c < n_chunks; c += T) {
        const auto next = checked_ml(max_sum, c, hq, flags);
        const float fc = next.y > 0.0f ? checked(expf(checked(next.x - m_all, flags)), flags) : 0.0f;
        factor[c] = fc;
        local_l = checked(local_l + checked(next.y * fc, flags), flags);
    }
    const float l = sum64(local_l);
    if (tid % 64 == 0) red[tid / 64] = l;
    __syncthreads();
    const float l_all = checked((red[0] + red[1]) + (red[2] + red[3]), flags);
    if (!(l_all > 0.0f)) flags |= mtp_attention_error_arithmetic;
    const float inv_l = l_all > 0.0f ? checked(1.0f / l_all, flags) : 0.0f;
    float acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c)
        acc = checked(acc + checked(partial[(c * 24 + hq) * D + tid] * factor[c], flags), flags);
    staged[hq * D + tid] = checked(acc * inv_l, flags);
    report(flags, workspace.error);
}
__global__ __launch_bounds__(T)
void publish(MtpAttentionQ4View<float> stages, MtpAttentionQ4View<float> outputs, const int* error) {
    if (__atomic_load_n(error, __ATOMIC_RELAXED) != 0) return;
    const int i = blockIdx.x * T + threadIdx.x;
    if (i < 6144) row(outputs, blockIdx.y)[i] = row(stages, blockIdx.y)[i];
}
} // namespace

hipError_t launch_mtp_attention_q4(MtpAttentionQ4View<const float> query, MtpAttentionQ4Cache cache,
    int batch, int first, MtpAttentionQ4View<float> output, MtpAttentionQ4Workspace w, hipStream_t stream) noexcept {
    if (!stream || !mtp_attention_detail::valid_arguments(query, cache, batch, first, output, w))
        return hipErrorInvalidValue;
    const auto chunks = static_cast<unsigned>(mtp_attention_q4_layout(cache.capacity).chunks);
    validate_queries<<<batch, T, 0, stream>>>(query, cache, first, w.error);
    auto status = hipGetLastError(); if (status != hipSuccess) return status;
    validate_scales<<<dim3(chunks, batch), T, 0, stream>>>(cache, first, w.error);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    validate_decoded<<<dim3(chunks, batch), T, 0, stream>>>(cache, first, w.error);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    dense_chunk<<<dim3(chunks, 2, batch), T, 0, stream>>>(query, cache, first, w);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    dense_combine<<<dim3(24, batch), T, 0, stream>>>(cache, first, w);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    publish<<<dim3(24, batch), T, 0, stream>>>(w.staged_output, output, w.error);
    return hipGetLastError();
}
} // namespace qwen
