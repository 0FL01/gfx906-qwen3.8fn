#include "mmq_wide.cuh"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <limits>

// ADAPT: 0FL01/mx-llama.cpp, production pin
// dcd685463d597d31f5ca759d32c94592a2740fa4, ggml/src/ggml-cuda/:
//   mmq-load-tiles.cuh::ggml_cuda_mmq_load_tiles_q5_0 (305..382),
//     ggml_cuda_mmq_load_tiles_q8_0 (463..534),
//     ggml_cuda_mmq_load_tiles_q6_K (947..1033);
//   mmq-vec-dot.cuh::ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a (110..154),
//     ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a (997..1029);
//   vecdotq.cuh::vec_dot_q5_0_q8_1_impl (175..200),
//     vec_dot_q8_0_q8_1_impl (243..258),
//     vec_dot_q6_K_q8_1_impl_mmq (650..675);
//   mmq-config-gcn.cuh / mmq-config-gfx906.cuh literal DP4A configurations,
//     mmq.cuh non-stream-K tile control and checked writeback;
//   common.cuh::ggml_cuda_dp4a and compiler-visible gfx906 DPP exchange.
// KEEP: unchanged canonical weights and existing Q8_1 -> 144-byte DS4 packer;
// Q4_0/Q4_1 forward the existing launcher, with no second implementation.
// ADAPT: 65-word expanded-code row stride, skewed FP32 scales, Q6 packed signed
// subscales. Q8 loads eight words per row/pass as in the gfx906 donor. Unlike
// donor clamped rows/padded K, every missing row/column/block32 owns zero LDS.
// Q6 super-scales have ONE writer/row, removing the donor duplicate stores.
// Q5 deliberately DOES NOT center codes: retain 0..31 and the full raw-half-sum
// correction. The donor's centered Q5-to-Q8 dot consumer drops that correction and is
// not ABI-compatible. Q8/Q6 read half d from DS4, not the donor float-scale ABI.
// Full ascending block32 Q5/Q8 dots and ascending Q6 64-value fragments retain
// explicit fmaf; no wave FP reduction, half-products or activation requantization.
// KEEP: Q5 I64/W4/J8, I128/W4/J16, I128/W8/J32,64; Q8 I128/W8/J8,16,32,64,128;
// Q6 I64/W4/J8,16,32,64. Select smallest listed J >= N, capped at the last J.
// Bounded K loops avoid the donor's problematic full unrolling. These literal
// configs still require parent GPU/codegen and paired performance qualification.
// WRITE OURS: borrowed API, host geometry/range checks and owned-tail masking.
// Donor order/disposition (RECON.md section16): furnace
// 905021dbad71c5056ef51f9fd45d545403fc989c repack-gcn is a repacked consumer
// (no canonical Q5_0 primitive missing here); no import/requantization. reinstinct
// 0b79e326351d90d4554a1c18df92da5d0ab692e8 has a different 40-byte float-scale
// activation ABI; design reference only, no code imported from either.
//
// MIT License (also preserved in third_party/mx-LICENSE)
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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "mmq_wide.hip requires strict floating-point arithmetic (no fast-math)"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "mmq_wide.hip requires gfx906 wave64"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace qwen {
namespace {

constexpr int code_stride = 65, blocks_per_tile = 8;
constexpr int q8_words = sizeof(Q8MmqBlock) / sizeof(std::uint32_t);
static_assert(sizeof(int) == 4 && sizeof(float) == 4 && q8_words == 36);
static_assert(sizeof(Q8_1) == 36 && offsetof(Q8_1, qs) == 4);
static_assert(sizeof(Q8MmqBlock) == 144 && offsetof(Q8MmqBlock, qs) == 16);

struct Range { std::uintptr_t begin, end; };

bool bytes_for(std::size_t a, std::size_t b, std::size_t unit,
               std::size_t& bytes) noexcept {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (a == 0 || b == 0 || b > max / a) return false;
    const auto count = a * b;
    if (unit > max / count) return false;
    bytes = count * unit;
    return true;
}

bool range(const void* pointer, std::size_t bytes, std::size_t alignment,
           Range& result) noexcept {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (!pointer || begin % alignment != 0 ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        return false;
    result = {begin, begin + bytes};
    return true;
}

bool overlaps(Range a, Range b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

bool dimension(int value) noexcept {
    return value > 0 && value <= mmq_device_max_dimension;
}

__device__ __forceinline__ float from_half(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// All canonical offsets and strides are two-byte aligned, NOT necessarily four.
__device__ __forceinline__ std::uint32_t word2(const std::uint8_t* source) {
    const auto* halves = reinterpret_cast<const std::uint16_t*>(source);
    return std::uint32_t(halves[0]) | (std::uint32_t(halves[1]) << 16);
}

__device__ __forceinline__ std::uint32_t word4(const void* source) {
    std::uint32_t value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

template<int I, int W>
__device__ __forceinline__ void load_q5(const std::uint8_t* weights,
        std::uint32_t* qs, float* df, int row0, int k0, int blocks32, int rows) {
    const int tx = threadIdx.x % 32, block = tx / 4, word = tx % 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 2 * W) {
        const int row = i0 + 2 * threadIdx.y + threadIdx.x / 32;
        std::uint32_t lo = 0, hi = 0;
        if (row0 + row < rows && k0 + block < blocks32) {
            const auto offset = (std::size_t(row0 + row) * blocks32 + k0 + block) * 22;
            const auto ql = word2(weights + offset + 6 + 4 * word);
            const auto qh = word2(weights + offset + 2) >> (4 * word);
            lo = ql & 0x0f0f0f0fu;
            lo |= (qh << 4) & 0x00000010u;
            lo |= (qh << 11) & 0x00001000u;
            lo |= (qh << 18) & 0x00100000u;
            lo |= (qh << 25) & 0x10000000u;
            hi = (ql >> 4) & 0x0f0f0f0fu;
            hi |= (qh >> 12) & 0x00000010u;
            hi |= (qh >> 5) & 0x00001000u;
            hi |= (qh << 2) & 0x00100000u;
            hi |= (qh << 9) & 0x10000000u;
        }
        // Each packed-word owner writes one low/high expanded pair, still unsigned.
        qs[row * code_stride + 8 * block + word] = lo;
        qs[row * code_stride + 8 * block + word + 4] = hi;
    }
    const int kb = threadIdx.x % blocks_per_tile;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 8 * W) {
        const int row = i0 + 8 * threadIdx.y + threadIdx.x / blocks_per_tile;
        float d = 0.0f;
        if (row0 + row < rows && k0 + kb < blocks32) {
            const auto offset = (std::size_t(row0 + row) * blocks32 + k0 + kb) * 22;
            d = from_half(*reinterpret_cast<const std::uint16_t*>(weights + offset));
        }
        df[row * blocks_per_tile + row / 4 + kb] = d;
    }
}

template<int I, int W>
__device__ __forceinline__ void load_q8(const std::uint8_t* weights,
        std::uint32_t* qs, float* df, int row0, int k0, int blocks32, int rows) {
    const int word = threadIdx.x % 8;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 8 * W) {
        const int row = i0 + 8 * threadIdx.y + threadIdx.x / 8;
#pragma unroll
        for (int block = 0; block < blocks_per_tile; ++block) {
            std::uint32_t bits = 0;
            if (row0 + row < rows && k0 + block < blocks32) {
                const auto offset = (std::size_t(row0 + row) * blocks32 + k0 + block) * 34;
                bits = word2(weights + offset + 2 + 4 * word);
            }
            qs[row * code_stride + 8 * block + word] = bits;
        }
        float d = 0.0f;
        if (row0 + row < rows && k0 + word < blocks32) {
            const auto offset = (std::size_t(row0 + row) * blocks32 + k0 + word) * 34;
            d = from_half(*reinterpret_cast<const std::uint16_t*>(weights + offset));
        }
        df[row * blocks_per_tile + row / 4 + word] = d;
    }
}

__device__ __forceinline__ std::uint32_t centered_q6(std::uint32_t bits) {
    // Per-byte subtraction without cross-byte borrow; exact [-32,31], no saturation.
    return ((bits | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
}

template<int I, int W>
__device__ __forceinline__ void load_q6(const std::uint8_t* weights,
        std::uint32_t* qs, float* df, std::uint32_t* sc,
        int row0, int k256, int blocks256, int rows) {
    const int tx = threadIdx.x % 32;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 2 * W) {
        const int row = i0 + 2 * threadIdx.y + threadIdx.x / 32;
        std::uint32_t lo = 0, hi = 0;
        if (row0 + row < rows && k256 < blocks256) {
            const auto offset = (std::size_t(row0 + row) * blocks256 + k256) * 210;
            const auto ql = word2(weights + offset + 4 * tx);
            const auto qh = word2(weights + offset + 128 +
                    4 * (8 * (tx / 16) + tx % 8)) >> ((tx & 8) >> 2);
            lo = centered_q6((ql & 0x0f0f0f0fu) | ((qh << 4) & 0x30303030u));
            hi = centered_q6(((ql >> 4) & 0x0f0f0f0fu) | (qh & 0x30303030u));
        }
        const int kq = 2 * tx - tx % 16;
        qs[row * code_stride + kq] = lo;
        qs[row * code_stride + kq + 16] = hi;
    }
    const int tid = 64 * threadIdx.y + threadIdx.x;
    // Unlike the donor's modulo-I writes, precisely one thread owns each scale.
    if (tid < I) {
        float d = 0.0f;
        if (row0 + tid < rows && k256 < blocks256) {
            const auto offset = (std::size_t(row0 + tid) * blocks256 + k256) * 210;
            d = from_half(*reinterpret_cast<const std::uint16_t*>(weights + offset + 208));
        }
        df[tid + tid / 32] = d;
    }
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 16 * W) {
        const int row = i0 + 16 * threadIdx.y + threadIdx.x / 4;
        const int word = threadIdx.x % 4;
        std::uint32_t bits = 0;
        if (row0 + row < rows && k256 < blocks256) {
            const auto offset = (std::size_t(row0 + row) * blocks256 + k256) * 210;
            bits = word2(weights + offset + 192 + 4 * word);
        }
        sc[4 * row + row / 8 + word] = bits;
    }
}

template<int J, int W>
__device__ __forceinline__ void load_activations(const Q8MmqBlock* input,
        std::uint32_t* tile, int column0, int k0, int half,
        int blocks32, int columns) {
    const int tid = 64 * threadIdx.y + threadIdx.x;
#pragma unroll
    for (int l0 = 0; l0 < J * q8_words; l0 += 64 * W) {
        const int l = l0 + tid;
        if (l < J * q8_words) {
            const int column = l / q8_words, word = l % q8_words;
            const int sub = word < 4 ? word : (word - 4) / 8;
            std::uint32_t bits = 0;
            // Do not trust even the unused part of a valid last K128 DS4 block.
            if (column0 + column < columns && k0 + 4 * half + sub < blocks32) {
                const auto block = std::size_t(k0 / 4 + half) * columns + column0 + column;
                bits = word4(reinterpret_cast<const std::uint8_t*>(input + block) + 4 * word);
            }
            tile[l] = bits;
        }
    }
}

template<int Word>
__device__ __forceinline__ int quad_word(int bits) {
    static_assert(Word >= 0 && Word < 4);
    return __builtin_amdgcn_mov_dpp(bits, Word * 0x55, 0xf, 0xf, false);
}

// Full quads participate even on invalid rows/columns. Compiler-visible DPP
// keeps the gfx906 VALU->DPP hazards scheduled; only raw words are exchanged.
__device__ __forceinline__ void activation_words(const std::uint32_t* y,
        int column, int block, int lane, int* u) {
    const int lo = int(y[column * q8_words + 4 + 8 * block + lane % 4]);
    const int hi = int(y[column * q8_words + 8 + 8 * block + lane % 4]);
    u[0] = quad_word<0>(lo); u[1] = quad_word<1>(lo);
    u[2] = quad_word<2>(lo); u[3] = quad_word<3>(lo);
    u[4] = quad_word<0>(hi); u[5] = quad_word<1>(hi);
    u[6] = quad_word<2>(hi); u[7] = quad_word<3>(hi);
}

template<TensorType Type>
__device__ __forceinline__ float accumulate32(const std::uint32_t* v,
        const int* u, float dw, std::uint32_t ds, float acc) {
    int dot = 0;
#pragma unroll
    for (int word = 0; word < 8; ++word)
        dot = __builtin_amdgcn_sdot4(v[word], u[word], dot, false);
    const float d8 = from_half(std::uint16_t(ds));
    if constexpr (Type == TensorType::Q5_0) {
        const float s_raw = from_half(std::uint16_t(ds >> 16));
        const float corrected = fmaf(float(dot), d8, -16.0f * s_raw);
        return fmaf(dw, corrected, acc);
    } else {
        const float dd = dw * d8; // FP32, deliberately NOT half-rounded.
        return fmaf(dd, float(dot), acc);
    }
}

__device__ __forceinline__ int subscale(std::uint32_t sc, int index) {
    const int value = int((sc >> (8 * index)) & 255u);
    return value < 128 ? value : value - 256;
}

__device__ __forceinline__ float accumulate64(const std::uint32_t* v,
        const int* u, const int* next, std::uint32_t sc, float d6,
        std::uint32_t ds_a, std::uint32_t ds_b, float acc) {
    int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        a0 = __builtin_amdgcn_sdot4(v[word], u[word], a0, false);
        a1 = __builtin_amdgcn_sdot4(v[word + 4], u[word + 4], a1, false);
        b0 = __builtin_amdgcn_sdot4(v[word + 8], next[word], b0, false);
        b1 = __builtin_amdgcn_sdot4(v[word + 12], next[word + 4], b1, false);
    }
    // Two int16-length dots times signed subscales, summed BEFORE int->float.
    // Each combined magnitude <= 2^24, including activations/subscales -128.
    const int scaled_a = a0 * subscale(sc, 0) + a1 * subscale(sc, 1);
    const int scaled_b = b0 * subscale(sc, 2) + b1 * subscale(sc, 3);
    float pair = fmaf(from_half(std::uint16_t(ds_a)), float(scaled_a), 0.0f);
    pair = fmaf(from_half(std::uint16_t(ds_b)), float(scaled_b), pair);
    return fmaf(d6, pair, acc);
}

template<TensorType Type, int I, int J, int W>
__global__ __launch_bounds__(64 * W, I == 128 ? 1 : 2)
void mmq32(const std::uint8_t* weights, const Q8MmqBlock* input,
           float* output, int blocks32, int rows, int columns) {
    static_assert(Type == TensorType::Q5_0 || Type == TensorType::Q8_0);
    static_assert(I == 64 || I == 128);
    static_assert(W == 4 || W == 8);
    static_assert(J % W == 0);
    // Owned LDS bytes: 4*(65*I + 8*I + I/4) + 144*J.
    // Q5: J8=19904, J16=39808, J32=42112, J64=46720.
    // Q8: J8=38656, J16=39808, J32=42112, J64=46720, J128=55936.
    // Every I128 tile exceeds 32 KiB: gfx906's 64 KiB LDS permits only
    // one resident block. I64/J8 still permits the two-block target.
    // Padding is never read; all referenced entries have one writer/phase.
    __shared__ std::uint32_t x_qs[I * code_stride];
    __shared__ float x_df[I * blocks_per_tile + I / 4];
    __shared__ std::uint32_t y[J * q8_words];
    const int lane = threadIdx.x, wave = threadIdx.y;
    const int row0 = blockIdx.x * I, column0 = blockIdx.y * J;
    float acc[J / W][I / 64]{};
    for (int k0 = 0; k0 < blocks32; k0 += blocks_per_tile) {
        if constexpr (Type == TensorType::Q5_0)
            load_q5<I, W>(weights, x_qs, x_df, row0, k0, blocks32, rows);
        else
            load_q8<I, W>(weights, x_qs, x_df, row0, k0, blocks32, rows);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            load_activations<J, W>(input, y, column0, k0, half, blocks32, columns);
            __syncthreads();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
            for (int block = 0; block < 4; ++block) {
                const int xb = 4 * half + block;
#pragma unroll
                for (int r = 0; r < I / 64; ++r) {
                    const int row = lane + 64 * r;
                    std::uint32_t v[8];
#pragma unroll
                    for (int word = 0; word < 8; ++word)
                        v[word] = x_qs[row * code_stride + 8 * xb + word];
                    const float dw = x_df[row * blocks_per_tile + row / 4 + xb];
#pragma unroll
                    for (int c = 0; c < J / W; ++c) {
                        const int column = W * c + wave;
                        int u[8];
                        activation_words(y, column, block, lane, u);
                        acc[c][r] = accumulate32<Type>(v, u, dw,
                                y[column * q8_words + block], acc[c][r]);
                    }
                }
            }
            __syncthreads(); // All readers finish before y or X can be overwritten.
        }
    }
#pragma unroll
    for (int c = 0; c < J / W; ++c) {
#pragma unroll
        for (int r = 0; r < I / 64; ++r) {
            const int column = column0 + W * c + wave, row = row0 + lane + 64 * r;
            if (column < columns && row < rows)
                output[std::size_t(column) * rows + row] = acc[c][r];
        }
    }
}

template<int J>
__global__ __launch_bounds__(256, 2)
void mmq6(const std::uint8_t* weights, const Q8MmqBlock* input,
          float* output, int blocks256, int rows, int columns) {
    constexpr int I = 64, W = 4;
    static_assert(J == 8 || J == 16 || J == 32 || J == 64);
    // Owned LDS: codes16640 + FP32 super-scales264 + packed subscales1056
    // +144*J; J8/16/32/64 =19112/20264/22568/27176 bytes, respectively.
    __shared__ std::uint32_t x_qs[I * code_stride];
    __shared__ float x_df[I + I / 32];
    __shared__ std::uint32_t x_sc[4 * I + I / 8];
    __shared__ std::uint32_t y[J * q8_words];
    const int lane = threadIdx.x, wave = threadIdx.y;
    const int row0 = blockIdx.x * I, column0 = blockIdx.y * J;
    float acc[J / W]{};
    for (int k256 = 0; k256 < blocks256; ++k256) {
        load_q6<I, W>(weights, x_qs, x_df, x_sc, row0, k256, blocks256, rows);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            load_activations<J, W>(input, y, column0, 8 * k256, half,
                    8 * blocks256, columns);
            __syncthreads();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
            for (int fragment = 0; fragment < 2; ++fragment) {
                const int xf = 2 * half + fragment, block = 2 * fragment;
                std::uint32_t v[16];
#pragma unroll
                for (int word = 0; word < 16; ++word)
                    v[word] = x_qs[lane * code_stride + 16 * xf + word];
                const auto sc = x_sc[4 * lane + lane / 8 + xf];
                const float d6 = x_df[lane + lane / 32];
#pragma unroll
                for (int c = 0; c < J / W; ++c) {
                    const int column = W * c + wave;
                    int u[8], next[8];
                    activation_words(y, column, block, lane, u);
                    activation_words(y, column, block + 1, lane, next);
                    acc[c] = accumulate64(v, u, next, sc, d6,
                            y[column * q8_words + block],
                            y[column * q8_words + block + 1], acc[c]);
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int c = 0; c < J / W; ++c) {
        const int column = column0 + W * c + wave, row = row0 + lane;
        if (column < columns && row < rows)
            output[std::size_t(column) * rows + row] = acc[c];
    }
}

template<TensorType Type, int I, int J, int W>
void enqueue32(QuantizedDeviceMatrix matrix, const Q8MmqBlock* input,
               int columns, float* output, hipStream_t stream) noexcept {
    const dim3 grid((matrix.output + I - 1) / I, (columns + J - 1) / J);
    hipLaunchKernelGGL((mmq32<Type, I, J, W>), grid, dim3(64, W), 0, stream,
            static_cast<const std::uint8_t*>(matrix.weights), input, output,
            matrix.input / 32, matrix.output, columns);
}

template<int J>
void enqueue6(QuantizedDeviceMatrix matrix, const Q8MmqBlock* input,
              int columns, float* output, hipStream_t stream) noexcept {
    const dim3 grid((matrix.output + 63) / 64, (columns + J - 1) / J);
    hipLaunchKernelGGL((mmq6<J>), grid, dim3(64, 4), 0, stream,
            static_cast<const std::uint8_t*>(matrix.weights), input, output,
            matrix.input / 256, matrix.output, columns);
}

} // namespace

hipError_t launch_mmq_linear(QuantizedDeviceMatrix matrix, const Q8MmqBlock* input,
        int columns, float* output, hipStream_t stream) noexcept {
    if (matrix.type == TensorType::Q4_0 || matrix.type == TensorType::Q4_1)
        return launch_mmq_q4(matrix, input, columns, output, stream);
    int block_elements = 0, block_bytes = 0;
    switch (matrix.type) {
    case TensorType::Q5_0: block_elements = 32; block_bytes = 22; break;
    case TensorType::Q8_0: block_elements = 32; block_bytes = 34; break;
    case TensorType::Q6_K: block_elements = 256; block_bytes = 210; break;
    default: return hipErrorNotSupported;
    }
    const bool head = matrix.type == TensorType::Q6_K &&
                      matrix.input == 2560 && matrix.output == 248320;
    if (!dimension(matrix.input) || matrix.input % block_elements != 0 ||
        (!dimension(matrix.output) && !head) ||
        columns < 1 || columns > mmq_device_max_columns)
        return hipErrorInvalidValue;
    const int blocks128 = (matrix.input + 127) / 128;
    std::size_t weight_bytes = 0, input_bytes = 0, output_bytes = 0;
    if (!bytes_for(std::size_t(matrix.output), std::size_t(matrix.input / block_elements),
                   std::size_t(block_bytes), weight_bytes) ||
        !bytes_for(std::size_t(columns), std::size_t(blocks128), sizeof(Q8MmqBlock), input_bytes) ||
        !bytes_for(std::size_t(columns), std::size_t(matrix.output), sizeof(float), output_bytes))
        return hipErrorInvalidValue;
    Range weights_range{}, input_range{}, output_range{};
    if (!range(matrix.weights, weight_bytes, 2, weights_range) ||
        !range(input, input_bytes, 16, input_range) ||
        !range(output, output_bytes, 4, output_range) ||
        overlaps(output_range, weights_range) || overlaps(output_range, input_range))
        return hipErrorInvalidValue;

    if (matrix.type == TensorType::Q5_0) {
        if (columns <= 8) enqueue32<TensorType::Q5_0, 64, 8, 4>(matrix, input, columns, output, stream);
        else if (columns <= 16) enqueue32<TensorType::Q5_0, 128, 16, 4>(matrix, input, columns, output, stream);
        else if (columns <= 32) enqueue32<TensorType::Q5_0, 128, 32, 8>(matrix, input, columns, output, stream);
        else enqueue32<TensorType::Q5_0, 128, 64, 8>(matrix, input, columns, output, stream);
    } else if (matrix.type == TensorType::Q8_0) {
        if (columns <= 8) enqueue32<TensorType::Q8_0, 128, 8, 8>(matrix, input, columns, output, stream);
        else if (columns <= 16) enqueue32<TensorType::Q8_0, 128, 16, 8>(matrix, input, columns, output, stream);
        else if (columns <= 32) enqueue32<TensorType::Q8_0, 128, 32, 8>(matrix, input, columns, output, stream);
        else if (columns <= 64) enqueue32<TensorType::Q8_0, 128, 64, 8>(matrix, input, columns, output, stream);
        else enqueue32<TensorType::Q8_0, 128, 128, 8>(matrix, input, columns, output, stream);
    } else {
        if (columns <= 8) enqueue6<8>(matrix, input, columns, output, stream);
        else if (columns <= 16) enqueue6<16>(matrix, input, columns, output, stream);
        else if (columns <= 32) enqueue6<32>(matrix, input, columns, output, stream);
        else enqueue6<64>(matrix, input, columns, output, stream);
    }
    return hipGetLastError();
}

} // namespace qwen
