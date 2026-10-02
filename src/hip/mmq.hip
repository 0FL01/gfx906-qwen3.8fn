#include "mmq.cuh"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <limits>

// ADAPT: 0FL01/mx-llama.cpp, pinned production revision
// dcd685463d597d31f5ca759d32c94592a2740fa4 (actual source read before port).
// ggml/src/ggml-cuda/:
//   mmq-load-tiles.cuh::ggml_cuda_mmq_load_tiles_q4_{0,1}, lines179..303;
//   mmq-vec-dot.cuh::ggml_cuda_mmq_vec_dot_q4_{0,1}_q8_1_dp4a, lines10..108;
//   vecdotq.cuh::vec_dot_q4_{0,1}_q8_1_impl (VDR=4, full block32);
//   mmq.cuh::mul_mat_q_process_tile, lines881..954, non-stream-K999..1066,
//   ggml_cuda_mmq_write_back_dp4a441..478, block_q8_1_mmq / DS4 / tile sizes;
//   mmq-config-gfx906.cuh -> mmq-config-gcn.cuh Q4 I64/J8,16,32,64/256threads;
//   common.cuh::ggml_cuda_dp4a and gfx906 DPP exchange/hazard policy.
// KEEP: canonical weight values, 36-byte Q8_1 raw-sum half ABI, DS4 ordering.
// ADAPT: donor 33-word packed-Q4 LDS row stride and skewed 8-scale row stride,
// K256 weight tile/two K128 activation phases, wave-column/lane-row ownership.
// Modifications: load Q4_1 with TWO-byte-safe words too; zero-fill ALL tails
// rather than donor clamped rows/unconditional padded K/columns; DPP quad
// activation-word broadcast uses compiler-visible mov_dpp (compiler schedules
// hazards), not opaque asm. Reuse four packed weight words across wave columns.
// No route IDs, output permutation, stream-K/fixups, MMA/framework or runtime.
// WRITE OURS: borrowed API/range checks and byte-only Q8_1 -> DS4 packer.
// Explicit FP policy: fmaf for Q4_0 correction+accumulation and Q4_1 dot+offset;
// Q4_1 half-RNE products once/full block, separate acc add. -ffp-contract=off
// elsewhere. Full blocks/ascending K/no FP wave reduction differ from MMVQ;
// this is NOT a bitwise MMVQ claim. Existing unchanged numerical gates apply.
// Donor disposition, mx -> furnace -> reinstinct (RECON.md section16):
// furnace 905021dbad71c5056ef51f9fd45d545403fc989c repack-gcn does not support
// canonical Q4_0/Q4_1: reject, no requantization. reinstinct
// 0b79e326351d90d4554a1c18df92da5d0ab692e8 grouped/repacked Q4 is a design
// reference only: its 40-byte float-scale/sum activation and centered-Q4
// correction are incompatible. No code from either is copied here.
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
#error "mmq.hip requires strict floating-point arithmetic (no fast-math)"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "mmq.hip requires gfx906 wave64"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace qwen {
namespace {

constexpr int tile_rows = 64, tile_k = 256, waves = 4, threads = 256;
constexpr int q4_words = 32, q4_stride = q4_words + 1;
constexpr int scales_per_row = tile_k / 32;
constexpr int scale_words = tile_rows * scales_per_row + tile_rows / 4;
constexpr int q8_words = sizeof(Q8MmqBlock) / sizeof(std::uint32_t);
static_assert(sizeof(int) == 4 && sizeof(float) == 4 && q8_words == 36);
static_assert(sizeof(Q4_0) == 18 && offsetof(Q4_0, qs) == 2);
static_assert(sizeof(Q4_1) == 20 && offsetof(Q4_1, qs) == 4);
static_assert(sizeof(Q8_1) == 36 && offsetof(Q8_1, qs) == 4);

struct Range { std::uintptr_t begin, end; };

bool bytes_for(std::size_t a, std::size_t b, std::size_t unit,
               std::size_t& bytes) noexcept {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (b > max / a) return false; // All validated dimensions are positive.
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

bool column_count(int columns) noexcept {
    return columns > 0 && columns <= mmq_device_max_columns;
}

__device__ __forceinline__ float from_half(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float half_product(float a, float b) {
    return __half2float(__float2half_rn(a * b));
}

// get_int_b2 adaptation: canonical weights promise alignment2, including Q4_1.
__device__ __forceinline__ std::uint32_t word2(const std::uint8_t* source) {
    const auto* halves = reinterpret_cast<const std::uint16_t*>(source);
    return std::uint32_t(halves[0]) | (std::uint32_t(halves[1]) << 16);
}

__device__ __forceinline__ std::uint32_t word4(const void* source) {
    // Activation words are aligned4; memcpy preserves raw bits without punning.
    std::uint32_t value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

__global__ void pack_q8(const Q8_1* input, Q8MmqBlock* output,
                       int blocks32, int columns, int count32) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count32) return;
    const int target = index / 4, sub = index % 4;
    const int column = target % columns, k32 = 4 * (target / columns) + sub;
    auto& dst = output[target];
    std::uint16_t d = 0, s = 0;
    // Every sub-block is exclusively owned by one thread; missing block32s
    // zero both DS entries and all codes without dereferencing the source.
    const Q8_1* src = nullptr;
    if (k32 < blocks32) {
        src = input + std::size_t(column) * blocks32 + k32;
        d = src->d;
        s = src->s;
    }
    dst.ds[2 * sub] = d;
    dst.ds[2 * sub + 1] = s;
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        const std::uint32_t bits = src ? word4(src->qs + 4 * word) : 0;
        __builtin_memcpy(dst.qs + 32 * sub + 4 * word, &bits, sizeof(bits));
    }
}

template<TensorType Type>
__device__ __forceinline__ void load_weights(const std::uint8_t* weights,
        std::uint32_t* qs, std::uint32_t* dm, int row0, int k0,
        int blocks32, int rows) {
    constexpr int bytes = Type == TensorType::Q4_0 ? 18 : 20;
    constexpr int codes_offset = Type == TensorType::Q4_0 ? 2 : 4;
    // 32 threads per packed K256 row, two rows per wave; eight rows/pass.
    const int tx = threadIdx.x % q4_words;
    const int block = tx / 4, word = tx % 4;
#pragma unroll
    for (int i0 = 0; i0 < tile_rows; i0 += 2 * waves) {
        const int row = i0 + 2 * threadIdx.y + threadIdx.x / q4_words;
        std::uint32_t bits = 0;
        if (row0 + row < rows && k0 + block < blocks32) {
            const auto offset = (std::size_t(row0 + row) * blocks32 + k0 + block) * bytes;
            bits = word2(weights + offset + codes_offset + 4 * word);
        }
        qs[row * q4_stride + tx] = bits;
    }
    // Eight scales/row, skew one word per four rows (donor bank-conflict fix).
    const int kb = threadIdx.x % scales_per_row;
#pragma unroll
    for (int i0 = 0; i0 < tile_rows; i0 += 8 * waves) {
        const int row = i0 + 8 * threadIdx.y + threadIdx.x / scales_per_row;
        std::uint32_t bits = 0;
        if (row0 + row < rows && k0 + kb < blocks32) {
            const auto offset = (std::size_t(row0 + row) * blocks32 + k0 + kb) * bytes;
            if constexpr (Type == TensorType::Q4_0)
                bits = *reinterpret_cast<const std::uint16_t*>(weights + offset);
            else
                bits = word2(weights + offset);
        }
        dm[row * scales_per_row + row / 4 + kb] = bits;
    }
}

template<int J>
__device__ __forceinline__ void load_activations(const Q8MmqBlock* input,
        std::uint32_t* tile, int column0, int k0, int half,
        int blocks32, int columns) {
    const int tid = 64 * threadIdx.y + threadIdx.x;
#pragma unroll
    for (int l0 = 0; l0 < J * q8_words; l0 += threads) {
        const int l = l0 + tid;
        if (l < J * q8_words) {
            const int column = l / q8_words, word = l % q8_words;
            const int sub = word < 4 ? word : (word - 4) / 8;
            std::uint32_t bits = 0;
            // Mask at block32 granularity even inside an existing last K128;
            // API never relies on caller-provided zeros in unused sub-blocks.
            if (column0 + column < columns && k0 + 4 * half + sub < blocks32) {
                const auto block = std::size_t(k0 / 4 + half) * columns + column0 + column;
                const auto* src = reinterpret_cast<const std::uint8_t*>(input + block);
                bits = word4(src + 4 * word);
            }
            tile[l] = bits;
        }
    }
}

// Each quad loads the four low/high activation words once, then shares them
// among its four row owners. Raw-bit DPP only: no FP reduction/reassociation.
// Controls 0x00/0x55/0xaa/0xff mean quad_perm:[w,w,w,w]. Compiler-visible
// intrinsic retains the donor's mandatory gfx906 VALU->DPP hazard scheduling.
template<int Word>
__device__ __forceinline__ int quad_word(int bits) {
    static_assert(Word >= 0 && Word < 4);
    return __builtin_amdgcn_mov_dpp(bits, Word * 0x55, 0xf, 0xf, false);
}

template<TensorType Type>
__device__ __forceinline__ float accumulate_block(const std::uint32_t* v,
        const int* u, std::uint32_t dm, std::uint32_t ds, float acc) {
    int dot = 0;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        dot = __builtin_amdgcn_sdot4(v[word] & 0x0f0f0f0fu, u[2 * word], dot, false);
        dot = __builtin_amdgcn_sdot4((v[word] >> 4) & 0x0f0f0f0fu, u[2 * word + 1], dot, false);
    }
    const float d4 = from_half(std::uint16_t(dm));
    const float d8 = from_half(std::uint16_t(ds));
    const float s8 = from_half(std::uint16_t(ds >> 16));
    if constexpr (Type == TensorType::Q4_0) {
        const float corrected = fmaf(float(dot), d8, -8.0f * s8);
        return fmaf(d4, corrected, acc);
    } else {
        const float m4 = from_half(std::uint16_t(dm >> 16));
        const float dd = half_product(d4, d8), ms = half_product(m4, s8);
        return acc + fmaf(float(dot), dd, ms);
    }
}

template<TensorType Type, int J>
__global__ __launch_bounds__(threads, 2)
void mmq(const std::uint8_t* weights, const Q8MmqBlock* input,
         float* output, int blocks32, int rows, int columns) {
    static_assert(Type == TensorType::Q4_0 || Type == TensorType::Q4_1);
    static_assert(J == 8 || J == 16 || J == 32 || J == 64);
    // Static LDS: 8448 Q4 code bytes +2112 half-pair bytes +144*J Q8 bytes.
    // All referenced entries have a unique writer each phase. Padding words
    // are never read. Peak J64 =19776 bytes, no dynamic/external workspace.
    __shared__ std::uint32_t x_qs[tile_rows * q4_stride];
    __shared__ std::uint32_t x_dm[scale_words];
    __shared__ std::uint32_t y[J * q8_words];
    const int lane = threadIdx.x, wave = threadIdx.y;
    const int row0 = blockIdx.x * tile_rows, column0 = blockIdx.y * J;
    float acc[J / waves]{};
    // No early exit of inactive rows/columns: full quads and all256 threads
    // participate in DPP and barriers; zero-owned LDS replaces invalid data.
    for (int k0 = 0; k0 < blocks32; k0 += scales_per_row) {
        load_weights<Type>(weights, x_qs, x_dm, row0, k0, blocks32, rows);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            load_activations<J>(input, y, column0, k0, half, blocks32, columns);
            __syncthreads();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
            for (int block = 0; block < 4; ++block) {
                const int xb = 4 * half + block;
                std::uint32_t v[4];
#pragma unroll
                for (int word = 0; word < 4; ++word)
                    v[word] = x_qs[lane * q4_stride + 4 * xb + word];
                const auto dm = x_dm[lane * scales_per_row + lane / 4 + xb];
#pragma unroll
                for (int c = 0; c < J / waves; ++c) {
                    const int column = waves * c + wave;
                    const int lo = int(y[column * q8_words + 4 + 8 * block + lane % 4]);
                    const int hi = int(y[column * q8_words + 8 + 8 * block + lane % 4]);
                    const int u[8] = {
                        quad_word<0>(lo), quad_word<0>(hi),
                        quad_word<1>(lo), quad_word<1>(hi),
                        quad_word<2>(lo), quad_word<2>(hi),
                        quad_word<3>(lo), quad_word<3>(hi)
                    };
                    acc[c] = accumulate_block<Type>(v, u, dm,
                            y[column * q8_words + block], acc[c]);
                }
            }
            __syncthreads(); // Readers finish before y/X can be overwritten.
        }
    }
#pragma unroll
    for (int c = 0; c < J / waves; ++c) {
        const int column = column0 + waves * c + wave, row = row0 + lane;
        if (column < columns && row < rows)
            output[std::size_t(column) * rows + row] = acc[c];
    }
}

template<TensorType Type, int J>
void enqueue(QuantizedDeviceMatrix matrix, const Q8MmqBlock* input,
             int columns, float* output, hipStream_t stream) noexcept {
    const dim3 grid((matrix.output + tile_rows - 1) / tile_rows, (columns + J - 1) / J);
    const dim3 block(64, waves);
    hipLaunchKernelGGL((mmq<Type, J>), grid, block, 0, stream,
            static_cast<const std::uint8_t*>(matrix.weights), input, output,
            matrix.input / 32, matrix.output, columns);
}

template<TensorType Type>
void enqueue_type(QuantizedDeviceMatrix matrix, const Q8MmqBlock* input,
                  int columns, float* output, hipStream_t stream) noexcept {
    if (columns <= 8) enqueue<Type, 8>(matrix, input, columns, output, stream);
    else if (columns <= 16) enqueue<Type, 16>(matrix, input, columns, output, stream);
    else if (columns <= 32) enqueue<Type, 32>(matrix, input, columns, output, stream);
    else enqueue<Type, 64>(matrix, input, columns, output, stream);
}

} // namespace

hipError_t launch_q8_1_to_mmq(const Q8_1* input, int width, int columns,
                            Q8MmqBlock* output, hipStream_t stream) noexcept {
    if (!dimension(width) || width % 32 || !column_count(columns))
        return hipErrorInvalidValue;
    const int blocks32 = width / 32, blocks128 = (blocks32 + 3) / 4;
    std::size_t input_bytes = 0, output_bytes = 0;
    if (!bytes_for(std::size_t(columns), std::size_t(blocks32), sizeof(Q8_1), input_bytes) ||
        !bytes_for(std::size_t(columns), std::size_t(blocks128), sizeof(Q8MmqBlock), output_bytes))
        return hipErrorInvalidValue;
    Range src{}, dst{};
    if (!range(input, input_bytes, 4, src) || !range(output, output_bytes, 16, dst) ||
        overlaps(src, dst))
        return hipErrorInvalidValue;
    const int count32 = blocks128 * columns * 4; // <=65536, after shape validation.
    hipLaunchKernelGGL(pack_q8, dim3((count32 + threads - 1) / threads), dim3(threads),
            0, stream, input, output, blocks32, columns, count32);
    return hipGetLastError();
}

hipError_t launch_mmq_q4(QuantizedDeviceMatrix matrix, const Q8MmqBlock* input,
                       int columns, float* output, hipStream_t stream) noexcept {
    if (matrix.type != TensorType::Q4_0 && matrix.type != TensorType::Q4_1)
        return hipErrorNotSupported;
    if (!dimension(matrix.input) || matrix.input % 32 || !dimension(matrix.output) ||
        !column_count(columns))
        return hipErrorInvalidValue;
    const int blocks32 = matrix.input / 32, blocks128 = (blocks32 + 3) / 4;
    const std::size_t block_bytes = matrix.type == TensorType::Q4_0 ? sizeof(Q4_0) : sizeof(Q4_1);
    std::size_t weight_bytes = 0, input_bytes = 0, output_bytes = 0;
    if (!bytes_for(std::size_t(matrix.output), std::size_t(blocks32), block_bytes, weight_bytes) ||
        !bytes_for(std::size_t(columns), std::size_t(blocks128), sizeof(Q8MmqBlock), input_bytes) ||
        !bytes_for(std::size_t(columns), std::size_t(matrix.output), sizeof(float), output_bytes))
        return hipErrorInvalidValue;
    Range weights_range{}, input_range{}, output_range{};
    if (!range(matrix.weights, weight_bytes, 2, weights_range) ||
        !range(input, input_bytes, 16, input_range) ||
        !range(output, output_bytes, 4, output_range) ||
        overlaps(output_range, weights_range) || overlaps(output_range, input_range))
        return hipErrorInvalidValue;
    if (matrix.type == TensorType::Q4_0)
        enqueue_type<TensorType::Q4_0>(matrix, input, columns, output, stream);
    else
        enqueue_type<TensorType::Q4_1>(matrix, input, columns, output, stream);
    return hipGetLastError();
}

} // namespace qwen
