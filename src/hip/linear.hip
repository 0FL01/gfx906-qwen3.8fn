#include "linear.cuh"
#include "dpp.cuh"

#include <hip/hip_fp16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// ADAPT: mx-llama.cpp, dcd685463d597d31f5ca759d32c94592a2740fa4,
// ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_q{4_0,4_1,5_0,8_0,6_K}_q8_1,
// their *_impl helpers, common.cuh::ggml_cuda_dp4a / gfx906 DPP reductions,
// mmvq.cu::mul_mat_vec_q (670..737), calc_nwarps / calc_rows_per_block /
// should_use_small_k, and quantize.cu::quantize_q8_1.
// Copyright (c) 2023-2026 The ggml authors; MIT license, complete notice in
// third_party/mx-LICENSE. Adapt only canonical primitives, no GGML runtime/types.
// KEEP: canonical on-disk bytes and our quant.hpp 36-byte raw-sum Q8_1 ABI.
// ADAPT: MMVQ fragments, two cooperating wave64s per row for N=1/2/3,
// production row tiling/reduction order, multi-column weight register reuse.
// WRITE OURS: borrowed-buffer API, range validation and numeric error handling.
// Donor order/decision: mx above; furnace
// 905021dbad71c5056ef51f9fd45d545403fc989c repack-gcn is a different repacked
// consumer, not a canonical replacement; reinstinct
// 0b79e326351d90d4554a1c18df92da5d0ab692e8 matvec multi-N pattern was inspected
// in R1. Its 40-byte activation ABI / quantized-sum correction are not imported.
// Modified FP contract: explicit fmaf reproduces the pinned gfx906 MMVQ native
// contractions (Q4/Q5 inner+acc, Q4_1 inner only, Q8 acc, Q6 pair+acc), while
// -ffp-contract=off and Q4_1 half-RNE products remain mandatory. Qualification:
// existing common-Q8 gate 2e-4+2e-5*|ref|, matched-input production MMVQ, then
// unchanged Session gates (logits .02+.002*|ref|, intermediates .002+.002*|ref|).

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "linear.hip requires strict floating-point arithmetic (no fast-math)"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "linear.hip requires gfx906 wave64"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace qwen {
namespace {

static_assert(sizeof(int) == 4 && sizeof(float) == 4);
static_assert(sizeof(Q4_0) == 18 && offsetof(Q4_0, qs) == 2);
static_assert(sizeof(Q4_1) == 20 && offsetof(Q4_1, qs) == 4);
static_assert(sizeof(Q8_1) == 36 && offsetof(Q8_1, qs) == 4);

struct Range { std::uintptr_t begin, end; };

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
    return value > 0 && value <= linear_device_max_dimension;
}

bool layout(TensorType type, int& elements, int& bytes) noexcept {
    switch (type) {
    case TensorType::Q4_0: elements = 32; bytes = 18; return true;
    case TensorType::Q4_1: elements = 32; bytes = 20; return true;
    case TensorType::Q5_0: elements = 32; bytes = 22; return true;
    case TensorType::Q8_0: elements = 32; bytes = 34; return true;
    case TensorType::Q6_K: elements = 256; bytes = 210; return true;
    default: return false;
    }
}

__device__ __forceinline__ float from_half(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ std::uint16_t to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

// Canonical blocks have 2-byte strides/alignment (e.g. Q8_0 stride34,
// Q6_K stride210). Do not replace this with a potentially unaligned int load.
// ADAPT mx vecdotq.cuh::get_int_b2 / qualified expert.hip::word2.
__device__ __forceinline__ std::uint32_t word2(const std::uint8_t* bytes, int index) {
    const auto* halves = reinterpret_cast<const std::uint16_t*>(bytes);
    return std::uint32_t(halves[2 * index]) |
           (std::uint32_t(halves[2 * index + 1]) << 16);
}

__global__ void quantize(const float* input, Q8_1* output, int elements, int* error) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return; // Whole width32 groups; no partial XOR sum.
    const int lane = index % 32;
    float x = input[index];
    int bad = !isfinite(x);
    if (bad) x = 0.0f;
    float amax = fabsf(x);
    for (int mask = 1; mask < 32; mask *= 2) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
        bad |= __shfl_xor(bad, mask, 32);
    }
    const float sum = sum32(x), d = amax / 127.0f;
    const auto hd = to_half(d), hs = to_half(sum);
    const bool representable = isfinite(sum) && (hs & 0x7c00) != 0x7c00 &&
                              (hd & 0x7c00) != 0x7c00 && (amax == 0.0f || hd != 0);
    const bool valid = !bad && representable;
    auto& block = output[index / 32];
    block.qs[lane] = valid && amax != 0.0f ?
                    static_cast<std::int8_t>(roundf(x / d)) : 0;
    if (lane == 0) {
        block.d = valid && amax != 0.0f ? hd : 0;
        block.s = valid && amax != 0.0f ? hs : 0;
        const int bits = (bad ? 1 : 0) | (representable ? 0 : 2);
        if (bits) atomicOr(error, bits);
    }
}

// Canonical MMVQ vdr=2, not a full block32 integer dot before scaling:
// Q4/Q5 iqs=0,2 each cover eight low + eight high nibbles (four SDOT4s);
// Q8 iqs=0,2,4,6 each cover eight contiguous signed codes (two SDOT4s).
// Integer dots are exact even for caller-supplied activation code -128.
struct Tile32 {
    std::uint32_t codes[4];
    float d, m;
};

template<TensorType Type>
__device__ __forceinline__ Tile32 load32(const std::uint8_t* block, int iqs) {
    Tile32 tile{};
    tile.d = from_half(*reinterpret_cast<const std::uint16_t*>(block));
    if constexpr (Type == TensorType::Q8_0) {
        for (int j = 0; j < 2; ++j) tile.codes[j] = word2(block + 2, iqs + j);
    } else {
        constexpr int offset = Type == TensorType::Q4_0 ? 2 :
                               Type == TensorType::Q4_1 ? 4 : 6;
        if constexpr (Type == TensorType::Q4_1)
            tile.m = from_half(*reinterpret_cast<const std::uint16_t*>(block + 2));
        std::uint32_t high = 0;
        if constexpr (Type == TensorType::Q5_0) high = word2(block + 2, 0);
        for (int j = 0; j < 2; ++j) {
            const auto packed = word2(block + offset, iqs + j);
            auto low = packed & 0x0f0f0f0fu;
            auto upper = (packed >> 4) & 0x0f0f0f0fu;
            if constexpr (Type == TensorType::Q5_0) {
                // ADAPT vec_dot_q5_0_q8_1_impl: qh bit i belongs to element i.
                const auto h = high >> (4 * (iqs + j));
                low |= (h << 4) & 0x00000010u;
                low |= (h << 11) & 0x00001000u;
                low |= (h << 18) & 0x00100000u;
                low |= (h << 25) & 0x10000000u;
                upper |= (h >> 12) & 0x00000010u;
                upper |= (h >> 5) & 0x00001000u;
                upper |= (h << 2) & 0x00100000u;
                upper |= (h << 9) & 0x10000000u;
            }
            tile.codes[2 * j] = low;
            tile.codes[2 * j + 1] = upper;
        }
    }
    return tile;
}

template<TensorType Type>
__device__ __forceinline__ float accumulate32(const Tile32& tile, const int* q8,
                                             float dx, float sx, float acc) {
    int dot = 0;
    constexpr int words = Type == TensorType::Q8_0 ? 2 : 4;
    for (int j = 0; j < words; ++j)
        dot = __builtin_amdgcn_sdot4(tile.codes[j], q8[j], dot, false);
    if constexpr (Type == TensorType::Q4_1) {
        // FAST_FP16_AVAILABLE is true on gfx906: __hmul2 in the donor rounds
        // both products to half RNE. Products of two finite halves are exact
        // in FP32 before this conversion. Split m*s equally over two fragments.
        const float dd = from_half(to_half(tile.d * dx));
        const float ms = from_half(to_half(tile.m * sx));
        // Native v_pk_mul_f16 -> inner FMA -> separate accumulator v_add_f32.
        return acc + fmaf(float(dot), dd, ms * 0.5f);
    } else if constexpr (Type == TensorType::Q8_0) {
        // vec_dot_q8_0_q8_1_impl<float>: FP32 product, NOT half-rounded.
        return fmaf(tile.d * dx, float(dot), acc);
    } else {
        constexpr float correction = Type == TensorType::Q4_0 ? 4.0f : 8.0f;
        // Raw half s is multiplied by an exact power of two; each fragment
        // subtracts half the block correction. Both native contractions matter.
        const float corrected = fmaf(float(dot), dx, -correction * sx);
        return fmaf(tile.d, corrected, acc);
    }
}

// Production GCN topology: dim3(64,2), both waves cooperate on every row.
// kbx=tid/(qi/vdr), kqs=vdr*(tid%(qi/vdr)), stride=vdr*128/qi.
// Merge wave1 into wave0 lane-wise BEFORE ascending-XOR DPP sum64; a sum of
// separately reduced waves would have a different floating-point order.
template<TensorType Type, int Columns, int Rows>
__global__ __launch_bounds__(128, 1)
void matrix32(const std::uint8_t* weights, const Q8_1* input,
              float* output, int blocks, int rows) {
    constexpr int bytes = Type == TensorType::Q4_0 ? 18 :
                          Type == TensorType::Q4_1 ? 20 :
                          Type == TensorType::Q5_0 ? 22 : 34;
    constexpr int fragments = Type == TensorType::Q8_0 ? 4 : 2;
    const int lane = threadIdx.x, tid = 64 * threadIdx.y + lane;
    const int row0 = blockIdx.x * Rows, iqs = 2 * (tid % fragments);
    float acc[Columns][Rows]{};
    for (int k = tid / fragments; k < blocks; k += 128 / fragments) {
        Tile32 tile[Rows]{};
        for (int r = 0; r < Rows; ++r) {
            if (row0 + r < rows)
                tile[r] = load32<Type>(weights +
                        (std::size_t(row0 + r) * blocks + k) * bytes, iqs);
        }
        for (int c = 0; c < Columns; ++c) {
            const auto& x = input[std::size_t(c) * blocks + k];
            const float dx = from_half(x.d), sx = from_half(x.s);
            // Q8_1 base alignment is checked as 4, and sizeof(Q8_1) is 36.
            const auto* q8 = reinterpret_cast<const int*>(x.qs);
            int codes[4];
            for (int j = 0; j < 2; ++j) {
                if constexpr (Type == TensorType::Q8_0) {
                    codes[j] = q8[iqs + j];
                } else {
                    codes[2 * j] = q8[iqs + j];
                    codes[2 * j + 1] = q8[iqs + j + 4];
                }
            }
            for (int r = 0; r < Rows; ++r)
                if (row0 + r < rows)
                    acc[c][r] = accumulate32<Type>(tile[r], codes, dx, sx, acc[c][r]);
        }
    }
    __shared__ float partial[Columns][Rows][64];
    if (threadIdx.y == 1)
        for (int c = 0; c < Columns; ++c) for (int r = 0; r < Rows; ++r)
            partial[c][r][lane] = acc[c][r];
    __syncthreads(); // All 128 threads, including idle fragments / invalid last row.
    if (threadIdx.y == 1) return;
    for (int c = 0; c < Columns; ++c) for (int r = 0; r < Rows; ++r) {
        const float value = sum64(acc[c][r] + partial[c][r][lane]);
        if (lane == r && row0 + r < rows)
            output[std::size_t(c) * rows + row0 + r] = value;
    }
}

struct Tile6 {
    std::uint32_t codes[2];
    int scales[2];
    float d;
};

__device__ __forceinline__ Tile6 load6(const std::uint8_t* block, int iqs) {
    // Canonical block: ql[128], qh[64], int8 scales[16], half d (offset208).
    // mx QI6_K=32, QR6_K=2, QI8_1=8. Each iqs represents a four-element
    // slice in two quarters 64 elements apart, not two adjacent Q8_1 blocks.
    const int half = iqs / 16, quarter = (iqs % 16) / 8;
    const int scale = half * 8 + (iqs % 16) / 4;
    const auto vl = word2(block, iqs);
    const auto vh = word2(block + 128, half * 8 + iqs % 8) >> (2 * quarter);
    const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
    Tile6 tile{};
    tile.d = from_half(*reinterpret_cast<const std::uint16_t*>(block + 208));
    for (int i = 0; i < 2; ++i) {
        const auto low = (vl >> (4 * i)) & 0x0f0f0f0fu;
        const auto high = ((vh >> (4 * i)) << 4) & 0x30303030u;
        // Exact bytewise (code-32), equivalent to donor __vsubss4 here:
        // setting each byte's high bit prevents borrow into the next byte;
        // XOR removes that bias, leaving signed int8 codes in [-32,31].
        tile.codes[i] = (((low | high) | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
        tile.scales[i] = scales[scale + 4 * i];
    }
    return tile;
}

template<int Columns, int Rows>
__global__ __launch_bounds__(128, 1)
void matrix6(const std::uint8_t* weights, const Q8_1* input,
             float* output, int blocks, int rows) {
    const int lane = threadIdx.x, tid = 64 * threadIdx.y + lane;
    const int row0 = blockIdx.x * Rows, iqs = tid % 32;
    float acc[Columns][Rows]{};
    // qi=32, vdr=1: 32 slices/block256; two waves cover four blocks/pass.
    for (int k = tid / 32; k < blocks; k += 4) {
        const int q8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        Tile6 tile[Rows]{};
        for (int r = 0; r < Rows; ++r)
            if (row0 + r < rows)
                tile[r] = load6(weights + (std::size_t(row0 + r) * blocks + k) * 210, iqs);
        for (int c = 0; c < Columns; ++c) {
            const auto* x = input + std::size_t(c) * blocks * 8 + k * 8 + q8_offset;
            int codes[2]; float dx[2];
            for (int i = 0; i < 2; ++i) {
                codes[i] = reinterpret_cast<const int*>(x[2 * i].qs)[iqs % 8];
                dx[i] = from_half(x[2 * i].d);
            }
            for (int r = 0; r < Rows; ++r) {
                if (row0 + r >= rows) continue;
                // Keep vec_dot_q6_K_q8_1_impl_mmvq grouping exactly: scale
                // multiplies the integer dot BEFORE FP32 d8, then d6 multiplies
                // the two-term sum. No half scale product and no raw-sum offset.
                float sum = 0.0f;
                for (int i = 0; i < 2; ++i) {
                    const int dot = __builtin_amdgcn_sdot4(tile[r].codes[i], codes[i], 0, false);
                    sum = fmaf(dx[i], float(dot * tile[r].scales[i]), sum);
                }
                acc[c][r] = fmaf(tile[r].d, sum, acc[c][r]);
            }
        }
    }
    __shared__ float partial[Columns][Rows][64];
    if (threadIdx.y == 1)
        for (int c = 0; c < Columns; ++c) for (int r = 0; r < Rows; ++r)
            partial[c][r][lane] = acc[c][r];
    __syncthreads();
    if (threadIdx.y == 1) return;
    for (int c = 0; c < Columns; ++c) for (int r = 0; r < Rows; ++r) {
        const float value = sum64(acc[c][r] + partial[c][r][lane]);
        if (lane == r && row0 + r < rows)
            output[std::size_t(c) * rows + row0 + r] = value;
    }
}

template<TensorType Type, int Columns, int Rows>
void enqueue_rows(QuantizedDeviceMatrix matrix, const Q8_1* input, float* output,
                  hipStream_t stream) noexcept {
    const dim3 grid((matrix.output + Rows - 1) / Rows), threads(64, 2);
    const auto* weights = static_cast<const std::uint8_t*>(matrix.weights);
    if constexpr (Type == TensorType::Q6_K)
        matrix6<Columns, Rows><<<grid, threads, 0, stream>>>(weights, input, output, matrix.input / 256, matrix.output);
    else
        matrix32<Type, Columns, Rows><<<grid, threads, 0, stream>>>(weights, input, output, matrix.input / 32, matrix.output);
}

template<TensorType Type, int Columns>
void enqueue_type(QuantizedDeviceMatrix matrix, const Q8_1* input, float* output,
                  hipStream_t stream) noexcept {
    if constexpr (Columns == 1) {
        // mmvq.cu::should_use_small_k, strict '<', for these five GCN types:
        // Q4/Q5 K<2048, Q8 K<1024, Q6 K<1024 use two rows/block; else one.
        constexpr int elements = Type == TensorType::Q6_K ? 256 : 32;
        constexpr int blocks_per_iter = Type == TensorType::Q6_K ? 4 :
                                        Type == TensorType::Q8_0 ? 32 : 64;
        if (matrix.input / elements < blocks_per_iter)
            enqueue_rows<Type, Columns, 2>(matrix, input, output, stream);
        else
            enqueue_rows<Type, Columns, 1>(matrix, input, output, stream);
    } else {
        enqueue_rows<Type, Columns, 2>(matrix, input, output, stream);
    }
}

template<int Columns>
void enqueue(QuantizedDeviceMatrix matrix, const Q8_1* input, float* output,
             hipStream_t stream) noexcept {
    switch (matrix.type) {
    case TensorType::Q4_0:
        enqueue_type<TensorType::Q4_0, Columns>(matrix, input, output, stream);
        break;
    case TensorType::Q4_1:
        enqueue_type<TensorType::Q4_1, Columns>(matrix, input, output, stream);
        break;
    case TensorType::Q5_0:
        enqueue_type<TensorType::Q5_0, Columns>(matrix, input, output, stream);
        break;
    case TensorType::Q8_0:
        enqueue_type<TensorType::Q8_0, Columns>(matrix, input, output, stream);
        break;
    case TensorType::Q6_K:
        enqueue_type<TensorType::Q6_K, Columns>(matrix, input, output, stream);
        break;
    default: break; // Unsupported types were rejected before enqueue.
    }
}

} // namespace

hipError_t launch_quantized_linear(QuantizedDeviceMatrix matrix, const Q8_1* input,
                                  int columns, float* output, hipStream_t stream) noexcept {
    int block_elements = 0, block_bytes = 0;
    if (!layout(matrix.type, block_elements, block_bytes)) return hipErrorNotSupported;
    const bool lm_head = matrix.type == TensorType::Q6_K &&
                        matrix.input == 2560 && matrix.output == 248320;
    if (!dimension(matrix.input) || (!dimension(matrix.output) && !lm_head) ||
        columns <= 0 || columns > linear_device_max_columns || matrix.input % block_elements)
        return hipErrorInvalidValue;
    // Products fit size_t32: ordinary Q8_0 <=285212672 bytes; the sole larger
    // shape is the actual Q6_K LM head, 248320*10*210=521472000 bytes.
    const auto rows = static_cast<std::size_t>(matrix.output);
    const auto width = static_cast<std::size_t>(matrix.input);
    const auto cols = static_cast<std::size_t>(columns);
    const auto weight_bytes = rows * (width / block_elements) * block_bytes;
    const auto input_bytes = cols * (width / 32) * sizeof(Q8_1);
    const auto output_bytes = cols * rows * sizeof(float);
    Range weights_range{}, input_range{}, output_range{};
    if (!range(matrix.weights, weight_bytes, 2, weights_range) ||
        !range(input, input_bytes, 4, input_range) ||
        !range(output, output_bytes, 4, output_range) ||
        overlaps(output_range, weights_range) || overlaps(output_range, input_range))
        return hipErrorInvalidValue;
    if (columns == 1) enqueue<1>(matrix, input, output, stream);
    else if (columns == 2) enqueue<2>(matrix, input, output, stream);
    else enqueue<3>(matrix, input, output, stream);
    return hipGetLastError();
}

hipError_t launch_quantize_q8_1(const float* input, Q8_1* output, int width,
                               int columns, int* error, hipStream_t stream) noexcept {
    if (!dimension(width) || !dimension(columns) || width % 32) return hipErrorInvalidValue;
    // <=2^28 elements; all byte products fit size_t32 and ptrdiff_t32.
    const auto elements = static_cast<std::size_t>(width) * static_cast<std::size_t>(columns);
    Range input_range{}, output_range{}, error_range{};
    if (!range(input, elements * sizeof(float), 4, input_range) ||
        !range(output, (elements / 32) * sizeof(Q8_1), 4, output_range) ||
        !range(error, sizeof(int), 4, error_range) ||
        overlaps(output_range, input_range) || overlaps(error_range, input_range) ||
        overlaps(error_range, output_range))
        return hipErrorInvalidValue;
    const int count = static_cast<int>(elements);
    quantize<<<(count + 255) / 256, 256, 0, stream>>>(input, output, count, error);
    return hipGetLastError();
}

} // namespace qwen
