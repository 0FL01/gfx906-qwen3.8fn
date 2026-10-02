#pragma once

#include "linear.cuh"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qwen {

inline constexpr int mmq_device_max_dimension = 16384;
inline constexpr int mmq_device_max_columns = 128;

// DS4 activation ABI, NOT a cast of four interleaved 36-byte Q8_1 blocks.
// ds = (d0,s0,d1,s1,d2,s2,d3,s3), original half bits; qs contains the four
// original block32 code arrays in order. s is the RAW float-input sum from
// quant.hpp, not a sum of int8 codes. Storage is [k128_block][column]:
// input[(block32/4)*columns+column], sub-block = block32%4.
struct alignas(16) Q8MmqBlock {
    std::uint16_t ds[8];
    std::int8_t qs[128];
};
static_assert(std::is_standard_layout_v<Q8MmqBlock> &&
              std::is_trivially_copyable_v<Q8MmqBlock>);
static_assert(alignof(Q8MmqBlock) == 16 && sizeof(Q8MmqBlock) == 144 &&
              offsetof(Q8MmqBlock, ds) == 0 && offsetof(Q8MmqBlock, qs) == 16);

// Borrowed device buffers, caller-owned capacities/residency/lifetimes and
// exclusive access ordered on the explicit stream, including nullptr when
// deliberately selecting HIP's default stream. Current device must be gfx906.
// Neither launch allocates, queries devices/streams, copies through a hidden
// stream, synchronizes, or inspects numeric validity. Trust finite static
// weights/Q8 values as in linear.cuh. Success means enqueued, not completed.
// Before any HIP call: invalid shapes, null/misaligned/wrapping ranges,
// size overflow or writable overlap return hipErrorInvalidValue. Unsupported
// matrix types return hipErrorNotSupported. Valid calls return hipGetLastError()
// after the kernel launch. Readable ranges may alias; output must be disjoint.
// Build mmq.hip for gfx906 with -ffp-contract=off and without fast-math.

// width positive <=16384, divisible by32; columns=1..128.
// input[column*(width/32)+block32] needs columns*(width/32)*36 bytes, aligned4.
// output[k128_block*columns+column] needs ceil(width/128)*columns*144 bytes,
// aligned16. ONLY rearranges original half d/s and int8 bytes, without a new
// quantization or sum. Missing block32s in the owned last K128 block are zeroed
// (both ds entries and all32 code bytes); no source padding may be read.
[[nodiscard]] hipError_t launch_q8_1_to_mmq(const Q8_1* input, int width,
        int columns, Q8MmqBlock* output, hipStream_t stream) noexcept;

// Canonical unchanged Q4_0/Q4_1 weights, contiguous [row][K/32], no padding.
// matrix.input and matrix.output positive <=16384, K divisible by32;
// columns=1..128. Required capacities/alignment:
//   weights: matrix.output*(K/32)*(18 or20) bytes, alignment2;
//   input:   ceil(K/128)*columns*144 bytes, alignment16;
//   output:  columns*matrix.output floats, alignment4.
// Output is overwritten in column-major [column][row], with no bias/alpha/beta.
// No external workspace. All row/column/K256 tails are checked and zero-filled
// into owned LDS; even unused block32 contents of the last input K128 are masked.
//
// GCN DP4A: I64, J8/16/32/64, dim3(64,4), K256, no stream-K.
// For rows<=640 use J8 column microtiles to expose more independent workgroups;
// otherwise select the smallest listed J >= columns, capped at64.
// Lane owns one row, wave owns columns wave+4*t. A weight tile is shared across
// columns via LDS, activation words via LDS/quad DPP; no cross-lane FP sum.
// Ascending full-block32 dots (eight signed SDOT4 with unsigned Q4 nibbles):
//   Q4_0: corrected=fmaf(float(dot),d8,-8*s8); acc=fmaf(d4,corrected,acc).
//   Q4_1: dd=half_RNE(d4*d8), ms=half_RNE(m4*s8), once per full block;
//         acc += fmaf(float(dot),dd,ms).
// All other FP operations are uncontracted. This explicitly retains donor
// contractions but differs from qualified MMVQ's two fragments and wave/DPP
// reduction. Common-Q8/unchanged linear and full-model gates must qualify it.
[[nodiscard]] hipError_t launch_mmq_q4(QuantizedDeviceMatrix matrix,
        const Q8MmqBlock* input, int columns, float* output,
        hipStream_t stream) noexcept;

// Literal gfx906 dispatch, shared with diagnostic metadata (not autotuning).
inline constexpr int mmq_q4_tile_j(int rows, int columns) noexcept {
    return rows <= 640 ? 8 : columns <= 8 ? 8 : columns <= 16 ? 16 : columns <= 32 ? 32 : 64;
}

} // namespace qwen
