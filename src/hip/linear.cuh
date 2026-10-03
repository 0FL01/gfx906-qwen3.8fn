#pragma once

#include "quant.hpp"
#include <hip/hip_runtime_api.h>

namespace qwen {

inline constexpr int linear_device_max_dimension = 16384;
inline constexpr int linear_device_max_columns = 3;
inline constexpr int linear_device_short_max_columns = 8;

// Borrowed canonical GGML weights: contiguous [output][input/block_elements]
// blocks, without padding or repack. input/output are element dimensions.
// Supported types (block_elements / block_bytes): Q4_0 32/18, Q4_1 32/20,
// Q5_0 32/22, Q8_0 32/34, Q6_K 256/210. No dense-type fallback.
struct QuantizedDeviceMatrix {
    TensorType type = TensorType::Q4_0;
    const void* weights = nullptr;
    int input = 0;
    int output = 0;
};

// All launches use the caller's current gfx906 device and explicitly supplied
// raw stream (including nullptr if the caller deliberately selects HIP's default
// stream). No allocation, device/stream query, copy, synchronization or exception.
// Caller owns allocation capacities, residency, lifetimes and ordering against
// uploads/other streams; pointer validation does not establish those properties.
// All dimensions are positive; zero-sized calls are rejected, not no-ops.
// Bad shapes, null/misaligned/wrapping pointers or writable overlap return
// hipErrorInvalidValue BEFORE any HIP call. Unsupported weight types return
// hipErrorNotSupported. Otherwise return hipGetLastError() after enqueue;
// success is not a completion guarantee or a check of tensor contents.
// Compile linear.hip for gfx906 without fast-math, with -ffp-contract=off.

// input/output dimensions <=16384, plus the actual Q6_K LM head [2560,248320];
// columns=1/2/3; input must be divisible by
// the weight type's block_elements. Tokens are contiguous columns:
// input[column*(matrix.input/32)+block], output[column*matrix.output+row].
// Required capacities: weights = output*(input/block_elements)*block_bytes;
// activations = columns*(input/32)*sizeof(Q8_1); output = columns*output floats.
// Weights need 2-byte alignment, Q8_1 activations and output 4-byte alignment.
// Output must be disjoint from both readable ranges; readable ranges may alias.
// No additional workspace. Output is overwritten (no bias, alpha or beta).
//
// Arithmetic uses the 36-byte Q8_1 ABI in quant.hpp, not dequantized F32 GEMM:
// Q4/Q5 use two block32 fragments (16 codes each): explicit inner FMA applies
// half the raw-sum correction (4*s8 / 8*s8), then weight-scale/accumulator FMA.
// Q4_1 uses half_RNE(d4*d8) and half_RNE(m4*s8), inner FMA with half the offset,
// then a separate accumulator add. Q8 uses four 8-code fragments with FP32
// d0*d8 and accumulator FMA. Q6_K retains two-word slices: integer dot*scale
// before FP32 conversion, explicit pair FMAs then weight-scale/accumulator FMA.
// Two cooperating wave64s (128 threads) merge lane-wise through LDS BEFORE
// ascending-XOR DPP sum64, unlike the ascending CPU row loop.
// These expressions intentionally differ from dequantize-then-F32-dot, notably
// raw-sum corrections and Q4_1 half products; compare with a common-Q8 oracle.
// Input half scales/sums and weight scales must already be valid/finite; this
// launch does not inspect numeric validity or the quantizer's sticky error flag.
[[nodiscard]] hipError_t launch_quantized_linear(QuantizedDeviceMatrix matrix,
        const Q8_1* input, int columns, float* output, hipStream_t stream) noexcept;

// Separate checked short-column seam, N=1..8. All other shape/type/range,
// alignment/overlap, arithmetic and lifetime contracts above are identical.
// Instantiates the same MMVQ templates, reusing weight registers across columns;
// no per-column launches, DS4 conversion or additional workspace. The existing
// launch_quantized_linear contract remains N=1..3 and still rejects N=4.
[[nodiscard]] hipError_t launch_quantized_linear_short(QuantizedDeviceMatrix matrix,
        const Q8_1* input, int columns, float* output, hipStream_t stream) noexcept;

// Reusable float -> canonical Q8_1 workspace producer, also usable for prefill:
// width is 32-divisible, width/columns are 1..16384 (linear N<=3, short N<=8).
// input[column*width+i]; output[column*(width/32)+block]. Required capacities:
// width*columns floats, (width/32)*columns Q8_1 blocks, one device int error.
// All three pointers need 4-byte alignment. Both writable ranges (output/error)
// must be disjoint from input and each other. No additional workspace.
// Codes use roundf(x/(amax/127)) BEFORE half RNE scale rounding; s is half RNE
// of the RAW float sum with ascending XOR stages 1/2/4/8/16. Zero blocks produce
// positive-zero d/s and zero codes, including all-negative-zero inputs.
//
// Caller initializes error=0 in stream order and checks it AFTER completion
// before consuming any dependent results. launch never clears or gates on it.
// Stored half d=0 is valid when original FP32 d>0: retain codes and raw sum.
// Sticky bits: 1 = nonfinite source, 2 = original FP32 scale zero for nonzero
// input, unsafe rounded code, or scale/raw-sum rounding to half infinity/NaN.
// A failing 32-value block is
// written as all-zero Q8_1; other blocks still quantize. Failure is per block.
// Reusing buffers after failure requires the caller to reset/check error again.
[[nodiscard]] hipError_t launch_quantize_q8_1(const float* input, Q8_1* output,
        int width, int columns, int* error, hipStream_t stream) noexcept;

} // namespace qwen
