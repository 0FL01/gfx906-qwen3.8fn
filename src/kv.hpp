#pragma once
#include "quant.hpp"
#include <cstddef>
#include <span>

namespace qwen {
// ADAPT mx cpy-utils.cuh::quantize_f32_q4_0_block, dcd685463d597d31f5ca759d32c94592a2740fa4.
// First maximum-absolute element wins ties. d = signed_max/-8, including -0
// for a zero block. Codes use original FP32 reciprocal, trunc(x*id+8.5),
// clamp at 15, low nibble first 16 / high nibble last 16; stored d is FP16 RNE.
// Finite inputs and representable nonzero scales required. Checks precede writes.
void quantize_q4(std::span<const float> input, std::span<Q4_0> output);
void dequantize_q4(std::span<const Q4_0> input, std::span<float> output);
// PORT mx fwht.cu: normalized Walsh-Hadamard, scale BEFORE ascending butterfly.
// Contiguous groups of 64 (V), 128 (index experiments), or 256 (Q/K).
// Exact in-place use supported; partial overlap rejected. H is its own inverse.
void hadamard(std::span<const float> input, std::span<float> output, int group);
}
