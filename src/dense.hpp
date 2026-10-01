#pragma once

#include "quant.hpp"

#include <cstddef>
#include <span>

namespace qwen {

// Read one element from complete canonical little-endian GGML blocks. The span
// must contain no partial/trailing block; element is a flat, zero-based index.
// Unaligned bytes are supported. F32/BF16 preserve IEEE bits, including NaNs;
// F16 uses half_to_float (which quiets signaling NaNs). No weight conversion,
// repacking, or activation quantization is performed.
[[nodiscard]] float tensor_element(TensorType type, std::span<const std::byte> data,
                                   std::size_t element);

// Numerical CPU oracle: matrix rows are contiguous canonical GGML blocks;
// input[column * matrix.input + k], output[column * matrix.output + row].
// Uses raw FP32 input and ascending-k scalar FP32 products/additions, starting
// at +0. Compile with -ffp-contract=off and without fast-math for this contract.
// Dimensions must be positive, each row block-divisible, and spans exactly
// sized. Output must not overlap input or weights (read-only overlap is OK).
// Invalid geometry, nonfinite input/weights, and nonfinite projection results
// throw std::invalid_argument BEFORE any output writes. A checked dry run then
// an identical write pass provides that guarantee without scratch allocations.
// No threads or per-call allocations on success. Caller must keep input/weights
// immutable for the whole call; existing output values need not be finite.
void matmul_f32(const QMatrix& matrix, std::span<const float> input,
                int columns, std::span<float> output);

} // namespace qwen
