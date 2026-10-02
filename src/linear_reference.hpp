#pragma once
#include "quant.hpp"

namespace qwen {
// Common-Q8 oracle for canonical Q4_0/Q4_1/Q5_0/Q8_0/Q6_K weights.
// Weight rows are contiguous; input/output tokens are column-major. Positive
// dimensions, input divisible by 32 (256 for Q6_K), exactly sized spans, N=1..3.
// Q8_1 is the existing 36-byte ABI with a half RAW input sum, not d*sum(qs).
// Throws invalid_argument for shape/range/alias/nonfinite/arithmetic failures,
// leaving all output bytes unchanged. Input and weights may overlap each other,
// but output may overlap neither. No threads or successful-call allocation.
// Caller keeps readable storage immutable throughout both preflight/publication
// passes. Requires IEEE FP32, round-to-nearest, and -ffp-contract=off (no fast-math).
// Q6 follows MMVQ four-element pair grouping; scalar ascending accumulation is
// an oracle for bounded-error GPU comparisons, not a bitwise parallel reduction.
void matmul_q8_reference(const QMatrix& matrix, std::span<const Q8_1> input,
                         int columns, std::span<float> output);
}
