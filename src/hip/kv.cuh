#pragma once
#include "quant.hpp"
#include <hip/hip_runtime.h>

namespace qwen {
// Borrowed contiguous device buffers on the supplied stream. No allocations,
// host synchronization or hidden default stream. Caller keeps buffers alive.
// Sticky error flag: 1 nonfinite input, 2 unrepresentable Q4 scale.
void launch_quantize_q4(const float *input,Q4_0 *output,int elements,int *error,hipStream_t stream);
void launch_dequantize_q4(const Q4_0 *input,float *output,int elements,hipStream_t stream);
void launch_hadamard(const float *input,float *output,int elements,int group,hipStream_t stream);
}
