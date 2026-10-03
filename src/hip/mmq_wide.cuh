#pragma once

#include "mmq.cuh"

namespace qwen {

// Canonical Q4_0/Q4_1/Q5_0/Q8_0/Q6_K wide projection on gfx906.
// KEEP the qualified 144-byte DS4 producer and borrowed explicit stream.
// Q4 forwards to launch_mmq_q4; other types use specialized donor LDS loaders.
// N=1..128, ordinary K/M=1..16384, K divisible by the weight block size.
// The sole large-output exception is Q6_K [K=2560,M=248320].
// Capacity/alignment: unchanged canonical weights alignment2; DS4 input
// ceil(K/128)*N blocks alignment16; output N*M floats alignment4.
// All partial row/column/K256 tiles are masked into owned storage; no neighbour
// weight/cache-slot padding. Values/scale bits must be finite; caller owns
// residency, allocation bounds, serialization and lifetime. No allocation,
// synchronization, device query, hidden stream, or activation requantization.
// Invalid host geometry/ranges return HIP errors before enqueue; results need
// completion and unchanged common-Q8/full-model qualification, not bit identity.
[[nodiscard]] hipError_t launch_mmq_linear(QuantizedDeviceMatrix matrix,
        const Q8MmqBlock* input, int columns, float* output,
        hipStream_t stream) noexcept;

} // namespace qwen
