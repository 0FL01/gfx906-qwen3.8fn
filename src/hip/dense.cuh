#pragma once
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

namespace qwen {
// Borrowed unchanged F32 values, including exact BF16->F32 conversion at load.
// Canonical [output][input] storage; contiguous column-major token activations.
struct DenseDeviceMatrix {
    const float* weights = nullptr;
    int input = 0;
    int output = 0;
};
// ADAPT: production MMVF for pair-aligned even K at N1..8; KEEP R0-qualified
// rocBLAS SGEMM at N9..128 and other valid geometry/alignment, not a GEMM library.
// Dimensions 1..16384, columns 1..128. Capacities: input*output weight floats,
// input*columns activation floats and output*columns destination floats.
// All pointers must be float-aligned, nonwrapping; output disjoint from reads.
// Geometry/ranges/null handle are rejected BEFORE calling rocBLAS. Contents,
// allocation capacities/device residency and serialization belong to caller.
// Caller exclusively owns a live handle. This sets its stream and HOST pointer
// mode, then enqueues MMVF or alpha=1,beta=0 transpose-A SGEMM; no allocations,
// copies, synchronization or device queries. rocBLAS may initialize its own
// workspace on first use: warm the handle outside performance measurements.
// Success means enqueue success, not completion or numerical validation.
[[nodiscard]] rocblas_status launch_dense_linear(rocblas_handle handle,
    DenseDeviceMatrix matrix, const float* input, int columns, float* output,
    hipStream_t stream) noexcept;

// Canonical MMVF arithmetic at logical N1..128 without the SGEMM fallback.
// Requires even input width and float2-aligned weights/activations; output is
// float-aligned. Full N8 microtiles share grid.y; the final N1..7 tile uses the
// existing kernel. Full logical ranges are validated before handle mutation or
// enqueue. Other ownership/lifetime contracts above apply. Old API unchanged.
[[nodiscard]] rocblas_status launch_dense_linear_tiled(rocblas_handle handle,
    DenseDeviceMatrix matrix, const float* input, int columns, float* output,
    hipStream_t stream) noexcept;
} // namespace qwen
