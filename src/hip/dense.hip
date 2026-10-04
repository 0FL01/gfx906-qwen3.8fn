#include "dense.cuh"
#include "dpp.cuh"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace qwen {
namespace {
struct Range { std::uintptr_t begin = 0, end = 0; };
bool range(const void* pointer, std::size_t floats, Range& r) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto bytes = floats * sizeof(float); // bounded dimensions below
    if (!address || address % alignof(float) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
    r = {address, address + bytes};
    return true;
}
bool overlaps(Range a, Range b) noexcept { return a.begin < b.end && b.begin < a.end; }
bool dimension(int d) noexcept { return d > 0 && d <= 16384; }

// ADAPT: mx dcd685463d597d31f5ca759d32c94592a2740fa4,
// ggml/src/ggml-cuda/mmvf.cu::mul_mat_vec_f and common.cuh block_reduce_sum.
// Copyright (c) 2023-2026 The ggml authors; MIT, third_party/mx-LICENSE.
// Exact BF16->F32 load conversion has already happened outside this API.
// The actual production binary uses low-then-high FMAC and two-stage padded
// wave reduction. Express contraction explicitly, not with global fast-math.
// KEEP rocBLAS for N9..128, odd K and valid float-aligned/non-pair-aligned inputs.
template<int Columns, int Threads, bool Tiled = false>
__global__ __launch_bounds__(Threads)
void mmvf(const float2* weights, const float2* input, int pairs, int rows,
          float* output) {
    if constexpr (Tiled) {
        const auto first = std::size_t(blockIdx.y) * Columns;
        input += first * pairs;
        output += first * rows;
    }
    const int tid = threadIdx.x, lane = tid % 64;
    const auto* row = weights + std::size_t(blockIdx.x) * pairs;
    float acc[Columns]{};
    for (int p = tid; p < pairs; p += Threads) {
        const float2 w = row[p];
        for (int c = 0; c < Columns; ++c) {
            const float2 x = input[std::size_t(c) * pairs + p];
            acc[c] = fmaf(w.x, x.x, acc[c]);
            acc[c] = fmaf(w.y, x.y, acc[c]);
        }
    }
    if constexpr (Threads == 128) {
        __shared__ float partial[64];
        if (tid < 64) partial[tid] = 0.0f;
        __syncthreads();
        for (int c = 0; c < Columns; ++c) {
            const float wave = sum64(acc[c]);
            if (lane == 0) partial[tid / 64] = wave;
            __syncthreads();
            if (tid < 64) acc[c] = sum64(partial[lane]);
            __syncthreads(); // Protect partial reuse for the next token.
        }
    } else {
        for (int c = 0; c < Columns; ++c) acc[c] = sum64(acc[c]);
    }
    if (tid < Columns)
        output[std::size_t(tid) * rows + blockIdx.x] = acc[tid];
}
template<int Columns, bool Tiled = false>
void enqueue_mmvf(DenseDeviceMatrix matrix, const float* input, float* output,
                  hipStream_t stream, int tiles = 1) noexcept {
    const auto* w = reinterpret_cast<const float2*>(matrix.weights);
    const auto* x = reinterpret_cast<const float2*>(input);
    if (matrix.input <= 128)
        mmvf<Columns, 64, Tiled><<<dim3(matrix.output, tiles), 64, 0, stream>>>(w, x, matrix.input / 2, matrix.output, output);
    else
        mmvf<Columns, 128, Tiled><<<dim3(matrix.output, tiles), 128, 0, stream>>>(w, x, matrix.input / 2, matrix.output, output);
}
} // namespace

rocblas_status launch_dense_linear(rocblas_handle handle, DenseDeviceMatrix matrix,
    const float* input, int columns, float* output, hipStream_t stream) noexcept {
    if (!handle) return rocblas_status_invalid_handle;
    if (!dimension(matrix.input) || !dimension(matrix.output) || columns < 1 || columns > 128)
        return rocblas_status_invalid_size;
    const auto k = static_cast<std::size_t>(matrix.input);
    const auto m = static_cast<std::size_t>(matrix.output);
    const auto n = static_cast<std::size_t>(columns);
    Range weights{}, x{}, y{};
    if (!range(matrix.weights, k * m, weights) || !range(input, k * n, x) ||
        !range(output, m * n, y) || overlaps(y, weights) || overlaps(y, x))
        return rocblas_status_invalid_pointer;
    auto status = rocblas_set_stream(handle, stream);
    if (status != rocblas_status_success) return status;
    status = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    if (status != rocblas_status_success) return status;
    if (columns <= 8 && matrix.input % 2 == 0 &&
        reinterpret_cast<std::uintptr_t>(matrix.weights) % alignof(float2) == 0 &&
        reinterpret_cast<std::uintptr_t>(input) % alignof(float2) == 0) {
        if (columns == 1) enqueue_mmvf<1>(matrix, input, output, stream);
        else if (columns == 2) enqueue_mmvf<2>(matrix, input, output, stream);
        else if (columns == 3) enqueue_mmvf<3>(matrix, input, output, stream);
        else if (columns == 4) enqueue_mmvf<4>(matrix, input, output, stream);
        else if (columns == 5) enqueue_mmvf<5>(matrix, input, output, stream);
        else if (columns == 6) enqueue_mmvf<6>(matrix, input, output, stream);
        else if (columns == 7) enqueue_mmvf<7>(matrix, input, output, stream);
        else enqueue_mmvf<8>(matrix, input, output, stream);
        return hipGetLastError() == hipSuccess ? rocblas_status_success : rocblas_status_internal_error;
    }
    const float alpha = 1.0f, beta = 0.0f;
    // GGUF [K,M] is a column-major K-by-M matrix, so A^T is M-by-K.
    return rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
        matrix.output, columns, matrix.input, &alpha, matrix.weights, matrix.input,
        input, matrix.input, &beta, output, matrix.output);
}
rocblas_status launch_dense_linear_tiled(rocblas_handle handle, DenseDeviceMatrix matrix,
    const float* input, int columns, float* output, hipStream_t stream) noexcept {
    if (!handle) return rocblas_status_invalid_handle;
    if (!dimension(matrix.input) || !dimension(matrix.output) || matrix.input % 2 != 0 ||
        columns < 1 || columns > 128) return rocblas_status_invalid_size;
    const auto k = static_cast<std::size_t>(matrix.input);
    const auto m = static_cast<std::size_t>(matrix.output);
    const auto n = static_cast<std::size_t>(columns);
    Range weights{}, x{}, y{};
    if (!range(matrix.weights, k * m, weights) || !range(input, k * n, x) ||
        !range(output, m * n, y) || overlaps(y, weights) || overlaps(y, x) ||
        reinterpret_cast<std::uintptr_t>(matrix.weights) % alignof(float2) != 0 ||
        reinterpret_cast<std::uintptr_t>(input) % alignof(float2) != 0)
        return rocblas_status_invalid_pointer;
    if (columns <= 8) return launch_dense_linear(handle, matrix, input, columns, output, stream);
    auto status = rocblas_set_stream(handle, stream);
    if (status != rocblas_status_success) return status;
    status = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    if (status != rocblas_status_success) return status;
    // KEEP production MMVF sums, FMAs and barriers; independent N8 tiles supply
    // more workgroups for small HC/router matrices without a wider reduction.
    const int tiles = columns / 8;
    enqueue_mmvf<8, true>(matrix, input, output, stream, tiles);
    if (hipGetLastError() != hipSuccess) return rocblas_status_internal_error;
    const int first = tiles * 8, tail = columns - first;
    if (tail == 0) return rocblas_status_success;
    return launch_dense_linear(handle, matrix, input + std::size_t(first) * k,
        tail, output + std::size_t(first) * m, stream);
}
} // namespace qwen
