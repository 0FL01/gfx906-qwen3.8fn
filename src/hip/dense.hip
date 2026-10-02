#include "dense.cuh"
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
    const float alpha = 1.0f, beta = 0.0f;
    // GGUF [K,M] is a column-major K-by-M matrix, so A^T is M-by-K.
    return rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
        matrix.output, columns, matrix.input, &alpha, matrix.weights, matrix.input,
        input, matrix.input, &beta, output, matrix.output);
}
} // namespace qwen
