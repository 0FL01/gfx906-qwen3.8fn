#include "dense.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// WRITE OURS: original byte-indexed CPU decoder, no donor source imported.
// Format/arithmetic reference: ggml (MIT), mx-llama.cpp revision
// dcd685463d597d31f5ca759d32c94592a2740fa4, ggml/src/ggml-quants.c,
// dequantize_row_q4_0/q4_1/q5_0/q8_0/q6_K. In particular Q6_K is
// ql[128], qh[64], signed scales[16], half d: 256 elements / 210 bytes.
// This oracle does not implement the donor's Q8 activation/dot-product ABI.

namespace qwen {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(message);
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        invalid("dense: size overflow");
    return a * b;
}

TypeLayout layout_for(TensorType type) {
    // The loader throws runtime_error for unknown on-disk types; this API's
    // caller-argument failures consistently use invalid_argument instead.
    switch (type) {
    case TensorType::F32: case TensorType::F16: case TensorType::BF16:
    case TensorType::Q4_0: case TensorType::Q4_1: case TensorType::Q5_0:
    case TensorType::Q8_0: case TensorType::Q6_K:
        return type_layout(type);
    }
    invalid("dense: unsupported tensor type");
}

unsigned byte(const std::byte* p) {
    return std::to_integer<unsigned>(*p);
}

std::uint16_t u16(const std::byte* p) {
    return static_cast<std::uint16_t>(byte(p) | (byte(p + 1) << 8));
}

std::uint32_t u32(const std::byte* p) {
    return std::uint32_t(byte(p)) | (std::uint32_t(byte(p + 1)) << 8) |
           (std::uint32_t(byte(p + 2)) << 16) | (std::uint32_t(byte(p + 3)) << 24);
}

int signed_byte(const std::byte* p) {
    const int value = static_cast<int>(byte(p));
    return value < 128 ? value : value - 256;
}

float decode(TensorType type, const std::byte* block, std::size_t i) {
    switch (type) {
    case TensorType::F32: return std::bit_cast<float>(u32(block));
    case TensorType::F16: return half_to_float(u16(block));
    case TensorType::BF16: return std::bit_cast<float>(std::uint32_t(u16(block)) << 16);
    case TensorType::Q4_0: {
        const int code = static_cast<int>((byte(block + 2 + i % 16) >> (4 * (i / 16))) & 15);
        return float(code - 8) * half_to_float(u16(block));
    }
    case TensorType::Q4_1: {
        const int code = static_cast<int>((byte(block + 4 + i % 16) >> (4 * (i / 16))) & 15);
        return float(code) * half_to_float(u16(block)) + half_to_float(u16(block + 2));
    }
    case TensorType::Q5_0: {
        const unsigned low = (byte(block + 6 + i % 16) >> (4 * (i / 16))) & 15;
        // qh is a little-endian 32-bit bitplane: bit i belongs to element i.
        const unsigned high = ((byte(block + 2 + i / 8) >> (i % 8)) & 1) << 4;
        return float(static_cast<int>(low | high) - 16) * half_to_float(u16(block));
    }
    case TensorType::Q8_0:
        return float(signed_byte(block + 2 + i)) * half_to_float(u16(block));
    case TensorType::Q6_K: {
        const std::size_t half = i / 128, quarter = (i % 128) / 32, lane = i % 32;
        const unsigned low = (byte(block + half * 64 + (quarter % 2) * 32 + lane) >>
                              (4 * (quarter / 2))) & 15;
        const unsigned high = ((byte(block + 128 + half * 32 + lane) >> (2 * quarter)) & 3) << 4;
        const int code = static_cast<int>(low | high) - 32;
        // Canonical ggml order is (d * signed_scale) * signed_code, not
        // d * (scale * code), and not FP16-rounded scale multiplication.
        return (half_to_float(u16(block + 208)) * float(signed_byte(block + 192 + i / 16))) *
               float(code);
    }
    }
    invalid("dense: unsupported tensor type");
}

struct Range { std::uintptr_t begin, end; };

Range range(const void* p, std::size_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(p);
    if (p == nullptr || bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        invalid("dense: invalid address range");
    return {begin, begin + bytes};
}

bool overlaps(Range a, Range b) {
    return a.begin < b.end && b.begin < a.end;
}

struct Shape {
    std::size_t width, rows, columns, row_bytes;
    TypeLayout layout;
};

Shape validate(const QMatrix& matrix, std::span<const float> input,
               int columns, std::span<float> output) {
    const auto layout = layout_for(matrix.type);
    if (matrix.input <= 0 || matrix.output <= 0 || columns <= 0 ||
        static_cast<std::size_t>(matrix.input) % layout.block_elements != 0)
        invalid("dense: dimensions must be positive and input block-divisible");
    const auto width = static_cast<std::size_t>(matrix.input);
    const auto rows = static_cast<std::size_t>(matrix.output);
    const auto cols = static_cast<std::size_t>(columns);
    const auto row_bytes = multiply(width / layout.block_elements, layout.block_bytes);
    const auto weight_bytes = multiply(row_bytes, rows);
    const auto input_count = multiply(width, cols), output_count = multiply(rows, cols);
    const auto input_bytes = multiply(input_count, sizeof(float));
    const auto output_bytes = multiply(output_count, sizeof(float));
    if (matrix.weights.size() != weight_bytes || input.size() != input_count ||
        output.size() != output_count)
        invalid("dense: matrix span size mismatch");
    const auto weights_range = range(matrix.weights.data(), weight_bytes);
    const auto input_range = range(input.data(), input_bytes);
    const auto output_range = range(output.data(), output_bytes);
    if (overlaps(output_range, weights_range) || overlaps(output_range, input_range))
        invalid("dense: output overlaps input or weights");
    for (const float value : input)
        if (!std::isfinite(value)) invalid("dense: nonfinite activation");
    return {width, rows, cols, row_bytes, layout};
}

template<bool Check>
float projection(TensorType type, const std::byte* row, const float* input, Shape shape) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float sum = 0.0f;
    for (std::size_t k = 0; k < shape.width; ++k) {
        const auto* block = row + (k / shape.layout.block_elements) * shape.layout.block_bytes;
        const float weight = decode(type, block, k % shape.layout.block_elements);
        if constexpr (Check)
            if (!std::isfinite(weight)) invalid("dense: nonfinite weight");
        const float product = weight * input[k];
        sum += product;
    }
    if constexpr (Check)
        if (!std::isfinite(sum)) invalid("dense: nonfinite projection");
    return sum;
}

} // namespace

float tensor_element(TensorType type, std::span<const std::byte> data, std::size_t element) {
    const auto layout = layout_for(type);
    if (data.size() % layout.block_bytes != 0)
        invalid("dense: tensor span contains a partial block");
    const auto block_index = element / layout.block_elements;
    if (block_index >= data.size() / layout.block_bytes)
        invalid("dense: tensor element out of range");
    range(data.data(), data.size());
    return decode(type, data.data() + block_index * layout.block_bytes, element % layout.block_elements);
}

void matmul_f32(const QMatrix& matrix, std::span<const float> input,
                int columns, std::span<float> output) {
    const QMatrix weights = matrix;
    const auto shape = validate(weights, input, columns, output);
    // Preflight every output, including late nonfinite weights and arithmetic
    // overflow. Recompute instead of allocating output-sized scratch memory.
    for (std::size_t column = 0; column < shape.columns; ++column)
        for (std::size_t row = 0; row < shape.rows; ++row)
            projection<true>(weights.type, weights.weights.data() + row * shape.row_bytes,
                             input.data() + column * shape.width, shape);
    for (std::size_t column = 0; column < shape.columns; ++column)
        for (std::size_t row = 0; row < shape.rows; ++row)
            output[column * shape.rows + row] =
                projection<false>(weights.type, weights.weights.data() + row * shape.row_bytes,
                                  input.data() + column * shape.width, shape);
}

} // namespace qwen
