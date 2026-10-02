#include "linear_reference.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// WRITE OURS: original scalar byte decoder, no GPU/donor implementation copied.
// Canonical GGML layouts and common-Q8 arithmetic follow mx dcd685463d597d31f5ca
// (ggml-quants.c / vecdotq.cuh); Q4 delegates to the qualified dot_q4_scalar.
// Q6's logical element index is decoded independently of packed SDOT4 words.
namespace qwen {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
// The existing Q4 scalar reads native half words; canonical GGUF is LE.
static_assert(std::endian::native == std::endian::little);

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(message);
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        invalid("linear reference: size overflow");
    return a * b;
}

TypeLayout layout_for(TensorType type) {
    switch (type) {
    case TensorType::Q4_0: return {32, 18};
    case TensorType::Q4_1: return {32, 20};
    case TensorType::Q5_0: return {32, 22};
    case TensorType::Q8_0: return {32, 34};
    case TensorType::Q6_K: return {256, 210};
    default: invalid("linear reference: unsupported weight type");
    }
}

unsigned byte(const std::byte* p) {
    return std::to_integer<unsigned>(*p);
}

std::uint16_t u16(const std::byte* p) {
    return static_cast<std::uint16_t>(byte(p) | (byte(p + 1) << 8));
}

int signed_byte(const std::byte* p) {
    const int value = static_cast<int>(byte(p));
    return value < 128 ? value : value - 256;
}

struct Range { std::uintptr_t begin, end; };

Range range(const void* p, std::size_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(p);
    if (p == nullptr || bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        invalid("linear reference: invalid address range");
    return {begin, begin + bytes};
}

bool overlaps(Range a, Range b) {
    return a.begin < b.end && b.begin < a.end;
}

template<bool Check> float finite(float value) {
    if constexpr (Check)
        if (!std::isfinite(value)) invalid("linear reference: nonfinite arithmetic");
    return value;
}

struct Shape {
    std::size_t blocks, q8_blocks, rows, columns, row_bytes;
    TypeLayout layout;
};

Shape validate(const QMatrix& matrix, std::span<const Q8_1> input,
               int columns, std::span<float> output) {
    const auto layout = layout_for(matrix.type);
    if (matrix.input <= 0 || matrix.output <= 0 || columns < 1 || columns > 3 ||
        static_cast<std::size_t>(matrix.input) % layout.block_elements != 0)
        invalid("linear reference: positive block-divisible dimensions and columns 1..3 required");
    const auto width = static_cast<std::size_t>(matrix.input);
    const auto rows = static_cast<std::size_t>(matrix.output);
    const auto cols = static_cast<std::size_t>(columns);
    const auto blocks = width / layout.block_elements, q8_blocks = width / 32;
    const auto row_bytes = multiply(blocks, layout.block_bytes);
    const auto weight_bytes = multiply(row_bytes, rows);
    const auto input_count = multiply(q8_blocks, cols), output_count = multiply(rows, cols);
    const auto input_bytes = multiply(input_count, sizeof(Q8_1));
    const auto output_bytes = multiply(output_count, sizeof(float));
    if (matrix.weights.size() != weight_bytes || input.size() != input_count ||
        output.size() != output_count)
        invalid("linear reference: matrix span size mismatch");
    const auto weights_range = range(matrix.weights.data(), weight_bytes);
    const auto input_range = range(input.data(), input_bytes);
    const auto output_range = range(output.data(), output_bytes);
    if (overlaps(output_range, weights_range) || overlaps(output_range, input_range))
        invalid("linear reference: output overlaps input or weights");
    // Validate s even for types that do not use the raw sum in their arithmetic.
    for (const auto& x : input) {
        finite<true>(half_to_float(x.d));
        finite<true>(half_to_float(x.s));
    }
    for (std::size_t b = 0; b < multiply(blocks, rows); ++b) {
        const auto* block = matrix.weights.data() + b * layout.block_bytes;
        finite<true>(half_to_float(u16(block + (matrix.type == TensorType::Q6_K ? 208 : 0))));
        if (matrix.type == TensorType::Q4_1) finite<true>(half_to_float(u16(block + 2)));
    }
    return {blocks, q8_blocks, rows, cols, row_bytes, layout};
}

// Decode a logical Q6 element j, not an SDOT4 word/byte lane. Low planes hold
// quarters 0/1 and 2/3 of each 128-element half; high planes hold two bits per quarter.
int q6_code(const std::byte* block, std::size_t j) {
    const auto half = j / 128, quarter = (j % 128) / 32, lane = j % 32;
    const auto low_at = half * 64 + (quarter % 2) * 32 + lane;
    const auto low = (byte(block + low_at) >> (4 * (quarter / 2))) & 15;
    const auto high = (byte(block + 128 + half * 32 + lane) >> (2 * quarter)) & 3;
    return static_cast<int>(low + 16 * high) - 32;
}

template<bool Check>
float dot32(TensorType type, const std::byte* block, const Q8_1& x) {
    if (type == TensorType::Q4_0 || type == TensorType::Q4_1)
        return finite<Check>(dot_q4_scalar(block, type, x));
    int dot = 0;
    for (std::size_t j = 0; j < 32; ++j) {
        int code;
        if (type == TensorType::Q8_0) code = signed_byte(block + 2 + j);
        else {
            const auto low = (byte(block + 6 + j % 16) >> (4 * (j / 16))) & 15;
            const auto high = (byte(block + 2 + j / 8) >> (j % 8)) & 1;
            code = static_cast<int>(low + 16 * high); // unsigned Q5: 0..31
        }
        dot += code * int(x.qs[j]); // int32 full-block dot, supports -128
    }
    const float dw = half_to_float(u16(block)), dx = half_to_float(x.d);
    if (type == TensorType::Q8_0)
        return finite<Check>(finite<Check>(dw * dx) * float(dot)); // no half product
    const float product = finite<Check>(float(dot) * dx);
    const float correction = finite<Check>(16.0f * half_to_float(x.s));
    return finite<Check>(dw * finite<Check>(product - correction));
}

template<bool Check>
float projection(TensorType type, const std::byte* row, const Q8_1* input, Shape shape) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float acc = 0.0f;
    for (std::size_t k = 0; k < shape.blocks; ++k) {
        const auto* block = row + k * shape.layout.block_bytes;
        if (type != TensorType::Q6_K) {
            acc = finite<Check>(acc + dot32<Check>(type, block, input[k]));
            continue;
        }
        const float d = half_to_float(u16(block + 208));
        for (std::size_t iqs = 0; iqs < 32; ++iqs) {
            float pair = 0.0f;
            for (std::size_t i = 0; i < 2; ++i) {
                const auto j = 128 * (iqs / 16) + 32 * ((iqs % 16) / 8) +
                               4 * (iqs % 8) + 64 * i;
                const auto& x = input[k * 8 + j / 32];
                int dot = 0;
                for (std::size_t t = 0; t < 4; ++t)
                    dot += q6_code(block, j + t) * int(x.qs[(j + t) % 32]);
                // |dot*subscale| <= 4*32*128*128 = 2^21; integer
                // multiplication precedes the exact int->float conversion.
                const int scaled_dot = dot * signed_byte(block + 192 + j / 16);
                const float term = finite<Check>(half_to_float(x.d) * float(scaled_dot));
                pair = finite<Check>(pair + term); // i=0, then i=1
            }
            acc = finite<Check>(acc + finite<Check>(d * pair));
        }
    }
    return acc;
}

} // namespace

void matmul_q8_reference(const QMatrix& matrix, std::span<const Q8_1> input,
                         int columns, std::span<float> output) {
    const QMatrix weights = matrix;
    const auto shape = validate(weights, input, columns, output);
    // Numeric failures (notably finite Q4_1 operands overflowing half products)
    // are atomic too. Recompute rather than allocate output-sized scratch.
    for (std::size_t c = 0; c < shape.columns; ++c)
        for (std::size_t r = 0; r < shape.rows; ++r)
            projection<true>(weights.type, weights.weights.data() + r * shape.row_bytes,
                             input.data() + c * shape.q8_blocks, shape);
    for (std::size_t c = 0; c < shape.columns; ++c)
        for (std::size_t r = 0; r < shape.rows; ++r)
            output[c * shape.rows + r] =
                projection<false>(weights.type, weights.weights.data() + r * shape.row_bytes,
                                  input.data() + c * shape.q8_blocks, shape);
}

} // namespace qwen
