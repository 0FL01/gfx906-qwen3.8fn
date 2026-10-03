#include "quant.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define QWEN_AVX2_TARGET __attribute__((target("avx2,fma,f16c")))
#define QWEN_X86_INTRINSICS 1
#else
#define QWEN_X86_INTRINSICS 0
#endif

namespace qwen {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(message);
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        invalid("quant: size overflow");
    return a * b;
}

std::uint16_t load_half(const void* address) {
    std::uint16_t bits;
    std::memcpy(&bits, address, sizeof(bits));
    return bits;
}

std::size_t block_bytes(TensorType type) {
    switch (type) {
    case TensorType::Q4_0: return sizeof(Q4_0);
    case TensorType::Q4_1: return sizeof(Q4_1);
    default: invalid("quant: expected Q4_0 or Q4_1 weights");
    }
}

int integer_dot_scalar(const std::uint8_t* codes, const Q8_1& activation) {
    int sum = 0;
    for (std::size_t i = 0; i < 16; ++i) {
        sum += int(codes[i] & 15) * int(activation.qs[i]);
        sum += int(codes[i] >> 4) * int(activation.qs[i + 16]);
    }
    return sum;
}

#if QWEN_X86_INTRINSICS
QWEN_AVX2_TARGET int integer_dot_avx2(const std::uint8_t* codes, const Q8_1& activation) {
    const auto packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
    const auto mask = _mm_set1_epi8(15);
    const auto low = _mm_and_si128(packed, mask);
    const auto high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    const auto nibbles = _mm256_set_m128i(high, low);
    const auto q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation.qs));
    // maddubs is unsigned nibble * signed int8, including -128. Its pair sum
    // cannot saturate: |a*x+b*y| <= 2*15*128 = 3840 < 32767. This is W4A8,
    // not a valid unsaturated scheme for arbitrary unsigned W8 values.
    const auto pairs = _mm256_maddubs_epi16(nibbles, q8);
    const auto quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    auto sum = _mm_add_epi32(_mm256_castsi256_si128(quads), _mm256_extracti128_si256(quads, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}
#endif

template<bool Avx2> int integer_dot(const std::uint8_t* codes, const Q8_1& activation) {
#if QWEN_X86_INTRINSICS
    if constexpr (Avx2) return integer_dot_avx2(codes, activation);
#endif
    return integer_dot_scalar(codes, activation);
}

// ADAPT arithmetic contract (no donor source copied): mx revision
// dcd685463d597d31f5ca759d32c94592a2740fa4, ggml-cuda/vecdotq.cuh,
// vec_dot_q4_0_q8_1_impl / vec_dot_q4_1_q8_1_impl. These are full 32-value
// block results, so the partial-thread correction factors collapse to 8 and 1.
template<TensorType Type>
float scaled_dot(int sum, float d4, float m4, const Q8_1& activation) {
    const float d8 = half_to_float(activation.d);
    const float s8 = half_to_float(activation.s);
    if constexpr (Type == TensorType::Q4_0) {
        return d4 * (float(sum) * d8 - 8.0f * s8);
    } else {
        // mx FAST_FP16_AVAILABLE multiplies the two half2 lanes in FP16
        // BEFORE conversion to FP32. A float product of two finite halfs is
        // exact; the explicit RNE half conversion supplies that rounding.
        const float d4d8 = half_to_float(float_to_half(d4 * d8));
        const float m4s8 = half_to_float(float_to_half(m4 * s8));
        return float(sum) * d4d8 + m4s8;
    }
}

template<bool Avx2, TensorType Type>
float dot_unchecked(const void* block, const Q8_1& activation) {
    const auto* bytes = static_cast<const std::uint8_t*>(block);
    constexpr std::size_t offset = Type == TensorType::Q4_0 ? 2 : 4;
    const float d4 = half_to_float(load_half(bytes));
    float m4 = 0.0f;
    if constexpr (Type == TensorType::Q4_1) m4 = half_to_float(load_half(bytes + 2));
    return scaled_dot<Type>(integer_dot<Avx2>(bytes + offset, activation), d4, m4, activation);
}

struct MatrixShape {
    std::size_t blocks, rows, columns, block_size;
};

MatrixShape validate(QMatrix weights, std::span<const Q8_1> input,
                     int columns, std::span<float> output) {
    const auto bytes = block_bytes(weights.type);
    if (weights.input <= 0 || weights.output <= 0 || columns <= 0 || weights.input % 32 != 0)
        invalid("quant: dimensions must be positive, input must be divisible by 32");
    const auto blocks = static_cast<std::size_t>(weights.input / 32);
    const auto rows = static_cast<std::size_t>(weights.output);
    const auto cols = static_cast<std::size_t>(columns);
    const auto weight_size = multiply(multiply(blocks, rows), bytes);
    const auto input_size = multiply(blocks, cols);
    const auto output_size = multiply(rows, cols);
    // Also validate the byte products, not merely the element counts. Pointer
    // distances must be representable even on a 32-bit scalar build.
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (weight_size > limit || multiply(input_size, sizeof(Q8_1)) > limit ||
        multiply(output_size, sizeof(float)) > limit)
        invalid("quant: size overflow");
    if (weights.weights.size() != weight_size || input.size() != input_size ||
        output.size() != output_size)
        invalid("quant: matrix span size mismatch");
    return {blocks, rows, cols, bytes};
}

template<bool Avx2, TensorType Type>
void matmul(QMatrix weights, std::span<const Q8_1> input,
            std::span<float> output, MatrixShape shape) {
    constexpr std::size_t code_offset = Type == TensorType::Q4_0 ? 2 : 4;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(weights.weights.data());
    // A bounded three-column tile reuses the row's weights/scales for the
    // decode/verify shapes, with no allocation, pool, or hidden threading.
    for (std::size_t column = 0; column < shape.columns; column += 3) {
        const auto count = std::min(std::size_t{3}, shape.columns - column);
        for (std::size_t row = 0; row < shape.rows; ++row) {
            std::array<float, 3> sums{};
            for (std::size_t k = 0; k < shape.blocks; ++k) {
                const auto* block = bytes + (row * shape.blocks + k) * shape.block_size;
                const float d4 = half_to_float(load_half(block));
                float m4 = 0.0f;
                if constexpr (Type == TensorType::Q4_1) m4 = half_to_float(load_half(block + 2));
                for (std::size_t c = 0; c < count; ++c) {
                    const auto& activation = input[(column + c) * shape.blocks + k];
                    sums[c] += scaled_dot<Type>(integer_dot<Avx2>(block + code_offset, activation),
                                                d4, m4, activation);
                }
            }
            for (std::size_t c = 0; c < count; ++c)
                output[(column + c) * shape.rows + row] = sums[c];
        }
    }
}

#if QWEN_X86_INTRINSICS
// Entire loop has the checked ISA: the integer dot can inline without a
// per-block target-function call/vzeroupper, and F16C replaces portable half
// bit manipulation. Scalar oracle remains untouched. Build uses
// -ffp-contract=off; the Clang scope also guards against accidental fusion.
template<TensorType Type>
QWEN_AVX2_TARGET void matmul_fast(QMatrix weights, std::span<const Q8_1> input,
                                std::span<float> output, MatrixShape shape) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    constexpr std::size_t offset = Type == TensorType::Q4_0 ? 2 : 4;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(weights.weights.data());
    for(std::size_t column=0;column<shape.columns;column+=3) {
        const auto count=std::min(std::size_t{3},shape.columns-column);
        for(std::size_t row=0;row<shape.rows;++row) {
            std::array<float,3> sums{};
            for(std::size_t k=0;k<shape.blocks;++k) {
                const auto* block=bytes+(row*shape.blocks+k)*shape.block_size;
                const float d4=_cvtsh_ss(load_half(block));
                float m4=0;
                if constexpr(Type==TensorType::Q4_1) m4=_cvtsh_ss(load_half(block+2));
                for(std::size_t c=0;c<count;++c) {
                    const auto &x=input[(column+c)*shape.blocks+k];
                    const int dot=integer_dot_avx2(block+offset,x);
                    const float dx=_cvtsh_ss(x.d),sx=_cvtsh_ss(x.s);
                    if constexpr(Type==TensorType::Q4_0) sums[c]+=d4*(float(dot)*dx-8.0f*sx);
                    else {
                        const float dd=_cvtsh_ss(_cvtss_sh(d4*dx,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
                        const float ms=_cvtsh_ss(_cvtss_sh(m4*sx,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
                        sums[c]+=float(dot)*dd+ms;
                    }
                }
            }
            for(std::size_t c=0;c<count;++c) output[(column+c)*shape.rows+row]=sums[c];
        }
    }
}
#endif

} // namespace

float half_to_float(std::uint16_t bits) {
    const auto sign = std::uint32_t(bits & 0x8000U) << 16;
    const auto exponent = (bits >> 10) & 31U;
    auto fraction = std::uint32_t(bits & 1023U);
    std::uint32_t result;
    if (exponent == 0) {
        if (fraction == 0) return std::bit_cast<float>(sign);
        std::uint32_t e = 113;
        while ((fraction & 1024U) == 0) {
            fraction <<= 1;
            --e;
        }
        result = sign | (e << 23) | ((fraction & 1023U) << 13);
    } else if (exponent == 31) {
        // Preserve sign/payload and quiet signaling NaNs, as F16C does.
        result = sign | 0x7f800000U | (fraction << 13);
        if (fraction != 0) result |= 0x00400000U;
    } else {
        result = sign | ((exponent + 112U) << 23) | (fraction << 13);
    }
    return std::bit_cast<float>(result);
}

std::uint16_t float_to_half(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = (bits >> 16) & 0x8000U;
    const auto exponent = (bits >> 23) & 255U;
    const auto fraction = bits & 0x007fffffU;
    std::uint32_t result;
    if (exponent == 255) {
        result = 0x7c00U;
        if (fraction != 0) result |= (fraction >> 13) | 0x0200U;
    } else if (exponent >= 143) {
        result = 0x7c00U;
    } else if (exponent >= 113) {
        const auto rounded = (fraction + 0x0fffU + ((fraction >> 13) & 1U)) >> 13;
        result = ((exponent - 112U) << 10) + rounded;
    } else if (exponent >= 102) {
        const auto significand = fraction | 0x00800000U;
        const auto shift = 126U - exponent; // 14..24; includes the 2^-25 zero tie.
        const auto truncated = significand >> shift;
        const auto remainder = significand & ((1U << shift) - 1U);
        const auto midpoint = 1U << (shift - 1U);
        result = truncated + (remainder > midpoint ||
                              (remainder == midpoint && (truncated & 1U) != 0));
    } else {
        result = 0;
    }
    return static_cast<std::uint16_t>(sign | result);
}

bool cpu_has_avx2() {
#if QWEN_X86_INTRINSICS
    static const bool available = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
               __builtin_cpu_supports("f16c");
    }();
    return available;
#else
    return false;
#endif
}

void quantize_q8(std::span<const float> input, std::span<Q8_1> output) {
    if (input.size() % 32 != 0 || output.size() != input.size() / 32)
        invalid("quant: Q8 input/output span size mismatch");
    // ADAPT mx quantize.cu::quantize_q8_1 at the pinned revision above: the
    // original FP32 scale selects codes; s stores the RAW float input sum.
    for (std::size_t block = 0; block < output.size(); ++block) {
        std::array<float, 32> sums;
        float amax = 0.0f;
        for (std::size_t i = 0; i < 32; ++i) {
            const float value = input[block * 32 + i];
            if (!std::isfinite(value)) invalid("quant: nonfinite activation");
            sums[i] = value;
            amax = std::max(amax, std::fabs(value));
        }
        Q8_1 result{};
        if (amax == 0.0f) {
            output[block] = result; // Exact positive zero ABI, even for -0 inputs.
            continue;
        }
        const float d = amax / 127.0f;
        if (d == 0.0f) invalid("quant: Q8 original FP32 scale underflows to zero for nonzero block");
        result.d = float_to_half(d);
        if ((result.d & 0x7c00U) == 0x7c00U) invalid("quant: Q8 scale overflows half");
        // mx ggml/src/ggml-cuda/quantize.cu:89..101 at dcd685463d597d31f5ca759d32c94592a2740fa4
        // stores half(d,sum) independently of codes selected with FP32 d.
        // A stored zero d is valid even with nonzero codes and a nonzero raw sum.
        // In-place paired updates are exactly the ascending XOR butterfly;
        // both lanes read the preceding stage, matching the GPU DPP order.
        for (std::size_t mask = 1; mask < 32; mask <<= 1) {
            for (std::size_t lane = 0; lane < 32; ++lane) {
                if ((lane & mask) == 0) {
                    const float sum = sums[lane] + sums[lane ^ mask];
                    sums[lane] = sum;
                    sums[lane ^ mask] = sum;
                }
            }
        }
        result.s = float_to_half(sums[0]);
        if (!std::isfinite(sums[0]) || (result.s & 0x7c00U) == 0x7c00U)
            invalid("quant: Q8 raw sum overflows half");
        for (std::size_t i = 0; i < 32; ++i) {
            const float code = std::roundf(input[block * 32 + i] / d);
            // FP32 subnormal d can make the ratio exceed the canonical range.
            // Check before conversion: an out-of-range float-to-int cast is UB.
            if (!std::isfinite(code) || code < -127.0f || code > 127.0f)
                invalid("quant: Q8 rounded code outside finite [-127,127] range");
            result.qs[i] = static_cast<std::int8_t>(code);
        }
        output[block] = result;
    }
}

float dot_q4_scalar(const void* block, TensorType type, const Q8_1& activation) {
    block_bytes(type);
    if (block == nullptr) invalid("quant: null Q4 block");
    if (type == TensorType::Q4_0) return dot_unchecked<false, TensorType::Q4_0>(block, activation);
    return dot_unchecked<false, TensorType::Q4_1>(block, activation);
}

float dot_q4_avx2(const void* block, TensorType type, const Q8_1& activation) {
    if (!cpu_has_avx2()) return dot_q4_scalar(block, type, activation);
    block_bytes(type);
    if (block == nullptr) invalid("quant: null Q4 block");
    if (type == TensorType::Q4_0) return dot_unchecked<true, TensorType::Q4_0>(block, activation);
    return dot_unchecked<true, TensorType::Q4_1>(block, activation);
}

void matmul_scalar(QMatrix weights, std::span<const Q8_1> input,
                   int columns, std::span<float> output) {
    const auto shape = validate(weights, input, columns, output);
    if (weights.type == TensorType::Q4_0)
        matmul<false, TensorType::Q4_0>(weights, input, output, shape);
    else matmul<false, TensorType::Q4_1>(weights, input, output, shape);
}

void matmul_avx2(QMatrix weights, std::span<const Q8_1> input,
                 int columns, std::span<float> output) {
    if (!cpu_has_avx2()) return matmul_scalar(weights, input, columns, output);
    const auto shape = validate(weights, input, columns, output);
#if QWEN_X86_INTRINSICS
    if (weights.type == TensorType::Q4_0)
        matmul_fast<TensorType::Q4_0>(weights, input, output, shape);
    else matmul_fast<TensorType::Q4_1>(weights, input, output, shape);
#else
    if (weights.type == TensorType::Q4_0)
        matmul<false, TensorType::Q4_0>(weights, input, output, shape);
    else matmul<false, TensorType::Q4_1>(weights, input, output, shape);
#endif
}

void matmul_avx2_baseline(QMatrix weights,std::span<const Q8_1> input,
                          int columns,std::span<float> output) {
    if(!cpu_has_avx2()) return matmul_scalar(weights,input,columns,output);
    const auto shape=validate(weights,input,columns,output);
    if(weights.type==TensorType::Q4_0) matmul<true,TensorType::Q4_0>(weights,input,output,shape);
    else matmul<true,TensorType::Q4_1>(weights,input,output,shape);
}

} // namespace qwen
