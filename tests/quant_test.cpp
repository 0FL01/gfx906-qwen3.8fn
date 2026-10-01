#include "quant.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define QWEN_TEST_F16C 1
#else
#define QWEN_TEST_F16C 0
#endif

namespace {

using qwen::Q4_0;
using qwen::Q4_1;
using qwen::Q8_1;
using qwen::TensorType;
std::size_t checks = 0;
std::size_t rejections = 0;
std::size_t finite_roundtrips = 0;
std::size_t matrix_cases = 0;
float max_matrix_parity_error = 0.0f;

void check(bool condition, const char* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

void near(float actual, float expected, const char* description) {
    check(std::isfinite(actual) && std::isfinite(expected) &&
          std::fabs(actual - expected) <= 1.0e-5f + 1.0e-6f * std::fabs(expected), description);
}

template<class Function> void rejected(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        ++rejections;
        check(true, "rejected invalid arguments");
        return;
    }
    throw std::runtime_error("invalid arguments unexpectedly accepted");
}

// Independent binary16 oracle: decode using the mathematical power-of-two
// definition, and encode by nearest-neighbor search over the finite lattice.
// It deliberately shares no production conversion bit-manipulation algorithm.
float reference_float(std::uint16_t h) {
    const int exponent = (h / 1024) % 32;
    const int fraction = h % 1024;
    float value;
    if (exponent == 31) {
        value = fraction == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else if (exponent == 0) {
        value = std::ldexp(float(fraction), -24);
    } else {
        value = std::ldexp(float(1024 + fraction), exponent - 25);
    }
    return (h & 0x8000U) != 0 ? -value : value;
}

std::uint16_t reference_half(float value) {
    const auto sign = std::uint16_t(std::signbit(value) ? 0x8000U : 0U);
    const double magnitude = std::fabs(double(value));
    if (std::isnan(value)) return std::uint16_t(sign | 0x7e00U);
    if (magnitude >= 65520.0) return std::uint16_t(sign | 0x7c00U);
    static const auto values = [] {
        std::array<float, 0x7c00> result{};
        for (std::size_t h = 0; h < result.size(); ++h)
            result[h] = reference_float(static_cast<std::uint16_t>(h));
        return result;
    }();
    const auto upper = std::lower_bound(values.begin(), values.end(), magnitude);
    if (upper == values.begin()) return sign;
    if (upper == values.end()) return std::uint16_t(sign | 0x7bffU);
    const auto hi = static_cast<std::size_t>(upper - values.begin());
    const double below = magnitude - double(values[hi - 1]);
    const double above = double(values[hi]) - magnitude;
    const auto index = below < above || (below == above && (hi % 2) != 0) ? hi - 1 : hi;
    return std::uint16_t(sign | index);
}

#if QWEN_TEST_F16C
__attribute__((target("avx2,fma,f16c")))
std::uint16_t hardware_half(float value) {
    return static_cast<std::uint16_t>(_cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

__attribute__((target("avx2,fma,f16c")))
float hardware_float(std::uint16_t h) {
    return _cvtsh_ss(h);
}
#endif

std::uint32_t random_bits() {
    static std::uint32_t state = 0x7058f931U;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void half_tests() {
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const auto h = static_cast<std::uint16_t>(bits);
        const float actual = qwen::half_to_float(h);
        const float expected = reference_float(h);
        if ((h & 0x7c00U) != 0x7c00U) {
            check(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(expected),
                  "half decode differs from mathematical definition");
            check(qwen::float_to_half(actual) == h, "finite half roundtrip failed");
            ++finite_roundtrips;
        } else if ((h & 1023U) == 0) {
            check(actual == expected && std::signbit(actual) == std::signbit(expected), "half infinity");
            check(qwen::float_to_half(actual) == h, "infinity roundtrip");
        } else {
            check(std::isnan(actual) && std::signbit(actual) == std::signbit(expected), "half NaN sign");
            check(qwen::float_to_half(actual) == std::uint16_t(h | 0x0200U), "NaN payload/quieting");
        }
#if QWEN_TEST_F16C
        if (qwen::cpu_has_avx2())
            check(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(hardware_float(h)),
                  "software half decode differs from F16C");
#endif
    }
    // Every adjacent finite positive half pair, including subnormal/normal
    // boundaries. Their midpoint is exactly representable as float.
    for (unsigned h = 0; h < 0x7bffU; ++h) {
        const float middle = (reference_float(static_cast<std::uint16_t>(h)) +
                              reference_float(static_cast<std::uint16_t>(h + 1))) * 0.5f;
        const auto even = static_cast<std::uint16_t>(h + (h & 1U));
        for (const bool negative : {false, true}) {
            const auto sign = static_cast<std::uint16_t>(negative ? 0x8000U : 0U);
            const float value = negative ? -middle : middle;
            check(qwen::float_to_half(value) == std::uint16_t(sign | even), "half midpoint not RNE");
            const float lower = std::nextafter(middle, 0.0f);
            const float upper = std::nextafter(middle, std::numeric_limits<float>::infinity());
            check(qwen::float_to_half(negative ? -lower : lower) == std::uint16_t(sign | h),
                  "half below midpoint");
            check(qwen::float_to_half(negative ? -upper : upper) == std::uint16_t(sign | (h + 1)),
                  "half above midpoint");
        }
    }
    for (const float value : {65504.0f, 65519.0f, 65520.0f, 65536.0f,
                              -65519.0f, -65520.0f, -0.0f,
                              std::numeric_limits<float>::denorm_min(),
                              -std::numeric_limits<float>::denorm_min(),
                              std::numeric_limits<float>::min(),
                              std::numeric_limits<float>::max()})
        check(qwen::float_to_half(value) == reference_half(value), "half corner conversion");
    for (std::size_t i = 0; i < 100000; ++i) {
        const auto bits = random_bits();
        const float value = std::bit_cast<float>(bits);
        const auto actual = qwen::float_to_half(value);
        if (std::isnan(value)) {
            const auto payload = ((bits & 0x007fffffU) >> 13) | 0x0200U;
            check(actual == std::uint16_t(((bits >> 16) & 0x8000U) | 0x7c00U | payload),
                  "float NaN payload/quieting");
        } else {
            check(actual == reference_half(value), "random float half RNE conversion");
        }
#if QWEN_TEST_F16C
        if (qwen::cpu_has_avx2())
            check(actual == hardware_half(value), "software half encode differs from F16C");
#endif
    }
}

Q8_1 reference_quantize(const std::array<float, 32>& values) {
    Q8_1 result{};
    float amax = 0.0f;
    for (const float value : values) amax = std::max(amax, std::fabs(value));
    if (amax == 0) return result;
    const float scale = amax / 127.0f;
    result.d = reference_half(scale);
    auto previous = values;
    // Literal simultaneous lane-XOR stages, independent of the production
    // paired-update reduction. Do not replace this with a double/sequential sum.
    for (const std::size_t mask : {1, 2, 4, 8, 16}) {
        auto next = previous;
        for (std::size_t lane = 0; lane < 32; ++lane)
            next[lane] = previous[lane] + previous[lane ^ mask];
        previous = next;
    }
    result.s = reference_half(previous[0]);
    for (std::size_t i = 0; i < 32; ++i) {
        const float ratio = values[i] / scale;
        // Promote after the specified FP32 division so adding 0.5 cannot
        // round a value immediately below a tie onto that tie.
        const double rounded = ratio < 0 ? std::ceil(double(ratio) - 0.5)
                                         : std::floor(double(ratio) + 0.5);
        result.qs[i] = static_cast<std::int8_t>(rounded);
    }
    return result;
}

Q8_1 quantized(const std::array<float, 32>& values) {
    Q8_1 output{};
    qwen::quantize_q8(values, std::span(&output, 1));
    const auto expected = reference_quantize(values);
    check(std::memcmp(&output, &expected, sizeof(output)) == 0, "quantized activation ABI mismatch");
    return output;
}

void quantize_tests() {
    std::array<float, 32> values{};
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = i % 2 == 0 ? -0.0f : 0.0f;
    auto result = quantized(values);
    const Q8_1 zero{};
    check(std::memcmp(&result, &zero, sizeof(zero)) == 0, "zero input must produce exact zero block");

    values.fill(0.0f);
    values[0] = 127.0f;
    const std::array<float, 8> ties{0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 126.5f, -126.5f};
    std::copy(ties.begin(), ties.end(), values.begin() + 1);
    result = quantized(values);
    const std::array<int, 8> codes{1, -1, 2, -2, 3, -3, 127, -127};
    for (std::size_t i = 0; i < codes.size(); ++i)
        check(result.qs[i + 1] == codes[i], "Q8 ties must round away from zero");
    values[1] = std::nextafter(0.5f, 0.0f);
    values[2] = -values[1];
    values[3] = std::nextafter(0.5f, 1.0f);
    values[4] = -values[3];
    result = quantized(values);
    check(result.qs[1] == 0 && result.qs[2] == 0 && result.qs[3] == 1 && result.qs[4] == -1,
          "Q8 values on either side of ties");

    values.fill(0.4f);
    values[0] = 127.0f;
    result = quantized(values);
    int integer_sum = 0;
    for (const auto code : result.qs) integer_sum += code;
    check(result.s == reference_half(139.4f), "Q8 must store half raw float input sum");
    check(result.s != reference_half(reference_float(result.d) * float(integer_sum)),
          "raw input sum incorrectly replaced by quantized reconstruction sum");

    values.fill(0.0f);
    values[0] = 1.0f;
    const float original_scale = 1.0f / 127.0f;
    values[1] = std::nextafter(original_scale * 0.5f, 0.0f);
    result = quantized(values);
    check(result.qs[1] == 0, "Q8 codes must use original FP32 scale");
    check(std::roundf(values[1] / reference_float(result.d)) == 1.0f,
          "original versus stored scale fixture must distinguish rounding");

    values.fill(0.0f);
    values[0] = 1048576.0f;
    values[1] = 0.03125f;
    values[2] = -1048576.0f;
    values[3] = 0.03125f;
    result = quantized(values);
    float sequential = 0.0f;
    for (const float value : values) sequential += value;
    check(result.s == 0 && reference_half(sequential) != 0,
          "ascending XOR sum must differ from sequential sum for cancellation fixture");

    values.fill(0.0f);
    values[0] = 127.0f * std::nextafter(std::ldexp(1.0f, -25), 1.0f);
    values[1] = -values[0];
    result = quantized(values);
    check(result.d == 1 && result.qs[0] == 127 && result.qs[1] == -127,
          "minimum surviving half scale must retain nonzero codes");

    for (std::size_t trial = 0; trial < 2048; ++trial) {
        for (float& value : values)
            value = float(int(random_bits() % 200001U) - 100000) / 4096.0f;
        quantized(values);
    }
    std::array<float, 96> multiple{};
    std::array<Q8_1, 3> outputs{};
    for (std::size_t block = 0; block < 3; ++block) {
        for (std::size_t i = 0; i < 32; ++i) {
            values[i] = float(int(i) - 16) * float(block + 1) / 8.0f;
            multiple[block * 32 + i] = values[i];
        }
        outputs[block] = reference_quantize(values);
    }
    std::array<Q8_1, 3> actual{};
    qwen::quantize_q8(multiple, actual);
    check(std::memcmp(actual.data(), outputs.data(), sizeof(actual)) == 0, "multi-block quantization");
    qwen::quantize_q8({}, {});
    check(true, "empty quantization");
    rejected([&] { qwen::quantize_q8(std::span(values).first(31), std::span(&result, 1)); });
    rejected([&] { qwen::quantize_q8(values, {}); });
    rejected([&] { qwen::quantize_q8(values, actual); });
    for (const float bad : {std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
        for (std::size_t lane = 0; lane < 32; ++lane) {
            values.fill(1.0f);
            values[lane] = bad;
            rejected([&] { qwen::quantize_q8(values, std::span(&result, 1)); });
        }
    }
    for (const float tiny : {std::numeric_limits<float>::denorm_min(),
                             std::numeric_limits<float>::min(), 0.000003f}) {
        values.fill(0.0f);
        values[0] = tiny;
        rejected([&] { qwen::quantize_q8(values, std::span(&result, 1)); });
    }
    for (const float huge : {std::numeric_limits<float>::max(), 10000000.0f}) {
        values.fill(0.0f);
        values[0] = huge;
        values[1] = -huge;
        rejected([&] { qwen::quantize_q8(values, std::span(&result, 1)); });
    }
    values.fill(3000.0f); // Scale is valid, but its raw sum is not finite in half.
    rejected([&] { qwen::quantize_q8(values, std::span(&result, 1)); });
}

// Wide integer reference first expands the 32 logical codes, then sums in
// int64. It includes signed -128 without int8 negation or saturating pairs.
float reference_dot(const void* address, TensorType type, const Q8_1& activation) {
    Q4_1 weights{};
    if (type == TensorType::Q4_0) {
        Q4_0 block{};
        std::memcpy(&block, address, sizeof(block));
        weights.d = block.d;
        std::copy(std::begin(block.qs), std::end(block.qs), weights.qs);
    } else {
        std::memcpy(&weights, address, sizeof(weights));
    }
    std::array<std::int64_t, 32> expanded{};
    for (std::size_t i = 0; i < 32; ++i)
        expanded[i] = i < 16 ? weights.qs[i] % 16 : weights.qs[i - 16] / 16;
    std::int64_t integer_dot = 0;
    for (std::size_t i = 0; i < 32; ++i)
        integer_dot += expanded[i] * std::int64_t(activation.qs[i]);
    const float d4 = reference_float(weights.d);
    const float d8 = reference_float(activation.d);
    const float raw_sum = reference_float(activation.s);
    if (type == TensorType::Q4_0)
        return d4 * (float(integer_dot) * d8 - 8.0f * raw_sum);
    const float d_product = reference_float(reference_half(d4 * d8));
    const float m_product = reference_float(reference_half(reference_float(weights.m) * raw_sum));
    return float(integer_dot) * d_product + m_product;
}

template<class Block> void check_dot(const Block& weights, TensorType type, const Q8_1& input) {
    const float expected = reference_dot(&weights, type, input);
    near(qwen::dot_q4_scalar(&weights, type, input), expected, "scalar dot versus wide reference");
    near(qwen::dot_q4_avx2(&weights, type, input), expected, "AVX2 dot versus wide reference");
}

void dot_tests() {
    Q4_0 zero_weights{0x3c00, {}};
    Q4_1 offset_weights{0x3c00, 0x0000, {}};
    Q8_1 input{0x3c00, 0x0000, {}};
    for (int code = 0; code < 16; ++code) {
        std::fill(std::begin(zero_weights.qs), std::end(zero_weights.qs), std::uint8_t(code * 17));
        std::fill(std::begin(offset_weights.qs), std::end(offset_weights.qs), std::uint8_t(code * 17));
        for (int value = -128; value <= 127; ++value) {
            std::fill(std::begin(input.qs), std::end(input.qs), static_cast<std::int8_t>(value));
            check_dot(zero_weights, TensorType::Q4_0, input);
            check_dot(offset_weights, TensorType::Q4_1, input);
            check(qwen::dot_q4_avx2(&zero_weights, TensorType::Q4_0, input) == float(32 * code * value),
                  "full-range maddubs must not saturate");
        }
    }
    const std::array<std::uint16_t, 8> scales{0x0000, 0x8000, 0x2800, 0xa800,
                                             0x3c01, 0xbc01, 0x0001, 0x8001};
    for (unsigned pattern = 0; pattern < 256; ++pattern) {
        for (std::size_t scale = 0; scale < scales.size(); ++scale) {
            zero_weights.d = scales[scale];
            offset_weights.d = scales[scale];
            offset_weights.m = scales[(scale + 3) % scales.size()];
            input.d = scales[(scale + 2) % scales.size()];
            input.s = scales[(scale + 4) % scales.size()];
            for (std::size_t lane = 0; lane < 16; ++lane) {
                const auto packed = static_cast<std::uint8_t>(pattern + lane * 37);
                zero_weights.qs[lane] = packed;
                offset_weights.qs[lane] = packed;
                input.qs[lane] = static_cast<std::int8_t>(int((pattern + lane * 19) % 256) - 128);
                input.qs[lane + 16] = static_cast<std::int8_t>(int((pattern + lane * 53) % 256) - 128);
            }
            check_dot(zero_weights, TensorType::Q4_0, input);
            check_dot(offset_weights, TensorType::Q4_1, input);
        }
    }
    // Independent raw-sum correction rather than a sum of activation codes.
    zero_weights.d = 0xbc00; // -1
    input.d = 0x3800; // 0.5
    input.s = 0x4200; // 3
    std::fill(std::begin(zero_weights.qs), std::end(zero_weights.qs), 0x00);
    std::fill(std::begin(input.qs), std::end(input.qs), std::int8_t{-128});
    check(qwen::dot_q4_scalar(&zero_weights, TensorType::Q4_0, input) == 24.0f,
          "Q4_0 signed scale and raw-sum offset");
    check_dot(zero_weights, TensorType::Q4_0, input);

    offset_weights.d = 0x3c01;
    offset_weights.m = 0xbc01;
    input.d = 0x3c01;
    input.s = 0x3c01;
    std::fill(std::begin(offset_weights.qs), std::end(offset_weights.qs), 0xff);
    std::fill(std::begin(input.qs), std::end(input.qs), std::int8_t{127});
    check_dot(offset_weights, TensorType::Q4_1, input);
    const float unrounded = float(32 * 15 * 127) * reference_float(offset_weights.d) * reference_float(input.d) +
                            reference_float(offset_weights.m) * reference_float(input.s);
    check(std::fabs(unrounded - reference_dot(&offset_weights, TensorType::Q4_1, input)) > 0.01f,
          "Q4_1 fixture must detect missing half-product rounding");
    offset_weights.d = 0x0400; // 2^-14 * 2^-14 rounds to half zero.
    offset_weights.m = 0x3800;
    input.d = 0x0400;
    input.s = 0x4200;
    check(qwen::dot_q4_scalar(&offset_weights, TensorType::Q4_1, input) == 1.5f,
          "Q4_1 half product underflow plus nonzero offset");
    check_dot(offset_weights, TensorType::Q4_1, input);
    // No natural struct alignment may be assumed for a tensor byte view.
    std::array<std::byte, sizeof(Q4_1) + 1> unaligned{};
    std::memcpy(unaligned.data() + 1, &offset_weights, sizeof(offset_weights));
    near(qwen::dot_q4_scalar(unaligned.data() + 1, TensorType::Q4_1, input), 1.5f, "unaligned scalar block");
    near(qwen::dot_q4_avx2(unaligned.data() + 1, TensorType::Q4_1, input), 1.5f, "unaligned AVX2 block");
    rejected([&] { qwen::dot_q4_scalar(nullptr, TensorType::Q4_0, input); });
    rejected([&] { qwen::dot_q4_avx2(nullptr, TensorType::Q4_1, input); });
    rejected([&] { qwen::dot_q4_scalar(&offset_weights, TensorType::Q8_0, input); });
    rejected([&] { qwen::dot_q4_avx2(&offset_weights, TensorType(999), input); });
}

void matrix_fixture(int width, int height, int columns, TensorType type, bool float_input) {
    const auto blocks = static_cast<std::size_t>(width / 32);
    const auto rows = static_cast<std::size_t>(height);
    const auto cols = static_cast<std::size_t>(columns);
    const auto block_size = type == TensorType::Q4_0 ? sizeof(Q4_0) : sizeof(Q4_1);
    const auto weight_size = blocks * rows * block_size;
    std::vector<std::byte> storage(weight_size + 2, std::byte{0x7d});
    const auto weights = std::span(storage).subspan(1, weight_size);
    const std::array<std::uint16_t, 8> scales{0x2800, 0xa800, 0x3401, 0xb401,
                                             0x0001, 0x8001, 0x3003, 0xb003};
    for (std::size_t i = 0; i < blocks * rows; ++i) {
        Q4_0 q0{};
        Q4_1 q1{};
        q0.d = q1.d = scales[random_bits() % scales.size()];
        q1.m = scales[random_bits() % scales.size()];
        for (std::size_t j = 0; j < 16; ++j)
            q0.qs[j] = q1.qs[j] = static_cast<std::uint8_t>(random_bits());
        if (type == TensorType::Q4_0) std::memcpy(weights.data() + i * block_size, &q0, block_size);
        else std::memcpy(weights.data() + i * block_size, &q1, block_size);
    }
    std::vector<Q8_1> input(blocks * cols);
    if (float_input) {
        std::vector<float> values(blocks * cols * 32);
        for (float& value : values) value = float(int(random_bits() % 8193U) - 4096) / 1024.0f;
        qwen::quantize_q8(values, input);
    } else {
        for (auto& activation : input) {
            activation.d = scales[random_bits() % scales.size()];
            activation.s = scales[random_bits() % scales.size()];
            for (auto& code : activation.qs)
                code = static_cast<std::int8_t>(int(random_bits() % 256U) - 128);
        }
    }
    constexpr float guard = -123456.0f;
    std::vector<float> scalar(rows * cols + 2, guard), avx2(rows * cols + 2, guard);
    const auto scalar_output = std::span(scalar).subspan(1, rows * cols);
    const auto avx2_output = std::span(avx2).subspan(1, rows * cols);
    const qwen::QMatrix matrix{type, width, height, weights};
    qwen::matmul_scalar(matrix, input, columns, scalar_output);
    qwen::matmul_avx2(matrix, input, columns, avx2_output);
    ++matrix_cases;
    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t row = 0; row < rows; ++row) {
            float expected = 0.0f;
            for (std::size_t k = 0; k < blocks; ++k)
                expected += reference_dot(weights.data() + (row * blocks + k) * block_size,
                                          type, input[c * blocks + k]);
            const auto index = c * rows + row;
            near(scalar_output[index], expected, "matrix scalar versus wide reference");
            near(avx2_output[index], expected, "matrix AVX2 versus wide reference");
            near(avx2_output[index], scalar_output[index], "matrix CPU/AVX2 parity");
            max_matrix_parity_error = std::max(max_matrix_parity_error,
                                               std::fabs(scalar_output[index] - avx2_output[index]));
        }
    }
    check(scalar.front() == guard && scalar.back() == guard && avx2.front() == guard && avx2.back() == guard,
          "matrix wrote outside output span");
    check(storage.front() == std::byte{0x7d} && storage.back() == std::byte{0x7d}, "matrix weight guards");
}

void matrix_tests() {
    // Literal one-block rows exercise column-major output ordering and raw
    // offsets without deriving the expected answers through reference_dot.
    std::array<Q8_1, 2> literal_input{};
    literal_input[0].d = literal_input[1].d = 0x3c00;
    literal_input[0].s = 0x4000; // 2
    literal_input[1].s = 0xc200; // -3
    for (std::size_t i = 0; i < 32; ++i) {
        literal_input[0].qs[i] = i < 16 ? 3 : -4;
        literal_input[1].qs[i] = -128;
    }
    for (const auto type : {TensorType::Q4_0, TensorType::Q4_1}) {
        std::array<Q4_0, 2> q0{};
        std::array<Q4_1, 2> q1{};
        for (std::size_t row = 0; row < 2; ++row) {
            q0[row].d = q1[row].d = 0x3c00;
            q1[row].m = 0x3800; // 0.5
            const std::uint8_t code = row == 0 ? 0x21 : 0x75;
            std::fill(std::begin(q0[row].qs), std::end(q0[row].qs), code);
            std::fill(std::begin(q1[row].qs), std::end(q1[row].qs), code);
        }
        const auto weights = type == TensorType::Q4_0
            ? std::span<const std::byte>(std::as_bytes(std::span(q0)))
            : std::span<const std::byte>(std::as_bytes(std::span(q1)));
        const std::array<float, 4> expected = type == TensorType::Q4_0
            ? std::array<float, 4>{-96.0f, -224.0f, -6120.0f, -24552.0f}
            : std::array<float, 4>{-79.0f, -207.0f, -6145.5f, -24577.5f};
        std::array<float, 4> scalar{}, avx2{};
        qwen::matmul_scalar({type, 32, 2, weights}, literal_input, 2, scalar);
        qwen::matmul_avx2({type, 32, 2, weights}, literal_input, 2, avx2);
        ++matrix_cases;
        check(scalar == expected && avx2 == expected, "literal matrix shape/order/offset fixture");
    }
    for (const auto type : {TensorType::Q4_0, TensorType::Q4_1}) {
        for (const int columns : {1, 2, 3}) {
            matrix_fixture(2560, 640, columns, type, false);
            matrix_fixture(640, 2560, columns, type, false);
        }
        for (const int columns : {1, 2, 3, 4, 7}) matrix_fixture(64, 5, columns, type, true);
        matrix_fixture(2560, 640, 3, type, true);
        matrix_fixture(640, 2560, 3, type, true);
    }

    using Matmul = void (*)(qwen::QMatrix, std::span<const Q8_1>, int, std::span<float>);
    std::array<std::byte, 41> bytes{};
    std::array<Q8_1, 3> input{};
    std::array<float, 3> output{};
    for (const Matmul multiply : {qwen::matmul_scalar, qwen::matmul_avx2}) {
        for (const TensorType type : {TensorType::Q4_0, TensorType::Q4_1}) {
            const auto size = type == TensorType::Q4_0 ? 36U : 40U;
            const qwen::QMatrix valid{type, 32, 2, std::span(bytes).first(size)};
            const auto valid_input = std::span(input).first(1);
            const auto valid_output = std::span(output).first(2);
            for (const int width : {0, -32, 1, 31, 33, std::numeric_limits<int>::max()}) {
                auto matrix = valid;
                matrix.input = width;
                rejected([&] { multiply(matrix, valid_input, 1, valid_output); });
            }
            for (const int height : {0, -1, std::numeric_limits<int>::max()}) {
                auto matrix = valid;
                matrix.output = height;
                rejected([&] { multiply(matrix, valid_input, 1, valid_output); });
            }
            for (const int columns : {0, -1, std::numeric_limits<int>::max()})
                rejected([&] { multiply(valid, valid_input, columns, valid_output); });
            for (const auto length : {0U, size - 1, size + 1}) {
                auto matrix = valid;
                matrix.weights = std::span(bytes).first(length);
                rejected([&] { multiply(matrix, valid_input, 1, valid_output); });
            }
            for (const auto length : {0U, 2U, 3U})
                rejected([&] { multiply(valid, std::span(input).first(length), 1, valid_output); });
            for (const auto length : {0U, 1U, 3U})
                rejected([&] { multiply(valid, valid_input, 1, std::span(output).first(length)); });
            // Huge legal, block-divisible int dimensions exercise checked size
            // handling without constructing a fictitiously huge/invalid span.
            auto huge = valid;
            huge.input = std::numeric_limits<int>::max() - 31;
            huge.output = std::numeric_limits<int>::max();
            rejected([&] { multiply(huge, {}, std::numeric_limits<int>::max(), {}); });
        }
        for (const TensorType type : {TensorType::F32, TensorType::F16, TensorType::Q5_0,
                                     TensorType::Q8_0, TensorType::Q6_K, TensorType::BF16, TensorType(999)})
            rejected([&] { multiply({type, 32, 2, bytes}, input, 1, output); });
    }
}

template<class Function> void group(std::string_view name, Function function) {
    const auto previous = checks;
    function();
    std::cout << name << ": " << checks - previous << " checks passed\n";
}

} // namespace

int main() {
    try {
        std::cout << "runtime AVX2/FMA/F16C: " << (qwen::cpu_has_avx2() ? "available" : "scalar fallback") << '\n';
        group("half", half_tests);
        group("quantize", quantize_tests);
        group("dot", dot_tests);
        group("matrix", matrix_tests);
        std::cout << "PASS: " << checks << " checks, " << rejections << " invalid cases rejected, "
                  << finite_roundtrips << " finite half roundtrips, " << matrix_cases << " matrix cases; "
                  << "max CPU/AVX2 matrix absolute error=" << max_matrix_parity_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
