#include "dense.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using qwen::TensorType;
constexpr std::array types{TensorType::F32, TensorType::F16, TensorType::BF16,
                           TensorType::Q4_0, TensorType::Q4_1, TensorType::Q5_0,
                           TensorType::Q8_0, TensorType::Q6_K};
std::size_t checks = 0, rejections = 0, matrix_cases = 0, block_cases = 0;

void check(bool condition, const char* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

void same(float actual, float expected, const char* description) {
    check(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(expected), description);
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

void put16(std::span<std::byte> data, std::size_t at, std::uint16_t bits) {
    data[at] = std::byte(bits & 255);
    data[at + 1] = std::byte(bits >> 8);
}

void put32(std::span<std::byte> data, std::size_t at, std::uint32_t bits) {
    for (std::size_t i = 0; i < 4; ++i) data[at + i] = std::byte((bits >> (8 * i)) & 255);
}

// Independent binary16 mathematical reference (no production half helper).
float half_reference(std::uint16_t bits) {
    const int exponent = (bits / 1024) % 32, fraction = bits % 1024;
    if (exponent == 31) {
        auto result = std::uint32_t(bits & 0x8000U) << 16;
        result |= 0x7f800000U | (std::uint32_t(fraction) << 13);
        if (fraction != 0) result |= 0x00400000U;
        return std::bit_cast<float>(result);
    }
    const float value = exponent == 0 ? std::ldexp(float(fraction), -24)
                                     : std::ldexp(float(1024 + fraction), exponent - 25);
    return (bits & 0x8000U) != 0 ? -value : value;
}

void ieee_readers() {
    // Odd starting address checks alignment-independent canonical LE reading.
    std::vector<std::byte> storage(1 + 65536 * 2);
    auto data = std::span(storage).subspan(1);
    for (std::size_t i = 0; i < 65536; ++i) put16(data, 2 * i, static_cast<std::uint16_t>(i));
    for (std::size_t i = 0; i < 65536; ++i) {
        same(qwen::tensor_element(TensorType::F16, data, i), half_reference(static_cast<std::uint16_t>(i)),
             "all F16 bitpatterns, including subnormals, signs, infinities and quieted NaN payloads");
        same(qwen::tensor_element(TensorType::BF16, data, i),
             std::bit_cast<float>(static_cast<std::uint32_t>(i) << 16), "all BF16 bitpatterns");
    }
    constexpr std::array<std::uint32_t, 24> bits{
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
        0x3f800001, 0x3f7fffff, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff};
    std::vector<std::byte> raw(1 + 4 * bits.size());
    auto f32 = std::span(raw).subspan(1);
    for (std::size_t i = 0; i < bits.size(); ++i) put32(f32, i * 4, bits[i]);
    for (std::size_t i = 0; i < bits.size(); ++i)
        same(qwen::tensor_element(TensorType::F32, f32, i), std::bit_cast<float>(bits[i]),
             "F32 exact LE IEEE bitpatterns");
}

struct Fixture {
    std::vector<std::byte> bytes;
    std::vector<float> dense;
};

// Construct canonical bytes from logical integer codes and scales, and produce
// a separate dense mathematical reference directly from those logical values.
// Expected values never use tensor_element or any other production decoder.
Fixture quant_block(TensorType type, std::span<const int> codes, std::uint16_t d,
                    std::uint16_t m = 0, std::span<const int> scales = {}) {
    const auto layout = qwen::type_layout(type);
    check(codes.size() == layout.block_elements, "test block size");
    Fixture result{std::vector<std::byte>(layout.block_bytes), std::vector<float>(codes.size())};
    const float scale = half_reference(d), offset = half_reference(m);
    if (type == TensorType::Q6_K) {
        check(scales.size() == 16, "test Q6_K scale count");
        put16(result.bytes, 208, d);
        for (std::size_t i = 0; i < 16; ++i) result.bytes[192 + i] = std::byte(scales[i] & 255);
        // Pack four quarters together, rather than production's single-index
        // lookup. ql low nibbles cover quarters 0/1, high nibbles 2/3.
        for (std::size_t half = 0; half < 2; ++half) {
            for (std::size_t lane = 0; lane < 32; ++lane) {
                std::array<unsigned, 4> q{};
                for (std::size_t quarter = 0; quarter < 4; ++quarter)
                    q[quarter] = static_cast<unsigned>(codes[half * 128 + quarter * 32 + lane] + 32);
                result.bytes[half * 64 + lane] = std::byte((q[0] & 15) | ((q[2] & 15) << 4));
                result.bytes[half * 64 + 32 + lane] = std::byte((q[1] & 15) | ((q[3] & 15) << 4));
                result.bytes[128 + half * 32 + lane] =
                    std::byte((q[0] / 16) | ((q[1] / 16) << 2) | ((q[2] / 16) << 4) | ((q[3] / 16) << 6));
            }
        }
        for (std::size_t i = 0; i < codes.size(); ++i)
            result.dense[i] = (scale * float(scales[i / 16])) * float(codes[i]);
        return result;
    }
    put16(result.bytes, 0, d);
    if (type == TensorType::Q4_1) put16(result.bytes, 2, m);
    if (type == TensorType::Q8_0) {
        for (std::size_t i = 0; i < 32; ++i) {
            result.bytes[2 + i] = std::byte(codes[i] & 255);
            result.dense[i] = float(codes[i]) * scale;
        }
        return result;
    }
    const std::size_t code_offset = type == TensorType::Q4_0 ? 2 : type == TensorType::Q4_1 ? 4 : 6;
    const int bias = type == TensorType::Q4_0 ? 8 : type == TensorType::Q4_1 ? 0 : 16;
    for (std::size_t i = 0; i < 32; ++i) {
        const unsigned q = static_cast<unsigned>(codes[i] + bias);
        result.bytes[code_offset + i % 16] |= std::byte((q % 16) << (i < 16 ? 0 : 4));
        if (type == TensorType::Q5_0 && q >= 16) result.bytes[2 + i / 8] |= std::byte(1U << (i % 8));
        result.dense[i] = float(codes[i]) * scale;
        if (type == TensorType::Q4_1) result.dense[i] += offset;
    }
    return result;
}

void read_block(TensorType type, const Fixture& fixture) {
    // Also tests flat indexing across the second block and unaligned storage.
    std::vector<std::byte> raw(1 + 2 * fixture.bytes.size());
    auto bytes = std::span(raw).subspan(1);
    std::copy(fixture.bytes.begin(), fixture.bytes.end(), bytes.begin());
    std::copy(fixture.bytes.begin(), fixture.bytes.end(), bytes.begin() + fixture.bytes.size());
    for (std::size_t block = 0; block < 2; ++block)
        for (std::size_t i = 0; i < fixture.dense.size(); ++i)
            same(qwen::tensor_element(type, bytes, block * fixture.dense.size() + i), fixture.dense[i],
                 "canonical quantized block exact decode");
    ++block_cases;
}

void quant_readers() {
    constexpr std::array<std::uint16_t, 7> scales{0x3c00, 0xb800, 0x3555, 0x0001, 0x8001, 0x8000, 0x7bff};
    for (const auto type : {TensorType::Q4_0, TensorType::Q4_1, TensorType::Q5_0}) {
        std::array<int, 32> codes{};
        for (std::size_t i = 0; i < 32; ++i) {
            // Q5 covers all 32 codes with unlike low/high halves. Q4 covers
            // all nibbles twice, with different ordering in the high half.
            if (type == TensorType::Q5_0) codes[i] = static_cast<int>((i * 13 + 7) % 32) - 16;
            else codes[i] = static_cast<int>((i < 16 ? i : 31 - i) % 16) -
                            (type == TensorType::Q4_0 ? 8 : 0);
        }
        for (const auto d : scales)
            read_block(type, quant_block(type, codes, d, type == TensorType::Q4_1 ? 0xbc00 : 0));
        if (type == TensorType::Q4_1) {
            for (const auto m : scales) read_block(type, quant_block(type, codes, 0xb555, m));
            read_block(type, quant_block(type, codes, 0x3c00, 0x7bff)); // FP32, not half-rounded d*q+m.
        }
    }
    // Individual Q5 bitplane bits isolate LE byte order and element 15/16/31.
    for (std::size_t bit = 0; bit < 32; ++bit) {
        Fixture fixture{std::vector<std::byte>(22), std::vector<float>(32, -16.0f)};
        put16(fixture.bytes, 0, 0x3c00);
        fixture.bytes[2 + bit / 8] = std::byte(1U << (bit % 8));
        fixture.dense[bit] = 0.0f;
        read_block(TensorType::Q5_0, fixture);
    }
    for (int first = -128; first < 128; first += 32) {
        std::array<int, 32> codes{};
        for (int i = 0; i < 32; ++i) codes[static_cast<std::size_t>(i)] = first + i;
        for (const auto d : scales) read_block(TensorType::Q8_0, quant_block(TensorType::Q8_0, codes, d));
    }

    constexpr std::array<int, 16> signed_scales{-128, 127, -1, 0, 1, -64, 63, 2,
                                                -2, 32, -32, 17, -17, 100, -100, 7};
    std::array<int, 256> codes{};
    for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<int>((i * 13 + i / 16) % 64) - 32;
    for (const auto d : scales) read_block(TensorType::Q6_K, quant_block(TensorType::Q6_K, codes, d, 0, signed_scales));
    for (int code = -32; code <= 31; ++code) {
        codes.fill(code);
        read_block(TensorType::Q6_K, quant_block(TensorType::Q6_K, codes, 0xb555, 0, signed_scales));
    }
    for (int first = -128; first < 128; first += 16) {
        std::array<int, 16> sc{};
        for (int i = 0; i < 16; ++i) sc[static_cast<std::size_t>(i)] = first + i;
        for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<int>((i * 7) % 64) - 32;
        read_block(TensorType::Q6_K, quant_block(TensorType::Q6_K, codes, 0x3555, 0, sc));
    }
    // Physical one-bit fixtures: sweep every ql/qh bit, independently deriving
    // its destination element. Both 128-element halves and every quarter.
    for (std::size_t at = 0; at < 192; ++at) {
        for (unsigned bit = 0; bit < 8; ++bit) {
            Fixture fixture{std::vector<std::byte>(210), std::vector<float>(256, -32.0f)};
            put16(fixture.bytes, 208, 0x3c00);
            std::fill(fixture.bytes.begin() + 192, fixture.bytes.begin() + 208, std::byte{1});
            fixture.bytes[at] = std::byte(1U << bit);
            std::size_t element;
            unsigned delta;
            if (at < 128) {
                element = (at / 64) * 128 + at % 64 + (bit < 4 ? 0 : 64);
                delta = 1U << (bit % 4);
            } else {
                const auto qh = at - 128;
                element = (qh / 32) * 128 + qh % 32 + (bit / 2) * 32;
                delta = 16U << (bit % 2);
            }
            fixture.dense[element] += float(delta);
            read_block(TensorType::Q6_K, fixture);
        }
    }
}

Fixture matrix_fixture(TensorType type, int width, int rows) {
    const auto layout = qwen::type_layout(type);
    const auto count = static_cast<std::size_t>(width) * static_cast<std::size_t>(rows);
    Fixture result{std::vector<std::byte>((count / layout.block_elements) * layout.block_bytes),
                   std::vector<float>(count)};
    if (layout.block_elements == 1) {
        for (std::size_t i = 0; i < count; ++i) {
            const float value = float(static_cast<int>((i * 17 + i / static_cast<std::size_t>(width)) % 61) - 30) / 64.0f;
            if (type == TensorType::F32) {
                result.dense[i] = value;
                put32(result.bytes, 4 * i, std::bit_cast<std::uint32_t>(value));
            } else if (type == TensorType::F16) {
                const auto bits = static_cast<std::uint16_t>((i % 2 ? 0x8000U : 0U) | ((i * 137) % 0x7c00U));
                result.dense[i] = half_reference(bits);
                put16(result.bytes, 2 * i, bits);
            } else {
                const auto bits = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
                result.dense[i] = std::bit_cast<float>(std::uint32_t(bits) << 16);
                put16(result.bytes, 2 * i, bits);
            }
        }
        return result;
    }
    for (std::size_t b = 0; b < count / layout.block_elements; ++b) {
        std::vector<int> codes(layout.block_elements);
        const int range = type == TensorType::Q8_0 ? 256 : type == TensorType::Q6_K ? 64 : type == TensorType::Q5_0 ? 32 : 16;
        const int bias = type == TensorType::Q4_1 ? 0 : range / 2;
        for (std::size_t i = 0; i < codes.size(); ++i)
            codes[i] = static_cast<int>((i * 13 + b * 7 + i / 16) % static_cast<std::size_t>(range)) - bias;
        std::array<int, 16> scales{};
        for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = static_cast<int>((i * 19 + b * 23) % 256) - 128;
        const auto d = static_cast<std::uint16_t>(b % 2 ? 0xb555 : 0x2801);
        const auto m = static_cast<std::uint16_t>(b % 2 ? 0x2c01 : 0xb101);
        const auto block = quant_block(type, codes, d, m, scales);
        std::copy(block.bytes.begin(), block.bytes.end(), result.bytes.begin() + b * layout.block_bytes);
        std::copy(block.dense.begin(), block.dense.end(), result.dense.begin() + b * layout.block_elements);
    }
    return result;
}

std::vector<float> dense_reference(std::span<const float> weights, std::span<const float> input,
                                   int width, int rows, int columns) {
    std::vector<float> result(static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns));
    for (int c = 0; c < columns; ++c) {
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                const float product = weights[static_cast<std::size_t>(row * width + k)] *
                                      input[static_cast<std::size_t>(c * width + k)];
                sum += product;
            }
            result[static_cast<std::size_t>(c * rows + row)] = sum;
        }
    }
    return result;
}

void matrices() {
    for (const auto type : types) {
        const auto block = static_cast<int>(qwen::type_layout(type).block_elements);
        const int width = block == 1 ? 7 : 2 * block, rows = 5;
        const auto fixture = matrix_fixture(type, width, rows);
        // Byte-misaligned complete matrix, preserving actual canonical bytes.
        std::vector<std::byte> raw(1 + fixture.bytes.size());
        std::copy(fixture.bytes.begin(), fixture.bytes.end(), raw.begin() + 1);
        const qwen::QMatrix matrix{type, width, rows, std::span(raw).subspan(1)};
        for (int columns = 1; columns <= 3; ++columns) {
            std::vector<float> input(static_cast<std::size_t>(width * columns));
            for (std::size_t i = 0; i < input.size(); ++i)
                input[i] = float(static_cast<int>((i * 29 + i / static_cast<std::size_t>(width) * 31) % 71) - 35) / 127.0f;
            const auto expected = dense_reference(fixture.dense, input, width, rows, columns);
            std::vector<float> guarded(expected.size() + 2, -1234.0f);
            auto output = std::span(guarded).subspan(1, expected.size());
            std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
            qwen::matmul_f32(matrix, input, columns, output);
            for (std::size_t i = 0; i < expected.size(); ++i) same(output[i], expected[i], "column-major FP32 dense reference parity");
            same(guarded.front(), -1234.0f, "output prefix guard");
            same(guarded.back(), -1234.0f, "output suffix guard");
            check(std::equal(fixture.bytes.begin(), fixture.bytes.end(), raw.begin() + 1), "weights remain unchanged");
            ++matrix_cases;
        }
    }
    // Ascending FP32 accumulation, not FP64 or a pairwise reduction.
    std::array<float, 3> weights{16777216.0f, 1.0f, -16777216.0f}, input{1.0f, 1.0f, 1.0f};
    std::array<float, 1> output{};
    qwen::matmul_f32({TensorType::F32, 3, 1, std::as_bytes(std::span(weights))}, input, 1, output);
    same(output[0], 0.0f, "scalar FP32 ascending sum, no double accumulation");
    // With fusion the second multiplication/addition would leave 2^-46.
    std::array<float, 2> w{-1.0f, 0x1.000002p0f}, x{0x1.000004p0f, 0x1.000002p0f};
    qwen::matmul_f32({TensorType::F32, 2, 1, std::as_bytes(std::span(w))}, x, 1, output);
    same(output[0], 0.0f, "separate FP32 multiply and add, no contraction");
    // Raw FP32 values too small for a viable Q8_1 activation scale still work.
    w = {1.0f, 0.0f};
    x = {1.0e-12f, 0.0f};
    qwen::matmul_f32({TensorType::F32, 2, 1, std::as_bytes(std::span(w))}, x, 1, output);
    same(output[0], x[0], "FP32 activation preserved without Q8 quantization");
}

template<class Function> void atomic_rejected(Function function, std::span<float> output) {
    const auto bytes = std::as_bytes(output);
    const std::vector<std::byte> before(bytes.begin(), bytes.end());
    rejected(function);
    check(std::equal(before.begin(), before.end(), bytes.begin()), "rejected call leaves all output bytes unchanged");
}

void invalid_shapes() {
    for (const auto type : types) {
        const auto block = static_cast<int>(qwen::type_layout(type).block_elements);
        const int width = block == 1 ? 3 : block;
        const auto fixture = matrix_fixture(type, width, 2);
        qwen::QMatrix matrix{type, width, 2, fixture.bytes};
        std::vector<float> input(static_cast<std::size_t>(width * 2), 0.25f), output(5, -321.0f);
        auto out = std::span(output).first(4);
        const auto fail = [&](qwen::QMatrix m, std::span<const float> x, int columns, std::span<float> y) {
            atomic_rejected([&] { qwen::matmul_f32(m, x, columns, y); }, output);
        };
        for (const int bad : {0, -1}) {
            auto m = matrix; m.input = bad; fail(m, input, 2, out);
            m = matrix; m.output = bad; fail(m, input, 2, out);
            fail(matrix, input, bad, out);
        }
        if (block != 1) {
            auto m = matrix; m.input = width - 1; fail(m, input, 2, out);
            m.input = width + 1; fail(m, input, 2, out);
        }
        auto m = matrix; m.weights = matrix.weights.first(matrix.weights.size() - 1); fail(m, input, 2, out);
        auto extra = fixture.bytes; extra.push_back(std::byte{0}); m.weights = extra; fail(m, input, 2, out);
        fail(matrix, std::span(input).first(input.size() - 1), 2, out);
        auto longer = input; longer.push_back(0.0f); fail(matrix, longer, 2, out);
        fail(matrix, input, 2, out.first(3));
        fail(matrix, input, 2, output);
        m = matrix; m.input = std::numeric_limits<int>::max(); m.output = std::numeric_limits<int>::max();
        fail(m, input, std::numeric_limits<int>::max(), out);

        rejected([&] { (void)qwen::tensor_element(type, {}, 0); });
        rejected([&] { (void)qwen::tensor_element(type, matrix.weights.first(matrix.weights.size() - 1), 0); });
        rejected([&] { (void)qwen::tensor_element(type, extra, 0); });
        rejected([&] { (void)qwen::tensor_element(type, matrix.weights, static_cast<std::size_t>(2 * width)); });
        rejected([&] { (void)qwen::tensor_element(type, matrix.weights, std::numeric_limits<std::size_t>::max()); });
    }
    std::array<float, 1> output{-321.0f}, input{1.0f};
    const auto unknown = static_cast<TensorType>(5);
    atomic_rejected([&] { qwen::matmul_f32({unknown, 1, 1, {}}, input, 1, output); }, output);
    rejected([&] { (void)qwen::tensor_element(unknown, {}, 0); });
}

void aliases() {
    std::array<float, 8> w{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<float, 16> storage{};
    std::fill(storage.begin(), storage.end(), 0.25f);
    const qwen::QMatrix matrix{TensorType::F32, 4, 2, std::as_bytes(std::span(w))};
    for (const auto offset : {std::size_t{0}, std::size_t{1}, std::size_t{7}}) {
        auto input = std::span(storage).first(8), output = std::span(storage).subspan(offset, 4);
        atomic_rejected([&] { qwen::matmul_f32(matrix, input, 2, output); }, storage);
    }
    auto output = std::span(storage).first(4), input = std::span(storage).subspan(3, 8);
    atomic_rejected([&] { qwen::matmul_f32(matrix, input, 2, output); }, storage);
    // Output touching, but not overlapping, input is valid.
    input = std::span(storage).first(8); output = std::span(storage).subspan(8, 4);
    qwen::matmul_f32(matrix, input, 2, output);
    same(output[0], 2.5f, "adjacent input/output allowed");

    for (const auto at : {std::size_t{0}, std::size_t{7}}) {
        const qwen::QMatrix m{TensorType::F32, 4, 2, std::as_bytes(std::span(storage).first(8))};
        auto out = std::span(storage).subspan(at, 2);
        atomic_rejected([&] { qwen::matmul_f32(m, std::span(w).first(4), 1, out); }, storage);
    }
    const qwen::QMatrix m{TensorType::F32, 4, 2, std::as_bytes(std::span(storage).subspan(1, 8))};
    atomic_rejected([&] { qwen::matmul_f32(m, std::span(w).first(4), 1, std::span(storage).first(2)); }, storage);
    std::array<float, 32> x{};
    const qwen::QMatrix packed{TensorType::Q4_0, 32, 1, std::as_bytes(std::span(storage)).first(18)};
    atomic_rejected([&] { qwen::matmul_f32(packed, x, 1, std::span(storage).subspan(4, 1)); }, storage);

    // Input and weight bytes are read-only and may alias each other.
    std::array<float, 1> y{};
    qwen::matmul_f32({TensorType::F32, 4, 1, std::as_bytes(std::span(w).first(4))}, std::span(w).first(4), 1, y);
    same(y[0], 30.0f, "read-only input/weights overlap allowed");
}

void nonfinite() {
    constexpr std::array<std::uint32_t, 4> bad_floats{0x7f800000, 0xff800000, 0x7fc12345, 0x7f800001};
    constexpr std::array<std::uint16_t, 4> bad_halfs{0x7c00, 0xfc00, 0x7e55, 0x7c01};
    constexpr std::array<std::uint16_t, 4> bad_bf16{0x7f80, 0xff80, 0x7fc1, 0x7f81};
    for (const auto type : types) {
        const auto layout = qwen::type_layout(type);
        const int width = 2 * static_cast<int>(layout.block_elements), rows = 3, columns = 3;
        const auto fixture = matrix_fixture(type, width, rows);
        std::vector<float> input(static_cast<std::size_t>(width * columns), 0.125f);
        std::vector<float> output(static_cast<std::size_t>(rows * columns), -321.0f);
        // Last activation: validation must cover all columns before any write.
        for (const auto bad : bad_floats) {
            input.back() = std::bit_cast<float>(bad);
            atomic_rejected([&] { qwen::matmul_f32({type, width, rows, fixture.bytes}, input, columns, output); }, output);
        }
        input.back() = 0.125f;
        for (std::size_t kind = 0; kind < bad_floats.size(); ++kind) {
            auto weights = fixture.bytes;
            const auto last_block = weights.size() - layout.block_bytes;
            if (type == TensorType::F32) put32(weights, last_block, bad_floats[kind]);
            else if (type == TensorType::BF16) put16(weights, last_block, bad_bf16[kind]);
            else put16(weights, last_block + (type == TensorType::Q6_K ? 208 : 0), bad_halfs[kind]);
            atomic_rejected([&] { qwen::matmul_f32({type, width, rows, weights}, input, columns, output); }, output);
            if (type == TensorType::Q4_1) {
                weights = fixture.bytes;
                put16(weights, last_block + 2, bad_halfs[kind]);
                atomic_rejected([&] { qwen::matmul_f32({type, width, rows, weights}, input, columns, output); }, output);
            }
        }
    }
    // Finite operands can overflow late in the last column. The checked dry run
    // rejects this too, before writing the valid early rows/columns.
    std::array<float, 4> weights{1.0f, 1.0f, 1.0f, 2.0f};
    std::array<float, 6> input{0.25f, 0.25f, 0.5f, 0.5f, 0.0f, std::numeric_limits<float>::max()};
    std::array<float, 6> output{};
    std::fill(output.begin(), output.end(), -321.0f);
    atomic_rejected([&] { qwen::matmul_f32({TensorType::F32, 2, 2, std::as_bytes(std::span(weights))}, input, 3, output); }, output);
    weights = {1.0f, 1.0f, 0.0f, 0.0f};
    input[4] = std::numeric_limits<float>::max();
    atomic_rejected([&] { qwen::matmul_f32({TensorType::F32, 2, 2, std::as_bytes(std::span(weights))}, input, 3, output); }, output);
}

} // namespace

int main() {
    try {
        ieee_readers();
        quant_readers();
        matrices();
        invalid_shapes();
        aliases();
        nonfinite();
        std::cout << "{\"test\":\"dense\",\"checks\":" << checks << ",\"rejections\":" << rejections
                  << ",\"block_cases\":" << block_cases << ",\"matrix_cases\":" << matrix_cases
                  << ",\"f16_patterns\":65536,\"bf16_patterns\":65536,\"max_matrix_error\":0,\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dense-test: " << error.what() << '\n';
        return 1;
    }
}
