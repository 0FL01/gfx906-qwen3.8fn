#include "linear_reference.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
std::size_t allocations = 0;
[[gnu::noinline]] void* allocate(std::size_t n) {
    ++allocations;
    if (auto* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void* allocate_aligned(std::size_t n, std::size_t alignment) {
    ++allocations;
    if (n == 0) n = 1;
    if (n > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc();
    n = ((n + alignment - 1) / alignment) * alignment;
    if (auto* p = std::aligned_alloc(alignment, n)) return p;
    throw std::bad_alloc();
}
}

[[gnu::noinline]] void* operator new(std::size_t n) { return allocate(n); }
[[gnu::noinline]] void* operator new[](std::size_t n) { return allocate(n); }
[[gnu::noinline]] void* operator new(std::size_t n, std::align_val_t a) {
    return allocate_aligned(n, static_cast<std::size_t>(a));
}
[[gnu::noinline]] void* operator new[](std::size_t n, std::align_val_t a) {
    return allocate_aligned(n, static_cast<std::size_t>(a));
}
[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {
using qwen::Q8_1;
using qwen::TensorType;
constexpr std::array types{TensorType::Q4_0, TensorType::Q4_1, TensorType::Q5_0,
                           TensorType::Q8_0, TensorType::Q6_K};
std::size_t checks = 0, rejections = 0, matrix_cases = 0, hot_calls = 0, hot_allocations = 0;
std::size_t q5_high_bits = 0, q6_packed_bits = 0, q6_position_codes = 0, q6_subscales = 0;
float max_error = 0.0f, max_q4_error = 0.0f;

void check(bool condition, const char* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

void same(float actual, float expected, const char* description) {
    check(std::isfinite(actual) && std::isfinite(expected), "nonfinite test comparison");
    max_error = std::max(max_error, std::fabs(actual - expected));
    check(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(expected), description);
}

// Explicit FP32 rounding at every arithmetic operation. Expected values use
// logical codes, double exact products, and a volatile float rounding barrier;
// they never invoke a production decoder or GPU word/bit-index expression.
float f32(double value) { volatile float result = static_cast<float>(value); return result; }
float add(float a, float b) { return f32(double(a) + double(b)); }
float sub(float a, float b) { return f32(double(a) - double(b)); }
float mul(float a, float b) { return f32(double(a) * double(b)); }

float half(std::uint16_t bits) {
    const int exponent = (bits / 1024) % 32, fraction = bits % 1024;
    const float value = exponent == 0 ? std::ldexp(float(fraction), -24) :
                        exponent < 31 ? std::ldexp(float(1024 + fraction), exponent - 25) :
                        fraction == 0 ? std::numeric_limits<float>::infinity() :
                                        std::numeric_limits<float>::quiet_NaN();
    return bits >= 0x8000 ? -value : value;
}

// Nearest lattice neighbor, ties to even: independent of float_to_half's bit
// arithmetic. All callers supply finite FP32 operands, including overflow ties.
float half_rne(float value) {
    const double magnitude = std::fabs(double(value));
    if (magnitude >= 65520.0) return std::copysign(std::numeric_limits<float>::infinity(), value);
    unsigned lo = 0, hi = 0x7c00;
    while (lo < hi) {
        const auto mid = (lo + hi) / 2;
        if (double(half(static_cast<std::uint16_t>(mid))) < magnitude) lo = mid + 1;
        else hi = mid;
    }
    unsigned nearest = lo;
    if (lo == 0x7c00) nearest = lo - 1;
    else if (lo > 0) {
        const double below = magnitude - double(half(static_cast<std::uint16_t>(lo - 1)));
        const double above = double(half(static_cast<std::uint16_t>(lo))) - magnitude;
        if (below < above || (below == above && (lo % 2) != 0)) nearest = lo - 1;
    }
    return std::copysign(half(static_cast<std::uint16_t>(nearest)), value);
}

struct Layout { std::size_t elements, bytes; };
Layout layout(TensorType type) {
    switch (type) {
    case TensorType::Q4_0: return {32, 18};
    case TensorType::Q4_1: return {32, 20};
    case TensorType::Q5_0: return {32, 22};
    case TensorType::Q8_0: return {32, 34};
    case TensorType::Q6_K: return {256, 210};
    default: throw std::runtime_error("bad test type");
    }
}

void put16(std::span<std::byte> bytes, std::size_t at, std::uint16_t bits) {
    bytes[at] = std::byte(bits % 256);
    bytes[at + 1] = std::byte(bits / 256);
}

struct Block {
    // Q4/Q5 codes are UNSIGNED (offsets are applied using raw sum); Q8 and
    // Q6 codes are signed. This is independent of the production storage.
    std::array<int, 256> codes{};
    std::array<int, 16> subscales{};
    std::uint16_t d = 0x3c00, m = 0;
};

std::vector<std::byte> pack(TensorType type, const Block& block) {
    std::vector<std::byte> bytes(layout(type).bytes);
    if (type == TensorType::Q6_K) {
        put16(bytes, 208, block.d);
        for (std::size_t s = 0; s < 16; ++s) bytes[192 + s] = std::byte(block.subscales[s] & 255);
        // Encode FOUR logical quarters together, no decoded-index lookup.
        for (std::size_t h = 0; h < 2; ++h) {
            for (std::size_t lane = 0; lane < 32; ++lane) {
                std::array<unsigned, 4> codes{};
                for (std::size_t q = 0; q < 4; ++q)
                    codes[q] = static_cast<unsigned>(32 + block.codes[h * 128 + q * 32 + lane]);
                bytes[h * 64 + lane] = std::byte(codes[0] % 16 + 16 * (codes[2] % 16));
                bytes[h * 64 + 32 + lane] = std::byte(codes[1] % 16 + 16 * (codes[3] % 16));
                bytes[128 + h * 32 + lane] = std::byte(codes[0] / 16 + 4 * (codes[1] / 16) +
                                                       16 * (codes[2] / 16) + 64 * (codes[3] / 16));
            }
        }
    } else {
        put16(bytes, 0, block.d);
        if (type == TensorType::Q8_0) {
            for (std::size_t j = 0; j < 32; ++j) bytes[2 + j] = std::byte(block.codes[j] & 255);
        } else {
            const auto at = type == TensorType::Q4_0 ? 2U : type == TensorType::Q4_1 ? 4U : 6U;
            if (type == TensorType::Q4_1) put16(bytes, 2, block.m);
            for (std::size_t j = 0; j < 16; ++j)
                bytes[at + j] = std::byte(block.codes[j] % 16 + 16 * (block.codes[j + 16] % 16));
            if (type == TensorType::Q5_0) {
                for (std::size_t group = 0; group < 4; ++group) {
                    unsigned plane = 0;
                    for (std::size_t bit = 0; bit < 8; ++bit)
                        plane += static_cast<unsigned>(block.codes[group * 8 + bit] / 16) * (1U << bit);
                    bytes[2 + group] = std::byte(plane);
                }
            }
        }
    }
    return bytes;
}

float expected32(TensorType type, const Block& block, const Q8_1& x) {
    int dot = 0;
    for (std::size_t j = 0; j < 32; ++j) dot += block.codes[j] * int(x.qs[j]);
    const float dw = half(block.d), dx = half(x.d), sx = half(x.s);
    if (type == TensorType::Q4_1)
        return add(mul(float(dot), half_rne(mul(dw, dx))), half_rne(mul(half(block.m), sx)));
    if (type == TensorType::Q8_0) return mul(mul(dw, dx), float(dot));
    const float correction = type == TensorType::Q4_0 ? 8.0f : 16.0f;
    return mul(dw, sub(mul(float(dot), dx), mul(correction, sx)));
}

float expected_row(TensorType type, std::span<const Block> blocks, std::span<const Q8_1> x) {
    float acc = 0.0f;
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const auto& block = blocks[b];
        if (type != TensorType::Q6_K) { acc = add(acc, expected32(type, block, x[b])); continue; }
        // Natural logical quarter traversal: pair quarters (0,2) then (1,3),
        // grouped into fours, within each 128-element half. This independently
        // expresses MMVQ FP32 grouping without reproducing packed word indexing.
        for (std::size_t h = 0; h < 2; ++h) {
            for (std::size_t first = 0; first < 2; ++first) {
                for (std::size_t group = 0; group < 8; ++group) {
                    float pair = 0.0f;
                    for (std::size_t q = first; q < 4; q += 2) {
                        int dot = 0;
                        for (std::size_t t = 0; t < 4; ++t)
                            dot += block.codes[h * 128 + q * 32 + group * 4 + t] *
                                   int(x[b * 8 + h * 4 + q].qs[group * 4 + t]);
                        const int scaled = dot * block.subscales[h * 8 + q * 2 + group / 4];
                        pair = add(pair, mul(half(x[b * 8 + h * 4 + q].d), float(scaled)));
                    }
                    acc = add(acc, mul(half(block.d), pair));
                }
            }
        }
    }
    return acc;
}

void call(const qwen::QMatrix& matrix, std::span<const Q8_1> x, int columns, std::span<float> y) {
    const auto before = allocations;
    qwen::matmul_q8_reference(matrix, x, columns, y);
    hot_allocations += allocations - before;
    ++hot_calls;
    check(allocations == before, "successful oracle allocated");
}

void verify(TensorType type, int rows, int columns, std::span<const Block> blocks,
            std::span<const Q8_1> input, std::span<const std::byte> weights) {
    const auto count = blocks.size() / static_cast<std::size_t>(rows);
    const auto width = count * layout(type).elements;
    const auto q8_count = width / 32;
    const qwen::QMatrix matrix{type, static_cast<int>(width), rows, weights};
    const std::vector<std::byte> saved_weights(weights.begin(), weights.end());
    const auto xb = std::as_bytes(input);
    const std::vector<std::byte> saved_input(xb.begin(), xb.end());
    std::vector<float> guarded(static_cast<std::size_t>(rows * columns) + 2, -12345.0f);
    auto output = std::span(guarded).subspan(1, guarded.size() - 2);
    std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
    call(matrix, input, columns, output);
    for (int c = 0; c < columns; ++c)
        for (int r = 0; r < rows; ++r)
            same(output[static_cast<std::size_t>(c * rows + r)],
                 expected_row(type, blocks.subspan(static_cast<std::size_t>(r) * count, count),
                              input.subspan(static_cast<std::size_t>(c) * q8_count, q8_count)),
                 "logical common-Q8 expected differs");
    if (type == TensorType::Q4_0 || type == TensorType::Q4_1) {
        std::vector<float> qualified(output.size());
        qwen::matmul_scalar(matrix, input, columns, qualified);
        for (std::size_t j = 0; j < output.size(); ++j) {
            max_q4_error = std::max(max_q4_error, std::fabs(output[j] - qualified[j]));
            same(output[j], qualified[j], "qualified existing Q4 scalar differs");
        }
    }
    same(guarded.front(), -12345.0f, "output prefix guard changed");
    same(guarded.back(), -12345.0f, "output suffix guard changed");
    check(std::equal(weights.begin(), weights.end(), saved_weights.begin()), "weight bytes changed");
    check(std::equal(xb.begin(), xb.end(), saved_input.begin()), "Q8 input bytes changed");
    ++matrix_cases;
}

std::vector<std::byte> pack_matrix(TensorType type, std::span<const Block> blocks) {
    std::vector<std::byte> bytes;
    for (const auto& block : blocks) {
        const auto packed = pack(type, block);
        bytes.insert(bytes.end(), packed.begin(), packed.end());
    }
    return bytes;
}

std::vector<Q8_1> activations(std::size_t count) {
    constexpr std::array<std::uint16_t, 8> d{0x3555, 0xb001, 0x0001, 0x8001, 0x3801, 0xb555, 0x0000, 0x8000};
    constexpr std::array<std::uint16_t, 8> s{0xb101, 0x2c01, 0x8001, 0x0001, 0x4c01, 0xbc01, 0x0000, 0x8000};
    std::vector<Q8_1> x(count);
    for (std::size_t b = 0; b < count; ++b) {
        x[b].d = d[b % d.size()]; x[b].s = s[(b * 3) % s.size()];
        for (std::size_t j = 0; j < 32; ++j)
            x[b].qs[j] = static_cast<std::int8_t>(static_cast<int>((j * 29 + b * 17) % 256) - 128);
    }
    return x;
}

std::vector<Block> matrix_blocks(TensorType type, std::size_t count) {
    constexpr std::array<std::uint16_t, 8> d{0x3555, 0xb555, 0x2801, 0xac01, 0x0001, 0x8001, 0x0000, 0x8000};
    std::vector<Block> blocks(count);
    const auto range = type == TensorType::Q8_0 ? 256 : type == TensorType::Q6_K ? 64 :
                       type == TensorType::Q5_0 ? 32 : 16;
    const int bias = type == TensorType::Q8_0 || type == TensorType::Q6_K ? range / 2 : 0;
    for (std::size_t b = 0; b < count; ++b) {
        blocks[b].d = d[b % d.size()]; blocks[b].m = d[(b * 3 + 1) % d.size()];
        for (std::size_t j = 0; j < layout(type).elements; ++j)
            blocks[b].codes[j] = static_cast<int>((j * 13 + b * 7 + j / 16) % static_cast<unsigned>(range)) - bias;
        for (std::size_t s = 0; s < 16; ++s)
            blocks[b].subscales[s] = static_cast<int>((s * 19 + b * 23) % 256) - 128;
    }
    return blocks;
}

void matrices() {
    for (const auto type : types) {
        // Weight-block boundaries relevant to wave64 dispatch. Q6 separately
        // covers Q8-block widths64/65 below (65 is invalid for block256).
        for (const auto count : {1, 2, 64, 65, 512}) {
            const int rows = count == 512 ? 3 : 5;
            const auto blocks = matrix_blocks(type, static_cast<std::size_t>(count * rows));
            const auto bytes = pack_matrix(type, blocks);
            std::vector<std::byte> unaligned(bytes.size() + 1);
            std::copy(bytes.begin(), bytes.end(), unaligned.begin() + 1);
            for (int columns = 1; columns <= 3; ++columns) {
                const auto x = activations(static_cast<std::size_t>(count * columns) * (type == TensorType::Q6_K ? 8 : 1));
                verify(type, rows, columns, blocks, x, std::span(unaligned).subspan(1));
            }
        }
        for (const int width : {2560, 6144, 10240, 16384}) {
            const auto count = static_cast<std::size_t>(width) / layout(type).elements;
            const auto blocks = matrix_blocks(type, count * 3);
            const auto bytes = pack_matrix(type, blocks);
            for (int columns = 1; columns <= 3; ++columns)
                verify(type, 3, columns, blocks, activations(static_cast<std::size_t>(width / 32 * columns)), bytes);
        }
    }
}

void q5_positions() {
    std::array<Block, 1> blocks{};
    std::array<Q8_1, 1> x{}; x[0].d = 0x3801; x[0].s = 0xb555;
    for (std::size_t j = 0; j < 32; ++j) {
        blocks[0].codes.fill(0); blocks[0].codes[j] = 16;
        auto bytes = pack(TensorType::Q5_0, blocks[0]);
        // Isolate each physical high bit with signed -128 activation. The raw
        // sum is deliberately unrelated to d8*sum(q8), testing offset semantics.
        for (std::size_t b = 2; b < 6; ++b)
            check(bytes[b] == std::byte(b == 2 + j / 8 ? 1U << (j % 8) : 0), "Q5 physical high bit");
        std::fill(std::begin(x[0].qs), std::end(x[0].qs), std::int8_t{0}); x[0].qs[j] = -128;
        verify(TensorType::Q5_0, 1, 1, blocks, x, bytes);
        ++q5_high_bits;
    }
    for (int code = 0; code < 32; ++code) {
        blocks[0].codes.fill(code);
        std::fill(std::begin(x[0].qs), std::end(x[0].qs), std::int8_t{127});
        verify(TensorType::Q5_0, 1, 1, blocks, x, pack(TensorType::Q5_0, blocks[0]));
    }
}

void q6_positions() {
    std::array<Block, 1> blocks{};
    blocks[0].subscales.fill(1);
    auto x = activations(8);
    // One physical bit at a time: invert its logical destination mathematically
    // from the four-quarter storage plane. Expected values use only the logical
    // delta and an activation at that destination, not production decoding.
    for (std::size_t at = 0; at < 192; ++at) {
        for (unsigned bit = 0; bit < 8; ++bit) {
            blocks[0].codes.fill(-32);
            std::vector<std::byte> bytes(210);
            put16(bytes, 208, blocks[0].d);
            std::fill(bytes.begin() + 192, bytes.begin() + 208, std::byte{1});
            bytes[at] = std::byte(1U << bit);
            std::size_t element; int delta;
            if (at < 128) {
                const auto h = at / 64, local = at % 64;
                element = h * 128 + local + (bit / 4) * 64;
                delta = 1 << (bit % 4);
            } else {
                const auto plane = at - 128;
                element = (plane / 32) * 128 + (bit / 2) * 32 + plane % 32;
                delta = 16 << (bit % 2);
            }
            blocks[0].codes[element] += delta;
            for (auto& a : x) std::fill(std::begin(a.qs), std::end(a.qs), std::int8_t{0});
            x[element / 32].qs[element % 32] = -128;
            x[element / 32].d = 0x3555;
            verify(TensorType::Q6_K, 1, 1, blocks, x, bytes);
            ++q6_packed_bits;
        }
    }
    // Each logical position exercises ALL 64 six-bit codes, including -32/31.
    for (std::size_t j = 0; j < 256; ++j) {
        blocks[0].codes.fill(0);
        for (auto& a : x) { a.d = 0x3555; a.s = 0xb555; std::fill(std::begin(a.qs), std::end(a.qs), std::int8_t{0}); }
        x[j / 32].qs[j % 32] = j % 2 ? -128 : 127;
        for (int code = -32; code <= 31; ++code) {
            blocks[0].codes[j] = code;
            verify(TensorType::Q6_K, 1, 1, blocks, x, pack(TensorType::Q6_K, blocks[0]));
            ++q6_position_codes;
        }
    }
    // Every subscale slot with all signed byte patterns; codes and activations
    // in every part of the slot prove the 16-element boundary and -128 support.
    blocks[0].codes.fill(-32);
    for (auto& a : x) { a.d = 0xb001; std::fill(std::begin(a.qs), std::end(a.qs), std::int8_t{-128}); }
    for (std::size_t s = 0; s < 16; ++s) {
        blocks[0].subscales.fill(0);
        for (int scale = -128; scale <= 127; ++scale) {
            blocks[0].subscales[s] = scale;
            verify(TensorType::Q6_K, 1, 1, blocks, x, pack(TensorType::Q6_K, blocks[0]));
            ++q6_subscales;
        }
    }
}

void arithmetic() {
    constexpr std::array<std::uint16_t, 9> scales{0x3c00, 0xbc00, 0x3555, 0xb555, 0x0001,
                                                 0x8001, 0x0000, 0x8000, 0x7bff};
    for (const auto type : types) {
        auto blocks = matrix_blocks(type, 1);
        auto x = activations(type == TensorType::Q6_K ? 8 : 1);
        for (const auto d : scales) {
            blocks[0].d = d;
            blocks[0].m = 0xb001;
            // Largest finite weight scale with tiny activation product remains
            // valid for Q4_1: overflow is tested explicitly in rejection fixtures.
            for (auto& a : x) { a.d = 0x0001; a.s = 0x3555; }
            verify(type, 1, 1, blocks, x, pack_matrix(type, blocks));
        }
    }
    std::array<Block, 1> blocks{};
    std::array<Q8_1, 1> x{}; x[0].d = 0x3c01; x[0].s = 0x3801;
    blocks[0].d = 0x3c01; blocks[0].m = 0xbc01; blocks[0].codes.fill(15);
    x[0].qs[0] = -128; x[0].qs[31] = 127;
    const float dd = mul(half(blocks[0].d), half(x[0].d));
    check(dd != half_rne(dd), "half product fixture distinguishes FP32 and half rounding");
    verify(TensorType::Q4_1, 1, 1, blocks, x, pack(TensorType::Q4_1, blocks[0]));
    const float fp32_q4 = add(mul(-15.0f, dd), mul(half(blocks[0].m), half(x[0].s)));
    check(expected32(TensorType::Q4_1, blocks[0], x[0]) != fp32_q4, "Q4_1 rounded products are observable");
    blocks[0].codes.fill(127);
    verify(TensorType::Q8_0, 1, 1, blocks, x, pack(TensorType::Q8_0, blocks[0]));
    check(expected32(TensorType::Q8_0, blocks[0], x[0]) != mul(half_rne(dd), -127.0f),
          "Q8_0 must not round its product to half");
    blocks[0].codes.fill(-128);
    std::fill(std::begin(x[0].qs), std::end(x[0].qs), std::int8_t{-128});
    verify(TensorType::Q8_0, 1, 1, blocks, x, pack(TensorType::Q8_0, blocks[0]));

    // Cancellation of ascending full-block accumulation: 2^24 + 1 - 2^24.
    std::array<Block, 3> cancellation{};
    std::array<Q8_1, 3> cx{};
    for (std::size_t b = 0; b < 3; ++b) { cancellation[b].codes[0] = 1; cx[b].qs[0] = 1; }
    cancellation[0].d = cancellation[2].d = 0x6c00; cx[0].d = cx[2].d = 0x6c00;
    cancellation[1].d = cx[1].d = 0x3c00; cx[2].qs[0] = -1;
    same(expected_row(TensorType::Q8_0, cancellation, cx), 0.0f, "FP32 cancellation fixture");
    verify(TensorType::Q8_0, 1, 1, cancellation, cx, pack_matrix(TensorType::Q8_0, cancellation));

    // Q6 grouping must apply d6 AFTER each two-term pair, then accumulate in
    // MMVQ slice order, rather than a raw element-by-element FP32 dot product.
    auto q6 = matrix_blocks(TensorType::Q6_K, 2);
    auto qx = activations(16);
    float elementwise = 0.0f;
    for (std::size_t b = 0; b < 2; ++b) {
        for (std::size_t j = 0; j < 256; ++j) {
            const float weight = mul(mul(half(q6[b].d), float(q6[b].subscales[j / 16])), float(q6[b].codes[j]));
            const float activation = mul(half(qx[b * 8 + j / 32].d), float(qx[b * 8 + j / 32].qs[j % 32]));
            elementwise = add(elementwise, mul(weight, activation));
        }
    }
    check(expected_row(TensorType::Q6_K, q6, qx) != elementwise, "Q6 fixture distinguishes common-Q8 MMVQ and raw dense arithmetic");
    verify(TensorType::Q6_K, 1, 1, q6, qx, pack_matrix(TensorType::Q6_K, q6));
}

template<class Function> void atomic_rejected(Function function, std::span<const std::byte> observed) {
    const std::vector<std::byte> saved(observed.begin(), observed.end());
    bool threw = false;
    try { function(); }
    catch (const std::invalid_argument&) { threw = true; ++rejections; }
    check(threw, "invalid call unexpectedly accepted");
    check(std::equal(saved.begin(), saved.end(), observed.begin()), "rejection mutated storage");
}

void shapes() {
    for (const auto type : types) {
        const auto blocks = matrix_blocks(type, 2);
        const auto bytes = pack_matrix(type, blocks);
        const auto width = static_cast<int>(layout(type).elements);
        auto x = activations(static_cast<std::size_t>(width / 32) * 2);
        std::array<float, 5> output{-321, -321, -321, -321, -321};
        auto out = std::span(output).first(4);
        const qwen::QMatrix matrix{type, width, 2, bytes};
        const auto fail = [&](qwen::QMatrix m, std::span<const Q8_1> a, int c, std::span<float> y) {
            atomic_rejected([&] { qwen::matmul_q8_reference(m, a, c, y); }, std::as_bytes(std::span(output)));
        };
        for (const int bad : {0, -1, std::numeric_limits<int>::min()}) {
            auto m = matrix; m.input = bad; fail(m, x, 2, out);
            m = matrix; m.output = bad; fail(m, x, 2, out);
            fail(matrix, x, bad, out);
        }
        fail(matrix, x, 4, out); fail(matrix, x, std::numeric_limits<int>::max(), out);
        auto m = matrix; m.input = width - 1; fail(m, x, 2, out);
        m.input = width + 1; fail(m, x, 2, out);
        m = matrix; m.weights = matrix.weights.first(matrix.weights.size() - 1); fail(m, x, 2, out);
        auto extra = bytes; extra.push_back(std::byte{0}); m.weights = extra; fail(m, x, 2, out);
        fail(matrix, std::span(x).first(x.size() - 1), 2, out);
        auto longer = x; longer.push_back(Q8_1{}); fail(matrix, longer, 2, out);
        fail(matrix, x, 2, out.first(3)); fail(matrix, x, 2, output);
        // Largest admissible int dimensions: multiplication/size checks must
        // reject without dereferencing the small supplied spans on any ABI.
        m = matrix; m.input = std::numeric_limits<int>::max() - (std::numeric_limits<int>::max() % width);
        m.output = std::numeric_limits<int>::max(); fail(m, x, 3, out);
        m = matrix; m.weights = {}; fail(m, x, 2, out);
        fail(matrix, {}, 2, out); fail(matrix, x, 2, {});
        // Nonempty spans require live memory; empty null spans above exercise
        // null rejection without inventing an invalid C++ pointer/count range.
    }
    const auto blocks = matrix_blocks(TensorType::Q6_K, 1);
    const auto bytes = pack_matrix(TensorType::Q6_K, blocks);
    auto x = activations(65);
    std::array<float, 1> y{-321};
    atomic_rejected([&] { qwen::matmul_q8_reference({TensorType::Q6_K, 65 * 32, 1, bytes}, x, 1, y); }, std::as_bytes(std::span(y)));
    // Q6 valid 64/512 activation-block boundaries in addition to the weight
    // boundaries covered by matrices().
    for (const int width : {64 * 32, 512 * 32}) {
        const auto bs = matrix_blocks(TensorType::Q6_K, static_cast<std::size_t>(width / 256));
        verify(TensorType::Q6_K, 1, 1, bs, activations(static_cast<std::size_t>(width / 32)), pack_matrix(TensorType::Q6_K, bs));
    }
    for (const auto unsupported : {TensorType::F32, TensorType::F16, TensorType::BF16, static_cast<TensorType>(5)})
        atomic_rejected([&] { qwen::matmul_q8_reference({unsupported, 32, 1, {}}, std::span(x).first(1), 1, y); }, std::as_bytes(std::span(y)));
}

void aliases() {
    const auto type = TensorType::Q8_0;
    const auto blocks = matrix_blocks(type, 3);
    const auto bytes = pack_matrix(type, blocks);
    auto x = activations(3);
    // Float object storage supplies valid output objects; packed weights are
    // read-only character views. These cover either edge and containment.
    std::array<float, 64> storage{};
    auto backing = std::as_writable_bytes(std::span(storage));
    std::copy(bytes.begin(), bytes.end(), backing.begin() + 4);
    const qwen::QMatrix matrix{type, 32, 3, std::span(backing).subspan(4, bytes.size())};
    for (const std::size_t at : {0, 1, 24, 25}) {
        auto out = std::span(storage).subspan(at, 3);
        atomic_rejected([&] { qwen::matmul_q8_reference(matrix, std::span(x).first(1), 1, out); }, backing);
    }
    // Place an actual float object at an aligned address inside Q8 code bytes.
    // Alignment/lifetime are valid; alias validation must run before any Q8
    // code reads. A separate fixture handles shared read-only input/weights.
    for (const std::size_t at : {4, 32}) {
        auto input = activations(1);
        auto* slot = reinterpret_cast<float*>(reinterpret_cast<std::byte*>(input.data()) + at);
        ::new (static_cast<void*>(slot)) float(-321.0f);
        auto out = std::span<float>(slot, 1);
        const auto w = pack(TensorType::Q8_0, blocks[0]);
        const qwen::QMatrix m{type, 32, 1, w};
        atomic_rejected([&] { qwen::matmul_q8_reference(m, input, 1, out); }, std::as_bytes(std::span(input)));
    }
    auto out = std::span(storage).subspan(27, 3); // past weight range [4,106)
    call(matrix, std::span(x).first(1), 1, out);
    for (std::size_t r = 0; r < 3; ++r) same(out[r], expected32(type, blocks[r], x[0]), "adjacent weight/output storage");
    // A Q8_1 also supplies a valid canonical Q8_0 byte view. Immutable readers
    // may overlap; outputs are still distinct. Derive its logical values here.
    Q8_1 shared{}; shared.d = 0x3555; shared.s = 0x8001;
    for (std::size_t j = 0; j < 32; ++j) shared.qs[j] = static_cast<std::int8_t>(static_cast<int>(j) - 16);
    Block logical{}; logical.d = shared.d;
    logical.codes[0] = 1; logical.codes[1] = -128; // little-endian half(s) bytes
    for (std::size_t j = 2; j < 32; ++j) logical.codes[j] = shared.qs[j - 2];
    std::array<float, 1> y{};
    call({type, 32, 1, std::as_bytes(std::span(&shared, 1)).first(34)}, std::span(&shared, 1), 1, y);
    same(y[0], expected32(type, logical, shared), "overlapping read-only weight/input accepted");
}

void nonfinite() {
    constexpr std::array<std::uint16_t, 4> bad{0x7c00, 0xfc00, 0x7e55, 0x7c01};
    for (const auto type : types) {
        const auto bs = matrix_blocks(type, 6);
        const auto canonical = pack_matrix(type, bs);
        const int width = static_cast<int>(2 * layout(type).elements);
        auto x = activations(static_cast<std::size_t>(width / 32 * 3));
        std::array<float, 9> output{}; std::fill(output.begin(), output.end(), -321.0f);
        const auto fail = [&](std::span<const std::byte> bytes) {
            atomic_rejected([&] { qwen::matmul_q8_reference({type, width, 3, bytes}, x, 3, output); }, std::as_bytes(std::span(output)));
        };
        for (const auto bits : bad) {
            const auto saved = x.back();
            x.back().d = bits; fail(canonical); x.back() = saved;
            x.back().s = bits; fail(canonical); x.back() = saved;
            auto poisoned = canonical;
            const auto last = poisoned.size() - layout(type).bytes;
            put16(poisoned, last + (type == TensorType::Q6_K ? 208 : 0), bits); fail(poisoned);
            if (type == TensorType::Q4_1) { poisoned = canonical; put16(poisoned, last + 2, bits); fail(poisoned); }
        }
        // No state is retained by the reference: immediate successful reuse.
        verify(type, 3, 3, bs, x, canonical);
    }
    // Finite operands overflow half only in the last row AND last column.
    // Even zero integer dots must not mask an invalid half product.
    auto bs = matrix_blocks(TensorType::Q4_1, 6);
    for (auto& b : bs) { b.d = 0x0001; b.m = 0x0001; b.codes.fill(0); }
    auto x = activations(6);
    for (auto& a : x) { a.d = 0x0001; a.s = 0x0001; }
    x.back().d = 0x7bff; bs.back().d = 0x7bff;
    std::array<float, 9> y{}; std::fill(y.begin(), y.end(), -321.0f);
    const auto fail = [&] {
        const auto bytes = pack_matrix(TensorType::Q4_1, bs);
        atomic_rejected([&] { qwen::matmul_q8_reference({TensorType::Q4_1, 64, 3, bytes}, x, 3, y); }, std::as_bytes(std::span(y)));
    };
    fail(); bs.back().codes.fill(15); fail();
    bs.back().d = 0x0001; bs.back().m = 0x7bff; x.back().s = 0x7bff; fail();
    // Opposite-sign infinities from finite half products are rejected too.
    bs.back().d = 0x7bff; bs.back().m = 0xfbff; fail();
    bs.back().d = bs.back().m = 0x0001;
    verify(TensorType::Q4_1, 3, 3, bs, x, pack_matrix(TensorType::Q4_1, bs));
}

} // namespace

int main() {
    try {
        check(std::fegetround() == FE_TONEAREST, "tests require round-to-nearest FP32");
        matrices(); q5_positions(); q6_positions(); arithmetic(); shapes(); aliases(); nonfinite();
        std::cout << "{\"test\":\"linear_reference\",\"checks\":" << checks << ",\"rejections\":" << rejections
                  << ",\"matrix_cases\":" << matrix_cases << ",\"q5_high_bits\":" << q5_high_bits
                  << ",\"q6_packed_bits\":" << q6_packed_bits << ",\"q6_position_codes\":" << q6_position_codes
                  << ",\"q6_subscales\":" << q6_subscales << ",\"hot_calls\":" << hot_calls
                  << ",\"hot_allocations\":" << hot_allocations << ",\"max_common_q8_error\":" << max_error
                  << ",\"max_qualified_q4_error\":" << max_q4_error << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "linear-reference-test: " << error.what() << '\n';
        return 1;
    }
}
