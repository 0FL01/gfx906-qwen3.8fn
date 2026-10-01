#pragma once
#include "model.hpp"
#include <cstdint>
#include <span>

namespace qwen {
// Canonical GGML Q4 blocks, no changed weight values or repack on the host.
struct Q4_0 { std::uint16_t d; std::uint8_t qs[16]; };
struct Q4_1 { std::uint16_t d, m; std::uint8_t qs[16]; };
// mx GPU Q8_1 activation ABI: half(d), half(sum of RAW float inputs), 32 int8.
// Scale uses amax/127 before half rounding; codes use roundf(x/original_d).
// Raw sum uses ascending XOR butterfly 1/2/4/8/16 (matching gfx906 DPP).
struct Q8_1 { std::uint16_t d, s; std::int8_t qs[32]; };
static_assert(sizeof(Q4_0)==18 && sizeof(Q4_1)==20 && sizeof(Q8_1)==36);

float half_to_float(std::uint16_t bits);
std::uint16_t float_to_half(float value);
bool cpu_has_avx2(); // Checks AVX2/FMA/F16C before target-attributed functions.
void quantize_q8(std::span<const float> input, std::span<Q8_1> output);
float dot_q4_scalar(const void *block, TensorType type, const Q8_1 &activation);
float dot_q4_avx2(const void *block, TensorType type, const Q8_1 &activation);
// Q4_1 retains mx FAST_FP16_AVAILABLE half-rounded d4*d8 and m4*s8 products;
// Q4_0 uses FP32 d4*(integer_dot*d8-8*s8). Both retain signed scales.
struct QMatrix {
    TensorType type;
    int input, output;
    std::span<const std::byte> weights;
};
// Column-major activations/outputs, contiguous blocks for each input column.
// Validates shapes and sizes. No threads/allocations in each matrix multiply.
void matmul_scalar(QMatrix weights, std::span<const Q8_1> input,
                   int columns, std::span<float> output);
void matmul_avx2(QMatrix weights, std::span<const Q8_1> input,
                 int columns, std::span<float> output);
// Diagnostic A/B baseline: original per-block AVX2 call + portable half scales.
// Not runtime dispatch; shares the scalar reference loop, no second backend.
void matmul_avx2_baseline(QMatrix weights, std::span<const Q8_1> input,
                        int columns, std::span<float> output);
}
