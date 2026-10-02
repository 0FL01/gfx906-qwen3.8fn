#include "session_ops.cuh"
#include "dpp.cuh"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Session operations require finite checks; compile without fast/finite-only math"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "session_ops.hip is specialized for gfx906 wave64"
#endif
#pragma clang fp contract(off)

// Donor-first: FP32 N1/2/3, Q24/K2/D256/rotary64, router512, expert640.
// ADAPT mx dcd685463d597d31f5ca759d32c94592a2740fa4, ggml/src/ggml-cuda/
// rope.cu::rope_neox / rope_multi: split-half pair j/j+rotary_dim/2 and text
// absolute positions. One thread owns both pair members for race-free exact
// alias. Caller-supplied frequencies replace powf/position arrays; no YaRN,
// norm, set_rows, strides or runtime import. Nonrotary tail remains FP32.
// ADAPT actual rope_multi binary: host powf(base,-2/64), device powf(scale,j);
// precompute the same FP32 values once, and retain explicit rotation FMAs with
// separately rounded sine products. Ordinary cosf/sinf, no fast intrinsics.
// ADAPT same revision common.cuh::warp_reduce_max shuffle fallback and
// topk-moe.cu::softmax_warp_inplace: logical width32, sixteen logits/lane,
// ascending FP32 sums and reciprocal multiply. KEEP qualified dpp.cuh::sum32,
// including mandatory VALU-to-DPP hazard waits; no LDS/global workspace.
// Checked furnace 905021dbad71c5056ef51f9fd45d545403fc989c rope.cu and RECON16
// donor map; reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8 has no different
// applicable Qwen PLE/HC equations for these standalone wiring operations.
// ADAPT mx unary.cu/unary.cuh direct sigmoid/SiLU; WRITE OURS interleaved split,
// pointwise add/scaled-add,
// borrowed-range checks and sticky-error sanitization. No extra forward norms;
// sigmoid Q gate is applied after inverse Hadamard, route weights after down.
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace qwen {
namespace {

constexpr int threads = 256;
constexpr int wave_size = 64;
constexpr int router_width = 32;
constexpr int head_dim = session_ops_qsa_head_dim;
constexpr int rotary_half = session_ops_qsa_rotary_dim / 2;
constexpr int pairs_per_head = head_dim / 2;
constexpr int experts = session_ops_router_experts;
static_assert(sizeof(float) == 4 && sizeof(int) == 4);
static_assert(head_dim == 256 && rotary_half == 32 && experts == 8 * wave_size);

struct Range {
    std::uintptr_t begin, end;
    bool writable, valid;
};

template<class T>
Range range(const T* pointer, std::size_t count, bool writable = false) noexcept {
    // Only the duplicate RoPE output range is omitted for exact alias.
    if (count == 0) return {0, 0, writable, true};
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (pointer == nullptr || begin % alignof(T) != 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(T))
        return {0, 0, writable, false};
    return {begin, begin + count * sizeof(T), writable, true};
}

template<std::size_t Count>
bool valid_ranges(const std::array<Range, Count>& buffers) noexcept {
    for (const auto& r : buffers) if (!r.valid) return false;
    for (std::size_t i = 0; i < Count; ++i) {
        const auto& a = buffers[i];
        if (a.begin == a.end) continue;
        for (std::size_t j = i + 1; j < Count; ++j) {
            const auto& b = buffers[j];
            if (b.begin != b.end && (a.writable || b.writable) &&
                a.begin < b.end && b.begin < a.end) return false;
        }
    }
    return true;
}

bool valid_tokens(int tokens) noexcept {
    return tokens >= 1 && tokens <= session_ops_max_tokens;
}

unsigned ceil_div(int count, int divisor) noexcept {
    // Positive int counts only; never use count+divisor-1 (INT_MAX may occur).
    return static_cast<unsigned>(count / divisor + (count % divisor != 0));
}

__device__ __forceinline__ float operand(float x, int& flags) {
    if (!isfinite(x)) { flags |= session_ops_error_invalid; return 0.0f; }
    return x;
}

__device__ __forceinline__ float arithmetic(float x, int& flags) {
    if (!isfinite(x)) { flags |= session_ops_error_arithmetic; return 0.0f; }
    return x;
}

__device__ __forceinline__ void report(int flags, int* error) {
    if (flags != 0) atomicOr(error, flags);
}

__device__ __forceinline__ float sigmoid(float x, int& flags) {
    // For finite negative x, exp overflow is expected: division gives +0.
    // Operands are validated by callers; validate the final result, not exp.
    return arithmetic(1.0f / (1.0f + expf(-x)), flags);
}

__global__ void split_q_gate(const float* projected, int count, float* query,
                             float* gate, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    const unsigned source = (i / head_dim) * (2 * head_dim) + i % head_dim;
    int flags = 0;
    const float q = operand(projected[source], flags);
    const float g = operand(projected[source + head_dim], flags);
    query[i] = flags == 0 ? q : 0.0f;
    gate[i] = flags == 0 ? g : 0.0f;
    report(flags, error);
}

__global__ void rope_frequencies(float theta_scale, float* output, int* error) {
    const int j = threadIdx.x;
    if (j >= rotary_half) return;
    int flags = 0;
    output[j] = arithmetic(powf(theta_scale, static_cast<float>(j)), flags);
    report(flags, error);
}

// ADAPT mx split-half rotation. Each thread reads BOTH inputs before either
// store. Tail pairs use [2*j,2*j+1] for j>=32, disjoint from rotary [0,64).
__global__ void qsa_rope(const float* input, const float* frequencies, int heads,
                        int start, int count, float* output, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    const unsigned row = i / pairs_per_head, pair = i % pairs_per_head;
    const unsigned base = row * head_dim;
    int flags = 0;
    unsigned first, second;
    float left, right;
    if (pair < rotary_half) {
        first = base + pair;
        second = first + rotary_half;
        const float a = operand(input[first], flags);
        const float b = operand(input[second], flags);
        const float frequency = operand(frequencies[pair], flags);
        const int position = start + static_cast<int>(row) / heads;
        const float angle = arithmetic(static_cast<float>(position) * frequency, flags);
        const float c = arithmetic(cosf(angle), flags);
        const float s = arithmetic(sinf(angle), flags);
        const float bs = arithmetic(b * s, flags);
        const float as = arithmetic(a * s, flags);
        left = arithmetic(fmaf(a, c, -bs), flags);
        right = arithmetic(fmaf(b, c, as), flags);
    } else {
        first = base + 2 * pair;
        second = first + 1;
        left = operand(input[first], flags);
        right = operand(input[second], flags);
    }
    output[first] = flags == 0 ? left : 0.0f;
    output[second] = flags == 0 ? right : 0.0f;
    report(flags, error);
}

__global__ void sigmoid_gate(const float* input, const float* gate, int count,
                            float* output, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    int flags = 0;
    const float x = operand(input[i], flags);
    const float g = operand(gate[i], flags);
    const float value = arithmetic(x * sigmoid(g, flags), flags);
    output[i] = flags == 0 ? value : 0.0f;
    report(flags, error);
}

__global__ void silu_up(const float* gate, const float* up, int count,
                       float* output, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    int flags = 0;
    const float g = operand(gate[i], flags);
    const float u = operand(up[i], flags);
    // Preserve the donor division, not g * an already-rounded sigmoid.
    const float activated = arithmetic(g / (1.0f + expf(-g)), flags);
    const float value = arithmetic(activated * u, flags);
    output[i] = flags == 0 ? value : 0.0f;
    report(flags, error);
}

__global__ void add(const float* a, const float* b, int count, float* output, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    int flags = 0;
    const float x = operand(a[i], flags);
    const float y = operand(b[i], flags);
    const float value = arithmetic(x + y, flags);
    output[i] = flags == 0 ? value : 0.0f;
    report(flags, error);
}

__global__ void scaled_add(const float* input, float scale, int count,
                          float* accumulator, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    int flags = 0;
    const float x = operand(input[i], flags);
    const float original = operand(accumulator[i], flags);
    const float contribution = arithmetic(scale * x, flags);
    const float value = arithmetic(original + contribution, flags);
    accumulator[i] = flags == 0 ? value : 0.0f;
    report(flags, error);
}

__global__ void shared_sigmoid_add(const float* input, const float* raw_gate, int count,
                                 float* accumulator, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    int flags = 0;
    const float x = operand(input[i], flags);
    const float original = operand(accumulator[i], flags);
    const float gate = operand(raw_gate[0], flags);
    // Expected exp overflow for finite negative gates yields valid +0 sigmoid.
    const float scale = arithmetic(1.0f / (1.0f + expf(-gate)), flags);
    const float contribution = arithmetic(scale * x, flags);
    const float value = arithmetic(original + contribution, flags);
    accumulator[i] = flags == 0 ? value : 0.0f;
    report(flags, error);
}

// The fused production router deliberately uses logical width32 on wave64.
__device__ __forceinline__ float router_max(float x) {
#pragma unroll
    for (int mask = 1; mask < router_width; mask *= 2)
        x = fmaxf(x, __shfl_xor(x, mask, router_width));
    return x;
}

__global__ __launch_bounds__(router_width)
void router_softmax(const float* logits, float* probabilities, int* error) {
    const int lane = threadIdx.x, base = blockIdx.x * experts;
    int flags = 0;
    float x[experts / router_width], weights[experts / router_width];
    float maximum = -INFINITY;
#pragma unroll
    for (int k = 0; k < experts / router_width; ++k) {
        x[k] = operand(logits[base + lane + k * router_width], flags);
        maximum = fmaxf(maximum, x[k]);
    }
    maximum = router_max(maximum);
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < experts / router_width; ++k) {
        const float shifted = arithmetic(x[k] - maximum, flags);
        weights[k] = arithmetic(expf(shifted), flags);
        total = arithmetic(total + weights[k], flags);
    }
    total = arithmetic(sum32(total), flags);
    if (!(total > 0.0f)) flags |= session_ops_error_arithmetic;
    const float inverse = total > 0.0f ? arithmetic(1.0f / total, flags) : 0.0f;
#pragma unroll
    for (int k = 0; k < experts / router_width; ++k) weights[k] = arithmetic(weights[k] * inverse, flags);
    // All lanes reach both reductions, including NaN/Inf input lanes. Reduce
    // the final flags BEFORE any store so one failure zeros every row element.
#pragma unroll
    for (int mask = 1; mask < router_width; mask *= 2)
        flags |= __shfl_xor(flags, mask, router_width);
#pragma unroll
    for (int k = 0; k < experts / router_width; ++k)
        probabilities[base + lane + k * router_width] = flags == 0 ? weights[k] : 0.0f;
    if (lane == 0) report(flags, error);
}

} // namespace

hipError_t launch_qsa_split_q_gate(const float* projected, int tokens, float* query,
        float* gate, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * session_ops_qsa_query_elements;
    if (!valid_ranges(std::array{range(projected, tokens * session_ops_qsa_projected_elements),
            range(query, count, true), range(gate, count, true), range(error, 1, true)}))
        return hipErrorInvalidValue;
    split_q_gate<<<ceil_div(count, threads), threads, 0, stream>>>(
            projected, count, query, gate, error);
    return hipGetLastError();
}

hipError_t launch_qsa_rope(const float* input, const float* inverse_frequencies, int tokens,
        int heads, int start_position, float* output, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens) || (heads != session_ops_qsa_query_heads &&
            heads != session_ops_qsa_key_heads) || start_position < 0 ||
            start_position > std::numeric_limits<int>::max() - (tokens - 1))
        return hipErrorInvalidValue;
    const int count = tokens * heads * head_dim;
    const bool inplace = input == output;
    if (!valid_ranges(std::array{range(input, count, inplace), range(inverse_frequencies, rotary_half),
            range(output, inplace ? 0 : count, true), range(error, 1, true)}))
        return hipErrorInvalidValue;
    qsa_rope<<<ceil_div(count / 2, threads), threads, 0, stream>>>(
            input, inverse_frequencies, heads, start_position, count / 2, output, error);
    return hipGetLastError();
}

hipError_t launch_rope_frequencies(float base, float* output, int* error,
        hipStream_t stream) noexcept {
    if (!std::isfinite(base) || base <= 0.0f ||
        !valid_ranges(std::array{range(output, rotary_half, true), range(error, 1, true)}))
        return hipErrorInvalidValue;
    const float theta_scale = std::pow(base, -2.0f / 64.0f);
    if (!std::isfinite(theta_scale) || !(theta_scale > 0.0f)) return hipErrorInvalidValue;
    rope_frequencies<<<1, rotary_half, 0, stream>>>(theta_scale, output, error);
    return hipGetLastError();
}

hipError_t launch_sigmoid_gate(const float* input, const float* gate, int count,
        float* output, int* error, hipStream_t stream) noexcept {
    if (count <= 0 || !valid_ranges(std::array{range(input, count), range(gate, count),
            range(output, count, true), range(error, 1, true)})) return hipErrorInvalidValue;
    sigmoid_gate<<<ceil_div(count, threads), threads, 0, stream>>>(input, gate, count, output, error);
    return hipGetLastError();
}

hipError_t launch_silu_up(const float* gate, const float* up, int count, float* output,
        int* error, hipStream_t stream) noexcept {
    if (count <= 0 || !valid_ranges(std::array{range(gate, count), range(up, count),
            range(output, count, true), range(error, 1, true)})) return hipErrorInvalidValue;
    silu_up<<<ceil_div(count, threads), threads, 0, stream>>>(gate, up, count, output, error);
    return hipGetLastError();
}

hipError_t launch_add(const float* a, const float* b, int count, float* output,
        int* error, hipStream_t stream) noexcept {
    if (count <= 0 || !valid_ranges(std::array{range(a, count), range(b, count),
            range(output, count, true), range(error, 1, true)})) return hipErrorInvalidValue;
    add<<<ceil_div(count, threads), threads, 0, stream>>>(a, b, count, output, error);
    return hipGetLastError();
}

hipError_t launch_scaled_add(const float* input, float scale, int count, float* accumulator,
        int* error, hipStream_t stream) noexcept {
    if (count <= 0 || !std::isfinite(scale) || !valid_ranges(std::array{range(input, count),
            range(accumulator, count, true), range(error, 1, true)})) return hipErrorInvalidValue;
    scaled_add<<<ceil_div(count, threads), threads, 0, stream>>>(input, scale, count, accumulator, error);
    return hipGetLastError();
}

hipError_t launch_router_softmax(const float* logits, int tokens, float* probabilities,
        int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * experts;
    if (!valid_ranges(std::array{range(logits, count), range(probabilities, count, true),
            range(error, 1, true)})) return hipErrorInvalidValue;
    router_softmax<<<tokens, router_width, 0, stream>>>(logits, probabilities, error);
    return hipGetLastError();
}

hipError_t launch_shared_sigmoid_add(const float* input, const float* raw_gate, int count,
        float* accumulator, int* error, hipStream_t stream) noexcept {
    if (count <= 0 || !valid_ranges(std::array{range(input, count), range(raw_gate, 1),
            range(accumulator, count, true), range(error, 1, true)})) return hipErrorInvalidValue;
    shared_sigmoid_add<<<ceil_div(count, threads), threads, 0, stream>>>(
            input, raw_gate, count, accumulator, error);
    return hipGetLastError();
}

} // namespace qwen
