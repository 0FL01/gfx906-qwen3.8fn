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

// Donor-first: FP32 microtiles N1..128, Q24/K2/D256/rotary64, router512,
// expert640. Extending the pointwise/independent-row grids changes no arithmetic;
// Session still slices a larger logical chunk before calling these launchers.
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
// Indexed MoE copy donor seams checked mx -> furnace -> reinstinct (RECON16):
// ADAPT mx dcd685463d597d31f5ca759d32c94592a2740fa4 getrows.cu::
// k_get_rows_float_vec/get_rows_cuda_float: indexed rows and coalesced 16B
// traffic, with 4B fallback. Furnace 905021dbad71c5056ef51f9fd45d545403fc989c
// getrows.cu has the same vector primitive and a short-row variant, not a
// bounded canonical Q8 route copy. Checked reinstinct
// 0b79e326351d90d4554a1c18df92da5d0ab692e8 kernels/moe_expert_sort.cpp::
// moe_gather_xq/moe_scatter_rows: KEEP separate unweighted copy semantics;
// its 40B BlockQ8/perm ABI is NOT imported. WRITE OURS bounded token/rank DTO,
// capacity checks and kernels for canonical 36B Q8_1 [N][80] and unweighted
// FP32 [N][2560], N1/2/3..128. Builtin constant-size byte copies replace donor
// int4 pointer punning; no dequantization, arithmetic, packer or runtime import.
// RouteGroups stays CPU-sorted; caller uploads metadata once/full layer and
// owns all streams/events/cache-slot dependencies. Rank fold remains unchanged.
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

constexpr int q8_row_bytes = moe_copy_q8_blocks * sizeof(Q8_1);
constexpr int down_row_bytes = moe_copy_width * sizeof(float);
static_assert(q8_row_bytes == 2880 && down_row_bytes == 10240);
static_assert(q8_row_bytes % 16 == 0 && down_row_bytes % 16 == 0);
static_assert(moe_copy_max_input_tokens * moe_copy_ranks * moe_copy_width < std::numeric_limits<int>::max());

template<class T>
Range copy_range(MoeCopyView<T> view, std::size_t required, bool writable = false) noexcept {
    // Validate the WHOLE view even when only a late-offset slice is accessed.
    auto r = range(view.data, view.capacity, writable);
    if (required == 0 || view.capacity < required ||
            reinterpret_cast<std::uintptr_t>(view.data) % 4 != 0) r.valid = false;
    return r;
}

bool valid_copy_slice(int input_tokens, int metadata_entries, int route_offset,
                      int count, int column_capacity) noexcept {
    // Subtraction only after bounding count; offset+count can never overflow.
    return input_tokens >= 1 && input_tokens <= moe_copy_max_input_tokens &&
        metadata_entries >= 1 && metadata_entries <= input_tokens * moe_copy_ranks &&
        count >= 1 && count <= session_ops_max_tokens && column_capacity >= count &&
        count <= metadata_entries && route_offset >= 0 && route_offset <= metadata_entries - count;
}

bool aligned16(const void* a, const void* b) noexcept {
    return (reinterpret_cast<std::uintptr_t>(a) | reinterpret_cast<std::uintptr_t>(b)) % 16 == 0;
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

// Address-only adaptation for the aggregate CPU projection handoff. Keep the
// original kernel above untouched: this exact small expression is checked
// against it (including quantized RAW-sum bytes) in the paired-middle fixture.
__global__ void silu_up_pairs(const float* paired, int count, float* output, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    constexpr unsigned width = silu_pairs_width;
    const unsigned base = (i / width) * (2 * width) + i % width;
    int flags = 0;
    const float g = operand(paired[base], flags);
    const float u = operand(paired[base + width], flags);
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

// WRITE OURS original-rank pointwise fold; the grouped execution order never
// enters this numerical reduction. Preserve ten separately rounded MUL/ADDs.
__global__ void routed_fold(const float* contributions, const float* weights,
                           int count, float* output, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    constexpr unsigned width = 2560, ranks = 10;
    const unsigned token = i / width, component = i % width;
    int flags = 0;
    float value = 0.0f;
#pragma unroll
    for (unsigned rank = 0; rank < ranks; ++rank) {
        const float x = operand(contributions[(token * ranks + rank) * width + component], flags);
        const float scale = operand(weights[token * ranks + rank], flags);
        const float contribution = arithmetic(scale * x, flags);
        value = arithmetic(value + contribution, flags);
    }
    output[i] = flags == 0 ? value : 0.0f;
    report(flags, error);
}

// Object representations accessed through unsigned char/builtin memcpy only:
// no Q8/float -> uint32_t/int4 lvalue alias, and no floating-point loads/stores.
template<int Bytes>
__device__ __forceinline__ void opaque_copy(unsigned char* destination, const unsigned char* source) {
    static_assert(Bytes == 4 || Bytes == 16);
    __builtin_memcpy(__builtin_assume_aligned(destination, Bytes),
                     __builtin_assume_aligned(source, Bytes), Bytes);
}

template<int Bytes>
__global__ void moe_gather_q8(const unsigned char* input, int input_tokens,
                             const MoeRouteIndex* routes, int route_offset, int count,
                             unsigned char* gathered, int* error) {
    const unsigned column = blockIdx.y;
    const unsigned unit = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= static_cast<unsigned>(count) || unit >= q8_row_bytes / Bytes) return;
    const auto route = routes[static_cast<unsigned>(route_offset) + column];
    const bool valid = route.token >= 0 && route.token < input_tokens &&
        route.rank >= 0 && route.rank < moe_copy_ranks;
    // Destination depends only on the already checked physical column.
    auto* destination = gathered + std::size_t(column) * q8_row_bytes + unit * Bytes;
    if (!valid) {
        if (unit == 0) report(session_ops_error_invalid, error);
        __builtin_memset(__builtin_assume_aligned(destination, Bytes), 0, Bytes);
        return;
    }
    // Form the indexed source ONLY after checking both signed indices.
    const auto* source = input + std::size_t(route.token) * q8_row_bytes + unit * Bytes;
    opaque_copy<Bytes>(destination, source);
}

template<int Bytes>
__global__ void moe_scatter(const unsigned char* down, int input_tokens,
                           const MoeRouteIndex* routes, int route_offset, int count,
                           unsigned char* contributions, int* error) {
    const unsigned column = blockIdx.y;
    const unsigned unit = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= static_cast<unsigned>(count) || unit >= down_row_bytes / Bytes) return;
    const auto route = routes[static_cast<unsigned>(route_offset) + column];
    if (route.token < 0 || route.token >= input_tokens || route.rank < 0 || route.rank >= moe_copy_ranks) {
        if (unit == 0) report(session_ops_error_invalid, error);
        return; // No down read, indexed address formation or payload write.
    }
    const auto row = std::size_t(route.token) * moe_copy_ranks + std::size_t(route.rank);
    auto* destination = contributions + row * down_row_bytes + unit * Bytes;
    const auto* source = down + std::size_t(column) * down_row_bytes + unit * Bytes;
    opaque_copy<Bytes>(destination, source);
}

template<bool RowBroadcast>
__global__ void shared_sigmoid_add(const float* input, const float* raw_gate, int count, int width,
                                 float* accumulator, int* error) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    int flags = 0;
    const float x = operand(input[i], flags);
    const float original = operand(accumulator[i], flags);
    const unsigned row = RowBroadcast ? i / static_cast<unsigned>(width) : 0;
    const float gate = operand(raw_gate[row], flags);
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

hipError_t launch_silu_up_pairs(const float* paired, int columns, float* output,
        int* error, hipStream_t stream) noexcept {
    if (columns < 1 || columns > silu_pairs_max_columns) return hipErrorInvalidValue;
    const int count = columns * silu_pairs_width;
    if (!valid_ranges(std::array{range(paired, count * 2), range(output, count, true),
            range(error, 1, true)})) return hipErrorInvalidValue;
    silu_up_pairs<<<ceil_div(count, threads), threads, 0, stream>>>(paired, count, output, error);
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

hipError_t launch_routed_fold(const float* contributions, const float* weights,
        int tokens, float* output, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * 2560;
    if (!valid_ranges(std::array{range(contributions, count * 10), range(weights, tokens * 10),
            range(output, count, true), range(error, 1, true)})) return hipErrorInvalidValue;
    routed_fold<<<ceil_div(count, threads), threads, 0, stream>>>(contributions, weights, count, output, error);
    return hipGetLastError();
}

hipError_t launch_moe_gather_q8(MoeCopyView<const Q8_1> input, int input_tokens,
        MoeCopyView<const MoeRouteIndex> routes, int metadata_entries, int route_offset,
        int count, int blocks_per_token, MoeCopyView<Q8_1> gathered, int column_capacity,
        int* error, hipStream_t stream) noexcept {
    if (!valid_copy_slice(input_tokens, metadata_entries, route_offset, count, column_capacity) ||
            blocks_per_token != moe_copy_q8_blocks ||
            column_capacity > std::numeric_limits<int>::max() / moe_copy_q8_blocks)
        return hipErrorInvalidValue;
    if (!valid_ranges(std::array{
            copy_range(input, std::size_t(input_tokens) * moe_copy_q8_blocks),
            copy_range(routes, metadata_entries),
            copy_range(gathered, std::size_t(column_capacity) * moe_copy_q8_blocks, true),
            range(error, 1, true)})) return hipErrorInvalidValue;
    const auto* source = reinterpret_cast<const unsigned char*>(input.data);
    auto* destination = reinterpret_cast<unsigned char*>(gathered.data);
    if (aligned16(source, destination)) {
        moe_gather_q8<16><<<dim3(ceil_div(q8_row_bytes / 16, threads), count), threads, 0, stream>>>(
                source, input_tokens, routes.data, route_offset, count, destination, error);
    } else {
        moe_gather_q8<4><<<dim3(ceil_div(q8_row_bytes / 4, threads), count), threads, 0, stream>>>(
                source, input_tokens, routes.data, route_offset, count, destination, error);
    }
    return hipGetLastError();
}

hipError_t launch_moe_scatter(MoeCopyView<const float> down, int input_tokens,
        MoeCopyView<const MoeRouteIndex> routes, int metadata_entries, int route_offset,
        int count, int width, int ranks, MoeCopyView<float> contributions,
        int column_capacity, int* error, hipStream_t stream) noexcept {
    if (!valid_copy_slice(input_tokens, metadata_entries, route_offset, count, column_capacity) ||
            width != moe_copy_width || ranks != moe_copy_ranks ||
            column_capacity > std::numeric_limits<int>::max() / moe_copy_width)
        return hipErrorInvalidValue;
    if (!valid_ranges(std::array{
            copy_range(down, std::size_t(column_capacity) * moe_copy_width),
            copy_range(routes, metadata_entries),
            copy_range(contributions, std::size_t(input_tokens) * moe_copy_ranks * moe_copy_width, true),
            range(error, 1, true)})) return hipErrorInvalidValue;
    const auto* source = reinterpret_cast<const unsigned char*>(down.data);
    auto* destination = reinterpret_cast<unsigned char*>(contributions.data);
    if (aligned16(source, destination)) {
        moe_scatter<16><<<dim3(ceil_div(down_row_bytes / 16, threads), count), threads, 0, stream>>>(
                source, input_tokens, routes.data, route_offset, count, destination, error);
    } else {
        moe_scatter<4><<<dim3(ceil_div(down_row_bytes / 4, threads), count), threads, 0, stream>>>(
                source, input_tokens, routes.data, route_offset, count, destination, error);
    }
    return hipGetLastError();
}

hipError_t launch_shared_sigmoid_add(const float* input, const float* raw_gate, int count,
        float* accumulator, int* error, hipStream_t stream) noexcept {
    if (count <= 0 || !valid_ranges(std::array{range(input, count), range(raw_gate, 1),
            range(accumulator, count, true), range(error, 1, true)})) return hipErrorInvalidValue;
    shared_sigmoid_add<false><<<ceil_div(count, threads), threads, 0, stream>>>(
            input, raw_gate, count, count, accumulator, error);
    return hipGetLastError();
}

hipError_t launch_shared_sigmoid_add_rows(const float* input, const float* raw_gate,
        int tokens, int width, float* accumulator, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens) || width <= 0 || width > std::numeric_limits<int>::max() / tokens)
        return hipErrorInvalidValue;
    const int count = tokens * width;
    if (!valid_ranges(std::array{range(input, count), range(raw_gate, tokens),
            range(accumulator, count, true), range(error, 1, true)})) return hipErrorInvalidValue;
    shared_sigmoid_add<true><<<ceil_div(count, threads), threads, 0, stream>>>(
            input, raw_gate, count, width, accumulator, error);
    return hipGetLastError();
}

} // namespace qwen
