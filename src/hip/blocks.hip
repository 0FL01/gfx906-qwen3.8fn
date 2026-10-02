#include "blocks.cuh"
#include "dpp.cuh"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "HC/PLE require finite checks; compile without fast/finite-only math"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "blocks.hip is specialized for gfx906 wave64"
#endif
#pragma clang fp contract(off)

// Donor-first disposition for FP32 N=1..128, H2560/C4/R320, index H128,
// QSA H256 (checked against pinned source, not selected by similar model name):
// ADAPT mx dcd685463d597d31f5ca759d32c94592a2740fa4, ggml/src/ggml-cuda/
// norm.cu::rms_norm_f32 and common.cuh::block_reduce / gfx906 DPP reductions.
// RMS: one group/block, H2560 uses 1024 threads/16 wave64 (stride 1024),
// H128/256 uses 256 threads/four wave64 (stride 256). ADAPT norm.cu:128-152,
// 304-322 and common.cuh:765-782: explicit square-accumulation FMA, ascending
// DPP sum64, then padded sum64 of LDS wave totals, sum/Group and rsqrtf(mean+eps).
// Confirmed in the production libggml-hip.so gfx906 code object: RMS square
// accumulation is v_fmac_f32 (1024-thread plain/fused: 0x9400/0x9d10);
// plain scale uses v_rsq_f32 at 0x95e8. Use existing dpp.cuh's PORT with its
// mandatory VALU/DPP waits. Folded gamma is direct; x*scale then *gamma stays
// separately rounded. PLE dot retains its existing four-wave/pairwise reduction.
// ADAPT furnace 905021dbad71c5056ef51f9fd45d545403fc989c, same directory,
// dsv4-hc.cu::dsv4_hc_pre_f32<true> and dsv4_hc_post_f32<false,false>:
// coalesced feature layout, ascending four-branch sigmoid mix/residual add.
// DO NOT PORT dsv4_hc_comb_f32: DeepSeek softmax/Sinkhorn is different math.
// repack-gcn.cu::hc_up_pre_f32 is a Qwen C4/R320 fusion candidate AFTER
// projection parity, not used here: it includes Q8 quantization and matmul.
// Copyright (c) 2023-2026 The ggml authors; MIT in third_party/mx-LICENSE.
// Checked reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8,
// kernels/rmsnorm.cpp: direct GGUF gamma confirmed; its full-LDS reduction is
// not imported (mx DPP primitive already available). No runtime/graph import.
// WRITE OURS PLE gate/causal history staging/publication: frozen src/hc.cpp and
// src/ple.cpp, HF a005fc82babfe8871d87746decad2dbee100a125 equations.

namespace qwen {
namespace {

constexpr int hidden = blocks_device_hidden;
constexpr int branches = blocks_device_branches;
constexpr int rank = blocks_device_rank;
constexpr int wide = blocks_device_wide;
constexpr int history_length = ple_device_history_length;
constexpr int history_elements = static_cast<int>(ple_device_history_elements);
constexpr int threads = 256;
static_assert(sizeof(float) == 4 && sizeof(int) == 4);

struct Range {
    std::uintptr_t begin, end;
    bool writable, valid;
};

template<class T>
Range range(const T* pointer, std::size_t count, bool writable = false) noexcept {
    if (count == 0) return {0, 0, writable, true};
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (pointer == nullptr || begin % alignof(T) != 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T))
        return {0, 0, writable, false};
    const auto bytes = count * sizeof(T);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        return {0, 0, writable, false};
    return {begin, begin + bytes, writable, true};
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
    return tokens >= 1 && tokens <= blocks_device_max_tokens;
}

bool valid_norm_shape(int width, int group) noexcept {
    return (group == 2560 && (width == 2560 || width == 10240)) ||
           (group == 128 && (width == 128 || width == 512)) ||
           (group == 256 && (width == 256 || width == 512 || width == 6144));
}

// Only bounded, positive validated counts reach ceil_div. Avoid n+d-1 overflow.
unsigned ceil_div(int n, int d) noexcept {
    return static_cast<unsigned>(n / d + (n % d != 0));
}

__device__ __forceinline__ float operand(float x, int& flags) {
    if (!isfinite(x)) { flags |= blocks_device_error_invalid; return 0.0f; }
    return x;
}

__device__ __forceinline__ float arithmetic(float x, int& flags) {
    if (!isfinite(x)) { flags |= blocks_device_error_arithmetic; return 0.0f; }
    return x;
}

__device__ __forceinline__ void report(int flags, int* error) {
    if (flags != 0) atomicOr(error, flags);
}

__device__ __forceinline__ float sigmoid(float x, int& flags) {
    // ADAPT pinned mx unary.cu: finite negative exponential overflow gives +0,
    // not a numeric error. Callers validate operands; check the final result.
    return arithmetic(1.0f / (1.0f + expf(-x)), flags);
}

__device__ __forceinline__ float silu(float x, int& flags) {
    // ADAPT unary.cuh division: preserve -0 for finite negative saturation.
    return arithmetic(x / (1.0f + expf(-x)), flags);
}

template<int Group>
__global__ __launch_bounds__(Group == hidden ? 1024 : 256)
void group_rms_norm(const float* input, const float* gamma, int width, int rows,
                    float epsilon, bool shared_gamma, float* output, int* error) {
    constexpr int stride = Group == hidden ? 1024 : 256;
    constexpr int components = (Group + stride - 1) / stride;
    __shared__ float wave_sums[64];
    __shared__ float group_scale;
    const int row = static_cast<int>(blockIdx.x), local = threadIdx.x;
    const int lane = local % 64, wave = local / 64;
    const int base = row * Group;
    const int gamma_base = shared_gamma ? 0 : base % width;
    int flags = 0;
    float x[components]{}, g[components]{};
    float squares = 0.0f;
    // Initialize every LDS lane, including padding beyond the 16/four totals.
    // All block threads participate in every barrier, even without a feature.
    if (local < 64) wave_sums[local] = 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < components; ++k) {
        const int d = local + k * stride;
        if (row < rows && d < Group) {
            x[k] = operand(input[base + d], flags);
            g[k] = operand(gamma[gamma_base + d], flags);
            squares = arithmetic(fmaf(x[k], x[k], squares), flags);
        }
    }
    squares = arithmetic(sum64(squares), flags);
    if (lane == 0) wave_sums[wave] = squares;
    __syncthreads();
    if (wave == 0) {
        const float total = arithmetic(sum64(wave_sums[lane]), flags);
        if (lane == 0) {
            // Donor mean is division by ncols, not multiplication by its inverse.
            const float mean = arithmetic(total / static_cast<float>(Group), flags);
            const float norm = arithmetic(mean + epsilon, flags);
            group_scale = arithmetic(rsqrtf(norm), flags);
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < components; ++k) {
        const int d = local + k * stride;
        if (row < rows && d < Group) {
            const float scaled = arithmetic(x[k] * group_scale, flags);
            output[base + d] = arithmetic(scaled * g[k], flags);
        }
    }
    report(flags, error);
}

__global__ void hc_lowrank_activation(const float* projection, int count,
                                     float* output, int* error) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int flags = 0;
    const float x = arithmetic(operand(projection[i], flags) / 4.0f, flags);
    output[i] = silu(x, flags);
    report(flags, error);
}

// ADAPT furnace dsv4_hc_pre_f32<true>: a thread owns one output feature,
// consecutive lanes access consecutive features in each of the four branches.
__global__ void hc_mix(const float* normalized, const float* projection, int count,
                       float* output, int* error) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int t = i / hidden, d = i % hidden;
    int flags = 0;
    float sum = 0.0f;
#pragma unroll
    for (int b = 0; b < branches; ++b) {
        const int j = t * wide + b * hidden + d;
        const float x = operand(normalized[j], flags);
        const float gate = sigmoid(operand(projection[j], flags), flags);
        const float term = arithmetic(gate * x, flags);
        sum = arithmetic(sum + term, flags);
    }
    output[i] = arithmetic(sum / 4.0f, flags);
    report(flags, error);
}

__global__ void hc_injection(const float* projection, int count, float* output, int* error) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int flags = 0;
    const float x = arithmetic(operand(projection[i], flags) / 4.0f, flags);
    output[i] = arithmetic(2.0f * sigmoid(x, flags), flags);
    report(flags, error);
}

// ADAPT furnace dsv4_hc_post_f32<false,false>; preserve the original residual,
// with activated injection supplied separately and no cross-branch remix.
__global__ void hc_combine(const float* residual, const float* block, const float* injection,
                           int count, float* output, int* error) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int t = i / wide, b = (i % wide) / hidden, d = i % hidden;
    int flags = 0;
    const float original = operand(residual[i], flags);
    const float value = operand(block[t * hidden + d], flags);
    float gate = operand(injection[t * branches + b], flags);
    if (gate < 0.0f || gate > 2.0f) { flags |= blocks_device_error_invalid; gate = 0.0f; }
    const float contribution = arithmetic(gate * value, flags);
    output[i] = arithmetic(original + contribution, flags);
    report(flags, error);
}

// ADAPT pinned mx SUM_ROWS(MUL(key,query)): eight independent accumulators,
// their ascending fold, logical sum32 and a zero-padded second sum32. Products
// remain separately rounded; this is not an FMA dot. The 60-CU target selects
// block512 for short decode rows, block128 for >=120 rows. Keep the source SCALE
// reciprocal multiply after the completed sum, rather than divide by sqrt(H).
template<int Block>
__global__ __launch_bounds__(Block)
void ple_gate(const float* key, const float* query, const float* value,
              float* output, int* error) {
    __shared__ float group_sums[32];
    __shared__ float group_gate;
    const int row = blockIdx.x, local = threadIdx.x;
    const int lane = local % 32, group = local / 32;
    const int base = row * hidden, value_base = (row / branches) * hidden;
    int flags = 0;
    float partials[8]{};
    if (local < 32) group_sums[local] = 0.0f;
    __syncthreads();
    for (int offset = local; offset < hidden; offset += 8 * Block) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = offset + j * Block;
            if (d < hidden) {
                const float a = operand(key[base + d], flags);
                const float b = operand(query[base + d], flags);
                partials[j] = arithmetic(partials[j] + arithmetic(a * b, flags), flags);
            }
        }
    }
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) dot = arithmetic(dot + partials[j], flags);
    dot = arithmetic(sum32(dot), flags);
    if (lane == 0) group_sums[group] = dot;
    __syncthreads();
    if (group == 0) {
        const float total = arithmetic(sum32(group_sums[lane]), flags);
        if (lane == 0) {
            const float s = arithmetic(fmaf(total, 0.019764235243201256f, 0.0f), flags);
            const float sign = s > 0.0f ? 1.0f : s < 0.0f ? -1.0f : 0.0f;
            const float root = arithmetic(sqrtf(fmaxf(fabsf(s), 1.0e-6f)), flags);
            group_gate = sigmoid(arithmetic(sign * root, flags), flags);
        }
    }
    __syncthreads();
    for (int d = local; d < hidden; d += Block) {
        const float v = operand(value[value_base + d], flags);
        output[base + d] = arithmetic(group_gate * v, flags);
    }
    report(flags, error);
}

template<bool Prefixes>
__global__ void ple_causal_conv(const float* gated, const float* norm_gated,
        const float* weights, const float* active_history, int tokens,
        const std::uint8_t* keep_mask, PleConvDeviceWorkspace workspace, int* error) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= wide) return;
    int flags = 0;
    float history[history_length], taps[ple_device_conv_kernel];
#pragma unroll
    for (int a = 0; a < history_length; ++a) {
        history[a] = operand(active_history[f * history_length + a], flags);
        if constexpr (Prefixes) workspace.prefixes[f * history_length + a] = history[a];
    }
#pragma unroll
    for (int k = 0; k < ple_device_conv_kernel; ++k)
        taps[k] = operand(weights[f * ple_device_conv_kernel + k], flags);
    for (int t = 0; t < tokens; ++t) {
        const int index = t * wide + f;
        float g = operand(gated[index], flags);
        float current = operand(norm_gated[index], flags);
        const unsigned keep = keep_mask == nullptr ? 1U : keep_mask[t];
        if (keep > 1U) flags |= blocks_device_error_invalid;
        if (keep != 1U) { g = 0.0f; current = 0.0f; }
        float conv = 0.0f;
#pragma unroll
        for (int k = 0; k < ple_device_conv_kernel; ++k) {
            // k=0,1,2: oldest ages 0,3,6 (lags 9,6,3); k=3: current.
            const float x = k == ple_device_conv_kernel - 1 ? current :
                            history[k * ple_device_conv_dilation];
            const float term = arithmetic(x * taps[k], flags);
            conv = arithmetic(conv + term, flags);
        }
        const float activated = silu(conv, flags);
        workspace.output[index] = arithmetic(g + activated, flags);
#pragma unroll
        for (int a = 0; a < history_length - 1; ++a) history[a] = history[a + 1];
        history[history_length - 1] = current;
        if constexpr (Prefixes) {
            const int base = (t + 1) * history_elements + f * history_length;
#pragma unroll
            for (int a = 0; a < history_length; ++a)
                workspace.prefixes[base + a] = history[a];
        }
    }
#pragma unroll
    for (int a = 0; a < history_length; ++a)
        workspace.history[f * history_length + a] = history[a];
    report(flags, error);
}

__global__ void validate_history(const float* staged, int* error) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < history_elements && !isfinite(staged[i]))
        atomicOr(error, blocks_device_error_invalid);
}

__global__ void publish_history(const float* staged, float* active, const int* error) {
    // A separate, stream-ordered kernel is the global completion boundary for
    // all producer/validation flags. No block-local flag check can replace it.
    if (*error != 0) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < history_elements) active[i] = staged[i];
}

} // namespace

hipError_t launch_group_rms_norm(const float* input, const float* gamma, int tokens,
        int width, int group_size, float epsilon, bool shared_gamma,
        float* output, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens) || !valid_norm_shape(width, group_size) ||
        !std::isfinite(epsilon) || epsilon <= 0.0f) return hipErrorInvalidValue;
    const int count = tokens * width;
    if (!valid_ranges(std::array{range(input, count), range(gamma, shared_gamma ? group_size : width),
                                  range(output, count, true), range(error, 1, true)}))
        return hipErrorInvalidValue;
    const int rows = count / group_size;
    if (group_size == hidden) {
        group_rms_norm<hidden><<<rows, 1024, 0, stream>>>(input, gamma, width, rows,
                epsilon, shared_gamma, output, error);
    } else if (group_size == 128) {
        group_rms_norm<128><<<rows, threads, 0, stream>>>(input, gamma,
                width, rows, epsilon, shared_gamma, output, error);
    } else {
        group_rms_norm<256><<<rows, threads, 0, stream>>>(input, gamma,
                width, rows, epsilon, shared_gamma, output, error);
    }
    return hipGetLastError();
}

hipError_t launch_hc_lowrank_activation(const float* down_projection, int tokens,
        float* activated, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * rank;
    if (!valid_ranges(std::array{range(down_projection, count), range(activated, count, true),
                                  range(error, 1, true)})) return hipErrorInvalidValue;
    hc_lowrank_activation<<<ceil_div(count, threads), threads, 0, stream>>>(
            down_projection, count, activated, error);
    return hipGetLastError();
}

hipError_t launch_hc_mix(const float* normalized, const float* up_projection, int tokens,
        float* mixed, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * hidden;
    if (!valid_ranges(std::array{range(normalized, tokens * wide), range(up_projection, tokens * wide),
                                  range(mixed, count, true), range(error, 1, true)}))
        return hipErrorInvalidValue;
    hc_mix<<<ceil_div(count, threads), threads, 0, stream>>>(normalized, up_projection,
            count, mixed, error);
    return hipGetLastError();
}

hipError_t launch_hc_injection(const float* inject_projection, int tokens,
        float* injection, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * branches;
    if (!valid_ranges(std::array{range(inject_projection, count), range(injection, count, true),
                                  range(error, 1, true)})) return hipErrorInvalidValue;
    hc_injection<<<ceil_div(count, threads), threads, 0, stream>>>(inject_projection,
            count, injection, error);
    return hipGetLastError();
}

hipError_t launch_hc_combine(const float* original_wide, const float* block, const float* injection,
        int tokens, float* output, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * wide;
    if (!valid_ranges(std::array{range(original_wide, count), range(block, tokens * hidden),
                                  range(injection, tokens * branches), range(output, count, true),
                                  range(error, 1, true)})) return hipErrorInvalidValue;
    hc_combine<<<ceil_div(count, threads), threads, 0, stream>>>(original_wide, block,
            injection, count, output, error);
    return hipGetLastError();
}

hipError_t launch_ple_gate(const float* norm_key, const float* norm_query, const float* shared_value,
        int tokens, float* output, int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * wide;
    if (!valid_ranges(std::array{range(norm_key, count), range(norm_query, count),
                                  range(shared_value, tokens * hidden), range(output, count, true),
                                  range(error, 1, true)})) return hipErrorInvalidValue;
    const int rows = tokens * branches;
    if (rows < 120)
        ple_gate<512><<<rows, 512, 0, stream>>>(norm_key, norm_query, shared_value, output, error);
    else
        ple_gate<128><<<rows, 128, 0, stream>>>(norm_key, norm_query, shared_value, output, error);
    return hipGetLastError();
}

hipError_t launch_ple_causal_conv(const float* gated, const float* norm_gated,
        const float* weights, const float* active_history, int tokens,
        const std::uint8_t* keep_mask, PleConvDeviceWorkspace workspace,
        int* error, hipStream_t stream) noexcept {
    if (!valid_tokens(tokens)) return hipErrorInvalidValue;
    const int count = tokens * wide;
    const auto prefix_count = workspace.prefixes == nullptr ? std::size_t{0} :
                              static_cast<std::size_t>(tokens + 1) * ple_device_history_elements;
    if (!valid_ranges(std::array{range(gated, count), range(norm_gated, count),
            range(weights, ple_device_conv_weight_elements), range(active_history, ple_device_history_elements),
            range(keep_mask, keep_mask == nullptr ? 0 : tokens), range(workspace.output, count, true),
            range(workspace.history, ple_device_history_elements, true),
            range(workspace.prefixes, prefix_count, true), range(error, 1, true)}))
        return hipErrorInvalidValue;
    if (workspace.prefixes != nullptr) {
        ple_causal_conv<true><<<wide / threads, threads, 0, stream>>>(gated, norm_gated,
                weights, active_history, tokens, keep_mask, workspace, error);
    } else {
        ple_causal_conv<false><<<wide / threads, threads, 0, stream>>>(gated, norm_gated,
                weights, active_history, tokens, keep_mask, workspace, error);
    }
    return hipGetLastError();
}

hipError_t launch_publish_ple_history(const float* staged, float* active_history,
        int* error, hipStream_t stream) noexcept {
    if (!valid_ranges(std::array{range(staged, ple_device_history_elements),
                                  range(active_history, ple_device_history_elements, true),
                                  range(error, 1, true)})) return hipErrorInvalidValue;
    const unsigned blocks = ceil_div(history_elements, threads);
    validate_history<<<blocks, threads, 0, stream>>>(staged, error);
    const hipError_t result = hipGetLastError();
    if (result != hipSuccess) return result;
    publish_history<<<blocks, threads, 0, stream>>>(staged, active_history, error);
    return hipGetLastError();
}

} // namespace qwen
