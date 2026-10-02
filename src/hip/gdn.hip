#include "gdn.cuh"
#include "dpp.cuh"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(__FAST_MATH__)
#error "GDN requires FP32 finite checks and must not be compiled with fast-math"
#endif
// Preserve the CPU oracle's separately rounded multiplication/addition. Do not
// contract a product into its following addition or use approximate exp/rsqrt.
#pragma clang fp contract(off)

// ADAPT: QK16/V48/D128, N=1..128, one session, FP32 [head][v][k] state.
// mx dcd685463d597d31f5ca759d32c94592a2740fa4, ggml/src/ggml-cuda/
// gated_delta_net.cu::gated_delta_net_cuda supplies the wave/column register ABI.
// furnace 905021dbad71c5056ef51f9fd45d545403fc989c, same file,
// gated_delta_net_cpw<2> and gated_delta_net_lds_wave64<16> supply decode CPW2
// and resident-state slab/prefetch/XOR swizzle. Copyright (c) 2023-2026 The ggml
// authors; mx/furnace MIT license text is in third_party/mx-LICENSE.
// The resident design was adapted by furnace from sixvolts/reinstinct
// 0b79e326351d90d4554a1c18df92da5d0ab692e8 kernels/gdn_recurrent_batched_v2.cpp,
// Apache-2.0 (third_party/Apache-2.0). Its opposite HBM orientation is NOT used.
// Modifications: fixed GGUF h%16 mapping, CPU decay-before-dot math, no fast
// intrinsics, chronological prefixes, checked causal conv/L2/RMS/sigmoid gate,
// finite/sanitized staging and chunk-atomic publication. No donor runtime code.

namespace qwen {
namespace {

constexpr int key_heads = gdn_device_key_heads;
constexpr int value_heads = gdn_device_value_heads;
constexpr int head_dim = gdn_device_head_dim;
constexpr int keys = key_heads * head_dim;
constexpr int values = gdn_device_value_elements;
constexpr int features = gdn_device_qkv_elements;
constexpr int recurrent_elements = gdn_device_recurrent_elements;
constexpr int history_elements = gdn_device_history_elements;
constexpr float query_scale = 0.08838834764831844055f; // 1/sqrt(128)

void check_launch(hipError_t result) {
    if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}

// Sanitize before storing or doing more arithmetic. Invalid scratch is never
// published, but cannot carry NaNs into later recurrence steps either. Flags
// accumulate in registers; only failing threads perform an atomic at kernel end.
__device__ __forceinline__ float input_finite(float x, int& flags) {
    if (!isfinite(x)) { flags |= 1; return 0.0f; }
    return x;
}

__device__ __forceinline__ float arithmetic_finite(float x, int& flags) {
    if (!isfinite(x)) { flags |= 2; return 0.0f; }
    return x;
}

__device__ __forceinline__ void report_error(int flags, int* error) {
    if (flags) atomicOr(error, flags);
}

__device__ __forceinline__ float sigmoid(float x) {
    // ADAPT production unary.cu: direct division, including valid +0 when
    // exp(-x) overflows for a finite negative operand. Final values are checked
    // by the caller; expected exponential overflow is not a numeric failure.
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus(float x) {
    return x > 20.0f ? x : logf(1.0f + expf(x));
}

struct Controls { float decay; float beta; float log_decay; };

__device__ __forceinline__ Controls controls(float alpha, float beta, float bias,
                                            float a, int& flags) {
    alpha = input_finite(alpha, flags);
    beta = input_finite(beta, flags);
    const float dt = arithmetic_finite(alpha + bias, flags);
    const float sp = arithmetic_finite(softplus(dt), flags);
    const float log_decay = arithmetic_finite(a * sp, flags);
    return {arithmetic_finite(expf(log_decay), flags), sigmoid(beta), log_decay};
}

__device__ __forceinline__ float parameter_a(float a, int& flags) {
    a = input_finite(a, flags);
    if (a > 0.0f) { flags |= 1; return 0.0f; }
    return a;
}

// Trace-only recomputation through the SAME scalar helper. This is not a
// capture of recurrence registers and never changes active state/output.
__global__ void control_witness(GdnDeviceParameters p, GdnDeviceInput in,
                               float* output, int* error) {
    const int h = threadIdx.x;
    if (h >= value_heads) return;
    int flags = 0;
    const float bias = input_finite(p.dt_bias[h], flags);
    const float a = parameter_a(p.ssm_a[h], flags);
    const Controls cb = controls(in.alpha[h], in.beta[h], bias, a, flags);
    output[h] = cb.log_decay;
    output[value_heads + h] = cb.beta;
    output[2 * value_heads + h] = cb.decay;
    report_error(flags, error);
}

// One feature/thread, with the causal token loop in registers. All history and
// prefixes contain RAW projections, including the pre-state in slot zero.
template<bool Prefixes>
__global__ void convolve(GdnDeviceParameters p, GdnDeviceInput in, int tokens,
                         const float* active_history, GdnDeviceWorkspace w) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= features) return;
    int flags = 0;
    float h0 = input_finite(active_history[f * 3], flags);
    float h1 = input_finite(active_history[f * 3 + 1], flags);
    float h2 = input_finite(active_history[f * 3 + 2], flags);
    const float w0 = input_finite(p.conv_weights[f * 4], flags);
    const float w1 = input_finite(p.conv_weights[f * 4 + 1], flags);
    const float w2 = input_finite(p.conv_weights[f * 4 + 2], flags);
    const float w3 = input_finite(p.conv_weights[f * 4 + 3], flags);
    if constexpr (Prefixes) {
        w.history_prefixes[f * 3] = h0;
        w.history_prefixes[f * 3 + 1] = h1;
        w.history_prefixes[f * 3 + 2] = h2;
    }
    for (int t = 0; t < tokens; ++t) {
        const float raw = input_finite(in.qkv[t * features + f], flags);
        // ADAPT mx ssm-conv.cu's compiled chronological FMA chain. This is
        // GDN's raw-projection convolution, not PLE's separate MUL/ADD graph.
        // Stored state/history/weights remain FP32; check each fused result.
        float x = arithmetic_finite(fmaf(h0, w0, 0.0f), flags);
        x = arithmetic_finite(fmaf(h1, w1, x), flags);
        x = arithmetic_finite(fmaf(h2, w2, x), flags);
        x = arithmetic_finite(fmaf(raw, w3, x), flags);
        w.convolved[t * features + f] = arithmetic_finite(x / (1.0f + expf(-x)), flags);
        h0 = h1; h1 = h2; h2 = raw;
        if constexpr (Prefixes) {
            const int base = (t + 1) * history_elements + f * 3;
            w.history_prefixes[base] = h0;
            w.history_prefixes[base + 1] = h1;
            w.history_prefixes[base + 2] = h2;
        }
    }
    w.history[f * 3] = h0;
    w.history[f * 3 + 1] = h1;
    w.history[f * 3 + 2] = h2;
    report_error(flags, w.error);
}

// One wave64 owns a Q or K vector. Keep the independent additive-L2 oracle's
// ascending separately-rounded FP32 sum and division for deterministic rounding
// at downstream Q8 thresholds; epsilon remains the HF 1e-6, never the fork floor.
// This D128-only reduction is deliberately serial in lane 0, with coalesced
// readback/stores. Performance is measured separately from correctness.
__global__ __launch_bounds__(256) void normalize_qk(float* convolved, int tokens, int* error) {
    const int row = blockIdx.x * 4 + threadIdx.x / 64;
    if (row >= tokens * (2 * key_heads)) return; // wave-uniform
    const int lane = threadIdx.x % 64;
    const int t = row / (2 * key_heads), h = row % (2 * key_heads);
    const int base = t * features + h * head_dim + lane;
    int flags = 0;
    const float x0 = convolved[base], x1 = convolved[base + 64];
    float denominator = 0.0f;
    if (lane == 0) {
        float sum = 0.0f;
        for (int j = 0; j < head_dim; ++j) {
            const float x = convolved[t * features + h * head_dim + j];
            sum = arithmetic_finite(sum + arithmetic_finite(x * x, flags), flags);
        }
        denominator = arithmetic_finite(sqrtf(arithmetic_finite(sum + 1.0e-6f, flags)), flags);
    }
    denominator = __shfl(denominator, 0, 64);
    const float y0 = arithmetic_finite(x0 / denominator, flags);
    const float y1 = arithmetic_finite(x1 / denominator, flags);
    convolved[base] = y0;
    convolved[base + 64] = y1;
    report_error(flags, error);
}

// ADAPT furnace gated_delta_net_cpw<2>: four waves/block, two V columns/wave,
// two K components/lane, Q/K reused across the two independent column updates.
// Unlike the donor's algebraic g*dot(S,K), dot(g*S,K) matches the CPU oracle's
// FP32 state decay and detects overflow before the dot product.
template<bool Prefixes>
__global__ __launch_bounds__(256, 2)
void recurrent_decode(GdnDeviceParameters p, GdnDeviceInput in,
                      const float* active_recurrent, GdnDeviceWorkspace w) {
    const int h = blockIdx.x, lane = threadIdx.x;
    const int col0 = (blockIdx.z * 4 + threadIdx.y) * 2;
    const int head_base = h * head_dim * head_dim;
    int flags = 0;
    float state[2][2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int index = head_base + (col0 + c) * head_dim + r * 64 + lane;
            state[c][r] = input_finite(active_recurrent[index], flags);
            if constexpr (Prefixes) w.recurrent_prefixes[index] = state[c][r];
        }
    }
    float decay = 0.0f, beta = 0.0f;
    if (lane == 0) {
        const float bias = input_finite(p.dt_bias[h], flags);
        const float a = parameter_a(p.ssm_a[h], flags);
        const Controls cb = controls(in.alpha[h], in.beta[h], bias, a, flags);
        decay = cb.decay; beta = cb.beta;
    }
    decay = __shfl(decay, 0, 64);
    beta = __shfl(beta, 0, 64);
    float q[2], k[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int index = (h % key_heads) * head_dim + r * 64 + lane;
        q[r] = w.convolved[index];
        k[r] = w.convolved[keys + index];
    }
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        float predicted = 0.0f;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            // ADAPT the verified mx recurrent kernel: PRE-state pair dot,
            // decay after reduction, explicit FMAs under contraction-off.
            predicted = arithmetic_finite(fmaf(state[c][r], k[r], predicted), flags);
        }
        predicted = arithmetic_finite(sum64(predicted), flags);
        const int col = col0 + c;
        const float value = w.convolved[2 * keys + h * head_dim + col];
        const float delta = arithmetic_finite(arithmetic_finite(fmaf(-decay, predicted, value), flags) * beta, flags);
        float read = 0.0f;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            state[c][r] = arithmetic_finite(fmaf(decay, state[c][r],
                arithmetic_finite(k[r] * delta, flags)), flags);
            read = arithmetic_finite(fmaf(state[c][r], q[r], read), flags);
            const int index = head_base + col * head_dim + r * 64 + lane;
            w.recurrent[index] = state[c][r];
            if constexpr (Prefixes) w.recurrent_prefixes[recurrent_elements + index] = state[c][r];
        }
        read = arithmetic_finite(arithmetic_finite(sum64(read), flags) * query_scale, flags);
        if (lane == 0) w.output[h * head_dim + col] = read;
    }
    report_error(flags, w.error);
}

// Exactly one wave per workgroup. Use the furnace wave barrier + compiler fence
// rather than a block fence that drains next-token global prefetches. All LDS
// accesses are ordered before peers use them, without waiting for global loads.
__device__ __forceinline__ void wave_sync() {
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

__device__ __forceinline__ float column_sum(float x) {
    x += __shfl_xor(x, 16, 64);
    return x + __shfl_xor(x, 32, 64);
}

// ADAPT furnace gated_delta_net_lds_wave64<16>, N>1. 48*8 wave workgroups;
// each owns a 16-column state slab for the entire causal loop. State occupies
// 8 KiB LDS, Q/K another 1 KiB; XOR swizzle avoids the 16-column bank conflict.
// Only ingress/egress and optional prefixes use the external unswizzled ABI.
template<bool Prefixes>
__global__ __launch_bounds__(64)
void recurrent_chunk(GdnDeviceParameters p, GdnDeviceInput in, int tokens,
                     const float* active_recurrent, GdnDeviceWorkspace w) {
    constexpr int cols = 16, groups = 4, per_group = head_dim / groups;
    __shared__ float state_lds[cols * head_dim];
    __shared__ float q_lds[head_dim];
    __shared__ float k_lds[head_dim];
    const int h = blockIdx.x, tid = threadIdx.x;
    const int group = tid / cols, local_col = tid % cols;
    const int tile_base = blockIdx.z * cols, col = tile_base + local_col;
    const int swizzle = 2 * local_col;
    const int head_base = h * head_dim * head_dim;
    int flags = 0;
    for (int i = tid; i < cols * head_dim; i += 64) {
        const int c = i / head_dim, k = i % head_dim;
        const int index = head_base + (tile_base + c) * head_dim + k;
        const float s = input_finite(active_recurrent[index], flags);
        state_lds[c * head_dim + (k ^ (2 * c))] = s;
        if constexpr (Prefixes) w.recurrent_prefixes[index] = s;
    }
    float bias = 0.0f, a = 0.0f;
    if (tid == 0) {
        bias = input_finite(p.dt_bias[h], flags);
        a = parameter_a(p.ssm_a[h], flags);
    }
    bias = __shfl(bias, 0, 64);
    a = __shfl(a, 0, 64);
    const float* q_base = w.convolved + (h % key_heads) * head_dim;
    const float* k_base = q_base + keys;
    const float* v_base = w.convolved + 2 * keys + h * head_dim + col;
    float pq[2], pk[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        pq[r] = q_base[tid + r * 64];
        pk[r] = k_base[tid + r * 64];
    }
    float pv = v_base[0], pa = in.alpha[h], pb = in.beta[h];
    float* state_col = state_lds + local_col * head_dim;
    for (int t = 0; t < tokens; ++t) {
        wave_sync(); // previous token finished reading Q/K and updating state
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            q_lds[tid + r * 64] = pq[r];
            k_lds[tid + r * 64] = pk[r];
        }
        const float value = pv, alpha = pa, raw_beta = pb;
        // Clamp the last prefetch to the current token, never read token N.
        const int next = t + 1 < tokens ? t + 1 : t;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            pq[r] = q_base[next * features + tid + r * 64];
            pk[r] = k_base[next * features + tid + r * 64];
        }
        pv = v_base[next * features];
        pa = in.alpha[next * value_heads + h];
        pb = in.beta[next * value_heads + h];
        wave_sync();
        float decay = 0.0f, beta = 0.0f;
        if (tid == 0) {
            const Controls cb = controls(alpha, raw_beta, bias, a, flags);
            decay = cb.decay; beta = cb.beta;
        }
        decay = __shfl(decay, 0, 64);
        beta = __shfl(beta, 0, 64);

        float state[per_group];
        float predicted = 0.0f;
#pragma unroll
        for (int l = 0; l < per_group; ++l) {
            const int k = l * groups + group;
            const float s = state_col[k ^ swizzle];
            state[l] = s;
            predicted = arithmetic_finite(fmaf(s, k_lds[k], predicted), flags);
        }
        predicted = arithmetic_finite(column_sum(predicted), flags);
        const float delta = arithmetic_finite(arithmetic_finite(fmaf(-decay, predicted, value), flags) * beta, flags);
        float read = 0.0f;
#pragma unroll
        for (int l = 0; l < per_group; ++l) {
            const int k = l * groups + group;
            const float s = arithmetic_finite(fmaf(decay, state[l],
                arithmetic_finite(k_lds[k] * delta, flags)), flags);
            state_col[k ^ swizzle] = s;
            read = arithmetic_finite(fmaf(s, q_lds[k], read), flags);
        }
        read = arithmetic_finite(arithmetic_finite(column_sum(read), flags) * query_scale, flags);
        if (group == 0) w.output[t * values + h * head_dim + col] = read;

        if constexpr (Prefixes) {
            wave_sync();
            const int prefix_base = (t + 1) * recurrent_elements + head_base;
            for (int i = tid; i < cols * head_dim; i += 64) {
                const int c = i / head_dim, k = i % head_dim;
                w.recurrent_prefixes[prefix_base + (tile_base + c) * head_dim + k] =
                    state_lds[c * head_dim + (k ^ (2 * c))];
            }
        }
    }
    wave_sync();
    for (int i = tid; i < cols * head_dim; i += 64) {
        const int c = i / head_dim, k = i % head_dim;
        w.recurrent[head_base + (tile_base + c) * head_dim + k] =
            state_lds[c * head_dim + (k ^ (2 * c))];
    }
    report_error(flags, w.error);
}

// ADAPT mx norm.cu::rms_norm_f32<256>, verified in the production gfx906 binary:
// logical32 partials, eight LDS totals (four zero), padded second sum32, rsqrt.
// One block/head/token. Preserve direct gamma and separate output/gate products;
// validate every Z/gamma even when an earlier stage raised a sticky error.
__global__ __launch_bounds__(256)
void normalize_output(GdnDeviceParameters p, GdnDeviceInput in, GdnDeviceWorkspace w) {
    __shared__ float totals[64];
    __shared__ float inverse;
    const int row = blockIdx.x, tid = threadIdx.x;
    const int base = row * head_dim + tid;
    int flags = 0;
    float x = 0.0f, z = 0.0f, gamma = 0.0f, sum = 0.0f;
    if (tid < 64) totals[tid] = 0.0f;
    __syncthreads();
    if (tid < head_dim) {
        x = arithmetic_finite(w.output[base], flags);
        z = input_finite(in.z[base], flags);
        gamma = input_finite(p.norm_weight[tid], flags);
        sum = arithmetic_finite(fmaf(x, x, 0.0f), flags);
    }
    sum = arithmetic_finite(sum32(sum), flags);
    if (tid % 32 == 0) totals[tid / 32] = sum;
    __syncthreads();
    if (tid < 32) {
        const float total = arithmetic_finite(sum32(totals[tid]), flags);
        if (tid == 0) {
            const float norm = arithmetic_finite(total / float(head_dim) + p.rms_epsilon, flags);
            inverse = arithmetic_finite(rsqrtf(norm), flags);
        }
    }
    __syncthreads();
    if (tid < head_dim) {
        float y = arithmetic_finite(arithmetic_finite(x * inverse, flags) * gamma, flags);
        w.output[base] = arithmetic_finite(y * sigmoid(z), flags);
    }
    report_error(flags, w.error);
}

// All producers on the same explicit stream are complete before this launch.
// No producer updates the flag here, so every block sees the same decision.
__global__ void publish(int tokens, float* active_recurrent, float* active_history,
                        float* output, GdnDeviceWorkspace w) {
    if (*w.error != 0) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < recurrent_elements) active_recurrent[i] = w.recurrent[i];
    if (i < history_elements) active_history[i] = w.history[i];
    if (i < tokens * values) output[i] = w.output[i];
}

struct BufferRange {
    std::uintptr_t address;
    std::size_t bytes;
    bool writable;
};

BufferRange buffer_range(const void* pointer, std::size_t bytes, bool writable) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes && (!pointer || address % alignof(float) != 0 ||
                  bytes > std::numeric_limits<std::uintptr_t>::max() - address))
        throw std::invalid_argument("GDN null/misaligned/wrapping buffer range");
    return {address, bytes, writable};
}

bool overlaps(BufferRange a, BufferRange b) {
    if (!a.bytes || !b.bytes) return false;
    return a.address <= b.address ? b.address - a.address < a.bytes : a.address - b.address < b.bytes;
}

void validate_buffers(GdnDeviceParameters p, GdnDeviceInput in, int tokens,
                      float* active_recurrent, float* active_history, float* output,
                      GdnDeviceWorkspace w) {
    if (tokens < 1 || tokens > gdn_device_max_tokens ||
        !std::isfinite(p.rms_epsilon) || p.rms_epsilon <= 0.0f)
        throw std::invalid_argument("GDN requires tokens 1..128 and positive finite RMS epsilon");
    if ((w.recurrent_prefixes == nullptr) != (w.history_prefixes == nullptr))
        throw std::invalid_argument("GDN needs both prefix buffers or neither");
    const auto n = std::size_t(tokens);
    constexpr auto f = sizeof(float);
    const auto prefix_slots = w.recurrent_prefixes ? n + 1 : 0;
    const std::array<BufferRange, 18> buffers{
        buffer_range(p.conv_weights, features * 4 * f, false),
        buffer_range(p.dt_bias, value_heads * f, false),
        buffer_range(p.ssm_a, value_heads * f, false),
        buffer_range(p.norm_weight, head_dim * f, false),
        buffer_range(in.qkv, n * features * f, false),
        buffer_range(in.z, n * values * f, false),
        buffer_range(in.alpha, n * value_heads * f, false),
        buffer_range(in.beta, n * value_heads * f, false),
        buffer_range(active_recurrent, recurrent_elements * f, true),
        buffer_range(active_history, history_elements * f, true),
        buffer_range(output, n * values * f, true),
        buffer_range(w.convolved, n * features * f, true),
        buffer_range(w.recurrent, recurrent_elements * f, true),
        buffer_range(w.history, history_elements * f, true),
        buffer_range(w.output, n * values * f, true),
        buffer_range(w.error, sizeof(int), true),
        buffer_range(w.recurrent_prefixes, prefix_slots * recurrent_elements * f, true),
        buffer_range(w.history_prefixes, prefix_slots * history_elements * f, true)};
    for (std::size_t i = 0; i < buffers.size(); ++i)
        for (std::size_t j = i + 1; j < buffers.size(); ++j)
            if ((buffers[i].writable || buffers[j].writable) && overlaps(buffers[i], buffers[j]))
                throw std::invalid_argument("GDN writable buffer overlap");
}

template<bool Prefixes>
void launch_impl(GdnDeviceParameters p, GdnDeviceInput in, int tokens,
                 float* active_recurrent, float* active_history, float* output,
                 GdnDeviceWorkspace w, hipStream_t stream) {
    convolve<Prefixes><<<(features + 255) / 256, 256, 0, stream>>>(p, in, tokens, active_history, w);
    check_launch(hipGetLastError());
    normalize_qk<<<tokens * (2 * key_heads) / 4, 256, 0, stream>>>(w.convolved, tokens, w.error);
    check_launch(hipGetLastError());
    if (tokens == 1) {
        recurrent_decode<Prefixes><<<dim3(value_heads, 1, head_dim / (4 * 2)), dim3(64, 4), 0, stream>>>(
            p, in, active_recurrent, w);
    } else {
        recurrent_chunk<Prefixes><<<dim3(value_heads, 1, head_dim / 16), 64, 0, stream>>>(
            p, in, tokens, active_recurrent, w);
    }
    check_launch(hipGetLastError());
    normalize_output<<<tokens * value_heads, 256, 0, stream>>>(p, in, w);
    check_launch(hipGetLastError());
    // At N<=128 the recurrent range is at least as large as the output range.
    publish<<<(recurrent_elements + 255) / 256, 256, 0, stream>>>(
        tokens, active_recurrent, active_history, output, w);
    check_launch(hipGetLastError());
}

}

void launch_gdn(GdnDeviceParameters parameters, GdnDeviceInput input, int tokens,
                float* active_recurrent, float* active_history, float* output,
                GdnDeviceWorkspace workspace, hipStream_t stream) {
    validate_buffers(parameters, input, tokens, active_recurrent, active_history, output, workspace);
    if (workspace.recurrent_prefixes)
        launch_impl<true>(parameters, input, tokens, active_recurrent, active_history, output, workspace, stream);
    else
        launch_impl<false>(parameters, input, tokens, active_recurrent, active_history, output, workspace, stream);
}

void launch_gdn_control_witness(GdnDeviceParameters parameters, GdnDeviceInput input,
                                float* output, int* error, hipStream_t stream) {
    const std::array<BufferRange, 6> buffers{
        buffer_range(parameters.dt_bias, value_heads * sizeof(float), false),
        buffer_range(parameters.ssm_a, value_heads * sizeof(float), false),
        buffer_range(input.alpha, value_heads * sizeof(float), false),
        buffer_range(input.beta, value_heads * sizeof(float), false),
        buffer_range(output, 3 * value_heads * sizeof(float), true),
        buffer_range(error, sizeof(int), true)};
    for (std::size_t i = 0; i < buffers.size(); ++i)
        for (std::size_t j = i + 1; j < buffers.size(); ++j)
            if ((buffers[i].writable || buffers[j].writable) && overlaps(buffers[i], buffers[j]))
                throw std::invalid_argument("GDN control witness writable buffer overlap");
    control_witness<<<1, 64, 0, stream>>>(parameters, input, output, error);
    check_launch(hipGetLastError());
}

}
