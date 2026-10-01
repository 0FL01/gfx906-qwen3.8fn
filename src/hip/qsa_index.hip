#include "qsa_index.cuh"
#include "dpp.cuh"

#include <hip/hip_fp16.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "QSA index requires finite checks; compile without fast/finite-only math"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "qsa_index.hip is specialized for gfx906 wave64"
#endif
#pragma clang fp contract(off)

// Donor-first, D128/Q4/K1/compress4, N=1..128, borrowed FP32 numeric cache:
// KEEP mx dcd685463d597d31f5ca759d32c94592a2740fa4, src/models/qwen4exp.cpp
// ::build_qsa_top_k: raw K cache roundtrip BEFORE chronological mean4, then
// direct folded gamma and RoPE; src/llama-memory-hybrid-idx.cpp::set_input_qsa
// uses block START positions. Do not import its expanded-position top-k.
// ADAPT furnace 905021dbad71c5056ef51f9fd45d545403fc989c,
// ggml/src/ggml-cuda/rope.cu::qsa_pool_norm_rope_f32<T,4> (943..1109):
// two rows/wave64, four adjacent channels/lane, register mean, half-wave RMS,
// LDS normalized rows and split-half rotary pairs. Replace history gathers by
// ONLY new roundtripped rows + active raw tail; explicit text frequencies,
// rotary prefix up to 128 (two frequencies/lane), no mrope/yarn guesses.
// ADAPT same file ::qsa_score_blocks_f32<4> (1207..1235): one wave/block,
// two adjacent channels/lane, existing mx dpp.cuh sum64, per-head ReLU then
// ascending sum4 / sqrt(128); no block bias or position expansion.
// Copyright (c) 2023-2026 The ggml authors. mx/furnace: MIT, license text in
// third_party/mx-LICENSE. DPP primitives retain the required VALU hazard waits.
// Checked reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8 donor map:
// its Q8 dense attention does not supply this QSA/Q4 numeric indexer contract.
// ADAPT the already qualified src/hip/kv.hip::quant/dequant into one pass:
// identical signed scale, first maximum tie, original-FP32-scale codes and
// stored-FP16-scale readback. Its host wrappers throw/allocate on launch errors;
// this adapter instead returns HIP status directly, without changing their API.
// WRITE OURS canonical tail/prefix staging and conditional subset publication;
// CPU semantic oracle: src/qsa_index.cpp. No retained-history copy, allocation,
// device query, default-stream substitution, host wait or hidden logical count.

namespace qwen {
namespace {

constexpr int dim = 128;
constexpr int heads = 4;
constexpr int ratio = 4;
constexpr int tail_elements = 3 * dim;
constexpr int threads = 256;
constexpr int error_invalid = 1;
constexpr int error_arithmetic = 2;
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && sizeof(Q4_0) == 18);

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

bool valid_config(const QsaIndexConfig& c) noexcept {
    return c.head_dim == dim && c.query_heads == heads && c.key_heads == 1 &&
           c.compress_ratio == ratio && c.capacity > 0 && c.capacity <= 131072 &&
           c.rotary_dim > 0 && c.rotary_dim <= dim && c.rotary_dim % 2 == 0 &&
           std::isfinite(c.rms_epsilon) && c.rms_epsilon > 0.0f &&
           std::isfinite(c.rope_attention_scale) && c.rope_attention_scale > 0.0f;
}

unsigned ceil_div(int n, int d) noexcept {
    // Only positive, bounded validated counts reach this helper.
    return static_cast<unsigned>(n / d + (n % d != 0));
}

__device__ __forceinline__ float operand(float x, int& flags) {
    if (!isfinite(x)) { flags |= error_invalid; return 0.0f; }
    return x;
}

__device__ __forceinline__ float arithmetic(float x, int& flags) {
    if (!isfinite(x)) { flags |= error_arithmetic; return 0.0f; }
    return x;
}

__device__ __forceinline__ float frequency(float x, int& flags) {
    if (!isfinite(x) || x <= 0.0f) { flags |= error_invalid; return 0.0f; }
    return x;
}

__device__ __forceinline__ void report(int flags, int* error) {
    if (flags != 0) atomicOr(error, flags);
}

// ADAPT qualified kv.hip quant/dequant, one width32 subgroup/Q4 block. Avoid a
// separate unpack launch and never cast NaN/Inf or an out-of-range float to int.
__global__ void q4_roundtrip(const float* input, int elements,
                            Q4_0* packed, float* output, int* error) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return; // elements and block size are multiples of 32
    const int lane = i % 32;
    int flags = 0;
    const float x = operand(input[i], flags);
    float absolute = fabsf(x), maximum = x;
    int first = lane;
#pragma unroll
    for (int mask = 1; mask < 32; mask *= 2) {
        const float other = __shfl_xor(absolute, mask, 32);
        const float value = __shfl_xor(maximum, mask, 32);
        const int where = __shfl_xor(first, mask, 32);
        const int bad = __shfl_xor(flags, mask, 32);
        if (other > absolute || (other == absolute && where < first)) {
            absolute = other; maximum = value; first = where;
        }
        flags |= bad;
    }
    const float d = (absolute == 0.0f ? 0.0f : maximum) / -8.0f;
    const auto hd = __half_as_ushort(__float2half_rn(d));
    const bool representable = (hd & 0x7c00) != 0x7c00 &&
                              (absolute == 0.0f || (hd & 0x7fff) != 0);
    if (!representable) flags |= error_arithmetic;
    const bool valid = flags == 0;
    const float inverse = arithmetic(valid && d != 0.0f ? 1.0f / d : 0.0f, flags);
    const float candidate = arithmetic(x * inverse + 8.5f, flags);
    int code = 8;
    if (valid && flags == 0) {
        // For qualified finite Q4 input this lies in [0.5,16.5]. This check is
        // deliberately BEFORE the cast, including when upstream scratch is bad.
        if (candidate >= 0.0f && candidate < 17.0f)
            code = min(15, static_cast<int>(candidate));
        else flags |= error_arithmetic;
    }
    const int high = __shfl_xor(code, 16, 32);
    auto& block = packed[i / 32];
    if (lane < 16) block.qs[lane] = static_cast<std::uint8_t>(code | (high << 4));
    const std::uint16_t stored = valid ? hd : 0x8000;
    if (lane == 0) block.d = stored;
    output[i] = arithmetic(__half2float(__ushort_as_half(stored)) * (code - 8), flags);
    report(flags, error);
}

// One channel/thread, small chronological tail in registers. Only the active
// start%4 rows are read; unused pre-state slots are canonical +0 even if their
// backing storage contains stale branch data. All optional prefixes are staged,
// and their logical lengths/checkpoint lifetime remain the caller's concern.
template<bool Prefixes>
__global__ void stage_tail(QsaIndexDeviceParameters p, const float* active_tail,
                          int phase, int tokens, QsaIndexAppendWorkspace w) {
    const int d = threadIdx.x;
    if (d >= dim) return;
    int flags = 0;
    // Validate all immutable parameters, including an append with no completed
    // block. Host validation cannot dereference borrowed device buffers.
    (void) operand(p.query_norm[d], flags);
    (void) operand(p.key_norm[d], flags);
    if (d < static_cast<int>(p.config.rotary_dim / 2))
        (void) frequency(p.inverse_frequencies[d], flags);
    float tail[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        tail[k] = k < phase ? operand(active_tail[k * dim + d], flags) : 0.0f;
        if constexpr (Prefixes) w.tail_prefixes[k * dim + d] = tail[k];
    }
    for (int t = 0; t < tokens; ++t) {
        const float x = arithmetic(w.roundtrip[t * dim + d], flags);
        if (phase == 3) {
#pragma unroll
            for (int k = 0; k < 3; ++k) tail[k] = 0.0f;
            phase = 0;
        } else {
            tail[phase] = x;
            ++phase;
        }
        if constexpr (Prefixes) {
#pragma unroll
            for (int k = 0; k < 3; ++k)
                w.tail_prefixes[(t + 1) * tail_elements + k * dim + d] = tail[k];
        }
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) w.staged_tail[k * dim + d] = tail[k];
    report(flags, w.error);
}

// ADAPT furnace half-wave pooling/norm/RoPE, eight row slots per workgroup.
// Pool=true gathers ONLY the newly completed mean4; Pool=false handles Q[4].
// Scalar accesses preserve the donor's four-adjacent-channel register layout
// without imposing float4/float2 alignment on the frozen float* API.
template<bool Pool>
__global__ __launch_bounds__(256)
void norm_rope(QsaIndexDeviceParameters p, const float* source, const float* tail,
               int rows, int start, float* output, int* error) {
    __shared__ float xs[8][dim];
    __shared__ float cs[8][dim / 2];
    __shared__ float sn[8][dim / 2];
    const int slot = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + slot;
    const bool active = row < rows;
    // Inactive half-waves still participate in DPP and the workgroup barrier.
    const int rc = active ? row : rows - 1;
    const int phase = start % ratio;
    const int position = Pool ? start - phase + rc * ratio : start - 1;
    const int half = static_cast<int>(p.config.rotary_dim / 2);
    int flags = 0;
    float v[4], gamma[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = 4 * lane + k;
        if constexpr (Pool) {
            float sum = 0.0f;
#pragma unroll
            for (int member = 0; member < ratio; ++member) {
                const int relative = rc * ratio + member;
                const float x = relative < phase ? operand(tail[relative * dim + d], flags) :
                    arithmetic(source[(relative - phase) * dim + d], flags);
                sum = member == 0 ? x : arithmetic(sum + x, flags);
            }
            v[k] = arithmetic(sum * 0.25f, flags);
            gamma[k] = operand(p.key_norm[d], flags);
        } else {
            v[k] = operand(source[rc * dim + d], flags);
            gamma[k] = operand(p.query_norm[d], flags);
            (void) operand(p.key_norm[d], flags);
        }
    }
    float square[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) square[k] = arithmetic(v[k] * v[k], flags);
    const float low = arithmetic(square[0] + square[1], flags);
    const float high = arithmetic(square[2] + square[3], flags);
    const float sum = arithmetic(sum32(arithmetic(low + high, flags)), flags);
    const float variance = arithmetic(sum / static_cast<float>(dim) + p.config.rms_epsilon, flags);
    const float inverse = arithmetic(1.0f / sqrtf(variance), flags);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float scaled = arithmetic(v[k] * inverse, flags);
        xs[slot][4 * lane + k] = arithmetic(scaled * gamma[k], flags);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int f = lane + 32 * k;
        if (f < half) {
            const float freq = frequency(p.inverse_frequencies[f], flags);
            const float angle = arithmetic(freq * static_cast<float>(position), flags);
            cs[slot][f] = arithmetic(cosf(angle) * p.config.rope_attention_scale, flags);
            sn[slot][f] = arithmetic(sinf(angle) * p.config.rope_attention_scale, flags);
        }
    }
    __syncthreads();
    if (active) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int d = 4 * lane + k;
            float value = xs[slot][d];
            if (d < 2 * half) {
                const int f = d % half;
                const float a = xs[slot][f], b = xs[slot][f + half];
                // Match the CPU's separate products and addition, not fused mad.
                const float left = arithmetic((d < half ? a : b) * cs[slot][f], flags);
                const float right = arithmetic((d < half ? -b : a) * sn[slot][f], flags);
                value = arithmetic(left + right, flags);
            }
            output[row * dim + d] = value;
        }
    }
    report(flags, error);
}

// A separate stream-ordered launch is the global boundary for ALL producer
// flags. Publish only [start/4, (start+tokens)/4) and the 384-element tail.
// Existing cache keys are neither read, copied nor recomputed during append.
__global__ void publish_append(const float* staged_keys, int new_blocks, int first,
                              const float* staged_tail, float* cache, float* tail,
                              const int* error) {
    if (*error != 0) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int key_elements = new_blocks * dim;
    if (i < key_elements) cache[first * dim + i] = staged_keys[i];
    else if (i < key_elements + tail_elements) tail[i - key_elements] = staged_tail[i - key_elements];
}

// ADAPT furnace one wave per completed block, four independent waves/CTA.
// n_blocks is visible/4, NOT capacity or the prepared future append length.
__global__ __launch_bounds__(256)
void score_blocks(const float* keys, const float* query, int n_blocks,
                  float* scores, int* error) {
    const int lane = threadIdx.x % 64;
    const int block = blockIdx.x * 4 + threadIdx.x / 64;
    if (block >= n_blocks) return; // whole wave only; no block barrier
    int flags = 0;
    const float k0 = operand(keys[block * dim + 2 * lane], flags);
    const float k1 = operand(keys[block * dim + 2 * lane + 1], flags);
    float total = 0.0f;
#pragma unroll
    for (int h = 0; h < heads; ++h) {
        const float q0 = arithmetic(query[h * dim + 2 * lane], flags);
        const float q1 = arithmetic(query[h * dim + 2 * lane + 1], flags);
        const float a = arithmetic(k0 * q0, flags);
        const float b = arithmetic(k1 * q1, flags);
        const float dot = arithmetic(sum64(arithmetic(a + b, flags)), flags);
        total = arithmetic(total + fmaxf(dot, 0.0f), flags);
    }
    const float score = arithmetic(total / sqrtf(static_cast<float>(dim)), flags);
    if (lane == 0) scores[block] = score;
    report(flags, error);
}

} // namespace

hipError_t launch_qsa_index_append(QsaIndexDeviceParameters p, const float* raw_keys,
        int start_consumed, int tokens, float* completed_keys, float* tail,
        QsaIndexAppendWorkspace w, hipStream_t stream) noexcept {
    if (!valid_config(p.config) || start_consumed < 0 || tokens < 1 || tokens > 128 ||
        static_cast<std::size_t>(start_consumed) > p.config.capacity ||
        static_cast<std::size_t>(tokens) > p.config.capacity - static_cast<std::size_t>(start_consumed))
        return hipErrorInvalidValue;
    const int phase = start_consumed % ratio;
    const int new_blocks = (phase + tokens) / ratio;
    // Validate the FULL declared constructor capacities, including the ceil
    // new_keys scratch even when no complete key is produced by this append.
    const auto scratch_blocks = ceil_div(phase + tokens, ratio);
    const auto cache_elements = (p.config.capacity / ratio) * dim;
    if (!valid_ranges(std::array{
            range(p.query_norm, dim), range(p.key_norm, dim),
            range(p.inverse_frequencies, p.config.rotary_dim / 2),
            range(raw_keys, tokens * dim), range(completed_keys, cache_elements, true),
            range(tail, tail_elements, true), range(w.packed, tokens * 4, true),
            range(w.roundtrip, tokens * dim, true), range(w.new_keys, scratch_blocks * dim, true),
            range(w.staged_tail, tail_elements, true),
            range(w.tail_prefixes, w.tail_prefixes == nullptr ? 0 : (tokens + 1) * tail_elements, true),
            range(w.error, 1, true)})) return hipErrorInvalidValue;

    const int elements = tokens * dim;
    q4_roundtrip<<<ceil_div(elements, threads), threads, 0, stream>>>(
            raw_keys, elements, w.packed, w.roundtrip, w.error);
    auto status = hipGetLastError();
    if (status != hipSuccess) return status;
    if (w.tail_prefixes != nullptr)
        stage_tail<true><<<1, dim, 0, stream>>>(p, tail, phase, tokens, w);
    else stage_tail<false><<<1, dim, 0, stream>>>(p, tail, phase, tokens, w);
    status = hipGetLastError();
    if (status != hipSuccess) return status;
    if (new_blocks != 0) {
        norm_rope<true><<<ceil_div(new_blocks, 8), threads, 0, stream>>>(
                p, w.roundtrip, tail, new_blocks, start_consumed, w.new_keys, w.error);
        status = hipGetLastError();
        if (status != hipSuccess) return status;
    }
    publish_append<<<ceil_div(new_blocks * dim + tail_elements, threads), threads, 0, stream>>>(
            w.new_keys, new_blocks, start_consumed / ratio, w.staged_tail, completed_keys, tail, w.error);
    return hipGetLastError();
}

hipError_t launch_qsa_index_score(QsaIndexDeviceParameters p, const float* raw_query,
        const float* completed_keys, int visible_tokens, float* normalized_query,
        float* scores, int* error, hipStream_t stream) noexcept {
    if (!valid_config(p.config) || visible_tokens < 1 ||
        static_cast<std::size_t>(visible_tokens) > p.config.capacity) return hipErrorInvalidValue;
    const int blocks = visible_tokens / ratio;
    // Protect the entire caller-owned cache from output aliases, including its
    // prepared future suffix; the scoring kernel reads ONLY the visible prefix.
    if (!valid_ranges(std::array{
            range(p.query_norm, dim), range(p.key_norm, dim),
            range(p.inverse_frequencies, p.config.rotary_dim / 2),
            range(raw_query, heads * dim), range(completed_keys, (p.config.capacity / ratio) * dim),
            range(normalized_query, heads * dim, true), range(scores, blocks, true),
            range(error, 1, true)})) return hipErrorInvalidValue;
    norm_rope<false><<<1, threads, 0, stream>>>(
            p, raw_query, nullptr, heads, visible_tokens, normalized_query, error);
    const auto status = hipGetLastError();
    if (status != hipSuccess || blocks == 0) return status;
    score_blocks<<<ceil_div(blocks, 4), threads, 0, stream>>>(
            completed_keys, normalized_query, blocks, scores, error);
    return hipGetLastError();
}

} // namespace qwen
