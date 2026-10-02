#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace qwen {

inline constexpr int blocks_device_max_tokens = 128;
inline constexpr int blocks_device_hidden = 2560;
inline constexpr int blocks_device_branches = 4;
inline constexpr int blocks_device_rank = 320;
inline constexpr int blocks_device_wide = 10240;
inline constexpr int ple_device_conv_kernel = 4;
inline constexpr int ple_device_conv_dilation = 3;
inline constexpr int ple_device_history_length = 9;
inline constexpr std::size_t ple_device_history_elements = 92160;
inline constexpr std::size_t ple_device_conv_weight_elements = 40960;
inline constexpr int blocks_device_error_invalid = 1;
inline constexpr int blocks_device_error_arithmetic = 2;

// Raw, borrowed-buffer gfx906 FP32 operations; N (tokens) is always 1..128.
// Every launcher requires an explicit stream argument, including nullptr only
// when the caller deliberately selects HIP's default stream. No allocation,
// copy, synchronization, device/stream/pointer query, or hidden stream is used.
// Caller owns capacities, device residency, buffer lifetimes, uploads, stream
// dependencies and exclusive persistent-state ownership until GPU completion.
//
// Before any HIP call, bad dimensions/epsilon, null required pointers, wrapping
// address ranges, bad alignment or ANY writable overlap return
// hipErrorInvalidValue. All float/int buffers require 4-byte alignment; masks
// require 1-byte alignment. Every writable accessed range (including error and
// scratch) must be disjoint from all other accessed ranges. Read-only ranges
// may overlap each other. In-place activation/norm/combine is NOT supported.
// Capacities listed below are ELEMENT counts, not bytes; validation establishes
// address-range arithmetic, not allocation size/accessibility. No exceptions.
// After enqueue, return hipGetLastError(); hipSuccess is not GPU completion.
//
// error: one borrowed device int, initialized to zero BY THE CALLER in stream
// order before the whole chain. Sticky atomic OR bits: 1 invalid numeric input,
// weight, history, gate or mask; 2 nonfinite FP32 arithmetic; 3 both. Launchers
// never clear it. All nonfinite operands/results are flagged and sanitized to
// zero before further arithmetic/storage; success-path arithmetic is unchanged.
// Pointwise outputs, convolution output/history and prefixes are PROVISIONAL:
// on any error their contents are not valid results, even if individually finite.
// Check error only after completion, before consumption/reuse. Clear it only
// after rejecting the failed chain and ordering all its readers/writers.
//
// Compile blocks.hip with -ffp-contract=off, without fast/finite-only math.
// FP32 products/adds round separately. RMS and PLE dot reduce per-thread
// ascending strided components, wave64 DPP XOR stages 1/2/4/8/16/32, then four
// LDS wave partials (pairwise (0+1)+(2+3) for H2560). CPU ascending sums differ
// in ordering; bit identity is not promised. Uses 1/sqrtf and stable expf-based
// sigmoid, not approximate rsqrt/fast exp. GGUF gamma already folds HF's +1;
// gamma multiplies directly. No projection, quantization or hidden-tap change.

// input/output [N][width]. Supported (group_size, width):
// (2560, 2560/10240), (128, 128/512), (256, 256/512/6144).
// Gamma is [width], repeated for each token, or [group_size] broadcast across
// groups when shared_gamma=true. Required capacities: input/output=N*width,
// gamma=shared_gamma ? group_size : width, error=1. No global workspace.
// Per group: output=(input*(1/sqrt(mean(input^2)+epsilon)))*gamma; epsilon>0.
[[nodiscard]] hipError_t launch_group_rms_norm(const float* input, const float* gamma,
        int tokens, int width, int group_size, float epsilon, bool shared_gamma,
        float* output, int* error, hipStream_t stream) noexcept;

// down_projection/activated [N][320]; required capacities N*320 each, error=1.
// activated=SiLU(down_projection/4). Takes the RAW projection before scaling.
[[nodiscard]] hipError_t launch_hc_lowrank_activation(const float* down_projection,
        int tokens, float* activated, int* error, hipStream_t stream) noexcept;

// normalized/up_projection [N][4][2560], mixed [N][2560]. Required capacities
// N*10240, N*10240, N*2560, error=1. up_projection is RAW, not sigmoid-activated.
// mixed[j]=(sum branches in order 0..3 sigmoid(up[b,j])*normalized[b,j])/4.
// Head-only mixing uses this same operation without injection.
[[nodiscard]] hipError_t launch_hc_mix(const float* normalized, const float* up_projection,
        int tokens, float* mixed, int* error, hipStream_t stream) noexcept;

// inject_projection/injection [N][4], capacities N*4 each, error=1.
// injection=2*sigmoid(inject_projection/4); input is RAW projection.
[[nodiscard]] hipError_t launch_hc_injection(const float* inject_projection,
        int tokens, float* injection, int* error, hipStream_t stream) noexcept;

// original_wide/output [N][4][2560], block [N][2560], injection [N][4].
// Capacities N*10240, N*2560, N*4, N*10240, error=1. Gate must be finite in
// [0,2]. output[b,j]=original_wide[b,j]+(injection[b]*block[j]). Input remains
// the original four-branch residual (including the full widened MTP tap).
[[nodiscard]] hipError_t launch_hc_combine(const float* original_wide, const float* block,
        const float* injection, int tokens, float* output, int* error,
        hipStream_t stream) noexcept;

// norm_key/norm_query/output [N][4][2560], shared_value [N][2560]. Capacities
// N*10240, N*10240, N*2560, N*10240, error=1. For each token/branch:
// s=dot(norm_key,norm_query)/sqrt(2560),
// gate=sigmoid(sign(s)*sqrt(max(abs(s),1e-6))), output=gate*shared_value.
// Both signs of zero have sign=0 and gate=0.5 (the floor does not change that).
// key/query normalization is explicit via launch_group_rms_norm above.
[[nodiscard]] hipError_t launch_ple_gate(const float* norm_key, const float* norm_query,
        const float* shared_value, int tokens, float* output, int* error,
        hipStream_t stream) noexcept;

struct PleConvDeviceWorkspace {
    float* output = nullptr;   // N*10240 provisional convolution output
    float* history = nullptr;  // 92160 staged post-chunk history
    // Optional (N+1)*92160, chronological slot n = after n consumed inputs;
    // slot 0 is the pre-chunk history, slot N is identical to history.
    float* prefixes = nullptr;
};

// gated/norm_gated [N][4][2560], weights [10240][4] oldest tap first,
// active_history [10240][9] oldest age first; all contiguous FP32.
// Capacities N*10240 each, weights=40960, active_history=92160, error=1, plus
// workspace capacities above. Optional keep_mask [N] contains only 0/1 (null
// means all ones). Mask zero replaces BOTH current gated/norm_gated by zero
// AFTER validating them; it still convolves history and appends a zero age.
// EOS is a hash-only boundary; this operation never clears history on EOS.
//
// conv=(((0+h[0]*w[0])+h[3]*w[1])+h[6]*w[2])+norm_gated*w[3],
// output=gated+SiLU(conv). History shifts one position and appends norm_gated,
// NOT gated/raw projection/convolved output. The causal token loop resides in
// each feature's thread registers. active_history is READ ONLY throughout.
// On any failing token the entire chunk remains speculative; launch a separate
// conditional publication below AFTER all contributing kernels on this stream.
[[nodiscard]] hipError_t launch_ple_causal_conv(const float* gated, const float* norm_gated,
        const float* weights, const float* active_history, int tokens,
        const std::uint8_t* keep_mask, PleConvDeviceWorkspace workspace,
        int* error, hipStream_t stream) noexcept;

// staged/active_history capacities 92160 each, error=1. staged can be either
// workspace.history or workspace.prefixes+n*92160, n in [0,N], for commit or
// rollback. A finite-validation kernel precedes a separate publication kernel;
// ONLY error==0 publishes. Any earlier numeric failure or nonfinite staged
// value leaves EVERY active history element untouched. Keep the same sticky
// error across the producing chain and publication; clearing it in between
// would incorrectly authorize invalid speculative state. Other streams need
// caller-supplied dependencies. Persistent consumed-token counts are caller
// state and may advance only after confirmed successful publication.
// This guarantee covers numeric failures, not HIP/device faults or concurrent
// misuse. Provisional output/prefixes must still be rejected if error!=0.
[[nodiscard]] hipError_t launch_publish_ple_history(const float* staged,
        float* active_history, int* error, hipStream_t stream) noexcept;

} // namespace qwen
