#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

namespace qwen {

// Fixed keep1 GGUF geometry; all buffers below contain contiguous FP32 values.
inline constexpr int gdn_device_key_heads = 16;
inline constexpr int gdn_device_value_heads = 48;
inline constexpr int gdn_device_head_dim = 128;
inline constexpr int gdn_device_conv_width = 4;
inline constexpr int gdn_device_max_tokens = 128;
inline constexpr std::size_t gdn_device_qkv_elements = 10240;
inline constexpr std::size_t gdn_device_value_elements = 6144;
inline constexpr std::size_t gdn_device_recurrent_elements = 786432;
inline constexpr std::size_t gdn_device_history_elements = 30720;

struct GdnDeviceParameters {
    // Borrowed read-only weights, despite the mutable pointer type. Conv taps
    // are [10240][4], oldest first; bias/A are [48], norm weight is [128].
    // ssm_a is already -exp(HF A_log), <= 0. Norm weight multiplies directly.
    float* conv_weights = nullptr;
    float* dt_bias = nullptr;
    float* ssm_a = nullptr;
    float* norm_weight = nullptr;
    float rms_epsilon = 1.0e-6f;
};

struct GdnDeviceInput {
    // Token-major RAW projections: qkv[t][Q2048,K2048,V6144], z[t][6144],
    // alpha/beta[t][48]. All V-side arrays have GGUF tiled head order h%16.
    const float* qkv = nullptr;
    const float* z = nullptr;
    const float* alpha = nullptr;
    const float* beta = nullptr;
};

struct GdnDeviceWorkspace {
    float* convolved = nullptr; // tokens*10240; SiLU conv V, normalized Q/K
    float* recurrent = nullptr; // 786432; staged next recurrent state
    float* history = nullptr;   // 30720; staged next raw convolution history
    float* output = nullptr;    // tokens*6144; staged normalized/gated output
    // Both null or both nonnull. Chronological slot 0 is pre-chunk state;
    // slot n is after n CONSUMED inputs. No reverse slots or layout conversion.
    float* recurrent_prefixes = nullptr; // (tokens+1)*786432
    float* history_prefixes = nullptr;   // (tokens+1)*30720
    // Caller resets to zero on stream before launch, and checks after completion
    // before consuming results. Sticky bitmask: 1 invalid input/weights/state,
    // 2 nonfinite FP32 arithmetic; 3 means both. Never reset by launch_gdn.
    int* error = nullptr;
};

// Borrowed device buffers on the caller's gfx906 device and explicit stream;
// no allocation, device query, host synchronization or hidden/default stream.
// tokens must be 1..128. Null/misaligned/overlapping pointers and bad tokens or
// epsilon throw invalid_argument; immediate HIP launch errors throw runtime_error.
// Allocation capacities/device residency/lifetimes are the caller's contract.
// All writable ranges must be disjoint from every other range; read-only input
// and weight ranges may overlap each other. The caller serializes state ownership.
//
// External recurrent[(h*128+v)*128+k] has K contiguous. History[feature*3+age]
// is oldest -> newest RAW projection, not activated/convolved values.
// Convolution uses SiLU; output is RMSNorm(epsilon supplied) * sigmoid(z).
// Q/K L2 epsilon is independently fixed to 1e-6; normalized Q includes 1/sqrt(128).
//
// Chunk-atomic numeric failure: only a final error==0 kernel publishes active
// recurrent/history and caller output. This is stronger than CPU per-token
// atomicity: ANY failing token leaves the ENTIRE active chunk/output unchanged.
// Scratch, including prefixes, is speculative and must not be consumed on error.
// This does not recover device faults or concurrent misuse. Compile without
// fast-math; state, intermediates and snapshots remain FP32.
void launch_gdn(GdnDeviceParameters parameters, GdnDeviceInput input, int tokens,
                float* active_recurrent, float* active_history, float* output,
                GdnDeviceWorkspace workspace, hipStream_t stream);

}
