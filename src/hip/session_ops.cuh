#pragma once

#include "quant.hpp"
#include <hip/hip_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qwen {

inline constexpr int session_ops_max_tokens = 128;
inline constexpr int session_ops_qsa_query_heads = 24;
inline constexpr int session_ops_qsa_key_heads = 2;
inline constexpr int session_ops_qsa_head_dim = 256;
inline constexpr int session_ops_qsa_rotary_dim = 64;
inline constexpr int session_ops_qsa_query_elements = 6144;
inline constexpr int session_ops_qsa_projected_elements = 12288;
inline constexpr int session_ops_router_experts = 512;
inline constexpr int session_ops_error_invalid = 1;
inline constexpr int session_ops_error_arithmetic = 2;

// Small borrowed-buffer FP32 launchers for the caller's current gfx906 device.
// All streams are explicit (nullptr deliberately selects HIP's default stream).
// No allocation, copy, synchronization, device/stream/pointer query or exception.
// Caller owns capacities, device residency, lifetimes and stream dependencies.
// Capacities below are FLOAT ELEMENT counts; error always needs one device int.
// There is no global scratch. Validation checks address arithmetic, not actual
// allocation capacities/accessibility. Every float/int range needs 4-byte
// alignment. Invalid shapes/counts, null pointers, wrapping ranges or writable
// overlaps return hipErrorInvalidValue BEFORE any HIP call. Read-only ranges
// may overlap. Only RoPE permits exact input/output alias; partial alias fails.
// Scaled-add reads/writes its single accumulator argument explicitly in-place,
// but its input must be disjoint. All other outputs must be out-of-place, and
// error must be disjoint from EVERY accessed range. General counts are positive
// int counts (1..INT_MAX); zero/negative counts are errors, not no-ops.
// After one enqueue each launcher returns hipGetLastError(), not GPU completion.
//
// Caller initializes error=0 in stream order before the whole speculative chain.
// Sticky atomic OR: bit 1 = nonfinite numeric input, bit 2 = nonfinite arithmetic.
// Never clears or gates on an earlier error. Nonfinite operands/intermediates
// are sanitized before further arithmetic. A failing pointwise result/pair is
// zeroed; softmax zeros its entire failing row. On error ALL outputs, including
// an overwritten accumulator/in-place RoPE, remain provisional: reject the
// chain after completion before publishing persistent state or using results.
// Reset error only after ordering all readers/writers of the rejected chain.
// Compile session_ops.hip with -ffp-contract=off and without fast/finite-only
// math. Products/adds round separately except the explicit donor RoPE fmaf;
// expf/cosf/sinf/powf are not fast intrinsics.

// projected [N][24][512], each head contains Q[256] then RAW gate[256].
// query/gate [N][24][256]; N=1..128. Capacities projected=N*12288,
// query=gate=N*6144. No normalization, activation or rearrangement within Q.
[[nodiscard]] hipError_t launch_qsa_split_q_gate(const float* projected, int tokens,
        float* query, float* gate, int* error, hipStream_t stream) noexcept;

// input/output [N][heads][256], heads=24 (main Q) or 2 (main K), N=1..128.
// Capacities input/output=N*heads*256, inverse_frequencies=32. Frequencies are
// supplied finite radians/position, including any caller-selected scaling.
// For token t use absolute position start_position+t; start_position>=0 and
// the last position must fit int. Rotate ONLY pairs j/j+32, j=0..31:
// a'=a*cos(theta)-b*sin(theta), b'=a*sin(theta)+b*cos(theta),
// theta=float(start_position+t)*inverse_frequencies[j]. Tail [64,256) is
// copied unchanged when finite (also validated/sanitized for exact alias).
// Text axes share this absolute position. No norm, YaRN or attention scaling.
// Rotation uses fmaf(a,c,-round(b*s)), fmaf(b,c,round(a*s)).
[[nodiscard]] hipError_t launch_qsa_rope(const float* input, const float* inverse_frequencies,
        int tokens, int heads, int start_position, float* output, int* error,
        hipStream_t stream) noexcept;

// Initialize 32 finite text frequencies with the actual mx two-stage expression:
// host theta_scale=powf(base,-2/64), device freq[j]=powf(theta_scale,float(j)).
// base must be finite and positive. Capacities output=32, error=1; disjoint.
// Call once during loading; an unrepresentable device result sets bit2/zeros it.
[[nodiscard]] hipError_t launch_rope_frequencies(float base, float* output,
        int* error, hipStream_t stream) noexcept;

// Capacities input=gate=output=count. output=input*sigmoid(RAW gate).
// In QSA call AFTER inverse Hadamard of the attention output.
[[nodiscard]] hipError_t launch_sigmoid_gate(const float* input, const float* gate,
        int count, float* output, int* error, hipStream_t stream) noexcept;

// Capacities gate=up=output=count. output=SiLU(gate)*up, with RAW projections.
// Expert intermediate width640 uses count=N*640; any positive int count works.
[[nodiscard]] hipError_t launch_silu_up(const float* gate, const float* up, int count,
        float* output, int* error, hipStream_t stream) noexcept;

inline constexpr int silu_pairs_width = 640, silu_pairs_max_columns = 30;
// Opt-in CPUlinear_GPUmiddle handoff: paired [columns][2][640] (gate then up),
// output [columns][640], columns=1..30. Implied exact capacities 1280*columns,
// 640*columns, error=1. Same numeric expression/rounding/sanitization as the
// UNCHANGED launch_silu_up; only addressing differs. All source/output/error
// ranges validated before ANY HIP call; no alias, allocation, transfer or sync.
[[nodiscard]] hipError_t launch_silu_up_pairs(const float* paired, int columns,
        float* output, int* error, hipStream_t stream) noexcept;

// Capacities a=b=output=count. output=a+b; out-of-place even for exact alias.
[[nodiscard]] hipError_t launch_add(const float* a, const float* b, int count,
        float* output, int* error, hipStream_t stream) noexcept;

// Capacities input=accumulator=count. accumulator[i] += scale*input[i]; scale
// is a finite HOST scalar (bad scale is rejected before enqueue). Input cannot
// alias accumulator, even exactly. Contributions must be serialized by caller;
// no atomics on the accumulator. Apply each routed weight AFTER expert down.
[[nodiscard]] hipError_t launch_scaled_add(const float* input, float scale, int count,
        float* accumulator, int* error, hipStream_t stream) noexcept;

// Wide Session's pointwise original-rank reduction, no atomic/group-order sum.
// contributions [N][10][2560] are UNWEIGHTED; weights [N][10] contain original
// normalized FP32 bits. N=1..128, output [N][2560]. Output/error are disjoint
// from all reads and each other; the general read-only alias rule applies.
// Starting at +0, left-fold ranks0..9, each contribution MUL and accumulator ADD
// rounded separately exactly as ten ordered scaled-add launches on finite data.
[[nodiscard]] hipError_t launch_routed_fold(const float* contributions, const float* weights,
        int tokens, float* output, int* error, hipStream_t stream) noexcept;

// Indexed MoE copies have a separate BYTE-ONLY contract from the numeric ops.
inline constexpr int moe_copy_max_input_tokens = 1024;
inline constexpr int moe_layerwise_max_input_tokens = 16384;
inline constexpr int moe_copy_width = 2560;
inline constexpr int moe_copy_ranks = 10;
inline constexpr int moe_copy_q8_blocks = 80;

// Explicit upload DTO, NOT a reinterpretation of RouteAssignment (which also
// has expert and weight). Convert each sorted assignment with {a.token,a.rank},
// retaining its position. Upload the whole sorted array ONCE per layer outside
// these helpers; reuse it for all microtiles. No weight is read/applied here.
struct MoeRouteIndex {
    std::int32_t token;
    std::int32_t rank;
};
static_assert(std::is_standard_layout_v<MoeRouteIndex> && std::is_trivially_copyable_v<MoeRouteIndex>);
static_assert(sizeof(std::int32_t) == 4 && sizeof(MoeRouteIndex) == 8 && alignof(MoeRouteIndex) == 4);
static_assert(offsetof(MoeRouteIndex, token) == 0 && offsetof(MoeRouteIndex, rank) == 4);
static_assert(std::is_standard_layout_v<Q8_1> && std::is_trivially_copyable_v<Q8_1>);
static_assert(sizeof(Q8_1) == 36 && offsetof(Q8_1, d) == 0 && offsetof(Q8_1, s) == 2 && offsetof(Q8_1, qs) == 4);

// A borrowed WHOLE contiguous buffer. capacity is in T elements (Q8 blocks,
// route DTOs, or float elements), not bytes; extra capacity is allowed and is
// included in address/overlap checks, not touched. No ownership or host access.
template<class T> struct MoeCopyView {
    T* data = nullptr;
    std::size_t capacity = 0;
};

// Both copies: input_tokens=1..1024 is the ACTUAL valid token count, not an
// allocation capacity; metadata_entries=1..input_tokens*10 is the valid prefix
// of routes. route_offset>=0, count=1..128, offset+count<=metadata_entries.
// column_capacity>=count describes the whole scratch matrix; its element count
// must fit INT_MAX. All dimensions are explicit and checked before submission.
// Every view must cover its declared shape. Whole declared capacities must fit
// PTRDIFF_MAX and nonwrapping addresses; all pointers need 4-byte alignment,
// including Q8_1 (whose native alignof is only 2). Aligned-16 payload pairs use
// constant 16-byte copies; otherwise constant 4-byte copies. 2880/10240-byte
// rows preserve either alignment. No strict-aliasing type punning.
// Writable whole views and the single int error cannot overlap any read view
// or one another. Read-only views may overlap. Invalid host calls return
// hipErrorInvalidValue before ANY HIP call, without flag/data mutation.
// Current device/stream, actual allocations/residency, slot lifetimes and event
// dependencies remain caller-owned. No allocations, transfers, queries or sync;
// one kernel per helper, then hipGetLastError(), not completion.
//
// BEFORE payload access each GPU entry checks BOTH token and rank against the
// actual input_tokens and top10. An invalid entry atomically ORs sticky bit1;
// gather zeros ONLY its destination column, scatter skips ONLY its write.
// Valid neighbors still copy even if the flag was already set. No flag clear.
// Accepted (token,rank) pairs must be unique, as guaranteed by RouteGroups;
// scatter has no payload atomics. Neither helper inspects Q8 half headers,
// codes/raw sums, float NaNs or weights. ALL payload bits are opaque and are
// preserved, including stored half d=0 with nonzero codes/raw sum. Numeric
// validation belongs to upstream producers/consumers, not these copy helpers.

// input [input_tokens][80], gathered [column_capacity][80]; blocks_per_token
// must equal 80. gathered[c] = input[routes[route_offset+c].token], c<count.
// No conversion/requantization or padding; other gathered columns untouched.
[[nodiscard]] hipError_t launch_moe_gather_q8(MoeCopyView<const Q8_1> input,
        int input_tokens, MoeCopyView<const MoeRouteIndex> routes, int metadata_entries,
        int route_offset, int count, int blocks_per_token, MoeCopyView<Q8_1> gathered,
        int column_capacity, int* error, hipStream_t stream) noexcept;

// down [column_capacity][2560], contributions [input_tokens][10][2560]; width
// must equal 2560 and ranks equal 10. Copy UNWEIGHTED down[c] to
// contributions[(token*10+rank)] for routes[route_offset+c], c<count. No fold,
// scaling, FP arithmetic or float interpretation; other rows untouched.
[[nodiscard]] hipError_t launch_moe_scatter(MoeCopyView<const float> down,
        int input_tokens, MoeCopyView<const MoeRouteIndex> routes, int metadata_entries,
        int route_offset, int count, int width, int ranks, MoeCopyView<float> contributions,
        int column_capacity, int* error, hipStream_t stream) noexcept;

// Separate full-prompt entry points. ONLY input_tokens' host bound differs
// (1..16384). All whole-buffer/overlap/DTO/sticky/opaque-byte rules above, and
// count<=128, are unchanged. Old entry points still reject input_tokens>1024.
[[nodiscard]] hipError_t launch_moe_gather_q8_layerwise(MoeCopyView<const Q8_1> input,
        int input_tokens, MoeCopyView<const MoeRouteIndex> routes, int metadata_entries,
        int route_offset, int count, int blocks_per_token, MoeCopyView<Q8_1> gathered,
        int column_capacity, int* error, hipStream_t stream) noexcept;
[[nodiscard]] hipError_t launch_moe_scatter_layerwise(MoeCopyView<const float> down,
        int input_tokens, MoeCopyView<const MoeRouteIndex> routes, int metadata_entries,
        int route_offset, int count, int width, int ranks, MoeCopyView<float> contributions,
        int column_capacity, int* error, hipStream_t stream) noexcept;

// Same separately rounded multiply/add, with one DEVICE raw gate scalar.
// Capacities input=accumulator=count, raw_gate=1. Computes the direct HIP
// sigmoid before broadcasting, avoiding a CPU-libm/backend rounding change.
// All read ranges must be disjoint from accumulator and error.
[[nodiscard]] hipError_t launch_shared_sigmoid_add(const float* input, const float* raw_gate,
        int count, float* accumulator, int* error, hipStream_t stream) noexcept;

// Token-major row broadcast with the SAME direct device sigmoid and separately
// rounded MUL/ADD as above: input/accumulator [tokens][width], raw_gate[tokens].
// tokens=1..128, width>0 and tokens*width<=INT_MAX. Capacities input/accumulator=
// tokens*width, raw_gate=tokens, error=1. Same borrowed stream/error/alias rules;
// validates the whole shape/ranges before any HIP call, no allocation or sync.
[[nodiscard]] hipError_t launch_shared_sigmoid_add_rows(const float* input, const float* raw_gate,
        int tokens, int width, float* accumulator, int* error, hipStream_t stream) noexcept;

// logits/probabilities [N][512], N=1..128; capacities N*512 each. Stable FP32
// exp(logit-row_max)/sum(exp(...)), one logical width32 group per row on wave64.
// Per-lane ascending strided sixteen-term sum, then qualified DPP XOR stages
// 1/2/4/8/16, matching production topk-moe.cu, followed by reciprocal multiply.
// All 512 logits participate; no top-k, renormalization of selected routes or
// sampling. A nonfinite input or arithmetic (including subtraction overflow
// for extreme finite logits) zeros the ENTIRE row and sets the sticky bits.
[[nodiscard]] hipError_t launch_router_softmax(const float* logits, int tokens,
        float* probabilities, int* error, hipStream_t stream) noexcept;

} // namespace qwen
