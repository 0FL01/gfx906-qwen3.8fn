# Kernel attribution

- `src/hip/dpp.cuh` ports only gfx906 DPP primitives from
  mx `ggml/src/ggml-cuda/common.cuh`, revision
  `dcd685463d597d31f5ca759d32c94592a2740fa4`.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`.
- `src/hip/expert.hip` adapts mx canonical SDOT4/Q4 arithmetic,
  Q8_1 activation quantization and byte-preserving Q4_0 planar packing
  from that revision (`vecdotq.cuh`, `quantize.cu`, `q8_repack/`).
  Runtime/graph/cache code is not imported.
- Multi-column, two-output-row weight reuse in `src/hip/expert.hip`
  is adapted from sixvolts/reinstinct
  `kernels/matvec_q4_0_repacked_batched.cpp`, revision
  `0b79e326351d90d4554a1c18df92da5d0ab692e8`, Apache-2.0 (see `Apache-2.0`).
  Modifications: canonical and planar Q4_0 plus Q4_1, mx 36-byte Q8_1
  with half scales/raw sums instead of donor 40-byte Q8/quantized-sum
  correction; mx DPP reduction, and bounded matrix/expert calls.
- `src/kv.cpp`, `src/hip/kv.hip` and the test-only serial pack in
  `src/kv_main.hip` adapt mx `cpy-utils.cuh` / `ggml-quants.c` Q4_0
  quantization from the same production revision. Modified: checked shapes,
  finite/representable scales, cooperative width32 first-maximum reduction.
  `src/hip/kv.hip` ports only the wave64 transform from mx `fwht.cu`.
- `tests/qsa_test.cpp` contains a CPU diagnostic adaptation of mx's strict
  bitonic comparison network (`argsort.cu`), with the full MIT notice inline.
  It emulates expanded-position selection; it is not a runtime backend.
  `src/qsa.cpp` is an original completed-block selector based on Qwen/HF
  Transformers `a005fc82babfe8871d87746decad2dbee100a125` semantics.
- `src/hip/gdn.hip` adapts the mx wave/column state ABI and furnace
  `ggml/src/ggml-cuda/gated_delta_net.cu` CPW2 decode and resident wave64
  slab kernels, revision `905021dbad71c5056ef51f9fd45d545403fc989c`.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`.
  Furnace's resident design derives from reinstinct
  `kernels/gdn_recurrent_batched_v2.cpp` at the Apache-2.0 revision above.
  Modifications: fixed tiled GGUF geometry, canonical FP32/FMA PRE-state dot
  and decay-after-reduction/update ordering, additive Q/K epsilon,
  chronological prefixes, N1/2/3 CPW2 resident register-state token loop,
  checked conv/L2/RMS/sigmoid preprocessing,
  no fast intrinsics, finite staging and chunk-atomic publication.
  No graph/cache/runtime code is imported. CPU dense/GDN oracles are original;
  format and architecture reference revisions are recorded in source comments.

- `src/hip/qsa_index.hip` adapts furnace `rope.cu` D128 pooling/RMS/RoPE
  and wave64 scoring at the furnace revision above. It reuses qualified mx
  Q4_0 arithmetic: first signed maximum, original FP32 codes, half-scale readback.
  Original append state, finite staging, chronological tails/prefixes and
  conditional publication replace donor ggml/runtime/layout-cache machinery.
- `src/hip/qsa_select.hip` adapts furnace `top-k.cu` radix histogram/cutoff/gather
  and the mx bitonic comparison pattern. Modified: unique score/ID rank keys,
  deterministic lower-ID ties, 512 whole blocks, actual tail/count, bounded scratch,
  invalid-input rejection and conditional publication, not expanded positions.
- `src/hip/attention.hip` adapts furnace `fattn.cu` D256/G12 64-key chunk and
  stable split-K merge. Modified: direct selected IDs, bounded canonical Q4_0
  to RNE FP16 chronological gather (bounded mx `argsort.cu` bitonic pattern),
  ascending within-chunk dot/value arithmetic and explicit internally widened
  exponential evaluation, no full mask/history copy, finite checks and conditional
  output publication. Copyright (c) 2023-2026 The ggml authors; full MIT notices
  appear inline and in `mx-LICENSE`. Reinstinct partial/merge at the revision
  above is a design/algebra reference, not an imported Q8 backend.

- `src/hip/linear.hip` adapts mx canonical Q4_0/Q4_1/Q5_0/Q8_0/Q6_K
  `vecdotq.cuh`, Q8_1 `quantize.cu` and qualified R1 register-reuse/DPP patterns.
  Modifications: borrowed checked buffers, N1/2/3 reuse, mx `mmvq.cu` two-wave
  topology and format-specific fragments with explicit FMA, canonical Q6 MMVQ
  slices, unchanged raw-sum/half-product ABI,
  representable-scale and finite activation errors. `linear_reference.cpp`
  is an original independently decoded common-Q8 scalar oracle.
- `src/hip/blocks.hip` adapts mx `norm.cu` strided/wave/LDS reductions and
  furnace `dsv4-hc.cu` coalesced pre/post patterns at the revisions above.
  Modifications: Qwen C4 sigmoid/SiLU/direct-gamma equations (not DeepSeek
  Sinkhorn), canonical logical reductions/direct unary expressions, strict finite
  arithmetic and provisional outputs. PLE dot/reduction/scale adapts mx
  `sumrows.cu`, `reduce_rows.cuh`, `scale.cu`; dilation3 state/prefix staging and
  conditional history publication are original.
  Reinstinct RMS is a layout/design reference, not an imported runtime.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`.

- `src/hip/dense.hip` adapts mx `mmvf.cu` pairwise FMA accumulation and padded
  two-stage DPP/LDS reductions for aligned even-K N1/2/3; other shapes retain
  the existing rocBLAS dependency. BF16 values are decoded exactly to F32.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`, same mx pin above.
- `src/hip/session_ops.hip` adapts mx `rope.cu`, `topk-moe.cu`, `common.cuh`
  and direct `unary.cu`/`unary.cuh` arithmetic. Modified: supplied text geometry,
  checked borrowed buffers, logical32 router and sticky-error sanitization.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`.
  Session/cache orchestration and separate diagnostic oracle programs are
  original code; no donor runtime, scheduler or model graph is imported.

- Short-window grouping in `src/session.hip` reuses the qualified canonical
  multi-column linear primitives above. Stable CPU route grouping, original-rank
  scatter/fold, same-stream slot lifetime and row-wise shared-gate orchestration
  are original code; no donor expert scheduler or runtime is imported.

- `src/hip/mmq.hip` adapts mx `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`,
  `mmq.cuh` and `vecdotq.cuh` canonical Q4 DP4A tile/load/dot/writeback seams
  at the mx revision above. Modified: literal gfx906 I64/K256/four-wave tiles,
  borrowed checked buffers, predicated owned-zero tails (including K640),
  alignment2 canonical loads, explicit FMA/half-product policy and unchanged-Q8
  byte transposition into DS4. The measured M≤640 J8 microtile dispatch is an
  original literal shape-specific adaptation; tile arithmetic is unchanged.
  No donor quantizer, allocator, stream-K,
  scheduler or whole GGML backend is imported.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`.

Upstream URLs and rejected donor options: RECON.md section 16.
