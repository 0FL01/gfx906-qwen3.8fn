# Kernel attribution

- `src/hip/dpp.cuh` ports only gfx906 DPP primitives from
  mx `ggml/src/ggml-cuda/common.cuh`, revision
  `dcd685463d597d31f5ca759d32c94592a2740fa4`.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`.
- `src/hip/expert.hip` adapts mx canonical SDOT4/Q4 arithmetic,
  Q8_1 activation quantization and byte-preserving Q4_0 planar packing
  from that revision (`vecdotq.cuh`, `quantize.cu`, `q8_repack/`).
  The CPU `src/quant.cpp` arithmetic contract and GPU quantizers in
  `src/hip/expert.hip` / `src/hip/linear.hip` retain the pinned
  `quantize.cu::quantize_q8_1` original-FP32-scale codes and independent raw
  input sum when a legitimate positive scale rounds to stored FP16 zero.
  The long-fixture repair corrects validation of this existing ABI, separately
  from optimization: nonzero-input blocks with original FP32 scale0, unsafe
  rounded codes±128, nonfinite values and FP16 Inf/NaN headers still reject
  before unsafe int8 conversion; logical32 reduction groups remain isolated
  within wave64.
  It introduces no weight/precision/epsilon/gate change, NaN clamp or all-zero
  replacement of valid blocks. No new donor component or dependency is added.
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
  chronological prefixes, exact N1..128 CPW2 register-state chronological token
  loop (extended from the earlier N1/2/3 dispatch), strengthened exact
  state/prefix/continuation and late-error atomic-publication fixtures,
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
  Modifications: borrowed checked buffers, N1/2/3 reuse plus a separate checked
  N1..8 short-column launcher over the same templates (the original N≤3 API
  remains unchanged), mx `mmvq.cu` two-wave
  topology and format-specific fragments with explicit FMA, canonical Q6 MMVQ
  slices, unchanged raw-sum/half-product ABI,
  checked finite headers/original FP32 scale and safe-code activation errors,
  including valid stored-half-zero scales as described above. `linear_reference.cpp`
  is an original independently decoded common-Q8 scalar oracle.
- `src/hip/blocks.hip` adapts mx `norm.cu` strided/wave/LDS reductions and
  furnace `dsv4-hc.cu` coalesced pre/post patterns at the revisions above.
  Modifications: Qwen C4 sigmoid/SiLU/direct-gamma equations (not DeepSeek
  Sinkhorn), canonical logical reductions/direct unary expressions, strict finite
  arithmetic and provisional outputs. PLE dot/reduction/scale adapts mx
  `sumrows.cu`, `reduce_rows.cuh`, `scale.cu`; dilation3 state/prefix staging and
  conditional history publication are original. Short PLE scheduling retains
  the existing 512-thread gate reduction for tiles of at most eight, with
  unchanged hash, normalization and convolution arithmetic.
  Reinstinct RMS is a layout/design reference, not an imported runtime.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`.

- `src/hip/dense.hip` adapts mx `mmvf.cu` pairwise FMA accumulation and padded
  two-stage DPP/LDS reductions for aligned even-K N1..8 (extended from N1/2/3);
  N9..128 and other unsupported MMVF shapes retain the existing rocBLAS
  dependency. BF16 values are decoded exactly to F32.
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
  Bounded prefill orchestration is original code informed by the pinned mx
  `src/llama-moecache.cpp`, `docs/development/moe-cache-prefill.md` and
  `ggml/src/ggml-cuda/ggml-cuda.cu` staging/layout-aware D2D designs, furnace
  `ggml-cuda.cu` pipeline ordering, and reinstinct `moe_expert_sort.cpp`,
  gather/scatter patterns and `src/runtime/pipeline.rs`. It groups routes once
  per full logical chunk, acquires one canonical triplet per expert group,
  and uses chronological nonexpert/shared/expert compute tiles of at most eight.
  Band16/two-stage-per-device `copy_ready`/`consumer_done` lifetime is original;
  donor scheduler, GPU-resident expert slab and whole-runtime infrastructure
  are not imported. This scheduling introduces no new weight/activation ABI,
  arithmetic or dependency; the existing mx/furnace MIT and reinstinct
  Apache-2.0 attribution above remains applicable to the adapted primitives.
  Completed-call `SessionRouteStats`/`route_stats()` diagnostics and the logical
  wide/real4K/16K long-prefill fixtures/collectors are original code. They measure
  assignments before physical tiling and introduce no new donor component or
  license. Completed remote/local actual-log validation and canonical journal
  recording reuse these original collectors without new donor code/dependencies.
  Diagnostic trace-range capture and CLI/primary-sampling integration
  are original code; the long correctness/sampling-prerequisite closure imports
  no donor whole runtime. Protocol1 full-request timing/output/expert-counter
  telemetry, the request collector and paired chunk A/B/A recording are original
  code. `tools/observe_vram.py` is an original Python-stdlib-only sampled global
  AMD driver-VRAM observer. Its request join in `tools/record_request.py` accepts
  the closed direct attached Docker argv shape, matches structured native options
  and prompt IDs to the request source, and validates both completed device
  observations. It imports no shell/runtime machinery, attests no binary/mount
  identity or HIP-index mapping, and does not turn samples into exact peaks or
  HIP-owned allocations. All8 actual raws were downloaded and passed local
  collection, including both observer joins; historical canonical30-row/2,150,750-byte
  history preservation and +9/-0 diff were verified. The accepted, closed
  measurement-only slice contains docs/CMake/observer/request tools/tests/results;
  experimental Session/GPU/R5/speculative changes are outside it. No new dependency,
  donor component or license is introduced by this telemetry/join.
  Completed frozen-b522/dirtytrue rocprofv3 diagnostics are measurement evidence,
  not an imported runtime or candidate qualification; PP+TG overlapping duration
  sums do not identify total latency or prove a RAM/expert-DMA bottleneck.
  The current Session/new route kernels/fixture and CPU-linear/GPU-middle
  correctness slice passed remote job765: strict CXX20/HIP20 gfx906 Release,
  warning gates/CTest35/35 (735.71s), both-GPU route-copy/paired-middle fixtures,
  full hybrid and default regressions/actual collectors. Artifacts retain snapshot
  `a4b55d84724ba15bbae7013d5a107b7671b7a409`/dirtytrue, not a future commit.
  The route candidate uses one 80*N-byte DTO upload per layer and indexed Q8 gather/down
  scatter per microtile<=8, preserving copied bits/expert fold order.
  `src/cpu_expert.hpp` / `src/cpu_expert.cpp`, persistent bounded worker pool,
  paired pinned-frame orchestration and two-stage gate/up then down scheduling,
  route-copy fixtures, hybrid fixture/collector and bounded input/witness/failure
  diagnostics are original code. The staged method reuses the already attributed
  canonical CPU quant arithmetic and GPU unary/Q8 primitives; paired addressing
  introduces no new donor algorithm. CPU gate/up/down surround canonical GPU
  SiLU/Q8 aggregated once per CPU-bearing layer; `force_cpu` means
  CPU_LINEAR_GPU_middle. The host-libm whole-expert path remains an oracle and
  is not full-model qualified (diagnostic755/759); no quantizer ABI/GPU-math,
  weight/precision/epsilon/gate change or tolerance waiver is introduced.
  Dedicated copy-stream middle buffers/flag, event-protected pinned-frame reuse,
  CPU-only rank contribution H2D and original-rank/shared fold are original
  orchestration; captured-reader/pending-ID/admission/error-reset fixtures include
  a synthetic failure, not an actual-inflight timing claim. Default cpu_workers=0
  retains the historical GPU-only path/allocations. Threads/standard-library
  facilities add no third-party dependency, donor runtime or new license.
  The hybrid protocol reports added-buffer capacities, not full RSS/worker stacks
  or measured all-owner old-buffer release; source-derived175 rejection checks
  are not serialized per-rejection observations. Missing stats/routes/input
  payloads/steady owned ledgers are not invented by the collector.
  Parent local collection of the downloaded272019-byte actual hybrid passed,
  with all compared metrics zero. Canonical ROOT journal has exactly one
  r5_hybrid append30→31 / 2167805 bytes; exact old Git HEAD30 byte prefix,
  parsed history and +1/-0 are verified. This bounded correctness slice is
  accepted, covering Session/CPU/ops, relevant fixtures/CMake/collector/docs/
  journal; accepted raw must not be appended again. Source remains a4b55d8/
  dirtytrue, not a future closure hash. Exclusive trace-off paired job
  1791073403122-769 is RUNNING with already-built765 versus savedb522/dirtytrue;
  it has no completed performance result and imports no donor code/dependency.
  No GPU-copy/hybrid performance win, measured dispatch threshold, fullR5 or
  MTP result is claimed. The three post765 attention files and three future
  Spec/R6 pure-math helper files are outside this accepted closure; they are
  not trained-MTP integration or actual HIP/model/performance qualification.
  Existing mx/furnace MIT and reinstinct Apache-2.0 attribution is retained.

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

- `src/hip/mmq_wide.hip` adapts the same pinned mx canonical Q5_0/Q8_0/Q6_K
  `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, `mmq.cuh` and `vecdotq.cuh` seams.
  Modified: literal format-specific gfx906 tiles, unchanged DS4 half headers,
  alignment2 loads, uniquely owned LDS writes and predicated zero tails.
  Q5 keeps unsigned codes and the original raw-sum correction rather than the
  donor's centered-code/code-sum approximation. Q6 retains signed subscale
  integer grouping before conversion; explicit FMA association is qualified
  against common-Q8 inputs. Q4 forwards to the existing implementation.
  I128 launch bounds reflect one-block LDS capacity, not claimed occupancy.
  No donor quantizer, allocation/runtime/configuration framework is imported.
  Copyright (c) 2023-2026 The ggml authors; MIT in `mx-LICENSE`.

Upstream URLs and rejected donor options: RECON.md section 16.
