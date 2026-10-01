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
  Modifications: fixed tiled GGUF geometry, decay-before-dot FP32 ordering,
  chronological prefixes, checked conv/L2/RMS/sigmoid preprocessing,
  no fast intrinsics, finite staging and chunk-atomic publication.
  No graph/cache/runtime code is imported. CPU dense/GDN oracles are original;
  format and architecture reference revisions are recorded in source comments.

Upstream URLs and rejected donor options: RECON.md section 16.
