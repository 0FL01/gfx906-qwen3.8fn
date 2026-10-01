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

Upstream URLs and rejected donor options: RECON.md section 16.
