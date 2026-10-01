# Qwen3.8-Flash-Next core for 2×gfx906

Standalone C++20/HIP core under implementation. The model is **not yet runnable**
in this core; `core-probe` is R0 hardware/build and `core-expert` is R1 execution
of one real routed expert, not the full network.
Scope and acceptance are in [PLAN.md](PLAN.md); current evidence in [STATE.md](STATE.md).

## Remote build and R0 probe

Mirror `CMakeLists.txt`, `src/`, `tools/`, `tests/` through `mi50-llama-remote`
to `/home/radneon/gfx906-core/src/`. Keep weights at their existing paths.
The working build image is `llama.cpp-gfx906:cmake-4.4.3`, ROCm 7.14,
Clang 23; production runtime image does not contain CMake.

```sh
docker run --rm --name core-build --entrypoint /bin/sh \
  -v /home/radneon/gfx906-core:/core \
  -e CORE_REVISION="$CORE_REVISION" -e CORE_DIRTY="$CORE_DIRTY" \
  llama.cpp-gfx906:cmake-4.4.3 /core/src/tools/build.sh
docker run --rm --name core-probe --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-probe -v /home/radneon/gfx906-core:/core \
  llama.cpp-gfx906:cmake-4.4.3 > /home/radneon/gfx906-core/runs/r0-probe.jsonl
python3 -B /home/radneon/gfx906-core/src/tools/record_probe.py \
  /home/radneon/gfx906-core/runs/r0-probe.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Before building, set `CORE_REVISION` to the synchronized controller commit and
`CORE_DIRTY` to `OFF` (clean) or `ON` (changed sources). No host `-march=native`; AVX2/FMA/F16C
are checked before target-attributed CPU code. GPU compilation explicitly targets
gfx906. Probe errors exit nonzero; transfer/consumer and both GEMMs are checked.
RAM read/FMA and concurrent read/H2D are synthetic hardware measurements, **not**
quantized expert throughput or full-request speed.

## R1: validated GGUF and one real expert

`Model` owns a checked GGUF v3 file descriptor and typed metadata/inventory;
only bounded header and explicitly requested slices are read. It supports the
eight actual target types, validates dimensions, block sizes, arithmetic,
alignment, overlaps and file bounds. No tensor mmap, whole-weight dequantization
or second RAM expert inventory. Sidecar sharing remains explicit.

```sh
docker run --rm --entrypoint /core/build/core-inspect \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf
docker run --rm --entrypoint /core/build/core-inspect \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
docker run --rm --name core-expert --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-expert -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 \
  /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/r1-expert-final.jsonl
python3 -B /home/radneon/gfx906-core/src/tools/record_expert.py \
  /home/radneon/gfx906-core/runs/r1-expert-final.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Layer0/expert0 retains canonical Q4_0 gate/up and Q4_1 down (2,867,200 bytes).
Activation ABI is **mx Q8_1, 36 bytes**: FP16 scale, FP16 sum of **raw** float
inputs, 32 signed codes. Codes use the original FP32 `amax/127` and `roundf`;
the raw sum uses ascending XOR 1/2/4/8/16. Q4_0 uses
`d4*(integer_dot*d8-8*raw_sum)`. Q4_1 keeps mx half-rounded `d4*d8` and
`m4*raw_sum`, not quietly changed FP32 products. CPU oracle uses portable RNE
half; production CPU uses checked AVX2/F16C in the complete matrix loop, no
saturating int16 sum (Q4 pair bound 3840), no implicit FMA reassociation.

GPU adapts mx SDOT4/DPP and multi-column weight reuse after donor inspection
(attribution in `third_party/NOTICE.md`). Canonical is the default; a byte-preserving
planar Q4_0 slot pack remains a qualification candidate, not a second full RAM
copy. Its activation ABI/correction is **not** reinstinct's 40-byte Q8 ABI.
Q4_1 remains canonical. Each wave computes two rows and reuses a block for up
to three columns. N=128 is a PP-sized fixture, **not** R4 grouped MMQ prefill.

Fixture gates were fixed before GPU execution: common-quant linear error
`2e-4 + 2e-5*|reference|`; float expert pipeline `2e-3 + 2e-4*|reference|`.
Tests include identical CPU/GPU packed bytes, half subnormals, rounding ties,
signed/zero scales, negative-zero blocks, int8 extrema, invalid input rejection
and valid reuse after rejection. CPU A/B/A compares old per-block calls against
the inlined/F16C path; GPU A/B/A compares canonical/planar/canonical separately
on both devices, never holding competing candidates simultaneously.

Measurements in `results.jsonl` use one pinned CPU worker and **repeated hot
weights**. GPU input is uploaded once before warmup; resident pipeline events
and completed wall time are distinct. Upload/pack includes weight allocations
and cold initialization effects, not isolated steady-state DMA. These are not
DDR miss throughput, cache policy, logits, TG or whole-request measurements.

## Existing production baseline (separate process)

No other GPU workload or heavy compiler during final measurements. If the named
container exists, inspect it first; do not blindly launch a second model instance.

```sh
sh /home/radneon/gfx906-core/src/tools/baseline-server.sh
python3 -B /home/radneon/gfx906-core/src/tools/baseline.py \
  --runs-dir /home/radneon/gfx906-core/runs \
  --results /home/radneon/gfx906-core/results.jsonl \
  --revision dcd685463d597d31f5ca759d32c94592a2740fa4
docker logs core-baseline > /home/radneon/gfx906-core/runs/r0-baseline-server.log 2>&1
docker stop --timeout 30 core-baseline
```

The client saves new exact token-ID fixtures, raw requests/responses and router
configuration, checks 32+64 and 4096+512 counts without prefix reuse, and uses
temperature 1.0 / top-p 0.95 / top-k 20. `ignore_eos=true` is recorded explicitly
to fix output lengths. These are not the unavailable original user's prompts.
Cache warmness is a protocol label, not measured occupancy; one short warmup is
not a claim of fully warm experts. Unavailable structured acceptance histograms
are `null`, not inferred from draft totals. A failed completion is never retried.

Local client tests (no ROCm required):

```sh
cmake -S . -B build -DCORE_WITH_HIP=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Or without local CMake: `python3 -B -m unittest discover -s tests -p baseline_test.py`.

## R2a: Q4 KV transforms and QSA selection boundaries

`kv.hpp` freezes canonical Q4_0 cache packing: the first maximum-absolute
element determines signed scale, codes use the original FP32 reciprocal,
and the FP16 scale is rounded to nearest-even. A zero block has scale `-0`
and sixteen `0x88` bytes, matching mx. Nonfinite inputs and unrepresentable
nonzero scales fail explicitly. `hip/kv.cuh` exposes allocation-free,
stream-ordered device-pointer launches; the caller owns buffer lifetimes and
checks the sticky quantization error flag before consuming invalid results.

The normalized Hadamard scales before its ascending butterfly, preserving
mx's wave64 register/shuffle order. Use Q/K256 after RoPE, V64 before cache
storage, and inverse V64 on attention output before gating. `core-kv` checks
exact CPU/GPU transform, packing and dequantization bytes on both cards,
including first-max sign ties, negative zero, FP16 subnormal scales, random
FP32 blocks, invalid inputs and valid reuse after rejection.

```sh
docker run --rm --name core-kv --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-kv \
  -v /home/radneon/gfx906-core:/core llama.cpp-gfx906:cmake-4.4.3 \
  > /home/radneon/gfx906-core/runs/r2-kv.jsonl \
  2> /home/radneon/gfx906-core/runs/r2-kv.err
docker run --rm --entrypoint /core/build/qsa-test \
  -v /home/radneon/gfx906-core:/core llama.cpp-gfx906:cmake-4.4.3 \
  > /home/radneon/gfx906-core/runs/r2-qsa.jsonl
python3 -B /home/radneon/gfx906-core/src/tools/record_kv.py \
  --kv-log /home/radneon/gfx906-core/runs/r2-kv.jsonl \
  --qsa-log /home/radneon/gfx906-core/runs/r2-qsa.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

These are synthetic resident primitive benchmarks, not attention or full
inference speed. The test-only mx serial pack / cooperative / serial A/B/A
uses identical rotated inputs and checks all three packed outputs. Cooperative
packing won on the measured N=1/2/3/128 forms; end-to-end qualification remains.

`qsa_select` takes exactly the fully visible block scores, selects whole blocks
and appends the actual tail; it returns counts rather than filled capacity.
Equal scores use lower block ID first (HF/PyTorch does not specify tied IDs).
Transformers reference is pinned to `a005fc82babfe8871d87746decad2dbee100a125`.
At 2052 visible tokens the reference selects 2048 valid IDs. A test-only CPU
emulation of production mx's padded HIP bitonic/expanded-position path selects
2051, including three valid tokens from the rejected block. This is concrete
source-derived ID divergence, not a baseline GPU/logits measurement. It marks
a semantic correction separately from optimization. R2 still needs numeric
pooled-key state and GPU selection/attention; HC/PLE CPU fixtures are below.

## R2b: GDN convolution, recurrent state and verification prefixes

`dense.hpp` is a checked scalar FP32 projection/embedding oracle for all eight
loaded GGUF types, including canonical Q5_0 and Q6_K. It reads unchanged packed
weights; it does **not** replace the production Q8 activation arithmetic.
`gdn.hpp` implements causal raw-input convolution, SiLU, Q/K L2 normalization,
beta/decay, FP32 recurrence, RMSNorm and **sigmoid** output gating. Its independent
double oracle includes HF grouped-head to GGUF tiled-head permutation tests.

The shared CPU/GPU recurrent ABI is `[V-head][V-component][K-component]`, with
K contiguous. Raw convolution history is `[qkv-feature][age]`, oldest first.
GGUF V head `h` uses Q/K head `h % 16`: mx's converter already reordered all
V-side tensors. Stored `ssm_a` is `-exp(HF A_log)`, not a log to exponentiate again.
Snapshots are chronological: slot n means n consumed inputs; verifying one
pending input plus two drafts restores slot `1 + accepted_drafts`.

`hip/gdn.cuh` borrows explicit-stream device buffers, without allocation or
host synchronization. Decode adapts furnace CPW2; chunks adapt its resident
16-column wave64/LDS slab with the same external state layout. Sticky error bits
are 1 for invalid values and 2 for nonfinite arithmetic. A final conditional
publication preserves the entire active chunk state/output on numeric failure;
scratch and speculative prefixes must not be consumed on error. CPU overflow
instead preserves the failing token and any earlier successful chunk tokens.

```sh
docker run --rm --name core-gdn --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-gdn \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/r2-gdn.jsonl \
  2> /home/radneon/gfx906-core/runs/r2-gdn.err
python3 -B /home/radneon/gfx906-core/src/tools/record_gdn.py \
  --raw /home/radneon/gfx906-core/runs/r2-gdn.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

The fixture loads layer0 projection/conv/control/norm weights, projects eight
synthetic hidden vectors with the CPU FP32 oracle, then checks both GPUs at
N=1/2/3/128 (the eight projections repeat for N128). It stops before `out_proj`;
this is not GPU projection qualification or full-network inference. Fixed
elementwise gates are output `2e-4 + 2e-4*|ref|`, state `2e-5 + 2e-4*|ref|`;
raw-history bytes are exact. It checks chronological prefixes, accept0/1/2,
two repeated rejection windows, split chunks, numeric/geometry failure atomicity
and valid reuse. CPU dense/GDN tests also pass ASan/UBSan.

Dispatch A/B/A compares validated CPW2 decode steps / resident chunk / decode
steps on identical inputs and zero state. Completed HIP events include reset
and the resident transaction, excluding projections, transfers and readback.
Resident chunks won on measured N2/3/128; numbers and raw provenance are in
`results.jsonl`, not an end-to-end PP/TG claim. Remote Release/CTest: 10/10.

## R2c: HC and PLE semantic actual-weight fixtures

`hc.hpp` retains token-major `[token][branch][feature]` residuals. It implements
branch-local RMS with direct stored gamma, low-rank SiLU/sigmoid mixing, injection
`2*sigmoid(projection/4)`, and combination with the **original** residual. The root
head collapses without injection or an extra norm; the MTP Tap remains the full
widened residual before collapse. Actual block0 attention/FFN and root weights
pass N1/2/3 sampled independent FP32-rounded HF equations and exact chunk prefixes.

`ple.hpp` reads exact multiplier bits/head ranges/EOS from metadata. Lookup reads
only selected canonical rows, preserving all sixteen logical keep1 heads. Its
layer returns an injection to add **once before attention HC**, with branch-local
RMS, signed-root sigmoid gating and four-tap depthwise convolution of dilation 3.
History holds nine prior normalized rows. EOS changes the hash segment, **not**
convolution history. Hash and convolution checkpoints are chronological; verify3
restores both at `1 + accepted_drafts`.

HF uses signed `torch.remainder`, whereas mx uses unsigned modulo. The selected
GGUF multipliers prove every valid-token product <= INT64_MAX; XOR cannot then
set the sign bit. Thus these hash formulas are equivalent for this variant's
entire declared vocabulary/history, despite differing on synthetic negative
hashes. This is **not a discovered baseline hash bug**. PLE EOS248044 remains
distinct from tokenizer EOS248046. Actual twelve-step fixtures exercise the
oldest dilation tap, EOS, a 10+2 chunk split and accept0/1/2 continuation.

```sh
docker run --rm --name core-hc-ple --entrypoint /bin/sh \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 -c \
  '/core/build/hc-test --model /models/qwen38-keep1-Q4_0.gguf && /core/build/ple-test --model /models/qwen38-keep1-Q4_0.gguf' \
  > /home/radneon/gfx906-core/runs/r2-hc-ple.jsonl \
  2> /home/radneon/gfx906-core/runs/r2-hc-ple.err
python3 -B /home/radneon/gfx906-core/src/tools/record_hc_ple.py \
  --raw /home/radneon/gfx906-core/runs/r2-hc-ple.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

This slice is CPU raw-FP32 semantic qualification, **not** GPU projections or
inference throughput. Gates remain HC `4e-6*(1+|reference|)` and PLE
`3e-6*(1+|reference|)`; sequence/split/restore outputs are bitwise exact. Hot calls
allocate no memory. Strict remote CTest, local ASan/UBSan and actual-weight
results are retained; GPU projections/HC/PLE launchers require separate gates.
