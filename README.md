# Qwen3.8-Flash-Next core for 2×gfx906

Standalone C++20/HIP core under implementation. `core-session` now executes the
own 48-layer model with Q4_0 K/V on both gfx906 GPUs. The 32-token teacher-forced
logits gate and final build/reset/generation pass. Prefill/MTP/serving remain in progress.
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
  > /home/radneon/gfx906-core/runs/r2-qsa-boundaries.jsonl
python3 -B /home/radneon/gfx906-core/src/tools/record_kv.py \
  --kv-log /home/radneon/gfx906-core/runs/r2-kv.jsonl \
  --qsa-log /home/radneon/gfx906-core/runs/r2-qsa-boundaries.jsonl \
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
a semantic correction separately from optimization. Numeric pooled-key state
and GPU selection/attention are qualified in R2d below.

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

## R2d: numerical index append, whole-block selection and Q4 sparse attention

`qsa_index.hpp` and `hip/qsa_index.cuh` retain only completed FP32 pooled keys
and at most three Q4-roundtripped raw keys. New keys are mean-pooled, direct-gamma
RMS-normalized and split-half rotated at block start `4*b`; queries rotate at
`visible-1`. Loaded text RoPE has dimension 64, base `1e7`, scale 1 and **INT32**
sections `[11,11,10,0]`. Append publishes only new keys and the final tail after
the sticky error is zero. Caller-owned logical length/checkpoint validity and
stream ordering must follow publication; small tail snapshots restore `1+a`
consumed inputs without copying pooled history. CPU guards reject stale prefixes.

`hip/qsa_select.cuh` adapts furnace radix selection to unique score/ID keys,
512 whole blocks and the actual tail. Ties choose lower block ID, including
signed zero. Selection is exact against the CPU selector on **identical scores**;
parallel score reductions may legitimately change cutoff IDs.

`attention.hpp` provides a checked common-gather CPU oracle and direct-FP32
diagnostic. `hip/attention.cuh` adapts furnace Q24/KV2/D256, `h/12` GQA, 64-key
chunks and FP32 stable merge. Only selected Q4 rows are gathered into RNE FP16;
the fixed workspace is 5,042,372 bytes (~4.81 MiB/query), not a full FP16 history.
Cache scales/IDs/query validate before use; sticky errors prevent publication.
Hadamard, projection, output gate and query RoPE are separate operations.

```sh
docker run --rm --name core-qsa --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-qsa \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/r2-qsa-gpu.jsonl \
  2> /home/radneon/gfx906-core/runs/r2-qsa-gpu.err
python3 -B /home/radneon/gfx906-core/src/tools/record_qsa.py \
  --raw /home/radneon/gfx906-core/runs/r2-qsa-gpu.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

The fixture uses actual layer3 indexer weights with eight CPU raw-FP32 synthetic
projections repeated in a prepared cache timeline. Attention uses synthetic Q4
data. Both GPUs pass append N1/2/3/128, every tail phase, chronological prefixes,
repeated rejection/restore, causal visibility through 131072, cutoff ties,
subnormals, exact counts/canaries, future/unselected poisoning and error reuse.
Fixed key/score/attention gates remain `2e-4 + 2e-4*abs(CPU reference)`;
Q4 bytes and raw-tail bytes are exact. Remote CTest16/16 and CPU ASan/UBSan pass.

Each resident metric is the mean of 20 individually completed HIP-event intervals,
validated after each repetition. Projection, transfers, reset/restore, flag clear
and validation are excluded. These are **not** full QSA, occupied-128K prompt,
inference PP/TG or speedup results. The measured attention cost (~1.1 ms at 2048
keys) remains a bottleneck to diagnose. `results.jsonl` preserves numerical scope
and provenance. CPU boundary and GPU logs now have distinct names; the earlier
CPU diagnostic log was overwritten by a filename collision, deterministically
replayed with every recorded row matching, and explicitly marked as recovered.

## R2e: reusable quantized projection and GPU HC/PLE primitives

`hip/linear.cuh` consumes unchanged canonical Q4_0/Q4_1/Q5_0/Q8_0/Q6_K
weights and the qualified 36-byte raw-sum Q8_1 activation ABI. N1/2/3 share
weight registers and two output rows; dimensions are currently bounded by
16384. Q5 retains the `-16*s8` correction, Q8 uses FP32 scale products and
Q6 keeps MMVQ four-element/two-quarter integer-subscale grouping. The independent
`linear_reference.hpp` scalar oracle checks these expressions, not a raw-FP32
dequantized dot. Weight/input scales must be validated before this borrowed-buffer
linear launch; the producer's sticky quantization error must be checked in order.

`hip/blocks.cuh` provides direct-gamma group RMS, HC SiLU/sigmoid mixing/injection,
original-residual combination, PLE signed-root gating and dilation3/conv4 history.
Parallel RMS/dot ordering has bounded error; identical normalized convolution
inputs give exact chronological history. Prefix slot `1+a`, rejected publication,
repeated accept0 and keep-mask behavior are checked. Numeric errors keep the entire
active PLE history unchanged; other outputs are provisional until error completion.

```sh
docker run --rm --name core-linear --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-linear \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/r2-linear-final.jsonl \
  2> /home/radneon/gfx906-core/runs/r2-linear-final.err
docker run --rm --name core-blocks --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-blocks \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/r2-blocks.jsonl \
  2> /home/radneon/gfx906-core/runs/r2-blocks.err
python3 -B /home/radneon/gfx906-core/src/tools/record_helpers.py \
  --linear /home/radneon/gfx906-core/runs/r2-linear-final.jsonl \
  --blocks /home/radneon/gfx906-core/runs/r2-blocks.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Check both process exit statuses before collecting; do not run concurrently.
Both GPUs pass 110 common-Q8 matrix cases, 123 quantization cases and all five
types, including signed-code extrema, odd rows, block boundaries, width16384 and
up to nine unchanged actual rows per type. Blocks use **same synthetic projections**
with actual gamma/F16 convolution, not actual matrix multiplication. Their N1/2/3/128
fixtures validate finite/alias/canary/error/reuse, root widened-tap preservation and
paired prefix history. Gates remain linear `2e-4+2e-5*|reference|`, blocks
`2e-4+2e-4*|reference|`; large-finite tests report absolute error relative to their
large input magnitudes, alongside the per-element bound ratio.

Remote CTest18/18 and local common-Q8 ASan/UBSan pass. Measurements are means of
20 individually completed resident HIP-event intervals with each result validated;
quantization/projections where absent, transfers, resets, references and validation
are excluded. This is component qualification, **not** a full projected HC/PLE
block, A/B speedup or request throughput. Subsequent projection and Session
qualification are described below. Numerical evidence/provenance live in the single `results.jsonl`.
## R3a: unchanged dense projection and full actual LM head

```sh
docker run --rm --name core-dense --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-dense \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/r3-dense.jsonl \
  2> /home/radneon/gfx906-core/runs/r3-dense.err
# Check the preceding exit status before starting the next command.
docker run --rm --name core-head --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-head \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/r3-head.jsonl \
  2> /home/radneon/gfx906-core/runs/r3-head.err
python3 -B /home/radneon/gfx906-core/src/tools/record_dense.py \
  --dense /home/radneon/gfx906-core/runs/r3-dense.jsonl \
  --head /home/radneon/gfx906-core/runs/r3-head.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Run sequentially without another GPU workload or heavy compilation. The borrowed
`hip/dense.cuh` wrapper uses a caller-owned rocBLAS handle/stream and canonical
`[K,M]` weights with transpose-A, alpha1/beta0. F32 values are unchanged; BF16 is
exactly converted once into F32, not requantized. Actual alpha F32 and index-K
BF16 matrices pass full ascending raw-FP32 oracle at N1/2/3/128; synthetic cases
cover odd, non-square and dimension16384 layouts. All outputs are checked, with
poisoned old output, canaries, immutable reads and pre-enqueue rejection cases.

The quantized wrapper's sole output-dimension extension is canonical Q6_K
`output.weight [2560,248320]`. `core-head` computes all rows for N1/2/3, verifies
every output is finite and full prefixes byte-exact, and compares 24 fixed rows
(including EOS/end-of-vocabulary coordinates) against the independent common-Q8
CPU oracle. It is not a full-output CPU oracle or whole-network logits fixture.
Both GPU gates were frozen before execution: dense `2e-4+2e-4*abs(ref)`, head
`2e-4+2e-5*abs(ref)`. Each reported event interval is completed and every one of
20 repetitions validated; upload, quantization, reset, CPU oracle and readback
are outside the resident linear measurement. No A/B or inference speed claim.
Compiled provenance is ce05879/dirty1, preserved in results.jsonl; CTest19/19
and 37 recorder tests passed. Direct 48-layer Session evidence follows.

## R3b: verified own 48-layer Session

`src/session.hpp` exposes `Session::step`, `reset` and `stats`; `src/session_main.cpp`
provides the token-ID CLI. Its direct graph preserves widened 4×2560 residuals,
zero-indexed PLE layer1, 36 GDN/12 QSA layers, softmax over all 512 experts with
top10 routing, all routed contributions plus the shared expert, route weights
after down projection, and root HC followed by the full 248320-row Q6_K head.
There are no extra ordinary attention/FFN/final norms. Canonical experts stay in
RAM; each layer has 112 immutable-weight GPU slots with same-stream reader,
upload and reuse ordering. The static 24/24 split passes the full 40 KiB residual
through pinned host memory. Q4_0 K/V, FP32 pooled index with raw-Q4 roundtrip,
FP32 GDN state and PLE hash/convolution history persist across steps. The runtime
links own HIP/static libraries and rocBLAS; `llama_decode` is only in separate
`tools/oracle.cpp`.

Verified teacher IDs `[248044, 100, ..., 130]`: **all 7,946,240 finite logit pairs
have zero numerical error**, with 32/32 argmax agreement. Required intermediates
also pass unchanged gates: logits `0.02 + 0.002*abs(ref)`, intermediates
`0.002 + 0.002*abs(ref)`, and exact `hc_init`. Remote artifacts under
`/home/radneon/gfx906-core/runs/` are `r3-session32-attention-order.jsonl`, trace
directory `r3-session32-attention-order`, `r3-session32-attention-order-logits.f32.bin`
and report `r3-compare32-attention-order.json`. Independent oracle
`r3-oracle32-hf-a` uses source/image-attested production
`dcd685463d597d31f5ca759d32c94592a2740fa4` and HF reference
`a005fc82babfe8871d87746decad2dbee100a125`, with opt-in additive GDN L2 and
FP32 gathered-QSA diagnostic corrections. This proves parity with that declared
oracle, not bitwise HF, unchanged production math or baseline performance;
the default production oracle remains unchanged.

The numerical binaries were compiled as `e9f1dfe57cf8fdc0abd9ba10ab91cfbe01d9e0db`,
dirty1; a later commit must not relabel those artifacts. Canonical attention
ordering and widened internal exponent evaluation are explicit accuracy changes;
the attention A/B/A was about 2× slower than the old topology. MMVQ/MMVF A/B/A
results qualify components only. This is ordered N=1 decode, before grouped
prefill, CPU miss-worker overlap, MTP2, user sampling, HTTP and long-context
qualification. `--generate 32 --ignore-eos` is greedy diagnosis: count actual
emitted tokens, and leave the final emitted token pending. The primary sampling
series remains temperature1.0/top-p0.95/top-k20.

Reproduction on the GPU host, using fresh names in the existing `runs/` parent
(`/core` is the container mount, not controller execution):

```sh
set -C  # Refuse overwriting redirected logs; trace/logits paths must also be fresh.
docker run --rm --name core-session-teacher --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-session \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 \
  --trace /core/runs/r3-session-repro32 \
  --logits /core/runs/r3-session-repro32-logits.f32.bin \
  /models/qwen38-keep1-Q4_0.gguf 248044 $(seq 100 130) \
  > /home/radneon/gfx906-core/runs/r3-session-repro32.jsonl \
  2> /home/radneon/gfx906-core/runs/r3-session-repro32.err
python3 -B /home/radneon/gfx906-core/src/tools/compare_session.py \
  --session-log /home/radneon/gfx906-core/runs/r3-session-repro32.jsonl \
  --session-logits /home/radneon/gfx906-core/runs/r3-session-repro32-logits.f32.bin \
  --session-trace /home/radneon/gfx906-core/runs/r3-session-repro32 \
  --oracle-dir /home/radneon/gfx906-core/runs/r3-oracle32-hf-a \
  --output /home/radneon/gfx906-core/runs/r3-compare-repro32.json
docker run --rm --name core-session-reset --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-session-test \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/r3-session-reset-repro.jsonl \
  2> /home/radneon/gfx906-core/runs/r3-session-reset-repro.err
docker run --rm --name core-session-generate --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-session \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 --generate 32 --ignore-eos \
  /models/qwen38-keep1-Q4_0.gguf 248044 \
  > /home/radneon/gfx906-core/runs/r3-generation-repro.jsonl \
  2> /home/radneon/gfx906-core/runs/r3-generation-repro.err
```

Check each exit status before the next command. `core-session-test` exercises
eight teacher steps, capacity4/slots1, invalid IDs/capacity rejection and bitwise
reset replay. Final closure job `1790961285745-474` exited0: strict build and
CTest21/21, comparison31 and recorder24 cases, final reset and 32 actual greedy
output tokens passed. Temporary fixtures respect TMPDIR; CTest uses its existing
build directory. Earlier failed logs remain preserved.

Generation completed in 2985.195379 ms with model load 69364.627872 ms recorded
separately; tracing was off, one input was consumed and the final emitted token
remained pending. This short greedy full-request observation is not steady TG,
prefill, user-sampling qualification or an A/B speedup. Raw logs are
`r3-session-closure-portable-build.log`, `r3-session-reset-final.jsonl` and
`r3-generation-final.jsonl` under the remote runs directory.
`tools/record_session.py --comparison ... --generation ... --reset ...
--results /home/radneon/gfx906-core/results.jsonl` validates before append.
The canonical journal contains one R3b record after the unchanged ten previous
records. The initial accidental `src/results.jsonl` record was validated against
the canonical append, then removed; no historical measurement was replaced.
R3b is closed; the per-GPU allocation/category ledger remains a separate R3
acceptance check before grouped prefill and hybrid scheduling.
## R3c: capacity/owner memory qualification

`core-memory MODEL.gguf` constructs one trace-free Session with capacity131072/112 slots,
checks both GPU allocation categories against independent live-Buffer RAII totals,
then consumes `[248044,100]`, resets and bitwise replays them. It checks steady
allocations/payload-read counters, at least 1 GiB free per GPU, current-device/stat/logit
preservation and recovery of all owned bytes after destruction without device reset.

```sh
docker run --rm --name core-memory --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-memory \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/NEW-memory.jsonl
python3 -B /home/radneon/gfx906-core/src/tools/record_memory.py \
  --raw /home/radneon/gfx906-core/runs/NEW-memory.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```
Use a fresh raw-log name and check both exit statuses. The verified full build runs
22 CTest gates; collector tests31 and the actual eight-row fixture pass.

`Session::memory()` synchronizes both streams outside the hot path. Categories cover
every owned GPU Buffer; total/free VRAM also sees HIP/rocBLAS private allocations.
Host capacities cover reported payload buffers, not full RAM; peak RSS is observed
separately. Expert read counts are logical payload loads, not a physical-SSD syscall
trace. This fixture qualifies capacity and two consumed tokens per pass, **not occupied
128K history, MTP workspace, prefill or inference speed**. Full measurements/provenance
remain in the single `results.jsonl` and raw log.

## R4a prerequisites: exact short GDN and stable route groups

The CPW2 GDN kernel now keeps state in registers across N2/3, with the same
arithmetic as sequential N1. Both GPUs pass bitwise output, recurrent-state,
raw-history and chronological-prefix comparisons from zero and occupied states,
including restore/continuation and late-error chunk-atomic publication. N≥4
still uses the qualified resident LDS kernel.

`RouteGroups` is a constructor-allocated CPU histogram/scan/scatter: ascending
expert IDs, stable token/rank order within each group, unchanged float weight
bits and every contribution retained. Invalid inputs leave prior views unchanged;
successful calls allocate nothing. Strict CPU and ASan/UBSan tests pass
491148 checks, 87 rejected cases and 184 allocation-checked hot calls.

```sh
docker run --rm --entrypoint /core/build/routes-test \
  -v /home/radneon/gfx906-core:/core llama.cpp-gfx906:cmake-4.4.3
docker run --rm --name core-gdn-short --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-gdn \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/NEW-gdn-short.jsonl
```

Use a fresh log and no competing GPU workload/heavy compilation. Full CTest23/23
and A/B/A pass; raw logs are `r4-short-build.log`, `r4-routes.jsonl` and
`r4-gdn-short-{a1,b,a2}.jsonl`. The `r4a_short_primitives` record preserves the
compiled revision/dirty flags and exact-fixture extension. N2/3 component latency
is below both A runs; N1 win is not confirmed, and N128 is unchanged. This closes
prerequisites only; short-window Session qualification follows below. Full chunk
prefill remains unqualified, with no full-request speed claim.

## R4a: grouped N2/3 Session qualification

`SessionConfig::max_batch_tokens` defaults to one; set it explicitly to three
before calling `Session::step_batch(ids)`. The returned logits are token-major
`[N][248320]`. Every ID/window/capacity check precedes state mutation. Failed
execution requires reset; argument rejection preserves the live logits and stats.

N2/3 use the qualified multi-column projections and stable expert grouping.
One canonical triplet acquisition serves every assignment in its group; unweighted
outputs scatter to original token/rank slots before the existing rank-order fold.
The original Q8 inputs remain separate from gathered inputs and down scratch.
QSA queries retain per-query causal visibility even after all window keys are
prepared. PLE/hash/convolution and exact short GDN advance chronologically.
Explicit batch capacity3 owns larger scratch/Q8/logits, a 122880-byte handoff,
and contribution buffers; default N1 retains its original allocation sizes/counts.

```sh
set -C
docker run --rm --name core-session-batch --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-session-batch-test \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/NEW-session-batch.jsonl
python3 -B /home/radneon/gfx906-core/src/tools/record_batch.py \
  --raw /home/radneon/gfx906-core/runs/NEW-session-batch.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Check both exit statuses and use a fresh log with no competing GPU workload.
The verified fixture uses one Session, capacity40/slots1/max-batch3, five schedules
including a five-token occupied prefix, and 200 output rows. All 39,731,200 replay
logits match N1 **bitwise**; 28 invalid windows preserve logits/stat timing/ledger,
and reset/continuation pass. Positive reuse is measured on matching teacher
suffixes after initial slot contents are overwritten, not assumed cold-cache parity.
Owners/categories/peak/host capacities and 144 constructor payload reads stay steady.
Individual-buffer geometry and post-destruction recovery are not fixture claims.

Closure job `1790973650970-503` exited0: strict build, CTest24/24, recorder38 cases,
default-N1 teacher32 with 7,946,240 finite pairs/error0, and the unchanged capacity131072
owned-memory ledger. Raw proofs are `r4-session-batch.jsonl`,
`r4-session-batch-final-build.log`, `r4-memory-default.jsonl` and
`r4-default-n1-comparison.json`. Compiled provenance is `997e197…`, dirty1;
the single journal adds one `r4a_grouped_session` after thirteen unchanged records.
This closes short-window self-parity/reuse, **not large-prompt PP, independent HF
parity, MTP or an end-to-end speedup**. Canonical bounded DS4/MMQ prefill follows.

## R4b primitives: canonical Q4 DS4/MMQ qualification

`qwen-mmq-gpu` exposes borrowed-buffer, explicit-stream launchers. DS4 is a
144-byte K128 block: four original half `(scale, raw_sum)` pairs followed by
128 signed codes, stored `[k128][column]`. The packer **transposes existing Q8_1
bytes**, without requantization; it is not a cast of four interleaved 36-byte blocks.
Q4_0/Q4_1 weights remain canonical. I64/K256/four-wave tiles use J8/16/32/64;
masked K/row/column tails are owned zeros, never bytes from a neighbour slot.

After source sync/build, run on the target with no competing GPU workload:

```sh
set -eu
set -C
docker run --rm --name core-mmq --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-mmq \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/NEW-mmq.jsonl \
  2> /home/radneon/gfx906-core/runs/NEW-mmq.err
python3 -B /home/radneon/gfx906-core/src/tools/record_mmq.py \
  --raw /home/radneon/gfx906-core/runs/NEW-mmq.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Job `1790977367105-506` passed strict build/CTest25/25 and both GPUs. Each device
checks 979 matrix cases, Q8/DS4 byte identity, Q4_1 half products, signed extrema,
poisoned missing subblocks, alignment2 weights and 89 host rejections. The frozen
full-output gate is `2e-4 + 2e-5*abs(common-Q8 CPU reference)`. All 90 A1/B/A2
measurements validate 20 individually completed event intervals and full outputs:
1800 intervals total. Quantization, packing, transfers, allocation, CPU reference
and readback are outside events. A is **diagnostic sliced N≤3 linear**, not a PP
production backend. Recorder32 tests passed; raw artifacts are `r4-mmq.jsonl`,
`r4-mmq-build.log` and `r4-mmq-codegen.log`, compiled `9269138…` dirty1.

The measured result is mixed: routed down wins at N≥8, gate at N128; HC loses at
every tested N. Standalone gfx906 metadata shows wave64 and private/spills0, not
a runtime occupancy trace. The journal adds one component record after fourteen
unchanged records. This closes **Q4 primitive correctness and component A/B/A**,
not large-prompt PP or a universal speedup. Small-M column parallelism and wide
Q5/Q8/Q6 projections must be qualified before bounded full-model PP integration.
