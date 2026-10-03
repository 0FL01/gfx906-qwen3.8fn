# Qwen3.8-Flash-Next core for 2×gfx906

Standalone C++20/HIP core under implementation. `core-session` now executes the
own 48-layer model with Q4_0 K/V on both gfx906 GPUs. R3 teacher32/reset/generation
are qualified; short and logical-wide prefill pass same-Session N1 full-logit
parity, including chunks128/129/single1024 and observed expert groups over 128.
Wide-slice closure includes strict Release build/CTest28/28, sequential
Session/wide/default-memory regressions and one validated journal append.
Full R4–R8, large-prompt prefill, MTP, serving and end-to-end speed goals remain open.
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
  llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'
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

Use the explicit `sh` invocation above after transfer: `tools/build.sh` is
nonexecutable (source mode100644, observed remote0664). Direct execution caused
job `1791010696634-716` to exit126 immediately; the explicit-shell retry succeeded.

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
host synchronization. At the R2b checkpoint, decode adapted furnace CPW2 and
chunks adapted its resident 16-column wave64/LDS slab with the same external
state layout. Current N1..128 dispatch uses the exact chronological CPW recurrence
described in the short-prefill checkpoint below; the R2b measurements retain their
historical implementation and provenance. Sticky error bits
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
weights and the qualified 36-byte raw-sum Q8_1 activation ABI. At the R2e checkpoint,
N1/2/3 shared weight registers and two output rows, with dimensions bounded by
16384. The current separate `launch_quantized_linear_short` seam extends that
reuse to N1..8; the original `launch_quantized_linear` N≤3 contract is unchanged.
The later sole large-output exception is the actual Q6_K LM head (R3a below).
Q5 retains the `-16*s8` correction, Q8 uses FP32 scale products and
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
R3b is closed; the separate per-GPU allocation/category acceptance check was
subsequently closed by R3c below.
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

At this checkpoint, the CPW2 GDN kernel kept state in registers across N2/3,
with the same arithmetic as sequential N1. Both GPUs pass bitwise output, recurrent-state,
raw-history and chronological-prefix comparisons from zero and occupied states,
including restore/continuation and late-error chunk-atomic publication. N≥4
then used the qualified resident LDS kernel. Current dispatch extends the exact
chronological CPW recurrence to N1..128; the A/B/A numbers below describe the
historical N2/3 change, not a new N128 speed measurement.

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
not large-prompt PP or a universal speedup. The following slice qualifies small-M
dispatch; wide Q5/Q8/Q6 projections still precede bounded full-model PP integration.

## R4b tuning: measured small-M column microtiles

`mmq_q4_tile_j` selects **J8 for M≤640**, otherwise the original smallest
J8/16/32/64 covering N, capped at 64. This changes dispatch only: canonical
weights, original Q8/DS4 bytes, dot arithmetic, masked tails and public validation
are unchanged. `core-mmq` protocol2 records the new selection and tile counts;
`record_mmq.py` still validates historical protocol1 without weakening its schema.

Job `1790979910034-519` ran three complete fixtures sequentially: saved owned
`build/core-mmq-microtile-baseline`, candidate `build/core-mmq`, baseline again.
The outer comparison uses only each fixture's **MMQ phase B**, not its internal
diagnostic sliced-linear phases. All three fixtures pass both GPUs and the same
full-output/byte/tail/canary gates. Raw files are
`runs/r4-mmq-microtile-{a1,b,a2}.jsonl`; candidate provenance is `e7686cd…`, dirty1,
while both baselines retain `9269138…`, dirty1.

For the eight changed actual gate/HC cases at N32/128, the candidate beats both
baselines on both devices: approximately **2.1/3.1× gate** and **2.2/3.6× HC**
resident speedup. These are paired component results, not full-request or PP
speedups. Unchanged down N128 varied approximately 6–8%; no universal improvement
or causal explanation for that variation is inferred.

Final strict Release build/CTest25/25 and all 75 recorder tests pass; the log is
`runs/r4-mmq-microtile-final-build.log`. The single canonical journal adds one
`r4b_mmq_microtiles` record after fifteen unchanged records, preserving all three
validated fixtures and thirty paired MMQ coordinates. Eight changed coordinates
must beat both baselines. One-shot append metadata fixes did not change numeric
code, raw data or any gate; the append was performed only once after validation.

For a fresh candidate run, use the previous section's `core-mmq` command and
`record_mmq.py`; filenames must be new. A paired replay uses the saved baseline
entrypoint, then current `core-mmq`, then the same baseline, with one GPU workload
at a time and no heavy compilation during events. Keep revisions/protocols from
their actual binaries. Wide-format and bounded full-model PP qualification follows.

## R4b wide projections: canonical Q5/Q8/Q6 and complete LM head

`launch_mmq_linear` accepts canonical Q4_0/Q4_1/Q5_0/Q8_0/Q6_K and the unchanged
144-byte DS4 input. Q4 forwards to the existing kernel. The other formats use
specialized LDS loaders and literal gfx906 tiles; Q5 preserves the **raw-sum**
correction, Q8 uses the FP32 scale product, and Q6 combines signed integer
subscale products before FP32 conversion. All partial tiles use owned zero
storage. Borrowed-buffer, explicit-stream, alignment/range/alias contracts remain;
there is no requantization, allocation or hidden synchronization in the launcher.
The sole large-output exception is Q6_K **[K2560,M248320]**, N≤128.

On the target, after the usual source sync/build, use new log names:

```sh
set -eu
set -C
docker run --rm --name core-mmq-wide-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-mmq-wide \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/NEW-mmq-wide.jsonl \
  2> /home/radneon/gfx906-core/runs/NEW-mmq-wide.err
python3 -B /home/radneon/gfx906-core/src/tools/record_mmq_wide.py \
  --raw /home/radneon/gfx906-core/runs/NEW-mmq-wide.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Strict Release build/CTest **26/26**, recorder **33 tests**, and both GPUs passed.
Each device covers 664 matrix cases, original Q8/DS4 byte identity, signed scales,
Q5 raw-sum differences, Q6 subscales, poisoned tails, host rejections, reuse and
canaries. The unchanged gate is `2e-4 + 2e-5*abs(common-Q8 CPU reference)`.
For the head, **all output elements are checked finite**, while numerical CPU
coverage is exactly the 24 emitted rows, including 248319—not the entire head.
The 90 A1/B/A2 records contain 1800 individually completed event intervals,
validated after every repetition. Allocation, model reads, transfers, quantizing,
packing, CPU reference, reset and readback are outside the event interval.
A is diagnostic sliced N≤3 linear, **not** a production PP backend.

Both GPUs show component wins at N32/128 for the actual Q5/Q8 shared-down matrices
and full Q6 head. N1/3 MMQ loses and must not replace the qualified short-column
path. This is neither a universal speedup nor full-model PP qualification.
I128 tiles required a one-block launch-bound hint because their LDS exceeds
32 KiB; the compiler error was fixed without suppressing warnings or gates.
Standalone gfx906 codegen is wave64: Q6 has no spills; Q5 J64 and Q8 J64/J128
report VGPR spills 30/3/55 and private bytes 124/16/208. This is static codegen
evidence, not a runtime occupancy trace or an end-to-end performance claim.

Raw artifacts: `runs/r4-mmq-wide.jsonl`, `r4-mmq-wide-build-fixed.log`,
`r4-mmq-wide-codegen-metadata.log`. Compiled provenance is `996addaf…`, dirty1.
After SSH lost job539's status, the persisted build/CTest log and separate
docker-wait job541 established the GPU process exit0; no second workload was
started blindly. The canonical journal adds one `r4b_mmq_wide` record after
sixteen unchanged records. These remain component results; the bounded Session
short-prefill checkpoint follows, while large-prompt qualification remains open.

## R4: qualified short-prefill slice closed; full R4 remains open

The accepted working source schedules all nonexpert `Matrix::apply` projections
as chronological tiles of at most eight columns. Dense MMVF covers N1..8;
the dense N9..128 fallback remains available. Quantized tiles use the separate
`launch_quantized_linear_short` N1..8 seam with the same canonical weights,
36-byte Q8_1 ABI, raw-sum/half-product arithmetic and range/lifetime contracts;
the original N≤3 API still rejects N4. Shared-expert projections also tile at
eight. Routed experts group routes **once per full logical chunk**, acquire one
canonical gate/up/down triplet per group, and compute microtiles of at most eight
assignments before original-rank scatter/fold. Band16, two stages per device and
the `copy_ready`/`consumer_done` producer/reader/reuse lifetime are retained.

PLE uses its existing 512-thread gate reduction for tiles of at most eight;
hash, normalization and convolution semantics are unchanged. GDN dispatch uses
the exact chronological CPW recurrence for N1..128 with strengthened state,
prefix, continuation and late-error atomic-publication checks. This checkpoint
adds no new arithmetic, weight, precision, epsilon or acceptance-gate changes.

GPU job `1791004619377-704` produced
`runs/r4-prefill-short-expert8-ple8.jsonl`: protocol1, **92 completed records**,
one Session with capacity40/slots1/max-batch32 and tracing off. It retains all
40 sequential N1 full-vocabulary rows, then resets the same Session for eight
N4 teacher windows, one N32 teacher window, and five N1 occupied-prefix steps
followed by N17+N10. Every phase adds eight N1 continuations; teacher IDs are
`[248044,100..130]`, continuation IDs `[131..138]`. Reset retains the expert
cache, so initial cold-cache equality is not assumed.

Across four timelines: 160 output rows/80 windows, **39,731,200 finite logits**
and **29,798,400 comparisons**, zero numerical violations and zero diagnostic
bit mismatches. The frozen gate remains `0.02 + 0.002*abs(reference)`; bitwise
or argmax equality is not a newly imposed acceptance requirement. Fifteen
invalid windows preserve the full active logits span, public stats and ledger,
then continue without reset. The 116 memory observations comprise 86 serialized
snapshots plus 30 rejection observations summarized by preservation proofs:
owned categories/counts/peaks, reported host/pinned capacities and constructor
payload-read counters stay steady. Workspace geometry is only an aggregate
floor; RAII cleanup is reported, **owned-buffer release/recovery is not measured**.

Compiled provenance is `15a025d9e105b704a628d0bbca003a08cb61b334`, **dirty1**.
The raw source literal `first_end_to_end_PP_gate_current_unqualified_candidate`
is a historical driver label and is preserved verbatim. The strict collector
qualifies the successful footer as **short PP self-parity**, without relabeling
the source or claiming independent HF parity. Parent validation of
`tools/record_prefill.py::collect(actual_raw)` passed; its local tests passed
36/36 in 142.573 s. Diagnostic trace32 job700 matched all 7392 nodes, including
640 FFN probes and the full head, exactly. Default-N1 job708 retained exact
original 6912 probes and full teacher32 logits. These are distinct checks;
the short fixture itself supplies only same-Session N1 reference evidence.

Reproduce on the GPU host after the usual source sync/build, with a fresh log
name in the existing `runs/` parent and one GPU workload at a time:

```sh
set -eu
set -C
docker run --rm --name core-prefill-short-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-test \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/NEW-prefill-short.jsonl \
  2> /home/radneon/gfx906-core/runs/NEW-prefill-short.err
python3 -B /home/radneon/gfx906-core/src/tools/record_prefill.py \
  --raw /home/radneon/gfx906-core/runs/NEW-prefill-short.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Check both exit statuses. `core-prefill-test MODEL.gguf` takes no adjustable
gates; the collector validates the complete frozen protocol before appending
one record to the explicit canonical journal. Diagnostic-only tracing is a
separate `core-prefill-trace --tokens32 MODEL.gguf NEW_TRACE_DIRECTORY` run;
use a fresh directory and its emitted manifest boundaries. A completed trace
capture does not itself establish that its numerical gate passed.

**Short-slice closure completed:** broad job `1791011326789-719` exited0 after
13m50s. Strict HIP/CXX Release build and **CTest27/27** passed; CTest total was
525.55 s. Dense-MMVF validation passed on both GPUs (480 cases reported), with
exact results. Quantized-short validation covered 747 cases/device, 1494 total,
plus 42 host rejections on each device; common-Q8 max absolute error was
`4.291534423828125e-06`, max bound ratio `0.01524`. The old API still rejects N4.
`core-gdn`, `core-session-test`, `core-session-batch-test` and `core-memory`
(capacity131072/slots112) then ran sequentially, each with process exit0.
Build/test durations are not inference-performance measurements.

Closure logs are `runs/r4-shortpp-final-regression-sh.log` and
`runs/r4-shortpp-final-{gdn,reset,batch,memory}.jsonl`. Separate validation through
`record_memory.collect(actual_raw)` passed, including steady owners, minimum free
VRAM of 7,166,787,584/6,686,539,776 bytes and complete owned-byte recovery after
destruction. This remains a capacity/regression check, not occupied128K evidence.

The canonical `/home/radneon/gfx906-core/results.jsonl` received exactly one
validated `r4b_short_prefill` append, **17 → 18 records**. Both the old 17-record
byte prefix and parsed JSON were unchanged; the downloaded controller journal
has one added line and no removed lines. The new record preserves original
`15a025d9e105b704a628d0bbca003a08cb61b334`/dirtytrue provenance.

This historical short fixture exercises chunks only through 32. The following
accepted wide slice qualifies logical chunks through 1024 and observes actual
expert groups over 128. The wide MMQ kernels above retain component qualification;
neither Session fixture establishes PP/TG speedup or independent HF parity.

## R4: qualified logical-wide1024 prefill slice; full R4 remains open

`SessionRouteStats` and `Session::route_stats()` expose completed-call logical
routing diagnostics separately from the unchanged historical `SessionStats` and
serialized protocols. `last_max_expert_group_assignments` is the maximum number
of assignments to **one expert in one layer**, across all 48 layers of the last
successfully completed call. `expert_groups_gt128` cumulatively counts
call/layer/expert groups with strictly more than 128 assignments since reset.
Construction and successful reset clear both fields; only successful execution
publishes new diagnostics. Invalid arguments and execution failure preserve the
previous successful values. Execution failure still requires Session reset.

The accepted slice retains canonical projection/shared/expert microtiles of at
most eight, chronological GDN slices of at most 128, one route grouping per full
logical chunk and one canonical triplet per expert group. Band16/two stages and
`copy_ready`/`consumer_done` lifetime are retained. Model weights, loaded tensor
precision, Q4_0 K/V, arithmetic, epsilon and frozen gates are unchanged.

Successful job `1791014580540-724` exited 0 after **22m16s**. Its binaries were
compiled as `88bd3e6b24ab1dc07c556f5093533d10a91ecd80`, **dirtytrue**; a later
code/documentation commit must not retag these artifacts. Strict HIP/CXX Release
build and **CTest28/28 (595.52 s)** passed, followed sequentially by
`core-session-test`, `core-session-batch-test`, `core-prefill-wide-test` and
`core-memory`, each with process exit 0. Build still needs the explicit
`sh /core/src/tools/build.sh` invocation above (source mode100644).

Actual raw `runs/r4-prefill-wide-a.jsonl` is **3,431,452 bytes**, protocol1:
one trace-free Session, capacity1056/slots1/max-batch1024. All 1056 full-vocabulary
N1 reference rows are retained. After resets, the 1024 teacher tokens are processed as eight
N128 chunks, seven N129 chunks plus N121, and a single N1024 chunk. Each of the
four phases includes 32 N1 continuations. Teacher IDs are `[248044,100..1122]`,
continuation IDs `[1123..1154]`; later chunk128/129 calls have genuinely occupied
prior history. Reset retains the expert cache, so cold-cache equality is not
claimed.

Across all phases: **4224 output rows / 1169 windows / 1181 JSONL records**,
**1,048,903,680 finite logits / 786,677,760 comparisons**, zero numerical
violations, diagnostic bit mismatches and maximum absolute error. The frozen
gate is `0.02 + 0.002*abs(reference)`; bitwise/argmax equality is not an additional
requirement. The observed maximum group contains **1017 assignments**; the
footer sums **950 groups over 128** across phases, including **874** in the
single1024 phase. These are actual completed-call logical routing observations,
not an inference from chunk length or proof of physical 128-column kernel/repack
execution.

Fifteen atomic argument rejections preserve the full live logits span,
`SessionStats`, `SessionRouteStats`, fixture guards and memory ledger, then
continue without reset. The **1205 memory observations** comprise 1175 serialized
snapshots plus 30 rejection observations summarized by preservation proofs.
Owned categories/counts/peaks, reported host/pinned capacities and constructor
payload-read counters remain steady. Minimum observed free VRAM is
**12,709,560,320 / 12,153,815,040 bytes** on GPU0/GPU1. This checks the owned
ledger and aggregate workspace floor, not individual-buffer geometry, full RAM
or wide-fixture post-destruction owned-byte recovery; RAII cleanup is reported.

Parent validation of `tools/record_prefill_wide.py::collect(actual_raw)` passed;
the parent local suite passed **18/18 in 47.192 s**. Logs are
`runs/r4-prefill-wide-a-build.log`, `r4-prefill-wide-a-reset.jsonl`,
`r4-prefill-wide-a-batch.jsonl` and `r4-prefill-wide-a-memory.jsonl`.
Separate `record_memory.collect` validation of the actual memory log passed for
the unchanged default capacity131072/slots112 regression; it did not append a
duplicate memory record.

Canonical `/home/radneon/gfx906-core/results.jsonl` now has exactly **19 records**:
one validated new `r4b_wide_prefill`, **18 → 19**, with the previous byte prefix
and parsed history unchanged. Compiled source `88bd3e6`/dirtytrue is retained.
The downloaded journal has one added line and no removed lines; controller
validation of the downloaded actual raw log also passes unchanged.

Reproduce on the GPU host after source sync and the explicit-shell build, using
fresh names in the existing `runs/` parent and one GPU workload at a time:

```sh
set -eu
set -C
docker run --rm --name core-prefill-wide-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-wide-test \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > /home/radneon/gfx906-core/runs/NEW-prefill-wide.jsonl \
  2> /home/radneon/gfx906-core/runs/NEW-prefill-wide.err
python3 -B /home/radneon/gfx906-core/src/tools/record_prefill_wide.py \
  --raw /home/radneon/gfx906-core/runs/NEW-prefill-wide.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Check both exit statuses. `core-prefill-wide-test MODEL.gguf` takes no adjustable
gates; the collector validates the complete protocol before appending one record
to the explicit canonical ROOT journal. Existing accepted raw data need no
second append.

This closes logical-wide1024 same-Session N1 self-parity, observed group>128 and
multi-microtile/stage-reuse correctness at this scale. Full 4K/16K teacher logits
and chunk boundaries beyond the 2052-token QSA budget, full requests with 512
actual outputs and peak VRAM/performance gates remain next. Prepared
`src/prefill_long_test.cpp`, sample CLI/sampler changes in `src/session_main.cpp`,
`src/session_cli.hpp`, `tests/session_cli_test.cpp` and sampling3 are not yet
accepted/integrated and are outside this wide closure. Neither throughput,
independent HF parity, occupied128K nor MTP is qualified by this fixture.
R4–R8 and the original temperature1.0/top-p0.95/top-k20 performance goals remain
open.
