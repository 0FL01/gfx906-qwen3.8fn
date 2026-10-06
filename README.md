# Qwen3.8-Flash-Next core for 2×gfx906

Standalone C++20/HIP core under implementation. `core-session` now executes the
own 48-layer model with Q4_0 K/V on both gfx906 GPUs. R3 teacher32/reset/generation
are qualified; short, logical-wide and real4K/16K teacher fixtures pass
same-Session N1 full-logit parity and continuation with observed expert groups
over 128. The **B8 attention / explicit CLI protocol2 correctness slice is
qualified**, with the component, full-model self-parity, actual request collectors
and default regressions passing. That B8 compiled snapshot remains
`77f89c3412fff65634df8b45b08fab1b3da028a0`/dirtytrue. Job
`1791083481832-775` ended exit1 after a successful strict build/CTest36/36 and
GPU/model execution: a collector incorrectly required two host-logit owners with
CPU workers off. Correcting that source-derived memory floor, without changing
numerical gates, enabled continuation job `1791087203796-820` to complete exit0
in 21m01s on the **same already-built binaries**, including CTest36/36 and all
remaining gates. The bounded correctness slice is accepted and pushed as
`3cc594d0783974aec4cf6f4657d8e0c6864139d2`; compiled provenance is unchanged.
The subsequent **trace-off B8 A/B/A job `1791090828619-823` completed exit0 in
52m44s** against the saved a4 baseline. B measured **35.5848 PP / 10.0650 TG at4K**
and **32.1364 PP / 9.7726 TG at16K**; mean-A/B PP gains are **1.3931× / 1.4200×**,
full-request gains **1.2711× / 1.3818×**. Detailed timings and fresh commands are below.
That bounded performance-measurement slice is **accepted, closed and pushed as
`ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2`**. The qualified775 attention binary
was promoted to the **one saved baseline before831**, retaining source77f/dirtytrue.
The subsequent **R6 prerequisite run831 completed with native exit0 / 46m12s**:
strict CXX20/HIP20 Release gfx906/all-warning/`-ffp-contract=off` build,
**CTest39/39**, actual sidecar descriptors, both-GPU dense-Q4 attention, target
checkpoint/tap/restore including wide1024, and default regressions passed.
Its artifacts retain **source `ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2`/dirtytrue**;
the current831 binary is not yet a performance baseline. MCP lost the background
status; a read-only Docker daemon die event recovered the original native exit0,
without repeating the run. Evidence and fresh commands are in the R6 section below.
Canonical remote **ROOT/results.jsonl is now50 records / 2,967,969 bytes**, exactly
two appends48→50 with the old48 byte prefix/parsed history verified remotely.
The parent's actual local restore collection and exact descriptor/dense-record
matches passed, with the old48 byte/parsed history preserved and actual **+2/-0 /
diff --check PASS**. The bounded **R6 prerequisite slice is accepted and closed,
correctness only**. The immediate parent action is its normal code/docs/journal
commit/push, then trained-forward teacher-reference qualification against an
independent donor oracle, full logits/state and stochastic windows.
Full R4–R8, exact peak-VRAM qualification, MTP, serving and the
400–600PP/30–40TG speed targets remain open; those targets are not met.
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
the raw sum uses ascending XOR 1/2/4/8/16. A positive original FP32 scale may
round to stored FP16 zero while retaining its original codes and raw sum;
the long-correctness repair below restores this pinned-mx contract. Q4_0 uses
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

At the historical wide-slice closure, canonical
`/home/radneon/gfx906-core/results.jsonl` reached **19 records**:
one validated new `r4b_wide_prefill`, **18 → 19**, with the previous byte prefix
and parsed history unchanged. Compiled source `88bd3e6`/dirtytrue is retained.
That downloaded wide-slice journal had one added line and no removed lines;
controller validation of its downloaded actual raw log passed unchanged.

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
multi-microtile/stage-reuse correctness at this scale. The later real4K/16K
correctness and primary-sampling prerequisites are qualified below; they retain
their own compiled provenance and do not expand the historical wide fixture's
scope. The later 512-output request comparisons are documented below;
exact peak VRAM, full R4 and the project speed targets remain open.

## R4: real4K/16K long correctness and primary-sampling prerequisites

Exclusive job `1791030847464-742` **completed with exit 0 after 1h15m35s**.
Binaries were compiled as `2e9848d43cf9f908dc8280f81e111c0cde86f01c`,
**dirtytrue**; a future commit must not replace this artifact provenance.
The full strict **CXX20/HIP20 Release gfx906 build and warning gates** passed,
with **CTest32/32 in 605.76 s**. After building, the actual Q8 capture passed
on both devices, followed by real16K then real4K full-logit fixtures and their
remote collectors. Reset/batch regressions exited0 with passed footers;
`record_memory.collect` passed on the actual default-capacity131072/slots112
memory raw. The job ended with the GPUs idle. These durations are correctness
and build observations, not benchmark results.

### Q8_1 validation-correctness repair, separate from optimization

Earlier job729 failed at N1 offset13206. Diagnostic job733 localized the
finite attention output at layer8, actual Q8 block117: its original FP32 scale
was about `2.728e-8`, positive but rounded to stored FP16 zero. The previous
guard incorrectly rejected this legitimate block. Pinned mx
`dcd685463d597d31f5ca759d32c94592a2740fa4`, `quantize.cu::quantize_q8_1`,
keeps the codes selected using the original FP32 scale and the independent raw
sum even when the stored half scale is zero.

The CPU and GPU repairs preserve those codes/raw sums. Nonzero-input blocks whose
original FP32 scale is zero, unsafe rounded codes outside [-127,127] (including
±128), nonfinite values, and FP16 Inf/NaN headers still reject before any unsafe
float-to-int8 conversion. GPU reductions isolate each logical32 half of wave64.
This is an accuracy/validation-correctness fix: no weights, precision, epsilon
or acceptance gates changed, and no NaN clamp or replacement with an all-zero
block was introduced. Target quantization passed **791975 checks/214 rejects**;
jobs740/742 prove the **192 actual blocks / 6144 captured F32 values** on both
GPUs, alongside boundary/rejection/reuse/linear checks. The repaired capture is
`runs/r4-q8-underflow-actual-b.jsonl`. Trace-range capture is diagnostic-only;
the default tracing behavior is unchanged.

### Actual long-fixture evidence

`core-prefill-long-test MODEL.gguf ROWS` requires explicit **4096 or 16384**.
One trace-free Session has capacity `ROWS+32` (4128/16416), **112 expert slots
per layer** and max logical batch1024. It retains all `ROWS+32` sequential N1
full-vocabulary reference rows, then resets the same Session for
`canonical1024` and `occupied5_then997` (five N1 steps, then N997 chunks and a
final remainder). All three phases include **32 N1 continuations**. Teacher IDs
are a new BOS248044-then-monotone family (`id[p]=99+p` for p≥1), not original
text, the user's prompt or a baseline-parity fixture. Reset retains the expert
cache; initial cold-cache equality is not claimed. Both target K and V remain
Q4_0 with unchanged loaded GGUF values/types.

The frozen full-vocabulary gate remains `0.02 + 0.002*abs(reference)`, with
all values finite and zero allowed violations; bit equality is diagnostic
only. Both actual raws have **zero violations, maximum absolute error, maximum
bound ratio and diagnostic bit mismatches**:

| Completed evidence | Real16K | Real4K |
| --- | ---: | ---: |
| JSONL records / phases | 129 / 3 | 93 / 3 |
| Timeline rows / actual windows | 49,248 / 16,518 | 12,384 / 4,206 |
| Full-vocabulary finite values | 12,229,263,360 | 3,075,194,880 |
| Full-vocabulary compared values | 8,152,842,240 | 2,050,129,920 |
| Atomic rejects / memory observations | 8 / 16,539 | 8 / 4,227 |
| Maximum logical expert group / groups>128 | 1024 / 28,064 | 1024 / 6,800 |
| Minimum observed free VRAM, GPU0 (bytes) | 5,184,978,944 | 5,260,476,416 |
| Minimum observed free VRAM, GPU1 (bytes) | 4,704,731,136 | 4,780,228,608 |
| Diagnostic completed-call wall sum (ms), **not benchmark** | 2,679,118.940426 | 607,945.098957 |

Actual accepted rows/call offsets cover visibility2047–2056 and all mod4 tail
phases beyond the QSA budget. This is logical causal-boundary evidence, **not a
GPU selected-ID trace**. Atomic argument rejection preserves the live logits,
public stats/route stats, fixture guards and memory ledger before continuation.
Routing counts describe assignments to one expert/layer before physical tiling;
canonical compute microtiles≤8 and chronological GDN slices≤128 are retained.
The full logits use a retained same-Session N1 self-reference, **not independent
HF parity**. Steady owned ledgers/aggregate floors and observed free VRAM do not
qualify individual-buffer capacities, full RAM, wide/long owned-buffer release,
peak VRAM, physical128-column tiles, occupied128K, performance or full R4.

Logs are `runs/r4-prefill-repaired-b-build.log`,
`runs/r4-prefill-long-b-{16384,4096}.jsonl`, and
`runs/r4-prefill-repaired-{reset,batch,memory}-b.jsonl`, plus the actual Q8 raw
above. Remote prevalidation of both actual long raws and parent local
`collect(downloaded_actual_raw)` for real16K and real4K **passed**. Parent
`record_prefill_long.main` appended **exactly two records** to canonical ROOT
`/home/radneon/gfx906-core/results.jsonl`, **19→21**. The old byte prefix and
parsed19-record history are exactly preserved. The last two records have kind
`r4b_long_prefill`, teacher rows **16384 then4096**, both `passed=true` and
`R4_complete_claim=false`; their original `2e9848d…`/dirtytrue source is retained.
The downloaded canonical journal is **1,572,766 bytes** locally; the expected
Git diff **+2/-0** is verified. The accepted slice retains its compiled artifact
provenance independently of the closure commit.

Reproduce on the GPU host after source sync and the explicit-shell build, with
fresh names and one GPU workload at a time:

```sh
set -eu
set -C
ROWS=16384  # Choose exactly 4096 or 16384; each is an explicit large-RAM fixture.
docker run --rm --name core-prefill-long-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-long-test \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf "$ROWS" \
  > /home/radneon/gfx906-core/runs/NEW-prefill-long-$ROWS.jsonl \
  2> /home/radneon/gfx906-core/runs/NEW-prefill-long-$ROWS.err
# Only after process exit0; validate before one append to the explicit ROOT journal.
python3 -B /home/radneon/gfx906-core/src/tools/record_prefill_long.py \
  --raw /home/radneon/gfx906-core/runs/NEW-prefill-long-$ROWS.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Check the collector exit status; repeat with the other ROWS value using a fresh
log. Existing actual raws must not be appended a second time. The source build
script remains mode100644 and requires `sh /core/src/tools/build.sh`.

### Integrated CLI/sampling prerequisite and remaining request gate

`core-session` accepts `--prefill-chunk 1..1024` (default1) and
`--sample --seed UINT64 --temperature 1.0 --top-p 0.95 --top-k 20`.
CLI/sampler integration is covered by the strict build/CTest above. Prior
job729's actual32-output primary-sampling smoke, seed42, gave the same IDs
with chunk1 and chunk32 and consumed exactly32 RNG draws; legacy greedy32
also passed. That historical smoke qualifies sampling prerequisites only;
the later actual512 request series has its own evidence below.
Greedy remains diagnostic-only; count actual emitted tokens and retain the
final emitted token as pending. A fresh short primary-sampling reproduction is:

```sh
set -eu
set -C
docker run --rm --name core-session-sampling-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-session \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 \
  --prefill-chunk 32 --generate 32 --ignore-eos --sample --seed 42 \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  /models/qwen38-keep1-Q4_0.gguf 248044 $(seq 100 130) \
  > /home/radneon/gfx906-core/runs/NEW-primary-sampling32.jsonl \
  2> /home/radneon/gfx906-core/runs/NEW-primary-sampling32.err
```

## R4: closed actual512 chunk A/B/A measurement slice; full R4 open

Exclusive performance job `1791040898917-746` **completed with exit 0 after
1h40m26s**. Its source/binaries were frozen at
`b522429c5933f493a5838317dc0bfc26ee4158aa`, **dirtytrue**; a later closure
commit must not retag these artifacts. The strict targeted `core-session`
build passed. New request-results and VRAM-observer CTest gates passed
**2/2 in 5.18 s**, containing **49 and 23 tests** respectively. The previous
full-build job742 passed **32/32**; this targeted measurement build did not
establish a subsequent full-build result. Latest job765 independently passed
**35/35**, with its separate source and correctness scope documented below.

### Inputs, ordering and completed-output accounting

At each length, four separate processes construct **fresh Sessions** with
112 expert slots per layer, Q4_0 target K/V and no prefix reuse. The source
records are request **protocol1**, primary stochastic sampling:
temperature1.0/top-p0.95/top-k20, seed12345, `--ignore-eos`, 512 actual outputs,
MTP off. Expert-cache warmness is recorded as unknown, not fully warm.

- **4K** uses the exact 4096 IDs saved by R0's generated archive/code text
  fixture: `ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json`.
- **16K** is four concatenated copies of those IDs, saved separately in
  `ROOT/runs/r4-fullrequest-b522-16k.prompt-ids.json`. It is not retokenized
  text, the original user's prompt, or a production-baseline16K fixture.
- The performance order is **A1 chunk128 → B chunk1024 → A2 chunk128**,
  unobserved and trace-free; then a **separate B_VRAM chunk1024** request at
  each length. Observed requests do not enter the performance comparison.
- Each request emits 512 tokens, performs **511 TG forwards / 512 RNG draws**,
  and consumes **4607 inputs at4K / 16895 at16K**. The first output comes from
  prefill and the last remains pending. Within each length, all four complete
  512-ID arrays are exactly equal. This is a measured diagnostic, not a universal
  bitwise-generation contract or baseline-RNG equivalence claim.

All **8/8 actual collectors**, including both request/observer joins, passed.
Rates below are completed wall measurements: PP uses the actual prompt count;
TG uses the 511 remaining output forwards. Request timing includes sampling
and CLI I/O, with model load and cleanup separate; it is not host enqueue-time
or a sum of overlapping GPU events.

| Length / trace-free run | PP tokens/s | TG tokens/s |
| --- | ---: | ---: |
| 4K A1, chunk128 | 13.7598351 | 10.0684460 |
| 4K B, chunk1024 | 24.8487818621 | 10.2822355241 |
| 4K A2, chunk128 | 13.4884100 | 9.9265480 |
| 16K A1, chunk128 | 12.6972088 | 9.6139253 |
| 16K B, chunk1024 | 22.2115183863 | 9.9343829824 |
| 16K A2, chunk128 | 12.5701114 | 9.7421264 |

| Completed paired measurement | 4K | 16K |
| --- | ---: | ---: |
| B PP elapsed (ms) | 164837.054095 | 737635.298724 |
| B TG elapsed (ms) | 49697.363847 | 51437.51765 |
| B full-request elapsed (ms), load separate | 214535.87483 | 789074.279015 |
| Mean A1/A2 PP elapsed (ms) | 300673.048345 | 1296885.815248 |
| Mean-A PP elapsed / B PP elapsed | 1.82406225345× | 1.75816669497× |
| Mean-A full-request elapsed / B full-request elapsed | 1.63977185394× | 1.71047221484× |

These are **same-own-config chunk comparisons only**. They establish no speedup
over the external production MTP2 baseline. The **400–600PP / 30–40TG targets
are not met**, and full-project/full-R4 performance remains open. Wide MMQ
component measurements are not relabeled as this Session's production path.

Full-request expert uploads are **1,139,677,491,200 bytes** in each 4K A run
versus **305,853,440,000 bytes** in B; at16K, each A uploads
**4,264,348,979,200 bytes** versus B's **890,465,689,600 bytes**.
These counters span the whole request. Per-phase upload bytes were not emitted,
so this is neither PP-only traffic nor proof that uploads are the residual
bottleneck.

### Separate sampled driver-VRAM observations

The stdlib-only `tools/observe_vram.py` samples at **0.1 s**, before spawn,
through the attached child lifetime, and after wait. Its exact scope is
`sampled_global_driver_VRAM_not_exact_instantaneous_peak`. Parent independently
verified the actual R0 HIP mapping: **HIP0 → 0000:05:00.0**,
**HIP1 → 0000:08:00.0**; the observer/collector itself does not attest HIP indices.
Both driver totals are **17,163,091,968 bytes**.

| Separate B_VRAM request | 4K | 16K |
| --- | ---: | ---: |
| Sample rounds per device | 2892 | 8643 |
| Observed maximum used, GPU0 (bytes) | 12,004,397,056 | 12,079,910,912 |
| Observed maximum used, GPU1 (bytes) | 12,487,217,152 | 12,562,612,224 |
| Observed minimum free, GPU0 (bytes) | 5,158,694,912 | 5,083,181,056 |
| Observed minimum free, GPU1 (bytes) | 4,675,874,816 | 4,600,479,744 |

These are **sampled global driver** values, not HIP-owned allocations, exact
instantaneous peaks or per-phase peaks. Observer elapsed includes Docker
startup/load/cleanup and observation overhead; it is not the native request
benchmark. `record_request.py --vram-log` validates the complete two-device
observer and a closed, direct attached Docker argv shape against the request
source. Shell-wrapped/detached commands are outside that join. Matching argv
does not attest binary or mount identity. No dependency/license was added.

### Raw evidence, journal status and fresh reproduction

Here `ROOT=/home/radneon/gfx906-core`. Actual request logs are
`ROOT/runs/r4-fullrequest-b522-{4k,16k}-{a1,b,a2,b-vram}.jsonl`; observers are
`ROOT/runs/r4-fullrequest-b522-{4k,16k}-b-vram-vram.jsonl`.
The same series preserves `r4-fullrequest-b522-fixture-source.json`,
`r4-fullrequest-b522-16k.prompt-ids.json`, `r4-fullrequest-b522-build.log` and
`r4-fullrequest-b522-series.log` under `ROOT/runs/`.

The canonical `ROOT/results.jsonl` received **exactly nine rows, 21→30**:
eight `r4_request` plus one `r4_prefill_ab`. All eight actual raws were downloaded;
local collection and both VRAM joins passed, and the complete seeded512-ID arrays
match across all four variants within each length. The downloaded local canonical
journal at this historical closure was **2,150,750 bytes / 30 rows**. Its old21-row
byte prefix and parsed history match Git HEAD exactly; numstat is **+9/-0**. Artifact/local-collector,
journal, local49/23-test suites, readback/status/diff/log and whitespace acceptance
gates are complete. **The actual512 measurement-only slice is accepted and
closed; full R4 remains open.** The accepted slice contains docs, CMake,
observer/request tools and tests, and results; experimental Session/GPU/R5/
speculative changes are outside it. Accepted
artifacts retain frozen b522/dirtytrue. Existing accepted raws must not be
appended again.

For a fresh trace-free 4K B run on the GPU host, use the existing fixed ID file
and fresh fixed log names. This reproduction explicitly sizes capacity4608;
the recorded source configuration remains authoritative for each actual run.
Use the explicit `sh /core/src/tools/build.sh` build invocation shown above
(source mode100644) with the synchronized source's actual revision/dirty flag.

```sh
set -eu
set -C
ROOT=/home/radneon/gfx906-core
PROMPT="$ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json"
docker run --rm --name core-fullrequest-4k-b-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-session \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 \
  --capacity 4608 --slots 112 --prefill-chunk 1024 \
  --generate 512 --ignore-eos --sample --seed 12345 \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  /models/qwen38-keep1-Q4_0.gguf \
  $(python3 -B -c 'import json,sys; print(*json.load(open(sys.argv[1])))' "$PROMPT") \
  > "$ROOT/runs/NEW-fullrequest-4k-b.jsonl" \
  2> "$ROOT/runs/NEW-fullrequest-4k-b.err"
# Only after exit0; this collector appends one validated new request record.
python3 -B "$ROOT/src/tools/record_request.py" \
  --raw "$ROOT/runs/NEW-fullrequest-4k-b.jsonl" \
  --results "$ROOT/results.jsonl"
```

For paired reproduction, run the same command sequentially as A1 with
`--prefill-chunk 128` / `NEW-fullrequest-4k-a1`, then B with1024, then A2
with128 / `NEW-fullrequest-4k-a2` (both `.jsonl` and `.err` names).
For16K use the saved `r4-fullrequest-b522-16k.prompt-ids.json`, capacity16896
and fresh `NEW-fullrequest-16k-{a1,b,a2}` names; do not tokenize again.
Check every run/collector exit status, with one GPU workload and no heavy
compilation or tracing. The three 512-output performance requests are separate
from the B_VRAM observation.

For a separate B_VRAM reproduction, prefix that direct Docker command with
`python3 -B "$ROOT/src/tools/observe_vram.py" --interval 0.1 --device-0 /sys/bus/pci/devices/0000:05:00.0 --device-1 /sys/bus/pci/devices/0000:08:00.0 --output "$ROOT/runs/NEW-fullrequest-4k-b-vram-vram.jsonl" --`,
retaining chunk1024 and using fresh child stdout/stderr names
`NEW-fullrequest-4k-b-vram.jsonl` / `NEW-fullrequest-4k-b-vram.err`.
After observer and child exit0, collect that new request with
`--vram-log "$ROOT/runs/NEW-fullrequest-4k-b-vram-vram.jsonl"` and the explicit
canonical `--results "$ROOT/results.jsonl"`. Keep Docker as the direct attached
argv after `--`; the observer requires a fresh output file.

### Current measured work and open gates

Parent rocprofv3 job `1791052186288-750` **completed exit0 in 2m14s**, using
the same frozen `b522429c5933f493a5838317dc0bfc26ee4158aa`/dirtytrue binary.
The diagnostic request used a **1024-token R0 text prefix / 32 outputs / 31 TG
forwards** and passed natively: PP **43116.686375 ms**, TG **4488.729555 ms**,
model load **70708.105268 ms**. These traced timings are diagnostic, not
performance qualification or a replacement for the trace-free actual512 series.

The profile aggregates **PP and TG together**. HIP/API, GPU and memory-DMA
intervals overlap; their duration sums are **not total request latency**, are
not additive across categories, and do not isolate PP-only costs.

| Profile category / operation | Calls or events | Summed duration (s) | Share of summed GPU duration |
| --- | ---: | ---: | ---: |
| HIP/API total | 2,760,730 | 36.11848 | — |
| `hipMemcpyAsync` API | 1,062,432 | 30.0312 | — |
| HIP launch API | 839,656 | 5.11047 | — |
| GPU total | 1,831,186 | 28.37233 | — |
| `fattn_dec_chunk` | 12,660 | 9.362744 | 33% |
| `Q4_0matrix32<2,8,2>` | 193,474 | 5.847864 | 20.61% |
| `copyBuffer` | 989,831 | 3.537115 | 12.47% |
| Q6 head `matrix6<8,2>` | 896 | 0.712313 | 2.51% |
| Memory DMA | 72,646 | 5.562979 | — |

Memory-DMA events are distinct from the nearly million small GPU D2D
`copyBuffer` operations. Neither the aggregate duration sums nor full-request
upload bytes prove a RAM/expert-DMA/memcpy residual bottleneck. The Q6 head is
not dominant in this diagnostic; head skipping is not the next major-win target.

The source-accounted narrow seam is **per-assignment GPU Q8 gather/down scatter**:
for1024 PP rows, `2 * (1024 * 10 * 48) = 983040` small D2D `hipMemcpyAsync`
calls. The qualified route-copy implementation replaces it with **one 80*N-byte
route DTO upload per layer** and indexed gather/scatter kernels per canonical microtile **≤8**,
preserving all copied bits and the existing expert-contribution fold order.
Job765 has now passed both-GPU route-copy and full-model hybrid correctness
gates. The completed trace-free comparison below shows a modest route-copy
PP/request gain, not dominance inferred from the roughly 83% memcpy share of
summed HIP/API durations. The CPU-linear scheduler remains opt-in
(`cpu_workers=0` by default); hybrid performance and admission thresholds remain open.

Raw profile stdout/stderr are `ROOT/runs/r4-profile-b522-1024.jsonl` / `.err`;
trace/stats CSVs remain under `ROOT/runs/r4-profile-b522-1024/`, with the nested
summary at `prefill_/core/runs/r4-profile-b522-1024-summary.txt.txt` relative to
that directory. The full **554 MB trace remains remote and was not downloaded**;
no model was copied for this profiling/documentation update.

### Accepted trace-off GPU-copy full-request measurement slice

Exclusive job **`1791073403122-769` completed exit0 in 57m33s**; GPUs were idle
at completion. It used the **already-built job765** candidate
`/core/build/core-session`, source
`a4b55d84724ba15bbae7013d5a107b7671b7a409`/dirtytrue, against saved
`/core/build/core-session-baseline`, source
`b522429c5933f493a5838317dc0bfc26ee4158aa`/dirtytrue. No rebuild or mirror of
future attention edits entered this series; these compiled macro strings remain
authoritative, independently of parent HEAD `6faf14ed96dc1b0d4353dc838323616a6f6c5313`.

For each **4K and 16K** length, the sequence is **A1 baseline → B candidate →
A2 baseline**, with **chunk1024 / slots112 on both binaries**, a fresh Session
per request and no prefix reuse. Primary sampling is temperature1.0/top-p0.95/
top-k20/seed12345 with ignoreEOS, MTP off and expert-cache warmness unknown.
All six requests emitted **512 actual outputs each / 3072 total**;
each performed 511 TG forwards / 512 RNG draws, with consumed4607/capacity4608
at4K and consumed16895/capacity16896 at16K. The three 512-ID arrays within each
length were identical, diagnostic-only. Inputs remain the exact archived R0
generated archive/code 4K IDs at the path above and four concatenated copies
for16K, not retokenized or original-user prompts. This is a same-own non-MTP
binary comparison, distinct from both the historical chunk comparison and an
external MTP2 baseline comparison.

| Length / B765, chunk1024 | PP tokens/s | TG tokens/s | Full request (ms) | Mean-A / B PP elapsed | Mean-A / B request elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 25.68871522181 | 10.07616889732 | 210162.620717 | 1.02718318141× | 1.01937816968× |
| 16K | 22.60272136899 | 9.85092544783 | 776743.230123 | 1.01999903571× | 1.01916904667× |

B is faster than **both A1 and A2 for PP and full request at each length**.
This is a modest 2–2.7% PP/about 1.9% request gain, not a large win or a qualified
TG/CPU-dispatch threshold. B full-request upload counters remain
305,853,440,000 bytes at4K / 890,465,689,600 bytes at16K, equal to the older B
configuration; they do not isolate PP traffic or establish bottleneck dominance.
Raw stdout/stderr are `ROOT/runs/r4-indexed-a4-{4k,16k}-{a1,b,a2}.jsonl` / `.err`;
series status and provenance are `ROOT/runs/r4-indexed-a4-series.log` and
`ROOT/runs/r4-indexed-a4-fixture-source.json`.

Remote canonical journal advanced **31→38**, exactly six `r4_request` records
and one `r4_indexed_route_copy_ab` (**2,497,537 bytes**), preserving the old byte
and parsed prefix. The first paired append failed on a missing model field
before writing, after six valid request appends. Parent re-collected the existing
37 records (timestamps excluded) and appended only the corrected paired record;
no duplicate or history rewrite occurred. Parent downloaded the canonical journal,
all six raw requests and fixture metadata. Current `record_request.collect`
**passed on all six locally**; exact source/footer/output counts/timings/throughput
match their remote journal entries, and the three 512-ID arrays per length match.
The local 38-record journal preserves both the **Git HEAD31 byte prefix and
parsed31-record history exactly**; actual git numstat **+7/-0** and diff whitespace
checks passed. **The measurement-only slice is accepted and closed**; do not
append these six requests or their paired record again. Full R4/R5 remains open.

At that measurement closure, local request tests **79/79 in 5.055s** and
attention-collector tests **28/28 in 43.004s** covered the CLI/protocol2/collector
contracts. The subsequent actual GPU/model/CLI qualification is recorded below.

The769 artifacts retain their historical b522→a4 binary identities. Fresh B8
paired reproduction follows the current-baseline recipe below, using saved exact
IDs/capacities, chunk1024 on all three runs and fresh stdout/stderr names.
Preserve each actual binary's revision/dirty flags. The769 series is complete.
During823, the qualified765 a4 binary was the **one saved baseline**,
`/core/build/core-session-baseline` (**1,556,208 bytes**), source
`a4b55d84724ba15bbae7013d5a107b7671b7a409`/dirtytrue. The completed B8 series below
used this binary against the already-built job775 candidate
`/core/build/core-session`, source77f/dirtytrue. Neither binary is retagged to a
future commit.

## R4: qualified B8 attention and explicit CLI2 correctness slice

### Strict build, collector repair and completed continuation

The correctness work used source snapshot
`77f89c3412fff65634df8b45b08fab1b3da028a0`. GPU artifacts retain that compiled
snapshot with **dirtytrue**; the accepted/pushed correctness closure is
`3cc594d0783974aec4cf6f4657d8e0c6864139d2`.
Job **`1791083481832-775` terminated exit1 in 26m29s**, after the full strict
**CXX20/HIP20 gfx906 Release/all-warning-gates build**, **CTest36/36 in 803.76s**,
both-GPU attention component PASS and original Model B8 executable **exit0**.
The failure was in the collector: it required two host-logit owners for
`cpu_workers=0`, whereas this Session owns only **`host_logits`, 1,017,118,720
bytes**. `working_logits` is allocated only with `cpu_workers>0`.

The parent corrected the **source-derived lower floor to one owner**, with no
numerical tolerance change, and added a regression: local **29/29 in 43.187s**.
Corrected actual B8 collection passed both locally and remotely. Continuation
job **`1791087203796-820` completed exit0 in 21m01s**, reusing the **same775
binaries**: full **CTest36/36 in 808.07s** with corrected collector tests, actual
B8 collection, CLI2 mixed/off32, default reset/batch/memory and strict actual
collection all passed. The separate default `core-memory` capacity131072/slots112
collector passed; this is not an occupied128K test or an additional journal append.

### Both-GPU attention component and actual Model B8

`core-attention-batch` is a model-free primitive fixture. Its successful raw has
**three JSON records: source plus one correctness record per GPU**, with **no
completion footer**. Each GPU passed **26 cases / 304 queries / 1,867,776
bit-compared values and 1,867,776 CPU-compared values**, maxabs
**1.1920928955078125e-7**, max-bound-ratio **0.0003692344547586807**, **23 device
rejects / 180 host rejects / 6 sticky checks**. The frozen component gate is
`2e-4 + 2e-4*abs(ref)`; parity with the old N1 primitive is bit-exact on these cases.

The new batch primitive **ADAPTs the current furnace-derived N1 path**, adding
an independent private query dimension while retaining exact old Q4→half RNE
gather, sorted selected IDs, 64-key split and merge arithmetic. Its typed borrowed
API checks capacities/strides/overlaps, uses the explicit stream and a shared
sticky flag, and suppresses all output publication on a detected query failure.
It adds no dependency; existing mx/furnace MIT and reinstinct Apache-2.0 pins apply.

The original `core-prefill-attention-test MODEL` completed **exit0 / 31 records**.
It uses **one same Session**, capacity2088/slots1/max1024/query-tile8/CPU workers0:

- Old N1 reference consumes all2088 rows: **2056 teacher + 32 continuation**.
- Enabled B8 consumes teacher chunks **1024/1024/8**, then32 N1 continuations.
- Occupied-prefix phase consumes **5 N1**, chunks **997/997/57**, then32 N1
  continuations.

Totals are **2163 completed calls / 6264 rows / 1,555,476,480 finite values /
1,036,984,320 full-vocabulary comparisons**. Numerical errors, violations and
diagnostic bit differences are **zero**; **4176 diagnostic argmax matches**.
The gate remains **`.02 + .002*abs(ref)`**; bit/argmax identity is not required.
Logical boundary coverage2047–2056 follows the accepted source rows, not a
GPU-observed visibility/selected-ID trace or an independent HF oracle.

Attention counters total **6180 batch API calls / 49,284 query rows / 6168
multiquery calls / 49,272 multiquery rows / 12 singleton-tail calls / max8**.
These count successfully completed **API invocations**, not physical kernels.
There are **2190 in-process memory observations / 26 serialized ledgers / 8
atomic rejects / 6 toggles / 6,073,162,240 preserved values**; individual toggle
states and the remaining observations are driver assertions, not additional raw
ledgers. Minimum observed free GPU0/1 is **12,671,549,440 / 12,115,804,160 bytes**.

Reported private workspace per GPU is **40,338,944 bytes**: four independent
buffers **40,142,336 bytes** plus **196,608 bytes reused from the f(15) prefix**,
not another allocation. Selection is separate, **82,080 bytes**. Actual f(15)
staging and f(17) output backing capacities are **50,331,648 bytes each**, already
in the workspace/owned ledger. This bounded self-reference does not qualify new
4K/16K B8 execution, performance, exact peak VRAM or measured owned-buffer release.

### Actual explicit CLI2 mixed/off32

Both requests used capacity64/slots1/chunk32, prompt **`[248044,100..130]`**,
generate32/ignoreEOS/sample/seed42 and primary **temperature1.0/top-p.95/top-k20**.
Mixed used **CPU workers1 / mixed / GPU-miss quota2 / attention tile8**; off used
**CPU workers0 / disabled / tile1**. Each emitted **32 actual outputs**, consumed63
rows, completed31 TG forwards, used32 RNG draws and one PP call. Output IDs were
identical, diagnostic-only. Protocol2 emitted the explicit execution knobs and
**26 hybrid + 2 route + 6 attention typed counters**; strict actual request
collection passed **locally and remotely**.

Mixed counters: **1488 short / 48 GPU-only-wide layers**, **11,255 CPU groups /
11,255 gate-up jobs / 11,255 down jobs / 11,255 middle columns / 1488 middle
batches**; paired H2D **57,625,600 bytes**, Q8 D2H **8,103,600 bytes**, CPU returns
**115,251,200 bytes**. GPU-hit groups649 / GPU-miss groups2976 / admitted2976 /
evicted1440. Diagnostic short TG wall times were **27789.104387 ms mixed versus
6521.876872 ms off**. This is not a proper paired performance series or a speed-gain
claim; CPU workers remain opt-in, not a promoted default or measured policy.
Raw **`candidate_unqualified` / `local_unqualified`** labels remain literal.
This documentation records the bounded qualification above without rewriting
Source records or claiming qualification for every exposed knob value.

### Raw evidence, journal and fresh reproduction

With `ROOT=/home/radneon/gfx906-core`, original raws are
`ROOT/runs/r4-attention-77f-a-{build.log,component.jsonl,model.jsonl}`; continuation
raws are `ROOT/runs/r4-attention-77f-b-{gates.log,cli2-mixed.jsonl,cli2-off.jsonl,reset.jsonl,batch.jsonl,memory.jsonl}`.
The parent downloaded the actual model/component/CLI2 raws and strict local
collection passed. At this correctness closure, canonical `ROOT/results.jsonl`
reached **41 records /
2,563,568 bytes**, **exactly three appends38→41**: one `r4_attention_prefill`, one
protocol2 mixed `r4_request`, one protocol2 off `r4_request`. The old38-record
byte prefix and parsed history are **exactly preserved**; source77f/dirtytrue is
unchanged. Parent downloaded the canonical journal and verified the exact old Git
HEAD38 byte prefix/parsed history and **+3/-0** diff. This bounded correctness slice
is accepted. **Never append these accepted logs again.**

For a fresh component/model reproduction on the GPU host, use the build contract
above with actual synchronized revision/dirty and explicit
`sh /core/src/tools/build.sh` (source mode100644). Run the following sequentially;
the model executable is a manual model gate, separate from CTest:

```sh
set -eu
set -C
ROOT=/home/radneon/gfx906-core
docker run --rm --name core-attention-batch-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-attention-batch \
  -v /home/radneon/gfx906-core:/core llama.cpp-gfx906:cmake-4.4.3 \
  > "$ROOT/runs/NEW-attention-component.jsonl" \
  2> "$ROOT/runs/NEW-attention-component.err"
docker run --rm --name core-prefill-attention-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-attention-test \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > "$ROOT/runs/NEW-attention-model.jsonl" \
  2> "$ROOT/runs/NEW-attention-model.err"
# Only after executable exit0; append ONE validated fresh model record to ROOT.
python3 -B "$ROOT/src/tools/record_prefill_attention.py" \
  --raw "$ROOT/runs/NEW-attention-model.jsonl" --results "$ROOT/results.jsonl"
```

For a fresh CLI2 mixed smoke, use the same Docker/device/mount contract with
entrypoint `/core/build/core-session` and arguments
`--capacity 64 --slots 1 --prefill-chunk 32 --generate 32 --ignore-eos --sample --seed 42 --temperature 1.0 --top-p 0.95 --top-k 20 --cpu-workers 1 --hybrid-mode mixed --gpu-miss-groups 2 --attention-query-tile 8 /models/qwen38-keep1-Q4_0.gguf 248044 $(seq 100 130)`.
For off use workers0/disabled/tile1; choose fresh stdout/stderr names for each.
Only after each executable exit0 collect with
`record_request.py --raw FRESH --results "$ROOT/results.jsonl"`.

### Accepted and closed trace-off B8 full-request A/B/A measurement slice

Exclusive job **`1791090828619-823` completed exit0 / 52m44s**, using the saved
job765 A binary above and **already-built job775 B**, without rebuild, heavy
compilation or another GPU workload. A retains source
`a4b55d84724ba15bbae7013d5a107b7671b7a409`/dirtytrue; B retains
`77f89c3412fff65634df8b45b08fab1b3da028a0`/dirtytrue. A uses the original
N1 attention/tile1; B explicitly uses **query tile8 / workers0 / hybrid disabled**.
B emits protocol2 with literal **`candidate_unqualified` / `local_unqualified`**
labels. These labels and both compiled Source records remain unchanged by this
measurement or a future closure commit.

Each length ran **A1→B→A2**, chunk1024/slots112 on both, fresh Session/no prefix
reuse/cache warmness unknown/MTP off. Sampling stayed **temperature1.0/top-p.95/
top-k20/seed12345/ignoreEOS**. All six emitted **512 actual outputs each / 3072
total**, with **511 TG forwards / 512 RNG draws per request**; consumed4607 at
capacity4608 for4K, consumed16895 at capacity16896 for16K. All three512-ID arrays
within each length match, **diagnostic-only**. Inputs are the saved exact R0
archive/code4K IDs and four concatenations for16K, not original-user prompts,
retokenized text or an external MTP baseline comparison.

| Length / run | PP elapsed (ms) | TG elapsed (ms) | Full request (ms) | PP tokens/s | TG tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K A1 | 160152.153592 | 50828.622536 | 210982.242626 | 25.5756785540 | 10.0533906784 |
| 4K B | 115105.478965 | 50769.941198 | 165876.889925 | 35.5847526706 | 10.0650106725 |
| 4K A2 | 160544.563734 | 50172.369423 | 210718.406950 | 25.5131653463 | 10.1848887321 |
| 16K A1 | 724478.209838 | 53140.681118 | 777620.365975 | 22.6148968699 | 9.6159851407 |
| 16K B | 509826.542585 | 52289.067381 | 562117.081524 | 32.1364201968 | 9.7725973247 |
| 16K A2 | 723419.610449 | 52437.016154 | 775858.095619 | 22.6479898573 | 9.7450243641 |

| Length | Mean-A PP (ms) | Mean-A request (ms) | Mean-A / B PP | Mean-A / B request |
| --- | ---: | ---: | ---: | ---: |
| 4K | 160348.358663 | 210850.324788 | 1.39305583109347× | 1.2711253802945932× |
| 16K | 723948.9101435 | 776739.230797 | 1.419990623620387× | 1.3818104027209437× |

**B is faster than both As in PP and full request at each length.** TG remains
about10 tokens/s; 400–600PP/30–40TG targets and full R4/R5–R8/final gates remain
open and unmet. Timings are completed wall measurements: PP excludes output
sampling, TG counts actual forwards (`outputs−1`), load is separate. This series
adds no GPU-event/profile/VRAM/exact-peak, independent-HF, marginal-RNG or CPU
dispatch qualification; model weights, precision, numerical gates and Q4 K/V
remain unchanged.

Raw stdout/stderr are **`ROOT/runs/r4-attention-77f-ab-{4k,16k}-{a1,b,a2}.jsonl`
/ `.err`**; series log and provenance are
`ROOT/runs/r4-attention-77f-ab-series.log` and
`ROOT/runs/r4-attention-77f-ab-fixture-source.json`.
Canonical **ROOT/results.jsonl advanced41→48 / 2,897,341 bytes**, exactly six
`r4_request` appends and one `r4_attention_batched_query_ab`; the paired record
retains source77f/dirtytrue and each request its own compiled source.
The old41-record byte prefix and parsed history were **verified remotely**;
all six strict remote collections passed. Parent downloaded all six raws,
fixture metadata and the canonical48-record journal. **All six actual local
collections passed**, with Source/completion/rates exactly matching the recorded
ROOT rows and three512-ID arrays per length identical, diagnostic-only. The
downloaded journal preserves the old **Git HEAD41 exact byte prefix and parsed
history**; actual **git diff numstat +7/-0 and diff --check passed**.
The bounded **performance-measurement slice is accepted, closed and pushed as
`ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2`**; its scope was the four docs and
journal, excluding the then-future prerequisite code. **Never append these seven
records again.** Before831, qualified775 was promoted to the **one saved baseline**:
`/core/build/core-session-baseline`, **1,750,328 bytes**, source77f/dirtytrue;
future A runs explicitly use query tile8 / workers0 / hybrid disabled.
The saved path no longer contains historical823 A/sourcea4. The current
`/core/build/core-session` is the831 sourceba446f9/dirtytrue build, correctness
qualified below but not performance-qualified or promoted. These changes do not
retag either the historical823 artifacts or the saved775 baseline.

Fresh tile8 request reproduction on the GPU host (choose new filenames before
every run). With current831 this is a new candidate run, not a replay of775 timings:

```sh
set -eu
set -C
ROOT=/home/radneon/gfx906-core
PROMPT="$ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json"
docker run --rm --name core-attention-ab-4k-b-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-session \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 \
  --capacity 4608 --slots 112 --prefill-chunk 1024 \
  --generate 512 --ignore-eos --sample --seed 12345 \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --cpu-workers 0 --hybrid-mode disabled --attention-query-tile 8 \
  /models/qwen38-keep1-Q4_0.gguf \
  $(python3 -B -c 'import json,sys; print(*json.load(open(sys.argv[1])))' "$PROMPT") \
  > "$ROOT/runs/NEW-attention-ab-4k-b.jsonl" \
  2> "$ROOT/runs/NEW-attention-ab-4k-b.err"
# Only after exit0; append this ONE fresh request, never an accepted historical raw.
python3 -B "$ROOT/src/tools/record_request.py" \
  --raw "$ROOT/runs/NEW-attention-ab-4k-b.jsonl" --results "$ROOT/results.jsonl"
```

Historical823 **A1 and A2** used the then-saved sourcea4 binary with the three
new flags `--cpu-workers`, `--hybrid-mode`, `--attention-query-tile` omitted.
That binary has been replaced at the saved path; use the archived823 raws for
that historical comparison. For new **A1 and A2**, use entrypoint
`/core/build/core-session-baseline` with explicit workers0 / hybrid disabled /
query tile8 and distinct `NEW-attention-ab-4k-a1` / `NEW-attention-ab-4k-a2`
container/stdout/stderr names; keep chunk1024/slots112.
For16K use prompt `ROOT/runs/r4-fullrequest-b522-16k.prompt-ids.json`, capacity16896
and fresh `NEW-attention-ab-16k-{a1,b,a2}` filenames. Run A1→B→A2 sequentially at
each length and collect each fresh raw only after executable exit0. Fixed raw
filenames identify the recorded variant; verify actual arguments, sampling,
prompt IDs and revision/dirty against **Source**, not the filename or current
Git HEAD. Preserve the literal raw labels. A rebuilt future binary must report
its actual source, not borrow823 provenance. Keep tracing off and one GPU workload
without heavy compilation.

New paired measurements use the qualified77f baseline with explicit tile8 and a
new candidate, checking each actual Source. Attention tile1/OFF and the original
API remain defaults. The R6 prerequisites below are **accepted and closed,
correctness only**, with actual remote/local and history checks passing; the
parent's normal prerequisite commit/push is next. Trained MTP, measured R5 policy,
remaining PP traffic, occupied long/API, converter and final speed goals stay open.

## R5: qualified CPU-linear/canonical-GPU-middle correctness slice

Remote **job765 completed exit0 / 34m46s**; the later exclusive job769 comparison
also completed, with its separate measurement scope above.
The full strict **CXX20/HIP20 Release gfx906** build passed all warning gates and
**CTest35/35 in 735.71s**. Both-GPU indexed route-copy and paired SiLU/Q8 fixtures,
the full hybrid fixture, default reset/batch/logical-wide1024/default-capacity
memory regressions and strict actual-log collectors all passed. Compiled source
is the **snapshot `a4b55d84724ba15bbae7013d5a107b7671b7a409`, dirtytrue**,
not a later documentation/closure commit.

### Execution method and the rejected whole-CPU path

The qualified method is **CPUlinear_GPUmiddle**: exact CPU gate/up projections,
**canonical GPU SiLU×up and Q8_1 quantization**, then exact CPU down projections.
Middle work is aggregated into **one batch per CPU-bearing layer**, with one
column per CPU assignment. `SessionHybridMode::force_cpu` retains its API name
but is labeled **CPU_LINEAR_GPU_middle**, not whole-CPU execution. Default
`cpu_workers=0` retains the historical GPU-only path and allocations.

The original host-libm whole-expert path remains a component oracle, **not a
qualified Session path**. Historical job755 failed the frozen full-model gate
at token100/offset1 (logits maxabs .223724, routed-down .018279). Job759's saved
actual **layer46/expert297** witness had identical CPU/GPU gate/up projections,
but 24 middle FP32 values differed at ULP scale. Q8 block11 kept identical codes
and scale while its **raw-sum half bits differed: 13734 versus 13733**. Primitive
bounds passed, yet the propagated full-model gate failed. Using the canonical
GPU nonlinear producer fixes this boundary without changing the quantizer ABI,
GPU math, weights, precision, epsilon or gates; **no tolerance waiver** was used.

The original bounded orchestration uses a persistent **1..15-worker CPU pool**,
fixed capacity **30 jobs**, and two stages (`gate_up`, `down`). Gate/up temporarily
uses the existing pinned down-output frame; it is reused for down only after
**`middle_ready` completes**, protecting its H2D source. Four dedicated GPU
middle buffers and a dedicated flag on `copy_stream` avoid racing the compute
stream's shared Q8/scratch/error state. Only CPU-dispatched rank contributions
are uploaded; `cpu_returned` is recorded **before** the compute-stream wait,
followed by original-rank fold and shared-expert addition. Captured cache readers,
pending-ID publication, admission and error/reset lifetimes are correctness
qualified. The submitted-admission failure is **synthetic**, not proof that CPU
or GPU work was still in flight at the throw or that useful overlap improves speed.

### Actual fixture evidence and observation limits

`core-session-hybrid-test MODEL` protocol1 uses **three sequential Session owners,
one simultaneous**, with max batches `[3,1,4]` and workers `[1,1,4]`. A retained
GPU-N1 reference covers BOS248044,100..130 plus eight continuations131..138.
The **11 phases / 326 compared windows** include hybrid-off N1/2/3,
force-CPU-linear/GPU-middle N1/2/3, mixed slots1, forced GPU misses, warm all-hit,
synthetic failure/reset replay, and N4 GPU fallback followed by short hybrid resume.

| Compared output | Rows | Values | Violations / maxabs / max-bound-ratio / diagnostic bit differences |
| --- | ---: | ---: | --- |
| Full vocabulary logits | 528 | 131,112,960 | All zero |
| All ten unweighted routed-down contributions at all48 layers | 508 | 624,230,400 | All zero |
| FFN output including shared expert at all48 layers | 508 | 62,423,040 | All zero |

Every compared operand is finite. Frozen gates remain **`.02+.002*abs(ref)`**
for logits and **`.002+.002*abs(ref)`** for intermediates. N4's20 rows have no
intermediate probe. Bit identity is a measured diagnostic, not a universal
CPU/GPU reduction or generation requirement. This is same-runtime GPU-N1
self-parity, not an independent HF oracle or a PP/TG benchmark.

Accepted compared-window deltas total **106,984 gate/up jobs and 106,984 down
jobs**, **137,581 GPU middle columns / 10,416 batches**, **704,414,720 H2D bytes**
for paired gate/up and **99,058,320 D2H bytes** for middle Q8. Reference execution
and the failed call are excluded. These are projection/transfer counters,
not request time or performance evidence.

The collector's **175 atomic-rejection checks** are **source-derived required
driver checks**, not serialized per-rejection observations. Protocol1 does not
serialize before/after historical stats, route-stats/physical-routing snapshots,
original input-probe payloads, steady per-category owned ledgers, constructor
payload reads or complete owned-buffer release. Do not reconstruct these as
additional measured fields. The model filename is a declared variant alias;
the actual model path is not emitted by this protocol.

Opt-in middle storage is **four independent buffers per GPU / 252,004 bytes**:
paired gate/up 153600, middleFP32 76800, middleQ8 21600, flag 4. These enter workspace
and the existing Buffer ledger; pinned middleQ8 capacity is **43,200 bytes total**.
`host_input_probe_bytes=855648` is **included** in total
`host_probe_bytes=33295968`, not added again. Reported capacities/metadata exclude
allocator slack, worker stacks and full RSS. Individual added-buffer capacities
are observable, but the hybrid protocol does not measure every old owned buffer's
release across all owners.

Default GPU-only regressions also passed: logical-wide1024 retained N1 reference
at **capacity1056 / slots1 / max1024**, with **786,677,760 comparisons and zero
error**. This is not a new4K/16K run. Default memory **capacity131072 / slots112**
passed both GPUs: owned **9,515,698,008 / 10,007,105,368 bytes**, free
**7,166,787,584 / 6,686,539,776 bytes**, and after destruction free
**16,968,876,032 bytes on each GPU**. This separate capacity/release fixture
does not establish occupied128K history or hybrid-protocol all-owner release.

### Raw evidence, canonical journal and fresh reproduction

With `ROOT=/home/radneon/gfx906-core`, actual logs are:

- `ROOT/runs/r5-middle-c-build.log`
- `ROOT/runs/r5-silu-pairs-c.jsonl`
- `ROOT/runs/r5-route-copy-c.jsonl`
- `ROOT/runs/r5-session-hybrid-middle-c.jsonl` (**348 records / 272,019 bytes**)
- `ROOT/runs/r5-middle-default-{reset,batch,wide,memory}-c.jsonl`

At this historical closure, canonical **ROOT/results.jsonl received one append,
30→31**, kind
`r5_hybrid`, passed=true, retaining snapshot a4b55d8/dirtytrue and the previous
30-record byte prefix/parsed history. **Never append this accepted raw again.**
Parent local `collect(downloaded_actual_r5-session-hybrid-middle-c)` **passed**
on the **272,019-byte** raw, with all compared metrics zero. The downloaded
canonical journal at that closure was **31 records / 2,167,805 bytes**; its exact
old Git HEAD 30-record byte prefix and parsed history are preserved, and **+1/-0 is verified**.
The bounded correctness slice is **accepted**. Its closure scope is Session/CPU/
ops, relevant fixtures, CMake, collector, these docs and journal; the three
post765 attention files and three future Spec files are excluded. Correctness
artifact source identities remain a4b55d8/dirtytrue, not a future closure hash.

For a **fresh** fixture on the GPU host, first use the explicit
`sh /core/src/tools/build.sh` build command above (script source mode100644),
with the source actually synchronized for that new run. Then:

```sh
set -eu
set -C
ROOT=/home/radneon/gfx906-core
docker run --rm --name core-hybrid-middle-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-session-hybrid-test \
  -v /home/radneon/gfx906-core:/core \
  -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  > "$ROOT/runs/NEW-session-hybrid-middle.jsonl" \
  2> "$ROOT/runs/NEW-session-hybrid-middle.err"
# Only after executable exit0; append ONE validated fresh record to ROOT only.
python3 -B "$ROOT/src/tools/record_hybrid.py" \
  --raw "$ROOT/runs/NEW-session-hybrid-middle.jsonl" \
  --results "$ROOT/results.jsonl"
```

This accepted correctness slice does not close fullR5, hybrid performance,
a measured dispatch threshold or MTP. The completed exclusive trace-off
comparison **job769** measured a modest GPU-copy PP/request gain using saved
baselineb522 versus already-built job765; its measurement slice is accepted
with local actual-log and exact history/+7/-0 proofs above. The qualified a4
binary was the saved baseline for823. B8/CLI2 correctness was accepted/pushed as
3cc594d after jobs775/820; separate trace-off B8 job823 is complete with the PP
and request gains above. Parent local six-raw collection/exact journal matches,
Git HEAD41 byte/parsed-prefix proof and +7/-0/diff --check passed; the bounded
performance-measurement slice is accepted, closed and pushed as ba446f9.
Qualified775/source77fdirtytrue is now the one saved baseline, promoted before831;
the current831/sourceba446f9dirtytrue build passed the bounded R6 prerequisite
gates below, now **accepted and closed, correctness only**, after parent actual
local validation/history/+2/-0 checks. Parent normal commit/push is immediate,
then trained-forward teacher-reference qualification proceeds. CPU-worker/admission-policy and PP weight
traffic measurements remain open; the earlier measurement-only commit provides
no trained MTP runtime proof.
Full R4/R5 performance, occupied128K, MTP2 and the **400–600PP / 30–40TG** targets
remain open and unmet.

## R6 prerequisites: actual descriptors, dense Q4 attention and target restore

### Native completion and strict gates

Run **`1791103589280-831`** used synchronized source
**`ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2`/dirtytrue**, not a future closure
commit. MCP reported **`state_lost` / `background_channel_closed_without_exit_status`
after14m15s**. The parent recovered the **original native exit0 without rerunning**
from a read-only Docker daemon **container die event** for
`core-r6-prerequisites-check`, ID
`35ed6fbda1d6c386cb1d84819addcaf4650460044619ba658867de52e73a0644`:
`exitCode=0`, `execDuration=2772s` (**46m12s**), `time=1791106362`,
`timeNano=1791106362809343377`. The downloaded ignored proof is
`ROOT/runs/r6-prerequisites-ba-a-exit-proof.json` (**719 bytes**).

The full **CXX20/HIP20 Release gfx906/all-warning/`-ffp-contract=off` build** passed
**CTest39/39 in1145.49s**. New `MtpModel` tests passed **666 checks / 451 rejects**;
speculative pure math passed **35,963,782 checks / 715 rejects / 5,522,274
allocation-checked hot calls with zero heap allocations**. Pure math does not
execute trained MTP. Subsequent actual descriptor, both-GPU dense component,
wide1024 target restore and default reset/batch/B8 attention/memory processes
all exited0; strict actual collection passed. Default capacity131072/slots112 is
an allocation/regression gate, not occupied128K history.

### Actual descriptor and model-free dense attention

`core-mtp-model TARGET SIDECAR` produced **36 records / 14,462 bytes** in
`ROOT/runs/r6-mtp-model-ba-a.jsonl`: all **32 role-typed descriptors**, checked
offsets/strides and actual inventory **Q8_0×19 / F32×11 / BF16×2**. BF16 is allowed
only for indexer Q/K. The checks preserve full top10 expert budget and confirm
blk.48 compression0/dense attention, full typed equality of all10 tokenizer
assets, and distinct target-owned **Q4_0 embedding / Q6_K output** views.
Descriptor-derived sidecar payload is **2,775,621,632 bytes**; gate/up/down expert
stride is **1,740,800 bytes each**. This CLI reads metadata/descriptors, not tensor
payload or a trained forward.

`core-mtp-attention` produced **4 records / 1,298 bytes** in
`ROOT/runs/r6-mtp-attention-ba-a.jsonl`, including both GPUs and a passing cleanup
footer with **live_resources=0**. Dense full-causal-prefix Q4 K/V covers B1..3 and
synthetic128K, old-N1 short bit parity, finite/source/bounds checks, future poison,
rejections, sticky publication and canaries. Each GPU passed **63 cases / 128
queries**, maxabs **2.9522925615310669e-6**, max-bound-ratio
**0.014517076073887787**, at frozen **`2e-4 + 2e-4*abs(ref)`**. Donor decision:
**KEEP** mx sum64, **ADAPT** the current furnace-derived64-key split/merge,
**WRITE OURS** bounded direct canonical Q4→FP16 RNE reads. This synthetic component
is not occupied full-model128K, trained MTP, independent HF or performance proof.

### Actual target checkpoint/tap/restore, including wide1024

`core-session-restore-test MODEL --wide-1024` produced protocol1 **56 records /
473,730 bytes** in `ROOT/runs/r6-target-restore-ba-a.jsonl`. Three owners execute
sequentially, one live at a time: primary cap40/max-batch3/CPU0, wide cap2048/
max-batch1024/CPU0, and CPU1 synthetic failure/reset recovery. Across all owners,
**644,390,400 full-vocabulary comparisons / 2595 rows** and **26,419,200 exact tap
values** passed: **maxerror/max-bound-ratio/diagnostic bit differences all zero**,
with the unchanged logit gate **`.02 + .002*abs(ref)`**. Aggregate phase totals are
**51 successful restores / 84 argument rejections / 137 steady-memory observations /
6 sticky rejections**; the footer's primary-only counts are50/83/133.

The45 primary windows cover N1..3 with **every retained prefix0..N**, all mod4
phases, EOS plus PLE hash/convolution, QSA tail, divergent suffixes and old-published
span/tap preservation. Wide PP checks **all1024 taps**. The target tap is the
**widened10240 residual before ROOT_HC**, published transactionally: invalid calls
and execution failure preserve the old publication; execution failure makes
checkpoint/restore unavailable until reset. Restore rewinds logical consumed-input
state and recurrent/conv/PLE/QSA-tail prefixes; it does **not rewind expert cache
slots, uploads or physical execution counters**, and does not copy full KV history.

Checkpoint/tap storage is opt-in, **default OFF**, with **87 independent GPU
Buffers**. Batch3 adds **236,851,200 bytes on GPU0 / 235,622,400 on GPU1**;
batch1024 keeps GPU0 at236,851,200 and adds **319,262,720 on GPU1**. Wide GPU1 has
two transactional tap buffers of **41,943,040 bytes each**. This is target
consumed-input self-parity/state qualification, not trained-sidecar rollback,
independent HF, occupied128K or throughput evidence.

### Raw evidence, journal boundary and fresh reproduction

With `ROOT=/home/radneon/gfx906-core`, the build/status log is
`ROOT/runs/r6-prerequisites-ba-a-build.log`; default raws are
`ROOT/runs/r6-default-{reset,batch,attention,memory}-ba-a.jsonl`.
Canonical **ROOT/results.jsonl alone** received exactly two appends **48→50 /
2,967,969 bytes**: `r6_target_restore` and `r6_mtp_prerequisites`. The old48-record
byte prefix and parsed history are exact remotely. The parent downloaded all
three new actual raws, exit proof and canonical journal. Parent read-only local
`record_restore.collect` passed again on the downloaded **473,730-byte** actual raw:
**644,390,400 comparisons / 26,419,200 exact tap values**, all errors zero.
The actual C++ descriptor36-record and dense4-record raws' Source/footer/inventory/
device fields **exactly match the new ROOT record**. Both the **Git HEAD old48
parsed history and byte prefix are exact against local50**; actual **git numstat
+2/-0 / diff --check PASS**. The bounded **R6 prerequisite slice is ACCEPTED/CLOSED,
correctness only; R6_complete_claim=false**. Parent normal code/docs/journal
commit/push is the immediate remaining action. Never duplicate-append the recorded
raws or retag their native Source.

Fresh reproductions run sequentially on the GPU host after the existing explicit
`sh /core/src/tools/build.sh` contract with the actual synchronized revision/dirty:

```sh
set -eu
set -C
ROOT=/home/radneon/gfx906-core
docker run --rm --name core-mtp-model-repro \
  --entrypoint /core/build/core-mtp-model \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf \
  /models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  > "$ROOT/runs/NEW-mtp-model.jsonl" 2> "$ROOT/runs/NEW-mtp-model.err"
docker run --rm --name core-mtp-attention-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-mtp-attention \
  -v /home/radneon/gfx906-core:/core llama.cpp-gfx906:cmake-4.4.3 \
  > "$ROOT/runs/NEW-mtp-attention.jsonl" 2> "$ROOT/runs/NEW-mtp-attention.err"
docker run --rm --name core-session-restore-repro \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --entrypoint /core/build/core-session-restore-test \
  -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro \
  llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf --wide-1024 \
  > "$ROOT/runs/NEW-target-restore.jsonl" 2> "$ROOT/runs/NEW-target-restore.err"
# Only after executable exit0; ONE strict fresh restore append to ROOT only.
python3 -B "$ROOT/src/tools/record_restore.py" \
  --raw "$ROOT/runs/NEW-target-restore.jsonl" --results "$ROOT/results.jsonl"
```

**Next:** parent normal prerequisite code/docs/journal commit/push, then apply/review
the trained-forward candidate and qualify teacher history against an actual
independent donor oracle, full logits/state and stochastic windows. Source-pinned
`qwen4exp.cpp:438..450` requires the **first Hnorm per2560 branch with distinct
4×2560 gamma**; treating it as one10240 reduction is rejected. HC mixer normalization
over the whole10240 remains unchanged. Ignored
`runs/r6-trained-mtp-forward.patch` (**73,983 bytes**) and
`runs/r4-layerwise-prefill.patch` (**61,985 bytes**) are both **unapplied/unqualified**
and carry no speed promise. Full R4 speed, R5 measured policy, R6 trained MTP,
R7 occupied long/tokenizer/API and R8 pack/final gates remain mandatory and open;
latest performance stays the accepted823 measurements above.

## R6 trained forward: sequential teacher and batching diagnostics (2026-10-04)

The candidate `MtpSession` executes the real Q8_0 blk48 sidecar, borrowing the
already-loaded target Q4_0 embedding and Q6_K output without duplicating them.
It supports N1..3, recursive widened-D proposals, full/explicit KV-only teacher
history, logical suffix restore, owned carry across target verification, and
transactional logits/tap/probe publication. All 512 sidecar experts are loaded
once on device1 (2,673,868,800 bytes). This is not a complete speculative sampler
or a performance-qualified MTP2 serving path.

The standalone `tools/mtp_teacher_oracle.cpp` links the existing source/image-
attested mx dcd685463d production libraries; it never enters the core hot path.
Its default uses the unmodified donor. `--dense-f32-control` instead evaluates
mathematical dense attention over the donor's actual Q4 cache decoded through
FP16 RNE, with unchanged query/mask/weights. That historical control is preserved.
A NEW, separate `--dense-short-canonical` flag requires the dense control and
is bounded to <=64 visible rows. It uses exp(double) rounded to F32 and the
rounded reciprocal/product form of the existing short GPU attention. This is
an arithmetic diagnostic, NOT a tolerance waiver, raw-donor parity or HF proof.
The independently expressed double CPU self-test checks both variants. Strict
standalone build and self-test passed 1,247,970 checks, 45 rejections and
1,222,656 comparisons (max bound ratio 0.000883514). The production libraries
and GPU attention implementation were not modified by this diagnostic.

Actual source snapshot: d2f948ae44f5172dd2395fdbf1723e0b1ddd6237, dirty=true.
The 32-row generated teacher is BOS248044,100..130. Source capture:
`ROOT/runs/r6-mtp-position26-d2-capture`; ROOT=/home/radneon/gfx906-core.
Do not retag these artifacts to a later closure commit.

Observed controls, all at frozen D atol=.002 / logits atol=.02 / rtol=.002:
- Original dense-FP32 control: numerical FAIL at position26; D3978 violations,
  logits68892 violations. Preserved under `r6-mtp-dense-f32-d2-oracle-n1`.
- Short-canonical sequential donor: native exit0, all32 rows PASS,
  327680 D values maxabs3.8146973e-6/maxratio.000205267;
  7946240 logits maxabs.020387292/maxratio.961748919, zero violations.
  Native job1791129410074-50, `r6-mtp-canonical-diagnostic-n1`.
- Short-canonical donor N2: D PASS; logits FAIL8216/maxratio2.599409.
  N3 also returns numerical_failure. These results MUST NOT be hidden by the N1 pass.
- DonorN1 vs donorN2 at position4: all captured inputs through FFN-HC are
  bit-identical. FFN differs by1.7762184e-5, widenedD by3.0159950e-5,
  head input by.026559830 and logits by.060904741. Thus the donor itself has
  observable batch-dependent amplification; this does not establish any wider
  accuracy claim. The diagnostic N2 capture is `r6-mtp-canonical-diagnostic-n2-sites`.
- The own-runtime chronological fixture checks N1/N2/N3 and ordinary/reset,
  KV-only prefix29, suffix poison/restore/replay and publication failure/reuse.
  Fresh job1791130233817-73 completed native exit0 after20m51s. Strict build
  and CTest39/39 passed (1149.99s); fresh teacher-width3 capture again has zero
  self error. Tokens, previous hidden, own D and all logits are byte-identical
  to the earlier N1 capture. Actual fresh arrays were re-compared to the frozen
  sequential donor arrays, with the same passing metrics above.
  Canonical ROOT/results.jsonl advanced50→51 with one bounded record; the
  entire previous50-record byte prefix was preserved on target and controller.

Sequential teacher agreement and own batching invariance are separate evidence
from a passing donor-batched comparison. Full R6 remains open: complete sampled
MTP2 windows, acceptance0/1/2, terminal/pending handling, broader/longer references
and matched end-to-end speed are not proved by this bounded32-row test.

Reproduce the diagnostic using the existing image/device/mount contract:
`tools/build-mtp-oracle.sh` requires the exact pinned donor tree and production
library revision and a NEW output named mtp-teacher-oracle. Its mandatory
`--self-test` must pass before atomic publication. Invoke that binary with
`MAIN SIDECAR CAPTURE_DIR NEW_OUTPUT_DIR --teacher-width 1 --dense-f32-control
--dense-short-canonical`; add `--capture-intermediates` only for diagnosis.
Never reuse an existing output directory. Use explicit /bin/sh for multi-step
commands reached through ssh: amude's login shell is not a POSIX-sh guarantee.

### R6 trained stochastic-window coordinator fixture (2026-10-04)

`core-mtp-window-test TARGET SIDECAR` is a runnable test-only coordinator, not
an enabled production CLI mode. It couples real MtpSession proposals to the
existing filtered SpeculativeSampler, target verify windows, teacher-history
rebuild and independent target(relative)/draft(absolute) prefix restoration.
The last emitted token remains pending and is excluded from both consumed
histories. The trained block is recursively reused for the second proposal.
No weights, precision, top-k, sampler formula or numerical gate was changed.

Actual source961150d3bc155025ec7e1dfa78eb4db47405fbf5/dirtytrue; strict target
build and job1791132994108-105 nativeexit0 in180s. Raw artifacts:
ROOT/runs/r6-window-961-live-build.log, -live.jsonl, -live.err.
Six replay-checked cases use generated8-token prompt BOS248044,100..106,
capacity48/cache112, temp1/top-p.95/top-k20 and explicit seeds. Budgets1/2/3
exercise the initial pending draw and reduced horizons; budgets32/41 exercise
many stochastic windows and the actual capacity boundary48; budget16 enables
EOS handling but no actual EOS occurred. No model-level EOS coverage is claimed.

Across55 replay-checked windows, acceptance histogram[0,1,2]=[33,10,12].
22,100,480 retained target logits,13,409,280 draft continuation logits and
13,409,280 ACTUAL restored-target continuation logits match independent
same-runtime sequential replay with maxabs0. The continuation check runs
BEFORE resetting owners, so it exercises the restored GDN/PLE/QSA state rather
than only comparing cached pre-restore verify output. Invisible suffixes are
subsequently overwritten by normal processing. This is state self-consistency,
not another independent model-reference qualification.

A seventh case executes32 outputs without any intermediate reset/replay.
Its entire output ID vector, acceptance histogram[10,6,3] and RNG draw counts
(proposal38,decision46) match the corresponding checked case exactly. The file
serializes the equality assertion/result and counters, not all output vectors.
The speculative proposal/initial stream uses seed12345; correction stream uses
seed XOR0x9e3779b97f4a7c15. No baseline-RNG-stream or throughput equality is claimed.

The initial simpler run97(nativeexit0) is retained as journal52. Strengthened
continuation/live run105 is journal53; the prior51-record byte-prefix history was
preserved and both records copied back to the controller. Intermediate
restore-only job102(nativeexit0) is raw diagnosis, not a duplicate journal append.
All timings include diagnostic work/load; they are NOT MTP throughput evidence.
Next: extract this tested orchestration into an opt-in reusable serving/CLI path,
add real token-ID EOS fixtures/longer histories, then trace-off matched MTP-off/on
requests. FullR6 and performance targets remain OPEN.

### Opt-in MTP runner and token-ID CLI (2026-10-04)

`MtpRunner` owns one target and its borrowing sidecar, preallocates sampler
workspaces, warms target/teacher history, and emits terminal-aware speculative
windows without resetting or replaying the full history. Invalid begin requests
are rejected before reset/RNG mutation; execution failure requires a fresh begin.
`core-mtp-run` exposes this path separately, leaving core-session unchanged:

```
core-mtp-run TARGET SIDECAR --sample --seed 12345 --generate 512 \
  --capacity 1024 --prefill-chunk 8 --attention-query-tile 8 --ignore-eos \
  248044 100 101 102 103 104 105 106
```

This is token-ID I/O only. GPU-only mode, explicit sampling seed and positive
output budget are required. CPU hybrid, tracing and full-logit export are rejected
before model load. Target verification reserves at least3 rows even if PPchunk1.
The prompt can use chunks up to1024; MTP teacher updates consume these taps in
N<=3 windows, retaining the previous target carry across chunk boundaries.
Target and sidecar cache/state/pending cursors are checked after every emission.
Structured output reports actual tokens, acceptance, RNG counts and completed
host-wall timings. PP includes target+teacher warmup; model load is separate.
Timing diagnostics are NOT automatically performance qualification.

Source f59c8dee226225fd743b14ec525116d7a6e40382/dirtytrue:
- Strict standalone CLI/API builds PASS. Help and7 malformed requests reject
  before model loading (including missing seed, CPU/trace-family unsupported
  mode, capacity/token/output overflow and unknown option).
- CLI32: native job1791133953326-136 exit0; outputs32/consumed39,
  acceptance[10,6,3], RNG38/46 match the checked coordinator.
- API job1791134154542-142 exit0: full32 output IDs identical for PPchunk1/8;
  repeated reset, active-state/RNG preservation after invalid begin, capacity48,
  initial/replacement custom stop-token handling and EOS-vs-budget precedence PASS.
  Those two stop-token tests use configured IDs; actual modelEOS248046 was NOT observed.
- Full strict build and CTest39/39 PASS1150.55s; job1791134454670-146
  nativeexit0/22m11s, including actual512 output smoke, consumed519, pending353.
  Windows213/acceptance[44,40,129], RNG426/594. Load77836.55ms excluded;
  completed request41560.08ms, PP957.607ms, decode40600.53ms/TG12.5860.
  The prompt is only8 generated tokens: NOT a4K/16K PP result or a speedup claim.

Raw prefix ROOT/runs/r6-run-f59: smoke32, api, preflight, full-regression and
smoke512 artifacts. Canonical journal54 records full actual output ID arrays,
source/completion data, API/preflight results and limits; previous53-record bytes
are preserved exactly on remote and controller. No independent long-model/HF,
actual modelEOS, tokenizer/HTTP or matched MTP-off/on speed claim. Next: matched
4K/16K full requests with512 outputs; retain the existing qualified baseline.

## Matched MTP off/on/off requests (2026-10-04)

Native series job `1791136455435-163` completed with exit 0 in 44m34s.
The already-built source remained `f59c8dee226225fd743b14ec525116d7a6e40382`,
dirty=true, for all six requests; it was not rebuilt or retagged for closure.
The target GGUF, exact saved 4096/16384 token-ID fixtures, seed 12345,
temperature 1, top-p .95, top-k 20, ignore-EOS, 512 actual outputs, chunk 1024,
112 expert slots per layer, attention tile 8 and CPU-off mode are matched.
Each process starts fresh with no prefix reuse; expert-cache warmth is unknown.
MTP owns its additional Q8 sidecar and workspace, so equal target slots do not
mean identical total VRAM. The original qualified core-session-baseline is intact.

| Request | MTP | PP tok/s | TG tok/s | Full request s | Load s, separate |
| --- | --- | ---: | ---: | ---: | ---: |
| 4k-A1 | off | 35.458543 | 9.908767 | 167.087150 | 74.851181 |
| 4k-B | on | 35.055268 | 10.187122 | 167.007184 | 75.681456 |
| 4k-A2 | off | 35.921420 | 10.080275 | 164.721197 | 80.333066 |
| 16k-A1 | off | 32.149104 | 9.635982 | 562.657292 | 72.282550 |
| 16k-B | on | 31.877007 | 10.365109 | 563.277182 | 75.490743 |
| 16k-A2 | off | 32.131512 | 9.649570 | 562.861627 | 76.878152 |

No stable end-to-end speedup is promoted. MTP request time is 167.007 s at 4K
against off repeats 167.087/164.721 s; at 16K it is 563.277 s against
562.657/562.862 s. Mean-off/on request ratios are 0.993395 and 0.999081.
MTP accepts 0/1/2 drafts in [26,31,141] and [24,35,139] windows respectively.
Both off repeats reproduce all 512 output IDs exactly within each length.
On/off output equality is not required: speculative sampling consumes different
random streams. Counts, pending/consumed contracts, finite timings and all six
native exits passed. This is performance accounting, not new long-logit or HF
parity qualification. Actual model-EOS handling is not qualified by ignore-EOS.

Both TG rates use 511 post-first-output tokens. MTP PP includes target and
teacher warmup plus first-output sampling; off PP excludes first-output sampling.
Full request is completed host wall including sampling/output IO, with load
separate. MTP verify_ms includes CPU distribution filtering and decisions, not
just GPU execution. These distinctions are retained in the journal.

Raw artifacts: `ROOT/runs/r6-matched-ec8-20261004/{4k,16k}-{A1,B,A2}`
with `.jsonl`, `.err`, `.manifest.json`, `.summary.json`; manifests retain full
actual Docker argv and native status. Fixtures are in the same directory.
Driver/validator: `ROOT/runs/r6-matched-ec8-{driver,collect}.py`. Do not blindly
rerun the append: canonical `ROOT/results.jsonl` now has 55 records, with the
previous 54-record byte prefix unchanged. The controller copied all six actual
raws, matched source/footer/output arrays to the journal, and reran the strict
existing collector on all four off requests. No duplicate historical append.

Next: phase-separated PP and MTP verification profiling, followed by targeted
optimization and a fresh pinned llama.cpp comparison. R4–R8 and the overall
performance goal remain open. The mutable production models.ini now names a
different model; matched donor runs must use explicit recovered launch arguments,
not that preset. No model or precision change is authorized by this measurement.

## Phase profiling and register-pressure hypothesis (2026-10-04)

Optional `-DCORE_BUILD_PROFILE_TOOLS=ON` adds two SEPARATE diagnostic targets:
`core-profile-prefill MODEL whitespace-token-ids.txt` and `core-mtp-profile`
(the latter uses the same arguments as core-mtp-run). They link the installed
ROCTX library; normal core-session/core-mtp-run do not gain that dependency or
replace their measured executables. Default builds compile ProfileRange to a
no-op. Completed ROCTX ranges isolate loading, target PP chunks, draft, target
verification including CPU distributions/decisions, and teacher/restore.

Native job1791139209698-232 exited0 in9m19s, compiled ec8ab03/dirtytrue. Both
strict diagnostic targets built successfully. Annotation-only8+32 preserved all
32 checked output IDs, acceptance[10,6,3] and RNG38/46. Seven malformed fixture
cases rejected before model loading, including an overflowing integer after a
valid4096-token prefix. Fresh diagnostic PP4K and MTP4K+128 completed; no full
CTest rerun is claimed for this annotation-only slice (prior39/39 is retained).

Run rocprofv3 with `--kernel-trace --memory-copy-trace --marker-trace
--output-format csv --output-directory FRESH -- EXECUTABLE ...`. The completed
PP marker encloses synchronous Session calls; no summed load/TG substitution.
Native analyzer job1791139851171-245 exited0 in1m02s. Zero kernel events crossed
the selected completed-phase boundaries. These are traced diagnostics, not
speed-qualified rates; kernel and DMA sums overlap and are not additive latency.

PP4K: wall126.1396s, 3,397,677 kernels, summed kernel duration97.2709s.
- Batched attention chunk:6,144 launches/39.7998s,40.92% of summed GPU duration
- Q4 canonical N8:781,636 launches/23.7616s,24.43%
- Dense MMVF N8:148,480 launches/10.0989s,10.38%
- Full Q6 head:3,584 launches/2.8318s,2.91%
- Kernel union per GPU:47.4781s/49.7645s; H2D summed7.3597s/7.2341s

MTP target verify:48 ranges/13.1496s wall,588,140 kernels/6.4473s summed GPU;
batched attention is2.0565s,31.90% of summed GPU. Draft wall0.5043s and
teacher/restore0.1231s are much smaller. PP target-only markers reproduce the
same dominant operations as non-MTP PP. This rules out treating draft cost or
LM-head skipping as the primary measured opportunity on this fixture.

Actual attention kernel metadata sample:VGPR_Count256, Scratch_Size4380,
LDS_Block_Size15872. Large private scratch and maximal VGPR count motivate a
register-pressure experiment: keep the exact ascending256-channel FP32 dot,
finite checks, Q4/half representation, FP64-exp-to-FP32 and reductions, but stop
fully unrolling its32 iterations. This is a hypothesis, not an implemented or
accepted acceleration. Donor furnace fattn.cu uses a different quarter-dot
reduction; do not silently adopt its arithmetic just to eliminate spills.

Raw directory ROOT/runs/r4-profile-ec8-20261004; CSVs are nested in each
*-trace directory. Summary ROOT/runs/r4-profile-ec8-summary.json; driver/analyzer
ROOT/runs/r4-profile-ec8-{driver,analyze}.py. Journal56 includes actual source,
completion records, preflight, phase summary and metadata sample; old55 bytes
are preserved and the copied controller summary matches exactly. Do not append
these artifacts again. Frozen correctness gates and full-plan scope are unchanged.

## Fresh pinned llama.cpp MTP2 baseline (2026-10-04)

Native job1791140372565-257 completed with exit0 in16m05s. Each of four requests
uses a fresh server process in the existing pinned image
`llama.cpp-gfx906:pp-stream-dcd685463d`, verified revision
`dcd685463d597d31f5ca759d32c94592a2740fa4`. HTTP is loopback inside a network-none
container, with no published port. Explicit historical launch arguments replace
the mutable production preset, which now selects another model.

Same actual4096/16384 token-ID fixtures, target GGUF/Q8 sidecar, requested
seed12345/temp1/top-p.95/top-k20/min-p0/repeat1/presence0,512 output tokens,
ignore-EOS, Q4 target/draft KV,112 MoE slots,1024 batch/ubatch,16 CPU threads,
layer split1,1 and MTP2. Every response passes the existing baseline validator,
reports no truncation and retains actual512 output IDs. No prefix reuse; cache
warmth is unknown. Server sampling serialization exposes top-p as FP32
0.949999988079071; this is preserved rather than rewritten as exact double.

| Request | PP tok/s | Donor TG tok/s | HTTP request s | Load-to-ready s, separate |
| --- | ---: | ---: | ---: | ---: |
| 4k-1 | 181.882351 | 14.848663 | 57.010654 | 125.923855 |
| 4k-2 | 181.652874 | 14.800414 | 57.151201 | 128.996698 |
| 16k-1 | 175.086729 | 10.093087 | 144.333988 | 136.127030 |
| 16k-2 | 173.935029 | 10.259562 | 144.124976 | 129.095101 |

Important headroom correction: the first attempt, job249, used capacity
4096+512 and the donor server truncated at511 outputs. The strict collector
rejected it; its raw response/logs remain in r6-llama-matched-20261004/4k-1,
not in the accepted512 series. The corrected series uses capacity prompt+1024:
5120/17408. Final occupied tokens_cached is4607/16895, not prefix reuse.
Previous own off/on/off used prompt+512 allocation, so do NOT call the allocated
capacities identical or present this as the final paired engine comparison.
All subsequent own baseline/candidate performance runs must use prompt+1024.

Donor TG divides512 by its predicted interval; own incremental TG divides511
by its post-first-output interval. Compare completed request wall times with
these timing boundaries disclosed. Structured0/1/2 acceptance is unavailable,
so no histogram is reconstructed from text totals. The donor's historical QSA
semantics are not an independent correctness oracle for the corrected core.

Accepted raw directory ROOT/runs/r6-llama-matched-headroom-20261004, including
image identity, launch/outer Docker argv, native exits, exact requests/responses
and summaries. Reproduction driver/client are ROOT/runs/r6-llama-matched-{host,client}.py.
Canonical journal57 preserves all56 previous records byte-for-byte; controller
copied the actual response set and reran all four strict validations. No final
speed win is claimed. The next own optimization remains under correctness gates.

## Canonical attention dot loop: bounded correctness closure (2026-10-04)

The candidate changes ONLY the two QK-dot loop unroll pragmas in attention.hip
from full unroll to unroll1. It keeps ascending FP32 products/additions, every
finite check, Q4-to-half values, double-exp-to-FP32 evaluation, selected-ID order,
chunk/reduction geometry and all acceptance gates. This targets register pressure,
not reduced precision or a new attention algorithm.

The model-free fixture now optionally supports `--capture FRESH.bin` and
`--compare SNAPSHOT.bin`. Default invocation/three-record output stays unchanged.
The witness contains native same-host typed case headers, all output/padding
values and a completion footer, which is written only after both GPUs pass.
Controller CPU sanity passed roundtrip plus6 fresh-path/corruption/truncation/
trailing-data/metadata/value rejection cases. This is a fixture, not model storage.

Old-kernel capture job1791141492191-268 and candidate comparison272 both exited0
in33s. Unchanged-source checksum was checked before capture. All52 cases and
3,745,280 values including padding are bit-identical across builds. Each GPU
also passes26 cases/304 queries,1,867,776 independent CPU comparisons at the
frozen2e-4+2e-4*abs(ref) gate,23 device rejections,180 host rejections and6 sticky
checks. CPU maximum absolute error remains1.1920928955078125e-7.

Actual traced kernel metadata and component duration sums:
- Batched:VGPR256/Scratch4380 -> VGPR52/Scratch0; LDS15872 unchanged
- Single-query:VGPR256/Scratch4372 -> VGPR52/Scratch0; LDS15872 unchanged
- Same110 batched launches:77.457911 ->17.018558ms (~4.55x)
- Same608 single-query launches:308.264752 ->85.367026ms (~3.61x)
These are traced component sums, not full-request speedups.

Full native job1791141889741-275 exited0 in30m13s: strict build and CTest39/39
passed1148.60s, then the unchanged actual48-layer prefill-attention fixture passed.
It completed2163 calls/6264 rows, inspected1,555,476,480 finite logits and compared
1,036,984,320 values with zero violations, zero diagnostic bit mismatches and
maxabs0. The full-model reference is N1 IN THIS CANDIDATE BUILD; it is not an
independent pre-change model-logit or HF reference. The separate component
snapshot supplies pre-change bit evidence. Logical2047–2056, reset/rejection,
state, memory and continuation checks retain their original scope and gates.
Remote and controller strict model collectors passed on the actual artifact.

Compiled provenance remains7df1c746d7decceea1ad04536fb83968c7975c43/dirtytrue.
Raw prefix ROOT/runs/r4-attention-roll:reference/candidate JSONL, binary witness,
component traces/summary, full-regression.log and model.jsonl/.err. Canonical
journal58 adds exactly one bounded-correctness record after57 unchanged records;
controller copied and matched actual component/model data. Do not append again.

The saved non-MTP core-session-baseline remains intact. The already-tested
f59 MTP executable was also preserved as core-mtp-run-baseline before rebuilding,
so one baseline runtime has separate off/on entrypoints; no models were copied.
A trace-off4K/16K A/B/A comparison against that MTP baseline is now running with
capacity prompt+1024,512 actual outputs and identical sampling. Full-request
speed promotion and all remaining R4–R8 goals are still open.

## Rolled attention: matched full-request promotion (2026-10-04)

Native job1791143846406-290 completed exit0 in41m55s. Each length used
A1(old MTP runtime) -> B(rolled attention) -> A2(old), with no tracing, competing
GPU workload or compilation. Same target/sidecar, exact saved token IDs,
capacity prompt+1024 (5120/17408),512 outputs,slots112,chunk1024,tile8,CPU0,
seed12345/temp1/top-p.95/top-k20 and ignore-EOS. All sessions are fresh with
no prefix reuse; expert-cache warmth is unknown. Load is recorded separately.

| Run | PP tok/s | TG tok/s | Completed request s | Load s, separate |
| --- | ---: | ---: | ---: | ---: |
| 4k-A1 | 35.554479 | 10.325402 | 164.694867 | 81.166275 |
| 4k-B | 48.298361 | 12.199959 | 126.693218 | 80.637726 |
| 4k-A2 | 35.166759 | 10.270663 | 166.228675 | 85.445873 |
| 16k-A1 | 32.099215 | 10.392727 | 559.588212 | 79.984882 |
| 16k-B | 46.173196 | 12.280926 | 396.448825 | 82.893748 |
| 16k-A2 | 31.851884 | 10.527262 | 562.923186 | 95.695762 |

The candidate beats BOTH baseline bookends for full request at each length:
mean-A/B ratios1.306003x at4K and1.415708x at16K. Every one of the512 output IDs,
acceptance histogram and proposal/decision RNG counts matches across each
triplet. This qualifies a bounded own-runtime improvement; it is not merely a
component timing inference. One B and two A repetitions were collected per length.

The pinned donor remains faster in full requests: about57s/144s versus the
new own126.693s/396.449s at4K/16K. Allocation headroom is now matched, but retain
the documented HTTP/CLI and TG-numerator differences and historical donor QSA
semantics. No llama.cpp victory,400–600PP/30–40TG,HF-reference or occupied128K
claim follows from this slice.

Raw directory ROOT/runs/r4-attention-roll-ab-20261004 retains all six full source,
output and completion streams, manifests, fixtures and pair summaries. The
controller copied and matched all actual raws/manifests against journal59,
which adds exactly one record after58 byte-identical historical records.
A source is f59c8de/dirtytrue; B source is7df1c74/dirtytrue, unchanged and never
retagged with closure commits. Native literal performance flags are preserved.
Do not append these accepted artifacts again.

After acceptance, core-mtp-run-baseline was promoted by byte-verified copy of
the tested rolled executable (still source7df1c74/dirtytrue). The prior saved
non-MTP core-session-baseline is untouched. Historical command paths alone do
not pin an executable forever; use each raw source revision/variant for replay.
Next is component-qualified canonical tiled-grid linear dispatch; no new Session
integration or end-to-end claim is included here.

## Canonical tiled-grid linear primitives (2026-10-04)

New checked launch_quantized_linear_tiled and launch_dense_linear_tiled accept
logicalN1..128 while retaining the exact original physicalN<=8 arithmetic.
Full N8 tiles occupy independent grid.y positions; a final N1..7 tail uses the
original short kernel. No wide-MMQ arithmetic, DS4 conversion, new precision,
workspace or changed reduction is introduced. Full logical readable/writable
ranges are checked before any enqueue; legacy N<=3/N<=8 APIs keep their limits.
Dense tiled mode requires even K and float2-aligned weights/input and has no
SGEMM fallback. The original dense API's fallback remains unchanged.

Manual core-linear-tiled is model-free. Native338 exited0 in1m09s after strict
build, the first fixture and unchanged linear-short-test/dense-mmvf-test; expanded
native344 exited0 in35s. Compiled source296ea56/dirtytrue is preserved. Expanded
coverage per GPU:300 cases,204,939,974 exact compared values including timed
reuse,81,352 CPU samples,21 host rejections and298 graph checks. Both GPUs pass;
max common-Q8 error6.4373016357421875e-6, max bound ratio0.0256371 at unchanged
2e-4+2e-5*abs(ref). CPU checks sample up to five spread rows across all columns;
full arrays are compared against chronological canonical N8 GPU execution.
Dense CPU checks use an exactly representable dyadic pattern, plus non-dyadic
GPU bit comparisons. Padding, immutable inputs, readonly aliasing, quantizer
raw-sum/subnormal headers, repeat/reuse, invalid ranges/handle preservation and
legacy limits pass. Captured logical calls contain exactly one or two kernel
nodes; this is a graph observation, not a full-model physical-dispatch count.

Resident event A/B/A timings cover86 device/shape/column coordinates, excluding
quantization, transfer and model costs. On both GPUs atN>=16, measured quantized
shapes and dense outputs<=512 beat both serial-N8 bookends. Examples atN128:
Q4 gate/up ~2x; Q4 down ~1.6x; HC4x10240 ~15.6–15.8x. Large F32 output2560 is
NEGATIVE: about0.76x atN128. N9 dense is not a stable win. Therefore the candidate
Session policy must be selective: logical tiles only when remainingN>=16;
quantized projections and dense output<=512, with legacy N8 otherwise. This is
component-derived policy, not an accepted model speedup or universal dispatch.

Raw prefix ROOT/runs/r4-linear-tiled-296, expanded.jsonl/.err and legacy
short/dense regression files. Journal60 preserves59 old records exactly and
includes all expanded timings, both device results and old-path regressions.
Controller copied the actual expanded stream and matched it to the journal.
The APIs are not yet wired into Session at this closure. Full-model state/
continuation and trace-off request measurements are mandatory after integration.

### Selective canonical tiled Session integration (2026-10-04)

Matrix dispatch now joins physical N8 tiles for logical remaining N>=16,
up to128, only for quantized projections or dense output<=512. Larger dense
retains the measured original N8 path. Expert gather/project/scatter similarly
uses bounded groups up to128 with the same physical arithmetic and original-rank
fold. No weights, workspace allocation, precision or QSA budget changes.

Source60d341ebbacbbe6fecd5fbea00dadf9376bc2474/dirtytrue:
selected build and both model fixtures (job356), then full strict build/39CTest
(job364,nativeexit0,1156.03s tests) pass. Short29,798,400 and wide1,036,984,320
logit comparisons have zero violations and diagnostic bit mismatches. These
are same-build N1 references, not independent HF proof; cross-build component
bit parity is recorded in the preceding primitive slice.

Actual reproduction uses tools/build.sh, core-prefill-test and
core-prefill-attention-test with the unchanged target GGUF. Raw logs are
ROOT/runs/r4-tiled-session-60d-{short,attention}.jsonl and
ROOT/runs/r4-tiled-session-60d-full-regression.log; ROOT means
/home/radneon/gfx906-core. Canonical results.jsonl record61 contains provenance.

One diagnostic4K+512 MTP request gave114.5936s, PP57.878/TG11.661, with every
output ID/acceptance/RNG/pending matching the rolled predecessor. This does not
qualify speed. Run python3 ROOT/runs/r4-tiled-session-ab.py for trace-off A/B/A
4K/16K+512, baseline core-mtp-run-baseline7df1c74/dirtytrue and candidate
core-mtp-run60d341e/dirtytrue, headroom1024, slots112, chunk1024, attention tile8,
seed12345, temperature1/top-p.95/top-k20, ignoreEOS. No competing GPU work/build.

Selective tiled Session speed qualification: native job1791150716560-382
finished exit0/34m10. Trace-off fresh-process A/B/A, capacity prompt+1024:
4K A126.127/127.737s vs B111.971s (meanA/B1.133616), PP58.510/TG12.177;
16K A397.990/397.574s vs B338.686s (1.174487), PP55.161/TG12.266.
All512 output IDs, acceptance histogram, sampler draws and pending state match
within each triplet. Allsix actual raw sources/footers/manifests were copied and
revalidated; canonical journal62. Raw ROOT/runs/r4-tiled-session-ab-20261004.
This accepts the selective dispatch for these workloads only. Pinned llama
full requests remain faster (~57/~144s); no final speed target is met.
A is rolled7df1c74/dirtytrue, B tiled60d341e/dirtytrue. Baseline executable still
contains A; no subsequent build/profiling should relabel these historical runs.

### Canonical fused SiLU/up and Q8 component

launch_silu_up_q8_1 joins the unchanged SiLU division/product/sanitization and
Q8_1 width32 scale/raw-sum/code operations. Middle FP32 remains materialized.
Width32..16384 divisible32, columns1..128; input aliases are allowed but all
writable ranges must be disjoint and aligned4. Sticky errors are preserved.

core-silu-q8 tests both gfx906 without a model: 288 cases/device,48,302,592
initial middle values and54,340,416Q8 bytes exact, including nonfinite inputs,
overflow, signed zero, tiny scales, sticky flags,31 host rejections,36 one-node
graph checks,144 same-owner numeric-failure recoveries,36 readable alias cases.
Original paired-middle/linear-short/dense-MMVF regressions pass. Jobs470/479
exit0; job475 failed solely on an incorrect build target name before tests.
Source85ff77818b28e1f936afcb71b0e67773f5b66b7a/dirtytrue, journal67,
raw ROOT/runs/r5-silu-q8-85f-expanded.jsonl. Build target core-silu-q8 then run
/core/build/core-silu-q8 in the usual exclusive gfx906 build-image container.
All24 resident width640 A/B/A coordinates improve1.345–1.577x over separate
launches; this is component evidence only, not integrated PP/TG or MTP speed.

Fused routed-expert Session integration passes native job1791156231841-489
(exit0/33m04): strict selected model/API targets, short and2088-row attention
fixtures, MTP request/reset/reject/custom-stop/capacity tests,39CTest1147.26s.
Short29,798,400 and wide1,036,984,320 logits compare with zero differences.
Compiled6c26e2abf9bfa14f175940f0a851fe829f464bd7/dirtytrue; raw
ROOT/runs/r5-fused-session-6c2-*, journal68. This is same-build N1 model
self-parity plus prior component two-launch reference, not independent HF.

Next paired driver ROOT/runs/r5-fused-session-ab.py rebuilds the ordinary
core-mtp-run between the actual separate/fused Session snapshots. The saved
core-mtp-run-baseline remains untouched and is checked unchanged. Builds and
model loads are outside request timings; no compilation overlaps inference.
Both variants preserve source6c26e2a/dirtytrue with explicit variant manifests,
prompt+1024 capacity,512 outputs,primary sampling and the same4K/16K fixtures.

Fused Session paired series job1791158371388-499 finished exit0/33m08.
4K+512: separate112.052/110.843s, fused111.536s (meanA/B0.999212): no stable
full-request win.16K+512: separate335.692/335.591s, fused331.419s
(meanA/B1.012739); PP56.355/TG12.559. All512 IDs/acceptance/RNG/pending exact.
This is a small16K benefit only; do not extrapolate component1.35–1.58x to the
runtime. Six actual raws/manifests revalidated, journal69; raw directory
ROOT/runs/r5-fused-session-ab-6c2. Saved baseline binary unchanged.

### Exact-zero sparse sampling path

The probability validator skips only exact signed zeros, which are already
valid mass. Residual positive-difference skips zero-support divisions while
preserving signed-zero output bits; its compensated mass loops ignore exact
zeros only. Every nonzero value, including the smallest FP32 subnormal, still
uses the original arithmetic, normalization checks and RNG. Generic Neumaier
Sum and distribution filtering are unchanged.

core-sampling-probe LOGITS --capture FRESH.bin or --compare EXISTING.bin uses
the first8 saved actual full-vocabulary logit rows and32 cases across primary,
unfiltered,greedy andtop-k1 configurations. Source2535cbf/dirtytrue:95,389,224
snapshot bytes match across builds (probabilities,residuals,acceptance,draws,RNG).
Sampling/speculative tests and explicit signed-zero/subnormal tests pass.
CPU0 finalA2/B3/A3 averages: distribution~.578ms unchanged,draw.623->.165ms,
acceptance1.176->.258ms,residual5.023->1.776ms. All timings are CPU components
at primarytop-k20, not MTP runtime gains. Native jobs511/515/524/525/529/530
exit0; journal70, raw ROOT/runs/r5-sampling-zero-*; snapshot footer is required.

Sparse sampling MTP qualification: API job1791161487780-534 exit0, then
trace-off paired job1791161686931-537 exit0/31m53.4K+512 separate112.812/110.077s
vs sparse109.526s (meanA/B1.017521), PP58.322/TG13.005.16K separate334.585/
333.669s vs sparse330.152s (1.012040), PP56.130/TG13.357. Every512 output ID,
acceptance histogram, RNG count and pending token match within both triplets.
Journal71, raw ROOT/runs/r5-sampling-zero-ab-253; source2535cbf/dirtytrue with
explicit sampler snapshots. The ordinary executable was rebuilt outside
measurements and restored to the candidate; saved baseline unchanged. These
are bounded1–2% full-request improvements over the own previous runtime.

### Last-row prefill output API

Session::prefill_last forwards the entire input window with unchanged state,
routes and complete layer47 tap, but requests only the final LM-head row.
step_batch and verify_window still return all rows. Existing bounds, pending
verify invalidation and borrowed-publication rules are unchanged; no new owner
or allocation. Root HC still runs for every input row. MtpRunner is not yet
using this API at this checkpoint.

core-prefill-last-test MODEL tests full-output32 reference against last-output
chunks1/2/3/8/17/32, continuations, mixed calls, invalid/capacity preservation,
pending verification and restore. Both checkpoint modes pass. Native562/568
exit0;35,261,440 checkpoint and2,483,200 ordinary values bit-identical,full tap
bytes exact,owned allocations steady. Constructorcapacity40/slots1/max32/tile8;
this is bounded self-parity, not independent HF or long-context qualification.
Source daee1efcd6e83c115175284d7ce9512045298b8d/dirtytrue, journal72/73,
raw ROOT/runs/r4-prefill-last-dae{,-expanded}.jsonl. No speed claim yet.

MtpRunner::begin now uses prefill_last for each prompt chunk while keeping every
target tap row for trained-MTP teacher warmup. Full strict build/MTP API/39CTest
job1791164948838-576 passed nativeexit0/22m00; tests1154.97s. Exact32 generation,
RNG/reset/reject/custom-stop/capacity48 fixture passed. Source48de1ba/dirtytrue,
journal74; raw ROOT/runs/r4-mtp-last-48d-*. No speed claim before paired requests.
Driver ROOT/runs/r4-last-head-ab.py toggles only actual MtpRunner source snapshots
between builds, outside timed inference; saved baseline remains unchanged.

Last-head paired series1791166462036-582 finished exit0/31m35.4K+512:
all-head110.443/108.628s versus last104.385s (meanA/B1.049345),PP62.177/TG13.271.
16K+512: all-head332.074/334.021s versus last323.481s (1.029575),PP57.615/TG13.066.
Every512 output ID,acceptance histogram,RNG andpending token matches inside
both triplets. Source48de1ba/dirtytrue, same original fixtures/sampling and
prompt+1024 capacity; builds outside timing, saved baseline unchanged.
Journal75 and actual copied raws ROOT/runs/r4-last-head-ab-48d. This is a bounded
own-runtime improvement, still slower than the pinned llama.cpp full requests.

The separate small-K Q4 one-wave experiment was not promoted. Native608 exited0:
309 cases/device,9,161,968 exact GPU values,99,356 CPU samples,19 host rejects,
308 graph checks and unchanged legacy tiled/short/dense regressions. K<=1024
provably leaves the old second wave empty; the candidate preserves acc+0 and
sum64 exactly. However,48 resident down-projection coordinates gave only19
wins against both controls,median ratio1.00688 and several regressions.
Runtime was never integrated; unused candidate API/test changes were reverted.
Journal76; raw ROOT/runs/r5-small-k-567.jsonl and exact experiment patch
ROOT/runs/r5-small-k-567.patch preserve the negative result. Cross-fixture
cache/timing differences are not accepted as source speedups.


The equal-accumulator Q4 tile4columns/4rows experiment was rejected as a broad
dispatch. Native624/631/633 all exit0, source8844338/dirtytrue. On each GPU,
462 cases compare503,678,810 values exactly against canonical short kernels,
115,228 CPU samples pass frozen gates,21 host rejects and460 graph checks pass.
Outer A/B/A has48 changed resident coordinates:28 beat both controls,median
meanA/B1.025915. However dominant Q4_0 down N32..128 regresses to0.871–0.897x.
Gate/up tail cases N20/28/36 benefit, but no narrow policy or model speed
qualification is claimed. Production kernel and fixture restored; experiment
patch and extended fixture preserved under runs/r4-tile4x4-884.patch and
runs/linear-tile4x4-test-884.hip. Journal77 and raw r4-tile4x4-{a1,b,a2}.jsonl.
No ordinary runtime relink or baseline overwrite occurred.


### 2026-10-05 opt-in layerwise prefill correctness checkpoint

Session::prefill_layerwise uses a separate constructor capacity (off by default),
routes a full window once per layer, and stages each active expert only once.
Legacy chunk<=1024 and verify APIs retain their bounds. Canonical selective
tiles<=128, fused SiLU/Q8, chronological GDN and rolled QSA math are unchanged.
Two full-N pageable activation owners and bounded pinned frames preserve DMA
lifetimes; all-row and final-row output modes retain every target teacher tap.
A separate16384-token route-copy bound reuses exact byte-copy kernels; default
1024 APIs still reject larger input. No new weight format or precision change.

Native649/654 model32/128 passed. Native667 model4096 and strict full build
passed before SSH transport exit255; the SAME Docker container continued.
Recovery wait676 observed container exit0 and39/39CTest passed in1151.06s;
nothing was rerun. Source c685d14a80bb94a5f7cab41afc0b4ddc73710401/dirtytrue.
Full-vocabulary N1/chunk1024 self-parity, all/last output, full taps, occupied
prefix, pending-owner rejects, restore/continuation, reset and steady buffers
pass frozen .02+.002abs(ref). These are self-parity gates, not independent HF.

Native663 copy tests passed both GPUs at16384 token capacity,327680 actual
token/rank pairs per GPU (two alignments),9310 gather and9310 scatter calls,
270host rejects/device, exact opaque bytes/canaries/sticky flags/read-only
sources and owner cleanup. Original copy fixture and491148-check CPU route
regression pass. Journal78; raw ROOT/runs/r4-layerwise-c685-{32,128,4096}.jsonl,
r4-layerwise-copy{,-legacy}-c685.jsonl and build/CTest logs.

At4096 target-only: each GPU originalQ8=11796480B,contributions419430400B,
DTO327680B; root tap167772160B EACH. Two host activation owners167772160B each.
Observed freeGPU0/1 initially4578082816/3764387840B, later4561305600/3745513472B.
All-row transactional output owners grow to8136949760B host; last-row does not
require that growth. This is measured target-only fit, NOT target+trained-MTP fit.
Warm-cache monotone-ID fixture uploads44.69GB oldchunk versus19.74GB layerwise,
but all-row wall includes fixture comparisons and layerwise is slower there.
No speed promotion from this fixture. Next integrate opt-in MTP, prove original
IDs/state and actual memory fit, then trace-off matched full-request A/B/A.


MtpRunner now accepts optional layerwise_prefill_capacity<=4096 (frame>=4).
core-mtp-run --layerwise-prefill N exposes it explicitly;0/default keeps the
original chunk path. Every full-window tap row is still consumed by sequential
trained teacher warmup, including cross-window carry. CLI source records both
frame and layerwise capacities. Six malformed/duplicate/bounds flags reject
before model loading. Selected strict build and both default/enabled API tests
passed job701 exit0/3m51; original32 IDs/RNG,reset/reject/custom stops/capacity48
hold with an8-token prompt, layerwise capacity16. Sourcebcd5320/dirtytrue,
journal79,raw r4-layerwise-mtp-bcd-*. This does not prove4096+sidecar fit or speed.
The preceding39CTest belongs to the Session slice; no new full-suite claim.

Paired driver ROOT/runs/r4-layerwise-mtp-ab.py compares ONE compiled binary with
--layerwise-prefill0/4096/0 on4K then16K+512. Originalchunk1024/slots112 and
sampling/capacity remain matched. No recompilation during timing; binary and
saved-baseline hashes checked. Driver is SSH-disconnect-resilient under nohup,
PID and native exit files r4-layerwise-mtp-ab-bcd.{pid,exit}; output directory
r4-layerwise-mtp-ab-bcd, series log same prefix. Actual emitted512 IDs/acceptance/
RNG/pending must match, not merely reported PP. Series still active at this commit.


The first layerwise4096 paired candidate exposed a stale integration bound:
MtpSession::save_target_carry still rejected target taps above1024 rows. The
4K-B process exited1 after target/teacher prefill, before output generation;
raw r4-layerwise-mtp-ab-bcd preserved and excluded from performance claims.
The carry validator now matches Session's16384 full-window bound and still
copies exactly one checked row. No arithmetic or memory allocation change.
Native720 strict selected build + new2051-token regression passed exit0/4m13:
default1024 chunks versus layerwise[1025,1025,1] give exactly the same32 IDs,
RNG counts and acceptance. Sourceb2014ce/dirtytrue, journal80.
Fresh matched series r4-layerwise-mtp-ab-b201 uses a Python detached wrapper
with native exit JSON; the first shell wrapper had an unrelated printf quoting
failure, so its empty exitfile is not evidence. Original candidate manifest
records native1. No failed run is reused as a speed comparison.


### 2026-10-05 layerwise4096 measured full-request improvement
Fresh same-build A/B/A completed nativeexit0 at05:21:28UTC, detachedPID1377858.
Sourceb2014ceabbe099f7eb4cf64471cfa9d9235466eb/dirtytrue, source/actual binary
unchanged across all six requests. Saved baseline hash unchanged. Candidate
explicit --layerwise-prefill4096; controls0; all use frame1024,slots112,tile8,
prompt+1024 capacity, original4K IDs/four-concat16K and primaryseed12345.
4K+512: A104.162715269/105.963112306s, B91.233442267s, meanA/B1.151583358.
B PP77.55286872/TG13.30181011.16K+512: A319.447977231/322.707701777s,
B267.236364684s,meanA/B1.201475106,PP71.70604842/TG13.18837323.
All512 output IDs,acceptance,RNG andpending token exact within both triplets.
Load is separate; no prefix reuse; expert cache warmness unknown. Both candidate
requests fit target+trained-MTP, but no new sampled/exact peak VRAM measurement
yet. These are qualified opt-in own-runtime wins, not default policy promotion,
llama.cpp win, independent HF or full-plan completion. Pinned donor~57s/~144s
remains faster. Journal81; actual copied raws/manifest/IDs verified under
ROOT/runs/r4-layerwise-mtp-ab-b201; failed earlier bcd series remains excluded.


### 2026-10-05 layerwise memory and phase diagnostics
Separate16K+512 observed candidate completed native0, sourceb2014ce/dirtytrue,
same512IDs/RNG as unobserved B. Fresh HIP query maps device0/1 to0000:05:00.0/
0000:08:00.0.3617 samples at0.1s over363.58s include load/request/cleanup:
global driver maxused12701913088/16393973760B, minfree4461178880/769118208B.
These are sampled global-driver extrema, NOT exact instantaneous or HIP-owned
peaks. Root/device1 has only~0.716GiB observed headroom. Increasing full-window
capacity8192 without a new budget is NOT justified. Journal82, actual observer
and request raws ROOT/runs/r4-layerwise-vram-b201.

Diagnostic-only ROCTX profile job749 exit0/4m20, analyzer751 exit0, sourcebd1fa883/
dirtytrue. Annotation smoke32 retains checked IDs/RNG.4K+128 profile target-PP
55.047s,1,081,432 kernels,summed41.739s,agent unions20.928/20.806s. Q4_0 tiled
matrix32<8,2>16.023s (38.39% of summed target kernels), attention9.302s(22.29%),
GDN1.683s,dense-tiled1.651s. H2D summed3.008/2.906s,D2H~.654/.653s.
Entire PP with teacher56.207s; verify128output10.799s/summed4.836s, draft.458s,
restore.094s. Zero boundary-crossing events. Sums overlap, tracing adds overhead;
these numbers are NOT paired wall-speed comparisons. Journal83, trace CSVs
remain ROOT/runs/r4-profile-layerwise-20261005; compact summary copied.
Next bounded candidate: canonical Q4 physical16columns/2rows, retaining exact
per-output reduction math; compare component A/B/A before any model promotion.


The canonical Q4 physical16columns/2rows experiment was rejected. A/B/A native
758/762/763 all exit0, source56496c8/dirtytrue. EachGPU512cases/530056502exact
GPU values,136600CPU samples,21host rejects and510graph checks pass. But only
1/42 changed coordinates beats both controls; meanA/B median0.82095,
range0.72506–1.02792. Production code/fixture restored; ordinary runtime never
rebuilt. Exact experiment patch and extended fixture preserved under runs,
raw r4-tile16x2-{a1,b,a2}.jsonl and journal84. No speculative register-pressure
explanation is claimed as a measured cause.

Next bounded hypothesis targets wasted lanes rather than a larger accumulator
tile: Q4 K640 has20 blocks/two fragments, so only40/128 oldthreads enter the K
loop. Pack six40-lane output rows in256threads, store original64-lane partials
with explicit old wave1+0, then use the same sum64 and original output lane.
This preserves per-output arithmetic but changes occupancy/reuse. Unimplemented
and unqualified at this checkpoint; measure both-GPU exact gates and A/B/A first.


The packed K640 Q4 component passes exact old-short parity on both GPUs:
530cases/device,530126594 exact GPU values,172996 CPU samples,21host rejects,
528graph checks. PhysicalN8/tails and all public bounds remain unchanged.
K640 uses exactly40 original fragments per output row; three rows fit120 of
128 threads. Original partial indices,+0wave1 merge,sum64 and globalrow%2
publication are preserved; only CTA ownership changes. Static LDS is6KiB,
no extra persistent workspace. Candidate only Q4_0/Q4_1,K640,M<=2560,N>=16.

Five-phase A1/B6/A2/B3/A3 native772/776/778/781/783 all exit0,source40cb728/
dirtytrue. Both 6row/256thread and3row/128thread versions beat both neighboring
controls at all36 resident down coordinates. Three-row meanA/B1.10007–1.52507,
median1.21848; main N128 Q4_0~1.22 and Q4_1~1.52. Three rows selected for
measured shapes; not a universal tuning claim. Journal85 and raw r4-packed640-*.
The separate K2560 row-owned variant failed timing: native790/792 exit0,
all correctness gates pass,0/18 beats controls,median0.77683,range.66454–.94614.
Only this extra K2560 change was removed. Patch retained, journal86.
Ordinary runtime was not rebuilt during component timing.

Selected three-row kernel is now under full-model regression, not yet a
full-request speed promotion. Detached gates PID1402124, prefix
ROOT/runs/r4-packed640-gates-40c: strict full build,short/attention-wide model,
layerwise128,MTP API and39CTest. Actual source remains40cb728/dirtytrue.


Packed K640 three-row full-model qualification completed native0 at06:53:52UTC,
source40cb728/dirtytrue, detachedPID1402124. Strict full build and39/39CTest
(1145.90s) pass. Strict collectors accept short29,798,400 and attention-wide
1,036,984,320 full-vocabulary comparisons, zero violations/diagnostic bit
differences, unchanged gates. Layerwise128 and original32-ID MTP API cap16,
RNG/reset/reject/custom-stop/cap48 fixtures pass. Journal87, actual raw
r4-packed640-gates-40c-* copied and independently re-collected on controller.
This qualifies correctness only. Next same-config layerwise4096 full-request
A/B/A switches only actual old/packed linear.hip snapshots between builds.
Saved baseline stays untouched. Source metadata remains the actual build base.


### 2026-10-05 packed640 full-request result: no promotion
Paired source52d7b62/dirtytrue, native0 at07:27:18UTC, all512IDs/RNG/acceptance
exact.4K A93.631/91.215s versusB93.489s,meanA/B0.988601;16K A268.791/267.035s
versusB267.740s,meanA/B1.000648. PP mean-A-time/B-time0.999198/0.999419:
the component gain does NOT establish a real PP or full-request gain.
Packed dispatch/kernel reverted from production; strengthened tests retained,
implementation remains reproducible in Git52d7b62 and exact runs artifacts.
Journal88, actual copied six raws/manifests/IDs under r4-packed640-ab-20261005.
The saved baseline stayed unchanged. Ordinary executable rebuild to restored
canonical kernel follows this source commit; no code-performance claim retained.

A deeper read of the existing layerwise trace separates the broad Q4 N8 label
by output rows inferred from grid shape: M10240 sums5.338s, M2560 4.292s,
M640 3.874s, M6144 1.384s, M320 .670s, M12288 .465s (Q4_0).
The label aggregates different projections; M2560 cannot isolate K640 from
other input widths. No causal cache/occupancy explanation is proven for the
flat result. Raw analysis r4-profile-layerwise-q4-grid-breakdown.json.
A later large-output N16 experiment must benchmark actual M6144/10240/12288,
not generalize the earlier small-matrix N16 rejection to those untested shapes.
Immediate next bounded experiment: defer checks only within pure attention
multiply/add chains, retaining end-of-chain sticky error and output publication.
No fastmath/FMA/order/tolerance changes; strengthened overflow and old-bit gates
are mandatory. Candidate/test artifacts are prepared but not active yet.


Deferred attention checking component A/B/A native842/844/845 passes, source22b9f0e/
dirtytrue.38cases/352queries perGPU,2162688 exact-N1 and CPU elements/device,
maxCPUabs1.19209e-7/maxratio.00036923;25device rejects,180host rejects and6sticky
checks/device pass. Cross-build snapshot76cases/4336640values including padding
matches exactly. Added Inf-plus-opposite-sign and finite-product/sum-overflow
tests preserve bit2 and unchanged final output. Candidate changes only batch
QK/PV pure additive chains: nonfinite values cannot become finite under later
adds; score/PV end checks still report before stream-ordered output publication.
Failed scratch is provisional; valid arithmetic/order/noFMA remain unchanged.
All24 resident API coordinates beat both controls,ratio1.03170–1.24901,
median1.10379; B3 near1.24 at long visibility,B8 long near1.03–1.04.
Journal89, r4-attention-deferred-*; snapshot remains onGPU host. No model or
fullrequest speed claim yet. Timings are optional --benchmark after capture/compare.

Large-projection N16 Q4 also regresses: native853/857/859 all0,
548cases/574166786 exactGPU values/device,184516CPU samples,21rejects/546graphs.
K2560,M6144/10240/12288,full16 multiples:0/24 beats controls,median.877386,
range.864493–.902627. Candidate removed, extra large-matrix tests retained.
Legacy raw source physical_tile_columns8 describes the default baseline; the
candidate's selected physical16 geometry is documented by its retained patch.
Raw values were not retagged. Initialjob848 ran the old fixture after a rejected
edit and is excluded; actual large-shape baseline is a1b. Journal90.

Only attention deferred checking is under model qualification now: detached
PID1431009,ROOT/runs/r4-attention-deferred-model-22b, selected strict build,
short/wide/layerwise128/MTPAPI plus six targeted CTests. No new full39 claim;
the last full39 run belongs to packed-kernel qualification (since reverted).


Attention deferred-check model gates completed native0 at08:31:46UTC,
source22b9f0e/dirtytrue,detachedPID1431009. Strict selected build,short29,798,400
andwide1,036,984,320 vocabulary comparisons pass with zero violations and
diagnostic bit differences; wide maxabs/maxratio0. Layerwise128 and original
MTP32 IDs/RNG/reset/reject/custom-stop/cap48 pass. Six targeted CTests passed
in72.61s (quant,sampling,speculative,CLI,attention oracle,attention result parser).
This is NOT a new full39 run, nor independent HF or a model-speed qualification.
Journal91, actual raw r4-attention-deferred-model-22b-* copied and short/wide
strict collectors independently rerun on controller. Next matched4K/16K+512
A/B/A changes only actual attention.hip snapshots, both withlayerwise4096.


### 2026-10-05 deferred attention checks: paired full-request qualification
A/B/A completed native0 at09:06:50UTC, source53d1153/dirtytrue, journal92.
Both variants use explicitlayerwise4096, frame1024, slots112, capacityprompt+1024;
only attention.hip changed, builds outside timing, preserved baseline unchanged.
4K+512: A93.663597/91.159666s, B91.140152s; PP78.853648, TG13.037670.
16K+512: A267.116887/267.692213s, B260.903173s; PP73.315235, TG13.652910.
PP mean-control time ratios1.019943/1.020554, candidate faster than both PP
controls. Request ratios1.013951/1.024919, but4K advantage over fastest control
only0.019514s: noise-floor full-request gain, not a robust4K speedup claim.
All512 IDs, acceptance, proposal/decision RNG and pending IDs exact pertriplet.
Six actual514-row logs, manifests and journal byteprefix independently checked
on controller. Ordinary runtime restored to candidate. This is a bounded
self-parity/speed qualification, not independent HF or llama.cpp victory.
Pinned donor remains faster(~57/~144s). Next: bounded two-GPU PP pipeline;
no precision, routing, QSA or tolerance changes. Full PLAN remains open.


### 2026-10-05 bounded host prefill coordinator prerequisite
Added one persistent producer plus caller consumer, exactly two borrowed slots,
ordered1..4096 windows, backpressure, callback failure propagation, cancel/drain
and destructor join. Callback must drain its own HIP/DMA before returning or
throwing; this coordinator only owns host metadata. Not integrated into Session.
Target strict Release:84 cases/363984 checks/16 rejects,20 repeated CTests pass.
ASan+UBSan with leak detection passes the same suite. Warmed64x17 jobs allocate
zero C++ new calls across all threads; metadata184B excludes stack/runtime and
payload. Payload ownership, order, reuse, full-queue cancellation, active callback
drain and first/middle/last failure are tested. No device or speed qualification.
TSan linking repaired by disabling test allocation overrides explicitly; runtime
cannot start due incompatible ASLR layout. No security settings were changed,
and no race-sanitizer pass is claimed. Series native66 is this limitation,
not a passing series status. Journal93/sourceadcd7f4dirty, actual log
runs/r4-prefill-pipeline-sanitizers-adc2.log. Reproduce normal suite:
cmake --build build --target prefill-pipeline-test; ctest --test-dir build
-R '^prefill-pipeline$' --output-on-failure --repeat until-fail:20.
Next integrate per-GPU mutable host frames and chronological two-stage PP,
then model/state/reuse gates and full requests. Full PLAN remains open.


### 2026-10-05: opt-in two-GPU prefill pipeline correctness

The existing 24/24 layer split now supports chronological subwindows with one
persistent GPU0 producer and the calling GPU1 consumer. Each stage owns its
mutable routing, pinned DTO and transfer buffers. Two bounded host handoff slots
transfer completed residual ownership; all six activation vectors have equal
capacity, so swaps do not allocate. No additional explicit GPU Buffer payload is
created. Default serial scheduling is unchanged. The initial pipeline limit is
a logical window of 4096 tokens, with an explicit subwindow size in 1..capacity;
CPU hybrid and tracing remain disabled for this path.

Absolute GDN/PLE/QSA chronology, original-rank arithmetic, all target tap rows and
full-window route diagnostics are preserved. Publication happens after both
stages finish. On failure, the producer and both GPU streams drain before reset
or owner release. Logits also use transactional staging without checkpoints.
The worker must never share the original PLE transfer buffer with GPU1.

Qualification used source 48f3d14ca08bb8a1dd7a02fc0e2a381fd60dc12a, dirty=true:
- Serial stage extraction passed the existing 128-token model fixture
- Pipeline 32/8 and 128/17 passed 125,642,240 and 441,443,840 comparisons
- Pipeline 4096/2048 passed 13,494,576,640 comparisons, including full vocabulary,
  hidden taps and repeated references. All three had zero gate violations,
  diagnostic bit mismatches, maximum absolute error and gate ratio
- Eight injected failure/reuse cases passed: first/middle/last windows on both
  stages, plus last-window failures without checkpoints. Previous published
  logits/taps/counters survive; execution and the checkpoint getter require reset
- MTP API original 32 IDs/RNG/acceptance, repeat/reject/custom-stop/capacity gates
  passed. The 2051-token carry test compared ordinary, serial layerwise and
  pipelined windows (1025/1025/1, pipeline 512), with 32 identical outputs
- Six invalid CLI options and help passed before model load
- Full strict build and 40/40 CTests passed in 1157.10 seconds

The first fault fixture incorrectly called checkpoint_state() on an invalidated
Session, contrary to its existing contract. That series exited 1 after the two
small model passes. The fixture was corrected to require the getter rejection
while independently checking preserved published bytes; no runtime relaxation.
The expanded series exited 0 at 10:36:56 UTC. Journal94 retains actual manifests,
footers and source revision; previous 93 records remain byte-identical.

Reproduction inside the established container:
- core-prefill-layerwise-test MODEL 4096 2048
- core-prefill-layerwise-test MODEL --pipeline-fault STAGE WINDOW [--no-tap]
- core-mtp-runner-test TARGET SIDECAR CLI32_IDS --pipeline
- core-mtp-runner-test TARGET SIDECAR --large-carry
- core-mtp-run ... --layerwise-prefill 4096 --prefill-pipeline 2048

Artifacts: ROOT/runs/r4-pipeline-expanded-48f/ and r4-pipeline-48f-{32x8,128x17}.
The latter are the successful portions of the first series. Host-only coordinator
sanitizers and their TSan runtime limitation remain journal93. This is bounded
same-engine correctness, not independent HF proof, occupied128K or a speed win.
Next: one-binary trace-off A/B/A at 4K/16K +512 outputs, pipeline 0 versus 2048.


### 2026-10-05: two-GPU pipeline full-request result

The trace-off A/B/A series completed natively at 11:12:52 UTC. One unchanged
runtime binary, source dc60e0f56da9d2b10bdbb5de845491470da8d0eb / dirty=true,
compared serial layerwise4096 against pipeline2048 within the same logical4096.
Capacity was prompt+1024, slots112, frame1024, attention tile8, primary sampling
(seed12345, temperature1, top-p.95, top-k20), fresh processes, no prefix reuse.

4K+512: A92.547317/92.184304s, B84.422057s; B PP88.353420, TG13.425791.
16K+512: A265.568701/260.969029s, B229.376431s; B PP85.327488, TG13.677163.
Candidate PP and full requests beat both controls at both sizes. Mean-control
request speedups were1.094096x/1.147759x; PP time ratios1.135659x/1.170804x.
TG differences are measurements, not evidence of a changed decode kernel:
prefill scheduling can change cache residency, and warmness is not isolated.

All512 output IDs, acceptance, proposal/decision RNG and pending IDs matched in
each triplet. Protected saved baseline stayed unchanged. Six actual514-row logs,
manifests, one runtime digest and the previous94 journal byte prefix were checked
again on the controller. Journal95 contains full evidence; raw directory is
ROOT/runs/r4-pipeline-ab-20261005. The selected configuration remains opt-in.
Pinned llama.cpp still wins the full requests (~57/~144s); full PLAN is open.
Next: separately observed VRAM and refreshed diagnostic GPU phase profile.


### 2026-10-05: pipeline memory and phase diagnostics

Separate observed16K+512 completed with the same full token/RNG trajectory as
the unobserved pipeline candidate. Fresh HIP PCI mapping was05:00.0/08:00.0.
3227 samples at0.1s over324.30s (load/request/cleanup) found max driver usage
12,701,908,992 /16,393,617,408 bytes and minimum free
4,461,182,976 /769,474,560 bytes, from17,163,091,968 bytes per device.
These are sampled global extrema, not exact instantaneous or per-phase peaks.
Journal96, ROOT/runs/r4-pipeline-vram-dc6; source dc60e0f dirty.

Diagnostic profile source6b78bb7 dirty completed natively at11:32:29 UTC.
The ordinary runtime binary was unchanged. Annotation32 and untraced/traced
4K+128 outputs/RNG/acceptance matched. Tracing perturbs latency: the untraced
request was55.019s, traced64.428s; neither is a new512-output speed result.

The traced target PP range was52.720290s, with1,200,175 kernel events and
42.078131 summed kernel seconds. Per-GPU kernel unions were21.303615/20.767612s,
union across both38.872606s, actual simultaneous kernel overlap3.198621s.
First-to-last kernel envelopes overlapped21.508125s, but envelopes INCLUDE
idle gaps; this is not21.5s of simultaneous GPU work. No boundary-crossing events.
Q4_0 tiled projections remain37.80% of summed kernel time (15.904007s);
attention21.02% (8.844621s), GDN1.682261s, dense tiled1.651964s.
H2D duration sums4.771855/4.720751s are not wall fractions or bandwidth:
this CSV does not include copy byte counts. The trace does not establish a
single causal explanation for the gaps. Journal97 retains complete summaries.
Artifacts: ROOT/runs/r4-profile-pipeline-20261005 and
r4-profile-pipeline-{summary,envelopes}.json.

Next bounded allocation hypothesis: Device currently reserves1024*248320*4
bytes for head logits on EACH GPU, although Matrix::apply already projects
at most128 rows per qualified tile. Queue a completed tile copy before reusing
a128-row output buffer. This would save889,978,880 bytes per GPU at frame1024,
without changing output vocabulary, arithmetic, host publication or head HC.
This is a SOURCE-DERIVED opportunity, not implemented or a measured saving yet.
Qualify bytes/parity and update versioned memory evidence before using the
headroom for larger logical pipeline windows. Full PLAN remains open.


### 2026-10-05: bounded128 device head-output staging

Device logits now hold min(frame,128) vocabulary rows. The head still executes
the same qualified projection tiles and computes every requested output row.
Each tile's D2H copy is ordered on the same stream before that buffer is reused;
the caller completes/checks the stream before host validation and publication.
Head HC, weights, precision, output vocabulary and host capacities are unchanged.
N1 and all constructor frames <=128 retain their previous buffer sizes.

The actual SessionDeviceMemory.head_logits_bytes field is included in workspace
and independent live-Buffer accounting. With frame1024, each GPU now owns
127,139,840 head bytes instead of1,017,118,720. Matching old/new loaded ledgers
confirm an EXACT889,978,880-byte reduction in both owned/workspace bytes on each
GPU; all other categories and buffer counts match. This is allocation evidence,
not a sampled driver-peak or speed claim.

Wide, long and attention fixtures now emit protocol2 with explicit head bytes.
Their readers retain the ORIGINAL protocol1 memory floors and accept archived
protocol1 records without rewriting them. Protocol2 requires the exact new
capacity; mixed versions, missing/false head data and undersized workspace reject.
No numerical tolerance changed. Five new test methods cover both versions and
failure cases across all three reader families; actual old/new attention logs
also passed their corresponding strict collectors.

Qualification source8ea78b03e3a9e4ceac1b580dbd9863eeb6427785 / dirty=true:
- Native wide786,677,760 and attention1,036,984,320 logit comparisons passed
  with zero gate violations and diagnostic bit mismatches
- Serial256 and pipeline256/subwindow129 each passed862,512,640 comparisons
  (including hidden taps and repeated references), with zero error/bit differences
- MTP API and2051-token carry passed the established32 IDs/RNG/acceptance gates
- Default capacity131072 memory fixture and strict collector passed; this remains
  allocation/ownership evidence, not occupied128K qualification
- Full strict build and41/41 CTests passed in1151.11 seconds

The series exited0 at12:50:59 UTC. Journal98 includes actual manifests/footers
and allocation deltas; previous97 records remain byte-identical. The controller
re-ran actual collectors. Artifacts: ROOT/runs/r4-head128-gates-8ea/.
The long emitter's protocol2 schema is covered by reader tests; no new native
4K/16K long-reference fixture is claimed for this slice.

Reproduce the existing container build with the actual source revision, then
core-prefill-wide-test MODEL; core-prefill-attention-test MODEL;
core-prefill-layerwise-test MODEL 256 [129]; core-mtp-runner-test TARGET SIDECAR
--large-carry; and the full CTest suite.
Next: observed16K+512 driver VRAM, then stage-sized buffers and larger logical
pipeline windows if the measured budget permits. Full PLAN remains open.


### 2026-10-05: bounded-head driver VRAM observation
Source58f3bbd dirty, separate16K+512 run, native0 at13:07:51 UTC. All512 IDs,
acceptance, RNG and pending matched the previous pipeline candidate.
Fresh HIP mapping05:00.0/08:00.0;3171 samples at0.1s over318.735s including
load/request/cleanup. Max used11,812,704,256 /15,504,785,408 bytes; minimum free
5,350,387,712 /1,658,306,560 bytes, total17,163,091,968 per device.
Observed max-usage differences from the previous pipeline run were
889,204,736 /888,832,000 bytes. These sampled driver differences are distinct
from the exact889,978,880-byte owned allocation reduction; runtime overhead and
sampling can differ. No exact instantaneous-peak or speed claim.
Journal99 and ROOT/runs/r4-head128-vram retain actual evidence. The old98 byte
prefix and actual copied request/observer logs were verified on the controller.

Next bounded change: separate logical pipeline/tap capacity from per-stage
scratch. Retain a2048-token stage but permit a logical window up to16384.
At frame1024, compared with the current logical4096/stage2048 allocation,
shrinking physical stage storage from4096 to2048 saves215,777,280 bytes/device;
growing two root taps from4096 to16384 costs1,006,632,960 bytes on GPU1.
Using observed free memory gives a SOURCE-DERIVED estimate867,450,880 root
bytes left. This is not a fit guarantee or implemented result; it needs actual
constructor/ownership, full-logit/tap/continuation/MTP gates and observed fit.
The longer fixture must avoid keeping two redundant full-vocabulary reference
arrays at16384 rows; retain one full N1 reference and validate the old1024 path
against it before comparing the candidate. Numerical gates remain unchanged.

### 2026-10-05: stage-sized pipeline and logical16K correctness
The opt-in pipeline now accepts logical windows up to16384 with stage tokens
up to4096, bounded by the coordinator's4096-window limit. Physical host/GPU
routing, activation, DTO and contribution owners use the stage row capacity;
published target taps retain every logical row. Serial allocations are unchanged.
MTP capacities above4096 require the pipeline. No arithmetic, precision, weights,
QSA budget, sampling or numerical gate changed. Default remains off.

Detached PID1504300, source d544b6ee45158f0bc92c2500da409899224a25f7/dirtytrue,
completed native0 at2026-10-05 15:06:07 UTC. Full strict build and41/41 CTests
passed (1156.08s). Seven CLI rejects, physical256x129 ownership, both last-window
stage failure/publication/reset cases and long-capacity MTP API passed.
Full16K/2048 compared29,497,640,960 values cumulatively, including repeated
full-vocabulary rows and hidden taps, with zero absolute error and bit mismatches.
The small256x129 fixture compared862,512,640 values with the same zero result.
This is same-engine N1 parity, not an independent HF reference.
Actual16387-token MTP carry compared logical4096 and16384 at stage2048:
all32 outputs, acceptance and RNG agree. Capacity boundaries, custom stop,
invalid-owner/reset/continuation tests retain their existing scope.

Native16K resource ledgers show stage_rows2048, contribution209715200B/device,
originalQ8 5898240B/device and root taps671088640B each. Logical rows are not
confused with physical stage extent. The fixture retains one full N1 vocabulary
reference rather than a duplicate second16GB array, while checking the old1024
path against it and preserving every hidden tap. Host swap was observed during
the large correctness fixture; no swap settings changed. Its timings are NOT
performance evidence. Normal512-output MTP runs are measured separately.

Actual raws/manifests: runs/r4-pipeline16k-gates-d544/. Journal100 and its old99
byte prefix were verified after copy to the controller. Protected baseline intact.
Next: same-binary A/B/A logical4096 versus16384, fixedstage2048 at4K/16K+512,
then driver VRAM observation. Current best qualified speed remains dc60e0f
84.422/229.376s; this correctness slice alone claims no speed or llama.cpp win.

### 2026-10-05: logical16K pipeline matched full-request measurements
Native A/B/A PID1520517 completed0 at15:37:12 UTC. Source
eaa778c00133891d3882f83942487da964108114/dirtytrue, one unchanged binary.
Both configurations use physicalstage2048/frame1024/tile8/slots112; logical
capacity A4096 versus B16384, fresh4K/16K prompts plus512 real output tokens,
capacityprompt+1024, primary sampling seed12345/temperature1/top-p.95/top-k20.
Load excluded; no prefix reuse; expert-cache warmness not isolated.

| prompt | A1 request | B request | A2 request | B PP | B TG |
|---|---:|---:|---:|---:|---:|
|4K|86.191997s|83.416036s|85.046992s|89.242071|13.620598|
|16K|234.192985s|183.372317s|231.931921s|112.303156|13.634030|

For16K, mean-control/full-request ratio1.270979 and PP-time ratio1.331639;
A PP84.069081/84.601701. Both request and PP beat both controls. The larger logical
window avoids repeating pipeline fill/drain and target/draft call boundaries.
At4K both capacities execute the same two pipeline subwindows: recorded
1.026415 request/1.015710 PP ratios are not evidence of a structural4K gain.
TG variation is not a decode-kernel improvement claim. All512 output IDs,
acceptance/RNG/pending agree across each triple; protected baseline untouched.
Pinned llama.cpp still finishes around57/144s, so full speed goal remains OPEN.

Actual six514-row logs/manifests, one executable digest, terminal status and old100
journal byte prefix verified on the controller; journal101. Raw directory:
runs/r4-pipeline16k-ab-20261005/. Explicit opt-in fast16K setting:
--layerwise-prefill16384 --prefill-pipeline2048. Default has not changed.
Separate driver VRAM observation PID1525727 is running; no new memory peak claim.

### 2026-10-05: logical16K/stage2048 observed driver VRAM
Separate PID1525727 completed native0 at15:43:58 UTC; sourceeaa778c/dirtytrue,
same executable SHA25653d77552893a77728d025c6c1346286f03b40329ad7e6c8316f97cda22cfd6fa
and same16K-B request as journal101. All512 IDs/acceptance/RNG/pending agree.
Fresh HIP PCI mapping05:00.0/08:00.0.2710 samples at0.1s over272.362s,
including load/request/cleanup: maxused11,596,705,792/16,294,354,944B;
minfree5,566,386,176/868,737,024B, total17,163,091,968B each.
These are sampled global driver extrema, not exact instantaneous peaks or a
new timing comparison. Journal102, actual copied raw request/observer/manifests
and the old101 byte prefix verified. Artifacts runs/r4-pipeline16k-vram/.

Next bounded configuration experiment: physicalstage4096 within logical16384.
Source-derived extra stage ownership215,777,280B/device leaves approximately
652,959,744B root headroom based on the sampled2048 run, not a fit guarantee.
A4097-token all-row/tap fixture will cover a full4096 stage plus partial carry,
then a separate same-binary16K+512 A/B/A will compare stage2048 versus4096.
No runtime arithmetic change or default promotion; qualify correctness before
timing and retain the current2048 path if4096 does not improve full requests.

### 2026-10-05: stage4096 within logical16K
The existing configuration (no runtime arithmetic/source changes) was qualified
with the FULL16384x4096 fixture, rather than the initially planned4097 case:
the fixture only accepts explicit bounded row cases.29,497,640,960 cumulative
logit/tap comparisons, zero errors/bits, all continuation/reset gates pass.
16387+32 CLI carry at stage2048/4096 gives identical IDs/acceptance/RNG/pending.
Strict selected build and two targeted CTests pass; latest full41-test regression
remains journal100 on the same runtime sources. Do not call this a new fullCTest.

DetachedPID1529544 native0 at16:57:54 UTC, sourcec3d7625 dirty.
SSH tunnel unavailable16:12..16:32; same process resumed observation,
no duplicate workload or network changes. Actual A/B/A16K+512:
A2048:186.827339s,PP110.805481,TG13.115193;
B4096:182.044469s,PP113.318528,TG13.641648;
A2048:185.103706s,PP111.225462,TG13.519534.
Mean request1.021539x and PP1.020749x; B beats both controls, all512 trajectories
exact. This modest result costs215,777,280B additional owned stage memory/GPU.
The real MTP request fits;4096-stage driver peak was not sampled. Default remains
unchanged; keep2048 as the lower-memory option. No4K benefit or llama.cpp win claimed.
Journal103, actual manifests/raws and old102 byte prefix verified on controller.
Raw roots runs/r4-stage4096-gates-c3d/ and runs/r4-stage4096-ab-20261005/.
Next bounded experiment targets4K: stage1024 versus2048 within logical4096,
to test whether four pipeline subwindows outweigh additional expert staging.
Qualify full4096 logits/taps and4099-token carry before paired512-output requests.

### 2026-10-05: stage1024 for4K rejected as a performance change
Existing configuration only, source3cb6749 dirty. NativePID1541061 completed0
at17:33:27UTC. Full4096x1024 passed13,494,576,640 cumulative logits/tap
comparisons with zero errors/bits;4099+32carry matches stage2048 exactly.
Strict selected build and two targeted CTests passed. Latest full41CTest
remains journal100 on unchanged runtime sources, not a new full regression.

Same-binary4K+512 A/B/A (logical4096, stage2048/1024/2048):
A1 request85.297698s,PP89.052903,TG13.002401;
B request85.556014s,PP85.555633,TG13.562085;
A2 request86.214582s,PP88.090998,TG12.866523.
All512 IDs/acceptance/RNG/pending match. B does NOT beat both request controls,
and PP is worse than both: mean-control PP-time ratio0.965973.
Mean request1.002339 is not a stable win; TG variation masks slower PP.
Do not promote stage1024 for4K. Existing opt-in support remains correct;
no runtime code/default changed. Stage2048 remains the measured4K choice.
Journal104, actual copied logs/manifests and old103 prefix verified.
Next diagnostic: completed-phase16K profiling at logical16384/stage2048,
with matching traced/untraced128-output trajectories and ordinary binary intact.
Use separate profile target/source stamp and fresh artifact paths, no speedclaim.

### 2026-10-05: current16K completed-phase diagnostic
PID1551076 completed native0 at17:54:17UTC. Profile source9f6f2be dirty;
ordinary3cb runtime digest unchanged. Annotation32 and traced/untraced128
trajectories pass. Untraced/traced16K+128 request155.971536/213.622188s:
tracing materially perturbs the run; none of these timings is512-output promotion.
Journal105, actual copied request logs/manifests/summary and old104 prefix verified.
Full CSVs remain on GPU host in runs/r4-profile-logical16k-20261005/mtp16k128-trace/.

Target PP completed197.308246s under trace,6,045,736 kernels.
Summed kernel183.679131s; perGPU unions92.147192/91.473480s;
union across both160.235462s, actual simultaneous kernel intersection23.385210s.
Zero events cross the completed phase boundaries. Sums overlap, not a wall breakdown.
Largest kernel sums: Q4_0 N8 matrix63.651312s(34.65%), attention40.208678s(21.89%).
Radix select and histogram each launch1,375,968 times, sums6.162286/5.717471s.
Both together are2,751,936 launches, a concrete launch-overhead candidate.
H2D sums18.922865/18.954299s; no byte counts in copy CSV, no bandwidth claim.
MTP verify46windows:10.992098s hostwall,4.542691s summed kernels; no decode claim.

Next bounded implementation: a borrowed-buffer QSA selector for up to8 chronological
queries, batching the EXISTING integer radix phases across the query grid.
Keep rank keys, signed-zero/tie order, eight radix bytes, whole blocks and actual tail.
Per-query constructor-owned histogram/state/candidate storage; bounded score rows.
Keep the existing N1 entry point unchanged in semantics. Validate extents/strides
and all aliases before enqueue. Separate scratch sorting/validation from final
batch publication so any sticky error prevents output publication across queries.
First qualify component IDs/counts against CPU oracle and old serial launches,
including mixed direct/radix boundary, tails, signed-zero/subnormal/ties,
NaN/Inf/sticky flags, invalid descriptors, guards, reuse and graph capture.
Only after component timing/parity integrate into attention_queries, preserving
all current full-model gates, memory ownership and matched request measurements.

### 2026-10-05: bounded batched QSA selector component
New launch_qsa_select_batch in hip/qsa_select.{cuh,hip} batches up to8 chronological
queries using the EXISTING integer radix algorithm, rank bits, signed-zero ties,
whole-block expansion and actual tail. Typed borrowed views declare extents and
strides; each query owns histogram/state/candidate rows. Sort/validation writes
scratch first; a separate kernel publishes ALL outputs only after a clean sticky
flag. Invalid descriptors enqueue nothing. No allocation, device queries or sync.
The scalar entry point preserves its semantics through non-batch instantiations.

Strict target build and core-qsa-select-batch pass on both gfx906 devices.
Each GPU:525 input cases,21,906,304 compared IDs/counts including padding,
23 host rejection cases (empty captured graphs),24 NaN/Inf cases,3 sticky cases,
12 graph replays; score-readonly and output/scratch/outer canaries pass.
Includes direct/radix and partial-grid boundaries, chronological tails,
random/tied/signed-zero/subnormal/extreme finite scores, reset/reuse.
Existing62-row core-qsa protocol and strict collector pass; targeted CTests3/3.
This is NOT a full-model qualification or new full41-test regression.

Source94579b9 dirty. Initial detachedPID1570573 completed native0 at18:11:49UTC.
Repeat job1791224178680-1196 passed with17-digit metric serialization;
same correctness and28/28 timing coordinates beat BOTH serial controls.
Resident20-repeat completed-GPU-event A/B/A ratios:
2-query min/median/max1.418483/1.671893/1.685357;
8-query4.348445/6.245553/6.442899.
These are selection-component timings, including launch gaps, not model speed.
Actual logs/collector and old105 journal byte prefix verified; journal106.
Reproduce: build core-qsa-select-batch and run on the two-GPU host; core-qsa MODEL.

Next integrate in Session::Layer::attention_queries only for count>1.
Keep serial index scoring arithmetic, but write each query into its own fixed32768
score row in f(16), then batch-select into existing ID/count rows. Grow f(16)
only when its existing backing storage is smaller than tile*32768; frame1024
already has enough. Histogram/state/candidates scale by attention_tile; tile1
keeps original capacities. Add explicit backing allocation evidence and full-model
N1/batched/logit/tap/MTP/failure regressions, then trace-off A/B/A full requests.
No model integration or inference-speed promotion is claimed by this component.

### 2026-10-05: batched QSA selection integrated and numerically qualified
Session::Layer::attention_queries now computes unchanged causal scores into
private rows of existing f(16), then selects up to eight queries per radix
grid. Attention tiles through128 remain supported using selector subgroups;
the final singleton uses the scalar entry point. Public ID/count storage still
covers the full attention tile. Per-query score, histogram, state and candidate
scratch is constructor-owned and reused in stream order. No hot allocation,
score arithmetic, selected rank order, precision, budget or tolerance changed.

At frame1024 the existing50,331,648-byte score backing already fits eight
32768-float rows, so it does not grow. Histograms/state/candidates become
262144/1024/32768 bytes each per device for tile8. Matching native old/new
attention ledgers prove exactly258,944 more owned/workspace bytes per GPU,
with all other categories and Buffer counts unchanged. Larger attention tiles
do not grow this selector prefix beyond eight rows. This is owned allocation
evidence, not a new sampled driver-peak measurement.

Compiled source54a4dbf3792145e3483ef2f61a2840897c17843f/dirtytrue.
Native qualification completed0 at21:35:28UTC, final resume PID1608246.
Two earlier attempts stopped BETWEEN phases due an external Telegram build
and a later RAM preflight while its linker was running. Completed phases were
retained; no numerical failure, relaxed memory guard or duplicated heavy run.
Full strict build and41/41 CTests pass in1154.97 seconds.
Native evidence includes:
- wide786,677,760 and attention1,036,984,320 compared values, zero violations/bits
- serial256 and pipeline256x129 each862,512,640 exact cumulative comparisons
- attention tiles16 and128:125,642,240 and441,443,840 exact comparisons
- full16384x2048:29,497,640,960 cumulative logits/tap comparisons, zero errors/bits
- both pipeline failure/publication/reset cases and actual16387-token MTP carry
- MTP API/long resource configuration,2051 carry and default capacity131072 memory
- actual strict wide/attention/default-memory collectors on GPU host and controller

Same-engine N1 parity is not an independent HF oracle; default capacity131072
is not occupied128K, and custom stop is not observed natural EOS. Large reference
fixture timing includes comparisons and host memory pressure, not speed evidence.
Journal107, actual copied manifests/raw footers and old106 byte prefix verified.
Artifacts: runs/r4-qsa-batch-model-54a/, with original/resume/resume2 terminal records.
Next: trace-off source-switched A/B/A4K and16K+512, same primary sampling,
stage2048 and logical4096/16384. Build only outside timing, preserve saved binaries.
Source-off snapshot runs/qsa-batch-session-off-54a.hip; driver r4-qsa-batch-ab.py.
No inference-speed promotion is claimed until those complete requests pass.

### 2026-10-05 exact batched QSA: paired full requests (journal108)
Source611895f dirty, trace-off source-switched Session A/B/A; same QSA
component library, scalar per-query selector versus bounded batched selector.
Both variants use frame1024/tile8/stage2048, logical4096/16384 respectively.
Full primary-sampling512 trajectories, acceptance/RNG/pending match exactly.
4K controls86.194940/85.825983s versus81.749504s; candidate PP92.904993,
TG13.568888. Mean-control request speedup1.052122; PP-time speedup1.049236.
16K controls187.302567/185.450542s versus175.970231s; candidate PP117.911638,
TG13.804565. Mean-control request speedup1.059137; PP-time speedup1.059900.
Candidate beats both controls for PP and complete request at both lengths.
Native PID1628582 completed0 at22:09:56UTC; candidate restored, protected
baseline unchanged. Actual six logs/manifests/input IDs copied and validated
on controller; journal old107 byte prefix preserved. Builds and load excluded
from request times, no prefix reuse, expert-cache warmness unknown.
This qualifies the bounded workload gain; it is not independent HF validation
or a llama.cpp win (historical matched donor still~57s/~144s).
Raw reproduction driver:runs/r4-qsa-batch-ab.py;
validator:runs/r4-qsa-batch-paired-record.py; raw:runs/r4-qsa-batch-ab-20261005.

User-directed next task: inspect Strata issue641 and related gfx906 changes
before another speculative optimization. Read-only source clone is pinned at
6f32ec070f23ced9f50e704d854d775da52591ab. Its published MI50 benchmark uses
Coder IQ1_M, fully VRAM-resident experts, MTP4 plus suffix drafting, greedy256,
int8 hybrid KV; direct comparison with our Q4_0/MTP2/sample512 is invalid.

### 2026-10-06 bounded GPU-resident residual qualification (journal 109)
Added opt-in SessionConfig.prefill_residual_device / --prefill-residual-device;
it requires the existing bounded pipeline and remains OFF by default.
A stage keeps one private FP32 residual backing and reuses the existing separate
frame scratch. HC, GDN, QSA, experts, CPU routing, FFN staging, reduction order,
finite-error checks, and whole-call publication remain unchanged.
The existing HC combine validates every operand/result with the sticky flag;
resident copies retain the completed error checks. Stage exit still validates
the downloaded residual. No new kernel or precision/weight conversion.

Native source f4d5e6166e8afb841422cdc7eaf58dc273bafb25 dirty:
- serial32 / resident32-stage8: 125,642,240 comparisons each, exact
- resident256-stage129: 862,512,640 comparisons, exact
- resident4096-stage2048: 13,494,576,640 comparisons, exact
- stage0 failure with taps and stage1 failure without taps preserve publication,
  including new diagnostics; reset permits correct reuse
- MTP API32/reset/custom-stop/capacity48 and 2051-token carry match prior IDs/RNG
- full strict build and all 41 CTests pass (1157.78 seconds)
- three production CLI preflight rejections pass without model files
- same target configuration at capacity17408/stage2048 proves exactly
  83,886,080 additional owned/workspace bytes and one buffer per GPU

SessionLayerwiseTransfers reports actual submitted residual, FFN, router and
routing-metadata extents and explicit loop-check counts for committed calls.
Failure/rejection retains publication; reset clears it. Head, PLE-internal and
expert transfers/checks are excluded. These are not hardware PCIe counters or
elapsed wait times. Their closed-form extents are asserted in native fixtures.

Short native PID1660234 completed0. Long PID1664601 finished all native phases
successfully; its final parser expected an obsolete CTest summary spelling and
exited1 after the native build exited0. Recovery verified all 41 individual
Passed rows and the current summary, retained the original failure receipt,
and did not rerun or alter test results. Finalizer completed0; journal109 and
actual logs/manifests were validated on the controller with the old108 byte
prefix intact. Runtime SHA256:
dc980a175142abdc4725bc077e5de02d5e2ead8e6746aa3e4d69174644d63508.
Protected historical baseline remains unchanged.

Reproduce the large native gate:
core-prefill-layerwise-test MODEL 4096 2048 8 --resident
Memory delta: core-prefill-layerwise-test MODEL --resident-memory
MTP path: core-mtp-run TARGET SIDECAR --sample --seed12345 --generate512
--capacity17408 --slots112 --prefill-chunk1024 --layerwise-prefill16384
--prefill-pipeline2048 --prefill-residual-device --attention-query-tile8
--ignore-eos --temperature1 --top-p0.95 --top-k20 TOKEN_IDS
(Separate option names and values when invoking the CLI.)

This is a bounded correctness/memory slice, not a speed promotion. Full16K
logit/tap qualification, independent HF, occupied128K and natural EOS remain
outside this result. Next is same-binary A/B/A4K/16K+512, changing only the
residual flag; retain it only with measured full-request benefit.

### 2026-10-06 resident residual full-request pair (journal 110)
Native PID1676011 completed0 at01:22:47UTC. Trace-off SAME binary
A1(off)/B(on)/A2(off), changing only --prefill-residual-device; actual inputs,
all manifests, source stamp and binary hashes validated. Source remains
f4d5e6166e8afb841422cdc7eaf58dc273bafb25 dirty, implementation commit0602367.
Logical4096/16384, stage2048, frame1024, attention8, slots112, output512;
temperature1/top-p0.95/top-k20/seed12345/ignore-EOS. Model load excluded.
No prefix reuse; expert-cache warmness unknown.

4096 prompt +512 output:
- 4k-A1: request83.464508964s, PP92.860259, TG12.984906
- 4k-B: request76.630826896s, PP105.181314, TG13.559101
- 4k-A2: request81.757773036s, PP92.883332, TG13.569619
Mean-control request speedup1.078040579x; PP speedup1.132543142x. Candidate beats BOTH controls for request and PP.

16384 prompt +512 output:
- 16k-A1: request178.951136696s, PP117.088573, TG13.095487
- 16k-B: request160.540751868s, PP133.168911, TG13.624024
- 16k-A2: request179.278323943s, PP116.444295, TG13.247354
Mean-control request speedup1.115696346x; PP speedup1.140481215x. Candidate beats BOTH controls for request and PP.

All512 output IDs, accepted counts, proposal/decision RNG draws and pending
last token match in each triple. Submitted residual H2D/D2H extents each fall
48x:16K each64,424,509,440 bytes ->1,342,177,280 bytes; additional resident
D2D128,849,018,880 bytes. These are API byte extents, not physical PCIe counters.
FFN, router and routing metadata extents stay unchanged; explicit loop-check
counts also match the implementation's formula. Resident adds80MiB per GPU.

Retain the opt-in candidate for these measured workloads; default remains OFF.
The full16K logit/tap reference is the next qualification step and is not implied
by matching512 generated outputs. Same-engine parity is not independent HF.
Pinned llama.cpp remains faster in total request (~57s4K/~144s16K); no victory
claim, no occupied128K or natural-EOS qualification. Both TG numerators here
are511; donor's is512. Full raw logs/manifests and old109 journal prefix were
independently validated on the controller.

### 2026-10-06 resident residual full16K qualification (journal111)
Native PID1680636 completed0 at02:10:53UTC. Same qualified f4d5e61 dirty
binaries, same model and frozen .02+.002*abs(reference) gates. Opt-in resident
residual, logical16384/stage2048/frame1024/attention8. Full all-vocabulary
logits and checkpoint taps:29,497,640,960 comparisons, maxabs0, maxratio0,
zero diagnostic bit mismatches. Includes occupied5, split calls, resets,
rejections and continuations; one retained full N1 reference. The separate
16387-token carry compares logical4096/16384 with resident enabled, matching
all32 output IDs, acceptance counts and RNG. Native durations2219.869s and
505.245s include fixture work and are NOT performance claims.

Actual raw logs/manifests, resident memory ownership and submitted-transfer
records were copied and independently validated on the controller; canonical
journal110 byte prefix preserved. Protected baseline and qualified ordinary
runtime SHA256 remain unchanged. This closes the resident candidate's full16K
same-engine numerical qualification, not independent HF, occupied128K or
natural EOS. Paired performance remains journal110:4K76.631s/PP105.181 and
16K160.541s/PP133.169. Pinned llama.cpp is still faster in total request.
Next: fresh annotated16K profile of the retained residual+QSA candidate,
with trace-off/traced128-output trajectory comparison and actual GPU timeline
unions. Do not infer bottlenecks solely from the pre-resident trace.

### 2026-10-06 retained-residual fresh profile (journal112)
Native PID1688034 completed0 at02:28:49UTC. Annotation-only build preserves
ordinary runtime SHA256dc980a...63508 and protected baseline. Exact32 smoke
and128-output trace-off/traced IDs/acceptance/RNG/pending match.
16K+128 ordinary133.434810s versus traced174.843363s; significant overhead,
not a speed comparison. Compiled f4d5e61 dirty, resident+batched-QSA enabled.

Actual completed target-PP range158.823253s;3,171,060 kernels, summed
169.882094s across both GPUs. Individual kernel timeline unions85.197480 /
84.657509s; union142.331615s, actual intersection27.523374s. No crossing
events. These are unions/intersections, not overlap of broad envelopes.
Top kernel sums: Q4_0 N8 matrix63.662384s(37.47%), sparse attention chunk
40.025397s(23.56%), GDN6.727816s, input validation/chronological bitonic sort
5.943099s, gather5.744329s, selected-scale preflight3.956389s. Index score
1.629774s, query norm/RoPE1.296272s; another QSA launch reduction is therefore
a smaller target than matrix/attention organization in this trace. H2D copy
sums16.496715/16.525057s, D2H0.395839/0.395995s. CSV has no byte column;
actual submitted byte extents are the separate Session diagnostics.

Next small candidate: distribute selected-scale preflight across multiple
CTAs/query rather than one CTA scanning all selected Q4 blocks. Keep the
existing separate global preflight boundary before any gather/decode, exact
invalid-bit classes and all-query failure-atomic publication. No math/precision
or selection change; N1 untouched. First require full component frozen gates,
cross-build output witness, failure/padding/graph checks and A/B/A; only then
model gates and full requests. Matrix planar packing remains a later measured
candidate, not a promise based on the top-kernel name.
