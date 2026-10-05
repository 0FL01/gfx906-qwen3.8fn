# Текущее состояние

## Задача (2026-10-05 01:45 UTC)

Завершить ВЕСЬ PLAN.md и превзойти llama.cpp в сопоставимых полных запросах.
Лично через SSH nc-lab, без engineering subagents. Document+commit+push каждый
законченный срез. R0–R3 закрыты; R4 speed/R5 dispatch/fullR6/R7/R8 ещё OPEN.
400–600PP/30–40TG и llama.cpp win НЕ достигнуты. Frozen gates/weights unchanged.

## Последний подтверждённый runtime

Sparse sampler paired job1791161686931-537 exit0/31m53, result pushed
daee1efcd6e83c115175284d7ce9512045298b8d; component350f8afe3f988af1a79a74f47e5f782f0c7051f7.
4K+512 A112.812/110.077s -> B109.526s (1.017521x),PP58.3219/TG13.0049.
16K+512 A334.585/333.669s -> B330.152s (1.012040x),PP56.1301/TG13.3572.
All512 IDs/acceptance/RNG/pending exact inside both triplets. Source2535cbf/dirtytrue,
explicit off/on sampler snapshots; capacity n+1024,slots112,chunk1024,tile8,
seed12345/temp1/top-p.95/top-k20/ignoreEOS. Load separate, expert warmness unknown.
Pinned donor dcd6854 same target/sidecar/headroom:4K~57s,16K~144s, still faster.
Donor QSA historical semantics are not an independent correctness oracle.

Current ordinary core-mtp-run is sparse-sampler2535cbf/dirtytrue (rolled attention,
selective tiled projections, fused SiLU/Q8). Saved core-mtp-run-baseline7df1c74
MUST stay untouched: overwrite was denied; no permission was obtained.
Safe comparisons rebuild ONLY ordinary executable between actual source variants,
never compile during timed inference; saved baseline checked unchanged.
Original core-session-baseline775/source77f89c3dirty (1750328B) also untouched.

## Текущий код / следующий физический шаг

New Session::prefill_last consumes the full window and retains ALL pre-head tap
rows, but projects/copies only final logits. All-row step_batch/verify unchanged.
No new allocations; root HC still computes all rows. MtpRunner NOT integrated.
Strict selected builds/model jobs562(exit0/2m32) and568(exit0/3m19) passed,
compiled daee1ef/dirtytrue. Checkpoint fixture35,261,440 values,ordinary2,483,200:
zero bit mismatches/maxabs. Nine checkpoint cases,26 invalids,full tap bits,
mixed APIs,pending invalidation,restore,continuations,capacity40/slots1/max32.
Both checkpoint modes keep owned allocations steady; RAII cleanup reported,
not a separate post-destruction VRAM recovery proof. No independent HF/speed claim.
Raw ROOT/runs/r4-prefill-last-dae{,-expanded}.jsonl and build/error logs.
Canonical journal73; controller copied actual raws,old71 byteprefix exact;
records72 initial and73 strengthened fixture are distinct, do not reappend.

NEXT after API commit: change only MtpRunner::begin target.step_batch(ids) to
target.prefill_last(ids). Run MTP runner API/full strict regression, then matched
4K/16K+512 with old/new MtpRunner source snapshots and preserved baseline.
No GPU job active after568. Do not relabel compiled sources to closure commits.

Then REVIEW/ADAPT runs/r4-layerwise-prefill.patch, still UNAPPLIED/UNQUALIFIED:
whole-window expert reuse may reduce repeated1024-chunk uploads. Its16K
contribution owners add1.608GiB/device plus1.26GiB extra root taps BEFORE trained
MTP2.67GB; never assume16K fits. Start opt-in4096 only after memory checks.
Retain current tiled128/fused primitives, correct offset/gather limits from old8.
Patch transfers~129GB widened residuals/16K plus HC/router: count total traffic.
No layerwise or two-GPU pipeline claim. Tokenizer/API,occupied128K and pack open.

## Принятые границы / не повторять

Rolled attention removed spills256VGPR/4380scratch ->52/0; same arithmetic,
cross-build3.745m component values exact, full model gate unchanged.
Selective tiled logical<=128 keeps physicalN8; dense output>512 stays oldN8
because batching it regressed. Both4K/16K paired accepted; primitives60d341e.
Fused SiLU/Q8 exact component1.35–1.58x translated to only~1.3%16K request gain,
no stable4K gain. Sparse CPU sampling skips EXACT zeros only, original generic
Neumaier Sum/filtering retained;95,389,224 snapshot bytes and signed-zero/tiny
residual tests exact. CPU draw/acceptance/residual components3.78/4.57/2.83x;
full requests only1–2% faster. Detail/provenance in README and results.jsonl.
Q4 tiled row4 and row1 candidates rejected on same-coordinate A/B/A, reverted.
Expert-only wideMMQ fails shortN32:7,559,835 violations,maxabs4.395458.
Tiny layer0 expert differences precede layer1 HC bound failure; mechanism not
independently proven. Reverted; never enable failed fixtures or loosen bounds.
Latest GPU phase profile was pre-fused/pre-sampler source60d341e: PP4K traced
72.995s/1.50m kernels, Q4 tiled33% and attention20% summed GPU time. Sums overlap.
Actual modelEOS and occupied128K remain unproven; custom-stop and capacity tests
do not close those. MTP independent sequential donor32 is bounded diagnostic,
same-build model self-reference is not independent HF proof.

## Машины / файлы

Controller: nc-lab:/home/opencode/ai/gfx906-qwen3.8fn, master, origin0FL01/gfx906-qwen3.8fn.
GPU: amude, radneon; SSH route/config in controller opencode.jsonc. Use SSH nc-lab.
Remote login shell is NOT assumed POSIX; explicit /bin/sh for set -eu/multi-step.
ROOT=/home/radneon/gfx906-core; source=$ROOT/src; build=$ROOT/build; runs=$ROOT/runs.
Canonical journal ONLY $ROOT/results.jsonl; controller receives verified copy.
Target model /home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf (75399121792B).
Sidecar /home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf (2786568256B).
Build image llama.cpp-gfx906:cmake-4.4.3, Clang23/HIP7.14.60850/rocBLAS5.5.
Donor production llama.cpp-gfx906:pp-stream-dcd685463d, source
/home/radneon/src/worktrees/qwen38-pp-trace-75, pin dcd685463d597d31f5ca759d32c94592a2740fa4.
Oracle libs $ROOT/build/oracle-production-libs; new oracle binary
$ROOT/build/mtp-canonical-diag/mtp-teacher-oracle; output dirs must be fresh.
Strict build: sh /core/src/tools/build.sh inside existing build container with
actual CORE_REVISION/CORE_DIRTY, /core bind ROOT, /models read-only model bind.
GPU tests: /dev/kfd,/dev/dri,video group; ONE GPU workload at a time.
16 physical/32logical CPU,oneNUMA; 2gfx906 wave64/CU60,17163091968B each.
LTE/reverse tunnel unreliable: inspect existing job/log after reconnect; no blind reruns.
