# Текущее состояние

## Активная задача (2026-10-04 22:28 UTC)

Цель: завершить ВЕСЬ план и превзойти llama.cpp на сопоставимых измерениях.
Лично, без subagents; document+commit+push каждый законченный срез.

PUSHED correctness296ea560825dfb1b066c1d2c5d93b61d13d40b3e: rolled attention,
39/39 CTest,1.036984320b same-build full-model logits zero error,3.745280m
cross-build component values exact. VGPR52/Scratch0 from256/4380.

Matched MTP A/B/A job1791143846406-290 COMPLETE exit0/41m55s; journal59 copied,
old58 prefix and all six actual sources/outputs/footers/manifests verified.
4K baseline164.695/166.229s -> candidate126.693s (1.3060x),PP48.298/TG12.200.
16K baseline559.588/562.923s -> candidate396.449s (1.4157x),PP46.173/TG12.281.
Every512 output ID/acceptance/RNG count identical within each triplet.
Capacity n+1024 matches pinned donor headroom. Donor still faster(~57/~144s).
A=f59c8de dirty; B=7df1c74 dirty; do not retag/duplicate accepted raw records.
/core/build/core-mtp-run-baseline NOW promotes rolled7df1c74, byte-verified.
Original non-MTP core-session-baseline stays untouched. No model copies.

Canonical tiled-grid primitive jobs338/344 COMPLETE exit0. Journal60 copied,
old59 bytes and actual expanded records verified. BothGPU300cases each,
204939974 exact compares/device,81352 CPU samples,21host rejects,298graph checks.
Legacy short/dense regressions pass; source296ea56/dirtytrue. Component APIs
support logicalN<=128 with physicalN8 grid tiles and original tails, no new math.
Measured candidate policy: only remainingN>=16; quantized or dense output<=512.
Large F32 output2560 is slower (~0.76x atN128) and must keep original N8 dispatch.
Primitive slice PUSHED60d341ebbacbbe6fecd5fbea00dadf9376bc2474.
Session integration is now a41-line diff: Matrix uses logical<=128 only for
remaining>=16 and quantized/small-output dense; expert gather/project groups
use<=128 with legacy<16 tails. All dot arithmetic remains physicalN<=8; no new
allocation or policy knob. Bounded full-model correctness passed; speed qualification pending.

Integration model job1791147708773-356 COMPLETE exit0/12m21s. Strict selected
build, short32 and full2088-row attention/continuation gates PASS; both remote
collectors pass. Short29798400 and wide1036984320 compared logits have zero
violations and diagnostic bit mismatches. Source60d341e/dirtytrue; not HF proof.

Single diagnostic probe1791148740791-363 COMPLETE exit0/3m22s:4K+512,
capacity5120. PP57.87835/TG11.66061/request114.59362s/load79.97922s separate.
All512 IDs/acceptance/RNG/pending match the prior rolled request. This is ONE
probe, not paired speed promotion; TG variation must be checked in the series.
Raw ROOT/runs/r4-tiled-session-60d-probe4k.{jsonl,err,manifest.json}.

Full strict build job1791149284220-364 COMPLETE nativeexit0/20m14s:
39/39 CTest passed1156.03s. Actual raw gates/probe copied to controller, collectors
rerun; canonical journal61 copied with old60-byteprefix unchanged.
Correctness integration PUSHED7f54d409f1e01c556aad5122efc8c9dc1545cc1c.
Matched trace-off job1791150716560-382 COMPLETE nativeexit0/34m10.
4K A126.127/127.737s -> B111.971s,1.133616x,PP58.51046/TG12.17687.
16K A397.990/397.574s -> B338.686s,1.174487x,PP55.16066/TG12.26576.
All512 output IDs/acceptance/RNG/pending identical inside both triplets.
Actual six raw runs/manifests copied and validated; journal62 old61 prefix exact.
A=rolled7df1c74/dirtytrue; B=tiled60d341e/dirtytrue; capacities5120/17408.
Selective dispatch accepted for measured workloads, not full-plan/llama win.
Baseline executable remains7df: attempted overwrite with accepted candidate was
blocked by action review; no overwrite happened. Do not circumvent the block.
Candidate core-mtp-run remains60d. Original core-session-baseline untouched.
Diagnostic jobs393/395 COMPLETE exit0; journal63, source60d341e/dirtytrue.
PP4K traced72.995s/1,503,743 kernels/summed46.932s; Q4 tiled15.570s(33.18%),
attention9.333s(19.89%), Q6head2.795s(5.96%). Dense tiled1.700s; large dense
0.958s. GPU unions22.471/24.450s, H2D7.460/7.233s; do not add overlaps.
MTP verify128outputs11.677s/588140kernels; Q4 paths lead, draft0.513s,
restore0.091s. No boundary-crossing trace events; diagnostic not speed result.
Q4 tiled physicalN8 row reuse2->4 REJECTED, source reverted. Native component
A/B/A jobs403/407/413 all exit0; exact values/CPU gates unchanged, but most
eligible shapes slower (roughly0.71–0.97x first-A/B). Journal64 retains paired
timings. Current runtime binaries untouched. Row1 experiment jobs413/419/426 also
passed all component gates but hurts dominantQ4_0 (~0.66–0.89x); Q4_1-only
1.05–1.09x is not integrated. Source restored; journal65.
Next: isolate expert-only wideMMQ from dense/PLE/GDN/projection changes in the
historical failed-wide candidate; current canonical graph and correctedQ8 stay.
Build/run only model fixture targets until qualified; current runtime preserved.
Further candidates require fresh measured priority: PP launch density/selection,
per-column dense reductions, two-GPU chunk overlap, grouped short MoE. Do not
mix another optimization into this still-unqualified integration slice.

R0–R3 closed. R4 speed, R5 measured dispatch, full R6, R7 occupied128K/tokenizer/API
and R8 safetensors/runtime-pack remain OPEN. Frozen correctness gates unchanged.
No 400–600PP/30–40TG or llama.cpp win claim.

## Текущий результат

### Новый window-срез 2026-10-04 17:01 UTC

core-mtp-window-test — test-only coordinator real proposals/SpeculativeSampler/
verify/teacher-rebuild/restore. Source961150d/dirtytrue; strict target build,
job1791132994108-105 nativeexit0/180s, ROOT/runs/r6-window-961-live.{jsonl,err}.
Six checked cases+uninterrupted32:55 checkedwindows/accept[33,10,12],
22100480target+13409280draft+13409280restored-target-continuation compares,
maxabs0. Live32 exactsame fullIDs/accept[10,6,3]/RNG38+46 as replayed32.
Capacity48 boundary reached; actual EOS NOT observed. No throughput evidence.
Canonical journal53, previous51byteprefix preserved; records52basic/53strengthened.
First boundedforward commit961150d PUSHED+remoteHEAD verified. GitHub key user
specified ~/.ssh/nc-lab; use per-command GIT_SSH_COMMAND with StrictHostKeyChecking.
No GPU job running after105. Next: production opt-in coordinator/CLI, realEOS
and longer-history tests, matched performance. FullR6 stillOPEN.


Compiled source d2f948ae44f5172dd2395fdbf1723e0b1ddd6237/dirtytrue, не closurecommit.
Job1791130233817-73 nativeexit0/20m51s: strict C++20/HIP20 Release gfx906,
CTest39/39 1149.99s; fresh own32 teacher-width3, allN1/2/3 selferror0,
reset/error/reuse/poison/restore/KV-only/carry-publication tests PASS.
ROOT/runs/r6-mtp-final-d2-regression.log, -self.jsonl, -capture/.
Fresh tokens/previoushidden/D/logits BYTE-identical to earlier position26 capture.
Independent sequential donor + explicit --dense-f32-control --dense-short-canonical:
D327680 maxabs3.8146973e-6/maxratio.000205267; logits7946240
maxabs.020387292/maxratio.961748919; zero violations at original gates.
Raw ROOT/runs/r6-mtp-canonical-diagnostic-n1; job1791129410074-50 nativeexit0.
CPU standalone oracle selftest1247970checks/45rejects/1222656double comparisons PASS.

LIMITS: generated BOS248044,100..130,32rows/cap64 only; not HF/trained-MTP2/performance.
Original direct-division control FAILposition26 (D3978/logits68892 violations).
Canonical batched donorN2/N3 also numerical FAIL. On p4 donorN1vsN2,
inputs through FFN-HC bitidentical; FFN1.7762e-5→D3.016e-5→head.02656→logits.06090.
Do not hide these controls or relax gates; core ownN1/N2/N3 remains invariant.
README contains exact scope/reproduction. Canonical ROOT/results.jsonl51records:
one r6_trained_forward_sequential_teacher append; old50 byteprefix unchanged,
remote/controller matched. Do not append accepted artifacts again.
No active GPU job at closure; check processes/logs before launching fresh work.

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

## Baseline / PP frontier

Preserve /core/build/core-session-baseline (1750328B, qualified775/source77f89c3dirtytrue).
Future A explicitly attentiontile8,CPU0disabled. Do NOT overwrite it with current build.
Accepted823 trace-off A/B/A: 4K PP35.58475/TG10.0650,request165.8769s;
16K PP32.13642/TG9.7726,request562.1171s;512outputs/MTPoff/loadseparate.
Own Matrix/apply and expert compute remain canonical<=8 despite logical1024.
Existing runs/r4-layerwise-prefill.patch remains UNAPPLIED/UNQUALIFIED:
retains <=8 kernels, CPU-staged residuals, no session_main integration.
Next PP investigation: fresh PP-only B8 profile, wide-kernel first-divergence,
separate last-logits and layerwise reuse, batched QSA score/select, then pipeline.
All prior prerequisites and raw source provenance are retained in README/PLAN/journal.
