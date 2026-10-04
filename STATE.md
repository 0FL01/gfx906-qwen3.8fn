# Текущее состояние

## Активная задача (2026-10-04 17:21 UTC)

Цель пользователя2026-10-04: завершить ВЕСЬ план и превзойти llama.cpp на
сопоставимых измерениях. Лично, без subagents; document+commit+push каждый срез.

Opt-in MtpRunner/core-mtp-run/API tests готовы к commit поверх f59c8de (PUSHED).
Файлы src/mtp_runner.{hpp,cpp},mtp_run.cpp,mtp_runner_test.cpp +CMake.
Full job1791134454670-146 nativeexit0/22m11s, strict build+CTest39/39 1150.55s.
CLI512:8prompt/512outputs/consumed519,213windows/accept[44,40,129],RNG426/594;
request41560.08ms/load77836.55ms separate/TG12.586. Smoke only, not4K/16K evidence.
API native142:PPchunk8 vs CLIchunk1 full32IDs equal;repeat/reset,invalidrequest
preservation,capacity48,2 custom stop-token branches PASS; actualEOS248046 unseen.
Help+7 CLI negative cases PASS. Canonical journal54/old53byteprefix preserved.
Raw prefix ROOT/runs/r6-run-f59. No active GPU process after146.
Next: commit/push this runner slice, then paired own MTP-off/on4K/16K+512;
external llama.cpp comparison and PP optimization. No 400–600PP/30–40TG claim.

R0–R3 закрыты. Bounded trained MTP forward/teacher diagnostic slice реализован
и проверен; полный R6/MTP2 ещё OPEN. Test-only stochastic coordinator уже
проверен; следующий код — reusable opt-in serving/CLI плюс реальный EOS
и длинная история. Не менять frozen tolerances.
R4 PP speed, R5 measured dispatch, R7 occupied32K/64K/128K+tokenizer/API,
R8 safetensors/runtime-pack также OPEN. Цели400–600PP/30–40TG не достигнуты.
Работать лично, без subagents; завершённые срезы документировать, commit/push.

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
