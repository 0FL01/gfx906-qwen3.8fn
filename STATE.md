# Current state

## Goal and constraints
Complete ALL of PLAN.md and beat pinned llama.cpp on comparable full requests.
Work personally through SSH nc-lab, without engineering subagents.
Document, commit and push each completed slice; verify remote master.
R0–R3 are closed; full R4–R8 and the 400–600 PP / 30–40 TG goals remain OPEN.
Do not relax numerical gates, weights, precision, QSA/top-k budgets or sampling.
Same-engine parity is not independent HF; capacity128K is not occupied128K;
custom stop is not natural EOS. R7 practical API/tokenizer/long context and
R8 original-safetensors streaming pack are still open.

## Latest completed slice
Batched QSA selector component was pushed as54a4dbf3792145e3483ef2f61a2840897c17843f.
Session integration now passed all gates, journal107; compiled source54a4dbf dirty.
Final native PID1608246 completed0 at2026-10-05 21:35:28UTC.
Full strict build and41/41 CTests passed in1154.97s.
Wide786,677,760 and attention1,036,984,320 comparisons have zero violations/bits.
Full16K has29,497,640,960 cumulative exact logits/tap comparisons.
Serial/pipeline256, attention tiles16/128, faults/reset, MTP API,2051/16387 carry,
default memory and actual strict collectors also passed.
Controller copied/validated actual logs and old106 journal byte prefix.
Raw directory: ROOT/runs/r4-qsa-batch-model-54a.
Original PID1575377 and resume1602356 stopped between phases on external
Telegram-build/low-RAM guards. Final resume2 preserved all completed phases.
The two earlier exit1 records are resource interruptions, not numeric failures.

Selector subgroups are bounded to8, while attention tiles through128 remain valid.
Score arithmetic is unchanged. Scalar tails retain the old selector entry point.
Histogram/state/candidate rows are private; final output publication follows checks.
At frame1024 existing f(16) already fits the score rows. Native ownership delta
is exactly258,944B/device; full attention ID/count storage is retained.
No model speed claim from this correctness slice.

## Next physical step
QSA paired result committed/pushed8cae1a4, journal108:4K81.749504s,
PP92.904993/TG13.568888;16K175.970231s,PP117.911638/TG13.804565.
All512 IDs/acceptance/RNG/pending exact; both beat both A controls.
User-priority Strata deployment completed and stopped23:28:58UTC; all setup,
start/stop/logs scripts remain in /home/radneon/strata. Both GPUs are free.
Source6f32ec0+documented compatibility/privacy patches; actualAPI/auth/SSE/
5053-token prompt/cancel-reuse/restart passed. Not an occupied128K proof.
Now implement submitted-byte/explicit-barrier diagnostics and opt-in bounded
GPU-resident residual for pipeline stages. Preserve math, CPU routing and
expert groups initially; compare one factor in a same-binary A/B/A. Full
numeric/tap/fault/reset/MTP gates and exact owned-memory accounting remain.
Use user-supplied research as hypotheses, not as measured oracles. See new
RECON section; do not repeat rejected N16 or expert-wide MMQ experiments.

## Latest qualified performance before this candidate
Current best16K+512 is182.044469s, PP113.318528/TG13.641648, using stage4096
within logical16384 (sourcec3d7625 dirty, journal103). Controls at stage2048
were186.827339/185.103706s; a modest~2% gain costs215,777,280B more per GPU.
Lower-memory stage2048 previously measured183.372317s, PP112.303156/TG13.634030
(journal101, sourceeaa778c dirty). Its sampled root free VRAM was868,737,024B.
Pinned llama.cpp remains around57s at4K and144s at16K, hence still faster overall.
Stage1024 for4K was rejected: PP85.56 versus89.05/88.09, no stable request win.
Default ordinary1024 configuration has not changed.

## Evidence and avoided repeats
Full numerical history and negative experiments: README.md/results.jsonl.
QSA component journal106:525 cases/GPU,43,812,608 ID/count/padding comparisons,
23 host rejects,24 nonfinite cases,3 sticky cases and12 graph replays/GPU.
28/28 resident A/B/A coordinates win; eight-query median6.2456x is NOT model speed.
Current16K profile journal105 found2,751,936 radix histogram/select launches,
motivating this candidate. Tracing materially perturbs latency.
Packed640, small/large N16, row-owned2560 and row1/row4 Q4 geometry did not give
stable model wins. Expert-wide MMQ failed frozen full-model gates; do not revive
it or relax bounds. Production Q4 dispatch remains canonical8x2.
Tiled attention rolled-dot/deferred checks are qualified. TSan could not start
due ASLR layout; no security settings changed and no TSan pass is claimed.

## Machines and safe execution
Controller: /home/opencode/ai/gfx906-qwen3.8fn, master,
origin git@github.com:0FL01/gfx906-qwen3.8fn.git; supplied Git key~/.ssh/nc-lab.
GPU amude/radneon: ROOT=/home/radneon/gfx906-core.
Source ROOT/src, build ROOT/build, runs ROOT/runs.
Canonical journal is ROOT/results.jsonl, NOT ROOT/src/results.jsonl.
Login shell is fish: use explicit /bin/sh or python3.
Read-only models: /home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf
and mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf. Do not duplicate/hash models.
Image llama.cpp-gfx906:cmake-4.4.3; bind ROOT:/core and models:/models:ro.
GPU flags: --device /dev/kfd --device /dev/dri --group-add video --ipc host.
Release HIP compiler /opt/rocm/llvm/bin/clang++, prefix/opt/rocm,
CORE_REVISION=actual40 and CORE_DIRTY=ON. Full build: sh /core/src/tools/build.sh.
Detached wrapper runs/run-detached-series.py writes prefix.exit.json.
One GPU workload; no heavy compilation during final performance requests.
After SSH interruption inspect the SAME PID/log; backoff1/2/4/8/10min.
Do not change transport settings or blindly repeat heavyweight runs.

Protected files MUST remain unchanged:
- ROOT/build/core-mtp-run-baseline, source7df, SHA256
  e1dd8b84a434ea72611c5cd67ccedb53d64a94705dbbd0e94741912545962a32
- ROOT/build/core-session-baseline, source77f dirty, original775 binary

Donor pin dcd685463d597d31f5ca759d32c94592a2740fa4.
Use isolated loopback/no-public-port runs,512 actual outputs, capacityprompt+1024,
temperature1/top-p.95/top-k20/seed12345. Donor TG numerator512 differs from own511.
Donor QSA differs from HF whole-block semantics; it is a speed reference, not oracle.
