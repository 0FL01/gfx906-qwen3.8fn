# Current state

## Goal and constraints
Finish ALL PLAN.md, beat pinned llama.cpp on matched complete requests.
User requires personal work through SSH nc-lab, no engineering subagents.
Document, commit and push each completed slice; verify remote master.
R0–R3 closed; fullR4–R8 OPEN.400–600PP/30–40TG and llama.cpp victory not achieved.
Never relax numeric gates, weights, precision, QSA/top-k budgets or sampling.
Same-engine parity is not independent HF; capacity128K is not occupied128K;
custom stop is not natural EOS. R7 practical API/tokenizer/long context andR8
original safetensors streaming pack remain open.

## Latest results
Canonical journal103, controller copy actualraws/manifests and old102 prefix verified.
Latest sourcec3d7625129f1a3c50bf32d96180d0859d6c81064/dirtytrue.
Stage4096 withinlogical16384:16K+512182.044469s,PP113.318528/TG13.641648.
Controlsstage2048:186.827339/185.103706s,PP110.805481/111.225462.
Meanrequest1.021539x,PP1.020749x; all512 IDs/RNG/acceptance/pending equal.
Extra215,777,280B/device owned stage memory; actual MTP fits, driverpeak notsampled.
Lower-memorystage2048 previous paired result183.372317s,PP112.303156/TG13.634030
vs234.193/231.932 logical4096controls; journal101, sourceeaa778cdirty.
4K B83.416036s vs86.192/85.047 but same2stages, notstructuralgain.
Pinned llama.cpp~57/~144s stillfaster. Defaultordinary1024 unchanged.
Full history/negative results/provenance: README.md/results.jsonl.

## Completed current slice
PID1529544 ROOT/runs/r4-stage4096-gates-c3d endednative0 at16:57:54UTC.
Strict selectedbuild+2targetedCTest; no runtime source changes.
Full16384x4096 fixture29,497,640,960 cumulative logits/taps compared,
zeroerrors/bits; repeated/reset/continuation/invalidowner gates pass.
16387+32CLI carry2048vs4096 exactlyequal. Then samebinary16K+512 A/B/A.
Latest full41CTest is journal100,1156.08s, same runtime source, not rerun here.
SSH outage16:12..16:32 recovered; SAME PID observed, no duplicate run.
Initial4097 fixture planned but corrected to allowed16384 BEFORE any native run.

## Next physical step
Prepare existingconfigurationstage1024 withinlogical4096 for4K prompt overlap:
selected build, full4096x1024 logits/taps,4099+32 carry vsstage2048,
then samebinary4K+512 A/B/A stage2048/1024/2048. No runtime arithmetic edits.
Read/build stamp current committed40hash dirtytrue. Unique runs/driver prefix.
No GPU workload active after1529544. Do not reuse old result paths.
Stage4096 remains optional; no default promotion, no4K speed claim.

## Memory
Head128 owned allocation saved889,978,880B/device; full41CTest journal98.
Stage-sized physicalscratch preserves full logical teacher taps.
Journal102 observerPID1525727 ended15:43:58UTC: logical16K/stage2048,
sameeaa binary/512trajectory,2710samples at0.1s:
minfree5,566,386,176/868,737,024B, maxused11,596,705,792/16,294,354,944B.
Global sampled extrema, not exactpeak. Source extra4096stage budget216MB/device
suggests~653MBroot free, not measurement. Full reference fixtures mayuseswap;
no system settingschanged, no speedclaimfromcorrectness runs.
Protected saved binaries MUST NOT be overwritten:
ROOT/build/core-mtp-run-baseline7df digest
e1dd8b84a434ea72611c5cd67ccedb53d64a94705dbbd0e94741912545962a32
ROOT/build/core-session-baseline775, source77fdirty.
Ordinary core-mtp-run currentlyc3ddirty, digest
a3bbf027d3d0fd70a0946c2a71e8f1f6d2dd6efcdcc21c742dc5a7510a89c41f.

## Avoid repeated failed work
Packed640, small/largeN16, row-owned2560 and row1/row4 Q4 geometry did notyield
stable model wins; production canonical8x2. Expert-wideMMQ failed frozenmodel
gate: do notenable orrelax bounds. Attention rolleddot/deferredchecks accepted.
Pipelineprofile journal97 has GPUkernel overlap3.199s and envelope21.508s
includingidle, NOT21.5s actual simultaneouscompute. Q4N8~38%,attention~21%
summedkernel durations, notwallbreakdown. No copybytes inCSV. TSan didnotstart
becauseASLRlayout; no securitysettingschanged, noTSanpassclaim.

## Machines and execution
Controller /home/opencode/ai/gfx906-qwen3.8fn, master,
origin git@github.com:0FL01/gfx906-qwen3.8fn.git, suppliedgitkey~/.ssh/nc-lab.
GPUamude/radneon, ROOT=/home/radneon/gfx906-core; srcROOT/src,
buildROOT/build,runsROOT/runs, canonicaljournalROOT/results.jsonl.
Loginfish: explicit /bin/sh or python3 forremote commands.
Models read-only /home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf and
mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf; do notduplicate/hash models.
Docker llama.cpp-gfx906:cmake-4.4.3, ROOT:/core,models:/models:ro,
--device /dev/kfd --device /dev/dri --group-add video --ipc host.
Release HIP /opt/rocm/llvm/bin/clang++, prefix/opt/rocm,
CORE_REVISION=actual40 CORE_DIRTY=ON; fullbuildsh/core/src/tools/build.sh.
Detached wrapper runs/run-detached-series.py writes prefix.exit.json.
One GPUworkload; builds outside timing. Poll samePID/log afterSSHdrop.
Backoff1/2/4/8/10min; no duplicateheavyload ornetworksettingschanges.
Donorpin dcd685463d597d31f5ca759d32c94592a2740fa4, isolatedloopback/noexposedports.
512actualoutputs, capacityprompt+1024,primarysampling1/.95/20/seed12345.
DonorTG512 numerator vsown511; comparefullrequestscope. DonorQSAsemantics
differ fromHF blockselection; donor is performance reference, notHF oracle.
