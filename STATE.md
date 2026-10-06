# Current state

## Goal and constraints
Complete all of PLAN.md and beat pinned llama.cpp on comparable full requests.
Work personally through SSH nc-lab, without engineering subagents.
Document, commit and push each completed slice; verify remote master.
R0–R3 are closed. Full R4–R8 and the400–600 PP /30–40 TG goals remain OPEN.
Do not relax numerical gates, weights, precision, QSA/top-k budgets or sampling.
Same-engine parity is not independent HF; capacity128K is not occupied128K;
custom stop is not natural EOS.

## Latest qualified implementation
Journal109: opt-in --prefill-residual-device, requiring the bounded pipeline.
Default OFF. One private residual buffer per GPU; existing frame scratch,
math, CPU routing, FFN staging and explicit loop checks are retained.
At stage2048, native off/on construction proves exactly83,886,080 extra owned
and workspace bytes, one extra buffer per GPU. No model-value conversion.
New committed-call transfer diagnostics count actual submitted byte extents
and explicit loop checks, not PCIe hardware traffic or wait durations.

Compiled source: f4d5e6166e8afb841422cdc7eaf58dc273bafb25 dirty.
Serial32/resident32/resident256 and full4096/stage2048 numerical gates are exact;
full4096 has13,494,576,640 comparisons, zero bit mismatches. Fault/reset and MTP
API32/2051-carry pass. Full strict build and41/41 CTests pass in1157.78 seconds.
Three production CLI rejections pass before any model file is opened.
Short PID1660234 completed0. Long PID1664601's native phases all passed,
but its final CTest-summary string check failed. Original exit1 is retained;
r4-resident-finalize-f4d.exit.json is0 after verifying all41 actual Passed rows.
No native test rerun or output alteration. Actual controller logs and old108
journal byte prefix validated. Full16K logits/taps and speed are NOT qualified.

## Next physical step
Copy runs/r4-resident-ab-f4d.py and r4-resident-ab-launch.py to GPU runs, then
launch the latter exactly once. They require the finalizer0 and qualified
runtime hash. Same binary, A/B/A4K/16K+512, only the residual flag changes;
assert full IDs/acceptance/RNG/pending, copy extents and loop-barrier counts.
Do not rebuild or edit GPU sources while this benchmark runs.
Runtime SHA256:
dc980a175142abdc4725bc077e5de02d5e2ead8e6746aa3e4d69174644d63508
After a retained candidate's pair, run the full16K reference before claiming
full16K logits/taps. Do not confuse diagnostic fixture wall times with speed.

## Confirmed performance baseline
Journal108, pushed8cae1a4: bounded batched QSA selector integration.
4K+512:81.749504s, PP92.904993 / TG13.568888.
16K+512:175.970231s, PP117.911638 / TG13.804565.
Mean-control request gains5.21% /5.91%; both beat both controls, all512
trajectories exact. These are BEFORE resident-residual optimization.
Pinned llama.cpp remains around57s/144s, so it is still faster overall.
Rejected stage1024 and Q4 N16/row/packed640 variants are documented in README.
Do not revive failed expert-wide MMQ or relax its frozen model gates.

## Strata prerequisite completed
User-directed deployment in /home/radneon/strata is built and live-tested.
Source6f32ec0 plus documented HIP-compatibility and API-only/privacy patches.
All12,288 Coder IQ1_M experts fit with explicit STRATA_STAGE_TRIM=1.
API/auth/SSE,5053-token prompt, disconnect/reuse and restart pass.
Final stop at2026-10-05 23:28:58UTC: container absent,8080 closed, autostart
removed. Both GPUs were freed. start.sh/stop.sh/logs.sh remain installed.
This is another model/quant and is not an engine benchmark or occupied128K proof.
Receipt: runs/strata-deployment-20261005.json; donor review: RECON.md.

## Machines, execution and protection
Controller: /home/opencode/ai/gfx906-qwen3.8fn, master,
origin git@github.com:0FL01/gfx906-qwen3.8fn.git; supplied Git key~/.ssh/nc-lab.
GPU amude/radneon ROOT=/home/radneon/gfx906-core.
Source ROOT/src, build ROOT/build, runs ROOT/runs.
Canonical journal: ROOT/results.jsonl, not ROOT/src/results.jsonl.
Fish login shell: wrap commands in explicit /bin/sh or invoke a Python file.
Models mounted read-only from /home/radneon/models-nvme:
qwen38-keep1-Q4_0.gguf and mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf.
Do not duplicate or hash models.
Image llama.cpp-gfx906:cmake-4.4.3, ROOT:/core and models:/models:ro.
GPU flags: --device /dev/kfd --device /dev/dri --group-add video --ipc host.
Release HIP compiler /opt/rocm/llvm/bin/clang++, prefix /opt/rocm.
Full build: CORE_REVISION=actual40 CORE_DIRTY=ON sh /core/src/tools/build.sh.
Detached wrapper runs/run-detached-series.py writes prefix.exit.json.
One GPU workload; no heavy compilation during final performance requests.
After SSH loss inspect the SAME PID/log. Backoff1/2/4/8/10 minutes; do not
change transport or duplicate a heavyweight operation.

Protected files must remain unchanged:
- ROOT/build/core-mtp-run-baseline, source7df, SHA256
  e1dd8b84a434ea72611c5cd67ccedb53d64a94705dbbd0e94741912545962a32
- ROOT/build/core-session-baseline, source77f dirty, original775 binary

Donor pin dcd685463d597d31f5ca759d32c94592a2740fa4; isolated loopback runs,
512 actual outputs, capacity prompt+1024, temperature1/top-p.95/top-k20/seed12345.
Its TG numerator512 differs from our511. Its QSA is a speed reference, not oracle.
R7 API/tokenizer/occupied long-context and R8 streaming original-safetensors pack
remain open. The user's October5 research is a hypothesis guide, not evidence
of fresh GPU results. Prioritize transfer/barrier organization before another
unmeasured matrix-geometry sweep.
