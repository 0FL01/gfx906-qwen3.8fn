# Текущее состояние

## Задача
Завершить весь PLAN.md и превзойти pinned llama.cpp в сопоставимых запросах.
Работать лично через SSH nc-lab, без engineering subagents. Каждый законченный
срез документировать, commit и push. R0–R3 закрыты; полный R4–R8 ещё OPEN.
400–600 PP / 30–40 TG и победа над llama.cpp не достигнуты.
Численные gates, weights, precision, top-k/QSA budget не ослаблять.

## Последний runtime / измерения
Pipeline2048 within layerwise4096, source dc60e0f dirty:
4K+51284.422057s, PP88.353420/TG13.425791;
16K+512229.376431s, PP85.327488/TG13.677163.
Matched A/B/A journal95: controls92.547/92.184s and265.569/260.969s.
Both PP/request beat both controls; mean request ratios1.094096/1.147759.
All512 IDs, acceptance, RNG, pending equal. Same binary in all six processes.
Serial deferred-attention prior result91.140/260.903s is retained in journal92.
Pinned llama.cpp~57/~144s still faster. Default remains ordinary1024;
fast configuration is explicit --layerwise-prefill4096 --prefill-pipeline2048.
Saved core-mtp-run-baseline7df and core-session-baseline775 MUST stay untouched.
Full history, actual provenance and negative experiments: results.jsonl/README.

## Текущий срез / следующий шаг
Двух-GPU pipeline correctness завершён: journal94, source48f3d14 dirty.
Expanded series PID1453820, ROOT/runs/r4-pipeline-expanded-48f, native0
2026-10-05 10:36:56 UTC. Full strict build, 40/40 CTests, 1157.10s.
32/8,128/17,4096/2048:125642240/441443840/13494576640 сравнений,
ошибок и bit mismatches0; счётчики включают taps и повторные references.
8 fault/reset cases, MTP32 API, carry2051 с окнами1025/1025/1,
6 CLI rejects проходят. Предыдущая ошибка только в fixture: запрещённый getter
checkpoint_state после invalidation; исправлено ожидание rejection, не runtime.
Actual downloaded raws/manifests и старый93 byte prefix проверены.

Pipeline default off; logical capacity<=4096, отдельные host frames для24/24
GPU stages и2 ring slots. Дополнительных explicit GPU buffers нет. Все tap rows,
absolute positions, transactional publication и failure drain сохранены.
Host-only coordinator push48f3d14, journal93:84cases,20 repeats, ASan/UBSan pass.
TSan не стартовал из-за ASLR layout; security settings не менялись.

Pipeline correctness PUSHED dc60e0f56da9d2b10bdbb5de845491470da8d0eb,
remote master verified. Ordinary core-mtp-run rebuilt with this revision/dirtytrue.
Paired PID1468638 completed native0 at11:12:52 UTC; journal95 validated.
Actual six raws/manifests/one runtime digest and old94 byte prefix verified.
ACTIVE separate VRAM observation: detached PID1474125, from11:15UTC;
ROOT/runs/r4-pipeline-vram-dc6.py, prefix ROOT/runs/r4-pipeline-vram-dc6
(.series.log/.exit.json/.pid), directory same prefix. Fresh HIP mapping,
16K+512 exact candidate argv, expected IDs/RNG from paired16K-B.
Do not rebuild/change ordinary binary until this observation completes.
After recording observed memory, rebuild ONLY core-mtp-profile for diagnostic
4K+128 phase tracing, then choose the next measured improvement.
No other GPU workload; full PLAN and llama.cpp win remain open.

## Не повторять
Packed640, small/large-output N16, row-owned2560 и row1/row4 Q4 geometry
не дали устойчивого model/coordinate выигрыша; production dispatch canonical8x2.
Expert-wideMMQ провалил frozen fullmodel gate; не включать/не ослаблять bounds.
Последний phaseprofile layerwise: PP55s, Q4N8~38%, attention~22% kernel-sum;
sums overlap, не wall breakdown. RootGPU ограничивает расширение fullwindow.
Same-build self-parity не independent HF; capacity128K не occupied128K;
custom stop не natural EOS. Полный trained-MTP oracle/R7/R8 ещё открыты.

## Машины / команды
Controller /home/opencode/ai/gfx906-qwen3.8fn, branchmaster, origin0FL01/gfx906-qwen3.8fn.
GPU amude/radneon, ROOT=/home/radneon/gfx906-core, sourceROOT/src, buildROOT/build,
runsROOT/runs, canonical journalROOT/results.jsonl (не ROOT/src/results).
Route controlleropencode.jsonc; explicit /bin/sh (login shell fish).
Models read-only /home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf and
mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf; не дублировать/не hash модели.
Docker llama.cpp-gfx906:cmake-4.4.3, bind ROOT:/core and models:/models:ro.
GPU: --device /dev/kfd --device /dev/dri --group-add video --ipc host.
Configure Release, HIP /opt/rocm/llvm/bin/clang++, prefix/opt/rocm,
CORE_REVISION=actual40 and CORE_DIRTY=ON; fullbuild sh /core/src/tools/build.sh.
CPU coordinator: buildtargetprefill-pipeline-test then
ctest --test-dir /core/build -R '^prefill-pipeline$' --repeat until-fail:20.
Donor pinned image llama.cpp-gfx906:pp-stream-dcd685463d, sourcepin
dcd685463d597d31f5ca759d32c94592a2740fa4; isolated no-public-port runs only.
512actualoutputs, capacityprompt+1024, primarysampling1/.95/20/seed12345.
One GPU workload, build outside timing. SSH drop: inspect same job first;
backoff1,2,4,8,10min, no duplicate heavyweight runs/network changes.
