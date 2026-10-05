# Текущее состояние

## Задача
Завершить весь PLAN.md и превзойти pinned llama.cpp в сопоставимых запросах.
Работать лично через SSH nc-lab, без engineering subagents. Каждый законченный
срез документировать, commit и push. R0–R3 закрыты; полный R4–R8 ещё OPEN.
400–600 PP / 30–40 TG и победа над llama.cpp не достигнуты.
Численные gates, weights, precision, top-k/QSA budget не ослаблять.

## Последний runtime / измерения
Attention-deferred source53d1153/dirtytrue, explicitlayerwise4096:
4K+51291.140152s, PP78.853648/TG13.037670;
16K+512260.903173s, PP73.315235/TG13.652910.
Journal92, result pushedadcd7f46af01a3edebe3c26c9fbb5ac09623b438.
A/B/A: PP~2% быстрее meancontrols на обоих размерах;4K fullrequest лишь
19.5ms быстрее лучшего A (шум),16K~2.49% лучше meancontrols.
Все512 IDs/acceptance/RNG/pending совпадают. Pinned llama.cpp~57/~144s быстрее.
Default остаётся ordinary1024, layerwise4096 opt-in. Ordinarycore-mtp-run
восстановлен на attention candidate. Saved core-mtp-run-baseline7df и
core-session-baseline775 НЕ перезаписывать: разрешения на это нет.
Полная история, raw provenance и отрицательные результаты: results.jsonl/README.

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

Следующий шаг: commit/push этого среза, затем runs/r4-pipeline-ab.py.
Скрипт пока НЕ запущен и revision file не создан. Собрать ordinary core-mtp-run
со stamp нового commit вне timing; одна неизменная binary, pipeline0 vs2048,
layerwise4096,4K/16K+512, capacity prompt+1024. Сохранённый baseline не трогать.
После paired result — измерить фактическую VRAM и выбрать следующий bottleneck.
GPU/CTest работы сейчас нет. Полный PLAN и победа над llama.cpp ещё не закрыты.

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
