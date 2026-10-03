# Текущее состояние

## Статус и текущая задача

R0–R3 закрыты: own48 Session/teacher32/reset и capacity131072 owned ledger. Short-PP и logical-wide1024 slices закрыты; полный R4–R8 и целевые скорости открыты.
Ближайшая задача — real4K/16K full teacher logits/continuation и causal chunk boundaries после 2052-token QSA budget, затем PP/full-request с 512 actual outputs/peak VRAM по обеим GPU с paired performance gates.
Подготовленные `src/prefill_long_test.cpp` и sample CLI/sampler (`src/session_main.cpp`, `src/session_cli.hpp`, `tests/session_cli_test.cpp`, sampling3) ещё не accepted/integrated; вне wide closure.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller checkout — исходник; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления; не запускать второй экземпляр вслепую.
Модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 bytes), источник пользователя https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 bytes); источник загрузки не установлен.
Baseline image `llama.cpp-gfx906:pp-stream-dcd685463d`; source `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`. Config `/home/radneon/llama/{docker-compose.yml,models.ini}`; наш baseline-контейнер остановлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, canonical journal только ROOT `/home/radneon/gfx906-core/results.jsonl`.
Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5; gfx906 GEMM проверены. Runtime image без CMake.
Hardware: 16 physical/32 logical CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60/VRAM 17 163 091 968 bytes. RECON без изменения машины/стека не повторять.

## Рабочие команды

Accepted wide build: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=88bd3e6b24ab1dc07c556f5093533d10a91ecd80 -e CORE_DIRTY=ON llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'`. Для новой сборки actual revision/dirty; compiled artifacts не retag новым commit. Explicit `sh` нужен: source build.sh100644.
Wide: `docker run --rm --name core-prefill-wide --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-wide-test -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf > /home/radneon/gfx906-core/runs/FRESH-prefill-wide.jsonl`.
После exit0: `python3 -B /home/radneon/gfx906-core/src/tools/record_prefill_wide.py --raw /home/radneon/gfx906-core/runs/FRESH-prefill-wide.jsonl --results /home/radneon/gfx906-core/results.jsonl`; проверить collector exit status, append ровно один record после validation. Existing accepted raw не append повторно.
Session: тот же Docker/device/mount контракт, entrypoint `/core/build/core-session`, `--generate 32 --ignore-eos MODEL 248044` (greedy diagnostic). Reset/batch/memory entrypoints `core-session-test`/`core-session-batch-test`/`core-memory`, только MODEL. Все repro/teacher/oracle/short commands — README.md.
Oracle teacher: `sh tools/build-oracle.sh`, production libs + `--all-layers --hf-gdn-l2-control --hf-qsa-f32-control --warm-cache 12 --cache-inserts 10` diagnostic-only, не bitwise HF/unchanged-math/performance baseline.
Fresh logs/trace paths в existing runs/; одна GPU-нагрузка, final measurement без тяжёлой сборки/tracing.

## Последний подтверждённый результат

Job `1791014580540-724` exit0/elapsed22m16s, compiled `88bd3e6b24ab1dc07c556f5093533d10a91ecd80` dirtytrue: strict HIP/CXX Release build/CTest28/28 (595.52s), затем sequential reset/batch/wide/default-memory, каждый exit0.
Actual raw `runs/r4-prefill-wide-a.jsonl` protocol1/3 431 452 bytes: одна Session cap1056/slots1/max1024; reference1056 N1, teacher1024 schedules 8×128/7×129+121/single1024, 32 N1 continuations в каждой фазе.
4224 total rows/1169 windows/1181 records: 1 048 903 680 finite logits/786 677 760 comparisons; zero violations/diagnostic bit mismatches/maxabs, frozen `.02+.002|ref|`. Same-Session N1 self-parity, не independent HF/fullR4/4K16K/throughput/cold-cache equality.
15 atomic rejects; 1205 memory observations (1175 serialized+30 summarized); guards/full live span/stats/routeStats preservation и steady owned ledgers. Min freeGPU0/1:12 709 560 320/12 153 815 040 bytes. Aggregate floor не individual-buffer/full-RAM proof; wide owned release не measured.
Observed max-group:1017; число groups>128:950 по всем фазам, single1024 phase:874. Это logical assignments к одному expert/layer, не physical 128-column kernel/repack execution proof.
`record_prefill_wide.collect(actual_raw)` passed; parent local18/18 passed за 47.192s. Actual `record_memory.collect(r4-prefill-wide-a-memory.jsonl)` passed(defaultcap131072/slots112 regression); duplicate memory record не добавлен.
Logs `runs/r4-prefill-wide-a-build.log`, `r4-prefill-wide-a-{reset,batch,memory}.jsonl`. Canonical ROOT journal ровно19 records: одна new `r4b_wide_prefill`,18→19, old byte prefix/parsed history unchanged/source88bd dirtytrue retained; downloaded journal +1/-0 и controller `collect(downloaded_actual_raw)` проверены.

## Контракт и границы

Separate `SessionRouteStats`/`route_stats()`: `last_max_expert_group_assignments` — last successful call max assignments к ОДНОМУ expert/layer по 48 слоям; `expert_groups_gt128` — cumulative call/layer/expert groups>128 since reset. Publish только success; constructor/successful reset zero; invalid arguments/execution failure preserve. Historical `SessionStats`/serialized protocols unchanged; execution failure требует reset.
Own Session: RAM experts/default112 GPU slots/layer, static24/24, Q4 KV/FP32 index/GDN/PLE persistent, default max_batch1/pinned40KiB. Accepted logical≤1024: canonical nonexpert/shared/expert microtiles≤8, routes once/full chunk/triplet once/group; GDN chronological slices≤128. Band16/two stages/copy_ready/consumer_done сохраняются; новые weights/precision/arithmetic/epsilon/gates не менялись.
Canonical attention dot/value ordering и internally widened exp — ранее маркированные accuracy changes, не speedup. Wide MMQ выигрыши component-only; PP/full-request скорость не established. Historical R3/short details — README.md/results.jsonl.
CLI greedy diagnostic-only; считать actual generated tokens, последний emitted token pending. Primary temperature1.0/top-p0.95/top-k20 ещё не qualified.

## Следующие действия

1. Завершить integration/qualification подготовленного long fixture: real4K, затем 16K, полный vocabulary N1/chunk logits, occupied-prefix/continuation и causal selected IDs на QSA budget boundaries≥2052; frozen gates не ослаблять.
2. После correctness проверить prepared sampler/CLI и full request с 512 actual outputs без prefix reuse; PP/TG/full elapsed/peak VRAM и paired performance. R5 CPU miss vs H2D+GPU/overlap остаётся.
Index raw128 без H/inverse; GDN [V][v][k], h%16; PLE hash/conv reset вместе; RoPE64 j/j+32/absolute position, sections ARRAY INT32.

## Не повторять без причины

Не скачивать full BF16 на заполненный SSD/не менять рабочий ROCm/не переносить wave32 assumptions. QSA fork не HF oracle без boundary fixture. Router ждать /models.current.status=loaded/model=current, не только /health. MCP transfer local_root ограничен checkout; запрет не обходить.
