# Текущее состояние

## Статус

R0–R3 закрыты (2026-10-02): own48 Session, teacher32/full logits, reset/replay/greedy32 и capacity131072 owner/allocation ledger проходят. R4–R8 и целевые скорости не закрыты.

## Задача

Qualified short-PP slice закрыт: job `1791004619377-704`/original compiled `15a025d9e105b704a628d0bbca003a08cb61b334` dirty1, broad closure job `1791011326789-719` exit0/strict Release build/CTest27/27. Текущая задача — large logical chunks/group>128/stage reuse, затем4K/16K PP/full-request/peak VRAM. R4–R8 и целевые скорости открыты.
Raw `runs/r4-prefill-short-expert8-ple8.jsonl` protocol1/92 records; collector `collect(actual_raw)` и local36/36 tests проходят. Canonical ROOT `/home/radneon/gfx906-core/results.jsonl` содержит18 records: ровно один validated `r4b_short_prefill` append, прежние17 byte-prefix/JSON unchanged, downloaded controller journal +1/-0. Original15a025d/dirtytrue сохранён; raw driver label `current_unqualified_candidate` исторический, не переименовывать.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller repository — текущий Git checkout; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления связи; не запускать второй экземпляр вслепую.
Основная модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 байт), источник по указанию пользователя: https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP sidecar: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 байт); источник загрузки не установлен.
Baseline image: `llama.cpp-gfx906:pp-stream-dcd685463d`; source: `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`.
Baseline config: `/home/radneon/llama/docker-compose.yml`, `/home/radneon/llama/models.ini` (не менялись); наш baseline-контейнер остановлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, один results.jsonl в корне. Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5, gfx906 GEMM проверены; runtime image не содержит CMake.
Actual hardware: 16 physical/32 logical allowed CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60, VRAM по17 163 091 968 bytes. Не повторять RECON без изменения машины/стека.

## Рабочие команды

После sync accepted dirty source: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=15a025d9e105b704a628d0bbca003a08cb61b334 -e CORE_DIRTY=ON llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'`. Для новой сборки указать actual revision/dirty. Initial job716 exit126: build.sh source100644/remote0664 nonexecutable; explicit `sh` retry719 прошёл, не math failure.
Session: `docker run --rm --name core-session --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-session -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 --generate 32 --ignore-eos /models/qwen38-keep1-Q4_0.gguf 248044`.
Teacher32: тот же entrypoint, `--trace /core/runs/NEW --logits /core/runs/NEW-logits.f32.bin /models/qwen38-keep1-Q4_0.gguf 248044 $(seq 100 130)`; parent runs/ должен существовать, trace/logits/log names свежие. Reset: entrypoint core-session-test, только MODEL. Полные build/fixtures/baseline/repro команды в README.md.
Oracle: `sh tools/build-oracle.sh` против build/oracle-production-libs; teacher reference использует `--all-layers --hf-gdn-l2-control --hf-qsa-f32-control --warm-cache 12 --cache-inserts 10` (diagnostic-only). compare_session.py сохраняет report/exit1 при failed gate; record_session.py --comparison/--generation/--reset/--results валидирует перед append. Одна GPU-нагрузка; final measurement без тяжёлой сборки/tracing.
Short PP: тот же Docker/device/mount контракт, entrypoint `/core/build/core-prefill-test`, только MODEL → свежий `runs/NEW-prefill-short.jsonl`; `python3 -B /home/radneon/gfx906-core/src/tools/record_prefill.py --raw /home/radneon/gfx906-core/runs/NEW-prefill-short.jsonl --results /home/radneon/gfx906-core/results.jsonl`. Проверить оба exit statuses; collector append ровно один record после полной validation. Полная команда/optional diagnostic trace — README.md.

## Последний подтверждённый результат

Short fixture: одна Session cap40/slots1/max32, retained N1 reference40, chunk4/chunk32/occupied-prefix5+N17+N10, по8 N1 continuations/phase. Все160 rows/80 windows:39 731 200 finite logits/29 798 400 comparisons, zero numerical violations/diagnostic bit mismatches; frozen `.02+.002|ref|`, bit/argmax identity не обязательны.15 atomic invalids;116 steady memory observations (86 serialized+30 summarized). Cleanup reported, owned-release не measured; workspace только aggregate floor, не individual-buffer proof/full-RAM accounting.
Trace32 job700: all7392 matched nodes/640FFN probes/fullhead exact; отдельный defaultN1 job708: original6912probes/fullteacher32 bitexact. Это не independent HF qualification short fixture. R3 reference `r3-oracle32-hf-a` — source/image-attested production с opt-in HF additive-GDN-L2/FP32 gathered-QSA controls, не bitwise HF/unchanged-math/performance baseline (README.md).
Broad719 elapsed13m50s/CTest525.55s: dense-MMVF both-GPU480 reported cases exact; linear-short747/device/1494 total,42 host rejects/device, common-Q8 maxabs4.291534423828125e-06/maxboundratio.01524, old API rejects N4. Sequential GDN/reset/batch/default memory(cap131072/slots112) process exits0; logs `runs/r4-shortpp-final-regression-sh.log`, `runs/r4-shortpp-final-{gdn,reset,batch,memory}.jsonl`. Actual `record_memory.collect` прошёл: steady owners, min free7 166 787 584/6 686 539 776 bytes и полный owned recovery. Это regression/capacity proof, не occupied128K/скорость; подробные historical R3 numerics — journal/README.md.

## Контракт и найденное ограничение

Own Session: RAM experts/default GPU112 slots/layer, same-stream slot ordering, static24/24, Q4 KV/FP32 index/GDN/PLE persistent; default max_batch_tokens1/pinned40KiB. Source bound logical≤1024, real short qualification только≤32. Nonexpert `Matrix::apply`/shared dense chronological tiles≤8 (MMVF1..8, fallback9..128 preserved), separate quantized-short≤8 со старым API≤3; expert microtiles≤8, routes once/full chunk, triplet once/group. Band16/two stages, `copy_ready`/`consumer_done` lifetime сохранены. PLE existing512-thread gate≤8/hash/norm/conv; GDN exact chronological CPW1..128/strengthened state+prefix+late-error. Новых arithmetic/weights/epsilon/gate changes нет. Exact graph/llama_decode-only-oracle контракт прежний; wide MMQ component-only, full PP speed не established.
Canonical ascending attention dot/value + chronological bounded gather и внутренний double-exp→FP32 — явные accuracy changes, selected multiset/repeatedIDs/public rankedIDs unchanged. A/B/A ~2× медленнее old topology, не speedup; MMVQ/MMVF — component qualification. GDN canonical recurrence+direct-logf softplus совпали с actual beta/log-decay/PRE/POST; старые softplus/prediction-FMA probes результата не меняли (RECON.md §10).
CLI greedy32 diagnostic-only, считать actual generated count; final emitted token остаётся pending. Primary series sampling temperature1.0/top-p0.95/top-k20 ещё впереди.

## Следующие действия

1. Проверить large logical chunks≤1024/expert group>128/multi-microtile stage reuse с causal/full-logit/reset/continuation gates. Max-group cardinality ещё не public API: доказать actual group sizes явно, не вывести их из chunk length.
2. Затем4K/16K без prefix reuse: PP/full-request/peak VRAM и paired performance; R5 CPU miss vs H2D+GPU/overlap остаётся. DS4 Q8 bits/padding owned. Sampler3 и `src/prefill_wide_test.cpp` uncommitted/unintegrated, вне short-PP slice. Wide-MMQ component wins не означают full PP speedup.
Не менять frozen gates/epsilon/weights. Index raw128 без H/inverse; GDN [V][v][k], h%16; PLE hash/conv reset вместе; RoPE64 j/j+32/absolute position, sections ARRAY INT32.

## Не повторять без причины

Не скачивать full BF16 checkpoint на заполненный SSD/не менять рабочий ROCm/не переносить wave32 assumptions. Не считать QSA fork эквивалентным HF без boundary fixture. Router /health не значит loaded model: ждать /models.current.status=loaded, model=current. MCP transfer local_root ограничен checkout; не обходить запрет.
