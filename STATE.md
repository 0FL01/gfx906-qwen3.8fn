# Текущее состояние

## Статус

R0, R1, R2a–R2e и R3a закрыты: primitives, dense projections/LM head (2026-10-02). R2–R8 целиком ещё не закрыты; модель новым core ещё не исполняется. Скорости PLAN.md остаются целями.

## Задача

R3b: прямой Session/48 layers с embeddings/LM head, layer split/cache и token-ID CLI; intermediates/teacher-forced logits. Все GPU projection/pointwise helpers integrated/qualified, не полный forward. No extra attention/FFN/final norm; Q/gate interleaved по512/head; MoE weights после down. Donor-first — PLAN.md/RECON.md §16. Основной KV target/MTP — Q4_0, expert layout canonical.

## Соединение и рабочие пути

Удалённое подключение: MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`; связь проверена 2026-10-01.
Controller repository: текущая рабочая директория OpenCode.
Основная модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 байт), источник по указанию пользователя: https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP sidecar: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 байт); источник загрузки не установлен.
Baseline image: `llama.cpp-gfx906:pp-stream-dcd685463d`; source: `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`.
Baseline config: `/home/radneon/llama/docker-compose.yml`, `/home/radneon/llama/models.ini` (не менялись). Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, один results.jsonl в корне. Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake 4.4.3, Clang 23, HIP 7.14.60850, rocBLAS 5.5; оба GEMM работают на gfx906. Runtime image не содержит CMake.

## Рабочие команды

После source sync: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=<синхронизированный-commit> -e CORE_DIRTY=OFF llama.cpp-gfx906:cmake-4.4.3 /core/src/tools/build.sh`.
Probe: `docker run --rm --name core-probe --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-probe -v /home/radneon/gfx906-core:/core llama.cpp-gfx906:cmake-4.4.3`.
Baseline: `sh /home/radneon/gfx906-core/src/tools/baseline-server.sh`, затем `python3 -B /home/radneon/gfx906-core/src/tools/baseline.py --runs-dir /home/radneon/gfx906-core/runs --results /home/radneon/gfx906-core/results.jsonl --revision dcd685463d597d31f5ca759d32c94592a2740fa4`.
Полные команды/log capture в README.md. Для dirty source задавать CORE_DIRTY=ON. Не совмещать model baseline и probe/тяжёлую сборку. Наш baseline-контейнер сейчас остановлен.
Expert: как probe, но entrypoint `/core/build/core-expert`, добавить `-v /home/radneon/models-nvme:/models:ro` и аргумент `/models/qwen38-keep1-Q4_0.gguf`; stdout → runs/r1-expert-final.jsonl. Inventory: core-inspect на обоих файлах без GPU devices. README.md содержит полные команды.
KV: как probe, entrypoint `/core/build/core-kv`; stdout → runs/r2-kv.jsonl. qsa-test без GPU → runs/r2-qsa-boundaries.jsonl; tools/record_kv.py валидирует/добавляет одну запись. Полные команды в README.md.
GDN: как expert, entrypoint `/core/build/core-gdn`, target GGUF; stdout → runs/r2-gdn.jsonl, stderr → runs/r2-gdn.err; tools/record_gdn.py --raw ... --results ... . Полная команда в README.md.
HC/PLE: CPU hc-test/ple-test --model target GGUF → runs/r2-hc-ple.jsonl; tools/record_hc_ple.py --raw ... --results ... . Команда в README.md.
QSA: как expert, entrypoint `/core/build/core-qsa`; stdout → runs/r2-qsa-gpu.jsonl, stderr → runs/r2-qsa-gpu.err; tools/record_qsa.py --raw ... --results ... . Не использовать имя CPU boundary log.
Linear/blocks: как expert, entrypoint `/core/build/core-linear` затем `/core/build/core-blocks`; logs runs/r2-linear-final.jsonl и r2-blocks.jsonl; tools/record_helpers.py --linear ... --blocks ... --results ... . Не запускать одновременно; полные команды README.md.
Dense/head: как expert, entrypoint core-dense затем core-head; logs runs/r3-dense.jsonl/r3-head.jsonl; tools/record_dense.py --dense ... --head ... --results ... . Команды README.md.

## Исходный baseline пользователя

2×16 GiB gfx906, Q4_0 keep1, layer split, cache112/inserts2, streaming+D2D, batch/ubatch1024, MTP2, q4_0 K/V, capacity131072.
512 input: PP 123.3 / TG 24.7; 4096 input: PP 220.7 / TG 23.7; 16384 input: PP 209.7 / TG 19.2.
Замеры сообщены пользователем, не перепроверены этим RECON. 32K/64K/128K в прежнем сообщении были экстраполяцией.

## Последний подтверждённый результат нового движка

R0 hardware: actual 16 physical/32 logical cores, один NUMA, AVX2/FMA/F16C; две wave64/CU60 gfx906 по17 163 091 968 bytes; copies/P2P/events/rocBLAS прошли. R1 target1224/sidecar32/expert прошли; CPU inlined/F16C принят A/B/A, GPU canonical. R2a Q4/Hadamard bytes exact; при2052 CPU source-emulation fork имеет3 extra IDs, не baseline GPU/logits доказательство. R2b GDN state/history/chunk/restore и R2c CPU actual HC/PLE gates прошли; hash baseline bug не найден. Подробные числа/сырые logs — один results.jsonl и README.md, не speed claims полного inference.
Новая baseline-серия, не оригинальные prompts: 4096+512 PP197.68/TG14.48, HTTP56.102s; sampling1/.95/20, MTP2/Q4 KV, один повтор без reuse. Occupancy/structured acceptance неизвестны; artifacts runs/r0-20261001T154816Z-4sffu3gz.
R2d: обе GPU прошли 62-row fixture: Q4/tail bytes, pooled/score gates, same-score exact IDs/counts, causal/future/unselected poison, chronological rollback/error/reuse. Pooled max error7.15e-7, scores4.77e-6, attention1.21e-5 (frozen gates2e-4+2e-4|ref|). Resident 128K score~95us/select~112us; attention2048~1.1ms — bottleneck, не ускорение. CPU raw-FP32 synthetic projections/caches, не occupied128K inference. Remote CTest16/16 и local ASan/UBSan прошли. Raw r2-qsa-gpu*, boundaries r2-qsa-boundaries.jsonl; CPU log recovered by exact deterministic replay после filename collision, отмечен в result. Compiled c100866 dirty1, не подменять новым commit.
R2e: обе GPU прошли 214+20-row common-Q8 linear/HC/PLE fixtures, gates unchanged. Remote CTest18/18/local common-Q8 ASan/UBSan; exact-byte producer/identical-input history и bounded parallel RMS. Это не full projection/inference; raw r2-linear-final/r2-blocks, compiled fa4ce49 dirty1. Ошибки fixture исправлены без изменения gates: read_slice по имени; tiny-scale max в каждом block; formatter переименован из-за std::quoted ADL.
R3a: dense/head обе GPU прошли, CTest19/19, recorder37. Gates unchanged; full dense output и head sampled oracle/finite/prefix bytes, не all-layer logits. Compiled ce05879 dirty1; raw r3-dense/r3-head. Числа в results.jsonl. SGEMM small shapes ~0.17ms — кандидат для профиля R3/R5, не переключать до full forward.

## Следующие действия

R3 Session/48 layers и separate baseline oracle. GDN own/HF L2 uses sum+1e-6, mx max(sum,eps²): диагностировать численную разницу отдельно от wiring. QSA block-selection difference отдельно от pure speedup. GDN [V][v][k], h%16, chronological slot n; PLE hash/conv возвращать совместно. RoPE64 split-half парыj/j+32, text pos абсолютный; sections ARRAY INT32. Source excerpts runs/donors; views r1-inventory.log. Worker pool/cold DDR/DMA — R3/R5.

## Не повторять без причины

Не скачивать весь BF16 checkpoint на заполненный SSD. Не менять рабочий ROCm. Не трактовать «MI50 32 GB» в fastfetch как VRAM. Не переносить wave32 assumptions. Не считать QSA fork эквивалентным HF без граничного теста. Router /health не означает loaded model: ждать /models.current.status=loaded и передавать model=current. После неудачной completion не повторять её вслепую. MCP transfer local_root ограничен checkout; временные source excerpts сохранять в runs/, не обходить запрет.
