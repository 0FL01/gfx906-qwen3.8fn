# Текущее состояние

## Статус

R0–R3 закрыты (2026-10-02): own48 Session, teacher32/full logits, reset/replay/greedy32 и capacity131072 owner/allocation ledger проходят. R4–R8 и целевые скорости не закрыты.

## Задача

R4a short-window Session закрыт: N2/3 grouped weight reuse, causal QSA, PLE/GDN chronology, 200 rows/full-logits bitwise replay, invalid/reset/ledger проходят. Job503 exit0: CTest24/24, default-N1 teacher32 error0 и unchanged capacity131072 owned allocation. Следующий срез R4b — подготовленные canonical DS4/MMQ primitives/fixture: прочитать driver, собрать и проверить обе GPU перед включением в bounded causal PP; R5 hybrid misses затем.
Raw runs/: `r4-session-batch.jsonl`, `r4-session-batch-final-build.log`, `r4-memory-default.jsonl`, `r4-default-n1-comparison.json`; journal14 records, прежние13 неизменны. Canonical journal только `/home/radneon/gfx906-core/results.jsonl`; не создавать src/results.jsonl.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller repository — текущий Git checkout; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления связи; не запускать второй экземпляр вслепую.
Основная модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 байт), источник по указанию пользователя: https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP sidecar: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 байт); источник загрузки не установлен.
Baseline image: `llama.cpp-gfx906:pp-stream-dcd685463d`; source: `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`.
Baseline config: `/home/radneon/llama/docker-compose.yml`, `/home/radneon/llama/models.ini` (не менялись); наш baseline-контейнер остановлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, один results.jsonl в корне. Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5, gfx906 GEMM проверены; runtime image не содержит CMake.
Actual hardware: 16 physical/32 logical allowed CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60, VRAM по17 163 091 968 bytes. Не повторять RECON без изменения машины/стека.

## Рабочие команды

После source sync: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=<синхронизированный-commit> -e CORE_DIRTY=OFF llama.cpp-gfx906:cmake-4.4.3 /core/src/tools/build.sh`.
Grouped/regression binaries compiled `997e197ea64e8b5a0c7983bd3cae29f4af379921` dirty1; старые CPW/route — 1322fd9 dirty1, R3 memory — c56cfa dirty1, teacher/gen/reset — e9f1dfe dirty1. Будущий commit не подменяет provenance уже записанных artifacts.
Session: `docker run --rm --name core-session --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-session -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 --generate 32 --ignore-eos /models/qwen38-keep1-Q4_0.gguf 248044`.
Teacher32: тот же entrypoint, `--trace /core/runs/NEW --logits /core/runs/NEW-logits.f32.bin /models/qwen38-keep1-Q4_0.gguf 248044 $(seq 100 130)`; parent runs/ должен существовать, trace/logits/log names свежие. Reset: entrypoint core-session-test, только MODEL. Полные build/fixtures/baseline/repro команды в README.md.
Oracle: `sh tools/build-oracle.sh` против build/oracle-production-libs; teacher reference использует `--all-layers --hf-gdn-l2-control --hf-qsa-f32-control --warm-cache 12 --cache-inserts 10` (diagnostic-only). compare_session.py сохраняет report/exit1 при failed gate; record_session.py --comparison/--generation/--reset/--results валидирует перед append. Одна GPU-нагрузка; final measurement без тяжёлой сборки/tracing.

## Последний подтверждённый результат

Teacher32 IDs `[248044,100..130]`: all7 946 240 finite logits numerical error0, argmax32/32; required intermediates проходят unchanged gates `.02+.002|ref|` logits/`.002+.002|ref|` intermediates/hc_init exact. Raw `runs/r3-session32-attention-order.jsonl`, одноимённый trace dir, `r3-session32-attention-order-logits.f32.bin`; report `r3-compare32-attention-order.json`.
Independent `r3-oracle32-hf-a`: production source/image-attested dcd685463d597d31f5ca759d32c94592a2740fa4, HF a005fc82babfe8871d87746decad2dbee100a125; opt-in additive-GDN-L2/FP32 gathered-QSA corrections. Не bitwise HF/unchanged-math/performance baseline; default production oracle неизменён.
Final reset8steps/capacity4/slots1/invalidIDs+capacity+bitwise replay passed. Greedy32/trace off: request2985.195379ms, load69364.627872ms отдельно; final token pending. Это short request, не steady TG/PP/sampling/A/B; полные numerics/provenance — results.jsonl/raw logs.
Capacity131072/slots112: min free VRAM6.67/6.23GiB, owned bytes восстановлены после destruction; no steady Buffer/read-counter growth. Это2 consumed tokens/pass, не occupied128K. Лог `r3-memory.jsonl`; full categories/RSS/private-memory limits — results.jsonl.

## Контракт и найденное ограничение

Own Session: RAM experts/GPU112 slots на слой, same-stream upload/reader/reuse ordering, static24/24, Q4 KV/FP32 index/GDN/PLE persistent. `step_batch` N1/2/3 с explicit max_batch_tokens; default1 сохраняет allocations/pinned40KiB, max3 увеличивает checked scratch/Q8/logits/handoff. Exact graph без ordinary attention/FFN/final norm; llama_decode только tools/oracle.cpp. Большой PP/CPU-worker overlap/MTP/sampling/HTTP/long-context ещё не квалифицированы.
Canonical ascending attention dot/value + chronological bounded gather и внутренний double-exp→FP32 — явные accuracy changes, selected multiset/repeatedIDs/public rankedIDs unchanged. A/B/A ~2× медленнее old topology, не speedup; MMVQ/MMVF — component qualification. GDN canonical recurrence+direct-logf softplus совпали с actual beta/log-decay/PRE/POST; старые softplus/prediction-FMA probes результата не меняли (RECON.md §10).
CLI greedy32 diagnostic-only, считать actual generated count; final emitted token остаётся pending. Primary series sampling temperature1.0/top-p0.95/top-k20 ещё впереди.

## Следующие действия

1. Квалифицировать prepared MMQ на actual gate/down/HC: unchanged Q8→DS4 layout bytes, Q4_1 half products, K640/tile tails и A/B/A против diagnostic sliced-linear; не считать последний production PP.
2. Bounded grouped Q4_0/Q4_1 staging/causal PP 4K/16K и peak VRAM; затем R5 CPU miss против H2D+GPU/overlap. DS4 MMQ144B/K128 не cast четырёх36B Q8; никогда не брать padding из соседнего cache slot.
Не менять frozen gates/epsilon/weights. Index raw128 без H/inverse; GDN [V][v][k], h%16; PLE hash/conv reset вместе; RoPE64 j/j+32/absolute position, sections ARRAY INT32.

## Не повторять без причины

Не скачивать full BF16 checkpoint на заполненный SSD/не менять рабочий ROCm/не переносить wave32 assumptions. Не считать QSA fork эквивалентным HF без boundary fixture. Router /health не значит loaded model: ждать /models.current.status=loaded, model=current. MCP transfer local_root ограничен checkout; не обходить запрет.
