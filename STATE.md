# Текущее состояние

## Статус

R0, R1, R2a–R2e, R3a и R3b закрыты (2026-10-02). Own48-layer Session: teacher32 logits/intermediates, final reset/replay и greedy32 прошли. R3 ещё требует per-GPU/category allocation ledger; R4–R8 и целевые скорости не закрыты.

## Задача

Commit/push verified R3b, затем R3 per-GPU ownership/allocation ledger и capacity131072 headroom; далее R4 grouped PP/R5 hybrid misses. Full job474 exit0, CTest21/21 (comparison31/recorder24), reset8steps/invalid-ID/capacity/replay и greedy32 passed.
Raw runs/: `r3-session-closure-portable-build.log`, `r3-session-reset-final.jsonl/.err`, `r3-generation-final.jsonl/.err`. Canonical results содержит11 records, прежние10 неизменны. Missing remote fixture/controller TMP path исправлены без gate changes; случайный src/results record проверен против canonical append и удалён, failed logs сохранены.

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
Текущие numerical binaries compiled `e9f1dfe57cf8fdc0abd9ba10ab91cfbe01d9e0db` dirty1; для этого source CORE_DIRTY=ON. Будущий commit не подменяет provenance уже записанных artifacts.
Session: `docker run --rm --name core-session --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-session -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 --generate 32 --ignore-eos /models/qwen38-keep1-Q4_0.gguf 248044`.
Teacher32: тот же entrypoint, `--trace /core/runs/NEW --logits /core/runs/NEW-logits.f32.bin /models/qwen38-keep1-Q4_0.gguf 248044 $(seq 100 130)`; parent runs/ должен существовать, trace/logits/log names свежие. Reset: entrypoint core-session-test, только MODEL. Полные build/fixtures/baseline/repro команды в README.md.
Oracle: `sh tools/build-oracle.sh` против build/oracle-production-libs; teacher reference использует `--all-layers --hf-gdn-l2-control --hf-qsa-f32-control --warm-cache 12 --cache-inserts 10` (diagnostic-only). compare_session.py сохраняет report/exit1 при failed gate; record_session.py --comparison/--generation/--reset/--results валидирует перед append. Одна GPU-нагрузка; final measurement без тяжёлой сборки/tracing.

## Последний подтверждённый результат

Teacher32 IDs `[248044,100..130]`: all7 946 240 finite logits numerical error0, argmax32/32; required intermediates проходят unchanged gates `.02+.002|ref|` logits/`.002+.002|ref|` intermediates/hc_init exact. Raw `runs/r3-session32-attention-order.jsonl`, одноимённый trace dir, `r3-session32-attention-order-logits.f32.bin`; report `r3-compare32-attention-order.json`.
Independent `r3-oracle32-hf-a`: production source/image-attested dcd685463d597d31f5ca759d32c94592a2740fa4, HF a005fc82babfe8871d87746decad2dbee100a125; opt-in additive-GDN-L2/FP32 gathered-QSA corrections. Не bitwise HF/unchanged-math/performance baseline; default production oracle неизменён.
Final reset8steps/capacity4/slots1/invalidIDs+capacity+bitwise replay passed. Greedy32/trace off: request2985.195379ms, load69364.627872ms отдельно; final token pending. Это short request, не steady TG/PP/sampling/A/B; полные numerics/provenance — results.jsonl/raw logs.

## Контракт и найденное ограничение

Own Session: RAM experts/GPU112 slots на слой, same-stream upload/reader/reuse ordering, static24/24, pinned40KiB handoff, Q4 KV/FP32 index/GDN/PLE persistent. Exact graph без ordinary attention/FFN/final norm; llama_decode только tools/oracle.cpp. Это orderedN1 decode, не grouped PP/CPU-worker overlap/MTP/sampling/HTTP/long-context qualification.
Canonical ascending attention dot/value + chronological bounded gather и внутренний double-exp→FP32 — явные accuracy changes, selected multiset/repeatedIDs/public rankedIDs unchanged. A/B/A ~2× медленнее old topology, не speedup; MMVQ/MMVF — component qualification. GDN canonical recurrence+direct-logf softplus совпали с actual beta/log-decay/PRE/POST; старые softplus/prediction-FMA probes результата не меняли (RECON.md §10).
CLI greedy32 diagnostic-only, считать actual generated count; final emitted token остаётся pending. Primary series sampling temperature1.0/top-p0.95/top-k20 ещё впереди.

## Следующие действия

1. Сверить фактические per-GPU/category allocations и headroom с capacity131072/R3 geometry; сохранить actual owner/steady allocation proof.
2. R4 route grouping → staged grouped Q4_0/Q4_1 MMQ (mx → furnace → reinstinct, RECON §16), causal chunk parity; затем R5 CPU miss против H2D+GPU и overlap на measured shapes. DS4 MMQ144B/K128 не cast четырёх36B Q8; K640 требует checked tail padding. N3 QSA projected buffer36864 floats/Q8 input960 blocks больше текущих32768/512.
Не менять frozen gates/epsilon/weights. Index raw128 без H/inverse; GDN [V][v][k], h%16; PLE hash/conv reset вместе; RoPE64 j/j+32/absolute position, sections ARRAY INT32.

## Не повторять без причины

Не скачивать full BF16 checkpoint на заполненный SSD/не менять рабочий ROCm/не переносить wave32 assumptions. Не считать QSA fork эквивалентным HF без boundary fixture. Router /health не значит loaded model: ждать /models.current.status=loaded, model=current. MCP transfer local_root ограничен checkout; не обходить запрет.
