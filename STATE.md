# Текущее состояние

## Статус

RECON по открытым исходникам и read-only remote RECON выполнены 2026-10-01. Связь, аппаратная topology, GGUF inventory и существующий baseline обнаружены (подробности в RECON.md, раздел 4). R0 не закрыт: новый HIP probe/kernel/copy, rocBLAS и baseline-прогон ещё не выполнялись. Реализации нового движка нет; скорости в PLAN.md остаются целями.

## Задача

Начать R0 из PLAN.md. После R0 перейти к R1 без повторного согласования архитектуры.

## Соединение и рабочие пути

Удалённое подключение: MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`; связь проверена 2026-10-01.
Controller repository: текущая рабочая директория OpenCode.
Основная модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 байт), источник по указанию пользователя: https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP sidecar: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 байт); источник загрузки не установлен.
Baseline image: `llama.cpp-gfx906:pp-stream-dcd685463d`; source: `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`.
Baseline config: `/home/radneon/llama/docker-compose.yml`, `/home/radneon/llama/models.ini`. Toolchain найден в Docker-образах; версии и работоспособность HIP/rocBLAS ещё проверить. Remote source/build/runs нового core ещё не созданы.

## Исходный baseline пользователя

2×16 GiB gfx906, Q4_0 keep1, layer split, cache112/inserts2, streaming+D2D, batch/ubatch1024, MTP2, q4_0 K/V, capacity131072.
512 input: PP 123.3 / TG 24.7; 4096 input: PP 220.7 / TG 23.7; 16384 input: PP 209.7 / TG 19.2.
Замеры сообщены пользователем, не перепроверены этим RECON. 32K/64K/128K в прежнем сообщении были экстраполяцией.

## Последний подтверждённый результат нового движка

Нет.

## Следующие действия

Через `mi50-llama-remote` проверить HIP compiler/runtime внутри найденного образа, выбрать source/build/runs нового core на корневом разделе и создать простой build/run цикл. Выполнить HIP device probe, kernel/copy smoke и rocBLAS GEMM, затем короткий baseline. В R1 реализовать loader найденных типов и первый expert fixture: у слоя 0 gate/up Q4_0, down Q4_1. Сначала non-speculative путь на выбранном GGUF без реквантизации, затем существующий MTP sidecar; другие локальные модели не включать в стартовый набор.

## Не повторять без причины

Не скачивать весь BF16 checkpoint на заполненный SSD. Не менять ROCm до проверки существующего. Не трактовать «MI50 32 GB» в fastfetch как фактическую VRAM. Не переносить wave32 assumptions из Strata. Не считать top-k QSA форка математически эквивалентным HF без граничного теста.
