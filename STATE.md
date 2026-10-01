# Текущее состояние

## Статус

R0 закрыт 2026-10-01: C++20/HIP исходник собран удалённо, kernel/copy/event ordering и rocBLAS проверены на обеих gfx906; сохранены hardware probe и два запроса production baseline. Это build/hardware slice, не inference нового core; R1–R8 ещё не закрыты. Скорости PLAN.md остаются целями.

## Задача

R1: валидированный GGUF loader и реальный expert слоя 0 (gate/up Q4_0, down Q4_1), scalar/AVX2/HIP parity и N=1/2/3/PP замеры. Loader уже написан, но ещё не включён в законченный срез. Canonical mx SDOT4/DPP первым; planar mx/reinstinct — кандидаты. Donor-first правило — PLAN.md и RECON.md, раздел 16. Основной KV target/MTP — Q4_0.

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

## Исходный baseline пользователя

2×16 GiB gfx906, Q4_0 keep1, layer split, cache112/inserts2, streaming+D2D, batch/ubatch1024, MTP2, q4_0 K/V, capacity131072.
512 input: PP 123.3 / TG 24.7; 4096 input: PP 220.7 / TG 23.7; 16384 input: PP 209.7 / TG 19.2.
Замеры сообщены пользователем, не перепроверены этим RECON. 32K/64K/128K в прежнем сообщении были экстраполяцией.

## Последний подтверждённый результат нового движка

R0 probe завершён с passed=true: actual 16 physical/32 logical cores, один NUMA, AVX2/FMA/F16C; две wave64/CU60 gfx906 по 17 163 091 968 bytes. Проверены H2D consumer, D2D readback, двусторонний P2P producer/consumer (20 epochs), оба rocBLAS GEMM (max_abs_error=0), CPU FMA + dual H2D. Числа в results.jsonl; сырые логи `/home/radneon/gfx906-core/runs/r0-{build.log,probe.jsonl,probe.err}`.
Новая baseline-серия (не пользовательские prompts): 32+64 — PP 12.00/TG 10.59, HTTP 8.715 s; 4096+512 — PP 197.68/TG 14.48, HTTP 56.102 s. Один повтор, без prefix reuse, temperature1/top-p.95/top-k20, ignore_eos=true, MTP2/Q4 KV. Cache cold/warm только по протоколу, occupancy неизвестна; короткий warmup не равен fully-warm. Structured acceptance histogram отсутствует, сохранён null. Artifacts `/home/radneon/gfx906-core/runs/r0-20261001T154816Z-4sffu3gz`; raw server log `r0-baseline-server.log`.
Проверки: удалённый Release build с -Werror; CTest baseline-client (15 cases) и те же 15 локальных Python tests прошли. RAM read/FMA — синтетический тест, не скорость эксперта.

## Следующие действия

Подключить src/model.cpp и 196 loader checks к CMake; проверить inventory обоих реальных файлов. Для expert fixture согласовать Q8_1 ABI (FP16 scale + FP16 raw FP32 input sum, roundf), Q4_1 correction и DPP reduction; проверять идентичные quantized activations до float-input pipeline. Source excerpts в игнорируемом runs/donors. Затем N=1/2/3 и PP128, CPU miss против resident GPU и H2D+GPU. Сначала non-speculative core без изменения весов, MTP sidecar позже.

## Не повторять без причины

Не скачивать весь BF16 checkpoint на заполненный SSD. Не менять рабочий ROCm. Не трактовать «MI50 32 GB» в fastfetch как VRAM. Не переносить wave32 assumptions. Не считать QSA fork эквивалентным HF без граничного теста. Router /health не означает loaded model: ждать /models.current.status=loaded и передавать model=current. После неудачной completion не повторять её вслепую. MCP transfer local_root ограничен checkout; временные source excerpts сохранять в runs/, не обходить запрет.
