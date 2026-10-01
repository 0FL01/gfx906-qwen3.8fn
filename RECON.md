# RECON и архитектура: Qwen3.8-Flash-Next на 2×gfx906

Дата проверки открытых источников: 2026-10-01.

Это архитектурное решение и разведка по исходникам с read-only remote RECON от 2026-10-01. Последующий R0 build/probe и новая baseline-серия также выполнены; результаты и ограничения находятся в разделе 4, STATE.md и results.jsonl. Полный inference нового core ещё не реализован.

## 1. Решение и границы

Делать отдельный C++20/HIP executable и небольшую core-библиотеку. Никакого универсального графа вычислений: заранее известная последовательность 48 слоёв и отдельный MTP-блок. Два физических GPU, один запрос, текстовый ввод, статический layer split. CPU выполняет назначенные ему MoE-experts, GPU держит attention/GDN, dense/shared weights и кеш routed experts.

Сначала читать имеющийся GGUF, затем поддержать собственный runtime-pack из safetensors. Первый этап сохраняет существующие значения весов. Квантование и восстановление полной PLE являются отдельными изменениями модели.

Конечный runtime не должен вызывать llama_decode для основного forward. Существующий fork нужен как эксплуатационный baseline, источник проверенных приёмов и средство выгрузить небольшие эталонные тензоры. Заимствовать маленький компонент полезнее, чем воспроизводить всё ради авторства.

Не включать в первую версию vision, LoRA, multi-user serving, TP/AllReduce, paged KV, speculative tree, distributed execution и многоархитектурный backend. Внешний API появится после CLI с реальной генерацией.

## 2. Что подтверждено

### Геометрия модели

Оригинальный checkpoint: qwen4_exp; 48 основных слоёв с чередованием трёх GDN и одного QSA. Hidden 2560; residual 4×2560; HC rank 320. Routed 512/top10, shared expert один, FFN intermediate 640. QSA: Q24/KV2, head 256; индексатор Q4/K1, head 128, блок 4, budget 2048. GDN: QK16/V48, head 128, conv4, output gate sigmoid. MTP: отдельный один слой, не две независимые головы. Native context 262144, но первая целевая ёмкость core 131072. [S1, S2]

В конфигурации HF `ple_layer_ids=[2]` использует однобазовую нумерацию; в нулевой нумерации это слой 1. `ple_embed_dim=2560` является полной шириной всех n-gram heads, а не шириной одного head. Эти соглашения подтверждены реализацией Transformers. [S3]

Имена `full_attention` в config обозначают здесь индексируемые QSA-слои; трактовать их как обычный dense Transformer без индексатора нельзя. [S3]

### Существующая модель пользователя

`qwen38-keep1-Q4_0.gguf`, около 75.4 GB. Keep1 сохраняет две из шестнадцати PLE hash-head через metadata и нулевые строки, но не сокращает routed-experts. Не переписывать архитектуру на «две головы общей шириной 320»: downstream PLE geometry остаётся частью исходной модели. Читать actual metadata и offsets. [S4]

Удалённые PLE-heads являются изменением модели, а не обычной квантизацией. Карточка варианта содержит небольшие оценки качества другой, Q3_K_XL, версии; они не подтверждают качество пользовательского Q4_0 на всех задачах. Предупреждение карточки об отсутствии MTP устарело относительно используемого fork: у fork есть отдельный MTP path. [S4, S5]

### Native instructions уже используются в fork

В `ggml/src/ggml-cuda/common.cuh` production-версии есть gfx906 `__builtin_amdgcn_sdot4`, FP16 dot2 и специализированные DPP reductions. Новый движок не получает эти возможности впервые. Возможный выигрыш: меньше загрузок, меньше launch/host waits, полезнее layout, специализированная обработка форм 1/2/3 и больших PP-чанков. [S6]

LLVM-тесты подтверждают `v_dot8_i32_i4` на gfx906. Это I4×I4: использование требует подходящих активаций или дополнительных операций разложения. Нельзя объявить его прямым бесплатным W4A8 kernel. [S7]

### Strata не является готовым gfx906-портом

В AMD backend Strata явно заявлены wave32 и RDNA3/RDNA4, wave64 исключён из текущего scope. Заимствовать алгоритмы и формы данных, но проверять shuffle, ballot, reductions, LDS и размер wave заново. [S8]

У Strata есть уже специализированные QSA, GDN, CPU/GPU MoE и speculative paths. Однако отдельные файлы содержат исторические комментарии и CUDA-специфичные решения. Не переносить весь проект или каждую страховку из него. [S9, S10]

## 3. Самое важное расхождение, которое нужно разобрать до оптимизаций QSA

В Transformers QSA выбирает до 512 ПОЛНЫХ блоков по 4 видимых токена и добавляет ФАКТИЧЕСКИЙ незавершённый хвост. В выходном буфере предусмотрено до 2051 позиции, но незаполненные элементы являются padding, а не дополнительными выбранными токенами. Pooling применяется до norm/RoPE, позиция блока берётся от его первого элемента. [S3]

В проверенном production fork `build_qsa_top_k()` разворачивает оценки блоков в оценки позиций и выполняет top-k шириной `min(n_kv,2048+4-1)`. Затем `build_attn_qsa()` строит полную маску и передаёт полные K/V в обычный attention dispatch. [S5]

Эти формы нельзя считать автоматически эквивалентными. Например, при 2052 видимых токенах HF выбирает 512 завершённых блоков, то есть 2048 токенов без хвоста. Внешне видимая ширина fork top-k равна 2051; нужно проверить actual IDs/mask, padding, bias и tie-handling. Это диагностическая гипотеза о семантике, не заявление об установленном end-to-end дефекте всех запусков.

Обязательный первый QSA-тест: длины 2047–2056, все четыре фазы хвоста, равные/нулевые scores и разные causal prefixes в prefill. Раздельно сравнить число ВАЛИДНЫХ выбранных позиций, сами IDs и итоговый attention output.

Если расхождение подтвердится, правильность архитектуры сверять с официальной reference, а старую скорость хранить как legacy baseline с явной пометкой. Не ослаблять тест для совпадения и не выдавать изменение алгоритма за только оптимизацию. Постоянный compatibility framework не нужен: достаточно маленького diagnostic oracle и объяснения.

## 4. Аппаратный RECON на машине

### Подтверждено на remote 2026-10-01

Подключение: MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`, Ubuntu 26.04.1, kernel `7.0.0-30-generic`. Проверка была read-only: без запуска контейнеров, загрузки полной модели, сборки и performance-прогонов.

**Зафиксированный стартовый набор GGUF:**

| Роль | Remote path | Размер, байт |
| --- | --- | ---: |
| Target keep1 | `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` | 75 399 121 792 |
| MTP shared sidecar | `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2 786 568 256 |

Источник target по указанию пользователя — Cyronius [S4]; карточка описывает Q4_0 keep1 из bartowski Q4_0. Локальные metadata согласуются с вариантом, но byte-for-byte идентичность удалённому файлу не проверялась. Источник загрузки sidecar не установлен. Сначала использовать target без реквантизации в non-speculative runtime, затем этот sidecar для MTP2. IQ1_M и остальные локальные модели не входят в стартовый набор.

Оба файла GGUF v3, `general.architecture=qwen4exp`. Для всех 1224 target и 32 sidecar тензоров проверены размеры по dimensions/type, block divisibility, alignment, отсутствие перекрытий и выходов за файл; ошибок нет. Это проверка inventory и диапазонов, не end-to-end correctness модели.

Target: 48 слоёв, hidden 2560, 512/top10 experts, FFN 640, native context 262144. Реальные типы (число тензоров): F32 460, F16 1, BF16 122, Q4_0 442, Q4_1 6, Q5_0 72, Q8_0 114, Q6_K 7. Все routed gate/up — Q4_0; routed down слоёв 0–5 — Q4_1, остальных 42 — Q4_0. Название файла не означает однородный Q4_0: первый expert слоя 0 требует Q4_1 down.

PLE: GGUF `ple.layers=[1]` (нулевая нумерация), таблица `[160, 40000085]` Q4_0. Сохранены головы 0 и 8 с vocab sizes 20000003 и 20000081 и offsets 0 и 20000003. Остальные 14 имеют vocab size 1 и offset 40000084; у общей строки прочитаны нулевые scales всех Q4_0 блоков. PLE EOS 248044 отличается от tokenizer EOS 248046; не объединять эти значения.

Sidecar: `nextn_shared_target_tensors=true`, `nextn_predict_layers=1`, тензоры блока `blk.48`, без собственных embedding/output. Типы: Q8_0 19, F32 11, BF16 2; все три routed expert tensors — Q8_0. Tokenizer обоих файлов: `gpt2`/`qwen35`, BOS 248044, EOS 248046. Совпадение полного tokenizer и поведение совместного MTP ещё не проверены.

**Аппаратная часть:** Xeon E5-2698B v3, 1 сокет, 16 физических cores/32 threads, AVX2/FMA/F16C, один NUMA-узел. RAM около 121 GiB (118 GiB available на момент проверки); частота DIMM не проверена. KFD сообщает две gfx906, wave64, 240 SIMD (60 CU). Sysfs VRAM — 17 163 091 968 байт на каждой карте (~15.98 GiB); HIP device properties ещё не получены. GPU BDF `0000:05:00.0` и `0000:08:00.0`. Root ports `00:02.0`/`00:03.0` показывают 8 GT/s ×16 (PCIe3); внутренние мосты GPU показывают 16 GT/s ×16, что не повышает полосу пути до CPU. P2P permission/copy и полоса не измерены.

Model NVMe `/dev/nvme0n1p1`: 239 GiB, свободно около 39 GiB. Корневой `/dev/sda2`: 468 GiB, свободно около 223 GiB. Сборку и runs нового core размещать на корневом разделе; модели не копировать.

**Существующий baseline:** образ `llama.cpp-gfx906:pp-stream-dcd685463d` (image ID prefix `e6be1a2fcba8`); исходники `/home/radneon/src/worktrees/qwen38-pp-trace-75`, HEAD `dcd685463d597d31f5ca759d32c94592a2740fa4`. Конфигурация `/home/radneon/llama/docker-compose.yml` и `models.ini` указывает оба выбранных GGUF: DIO, cache112/inserts2, layer split 1:1, batch/ubatch1024, threads16, Q4_0 K/V, capacity131072, MTP2 на ROCm1, temperature1.0/top-p0.95/top-k20, min-p0, repeat-penalty1. Streaming и prefill D2D включены, подробный trace выключен. Запущенных Docker-контейнеров на момент проверки нет; baseline не воспроизводился.

ROCm отсутствует в host `/opt`; доступные gfx906 Docker-образы используют существующий стек. Найденный `Dockerfile-build-llama` использует base `mixa3607/llama.cpp-gfx906:b10808-rocm-7.14-mxxm-20260826041541-pre`, `HIPCXX` из `hipconfig`, `AMDGPU_TARGETS=gfx906`. Тег образа не является проверкой compiler/runtime версии: точные версии, HIP kernel и rocBLAS GEMM ещё проверить внутри контейнера. Доступ пользователя к `/dev/kfd` и render nodes есть.

### Исполнимый R0 после read-only RECON, 2026-10-01

Source mirror `/home/radneon/gfx906-core/src`, build/runs на том же корневом разделе; существующие веса не копировались. `llama.cpp-gfx906:cmake-4.4.3` содержит CMake 4.4.3, `/opt/rocm/llvm/bin/clang++` Clang 23, HIP runtime/driver 7.14.60850 и rocBLAS 5.5. Production runtime image не содержит CMake. C++20/HIP20 с явным gfx906 собраны без смены стека; обе rocBLAS SGEMM fixtures прошли с max_abs_error=0.

HIP properties подтвердили обе gfx906:sramecc+:xnack-, wave64, CU60, 8 async engines и 17 163 091 968 VRAM bytes на GPU, включая BDF из topology. Runtime CPU probe подтвердил 16 физических cores/32 разрешённых logical CPUs, один NUMA, AVX2/FMA/F16C. Kernel/H2D consumer, D2D readback и event-ordered P2P producer/consumer (20 epochs, оба направления) проверены. P2P доступен, но измеренный последний 4-MiB transfer — около 3 GB/s; это диагностическая точка, не основание считать P2P быстрее staging. RAM read/FMA и overlap с dual H2D синтетические, не CPU expert GEMV. Все численные измерения — в одном results.jsonl и raw r0-probe.jsonl.

Production baseline выполнен отдельно с MTP2/Q4 target+draft KV, тем же preset и traces off, после чего наш core-baseline остановлен. Новые fixtures сохраняют exact 32/4096 prompt IDs и 64/512 output IDs: это не оригинальные prompts пользователя. PP/TG/HTTP и raw artifacts записаны в results.jsonl; prefix reuse выключен, temperature1/top-p.95/top-k20, ignore_eos=true. Один короткий запрос перед 4K не доказывает полностью прогретый expert cache. Histogram 0/1/2 отсутствует в structured response и оставлен null. Router требует model=current в tokenize/completion и ожидания /models status=loaded; одного /health недостаточно.

### Controller и GPU-host

На controller остаётся единственный редактируемый Git checkout. На GPU-host выбрать одну директорию с `src/`, `build/` и `runs/`; существующие модели остаются по найденным путям. `src/` является зеркалом исходников, не второй независимо редактируемой копией. Если аварийное исправление сделано на GPU-host, сначала вернуть его в Git checkout, затем выполнять следующий sync.

Пересылать изменения исходников/маленьких fixtures обычным rsync по SSH. Исключить `.git`, build, runs, веса и скачанные checkpoints. На стороне GPU-host сохранять build и run logs между итерациями; повторная полная сборка нужна при смене compiler/flags, а не при каждом изменении функции. Не использовать удаление файлов при sync для общей директории с моделями. Никакой собственной системы deploy.

Сначала SSH-команды могут быть foreground. Для длинного benchmark достаточно имеющегося tmux либо одного отслеживаемого процесса с PID и логом. После обрыва проверить этот процесс, не запускать второй. Сохранять exit status удалённой сборки и запуска. Heavy compile, conversion и финальный CPU/GPU performance run не исполнять одновременно. Тонкий remote helper добавить только когда уже повторяются одни и те же команды.

### Что обнаружить

HIP device properties обеих GPU: `gcnArchName`, `totalGlobalMem`, `warpSize`, CU count, `asyncEngineCount`; свободная память после остановки других нагрузок. PCI BDF каждой карты; link speed/width под нагрузкой; P2P permission и фактическая корректность копий. Не путать локальный D2D и межкарточный transfer.

CPU: физические core IDs, SMT siblings, AVX2/FMA/F16C, NUMA nodes и их distance; доступная память и рабочая частота DIMM по данным системы. Односокетная машина не означает автоматически два NUMA-узла из-за двух GPU. Если узел один, NUMA-сложность заканчивается на affinity/first-touch.

Рабочий toolchain: версия и путь compiler/runtime/rocBLAS, фактический gfx906 GEMM, доступный profiler. Не требовать конкретный номер ROCm из интернета. Уже работающий локальный stack является предпочтительной точкой старта; HIP compiler и rocBLAS payload проверить отдельно. [S11]

### Минимальные микрозамеры

Замерить последовательное и экспертоподобное чтение RAM на нескольких размерах pool/числах физических потоков; размер массива должен превосходить LLC. Не выдавать memcpy bandwidth за bandwidth CPU GEMV.

H2D из обычного и pinned host memory: малые transfers активаций и transfers порядка одного эксперта, группы экспертов, большого staging tile. Измерить обе GPU отдельно и одновременно. Затем тот же transfer совместно с CPU expert compute. Этот последний тест важнее суммы двух изолированных пиков.

HBM read/device copy; inter-GPU copy в обе стороны; empty launch/event latency; один маленький и один PP-sized GEMM. Доказать producer-copy-consumer ordering повторяющимся проверяемым шаблоном.

Полученные значения являются входом в scheduler. Не тратить первую итерацию на все возможные counters и тысячи размеров.

### Физические ограничения

Из заявленной DDR3-1600 quad-channel следует теоретический предел 51.2 GB/s: 4×8×1.6. Это расчёт, не результат машины. PCIe3×16 имеет около 15.75 GB/s до transaction overhead; link topology ещё нужно установить. Два одновременных H2D и CPU experts конкурируют за DDR3.

При layer split один decode идёт последовательно через обе половины сети. Суммировать пиковые TFLOPS/HBM двух GPU и делить на активные параметры для прогноза TG неправильно. Две карты прежде всего дают вместимость; PP позже можно pipeline-ить по чанкам.

### Toolchain и ISA

CMake: языки CXX и HIP, стандарт 20, явный gfx906. Host flags задавать только CXX target, device flags только HIP. Не передавать `-march=native` с контроллера или AVX2 flags в device compilation. CMake имеет штатный `CMAKE_HIP_ARCHITECTURES`; при native HIP language использовать подходящий ROCm clang, не подменять CMake compiler wrapper-ом без проверки. [S12]

Для throughput kernels сначала обычный HIP и compiler intrinsics. Inline AMDGCN asm применять после disassembly/микрозамера, локально. Проверять register pressure, spills, LDS usage и wave64. Не отключать globally FP-ограничения `fast-math`, чтобы скрыть numerical bugs.

Основной GPU dot-path: W4A8/INT32 accumulation со scales. FP16 dot2 с FP32 accumulation исследовать для подходящих attention/dense forms. GDN state и reductions/softmax/router держать в FP32 в начальной корректной реализации.

CPU Q4×Q8: AVX2 unpack, integer dot и scale application. `maddubs` имеет unsigned×signed semantics и saturating pair sums. Для Q4 unpack в 0..15 допустим отдельный offset-corrected путь; доказать отсутствие saturation на поддерживаемом диапазоне, отдельно проверять -128. На W8A8 такую арифметику автоматически не распространять. [S13]

## 5. Минимальная структура кода

Структура ориентировочная. Не создавать пустые файлы и абстракции заранее.

```text
src/
  main.cpp                 CLI / benchmark commands
  model.cpp                GGUF views, потом runtime-pack
  runtime.cpp              фиксированная последовательность слоёв
  memory.cpp               host arena, GPU buffers, expert slots
  moe.cpp                  routes, CPU/GPU assignment, cache admission
  mtp.cpp                  draft / verify / commit state
  cpu/                     AVX2 expert kernels и постоянный worker pool
  hip/                     GEMV/GEMM, HC, GDN, QSA, маленькие операции
  sampling.cpp             сначала greedy, затем target sampling/MTP correction
  server.cpp               поздний минимальный serving
  ...                      добавлять только реально используемое

tools/
  remote.sh                один sync/build/run helper по необходимости
  convert.py               GGUF metadata/safetensors -> простой pack
  bench.py                 несколько fixtures, JSONL, summary

tests/
  ...                      конкретные математические и state tests
```

Runtime вызывает конкретные функции, не строит graph IR. Host-visible TensorView достаточно хранить pointer, type, dimensions и strides. Ownership у model/session/arena, не у каждого промежуточного tensor. Несколько switch по реальным weight types лучше системы регистрации kernels.

Предусмотреть прямые операции `prefill`, `decode_one`, `verify_window`, `commit_prefix`, `reset`. Имена являются проектируемым интерфейсом, готовых команд/функций пока нет. MTP не должен вынуждать переделывать слой после его реализации: операции получают абсолютные позиции и число токенов, короткое окно поддерживается как отдельная форма исполнения.

## 6. Память и размещение

### Host

Один canonical host inventory routed-experts. На первом этапе никаких двух полных копий CPU/GPU форматов в RAM. CPU-friendly storage и GPU layout могут отличаться, но repack выполнять для загружаемого tile/cache slot либо одноразово при подготовке pack.

PLE lookup-таблица остаётся в RAM. Из неё выбираются строки нужных n-grams; целую таблицу на каждый токен не копировать. CPU gather сначала простейший, затем batch gather/prefetch. Нулевые головы keep1 остаются точными нулями. Строки можно подготовить до PLE-layer, когда input token IDs уже известны.

Модель сначала грузить явным чтением в предсказуемую host memory: у текущего пользователя already-qualified DIO stack. Не предполагать, что mmap обязательно плох на новом runtime, но и не делать demand paging необходимой частью benchmark.

### GPU

На каждой GPU: её dense/shared weights, GDN state/conv state, K/V и индексатор её QSA-слоёв, expert-cache slots, фиксированный workspace. Последняя GPU держит LM output и MTP. Target embeddings и output matrix не считать связанными: original config содержит untied weights; MTP может заимствовать обе разные матрицы target. [S1, S5]

Первый split около 24/24, но выбрать границу по реальному VRAM budget с учётом draft, а не упрямо по равенству слоёв. У каждого слоя ровно один GPU owner. KV и его эксперты не переезжают между картами на каждом токене.

Одна сессия: сплошные заранее выделенные K/V массивы и счётчик committed length. Число выданных токенов не отождествлять с числом уже потреблённых forward: pending sampled token может ещё отсутствовать в состоянии. Откат не требует стирания всего хвоста: недоступные позиции исключаются по length и позже перезаписываются. Prefix editing сначала решать reset+prefill, а append-history reuse поддержать после correctness. Произвольные snapshots на диск не нужны для MVP.

### Начальная точность кешей

Для маленьких correctness fixtures использовать FP32/FP16 где это упрощает эталон. Основной режим target и MTP требует Q4_0 для обоих K/V, включая rotations и scale semantics baseline. Q8_KV — только отдельный сравнительный эксперимент, не замена обязательного Q4 и не весовая квантизация. Pooled index keys сначала FP32; raw index keys перед pooling проходят Q4_0 quantize/dequantize как в baseline.

### Расчёт вместимости, не замер

Один routed expert содержит 3×2560×640=4,915,200 параметров. Q4_0 имеет 18 байт/32 значения, поэтому один эксперт занимает 2,764,800 байт без дополнительного padding. Все 48×512 экспертов: 67,947,724,800 байт. Кеш 112 на каждый слой: 14,863,564,800 байт, около 13.84 GiB суммарно. [S1, S14]

В core 112 является baseline для сравнения, не обязательным оптимумом. Вместимость и hit rate различаются. MTP/shared/dense веса в расчёт выше не включены.

При 128K контекста только основные QSA K/V без padding занимают 3 GiB FP16 или 864 MiB при 4.5 bpw Q4_0. Pooled index keys FP32 занимают 192 MiB. Эти оценки получены из геометрии, MTP и temporary workspace считаются отдельно.

GDN state основных 36 слоёв при форме 48×128×128 FP32 составляет 108 MiB. Это объясняет, почему несколько коротких speculative states можно разместить в VRAM, но не доказывает бесплатность их записи. Наивная копия такого state каждый шаг должна попасть в profiler, а не исчезнуть из бюджета.

Сначала вывести реальные категории VRAM/RAM после загрузки и сравнить с расчётом. Остаток после weights/state/workspace/draft и небольшого headroom отдаётся expert cache. Не изобретать сложный allocator: одна-две крупные arenas с offsets достаточны.

## 7. Исполнение MoE

### Decode и verify 1–3

Для слоя получить реальные routes и normalized weights каждой позиции. Сгруппировать назначения по expert ID. Кеш hits исполняются на owning GPU, misses первоначально исполняются CPU. CPU получает небольшой input batch/routes, GPU одновременно считает hits/shared expert, затем результаты суммируются в фиксированном порядке.

Один expert, назначенный нескольким позициям, должен читаться tile за tile с обновлением нескольких аккумуляторов. Просто трижды вызвать GEMV из wrapper `verify3` недостаточно. Gate/up можно слить; между gate/up и down всё равно есть нелинейность, поэтому не обещать произвольное слияние всей цепочки в один kernel.

Минимальная форма работы CPU: постоянный bounded worker pool, без создания потоков на слой и без nested OpenMP. Подобрать несколько значений числа физических workers; один host coordination thread не должен быть вытеснен всей машиной AVX2-workload.

Дальше измерить выбор CPU miss против H2D miss+GPU. Модель времени должна включать serial round-trips, queue wait, VRAM availability, reuse across positions и конкуренцию за DDR3. Сначала простое пороговое правило по измерениям, не learned scheduler.

Prefetch будущего слоя не является знанием будущих routes. По умолчанию использовать только уже известные routes; any predictive prefetch является поздним advisory read, не заменой реального выбора экспертов.

### Кеш экспертов

Начать с фиксированных per-layer slots и простой ranked/LRU admission, близкой существующему baseline. Один прямой expert->slot map; нет необходимости в concurrent hash map. Admission из accepted history предпочтительнее засорения отвергнутыми drafts, но cache hits для verification учитываются целиком.

Slot можно заменить только после завершения предыдущего GPU consumer. Новую map entry публиковать после завершения всех необходимых uploads/repack. Достаточно slot state и HIP events. Никакой lock-free CPU↔GPU очереди на fine-grained memory в первой реализации.

### Prefill

Отдельный алгоритм: маршрутизация большого чанка, группировка token assignments по эксперту, staged grouped GEMM на GPU owner, scatter-add. Начальные chunk sizes 1024/2048, затем 4096 и 8192 только при подходящем workspace. Отдельно сохранить быстрый короткий PP для 512 input, не заполняя искусственно большой batch.

Для почти равномерного routing среднее assignments/expert равно B×10/512. Поэтому увеличение B с 1024 до 4096 меняет среднюю полезную работу на эксперт с 20 до 80 позиций. Это математическая иллюстрация амортизации, не измерение распределения модели.

Double buffering экспертных групп позволяет перекрывать передачу следующей группы с текущим GEMM. Не staging всей модели: группе достаточно нескольких экспертов/тайлов. Кеш во время PP можно частично использовать как workspace; восстановление горячих экспертов перед TG обязательно включать в полный request time.

Проверить обе GPU: layer split не гарантирует, что общий loader/scheduler прежнего fork сбалансировал host-expert PP. Новый runtime должен явно отдавать работу owner-у. Pipeline двух PP-чанков через split добавлять после корректного single-in-flight пути, не раньше.

## 8. HIP streams и синхронизация

Начальная конструкция на GPU: один compute stream и один transfer stream, non-blocking; третий только при обнаруженной конфликтующей нагрузке. Все handles и buffers имеют явного owner-а. Один host coordinator достаточен для первой версии.

Цепочка зависимости:

```text
заполнить pinned staging
→ H2D на transfer stream
→ ready event
→ compute stream ждёт ready
→ kernel потребляет slot
→ consumed event
→ следующий reuse/repack ждёт consumed
```

Сначала допустим синхронный fallback для проверки, затем events. Не удалять ожидания, опираясь только на то, что API называется Async. Выполнение streams не обещает автоматически полезного overlap; kernels, DMA и host конкурируют за общие ресурсы. [S15]

Pinned host memory ускоряет transfer, но прямое чтение такой памяти GPU остаётся PCIe-access, не HBM. Write-combined memory плохо подходит canonical weights, которые читает CPU. Для NUMA есть соответствующие allocation flags, однако применять их по обнаруженной topology. [S16]

Сначала иметь маленькие pinned staging buffers. Затем проверить разовую регистрацию нужного host arena/ranges, если это улучшает измерения и устойчиво работает. Не pin/unpin десятки GB на каждой итерации и не предполагать, что full-arena registration обязательно доступна. Fallback с дополнительной host copy честно учитывать в трафике RAM.

P2P включать после реального теста. При отсутствии корректного P2P достаточно transfer через host для небольшого residual на границе split; это не повод строить RCCL infrastructure.

HIP graphs позднее, после корректного hot path. Динамические position, length, active-count и slot maps нельзя заморозить как host scalar arguments в captured graph. Для них достаточно небольшого device step descriptor. Не захватывать CPU-routing/handoff в «один whole-engine graph». Сначала несколько стабильных GPU участков. [S9, S15]

## 9. QSA: точная разреженность вместо полной маски

На каждом QSA-слое держать готовые pooled/normed/rotated keys завершённых блоков. Для decode обновляется максимум один завершившийся блок и хвост. Не собирать заново весь пул из истории.

Зафиксировать порядок: raw-key precision/rounding → pooling → norm → RoPE по позиции начала блока. Для правильного causal prefill ключ блока становится видимым только когда все четыре токена уже видимы конкретному query. Нельзя подготовить ключ из будущих токенов чанка и открыть его ранней позиции. [S3]

Score на decode: dot с четырьмя query heads, ReLU по каждому head, затем sum. Top-k работает по блокам reference-семантики. Буфер до 2051 IDs является capacity, actual count хранится отдельно. Attention читает выбранные K/V, не разворачивает selection в плотную `[context×batch]` маску.

Первый GPU-путь: Q4 gather/dequant только выбранных rows в bounded FP16 scratch + адаптированный специализированный sparse attention (раздел 16), с FP32 reductions/softmax. Затем сравнить с прямым indexed Q4 attention. Не считать второй вариант быстрее заранее: нерегулярное чтение, reuse между query heads и launch overhead могут изменить выбор.

Для PP считать scores tiles-ами по queries/blocks и не держать промежуточные данные на все heads×context×chunk. Одно FP32 представление 128K×4K уже равно 2 GiB. Тайлование требуется до увеличения chunk.

Индексатор всё равно оценивает растущую историю: оптимизация устраняет повторную подготовку и плотные промежуточные представления, но не превращает полный QSA decode в O(1) по истории.

## 10. GDN и HC

Scalar/CPU reference GDN реализовать буквально с проверкой intermediate states; GPU decode/chunked/resident-state пути адаптировать из раздела 16. Выбрать единый state layout для decode, PP, verify и checkpoint/restore. Последовательный host-loop с отдельными launch на каждый токен допустим только как reference, не как производительный PP.

Критично сохранить beta, decay, normalization Q/K, causal convolution и sigmoid output gate. Не наследовать SiLU-gate из похожей Qwen-архитектуры автоматически. Проверять recurrent и chunked outputs на одной последовательности. [S3, S5]

HC не является обычным residual-add. Есть четыре ветви, grouped normalization, low-rank mix, gates и inject. Между GPU передавать весь нужный residual, не только vector 2560. Форму tap для MTP считать отдельным контрактом. [S3, S5]

## 11. MTP2

Сначала target decode корректен без speculation. Затем отдельный draft head из существующего sidecar, sharing target embeddings/output там, где это действительно предусмотрено checkpoint. MTP может содержать собственные attention/MoE weights; это не просто маленький linear layer. Сверить inventory и VRAM.

На MTP передаётся widened residual до финального HC mixer, с корректным token/hidden shift. В fork этому соответствует `t_h_nextn`; порядок входов и собственный head mixer нельзя угадывать. Draft prefill должен построить состояние для всей предыдущей истории, не только последнего токена. [S5]

### Состояние окна

Есть committed position и tentative окно. Для QSA KV откат обычно сводится к length; отдельно вернуть raw tail/pooled-tail metadata. Для GDN нужны восстановимые recurrent/conv states, для PLE convolution/history тоже. Состояние RNG определяется выбранным sampling algorithm, а не требованием совпадения seed-stream с non-speculative baseline.

Первый рабочий вариант: небольшой GPU checkpoint до verify и восстановление/replay принятого префикса при reject. Это медленнее, но просто. Следующий вариант: сохранить state после каждой из максимум 3 проверяемых позиций и выбрать нужный prefix без полного повторного model forward. Не проектировать общий transactional state store.

Проверить accept0/1/2, forced all-reject, all-accept, переход через границу QSA-block, EOS и продолжение после prefix commit. Сравнить дальнейшие logits с отдельным sequential replay только реально committed токенов.

### Sampling

Сначала greedy path и deterministic fixtures. Затем temperature/top-k/top-p как у пользователя. Для точного stochastic speculation нужно учитывать фактические proposal distribution q и target distribution p, acceptance и residual distribution при reject. Просто принять draft при совпадении с одним независимым target sample не является общим эквивалентом standard speculative sampling. [S17]

Не сокращать vocabulary в первом MTP. Позже допустим отдельный эксперимент с корректно заданной q, но не удалением токенов из target distribution. Изменение acceptance на русском/коде измерять отдельно.

### Экономика

Для полного окна 2 draft-позиций E[выходных токенов]=1+p1+p1*p2, где p2 условна на принятие первой. TG определяется полным временем draft+verify+commit/restore, не одним acceptance%. Ускорять verify совместным чтением экспертных tiles и короткими GPU GEMM, а не тремя последовательными decode.

## 12. Загрузка и собственный pack

Начать с actual tensor inventory существующего GGUF: имя, type, shape, strides/packing, bytes. Поддержать только встречающиеся formats; unexpected type должен давать понятную ошибку. Отдельный fallback dequant для редкого маленького non-hot tensor допустим, но не скрытое разворачивание всех routed weights в FP16.

Для safetensors достаточно Python converter с `safe_open`/slices и небольшим числом зависимостей. SafeTensor позволяет отдельную загрузку/срезы тензоров; полный checkpoint не обязан единовременно быть размещён в RAM. [S18]

Converter делает один простой формат: `meta.json`, `weights.bin` и tokenizer assets. PLE может быть отдельным payload для удобства крупной таблицы. Хранить actual offsets uint64, dimensions, dtype/quant, layout version, alignment. Проверять диапазоны чтения, не sha256sum. Не нужна контейнерная файловая система или tensor DB.

Нормализовать HF naming/packing один раз: fused gate_up, Q/gate interleave, indexer QK split, HC shape, MTP e/h projection, RMSNorm stored-weight convention. Для PLE сохранить numeric order 128 частей, head offsets, signed/hash/modulo semantics и EOS-boundaries. Проверить slices на маленьком source tensor перед потоковой обработкой всех shards. [S3]

Не грузить весь BF16 checkpoint в 128 GB. Для исходного масштаба параметров это сотни GB весов. Не держать full BF16 + промежуточный GGUF + новый pack как обязательные стадии. Скачивать/читать по shard, квантовать по tensor/tile и освобождать временное. Если цель состоит только в repack существующего GGUF, вообще не скачивать safetensors.

По пользовательскому fastfetch раздел моделей почти заполнен. В R0 найти реальное свободное место; root volume и model volume могут иметь разные бюджеты. Не удалять исходные нужные веса ради scratch, если есть другая подходящая директория.

## 13. Собственная квантизация

Первая цель: быстрый numerically equivalent repack Q4_0. Вторая: калиброванный 4-bit формат с меньшими накладными расходами и/или лучшим качеством. Третья: 3.x bpw expert format, только если CPU/GPU распаковка и end-to-end выигрыш оправдывают сложность. W4A4/dot8 является поздним исследованием, не основой первой работающей системы.

Калибровка должна включать реальные code/Russian/English/tool workloads и достаточное число маршрутизированных экспертов. Не заявлять качество cold experts по нескольким prompts. Router, normalization, HC, indexer и MTP не опускать в низкую точность вместе со всеми экспертами без отдельной причины.

Сравнить как минимум: current keep1 Q4_0; repack тех же весов; новый expert quant keep1; новый quant с full PLE. Full PLE может улучшить качество, но это другой вариант модели. Любое падение draft acceptance входит в стоимость квантизации.

## 14. Минимальная проверка и метрики

Тестировать границы, которые меняют дискретные решения: PLE hash, QSA selected IDs/causal mask, route IDs, commit/reject. Для numeric kernels измерять max absolute error, relative L2 и downstream logits на одинаковом input. Tolerances выбирать по precision и эталону до принятия оптимизации; универсальная одна константа на весь двигатель не нужна.

Не требовать бит-в-бит одинакового generated text от произвольных GPU/CPU reduction orders. Вместо этого держать teacher-forced inputs и смотреть величину numerical drift, margin у top-k границы и изменение quality. После изменения algorithm semantics явно отделить исправление от FP-noise.

Режимы benchmark: full PP, incremental PP, TG, MTP2, total request. Для kernel tests фиксировать input/routes, для integration tests сохранять реальные routes и acceptance статистику. Warm expert cache достигается одинаковым протоколом прогрева, а не reuse готового prompt state.

Каждый принятый performance result хранит хотя бы вариант кода/модели/KV, actual token counts, context, sampling, PP/TG/request time, cold/warm state и raw log. Дополнительно собирать bytes и component timings только там, где они нужны текущей гипотезе. Никаких Prometheus/Grafana/benchmark DB на старте.

Profiler optional: использовать доступный корректно работающий rocprof/rocprofv3, CPU perf и локальные HIP events. Не блокировать разработку на установке самого нового profiler. Legacy rocprof документирует gfx906; это не гарантия совместимости конкретного нового toolchain. [S19]

## 15. Карта первичных источников

Открытые исходники могут измениться после даты RECON. Во время работы достаточно записать обычные Git revisions используемых checkout, без отдельного hashing pipeline.

S1. Оригинальная конфигурация: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json

S2. Официальная model card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next

S3. Transformers, реализация Qwen Team/Hugging Face: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modular_qwen4_exp.py
Читать `Qwen4ExpTextConfig`, `Qwen4ExpTextQSAIndexer`, `Qwen4ExpTextGatedResidual`, `Qwen4ExpTextNGramEmbedding`, `Qwen4ExpTextPLELayer`; математический reference и соглашения config.

S4. Фактический keep1-вариант: https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF

S5. Production model graph пользователя: https://github.com/0FL01/mx-llama.cpp/blob/dcd685463d597d31f5ca759d32c94592a2740fa4/src/models/qwen4exp.cpp
Читать load_arch_hparams/tensors, build_qsa_top_k, build_attn_qsa, build_norm_gated, `t_h_nextn` и MTP-specific branch.

S6. Native gfx906 пути fork: https://github.com/0FL01/mx-llama.cpp/blob/dcd685463d597d31f5ca759d32c94592a2740fa4/ggml/src/ggml-cuda/common.cuh
Читать ggml_cuda_dp4a, ggml_cuda_mad и DPP reductions. Общая документация оптимизаций: https://github.com/0FL01/mx-llama.cpp/blob/master/FEATURES.md

S7. LLVM codegen test: https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/llvm.amdgcn.sdot8.ll
Ориентироваться на явный gfx906 codegen и реальные compiler tests, не на предположение о generic gfx9.

S8. Strata AMD backend: https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md

S9. Strata QSA contract: https://github.com/Niko1221/Strata/blob/main/include/strata/kernels/qsa.hpp
Также найденные точки входа: include/strata/kernels/qsa_decode_attn.hpp; src/kernels/cuda/qsa_decode_attn.cu; src/kernels/cuda/native_qsa_indexer.cu; src/kernels/cuda/qsa_select.cu. Их полная корректность этим RECON не аттестована.

S10. Strata implementation/bench context: https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md
Найденные GDN точки входа: src/kernels/cuda/gdn.cu, src/kernels/cuda/fused_gdn.cu, include/strata/kernels/native_gdn.hpp. Это карта для чтения, не готовый wave64-порт.

S11. Конкретный пример несовпадения toolchain/library payload: https://github.com/ROCm/TheRock/issues/1844
Исторический issue не использовать как утверждение, что все современные ROCm сборки не поддерживают gfx906.

S12. CMake HIP target: https://cmake.org/cmake/help/latest/variable/CMAKE_HIP_ARCHITECTURES.html

S13. LLVM/Clang AVX2 intrinsics: https://clang.llvm.org/doxygen/avx2intrin_8h.html

S14. Q4_0 block definition: https://github.com/0FL01/mx-llama.cpp/blob/dcd685463d597d31f5ca759d32c94592a2740fa4/ggml/src/ggml-common.h

S15. HIP streams/events/overlap: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html

S16. HIP host memory/NUMA/pinning: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html

S17. Speculative sampling: https://arxiv.org/abs/2211.17192

S18. SafeTensors partial loading: https://huggingface.co/docs/safetensors/index

S19. Legacy profiler support: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/doxygen/html/index.html
Актуальный tracing interface: https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html

Дополнительно для локального изучения production streaming: ggml/src/ggml-cuda/mmq.cu в указанном fork. Там есть copy-group events и диагностические host waits; удалять synchronization без producer-consumer test нельзя.

## 16. Доноры hot gfx906 kernels и layouts

Read-only source RECON 2026-10-01: проверены production FEATURES.md/model/cache/kernel paths, полный furnace README, reinstinct docs/ARCHITECTURE.md и относящиеся kernels/dispatch/packers. Новых performance-прогонов не было. Порядок проверки перед hot kernel/layout: mx → furnace → reinstinct; правило выбора и gates находятся в PLAN.md. Донор алгоритма не становится архитектурным oracle.

### Зафиксированные ревизии и первое чтение

| Источник | Revision | Первое чтение / роль |
| --- | --- | --- |
| Production mx [S5, S6] | `dcd685463d597d31f5ca759d32c94592a2740fa4` | Remote checkout из раздела 4; FEATURES.md, src/models/qwen4exp.cpp, common.cuh, mmq.cu, GDN и expert cache; эксплуатационный baseline |
| [Публичный mx master](https://github.com/0FL01/mx-llama.cpp/tree/43dec4acba459b3116bba8cfcdb049bb2adc6540) | `43dec4acba459b3116bba8cfcdb049bb2adc6540` | Дополнительный донор; отличается от production, не подменяет его image/config |
| [furnace gfx906-perf](https://github.com/sixvolts/llamacpp-gfx906-furnace/tree/905021dbad71c5056ef51f9fd45d545403fc989c) | `905021dbad71c5056ef51f9fd45d545403fc989c` | README целиком перед QSA/GDN/quant matmul, затем src/models/qwen4exp.cpp, repack-gcn.cu/.cuh и kernels ниже |
| [reinstinct main](https://github.com/sixvolts/reinstinct/tree/0b79e326351d90d4554a1c18df92da5d0ab692e8) | `0b79e326351d90d4554a1c18df92da5d0ab692e8` | docs/ARCHITECTURE.md, затем kernels и src/runtime/pipeline.rs; primitive/layout reference, не runtime для портирования |

mx/furnace — MIT; reinstinct объявляет Apache-2.0 в Cargo.toml. При извлечении кода проверить и сохранить относящиеся лицензии, provenance и attribution. Не импортировать Rust runtime, ggml scheduler, server, JIT/hsaco loader или multi-user infrastructure.

### Карта решений для стартового GGUF

**KEEP** — сохранить проверенное решение/контракт; **PORT** — извлечь небольшой готовый primitive; **ADAPT** — адаптировать под наши formats/shapes/state; **WRITE OURS** — собственная часть без подходящего готового решения. Кандидаты выбираются по parity и A/B, не по названию проекта. Для mx/furnace пути kernels ниже относительно `ggml/src/ggml-cuda/`, для reinstinct — `kernels/`, если не указан другой prefix.

| Операция | Решение | Донор / конкретное применение |
| --- | --- | --- |
| Wave64/DPP sum/max, sdot4 | PORT | mx common.cuh; reinstinct gfx906_dpp.h для сравнения. Сохранить VALU→DPP hazard waits и cross-lane semantics |
| Canonical Q4_0/Q4_1 GEMV/MMQ | PORT / ADAPT | mx mmvq.cu, mmq.cu, vecdotq.cuh; scale/offset Q4_1 и согласованный activation contract |
| Planar Q4_0, N=1/2/3 | ADAPT, кандидат | mx q8_repack/repack-q4-0-host.h, q8_repack/repack-common.cuh, q8_repack/repack-kernels.cuh; reinstinct matvec_q4_0_repacked.cpp и matvec_q4_0_repacked_batched.cpp |
| AVX2 experts и route union | WRITE OURS / ADAPT | Canonical quant arithmetic как reference; один tile и несколько CPU/GPU аккумуляторов после группировки expert IDs |
| Grouped MoE PP | ADAPT | mx mmq.cu и staging; reinstinct moe_expert_sort.cpp, mmq_gemm_q4_0_repacked.cpp, mmq_gemm_q4k_grouped.cpp и scatter/combine. Grouped Q4_K не является готовым Q4_0/Q4_1 kernel |
| Q8_0 / реально нужные K-quants | ADAPT | mx q8_repack; furnace repack-gcn.cu; reinstinct matvec_q8_0_repacked.cpp, matvec_q6k_repacked.cpp. Только actual tensor types, без реквантизации |
| GDN decode / multi-token / PP | PORT / ADAPT | mx gated_delta_net.cu и gated_delta_net_chunk.cu; furnace gated_delta_net.cu; reinstinct gdn_recurrent_step_fused.cpp и gdn_recurrent_batched_v2.cpp |
| QSA pooling / scoring | ADAPT | furnace rope.cu, dispatch ggml-cuda.cu; Q4 raw-key roundtrip, новые completed blocks, FP32 pooled keys, N=1/2/3 и query tiles |
| Численный pooled-key cache | WRITE OURS | Single-session append, committed block count/raw tail, causal visibility и rollback; furnace host layout cache не хранит готовые numeric pooled keys |
| Block top-k | ADAPT | furnace top-k.cu; официальный block-select, actual tail/count и проверенный tie contract вместо expanded-position selection |
| Sparse Q4 attention | ADAPT | furnace fattn.cu: selected IDs → bounded Q4 gather/dequant → sparse FP16 kernel; reinstinct attn_partial_q8.cpp / attn_merge.cpp для split-K и softmax merge, не для замены KV-format |
| HC fusion | ADAPT после parity | furnace dsv4-hc.cu и repack-gcn.cu HC-up/pre paths; сохранить 4 ветви, gates и MTP tap |
| Expert cache / PP residency reuse | KEEP / ADAPT | mx src/llama-moecache.cpp, docs/development/moe-cache-prefill.md, staging ggml-cuda.cu; собственные slot maps/events и layout-aware D2D |
| CPU miss vs H2D+GPU miss / overlap | WRITE OURS | Наш AVX2/DDR3 scheduler, admission и route reuse N=2/3; измерять RAM/DMA/compute конкуренцию |
| Multi-GPU staging / PP pipeline | ADAPT | furnace ggml-cuda.cu, reinstinct src/runtime/pipeline.rs; events и bounded buffers, полный HC residual, не 2× single-sequence TG |
| HIP graphs | ADAPT позднее | Стабильные GPU участки, device step descriptor для position/length/maps; не whole-engine capture CPU handoff |

### Ограничения переноса, подтверждённые исходниками

- **Q4 planar уже есть в production mx, но не доказано быстрее.** q8_repack/Q4_0-qualification.md описывает byte-preserving Q4_0 и default-off `GGML_CUDA_REPACK_Q4_0`. В квалификации на том же keep1 GGUF, но tensor split/host-expert конфигурации, TG ухудшился примерно на 0.5–4.1%; это чужой сохранённый результат, не новый наш замер. Q4_1 planar не покрыт. У mx и reinstinct различаются padding rules; на K=2560/640 выбирать по измерению, переносить packer вместе с kernel.
- **Activation ABI не общий.** mx использует Q8_1/специализированные MMQ layouts; reinstinct BlockQ8 содержит FP32 scale/sum и 32 int8. Q4 offset correction и FP association также различаются. Проверять одинаковые квантованные inputs отдельно от float→quant pipeline; не смешивать форматы из разных доноров.
- **furnace repack-gcn не поддерживает Q4_0/Q4_1.** Есть Q3_K/Q4_K/Q5_K/Q6_K, опциональные Q8_0/Q5_1. Нельзя переводить routed weights в другой quant ради этого kernel. Small-MoE dispatch по отдельным assignments не заменяет загрузку одного weight tile для нескольких совпавших routes.
- **QSA cache и fusion требуют доработки.** LLAMA_QSA_NO_MS_CACHE управляет host block-layout cache; pooling всё ещё читает историю. Fused pool принимает F16/F32 raw keys, не Q4. Sparse attention принимает F16 K/V и получает IDs через маску. Адаптировать под наш numeric cache и прямые IDs, не создавать полную FP16 историю/маску. Expanded-position top-k и atomic tie order не считать официальной семантикой (раздел 3).
- **GDN state layouts различаются.** У furnace resident-state путь Wave64/LDS, у reinstinct batched_v2 — swizzle/prefetch; старый step_fused_batched читает/пишет HBM state на каждом токене. У mx уже есть chunked PP. На H48/D128 сравнить эти варианты, не смешивать транспонированные state layouts или snapshot conventions; проверить state после каждого accepted prefix.
- **PP reuse переносить без layout/OOB ошибок.** mx квалифицировал D2D reuse resident experts для PP; копирование canonical tail padding не должно читать следующий cache slot. Repacked slot нельзя слепо копировать в canonical MMQ tensor. Учитывать H2D/D2D/repack и восстановление cache в полном request time.
- **Чужая topology не наша.** furnace измерял 4×32 GB, другие quant/config и multi-user workload; его P2P fallback вызван проблемой того HIP-стека. Наш P2P проверяется в R0. reinstinct GPU-resident runtime/loader может реквантовать исходные weights, а pipeline удерживает все PP activations; взять ordering, но использовать bounded staging и unchanged weights.
- **Q8 KV не становится основным режимом.** reinstinct хранит int8 с FP32 scale на token/head, не ggml Q8_0 block32/FP16 scale, и его dense GQA paths не являются QSA. При 128K target+MTP Q4_0 K/V — 936 MiB; обычный Q8_0 был бы 1768 MiB, custom per-head Q8 reinstinct — 1690 MiB, без индекса/workspace. Это расчёт; Q8 допускается только отдельным экспериментом. reinstinct также не является Qwen3.8-Flash-Next logits oracle.

### Навигация по feature switches furnace

- `GGML_CUDA_NO_QSA_POOL`, `GGML_CUDA_NO_QSA_SCORE`: ggml-cuda.cu → rope.cu; block selection — отдельно top-k.cu.
- `GGML_CUDA_NO_FA_GATHER`, `GGML_CUDA_NO_FA_DEC`: fattn.cu.
- `GGML_CUDA_NO_Q8_MULTI`, `GGML_CUDA_NO_MOE_SMALL`, `GGML_CUDA_NO_Q8_NC`, `GGML_CUDA_NO_HC_UP_PRE`, `GGML_CUDA_NO_Q8_EMIT`: repack-gcn.cu.
- `GGML_CUDA_NO_F32_NC`: ggml-cuda.cu; `LLAMA_QSA_NO_MS_CACHE`: src/llama-memory-hybrid-idx.cpp.

Наличие switch или kernel не доказывает его активацию в preset и выигрыш на наших shapes. Новую работу концентрировать на GPU expert residency, CPU/H2D miss scheduling, overlap, shared route loading N=2/3 и grouped PP; новый expert quant — после same-Q4 runtime.
