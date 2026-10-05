# План реализации и критерии готовности

Этот файл задаёт очередь работы, а не календарное обещание. Переходы определяются работающим кодом и измерениями. Архитектурные основания и ссылки находятся в RECON.md. Начать с R0; после возобновления читать текущую задачу из STATE.md.

## Что считается результатом

**User reaffirmed2026-10-04:** finish the ENTIRE development plan, personally without subagents; document and commit/push every completed slice. Goal is to beat llama.cpp on comparable end-to-end measurements, not just own microbenchmarks. No correctness-gate waiver or guaranteed speed claim.

Самостоятельный C++20/HIP runtime действительно выполняет модель на выделенной машине, а не вызывает llama_decode внутри новой оболочки. Он поддерживает текст, обе gfx906, RAM-experts с GPU-кешем, Q4_0 для обоих K/V-кешей target и MTP, полноценный prefill, корректный MTP2, append-history reuse и проверенный занятый контекст около 128K. Конвертер из оригинальных safetensors является отдельным обязательным результатом. Новый низкобитный квант является исследованием после рабочего pack, а не условием для первого запуска.

Сначала функциональный результат на существующих весах. Затем выигрыш полного запроса. API-обвязка не должна задерживать inference core.

## Правило перед hot kernel и выбором GPU layout

Для текущей операции проверить production mx, затем furnace и reinstinct по pinned revisions и карте RECON.md, раздел 16. При первом использовании ревизии прочитать mx FEATURES.md, furnace README полностью и reinstinct docs/ARCHITECTURE.md; затем читать относящиеся к операции kernels, dispatch и packer. Не повторять полный RECON для каждого kernel и не создавать новую подготовительную веху: R0 остаётся ближайшей задачей.

В соответствующем участке кода/RECON кратко зафиксировать:

1. Операцию, фактические shapes/dtypes и режимы N=1/2/3/prefill.
2. Донорские file/function/revision и решение **KEEP / PORT / ADAPT / WRITE OURS**: сохранить проверенное решение / извлечь небольшой primitive / адаптировать / написать своё при отсутствии подходящего решения.
3. Согласованный контракт packer/kernel: activation quantization и offset correction, scales, plane order, strides/padding, state layout и lifetime buffers. Одинаковое имя Q8 не гарантирует одинаковый ABI или FP-арифметику.
4. Малый parity fixture и ближайший reference для парного A/B; учесть repack/transfer/workspace, а не только kernel time.

Scalar/CPU oracle разрешён; намеренно generic/медленный GPU production path, который сразу заменяется известным gfx906 kernel, не нужен. Переносить primitives, а не runtime/server/graph infrastructure; сохранять лицензии и attribution. Микробенч допускает кандидат в сквозной runtime, но окончательный выбор основного пути требует проверки полного запроса. Если подходящих доноров нет, писать минимальный собственный kernel и сравнивать с ближайшим reference.

## R0. Подключение, рабочая сборка, аппаратный baseline

**Действия.** Связь, модели, baseline image/source/config и аппаратная topology обнаружены 2026-10-01 (RECON.md, раздел 4; пути в STATE.md). Через MCP `mi50-llama-remote` проверить версии и работоспособность существующего контейнерного gfx906 toolchain, выбрать рабочие source/build/runs нового core на корневом разделе. Не обновлять систему ради свежего номера версии. Настроить один цикл source sync → remote build → remote run → получить результат. Остановить конкурирующий llama-server перед нагрузкой своего процесса; baseline запускать отдельно.

Написать маленький device probe и реальные HIP smoke kernels. Проверить обе GPU, VRAM, wave size, CPU flags/физические cores, NUMA и PCI topology. Выполнить ограниченную серию RAM/H2D/P2P/launch/GEMM проверок из RECON, включая CPU compute одновременно с H2D. Недоступный performance counter не блокирует разработку: использовать wall time, HIP events и доступные системные данные.

Воспроизвести короткий baseline и 4K+512 на нынешнем форке, сохранить точный вход. Если исходные пользовательские prompts отсутствуют, создать и сохранить новые fixtures, явно пометить, что это новая серия. Не обещать воспроизведение 23.7TG на другом тексте.

**Готово:** исходник с контроллера собран и выполнен на обеих GPU; корректность kernel/copy проверена; есть реальные характеристики машины и первый лог baseline. STATE.md содержит рабочие команды. Не создавать отдельный framework аппаратного RECON.

**Закрыт 2026-10-01:** core-probe, удалённый Release build/CTest, обе HIP/rocBLAS fixtures и новая baseline-серия 32+64 / 4096+512; команды и ограничения в README.md/STATE.md, числа в results.jsonl. Это не завершение inference core и не speedup относительно исходных пользовательских prompts.

## R1. Загрузка весов и один исполнимый эксперт

**Действия.** Реализовать loader зафиксированного GGUF по проверенному inventory в RECON.md: имена, размеры, реальные types/strides и metadata. Target содержит F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q8_0/Q6_K; sidecar — F32/BF16/Q8_0. Поддержать необходимые типы, не весь каталог GGUF. Можно использовать существующий parser. Нормализовать views без полной перекопировки модели. Идентифицировать PLE, HC, GDN/QSA, shared expert, LM head и sidecar.

Извлечь один реальный routed expert слоя 0: gate/up Q4_0, down Q4_1 (Q4_1 down также у слоёв 1–5; у остальных down Q4_0). Сделать scalar/reference, AVX2 и HIP Q4×Q8 реализацию его gate/up → SiLU×up → down с сохранением scale/offset Q4_1. Проверять одинаковые квантованные входы и веса, чтобы изолировать ошибки kernel от ошибок activation quantization. Затем проверить float input → quantization → matmul.

HIP начать с адаптации canonical DP4A и готовых Wave64/DPP primitives mx, не с generic GPU matmul. Planar Q4_0 из mx q8_repack и reinstinct matvec_q4_0_repacked* сравнить как кандидаты на формах 2560↔640 и N=1/2/3; Q4_1 down обязателен независимо от наличия planar-пути. Packer и activation arithmetic переносить согласованно, padding выбирать по измерению наших shapes. Сохранить один canonical host inventory; repack делать для GPU slot/tile, без второй полной копии экспертов в RAM.

Измерить 1/2/3 позиции и один PP-sized batch. На CPU применять постоянный worker pool только там, где он нужен; не создавать потоки на каждый matmul. Проверить диапазоны unpack, scale, signedness, суммы и отсутствие saturation в выбранной AVX2-схеме.

**Готово:** реальный expert выдаёт проверенный результат CPU/GPU; зафиксированы donor/layout/activation contract, время и трафик для проверенных форм. Для correctness полный FP32 checkpoint в RAM не нужен: достаточно выбранных tensors/fixtures.

**Закрыт 2026-10-01:** Model проверил оба GGUF; core-expert исполнил исходный expert0/layer0 scalar/AVX2/HIP для N=1/2/3/128 на обеих GPU. Q8_1 ABI и Q4_1 half-product semantics зафиксированы в quant.hpp/README.md; mx DPP/SDOT4 и адаптированное multi-column reuse с лицензиями в third_party/. CPU inlined/F16C принят после A/B/A и точной parity; planar не дал универсального выигрыша, основной layout canonical. Числа, scope и raw logs в results.jsonl. Это ещё не сквозной inference, DDR-miss scheduler или полноценный grouped PP.

## R2. Минимальные блоки архитектуры

**Действия.** Реализовать HC mixing/injection, GDN state update и convolution, QSA indexer/attention, PLE lookup/hash/convolution и нормализации. Использовать маленькие fixtures и выборочные слои с реальными весами. Пока без fusion ради fusion.

GDN GPU-пути адаптировать из mx decode/chunked и furnace resident-state Wave64/LDS; reinstinct batched_v2 использовать как дополнительный primitive reference. До переноса выбрать единый state layout для decode/prefill/verify/checkpoint и проверить QK16→V48 mapping, beta/decay и sigmoid output gate. HC сначала проверить по отдельным операциям, затем рассматривать готовые furnace fusion-кандидаты.

Реализовать reference и HIP запись/чтение Q4_0 K/V: блок 32 значения, 16 байт кодов + FP16 scale, фактически 4.5 bpw. Сохранить signed scale, округление и nibble packing baseline; коды вычислять с исходным FP32 scale, не с округлённым сохранённым scale. Проверить нулевые блоки и сравнить CPU/HIP packing, dequantization и attention на одинаковых квантованных данных и выбранных IDs. Перенести нормализованный Hadamard: Q/K блоками 256 после RoPE, V блоками 64 перед записью, обратное преобразование выхода по V до gate/output projection.

Индексатор: raw keys сначала Q4_0 quantize/dequantize как в baseline, затем pooling → norm → RoPE. Хранить FP32 pooled keys завершённых блоков и небольшой raw-хвост; не сохранять полную raw-историю или неиспользуемый index V. Проверить последовательное обновление против обработки чанком. Семантику block-select/хвоста проверять отдельно от формата KV, чтобы её исправление не выдавать за чистую оптимизацию.

Адаптировать furnace fused pooling/scoring и radix top-k под Q4 raw-key roundtrip, завершённые блоки и actual count. Численный append-cache pooled keys и его rollback реализовать самим: furnace кеширует host layout, не готовые GPU pooled keys. Не переносить expanded-position top-k или atomic tie order как математическую спецификацию; выбор IDs и границы scores сверять с reference.

Обязательная QSA-проверка: 2052 видимые позиции, 513 законченных блоков, хвост 0. Проверить официальный block-select и путь текущего fork отдельно. Дополнить длинами 2047–2056, равными/нулевыми scores, каждым остатком по 4, разными границами prefill-чанка. Считать фактический selection count отдельно от вместимости буфера 2051. Не объявлять fork ошибочным или эквивалентным по одному комментарию.

Проверить PLE EOS/history и однобазовый layer id; keep1 имеет сохранённую логическую структуру с нулевыми головами. GDN output gate здесь sigmoid. В QSA проверить соответствие query-head → KV-head, RoPE и causal visibility.

**Готово:** каждый блок проходит малые тесты, включая последовательное исполнение против обработки чанком. Обнаруженная численная/алгоритмическая разница с fork описана конкретно. Не требовать побитового совпадения всей генерации, если меняется FP-порядок.

**Срез R2a закрыт 2026-10-01, R2 ещё не закрыт:** Q4_0 KV packing/dequant и Hadamard CPU/HIP проходят byte parity на обеих GPU; cooperative pack проверен против mx serial A/B/A. CPU block-select/actual-tail oracle проверил 2047–2056 и causal chunk prefixes. Source-derived эмуляция pinned HIP fork показывает при 2052 valid count2051 против reference2048; это семантическое исправление, не чистая оптимизация и не GPU/logits замер baseline. Команды в README.md, сырые logs/числа в results.jsonl. Следующие срезы: GDN/conv единый state, HC/PLE actual weights, numeric QSA append/rollback и sparse attention.

**Срез R2b закрыт 2026-10-01:** CPU dense oracle поддерживает все actual GGUF types; GDN/conv scalar и адаптированные gfx906 CPW2/resident chunks имеют единый FP32 state/raw-history ABI и GGUF mapping h%16. Loaded layer0 projections через CPU oracle → GPU recurrence/norm/sigmoid проходят N1/2/3/128, chronological prefix/accept0/1/2, repeated reject, split-chunk и error+reuse fixtures на обеих GPU. Dispatch A/B/A поддержал resident chunks; remote CTest10/10 и CPU ASan/UBSan прошли. Это fixture до out_proj, не GPU projections или полный inference. Scope/числа — results.jsonl, команды — README.md. Следующие срезы HC/PLE и numeric QSA остаются открыты.

**Срез R2c закрыт 2026-10-02, R2 ещё не закрыт:** HC/PLE CPU oracles и loaded-weight fixtures проверяют attention/FFN/root HC N1/2/3, widened MTP Tap, logical keep1 lookup/hash/EOS, dilation9 history, chunk и accept0/1/2 restore. Actual multiplier bound доказывает signed/unsigned hash equivalence для всей declared vocabulary выбранного GGUF; синтетический negative hash не считается багом baseline. Это raw-FP32 CPU semantic qualification, не GPU projections или inference speed. Числа/контракт — results.jsonl/README.md. Следующий срез: numeric QSA pooled append/rollback и GPU selection/attention; GPU dense/HC/PLE integration остаётся перед R3 forward.

**Срез R2d закрыт 2026-10-02:** CPU/HIP indexer хранит только FP32 completed keys и Q4-roundtrip хвост, append публикует новые блоки без копирования истории. Actual indexer weights/metadata, все tail phases/N1/2/3/128, chronological prefixes/repeated reject/restore, visibility до131072 и future poison проверены на обеих GPU. Furnace radix адаптирован к whole-block selection с exact IDs на одинаковых scores; bounded Q4→FP16 gather/attention Q24/KV2/D256 проходит common-gather gate и selected/unselected/error fixtures. Remote CTest16/16, CPU ASan/UBSan прошли. Это component fixture с CPU synthetic projections, не GPU projections, полный QSA block или inference speed; GPU dense/HC/PLE integration ещё нужна перед R3. Attention ~1.1 ms на2048 keys — измеренный bottleneck для последующей диагностики, не доказательство ускорения. Числа/scope/raw provenance — results.jsonl/README.md.

**Срез R2e закрыт 2026-10-02:** reusable canonical Q4/Q5/Q8/Q6 GPU linear проходит независимый common-Q8 oracle на обеих GPU для N1/2/3, extrema/odd rows/width16384 и выбранных unchanged actual rows. HC/PLE GPU RMS/pointwise/history launchers проверены N1/2/3/128 на одинаковых synthetic projections с actual gamma/F16 conv, chronological prefixes/restore, rejected publication и error+reuse. Gates не менялись; remote CTest18/18, CPU common-Q8 ASan/UBSan прошли. Это не полный projected HC/PLE block или speedup. Числа/scope — results.jsonl/README.md. Следующий работающий срез R3: F32/BF16 GPU projection и Session, embeddings/LM head, actual all-layer logits; текущий linear output limit расширять только для проверенного LM head.

## R3. Первый сквозной самостоятельный runtime

**Срез R3a закрыт 2026-10-02, R3 ещё не закрыт:** borrowed rocBLAS projection проверен на unchanged F32 и exact BF16→F32 values, N1/2/3/128, odd/non-square/max dimensions и beta-zero canaries на обеих GPU. Canonical Q6_K linear расширен только для actual LM head [2560,248320]; все output values finite, full prefix bytes exact, 24 выбранные строки сверены с common-Q8 oracle при N1/2/3. Remote CTest19/19 и recorder tests37 прошли, gates unchanged. Это не all-layer logits/full inference или A/B speedup; resident times/provenance — results.jsonl, команды — README.md. Далее прямой Session/48 layers и baseline teacher-forced/intermediate oracle.

**Срез R3b закрыт 2026-10-02:** собственный Session/CLI исполняет все48 слоёв, static24/24 split с pinned40KiB widened-residual handoff, canonical RAM experts/112 GPU slots на слой и ordered Q4_0 K/V decode. Teacher32 IDs `[248044,100..130]`: все7 946 240 logits finite и numerical error0, required intermediates проходят исходные gates, argmax32/32. Reference — независимый source/image-attested production oracle с opt-in HF additive-GDN-L2/FP32 gathered-QSA corrections, не bitwise HF или unchanged-performance baseline (RECON.md §10). Compiled provenance `e9f1dfe57cf8fdc0abd9ba10ab91cfbe01d9e0db` dirty1; artifacts/repro — README.md.

Final job `1790961285745-474` exit0: strict full build/CTest21/21, reset8steps/capacity4/slots1 и greedy32 прошли; recorder добавил одну validated R3b запись в canonical journal. Числа полного короткого запроса — results.jsonl, не steady TG/PP/A/B claim. Canonical attention accuracy A/B/A примерно2× медленнее старой topology, не acceleration claim. Это N=1 ordered decode, не grouped PP/CPU miss overlap/MTP/sampling/HTTP/long-context qualification. Оставшийся allocation ledger закрывается R3c ниже; цели R4–R8 сохраняются.

**R3c и R3 закрыты 2026-10-02:** capacity131072/slots112 проверены одной Session на обеих GPU. Полный category traversal совпадает с независимыми live-Buffer counters; static24/24 owners, расчёт Q4 KV/pooled index/GDN/PLE/cache, steady step/reset/replay и recovery всех owned bytes после destruction проходят. Все144 expert payload loads остаются только в constructor; step использует RAM, counters после load неизменны (не physical-SSD syscall trace). Full build job488 exit0/CTest22/22, actual memory fixture/collector31 tests проходят; одна R3c запись в canonical journal, прежние11 неизменны. Полные allocations/headroom/provenance — results.jsonl, воспроизведение — README.md. Это capacity proof с двумя consumed tokens на pass, не occupied128K/MTP/performance qualification; далее R4/R5.

**Действия.** Соединить 48 слоёв в прямую последовательность вызовов. Один Session с заранее выделенными buffers; статический layer split. CPU misses и GPU hits первоначально могут синхронизироваться консервативно. Кеш и очередь копий должны сначала быть корректными.

Сохранить owner/residency и producer-consumer контракты mx expert cache в собственных slots/maps/events, без импорта ggml scheduler. Публиковать slot только после всех uploads/repack, заменять только после readers. Между GPU передавать нужный residual 4×2560; P2P или ограниченный pinned staging выбирать по проверке R0, не по настройке чужого стека.

Добавить embeddings, LM projection, greedy sampling и текстовую CLI через готовый tokenizer либо сначала CLI с token IDs. Выполнить короткий prompt и 32–128 output tokens с Q4_0 K/V. Сравнить выбранные intermediate activations и teacher-forced logits. FP32/FP16-KV допустим только для маленьких correctness fixtures, не как основной режим или замена Q4_0 в performance A/B.

Сверить выделенную память с геометрией: при capacity 131072 основной Q4_0 K/V — 864 MiB; будущий MTP K/V — ещё 72 MiB, FP32 pooled index основных и MTP слоёв — 208 MiB. Всего кеши и индекс — 1144 MiB на обе GPU без padding/workspace; это расчёт, не замер. Weights, expert cache, GDN/conv states, speculative checkpoints и workspace учитывать отдельно по каждой GPU с headroom. Память зависит от выделенной capacity, а не только занятой истории.

Проверить, что обе половины сети действительно исполняются на назначенных GPU и что процесс не читает expert weights с SSD во время прогретой генерации.

**Готово:** собственный бинарник самостоятельно обрабатывает запрос с Q4_0 K/V; все слои/PLE работают; нет NaN, утечек, бесконтрольного роста памяти и скрытого llama_decode. Выделенная память по категориям и GPU сверена с расчётом. Скорость на этом этапе может быть ниже baseline. Не бросать core ради преждевременной оптимизации одного kernel.

## R4. Полноценный prefill

**R4a prerequisites закрыты (2026-10-02).** N2/3 GDN CPW2 держит state в регистрах и повторяет exact N1 arithmetic; outputs/state/history/все prefixes и continuation совпали побитово на обеих GPU. Stable CPU RouteGroups сохраняет token/rank/raw weight bits без hot allocations. CTest23/23, 491148 route checks и A/B/A GDN прошли. N2/3 component latency ниже обоих A, N1 win не подтверждён.

**R4a grouped short-window Session закрыт, не весь R4.** `step_batch` N2/3 исполняет multi-column projections и один triplet upload на expert group, затем scatter и original-rank contribution fold. Пять schedules/200 rows дают 39 731 200 bitwise logit comparisons с N1; causal QSA, PLE/GDN chronology, 28 invalid windows, reset/replay и steady owners/read counters проходят. Slot1 matching-suffix reuse подтверждён без предположения одинакового initial cache. Full CTest24/24, default-N1 teacher32 error0 и прежний capacity131072 allocation ledger проходят. Это short-window self-parity/reuse, не квалификация большого PP или end-to-end ускорения; следующий срез — bounded canonical DS4/MMQ и causal chunk PP.

**R4b Q4 MMQ primitives закрыты, не весь R4.** Canonical Q4_0/Q4_1, byte-preserving Q8→DS4 и masked K/row/column tails прошли обе GPU: 979 matrix cases на устройство, unchanged common-Q8 gates, 1800 validated completed-event intervals, full CTest25/25 и recorder32 cases. A/B/A против diagnostic sliced-linear показал выигрыш down при N≥8 и gate при N128, но первоначальный HC проигрывает на всех проверенных N. Это корректный component baseline, не production PP или универсальное ускорение; последующий microtile-срез ниже, затем wide-format projections и bounded grouped staging.

**R4b small-M microtiles закрыты.** Literal J8 для M≤640 меняет только dispatch. Paired MMQ-baseline/candidate/baseline на обеих GPU подтвердил выигрыш всех восьми изменённых actual gate/HC N32/128 cases против обоих A; все три полных fixtures сохраняют frozen gates/byte/tail proofs. Full CTest25/25 и recorder75 cases проходят; protocol1 baseline и protocol2 candidate валидируются отдельно. Неизменённый down N128 показал ~6–8% variation, поэтому нет universal/end-to-end/PP speedup claim. Переход к specialized Q5/Q8/Q6 и bounded causal PP; full-request gate остаётся обязательным.

R4b wide primitives закрыты: canonical Q5_0/Q8_0/Q6_K используют специализированные LDS loaders и прежний byte-preserving DS4; Q4 forwarding не дублирует kernel. Обе GPU проходят common-Q8 gates, 664 correctness cases/device и 1800 validated completed intervals; весь head finite, CPU numeric coverage — 24 rows, не весь vocabulary. Strict CTest26/26 и recorder33 проходят. Paired N32/128 component results выигрывают на всех трёх actual bindings; N1/3 MMQ медленнее и не заменяет MMVQ. I128 occupancy hint исправлен по фактическому LDS; standalone codegen отмечает spills у широких Q5/Q8 tiles. Это component qualification; bounded Session short checkpoint ниже не закрывает большой PP/full-request gates.

**Qualified short-PP slice закрыт, не весь R4.** Исторический compiled `15a025d9e105b704a628d0bbca003a08cb61b334` dirty1: одна Session capacity40/slots1/max32 прошла N1/chunk4/chunk32/occupied-prefix full-logit self-parity, atomic rejects и steady ledger. Broad job `1791011326789-719` exit0/strict Release/CTest27/27 и sequential regressions подтвердили срез; одна validated `r4b_short_prefill` запись17→18 сохранила прежнюю историю. Подробные численные результаты, raw labels, ограничения и воспроизведение — README.md/results.jsonl.

Accepted scheduling: nonexpert `Matrix::apply` и shared dense — chronological tiles≤8, dense MMVF N1..8 с сохранённым fallback N9..128; separate quantized-short API≤8 сохраняет36-byte Q8_1/FP-order и прежний API≤3. Expert microtiles≤8, grouping once/full logical chunk и один triplet/group; band16/two stages и `copy_ready`/`consumer_done` lifetime неизменны. PLE сохраняет512-thread gate reduction на≤8, hash/norm/conv; GDN теперь exact chronological CPW N1..128 с усиленными state/prefix/late-error fixtures. Новых arithmetic/weights/precision/epsilon/gate changes нет. Trace32 job700: all7392 matched nodes/640FFN probes/full head exact; отдельный defaultN1 job708: original6912 probes/fullteacher32 bit-exact. Short fixture не является independent HF reference или speed evidence.

**Qualified logical-wide1024 slice закрыт, не весь R4.** Job `1791014580540-724` exit0/elapsed22m16s, compiled `88bd3e6b24ab1dc07c556f5093533d10a91ecd80` dirtytrue: strict HIP/CXX Release build/CTest28/28 (595.52s), затем sequential `core-session-test`, `core-session-batch-test`, `core-prefill-wide-test`, `core-memory`, каждый exit0. Build по-прежнему запускается explicit `sh`: source `tools/build.sh`100644. Новая ревизия commit не переименовывает provenance этих artifacts.

Отдельные `SessionRouteStats`/`route_stats()` публикуются только после успешного call: last-call max assignments к ОДНОМУ expert в одном layer, максимум по 48 слоям; cumulative call/layer/expert groups>128 с момента reset. Constructor/successful reset обнуляют, invalid arguments и execution failure сохраняют последние успешные diagnostics; старые `SessionStats` и serialized protocols неизменны. Logical groups не являются physical 128-column kernel/repack proof.

Actual `runs/r4-prefill-wide-a.jsonl` protocol1/3 431 452 bytes: одна Session cap1056/slots1/max1024, retained 1056 N1 reference, teacher1024 schedules 8×128/7×129+121/single1024 и 32 N1 continuations в каждой фазе. Все 4224 rows/1169 windows/1181 records: 1 048 903 680 finite logits/786 677 760 comparisons, zero violations/bit mismatches/maxabs при frozen `.02+.002|ref|`. 15 atomic rejects, 1205 memory observations (1175 serialized+30 summarized), fixture guards/full-span/stats/route-stats preservation и steady owned ledger проходят. Observed max-group:1017; число groups>128:950 по всем фазам, single1024 phase:874; это actual logical routing evidence. Min freeGPU0/1:12 709 560 320/12 153 815 040 bytes. Canonical microtiles≤8/GDN slices≤128 и model/precision/epsilon/gates сохраняют математику. Это same-Session N1 self-parity, не independent HF/throughput/cold-cache equality/4K16K/fullR4; aggregate memory floor не доказывает individual-buffer capacities или wide-fixture owned release.

`record_prefill_wide.collect(actual_raw)` и parent local18/18 (47.192s) прошли; actual `record_memory.collect(r4-prefill-wide-a-memory.jsonl)` отдельно прошёл для defaultcap131072/slots112, без duplicate append. Logs `runs/r4-prefill-wide-a-build.log`, `-reset.jsonl`, `-batch.jsonl`, `-memory.jsonl`. На историческом wide closure canonical ROOT `/home/radneon/gfx906-core/results.jsonl` достиг19 records: одна new `r4b_wide_prefill`,18→19, прежний byte prefix/parsed history неизменны, source88bd/dirtytrue сохранён; downloaded wide journal +1/-0 и controller `collect(downloaded_actual_raw)` проверены. Полные команды — README.md.

**Real4K/16K long-correctness и sampling prerequisites qualified; полный R4 открыт.** Exclusive job `1791030847464-742` completed exit0/elapsed1h15m35s: полный strict CXX20/HIP20 Release gfx906 build с warning gates, CTest32/32 (605.76s), actual Q8-underflow capture на обеих GPU, затем real16K→real4K full-logit fixtures/remote collectors PASS и sequential reset/batch/default-memory exit0/passed footers. Actual `record_memory.collect` прошёл для capacity131072/slots112. Compiled provenance строго `2e9848d43cf9f908dc8280f81e111c0cde86f01c`/dirtytrue, не будущий commit; после этого исторического job GPU были idle.

Каждая long fixture использует одну Session capROWS+32/slots112/max1024, retained N1 full-vocabulary reference, `canonical1024` и occupied5N1→chunks997, с32 N1 continuations в каждой из трёх фаз. Real16K:129 records/49248 timeline rows/16518 windows; real4K:93 records/12384 rows/4206 windows. Все full-vocabulary values finite; zero violations/maxabs/maxboundratio/diagnostic bit mismatches при frozen `.02+.002*abs(ref)`. Bit equality diagnostic-only. Teacher BOS248044→monotone IDs — новая teacher-family, не original text/user/baseline parity. Observed2047–2056/mod4 — actual logical causal visibility, не GPU selected-ID trace. Q4_0 обоих target K/V unchanged. Это same-Session N1 self-parity, не independent HF/physical128-tile/individual-buffer/full-RAM/owned-release/peak-VRAM/performance/fullR4 proof; wall sums correctness-only. Полные числа и ограничения — README.md, raws `runs/r4-prefill-long-b-{16384,4096}.jsonl`.

**Q8_1 accuracy/validation-correctness repair отдельно от optimization.** Job729 N1 offset13206 был ошибочно отвергнутым valid tiny Q8 block; job733 показал finite ATTENTION OUTPUT layer8/block117, original FP32 d>0→stored half0. Pinned mx `dcd685…` сохраняет original codes/raw sum. CPU/оба GPU теперь сохраняют тот же контракт; nonzero-input blocks с original FP32 d0, unsafe rounded±128, nonfinite/FP16 InfNaN остаются reject до unsafe int8 cast, logical32 halves wave64 изолированы. Без weight/precision/epsilon/gate changes, NaN clamp или all-zero replacement. Target quant791975 checks/214 rejects и actual192-block/6144-F32 capture на обеих GPU прошли jobs740/742. Diagnostic trace range не меняет default tracing.

Long fixture и primary-sampling CLI/sampler integrated в strict build; prior job729 actual32-output smoke seed42/chunk1-vs32 дал одинаковые IDs/RNG32, legacy greedy32 passed. Primary preset temperature1.0/top-p0.95/top-k20 сохранён. Этот исторический срез закрывает long correctness/sampling prerequisites; последующая actual512 серия и observer joins имеют отдельное подтверждение ниже. Future CPU expert/shadow и speculative helpers не runtime-qualified этим срезом. Existing MIT mx/furnace и Apache-2.0 reinstinct attribution сохраняется, новых dependencies/donor whole runtime нет.

Remote prevalidation обоих actual long raws и parent LOCAL `collect(downloaded_actual16K/4K)` PASS. Parent `record_prefill_long.main` appended EXACTLY два records в canonical ROOT19→21; old byte prefix и parsed19-record history exact preserved. Последние два `r4b_long_prefill`:teacher rows16384 затем4096, оба passed=true/R4_complete_claim=false, source2e9848d/dirtytrue unchanged. Downloaded canonical journal локально1 572 766B; gitdiff+2/-0 verified. Logs `runs/r4-prefill-repaired-b-build.log`, `r4-q8-underflow-actual-b.jsonl`, `r4-prefill-repaired-{reset,batch,memory}-b.jsonl`; команды `core-prefill-long-test MODEL 4096|16384` и `record_prefill_long.py --raw ... --results ROOT/results.jsonl` только после exit0 — README.md. Explicit `sh` build100644 сохраняется.

**Actual512 same-own-config measurement-only slice принят/закрыт; полный R4 открыт.** Performance job `1791040898917-746` completed exit0/1h40m26s на FROZEN `b522429c5933f493a5838317dc0bfc26ee4158aa`/dirtytrue, не future commit. Strict targeted `core-session` build PASS; новые request-results/vram-observer CTest2/2 (5.18s), suites49/23 PASS. Исторический full32/32 job742 остаётся закрытым; targeted job746 не подтверждал subsequent full suite. Отдельный latest job765 прошёл full35/35 со своим source/scope ниже. Все8 fresh Session requests protocol1/slots112/MTPoff/no prefix reuse/ignoreEOS/primary temperature1/top-p.95/top-k20/seed12345 выдали512 actual outputs: TGforwards511/RNG512, consumed4607/16895. Exact4K input — R0 generated archive/code IDs `ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json`; 16K — четыре конкатенации тех же IDs, не retokenized/original-user/baseline16K prompt. Все четыре512-ID arrays внутри каждой длины exact equal, diagnostic-only, не universal bit-generation contract.

Performance порядок на каждой длине: unobserved trace-free A1chunk128→B1024→A2chunk128, затем отдельный B_VRAM. B4K PP24.8487818621/TG10.2822355241, total214535.87483ms; B16K PP22.2115183863/TG9.9343829824, total789074.279015ms. Mean-A/B PP speedup1.82406225345×/1.75816669497×, full-request1.63977185394×/1.71047221484×. Только сравнение конфигураций собственного non-MTP core; externalMTP2baseline speedup не заявляется. Цели400–600PP/30–40TG НЕ достигнуты; full-project/fullR4performance OPEN. Full-request expert uploads4K: eachA1139677491200B/B305853440000B;16K eachA4264348979200B/B890465689600B. Per-phase bytes не emitted: нет PP-only traffic claim или доказательства residual bottleneck. Полные A/B/A rates/timings — README.md/raws.

Отдельный observer0.1s: scope `sampled_global_driver_VRAM_not_exact_instantaneous_peak`, не HIP-owned/exact/per-phase peak. Parent independently verified actual R0 HIP0BDF0000:05:00.0/HIP1BDF0000:08:00.0; collector сам mapping не attest. Driver totals17163091968B каждый;4K samples2892/maxused12004397056/12487217152B/minfree5158694912/4675874816B;16K samples8643/maxused12079910912/12562612224B/minfree5083181056/4600479744B. Все8 actual raws downloaded/localcollect PASS, включая оба closed direct-Docker-argv observer joins; own-seeded512 IDs exact equal across4 variants per length. Canonical ROOT/results.jsonl appended EXACT9 rows21→30:8 `r4_request`+1 `r4_prefill_ab`. Local canonical30 rows/2 150 750B: old byte prefix/parsed21 history compared Git HEAD EXACT, numstat+9/-0 verified. Artifact/journal/local49+23/readback/status/diff/log/whitespace gates complete; accepted/closed measurement-only slice ONLY docs/CMake/observer+request tools/tests/results, experimental Session/GPU/R5/speculative outside. Raw paths/repro/stdlib observer+optional `record_request.py --vram-log` — README.md; accepted raws не append повторно, b522/dirtytrue не retag.

**Завершённый diagnostic profile, не performance:** rocprofv3 job `1791052186288-750` COMPLETE exit0/2m14s на frozen `b522429c5933f493a5838317dc0bfc26ee4158aa`/dirtytrue; prefix1024 R0text/outputs32/TG31, native PASS PP43116.686375ms/TG4488.729555ms/load70708.105268ms. Profile включает PP+TG; overlapping sums НЕ total latency/НЕ PP-only attribution: HIP2760730 calls/36.11848s, hipMemcpyAsync1062432/30.0312s, launch839656/5.11047s; GPU1831186/28.37233s. fattn_dec_chunk12660/9.362744s (33% GPU sum), Q4_0matrix32<2,8,2>193474/5.847864s (20.61%), copyBuffer989831/3.537115s (12.47%), Q6 head matrix6<8,2>896/.712313s (2.51%). MEMDMA72646/5.562979s — не nearly-million D2D copyBuffer calls; RAM/expert-DMA/memcpy bottleneck не доказан суммами или upload bytes. Head не dominant, headskip не major-win priority. Raws `ROOT/runs/r4-profile-b522-1024.jsonl`/`.err`, trace/stats CSV directory `ROOT/runs/r4-profile-b522-1024/`, nested summary `prefill_/core/runs/r4-profile-b522-1024-summary.txt.txt`; full554MB trace remote/not downloaded, model not copied.

**GPU-copy measurement-срез принят/закрыт; полный R4 открыт.** Source-accounted per-assignment Q8 gather/down scatter (`2*(1024*10*48)=983040` small D2D copies для1024 PP) заменён одним80*N-byte route DTO upload/layer и indexed gather/scatter per microtile≤8, с сохранением copied bits/expert fold order. Job765 подтвердил обеGPU route-copy/CPUlinear_GPUmiddle/full-model/default gates; correctness-срез принят, см. R5. Exclusive TRACE-OFF job `1791073403122-769` COMPLETED exit0/57m33s, GPUidle: saved `/core/build/core-session-baseline` SOURCE `b522429c5933f493a5838317dc0bfc26ee4158aa`/dirtytrue против ALREADY-BUILT765 `/core/build/core-session` SOURCE `a4b55d84724ba15bbae7013d5a107b7671b7a409`/dirtytrue; без rebuild/future-attention mirror, не retag parentHEAD6faf14ed. На каждой4K/16K длине A1→B→A2, ОБА chunk1024/slots112/fresh/noPrefixReuse/warmnessunknown/MTPoff/primary1/.95/20/seed12345/ignoreEOS. Все6 requests дали512actual outputs каждый (3072total), TG511/RNG512, consumed4607|16895/cap4608|16896; три512-ID массива внутри длины identical diagnostic-only. Exact R0 generated archive/code4K IDs/four-concat16K, не retokenized/original-user/externalMTP2comparison. Raws `ROOT/runs/r4-indexed-a4-{4k,16k}-{a1,b,a2}.jsonl`/`.err`, sameprefix `series.log`/`fixture-source.json`; подробные числа — results/raws, summary/repro — README.md.

B4K PP25.68871522181/TG10.07616889732, B16K PP22.60272136899/TG9.85092544783. Mean-A/B PP1.02718318141×/1.01999903571×;request1.01937816968×/1.01916904667×. B faster BOTH A дляPP/request на обеих длинах: modest2–2.7% PP/about1.9%request, не largewin/TGthreshold. B uploads305853440000B/890465689600B sameoldB, fullrequest, неPP-only. Rocprof memcpy83% API-duration sum не dominance proof; tracefree gain modest. Canonical ROOT31→38 EXACT6 `r4_request`+1 `r4_indexed_route_copy_ab`,2497537B. Initial paired append missingmodel failed BEFORE write после6 validappends; parent re-collected37existing records excepttimestamps и addedONLY corrected paired сmodel, no duplicate/historyrewrite. Parent LOCALcanonical/all6raws/fixturemetadata downloaded; currentrecord_request.collect all6 PASS/exactsource-footer-counts-timings-throughput matchremotejournal/three512IDsperlength same. GitHEAD31 byteprefixANDparsed31 history EXACT preserved; ACTUALgitnumstat+7/-0/diffcheckPASS. Measurement-срез ACCEPTED/CLOSED, canonical38/acceptedraws НЕappendagain; fullR4/R5goalsOPEN.

**B8 attention / CLI2 correctness-срез qualified; полный R4 открыт.** Исходный HEAD среза `77f89c3412fff65634df8b45b08fab1b3da028a0`; compiled Source SNAPSHOT77f/dirtytrue неизменен и НЕ retag future commit. Job `1791083481832-775` terminal exit1/26m29s ПОСЛЕ full strict CXX20/HIP20 gfx906 Release/all-warning build/CTest36/36 (803.76s), обеGPU attention component PASS и original ModelB8 executable exit0/31records. Ошибка collector требовала2 host-logit owners при cpu0; actual Session только `host_logits`1017118720B, `working_logits` только cpu_workers>0. Parent исправил source-derived lower floor1owner, НЕ numerical tolerance, добавил regression local29/29 (43.187s). Corrected actual LOCAL/remote B8 collect PASS. Continuation `1791087203796-820` COMPLETE exit0/21m01s на SAME already-built775 binaries: fullCTest36/36 (808.07s) с correctedtests→actualB8collect→CLI2mixed/off32→defaultreset/batch/memory+strictcollect ALLPASS. Defaultcore-memory cap131072/slots112 PASS — неoccupied128K/новыйappend.

Component raw: EXACT3 JSONrecords source+twoGPU, НЕ footer; each26cases/304queries/1867776bit+1867776CPUvalues, maxabs1.1920928955078125e-7/maxboundratio.0003692344547586807,23device/180hostrejects/6sticky. Batchprimitive ADAPT currentfurnace-derivedN1: private query dimension, exactoldQ4→halfRNE/sortedselectedID/64-keysplit/merge, bounded typed borrowed API/sticky atomic publication. Existing mx/furnace MIT/reinstinctApache pins unchanged, new owner/API/CLI/schemafixtures original, no newdependency.

Actual ModelB8: ONEsameSession cap2088/slots1/max1024/tile8/cpu0, oldN1 all2088 reference (2056teacher+32continuation), enabled1024/1024/8+32N1, occupied5N1→997/997/57+32N1. 2163completedcalls/6264rows/1555476480finite/1036984320fullvocabcompared; ZEROerrors/violations/diagnosticbits,4176argmaxdiagnosticmatches при frozen `.02+.002*abs(ref)`/bitnotrequired. Logical2047–2056 coverage НЕ GPU selected-ID/visibility trace/independentHF/new4K16K/perf/exactpeak/release. 2190memoryobs/26serialized/8atomicrejects/6toggles/preserved6073162240values; individualtoggles/неserializedobs driverassertions-only. Completed attention APIcounts6180calls/49284rows/6168multi/49272multirows/12singleton/max8 НЕkernels. Actualprivate40338944B/GPU = fourbuffers40142336+reusedf(15)prefix196608, НЕextraallocation; selection82080 separate, f(15)/f(17) backing50331648B each alreadyledgered. MinfreeGPU0/1:12671549440/12115804160B.

Actual CLI2: cap64/slots1/chunk32/prompt[BOS248044,100..130]/generate32/ignoreEOS/sampleSeed42/primary1/.95/20; mixedcpu1/modeMixed/quota2/tile8 vsOFFcpu0/disabled/tile1. BOTH32actualoutputs/63consumed/31TG/32RNG/PPcalls1, IDsidentical diagnostic-only. Nativeprotocol2 explicitknobs/26hybrid+2route+6attention typedcounters, actualcollect LOCAL+remote PASS. Mixedshort1488/wide48/cpuGroups11255/gateup11255/down11255/middlecols11255/batches1488/pairedH2D57625600B/Q8D2H8103600B/returns115251200B/GPUhitgroups649/GPUmissgroups2976/admitted2976/evicted1440. Shortdiagnostic TGmixed27789.104387ms vsOFF6521.876872ms НЕ properpairedperformance/speedgain/policyselection; cpu0default unchanged. Literal raw `candidate_unqualified`/`local_unqualified` labels preserved; bounded qualification here НЕ Source rewrite/allknobs qualification.

Raws ROOT/runs/`r4-attention-77f-a-{build.log,component.jsonl,model.jsonl}`, continuation `r4-attention-77f-b-{gates.log,cli2-mixed.jsonl,cli2-off.jsonl,reset.jsonl,batch.jsonl,memory.jsonl}`. Parent LOCAL downloaded actual model/component/CLI2 raws, strictLOCALcollect всех3actual model/mixed/off raws PASS. Canonical ROOT/results.jsonl41/2563568B downloadedLOCAL: EXACT3appends38→41 [`r4_attention_prefill`,proto2mixed `r4_request`,proto2off `r4_request`]. Parent verified GitHEAD38 EXACTbyteprefix+parsedhistory, actualGit+3/-0 и gitdiffcheckPASS; artifact Source SNAPSHOT77f/dirtytrue preserved, НЕ retag closure/futurecommit. Bounded correctness-срез ACCEPTED/CLOSED; полный R4/performance остаётсяOPEN. Acceptedlogs NEVERappendagain. Repro: existingDockercontract/explicit `sh /core/src/tools/build.sh`100644, `core-attention-batch`/manual `core-prefill-attention-test MODEL`, then `record_prefill_attention.py --raw FRESH --results ROOT/results.jsonl` ONLYafterexit0 — README.md.

**B8 correctness принят/pushed `3cc594d0783974aec4cf6f4657d8e0c6864139d2`; отдельный bounded performance-measurement slice ACCEPTED/CLOSED.** Exclusive trace-off job `1791090828619-823` exit0/52m44s: saved765 A `/core/build/core-session-baseline`1556208B/source `a4b55d84724ba15bbae7013d5a107b7671b7a409`/dirtytrue против already-built775 B `/core/build/core-session`/source `77f89c3412fff65634df8b45b08fab1b3da028a0`/dirtytrue. Без rebuild/heavycompile/otherGPU. На каждой4K/16K длине A1→B→A2: ОБАchunk1024/slots112/fresh/noPrefixReuse/cachewarmunknown/CPU0disabled/MTPoff/primary1/.95/20seed12345ignoreEOS; AoriginalN1tile1, Bexplicitquerytile8/protocol2. Literal `candidate_unqualified`/`local_unqualified` и Source не переписывать под futurecommit. Exact R0archive/code4K IDs/fourconcat16K, неoriginaluser/retokenized/externalMTPbaseline. Все6strictRemoteCollectPASS,512actualoutputs каждый/3072total/TG511/RNG512; три512-IDarrays внутри длины identical diagnostic-only.

B4K PP35.5847526706/TG10.0650106725/request165876.889925ms; B16K PP32.1364201968/TG9.7725973247/request562117.081524ms. Mean-A/B PP1.39305583109347×/1.419990623620387×, request1.2711253802945932×/1.3818104027209437×; B быстрее ОБОИХ As поPP/request на обеих длинах. TG остаётся~10/targets400–600PP/30–40TG unmet. Completedwall PP безoutputsampling/TGoutputs−1/loadseparate; нет новых GPUevents/profile/VRAM/exactpeak/HF/marginalRNG/CPUdispatch claims. Полные6 timings/rates и fixedrawfilename/argv-vs-Source/fresh-repro — README.md; ROOT/runs/`r4-attention-77f-ab-{4k,16k}-{a1,b,a2}.jsonl`/`.err`, `r4-attention-77f-ab-series.log`/`-fixture-source.json`.

На историческом823 closure canonical ROOT/results.jsonl EXACT7appends41→48/2897341B:6 `r4_request`+1 `r4_attention_batched_query_ab`, pairedsource77fdirtytrue/requestactualSource unchanged. Old41 byteprefix+parsedhistory verifiedREMOTE. Parent downloaded6raws/fixturemetadata/canonical48journal: все6actualLOCALcollectPASS/Source+complete+rates EXACTmatchROOTrows/three512IDsarrays withinlengthsame diagnostic-only; oldGitHEAD41 EXACTbyteprefix+parsedhistory и actualgitdiffnumstat+7/-0/diff--checkPASS. Bounded performance-measurement slice ACCEPTED/CLOSED/pushed `ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2`; NEVERduplicateappend. Тогда commit содержал только четыре docs+journal, prerequisite code вне scope; compiled77fdirtytrue неretag.

**Текущая задача / оставшиеся критерии R4:** qualified775attention/source77fdirtytrue уже ONEbaseline `/core/build/core-session-baseline`1750328B, promoted BEFORE831; futureA явноtile8/CPU0disabled. Исторический savedA/sourcea4 НЕподменять77f в823raws. Currentcore-session — build831/sourceba446f9dirtytrue, не новый performancebaseline. Actualstrict checkpoint/tap/restore --wide-1024, bothGPUdense-Q4/actualMtpModel и DEFAULTregressions PASS (R6 ниже); boundedR6prerequisite ACCEPTED/CLOSED correctnessONLY после parentactualLOCAL/history/+2/-0/diffcheckPASS. Immediateparent normalcommit/push, затем trained-forward teacher-reference qualification. Ignored `runs/r4-layerwise-prefill.patch`61985B UNAPPLIED/UNQUALIFIED; PPweighttraffic и R5CPUworkers/admission-policy требуют новых measurements, не bottleneck-доказательства по overlapping sums. Attentiontile1/OFF/originalAPIdefault unchanged; APIcounts НЕkernelcounts. Hybrid speedwin/dispatchthreshold, exactpeakVRAM, fullR4/R5–R8/marginals, occupied128K/MTP2/finaltargets OPEN; wideMMQ component-only.

**Действия.** Адаптировать mx staged MMQ и reinstinct sort/gather/grouped tiling/scatter для grouped expert GEMM и загрузки tiles наших Q4_0/Q4_1 экспертов. Не предполагать GPU-resident slab всех экспертов. Сравнить mx chunked GDN и furnace resident-state вариант на реальной геометрии; не делать host loop с тысячами decode launches.

QSA обрабатывает запросы тайлами с per-query causal visibility; не создаёт полноконтекстные промежуточные маски для всего большого чанка. Постоянный K/V остаётся Q4_0; сначала gather + dequantize только выбранных rows в ограниченный FP16 scratch, адаптированный furnace sparse attention с FP32 reductions/softmax. Использовать selected IDs непосредственно, без масочного frontend донора. До 2051 IDs требуют около 4 MiB K/V scratch на query; размер query tile выбирать по фиксированному workspace budget. Не распаковывать всю историю K/V в FP16. Прямой indexed Q4 attention сравнить с gather после корректного первого пути.

Сначала 1024/2048, затем 4096/8192 только при измеренном выигрыше и доступном workspace. Dense GEMM разрешено брать из rocBLAS после shape-specific проверки. Сравнить dequant+GEMM и quantized путь на настоящем распределении токенов по экспертам.

Сохранить полезное reuse resident expert weights в PP из mx prefill D2D; canonical/repacked layouts и padding должны соответствовать consumer-у, не делать слепой D2D repacked slot в canonical tensor. Добавить compute/copy overlap через события и bounded double buffering. Для каждой GPU измерить полезную работу и простои. После одночанковой корректности адаптировать event-ordered pipeline двух prefill-чанков между GPU из furnace/reinstinct. Если он не окупается, оставить более простой вариант. Учесть стоимость восстановления expert cache после заимствования VRAM под PP; две GPU не обещают 2× single-sequence TG.

**Готово:** 4K и 16K prompt без prefix reuse обрабатываются корректно с Q4_0 K/V; границы чанков не меняют causal смысл; есть PP, полное время запроса и peak VRAM по GPU. QSA workspace ограничен выбранными rows/query tiles, нет полноконтекстной FP16-копии K/V или ложного увеличения PP за счёт незаметного reuse.

## R5. Decode, cache и короткий multi-token MoE

**CPUlinear_GPUmiddle correctness-срез принят; полный R5/performance открыт.** Remote job765 COMPLETED exit0/34m46s; последующая exclusive серия769 также COMPLETED со своим measurement scope выше. Полный strict CXX20/HIP20 Release gfx906/all-warning-gates build, CTest35/35 (735.71s), обеGPU route-copy/paired-middle fixtures, full hybrid348-record fixture и default reset/batch/wide1024/capacity131072 memory/strict actual collectors PASS. Compiled source — SNAPSHOT `a4b55d84724ba15bbae7013d5a107b7671b7a409`/dirtytrue, не future closure commit. Default `cpu_workers=0` сохраняет прежний GPU-only path/allocations.

Метод: exact CPU gate/up → CANONICAL GPU SiLU×up/Q8_1 middle (один aggregate batch на CPU-bearing layer) → exact CPU down. API `force_cpu` имеет diagnostic label **CPU_LINEAR_GPU_middle**, не wholeCPU. Original host-libm whole-expert path остаётся oracle и не full-model-qualified: job755 frozen gate FAIL token100/offset1; job759 actual layer46/expert297 witness имел identical gate/up,24 middle FP32 ULP-scale differences и один block11 raw-sum half-header difference13734vs13733 при одинаковых codes/d. Primitive bounds проходили, propagated full gate — нет. Quantizer ABI/GPU math/weights/precision/epsilon/gates неизменны; tolerance waiver отсутствует.

Original bounded scheduler: persistent1..15 workers/fixed30 jobs, stages `gate_up`/`down`; existing pinned output frame сначала paired gate/up, reuse только после `middle_ready`. Dedicated copy-stream middle buffers/flag устраняют гонку с compute Q8/scratch/error. Только CPU-rank H2D, record `cpu_returned` перед compute wait, original-rank fold плюс shared expert. Captured cache readers/pending IDs/admission/error/reset lifetimes qualified; synthetic accepted-CPU+queued-admission failure не actual-inflight timing/overlap-speed proof.

Actual protocol1:3 sequential owners/1 simultaneous, batches[3,1,4]/workers[1,1,4], retained GPU-N1 teacher BOS248044,100..130+continuations131..138.11 phases/326 windows/528 full-vocab rows/131112960 values и508 intermediate rows/624230400 unweighted-down/62423040 FFN values: all finite, zero violations/maxabs/maxboundratio/diagnostic bit differences при frozen logits `.02+.002*abs(ref)`/intermediates `.002+.002*abs(ref)`. ForceCPU-linear/GPU-middle N1/2/3, mixedslots1/forcedGPU/warmallhit, synthetic failure+reset и5 N4 GPU fallbacks→short resume PASS. Bit identity diagnostic-only; same-runtime self-reference не independentHF/performance. Compared-window totals:gateUp/down106984 jobs each,middle137581 columns/10416 batches,pairedH2D704414720B/Q8D2H99058320B; reference/failed-call excluded.

New middle capacities perGPU:4 independent buffers252004B (153600/76800/21600/4), pinnedmiddle43200B total. Input probe855648B INCLUDED in total hostprobe33295968B. Capacities/explicit metadata не RSS/stacks/allocator slack. Collector175 rejection checks SOURCE-DERIVED, не serialized per-rejection proof; protocol не выдаёт before/after stats/route/physical-routing snapshots, original input payloads, steady owned-category/read ledgers или complete all-owner owned release. Added-buffer capacities observable; остальные отсутствующие observations не восстанавливать по assertions. Separate defaultwide cap1056/slots1/max1024:786677760 comparisons/zeroerror, не 4K16K rerun. Defaultmemory cap131072/slots112 PASS:owned9515698008/10007105368B,free7166787584/6686539776B,afterdestroy16968876032B обеGPU; это allocation/release, не occupied128K.

Logs ROOT/runs/`r5-middle-c-build.log`, `r5-silu-pairs-c.jsonl`, `r5-route-copy-c.jsonl`, `r5-session-hybrid-middle-c.jsonl`, `r5-middle-default-{reset,batch,wide,memory}-c.jsonl`. ROOT=`/home/radneon/gfx906-core`. На историческом correctnessclosure canonical ROOT/results.jsonl: EXACTLY ONE append30→31 kind`r5_hybrid`/passedtrue, accepted raw НЕ append повторно. Parent LOCAL `collect(downloaded_actual_r5-session-hybrid-middle-c)`272019B PASS/all compared metrics zero; тогдаlocal journal31/2167805B EXACT old GitHEAD30 byte prefix/parsed history и +1/-0 verified. Bounded correctness closure accepted: ONLY Session/CPU/ops, relevant fixtures/CMake/collector/docs/journal; post765 attention3 и futureSpec3 excluded. Artifacts sourcea4b55d8/dirtytrue preserved, не closurecommit6faf14ed. Fresh repro — explicit `sh /core/src/tools/build.sh` (100644), `core-session-hybrid-test MODEL`, затем `record_hybrid.py --raw FRESH --results ROOT/results.jsonl` строго после executable exit0; полный Docker contract — README.md.

Accepted paired baselineb522 vs ALREADY-BUILT765 job769 дал modestGPU-copy PP/request gain; parent LOCALactualcollect/exact31history/+7/-0 PASS. Qualified765a4 был ONEsavedbaseline для823; B8/CLI2 correctness775/820 accepted/pushed3cc594d, compiled77fdirtytrue/неfuturecommit. Trace-off B8 A/B/A job823 COMPLETEexit0 с PP/request выигрышем выше; boundedmeasurement ACCEPTED/CLOSED/pushedba446f9 после parentLOCAL6collect/exactROOTmatches/GitHEAD41 byte+parsedprefix/+7/-0/diffcheckPASS. Qualified77577f promoted ONEbaseline BEFORE831; R6 prerequisite ACCEPTED/CLOSED correctnessONLY/sourceba446f9dirtytrue после actualLOCAL/history/+2/-0/diffcheckPASS, parent normalcommit/push immediate→trainedforwardteacherreference, R5policy/PPtraffic всё ещё measurementOPEN. CLI2mixed/off32 timings толькоdiagnostic, НЕproperpairedperformance/CPUdefaultpromotion. Full-request uploads/overlapping PP+TG sums не заменяют CPUmiss vs H2D+GPU/overlap measurements. Hybrid speedup/dispatchthreshold/fullR5/MTP/400–600PP/30–40TG не подтверждены. Тогда prerequisitecode excludedfrommeasurement-onlycommit; новая831 qualification имеет отдельный scope. Existing mx/furnace MIT/reinstinct Apache pins сохранены, new orchestration/fixtures/diagnostics original, Threads/stdlib без нового donor runtime/dependency.

**Действия.** Оптимизировать по профилю, а не по списку доступных инструкций. Подобрать CPU threads/affinity/first-touch и простой cache admission. Сначала baseline 112 slots/layer, затем разумный бюджет из свободной VRAM. Один owner на слой, никаких remote expert RPC.

Для 1/2/3 позиций объединять routes по expert ID на CPU и GPU, читать весовой tile один раз на несколько аккумуляторов. Адаптировать reinstinct multi-column matvec после группировки: несколько отдельных GEMV по assignments не дают этого reuse. Собственный гибридный scheduler должен загрузить эксперта один раз для всех позиций окна, перекрывать AVX2, DDR reads, DMA и GPU compute при измеренном выигрыше. Fuse gate/up при выигрыше. Проверить все-hit/all-miss/mixed и повторяющиеся IDs, переходы cache slot FREE → LOADING → READY → IN_USE. Не публиковать новый ID до окончания всех нужных копий; не перезаписывать slot до завершения readers.

Сравнить CPU miss и H2D+GPU miss на измеренных размерах с учётом reuse, repack и конкуренции за DDR3. Не загружать все misses на GPU по религиозному принципу. Минимизировать лишние round-trips и выделения памяти. Готовые DPP/sdot4 primitives уже используются с R1; новые inline AMDGCN изменения делать для найденного горячего места после disassembly и A/B. HIP graphs адаптировать позднее, если submission overhead действительно значим; positions/lengths/slot maps передавать через device descriptor, не замораживать host scalars и CPU handoff в whole-engine graph.

**Готово:** есть отдельные 1/2/3-token block measurements, cache accounting и сквозной non-MTP baseline. Микроускорение принимается в основной путь только после проверки полного запроса.

Порядок R4/R5 допускается менять после R3 по измеренному узкому месту. Их результаты нужны до окончательной оценки MTP2.

## R6. MTP2 с корректным состоянием

**2026-10-04: bounded trained-forward/teacher slice implemented.** Own MtpSession
executes real sidecar and borrows target embedding/head. Fresh strict build,
CTest39/39, chronological32 N1/N2/N3 self-check/reset/rollback/poison/KV-only
passed; independent sequential donor with explicitly selected short-canonical
CPU dense correction passed all327680 D/7946240 logits at unchanged gates.
Raw donor/direct-division and batched-donor N2/N3 are NOT qualified: retain their
numerical failures and the documented batch-sensitive FFN/head witness.
Canonical journal51 includes successes AND failed controls. No wholeR6,
HF equivalence, longer-history or MTP throughput claim. See README R6 trained forward.
The test-only trained coordinator now couples SpeculativeSampler to target verify,
teacher rebuild and restore. Seven native cases pass, including55 replay-checked
windows with actual0/1/2 acceptance[33,10,12], restored target/draft continuation
and an uninterrupted32-output equivalence check. EOS-enabled case emitted no EOS.
Opt-in reusable MtpRunner and core-mtp-run now pass strict build/39CTest, API request/reset/rejection/stop fixtures and generated8+512 output smoke. Journal54. Actual modelEOS and longer-reference qualification remain open. Next: matched4K/16K MTP-off/on, external llama.cpp baseline and measured optimization; no speed promotion from smoke runs.


**Bounded R6 prerequisite slice ACCEPTED/CLOSED, correctness ONLY; R6_complete_claim=false.** Actual run831 DONE/native exit0. Snapshot строго `ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2`/dirtytrue, не futurecommit. MCP job `1791103589280-831` получил `state_lost/background_channel_closed_without_exit_status` после14m15s; READONLY Docker daemon die event ORIGINAL контейнера `core-r6-prerequisites-check` установил nativeexit0/2772s46m12s без повторного run. ContainerID `35ed6fbda1d6c386cb1d84819addcaf4650460044619ba658867de52e73a0644`, time1791106362/timeNano1791106362809343377; downloadedignored proof `runs/r6-prerequisites-ba-a-exit-proof.json`719B. Strictfull CXX20/HIP20 Release gfx906/allwarnings/ffp-contractoff PASS, CTest39/39 actual1145.49s. MtpModel666checks/451rejects и Specpuremath35963782checks/715rejects/5522274hotcalls zeroheap PASS; это НЕ trainedMTP.

Actual `core-mtp-model TARGET SIDECAR`: `runs/r6-mtp-model-ba-a.jsonl`36records/14462B,32roletyped descriptors/offsets/strides, actualQ8_0×19/F32×11/BF16×2 толькоindexQ/K; top10/fulltypedtokenizer10assets/blk48compression0dense/target-owneddistinctembeddingQ4_0/outputQ6_K checks PASS. Descriptor-derivedpayload2775621632B/eachgate-up-downstride1740800B; payloadread/trainedexecution нет. Actual `core-mtp-attention` обеGPU: `runs/r6-mtp-attention-ba-a.jsonl`4records/1298B, B1..3/densefullcausalprefixQ4KV/synthetic128K, frozen `2e-4+2e-4*abs(ref)`/oldN1shortbitparity/finite/bounds/poison/reject/sticky/canaries PASS; cleanupfooter live_resources0. KEEPmxsum64/ADAPTcurrentfurnace64splitmerge/WRITEOURSdirectcanonicalQ4→FP16RNE; неoccupiedfullmodel128K/HF/trainedforward/performance.

Actual target `core-session-restore-test MODEL --wide-1024`: `runs/r6-target-restore-ba-a.jsonl`473730B/protocol1/56records/3sequentialowners, primarycap40batch3CPU0/widecap2048batch1024CPU0/CPU1syntheticfailure+reset. Acrossallowners644390400fullvocabcomparisons/2595rows и26419200EXACTtapvalues, allmaxerror/maxratio/diagnosticbitdiff ZERO при unchanged `.02+.002*abs(ref)`. Aggregates51successfulrestores/84argumentrejects/137steadyMem/6stickyrejects; footerprimary-only50/83/133.45primarywindows N1..3×ALLretainedprefix0..N/mod4/EOS+PLEhash-conv/QSAtail/divergentsuffix/oldpublishedspan-tapfailurepreservation, widePPALL1024taps PASS. Tap widened10240 BEFORE ROOT_HC; restore logicalcursor/recurrent/conv/PLE/tails, НЕ rewind expert slots/uploads/physicalcounters. Invalid/executionfailure preserves oldpublication; executionfailure requiresreset.87optinGPUBuffers/defaultOFF: batch3GPU0/1 236851200/235622400B; batch1024GPU0same/GPU1 319262720B, GPU1two transactionaltaps41943040B EACH. Это target consumed-input self-parity/state proof, не trainedMTP rollback/HF/throughput.

`runs/r6-prerequisites-ba-a-build.log` и `runs/r6-default-{reset,batch,attention,memory}-ba-a.jsonl`: все model/default processes nativeexit0/strictcollectPASS. Defaultcap131072/slots112 — НЕ occupied128K. ONLYcanonicalROOT `/home/radneon/gfx906-core/results.jsonl` EXACTTWOappends48→50/2967969B [`r6_target_restore`,`r6_mtp_prerequisites`]; remoteold48byteprefix+parsedhistory EXACT. All3actualraws+exitproof+canonical downloaded; parentREADONLYactualLOCAL `record_restore.collect`473730B PASSagain/allerrorsZERO644390400compares26419200tap; descriptor36+dense4 actualSource/footer/inventory/device EXACTmatchesROOTnewrecord. GitHEADold48parsedhistory ANDbyteprefix EXACTagainstlocal50; actualgitnumstat+2/-0/diffcheckPASS. Boundedprerequisite ACCEPTED/CLOSED correctnessONLY. НЕ duplicateappend acceptedraws/retagSource. Freshrepro/explicit`sh`build/strict`record_restore.py --raw FRESH --results ROOT/results.jsonl` толькоafterexit0 — README.md.

**Ближайшая задача:** parent immediate normalcommit/push acceptedprerequisitecode/docs/journal, затем apply/review trained-forward candidate и actualindependentdonororacle/fullLogits/state/teacherhistory/stochasticwindows. FIRST Hnorm — perbranch2560 с distinctgamma4×2560 по pinned `qwen4exp.cpp:438..450`; whole10240 reduction REJECTED, HCmixnorm whole10240 unchanged. Ignored `runs/r6-trained-mtp-forward.patch`73983B и `runs/r4-layerwise-prefill.patch`61985B BOTHUNAPPLIED/UNQUALIFIED/no speedpromise. Qualified77577fbaseline promoted before831 остаётся performancebaseline; current831 НЕperformancequalified. R6trainedMTP/fullR4speed/R5measuredpolicy/R7occupiedlong-tokenizer-API/R8pack-finalgates остаются обязательными OPEN.

**Действия.** Загрузить настоящий MTP sidecar, проверить его tap, shift, shared embeddings/output и собственное history state. Его K/V хранить в Q4_0 с теми же правилами packing/rotation, что у target; Q8_0 весов sidecar не определяет формат KV. Сначала greedy verify, потом стохастическая схема с правильной correction. Не заменять trained MTP маленькой случайной draft-моделью.

Явно различать три позиции: выданные пользователю токены, уже потреблённые forward токены и pending sampled token. Bonus/replacement token может ещё не входить в KV/state. Уточнить этот контракт в коде до optimization, иначе ошибки на один токен маскируются правдоподобным текстом.

Target restore уже квалифицирован с краткими chronological recurrent/conv/PLE/QSA-tail prefix states для0..3 consumedinputs и логическими KV lengths. Для trained MTP дополнить его собственным history/length state и проверить paired rollback. Вернуть число видимых pooled blocks и raw-хвост target/MTP; отвергнутые позиции Q4 K/V исключать по lengths и перезаписывать перед новым чтением. Никогда не копировать весь128K KV на каждое speculative окно или откатывать physicalexpertcache/uploads/counters вместо логического состояния.

Verify использует grouped N=2/3 expert path R5, а не несколько самостоятельных decode. Адаптировать поддержку коротких GDN prefix states mx/furnace к единому state layout R2 и контракту consumed/pending/committed; серверный checkpoint scheduler донора не нужен. Проверить parity состояния после выбора каждого принятого префикса.

Тесты: приняты 0/1/2 drafts, all-reject несколько окон подряд, завершение EOS, предел контекста, блок QSA завершился speculative токеном и откатился, PLE пересёк EOS, reuse после генерации. Проверить состояние против последовательного исполнения тех же фактически потреблённых токенов. Для stochastic correction добавить маленький тест распределений, не требующий запуска всей модели.

**Готово:** MTP2 работает корректно на тех же sampling settings с Q4_0 K/V у target и draft; есть acceptance histogram и время draft/verify/restore. Скорость считается по реальным output tokens. Сравнение с MTP off сначала с одинаковым бюджетом cache; перераспределение freed draft VRAM является отдельным экспериментом.

## R7. Длинный контекст и практическое использование

**Действия.** Последовательно проверить занятые 32K, 64K и около 128K. Для capacity 131072 оставлять место под 512 outputs, special tokens и необходимое speculative lookahead. У самой границы уменьшать draft horizon, не выходить за buffers.

На каждой длине использовать Q4_0 K/V у target/MTP, проверять causal visibility и выбранные teacher-forced logits против reference; FP32/FP16 эталон ограничивать малыми fixtures. Сохранять capacity, фактическую длину, выделенные KV/index/state/workspace bytes и peak VRAM по каждой GPU, включая prefill и verify/restore. Проверить отсутствие роста workspace до полноконтекстной FP16-копии K/V.

Отдельно измерить cold/full PP, warm expert cache, incremental PP после длинной истории и TG на этой истории. Добавить несколько retrieval fixtures с ответом в начале/середине/конце, но не выдавать их за полную оценку качества.

Проверить cancellation между безопасными boundaries и повторный запрос; append-history reuse; EOS/stop handling и корректный tokenizer/chat template. После рабочей CLI добавить минимальный streaming API готовой библиотекой или тонким внешним адаптером. Tool-call output сериализуется по template, ядро не исполняет инструменты за клиента.

**Готово:** реальный длинный ввод обработан с Q4_0 K/V, 512 outputs измерены, continuation корректен; память укладывается в VRAM каждой GPU с headroom. Capacity allocation не считается тестом 128K. Есть практическая команда запуска/endpoint без UI, multi-user scheduler и model zoo.

## R8. Runtime-pack из оригинальных safetensors

**Действия.** Сделать потоковую конвертацию shard/tensor за раз в простой pack с metadata и offsets. Нормализовать fused tensor layouts, HC/norm conventions, MTP tap/projections и 128 частей PLE в правильном порядке. Не выделять полный BF16 checkpoint в RAM. Не требовать размещения всех исходных shards и нескольких полных output packs на одном разделе.

Начать с воспроизводимого Q4 pack и reference fixtures на отдельных оригинальных tensors. Явно выбрать full-PLE или keep1 как вариант; полная PLE не считается теми же весами, что baseline. Если нужно скачивать, сохранять только рабочий набор shards и завершённый pack; обычное возобновление передачи допустимо, собственная система аттестации файлов не нужна.

Затем repack без изменения чисел и calibrated mixed precision. Исследовать Q3.x только если CPU/GPU kernels и качество оправдывают формат. Не считать цель 3.25–3.75 bpw требованием любой ценой. Full-PLE Q3.x может оказаться интереснее keep1 Q4, но сравнение должно включать качество и acceptance, а не только tok/s.

**Готово:** core читает pack, сделанный из оригинальных safetensors; происхождение, shapes и выбранная precision понятны; есть fixtures и end-to-end контроль. Новый квант допускается в основной режим по совместному результату качества, памяти и latency. Если Q3 не окупился, оставить рабочий Q4 и отрицательный результат, а не подгонять тест.

## Целевые показатели

Числа ниже являются инженерными целями до профилирования новой реализации, не обещанием и не критериями математической корректности. Пересматривать их после R0/R3 по реальной critical path.

**Известный исходный результат пользователя:** 4K PP 220.7/TG 23.7; 16K PP 209.7/TG 19.2; MTP2, warm expert cache, 512 outputs. Более длинные точки раньше не измерялись.

**Первая полезная веха после функциональной готовности:** на сопоставимой серии 4K получить примерно 350PP/30TG, на 16K примерно 400PP/28TG с MTP2 без скрытого изменения весов/KV. Это ориентир выбора работы, не причина пропускать правильность или останавливаться навсегда ниже цели.

**Основная цель same-Q4 runtime:** 400–600PP и 30–40TG на 4–16K; полный 16K+512 запрос порядка 40–60 с вместо пользовательских 104.7 с. Время вычислено как N/PP +511/TG, без небольших serving расходов и с первым токеном из prefill. На каждом реальном run сохранять непосредственно измеренное полное время.

**Цель остаётся открытой и НЕ достигнута:** latest actual non-MTP B775querytile8/chunk1024 в completedjob823 дал4K PP35.5848/TG10.0650 и16K PP32.1364/TG9.7726; boundedperformance-measurement slice accepted/closed с parentLOCAL/history/+7/-0/diffcheckPASS. Own chunk/GPU-copy/attention A/B/A выше не являются external MTP2 baseline speedup; series fixtures и sampling/RNG provenance сохраняются отдельно.

**Условная цель длинного контекста:** 64K около 300–600PP/23–34TG; около 128K около 250–500PP/20–30TG. Эти цели требуют устранения значительной части полноконтекстной работы QSA. Если измеренный наклон latency остаётся прежним, указать фактические более низкие результаты. Линейную модель по двум точкам не выдавать за прогноз архитектуры.

**Stretch после собственного удачного кванта:** около 500–850PP/35–50TG на 16K. Качество, CPU unpack и MTP acceptance могут сделать этот вариант невыгодным. 80–100TG не является требованием проекта.

Не ускорять один показатель за счёт намного худшего другого молча. Для каждого важного выбора сравнивать оба режима: большой новый prompt и продолжение существующей истории.

## Минимальный контур контроля

После небольшой правки: затронутый fixture и smoke. После performance-кандидата: парные прогоны на коротком и среднем запросах. После архитектурной вехи: несколько prompt families, 512 outputs и длинные границы. Подробный trace выключать в окончательном benchmark.

Хранить в results.jsonl: revision/dirty, вариант модели/KV, input/output token counts, prefix-cache состояние, expert-cache состояние, sampling, PP time, TG time, total time, accepted0/1/2 и краткие RAM/VRAM данные. В диагностических runs добавлять CPU/H2D/P2P bytes и stage times по необходимости. Это один файл, не новая observability-платформа.

Кандидат отклоняется при необъяснённой correctness-регрессии, подтверждённом ухудшении полного запроса или выигрыше только на игрушечной форме. Недостаточный выигрыш означает следующую гипотезу, а не создание универсального autotuner.

В STATE.md оставлять следующую физически выполнимую команду или изменение. Не заканчивать запись словами «продолжить оптимизацию» без указания конкретного участка.

### 2026-10-04 matched MTP measurement checkpoint

Own off/on/off 4K and 16K, 512 outputs each, completed natively (job163 exit0).
Canonical journal55 preserves old54 bytes and actual compiled f59c8de/dirty provenance.
No full-request gain: mean-off/on ratios0.993395/0.999081. Draft acceptance is high
([26,31,141]/[24,35,139]); target verification plus CPU filtering/decision dominates
the MTP decode timer. Do not promote smoke TG12.59 or this non-winning result.
Next is phase-separated current profiling, targeted measured optimization, and
matched pinned llama.cpp baseline. The entire plan remains active and unfinished.

### 2026-10-04 current profile guides the next candidate

Separate completed PP and MTP-verify ROCTX profiles are captured (journal56).
Attention chunk is40.92% of summed PP GPU duration and31.90% of verify GPU;
metadata sample has256 VGPR and4380 scratch. Test a rolled canonical dot loop
before changing attention math. Keep all FP checks/order/precision and gates;
check compiled metadata, component parity, full-model continuation and paired
trace-off requests before promotion. Q4 N8 and dense N8 are next measured PP
costs; head skipping is only2.91% of summed PP GPU duration here.

### 2026-10-04 pinned donor baseline and headroom

Fresh pinned llama.cpp MTP2 completed4K/16K x2 with512 actual outputs (journal57).
4K HTTP57.011/57.151s,16K144.334/144.125s. An initial511-output truncated run was
rejected and preserved separately. Use capacity prompt+1024 for all future own
baseline/candidate comparisons; previous own series used prompt+512. Preserve
TG numerator/timing-boundary differences and donor QSA semantic limitations.

### 2026-10-04 rolled canonical attention correctness checkpoint

Only QK-dot loop unrolling changed. Old/candidate component snapshot3,745,280
values is bit-identical; scratch is0 and VGPR52 (was256/4380). Full strict build,
39/39 CTest and actual48-layer N1/B8/continuation fixture pass with1,036,984,320
zero-error compared logits. Journal58; compiled7df1c74/dirtytrue. Component timing
improvement is diagnostic. Paired trace-off MTP baseline/candidate4K/16K+512,
capacity prompt+1024, is the next promotion gate. Full project remains open.

### 2026-10-04 rolled attention measured promotion

Trace-off MTP A/B/A4K/16K+512 completed with identical full output/acceptance/RNG
trajectories. Candidate requests126.693/396.449s beat both old bookends; mean
speedups1.3060x/1.4157x. Journal59; old58 bytes retained. Rolled MTP executable
is the new MTP baseline, source7df1c74/dirtytrue. Pinned llama.cpp still wins
full requests(~57/~144s). Continue measured PP work; whole-plan goals remain open.

### 2026-10-04 canonical tiled-grid component checkpoint

LogicalN<=128 via parallel physicalN8 tiles passes both GPUs,409,879,948 exact
compares,162,704 CPU samples,42 host rejects and596 graph checks. Legacy APIs
pass. Journal60/source296ea56dirty. Resident shape tests support quantized and
small-output dense atN>=16; large F32 output2560 regresses and is excluded.
Next: selective Session integration, frozen full-model gates and matched full
requests. No component number is a model speed claim.


### 2026-10-05 layerwise scheduling prerequisite
Opt-in full-window routing and bounded GPU frames pass32/128/4096 full-vocabulary
self-parity, full teacher taps, occupied continuation and ownership gates.
Both-GPU opaque byte-copy16384 bound and full39CTest pass; details README/journal78.
MTP integration, representative full-request win and16K fit remain OPEN.


### 2026-10-05 layerwise4096 full-request result
Opt-in trained MTP with layerwise4096 improves matched4K+512 to91.233s
(meanA/B1.15158) and16K+512 to267.236s (1.20148), all IDs/RNG/acceptance exact.
Journal81/README retain actual sourceb2014ce and raw evidence. Native exit0.
Still behind pinned llama.cpp57s/144s. Next sampled VRAM and refreshed phase
profile before choosing another kernel/scheduling change; full R4–R8 remain open.


### 2026-10-05 deferred attention checks: paired full-request qualification
A/B/A completed native0 at09:06:50UTC, source53d1153/dirtytrue, journal92.
Both variants use explicitlayerwise4096, frame1024, slots112, capacityprompt+1024;
only attention.hip changed, builds outside timing, preserved baseline unchanged.
4K+512: A93.663597/91.159666s, B91.140152s; PP78.853648, TG13.037670.
16K+512: A267.116887/267.692213s, B260.903173s; PP73.315235, TG13.652910.
PP mean-control time ratios1.019943/1.020554, candidate faster than both PP
controls. Request ratios1.013951/1.024919, but4K advantage over fastest control
only0.019514s: noise-floor full-request gain, not a robust4K speedup claim.
All512 IDs, acceptance, proposal/decision RNG and pending IDs exact pertriplet.
Six actual514-row logs, manifests and journal byteprefix independently checked
on controller. Ordinary runtime restored to candidate. This is a bounded
self-parity/speed qualification, not independent HF or llama.cpp victory.
Pinned donor remains faster(~57/~144s). Next: bounded two-GPU PP pipeline;
no precision, routing, QSA or tolerance changes. Full PLAN remains open.

2026-10-05: bounded host pipeline coordinator prerequisite qualified (journal93).
Session/HIP integration, causal model gates and measured overlap remain OPEN.

2026-10-05: the bounded two-GPU pipeline correctness slice is qualified
(journal94, source48f3d14 dirty). Logical4096/subwindow2048, short irregular
windows, eight failure/reset cases, trained MTP carry and full40 CTests pass.
Keep it opt-in. Full-request A/B/A and observed VRAM remain the promotion gates;
the full R4 speed target and remaining R5–R8 are still open.


### 2026-10-05: two-GPU pipeline full-request result

The trace-off A/B/A series completed natively at 11:12:52 UTC. One unchanged
runtime binary, source dc60e0f56da9d2b10bdbb5de845491470da8d0eb / dirty=true,
compared serial layerwise4096 against pipeline2048 within the same logical4096.
Capacity was prompt+1024, slots112, frame1024, attention tile8, primary sampling
(seed12345, temperature1, top-p.95, top-k20), fresh processes, no prefix reuse.

4K+512: A92.547317/92.184304s, B84.422057s; B PP88.353420, TG13.425791.
16K+512: A265.568701/260.969029s, B229.376431s; B PP85.327488, TG13.677163.
Candidate PP and full requests beat both controls at both sizes. Mean-control
request speedups were1.094096x/1.147759x; PP time ratios1.135659x/1.170804x.
TG differences are measurements, not evidence of a changed decode kernel:
prefill scheduling can change cache residency, and warmness is not isolated.

All512 output IDs, acceptance, proposal/decision RNG and pending IDs matched in
each triplet. Protected saved baseline stayed unchanged. Six actual514-row logs,
manifests, one runtime digest and the previous94 journal byte prefix were checked
again on the controller. Journal95 contains full evidence; raw directory is
ROOT/runs/r4-pipeline-ab-20261005. The selected configuration remains opt-in.
Pinned llama.cpp still wins the full requests (~57/~144s); full PLAN is open.
Next: separately observed VRAM and refreshed diagnostic GPU phase profile.

2026-10-05: bounded128 head-output staging is qualified (journal98).
Actual owned saving889,978,880B per GPU at frame1024; full output/math retained.
Native wide/attention/pipeline/MTP/default-memory gates and full41 CTests pass.
Protocol2 reports the head capacity; protocol1 retains its original bounds.
Driver peak observation and larger-window PP remain the next steps.

**2026-10-05 bounded R4 update:** stage-sized pipeline logical16K/2048 passed full-vocabulary/tap/continuation and trained16387 carry gates, plus41/41CTest; journal100. No speed promotion until paired512-output requests. Full R4–R8 and llama.cpp victory remain open.

**2026-10-05 R4 measured update:** journal101, logical16K/stage2048 gives183.372s for16K+512 versus234.193/231.932s controls, PP112.303, exact512-output trajectories.4K schedule unchanged; no structural4K speed claim. Pinned llama.cpp~144s remains faster; full PLAN remains OPEN.

**2026-10-05 R4 configuration result:** stage4096/logical16K passes full16K parity and16387carry;16K+512182.044s versus186.827/185.104s controls (~2.15%). Extra216MB/device; no default change or llama.cpp win. Journal103. Next4K stage1024 overlap test; all R4–R8 completion criteria remain open.

**2026-10-05 R4 negative result:** stage1024 for4K passed full parity/carry, but PP85.556 versus89.053/88.091 and request85.556s did not beat both controls. Not promoted; journal104. Next current16K completed-phase profile; full goals stay open.

**2026-10-05 next R4 candidate:** current16K profile journal105 exposes2,751,936 radix histogram/select launches. Implement bounded up-to8-query selection on the existing integer rank contract; component qualification first, then full-model integration. Traced197.3s is diagnostic, not a speed measurement.

**2026-10-05 R4 primitive complete:** boundedQSA batch selector passes both-GPU CPU/serial parity, invalid/sticky/graph/guard tests and legacyQSA gates;28/28 resident coordinates beat controls (8-query median6.25x). Journal106. Component only; Session integration and full-model/performance gates next.
