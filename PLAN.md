# План реализации и критерии готовности

Этот файл задаёт очередь работы, а не календарное обещание. Переходы определяются работающим кодом и измерениями. Архитектурные основания и ссылки находятся в RECON.md. Начать с R0; после возобновления читать текущую задачу из STATE.md.

## Что считается результатом

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

**Действия.** Соединить 48 слоёв в прямую последовательность вызовов. Один Session с заранее выделенными buffers; статический layer split. CPU misses и GPU hits первоначально могут синхронизироваться консервативно. Кеш и очередь копий должны сначала быть корректными.

Сохранить owner/residency и producer-consumer контракты mx expert cache в собственных slots/maps/events, без импорта ggml scheduler. Публиковать slot только после всех uploads/repack, заменять только после readers. Между GPU передавать нужный residual 4×2560; P2P или ограниченный pinned staging выбирать по проверке R0, не по настройке чужого стека.

Добавить embeddings, LM projection, greedy sampling и текстовую CLI через готовый tokenizer либо сначала CLI с token IDs. Выполнить короткий prompt и 32–128 output tokens с Q4_0 K/V. Сравнить выбранные intermediate activations и teacher-forced logits. FP32/FP16-KV допустим только для маленьких correctness fixtures, не как основной режим или замена Q4_0 в performance A/B.

Сверить выделенную память с геометрией: при capacity 131072 основной Q4_0 K/V — 864 MiB; будущий MTP K/V — ещё 72 MiB, FP32 pooled index основных и MTP слоёв — 208 MiB. Всего кеши и индекс — 1144 MiB на обе GPU без padding/workspace; это расчёт, не замер. Weights, expert cache, GDN/conv states, speculative checkpoints и workspace учитывать отдельно по каждой GPU с headroom. Память зависит от выделенной capacity, а не только занятой истории.

Проверить, что обе половины сети действительно исполняются на назначенных GPU и что процесс не читает expert weights с SSD во время прогретой генерации.

**Готово:** собственный бинарник самостоятельно обрабатывает запрос с Q4_0 K/V; все слои/PLE работают; нет NaN, утечек, бесконтрольного роста памяти и скрытого llama_decode. Выделенная память по категориям и GPU сверена с расчётом. Скорость на этом этапе может быть ниже baseline. Не бросать core ради преждевременной оптимизации одного kernel.

## R4. Полноценный prefill

**Действия.** Адаптировать mx staged MMQ и reinstinct sort/gather/grouped tiling/scatter для grouped expert GEMM и загрузки tiles наших Q4_0/Q4_1 экспертов. Не предполагать GPU-resident slab всех экспертов. Сравнить mx chunked GDN и furnace resident-state вариант на реальной геометрии; не делать host loop с тысячами decode launches.

QSA обрабатывает запросы тайлами с per-query causal visibility; не создаёт полноконтекстные промежуточные маски для всего большого чанка. Постоянный K/V остаётся Q4_0; сначала gather + dequantize только выбранных rows в ограниченный FP16 scratch, адаптированный furnace sparse attention с FP32 reductions/softmax. Использовать selected IDs непосредственно, без масочного frontend донора. До 2051 IDs требуют около 4 MiB K/V scratch на query; размер query tile выбирать по фиксированному workspace budget. Не распаковывать всю историю K/V в FP16. Прямой indexed Q4 attention сравнить с gather после корректного первого пути.

Сначала 1024/2048, затем 4096/8192 только при измеренном выигрыше и доступном workspace. Dense GEMM разрешено брать из rocBLAS после shape-specific проверки. Сравнить dequant+GEMM и quantized путь на настоящем распределении токенов по экспертам.

Сохранить полезное reuse resident expert weights в PP из mx prefill D2D; canonical/repacked layouts и padding должны соответствовать consumer-у, не делать слепой D2D repacked slot в canonical tensor. Добавить compute/copy overlap через события и bounded double buffering. Для каждой GPU измерить полезную работу и простои. После одночанковой корректности адаптировать event-ordered pipeline двух prefill-чанков между GPU из furnace/reinstinct. Если он не окупается, оставить более простой вариант. Учесть стоимость восстановления expert cache после заимствования VRAM под PP; две GPU не обещают 2× single-sequence TG.

**Готово:** 4K и 16K prompt без prefix reuse обрабатываются корректно с Q4_0 K/V; границы чанков не меняют causal смысл; есть PP, полное время запроса и peak VRAM по GPU. QSA workspace ограничен выбранными rows/query tiles, нет полноконтекстной FP16-копии K/V или ложного увеличения PP за счёт незаметного reuse.

## R5. Decode, cache и короткий multi-token MoE

**Действия.** Оптимизировать по профилю, а не по списку доступных инструкций. Подобрать CPU threads/affinity/first-touch и простой cache admission. Сначала baseline 112 slots/layer, затем разумный бюджет из свободной VRAM. Один owner на слой, никаких remote expert RPC.

Для 1/2/3 позиций объединять routes по expert ID на CPU и GPU, читать весовой tile один раз на несколько аккумуляторов. Адаптировать reinstinct multi-column matvec после группировки: несколько отдельных GEMV по assignments не дают этого reuse. Собственный гибридный scheduler должен загрузить эксперта один раз для всех позиций окна, перекрывать AVX2, DDR reads, DMA и GPU compute при измеренном выигрыше. Fuse gate/up при выигрыше. Проверить все-hit/all-miss/mixed и повторяющиеся IDs, переходы cache slot FREE → LOADING → READY → IN_USE. Не публиковать новый ID до окончания всех нужных копий; не перезаписывать slot до завершения readers.

Сравнить CPU miss и H2D+GPU miss на измеренных размерах с учётом reuse, repack и конкуренции за DDR3. Не загружать все misses на GPU по религиозному принципу. Минимизировать лишние round-trips и выделения памяти. Готовые DPP/sdot4 primitives уже используются с R1; новые inline AMDGCN изменения делать для найденного горячего места после disassembly и A/B. HIP graphs адаптировать позднее, если submission overhead действительно значим; positions/lengths/slot maps передавать через device descriptor, не замораживать host scalars и CPU handoff в whole-engine graph.

**Готово:** есть отдельные 1/2/3-token block measurements, cache accounting и сквозной non-MTP baseline. Микроускорение принимается в основной путь только после проверки полного запроса.

Порядок R4/R5 допускается менять после R3 по измеренному узкому месту. Их результаты нужны до окончательной оценки MTP2.

## R6. MTP2 с корректным состоянием

**Действия.** Загрузить настоящий MTP sidecar, проверить его tap, shift, shared embeddings/output и собственное history state. Его K/V хранить в Q4_0 с теми же правилами packing/rotation, что у target; Q8_0 весов sidecar не определяет формат KV. Сначала greedy verify, потом стохастическая схема с правильной correction. Не заменять trained MTP маленькой случайной draft-моделью.

Явно различать три позиции: выданные пользователю токены, уже потреблённые forward токены и pending sampled token. Bonus/replacement token может ещё не входить в KV/state. Уточнить этот контракт в коде до optimization, иначе ошибки на один токен маскируются правдоподобным текстом.

Первый вариант restore: checkpoint recurrent/conv/PLE/tail state + логические KV lengths, затем при необходимости replay принятого префикса. Вернуть число видимых pooled blocks и raw-хвост target/MTP; отвергнутые позиции Q4 K/V исключать по lengths и перезаписывать перед новым чтением. После проверки заменить дорогой replay краткими prefix states, сохраняемыми внутри 1–3-step recurrence. Никогда не копировать весь 128K KV на каждое speculative окно.

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

**Условная цель длинного контекста:** 64K около 300–600PP/23–34TG; около 128K около 250–500PP/20–30TG. Эти цели требуют устранения значительной части полноконтекстной работы QSA. Если измеренный наклон latency остаётся прежним, указать фактические более низкие результаты. Линейную модель по двум точкам не выдавать за прогноз архитектуры.

**Stretch после собственного удачного кванта:** около 500–850PP/35–50TG на 16K. Качество, CPU unpack и MTP acceptance могут сделать этот вариант невыгодным. 80–100TG не является требованием проекта.

Не ускорять один показатель за счёт намного худшего другого молча. Для каждого важного выбора сравнивать оба режима: большой новый prompt и продолжение существующей истории.

## Минимальный контур контроля

После небольшой правки: затронутый fixture и smoke. После performance-кандидата: парные прогоны на коротком и среднем запросах. После архитектурной вехи: несколько prompt families, 512 outputs и длинные границы. Подробный trace выключать в окончательном benchmark.

Хранить в results.jsonl: revision/dirty, вариант модели/KV, input/output token counts, prefix-cache состояние, expert-cache состояние, sampling, PP time, TG time, total time, accepted0/1/2 и краткие RAM/VRAM данные. В диагностических runs добавлять CPU/H2D/P2P bytes и stage times по необходимости. Это один файл, не новая observability-платформа.

Кандидат отклоняется при необъяснённой correctness-регрессии, подтверждённом ухудшении полного запроса или выигрыше только на игрушечной форме. Недостаточный выигрыш означает следующую гипотезу, а не создание универсального autotuner.

В STATE.md оставлять следующую физически выполнимую команду или изменение. Не заканчивать запись словами «продолжить оптимизацию» без указания конкретного участка.
