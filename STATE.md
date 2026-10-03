# Текущее состояние

## Статус и текущая задача

R0–R3 закрыты: own48 Session/teacher32/reset и capacity131072 owned ledger. Short/logical-wide1024 и real4K/16K long-correctness slices qualified; полный R4–R8 и целевые скорости открыты.
Текущая задача — strict target/bothGPU route-copy + current CPUshadow/full-model gates, затем trace-off paired validation. Proven source seam: per-assignment Q8 gather/down scatter `2*(1024*10*48)=983040` small D2D calls; candidate ONE80*N-byte route DTO/layer + indexed micro≤8 preserves bits/fold, ещё unqualified.
Performance job `1791040898917-746` COMPLETED exit0/1h40m26s: strict targeted core-session build, request-results/vram-observer CTest2/2 (5.18s; suites49/23) PASS,8 fresh actual512 requests/collectors PASS включая2 observer joins. Source FROZEN `b522429c5933f493a5838317dc0bfc26ee4158aa` dirtytrue, не future commit.
Previous full job `1791030847464-742` закрыт: strict CXX20/HIP20 Release gfx906/warnings, CTest32/32 (605.76s), actualQ8/long16K→4K/reset/batch/default-memory PASS; source2e9848d/dirtytrue. Full33/33 ещё НЕ подтверждён.
Все8 raws downloaded/localcollect+2VRAMjoins PASS; local canonical30 rows/2150750B, EXACT9 appends21→30:8*r4_request+1*r4_prefill_ab. Old21 byte prefix/parsedhistory compared Git HEAD EXACT, numstat+9/-0 verified. Actual512 measurement-only slice accepted/closed; local49+23/readback/status/diff/log/whitespace gates PASS; accepted raws не append повторно. Historical long19→21/+2/-0 verified.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller checkout — исходник; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления; не запускать второй экземпляр вслепую.
Модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 bytes), источник пользователя https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 bytes); источник загрузки не установлен.
Baseline image `llama.cpp-gfx906:pp-stream-dcd685463d`; source `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`. Config `/home/radneon/llama/{docker-compose.yml,models.ini}`; наш baseline-контейнер остановлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, canonical journal только ROOT `/home/radneon/gfx906-core/results.jsonl`.
Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5; gfx906 GEMM проверены. Runtime image без CMake.
Hardware:16 physical/32 logical CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60/driverVRAM17163091968B. Parent verified actualR0 HIP0BDF0000:05:00.0/HIP1BDF0000:08:00.0; RECON без изменения машины/стека не повторять.

## Рабочие команды

Последний completed FULL build(job742): `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=2e9848d43cf9f908dc8280f81e111c0cde86f01c -e CORE_DIRTY=ON llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'`. Для новой сборки actual revision/dirty; explicit `sh` нужен: source build.sh100644. Job746 был targeted build, не full33 qualification.
Long: `docker run --rm --name core-prefill-long --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-long-test -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf 16384 > /home/radneon/gfx906-core/runs/FRESH-prefill-long-16384.jsonl`; ROWS ровно4096 или16384.
Только после exit0: `python3 -B /home/radneon/gfx906-core/src/tools/record_prefill_long.py --raw /home/radneon/gfx906-core/runs/FRESH-prefill-long-16384.jsonl --results /home/radneon/gfx906-core/results.jsonl`; проверить collector exit status, append один validated record. Existing actual raw не append повторно.
Request: тот же Docker/device/mount контракт, entrypoint `/core/build/core-session`; `--capacity 4608 --slots 112 --prefill-chunk 128|1024 --generate 512 --ignore-eos --sample --seed 12345 --temperature 1.0 --top-p 0.95 --top-k 20 MODEL exact4KIDs`;128|1024 означает выбрать одно значение. Fresh16K reproduction capacity16896/exact saved16KIDs; fixed-filename commands — README.md, recorded source config authoritative.
Request collect: `python3 -B /home/radneon/gfx906-core/src/tools/record_request.py --raw FRESH_REQUEST --results /home/radneon/gfx906-core/results.jsonl` после exit0; optional `--vram-log FRESH_OBSERVER`. Stdlib observer0.1s; request join требует direct attached Docker argv, не shell shape; full repro README.md.
Oracle teacher: `sh tools/build-oracle.sh`, production libs + `--all-layers --hf-gdn-l2-control --hf-qsa-f32-control --warm-cache 12 --cache-inserts 10` diagnostic-only, не bitwise HF/unchanged-math/performance baseline.
Fresh logs/trace paths в existing runs/; одна GPU-нагрузка, final measurement без тяжёлой сборки/tracing.

## Последний подтверждённый результат

Каждая длина: fresh A1chunk128→B1024→A2chunk128 unobserved/tracefree, затем отдельный B_VRAM; protocol1/primary1/.95/20/seed12345/ignoreEOS/slots112/no prefixreuse/MTPoff.512 emitted/TGforwards511/RNG512/consumed4607|16895;512-ID arrays exact same across4 variants per length, diagnostic-only.
4K exact R0 generated archive/code IDs `ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json`;16K четыре конкатенации этих IDs в `runs/r4-fullrequest-b522-16k.prompt-ids.json`, не retokenized/original-user/baseline16K. Warmness unknown.
B4K PP24.8487818621/TG10.2822355241,PP164837.054095/TG49697.363847/total214535.87483ms; B16K PP22.2115183863/TG9.9343829824,PP737635.298724/TG51437.51765/total789074.279015ms. Mean-A/B PP1.82406225345×/1.75816669497×;request1.63977185394×/1.71047221484×. Load separate; same-own-config ONLY, no externalMTP2 speedup. Targets400–600PP/30–40TG НЕ met.
Full-request uploads4K eachA1139677491200B/B305853440000B;16K eachA4264348979200B/B890465689600B. Per-phase bytes not emitted: не PP-only traffic/не residual-bottleneck proof.
B_VRAM0.1s scope `sampled_global_driver_VRAM_not_exact_instantaneous_peak`:4K samples2892/maxused12004397056/12487217152B/minfree5158694912/4675874816B;16K samples8643/maxused12079910912/12562612224B/minfree5083181056/4600479744B. Global driver, не HIP-owned/exact/per-phase peaks.
Raw `runs/r4-fullrequest-b522-{4k,16k}-{a1,b,a2,b-vram}.jsonl`, observers `*-b-vram-vram.jsonl`, same-prefix fixture-source.json/16k.prompt-ids.json/build.log/series.log; точные rates/scopes/repro — README.md.
Rocprofv3 job `1791052186288-750` COMPLETE exit0/2m14s, FROZENb522/dirtytrue; prefix1024R0text/32outputs/TG31 nativePASS PP43116.686375/TG4488.729555/load70708.105268ms. Raws `runs/r4-profile-b522-1024.jsonl`/`.err`, trace/statsCSV same-name directory/nested summary — README.md; full554MB trace remote/not downloaded, model not copied.
Profile PP+TG overlapping sums, НЕ total latency/НЕ PP-only: HIP2760730/36.11848s, hipMemcpyAsync1062432/30.0312s, launch839656/5.11047s; GPU1831186/28.37233s. MEMDMA72646/5.562979s distinct from D2D copyBuffer; sums/bytes не доказывают RAM/expert-DMA/memcpy residual bottleneck.
GPU: fattn_dec_chunk12660/9.362744s33%, Q4_0matrix32<2,8,2>193474/5.847864s20.61%, copyBuffer989831/3.537115s12.47%, Q6headmatrix6<8,2>896/.712313s2.51% (GPU-duration sum shares). Head не dominant; headskip не major-win priority; profiling diagnostic-only, не performance qualification.
Historical long raws source2e/dirtytrue: zero violations/maxabs/maxboundratio/diagnostic bits при frozen `.02+.002*abs(ref)`/allfinite; N1/canonical1024/occupied5→997 +32 continuations, causal2047–2056/mod4. Same-Session self-reference, не independentHF/selected-IDtrace/physical128-tile/individual-capacity/fullRAM/owned-release/peak/performance proof.
Q8 repair(originalFP32 d>0→storedhalf0 keeps codes/rawsum) — validation correctness, не optimization. Originald0/unsafe±128/nonfinite/halfInfNaN reject; logical32 halves isolated. Quant791975/214rejects и actual192blocks/6144F32 обеGPU PASS; weights/precision/epsilon/gates unchanged. Historical logs — README.md.

## Контракт и границы

Separate `SessionRouteStats`/`route_stats()`: last-call max assignments к одному expert/layer и cumulative call/layer/expert groups>128 since reset. Publish только success; constructor/successful reset zero; invalid arguments/execution failure preserve. Historical `SessionStats`/protocols unchanged; execution failure требует reset.
Own Session: RAM experts/default112 GPU slots/layer, static24/24, Q4 KV/FP32 index/GDN/PLE persistent, default max_batch1/pinned40KiB. Accepted logical≤1024: canonical nonexpert/shared/expert microtiles≤8, routes once/full chunk/triplet once/group; GDN chronological slices≤128. Band16/two stages/copy_ready/consumer_done сохраняются; новые weights/precision/arithmetic/epsilon/gates не менялись.
Canonical attention ordering/widened exp — accuracy changes; wide MMQ wins component-only. Actual512 own-chunk PP/request comparison не external baseline/fullR4 qualification. Последний emitted token pending; greedy/trace-range diagnostic-only.
Parent-local R5 Session/new route kernels/fixture и CPUexpert/shadow/hybrid patch experimental/defaultOFF/prepared/unmirrored/unqualified; нет actual HIP/model/speed результата. R6 pure-math speculative helpers prepared, не trainedMTP/restore/performance qualification.

## Следующие действия

1. Sync/strict target build candidate; actual route-copy fixture обеGPU (all bits/fold), current CPUshadow/hybrid экспертные суммы/cold-hit-miss/slot-lifetime и full-model/full33 gates. Existing targeted gates не full33 proof.
2. После candidate gates trace-off paired full-request validation; actual HIP/model/speed qualification candidate ещё OPEN. Accepted measurements retain compiledb522/dirtytrue/historical2e/dirtytrue, не future candidate revision. Exactpeak/fullR4/R5–R8/occupied128K/MTP2/400–600PP/30–40TG OPEN.
Index raw128 без H/inverse; GDN [V][v][k], h%16; PLE hash/conv reset вместе; RoPE64 j/j+32/absolute position, sections ARRAY INT32.

## Не повторять без причины

Не скачивать full BF16 на заполненный SSD/не менять рабочий ROCm/не переносить wave32 assumptions. QSA fork не HF oracle без boundary fixture. Router ждать /models.current.status=loaded/model=current, не только /health. MCP transfer local_root ограничен checkout; запрет не обходить.
