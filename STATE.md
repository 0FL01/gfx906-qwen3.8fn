# Текущее состояние

## Статус и текущая задача

R0–R3 закрыты: own48 Session/teacher32/reset и capacity131072 owned ledger. Short/logical-wide1024 и real4K/16K long-correctness slices qualified; полный R4–R8 и целевые скорости открыты.
Latest correctness job765 COMPLETED exit0/34m46s: full strict CXX20/HIP20 gfx906 Release/allwarninggates/CTest35/35 (735.71s), обеGPU route-copy/paired-middle, full hybrid348records/defaultreset/batch/wide1024/cap131072 memory/strict actual collectors PASS. Source SNAPSHOTa4b55d84724ba15bbae7013d5a107b7671b7a409/dirtytrue, не future commit.
Latest exclusive TRACE-OFF paired job `1791073403122-769` COMPLETED exit0/57m33s, GPUidle. FrozenA `/core/build/core-session-baseline` SOURCEb522429c5933f493a5838317dc0bfc26ee4158aa/dirtytrue vs B765 `/core/build/core-session` SOURCEa4b55d84724ba15bbae7013d5a107b7671b7a409/dirtytrue; не parentHEAD6faf14ed96dc1b0d4353dc838323616a6f6c5313.
Qualified CPUlinear_GPUmiddle: exact CPU gate/up/down, canonical GPU SiLU/Q8 middle once/CPU-bearing layer; force_cpu labelCPU_LINEAR_GPU_middle, не wholeCPU. Hybrid11phases/326windows/528vocabRows131112960values/508intermediateRows624230400down+62423040FFN:finite/allzeroerror/violations/diagnosticbits; bitidentity diagnostic-only.
Correctness765 ACCEPTED; measurement769 ACCEPTED/CLOSED: LOCALcanonical38/2497537B+6raws+fixturemetadata downloaded/all6collectPASS/exactsource-footer-counts-timings-rates matchjournal. GitHEAD31 byteprefixANDparsed31 preserved, ACTUAL+7/-0/diffcheckPASS; EXACT6r4_request+1r4_indexed_route_copy_ab. NEVERappendagain/sourceb522+a4dirtytrue preserved.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller checkout — исходник; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления; не запускать второй экземпляр вслепую.
Модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 bytes), источник пользователя https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 bytes); источник загрузки не установлен.
Baseline image `llama.cpp-gfx906:pp-stream-dcd685463d`; source `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`. Config `/home/radneon/llama/{docker-compose.yml,models.ini}`; наш baseline-контейнер остановлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, canonical journal только ROOT `/home/radneon/gfx906-core/results.jsonl`.
Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5; gfx906 GEMM проверены. Runtime image без CMake.
Hardware:16 physical/32 logical CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60/driverVRAM17163091968B. Parent verified actualR0 HIP0BDF0000:05:00.0/HIP1BDF0000:08:00.0; RECON без изменения машины/стека не повторять.

## Рабочие команды

Последний FULL build765 PASS: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=a4b55d84724ba15bbae7013d5a107b7671b7a409 -e CORE_DIRTY=ON llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'`. Новая сборка требует actual revision/dirty; explicit `sh`, source script100644.
Hybrid fresh: `docker run --rm --name core-hybrid-middle-repro --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-session-hybrid-test -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf > /home/radneon/gfx906-core/runs/NEW-session-hybrid-middle.jsonl`; fresh stderr/fullcontract и long4096|16384 — README.md.
Только после executable exit0: `python3 -B /home/radneon/gfx906-core/src/tools/record_hybrid.py --raw /home/radneon/gfx906-core/runs/NEW-session-hybrid-middle.jsonl --results /home/radneon/gfx906-core/results.jsonl`; проверить collector exitstatus/ONEfresh append. Accepted raw не append повторно.
Request: тот же Docker/device/mount контракт, entrypoint `/core/build/core-session`; `--capacity 4608 --slots 112 --prefill-chunk 128|1024 --generate 512 --ignore-eos --sample --seed 12345 --temperature 1.0 --top-p 0.95 --top-k 20 MODEL exact4KIDs`;128|1024 означает выбрать одно значение. Fresh16K reproduction capacity16896/exact saved16KIDs; fixed-filename commands — README.md, recorded source config authoritative.
Request collect: `python3 -B /home/radneon/gfx906-core/src/tools/record_request.py --raw FRESH_REQUEST --results /home/radneon/gfx906-core/results.jsonl` после exit0; optional `--vram-log FRESH_OBSERVER`. Stdlib observer0.1s; request join требует direct attached Docker argv, не shell shape; full repro README.md.
Oracle teacher: `sh tools/build-oracle.sh`, production libs + `--all-layers --hf-gdn-l2-control --hf-qsa-f32-control --warm-cache 12 --cache-inserts 10` diagnostic-only, не bitwise HF/unchanged-math/performance baseline.
Currentcandidate explicitCPU/attention CLI/requestprotocol2/collectors locallytested: request79/5.055s+attentioncollector28/43.004s PASS, НЕHIP/GPUattention/ModelB8. Cleanupsourcechecks неruntimeproof; runtimequerybatchstats НЕkernelcounts; attentiontile1/OFF/originalAPIdefault unchanged, CPUlinear_GPUmiddle retains765/full35qualification.

## Последний подтверждённый результат

769 каждая длина: fresh A1baseline→B765→A2baseline, ОБА chunk1024/slots112/noPrefixReuse/warmnessunknown/MTPoff/protocol1/primary1/.95/20/seed12345/ignoreEOS.6runs×512actualoutputs=3072;TG511/RNG512/consumed4607|16895/cap4608|16896;512IDs same withinlength diagnostic-only.
4K exact R0 generated archive/code IDs `ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json`;16K четыре конкатенации этих IDs в `runs/r4-fullrequest-b522-16k.prompt-ids.json`, не retokenized/original-user/baseline16K. Warmness unknown.
B4K PP25.68871522181/TG10.07616889732/total210162.620717ms;B16K PP22.60272136899/TG9.85092544783/total776743.230123ms. Mean-A/B PP1.02718318141×/1.01999903571×;request1.01937816968×/1.01916904667×. B beatsBOTHas PP/request eachlength: modest2–2.7%PP/about1.9%request, неlargewin/TGthreshold;400–600PP/30–40TG НЕmet.
Bfullrequest uploads4K305853440000B/16K890465689600B sameoldB; неPP-only/residual-dominanceproof. Initialpairedappend missingmodel failedBEFOREwrite после6validappends; parentre-collected37existing excepttimestamps thenaddedONLYonecorrectpaired, no duplicate/historyrewrite.
769raws ROOT/runs/r4-indexed-a4-{4k,16k}-{a1,b,a2}.jsonl/.err, sameprefix fixture-source.json/series.log. Detailedsixrun timings — results/raws; summary/freshDockerrepro README.md. Loadseparate/sameownbinarycomparison, неexternalMTP2/originaluser.
Historical746 chunk128/1024/128 measurement accepted21→30/+9/-0:PP1.824×/1.758×;request1.640×/1.710×. Separate0.1s B_VRAM globaldriversamples notexactpeak/HIPowned; raw runs/r4-fullrequest-b522-* +observer/fixture/series/build, details README/results.
Rocprofv3 job `1791052186288-750` COMPLETE exit0/2m14s, FROZENb522/dirtytrue; prefix1024R0text/32outputs/TG31 nativePASS PP43116.686375/TG4488.729555/load70708.105268ms. Raws `runs/r4-profile-b522-1024.jsonl`/`.err`, trace/statsCSV same-name directory/nested summary — README.md; full554MB trace remote/not downloaded, model not copied.
Profile PP+TG overlapping sums, НЕ total latency/НЕ PP-only: HIP2760730/36.11848s, hipMemcpyAsync1062432/30.0312s, launch839656/5.11047s; GPU1831186/28.37233s. MEMDMA72646/5.562979s distinct from D2D copyBuffer; sums/bytes не доказывают RAM/expert-DMA/memcpy residual bottleneck.
GPU: fattn_dec_chunk12660/9.362744s33%, Q4_0matrix32<2,8,2>193474/5.847864s20.61%, copyBuffer989831/3.537115s12.47%, Q6headmatrix6<8,2>896/.712313s2.51% (GPU-duration sum shares). Head не dominant; headskip не major-win priority; profiling diagnostic-only, не performance qualification.
Job765 defaultwide cap1056/slots1/max1024:786677760 comparisons/zeroerror, не4K16Krerun. Memory cap131072/slots112:owned9515698008/10007105368B,free7166787584/6686539776B,afterdestroy16968876032B eachPASS; capacity/release, неoccupied128K. Historical real4K/16K correctness source2e/dirtytrue — README.md.
Brief diagnostic: wholeCPU libm oracle fullgate FAILED755 offset1; saved759 actualL46/E297 identicalgate/up,24 middleULP-scale diffs/oneblock11 rawsumhalf13734vs13733(codes/d same), primitiveboundsPASS/fullgateFAIL. Qualified path keeps canonicalGPUmiddle; ABI/GPUmath/weights/precision/epsilon/gates unchanged/no tolerancewaiver.

## Контракт и границы

Separate `SessionRouteStats`/`route_stats()`: last-call max assignments к одному expert/layer и cumulative call/layer/expert groups>128 since reset. Publish только success; constructor/successful reset zero; invalid arguments/execution failure preserve. Historical `SessionStats`/protocols unchanged; execution failure требует reset.
Own Session: RAM experts/default112 GPU slots/layer, static24/24, Q4 KV/FP32 index/GDN/PLE persistent, default max_batch1/pinned40KiB. Accepted logical≤1024: canonical nonexpert/shared/expert microtiles≤8, routes once/full chunk/triplet once/group; GDN chronological slices≤128. Band16/two stages/copy_ready/consumer_done сохраняются; новые weights/precision/arithmetic/epsilon/gates не менялись.
Canonical attention ordering/widened exp — accuracy changes; wide MMQ wins component-only. Actual512 own-chunk PP/request comparison не external baseline/fullR4 qualification. Последний emitted token pending; greedy/trace-range diagnostic-only.
Qualified partialpipeline:persistent1..15workers/fixed30jobs/gate_up→down, pinnedframe reuse aftermiddle_ready, dedicatedcopyStreamflag, CPUrankH2D/eventbeforecomputeWait/orderedfold+shared; cachedreaders/pendingIDs/errorreset syntheticqualified, не inflighttiming. Newmiddle4buffers252004B/GPU+pinned43200B; hostInputProbe855648 INCLUDEDtotal33295968B, неRSS/stacks/all-old-buffer-release.175rejects source-derived/notserialized; missingstats/routes/inputpayload/ownedledgers не observations.

## Следующие действия

1. NEXT продвинуть qualifieda4binary как ONEcurrentbaseline, затем fullstrict36candidate build/CTest; actualrevision/dirty/explicitsh. Old765 full35 PASS неqualification новых36gates; frozen769 artifactmacros неretag.
2. Actual bothGPUattentionbatch+ModelB8Session full-logits/visibility2047–2056/privatebuffer/defaultregressions, затемCPUworkers/admissionpolicy. CLI/protocol2localPASS неHIPproof; futureSpec3 excluded. Hybridwin/threshold/FULLR4R5/MTP/exactpeak/occupied128K/targets OPEN/unmet; defaultcpu0.
Index raw128 без H/inverse; GDN [V][v][k], h%16; PLE hash/conv reset вместе; RoPE64 j/j+32/absolute position, sections ARRAY INT32.

## Не повторять без причины

Не скачивать full BF16 на заполненный SSD/не менять рабочий ROCm/не переносить wave32 assumptions. QSA fork не HF oracle без boundary fixture. Router ждать /models.current.status=loaded/model=current, не только /health. MCP transfer local_root ограничен checkout; запрет не обходить.
