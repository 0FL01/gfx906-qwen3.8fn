# Текущее состояние

## Статус и текущая задача

R0–R3 закрыты; short/logical-wide1024/real4K16K correctness qualified. B8 attention/CLI2 bounded correctness qualified; полный R4–R8/400–600PP/30–40TG OPEN/unmet. NEXT trace-off B8 A/B/A против saved765a4 baseline.
Исходный HEAD среза `77f89c3412fff65634df8b45b08fab1b3da028a0`; GPU Source SNAPSHOT77f/dirtytrue NEVER retag closure commit. Candidate `/core/build/core-session` already-built775.
Job `1791083481832-775` terminal exit1/26m29s AFTER strictfullCXX20/HIP20gfx906Release/allwarnings build/CTest36/36 (803.76s), bothGPUattentioncomponentPASS и originalModelB8exeEXIT0/31records.
Failurecollector требовал2hostlogitowners приcpu0; actualhost_logits1017118720B, working_logits толькоcpu_workers>0. Parentfixed source-derivedlowerfloor1owner, НЕnumericaltolerance; regressionLOCAL29/29 (43.187s), actualLOCAL/remotecollectB8PASS.
Continuation `1791087203796-820` COMPLETEexit0/21m01s SAMEalready-built775binaries: fullCTest36/36 (808.07s)/correctedtests→B8collect→CLI2mixed/off32→defaultreset/batch/memory+strictcollect ALLPASS; cap131072/slots112 memory неoccupied128K/duplicateappend.
Canonical remoteROOT/results.jsonl41/2563568B EXACT3appends38→41 [r4_attention_prefill,proto2mixedr4_request,proto2offr4_request], old38byteprefix+parsedhistoryEXACT/source77fdirtytrue. Downloaded local journal/Git38 history and +3/-0 verified; correctness slice accepted. Acceptedlogs NEVERappendagain.
ParentLOCALdownloaded actualModel/component/CLI2raws/strictlocalcollect ALLPASS. Historical765/769 closures accepted; actual512/route-copy paired measurement details README.md/results/raws, B8 has NOpairedperformanceyet.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller checkout — исходник; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления; не запускать второй экземпляр вслепую.
Модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 bytes), источник пользователя https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 bytes); источник загрузки не установлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, canonical journal только ROOT `/home/radneon/gfx906-core/results.jsonl`.
Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5; gfx906 GEMM проверены. Runtime image без CMake.
Production baseline image `llama.cpp-gfx906:pp-stream-dcd685463d`; source `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`. Config `/home/radneon/llama/{docker-compose.yml,models.ini}`; наш baseline-контейнер остановлен.
Hardware:16 physical/32 logical CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60/driverVRAM17163091968B. Parent verified actualR0 HIP0BDF0000:05:00.0/HIP1BDF0000:08:00.0; RECON без изменения машины/стека не повторять.

## Рабочие команды

Последний strictfullbuild775 PASS: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=77f89c3412fff65634df8b45b08fab1b3da028a0 -e CORE_DIRTY=ON llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'`. Новая сборка требует actualsynchronizedrevision/dirty; explicitsh/source100644;820 неrebuild.
Fresh model gate: `docker run --rm --name core-prefill-attention-repro --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-attention-test -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf > /home/radneon/gfx906-core/runs/NEW-attention-model.jsonl`; freshstderr/core-attention-batch/manualmodelgate fullcontract README.md.
ONLYafterexeexit0: `python3 -B /home/radneon/gfx906-core/src/tools/record_prefill_attention.py --raw /home/radneon/gfx906-core/runs/NEW-attention-model.jsonl --results /home/radneon/gfx906-core/results.jsonl`; проверитьexitstatus/ONEfreshappend. AcceptedrawsNEVERappendagain.
CLI2fresh sameDockercontract/entrypointcore-session: `--capacity 64 --slots 1 --prefill-chunk 32 --generate 32 --ignore-eos --sample --seed 42 --temperature 1.0 --top-p .95 --top-k 20 --cpu-workers 1 --hybrid-mode mixed --gpu-miss-groups 2 --attention-query-tile 8 MODEL 248044 $(seq 100 130)`; OFFworkers0/disabled/tile1/freshlogs — README.md.
Requestcollect: `python3 -B /home/radneon/gfx906-core/src/tools/record_request.py --raw FRESH_REQUEST --results /home/radneon/gfx906-core/results.jsonl` ONLYafterexit0; optionalVRAMjoin directattachedDockerargv/0.1s driver samples НЕexactpeak — README.md.
Literalraw `candidate_unqualified`/`local_unqualified` labels preserved; parent boundedqualification неSource rewrite/allknobsproof. Nativeproto2explicitknobs+26hybrid/2route/6attention typedcounters actualLOCAL+remotecollectPASS.

## Последний подтверждённый результат

Attentioncomponent EXACT3JSONrecords source+twoGPU/НЕfooter; each26cases304queries1867776bit+1867776CPUvalues/maxabs1.1920928955078125e-7/maxboundratio.0003692344547586807/23device180hostrejects6stickyPASS.
ModelB8 ONEsameSession cap2088/slots1/max1024/tile8/cpu0: oldN1 all2088 reference; enabled1024/1024/8+32N1; occupied5N1→997/997/57+32N1. Teacher2056+continuation32, logicalboundary2047..2056 НЕGPUselectedID/visibilitytrace/independentHF/new4K16K.
2163completedcalls/6264rows/1555476480finite/1036984320fullvocabcompared/ZEROerrorsviolationsdiagnosticbits/4176argmaxdiagnosticmatches; frozen `.02+.002*abs(ref)`, bit/argmaxnotrequired; timingcorrectnessonly. CompletedattentionAPIcounts6180calls49284rows6168multi49272multirows12singletonmax8 НЕphysicalkernels.
2190memoryobservations/26serializedledgers/8atomicrejects/6toggles/preserved6073162240values; individualtoggles/неserializedobservations driverassertions-only, notadditionalrawledgers.
Actualprivate40338944B/GPU=fourbuffers40142336+reusedf(15)prefix196608 НЕextraallocation; selection82080separate, f(15)/f(17)actualbacking50331648B each alreadyworkspace/ownedledger. MinfreeGPU0/1:12671549440/12115804160B; НЕexactpeak/fullRAM/individualreleaseproof.
ActualCLI2 cap64slots1chunk32prompt[BOS248044,100..130]generate32ignoreEOSsampleSeed42primary1/.95/20: mixedcpu1/mixedquota2tile8 vsOFFcpu0/disabledtile1; BOTH32outputs63consumed31TG32RNG1PPcall/IDsidenticaldiagnostic.
Mixedshort1488/wide48/cpuGroups11255/gateup11255/down11255/middlecols11255/batches1488/pairedH2D57625600B/Q8D2H8103600B/returns115251200B/GPUhitgroups649/GPUmissgroups2976/admitted2976/evicted1440.
Shortdiagnostic TGmixed27789.104387ms vsOFF6521.876872ms НЕproperpairedperformance/speedgain/policythreshold; cpu0default НЕpromoted.
Lastqualifiedperformance769: B7654K PP25.68871522181/TG10.07616889732,16K PP22.60272136899/TG9.85092544783; modest2–2.7%PP/about1.9%request. FullR4/targetsOPEN; raw r4-indexed-a4-* and historicalprofile750 — README.md.
RawROOT/runs/r4-attention-77f-a-{build.log,component.jsonl,model.jsonl}; continuation r4-attention-77f-b-{gates.log,cli2-mixed.jsonl,cli2-off.jsonl,reset.jsonl,batch.jsonl,memory.jsonl}; source77fdirtytrue unchanged.

## Контракт и границы

Own Session: RAM experts/default112 GPU slots/layer, static24/24, Q4 KV/FP32 index/GDN/PLE persistent, default max_batch1/pinned40KiB. Accepted logical≤1024: canonical nonexpert/shared/expert microtiles≤8, routes once/full chunk/triplet once/group; GDN chronological slices≤128. Band16/two stages/copy_ready/consumer_done сохраняются; новые weights/precision/arithmetic/epsilon/gates не менялись.
SeparateSessionRouteStats last-callmax assignments одномуexpert/layer+cumulativegroups>128 since reset; publishonlysuccess/resetzero/invalid+executionfailurepreserve, failure требуетreset. Lastemittedtoken pending; greedy/trace diagnostic-only.
Batchprimitive ADAPT currentfurnace-derivedN1/privatequerydimension/exactoldQ4→halfRNE/sortedselectedID/64-keysplit/merge; mx/furnaceMIT/reinstinctApachepinsunchanged. Originalboundedowner/API/CLI/schemafixtures/no newdependency; tile1/OFF/originalAPIdefault unchanged.
QualifiedCPUlinear_GPUmiddle exactCPUgateup/down+canonicalGPUSiLU/Q8 onceCPU-bearinglayer; force_cpu НЕwholeCPU. WholeCPUlibm755/759 fullgatefailed, primitiveboundspassed — no tolerancewaiver/repeat; pipeline/lifetime detailsREADME.md.
FutureunlinkedSpec3/MtpModel3/newinspector/ignoredcheckpointpatch excludedfromB8CLI2slice; no trainedMTP/longAPI/converter qualification. Exactpeak/occupied128K/fullR4/R5policy/R6–R8/finalspeeds remainOPEN.

## Следующие действия

1. ONEcurrentbaseline `/core/build/core-session-baseline`1556208B SOURCEa4b55d84724ba15bbae7013d5a107b7671b7a409/dirtytrue qualified765, alreadycopied; НЕoverwrite beforepairedB8performance. Candidatealready775/source77fdirtytrue;775/820TERMINAL, неresume/relaunch.
2. Trace-off own A1baseline→Btile8→A2baseline ОБАchunk1024slots112/fresh/CPU0disabled/MTPoff/noPrefixReuse/warmnessunknown/primary1/.95/20/seed12345×512ignoreEOS; oneGPUload/noheavycompile. Baselineoriginaltile1/no newCLIflags; Bexplicitworkers0/disabled/tile8.
Fixed4KIDs `ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json`;16K fourconcat `ROOT/runs/r4-fullrequest-b522-16k.prompt-ids.json`, неretokenize/originaluser. Capacities4608/16896, freshlogs/fullDockercontractREADME.md.
Brequestargs: `--capacity 4608 --slots 112 --prefill-chunk 1024 --generate 512 --ignore-eos --sample --seed 12345 --temperature 1.0 --top-p .95 --top-k 20 --cpu-workers 0 --hybrid-mode disabled --attention-query-tile 8 MODEL exact4KIDs`;16K capacity16896/exactsavedIDs.
3. После B8 performance — CPUworkers/admission-policy measurements, fullR4/R5→trainedMTP/R7occupiedlongAPI/R8converter. No newB8speedclaim доpairedactual512; no duplicateappends.
Indexraw128 безH/inverse; GDN[V][v][k],h%16; PLEhash/convresetвместе; RoPE64j/j+32/absolute, sectionsARRAYINT32. НеfullBF16download/ROCmchange/wave32; QSAfork неHForacle; MCPtransferlocal_root толькоcheckout.
