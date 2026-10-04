# Текущее состояние

## Статус и текущая задача

R0–R3 закрыты; B8 attention/CLI2 correctness accepted/pushed `3cc594d0783974aec4cf6f4657d8e0c6864139d2`. Полный R4–R8/400–600PP/30–40TG OPEN/unmet.
Trace-off B8 A/B/A job `1791090828619-823` COMPLETEexit0/52m44s/boundedPERFORMANCE-MEASUREMENT ACCEPTED/CLOSED. Next parent5filesdocs+journal commit/push→qualified77fattentionONEbaseline→newactualstrictR6prereqgates/DEFAULTregressions.
Historical823 A saved765baseline `/core/build/core-session-baseline`1556208B/source `a4b55d84724ba15bbae7013d5a107b7671b7a409`dirtytrue; B `/core/build/core-session`already775/source `77f89c3412fff65634df8b45b08fab1b3da028a0`dirtytrue. No rebuild/heavycompile/otherGPU; NEVERretag futurecommit.
Canonical remoteROOT/results.jsonl48/2897341B EXACT7appends41→48 [6r4_request+1r4_attention_batched_query_ab]; old41byteprefix+parsedhistory verifiedREMOTE/all6strictRemoteCollectPASS. Accepted/historical raws NEVERduplicateappend.
Parent downloaded6raws/fixturemetadata/canonical48journal; all6actualLOCALcollectPASS/Source+complete+rates EXACTmatchROOTrows/withinlength512IDs same diagnostic-only. OldGitHEAD41 EXACTbyteprefix+parsedhistory/actualgitnumstat+7/-0/diff--checkPASS. Pairedsource77fdirtytrue preserved.
HISTORICAL correctness775 strictbuild/CTest36/36 (803.76s)+bothGPUcomponent+modelEXIT0; collector1hostlogitfloor repair НЕtolerance. Historical820 SAME775 fullCTest36/36 (808.07s)/CLI2/defaultreset,batch,memory/collectPASS. НЕnewR6gates; correctness41/Git38prefix/+3/-0 verified before3cc594d.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller checkout — исходник; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления; не запускать второй экземпляр вслепую.
Модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 bytes), источник пользователя https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 bytes); источник загрузки не установлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, canonical journal только ROOT `/home/radneon/gfx906-core/results.jsonl`.
Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5; gfx906 GEMM проверены. Runtime image без CMake.
Production baseline image `llama.cpp-gfx906:pp-stream-dcd685463d`; source `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`. Config `/home/radneon/llama/{docker-compose.yml,models.ini}`; наш baseline-контейнер остановлен.
Hardware:16 physical/32 logical CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60/driverVRAM17163091968B. Parent verified actualR0 HIP0BDF0000:05:00.0/HIP1BDF0000:08:00.0; RECON без изменения машины/стека не повторять.

## Рабочие команды

Последний strictfullbuild775 PASS: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=77f89c3412fff65634df8b45b08fab1b3da028a0 -e CORE_DIRTY=ON llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'`. Новыйsnapshotbuild требует actualsynchronizedrevision/dirty, не77fпоинерции; explicitsh/source100644;820/823 неrebuild.
Fresh model gate: `docker run --rm --name core-prefill-attention-repro --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-attention-test -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf > /home/radneon/gfx906-core/runs/NEW-attention-model.jsonl`; freshstderr/core-attention-batch/manualmodelgate fullcontract README.md.
ONLYafterexeexit0: `python3 -B /home/radneon/gfx906-core/src/tools/record_prefill_attention.py --raw /home/radneon/gfx906-core/runs/NEW-attention-model.jsonl --results /home/radneon/gfx906-core/results.jsonl`; проверитьexitstatus/ONEfreshappend. AcceptedrawsNEVERappendagain.
CLI2fresh sameDockercontract/entrypointcore-session: `--capacity 64 --slots 1 --prefill-chunk 32 --generate 32 --ignore-eos --sample --seed 42 --temperature 1.0 --top-p .95 --top-k 20 --cpu-workers 1 --hybrid-mode mixed --gpu-miss-groups 2 --attention-query-tile 8 MODEL 248044 $(seq 100 130)`; OFFworkers0/disabled/tile1/freshlogs — README.md.
Requestcollect: `python3 -B /home/radneon/gfx906-core/src/tools/record_request.py --raw FRESH_REQUEST --results /home/radneon/gfx906-core/results.jsonl` ONLYafterexit0; optionalVRAMjoin directattachedDockerargv/0.1s driver samples НЕexactpeak — README.md.
Literalraw `candidate_unqualified`/`local_unqualified` labels preserved; parent boundedqualification неSource rewrite/allknobsproof. Nativeproto2explicitknobs+26hybrid/2route/6attention typedcounters actualLOCAL+remotecollectPASS.

## Последний подтверждённый результат

823 ОБАchunk1024/slots112/fresh/noPrefixReuse/cachewarmunknown/CPU0disabled/MTPoff/primary1/.95/20seed12345ignoreEOS512; AoriginalN1tile1/Bexplicitquerytile8protocol2. ExactR0archive/code4K IDs/fourconcat16K, неoriginaluser/retokenized/externalMTPbaseline.
Все6requests512actualoutputs/3072total, eachTG511/RNG512, consumed4607|16895/cap4608|16896; three512-IDarrays withinlength equal diagnostic-only. Rawlabels candidate_unqualified/local_unqualified unchanged.
4K A1/B/A2 PP160152.153592/115105.478965/160544.563734ms; request210982.242626/165876.889925/210718.406950ms. B PP35.5847526706/TG10.0650106725; meanA/B PP1.39305583109347×/request1.2711253802945932×.
16K A1/B/A2 PP724478.209838/509826.542585/723419.610449ms; request777620.365975/562117.081524/775858.095619ms. B PP32.1364201968/TG9.7725973247; meanA/B PP1.419990623620387×/request1.3818104027209437×.
B fasterBOTHAs PP/request EACHlength; TG~10/targetsUNMET. Completedwall PPwithoutoutputsampling/TGoutputs−1/loadseparate; no newGPUevents/profile/VRAM/exactpeak/HF/marginalRNG/CPUdispatchclaims.
RawsROOT/runs/r4-attention-77f-ab-{4k,16k}-{a1,b,a2}.jsonl/.err, r4-attention-77f-ab-series.log/-fixture-source.json. Sixexacttimings/rates/freshrepro/argv-vs-Source — README.md; canonicalROOTonly.
CorrectnessB8 cap2088/slots1/max1024/tile8/cpu0: teacher2056+32continuation oldN1/enabled1024/occupied5→997; 1036984320fullvocabcomparisons ZEROerrors/frozen `.02+.002*abs(ref)`/bitdiagnostic. Logical2047..2056 НЕGPUvisibilitytrace/HForacle.
Private40338944B/GPU includes4buffers40142336+reusedf(15)196608; selection82080separate/f15,f17backing50331648each alreadyledgered. Component26cases/304queries perGPU PASS; APIcounts НЕphysicalkernels.
CLI2mixed/off32 actualLOCAL+remotePASS; TG27789.104387vs6521.876872ms diagnostic-only/НЕpolicywin/defaultpromotion. Historical769 modest2–2.7%PP/~1.9%request; correctness/model/CLI2 rawdetails README.md.

## Контракт и границы

Own Session: RAM experts/default112 GPU slots/layer, static24/24, Q4 KV/FP32 index/GDN/PLE persistent, default max_batch1/pinned40KiB. Accepted logical≤1024: canonical nonexpert/shared/expert microtiles≤8, routes once/full chunk/triplet once/group; GDN chronological slices≤128. Band16/two stages/copy_ready/consumer_done сохраняются; новые weights/precision/arithmetic/epsilon/gates не менялись.
SeparateSessionRouteStats last-callmax assignments одномуexpert/layer+cumulativegroups>128 since reset; publishonlysuccess/resetzero/invalid+executionfailurepreserve, failure требуетreset. Lastemittedtoken pending; greedy/trace diagnostic-only.
Batchprimitive ADAPT currentfurnace-derivedN1/privatequerydimension/exactoldQ4→halfRNE/sortedselectedID/64-keysplit/merge; mx/furnaceMIT/reinstinctApachepinsunchanged. Originalboundedowner/API/CLI/schemafixtures/no newdependency; tile1/OFF/originalAPIdefault unchanged.
QualifiedCPUlinear_GPUmiddle exactCPUgateup/down+canonicalGPUSiLU/Q8 onceCPU-bearinglayer; force_cpu НЕwholeCPU. WholeCPUlibm755/759 fullgatefailed, primitiveboundspassed — no tolerancewaiver/repeat; pipeline/lifetime detailsREADME.md.
LOCALONLY Sessioncheckpoint/tap/restore+newfixture, MtpModelguards/Specmath/denseQ4attention НЕmirrored/compiled/promoted, excludedfrommeasurement-onlyslice. No trainedMTP/longAPI/converterqualification; exactpeak/occupied128K/fullR4/R5policy/R6–R8/finalspeedsOPEN.

## Следующие действия

1. Parent next normalcommit/push ONLYREADME.md/PLAN.md/STATE.md/third_party/NOTICE.md/results.jsonl; excludes ALLSessioncheckpoint/MtpModel/Spec/dense/inspectorfuturecode. 823 ACCEPTED/CLOSED — no duplicateappend/newHEADclaim/Source rewrite.
2. AFTERparentcommit/push promotequalified775attention/source77fdirtytrue/explicittile8 ONEbaseline, then actualHIP Sessionverify_window/restore_prefix/target_tap fixture vs consumed-input sequential state/prefix0..N/repeatedreject/EOS/PLE/QSA-tail/error+reset; denseQ4component BOTHGPU/actualdescriptor gates.
Mtpguards:32exactroles, BF16indexQ/KONLY, fulltop-k10budget/blk.48compression0dense, everytypedtokenizerfield equal; actualdonorreader convention. Descriptor/puremath НЕtrainedMTPforwardproof.
3. Mirror/build NEWSNAPSHOTactualrevisiondirty; actualfixtureexit0+collect, fullrelevantCTest+DEFAULTreset/batch/memory; keepR6candidate outside823measurement-onlycommit. Then R5workers/admissionpolicy+remainingPPweighttraffic measurements.
Freshperformance commandsfixedNEWfilenames/actualargv-vs-Source README.md;4K ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json,16K ROOT/runs/r4-fullrequest-b522-16k.prompt-ids.json; cap4608/16896, neverretokenize/reuseacceptedrecords.
Indexraw128 безH/inverse; GDN[V][v][k],h%16; PLEhash/convresetвместе; RoPE64j/j+32/absolute, sectionsARRAYINT32. НеfullBF16download/ROCmchange/wave32; QSAfork неHForacle; MCPtransferlocal_root толькоcheckout.
