# Текущее состояние

## Статус и текущая задача

R0–R3 закрыты; B8 attention/CLI2 correctness accepted/pushed `3cc594d0783974aec4cf6f4657d8e0c6864139d2`. Полный R4–R8/400–600PP/30–40TG OPEN/unmet.
Trace-off B8 A/B/A823 ACCEPTED/CLOSED/pushed `ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2`; B8artifacts775/820/823B Source77fdirtytrue не retag.
BoundedR6prerequisite ACCEPTED/CLOSED correctnessONLY;831 DONE/nativeEXIT0/46m12s, compiled `ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2`/dirtytrue. Strict CXX20/HIP20 Release gfx906/allwarnings/ffp-contractoff, CTest39/39 actual1145.49s и actual/defaultgates PASS. R6_complete_claim=false/НЕtrainedMTP.
MCP831 state_lost/background_channel_closed_without_exit_status после14m15; READONLYDockerdaemon originaldieevent recoveredexit0/execDuration2772s, НЕrepeat. ID35ed6fbda1d6c386cb1d84819addcaf4650460044619ba658867de52e73a0644/namecore-r6-prerequisites-check/time1791106362/timeNano1791106362809343377; ignoredruns/r6-prerequisites-ba-a-exit-proof.json719B downloaded.
CanonicalROOT/results.jsonl50/2967969B EXACT2appends48→50 [`r6_target_restore`,`r6_mtp_prerequisites`]; remoteold48byteprefix+parsedhistory EXACT, parentGitHEADold48byteprefix ANDparsedhistory EXACTagainstlocal50/actualgitnumstat+2/-0/diffcheckPASS. Acceptedraws NEVERduplicateappend/retagSource.
All3actualraws+exitproof+canonical downloaded; parentREADONLYactualLOCALrestore473730Bcollect PASSagain/ZEROerrors644390400compares26419200tap; C++descriptor36+dense4 Source/footer/inventory/device EXACTmatchesROOTnewrecord. Immediateparent normalcommit/push acceptedprerequisitecode/docs/journal, затем trainedforwardteacherreference.
ONEbaseline PROMOTED BEFORE831: `/core/build/core-session-baseline`1750328B/qualified775/source `77f89c3412fff65634df8b45b08fab1b3da028a0`dirtytrue; futureA explicitlytile8/CPU0disabled. Currentcore-session build831/sourceba446f9dirtytrue НЕperformancebaseline. Historical823A sourcea4/1556208B не подменять77f.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller checkout — исходник; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления; не запускать второй экземпляр вслепую.
Модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 bytes), источник пользователя https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 bytes); источник загрузки не установлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, canonical journal только ROOT `/home/radneon/gfx906-core/results.jsonl`.
Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5; gfx906 GEMM проверены. Runtime image без CMake.
Production baseline image `llama.cpp-gfx906:pp-stream-dcd685463d`; source `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`. Config `/home/radneon/llama/{docker-compose.yml,models.ini}`; наш baseline-контейнер остановлен.
Hardware:16 physical/32 logical CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60/driverVRAM17163091968B. Parent verified actualR0 HIP0BDF0000:05:00.0/HIP1BDF0000:08:00.0; RECON без изменения машины/стека не повторять.

## Рабочие команды

Последний strictfullbuild831 PASS: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=ba446f9266ab2dc4c0aa4e73d8930a5534c4b2a2 -e CORE_DIRTY=ON llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'`. Freshbuild требует ACTUALsynchronizedrevision/dirty; explicitsh/source100644. Baseline не rebuild во время measurements.
Fresh target gate: sameDockerdevice/mountcontract README.md, entrypoint `/core/build/core-session-restore-test`, `/models/qwen38-keep1-Q4_0.gguf --wide-1024`, freshROOT/runs/NEW-target-restore.jsonl/.err; sequential/nootherGPU.
ONLYafterexeexit0: `python3 -B /home/radneon/gfx906-core/src/tools/record_restore.py --raw /home/radneon/gfx906-core/runs/NEW-target-restore.jsonl --results /home/radneon/gfx906-core/results.jsonl`; strictcollect/ONEfreshROOTappend, acceptedrawsNEVERappendagain.
Descriptor `core-mtp-model TARGET SIDECAR`; model-free bothGPU `core-mtp-attention`; full fresh commands/defaultregression contracts — README.md. Actualbuildlog ROOT/runs/r6-prerequisites-ba-a-build.log.
Freshrequest/record_request.py commands/fixedNEWfilenames/actualargv-vs-Source — README.md; exact4K ROOT/runs/r0-20261001T154816Z-4sffu3gz/4k.prompt-ids.json,16K ROOT/runs/r4-fullrequest-b522-16k.prompt-ids.json, cap4608/16896/chunk1024/slots112/primary1/.95/20/seed12345/512ignoreEOS/MTPoff.
Literalraw `candidate_unqualified`/`local_unqualified` labels preserved; boundedqualification неSource rewrite/allknobsproof. Optional0.1sdriverVRAMsamples НЕexactpeak; historical775collector1hostlogitfloorrepair НЕtolerance.

## Последний подтверждённый результат

831 MtpModel666checks/451rejects; Specpuremath35963782checks/715rejects/5522274hotcalls zeroheap. Actualmodelraw36records14462B:32typedroles/offsets/strides, Q8×19/F32×11/BF16indexQ-KONLY; top10/fulltyped10tokenizerassets/blk48compression0/distincttargetembeddingQ4-outputQ6; payloadderived2775621632B/strides1740800B, НЕpayloadread/execution.
BothGPUdense raw4records1298B: B1..3/fullcausalQ4KV/synthetic128K, frozen `2e-4+2e-4*abs(ref)`/oldN1shortbitexact/finite-bounds-poison-reject-sticky-canaries PASS/footerlive_resources0; НЕoccupiedfullmodel128K/trainedforward/perf/HF.
Targetraw473730B/protocol1/56records/3sequentialowners primarycap40batch3CPU0,widecap2048batch1024CPU0,CPU1syntheticfailure+reset:644390400fullvocabcompares2595rows/26419200EXACTtapvalues, maxerror/maxratio/diagnosticbitdiffZERO/unchanged `.02+.002*abs(ref)`; aggregate51restores84argrejects137steadyMem6sticky, footerprimary50/83/133.
45primarywindows N1..3×allretainedprefix0..N/mod4+EOS/PLEhash-conv/QSAtail/divergentsuffix/oldpublishedspan-tapfailurepreservation; widePPALL1024taps PASS.87optinGPUBuffers/defaultOFF: batch3GPU0/1 236851200/235622400B; batch1024GPU0same/GPU1 319262720B, twoGPU1tapbuffers41943040B EACH.
Raws ROOT/runs/r6-{mtp-model,mtp-attention,target-restore}-ba-a.jsonl; allDEFAULTreset/batch/B8/memorynativeEXIT0+strictcollectPASS, ROOT/runs/r6-default-{reset,batch,attention,memory}-ba-a.jsonl. Defaultcap131072slots112 НЕoccupied128K.
823 ОБАchunk1024/slots112/fresh/noPrefixReuse/cachewarmunknown/CPU0disabled/MTPoff/primary1/.95/20seed12345ignoreEOS512; AoriginalN1tile1/Bexplicitquerytile8protocol2. ExactR0archive/code4K IDs/fourconcat16K, неoriginaluser/retokenized/externalMTPbaseline.
Все6requests512actualoutputs/3072total, eachTG511/RNG512, consumed4607|16895/cap4608|16896; three512-IDarrays withinlength equal diagnostic-only. Rawlabels candidate_unqualified/local_unqualified unchanged.
4K A1/B/A2 PP160152.153592/115105.478965/160544.563734ms; request210982.242626/165876.889925/210718.406950ms. B PP35.5847526706/TG10.0650106725; meanA/B PP1.39305583109347×/request1.2711253802945932×.
16K A1/B/A2 PP724478.209838/509826.542585/723419.610449ms; request777620.365975/562117.081524/775858.095619ms. B PP32.1364201968/TG9.7725973247; meanA/B PP1.419990623620387×/request1.3818104027209437×.
B fasterBOTHAs PP/request EACHlength; TG~10/targetsUNMET. Completedwall PPwithoutoutputsampling/TGoutputs−1/loadseparate; no newGPUevents/profile/VRAM/exactpeak/HF/marginalRNG/CPUdispatchclaims.
Performance raws ROOT/runs/r4-attention-77f-ab-{4k,16k}-{a1,b,a2}.jsonl/.err и r4-attention-77f-ab-series.log/-fixture-source.json; historicalcanonical48/old41exact/+7/-0/all6LOCALcollect PASS. CorrectnessB8 1036984320comparesZERO/CLI2mixed-off32 PASS, shortTGdiagnostic-only/НЕpolicywin; detailsREADME.md.

## Контракт и границы

Own Session: RAM experts/default112 GPU slots/layer, static24/24, Q4 KV/FP32 index/GDN/PLE persistent, default max_batch1/pinned40KiB. Accepted logical≤1024: canonical nonexpert/shared/expert microtiles≤8, routes once/full chunk/triplet once/group; GDN chronological slices≤128. Band16/two stages/copy_ready/consumer_done сохраняются; новые weights/precision/arithmetic/epsilon/gates не менялись.
Targettap widened10240 BEFORE ROOT_HC, transactionaloldpublicationpreservedinvalid/failure; failure requiresreset. Restore logicalcursor/recurrent-conv-PLE-QSAtails, НЕrewindexpertslots/uploads/physicalcounters/fullKVcopy. Lastemittedtoken pending; targetselfparity НЕtrainedsidecarrollback/HForacle.
Dense donor KEEPmxsum64/ADAPTcurrentfurnace64splitmerge/WRITEOURSboundedcanonicalQ4→FP16RNE; existing mx/furnaceMIT/reinstinctApachepinsunchanged/no newdependency. Attentiontile1/OFF/originalAPIdefault unchanged; APIcounts НЕkernels.
QualifiedCPUlinear_GPUmiddle exactCPUgateup/down+canonicalGPUSiLU/Q8 onceCPU-bearinglayer; force_cpu НЕwholeCPU. WholeCPUlibm755/759 fullgatefailed, primitiveboundspassed — no tolerancewaiver/repeat; pipeline/lifetime detailsREADME.md.
FIRST Hnorm perbranch2560/distinctgamma4×2560 per pinnedqwen4exp.cpp438..450; whole10240firstnorm REJECTED, HCmixnorm whole10240 unchanged. Indexraw128безH/inverse,GDN[V][v][k]/h%16,PLEhash-convresetвместе,RoPE64j/j+32/absolute/sectionsARRAYINT32.

## Следующие действия

1. Parent immediate normalcommit/push acceptedprerequisitecode+docs+journal; actualLOCAL/history/+2/-0/diffcheck PASS, boundedcorrectnessslice ACCEPTED/CLOSED. НЕ rerun831/duplicateappend/retagba446f9dirtytrue; nativeexitproof уже получен.
2. THEN apply/review ignored `runs/r6-trained-mtp-forward.patch`73983B; independentactualdonororacle/fullLogits/state/teacherhistory/stochasticwindows, затем R5policy/PPtraffic measurements. `runs/r4-layerwise-prefill.patch`61985B тоже UNAPPLIED/UNQUALIFIED/no speedpromise.
FullR0–R8 обязательны: R4speed/R5measuredpolicy/R6TRAINEDMTP/R7occupiedlong-tokenizer-API/R8pack-finalgates OPEN. НеfullBF16download/ROCmchange/wave32; QSAfork НЕHForacle; MCPtransferlocal_root толькоcheckout.
