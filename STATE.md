# Текущее состояние

## Статус и текущая задача

R0–R3 закрыты: own48 Session/teacher32/reset и capacity131072 owned ledger. Short/logical-wide1024 и real4K/16K long-correctness slices qualified; полный R4–R8 и целевые скорости открыты.
Текущая задача — paired4K/16K full requests с512 actual outputs без prefix reuse: fixed R0 text fixture IDs4K и отдельно сохранённая новая16K text fixture, primary sampling, PP/TG/full elapsed и sampled observed VRAM обеих GPU; затем peak-VRAM/paired performance gates и следующий измеренный bottleneck.
Exclusive job `1791030847464-742` completed exit0/elapsed1h15m35s: strict CXX20/HIP20 Release gfx906/warning gates, CTest32/32 (605.76s), actual Q8 capture both GPUs, real16K→4K/remote collectors PASS, reset/batch exit0/passed footers и actual default-memory collector PASS. GPU idle.
Compiled `2e9848d43cf9f908dc8280f81e111c0cde86f01c` dirtytrue; future commit не relabel artifacts. Intended closure — long correctness/sampling prerequisites; parent VRAM observer/synthetic tests и future CPU expert/shadow/speculative helpers вне runtime qualification.
Parent LOCAL collect(downloaded actual16K/4K) PASS; remote prevalidated оба raws, `record_prefill_long.main` appended ровно два ROOT records19→21. Old byte prefix/parsed19 exact preserved; последние два `r4b_long_prefill` rows16384→4096 passed=true/R4_complete_claim=false/source2e dirtytrue. Downloaded canonical journal1 572 766B; gitdiff+2/-0 verified.

## Соединение и рабочие пути

MCP `mi50-llama-remote`, хост `amude`, пользователь `radneon`. Controller checkout — исходник; сборка/GPU/final CPU measurements только remote. При обрыве проверить существующий процесс/лог после восстановления; не запускать второй экземпляр вслепую.
Модель: `/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf` (75 399 121 792 bytes), источник пользователя https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF.
MTP: `/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2 786 568 256 bytes); источник загрузки не установлен.
Baseline image `llama.cpp-gfx906:pp-stream-dcd685463d`; source `/home/radneon/src/worktrees/qwen38-pp-trace-75`, revision `dcd685463d597d31f5ca759d32c94592a2740fa4`. Config `/home/radneon/llama/{docker-compose.yml,models.ini}`; наш baseline-контейнер остановлен.
Remote mirror/build/runs: `/home/radneon/gfx906-core/{src,build,runs}`, canonical journal только ROOT `/home/radneon/gfx906-core/results.jsonl`.
Build image `llama.cpp-gfx906:cmake-4.4.3`: CMake4.4.3/Clang23/HIP7.14.60850/rocBLAS5.5; gfx906 GEMM проверены. Runtime image без CMake.
Hardware: 16 physical/32 logical CPU cores, один NUMA, AVX2/FMA/F16C; обе gfx906 wave64/CU60/VRAM 17 163 091 968 bytes. RECON без изменения машины/стека не повторять.

## Рабочие команды

Latest actual build: `docker run --rm --name core-build --entrypoint /bin/sh -v /home/radneon/gfx906-core:/core -e CORE_REVISION=2e9848d43cf9f908dc8280f81e111c0cde86f01c -e CORE_DIRTY=ON llama.cpp-gfx906:cmake-4.4.3 -c 'sh /core/src/tools/build.sh'`. Для новой сборки actual revision/dirty; explicit `sh` нужен: source build.sh100644.
Long: `docker run --rm --name core-prefill-long --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined --entrypoint /core/build/core-prefill-long-test -v /home/radneon/gfx906-core:/core -v /home/radneon/models-nvme:/models:ro llama.cpp-gfx906:cmake-4.4.3 /models/qwen38-keep1-Q4_0.gguf 16384 > /home/radneon/gfx906-core/runs/FRESH-prefill-long-16384.jsonl`; ROWS ровно4096 или16384.
Только после exit0: `python3 -B /home/radneon/gfx906-core/src/tools/record_prefill_long.py --raw /home/radneon/gfx906-core/runs/FRESH-prefill-long-16384.jsonl --results /home/radneon/gfx906-core/results.jsonl`; проверить collector exit status, append один validated record. Existing actual raw не append повторно.
Session: тот же Docker/device/mount контракт, entrypoint `/core/build/core-session`; `--prefill-chunk 32 --generate 32 --ignore-eos --sample --seed 42 --temperature 1.0 --top-p 0.95 --top-k 20 MODEL 248044 $(seq 100 130)` — short prerequisite. Reset/batch/memory entrypoints `core-session-test`/`core-session-batch-test`/`core-memory`, только MODEL; полные repro — README.md.
Oracle teacher: `sh tools/build-oracle.sh`, production libs + `--all-layers --hf-gdn-l2-control --hf-qsa-f32-control --warm-cache 12 --cache-inserts 10` diagnostic-only, не bitwise HF/unchanged-math/performance baseline.
Fresh logs/trace paths в existing runs/; одна GPU-нагрузка, final measurement без тяжёлой сборки/tracing.

## Последний подтверждённый результат

Real16K:129 records/3 phases/49248 rows/16518 windows; 12 229 263 360 finite/8 152 842 240 compared full-vocab values. 8 atomic rejects/16539 memory observations; logicalmax1024/groups>128:28064; minfreeGPU0/1:5 184 978 944/4 704 731 136B; correctness wall2 679 118.940426ms, NOT benchmark.
Real4K:93 records/12384 rows/4206 windows; 3 075 194 880 finite/2 050 129 920 compared; 8 rejects/4227 memory observations; logicalmax1024/groups>128:6800; minfreeGPU0/1:5 260 476 416/4 780 228 608B; diagnostic wall607 945.098957ms, NOT benchmark.
Оба actual raws:zero violations/maxabs/maxboundratio/diagnostic bit mismatches, frozen `.02+.002*abs(ref)` и all-finite; bit equality diagnostic-only. CapROWS+32/slots112/max1024; retained N1 reference/canonical1024/occupied5N1→997, 32 N1 continuations в каждой фазе.
Teacher BOS248044→monotone — новая ID-family, не original text/user/baseline parity. Same-Session self-reference не independent HF; observed2047–2056/mod4 causal coverage не GPU selected-ID trace. Нет physical128-tile/individual-capacity/full-RAM/wide-owned-release/peak-VRAM/performance/fullR4 claim.
Previous job729 failureN1 offset13206: job733 trace finite ATTENTION OUTPUT layer8/block117, FP32 d≈2.728e-8>0→storedhalf0 ошибочно rejected. Pinned mx keeps original codes/raw sum; CPU/оба GPU repair сохраняет контракт, не optimization/NaN clamp/all-zero replacement.
Nonzero-input blocks с original FP32 d0/unsafe±128/nonfinite/FP16 InfNaN still reject до unsafe int8 cast; logical32 halves wave64 isolated. Target quant791975 checks/214 rejects и actual192 blocks/6144 F32 both-GPU proof jobs740/742 PASS; weights/precision/epsilon/gates unchanged.
Logs `runs/r4-prefill-repaired-b-build.log`, `r4-q8-underflow-actual-b.jsonl`, `r4-prefill-long-b-{16384,4096}.jsonl`, `r4-prefill-repaired-{reset,batch,memory}-b.jsonl`. Actual `record_memory.collect` PASS(defaultcap131072/slots112), duplicate memory append не требуется.

## Контракт и границы

Separate `SessionRouteStats`/`route_stats()`: last-call max assignments к одному expert/layer и cumulative call/layer/expert groups>128 since reset. Publish только success; constructor/successful reset zero; invalid arguments/execution failure preserve. Historical `SessionStats`/protocols unchanged; execution failure требует reset.
Own Session: RAM experts/default112 GPU slots/layer, static24/24, Q4 KV/FP32 index/GDN/PLE persistent, default max_batch1/pinned40KiB. Accepted logical≤1024: canonical nonexpert/shared/expert microtiles≤8, routes once/full chunk/triplet once/group; GDN chronological slices≤128. Band16/two stages/copy_ready/consumer_done сохраняются; новые weights/precision/arithmetic/epsilon/gates не менялись.
Canonical attention dot/value ordering и internally widened exp — ранее маркированные accuracy changes, не speedup. Wide MMQ выигрыши component-only; PP/full-request скорость не established. Historical R3/short details — README.md/results.jsonl.
CLI/sampler integrated strict; prior729 greedy32/primary32 seed42 chunk1vs32 IDs same/RNG32. Primary temperature1.0/top-p0.95/top-k20; full512 series ещё не qualified. Greedy diagnostic-only, actual output count; последний emitted token pending. Trace-range diagnostic-only/default unchanged.

## Следующие действия

1. Для следующих binaries указать actual commit/dirty; accepted long artifacts сохраняют compiled2e/dirtytrue. Не append accepted raws повторно и не retag старые logs.
2. Выполнить paired4K/16K full512 requests на fixed4K/new16K text IDs с primary sampling, PP/TG/full elapsed/observed VRAM и paired performance; затем measured bottleneck. R5 CPU miss vs H2D+GPU/overlap открыт.
Index raw128 без H/inverse; GDN [V][v][k], h%16; PLE hash/conv reset вместе; RoPE64 j/j+32/absolute position, sections ARRAY INT32.

## Не повторять без причины

Не скачивать full BF16 на заполненный SSD/не менять рабочий ROCm/не переносить wave32 assumptions. QSA fork не HF oracle без boundary fixture. Router ждать /models.current.status=loaded/model=current, не только /health. MCP transfer local_root ограничен checkout; запрет не обходить.
