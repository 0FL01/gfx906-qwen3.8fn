# Текущее состояние

## Задача
Завершить весь PLAN.md и превзойти pinned llama.cpp в сопоставимых запросах.
Работать лично через SSH nc-lab, без engineering subagents. Каждый законченный
срез документировать, commit и push. R0–R3 закрыты; полный R4–R8 ещё OPEN.
400–600 PP / 30–40 TG и победа над llama.cpp не достигнуты.
Численные gates, weights, precision, top-k/QSA budget не ослаблять.

## Последний runtime / измерения
Latest opt-in logical16384/stage2048, sourceeaa778c dirty:
16K+512183.372317s, PP112.303156/TG13.634030.
A/B/A journal101 controls234.192985/231.931921s, PP84.069081/84.601701.
Meanrequest1.270979x, PP1.331639x. All512 IDs/RNG/acceptance/pending equal.
4K B83.416036s vs86.191997/85.046992, PP89.242071/TG13.620598,
but same two-stage schedule: no structural4K gain claim.
Previous dc60 pipeline logical4096 speed84.422/229.376s is journal95.
Pinned llama.cpp~57/~144s still faster. Default remains ordinary1024.
Saved core-mtp-run-baseline7df and core-session-baseline775 MUST stay untouched.
Full history and provenance: results.jsonl/README.

## Текущий срез / следующий шаг
Двух-GPU pipeline correctness завершён: journal94, source48f3d14 dirty.
Expanded series PID1453820, ROOT/runs/r4-pipeline-expanded-48f, native0
2026-10-05 10:36:56 UTC. Full strict build, 40/40 CTests, 1157.10s.
32/8,128/17,4096/2048:125642240/441443840/13494576640 сравнений,
ошибок и bit mismatches0; счётчики включают taps и повторные references.
8 fault/reset cases, MTP32 API, carry2051 с окнами1025/1025/1,
6 CLI rejects проходят. Предыдущая ошибка только в fixture: запрещённый getter
checkpoint_state после invalidation; исправлено ожидание rejection, не runtime.
Actual downloaded raws/manifests и старый93 byte prefix проверены.

Pipeline default off; logical capacity<=4096, отдельные host frames для24/24
GPU stages и2 ring slots. Дополнительных explicit GPU buffers нет. Все tap rows,
absolute positions, transactional publication и failure drain сохранены.
Host-only coordinator push48f3d14, journal93:84cases,20 repeats, ASan/UBSan pass.
TSan не стартовал из-за ASLR layout; security settings не менялись.

Pipeline correctness PUSHED dc60e0f56da9d2b10bdbb5de845491470da8d0eb,
remote master verified. Ordinary core-mtp-run rebuilt with this revision/dirtytrue.
Paired PID1468638 completed native0 at11:12:52 UTC; journal95 validated.
Actual six raws/manifests/one runtime digest and old94 byte prefix verified.
VRAM PID1474125 completed native0 at11:21:21 UTC; journal96 validated.
Separate16K+512, exact IDs/RNG to paired B. Fresh HIP mapping05:00/08:00.
3227 samples at0.1s,324.30s including load/request/cleanup:
max used12701908992/16393617408B; min free4461182976/769474560B.
Sampled global driver extrema, not exact instantaneous/per-phase peaks.
Actual copied raws and previous95 byte prefix verified.

Profile PID1476852 completed native0 at11:32:29 UTC; journal97 verified.
Source6b78bb7 dirty; ordinary dc60e0f binary unchanged. Actual32 and128
annotation trajectories match. Target PP traced52.720s; GPU kernel sums42.078s,
unions21.304/20.768s, intersection3.199s. Envelopes overlap21.508s INCLUDINGidle.
Q4 tiled15.904s (37.8% sum), attention8.845s (21.0%). Trace overhead material:
untraced/traced128-output requests55.019/64.428s; no speed claim from trace.
No copy byte counts in this CSV. Details/raw paths in README/journal97.

Bounded-head128 correctness COMPLETE, journal98; source8ea78b0 dirty.
Detached1483257 ended native0 at12:50:59 UTC. Full strict build41/41 CTests,
1151.11s. Native wide786677760/attention1036984320 zero errors/bits;
serial256/pipeline256x129 each862512640 exact; MTP API/carry and default-memory
capacity131072 pass. Actual controller collectors and old97 byte prefix verified.
Default-memory is allocation proof, not occupied128K.

Device.logits=min(frame,128)*V, ordered per-tile copies, full host output.
N1/short<=128 shapes unchanged. Actual head Buffer127139840B at frame1024;
matching old/new native ledgers show889978880B less owned/workspace per GPU,
other categories/counts unchanged. Not a driver peak measurement yet.
Wide/long/attention protocol2 reports exact head bytes; protocol1 ORIGINAL floors
remain enforced. New head-layout test plus actual old/new collectors pass.
No new native long4K/16K reference run claimed. Source-off snapshot:
controller runs/head-logits-off-8ea.hip.

Head128 slice PUSHED58f3bbdbea1415c52894f82447b580e1a73536fd; remote verified.
Ordinary core-mtp-run rebuilt with58f3bbd/dirtytrue.
Head128 VRAM observation completed native0 at13:07:51 UTC; journal99.
PID1497345, ROOT/runs/r4-head128-vram, source58f3bbd dirty.
All512 IDs/RNG/acceptance/pending match paired pipeline B. Fresh HIP mapping.
3171 samples/0.1s: max used11812704256/15504785408B; min free
5350387712/1658306560B. Sampled global driver extrema, not exact peak or speed.
Old98 byte prefix and actual request/observer logs verified.

Next slice: stage-sized pipeline resources with logical capacity up to16384,
fixed2048 stage. Current physical arrays still reserve4096 rows. Allocate
original Q8/contributions/DTO and six host activation vectors for the stage,
while published root taps keep ALL actual logical rows. Default serial path
keeps its logical-capacity allocation. Report logical and physical capacities.
Respect coordinator max4096 windows and constructor stage bound<=4096.
Estimated root free after stage shrink and16K taps867450880B, NOT fit proof.
Native long fixture should retain one full N1 reference (validate old1024 path
against it) rather than two redundant16GB arrays; preserve full tap evidence.
Then actual MTP long carry and full41CTest, followed matched full requests.
Stage-sized logical16K/2048 correctness COMPLETE, journal100.
Detached1504300 native0 at2026-10-05 15:06:07 UTC, d544b6e dirty.
Full41/41CTest1156.08s; full16K29,497,640,960 cumulative comparisons,
zero absolute error/bit differences; small256x129862,512,640 exact.
MTP long resources, two last-window faults and16387 carry32IDs/RNG pass.
Actual copied logs/manifests and old99 byte prefix verified. Large correctness
fixture used host swap, no speed claim; one full reference and all taps retained.
Slice PUSHED eaa778c00133891d3882f83942487da964108114; remote verified.
Ordinary runtime rebuilt with eaa778c/dirtytrue.
Paired A/B/A PID1520517 completednative0 at15:37:12UTC.
ROOT/runs/r4-pipeline16k-ab-20261005; journal101 actualsixlogs verified.
One unchanged eaa778cdirty binary, logical4096vs16384, fixedstage2048.
Separate driver VRAM observer PID1525727 endednative0 at15:43:58UTC;
ROOT/runs/r4-pipeline16k-vram, journal102 verified.
Minfree5,566,386,176/868,737,024B,2710samples; exact512IDs/RNG; no speedclaim.
Next: existing opt-in4096 stage withinlogical16K, full4097-row logits/taps
and16K+512 A/B/A stage2048vs4096. Extra215,777,280B/device bysource,
estimated root652,959,744B left is NOT fit proof. No runtime arithmetic edit.
No GPU workload active; prepare unique driver/source stamp and launch.
Full PLAN and llama.cpp victory are still open.

## Не повторять
Packed640, small/large-output N16, row-owned2560 и row1/row4 Q4 geometry
не дали устойчивого model/coordinate выигрыша; production dispatch canonical8x2.
Expert-wideMMQ провалил frozen fullmodel gate; не включать/не ослаблять bounds.
Последний phaseprofile layerwise: PP55s, Q4N8~38%, attention~22% kernel-sum;
sums overlap, не wall breakdown. RootGPU ограничивает расширение fullwindow.
Same-build self-parity не independent HF; capacity128K не occupied128K;
custom stop не natural EOS. Полный trained-MTP oracle/R7/R8 ещё открыты.

## Машины / команды
Controller /home/opencode/ai/gfx906-qwen3.8fn, branchmaster, origin0FL01/gfx906-qwen3.8fn.
GPU amude/radneon, ROOT=/home/radneon/gfx906-core, sourceROOT/src, buildROOT/build,
runsROOT/runs, canonical journalROOT/results.jsonl (не ROOT/src/results).
Route controlleropencode.jsonc; explicit /bin/sh (login shell fish).
Models read-only /home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf and
mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf; не дублировать/не hash модели.
Docker llama.cpp-gfx906:cmake-4.4.3, bind ROOT:/core and models:/models:ro.
GPU: --device /dev/kfd --device /dev/dri --group-add video --ipc host.
Configure Release, HIP /opt/rocm/llvm/bin/clang++, prefix/opt/rocm,
CORE_REVISION=actual40 and CORE_DIRTY=ON; fullbuild sh /core/src/tools/build.sh.
CPU coordinator: buildtargetprefill-pipeline-test then
ctest --test-dir /core/build -R '^prefill-pipeline$' --repeat until-fail:20.
Donor pinned image llama.cpp-gfx906:pp-stream-dcd685463d, sourcepin
dcd685463d597d31f5ca759d32c94592a2740fa4; isolated no-public-port runs only.
512actualoutputs, capacityprompt+1024, primarysampling1/.95/20/seed12345.
One GPU workload, build outside timing. SSH drop: inspect same job first;
backoff1,2,4,8,10min, no duplicate heavyweight runs/network changes.
