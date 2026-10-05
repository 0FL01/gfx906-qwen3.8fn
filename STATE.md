# Текущее состояние

## Задача (2026-10-05 01:45 UTC)

Завершить ВЕСЬ PLAN.md и превзойти llama.cpp в сопоставимых полных запросах.
Лично через SSH nc-lab, без engineering subagents. Document+commit+push каждый
законченный срез. R0–R3 закрыты; R4 speed/R5 dispatch/fullR6/R7/R8 ещё OPEN.
400–600PP/30–40TG и llama.cpp win НЕ достигнуты. Frozen gates/weights unchanged.

## Последний подтверждённый runtime

Opt-in layerwise4096 b2014ce/dirtytrue:4K91.233s,16K267.236s (both+512),
PP77.553/71.706 and TG13.302/13.188. Journal81, paired both-controls win.
Default configuration remains ordinary1024; prior last-head result below.

Last-head48de1ba/dirtytrue:4K104.385s PP62.1766/TG13.2706;16K323.481s
PP57.6149/TG13.0663. Paired result pushed5677a87b2bee81ff25ff41f5918a4ac447bbdbb0.
The preceding sampler slice is recorded below for comparison.

Sparse sampler paired job1791161686931-537 exit0/31m53, result pushed
daee1efcd6e83c115175284d7ce9512045298b8d; component350f8afe3f988af1a79a74f47e5f782f0c7051f7.
4K+512 A112.812/110.077s -> B109.526s (1.017521x),PP58.3219/TG13.0049.
16K+512 A334.585/333.669s -> B330.152s (1.012040x),PP56.1301/TG13.3572.
All512 IDs/acceptance/RNG/pending exact inside both triplets. Source2535cbf/dirtytrue,
explicit off/on sampler snapshots; capacity n+1024,slots112,chunk1024,tile8,
seed12345/temp1/top-p.95/top-k20/ignoreEOS. Load separate, expert warmness unknown.
Pinned donor dcd6854 same target/sidecar/headroom:4K~57s,16K~144s, still faster.
Donor QSA historical semantics are not an independent correctness oracle.

Current ordinary core-mtp-run is last-head48de1ba/dirtytrue (rolled attention,
selective tiled projections, fused SiLU/Q8 and sparse sampling). Saved core-mtp-run-baseline7df1c74
MUST stay untouched: overwrite was denied; no permission was obtained.
Safe comparisons rebuild ONLY ordinary executable between actual source variants,
never compile during timed inference; saved baseline checked unchanged.
Original core-session-baseline775/source77f89c3dirty (1750328B) also untouched.

## Текущий код / следующий физический шаг

New Session::prefill_last consumes the full window and retains ALL pre-head tap
rows, but projects/copies only final logits. All-row step_batch/verify unchanged.
No new allocations; root HC still computes all rows. API PUSHED48de1ba23172d0d8ddf301e5742099caed1dc7ab.
MtpRunner integration PUSHED3c1ee722bea7251df1eea4d8201e96e4eb20aec9.
Strict selected builds/model jobs562(exit0/2m32) and568(exit0/3m19) passed,
compiled daee1ef/dirtytrue. Checkpoint fixture35,261,440 values,ordinary2,483,200:
zero bit mismatches/maxabs. Nine checkpoint cases,26 invalids,full tap bits,
mixed APIs,pending invalidation,restore,continuations,capacity40/slots1/max32.
Both checkpoint modes keep owned allocations steady; RAII cleanup reported,
not a separate post-destruction VRAM recovery proof. No independent HF/speed claim.
Raw ROOT/runs/r4-prefill-last-dae{,-expanded}.jsonl and build/error logs.
Canonical journal74; latest copied API/CTest raw and old73 byteprefix exact;
records72 initial and73 strengthened fixture are distinct, do not reappend.

Full build/MTP API/39CTest job1791164948838-576 COMPLETEexit0/22m00,
source48de1ba/dirtytrue;39CTest1154.97s and MTP API pass.
Raw ROOT/runs/r4-mtp-last-48d-{build.log,api.jsonl,api.err,ctest.log}.
Last-head paired job1791166462036-582 COMPLETEexit0/31m35; journal75,
old74 prefix and actual six raws/footers/512 IDs/manifests verified.
4K all-head110.443/108.628s -> last104.385s (1.049345x),PP62.1766/TG13.2706.
16K all-head332.074/334.021s -> last323.481s (1.029575x),PP57.6149/TG13.0663.
Both candidates beat both controls; all512 IDs/acceptance/RNG/pending exact.
Source48de1ba/dirtytrue; explicit MtpRunner snapshots; same sampling/capacity.
Saved baseline7df unchanged. Ordinary core-mtp-run restored to last-head candidate.
No GPU work active after582. Full-plan/llama.cpp win remains open.

Small-K one-wave experiment608 COMPLETEexit0/1m18:309cases/device,
9,161,968 exact GPU values and99,356 CPU samples/device,19host rejects,
308graph checks. Legacy tiled/short/dense pass. But only19/48 resident down
coordinates beat both controls; median ratio1.00688, with regressions.
NOT promoted; own candidate API/test changes reverted, runtime never integrated.
Patch/test preserved in runs/r5-small-k-567.patch and source artifact; journal76.
Same original tiled fixture confirmed its baseline timings unchanged; do not
compare cross-fixture/cache regimes as a speedup.

Tile4x4 experiment624/631/633 completed exit0; journal77. Exact component
gates pass, but broad Q4_0 down geometry regresses; rejected/restored.
48 coordinates,28 beatboth,median1.025915; dominant N32..128 down0.871–0.897x.
Only narrow gate/up tails show useful potential; not model-qualified.
All candidate source/extended fixture preserved in runs; production untouched.

Layerwise prefill correctness slice PASSED, opt-in defaultoff. Sourcec685d14/
dirtytrue; model32job649,128job654 exit0.4096job667 model/fullbuild passed,
then SSH disconnected255; SAME container finished39CTest1151.06s, recovered
with docker-wait job676 exit0. No rerun. Journal78 copied and verified.
Byte-copy16K bound job663 exit0/4m18:bothGPUs327680pairs each,9310gather/scatter,
270host rejects; original copy and491148-check CPU routes pass.
Target-only4096 freeGPU0/1~4.56/3.75GB; trained-MTP fit NOT checked yet.
Warm monotone-ID fixture halves expert upload bytes but all-row is slower;
no component/model speed promotion. Qualified correctness only, not fullR4.

MtpRunner/CLI opt-in integration passes bounded API and six early CLI rejects:
job1791174641902-701 exit0/3m51; cap0/cap16 actual32 IDs/RNG/reset/cap48
match old expected fixture, custom stop only. Sourcebcd5320/dirtytrue; journal79.

Paired bcd series STOPPED:4K-A1 completed105.542s, then4K-B nativeexit1
at MtpSession::save_target_carry: stale hard cap tap.rows<=1024. Constructor
and full4096 target/teacher had run, but no generation/performance result.
Failed raws preserved under runs/r4-layerwise-mtp-ab-bcd. No A/B speed claim.
Old shell exit-wrapper also misquoted printf (empty exitfile); actual candidate
manifest records native1 and traceback, no active GPU after failure.

Carry validator repair PASSED job1791175643524-720 exit0/4m13:
2051prompt, default1024 vs layerwise[1025,1025,1], exact32IDs/RNG/acceptance.
Sourceb2014ce/dirtytrue, journal80; failed paired bcd run retained/excluded.
No arithmetic/allocation change, carry still copies one checked row.

Fresh paired series COMPLETE native0 at05:21:28UTC,sourceb2014ce/dirtytrue,
journal81.4K A104.163/105.963→B91.233s,PP77.5529/TG13.3018,meanA/B1.15158.
16K A319.448/322.708→B267.236s,PP71.7060/TG13.1884,meanA/B1.20148.
All512 IDs/acceptance/RNG/pending exact inside triplets; saved baseline untouched.
Raw ROOT/runs/r4-layerwise-mtp-ab-b201 and .exit.json verified and copied.
Layerwise4096 remains explicit opt-in; both target+MTP requests fit, no new peak
VRAM evidence yet. Earlier bcd failed series excluded. Fullplan/llama win open.

VRAM observer COMPLETE native0, journal82, sourceb201/dirtytrue,3617samples.
Fresh HIPmapping0=05:00.0/1=08:00.0;16K+512 exactIDs/RNG. Minfree4.461GB/
0.769GB (global sampled, not exactpeak). Do NOT increase8192window blindly.
Root observed headroom is only~0.716GiB with target+MTP at current settings.

Fresh profile749 exit0/4m20 and analyzer751 exit0,sourcebd1fa883/dirtytrue,
journal83. TargetPP55.047s/1.081Mkernels,sum41.739s;Q4tiled16.023s38.39%,
attention9.302s22.29%,GDN1.683s,dense1.651s. H2Dsum3.008/2.906,D2H.654/.653.
No crossingevents. Diagnostic only. Original runtimeb201 unchanged; core-mtp-
profile separately built. No GPU work active after profile.

PhysicalCols16/Rows2 Q4 candidate REJECTED:758/762/763 native0, source56496c8/
dirtytrue.512cases and530056502GPU values/device exact,136600CPU samples,
21rejects/510graph checks. Only1/42 wins,median0.82095,range.72506–1.02792.
Code/fixture restored; patch preserved, journal84. No model runtime rebuild.

Selected Q4 K640 packed3rows/128threads component passes both GPUs530cases,
530126594exactvalues,172996CPU samples,21rejects/528graphs. Both rows6 and
rows3 win all36 A/B/A down coordinates; rows3 range1.10007–1.52507,median1.21848.
Native772/776/778/781/783 all0,source40cb728/dirtytrue,journal85. OriginalN8/
tails/math order preserved; staticLDS6KiB,no new long-lived VRAM. Candidate
active onlyQ4_0/Q4_1,K640,M<=2560,N>=16. K2560 extra row-owned variant
REJECTED0/18wins,median.77683 (jobs790/792),patch preserved,journal86.
Actual raws copied/verified; ordinary runtime was not rebuilt during timings.

Packed640 full-model gates COMPLETE native0 at06:53:52UTC,source40cb728/
dirtytrue, journal87. Strict fullbuild/39CTest1145.90s pass. Short29.8M/
wide1.03698B vocabulary comparisons zeroerrors/bits, layerwise128 and MTPAPI
pass. Actual raws copied and strictshort/wide collectors rechecked on controller.

Packed correctness pushed52d7b62; fullrequest series COMPLETE native0 at07:27:18,
source52d7b62/dirtytrue,journal88.4K B93.489s versus93.631/91.215;16K B267.740s
versus268.791/267.035. PP ratios.999198/.999419, no stable improvement.
All512IDs/RNG exact. Production packed kernel REVERTED in source, expanded
tests retained; reproducible implementation Git52d7b62 and runs artifacts.
Ordinary executable still needs rebuild after this source commit (last build
was packed-on restoration). Protected baseline7df stays untouched.
Best qualified fullrequest improvement remains layerwise b201:91.233/267.236s.

Existing trace Q4_0 N8 split by grid-inferred output M:10240=5.338s,
2560=4.292s,640=3.874s,6144=1.384s,320=.670s,12288=.465s. M alone does NOT
identify K, so don't attribute allM2560 to down. Analysis saved/copied.
LargeM N16 remains untested; earlier negative candidate restrictedM<=2560.

NEXT attention deferred-check component A/B/A. Prepared ONLY under runs:
attention-deferred-{off,on}.hip and attention-deferred-test.hip. Production
attention source untouched. Fixture adds two overflow cases, optional
--benchmark after --capture/--compare,12resident coordinates/GPU and old bits.
First sync strengthened fixture, build restored core-mtp-run +core-attention-
batch with actual newHEAD/dirtytrue, capture fresh A1 snapshot. Then swap ONLY
attention.hip candidate for B(compare same snapshot), restoreold for A2.
Must preserve nonfinite detection/publication and exact valid operation order.
No new GPU work running after paired completion. FullPLAN/llama win open.

## Принятые границы / не повторять

Rolled attention removed spills256VGPR/4380scratch ->52/0; same arithmetic,
cross-build3.745m component values exact, full model gate unchanged.
Selective tiled logical<=128 keeps physicalN8; dense output>512 stays oldN8
because batching it regressed. Both4K/16K paired accepted; primitives60d341e.
Fused SiLU/Q8 exact component1.35–1.58x translated to only~1.3%16K request gain,
no stable4K gain. Sparse CPU sampling skips EXACT zeros only, original generic
Neumaier Sum/filtering retained;95,389,224 snapshot bytes and signed-zero/tiny
residual tests exact. CPU draw/acceptance/residual components3.78/4.57/2.83x;
full requests only1–2% faster. Detail/provenance in README and results.jsonl.
Q4 tiled row4 and row1 candidates rejected on same-coordinate A/B/A, reverted.
Expert-only wideMMQ fails shortN32:7,559,835 violations,maxabs4.395458.
Tiny layer0 expert differences precede layer1 HC bound failure; mechanism not
independently proven. Reverted; never enable failed fixtures or loosen bounds.
Latest GPU phase profile was pre-fused/pre-sampler source60d341e: PP4K traced
72.995s/1.50m kernels, Q4 tiled33% and attention20% summed GPU time. Sums overlap.
Actual modelEOS and occupied128K remain unproven; custom-stop and capacity tests
do not close those. MTP independent sequential donor32 is bounded diagnostic,
same-build model self-reference is not independent HF proof.

## Машины / файлы

Controller: nc-lab:/home/opencode/ai/gfx906-qwen3.8fn, master, origin0FL01/gfx906-qwen3.8fn.
GPU: amude, radneon; SSH route/config in controller opencode.jsonc. Use SSH nc-lab.
Remote login shell is NOT assumed POSIX; explicit /bin/sh for set -eu/multi-step.
ROOT=/home/radneon/gfx906-core; source=$ROOT/src; build=$ROOT/build; runs=$ROOT/runs.
Canonical journal ONLY $ROOT/results.jsonl; controller receives verified copy.
Target model /home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf (75399121792B).
Sidecar /home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf (2786568256B).
Build image llama.cpp-gfx906:cmake-4.4.3, Clang23/HIP7.14.60850/rocBLAS5.5.
Donor production llama.cpp-gfx906:pp-stream-dcd685463d, source
/home/radneon/src/worktrees/qwen38-pp-trace-75, pin dcd685463d597d31f5ca759d32c94592a2740fa4.
Oracle libs $ROOT/build/oracle-production-libs; new oracle binary
$ROOT/build/mtp-canonical-diag/mtp-teacher-oracle; output dirs must be fresh.
Strict build: sh /core/src/tools/build.sh inside existing build container with
actual CORE_REVISION/CORE_DIRTY, /core bind ROOT, /models read-only model bind.
GPU tests: /dev/kfd,/dev/dri,video group; ONE GPU workload at a time.
16 physical/32logical CPU,oneNUMA; 2gfx906 wave64/CU60,17163091968B each.
LTE/reverse tunnel unreliable: inspect existing job/log after reconnect; no blind reruns.
