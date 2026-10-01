# Qwen3.8-Flash-Next core for 2×gfx906

Standalone C++20/HIP core under implementation. The model is **not yet runnable**
in this core; `core-probe` is the verified R0 hardware/build slice, not inference.
Scope and acceptance are in [PLAN.md](PLAN.md); current evidence in [STATE.md](STATE.md).

## Remote build and R0 probe

Mirror `CMakeLists.txt`, `src/`, `tools/`, `tests/` through `mi50-llama-remote`
to `/home/radneon/gfx906-core/src/`. Keep weights at their existing paths.
The working build image is `llama.cpp-gfx906:cmake-4.4.3`, ROCm 7.14,
Clang 23; production runtime image does not contain CMake.

```sh
docker run --rm --name core-build --entrypoint /bin/sh \
  -v /home/radneon/gfx906-core:/core \
  -e CORE_REVISION="$CORE_REVISION" -e CORE_DIRTY="$CORE_DIRTY" \
  llama.cpp-gfx906:cmake-4.4.3 /core/src/tools/build.sh
docker run --rm --name core-probe --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --security-opt seccomp=unconfined \
  --entrypoint /core/build/core-probe -v /home/radneon/gfx906-core:/core \
  llama.cpp-gfx906:cmake-4.4.3 > /home/radneon/gfx906-core/runs/r0-probe.jsonl
python3 -B /home/radneon/gfx906-core/src/tools/record_probe.py \
  /home/radneon/gfx906-core/runs/r0-probe.jsonl \
  --results /home/radneon/gfx906-core/results.jsonl
```

Before building, set `CORE_REVISION` to the synchronized controller commit and
`CORE_DIRTY` to `OFF` (clean) or `ON` (changed sources). No host `-march=native`; AVX2/FMA/F16C
are checked before target-attributed CPU code. GPU compilation explicitly targets
gfx906. Probe errors exit nonzero; transfer/consumer and both GEMMs are checked.
RAM read/FMA and concurrent read/H2D are synthetic hardware measurements, **not**
quantized expert throughput or full-request speed.

## Existing production baseline (separate process)

No other GPU workload or heavy compiler during final measurements. If the named
container exists, inspect it first; do not blindly launch a second model instance.

```sh
sh /home/radneon/gfx906-core/src/tools/baseline-server.sh
python3 -B /home/radneon/gfx906-core/src/tools/baseline.py \
  --runs-dir /home/radneon/gfx906-core/runs \
  --results /home/radneon/gfx906-core/results.jsonl \
  --revision dcd685463d597d31f5ca759d32c94592a2740fa4
docker logs core-baseline > /home/radneon/gfx906-core/runs/r0-baseline-server.log 2>&1
docker stop --timeout 30 core-baseline
```

The client saves new exact token-ID fixtures, raw requests/responses and router
configuration, checks 32+64 and 4096+512 counts without prefix reuse, and uses
temperature 1.0 / top-p 0.95 / top-k 20. `ignore_eos=true` is recorded explicitly
to fix output lengths. These are not the unavailable original user's prompts.
Cache warmness is a protocol label, not measured occupancy; one short warmup is
not a claim of fully warm experts. Unavailable structured acceptance histograms
are `null`, not inferred from draft totals. A failed completion is never retried.

Local client tests (no ROCm required):

```sh
cmake -S . -B build -DCORE_WITH_HIP=OFF
ctest --test-dir build --output-on-failure
```

Or without local CMake: `python3 -B -m unittest discover -s tests -p baseline_test.py`.
