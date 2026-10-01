#!/bin/sh
set -eu
# Run on amude, separately from core-probe or other GPU workloads.
# Existing production preset; local-only HTTP endpoint, no API secrets.
if docker container inspect core-baseline >/dev/null 2>&1; then
  state=$(docker container inspect --format '{{.State.Status}}' core-baseline)
  if [ "$state" != exited ] && [ "$state" != created ]; then
    printf '%s\n' "core-baseline is $state; inspect it before restarting" >&2
    exit 1
  fi
  docker rm core-baseline
fi
docker run -d --name core-baseline \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -p 127.0.0.1:18080:8080 \
  -v /home/radneon/models-nvme:/models:ro \
  -v /home/radneon/llama/models.ini:/config/models.ini:ro \
  -e HIP_VISIBLE_DEVICES=0,1 -e LLAMA_REUSE_NO_DRAIN=1 \
  -e GGML_ENABLE_CUSTOM_AR=1 -e LLAMA_ENABLE_MTP_OPT=1 -e GGML_CUDA_P2P=1 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 -e GPU_MAX_HW_QUEUES=8 \
  -e LLAMA_MOE_CACHE_PREFILL_D2D=1 -e GGML_CUDA_MOE_PREFILL_STREAM=1 \
  -e GGML_SCHED_MOE_PREFILL_OWNER=0 -e GGML_CUDA_MOE_PREFILL_TRACE=0 \
  -e GGML_CUDA_MOE_PREFILL_DEVICE_TRACE=0 \
  --entrypoint /app/llama-server llama.cpp-gfx906:pp-stream-dcd685463d \
  --host 0.0.0.0 --port 8080 --models-preset /config/models.ini \
  --models-max 1 --models-autoload
