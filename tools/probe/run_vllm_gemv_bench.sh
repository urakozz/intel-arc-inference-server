#!/usr/bin/env bash
# Throwaway Spec 1.7 experiment wrapper. Run on the B70 box; no installs.
set -euo pipefail

# t214 is gone from the box since 2026-09-09; rows already recorded under it
# keep their own label.
BENCH_IMAGE="${BENCH_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest}"
BENCH_REPO_DIR="${BENCH_REPO_DIR:-/home/user/b70-inference-server}"
BENCH_HF_CACHE="${BENCH_HF_CACHE:-/home/user/.cache/huggingface}"
BENCH_SNAP="${BENCH_SNAP:-/hf/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/2a9077667e28aa53e61d91bdee5d7962e8674668}"
BENCH_UID="$(id -u)"
BENCH_GID="$(id -g)"
BENCH_RENDER_GID="$(getent group render | cut -d: -f3)"
BENCH_VIDEO_GID="$(getent group video | cut -d: -f3)"

exec docker run --rm --entrypoint python3 \
  -u "$BENCH_UID:$BENCH_GID" \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$BENCH_RENDER_GID" --group-add "$BENCH_VIDEO_GID" \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e ZE_FLAT_HIERARCHY=FLAT \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CCL_ZE_IPC_EXCHANGE=sockets \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HOME=/scratch/hf -e HOME=/scratch -e PYTHONUNBUFFERED=1 \
  -v "$BENCH_REPO_DIR:/ws" -w /ws \
  -v "$BENCH_HF_CACHE:/hf:ro" -v /tmp:/scratch \
  "$BENCH_IMAGE" /ws/tools/probe/vllm_gemv_bench.py --snap "$BENCH_SNAP" "$@"
