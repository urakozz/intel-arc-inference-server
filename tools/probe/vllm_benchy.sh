#!/bin/bash
# On the box, from a tree root: start vLLM (the box's XPU image) under the shared GPU lock, run
# llama-benchy against it, stop the container - serve_benchy.sh's twin, so one client, one set of
# flags and one card measure both servers (queue row 41).
#   tools/probe/vllm_benchy.sh [llama-benchy args...]
# Env: MODEL (repo id or snapshot dir under HF_CACHE; default the W4A16 g64 checkpoint), MAX_LEN
# (40960), MAX_SEQS (8: llama-benchy's concurrency 8), KV (auto | fp8), MTP (0: no speculation;
# N: the checkpoint's MTP head with N speculative tokens), VLLM_ARGS (extra vllm serve flags),
# IMAGE (VLLM_IMAGE, default the oracle image), PORT (8012), AFFINITY (the container's
# ZE_AFFINITY_MASK, default 0), BENCHY_VERSION (pins llama-benchy), WAIT_S (1800: startup with
# graph capture and compilation).
# The serve flags are docs/BENCHMARKS.md's recorded baseline ("vLLM: how the baseline is served")
# with MAX_SEQS / KV / MTP on top. Prints the vLLM version the image carries (read it, never the tag).
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL="${MODEL:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}"
MAX_LEN="${MAX_LEN:-40960}"
MAX_SEQS="${MAX_SEQS:-8}"
KV="${KV:-auto}"
MTP="${MTP:-0}"
IMAGE="${IMAGE:-${VLLM_IMAGE:-vllm-xpu-env-next-p314-t216-vxkp0:latest}}"
PORT="${PORT:-8012}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
NAME="b70-vllm-benchy-$$"
export PATH="$HOME/.local/bin:$PATH"
spec=()
if [ "$MTP" != 0 ]; then
  spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
fi
exec 9>"$HOME/b70-gpu.lock"
echo "waiting for the GPU lock"; flock 9; echo "lock held"
uptime
# Not --rm: the container's log is kept in ~/benchy-vllm.log even when vLLM dies at startup.
trap 'docker logs "$NAME" > "$HOME/benchy-vllm.log" 2>&1 || true; docker rm -f "$NAME" > /dev/null 2>&1 || true' EXIT
# The HF cache is mounted at its own path as well, so a snapshot dir's blob links resolve.
# shellcheck disable=SC2086 # VLLM_ARGS is a flag list
docker run -d --name "$NAME" \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e ZE_AFFINITY_MASK="${AFFINITY:-0}" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZE_FLAT_HIERARCHY=FLAT \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e HF_HUB_OFFLINE=1 \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$HF_CACHE:$HF_CACHE:ro" \
  "$IMAGE" "$MODEL" \
  --served-model-name b70 \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len "$MAX_LEN" --kv-cache-dtype "$KV" --max-num-seqs "$MAX_SEQS" \
  --language-model-only --trust-remote-code \
  --compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}' \
  "${spec[@]}" ${VLLM_ARGS:-} > /dev/null
for _ in $(seq 1 "${WAIT_S:-1800}"); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || {
    echo "vLLM exited"; docker logs --tail 40 "$NAME" 2>&1 || true; exit 1; }
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "vLLM not up after ${WAIT_S:-1800} s"; exit 1; }
echo "server up"
docker exec "$NAME" python3 -c 'import vllm; print("vllm version", vllm.__version__)' || true
benchy=llama-benchy${BENCHY_VERSION:+@$BENCHY_VERSION}
echo "client: $benchy, server: vLLM kv $KV, mtp $MTP, max-num-seqs $MAX_SEQS, max-len $MAX_LEN ${VLLM_ARGS:-}"
uvx "$benchy" --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --served-model-name b70 --tokenizer "$MODEL" "$@"
uptime
