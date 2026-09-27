#!/bin/bash
# On the box, from a tree root: start b70-serve under the shared GPU lock, run
# llama-benchy against it, stop the server.
#   tools/probe/serve_benchy.sh [llama-benchy args...]
# e.g. tools/probe/detach.sh ~/benchy.log tools/probe/serve_benchy.sh --pp 4096 --tg 256
# Env: MODEL (repo id or snapshot; default the W4A16 g64 checkpoint), MAX_LEN
# (16384), PORT (8011), SERVE_ARGS (extra b70-serve flags).
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL="${MODEL:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}"
MAX_LEN="${MAX_LEN:-16384}"
PORT="${PORT:-8011}"
export PATH="$HOME/.local/bin:$PATH"
exec 9>"$HOME/b70-gpu.lock"
echo "waiting for the GPU lock"; flock 9; echo "lock held"
uptime
ZE_AFFINITY_MASK=0 build/src/cli/b70-serve "$MODEL" --max-len "$MAX_LEN" --port "$PORT" \
  --served-name b70 ${SERVE_ARGS:-} > "$HOME/benchy-serve.log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null || true' EXIT
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
  kill -0 $srv 2>/dev/null || { echo "server exited"; cat "$HOME/benchy-serve.log"; exit 1; }
  sleep 1
done
echo "server up"
uvx llama-benchy --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
  --served-model-name b70 --tokenizer "$MODEL" "$@"
uptime
