#!/bin/bash
# On the box, from a tree root: C3. Replay a request log against b70-serve with the prefix
# cache on and then off (same build, max_len 131072, device 0), under the shared GPU lock,
# and compare the two runs.
#   tools/prefix/replay_ab.sh <log dir> <out prefix>
# e.g. tools/probe/detach.sh ~/c3.log tools/prefix/replay_ab.sh tests/golden/opencode/session1 ~/c3
# Writes <out prefix>.on.json, .off.json, .on.serve.log, .off.serve.log. Env: MODEL, PORT
# (8012), GB (the cache-on budget, 32), ARMS ("on off"), SERVE_ARGS (extra b70-serve flags,
# e.g. "--log-requests DIR").
set -euo pipefail
cd "$(dirname "$0")/../.."
log="$1"
out="$2"
MODEL="${MODEL:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}"
PORT="${PORT:-8012}"
GB="${GB:-32}"
exec 9>"$HOME/b70-gpu.lock"
echo "waiting for the GPU lock"; flock 9; echo "lock held"
for arm in ${ARMS:-on off}; do
  gb=$GB; [ "$arm" = off ] && gb=0
  echo "== cache $arm (--prefix-cache-gb $gb) =="; uptime
  ZE_AFFINITY_MASK=0 build/src/cli/b70-serve "$MODEL" --max-len 131072 --port "$PORT" \
    --served-name b70 --prefix-cache-gb "$gb" ${SERVE_ARGS:-} > "$out.$arm.serve.log" 2>&1 &
  srv=$!
  for _ in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    kill -0 $srv 2>/dev/null || { echo "server exited"; tail -20 "$out.$arm.serve.log"; exit 1; }
    sleep 1
  done
  python3 tools/prefix/replay_log.py run "$log" "http://127.0.0.1:$PORT" "$out.$arm.json"
  kill $srv; wait $srv 2>/dev/null || true
  uptime
done
if [ "${ARMS:-on off}" = "on off" ]; then
  python3 tools/prefix/replay_log.py compare "$out.on.json" "$out.off.json"
fi
