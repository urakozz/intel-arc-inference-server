#!/usr/bin/env bash
# Start b70-serve, prove it serves, stop it - ON THE BOX, from a tree root, under the
# caller's GPU lock (this script does not take it).
#   tools/box_validate/serve_probe.sh <snapshot or repo> [b70-serve flags...]
# Waits for /v1/models (WAIT seconds, default 900: a load at --max-len auto plans first),
# sends one short greedy chat request, records the device's memory while the server is up
# (xpu-smi, when installed), stops the server and prints its log - the startup lines carry
# what the rows ask for (`max_len: auto -> N`, `memory: ...`, `prefix cache: ...`,
# `draft vocab ...`). Exit 0 when the request was answered.
# Env: PORT (8013), WAIT (900), DEVICE (0, for xpu-smi), SEEDED=1 (also send one seeded
# sampled request twice; the two completions must be identical).
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 1 ] || { echo "usage: $0 <snapshot> [b70-serve flags...]" >&2; exit 2; }
model="$1"; shift
port="${PORT:-8013}"
log="${TMPDIR:-/tmp}/b70-serve-probe.$$.log"
echo "serve_probe: build/src/cli/b70-serve $model --port $port --served-name b70 $*"
build/src/cli/b70-serve "$model" --port "$port" --served-name b70 "$@" > "$log" 2>&1 &
srv=$!
stop() { kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; }
trap stop EXIT
up=0
for _ in $(seq 1 "${WAIT:-900}"); do
  if curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then up=1; break; fi
  kill -0 "$srv" 2>/dev/null || break
  sleep 1
done
rc=1
if [ "$up" = 1 ]; then
  resp=$(curl -sf -m 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"b70","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0}')
  rc=$?
  echo "response: ${resp:-<none>}"
  if [ "${SEEDED:-0}" = 1 ] && [ "$rc" = 0 ]; then
    # a seeded sampled request twice: the two completions must be bitwise identical
    body='{"model":"b70","messages":[{"role":"user","content":"Write two sentences about the sea."}],"max_tokens":64,"temperature":1.0,"top_p":0.95,"seed":1234}'
    one=$(curl -sf -m 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" \
          | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["choices"][0]["message"]))')
    two=$(curl -sf -m 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" \
          | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["choices"][0]["message"]))')
    echo "seeded 1: $one"
    echo "seeded 2: $two"
    if [ -n "$one" ] && [ "$one" = "$two" ]; then echo "SEEDED identical"; else echo "SEEDED DIFFER"; rc=1; fi
  fi
  if command -v xpu-smi > /dev/null 2>&1; then
    echo "device memory while serving (xpu-smi stats -d ${DEVICE:-0}):"
    xpu-smi stats -d "${DEVICE:-0}" 2>/dev/null | grep -iE 'memory' | sed 's/^/  /'
  fi
else
  echo "serve_probe: the server did not come up"
fi
stop
trap - EXIT
echo "--- b70-serve log"
cat "$log"
rm -f "$log"
if [ "$rc" = 0 ]; then echo "SERVE_PROBE OK"; else echo "SERVE_PROBE FAILED"; fi
[ "$rc" = 0 ]
