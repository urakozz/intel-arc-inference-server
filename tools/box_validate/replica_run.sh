#!/usr/bin/env bash
# One arm of row 77 (plan 13b, b70-serve --parallel replica), ON THE BOX, from a tree root, under the caller's GPU
# lock (this script does not take it): start b70-serve on the cards ZE_AFFINITY_MASK names with the arm's flags, run
# tools/box_validate/replica_check.py run over it, stop the server.
#   tools/box_validate/replica_run.sh <out dir> <cards> <snapshot> [b70-serve flags...] -- [replica_check.py run flags]
# <cards>: the ZE_AFFINITY_MASK value (0,1 for --parallel replica; 0 or 1 for a one-card arm).
# Writes <out dir>/requests.jsonl, <out dir>/server-log/ (the --log-requests records: the slot that served each
# request) and <out dir>/server.log; prints the client's lines and the server's startup lines. Exit: the client's,
# 1 when the server did not come up.
# Env: PORT (8013), WAIT (1200 s for /v1/models: replica loads the model twice), B70_SERVE (default this tree's
# build/src/cli/b70-serve).
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 3 ] || { echo "usage: $0 <out dir> <cards> <snapshot> [serve flags] -- [client flags]" >&2; exit 2; }
out="$1" cards="$2" model="$3"
shift 3
serve=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do serve+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
port="${PORT:-8013}"
mkdir -p "$out/server-log"
bin="${B70_SERVE:-build/src/cli/b70-serve}"
echo "replica_run: ZE_AFFINITY_MASK=$cards $bin $model --port $port --log-requests $out/server-log ${serve[*]+${serve[*]}}"
ZE_AFFINITY_MASK="$cards" "$bin" "$model" --port "$port" --log-requests "$out/server-log" ${serve[@]+"${serve[@]}"} \
  > "$out/server.log" 2>&1 &
srv=$!
stop() { kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; }
trap stop EXIT
up=0
for _ in $(seq 1 "${WAIT:-1200}"); do
  if curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then up=1; break; fi
  kill -0 "$srv" 2>/dev/null || break
  sleep 1
done
rc=1
if [ "$up" = 1 ]; then
  python3 tools/box_validate/replica_check.py run --url "http://127.0.0.1:$port" --out "$out/requests.jsonl" "$@"
  rc=$?
else
  echo "replica_run: the server did not come up"
fi
stop
trap - EXIT
echo "--- b70-serve [ZE_AFFINITY_MASK=$cards], startup lines"
grep -E 'devices:|card [01]:|max_len|memory:|mtp|spec lookup|prefix cache:|replica|error|refus' "$out/server.log" | grep -v '^prefix: ' | head -40
[ "$up" = 1 ] || tail -20 "$out/server.log"
echo "REPLICA_RUN [$cards] rc=$rc"
exit "$rc"
