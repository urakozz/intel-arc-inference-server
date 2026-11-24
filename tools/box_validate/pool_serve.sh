#!/usr/bin/env bash
# One server run of the KV page pool's checks (plan 31 Task 7), ON THE BOX, from a tree root, under the caller's GPU
# lock (this script does not take it): start b70-serve with the run's flags, run one tools/box_validate/pool_serve.py
# case over it, stop the server.
#   tools/box_validate/pool_serve.sh <out dir> <case> <snapshot> [b70-serve flags...] [-- pool_serve.py flags]
# Writes <out dir>/<case>.json (the case's ids) and <out dir>/server.log; prints the case's lines and the server's
# slots / pool / eviction lines. Env: PORT (8013), WAIT (900 s), B70_SERVE (default this tree's build/src/cli/b70-serve;
# another build for a reference).
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 3 ] || { echo "usage: $0 <out dir> <case> <snapshot> [serve flags] -- [case flags]" >&2; exit 2; }
out="$1" case="$2" model="$3"
shift 3
serve=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do serve+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
port="${PORT:-8013}"
mkdir -p "$out"
bin="${B70_SERVE:-build/src/cli/b70-serve}"
echo "pool_serve: [$case] $bin $model --port $port ${serve[*]+${serve[*]}}"
"$bin" "$model" --port "$port" ${serve[@]+"${serve[@]}"} > "$out/server.log" 2>&1 &
srv=$!
stop() { tools/box_stop.sh child "$srv"; wait "$srv" 2>/dev/null; }
trap stop EXIT
up=0
for _ in $(seq 1 "${WAIT:-900}"); do
  if curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then up=1; break; fi
  kill -0 "$srv" 2>/dev/null || break
  sleep 1
done
rc=1
if [ "$up" = 1 ]; then
  python3 tools/box_validate/pool_serve.py "$case" --url "http://127.0.0.1:$port" --out "$out/$case.json" "$@"
  rc=$?
else
  echo "pool_serve: the server did not come up"
fi
stop
trap - EXIT
grep -E '^slots:|kv pool|max_len|left the card|refus|error' "$out/server.log" | head -20
[ "$up" = 1 ] || tail -20 "$out/server.log"
echo "POOL_SERVE [$case] rc=$rc"
exit "$rc"
