#!/usr/bin/env bash
# One arm of r78.speed (plan 13c Task 5), ON THE BOX, from a tree root, under the caller's GPU lock (this script does
# not take it): start b70-serve with the arm's flags, run tools/box_validate/slots_bench.py over it (1 / 2 / 4 / 8
# requests at once by default), stop the server.
#   tools/box_validate/slots_serve_bench.sh <out dir> <snapshot> <label> [b70-serve flags...] [-- slots_bench.py flags]
# Writes <out dir>/server-log/ (the --log-requests records the bench reads), <out dir>/server.log and
# <out dir>/bench.jsonl; prints the bench's lines and the server's startup lines. Exit: the bench's, 1 when the server
# did not come up. Env: PORT (8013), WAIT (900 s), B70_SERVE (default this tree's build/src/cli/b70-serve).
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 3 ] || { echo "usage: $0 <out dir> <snapshot> <label> [serve flags] [-- bench flags]" >&2; exit 2; }
out="$1" model="$2" label="$3"
shift 3
serve=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do serve+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
port="${PORT:-8013}"
mkdir -p "$out/server-log"
bin="${B70_SERVE:-build/src/cli/b70-serve}"
echo "slots_serve_bench: [$label] $bin $model --port $port --log-requests $out/server-log ${serve[*]+${serve[*]}}"
"$bin" "$model" --port "$port" --log-requests "$out/server-log" ${serve[@]+"${serve[@]}"} > "$out/server.log" 2>&1 &
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
  python3 tools/box_validate/slots_bench.py --url "http://127.0.0.1:$port" --server-log "$out/server-log" \
    --label "$label" --out "$out/bench.jsonl" "$@"
  rc=$?
else
  echo "slots_serve_bench: the server did not come up"
fi
stop
trap - EXIT
echo "--- b70-serve [$label], startup lines"
grep -E 'slots:|max_len|memory:|mtp|kv cache|error|refus' "$out/server.log" | grep -v '^prefix: \|^request: ' | head -20
[ "$up" = 1 ] || tail -20 "$out/server.log"
exit "$rc"
