#!/usr/bin/env bash
# One arm of a server row, ON THE BOX, from a tree root, under the caller's GPU lock (this
# script does not take it): start b70-serve with the arm's flags, run serve_client.py over
# the prompt sets, stop the server.
#   tools/box_validate/serve_run.sh <out dir> <label> <round> <snapshot> [b70-serve flags...] \
#       -- [serve_client.py run flags...]
# Writes <out dir>/requests.jsonl (one line per request), <out dir>/server-log/ (the server's
# --log-requests records, which give the client decode-only t/s) and <out dir>/server.log;
# prints the client's per-request lines and the server's startup lines (memory, max_len,
# mtp / spec / draft vocab / prefix cache). Exit: the client's (0 when every request was
# answered), 1 when the server did not come up.
# Env: PORT (8013), WAIT (900 s for /v1/models), B70_SERVE (the binary; default this tree's
# build/src/cli/b70-serve - test_box_validate.py points it at a stand-in).
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 4 ] || { echo "usage: $0 <out dir> <label> <round> <snapshot> [serve flags] -- [client flags]" >&2; exit 2; }
out="$1" label="$2" round="$3" model="$4"
shift 4
serve=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do serve+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
port="${PORT:-8013}"
mkdir -p "$out/server-log"
bin="${B70_SERVE:-build/src/cli/b70-serve}"
echo "serve_run: [$label r$round] $bin $model --port $port --served-name b70 --log-requests $out/server-log ${serve[*]+${serve[*]}}"
"$bin" "$model" --port "$port" --served-name b70 --log-requests "$out/server-log" ${serve[@]+"${serve[@]}"} \
  > "$out/server.log" 2>&1 &
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
  python3 tools/box_validate/serve_client.py run --url "http://127.0.0.1:$port" --label "$label" \
    --round "$round" --server-log "$out/server-log" --out "$out/requests.jsonl" "$@"
  rc=$?
else
  echo "serve_run: the server did not come up"
fi
stop
trap - EXIT
echo "--- b70-serve [$label r$round], startup and per-request lines"
grep -E 'max_len|memory:|mtp|spec lookup|lookup:|draft vocab|prefix cache:|lm_head|error|refus' "$out/server.log" | grep -v '^prefix: ' | head -40
[ "$up" = 1 ] || tail -20 "$out/server.log"
echo "SERVE_RUN [$label r$round] rc=$rc"
exit "$rc"
