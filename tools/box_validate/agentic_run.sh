#!/usr/bin/env bash
# A recorded coding session as a live agent loop through b70-serve, ON THE BOX, from a tree root, under the caller's GPU
# lock (this script does not take it): start b70-serve with the arm's flags, run tools/prefix/agentic_loop.py over the
# transcript, stop the server, print the report (spec 24e Task 5: cached_tokens, TTFT, decode t/s, the hit rate and the turns
# whose call the template re-renders in another order than the model generated it).
#   tools/box_validate/agentic_run.sh <out dir> <snapshot> <transcript.json> [b70-serve flags...]
# Writes <out dir>/run.json (one record per request), <out dir>/server-log/ (the server's --log-requests records, which give the
# report the ideal prefix reuse), <out dir>/server.log and <out dir>/report.md. Exit: the loop's (0 when it ran), 1 when the
# server did not come up.
# Env: PORT (8013), WAIT (900 s for /v1/models), TURNS (12), MAX_TOKENS (1024), B70_SERVE (the binary; default this tree's
# build/src/cli/b70-serve - test_box_validate.py points it at a stand-in).
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 3 ] || { echo "usage: $0 <out dir> <snapshot> <transcript.json> [b70-serve flags...]" >&2; exit 2; }
out="$1" model="$2" transcript="$3"
shift 3
port="${PORT:-8013}"
mkdir -p "$out/server-log"
bin="${B70_SERVE:-build/src/cli/b70-serve}"
echo "agentic_run: $bin $model --port $port --served-name b70 --log-requests $out/server-log $*"
"$bin" "$model" --port "$port" --served-name b70 --log-requests "$out/server-log" "$@" > "$out/server.log" 2>&1 &
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
  python3 tools/prefix/agentic_loop.py run "$transcript" "http://127.0.0.1:$port" "$out/run.json" \
    --turns "${TURNS:-12}" --max-tokens "${MAX_TOKENS:-1024}"
  rc=$?
else
  echo "agentic_run: the server did not come up"
fi
stop
trap - EXIT
if [ "$rc" = 0 ]; then
  python3 tools/prefix/agentic_loop.py report "$out/run.json" "$out/server-log" | tee "$out/report.md"
fi
echo "--- b70-serve, startup lines"
grep -E 'max_len|memory:|prefix cache:|lm_head|error|refus|b70-serve:' "$out/server.log" | grep -v '^prefix: ' | head -20
[ "$up" = 1 ] || tail -20 "$out/server.log"
echo "AGENTIC_RUN rc=$rc"
exit "$rc"
