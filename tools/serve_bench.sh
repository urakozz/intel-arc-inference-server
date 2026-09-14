#!/usr/bin/env bash
# tools/serve_bench.sh -- starts b70-serve, runs the standing llama-benchy
# command against it three times, stops the server, and prints the tg256
# rows, the coherence answer, and a derived HTTP-inclusive pp figure (plan
# 7d Task 3: bar 4 -- llama-benchy end to end -- and the inputs bar 5 and the
# HTTP-prefill-cost derivation need).
#
#   tools/serve_bench.sh                 build, 3 llama-benchy invocations
#   tools/serve_bench.sh --no-build      skip the box build (sync only)
#   tools/serve_bench.sh --runs 5        5 invocations instead of 3
#
# **Why pp is derived from e2e_ttft, not llama-benchy's own pp{N} row.**
# llama-benchy's `pp_throughput` is `prompt_tokens / est_ppt`, and
# `est_ppt = ttfr - latency`, where `ttfr` is the arrival time of the FIRST
# streamed SSE frame that carries `choices` at all. For a compliant OpenAI
# chat stream, `b70-serve` (src/server/server.cc, protocol_test.cc case 5's
# anti-buffering bar) writes the role-delta frame BEFORE `generate()` even
# starts -- correct, tested protocol behaviour -- so `ttfr` measures "time to
# the role frame", not "time to first generated token", and `est_ppt` comes
# out sub-millisecond and the printed pp figure is a many-million-t/s
# artifact (measured: 9.8M +/- 5.5M t/s in the probe run that motivated this
# comment). llama-benchy's OWN `e2e_ttft` field (`first_token_ts - start_ts`,
# gated on a non-empty `delta.content`) is exactly "time to the first real
# generated token" -- the same first-token-inclusive window
# tools/bench_decode.sh's `--pp` row uses on the device side -- so
# `pp_http = prompt_tokens / (e2e_ttft_ms / 1000)` is the comparable,
# HTTP-inclusive number. Both figures are recorded; the llama-benchy pp
# figure is kept for the record but is not the number bar 5's arithmetic
# uses.
#
# Adapted to the box's unreliable WiFi (fact recorded in the T4 dispatch):
# every step that can run longer than about a minute -- the server itself,
# and each llama-benchy invocation -- is launched detached (setsid nohup) on
# the box and polled with short, separately-connecting ssh calls, so a
# mid-run WiFi drop costs at most one retry rather than the whole session.
#
# Env: MODEL (default: the gate checkpoint), PORT (8000), SERVED_NAME (b70),
#      RUNS (3 llama-benchy invocations; each is itself 3 runs -- llama-
#      benchy's own --runs default), BOX, REMOTE_DIR, JOBS (tools/box.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/home/user/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-b70}"
RUNS="${RUNS:-3}"
BUILD=1
BOX="${BOX:-user@box}"
REMOTE_DIR="${REMOTE_DIR:-b70-inference-server}"
SSH="ssh -n -o BatchMode=yes"

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --runs) RUNS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    -h|--help)
      awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "serve_bench.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ "$SHA" != unknown ] && ! git diff --quiet HEAD 2>/dev/null; then
  SHA="$SHA-dirty"
fi

idle_check() {
  $SSH "$BOX" '
    n=$(docker ps -q 2>/dev/null | wc -l)
    holders=""
    for f in /proc/*/fdinfo/*; do
      grep -l drm-driver "$f" >/dev/null 2>&1 || continue
      pid=$(echo "$f" | cut -d/ -f3)
      holders="$holders $pid:$(cat /proc/$pid/comm 2>/dev/null)"
    done
    echo "docker=$n drm_holders=[$holders]"
    if [ "$n" -eq 0 ] && [ -z "$holders" ]; then echo "GRADE=RECORD"; else echo "GRADE=ITERATE"; fi'
}

echo "bench: b70-serve vs llama-benchy, sha $SHA, model $MODEL, $RUNS invocation(s)" >&2
echo "=== idle proof (before) ===" >&2
idle_check | tee /dev/stderr | grep -q 'GRADE=RECORD' && BEFORE_GRADE=RECORD || BEFORE_GRADE=ITERATE

if [ "$BUILD" = 1 ]; then
  tools/box.sh build
else
  tools/box.sh sync
fi

echo "=== starting b70-serve (detached, device 0 default) ===" >&2
PID="$($SSH "$BOX" "cd '$REMOTE_DIR' && setsid nohup ./build/src/cli/b70-serve '$MODEL' --port $PORT --served-name '$SERVED_NAME' > \$HOME/b70-serve-bench.log 2>&1 < /dev/null & echo \$!")"
echo "server pid: $PID" >&2

PID_STILL_RUNNING=1
cleanup() {
  if [ "$PID_STILL_RUNNING" = 1 ]; then
    echo "=== stopping b70-serve (pid $PID) ===" >&2
    $SSH "$BOX" "kill $PID 2>/dev/null || true"
    sleep 1
    $SSH "$BOX" "kill -0 $PID 2>/dev/null && kill -9 $PID || true"
    PID_STILL_RUNNING=0
  fi
}
trap cleanup EXIT

echo "=== waiting for /v1/models ===" >&2
ready=0
for i in $(seq 1 60); do
  if $SSH "$BOX" "curl -s -m 2 http://127.0.0.1:$PORT/v1/models 2>/dev/null" | grep -q '"object":"list"'; then
    ready=1
    break
  fi
  sleep 2
done
[ "$ready" = 1 ] || { echo "serve_bench.sh: server never answered /v1/models -- see \$HOME/b70-serve-bench.log on the box" >&2; exit 1; }
echo "server ready" >&2

echo "=== coherence probe: What is the capital of France? ===" >&2
COHERENCE_ANSWER="$($SSH "$BOX" "curl -s -m 30 http://127.0.0.1:$PORT/v1/chat/completions -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32}'" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "(parse failed)")"
echo "answer: $COHERENCE_ANSWER" >&2

tg_vals=()
e2e_vals=()
ppraw_vals=()
for i in $(seq 1 "$RUNS"); do
  echo "=== llama-benchy invocation $i/$RUNS (detached) ===" >&2
  remote_log="\$HOME/serve_bench_$i.log"
  remote_json="\$HOME/serve_bench_$i.json"
  # -n so llama-benchy's own --runs 3 is honoured (median-over-3 inside this
  # one invocation, matching how the vLLM rows in docs/BENCHMARKS.md were
  # taken); the invocation itself runs detached because it can exceed a
  # minute (warmup + coherence + latency probe + 3 timed runs).
  BENCH_PID="$($SSH "$BOX" "cd '$REMOTE_DIR' && setsid nohup ~/.local/bin/uvx llama-benchy --base-url http://127.0.0.1:$PORT/v1 --model '$SERVED_NAME' --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation --format json --save-result $remote_json > $remote_log 2>&1 < /dev/null & echo \$!")"
  echo "  llama-benchy pid: $BENCH_PID" >&2
  done_flag=0
  for _ in $(seq 1 90); do  # up to ~4.5 min per invocation
    if $SSH "$BOX" "test -f $remote_json"; then
      done_flag=1
      break
    fi
    if ! $SSH "$BOX" "kill -0 $BENCH_PID 2>/dev/null"; then
      # process exited without producing a result file: a real failure
      break
    fi
    sleep 3
  done
  [ "$done_flag" = 1 ] || {
    echo "serve_bench.sh: invocation $i did not produce $remote_json -- log follows" >&2
    $SSH "$BOX" "cat $remote_log" >&2 || true
    exit 1
  }
  $SSH "$BOX" "grep -q 'Coherence test PASSED' $remote_log" || {
    echo "serve_bench.sh: invocation $i's coherence test did not pass -- log follows" >&2
    $SSH "$BOX" "cat $remote_log" >&2
    exit 1
  }
  read -r tg e2e ppraw < <($SSH "$BOX" "jq -r '[.benchmarks[0].tg_throughput.mean, .benchmarks[0].e2e_ttft.mean, .benchmarks[0].pp_throughput.mean] | @tsv' $remote_json")
  echo "  tg256 = $tg t/s, e2e_ttft = $e2e ms, llama-benchy's own (unusable) pp = $ppraw t/s" >&2
  tg_vals+=("$tg")
  e2e_vals+=("$e2e")
  ppraw_vals+=("$ppraw")
done

cleanup
trap - EXIT

echo "=== idle proof (after) ===" >&2
idle_check | tee /dev/stderr | grep -q 'GRADE=RECORD' && AFTER_GRADE=RECORD || AFTER_GRADE=ITERATE

GRADE=ITERATE
[ "$BEFORE_GRADE" = RECORD ] && [ "$AFTER_GRADE" = RECORD ] && GRADE=RECORD
echo "grade: before=$BEFORE_GRADE after=$AFTER_GRADE -> $GRADE" >&2

printf '%s\n' "${tg_vals[@]}" "${e2e_vals[@]}" "${ppraw_vals[@]}" | awk -v n="$RUNS" -v grade="$GRADE" -v sha="$SHA" '
  NR <= n            { tg[NR] = $1 + 0 }
  NR > n && NR <= 2*n { e2e[NR-n] = $1 + 0 }
  NR > 2*n           { ppraw[NR-2*n] = $1 + 0 }
  END {
    for (i = 1; i <= n; i++) for (j = i+1; j <= n; j++) if (tg[j] < tg[i]) { t=tg[i]; tg[i]=tg[j]; tg[j]=t }
    for (i = 1; i <= n; i++) for (j = i+1; j <= n; j++) if (e2e[j] < e2e[i]) { t=e2e[i]; e2e[i]=e2e[j]; e2e[j]=t }
    mid = int((n + 1) / 2)
    tg_med = tg[mid]; e2e_med = e2e[mid]
    pp_med = 4096 / (e2e_med / 1000)
    printf "sha %s -- %s grade\n", sha, grade
    printf "tg256 http median %.2f t/s over %d invocation(s); min %.2f, max %.2f\n", tg_med, n, tg[1], tg[n]
    printf "e2e_ttft (pp4096, first-token-inclusive) median %.2f ms over %d invocation(s); min %.2f, max %.2f\n", e2e_med, n, e2e[1], e2e[n]
    printf "pp4096 http (derived from e2e_ttft) median %.2f t/s\n", pp_med
  }' >&2

echo "server-log-only (Task 5) generation timing lines, if any, are in \$HOME/b70-serve-bench.log on the box" >&2
