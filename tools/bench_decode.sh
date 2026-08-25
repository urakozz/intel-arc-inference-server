#!/usr/bin/env bash
# The recorded decode benchmark - the harness behind every b70-decode row in
# docs/BENCHMARKS.md.
#
#   tools/bench_decode.sh                        3 runs at depth 4096, tg 256
#   tools/bench_decode.sh --depth 64             the doc 07 #12 experiment
#   tools/bench_decode.sh --runs 1 --no-build    one run against what is already built
#
# Three things it exists to get right, none of which a bare ssh line does:
#
# 1. **The git sha.** `tools/box.sh sync` rsyncs the tree *without* `.git`, so
#    the box cannot name its own commit and `b70-decode --bench` would print
#    `unknown` in a row that is meant to be a record. The sha is read here, on
#    the Mac, and threaded to the remote process through its environment. A
#    dirty worktree is recorded as `<sha>-dirty`: a measurement of code that is
#    not the commit must not be filed under the commit.
# 2. **Repetition and the median.** One run is an anecdote. Three runs give the
#    median that gets recorded and the spread that says whether the median
#    means anything. Both are printed; neither is smoothed.
# 3. **The exact invocation, committed.** Whoever re-runs this in six months
#    gets the same depth, the same tg, the same checkpoint and the same
#    default --max-len, without reconstructing them from a doc.
#
# The box must be otherwise idle - no vLLM container, no other GPU work - or
# the number measures the contention instead. Env: BOX, REMOTE_DIR, JOBS (see
# tools/box.sh), MODEL, DEPTH, TG, RUNS.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ}"
DEPTH="${DEPTH:-4096}"
TG="${TG:-256}"
RUNS="${RUNS:-3}"
BUILD=1

while [ $# -gt 0 ]; do
  case "$1" in
    --depth) DEPTH="$2"; shift 2 ;;
    --tg)    TG="$2";    shift 2 ;;
    --runs)  RUNS="$2";  shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    -h|--help)
      awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "bench_decode.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ "$SHA" != unknown ] && ! git diff --quiet HEAD 2>/dev/null; then
  SHA="$SHA-dirty"
fi

if [ "$BUILD" = 1 ]; then
  tools/box.sh build
else
  tools/box.sh sync
fi

if ! [ "$RUNS" -ge 1 ] 2>/dev/null; then
  echo "bench_decode.sh: --runs must be a positive integer, got '$RUNS'" >&2
  exit 2
fi

echo "bench: $MODEL, depth $DEPTH, tg $TG, $RUNS run(s), sha $SHA" >&2

# `rows` is only ever expanded after at least one run, which the --runs guard
# above enforces -- an empty array would trip `set -u` on older bash (3.2, which
# is what macOS ships).
rows=()
for i in $(seq 1 "$RUNS"); do
  echo "=== run $i/$RUNS ===" >&2
  # box.sh run joins its arguments and executes them remotely; B70_GIT_SHA as a
  # command prefix is what puts the Mac's sha in the remote process.
  row="$(tools/box.sh run "B70_GIT_SHA='$SHA' ./build/src/cli/b70-decode '$MODEL' --bench --depth $DEPTH --tg $TG")"
  echo "$row"
  rows+=("$row")
done

printf '%s\n' "${rows[@]}" | awk -F'|' '
  { ts[NR] = $5 + 0; mspt[NR] = $6 + 0 }
  END {
    if (NR < 1) exit 0
    for (i = 1; i <= NR; i++) for (j = i + 1; j <= NR; j++)
      if (ts[j] < ts[i]) { t = ts[i]; ts[i] = ts[j]; ts[j] = t
                           t = mspt[i]; mspt[i] = mspt[j]; mspt[j] = t }
    mid = int((NR + 1) / 2)
    # The parentheses around the ternary are load-bearing, and this is not a
    # macOS quirk: in POSIX awk grammar a bare `>` inside a print/printf
    # argument list parses as an output redirection, so `a > 0 ? b : c` is
    # ambiguous there. Hoisting it into its own assignment settles it portably.
    # (Getting this wrong is what made the first run of this script print every
    # row and then die at the summary -- the rows were fine, the median was not.)
    pct = (ts[mid] > 0) ? (100 * (ts[NR] - ts[1]) / ts[mid]) : 0
    printf "median %.2f t/s (%.2f ms/token) over %d run(s); min %.2f, max %.2f, spread %.2f (%.2f%%)\n",
           ts[mid], mspt[mid], NR, ts[1], ts[NR], ts[NR] - ts[1], pct
  }' >&2
