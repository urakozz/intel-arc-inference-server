#!/usr/bin/env bash
# The recorded decode benchmark - the harness behind every b70-decode row in
# docs/BENCHMARKS.md.
#
#   tools/bench_decode.sh                        3 runs at depth 4096, tg 256
#   tools/bench_decode.sh --depth 64             the doc 07 #12 experiment
#   tools/bench_decode.sh --runs 1 --no-build    one run against what is already built
#   tools/bench_decode.sh --prefill-length 4096  3 runs of spec 2's PREFILL at kC = 2048
#   tools/bench_decode.sh --prefill-length 4096 --prefill-chunk 1024
#   tools/bench_decode.sh --prefill-length 4096 --prefill-backend l0    spec 2.1's L0 GEMM backend
#   tools/bench_decode.sh --depth 4096 --max-len 131072 spec 6: the 128k decode variants
#   B70_PREFILL_ATTN=composed tools/bench_decode.sh --prefill-length 4096   spec 6's reference attention
#   tools/bench_decode.sh --lm-head int8         spec 9: the int8 lm_head (rows say int8-head)
#
# `--prefill-length N` replaces `--depth N` with `Engine::prefill` (spec 2, plan 6e Task 1)
# and makes the binary print a SECOND markdown row -- `| ... <backend> pp | N |
# C | ms | t/s |` -- beside the unchanged tg row. This script then medians
# BOTH: the tg row on t/s as it always has, and the pp row on its own t/s. It
# is exclusive with --depth for the same reason the CLI refuses the pair: the
# prefilled ids ARE the depth. `--prefill-backend sycl-tla|l0|l0-int8` (spec 2.1 §3.5, spec 5)
# belongs to --prefill-length the same way --prefill-chunk does, and both PP_BACKEND (env) and
# --prefill-backend (flag) are refused without it. (These were --pp, --pp-chunk and --pp-backend
# until 2026-10-06, when b70-decode's --pp became vLLM's pipeline-parallel size; the old
# spellings are refused here naming the new ones.)
#
# **The grade is a property of the box, not of this script, so this script
# measures it instead of asserting it.** Record grade needs a provably idle
# box: zero containers and zero DRM fd holders on every card. From 2026-09-04
# to 2026-09-09 that was never true - two desktop daemons (baobab, ptyxis)
# re-acquired render-node fds every few minutes, and every row taken in that
# window says ITERATE. They were killed and the box rebooted on 2026-09-09, and
# the check below now prints RECORD when it finds nothing holding the GPU.
# Quote whichever word it printed; never upgrade a row's grade after the fact.
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
# tools/box.sh), MODEL, DEPTH, TG, RUNS, MAX_LEN, ZE_AFFINITY_MASK, B70_PREFILL_ATTN.
#
# `B70_PREFILL_ATTN` (spec 6: `flash`, the default, or `composed`) is forwarded exactly
# like ZE_AFFINITY_MASK below, for the same reason: ssh carries no environment. `--max-len`
# (or MAX_LEN) is passed through to b70-decode's own flag; unset adds nothing, so the
# invocation is byte-for-byte the one every earlier row used.
#
# `ZE_AFFINITY_MASK` needs the same treatment as the sha and for the same
# reason: `ssh` carries no environment, so until now every row this harness
# took ran on the box's default device 0 no matter what the caller exported -
# silently, which is the bad kind of wrong. Set it on the Mac and it is written
# into the remote command line; leave it unset and nothing is added, which is
# byte-for-byte the invocation every earlier row used. The device that ran is
# printed in the bench header so it lands in the log beside the number.
set -euo pipefail
# Machine address. Set BOX in the environment, or put it in tools/box.env
# (untracked; copy tools/box.env.example). There is no default: a wrong
# address should fail loudly rather than quietly talk to the wrong machine.
_box_env="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/box.env"
# shellcheck source=/dev/null
[ -r "$_box_env" ] && . "$_box_env"
: "${BOX:?set BOX=user@host in the environment or in tools/box.env (see tools/box.env.example)}"
cd "$(dirname "$0")/.."

MODEL="${MODEL:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}"
DEPTH="${DEPTH:-4096}"
TG="${TG:-256}"
RUNS="${RUNS:-3}"
PP="${PP:-}"
PP_CHUNK="${PP_CHUNK:-}"
PP_BACKEND="${PP_BACKEND:-}"
MAX_LEN="${MAX_LEN:-}"
LM_HEAD="${LM_HEAD:-}"
BUILD=1

# `set -u` is on, so a bare `--depth` at the end of the line would abort with
# `$2: unbound variable` -- bash's message about this script's internals, not
# this script's message about the operator's command. Every value-taking flag
# goes through this first, so a missing (or empty) value takes the same exit-2
# path as an unknown argument. `need_value "$@"` sees the flag as $1 and its
# value, if there is one, as $2; the `||` short-circuits so `-z "$2"` is never
# reached when $2 does not exist.
need_value() {
  if [ "$#" -lt 2 ] || [ -z "$2" ]; then
    echo "bench_decode.sh: $1 needs a value" >&2
    exit 2
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --depth) need_value "$@"; DEPTH="$2"; DEPTH_SET=1; shift 2 ;;
    --prefill-length)  need_value "$@"; PP="$2";       shift 2 ;;
    --prefill-chunk)   need_value "$@"; PP_CHUNK="$2"; shift 2 ;;
    --prefill-backend) need_value "$@"; PP_BACKEND="$2"; shift 2 ;;
    --pp|--pp-chunk|--pp-backend)
      case "$1" in
        --pp) new="--prefill-length N (b70-decode's --pp is the pipeline-parallel size now)" ;;
        --pp-chunk) new="--prefill-chunk C" ;;
        *) new="--prefill-backend B" ;;
      esac
      echo "bench_decode.sh: $1 was renamed on 2026-10-06: $new" >&2
      exit 2 ;;
    --tg)    need_value "$@"; TG="$2";    shift 2 ;;
    --runs)  need_value "$@"; RUNS="$2";  shift 2 ;;
    --model) need_value "$@"; MODEL="$2"; shift 2 ;;
    --max-len) need_value "$@"; MAX_LEN="$2"; shift 2 ;;
    --lm-head) need_value "$@"; LM_HEAD="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    -h|--help)
      awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "bench_decode.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# Everything the arguments alone can decide is decided before the box is
# touched. `--runs 0` used to be caught *after* a full remote rebuild, which
# spent a couple of minutes on the box to reject four characters.
if ! [ "$RUNS" -ge 1 ] 2>/dev/null; then
  echo "bench_decode.sh: --runs must be a positive integer, got '$RUNS'" >&2
  exit 2
fi
# The prefill arguments are validated here as well as in the CLI, for the same
# reason --runs is: a rejection that costs a two-minute remote rebuild first is
# a worse harness than one that costs nothing.
if [ -n "$PP" ]; then
  if ! [ "$PP" -ge 1 ] 2>/dev/null; then
    echo "bench_decode.sh: --prefill-length must be a positive integer, got '$PP'" >&2
    exit 2
  fi
  if [ -n "${DEPTH_SET:-}" ]; then
    echo "bench_decode.sh: --prefill-length and --depth are exclusive (the prefilled ids ARE the depth)" >&2
    exit 2
  fi
elif [ -n "$PP_CHUNK" ]; then
  echo "bench_decode.sh: --prefill-chunk belongs to --prefill-length" >&2
  exit 2
elif [ -n "$PP_BACKEND" ]; then
  echo "bench_decode.sh: --prefill-backend belongs to --prefill-length" >&2
  exit 2
fi

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ "$SHA" != unknown ] && ! git diff --quiet HEAD 2>/dev/null; then
  SHA="$SHA-dirty"
fi

if [ "$BUILD" = 1 ]; then
  tools/box.sh build
else
  tools/box.sh sync
fi

# An empty prefix when the caller sets nothing: the remote command line then
# matches every row taken before this flag existed.
AFFINITY=""
if [ -n "${ZE_AFFINITY_MASK:-}" ]; then
  AFFINITY="ZE_AFFINITY_MASK='$ZE_AFFINITY_MASK' "
fi
if [ -n "${B70_PREFILL_ATTN:-}" ]; then
  AFFINITY="${AFFINITY}B70_PREFILL_ATTN='$B70_PREFILL_ATTN' "
fi
MAXLEN_ARGS=""
[ -n "$MAX_LEN" ] && MAXLEN_ARGS=" --max-len $MAX_LEN"
# Spec 9: unset adds nothing (b70-decode's default is the checkpoint's bf16 head).
[ -n "$LM_HEAD" ] && MAXLEN_ARGS="$MAXLEN_ARGS --lm-head $LM_HEAD"

if [ -n "$PP" ]; then
  MODE_ARGS="--prefill-length $PP"
  [ -n "$PP_CHUNK" ] && MODE_ARGS="$MODE_ARGS --prefill-chunk $PP_CHUNK"
  [ -n "$PP_BACKEND" ] && MODE_ARGS="$MODE_ARGS --prefill-backend $PP_BACKEND"
  echo "bench: $MODEL, PREFILL $PP ids (chunk ${PP_CHUNK:-default kC}), tg $TG, $RUNS run(s), sha $SHA, device ${ZE_AFFINITY_MASK:-unset (box default 0)}, backend ${PP_BACKEND:-build default}, attention ${B70_PREFILL_ATTN:-default (flash)}, max_len ${MAX_LEN:-default}" >&2
  echo "       iterate grade unless the box is provably idle -- see this script's header" >&2
else
  MODE_ARGS="--depth $DEPTH"
  echo "bench: $MODEL, depth $DEPTH, tg $TG, $RUNS run(s), sha $SHA, device ${ZE_AFFINITY_MASK:-unset (box default 0)}, max_len ${MAX_LEN:-default}" >&2
fi

# `rows` is only ever expanded after at least one run, which the --runs guard
# (now above the build, so it costs nothing to trip) enforces -- an empty array
# would trip `set -u` on older bash (3.2, which is what macOS ships).
rows=()
pp_rows=()
for i in $(seq 1 "$RUNS"); do
  echo "=== run $i/$RUNS ===" >&2
  # box.sh run joins its arguments and executes them remotely; B70_GIT_SHA as a
  # command prefix is what puts the Mac's sha in the remote process. Both stdout
  # rows come back together, so they are split on the ` pp |` marker the CLI
  # writes rather than on line order.
  out="$(tools/box.sh run "${AFFINITY}B70_GIT_SHA='$SHA' ./build/src/cli/b70-decode '$MODEL' --bench $MODE_ARGS --tg $TG$MAXLEN_ARGS")"
  echo "$out"
  while IFS= read -r line; do
    case "$line" in
      *" pp |"*) pp_rows+=("$line") ;;
      "| b70-decode "*) rows+=("$line") ;;
    esac
  done <<< "$out"
done

# The pp row first when there is one: it is what a --prefill-length invocation was for.
# Columns are `| b70-decode <sha> <backend> pp | <N ids> | <chunk> | <ms total> | <t/s> |`,
# so $5 is the millisecond total and $6 the rate -- the same two positions the
# tg row uses, which is why one awk shape serves both.
if [ "${#pp_rows[@]}" -gt 0 ]; then
  # The grade, measured on the box at the moment the rows were taken rather
  # than asserted here. A holder that appears mid-run is exactly what this is
  # meant to catch, so it is sampled AFTER the runs, not before.
  # Same default and env name as tools/box.sh; this probe is a bare ssh rather
  # than `box.sh run` because box.sh re-expands its arguments and would mangle
  # the quoting in the loop below.
  grade=$(ssh -o BatchMode=yes "$BOX" '
    n=$(docker ps -q 2>/dev/null | wc -l)
    for f in /proc/*/fdinfo/*; do
      grep -l drm-driver "$f" >/dev/null 2>&1 && n=$((n+1))
    done
    [ "$n" -eq 0 ] && echo RECORD || echo ITERATE' 2>/dev/null) || grade=UNKNOWN
  printf '%s\n' "${pp_rows[@]}" | awk -F'|' -v grade="${grade:-UNKNOWN}" '
    { ms[NR] = $5 + 0; ts[NR] = $6 + 0 }
    END {
      if (NR < 1) exit 0
      for (i = 1; i <= NR; i++) for (j = i + 1; j <= NR; j++)
        if (ts[j] < ts[i]) { t = ts[i]; ts[i] = ts[j]; ts[j] = t
                             t = ms[i]; ms[i] = ms[j]; ms[j] = t }
      mid = int((NR + 1) / 2)
      pct = (ts[mid] > 0) ? (100 * (ts[NR] - ts[1]) / ts[mid]) : 0
      printf "pp median %.2f t/s (%.1f ms total) over %d run(s); min %.2f, max %.2f, spread %.2f (%.2f%%) -- %s grade\n",
             ts[mid], ms[mid], NR, ts[1], ts[NR], ts[NR] - ts[1], pct, grade
    }' >&2
fi

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
