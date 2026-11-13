#!/bin/bash
# Phase 2: interleaved pairs, a control build (CTRL, e.g. main's tree) against a candidate build (CAND), Ornith ppN
# on one card. ON THE BOX under a card lock:
#   CTRL=<dir with build/> CAND=<dir with build/> tools/gpu_lock.sh card <N> -- tools/probe/pf_phase2_ab.sh [N] [rounds] [backend] [profile]
# profile=1 adds B70_PREFILL_PROFILE=1 (the phase table; t/s then ~4 % low).
N="${1:-4096}"; R="${2:-3}"; BE="${3:-l0}"; PROF="${4:-0}"
: "${CTRL:?CTRL=<the control build directory>}" "${CAND:?CAND=<the candidate build directory>}"
SNAP="${SNAP_ORNITH:-urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ}"
run() {   # run <label> <build dir>
  local p=""; [ "$PROF" = 1 ] && p="B70_PREFILL_PROFILE=1"
  env $p "$2"/src/cli/b70-decode "$SNAP" --bench --prefill-length "$N" --tg 32 \
    --max-len 32768 --parallel 1 --prefill-backend "$BE" 2>&1 |
    grep -E '^pp:|^  [a-z_A-Z0-9^]+ +[0-9.]+ +[0-9.]+ +[0-9.]+|TOTAL' | sed "s/^/[$1] /"
}
for r in $(seq "$R"); do
  if [ $((r % 2)) = 1 ]; then run ctrl "$CTRL"; run cand "$CAND"; else run cand "$CAND"; run ctrl "$CTRL"; fi
done
