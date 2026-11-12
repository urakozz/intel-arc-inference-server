#!/bin/bash
# Phase 2 Task 8: Flash-Next pp<N> at --layers 18 (Intel's checkpoint, one card), a control build (CTRL) against a
# candidate build (CAND), interleaved, R rounds (r33.speed's one-card row). ~52 GB of host RAM a run (the PLE table):
# one at a time. ON THE BOX under a card lock:
#   CTRL=<build dir> CAND=<build dir> tools/gpu_lock.sh card <N> -- tools/probe/pf_phase2_ab_q4exp.sh [N] [R]
N="${1:-4096}"; R="${2:-1}"
: "${CTRL:?CTRL=<the control build directory>}" "${CAND:?CAND=<the candidate build directory>}"
SNAP="${SNAP_Q4EXP_INTEL:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}"
run() {
  "$2"/src/cli/b70-decode "$SNAP" --bench --layers 18 --prefill-length "$N" --tg 16 --max-len 33024 2>&1 |
    grep -E '^pp:' | sed "s/^/[$1] /"
}
for r in $(seq "$R"); do
  if [ $((r % 2)) = 1 ]; then run ctrl "$CTRL"; run cand "$CAND"; else run cand "$CAND"; run ctrl "$CTRL"; fi
done
