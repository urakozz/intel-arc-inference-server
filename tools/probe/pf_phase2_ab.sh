#!/bin/bash
# Phase 2: interleaved pairs, today's kernels (B70_PF_PHASE2=old) against the branch's, Ornith ppN on one card.
# ON THE BOX under a card lock: tools/gpu_lock.sh card 0 -- tools/probe/pf_phase2_ab.sh [N] [rounds] [backend] [profile]
# profile=1 adds B70_PREFILL_PROFILE=1 (the phase table; t/s then ~4 % low).
N="${1:-4096}"; R="${2:-3}"; BE="${3:-l0}"; PROF="${4:-0}"
SNAP="${SNAP_ORNITH:-urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ}"
run() {
  local p=""; [ "$PROF" = 1 ] && p="B70_PREFILL_PROFILE=1"
  env B70_PF_PHASE2="$1" $p "${BUILD:-build}"/src/cli/b70-decode "$SNAP" --bench --prefill-length "$N" --tg 32 \
    --max-len 32768 --parallel 1 --prefill-backend "$BE" 2>&1 |
    grep -E '^pp:|^  [a-z_A-Z0-9^]+ +[0-9.]+ +[0-9.]+ +[0-9.]+|TOTAL' | sed "s/^/[$1] /"
}
for r in $(seq "$R"); do
  if [ $((r % 2)) = 1 ]; then run old; run new; else run new; run old; fi
done
