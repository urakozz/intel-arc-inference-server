#!/bin/bash
# Phase 2 Task 1 (docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md): the per-launch
# prefill trace of Ornith at pp4096 on one card, l0 and l0-int8 (B70_PREFILL_PROFILE=2), two
# rounds each. Run ON THE BOX from the repo root, under the card's lock:
#   tools/gpu_lock.sh card 0 -- tools/probe/pf_phase2_trace.sh [N] [snapshot]
# Prints each run's pp line, phase table and (phase, entry) trace.
set -u
N="${1:-4096}"
SNAP="${2:-${SNAP_ORNITH:-urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ}}"
for round in 1 2; do
  for be in l0 l0-int8; do
    echo "=== round $round backend $be pp$N"
    B70_PREFILL_PROFILE=2 build/src/cli/b70-decode "$SNAP" --bench --prefill-length "$N" --tg 32 \
      --max-len 32768 --parallel 1 --prefill-backend "$be" 2>&1 |
      grep -E '^pp:|^  [a-z_A-Z0-9^]+ +[0-9.]+ +[0-9.]+|^    [a-z_]|TOTAL|instrumented|^prefill backend'
  done
done
