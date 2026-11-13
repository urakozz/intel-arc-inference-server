#!/bin/bash
# Phase 2 Task 1: dump one Ornith pp4096 prefill's GDN intra-chunk inputs (B70_PF_GDN_DUMP, calls 0 / 9 / 21 =
# layers 0 / 12 / 28 of the first chunk), then probe_gdn_intra on them. ON THE BOX, from the repo root, under a
# card lock: tools/gpu_lock.sh card 1 -- tools/probe/pf_phase2_gdn_probe.sh <dump dir> [probe args...]
set -eu
D="$1"; shift
SNAP="${SNAP_ORNITH:-urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ}"
mkdir -p "$D"
if [ ! -f "$D/gdn21.txt" ]; then
  B70_PF_GDN_DUMP="$D" build/src/cli/b70-decode "$SNAP" --bench --prefill-length 4096 --tg 4 --max-len 8192 \
    --parallel 1 2>&1 | grep -E '^pp:'
fi
build/tools/probe/probe_gdn_intra --dump "$D" "$@"
