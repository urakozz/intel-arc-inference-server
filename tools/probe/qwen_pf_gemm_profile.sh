#!/usr/bin/env bash
# On the box, from a tree's root: Qwen3.8 `l0` prefill phase profile at pp4096 and pp16384, one card
# (record: docs/probes/probe-qwen-pf-gemm-2026-11-13.md, Task 1). B70_PREFILL_PROFILE=2 (the phase
# table plus the per-launch trace by kernel name), then one plain run of each for the wall.
# Usage: qwen_pf_gemm_profile.sh CARD [BUILD]   (takes `tools/gpu_lock.sh card CARD` itself)
set -u
card=${1:?card}
build=${2:-build}
out="$HOME/b70/logs/qwen-pf-gemm"
mkdir -p "$out"
snap=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
tools/gpu_lock.sh card "$card" -- bash -c "
  for n in 4096 16384; do
    echo \"== \$(date '+%F %T') profile \$n\"
    B70_PREFILL_PROFILE=2 $build/src/cli/b70-decode $snap --bench --prefill-length \$n --tg 16 \
      --max-len 32768 --parallel 1 --prefill-backend l0 > $out/profile-l0-\$n.log 2>&1
    grep -E '^pp:' $out/profile-l0-\$n.log
    echo \"== \$(date '+%F %T') plain \$n\"
    $build/src/cli/b70-decode $snap --bench --prefill-length \$n --tg 16 \
      --max-len 32768 --parallel 1 --prefill-backend l0 > $out/plain-l0-\$n.log 2>&1
    grep -E '^pp:' $out/plain-l0-\$n.log
  done
"
echo "done $(date '+%F %T')"
