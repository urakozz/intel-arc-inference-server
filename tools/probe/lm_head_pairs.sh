#!/usr/bin/env bash
# Spec 9 H1/H3 (plan 9b Task 3 Step 4): the int8 lm_head against the bf16 one, as
# interleaved pairs after a warm-up, ON THE BOX from a tree root, under the GPU lock:
#   flock ~/b70-gpu.lock tools/probe/lm_head_pairs.sh <snapshot> [pairs = 3]
# Each run is `b70-decode --bench --pp 4096 --tg 256`: prefill 4096 ids (H3's pp4096 row)
# and then decode 256 at that depth (H1's tg row). Pair i runs bf16 then int8 for odd i
# and int8 then bf16 for even i, so a drift in clocks lands on both arms. Then one
# `--profile` per head at depth 512 for the head launch's in-situ time (Review Focus 1:
# its GB/s; the launch is depth-independent). `uptime` before and after.
set -euo pipefail
SNAP="$1"
PAIRS="${2:-3}"
D=build/src/cli/b70-decode
export B70_GIT_SHA="${B70_GIT_SHA:-unknown}"
run() { "$D" "$SNAP" --bench --pp 4096 --tg 256 --lm-head "$1" 2> /dev/null | grep '^| b70-decode'; }
uptime
echo "warm-up"; run bf16 > /dev/null
for i in $(seq "$PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then order="bf16 int8"; else order="int8 bf16"; fi
  for h in $order; do echo "pair $i $h"; run "$h"; done
done
for h in bf16 int8; do
  echo "profile $h"
  "$D" "$SNAP" --profile --depth 512 --steps 32 --lm-head "$h" 2> /dev/null |
    grep -iE 'gemv_i8w|gemv_bf16_M1_K5120_N248320|lm_head|total' | head -8
done
uptime
