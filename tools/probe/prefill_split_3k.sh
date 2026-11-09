#!/usr/bin/env bash
# On the box, under tools/gpu_lock.sh box: open-work 3k's whole measurement, one lock.
# Record: docs/probes/probe-prefill-chunk-split-2026-11-10.md.
#   1. probe_dpas_rate --sweep (the card's DPAS peaks)
#   2. probe_prefill_split at 2048 and 921 rows (the 15,257-id prompt's last chunk): the int4
#      linears of one chunk on l0 and l0-int8, itemised, over 3i's distinct-layer walk
#   3. b70-decode --bench --prefill-length 2048 / 15257, bf16 KV, l0 vs l0-int8: warm-up, then three
#      interleaved rounds unprofiled (walls), then two interleaved rounds with B70_PREFILL_PROFILE=1
#   4. probe_dpas_rate --sweep again (clock state after the run)
# Usage: prefill_split_3k.sh SNAPSHOT [BUILD]
set -euo pipefail
snap=${1:?snapshot}
build=${2:-build}
out="$HOME/b70/logs/3k"
mkdir -p "$out"
unset B70_PREFILL_PROFILE

stamp() { echo "== $(date '+%F %T') $*"; }

eng() {   # backend n log
  "$build/src/cli/b70-decode" "$snap" --bench --prefill-length "$2" --prefill-backend "$1" \
    --max-len 32768 --tg 16 --parallel 1 --kv-cache bf16 > "$out/$3.log" 2>&1
  sed -n '/^pp:/p' "$out/$3.log"
}

stamp dpas-pre
"$build/tools/probe/probe_dpas_rate" --sweep > "$out/dpas-pre.log" 2>&1
grep -E "bf16|i8" "$out/dpas-pre.log" | head -20 || true

stamp split
"$build/tools/probe/probe_prefill_split" --rows 2048,921 --rounds 7 > "$out/split.log" 2>&1
grep '^|' "$out/split.log" || true

stamp engine-warmup
eng l0 2048 warmup-l0-2048
eng l0-int8 2048 warmup-l0-int8-2048

for round in 1 2 3; do
  backends=(l0 l0-int8)
  ((round % 2 == 0)) && backends=(l0-int8 l0)
  for n in 2048 15257; do
    for b in "${backends[@]}"; do
      stamp "plain $b $n r$round"
      eng "$b" "$n" "plain-$b-$n-r$round"
    done
  done
done

export B70_PREFILL_PROFILE=1
for round in 1 2; do
  backends=(l0 l0-int8)
  ((round % 2 == 0)) && backends=(l0-int8 l0)
  for n in 2048 15257; do
    for b in "${backends[@]}"; do
      stamp "profile $b $n r$round"
      eng "$b" "$n" "profile-$b-$n-r$round"
    done
  done
done
unset B70_PREFILL_PROFILE

stamp dpas-post
"$build/tools/probe/probe_dpas_rate" --sweep > "$out/dpas-post.log" 2>&1
grep -E "bf16|i8" "$out/dpas-post.log" | head -20 || true
stamp done
