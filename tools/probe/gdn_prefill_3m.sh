#!/usr/bin/env bash
# On the box, under tools/gpu_lock.sh box: open-work 3m's whole timed measurement, one lock.
# Record: docs/probes/probe-gdn-prefill-2026-11-10.md.
#   1. probe_fma_rate (fp32 vector peak), probe_dpas_rate --sweep (bf16 DPAS peak), probe_gdn_scan_occ
#      (the production scan over the first h heads: residency)
#   2. b70-decode --bench, l0, bf16 KV, real weights: one warm-up, then three rounds of four arms
#      (2048 / 15257 ids, plain / B70_PREFILL_PROFILE=1), the arm order rotating each round
#   3. probe_fma_rate again (clock state after the run)
# Usage: gdn_prefill_3m.sh SNAPSHOT [BUILD]
set -euo pipefail
snap=${1:?snapshot}
build=${2:-build}
out="$HOME/b70/logs/3m"
mkdir -p "$out"
unset B70_PREFILL_PROFILE

stamp() { echo "== $(date '+%F %T') $*"; }

eng() {   # mode(plain|profile) n log
  if [ "$1" = profile ]; then
    B70_PREFILL_PROFILE=1 "$build/src/cli/b70-decode" "$snap" --bench --prefill-length "$2" \
      --prefill-backend l0 --max-len 32768 --tg 16 --parallel 1 --kv-cache bf16 > "$out/$3.log" 2>&1
  else
    "$build/src/cli/b70-decode" "$snap" --bench --prefill-length "$2" \
      --prefill-backend l0 --max-len 32768 --tg 16 --parallel 1 --kv-cache bf16 > "$out/$3.log" 2>&1
  fi
  sed -n '/^pp:/p' "$out/$3.log"
}

stamp "uptime: $(uptime)"
stamp fma-pre
"$build/tools/probe/probe_fma_rate" > "$out/fma-pre.log" 2>&1
grep -E "^\||peak|checksum" "$out/fma-pre.log" || true
stamp dpas-pre
"$build/tools/probe/probe_dpas_rate" --sweep > "$out/dpas-pre.log" 2>&1
grep -E "bf16_bf16" "$out/dpas-pre.log" | tail -1 || true

stamp scan-occ
"$build/tools/probe/probe_gdn_scan_occ" --rounds 7 > "$out/scan-occ.log" 2>&1
grep -E "^SANITY|^\|" "$out/scan-occ.log" || true

stamp engine-warmup
eng plain 2048 warmup-2048

arms=("plain 2048" "plain 15257" "profile 2048" "profile 15257")
for round in 1 2 3; do
  for i in 0 1 2 3; do
    a=${arms[$(((i + round - 1) % 4))]}
    set -- $a
    stamp "$1 $2 r$round"
    eng "$1" "$2" "$1-$2-r$round"
  done
done

stamp fma-post
"$build/tools/probe/probe_fma_rate" > "$out/fma-post.log" 2>&1
grep -E "peak|checksum" "$out/fma-post.log" || true
stamp "uptime: $(uptime)"
stamp done
