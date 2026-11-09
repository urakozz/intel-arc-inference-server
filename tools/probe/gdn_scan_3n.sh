#!/usr/bin/env bash
# On the box, under tools/gpu_lock.sh box: open-work 3n's timed before / after, one lock.
# Record: docs/probes/probe-gdn-prefill-2026-11-10.md (section 8).
#   1. probe_gdn_scan_occ (this tree's build): the packed scan over h heads, the pre-3n staged entry
#      and candidate 2 (_wg8), in one rotation
#   2. b70-decode --bench, l0, bf16 KV, real weights, BEFORE (main's build) and AFTER (this tree's):
#      one warm-up each, then three rounds of six arms (before / after x 2048 plain, 15257 plain,
#      2048 B70_PREFILL_PROFILE=1), the arm order rotating each round
# Usage: gdn_scan_3n.sh SNAPSHOT BEFORE_BUILD AFTER_BUILD
set -euo pipefail
snap=${1:?snapshot}
before=${2:?before build}
after=${3:?after build}
out="$HOME/b70/logs/3n"
mkdir -p "$out"
unset B70_PREFILL_PROFILE

stamp() { echo "== $(date '+%F %T') $*"; }

eng() {   # build-name(before|after) mode(plain|profile) n log
  local b
  if [ "$1" = before ]; then b=$before; else b=$after; fi
  if [ "$2" = profile ]; then
    B70_PREFILL_PROFILE=1 "$b/src/cli/b70-decode" "$snap" --bench --prefill-length "$3" \
      --prefill-backend l0 --max-len 32768 --tg 16 --parallel 1 --kv-cache bf16 > "$out/$4.log" 2>&1
  else
    "$b/src/cli/b70-decode" "$snap" --bench --prefill-length "$3" \
      --prefill-backend l0 --max-len 32768 --tg 16 --parallel 1 --kv-cache bf16 > "$out/$4.log" 2>&1
  fi
  sed -n '/^pp:/p' "$out/$4.log"
}

stamp "uptime: $(uptime)"
stamp scan-occ
"$after/tools/probe/probe_gdn_scan_occ" --rounds 7 > "$out/scan-occ.log" 2>&1
grep -E "^SANITY|^\|" "$out/scan-occ.log" || true

stamp engine-warmup
eng before plain 2048 warmup-before
eng after plain 2048 warmup-after

arms=("before plain 2048" "after plain 2048" "before plain 15257" "after plain 15257"
      "before profile 2048" "after profile 2048")
for round in 1 2 3; do
  for i in 0 1 2 3 4 5; do
    a=${arms[$(((i + 2 * (round - 1)) % 6))]}
    set -- $a
    stamp "$1 $2 $3 r$round"
    eng "$1" "$2" "$3" "$1-$2-$3-r$round"
  done
done
stamp "uptime: $(uptime)"
stamp done
