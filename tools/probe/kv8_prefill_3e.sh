#!/usr/bin/env bash
# On the box, under tools/gpu_lock.sh card N: item 3e's profile and paired timings.
# Usage: kv8_prefill_3e.sh profile|bench SNAPSHOT BUILD_BEFORE [BUILD_AFTER]
set -euo pipefail
mode=${1:?profile or bench}
snap=${2:?snapshot}
before=${3:?baseline build directory}
after=${4:-$before}
unset B70_PREFILL_PROFILE
out="$HOME/b70/logs/3e"
mkdir -p "$out"

run() {
  local build=$1 kv=$2 n=$3 log=$4
  local max_len=131072
  if [ "$mode" = profile ]; then max_len=65536; fi
  "$build/src/cli/b70-decode" "$snap" --bench --prefill-length "$n" \
    --max-len "$max_len" --tg 16 --parallel 1 --kv-cache "$kv" > "$out/$log.log" 2>&1
  sed -n '/^pp:/p; /^  attn_flash /p; /^  TOTAL /p' "$out/$log.log"
}

case "$mode" in
  profile)
    for kv in bf16 int8; do
      run "$before" "$kv" 15257 "profile-warmup-$kv"
    done
    export B70_PREFILL_PROFILE=1
    for kv in bf16 int8; do
      run "$before" "$kv" 15257 "profile-$kv"
    done
    ;;
  bench)
    [ "$before" != "$after" ] || { echo "bench needs distinct before/after builds" >&2; exit 2; }
    for build in "$before" "$after"; do
      for kv in bf16 int8; do
        run "$build" "$kv" 15257 "warmup-${build##*/}-$kv"
      done
    done
    for n in 15257 31000 65000 128000; do
      for round in 1 2 3; do
        builds=("$before" "$after")
        forms=(bf16 int8)
        if ((round % 2 == 0)); then
          builds=("$after" "$before")
          forms=(int8 bf16)
        fi
        for kv in "${forms[@]}"; do
          for build in "${builds[@]}"; do
            log="bench-${build##*/}-$kv-$n-r$round"
            echo "$log"
            run "$build" "$kv" "$n" "$log"
          done
        done
      done
    done
    ;;
  *) echo "expected profile or bench" >&2; exit 2 ;;
esac
