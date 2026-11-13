#!/usr/bin/env bash
# On the box, from the branch tree's root: Qwen3.8 `l0` prompt pass, one card - main's build at
# `--prefill-chunk A` against this tree's build at `--prefill-chunk B`, interleaved pairs after a
# warm-up (record: docs/probes/probe-qwen-pf-gemm-2026-11-13.md, section 6).
# Usage: qwen_chunk4096_bench.sh CARD MAIN_TREE ROUNDS "LENGTHS" [A B] [BACKEND] [PROFILE]
#   PROFILE=1 adds one B70_PREFILL_PROFILE=1 run per length and arm after the rounds.
# Takes `tools/gpu_lock.sh card CARD` itself, once for the whole series.
set -u
card=${1:?card}
main=${2:?main tree}
rounds=${3:-3}
lengths=${4:-"4096 16384 32768"}
ca=${5:-2048}
cb=${6:-4096}
backend=${7:-l0}
profile=${8:-0}
out="$HOME/b70/logs/qwen-chunk-4096"
mkdir -p "$out"
snap=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
tag="$backend-$(date +%H%M%S)"
tools/gpu_lock.sh card "$card" -- bash -c "
  one() {   # tree chunk n log [env]
    \$5 \$1/build/src/cli/b70-decode $snap --bench --prefill-length \$3 --tg 16 --max-len 34816 \
      --parallel 1 --prefill-backend $backend --prefill-chunk \$2 > \$4 2>&1
    echo \"\$(date '+%T') \$1 chunk \$2 pp\$3: \$(grep -E '^pp:' \$4)\"
  }
  one $main $ca 4096 $out/warm-$tag-a.log
  one \$PWD $cb 4096 $out/warm-$tag-b.log
  for r in \$(seq 1 $rounds); do
    for n in $lengths; do
      one $main $ca \$n $out/r\$r-$tag-main-\$n.log
      one \$PWD $cb \$n $out/r\$r-$tag-branch-\$n.log
    done
  done
  if [ $profile = 1 ]; then
    for n in $lengths; do
      one $main $ca \$n $out/prof-$tag-main-\$n.log 'env B70_PREFILL_PROFILE=1'
      one \$PWD $cb \$n $out/prof-$tag-branch-\$n.log 'env B70_PREFILL_PROFILE=1'
    done
  fi
"
echo "done $(date '+%F %T')"
