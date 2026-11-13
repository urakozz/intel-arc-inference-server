#!/usr/bin/env bash
# On the box, from the branch tree's root: b70-serve's start-up plan on one card (context, slots, prefill chunk
# cap) for main's build and this tree's, per recipe of docs/18-running-models.md - the server is stopped once it
# listens (record: docs/probes/probe-qwen-pf-gemm-2026-11-13.md, section 6).
# Usage: qwen_chunk4096_plans.sh CARD MAIN_TREE    (takes `tools/gpu_lock.sh card CARD` itself)
set -u
card=${1:?card}
main=${2:?main tree}
out="$HOME/b70/logs/qwen-chunk-4096"
mkdir -p "$out"
snap=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
tools/gpu_lock.sh card "$card" -- bash -c "
  one() {   # tree tag args...
    t=\$1; tag=\$2; shift 2
    \$t/build/src/cli/b70-serve $snap --port 8018 --parallel 1 \"\$@\" > $out/plan-\$tag.log 2>&1 &
    srv=\$!
    for _ in \$(seq 1 600); do
      curl -sf http://127.0.0.1:8018/v1/models > /dev/null && break
      kill -0 \$srv 2>/dev/null || break
      sleep 1
    done
    kill \$srv 2>/dev/null; wait \$srv 2>/dev/null
    echo \"== \$tag (\$*)\"
    grep -E '^max_len|prefill chunk|^slots|^memory|total' $out/plan-\$tag.log | head -8
  }
  i=0
  for args in '' '--kv-cache int8' '--prefill-backend l0-int8' '--max-num-seqs 4' '--max-num-seqs 4 --kv-cache int8'; do
    i=\$((i + 1))
    one $main main-\$i \$args
    one \$PWD branch-\$i \$args
  done
"
echo "done $(date '+%F %T')"
