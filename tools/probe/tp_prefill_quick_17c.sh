#!/bin/bash
# Spec 17c: one quick (not interleaved, one run each) pp row per arm at the given lengths, to see where TP prefill
# stands before the interleaved r80.speed: tools/gpu_lock.sh box -- tools/probe/tp_prefill_quick_17c.sh 2048 16384
M=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
for d in "$@"; do
  for arm in "--parallel 1" "--parallel tp"; do
    echo "== $d $arm"
    build/src/cli/b70-decode $M --bench --prefill-length $d --tg 8 --max-len 32768 $arm 2>&1 | grep -E '^\| b70-decode|^pp:|b70-decode:'
  done
done
