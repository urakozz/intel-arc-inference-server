#!/bin/bash
# Spec 6 K3: passkey retrieval at ~120k context, run ON THE BOX from the repo root.
#   tools/probe/passkey.sh [pp-backend ...]      default: l0-int8 l0
# For each placement (0.05, 0.5, 0.95): tools/probe/passkey.py builds the prompt in the
# oracle container, b70-decode prefills it at --max-len 131072 and generates 8 ids
# greedily, the container decodes them, and the placement PASSES if "71432" is in the text.
# Files go to /tmp/passkey (small: ~1 MB each). Prints one line per (backend, placement)
# and a summary "passkey <backend>: k/3". Exit 0 iff every backend got 3/3.
# Env: MODEL (default the gate checkpoint), ZE_AFFINITY_MASK (default 0).
cd "$(dirname "$0")/../.."
MODEL="${MODEL:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
backends=("$@")
[ ${#backends[@]} -eq 0 ] && backends=(l0-int8 l0)
dir=/tmp/passkey
mkdir -p "$dir"
places="0.05 0.5 0.95"
tools/oracle/run_in_container.sh "for p in $places; do python3 tools/probe/passkey.py \"\$SNAP\" \$p /scratch/passkey/p\$p.ids; done" || exit 1
ok_all=1
for b in "${backends[@]}"; do
  pass=0
  for p in $places; do
    ids=$(./build/src/cli/b70-decode "$MODEL" --ids "$dir/p$p.ids" --n 8 --prefill \
          --pp-backend "$b" --max-len 131072 2> "$dir/p$p.$b.log" | tr '\n' ' ')
    text=$(tools/oracle/run_in_container.sh "python3 -P tools/oracle/tokenize.py \"\$SNAP\" decode $ids" 2>/dev/null)
    if printf '%s' "$text" | grep -q 71432; then verdict=PASS; pass=$((pass+1)); else verdict=FAIL; fi
    prefill=$(grep -E "^prefill:|memory:" "$dir/p$p.$b.log" | tr '\n' ' ')
    echo "passkey $b placement $p: ids [$ids] -> \"$text\" -- $verdict  ($prefill)"
  done
  echo "passkey $b: $pass/3"
  [ $pass -eq 3 ] || ok_all=0
done
[ $ok_all -eq 1 ]
