#!/bin/bash
# Spec 17c: b70-decode --parallel tp --prefill against one card's --prefill on one prompt (the ids side by side and
# cmp'd), under one whole-box lock: tools/gpu_lock.sh box -- tools/probe/tp_prefill_smoke_17c.sh [prompt] [n]
M=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
P=tests/golden/prompts/${1:-prose}.ids
N=${2:-24}
D=${TMPDIR:-/tmp}/b70-17c-smoke
mkdir -p "$D"
echo "== one card"; build/src/cli/b70-decode $M --ids $P --n $N --prefill --parallel 1 --max-len 16384 > "$D/one.txt"; echo rc=$?
echo "== tp"; build/src/cli/b70-decode $M --ids $P --n $N --prefill --parallel tp --max-len 16384 > "$D/tp.txt"; echo rc=$?
paste "$D/one.txt" "$D/tp.txt" | tr '\n' ' '; echo
cmp "$D/one.txt" "$D/tp.txt" && echo IDS_IDENTICAL
