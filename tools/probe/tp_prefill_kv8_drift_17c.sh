#!/bin/bash
# Spec 17c: how fragile the int8-KV long run is to a legitimate rounding change - 64 ids after the long prompt on
# one card by prefill and by decode replays (ingest), and under --parallel tp by prefill and by ingest, all
# --kv-cache int8 (KV=bf16 for the other form; N ids, LEN the max-len); the first position each pair differs at.
# Under one box lock:
#   tools/gpu_lock.sh box -- tools/probe/tp_prefill_kv8_drift_17c.sh [prompt]
M=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
P=tests/golden/prompts/${1:-long}.ids
D=${TMPDIR:-/tmp}/b70-17c-drift-${1:-long}; mkdir -p "$D"
N=${N:-64} LEN=${LEN:-16384} KV=${KV:-int8}
run() { build/src/cli/b70-decode $M --ids $P --n $N --kv-cache $KV --max-len $LEN "$@" 2>/dev/null; }
run --parallel 1 --prefill > "$D/one_pf.ids"
run --parallel 1 > "$D/one_in.ids"
run --parallel tp --prefill > "$D/tp_pf.ids"
run --parallel tp > "$D/tp_in.ids"
first() { paste "$D/$1.ids" "$D/$2.ids" | awk -v a="$1" -v b="$2" '$1 != $2 { print a " vs " b ": first differs at id " NR - 1 " (" $1 " / " $2 ")"; f = 1; exit } END { if (!f) print a " vs " b ": identical ids" }'; }
for p in "one_pf one_in" "one_pf tp_pf" "one_in tp_in" "tp_pf tp_in" "one_in tp_pf"; do first $p; done
