#!/bin/bash
# Spec 23e Task 6 (MG5): passkey retrieval for Muse-Glimmer-30B at its context 131072 on ONE card (the default, spec 23
# decision 7), ON THE BOX from the repo root (opt-in; box queue row 46's r46.passkey) - tools/probe/kolibri_passkey.sh's
# shape carried to Muse.
#   tools/probe/muse_passkey.sh [lm-head ...]      default: int8 (the served head, spec 9)
# For each placement (0.05, 0.5, 0.95): tools/probe/passkey.py builds the prompt with Muse's tokenizer in the oracle
# container (no special tokens), this script puts Muse's BOS (200000 - the template writes it before every prompt; a
# continuation without it is off the trained distribution) in front, b70-decode prefills it (23d's l0 prefill, one card,
# --max-len MAX_LEN) and generates 8 ids greedily, the container decodes them, and the placement PASSES if "71432" is
# in the text. Prints one line per (head, placement) with b70-decode's prefill and decode lines (decode at this depth
# reads ~1.74 GB of full-layer KV a token beside the 14.72 GB of weights and rings, derived - the decode speed at
# depth is the row's second number) and "passkey muse <head>: k/3"; exit 0 iff every head got 3/3 (the gate: 3/3 with
# the int8 head).
# Env: MODEL (the int4 checkpoint, either Vishva007 spelling), ORACLE_MODEL (its HF cache dir, for the tokenizer: any of
# the three repos - tokenizer.json is identical in all), MAX_LEN (131072: decision 6's ceiling), N_TARGET (130000 prompt
# ids: BOS + 8 generated fit under MAX_LEN), DECODE_ARGS (extra b70-decode flags, e.g. '--pp 2'), ZE_AFFINITY_MASK.
# Hold ~/b70-gpu.lock (flock) around it on a shared box; it runs for a while (three ~130k prefills).
cd "$(dirname "$0")/../.." || exit 2
MODEL="${MODEL:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}"
export ORACLE_MODEL="${ORACLE_MODEL:-models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound}"
MAX_LEN="${MAX_LEN:-131072}"
N_TARGET="${N_TARGET:-130000}"
heads=("$@")
[ ${#heads[@]} -eq 0 ] && heads=(int8)
dir=/tmp/muse-passkey
mkdir -p "$dir"
places="0.05 0.5 0.95"
tools/oracle/run_in_container.sh "mkdir -p /scratch/muse-passkey; for p in $places; do python3 tools/probe/passkey.py \"\$SNAP\" \$p /scratch/muse-passkey/p\$p.ids $N_TARGET; done" || exit 1
ok_all=1
for h in "${heads[@]}"; do
  pass=0
  for p in $places; do
    [ -s "$dir/p$p.ids" ] || { echo "muse_passkey: no $dir/p$p.ids (the container's /scratch is $dir?)" >&2; exit 1; }
    { echo 200000; cat "$dir/p$p.ids"; } > "$dir/p$p.bos.ids"
    # shellcheck disable=SC2086 # DECODE_ARGS is a list of flags
    ids=$(./build/src/cli/b70-decode "$MODEL" --ids "$dir/p$p.bos.ids" --n 8 --prefill --lm-head "$h" \
          --max-len "$MAX_LEN" ${DECODE_ARGS:-} 2> "$dir/p$p.$h.log" | tr '\n' ' ')
    text=$(tools/oracle/run_in_container.sh "python3 -P tools/oracle/tokenize.py \"\$SNAP\" decode $ids" 2>/dev/null)
    if printf '%s' "$text" | grep -q 71432; then verdict=PASS; pass=$((pass+1)); else verdict=FAIL; fi
    info=$(grep -E "^max_len|^split|^prefill:|^generate:|memory" "$dir/p$p.$h.log" | tr '\n' ' ')
    echo "passkey muse $h placement $p: ids [$ids] -> \"$text\" -- $verdict  ($info)"
  done
  echo "passkey muse $h: $pass/3"
  [ $pass -eq 3 ] || ok_all=0
done
[ $ok_all -eq 1 ]
