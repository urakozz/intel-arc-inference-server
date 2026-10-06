#!/bin/bash
# Spec 20e Task 4 (KL4): passkey retrieval for Kolibri-1 at its trained 262144 on TWO cards, ON THE BOX from
# the repo root (opt-in; box queue row 28's r28.passkey) - tools/probe/k2_passkey.sh's shape carried to
# Kolibri.
#   tools/probe/kolibri_passkey.sh [lm-head ...]      default: int8 (the served head, spec 9)
# For each placement (0.05, 0.5, 0.95): tools/probe/passkey.py builds the prompt with Kolibri's tokenizer in
# the oracle container (no BOS: Kolibri has none, spec 20 §1), b70-decode prefills it (20d's l0 prefill,
# --pp 2 - Kolibri's default - at --max-len MAX_LEN) and generates 8 ids greedily, the container decodes
# them, and the placement PASSES if "71432" is in the text. Prints one line per (head, placement) with
# b70-decode's prefill and decode lines (decode at this depth reads ~5.4 GB of full-layer KV a token,
# derived - more than the weights' ~2.4 GB) and "passkey kolibri <head>: k/3"; exit 0 iff every head got
# 3/3 (the gate: 3/3 with the int8 head).
# Env: MODEL (the int4 checkpoint, spec 20b), ORACLE_MODEL (its HF cache dir, for the tokenizer), MAX_LEN
# (262144: spec 20 decision 3's ceiling), N_TARGET (260000 prompt ids: 8 generated fit under MAX_LEN),
# DECODE_ARGS (extra b70-decode flags, e.g. '--pipeline-handoff peer'), ZE_AFFINITY_MASK (0,1: both cards).
# Hold ~/b70-gpu.lock (flock) around it on a shared box; it runs for a while (three ~260k prefills).
cd "$(dirname "$0")/../.." || exit 2
MODEL="${MODEL:-urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ}"
export ORACLE_MODEL="${ORACLE_MODEL:-models--urakozz--Kolibri-1-W4A16-g64-AutoRound-GPTQ}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"
MAX_LEN="${MAX_LEN:-262144}"
N_TARGET="${N_TARGET:-260000}"
heads=("$@")
[ ${#heads[@]} -eq 0 ] && heads=(int8)
dir=/tmp/kolibri-passkey
mkdir -p "$dir"
places="0.05 0.5 0.95"
tools/oracle/run_in_container.sh "mkdir -p /scratch/kolibri-passkey; for p in $places; do python3 tools/probe/passkey.py \"\$SNAP\" \$p /scratch/kolibri-passkey/p\$p.ids $N_TARGET; done" || exit 1
ok_all=1
for h in "${heads[@]}"; do
  pass=0
  for p in $places; do
    [ -s "$dir/p$p.ids" ] || { echo "kolibri_passkey: no $dir/p$p.ids (the container's /scratch is $dir?)" >&2; exit 1; }
    # shellcheck disable=SC2086 # DECODE_ARGS is a list of flags
    ids=$(./build/src/cli/b70-decode "$MODEL" --ids "$dir/p$p.ids" --n 8 --prefill --lm-head "$h" \
          --max-len "$MAX_LEN" ${DECODE_ARGS:-} 2> "$dir/p$p.$h.log" | tr '\n' ' ')
    text=$(tools/oracle/run_in_container.sh "python3 -P tools/oracle/tokenize.py \"\$SNAP\" decode $ids" 2>/dev/null)
    if printf '%s' "$text" | grep -q 71432; then verdict=PASS; pass=$((pass+1)); else verdict=FAIL; fi
    info=$(grep -E "^max_len|^split|^prefill:|^generate:|memory" "$dir/p$p.$h.log" | tr '\n' ' ')
    echo "passkey kolibri $h placement $p: ids [$ids] -> \"$text\" -- $verdict  ($info)"
  done
  echo "passkey kolibri $h: $pass/3"
  [ $pass -eq 3 ] || ok_all=0
done
[ $ok_all -eq 1 ]
