#!/bin/bash
# Spec 18e Task 1: passkey retrieval for K2-Horizon near the one-card int8-KV ceiling, ON THE BOX
# from the repo root (opt-in, by hand; spec 6 K3's passkey.sh carried to K2).
#   tools/probe/k2_passkey.sh [kv-cache ...]      default: int8
# For each placement (0.05, 0.5, 0.95): tools/probe/passkey.py builds the prompt with K2's tokenizer
# in the oracle container (BOS 0 prepended: every K2 prompt leads with it), b70-decode prefills it
# (K2's l0 prefill) at --max-len MAX_LEN with --kv-cache <form> and generates 8 ids greedily, the
# container decodes them, and the placement PASSES if "71432" is in the text. Prints one line per
# (form, placement) and "passkey k2 <form>: k/3"; exit 0 iff every form got 3/3.
# The ceiling (k2_plan_test, 32.53 GB card, 1.5 GB reserve, the prefill scratch planned): int8 KV
# 83968 (bf16 head) / 90368 (--lm-head int8); bf16 KV 42752 / 45824. N_TARGET defaults to ~95 % of
# the int8 / bf16-head ceiling: MAX_LEN=auto must print "max_len: auto -> 83968" (the run fails
# loudly if the prompt does not fit). A bf16 form needs N_TARGET=40000.
# Env: MODEL (the K2 int4 checkpoint), ORACLE_MODEL (its HF cache dir), MAX_LEN (auto), N_TARGET
# (79000), DECODE_ARGS (extra b70-decode flags, e.g. '--lm-head int8' with N_TARGET=85000),
# ZE_AFFINITY_MASK (0). Hold ~/b70-gpu.lock (flock) around it on a shared box.
cd "$(dirname "$0")/../.." || exit 2
MODEL="${MODEL:-urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ}"
export ORACLE_MODEL="${ORACLE_MODEL:-models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
MAX_LEN="${MAX_LEN:-auto}"
N_TARGET="${N_TARGET:-79000}"
forms=("$@")
[ ${#forms[@]} -eq 0 ] && forms=(int8)
dir=/tmp/k2-passkey
mkdir -p "$dir"
places="0.05 0.5 0.95"
tools/oracle/run_in_container.sh "mkdir -p /scratch/k2-passkey; for p in $places; do python3 tools/probe/passkey.py \"\$SNAP\" \$p /scratch/k2-passkey/raw\$p.ids $((N_TARGET - 1)); done" || exit 1
for p in $places; do { echo 0; cat "$dir/raw$p.ids"; } > "$dir/p$p.ids"; done   # BOS 0 first
ok_all=1
for f in "${forms[@]}"; do
  pass=0
  for p in $places; do
    # shellcheck disable=SC2086 # DECODE_ARGS is a list of flags
    ids=$(./build/src/cli/b70-decode "$MODEL" --ids "$dir/p$p.ids" --n 8 --prefill --kv-cache "$f" \
          --max-len "$MAX_LEN" ${DECODE_ARGS:-} 2> "$dir/p$p.$f.log" | tr '\n' ' ')
    text=$(tools/oracle/run_in_container.sh "python3 -P tools/oracle/tokenize.py \"\$SNAP\" decode $ids" 2>/dev/null)
    if printf '%s' "$text" | grep -q 71432; then verdict=PASS; pass=$((pass+1)); else verdict=FAIL; fi
    info=$(grep -E "^max_len|^prefill:|memory:" "$dir/p$p.$f.log" | tr '\n' ' ')
    echo "passkey k2 $f placement $p: ids [$ids] -> \"$text\" -- $verdict  ($info)"
  done
  echo "passkey k2 $f: $pass/3"
  [ $pass -eq 3 ] || ok_all=0
done
[ $ok_all -eq 1 ]
