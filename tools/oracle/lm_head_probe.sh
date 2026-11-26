#!/bin/bash
# Spec 9 P0 (plan 9a): the int8 lm_head probe's production run, on the GPU machine, from the repo root.
#   tools/probe/detach.sh $B70_DATA/logs/probe/spec9a.log tools/oracle/lm_head_probe.sh
# Stages its inputs into $B70_DATA/oracle-out/qwen38/lm-head-probe/ (run_in_container.sh mounts it):
#   - golden: the three prompts + the 32 greedy golden tokens of the W4A16 g64
#     checkpoint ($B70_DATA/oracle-out/qwen38/golden/*.tokens);
#   - toolcall: A4 scenarios' prompts + the bf16 oracle's outputs
#     (TOOLCALL, the toolcall set's toolcall-out/*.bf16.ids): one per task type on `box` first,
#     then t1/t2/t6 on the other five files (the short-context ones; the long
#     t3/t4/t5 on the other files cost ~30 min of CPU forward each).
# Then one dump (the model, ~60 GB) and one analyze (the head only).
set -euo pipefail
OUT=${B70_DATA:?set B70_DATA, the data root}/oracle-out/qwen38/lm-head-probe
GOLDEN=${GOLDEN:-$B70_DATA/oracle-out/qwen38/golden}
TOOLCALL=${TOOLCALL:?set TOOLCALL to the toolcall-out directory of the toolcall set}
THREADS=${ORACLE_THREADS:-40}
mkdir -p "$OUT"
src=""
for p in prose code cjk; do
  cp "$GOLDEN/$p.tokens" "$OUT/"
  src+=" --source golden/$p:/ws/tests/golden/prompts/$p.ids:$OUT/$p.tokens"
done
names=""
for t in t1_define t2_explain t6_usage t3_rename t4_comment t5_test; do names+=" $t-box"; done
for f in linear_l0 loader openai rotation step; do
  for t in t1_define t2_explain t6_usage; do names+=" $t-$f"; done
done
for n in $names; do
  cp "$TOOLCALL/$n.bf16.ids" "$OUT/"
  src+=" --source toolcall/$n:/ws/tests/golden/toolcall/$n.ids:$OUT/$n.bf16.ids"
done
free -g | sed -n 2p
if [ "${SKIP_DUMP:-0}" != 1 ]; then
  ORACLE_THREADS=$THREADS tools/oracle/run_in_container.sh \
    "python3 tools/oracle/lm_head_probe.py dump \"\$SNAP\" --out $OUT/hiddens.safetensors $src"
fi
ORACLE_THREADS=$THREADS tools/oracle/run_in_container.sh \
  "python3 tools/oracle/lm_head_probe.py analyze \"\$SNAP\" --hiddens $OUT/hiddens.safetensors --out $OUT/result.json"
