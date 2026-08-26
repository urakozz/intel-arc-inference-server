#!/usr/bin/env bash
# Regenerates the three golden dumps, one after another (61 GiB peak each), from
# the repo root ON THE BOX - the Mac never loads the model. dump.py takes one
# --prompt/--out pair, so this is three container invocations and the model
# reloads (~2 min) per prompt; ~17 min serial on an idle box. No numerics are
# touched here.
#
#   tools/box.sh sync && ssh <box>
#   cd ~/b70-inference-server
#   OUT_DIR=... setsid nohup tools/oracle/golden.sh > "$OUT_DIR"/golden.log 2>&1 </dev/null &
#
# (detached, so the run outlives the ssh session). Prompt ids come from
# $IDS_DIR, which defaults to the COMMITTED tests/golden/prompts - see
# "Running it" in tools/oracle/README.md.
#
# **One golden set per checkpoint, in its own directory.** The gate's whole
# assertion is that the engine reproduces the oracle's ids on the SAME weights,
# so a golden set belongs to exactly one checkpoint and overwriting one with
# another's output silently changes what the gate means. OUT_DIR and the
# snapshot are therefore both parameters, and neither has a default that
# clobbers the other checkpoint's set:
#
#   the published bf16-lm_head checkpoint (default, HF cache):
#     tools/oracle/golden.sh
#   the self-quantised int4-lm_head one:
#     OUT_DIR=oracle-out-rtn \
#     ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 \
#     ORACLE_THREADS=28 tools/oracle/golden.sh
#
# Env: OUT_DIR, IDS_DIR, plus everything run_in_container.sh reads
# (ORACLE_SNAP / ORACLE_MODEL, ORACLE_THREADS, ORACLE_IMAGE).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT_DIR="${OUT_DIR:-oracle-out}"
# The prompt ids are the SAME file for every checkpoint - the tokenizer is
# identical, so the committed tests/golden/prompts/*.ids are valid for both and
# re-tokenizing would be a chance to change them by accident.
IDS_DIR="${IDS_DIR:-tests/golden/prompts}"
mkdir -p "$OUT_DIR"
echo "=== golden set -> $OUT_DIR   ids <- $IDS_DIR   snap: ${ORACLE_SNAP:-HF cache/${ORACLE_MODEL:-default}}"
for p in prose code cjk; do
  echo "=== $p  start $(date -Is)"
  /usr/bin/time -v tools/oracle/run_in_container.sh \
    'python3 tools/oracle/dump.py "$SNAP" --prompt /ws/'"$IDS_DIR/$p"'.ids \
       --out /ws/'"$OUT_DIR/$p"'.golden.safetensors --gen 32' \
    > "$OUT_DIR/$p".log 2>&1
  echo "=== $p  done  $(date -Is)  rc=$?"
  tail -3 "$OUT_DIR/$p".log
done
echo "=== ALL DONE $(date -Is)"
