#!/bin/bash
# Spec 12 §8 "Before 12b": plan 12a repeated on Qwen3.8 itself, on the Mac CPU, overnight.
# The bf16 base checkpoint (Qwen/Qwen3.8-27B) in a Linux container capped at 28 GB, the model
# layer-streamed (tools/oracle/stream.py), the five int8 KV schemes carried in the batch
# (tools/oracle/kv_int8_probe.py). Run from the repo (or worktree) root on the Mac:
#
#   tools/oracle/kv8_qwen38_repeat.sh            # everything, in order of value (~5-7 h)
#   STEPS="golden a4" tools/oracle/kv8_qwen38_repeat.sh   # a subset: check golden a4 long replay
#
# Detach it (it outlives the terminal): nohup tools/oracle/kv8_qwen38_repeat.sh > /dev/null 2>&1 &
# then `tail -f oracle-out-12a-qwen38/repeat.log`. Every output is in OUT (git-ignored).
#
# Steps (each skipped when its output exists, so a re-run resumes):
#   check   the probe's unit test (seconds, no weights) and the tokenizer round trip
#   golden  prose / code / cjk + their 32 golden greedy ids teacher-forced (the 2026-08-24
#           oracle's continuations, tools/oracle/README.md), all six variants: the decision-row
#           table - the stop rule against l0-int8's 0.999931742 and the Q2 golden tolerance
#   a4      the two A4 tool-call prompts (tests/golden/toolcall/t1_define-linear_l0,
#           t2_explain-box), six variants - the f32attn envelope
#   long    long32k.ids[:LONG_N] (4096; 8192 if the night allows), five variants, with the
#           q / K / V capture - the 512-2047 / 2048-4095 buckets and the last row (Q3's 32k bar)
#   replay  the fp64 attention replay of that capture at 1k / 2k / 4k real, 8k / 16k / 32k
#           tiled - K's outlier channels and V's per-token term, flat with depth or not
# Then `kv_int8_probe.py summary` over every run into OUT/summary.md.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT="${OUT:-oracle-out-12a-qwen38}"
REPO_ID="${REPO_ID:-Qwen/Qwen3.8-27B}"
IMAGE="${IMAGE:-agnes-ref-img:latest}"      # python 3.12, torch 2.14.1+cpu, transformers 5.15.0, g++
NAME="${NAME:-kv8-ref}"
LONG_N="${LONG_N:-4096}"
THREADS="${THREADS:-16}"
STEPS="${STEPS:-check golden a4 long replay}"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$OUT"
LOG="$OUT/repeat.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') kv8 Qwen3.8 repeat: steps [$STEPS], out $OUT"

# The download: every shard the index names present and complete.
CACHE="$HF/hub/models--${REPO_ID//\//--}"
SNAP_HOST=$(ls -d "$CACHE"/snapshots/*/ 2>/dev/null | head -1)
[ -n "$SNAP_HOST" ] || { echo "no snapshot of $REPO_ID under $CACHE: hf download $REPO_ID first"; exit 2; }
python3 - "$SNAP_HOST" <<'EOF' || exit 2
import json, os, sys
snap = sys.argv[1]
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(snap, s))]
if missing:
    sys.exit(f"download incomplete: {len(missing)} of {len(shards)} shards missing ({missing[:3]}...)")
print(f"checkpoint complete: {len(shards)} shards, {len(idx['weight_map'])} tensors")
EOF
SNAP="/hf/hub/models--${REPO_ID//\//--}/snapshots/$(basename "$SNAP_HOST")"

# The container: the repo at /ws, the HF cache read-only at /hf, 28 GB (tools/oracle/README.md,
# "The Mac path": uncapped, the VM's page cache over the checkpoint swaps macOS to a crawl).
if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --memory 28g --memory-swap 28g -v "$HF":/hf:ro -v "$PWD":/ws \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/root/torch_ext \
    -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -w /ws "$IMAGE" sleep infinity >/dev/null ||
    { echo "docker run failed"; exit 2; }
fi
run() {   # one probe invocation in the container, timed, into the log
  echo "--- $(date '+%T') $*"
  /usr/bin/time -p docker exec "$NAME" python3 tools/oracle/kv_int8_probe.py "$@"
}
has() { [[ " $STEPS " == *" $1 "* ]]; }

if has check; then
  docker exec "$NAME" python3 tools/oracle/test_kv_int8_probe.py || { echo "probe unit test FAILED"; exit 1; }
  run tokcheck "$SNAP" tests/golden/prompts/long32k.ids tests/golden/prompts/prose.ids \
    tests/golden/toolcall/t1_define-linear_l0.ids
fi

if has golden; then
  # The 32 greedy ids of the 2026-08-24 oracle run (Qwen3.8 W4A16, bf16 head; tools/oracle/
  # README.md "Sanity checks"), teacher-forced: the decision rows are the prompt's last row and
  # these 32. They are Qwen3.8's own greedy continuations, so the rows are the gate's rows.
  echo "3113 7810 279 1118 479 654 8980 1000 381 1142 440 279 1834 725 2213 13 3113 11292 279 4220 6092 1000 381 6992 11 321 539 5600 279 72103 1000 381" > "$OUT/prose.cont32.ids"
  echo "271 727 40523 17 19490 11 750 11 15131 1590 198 262 460 498 1030 8474 3620 11 750 681 15131 8 364 343 303 2663 60 271 727 40523 18 19490" > "$OUT/code.cont32.ids"
  echo "29545 271 95815 108553 97663 108447 96494 3709 98844 95895 97771 95726 114183 101650 100700 1710 271 550 220 99737 96863 271 99737 96863 95761 105064 97463 95793 100830 98252 96019 115534" > "$OUT/cjk.cont32.ids"
  for p in prose code cjk; do
    [ -f "$OUT/$p.pt" ] && { echo "$p: done"; continue; }
    run run "$SNAP" --fp32-matmul --ids "tests/golden/prompts/$p.ids" --cont-ids "$OUT/$p.cont32.ids" \
      --out "$OUT/$p.pt"
  done
fi

if has a4; then
  for t in t1_define-linear_l0 t2_explain-box; do
    [ -f "$OUT/$t.pt" ] && { echo "$t: done"; continue; }
    run run "$SNAP" --fp32-matmul --ids "tests/golden/toolcall/$t.ids" --out "$OUT/$t.pt"
  done
fi

if has long; then
  if [ -f "$OUT/long${LONG_N}.pt" ]; then echo "long: done"; else
    run run "$SNAP" --fp32-matmul --ids tests/golden/prompts/long32k.ids --n "$LONG_N" \
      --variants bf16,pt,kivi,rot,rotkv --capture "$OUT/long${LONG_N}.cap.safetensors" \
      --out "$OUT/long${LONG_N}.pt"
  fi
fi

if has replay; then
  if [ -f "$OUT/long${LONG_N}.replay.json" ]; then echo "replay: done"; else
    real="1024,2048,4096"; [ "$LONG_N" -ge 8192 ] && real="1024,2048,4096,8192"
    run replay "$OUT/long${LONG_N}.cap.safetensors" --depths "$real" --tiles 8192,16384,32768 \
      --out "$OUT/long${LONG_N}.replay.json" | tee "$OUT/long${LONG_N}.replay.md"
  fi
fi

pts=$(ls "$OUT"/*.pt 2>/dev/null)
if [ -n "$pts" ]; then
  # shellcheck disable=SC2086
  docker exec "$NAME" python3 tools/oracle/kv_int8_probe.py summary $pts > "$OUT/summary.md" 2>&1
  echo "summary: $OUT/summary.md"
fi
echo "=== $(date '+%F %T') done"
