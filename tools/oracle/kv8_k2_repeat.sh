#!/bin/bash
# Spec 18e Task 1, Review Focus 1: the 12a int8-KV probe repeated on K2-Horizon (rotkv at head_dim
# 128, tools/oracle/k2_kv8_probe.py over 18a's tools/oracle/k2_ref.py), on the Mac CPU. The
# operator's int4 checkpoint (what the engine runs) in a Linux container capped at 28 GB, the model
# layer at a time, the variants (bf16 reference, rotkv = the engine's scheme, per token, fp32
# attention) each with its own residual stream and KV cache. Run from the repo (or worktree) root:
#
#   hf download urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ     # 22.2 GB, once
#   tools/oracle/kv8_k2_repeat.sh                                            # everything (~4-8 h, est.)
#   STEPS="check golden" tools/oracle/kv8_k2_repeat.sh                       # a subset
#
# Detach it: nohup tools/oracle/kv8_k2_repeat.sh > /dev/null 2>&1 &  then
# `tail -f oracle-out-18e-k2/repeat.log`. Every output is in OUT (git-ignored: oracle-out*).
#
# Steps (each skipped when its output exists, so a re-run resumes):
#   check   the probe's and the reference's unit tests (tiny weights, seconds) - no checkpoint read
#   tok     K2's tokenisation, BOS 0 first: prose / code / cjk (tokenize.py --bos) and a long natural
#           prompt (tools/probe/mk_long_ids.py over docs/*.md + src/**/*.cc, BOS prepended), LONG_N ids
#   golden  prose / code / cjk + 32 continuation ids, all four variants: the decision-row table
#           (logits cos / KL / argmax at near-ties against bf16) and the routing diagnostic. The
#           continuation is oracle-out-k2/<p>.golden.safetensors' tokens when 18a's golden set is
#           here (teacher-forced, one prompt pass), else 32 greedy steps through the caches on the
#           reference's own argmax (decode's q path: fp32 q_rot)
#   long    the long prompt [:LONG_N] (4096 default), bf16 + rotkv + pt, with the q / K / V capture
#           - the 512-2047 / 2048-4095 / last-row buckets
#   replay  the fp64 attention-only replay of that capture at 1k / 2k / 4k real and 16k / 32k / 64k
#           tiled (the one-card int8 ceiling is ~92k, k2_plan_test): flat with depth or not
# Then `k2_kv8_probe.py summary` over every run into OUT/summary.md.
#
# What decides (plan 18e Review Focus 1; spec 12 §8's method): rotkv's golden decision-row mean
# logit cosine drop and KL against bf16, set beside l0-int8 / the f32attn envelope; the routing
# diagnostic's extra differing rows x layers over bf16 (K2's routers are tie-sensitive); the replay
# per-depth cosine. The tolerances for the K2 _kv8 gates (k2_golden_kv8_test's tie rule stays; the
# kernel bars of k2_kv8_kernels_test) are re-derived from these numbers.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT="${OUT:-oracle-out-18e-k2}"
REPO_ID="${REPO_ID:-urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ}"
IMAGE="${IMAGE:-agnes-ref-img:latest}"      # python 3.12, torch 2.14.1+cpu, transformers 5.15.0, g++, ninja
NAME="${NAME:-kv8-k2-ref}"
LONG_N="${LONG_N:-4096}"
THREADS="${THREADS:-16}"
GOLDEN="${GOLDEN:-oracle-out-k2}"           # 18a's golden set, if present (box-made; copy it here)
STEPS="${STEPS:-check tok golden long replay}"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$OUT"
LOG="$OUT/repeat.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') kv8 K2-Horizon repeat: steps [$STEPS], out $OUT"
has() { [[ " $STEPS " == *" $1 "* ]]; }

# The container: the repo at /ws, the HF cache read-only at /hf, 28 GB (the VM's page cache over the
# shards otherwise swaps macOS to a crawl - tools/oracle/README.md, "The Mac path").
start() {
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    docker rm -f "$NAME" >/dev/null 2>&1
    docker run -d --name "$NAME" --memory 28g --memory-swap 28g -v "$HF":/hf:ro -v "$PWD":/ws \
      -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/root/torch_ext \
      -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -w /ws "$IMAGE" sleep infinity >/dev/null ||
      { echo "docker run failed"; exit 2; }
  fi
}
start
run() {   # one probe invocation in the container, timed, into the log
  echo "--- $(date '+%T') $*"
  /usr/bin/time -p docker exec "$NAME" python3 tools/oracle/k2_kv8_probe.py "$@"
}

if has check; then
  docker exec "$NAME" python3 tools/oracle/test_k2_kv8_probe.py || { echo "probe unit test FAILED"; exit 1; }
fi
if [ "$STEPS" = "check" ]; then echo "=== $(date '+%F %T') done (check only)"; exit 0; fi

# The download: every shard the index names present and complete.
CACHE="$HF/hub/models--${REPO_ID//\//--}"
SNAP_HOST=$(ls -d "$CACHE"/snapshots/*/ 2>/dev/null | head -1)
[ -n "$SNAP_HOST" ] || { echo "no snapshot of $REPO_ID under $CACHE: hf download $REPO_ID first (22.2 GB)"; exit 2; }
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

if has tok; then
  for p in prose code cjk; do
    [ -s "$OUT/$p.ids" ] && { echo "$p.ids: done"; continue; }
    docker exec "$NAME" python3 tools/oracle/tokenize.py "$SNAP" encode --bos "tests/golden/prompts/$p.txt" \
      > "$OUT/$p.ids" || { echo "tokenize $p FAILED"; rm -f "$OUT/$p.ids"; exit 1; }
  done
  if [ ! -s "$OUT/long$LONG_N.ids" ]; then
    docker exec "$NAME" python3 tools/probe/mk_long_ids.py "$SNAP" "/ws/$OUT/long.raw.ids" "$((LONG_N - 1))" &&
      { echo 0; cat "$OUT/long.raw.ids"; } > "$OUT/long$LONG_N.ids" || { echo "long prompt FAILED"; exit 1; }
  fi
  wc -w "$OUT"/*.ids
fi

if has golden; then
  for p in prose code cjk; do
    [ -f "$OUT/$p.pt" ] && { echo "$p: done"; continue; }
    if [ -s "$GOLDEN/$p.golden.safetensors" ]; then
      docker exec "$NAME" python3 -c "
from safetensors import safe_open
print(' '.join(map(str, safe_open('$GOLDEN/$p.golden.safetensors', 'pt').get_tensor('tokens').tolist())))" \
        > "$OUT/$p.cont32.ids" || exit 1
      cmp -s "$GOLDEN/$p.ids" "$OUT/$p.ids" || echo "WARNING: $GOLDEN/$p.ids differs from $OUT/$p.ids - using $GOLDEN's"
      run run "$SNAP" --ids "$GOLDEN/$p.ids" --cont-ids "$OUT/$p.cont32.ids" --gen-cont 32 --out "$OUT/$p.pt"
    else
      run run "$SNAP" --ids "$OUT/$p.ids" --gen 32 --out "$OUT/$p.pt"
    fi
  done
fi

if has long; then
  if [ -f "$OUT/long$LONG_N.pt" ]; then echo "long: done"; else
    run run "$SNAP" --ids "$OUT/long$LONG_N.ids" --variants bf16,rotkv,pt \
      --capture "$OUT/long$LONG_N.cap.safetensors" --out "$OUT/long$LONG_N.pt"
  fi
fi

if has replay; then
  if [ -f "$OUT/long$LONG_N.replay.json" ]; then echo "replay: done"; else
    real="1024,2048,4096"; [ "$LONG_N" -ge 8192 ] && real="1024,2048,4096,8192"
    run replay "$OUT/long$LONG_N.cap.safetensors" --depths "$real" --tiles 16384,32768,65536 \
      --out "$OUT/long$LONG_N.replay.json" | tee "$OUT/long$LONG_N.replay.md"
  fi
fi

pts=$(ls "$OUT"/*.pt 2>/dev/null)
if [ -n "$pts" ]; then
  # shellcheck disable=SC2086
  docker exec "$NAME" python3 tools/oracle/k2_kv8_probe.py summary $pts > "$OUT/summary.md" 2>&1
  echo "summary: $OUT/summary.md"
fi
echo "=== $(date '+%F %T') done"
