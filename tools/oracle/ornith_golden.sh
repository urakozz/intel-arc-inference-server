#!/bin/bash
# tools/oracle/ornith_golden.sh - spec 15a's Ornith reference ON THE MAC, from the int4 checkpoint
# (tools/oracle/ornith_ref.py; plan 15a "The reference from the int4 checkpoint").
#
#   tools/oracle/ornith_golden.sh            # from the repo (or worktree) root; detached
#   tools/oracle/ornith_golden.sh --status   # where it is
#
# One container (`ornith-ref`, image agnes-ref-img:latest: torch 2.14.1+cpu, transformers 5.15.0,
# g++ / ninja for stream.py's C++ dequant), capped at 28 GB like every Mac oracle run (the page
# cache over the 23 GB checkpoint counts against the cap; uncapped, the VM grows and macOS swaps -
# tools/oracle/README.md "The Mac path"). It runs `facts` first (stops if the checkpoint is not
# the one spec 15 §13 read), then prose, code, cjk serially into oracle-out-ornith/
# (<p>.golden.safetensors, <p>.log) with the MTP head's M1 reference into oracle-out-ornith-mtp/
# (m1/<p>.mtp.safetensors, <p>.cont256.ids). Resumable: a prompt whose golden file and M1 file
# both exist is skipped. Detached (`docker run -d`): it outlives the terminal; poll with --status.
#
# Needs: the checkpoint in the HF cache, complete -
#   uvx --from huggingface_hub hf download urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
# (22.99 GB; plan 15a). Refuses to start while another agnes-ref-img container runs (the
# Mac is shared: a second 28 GB oracle would swap), unless FORCE=1.
#
# Env: ORACLE_THREADS (default 12), MEM (default 28g), OUT_DIR, MTP_OUT_DIR, PROMPTS
# ("prose code cjk"), IMAGE.
set -eu
cd "$(dirname "$0")/../.."
IMAGE=${IMAGE:-agnes-ref-img:latest}
MEM=${MEM:-28g}
THREADS=${ORACLE_THREADS:-12}
OUT=${OUT_DIR:-oracle-out-ornith}
MTP=${MTP_OUT_DIR:-oracle-out-ornith-mtp}
PROMPTS=${PROMPTS:-prose code cjk}
NAME=ornith-ref
MODEL=models--urakozz--Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
HF=${HF_CACHE:-$HOME/.cache/huggingface}

if [ "${1:-}" = "--status" ]; then
  docker ps -a --filter "name=^$NAME\$" --format '{{.Names}} {{.Status}}'
  for p in $PROMPTS; do
    printf '%-6s golden %-3s m1 %-3s  ' "$p" \
      "$([ -s "$OUT/$p.golden.safetensors" ] && echo yes || echo no)" \
      "$([ -s "$MTP/m1/$p.mtp.safetensors" ] && echo yes || echo no)"
    tail -1 "$OUT/$p.log" 2>/dev/null || echo
  done
  exit 0
fi

snap=$(ls -d "$HF/hub/$MODEL/snapshots/"*/ 2>/dev/null | head -1)
[ -n "$snap" ] || { echo "no snapshot of $MODEL under $HF/hub (download it first, plan 15a)" >&2; exit 2; }
if ls "$HF/hub/$MODEL/blobs/"*.incomplete > /dev/null 2>&1; then
  echo "the download is not complete ($HF/hub/$MODEL/blobs/*.incomplete)" >&2; exit 2
fi
for f in model-0000{1,2,3,4,5}-of-00005.safetensors model_extra_tensors.safetensors tokenizer.json; do
  [ -e "$snap$f" ] || { echo "$snap$f is missing" >&2; exit 2; }
done
others=$(docker ps --filter "ancestor=$IMAGE" --format '{{.Names}}' | grep -vx "$NAME" || true)
if [ -n "$others" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "another $IMAGE container is running ($others): wait for it, or FORCE=1" >&2; exit 2
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME exists: --status, or 'docker rm $NAME' once it has exited" >&2; exit 2
fi
mkdir -p "$OUT" "$MTP/m1"
rel=${snap#"$HF/"}
docker run -d --name "$NAME" --memory "$MEM" --memory-swap "$MEM" \
  -v "$HF":/hf:ro -v "$PWD":/ws -w /ws \
  -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/tmp/torch_ext \
  -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" \
  "$IMAGE" sh -c "
    S=/hf/$rel
    python3 tools/oracle/ornith_ref.py facts \"\$S\" > /ws/$OUT/facts.log 2>&1 || exit 1
    for p in $PROMPTS; do
      [ -s /ws/$OUT/\$p.golden.safetensors ] && [ -s /ws/$MTP/m1/\$p.mtp.safetensors ] && continue
      python3 tools/oracle/ornith_ref.py run \"\$S\" --prompt /ws/tests/golden/prompts/\$p.ids \
        --out /ws/$OUT/\$p.golden.safetensors --gen 32 --mtp-out /ws/$MTP \
        > /ws/$OUT/\$p.log 2>&1 || exit 1
    done
    echo done > /ws/$OUT/DONE"
echo "started $NAME (detached): tools/oracle/ornith_golden.sh --status; logs in $OUT/"
