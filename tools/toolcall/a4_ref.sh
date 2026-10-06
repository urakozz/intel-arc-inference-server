#!/bin/bash
# tools/toolcall/a4_ref.sh - the A4 tool-call set and its CPU reference run for a MoE model
# (spec 15e Task 3: Ornith; spec 18d K4: K2-Horizon), from the repo (or worktree) root.
#
#   tools/toolcall/a4_ref.sh <ornith|k2> set       the model's set: make_set.py --from
#                                                   tests/golden/toolcall (the 36 conversations of
#                                                   Qwen3.8's set, re-rendered and re-tokenised by
#                                                   the checkpoint's own template) -> $OUT/set
#   tools/toolcall/a4_ref.sh <ornith|k2> ref       the reference: oracle_generate.py over $OUT/set
#                                                   -> $OUT/<name>.bf16.{ids,txt} (resumable)
#   tools/toolcall/a4_ref.sh <ornith|k2> status    what is done
#
# The references (tools/toolcall/oracle_generate.py, by config.json's model_type):
#   ornith  tools/oracle/ornith_ref.py's layer-streamed Qwen3_5MoeForCausalLM on the INT4
#           checkpoint dequantised (15a: the engine's own weights), 192 new ids;
#   k2      tools/oracle/k2_ref.py's port in bf16 mode on the int4 checkpoint dequantised (18a's
#           reference; MODEL_DIR=<the bf16 original's snapshot> for the true bf16 model), 512 new
#           ids (its replies open with reasoning; the set renders reasoning_effort low).
# Then on the box: tools/toolcall/engine_generate.sh with the same set and N_NEW, and
# tools/toolcall/score.py <dir> bf16 <engine runs> (box-validation-queue rows 16 / 25).
#
# Where it runs:
#   the Mac  one container (image agnes-ref-img:latest - torch 2.14.1+cpu, transformers 5.15, g++ /
#            ninja for stream.py's C++ dequant) capped at MEM (28g), like every Mac oracle run;
#            `ref` is DETACHED (docker run -d, name a4-ref-<model>): poll with `status`. Refuses
#            while another container of that image runs (the Mac holds one 28 GB oracle at a
#            time) unless FORCE=1. The checkpoint must be complete in the HF cache:
#              uvx --from huggingface_hub hf download urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
#              uvx --from huggingface_hub hf download urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ
#   the box  tools/oracle/run_in_container.sh (ORACLE_IMAGE), in the foreground (the runbook's
#            stages run detached already): r16.a4_ref / r25.a4_ref.
# Time (estimated, not measured): `set` minutes; `ref` hours - a layer-streamed decode step
# reads every layer's dense weights from the page cache: 36 scenarios x up to 192 (Ornith) / 512
# (K2) ids. Each scenario's prompt forward and per-step seconds are in $OUT/ref.log.
#
# Env: OUT (oracle-out-<model>-a4; git-ignored like every oracle-out*), MODEL_DIR (a snapshot
# directory instead of the HF cache's), NEW_TOKENS, KWARGS (K2's template variables, JSON;
# default make_set.py's {"tool_call_format": "xml", "reasoning_effort": "low"}), IMAGE, MEM,
# ORACLE_THREADS (12), HF_CACHE, FORCE.
set -eu
cd "$(dirname "$0")/../.."
m=${1:-}
step=${2:-}
case "$m" in
  ornith) repo=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ; new=${NEW_TOKENS:-192} ;;
  k2) repo=urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ; new=${NEW_TOKENS:-512} ;;
  *) sed -n '2,40p' "$0" >&2; exit 2 ;;
esac
case "$step" in set | ref | status) ;; *) sed -n '2,40p' "$0" >&2; exit 2 ;; esac
cache_name="models--$(printf '%s' "$repo" | sed 's|/|--|g')"
OUT=${OUT:-oracle-out-$m-a4}
NAME=a4-ref-$m
HF=${HF_CACHE:-$HOME/.cache/huggingface}
THREADS=${ORACLE_THREADS:-12}
mac=0
[ "$(uname -s)" = Darwin ] && mac=1
kw=()
[ -n "${KWARGS:-}" ] && kw=(--kwargs "$KWARGS")

if [ "$step" = status ]; then
  [ $mac = 1 ] && docker ps -a --filter "name=^$NAME\$" --format '{{.Names}} {{.Status}}'
  n_set=$(ls "$OUT"/set/*.ids 2> /dev/null | wc -l | tr -d ' ')
  n_ref=$(ls "$OUT"/*.bf16.txt 2> /dev/null | wc -l | tr -d ' ')
  echo "$m: set $n_set/36 scenarios ($OUT/set), reference $n_ref/36 ($OUT/*.bf16.txt)$([ -f "$OUT/DONE" ] && echo ', DONE')"
  tail -3 "$OUT/ref.log" 2> /dev/null || true
  exit 0
fi

# The snapshot: MODEL_DIR, or the HF cache's (complete: config, tokenizer, every shard).
if [ -n "${MODEL_DIR:-}" ]; then
  snap=${MODEL_DIR%/}/
else
  snap=$(ls -d "$HF/hub/$cache_name/snapshots/"*/ 2> /dev/null | head -1)
  [ -n "$snap" ] || { echo "a4_ref: no snapshot of $repo under $HF/hub - first: uvx --from huggingface_hub hf download $repo" >&2; exit 2; }
  if ls "$HF/hub/$cache_name/blobs/"*.incomplete > /dev/null 2>&1; then
    echo "a4_ref: the download of $repo is not complete ($HF/hub/$cache_name/blobs/*.incomplete)" >&2; exit 2
  fi
fi
for f in config.json generation_config.json tokenizer.json chat_template.jinja; do
  [ -e "$snap$f" ] || { echo "a4_ref: $snap$f is missing" >&2; exit 2; }
done
python3 - "$snap" << 'EOF' || exit 2
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
files = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else set()
missing = [f for f in sorted(files) if not os.path.exists(os.path.join(d, f))]
if missing:
    sys.exit(f"a4_ref: {d} lacks {len(missing)} shards, e.g. {missing[:2]}")
EOF
if [ "$step" = set ]; then
  cmd_set="python3 tools/toolcall/make_set.py --from tests/golden/toolcall"
else
  [ -s "$OUT/set/manifest.json" ] || { echo "a4_ref: no set in $OUT/set - run '$0 $m set' first" >&2; exit 2; }
  cmd_ref="python3 tools/toolcall/oracle_generate.py"
fi

# The container sees this tree only (mounted at /ws): a run writes into a real directory inside
# it. On the box OUT may be elsewhere (the runbook keeps data in the data tree, linked into the
# trees): the run then works in oracle-out-<model>-a4.partial/, seeded with what OUT already
# holds (so it resumes), and copies everything to OUT when the container exits.
case "$OUT" in /*) inside=0 ;; *) inside=1 ;; esac
[ -L "$OUT" ] && inside=0
if [ $mac = 0 ]; then   # the box: the reference image, foreground
  work=$OUT
  if [ $inside = 0 ]; then
    work=oracle-out-$m-a4.partial
    mkdir -p "$work" "$OUT"
    cp -R "$OUT/." "$work/"
  fi
  mkdir -p "$work/set"
  if [ -n "${MODEL_DIR:-}" ]; then export ORACLE_SNAP="${MODEL_DIR%/}"; else export ORACLE_MODEL="$cache_name"; fi
  [ -n "${ORACLE_THREADS:-}" ] && export ORACLE_THREADS
  rc=0
  if [ "$step" = set ]; then
    tools/oracle/run_in_container.sh "$cmd_set \"\$SNAP\" /ws/$work/set ${kw[*]+$(printf "'%s' " "${kw[@]}")}" || rc=$?
  else
    tools/oracle/run_in_container.sh "$cmd_ref \"\$SNAP\" /ws/$work/set /ws/$work --new-tokens $new" 2>&1 | tee -a "$work/ref.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] && echo done > "$work/DONE"
  fi
  if [ "$work" != "$OUT" ]; then cp -R "$work/." "$OUT/"; echo "a4_ref: copied $work -> $OUT"; fi
  exit "$rc"
fi

# The Mac: agnes-ref-img, capped, one oracle container at a time; OUT inside this tree.
[ $inside = 1 ] || { echo "a4_ref: on the Mac OUT must be a directory inside this tree (got $OUT)" >&2; exit 2; }
mkdir -p "$OUT/set"
IMAGE=${IMAGE:-agnes-ref-img:latest}
MEM=${MEM:-28g}
others=$(docker ps --filter "ancestor=$IMAGE" --format '{{.Names}}' | grep -vx "$NAME" || true)
if [ -n "$others" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "a4_ref: another $IMAGE container is running ($others): wait for it, or FORCE=1" >&2; exit 2
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "a4_ref: container $NAME exists: '$0 $m status', or 'docker rm $NAME' once it has exited" >&2; exit 2
fi
if [ -n "${MODEL_DIR:-}" ]; then mount=(-v "${MODEL_DIR%/}":/snap:ro); S=/snap; else mount=(-v "$HF":/hf:ro); S="/hf/${snap#"$HF/"}"; fi
run=(docker run --name "$NAME" --memory "$MEM" --memory-swap "$MEM" "${mount[@]}" -v "$PWD":/ws -w /ws
     -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/tmp/torch_ext
     -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS")
if [ "$step" = set ]; then
  "${run[@]}" --rm --entrypoint python3 "$IMAGE" tools/toolcall/make_set.py --from tests/golden/toolcall \
    "$S" "/ws/$OUT/set" ${kw[@]+"${kw[@]}"}
  exit $?
fi
"${run[@]}" -d "$IMAGE" sh -c "$cmd_ref $S /ws/$OUT/set /ws/$OUT --new-tokens $new >> /ws/$OUT/ref.log 2>&1 && echo done > /ws/$OUT/DONE"
echo "a4_ref: started $NAME (detached): '$0 $m status'; the log is $OUT/ref.log"
