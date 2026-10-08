#!/bin/bash
# tools/toolcall/a4_ref.sh - the A4 tool-call set and its CPU reference run for a MoE model
# (spec 15e Task 3: Ornith; spec 18d K4: K2-Horizon; spec 20e KL4: Kolibri-1), from the repo (or
# worktree) root.
#
#   tools/toolcall/a4_ref.sh <ornith|k2|kolibri> set   the model's set: make_set.py --from
#                                                   tests/golden/toolcall (the 36 conversations of
#                                                   Qwen3.8's set, re-rendered and re-tokenised by
#                                                   the checkpoint's own template) -> $OUT/set;
#                                                   kolibri: the committed German set
#                                                   tests/golden/toolcall-kolibri-de (make_set.py
#                                                   --lang de, made from the release's tokenizer,
#                                                   which 20b's export copies) copied to $OUT/set
#   tools/toolcall/a4_ref.sh <ornith|k2> ref       the reference: oracle_generate.py over $OUT/set
#                                                   -> $OUT/<name>.bf16.{ids,txt} (resumable)
#   tools/toolcall/a4_ref.sh <ornith|k2> status    what is done
#
# The references (tools/toolcall/oracle_generate.py, by config.json's model_type):
#   ornith  tools/oracle/ornith_ref.py's layer-streamed Qwen3_5MoeForCausalLM on the INT4
#           checkpoint dequantised (15a: the engine's own weights), 192 new ids;
#   k2      tools/oracle/k2_ref.py's port in bf16 mode on the int4 checkpoint dequantised (18a's
#           reference; MODEL_DIR=<the bf16 original's snapshot> for the true bf16 model), 512 new
#           ids (its replies open with reasoning; the set renders reasoning_effort low);
#   kolibri tools/oracle/kolibri_ref.py's KolibriRef in bf16 mode (`--model kolibri`) on the 156 GB
#           bf16 SOURCE (Aleph-Alpha/Kolibri-1-BF16 - KL4's reference; it runs wherever 20b runs,
#           decision 1: MODEL_DIR, DEVICE=cuda|xpu where there is one), 192 new ids (the set renders
#           thinking off). Score with `score.py --format hermes` (0 parse failures of the reference:
#           the hard bar).
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
# Batched (BATCH, default auto; BATCH=1 is the old one-scenario-at-a-time run): oracle_generate.py
# --batch runs up to BATCH scenarios through each layer together (layer-major), each layer's
# weights dequantised once per pass for all of them, a finished scenario's slot refilled at once;
# every sequence's arithmetic stays its own (no op sees two sequences), so the ids are bitwise the
# sequential run's (tools/toolcall/test_oracle_batched.py: ids and every logits row, tiny Ornith /
# K2 / Kolibri, ragged prompts, different EOS points). RESIDENT (default auto; none = off) keeps
# dequantised weights resident once read - every layer's dense part, then whole layers of
# experts - in what the cap leaves (also bitwise: the same values, read from RAM instead of
# re-dequantised). The memory plan (estimates, from config.json and the container's cap - MEM)
# is the log's first lines (on the box, no cap: MemAvailable at start, or MEM=<N>g). ONLY=a,b,...
# runs just those scenarios (the cross-check);
# NAME the container (a4-ref-<model>).
#
# Time: `set` minutes. `ref`, Ornith sequential MEASURED on the Mac (2026-10-08, 6 of 36, 12
# threads): 920-2694 s a scenario, ~29 min mean, ~18 h the set; derived from two scenarios of
# near-equal prompt length, a ~2800-id prompt forward ~1900 s and a decode step ~6 s (~8.7 s in
# py-spy's window). py-spy over 800 s of it: a decode step is 38% routed-expert dequant + copy
# (per layer and expert; a batch shares only the experts its rows have in common, ~6% fewer at 4
# rows), 23% the GDN's bf16 depthwise conv1d (76 ms a call, per sequence), 13% grouped_mm, 9%
# linears, 6% attention, 5% GDN - per sequence - and 1% the dense layers' (prefetched) dequant.
# So for Ornith (ESTIMATED): batching alone ~5%; with RESIDENT=auto at MEM=60g (26 of 40 expert
# layers resident) ~1.25x - ~14 h for 36, ~11.5 h for the 29 left. K2 (512 ids; its int4 step
# ~7 s, ~70% of it dequant - tools/oracle/README.md's estimate, not measured): ~1-1.5 days
# sequential, ~0.6-1 day batched (8) + RESIDENT at 54-60g (ESTIMATED). Each scenario's seconds,
# and every prompt pass's (and every 25th pass's) seconds, are in $OUT/ref.log. Ornith batched
# needs transformers 5.15 (agnes-ref-img; ornith_ref.layer_major refuses another version).
#
# Env: OUT (oracle-out-<model>-a4; git-ignored like every oracle-out*), DEVICE (kolibri's
# reference: cpu, cuda or xpu), MODEL_DIR (a snapshot
# directory instead of the HF cache's), NEW_TOKENS, KWARGS (K2's template variables, JSON;
# default make_set.py's {"tool_call_format": "xml", "reasoning_effort": "low"}), IMAGE, MEM,
# ORACLE_THREADS (12), HF_CACHE, FORCE, BATCH (auto), RESIDENT (auto), ONLY, NAME.
set -eu
cd "$(dirname "$0")/../.."
m=${1:-}
step=${2:-}
case "$m" in
  ornith) repo=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ; new=${NEW_TOKENS:-192} ;;
  k2) repo=urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ; new=${NEW_TOKENS:-512} ;;
  kolibri) repo=Aleph-Alpha/Kolibri-1-BF16; new=${NEW_TOKENS:-192} ;;
  *) sed -n '2,/^set -eu/p' "$0" | sed '$d' >&2; exit 2 ;;
esac
case "$step" in set | ref | status) ;; *) sed -n '2,/^set -eu/p' "$0" | sed '$d' >&2; exit 2 ;; esac
cache_name="models--$(printf '%s' "$repo" | sed 's|/|--|g')"
OUT=${OUT:-oracle-out-$m-a4}
NAME=${NAME:-a4-ref-$m}
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
tmpl=chat_template.jinja
[ "$m" = kolibri ] && tmpl=tokenizer_config.json   # Kolibri's template is tokenizer_config.json's string
for f in config.json generation_config.json tokenizer.json $tmpl; do
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
if [ "$step" = set ] && [ "$m" = kolibri ]; then   # the committed German set (spec 20e)
  mkdir -p "$OUT/set"
  cp tests/golden/toolcall-kolibri-de/* "$OUT/set/"
  echo "a4_ref: kolibri set: tests/golden/toolcall-kolibri-de -> $OUT/set ($(ls "$OUT"/set/*.ids | wc -l | tr -d ' ') scenarios)"
  exit 0
fi
if [ "$step" = set ]; then
  cmd_set="python3 tools/toolcall/make_set.py --from tests/golden/toolcall"
else
  [ -s "$OUT/set/manifest.json" ] || { echo "a4_ref: no set in $OUT/set - run '$0 $m set' first" >&2; exit 2; }
  cmd_ref="python3 tools/toolcall/oracle_generate.py"
  [ "$m" = kolibri ] && cmd_ref="$cmd_ref --model kolibri --device ${DEVICE:-cpu}"
  BATCH=${BATCH:-auto}
  RESIDENT=${RESIDENT:-auto}
  [[ "$BATCH" =~ ^(auto|[1-9][0-9]*)$ ]] || { echo "a4_ref: BATCH is auto or a count >= 1 (got $BATCH)" >&2; exit 2; }
  [[ "$RESIDENT" =~ ^(auto|none)$ ]] || { echo "a4_ref: RESIDENT is auto or none (got $RESIDENT)" >&2; exit 2; }
  cmd_ref="$cmd_ref --batch $BATCH --resident $RESIDENT"
  if [ -n "${ONLY:-}" ]; then
    [[ "$ONLY" =~ ^[A-Za-z0-9_.,-]+$ ]] || { echo "a4_ref: ONLY is a comma-separated list of scenario names (got $ONLY)" >&2; exit 2; }
    cmd_ref="$cmd_ref --only $ONLY"
  fi
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
  # No container cap here: the memory plan reads MemAvailable at start, or MEM (e.g. 60g) when set
  if [ "$step" = ref ] && [ -n "${MEM:-}" ]; then
    [[ "$MEM" =~ ^[0-9]+g$ ]] || { echo "a4_ref: MEM is <N>g (got $MEM)" >&2; exit 2; }
    cmd_ref="ORACLE_MEM_GB=${MEM%g} $cmd_ref"
  fi
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
