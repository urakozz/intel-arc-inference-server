#!/bin/bash
# tools/toolcall/a4_ref.sh - the A4 tool-call set and its CPU reference run for a MoE model
# (spec 15e Task 3: Ornith; spec 20e KL4: Kolibri-1; spec 21-QFN-v1e Task 5:
# Qwen3.8-Flash-Next) - and for Qwen3.8 itself - from the repo (or worktree) root.
#
#   tools/toolcall/a4_ref.sh qwen set|ref|status    Qwen3.8's own A4 reference (the one r11.a4 / r75.gates score
#                                                   against, A4_REF_DIR; the earlier one was lost): `set` copies
#                                                   the committed set tests/golden/toolcall (Qwen3.8's 36, the
#                                                   set the engine runs) to $OUT/set; `ref` runs
#                                                   oracle_generate.py --model qwen on the bf16 original
#                                                   Qwen/Qwen3.8-27B (54 GB): Qwen3_5ForCausalLM layer-streamed,
#                                                   the first K layers kept once read (K from the memory plan
#                                                   under the cap: ~54 of 64 at a 64g cap), bf16 GEMMs as fp32
#                                                   GEMMs rounded to bf16 (cpu_bf16.py), HF generate, 384 new ids,
#                                                   one scenario at a time -> $OUT/<name>.bf16.{ids,txt}
#                                                   (resumable; no .gap). OUT default $B70_DATA/oracle-out/qwen38/a4.
#   tools/toolcall/a4_ref.sh <ornith|kolibri|qwen4exp> set   the model's set: make_set.py --from
#                                                   tests/golden/toolcall (the 36 conversations of
#                                                   Qwen3.8's set, re-rendered and re-tokenised by
#                                                   the checkpoint's own template) -> $OUT/set;
#                                                   kolibri: the committed German set
#                                                   tests/golden/toolcall-kolibri-de (make_set.py
#                                                   --lang de, made from the release's tokenizer,
#                                                   which 20b's export copies) copied to $OUT/set;
#                                                   qwen4exp: make_set.py --from tests/golden/toolcall
#                                                   on the ORIGINAL's small files (TOK_DIR, default
#                                                   the HF cache's Qwen/Qwen3.8-Flash-Next: config,
#                                                   tokenizer.json, chat_template.jinja - Intel's
#                                                   checkpoint ships Qwen3.8's tokenizer.json, without
#                                                   the original's \p{M} split, spec 21-QFN-v1e) with
#                                                   transformers 5.19.0; the template is byte-identical
#                                                   to Qwen3.8's, so a scenario's ids differ only where
#                                                   the split does - the log (and $OUT/set/SAME) says
#                                                   how many of the 36 are Qwen3.8's ids exactly
#   tools/toolcall/a4_ref.sh <ornith|qwen4exp> ref     the reference: oracle_generate.py over $OUT/set
#                                                   -> $OUT/<name>.bf16.{ids,txt} (resumable)
#   tools/toolcall/a4_ref.sh <ornith|qwen> status     what is done
#   tools/toolcall/a4_ref.sh inkling set|ref|status   Inkling 2 Mini: `set` is make_set.py --from the 36 of
#                                                   tests/golden/toolcall, re-rendered by the checkpoint's
#                                                   own template (KWARGS default {"reasoning_effort": "none"},
#                                                   thinking off) -> $OUT/set; `ref` is oracle_generate.py
#                                                   --model inkling: tools/oracle/inkling_ref.py's bf16 CPU
#                                                   forward (the prompt checks' one), greedy, 384 new ids,
#                                                   stopping at the EOS id, in the foreground on the Linux
#                                                   machine (no container: PYTHON names the interpreter that
#                                                   has torch, safetensors and transformers, PYTHONPATH what it
#                                                   needs besides, INK_THREADS the threads, default 8)
#                                                   -> $OUT/<name>.bf16.{ids,txt,gap} (resumable).
#                                                   OUT default $B70_DATA/oracle-out/inkling/a4.
#
# The references (tools/toolcall/oracle_generate.py, by config.json's model_type):
#   ornith  tools/oracle/ornith_ref.py's layer-streamed Qwen3_5MoeForCausalLM on the INT4
#           checkpoint dequantised (15a: the engine's own weights), 384 new ids;
#   kolibri tools/oracle/kolibri_ref.py's KolibriRef in bf16 mode (`--model kolibri`) on the 156 GB
#           bf16 SOURCE (Aleph-Alpha/Kolibri-1-BF16 - KL4's reference; it runs wherever 20b runs,
#           decision 1: MODEL_DIR, DEVICE=cuda|xpu where there is one), 384 new ids (the set renders
#           thinking off). Score with `score.py --format hermes` (0 parse failures of the reference:
#           the hard bar).
#   qwen4exp tools/oracle/qwen4exp_ref.py's layer-streamed model (spec 21-QFN-v1a) on Intel's W4A16
#           AutoRound checkpoint (Intel/Qwen3.8-Flash-Next-W4A16-AutoRound: int4 g128 experts
#           dequantised, the PLE table from <snapshot>/ple-int8/ or <snapshot>-ple-int8/ when present - the engine's format),
#           384 new ids, transformers 5.19.0 from the qwen4exp site first on PYTHONPATH (SITE, default
#           $B70_DATA/oracle-out/flash-next/python-site - tools/oracle/qwen4exp_env.sh makes it in the container once). Q4_LAYERS=N truncates as the engine's --layers N (a test run;
#           the A4 reference is the whole model). Prepared for spec 22: the engine's A4 run on the
#           whole model waits for it (validation row 34's r34.a4_ref makes the reference).
#           Score with `score.py` (Qwen XML; 0 parse failures of the reference).
# Then on the GPU machine: tools/toolcall/engine_generate.sh with the same set and N_NEW, and
# tools/toolcall/score.py <dir> bf16 <engine runs> (validation rows 16 / 25).
#
# Where it runs:
#   the Mac  one container (image agnes-ref-img:latest - torch 2.14.1+cpu, transformers 5.15, g++ /
#            ninja for stream.py's C++ dequant) capped at MEM (28g), like every Mac oracle run;
#            `ref` is DETACHED (docker run -d, name a4-ref-<model>): poll with `status`. Refuses
#            while another container of that image runs (the Mac holds one 28 GB oracle at a
#            time) unless FORCE=1. The checkpoint must be complete in the HF cache:
#              uvx --from huggingface_hub hf download urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
#   Linux    tools/oracle/run_in_container.sh (ORACLE_IMAGE), in the foreground (run it detached
#            yourself): r16.a4_ref.
# Batched (BATCH, default auto; BATCH=1 is the old one-scenario-at-a-time run): oracle_generate.py
# --batch runs up to BATCH scenarios through each layer together (layer-major), each layer's
# weights dequantised once per pass for all of them, a finished scenario's slot refilled at once;
# every sequence's arithmetic stays its own (no op sees two sequences), so the ids are bitwise the
# sequential run's (tools/toolcall/test_oracle_batched.py: ids and every logits row, tiny Ornith /
# Kolibri, ragged prompts, different EOS points). RESIDENT (default auto; none = off) keeps
# dequantised weights resident once read - every layer's dense part, then whole layers of
# experts - in what the cap leaves (also bitwise: the same values, read from RAM instead of
# re-dequantised). The memory plan (estimates, from config.json and the container's cap - MEM)
# is the log's first lines (on Linux: run_in_container.sh's cap, 64g by default, or MEM=<N>g). ONLY=a,b,...
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
# layers resident) ~1.25x - ~14 h for 36, ~11.5 h for the 29 left. Each scenario's seconds,
# and every prompt pass's (and every 25th pass's) seconds, are in $OUT/ref.log. Ornith batched
# needs transformers 5.15 (agnes-ref-img; ornith_ref.layer_major refuses another version).
# Qwen3.8 (qwen, ESTIMATED from the test machine's fp32 rate and memory bandwidth, not measured): a prompt pass
# (960-2990 ids, mean ~1900) ~2-4 min, a decode step ~5-8 s (10 of 64 layers re-read, every bf16
# weight upcast to fp32 per call), up to 384 steps a scenario - ~10-16 h the set, one scenario at a
# time (the dense path does not batch).
#
# Env: OUT ($B70_DATA/oracle-out/<model>/a4, Qwen3.8-Flash-Next's flash-next/a4-ours: tools/oracle/README.md), DEVICE (kolibri's
# reference: cpu, cuda or xpu), MODEL_DIR (a snapshot
# directory instead of the HF cache's), NEW_TOKENS, KWARGS (the template variables, JSON;
# default make_set.py's), IMAGE, MEM,
# ORACLE_THREADS (12), INK_THREADS, PYTHON (inkling only), HF_CACHE, FORCE, BATCH (auto; qwen: ignored), RESIDENT (auto), ONLY, NAME, FOREGROUND; qwen4exp only:
# TOK_DIR (the original's small files for `set`), SITE, Q4_LAYERS, Q4_PLE.
set -eu
cd "$(dirname "$0")/../.."
m=${1:-}
step=${2:-}
case "$m" in
  qwen) repo=Qwen/Qwen3.8-27B; new=${NEW_TOKENS:-384}; set_dir=qwen38/a4 ;;   # the bf16 original (MODEL_DIR for another snapshot)
  ornith) repo=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ; new=${NEW_TOKENS:-384}; set_dir=ornith/a4 ;;
  kolibri) repo=Aleph-Alpha/Kolibri-1-BF16; new=${NEW_TOKENS:-384}; set_dir=kolibri/a4 ;;
  inkling) repo=thinkingmachines/Inkling-2-Mini; new=${NEW_TOKENS:-384}; set_dir=inkling/a4 ;;
  qwen4exp) repo=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound; new=${NEW_TOKENS:-384}; set_dir=flash-next/a4-ours ;;
  *) sed -n '2,/^set -eu/p' "$0" | sed '$d' >&2; exit 2 ;;
esac
case "$step" in set | ref | status) ;; *) sed -n '2,/^set -eu/p' "$0" | sed '$d' >&2; exit 2 ;; esac
if [ "$m" = qwen ] && [ "$step" = set ]; then   # Qwen3.8's own committed set: the one r11.a4's engine run reads
  OUT=${OUT:-${B70_DATA:?set B70_DATA, the data root}/oracle-out/$set_dir}
  mkdir -p "$OUT/set"
  cp tests/golden/toolcall/* "$OUT/set/"
  echo "a4_ref: qwen set: tests/golden/toolcall -> $OUT/set ($(ls "$OUT"/set/*.ids | wc -l | tr -d ' ') scenarios)"
  exit 0
fi
# Qwen3.8-Flash-Next's set is the ORIGINAL's tokenizer and template (its small files), not Intel's.
if [ "$m" = qwen4exp ] && [ "$step" = set ]; then
  repo=Qwen/Qwen3.8-Flash-Next
  if [ -n "${TOK_DIR:-}" ]; then MODEL_DIR=$TOK_DIR; else unset MODEL_DIR; fi
fi
cache_name="models--$(printf '%s' "$repo" | sed 's|/|--|g')"
OUT=${OUT:-${B70_DATA:?set B70_DATA, the data root}/oracle-out/$set_dir}
NAME=${NAME:-a4-ref-$m}
run_tag=bf16      # the outputs' tag: <name>.<run_tag>.{ids,txt,gap}
HF=${HF_CACHE:-$HOME/.cache/huggingface}
THREADS=${ORACLE_THREADS:-12}
mac=0
[ "$(uname -s)" = Darwin ] && mac=1
py=${PYTHON:-python3}
[ "$m" = inkling ] && KWARGS=${KWARGS:-'{"reasoning_effort": "none"}'}   # the template's thinking switch
kw=()
[ -n "${KWARGS:-}" ] && kw=(--kwargs "$KWARGS")

if [ "$step" = status ]; then
  [ $mac = 1 ] && docker ps -a --filter "name=^$NAME\$" --format '{{.Names}} {{.Status}}'
  n_set=$(ls "$OUT"/set/*.ids 2> /dev/null | wc -l | tr -d ' ')
  n_ref=$(ls "$OUT"/*."$run_tag".txt 2> /dev/null | wc -l | tr -d ' ')
  dn=DONE
  [ -f "$OUT/set/SAME" ] && echo "$m: $(cat "$OUT/set/SAME")"
  echo "$m: set $n_set/36 scenarios ($OUT/set), reference $n_ref/36 ($OUT/*.$run_tag.txt)$([ -f "$OUT/$dn" ] && echo ', DONE')"
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
# (The original's small files for qwen4exp's set: its index names shards that are not there.)
[ "$m" = qwen4exp ] && [ "$step" = set ] || python3 - "$snap" << 'EOF' || exit 2
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {}
files = {v for k, v in wm.items() if ".ngram_embedding.shard_" not in k}   # the bf16 n-gram shards are not downloaded
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
  cmd_set="$py tools/toolcall/make_set.py --from tests/golden/toolcall"
else
  [ -s "$OUT/set/manifest.json" ] || { echo "a4_ref: no set in $OUT/set - run '$0 $m set' first" >&2; exit 2; }
  cmd_ref="$py tools/toolcall/oracle_generate.py"
  [ "$m" = inkling ] && cmd_ref="$cmd_ref --model inkling"
  [ "$m" = qwen ] && cmd_ref="$cmd_ref --model qwen"
  [ "$m" = kolibri ] && cmd_ref="$cmd_ref --model kolibri --device ${DEVICE:-cpu}"
  [ "$m" = qwen4exp ] && cmd_ref="$cmd_ref --model qwen4exp"
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
done_file=DONE

# qwen4exp: transformers 5.19.0 from the site directory first on PYTHONPATH (made in the container
# once by tools/oracle/qwen4exp_env.sh - pip --no-deps --target, the image never changed), the
# reference's knobs passed through.
pre=""
q4env=()
if [ "$m" = qwen4exp ]; then
  SITE=${SITE:-${B70_DATA:?set B70_DATA, the data root}/oracle-out/flash-next/python-site}
  [[ "$SITE" =~ ^[A-Za-z0-9_./-]+$ ]] || { echo "a4_ref: SITE=$SITE" >&2; exit 2; }
  pre="tools/oracle/qwen4exp_env.sh $SITE > /dev/null && export PYTHONPATH=$SITE && "
  for v in Q4_LAYERS Q4_PLE; do
    x=${!v:-}
    [ -n "$x" ] || continue
    [[ "$x" =~ ^[A-Za-z0-9_.:/-]+$ ]] || { echo "a4_ref: $v=$x" >&2; exit 2; }
    q4env+=(-e "$v=$x")
    [ "$step" = ref ] && cmd_ref="$v=$x $cmd_ref"
  done
fi
# same_ids - how many of the set's scenarios are Qwen3.8's ids exactly (qwen4exp: the \p{M} split)
same_ids() {
  local n=0 t=0 f
  for f in "$1"/*.ids; do
    t=$((t + 1))
    cmp -s "$f" "tests/golden/toolcall/$(basename "$f")" && n=$((n + 1))
  done
  echo "same ids as Qwen3.8's set (tests/golden/toolcall): $n/$t scenarios" | tee "$1/SAME"
}

# Inkling: the interpreter on the Linux machine, no container; the run's threads are INK_THREADS.
if [ "$m" = inkling ]; then
  [ $mac = 0 ] || { echo "a4_ref: inkling runs on the Linux machine only" >&2; exit 2; }
  mkdir -p "$OUT/set"
  export OMP_NUM_THREADS=${INK_THREADS:-8} MKL_NUM_THREADS=${INK_THREADS:-8} CUDA_VISIBLE_DEVICES=
  if [ "$step" = set ]; then
    $cmd_set "$snap" "$OUT/set" ${kw[@]+"${kw[@]}"}
    exit $?
  fi
  $cmd_ref "$snap" "$OUT/set" "$OUT" --new-tokens "$new" 2>&1 | tee -a "$OUT/ref.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && echo done > "$OUT/$done_file"
  exit "$rc"
fi

# The container sees this tree at /ws (its working directory, so a relative OUT is the same path inside) and
# $B70_DATA/oracle-out at its own path: the run writes into OUT itself, and a killed run resumes there.
if [ $mac = 0 ]; then   # Linux: the reference image, foreground
  mkdir -p "$OUT/set"
  if [ -n "${MODEL_DIR:-}" ]; then export ORACLE_SNAP="${MODEL_DIR%/}"; else export ORACLE_MODEL="$cache_name"; fi
  [ -n "${ORACLE_THREADS:-}" ] && export ORACLE_THREADS
  rc=0
  # The container's cap is run_in_container.sh's (64g by default, swap off; above it only with the
  # cards idle): MEM (e.g. 48g) sets both the cap and the memory plan's budget, so they agree; unset,
  # the plan reads the cap from the container's memory.max.
  if [ "$step" = ref ] && [ -n "${MEM:-}" ]; then
    [[ "$MEM" =~ ^[0-9]+g$ ]] || { echo "a4_ref: MEM is <N>g (got $MEM)" >&2; exit 2; }
    cmd_ref="ORACLE_MEM_GB=${MEM%g} $cmd_ref"
    export ORACLE_MEMORY="$MEM"
  fi
  if [ "$step" = set ]; then
    tools/oracle/run_in_container.sh "$pre$cmd_set \"\$SNAP\" $OUT/set ${kw[*]+$(printf "'%s' " "${kw[@]}")}" || rc=$?
    [ "$rc" = 0 ] && [ "$m" = qwen4exp ] && same_ids "$OUT/set"
  else
    tools/oracle/run_in_container.sh "$pre$cmd_ref \"\$SNAP\" $OUT/set $OUT --new-tokens $new" 2>&1 | tee -a "$OUT/ref.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] && echo done > "$OUT/$done_file"
  fi
  exit "$rc"
fi

# The Mac: agnes-ref-img, capped, one oracle container at a time; OUT mounted as on Linux.
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
run=(docker run --name "$NAME" --memory "$MEM" --memory-swap "$MEM" "${mount[@]}" -v "$PWD":/ws -w /ws -v "$B70_DATA/oracle-out:$B70_DATA/oracle-out"
     -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/tmp/torch_ext
     -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" ${q4env[@]+"${q4env[@]}"})
if [ "$step" = set ] && [ "$m" = qwen4exp ]; then   # the site first, then the set
  "${run[@]}" --rm --entrypoint sh "$IMAGE" -c "$pre$cmd_set $S $OUT/set"
  [ "$m" = qwen4exp ] && same_ids "$OUT/set"
  exit 0
fi
if [ "$step" = set ]; then
  "${run[@]}" --rm --entrypoint python3 "$IMAGE" tools/toolcall/make_set.py --from tests/golden/toolcall \
    "$S" "$OUT/set" ${kw[@]+"${kw[@]}"}
  exit $?
fi
if [ "${FOREGROUND:-0}" = 1 ]; then   # attached: the caller holds the Mac's lock for the whole run
  "${run[@]}" --rm "$IMAGE" sh -c "$pre$cmd_ref $S $OUT/set $OUT --new-tokens $new >> $OUT/ref.log 2>&1 && echo done > $OUT/$done_file"
  exit $?
fi
"${run[@]}" -d "$IMAGE" sh -c "$pre$cmd_ref $S $OUT/set $OUT --new-tokens $new >> $OUT/ref.log 2>&1 && echo done > $OUT/$done_file"
echo "a4_ref: started $NAME (detached): '$0 $m status'; the log is $OUT/ref.log"
