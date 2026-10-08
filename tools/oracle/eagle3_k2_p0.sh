#!/bin/bash
# The EAGLE3 K2 P0 on the Mac (docs/probe-eagle3-k2-2026-10-08.md; tools/oracle/README.md "The
# EAGLE3 K2 P0"): is EAGLE-3 (Siladrim/K2-Horizon-MoVA-36B-A4B-EAGLE3) worth building for
# K2-Horizon? Measured acceptance (teacher-forced, greedy) and a verify cost derived from the
# recorded routes, against spec 19e's prompt lookup. Run from the repo (or worktree) root:
#
#   tools/oracle/eagle3_k2_p0.sh                     # everything, resumable
#   DRY_RUN=1 tools/oracle/eagle3_k2_p0.sh           # the plan and estimates: no container, no model
#   STEPS="accept summary" ARMS=bf16 tools/oracle/eagle3_k2_p0.sh
#
# Detach it (it outlives the terminal): nohup tools/oracle/eagle3_k2_p0.sh > /dev/null 2>&1 &
# then `tail -f oracle-out-eagle3-k2/p0.log`. Every output is in OUT (git-ignored).
#
# Steps (each resumable; a model step refuses while ANOTHER agnes-ref-img container runs - one
# oracle at a time on this Mac's 62.8 GB Docker - unless FORCE=1; it never stops one):
#   a4       PREREQUISITE: K2's A4 set and reference continuations, tools/toolcall/a4_ref.sh k2
#            set + ref -> A4_OUT (oracle-out-k2-a4; the ref is hours, detached in its own
#            container a4-ref-k2). Done when A4_OUT/DONE exists; while it runs, or right after
#            starting it, the driver stops here (exit 4) - re-run once DONE. A4_PARTIAL=1 goes on
#            with the scenarios already done; A4=0 leaves the A4 set out (golden only).
#   check    the unit tests (tiny models): test_eagle3_ref, test_k2_taps, test_eagle3_accept,
#            test_eagle3_cost, test_dflash_accept (accept_rows is reused)
#   facts    eagle3_ref.py facts: the drafter's tensors, d2t round trip, its embedding vs K2's bytes
#   golden   the golden prompts (prose / code / cjk) with K2 greedy continuations: GOLDEN_DIR
#            (oracle-out-k2, 18a's k2_oracle.sh: <p>.ids + <p>.golden.safetensors, 32 ids) if
#            present, else tokenize.py --bos + k2_ref.py run --gen GOLDEN_GEN (128) here
#   sources  OUT/dumps/sources.json: golden/<p> + a4/<scenario> (each with a reference)
#   dump     k2_taps.py run: one K2 forward per source -> OUT/dumps/*.k2taps.safetensors
#   accept   eagle3_accept.py run: ARMS (bf16,int8,int8h,int4h), K 1..5 -> OUT/accept/accept.*.json
#   lookup   eagle3_accept.py lookup: spec 19e's prompt lookup on the same anchors
#   cost     eagle3_cost.py run: verify bytes at M = 1..6 from the routes, the drafter's bytes
#   summary  eagle3_accept.py summary -> OUT/summary.md (E_K, a_j, S_K at DEPTH)
#
# Env: OUT, A4_OUT, GOLDEN_DIR, GOLDEN_GEN, K2_REPO (the int4 checkpoint the engine runs),
# DRAFTER_REPO, IMAGE, MEM (54g), THREADS (16), STEPS, ARMS, WINDOW (query | anchor), DEPTH
# (4096), MATMUL (fp32 | bf16), KEEP=1 (keep the container at exit), FORCE=1, A4, A4_PARTIAL,
# K2_EMBED=1 (also the arms with K2's embedding, `<arm>-k2emb`: when facts finds it differs).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT="${OUT:-oracle-out-eagle3-k2}"
A4_OUT="${A4_OUT:-oracle-out-k2-a4}"
GOLDEN_DIR="${GOLDEN_DIR:-oracle-out-k2}"
GOLDEN_GEN="${GOLDEN_GEN:-128}"
K2_REPO="${K2_REPO:-urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ}"
DRAFTER_REPO="${DRAFTER_REPO:-Siladrim/K2-Horizon-MoVA-36B-A4B-EAGLE3}"
IMAGE="${IMAGE:-agnes-ref-img:latest}"     # python 3.12, torch 2.14.1+cpu, transformers 5.15.0, g++ / ninja
MEM="${MEM:-54g}"
THREADS="${THREADS:-16}"
STEPS="${STEPS:-a4 check facts golden sources dump accept lookup cost summary}"
ARMS="${ARMS:-bf16,int8,int8h,int4h}"
WINDOW="${WINDOW:-query}"
DEPTH="${DEPTH:-4096}"
MATMUL="${MATMUL:-fp32}"
A4="${A4:-1}"
DRY_RUN="${DRY_RUN:-}"
NAME="${NAME:-eagle3-k2-ref}"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
case "$OUT" in /*) echo "OUT must be a directory inside this tree (the container sees it as /ws)"; exit 2 ;; esac
mkdir -p "$OUT"
LOG="$OUT/p0.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') EAGLE3 K2 P0: steps [$STEPS], arms $ARMS, window $WINDOW, out $OUT${DRY_RUN:+ (dry run)}"
has() { [[ " $STEPS " == *" $1 "* ]]; }

# --- snapshots (never downloaded here) -----------------------------------------------------
snap_of() {   # repo -> host snapshot dir ("" if none)
  ls -d "$HF/hub/models--${1//\//--}"/snapshots/*/ 2> /dev/null | head -1
}
complete() {  # host snapshot dir, files... -> 0 if every file resolves and no blob is .incomplete
  local s="$1"; shift
  [ -n "$s" ] || return 1
  ls "$s"/../../blobs/*.incomplete > /dev/null 2>&1 && return 1
  for f in "$@"; do [ -e "$s/$f" ] || return 1; done
  if [ -f "$s/model.safetensors.index.json" ]; then
    python3 -c "import json,os,sys; d=sys.argv[1]; m=set(json.load(open(os.path.join(d,'model.safetensors.index.json')))['weight_map'].values()); sys.exit(0 if all(os.path.exists(os.path.join(d,f)) for f in m) else 1)" "$s" || return 1
  fi
}
K2_HOST=$(snap_of "$K2_REPO")
E3_HOST=$(snap_of "$DRAFTER_REPO")
K2_OK=0; complete "$K2_HOST" config.json tokenizer.json model.safetensors.index.json && K2_OK=1
E3_OK=0; complete "$E3_HOST" config.json model.safetensors && E3_OK=1
in_c() { echo "/hf/${1#"$HF/"}"; }       # host snapshot path -> the container's
K2_SNAP=""; [ -n "$K2_HOST" ] && K2_SNAP=$(in_c "${K2_HOST%/}")
E3_SNAP=""; [ -n "$E3_HOST" ] && E3_SNAP=$(in_c "${E3_HOST%/}")
echo "K2 $K2_REPO: ${K2_HOST:-not in $HF/hub} ($([ $K2_OK = 1 ] && echo complete || echo 'INCOMPLETE / missing: uvx --from huggingface_hub hf download '"$K2_REPO"))"
echo "drafter $DRAFTER_REPO: ${E3_HOST:-not in $HF/hub} ($([ $E3_OK = 1 ] && echo complete || echo 'INCOMPLETE / missing: uvx --from huggingface_hub hf download '"$DRAFTER_REPO"))"

# --- the A4 prerequisite's state --------------------------------------------------------------
a4_done=$(ls "$A4_OUT"/*.bf16.txt 2> /dev/null | wc -l | tr -d ' ')
a4_total=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$A4_OUT/set/manifest.json" 2> /dev/null || echo 36)
a4_running=$(docker ps --filter "name=^a4-ref-k2\$" --format '{{.Names}}' 2> /dev/null)
echo "A4 ($A4_OUT): $a4_done/$a4_total reference continuations$([ -f "$A4_OUT/DONE" ] && echo ', DONE')$([ -n "$a4_running" ] && echo ', the reference container a4-ref-k2 is RUNNING')"

# --- the source list (the same for every step) -------------------------------------------------
GOLD="$OUT/golden"
mkdir -p "$GOLD"
SRC=()
for p in prose code cjk; do SRC+=(--source "golden/$p:$GOLD/$p.ids:$GOLD/$p.cont.ids"); done
[ "$A4" = 1 ] && SRC+=(--a4-ref "$A4_OUT")

# =============================================================================================
if [ -n "$DRY_RUN" ]; then
  # No container: the host's python3 runs every script's --dry-run (they need no torch for it).
  # Sizes not yet known are PLANNED: the golden prompts' committed ids + BOS with GOLDEN_GEN
  # ids; the A4 scenarios without a reference at 512 ids (a4_ref.sh k2's cap: an upper bound),
  # prompt sizes from A4_OUT/set/manifest.json or Qwen3.8's set (a proxy for K2's tokeniser).
  DRY="$OUT/dry-run"
  mkdir -p "$DRY"
  PLAN=()
  for p in prose code cjk; do
    if [ ! -s "$GOLD/$p.cont.ids" ]; then
      n=$(( $(wc -w < tests/golden/prompts/$p.ids) + 1 ))
      g=$GOLDEN_GEN
      [ -s "$GOLDEN_DIR/$p.golden.safetensors" ] && [ -s "$GOLDEN_DIR/$p.ids" ] && { n=$(wc -w < "$GOLDEN_DIR/$p.ids"); g=32; }
      PLAN+=(--plan "golden/$p:$n:$g")
    fi
  done
  MAN="$A4_OUT/set/manifest.json"; [ -s "$MAN" ] || MAN=tests/golden/toolcall/manifest.json
  [ "$A4" = 1 ] && PLAN+=(--plan-manifest "$MAN:512")
  DRY_SRC=()
  for p in prose code cjk; do [ -s "$GOLD/$p.cont.ids" ] && DRY_SRC+=(--source "golden/$p:$GOLD/$p.ids:$GOLD/$p.cont.ids"); done
  [ "$A4" = 1 ] && [ -s "$A4_OUT/set/manifest.json" ] && DRY_SRC+=(--a4-ref "$A4_OUT")
  echo "--- the plan (steps: $STEPS)"
  if has a4; then
    if [ "$A4" != 1 ]; then echo "a4: skipped (A4=0: golden only)"
    elif [ -f "$A4_OUT/DONE" ]; then echo "a4: done"
    elif [ -n "$a4_running" ]; then echo "a4: running in a4-ref-k2 - the driver would stop here (exit 4) until $A4_OUT/DONE"
    else echo "a4: would run tools/toolcall/a4_ref.sh k2 set (if no $A4_OUT/set) and tools/toolcall/a4_ref.sh k2 ref (detached, 28 GB, its own container), then stop (exit 4); ESTIMATED ~1-1.5 days (36 scenarios x up to 512 layer-streamed decode steps at ~7 s, tools/oracle/README.md's K2 int4 estimate)"; fi
  fi
  has check && echo "check: python3 tools/oracle/test_{eagle3_ref,k2_taps,eagle3_accept,eagle3_cost,dflash_accept}.py in $NAME (~10 min measured on 2 CPUs, less at $THREADS threads)"
  has facts && echo "facts: python3 tools/oracle/eagle3_ref.py facts <drafter> --k2 <K2> (reads the drafter 1.67 GB + K2's embedding 1.28 GB; ~1 min)"
  if has golden; then
    for p in prose code cjk; do
      if [ -s "$GOLD/$p.cont.ids" ]; then echo "golden/$p: done"
      elif [ -s "$GOLDEN_DIR/$p.golden.safetensors" ]; then echo "golden/$p: from $GOLDEN_DIR (18a's 32 greedy ids)"
      else echo "golden/$p: tokenize.py --bos + k2_ref.py run --gen $GOLDEN_GEN (ESTIMATED ~$(( GOLDEN_GEN * 7 / 60 + 2 )) min: ~7 s a layer-streamed int4 step)"; fi
    done
  fi
  has sources && echo "sources: k2_taps.py sources --out-dir $OUT/dumps ${SRC[*]}"
  if has dump || has sources; then
    echo "dump: k2_taps.py run <K2> --out-dir $OUT/dumps --matmul $MATMUL (planned sizes below)"
    python3 tools/oracle/k2_taps.py run "${K2_SNAP:-<K2 snapshot>}" --out-dir "$DRY" --matmul "$MATMUL" --dry-run \
      ${DRY_SRC[@]+"${DRY_SRC[@]}"} ${PLAN[@]+"${PLAN[@]}"} || exit 1
  fi
  if has accept; then
    echo "accept: eagle3_accept.py run --dumps $OUT/dumps --out-dir $OUT/accept --arms $ARMS --drafter <drafter> --window $WINDOW"
    python3 tools/oracle/eagle3_accept.py run --dumps "$DRY" --out-dir "$DRY/accept" --arms "$ARMS" --window "$WINDOW" --dry-run || exit 1
  fi
  has lookup && python3 tools/oracle/eagle3_accept.py lookup --dumps "$DRY" --out-dir "$DRY/accept" --dry-run
  if has cost; then
    echo "cost: eagle3_cost.py run --dumps $OUT/dumps --out $OUT/accept/cost.json (seconds; the byte model, derived:)"
    python3 tools/oracle/eagle3_cost.py model || exit 1
  fi
  has summary && echo "summary: eagle3_accept.py summary --out-dir $OUT/accept --cost $OUT/accept/cost.json --depth $DEPTH --md $OUT/summary.md"
  echo "container (not started in a dry run): $NAME from $IMAGE, --memory $MEM, $THREADS threads; refuses while another $IMAGE container runs ($(docker ps --filter "ancestor=$IMAGE" --format '{{.Names}}' 2> /dev/null | tr '\n' ' '))"
  echo "=== $(date '+%F %T') dry run done"
  exit 0
fi
# =============================================================================================

others() { docker ps --filter "ancestor=$IMAGE" --format '{{.Names}}' | grep -vx "$NAME" || true; }
drop_container() { docker rm -f "$NAME" > /dev/null 2>&1 || true; }       # ours only, by name
ensure_container() {
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2> /dev/null)" = "true" ]; then return 0; fi
  local o; o=$(others)
  if [ -n "$o" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "another $IMAGE container is running ($o): one oracle at a time on this Mac - wait for it, or FORCE=1"
    exit 3
  fi
  drop_container
  docker run -d --name "$NAME" --memory "$MEM" --memory-swap "$MEM" -v "$HF":/hf:ro -v "$PWD":/ws \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/tmp/torch_ext \
    -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -w /ws "$IMAGE" sleep infinity > /dev/null ||
    { echo "docker run failed"; exit 2; }
  echo "container $NAME started (--memory $MEM, $THREADS threads)"
}
[ "${KEEP:-0}" = 1 ] || trap drop_container EXIT
x() { echo "--- $(date '+%T') $*"; ensure_container; /usr/bin/time -p docker exec "$NAME" "$@"; }
need_k2() { [ $K2_OK = 1 ] || { echo "$1: the K2 checkpoint is not complete in the HF cache ($K2_REPO)"; exit 2; }; }
need_e3() { [ $E3_OK = 1 ] || { echo "$1: the drafter is not complete in the HF cache ($DRAFTER_REPO)"; exit 2; }; }

if has a4 && [ "$A4" = 1 ]; then
  if [ -f "$A4_OUT/DONE" ]; then echo "a4: done ($a4_done/$a4_total)"
  elif [ "${A4_PARTIAL:-0}" = 1 ]; then echo "a4: A4_PARTIAL=1 - going on with $a4_done/$a4_total scenarios"
  elif [ -n "$a4_running" ]; then
    tools/toolcall/a4_ref.sh k2 status
    echo "a4: the reference runs in a4-ref-k2; re-run this driver once $A4_OUT/DONE exists (or A4_PARTIAL=1 after stopping it)"
    exit 4
  else
    need_k2 a4
    drop_container                                     # a4_ref.sh holds the one oracle slot itself
    # a4_ref.sh reads MEM / IMAGE / HF_CACHE from the environment: give it its own (28 GB)
    a4env=(env OUT="$A4_OUT" MEM="${A4_MEM:-28g}" IMAGE="$IMAGE" HF_CACHE="$HF")
    if [ ! -s "$A4_OUT/set/manifest.json" ]; then
      "${a4env[@]}" tools/toolcall/a4_ref.sh k2 set || { echo "a4: a4_ref.sh k2 set FAILED"; exit 1; }
    fi
    "${a4env[@]}" tools/toolcall/a4_ref.sh k2 ref || { echo "a4: a4_ref.sh k2 ref did not start (see above)"; exit 1; }
    echo "a4: started; re-run this driver once $A4_OUT/DONE exists ('tools/toolcall/a4_ref.sh k2 status')"
    exit 4
  fi
fi

if has check; then
  for t in test_eagle3_ref test_k2_taps test_eagle3_accept test_eagle3_cost test_dflash_accept; do
    x python3 tools/oracle/$t.py > "$OUT/$t.log" 2>&1 || { tail -20 "$OUT/$t.log"; echo "$t FAILED"; exit 1; }
  done
  echo "unit tests: PASS (logs in $OUT)"
fi

if has facts; then
  need_e3 facts
  K2ARG=(); [ -n "$K2_SNAP" ] && K2ARG=(--k2 "$K2_SNAP")   # partial is fine: facts reads the embedding shard + tokenizer only
  x python3 tools/oracle/eagle3_ref.py facts "$E3_SNAP" ${K2ARG[@]+"${K2ARG[@]}"} > "$OUT/facts.log" 2>&1 || { cat "$OUT/facts.log"; echo "facts FAILED"; exit 1; }
  cat "$OUT/facts.log"
fi

if has golden; then
  for p in prose code cjk; do
    [ -s "$GOLD/$p.cont.ids" ] && { echo "golden/$p: done"; continue; }
    if [ -s "$GOLDEN_DIR/$p.golden.safetensors" ] && [ -s "$GOLDEN_DIR/$p.ids" ]; then
      cp "$GOLDEN_DIR/$p.ids" "$GOLD/$p.ids"
      x python3 tools/oracle/k2_taps.py golden-ids "$GOLDEN_DIR/$p.golden.safetensors" "$GOLD/$p.cont.ids.tmp" || exit 1
    else
      need_k2 golden
      x sh -c "python3 tools/oracle/tokenize.py '$K2_SNAP' encode --bos tests/golden/prompts/$p.txt > $GOLD/$p.ids" || exit 1
      x python3 tools/oracle/k2_ref.py run "$K2_SNAP" --prompt "$GOLD/$p.ids" --out "$GOLD/$p.run.safetensors" \
        --gen "$GOLDEN_GEN" --max-prompt 4096 > "$GOLD/$p.log" 2>&1 || { tail -20 "$GOLD/$p.log"; echo "golden/$p FAILED"; exit 1; }
      x python3 tools/oracle/k2_taps.py golden-ids "$GOLD/$p.run.safetensors" "$GOLD/$p.cont.ids.tmp" || exit 1
    fi
    mv "$GOLD/$p.cont.ids.tmp" "$GOLD/$p.cont.ids"
    echo "golden/$p: $(wc -w < "$GOLD/$p.ids") prompt + $(wc -w < "$GOLD/$p.cont.ids") greedy ids"
  done
fi

if has sources; then
  x python3 tools/oracle/k2_taps.py sources --out-dir "$OUT/dumps" "${SRC[@]}" || exit 1
fi

if has dump; then
  need_k2 dump
  x python3 tools/oracle/k2_taps.py run "$K2_SNAP" --out-dir "$OUT/dumps" --matmul "$MATMUL" "${SRC[@]}" \
    || { echo "dump FAILED"; exit 1; }
fi

if has accept; then
  need_e3 accept
  x python3 tools/oracle/eagle3_accept.py run --dumps "$OUT/dumps" --out-dir "$OUT/accept" --arms "$ARMS" \
    --drafter "$E3_SNAP" --window "$WINDOW" || { echo "accept FAILED"; exit 1; }
  if [ "${K2_EMBED:-0}" = 1 ]; then              # the drafter with K2's embedding (if facts says they differ)
    need_k2 accept
    x python3 tools/oracle/eagle3_accept.py run --dumps "$OUT/dumps" --out-dir "$OUT/accept" --arms "$ARMS" \
      --drafter "$E3_SNAP" --window "$WINDOW" --embed-from "$K2_SNAP" || { echo "accept (K2 embedding) FAILED"; exit 1; }
  fi
fi

if has lookup; then
  x python3 tools/oracle/eagle3_accept.py lookup --dumps "$OUT/dumps" --out-dir "$OUT/accept" || { echo "lookup FAILED"; exit 1; }
fi

if has cost; then
  CFG=tools/oracle/third_party/eagle3_k2/config.json
  [ $E3_OK = 1 ] && CFG="$E3_SNAP/config.json"
  x python3 tools/oracle/eagle3_cost.py run --dumps "$OUT/dumps" --out "$OUT/accept/cost.json" --drafter-config "$CFG" \
    > "$OUT/cost.log" 2>&1 || { cat "$OUT/cost.log"; echo "cost FAILED"; exit 1; }
  cat "$OUT/cost.log"
fi

if has summary; then
  x python3 tools/oracle/eagle3_accept.py summary --out-dir "$OUT/accept" --cost "$OUT/accept/cost.json" \
    --depth "$DEPTH" --md "$OUT/summary.md" > /dev/null || { echo "summary FAILED"; exit 1; }
  echo "summary: $OUT/summary.md"
fi
echo "=== $(date '+%F %T') done"
