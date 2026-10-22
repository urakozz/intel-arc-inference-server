#!/bin/bash
# Spec 19a Tasks 2-3 on the Mac: tap dumps from the Qwen3.8 bf16 CPU reference and teacher-forced
# DFlash acceptance. The bf16 base checkpoint (Qwen/Qwen3.8-27B) layer-streamed in a Linux
# container capped at 28 GB (tools/oracle/dump_taps.py), then the drafter arms over the dumps
# (tools/oracle/dflash_accept.py). Run from the repo (or worktree) root on the Mac:
#
#   tools/oracle/dflash_p0.sh                        # everything, resumable
#   DRY_RUN=1 tools/oracle/dflash_p0.sh              # the plans and estimates; loads no model
#   STEPS="accept" ARMS=bf16 tools/oracle/dflash_p0.sh
#
# Detach it (it outlives the terminal): nohup tools/oracle/dflash_p0.sh > /dev/null 2>&1 &
# then `tail -f oracle-out-19a/p0.log`. Every output is in OUT (git-ignored).
#
# It refuses to start a model step while the spec 12a repeat (kv8_qwen38_repeat.sh, also 28 GB)
# runs: Docker has 62.8 GB on this Mac (IGNORE_REPEAT=1 overrides).
#
# Steps (each resumable: dumps and accept results already written are skipped):
#   check    the unit tests of both scripts (tiny models, seconds)
#   sources  OUT/dumps/sources.json + rank.py request logs: the golden prompts with their
#            recorded greedy continuations (GOLDEN_CONT_DIR/{prose,code,cjk}.ids, else the 32
#            ids of the 2026-08-24 oracle below), the A4 set (34 scenarios, reference text
#            re-encoded), EXTRA_SOURCES ("corpus/label:PROMPT_IDS:CONT_IDS ...", e.g. a
#            prose / reasoning set the engine recorded)
#   dump     dump_taps.py run (one streamed forward per batch of sources)
#   rank     tools/draft_vocab/rank.py over the same sources -> OUT/draft_vocab.ranked.ids (the
#            V' "-r" arms; in-sample, so an upper bound for a ranked V')
#   accept   dflash_accept.py run, ARMS (default bf16,int8,int8h,w4a16), V' 32k/64k/128k on bf16
#
# MODEL=muse (spec 23f Task 3): Muse-Glimmer's assistant over muse_taps.py's dumps (DUMPS, default
# oracle-out-muse-taps/dumps - tools/box_validate/muse_oracle.sh <data> taps makes them), OUT default
# oracle-out-muse-p0; steps (STEPS, default "check accept lookup"):
#   check    test_dflash_accept.py, test_muse_taps.py and test_dflash_ref.py (the Muse form against transformers
#            5.19.0: SITE, default oracle-out-q4exp/site, first on PYTHONPATH)
#   accept   dflash_accept.py run --model muse (ARMS default bf16,int8,int8h,int4g64,int4g64h,dflash2; K 1..15 from one
#            block per anchor; the sampled estimate for q raw and capped) -> OUT/accept/{accept.*.json,summary.md}
#   lookup   tools/spec/lookup_accept.py --dumps DUMPS --max-k 15 (n = 2..6) -> OUT/lookup.{txt,json}
#
# MODEL=lilicorr (spec 26a Tasks 2-3): the LiLiCorr drafter over Bolt's tap dumps (DUMPS, default
# oracle-out-bolt-taps/dumps - tools/box_validate/bolt_oracle.sh <data> taps makes them), OUT default
# oracle-out-lilicorr-p0; steps (STEPS, default "check accept lookup"):
#   check    test_dflash_ref.py (the LiLiCorr form) and test_dflash_accept.py (incl. the tiny pair: 25a's tiny Bolt +
#            lilicorr_make_tiny; SITE first on PYTHONPATH)
#   accept   dflash_accept.py run --model bolt-lilicorr (ARMS default exact,int8,int4g64,exacth,int8h,int4g64h; K 1..7
#            from one 8-row block per anchor; the sampled estimate for q lattice and lattice+lp) -> OUT/accept/; SKIP
#            (77) while the drafter's weights are not in the HF cache
#   lookup   tools/spec/lookup_accept.py --dumps DUMPS --max-k 7 -> OUT/lookup.{txt,json}
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
if [ "${MODEL:-qwen38}" = lilicorr ]; then
  OUT="${OUT:-oracle-out-lilicorr-p0}"
  DUMPS="${DUMPS:-oracle-out-bolt-taps/dumps}"
  SITE="${SITE:-oracle-out-q4exp/site}"
  IMAGE="${IMAGE:-agnes-ref-img:latest}"
  THREADS="${THREADS:-12}"
  STEPS="${STEPS:-check accept lookup}"
  ARMS="${ARMS:-exact,int8,int4g64,exacth,int8h,int4g64h}"
  MEM="${MEM:-36g}"
  HF="${HF_HOME:-$HOME/.cache/huggingface}"
  DRY_RUN="${DRY_RUN:-}"
  mkdir -p "$OUT"
  exec > >(tee -a "$OUT/p0.log") 2>&1
  echo "=== $(date '+%F %T') LiLiCorr P0: steps [$STEPS], arms $ARMS, dumps $DUMPS, out $OUT${DRY_RUN:+ (dry run)}"
  has() { [[ " $STEPS " == *" $1 "* ]]; }
  if { has accept || has lookup; } && [ ! -d "$DUMPS" ]; then
    echo "no $DUMPS (tools/box_validate/bolt_oracle.sh <data> taps first)"
    exit 2
  fi
  oabs=$(cd "$OUT" && pwd -P) sabs="" dmount=()
  [ -d "$SITE" ] && sabs=$(cd "$SITE" && pwd -P)
  [ -d "$DUMPS" ] && dmount=(-v "$(cd "$DUMPS" && pwd -P)":/dumps:ro)
  # One model computation on the Mac's CPU at a time: every container waits for the shared lock (fcntl, B70_CPU_LOCK).
  lock=(python3 -c 'import fcntl, subprocess, sys
f = open(sys.argv[1], "a+")
fcntl.flock(f, fcntl.LOCK_EX)
sys.exit(subprocess.run(sys.argv[2:]).returncode)' "${B70_CPU_LOCK:-/tmp/b70-cpu-model.lock}")
  run() {
    echo "--- $(date '+%T') $*"
    /usr/bin/time -p "${lock[@]}" docker run --rm --memory "$MEM" --memory-swap "$MEM" -v "$HF":/hf:ro -v "$PWD":/ws \
      "${dmount[@]}" -v "$oabs":/out ${sabs:+-v "$sabs":/site:ro -e PYTHONPATH=/site} -e HF_HOME=/hf \
      -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TQDM_DISABLE=1 -e OMP_NUM_THREADS="$THREADS" \
      -e MKL_NUM_THREADS="$THREADS" -e LILICORR_TINY_DIR=/out/tiny -w /ws "$IMAGE" "$@"
  }
  rc=0
  if has check; then
    for t in test_dflash_ref test_dflash_accept; do
      run python3 "tools/oracle/$t.py" > "$OUT/$t.log" 2>&1 || { tail -20 "$OUT/$t.log"; echo "$t FAILED"; exit 1; }
    done
    echo "unit tests: PASS (logs in $OUT)"
  fi
  if has accept; then
    run python3 tools/oracle/dflash_accept.py run --model bolt-lilicorr --dumps /dumps --out-dir /out/accept \
      --arms "$ARMS" --batch "${BATCH:-8}" ${DRY_RUN:+--dry-run}
    r=$?
    if [ "$r" = 77 ]; then echo "accept SKIPPED (the drafter's weights: PENDING-WEIGHTS)"; rc=77
    elif [ "$r" != 0 ]; then echo "accept FAILED"; exit 1; fi
  fi
  if has lookup; then
    python3 tools/spec/lookup_accept.py --dumps "$DUMPS" --max-k 7 --mc-seeds 4 --json "$OUT/lookup.json" \
      > "$OUT/lookup.txt" || { echo "lookup FAILED"; exit 1; }
    sed -n '/(all)/,/^Verify share/p' "$OUT/lookup.txt" | head -14
  fi
  echo "=== $(date '+%F %T') done"
  exit "$rc"
fi
if [ "${MODEL:-qwen38}" = muse ]; then
  OUT="${OUT:-oracle-out-muse-p0}"
  DUMPS="${DUMPS:-oracle-out-muse-taps/dumps}"
  SITE="${SITE:-oracle-out-q4exp/site}"
  IMAGE="${IMAGE:-agnes-ref-img:latest}"
  THREADS="${THREADS:-12}"
  STEPS="${STEPS:-check accept lookup}"
  ARMS="${ARMS:-bf16,int8,int8h,int4g64,int4g64h,dflash2}"
  MEM="${MEM:-36g}"
  HF="${HF_HOME:-$HOME/.cache/huggingface}"
  DRY_RUN="${DRY_RUN:-}"
  mkdir -p "$OUT"
  exec > >(tee -a "$OUT/p0.log") 2>&1
  echo "=== $(date '+%F %T') Muse DFlash P0: steps [$STEPS], arms $ARMS, dumps $DUMPS, out $OUT${DRY_RUN:+ (dry run)}"
  [ -d "$DUMPS" ] || { echo "no $DUMPS (tools/box_validate/muse_oracle.sh <data> taps first)"; exit 2; }
  dabs=$(cd "$DUMPS" && pwd -P) oabs=$(cd "$OUT" && pwd -P) sabs=""
  [ -d "$SITE" ] && sabs=$(cd "$SITE" && pwd -P)
  # One model computation on the Mac's CPU at a time: every container waits for the shared lock (fcntl, B70_CPU_LOCK).
  lock=(python3 -c 'import fcntl, subprocess, sys
f = open(sys.argv[1], "a+")
fcntl.flock(f, fcntl.LOCK_EX)
sys.exit(subprocess.run(sys.argv[2:]).returncode)' "${B70_CPU_LOCK:-/tmp/b70-cpu-model.lock}")
  run() {
    echo "--- $(date '+%T') $*"
    /usr/bin/time -p "${lock[@]}" docker run --rm --memory "$MEM" --memory-swap "$MEM" -v "$HF":/hf:ro -v "$PWD":/ws \
      -v "$dabs":/dumps:ro -v "$oabs":/out ${sabs:+-v "$sabs":/site:ro -e PYTHONPATH=/site} -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
      -e PYTHONUNBUFFERED=1 -e TQDM_DISABLE=1 -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -w /ws \
      "$IMAGE" "$@"
  }
  has() { [[ " $STEPS " == *" $1 "* ]]; }
  if has check; then
    for t in test_dflash_accept test_muse_taps test_dflash_ref; do
      run python3 "tools/oracle/$t.py" > "$OUT/$t.log" 2>&1 || { tail -20 "$OUT/$t.log"; echo "$t FAILED"; exit 1; }
    done
    echo "unit tests: PASS (logs in $OUT)"
  fi
  if has accept; then
    run python3 tools/oracle/dflash_accept.py run --model muse --dumps /dumps --out-dir /out/accept \
      --arms "$ARMS" --batch "${BATCH:-8}" ${DRY_RUN:+--dry-run} || { echo "accept FAILED"; exit 1; }
  fi
  if has lookup; then
    python3 tools/spec/lookup_accept.py --dumps "$DUMPS" --max-k 15 --mc-seeds 4 --json "$OUT/lookup.json" \
      > "$OUT/lookup.txt" || { echo "lookup FAILED"; exit 1; }
    sed -n '/(all)/,/^Verify share/p' "$OUT/lookup.txt" | head -14
  fi
  echo "=== $(date '+%F %T') done"
  exit 0
fi
OUT="${OUT:-oracle-out-19a}"
REPO_ID="${REPO_ID:-Qwen/Qwen3.8-27B}"
IMAGE="${IMAGE:-agnes-ref-img:latest}"      # python 3.12, torch 2.14.1+cpu, transformers 5.15.0
THREADS="${THREADS:-16}"
STEPS="${STEPS:-check sources dump rank accept}"
ARMS="${ARMS:-bf16,int8,int8h,w4a16}"
VOCAB="${VOCAB:-32k,64k,128k}"
BATCH_TOKENS="${BATCH_TOKENS:-8192}"
DRY_RUN="${DRY_RUN:-}"
if [ -n "$DRY_RUN" ]; then NAME="${NAME:-dflash-dry}"; MEM="${MEM:-6g}"; else NAME="${NAME:-dflash-ref}"; MEM="${MEM:-28g}"; fi
HF="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$OUT"
LOG="$OUT/p0.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') DFlash P0: steps [$STEPS], arms $ARMS, out $OUT${DRY_RUN:+ (dry run)}"

if [ -z "$DRY_RUN" ] && [ -z "${IGNORE_REPEAT:-}" ] && pgrep -f kv8_qwen38_repeat > /dev/null; then
  echo "the spec 12a repeat (kv8_qwen38_repeat.sh) is running: not starting a 28 GB job beside it"
  exit 3
fi

CACHE="$HF/hub/models--${REPO_ID//\//--}"
SNAP_HOST=$(ls -d "$CACHE"/snapshots/*/ 2>/dev/null | head -1)
[ -n "$SNAP_HOST" ] || { echo "no snapshot of $REPO_ID under $CACHE"; exit 2; }
SNAP="/hf/hub/models--${REPO_ID//\//--}/snapshots/$(basename "$SNAP_HOST")"

if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
  docker rm -f "$NAME" > /dev/null 2>&1
  docker run -d --name "$NAME" --memory "$MEM" --memory-swap "$MEM" -v "$HF":/hf:ro -v "$PWD":/ws \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -w /ws "$IMAGE" sleep infinity > /dev/null ||
    { echo "docker run failed"; exit 2; }
fi
x() { echo "--- $(date '+%T') $*"; /usr/bin/time -p docker exec "$NAME" "$@"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }

# The sources (the same list for every step).
GOLD="$OUT/golden"
mkdir -p "$GOLD"
if [ -n "${GOLDEN_CONT_DIR:-}" ]; then
  for p in prose code cjk; do cp "$GOLDEN_CONT_DIR/$p.ids" "$GOLD/$p.cont.ids"; done
else
  # The 32 greedy ids of the 2026-08-24 oracle run (Qwen3.8 W4A16, bf16 head; tools/oracle/
  # README.md "Sanity checks"), as kv8_qwen38_repeat.sh teacher-forces them.
  echo "3113 7810 279 1118 479 654 8980 1000 381 1142 440 279 1834 725 2213 13 3113 11292 279 4220 6092 1000 381 6992 11 321 539 5600 279 72103 1000 381" > "$GOLD/prose.cont.ids"
  echo "271 727 40523 17 19490 11 750 11 15131 1590 198 262 460 498 1030 8474 3620 11 750 681 15131 8 364 343 303 2663 60 271 727 40523 18 19490" > "$GOLD/code.cont.ids"
  echo "29545 271 95815 108553 97663 108447 96494 3709 98844 95895 97771 95726 114183 101650 100700 1710 271 550 220 99737 96863 271 99737 96863 95761 105064 97463 95793 100830 98252 96019 115534" > "$GOLD/cjk.cont.ids"
fi
SRC=()
for p in prose code cjk; do SRC+=(--source "$p/golden:tests/golden/prompts/$p.ids:$GOLD/$p.cont.ids"); done
for s in ${EXTRA_SOURCES:-}; do SRC+=(--source "$s"); done
SRC+=(--a4 tests/golden/toolcall --a4-expected tests/server/toolcall_expected.json --tokenizer "$SNAP/tokenizer.json")

if has check; then
  x python3 tools/oracle/test_dump_taps.py > "$OUT/test_dump_taps.log" 2>&1 || { tail -20 "$OUT/test_dump_taps.log"; echo "test_dump_taps FAILED"; exit 1; }
  x python3 tools/oracle/test_dflash_accept.py > "$OUT/test_dflash_accept.log" 2>&1 || { tail -20 "$OUT/test_dflash_accept.log"; echo "test_dflash_accept FAILED"; exit 1; }
  echo "unit tests: PASS (logs in $OUT)"
fi

if has sources || [ -n "$DRY_RUN" ]; then
  x python3 tools/oracle/dump_taps.py sources --out-dir "$OUT/dumps" --rank-logs "$OUT/ranklogs" "${SRC[@]}" || exit 1
fi

if has dump; then
  x python3 tools/oracle/dump_taps.py run "$SNAP" --out-dir "$OUT/dumps" --batch-tokens "$BATCH_TOKENS" \
    ${DRY_RUN:+--dry-run} "${SRC[@]}" || { echo "dump FAILED"; exit 1; }
fi

if has rank; then
  if [ -n "$DRY_RUN" ]; then echo "rank: rank.py over $OUT/ranklogs -> $OUT/draft_vocab.ranked.ids (skipped: dry run)"
  elif [ -f "$OUT/draft_vocab.ranked.ids" ]; then echo "rank: done"
  else x python3 tools/draft_vocab/rank.py "$OUT/ranklogs" --vocab-used 248077 -o "$OUT/draft_vocab.ranked.ids" || exit 1
  fi
fi

if has accept; then
  RANKED=()
  [ -f "$OUT/draft_vocab.ranked.ids" ] || [ -n "$DRY_RUN" ] && RANKED=(--draft-vocab-ids "$OUT/draft_vocab.ranked.ids")
  x python3 tools/oracle/dflash_accept.py run --dumps "$OUT/dumps" --out-dir "$OUT/accept" --arms "$ARMS" \
    --vocab "$VOCAB" "${RANKED[@]}" ${DRY_RUN:+--dry-run} || { echo "accept FAILED"; exit 1; }
fi
if [ -n "$DRY_RUN" ]; then docker rm -f "$NAME" > /dev/null 2>&1; fi
echo "=== $(date '+%F %T') done"
