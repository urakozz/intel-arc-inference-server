#!/usr/bin/env bash
# One command: z-lab/Qwen3.8-27B-DFlash2 (bf16) -> our AutoRound W4A16 g64 symmetric drafter, tuned on the
# QUANTISED target's taps over the training opencode recordings (docs/open-work.md, "DFlash drafter quant").
#
#   tools/quantize/quantize_dflash.sh start              start (or resume) the run, detached; on the box
#   tools/quantize/quantize_dflash.sh status             stages done, taps dumped, AutoRound's block, the result
#   tools/quantize/quantize_dflash.sh stop               stop the run (tools/box_stop.sh dflash-quant); start resumes
#   tools/quantize/quantize_dflash.sh selftest           quant/export/check on a tiny random drafter (~2 min)
#   tools/quantize/quantize_dflash.sh run [--only S | --from S]   the stages in the foreground (what start runs)
#
# Stages of `run` (a finished stage leaves $WORK/.done-<stage>; a rerun skips it):
#   sources  tools/quantize/dflash/sources.py: the requests of RECORDINGS (default opencode-session-int8 and
#            opencode-session; opencode-session-2 is held out and refused, A4 / golden never enter) with >= 8
#            continuation ids, cut at 256 ids and the first EOS, minus those already dumped
#   taps     tools/probe/probe_dflash_iter --engine-taps on card CARD (tools/gpu_lock.sh card): the quantised
#            target TARGET teacher-forced over each request, its taps at the drafter's layers 5/19/33/47/61
#            over the last 2048 prompt positions and the continuation, one dump per request in $WORK/taps
#            (~8 GB); resumable per request (a rerun dumps only what is missing). The tree is built if the
#            probe is missing.
#   quant    AutoRound (tools/quantize/dflash/autoround_run.py) on the real non-causal DFlash2DraftModel
#            forward (tools/quantize/dflash/model.py = tools/oracle/dflash_ref.py's, batched): W4A16, group 64,
#            symmetric, auto_round:auto_gptq, ITERS (200) sign-SGD iterations, NSAMPLES (128) samples of
#            ANCHORS (32) blocks of 8 at the recordings' continuation positions, BATCH (8), seed 42, no
#            quant_lm_head; on card CARD, or the box CPU (DEVICE=cpu, /tmp/b70-cpu-model.lock). Restarts from
#            the beginning after a stop (the raw export moved to ~/b70/trash/<date>).
#   export   tools/quantize/dflash/finalize.py: checkpoint names, unpacked tensors bitwise the source's, the
#            source's config.json + quantization_config, dflash_recipe.json, README.md with the provenance
#   check    tools/quantize/check_gptq_export.py --dflash --source: packed = every drafter linear (fc and the
#            selector projection included), norms / conv bases / codebooks the source's bytes, commit and
#            recipe recorded; the last line is "DFLASH QUANT: OK <export>" or "DFLASH QUANT: FAIL ..."
#
# Output: OUT = QUANT_ROOT/Qwen3.8-27B-DFlash2-W4A16-g64-<date>/ (the checkpoint only: config.json,
# model.safetensors, dflash_recipe.json, README.md); the work beside it in $OUT.work/ (sources.json, taps/, raw/,
# .done-*, RESULT, run.pid, run.log -> LOGDIR/run-<time>.log); QUANT_ROOT/dflash-current points at OUT.
#
# AutoRound: the LATEST of the operator's fork ~/auto-round (AR_REPO), not a fixed pin (operator 2026-11-13): the
# hash branch AR_REF (local-patches-v2 = upstream main + the cached-OpenCodeInstruct commit) has when `start` or
# `run` begins, resolved by tools/quantize/ar_latest.sh and recorded (hash + upstream base) in the recipe and the
# README. No patch file (AR_PATCH=0: auto-round-local.patch's BOS/EOS fix is upstream, #2591, and the calibration
# rows here are our own tensors). A run keeps the hash it started with ($OUT.work/ar_commit): `start` on a run
# whose branch has moved refuses (AR_COMMIT=<recorded hash> resumes it on the old build). The old pin is the
# branch pin-a6cf7487-2026-11-13. Installed by tools/quantize/ar_pin.sh beside VENVPY, transformers 5.19.0 in SITE.
#
# Overridables (environment): DRAFTER (z-lab/Qwen3.8-27B-DFlash2 or a snapshot dir), TARGET
# (urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ or a dir), RECORDINGS (space-separated dirs), QUANT_ROOT
# ($HOME/b70/hdd/quant), OUT (default: the run `dflash-current` points at, else a new dated one), TAPS ($WORK/taps;
# another run's taps dir to reuse it), DEVICE (xpu: card CARD | cpu), CARD (1), ITERS, NSAMPLES, ANCHORS, BATCH,
# CONTEXT (2304 context rows), RAM_CAP_GB (64), RAM_MIN_GB (40), DISK_MIN_GB (20), PROBE, VENVPY, SITE, AR_REPO, AR_REF,
# LOGDIR ($HOME/b70/logs/dflash-quant).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"         # tools/
REPO="$(cd "$HERE/.." && pwd)"
Q="$HERE/quantize/dflash"
SELF="$HERE/quantize/quantize_dflash.sh"
DRAFTER="${DRAFTER:-z-lab/Qwen3.8-27B-DFlash2}"
TARGET="${TARGET:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}"
RECORDINGS="${RECORDINGS:-$HOME/b70/logs/opencode-session-int8 $HOME/b70/logs/opencode-session}"
QUANT_ROOT="${QUANT_ROOT:-$HOME/b70/hdd/quant}"
LOGDIR="${LOGDIR:-$HOME/b70/logs/dflash-quant}"
DEVICE="${DEVICE:-xpu}"
CARD="${CARD:-1}"
ITERS="${ITERS:-200}"
NSAMPLES="${NSAMPLES:-128}"
ANCHORS="${ANCHORS:-32}"
BATCH="${BATCH:-8}"
CONTEXT="${CONTEXT:-2304}"
SEED=42
RAM_CAP_GB="${RAM_CAP_GB:-64}"
RAM_MIN_GB="${RAM_MIN_GB:-40}"
DISK_MIN_GB="${DISK_MIN_GB:-20}"
PROBE="${PROBE:-$REPO/build/tools/probe/probe_dflash_iter}"
VENVPY="${VENVPY:-$HOME/auto-round/.venv/bin/python}"
SITE="${SITE:-$HOME/.cache/q4exp-site-py312}"
. "$HERE/quantize/ar_latest.sh"     # AR_COMMIT = the latest AutoRound (branch AR_REF of AR_REPO), see there
CURRENT="$QUANT_ROOT/dflash-current"
STAGES=(sources taps quant export check)

die() { echo "quantize_dflash: $*" >&2; exit 2; }

snapshot() {   # $1 = repo id or dir -> its snapshot directory (refs/main of the HF cache entry, or the dir)
  if [ -f "$1/config.json" ]; then echo "$1"; return; fi
  local e="$HOME/.cache/huggingface/hub/models--${1//\//--}"
  [ -f "$e/refs/main" ] || die "$1 is not in the HF cache ($e)"
  echo "$e/snapshots/$(cat "$e/refs/main")"
}

env_py() {     # the latest AutoRound beside VENVPY, transformers 5.19.0 in SITE; sets PKG
  ar_resolve || exit 2
  [ -x "$VENVPY" ] || die "VENVPY=$VENVPY is not an executable python"
  if [ ! -d "$SITE/transformers-5.19.0.dist-info" ]; then
    local uv; uv=$(command -v uv || echo "$HOME/.local/bin/uv")
    "$uv" pip install -q --python "$VENVPY" --no-deps --target "$SITE" -r "$HERE/oracle/qwen4exp_requirements.txt"
  fi
  local here_save="$PWD"
  . "$HERE/quantize/ar_pin.sh"      # cd /tmp (ar_pin's rule: no auto-round checkout as cwd)
  cd "$here_save"
  export PYTHONPATH="$PKG:$SITE" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=4
}

mem_avail_gb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
disk_free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }

# locked <cmd...>: the device's lock, then a memory-capped scope
locked() {
  local cap=(systemd-run --user --scope -q -p "MemoryMax=${RAM_CAP_GB}G" -p MemorySwapMax=0)
  if [ "$DEVICE" = xpu ]; then
    "$HERE/gpu_lock.sh" card "$CARD" -- "${cap[@]}" "$@"
  else
    flock /tmp/b70-cpu-model.lock "${cap[@]}" "$@"
  fi
}

to_trash() {   # move a leftover aside (nothing is deleted on the box)
  local t; t="$HOME/b70/trash/$(date +%F)"; mkdir -p "$t"
  mv "$1" "$t/$(basename "$1")-$(date +%H%M%S)"
}

stage_sources() {
  # shellcheck disable=SC2086
  python3 "$Q/sources.py" $RECORDINGS --generation-config "$TSNAP/generation_config.json" \
    --out "$WORK/sources.json" --done "$TAPS"
}

stage_taps() {
  mkdir -p "$TAPS"
  if [ "$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$WORK/sources.json")" != 0 ]; then
    if [ ! -x "$PROBE" ]; then
      echo "building $REPO/build (the probe loads the engine's kernel binaries, so the whole tree)"
      [ -f "$REPO/build/CMakeCache.txt" ] || cmake -S "$REPO" -B "$REPO/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
      cmake --build "$REPO/build" -j44 > "$WORK/probe-build.log" 2>&1 ||
        die "building $REPO/build failed: $WORK/probe-build.log"
    fi
    (cd "$REPO" && "$HERE/gpu_lock.sh" card "$CARD" -- "$PROBE" "$TSNAP" "$DSNAP" --format int8 \
      --engine-taps "$WORK/sources.json" "$TAPS" --anchors 1)
  fi
  stage_sources
  python3 -c 'import json, sys; sys.exit(len(json.load(open(sys.argv[1]))) != 0)' "$WORK/sources.json" ||
    die "taps: requests still without a complete dump (the lines above)"
}

stage_quant() {
  [ -e "$WORK/raw" ] && to_trash "$WORK/raw"
  env_py
  AR_WORK_SPACE="$WORK/ar-work" locked "$VENVPY" "$Q/autoround_run.py" --drafter "$DSNAP" --target "$TSNAP" \
    --taps "$TAPS" --out "$WORK/raw" --device "$DEVICE" --iters "$ITERS" --nsamples "$NSAMPLES" \
    --anchors "$ANCHORS" --batch-size "$BATCH" --seed "$SEED" --context "$CONTEXT"
}

stage_export() {
  [ -e "$OUT/model.safetensors" ] && to_trash "$OUT/model.safetensors"
  env_py
  "$VENVPY" "$Q/finalize.py" "$WORK/raw" "$DSNAP" "$OUT"
}

stage_check() {
  env_py
  if "$VENVPY" "$HERE/quantize/check_gptq_export.py" "$OUT" --dflash --source "$DSNAP"; then
    cat >> "$OUT/README.md" <<EOF
- checked by \`tools/quantize/check_gptq_export.py --dflash --source\` (VERIFIED $(date -u +%Y-%m-%dT%H:%MZ));
  source \`$DRAFTER\` at \`$(basename "$DSNAP")\`, taps of \`$TARGET\` at \`$(basename "$TSNAP")\`
EOF
    echo "DFLASH QUANT: OK $OUT" | tee "$WORK/RESULT"
  else
    echo "DFLASH QUANT: FAIL - the check refused $OUT (the lines above say what)" | tee "$WORK/RESULT"
    return 1
  fi
}

run_stages() {
  local only="" from="" hit=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --only) only=$2; shift 2 ;; --from) from=$2; shift 2 ;;
      *) die "run: unknown argument $1" ;;
    esac
  done
  [ -n "$OUT" ] || die "OUT is not set and $CURRENT does not exist - use start"
  WORK="$OUT.work"; TAPS="${TAPS:-$WORK/taps}"
  mkdir -p "$WORK"
  ar_resolve || exit 2; ar_run_guard "$WORK" || exit 2
  DSNAP=$(snapshot "$DRAFTER"); TSNAP=$(snapshot "$TARGET")
  echo "quantize_dflash: $(date -u +%FT%TZ) DRAFTER=$DRAFTER ($DSNAP) TARGET=$TARGET ($TSNAP) OUT=$OUT" \
       "TAPS=$TAPS DEVICE=$DEVICE CARD=$CARD ITERS=$ITERS NSAMPLES=$NSAMPLES ANCHORS=$ANCHORS BATCH=$BATCH" \
       "auto-round $AR_COMMIT (base ${AR_BASE:0:8}, $AR_REF)"
  for s in "${STAGES[@]}"; do
    [ "$s" = "$from" ] && hit=1
    if [ -n "$only" ] && [ "$s" != "$only" ]; then continue; fi
    if [ -z "$only" ] && [ -n "$from" ] && [ $hit = 0 ]; then continue; fi
    if [ -z "$only$from" ] && [ -f "$WORK/.done-$s" ]; then echo "== $s: done $(cat "$WORK/.done-$s")"; continue; fi
    echo "== $s $(date -u +%FT%TZ)"
    "stage_$s"
    date -u +%FT%TZ > "$WORK/.done-$s"
  done
}

resolve_out() { [ -n "${OUT:-}" ] || OUT=$( [ -L "$CURRENT" ] && readlink -f "$CURRENT" || true); }

case "${1:-}" in
  start)
    mkdir -p "$QUANT_ROOT" "$LOGDIR"
    resolve_out
    if [ -n "$OUT" ] && [ -f "$OUT.work/run.pid" ] && kill -0 "$(cat "$OUT.work/run.pid")" 2>/dev/null; then
      die "already running (pid $(cat "$OUT.work/run.pid"), $OUT) - see: $SELF status"
    fi
    if [ -n "$OUT" ] && [ -f "$OUT.work/.done-check" ]; then
      die "the run in $OUT is finished ($(cat "$OUT.work/RESULT")); for a new one: OUT=<new dir> $SELF start"
    fi
    snapshot "$DRAFTER" > /dev/null; snapshot "$TARGET" > /dev/null
    ar_resolve || exit 2
    [ -z "$OUT" ] || ar_run_guard "$OUT.work" || exit 2
    for r in $RECORDINGS; do [ -d "$r" ] || die "recording $r does not exist"; done
    m=$(mem_avail_gb); [ "$m" -ge "$RAM_MIN_GB" ] || die "MemAvailable ${m} GB < RAM_MIN_GB $RAM_MIN_GB - something else is using the RAM"
    d=$(disk_free_gb "$QUANT_ROOT"); [ "$d" -ge "$DISK_MIN_GB" ] || die "$QUANT_ROOT has ${d} GB free < DISK_MIN_GB $DISK_MIN_GB"
    if [ -z "$OUT" ]; then
      OUT="$QUANT_ROOT/Qwen3.8-27B-DFlash2-W4A16-g64-$(date +%F)"
      mkdir -p "$OUT.work"; ln -sfn "$OUT" "$CURRENT"
      echo "new run: $OUT"
    else
      mkdir -p "$OUT.work"
      echo "resuming: $OUT"
    fi
    log="$LOGDIR/run-$(date +%Y%m%d-%H%M%S).log"
    ln -sfn "$log" "$OUT.work/run.log"
    OUT="$OUT" setsid nohup "$SELF" run > "$log" 2>&1 < /dev/null &
    echo $! > "$OUT.work/run.pid"
    echo "started (pid $!), log $log"
    echo "progress: $SELF status" ;;
  status)
    resolve_out
    [ -n "$OUT" ] || die "no run yet ($CURRENT does not exist)"
    WORK="$OUT.work"; TAPS="${TAPS:-$WORK/taps}"
    echo "run: $OUT"
    for s in "${STAGES[@]}"; do printf '  %-7s %s\n' "$s" "$(cat "$WORK/.done-$s" 2>/dev/null || echo -)"; done
    if [ -f "$WORK/RESULT" ]; then cat "$WORK/RESULT"; exit 0; fi
    if [ -f "$WORK/run.pid" ] && kill -0 "$(cat "$WORK/run.pid")" 2>/dev/null; then
      echo "RUNNING (pid $(cat "$WORK/run.pid")), MemAvailable $(mem_avail_gb) GB, $(disk_free_gb "$QUANT_ROOT") GB free"
    else
      echo "NOT RUNNING and not finished - resume with: $SELF start"
    fi
    if [ -f "$WORK/sources.json" ]; then
      echo "-- taps: $(ls "$TAPS"/*.taps.safetensors 2>/dev/null | wc -l) request dumps in $TAPS," \
           "$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$WORK/sources.json") still to dump at the last sources pass"
    fi
    if [ -e "$WORK/run.log" ]; then
      echo "-- AutoRound (the block being tuned, of encoder / layers.0-4 / candidate_selector):"
      tr '\r' '\n' < "$WORK/run.log" | grep -E "^Quantizing (encoder|layers\.[0-9]+|candidate_selector)" | tail -1
      echo "-- last log lines:"
      tr '\r' '\n' < "$WORK/run.log" | grep -v -E "^(Quantizing|packing|Writing)" | grep -v '^\s*$' | tail -5
    fi ;;
  stop)
    exec "$HERE/box_stop.sh" dflash-quant ;;
  selftest)
    W="$QUANT_ROOT/selftest-dflash-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$W"
    env_py
    "$VENVPY" "$Q/test_dflash_quant.py" --make-tiny "$W/tiny"
    DRAFTER="$W/tiny/drafter" TARGET="$W/tiny/target" TAPS="$W/tiny/taps" OUT="$W/out" ITERS=2 NSAMPLES=4 \
      ANCHORS=3 BATCH=2 CONTEXT=64 "$SELF" run --from quant ;;
  run)
    shift
    resolve_out
    run_stages "$@" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
