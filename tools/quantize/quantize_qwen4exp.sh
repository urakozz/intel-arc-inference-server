#!/usr/bin/env bash
# One command: Qwen/Qwen3.8-Flash-Next (bf16, 360 GB) -> our AutoRound int4 g64 symmetric export, tuned,
# the output head and the MTP head's experts included - spec 21 v2 §2, plan 21v2a Tasks 1-3, run on the box.
#
#   tools/quantize/quantize_qwen4exp.sh check-download [--sha]   the original complete? (144 files, sizes / sha256)
#   tools/quantize/quantize_qwen4exp.sh start                    start (or resume) the run, detached
#   tools/quantize/quantize_qwen4exp.sh status                   blocks done, s per block, the ETA, RAM, the result
#   tools/quantize/quantize_qwen4exp.sh stop                     stop the run (TERM to its process group); start resumes
#   tools/quantize/quantize_qwen4exp.sh selftest                 the whole pipeline on a tiny model (~5 min)
#   tools/quantize/quantize_qwen4exp.sh run [--only S | --from S]   the stages in the foreground (what start runs)
#
# Stages of `run` (a finished stage leaves $OUT/.done-<stage>; a rerun skips it):
#   quant  AutoRound (tools/quantize/qwen4exp/autoround_run.py): W4A16, group 64, symmetric, format
#          auto_round:auto_gptq, tuned sign-SGD ITERS (200) on NSAMPLES (128) x SEQLEN (2048) rows of DATASET
#          (nvidia/OpenCodeInstruct through the chat template, from the HF cache; refused if any of its 50
#          parquet shards is missing - operator 2026-11-12: the model is for agentic coding), low CPU memory (the fused-MoE checkpoint
#          is read one block at a time), low GPU memory; on card CARD (1) through tools/gpu_lock.sh, or the box
#          CPU (DEVICE=cpu, /tmp/b70-cpu-model.lock). Per-block resume: AR_RESUME_DIR=$OUT/resume (two fp32
#          files of the calibration activations, 21.5 GB, overwritten each block) - after a stop, `start` again
#          continues at the first unfinished block. The PLE table (102 GB) is streamed once into a raw file in
#          PLE_DIR (NVMe) and read from there by mmap (autoround_run.py: AutoRound's own load would need ~190 GB).
#   bf16   every tensor AutoRound did not quantise is bitwise the source's (the ignore list, routers, norms, the
#          PLE table, ...); on DEVICE=cpu its fp32 widenings are restored to bf16 (tools/quantize/qwen4exp/bf16_same.py)
#   head   lm_head packed by the engine's RTN g64 rule and declared 4-bit (tools/quantize/qwen4exp/head_rtn.py):
#          AutoRound's quant_lm_head would turn its low-CPU-memory mode and the resume off, for a head it only
#          RTNs anyway (with BF16 scales the loader refuses)
#   ple    the PLE table to the engine's int8 file, $EXPORT-ple-int8 (tools/quantize/qwen4exp/ple_int8.py)
#   check  tools/quantize/qwen4exp/check.py (the engine's rules for this family, the int8 PLE file sampled bit
#          for bit) and tools/quantize/check_gptq_export.py --lm-head quant --source (every linear packed or
#          declared bf16; the vision tower, which the engine never loads, not counted); provenance appended to
#          the export's README.md; the last line is "FLASH-NEXT QUANT: OK <export>" or "FLASH-NEXT QUANT: FAIL ..." (also in $OUT/RESULT)
#
# Disks: QUANT_ROOT (the 8 TB disk: the export grows ~1.4 GB a layer, +102 GB of bf16 PLE table after layer 1,
# ~172 GB at the end, + the int8 PLE file 51.8 GB + resume/ 21.5 GB = ~250 GB) and PLE_DIR (NVMe, 102 GB while
# `quant` runs; move it to ~/b70/trash afterwards).
# Guards: `start` refuses unless the download check passes, MemAvailable >= RAM_MIN_GB (100), QUANT_ROOT has
# >= DISK_MIN_GB (300) and PLE_DIR's disk >= 110 GB free (or holds the file already); AutoRound runs in a systemd scope capped at RAM_CAP_GB
# (115) with no swap, so a memory overrun kills the run (resumable) instead of the box.
#
# AutoRound: AR_COMMIT a6cf7487 of the operator's fork ~/auto-round (AR_REPO; the commit is only there): upstream
# main b58069ea (2026-11-11, version string 0.17.0) + one local commit that reads OpenCodeInstruct's cached parquet
# and loads snapshots local-only; no patch file (AR_PATCH=0): the BOS/EOS fix of auto-round-local.patch is
# upstream (#2591) and nothing on this path reads WrapperLinear.in_features. Why not the Qwen3.8 scripts' 6afaecdb: b58069ea adds the per-block resume
# (AR_RESUME_DIR), the vectorised QSA indexer (#2579, the 12 QSA blocks) and routed MoE forward (#2615);
# both commits pin Qwen4's ngram table on the CPU and export the MTP head (tools/quantize/README.md,
# "Qwen3.8-Flash-Next: our AutoRound export"). Installed by tools/quantize/ar_pin.sh beside VENVPY
# (~/auto-round/.venv: torch 2.14 + XPU), with transformers 5.19.0 (the reference's pin,
# tools/oracle/qwen4exp_requirements.txt) in SITE first on PYTHONPATH.
#
# Overridables (environment): MODEL (Qwen/Qwen3.8-Flash-Next or a snapshot dir), QUANT_ROOT
# ($HOME/b70/hdd/quant), OUT (default: the run `current` points at, else QUANT_ROOT/Qwen3.8-Flash-Next-
# W4A16-g64-<date>), PLE_DIR ($HOME/b70/scratch/flash-next-ple), DEVICE (xpu: card CARD | cpu), CARD (1), ITERS, NSAMPLES, SEQLEN, BATCH (8), RAM_CAP_GB, RAM_MIN_GB,
# DISK_MIN_GB, DATASET, VENVPY, SITE, AR_REPO, LOGDIR ($HOME/b70/logs/flash-next-quant).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"         # tools/
Q="$HERE/quantize/qwen4exp"
SELF="$HERE/quantize/quantize_qwen4exp.sh"
MODEL="${MODEL:-Qwen/Qwen3.8-Flash-Next}"
QUANT_ROOT="${QUANT_ROOT:-$HOME/b70/hdd/quant}"
LOGDIR="${LOGDIR:-$HOME/b70/logs/flash-next-quant}"
DEVICE="${DEVICE:-xpu}"
CARD="${CARD:-1}"
ITERS="${ITERS:-200}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
BATCH="${BATCH:-8}"
RAM_CAP_GB="${RAM_CAP_GB:-115}"
RAM_MIN_GB="${RAM_MIN_GB:-100}"
DISK_MIN_GB="${DISK_MIN_GB:-300}"
PLE_DIR="${PLE_DIR:-$HOME/b70/scratch/flash-next-ple}"
VENVPY="${VENVPY:-$HOME/auto-round/.venv/bin/python}"
SITE="${SITE:-$HOME/.cache/q4exp-site-py312}"
DATASET="${DATASET:-nvidia/OpenCodeInstruct:apply_chat_template}"
export AR_COMMIT=a6cf748739f2dd58e6c16b0ac849e92af7396e32 AR_PATCH=0 AR_REPO="${AR_REPO:-$HOME/auto-round}"
CURRENT="$QUANT_ROOT/flash-next-current"
STAGES=(quant bf16 head ple check)

die() { echo "quantize_qwen4exp: $*" >&2; exit 2; }

snapshot() {   # MODEL -> its snapshot directory (refs/main of the HF cache entry, or MODEL itself)
  if [ -f "$MODEL/config.json" ]; then echo "$MODEL"; return; fi
  local e="$HOME/.cache/huggingface/hub/models--${MODEL//\//--}"
  [ -f "$e/refs/main" ] || die "$MODEL is not in the HF cache ($e)"
  echo "$e/snapshots/$(cat "$e/refs/main")"
}

env_py() {     # the pinned AutoRound beside VENVPY, transformers 5.19.0 in SITE; sets PKG
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
    AR_DISABLE_META_LOAD=1 flock /tmp/b70-cpu-model.lock "${cap[@]}" "$@"
  fi
}

export_dir() { dirname "$(find "$OUT/export" -name config.json -printf '%d %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2)"; }

stage_quant() {
  mkdir -p "$OUT/work" "$OUT/resume"
  AR_WORK_SPACE="$OUT/work" AR_RESUME_DIR="$OUT/resume" locked "$VENVPY" "$Q/autoround_run.py" \
    --model "$SNAP" --out "$OUT/export" --device "$DEVICE" --iters "$ITERS" --nsamples "$NSAMPLES" \
    --seqlen "$SEQLEN" --batch-size "$BATCH" --ple-dir "$PLE_DIR" --dataset "$DATASET"
}

stage_bf16() {
  "$VENVPY" "$Q/bf16_same.py" "$(export_dir)" "$SNAP"
}

stage_head() {
  "$VENVPY" "$Q/head_rtn.py" "$(export_dir)"
}

stage_ple() {
  local ex; ex=$(export_dir)
  "$VENVPY" "$Q/ple_int8.py" "$ex" "$ex-ple-int8"
}

stage_check() {
  local ex ok=1; ex=$(export_dir)
  "$VENVPY" "$Q/check.py" "$ex" --ple "$ex-ple-int8" || ok=0
  "$VENVPY" "$HERE/quantize/check_gptq_export.py" "$ex" --group 64 --lm-head quant --source "$SNAP" \
    --skip-prefix model.visual. || ok=0
  if [ $ok = 1 ]; then
    cat >> "$ex/README.md" <<EOF

## Quantisation provenance

- tool: auto-round at \`$AR_COMMIT\` (the operator's fork: intel/auto-round \`b58069ea\` + the cached-OpenCodeInstruct commit; version string $("$VENVPY" -c 'import auto_round; print(auto_round.__version__)'))
- recipe: \`tools/quantize/quantize_qwen4exp.sh\` - W4A16, group 64, symmetric, format
  auto_round:auto_gptq, tuned: $ITERS iterations, $NSAMPLES x $SEQLEN rows of \`$DATASET\`, batch $BATCH;
  kept bf16: routers, hyper-connections, shared_expert_gate, GDN in_proj_a / in_proj_b, the QSA indexer, the
  PLE layer, norms, the embedding, vision, the MTP head's attention and shared expert; the MTP head's routed
  experts RTN int4 g64 (not tuned: transformers does not build the head); \`lm_head\` RTN int4 g64 by
  the engine's rule (\`tools/quantize/qwen4exp/head_rtn.py\`); qwen4exp_recipe.json has the call
- source: \`$MODEL\` at \`$(basename "$SNAP")\`; made $(date -u +%Y-%m-%dT%H:%MZ)
- checked by \`tools/quantize/qwen4exp/check.py\` (with the int8 PLE file \`$(basename "$ex")-ple-int8\`) and
  \`tools/quantize/check_gptq_export.py --lm-head quant --source\`
EOF
    echo "FLASH-NEXT QUANT: OK $ex" | tee "$OUT/RESULT"
  else
    echo "FLASH-NEXT QUANT: FAIL - a check refused $ex (the lines above say what)" | tee "$OUT/RESULT"
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
  SNAP=$(snapshot)
  env_py
  echo "quantize_qwen4exp: $(date -u +%FT%TZ) MODEL=$MODEL ($SNAP) OUT=$OUT DEVICE=$DEVICE CARD=$CARD ITERS=$ITERS" \
       "NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN BATCH=$BATCH DATASET=$DATASET auto-round ${AR_COMMIT:0:8}"
  for s in "${STAGES[@]}"; do
    [ "$s" = "$from" ] && hit=1
    if [ -n "$only" ] && [ "$s" != "$only" ]; then continue; fi
    if [ -z "$only" ] && [ -n "$from" ] && [ $hit = 0 ]; then continue; fi
    if [ -z "$only$from" ] && [ -f "$OUT/.done-$s" ]; then echo "== $s: done $(cat "$OUT/.done-$s")"; continue; fi
    echo "== $s $(date -u +%FT%TZ)"
    "stage_$s"
    date -u +%FT%TZ > "$OUT/.done-$s"
  done
}

resolve_out() { [ -n "${OUT:-}" ] || OUT=$( [ -L "$CURRENT" ] && readlink -f "$CURRENT" || true); }

case "${1:-}" in
  check-download)
    shift
    "$VENVPY" "$Q/check_download.py" --repo "$MODEL" "$@" ;;
  start)
    mkdir -p "$QUANT_ROOT" "$LOGDIR"
    resolve_out
    if [ -n "$OUT" ] && [ -f "$OUT/run.pid" ] && kill -0 "$(cat "$OUT/run.pid")" 2>/dev/null; then
      die "already running (pid $(cat "$OUT/run.pid"), $OUT) - see: $SELF status"
    fi
    if [ -n "$OUT" ] && [ -f "$OUT/.done-check" ]; then
      die "the run in $OUT is finished ($(cat "$OUT/RESULT")); for a new one: OUT=<new dir> $SELF start"
    fi
    [ -f "$MODEL/config.json" ] || "$VENVPY" "$Q/check_download.py" --repo "$MODEL" || die "the download is not complete"
    if [ "${DATASET%%:*}" = nvidia/OpenCodeInstruct ]; then
      n=$(ls "$HOME"/.cache/huggingface/hub/datasets--nvidia--OpenCodeInstruct/snapshots/*/data/train-*.parquet 2>/dev/null | wc -l)
      [ "$n" = 50 ] || die "calibration data: $n of 50 nvidia/OpenCodeInstruct parquet shards in the HF cache"
    fi
    m=$(mem_avail_gb); [ "$m" -ge "$RAM_MIN_GB" ] || die "MemAvailable ${m} GB < RAM_MIN_GB $RAM_MIN_GB - something else is using the RAM"
    d=$(disk_free_gb "$QUANT_ROOT"); [ "$d" -ge "$DISK_MIN_GB" ] || die "$QUANT_ROOT has ${d} GB free < DISK_MIN_GB $DISK_MIN_GB"
    mkdir -p "$PLE_DIR"
    if ! ls "$PLE_DIR"/*.bf16.json >/dev/null 2>&1; then
      d=$(disk_free_gb "$PLE_DIR"); [ "$d" -ge 110 ] || die "$PLE_DIR has ${d} GB free < 110 (the 102 GB PLE file)"
    fi
    if [ -z "$OUT" ]; then
      OUT="$QUANT_ROOT/Qwen3.8-Flash-Next-W4A16-g64-$(date +%F)"
      mkdir -p "$OUT"; ln -sfn "$OUT" "$CURRENT"
      echo "new run: $OUT"
    else
      echo "resuming: $OUT"
    fi
    log="$LOGDIR/run-$(date +%Y%m%d-%H%M%S).log"
    ln -sfn "$log" "$OUT/run.log"
    OUT="$OUT" setsid nohup "$SELF" run > "$log" 2>&1 < /dev/null &
    echo $! > "$OUT/run.pid"
    echo "started (pid $!), log $log"
    echo "progress: $SELF status" ;;
  status)
    resolve_out
    [ -n "$OUT" ] || die "no run yet ($CURRENT does not exist)"
    echo "run: $OUT"
    for s in "${STAGES[@]}"; do printf '  %-6s %s\n' "$s" "$(cat "$OUT/.done-$s" 2>/dev/null || echo -)"; done
    if [ -f "$OUT/RESULT" ]; then cat "$OUT/RESULT"; exit 0; fi
    if [ -f "$OUT/run.pid" ] && kill -0 "$(cat "$OUT/run.pid")" 2>/dev/null; then
      pid=$(cat "$OUT/run.pid")
      rss=$(ps -o rss= -g "$(ps -o sid= -p "$pid" | tr -d ' ')" 2>/dev/null | awk '{s += $1} END {printf "%.1f", s / 1048576}')
      echo "RUNNING (pid $pid), RSS of the run ${rss} GiB, MemAvailable $(mem_avail_gb) GB, $(disk_free_gb "$QUANT_ROOT") GB free"
    else
      echo "NOT RUNNING and not finished - resume with: $SELF start"
    fi
    if [ -e "$OUT/run.log" ]; then
      echo "-- blocks (AutoRound's bar: block, done/48 [elapsed<remaining, s per block]; on a resume it counts this process):"
      tr '\r' '\n' < "$OUT/run.log" | grep -E "^Quantizing model\.language_model\.layers\.[0-9]+:" | tail -1
      echo "-- seconds per finished block (AutoRound's per-block memory lines; the first includes the warm-up):"
      grep -E "INFO device.py .*'peak_ram'" "$OUT/run.log" | awk '{ split($2, t, ":"); s = t[1] * 3600 + t[2] * 60 + t[3];
        if (p != "") { d = s - p; if (d < 0) d += 86400; printf " %d", d }; p = s } END { print "" }'
      echo "-- last log lines:"
      tr '\r' '\n' < "$OUT/run.log" | grep -v -E "^(Quantizing|packing|Writing|Map|Filter)" | grep -v '^\s*$' | tail -5
    fi ;;
  stop)
    resolve_out
    [ -n "$OUT" ] && [ -f "$OUT/run.pid" ] && kill -0 "$(cat "$OUT/run.pid")" 2>/dev/null || die "no run is running"
    kill -TERM -- "-$(cat "$OUT/run.pid")"
    echo "stopped the run in $OUT (pid $(cat "$OUT/run.pid") and its children); resume with: $SELF start" ;;
  selftest)
    W="$QUANT_ROOT/selftest-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$W"
    SNAP0=$(MODEL="$MODEL" snapshot 2>/dev/null || true)
    proc="$SNAP0"; [ -f "$proc/preprocessor_config.json" ] ||
      proc=$(ls -d "$HOME"/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound/snapshots/*/ | grep -v ple-int8 | head -1)
    tok=$(ls -d "$HOME"/.cache/huggingface/hub/models--qikp--tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next/snapshots/*/ | head -1)
    env_py
    "$VENVPY" "$Q/make_tiny_vl.py" "$W/tiny" --tokenizer "$tok" --processor "$proc"
    MODEL="$W/tiny" OUT="$W/out" PLE_DIR="$W/ple" ITERS=2 NSAMPLES=8 SEQLEN=128 BATCH=4 "$SELF" run ;;
  run)
    shift
    resolve_out
    run_stages "$@" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
