#!/usr/bin/env bash
# One command: ricursiveAI/Ripple-65B-Preview (bf16, 130 GB, remote code) -> our AutoRound int4 g64 symmetric
# export, tuned; lm_head and the embedding stay bf16 (the engine builds its int8 head at load). On the GPU machine.
#
#   tools/quantize/quantize_ripple.sh start                    start (or resume) the run, detached
#   tools/quantize/quantize_ripple.sh status                   blocks done, s per block, the ETA, RAM, the result
#   tools/quantize/quantize_ripple.sh stop                     stop the run; start resumes
#   tools/quantize/quantize_ripple.sh selftest                 the whole pipeline on a tiny random Ripple, CPU
#                                                              (both attention arms, ~5 min)
#   tools/quantize/quantize_ripple.sh run [--only S | --from S]   the stages in the foreground (what start runs)
#   ATTN=bf16 tools/quantize/quantize_ripple.sh start          the mixed-precision arm: attention bf16, MLP int4
#
# Stages of `run` (a finished stage leaves $OUT/.done-<stage>; a rerun skips it):
#   quant  AutoRound (tools/quantize/ripple/autoround_run.py): W4A16, group 64, symmetric, format
#          auto_round:auto_gptq, tuned sign-SGD ITERS (200) on NSAMPLES (128) x SEQLEN (2048) rows of DATASET
#          (nvidia/OpenCodeInstruct through the chat template, from the HF cache; refused if any of its 50 parquet
#          shards is missing), batch BATCH (8) x GAS (1); the model through its remote code (trust_remote_code);
#          low CPU memory as a meta skeleton with each block read from the checkpoint when it is tuned
#          (AR_DISK_STREAM_MODEL=1: AutoRound does that by itself only for fused-MoE checkpoints, and 130 GB of
#          bf16 does not fit the RAM); a guard refuses any decoder-layer call without its value residual; low GPU
#          memory; on card CARD (1) under its lock, or the CPU (DEVICE=cpu,
#          the CPU lock B70_CPU_LOCK when set). Per-block resume: AR_RESUME_DIR=$OUT/resume (two fp32 files of the
#          calibration activations, 2 x 5.4 GB, overwritten each block) - after a stop, `start` again continues at
#          the first unfinished block.
#          ATTN=int4 (default): every linear of the 80 blocks int4. ATTN=bf16: self_attn q/k/v/o/g kept bf16
#          (extra_config bits 16, which the loader accepts today), only the MLP int4 - the fallback if int4
#          attention costs too much quality; int8 attention would be this export with the engine quantising the
#          bf16 attention to int8 at load (its lm_head int8 path), as the loader takes int4 and bf16 only.
#   scales every int4 group scale in f16 (tools/quantize/qwen4exp/scales_f16.py; a tuned export is already f16)
#   bf16   every tensor AutoRound did not quantise is bitwise the source's (tools/quantize/qwen4exp/bf16_same.py;
#          on DEVICE=cpu its fp32 widenings are restored to bf16)
#   check  tools/quantize/ripple/check.py (config, remote code, tokenizer and template the source's, the attention
#          arm; RELOAD=1: the export reloaded by transformers with trust_remote_code against the engine's
#          dequantisation) and tools/quantize/check_gptq_export.py --lm-head keep --source; provenance appended to
#          the export's README.md; the last line is "RIPPLE QUANT: OK <export>" or "RIPPLE QUANT: FAIL ..."
#          (also in $OUT/RESULT)
#
# Disk: QUANT_ROOT ($B70_DATA/quant on the NVMe): the export ~36 GB (ATTN=bf16: ~56 GB) + resume/ 10.7 GB; `start`
# wants DISK_MIN_GB (80) free. RAM: `start` wants MemAvailable >= RAM_MIN_GB (45); AutoRound runs in a systemd
# scope capped at RAM_CAP_GB (64) with no swap, so an overrun kills the run (resumable), never the machine.
#
# AutoRound: the latest of our fork ~/auto-round (branch AR_REF, tools/quantize/ar_latest.sh), the same build rule
# as the Flash-Next script; a run keeps the hash it started with ($OUT/ar_commit) and the attention arm ($OUT/attn).
# Installed by tools/quantize/ar_pin.sh beside VENVPY (~/auto-round/.venv), transformers 5.19.0 in SITE.
#
# Overridables (environment): MODEL (the pinned snapshot below, or another snapshot dir), QUANT_ROOT
# ($B70_DATA/quant), OUT (default: the run `ripple-current` points at - `start` points it at the run it starts - else
# QUANT_ROOT/Ripple-65B-Preview-W4A16-g64[-attn-bf16]-<date>), ATTN (int4 | bf16), DEVICE (xpu: card CARD | cpu),
# CARD (1), ITERS, NSAMPLES, SEQLEN, BATCH, GAS, RAM_CAP_GB, RAM_MIN_GB, DISK_MIN_GB, DATASET, RELOAD, VENVPY, SITE, AR_REPO, AR_REF,
# LOGDIR ($B70_DATA/logs/ripple-quant), SCRATCH ($B70_DATA/scratch/ripple-quant, the selftest's directory).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"         # tools/
Q="$HERE/quantize/ripple"
Q4="$HERE/quantize/qwen4exp"
SELF="$HERE/quantize/quantize_ripple.sh"
MODEL="${MODEL:-$HOME/.cache/huggingface/hub/models--ricursiveAI--Ripple-65B-Preview/snapshots/d58dbcf43344e27494a40cec6727e180450f4a6b}"
QUANT_ROOT="${QUANT_ROOT:-${B70_DATA:?set B70_DATA, the data root}/quant}"
LOGDIR="${LOGDIR:-${B70_DATA:?set B70_DATA, the data root}/logs/ripple-quant}"
SCRATCH="${SCRATCH:-${B70_DATA:?set B70_DATA, the data root}/scratch/ripple-quant}"
ATTN="${ATTN:-int4}"
DEVICE="${DEVICE:-xpu}"
CARD="${CARD:-1}"
ITERS="${ITERS:-200}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
BATCH="${BATCH:-8}"
GAS="${GAS:-1}"
RAM_CAP_GB="${RAM_CAP_GB:-64}"
RAM_MIN_GB="${RAM_MIN_GB:-45}"
DISK_MIN_GB="${DISK_MIN_GB:-80}"
RELOAD="${RELOAD:-0}"
VENVPY="${VENVPY:-$HOME/auto-round/.venv/bin/python}"
SITE="${SITE:-$HOME/.cache/q4exp-site-py312}"
DATASET="${DATASET:-nvidia/OpenCodeInstruct:apply_chat_template}"
. "$HERE/quantize/ar_latest.sh"     # AR_COMMIT = the latest AutoRound (branch AR_REF of AR_REPO), see there
CURRENT="$QUANT_ROOT/ripple-current"
STAGES=(quant scales bf16 check)

die() { echo "quantize_ripple: $*" >&2; exit 2; }
case "$ATTN" in int4|bf16) ;; *) die "ATTN must be int4 or bf16, not '$ATTN'" ;; esac
[ -f "$MODEL/config.json" ] || die "MODEL=$MODEL is not a snapshot directory (no config.json)"

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
    # through the machine's card wrapper when B70_CARD_RUN names one (<wrapper> <card> -- <cmd>), else by affinity alone
    if [ -n "${B70_CARD_RUN:-}" ]; then $B70_CARD_RUN "$CARD" -- "${cap[@]}" "$@"
    else ZE_AFFINITY_MASK="$CARD" "${cap[@]}" "$@"; fi
  else
    ${B70_CPU_LOCK:+flock "$B70_CPU_LOCK"} "${cap[@]}" "$@"
  fi
}

export_dir() { dirname "$(find "$OUT/export" -name config.json -printf '%d %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2)"; }

attn_guard() {   # a run keeps its attention arm
  if [ -f "$OUT/attn" ]; then
    [ "$(cat "$OUT/attn")" = "$ATTN" ] || die "the run in $OUT is ATTN=$(cat "$OUT/attn"); this call says ATTN=$ATTN"
  else
    mkdir -p "$OUT"; echo "$ATTN" > "$OUT/attn"
  fi
}

stage_quant() {
  mkdir -p "$OUT/work" "$OUT/resume"
  AR_WORK_SPACE="$OUT/work" AR_RESUME_DIR="$OUT/resume" locked "$VENVPY" "$Q/autoround_run.py" \
    --model "$MODEL" --out "$OUT/export" --device "$DEVICE" --attn "$ATTN" --iters "$ITERS" \
    --nsamples "$NSAMPLES" --seqlen "$SEQLEN" --batch-size "$BATCH" --gas "$GAS" --dataset "$DATASET"
}

stage_scales() {
  "$VENVPY" "$Q4/scales_f16.py" "$(export_dir)"
}

stage_bf16() {
  "$VENVPY" "$Q4/bf16_same.py" "$(export_dir)" "$MODEL"
}

stage_check() {
  local ex ok=1 reload=(); ex=$(export_dir)
  if [ "$RELOAD" = 1 ]; then
    reload=(--reload)
    ${B70_CPU_LOCK:+flock "$B70_CPU_LOCK"} "$VENVPY" "$Q/check.py" "$ex" "$MODEL" --attn "$ATTN" "${reload[@]}" || ok=0
  else
    "$VENVPY" "$Q/check.py" "$ex" "$MODEL" --attn "$ATTN" || ok=0
  fi
  "$VENVPY" "$HERE/quantize/check_gptq_export.py" "$ex" --group 64 --lm-head keep --source "$MODEL" || ok=0
  if [ $ok = 1 ]; then
    cat >> "$ex/README.md" <<EOF

## Quantisation provenance

- tool: auto-round at \`$AR_COMMIT\` (our fork, branch \`$AR_REF\`: intel/auto-round main \`$AR_BASE\` + the cached-OpenCodeInstruct commit; version string $("$VENVPY" -c 'import auto_round; print(auto_round.__version__)'))
- recipe: \`tools/quantize/quantize_ripple.sh\` - W4A16, group 64, symmetric, format auto_round:auto_gptq,
  $( [ "$ITERS" = 0 ] && echo "RTN on" || echo "tuned: $ITERS iterations on") $NSAMPLES x $SEQLEN rows of \`$DATASET\`, batch $BATCH x $GAS;
  int4: $( [ "$ATTN" = bf16 ] && echo "the MLP (gate / up / down); self_attn q / k / v / o / g kept bf16" || echo "every linear of the 80 blocks (self_attn q / k / v / o / g, MLP gate / up / down)");
  kept bf16: norms, q / k norms, value_res_lambda, the embedding, \`lm_head\` (the engine builds its int8 head at
  load); ripple_recipe.json has the call
- source: \`ricursiveAI/Ripple-65B-Preview\` at \`$(basename "$MODEL")\`; made $(date -u +%Y-%m-%dT%H:%MZ)
- checked by \`tools/quantize/ripple/check.py\` and \`tools/quantize/check_gptq_export.py --lm-head keep --source\`
EOF
    echo "RIPPLE QUANT: OK $ex" | tee "$OUT/RESULT"
  else
    echo "RIPPLE QUANT: FAIL - a check refused $ex (the lines above say what)" | tee "$OUT/RESULT"
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
  env_py
  ar_run_guard "$OUT" || exit 2
  attn_guard
  echo "quantize_ripple: $(date -u +%FT%TZ) MODEL=$MODEL OUT=$OUT ATTN=$ATTN DEVICE=$DEVICE CARD=$CARD ITERS=$ITERS" \
       "NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN BATCH=$BATCH GAS=$GAS DATASET=$DATASET auto-round $AR_COMMIT" \
       "(base ${AR_BASE:0:8}, $AR_REF)"
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
  start)
    mkdir -p "$QUANT_ROOT" "$LOGDIR"
    resolve_out
    if [ -n "$OUT" ] && [ -f "$OUT/run.pid" ] && kill -0 "$(cat "$OUT/run.pid")" 2>/dev/null; then
      die "already running (pid $(cat "$OUT/run.pid"), $OUT) - see: $SELF status"
    fi
    if [ -n "$OUT" ] && [ -f "$OUT/.done-check" ]; then
      die "the run in $OUT is finished ($(cat "$OUT/RESULT")); for a new one: OUT=<new dir> $SELF start"
    fi
    if [ "${DATASET%%:*}" = nvidia/OpenCodeInstruct ]; then
      n=$(ls "$HOME"/.cache/huggingface/hub/datasets--nvidia--OpenCodeInstruct/snapshots/*/data/train-*.parquet 2>/dev/null | wc -l)
      [ "$n" = 50 ] || die "calibration data: $n of 50 nvidia/OpenCodeInstruct parquet shards in the HF cache"
    fi
    ar_resolve || exit 2
    [ -z "$OUT" ] || ar_run_guard "$OUT" || exit 2
    [ -z "$OUT" ] || attn_guard
    m=$(mem_avail_gb); [ "$m" -ge "$RAM_MIN_GB" ] || die "MemAvailable ${m} GB < RAM_MIN_GB $RAM_MIN_GB - something else is using the RAM"
    d=$(disk_free_gb "$QUANT_ROOT"); [ "$d" -ge "$DISK_MIN_GB" ] || die "$QUANT_ROOT has ${d} GB free < DISK_MIN_GB $DISK_MIN_GB"
    if [ -z "$OUT" ]; then
      OUT="$QUANT_ROOT/Ripple-65B-Preview-W4A16-g64$( [ "$ATTN" = bf16 ] && echo -attn-bf16)-$(date +%F)"
      echo "new run: $OUT"
    elif [ -f "$OUT/run.pid" ]; then
      echo "resuming: $OUT"
      # a FAIL of the previous attempt is not this run's result (status reads RESULT)
      [ ! -f "$OUT/RESULT" ] || mv "$OUT/RESULT" "$OUT/RESULT.prev"
    else
      echo "new run: $OUT"
    fi
    # `current` names the run started last: status, stop and start without OUT act on it
    OUT=$(readlink -m "$OUT"); mkdir -p "$OUT"; ln -sfn "$OUT" "$CURRENT"
    log="$LOGDIR/run-$(date +%Y%m%d-%H%M%S).log"
    ln -sfn "$log" "$OUT/run.log"
    OUT="$OUT" setsid nohup "$SELF" run > "$log" 2>&1 < /dev/null &
    echo $! > "$OUT/run.pid"
    echo "started (pid $!), log $log"
    echo "progress: $SELF status" ;;
  status)
    resolve_out
    [ -n "$OUT" ] || die "no run yet ($CURRENT does not exist)"
    echo "run: $OUT (ATTN=$(cat "$OUT/attn" 2>/dev/null || echo ?))"
    for s in "${STAGES[@]}"; do printf '  %-6s %s\n' "$s" "$(cat "$OUT/.done-$s" 2>/dev/null || echo -)"; done
    if [ -f "$OUT/run.pid" ] && kill -0 "$(cat "$OUT/run.pid")" 2>/dev/null; then
      pid=$(cat "$OUT/run.pid")
      rss=$(ps -o rss= -g "$(ps -o sid= -p "$pid" | tr -d ' ')" 2>/dev/null | awk '{s += $1} END {printf "%.1f", s / 1048576}')
      echo "RUNNING (pid $pid), RSS of the run ${rss} GiB, MemAvailable $(mem_avail_gb) GB, $(disk_free_gb "$QUANT_ROOT") GB free"
    elif [ -f "$OUT/RESULT" ]; then
      cat "$OUT/RESULT"; exit 0
    else
      echo "NOT RUNNING and not finished - resume with: $SELF start"
    fi
    if [ -e "$OUT/run.log" ]; then
      echo "-- blocks (AutoRound's bar: block, done/80 [elapsed<remaining, s per block]; on a resume it counts this process):"
      tr '\r' '\n' < "$OUT/run.log" | grep -E "^Quantizing model\.layers\.[0-9]+:" | tail -1
      echo "-- seconds per finished block (AutoRound's per-block memory lines; the first includes the warm-up):"
      grep -E "INFO device.py .*'peak_ram'" "$OUT/run.log" | awk '{ split($2, t, ":"); s = t[1] * 3600 + t[2] * 60 + t[3];
        if (p != "") { d = s - p; if (d < 0) d += 86400; printf " %d", d }; p = s } END { print "" }'
      echo "-- last log lines:"
      tr '\r' '\n' < "$OUT/run.log" | grep -v -E "^(Quantizing|packing|Writing|Map|Filter)" | grep -v '^\s*$' | tail -5
    fi ;;
  stop)   # the run CURRENT points at (the last started): TERM to its process group
    [ -z "${OUT:-}" ] || [ "$(readlink -m "$OUT")" = "$(readlink -f "$CURRENT")" ] ||
      die "OUT=$OUT is not the run $CURRENT points at (the last one started), which is the one stop stops"
    run=$(readlink -f "$CURRENT" 2>/dev/null || true); p=$(cat "$run/run.pid" 2>/dev/null || true)
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null || die "no run is running"
    kill -TERM -- "-$p"
    echo "stopped the run in $run (pid $p and its children); resume with: $SELF start" ;;
  selftest)
    W="$SCRATCH/selftest-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$W" "$LOGDIR"
    env_py
    "$VENVPY" "$Q/make_tiny.py" "$W/tiny" --source "$MODEL"
    for arm in int4 bf16; do
      MODEL="$W/tiny" OUT="$W/out-$arm" ATTN=$arm DEVICE=cpu ITERS=10 NSAMPLES=8 SEQLEN=128 BATCH=4 GAS=1 \
        RAM_CAP_GB=16 RELOAD=1 "$SELF" run
    done
    echo "RIPPLE SELFTEST: OK both arms ($W)" ;;
  run)
    shift
    resolve_out
    run_stages "$@" ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
