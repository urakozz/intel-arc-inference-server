#!/usr/bin/env bash
# One command: nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 (bf16, 249 GB) -> our AutoRound int4 g64 symmetric
# export, tuned, text-only, with the MTP layer - run on the GPU machine.
#
#   tools/quantize/quantize_nemotron.sh check-download [--sha]   the original complete? (sizes / sha256 vs the hub)
#   tools/quantize/quantize_nemotron.sh start                    start (or resume) the run, detached
#   tools/quantize/quantize_nemotron.sh status                   blocks done, s per block, RAM, disk, the result
#   tools/quantize/quantize_nemotron.sh stop                     stop the run; start resumes
#   tools/quantize/quantize_nemotron.sh selftest                 the whole pipeline on a tiny model (CPU, ~10 min)
#   tools/quantize/quantize_nemotron.sh run [--only S | --from S]   the stages in the foreground (what start runs)
#
# Stages of `run` (a finished stage leaves $OUT/.done-<stage>; a rerun skips it):
#   quant  AutoRound (tools/quantize/nemotron/autoround_run.py): W4A16, group 64, symmetric, format
#          auto_round:auto_gptq, tuned sign-SGD ITERS (200) on NSAMPLES (128) x SEQLEN (2048) rows of DATASET
#          (nvidia/OpenCodeInstruct through the chat template, from the HF cache; refused if any of its 50
#          parquet shards is missing - the model is for agentic coding), low CPU memory (a meta skeleton, one block
#          of the 249 GB original materialised at a time), low GPU memory; on card CARD (1, the quant card: card 0 stays free) through
#          its lock, or the CPU (DEVICE=cpu, the CPU lock B70_CPU_LOCK when set). Quantised: every block's routed
#          experts, shared expert, latent projections, Mamba in/out_proj, attention q/k/v/o; kept bf16: routers,
#          conv1d, norms, A_log / D / dt_bias, the embedding, lm_head (the engine builds its int8 head at load).
#          The MTP layer is copied by AutoRound at export, its 512 routed experts RTN int4 g64, the rest bf16.
#          Per-block resume: AR_RESUME_DIR=$OUT/resume (two fp32 files of the calibration activations,
#          128 x 2048 x 4096 x 4 B = 4.3 GB each, overwritten each block) - after a stop, `start` again continues at
#          the first unfinished block.
#   strip  the vision tower and projector dropped from the export (tools/quantize/nemotron/strip_vision.py)
#   scales every int4 group scale in f16 (tools/quantize/qwen4exp/scales_f16.py): AutoRound keeps the Mamba
#          out_proj scales bf16 for this family; converted exactly or refused
#   bf16   every tensor AutoRound did not quantise is bitwise the source's (tools/quantize/qwen4exp/bf16_same.py):
#          the routers and their F32 bias, norms, conv, A_log / D / dt_bias, embedding, lm_head, the MTP layer's
#          bf16 part; fp32 widenings (DEVICE=cpu, and A_log / D / dt_bias always: AutoRound restores them in fp32)
#          go back to the source's bf16
#   check  tools/quantize/nemotron/check.py (the engine's int4 rules, every name / dtype / shape the reference
#          reads, the whole MTP layer present, no vision); provenance appended to the export's README.md; the last
#          line is "NEMOTRON QUANT: OK <export>" or "NEMOTRON QUANT: FAIL ..." (also in $OUT/RESULT)
#
# Disk: QUANT_ROOT (the 8 TB disk): the export ~68 GB (experts 59.9, MTP experts 1.5, dense 3.6, embedding and
# head 2.1, the MTP layer's bf16 part 0.25) + vision 1.7 until `strip` + resume/ 8.6 GB; `start` wants DISK_MIN_GB
# (150) free. The source is read one block at a time (5.6 GB a MoE block) from wherever MODEL lives.
# Guards: `start` refuses unless the download check passes, MemAvailable >= RAM_MIN_GB (80), QUANT_ROOT has
# >= DISK_MIN_GB free; AutoRound runs in a systemd scope capped at RAM_CAP_GB (100) with no swap, so a memory
# overrun kills the run (resumable) instead of the machine.
#
# AutoRound: the LATEST of our fork ~/auto-round (AR_REPO, branch AR_REF local-patches-v2, ar_latest.sh), recorded
# in $OUT/ar_commit; a run never switches AutoRound (resume with AR_COMMIT=<recorded hash>). Installed by
# tools/quantize/ar_pin.sh beside VENVPY (~/auto-round/.venv: torch 2.14 + XPU), with transformers 5.19.0 in SITE
# first on PYTHONPATH (the Flash-Next run's site).
#
# Overridables (environment): MODEL (the HF id, looked up in HF_HUB then ~/.cache/huggingface/hub, or a snapshot
# dir), HF_HUB (~/.cache/huggingface/hub), QUANT_ROOT ($B70_DATA/quant), OUT (default: the run `nemotron-current`
# points at - `start` points it at the run it starts - else QUANT_ROOT/Nemotron-3.5-Super-W4A16-g64-<date>), DEVICE
# (xpu: card CARD | cpu), CARD (1), ITERS, NSAMPLES, SEQLEN, BATCH (8), RAM_CAP_GB, RAM_MIN_GB, DISK_MIN_GB, DATASET, VENVPY, SITE, AR_REPO, AR_REF,
# LOGDIR ($B70_DATA/logs/nemotron-quant), SCRATCH ($B70_DATA/scratch/nemotron-quant: the selftest's files).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"         # tools/
Q="$HERE/quantize/nemotron"
SELF="$HERE/quantize/quantize_nemotron.sh"
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16}"
HF_HUB="${HF_HUB:-$HOME/.cache/huggingface/hub}"
QUANT_ROOT="${QUANT_ROOT:-${B70_DATA:?set B70_DATA, the data root}/quant}"
LOGDIR="${LOGDIR:-${B70_DATA:?set B70_DATA, the data root}/logs/nemotron-quant}"
SCRATCH="${SCRATCH:-${B70_DATA:?set B70_DATA, the data root}/scratch/nemotron-quant}"
DEVICE="${DEVICE:-xpu}"
CARD="${CARD:-1}"
ITERS="${ITERS:-200}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
BATCH="${BATCH:-8}"
RAM_CAP_GB="${RAM_CAP_GB:-100}"
RAM_MIN_GB="${RAM_MIN_GB:-80}"
DISK_MIN_GB="${DISK_MIN_GB:-150}"
VENVPY="${VENVPY:-$HOME/auto-round/.venv/bin/python}"
SITE="${SITE:-$HOME/.cache/q4exp-site-py312}"
DATASET="${DATASET:-nvidia/OpenCodeInstruct:apply_chat_template}"
. "$HERE/quantize/ar_latest.sh"     # AR_COMMIT = the latest AutoRound (branch AR_REF of AR_REPO), see there
CURRENT="$QUANT_ROOT/nemotron-current"
STAGES=(quant strip scales bf16 check)

die() { echo "quantize_nemotron: $*" >&2; exit 2; }

snapshot() {   # MODEL -> its snapshot directory (refs/main of the HF cache entry, or MODEL itself)
  if [ -f "$MODEL/config.json" ]; then echo "$MODEL"; return; fi
  local hub e
  for hub in "$HF_HUB" "$HOME/.cache/huggingface/hub"; do
    e="$hub/models--${MODEL//\//--}"
    if [ -f "$e/refs/main" ]; then echo "$e/snapshots/$(cat "$e/refs/main")"; return; fi
  done
  die "$MODEL is in neither HF cache ($HF_HUB, ~/.cache/huggingface/hub)"
}

env_py() {     # the latest AutoRound beside VENVPY, transformers 5.19.0 in SITE; sets PKG
  ar_resolve || exit 2
  [ -x "$VENVPY" ] || die "VENVPY=$VENVPY is not an executable python"
  [ -d "$SITE/transformers-5.19.0.dist-info" ] ||
    "$(command -v uv || echo "$HOME/.local/bin/uv")" pip install -q --python "$VENVPY" --no-deps --target "$SITE" \
      -r "$HERE/oracle/qwen4exp_requirements.txt"
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

stage_quant() {
  mkdir -p "$OUT/work" "$OUT/resume"
  AR_WORK_SPACE="$OUT/work" AR_RESUME_DIR="$OUT/resume" locked "$VENVPY" "$Q/autoround_run.py" \
    --model "$SNAP" --out "$OUT/export" --device "$DEVICE" --iters "$ITERS" --nsamples "$NSAMPLES" \
    --seqlen "$SEQLEN" --batch-size "$BATCH" --dataset "$DATASET"
}

stage_strip() {
  "$VENVPY" "$Q/strip_vision.py" "$(export_dir)"
}

stage_scales() {
  "$VENVPY" "$HERE/quantize/qwen4exp/scales_f16.py" "$(export_dir)"
}

stage_bf16() {
  "$VENVPY" "$HERE/quantize/qwen4exp/bf16_same.py" "$(export_dir)" "$SNAP"
}

stage_check() {
  local ex; ex=$(export_dir)
  if "$VENVPY" "$Q/check.py" "$ex"; then
    cat >> "$ex/README.md" <<EOF

## Quantisation provenance

- source: \`$MODEL\` at \`$(basename "$SNAP")\` (NVIDIA, licence OpenMDW-1.1: https://openmdw.ai/license/1-1/ - a
  copy of the licence and NVIDIA's notices must travel with any published copy of this export); made $(date -u +%Y-%m-%dT%H:%MZ)
- tool: auto-round at \`$AR_COMMIT\` (our fork, branch \`$AR_REF\`: intel/auto-round main \`$AR_BASE\` + the cached-OpenCodeInstruct commit; version string $("$VENVPY" -c 'import auto_round; print(auto_round.__version__)'))
- recipe: \`tools/quantize/quantize_nemotron.sh\` - W4A16, group 64, symmetric, format auto_round:auto_gptq,
  $( [ "$ITERS" = 0 ] && echo "RTN weighted by the activation importance of" || echo "tuned: $ITERS iterations on") $NSAMPLES x $SEQLEN rows of \`$DATASET\`, batch $BATCH;
  int4: every block's routed and shared experts, latent projections, Mamba in_proj / out_proj, attention q/k/v/o, and
  the MTP layer's routed experts (RTN, not tuned: transformers does not build the MTP layer); kept bf16 as the
  source ships them: routers (F32 bias), conv1d, norms, A_log / D / dt_bias, the embedding, \`lm_head\` (the engine
  builds its int8 head at load), the MTP layer's eh_proj / attention / shared expert / latent projections; the
  vision tower and projector dropped (text only); nemotron_recipe.json has the call
- checked by \`tools/quantize/nemotron/check.py\`; every unquantised tensor bitwise the source's (\`bf16_same.py\`)
EOF
    echo "NEMOTRON QUANT: OK $ex" | tee "$OUT/RESULT"
  else
    echo "NEMOTRON QUANT: FAIL - check.py refused $ex (the lines above say what)" | tee "$OUT/RESULT"
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
  ar_run_guard "$OUT" || exit 2
  echo "quantize_nemotron: $(date -u +%FT%TZ) MODEL=$MODEL ($SNAP) OUT=$OUT DEVICE=$DEVICE CARD=$CARD ITERS=$ITERS" \
       "NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN BATCH=$BATCH DATASET=$DATASET auto-round $AR_COMMIT (base ${AR_BASE:0:8}, $AR_REF)"
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

# selftest_ref <tiny> <export> <dir>: the reference against transformers (fp32) on the tiny, then the bf16 tiny and
# the int4 export through the reference, compared, MTP drafts included - each a CPU job under the CPU lock. The bar
# is loose on purpose: int4 noise on a random tiny model moves its near-flat logits a lot (mean cosine ~0.9 seen)
selftest_ref() {
  local ids="1,10,4,250,1000,3,77,12,5000,42,9,128,7,64,31,8" ref="$HERE/oracle/nemotron_ref.py"
  local cpu=(${B70_CPU_LOCK:+flock "$B70_CPU_LOCK"} systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 "$VENVPY")
  "${cpu[@]}" "$ref" hfcheck "$1" --prompt "$ids"
  "${cpu[@]}" "$ref" run "$1" --prompt "$ids" --out "$3/ref-bf16.safetensors" --gen 4 --draft 3
  "${cpu[@]}" "$ref" run "$2" --prompt "$ids" --out "$3/ref-int4.safetensors" --gen 4 --draft 3
  "$VENVPY" "$ref" compare "$3/ref-bf16.safetensors" "$3/ref-int4.safetensors" --min-mean-cos 0.8 --mtp
}

case "${1:-}" in
  check-download)
    shift
    HF_HUB_CACHE="$HF_HUB" "$VENVPY" "$HERE/quantize/qwen4exp/check_download.py" --repo "$MODEL" "$@" ;;
  start)
    mkdir -p "$QUANT_ROOT" "$LOGDIR"
    resolve_out
    if [ -n "$OUT" ] && [ -f "$OUT/run.pid" ] && kill -0 "$(cat "$OUT/run.pid")" 2>/dev/null; then
      die "already running (pid $(cat "$OUT/run.pid"), $OUT) - see: $SELF status"
    fi
    if [ -n "$OUT" ] && [ -f "$OUT/.done-check" ]; then
      die "the run in $OUT is finished ($(cat "$OUT/RESULT")); for a new one: OUT=<new dir> $SELF start"
    fi
    [ -f "$MODEL/config.json" ] || HF_HUB_CACHE="$HF_HUB" "$VENVPY" "$HERE/quantize/qwen4exp/check_download.py" \
      --repo "$MODEL" || die "the download is not complete"
    if [ "${DATASET%%:*}" = nvidia/OpenCodeInstruct ]; then
      n=$(ls "$HOME"/.cache/huggingface/hub/datasets--nvidia--OpenCodeInstruct/snapshots/*/data/train-*.parquet 2>/dev/null | wc -l)
      [ "$n" = 50 ] || die "calibration data: $n of 50 nvidia/OpenCodeInstruct parquet shards in the HF cache"
    fi
    ar_resolve || exit 2
    [ -z "$OUT" ] || ar_run_guard "$OUT" || exit 2
    m=$(mem_avail_gb); [ "$m" -ge "$RAM_MIN_GB" ] || die "MemAvailable ${m} GB < RAM_MIN_GB $RAM_MIN_GB - something else is using the RAM"
    d=$(disk_free_gb "$QUANT_ROOT"); [ "$d" -ge "$DISK_MIN_GB" ] || die "$QUANT_ROOT has ${d} GB free < DISK_MIN_GB $DISK_MIN_GB"
    if [ -z "$OUT" ]; then
      OUT="$QUANT_ROOT/Nemotron-3.5-Super-W4A16-g64-$(date +%F)"
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
    echo "run: $OUT"
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
      echo "-- blocks (AutoRound's bar: block, done/88 [elapsed<remaining, s per block]; on a resume it counts this process):"
      tr '\r' '\n' < "$OUT/run.log" | grep -E "^Quantizing language_model\.model\.layers\.[0-9]+:" | tail -1
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
    SRC=$(snapshot)
    env_py
    "$VENVPY" "$HERE/oracle/nemotron_make_tiny.py" "$W/tiny" --source "$SRC"
    MODEL="$W/tiny" OUT="$W/out" DEVICE="${SELFTEST_DEVICE:-cpu}" RAM_CAP_GB=20 ITERS=2 NSAMPLES=8 SEQLEN=128 BATCH=4 "$SELF" run
    OUT="$W/out"
    selftest_ref "$W/tiny" "$(export_dir)" "$W"
    echo "NEMOTRON SELFTEST: OK $W" ;;
  run)
    shift
    resolve_out
    run_stages "$@" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
