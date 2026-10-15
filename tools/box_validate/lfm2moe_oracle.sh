#!/usr/bin/env bash
# Spec 24a: the LFM2.5-24B-A2B (lfm2_moe) CPU reference runs (qwen4exp_oracle.sh's shape) - ON THE BOX CPU, and on
# the Mac (operator, 2026-10-15: the Mac has the room and the bf16 weights).
#   tools/box_validate/lfm2moe_oracle.sh <data tree> tests|ppl|hfcheck|golden|export <dir>|a4
#
# Every mode runs tools/oracle/lfm2moe_ref.py in the oracle image (tools/oracle/run_in_container.sh) with
# transformers 5.19.0 from a site directory BESIDE the image (tools/oracle/qwen4exp_env.sh: pip --no-deps --target;
# the image is never changed) first on PYTHONPATH.
#   tests         the Python tests on the tiny models: test_lfm2moe_ref.py, test_lfm2moe_make_tiny.py, the toolcall
#                 tests (test_score.py, test_make_set.py, test_oracle_batched.py test_lfm2moe) (r48.host)
#   ppl           perplexity of the bf16 original on prose.txt, code.txt (BOS first) and the agentic transcript
#                 (lfm2moe_agentic.ids): finite, prose below an ESTIMATED sanity ceiling of 20
#                 -> <data>/oracle-out-lfm2moe/ppl.log
#   hfcheck       transformers' own model against the port (the restated ops chained) on real weights, --layers 6,
#                 2048 ids of lfm2moe_4k + 4 cached steps: BITWISE or exit 1 -> <data>/oracle-out-lfm2moe/hfcheck.log
#   golden        the golden sets on the bf16 original, every Task 5 prompt (lfm2moe_short / 4k / agentic / 32k,
#                 --gen 32) -> <data>/oracle-out-lfm2moe/<p>.{ids,golden.safetensors,log}; each log's gap lines
#                 (the 4th / 5th sel per layer: B70_LFM_TIE_TOL's evidence)
#   export <dir>  the same prompts on an int4 export dequantised (24q's RTN / AutoRound, 24b's synthetics; mounted
#                 as ORACLE_SNAP): the engine-format reference every F3 gate reads -> <data>/oracle-out-lfm2moe-<name>/
#                 (<name> = the directory's base name)
#   a4            tools/toolcall/a4_ref.sh lfm2moe set + ref (36 scenarios, greedy, 2048 new ids, batched)
#                 -> <data>/oracle-out-lfm2moe-a4/ (HOURS; resumable per scenario)
# Golden runs are per prompt resumable (written into <tree>/oracle-out-lfm2moe*.partial/, moved when complete, the
# golden file last). Exit 77 (SKIP) when the memory is short (box: MemAvailable < LFM_REF_MIN_GB - 8 tests, 64 real;
# Mac: the container cap LFM_MEM_GB, default 40, below the floor - 8 tests, 36 real) or the snapshot is absent or
# still downloading (*.incomplete blobs). DRY_RUN=1 prints every command, the floor and the ESTIMATED time and peak
# RSS per prompt, runs nothing.
# The Mac: ORACLE_IMAGE agnes-ref-img:latest, every container capped at LFM_MEM_GB (run_in_container.sh's
# ORACLE_MEMORY), and the whole run holds flock(2) on LFM_LOCK (default /tmp/b70-cpu-model.lock: one model job on
# the Mac's CPU at a time, across agents) - it waits for the lock after the SKIP checks.
# Env: SNAP_LFM2MOE (default models--LiquidAI--LFM2.5-24B-A2B), LFM_REF_MIN_GB, LFM_MEM_GB, LFM_GEN (32),
#      LFM_PROMPTS (lfm2moe_short lfm2moe_4k lfm2moe_agentic lfm2moe_32k), LFM_LOCK, LFM_LOCK_HELD (1: the
#      caller already holds LFM_LOCK - do not take it again), HF_HOME, ORACLE_IMAGE,
#      ORACLE_THREADS.
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 2 ] || { echo "usage: $0 <data tree> tests|ppl|hfcheck|golden|export <dir>|a4" >&2; exit 2; }
data="$1" mode="$2" arg="${3:-}"
case "$mode" in tests|ppl|hfcheck|golden|a4) ;;
  export) [ -n "$arg" ] || { echo "lfm2moe_oracle: export needs the export's directory" >&2; exit 2; } ;;
  *) echo "lfm2moe_oracle: mode $mode is not one of tests|ppl|hfcheck|golden|export <dir>|a4" >&2; exit 2 ;;
esac
mac=0
[ "$(uname -s)" = Darwin ] && mac=1
dry="${DRY_RUN:-0}"

hf="${HF_HOME:-$HOME/.cache/huggingface}"
gen="${LFM_GEN:-32}"
[[ "$gen" =~ ^[0-9]+$ ]] || { echo "lfm2moe_oracle: LFM_GEN=$gen" >&2; exit 2; }
prompts="${LFM_PROMPTS:-lfm2moe_short lfm2moe_4k lfm2moe_agentic lfm2moe_32k}"
TINY=models--yujiepan--lfm2-moe-tiny-random
REAL="${SNAP_LFM2MOE:-models--LiquidAI--LFM2.5-24B-A2B}"
if [ $mac = 1 ]; then
  SITE=oracle-out-q4exp/site
  export ORACLE_IMAGE="${ORACLE_IMAGE:-agnes-ref-img:latest}"
  export ORACLE_MEMORY="${LFM_MEM_GB:-40}g"
  export ORACLE_THREADS="${ORACLE_THREADS:-12}"
else
  SITE=oracle-out-q4exp-site
fi
case "$mode" in tests) need="${LFM_REF_MIN_GB:-8}"; model="$TINY" ;; *) model="$REAL" ;; esac
if [ "$mode" != tests ]; then
  if [ $mac = 1 ]; then need="${LFM_REF_MIN_GB:-36}"; else need="${LFM_REF_MIN_GB:-64}"; fi
fi
snapdir=$(ls -d "$hf/hub/$model"/snapshots/*/ 2>/dev/null | head -1)
if [ "$mode" = export ]; then   # the export's own directory is the checkpoint (the bf16 original is not read)
  model="$arg"
  snapdir=""
  [ -f "$arg/config.json" ] && snapdir="$arg"
fi
if [ $mac = 1 ]; then
  avail="${LFM_MEM_GB:-40}"
  memwhat="the container cap LFM_MEM_GB"
else
  avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
  memwhat="MemAvailable"
fi
echo "lfm2moe_oracle: $mode${arg:+ $arg} -> $data; checkpoint $model (${snapdir:-absent}); $memwhat ${avail} GB (need $need)"

if [ "$dry" = 1 ]; then
  echo "DRY_RUN: RAM floor $need GB ($memwhat now ${avail} GB); site $SITE (qwen4exp_env.sh)$([ $mac = 1 ] && echo "; Mac: image $ORACLE_IMAGE, --memory $ORACLE_MEMORY, $ORACLE_THREADS threads, the lock ${LFM_LOCK:-/tmp/b70-cpu-model.lock}")"
  echo "DRY_RUN: MEASURED on the Mac (2026-10-15: agnes-ref-img, 12 threads, --memory 40g, the bf16 original"
  echo "  layer-streamed; ru_maxrss counts the mmap'd shard pages, so 'peak RSS' sits at the cap - the anon part is"
  echo "  smaller) - the box (22 Haswell cores) is ESTIMATED at ~1-2x these:"
  echo "    lfm2moe_short (100 ids) 881 s (prompt 658 s, 32 steps 204 s)   hfcheck --layers 6 (2048 ids + 4 steps) 390 s"
  echo "    lfm2moe_4k 2923 s (prompt 2496 s whole, 32 steps 395 s)         ppl (46 + 55 + 4096 ids) 2905 s"
  echo "    lfm2moe_agentic (12474 ids, 7 chunks of 2048): ~2.5-3.5 h ESTIMATED from the 4k prompt rate"
  echo "    lfm2moe_32k (16 chunks of 2048, eager scores up to [32][2048][32768] fp32 ~8.6 GB a layer call):"
  echo "      ~6-8 h ESTIMATED - above the Mac's 6 h rule: a box run"
  echo "    a4: 36 scenarios x up to 2048 new ids (reasoning first; ~10 s a decode step measured on the Mac at"
  echo "      100 ids) ~1-3 days ESTIMATED even batched - a box run; tests (tiny models) ~15-25 min measured"
fi
if [ "$dry" != 1 ] && [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON lfm2moe_oracle: $memwhat ${avail} GB < $need GB (another job holds the RAM)"
  exit 77
fi
if [ "$dry" != 1 ] && [ -z "$snapdir" ]; then
  case "$mode" in
    tests) echo "SKIP_REASON lfm2moe_oracle: $TINY is not in $hf/hub (hf download yujiepan/lfm2-moe-tiny-random - 10 MB)" ;;
    export) echo "SKIP_REASON lfm2moe_oracle: $arg has no config.json (24q's export, or 24b's synthetic checkpoint)" ;;
    *) echo "SKIP_REASON lfm2moe_oracle: $REAL is not in $hf/hub (hf download LiquidAI/LFM2.5-24B-A2B: 47.96 GB, df -h ~ first)" ;;
  esac
  exit 77
fi
if [ "$dry" != 1 ] && [ "$mode" != tests ] && [ "$mode" != export ] && ls "$hf/hub/$model/blobs/"*.incomplete > /dev/null 2>&1; then
  echo "SKIP_REASON lfm2moe_oracle: the download of $model is not complete ($hf/hub/$model/blobs/*.incomplete)"
  exit 77
fi
# The Mac: one model job at a time on its CPU (the operator's rule) - re-run this script under flock(2), AFTER the
# SKIP checks above (a run that would SKIP never waits for the lock). A caller that already holds the lock sets
# LFM_LOCK_HELD=1 (flock(2) on a second open of the file would wait for itself).
if [ $mac = 1 ] && [ "$dry" != 1 ] && [ -z "${LFM_LOCK_HELD:-}" ]; then
  lock="${LFM_LOCK:-/tmp/b70-cpu-model.lock}"
  echo "lfm2moe_oracle: waiting for $lock ($(date '+%F %T'))"
  exec env LFM_LOCK_HELD=1 python3 -c 'import fcntl, subprocess, sys
f = open(sys.argv[1], "a")
fcntl.flock(f, fcntl.LOCK_EX)
print("lfm2moe_oracle: holding the lock", flush=True)
sys.exit(subprocess.call(sys.argv[2:]))' "$lock" "$PWD/tools/box_validate/lfm2moe_oracle.sh" "$@"
fi
rc=0

# in_box <command> - one oracle-image run with the 5.19.0 site first on PYTHONPATH ($SNAP = the checkpoint).
in_box() {
  local cmd="tools/oracle/qwen4exp_env.sh /ws/$SITE > /dev/null && export PYTHONPATH=/ws/$SITE && $1"
  if [ -n "${ORACLE_SNAP:-}" ]; then
    if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_SNAP=$ORACLE_SNAP tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
    tools/oracle/run_in_container.sh "$cmd"
    return
  fi
  if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_MODEL=$model tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
  ORACLE_MODEL="$model" tools/oracle/run_in_container.sh "$cmd"
}

# golden <out dir> <partial dir> <prompt name> <extra run args>
golden() {
  local out="$1" part="$2" p="$3" extra="$4"
  if [ -s "$out/$p.golden.safetensors" ] && [ -s "$out/$p.ids" ]; then
    echo "lfm2moe_oracle: $p kept ($out/$p.golden.safetensors)"
    return 0
  fi
  [ "$dry" = 1 ] || mkdir -p "$out" "$part"
  echo "lfm2moe_oracle: $p - lfm2moe_ref.py run --gen $gen $extra ($(date '+%F %T'))"
  if ! in_box "cp tests/golden/prompts/$p.ids /ws/$part/$p.ids && python3 tools/oracle/lfm2moe_ref.py run \"\$SNAP\" \
        --prompt /ws/$part/$p.ids --out /ws/$part/$p.golden.safetensors --gen $gen $extra > /ws/$part/$p.log 2>&1"; then
    echo "lfm2moe_oracle: $p FAILED"
    tail -20 "$part/$p.log" 2>/dev/null
    rc=1
    return 1
  fi
  [ "$dry" = 1 ] && return 0
  for f in "$p.ids" "$p.log" "$p.golden.safetensors"; do   # the golden file last: it marks done
    mv -f "$part/$f" "$out/$f"
  done
  echo "lfm2moe_oracle: $p done ($(date '+%F %T')); $(grep -E '^wall' "$out/$p.log")"
}

gaps() {   # the routing gap distributions: B70_LFM_TIE_TOL's evidence (spec 24 F3)
  echo "--- the 4th / 5th sel gap per layer (B70_LFM_TIE_TOL for 24c / 24d is set from these), walls and RSS"
  # shellcheck disable=SC2086 # the glob is meant
  [ "$dry" = 1 ] || grep -H -E 'sel gap|^wall|^prompt forward|^greedy' $1 2>/dev/null
}

case "$mode" in
  tests)
    in_box "python3 tools/oracle/test_lfm2moe_make_tiny.py && python3 tools/oracle/test_lfm2moe_ref.py && \
      python3 tools/toolcall/test_score.py && python3 tools/toolcall/test_make_set.py && \
      python3 tools/toolcall/test_oracle_batched.py test_lfm2moe" || rc=1
    ;;
  ppl)
    [ "$dry" = 1 ] || mkdir -p "$data/oracle-out-lfm2moe"
    in_box "python3 tools/oracle/lfm2moe_ref.py ppl \"\$SNAP\" --text prose=tests/golden/prompts/prose.txt \
      --text code=tests/golden/prompts/code.txt --ids agentic=tests/golden/prompts/lfm2moe_agentic.ids \
      > /ws/oracle-out-lfm2moe.ppl.log 2>&1" || rc=1
    if [ "$dry" != 1 ]; then
      mv -f oracle-out-lfm2moe.ppl.log "$data/oracle-out-lfm2moe/ppl.log"
      cat "$data/oracle-out-lfm2moe/ppl.log"
      awk '/^ppl / { n++; if ($3 ~ /nan|inf/ || /NOT SANE/) { print "lfm2moe_oracle: " $2 " perplexity " $3 " is not finite"; bad = 1 } }
           /^ppl prose:/ { if ($3 + 0 > 20) { print "lfm2moe_oracle: prose perplexity " $3 " is not sane (ceiling 20, ESTIMATED)"; bad = 1 } }
           END { if (n < 3) { print "lfm2moe_oracle: " n + 0 " of 3 perplexity lines"; bad = 1 }; exit bad }' \
        "$data/oracle-out-lfm2moe/ppl.log" || rc=1
    fi
    ;;
  hfcheck)
    [ "$dry" = 1 ] || mkdir -p "$data/oracle-out-lfm2moe"
    in_box "python3 tools/oracle/lfm2moe_ref.py hfcheck \"\$SNAP\" --layers 6 --prompt tests/golden/prompts/lfm2moe_4k.ids \
      --n 2048 --gen 4 > /ws/oracle-out-lfm2moe.hfcheck.log 2>&1" || rc=1
    if [ "$dry" != 1 ]; then
      mv -f oracle-out-lfm2moe.hfcheck.log "$data/oracle-out-lfm2moe/hfcheck.log"
      tail -5 "$data/oracle-out-lfm2moe/hfcheck.log"
    fi
    ;;
  golden)
    for p in $prompts; do
      golden "$data/oracle-out-lfm2moe" oracle-out-lfm2moe.partial "$p" ""
    done
    gaps "$data/oracle-out-lfm2moe/*.log"
    ;;
  export)
    name=$(basename "${arg%/}")
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "lfm2moe_oracle: export directory name $name" >&2; exit 2; }
    export ORACLE_SNAP
    ORACLE_SNAP="$(cd "$arg" 2>/dev/null && pwd || echo "$arg")"
    for p in $prompts; do
      golden "$data/oracle-out-lfm2moe-$name" "oracle-out-lfm2moe-$name.partial" "$p" "--weights int4"
    done
    unset ORACLE_SNAP
    gaps "$data/oracle-out-lfm2moe-$name/*.log"
    ;;
  a4)
    out="$data/oracle-out-lfm2moe-a4"
    if [ "$dry" = 1 ]; then
      echo "DRY_RUN: OUT=$out tools/toolcall/a4_ref.sh lfm2moe set; OUT=$out tools/toolcall/a4_ref.sh lfm2moe ref"
    else
      if [ ! -s "$out/set/manifest.json" ]; then
        OUT="$out" FOREGROUND=1 tools/toolcall/a4_ref.sh lfm2moe set || rc=1
      fi
      # the Mac's container cap is the memory plan's; the box plans from MemAvailable (a4_ref.sh, no cap)
      if [ $mac = 1 ]; then a4mem="${LFM_MEM_GB:-40}g"; else a4mem="${MEM:-}"; fi
      [ "$rc" = 0 ] && { OUT="$out" FOREGROUND=1 MEM="$a4mem" tools/toolcall/a4_ref.sh lfm2moe ref || rc=1; }
      OUT="$out" tools/toolcall/a4_ref.sh lfm2moe status
    fi
    ;;
esac
exit "$rc"
