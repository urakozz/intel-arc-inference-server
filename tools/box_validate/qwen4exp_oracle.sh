#!/usr/bin/env bash
# Spec 21a: the Qwen3.8-Flash-Next (qwen4_exp) CPU reference runs, ON THE BOX CPU (kolibri_oracle.sh's shape).
#   tools/box_validate/qwen4exp_oracle.sh <data tree> tests|tiny|intel|intel-layers N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8
#
# Every mode runs tools/oracle/qwen4exp_ref.py in the oracle image (tools/oracle/run_in_container.sh) with
# transformers 5.19.0 from a site directory BESIDE the image (tools/oracle/qwen4exp_env.sh: pip --no-deps
# --target <tree>/oracle-out-q4exp-site, made once, ~60 MB; the image is never changed) first on PYTHONPATH.
#   tests          the Python tests (test_qwen4exp_ref.py, test_qwen4exp_mtp.py) on the tiny model (r30.host)
#   tiny           golden runs on the tiny model -> <data>/oracle-out-q4exp-tiny/ (q4exp_short, q4exp_4k)
#   intel          Intel's 181 GB checkpoint, every golden prompt (q4exp_short / 4k / 8k / 32k / agentic, --gen 32)
#                  -> <data>/oracle-out-q4exp/<p>.{ids,golden.safetensors,log}
#   intel-layers N the reference truncated to N layers (what 21c's --layers N gate reads) on short / 4k / agentic
#                  -> <data>/oracle-out-q4exp-L<N>/
#   ours           our AutoRound g64 checkpoint (21q) when it exists -> <data>/oracle-out-q4exp-ours/
#   ppl            perplexity on prose.txt, code.txt and the agentic transcript (F1's third bullet: finite and
#                  below a sanity ceiling of 30 on prose, ESTIMATED) -> <data>/oracle-out-q4exp/ppl.log
#   hfcheck        transformers' own per-query indexer against the port's cache on real weights, --layers 4,
#                  2100 ids + 4 steps: BITWISE or exit 1 -> <data>/oracle-out-q4exp/hfcheck.log
#   synth-ckpt     spec 21b: the two synthetic real-width checkpoints (tools/quantize/qwen4exp/make_synth.py --layers 4
#                  --mtp, ours and Intel's form, from the ORIGINAL's small files - config.json and the tokenizer
#                  files of Qwen/Qwen3.8-Flash-Next, a few MB, not its weights), their PLE int8 files (ple_int8.py)
#                  and check.py -> <data>/oracle-out-q4exp-synth/{ours,intel}/{ckpt,ckpt-ple-int8} (r31.synth)
#   ple-int8       spec 21b: Intel's checkpoint's 128 bf16 PLE shards -> the int8 file the engine pins,
#                  <Intel snapshot>-ple-int8/ (ple_int8.py --scale ${Q4_PLE_SCALE:-bf16}; ~51.8 GB written; df -h first;
#                  written in the tree, then moved beside the snapshot) and check.py --ple (r31.ple_convert)
#   trace          teacher-forced routing traces for spec 22 P0.6 / P0.8: the 36 A4 scenarios
#                  (tests/golden/toolcall/*.ids), q4exp_agentic, code, prose, and the opencode recording when
#                  OPENCODE_LOG is set -> <data>/oracle-out-q4exp-traces/<name>.routes.safetensors
# Golden runs are per prompt resumable (written into <tree>/oracle-out-q4exp*.partial/, moved when complete, the
# golden file last); traces skip sources already written. Exit 77 (SKIP) when MemAvailable < Q4_REF_MIN_GB
# (24 tiny / tests, 64 real) or the snapshot is absent. DRY_RUN=1 prints the commands, the RAM floor and an
# ESTIMATED time per prompt, runs nothing.
# Env: SNAP_Q4EXP_INTEL (default models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound), SNAP_Q4EXP_OURS,
#      Q4_PLE (bf16 = the snapshot's own 128 shards, or int8:<dir>; default int8:<snapshot>-ple-int8 when 21b's
#      file exists - the engine-format reference, spec 21 F3 - else bf16), Q4_REF_MIN_GB, Q4_GEN (32),
#      OPENCODE_LOG, HF_HOME, ORACLE_IMAGE, ORACLE_THREADS.
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 2 ] || { echo "usage: $0 <data tree> tests|tiny|intel|intel-layers N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8" >&2; exit 2; }
data="$1" mode="$2" nl="${3:-}"
case "$mode" in tests|tiny|intel|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8) ;;
  intel-layers) [[ "$nl" =~ ^[0-9]+$ ]] && [ "$nl" -ge 4 ] || { echo "qwen4exp_oracle: intel-layers N needs N >= 4 (the first QSA layer: a cached run needs one)" >&2; exit 2; } ;;
  *) echo "qwen4exp_oracle: mode $mode is not one of tests|tiny|intel|intel-layers N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8" >&2; exit 2 ;;
esac
hf="${HF_HOME:-$HOME/.cache/huggingface}"
dry="${DRY_RUN:-0}"
gen="${Q4_GEN:-32}"
TINY=models--qikp--tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next
INTEL="${SNAP_Q4EXP_INTEL:-models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound}"
OURS="${SNAP_Q4EXP_OURS:-models--urakozz--Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ}"
ORIG="${SNAP_Q4EXP_ORIG:-models--Qwen--Qwen3.8-Flash-Next}"   # spec 21b: its small files only (synth-ckpt)
SITE=oracle-out-q4exp-site
case "$mode" in tests|tiny|synth-ckpt|ple-int8) need="${Q4_REF_MIN_GB:-24}" ;; *) need="${Q4_REF_MIN_GB:-64}" ;; esac
case "$mode" in tests|tiny) model="$TINY" ;; ours) model="$OURS" ;; synth-ckpt) model="$ORIG" ;; *) model="$INTEL" ;; esac
snapdir=$(ls -d "$hf/hub/$model"/snapshots/*/ 2>/dev/null | head -1)
avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
echo "qwen4exp_oracle: $mode${nl:+ $nl} -> $data; checkpoint $model (${snapdir:-absent}); MemAvailable ${avail} GB (need $need)"

# The PLE source (spec 21 F3: the engine-format reference reads 21b's int8 file).
ple="${Q4_PLE:-}"
if [ -z "$ple" ]; then
  if [ -n "$snapdir" ] && [ -f "${snapdir%/}-ple-int8/model.safetensors.index.json" ]; then
    ple="int8:/hf/hub/$model/snapshots/$(basename "$snapdir")-ple-int8"
  else
    ple=bf16
  fi
fi

if [ "$dry" = 1 ]; then
  echo "DRY_RUN: RAM floor Q4_REF_MIN_GB=$need (MemAvailable now ${avail} GB); PLE source $ple; site $SITE (qwen4exp_env.sh)"
  echo "DRY_RUN: ESTIMATED times on the box CPU (22 Haswell cores, bf16 matmuls without native support; Intel's"
  echo "  checkpoint layer-streamed: per forward each layer's bf16 dense arm (~0.15 GB) read and its routed experts"
  echo "  dequantised - a 2048-row chunk touches ~all 512 (1.34 GB int4 -> 2.5 GB bf16), a decode step 10):"
  echo "    q4exp_short (103 ids) ~10 min (32 decode steps dominate) | q4exp_4k ~20 min | q4exp_8k ~35 min"
  echo "    q4exp_32k ~1.5-2 h (16 chunks of 2048; 12 QSA layers' eager attention over up to 32k keys)"
  echo "    q4exp_agentic (~12k ids) ~45 min | intel total ~3-4 h | intel-layers 4 ~10 min, 18 ~40 min (3 prompts)"
  echo "    ppl ~30 min | hfcheck --layers 4 ~15 min | trace (~50k tokens, batch 8, no generation) ~1-1.5 h"
fi
if [ "$dry" != 1 ] && [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON qwen4exp_oracle: MemAvailable ${avail} GB < Q4_REF_MIN_GB $need (another job holds the RAM)"
  exit 77
fi
if [ -z "$snapdir" ] && [ "$dry" != 1 ]; then
  case "$mode" in
    tests|tiny) echo "SKIP_REASON qwen4exp_oracle: $TINY is not in $hf/hub (hf download qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next - 124 MB)" ;;
    synth-ckpt) echo "SKIP_REASON qwen4exp_oracle: $ORIG is not in $hf/hub (its small files only: hf download Qwen/Qwen3.8-Flash-Next config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja - a few MB)" ;;
    ours) echo "SKIP_REASON qwen4exp_oracle: $OURS is not in $hf/hub (spec 21q's checkpoint; decision 6)" ;;
    *) echo "SKIP_REASON qwen4exp_oracle: $INTEL is not in $hf/hub (hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound: 181 GB, df -h ~ first; the 128 PLE shards alone are 102.4 GB)" ;;
  esac
  exit 77
fi
rc=0

# in_box <command> - one oracle-image run with the 5.19.0 site first on PYTHONPATH ($SNAP = the checkpoint).
in_box() {
  local cmd="tools/oracle/qwen4exp_env.sh /ws/$SITE > /dev/null && export PYTHONPATH=/ws/$SITE && $1"
  if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_MODEL=$model tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
  ORACLE_MODEL="$model" tools/oracle/run_in_container.sh "$cmd"
}

# golden <out dir> <partial dir> <prompt name> <extra run args>
golden() {
  local out="$1" part="$2" p="$3" extra="$4"
  if [ -s "$out/$p.golden.safetensors" ] && [ -s "$out/$p.ids" ]; then
    echo "qwen4exp_oracle: $p kept ($out/$p.golden.safetensors)"
    return 0
  fi
  [ "$dry" = 1 ] || mkdir -p "$out" "$part"
  echo "qwen4exp_oracle: $p - qwen4exp_ref.py run --gen $gen $extra ($(date '+%F %T'))"
  if ! in_box "cp tests/golden/prompts/$p.ids /ws/$part/$p.ids && python3 tools/oracle/qwen4exp_ref.py run \"\$SNAP\" \
        --prompt /ws/$part/$p.ids --out /ws/$part/$p.golden.safetensors --gen $gen --ple $ple $extra \
        > /ws/$part/$p.log 2>&1"; then
    echo "qwen4exp_oracle: $p FAILED"
    tail -20 "$part/$p.log" 2>/dev/null
    rc=1
    return 1
  fi
  [ "$dry" = 1 ] && return 0
  for f in "$p.ids" "$p.log" "$p.golden.safetensors"; do   # the golden file last: it marks done
    mv -f "$part/$f" "$out/$f"
  done
  echo "qwen4exp_oracle: $p done ($(date '+%F %T'))"
}

gaps() {   # the gap distributions (decision 3's tau, R2's MoE tolerance)
  echo "--- the selection / routing gap distributions (decision 3: tau from the QSA 512th/513th gap; R2: the MoE 10th/11th)"
  # shellcheck disable=SC2086 # the glob is meant
  [ "$dry" = 1 ] || grep -H -E 'gap|^wall|^prompt forward|^greedy' $1 2>/dev/null
}

case "$mode" in
  tests)
    in_box "python3 tools/oracle/test_qwen4exp_ref.py && python3 tools/oracle/test_qwen4exp_mtp.py" || rc=1
    ;;
  synth-ckpt)
    # spec 21b Task 1: written in the tree (the container mounts only it), moved under <data> when checked; an
    # older incomplete set is moved aside (<f>.old-<epoch>), never deleted.
    for f in ours intel; do
      if [ -s "$data/oracle-out-q4exp-synth/$f/ckpt-ple-int8/model.safetensors.index.json" ]; then
        echo "qwen4exp_oracle: synthetic $f kept ($data/oracle-out-q4exp-synth/$f)"
        continue
      fi
      part=oracle-out-q4exp-synth.partial/$f
      echo "qwen4exp_oracle: synthetic $f - make_synth.py --layers 4 --mtp, ple_int8.py, check.py ($(date '+%F %T'))"
      if ! in_box "mkdir -p /ws/$part && python3 tools/quantize/qwen4exp/make_synth.py /ws/$part/ckpt --layers 4 --form $f \
            --mtp --tokenizer \"\$SNAP\" && python3 tools/quantize/qwen4exp/ple_int8.py /ws/$part/ckpt /ws/$part/ckpt-ple-int8 && \
            python3 tools/quantize/qwen4exp/check.py /ws/$part/ckpt --ple /ws/$part/ckpt-ple-int8"; then
        echo "qwen4exp_oracle: synthetic $f FAILED"
        rc=1
        continue
      fi
      [ "$dry" = 1 ] && continue
      mkdir -p "$data/oracle-out-q4exp-synth"
      [ -e "$data/oracle-out-q4exp-synth/$f" ] && mv "$data/oracle-out-q4exp-synth/$f" "$data/oracle-out-q4exp-synth/$f.old-$(date +%s)"
      mv "$part" "$data/oracle-out-q4exp-synth/$f"
      echo "qwen4exp_oracle: synthetic $f done ($(date '+%F %T'))"
    done
    ;;
  ple-int8)
    # spec 21b Task 1 on the real table: Intel's shards (the original's bf16 rows as shipped) -> <snapshot>-ple-int8/.
    dest="${snapdir%/}-ple-int8"
    if [ -s "$dest/model.safetensors.index.json" ]; then
      echo "qwen4exp_oracle: the PLE int8 file is there ($dest)"
    else
      df -h . "$hf" 2>/dev/null | tail -2
      echo "qwen4exp_oracle: ple_int8.py on $model -> $dest (~51.8 GB written, ESTIMATED ~0.5-1 h) ($(date '+%F %T'))"
      if in_box "mkdir -p /ws/oracle-out-q4exp-ple.partial && python3 tools/quantize/qwen4exp/ple_int8.py \"\$SNAP\" \
            /ws/oracle-out-q4exp-ple.partial/ple --scale ${Q4_PLE_SCALE:-bf16} && python3 tools/quantize/qwen4exp/check.py \
            \"\$SNAP\" --ple /ws/oracle-out-q4exp-ple.partial/ple"; then
        [ "$dry" = 1 ] || mv oracle-out-q4exp-ple.partial/ple "$dest"
        echo "qwen4exp_oracle: the PLE int8 file written ($dest, $(date '+%F %T'))"
      else
        echo "qwen4exp_oracle: ple_int8.py FAILED"
        rc=1
      fi
    fi
    ;;
  tiny)
    for p in q4exp_short q4exp_4k; do
      golden "$data/oracle-out-q4exp-tiny" oracle-out-q4exp-tiny.partial "$p" "--ple bf16"
    done
    gaps "$data/oracle-out-q4exp-tiny/*.log"
    ;;
  intel|ours)
    out="$data/oracle-out-q4exp"; part=oracle-out-q4exp.partial
    [ "$mode" = ours ] && { out="$data/oracle-out-q4exp-ours"; part=oracle-out-q4exp-ours.partial; }
    for p in q4exp_short q4exp_4k q4exp_8k q4exp_agentic q4exp_32k; do
      golden "$out" "$part" "$p" ""
    done
    gaps "$out/*.log"
    ;;
  intel-layers)
    for p in q4exp_short q4exp_4k q4exp_agentic; do
      golden "$data/oracle-out-q4exp-L$nl" "oracle-out-q4exp-L$nl.partial" "$p" "--layers $nl"
    done
    gaps "$data/oracle-out-q4exp-L$nl/*.log"
    ;;
  ppl)
    [ "$dry" = 1 ] || mkdir -p "$data/oracle-out-q4exp"
    in_box "python3 tools/oracle/qwen4exp_ref.py ppl \"\$SNAP\" --ple $ple --text prose=tests/golden/prompts/prose.txt \
      --text code=tests/golden/prompts/code.txt --ids agentic=tests/golden/prompts/q4exp_agentic.ids \
      > /ws/oracle-out-q4exp.ppl.log 2>&1" || rc=1
    if [ "$dry" != 1 ]; then
      mv -f oracle-out-q4exp.ppl.log "$data/oracle-out-q4exp/ppl.log"
      cat "$data/oracle-out-q4exp/ppl.log"
      awk '/^ppl prose:/ { if ($3 + 0 > 30 || $3 != $3 + 0) { print "qwen4exp_oracle: prose perplexity " $3 " is not sane (ceiling 30, ESTIMATED)"; bad = 1 } }
           END { exit bad }' "$data/oracle-out-q4exp/ppl.log" || rc=1
    fi
    ;;
  hfcheck)
    [ "$dry" = 1 ] || mkdir -p "$data/oracle-out-q4exp"
    in_box "python3 tools/oracle/qwen4exp_ref.py hfcheck \"\$SNAP\" --layers 4 --n 2100 --gen 4 --ple $ple \
      > /ws/oracle-out-q4exp.hfcheck.log 2>&1" || rc=1
    if [ "$dry" != 1 ]; then
      mv -f oracle-out-q4exp.hfcheck.log "$data/oracle-out-q4exp/hfcheck.log"
      tail -5 "$data/oracle-out-q4exp/hfcheck.log"
    fi
    ;;
  trace)
    srcs=""
    for f in tests/golden/toolcall/*.ids; do srcs="$srcs --source $(basename "$f" .ids):$f"; done
    srcs="$srcs --source q4exp_agentic:tests/golden/prompts/q4exp_agentic.ids --source code:tests/golden/prompts/code.ids"
    srcs="$srcs --source prose:tests/golden/prompts/prose.ids"
    oc=""
    if [ -n "${OPENCODE_LOG:-}" ]; then
      if [ -d "$OPENCODE_LOG" ]; then
        mkdir -p oracle-out-q4exp-opencode && cp -r "$OPENCODE_LOG"/. oracle-out-q4exp-opencode/
        oc="--opencode /ws/oracle-out-q4exp-opencode"
      else
        echo "qwen4exp_oracle: OPENCODE_LOG=$OPENCODE_LOG is not a directory - traced without it"
      fi
    fi
    [ "$dry" = 1 ] || mkdir -p "$data/oracle-out-q4exp-traces" oracle-out-q4exp-traces.partial
    in_box "python3 tools/oracle/qwen4exp_ref.py trace \"\$SNAP\" --ple $ple $srcs $oc --out /ws/oracle-out-q4exp-traces.partial \
      > /ws/oracle-out-q4exp-traces.partial/trace.log 2>&1" || rc=1
    if [ "$dry" != 1 ]; then
      for f in oracle-out-q4exp-traces.partial/*.routes.safetensors oracle-out-q4exp-traces.partial/trace.log; do
        [ -e "$f" ] && mv -f "$f" "$data/oracle-out-q4exp-traces/"
      done
      grep -E '^trace ' "$data/oracle-out-q4exp-traces/trace.log" | tail -45
    fi
    ;;
esac
exit "$rc"
