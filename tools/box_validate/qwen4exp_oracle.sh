#!/usr/bin/env bash
# Spec 21a: the Qwen3.8-Flash-Next (qwen4_exp) CPU reference runs, ON THE BOX CPU (kolibri_oracle.sh's shape).
#   tools/box_validate/qwen4exp_oracle.sh <data tree> tests|tiny|intel|intel-layers N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8|synth|mtp-fixture|mtp-accept|reap|curve|project|prune|prune-a4
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
#   synth          spec 21c: the golden sets of the two synthetic checkpoints (synth-ckpt's steps first when they
#                  are absent): qwen4exp_ref.py run on each for q4exp_short / 4k / agentic (--gen 32) with the
#                  engine-format PLE (their own ckpt-ple-int8, copied into the tree: the container mounts only it and
#                  the snapshot) -> <data>/oracle-out-q4exp-synth/{ours,intel}/<p>.{ids,golden.safetensors,log}
#                  (r32.oracle_synth; ESTIMATED ~0.5-1 h a set on the box CPU: 4 real-width layers, every expert)
#   trace          teacher-forced routing traces for spec 22 P0.6 / P0.8: the 36 A4 scenarios
#                  (tests/golden/toolcall/*.ids), q4exp_agentic, code, prose, and the opencode recording when
#                  OPENCODE_LOG is set -> <data>/oracle-out-q4exp-traces/<name>.routes.safetensors
#   mtp-fixture    spec 21e Task 3 (M1's data): tools/oracle/qwen4exp_mtp_fixture.py - 21a's MTP port on the engine-
#                  format head (bf16 experts RTN int4 g64), both norm forms, 8 samples of q4exp_short - on each synthetic
#                  checkpoint (their own ckpt-ple-int8) -> <data>/oracle-out-q4exp-mtp/{ours,intel}/mtp_{single,
#                  per_stream}.safetensors, and on Intel's at --layers 18 when it is there (and MemAvailable >= 64 GB)
#                  -> <data>/oracle-out-q4exp-mtp/intel-L18/ (r34.fixture; what qwen4exp_mtp_test's M1 reads)
#   mtp-accept     spec 21e Task 3 step 4 (decision 4): qwen4exp_mtp.py accept on Intel's checkpoint - teacher-forced
#                  greedy acceptance by depth, K <= 3, the selection reused and fresh, over 21a's golden continuations
#                  of q4exp_short and q4exp_agentic (<data>/oracle-out-q4exp, `intel` first) and the A4 set's first
#                  four scenarios (their own 64-id continuations), per norm form ->
#                  <data>/oracle-out-q4exp-mtp/accept_{single,per_stream}.json (r34.accept; hours)
#   reap           spec 22 P0.8 (plan 22a): offload_curve.py reap over <data>/oracle-out-q4exp-traces (no checkpoint):
#                  REAP and freq masks at PRUNE_KEPT (0.5,0.625,0.75,0.875) from PRUNE_SOURCES (A4,agentic,opencode) -
#                  in sample, --holdout-a4, and decision 9's --kfold 4 --fold all -> <data>/oracle-out-q4exp-prune/masks/
#                  (+ reap.json, folds.json) (r36.reap; minutes)
#   curve          spec 22 P0.6: offload_curve.py counts / curve / swap / mtp / prefill / profile over the traces (no
#                  checkpoint) -> <data>/oracle-out-q4exp-offload/*.{txt,json}, the profile -> oracle-out-q4exp-traces/
#                  profile.safetensors (r36.curve; minutes to an hour: swap at S 1 is a step-by-step simulation)
#   project        spec 22 P0.7: offload_curve.py prune-report (with r36.prune's maskevals) and project over Task 1's
#                  budget, r36's po-*.txt lines and the curves -> oracle-out-q4exp-offload/{prune,project}.{txt,json},
#                  the last line STOP / GO-CANDIDATE (r36.project)
#   prune          spec 22 P0.8: qwen4exp_ref.py maskeval for every REAP mask (in sample, -noa4) over q4exp_short, 4k,
#                  agentic (+ q4exp_32k with PRUNE_32K=1; r30.golden's sets) and per A4 scenario its in-sample masks and
#                  its own fold's, teacher-forced over the full reference's continuation (r34.a4_ref), the full rows
#                  cached once -> <data>/oracle-out-q4exp-prune/maskeval/<p>.<tag>.json (r36.prune; HOURS, resumable)
#   prune-a4       spec 22 P0.8: tools/toolcall/a4_ref.sh qwen4exp ref with MASK= every REAP mask whose experts fit
#                  two cards whole (Task 1's budget at 32k with the head: <data>/oracle-out-q4exp-offload/budget.json -
#                  r36.host - or PRUNE_BUDGET) - the all-traces mask on the 36 scenarios and each fold's mask on its 9 - then score.py
#                  -> <data>/oracle-out-q4exp-a4/<name>.mask-<tag>.{ids,txt}, oracle-out-q4exp-prune/a4_score.txt
#                  (r36.prune_a4; DAYS at 3 fractions x 72 generations, ESTIMATED; resumable per scenario)
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
[ $# -ge 2 ] || { echo "usage: $0 <data tree> tests|tiny|intel|intel-layers N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8|synth|mtp-fixture|mtp-accept|reap|curve|project|prune|prune-a4" >&2; exit 2; }
data="$1" mode="$2" nl="${3:-}"
case "$mode" in tests|tiny|intel|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8|synth|mtp-fixture|mtp-accept|reap|curve|project|prune|prune-a4) ;;
  intel-layers) [[ "$nl" =~ ^[0-9]+$ ]] && [ "$nl" -ge 4 ] || { echo "qwen4exp_oracle: intel-layers N needs N >= 4 (the first QSA layer: a cached run needs one)" >&2; exit 2; } ;;
  *) echo "qwen4exp_oracle: mode $mode is not one of tests|tiny|intel|intel-layers N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8|synth|mtp-fixture|mtp-accept|reap|curve|project|prune|prune-a4" >&2; exit 2 ;;
esac
hf="${HF_HOME:-$HOME/.cache/huggingface}"
dry="${DRY_RUN:-0}"
gen="${Q4_GEN:-32}"
TINY=models--qikp--tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next
INTEL="${SNAP_Q4EXP_INTEL:-models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound}"
OURS="${SNAP_Q4EXP_OURS:-models--urakozz--Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ}"
ORIG="${SNAP_Q4EXP_ORIG:-models--Qwen--Qwen3.8-Flash-Next}"   # spec 21b: its small files only (synth-ckpt)
SITE=oracle-out-q4exp-site
prune_sources="${PRUNE_SOURCES:-A4,agentic,opencode}"   # spec 22 §4b: coding / agentic traces only
prune_kept="${PRUNE_KEPT:-0.5,0.625,0.75,0.875}"
[[ "$prune_sources" =~ ^[A-Za-z0-9_,.-]+$ ]] && [[ "$prune_kept" =~ ^[0-9.,]+$ ]] || {
  echo "qwen4exp_oracle: PRUNE_SOURCES / PRUNE_KEPT ($prune_sources / $prune_kept)" >&2; exit 2; }
case "$mode" in tests|tiny|synth-ckpt|ple-int8) need="${Q4_REF_MIN_GB:-24}" ;; reap|curve|project) need="${Q4_REF_MIN_GB:-8}" ;; synth|mtp-fixture) need="${Q4_REF_MIN_GB:-32}" ;; *) need="${Q4_REF_MIN_GB:-64}" ;; esac
case "$mode" in tests|tiny) model="$TINY" ;; ours) model="$OURS" ;; synth-ckpt|synth) model="$ORIG" ;; *) model="$INTEL" ;; esac
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
  echo "    reap ~minutes (numpy over the traces) | prune: a golden prompt ~(its row-30 wall) x (1 + masks) - short /"
  echo "    4k / agentic ~75 min each x 9 forwards ~11 h; an A4 scenario (~2.7k ids teacher-forced, one forward) ~10 min x"
  echo "    (1 full + 8 masks) ~1.5 h, x 36 ~2.3 days | prune-a4: 72 generations a fitting fraction (36 + 4 x 9) at row"
  echo "    34's ~30 min each ~1.5 days a fraction, ~4.5 days at 50 / 62.5 / 75 % (ESTIMATED from row 30's walls)"
fi
if [ "$dry" != 1 ] && [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON qwen4exp_oracle: MemAvailable ${avail} GB < Q4_REF_MIN_GB $need (another job holds the RAM)"
  exit 77
fi
synth_have() {   # both synthetic checkpoints and their PLE files are there
  for f in ours intel; do
    [ -s "$data/oracle-out-q4exp-synth/$f/ckpt-ple-int8/model.safetensors.index.json" ] || return 1
  done
}
if { [ "$mode" = reap ] || [ "$mode" = curve ]; } && [ "$dry" != 1 ] && ! ls "$data/oracle-out-q4exp-traces"/*.routes.safetensors > /dev/null 2>&1; then
  echo "SKIP_REASON qwen4exp_oracle: no routing traces in $data/oracle-out-q4exp-traces (r30.traces)"
  exit 77
fi
# reap reads the traces only (no checkpoint): run_in_container is given the tree's config-only stand-in as ORACLE_SNAP
case "$mode" in reap|curve|project) snapdir="$PWD/tests/model/qwen4exp/" ;; esac
if [ -z "$snapdir" ] && [ "$dry" != 1 ] && ! { { [ "$mode" = synth ] || [ "$mode" = mtp-fixture ]; } && synth_have; }; then
  case "$mode" in
    mtp-fixture) echo "SKIP_REASON qwen4exp_oracle: neither the synthetic checkpoints (r31.synth) nor $INTEL are there" ;;
    synth) echo "SKIP_REASON qwen4exp_oracle: the synthetic checkpoints are absent and $ORIG is not in $hf/hub to make them (its small files only: hf download Qwen/Qwen3.8-Flash-Next config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja - a few MB)" ;;
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
  if [ "$mode" = reap ] || [ "$mode" = curve ] || [ "$mode" = project ]; then   # no checkpoint read: the config-only stand-in satisfies run_in_container's $SNAP
    if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_SNAP=${snapdir%/} tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
    ORACLE_SNAP="${snapdir%/}" tools/oracle/run_in_container.sh "$cmd"
    return
  fi
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
  synth)
    # spec 21c Task 1: the checkpoints first (synth-ckpt's steps, a process of their own) when absent, then per form
    # the golden prompts against its own snapshot (ORACLE_SNAP, mounted at /snap) and its own PLE file (copied into
    # the tree: ~3 MB at the synthetic table's base 1000; the container mounts only the tree and the snapshot).
    if ! synth_have; then
      echo "qwen4exp_oracle: the synthetic checkpoints are absent - synth-ckpt first"
      "$0" "$data" synth-ckpt || rc=1
    fi
    for f in ours intel; do
      out="$data/oracle-out-q4exp-synth/$f"; part="oracle-out-q4exp-synth.partial/$f-golden"
      if [ "$dry" != 1 ] && [ ! -s "$out/ckpt-ple-int8/model.safetensors.index.json" ]; then
        echo "qwen4exp_oracle: synthetic $f absent after synth-ckpt"; rc=1; continue
      fi
      if [ "$dry" != 1 ] && [ ! -s "$part/ple/model.safetensors.index.json" ]; then
        mkdir -p "$part" && cp -r "$out/ckpt-ple-int8" "$part/ple.partial" && mv "$part/ple.partial" "$part/ple"
      fi
      export ORACLE_SNAP
      ORACLE_SNAP="$(cd "$out/ckpt" 2>/dev/null && pwd || echo "$out/ckpt")"
      [ "$dry" = 1 ] && echo "DRY_RUN: ORACLE_SNAP=$ORACLE_SNAP (the $f form), PLE int8:/ws/$part/ple"
      ple="int8:/ws/$part/ple"
      for p in q4exp_short q4exp_4k q4exp_agentic; do
        golden "$out" "$part" "$p" ""
      done
      unset ORACLE_SNAP
    done
    gaps "$data/oracle-out-q4exp-synth/*/*.log"
    ;;
  mtp-fixture)
    # spec 21e: per synthetic form against its own snapshot (ORACLE_SNAP) and PLE file (copied into the tree, as
    # synth does); then Intel's at --layers 18. Written into the tree, the files moved when both are there.
    fixture() {   # fixture <out dir> <partial dir> <extra args>
      local out="$1" part="$2" extra="$3"
      if [ -s "$out/mtp_single.safetensors" ] && [ -s "$out/mtp_per_stream.safetensors" ]; then
        echo "qwen4exp_oracle: MTP fixture kept ($out)"; return 0
      fi
      [ "$dry" = 1 ] || mkdir -p "$out" "$part"
      echo "qwen4exp_oracle: MTP fixture -> $out ($extra) ($(date '+%F %T'))"
      if ! in_box "python3 tools/oracle/qwen4exp_mtp_fixture.py \"\$SNAP\" /ws/$part $extra > /ws/$part/fixture.log 2>&1"; then
        echo "qwen4exp_oracle: MTP fixture $out FAILED"; tail -20 "$part/fixture.log" 2>/dev/null; rc=1; return 1
      fi
      [ "$dry" = 1 ] && return 0
      for f in fixture.log mtp_per_stream.safetensors mtp_single.safetensors; do mv -f "$part/$f" "$out/$f"; done
      tail -3 "$out/fixture.log"
    }
    for f in ours intel; do
      src="$data/oracle-out-q4exp-synth/$f"; part="oracle-out-q4exp-mtp.partial/$f"
      if [ "$dry" != 1 ] && [ ! -s "$src/ckpt-ple-int8/model.safetensors.index.json" ]; then
        echo "qwen4exp_oracle: synthetic $f absent (r31.synth: qwen4exp_oracle.sh synth-ckpt) - its fixture skipped"; continue
      fi
      if [ "$dry" != 1 ] && [ ! -s "$part/ple/model.safetensors.index.json" ]; then
        mkdir -p "$part" && cp -r "$src/ckpt-ple-int8" "$part/ple.partial" && mv "$part/ple.partial" "$part/ple"
      fi
      export ORACLE_SNAP
      ORACLE_SNAP="$(cd "$src/ckpt" 2>/dev/null && pwd || echo "$src/ckpt")"
      fixture "$data/oracle-out-q4exp-mtp/$f" "$part" "--ple int8:/ws/$part/ple"
      unset ORACLE_SNAP
    done
    if [ -n "$snapdir" ] || [ "$dry" = 1 ]; then
      if [ "$dry" != 1 ] && [ "${avail:-0}" -lt "${Q4_REF_REAL_GB:-64}" ]; then
        echo "qwen4exp_oracle: Intel's --layers 18 fixture skipped: MemAvailable ${avail} GB < ${Q4_REF_REAL_GB:-64}"
      else
        fixture "$data/oracle-out-q4exp-mtp/intel-L18" oracle-out-q4exp-mtp.partial/intel-L18 "--layers 18 --ple $ple"
      fi
    else
      echo "qwen4exp_oracle: $INTEL absent - the intel-L18 fixture (qwen4exp_mtp_intel_test) skipped"
    fi
    ;;
  mtp-accept)
    # spec 21e decision 4: both norm forms; the golden continuations copied into the tree (the container mounts only it).
    part=oracle-out-q4exp-mtp.partial/accept
    [ "$dry" = 1 ] || mkdir -p "$part/golden" "$data/oracle-out-q4exp-mtp"
    for p in q4exp_short q4exp_agentic; do
      if [ -s "$data/oracle-out-q4exp/$p.golden.safetensors" ]; then
        [ "$dry" = 1 ] || cp -f "$data/oracle-out-q4exp/$p.golden.safetensors" "$part/golden/"
      else
        echo "qwen4exp_oracle: no $data/oracle-out-q4exp/$p.golden.safetensors (qwen4exp_oracle.sh intel) - accept runs without it"
      fi
    done
    srcs=""
    for f in $(ls tests/golden/toolcall/*.ids | head -4); do srcs="$srcs --source $(basename "$f" .ids):$f"; done
    for n in single per_stream; do
      if [ -s "$data/oracle-out-q4exp-mtp/accept_$n.json" ]; then echo "qwen4exp_oracle: accept_$n kept"; continue; fi
      echo "qwen4exp_oracle: acceptance, norm $n ($(date '+%F %T'))"
      if in_box "python3 tools/oracle/qwen4exp_mtp.py accept \"\$SNAP\" --out /ws/$part/accept_$n.json --norm $n \
            --select both --golden /ws/$part/golden $srcs --k 3 --ple $ple > /ws/$part/accept_$n.log 2>&1"; then
        if [ "$dry" != 1 ]; then
          mv -f "$part/accept_$n.log" "$part/accept_$n.json" "$data/oracle-out-q4exp-mtp/"
          grep -E '^pooled' "$data/oracle-out-q4exp-mtp/accept_$n.log"
        fi
      else
        echo "qwen4exp_oracle: acceptance $n FAILED"; tail -20 "$part/accept_$n.log" 2>/dev/null; rc=1
      fi
    done
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
  reap)
    # spec 22 P0.8: REAP / freq masks from the traces (offload_curve.py: numpy, no checkpoint read) - in sample, without
    # the A4 scenarios, and decision 9's 4 folds; the traces copied into the tree (the container mounts only it).
    tr="$data/oracle-out-q4exp-traces"
    part=oracle-out-q4exp-prune.partial
    [ "$dry" = 1 ] || { mkdir -p "$part/traces" "$part/masks" "$data/oracle-out-q4exp-prune/masks"
                        cp -f "$tr"/*.routes.safetensors "$part/traces/"; }
    for arm in "" "--holdout-a4" "--kfold 4 --fold all"; do
      in_box "python3 tools/oracle/offload_curve.py reap --traces /ws/$part/traces --sources $prune_sources --kept $prune_kept \
        --criterion reap,freq --out /ws/$part/masks $arm >> /ws/$part/reap.log 2>&1" || rc=1
    done
    if [ "$dry" != 1 ]; then
      cp -f "$part"/masks/* "$data/oracle-out-q4exp-prune/masks/"
      grep -E '^reap' "$part/reap.log" | tail -40
    fi
    ;;
  curve)
    # spec 22 P0.6: offload_curve.py over the traces (no checkpoint read) -> <data>/oracle-out-q4exp-offload/
    # {counts,curve,swap,mtp,prefill}.{txt,json} and the routing profile <data>/oracle-out-q4exp-traces/profile.safetensors
    # (decision 8's source). swap's copy time: r36.copy's one-expert copy on the copy queue (po-copy.txt), else spec 22 §1's
    # measured 12.2 GB/s on device 0 (0.214 ms, marked); its step time SWAP_STEP_MS (25: the stopping rule's 40 t/s);
    # swap / mtp at the two-card f of Task 1's budget at 32k (the head + prefill; CURVE_F overrides).
    tr="$data/oracle-out-q4exp-traces" off="$data/oracle-out-q4exp-offload" part=oracle-out-q4exp-offload.partial
    cms="" f="${CURVE_F:-}" sms="${SWAP_STEP_MS:-25}"
    [ -s "$off/po-copy.txt" ] && cms=$(awk '/^po copy / && / shape=expert / && / list=regular / && / ordinal=copy / { for (i = 1; i <= NF; ++i) if ($i ~ /^us=/) { sub("us=", "", $i); print $i / 1000; exit } }' "$off/po-copy.txt")
    [ -n "$cms" ] || { cms=0.214; echo "qwen4exp_oracle: curve: no measured expert copy (r36.copy) - swap uses 0.214 ms (spec 22 §1's 12.2 GB/s, MEASURED on device 0 2026-09-27, not this row's)"; }
    if [ -z "$f" ] && [ -s "$off/budget.json" ]; then
      f=$(python3 -c "import json,sys; b=[json.loads(l.split('budget-json ',1)[-1]) for l in open(sys.argv[1]) if '{' in l]; b=[x for x in b if x['devices']==2 and x['max_len']==32768 and x['mtp'] and x['prefill'] and x['dense']=='int4']; print(round(b[0]['f'], 4))" "$off/budget.json")
    fi
    f=${f:-0.79}
    [[ "$cms" =~ ^[0-9.]+$ ]] && [[ "$f" =~ ^[0-9.]+$ ]] && [[ "$sms" =~ ^[0-9.]+$ ]] || { echo "qwen4exp_oracle: curve: copy-ms $cms / f $f / step $sms" >&2; exit 2; }
    [ "$dry" = 1 ] || { mkdir -p "$part/traces" "$off"; cp -f "$tr"/*.routes.safetensors "$part/traces/"
                        [ -s "$off/budget.json" ] && cp -f "$off/budget.json" "$part/"; }
    oc="python3 /ws/tools/oracle/offload_curve.py"
    bud=""
    [ -s "$off/budget.json" ] && bud="--budget budget.json"
    in_box "cd /ws/$part && $oc counts --traces traces --json counts.json > counts.txt && \
      $oc curve --traces traces --split hash --f 0.30:1.00:0.01 $bud --json curve.json > curve.txt && \
      $oc swap --traces traces --f $f --S 1,4,16,64 --B 2.6112,26.112,261.12 --W 64,256,1024 --copy-ms $cms --step-ms $sms \
        --json swap.json > swap.txt && \
      $oc mtp --traces traces --K 1,2,3,4 --f $f --json mtp.json > mtp.txt && \
      $oc prefill --traces traces --chunk 2048,4096 --f 0.6:0.9:0.05 --json prefill.json > prefill.txt && \
      $oc profile --traces traces --out profile.safetensors --json profile.json > profile.txt" || rc=1
    if [ "$dry" != 1 ]; then
      for x in counts curve swap mtp prefill profile; do cp -f "$part/$x.txt" "$part/$x.json" "$off/" 2>/dev/null; done
      cp -f "$part/profile.safetensors" "$tr/profile.safetensors" 2>/dev/null
      tail -6 "$part/curve.txt"; tail -8 "$part/swap.txt"; cat "$part/mtp.txt" "$part/profile.txt"
    fi
    ;;
  project)
    # spec 22 P0.7: prune-report (when r36.prune's maskevals exist) and the projection -> <data>/oracle-out-q4exp-offload/
    # {prune,project}.{txt,json}; the last line STOP or GO-CANDIDATE. Inputs: Task 1's budget, r36's po-*.txt lines, the
    # curve / swap / mtp / prefill JSON (r36.curve), r34.accept's accept_single.json when present; PROJECT_BW (row 32's
    # measured rate, GB/s) and PROJECT_PF_MS (row 33's compute a chunk, 2048:X[,4096:Y]) when the box has them.
    off="$data/oracle-out-q4exp-offload" part=oracle-out-q4exp-offload.partial
    if [ "$dry" != 1 ]; then
      [ -s "$off/budget.json" ] && [ -s "$off/curve.json" ] || {
        echo "SKIP_REASON qwen4exp_oracle: project needs $off/budget.json (r36.host) and curve.json (r36.curve)"; exit 77; }
      mkdir -p "$part/maskeval" "$part/a4"
      cat "$off"/po-*.txt > "$part/p0.txt" 2>/dev/null || : > "$part/p0.txt"
      for x in budget curve swap mtp prefill; do [ -s "$off/$x.json" ] && cp -f "$off/$x.json" "$part/"; done
      [ -s "$data/oracle-out-q4exp-mtp/accept_single.json" ] && cp -f "$data/oracle-out-q4exp-mtp/accept_single.json" "$part/accept.json"
      cp -f "$data"/oracle-out-q4exp-prune/maskeval/*.json "$part/maskeval/" 2>/dev/null
      cp -f "$data"/oracle-out-q4exp-a4/*.txt "$part/a4/" 2>/dev/null
    fi
    opt=""
    for x in swap mtp prefill accept; do [ "$dry" = 1 ] || [ -s "$part/$x.json" ] && opt="$opt --$x $x.json"; done
    [ -n "${PROJECT_BW:-}" ] && [[ "$PROJECT_BW" =~ ^[0-9.]+$ ]] && opt="$opt --bw $PROJECT_BW"
    [ -n "${PROJECT_PF_MS:-}" ] && [[ "$PROJECT_PF_MS" =~ ^[0-9.:,]+$ ]] && opt="$opt --prefill-compute-ms $PROJECT_PF_MS"
    oc="python3 /ws/tools/oracle/offload_curve.py"
    if [ "$dry" = 1 ] || ls "$part"/maskeval/*.json > /dev/null 2>&1; then
      in_box "cd /ws/$part && $oc prune-report --dir maskeval --a4 a4 --budget budget.json --json prune.json > prune.txt" || rc=1
    fi
    in_box "cd /ws/$part && $oc project --budget budget.json --p0 p0.txt --curve curve.json$opt --json project.json > project.txt" || rc=1
    if [ "$dry" != 1 ]; then
      for x in prune project; do cp -f "$part/$x.txt" "$part/$x.json" "$off/" 2>/dev/null; done
      [ -s "$part/prune.txt" ] && cat "$part/prune.txt"
      cat "$part/project.txt"
    fi
    ;;
  prune)
    # spec 22 P0.8: maskeval for every REAP mask (in sample and without A4) over the golden prompts, and per A4
    # scenario the in-sample masks and its own fold's (decision 9: graded on a mask that never saw it), teacher-forced
    # over the full reference's continuation (r34.a4_ref), the full model's rows cached once a scenario. One
    # maskeval process a prompt / scenario (the model built once for all its masks); resumable per (prompt, mask).
    masks="$data/oracle-out-q4exp-prune/masks"
    part=oracle-out-q4exp-prune.partial
    if [ "$dry" != 1 ] && ! ls "$masks"/mask-reap-*.safetensors > /dev/null 2>&1; then
      echo "SKIP_REASON qwen4exp_oracle: no REAP masks in $masks (r36.reap: qwen4exp_oracle.sh <data> reap)"; exit 77
    fi
    [ "$dry" = 1 ] || { mkdir -p "$part/masks" "$part/golden" "$part/a4" "$part/eval"; cp -f "$masks"/* "$part/masks/"; }
    prompts="q4exp_short q4exp_4k q4exp_agentic"
    [ "${PRUNE_32K:-0}" = 1 ] && prompts="$prompts q4exp_32k"
    in_sample=$(cd "$part/masks" 2>/dev/null && ls mask-reap-*.safetensors 2>/dev/null | grep -v -- '-fold' | tr '\n' ' ')
    [ "$dry" = 1 ] && [ -z "$in_sample" ] && in_sample="mask-reap-0.5.safetensors mask-reap-0.5-noa4.safetensors (...)"
    for p in $prompts; do
      if [ ! -s "$data/oracle-out-q4exp/$p.golden.safetensors" ] && [ "$dry" != 1 ]; then
        echo "qwen4exp_oracle: prune: $p has no golden set ($data/oracle-out-q4exp: r30.golden) - skipped"; continue
      fi
      [ "$dry" = 1 ] || cp -f "$data/oracle-out-q4exp/$p.golden.safetensors" "$part/golden/"
      args=""
      for mfile in $in_sample; do
        t=${mfile%.safetensors}
        [ -s "$part/eval/$p.$t.json" ] || args="$args --expert-mask /ws/$part/masks/$mfile"
      done
      [ -n "$args" ] || { echo "qwen4exp_oracle: prune: $p kept (every mask evaluated)"; continue; }
      echo "qwen4exp_oracle: prune: $p - maskeval ($(date '+%F %T'))"
      in_box "python3 tools/oracle/qwen4exp_ref.py maskeval \"\$SNAP\" --ple $ple --golden /ws/$part/golden/$p.golden.safetensors \
        --prompt tests/golden/prompts/$p.ids $args --json-dir /ws/$part/eval >> /ws/$part/maskeval.log 2>&1" || rc=1
    done
    a4="$data/oracle-out-q4exp-a4"
    if [ ! -s "$a4/set/manifest.json" ] && [ "$dry" = 1 ]; then
      echo "DRY_RUN: per A4 scenario <n> of $a4/set (36; fold i from masks/folds.json): ORACLE_MODEL=$model" \
           "tools/oracle/run_in_container.sh '... qwen4exp_ref.py maskeval \"\$SNAP\" --ple $ple --prompt <n>.ids --cont" \
           "<n>.bf16.ids --full-cache <n>.full.safetensors --expert-mask mask-reap-<kept>.safetensors --expert-mask" \
           "mask-reap-<kept>-fold<i>.safetensors (each kept) --json-dir /ws/$part/eval'"
    elif [ ! -s "$a4/set/manifest.json" ]; then
      echo "qwen4exp_oracle: prune: no A4 reference in $a4 (r34.a4_ref) - the A4 maskevals skipped"
    else
      for f in "$a4"/set/*.ids; do
        n=$(basename "$f" .ids)
        [ -s "$a4/$n.bf16.ids" ] || { [ "$dry" = 1 ] || echo "qwen4exp_oracle: prune: $n has no full reference - skipped"; continue; }
        fold=""
        [ -s "$part/masks/folds.json" ] && fold=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['folds'].get(sys.argv[2], ''))" "$part/masks/folds.json" "$n")
        [ "$dry" != 1 ] || fold=${fold:-0}
        [ "$dry" = 1 ] || cp -f "$f" "$a4/$n.bf16.ids" "$part/a4/"
        args=""
        for mfile in $in_sample; do
          case "$mfile" in *-noa4.safetensors) continue ;; esac
          t=${mfile%.safetensors}
          [ -s "$part/eval/$n.$t.json" ] || args="$args --expert-mask /ws/$part/masks/$mfile"
          if [ -n "$fold" ]; then
            tf="$t-fold$fold"
            [ -s "$part/eval/$n.$tf.json" ] || args="$args --expert-mask /ws/$part/masks/$tf.safetensors"
          fi
        done
        [ -n "$args" ] || continue
        echo "qwen4exp_oracle: prune: A4 $n (fold ${fold:-none}) - maskeval ($(date '+%F %T'))"
        in_box "python3 tools/oracle/qwen4exp_ref.py maskeval \"\$SNAP\" --ple $ple --prompt /ws/$part/a4/$n.ids \
          --cont /ws/$part/a4/$n.bf16.ids --full-cache /ws/$part/a4/$n.full.safetensors $args --json-dir /ws/$part/eval \
          >> /ws/$part/maskeval.log 2>&1" || rc=1
      done
    fi
    if [ "$dry" != 1 ]; then
      mkdir -p "$data/oracle-out-q4exp-prune/maskeval"
      cp -f "$part"/eval/*.json "$data/oracle-out-q4exp-prune/maskeval/" 2>/dev/null
      grep -E '^maskeval ' "$part/maskeval.log" | tail -40
    fi
    ;;
  prune-a4)
    # spec 22 P0.8: the A4 set generated under every REAP mask whose experts fit two cards whole (Task 1's budget: the
    # kept experts' bytes <= the two-card cache at 32k with the MTP head and the prefill scratch) - in sample (the
    # all-traces mask, 36 scenarios) and 4-fold held out (each fold's 9 scenarios on its fold's mask) - then score.py
    # against the full reference (r34.a4_ref) -> <data>/oracle-out-q4exp-a4/<name>.mask-<tag>.{ids,txt,gap}.
    masks="$data/oracle-out-q4exp-prune/masks"
    a4="$data/oracle-out-q4exp-a4"
    budget="${PRUNE_BUDGET:-$data/oracle-out-q4exp-offload/budget.json}"
    if [ "$dry" != 1 ]; then
      ls "$masks"/mask-reap-*.safetensors > /dev/null 2>&1 || { echo "SKIP_REASON qwen4exp_oracle: no REAP masks in $masks (r36.reap)"; exit 77; }
      [ -s "$a4/set/manifest.json" ] || { echo "SKIP_REASON qwen4exp_oracle: no A4 set / reference in $a4 (r34.a4_ref)"; exit 77; }
      [ -s "$budget" ] || { echo "SKIP_REASON qwen4exp_oracle: no budget lines at $budget (r36.host: qwen4exp_plan_test --budget-json)"; exit 77; }
      fit=$(python3 - "$budget" "$prune_kept" << 'EOF'
import json, sys
lines = [json.loads(l.split("budget-json ", 1)[-1]) for l in open(sys.argv[1]) if "{" in l]
b = [x for x in lines if x["devices"] == 2 and x["max_len"] == 32768 and x["mtp"] and x["dense"] == "int4"]
b.sort(key=lambda x: not x["prefill"])
b = b[0]
print(" ".join(f"{k:g}" for k in map(float, sys.argv[2].split(","))
               if b["layers"] * round(b["experts"] * k) * b["expert_bytes"] <= b["cache_total"]))
EOF
)
    else
      fit="0.5 0.625 0.75"   # DRY_RUN: Task 1's derived two-card cache at 32k with the head + prefill, 50.76 GB
    fi
    echo "qwen4exp_oracle: prune-a4: kept fractions whose experts fit two cards whole at 32k: ${fit:-none}$([ "$dry" = 1 ] && echo ' (DRY_RUN: from the Mac-derived budget)')"
    tags=""
    for k in $fit; do
      for t in "mask-reap-$k" "mask-reap-$k-fold0" "mask-reap-$k-fold1" "mask-reap-$k-fold2" "mask-reap-$k-fold3"; do
        o=""
        case "$t" in *-fold*) i=${t##*-fold}
          if [ "$dry" != 1 ]; then
            o=$(python3 -c "import json,sys; f=json.load(open(sys.argv[1]))['folds']; print(','.join(sorted(n for n, v in f.items() if v == int(sys.argv[2]))))" "$masks/folds.json" "$i")
          else
            o="FOLD${i}_SCENARIOS"
          fi ;;
        esac
        tags="$tags $t"
        if [ "$dry" = 1 ]; then
          echo "DRY_RUN: OUT=$a4 MASK=$masks/$t.safetensors${o:+ ONLY=$o} tools/toolcall/a4_ref.sh qwen4exp ref"
          continue
        fi
        [ -f "$a4/DONE.$t" ] && { echo "qwen4exp_oracle: prune-a4: $t kept ($a4/DONE.$t)"; continue; }
        echo "qwen4exp_oracle: prune-a4: $t${o:+ on $o} ($(date '+%F %T'))"
        OUT="$a4" MASK="$masks/$t.safetensors" ONLY="$o" tools/toolcall/a4_ref.sh qwen4exp ref || rc=1
      done
    done
    if [ "$dry" = 1 ]; then
      echo "DRY_RUN: python3 tools/toolcall/score.py $a4 bf16$tags > $data/oracle-out-q4exp-prune/a4_score.txt"
    elif [ -z "$fit" ]; then
      echo "qwen4exp_oracle: prune-a4: no REAP fraction fits two cards whole at 32k - nothing generated"
    elif [ -n "$tags" ]; then
      mkdir -p "$data/oracle-out-q4exp-prune"
      python3 tools/toolcall/score.py "$a4" bf16 $tags > "$data/oracle-out-q4exp-prune/a4_score.txt" || rc=1
      tail -$(( $(echo "$tags" | wc -w) + 2 )) "$data/oracle-out-q4exp-prune/a4_score.txt"
    fi
    ;;
esac
exit "$rc"
