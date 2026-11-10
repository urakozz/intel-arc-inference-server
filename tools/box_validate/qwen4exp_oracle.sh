#!/usr/bin/env bash
# Spec 21a: the Qwen3.8-Flash-Next (qwen4_exp) CPU reference runs, ON THE BOX CPU (kolibri_oracle.sh's shape).
#   tools/box_validate/qwen4exp_oracle.sh <data tree> tests|tiny|intel|intel-layers N|intel-f32 N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8|synth|mtp-fixture|mtp-accept|curve|project
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
#   intel-f32 N    fp32 prefill judge, 4500 cycled ids + 32 greedy, 64g cap -> oracle-out-q4exp-f32-L<N>/
#   intel-f32-aligned N  paired fp32 companions replaying f32-judged engine routes from
#                  oracle-out-q4exp-align-L<N>.partial/engine.safetensors, same fixed 32-token continuation
#                  -> oracle-out-q4exp-f32-aligned-L<N>/ (64g cap; rejects remain gate failures)
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
#                  (r32.oracle_synth; ESTIMATED ~0.5-1 h a set on the box CPU: 4 real-width layers, every expert);
#                  a kept set without the gates' score tensors (qsa.score, route.top - since 2026-11-04) is moved to
#                  ~/b70/trash/<date>/ and made again
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
#   curve          spec 22 P0.6: offload_curve.py counts / curve / swap / mtp / prefill / profile over the traces (no
#                  checkpoint) -> <data>/oracle-out-q4exp-offload/*.{txt,json}, the profile -> oracle-out-q4exp-traces/
#                  profile.safetensors (r36.curve; minutes to an hour: swap at S 1 is a step-by-step simulation)
#   project        spec 22 P0.7: offload_curve.py project over Task 1's budget, r36's po-*.txt lines and the
#                  curves -> oracle-out-q4exp-offload/project.{txt,json}, the last line STOP / GO-CANDIDATE (r36.project)
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
[ $# -ge 2 ] || { echo "usage: $0 <data tree> tests|tiny|intel|intel-layers N|intel-f32 N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8|synth|mtp-fixture|mtp-accept|curve|project" >&2; exit 2; }
data="$1" mode="$2" nl="${3:-}"
case "$mode" in tests|tiny|intel|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8|synth|mtp-fixture|mtp-accept|curve|project) ;;
  intel-layers|intel-f32|intel-f32-aligned) [[ "$nl" =~ ^[0-9]+$ ]] && [ "$nl" -ge 4 ] || { echo "qwen4exp_oracle: $mode N needs N >= 4 (the first QSA layer: a cached run needs one)" >&2; exit 2; } ;;
  *) echo "qwen4exp_oracle: mode $mode is not one of tests|tiny|intel|intel-layers N|intel-f32 N|ours|ppl|hfcheck|trace|synth-ckpt|ple-int8|synth|mtp-fixture|mtp-accept|curve|project" >&2; exit 2 ;;
esac
hf="${HF_HOME:-$HOME/.cache/huggingface}"
dry="${DRY_RUN:-0}"
gen="${Q4_GEN:-32}"
TINY=models--qikp--tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next
INTEL="${SNAP_Q4EXP_INTEL:-models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound}"
OURS="${SNAP_Q4EXP_OURS:-models--urakozz--Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ}"
ORIG="${SNAP_Q4EXP_ORIG:-models--Qwen--Qwen3.8-Flash-Next}"   # spec 21b: its small files only (synth-ckpt)
SITE=oracle-out-q4exp-site
case "$mode" in tests|tiny|synth-ckpt|ple-int8) need="${Q4_REF_MIN_GB:-24}" ;; curve|project) need="${Q4_REF_MIN_GB:-8}" ;; synth|mtp-fixture) need="${Q4_REF_MIN_GB:-32}" ;; *) need="${Q4_REF_MIN_GB:-64}" ;; esac
case "$mode" in tests|tiny) model="$TINY" ;; ours) model="$OURS" ;; synth-ckpt|synth) model="$ORIG" ;; *) model="$INTEL" ;; esac
snapdir=$(ls -d "$hf/hub/$model"/snapshots/*/ 2>/dev/null | head -1)
if [ "$mode" = intel-f32 ] || [ "$mode" = intel-f32-aligned ]; then
  snapdir=""
  if [ -r "$hf/hub/$model/refs/main" ]; then
    snapdir="$hf/hub/$model/snapshots/$(cat "$hf/hub/$model/refs/main")"
    [ -f "$snapdir/config.json" ] || snapdir=""
  else
    for candidate in "$hf/hub/$model"/snapshots/*; do
      [ -f "$candidate/config.json" ] && { snapdir="$candidate"; break; }
    done
  fi
fi
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
if [ "$mode" = intel-f32 ] || [ "$mode" = intel-f32-aligned ]; then
  engine_ple="int8:/hf/hub/$model/snapshots/$(basename "${snapdir:-absent}")-ple-int8"
  if [ -z "$snapdir" ] || [ ! -s "${snapdir%/}-ple-int8/model.safetensors.index.json" ] || [ "$ple" != "$engine_ple" ]; then
    echo "qwen4exp_oracle: intel-f32 needs the checkpoint's sibling int8 PLE file (the engine's table)" >&2
    exit 2
  fi
  ple_dir=$(cd "${snapdir%/}-ple-int8" && pwd -P)
  ple=int8:/ple
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
synth_have() {   # both synthetic checkpoints and their PLE files are there
  for f in ours intel; do
    [ -s "$data/oracle-out-q4exp-synth/$f/ckpt-ple-int8/model.safetensors.index.json" ] || return 1
  done
}
if [ "$mode" = curve ] && [ "$dry" != 1 ] && ! ls "$data/oracle-out-q4exp-traces"/*.routes.safetensors > /dev/null 2>&1; then
  echo "SKIP_REASON qwen4exp_oracle: no routing traces in $data/oracle-out-q4exp-traces (r30.traces)"
  exit 77
fi
# curve / project read the traces and JSON only (no checkpoint): run_in_container is given the tree's config-only
# stand-in as ORACLE_SNAP
case "$mode" in curve|project) snapdir="$PWD/tests/model/qwen4exp/" ;; esac
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
  if [ "$mode" = intel-f32 ] || [ "$mode" = intel-f32-aligned ]; then
    if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_MEMORY=${ORACLE_MEMORY:-64g} ORACLE_SNAP=$snapdir ORACLE_PLE=$ple_dir tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
    ORACLE_SNAP="$snapdir" ORACLE_PLE="$ple_dir" tools/oracle/run_in_container.sh "$cmd"
    return
  fi
  if [ "$mode" = curve ] || [ "$mode" = project ]; then   # no checkpoint read: the config-only stand-in satisfies run_in_container's $SNAP
    if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_SNAP=${snapdir%/} tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
    ORACLE_SNAP="${snapdir%/}" tools/oracle/run_in_container.sh "$cmd"
    return
  fi
  if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_MODEL=$model tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
  ORACLE_MODEL="$model" tools/oracle/run_in_container.sh "$cmd"
}

# has_scores <golden file> - its header names the score judge's tensors (qsa.score.L* when it has QSA selection rows,
# route.top.p.L*): qwen4exp_ref.py run writes them since 2026-11-04; older sets judge by the gaps (golden_test)
has_scores() {
  python3 -c 'import json, struct, sys
f = open(sys.argv[1], "rb"); h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
sel = any(k.startswith("qsa.sel.") for k in h); sc = any(k.startswith("qsa.score.") for k in h)
sys.exit(0 if any(k.startswith("route.top.p.") for k in h) and (sc or not sel) else 1)' "$1"
}

# golden <out dir> <partial dir> <prompt name> <extra run args> [scores]
# `scores` (the synthetic sets: minutes a prompt): a kept set without the score judge's tensors is moved to
# ~/b70/trash/<date>/ (the operator removes it) and made again; the real-checkpoint sets (hours) are kept as they are.
golden() {
  local out="$1" part="$2" p="$3" extra="$4" scores="${5:-}"
  if [ -n "$scores" ] && [ "$dry" != 1 ] && [ -s "$out/$p.golden.safetensors" ] && ! has_scores "$out/$p.golden.safetensors"; then
    local trash="$HOME/b70/trash/$(date +%F)/qwen4exp-golden-noscores/$(basename "$(dirname "$out")")-$(basename "$out")"
    echo "qwen4exp_oracle: $p has no block / router scores (qsa.score, route.top) - the old set to $trash, made again"
    mkdir -p "$trash" && for f in "$p.golden.safetensors" "$p.log"; do
      [ -e "$out/$f" ] && mv -f "$out/$f" "$trash/$f"
    done
  fi
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
  intel-f32-aligned)
    export ORACLE_MEMORY=64g   # also overrides an inherited larger cap
    out="$data/oracle-out-q4exp-f32-aligned-L$nl" part="oracle-out-q4exp-align-L$nl.partial"
    if [ -s "$out/q4exp_4k.state.safetensors" ]; then
      echo "qwen4exp_oracle: aligned f32 L$nl kept"
    else
      if [ "$dry" != 1 ]; then
        [ -s "$part/engine.safetensors" ] || { echo "need qwen4exp_route_capture -> $part/engine.safetensors" >&2; exit 2; }
        mkdir -p "$out" "$part"
        cp -f "$data/oracle-out-q4exp-f32-L$nl/q4exp_4k.state.safetensors" "$part/original.safetensors" || exit 1
      fi
      in_box "python3 tools/oracle/qwen4exp_align.py split /ws/$part/engine.safetensors /ws/$part" || exit 1
      for path in prefill decode; do
        if [ -s "$part/$path.state.safetensors" ]; then continue; fi
        in_box "python3 tools/oracle/qwen4exp_ref.py run \"\$SNAP\" --dtype fp32 --lm-head int8 --layers $nl \
          --cycle 4500 --state-dump --gen 32 --chunk 2048 --act-tail 0 --logits-tail 1 --ple $ple \
          --prompt tests/golden/prompts/q4exp_4k.ids --engine-routes /ws/$part/$path.routes.safetensors \
          --out /ws/$part/$path.state.safetensors > /ws/$part/$path.log 2>&1" || exit 1
      done
      in_box "python3 tools/oracle/qwen4exp_align.py merge /ws/$part/original.safetensors /ws/$part \
        /ws/$part/paired.safetensors > /ws/$part/merge.log 2>&1" || exit 1
      if [ "$dry" != 1 ]; then
        cp -f "$part/merge.log" "$out/q4exp_4k.log" || exit 1
        mv -f "$part/paired.safetensors" "$out/q4exp_4k.state.safetensors" || exit 1
        cat "$out/q4exp_4k.log"
      fi
    fi
    ;;
  intel-f32)
    out="$data/oracle-out-q4exp-f32-L$nl" part="oracle-out-q4exp-f32-L$nl.partial"
    if [ -s "$out/q4exp_4k.state.safetensors" ]; then
      echo "qwen4exp_oracle: f32 L$nl kept ($out/q4exp_4k.state.safetensors)"
    else
      [ "$dry" = 1 ] || mkdir -p "$out" "$part"
      echo "qwen4exp_oracle: f32 L$nl, 4500 cycled ids, 32 greedy tokens, ORACLE_MEMORY=64g"
      if ORACLE_MEMORY=64g in_box "python3 tools/oracle/qwen4exp_ref.py run \"\$SNAP\" \
            --prompt tests/golden/prompts/q4exp_4k.ids --out /ws/$part/q4exp_4k.state.safetensors \
            --dtype fp32 --lm-head int8 --layers $nl --cycle 4500 --state-dump --chunk 2048 --gen 32 \
            --act-tail 0 --logits-tail 1 --ple $ple > /ws/$part/q4exp_4k.log 2>&1"; then
        if [ "$dry" != 1 ]; then
          mv -f "$part/q4exp_4k.log" "$out/q4exp_4k.log"
          mv -f "$part/q4exp_4k.state.safetensors" "$out/q4exp_4k.state.safetensors"
          tail -3 "$out/q4exp_4k.log"
        fi
      else
        echo "qwen4exp_oracle: f32 L$nl FAILED"
        tail -20 "$part/q4exp_4k.log" 2>/dev/null
        rc=1
      fi
    fi
    ;;
  tests)
    in_box "python3 tools/oracle/test_qwen4exp_routes.py && python3 tools/oracle/test_qwen4exp_ref.py && python3 tools/oracle/test_qwen4exp_mtp.py" || rc=1
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
    elif [ -s "${BV_DATA:-.}/oracle-out-q4exp-ple-int8/model.safetensors.index.json" ]; then
      echo "qwen4exp_oracle: the PLE int8 file is in ${BV_DATA:-.}/oracle-out-q4exp-ple-int8, not beside the snapshot: link or" \
           "move it to $dest (the engine reads <snapshot>-ple-int8/ only)" >&2
      exit 1
    else
      df -h . "$hf" "$(dirname "$dest")" 2>/dev/null | tail -3
      # The destination sits beside the snapshot - on the box a bind mount of the operator's USB disk, another
      # filesystem than the tree: the final mv is a copy there. Refuse up front when it cannot hold the file
      # (2026-11-03: the copy filled the USB disk and the stage still reported the file written).
      ple_need_gb=56
      dest_free_gb=$(df -P -k "$(dirname "$dest")" 2>/dev/null | awk 'NR == 2 { print int($4 / 1048576) }')
      if [ "$dry" != 1 ] && [ "${dest_free_gb:-0}" -lt "$ple_need_gb" ] \
         && [ ! -s oracle-out-q4exp-ple.partial/ple/model.safetensors.index.json ]; then
        echo "qwen4exp_oracle: ple-int8 REFUSED: $(dirname "$dest") has ${dest_free_gb:-?} GB free, the file needs $ple_need_gb"
        rc=1
      elif [ -s oracle-out-q4exp-ple.partial/ple/model.safetensors.index.json ] \
         || in_box "mkdir -p /ws/oracle-out-q4exp-ple.partial && python3 tools/quantize/qwen4exp/ple_int8.py \"\$SNAP\" \
            /ws/oracle-out-q4exp-ple.partial/ple --scale ${Q4_PLE_SCALE:-bf16} && python3 tools/quantize/qwen4exp/check.py \
            \"\$SNAP\" --ple /ws/oracle-out-q4exp-ple.partial/ple"; then
        echo "qwen4exp_oracle: ple_int8.py ACCEPTED in oracle-out-q4exp-ple.partial/ple ($(date '+%F %T'))"
        if [ "$dry" = 1 ]; then
          :
        elif [ -e "$dest" ]; then
          echo "qwen4exp_oracle: ple-int8 FAILED: $dest exists without its index (a partial copy) - remove it first; the file stays in oracle-out-q4exp-ple.partial/ple"
          rc=1
        elif [ "${dest_free_gb:-0}" -lt "$ple_need_gb" ]; then
          echo "qwen4exp_oracle: ple-int8 FAILED: $(dirname "$dest") has ${dest_free_gb:-?} GB free, the file needs $ple_need_gb - left in oracle-out-q4exp-ple.partial/ple"
          rc=1
        elif mv oracle-out-q4exp-ple.partial/ple "$dest"; then
          echo "qwen4exp_oracle: the PLE int8 file written ($dest, $(date '+%F %T'))"
        else
          echo "qwen4exp_oracle: ple-int8 FAILED: mv to $dest (a partial copy may be left there; the file stays in oracle-out-q4exp-ple.partial/ple)"
          rc=1
        fi
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
        golden "$out" "$part" "$p" "" scores
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
    # spec 22 P0.7: the projection -> <data>/oracle-out-q4exp-offload/project.{txt,json}; the last line STOP or
    # GO-CANDIDATE. Inputs: Task 1's budget, r36's po-*.txt lines, the curve / swap / mtp / prefill JSON (r36.curve),
    # r34.accept's accept_single.json when present; PROJECT_BW (row 32's measured rate, GB/s) and PROJECT_PF_MS (row
    # 33's compute a chunk, 2048:X[,4096:Y]) when the box has them.
    off="$data/oracle-out-q4exp-offload" part=oracle-out-q4exp-offload.partial
    if [ "$dry" != 1 ]; then
      [ -s "$off/budget.json" ] && [ -s "$off/curve.json" ] || {
        echo "SKIP_REASON qwen4exp_oracle: project needs $off/budget.json (r36.host) and curve.json (r36.curve)"; exit 77; }
      mkdir -p "$part"
      cat "$off"/po-*.txt > "$part/p0.txt" 2>/dev/null || : > "$part/p0.txt"
      for x in budget curve swap mtp prefill; do [ -s "$off/$x.json" ] && cp -f "$off/$x.json" "$part/"; done
      [ -s "$data/oracle-out-q4exp-mtp/accept_single.json" ] && cp -f "$data/oracle-out-q4exp-mtp/accept_single.json" "$part/accept.json"
    fi
    opt=""
    for x in swap mtp prefill accept; do [ "$dry" = 1 ] || [ -s "$part/$x.json" ] && opt="$opt --$x $x.json"; done
    [ -n "${PROJECT_BW:-}" ] && [[ "$PROJECT_BW" =~ ^[0-9.]+$ ]] && opt="$opt --bw $PROJECT_BW"
    [ -n "${PROJECT_PF_MS:-}" ] && [[ "$PROJECT_PF_MS" =~ ^[0-9.:,]+$ ]] && opt="$opt --prefill-compute-ms $PROJECT_PF_MS"
    oc="python3 /ws/tools/oracle/offload_curve.py"
    in_box "cd /ws/$part && $oc project --budget budget.json --p0 p0.txt --curve curve.json$opt --json project.json > project.txt" || rc=1
    if [ "$dry" != 1 ]; then
      cp -f "$part/project.txt" "$part/project.json" "$off/" 2>/dev/null
      cat "$part/project.txt"
    fi
    ;;
esac
exit "$rc"
