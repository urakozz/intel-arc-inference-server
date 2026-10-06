#!/usr/bin/env bash
# Spec 20c's Kolibri-1 golden sets, ON THE BOX CPU (k2_oracle.sh's shape): what kolibri_golden_test,
# kolibri_decode_test and kolibri_partial_test read.
#   tools/box_validate/kolibri_oracle.sh <data tree> synth|real
#
# synth  (no 20b needed) <data>/oracle-out-kolibri-synth/{int4attn,bf16attn}/: per arm ckpt/ (a
#        real-width 5-layer synthetic checkpoint, tools/quantize/kolibri/make_synth.py --layers 5,
#        ~6.2 GB each, made if absent), then for prose and de_prose: tokenize.py <ckpt> encode (no
#        --bos: Kolibri has none) -> <p>.ids, kolibri_ref.py run --gen 32 -> <p>.golden.safetensors
#        (+ <p>.log). The tokenizer comes from KOL_TOKENIZER_MODEL's snapshot in the HF cache (default
#        models--Aleph-Alpha--Kolibri-1-BF16: tokenizer.json and its config only - nothing is
#        downloaded here); SKIP 77 when it is not there. Also writes kolibri_bench.ids: the first 42
#        ids of de_prose.ids, the source of cli/kolibri_decode.h's kKolibriBenchPrompt (re-bake it
#        when it differs: plan 20c Task 5 Step 3).
# real   (needs 20b's checkpoint) <data>/oracle-out-kolibri/: prose, code, de_prose (tokenize.py) and
#        de_chat (kolibri_chat_ids.py over tests/golden/prompts/de_chat.json) on KOL_ORACLE_MODEL
#        (default models--urakozz--Kolibri-1-W4A16-g64-AutoRound-GPTQ), each -> <p>.ids,
#        <p>.golden.safetensors, <p>.log; SKIP 77 when the checkpoint is not in the HF cache.
#
# Per prompt resumable: written into <tree>/oracle-out-kolibri*.partial/ and moved into the data tree
# once complete (the golden file last: it marks done). Finally every log's `MoE 6th/7th selection gap`
# line - the evidence B70_KOL_TIE_TOL (kolibri_golden_test, 1e-2 proposed) is set from.
# Exit 77 (SKIP) when MemAvailable is below KOL_REF_MIN_GB (24 synth / 48 real; the box is shared).
# Env: KOL_ORACLE_MODEL, KOL_TOKENIZER_MODEL, KOL_REF_MIN_GB, HF_HOME, ORACLE_IMAGE, ORACLE_THREADS.
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -eq 2 ] || { echo "usage: $0 <data tree> synth|real" >&2; exit 2; }
data="$1" mode="$2"
case "$mode" in synth|real) ;; *) echo "kolibri_oracle: mode $mode is not synth or real" >&2; exit 2 ;; esac
hf="${HF_HOME:-$HOME/.cache/huggingface}"
if [ "$mode" = synth ]; then need="${KOL_REF_MIN_GB:-24}"; else need="${KOL_REF_MIN_GB:-48}"; fi
avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
echo "kolibri_oracle: $mode -> $data; MemAvailable ${avail} GB (need $need)"
if [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON kolibri_oracle: MemAvailable ${avail} GB < KOL_REF_MIN_GB $need (another job holds the RAM)"
  exit 77
fi
rc=0

# done_or_run <out dir> <partial dir> <prompt> <ORACLE_MODEL or -> <ORACLE_SNAP or -> <ids command>
# The ids command runs in the container with $SNAP the checkpoint and writes /ws/<partial>/<p>.ids.
done_or_run() {
  local out="$1" part="$2" p="$3" om="$4" os="$5" idcmd="$6"
  if [ -s "$out/$p.golden.safetensors" ] && [ -s "$out/$p.ids" ]; then
    echo "kolibri_oracle: $p kept ($out/$p.golden.safetensors)"
    return 0
  fi
  echo "kolibri_oracle: $p - ids, then kolibri_ref.py run --gen 32 ($(date '+%F %T'))"
  local envs=()
  [ "$om" != - ] && envs+=("ORACLE_MODEL=$om")
  [ "$os" != - ] && envs+=("ORACLE_SNAP=$os")
  if ! env "${envs[@]}" tools/oracle/run_in_container.sh \
      "$idcmd > /ws/$part/$p.ids && \
       python3 tools/oracle/kolibri_ref.py run \"\$SNAP\" --prompt /ws/$part/$p.ids \
         --out /ws/$part/$p.golden.safetensors --gen 32 > /ws/$part/$p.log 2>&1"; then
    echo "kolibri_oracle: $p FAILED"
    tail -20 "$part/$p.log" 2>/dev/null
    rc=1
    return 1
  fi
  for f in "$p.ids" "$p.log" "$p.golden.safetensors"; do   # the golden file last: it marks done
    mv -f "$part/$f" "$out/$f"
  done
  echo "kolibri_oracle: $p done ($(date '+%F %T'))"
}

if [ "$mode" = synth ]; then
  tokm="${KOL_TOKENIZER_MODEL:-models--Aleph-Alpha--Kolibri-1-BF16}"
  tokdir=$(ls -d "$hf/hub/$tokm"/snapshots/*/ 2>/dev/null | head -1)
  if [ -z "$tokdir" ] || [ ! -f "$tokdir/tokenizer.json" ]; then
    echo "SKIP_REASON kolibri_oracle: no tokenizer.json under $hf/hub/$tokm (hf download Aleph-Alpha/Kolibri-1-BF16 tokenizer.json tokenizer_config.json generation_config.json - a few MB)"
    exit 77
  fi
  for arm in int4attn bf16attn; do
    out="$data/oracle-out-kolibri-synth/$arm" part="oracle-out-kolibri-synth.partial/$arm"
    mkdir -p "$out" "$part"
    if [ ! -s "$out/ckpt/model.safetensors.index.json" ]; then
      echo "kolibri_oracle: $arm - make_synth.py --layers 5 --attn ${arm%attn} ($(date '+%F %T'))"
      if ! ORACLE_MODEL="$tokm" tools/oracle/run_in_container.sh \
          "python3 tools/quantize/kolibri/make_synth.py /ws/$part/ckpt --layers 5 --attn ${arm%attn} \
             --tokenizer \"\$SNAP\" && python3 tools/quantize/kolibri/check.py /ws/$part/ckpt --attn ${arm%attn}" \
          > "$part/make_synth.log" 2>&1; then
        echo "kolibri_oracle: $arm make_synth FAILED"; tail -20 "$part/make_synth.log"; rc=1; continue
      fi
      mv "$part/ckpt" "$out/ckpt"
    fi
    for p in prose de_prose; do
      done_or_run "$out" "$part" "$p" - "$(cd "$out/ckpt" && pwd)" \
        "python3 tools/oracle/tokenize.py \"\$SNAP\" encode tests/golden/prompts/$p.txt"
    done
  done
  ids="$data/oracle-out-kolibri-synth/int4attn/de_prose.ids"
  if [ -s "$ids" ]; then
    tr -s ' \n' '\n' < "$ids" | head -42 | tr '\n' ' ' > "$data/oracle-out-kolibri-synth/kolibri_bench.ids"
    echo >> "$data/oracle-out-kolibri-synth/kolibri_bench.ids"
    echo "kolibri_oracle: kolibri_bench.ids (the first 42 ids of de_prose; compare with tests/golden/prompts/kolibri_bench.ids)"
    diff -q <(tr -s ' \n' '\n' < "$data/oracle-out-kolibri-synth/kolibri_bench.ids") \
            <(tr -s ' \n' '\n' < tests/golden/prompts/kolibri_bench.ids) > /dev/null ||
      echo "kolibri_oracle: NOTE the committed kolibri_bench.ids differs - re-bake kKolibriBenchPrompt"
  fi
  logs="$data/oracle-out-kolibri-synth/*/*.log"
else
  model="${KOL_ORACLE_MODEL:-models--urakozz--Kolibri-1-W4A16-g64-AutoRound-GPTQ}"
  if ! ls -d "$hf/hub/$model"/snapshots/*/ > /dev/null 2>&1; then
    echo "SKIP_REASON kolibri_oracle: $model is not in $hf/hub (spec 20b's checkpoint, decision 1)"
    exit 77
  fi
  out="$data/oracle-out-kolibri" part="oracle-out-kolibri.partial"
  mkdir -p "$out" "$part"
  for p in prose code de_prose; do
    done_or_run "$out" "$part" "$p" "$model" - \
      "python3 tools/oracle/tokenize.py \"\$SNAP\" encode tests/golden/prompts/$p.txt"
  done
  done_or_run "$out" "$part" de_chat "$model" - \
    "python3 tools/oracle/kolibri_chat_ids.py \"\$SNAP\" tests/golden/prompts/de_chat.json"
  logs="$data/oracle-out-kolibri/*.log"
fi
echo "--- the selection-gap distribution (B70_KOL_TIE_TOL; 1e-2 proposed in kolibri_golden_test.cc)"
# shellcheck disable=SC2086 # the glob is meant
grep -H -E 'selection gap' $logs 2>/dev/null
exit "$rc"
