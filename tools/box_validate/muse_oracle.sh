#!/usr/bin/env bash
# Spec 23a: the Muse-Glimmer-30B CPU reference runs (kolibri_oracle.sh's / qwen4exp_oracle.sh's shape), on the BOX
# CPU - and on the Mac (Docker Desktop: ORACLE_IMAGE agnes-ref-img, the container capped by ORACLE_MEMORY).
#   tools/box_validate/muse_oracle.sh <data tree> tests|tiny|facts|ppl|hfcheck|golden|golden-bf16|a4|gateq|taps|synth-ckpt|synth-tests
#
# Every container mode runs in the oracle image (tools/oracle/run_in_container.sh) with transformers 5.19.0 from spec
# 21a's site BESIDE the image (tools/oracle/qwen4exp_env.sh; the box's oracle-out-q4exp-site, the Mac's
# oracle-out-q4exp/site - the same site qwen4exp uses, not a second one) first on PYTHONPATH.
#   tests        test_muse_ref.py, test_muse_gateq.py and the toolcall tests on the tiny model (r42.host)
#   tiny         golden runs on the tiny model (window 2048 crossed by muse_4k) -> <data>/oracle-out-muse-tiny/
#   facts        muse_facts.py on the host (small files, headers, the FULL qzeros / g_idx scan of the int4 shards:
#                ~210 MB read), then in the container the 5.19.0 citations, the one-message render and
#                muse_ref.py facts (the names) -> <data>/oracle-out-muse/facts.log (r42.facts)
#   ppl          muse_ref.py ppl on prose.txt, code.txt, muse_4k.ids and muse_agentic.ids for each form present
#                (GPTQ dequantised; the bf16 original) -> <data>/oracle-out-muse{,-bf16}/ppl.log (sanity: finite, prose
#                and 4k below 30 - ESTIMATED ceiling)
#   hfcheck      muse_ref.py hfcheck --layers 4 on the first 2100 ids of muse_4k for each form present: streamed =
#                resident = layer-major, BITWISE or exit 1 -> <data>/oracle-out-muse{,-bf16}/hfcheck.log
#   golden       the GPTQ form, every golden prompt (muse_short_prose / short_code / 4k / 8k / 32k / agentic),
#                --gen 32 --taps -> <data>/oracle-out-muse/<p>.{ids,golden.safetensors,log}
#   golden-bf16  the bf16 original on muse_short_prose and muse_agentic -> <data>/oracle-out-muse-bf16/
#   a4           tools/toolcall/a4_ref.sh muse set + ref (both forms, 512 new ids) -> <data>/oracle-out-muse-a4/
#                <name>.{gptq,bf16}.{ids,txt,gap} (resumable per scenario), then score.py --format atem
#   gateq        muse_gateq.py kl (teacher-forced over the bf16 run's A4 continuations, + prose / code) and verdict
#                (spec 23 §3.3's bars, decision 5: a RECORDED measurement - the checkpoint is kept, operator ruling
#                2026-10-15) -> <data>/oracle-out-muse-a4/{gateq.json,gateq.log}
#   synth-ckpt   spec 23b: tools/quantize/muse/make_synth.py - a real-width synthetic checkpoint of the first
#                MUSE_SYNTH_LAYERS (4) layers in the -GPTQ spelling (random weights, 23a's RTN packer; ~6.4 GB at 4) - then
#                tools/quantize/muse/check.py (names = muse_ref.expected_names, the data asserts, every linear within
#                RTN's bound of its regenerated twin) -> <data>/oracle-out-muse-synth/ckpt<N>/ + ckpt<N>.ACCEPTED (the
#                check's log; written last, it marks done). Needs only the original's and the -GPTQ repo's SMALL files.
#   synth-tests  spec 23b: tools/quantize/muse/test_muse_quant.py (make_synth / check.py / the packer; the same small files)
#   taps         for 23f P0: the residual after layers 1, 13, 25, 37, 49 over [prompt + the GPTQ form's own A4
#                continuation] per scenario (teacher-forced, no generation) -> <data>/oracle-out-muse-taps/
# Golden runs are per prompt resumable (written into <tree>/oracle-out-muse*.partial/, moved when complete, the golden
# file last). Exit 77 (SKIP) when the RAM is short (MUSE_REF_MIN_GB: 24 tests / tiny / facts, 40 the GPTQ form, 72 the
# bf16 one; the box: MemAvailable; the Mac: the ORACLE_MEMORY cap) or a snapshot is absent / incomplete (every shard
# of its index, no *.incomplete blob - never download here: df -h first, then hf download by hand).
# DRY_RUN=1 prints the commands, the RAM floor and an ESTIMATED time per prompt, runs nothing.
# Env: SNAP_MUSE (default: models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound when complete, else -AutoRound-GPTQ), SNAP_MUSE_BF16
#      (models--meta-models--Muse-Glimmer-30B), SNAP_MUSE_ASSIST (models--meta-models--Muse-Glimmer-30B-assistant),
#      MUSE_PROMPTS (golden's prompts, default all six), MUSE_REF_MIN_GB, MUSE_GEN (32), HF_HOME, ORACLE_IMAGE, ORACLE_THREADS, ORACLE_MEMORY (the Mac: 40g).
set -u
cd "$(dirname "$0")/../.." || exit 2
MODES="tests|tiny|facts|ppl|hfcheck|golden|golden-bf16|a4|gateq|taps|synth-ckpt|synth-tests"
[ $# -eq 2 ] || { echo "usage: $0 <data tree> $MODES" >&2; exit 2; }
data="$1" mode="$2"
case "|$MODES|" in *"|$mode|"*) ;; *) echo "muse_oracle: mode $mode is not one of $MODES" >&2; exit 2 ;; esac
hf="${HF_HOME:-$HOME/.cache/huggingface}"
dry="${DRY_RUN:-0}"
gen="${MUSE_GEN:-32}"
[[ "$gen" =~ ^[0-9]+$ ]] || { echo "muse_oracle: MUSE_GEN=$gen" >&2; exit 2; }
TINY=models--optimum-intel-internal-testing--tiny-random-muse-glimmer
GPTQ="${SNAP_MUSE:-}"      # the int4 checkpoint: set below (either Vishva007 spelling) unless given
BF16="${SNAP_MUSE_BF16:-models--meta-models--Muse-Glimmer-30B}"
ASSIST="${SNAP_MUSE_ASSIST:-models--meta-models--Muse-Glimmer-30B-assistant}"
PROMPTS="${MUSE_PROMPTS:-muse_short_prose muse_short_code muse_4k muse_8k muse_32k muse_agentic}"   # golden's set
for p in $PROMPTS; do
  [ -s "tests/golden/prompts/$p.ids" ] || { echo "muse_oracle: MUSE_PROMPTS names $p: no tests/golden/prompts/$p.ids" >&2; exit 2; }
done
mac=0
[ "$(uname -s)" = Darwin ] && mac=1
if [ $mac = 1 ]; then
  SITE=oracle-out-q4exp/site
  export ORACLE_IMAGE="${ORACLE_IMAGE:-agnes-ref-img:latest}" ORACLE_MEMORY="${ORACLE_MEMORY:-40g}"
  export ORACLE_THREADS="${ORACLE_THREADS:-12}"
else
  SITE=oracle-out-q4exp-site
fi
case "$mode" in
  tests|tiny|facts|synth-ckpt|synth-tests) need="${MUSE_REF_MIN_GB:-24}" ;;
  golden-bf16) need="${MUSE_REF_MIN_GB:-72}" ;;
  *) need="${MUSE_REF_MIN_GB:-40}" ;;
esac
if [ -r /proc/meminfo ]; then
  avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo)
else                      # the Mac: the container's cap is what the run gets
  avail="${ORACLE_MEMORY%g}"
fi
echo "muse_oracle: $mode -> $data; MemAvailable ${avail:-0} GB (need $need)$([ $mac = 1 ] && echo "; Mac: $ORACLE_IMAGE capped $ORACLE_MEMORY")"

snap_of() { ls -d "$hf/hub/$1"/snapshots/*/ 2>/dev/null | head -1; }
# complete <cache dir name>: config.json, every shard the index names, no *.incomplete blob (a download in flight)
complete() {
  local d; d=$(snap_of "$1")
  [ -n "$d" ] && [ -f "$d/config.json" ] || return 1
  ls "$hf/hub/$1/blobs/"*.incomplete > /dev/null 2>&1 && return 1
  python3 - "$d" <<'EOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
files = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
sys.exit(1 if [f for f in files if not os.path.exists(os.path.join(d, f))] else 0)
EOF
}
# The int4 checkpoint: the same export in two spellings (spec 23 §3.1) - the auto_round one (no g_idx; the Mac's
# download) when complete, else the GPTQ one (g_idx, checked == i // 64).
if [ -z "$GPTQ" ]; then
  GPTQ=models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound-GPTQ
  complete models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound && GPTQ=models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound
fi
echo "muse_oracle: the int4 checkpoint is $GPTQ"
forms=()     # the checkpoint forms a mode can run on
complete "$GPTQ" && forms+=(gptq)
complete "$BF16" && forms+=(bf16)
model_of() { if [ "$1" = bf16 ]; then echo "$BF16"; else echo "$GPTQ"; fi; }
outdir_of() { if [ "$1" = bf16 ]; then echo "$data/oracle-out-muse-bf16"; else echo "$data/oracle-out-muse"; fi; }

if [ "$dry" = 1 ]; then
  echo "DRY_RUN: RAM floor MUSE_REF_MIN_GB=$need; site $SITE (qwen4exp_env.sh); forms present: ${forms[*]:-none}"
  echo "DRY_RUN: ESTIMATED times on the box CPU (22 Haswell cores, bf16 matmuls without native support), layer-streamed:"
  echo "  a forward reads 25.2 B linear parameters - the GPTQ form dequantises all of them (int4 13.4 GB -> bf16 50.3 GB,"
  echo "  stream.dequant_t: ~3-5 s a pass) - and computes ~50 GFLOP a token in the linears + the 13 full layers' attention:"
  echo "    muse_short_prose / short_code (42 / 53 ids) ~5 min each (32 decode steps at ~5-10 s dominate)"
  echo "    muse_4k ~20 min | muse_8k ~40 min | muse_32k ~2.5-3 h (16 chunks of 2048) | muse_agentic (13.4k ids) ~1-1.5 h"
  echo "    golden total ~5-6 h | golden-bf16 (short + agentic) ~1.5-2 h (no dequant; 55.7 GB read a pass)"
  echo "    facts: the full scan ~210 MB of reads, minutes | ppl ~30-40 min a form | hfcheck --layers 4 ~10 min a form"
  echo "    a4: 36 scenarios x 2 forms x (a ~1.8k-id prompt ~5-8 min + up to 512 decode steps); batched (8, layer-major,"
  echo "      one dequant a pass for all) ~0.5-1 day a form | gateq: 36 teacher-forced forwards a form ~3-5 h | taps ~3-4 h"
  echo "  The Mac (agnes-ref-img, i9-9980HK, $ORACLE_MEMORY cap if set): ~2-3x the box's compute times; the bf16 form reads"
  echo "  55.7 GB a pass from disk once the page cache (64 GB RAM, shared) cannot hold it - its A4 ~2-4 days (ESTIMATED)."
fi
if [ "$dry" != 1 ] && [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON muse_oracle: ${avail:-0} GB available < MUSE_REF_MIN_GB $need (another job holds the RAM; the Mac: ORACLE_MEMORY)"
  exit 77
fi
case "$mode" in
  tests|tiny) complete "$TINY" || [ "$dry" = 1 ] || {
    echo "SKIP_REASON muse_oracle: $TINY is not in $hf/hub (hf download optimum-intel-internal-testing/tiny-random-muse-glimmer - 27 MB)"; exit 77; } ;;
  facts|golden|taps) complete "$GPTQ" || [ "$dry" = 1 ] || {
    echo "SKIP_REASON muse_oracle: $GPTQ is absent or incomplete in $hf/hub (Vishva007/Muse-Glimmer-30B-W4A16-AutoRound or -GPTQ, 22.8 GB; df -h first)"; exit 77; }
    if [ "$mode" = facts ] && [ "$dry" != 1 ]; then   # the original's small files (config, tokenizer, template)
      b=$(snap_of "$BF16")
      for f in config.json tokenizer.json chat_template.jinja model.safetensors.index.json; do
        [ -n "$b" ] && [ -s "$b/$f" ] || {
          echo "SKIP_REASON muse_oracle: facts needs the original's small files ($BF16: $f - hf download meta-models/Muse-Glimmer-30B config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja model.safetensors.index.json)"
          exit 77; }
      done
    fi ;;
  synth-ckpt|synth-tests) [ "$dry" = 1 ] || {
    b=$(snap_of "$BF16") g=$(snap_of models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound-GPTQ)
    for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
      [ -n "$b" ] && [ -s "$b/$f" ] || {
        echo "SKIP_REASON muse_oracle: synth-ckpt and synth-tests need the original's small files ($BF16: $f - hf download meta-models/Muse-Glimmer-30B config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json)"
        exit 77; }
    done
    [ -n "$g" ] && [ -s "$g/config.json" ] || {
      echo "SKIP_REASON muse_oracle: synth-ckpt needs the -GPTQ repo's config.json (hf download Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ config.json)"
      exit 77; }
  } ;;
  golden-bf16) complete "$BF16" || [ "$dry" = 1 ] || {
    echo "SKIP_REASON muse_oracle: $BF16 is absent or incomplete in $hf/hub (meta-models/Muse-Glimmer-30B, 59.6 GB; df -h first)"; exit 77; } ;;
  a4|gateq) { complete "$GPTQ" && complete "$BF16"; } || [ "$dry" = 1 ] || {
    echo "SKIP_REASON muse_oracle: gate Q needs both $GPTQ and $BF16 complete in $hf/hub (22.8 + 59.6 GB)"; exit 77; } ;;
  ppl|hfcheck) [ ${#forms[@]} -gt 0 ] || [ "$dry" = 1 ] || {
    echo "SKIP_REASON muse_oracle: neither $GPTQ nor $BF16 is complete in $hf/hub"; exit 77; } ;;
esac
[ ${#forms[@]} -gt 0 ] || forms=(gptq)      # DRY_RUN with nothing downloaded: show the GPTQ form's commands
rc=0

# in_box <cache dir name> <command> - one oracle-image run, the 5.19.0 site first on PYTHONPATH, $SNAP = the snapshot.
in_box() {
  local cmd="tools/oracle/qwen4exp_env.sh /ws/$SITE > /dev/null && export PYTHONPATH=/ws/$SITE && $2"
  if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_MODEL=$1 tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
  ORACLE_MODEL="$1" tools/oracle/run_in_container.sh "$cmd"
}

# golden <cache dir name> <out dir> <partial dir> <prompt name> <extra run args>
golden() {
  local model="$1" out="$2" part="$3" p="$4" extra="$5"
  if [ -s "$out/$p.golden.safetensors" ] && [ -s "$out/$p.ids" ]; then
    echo "muse_oracle: $p kept ($out/$p.golden.safetensors)"
    return 0
  fi
  [ "$dry" = 1 ] || mkdir -p "$out" "$part"
  echo "muse_oracle: $p - muse_ref.py run --gen $gen $extra ($(date '+%F %T'))"
  if ! in_box "$model" "cp tests/golden/prompts/$p.ids /ws/$part/$p.ids && python3 tools/oracle/muse_ref.py run \"\$SNAP\" \
        --prompt /ws/$part/$p.ids --out /ws/$part/$p.golden.safetensors --gen $gen $extra > /ws/$part/$p.log 2>&1"; then
    echo "muse_oracle: $p FAILED"
    tail -20 "$part/$p.log" 2>/dev/null
    rc=1
    return 1
  fi
  [ "$dry" = 1 ] && return 0
  for f in "$p.ids" "$p.log" "$p.golden.safetensors"; do   # the golden file last: it marks done
    mv -f "$part/$f" "$out/$f"
  done
  echo "muse_oracle: $p done ($(date '+%F %T'))"
}

summary() {   # the cap_tie counts and the raw top-1 / top-2 gap distribution of every log (MG3's tie rule)
  echo "--- cap_tie and the raw gap (spec 23 §1.1: rows whose capped top-2 tie are undetermined)"
  # shellcheck disable=SC2086 # the glob is meant
  [ "$dry" = 1 ] || grep -H -E '^cap_tie|top-1 - top-2 gap|^wall|^prompt forward|^greedy|ppl' $1 2>/dev/null
}

case "$mode" in
  tests)
    in_box "$TINY" "python3 tools/oracle/test_muse_ref.py && python3 tools/oracle/test_muse_gateq.py && \
      python3 tools/toolcall/test_make_set.py && python3 tools/toolcall/test_score.py" || rc=1
    ;;
  tiny)
    for p in muse_short_prose muse_4k; do
      golden "$TINY" "$data/oracle-out-muse-tiny" oracle-out-muse-tiny.partial "$p" "--taps" || true
    done
    summary "$data/oracle-out-muse-tiny/*.log"
    ;;
  facts)
    out="$data/oracle-out-muse"
    [ "$dry" = 1 ] || mkdir -p "$out"
    g=$(snap_of "$GPTQ") b=$(snap_of "$BF16") a=$(snap_of "$ASSIST")
    [ -n "$b" ] || b="$hf/hub/$BF16/snapshots/<the original's small files>"
    host="python3 tools/oracle/muse_facts.py ${b%/} ${g%/} ${a:+--assistant ${a%/}} --scan-quant --no-render \
      --section config,tensors,quant,bytes,tokens,drafter --cache oracle-out-muse-facts"
    if [ "$dry" = 1 ]; then
      echo "DRY_RUN: $host > $out/facts.log"
    elif ! $host > "$out/facts.log" 2>&1; then
      echo "muse_oracle: muse_facts.py FAILED"; tail -20 "$out/facts.log"; rc=1
    fi
    bc="/hf/hub/$BF16/snapshots/$(basename "${b%/}")"
    cmd="python3 tools/oracle/muse_facts.py $bc \"\$SNAP\" --section cites,tokens && \
      python3 tools/oracle/muse_ref.py facts \"\$SNAP\""
    if [ "$dry" = 1 ]; then
      in_box "$GPTQ" "$cmd"
    else
      in_box "$GPTQ" "$cmd" >> "$out/facts.log" 2>&1 || rc=1
      grep -E 'FULL SCAN|citations hold|text names|BOS count' "$out/facts.log"
    fi
    ;;
  ppl|hfcheck)
    for f in "${forms[@]}"; do
      out=$(outdir_of "$f")
      [ "$dry" = 1 ] || mkdir -p "$out"
      if [ "$mode" = ppl ]; then
        cmd="python3 tools/oracle/muse_ref.py ppl \"\$SNAP\" --text prose=tests/golden/prompts/prose.txt \
          --text code=tests/golden/prompts/code.txt --ids muse_4k=tests/golden/prompts/muse_4k.ids \
          --ids muse_agentic=tests/golden/prompts/muse_agentic.ids --max-tokens 16384"
      else
        cmd="python3 tools/oracle/muse_ref.py hfcheck \"\$SNAP\" --layers 4 --prompt tests/golden/prompts/muse_4k.ids \
          --n 2100 --gen 4"
      fi
      echo "muse_oracle: $mode on the $f form ($(date '+%F %T'))"
      if [ "$dry" = 1 ]; then in_box "$(model_of "$f")" "$cmd"; continue; fi
      in_box "$(model_of "$f")" "$cmd" > "$out/$mode.log" 2>&1 || { echo "muse_oracle: $mode $f FAILED"; rc=1; }
      tail -6 "$out/$mode.log"
      if [ "$mode" = ppl ]; then   # finite and below the ESTIMATED sanity ceiling of 30 on prose and the 4k text
        awk '/^ppl (prose|muse_4k):/ { v = $3 + 0; if (!(v > 0 && v < 30)) bad = 1 }
             END { exit bad }' "$out/ppl.log" || { echo "muse_oracle: ppl ($f) outside (0, 30) - a wrong port?"; rc=1; }
      fi
    done
    ;;
  golden)
    for p in $PROMPTS; do
      golden "$GPTQ" "$data/oracle-out-muse" oracle-out-muse.partial "$p" "--taps" || true
    done
    summary "$data/oracle-out-muse/*.log"
    ;;
  golden-bf16)
    for p in muse_short_prose muse_agentic; do
      golden "$BF16" "$data/oracle-out-muse-bf16" oracle-out-muse-bf16.partial "$p" "" || true
    done
    summary "$data/oracle-out-muse-bf16/*.log"
    ;;
  a4)
    # tools/toolcall/a4_ref.sh muse: the committed set, then both references (resumable; the box: foreground)
    a4env=(OUT="$data/oracle-out-muse-a4")
    if [ $mac = 1 ] && [ "$dry" != 1 ]; then
      # the Mac: a4_ref.sh runs one DETACHED container a form, OUT inside this tree; each call advances one step
      [ "$(cd "$data" && pwd -P)" = "$(pwd -P)" ] || { echo "muse_oracle: on the Mac a4 needs the data tree = this tree (.)" >&2; exit 2; }
      OUT=oracle-out-muse-a4 tools/toolcall/a4_ref.sh muse set || exit 1
      for f in gptq bf16; do
        [ -f "oracle-out-muse-a4/DONE.$f" ] && continue
        echo "muse_oracle: a4 - starting the $f reference detached (re-run this mode after it: FORM=$f tools/toolcall/a4_ref.sh muse status)"
        OUT=oracle-out-muse-a4 FORM=$f tools/toolcall/a4_ref.sh muse ref; exit $?
      done
      python3 tools/toolcall/score.py --format atem oracle-out-muse-a4 bf16 gptq | tee oracle-out-muse-a4/score.txt
      exit 0
    fi
    if [ "$dry" = 1 ]; then
      echo "DRY_RUN: ${a4env[*]} tools/toolcall/a4_ref.sh muse set && ${a4env[*]} FORM=both tools/toolcall/a4_ref.sh muse ref"
      echo "DRY_RUN: python3 tools/toolcall/score.py --format atem $data/oracle-out-muse-a4 bf16 gptq"
    else
      env "${a4env[@]}" tools/toolcall/a4_ref.sh muse set || rc=1
      env "${a4env[@]}" FORM=both tools/toolcall/a4_ref.sh muse ref || rc=1
      python3 tools/toolcall/score.py --format atem "$data/oracle-out-muse-a4" bf16 gptq \
        | tee "$data/oracle-out-muse-a4/score.txt" || rc=1
    fi
    ;;
  gateq)
    out="$data/oracle-out-muse-a4"
    if [ "$dry" != 1 ] && [ "$(ls "$out"/*.bf16.txt 2>/dev/null | wc -l)" -ne 36 ]; then
      echo "SKIP_REASON muse_oracle: gateq needs the 36 bf16 A4 references in $out (mode a4 first)"; exit 77
    fi
    # the bf16 rows first (cached beside the run), then the GPTQ form's rows against them; the A4 set is in-tree
    [ "$dry" = 1 ] || mkdir -p oracle-out-muse-gateq.partial
    [ "$dry" = 1 ] || cp -f "$out"/*.bf16.ids oracle-out-muse-gateq.partial/
    bsnap="/hf/hub/$BF16/snapshots/$(basename "$(snap_of "$BF16")")"
    cmd="python3 tools/oracle/muse_gateq.py kl /ws/oracle-out-muse-gateq.partial \"\$SNAP\" \
      --set tests/golden/toolcall-muse --bf16 $bsnap --text prose=tests/golden/prompts/muse_short_prose.ids \
      --text code=tests/golden/prompts/muse_short_code.ids --text muse_4k=tests/golden/prompts/muse_4k.ids \
      --out /ws/oracle-out-muse-gateq.partial/gateq.json"
    if [ "$dry" = 1 ]; then
      in_box "$GPTQ" "$cmd"
    else
      in_box "$GPTQ" "$cmd" > "$data/gateq.kl.log" 2>&1 || { rc=1; tail -20 "$data/gateq.kl.log"; }
    fi
    if [ "$dry" != 1 ] && [ "$rc" = 0 ]; then
      cp -f oracle-out-muse-gateq.partial/gateq.json "$out/gateq.json"
      mv -f "$data/gateq.kl.log" "$out/gateq.kl.log"
      python3 tools/oracle/muse_gateq.py verdict "$out" "$out/gateq.json" | tee "$out/gateq.log"
      [ "${PIPESTATUS[0]}" = 0 ] || rc=1
    fi
    ;;
  synth-tests)
    in_box models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound-GPTQ "python3 tools/quantize/muse/test_muse_quant.py" || rc=1
    ;;
  synth-ckpt)
    nl="${MUSE_SYNTH_LAYERS:-4}"
    [[ "$nl" =~ ^[0-9]+$ ]] && [ "$nl" -ge 1 ] && [ "$nl" -le 52 ] || { echo "muse_oracle: MUSE_SYNTH_LAYERS=$nl (1..52)" >&2; exit 2; }
    out="$data/oracle-out-muse-synth" part=oracle-out-muse-synth.partial ck="ckpt$nl"
    if [ -f "$out/$ck.ACCEPTED" ] && [ -s "$out/$ck/model.safetensors.index.json" ]; then
      echo "muse_oracle: $out/$ck kept (ACCEPTED)"
    else
      GQ=models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound-GPTQ
      bc="/hf/hub/$BF16/snapshots/$(basename "$(snap_of "$BF16")")"
      [ "$dry" = 1 ] || mkdir -p "$out" "$part"
      echo "muse_oracle: synth-ckpt - make_synth.py --layers $nl, then check.py ($(date '+%F %T'))"
      if in_box "$GQ" "python3 tools/quantize/muse/make_synth.py /ws/$part/$ck --layers $nl --tokenizer $bc \
            --gptq \"\$SNAP\" > /ws/$part/$ck.log 2>&1 && \
            python3 tools/quantize/muse/check.py /ws/$part/$ck > /ws/$part/$ck.check.log 2>&1"; then
        if [ "$dry" != 1 ]; then
          tail -1 "$part/$ck.check.log"
          [ -e "$out/$ck" ] && mv -f "$out/$ck" "$out/$ck.stale.$(date +%s)"
          mv -f "$part/$ck" "$out/$ck" && mv -f "$part/$ck.log" "$out/$ck.log" && mv -f "$part/$ck.check.log" "$out/$ck.ACCEPTED"
          du -sh "$out/$ck"
        fi
      else
        echo "muse_oracle: synth-ckpt FAILED"
        tail -20 "$part/$ck.log" "$part/$ck.check.log" 2>/dev/null
        rc=1
      fi
    fi
    ;;
  taps)
    a4="$data/oracle-out-muse-a4"
    if [ "$dry" != 1 ] && [ "$(ls "$a4"/*.gptq.ids 2>/dev/null | wc -l)" -ne 36 ]; then
      echo "SKIP_REASON muse_oracle: taps needs the 36 GPTQ-form A4 continuations in $a4 (mode a4 first)"; exit 77
    fi
    out="$data/oracle-out-muse-taps" part=oracle-out-muse-taps.partial
    [ "$dry" = 1 ] || mkdir -p "$out" "$part"
    for f in tests/golden/toolcall-muse/*.ids; do
      n=$(basename "$f" .ids)
      [ -s "$out/$n.golden.safetensors" ] && { echo "muse_oracle: taps $n kept"; continue; }
      [ "$dry" = 1 ] || { tr -s ' \n' ' ' < "$f"; tr -s ' \n' ' ' < "$a4/$n.gptq.ids"; echo; } > "$part/$n.ids"
      in_box "$GPTQ" "python3 tools/oracle/muse_ref.py run \"\$SNAP\" --prompt /ws/$part/$n.ids \
        --out /ws/$part/$n.golden.safetensors --gen 0 --taps --act-tail 0 --logits-tail 1 > /ws/$part/$n.log 2>&1" \
        || { echo "muse_oracle: taps $n FAILED"; rc=1; continue; }
      [ "$dry" = 1 ] || for x in "$n.ids" "$n.log" "$n.golden.safetensors"; do mv -f "$part/$x" "$out/$x"; done
    done
    ;;
esac
exit "$rc"
