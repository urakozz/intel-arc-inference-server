#!/usr/bin/env bash
# Spec 25a: the Nemotron 3.5 Bolt 74B-A9B CPU reference runs (kolibri_oracle.sh's shape), ON THE BOX
# CPU - the Mac runs `tests` only (the real checkpoints need more RAM than its 40 GB container cap; BOLT_REF_MIN_GB).
#   tools/box_validate/bolt_oracle.sh <data tree> tests|ppl|hfcheck|golden|golden-bf16|export <dir>|a4|a4-bf16|gateq
#
# Every mode runs tools/oracle/bolt_ref.py (the checkpoint's own remote code, layer-streamed, its Mamba mixers on the
# kernel path through bolt_mamba_ops, an fp32 SSM / conv cache) in the oracle image (tools/oracle/run_in_container.sh)
# with transformers 5.19.0 from spec 21a's site BESIDE the image (tools/oracle/qwen4exp_env.sh; the box's
# oracle-out-q4exp-site, the Mac's oracle-out-q4exp/site) first on PYTHONPATH.
#   tests        test_bolt_make_tiny.py, test_bolt_ref.py, bolt_tiny_check.py and the toolcall tests (test_score.py,
#                test_make_set.py, test_oracle_batched.py test_bolt) on tiny puzzles built from the pinned snapshot's
#                SMALL files (r57.host)
#   ppl          perplexity on prose.txt, code.txt (no BOS) and bolt_agentic.ids for each form present (our g64 export
#                dequantised; the BF16 original) -> <data>/oracle-out-bolt{,-bf16}/ppl.log; finite, code below an
#                ESTIMATED sanity ceiling of 10 (recorded per text)
#   hfcheck      bolt_ref.py hfcheck --layers 12 on 2048 ids of bolt_4k + 4 cached steps for each form present: the
#                port = its restated chain BITWISE (exit 1 otherwise), R's torch fallback recorded; fp32 when
#                MemAvailable >= BOLT_FP32_GB (64), else bf16 -> <data>/oracle-out-bolt{,-bf16}/hfcheck.log
#   golden       our g64 export dequantised (the engine-format reference every B3 gate reads), every golden prompt
#                (BOLT_PROMPTS: bolt_short bolt_4k bolt_agentic bolt_32k), --gen 32 --taps -> <data>/oracle-out-bolt/
#                <p>.{ids,golden.safetensors,log}; each log's gap lines (the 9th / 10th sel per MoE block:
#                B70_BOLT_TIE_TOL's evidence, Decision 11) and SSM-state magnitudes
#   golden-bf16  the BF16 original on BOLT_BF16_PROMPTS (bolt_short bolt_agentic) -> <data>/oracle-out-bolt-bf16/
#   export <dir> the golden prompts on an int4 checkpoint dequantised (25b's synthetics; mounted as ORACLE_SNAP)
#                -> <data>/oracle-out-bolt-<name>/ (<name> = the directory's base name)
#   a4 / a4-bf16 tools/toolcall/a4_ref.sh bolt set + ref FORM=int4 / FORM=bf16 (36 scenarios, greedy, 2048 new ids,
#                reasoning on, batched) -> <data>/oracle-out-bolt-a4/<name>.{int4,bf16}.{ids,txt,gap} (DAYS,
#                ESTIMATED; resumable per scenario); score.py --format qwen-think once both exist
#   gateq        tools/oracle/bolt_gateq.py kl (teacher-forced over the BF16 A4 continuations + prose / code / 4k)
#                and verdict -> <data>/oracle-out-bolt-a4/{gateq.json,gateq.log}: spec 25 B5's table, a RECORDED
#                measurement (Decision 3)
# Golden runs are per prompt resumable (written into <tree>/oracle-out-bolt*.partial/, moved when complete, the golden
# file last). The int4 form's output directories (oracle-out-bolt, oracle-out-bolt-a4) carry a SNAPSHOT file naming the
# snapshot they were made from (<cache dir>/snapshots/<sha>; data.sh's keys require our export's): one without it, or
# naming another, is another checkpoint's (the 2026-10-22/23 ones: a third-party g128 export) - it is set aside whole as
# <dir>.stale-<time> (never deleted) and the run starts it again. Exit 77 (SKIP) when the RAM is short (BOLT_REF_MIN_GB:
# 8 tests, 48 the int4 form, 96 the BF16 one - ESTIMATED; the box: MemAvailable; the Mac: the ORACLE_MEMORY cap) or a snapshot is absent / incomplete (every shard
# of its index, no *.incomplete blob - never downloaded here: df -h ~ first, then hf download by hand, pinned):
#   hf download urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ --revision 8932aa8d1844f91056c5b9433448075a842b9156
#   hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 --revision 567021037ff88364143e24ea05e6abd0f72086ff
# DRY_RUN=1 prints every command, the floor and an ESTIMATED time per prompt, runs nothing.
# The Mac: ORACLE_IMAGE agnes-ref-img, the container capped at BOLT_MEM_GB (40), the whole run under flock(2) on
# BOLT_LOCK (/tmp/b70-cpu-model.lock: one model job on the Mac's CPU at a time).
# Env: SNAP_BOLT (models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ), SNAP_BOLT_BF16
#      (models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16), BOLT_PROMPTS, BOLT_BF16_PROMPTS, BOLT_GEN (32),
#      BOLT_REF_MIN_GB, BOLT_FP32_GB, BOLT_KEEP ('' | all: blocks resident once read),
#      BOLT_MEM_GB, BOLT_LOCK, BOLT_LOCK_HELD, HF_HOME, ORACLE_IMAGE, ORACLE_THREADS.
set -u
cd "$(dirname "$0")/../.." || exit 2
MODES="tests|ppl|hfcheck|golden|golden-bf16|export <dir>|a4|a4-bf16|gateq"
[ $# -ge 2 ] || { echo "usage: $0 <data tree> $MODES" >&2; exit 2; }
data="$1" mode="$2" arg="${3:-}"
case "$mode" in tests|ppl|hfcheck|golden|golden-bf16|a4|a4-bf16|gateq) ;;
  export) [ -n "$arg" ] || { echo "bolt_oracle: export needs the export's directory" >&2; exit 2; } ;;
  *) echo "bolt_oracle: mode $mode is not one of $MODES" >&2; exit 2 ;;
esac
mac=0
[ "$(uname -s)" = Darwin ] && mac=1
dry="${DRY_RUN:-0}"
hf="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_CACHE="${HF_CACHE:-$hf}"    # run_in_container.sh mounts HF_CACHE at /hf: the cache this script reads
gen="${BOLT_GEN:-32}"
[[ "$gen" =~ ^[0-9]+$ ]] || { echo "bolt_oracle: BOLT_GEN" >&2; exit 2; }
keep="${BOLT_KEEP:-}"
[ -z "$keep" ] || [[ "$keep" =~ ^(all|[0-9,]+)$ ]] || { echo "bolt_oracle: BOLT_KEEP=$keep" >&2; exit 2; }
prompts="${BOLT_PROMPTS:-bolt_short bolt_4k bolt_agentic bolt_32k}"
bprompts="${BOLT_BF16_PROMPTS:-bolt_short bolt_agentic}"
for p in $prompts $bprompts; do
  [ -s "tests/golden/prompts/$p.ids" ] || { echo "bolt_oracle: no tests/golden/prompts/$p.ids" >&2; exit 2; }
done
INT4="${SNAP_BOLT:-models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ}"
BF16="${SNAP_BOLT_BF16:-models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16}"
if [ $mac = 1 ]; then
  SITE=oracle-out-q4exp/site
  export ORACLE_IMAGE="${ORACLE_IMAGE:-agnes-ref-img:latest}" ORACLE_MEMORY="${BOLT_MEM_GB:-40}g"
  export ORACLE_THREADS="${ORACLE_THREADS:-12}"
  avail="${BOLT_MEM_GB:-40}" memwhat="the container cap BOLT_MEM_GB"
else
  SITE=oracle-out-q4exp-site
  avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
  memwhat="MemAvailable"
fi
case "$mode" in
  tests) need="${BOLT_REF_MIN_GB:-8}" ;;
  golden-bf16|a4-bf16|gateq) need="${BOLT_REF_MIN_GB:-96}" ;;
  *) need="${BOLT_REF_MIN_GB:-48}" ;;
esac
echo "bolt_oracle: $mode${arg:+ $arg} -> $data; $memwhat ${avail:-0} GB (need $need)"

snap_of() { ls -d "$hf/hub/$1"/snapshots/*/ 2>/dev/null | head -1; }
# complete <cache dir name>: config.json, every shard the index names, no *.incomplete blob written in the last hour (a
# download in flight; an older one is an interrupted download's leftover - the box's copy of our export has 8)
complete() {
  local d; d=$(snap_of "$1")
  [ -n "$d" ] && [ -f "$d/config.json" ] || return 1
  [ -n "$(find "$hf/hub/$1/blobs/" -maxdepth 1 -name '*.incomplete' -mmin -60 2>/dev/null | head -1)" ] && return 1
  python3 - "$d" <<'EOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
files = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
sys.exit(1 if [f for f in files if not os.path.exists(os.path.join(d, f))] else 0)
EOF
}
small() { local d; d=$(snap_of "$1"); [ -n "$d" ] && [ -s "$d/modeling_nemotron_h.py" ] && [ -s "$d/config.json" ]; }
forms=()
complete "$INT4" && forms+=(int4)
complete "$BF16" && forms+=(bf16)
model_of() { if [ "$1" = bf16 ]; then echo "$BF16"; else echo "$INT4"; fi; }
# own_dir <out dir>: make <out> this int4 snapshot's - set a directory whose SNAPSHOT file names another (or that has none)
# aside whole as <out>.stale-<time>, then write SNAPSHOT. snap_want: what SNAPSHOT says (data.sh compares it to ours).
snap_want() { echo "$INT4/snapshots/$(basename "$(snap_of "$INT4")")"; }
own_dir() {
  local out="$1" want aside
  want=$(snap_want)
  if [ -z "$(snap_of "$INT4")" ]; then   # nothing to name (a4-bf16 needs only the BF16 original): touch nothing
    echo "bolt_oracle: $INT4 is not in $hf/hub - $out is neither set aside nor claimed"
    [ "$dry" = 1 ] || mkdir -p "$out"
    return 0
  fi
  if [ -e "$out" ] && [ "$(cat "$out/SNAPSHOT" 2>/dev/null)" != "$want" ]; then
    aside="$out.stale-$(date +%Y%m%d-%H%M%S)"
    echo "bolt_oracle: $out was made from $(cat "$out/SNAPSHOT" 2>/dev/null || echo 'another checkpoint (no SNAPSHOT file)'), not $want - set aside as $aside"
    [ "$dry" = 1 ] || mv "$out" "$aside" || return 1
  fi
  [ "$dry" = 1 ] || { mkdir -p "$out" && printf '%s\n' "$want" > "$out/SNAPSHOT"; }
}
outdir_of() { if [ "$1" = bf16 ]; then echo "$data/oracle-out-bolt-bf16"; else echo "$data/oracle-out-bolt"; fi; }

if [ "$dry" = 1 ]; then
  echo "DRY_RUN: RAM floor $need GB ($memwhat ${avail:-0} GB); site $SITE; forms present: ${forms[*]:-none}"
  echo "DRY_RUN: ESTIMATED times on the box CPU (22 Haswell cores), layer-streamed - nothing measured on real weights yet:"
  echo "  a token reads 9.05 B active parameters (18 GFLOP); a 2048-row prefill chunk routes to ~all 13,824 experts, so a"
  echo "  chunk dequantises ~66 B expert + 7 B dense weights (int4 -> bf16, stream.dequant_t ~1-2 GB/s: ~1.5-3 min) and"
  echo "  computes ~37 TFLOP (~4-8 min at 100-150 GFLOP/s); a decode step re-dequantises the 7 B dense weights + 360"
  echo "  experts: ~10-15 s (BOLT_KEEP=all keeps the dense blocks resident, ~13.6 GB bf16: ~3-5 s a step)."
  echo "    bolt_short (96 ids + 32 steps) ~10 min | bolt_4k (2 chunks) ~20-30 min | bolt_agentic (12.7k ids, 7 chunks)"
  echo "    ~1.5-2 h | bolt_32k (16 chunks; eager scores up to [32][2048][32768] fp32 ~8.6 GB a layer call) ~4-6 h"
  echo "    golden-bf16: x ~1.5 (no dequant, 150 GB read a pass from the page cache or disk) | ppl ~1 h a form"
  echo "    hfcheck --layers 12 ~15 min a form | a4: 36 x (~2k-id prompt + up to 2048 steps), batched 4: ~2-4 days a form"
  echo "    gateq: 72 teacher-forced passes ~1 day"
fi
if [ "$dry" != 1 ] && [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON bolt_oracle: $memwhat ${avail:-0} GB < BOLT_REF_MIN_GB $need (the real checkpoints need the box)"
  exit 77
fi
miss() { echo "SKIP_REASON bolt_oracle: $1"; exit 77; }
if [ "$dry" != 1 ]; then
  case "$mode" in
    tests) small "$INT4" || miss "the pinned snapshot's small files are not in $hf/hub (hf download urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ --revision 8932aa8d1844f91056c5b9433448075a842b9156 --exclude '*.safetensors')" ;;
    golden) complete "$INT4" || miss "$INT4 is absent or incomplete (hf download urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ --revision 8932aa8d...: 41 GB, df -h ~ first)" ;;
    a4) complete "$INT4" || miss "$INT4 is absent or incomplete (41 GB)" ;;
    golden-bf16|a4-bf16) complete "$BF16" || miss "$BF16 is absent or incomplete (hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 --revision 56702103...: 149.8 GB, df -h ~ first)" ;;
    gateq) { complete "$INT4" && complete "$BF16"; } || miss "gate Q needs both $INT4 and $BF16 complete (41 + 149.8 GB)" ;;
    ppl|hfcheck) [ ${#forms[@]} -gt 0 ] || miss "neither $INT4 nor $BF16 is complete in $hf/hub" ;;
    export) [ -f "$arg/config.json" ] || miss "$arg has no config.json (25b's synthetic checkpoint)" ;;
  esac
fi
[ ${#forms[@]} -gt 0 ] || forms=(int4)      # DRY_RUN with nothing downloaded: show the int4 form's commands
# The Mac: one model job at a time on its CPU - re-run this script under flock(2) AFTER the SKIP checks.
if [ $mac = 1 ] && [ "$dry" != 1 ] && [ -z "${BOLT_LOCK_HELD:-}" ]; then
  lock="${BOLT_LOCK:-/tmp/b70-cpu-model.lock}"
  echo "bolt_oracle: waiting for $lock ($(date '+%F %T'))"
  exec env BOLT_LOCK_HELD=1 python3 -c 'import fcntl, subprocess, sys
f = open(sys.argv[1], "a")
fcntl.flock(f, fcntl.LOCK_EX)
print("bolt_oracle: holding the lock", flush=True)
sys.exit(subprocess.call(sys.argv[2:]))' "$lock" "$PWD/tools/box_validate/bolt_oracle.sh" "$@"
fi
rc=0
kflag=""
[ -n "$keep" ] && kflag="--keep $keep"

# in_box <cache dir name> <command> - one oracle-image run, the 5.19.0 site first on PYTHONPATH, $SNAP = the snapshot.
in_box() {
  local cmd="tools/oracle/qwen4exp_env.sh /ws/$SITE > /dev/null && export PYTHONPATH=/ws/$SITE && $2"
  if [ -n "${ORACLE_SNAP:-}" ]; then
    if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_SNAP=$ORACLE_SNAP tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
    tools/oracle/run_in_container.sh "$cmd"
    return
  fi
  if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_MODEL=$1 tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
  ORACLE_MODEL="$1" tools/oracle/run_in_container.sh "$cmd"
}

# golden <cache dir name> <out dir> <partial dir> <prompt> <gen> <extra run args>
golden() {
  local model="$1" out="$2" part="$3" p="$4" g="$5" extra="$6"
  if [ -s "$out/$p.golden.safetensors" ] && [ -s "$out/$p.ids" ]; then
    echo "bolt_oracle: $p kept ($out/$p.golden.safetensors)"
    return 0
  fi
  [ "$dry" = 1 ] || mkdir -p "$out" "$part"
  echo "bolt_oracle: $p - bolt_ref.py run --gen $g $extra ($(date '+%F %T'))"
  if ! in_box "$model" "cp tests/golden/prompts/$p.ids /ws/$part/$p.ids && python3 tools/oracle/bolt_ref.py run \"\$SNAP\" \
        --prompt /ws/$part/$p.ids --out /ws/$part/$p.golden.safetensors --gen $g $kflag $extra > /ws/$part/$p.log 2>&1"; then
    echo "bolt_oracle: $p FAILED"
    tail -20 "$part/$p.log" 2>/dev/null
    rc=1
    return 1
  fi
  [ "$dry" = 1 ] && return 0
  for f in "$p.ids" "$p.log" "$p.golden.safetensors"; do   # the golden file last: it marks done
    mv -f "$part/$f" "$out/$f"
  done
  echo "bolt_oracle: $p done ($(date '+%F %T')); $(grep -E '^wall' "$out/$p.log")"
}

summary() {   # the routing gaps (B70_BOLT_TIE_TOL's evidence, Decision 11), the SSM state magnitudes, walls and RSS
  echo "--- the 9th / 10th sel gap per MoE block, the SSM state after the prompt, walls"
  # shellcheck disable=SC2086 # the glob is meant
  [ "$dry" = 1 ] || grep -H -E 'sel gap|SSM state|^wall|^prompt forward|^greedy' $1 2>/dev/null
}

case "$mode" in
  tests)
    in_box "$INT4" "python3 tools/oracle/bolt_tiny_check.py \"\$SNAP\" && python3 tools/oracle/test_bolt_make_tiny.py && \
      python3 tools/oracle/test_bolt_ref.py && python3 tools/toolcall/test_score.py && \
      python3 tools/toolcall/test_make_set.py && python3 tools/toolcall/test_oracle_batched.py test_bolt" || rc=1
    ;;
  ppl|hfcheck)
    for f in "${forms[@]}"; do
      out=$(outdir_of "$f")
      if [ "$f" = int4 ]; then own_dir "$out" || { rc=1; continue; }; elif [ "$dry" != 1 ]; then mkdir -p "$out"; fi
      if [ "$mode" = ppl ]; then
        cmd="python3 tools/oracle/bolt_ref.py ppl \"\$SNAP\" --weights $f $kflag --text prose=tests/golden/prompts/prose.txt \
          --text code=tests/golden/prompts/code.txt --ids agentic=tests/golden/prompts/bolt_agentic.ids --max-tokens 16384"
      else
        dt=bf16
        [ "${avail:-0}" -ge "${BOLT_FP32_GB:-64}" ] && dt=fp32
        cmd="python3 tools/oracle/bolt_ref.py hfcheck \"\$SNAP\" --weights $f --layers 12 --dtype $dt \
          --prompt tests/golden/prompts/bolt_4k.ids --n 2048 --gen 4"
      fi
      echo "bolt_oracle: $mode on the $f form ($(date '+%F %T'))"
      if [ "$dry" = 1 ]; then in_box "$(model_of "$f")" "$cmd"; continue; fi
      in_box "$(model_of "$f")" "$cmd > /ws/oracle-out-bolt.$mode.$f.log 2>&1" || { echo "bolt_oracle: $mode $f FAILED"; rc=1; }
      mv -f "oracle-out-bolt.$mode.$f.log" "$out/$mode.log"
      tail -6 "$out/$mode.log"
      if [ "$mode" = ppl ]; then   # finite; code below an ESTIMATED sanity ceiling of 10
        awk '/^ppl / { n++; if ($3 ~ /nan|inf/ || /NOT SANE/) bad = 1 }
             /^ppl code:/ { if ($3 + 0 > 10) { print "bolt_oracle: code perplexity " $3 " above the ESTIMATED ceiling 10"; bad = 1 } }
             END { if (n < 3) bad = 1; exit bad }' "$out/ppl.log" || { echo "bolt_oracle: ppl ($f) not sane"; rc=1; }
      fi
    done
    ;;
  golden)
    own_dir "$data/oracle-out-bolt" || exit 1
    for p in $prompts; do
      golden "$INT4" "$data/oracle-out-bolt" oracle-out-bolt.partial "$p" "$gen" "--weights int4 --taps" || true
    done
    summary "$data/oracle-out-bolt/bolt_*.log"
    ;;
  golden-bf16)
    for p in $bprompts; do
      golden "$BF16" "$data/oracle-out-bolt-bf16" oracle-out-bolt-bf16.partial "$p" "$gen" "--weights bf16" || true
    done
    summary "$data/oracle-out-bolt-bf16/bolt_*.log"
    ;;
  export)
    name=$(basename "${arg%/}")
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bolt_oracle: export directory name $name" >&2; exit 2; }
    export ORACLE_SNAP
    ORACLE_SNAP="$(cd "$arg" 2>/dev/null && pwd || echo "$arg")"
    for p in $prompts; do
      golden export "$data/oracle-out-bolt-$name" "oracle-out-bolt-$name.partial" "$p" "$gen" "--weights int4 --taps" || true
    done
    unset ORACLE_SNAP
    summary "$data/oracle-out-bolt-$name/bolt_*.log"
    ;;
  a4|a4-bf16)
    out="$data/oracle-out-bolt-a4"
    f=int4
    [ "$mode" = a4-bf16 ] && f=bf16
    own_dir "$out" || exit 1
    if [ "$dry" = 1 ]; then
      echo "DRY_RUN: OUT=$out tools/toolcall/a4_ref.sh bolt set; OUT=$out FORM=$f tools/toolcall/a4_ref.sh bolt ref"
      echo "DRY_RUN: python3 tools/toolcall/score.py --format qwen-think $out bf16 int4   (once both forms are done)"
    else
      [ -s "$out/set/manifest.json" ] || { OUT="$out" tools/toolcall/a4_ref.sh bolt set || rc=1; }
      if [ $mac = 1 ]; then a4mem="${BOLT_MEM_GB:-40}g"; else a4mem="${MEM:-}"; fi
      [ "$rc" = 0 ] && { OUT="$out" FORM=$f FOREGROUND=1 MEM="$a4mem" tools/toolcall/a4_ref.sh bolt ref || rc=1; }
      OUT="$out" FORM=$f tools/toolcall/a4_ref.sh bolt status
      if [ -f "$out/DONE.int4" ] && [ -f "$out/DONE.bf16" ]; then
        python3 tools/toolcall/score.py --format qwen-think "$out" bf16 int4 | tee "$out/score.txt"
      elif [ -f "$out/DONE.$f" ]; then
        python3 tools/toolcall/score.py --format qwen-think "$out" "$f" "$f" | tail -2
      fi
    fi
    ;;
  gateq)
    out="$data/oracle-out-bolt-a4"
    if [ "$dry" != 1 ] && [ "$(cat "$out/SNAPSHOT" 2>/dev/null)" != "$(snap_want)" ]; then
      miss "gateq needs $out made from $(snap_want) (its SNAPSHOT file; modes a4 and a4-bf16 first)"
    fi
    if [ "$dry" != 1 ] && [ "$(ls "$out"/*.bf16.txt 2>/dev/null | wc -l)" -ne 36 ]; then
      miss "gateq needs the 36 BF16 A4 references in $out (mode a4-bf16 first)"
    fi
    [ "$dry" = 1 ] || { mkdir -p oracle-out-bolt-gateq.partial && cp -f "$out"/*.bf16.ids oracle-out-bolt-gateq.partial/; }
    bsnap="/hf/hub/$BF16/snapshots/$(basename "$(snap_of "$BF16")")"
    cmd="python3 tools/oracle/bolt_gateq.py kl /ws/oracle-out-bolt-gateq.partial \"\$SNAP\" --set tests/golden/toolcall-bolt \
      --bf16 $bsnap --text prose=tests/golden/prompts/bolt_short_prose.ids --text code=tests/golden/prompts/bolt_short_code.ids \
      --text bolt_4k=tests/golden/prompts/bolt_4k.ids --out /ws/oracle-out-bolt-gateq.partial/gateq.json"
    if [ "$dry" = 1 ]; then
      in_box "$INT4" "$cmd"
      echo "DRY_RUN: python3 tools/oracle/bolt_gateq.py verdict $out $out/gateq.json"
    else
      in_box "$INT4" "$cmd > /ws/oracle-out-bolt-gateq.partial/kl.log 2>&1" || { rc=1; tail -20 oracle-out-bolt-gateq.partial/kl.log; }
      if [ "$rc" = 0 ]; then
        cp -f oracle-out-bolt-gateq.partial/gateq.json oracle-out-bolt-gateq.partial/kl.log "$out/"
        python3 tools/oracle/bolt_gateq.py verdict "$out" "$out/gateq.json" | tee "$out/gateq.log"
      fi
    fi
    ;;
esac
exit "$rc"
