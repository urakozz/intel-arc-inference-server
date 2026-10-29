#!/usr/bin/env bash
# Spec 25b: the Nemotron 3.5 Bolt synthetic real-width checkpoints and their Python tests (the synth-ckpt /
# synth-tests shape of the other models' oracle scripts) - ON THE BOX CPU, and on the Mac (the Mac holds /tmp/b70-cpu-model.lock, one model job at a time).
#   tools/box_validate/bolt_synth.sh <data tree> synth-ckpt|synth-tests
#
#   synth-ckpt   tools/quantize/bolt/make_synth.py for each N in BOLT_SYNTH_LAYERS (default "8 12": N 8 the host tests',
#                N 12 the card tests'), three checkpoints each - the GPTQ g128 spelling (gptq-g128), its EXACT g64 expansion
#                (--expand-from: the pair whose device bytes and logits must be identical) and a native g64 packing - then
#                check.py on each -> <data>/oracle-out-bolt-synth/n<N>-{g128,g64x,g64}/ + <name>.ACCEPTED (the check's log,
#                written last: it marks done). N 8 ~3.9 GB, N 12 ~5.5 GB a checkpoint (measured on the Mac). Needs no
#                weight download: our export's pinned small files only (config / tokenizer / template / generation
#                config, for the copies and the remote code's name list).
#   synth-tests  tools/quantize/bolt/test_bolt_synth.py (the packer, the exact expansion, the config, widths), and with
#                the N 8 pair present its --e2e (check.py ACCEPTED on both, bolt_ref.build_streamed logits on 16 ids bitwise
#                equal)
# Exit 77 (SKIP) when the small files are absent or the memory is short (BOLT_SYNTH_MIN_GB 24; the Mac: the container cap
# BOLT_MEM_GB, default 40). DRY_RUN=1 prints every command and runs nothing.
# Env: SNAP_BOLT (default models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ), BOLT_SYNTH_LAYERS,
#      BOLT_SYNTH_MIN_GB, BOLT_MEM_GB, BOLT_LOCK, BOLT_LOCK_HELD, HF_HOME, ORACLE_IMAGE, ORACLE_THREADS.
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 2 ] || { echo "usage: $0 <data tree> synth-ckpt|synth-tests" >&2; exit 2; }
data="$1" mode="$2"
case "$mode" in synth-ckpt|synth-tests) ;; *) echo "bolt_synth: mode $mode is not synth-ckpt|synth-tests" >&2; exit 2 ;; esac
mac=0
[ "$(uname -s)" = Darwin ] && mac=1
dry="${DRY_RUN:-0}"
hf="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_CACHE="${HF_CACHE:-$hf}"
INT4="${SNAP_BOLT:-models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ}"
layers="${BOLT_SYNTH_LAYERS:-8 12}"
for n in $layers; do
  [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 8 ] && [ "$n" -le 88 ] || { echo "bolt_synth: BOLT_SYNTH_LAYERS entry $n (8..88)" >&2; exit 2; }
done
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
need="${BOLT_SYNTH_MIN_GB:-24}"
snapdir=$(ls -d "$hf/hub/$INT4"/snapshots/*/ 2>/dev/null | head -1)
echo "bolt_synth: $mode -> $data; $INT4 (${snapdir:-absent}); $memwhat ${avail:-0} GB (need $need)"
if [ "$dry" != 1 ] && [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON bolt_synth: $memwhat ${avail:-0} GB < $need GB"
  exit 77
fi
if [ "$dry" != 1 ] && { [ -z "$snapdir" ] || [ ! -s "$snapdir/config.json" ]; }; then
  echo "SKIP_REASON bolt_synth: the pinned small files are not in $hf/hub (hf download urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ --revision 8932aa8d1844f91056c5b9433448075a842b9156 --exclude '*.safetensors')"
  exit 77
fi
if [ $mac = 1 ] && [ "$dry" != 1 ] && [ -z "${BOLT_LOCK_HELD:-}" ]; then
  lock="${BOLT_LOCK:-/tmp/b70-cpu-model.lock}"
  echo "bolt_synth: waiting for $lock ($(date '+%F %T'))"
  exec env BOLT_LOCK_HELD=1 python3 -c 'import fcntl, subprocess, sys
f = open(sys.argv[1], "a")
fcntl.flock(f, fcntl.LOCK_EX)
print("bolt_synth: holding the lock", flush=True)
sys.exit(subprocess.call(sys.argv[2:]))' "$lock" "$PWD/tools/box_validate/bolt_synth.sh" "$@"
fi
rc=0
# in_box <command>: one oracle-image run, the 5.19.0 site first on PYTHONPATH, $SNAP = the pinned snapshot (small files).
in_box() {
  local cmd="tools/oracle/qwen4exp_env.sh /ws/$SITE > /dev/null && export PYTHONPATH=/ws/$SITE && $1"
  if [ "$dry" = 1 ]; then echo "DRY_RUN: ORACLE_MODEL=$INT4 tools/oracle/run_in_container.sh '$cmd'"; return 0; fi
  ORACLE_MODEL="$INT4" tools/oracle/run_in_container.sh "$cmd"
}
out="$data/oracle-out-bolt-synth" part=oracle-out-bolt-synth.partial
case "$mode" in
  synth-ckpt)
    for nl in $layers; do
      if [ -f "$out/n$nl-g128.ACCEPTED" ] && [ -f "$out/n$nl-g64x.ACCEPTED" ] && [ -f "$out/n$nl-g64.ACCEPTED" ] \
         && [ -s "$out/n$nl-g128/model.safetensors.index.json" ]; then
        echo "bolt_synth: $out/n$nl-{g128,g64x,g64} kept (ACCEPTED)"
        continue
      fi
      [ "$dry" = 1 ] || { mkdir -p "$out" "$part"; }
      echo "bolt_synth: synth-ckpt N $nl - make_synth.py g128, its exact g64 expansion, a native g64; check.py on each ($(date '+%F %T'))"
      # one container run: the g64x form reads the g128 directory while it is still under the tree (/ws/$part)
      cmd="python3 tools/quantize/bolt/make_synth.py /ws/$part/n$nl-g128 --layers $nl --form gptq-g128 --tokenizer \"\$SNAP\" \
            > /ws/$part/n$nl-g128.log 2>&1 && python3 tools/quantize/bolt/check.py /ws/$part/n$nl-g128 > /ws/$part/n$nl-g128.check.log 2>&1 && \
            python3 tools/quantize/bolt/make_synth.py /ws/$part/n$nl-g64x --layers $nl --form ar-g64 --expand-from /ws/$part/n$nl-g128 \
              --tokenizer \"\$SNAP\" > /ws/$part/n$nl-g64x.log 2>&1 && \
            python3 tools/quantize/bolt/check.py /ws/$part/n$nl-g64x > /ws/$part/n$nl-g64x.check.log 2>&1 && \
            python3 tools/quantize/bolt/make_synth.py /ws/$part/n$nl-g64 --layers $nl --form ar-g64 --tokenizer \"\$SNAP\" \
              > /ws/$part/n$nl-g64.log 2>&1 && python3 tools/quantize/bolt/check.py /ws/$part/n$nl-g64 > /ws/$part/n$nl-g64.check.log 2>&1"
      if in_box "$cmd"; then
        if [ "$dry" != 1 ]; then
          for form in g128 g64x g64; do
            name="n$nl-$form"
            tail -1 "$part/$name.check.log"
            [ -e "$out/$name" ] && mv -f "$out/$name" "$out/$name.stale.$(date +%s)"
            mv -f "$part/$name" "$out/$name" && mv -f "$part/$name.log" "$out/$name.log" && \
              mv -f "$part/$name.check.log" "$out/$name.ACCEPTED"
            du -sh "$out/$name"
          done
        fi
      else
        echo "bolt_synth: synth-ckpt N $nl FAILED"
        tail -20 "$part"/n"$nl"-*.log 2>/dev/null
        rc=1
      fi
    done
    ;;
  synth-tests)
    in_box "python3 tools/quantize/bolt/test_bolt_synth.py" || rc=1
    ;;
esac
exit "$rc"
