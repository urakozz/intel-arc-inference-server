#!/usr/bin/env bash
# Spec 18a's real-weight K2-Horizon reference, ON THE BOX CPU (tools/oracle/README.md "The
# K2-Horizon reference"): the golden set k2_golden_test reads, into <data tree>/oracle-out-k2.
#   tools/box_validate/k2_oracle.sh <data tree>
# Run from the tree under test (its tools/oracle/k2_ref.py runs; the container mounts the tree
# as /ws). Per prompt of prose / code / cjk whose <p>.golden.safetensors is not in the data
# tree yet: tokenize.py --bos (K2's prompts start with BOS 0) -> <p>.ids, then `k2_ref.py run
# --gen 32` -> <p>.golden.safetensors (+ <p>.log with the MoE 8th/9th and MoVA 4th/5th
# selection-gap distribution that sets B70_K2_TIE_TOL). Each prompt is written to
# <tree>/oracle-out-k2.partial/ (box.sh sync leaves oracle-out-* alone) and moved into the
# data tree once complete, so a killed run resumes at the prompt it was on. Finally the gap
# lines of every log are printed. ~5-6 min per prompt on the int4 checkpoint (estimated).
# Exit 77 (SKIP) when MemAvailable is below K2_REF_MIN_GB (the box is shared: check before a
# CPU oracle run).
# Env: K2_ORACLE_MODEL (the HF cache dir name; default the int4 checkpoint the engine runs,
# so the golden tokens are the ones it can match), K2_REF_MIN_GB (32), K2_HFCHECK=1 (also
# `k2_ref.py hfcheck` on prose against the vendored model - needs transformers >= 5.13 in the
# image; its result is printed, never fatal), ORACLE_IMAGE, ORACLE_THREADS.
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -eq 1 ] || { echo "usage: $0 <data tree>" >&2; exit 2; }
data="$1"
model="${K2_ORACLE_MODEL:-models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ}"
need="${K2_REF_MIN_GB:-32}"
avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
echo "k2_oracle: $model -> $data/oracle-out-k2; MemAvailable ${avail} GB (need $need)"
if [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON k2_oracle: MemAvailable ${avail} GB < K2_REF_MIN_GB $need (another job holds the RAM)"
  exit 77
fi
mkdir -p "$data/oracle-out-k2" oracle-out-k2.partial
rc=0
for p in prose code cjk; do
  if [ -s "$data/oracle-out-k2/$p.golden.safetensors" ] && [ -s "$data/oracle-out-k2/$p.ids" ]; then
    echo "k2_oracle: $p kept ($data/oracle-out-k2/$p.golden.safetensors)"
    continue
  fi
  echo "k2_oracle: $p - tokenize (--bos), then k2_ref.py run --gen 32 ($(date '+%F %T'))"
  if ! ORACLE_MODEL="$model" tools/oracle/run_in_container.sh \
      "python3 tools/oracle/tokenize.py \"\$SNAP\" encode --bos tests/golden/prompts/$p.txt > /ws/oracle-out-k2.partial/$p.ids && \
       python3 tools/oracle/k2_ref.py run \"\$SNAP\" --prompt /ws/oracle-out-k2.partial/$p.ids \
         --out /ws/oracle-out-k2.partial/$p.golden.safetensors --gen 32 > /ws/oracle-out-k2.partial/$p.log 2>&1"; then
    echo "k2_oracle: $p FAILED"
    tail -20 "oracle-out-k2.partial/$p.log" 2>/dev/null
    rc=1
    continue
  fi
  for f in "$p.ids" "$p.log" "$p.golden.safetensors"; do   # the golden file last: it marks done
    mv -f "oracle-out-k2.partial/$f" "$data/oracle-out-k2/$f"
  done
  echo "k2_oracle: $p done ($(date '+%F %T'))"
done
if [ "${K2_HFCHECK:-0}" = 1 ] && [ -s "$data/oracle-out-k2/prose.golden.safetensors" ]; then
  cp "$data/oracle-out-k2/prose.ids" "$data/oracle-out-k2/prose.golden.safetensors" oracle-out-k2.partial/
  ORACLE_MODEL="$model" tools/oracle/run_in_container.sh \
    "python3 tools/oracle/k2_ref.py hfcheck \"\$SNAP\" --prompt /ws/oracle-out-k2.partial/prose.ids \
       --against /ws/oracle-out-k2.partial/prose.golden.safetensors" > "$data/oracle-out-k2/hfcheck.log" 2>&1
  echo "k2_oracle: hfcheck rc $? (not fatal):"; tail -5 "$data/oracle-out-k2/hfcheck.log"
fi
echo "--- the selection-gap distribution (B70_K2_TIE_TOL; 1e-3 proposed in k2_golden_test.cc)"
for p in prose code cjk; do
  grep -H -E 'selection gap|#==0' "$data/oracle-out-k2/$p.log" 2>/dev/null
done
ls -la "$data/oracle-out-k2"
exit "$rc"
