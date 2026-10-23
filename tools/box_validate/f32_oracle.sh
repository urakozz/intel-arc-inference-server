#!/usr/bin/env bash
# The Ornith / K2 token gates' JUDGE (operator ruling 2026-10-23), ON THE BOX CPU: the f32
# reference teacher-forced on the bf16 golden's own tokens, so its logits rows are the golden's
# contexts row for row (tests/golden/golden_common.h, `--judge=`).
#   tools/box_validate/f32_oracle.sh <data tree> ornith|k2
# Per prompt of prose / code / cjk whose <p>.golden.safetensors is not in
# <data tree>/oracle-out-<model>-f32 yet: the forced ids are <data tree>/oracle-out-<model>/<p>.golden
# .safetensors's `tokens`, then
#   ornith: tools/oracle/ornith_ref.py run --dtype f32 --force <ids> --gen 32  (tests/golden/prompts/<p>.ids)
#   k2:     tools/oracle/k2_ref.py run --mode f32 --force <ids> --gen 32       (oracle-out-k2/<p>.ids)
# into <tree>/oracle-out-<model>-f32.partial/ (box.sh sync leaves oracle-out-* alone), moved into the
# data tree once complete, so a killed run resumes at the prompt it was on.
# Measured 2026-10-23 (box, 22 threads): Ornith ~1.5 min a prompt, peak RSS 18-21 GiB; K2 ~2.5 min
# a prompt, peak RSS ~17 GiB. One CPU model job at a time: the whole run holds
# /tmp/b70-cpu-model.lock, and it exits 77 (SKIP) unless MemAvailable >= F32_REF_MIN_GB (95: the
# 70 GB the box keeps for the other jobs + this run's peak).
# Env: F32_REF_MIN_GB, ORACLE_THREADS (22), ORACLE_IMAGE, SNAP_ORNITH_CACHE / SNAP_K2_CACHE (the HF
# cache directory names; default the int4 checkpoints the engine runs).
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -eq 2 ] || { echo "usage: $0 <data tree> ornith|k2" >&2; exit 2; }
data="$1" m="$2"
case "$m" in
  ornith) model="${SNAP_ORNITH_CACHE:-models--urakozz--Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ}" ;;
  k2)     model="${SNAP_K2_CACHE:-models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ}" ;;
  *) echo "f32_oracle: model is ornith or k2, not $m" >&2; exit 2 ;;
esac
src="$data/oracle-out-$m" out="$data/oracle-out-$m-f32" part="oracle-out-$m-f32.partial"
need="${F32_REF_MIN_GB:-95}"
exec 9> /tmp/b70-cpu-model.lock
echo "f32_oracle: waiting for /tmp/b70-cpu-model.lock ($(date '+%F %T'))"
flock 9
avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
echo "f32_oracle: $m ($model) -> $out; MemAvailable ${avail} GB (need $need)"
if [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON f32_oracle: MemAvailable ${avail} GB < F32_REF_MIN_GB $need (another job holds the RAM)"
  exit 77
fi
mkdir -p "$out" "$part"
rc=0
for p in prose code cjk; do
  if [ -s "$out/$p.golden.safetensors" ]; then echo "f32_oracle: $p kept"; continue; fi
  [ -s "$src/$p.golden.safetensors" ] || { echo "f32_oracle: $src/$p.golden.safetensors is missing (the bf16 golden first)"; rc=1; continue; }
  python3 - "$src/$p.golden.safetensors" "$part/$p.force.ids" <<'EOF'
import json, struct, sys
import numpy as np
f = open(sys.argv[1], "rb")
n = struct.unpack("<Q", f.read(8))[0]
h = json.loads(f.read(n))
t = h["tokens"]
assert t["dtype"] == "I32", t
s, e = t["data_offsets"]
f.seek(8 + n + s)
ids = np.frombuffer(f.read(e - s), np.int32)
open(sys.argv[2], "w").write(" ".join(map(str, ids.tolist())) + "\n")
EOF
  if [ "$m" = ornith ]; then
    prompt="tests/golden/prompts/$p.ids"
    cmd="python3 tools/oracle/ornith_ref.py run \"\$SNAP\" --prompt $prompt --out /ws/$part/$p.golden.safetensors --gen 32 --dtype f32 --force /ws/$part/$p.force.ids"
  else
    cp "$src/$p.ids" "$part/$p.ids"
    cmd="python3 tools/oracle/k2_ref.py run \"\$SNAP\" --prompt /ws/$part/$p.ids --out /ws/$part/$p.golden.safetensors --gen 32 --mode f32 --force /ws/$part/$p.force.ids"
  fi
  echo "f32_oracle: $p start ($(date '+%F %T'))"
  if ! ORACLE_THREADS="${ORACLE_THREADS:-22}" ORACLE_MODEL="$model" tools/oracle/run_in_container.sh \
      "$cmd > /ws/$part/$p.log 2>&1"; then
    echo "f32_oracle: $p FAILED"; tail -20 "$part/$p.log" 2>/dev/null; rc=1; continue
  fi
  for f in "$p.force.ids" "$p.log" "$p.golden.safetensors"; do   # the golden file last: it marks done
    mv -f "$part/$f" "$out/$f"
  done
  [ "$m" = k2 ] && mv -f "$part/$p.ids" "$out/$p.ids"
  echo "f32_oracle: $p done ($(date '+%F %T'))"; tail -2 "$out/$p.log"
done
ls -la "$out"
exit "$rc"
