#!/usr/bin/env bash
# Regenerates the three golden dumps, one after another (61 GiB peak each), from
# the repo root ON THE BOX - the Mac never loads the model. dump.py takes one
# --prompt/--out pair, so this is three container invocations and the model
# reloads (~2 min) per prompt; ~17 min serial. No numerics are touched here.
#
#   tools/box.sh sync && ssh <box>
#   cd ~/b70-inference-server
#   setsid nohup tools/oracle/golden.sh > oracle-out/golden.log 2>&1 </dev/null &
#
# (detached, so the run outlives the ssh session). Prompt ids must already exist
# as oracle-out/<p>.ids - see "Running it" in tools/oracle/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p oracle-out
for p in prose code cjk; do
  echo "=== $p  start $(date -Is)"
  /usr/bin/time -v tools/oracle/run_in_container.sh \
    'python3 tools/oracle/dump.py "$SNAP" --prompt /ws/oracle-out/'"$p"'.ids \
       --out /ws/oracle-out/'"$p"'.golden.safetensors --gen 32' \
    > oracle-out/"$p".log 2>&1
  echo "=== $p  done  $(date -Is)  rc=$?"
  tail -3 oracle-out/"$p".log
done
echo "=== ALL DONE $(date -Is)"
