#!/bin/bash
# Spec 6 P1's timed sweep, run ON THE BOX and detached (the WiFi drops long ssh
# sessions): every arm against the composed path at pos 2048 / 16384 / 30720, C 2048.
# Log: /tmp/pfa_sweep.log, ending in ALLDONE. Poll it; the GPU must be idle.
cd "$(dirname "$0")/../.."
setsid nohup sh -c '
  for p in 2048 16384 30720; do
    ZE_AFFINITY_MASK=0 ./build/tools/probe/probe_flash_attn $p 2048 --arms all --time
  done
  echo ALLDONE' > /tmp/pfa_sweep.log 2>&1 < /dev/null &
echo "started, log /tmp/pfa_sweep.log"
