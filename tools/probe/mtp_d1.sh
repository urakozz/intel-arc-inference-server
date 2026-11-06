#!/bin/bash
# Spec 8 D1/D2 (plan 8c Task 4): interleaved arms of `mtp_gpu_test --bench K`, one
# process per arm (K = 0 loads no head: today's server), 3 rounds in rotated order,
# every process under the shared GPU lock. From the repo root on the box:
#   tools/probe/detach.sh ~/b70/logs/probe/mtp_d1.log tools/probe/mtp_d1.sh [greedy|sampled] [depth]
# Median per (K, prompt) is taken by tools/probe/mtp_d1_table.py over the log.
# MAX_LEN (spec 8 §12): the engine length, passed as --max-len (A12's 32k / 64k rows need
# 65536 / 131072); unset, mtp_gpu_test's default 16384 as before.
# VATTN=1 (spec 28c, plan 28c Task 7): every K > 0 arm runs as two processes, the verify attention forced to v2
# (--vattn v2) and the capture's rule (--vattn rule), their order alternating by round; BENCH lines carry vattn=.
# LOCK: the command each process runs under (default: the whole-box flock, as before; e.g.
# LOCK="tools/gpu_lock.sh card 1 --" for one card beside a job on the other).
set -u
mode="${1:-greedy}"
depth="${2:-4096}"
snap="${SNAP:-$(grep -m1 "^B70_TEST_SNAPSHOT" build/CMakeCache.txt | cut -d= -f2)}"
flag=""
[ "$mode" = sampled ] && flag="--sampled"
lock="${LOCK:-flock $HOME/b70-gpu.lock}"
round=0
for order in "0 3 1 2" "2 1 3 0" "0 3 1 2"; do
  round=$((round + 1))
  for k in $order; do
    arms="rule"
    if [ "${VATTN:-0}" = 1 ] && [ "$k" -gt 0 ]; then
      if [ $((round % 2)) = 1 ]; then arms="v2 rule"; else arms="rule v2"; fi
    fi
    for va in $arms; do
      echo "ROUND order=[$order] k=$k vattn=$va $(date +%T) $(uptime | sed 's/.*load/load/')"
      # shellcheck disable=SC2086 # $lock, $flag and the --max-len pair are word lists
      $lock build/tests/mtp_gpu_test "$snap" l0-int8 --bench "$k" $flag \
        --depth "$depth" ${MAX_LEN:+--max-len $MAX_LEN} --vattn "$va" 2>/dev/null | grep '^BENCH'
    done
  done
done
