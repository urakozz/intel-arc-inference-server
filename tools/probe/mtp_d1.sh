#!/bin/bash
# Spec 8 D1/D2 (plan 8c Task 4): interleaved arms of `mtp_gpu_test --bench K`, one
# process per arm (K = 0 loads no head: today's server), 3 rounds in rotated order,
# every process under the shared GPU lock. From the repo root on the box:
#   tools/probe/detach.sh ~/mtp_d1.log tools/probe/mtp_d1.sh [greedy|sampled] [depth]
# Median per (K, prompt) is taken by tools/probe/mtp_d1_table.py over the log.
set -u
mode="${1:-greedy}"
depth="${2:-4096}"
snap="${SNAP:-$(grep -m1 "^B70_TEST_SNAPSHOT" build/CMakeCache.txt | cut -d= -f2)}"
flag=""
[ "$mode" = sampled ] && flag="--sampled"
for order in "0 3 1 2" "2 1 3 0" "0 3 1 2"; do
  for k in $order; do
    echo "ROUND order=[$order] k=$k $(date +%T) $(uptime | sed 's/.*load/load/')"
    flock "$HOME/b70-gpu.lock" build/tests/mtp_gpu_test "$snap" l0-int8 --bench "$k" $flag \
      --depth "$depth" 2>/dev/null | grep '^BENCH'
  done
done
