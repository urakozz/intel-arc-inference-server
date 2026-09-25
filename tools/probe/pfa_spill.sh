#!/bin/bash
# Per-arm spill lines for probe_flash_attn.cl: compiles each arm alone with the build's
# ocloc and flags (a parallel build interleaves the warnings, so the build log cannot
# attribute them). Run on the box: bash tools/probe/pfa_spill.sh ["KT RPW HPW QREG" ...]
cd "$(dirname "$0")/../.."
OC=$(grep OCLOC_EXECUTABLE: build/CMakeCache.txt | cut -d= -f2)
DEV=$(grep B70_OCLOC_DEVICE build/CMakeCache.txt | head -1 | cut -d= -f2)
mkdir -p /tmp/pfa_spill
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=("32 16 6 1" "32 32 6 1" "64 16 6 1" "32 16 3 1" "32 16 1 1" "32 16 6 0" "64 16 6 0" "32 32 3 1")
for arm in "${ARMS[@]}"; do
  set -- $arm
  n=pfa_KT$1_R$2_H$3_Q$4
  l=$("$OC" compile -file tools/probe/probe_flash_attn.cl -device "${DEV:-bmg-g31}" -output "$n" \
        -output_no_suffix -out_dir /tmp/pfa_spill \
        -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -DKT=$1 -DRPW=$2 -DHPW=$3 -DQREG=$4 -cl-intel-256-GRF-per-thread" \
        2>&1 | grep -E "spill|error" | tail -1)
  echo "$n: ${l:-no spill line}"
done
