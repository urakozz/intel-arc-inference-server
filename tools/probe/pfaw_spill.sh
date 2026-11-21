#!/bin/bash
# Spec 38a Task 1: the spill line of each probe arm of pf_flash_attn.cl's PAIR / WIDE forms (tools/probe/CMakeLists.txt's
# pfas_* / pfaw* / pfaq_* binaries), each compiled alone with the build's ocloc and flags (a parallel build interleaves
# the warnings, so the build log cannot attribute them). CPU only, no GPU lock. Run on the box from the tree's root:
#   bash tools/probe/pfaw_spill.sh
cd "$(dirname "$0")/../.."
OC=$(grep OCLOC_EXECUTABLE: build/CMakeCache.txt | cut -d= -f2)
DEV=$(grep B70_OCLOC_DEVICE build/CMakeCache.txt | head -1 | cut -d= -f2)
OUT=$HOME/b70/tmp/perf112/spill
mkdir -p "$OUT"
C="-DKT=64 -DQREG=0 -DEXP2=1"
ARMS=(
  "pfal_H2|$C -DRPW=16 -DHPW=2 -DPAIR=1"
  "pfas_pair_H2|$C -DRPW=16 -DHPW=2 -DPAIR=1 -DFA_SKELETON=1"
  "pfaq_pair_H2|$C -DRPW=16 -DHPW=2 -DPAIR=1 -DPAIR_QSLM=4"
  "pfaw_H1|$C -DRPW=32 -DHPW=1 -DWIDE=1 -DWIDE_AHEAD=1"
  "pfaw_H2|$C -DRPW=32 -DHPW=2 -DWIDE=1 -DWIDE_AHEAD=1"
  "pfaw0_H1|$C -DRPW=32 -DHPW=1 -DWIDE=1 -DWIDE_AHEAD=0"
  "pfawq1_H1|$C -DRPW=32 -DHPW=1 -DWIDE=1 -DWIDE_AHEAD=1 -DWIDE_Q1=1"
  "pfas_wide_H1|$C -DRPW=32 -DHPW=1 -DWIDE=1 -DWIDE_AHEAD=1 -DFA_SKELETON=1"
  "pfas_wide0_H1|$C -DRPW=32 -DHPW=1 -DWIDE=1 -DWIDE_AHEAD=0 -DFA_SKELETON=1"
)
for arm in "${ARMS[@]}"; do
  n=${arm%%|*}
  l=$("$OC" compile -file src/kernels/prefill/pf_flash_attn.cl -device "${DEV:-bmg-g31}" -output "$n" \
        -output_no_suffix -out_dir "$OUT" \
        -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -Isrc/kernels ${arm#*|} -cl-intel-256-GRF-per-thread" \
        2>&1 | grep -iE "spill|error" | tail -1)
  echo "$n: ${l:-no spill line}"
done
