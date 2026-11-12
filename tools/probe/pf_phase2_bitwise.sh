#!/bin/bash
# Phase 2: end-to-end bitwise evidence for the bitwise tasks - each Ornith prefill gate test run verbose with
# today's kernels (B70_PF_PHASE2=old) and the branch's, the two logs diffed after dropping timings and load lines
# (every printed logit cosine, gdn_state cosine and token is 9 digits; a bitwise walk prints the same file).
# ON THE BOX under a card lock: tools/gpu_lock.sh card 1 -- tools/probe/pf_phase2_bitwise.sh <outdir> [test...]
set -u
OUT="$1"; shift
TESTS="${*:-prefill_gate_ornith_l0_test prefill_gate_ornith_l0-int8_test prefill_gate_ornith_l0_c16_test}"
mkdir -p "$OUT"
for t in $TESTS; do
  for v in old new; do
    B70_PF_PHASE2=$v ctest --test-dir "${BUILD:-build}" -V -R "^$t\$" 2>&1 |
      grep -vE 'sec$|Test time|load +[0-9]|repacked in|Start|Test #|tests passed|Total Test|^[0-9]+: Test command|UpdateCTest|Constructing|Checking test|Test project|Parse|Running|Updating|ms\b|t/s' > "$OUT/$t.$v.log"
  done
  if diff -q "$OUT/$t.old.log" "$OUT/$t.new.log" > /dev/null; then echo "BITWISE $t: identical ($(wc -l < "$OUT/$t.new.log") lines)"
  else echo "DIFFER $t: $(diff "$OUT/$t.old.log" "$OUT/$t.new.log" | grep -c '^[<>]') lines"; fi
done
