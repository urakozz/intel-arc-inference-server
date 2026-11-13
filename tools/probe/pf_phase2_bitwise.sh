#!/bin/bash
# Phase 2: end-to-end bitwise evidence for a bitwise task - each test run verbose from a control build (CTRL, e.g.
# main's tree) and a candidate build (CAND), the two logs diffed after dropping timings, load, memory and
# launch-count lines (every printed logit / state cosine is 9 digits and every token is printed; a bitwise walk
# prints the same file). ON THE BOX under a card lock:
#   CTRL=<build dir> CAND=<build dir> tools/gpu_lock.sh card <N> -- tools/probe/pf_phase2_bitwise.sh <outdir> test...
set -u
OUT="$1"; shift
: "${CTRL:?CTRL=<the control build directory>}" "${CAND:?CAND=<the candidate build directory>}"
mkdir -p "$OUT"
for t in "$@"; do
  for v in ctrl cand; do
    dir=$CTRL; [ $v = cand ] && dir=$CAND
    ctest --test-dir "$dir" -V -R "^$t\$" 2>&1 |
      grep -vE 'sec$|Test time|load +[0-9]|repacked in|Start|Test #|tests passed|Total Test|^[0-9]+: Test command|UpdateCTest|Constructing|Checking test|Test project|Parse|Running|Updating|ms\b|t/s|sec.proc|MemAvailable|launches|Working Directory' |
      sed -E 's/^[0-9]+: //; s#/b70/trees/[^/]+/#/b70/trees/<tree>/#g' > "$OUT/$t.$v.log"   # the build's own tree in paths
  done
  if diff -q "$OUT/$t.ctrl.log" "$OUT/$t.cand.log" > /dev/null; then echo "BITWISE $t: identical ($(wc -l < "$OUT/$t.cand.log") lines)"
  else echo "DIFFER $t: $(diff "$OUT/$t.ctrl.log" "$OUT/$t.cand.log" | grep -c '^[<>]') lines"; fi
done
