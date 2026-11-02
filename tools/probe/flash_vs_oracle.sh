#!/bin/bash
# Spec 6 K3a, as the operator redefined it on 2026-09-26: flash is no further from the CPU
# oracle than composed, per backend. Registered as ctest `flash_vs_oracle_test`; run from the
# repo root on the box (build in ./build).
#   tools/probe/flash_vs_oracle.sh [oracle dir] [snapshot]
# ctest passes B70_TEST_ORACLE_DIR and B70_TEST_SNAPSHOT, as the other golden registrations
# do; by hand the defaults are ./oracle-out-primary and the checkpoint's repo id (refs/main).
#
# The golden gate (prose, code, cjk; 32 greedy steps each = 96 decision rows) in both
# attention modes on l0 and l0-int8. Per run: the mean and worst per-step logit cosine
# against the oracle, and the mean GDN-state layer cosine after the prefill (diagnostic).
# Bar, per backend: mean(flash) >= mean(composed) - tol, and every gate run itself passes.
# tol is 1e-6, except l0 at 1e-5: the operator accepted exp2's 7.6e-6 miss there on
# 2026-09-27 (spec 6 §9). l0-int8, the default, keeps 1e-6.
# Exit 0 iff both hold on both backends; 77 (ctest SKIP, as golden_gate_test) when the gate
# reports the golden set absent - a tree without the oracle-out* links
# (tools/box_validate/data.sh link); 1 otherwise. PASS is printed only from 96 parsed rows
# per mode and two gate runs that exited 0.
cd "$(dirname "$0")/../.."
oracle=${1:-oracle-out-primary}
snap=${2:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}
# Scratch in /tmp, as every tools/probe/*.sh keeps its own.
out=/tmp/flash_vs_oracle
mkdir -p $out
num='^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$'
rc=0
for b in "l0" "l0-int8 1"; do
  bn=${b%% *}
  tol=1e-6; [ "$bn" = l0 ] && tol=1e-5
  gate_ok=1
  for m in composed flash; do
    : > $out/$m-$bn.mean
    # shellcheck disable=SC2086
    ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} B70_PREFILL_ATTN=$m ./build/tests/prefill_gate_test \
      "$oracle" tests/golden/prompts "$snap" \
      prose,code,cjk 0 $b > $out/$m-$bn.log 2>&1
    grc=$?
    if [ $grc -eq 77 ]; then
      grep -m1 '^SKIP' $out/$m-$bn.log
      echo "flash_vs_oracle: SKIP - the golden gate found no golden set in $oracle ($out/$m-$bn.log)"
      exit 77
    fi
    [ $grc -eq 0 ] || { echo "$m $bn: the golden gate FAILED, exit $grc ($out/$m-$bn.log)"; gate_ok=0; rc=1; }
    awk -v tag="$m-$bn" -v mf="$out/$m-$bn.mean" '
      /^ +[0-9]+ +[0-9]+ +[0-9]+ +(yes|TIE|no|NO)/ { n++; s += $5; if (min == "" || $5 < min) min = $5 }
      /^ +[0-9]+ +0\.[0-9]+ +[0-9.e+-]+ +[0-9.]+ +[0-9.]+$/ { g++; gs += $2 }
      END { if (n == 0) { printf "%-18s no decision rows in the log\n", tag; print "none 0" > mf; exit }
            printf "%-18s logit-cos vs oracle: mean %.9f, worst %.9f over %d steps; gdn layer cos mean %.9f (%d)\n", tag, s/n, min, n, g ? gs/g : 0, g
            printf "%.12f %d\n", s/n, n > mf }' $out/$m-$bn.log
  done
  fm=; fn=; cm=; cn=
  read -r fm fn < $out/flash-$bn.mean
  read -r cm cn < $out/composed-$bn.mean
  if [ "$fn" != 96 ] || [ "$cn" != 96 ]; then
    echo "$bn: K3a FAIL, expected 96 decision rows per mode, got flash ${fn:-none} composed ${cn:-none}"; rc=1; continue
  fi
  if ! [[ "$fm" =~ $num && "$cm" =~ $num ]]; then
    echo "$bn: K3a FAIL, unparsable means: flash '$fm' composed '$cm'"; rc=1; continue
  fi
  if [ $gate_ok -ne 1 ]; then
    echo "$bn: K3a FAIL, a golden gate run failed (means flash $fm, composed $cm)"; rc=1
  elif awk -v f="$fm" -v c="$cm" -v t="$tol" 'BEGIN { exit !(f + 0 >= c - t) }'; then
    echo "$bn: K3a PASS, mean flash $fm >= composed $cm - $tol"
  else
    echo "$bn: K3a FAIL, mean flash $fm < composed $cm - $tol"; rc=1
  fi
done
exit $rc
