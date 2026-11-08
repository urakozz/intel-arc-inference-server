#!/bin/bash
# Spec 12b Q2/Q3 on the golden set: the int8 KV cache no further from the CPU oracle than
# the bf16 one, less a tolerance. Registered as ctest `kv8_vs_oracle_test`; run from the repo
# root on the box (build in ./build), B70_KV_CACHE selecting the cache.
#   tools/probe/kv8_vs_oracle.sh [oracle dir] [snapshot]
# ctest passes B70_TEST_ORACLE_DIR and B70_TEST_SNAPSHOT, as the other golden registrations
# do; by hand the defaults are ./oracle-out-primary and the checkpoint's repo id (refs/main).
#
# The golden gate (prose, code, cjk; 32 greedy steps each = 96 decision rows) with bf16 KV and
# with int8 KV on l0-int8 and l0, flash attention. Per run: the mean and worst per-step logit
# cosine against the oracle. Bar, per backend: mean(int8) >= mean(bf16) - tol, tol = 1e-4
# (confirmed by the Qwen3.8 repeat of 12a: rotkv's golden 1 - cos against bf16 KV is 3.3e-5 /
# 5.0e-5 / 3.3e-5 on prose / code / cjk, docs/probes/probe-int8-kv-qwen38-2026-10-06.md), and
# every gate run itself passes (the tie rule's allowance unchanged, spec 12 Q2).
# Exit 0 iff both hold on both backends; 77 (ctest SKIP, as golden_gate_test) when the gate
# reports the golden set absent - a tree without the oracle-out* links
# (tools/box_validate/data.sh link); 1 otherwise. PASS is printed only from 96 parsed rows
# per form and two gate runs that exited 0.
cd "$(dirname "$0")/../.."
oracle=${1:-oracle-out-primary}
snap=${2:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}
# Keep the four gate logs together; detached box jobs supply their own TMPDIR.
out=${TMPDIR:-/tmp}/kv8_vs_oracle
mkdir -p $out
num='^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$'
rc=0
tol=${KV8_TOL:-1e-4}
for b in "l0-int8 1" "l0"; do
  bn=${b%% *}
  gate_ok=1
  for kv in bf16 int8; do
    : > $out/$kv-$bn.mean
    # shellcheck disable=SC2086
    ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} B70_KV_CACHE=$kv ./build/tests/prefill_gate_test \
      "$oracle" tests/golden/prompts "$snap" \
      prose,code,cjk 0 $b > $out/$kv-$bn.log 2>&1
    grc=$?
    if [ $grc -eq 77 ]; then
      grep -m1 '^SKIP' $out/$kv-$bn.log
      echo "kv8_vs_oracle: SKIP - the golden gate found no golden set in $oracle ($out/$kv-$bn.log)"
      exit 77
    fi
    [ $grc -eq 0 ] || { echo "$kv $bn: the golden gate FAILED, exit $grc ($out/$kv-$bn.log)"; gate_ok=0; rc=1; }
    awk -v tag="$kv-$bn" -v mf="$out/$kv-$bn.mean" '
      /^ +[0-9]+ +[0-9]+ +[0-9]+ +(yes|TIE|no|NO)/ { n++; s += $5; if (min == "" || $5 < min) min = $5 }
      END { if (n == 0) { printf "%-14s no decision rows in the log\n", tag; print "none 0" > mf; exit }
            printf "%-14s logit-cos vs oracle: mean %.9f, worst %.9f over %d steps\n", tag, s/n, min, n
            printf "%.12f %d\n", s/n, n > mf }' $out/$kv-$bn.log
  done
  km=; kn=; bm=; bnn=
  read -r km kn < $out/int8-$bn.mean
  read -r bm bnn < $out/bf16-$bn.mean
  if [ "$kn" != 96 ] || [ "$bnn" != 96 ]; then
    echo "$bn: Q2/Q3 golden FAIL, expected 96 decision rows per form, got int8 ${kn:-none} bf16 ${bnn:-none}"; rc=1; continue
  fi
  if ! [[ "$km" =~ $num && "$bm" =~ $num ]]; then
    echo "$bn: Q2/Q3 golden FAIL, unparsable means: int8 KV '$km' bf16 KV '$bm'"; rc=1; continue
  fi
  if [ $gate_ok -ne 1 ]; then
    echo "$bn: Q2/Q3 golden FAIL, a golden gate run failed (means int8 KV $km, bf16 KV $bm)"; rc=1
  elif awk -v k="$km" -v f="$bm" -v t="$tol" 'BEGIN { exit !(k + 0 >= f - t) }'; then
    echo "$bn: Q2/Q3 golden PASS, mean int8 KV $km >= bf16 KV $bm - $tol"
  else
    echo "$bn: Q2/Q3 golden FAIL, mean int8 KV $km < bf16 KV $bm - $tol"; rc=1
  fi
done
exit $rc
