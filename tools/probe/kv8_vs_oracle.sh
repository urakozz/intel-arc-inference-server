#!/bin/bash
# Spec 12b Q2/Q3 on the golden set: the int8 KV cache no further from the CPU oracle than
# the bf16 one, less a tolerance. Registered as ctest `kv8_vs_oracle_test`; run from the repo
# root on the box (build in ./build, golden set in ./oracle-out-primary) - flash_vs_oracle.sh's
# arrangement, with B70_KV_CACHE in place of B70_PREFILL_ATTN.
#
# The golden gate (prose, code, cjk; 32 greedy steps each = 96 decision rows) with bf16 KV and
# with int8 KV on l0-int8 and l0, flash attention. Per run: the mean and worst per-step logit
# cosine against the oracle. Bar, per backend: mean(int8) >= mean(bf16) - tol, tol = 1e-4
# (confirmed by the Qwen3.8 repeat of 12a: rotkv's golden 1 - cos against bf16 KV is 3.3e-5 /
# 5.0e-5 / 3.3e-5 on prose / code / cjk, docs/probe-int8-kv-qwen38-2026-10-06.md), and
# every gate run itself passes (the tie rule's allowance unchanged, spec 12 Q2).
# Exit 0 iff both hold on both backends.
cd "$(dirname "$0")/../.."
out=/tmp/kv8_vs_oracle
mkdir -p $out
rc=0
tol=${KV8_TOL:-1e-4}
for b in "l0-int8 1" "l0"; do
  bn=${b%% *}
  for kv in bf16 int8; do
    # shellcheck disable=SC2086
    ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} B70_PREFILL_ATTN=flash B70_KV_CACHE=$kv ./build/tests/prefill_gate_test \
      oracle-out-primary tests/golden/prompts urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
      prose,code,cjk 0 $b > $out/$kv-$bn.log 2>&1 || { echo "$kv $bn: the golden gate FAILED ($out/$kv-$bn.log)"; rc=1; }
    awk -v tag="$kv-$bn" '
      /^ +[0-9]+ +[0-9]+ +[0-9]+ +(yes|TIE|no|NO)/ { n++; s += $5; if (min == "" || $5 < min) min = $5 }
      END { printf "%-14s logit-cos vs oracle: mean %.9f, worst %.9f over %d steps\n", tag, s/n, min, n
            printf "%.12f %d\n", s/n, n > "/dev/stderr" }' $out/$kv-$bn.log 2> $out/$kv-$bn.mean
  done
  read -r km kn < $out/int8-$bn.mean
  read -r bm bnn < $out/bf16-$bn.mean
  if [ "${kn:-0}" -ne 96 ] || [ "${bnn:-0}" -ne 96 ]; then
    echo "$bn: expected 96 decision rows per form, got int8 ${kn:-0} bf16 ${bnn:-0}"; rc=1; continue
  fi
  if awk -v k="$km" -v f="$bm" -v t="$tol" 'BEGIN { exit !(k >= f - t) }'; then
    echo "$bn: Q2/Q3 golden PASS, mean int8 KV $km >= bf16 KV $bm - $tol"
  else
    echo "$bn: Q2/Q3 golden FAIL, mean int8 KV $km < bf16 KV $bm - $tol"; rc=1
  fi
done
exit $rc
