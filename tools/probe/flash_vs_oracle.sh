#!/bin/bash
# Spec 6 K3a, as the operator redefined it on 2026-09-26: flash is no further from the CPU
# oracle than composed, per backend. Registered as ctest `flash_vs_oracle_test`; run from the
# repo root on the box (build in ./build, golden set in ./oracle-out-primary).
#
# The golden gate (prose, code, cjk; 32 greedy steps each = 96 decision rows) in both
# attention modes on l0 and l0-int8. Per run: the mean and worst per-step logit cosine
# against the oracle, and the mean GDN-state layer cosine after the prefill (diagnostic).
# Bar, per backend: mean(flash) >= mean(composed) - 1e-6, and every gate run itself passes.
# Exit 0 iff both hold on both backends.
cd "$(dirname "$0")/../.."
out=/tmp/flash_vs_oracle
mkdir -p $out
rc=0
for b in "l0" "l0-int8 1"; do
  bn=${b%% *}
  for m in composed flash; do
    # shellcheck disable=SC2086
    ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} B70_PREFILL_ATTN=$m ./build/tests/prefill_gate_test \
      oracle-out-primary tests/golden/prompts urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
      prose,code,cjk 0 $b > $out/$m-$bn.log 2>&1 || { echo "$m $bn: the golden gate FAILED ($out/$m-$bn.log)"; rc=1; }
    awk -v tag="$m-$bn" '
      /^ +[0-9]+ +[0-9]+ +[0-9]+ +(yes|TIE|no|NO)/ { n++; s += $5; if (min == "" || $5 < min) min = $5 }
      /^ +[0-9]+ +0\.[0-9]+ +[0-9.e+-]+ +[0-9.]+ +[0-9.]+$/ { g++; gs += $2 }
      END { printf "%-18s logit-cos vs oracle: mean %.9f, worst %.9f over %d steps; gdn layer cos mean %.9f (%d)\n", tag, s/n, min, n, gs/g, g
            printf "%.12f %d\n", s/n, n > "/dev/stderr" }' $out/$m-$bn.log 2> $out/$m-$bn.mean
  done
  read -r fm fn < $out/flash-$bn.mean
  read -r cm cn < $out/composed-$bn.mean
  if [ "${fn:-0}" -ne 96 ] || [ "${cn:-0}" -ne 96 ]; then
    echo "$bn: expected 96 decision rows per mode, got flash ${fn:-0} composed ${cn:-0}"; rc=1; continue
  fi
  if awk -v f="$fm" -v c="$cm" 'BEGIN { exit !(f >= c - 1e-6) }'; then
    echo "$bn: K3a PASS, mean flash $fm >= composed $cm - 1e-6"
  else
    echo "$bn: K3a FAIL, mean flash $fm < composed $cm - 1e-6"; rc=1
  fi
done
exit $rc
