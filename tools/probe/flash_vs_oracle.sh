#!/bin/bash
# Spec 6 K3 diagnosis: which of flash and composed is closer to the CPU oracle? Run ON THE
# BOX from the repo root. The golden gate (prose, code, cjk; 32 greedy steps each) in both
# modes on l0 and l0-int8; prints, per mode, the mean and the worst of the per-step
# logit cosine against the oracle, and the mean GDN-state layer cosine after the prefill.
out=/tmp/flash_vs_oracle
mkdir -p $out
for b in "l0" "l0-int8 1"; do
  for m in composed flash; do
    tag="$m-${b%% *}"
    # shellcheck disable=SC2086
    ZE_AFFINITY_MASK=0 B70_PREFILL_ATTN=$m ./build/tests/prefill_gate_test oracle-out-primary \
      tests/golden/prompts urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ prose,code,cjk 0 $b \
      > $out/$tag.log 2>&1
    awk -v tag="$tag" '
      /^ +[0-9]+ +[0-9]+ +[0-9]+ +(yes|TIE|no|NO)/ { n++; s += $5; if (min == "" || $5 < min) min = $5; if (1 - $5 > 0) l += log(1 - $5) }
      /^ +[0-9]+ +0\.[0-9]+ +[0-9.e+-]+ +[0-9.]+ +[0-9.]+$/ { g++; gs += $2 }
      END { printf "%-18s logit-cos vs oracle: mean %.9f, worst %.9f, geo-mean(1-cos) %.3e over %d steps; gdn layer cos mean %.9f (%d)\n", tag, s/n, min, exp(l/n), n, gs/g, g }' $out/$tag.log
  done
done
