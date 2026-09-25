#!/usr/bin/env bash
# Spec 6 P0: today's attention costs. Diagnostic rows (profiled runs add waits);
# the unprofiled rows are the bench numbers. Run from the Mac, on an idle box.
#
# Deviations from the plan's draft, each forced by the tools as they are:
#   * The profiled run is a direct `tools/box.sh run`, not bench_decode.sh:
#     bench_decode.sh forwards only ZE_AFFINITY_MASK and B70_GIT_SHA over ssh,
#     so B70_PREFILL_PROFILE=1 set on the Mac never reached the box.
#   * The phase names are profile.cc's: attn_QK^T, attn_softmax, attn_PV.
#   * pp16384 does not fit: `--bench` refuses depth + tg > max_len, and max_len
#     is 16384 (the only compiled decode attention). The deep rows are pp14336
#     (7 full chunks) and pp16128 (the deepest that fits with tg 256).
set -euo pipefail
cd "$(dirname "$0")/../.."
export ZE_AFFINITY_MASK=0
MODEL="${MODEL:-urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}"
tools/box.sh sync
echo "## pp4096, profiled (attention phases: attn_QK^T, attn_softmax, attn_PV)"
tools/box.sh run "ZE_AFFINITY_MASK=0 B70_PREFILL_PROFILE=1 ./build/src/cli/b70-decode '$MODEL' --bench --pp 4096 --tg 256 2>&1" | \
  grep -E "attn_(prep|QK|softmax|PV|gate)|phase +wait_ms| pp \|" || true
for pp in 4096 8192 14336 16128; do
  echo "## pp$pp, unprofiled, median of 3"
  tools/bench_decode.sh --pp "$pp" --runs 3 --no-build 2>&1 | grep -E " pp \||pp median"
done
for d in 4096 16000; do
  echo "## decode at depth $d, max_len 16384, median of 3"
  tools/bench_decode.sh --depth "$d" --runs 3 --no-build 2>&1 | grep -E "^\| b70-decode|^median"
done
