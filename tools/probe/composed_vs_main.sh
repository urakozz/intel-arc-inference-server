#!/bin/bash
# Spec 6 (plan 6b Task 2): is the composed path bitwise what it was? Run ON THE BOX from
# the repo root, with a build of the base commit at ~/b70-main/build (tools/box.sh build
# with REMOTE_DIR=b70-main from an export of that commit). Runs the golden gate on l0 and
# l0-int8, and the multi-chunk c16 pair, in both builds (this one with
# B70_PREFILL_ATTN=composed), and diffs every printed number (timing lines dropped).
# Identical output means identical logits to the printed digits on every row.
here=$(pwd)
main=$HOME/b70-main
[ -e "$main/oracle-out-primary" ] || ln -s "$here/oracle-out-primary" "$main/oracle-out-primary"
out=/tmp/composed_vs_main
mkdir -p $out
rc=0
for args in "prose,code,cjk 0 l0" "prose,code,cjk 0 l0-int8 1" "prose,code,cjk 16 l0" "prose,code,cjk 16 l0-int8 1"; do
  tag=$(echo "$args" | tr ' ,' '__')
  # shellcheck disable=SC2086
  (cd "$main" && ZE_AFFINITY_MASK=0 ./build/tests/prefill_gate_test oracle-out-primary tests/golden/prompts \
     urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ $args) > $out/main_$tag.log 2>&1
  # shellcheck disable=SC2086
  ZE_AFFINITY_MASK=0 B70_PREFILL_ATTN=composed ./build/tests/prefill_gate_test oracle-out-primary \
     tests/golden/prompts urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ $args > $out/comp_$tag.log 2>&1
  if diff <(grep -vE " ms|t/s|seconds|load" $out/main_$tag.log) <(grep -vE " ms|t/s|seconds|load" $out/comp_$tag.log) > $out/diff_$tag.txt; then
    echo "$args: IDENTICAL ($(wc -l < $out/comp_$tag.log) lines)"
  else
    echo "$args: DIFFER ($(wc -l < $out/diff_$tag.txt) diff lines, $out/diff_$tag.txt)"; rc=1
  fi
done
exit $rc
