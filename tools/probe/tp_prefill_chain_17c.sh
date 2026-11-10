#!/bin/bash
# Spec 17c: the timing diagnosis (tp_prefill_test time: with / without crossings) at 2048 and 16384 ids, then row
# 80's correctness tests in order - under ONE whole-box lock:
#   tools/gpu_lock.sh box -- tools/probe/tp_prefill_chain_17c.sh [time|tests|cli|all]
M=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
what=${1:-all}
if [ "$what" = time ] || [ "$what" = all ]; then
  for n in 2048 16384; do build/tests/tp_prefill_test $M time $n 2>&1 | grep -E '^time|error|what'; done
fi
if [ "$what" = tests ] || [ "$what" = all ]; then
  ctest --test-dir build -R "${TESTS:-^(tp_prefill_test|tp_prefill_kv8_test|tp_decode_test|tp_decode_kv8_test|tp_fail_test)$}" \
    --output-on-failure 2>&1 | grep -vE '^(load|  [a-z])' | tail -150
fi
if [ "$what" = cli ] || [ "$what" = all ]; then   # r80.cli by hand
  D=${TMPDIR:-/tmp}/b70-17c-cli; mkdir -p "$D"
  build/src/cli/b70-decode $M --ids tests/golden/prompts/long.ids --n 64 --prefill --parallel 1 > "$D/one.ids"; echo "one rc=$?"
  build/src/cli/b70-decode $M --ids tests/golden/prompts/long.ids --n 64 --prefill --parallel tp > "$D/tp.ids" 2> "$D/tp.err"; echo "tp rc=$?"
  cmp "$D/one.ids" "$D/tp.ids" && echo "cli: ids identical" || echo "cli: ids differ (tp_prefill_test judges)"
  grep -E '^(engine|memory|prefill|pp|generate):' "$D/tp.err"
  ctest --test-dir build -R '^cli_reject_tp_' 2>&1 | grep -E 'tests passed|Failed'
fi
