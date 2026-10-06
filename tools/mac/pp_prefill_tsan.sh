#!/bin/bash
# tools/mac/pp_prefill_tsan.sh - the pipeline-parallel protocol tests under ThreadSanitizer,
# on the Mac (or any host with clang): tests/runtime/pp_prefill_protocol_test.cc (spec 16c's
# prefill order on two host threads) and tests/runtime/pp_protocol_test.cc (spec 16b's peer
# hand-off). Each test is rebuilt from its source with -fsanitize=thread against the host
# build's planner archive (cmake --preset mac-host builds it), so every per-chunk resource
# the order hands between the "devices" is checked for a happens-before edge. Exit 0 when
# both run clean (ThreadSanitizer exits 66 on a report).
#
#   tools/mac/pp_prefill_tsan.sh [build dir, default build/mac-host]
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
build=${1:-$root/build/mac-host}
plan=$build/src/runtime/libb70_plan.a
model=$build/src/model/libb70_model.a
if [ ! -f "$plan" ] || [ ! -f "$model" ]; then
  echo "pp_prefill_tsan: no $plan / $model - cmake --preset mac-host && cmake --build --preset mac-host first" >&2
  exit 2
fi
out=$build/tsan
mkdir -p "$out"
cxx=${CXX:-clang++}
for t in pp_prefill_protocol_test pp_protocol_test; do
  "$cxx" -std=c++17 -O1 -g -fsanitize=thread -Wall -Wextra -Werror \
    -I"$root/src" -I"$root/tests" "$root/tests/runtime/$t.cc" "$plan" "$model" -o "$out/$t"
  echo "== $t (ThreadSanitizer)"
  TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1" "$out/$t"
done
echo "pp_prefill_tsan: both clean"
