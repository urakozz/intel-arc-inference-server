#!/bin/bash
# ctest on the box's build dir with extra environment, device 0:
#   tools/probe/ctest_env.sh [VAR=value ...] -- <ctest args...>
# e.g. tools/probe/ctest_env.sh B70_PREFILL_ATTN=composed -- -R prefill_gate_l0_test
cd "$(dirname "$0")/../.."
envs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
exec env ZE_AFFINITY_MASK=0 "${envs[@]}" ctest --test-dir build --output-on-failure "$@"
