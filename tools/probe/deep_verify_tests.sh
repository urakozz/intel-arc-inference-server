#!/usr/bin/env bash
# Spec 32 / 32e: the deep verify lists' checks on one card, from the tree root after the build - the kernel tests (every
# 32- / 64-row build bitwise the 16-row one), the 16-row and head lists over the new pending layout, the 32- and 64-row
# lists bitwise the 16-row list (mtp_verify_deep_test), each test under its own card lock. Run detached; one log per test under OUT.
#   [TESTS="..."] tools/probe/deep_verify_tests.sh CARD OUT
set -u
card="$1" out="$2"
mkdir -p "$out"
for t in ${TESTS:-argmax_test gdn_step_slots_test gemm_sm_test attn_verify_dpas_test mtp_verify_wide_test mtp_verify_deep_test mtp_verify_test}; do
  echo "== $t $(date '+%F %T')"
  tools/gpu_lock.sh card "$card" -- ctest --test-dir build --output-on-failure -R "^$t\$" > "$out/$t.log" 2>&1
  echo "rc=$? ($t)"
done
echo DEEP_VERIFY_TESTS_DONE
