#!/bin/bash
# Phase 2: the Ornith prefill gates on the branch, under a card lock (ctest output to stdout).
#   tools/gpu_lock.sh card 1 -- tools/probe/pf_phase2_gates.sh [ctest regex]
RE="${1:-^(pf_norm_fused_test|gdn_front_test|prefill_gate_ornith_.*|prefill_split_ornith_.*|ornith_prefill_.*|golden_gate_ornith_test|ornith_mtp_names_test)$}"
ctest --test-dir "${BUILD:-build}" --output-on-failure -R "$RE"
