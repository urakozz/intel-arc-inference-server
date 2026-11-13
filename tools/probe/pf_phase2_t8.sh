#!/bin/bash
# Phase 2 Task 8's gates in one card job, one Intel-checkpoint run at a time (each pins ~52 GB of host RAM):
#   CTRL=<main's build> CAND=<branch build> tools/gpu_lock.sh card <N> -- tools/probe/pf_phase2_t8.sh
D="$(dirname "$0")"
BUILD="$CAND" "$D/pf_phase2_gates.sh" "^(qwen4exp_pf_variant_names_test|qwen4exp_pf_kernels_test|qwen4exp_plan_test|qwen4exp_prefill_gdn_test|qwen4exp_prefill_synth_(ours|intel|eager)_test|prefill_split_qwen4exp_test|qwen4exp_golden_prefill_synth_(ours|intel)(_c16|_i8head|_inject|_eager)?_test|qwen4exp_prefill_intel_test)$"
"$D/pf_phase2_bitwise.sh" "$HOME/b70/logs/moe-prefill-phase2/t8-bitwise" qwen4exp_prefill_synth_ours_test \
  qwen4exp_prefill_synth_intel_test qwen4exp_golden_prefill_synth_ours_test
"$D/pf_phase2_ab_q4exp.sh" 4096 2
