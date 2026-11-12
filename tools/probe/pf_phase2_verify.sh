#!/bin/bash
# Phase 2: a task's landing check, ON THE BOX under a card lock (BUILD = the candidate build dir):
#   BUILD=<build dir> tools/gpu_lock.sh card <N> -- tools/probe/pf_phase2_verify.sh
# The Ornith set (pf_phase2_gates.sh), then the host / kernel / Flash-Next tests the phase 2 files touch.
D="$(dirname "$0")"
"$D/pf_phase2_gates.sh"
"$D/pf_phase2_gates.sh" "^(context_test|pf_prep_test|gdn_conv_test|gdn_wy_test|gdn_chunk_test|variant_names_test|memory_plan_test|prefill_smoke_test|ornith_decode_test|pf_moe_test|qwen4exp_plan_test|qwen4exp_pf_variant_names_test|qwen4exp_pf_kernels_test|prefill_split_qwen4exp_test|qwen4exp_prefill_gdn_test|qwen4exp_prefill_synth_ours_test|qwen4exp_golden_prefill_synth_ours_test)$"
