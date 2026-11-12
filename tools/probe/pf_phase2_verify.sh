#!/bin/bash
# Phase 2: the landing check of the bitwise tasks, ON THE BOX under card 1's lock (BUILD = the build dir):
#   tools/gpu_lock.sh card 1 -- tools/probe/pf_phase2_verify.sh <outdir>
# 1. the Ornith set (pf_phase2_gates.sh), 2. host / kernel / Flash-Next tests, 3. end-to-end bitwise against
# B70_PF_PHASE2=old (pf_phase2_bitwise.sh) for Ornith and Flash-Next.
OUT="$1"
D="$(dirname "$0")"
"$D/pf_phase2_gates.sh"
"$D/pf_phase2_gates.sh" "^(context_test|pf_prep_test|gdn_conv_test|gdn_wy_test|gdn_chunk_test|variant_names_test|memory_plan_test|prefill_smoke_test|ornith_decode_test|pf_moe_test|qwen4exp_plan_test|qwen4exp_pf_variant_names_test|qwen4exp_pf_kernels_test|prefill_split_qwen4exp_test|qwen4exp_prefill_gdn_test)$"
"$D/pf_phase2_bitwise.sh" "$OUT" prefill_gate_ornith_l0_test prefill_gate_ornith_l0-int8_test prefill_gate_ornith_l0_c16_test \
  ornith_prefill_l0_test qwen4exp_prefill_synth_ours_test qwen4exp_golden_prefill_synth_ours_test
