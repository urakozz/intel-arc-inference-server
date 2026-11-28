# Each test is add_executable + b70_target_kernel_dir + add_test. No framework:
# tests/check.h exits non-zero on the first failed CHECK. The exception is the
# `b70_cli_reject` block at the bottom, which has no source file of its own: it
# drives the shipped `b70-decode` binary and asserts its exit code.

add_executable(copy_test l0/copy_test.cc)
target_include_directories(copy_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(copy_test PRIVATE b70_l0)
add_test(NAME copy_test COMMAND copy_test)

add_executable(launch_test l0/launch_test.cc)
target_include_directories(launch_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(launch_test PRIVATE b70_l0)
b70_target_kernel_dir(launch_test)
add_dependencies(launch_test kernel_noop)
add_test(NAME launch_test COMMAND launch_test)

add_executable(arg_capture_test l0/arg_capture_test.cc)
target_include_directories(arg_capture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(arg_capture_test PRIVATE b70_l0)
b70_target_kernel_dir(arg_capture_test)
add_dependencies(arg_capture_test kernel_noop)
add_test(NAME arg_capture_test COMMAND arg_capture_test)

# Kernel-timestamp events (plan 4's profiler foundation): the device timer
# calibration and the signal parameter on launch, on a two-launch regular list.
# Same shape as launch_test - a GPU and the noop binary, no checkpoint - so it
# carries no label, like every other test in this l0 group.
add_executable(event_test l0/event_test.cc)
target_include_directories(event_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(event_test PRIVATE b70_l0)
b70_target_kernel_dir(event_test)
add_dependencies(event_test kernel_noop)
add_test(NAME event_test COMMAND event_test)

# l0::MemKind::HostLocked (host memory locked against swap): mlock on a driver allocation locks nothing, a HostLocked
# allocation under a covering limit is locked and read by a copy and an indirect kernel read on every card, and under a
# limit that does not cover it is plain Host memory. Two cards when visible.
add_executable(host_lock_test l0/host_lock_test.cc)
target_include_directories(host_lock_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(host_lock_test PRIVATE b70_l0)
b70_target_kernel_dir(host_lock_test)
add_dependencies(host_lock_test kernel_host_read)
add_test(NAME host_lock_test COMMAND host_lock_test)

# The prefill execution context (spec 2.1 §3.5): the Level Zero list, through
# b70_prefill_host.
add_executable(context_test prefill/context_test.cc)
target_include_directories(context_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(context_test PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(context_test)
add_dependencies(context_test kernel_noop kernel_pf_probe_chain)
add_test(NAME context_test COMMAND context_test)

# Spec 2.1 S0: pf_gemm at every prefill shape against a host fp64 reference (bar 1, re-stated
# 2026-10-27 as a cos / max-ulp bound: tests/prefill/pf_gemm_test.cc).
add_executable(pf_gemm_test prefill/pf_gemm_test.cc)
target_include_directories(pf_gemm_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pf_gemm_test PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(pf_gemm_test)
add_dependencies(pf_gemm_test kernel_pf_gemm_T0 kernel_pf_gemm_T1 kernel_pf_gemm_T0_SILU
                 kernel_pf_silu_mul)
foreach(_t A2_B2_S4 A4_B2_S4)   # open-work 3a: gemm_l0's short-chunk tiles
  add_dependencies(pf_gemm_test kernel_pf_gemm_small_${_t} kernel_pf_gemm_small_${_t}_SILU)
endforeach()
foreach(_t A4_B1_S1_L0 A4_B1_S1_L1 A8_B1_S2_L0 A8_B1_S2_L1 A1_B2_S4_L0_SILU A4_B2_S1_L0_SILU)   # open-work 3b
  add_dependencies(pf_gemm_test kernel_pf_gemm_q4_${_t})
endforeach()
add_dependencies(pf_gemm_test kernel_pf_dequant_slab_K5120_N34816_L0 kernel_pf_dequant_slab_K5120_N16384_L1
                 kernel_pf_dequant_slab_K17408_N5120_L0)
add_dependencies(pf_gemm_test kernel_pf_gemm_q4_A8_B1_S4_L0_SLM_SILU)
add_test(NAME pf_gemm_test COMMAND pf_gemm_test)
set_tests_properties(pf_gemm_test PROPERTIES TIMEOUT 1800)

# 3g: unused setup, bitwise output and later slab fallback, without a checkpoint.
add_executable(linear_l0_lazy_test prefill/linear_l0_lazy_test.cc)
target_include_directories(linear_l0_lazy_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(linear_l0_lazy_test PRIVATE b70_prefill_host b70_runtime b70_l0)
b70_target_kernel_dir(linear_l0_lazy_test)
foreach(_layout 0 1)
  add_ocloc_kernel(pf_dequant_slab_K256_N2048_L${_layout}
    SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/prefill/pf_dequant_slab.cl
    DEFINES K=256 N=2048 LAYOUT=${_layout})
  add_dependencies(linear_l0_lazy_test kernel_pf_dequant_slab_K256_N2048_L${_layout})
endforeach()
foreach(_t A4_B1_S1_L0 A4_B1_S1_L1 A8_B1_S2_L0 A8_B1_S2_L1
           A1_B2_S4_L0_SILU A4_B2_S1_L0_SILU A8_B1_S4_L0_SLM_SILU)
  add_dependencies(linear_l0_lazy_test kernel_pf_gemm_q4_${_t})
endforeach()
add_dependencies(linear_l0_lazy_test kernel_pf_gemm_small_A2_B2_S4
                 kernel_pf_gemm_small_A2_B2_S4_SILU)
add_test(NAME linear_l0_lazy_test COMMAND linear_l0_lazy_test)
set_tests_properties(linear_l0_lazy_test PROPERTIES TIMEOUT 120)

add_executable(pf_int8_test prefill/pf_int8_test.cc)
target_include_directories(pf_int8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pf_int8_test PRIVATE b70_l0)
b70_target_kernel_dir(pf_int8_test)
add_dependencies(pf_int8_test kernel_pf_quant_had_K3072 kernel_pf_quant_had_K8704
                 kernel_pf_requant_rot_L0_PAD kernel_pf_requant_rot_L1_PAD
                 kernel_pf_quant_had_K5120 kernel_pf_quant_had_K6144
                 kernel_pf_quant_had_K17408 kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1
                 kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU)
add_test(NAME pf_int8_test COMMAND pf_int8_test)

# Spec 6 K1 (plan 6b Task 1): pf_flash_attn against an fp64 reference.
add_executable(pf_flash_attn_test prefill/pf_flash_attn_test.cc)
target_include_directories(pf_flash_attn_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pf_flash_attn_test PRIVATE b70_l0)
b70_target_kernel_dir(pf_flash_attn_test)
add_dependencies(pf_flash_attn_test kernel_pf_flash_attn)
add_test(NAME pf_flash_attn_test COMMAND pf_flash_attn_test)
set_tests_properties(pf_flash_attn_test PROPERTIES TIMEOUT 1800)

add_executable(linear_i8_test prefill/linear_i8_test.cc)
target_include_directories(linear_i8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(linear_i8_test PRIVATE b70_prefill_host b70_runtime b70_l0)
b70_target_kernel_dir(linear_i8_test)
add_dependencies(linear_i8_test kernel_pf_quant_had_K5120 kernel_pf_quant_had_K6144
                 kernel_pf_quant_had_K17408 kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1
                 kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU)
add_test(NAME linear_i8_test COMMAND linear_i8_test)
add_test(NAME linear_i8_cap4096_test COMMAND linear_i8_test --chunk-cap 4096)   # plan 5d: the 4096-row scratch

# --- spec 3 T3: HTTP server (host-only mock tests) ---
add_executable(openai_test server/openai_test.cc)
target_include_directories(openai_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(openai_test PRIVATE b70_server)
add_test(NAME openai_test COMMAND openai_test)

# Spec 7 C6: the Qwen XML tool-call parser over the golden tool-call outputs.
add_executable(toolcall_test server/toolcall_test.cc)
target_include_directories(toolcall_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(toolcall_test PRIVATE b70_server)
add_test(NAME toolcall_test COMMAND toolcall_test ${CMAKE_SOURCE_DIR}/tests/server/toolcall_expected.json
  ${CMAKE_SOURCE_DIR}/tests/golden/toolcall)

# Spec 7 C4 (plan 7c): the prefix cache store and plan, and the request path over a mock.
add_executable(prefix_cache_test server/prefix_cache_test.cc)
target_include_directories(prefix_cache_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(prefix_cache_test PRIVATE b70_server)
add_test(NAME prefix_cache_test COMMAND prefix_cache_test)
add_executable(prefix_server_test server/prefix_server_test.cc)
target_include_directories(prefix_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(prefix_server_test PRIVATE b70_server)
add_test(NAME prefix_server_test COMMAND prefix_server_test)

# Spec 8 M4 (plan 8c): the speculative acceptance rule, host only.
add_executable(spec_accept_test server/spec_accept_test.cc)
target_include_directories(spec_accept_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(spec_accept_test PRIVATE b70_server)
add_test(NAME spec_accept_test COMMAND spec_accept_test)

# Spec 19e (plan 19e Task 2): the prompt-lookup matcher against a brute force, host only.
add_executable(prompt_lookup_test server/prompt_lookup_test.cc)
target_include_directories(prompt_lookup_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(prompt_lookup_test PRIVATE b70_server)
add_test(NAME prompt_lookup_test COMMAND prompt_lookup_test)

# Spec 28 N6 (plan 28b): the golden tie rule for the speculative greedy gates (tests/server/greedy_tie.h).
add_executable(greedy_tie_test server/greedy_tie_test.cc)
target_include_directories(greedy_tie_test PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/server)
add_test(NAME greedy_tie_test COMMAND greedy_tie_test)
# Spec 28 N4 / N6's grading (tests/server/spec_judge.h): awaiting vs judged, the exit code (77 = awaiting a reference).
add_executable(spec_judge_test server/spec_judge_test.cc)
target_include_directories(spec_judge_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/tests/server ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(spec_judge_test PRIVATE b70_loader b70_model)
add_test(NAME spec_judge_test COMMAND spec_judge_test)

# Spec 8 §3.6 (plan 8c): the server over a scripted speculative engine, host only.
add_executable(mtp_server_test server/mtp_server_test.cc)
target_include_directories(mtp_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(mtp_server_test PRIVATE b70_server)
add_test(NAME mtp_server_test COMMAND mtp_server_test)
# b70-serve's reconfigure in place (POST /v1/control/reconfigure, cli/serve_reconfigure.h), host only.
add_executable(reconfigure_server_test server/reconfigure_server_test.cc)
target_include_directories(reconfigure_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(reconfigure_server_test PRIVATE b70_server)
add_test(NAME reconfigure_server_test COMMAND reconfigure_server_test)

# Spec 19e (plan 19e Task 2): the server with external prompt lookup over a mock engine that verifies
# external drafts, host only.
add_executable(lookup_server_test server/lookup_server_test.cc)
target_include_directories(lookup_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(lookup_server_test PRIVATE b70_server)
add_test(NAME lookup_server_test COMMAND lookup_server_test)

# Spec 19f (plan 19f Task 3): prompt lookup inside DFlash - the policy (ENTER / STAY / EXIT, the window, K caps) and the
# server over a mock DFlash engine (responses byte-identical to plain, the proposals, the warm ring), host only.
add_executable(lookup_in_dflash_test server/lookup_in_dflash_test.cc)
target_include_directories(lookup_in_dflash_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(lookup_in_dflash_test PRIVATE b70_server)
add_test(NAME lookup_in_dflash_test COMMAND lookup_in_dflash_test)
add_executable(dflash_lookup_server_test server/dflash_lookup_server_test.cc)
target_include_directories(dflash_lookup_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_lookup_server_test PRIVATE b70_server)
add_test(NAME dflash_lookup_server_test COMMAND dflash_lookup_server_test)
# Open-work 27: prompt lookup inside MTP through the server - identical output, the policy's proposals, the per-step
# choice kept on the MTP iterations, the refusals.
add_executable(mtp_lookup_server_test server/mtp_lookup_server_test.cc)
target_include_directories(mtp_lookup_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(mtp_lookup_server_test PRIVATE b70_server)
add_test(NAME mtp_lookup_server_test COMMAND mtp_lookup_server_test)

# Spec 8 §10: the --spec mtp policy on synthetic acceptance sequences, host only.
add_executable(adaptive_k_test server/adaptive_k_test.cc)
target_include_directories(adaptive_k_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(adaptive_k_test PRIVATE b70_server)
add_test(NAME adaptive_k_test COMMAND adaptive_k_test)

# Open-work 13: the per-step draft choice (server::StepK) from given probabilities, costs and calibration, host only.
add_executable(step_k_test server/step_k_test.cc)
target_include_directories(step_k_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(step_k_test PRIVATE b70_server)
add_test(NAME step_k_test COMMAND step_k_test)

# Spec 13 B4 (plan 13b Task 1): the batching scheduler over a mock batched engine, host only.
add_executable(scheduler_test server/scheduler_test.cc)
target_include_directories(scheduler_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(scheduler_test PRIVATE b70_server)
add_test(NAME scheduler_test COMMAND scheduler_test)

# Plan 13b Task 3: cli::ReplicaEngine (b70-serve --parallel replica: one engine per card behind the scheduler) over
# two host mock engines - threads, bitwise per card, a card's failure, a restore across cards, the prefill cut. Host only.
add_executable(replica_engine_test cli/replica_engine_test.cc)
target_include_directories(replica_engine_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(replica_engine_test PRIVATE b70_server)
add_test(NAME replica_engine_test COMMAND replica_engine_test)
# Plan 13b Task 4: the server's scheduler path over HTTP (cli::ReplicaEngine over two mock cards) against the serial
# server - replies equal alone and concurrent, streamed, split-last, a disconnect, the 503, the request log. Host only.
add_executable(replica_server_test server/replica_server_test.cc)
target_include_directories(replica_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(replica_server_test PRIVATE b70_server)
add_test(NAME replica_server_test COMMAND replica_server_test)

# Open-work 27 under --parallel replica: lookup inside MTP per request on each card, against the serial server.
add_executable(replica_lookup_server_test server/replica_lookup_server_test.cc)
target_include_directories(replica_lookup_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(replica_lookup_server_test PRIVATE b70_server)
add_test(NAME replica_lookup_server_test COMMAND replica_lookup_server_test)

add_executable(protocol_test server/protocol_test.cc)
target_include_directories(protocol_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(protocol_test PRIVATE b70_server)
add_test(NAME protocol_test COMMAND protocol_test)

# P3 is raw Level Zero/OpenCL C only: it requires a B70 and its kernels.
add_executable(dequant_test prefill/dequant_test.cc)
target_include_directories(dequant_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dequant_test PRIVATE b70_l0)
b70_target_kernel_dir(dequant_test)
add_dependencies(dequant_test
  kernel_pf_dequant_tile_K256_N64_L0_T0
  kernel_pf_dequant_tile_K256_N64_L0_T1
  kernel_pf_dequant_tile_K256_N64_L1_T0
  kernel_pf_dequant_tile_K256_N64_L1_T1)
add_test(NAME dequant_test COMMAND dequant_test)

# Spec 2.1 S0: the slab dequant against the full one, every slab, production layouts.
add_executable(pf_dequant_slab_test prefill/pf_dequant_slab_test.cc)
target_include_directories(pf_dequant_slab_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pf_dequant_slab_test PRIVATE b70_l0)
b70_target_kernel_dir(pf_dequant_slab_test)
add_dependencies(pf_dequant_slab_test
  kernel_pf_dequant_slab_K5120_N16384_L1 kernel_pf_dequant_slab_K6144_N5120_L0
  kernel_pf_dequant_slab_K5120_N34816_L0 kernel_pf_dequant_slab_K17408_N5120_L0
  kernel_pf_dequant_slab_K5120_N14336_L0
  kernel_pf_dequant_tile_K5120_N16384_L1_T0 kernel_pf_dequant_tile_K6144_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N34816_L0_T0 kernel_pf_dequant_tile_K17408_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N14336_L0_T0)
add_test(NAME pf_dequant_slab_test COMMAND pf_dequant_slab_test)
set_tests_properties(pf_dequant_slab_test PROPERTIES TIMEOUT 900)

# The runtime-`M`, S = 1 small kernels (spec 2, plan 6b Tasks 3-5). Like
# dequant_test these are raw Level Zero and OpenCL C: they need a B70 and their
# own binaries, and carry no label. Each also loads the DECODE binary it must be bit-identical to at M = 1,
# which is why the dependency lists below name both sides.
#
# `tests/kernels` is on the include path because the prefill references are
# mostly a redirection to the decode ones (tests/prefill/pf_ref.h says why).
add_executable(pf_prep_test prefill/pf_prep_test.cc)
target_include_directories(pf_prep_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(pf_prep_test PRIVATE b70_l0)
b70_target_kernel_dir(pf_prep_test)
add_dependencies(pf_prep_test
  kernel_pf_embed_gather kernel_pf_res_fold_K5120_SP0_G20 kernel_pf_res_fold_K5120_SP1_G20
  kernel_pf_norm_finish_K5120_G20_W20 kernel_pf_silu_mul
  kernel_embed_gather_M1 kernel_prep_res_fold_M1_K5120_SP0_G20
  kernel_prep_res_fold_M1_K5120_SP1_G20 kernel_prep_norm_finish_M1_K5120_G20_W20
  kernel_prep_silu_mul_M1)
add_test(NAME pf_prep_test COMMAND pf_prep_test)

# The residual fold + RMSNorm in one launch against the pair, every byte (Qwen3.8's 5120 here; Ornith's 2048 below).
add_executable(pf_norm_fused_test prefill/pf_norm_fused_test.cc)
target_include_directories(pf_norm_fused_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(pf_norm_fused_test PRIVATE b70_l0)
b70_target_kernel_dir(pf_norm_fused_test)
add_dependencies(pf_norm_fused_test kernel_pf_res_fold_K5120_SP0_G20 kernel_pf_res_fold_K5120_SP1_G20
                 kernel_pf_norm_finish_K5120_G20_W20 kernel_pf_res_norm_K5120_SP0_G20 kernel_pf_res_norm_K5120_SP1_G20)
add_test(NAME pf_norm_fused_qwen_test COMMAND pf_norm_fused_test 5120)
set_tests_properties(pf_norm_fused_qwen_test PROPERTIES LABELS "prefill")
if(B70_MTP)   # PERF-85: the MTP head's prompt-pass forms (_Z, pf_mtp_front) against the launches they replace
  add_dependencies(pf_norm_fused_test kernel_pf_res_fold_K5120_SP1_G20_Z kernel_pf_res_norm_K5120_SP1_G20_Z
                   kernel_pf_embed_gather kernel_pf_norm_finish_K5120_G20_W20_X10240 kernel_pf_mtp_front_K5120_G20)
  add_test(NAME pf_norm_fused_mtp_qwen_test COMMAND pf_norm_fused_test 5120 mtp)
  set_tests_properties(pf_norm_fused_mtp_qwen_test PROPERTIES LABELS "prefill;mtp")
endif()

add_executable(pf_gemv_test prefill/pf_gemv_test.cc)
target_include_directories(pf_gemv_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(pf_gemv_test PRIVATE b70_l0)
b70_target_kernel_dir(pf_gemv_test)
add_dependencies(pf_gemv_test kernel_pf_ab_proj kernel_gemv_bf16_M1_K5120_N128_C16_S16)
if(B70_ORNITH)   # phase 3 Task 2: the ab_gemm case (Ornith's a||b by tiled_gemm, pf_gdn_gate at pitch 256)
  target_compile_definitions(pf_gemv_test PRIVATE B70_PF_AB_GEMM=1)
  target_link_libraries(pf_gemv_test PRIVATE b70_prefill_host)
  add_dependencies(pf_gemv_test kernel_pf_route_slab_K2048_N128_P256 kernel_pf_gemm_T0
                   kernel_pf_gemm_small_A2_B2_S4 kernel_pf_gemm_small_A4_B2_S4
                   kernel_pf_gdn_conv_GK16V32 kernel_pf_gdn_conv_GK16V32_AB256)
endif()
add_test(NAME pf_gemv_test COMMAND pf_gemv_test)

add_executable(pf_attn_test prefill/pf_attn_test.cc)
target_include_directories(pf_attn_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(pf_attn_test PRIVATE b70_l0)
b70_target_kernel_dir(pf_attn_test)
add_dependencies(pf_attn_test kernel_pf_gated_head kernel_pf_attn_prep
  kernel_prep_gated_head_M1 kernel_attn_prep_M1)
add_test(NAME pf_attn_test COMMAND pf_attn_test)

# gdn_chunk's kernels (plan 6b Tasks 6-8). These drive the prefill execution
# context. No checkpoint - one synthetic GDN layer.
add_executable(gdn_conv_test prefill/gdn_conv_test.cc)
target_include_directories(gdn_conv_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gdn_conv_test PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(gdn_conv_test)
add_dependencies(gdn_conv_test kernel_pf_gdn_conv)
add_test(NAME gdn_conv_test COMMAND gdn_conv_test)
set_tests_properties(gdn_conv_test PROPERTIES TIMEOUT 1800)

add_executable(gdn_wy_test prefill/gdn_wy_test.cc)
target_include_directories(gdn_wy_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gdn_wy_test PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(gdn_wy_test)
add_dependencies(gdn_wy_test kernel_pf_gdn_wy)
add_test(NAME gdn_wy_test COMMAND gdn_wy_test)
set_tests_properties(gdn_wy_test PROPERTIES TIMEOUT 1800)

# gdn_chunk end to end. It links b70_runtime for PrefillScratch (2.264 GB of
# device scratch, the production allocation) and drives decode's own
# gdn_step_M1 / prep_gated_head_M1 as the self-consistency oracle.
add_executable(gdn_chunk_test prefill/gdn_chunk_test.cc)
target_include_directories(gdn_chunk_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gdn_chunk_test PRIVATE b70_prefill_host b70_runtime b70_l0)
b70_target_kernel_dir(gdn_chunk_test)
add_dependencies(gdn_chunk_test kernel_pf_gdn_conv kernel_pf_gdn_wy kernel_pf_gdn_scan kernel_pf_gdn_front_PL kernel_pf_gdn_scan_OB kernel_pf_gated_head_OB kernel_pf_gdn_wy_WB kernel_pf_gdn_intra
  kernel_pf_gated_head kernel_gdn_step_M1 kernel_prep_gated_head_M1)
add_test(NAME gdn_chunk_test COMMAND gdn_chunk_test)
set_tests_properties(gdn_chunk_test PROPERTIES TIMEOUT 1800)
# The walk binds the SPLIT scan and the vector solve (gdn.cc; their B70_PREFILL_GDN_SCAN /
# B70_PREFILL_GDN_SOLVE selectors went in the zero-config cleanup). These rows fail if either
# entry moves without the record and the gate evidence moving with it. The solve row observes
# `KernelCache::kernels()` rather than the output: the register entry (still compiled, graded
# by gdn_wy_test) is bit-identical by construction.
add_test(NAME gdn_scan_default_dispatch_test COMMAND gdn_chunk_test --dispatch-split)
add_test(NAME gdn_solve_default_dispatch_test COMMAND gdn_chunk_test --dispatch-solve-vector)
# Plan 5d: the 4096-position bars as ONE call at chunk cap 4096 (--chunk-cap, tests/chunk_cap_arg.h).
add_test(NAME gdn_chunk_cap4096_test COMMAND gdn_chunk_test --chunk-cap 4096)
set_tests_properties(gdn_chunk_cap4096_test PROPERTIES TIMEOUT 1800)

# The composed attention (ruling A14), end to end against decode's own
# reference. No checkpoint -- one synthetic FA layer over a synthetic cache. The host
# reference is the expensive half (attn_ref::decode walks every block of a
# 4352-position cache twice, once per q dtype), hence the timeout.
add_executable(attn_chunk_test prefill/attn_chunk_test.cc)
target_include_directories(attn_chunk_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(attn_chunk_test PRIVATE b70_prefill_host b70_runtime b70_l0)
b70_target_kernel_dir(attn_chunk_test)
add_dependencies(attn_chunk_test kernel_pf_attn kernel_pf_attn_prep_q16 kernel_pf_flash_attn kernel_pf_flash_attn_pair
                 kernel_pf_gemm_T0 kernel_pf_gemm_T1)
add_test(NAME attn_chunk_test COMMAND attn_chunk_test)
set_tests_properties(attn_chunk_test PROPERTIES TIMEOUT 3600)
if(B70_MTP)   # PERF-85: attn_prep over the kv heads only (the MTP head's prompt pass), bitwise the full grid's K / V
  add_executable(pf_attn_prep_kvo_test prefill/pf_attn_prep_kvo_test.cc)
  target_include_directories(pf_attn_prep_kvo_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                             ${CMAKE_SOURCE_DIR}/tests/kernels ${CMAKE_SOURCE_DIR}/tests/prefill)
  target_link_libraries(pf_attn_prep_kvo_test PRIVATE b70_l0)
  b70_target_kernel_dir(pf_attn_prep_kvo_test)
  add_dependencies(pf_attn_prep_kvo_test kernel_pf_attn_prep_q16 kernel_pf_attn_prep_q16_KVO)
  add_test(NAME pf_attn_prep_kvo_qwen_test COMMAND pf_attn_prep_kvo_test 24 4)
  set_tests_properties(pf_attn_prep_kvo_qwen_test PROPERTIES LABELS "prefill;mtp")
endif()

# D3's scan-only arithmetic discriminator is direct Level Zero/OpenCL C, so it
# must run in both configurations.  The fixtures make removal of either BF16
# low chain observable without re-implementing the scan.
add_executable(gdn_scan_split_test prefill/gdn_scan_split_test.cc)
target_include_directories(gdn_scan_split_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gdn_scan_split_test PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(gdn_scan_split_test)
add_dependencies(gdn_scan_split_test kernel_pf_gdn_scan)
add_test(NAME gdn_scan_split_test COMMAND gdn_scan_split_test)
set_tests_properties(gdn_scan_split_test PROPERTIES TIMEOUT 1800)


add_executable(int4_test kernels/int4_test.cc)
target_include_directories(int4_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME int4_test COMMAND int4_test)

# The kernel directory lookup (src/kernels/kernel_dir.cc): compiled-in path first, <executable>/../kernels in an
# extracted release tarball, a plain sentence naming both when neither exists. Host only.
add_executable(kernel_dir_test kernels/kernel_dir_test.cc)
target_include_directories(kernel_dir_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kernel_dir_test PRIVATE b70_kernel_dir)
add_test(NAME kernel_dir_test COMMAND kernel_dir_test)

# Spec 28 (plan 28a): the small-M GEMM's fp64 reference and gemv's fp32 host twin. Host only.
add_executable(gemm_sm_ref_test kernels/gemm_sm_ref_test.cc)
target_include_directories(gemm_sm_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
add_test(NAME gemm_sm_ref_test COMMAND gemm_sm_ref_test)

# Spec 28b: the production small-M GEMM. gemm_sm_names_test (host): kernels::kSmallMTuning against the CMake
# block's binaries. gemm_sm_test (device): N2 against fp64 and gemv, N3 property R, every row at every built M.
get_property(B70_GEMM_SM_KERNELS GLOBAL PROPERTY B70_GEMM_SM_KERNELS)
get_property(B70_GEMM_SM_CONSUMERS GLOBAL PROPERTY B70_GEMM_SM_CONSUMERS)
if(B70_MTP AND B70_DFLASH AND B70_VERIFY_M8 AND B70_ORNITH AND B70_Q4EXP)
  add_executable(gemm_sm_names_test kernels/gemm_sm_names_test.cc)
  target_include_directories(gemm_sm_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  b70_target_kernel_dir(gemm_sm_names_test)
  add_test(NAME gemm_sm_names_test COMMAND gemm_sm_names_test ${B70_GEMM_SM_KERNELS} -- ${B70_GEMM_SM_CONSUMERS})

  # The capture rule (runtime/gemv_kind.h): M = 1 on gemv for every list, M >= 2 on gemm_sm, the override.
  add_executable(gemv_kind_test runtime/gemv_kind_test.cc)
  target_include_directories(gemv_kind_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(gemv_kind_test PRIVATE b70_model)
  b70_target_kernel_dir(gemv_kind_test)
  add_test(NAME gemv_kind_test COMMAND gemv_kind_test)

  add_executable(gemm_sm_test kernels/gemm_sm_test.cc)
  target_include_directories(gemm_sm_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(gemm_sm_test PRIVATE b70_l0)
  b70_target_kernel_dir(gemm_sm_test)
  add_dependencies(gemm_sm_test ${B70_GEMM_SM_KERNELS})
  get_property(_pi8h GLOBAL PROPERTY B70_PI8H_KERNELS)   # PERF-111: the head rows' n-major twins
  add_dependencies(gemm_sm_test ${_pi8h})
  add_test(NAME gemm_sm_test COMMAND gemm_sm_test)
  set_tests_properties(gemm_sm_test PROPERTIES TIMEOUT 1800)
endif()

# Spec 28c (plan 28c Task 1): the verify attention's fp64 reference at M rows (bf16 and int8 KV) and
# v2's block view. Host only.
add_executable(attn_verify_ref_test kernels/attn_verify_ref_test.cc)
target_include_directories(attn_verify_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
add_test(NAME attn_verify_ref_test COMMAND attn_verify_ref_test)

# Spec 28c (plan 28c Task 5): the capture's verify-attention rule (runtime/verify_attn.h), host only; the built
# attn_verify_dpas names are its arguments (every (M, KV form) the rule can bind must be on a CMake line).
set(_vattn_built)
foreach(MM 2 3 4 5 6 7 8 16 32 64)   # 16: open-work 28's wide list; 32 / 64: spec 32's deep lists
  foreach(SFX "" _KV8 _Q12KV2 _Q12KV2_KV8 _Q16KV2)   # spec 17d: the per-card 12 / 2 builds; PERF-56: Ornith's 16 / 2
    if(TARGET kernel_attn_verify_dpas_M${MM}_T32${SFX})
      list(APPEND _vattn_built attn_verify_dpas_M${MM}_T32${SFX})
    endif()
  endforeach()
endforeach()
# PERF-120: the 8-row lists' 8 key blocks (runtime::verify_attn_blocks), unpaged and paged, with their reduce
foreach(SFX "" _KV8 _Q12KV2 _Q12KV2_KV8 _PG${KV_PAGE} _KV8_PG${KV_PAGE})
  foreach(K attn_verify_dpas attn_v2)
    if(TARGET kernel_${K}_M8_T8${SFX})
      list(APPEND _vattn_built ${K}_M8_T8${SFX})
    endif()
  endforeach()
endforeach()
add_executable(verify_attn_test runtime/verify_attn_test.cc)
target_include_directories(verify_attn_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(verify_attn_test PRIVATE b70_model)
b70_target_kernel_dir(verify_attn_test)
add_test(NAME verify_attn_test COMMAND verify_attn_test ${_vattn_built})

# Spec 28c (plan 28c Task 6): the production attn_verify_dpas binaries + v2's reduce on the card - N1, property R,
# against v2 at M = 1. Part S always; part D on plan 28c Task 4's capture when B70_VERIFY_ATTN_DUMP names it.
if(B70_VERIFY_DPAS AND B70_MTP AND B70_VERIFY_M8 AND B70_KV8)
  set(B70_VERIFY_ATTN_DUMP "" CACHE PATH "plan 28c's verify-attention capture")
  if(NOT B70_VERIFY_ATTN_DUMP AND DEFINED ENV{B70_DATA})
    set(B70_VERIFY_ATTN_DUMP "$ENV{B70_DATA}/logs/spec28c/vattn-dump" CACHE PATH "plan 28c's verify-attention capture" FORCE)
  endif()
  add_executable(attn_verify_dpas_test kernels/attn_verify_dpas_test.cc)
  target_include_directories(attn_verify_dpas_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                             ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(attn_verify_dpas_test PRIVATE b70_l0)
  b70_target_kernel_dir(attn_verify_dpas_test)
  set(_vdpas_deps)
  foreach(MM 2 3 4 5 6 7 8)
    list(APPEND _vdpas_deps kernel_attn_verify_dpas_M${MM}_T32 kernel_attn_v2_M${MM}_T32)
  endforeach()
  foreach(MM 2 3 4 5 6 7 8)
    list(APPEND _vdpas_deps kernel_attn_verify_dpas_M${MM}_T32_KV8 kernel_attn_v2_M${MM}_T32_KV8)
  endforeach()
  add_dependencies(attn_verify_dpas_test ${_vdpas_deps} kernel_attn_v2_M1_T32 kernel_attn_v2_M1_T32_KV8)
  add_test(NAME attn_verify_dpas_test COMMAND attn_verify_dpas_test ${B70_VERIFY_ATTN_DUMP})
  set_tests_properties(attn_verify_dpas_test PROPERTIES TIMEOUT 3600)
  add_test(NAME attn_verify_dpas_kv8_m8_test COMMAND attn_verify_dpas_test --kv8-m8)
  set_tests_properties(attn_verify_dpas_kv8_m8_test PROPERTIES LABELS "kv8;dflash" TIMEOUT 300)
  if(B70_ORNITH)   # PERF-56: Ornith's heads in its wide lookup verify lists (row chunks of 8), against v2 at M = 1
    add_dependencies(attn_verify_dpas_test kernel_attn_v2_M1_T32_Q16KV2)
    foreach(MM 16 32 64)
      add_dependencies(attn_verify_dpas_test kernel_attn_verify_dpas_M${MM}_T32_Q16KV2 kernel_attn_v2_M${MM}_T32_Q16KV2)
    endforeach()
    add_test(NAME attn_verify_dpas_ornith_wide_test COMMAND attn_verify_dpas_test --ornith-wide)
    set_tests_properties(attn_verify_dpas_ornith_wide_test PROPERTIES LABELS "ornith;mtp" TIMEOUT 600)
  endif()
endif()

# 3l: the production attn_verify_dpas[_kv8] against its frozen copy (kernels/attn_verify_control.cl, main 865da900's
# source with main's levers and options, hardcoded here so a lever change shows up as a mismatch), every partial and
# reduced word, bf16 and int8 at M = 2..8 (2..4 without B70_VERIFY_M8).
if(B70_VERIFY_DPAS AND B70_MTP AND B70_KV8)
  set(_vctl_m 2 3 4)
  if(B70_VERIFY_M8)
    list(APPEND _vctl_m 5 6 7 8)
  endif()
  set(_vctl_deps)
  foreach(MM ${_vctl_m})
    foreach(_kv 0 1)
      set(_sfx "")
      if(_kv)
        set(_sfx _KV8)
      endif()
      add_ocloc_kernel(attn_verify_control_M${MM}${_sfx} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/attn_verify_control.cl
        DEFINES CTRL_POS=0 CTRL_NACT=1 M=${MM} TGT=32 KT=32 PACK_H=0 KS=4 DSPLIT=256 QSPLIT=1 PSPLIT=1 KV8=${_kv}
        OPTIONS -cl-intel-256-GRF-per-thread)
      list(APPEND _vctl_deps kernel_attn_verify_control_M${MM}${_sfx} kernel_attn_verify_dpas_M${MM}_T32${_sfx}
                             kernel_attn_v2_M${MM}_T32${_sfx})
    endforeach()
  endforeach()
  if(B70_VERIFY_M8)   # spec 33: the 16-row line (open-work 28's levers, KS 1 DSPLIT 128) against the same frozen copy
    add_ocloc_kernel(attn_verify_control_M16 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/attn_verify_control.cl
      DEFINES CTRL_POS=0 CTRL_NACT=1 M=16 TGT=32 KT=32 PACK_H=0 KS=1 DSPLIT=128 QSPLIT=1 PSPLIT=1 KV8=0
      OPTIONS -cl-intel-256-GRF-per-thread)
    list(APPEND _vctl_deps kernel_attn_verify_control_M16 kernel_attn_verify_dpas_M16_T32 kernel_attn_v2_M16_T32)
  endif()
  add_executable(attn_verify_bitwise_test kernels/attn_verify_bitwise_test.cc)
  target_include_directories(attn_verify_bitwise_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(attn_verify_bitwise_test PRIVATE b70_l0)
  b70_target_kernel_dir(attn_verify_bitwise_test)
  add_dependencies(attn_verify_bitwise_test ${_vctl_deps})
  if(B70_VERIFY_M8)
    target_compile_definitions(attn_verify_bitwise_test PRIVATE B70_VERIFY_TEST_M8=1)
  endif()
  add_test(NAME attn_verify_bitwise_test COMMAND attn_verify_bitwise_test)
  set_tests_properties(attn_verify_bitwise_test PROPERTIES LABELS kv8 TIMEOUT 900)
endif()

add_executable(gemv_test kernels/gemv_test.cc)
target_include_directories(gemv_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_test)
add_dependencies(gemv_test
  kernel_gemv_control_M1_K6144_N5120_S4_L0
  kernel_gemv_M1_K6144_N5120_S4_L0 kernel_gemv_M1_K5120_N14336_S2_L0
  kernel_gemv_M1_K5120_N16384_S1_L1 kernel_gemv_M1_K5120_N34816_S8_L0
  kernel_gemv_M1_K17408_N5120_S4_L0 kernel_gemv_M2_K6144_N5120_S1_L0)
if(B70_TP AND B70_Q4EXP)   # PERF-108: the tp cards' qkv||z (_KS4)
  add_dependencies(gemv_test kernel_gemv_M1_K5120_N8192_S1_L1_KS4 kernel_gemv_M1_K2560_N8192_S1_L0_KS4)
endif()
add_test(NAME gemv_test COMMAND gemv_test)

add_executable(gemv_bf16_test kernels/gemv_bf16_test.cc)
target_include_directories(gemv_bf16_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_bf16_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_bf16_test)
# Both a||b tilings: the test runs the `{64, 1}` build and the retiled one the
# runtime binds over one set of inputs, and holds each - and their difference -
# to the reference tolerance (spec 1.5 lever L2 reorders the sum).
add_dependencies(gemv_bf16_test kernel_gemv_bf16_M1_K5120_N128
  kernel_gemv_bf16_M1_K5120_N128_C16_S16 kernel_gemv_bf16_M1_K5120_N248320)
# The verify rows on the DPAS: the 2..8-row and ROWG builds (item 40) against the one-row ones.
if(B70_MTP AND B70_VERIFY_M8)
  foreach(_kn K5120_N128_C16_S16 K10240_N5120 K5120_N14336)
    foreach(_m 1 2 3 4 5 6 7 8)
      add_dependencies(gemv_bf16_test kernel_gemv_bf16_M${_m}_${_kn})
    endforeach()
    add_dependencies(gemv_bf16_test kernel_gemv_bf16_M8_${_kn}_RG)
  endforeach()
endif()
add_test(NAME gemv_bf16_test COMMAND gemv_bf16_test)

# Spec 9 (plan 9b Task 2): the int8 lm_head GEMV at its real shape, M = 1 and 4, against
# an fp64 reference; prints the launch's GB/s.
add_executable(gemv_i8w_test kernels/gemv_i8w_test.cc)
target_include_directories(gemv_i8w_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_i8w_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_i8w_test)
add_dependencies(gemv_i8w_test kernel_gemv_i8w_M1_K5120_N248320 kernel_gemv_i8w_M4_K5120_N248320
                 kernel_pi8h_gemv_i8w_M1_K5120_N248320_NM kernel_pi8h_gemv_i8w_M4_K5120_N248320_NM)   # PERF-111
add_test(NAME gemv_i8w_test COMMAND gemv_i8w_test)
set_tests_properties(gemv_i8w_test PROPERTIES TIMEOUT 900)

add_executable(prep_test kernels/prep_test.cc)
target_include_directories(prep_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(prep_test PRIVATE b70_l0)
b70_target_kernel_dir(prep_test)
# Both shapes of the residual norm: the single-work-group `prep_res_norm` and
# the `prep_res_fold` + `prep_norm_finish` pair the runtime binds since spec
# 1.5's lever L1. The test runs the pair against the single kernel's own
# reference (`resid` bit-identical, `x` to the ruled ulp bar for the reordered
# Sigma), so both must exist - the same arrangement lever L2 left for
# gemv_bf16's two tilings. `_W1` is stage B's single-work-group finish, run to
# prove the grid changes no bits.
add_dependencies(prep_test kernel_prep_res_norm_M1_K5120_SP0 kernel_prep_res_norm_M1_K5120_SP4
  kernel_prep_res_fold_M1_K5120_SP0_G20 kernel_prep_res_fold_M1_K5120_SP4_G20
  kernel_prep_norm_finish_M1_K5120_G20_W20 kernel_prep_norm_finish_M1_K5120_G20_W1
  kernel_prep_res_fold_M1_K2048_SP0_G20 kernel_prep_res_fold_M1_K2048_SP4_G20 kernel_prep_norm_finish_M1_K2048_G20_W20
  kernel_prep_res_fold_M1_K2560_SP0_G20 kernel_prep_norm_finish_M1_K2560_G20_W20
  kernel_prep_silu_mul_M1 kernel_prep_gated_head_M1)
add_test(NAME prep_test COMMAND prep_test)

# The GDN decode step. Its reference (gdn_ref.h) reproduces the kernel's tree
# orders exactly, so the bars are ulp-tight and the conv ring is bit-exact.
add_executable(gdn_step_test kernels/gdn_step_test.cc)
target_include_directories(gdn_step_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gdn_step_test PRIVATE b70_l0)
b70_target_kernel_dir(gdn_step_test)
add_dependencies(gdn_step_test kernel_gdn_step_M1 kernel_gdn_step_M2)
add_test(NAME gdn_step_test COMMAND gdn_step_test)
# Spec 8 (plan 8b Task 4): the SPEC_SLOTS build's per-row state against M = 1 steps.
if(B70_MTP)
  add_executable(gdn_step_slots_test kernels/gdn_step_slots_test.cc)
  target_include_directories(gdn_step_slots_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(gdn_step_slots_test PRIVATE b70_l0)
  b70_target_kernel_dir(gdn_step_slots_test)
  add_dependencies(gdn_step_slots_test kernel_gdn_step_M1 kernel_gdn_step_slots_M1
    kernel_gdn_step_slots_M2 kernel_gdn_step_slots_M3 kernel_gdn_step_slots_M4)
  foreach(MM RANGE 1 8)   # spec 19 v3d: DFlash's 8-slot builds too (B70_VERIFY_M8, on with MTP by default)
    add_dependencies(gdn_step_slots_test kernel_gdn_step_slots_M${MM}_N8)
  endforeach()
  add_test(NAME gdn_step_slots_test COMMAND gdn_step_slots_test)
endif()
# Spec 19a (plan 19a Task 4): every name probe_mtp_steps' verify lists at M = 5..8 and
# probe_draft_cost bind is built by src/kernels/CMakeLists.txt's B70_VERIFY_M8 block (or
# reused), and the block builds nothing they do not bind. Host only (names, no device).
if(B70_VERIFY_M8 AND B70_MTP)
  set(B70_VERIFY_M8_KERNELS)
  foreach(MM 5 6 7 8)
    list(APPEND B70_VERIFY_M8_KERNELS
      kernel_gemv_M${MM}_K5120_N16384_S1_L1 kernel_gemv_M${MM}_K6144_N5120_S4_L0
      kernel_gemv_M${MM}_K5120_N34816_S8_L0 kernel_gemv_M${MM}_K17408_N5120_S4_L0
      kernel_gemv_M${MM}_K5120_N14336_S2_L0 kernel_gemv_i8w_M${MM}_K5120_N248320
      kernel_gemv_bf16_M${MM}_K5120_N128_C16_S16 kernel_prep_res_fold_M${MM}_K5120_SP0_G20
      kernel_prep_res_fold_M${MM}_K5120_SP4_G20 kernel_prep_norm_finish_M${MM}_K5120_G20_W20
      kernel_prep_silu_mul_M${MM} kernel_prep_gated_head_M${MM} kernel_embed_gather_M${MM}
      kernel_argmax_stage1_M${MM} kernel_attn_prep_M${MM} kernel_attn_v2_M${MM}_T${ATTN_V2_TGT}
      kernel_gdn_step_slots_M${MM}_N8 kernel_gemv_bf16_M${MM}_K10240_N5120
      kernel_gemv_bf16_M${MM}_K5120_N14336 kernel_prep_norm_finish_M${MM}_K5120_G20_W20_X10240_P2
      kernel_prep_res_fold_M${MM}_K5120_SP0_G20_P2 kernel_prep_res_fold_M${MM}_K5120_SP1_G20_Z kernel_attn_prep_M${MM}_S1)
  endforeach()
  foreach(KN K5120_N6144 K4096_N5120 K5120_N34816 K17408_N5120 K25600_N5120 K5120_N2048)
    list(APPEND B70_VERIFY_M8_KERNELS kernel_gemv_i8w_M8_${KN} kernel_gemv_bf16_M8_${KN})
  endforeach()
  foreach(KN K5120_N1280 K5120_N256)
    list(APPEND B70_VERIFY_M8_KERNELS kernel_gemv_i8w_M8_${KN} kernel_gemv_bf16_M8_${KN}_C16_S16)
  endforeach()
  list(APPEND B70_VERIFY_M8_KERNELS kernel_gemv_M8_K5120_N6144_S4_L0 kernel_gemv_M8_K4096_N5120_S4_L0
    kernel_gemv_M8_K5120_N1280_S16_L0 kernel_gemv_M8_K5120_N2048_S8_L0
    kernel_gemv_M8_K25600_N5120_S8_L0 kernel_gemv_M8_K5120_N256_S16_L0)
  set(_v19a_reused kernel_argmax_stage2)
  get_property(_v28b_gemm_sm GLOBAL PROPERTY B70_GEMM_SM_KERNELS)   # spec 28b: the verify lists' gemm_sm
  get_property(_v28b_cons GLOBAL PROPERTY B70_GEMM_SM_CONSUMERS)    # and gate||up's SiLU at S 1
  list(APPEND _v19a_reused ${_v28b_gemm_sm} ${_v28b_cons})
  foreach(t ${B70_VERIFY_M8_KERNELS} ${_v19a_reused})
    if(NOT TARGET ${t})
      message(FATAL_ERROR "spec 19a: ${t} is not built (src/kernels/CMakeLists.txt)")
    endif()
  endforeach()
  add_executable(verify_m8_names_test kernels/verify_m8_names_test.cc)
  target_include_directories(verify_m8_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tools/probe)
  target_link_libraries(verify_m8_names_test PRIVATE b70_model)
  b70_target_kernel_dir(verify_m8_names_test)
  add_test(NAME verify_m8_names_test COMMAND verify_m8_names_test ${B70_VERIFY_M8_KERNELS} --
    ${_v19a_reused})
endif()

# The decode-attention trio (attn_prep, shared with v2; attn_decode / attn_reduce, v1 - the
# test-only reference attn_v2_test grades v2 against since the zero-config cleanup). Its
# reference (attn_ref.h) reproduces the kernels'
# norm tree, score tree, wave order and block-merge order exactly, so the KV
# cache and the norm/RoPE outputs are held bit-exact and only the `exp`-carrying
# tensors have a tolerance. M = 2 is run, not just compiled: it is the only
# cover for per-m causal masking.
add_executable(attn_test kernels/attn_test.cc)
target_include_directories(attn_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(attn_test PRIVATE b70_l0)
b70_target_kernel_dir(attn_test)
add_dependencies(attn_test kernel_attn_prep_M1
  kernel_attn_decode_M1_L4096_B${ATTN_BLOCK} kernel_attn_reduce_M1_L4096_B${ATTN_BLOCK}
  kernel_attn_prep_M2
  kernel_attn_decode_M2_L4096_B${ATTN_BLOCK} kernel_attn_reduce_M2_L4096_B${ATTN_BLOCK})
add_test(NAME attn_test COMMAND attn_test)

# Spec 10 A1 (plan 10b Task 1): decode attention v2 against v1. Part S (synthetic)
# always; part D on plan 10a's real capture when B70_ATTN_V2_DUMP names it (default
# $B70_DATA/scratch/spec10a-dump; without B70_DATA part D skips).
set(B70_ATTN_V2_DUMP "" CACHE PATH "plan 10a's decode-attention capture")
if(NOT B70_ATTN_V2_DUMP AND DEFINED ENV{B70_DATA})
  set(B70_ATTN_V2_DUMP "$ENV{B70_DATA}/scratch/spec10a-dump" CACHE PATH "plan 10a's decode-attention capture" FORCE)
endif()
add_executable(attn_v2_test kernels/attn_v2_test.cc)
target_include_directories(attn_v2_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(attn_v2_test PRIVATE b70_l0)
b70_target_kernel_dir(attn_v2_test)
add_dependencies(attn_v2_test
  kernel_attn_decode_M1_L4096_B${ATTN_BLOCK} kernel_attn_reduce_M1_L4096_B${ATTN_BLOCK}
  kernel_attn_decode_M1_L131072_B${ATTN_BLOCK} kernel_attn_reduce_M1_L131072_B${ATTN_BLOCK}
  kernel_attn_v2_M1_T${ATTN_V2_TGT} kernel_attn_v2_M2_T${ATTN_V2_TGT}
  kernel_attn_v2_M3_T${ATTN_V2_TGT} kernel_attn_v2_M4_T${ATTN_V2_TGT})
add_test(NAME attn_v2_test COMMAND attn_v2_test ${B70_ATTN_V2_DUMP})
set_tests_properties(attn_v2_test PROPERTIES TIMEOUT 900)

# Spec 15c: the MoE block's host reference (tests/kernels/moe_ref.h) against the
# spec's formula and an independent double-precision MoE - top-k ties to the lower id,
# the fixed slot order, SiLU x up, the layout-1 GEMV. Host only.
add_executable(moe_ref_test kernels/moe_ref_test.cc)
target_include_directories(moe_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME moe_ref_test COMMAND moe_ref_test)
# ... and moe.cl on the card against it at Ornith's shape (needs a B70, no checkpoint):
# the router GEMV + route, ties, every slot of gate||up against its own expert, down's
# fixed-order sum + gate + residual fold, bitwise replay.
if(B70_ORNITH)
  add_executable(moe_test kernels/moe_test.cc)
  target_include_directories(moe_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(moe_test PRIVATE b70_l0)
  b70_target_kernel_dir(moe_test)
  add_dependencies(moe_test kernel_moe_M1_E256_T8_D2048_I512 kernel_gemv_bf16_M1_K2048_N272_C16_S16)
  add_test(NAME moe_test COMMAND moe_test)
  set_tests_properties(moe_test PROPERTIES LABELS ornith TIMEOUT 900)
endif()

# The two kernels at a token's boundary. Both read the shared control block at
# *execution* time, which is what the replayed list needs and what a kernel
# argument (captured at append - tests/l0/arg_capture_test.cc) could not give.
add_executable(embed_gather_test kernels/embed_gather_test.cc)
target_include_directories(embed_gather_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(embed_gather_test PRIVATE b70_l0 b70_model)
b70_target_kernel_dir(embed_gather_test)
add_dependencies(embed_gather_test kernel_embed_gather_M1)
add_test(NAME embed_gather_test COMMAND embed_gather_test)

add_executable(argmax_test kernels/argmax_test.cc)
target_include_directories(argmax_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(argmax_test PRIVATE b70_l0 b70_model)
b70_target_kernel_dir(argmax_test)
add_dependencies(argmax_test kernel_argmax_stage1_M1 kernel_argmax_stage2)
add_test(NAME argmax_test COMMAND argmax_test)

# Open-work 13: draft_prob.cl against a host reference (the whole row, a TP card's half, the halves merged).
if(B70_TP AND B70_MTP)
  add_executable(draft_prob_test kernels/draft_prob_test.cc)
  target_include_directories(draft_prob_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(draft_prob_test PRIVATE b70_l0 b70_model)
  b70_target_kernel_dir(draft_prob_test)
  add_dependencies(draft_prob_test kernel_draft_prob_N248320_U248077 kernel_draft_prob_N124160_U124160
                   kernel_draft_prob_N124160_U123917)
  add_test(NAME draft_prob_test COMMAND draft_prob_test)
endif()

add_executable(json_test common/json_test.cc)
target_include_directories(json_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME json_test COMMAND json_test)

# Spec 12b: the int8 KV cache's rotkv arithmetic on the host (src/common/kv8.h) - the
# probe's signs, binary16 RNE, the rotation and the quantiser. No device.
add_executable(kv8_test common/kv8_test.cc)
target_include_directories(kv8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME kv8_test COMMAND kv8_test)
# ... and the f32 judges' simulation of it (tools/oracle/int8_sim.py kv8_*, qwen_forced.py --kv8) against kv8.h bit
# for bit on tools/oracle/kv8_sim_fixture.py's rows. No device.
add_executable(kv8_sim_test common/kv8_sim_test.cc)
target_include_directories(kv8_sim_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kv8_sim_test PRIVATE b70_loader)
add_test(NAME kv8_sim_test COMMAND kv8_sim_test ${CMAKE_CURRENT_SOURCE_DIR}/loader/kv8_sim_fixture.safetensors)

# Fix round A: the pieces a large host <-> device copy is cut into (src/common/copy_chunks.h). No device.
add_executable(copy_chunks_test common/copy_chunks_test.cc)
target_include_directories(copy_chunks_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME copy_chunks_test COMMAND copy_chunks_test)

add_executable(safetensors_test loader/safetensors_test.cc)
target_include_directories(safetensors_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(safetensors_test PRIVATE b70_loader)
add_test(NAME safetensors_test COMMAND safetensors_test)

# Spec 9 (plan 9b Task 1): the int8 lm_head's host quantisation, on synthetic rows.
add_executable(lm_head_int8_test loader/lm_head_int8_test.cc)
target_include_directories(lm_head_int8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(lm_head_int8_test PRIVATE b70_loader)
add_test(NAME lm_head_int8_test COMMAND lm_head_int8_test)

add_executable(quant_test loader/quant_test.cc)
target_include_directories(quant_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(quant_test PRIVATE b70_loader)
add_test(NAME quant_test COMMAND quant_test)

# Spec 20 §9: the compressed-tensors symmetric pack-quantized loader - exact repack onto
# the GPTQ layout (g64 / g128, f16 / bf16 scales), the startup note, the refusals, and
# real repos' quantization_config (tests/loader/ct/). Host only.
add_executable(ct_loader_test loader/ct_loader_test.cc)
target_include_directories(ct_loader_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(ct_loader_test PRIVATE b70_loader)
add_test(NAME ct_loader_test COMMAND ct_loader_test ${CMAKE_SOURCE_DIR}/tests/loader/ct)

add_executable(dequant_fixture_test loader/dequant_fixture_test.cc)
target_include_directories(dequant_fixture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dequant_fixture_test PRIVATE b70_loader)
add_test(NAME dequant_fixture_test COMMAND dequant_fixture_test ${CMAKE_SOURCE_DIR}/tests/golden/dequant_fixture.safetensors)

# Spec 15c: the MoE layer repack (loader/moe.h) on a synthetic per-expert GPTQ
# checkpoint the test writes itself (both per-expert forms, g128, the refusals) and
# Ornith's MoE byte figures. Host only.
add_executable(ornith_repack_test loader/ornith_repack_test.cc)
target_include_directories(ornith_repack_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(ornith_repack_test PRIVATE b70_loader)
add_test(NAME ornith_repack_test COMMAND ornith_repack_test)

# Spec 15 §13: the GDN a||b in the checkpoint's form (loader/ab.h) - the int4 checkpoint's
# a||b repacked exactly with its zero padding, its bf16 prefill copy, the bf16 row
# untouched - on a fixture the test writes at Ornith's real a/b shape. Host only.
add_executable(ab_int4_test loader/ab_int4_test.cc)
target_include_directories(ab_int4_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(ab_int4_test PRIVATE b70_loader)
add_test(NAME ab_int4_test COMMAND ab_int4_test)

# Spec 6 §10: config.json's max_position_embeddings (text_config or top level), the
# bound of --max-len. Host only; inline JSON.
# b70-serve --prefix-cache-gb auto: host-only, the size rule over /proc/meminfo text.
add_executable(prefix_cache_size_test cli/prefix_cache_size_test.cc)
target_include_directories(prefix_cache_size_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME prefix_cache_size_test COMMAND prefix_cache_size_test)

add_executable(trained_context_test loader/trained_context_test.cc)
target_include_directories(trained_context_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(trained_context_test PRIVATE b70_loader)
add_test(NAME trained_context_test COMMAND trained_context_test)

add_executable(qwen35_test model/qwen35_test.cc)
target_include_directories(qwen35_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen35_test PRIVATE b70_model)
add_test(NAME qwen35_test COMMAND qwen35_test)

# Spec 14 §3.1: the model descriptor (host-only).
add_executable(model_desc_test model/model_desc_test.cc)
target_include_directories(model_desc_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(model_desc_test PRIVATE b70_model)
add_test(NAME model_desc_test COMMAND model_desc_test)

# Spec 15b: the kernel variant names built from a descriptor - main's names at
# Qwen3.8's shapes, suffixed ones at Ornith's. Host only; names only.
add_executable(variant_names_test kernels/variant_names_test.cc)
target_include_directories(variant_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(variant_names_test PRIVATE b70_model)
b70_target_kernel_dir(variant_names_test)
add_test(NAME variant_names_test COMMAND variant_names_test)

# The model table and src/kernels/CMakeLists.txt are two lists that have to
# agree: every row's compiled variant must exist before plan 3 binds it. No
# device, no checkpoint - just the filenames.
add_executable(kernel_table_test kernels/kernel_table_test.cc)
target_include_directories(kernel_table_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kernel_table_test PRIVATE b70_model)
b70_target_kernel_dir(kernel_table_test)
add_dependencies(kernel_table_test
  kernel_gemv_M1_K5120_N16384_S1_L1 kernel_gemv_M1_K6144_N5120_S4_L0
  kernel_gemv_M1_K5120_N34816_S8_L0 kernel_gemv_M1_K17408_N5120_S4_L0
  kernel_gemv_M1_K5120_N14336_S2_L0 kernel_gemv_M1_K5120_N248320_S1_L1
  kernel_gemv_bf16_M1_K5120_N128_C16_S16 kernel_gemv_bf16_M1_K5120_N248320
  kernel_gemv_i8w_M1_K5120_N248320)
add_test(NAME kernel_table_test COMMAND kernel_table_test)

# Every decode allocation, sized from the model constants: the two byte totals
# are pinned so a size change has to be argued for in the test's arithmetic.
add_executable(buffers_test runtime/buffers_test.cc)
target_include_directories(buffers_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(buffers_test PRIVATE b70_runtime b70_l0 b70_model)
add_test(NAME buffers_test COMMAND buffers_test)

# Spec 6 §10 (max_len auto): the memory planner on the host - buffers_test's table
# from the size functions alone, spec 6's measured 128k line, max_len_that_fits's
# contract. No device, no checkpoint.
add_executable(memory_plan_test runtime/memory_plan_test.cc)
target_include_directories(memory_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(memory_plan_test PRIVATE b70_plan)
add_test(NAME memory_plan_test COMMAND memory_plan_test)
# The startup log's VRAM chart (runtime/vram_bars.h), host only: 64 cells, free stretches, shades, wrap, ASCII.
add_executable(vram_bars_test runtime/vram_bars_test.cc)
target_include_directories(vram_bars_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(vram_bars_test PRIVATE b70_plan)
add_test(NAME vram_bars_test COMMAND vram_bars_test)
# Spec 13a: the batching probe's row table, its rules and the layouts' memory (runtime/batch_rows.h), host only.
add_executable(batch_rows_test runtime/batch_rows_test.cc)
target_include_directories(batch_rows_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(batch_rows_test PRIVATE b70_plan)
add_test(NAME batch_rows_test COMMAND batch_rows_test)

# Spec 13c / spec 31: the planner's sequences of one card (plan_pool, pool_that_fits), host only.
add_executable(pool_plan_test runtime/pool_plan_test.cc)
target_include_directories(pool_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pool_plan_test PRIVATE b70_plan)
add_test(NAME pool_plan_test COMMAND pool_plan_test)
# Spec 31 (plan 31 Task 3): the slots' KV page allocator (runtime/kv_pool.h), host only.
add_executable(kv_pool_test runtime/kv_pool_test.cc)
target_include_directories(kv_pool_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kv_pool_test PRIVATE b70_plan)
add_test(NAME kv_pool_test COMMAND kv_pool_test)

# The five checkpoint-labelled tests below (loader::load, argv[1] or [3])
# default with no argument to the HF repo id
# "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ", which loader::resolve_snapshot
# (src/loader/snapshot.cc) follows through the cache's refs/main pointer -
# whatever revision was downloaded last, not necessarily one this
# loader accepts: refs/main has pointed at a snapshot whose config.json carries
# a broken non-exclusion dynamic-quant rule that src/loader/quant.cc correctly
# refuses (see the benchmark record). A bare `ctest` must not depend on that pointer,
# so every add_test below for these five passes a snapshot directory
# explicitly - resolve_snapshot takes a directory verbatim, bypassing
# refs/main entirely. An overridable default rather than a hardcoded path: a
# CMake cache variable (B70_TEST_SNAPSHOT), defaulted to the gate checkpoint's
# known-good snapshot, resolved against $HF_HOME (or ~/.cache/huggingface) the
# same way the loader itself does. Override with -DB70_TEST_SNAPSHOT=<dir> at
# configure time if our cache moves the working snapshot; this never
# mutates the HF cache itself.
#
# 2026-09-09 (ruling A31): the gate checkpoint is
# urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ at snapshot 84575a1 (GPTQ g64
# sym, bf16 lm_head, exclusion-only dynamic rules). The previous default,
# Vishva007's 2a90776, was deleted from the test machine on 2026-09-09; its golden set in
# `oracle-out/` belongs to THAT checkpoint's weights and is not reused - the
# golden registrations below take B70_TEST_ORACLE_DIR, one golden set per
# checkpoint (tools/oracle/README.md), and SKIP (77) until it has
# dumped it.
if(DEFINED ENV{HF_HOME})
  set(_b70_hf_home "$ENV{HF_HOME}")
else()
  set(_b70_hf_home "$ENV{HOME}/.cache/huggingface")
endif()
# b70_hf_snapshot(VAR REPO_DIR PINNED SUFFIX TYPE DOC): VAR = REPO_DIR's snapshot in the HF cache - the one refs/main
# names (what is downloaded), PINNED when the cache has no refs/main - plus SUFFIX (a file inside, or empty). A cached
# value that no longer exists (the snapshot moved on: a re-download, a new upstream revision) is replaced; one that
# exists - an override with -D - is kept. (kolibri_tokenizer_test skipped for weeks on a pinned id the test machine no longer
# had, 2026-11-07.)
function(b70_hf_snapshot var repo pinned suffix type doc)
  set(_sha "${pinned}")
  if(EXISTS "${_b70_hf_home}/hub/${repo}/refs/main")
    file(READ "${_b70_hf_home}/hub/${repo}/refs/main" _sha)
    string(STRIP "${_sha}" _sha)
  endif()
  if(NOT DEFINED CACHE{${var}} OR NOT EXISTS "$CACHE{${var}}")
    set(${var} "${_b70_hf_home}/hub/${repo}/snapshots/${_sha}${suffix}" CACHE ${type} "${doc}" FORCE)
  endif()
endfunction()
b70_hf_snapshot(B70_TEST_SNAPSHOT models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
  84575a18f209992ef96d819b31f924b489e3d55d "" PATH "Checkpoint snapshot directory the checkpoint-labelled tests load")
# The CPU reference sets every golden / judged test reads: $B70_DATA/oracle-out/<model>/<set> (tools/oracle/README.md).
# Without B70_DATA the root does not exist and those tests skip (77). A build tree configured before the sets moved
# there caches their old in-tree paths (<source>/oracle-out-<name>): those entries are dropped, so the defaults below
# apply.
if(DEFINED ENV{B70_DATA})
  set(B70_ORACLE "$ENV{B70_DATA}/oracle-out" CACHE PATH "The CPU reference sets' root" FORCE)
elseif(NOT DEFINED CACHE{B70_ORACLE})
  set(B70_ORACLE "${CMAKE_BINARY_DIR}/no-reference-data" CACHE PATH "The CPU reference sets' root (B70_DATA unset)")
endif()
get_cmake_property(_b70_cache CACHE_VARIABLES)
foreach(v IN LISTS _b70_cache)
  if("$CACHE{${v}}" MATCHES "^${CMAKE_SOURCE_DIR}/oracle-out-")
    unset(${v} CACHE)
  endif()
endforeach()
set(B70_TEST_ORACLE_DIR "${B70_ORACLE}/qwen38/golden"
  CACHE PATH "The CPU-oracle golden set dumped from B70_TEST_SNAPSHOT (tools/oracle/golden.sh OUT_DIR)")

# Needs the real 19 GB checkpoint and a B70: label `checkpoint` so it can be
# excluded (`ctest -LE checkpoint`) where the model or the card is absent.
add_executable(load_checkpoint_test loader/load_checkpoint_test.cc)
target_include_directories(load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(load_checkpoint_test PRIVATE b70_loader b70_model)
add_test(NAME load_checkpoint_test COMMAND load_checkpoint_test ${B70_TEST_SNAPSHOT})
set_tests_properties(load_checkpoint_test PROPERTIES LABELS checkpoint TIMEOUT 600)

# Every variant runtime::build binds, by name: a missing .bin is a
# zeModuleCreate failure minutes into a load otherwise (kernel_table_test makes
# the same point for the model table's GEMV rows). One list, because all three
# tests below capture the same decode list. (src/cli/b70-decode deliberately
# depends on ALL kernels instead - see src/cli/CMakeLists.txt.)
#
# None of the three calls `b70_target_kernel_dir`, and that is not an omission:
# they *build* the variants (add_dependencies) but never name a path, because
# `runtime::build` is what opens the modules and B70_KERNEL_DIR is PRIVATE to
# b70_runtime's own compile of capture.cc. Adding the call back would define an
# unused macro in three translation units that do not include kernels/kernels.h.
set(B70_DECODE_LIST_KERNELS
  kernel_embed_gather_M1 kernel_argmax_stage1_M1 kernel_argmax_stage2
  kernel_prep_res_fold_M1_K5120_SP0_G20 kernel_prep_res_fold_M1_K5120_SP4_G20
  kernel_prep_norm_finish_M1_K5120_G20_W20
  kernel_prep_silu_mul_M1 kernel_prep_gated_head_M1 kernel_gdn_step_M1
  kernel_attn_prep_M1 kernel_attn_v2_M1_T${ATTN_V2_TGT}
  kernel_gemv_M1_K5120_N16384_S1_L1 kernel_gemv_M1_K5120_N14336_S2_L0
  kernel_gemv_M1_K6144_N5120_S4_L0 kernel_gemv_M1_K5120_N34816_S8_L0
  kernel_gemv_M1_K17408_N5120_S4_L0 kernel_gemv_M1_K5120_N248320_S1_L1
  kernel_gemv_bf16_M1_K5120_N128_C16_S16 kernel_gemv_bf16_M1_K5120_N248320
  kernel_gemv_i8w_M1_K5120_N248320)

# The captured decode list, replayed: spec §8.7's determinism proof and the
# acceptance for every kernel in plan 3. Needs the real checkpoint and a B70
# (label `checkpoint`), and it is the one test that runs all 774 kernels.
add_executable(replay_determinism_test runtime/replay_determinism_test.cc)
target_include_directories(replay_determinism_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(replay_determinism_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(replay_determinism_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME replay_determinism_test COMMAND replay_determinism_test ${B70_TEST_SNAPSHOT})
set_tests_properties(replay_determinism_test PROPERTIES LABELS checkpoint TIMEOUT 900)
# Spec 9 L4: replay determinism with the int8 lm_head (quantised at load).
add_test(NAME replay_determinism_i8head_test COMMAND replay_determinism_test ${B70_TEST_SNAPSHOT} int8)
set_tests_properties(replay_determinism_i8head_test PROPERTIES LABELS checkpoint TIMEOUT 900)

# The same list driven through runtime::Engine - ingestion, greedy generation,
# reset, the max_len wall and the timing the CLI prints. Same checkpoint, same
# card, so the same label; two runs of 16 + 8 tokens rather than three of 16 + 8
# with full state snapshots, so it is the cheaper of the two.
add_executable(engine_smoke_test runtime/engine_smoke_test.cc)
target_include_directories(engine_smoke_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(engine_smoke_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(engine_smoke_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME engine_smoke_test COMMAND engine_smoke_test ${B70_TEST_SNAPSHOT})
set_tests_properties(engine_smoke_test PROPERTIES LABELS checkpoint TIMEOUT 900)

# The same list captured twice - plain and profiled (plan 4 spec §3.5). It is
# the profiler's calibration: the instrumented list must generate the same 8
# tokens as the plain one, its 774 timestamps must tile the step in walk order,
# and their sum must fit inside the fence wall. Same checkpoint and card as the
# two above, so the same label; ~49 replays of the real step, no oracle dumps.
add_executable(profile_capture_test runtime/profile_capture_test.cc)
target_include_directories(profile_capture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(profile_capture_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(profile_capture_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME profile_capture_test COMMAND profile_capture_test ${B70_TEST_SNAPSHOT})
set_tests_properties(profile_capture_test PROPERTIES LABELS checkpoint TIMEOUT 900)

# The golden gate (spec §11): the engine against the CPU oracle, 3 prompts x 32
# greedy tokens, element-exact, plus the per-layer / GDN-state / logit-row
# diagnostics. Needs what engine_smoke_test needs AND the ~1 GB of oracle dumps
# that live only on the GPU machine - hence the second label and the SKIP_RETURN_CODE:
# absent goldens make the test exit 77, which ctest reports as SKIP rather than
# a failure. Both directories are passed absolute so the test does not depend on
# ctest's working directory; the prompt ids are committed, so a missing .ids
# file is a real failure and not a skip.
add_executable(golden_gate_test golden/golden_gate_test.cc)
target_include_directories(golden_gate_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(golden_gate_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(golden_gate_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME golden_gate_test COMMAND golden_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT})
set_tests_properties(golden_gate_test PROPERTIES
  LABELS "checkpoint;golden" SKIP_RETURN_CODE 77 TIMEOUT 1800)
# Spec 9 L3: the same gate with the int8 lm_head (argv[4]); the oracle stays bf16.
add_test(NAME golden_gate_i8head_test COMMAND golden_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} int8)
set_tests_properties(golden_gate_i8head_test PROPERTIES
  LABELS "checkpoint;golden" SKIP_RETURN_CODE 77 TIMEOUT 1800)

# --- the prefill gates (spec 2 §6; spec 2.1 §2 bar 3) -------------------------------------
# Registered HERE rather than beside the kernel tests above because these need
# B70_TEST_SNAPSHOT and B70_DECODE_LIST_KERNELS, both defined between there and here.
#
# Every binary the prefill walk opens. The decode list is part of it because
# `step_head`'s lm_head and its two argmax stages ARE decode's own M = 1
# binaries -- the prefill path re-uses them rather than compiling a second
# copy, which is also why the module count below is not a new number. Since
# spec 2.1 S2/S3 the L0 backend's own binaries are in the list too: pf_gemm IS
# its linears' and its attention's GEMM, and the slab dequants feed them.
set(B70_PREFILL_KERNELS
  kernel_pf_gemm_small_A2_B2_S4 kernel_pf_gemm_small_A4_B2_S4
  kernel_pf_gemm_small_A2_B2_S4_SILU kernel_pf_gemm_small_A4_B2_S4_SILU
  kernel_pf_gemm_q4_A4_B1_S1_L0 kernel_pf_gemm_q4_A4_B1_S1_L1
  kernel_pf_gemm_q4_A8_B1_S2_L0 kernel_pf_gemm_q4_A8_B1_S2_L1
  kernel_pf_gemm_q4_A1_B2_S4_L0_SILU kernel_pf_gemm_q4_A4_B2_S1_L0_SILU
  kernel_pf_gemm_q4_A8_B1_S4_L0_SLM_SILU
  kernel_pf_embed_gather kernel_pf_res_fold_K5120_SP0_G20
  kernel_pf_res_fold_K5120_SP1_G20 kernel_pf_norm_finish_K5120_G20_W20
  kernel_pf_silu_mul kernel_pf_gated_head kernel_pf_ab_proj kernel_pf_route_slab_K5120_N128_P256 kernel_pf_gdn_conv_AB256
  kernel_pf_attn_prep_q16 kernel_pf_attn kernel_pf_flash_attn kernel_pf_flash_attn_pair kernel_pf_flash_attn_split kernel_pf_flash_merge
  kernel_pf_gdn_conv kernel_pf_gdn_wy kernel_pf_gdn_scan
  kernel_pf_gdn_front_PL kernel_pf_gdn_scan_OB kernel_pf_gated_head_OB kernel_pf_gdn_wy_WB kernel_pf_gdn_intra
  kernel_pf_dequant_tile_K5120_N16384_L1_T0
  kernel_pf_dequant_tile_K6144_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N34816_L0_T0
  kernel_pf_dequant_tile_K17408_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N14336_L0_T0
  kernel_pf_gemm_T0 kernel_pf_gemm_T1 kernel_pf_gemm_T0_SILU
  kernel_pf_dequant_slab_K5120_N16384_L1 kernel_pf_dequant_slab_K6144_N5120_L0
  kernel_pf_dequant_slab_K5120_N34816_L0 kernel_pf_dequant_slab_K17408_N5120_L0
  kernel_pf_dequant_slab_K5120_N14336_L0
  # Open-work 18: the walk of a chunk wider than 2048 rows.
  kernel_pf_gemm_T0_R kernel_pf_gemm_T0_SILU_R
  kernel_pf_dequant_slab_K5120_N16384_L1_BS kernel_pf_dequant_slab_K6144_N5120_L0_BS
  kernel_pf_dequant_slab_K5120_N34816_L0_BS kernel_pf_dequant_slab_K17408_N5120_L0_BS
  kernel_pf_dequant_slab_K5120_N14336_L0_BS
  # Spec 5's h8 path, which `--prefill-backend l0-int8` runs (pf_colmax_rot is in the
  # pf_requant_rot binaries).
  kernel_pf_quant_had_K5120 kernel_pf_quant_had_K6144 kernel_pf_quant_had_K17408
  kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1 kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU
  ${B70_DECODE_LIST_KERNELS})

# The three spec 2 §6 gates. `tests/golden` is on the include path for
# golden_common.h -- the decode gate's machinery, extracted verbatim, so the
# prefill gate grades on the same code rather than on a copy of it.
add_executable(prefill_lazy_setup_test prefill/prefill_lazy_setup_test.cc)
target_include_directories(prefill_lazy_setup_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(prefill_lazy_setup_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(prefill_lazy_setup_test ${B70_PREFILL_KERNELS})
add_test(NAME prefill_lazy_setup_test COMMAND prefill_lazy_setup_test ${B70_TEST_SNAPSHOT}
  bf16 ${CMAKE_CURRENT_BINARY_DIR}/prefill-lazy-setup-bf16.raw)
set_tests_properties(prefill_lazy_setup_test PROPERTIES LABELS "checkpoint;prefill" TIMEOUT 180)
if(B70_KV8)
  add_test(NAME prefill_lazy_setup_kv8_test COMMAND prefill_lazy_setup_test ${B70_TEST_SNAPSHOT}
    int8 ${CMAKE_CURRENT_BINARY_DIR}/prefill-lazy-setup-int8.raw)
  set_tests_properties(prefill_lazy_setup_kv8_test PROPERTIES LABELS "checkpoint;prefill;kv8" TIMEOUT 180)
endif()
foreach(t prefill_gate prefill_consistency prefill_determinism prefill_int8)
  add_executable(${t}_test prefill/${t}_test.cc)
  target_include_directories(${t}_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(${t}_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  add_dependencies(${t}_test ${B70_PREFILL_KERNELS})
endforeach()
# Spec 6 K3a (plan 6b Task 4): flash_long_test, the 32704-id prefill of the committed prompt in
#   both modes (flash, and the composed reference set by the test) on both backends, every logit
#   finite; flash-vs-composed printed as information. A 32k CPU oracle
#   (tools/oracle/last_logits.py) measured 11359 s to reach 8192 ids, so it is not part of the
#   suite; given one, `flash_long_test ... <file>` applies the bar at depth. Its `--128k <ids>`
#   mode (K3b, determinism at max_len 131072) is run by hand. (flash_vs_oracle_test,
#   K3a's golden-set bar, ran the gate under B70_PREFILL_ATTN=composed; it went with the variable
#   in the zero-config cleanup, its bar long met.)
add_executable(flash_long_test prefill/flash_long_test.cc)
target_include_directories(flash_long_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(flash_long_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(flash_long_test ${B70_PREFILL_KERNELS})
add_test(NAME flash_long_test COMMAND flash_long_test ${B70_TEST_SNAPSHOT}
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long32k.ids)
set_tests_properties(flash_long_test PROPERTIES LABELS "checkpoint;prefill"
  SKIP_RETURN_CODE 77 TIMEOUT 3600)
# Prefill continuation (fix-prefill-continuation): N ids in one prefill call against two
# calls split at 1..2049 and against M = 1 decode; per-row KV cosine per FA layer, GDN
# state, last-row logits. Bars in the source.
add_executable(prefill_split_test prefill/prefill_split_test.cc)
target_include_directories(prefill_split_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(prefill_split_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(prefill_split_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
foreach(b l0 l0-int8)
  add_test(NAME prefill_split_${b}_test COMMAND prefill_split_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long32k.ids ${b})
  set_tests_properties(prefill_split_${b}_test PROPERTIES LABELS "checkpoint;prefill"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  # Plan 5d: at chunk cap 4096 (one 2600-row chunk; the split at 2048 = the 2048 chunking, bitwise).
  add_test(NAME prefill_split_${b}_cap4096_test COMMAND prefill_split_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long32k.ids ${b} --chunk-cap 4096)
  set_tests_properties(prefill_split_${b}_cap4096_test PROPERTIES LABELS "checkpoint;prefill"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
endforeach()
# The prompt pass's kernels need no scratch: the walk's pf_* binaries, every entry (its immediate list keeps the
# scratch it first got, which a decode list's larger scratch frees).
add_executable(prefill_scratch_test prefill/prefill_scratch_test.cc)
target_include_directories(prefill_scratch_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(prefill_scratch_test PRIVATE b70_l0)
b70_target_kernel_dir(prefill_scratch_test)
add_dependencies(prefill_scratch_test ${B70_PREFILL_KERNELS})
list(TRANSFORM B70_PREFILL_KERNELS REPLACE "^kernel_" "" OUTPUT_VARIABLE _prefill_bins)
add_test(NAME prefill_scratch_test COMMAND prefill_scratch_test ${_prefill_bins})
# Spec 7 C1 (plan 7b): state and KV snapshots restore bitwise; the block hook.
add_executable(snapshot_test runtime/snapshot_test.cc)
target_include_directories(snapshot_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(snapshot_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(snapshot_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
add_test(NAME snapshot_test COMMAND snapshot_test ${B70_TEST_SNAPSHOT})
# Open-work 18: at chunk cap 4096 the hooks of D-F are unchanged (the tail) and case G checks the mixed chunks.
add_test(NAME snapshot_cap4096_test COMMAND snapshot_test ${B70_TEST_SNAPSHOT} --chunk-cap 4096)
set_tests_properties(snapshot_test snapshot_cap4096_test PROPERTIES LABELS checkpoint
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 900)
# Spec 5 T3 (plan 5b Task 2): l0-int8 on the engine - a smoke cosine against l0 and the
# launch delta step_chunk_launches derives (PERF-113: every linear an h8 walk). The int8 kernels are in
# B70_PREFILL_KERNELS above.
add_test(NAME prefill_int8_test COMMAND prefill_int8_test ${B70_TEST_SNAPSHOT})
set_tests_properties(prefill_int8_test PROPERTIES LABELS "checkpoint;prefill"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600)
# l0, every build.
add_test(NAME prefill_gate_l0_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 0 l0)
set_tests_properties(prefill_gate_l0_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
# Spec 5 T4 (plan 5b Task 3): the same gate on l0-int8, for A2/A3 against the l0 run above
# (tools/rotate/gate_compare.py <l0 log> <int8 log>). The gate's own verdict is the
# token rule; A2/A3 are the comparison's.
add_test(NAME prefill_gate_int8_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 0 l0-int8 1)
set_tests_properties(prefill_gate_int8_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
# Spec 9 L3: the l0 and l0-int8 gates again with the int8 lm_head (argv[8]). argv[7] is
# each backend's standing near-tie allowance (0 on l0, 1 on l0-int8), unchanged - except l0 with the int8 head,
# which takes the same one: cjk row 9, accepted by the owner 2026-11-23 (the engine's 105874 vs the golden's 95895,
# 0.003 apart in the engine's logits and one bf16 ulp apart in the golden, 13.9375 vs 13.875).
add_test(NAME prefill_gate_l0_i8head_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 0 l0 1 int8)
add_test(NAME prefill_gate_int8_i8head_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 0 l0-int8 1 int8)
set_tests_properties(prefill_gate_l0_i8head_test prefill_gate_int8_i8head_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
add_test(NAME prefill_gate_int8_c1000_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 1000 l0-int8 1)
set_tests_properties(prefill_gate_int8_c1000_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
# Multi-chunk on the golden set: chunk 16 puts every prompt through 3 or 4 chunks (the
# quantiser scratch reused across chunks, KV and GDN carried). The plan's multi-chunk pin
# is prefill_gate_long_int8_test below (2820 ids at C = 1024); this pair covers chunk counts
# the long prompt does not.
add_test(NAME prefill_gate_l0_c16_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 16 l0)
add_test(NAME prefill_gate_int8_c16_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 16 l0-int8 1)
set_tests_properties(prefill_gate_l0_c16_test prefill_gate_int8_c16_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
# 3l (2): short turns deep in a session (16 / 64 / 128 rows at 16000 / 32000 keys) with the split-K flash and
# without, each run judged by its own f32 reference (B70_FORCED_JUDGE; SKIP 77 without one). Both KV forms.
add_executable(prefill_turns_test prefill/prefill_turns_test.cc)
target_include_directories(prefill_turns_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(prefill_turns_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(prefill_turns_test ${B70_PREFILL_KERNELS})
add_test(NAME prefill_turns_test COMMAND prefill_turns_test ${B70_TEST_SNAPSHOT} bf16)
set_tests_properties(prefill_turns_test PROPERTIES LABELS "checkpoint;golden;prefill"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600)
if(B70_KV8)
  add_dependencies(prefill_turns_test kernel_pf_attn_prep_kv8 kernel_pf_flash_attn_kv8_raw kernel_pf_attn_gate_kv8
    kernel_pf_flash_attn_kv8_raw_pair kernel_pf_flash_attn_kv8_split kernel_pf_flash_attn_kv8_raw_split)
  add_test(NAME prefill_turns_kv8_test COMMAND prefill_turns_test ${B70_TEST_SNAPSHOT} int8)
  set_tests_properties(prefill_turns_kv8_test PROPERTIES LABELS "checkpoint;golden;prefill;kv8"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
add_test(NAME prefill_consistency_l0_test COMMAND prefill_consistency_test
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} l0)
add_test(NAME prefill_determinism_l0_test COMMAND prefill_determinism_test
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} l0)
set_tests_properties(prefill_consistency_l0_test prefill_determinism_l0_test PROPERTIES
  LABELS "checkpoint;prefill" TIMEOUT 3600)
# Plan 5d: the gates at chunk cap 4096 (--chunk-cap, tests/chunk_cap_arg.h): every chunk-0 case
# runs at the cap, so a prompt above 2048 positions is ONE chunk.
add_test(NAME prefill_gate_l0_cap4096_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 0 l0 --chunk-cap 4096)
add_test(NAME prefill_gate_int8_cap4096_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 0 l0-int8 1 --chunk-cap 4096)
set_tests_properties(prefill_gate_l0_cap4096_test prefill_gate_int8_cap4096_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
add_test(NAME prefill_consistency_l0_cap4096_test COMMAND prefill_consistency_test
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} l0 --chunk-cap 4096)
add_test(NAME prefill_determinism_l0_cap4096_test COMMAND prefill_determinism_test
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} l0 --chunk-cap 4096)
set_tests_properties(prefill_consistency_l0_cap4096_test prefill_determinism_l0_cap4096_test PROPERTIES
  LABELS "checkpoint;prefill" TIMEOUT 3600)

# `Engine::prefill` end to end: the chunk walk, the Control handoff, the
# launch arithmetic and the rejections (plan 6b Task 10), on the L0 backend; section 8
# (ruling E1) measures the L0 chunk's live launch delta.
add_executable(prefill_smoke_test prefill/prefill_smoke_test.cc)
target_include_directories(prefill_smoke_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(prefill_smoke_test PRIVATE b70_prefill_host b70_runtime b70_loader
                      b70_l0 b70_model)
add_dependencies(prefill_smoke_test ${B70_PREFILL_KERNELS})
add_test(NAME prefill_smoke_test COMMAND prefill_smoke_test ${B70_TEST_SNAPSHOT})
set_tests_properties(prefill_smoke_test PROPERTIES LABELS "checkpoint;prefill" TIMEOUT 1800)
# Plan 5d: the smoke at chunk cap 4096 (the scratch, the Int8State and the chunk walk at 4096 rows).
add_test(NAME prefill_smoke_cap4096_test COMMAND prefill_smoke_test ${B70_TEST_SNAPSHOT} --chunk-cap 4096)
set_tests_properties(prefill_smoke_cap4096_test PROPERTIES LABELS "checkpoint;prefill" TIMEOUT 1800)

# The int4-lm_head checkpoint (spec 1.6 §5.1, tools/quantize_qwen38_rtn.sh).
# It never went through `hf download`, so it is a plain path and not a cache
# lookup; the same override rule as B70_TEST_SNAPSHOT applies.
set(B70_RTN_SNAPSHOT "$ENV{HOME}/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64"
  CACHE PATH "The int4-lm_head checkpoint the RTN-labelled gates load")

if(EXISTS "${B70_RTN_SNAPSHOT}/config.json")
  add_test(NAME prefill_gate_rtn_test COMMAND prefill_gate_test
    ${B70_ORACLE}/qwen38/golden-rtn ${CMAKE_SOURCE_DIR}/tests/golden/prompts
    ${B70_RTN_SNAPSHOT})
  set_tests_properties(prefill_gate_rtn_test PROPERTIES
    LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
else()
  message(STATUS "b70: RTN checkpoint absent (${B70_RTN_SNAPSHOT}) -- prefill_gate_rtn_test not registered")
endif()

# Spec 2 §6.2: the multi-chunk golden gate. ONE oracle prompt of >= 2048 ids
# (`tools/oracle/make_long_prompt.sh` builds 2820 from the three committed id
# streams) gated at C = 1024, so every chunk boundary is crossed in
# attention-over-cache AND in the GDN state carry.
#
# **Plan 6b marked this row "expected-limited under L1" and that limitation is
# GONE.** It was the retired `PrefillScratch::kAttnC = 64` sub-chunk of ruling
# R5; ruling A14's composed attention runs the whole chunk in one pass, and
# `prefill_gate_test` at chunk 16 and 1024 already measures 62/62 determined
# rows exact on the short prompts. What is still untested here and nowhere
# else is DEPTH: 2820 positions of chunked GDN carry against 141.
#
# The long prompt's reference is in the primary golden set beside the short three
# (`long.golden.safetensors`, `PROMPTS=long tools/oracle/golden.sh`, tools/oracle/README.md);
# the test SKIPs (77) until it is there. It was wired to the RTN checkpoint until
# 2026-11-12, which left the test machine, so neither test was registered.
add_test(NAME prefill_gate_long_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts
  ${B70_TEST_SNAPSHOT} long 1024)
set_tests_properties(prefill_gate_long_test PROPERTIES
  LABELS "checkpoint;golden;prefill;long" SKIP_RETURN_CODE 77 TIMEOUT 7200)
# Spec 5 (plan 5b Task 3): the same 2820-id, two-chunk gate on l0-int8.
add_test(NAME prefill_gate_long_int8_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts
  ${B70_TEST_SNAPSHOT} long 1024 l0-int8 1)
set_tests_properties(prefill_gate_long_int8_test PROPERTIES
  LABELS "checkpoint;golden;prefill;long" SKIP_RETURN_CODE 77 TIMEOUT 7200)

# --- b70-decode's host-side rejections ---------------------------------------
# The product binary had no automated test at all. These are the argument
# rejections that short-circuit BEFORE `l0::Context` is constructed
# (src/cli/b70_decode.cc: the parse loop, the mode checks, then `read_ids`) - so
# they need neither a GPU nor the 19 GB checkpoint, and they carry no label:
# they belong in the plain suite that runs everywhere.
#
# Each case runs the real binary and requires **exit 1**, the code `main`'s
# catch-all returns. `WILL_FAIL TRUE` would have been shorter and wrong: it
# accepts *any* non-zero status, and a segfault is non-zero, so a CLI that
# crashed on a bad flag would pass it. The `sh -c` wrapper pins the exact
# number, so exit 0, exit 2 and a signal all fail.
#
# It also greps the output for the `device:` line the CLI prints immediately
# after `l0::Context` is built. That is the guard on this block's own premise:
# if one of these rejections ever moved to after the device is opened it would
# still exit 1 (a bogus model path throws in the loader), the test would still
# pass, and it would silently have stopped testing what it says it tests.
# `EXPECT <substring>` pins WHICH rejection fired. Without it a case only
# asserts "exit 1 before the device", which every unknown flag satisfies too -
# so a test for a flag the parser does not know yet passes for the wrong reason
# and keeps passing if the feature is deleted. The default expectation is
# `b70-decode:`, the prefix `main`'s catch-all prints on every rejection (and
# which `usage()`'s own "b70-decode <snapshot-or-repo>" lines do not match), so
# the nine cases written before this argument existed gain a free check and
# none of them changes.
function(b70_cli_reject name)
  cmake_parse_arguments(R "" "EXPECT" "" ${ARGN})
  if(NOT R_EXPECT)
    set(R_EXPECT "b70-decode:")
  endif()
  add_test(NAME ${name} COMMAND sh -c [==[
expect=$1
shift
out=$("$@" 2>&1)
rc=$?
printf '%s\n' "$out" >&2
if [ "$rc" -ne 1 ]
then
  echo "b70_cli_reject: expected exit 1, got $rc" >&2
  exit 1
fi
if printf '%s\n' "$out" | grep -q '^device:'
then
  echo "b70_cli_reject: the rejection happened after l0::Context was built" >&2
  exit 1
fi
if ! printf '%s\n' "$out" | grep -qF -- "$expect"
then
  echo "b70_cli_reject: exit 1, but the message does not contain '$expect'" >&2
  exit 1
fi
]==] b70_cli_reject "${R_EXPECT}" $<TARGET_FILE:b70-decode> ${R_UNPARSED_ARGUMENTS})
  set_tests_properties(${name} PROPERTIES TIMEOUT 60)
endfunction()

# The model argument is a path that does not exist, on purpose: every case below
# must reject before anything tries to open it, and if one stops doing so the
# `device:` grep above turns that into a failure rather than a false pass.
set(B70_NO_MODEL /nonexistent/b70-model)
# tests/cli/ok.ids is three ids from the bench prompt, all in vocabulary. Cases
# that are not about the file's contents use it deliberately, so that a pass
# means the flag combination was rejected and not that the file was bad.
set(B70_OK_IDS ${CMAKE_CURRENT_SOURCE_DIR}/cli/ok.ids)
# tests/cli/oov.ids is one legal id followed by 248320 - `Qwen35::kVocab`
# exactly, the first row `read_ids` rejects (rows [kVocabUsed, kVocab) are
# legitimate; beyond kVocab embed_gather would read past the allocation).
set(B70_OOV_IDS ${CMAKE_CURRENT_SOURCE_DIR}/cli/oov.ids)

# The flag parser's own refusals, and every flag accepted (the run gets past the parser to the model, which does not
# exist) - pinned before the flags became one struct (src/cli/flags.h).
b70_cli_reject(cli_reject_flag_needs_value ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n EXPECT "--n needs a value")
b70_cli_reject(cli_reject_flag_u32_text ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n x
  EXPECT "--n expects a non-negative integer, got 'x'")
b70_cli_reject(cli_reject_flag_u32_range ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4294967296
  EXPECT "--n is out of range: 4294967296")
b70_cli_reject(cli_reject_flag_unknown ${B70_NO_MODEL} --bogus EXPECT "unknown option '--bogus'")
b70_cli_reject(cli_reject_flag_extra ${B70_NO_MODEL} extra EXPECT "unexpected extra argument 'extra'")
b70_cli_reject(cli_reject_flag_no_model --ids ${B70_OK_IDS} --n 4 EXPECT "a snapshot directory or HF repo id is required")
b70_cli_reject(cli_reject_flag_layers_zero ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --layers 0
  EXPECT "--layers 0 would load no layer")
b70_cli_reject(cli_reject_flag_kv_value ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --kv-cache fp8
  EXPECT "--kv-cache / B70_KV_CACHE fp8: expected bf16 or int8")
b70_cli_reject(cli_reject_flag_parallel_value ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --parallel 3
  EXPECT "--parallel expects auto, 1, pp, replica or tp, got '3'")
b70_cli_reject(cli_accept_flags_ids ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --prefill --max-len auto --kv-cache bf16
  --parallel 1 --spec off --prefill-chunk 256 --prefill-backend l0 EXPECT "not a directory and not an <org>/<name> repo id")
b70_cli_reject(cli_accept_flags_bench ${B70_NO_MODEL} --bench --tg 8 --prefill-length 64 --prefill-chunk 64
  --prefill-backend l0-int8 --max-len 4096 --layers 4 EXPECT "--layers belongs to Kolibri-1's and Qwen3.8-Flash-Next's")
b70_cli_reject(cli_accept_flags_profile ${B70_NO_MODEL} --profile --depth 64 --steps 2 --repeats 1 --pipeline-split auto
  EXPECT "it has no --parallel pp")
# --help: exit 0, every flag named.
add_test(NAME cli_decode_help COMMAND sh -c [==[
bin=$1
shift
out=$("$bin" --help 2>&1) || { echo "cli_decode_help: --help did not exit 0" >&2; exit 1; }
for f in "$@"; do printf '%s\n' "$out" | grep -qF -- "$f" || { echo "cli_decode_help: --help does not name $f" >&2; exit 1; }; done
]==] cli_decode_help $<TARGET_FILE:b70-decode> --ids --n --prefill --spec --spec-max --draft-model --max-len --parallel
  --pipeline-split --layers --bench --depth --tg --prefill-length --prefill-chunk --profile --steps --repeats --kv-cache
  --prefill-backend)

# "exactly one of --ids and --bench" - the two modes are exclusive.
b70_cli_reject(cli_reject_ids_and_bench ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --bench)
# "--n belongs to --ids" - --bench sizes its run with --depth and --tg.
b70_cli_reject(cli_reject_n_with_bench ${B70_NO_MODEL} --bench --n 4)
# "--depth and --tg belong to --bench" - the mirror of the case above.
b70_cli_reject(cli_reject_depth_with_ids ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --depth 64)
b70_cli_reject(cli_reject_tg_with_ids ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --tg 8)
# "--prefill-backend belongs to --prefill-length and --prefill" (spec 2.1).
b70_cli_reject(cli_reject_prefill_backend_with_ids ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4
  --prefill-backend l0 EXPECT "--prefill-backend belongs to --prefill-length and --prefill")
# "--prefill-backend expects l0 or l0-int8".
b70_cli_reject(cli_reject_prefill_backend_value ${B70_NO_MODEL} --bench --prefill-length 64
  --prefill-backend cuda EXPECT "--prefill-backend expects l0 or l0-int8")
# PERF-114: l0-int8 on a dense model runs on the int8 context - an explicit --kv-cache bf16 is refused (the missing
# model reads as dense); --turbo is gone, an unknown flag.
b70_cli_reject(cli_reject_int8_kv_bf16 ${B70_NO_MODEL} --bench --prefill-length 64
  --prefill-backend l0-int8 --kv-cache bf16 EXPECT "--prefill-backend l0-int8 runs on the int8 context")
b70_cli_reject(cli_reject_turbo_unknown ${B70_NO_MODEL} --bench --prefill-length 64
  --prefill-backend l0-int8 --turbo EXPECT "unknown option")
# PERF-113: the int8 prompt pass's startup notice - printed with l0-int8 on the dense model, not with l0, and (below,
# B70_ORNITH) not on a MoE model, whose l0-int8 is the W4A8 experts the numbers did not measure. A real load (the
# model's descriptor decides), so `checkpoint`; each run must reach its "prefill backend:" line.
function(b70_int8_notice name want snapshot backend)
  add_test(NAME ${name} COMMAND sh -c [==[
want=$1
shift
out=$("$@" 2>&1)
printf '%s
' "$out" >&2
printf '%s
' "$out" | grep -q '^prefill backend: ' || { echo "b70_int8_notice: the run never reached its prefill" >&2; exit 1; }
if printf '%s
' "$out" | grep -qF 'the int8 prompt pass is on'; then got=1; else got=0; fi
[ "$got" = "$want" ] || { echo "b70_int8_notice: notice printed $got, expected $want" >&2; exit 1; }
]==] b70_int8_notice ${want} $<TARGET_FILE:b70-decode> ${snapshot} --bench --prefill-length 64 --tg 1
    --prefill-backend ${backend})
  set_tests_properties(${name} PROPERTIES LABELS checkpoint TIMEOUT 600)
endfunction()
b70_int8_notice(cli_int8_notice 1 ${B70_TEST_SNAPSHOT} l0-int8)
b70_int8_notice(cli_no_int8_notice 0 ${B70_TEST_SNAPSHOT} l0)
# "unknown option" - anything starting with '-' the parse loop does not know.
b70_cli_reject(cli_reject_unknown_flag ${B70_NO_MODEL} --bench --frobnicate)
# "a snapshot directory or HF repo id is required" - no positional argument.
b70_cli_reject(cli_reject_no_model --bench)
# "--ids needs --n" - the mode is chosen but not sized.
b70_cli_reject(cli_reject_ids_without_n ${B70_NO_MODEL} --ids ${B70_OK_IDS})
# "cannot open --ids file" - a typo'd prompt path must not cost a 13-second load.
b70_cli_reject(cli_reject_missing_ids_file ${B70_NO_MODEL} --ids /nonexistent/prompt.ids --n 4)
# "id 248320 is outside the vocabulary" - read_ids' bound, from the fixture.
b70_cli_reject(cli_reject_oov_id ${B70_NO_MODEL} --ids ${B70_OOV_IDS} --n 4)
# --profile is a third mode, exclusive with the other two: it replays an
# INSTRUMENTED list (774 host-scope event signals per step), so it can never
# produce a bench row, and it sizes its own run with --depth and --steps rather
# than with --ids/--n. Both rejections are the mode check, host-side - and both
# name the message, because "exit 1 before the device" is also what an unknown
# flag does: without EXPECT these two would pass on a binary that had never
# heard of --profile.
b70_cli_reject(cli_reject_profile_and_bench ${B70_NO_MODEL} --profile --bench
  EXPECT "exactly one of --ids, --bench and --profile")
b70_cli_reject(cli_reject_profile_and_ids ${B70_NO_MODEL} --profile --ids ${B70_OK_IDS} --n 4
  EXPECT "exactly one of --ids, --bench and --profile")
# "--depth 0 would ingest nothing" - both synthetic modes measure a step at a
# context depth, and depth 0 is not one.
b70_cli_reject(cli_reject_depth_zero_bench ${B70_NO_MODEL} --bench --depth 0
  EXPECT "--depth 0 would ingest nothing")
b70_cli_reject(cli_reject_depth_zero_profile ${B70_NO_MODEL} --profile --depth 0
  EXPECT "--depth 0 would ingest nothing")
# `--repeats` is repeated-run averaging of the INSTRUMENTED replays and belongs
# to --profile alone (the memo's §5.4 item; src/cli/b70_decode.cc). Both
# rejections are host-side and land before l0::Context, like every case above.
b70_cli_reject(cli_reject_repeats_with_bench ${B70_NO_MODEL} --bench --repeats 5
  EXPECT "--repeats belongs to --profile")
b70_cli_reject(cli_reject_repeats_zero ${B70_NO_MODEL} --profile --repeats 0
  EXPECT "--repeats 0 would profile nothing")
# `--prefill-length` / `--prefill-chunk` / `--prefill` (spec 2, interfaces.md's CLI contract and
# ruling A8; `--pp` / `--pp-chunk` until the 2026-10-06 rename).
# These five are host-side argument validation. Each names its message: "exit 1 before the device" is
# also what an unknown flag does, and without EXPECT these cases would pass on a
# binary that had never heard of the flag.
b70_cli_reject(cli_reject_prefill_length_without_bench ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4
  --prefill-length 64 EXPECT "--prefill-length belongs to --bench")
b70_cli_reject(cli_reject_prefill_length_and_depth ${B70_NO_MODEL} --bench --prefill-length 64
  --depth 64 EXPECT "--prefill-length and --depth are exclusive")
b70_cli_reject(cli_reject_prefill_length_zero ${B70_NO_MODEL} --bench --prefill-length 0
  EXPECT "--prefill-length 0 would prefill nothing")
b70_cli_reject(cli_reject_prefill_chunk_without_length ${B70_NO_MODEL} --bench --prefill-chunk 1024
  EXPECT "--prefill-chunk belongs to --prefill-length")
# `--prefill` is the ids-mode equivalent of `--prefill-length`; it must not be accepted as
# an unknown flag or leak as far as l0::Context.
b70_cli_reject(cli_reject_prefill_without_ids ${B70_NO_MODEL} --bench --prefill
  EXPECT "--prefill belongs to --ids")
# Spec 6 §10: --max-len is `auto` or a positive integer, parsed host-side, before the snapshot or
# the device is touched.
b70_cli_reject(cli_reject_max_len_zero ${B70_NO_MODEL} --bench --max-len 0
  EXPECT "--max-len 0 is not a model length")
b70_cli_reject(cli_reject_max_len_word ${B70_NO_MODEL} --bench --max-len big
  EXPECT "--max-len expects auto or a positive integer")
# Spec 8: `--spec mtp` is --ids mode's greedy speculative decoding (queue row 6's
# long-context checks without the server); parsed and placed before the device like the rest.
b70_cli_reject(cli_reject_mtp_with_bench ${B70_NO_MODEL} --bench --spec mtp
  EXPECT "--spec belongs to --ids")
b70_cli_reject(cli_reject_mtp_value ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --spec mtp --spec-max 4
  EXPECT "expects 1..3")


# --- spec 3 T1: tokenizer (host-only; needs the checkpoint's tokenizer.json) ---
if(B70_TOKENIZER_ENABLED)
  add_executable(tokenizer_smoke_test tokenizer/tokenizer_smoke_test.cc)
  target_include_directories(tokenizer_smoke_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(tokenizer_smoke_test PRIVATE b70_tokenizer)
  add_test(NAME tokenizer_smoke_test COMMAND tokenizer_smoke_test)
  # A tokenizer.json's truncation / padding never apply (a quantised repo has shipped truncation at
  # 2048, which cut every long prompt's tail): a synthetic file, no data needed.
  add_executable(no_truncation_test tokenizer/no_truncation_test.cc)
  target_include_directories(no_truncation_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(no_truncation_test PRIVATE b70_tokenizer)
  add_test(NAME no_truncation_test COMMAND no_truncation_test)
  add_executable(parity_test tokenizer/parity_test.cc)
  target_include_directories(parity_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(parity_test PRIVATE b70_tokenizer)
  add_test(NAME parity_test COMMAND parity_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
  add_executable(template_test tokenizer/template_test.cc)
  target_include_directories(template_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(template_test PRIVATE b70_tokenizer b70_server)
  add_test(NAME template_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
  add_executable(streamer_test tokenizer/streamer_test.cc)
  target_include_directories(streamer_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(streamer_test PRIVATE b70_tokenizer)
  add_test(NAME streamer_test COMMAND streamer_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
  # The server's chat prompt ids are transformers' (Qwen3.8, key order kept): every A4 and edit / copy request's
  # committed .ids, and one session per tests/server/prompt_order_cases.json case whose turn N+1 (the server's own tool
  # calls echoed back) starts with turn N's prompt + generated ids + EOS. Exit 77 without the snapshot.
  add_executable(prompt_order_test server/prompt_order_test.cc)
  target_include_directories(prompt_order_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(prompt_order_test PRIVATE b70_server b70_tokenizer)
  add_test(NAME prompt_order_test COMMAND prompt_order_test ${CMAKE_SOURCE_DIR})
  set_tests_properties(prompt_order_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

  add_executable(golden_server_test server/golden_server_test.cc)
  target_include_directories(golden_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(golden_server_test PRIVATE b70_server)
  add_dependencies(golden_server_test b70-serve b70-decode)
  add_test(NAME golden_server_test COMMAND golden_server_test
    $<TARGET_FILE:b70-serve> $<TARGET_FILE:b70-decode> ${CMAKE_SOURCE_DIR}/tests/golden/prompts
    ${B70_TEST_SNAPSHOT})
  set_tests_properties(golden_server_test PROPERTIES TIMEOUT 600)

  # Host sampling (spec 3.5, plan 7d Task 5): the sampler function alone, over
  # synthetic logits rows -- no device, no checkpoint, no Engine instantiated.
  # It includes serve_adapters.h (where `sample()` and `EngineAdapter` live),
  # so it links what b70-serve links (b70_prefill_host) even though this test
  # never constructs an EngineAdapter.
  add_executable(sampling_test server/sampling_test.cc)
  target_include_directories(sampling_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(sampling_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
    b70_model b70_l0 b70_prefill_host)
  add_test(NAME sampling_test COMMAND sampling_test)

  # Spec 7 C2 (plan 7c Task 3): the prefix cache's request path on the card, cached against
  # cold under the tie-aware rule, on both L0 backends. serve_adapters.h again, so it links
  # what b70-serve links.
  add_executable(prefix_gpu_test server/prefix_gpu_test.cc)
  target_include_directories(prefix_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(prefix_gpu_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
    b70_model b70_l0 b70_prefill_host)
  add_dependencies(prefix_gpu_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
  # l0 takes the near-tie allowance of 1 too (spec 7 §8): sequence e's
  # row 30 flip (0.108 bf16 ulp) is the GDN chunk-boundary rounding class of
  # docs/08-decode-vs-prefill.md ("How prefill got here"), not a cache fault.
  add_test(NAME prefix_gpu_l0_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0 1)
  add_test(NAME prefix_gpu_int8_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0-int8)
  # The opt-in split_last path (b70-serve --prefix-split-last), same sequences, with l0's near-tie allowance of 1:
  # the split only feeds the prompt's last id by a decode replay, and sequence e's row 30 is the same sub-ulp flip.
  add_test(NAME prefix_gpu_split_l0_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0 1 1)
  add_test(NAME prefix_gpu_split_int8_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0-int8 1 1)
  # Open-work 18: the same sequences at 4096-row chunks (Qwen3.8's default where the plan has room).
  add_test(NAME prefix_gpu_l0_cap4096_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0 1 --chunk-cap 4096)
  set_tests_properties(prefix_gpu_l0_test prefix_gpu_int8_test prefix_gpu_split_l0_test
    prefix_gpu_split_int8_test prefix_gpu_l0_cap4096_test PROPERTIES LABELS checkpoint TIMEOUT 1800)
  # Spec 8 M5 (plan 8c): C2 with the MTP head loaded and the adapter at --spec mtp.
  if(B70_MTP)
    add_test(NAME prefix_gpu_mtp_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
      ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0-int8 1 0 3)
    set_tests_properties(prefix_gpu_mtp_test PROPERTIES LABELS checkpoint TIMEOUT 1800)
  endif()
endif()

# --- spec 8 (plan 8b): MTP in the engine ------------------------------------------------
# Every kernel binary is an ALL target, so these depend on the whole kernel set the way the
# CLIs do (src/cli/CMakeLists.txt) rather than on a hand-kept list of the MTP variants.
if(B70_MTP)
  get_property(_b70_mtp_kernels DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels
               PROPERTY BUILDSYSTEM_TARGETS)
  get_property(_b70_mtp_pf_kernels DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
               PROPERTY BUILDSYSTEM_TARGETS)
  foreach(t mtp_head mtp_prefill mtp_verify)
    if(t STREQUAL "mtp_prefill")
      set(_src prefill/${t}_test.cc)
    else()
      set(_src runtime/${t}_test.cc)
    endif()
    add_executable(${t}_test ${_src})
    target_include_directories(${t}_test PRIVATE ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
    target_link_libraries(${t}_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
    add_dependencies(${t}_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
  endforeach()
  # M1: the head's draft logits against tools/oracle/mtp_ref.py --dump (SKIP without it).
  add_test(NAME mtp_head_test COMMAND mtp_head_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_ORACLE}/qwen38/mtp-head/m1
    ${B70_ORACLE}/qwen38/mtp-head)
  set_tests_properties(mtp_head_test PROPERTIES LABELS "checkpoint;golden;mtp"
    SKIP_RETURN_CODE 77 TIMEOUT 1800)
  # Task 3 / Review Focus 2: the head's KV after prefill against M = 1 fills.
  add_test(NAME mtp_prefill_test COMMAND mtp_prefill_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(mtp_prefill_test PROPERTIES LABELS "checkpoint;prefill;mtp"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  add_test(NAME mtp_prefill_cap4096_test COMMAND mtp_prefill_test ${B70_TEST_SNAPSHOT} --chunk-cap 4096)   # plan 5d
  set_tests_properties(mtp_prefill_cap4096_test PROPERTIES LABELS "checkpoint;prefill;mtp"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  # M2 + Review Focus 1-5: verify at M = k + 1, per-row GDN slots, commit, snapshots, replay.
  # Spec 28 N4 / N6 (tests/server/spec_judge.h): Qwen3.8's speculative output graded by the f32 judges of its forced
  # contexts - tools/oracle/qwen_forced.py forced-qwen makes them on the CPU into the data tree's
  # $B70_DATA/oracle-out/qwen38/f32{,-i8head}-forced; the bf16 head's for the checkpoint-head tests, the
  # int8 head's for the tests that load LmHeadForm::Int8. A divergence whose judge is not there yet AWAITS it: the
  # test exits 77 (SKIP) when nothing else failed.
  set(B70_QWEN_FORCED_JUDGE "${B70_ORACLE}/qwen38/f32-forced" CACHE PATH
    "Qwen3.8's f32 judges of the forced contexts, bf16 head (spec_judge.h's B70_FORCED_JUDGE)")
  set(B70_QWEN_FORCED_JUDGE_I8HEAD "${B70_ORACLE}/qwen38/f32-i8head-forced" CACHE PATH
    "Qwen3.8's f32 judges of the forced contexts, the engine's int8 head")
  # The judge by prefill form: the tests that prefill with l0-int8 (every mtp_gpu_test
  # registration) are judged by references whose prompt pass applies the same int8 rounding - f32_refs.sh
  # forced-qwen-pfi8 (qwen_forced.py --pfi8-re), both head forms in one dir (the keys carry the form). PERF-113: int8
  # on every prompt linear, so a new dir and variable name - the old ones hold the MLP-only form's references, and a
  # build tree's cache would keep the old path.
  set(B70_QWEN_FORCED_JUDGE_PFI8ALL "${B70_ORACLE}/qwen38/f32-pfi8all-forced" CACHE PATH
    "Qwen3.8's f32 judges of the forced contexts with the l0-int8 prompt pass (the mtp_gpu_test registrations)")
  # The judge by KV form: the int8 KV cache's twins (mtp_gpu_kv8_test, the DFlash *_kv8_test) are judged by references
  # whose attention reads K / V through the same int8 rounding - f32_refs.sh forced-qwen-kv8 / forced-qwen-pfi8-kv8
  # (qwen_forced.py --kv8: int8_sim.kv8_fq, bitwise kv8.h's codes, kv8_sim_test).
  set(B70_QWEN_FORCED_JUDGE_PFI8ALL_KV8 "${B70_ORACLE}/qwen38/f32-pfi8all-kv8-forced" CACHE PATH
    "Qwen3.8's f32 judges with the l0-int8 prompt pass and the int8 KV cache (mtp_gpu_kv8_test)")
  set(B70_QWEN_FORCED_JUDGE_I8HEAD_KV8 "${B70_ORACLE}/qwen38/f32-i8head-kv8-forced" CACHE PATH
    "Qwen3.8's f32 judges with the int8 head and the int8 KV cache (the DFlash *_kv8_test)")
  # The bf16-unstable rule (2026-11-04, spec_judge.h): standard transformers bf16 references, top-k only
  # (tools/oracle/qwen_bf16_top.py: <key>.bf16top.safetensors, both head forms in one dir), read as B70_FORCED_BF16.
  set(B70_QWEN_FORCED_BF16TOP "${B70_ORACLE}/qwen38/bf16top-forced" CACHE PATH
    "Qwen3.8's standard-bf16 top-k references of the forced contexts (spec_judge.h's bf16-unstable rule)")
  add_test(NAME mtp_verify_test COMMAND mtp_verify_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(mtp_verify_test PROPERTIES LABELS "checkpoint;mtp" SKIP_RETURN_CODE 77
    ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  # Open-work 28: the 16-row verify list (W16 alone; the int8 head, the int8 head's judges).
  add_test(NAME mtp_verify_wide_test COMMAND mtp_verify_test ${B70_TEST_SNAPSHOT} 16384 rule wide)
  set_tests_properties(mtp_verify_wide_test PROPERTIES LABELS "checkpoint;mtp" SKIP_RETURN_CODE 77
    ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 1800)
  # Spec 32: the 32- and 64-row lists bitwise the 16-row list's rows (no judge: the 16-row list's record carries over).
  add_test(NAME mtp_verify_deep_test COMMAND mtp_verify_test ${B70_TEST_SNAPSHOT} 16384 rule deep)
  set_tests_properties(mtp_verify_deep_test PROPERTIES LABELS "checkpoint;mtp" SKIP_RETURN_CODE 77
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 1800)
  # Spec 13c (plan 13c Tasks 2-3): runtime::Engine's sequence slots on the KV page pool - S1 a slot alone is today's
  # engine, S2 the slots' isolation (prefill, decode, reset, snapshots), S3 B1 / B3 for the row kernels (bitwise, mixed K and depth), S4 B1
  # by the f32 judge (the int8 head's references; a divergence without one AWAITS it: 77), S5 determinism. Both KV forms.
  if(B70_VERIFY_M8 AND B70_KV8)   # the row builds (src/kernels: B70_MTP, B70_VERIFY_M8, B70_KV8)
    add_executable(pool_test runtime/pool_test.cc)
    target_include_directories(pool_test PRIVATE ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/tests/server)
    target_link_libraries(pool_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
    add_dependencies(pool_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
    add_test(NAME pool_test COMMAND pool_test ${B70_TEST_SNAPSHOT})
    add_test(NAME pool_kv8_test COMMAND pool_test ${B70_TEST_SNAPSHOT})
    add_test(NAME pool_nomtp_test COMMAND pool_test ${B70_TEST_SNAPSHOT} --mtp off)
    # Spec 31 (plan 31 Task 5): the KV page pool's own cases (pool_test.cc's --pool: pages in any order, reset, growth,
    # a full pool, shared prefix pages).
    add_test(NAME pool_pages_test COMMAND pool_test ${B70_TEST_SNAPSHOT} --pool)
    add_test(NAME pool_pages_kv8_test COMMAND pool_test ${B70_TEST_SNAPSHOT} --pool)
    set_tests_properties(pool_pages_test PROPERTIES LABELS "checkpoint;mtp;pool" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 3600)
    set_tests_properties(pool_pages_kv8_test PROPERTIES LABELS "checkpoint;mtp;pool;kv8" SKIP_RETURN_CODE 77 ENVIRONMENT "B70_KV_CACHE=int8"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
    set_tests_properties(pool_nomtp_test PROPERTIES LABELS "checkpoint;pool" SKIP_RETURN_CODE 77
      ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
    set_tests_properties(pool_test PROPERTIES LABELS "checkpoint;mtp;pool" SKIP_RETURN_CODE 77
      ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
    set_tests_properties(pool_kv8_test PROPERTIES LABELS "checkpoint;mtp;pool;kv8" SKIP_RETURN_CODE 77
      ENVIRONMENT "B70_KV_CACHE=int8;B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  endif()
  # Plan 8c: M3 (greedy MTP is identical to plain), RF1 on the card, M5, M4 end to end,
  # through EngineAdapter (the server's engine calls); and the D1/D2 bench (--bench K).
  if(B70_TOKENIZER_ENABLED)
    add_executable(mtp_gpu_test server/mtp_gpu_test.cc)
    target_include_directories(mtp_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
    target_link_libraries(mtp_gpu_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
      b70_model b70_l0 b70_prefill_host)
    add_dependencies(mtp_gpu_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
    add_test(NAME mtp_gpu_test COMMAND mtp_gpu_test ${B70_TEST_SNAPSHOT} l0-int8)
    set_tests_properties(mtp_gpu_test PROPERTIES LABELS "checkpoint;mtp" SKIP_RETURN_CODE 77
      ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_PFI8ALL};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 5400)
    # Open-work 27: prompt lookup inside MTP through EngineAdapter - D2 against the plain loop on golden, A4 and the
    # edit / copy set's 8k / 32k prompts by the f32 judge (the int8 head's references), the warm head, a truncate
    # inside a lookup burst, a seeded sampled run reproducible.
    add_executable(mtp_lookup_gpu_test server/mtp_lookup_gpu_test.cc)
    target_include_directories(mtp_lookup_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
    target_link_libraries(mtp_lookup_gpu_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
      b70_model b70_l0 b70_prefill_host)
    add_dependencies(mtp_lookup_gpu_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
    add_test(NAME mtp_lookup_gpu_test COMMAND mtp_lookup_gpu_test ${B70_TEST_SNAPSHOT})
    set_tests_properties(mtp_lookup_gpu_test PROPERTIES LABELS "checkpoint;mtp" SKIP_RETURN_CODE 77
      ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 7200)
  endif()
  # Spec 8 §12 (MTP at every context length; validation queue row 6): mtp_verify_test (M2, RF3 at
  # max_len - 4, RF4 deep - spec 7 C1 with MTP at ~70k on the 131072 run) and mtp_gpu_test's
  # M3 gate at max_len 32768 and 131072. Label `longctx`. Each exits 77 (SKIP) at once unless
  # B70_LONGCTX_TESTS=1, so a plain `ctest` costs what it did; the validation run (r6.mtp_long)
  # sets it. The variable is read here, by the wrapper, not by the tests.
  set(_b70_longctx_gate [==[
if [ "${B70_LONGCTX_TESTS:-0}" != 1 ]
then
  echo "SKIP: a long-context MTP variant (validation queue row 6); B70_LONGCTX_TESTS=1 runs it"
  exit 77
fi
exec "$@"
]==])
  foreach(_len 32768 131072)
    if(_len EQUAL 32768)
      set(_tag 32k)
    else()
      set(_tag 128k)
    endif()
    add_test(NAME mtp_verify_${_tag}_test COMMAND sh -c "${_b70_longctx_gate}" b70_longctx
      $<TARGET_FILE:mtp_verify_test> ${B70_TEST_SNAPSHOT} ${_len})
    set_tests_properties(mtp_verify_${_tag}_test PROPERTIES LABELS "checkpoint;mtp;longctx"
      SKIP_RETURN_CODE 77 ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 7200)
    if(TARGET mtp_gpu_test)
      add_test(NAME mtp_gpu_${_tag}_test COMMAND sh -c "${_b70_longctx_gate}" b70_longctx
        $<TARGET_FILE:mtp_gpu_test> ${B70_TEST_SNAPSHOT} l0-int8 --max-len ${_len})
      # The bf16-unstable rule's references as for mtp_gpu_test (the same prompts and positions;
      # their own contexts' bf16 top-k files come from f32_refs.sh bf16top-qwen over r69.forced_dumps' longctx dumps).
      set_tests_properties(mtp_gpu_${_tag}_test PROPERTIES LABELS "checkpoint;mtp;longctx"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_PFI8ALL};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 7200)
    endif()
  endforeach()
endif()

# Spec 6 §10 (max_len auto): the memory plan against the card - every planned component
# and every buffer equal to what the engine allocated, at max_len 16384 and at `auto`
# (the largest the plan fits with the default 1.5 GB reserve, reached as the CLIs reach
# it). At `auto` the engine fills the card to the reserve, so this is also where a
# too-small reserve shows up as an allocation failure. Depends on the whole kernel set,
# like the CLIs: the MTP variant captures the draft and verify lists.
get_property(_b70_plan_kernels DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
get_property(_b70_plan_pf_kernels DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
             PROPERTY BUILDSYSTEM_TARGETS)
add_executable(memory_plan_box_test runtime/memory_plan_box_test.cc)
target_include_directories(memory_plan_box_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(memory_plan_box_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(memory_plan_box_test ${_b70_plan_kernels} ${_b70_plan_pf_kernels})
add_test(NAME memory_plan_box_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT})
set(_b70_plan_box_tests memory_plan_box_test)
if(B70_MTP)
  add_test(NAME memory_plan_box_mtp_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT} mtp)
  list(APPEND _b70_plan_box_tests memory_plan_box_mtp_test)
endif()
set_tests_properties(${_b70_plan_box_tests} PROPERTIES LABELS checkpoint
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 1800)
# --- Spec 15c: Ornith 1.5 35B-A3B decode on the card (label `ornith`) ----------------
# The int4 checkpoint is our AutoRound W4A16 g64 export,
# urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ (spec 15 decision 1, settled in §13);
# B70_ORNITH_SNAPSHOT names it (resolved in the HF cache). Until it is on the machine
# ornith_decode_test SKIPs (77), and the golden registrations SKIP on the absent
# `$B70_DATA/oracle-out/ornith/golden` (15a's golden set, plan 15a) before they load anything. Its
# tokenizer.json gives the committed tests/golden/prompts ids (checked, spec 15 §13), so
# B70_ORNITH_PROMPTS_DIR stays the committed directory.
#   R3 (decode): golden_gate_test, descriptor-driven since 15c (the tap is the layer
#       output when the FFN folds inside the layer), bf16 and int8 heads;
#   R2 + the list + replay bitwise + the plan: ornith_decode_test.
if(B70_ORNITH)
  set(B70_ORNITH_SNAPSHOT "urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ"
    CACHE STRING "Ornith int4 checkpoint (repo id resolved in the HF cache, or a snapshot directory)")
  set(B70_ORNITH_ORACLE_DIR "${B70_ORACLE}/ornith/golden"
    CACHE PATH "The CPU-oracle golden set of Ornith (spec 15a), with router_logits.L* for R2")
  # the token gates' judge (tests/golden/golden_common.h, `--judge=`) -
  # the f32 teacher-forced reference, tools/oracle/ornith_ref.py <data tree> ornith.
  set(B70_ORNITH_F32_DIR "${B70_ORACLE}/ornith/f32"
    CACHE PATH "Ornith's f32 teacher-forced reference (the golden gates' judge)")
  # ... and its quantisation-aware twins (tools/oracle/int8_sim.py; f32_oracle.sh ornith-i8head / ornith-pfi8):
  # the int8 head, and the prompt forward through spec 5's l0-int8 linears.
  set(B70_ORNITH_F32_I8HEAD_DIR "${B70_ORACLE}/ornith/f32-i8head"
    CACHE PATH "Ornith's f32 reference through the int8 head (the _i8head gate's judge)")
  set(B70_ORNITH_F32_PFI8_DIR "${B70_ORACLE}/ornith/f32-pfi8"
    CACHE PATH "Ornith's f32 reference with the l0-int8 prefill linears (the l0-int8 prefill gates' judge)")
  set(B70_ORNITH_PROMPTS_DIR "${CMAKE_SOURCE_DIR}/tests/golden/prompts"
    CACHE PATH "Ornith's prompt ids (the committed ones: the int4 checkpoint's tokenizer gives them)")
  b70_int8_notice(cli_no_int8_notice_ornith 0 ${B70_ORNITH_SNAPSHOT} l0-int8)   # PERF-113: no Qwen3.8 numbers here
  # The greedy argmax is Qwen3.8's argmax_stage1_M1 (vocab_used 248077, spec 15 §13); a||b is
  # the int4 checkpoint's gemv_M1_K2048_N128_S1_L1, its bf16 form kept for a checkpoint that
  # ships a||b bf16.
  set(B70_ORNITH_DECODE_KERNELS
    kernel_embed_gather_M1_D2048 kernel_argmax_stage1_M1 kernel_argmax_stage2
    kernel_prep_res_fold_M1_K2048_SP0_G20 kernel_prep_res_fold_M1_K2048_SP4_G20
    kernel_prep_norm_finish_M1_K2048_G20_W20 kernel_prep_gated_head_M1_GK16V32
    kernel_gdn_step_M1_GK16V32 kernel_attn_prep_M1_Q16KV2 kernel_attn_v2_M1_T${ATTN_V2_TGT}_Q16KV2
    kernel_gemv_M1_K2048_N12288_S1_L1 kernel_gemv_M1_K4096_N2048_S4_L0
    kernel_gemv_M1_K2048_N9216_S2_L0 kernel_gemv_M1_K2048_N248320_S1_L1
    kernel_gemv_M1_K2048_N128_S1_L1
    kernel_gemv_bf16_M1_K2048_N128_C16_S16 kernel_gemv_bf16_M1_K2048_N272_C16_S16
    kernel_gemv_bf16_M1_K2048_N248320 kernel_gemv_i8w_M1_K2048_N248320
    kernel_moe_M1_E256_T8_D2048_I512)
  add_dependencies(golden_gate_test ${B70_ORNITH_DECODE_KERNELS})
  add_test(NAME golden_gate_ornith_test COMMAND golden_gate_test
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT} --judge=${B70_ORNITH_F32_DIR})
  add_test(NAME golden_gate_ornith_i8head_test COMMAND golden_gate_test
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT} int8 --judge=${B70_ORNITH_F32_I8HEAD_DIR}
    --accept=code:3,cjk:15)   # known divergences, 2026-10-23 (debug 2026-10-26)
  set_tests_properties(golden_gate_ornith_test golden_gate_ornith_i8head_test PROPERTIES
    LABELS "checkpoint;golden;ornith" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  # PERF-64: the served form (the int8 head) over the int8 KV cache, by the same f32 judge and the bf16 list plus
  # prose row 18, accepted by the owner 2026-11-22: the engine picks 32 (the bf16 golden's token), the f32 judge 760
  # by 0.456 logit (7 bf16 ulp; the bf16 cache passes this row).
  add_test(NAME golden_gate_ornith_i8head_kv8_test COMMAND golden_gate_test
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT} int8 --judge=${B70_ORNITH_F32_I8HEAD_DIR}
    --accept=code:3,cjk:15,prose:18=32)
  set_tests_properties(golden_gate_ornith_i8head_kv8_test PROPERTIES LABELS "checkpoint;golden;ornith;kv8"
    SKIP_RETURN_CODE 77 TIMEOUT 3600 ENVIRONMENT "B70_KV_CACHE=int8")

  add_executable(ornith_decode_test golden/ornith_decode_test.cc)
  target_include_directories(ornith_decode_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  # Engine::memory_use / memory_line live in b70_prefill_host (engine_prefill.cc), as for ornith_mtp_test.
  target_link_libraries(ornith_decode_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  add_dependencies(ornith_decode_test ${B70_ORNITH_DECODE_KERNELS})
  # R2's judge is the f32 reference (the ruling above); the routes are teacher-forced, so the head form does not
  # move them and both registrations take the checkpoint-head judge.
  add_test(NAME ornith_decode_test COMMAND ornith_decode_test ${B70_ORNITH_SNAPSHOT}
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} --judge=${B70_ORNITH_F32_DIR})
  add_test(NAME ornith_decode_i8head_test COMMAND ornith_decode_test ${B70_ORNITH_SNAPSHOT}
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} int8 --judge=${B70_ORNITH_F32_DIR})
  set_tests_properties(ornith_decode_test ornith_decode_i8head_test PROPERTIES
    LABELS "checkpoint;golden;ornith" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()

# --- Spec 12b: the int8 KV cache (`--kv-cache int8`, src/kernels/kv8.cl) ----------------
# Written blind on the Mac; the validation queue row 11 is the order
# to run these in. `kv8` labels all of them: `ctest -L kv8`.
if(B70_KV8)
  set(B70_KV8_KERNELS
    kernel_attn_prep_M1_KV8 kernel_attn_prep_M2_KV8 kernel_attn_prep_M3_KV8 kernel_attn_prep_M4_KV8
    kernel_attn_v2_M1_T${ATTN_V2_TGT}_KV8 kernel_attn_v2_M2_T${ATTN_V2_TGT}_KV8
    kernel_attn_v2_M3_T${ATTN_V2_TGT}_KV8 kernel_attn_v2_M4_T${ATTN_V2_TGT}_KV8
    kernel_pf_attn_prep_kv8 kernel_pf_flash_attn_kv8 kernel_pf_flash_attn_kv8_raw kernel_pf_flash_attn_kv8_raw_pair
    kernel_pf_attn_gate_kv8 kernel_pf_flash_attn_kv8_split kernel_pf_flash_attn_kv8_raw_split
    kernel_pf_attn_prep_kv8_I8S kernel_pf_flash_attn_kv8_raw_pair_I8S)   # PERF-114 (the int8 pass's scores)
  if(B70_MTP)
    list(APPEND B70_KV8_KERNELS kernel_attn_prep_M1_S1_KV8 kernel_attn_prep_M2_S1_KV8
      kernel_attn_prep_M3_S1_KV8 kernel_attn_prep_M4_S1_KV8)
  endif()

  # The kernels against tests/kernels/kv8_ref.h, no checkpoint: the writers BITWISE (decode
  # M = 1..4, the MTP head's S1, prefill), decode v2 over int8 against the fp64 attention
  # (cosine >= 0.99999 per head) with verify rows bitwise the M = 1 steps, the flash + gate
  # (pf_o >= 0.99999; gated >= 0.9999, PROVISIONAL).
  add_executable(kv8_kernels_test kernels/kv8_kernels_test.cc)
  target_include_directories(kv8_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(kv8_kernels_test PRIVATE b70_l0)
  b70_target_kernel_dir(kv8_kernels_test)
  add_dependencies(kv8_kernels_test ${B70_KV8_KERNELS})
  if(B70_VERIFY_M8)
    target_compile_definitions(kv8_kernels_test PRIVATE B70_TEST_VERIFY_M8=1)
    foreach(MM 5 6 7 8)
      add_dependencies(kv8_kernels_test kernel_attn_prep_M${MM}_KV8 kernel_attn_v2_M${MM}_T${ATTN_V2_TGT}_KV8)
    endforeach()
  endif()
  add_test(NAME kv8_kernels_test COMMAND kv8_kernels_test)
  set_tests_properties(kv8_kernels_test PROPERTIES LABELS kv8 TIMEOUT 1800)

  add_ocloc_kernel(kv8_flash_control SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kv8_flash_control.cl
                   DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_executable(kv8_flash_bitwise_test kernels/kv8_flash_bitwise_test.cc)
  target_include_directories(kv8_flash_bitwise_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kv8_flash_bitwise_test PRIVATE b70_l0)
  b70_target_kernel_dir(kv8_flash_bitwise_test)
  add_dependencies(kv8_flash_bitwise_test kernel_kv8_flash_control
    kernel_pf_flash_attn_kv8 kernel_pf_flash_attn_kv8_raw kernel_pf_flash_attn_kv8_raw_pair kernel_pf_attn_gate_kv8)
  # 3l (2): the split-K short-chunk attention (both KV forms) against today's kernels and fp64.
  add_executable(flash_split_test prefill/flash_split_test.cc)
  target_include_directories(flash_split_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/prefill)
  target_link_libraries(flash_split_test PRIVATE b70_l0)
  b70_target_kernel_dir(flash_split_test)
  add_dependencies(flash_split_test kernel_pf_flash_attn kernel_pf_flash_attn_split kernel_pf_flash_merge
    kernel_pf_flash_attn_kv8_raw kernel_pf_flash_attn_kv8_raw_split kernel_pf_flash_attn_kv8_split)
  add_test(NAME flash_split_test COMMAND flash_split_test)
  set_tests_properties(flash_split_test PROPERTIES LABELS kv8 TIMEOUT 1800)
  add_test(NAME kv8_flash_bitwise_test COMMAND kv8_flash_bitwise_test)
  set_tests_properties(kv8_flash_bitwise_test PROPERTIES LABELS kv8 TIMEOUT 1800)
  add_test(NAME kv8_flash_raw_bitwise_test COMMAND kv8_flash_bitwise_test
    ${B70_KERNEL_DIR}/pf_flash_attn_kv8_raw.bin --raw)
  set_tests_properties(kv8_flash_raw_bitwise_test PROPERTIES LABELS kv8 TIMEOUT 1800)
  add_test(NAME kv8_flash_raw_pair_bitwise_test COMMAND kv8_flash_bitwise_test
    ${B70_KERNEL_DIR}/pf_flash_attn_kv8_raw_pair.bin --raw --pair)
  set_tests_properties(kv8_flash_raw_pair_bitwise_test PROPERTIES LABELS kv8 TIMEOUT 1800)
  # PERF-114 (the int8 prompt pass): the int8-score writer and pair (kv8.cl KV8_I8S) - the writer bitwise its host twin, the pair
  # bitwise the bf16 pair on power-of-two q scales (and paged bitwise contiguous), real scales against the CPU twin.
  add_executable(kv8_i8s_test kernels/kv8_i8s_test.cc)
  target_include_directories(kv8_i8s_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(kv8_i8s_test PRIVATE b70_l0)
  b70_target_kernel_dir(kv8_i8s_test)
  add_dependencies(kv8_i8s_test kernel_pf_attn_prep_kv8_I8S kernel_pf_flash_attn_kv8_raw_pair
    kernel_pf_flash_attn_kv8_raw_pair_I8S)
  if(B70_MTP AND B70_VERIFY_M8)
    add_dependencies(kv8_i8s_test kernel_pf_flash_attn_kv8_raw_pair_I8S_PG${KV_PAGE})
  endif()
  if(B70_TP)
    add_dependencies(kv8_i8s_test kernel_pf_flash_attn_kv8_raw_Q12KV2 kernel_pf_flash_attn_kv8_raw_pair_I8S_Q12KV2)
  endif()
  add_test(NAME kv8_i8s_test COMMAND kv8_i8s_test)
  set_tests_properties(kv8_i8s_test PROPERTIES LABELS kv8 SKIP_RETURN_CODE 77 TIMEOUT 1800)
  # PERF-64: the int8 KV cache's binaries at Ornith's 16 / 2 heads bitwise their Qwen3.8 twins on the same heads (the
  # writers, decode v2, the flash forms and the gate), and its paged prefill twins bitwise the unpaged ones.
  if(B70_ORNITH)
    add_executable(kv8_ornith_test kernels/kv8_ornith_test.cc)
    target_include_directories(kv8_ornith_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(kv8_ornith_test PRIVATE b70_l0)
    b70_target_kernel_dir(kv8_ornith_test)
    get_property(_kv8o_k DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
    get_property(_kv8o_pf DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
    add_dependencies(kv8_ornith_test ${_kv8o_k} ${_kv8o_pf})
    add_test(NAME kv8_ornith_test COMMAND kv8_ornith_test)
    set_tests_properties(kv8_ornith_test PROPERTIES LABELS "kv8;ornith" TIMEOUT 1800)
  endif()

  add_ocloc_kernel(kv8_scale_test SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kv8_scale_test.cl
    DEFINES KV8_FLASH=1 KV8_RAW=1 KT=64 RPW=8 HPW=6 QREG=0 EXP2=1
    DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/kv8.cl OPTIONS -cl-intel-256-GRF-per-thread)
  add_executable(kv8_scale_test kernels/kv8_scale_test.cc)
  target_include_directories(kv8_scale_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kv8_scale_test PRIVATE b70_l0)
  b70_target_kernel_dir(kv8_scale_test)
  add_dependencies(kv8_scale_test kernel_kv8_scale_test)
  add_test(NAME kv8_scale_test COMMAND kv8_scale_test)
  set_tests_properties(kv8_scale_test PROPERTIES LABELS kv8 TIMEOUT 60)

  # 3f: production and probe loads versus the frozen reader.
  foreach(_m 1 2 3 4 8)
    set(_rows "")
    if(_m EQUAL 8)
      set(_rows BATCH_ROWS=1 BROW_WORDS=8 BROW_POS=0 BROW_KVB=1 BROW_SLOT=4)
    endif()
    foreach(_kind 0 1 2)
      set(_name kv8_decode_probe${_kind}_M${_m})
      add_ocloc_kernel(${_name} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kv8_decode_control.cl
        DEFINES CTRL_POS=0 CTRL_NACT=1 M=${_m} TGT=32 KV8_PROBE_KIND=${_kind} ${_rows}
        DEPENDS ${CMAKE_SOURCE_DIR}/tools/probe/kv8_decode_loads.cl)
      list(APPEND KV8_DECODE_PROBE_KERNELS kernel_${_name})
    endforeach()
  endforeach()
  add_ocloc_kernel(kv8_decode_scale_prod SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kv8_decode_scale_prod.cl
    DEFINES CTRL_POS=0 CTRL_NACT=1 KV8_DEC=1 M=1 TGT=32 DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/kv8.cl)
  add_executable(kv8_decode_bitwise_test kernels/kv8_decode_bitwise_test.cc)
  target_include_directories(kv8_decode_bitwise_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kv8_decode_bitwise_test PRIVATE b70_l0)
  b70_target_kernel_dir(kv8_decode_bitwise_test)
  add_dependencies(kv8_decode_bitwise_test ${KV8_DECODE_PROBE_KERNELS} kernel_kv8_decode_scale_prod
    kernel_attn_v2_M1_T32_KV8 kernel_attn_v2_M2_T32_KV8 kernel_attn_v2_M3_T32_KV8
    kernel_attn_v2_M4_T32_KV8)
  if(TARGET kernel_attn_v2_M8_T32_KV8_ROWS)
    target_compile_definitions(kv8_decode_bitwise_test PRIVATE B70_KV8_TEST_ROWS=1)
    add_dependencies(kv8_decode_bitwise_test kernel_attn_v2_M8_T32_KV8_ROWS)
    # 3j: the production row reader at M = 2..7 too, each against the frozen reader's BATCH_ROWS build.
    foreach(_m 2 3 4 5 6 7)
      add_ocloc_kernel(kv8_decode_probe0_M${_m}_ROWS SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kv8_decode_control.cl
        DEFINES CTRL_POS=0 CTRL_NACT=1 M=${_m} TGT=32 KV8_PROBE_KIND=0
                BATCH_ROWS=1 BROW_WORDS=8 BROW_POS=0 BROW_KVB=1 BROW_SLOT=4
        DEPENDS ${CMAKE_SOURCE_DIR}/tools/probe/kv8_decode_loads.cl)
      add_dependencies(kv8_decode_bitwise_test kernel_kv8_decode_probe0_M${_m}_ROWS kernel_attn_v2_M${_m}_T32_KV8_ROWS)
    endforeach()
  endif()
  add_test(NAME kv8_decode_bitwise_test COMMAND kv8_decode_bitwise_test)
  set_tests_properties(kv8_decode_bitwise_test PROPERTIES LABELS kv8 TIMEOUT 600)

  # The gates of spec 12 §4 over the int8 cache: the SAME binaries and command lines as the
  # bf16 registrations they twin, with B70_KV_CACHE=int8 (runtime::default_kv_cache - the
  # Engine's, the planner's and b70-decode's default). Bars unchanged unless named:
  #   Q2   golden_gate_kv8 / _i8head_kv8 (decode), prefill_gate_{int8,l0}_kv8 - the tie
  #        rule's allowance unchanged; kv8_vs_oracle_test - mean logit cos vs the oracle
  #        within 1e-4 of bf16 KV's (tools/probe/kv8_vs_oracle.sh; the Qwen3.8 repeat of 12a).
  #   Q3   flash_long_kv8_test - 1 - cos(int8 KV, bf16 KV) <= 2e-3 at each of 4k..32k and
  #        <= 5e-4 averaged over them (the Qwen3.8 repeat of 12a);
  #        passkey 3/3 at 120k (and at 250k, which only int8 fits) is run by hand:
  #        B70_KV_CACHE=int8 tools/probe/passkey.sh l0-int8.
  #   Q4   A4 by hand: B70_KV_CACHE=int8 tools/toolcall/engine_generate.sh into its own dir.
  #   Q5   replay / prefill determinism and replay bitwise; snapshot_kv8 (spec 7, int8 rows +
  #        scales); mtp_verify_kv8 (spec 8 M2: verify rows bitwise the plain steps) and
  #        mtp_gpu_kv8 (greedy MTP identical to plain); prefix_gpu_*_kv8; the planner against
  #        the allocations (memory_plan_box_kv8).
  set(_kv8_env "B70_KV_CACHE=int8")
  macro(b70_kv8_twin name)
    if(TEST ${name}_kv8_test)
      set_tests_properties(${name}_kv8_test PROPERTIES ENVIRONMENT "${_kv8_env}")
      get_test_property(${name}_test LABELS _kv8_labels)
      if(NOT _kv8_labels)
        set(_kv8_labels "")
      endif()
      list(APPEND _kv8_labels kv8)
      set_tests_properties(${name}_kv8_test PROPERTIES LABELS "${_kv8_labels}")
      get_test_property(${name}_test TIMEOUT _kv8_timeout)
      if(_kv8_timeout)
        set_tests_properties(${name}_kv8_test PROPERTIES TIMEOUT ${_kv8_timeout})
      endif()
      get_test_property(${name}_test SKIP_RETURN_CODE _kv8_skip)
      if(_kv8_skip)
        set_tests_properties(${name}_kv8_test PROPERTIES SKIP_RETURN_CODE ${_kv8_skip})
      endif()
      get_test_property(${name}_test WORKING_DIRECTORY _kv8_wd)
      if(_kv8_wd)
        set_tests_properties(${name}_kv8_test PROPERTIES WORKING_DIRECTORY ${_kv8_wd})
      endif()
    endif()
  endmacro()

  foreach(exe replay_determinism_test engine_smoke_test golden_gate_test prefill_gate_test prefill_lazy_setup_test
              prefill_determinism_test prefill_consistency_test snapshot_test
              flash_long_test memory_plan_box_test mtp_verify_test mtp_gpu_test prefix_gpu_test)
    if(TARGET ${exe})
      add_dependencies(${exe} ${B70_KV8_KERNELS})
    endif()
  endforeach()

  add_test(NAME replay_determinism_kv8_test COMMAND replay_determinism_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(replay_determinism)
  add_test(NAME engine_smoke_kv8_test COMMAND engine_smoke_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(engine_smoke)
  add_test(NAME golden_gate_kv8_test COMMAND golden_gate_test
    ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(golden_gate)
  add_test(NAME golden_gate_i8head_kv8_test COMMAND golden_gate_test
    ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} int8)
  b70_kv8_twin(golden_gate_i8head)
  add_test(NAME prefill_gate_int8_kv8_test COMMAND prefill_gate_test
    ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
    prose,code,cjk 0 l0-int8 1)
  b70_kv8_twin(prefill_gate_int8)
  add_test(NAME prefill_gate_l0_kv8_test COMMAND prefill_gate_test
    ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
    prose,code,cjk 0 l0)
  b70_kv8_twin(prefill_gate_l0)
  add_test(NAME kv8_vs_oracle_test COMMAND ${CMAKE_SOURCE_DIR}/tools/probe/kv8_vs_oracle.sh
    ${B70_TEST_ORACLE_DIR} ${B70_TEST_SNAPSHOT})
  set_tests_properties(kv8_vs_oracle_test PROPERTIES LABELS "checkpoint;golden;prefill;kv8"
    SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  add_test(NAME flash_long_kv8_test COMMAND flash_long_test ${B70_TEST_SNAPSHOT} --kv8
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long32k.ids)
  set_tests_properties(flash_long_kv8_test PROPERTIES LABELS "checkpoint;prefill;kv8"
    SKIP_RETURN_CODE 77 TIMEOUT 5400)
  add_test(NAME prefill_determinism_l0_kv8_test COMMAND prefill_determinism_test
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} l0)
  b70_kv8_twin(prefill_determinism_l0)
  add_test(NAME prefill_consistency_l0_kv8_test COMMAND prefill_consistency_test
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} l0)
  b70_kv8_twin(prefill_consistency_l0)
  add_test(NAME snapshot_kv8_test COMMAND snapshot_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(snapshot)
  add_test(NAME memory_plan_box_kv8_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(memory_plan_box)
  if(TEST mtp_verify_test)
    add_test(NAME mtp_verify_kv8_test COMMAND mtp_verify_test ${B70_TEST_SNAPSHOT})
    b70_kv8_twin(mtp_verify)
    set_property(TEST mtp_verify_kv8_test APPEND PROPERTY ENVIRONMENT
      "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE}")
    add_test(NAME memory_plan_box_mtp_kv8_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT} mtp)
    b70_kv8_twin(memory_plan_box_mtp)
  endif()
  if(TEST mtp_gpu_test)
    add_test(NAME mtp_gpu_kv8_test COMMAND mtp_gpu_test ${B70_TEST_SNAPSHOT} l0-int8)
    b70_kv8_twin(mtp_gpu)
    set_property(TEST mtp_gpu_kv8_test APPEND PROPERTY ENVIRONMENT
      "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_PFI8ALL_KV8}" "B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}")
  endif()
  if(TEST prefix_gpu_int8_test)
    add_test(NAME prefix_gpu_int8_kv8_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
      ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0-int8)
    b70_kv8_twin(prefix_gpu_int8)
  endif()
  if(TEST prefix_gpu_mtp_test)
    add_test(NAME prefix_gpu_mtp_kv8_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
      ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0-int8 1 0 3)
    b70_kv8_twin(prefix_gpu_mtp)
  endif()
endif()

# --- Spec 15d: Ornith 1.5 35B-A3B prefill (label `ornith`) ---------------------------------
# Written blind on the Mac (the validation queue row 13 is the
# order to run these in). The host reference test runs everywhere; the rest need a B70, and
# the checkpoint ones SKIP (77) until the Ornith int4 checkpoint (15a) and its golden set
# `$B70_DATA/oracle-out/ornith/golden` are on the machine, as 15c's do.
#   the reference itself: pf_moe_ref_test (host) - the sort's invariants and tile bound,
#       row independence, the combine == decode's, the whole chunk against decode's chain;
#   the kernels: pf_moe_test (a B70, no checkpoint) - sort / gathers / combine exact,
#       grouped == dense GEMMs bitwise, row independence and replay on the card;
#   R3 (prefill): prefill_gate_test on $B70_DATA/oracle-out/ornith/golden, l0 and l0-int8, one chunk and
#       chunks of 16;
#   continuation: prefill_split_test on Ornith (splits at multiples of 64 bitwise; the other
#       bars are Qwen3.8's, PROVISIONAL here until the test machine measures Ornith's);
#   the walk, determinism + replay, chunking, prefill-vs-decode routes and tokens, R2 on the
#       prefill path: ornith_prefill_test.
add_executable(pf_moe_ref_test kernels/pf_moe_ref_test.cc)
target_include_directories(pf_moe_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME pf_moe_ref_test COMMAND pf_moe_ref_test)
if(B70_ORNITH)
  set(B70_ORNITH_PREFILL_KERNELS
    kernel_pf_res_fold_K2048_SP0_G20 kernel_pf_res_fold_K2048_SP1_G20
    kernel_pf_norm_finish_K2048_G20_W20 kernel_pf_embed_gather_D2048 kernel_pf_ab_proj_D2048
    kernel_pf_dequant_slab_K2048_N12288_L1 kernel_pf_dequant_slab_K4096_N2048_L0
    kernel_pf_dequant_slab_K2048_N9216_L0 kernel_pf_quant_had_K2048 kernel_pf_quant_had_K4096
    kernel_pf_gdn_conv_GK16V32 kernel_pf_gdn_wy_GK16V32 kernel_pf_gdn_scan_GK16V32
    kernel_pf_gated_head_GK16V32 kernel_pf_gdn_intra_GK16V32 kernel_pf_gdn_front_GK16V32 kernel_pf_res_norm_K2048_SP0_G20 kernel_pf_res_norm_K2048_SP1_G20
    kernel_pf_attn_prep_q16_Q16KV2 kernel_pf_attn_Q16KV2
    kernel_pf_flash_attn_Q16KV2 kernel_pf_flash_attn_pair_Q16KV2 kernel_pf_flash_attn_pairg_Q16KV2
    kernel_pf_flash_attn_split_Q16KV2 kernel_pf_flash_merge_Q16KV2 kernel_pf_flash_mergeg_Q16KV2 kernel_pf_moe_router_K2048_N272 kernel_pf_moe_E256_T8_D2048_I512
    kernel_pf_moe_combine_norm_E256_T8_D2048_I512_G20
    kernel_pf_route_slab_K2048_N272_P512 kernel_pf_route_slab_K2048_N128_P256 kernel_moe_M1_E256_T8_D2048_I512_R512
    kernel_pf_gdn_conv_GK16V32_AB256 kernel_pf_gemm_small_A2_B2_S4 kernel_pf_gemm_small_A4_B2_S4
    kernel_pf_moe_sort_mw_E256_T8 kernel_pf_moe_q4gemm_K2048_N1024_SILU kernel_pf_moe_q4gemm_K2048_N1024_SILU_SHX kernel_pf_moe_q4gemm_K512_N2048
    kernel_pf_moe_w4a8_K2048_N1024
    kernel_pf_moe_gemm_K2048_N1024_SILU kernel_pf_moe_gemm_i8_K2048_N1024_SILU
    kernel_pf_moe_gemm_K512_N2048
    kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1 kernel_pf_gemm_T0 kernel_pf_gemm_T0_SILU
    kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU
    ${B70_ORNITH_DECODE_KERNELS})

  add_executable(gdn_front_test prefill/gdn_front_test.cc)
  target_include_directories(gdn_front_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                             ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(gdn_front_test PRIVATE b70_l0)
  b70_target_kernel_dir(gdn_front_test)
  add_dependencies(gdn_front_test kernel_pf_gdn_conv_GK16V32 kernel_pf_gdn_front_GK16V32 kernel_pf_gdn_conv
                   kernel_pf_gdn_front_PL)
  add_test(NAME gdn_front_test COMMAND gdn_front_test)
  set_tests_properties(gdn_front_test PROPERTIES LABELS "ornith;prefill")

  add_dependencies(pf_norm_fused_test kernel_pf_res_fold_K2048_SP0_G20 kernel_pf_res_fold_K2048_SP1_G20
                   kernel_pf_norm_finish_K2048_G20_W20 kernel_pf_res_norm_K2048_SP0_G20 kernel_pf_res_norm_K2048_SP1_G20)
  add_test(NAME pf_norm_fused_test COMMAND pf_norm_fused_test 2048)
  set_tests_properties(pf_norm_fused_test PROPERTIES LABELS "ornith;prefill")
  if(B70_MTP)   # PERF-85 at Ornith's hidden
    add_dependencies(pf_norm_fused_test kernel_pf_res_fold_K2048_SP1_G20_Z kernel_pf_res_norm_K2048_SP1_G20_Z
                     kernel_pf_embed_gather_D2048 kernel_pf_norm_finish_K2048_G20_W20_X4096 kernel_pf_mtp_front_K2048_G20)
    add_test(NAME pf_norm_fused_mtp_test COMMAND pf_norm_fused_test 2048 mtp)
    set_tests_properties(pf_norm_fused_mtp_test PROPERTIES LABELS "ornith;prefill;mtp")
    add_dependencies(pf_attn_prep_kvo_test kernel_pf_attn_prep_q16_Q16KV2 kernel_pf_attn_prep_q16_Q16KV2_KVO)
    add_test(NAME pf_attn_prep_kvo_ornith_test COMMAND pf_attn_prep_kvo_test 16 2)
    set_tests_properties(pf_attn_prep_kvo_ornith_test PROPERTIES LABELS "ornith;prefill;mtp")
  endif()

  add_executable(pf_moe_test kernels/pf_moe_test.cc)
  target_include_directories(pf_moe_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(pf_moe_test PRIVATE b70_prefill_host b70_l0)
  b70_target_kernel_dir(pf_moe_test)
  add_dependencies(pf_moe_test ${B70_ORNITH_PREFILL_KERNELS})
  add_dependencies(attn_chunk_test kernel_pf_flash_attn_Q16KV2 kernel_pf_flash_attn_pair_Q16KV2 kernel_pf_flash_attn_pairg_Q16KV2
    kernel_pf_flash_attn_split_Q16KV2 kernel_pf_flash_merge_Q16KV2 kernel_pf_flash_mergeg_Q16KV2)
  # 3l (2) at Ornith's heads: the split against the pair form and fp64, the gated merge bitwise the merge + pf_attn_gate;
  # the HPW 4 split is a timing arm only (flash_split_test --ornith --time).
  if(TARGET flash_split_test)
    add_ocloc_kernel(pf_flash_attn_split_Q16KV2_H4 SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/prefill/pf_flash_attn.cl
                     DEFINES KT=64 RPW=8 HPW=4 QREG=0 EXP2=1 SPLIT=1 FA_Q_HEADS=16 FA_KV_HEADS=2
                     OPTIONS -cl-intel-256-GRF-per-thread)
    add_dependencies(flash_split_test kernel_pf_flash_attn_pair_Q16KV2 kernel_pf_flash_attn_pairg_Q16KV2
      kernel_pf_flash_attn_split_Q16KV2 kernel_pf_flash_attn_split_Q16KV2_H4 kernel_pf_flash_merge_Q16KV2
      kernel_pf_flash_mergeg_Q16KV2 kernel_pf_attn_Q16KV2)
    add_test(NAME flash_split_ornith_test COMMAND flash_split_test --ornith)
    set_tests_properties(flash_split_ornith_test PROPERTIES LABELS "ornith;prefill" TIMEOUT 1800)
  endif()
  add_test(NAME pf_moe_test COMMAND pf_moe_test)
  set_tests_properties(pf_moe_test PROPERTIES LABELS "ornith;prefill" TIMEOUT 1800)

  add_executable(ornith_prefill_test golden/ornith_prefill_test.cc)
  target_include_directories(ornith_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(ornith_prefill_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  add_dependencies(ornith_prefill_test ${B70_ORNITH_PREFILL_KERNELS})
  add_dependencies(prefill_gate_test ${B70_ORNITH_PREFILL_KERNELS})
  add_dependencies(prefill_split_test ${B70_ORNITH_PREFILL_KERNELS})
  foreach(b l0 l0-int8)
    set(_near 0)
    set(_judge ${B70_ORNITH_F32_DIR})
    # Known divergences (debug 2026-10-26; golden_common.h): exactly
    # the rows that failed that day - golden_common.h AcceptList. cjk 15's engine id at c16 is 99737 (the l0 c16 runs of
    # 2026-11-12 / 13), so the entry names it (item 30 (4)).
    set(_acc_one "")
    set(_acc_c16 "cjk:15=99737")
    if(b STREQUAL "l0-int8")
      # Plan 2026-11-12: l0-int8's experts are W4A8, judged by the PLAIN f32 reference (the pfi8 judge
      # simulates the rotated int8 form it replaced) under the gate's near-miss rule (prefill_gate_test.cc), no
      # per-row accept lists - except cjk row 15 (f32 margin 3.02 ulp; l0 misses it at c16 too), accepted
      # 2026-11-15, for the engine's id 99737 there only (item 30 (4): golden_common.h AcceptList's `=id`).
      set(_near 1)   # the l0-int8 gate's standing near-tie allowance (spec 5), as Qwen3.8's
      set(_acc_one "cjk:15=99737")
      set(_acc_c16 "cjk:15=99737")
    endif()
    add_test(NAME prefill_gate_ornith_${b}_test COMMAND prefill_gate_test
      ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT} --judge=${_judge}
      --accept=${_acc_one} prose,code,cjk 0 ${b} ${_near})
    add_test(NAME prefill_gate_ornith_${b}_c16_test COMMAND prefill_gate_test
      ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT} --judge=${_judge}
      --accept=${_acc_c16} prose,code,cjk 16 ${b} ${_near})
    add_test(NAME prefill_split_ornith_${b}_test COMMAND prefill_split_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR}/long32k.ids ${b})
    # Its routes and token rows go to the PLAIN f32 judge on both backends (l0-int8 too,
    # under the same no-further-from-f32-than-the-bf16-golden rule); the l0-int8 token gate keeps pfi8 above.
    add_test(NAME ornith_prefill_${b}_test COMMAND ornith_prefill_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} ${b} ${B70_ORNITH_ORACLE_DIR} --judge=${B70_ORNITH_F32_DIR})
    set_tests_properties(prefill_gate_ornith_${b}_test prefill_gate_ornith_${b}_c16_test
      PROPERTIES LABELS "checkpoint;golden;prefill;ornith" SKIP_RETURN_CODE 77 TIMEOUT 3600)
    set_tests_properties(prefill_split_ornith_${b}_test ornith_prefill_${b}_test PROPERTIES
      LABELS "checkpoint;prefill;ornith" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 3600)
    # the split runs judged by the f32 reference of long32k[0, 2600), no further from it
    # than the bf16 reference (tools/oracle/qwen_forced.py split; absent: the fixed bars).
    set_tests_properties(prefill_split_ornith_${b}_test PROPERTIES ENVIRONMENT
      "B70_SPLIT_REFS=${B70_ORACLE}/ornith/split/long32k-2600-f32.golden.safetensors,${B70_ORACLE}/ornith/split/long32k-2600-bf16.golden.safetensors")
    # PERF-64: the prompt pass over the int8 KV cache, by the same f32 judge; cjk row 15 at chunk 0 accepted by the
    # owner 2026-11-22 for the engine's 99737 (f32 97663 by 0.299 logit, 3.02 ulp - the row the c16 list accepts).
    if(b STREQUAL "l0")
      add_test(NAME prefill_gate_ornith_l0_kv8_test COMMAND prefill_gate_test
        ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT} --judge=${_judge}
        --accept=cjk:15=99737 prose,code,cjk 0 l0 0)
      set_tests_properties(prefill_gate_ornith_l0_kv8_test PROPERTIES LABELS "checkpoint;golden;prefill;ornith;kv8"
        SKIP_RETURN_CODE 77 TIMEOUT 3600 ENVIRONMENT "B70_KV_CACHE=int8")
    endif()
    # l0-int8 on a MoE model is opt-in and its accuracy gates are diagnostic
    # (the tests print their verdicts and do not fail on them); prefill_gate's quantisation-aware judge (pfi8: a miss
    # there is a kernel error, not an accuracy loss) still gates.
    if(b STREQUAL "l0-int8")
      set_property(TEST prefill_split_ornith_${b}_test ornith_prefill_${b}_test APPEND PROPERTY LABELS diagnostic)
    endif()
  endforeach()
  # Open-work 38: the prompt pass with the mixer linears held in bf16 bitwise the pass that unpacks them, at the
  # chunk caps 2048 and 4096; the scratch grows by the plan's dense_slab_bytes.
  add_executable(prefill_dense_bf16_test prefill/prefill_dense_bf16_test.cc)
  target_include_directories(prefill_dense_bf16_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(prefill_dense_bf16_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  add_dependencies(prefill_dense_bf16_test ${B70_ORNITH_PREFILL_KERNELS})
  add_test(NAME prefill_dense_bf16_ornith_test COMMAND prefill_dense_bf16_test ${B70_ORNITH_SNAPSHOT}
    ${B70_ORNITH_PROMPTS_DIR})
  add_test(NAME prefill_dense_bf16_ornith_cap4096_test COMMAND prefill_dense_bf16_test ${B70_ORNITH_SNAPSHOT}
    ${B70_ORNITH_PROMPTS_DIR} --chunk-cap 4096)
  set_tests_properties(prefill_dense_bf16_ornith_test prefill_dense_bf16_ornith_cap4096_test PROPERTIES
    LABELS "checkpoint;prefill;ornith" SKIP_RETURN_CODE 77 TIMEOUT 3600)

endif()



# ==== Spec 15e: Ornith 1.5 35B-A3B served, and its MoE MTP head (label `ornith`) == (begin) ==
# One delimited block for every 15e registration (the branches in flight beside it edit
# their own blocks). Host tests first - they run on the Mac - then the
# card's, which SKIP (77) until the Ornith int4 checkpoint (15a) is on the machine.
#
# Task 1: the chat template. Ornith's chat_template.jinja is Qwen3.5's, not Qwen3.8's (no
# reasoning-effort block; every assistant turn keeps its <think> block): rendered by the
# engine on six message lists against transformers' render (tools/tokenizer/
# dump_agnes_template.py with the checkpoint's two template files, vendored at
# tests/tokenizer/ornith/ - MIT, ornith-ai/Ornith-1.5-35B-A3B at 10fbf86f) ...
if(B70_TOKENIZER_ENABLED)
  add_test(NAME template_ornith_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/ornith ornith_template_cases.json ornith)
  set_tests_properties(template_ornith_test PROPERTIES LABELS ornith SKIP_RETURN_CODE 77)
  # ... and the server over that template (mock engine and tokenizer): the prompt is
  # transformers' render, Ornith-format tool calls come back as OpenAI tool calls (buffered
  # and streamed), reasoning splits at </think>.
  add_executable(ornith_server_test server/ornith_server_test.cc)
  target_include_directories(ornith_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(ornith_server_test PRIVATE b70_server b70_tokenizer)
  add_test(NAME ornith_server_test COMMAND ornith_server_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
  set_tests_properties(ornith_server_test PROPERTIES LABELS ornith SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
#
# Task 2: the MoE MTP head (spec 15e). Host, on the Mac too: the RTN quantiser and the
# head's MoE layer repacked from a synthetic checkpoint in the published head's naming,
# the head's byte figures and the memory planner with the head (ornith_mtp_head_test) ...
add_executable(ornith_mtp_head_test loader/ornith_mtp_head_test.cc)
target_include_directories(ornith_mtp_head_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(ornith_mtp_head_test PRIVATE b70_loader b70_plan)
add_test(NAME ornith_mtp_head_test COMMAND ornith_mtp_head_test)
set_tests_properties(ornith_mtp_head_test PROPERTIES LABELS ornith)
if(B70_ORNITH AND B70_MTP)
  # ... every binary Ornith's verify / draft lists and the prefill's head KV fill bind,
  # formed as capture.cc / step.cc form them, against the kernels' spec 15e block (which
  # must all be bound) and the binaries it reuses. A name missing from either kernel CMake
  # block fails at configure time here (if(NOT TARGET)), the names test on any host.
  set(B70_ORNITH_MTP_KERNELS)
  foreach(MM 2 3 4)
    list(APPEND B70_ORNITH_MTP_KERNELS
      kernel_gemv_M${MM}_K2048_N12288_S1_L1 kernel_gemv_M${MM}_K4096_N2048_S4_L0
      kernel_gemv_M${MM}_K2048_N9216_S2_L0 kernel_gemv_bf16_M${MM}_K2048_N128_C16_S16
      kernel_gemv_M${MM}_K2048_N128_S1_L1
      kernel_gemv_bf16_M${MM}_K2048_N272_C16_S16 kernel_gemv_bf16_M${MM}_K2048_N248320
      kernel_gemv_i8w_M${MM}_K2048_N248320 kernel_prep_res_fold_M${MM}_K2048_SP0_G20
      kernel_prep_res_fold_M${MM}_K2048_SP4_G20 kernel_prep_norm_finish_M${MM}_K2048_G20_W20
      kernel_embed_gather_M${MM}_D2048 kernel_prep_gated_head_M${MM}_GK16V32
      kernel_attn_prep_M${MM}_Q16KV2 kernel_attn_v2_M${MM}_T${ATTN_V2_TGT}_Q16KV2
      kernel_moe_M${MM}_E256_T8_D2048_I512)
  endforeach()
  foreach(MM 1 2 3 4)
    list(APPEND B70_ORNITH_MTP_KERNELS
      kernel_gdn_step_slots_M${MM}_G30_GK16V32 kernel_gdn_step_defer_M${MM}_G30_GK16V32 kernel_gemv_bf16_M${MM}_K4096_N2048
      kernel_gemv_bf16_M${MM}_K2048_N9216 kernel_prep_norm_finish_M${MM}_K2048_G20_W20_X4096_P2
      kernel_prep_res_fold_M${MM}_K2048_SP0_G20_P2 kernel_prep_res_fold_M${MM}_K2048_SP1_G20_Z kernel_attn_prep_M${MM}_Q16KV2_S1)
  endforeach()
  list(APPEND B70_ORNITH_MTP_KERNELS kernel_prep_res_fold_M1_K2048_SP1_G20)
  # PERF-85: the head's prompt pass binds the one-launch forms; pf_res_fold _Z and pf_norm_finish _X4096 stay built
  # as pf_norm_fused_mtp_test's reference.
  list(APPEND B70_ORNITH_MTP_KERNELS kernel_pf_bf16_slab_K4096 kernel_pf_bf16_slab_K2048
    kernel_pf_mtp_front_K2048_G20 kernel_pf_res_norm_K2048_SP1_G20_Z kernel_pf_attn_prep_q16_Q16KV2_KVO)
  # PERF-56: the wide lookup verify lists' own builds (src/kernels' PERF-56 block; empty without B70_VERIFY_M8).
  get_property(_o56_wide GLOBAL PROPERTY B70_ORNITH_WIDE_KERNELS)
  list(APPEND B70_ORNITH_MTP_KERNELS ${_o56_wide})
  # What the lists reuse: 15c's decode list, 15d's prefill kernels, spec 8's shared ones.
  set(_o15e_reused ${B70_ORNITH_DECODE_KERNELS}
    kernel_argmax_stage1_M2 kernel_argmax_stage1_M3 kernel_argmax_stage1_M4
    kernel_pf_res_fold_K2048_SP0_G20 kernel_pf_res_fold_K2048_SP1_G20
    kernel_pf_norm_finish_K2048_G20_W20 kernel_pf_embed_gather_D2048
    kernel_pf_attn_prep_q16_Q16KV2 kernel_pf_res_norm_K2048_SP0_G20)
  if(_o56_wide)   # PERF-56: Qwen3.8's argmax pair at the wide lists' rows (the same vocabulary)
    list(APPEND _o15e_reused kernel_argmax_stage1_M16 kernel_argmax_stage1_M32 kernel_argmax_stage1_M64
      kernel_argmax_stage2_W16 kernel_argmax_stage2_W32 kernel_argmax_stage2_W64)
  endif()
  get_property(_o28b_gemm_sm GLOBAL PROPERTY B70_GEMM_SM_KERNELS)   # spec 28b: the verify lists' dense gemm_sm
  list(APPEND _o15e_reused ${_o28b_gemm_sm})
  foreach(t ${B70_ORNITH_MTP_KERNELS} ${_o15e_reused})
    if(NOT TARGET ${t})
      message(FATAL_ERROR "spec 15e: ${t} is not built (src/kernels/CMakeLists.txt)")
    endif()
  endforeach()
  add_executable(ornith_mtp_names_test kernels/ornith_mtp_names_test.cc)
  target_include_directories(ornith_mtp_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(ornith_mtp_names_test PRIVATE b70_model)
  b70_target_kernel_dir(ornith_mtp_names_test)
  add_test(NAME ornith_mtp_names_test COMMAND ornith_mtp_names_test ${B70_ORNITH_MTP_KERNELS} --
    ${_o15e_reused})
  set_tests_properties(ornith_mtp_names_test PROPERTIES LABELS ornith)

  # On the card, no checkpoint: moe.cl at M = 2..4, every row bitwise M = 1's (the verify
  # lists' MoE block, Review Focus 3).
  add_executable(moe_m_test kernels/moe_m_test.cc)
  target_include_directories(moe_m_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(moe_m_test PRIVATE b70_l0)
  b70_target_kernel_dir(moe_m_test)
  add_dependencies(moe_m_test kernel_moe_M1_E256_T8_D2048_I512 kernel_moe_M2_E256_T8_D2048_I512
    kernel_moe_M3_E256_T8_D2048_I512 kernel_moe_M4_E256_T8_D2048_I512)
  if(B70_VERIFY_M8)   # PERF-56: the wide lookup verify lists' rows
    add_dependencies(moe_m_test kernel_moe_M16_E256_T8_D2048_I512 kernel_moe_M32_E256_T8_D2048_I512
      kernel_moe_M64_E256_T8_D2048_I512 kernel_moe_M16_E256_T8_D2048_I512_U kernel_moe_M32_E256_T8_D2048_I512_U
      kernel_moe_M4_E256_T8_D2048_I512_U)
  endif()
  add_test(NAME moe_m_test COMMAND moe_m_test)
  set_tests_properties(moe_m_test PROPERTIES LABELS "ornith;mtp" TIMEOUT 900)

  # On the card with the checkpoint (SKIP 77 without it): the lists, M2 bitwise rows
  # (logits, GDN slots, every layer's routes), M3 greedy --spec mtp == --spec off against a
  # head-less engine, the head's route row; both lm_head forms. And M1 - the head's draft
  # logits against tools/oracle/mtp_ref.py's dump on Ornith ($B70_DATA/oracle-out/ornith/mtp-head, spec
  # 15a's; SKIP without it) - through spec 8's own mtp_head_test.
  if(TARGET mtp_head_test)
    get_property(_o15e_k DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
    get_property(_o15e_pf DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
    add_executable(ornith_mtp_test runtime/ornith_mtp_test.cc)
    b70_target_kernel_dir(ornith_mtp_test)
    target_include_directories(ornith_mtp_test PRIVATE ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
    target_link_libraries(ornith_mtp_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
    add_dependencies(ornith_mtp_test ${_o15e_k} ${_o15e_pf})
    add_test(NAME ornith_mtp_test COMMAND ornith_mtp_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} int8)
    add_test(NAME ornith_mtp_bf16head_test COMMAND ornith_mtp_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} bf16)
    add_test(NAME mtp_head_ornith_test COMMAND mtp_head_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} ${B70_ORACLE}/ornith/mtp-head/m1
      ${B70_ORACLE}/ornith/mtp-head)
    set_tests_properties(ornith_mtp_test ornith_mtp_bf16head_test mtp_head_ornith_test PROPERTIES
      LABELS "checkpoint;mtp;ornith" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 3600)
    # M1 judged by the f32 reference of the same rows (tools/oracle/qwen_forced.py
    # ornith-mtp), the engine missing no more rows than the bf16 reference.
    set_tests_properties(mtp_head_ornith_test PROPERTIES
      ENVIRONMENT "B70_MTP_JUDGE=${B70_ORACLE}/ornith/mtp-head-f32/m1")
    # r16.mtp's f32 judges (spec 28 N6): the forced contexts of each head form (B70_DUMP_FORCED ->
    # oracle-forced-dumps/ornith-mtp, f32_refs.sh forced ornith ... [-i8head --lm-head int8]).
    set_tests_properties(ornith_mtp_test PROPERTIES
      ENVIRONMENT "B70_FORCED_JUDGE=${B70_ORACLE}/ornith/f32-i8head-forced")
    set_tests_properties(ornith_mtp_bf16head_test PROPERTIES
      ENVIRONMENT "B70_FORCED_JUDGE=${B70_ORACLE}/ornith/f32-forced")
    # PERF-64: the MTP verify and draft lists over the int8 KV cache, by the int8 head's forced judges.
    add_test(NAME ornith_mtp_kv8_test COMMAND ornith_mtp_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} int8)
    set_tests_properties(ornith_mtp_kv8_test PROPERTIES LABELS "checkpoint;mtp;ornith;kv8" SKIP_RETURN_CODE 77
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600
      ENVIRONMENT "B70_FORCED_JUDGE=${B70_ORACLE}/ornith/f32-i8head-forced;B70_KV_CACHE=int8")
    # Plan 31 Task 12: Ornith's sequence slots - pool_test's cases on Ornith (pool_test.cc), with S0 against the
    # one-request engine; S4 judged by the int8 head's f32 references of the forced contexts.
    if(TARGET pool_test)
      add_dependencies(pool_test ${_o15e_k} ${_o15e_pf})
      add_test(NAME ornith_pool_test COMMAND pool_test ${B70_ORNITH_SNAPSHOT} --one-slot)
      set_tests_properties(ornith_pool_test PROPERTIES LABELS "checkpoint;mtp;pool;ornith" SKIP_RETURN_CODE 77
        ENVIRONMENT "B70_FORCED_JUDGE=${B70_ORACLE}/ornith/f32-i8head-forced"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
      # Plan 31 Task 14: the KV page pool's cases on Ornith (pool_test.cc's --pool: P1-P5).
      add_test(NAME ornith_pool_pages_test COMMAND pool_test ${B70_ORNITH_SNAPSHOT} --pool)
      set_tests_properties(ornith_pool_pages_test PROPERTIES LABELS "checkpoint;mtp;pool;ornith" SKIP_RETURN_CODE 77
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
      # PERF-64: both over the int8 KV cache (the restore and reset cases included: S2, P2).
      add_test(NAME ornith_pool_kv8_test COMMAND pool_test ${B70_ORNITH_SNAPSHOT} --one-slot)
      add_test(NAME ornith_pool_pages_kv8_test COMMAND pool_test ${B70_ORNITH_SNAPSHOT} --pool)
      set_tests_properties(ornith_pool_kv8_test ornith_pool_pages_kv8_test PROPERTIES
        LABELS "checkpoint;mtp;pool;ornith;kv8" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600
        ENVIRONMENT "B70_FORCED_JUDGE=${B70_ORACLE}/ornith/f32-i8head-forced;B70_KV_CACHE=int8")
    endif()
  endif()
  # PERF-56: mtp_verify_test's W16 / W32 / W64 cases on Ornith one card - its int8 head, its int8-head forced judges
  # (W16's rows against the plain steps; W32 / W64 bitwise the 16-row list's rows, no judge).
  if(TARGET mtp_verify_test AND B70_VERIFY_M8)
    add_test(NAME mtp_verify_wide_ornith_test COMMAND mtp_verify_test ${B70_ORNITH_SNAPSHOT} 16384 rule wide)
    set_tests_properties(mtp_verify_wide_ornith_test PROPERTIES LABELS "checkpoint;mtp;ornith" SKIP_RETURN_CODE 77
      ENVIRONMENT "B70_FORCED_JUDGE=${B70_ORACLE}/ornith/f32-i8head-forced"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 1800)
    add_test(NAME mtp_verify_deep_ornith_test COMMAND mtp_verify_test ${B70_ORNITH_SNAPSHOT} 16384 rule deep)
    set_tests_properties(mtp_verify_deep_ornith_test PROPERTIES LABELS "checkpoint;mtp;ornith" SKIP_RETURN_CODE 77
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 1800)
  endif()
endif()
# ==== Spec 15e ============================================================= (end) ====

# The renderer extensions - third_party/minja's b70 patches (third_party/VERSIONS) - each
# against Jinja2's own render (tools/tokenizer/dump_minja_ext.py).
add_executable(minja_ext_test tokenizer/minja_ext_test.cc)
target_include_directories(minja_ext_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(minja_ext_test PRIVATE b70_third_party_minja)
add_test(NAME minja_ext_test COMMAND minja_ext_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
# b70-serve's refusals before the device (b70-decode's b70_cli_reject, for b70-serve).
if(TARGET b70-serve)
  function(b70_serve_reject name)
    cmake_parse_arguments(R "" "EXPECT" "" ${ARGN})
    add_test(NAME ${name} COMMAND sh -c [==[
expect=$1
shift
out=$("$@" 2>&1)
rc=$?
printf '%s\n' "$out" >&2
if [ "$rc" -ne 1 ]
then
  echo "b70_serve_reject: expected exit 1, got $rc" >&2
  exit 1
fi
if printf '%s\n' "$out" | grep -q '^device:'
then
  echo "b70_serve_reject: the rejection happened after l0::Context was built" >&2
  exit 1
fi
if ! printf '%s\n' "$out" | grep -qF -- "$expect"
then
  echo "b70_serve_reject: exit 1, but the message does not contain '$expect'" >&2
  exit 1
fi
]==] b70_serve_reject "${R_EXPECT}" $<TARGET_FILE:b70-serve> ${R_UNPARSED_ARGUMENTS})
    set_tests_properties(${name} PROPERTIES TIMEOUT 60)
  endfunction()
  # The flag parser's own refusals, and every flag accepted (pinned before the flags became one struct, src/cli/flags.h).
  b70_serve_reject(cli_reject_serve_flag_needs_value ${B70_NO_MODEL} --port EXPECT "--port needs a value")
  b70_serve_reject(cli_reject_serve_flag_u32 ${B70_NO_MODEL} --port x EXPECT "--port expects a non-negative integer, got 'x'")
  b70_serve_reject(cli_reject_serve_flag_port ${B70_NO_MODEL} --port 65536 EXPECT "--port is out of range")
  b70_serve_reject(cli_reject_serve_flag_seqs_zero ${B70_NO_MODEL} --max-num-seqs 0
    EXPECT "--max-num-seqs 0 would serve no request")
  b70_serve_reject(cli_reject_serve_flag_layers_zero ${B70_NO_MODEL} --layers 0 EXPECT "--layers 0 would load no layer")
  b70_serve_reject(cli_reject_serve_flag_cache_gb ${B70_NO_MODEL} --prefix-cache-gb x
    EXPECT "--prefix-cache-gb expects a non-negative integer, got 'x'")
  b70_serve_reject(cli_reject_serve_flag_unknown ${B70_NO_MODEL} --bogus EXPECT "unknown option '--bogus'")
  b70_serve_reject(cli_reject_serve_flag_extra ${B70_NO_MODEL} extra EXPECT "unexpected extra argument 'extra'")
  b70_serve_reject(cli_reject_serve_flag_no_model --port 1 EXPECT "a snapshot directory or HF repo id is required")
  b70_serve_reject(cli_accept_serve_flags ${B70_NO_MODEL} --host 127.0.0.1 --port 8001 --max-len 8192 --max-num-seqs 1
    --log-requests /nonexistent/log --prefix-cache-gb 0 --prefix-split-last --spec mtp --spec-max 2 --prefill-backend l0
    --kv-cache bf16 --parallel 1 EXPECT "cannot create directories: Permission denied [/nonexistent/log]")
  b70_serve_reject(cli_accept_serve_flags_draft ${B70_NO_MODEL} --spec dflash --draft-model /nonexistent/drafter
    --prefix-cache-gb auto --parallel auto --max-len auto
    EXPECT "not a directory and not an <org>/<name> repo id: /nonexistent/drafter")
  add_test(NAME cli_serve_help COMMAND sh -c [==[
bin=$1
shift
out=$("$bin" --help 2>&1) || { echo "cli_serve_help: --help did not exit 0" >&2; exit 1; }
for f in "$@"; do printf '%s\n' "$out" | grep -qF -- "$f" || { echo "cli_serve_help: --help does not name $f" >&2; exit 1; }; done
]==] cli_serve_help $<TARGET_FILE:b70-serve> --spec --spec-max --draft-model --max-num-seqs --max-len --parallel
    --pipeline-split --host --port --log-requests --prefix-cache-gb --prefix-split-last --layers --kv-cache
    --prefill-backend)
  # The flag tables (README.md: b70-serve, docs/18-development.md: b70-decode) against both binaries' generated --help:
  # the same flags, the same defaults.
  add_test(NAME cli_readme_flags COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/cli/readme_flags_test.py
    ${CMAKE_SOURCE_DIR}/docs/17-running-models.md ${CMAKE_SOURCE_DIR}/docs/18-development.md $<TARGET_FILE:b70-serve> $<TARGET_FILE:b70-decode>)
endif()


# ==== Spec 16b: pipeline parallel decode across two B70s ================================
# Written blind on the Mac (no B70, and no multi-device Level Zero on a Mac at all);
# the validation queue row 22 is the order to run these in.
# `pp` labels the device ones: `ctest -L pp`.
#
# Host, no device: the planner (stages, weights by device, the per-device plan, auto split
# and auto length for Qwen3.8 / Ornith), and b70-decode's
# --parallel flag.
add_executable(pipeline_plan_test runtime/pipeline_plan_test.cc)
target_include_directories(pipeline_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pipeline_plan_test PRIVATE b70_plan)
add_test(NAME pipeline_plan_test COMMAND pipeline_plan_test)
find_package(Threads REQUIRED)
# Spec 17a (plan 17a Task 1): the TP all-reduce's crossing protocol (tools/probe/tp_protocol.h, the host mirror of
# probe_tp.cl's tp_send / tp_wait and probe_tp's CPU-sum thread) on host threads - every route, stamps, the two-slot
# landing, the fixed-order sum bitwise on both "cards", a skipped data write caught, a dropped write timed out.
add_executable(tp_crossing_protocol_test runtime/tp_crossing_protocol_test.cc)
target_include_directories(tp_crossing_protocol_test PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tools/probe)
target_link_libraries(tp_crossing_protocol_test PRIVATE Threads::Threads)
add_test(NAME tp_crossing_protocol_test COMMAND tp_crossing_protocol_test)
add_executable(pipeline_args_test cli/pipeline_args_test.cc)
target_include_directories(pipeline_args_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pipeline_args_test PRIVATE b70_plan)
add_test(NAME pipeline_args_test COMMAND pipeline_args_test)

# On the test machine, both cards and the checkpoint; SKIP (77) with fewer than two GPUs or without
# peer access. P1 (pp_decode_test): ingest + 64 greedy ids and a spec 7 restore at 4395,
# --parallel pp, the auto split and the cuts at 5 and 62 - ids, logits,
# every layer's GDN state / conv ring / KV and both Control blocks BITWISE the single card on
# device 0; a reset and the whole run again, bitwise. P4 (pp_fail_test): a lost hand-off
# throws within its bound, never hangs, and reset() recovers.
add_executable(pp_decode_test runtime/pp_decode_test.cc)
target_include_directories(pp_decode_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(pp_decode_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(pp_decode_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
add_test(NAME pp_decode_test COMMAND pp_decode_test ${B70_TEST_SNAPSHOT})
add_test(NAME pp_decode_i8head_test COMMAND pp_decode_test ${B70_TEST_SNAPSHOT} int8)
add_executable(pp_fail_test runtime/pp_fail_test.cc)
target_include_directories(pp_fail_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(pp_fail_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(pp_fail_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME pp_fail_test COMMAND pp_fail_test ${B70_TEST_SNAPSHOT})
set_tests_properties(pp_decode_test pp_decode_i8head_test pp_fail_test PROPERTIES
  LABELS "checkpoint;pp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600)
if(B70_KV8 AND COMMAND b70_kv8_twin)
  add_dependencies(pp_decode_test ${B70_KV8_KERNELS})
  add_test(NAME pp_decode_kv8_test COMMAND pp_decode_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(pp_decode)
endif()

# b70-decode's --parallel refusals, before any device (cli/pipeline_args.h), and P4's missing
# card: with ZE_AFFINITY_MASK=0 the driver shows one GPU and --parallel pp must stop with the
# count, before the load (the model path does not exist).
b70_cli_reject(cli_reject_pipeline_value ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --parallel 3
  EXPECT "--parallel expects auto, 1, pp, replica or tp")
# (Spec 16c lifted 16b's two prefill refusals; what it refuses is in its own block below.
# Spec 16d lifted the --spec mtp one, cli_reject_pipeline_mtp: --parallel pp --spec mtp runs.)
# Plan 13b: two requests at once is b70-serve's.
b70_cli_reject(cli_reject_pipeline_replica_decode ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --parallel replica
  EXPECT "--parallel replica is b70-serve's")
b70_cli_reject(cli_reject_pipeline_split_alone ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --parallel 1
  --pipeline-split 32 EXPECT "--pipeline-split belongs to --parallel pp")
# --pipeline-handoff went 2026-11-13 (the peer hand-off measured no gain): an unknown option now.
b70_cli_reject(cli_reject_pipeline_handoff_removed ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --parallel pp
  --pipeline-handoff copy EXPECT "unknown option '--pipeline-handoff'")
add_test(NAME cli_reject_pipeline_one_gpu COMMAND sh -c [==[
out=$(ZE_AFFINITY_MASK=0 "$@" 2>&1)
rc=$?
printf '%s\n' "$out" >&2
[ "$rc" -eq 1 ] || { echo "expected exit 1, got $rc" >&2; exit 1; }
printf '%s\n' "$out" | grep -qF 'needs two GPUs and Level Zero shows 1' ||
  { echo "the refusal does not name the device count" >&2; exit 1; }
if printf '%s\n' "$out" | grep -q '^devices:'; then echo "a device was opened" >&2; exit 1; fi
exit 0
]==] cli_reject_pipeline_one_gpu $<TARGET_FILE:b70-decode> ${B70_NO_MODEL} --ids ${B70_OK_IDS}
  --n 4 --parallel pp)
set_tests_properties(cli_reject_pipeline_one_gpu PROPERTIES LABELS pp TIMEOUT 60)
# ==== Spec 16b ============================================================= (end) ====

# ==== Spec 17b: tensor parallel decode across two B70s (`--parallel tp`) ================================
# Validation queue row 79 (the validation queue). Host, no device: the per-card descriptor, plan and kernel names
# (tp_desc_test); the per-card slicing rules, card tiles = one-card tiles, the one-card gather of state and KV
# (tp_slice_test); the crossing's protocol on two host threads, both forms, a stale and a dropped send
# (tp_protocol_test); the bf16 wire's packing and fixed sum order as a host twin on random data (tp_wire_bf16_test).
add_executable(tp_desc_test model/tp_desc_test.cc)
target_include_directories(tp_desc_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(tp_desc_test PRIVATE b70_plan b70_model)
b70_target_kernel_dir(tp_desc_test)
add_test(NAME tp_desc_test COMMAND tp_desc_test)
add_executable(tp_slice_test loader/tp_slice_test.cc)
target_include_directories(tp_slice_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(tp_slice_test PRIVATE b70_loader b70_plan b70_model)
add_test(NAME tp_slice_test COMMAND tp_slice_test)
add_executable(tp_protocol_test runtime/tp_protocol_test.cc)
target_include_directories(tp_protocol_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(tp_protocol_test PRIVATE b70_model Threads::Threads)
add_test(NAME tp_protocol_test COMMAND tp_protocol_test)
set_tests_properties(tp_protocol_test PROPERTIES TIMEOUT 300)
add_executable(tp_wire_bf16_test runtime/tp_wire_bf16_test.cc)
target_include_directories(tp_wire_bf16_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(tp_wire_bf16_test PRIVATE b70_model Threads::Threads)
add_test(NAME tp_wire_bf16_test COMMAND tp_wire_bf16_test)
set_tests_properties(tp_wire_bf16_test PROPERTIES TIMEOUT 300)

# Both cards; SKIP (77) with fewer than two GPUs or without peer access both ways.
# tp_kernels_test (no checkpoint): the crossing against prep_res_fold, bitwise (slices) / the host's sum (folded),
# 300 replays. tp_decode_test (T1 / T2): the prose prompt + 64 greedy ids against one card on card 0 - both crossings
# judged by the golden tie rule (slices since PERF-108's in-work-group K split of a card's qkv||z), both T1 bitwise
# between the cards and replayed after reset bitwise; its `bench` mode times one crossing (row 79's speed stage). tp_fail_test (T4): a card's list
# dropped, the step throws within its bounds, reset() recovers.
add_executable(tp_kernels_test kernels/tp_kernels_test.cc)
target_include_directories(tp_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(tp_kernels_test PRIVATE b70_l0 b70_model)
b70_target_kernel_dir(tp_kernels_test)
add_dependencies(tp_kernels_test ${B70_TP_KERNELS} kernel_prep_res_fold_M1_K5120_SP4_G20)
if(B70_TP_STRESS_KERNELS)   # the `stress` mode's both trailer layouts at M = 1..8 (B70_VERIFY_M8)
  add_dependencies(tp_kernels_test ${B70_TP_STRESS_KERNELS})
endif()
add_test(NAME tp_kernels_test COMMAND tp_kernels_test)
# PERF-44: the folded crossing's tp_send as the linear's tail against the plain linear + tp_send, both sides on ONE card
# (no checkpoint; ctest -R tp_fold_send_test): bitwise at M 1 (gemv) and the verify rows (gemm_sm) built.
add_executable(tp_fold_send_test kernels/tp_fold_send_test.cc)
target_include_directories(tp_fold_send_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(tp_fold_send_test PRIVATE b70_l0 b70_model)
b70_target_kernel_dir(tp_fold_send_test)
add_dependencies(tp_fold_send_test ${B70_TP_KERNELS})
set(_tfs_m 1)
if(B70_MTP)
  set(_tfs_m 1 2 3 4)
  if(B70_VERIFY_M8)
    set(_tfs_m 1 2 3 4 5 6 7 8)
  endif()
  add_dependencies(tp_fold_send_test ${B70_TP_VERIFY_KERNELS})
  foreach(MM ${_tfs_m})
    if(NOT MM EQUAL 1)
      add_dependencies(tp_fold_send_test kernel_gemm_sm_M${MM}_K3072_N5120_S4_L0_mg_b2d
                       kernel_gemm_sm_M${MM}_K8704_N5120_S4_L0_mg_b2d kernel_tp_cross_M${MM}_K5120_G20_S4_F_R0
                       kernel_tp_cross_M${MM}_K5120_G20_S4_F_R1)
    endif()
  endforeach()
endif()
add_test(NAME tp_fold_send_test COMMAND tp_fold_send_test ${_tfs_m})
set_tests_properties(tp_fold_send_test PROPERTIES LABELS tp SKIP_RETURN_CODE 77 TIMEOUT 600)
add_executable(tp_decode_test runtime/tp_decode_test.cc)
target_include_directories(tp_decode_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/tests/server)
target_link_libraries(tp_decode_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(tp_decode_test ${B70_DECODE_LIST_KERNELS} ${B70_TP_KERNELS})
add_test(NAME tp_decode_test COMMAND tp_decode_test ${B70_TEST_SNAPSHOT} both)
add_executable(tp_fail_test runtime/tp_fail_test.cc)
target_include_directories(tp_fail_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(tp_fail_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(tp_fail_test ${B70_TP_KERNELS})
add_test(NAME tp_fail_test COMMAND tp_fail_test ${B70_TEST_SNAPSHOT})
set_tests_properties(tp_kernels_test PROPERTIES LABELS tp SKIP_RETURN_CODE 77 TIMEOUT 600)
set_tests_properties(tp_decode_test tp_fail_test PROPERTIES
  LABELS "checkpoint;tp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600)
if(B70_KV8 AND COMMAND b70_kv8_twin)
  add_test(NAME tp_decode_kv8_test COMMAND tp_decode_test ${B70_TEST_SNAPSHOT} both)
  b70_kv8_twin(tp_decode)
endif()

# Spec 17c: tensor-parallel prefill on both cards (both cards visible) - tp_prefill_test against one card's Engine::prefill
# on card 0: T1 (Controls, the following decode steps' residual taps) bitwise between the cards, T2 judged by the
# golden tie rule (ids), chunk invariance (1024 vs 2048) and replay bitwise; bf16 and (the _kv8 twin) int8 KV.
add_executable(tp_prefill_test runtime/tp_prefill_test.cc)
target_include_directories(tp_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/tests/server)
target_link_libraries(tp_prefill_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(tp_prefill_test ${B70_PREFILL_KERNELS} ${B70_TP_KERNELS} ${B70_TP_PREFILL_KERNELS})
# A divergence that is not a near-tie goes to the int8 head's f32 judge of its context (forced_judge.h; the judges:
# tools/oracle/qwen_forced.py forced-qwen with QWEN_KEYS_RE='^tp_prefill_'), and awaits it as SKIP (77).
set(_tp_pf_judge "${B70_ORACLE}/qwen38/f32-i8head-forced")
add_test(NAME tp_prefill_test COMMAND tp_prefill_test ${B70_TEST_SNAPSHOT})
set_tests_properties(tp_prefill_test PROPERTIES
  LABELS "checkpoint;tp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600
  ENVIRONMENT "B70_FORCED_JUDGE=${_tp_pf_judge}")
add_executable(tp_prefill_int8_test runtime/tp_prefill_int8_test.cc)
target_include_directories(tp_prefill_int8_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(tp_prefill_int8_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(tp_prefill_int8_test ${B70_PREFILL_KERNELS} ${B70_TP_KERNELS} ${B70_TP_PREFILL_KERNELS})
add_test(NAME tp_prefill_int8_test COMMAND tp_prefill_int8_test ${B70_TEST_SNAPSHOT})
set_tests_properties(tp_prefill_int8_test PROPERTIES
  LABELS "checkpoint;tp;prefill" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  SKIP_RETURN_CODE 77 TIMEOUT 3600 ENVIRONMENT "B70_FORCED_JUDGE=${_tp_pf_judge}")
if(B70_KV8)
  # PERF-107: the int8 pass on the int8 cache (its served form), a follow-up turn's tail as split chunks; the same
  # contexts and f32 judges as tp_prefill_int8_test.
  add_test(NAME tp_prefill_int8_kv8_test COMMAND tp_prefill_int8_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(tp_prefill_int8_kv8_test PROPERTIES
    LABELS "checkpoint;tp;prefill;kv8" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600
    ENVIRONMENT "B70_KV_CACHE=int8;B70_FORCED_JUDGE=${_tp_pf_judge}")
  add_test(NAME tp_prefill_kv8_test COMMAND tp_prefill_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(tp_prefill_kv8_test PROPERTIES
    LABELS "checkpoint;tp;kv8" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600
    ENVIRONMENT "B70_KV_CACHE=int8;B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD_KV8}")
endif()

# Spec 17d: MTP, the snapshots' restore and the server's calls under tensor parallel (both cards visible) - tp_mtp_test:
# speculative greedy ids at K = 1..3 against plain TP decode (the golden tie rule, then the f32 judge of a non-tie's
# context: tools/oracle/qwen_forced.py forced-qwen with QWEN_KEYS_RE='^tp_mtp_'), T1 of the head between the cards,
# the logits gathers (D9), the block hook and a restore bitwise the cold run, one card's snapshot into TP and back
# (judged), a dropped verify step bounded; bf16 and (the _kv8 twin) int8 KV. `cost <depth>` measures the TP cost table.
add_executable(tp_mtp_test runtime/tp_mtp_test.cc)
target_include_directories(tp_mtp_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/tests/server)
target_link_libraries(tp_mtp_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(tp_mtp_test ${B70_PREFILL_KERNELS} ${B70_TP_KERNELS} ${B70_TP_PREFILL_KERNELS} ${B70_TP_VERIFY_KERNELS})
add_test(NAME tp_mtp_test COMMAND tp_mtp_test ${B70_TEST_SNAPSHOT})
set_tests_properties(tp_mtp_test PROPERTIES
  LABELS "checkpoint;tp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600
  ENVIRONMENT "B70_FORCED_JUDGE=${_tp_pf_judge};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}")
if(B70_KV8)
  add_test(NAME tp_mtp_kv8_test COMMAND tp_mtp_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(tp_mtp_kv8_test PROPERTIES
    LABELS "checkpoint;tp;kv8" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600
    ENVIRONMENT "B70_KV_CACHE=int8;B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD_KV8};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}")
endif()

# Spec 19 v3c: DFlash under tensor parallel (both cards visible) - tp_dflash_test: speculative greedy ids at K = 7, 3, 1
# with the drafter split over both cards against plain TP decode (the golden tie rule, then the f32 judge:
# QWEN_KEYS_RE='^tp_dflash_'), T1 of the drafter halves between the cards, a restore bitwise, a dropped draft list
# bounded. `<ckpt> <drafter> cost <depth>` measures plan 19v3c's verify table and round. Built where the drafter's TP
# kernels are (B70_TP, B70_DFLASH, B70_VERIFY_M8, B70_MTP).
if(B70_TP AND B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(tp_dflash_test runtime/tp_dflash_test.cc)
  target_include_directories(tp_dflash_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/tests/server)
  target_link_libraries(tp_dflash_test PRIVATE b70_dflash_runtime b70_prefill_host b70_runtime b70_loader b70_l0
    b70_model)
  add_test(NAME tp_dflash_test COMMAND tp_dflash_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(tp_dflash_test PROPERTIES
    LABELS "checkpoint;tp;dflash" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600
    ENVIRONMENT "B70_FORCED_JUDGE=${_tp_pf_judge};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}")
endif()

# b70-decode --parallel tp's refusals, before any device (cli/pipeline_args.h check_tp), and the missing card.
b70_cli_reject(cli_reject_tp_spec ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --parallel tp --spec dflash
  --draft-model ${B70_NO_MODEL} EXPECT "b70-decode --parallel tp drafts with the MTP head only")
b70_cli_reject(cli_reject_tp_split ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --parallel tp --pipeline-split 32
  EXPECT "--pipeline-split belongs to --parallel pp")
b70_cli_reject(cli_reject_tp_profile ${B70_NO_MODEL} --profile --parallel tp
  EXPECT "it has no --parallel tp")
add_test(NAME cli_reject_tp_one_gpu COMMAND sh -c [==[
out=$(ZE_AFFINITY_MASK=0 "$@" 2>&1)
rc=$?
printf '%s\n' "$out" >&2
[ "$rc" -eq 1 ] || { echo "expected exit 1, got $rc" >&2; exit 1; }
printf '%s\n' "$out" | grep -qF -- '--parallel tp needs two GPUs and Level Zero shows 1' ||
  { echo "the refusal does not name the device count" >&2; exit 1; }
exit 0
]==] cli_reject_tp_one_gpu $<TARGET_FILE:b70-decode> ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --parallel tp)
set_tests_properties(cli_reject_tp_one_gpu PROPERTIES LABELS tp TIMEOUT 60)
# ==== Spec 17b ============================================================= (end) ====

# ==== Spec 16c: pipeline parallel PREFILL across two B70s (the chunk pipeline) ==========
# Written blind on the Mac, like 16b; the validation queue, row 23 is the order to run the
# device ones in (`ctest -L pp`).
#
# Host, no device: the chunks (Engine::prefill's, exactly), the host's order and its rules
# (every mutation that drops one caught), the landing slots and the per-device plan with a
# prefill path (runtime/pipeline_prefill_plan.h); and that order run by its one executor over
# two host threads standing in for the devices - a slowed device 1
# (back-pressure), spec 7's hooks from the shadows, a throwing hook, a lost hand-off. Clean
# under ThreadSanitizer (measured on the retired Mac tooling, 2026-10).
add_executable(pipeline_prefill_plan_test runtime/pipeline_prefill_plan_test.cc)
target_include_directories(pipeline_prefill_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pipeline_prefill_plan_test PRIVATE b70_plan)
add_test(NAME pipeline_prefill_plan_test COMMAND pipeline_prefill_plan_test)
add_executable(pp_prefill_protocol_test runtime/pp_prefill_protocol_test.cc)
target_include_directories(pp_prefill_protocol_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pp_prefill_protocol_test PRIVATE b70_plan Threads::Threads)
add_test(NAME pp_prefill_protocol_test COMMAND pp_prefill_protocol_test)
set_tests_properties(pp_prefill_protocol_test PROPERTIES TIMEOUT 120)

# On the test machine, both cards and the checkpoint; SKIP (77) with fewer than two GPUs or without peer
# access. P1 / P2 (pp_prefill_test): 4103 ids (3 chunks), 32768 (16), 2, a prefill_split
# continuation at 2500 with chunk 1000, spec 7's hooks at 2048 / 4096 (from the shadows) and
# 5000, a hook that throws, a one-card snapshot continued on two cards - logits, every layer's
# GDN state / conv ring / KV, the Control blocks and 16 decoded ids BITWISE one card on device
# 0, at the auto split and the cuts at 5 and 62; l0-int8 (default), l0, the
# int8 head, the int8 KV cache. Back-pressure and P4 (pp_prefill_fail_test): device 1 held back
# artificially - bitwise the clean run; a lost hand-off throws within its bound and reset()
# recovers.
add_executable(pp_prefill_test runtime/pp_prefill_test.cc)
target_include_directories(pp_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(pp_prefill_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(pp_prefill_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
add_test(NAME pp_prefill_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0-int8)
add_test(NAME pp_prefill_l0_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0)
add_test(NAME pp_prefill_i8head_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0-int8 int8)
add_test(NAME pp_prefill_cap4096_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0-int8 --chunk-cap 4096)   # plan 5d
add_executable(pp_prefill_fail_test runtime/pp_prefill_fail_test.cc)
target_include_directories(pp_prefill_fail_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(pp_prefill_fail_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(pp_prefill_fail_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
add_test(NAME pp_prefill_fail_test COMMAND pp_prefill_fail_test ${B70_TEST_SNAPSHOT})
set_tests_properties(pp_prefill_test pp_prefill_l0_test pp_prefill_i8head_test pp_prefill_cap4096_test pp_prefill_fail_test
  PROPERTIES LABELS "checkpoint;pp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77
  TIMEOUT 5400)
if(B70_KV8 AND COMMAND b70_kv8_twin)
  add_dependencies(pp_prefill_test ${B70_KV8_KERNELS})
  add_test(NAME pp_prefill_kv8_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0-int8)
  b70_kv8_twin(pp_prefill)
endif()

# b70-decode --parallel pp runs --prefill and --bench --prefill-length through the pipeline.
# ==== Spec 16c ============================================================= (end) ====

# ==== Spec 16d: pipeline parallel integration (b70-serve --parallel pp, MTP across the split) ====
# Written blind on the Mac, like 16b / 16c; the validation queue, row 27 is the order to run
# the device ones in (`ctest -L pp`).
#
# Host, no device: the planner's server terms (the MTP head + embedding replica + buffers on
# device 1, device 0's verify slots, the prefix cache's block shadows), the stage verify
# lists' launch arithmetic, spec 7's KV snapshot over two devices = the one-card layout byte
# for byte with the head's layer (pp_serve_plan_test); b70-serve's adapter over a device-free
# two-device engine through spec 7's request path and the HTTP server - snapshots moving
# between one card and two, MTP / lookup iterations, sampling (pp_serve_test).
add_executable(pp_serve_plan_test runtime/pp_serve_plan_test.cc)
target_include_directories(pp_serve_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pp_serve_plan_test PRIVATE b70_plan)
add_test(NAME pp_serve_plan_test COMMAND pp_serve_plan_test)
add_executable(pp_serve_test server/pp_serve_test.cc)
target_include_directories(pp_serve_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pp_serve_test PRIVATE b70_server b70_plan)
add_test(NAME pp_serve_test COMMAND pp_serve_test)
set_tests_properties(pp_serve_test PROPERTIES TIMEOUT 300)

# On the test machine, both cards and the checkpoint; SKIP (77) with fewer than two GPUs or without peer
# access. Review Focus 1-3 (pp_mtp_test): MTP across the split BITWISE one card's on device 0 -
# the prose prompt's prefill (the head's KV filled on device 1), 64 speculative iterations (K 3,
# 1, 2, 0: every draft and verify logits row, the ids, the session with the live verify slot,
# the head's hidden row and KV layer), M2 on both engines (4 plain rows == one M = 4 verify),
# a 4103-id hooked prefill (spec 7 at 2048 / 4096 from the shadows, with the head's row), a
# one-card snapshot continued on two cards; the auto split, the cuts at 5 and
# 62; both Control blocks equal after every call. The int8 head and the int8 KV twins.
if(B70_MTP)
  add_executable(pp_mtp_test runtime/pp_mtp_test.cc)
  target_include_directories(pp_mtp_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(pp_mtp_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  add_dependencies(pp_mtp_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
  add_test(NAME pp_mtp_test COMMAND pp_mtp_test ${B70_TEST_SNAPSHOT})
  add_test(NAME pp_mtp_i8head_test COMMAND pp_mtp_test ${B70_TEST_SNAPSHOT} int8)
  set_tests_properties(pp_mtp_test pp_mtp_i8head_test PROPERTIES
    LABELS "checkpoint;pp;mtp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 5400)
  if(B70_KV8 AND COMMAND b70_kv8_twin)
    add_test(NAME pp_mtp_kv8_test COMMAND pp_mtp_test ${B70_TEST_SNAPSHOT})
    b70_kv8_twin(pp_mtp)
  endif()
endif()
# Review Focus 2 / 5 and spec 16 P2 (spec 7's C2 with --parallel pp): the prefix cache's request path
# through b70-serve's adapters, two cards BITWISE one card - the plans, the final turn's ids and
# rows, the session - on sequences a-f; the split_last path, MTP (K 3, step_many),
# the int8 head; the int8 KV twin.
if(B70_TOKENIZER_ENABLED)
  add_executable(pp_prefix_gpu_test server/pp_prefix_gpu_test.cc)
  target_include_directories(pp_prefix_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(pp_prefix_gpu_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
    b70_model b70_l0 b70_prefill_host)
  add_dependencies(pp_prefix_gpu_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
  set(_pp_prompts ${CMAKE_SOURCE_DIR}/tests/golden/prompts)
  add_test(NAME pp_prefix_gpu_test COMMAND pp_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${_pp_prompts} 0 0)
  add_test(NAME pp_prefix_gpu_split_test COMMAND pp_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${_pp_prompts} 0 1 int8)
  set(_pp_prefix pp_prefix_gpu_test pp_prefix_gpu_split_test)
  if(B70_MTP)
    add_dependencies(pp_prefix_gpu_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
    add_test(NAME pp_prefix_gpu_mtp_test COMMAND pp_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${_pp_prompts} 3 1 int8)
    list(APPEND _pp_prefix pp_prefix_gpu_mtp_test)
  endif()
  set_tests_properties(${_pp_prefix} PROPERTIES
    LABELS "checkpoint;pp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  if(B70_KV8 AND COMMAND b70_kv8_twin)
    add_dependencies(pp_prefix_gpu_test ${B70_KV8_KERNELS})
    add_test(NAME pp_prefix_gpu_kv8_test COMMAND pp_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${_pp_prompts} 0 copy 1 int8)
    b70_kv8_twin(pp_prefix_gpu)
  endif()
endif()

# b70-serve's --parallel refusals before the device (cli/pipeline_serve.h): the model's with
# tests/model/kolibri1 (config.json only), the rest before the snapshot is resolved (the model path does not exist).
if(TARGET b70-serve AND COMMAND b70_serve_reject)
  b70_serve_reject(cli_reject_serve_pp_value ${B70_NO_MODEL} --parallel 4096
    EXPECT "--parallel expects auto, 1, pp, replica or tp")
  b70_serve_reject(cli_reject_serve_pp_split_alone ${B70_NO_MODEL} --parallel 1 --pipeline-split 20
    EXPECT "--pipeline-split belongs to --parallel pp")
  b70_serve_reject(cli_reject_serve_pp_handoff_removed ${B70_NO_MODEL} --parallel pp --pipeline-handoff copy
    EXPECT "unknown option '--pipeline-handoff'")
  # Plan 13b: --parallel replica is Qwen3.8's and Ornith's - other models by name (their config.json), an unreadable
  # checkpoint, the pp override flags; all before the device.
  b70_serve_reject(cli_reject_serve_replica_kolibri ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --parallel replica
    EXPECT "has no replica engine")
  b70_serve_reject(cli_reject_serve_replica_no_model ${B70_NO_MODEL} --parallel replica
    EXPECT "no readable config.json")
  b70_serve_reject(cli_reject_serve_replica_split ${B70_NO_MODEL} --parallel replica --pipeline-split 20
    EXPECT "--pipeline-split belongs to --parallel pp")
  set_tests_properties(cli_reject_serve_pp_value
    cli_reject_serve_pp_split_alone cli_reject_serve_pp_handoff_removed
    cli_reject_serve_replica_kolibri cli_reject_serve_replica_no_model
    cli_reject_serve_replica_split
    PROPERTIES LABELS pp)
endif()
# ==== Spec 16d ============================================================= (end) ====

# ==== Spec 20c: Kolibri-1 decode (label kolibri) ==== (begin) ====
# One delimited block for every 20c registration (spec 16c's and 20d's blocks stay apart). Host
# tests first (they run on the Mac), then the card's (GPU only; validation
# queue row 24). Real-checkpoint tests SKIP (77) until spec 20b's checkpoint exists; the synthetic
# checkpoints (tools/oracle/kolibri_ref.py synth) carry KL2 / KL3 before it.
# Task 2: the model table.
add_executable(kolibri1_test model/kolibri1_test.cc)
target_include_directories(kolibri1_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri1_test PRIVATE b70_kolibri_model)
add_test(NAME kolibri1_test COMMAND kolibri1_test ${CMAKE_SOURCE_DIR}/tests/model/kolibri1/config.json
  ${CMAKE_SOURCE_DIR}/tests/model/kolibri1/synth5_bf16attn.json)
set_tests_properties(kolibri1_test PROPERTIES LABELS kolibri)
# Task 3: the loader's host half on a synthetic checkpoint (both attention arms), the RoPE table
# against torch's; then the card's load of the synthetic checkpoints and of the real one truncated.
foreach(t kolibri1_repack_test kolibri1_rope_test)
  add_executable(${t} loader/${t}.cc)
  target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(${t} PRIVATE b70_kolibri_loader)
  add_test(NAME ${t} COMMAND ${t})
  set_tests_properties(${t} PROPERTIES LABELS kolibri)
endforeach()
set(B70_KOLIBRI_SNAPSHOT "urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ" CACHE STRING "Kolibri-1 int4 checkpoint (spec 20b)")
set(B70_KOLIBRI_SYNTH_DIR "${B70_ORACLE}/kolibri/synth" CACHE PATH
  "Task 1's synthetic checkpoints and golden sets (tools/oracle/kolibri_ref.py synth)")
add_executable(kolibri1_load_checkpoint_test loader/kolibri1_load_checkpoint_test.cc)
target_include_directories(kolibri1_load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri1_load_checkpoint_test PRIVATE b70_kolibri_loader b70_l0)
add_test(NAME kolibri1_load_synth_int4attn_test COMMAND kolibri1_load_checkpoint_test ${B70_KOLIBRI_SYNTH_DIR}/int4attn/ckpt)
add_test(NAME kolibri1_load_synth_bf16attn_test COMMAND kolibri1_load_checkpoint_test ${B70_KOLIBRI_SYNTH_DIR}/bf16attn/ckpt int8)
add_test(NAME kolibri1_load_partial_test COMMAND kolibri1_load_checkpoint_test ${B70_KOLIBRI_SNAPSHOT} bf16 30)
set_tests_properties(kolibri1_load_synth_int4attn_test kolibri1_load_synth_bf16attn_test kolibri1_load_partial_test
  PROPERTIES LABELS "checkpoint;kolibri" SKIP_RETURN_CODE 77 TIMEOUT 1800)
# Task 4: the kernels' host twin (tests/kernels/kolibri_ref.h) against kolibri_ref.py's own outputs
# (tests/kernels/kolibri_fixture.h) and the semantics' properties. Host only, headers only.
add_executable(kolibri_ref_test kernels/kolibri_ref_test.cc)
target_include_directories(kolibri_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME kolibri_ref_test COMMAND kolibri_ref_test)
set_tests_properties(kolibri_ref_test PROPERTIES LABELS kolibri)
if(B70_KOLIBRI)
  # Every binary a Kolibri decode list binds, any arm / head / attention (src/kernels/CMakeLists.txt's
  # Kolibri block).
  set(B70_KOLIBRI_DECODE_KERNELS
    kernel_kol_embed_gather_M1_D2560_V128000 kernel_kol_argmax_stage1_M1_N128000_V128000
    kernel_kol_argmax_stage2_N128000 kernel_prep_res_fold_M1_K2560_SP0_G20
    kernel_prep_res_fold_M1_K2560_SP1_G20_Z kernel_prep_res_fold_M1_K2560_SP4_G20_Z
    kernel_kol_norm_M1_K2560_G20_W20 kernel_kol_post_add_M1_K2560_G20
    kernel_kol_attn_prep_M1_N7168_S2_Q48KV4_R4096 kernel_kol_attn_prep_M1_N7168_S2_Q48KV4_F
    kernel_kol_attn_prep_M1_N7168_S1_Q48KV4_R4096 kernel_kol_attn_prep_M1_N7168_S1_Q48KV4_F
    kernel_kol_attn_M1_T${ATTN_V2_TGT}_Q48KV4_W513_R4096 kernel_kol_attn_M1_T${ATTN_V2_TGT}_Q48KV4_F
    kernel_kol_attn_eager_M1_T${ATTN_V2_TGT}_Q48KV4_W513_R4096 kernel_kol_attn_eager_M1_T${ATTN_V2_TGT}_Q48KV4_F
    kernel_kol_route_M1_E384_T6_N512_L256 kernel_kol_moe_M1_E384_T6_D2560_I512_SH
    kernel_gemv_M1_K2560_N7168_S2_L0 kernel_gemv_M1_K6144_N2560_S4_L0
    kernel_gemv_bf16_M1_K2560_N7168 kernel_gemv_bf16_M1_K6144_N2560 kernel_gemv_bf16_M1_K2560_N512_C16_S16
    kernel_gemv_bf16_M1_K2560_N128000 kernel_gemv_i8w_M1_K2560_N128000)
  # Host: the names the capture binds (kernels::kolibri::decode_variants) are exactly this list.
  add_executable(kolibri_variant_names_test kernels/kolibri_variant_names_test.cc)
  target_include_directories(kolibri_variant_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kolibri_variant_names_test PRIVATE b70_kolibri_model)
  b70_target_kernel_dir(kolibri_variant_names_test)
  add_test(NAME kolibri_variant_names_test COMMAND kolibri_variant_names_test ${B70_KOLIBRI_DECODE_KERNELS})
  set_tests_properties(kolibri_variant_names_test PROPERTIES LABELS kolibri)
  # K1 on the card: every Kolibri kernel at the real shapes against kolibri_ref.h (no checkpoint).
  add_executable(kolibri_kernels_test kernels/kolibri_kernels_test.cc)
  target_include_directories(kolibri_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kolibri_kernels_test PRIVATE b70_l0 b70_kolibri_loader)
  b70_target_kernel_dir(kolibri_kernels_test)
  add_dependencies(kolibri_kernels_test ${B70_KOLIBRI_DECODE_KERNELS})
  add_test(NAME kolibri_kernels_test COMMAND kolibri_kernels_test)
  set_tests_properties(kolibri_kernels_test PROPERTIES LABELS kolibri TIMEOUT 900)
endif()
# Task 5: the runtime. The planner, the launch counts and the split by bytes on the host ...
add_executable(kolibri_plan_test runtime/kolibri_plan_test.cc)
target_include_directories(kolibri_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri_plan_test PRIVATE b70_kolibri_plan)
add_test(NAME kolibri_plan_test COMMAND kolibri_plan_test)
set_tests_properties(kolibri_plan_test PROPERTIES LABELS kolibri)
# ... and on the card: plan == allocation, K3 replay determinism and the ring (kolibri_decode_test); KL2,
# the tie-aware gate with the routing diagnostic (kolibri_golden_test) on both synthetic arms, flash and
# eager, and the int8 head; the real checkpoint truncated to 30 layers on one card (kolibri_partial_test,
# development mode; needs spec 20b). SKIP 77 without the data (tools/oracle/kolibri_ref.py).
if(B70_KOLIBRI)
  set(B70_KOLIBRI_ORACLE_DIR "${B70_ORACLE}/kolibri/golden"
    CACHE PATH "The real checkpoint's golden set (kolibri_oracle.sh real): <prompt>.ids + .golden.safetensors")
  # Kolibri's golden gates judged by the f32 reference teacher-forced on each
  # golden's tokens (kolibri_oracle.sh real-f32 / synth-f32; kolibri_golden_test.cc / kolibri_partial_test.cc
  # `--judge=`): the engine no further from f32 than the bf16 golden.
  set(B70_KOLIBRI_F32_DIR "${B70_ORACLE}/kolibri/f32"
    CACHE PATH "The real checkpoint's f32 judge (kolibri_oracle.sh real-f32)")
  set(B70_KOLIBRI_SYNTH_F32_DIR "${B70_ORACLE}/kolibri/synth-f32"
    CACHE PATH "The synthetic arms' f32 judges, <arm>/ (kolibri_oracle.sh synth-f32)")
  foreach(t kolibri_decode_test kolibri_golden_test kolibri_partial_test)
    if(t STREQUAL "kolibri_decode_test")
      add_executable(${t} runtime/${t}.cc)
    else()
      add_executable(${t} golden/${t}.cc)
    endif()
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
      ${CMAKE_SOURCE_DIR}/tests/runtime)
    target_link_libraries(${t} PRIVATE b70_kolibri_runtime b70_l0)
    b70_target_kernel_dir(${t})
    add_dependencies(${t} ${B70_KOLIBRI_DECODE_KERNELS})
  endforeach()
  set(_kol_synth ${B70_KOLIBRI_SYNTH_DIR})
  add_test(NAME kolibri_decode_synth_test COMMAND kolibri_decode_test ${_kol_synth}/int4attn/ckpt
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/kolibri_bench.ids)
  add_test(NAME kolibri_golden_synth_int4attn_test COMMAND kolibri_golden_test ${_kol_synth}/int4attn/ckpt
    ${_kol_synth}/int4attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/int4attn)
  add_test(NAME kolibri_golden_synth_bf16attn_test COMMAND kolibri_golden_test ${_kol_synth}/bf16attn/ckpt
    ${_kol_synth}/bf16attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/bf16attn)
  add_test(NAME kolibri_golden_synth_int4attn_eager_test COMMAND kolibri_golden_test ${_kol_synth}/int4attn/ckpt
    ${_kol_synth}/int4attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/int4attn)
  add_test(NAME kolibri_golden_synth_bf16attn_eager_test COMMAND kolibri_golden_test ${_kol_synth}/bf16attn/ckpt
    ${_kol_synth}/bf16attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/bf16attn)
  set_tests_properties(kolibri_golden_synth_int4attn_eager_test kolibri_golden_synth_bf16attn_eager_test
    PROPERTIES ENVIRONMENT "B70_KOLIBRI_ATTN=eager")
  add_test(NAME kolibri_golden_synth_i8head_test COMMAND kolibri_golden_test ${_kol_synth}/int4attn/ckpt
    ${_kol_synth}/int4attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/int4attn int8)
  add_test(NAME kolibri_partial_test COMMAND kolibri_partial_test ${B70_KOLIBRI_SNAPSHOT} ${B70_KOLIBRI_ORACLE_DIR} --judge=${B70_KOLIBRI_F32_DIR} 30)
  add_test(NAME kolibri_partial_eager_test COMMAND kolibri_partial_test ${B70_KOLIBRI_SNAPSHOT} ${B70_KOLIBRI_ORACLE_DIR} --judge=${B70_KOLIBRI_F32_DIR} 30)
  set_tests_properties(kolibri_partial_eager_test PROPERTIES ENVIRONMENT "B70_KOLIBRI_ATTN=eager")
  # eager is diagnostic - under the f32 judge these print their verdicts, never fail.
  set_property(TEST kolibri_golden_synth_int4attn_eager_test kolibri_golden_synth_bf16attn_eager_test
    kolibri_partial_eager_test APPEND PROPERTY LABELS diagnostic)
  set_tests_properties(kolibri_decode_synth_test kolibri_golden_synth_int4attn_test kolibri_golden_synth_bf16attn_test
    kolibri_golden_synth_int4attn_eager_test kolibri_golden_synth_bf16attn_eager_test kolibri_golden_synth_i8head_test
    kolibri_partial_test kolibri_partial_eager_test
    PROPERTIES LABELS "checkpoint;golden;kolibri" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# Task 6: two cards on spec 16b's pieces - --parallel 1 against --parallel pp bitwise on the synthetic checkpoint
# (the synthetic), the real one across splits 25 / 20, P4's lost hand-off, replay (kolibri_pp_test);
# KL2 and K3 on the real checkpoint across two cards (needs spec 20b). SKIP 77 with one GPU or no data.
if(B70_KOLIBRI)
  add_executable(kolibri_pp_test runtime/kolibri_pp_test.cc)
  target_include_directories(kolibri_pp_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(kolibri_pp_test PRIVATE b70_kolibri_runtime b70_l0)
  add_dependencies(kolibri_pp_test ${B70_KOLIBRI_DECODE_KERNELS})
  add_test(NAME kolibri_pp_test COMMAND kolibri_pp_test ${B70_KOLIBRI_SYNTH_DIR}/int4attn/ckpt ${B70_KOLIBRI_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/kolibri_bench.ids)
  set_tests_properties(kolibri_pp_test PROPERTIES LABELS "checkpoint;kolibri;pp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  add_test(NAME kolibri_golden_test COMMAND kolibri_golden_test ${B70_KOLIBRI_SNAPSHOT} ${B70_KOLIBRI_ORACLE_DIR} --judge=${B70_KOLIBRI_F32_DIR} pp2)
  add_test(NAME kolibri_golden_i8head_test COMMAND kolibri_golden_test ${B70_KOLIBRI_SNAPSHOT} ${B70_KOLIBRI_ORACLE_DIR} --judge=${B70_KOLIBRI_F32_DIR}
    int8 pp2)
  add_test(NAME kolibri_golden_eager_test COMMAND kolibri_golden_test ${B70_KOLIBRI_SNAPSHOT} ${B70_KOLIBRI_ORACLE_DIR} --judge=${B70_KOLIBRI_F32_DIR} pp2)
  set_tests_properties(kolibri_golden_eager_test PROPERTIES ENVIRONMENT "B70_KOLIBRI_ATTN=eager")
  set_property(TEST kolibri_golden_eager_test APPEND PROPERTY LABELS diagnostic)   # ruling 2026-10-29
  add_test(NAME kolibri_decode_test COMMAND kolibri_decode_test ${B70_KOLIBRI_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/kolibri_bench.ids pp2)
  set_tests_properties(kolibri_golden_test kolibri_golden_i8head_test kolibri_golden_eager_test
    kolibri_decode_test PROPERTIES LABELS "checkpoint;golden;kolibri;pp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# b70-decode's model_type dispatch and Kolibri's refusals, before the device: tests/model/kolibri1/ holds
# only the checkpoint's config.json, which is all the dispatch and the --parallel 1 fit check read.
set(_kol_cfg ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1)
b70_cli_reject(cli_reject_kolibri_one_card ${_kol_cfg} --ids ${B70_OK_IDS} --n 4 --parallel 1 EXPECT "--parallel pp")
b70_cli_reject(cli_reject_kolibri_mtp ${_kol_cfg} --ids ${B70_OK_IDS} --n 4 --spec mtp EXPECT "no MTP head")
b70_cli_reject(cli_reject_kolibri_kv8 ${_kol_cfg} --ids ${B70_OK_IDS} --n 4 --kv-cache int8 EXPECT "bf16 only")
b70_cli_reject(cli_reject_kolibri_layers_qwen ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --layers 3
  EXPECT "--layers belongs to Kolibri-1")
# (20c's cli_reject_kolibri_serve - b70-serve refusing kolibri1 - is gone: spec 20e serves it; its refusals
# are the cli_reject_serve_kolibri_* tests of the spec 20e block.)
# ==== Spec 20c: Kolibri-1 decode ==== (end) ====

# ==== Spec 20d: Kolibri-1 prefill (labels kolibri;prefill) ==== (begin) ====
# One delimited block for every 20d registration, after 20c's. Written blind on the Mac (validation queue row
# 26): host tests first (the host checks run them), then the card's; the checkpoint ones SKIP (77)
# until the synthetic sets (kolibri_oracle.sh synth) or, for the real checkpoint, spec 20b exist.
# Task 1: the prefill kernels' host twin (tests/kernels/kolibri_pf_ref.h) - the 384-expert sort and its
# adversaries, the combine == 20c's decode chain and torch's fixture row, the windowed flash walk over
# the ring in fp64, eager_window against the fixture's eager rows, the bf16 slab tail. Headers only.
add_executable(kolibri_pf_ref_test kernels/kolibri_pf_ref_test.cc)
target_include_directories(kolibri_pf_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME kolibri_pf_ref_test COMMAND kolibri_pf_ref_test)
set_tests_properties(kolibri_pf_ref_test PROPERTIES LABELS "kolibri;prefill")
if(B70_KOLIBRI)
  # Task 2: every binary the prefill walk binds, both attention arms and forms (src/kernels/CMakeLists.txt's
  # spec 20d block; pf_gemm_T0 and 20c's kol_route are reused).
  set(B70_KOLIBRI_PREFILL_KERNELS
    kernel_kol_pf_embed_gather_D2560_V128000 kernel_pf_res_fold_K2560_SP0_G20 kernel_pf_res_fold_K2560_SP1_G20_Z
    kernel_kol_norm_M2048_K2560_G20_W20 kernel_kol_post_add_M2048_K2560_G20
    kernel_pf_int4_slab_K2560_N7168 kernel_pf_int4_slab_K6144_N2560
    kernel_kol_pf_bf16_slab_K2560_N7168 kernel_kol_pf_bf16_slab_K6144_N2560 kernel_pf_gemm_T0
    kernel_kol_attn_prep_M2048_N7168_S1_Q48KV4_R4096 kernel_kol_attn_prep_M2048_N7168_S1_Q48KV4_F
    kernel_kol_pf_flash_attn_Q48KV4_W513_R4096 kernel_kol_pf_flash_attn_Q48KV4_W513_R4096_EAGER
    kernel_kol_pf_flash_attn_Q48KV4_F kernel_kol_pf_flash_attn_Q48KV4_F_EAGER
    kernel_pf_moe_router_K2560_N512 kernel_kol_route_M1_E384_T6_N512_L256 kernel_kol_pf_moe_E384_T6_D2560_I512_L256
    kernel_pf_moe_gemm_K2560_N1024_SILU kernel_pf_moe_gemm_K512_N2560)
  # Host: the names the walk binds (kernels::kolibri::prefill_variants) are exactly this list.
  add_executable(kolibri_pf_variant_names_test kernels/kolibri_pf_variant_names_test.cc)
  target_include_directories(kolibri_pf_variant_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kolibri_pf_variant_names_test PRIVATE b70_kolibri_model)
  b70_target_kernel_dir(kolibri_pf_variant_names_test)
  add_test(NAME kolibri_pf_variant_names_test COMMAND kolibri_pf_variant_names_test ${B70_KOLIBRI_PREFILL_KERNELS})
  set_tests_properties(kolibri_pf_variant_names_test PROPERTIES LABELS "kolibri;prefill")
  # K1 for the prefill on the card, no checkpoint: every prefill kernel at the real shapes against
  # kolibri_pf_ref.h, grouped == dense (pf_gemm_T0_SILU / pf_gemm_T0), the windowed flash against fp64.
  add_executable(kolibri_pf_kernels_test kernels/kolibri_pf_kernels_test.cc)
  target_include_directories(kolibri_pf_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kolibri_pf_kernels_test PRIVATE b70_prefill_host b70_kolibri_loader b70_l0)
  b70_target_kernel_dir(kolibri_pf_kernels_test)
  add_dependencies(kolibri_pf_kernels_test ${B70_KOLIBRI_PREFILL_KERNELS} ${B70_KOLIBRI_DECODE_KERNELS}
    kernel_pf_gemm_T0_SILU)
  add_test(NAME kolibri_pf_kernels_test COMMAND kolibri_pf_kernels_test)
  set_tests_properties(kolibri_pf_kernels_test PROPERTIES LABELS "kolibri;prefill" TIMEOUT 1800)
endif()
# Task 3: the prefill walk on the card (GPU only; SKIP 77 without the synthetic sets / the real checkpoint).
#   kolibri_prefill_test: the walk's launches and the plan, K3 (two prefills, recorded == replayed, chunks
#       of 16 == chunks of 2048, all bitwise), Review Focus 4 (KV / rings / routes / tokens against decode's
#       fill); its `split` mode is prefill_split_kolibri_test (Review Focus 5), its `pp` mode one card
#       against two, bitwise (needs two GPUs);
#   kolibri_golden_test's `prefill` mode: KL2 on prefill (the synthetic arms; the real checkpoint after 20b).
if(B70_KOLIBRI)
  add_executable(kolibri_prefill_test runtime/kolibri_prefill_test.cc)
  target_include_directories(kolibri_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(kolibri_prefill_test PRIVATE b70_kolibri_prefill b70_l0)
  add_dependencies(kolibri_prefill_test ${B70_KOLIBRI_PREFILL_KERNELS} ${B70_KOLIBRI_DECODE_KERNELS})
  set(_kol_synth ${B70_KOLIBRI_SYNTH_DIR})
  set(_kol_ids ${CMAKE_SOURCE_DIR}/tests/golden/prompts/kolibri_bench.ids)
  add_test(NAME kolibri_prefill_synth_int4attn_test COMMAND kolibri_prefill_test ${_kol_synth}/int4attn/ckpt ${_kol_ids} all)
  add_test(NAME kolibri_prefill_synth_bf16attn_test COMMAND kolibri_prefill_test ${_kol_synth}/bf16attn/ckpt ${_kol_ids} all)
  add_test(NAME kolibri_prefill_synth_eager_test COMMAND kolibri_prefill_test ${_kol_synth}/int4attn/ckpt ${_kol_ids} all)
  set_tests_properties(kolibri_prefill_synth_eager_test PROPERTIES ENVIRONMENT "B70_KOLIBRI_ATTN=eager")
  add_test(NAME prefill_split_kolibri_test COMMAND kolibri_prefill_test ${_kol_synth}/int4attn/ckpt ${_kol_ids} split)
  set_tests_properties(kolibri_prefill_synth_int4attn_test kolibri_prefill_synth_bf16attn_test
    kolibri_prefill_synth_eager_test prefill_split_kolibri_test
    PROPERTIES LABELS "checkpoint;kolibri;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  # two cards (16b's pieces): one card against --parallel pp --pipeline-split 3 bitwise on the synthetic checkpoint,
  # the split test across two cards; the real checkpoint on two cards (needs spec 20b)
  add_test(NAME kolibri_prefill_pp_test COMMAND kolibri_prefill_test ${_kol_synth}/int4attn/ckpt ${_kol_ids} pp pp2:3)
  add_test(NAME prefill_split_kolibri_pp_test COMMAND kolibri_prefill_test ${_kol_synth}/int4attn/ckpt ${_kol_ids}
    split pp2:3)
  add_test(NAME kolibri_prefill_test COMMAND kolibri_prefill_test ${B70_KOLIBRI_SNAPSHOT} ${_kol_ids} all pp2)
  add_test(NAME prefill_split_kolibri_real_test COMMAND kolibri_prefill_test ${B70_KOLIBRI_SNAPSHOT} ${_kol_ids} split pp2)
  set_tests_properties(kolibri_prefill_pp_test prefill_split_kolibri_pp_test kolibri_prefill_test
    prefill_split_kolibri_real_test PROPERTIES LABELS "checkpoint;kolibri;prefill;pp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  # KL2 on prefill: 20c's golden test in its prefill mode (one chunk, and chunks of 16), both synthetic arms,
  # eager, the int8 head; the real checkpoint on two cards after 20b.
  target_link_libraries(kolibri_golden_test PRIVATE b70_kolibri_prefill)
  add_dependencies(kolibri_golden_test ${B70_KOLIBRI_PREFILL_KERNELS})
  add_test(NAME kolibri_golden_prefill_synth_int4attn_test COMMAND kolibri_golden_test ${_kol_synth}/int4attn/ckpt
    ${_kol_synth}/int4attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/int4attn prefill)
  add_test(NAME kolibri_golden_prefill_synth_bf16attn_test COMMAND kolibri_golden_test ${_kol_synth}/bf16attn/ckpt
    ${_kol_synth}/bf16attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/bf16attn prefill)
  add_test(NAME kolibri_golden_prefill_synth_c16_test COMMAND kolibri_golden_test ${_kol_synth}/int4attn/ckpt
    ${_kol_synth}/int4attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/int4attn prefill:16)
  add_test(NAME kolibri_golden_prefill_synth_i8head_test COMMAND kolibri_golden_test ${_kol_synth}/int4attn/ckpt
    ${_kol_synth}/int4attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/int4attn int8 prefill)
  add_test(NAME kolibri_golden_prefill_synth_eager_test COMMAND kolibri_golden_test ${_kol_synth}/int4attn/ckpt
    ${_kol_synth}/int4attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/int4attn prefill)
  set_tests_properties(kolibri_golden_prefill_synth_eager_test PROPERTIES ENVIRONMENT "B70_KOLIBRI_ATTN=eager")
  set_property(TEST kolibri_golden_prefill_synth_eager_test APPEND PROPERTY LABELS diagnostic)   # ruling 2026-10-29
  add_test(NAME kolibri_golden_prefill_synth_pp_test COMMAND kolibri_golden_test ${_kol_synth}/int4attn/ckpt
    ${_kol_synth}/int4attn --judge=${B70_KOLIBRI_SYNTH_F32_DIR}/int4attn pp2:3 prefill)
  set_tests_properties(kolibri_golden_prefill_synth_int4attn_test kolibri_golden_prefill_synth_bf16attn_test
    kolibri_golden_prefill_synth_c16_test kolibri_golden_prefill_synth_i8head_test
    kolibri_golden_prefill_synth_eager_test kolibri_golden_prefill_synth_pp_test
    PROPERTIES LABELS "checkpoint;golden;kolibri;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  add_test(NAME kolibri_golden_prefill_test COMMAND kolibri_golden_test ${B70_KOLIBRI_SNAPSHOT} ${B70_KOLIBRI_ORACLE_DIR} --judge=${B70_KOLIBRI_F32_DIR}
    pp2 prefill)
  add_test(NAME kolibri_golden_prefill_c16_test COMMAND kolibri_golden_test ${B70_KOLIBRI_SNAPSHOT}
    ${B70_KOLIBRI_ORACLE_DIR} --judge=${B70_KOLIBRI_F32_DIR} pp2 prefill:16)
  add_test(NAME kolibri_golden_prefill_i8head_test COMMAND kolibri_golden_test ${B70_KOLIBRI_SNAPSHOT}
    ${B70_KOLIBRI_ORACLE_DIR} --judge=${B70_KOLIBRI_F32_DIR} int8 pp2 prefill)
  set_tests_properties(kolibri_golden_prefill_test kolibri_golden_prefill_c16_test kolibri_golden_prefill_i8head_test
    PROPERTIES LABELS "checkpoint;golden;kolibri;prefill;pp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# Task 3: b70-decode's Kolibri prefill backends and chunk, before the device (tests/model/kolibri1: config.json
# only). 20c's cli_reject_kolibri_prefill (--prefill naming spec 20d) is gone: Kolibri prefills now.
b70_cli_reject(cli_reject_kolibri_prefill_int8 ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --bench --prefill-length 64
  --prefill-backend l0-int8 EXPECT "1024-k Hadamard blocks")
b70_cli_reject(cli_reject_kolibri_prefill_chunk ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --ids ${B70_OK_IDS} --n 4
  --prefill --prefill-chunk 4096 EXPECT "exceeds Kolibri-1's prefill chunk 2048")
# ==== Spec 20d: Kolibri-1 prefill ==== (end) ====
# ==== Spec 20e: Kolibri-1 served (label kolibri) ==== (begin) ====
if(B70_TOKENIZER_ENABLED)
  # Kolibri-1's chat template - Aleph-Alpha/Kolibri-1-BF16 at 8c8b3489's tokenizer_config.json "chat_template"
  # (the release ships no chat_template.jinja; chat::Template reads the string), vendored with its
  # generation_config at tests/tokenizer/kolibri/ (Apache-2.0) - on 16 message lists (every reasoning_effort,
  # enable_thinking off, tools with and without a system turn, call history, parallel calls and their tool
  # results, reasoning before and after the last query, the `reasoning` field, <think> inside content, German)
  # against transformers' apply_chat_template (tools/tokenizer/dump_kolibri.py).
  add_test(NAME template_kolibri_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/kolibri kolibri_template_cases.json kolibri)
  # ... and its 128k tokenizer against HF's (the id count and the two ids the head has but the tokenizer does
  # not define, every tag, a German-heavy text set, corpus.txt's digests, three chat renders' ids).
  # tokenizer.json (9.5 MB) is not vendored: the BF16 release's in the HF cache (20b's export copies it
  # unchanged), or -DB70_KOLIBRI_TOKENIZER_JSON / the environment variable of the same name. Absent:
  # disabled on the Mac (host_only), SKIP (77) elsewhere.
  b70_hf_snapshot(B70_KOLIBRI_TOKENIZER_JSON models--Aleph-Alpha--Kolibri-1-BF16
    8c8b34899c7bc77b349f7916b07158e3ba29456a /tokenizer.json FILEPATH
    "Kolibri-1's tokenizer.json for kolibri_tokenizer_test (-D or the environment variable)")
  add_executable(kolibri_tokenizer_test tokenizer/kolibri_tokenizer_test.cc)
  target_include_directories(kolibri_tokenizer_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kolibri_tokenizer_test PRIVATE b70_tokenizer)
  add_test(NAME kolibri_tokenizer_test COMMAND kolibri_tokenizer_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/kolibri ${B70_KOLIBRI_TOKENIZER_JSON})
  set_tests_properties(template_kolibri_test kolibri_tokenizer_test PROPERTIES LABELS kolibri SKIP_RETURN_CODE 77)
  # The 2026-11-04 tool-call item: a 20-turn tool session through the server with the vendored template (every prompt
  # the template's render, by induction; reasoning echoed or not; repaired raw-newline calls), the thinking switches
  # (top-level reasoning_effort included) and a max_tokens cut inside the reasoning, over a scripted engine.
  add_executable(kolibri_session_test server/kolibri_session_test.cc)
  target_include_directories(kolibri_session_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(kolibri_session_test PRIVATE b70_server b70_tokenizer)
  add_test(NAME kolibri_session_test COMMAND kolibri_session_test ${CMAKE_SOURCE_DIR}/tests/tokenizer/kolibri)
  set_tests_properties(kolibri_session_test PROPERTIES LABELS kolibri)
endif()
# Task 2 (host): Kolibri's reasoning (opened by the model) and hermes JSON tool calls - server/toolcall_kolibri.h
# over the template renders above, the Review Focus cases, German bytes and a 2000-output fuzz, every parse
# split-invariant; the server's kolibri1 dispatch (kwargs, list content, both EOS ids, streamed frames) and the
# model-scoped sampling defaults from the vendored generation_config.json, over a mock engine.
add_executable(toolcall_kolibri_test server/toolcall_kolibri_test.cc)
target_include_directories(toolcall_kolibri_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(toolcall_kolibri_test PRIVATE b70_server)
add_test(NAME toolcall_kolibri_test COMMAND toolcall_kolibri_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
add_executable(kolibri_server_test server/kolibri_server_test.cc)
target_include_directories(kolibri_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri_server_test PRIVATE b70_server)
add_test(NAME kolibri_server_test COMMAND kolibri_server_test ${CMAKE_SOURCE_DIR}/tests/tokenizer/kolibri)
set_tests_properties(toolcall_kolibri_test kolibri_server_test PROPERTIES LABELS kolibri)
# Task 3 (host): the engine behind the server without a device - cli::kolibri::KolibriEngineAdapterT over a
# fake with KolibriEngine's interface whose rings and full KV are host buffers in KolibriBuffers' layout (one
# card and two): runtime::kolibri::state_runs / kv_runs on the real shapes (41,943,040 B of state, 20,480 B of
# KV a position, one host layout under --parallel 1 and --parallel pp), save / load on host buffers, Review Focus 4 through
# spec 7's PrefixSession (restores at 2048, 4096, 2049, 5000 and 300, every turn == a cold run), a snapshot
# moving between one card and two, and the server over HTTP (greedy == the decode run, a cached second turn,
# seeded sampling with the model's defaults, no id the tokenizer lacks).
add_executable(kolibri_snapshot_test runtime/kolibri_snapshot_test.cc)
target_include_directories(kolibri_snapshot_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri_snapshot_test PRIVATE b70_server b70_kolibri_plan)
add_test(NAME kolibri_snapshot_test COMMAND kolibri_snapshot_test)
set_tests_properties(kolibri_snapshot_test PROPERTIES LABELS kolibri)
# On the card (GPU only; SKIP 77 without the synthetic sets / the real checkpoint): kolibri_snapshot_gpu_test -
# KolibriEngine's snapshot calls (Review Focus 4): restores at the block end 4096 (bitwise the cold run), at the
# request ends 2049 and 300 (bitwise the session that never left), on one card and on two (pp2:3); `cross`: a
# snapshot taken on two cards continues on one, and the reverse (spec 16b's rule; two GPUs). The real
# checkpoint on two cards after spec 20b.
if(B70_KOLIBRI)
  add_executable(kolibri_snapshot_gpu_test runtime/kolibri_snapshot_gpu_test.cc)
  target_include_directories(kolibri_snapshot_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(kolibri_snapshot_gpu_test PRIVATE b70_kolibri_prefill b70_l0)
  add_dependencies(kolibri_snapshot_gpu_test ${B70_KOLIBRI_PREFILL_KERNELS} ${B70_KOLIBRI_DECODE_KERNELS})
  set(_kol_synth ${B70_KOLIBRI_SYNTH_DIR})
  set(_kol_ids ${CMAKE_SOURCE_DIR}/tests/golden/prompts/kolibri_bench.ids)
  add_test(NAME kolibri_snapshot_gpu_synth_test COMMAND kolibri_snapshot_gpu_test ${_kol_synth}/int4attn/ckpt ${_kol_ids})
  add_test(NAME kolibri_snapshot_gpu_synth_pp_test COMMAND kolibri_snapshot_gpu_test ${_kol_synth}/int4attn/ckpt
    ${_kol_ids} pp2:3)
  add_test(NAME kolibri_snapshot_gpu_cross_test COMMAND kolibri_snapshot_gpu_test ${_kol_synth}/int4attn/ckpt
    ${_kol_ids} cross)
  add_test(NAME kolibri_snapshot_gpu_test COMMAND kolibri_snapshot_gpu_test ${B70_KOLIBRI_SNAPSHOT} ${_kol_ids} pp2)
  set_tests_properties(kolibri_snapshot_gpu_synth_test kolibri_snapshot_gpu_synth_pp_test kolibri_snapshot_gpu_cross_test
    kolibri_snapshot_gpu_test PROPERTIES LABELS "checkpoint;kolibri" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# b70-serve's Kolibri refusals, before the device (tests/model/kolibri1: config.json only - all the dispatch,
# cli::kolibri::check_args and the --parallel 1 fit check read).
if(TARGET b70-serve AND COMMAND b70_serve_reject)
  set(_kol_cfg ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1)
  b70_serve_reject(cli_reject_serve_kolibri_spec_mtp ${_kol_cfg} --spec mtp EXPECT "no MTP head")
  b70_serve_reject(cli_reject_serve_kolibri_kv8 ${_kol_cfg} --kv-cache int8 EXPECT "bf16 only")
  b70_serve_reject(cli_reject_serve_kolibri_prefill_int8 ${_kol_cfg} --prefill-backend l0-int8
    EXPECT "1024-k Hadamard blocks")
  b70_serve_reject(cli_reject_serve_kolibri_one_card ${_kol_cfg} --parallel 1 EXPECT "run it with --parallel pp")
  b70_serve_reject(cli_reject_serve_kolibri_max_len ${_kol_cfg} --max-len 524288 EXPECT "a longer context is not supported")
  set_tests_properties(cli_reject_serve_kolibri_spec_mtp
    cli_reject_serve_kolibri_kv8 cli_reject_serve_kolibri_prefill_int8 cli_reject_serve_kolibri_one_card
    cli_reject_serve_kolibri_max_len PROPERTIES LABELS kolibri)
endif()
# ==== Spec 20e: Kolibri-1 served ==== (end) ====

# ==== Spec 21-QFN-v1b: Qwen3.8-Flash-Next descriptor and loader (label qwen4exp) ==== (begin) ====
# One delimited block for every 21-QFN-v1b registration. Host tests first (they run on the Mac:
# the host checks), then the card's (GPU only; validation queue row 31): the loads SKIP (77)
# until the synthetic checkpoints (tools/quantize/qwen4exp/make_synth.py + ple_int8.py, r31.synth) or
# Intel's checkpoint and its PLE file (r31.ple_convert) are on the GPU machine.
# Task 2: the model table (the configs are built in code: the original's config.json is not vendored).
add_executable(qwen4exp_test model/qwen4exp_test.cc)
target_include_directories(qwen4exp_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen4exp_test PRIVATE b70_qwen4exp_model)
add_test(NAME qwen4exp_test COMMAND qwen4exp_test)
set_tests_properties(qwen4exp_test PROPERTIES LABELS qwen4exp)
# Task 3: the loader's host half on synthetic real-width checkpoints in both forms (written by the test).
add_executable(qwen4exp_repack_test loader/qwen4exp_repack_test.cc)
target_include_directories(qwen4exp_repack_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen4exp_repack_test PRIVATE b70_qwen4exp_loader)
add_test(NAME qwen4exp_repack_test COMMAND qwen4exp_repack_test)
set_tests_properties(qwen4exp_repack_test PROPERTIES LABELS qwen4exp TIMEOUT 900)
# Task 4: the PLE table's host half - the hash against transformers' (tests/loader/qwen4exp_ple_fixture.h,
# tools/oracle/qwen4exp_ple_fixture.py), the int8 file per head, the refusals by name.
add_executable(qwen4exp_ple_test loader/qwen4exp_ple_test.cc)
target_include_directories(qwen4exp_ple_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/tests/loader)
target_link_libraries(qwen4exp_ple_test PRIVATE b70_qwen4exp_loader)
add_test(NAME qwen4exp_ple_test COMMAND qwen4exp_ple_test)
set_tests_properties(qwen4exp_ple_test PROPERTIES LABELS qwen4exp)
# Task 5: the memory plan - N per card, the split by bytes, the full model refused until spec 22. Host only.
add_executable(qwen4exp_plan_test runtime/qwen4exp_plan_test.cc)
target_include_directories(qwen4exp_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen4exp_plan_test PRIVATE b70_qwen4exp_plan)
add_test(NAME qwen4exp_plan_test COMMAND qwen4exp_plan_test)
set_tests_properties(qwen4exp_plan_test PROPERTIES LABELS qwen4exp)
# Task 6: the loader on the card - the synthetic checkpoints (both forms, with the MTP head) and Intel's
# checkpoint at 18 layers (one card's fit with its bf16 dense layers, development mode); each with its PLE int8
# file beside it (<snapshot>-ple-int8/). SKIP 77 without the data.
set(B70_Q4EXP_SYNTH_DIR "${B70_ORACLE}/flash-next/synth" CACHE PATH
  "Spec 21-QFN-v1b's synthetic checkpoints: <dir>/{ours,intel}/ckpt and ckpt-ple-int8 (validation: r31.synth)")
set(B70_Q4EXP_INTEL_SNAPSHOT "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound" CACHE STRING
  "Intel's Qwen3.8-Flash-Next W4A16 AutoRound checkpoint (181 GB; its PLE file: r31.ple_convert)")
# The host half over the synthetic files themselves (make_synth.py's writer against the C++ reader; no device,
# minutes: the MTP head's RTN dominates) - label checkpoint, so the host checks leave it to the validation stage r31.load.
add_executable(qwen4exp_synth_host_test loader/qwen4exp_synth_host_test.cc)
target_include_directories(qwen4exp_synth_host_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen4exp_synth_host_test PRIVATE b70_qwen4exp_loader)
add_test(NAME qwen4exp_synth_host_test COMMAND qwen4exp_synth_host_test ${B70_Q4EXP_SYNTH_DIR})
set_tests_properties(qwen4exp_synth_host_test PROPERTIES LABELS "checkpoint;qwen4exp" SKIP_RETURN_CODE 77 TIMEOUT 1800)
add_executable(qwen4exp_load_checkpoint_test loader/qwen4exp_load_checkpoint_test.cc)
target_include_directories(qwen4exp_load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen4exp_load_checkpoint_test PRIVATE b70_qwen4exp_loader b70_l0)
add_test(NAME qwen4exp_load_synth_ours_test COMMAND qwen4exp_load_checkpoint_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt 0 mtp)
add_test(NAME qwen4exp_load_synth_intel_test COMMAND qwen4exp_load_checkpoint_test ${B70_Q4EXP_SYNTH_DIR}/intel/ckpt 0 mtp)
add_test(NAME qwen4exp_load_intel_layers_test COMMAND qwen4exp_load_checkpoint_test ${B70_Q4EXP_INTEL_SNAPSHOT} 18)
set_tests_properties(qwen4exp_load_synth_ours_test qwen4exp_load_synth_intel_test qwen4exp_load_intel_layers_test
  PROPERTIES LABELS "checkpoint;qwen4exp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
# ==== Spec 21-QFN-v1b: Qwen3.8-Flash-Next descriptor and loader ==== (end) ====

# ==== Spec 21-QFN-v1c: Qwen3.8-Flash-Next decode (label qwen4exp) ==== (begin) ====
# One delimited block for every 21-QFN-v1c registration, after 21-QFN-v1b's. Written blind on the Mac (validation queue row 32): host
# tests first (the host checks run them), then the card's (GPU only); the checkpoint ones SKIP (77) until the
# synthetic golden sets (tools/oracle/qwen4exp_ref.py synth) or Intel's checkpoint and 21-QFN-v1a's sets exist.
# Task 1: the kernels' host twin (tests/kernels/qwen4exp_ref.h) against qwen4exp_ref.py's restated ops and
# transformers' modules (tests/kernels/qwen4exp_fixture.h) and the semantics' properties. Host only; the loader's
# host half for the RoPE table (q4_rope_table) and the PLE hash.
add_executable(qwen4exp_ref_test kernels/qwen4exp_ref_test.cc)
target_include_directories(qwen4exp_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen4exp_ref_test PRIVATE b70_qwen4exp_loader)
add_test(NAME qwen4exp_ref_test COMMAND qwen4exp_ref_test)
set_tests_properties(qwen4exp_ref_test PROPERTIES LABELS qwen4exp TIMEOUT 600)
add_executable(qwen4exp_f32_judge_test runtime/qwen4exp_f32_judge_test.cc)
target_include_directories(qwen4exp_f32_judge_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME qwen4exp_f32_judge_test COMMAND qwen4exp_f32_judge_test)
set_tests_properties(qwen4exp_f32_judge_test PROPERTIES LABELS qwen4exp)
if(B70_Q4EXP)
  # Task 2: every binary a Qwen3.8-Flash-Next decode list binds, any dense arm / head / attention / PLE scale form
  # (src/kernels/CMakeLists.txt's spec 21-QFN-v1c block; the reused binaries other blocks build - Qwen3.8's gdn_step_M1 and
  # argmax, Kolibri's 6144 x 2560 int4 and bf16 - are bound by name and listed too).
  set(B70_Q4EXP_DECODE_KERNELS
    kernel_embed_gather_M1_D2560 kernel_q4_hc_combine_norm_M1_E kernel_q4_hc_combine_norm_M1_S1
    kernel_q4_hc_combine_norm_M1_S8 kernel_q4_hc_combine_norm_M1_Y kernel_q4_hc_combine_norm_M1_Y_NN
    kernel_q4_hc_combine_norm_M1_X kernel_q4_hc_up_mix_M1_I_B kernel_q4_hc_up_mix_M1_B
    kernel_gemv_bf16_M1_K10240_N336_C16_S16 kernel_gemv_bf16_M1_K10240_N320_C16_S16
    kernel_q4_hc_up_mix_M1_I_Q8_B kernel_q4_hc_up_mix_M1_Q8_B kernel_q4_hc_down_i8_M1_N336_KS32
    kernel_q4_hc_down_i8_M1_N320_KS32
    kernel_gemv_M1_K2560_N16384_S1_L0 kernel_gemv_bf16_M1_K2560_N16384 kernel_gemv_bf16_M1_K2560_N128_C16_S16
    kernel_gdn_step_M1 kernel_prep_gated_head_M1_SIG kernel_gemv_M1_K6144_N2560_S8_L0 kernel_gemv_bf16_M1_K6144_N2560
    kernel_gemv_M1_K2560_N13312_S2_L0 kernel_gemv_bf16_M1_K2560_N13312 kernel_gemv_bf16_M1_K2560_N640_C16_S16
    kernel_attn_prep_M1_Q24KV2 kernel_attn_prep_M1_Q24KV2_S1 kernel_q4_qsa_M1_T512_W1024_I4H kernel_q4_qsa_attn_M1_T32
    kernel_q4_qsa_attn_eager_M1 kernel_gemv_bf16_M1_K2560_N528_C16_S16 kernel_q4_route_M1_E512_T10_N528_L256
    kernel_q4_moe_M1_E512_T10_D2560_I640_SH4 kernel_q4_moe_M1_E512_T10_D2560_I640_SHB
    kernel_q4_moe_M1_E512_T10_D2560_I640_SH4_TBL kernel_q4_moe_M1_E512_T10_D2560_I640_SHB_TBL kernel_q4_expert_check
    kernel_q4_ple_gather_M1_BF16
    kernel_q4_ple_gather_M1_F32 kernel_q4_ple_block_M1 kernel_gemv_bf16_M1_K2560_N12800
    kernel_gemv_bf16_M1_K2560_N248320 kernel_gemv_i8w_M1_K2560_N248320 kernel_argmax_stage1_M1 kernel_argmax_stage2)
  # Host: the names the capture binds (kernels::qwen4exp::decode_variants) are exactly this list.
  add_executable(qwen4exp_variant_names_test kernels/qwen4exp_variant_names_test.cc)
  target_include_directories(qwen4exp_variant_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_variant_names_test PRIVATE b70_qwen4exp_model)
  b70_target_kernel_dir(qwen4exp_variant_names_test)
  add_test(NAME qwen4exp_variant_names_test COMMAND qwen4exp_variant_names_test ${B70_Q4EXP_DECODE_KERNELS})
  set_tests_properties(qwen4exp_variant_names_test PROPERTIES LABELS qwen4exp)
  # F2 on the card: every binary of the block at the real shapes against qwen4exp_ref.h (no checkpoint);
  # --bench-attn (decision 10's evidence: attn_v2 at 24 / 2 against the sparse kernel) and --ple-rate (Task 5) opt-in.
  add_executable(qwen4exp_kernels_test kernels/qwen4exp_kernels_test.cc)
  target_include_directories(qwen4exp_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_kernels_test PRIVATE b70_l0 b70_qwen4exp_loader)
  b70_target_kernel_dir(qwen4exp_kernels_test)
  add_dependencies(qwen4exp_kernels_test ${B70_Q4EXP_DECODE_KERNELS} kernel_attn_v2_M1_T${ATTN_V2_TGT}_Q24KV2
    kernel_q4_hc_up_mix_M2048_I_D512_R4C64_Q8 kernel_q4_hc_slab_i8_N336    # PERF-66's prefill forms (test 8)
    kernel_q4_qsa_M1_T512_W1024 kernel_q4_qsa_M2048_T512_W1024_PF)   # test 9, --bench-qsa-i4: the bf16 reference forms
  foreach(MM 1 2 3 4)   # PERF-106's --bench-hc8: lead 4's pair (the down _A, the up-mix _A; _R<M>C16 at M >= 2)
    if(MM GREATER 1)
      add_dependencies(qwen4exp_kernels_test kernel_q4_hc_up_mix_M${MM}_I_R${MM}C16_Q8_B_A)
    else()
      add_dependencies(qwen4exp_kernels_test kernel_q4_hc_up_mix_M1_I_Q8_B_A)
    endif()
    add_dependencies(qwen4exp_kernels_test kernel_q4_hc_down_i8_M${MM}_N336_KS32_A)
  endforeach()
  add_test(NAME qwen4exp_kernels_test COMMAND qwen4exp_kernels_test)
  set_tests_properties(qwen4exp_kernels_test PROPERTIES LABELS qwen4exp SKIP_RETURN_CODE 77 TIMEOUT 1800)
  # Task 3: the runtime on the card - F4 on decode (plan == allocation, two runs bitwise, the injected capture fed
  # the free run's lists = the free run) on both synthetic checkpoints past the selection's first cut (2200 ids of
  # q4exp_4k); F3 (tokens tie-aware, routing, gate S, PLE ids; free, int8 head, eager, injected) on both synthetic
  # golden sets (qwen4exp_oracle.sh synth); the development mode on Intel's checkpoint at --layers 18 against 21-QFN-v1a's
  # full and truncated sets. SKIP 77 without the data.
  set(B70_Q4EXP_ORACLE_DIR "${B70_ORACLE}/flash-next/golden"
    CACHE PATH "21-QFN-v1a's golden sets on Intel's checkpoint (qwen4exp_oracle.sh intel): <prompt>.ids + .golden.safetensors")
  foreach(t qwen4exp_decode_test qwen4exp_decode_ref_probe qwen4exp_golden_test qwen4exp_partial_test)
    if(t STREQUAL "qwen4exp_decode_ref_probe")
      add_executable(${t} ${CMAKE_SOURCE_DIR}/tools/probe/qwen4exp_decode_ref.cc)
    elseif(t STREQUAL "qwen4exp_decode_test")
      add_executable(${t} runtime/${t}.cc)
    else()
      add_executable(${t} golden/${t}.cc)
    endif()
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
      ${CMAKE_SOURCE_DIR}/tests/runtime)
    target_link_libraries(${t} PRIVATE b70_qwen4exp_runtime b70_l0)
    b70_target_kernel_dir(${t})
    add_dependencies(${t} ${B70_Q4EXP_DECODE_KERNELS})
  endforeach()
  set(_q4s ${B70_Q4EXP_SYNTH_DIR})
  set(_q4tests "")
  foreach(f ours intel)
    add_test(NAME qwen4exp_decode_synth_${f}_test COMMAND qwen4exp_decode_test ${_q4s}/${f}/ckpt
      ${CMAKE_SOURCE_DIR}/tests/golden/prompts/q4exp_4k.ids 2200)
    add_test(NAME qwen4exp_golden_synth_${f}_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f})
    add_test(NAME qwen4exp_golden_synth_${f}_i8head_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f} int8)
    add_test(NAME qwen4exp_golden_synth_${f}_eager_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f})
    set_tests_properties(qwen4exp_golden_synth_${f}_eager_test PROPERTIES ENVIRONMENT "B70_Q4_ATTN=eager")
    add_test(NAME qwen4exp_golden_synth_${f}_inject_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f} inject)
    list(APPEND _q4tests qwen4exp_decode_synth_${f}_test qwen4exp_golden_synth_${f}_test qwen4exp_golden_synth_${f}_i8head_test
      qwen4exp_golden_synth_${f}_eager_test qwen4exp_golden_synth_${f}_inject_test)
  endforeach()
  add_test(NAME qwen4exp_partial_test COMMAND qwen4exp_partial_test ${B70_Q4EXP_INTEL_SNAPSHOT} ${B70_Q4EXP_ORACLE_DIR}
    ${B70_Q4EXP_ORACLE_DIR}-L18 18 int8)
  set_tests_properties(${_q4tests} qwen4exp_partial_test
    PROPERTIES LABELS "checkpoint;golden;qwen4exp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  # Task 4: two cards on spec 16b's pieces - --parallel 1 against --parallel pp (splits 2 and 1) bitwise on the
  # ours synthetic, Intel's at the planner's N for two cards (layers_that_fit) split s against s + 1, P4's lost
  # hand-off, the landing layout. SKIP 77 with one GPU or no data.
  add_executable(qwen4exp_pp_test runtime/qwen4exp_pp_test.cc)
  target_include_directories(qwen4exp_pp_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(qwen4exp_pp_test PRIVATE b70_qwen4exp_runtime b70_l0)
  add_dependencies(qwen4exp_pp_test ${B70_Q4EXP_DECODE_KERNELS})
  add_test(NAME qwen4exp_pp_test COMMAND qwen4exp_pp_test ${_q4s}/ours/ckpt ${B70_Q4EXP_INTEL_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/q4exp_short.ids)
  set_tests_properties(qwen4exp_pp_test PROPERTIES LABELS "checkpoint;qwen4exp;pp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  # Plan 21-QFN-v2b Task 2: tp against pp at --layers 18 on our RTN export, every expert resident (B70_Q4EXP_RTN_SNAPSHOT).
  if(B70_TP)
    set(B70_Q4EXP_RTN_SNAPSHOT "" CACHE STRING "Our RTN int4 g64 export of Qwen3.8-Flash-Next (plan 21-QFN-v2b's tp test)")
    add_executable(qwen4exp_tp_test runtime/qwen4exp_tp_test.cc)
    target_include_directories(qwen4exp_tp_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
      ${CMAKE_SOURCE_DIR}/tests/runtime)
    target_link_libraries(qwen4exp_tp_test PRIVATE b70_qwen4exp_prefill b70_l0)   # forcepf: prefills
    add_dependencies(qwen4exp_tp_test ${B70_Q4EXP_DECODE_KERNELS} kernel_gemv_M1_K2560_N8192_S1_L0_KS4
      kernel_gemv_M1_K2560_N6656_S2_L0 kernel_gemv_M1_K3072_N2560_S4_L0_TP kernel_gemv_i8w_M1_K2560_N124160
      kernel_prep_gated_head_M1_GK8V24_SIG kernel_gdn_step_M1_GK8V24 kernel_attn_prep_M1_Q12KV1
      kernel_q4_qsa_attn_M1_T32_Q12KV1 kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TP_R0
      kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TP_R1 kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TBL_TP_R0
      kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TBL_TP_R1
      kernel_tp_cross_M1_K2560_G20_S4_X_R0_sum_W64 kernel_tp_cross_M1_K2560_G20_S4_X_R1_sum_W64
      kernel_tp_argmax_M1_N124160_U124160_B0 kernel_tp_argmax_M1_N124160_U123917_B124160)
    foreach(mode pp tp)   # pp writes the reference rows tp reads: run in this order (ctest -R qwen4exp_tp_)
      add_test(NAME qwen4exp_tp_${mode}_test COMMAND qwen4exp_tp_test ${B70_Q4EXP_RTN_SNAPSHOT}
        ${CMAKE_SOURCE_DIR}/tests/golden/prompts/q4exp_4k.ids ${mode} ${CMAKE_BINARY_DIR}/qwen4exp_tp_rows.f32)
      set_tests_properties(qwen4exp_tp_${mode}_test PROPERTIES LABELS "checkpoint;qwen4exp;tp" SKIP_RETURN_CODE 77
        TIMEOUT 3600)
    endforeach()
    set_tests_properties(qwen4exp_tp_tp_test PROPERTIES DEPENDS qwen4exp_tp_pp_test)
  endif()
endif()
# b70-decode's model_type dispatch and this family's refusals, before the device: tests/model/qwen4exp/ holds only a
# model_type stub config.json (spec 21-QFN-v1 §13 departure 1: the checkpoint's own config is not vendored), which is all the
# dispatch and the refusals read.
set(_q4_cfg ${CMAKE_CURRENT_SOURCE_DIR}/model/qwen4exp)
b70_cli_reject(cli_reject_qwen4exp_mtp ${_q4_cfg} --ids ${B70_OK_IDS} --n 4 --spec mtp EXPECT "its MTP head is served by b70-serve")
# (spec 21-QFN-v1e removed cli_reject_qwen4exp_serve: b70-serve serves this family now - its refusals are 21-QFN-v1e's block's)
# ==== Spec 21-QFN-v1c: Qwen3.8-Flash-Next decode ==== (end) ====

# ==== Spec 21-QFN-v1d: Qwen3.8-Flash-Next prefill (labels qwen4exp;prefill) ==== (begin) ====
# One delimited block for every 21-QFN-v1d registration, after 21-QFN-v1c's. Written blind on the Mac (validation queue row 33): host
# tests first (the host checks run them), then the card's (GPU only); the checkpoint ones SKIP (77) until the
# synthetic golden sets (qwen4exp_oracle.sh synth) or Intel's checkpoint and 21-QFN-v1a's sets exist.
# Task 1: the prefill kernels' host twin (tests/kernels/qwen4exp_pf_ref.h) - the 512-expert sort and its adversaries,
# the combine == 21-QFN-v1c's decode chain, dense / sparse rows, the sparse attention over row lists in fp64 and eager, the
# indexer and PLE over a chunk == decode's M = 1 chains with their rings. Headers only (+ the PLE hash header).
add_executable(qwen4exp_pf_ref_test kernels/qwen4exp_pf_ref_test.cc)
target_include_directories(qwen4exp_pf_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen4exp_pf_ref_test PRIVATE b70_qwen4exp_loader)
add_test(NAME qwen4exp_pf_ref_test COMMAND qwen4exp_pf_ref_test)
set_tests_properties(qwen4exp_pf_ref_test PROPERTIES LABELS "qwen4exp;prefill" TIMEOUT 900)
if(B70_Q4EXP)
  # Task 2: every binary a prefill chunk binds, both dense arms / attention forms / PLE scale forms
  # (src/kernels/CMakeLists.txt's spec 21-QFN-v1d block; the reused binaries other blocks build - pf_gemm_T0, Qwen3.8's
  # pf_gdn_*, 21-QFN-v1c's q4_route_M1, Kolibri's 6144 x 2560 int4 slab - are bound by name and listed too).
  set(B70_Q4EXP_PREFILL_KERNELS
    kernel_pf_embed_gather_D2560 kernel_q4_hc_combine_norm_M2048_E kernel_q4_hc_combine_norm_M2048_S1
    kernel_q4_hc_combine_norm_M2048_Y kernel_q4_hc_combine_norm_M2048_X kernel_q4_hc_combine_norm_M2048_Y_NN
    kernel_q4_pf_bf16_slab_K10240_N336 kernel_pf_gemm_T0 kernel_q4_hc_up_mix_M2048_I_D512_R4C64 kernel_pf_ab_proj_D2560
    kernel_q4_hc_up_mix_M2048_I_D512_R4C64_Q8 kernel_q4_hc_slab_i8_N336
    kernel_pf_gdn_conv kernel_pf_gdn_intra kernel_pf_gdn_scan_OB kernel_pf_gdn_front_PL kernel_pf_gdn_front_PL_PB
    kernel_pf_gemm_T0_B16 kernel_pf_gated_head_SIG_OB_PB
    kernel_pf_route_slab_K2560_N528_P768 kernel_pf_route_slab_K2560_N128_P256
    kernel_q4_route_M1_E512_T10_N528_L256_P768 kernel_pf_gdn_conv_AB256
    kernel_pf_moe_sort_mw_E512_T10 kernel_pf_moe_q4gemm_K2560_N1280_SILU_SHX kernel_pf_moe_q4gemm_K640_N2560
    kernel_pf_moe_q4gemm_K2560_N1280_SILU_TBL_SHX kernel_pf_moe_q4gemm_K640_N2560_TBL
    kernel_pf_gated_head_SIG_OB kernel_q4_pf_bf16_slab_K2560_N640
    kernel_pf_attn_prep_q16_Q24KV2 kernel_q4_qsa_M2048_T512_W1024_PF_I4H kernel_pf_flash_attn_Q24KV2
    kernel_q4_pf_sparse_attn_Q24KV2 kernel_q4_pf_sparse_attn_Q24KV2_EAGER kernel_pf_attn_Q24KV2
    kernel_pf_moe_router_K2560_N528 kernel_q4_route_M1_E512_T10_N528_L256 kernel_q4_pf_moe_E512_T10_D2560_I640_L256
    kernel_pf_moe_gemm_K2560_N1280_SILU kernel_pf_moe_gemm_K640_N2560 kernel_q4_ple_gather_M2048_BF16_PF
    kernel_q4_ple_gather_M2048_F32_PF kernel_q4_pf_bf16_slab_K2560_N12800 kernel_q4_pf_ple_C2048
    kernel_pf_dequant_slab_K2560_N16384_L0 kernel_pf_dequant_slab_K2560_N13312_L0 kernel_pf_int4_slab_K6144_N2560
    kernel_q4_pf_bf16_slab_K2560_N16384 kernel_q4_pf_bf16_slab_K2560_N13312 kernel_q4_pf_bf16_slab_K6144_N2560
    # PERF-36: the 4096-row chunk's builds (M = 4096, the block-store slabs, pf_gemm_T0_R)
    kernel_q4_hc_combine_norm_M4096_E kernel_q4_hc_combine_norm_M4096_S1 kernel_q4_hc_combine_norm_M4096_Y
    kernel_q4_hc_combine_norm_M4096_X kernel_q4_hc_combine_norm_M4096_Y_NN kernel_q4_hc_up_mix_M4096_I_D512_R4C64
    kernel_q4_hc_up_mix_M4096_I_D512_R4C64_Q8
    kernel_q4_qsa_M4096_T512_W1024_PF_I4H kernel_q4_ple_gather_M4096_BF16_PF kernel_q4_ple_gather_M4096_F32_PF
    kernel_q4_pf_ple_C4096 kernel_pf_dequant_slab_K2560_N16384_L0_BS kernel_pf_dequant_slab_K2560_N13312_L0_BS
    kernel_pf_gemm_T0_R kernel_pf_gemm_T0_B16_R)
  # Host: the names the walk binds (kernels::qwen4exp::prefill_variants) are exactly this list.
  add_executable(qwen4exp_pf_variant_names_test kernels/qwen4exp_pf_variant_names_test.cc)
  target_include_directories(qwen4exp_pf_variant_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_pf_variant_names_test PRIVATE b70_qwen4exp_model)
  b70_target_kernel_dir(qwen4exp_pf_variant_names_test)
  add_test(NAME qwen4exp_pf_variant_names_test COMMAND qwen4exp_pf_variant_names_test ${B70_Q4EXP_PREFILL_KERNELS})
  set_tests_properties(qwen4exp_pf_variant_names_test PROPERTIES LABELS "qwen4exp;prefill")
  # F2 for the prefill on the card, no checkpoint: every prefill kernel at the real shapes against qwen4exp_pf_ref.h,
  # grouped == dense (pf_gemm_T0_SILU / pf_gemm_T0), the dense and the sparse flash against fp64, the eager chain.
  add_executable(qwen4exp_pf_kernels_test kernels/qwen4exp_pf_kernels_test.cc)
  target_include_directories(qwen4exp_pf_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_pf_kernels_test PRIVATE b70_prefill_host b70_qwen4exp_loader b70_l0)
  b70_target_kernel_dir(qwen4exp_pf_kernels_test)
  add_dependencies(qwen4exp_pf_kernels_test ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_DECODE_KERNELS} kernel_pf_gemm_T0_SILU
    kernel_q4_qsa_M2048_T512_W1024_PF)   # the bf16 reference form
  add_test(NAME qwen4exp_pf_kernels_test COMMAND qwen4exp_pf_kernels_test)
  set_tests_properties(qwen4exp_pf_kernels_test PROPERTIES LABELS "qwen4exp;prefill" SKIP_RETURN_CODE 77 TIMEOUT 1800)
  # Plan 21-QFN-v3a Task 1: the int8 KV cache's writers (bitwise kv8.h) and readers (against an fp64 twin on the same
  # codes) at 24 / 2 heads - tests/kernels/qwen4exp_kv8_test.cc (GPU only).
  if(B70_KV8)
    add_executable(qwen4exp_kv8_test kernels/qwen4exp_kv8_test.cc)
    target_include_directories(qwen4exp_kv8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(qwen4exp_kv8_test PRIVATE b70_l0)
    b70_target_kernel_dir(qwen4exp_kv8_test)
    add_dependencies(qwen4exp_kv8_test kernel_attn_prep_M1_Q24KV2 kernel_attn_prep_M1_Q24KV2_KV8
      kernel_pf_attn_prep_q16_Q24KV2 kernel_pf_attn_prep_kv8_Q24KV2 kernel_pf_attn_prep_kv8_Q24KV2_KVO
      kernel_q4_qsa_attn_M1_T32_KV8 kernel_pf_flash_attn_kv8_Q24KV2 kernel_q4_pf_sparse_attn_Q24KV2_KV8
      kernel_pf_attn_gate_kv8_Q24KV2)
    add_test(NAME qwen4exp_kv8_test COMMAND qwen4exp_kv8_test)
    set_tests_properties(qwen4exp_kv8_test PROPERTIES LABELS "qwen4exp;prefill" SKIP_RETURN_CODE 77 TIMEOUT 1800)
  endif()
  # Task 3: the prefill walk on the card (GPU only; SKIP 77 without the synthetic checkpoints / Intel's).
  #   qwen4exp_prefill_test: the walk's launches and the plan, F4 (two prefills, recorded == replayed, chunks of 64 ==
  #       2048 bitwise, 16 printed), Review Focus 5 (the state, routes and selections against decode's fill, tokens),
  #       the injected run chunked == whole; its `split` mode (multiples of 64 bitwise, the rest within bars), its `pp`
  #       mode (one card against two, bitwise; P4), its `gdn` mode (gdn_chunk_q4 against decode's gdn_step, no
  #       checkpoint);
  #   qwen4exp_golden_test's `prefill` mode: F3 on prefill (both synthetics; Intel's --layers 18 after 21-QFN-v1a's data).
  add_executable(qwen4exp_prefill_test runtime/qwen4exp_prefill_test.cc)
  b70_target_kernel_dir(qwen4exp_prefill_test)
  target_include_directories(qwen4exp_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(qwen4exp_prefill_test PRIVATE b70_qwen4exp_prefill b70_l0)
  add_dependencies(qwen4exp_prefill_test ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_DECODE_KERNELS})
  add_executable(qwen4exp_route_capture ${CMAKE_SOURCE_DIR}/tools/probe/qwen4exp_route_capture.cc)
  b70_target_kernel_dir(qwen4exp_route_capture)
  target_include_directories(qwen4exp_route_capture PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_route_capture PRIVATE b70_qwen4exp_prefill b70_l0)
  add_dependencies(qwen4exp_route_capture ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_DECODE_KERNELS})
  set(_q4s ${B70_Q4EXP_SYNTH_DIR})
  set(_q4ids ${CMAKE_SOURCE_DIR}/tests/golden/prompts/q4exp_4k.ids)
  set(_q4pf "")
  foreach(f ours intel)
    add_test(NAME qwen4exp_prefill_synth_${f}_test COMMAND qwen4exp_prefill_test ${_q4s}/${f}/ckpt ${_q4ids} all)
    list(APPEND _q4pf qwen4exp_prefill_synth_${f}_test)
  endforeach()
  add_test(NAME qwen4exp_prefill_synth_eager_test COMMAND qwen4exp_prefill_test ${_q4s}/ours/ckpt ${_q4ids} all)
  set_tests_properties(qwen4exp_prefill_synth_eager_test PROPERTIES ENVIRONMENT "B70_Q4_ATTN=eager")
  add_test(NAME prefill_split_qwen4exp_test COMMAND qwen4exp_prefill_test ${_q4s}/ours/ckpt ${_q4ids} split)
  # PERF-36: the 4096-row chunk against 2048-row chunks, bitwise (both dense arms)
  foreach(f ours intel)
    add_test(NAME qwen4exp_prefill_wide_${f}_test COMMAND qwen4exp_prefill_test ${_q4s}/${f}/ckpt ${_q4ids} wide)
    list(APPEND _q4pf qwen4exp_prefill_wide_${f}_test)
  endforeach()
  add_test(NAME qwen4exp_prefill_gdn_test COMMAND qwen4exp_prefill_test - - gdn)
  add_test(NAME qwen4exp_prefill_gdn_pb_test COMMAND qwen4exp_prefill_test - - gdn-pb)
  set_tests_properties(${_q4pf} qwen4exp_prefill_synth_eager_test prefill_split_qwen4exp_test
    PROPERTIES LABELS "checkpoint;qwen4exp;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  set_tests_properties(qwen4exp_prefill_gdn_test qwen4exp_prefill_gdn_pb_test PROPERTIES LABELS "qwen4exp;prefill"
    SKIP_RETURN_CODE 77 TIMEOUT 1800)
  # two cards (16b's pieces): one card against --parallel pp --pipeline-split 2 bitwise on the ours synthetic, the split test
  # across two cards; Intel's truncated model on two cards at the planner's N with prefill (37 at 32768: 38 does not fit
  # beside the prefill scratch)
  add_test(NAME qwen4exp_prefill_pp_test COMMAND qwen4exp_prefill_test ${_q4s}/ours/ckpt ${_q4ids} pp pp2:2)
  # plan 21-QFN-v3a: the int8 KV cache - F4, chunking, prefill against decode (all), the wide chunk, pp against one card
  add_test(NAME qwen4exp_prefill_synth_kv8_test COMMAND qwen4exp_prefill_test ${_q4s}/ours/ckpt ${_q4ids} all kv8)
  add_test(NAME qwen4exp_prefill_wide_kv8_test COMMAND qwen4exp_prefill_test ${_q4s}/ours/ckpt ${_q4ids} wide kv8)
  add_test(NAME qwen4exp_prefill_pp_kv8_test COMMAND qwen4exp_prefill_test ${_q4s}/ours/ckpt ${_q4ids} pp pp2:2 kv8)
  set_tests_properties(qwen4exp_prefill_synth_kv8_test qwen4exp_prefill_wide_kv8_test qwen4exp_prefill_pp_kv8_test
    PROPERTIES LABELS "checkpoint;qwen4exp;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  add_test(NAME prefill_split_qwen4exp_pp_test COMMAND qwen4exp_prefill_test ${_q4s}/ours/ckpt ${_q4ids} split pp2:2)
  foreach(n 18 37)
    set(_q4judge${n} "")
    # Relative counts require both complete routing audits; legacy single-reference dumps lack them.
    set(_q4ref "${B70_ORACLE}/flash-next/f32-aligned-L${n}/q4exp_4k.state.safetensors")
    if(EXISTS "${_q4ref}")
      set(_q4judge${n} "judge:${_q4ref}")
    endif()
  endforeach()
  add_test(NAME qwen4exp_prefill_intel_test COMMAND qwen4exp_prefill_test ${B70_Q4EXP_INTEL_SNAPSHOT} ${_q4ids} all
    layers:18 int8 ${_q4judge18})
  set_tests_properties(qwen4exp_prefill_intel_test
    PROPERTIES LABELS "checkpoint;qwen4exp;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  add_test(NAME qwen4exp_prefill_intel_pp_test COMMAND qwen4exp_prefill_test ${B70_Q4EXP_INTEL_SNAPSHOT} ${_q4ids} all pp2
    layers:37 int8 ${_q4judge37})
  set_tests_properties(qwen4exp_prefill_pp_test prefill_split_qwen4exp_pp_test qwen4exp_prefill_intel_pp_test
    PROPERTIES LABELS "checkpoint;qwen4exp;prefill;pp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  # F3 on prefill: 21-QFN-v1c's golden test in its prefill mode (one chunk, chunks of 16, the int8 head, injected, eager) on
  # both synthetic golden sets; Intel's --layers 18 after 21-QFN-v1a's sets exist.
  target_link_libraries(qwen4exp_golden_test PRIVATE b70_qwen4exp_prefill)
  add_dependencies(qwen4exp_golden_test ${B70_Q4EXP_PREFILL_KERNELS})
  set(_q4g "")
  foreach(f ours intel)
    add_test(NAME qwen4exp_golden_prefill_synth_${f}_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f} prefill)
    add_test(NAME qwen4exp_golden_prefill_synth_${f}_c16_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f}
      prefill:16)
    add_test(NAME qwen4exp_golden_prefill_synth_${f}_i8head_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f}
      int8 prefill)
    add_test(NAME qwen4exp_golden_prefill_synth_${f}_inject_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f}
      inject prefill)
    add_test(NAME qwen4exp_golden_prefill_synth_${f}_eager_test COMMAND qwen4exp_golden_test ${_q4s}/${f}/ckpt ${_q4s}/${f}
      prefill)
    set_tests_properties(qwen4exp_golden_prefill_synth_${f}_eager_test PROPERTIES ENVIRONMENT "B70_Q4_ATTN=eager")
    list(APPEND _q4g qwen4exp_golden_prefill_synth_${f}_test qwen4exp_golden_prefill_synth_${f}_c16_test
      qwen4exp_golden_prefill_synth_${f}_i8head_test qwen4exp_golden_prefill_synth_${f}_inject_test
      qwen4exp_golden_prefill_synth_${f}_eager_test)
  endforeach()
  add_test(NAME qwen4exp_golden_prefill_synth_pp_test COMMAND qwen4exp_golden_test ${_q4s}/ours/ckpt ${_q4s}/ours pp2:2
    prefill)
  add_test(NAME qwen4exp_golden_prefill_intel_test COMMAND qwen4exp_golden_test ${B70_Q4EXP_INTEL_SNAPSHOT}
    ${B70_Q4EXP_ORACLE_DIR}-L18 layers:18 int8 prefill)
  list(APPEND _q4g qwen4exp_golden_prefill_synth_pp_test qwen4exp_golden_prefill_intel_test)
  set_tests_properties(${_q4g} PROPERTIES LABELS "checkpoint;golden;qwen4exp;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# Task 3: b70-decode's Qwen3.8-Flash-Next prefill backends and chunk, before the device (tests/model/qwen4exp: the
# model_type stub). 21-QFN-v1c's cli_reject_qwen4exp_prefill (--prefill naming spec 21-QFN-v1d) is gone: this family prefills now.
set(_q4_cfg ${CMAKE_CURRENT_SOURCE_DIR}/model/qwen4exp)
b70_cli_reject(cli_reject_qwen4exp_prefill_int8 ${_q4_cfg} --layers 4 --bench --prefill-length 64 --prefill-backend l0-int8
  EXPECT "1024-k Hadamard blocks")
# ==== Spec 21-QFN-v1d: Qwen3.8-Flash-Next prefill ==== (end) ====
# Plan 21-QFN-v2b Task 4: the tensor-parallel prefill against pp's at --layers 18 on our RTN export, every expert resident
# (B70_Q4EXP_RTN_SNAPSHOT): pp writes its prefill's rows, tp reads them (ctest -R qwen4exp_tp_prefill_, in this order).
if(B70_Q4EXP AND B70_TP)
  set(B70_Q4EXP_TP_KERNELS kernel_gemv_M1_K2560_N8192_S1_L0_KS4 kernel_gemv_M1_K2560_N6656_S2_L0
    kernel_gemv_M1_K3072_N2560_S4_L0_TP kernel_gemv_i8w_M1_K2560_N124160 kernel_prep_gated_head_M1_GK8V24_SIG
    kernel_gdn_step_M1_GK8V24 kernel_attn_prep_M1_Q12KV1 kernel_q4_qsa_attn_M1_T32_Q12KV1
    kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TP_R0 kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TP_R1
    kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TBL_TP_R0 kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TBL_TP_R1
    kernel_tp_cross_M1_K2560_G20_S4_X_R0_sum_W64 kernel_tp_cross_M1_K2560_G20_S4_X_R1_sum_W64
    kernel_tp_argmax_M1_N124160_U124160_B0 kernel_tp_argmax_M1_N124160_U123917_B124160)
  set(B70_Q4EXP_TP_PREFILL_KERNELS kernel_pf_dequant_slab_K2560_N8192_L0 kernel_pf_int4_slab_K2560_N6656
    kernel_pf_int4_slab_K3072_N2560 kernel_q4_pf_bf16_slab_K2560_N8192 kernel_q4_pf_bf16_slab_K2560_N6656
    kernel_q4_pf_bf16_slab_K3072_N2560 kernel_pf_gdn_front_GK8V24_PL kernel_pf_gdn_conv_GK8V24_AB256
    kernel_pf_gated_head_GK8V24_SIG_OB kernel_pf_gdn_conv_GK8V24 kernel_pf_gdn_intra_GK8V24 kernel_pf_gdn_scan_GK8V24_OB
    kernel_pf_gdn_front_GK8V24_PL_PB kernel_pf_gated_head_GK8V24_SIG_OB_PB
    kernel_pf_attn_prep_q16_Q12KV1 kernel_pf_attn_Q12KV1 kernel_pf_flash_attn_Q12KV1 kernel_q4_pf_sparse_attn_Q12KV1
    kernel_pf_moe_q4gemm_K2560_N640_SILU_SHX kernel_pf_moe_q4gemm_K2560_N640_SILU_TBL_SHX kernel_pf_moe_q4gemm_K320_N2560_F32
    kernel_pf_moe_q4gemm_K320_N2560_F32_TBL kernel_q4_pf_moe_combine_f32_E512_T10_D2560
    kernel_q4_hc_combine_norm_M2048_S2 kernel_q4_hc_combine_norm_M2048_S2_NN kernel_q4_hc_combine_norm_M1_S2
    kernel_q4_hc_combine_norm_M4096_S2 kernel_q4_hc_combine_norm_M4096_S2_NN kernel_pf_dequant_slab_K2560_N8192_L0_BS)
  add_executable(qwen4exp_tp_prefill_test runtime/qwen4exp_tp_prefill_test.cc)
  b70_target_kernel_dir(qwen4exp_tp_prefill_test)
  target_include_directories(qwen4exp_tp_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(qwen4exp_tp_prefill_test PRIVATE b70_qwen4exp_prefill b70_l0)
  add_dependencies(qwen4exp_tp_prefill_test ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_DECODE_KERNELS}
    ${B70_Q4EXP_TP_KERNELS} ${B70_Q4EXP_TP_PREFILL_KERNELS})
  if(TARGET qwen4exp_tp_test)   # its forcepf: rows prefill
    add_dependencies(qwen4exp_tp_test ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_TP_PREFILL_KERNELS})
  endif()
  foreach(mode pp tp)
    add_test(NAME qwen4exp_tp_prefill_${mode}_test COMMAND qwen4exp_tp_prefill_test ${B70_Q4EXP_RTN_SNAPSHOT}
      ${CMAKE_SOURCE_DIR}/tests/golden/prompts/q4exp_4k.ids ${mode} ${CMAKE_BINARY_DIR}/qwen4exp_tp_pf_rows.f32)
    set_tests_properties(qwen4exp_tp_prefill_${mode}_test PROPERTIES LABELS "checkpoint;qwen4exp;tp" SKIP_RETURN_CODE 77
      TIMEOUT 3600)
  endforeach()
  set_tests_properties(qwen4exp_tp_prefill_tp_test PROPERTIES DEPENDS qwen4exp_tp_prefill_pp_test)
  # PERF-27: the MoE prompt-pass glue against the forms it replaced, no checkpoint - the prefill's one-sub-group
  # q4_route against decode's (every word, random and tied logits), q4_pf_gather_routed + the _SHX gate||up against
  # q4_pf_gather + the plain binary (one card's 2560 x 1280 and a tp card's 2560 x 640, direct and _TBL), the
  # 16-byte combines against the one-column ones (_C1); PERF-25: the four direct expert GEMMs against their _DB twins
  # (the register plan they replaced). `--time` adds the kernel times and the builds' spills.
  add_executable(qwen4exp_pf_glue_test kernels/qwen4exp_pf_glue_test.cc)
  target_include_directories(qwen4exp_pf_glue_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_pf_glue_test PRIVATE b70_l0 b70_qwen4exp_loader)
  b70_target_kernel_dir(qwen4exp_pf_glue_test)
  add_dependencies(qwen4exp_pf_glue_test ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_TP_PREFILL_KERNELS}
    kernel_q4_route_M1_E512_T10_N528_L256 kernel_pf_moe_q4gemm_K2560_N1280_SILU kernel_pf_moe_q4gemm_K2560_N1280_SILU_TBL
    kernel_pf_moe_q4gemm_K2560_N640_SILU kernel_pf_moe_q4gemm_K2560_N640_SILU_TBL
    kernel_q4_pf_moe_E512_T10_D2560_I640_L256_C1 kernel_q4_pf_moe_combine_f32_E512_T10_D2560_C1
    kernel_pf_moe_q4gemm_K2560_N1280_SILU_SHX_DB kernel_pf_moe_q4gemm_K640_N2560_DB
    kernel_pf_moe_q4gemm_K2560_N640_SILU_SHX_DB kernel_pf_moe_q4gemm_K320_N2560_F32_DB)
  add_test(NAME qwen4exp_pf_glue_test COMMAND qwen4exp_pf_glue_test)
  set_tests_properties(qwen4exp_pf_glue_test PROPERTIES LABELS "qwen4exp;prefill" SKIP_RETURN_CODE 77 TIMEOUT 1800)
  # PERF-20 / PERF-21: the GDN walk (gdn_chunk_q4: front _PL, scan _OB, head _SIG_OB) against the plain forms it
  # replaced, no checkpoint - state, ring and y bitwise at one card's 16 / 48 heads and a tp card's 8 / 24, and PERF-8's
  # walk on bf16 partials (_PB) against the old on fp32; `--time`
  # adds the three kernels' times old against new at C 2048 and the builds' spills.
  add_executable(qwen4exp_pf_gdn_bundle_test kernels/qwen4exp_pf_gdn_bundle_test.cc)
  target_include_directories(qwen4exp_pf_gdn_bundle_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_pf_gdn_bundle_test PRIVATE b70_qwen4exp_prefill b70_l0)
  b70_target_kernel_dir(qwen4exp_pf_gdn_bundle_test)
  add_dependencies(qwen4exp_pf_gdn_bundle_test ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_TP_PREFILL_KERNELS}
    kernel_pf_gdn_front kernel_pf_gdn_scan kernel_pf_gated_head_SIG kernel_pf_gdn_front_GK8V24 kernel_pf_gdn_scan_GK8V24
    kernel_pf_gated_head_GK8V24_SIG)
  add_test(NAME qwen4exp_pf_gdn_bundle_test COMMAND qwen4exp_pf_gdn_bundle_test)
  set_tests_properties(qwen4exp_pf_gdn_bundle_test PROPERTIES LABELS "qwen4exp;prefill" SKIP_RETURN_CODE 77 TIMEOUT 1800)
  add_dependencies(qwen4exp_route_capture ${B70_Q4EXP_TP_KERNELS} ${B70_Q4EXP_TP_PREFILL_KERNELS})   # its tp mode
  add_dependencies(qwen4exp_prefill_test ${B70_Q4EXP_TP_PREFILL_KERNELS})   # its gdn-pb mode's tp heads
  # PERF-44 (one card, no checkpoint): out_proj / o_proj's send-folded gemv + the wired tp_sum bitwise the plain gemv +
  # tp_send + tp_sum, two sides on one device each writing the other's landing; M 1, and 2..4 with the MTP builds.
  set(_tpw_m 1)
  if(B70_MTP)
    list(APPEND _tpw_m 2 3 4)
  endif()
  # PERF-109: at the S4 split, and the S2 split's plain gemv as the judged control (--bench: the S2 fused pair and one
  # card's S4 gemv against the new)
  set(_tpw_kernels kernel_gemv_M1_K3072_N2560_S2_L0_TP kernel_tp_cross_M1_K2560_G20_S2_X_R0_sum_W64
    kernel_tp_cross_M1_K2560_G20_S2_X_R1_sum_W64 kernel_gemv_M1_K6144_N2560_S8_L0)
  foreach(MM IN LISTS _tpw_m)
    list(APPEND _tpw_kernels kernel_gemv_M${MM}_K3072_N2560_S4_L0 kernel_gemv_M${MM}_K3072_N2560_S4_L0_TP
      kernel_gemv_M${MM}_K3072_N2560_S2_L0)
    foreach(R 0 1)
      list(APPEND _tpw_kernels kernel_tp_cross_M${MM}_K2560_G20_S4_X_R${R}_sum kernel_tp_cross_M${MM}_K2560_G20_S4_X_R${R}_sum_W64)
    endforeach()
  endforeach()
  add_executable(qwen4exp_tp_wire_test kernels/qwen4exp_tp_wire_test.cc)
  target_include_directories(qwen4exp_tp_wire_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_tp_wire_test PRIVATE b70_l0 b70_model)
  b70_target_kernel_dir(qwen4exp_tp_wire_test)
  add_dependencies(qwen4exp_tp_wire_test ${_tpw_kernels})
  if(TARGET kernel_gemv_M1_K6144_N2560_S4_L0)   # Kolibri's o_proj: --bench's one-card S4 control
    add_dependencies(qwen4exp_tp_wire_test kernel_gemv_M1_K6144_N2560_S4_L0)
  endif()
  add_test(NAME qwen4exp_tp_wire_test COMMAND qwen4exp_tp_wire_test ${_tpw_m})
  set_tests_properties(qwen4exp_tp_wire_test PROPERTIES LABELS "qwen4exp;tp" SKIP_RETURN_CODE 77 TIMEOUT 600)
endif()
# ==== Spec 21-QFN-v1e: Qwen3.8-Flash-Next served (label qwen4exp) ==== (begin) ====
# Task 1 (host): the tokens and the template. 21-QFN-v1a found the original's tokenizer.json NOT Qwen3.8's (the split regex
# adds \p{M}; Intel's checkpoint ships Qwen3.8's file), so qwen4exp_tokenizer_test holds the engine's tokenizer to
# tools/tokenizer/dump_qwen4exp.py's fixture (tests/tokenizer/qwen4exp_tokenizer.json: 24 texts, the added tokens
# both ways, corpus.txt's digests, three renders' ids) and the template to its hash (c3cf9e34... = Qwen3.8's)
# and the agnes_template_* renders (Qwen3.8's template, named after the dropped model they were first dumped for). Nothing of the release is vendored (its licence text unread): the original's small
# files are the synthetic checkpoint's (21-QFN-v1b's make_synth.py copies them; validation: r31.synth), or
# -DB70_Q4EXP_TOKENIZER_JSON / the environment variable of the same name (a directory holding the original's
# tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json). Absent: disabled on the
# Mac (host_only), SKIP (77) elsewhere.
if(B70_TOKENIZER_ENABLED)
  set(B70_Q4EXP_TOKENIZER_JSON "${B70_Q4EXP_SYNTH_DIR}/ours/ckpt/tokenizer.json"
    CACHE FILEPATH "Qwen3.8-Flash-Next's (the original's) tokenizer.json for qwen4exp_tokenizer_test, its small files beside it")
  get_filename_component(_q4_tok_dir "${B70_Q4EXP_TOKENIZER_JSON}" DIRECTORY)
  add_executable(qwen4exp_tokenizer_test tokenizer/qwen4exp_tokenizer_test.cc)
  target_include_directories(qwen4exp_tokenizer_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_tokenizer_test PRIVATE b70_tokenizer)
  add_test(NAME qwen4exp_tokenizer_test COMMAND qwen4exp_tokenizer_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${B70_Q4EXP_TOKENIZER_JSON})
  # ... and the server over that template (mock engine and tokenizer, cli::qwen4exp::chat_options - b70-serve's
  # dispatch's options): ChatFormat Qwen, the prompt = transformers' render, two parallel Qwen XML calls back as
  # OpenAI tool calls (buffered and streamed), reasoning on / off, both EOS ids stop, greedy without sampling fields
  # (decision 11's interim).
  add_executable(qwen4exp_server_test server/qwen4exp_server_test.cc)
  target_include_directories(qwen4exp_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_server_test PRIVATE b70_server b70_tokenizer)
  add_test(NAME qwen4exp_server_test COMMAND qwen4exp_server_test ${CMAKE_SOURCE_DIR}/tests/tokenizer ${_q4_tok_dir})
  set_tests_properties(qwen4exp_tokenizer_test qwen4exp_server_test PROPERTIES LABELS qwen4exp SKIP_RETURN_CODE 77
    TIMEOUT 300)
endif()
# Task 2: the prefix cache's snapshots. Host: the layouts (state_runs / kv_runs) over simulated devices - the bytes,
# one host layout under --parallel 1 and --parallel pp (every split of --layers 4 and 8), round trips across placements, the
# history before position 0 (EOS ids, zero conv rows), the open block's raw keys at 2049..2052 and the head's.
add_executable(qwen4exp_snapshot_test runtime/qwen4exp_snapshot_test.cc)
target_include_directories(qwen4exp_snapshot_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen4exp_snapshot_test PRIVATE b70_qwen4exp_plan)
add_test(NAME qwen4exp_snapshot_test COMMAND qwen4exp_snapshot_test)
set_tests_properties(qwen4exp_snapshot_test PROPERTIES LABELS qwen4exp TIMEOUT 300)
# Spec 39 Task 1: runtime/gpu_state.h - the rules and walks on a fixture, Flash-Next's reflected sums against the
# hand-written ones. Host only. Then the two builds that must fail (excluded from all; each test builds its target and
# passes on the refusal's text): a sizes member without an annotation, a buffer without a sizes member.
add_executable(gpu_state_test runtime/gpu_state_test.cc)
target_include_directories(gpu_state_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(gpu_state_test PRIVATE b70_qwen4exp_plan)
add_test(NAME gpu_state_test COMMAND gpu_state_test)
set_tests_properties(gpu_state_test PROPERTIES LABELS qwen4exp)
foreach(_case unannotated unplanned)
  string(TOUPPER ${_case} _CASE)
  add_library(gpu_state_fail_${_case} OBJECT EXCLUDE_FROM_ALL runtime/gpu_state_fail.cc)
  target_include_directories(gpu_state_fail_${_case} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(gpu_state_fail_${_case} PRIVATE GPU_STATE_FAIL_${_CASE})
  add_test(NAME gpu_state_${_case}_test
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target gpu_state_fail_${_case})
endforeach()
set_tests_properties(gpu_state_unannotated_test PROPERTIES LABELS qwen4exp
                     PASS_REGULAR_EXPRESSION "Sizes::idx_q4 has no GPU state annotation")
set_tests_properties(gpu_state_unplanned_test PROPERTIES LABELS qwen4exp
                     PASS_REGULAR_EXPRESSION "Bufs::idx_q4 is a buffer without exactly one sizes member")
# On the card (GPU only; SKIP 77 without the synthetic checkpoints / Intel's): qwen4exp_snapshot_gpu_test - restores at
# the block ends 2048 / 4096 bitwise the cold run, at the request ends 2049..2052 (the QSA cut), 5000 and 1 bitwise the
# session that never left; one card, two (pp2:2), and `cross` (a two-card snapshot continues on one card and the
# reverse); Intel's --layers 18 when present.
if(B70_Q4EXP)
  add_executable(qwen4exp_snapshot_gpu_test runtime/qwen4exp_snapshot_gpu_test.cc)
  target_include_directories(qwen4exp_snapshot_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(qwen4exp_snapshot_gpu_test PRIVATE b70_qwen4exp_prefill b70_l0)
  add_dependencies(qwen4exp_snapshot_gpu_test ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_DECODE_KERNELS})
  set(_q4snap_ids ${CMAKE_SOURCE_DIR}/tests/golden/prompts/q4exp_4k.ids)
  add_test(NAME qwen4exp_snapshot_gpu_synth_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt
    ${_q4snap_ids})
  add_test(NAME qwen4exp_snapshot_gpu_synth_intel_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_SYNTH_DIR}/intel/ckpt
    ${_q4snap_ids})
  add_test(NAME qwen4exp_snapshot_gpu_synth_pp_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt
    ${_q4snap_ids} pp2:2)
  add_test(NAME qwen4exp_snapshot_gpu_cross_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt
    ${_q4snap_ids} cross)
  add_test(NAME qwen4exp_snapshot_gpu_intel_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_INTEL_SNAPSHOT}
    ${_q4snap_ids} layers:18 int8)
  # spec 21-QFN-v1e Task 3: the same with the MTP head (its tail, R, K / V and keys ride the snapshots; continuations at K = 2)
  add_test(NAME qwen4exp_snapshot_gpu_synth_mtp_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt
    ${_q4snap_ids} mtp)
  add_test(NAME qwen4exp_snapshot_gpu_cross_mtp_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt
    ${_q4snap_ids} cross mtp)
  # plan 21-QFN-v3a: the int8 KV cache - a prefix cache hit bitwise cold, one card with the head, and pp
  add_test(NAME qwen4exp_snapshot_gpu_synth_kv8_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt
    ${_q4snap_ids} mtp kv8)
  add_test(NAME qwen4exp_snapshot_gpu_synth_pp_kv8_test COMMAND qwen4exp_snapshot_gpu_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt
    ${_q4snap_ids} pp2:2 kv8)
  set_tests_properties(qwen4exp_snapshot_gpu_synth_test qwen4exp_snapshot_gpu_synth_intel_test
    qwen4exp_snapshot_gpu_synth_pp_test qwen4exp_snapshot_gpu_cross_test qwen4exp_snapshot_gpu_intel_test
    qwen4exp_snapshot_gpu_synth_mtp_test qwen4exp_snapshot_gpu_cross_mtp_test qwen4exp_snapshot_gpu_synth_kv8_test
    qwen4exp_snapshot_gpu_synth_pp_kv8_test
    PROPERTIES LABELS "checkpoint;qwen4exp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# Task 3: the MTP head and spec 8's lists. B70_Q4EXP_MTP_KERNELS is src/kernels/CMakeLists.txt's spec 21-QFN-v1e block (the
# verify lists' M = 2..4 binaries, gdn_step_defer_M<1..4>_G1 (PERF-51), q4_pf_ple_C4, q4_mtp at M = 1..4 and _PF, the fc's);
# the lists also bind 21-QFN-v1c's decode binaries, 21-QFN-v1d's prefill binaries and Qwen3.8's argmax_stage1_M<2..4> by name,
# and (open-work 49) the draft steps Qwen3.8's draft_prob over the whole row.
if(B70_Q4EXP AND B70_MTP)
  set(_q4moe_per_row "")
  set(B70_Q4EXP_MTP_KERNELS kernel_q4_pf_ple_C4 kernel_q4_mtp_M2048_SINGLE_PF kernel_q4_mtp_M2048_STREAM_PF
    kernel_q4_mtp_M4096_SINGLE_PF kernel_q4_mtp_M4096_STREAM_PF kernel_q4_pf_bf16_slab_K2560_N2560
    kernel_pf_attn_prep_q16_Q24KV2_KVO)   # PERF-85
  foreach(MM 1 2 3 4)
    list(APPEND B70_Q4EXP_MTP_KERNELS kernel_gdn_step_defer_M${MM}_G1 kernel_gemv_bf16_M${MM}_K2560_N2560
      kernel_q4_mtp_M${MM}_SINGLE kernel_q4_mtp_M${MM}_STREAM)
    if(NOT MM EQUAL 1)
      list(APPEND B70_Q4EXP_MTP_KERNELS kernel_gemv_M${MM}_K2560_N16384_S1_L0 kernel_gemv_M${MM}_K2560_N13312_S2_L0
        kernel_gemv_M${MM}_K6144_N2560_S8_L0 kernel_gemv_bf16_M${MM}_K2560_N16384 kernel_gemv_bf16_M${MM}_K2560_N13312
        kernel_gemv_bf16_M${MM}_K6144_N2560 kernel_gemv_bf16_M${MM}_K2560_N128_C16_S16
        kernel_gemv_bf16_M${MM}_K10240_N336_C16_S16 kernel_gemv_bf16_M${MM}_K10240_N320_C16_S16
        kernel_gemv_bf16_M${MM}_K2560_N640_C16_S16 kernel_gemv_bf16_M${MM}_K2560_N528_C16_S16
        kernel_gemv_bf16_M${MM}_K2560_N12800 kernel_gemv_bf16_M${MM}_K2560_N248320 kernel_gemv_i8w_M${MM}_K2560_N248320
        kernel_embed_gather_M${MM}_D2560 kernel_prep_gated_head_M${MM}_SIG kernel_attn_prep_M${MM}_Q24KV2
        kernel_attn_prep_M${MM}_Q24KV2_S1 kernel_q4_hc_combine_norm_M${MM}_E kernel_q4_hc_combine_norm_M${MM}_S1
        kernel_q4_hc_combine_norm_M${MM}_S8 kernel_q4_hc_combine_norm_M${MM}_Y kernel_q4_hc_combine_norm_M${MM}_Y_NN
        kernel_q4_hc_combine_norm_M${MM}_X kernel_q4_hc_up_mix_M${MM}_I_B kernel_q4_hc_up_mix_M${MM}_B
        kernel_q4_hc_up_mix_M${MM}_I_Q8_B kernel_q4_hc_up_mix_M${MM}_Q8_B
        kernel_q4_hc_down_i8_M${MM}_N336_KS32 kernel_q4_hc_down_i8_M${MM}_N320_KS32
        kernel_q4_ple_gather_M${MM}_BF16 kernel_q4_ple_gather_M${MM}_F32 kernel_q4_qsa_M${MM}_T512_W1024_I4H
        kernel_q4_qsa_attn_M${MM}_T32 kernel_q4_qsa_attn_eager_M${MM} kernel_q4_route_M${MM}_E512_T10_N528_L256
        kernel_q4_moe_M${MM}_E512_T10_D2560_I640_SH4_U kernel_q4_moe_M${MM}_E512_T10_D2560_I640_SHB_U
        kernel_q4_moe_M${MM}_E512_T10_D2560_I640_SH4_TBL_U kernel_q4_moe_M${MM}_E512_T10_D2560_I640_SHB_TBL_U)
      # the per-row MoE at M rows: the union's reference (qwen4exp_moe_union_test), no list binds it
      list(APPEND _q4moe_per_row kernel_q4_moe_M${MM}_E512_T10_D2560_I640_SH4 kernel_q4_moe_M${MM}_E512_T10_D2560_I640_SHB
        kernel_q4_moe_M${MM}_E512_T10_D2560_I640_SH4_TBL kernel_q4_moe_M${MM}_E512_T10_D2560_I640_SHB_TBL)
    endif()
  endforeach()
  # Spec 28d: the verify lists' dense int4 projections and int8 head at M 2..4 on gemm_sm (spec 28b's block builds
  # them, kSmallMTuning's Flash-Next rows) - in this list so every MTP test depends on them; the gemv / gemv_i8w ones
  # above stay bound under the tests' override (CaptureOptions::force_linear).
  foreach(MM 2 3 4)
    list(APPEND B70_Q4EXP_MTP_KERNELS kernel_gemm_sm_M${MM}_K2560_N16384_S1_L0_mg kernel_gemm_sm_M${MM}_K2560_N13312_S2_L0_mg
      kernel_gemm_sm_M${MM}_K6144_N2560_S8_L0_mg kernel_gemm_sm_i8w_M${MM}_K2560_N248320_wg)
  endforeach()
  set(_q4mtp_reused ${B70_Q4EXP_DECODE_KERNELS} ${B70_Q4EXP_PREFILL_KERNELS} kernel_argmax_stage1_M2
    kernel_argmax_stage1_M3 kernel_argmax_stage1_M4 kernel_draft_prob_N248320_U248077 ${_q4moe_per_row})
  if(TARGET qwen4exp_snapshot_gpu_test)   # its `mtp` mode binds the head's lists
    add_dependencies(qwen4exp_snapshot_gpu_test ${B70_Q4EXP_MTP_KERNELS})
  endif()
  # Host: the names the lists bind (kernels::qwen4exp verify / draft / mtp_prefill_variants) are built, the block's
  # binaries are all bound; the launch counts of every list (runtime::qwen4exp's).
  add_executable(qwen4exp_mtp_names_test kernels/qwen4exp_mtp_names_test.cc)
  target_include_directories(qwen4exp_mtp_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_mtp_names_test PRIVATE b70_qwen4exp_plan)
  b70_target_kernel_dir(qwen4exp_mtp_names_test)
  add_test(NAME qwen4exp_mtp_names_test COMMAND qwen4exp_mtp_names_test ${B70_Q4EXP_MTP_KERNELS} --reused ${_q4mtp_reused})
  set_tests_properties(qwen4exp_mtp_names_test PROPERTIES LABELS qwen4exp)
  # PERF-85: the head's attn_prep over the kv heads only, every K / V word against the full grid of the same source at
  # 24 / 2 (the executable is the MTP block's above when that builds it - Qwen3.8's and Ornith's heads)
  if(NOT TARGET pf_attn_prep_kvo_test)
    add_executable(pf_attn_prep_kvo_test prefill/pf_attn_prep_kvo_test.cc)
    target_include_directories(pf_attn_prep_kvo_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                               ${CMAKE_SOURCE_DIR}/tests/kernels ${CMAKE_SOURCE_DIR}/tests/prefill)
    target_link_libraries(pf_attn_prep_kvo_test PRIVATE b70_l0)
    b70_target_kernel_dir(pf_attn_prep_kvo_test)
  endif()
  add_dependencies(pf_attn_prep_kvo_test kernel_pf_attn_prep_q16_Q24KV2 kernel_pf_attn_prep_q16_Q24KV2_KVO)
  add_test(NAME pf_attn_prep_kvo_fn_test COMMAND pf_attn_prep_kvo_test 24 2)
  set_tests_properties(pf_attn_prep_kvo_fn_test PROPERTIES LABELS "qwen4exp;prefill;mtp")
  # On the card (GPU only; SKIP 77 without the synthetic checkpoints / Intel's): qwen4exp_mtp_test - M1 the head
  # against 21-QFN-v1a's port on tools/oracle/qwen4exp_mtp_fixture.py's data (both norm forms; SKIPs without the fixture),
  # M2 verify rows bitwise the plain steps, M3 greedy lossless at K = 1..3 (and the head off), the verify's cost and
  # per-layer expert union; one card and two (pp2:2); Intel's --layers 18 when present.
  add_executable(qwen4exp_mtp_test runtime/qwen4exp_mtp_test.cc)
  b70_target_kernel_dir(qwen4exp_mtp_test)
  target_include_directories(qwen4exp_mtp_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(qwen4exp_mtp_test PRIVATE b70_qwen4exp_prefill b70_l0)
  add_dependencies(qwen4exp_mtp_test ${B70_Q4EXP_MTP_KERNELS} ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_DECODE_KERNELS})
  set(B70_Q4EXP_MTP_FIXTURE_DIR "${B70_ORACLE}/flash-next/mtp-fixture" CACHE PATH
    "tools/oracle/qwen4exp_mtp_fixture.py's M1 data: <dir>/<ours|intel>/mtp_{single,per_stream}.safetensors (validation: r34.fixture)")
  set(_q4m ${B70_Q4EXP_MTP_FIXTURE_DIR})
  set(_q4m_ids ${CMAKE_SOURCE_DIR}/tests/golden/prompts/q4exp_4k.ids)
  add_test(NAME qwen4exp_mtp_synth_test COMMAND qwen4exp_mtp_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m}/ours ${_q4m_ids})
  add_test(NAME qwen4exp_mtp_synth_intel_test COMMAND qwen4exp_mtp_test ${B70_Q4EXP_SYNTH_DIR}/intel/ckpt ${_q4m}/intel
    ${_q4m_ids})
  add_test(NAME qwen4exp_mtp_synth_eager_test COMMAND qwen4exp_mtp_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m}/ours
    ${_q4m_ids} m2 m3)
  set_tests_properties(qwen4exp_mtp_synth_eager_test PROPERTIES ENVIRONMENT "B70_Q4_ATTN=eager")
  add_test(NAME qwen4exp_mtp_synth_pp_test COMMAND qwen4exp_mtp_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m}/ours
    ${_q4m_ids} m2 m3 cost pp2:2)
  add_test(NAME qwen4exp_mtp_intel_test COMMAND qwen4exp_mtp_test ${B70_Q4EXP_INTEL_SNAPSHOT} ${_q4m}/intel-L18 ${_q4m_ids}
    layers:18 int8)
  set_tests_properties(qwen4exp_mtp_synth_test qwen4exp_mtp_synth_intel_test qwen4exp_mtp_synth_eager_test
    qwen4exp_mtp_synth_pp_test qwen4exp_mtp_intel_test
    PROPERTIES LABELS "checkpoint;qwen4exp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  # Plan 21-QFN-v2a Task 8, the expert tier on the ours synthetic (tests/runtime/qwen4exp_tier_test.cc): O0 / O1 - the
  # direct form against the table at all / none / random:0.5:7 / adversarial, prefill, decode and verify at M = 2..4,
  # one card and two (device 1's experts in host memory); O3 - the load check refuses a corrupted mirror, arena or
  # table entry.
  add_executable(qwen4exp_tier_test runtime/qwen4exp_tier_test.cc)
  b70_target_kernel_dir(qwen4exp_tier_test)
  target_include_directories(qwen4exp_tier_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(qwen4exp_tier_test PRIVATE b70_qwen4exp_prefill b70_l0)
  add_dependencies(qwen4exp_tier_test ${B70_Q4EXP_MTP_KERNELS} ${B70_Q4EXP_PREFILL_KERNELS} ${B70_Q4EXP_DECODE_KERNELS})
  add_test(NAME qwen4exp_tier_synth_test COMMAND qwen4exp_tier_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m_ids} 1)
  add_test(NAME qwen4exp_tier_synth_pp_test COMMAND qwen4exp_tier_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m_ids} pp2)
  add_test(NAME qwen4exp_tier_check_test COMMAND qwen4exp_tier_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m_ids} check)
  # PERF-124: the host-memory experts on the CPU (--host-experts cpu: every one on the CPU), pp2, bitwise the direct run
  add_test(NAME qwen4exp_tier_synth_pp_cpu_test COMMAND qwen4exp_tier_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m_ids} pp2 cpu)
  set_tests_properties(qwen4exp_tier_synth_test qwen4exp_tier_check_test
    PROPERTIES LABELS "checkpoint;qwen4exp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  set_tests_properties(qwen4exp_tier_synth_pp_test qwen4exp_tier_synth_pp_cpu_test PROPERTIES LABELS "checkpoint;qwen4exp;pp" SKIP_RETURN_CODE 77
    TIMEOUT 3600)
  # Open-work item 48: the union-read MoE (q4_moe_union.cl) bitwise the per-row q4_moe at M = 2..4, one card's I 640
  # (_SH4 / _SHB, direct / table with host-memory experts) and tp's I 320 halves; the counting builds (each chosen
  # expert read once a verify); --time opt-in. Needs a B70, no checkpoint.
  if(B70_TP)
    set(_q4u_kernels kernel_q4_moe_M4_E512_T10_D2560_I640_SH4_TBL_U_CNT kernel_q4_moe_M4_E512_T10_D2560_I320_SH4_TBL_TP_R0_U_CNT)
    foreach(MM 2 3 4)
      foreach(F SH4 SHB SH4_TBL SHB_TBL)
        list(APPEND _q4u_kernels kernel_q4_moe_M${MM}_E512_T10_D2560_I640_${F} kernel_q4_moe_M${MM}_E512_T10_D2560_I640_${F}_U)
      endforeach()
      foreach(F TP_R0 TP_R1 TBL_TP_R0 TBL_TP_R1)
        list(APPEND _q4u_kernels kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_${F} kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_${F}_U)
      endforeach()
    endforeach()
    add_executable(qwen4exp_moe_union_test kernels/qwen4exp_moe_union_test.cc)
    target_include_directories(qwen4exp_moe_union_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(qwen4exp_moe_union_test PRIVATE b70_l0 b70_qwen4exp_loader)
    b70_target_kernel_dir(qwen4exp_moe_union_test)
    add_dependencies(qwen4exp_moe_union_test ${_q4u_kernels})
    add_test(NAME qwen4exp_moe_union_test COMMAND qwen4exp_moe_union_test)
    set_tests_properties(qwen4exp_moe_union_test PROPERTIES LABELS qwen4exp SKIP_RETURN_CODE 77 TIMEOUT 1800)
  endif()
  # Plan 21-QFN-v2b Task 5 Step 0: the MTP head under --parallel tp (GPU only; our RTN export, 18 layers): the plain run's
  # greedy ids, then the head's plain steps, spec 8's loop at K 1..3 and a restored snapshot against them.
  if(B70_TP AND TARGET qwen4exp_tp_prefill_test)
    set(B70_Q4EXP_TP_MTP_KERNELS kernel_draft_prob_N124160_U124160 kernel_draft_prob_N124160_U123917)   # open-work 49
    foreach(MM 1 2 3 4)
      list(APPEND B70_Q4EXP_TP_MTP_KERNELS kernel_gdn_step_defer_M${MM}_GK8V24_G1)
      if(NOT MM EQUAL 1)
        list(APPEND B70_Q4EXP_TP_MTP_KERNELS kernel_gemv_M${MM}_K2560_N8192_S1_L0_KS4 kernel_gemv_M${MM}_K2560_N6656_S2_L0
          kernel_gemv_M${MM}_K3072_N2560_S4_L0_TP kernel_gemv_i8w_M${MM}_K2560_N124160 kernel_prep_gated_head_M${MM}_GK8V24_SIG
          kernel_attn_prep_M${MM}_Q12KV1 kernel_q4_qsa_attn_M${MM}_T32_Q12KV1
          kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TP_R0 kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TP_R1
          kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TBL_TP_R0 kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TBL_TP_R1
          kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TP_R0_U kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TP_R1_U
          kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TBL_TP_R0_U kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TBL_TP_R1_U
          kernel_tp_cross_M${MM}_K2560_G20_S4_X_R0_sum_W64 kernel_tp_cross_M${MM}_K2560_G20_S4_X_R1_sum_W64
          kernel_tp_argmax_M${MM}_N124160_U124160_B0 kernel_tp_argmax_M${MM}_N124160_U123917_B124160)
      endif()
    endforeach()
    add_executable(qwen4exp_tp_mtp_test runtime/qwen4exp_tp_mtp_test.cc)
    b70_target_kernel_dir(qwen4exp_tp_mtp_test)
    target_include_directories(qwen4exp_tp_mtp_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
      ${CMAKE_SOURCE_DIR}/tests/runtime)
    target_link_libraries(qwen4exp_tp_mtp_test PRIVATE b70_qwen4exp_prefill b70_l0)
    add_dependencies(qwen4exp_tp_mtp_test ${B70_Q4EXP_TP_MTP_KERNELS} ${B70_Q4EXP_MTP_KERNELS} ${_q4mtp_reused}
      ${B70_Q4EXP_TP_KERNELS})
    foreach(mode plain mtp)
      add_test(NAME qwen4exp_tp_mtp_${mode}_test COMMAND qwen4exp_tp_mtp_test ${B70_Q4EXP_RTN_SNAPSHOT}
        ${CMAKE_SOURCE_DIR}/tests/golden/prompts/q4exp_4k.ids ${mode} ${CMAKE_BINARY_DIR}/qwen4exp_tp_mtp_ids.u32)
      set_tests_properties(qwen4exp_tp_mtp_${mode}_test PROPERTIES LABELS "checkpoint;qwen4exp;tp" SKIP_RETURN_CODE 77
        TIMEOUT 3600)
    endforeach()
    set_tests_properties(qwen4exp_tp_mtp_mtp_test PROPERTIES DEPENDS qwen4exp_tp_mtp_plain_test)
    # PERF-124 under tp (qwen4exp_tier_test tp2, the ours synthetic): each card's half experts in host memory read over
    # PCIe, then (cpu) computed on the CPU; plain and with the head, bitwise tp's all-resident run, the ids pp2's.
    set(_q4tp_cpu)
    foreach(R 0 1)
      list(APPEND _q4tp_cpu kernel_q4_moe_M1_E512_T10_D2560_I320_SH4_TBL_TP_R${R}_CPU)
      foreach(MM 2 3 4)
        list(APPEND _q4tp_cpu kernel_q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TBL_TP_R${R}_U_CPU)
      endforeach()
    endforeach()
    add_dependencies(qwen4exp_tier_test ${B70_Q4EXP_TP_MTP_KERNELS} ${B70_Q4EXP_TP_KERNELS} ${_q4tp_cpu})
    add_test(NAME qwen4exp_tier_synth_tp_test COMMAND qwen4exp_tier_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m_ids} tp2)
    add_test(NAME qwen4exp_tier_synth_tp_cpu_test COMMAND qwen4exp_tier_test ${B70_Q4EXP_SYNTH_DIR}/ours/ckpt ${_q4m_ids}
      tp2 cpu)
    set_tests_properties(qwen4exp_tier_synth_tp_test qwen4exp_tier_synth_tp_cpu_test PROPERTIES
      LABELS "checkpoint;qwen4exp;tp" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  endif()
endif()
# Task 4: b70-serve serves this family (src/cli/qwen4exp_serve.h; 21-QFN-v1c's cli_reject_qwen4exp_serve is gone). The
# adapter: spec 16d's cli::pp::PipelineEngineAdapterT instantiated over Qwen4ExpEngine (an object library - the Mac's
# Level Zero syntax check compiles it; b70-serve links the same instantiation). The refusals before the device, each by
# name (tests/model/qwen4exp: the model_type stub; tests/model/qwen4exp-q38tok: the stub with Qwen3.8's split regex).
if(B70_Q4EXP)
  add_library(qwen4exp_adapter_compile OBJECT cli/qwen4exp_adapter_compile.cc)
  target_include_directories(qwen4exp_adapter_compile PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(qwen4exp_adapter_compile PRIVATE b70_qwen4exp_runtime b70_server)
endif()
if(TARGET b70-serve)
  set(_q4_cfg ${CMAKE_CURRENT_SOURCE_DIR}/model/qwen4exp)
  b70_serve_reject(cli_reject_serve_qwen4exp_prefill_int8 ${_q4_cfg} --layers 4 --prefill-backend l0-int8
    EXPECT "1024-k Hadamard blocks")
  b70_serve_reject(cli_reject_serve_qwen4exp_split_alone ${_q4_cfg} --layers 4 --parallel 1 --pipeline-split 2
    EXPECT "--pipeline-split belongs to --parallel pp")
  b70_serve_reject(cli_reject_serve_qwen4exp_tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/model/qwen4exp-q38tok --layers 4
    EXPECT "Serve a snapshot that ships")
  b70_serve_reject(cli_reject_serve_layers_other ${CMAKE_CURRENT_SOURCE_DIR}/model/ornith --layers 4
    EXPECT "belongs to Qwen3.8-Flash-Next")
  set_tests_properties(
    cli_reject_serve_qwen4exp_prefill_int8
    cli_reject_serve_qwen4exp_split_alone cli_reject_serve_qwen4exp_tokenizer
    cli_reject_serve_layers_other PROPERTIES LABELS qwen4exp)
endif()
# ==== Spec 21-QFN-v1e: Qwen3.8-Flash-Next served ==== (end) ====

# ==== Spec 22a: the expert-offload probe ==== (begin) ====
# tools/probe/probe_offload_moe.cl is q4_moe.cl with its two expert-address lines as a pointer-table read
# (tools/probe/mk_offload_kernel.py); the drift test reruns the generator in --check mode. Host only.
add_executable(probe_offload_kernel_drift_test kernels/probe_offload_kernel_drift_test.cc)
add_test(NAME probe_offload_kernel_drift_test COMMAND probe_offload_kernel_drift_test ${CMAKE_SOURCE_DIR})
set_tests_properties(probe_offload_kernel_drift_test PROPERTIES LABELS host)
# ==== Spec 22a: the expert-offload probe ==== (end) ====

# b70-serve refuses another model's flags on a Kolibri-1 config, before the device: --spec dflash without Kolibri's own
# drafter (spec 19c). (Written for spec 23e; Muse-Glimmer-30B was dropped 2026-10-27.)
if(TARGET b70-serve AND COMMAND b70_serve_reject)
  b70_serve_reject(cli_reject_serve_spec_dflash_other ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --spec dflash
    EXPECT "--spec dflash on Kolibri-1 needs --draft-model")   # Kolibri DSpark: its drafter by name
endif()

# ==== Spec 19b: the DFlash drafter (label dflash) ==== (begin) ====
# Task 1: the descriptor (the published config vendored at tests/model/dflash/) and the loader's host half on tiny
# synthetic checkpoints the test writes itself (tests/loader/dflash_fixture.h). Host only.
add_executable(dflash_loader_test loader/dflash_loader_test.cc)
target_include_directories(dflash_loader_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_loader_test PRIVATE b70_dflash_loader)
add_test(NAME dflash_loader_test COMMAND dflash_loader_test
  ${CMAKE_CURRENT_SOURCE_DIR}/model/dflash/qwen38_dflash2_config.json)
set_tests_properties(dflash_loader_test PROPERTIES LABELS dflash)
# Spec 19 v3c Task 2: the drafter's halves for --parallel tp (loader/dflash_tp_split.h) in every device form, and a TP
# card's plan with its half at 262,144 positions. Host only.
add_executable(dflash_tp_split_test loader/dflash_tp_split_test.cc)
target_include_directories(dflash_tp_split_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_tp_split_test PRIVATE b70_dflash_loader b70_dflash_plan b70_plan b70_model)
add_test(NAME dflash_tp_split_test COMMAND dflash_tp_split_test)
set_tests_properties(dflash_tp_split_test PROPERTIES LABELS "dflash;tp")
add_executable(dflash_head_test loader/dflash_head_test.cc)
target_include_directories(dflash_head_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_head_test PRIVATE b70_dflash_loader)
add_test(NAME dflash_head_test COMMAND dflash_head_test)
set_tests_properties(dflash_head_test PROPERTIES LABELS dflash)
# Task 1 (device half): load_drafter beside the int4 gate checkpoint (int8 head), both formats - the bytes against
# the layout formula, linears / norms / codebook rows / the mask row read back from the card. SKIP 77 without either
# snapshot. B70_DFLASH_SNAPSHOT: a repo id (resolved in the HF cache) or a directory.
set(B70_DFLASH_SNAPSHOT "z-lab/Qwen3.8-27B-DFlash2" CACHE STRING
  "The DFlash drafter the dflash-labelled checkpoint tests load (spec 19b)")
add_executable(dflash_load_checkpoint_test loader/dflash_load_checkpoint_test.cc)
target_include_directories(dflash_load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_load_checkpoint_test PRIVATE b70_dflash_loader b70_l0)
add_test(NAME dflash_load_checkpoint_test COMMAND dflash_load_checkpoint_test ${B70_TEST_SNAPSHOT}
  ${B70_DFLASH_SNAPSHOT})
set_tests_properties(dflash_load_checkpoint_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77
  TIMEOUT 1800)
# The AutoRound int4 drafter (B70_DFLASH_PACKED_SNAPSHOT): its packed linears on the card, every one unpacked bitwise
# the CPU dequant of the shard.
set(B70_DFLASH_PACKED_SNAPSHOT "urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound" CACHE STRING
  "The AutoRound int4 DFlash drafter the packed checkpoint tests load")
add_test(NAME dflash_load_packed_test COMMAND dflash_load_checkpoint_test ${B70_TEST_SNAPSHOT}
  ${B70_DFLASH_PACKED_SNAPSHOT})
set_tests_properties(dflash_load_packed_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77 TIMEOUT 1800)
# Task 2: the host twin (tests/kernels/dflash_ref.h) against tools/oracle/dflash_ref.py's own outputs
# (tests/loader/dflash_ref_fixture.safetensors, tools/oracle/dflash_fixture.py). Host only.
add_executable(dflash_ref_test kernels/dflash_ref_test.cc)
target_include_directories(dflash_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_ref_test PRIVATE b70_loader b70_dflash_model)
add_test(NAME dflash_ref_test COMMAND dflash_ref_test ${CMAKE_CURRENT_SOURCE_DIR}/loader/dflash_ref_fixture.safetensors)
set_tests_properties(dflash_ref_test PROPERTIES LABELS dflash)
# Task 2: every binary the draft / projection lists bind (kernels::dflash::draft_variants) is built - by the Spec 19b
# block (B70_DFLASH_KERNELS) or reused (B70_VERIFY_M8's drafter GEMVs and int8 heads) - and the block builds nothing
# unbound. Host only (names, no device).
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  set(B70_DFLASH_KERNELS kernel_df_prep_M8_H5120_I17408_Q32KV8_D128_G16T2_R2048_V248320
    kernel_df_attn_M8_Q32KV8_D128_R2048_W2048_P128 kernel_df_topk_R7_N248320_U248077
    kernel_df_select_M8_K16_Q16_RK256_V248320 kernel_df_select_v1_M8_K16 kernel_gemv_bf16_M7_K5120_N248320)
  list(APPEND B70_DFLASH_KERNELS kernel_df_prep_M8_H5120_I17408_Q32KV8_D128_G16T2_R2048_V248320_F8Q4R4U8C16H16K8
       kernel_df_select_M8_K16_Q16_RK256_V248320_H16)
  foreach(RPW 1 2 4)
    list(APPEND B70_DFLASH_KERNELS kernel_df_attn_M8_Q32KV8_D128_R2048_W2048_P128_H1_B${RPW})
  endforeach()
  list(APPEND B70_DFLASH_KERNELS kernel_df_candidate_R7_N248320_U248077_K64
       kernel_df_rerank_R7_K5120_U248077_C64_Q4)
  list(APPEND B70_DFLASH_KERNELS kernel_df_candidate_R7_N248320_U248077_K80 kernel_df_rerank_R7_K5120_U248077_C80_Q4)
  list(APPEND B70_DFLASH_KERNELS kernel_df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas   # spec 19 v3d Task 6
       kernel_df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas_k2_g256_blk   # plan 41k: KS 2 by 2D reads at 256 GRF
       kernel_df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas_k4_g256)   # plan 19v3b: KS 4 at 256 GRF
  list(APPEND B70_DFLASH_KERNELS kernel_df_proj_wide_H5120_KV8_D128_R2048)
  set(_v19b_reused kernel_gemv_i8w_M7_K5120_N248320)
  foreach(KN K5120_N6144 K4096_N5120 K5120_N34816 K17408_N5120 K25600_N5120 K5120_N2048)
    list(APPEND _v19b_reused kernel_gemv_i8w_M8_${KN} kernel_gemv_bf16_M8_${KN})
  endforeach()
  foreach(KN K5120_N1280 K5120_N256)
    list(APPEND _v19b_reused kernel_gemv_i8w_M8_${KN} kernel_gemv_bf16_M8_${KN}_C16_S16)
  endforeach()
  # Spec 28d: the int8 linears and heads on gemm_sm_i8w (spec 28b's block builds them; gemv_i8w above stays bound
  # under the tests' override, CaptureOptions::force_linear).
  foreach(KN K5120_N6144 K4096_N5120 K5120_N34816 K17408_N5120 K25600_N5120 K5120_N2048 K5120_N1280 K5120_N256)
    list(APPEND _v19b_reused kernel_gemm_sm_i8w_M8_${KN}_wg)
    list(APPEND _v19b_reused kernel_gemm_sm_M8_${KN}_S1_L1_mg kernel_gemv_M8_${KN}_S1_L1)
  endforeach()
  list(APPEND _v19b_reused kernel_gemm_sm_M8_K5120_N6144_S4_L1_mg kernel_gemv_M8_K5120_N6144_S4_L1)
  list(APPEND _v19b_reused kernel_gemm_sm_M8_K4096_N5120_S4_L1_mg kernel_gemv_M8_K4096_N5120_S4_L1)
  list(APPEND _v19b_reused kernel_gemm_sm_M8_K5120_N34816_S8_L1_mg kernel_gemv_M8_K5120_N34816_S8_L1)
  list(APPEND _v19b_reused kernel_gemm_sm_M8_K17408_N5120_S4_L1_mg kernel_gemv_M8_K17408_N5120_S4_L1)
  list(APPEND _v19b_reused kernel_gemm_sm_M8_K25600_N5120_S8_L1_mg kernel_gemv_M8_K25600_N5120_S8_L1)
  list(APPEND _v19b_reused kernel_gemm_sm_M8_K5120_N2048_S8_L1_mg kernel_gemv_M8_K5120_N2048_S8_L1)
  list(APPEND _v19b_reused kernel_gemm_sm_M8_K5120_N1280_S16_L1_mg kernel_gemv_M8_K5120_N1280_S16_L1)
  list(APPEND _v19b_reused kernel_gemm_sm_M8_K5120_N256_S16_L1_mg kernel_gemv_M8_K5120_N256_S16_L1)
  list(APPEND _v19b_reused kernel_gemm_sm_i8w_M7_K5120_N248320_wg)
  list(APPEND _v19b_reused kernel_gemm_sm_M7_K5120_N248320_S1_L1_mg kernel_gemv_M7_K5120_N248320_S1_L1
       kernel_gemm_sm_M7_K5120_N248320_S1_L3_mg)   # spec 19 v3d: the 2-bit candidate head
  foreach(t ${B70_DFLASH_KERNELS} ${_v19b_reused})
    if(NOT TARGET ${t})
      message(FATAL_ERROR "spec 19b: ${t} is not built (src/kernels/CMakeLists.txt)")
    endif()
  endforeach()
  add_executable(dflash_variant_names_test kernels/dflash_variant_names_test.cc)
  target_include_directories(dflash_variant_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(dflash_variant_names_test PRIVATE b70_dflash_model)
  b70_target_kernel_dir(dflash_variant_names_test)
  add_test(NAME dflash_variant_names_test COMMAND dflash_variant_names_test
    ${CMAKE_CURRENT_SOURCE_DIR}/model/dflash/qwen38_dflash2_config.json ${B70_DFLASH_KERNELS} -- ${_v19b_reused})
  set_tests_properties(dflash_variant_names_test PROPERTIES LABELS dflash)
  # Task 2 on the card: every df_* kernel at the drafter's real shapes against the host twin, random data (no
  # checkpoint): bitwise where no transcendental is involved, within 1 bf16 ulp through exp, the window / wrap / K
  # cases, top-16 with planted ties / NaN / the mask, both head forms of the walk.
  add_executable(dflash_kernels_test kernels/dflash_kernels_test.cc)
  target_include_directories(dflash_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(dflash_kernels_test PRIVATE b70_l0)
  b70_target_kernel_dir(dflash_kernels_test)
  add_dependencies(dflash_kernels_test ${B70_DFLASH_KERNELS} kernel_gemv_M8_K5120_N2048_S8_L1)
  add_executable(dflash_head_kernels_test kernels/dflash_head_kernels_test.cc)
  target_include_directories(dflash_head_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(dflash_head_kernels_test PRIVATE b70_l0)
  b70_target_kernel_dir(dflash_head_kernels_test)
  add_dependencies(dflash_head_kernels_test kernel_df_candidate_R7_N248320_U248077_K64
                   kernel_df_rerank_R7_K5120_U248077_C64_Q4)
  # PERF-111: the rerank in today's n-major order, test-only twins - its output must be the production build's
  foreach(CC 64 80)
    add_ocloc_kernel(df_rerank_R7_K5120_U248077_C${CC}_Q4_NM SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/dflash/df_rerank.cl
                     DEFINES K=5120 USED=248077 C=${CC} N=248320 GEMV_KMAJOR=0)
    add_dependencies(dflash_head_kernels_test kernel_df_rerank_R7_K5120_U248077_C${CC}_Q4_NM)
  endforeach()
  add_test(NAME dflash_head_kernels_test COMMAND dflash_head_kernels_test)
  set_tests_properties(dflash_head_kernels_test PROPERTIES LABELS dflash TIMEOUT 900)
  add_test(NAME dflash_kernels_test COMMAND dflash_kernels_test)
  set_tests_properties(dflash_kernels_test PROPERTIES LABELS dflash TIMEOUT 900)
endif()
# Task 3: the drafter's buffers through the planner (sizes, DrafterPlan, the memory line, max_len auto), the RoPE
# table against the twin, the launch formulas. Host only.
add_executable(dflash_plan_test runtime/dflash_plan_test.cc)
target_include_directories(dflash_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_plan_test PRIVATE b70_dflash_plan b70_dflash_loader)
add_test(NAME dflash_plan_test COMMAND dflash_plan_test ${CMAKE_CURRENT_SOURCE_DIR}/model/dflash/qwen38_dflash2_config.json)
set_tests_properties(dflash_plan_test PROPERTIES LABELS dflash)
# Task 3 on the card: the draft and projection lists end to end - the real drafter (B70_DFLASH_SNAPSHOT, both formats)
# beside a synthetic target (random embedding, random int8 / bf16 heads at Qwen3.8's shapes) against the host twin:
# the projection across the ring's wrap, a draft at p 2053, determinism, K by the control block, a sampled walk, a
# SKIP 77 without the drafter.
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(dflash_buffers_test runtime/dflash_buffers_test.cc)
  target_include_directories(dflash_buffers_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(dflash_buffers_test PRIVATE b70_dflash_runtime)
  add_test(NAME dflash_buffers_test COMMAND dflash_buffers_test ${CMAKE_CURRENT_SOURCE_DIR}/model/dflash/qwen38_dflash2_config.json)
  set_tests_properties(dflash_buffers_test PROPERTIES LABELS dflash TIMEOUT 60)
  add_executable(dflash_draft_test runtime/dflash_draft_test.cc)
  b70_target_kernel_dir(dflash_draft_test)
  target_include_directories(dflash_draft_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(dflash_draft_test PRIVATE b70_dflash_runtime)
  add_dependencies(dflash_draft_test ${B70_DFLASH_KERNELS} ${_v19b_reused})
  add_test(NAME dflash_draft_test COMMAND dflash_draft_test ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(dflash_draft_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  add_test(NAME dflash_draft_packed_test COMMAND dflash_draft_test ${B70_DFLASH_PACKED_SNAPSHOT})
  set_tests_properties(dflash_draft_packed_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  add_executable(dflash_profile_test runtime/dflash_profile_test.cc)
  target_include_directories(dflash_profile_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(dflash_profile_test PRIVATE b70_dflash_runtime b70_prefill_host b70_runtime)
  add_dependencies(dflash_profile_test ${B70_DFLASH_KERNELS} ${_v19b_reused})
  add_test(NAME dflash_profile_test COMMAND dflash_profile_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(dflash_profile_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77 TIMEOUT 900)
endif()
# ==== Spec 19b ==== (end) ====

# ==== Spec 19c: DFlash in the decode loop ==== (begin) ====
# The binaries the Engine's tap verify lists add (src/kernels/CMakeLists.txt's Spec 19c block): checked to exist as
# targets, as 19b's block does.
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  set(B70_DFLASH_VERIFY_KERNELS)
  foreach(MM 1 2 3 4)
    list(APPEND B70_DFLASH_VERIFY_KERNELS kernel_gdn_step_slots_M${MM}_N8)
  endforeach()
  foreach(MM 5 6 8)
    list(APPEND B70_DFLASH_VERIFY_KERNELS kernel_gemv_bf16_M${MM}_K5120_N248320)
  endforeach()
  foreach(t ${B70_DFLASH_VERIFY_KERNELS})
    if(NOT TARGET ${t})
      message(FATAL_ERROR "spec 19c: ${t} is not built (src/kernels/CMakeLists.txt)")
    endif()
  endforeach()
endif()
# Host: the arithmetic the loop rests on
# (the tap sites, the verify slots, the planner's terms, the cost table, the sparse-q acceptance). On the card: the engine
# with the real drafter beside the gate checkpoint (dflash_engine_test).
# Task 3 / D4: lossless sampling with the drafter's sparse q (server::accept_sparse), 10^6 seeded draws. Host only.
add_executable(dflash_accept_test server/dflash_accept_test.cc)
target_include_directories(dflash_accept_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_accept_test PRIVATE b70_server)
add_test(NAME dflash_accept_test COMMAND dflash_accept_test)
set_tests_properties(dflash_accept_test PROPERTIES LABELS dflash TIMEOUT 600)
add_executable(dflash_engine_host_test runtime/dflash_engine_host_test.cc)
target_include_directories(dflash_engine_host_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_engine_host_test PRIVATE b70_plan b70_server PkgConfig::ZE)
add_test(NAME dflash_engine_host_test COMMAND dflash_engine_host_test)
set_tests_properties(dflash_engine_host_test PROPERTIES LABELS dflash)
add_test(NAME dflash_engine_accept_oracle_test COMMAND python3
  ${CMAKE_SOURCE_DIR}/tools/oracle/test_dflash_engine_accept.py)
set_tests_properties(dflash_engine_accept_oracle_test PROPERTIES LABELS dflash)
# The engine with the real drafter beside the gate checkpoint (int8 head, bf16 KV): the lists' counts, the taps and
# the commit projection (Task 1), D3 at M = 1..8, partial acceptance (the GDN state at the last accepted row, the
# ring), D2 on the card at K 7 / 3 / 1, the last positions of max_len. SKIP 77 without either checkpoint.
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(dflash_engine_test runtime/dflash_engine_test.cc)
  target_include_directories(dflash_engine_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(dflash_engine_test PRIVATE b70_dflash_runtime b70_prefill_host b70_runtime b70_loader
    b70_l0 b70_model)
  add_dependencies(dflash_engine_test ${B70_DFLASH_KERNELS} ${_v19b_reused} ${B70_DFLASH_VERIFY_KERNELS})
  add_test(NAME dflash_engine_test COMMAND dflash_engine_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(dflash_engine_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77
    ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 5400)
  # Plan 19v3b Task 3: the drafter beside sequence slots - a session per slot, bitwise the one-slot engine, isolated.
  add_executable(dflash_pool_test runtime/dflash_pool_test.cc)
  target_include_directories(dflash_pool_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(dflash_pool_test PRIVATE b70_dflash_runtime b70_prefill_host b70_runtime b70_loader
    b70_l0 b70_model b70_server)
  add_dependencies(dflash_pool_test ${B70_DFLASH_KERNELS} ${_v19b_reused} ${B70_DFLASH_VERIFY_KERNELS})
  add_test(NAME dflash_pool_test COMMAND dflash_pool_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(dflash_pool_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  # Plan 40a Task 1: the drafter beside the KV page pool - pages in any order, reset, a longer prompt: bitwise the
  # one-slot engine.
  add_test(NAME dflash_pool_pages_test COMMAND dflash_pool_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT} --pool)
  set_tests_properties(dflash_pool_pages_test PROPERTIES LABELS "checkpoint;dflash;pool" SKIP_RETURN_CODE 77
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  # The DFlash verify at M = 2..8 on the int8 KV cache against the plain loop: v2 + gemv lists bitwise (ids and the
  # KV rows they wrote) at every K, the production / gemm_sm / DPAS arms' first difference recorded.
  add_executable(dflash_verify_kv8_test runtime/dflash_verify_kv8_test.cc)
  target_include_directories(dflash_verify_kv8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(dflash_verify_kv8_test PRIVATE b70_dflash_runtime b70_prefill_host b70_runtime b70_loader
    b70_l0 b70_model)
  add_dependencies(dflash_verify_kv8_test ${B70_DFLASH_KERNELS} ${_v19b_reused} ${B70_DFLASH_VERIFY_KERNELS})
  add_test(NAME dflash_verify_kv8_test COMMAND dflash_verify_kv8_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(dflash_verify_kv8_test PROPERTIES LABELS "checkpoint;dflash;kv8" SKIP_RETURN_CODE 77
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 5400)
endif()
# Through EngineAdapter (what b70-serve's loop calls): D2 at K 7 / 3 / auto against the plain loop on the golden and A4
# prompts, truncate_to inside a burst, seeded sampling reproducible. SKIP 77 without either checkpoint.
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP AND B70_TOKENIZER_ENABLED)
  add_executable(dflash_gpu_test server/dflash_gpu_test.cc)
  target_include_directories(dflash_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(dflash_gpu_test PRIVATE b70_server b70_tokenizer b70_dflash_runtime b70_prefill_host
    b70_runtime b70_loader b70_model b70_l0)
  add_dependencies(dflash_gpu_test ${B70_DFLASH_KERNELS} ${_v19b_reused} ${B70_DFLASH_VERIFY_KERNELS})
  add_test(NAME dflash_gpu_test COMMAND dflash_gpu_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(dflash_gpu_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77
    ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 7200)
  # Spec 19f (Review Focus 3-4): prompt lookup inside DFlash through EngineAdapter - D2 against the plain loop on golden, A4
  # and the edit / copy set's 8k / 32k prompts, the warm ring (bitwise the plain loop's after lookup runs), a seeded
  # sampled lookup run reproducible.
  add_executable(dflash_lookup_gpu_test server/dflash_lookup_gpu_test.cc)
  target_include_directories(dflash_lookup_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(dflash_lookup_gpu_test PRIVATE b70_server b70_tokenizer b70_dflash_runtime b70_prefill_host
    b70_runtime b70_loader b70_model b70_l0)
  add_dependencies(dflash_lookup_gpu_test ${B70_DFLASH_KERNELS} ${_v19b_reused} ${B70_DFLASH_VERIFY_KERNELS})
  add_test(NAME dflash_lookup_gpu_test COMMAND dflash_lookup_gpu_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(dflash_lookup_gpu_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77
    ENVIRONMENT "B70_FORCED_JUDGE=${B70_QWEN_FORCED_JUDGE_I8HEAD};B70_FORCED_BF16=${B70_QWEN_FORCED_BF16TOP}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 7200)
endif()
# b70-decode --spec dflash's refusals before the device.
b70_cli_reject(cli_reject_dflash_bench ${B70_NO_MODEL} --bench --spec dflash EXPECT "--spec belongs to --ids")
b70_cli_reject(cli_reject_dflash_spec_max ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --spec dflash --spec-max 9
  EXPECT "expects 1..7")
b70_cli_reject(cli_reject_dflash_kolibri ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --ids ${B70_OK_IDS} --n 4
  --spec dflash EXPECT "b70-decode runs Kolibri-1 plainly")
# spec 19d: b70-decode stays one card (b70-serve --parallel pp serves the drafter).
b70_cli_reject(cli_reject_dflash_pp ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --spec dflash --parallel pp
  --draft-model urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound EXPECT "b70-decode --spec dflash runs on one card")
# Qwen3.8's --spec dflash has no default drafter: the refusal names the two choices.
b70_cli_reject(cli_reject_dflash_no_draft_model ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --spec dflash
  EXPECT "needs --draft-model")
set_tests_properties(cli_reject_dflash_bench cli_reject_dflash_spec_max
  cli_reject_dflash_kolibri cli_reject_dflash_pp cli_reject_dflash_no_draft_model PROPERTIES LABELS dflash)
# b70-serve --spec dflash's refusals before the device (cli/dflash_args.h).
if(TARGET b70-serve AND COMMAND b70_serve_reject)
  b70_serve_reject(cli_reject_serve_dflash_model_alone ${B70_NO_MODEL} --draft-model x EXPECT "needs --spec dflash")
  b70_serve_reject(cli_reject_serve_dflash_spec_max ${B70_NO_MODEL} --spec dflash --spec-max 8 EXPECT "expects 1..7")
  set_tests_properties(cli_reject_serve_dflash_model_alone cli_reject_serve_dflash_spec_max
    PROPERTIES LABELS dflash)
  b70_serve_reject(cli_reject_serve_dflash_no_draft_model ${B70_NO_MODEL} --spec dflash
    EXPECT "needs --draft-model")
  b70_serve_reject(cli_reject_serve_dflash_no_draft_model_hint ${B70_NO_MODEL} --spec dflash
    EXPECT "--draft-model urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound (our packed int4 drafter")
  b70_serve_reject(cli_reject_serve_spec_value ${B70_NO_MODEL} --spec lookup
    EXPECT "--spec expects auto, off, mtp or dflash, got 'lookup'")
  set_tests_properties(cli_reject_serve_spec_value cli_reject_serve_dflash_no_draft_model
    cli_reject_serve_dflash_no_draft_model_hint
    PROPERTIES LABELS dflash)
endif()
# ==== Spec 19c ==== (end) ====
# ==== Spec 19d: DFlash in the server ==== (begin) ====
# Host: the drafter's ring across a prefix-cache
# restore through PrefixSession (decision 5), the store keyed by the drafter. On the card: the same.
add_executable(dflash_prefix_test server/dflash_prefix_test.cc)
target_include_directories(dflash_prefix_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_prefix_test PRIVATE b70_server)
add_test(NAME dflash_prefix_test COMMAND dflash_prefix_test)
set_tests_properties(dflash_prefix_test PROPERTIES LABELS dflash)
# Task 2: speculative decoding under the batching scheduler (one drafter ring per slot) against the mock batched
# engine's speculative mode. Host only.
add_executable(dflash_scheduler_test server/dflash_scheduler_test.cc)
target_include_directories(dflash_scheduler_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dflash_scheduler_test PRIVATE b70_server)
add_test(NAME dflash_scheduler_test COMMAND dflash_scheduler_test)
set_tests_properties(dflash_scheduler_test PROPERTIES LABELS dflash)
# --spec dflash --parallel pp without a device: PipelineEngineAdapterT's DFlash path over a fake two-card engine with a tap
# drafter (lossless greedy, seeded sampling, a restored turn drafting as the uninterrupted one, the refusals).
add_executable(pp_dflash_serve_test server/pp_dflash_serve_test.cc)
target_include_directories(pp_dflash_serve_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pp_dflash_serve_test PRIVATE b70_server b70_plan)
add_test(NAME pp_dflash_serve_test COMMAND pp_dflash_serve_test)
set_tests_properties(pp_dflash_serve_test PROPERTIES LABELS dflash)
# The same on the card, through PrefixSession over EngineAdapter (--spec dflash): the ring after a restore bitwise the
# uninterrupted session's (and a cold run's after a block-end restore with a long tail), the continuation's ids and
# per-iteration acceptance identical; a cold-ring control printed. SKIP 77 without either checkpoint.
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP AND B70_TOKENIZER_ENABLED)
  add_executable(dflash_prefix_gpu_test server/dflash_prefix_gpu_test.cc)
  target_include_directories(dflash_prefix_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(dflash_prefix_gpu_test PRIVATE b70_server b70_tokenizer b70_dflash_runtime b70_prefill_host
    b70_runtime b70_loader b70_model b70_l0)
  add_dependencies(dflash_prefix_gpu_test ${B70_DFLASH_KERNELS} ${_v19b_reused} ${B70_DFLASH_VERIFY_KERNELS})
  add_test(NAME dflash_prefix_gpu_test COMMAND dflash_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(dflash_prefix_gpu_test PROPERTIES LABELS "checkpoint;dflash" SKIP_RETURN_CODE 77
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
endif()
# --spec dflash --parallel pp on two cards, BITWISE the one-card drafter (the copy hand-off at the auto split and at 20): the
# prefill's ring, every verify's rows, the drafts and ids, the session and the hooked snapshots; a one-card snapshot
# continued on two cards. SKIP 77 with fewer than two GPUs, without peer access or either checkpoint.
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(pp_dflash_test runtime/pp_dflash_test.cc)
  target_include_directories(pp_dflash_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(pp_dflash_test PRIVATE b70_dflash_runtime b70_prefill_host b70_runtime b70_loader b70_l0
    b70_model)
  add_dependencies(pp_dflash_test ${B70_DFLASH_KERNELS} ${_v19b_reused} ${B70_DFLASH_VERIFY_KERNELS})
  add_test(NAME pp_dflash_test COMMAND pp_dflash_test ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
  set_tests_properties(pp_dflash_test PROPERTIES LABELS "checkpoint;dflash;pp" SKIP_RETURN_CODE 77
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 5400)
endif()
# Spec 19v3a: same checkpoint tests over the int8 cache. Keep the original environment and timeouts; the judge is the
# int8 KV cache's (B70_QWEN_FORCED_JUDGE_I8HEAD_KV8 in place of _I8HEAD).
if(B70_KV8 AND B70_VERIFY_DPAS)
  foreach(t dflash_engine_test dflash_gpu_test dflash_prefix_gpu_test pp_dflash_test)
    if(TARGET ${t})
      string(REPLACE "_test" "_kv8_test" twin ${t})
      add_test(NAME ${twin} COMMAND ${t} ${B70_TEST_SNAPSHOT} ${B70_DFLASH_SNAPSHOT})
      get_test_property(${t} TIMEOUT _df_timeout)
      get_test_property(${t} LABELS _df_labels)
      get_test_property(${t} ENVIRONMENT _df_env)
      set_tests_properties(${twin} PROPERTIES LABELS "${_df_labels};kv8" SKIP_RETURN_CODE 77
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT ${_df_timeout} ENVIRONMENT "B70_KV_CACHE=int8")
      if(_df_env)
        string(REPLACE "${B70_QWEN_FORCED_JUDGE_I8HEAD}" "${B70_QWEN_FORCED_JUDGE_I8HEAD_KV8}" _df_env "${_df_env}")
        set_property(TEST ${twin} APPEND PROPERTY ENVIRONMENT ${_df_env})
      endif()
      foreach(MM 5 6 7 8)
        add_dependencies(${t} kernel_attn_prep_M${MM}_KV8 kernel_attn_v2_M${MM}_T32_KV8
          kernel_attn_verify_dpas_M${MM}_T32_KV8)
      endforeach()
    endif()
  endforeach()
endif()
# ==== Spec 19d ==== (end) ====

# Spec 17a (plan 17a Task 1): probe_tp's tensor-parallel shapes (tools/probe/tp_shapes.h) from ModelDesc through a
# tp_rank / tp_size view - spec 17 §R2.5's per-card linears, 128 / 144 all-reduces of 20 KiB. Host only.
add_executable(tp_shapes_test kernels/tp_shapes_test.cc)
target_include_directories(tp_shapes_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/tools/probe)
target_link_libraries(tp_shapes_test PRIVATE b70_model)
add_test(NAME tp_shapes_test COMMAND tp_shapes_test)

# ==== PERF-53 (plan 21-QFN-v3b): Qwen3.8-Flash-Next's DFlash 2 drafter (labels qwen4exp;dflash) ==== (begin) ====
# The descriptor: the published config held to its kernels and to Flash-Next. Host only.
add_executable(flash_next_dflash_desc_test model/flash_next_dflash_desc_test.cc)
target_include_directories(flash_next_dflash_desc_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(flash_next_dflash_desc_test PRIVATE b70_dflash_model b70_qwen4exp_model)
add_test(NAME flash_next_dflash_desc_test
         COMMAND flash_next_dflash_desc_test ${CMAKE_CURRENT_SOURCE_DIR}/model/flash_next_dflash/config.json)
set_tests_properties(flash_next_dflash_desc_test PROPERTIES LABELS "qwen4exp;dflash")
# The loader arm at kFlashNext (loader::flash_next): the published config, the shape set's linear pairs, a tiny
# synthetic checkpoint in both load forms (Int8 at load, packed Int4G64). Host only. With the release in the HF
# cache, the header check runs against it too (the snapshot as the test's second argument).
add_executable(flash_next_dflash_loader_test loader/flash_next_dflash_loader_test.cc)
target_include_directories(flash_next_dflash_loader_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(flash_next_dflash_loader_test PRIVATE b70_dflash_loader b70_qwen4exp_model)
if(NOT DEFINED CACHE{B70_FLASH_NEXT_DFLASH2})
  b70_hf_snapshot(B70_FLASH_NEXT_DFLASH2 models--incoai--Qwen3.8-Flash-Next-DFlash2
    "" "" PATH "Qwen3.8-Flash-Next-DFlash2 snapshot for the loader arm's header check (empty: the synthetic parts only)")
  if(NOT EXISTS "${B70_FLASH_NEXT_DFLASH2}")
    set(B70_FLASH_NEXT_DFLASH2 "" CACHE PATH "Qwen3.8-Flash-Next-DFlash2 snapshot for the loader arm's header check (empty: the synthetic parts only)" FORCE)
  endif()
endif()
if(B70_FLASH_NEXT_DFLASH2)
  add_test(NAME flash_next_dflash_loader_test COMMAND flash_next_dflash_loader_test
           ${CMAKE_CURRENT_SOURCE_DIR}/model/flash_next_dflash/config.json "${B70_FLASH_NEXT_DFLASH2}")
else()
  add_test(NAME flash_next_dflash_loader_test COMMAND flash_next_dflash_loader_test
           ${CMAKE_CURRENT_SOURCE_DIR}/model/flash_next_dflash/config.json)
endif()
set_tests_properties(flash_next_dflash_loader_test PROPERTIES LABELS "qwen4exp;dflash")
# Every binary the drafter's lists name is compiled (src/kernels/CMakeLists.txt's PERF-53 block and spec 19b's). Host
# only (reads the kernel directory).
if(B70_Q4EXP AND B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(flash_next_dflash_names_test kernels/flash_next_dflash_names_test.cc)
  target_include_directories(flash_next_dflash_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  b70_target_kernel_dir(flash_next_dflash_names_test)
  add_test(NAME flash_next_dflash_names_test COMMAND flash_next_dflash_names_test)
  set_tests_properties(flash_next_dflash_names_test PROPERTIES LABELS "qwen4exp;dflash")
endif()
# ==== PERF-53 (plan 21-QFN-v3b): Qwen3.8-Flash-Next's DFlash 2 drafter ==== (end) ====

# ==== Kolibri DSpark: speculative decoding for Kolibri-1 (label kolibri) ==== (begin) ====
# intuit/Kolibri-1-dspark beside Kolibri-1 (docs/17-running-models.md "Kolibri-1 with DSpark"). Host tests run on the Mac's
# build tree and the test machine's; the card tests wait for the first run (after Flash-Next).
# The descriptor: the speculators config -> the DSpark form, held to its kernels and to Kolibri-1. Host only.
add_executable(dspark_desc_test model/dspark_desc_test.cc)
target_include_directories(dspark_desc_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dspark_desc_test PRIVATE b70_dflash_model b70_kolibri_model)
add_test(NAME dspark_desc_test COMMAND dspark_desc_test ${CMAKE_CURRENT_SOURCE_DIR}/model/dspark/config.json)
set_tests_properties(dspark_desc_test PROPERTIES LABELS "kolibri;dflash")
# The loader's host half on a tiny DSpark checkpoint the test writes: names both ways, the Markov tables as the codebooks,
# the bytes formula, a target by its parts. Host only.
add_executable(dspark_loader_test loader/dspark_loader_test.cc)
target_include_directories(dspark_loader_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dspark_loader_test PRIVATE b70_dflash_loader)
add_test(NAME dspark_loader_test COMMAND dspark_loader_test)
set_tests_properties(dspark_loader_test PROPERTIES LABELS "kolibri;dflash")
# The drafter's buffers, launch counts, snapshot bytes and key (runtime/dflash/dflash_sizes.h). Host only.
add_executable(dspark_plan_test runtime/dspark_plan_test.cc)
target_include_directories(dspark_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dspark_plan_test PRIVATE b70_dflash_plan b70_dflash_loader)
add_test(NAME dspark_plan_test COMMAND dspark_plan_test ${CMAKE_CURRENT_SOURCE_DIR}/model/dspark/config.json)
set_tests_properties(dspark_plan_test PROPERTIES LABELS "kolibri;dflash")
# Kolibri's verify lists in the plan: the scratch at 8 rows, the second Control block, the verify link, the launch counts,
# the drafter's prefill tail rule, and the two-card plan with the drafter and its replica on device 1 (DERIVED). Host only.
add_executable(kolibri_spec_plan_test runtime/kolibri_spec_plan_test.cc)
target_include_directories(kolibri_spec_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri_spec_plan_test PRIVATE b70_kolibri_plan)
add_test(NAME kolibri_spec_plan_test COMMAND kolibri_spec_plan_test)
set_tests_properties(kolibri_spec_plan_test PROPERTIES LABELS kolibri)
# The adapter's speculative loop over a device-free engine and a fake drafter: greedy step_many == plain step() token for token
# at every K (the masked bonus ids among them), truncate_to inside a run, the deferred commit, the sampled path's seed and
# defined ids. Host only.
add_executable(kolibri_spec_test server/kolibri_spec_test.cc)
target_include_directories(kolibri_spec_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri_spec_test PRIVATE b70_server)
add_test(NAME kolibri_spec_test COMMAND kolibri_spec_test)
set_tests_properties(kolibri_spec_test PROPERTIES LABELS kolibri)
# On the cards (GPU only, NOT YET RUN: the first card run after Flash-Next; SKIP 77 without the data):
#   kolibri_verify_test   the verify lists against the plain decode list - every row's argmax the plain run's, its logits row
#                         cosine >= 0.99999 (bitwise expected, counted), partial commits, greedy lossless with the plain
#                         run's own ids as drafts, the launches at M 2..8 - on the synthetic set (one card, two) and the real
#                         checkpoint on two cards;
#   kolibri_dspark_test   the real Kolibri-1 with the DSpark drafter through b70-serve's glue: greedy step_many == plain token
#                         for token (D2), the tokens a verify (the drafter wired right), a seeded sampled request.
set(B70_DSPARK_SNAPSHOT "intuit/Kolibri-1-dspark" CACHE STRING "Kolibri-1's DSpark drafter (Kolibri DSpark)")
if(B70_KOLIBRI AND B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(kolibri_verify_test runtime/kolibri_verify_test.cc)
  target_include_directories(kolibri_verify_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(kolibri_verify_test PRIVATE b70_kolibri_prefill b70_l0 b70_server)
  add_executable(kolibri_dspark_test runtime/kolibri_dspark_test.cc)
  target_include_directories(kolibri_dspark_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/runtime)
  target_link_libraries(kolibri_dspark_test PRIVATE b70_kolibri_prefill b70_dflash_runtime b70_l0 b70_server)
  b70_target_kernel_dir(kolibri_dspark_test)
  set(_kds_synth ${B70_KOLIBRI_SYNTH_DIR}/int4attn/ckpt)
  set(_kds_ids ${CMAKE_SOURCE_DIR}/tests/golden/prompts/kolibri_bench.ids)
  add_test(NAME kolibri_verify_synth_test COMMAND kolibri_verify_test ${_kds_synth} ${_kds_ids})
  add_test(NAME kolibri_verify_synth_pp_test COMMAND kolibri_verify_test ${_kds_synth} ${_kds_ids} pp2:3)
  add_test(NAME kolibri_verify_test COMMAND kolibri_verify_test ${B70_KOLIBRI_SNAPSHOT} ${_kds_ids} pp2 int8)
  add_test(NAME kolibri_dspark_test COMMAND kolibri_dspark_test ${B70_KOLIBRI_SNAPSHOT} ${B70_DSPARK_SNAPSHOT} ${_kds_ids})
  set_tests_properties(kolibri_verify_synth_test kolibri_verify_synth_pp_test kolibri_verify_test kolibri_dspark_test
    PROPERTIES LABELS "checkpoint;kolibri" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# b70-serve's --spec refusals for Kolibri-1, before the device (tests/model/kolibri1: config.json only; the drafter's checks
# read tests/model/dspark: config.json only, so a drafter refusal is the safetensors' absence or the config's own).
if(TARGET b70-serve AND COMMAND b70_serve_reject)
  set(_kds_cfg ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1)
  b70_serve_reject(cli_reject_serve_kolibri_dspark_no_model ${_kds_cfg} --spec dflash EXPECT "intuit/Kolibri-1-dspark")
  b70_serve_reject(cli_reject_serve_kolibri_dspark_spec_max ${_kds_cfg} --spec dflash --spec-max 8
    --draft-model ${CMAKE_CURRENT_SOURCE_DIR}/model/dspark EXPECT "expects 1..7")
  set_tests_properties(cli_reject_serve_kolibri_dspark_no_model
    cli_reject_serve_kolibri_dspark_spec_max PROPERTIES LABELS kolibri)
endif()
# Every binary the verify lists and the drafter's lists name is compiled (src/kernels/CMakeLists.txt's Kolibri DSpark
# block). Host only (reads the kernel directory).
if(B70_KOLIBRI AND B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(dspark_names_test kernels/dspark_names_test.cc)
  target_include_directories(dspark_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(dspark_names_test PRIVATE b70_kolibri_model)
  b70_target_kernel_dir(dspark_names_test)
  add_test(NAME dspark_names_test COMMAND dspark_names_test)
  set_tests_properties(dspark_names_test PROPERTIES LABELS "kolibri;dflash")
endif()
# ==== Kolibri DSpark ==== (end) ====
# Fix round A: every test executable declared above gets a line-buffered stdout
# (common/line_buffered_stdout.cc - a static initializer, so no main changes): crash logs
# keep the phase line printed just before the crash, under any launcher. Keep this last.
get_property(_b70_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _b70_test_targets)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_sources(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common/line_buffered_stdout.cc)
  endif()
endforeach()

# tools/test_launcher.sh names a test's trace after the test.
get_property(_b70_all_tests DIRECTORY PROPERTY TESTS)
foreach(_t IN LISTS _b70_all_tests)
  set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT B70_TEST_NAME=${_t})
endforeach()

add_executable(expert_fp_test loader/expert_fp_test.cc)
target_include_directories(expert_fp_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(expert_fp_test PRIVATE b70_loader)
add_test(NAME expert_fp_test COMMAND expert_fp_test)
# The Inkling tests' model snapshots: a -D value always wins; otherwise the HF cache's refs/main
# snapshot of the model (b70_hf_snapshot, like B70_TEST_SNAPSHOT above) - the drafter preferring our
# packed int4 export over the bf16 release. Not in the cache: empty, the tests stay unregistered.
if(NOT B70_INKLING_SNAPSHOT)
  b70_hf_snapshot(B70_INKLING_SNAPSHOT models--thinkingmachines--Inkling-2-Mini
    b85c3cfa615fc413cc9af3379f491b963f3b586a "" PATH "Inkling-2-Mini snapshot for the sampled expert format test")
  if(NOT EXISTS "${B70_INKLING_SNAPSHOT}")
    set(B70_INKLING_SNAPSHOT "" CACHE PATH "Inkling-2-Mini snapshot for the sampled expert format test" FORCE)
  endif()
endif()
if(NOT B70_INKLING_DFLASH2)
  b70_hf_snapshot(B70_INKLING_DFLASH2 models--urakozz--Inkling-2-Mini-DFlash2-W4A16-g64-AutoRound
    8844a79cdd01d2d6094d8abea416ab6346fee112 "" PATH "the DFlash 2 drafter's snapshot (the DFlash 2 drafter's tests; empty: skipped)")
  if(NOT EXISTS "${B70_INKLING_DFLASH2}")
    b70_hf_snapshot(B70_INKLING_DFLASH2 models--incoai--Inkling-2-Mini-DFlash2
      4f9088ea66c21f2b99ed657d1941c306b03e84d2 "" PATH "the DFlash 2 drafter's snapshot (the DFlash 2 drafter's tests; empty: skipped)")
  endif()
  if(NOT EXISTS "${B70_INKLING_DFLASH2}")
    set(B70_INKLING_DFLASH2 "" CACHE PATH "the DFlash 2 drafter's snapshot (the DFlash 2 drafter's tests; empty: skipped)" FORCE)
  endif()
endif()
# The bf16 release on its own (the int8-at-load path's loader test checks its bytes): a -D value wins, else the cache's.
if(NOT B70_INKLING_DFLASH2_BF16)
  b70_hf_snapshot(B70_INKLING_DFLASH2_BF16 models--incoai--Inkling-2-Mini-DFlash2
    4f9088ea66c21f2b99ed657d1941c306b03e84d2 "" PATH "the bf16 DFlash 2 release's snapshot (the int8-at-load loader test; empty: skipped)")
  if(NOT EXISTS "${B70_INKLING_DFLASH2_BF16}")
    set(B70_INKLING_DFLASH2_BF16 "" CACHE PATH "the bf16 DFlash 2 release's snapshot (the int8-at-load loader test; empty: skipped)" FORCE)
  endif()
endif()
message(STATUS "Inkling model snapshots: Inkling-2-Mini=${B70_INKLING_SNAPSHOT} DFlash2=${B70_INKLING_DFLASH2} DFlash2 bf16=${B70_INKLING_DFLASH2_BF16}")
if(B70_INKLING_SNAPSHOT)
  add_test(NAME expert_fp_release_test COMMAND expert_fp_test "${B70_INKLING_SNAPSHOT}")
endif()

add_executable(expert_fp_gpu_test kernels/expert_fp_gpu_test.cc)
target_include_directories(expert_fp_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(expert_fp_gpu_test PRIVATE b70_loader b70_l0)
b70_target_kernel_dir(expert_fp_gpu_test)
add_dependencies(expert_fp_gpu_test kernel_expert_fp_bytes_NVFP4 kernel_expert_fp_bytes_MXFP8)
foreach(fmt NVFP4 MXFP8)
  add_dependencies(expert_fp_gpu_test kernel_kol_moe_E128_T8_D2304_I1152_${fmt}
                   kernel_pf_moe_q4gemm_K2304_N2304_SILU_${fmt} kernel_pf_moe_q4gemm_K1152_N2304_${fmt})
endforeach()
add_test(NAME expert_fp_gpu_test COMMAND expert_fp_gpu_test)

add_executable(inkling2_pf_moe_test kernels/inkling2_pf_moe_test.cc)
target_include_directories(inkling2_pf_moe_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_pf_moe_test PRIVATE b70_inkling2_loader b70_l0)
b70_target_kernel_dir(inkling2_pf_moe_test)
add_dependencies(inkling2_pf_moe_test kernel_kol_route_E128_T8_N128_L128_RENORM
                 kernel_pf_moe_sort_mw_E128_T8_INK kernel_kol_pf_moe_E128_T8_D2304_I1152_INK)
foreach(fmt NVFP4 MXFP8)
  add_dependencies(inkling2_pf_moe_test kernel_pf_moe_q4gemm_K2304_N2304_F32_INK_${fmt}
                   kernel_pf_moe_q4gemm_K1152_N2304_F32_INK_${fmt})
endforeach()
add_test(NAME inkling2_pf_moe_test COMMAND inkling2_pf_moe_test)

# Plan 41e: the int8 prompt pass's linears (rotation, quantiser, pf_gemm_i8) against the fp32 product.
add_executable(inkling2_int8_linear_test kernels/inkling2_int8_linear_test.cc)
target_include_directories(inkling2_int8_linear_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_int8_linear_test PRIVATE b70_l0)
b70_target_kernel_dir(inkling2_int8_linear_test)
add_dependencies(inkling2_int8_linear_test kernel_ink_int8_K2304 kernel_ink_int8_K3072 kernel_pf_gemm_i8)
add_test(NAME inkling2_int8_linear_test COMMAND inkling2_int8_linear_test)

add_executable(inkling2_pf_attn_test kernels/inkling2_pf_attn_test.cc)
target_include_directories(inkling2_pf_attn_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_pf_attn_test PRIVATE b70_l0)
b70_target_kernel_dir(inkling2_pf_attn_test)
add_dependencies(inkling2_pf_attn_test kernel_kol_pf_flash_attn_Q24KV8_REL16_SINK_GATE_W1024_R4096
                 kernel_kol_pf_flash_attn_Q24KV8_REL16_SINK_GATE_F kernel_ink_prep_Q24KV8_D128_REL16_PREFILL
                 kernel_kol_pf_flash_attn_Q24KV8_REL16_SINK_GATE_W1024_R4096_PAIR
                 kernel_kol_pf_flash_attn_Q24KV8_REL16_SINK_GATE_F_PAIR)
add_test(NAME inkling2_pf_attn_test COMMAND inkling2_pf_attn_test)

# PERF-102 (plan 41d): the int8 cache's writer bitwise kv8.h at head_dim 128, its decode and prompt readers against a
# double twin.
add_executable(inkling2_kv8_test kernels/inkling2_kv8_test.cc)
target_include_directories(inkling2_kv8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_kv8_test PRIVATE b70_l0)
b70_target_kernel_dir(inkling2_kv8_test)
add_dependencies(inkling2_kv8_test kernel_ink_prep_Q24KV8_D128_REL16_FUSED kernel_ink_prep_Q24KV8_D128_REL16_FUSED_KV8
                 kernel_kol_attn_M1_T32_Q24KV8_REL16E1024_SINK_GATE_F_KV8
                 kernel_kol_attn_M8_T32_Q24KV8_REL16E1024_SINK_GATE_F_KV8
                 kernel_kol_pf_flash_attn_Q24KV8_REL16_SINK_GATE_F_KV8
                 kernel_kol_pf_flash_attn_Q24KV8_REL16_SINK_GATE_F_KV8_PAIR
                 kernel_kol_pf_flash_attn_Q24KV8_REL16_SINK_GATE_F_KV8_I8S_PAIR)
add_test(NAME inkling2_kv8_test COMMAND inkling2_kv8_test)

add_executable(inkling2_chunks_test runtime/inkling2_chunks_test.cc)
target_include_directories(inkling2_chunks_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_chunks_test PRIVATE b70_inkling2_runtime)
foreach(name IN LISTS B70_INKLING_KERNELS B70_INKLING_PREFILL_KERNELS)
  add_dependencies(inkling2_chunks_test kernel_${name})
endforeach()
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_chunks_test COMMAND inkling2_chunks_test "${B70_INKLING_SNAPSHOT}")
endif()

add_executable(inkling2_host_test loader/inkling2_host_test.cc)
target_include_directories(inkling2_host_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_host_test PRIVATE b70_inkling2_plan)
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_host_test COMMAND inkling2_host_test "${B70_INKLING_SNAPSHOT}")
endif()

add_executable(inkling2_prefill_plan_test runtime/inkling2_prefill_plan_test.cc)
target_include_directories(inkling2_prefill_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_prefill_plan_test PRIVATE b70_inkling2_plan)
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_prefill_plan_test COMMAND inkling2_prefill_plan_test "${B70_INKLING_SNAPSHOT}")
endif()

add_executable(inkling2_pp_plan_test runtime/inkling2_pp_plan_test.cc)
target_include_directories(inkling2_pp_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_pp_plan_test PRIVATE b70_inkling2_plan)
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_pp_plan_test COMMAND inkling2_pp_plan_test "${B70_INKLING_SNAPSHOT}")
endif()

# `--parallel tp`'s placement and plan (host, the snapshot's headers).
add_executable(inkling2_tp_plan_test runtime/inkling2_tp_plan_test.cc)
target_include_directories(inkling2_tp_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_tp_plan_test PRIVATE b70_inkling2_plan)
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_tp_plan_test COMMAND inkling2_tp_plan_test "${B70_INKLING_SNAPSHOT}")
endif()

add_executable(inkling2_variant_names_test kernels/inkling2_variant_names_test.cc)
target_include_directories(inkling2_variant_names_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_variant_names_test PRIVATE b70_inkling2_model)
add_test(NAME inkling2_variant_names_test COMMAND inkling2_variant_names_test ${B70_INKLING_KERNELS})

# ==== Plan 41k: Inkling 2 Mini's DFlash 2 drafter (label inkling) ==== (begin) ====
# The descriptor: the published config held to its kernels and to Inkling. Host only.
add_executable(inkling2_dflash_desc_test model/inkling2_dflash_desc_test.cc)
target_include_directories(inkling2_dflash_desc_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_dflash_desc_test PRIVATE b70_dflash_model)
add_test(NAME inkling2_dflash_desc_test
         COMMAND inkling2_dflash_desc_test ${CMAKE_CURRENT_SOURCE_DIR}/model/inkling2_dflash/config.json)
set_tests_properties(inkling2_dflash_desc_test PROPERTIES LABELS inkling)
# Every binary the drafter's lists name is compiled (src/kernels/CMakeLists.txt's plan 41k block and spec 19b's). Host
# only (reads the kernel directory).
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(inkling2_dflash_names_test kernels/inkling2_dflash_names_test.cc)
  target_include_directories(inkling2_dflash_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(inkling2_dflash_names_test PRIVATE b70_inkling2_model)
  b70_target_kernel_dir(inkling2_dflash_names_test)
  add_test(NAME inkling2_dflash_names_test COMMAND inkling2_dflash_names_test)
  set_tests_properties(inkling2_dflash_names_test PROPERTIES LABELS inkling)
endif()
# The loader arm on the real checkpoints (the drafter in DSpark's place, its device form bitwise; refusals). Host.
add_executable(inkling2_dflash_loader_test loader/inkling2_dflash_loader_test.cc)
target_include_directories(inkling2_dflash_loader_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_dflash_loader_test PRIVATE b70_inkling2_loader)
if(B70_INKLING_SNAPSHOT AND B70_INKLING_DFLASH2_BF16)
  add_test(NAME inkling2_dflash_loader_test
           COMMAND inkling2_dflash_loader_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_DFLASH2_BF16}")
  set_tests_properties(inkling2_dflash_loader_test PROPERTIES LABELS inkling)
endif()
# The draft list on the card against the CPU reference on a tap dump's anchors: the code and chat sets'
# $B70_ORACLE/inkling/df2-drafts/<set>.txt (dflash_accept.py drafts' text; the test skips 77 without it).
add_executable(inkling2_dflash_draft_test golden/inkling2_dflash_draft_test.cc)
target_include_directories(inkling2_dflash_draft_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_dflash_draft_test PRIVATE b70_inkling2_runtime)
if(B70_INKLING_SNAPSHOT AND B70_INKLING_DFLASH2)
  foreach(set code chat)
    add_test(NAME inkling2_dflash_draft_${set}_test COMMAND inkling2_dflash_draft_test "${B70_INKLING_SNAPSHOT}"
             --draft-model "${B70_INKLING_DFLASH2}" "${B70_ORACLE}/inkling/df2-drafts/${set}.txt")
    set_tests_properties(inkling2_dflash_draft_${set}_test PROPERTIES LABELS "inkling;gpu" SKIP_RETURN_CODE 77)
  endforeach()
endif()
# ==== Plan 41k: Inkling 2 Mini's DFlash 2 drafter ==== (end) ====

add_executable(inkling2_ops_test kernels/inkling2_ops_test.cc)
target_include_directories(inkling2_ops_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME inkling2_ops_test COMMAND inkling2_ops_test)

# Plan 21-QFN-v3e Task 1: the taps kernel's host twin (tests/kernels/tap_mean_ref.h, the same arithmetic as
# kernels/tap_mean.cl) on random data. Host only; the card test comes with the lists (Task 2).
add_executable(tap_mean_test kernels/tap_mean_test.cc)
target_include_directories(tap_mean_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME tap_mean_test COMMAND tap_mean_test)
set_tests_properties(tap_mean_test PROPERTIES LABELS "qwen4exp;dflash")

add_executable(inkling2_golden_test golden/inkling2_golden_test.cc)
target_include_directories(inkling2_golden_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_golden_test PRIVATE b70_inkling2_model b70_loader)
set(B70_INKLING_GOLDEN "" CACHE FILEPATH "Inkling FP32 CPU golden for the judge's self-check")
if(B70_INKLING_GOLDEN)
  add_test(NAME inkling2_golden_file_test COMMAND inkling2_golden_test "${B70_INKLING_GOLDEN}")
endif()

add_executable(inkling2_ops_gpu_test kernels/inkling2_ops_gpu_test.cc)
target_include_directories(inkling2_ops_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_ops_gpu_test PRIVATE b70_l0)
b70_target_kernel_dir(inkling2_ops_gpu_test)
add_dependencies(inkling2_ops_gpu_test kernel_ink_glue_H2304 kernel_ink_glue_H2304_CTRL kernel_ink_glue_H2304_CTRL_SPLIT
                 kernel_ink_prep_Q24KV8_D128_REL16
                 kernel_kol_route_E128_T8_N128_L128_RENORM kernel_kol_route_E128_T8_N128_L128_RENORM_M1_FUSED
                 kernel_kol_attn_M1_T32_Q24KV8_REL16E1024_SINK_GATE_F
                 kernel_kol_moe_M2_E128_T8_D2304_I1152_CAP_NVFP4
                 kernel_kol_moe_M2_E128_T8_D2304_I1152_CAP_MXFP8)
add_test(NAME inkling2_ops_gpu_test COMMAND inkling2_ops_gpu_test)

add_executable(inkling2_decode_test golden/inkling2_decode_test.cc)
target_include_directories(inkling2_decode_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_decode_test PRIVATE b70_inkling2_runtime)
foreach(name IN LISTS B70_INKLING_KERNELS)
  add_dependencies(inkling2_decode_test kernel_${name})
endforeach()
set(B70_INKLING_PROMPT "" CACHE FILEPATH "Token IDs for the Inkling FP32 CPU golden")
# The pp / tp tests' short prompt after their synthetic context: any real ids; the oracle set's golden code prompt.
set(B70_INKLING_BRIEF "${B70_ORACLE}/inkling/golden/code.ids" CACHE FILEPATH "Token IDs the Inkling pp / tp tests end with")
if(B70_INKLING_SNAPSHOT AND B70_INKLING_GOLDEN AND B70_INKLING_PROMPT)
  add_test(NAME inkling2_decode_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
           "${B70_INKLING_PROMPT}" "${B70_INKLING_GOLDEN}")
  add_test(NAME inkling2_draft_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
           "${B70_INKLING_PROMPT}" "${B70_INKLING_GOLDEN}" draft)
endif()

# Drafted ids equal plain ids after a real prompt's prompt pass: 200 ids (bf16) and ~30k (both caches). A verify row is
# not bitwise the one-row step, so drafted ids may leave plain's at a near-tie (the int8 cache's 200-id case does at
# id 7): these prompts are the ones that meet none in 128 ids; the golden ones below judge a difference.
if(B70_INKLING_SNAPSHOT)
  foreach(c "200;prose-followup-200;bf16" "30000;prose-32768;bf16 kv8")
    list(GET c 0 len)
    list(GET c 1 file)
    list(GET c 2 caches)
    separate_arguments(caches)
    foreach(kv IN LISTS caches)
      add_test(NAME inkling2_same_ids_${len}_${kv}_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}" same
               "${B70_ORACLE}/inkling/prose/${file}.ids" ${len} 128 ${kv})
      set_tests_properties(inkling2_same_ids_${len}_${kv}_test PROPERTIES LABELS inkling SKIP_RETURN_CODE 77)
    endforeach()
  endforeach()
  # The two prompts with an f32 golden, both caches: where drafted ids leave plain's, the reference row accepts both.
  foreach(g "compat;prose.compat" "4096;prose-4096.fp32")
    list(GET g 0 tag)
    list(GET g 1 file)
    foreach(kv bf16 kv8)
      add_test(NAME inkling2_same_ids_golden_${tag}_${kv}_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
               same-golden "${B70_ORACLE}/inkling/prose/${file}.golden.safetensors" ${kv})
      set_tests_properties(inkling2_same_ids_golden_${tag}_${kv}_test PROPERTIES LABELS inkling SKIP_RETURN_CODE 77)
    endforeach()
  endforeach()
  # The int8 cache's ids, plain and drafted, judged by its own f32 reference (--kv int8; skipped, 77, until it is made).
  foreach(s "code;golden/code" "chat;golden/chat" "agentic;golden/agentic" "prose4096;prose/prose-4096")
    list(GET s 0 tag)
    list(GET s 1 stem)
    # the bf16 cache by the plain f32 reference beside them: the int8 cache's comparison
    add_test(NAME inkling2_golden_${tag}_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
             "${B70_ORACLE}/inkling/${stem}.ids" "${B70_ORACLE}/inkling/${stem}.fp32.golden.safetensors")
    add_test(NAME inkling2_golden_${tag}_draft_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
             "${B70_ORACLE}/inkling/${stem}.ids" "${B70_ORACLE}/inkling/${stem}.fp32.golden.safetensors" draft)
    add_test(NAME inkling2_golden_${tag}_kv8_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
             "${B70_ORACLE}/inkling/${stem}.ids" "${B70_ORACLE}/inkling/${stem}.fp32-kv8.golden.safetensors" kv8)
    add_test(NAME inkling2_golden_${tag}_kv8_draft_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
             "${B70_ORACLE}/inkling/${stem}.ids" "${B70_ORACLE}/inkling/${stem}.fp32-kv8.golden.safetensors" draft kv8)
    set_tests_properties(inkling2_golden_${tag}_test inkling2_golden_${tag}_draft_test inkling2_golden_${tag}_kv8_test
                         inkling2_golden_${tag}_kv8_draft_test PROPERTIES LABELS inkling SKIP_RETURN_CODE 77)
  endforeach()
endif()

# Plan 41k: the same judgements with the DFlash 2 drafter in DSpark's place (dflash=DIR) - drafted ids equal plain or
# judged by the f32 golden, both caches, and the accept mode's one-submission round against the three calls.
if(B70_INKLING_SNAPSHOT AND B70_INKLING_DFLASH2)
  foreach(g "compat;prose.compat" "4096;prose-4096.fp32")
    list(GET g 0 tag)
    list(GET g 1 file)
    foreach(kv bf16 kv8)
      add_test(NAME inkling2_dflash_same_ids_golden_${tag}_${kv}_test COMMAND inkling2_decode_test
               "${B70_INKLING_SNAPSHOT}" same-golden "${B70_ORACLE}/inkling/prose/${file}.golden.safetensors" ${kv}
               "dflash=${B70_INKLING_DFLASH2}")
      set_tests_properties(inkling2_dflash_same_ids_golden_${tag}_${kv}_test PROPERTIES LABELS inkling
                           SKIP_RETURN_CODE 77)
    endforeach()
  endforeach()
  foreach(s "code;golden/code" "chat;golden/chat" "agentic;golden/agentic" "prose4096;prose/prose-4096")
    list(GET s 0 tag)
    list(GET s 1 stem)
    add_test(NAME inkling2_dflash_golden_${tag}_kv8_draft_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
             "${B70_ORACLE}/inkling/${stem}.ids" "${B70_ORACLE}/inkling/${stem}.fp32-kv8.golden.safetensors" draft kv8
             "dflash=${B70_INKLING_DFLASH2}")
    set_tests_properties(inkling2_dflash_golden_${tag}_kv8_draft_test PROPERTIES LABELS inkling SKIP_RETURN_CODE 77)
  endforeach()
  add_test(NAME inkling2_dflash_accept_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}" accept
           "${B70_ORACLE}/inkling/golden/code.ids" "${B70_ORACLE}/inkling/golden/chat.ids"
           "dflash=${B70_INKLING_DFLASH2}")
  set_tests_properties(inkling2_dflash_accept_test PROPERTIES LABELS inkling SKIP_RETURN_CODE 77)
endif()

# `--parallel pp` against one card, bitwise (both cards; skipped, 77, with one).
add_executable(inkling2_pp_test runtime/inkling2_pp_test.cc)
target_include_directories(inkling2_pp_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_pp_test PRIVATE b70_inkling2_runtime)
foreach(name IN LISTS B70_INKLING_KERNELS B70_INKLING_PREFILL_KERNELS)
  add_dependencies(inkling2_pp_test kernel_${name})
endforeach()
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_pp_test COMMAND inkling2_pp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}")
  add_test(NAME inkling2_pp_kv8_test COMMAND inkling2_pp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}" kv8)
  add_test(NAME inkling2_pp_int8_test COMMAND inkling2_pp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}" int8)
  set_tests_properties(inkling2_pp_test inkling2_pp_kv8_test inkling2_pp_int8_test PROPERTIES SKIP_RETURN_CODE 77)
  if(B70_INKLING_DFLASH2)   # plan 41k: the DFlash 2 drafter on card 1, both caches
    add_test(NAME inkling2_dflash_pp_test COMMAND inkling2_pp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}"
             "dflash=${B70_INKLING_DFLASH2}")
    add_test(NAME inkling2_dflash_pp_kv8_test COMMAND inkling2_pp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}"
             kv8 "dflash=${B70_INKLING_DFLASH2}")
    set_tests_properties(inkling2_dflash_pp_test inkling2_dflash_pp_kv8_test PROPERTIES LABELS inkling
                         SKIP_RETURN_CODE 77)
  endif()
endif()

# `--parallel tp` against itself where the order is kept (two cards; skipped, 77, with one); its ids against the f32
# judge are inkling2_tp_decode_test / inkling2_tp_draft_test.
add_executable(inkling2_tp_test runtime/inkling2_tp_test.cc)
target_include_directories(inkling2_tp_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_tp_test PRIVATE b70_inkling2_runtime)
foreach(name IN LISTS B70_INKLING_KERNELS B70_INKLING_PREFILL_KERNELS)
  add_dependencies(inkling2_tp_test kernel_${name})
endforeach()
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_tp_test COMMAND inkling2_tp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}")
  add_test(NAME inkling2_tp_kv8_test COMMAND inkling2_tp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}" kv8)
  add_test(NAME inkling2_tp_int8_test COMMAND inkling2_tp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}" int8)
  set_tests_properties(inkling2_tp_test inkling2_tp_kv8_test inkling2_tp_int8_test PROPERTIES SKIP_RETURN_CODE 77)
  if(B70_INKLING_DFLASH2)   # plan 41k: the DFlash 2 drafter whole on both cards
    add_test(NAME inkling2_dflash_tp_test COMMAND inkling2_tp_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_BRIEF}"
             "dflash=${B70_INKLING_DFLASH2}")
    set_tests_properties(inkling2_dflash_tp_test PROPERTIES LABELS inkling SKIP_RETURN_CODE 77)
  endif()
endif()
if(B70_INKLING_SNAPSHOT AND B70_INKLING_GOLDEN AND B70_INKLING_PROMPT)
  add_test(NAME inkling2_tp_decode_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
           "${B70_INKLING_PROMPT}" "${B70_INKLING_GOLDEN}" tp)
  add_test(NAME inkling2_tp_draft_test COMMAND inkling2_decode_test "${B70_INKLING_SNAPSHOT}"
           "${B70_INKLING_PROMPT}" "${B70_INKLING_GOLDEN}" draft tp)
  set_tests_properties(inkling2_tp_decode_test inkling2_tp_draft_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_executable(inkling2_prefill_test golden/inkling2_prefill_test.cc)
target_include_directories(inkling2_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_prefill_test PRIVATE b70_inkling2_runtime)
foreach(name IN LISTS B70_INKLING_KERNELS B70_INKLING_PREFILL_KERNELS)
  add_dependencies(inkling2_prefill_test kernel_${name})
endforeach()
if(B70_INKLING_SNAPSHOT AND B70_INKLING_GOLDEN)
  add_test(NAME inkling2_prefill_prose_test COMMAND inkling2_prefill_test "${B70_INKLING_SNAPSHOT}" "${B70_INKLING_GOLDEN}")
endif()
# Plan 41d's judge sets (tools/oracle/inkling_refs.sh; skipped, 77, until they are made): the golden prompts with the
# bf16 cache and every set with `--kv-cache int8` against its int8-aware references.
if(B70_INKLING_SNAPSHOT)
  set(_ink "${B70_ORACLE}/inkling")
  foreach(p code chat agentic)
    add_test(NAME inkling2_prefill_${p}_test COMMAND inkling2_prefill_test "${B70_INKLING_SNAPSHOT}"
             "${_ink}/golden/${p}.fp32.golden.safetensors" "${_ink}/golden/${p}.bf16.golden.safetensors")
    add_test(NAME inkling2_prefill_${p}_kv8_test COMMAND inkling2_prefill_test "${B70_INKLING_SNAPSHOT}"
             "${_ink}/golden/${p}.fp32-kv8.golden.safetensors" "${_ink}/golden/${p}.bf16-kv8.golden.safetensors" kv8)
    # Plan 41e: the int8 prompt pass (on the int8 cache) against the same int8-aware references.
    add_test(NAME inkling2_prefill_${p}_int8_test COMMAND inkling2_prefill_test "${B70_INKLING_SNAPSHOT}"
             "${_ink}/golden/${p}.fp32-kv8.golden.safetensors" "${_ink}/golden/${p}.bf16-kv8.golden.safetensors" int8)
    set_tests_properties(inkling2_prefill_${p}_test inkling2_prefill_${p}_kv8_test inkling2_prefill_${p}_int8_test
                         PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
  foreach(n 4096 32768)
    foreach(mode kv8 int8)
      add_test(NAME inkling2_prefill_prose${n}_${mode}_test COMMAND inkling2_prefill_test "${B70_INKLING_SNAPSHOT}"
               "${_ink}/prose/prose-${n}.fp32-kv8.golden.safetensors"
               "${_ink}/prose/prose-${n}.bf16-kv8.golden.safetensors" ${mode})
      set_tests_properties(inkling2_prefill_prose${n}_${mode}_test PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
  endforeach()
endif()
foreach(size 4K 32K)
  set(B70_INKLING_PREFILL_${size} "" CACHE FILEPATH "Inkling ${size} FP32 prompt golden")
  set(B70_INKLING_PREFILL_${size}_BF16 "" CACHE FILEPATH "Inkling ${size} BF16 prompt golden twin")
  if(B70_INKLING_SNAPSHOT AND B70_INKLING_PREFILL_${size} AND B70_INKLING_PREFILL_${size}_BF16)
    add_test(NAME inkling2_prefill_${size}_test COMMAND inkling2_prefill_test "${B70_INKLING_SNAPSHOT}"
             "${B70_INKLING_PREFILL_${size}}" "${B70_INKLING_PREFILL_${size}_BF16}")
  endif()
endforeach()
# Plan 41f Task 1: Inkling's chat template (thinkingmachines/Inkling-2-Mini at b85c3cfa's chat_template.jinja, vendored with
# its tokenizer_config / generation_config at tests/tokenizer/inkling2/, Apache-2.0) on 20 message lists (every effort level
# and a number, tools, a call history, tool results, reasoning, content parts, an image part; an unknown effort and an
# audio part refused by the template) against transformers' apply_chat_template (tools/tokenizer/dump_inkling2.py); and its
# tokenizer against HF's (ignore_merges words, the corpus, every render's ids, the A4 set's ids once the CPU queue made it).
if(B70_TOKENIZER_ENABLED)
  add_test(NAME template_inkling2_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/inkling2 inkling2_template_cases.json inkling2)
  b70_hf_snapshot(B70_INKLING2_TOKENIZER_JSON models--thinkingmachines--Inkling-2-Mini
    b85c3cfa615fc413cc9af3379f491b963f3b586a /tokenizer.json FILEPATH
    "Inkling 2 Mini's tokenizer.json for inkling2_tokenizer_test (-D or the environment variable)")
  add_executable(inkling2_tokenizer_test tokenizer/inkling2_tokenizer_test.cc)
  target_include_directories(inkling2_tokenizer_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(inkling2_tokenizer_test PRIVATE b70_tokenizer)
  add_test(NAME inkling2_tokenizer_test COMMAND inkling2_tokenizer_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/inkling2 ${B70_INKLING2_TOKENIZER_JSON} ${B70_ORACLE}/inkling/a4/set)
  set_tests_properties(template_inkling2_test inkling2_tokenizer_test PROPERTIES LABELS inkling SKIP_RETURN_CODE 77)
endif()
# The serial server's prompt encode (tok::Tokenizer::encode_extending: the ids of the previous prompt's shared start
# kept) equals encode on growing, edited, cut and unrelated conversations - Qwen3.8's (and Ornith's renders), Inkling's
# and Kolibri's tokenizers; each one absent is skipped.
if(B70_TOKENIZER_ENABLED)
  add_executable(tokenizer_extend_test tokenizer/tokenizer_extend_test.cc)
  target_include_directories(tokenizer_extend_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(tokenizer_extend_test PRIVATE b70_tokenizer)
  add_test(NAME tokenizer_extend_test COMMAND tokenizer_extend_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    default=template_ default=ornith_template_ ${B70_INKLING2_TOKENIZER_JSON}=inkling2_template_
    ${B70_KOLIBRI_TOKENIZER_JSON}=kolibri_template_)
  set_tests_properties(tokenizer_extend_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
# Plan 41f Task 3 (host, the checkpoint's headers): Inkling's served context on one card - with and without the drafter,
# bf16 and int8 cache (replica plans the same on each card), and the refusal of what does not fit.
add_executable(inkling2_plan_test runtime/inkling2_plan_test.cc)
target_include_directories(inkling2_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_plan_test PRIVATE b70_inkling2_plan b70_runtime)
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_plan_test COMMAND inkling2_plan_test "${B70_INKLING_SNAPSHOT}" ${B70_INKLING_DFLASH2})
  set_tests_properties(inkling2_plan_test PROPERTIES LABELS inkling)
endif()
# Plan 41f Task 4 (card): Inkling's prefix-cache snapshots through server::PrefixSession and the serving adapter - a
# follow-up turn, a fork after 4096, the same prompt again (the head image), --prefix-split-last and a reset after a
# long session, each bitwise its cold run, plain and drafted; bf16 and int8 cache.
add_executable(inkling2_snapshot_test runtime/inkling2_snapshot_test.cc)
target_include_directories(inkling2_snapshot_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(inkling2_snapshot_test PRIVATE b70_inkling2_runtime b70_server)
foreach(name IN LISTS B70_INKLING_KERNELS B70_INKLING_PREFILL_KERNELS)
  add_dependencies(inkling2_snapshot_test kernel_${name})
endforeach()
if(B70_INKLING_SNAPSHOT)
  add_test(NAME inkling2_snapshot_test COMMAND inkling2_snapshot_test "${B70_INKLING_SNAPSHOT}")
  add_test(NAME inkling2_snapshot_kv8_test COMMAND inkling2_snapshot_test "${B70_INKLING_SNAPSHOT}" kv8)
  add_test(NAME inkling2_snapshot_pp_test COMMAND inkling2_snapshot_test "${B70_INKLING_SNAPSHOT}" pp)
  add_test(NAME inkling2_snapshot_pp_kv8_test COMMAND inkling2_snapshot_test "${B70_INKLING_SNAPSHOT}" kv8 pp)
  set_tests_properties(inkling2_snapshot_test inkling2_snapshot_kv8_test inkling2_snapshot_pp_test
                       inkling2_snapshot_pp_kv8_test PROPERTIES LABELS inkling SKIP_RETURN_CODE 77
                       TIMEOUT 3600)
  if(B70_INKLING_DFLASH2)   # plan 41k: the DFlash 2 drafter's rings in the snapshots, one card, both caches
    add_test(NAME inkling2_dflash_snapshot_test COMMAND inkling2_snapshot_test "${B70_INKLING_SNAPSHOT}"
             "dflash=${B70_INKLING_DFLASH2}")
    add_test(NAME inkling2_dflash_snapshot_kv8_test COMMAND inkling2_snapshot_test "${B70_INKLING_SNAPSHOT}" kv8
             "dflash=${B70_INKLING_DFLASH2}")
    set_tests_properties(inkling2_dflash_snapshot_test inkling2_dflash_snapshot_kv8_test PROPERTIES LABELS inkling
                         SKIP_RETURN_CODE 77 TIMEOUT 3600)
  endif()
endif()
# Plan 41f Task 2 (host): Inkling's reasoning, text and JSON call blocks (server/toolcall_inkling2.h) - the template's
# round trip, drift, refusals, cuts and a 1000-output fuzz, every parse split-invariant.
add_executable(toolcall_inkling2_test server/toolcall_inkling2_test.cc)
target_include_directories(toolcall_inkling2_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(toolcall_inkling2_test PRIVATE b70_server)
add_test(NAME toolcall_inkling2_test COMMAND toolcall_inkling2_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
set_tests_properties(toolcall_inkling2_test PROPERTIES LABELS inkling)
# PERF-124: the host-RAM expert on the CPU (src/cpu): AVX2 (and AVX-512 where present) against the scalar fp32 twin. Host only.
add_executable(cpu_expert_test cpu/cpu_expert_test.cc)
target_include_directories(cpu_expert_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests/cpu)
target_link_libraries(cpu_expert_test PRIVATE b70_cpu)
add_test(NAME cpu_expert_test COMMAND cpu_expert_test)
set_tests_properties(cpu_expert_test PROPERTIES LABELS cpu TIMEOUT 600)
# PERF-124 step 4: the host-RAM expert from its layout-1 blocks (src/cpu/expert_q4.cc) against q4_moe.cl's host twin,
# bitwise. Host only.
add_executable(cpu_expert_q4_test cpu/cpu_expert_q4_test.cc)
target_include_directories(cpu_expert_q4_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(cpu_expert_q4_test PRIVATE b70_cpu b70_qwen4exp_loader)
add_test(NAME cpu_expert_q4_test COMMAND cpu_expert_q4_test)
set_tests_properties(cpu_expert_q4_test PROPERTIES LABELS cpu TIMEOUT 600)
