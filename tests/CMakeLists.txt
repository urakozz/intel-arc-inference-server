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

# The only test here that needs neither a GPU nor a kernel: device_index_from_env
# is a pure parse of ONEAPI_DEVICE_SELECTOR (doc 04, "Device selection").
add_executable(device_select_test l0/device_select_test.cc)
target_include_directories(device_select_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(device_select_test PRIVATE b70_l0)
add_test(NAME device_select_test COMMAND device_select_test)

# The prefill execution context (spec 2.1 §3.5). `Context` is g++-only and
# links only b70_l0 through b70_prefill_host, so this test is built and
# registered in EVERY configuration (ruling B4). In the enabled build it ALSO
# links libb70_prefill.so -- the test still proves !cx.has_sycl(), because
# linking the .so creates no SYCL queue by itself.
add_executable(context_test prefill/context_test.cc)
target_include_directories(context_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(context_test PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(context_test)
add_dependencies(context_test kernel_noop kernel_pf_probe_chain)
add_test(NAME context_test COMMAND context_test)
if(B70_PREFILL_ENABLED)
  b70_link_prefill(context_test)
endif()

if(B70_PREFILL_ENABLED)
  # The batched Xe bf16 GEMM (rulings A14/A15, plan 6d-composed Task 1). Same
  # guard: it calls into libb70_prefill.so, so with -DB70_PREFILL=OFF this
  # block does not exist and the suite is what it was. No kernel binaries -
  # the only device code it runs is sycl-tla's. `Context` and
  # `gemm_bf16_batched*` now live in b70_prefill_host (ruling B5).
  add_executable(gemm_batched_test prefill/gemm_batched_test.cc)
  target_include_directories(gemm_batched_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(gemm_batched_test PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(gemm_batched_test)
  add_test(NAME gemm_batched_test COMMAND gemm_batched_test)
  set_tests_properties(gemm_batched_test PROPERTIES TIMEOUT 1800)

  # Spec 2.1 S0: pf_gemm bitwise against sycl-tla at every prefill shape (bar 1).
  add_executable(pf_gemm_test prefill/pf_gemm_test.cc)
  target_include_directories(pf_gemm_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(pf_gemm_test PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(pf_gemm_test)
  b70_target_kernel_dir(pf_gemm_test)
  add_dependencies(pf_gemm_test kernel_pf_gemm_T0 kernel_pf_gemm_T1 kernel_pf_gemm_T0_SILU
                   kernel_pf_silu_mul)
  add_test(NAME pf_gemm_test COMMAND pf_gemm_test)
  set_tests_properties(pf_gemm_test PROPERTIES TIMEOUT 1800)

  add_executable(pf_int8_test prefill/pf_int8_test.cc)
  target_include_directories(pf_int8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(pf_int8_test PRIVATE b70_l0)
  b70_target_kernel_dir(pf_int8_test)
  add_dependencies(pf_int8_test kernel_pf_quant_had_K5120 kernel_pf_quant_had_K6144
                   kernel_pf_quant_had_K17408 kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1
                   kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU kernel_pf_quant_had_K19456)
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
  b70_link_prefill(linear_i8_test)
  b70_target_kernel_dir(linear_i8_test)
  add_dependencies(linear_i8_test kernel_pf_quant_had_K5120 kernel_pf_quant_had_K6144
                   kernel_pf_quant_had_K17408 kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1
                   kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU)
  add_test(NAME linear_i8_test COMMAND linear_i8_test)
endif()

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

# Spec 8 §3.6 (plan 8c): the server over a scripted speculative engine, host only.
add_executable(mtp_server_test server/mtp_server_test.cc)
target_include_directories(mtp_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(mtp_server_test PRIVATE b70_server)
add_test(NAME mtp_server_test COMMAND mtp_server_test)

# Spec 19e (plan 19e Task 2): the server with --spec lookup over a mock engine that verifies
# external drafts, host only.
add_executable(lookup_server_test server/lookup_server_test.cc)
target_include_directories(lookup_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(lookup_server_test PRIVATE b70_server)
add_test(NAME lookup_server_test COMMAND lookup_server_test)

# Spec 8 §10: the --mtp auto policy on synthetic acceptance sequences, host only.
add_executable(adaptive_k_test server/adaptive_k_test.cc)
target_include_directories(adaptive_k_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(adaptive_k_test PRIVATE b70_server)
add_test(NAME adaptive_k_test COMMAND adaptive_k_test)

# Spec 13 B4 (plan 13c Task 1): the batching scheduler over a mock batched engine, host only.
add_executable(scheduler_test server/scheduler_test.cc)
target_include_directories(scheduler_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(scheduler_test PRIVATE b70_server)
add_test(NAME scheduler_test COMMAND scheduler_test)

add_executable(protocol_test server/protocol_test.cc)
target_include_directories(protocol_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(protocol_test PRIVATE b70_server)
add_test(NAME protocol_test COMMAND protocol_test)

# P3 is raw Level Zero/OpenCL C only: it requires a B70 and its kernels but
# never the optional SYCL component, so it intentionally has no guard.
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
  kernel_pf_dequant_slab_K5120_N38912_L0 kernel_pf_dequant_slab_K19456_N5120_L0
  kernel_pf_dequant_tile_K5120_N38912_L0_T0 kernel_pf_dequant_tile_K19456_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N16384_L1_T0 kernel_pf_dequant_tile_K6144_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N34816_L0_T0 kernel_pf_dequant_tile_K17408_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N14336_L0_T0)
add_test(NAME pf_dequant_slab_test COMMAND pf_dequant_slab_test)
set_tests_properties(pf_dequant_slab_test PROPERTIES TIMEOUT 900)

# The runtime-`M`, S = 1 small kernels (spec 2, plan 6b Tasks 3-5). Like
# dequant_test these are raw Level Zero and OpenCL C: they need a B70 and their
# own binaries, never the optional SYCL component, so they carry no guard and no
# label. Each also loads the DECODE binary it must be bit-identical to at M = 1,
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

add_executable(pf_gemv_test prefill/pf_gemv_test.cc)
target_include_directories(pf_gemv_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(pf_gemv_test PRIVATE b70_l0)
b70_target_kernel_dir(pf_gemv_test)
add_dependencies(pf_gemv_test kernel_pf_ab_proj kernel_gemv_bf16_M1_K5120_N128_C16_S16)
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
# context, so they carry the same B70_PREFILL_ENABLED guard context_test does:
# with -DB70_PREFILL=OFF the whole gdn_chunk stream is absent and the suite is
# exactly what it was. No checkpoint - one synthetic GDN layer.
if(B70_PREFILL_ENABLED)
  add_executable(gdn_conv_test prefill/gdn_conv_test.cc)
  target_include_directories(gdn_conv_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(gdn_conv_test PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(gdn_conv_test)
  b70_target_kernel_dir(gdn_conv_test)
  add_dependencies(gdn_conv_test kernel_pf_gdn_conv)
  add_test(NAME gdn_conv_test COMMAND gdn_conv_test)
  set_tests_properties(gdn_conv_test PROPERTIES TIMEOUT 1800)

  add_executable(gdn_wy_test prefill/gdn_wy_test.cc)
  target_include_directories(gdn_wy_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(gdn_wy_test PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(gdn_wy_test)
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
  b70_link_prefill(gdn_chunk_test)
  b70_target_kernel_dir(gdn_chunk_test)
  add_dependencies(gdn_chunk_test kernel_pf_gdn_conv kernel_pf_gdn_wy kernel_pf_gdn_scan
    kernel_pf_gated_head kernel_gdn_step_M1 kernel_prep_gated_head_M1)
  add_test(NAME gdn_chunk_test COMMAND gdn_chunk_test)
  set_tests_properties(gdn_chunk_test PROPERTIES TIMEOUT 1800)
  add_test(NAME gdn_scan_invalid_selector_test COMMAND gdn_chunk_test --invalid-selector)
  # Since 2026-09-23 an unset selector binds the SPLIT scan (gdn.cc's
  # `gdn_scan_entry`). This row is what fails if that default moves without the
  # record and the gate evidence moving with it.
  add_test(NAME gdn_scan_default_dispatch_test COMMAND gdn_chunk_test --dispatch-split)
  add_test(NAME gdn_scan_vector_dispatch_test COMMAND gdn_chunk_test --dispatch-vector)
  add_test(NAME gdn_scan_split_dispatch_test COMMAND gdn_chunk_test --dispatch-split)
  set_tests_properties(gdn_scan_invalid_selector_test PROPERTIES ENVIRONMENT "B70_PREFILL_GDN_SCAN=bad")
  set_tests_properties(gdn_scan_default_dispatch_test PROPERTIES ENVIRONMENT "B70_PREFILL_GDN_SCAN=")
  set_tests_properties(gdn_scan_vector_dispatch_test PROPERTIES ENVIRONMENT "B70_PREFILL_GDN_SCAN=vector")
  set_tests_properties(gdn_scan_split_dispatch_test PROPERTIES ENVIRONMENT "B70_PREFILL_GDN_SCAN=dpas_split")

  # Stage S4's solve selector, registered parallel to the scan's four rows.
  # The dispatch rows observe `KernelCache::kernels()` rather than the output:
  # the two solve entries are bit-identical by construction, so an output
  # comparison could not tell which one ran (gdn_chunk_test.cc,
  # `check_solve_dispatch`). The full end-to-end run is registered a second
  # time under the register selector so every band, boundary and determinism
  # bar in the file is carried under both.
  add_test(NAME gdn_solve_invalid_selector_test COMMAND gdn_chunk_test --invalid-selector)
  add_test(NAME gdn_solve_default_dispatch_test COMMAND gdn_chunk_test --dispatch-solve-vector)
  add_test(NAME gdn_solve_vector_dispatch_test COMMAND gdn_chunk_test --dispatch-solve-vector)
  add_test(NAME gdn_solve_register_dispatch_test COMMAND gdn_chunk_test --dispatch-solve-register)
  add_test(NAME gdn_chunk_solve_register_test COMMAND gdn_chunk_test)
  set_tests_properties(gdn_solve_invalid_selector_test PROPERTIES ENVIRONMENT "B70_PREFILL_GDN_SOLVE=bad")
  set_tests_properties(gdn_solve_default_dispatch_test PROPERTIES ENVIRONMENT "B70_PREFILL_GDN_SOLVE=")
  set_tests_properties(gdn_solve_vector_dispatch_test PROPERTIES ENVIRONMENT "B70_PREFILL_GDN_SOLVE=vector")
  set_tests_properties(gdn_solve_register_dispatch_test PROPERTIES ENVIRONMENT "B70_PREFILL_GDN_SOLVE=register")
  set_tests_properties(gdn_chunk_solve_register_test PROPERTIES
    ENVIRONMENT "B70_PREFILL_GDN_SOLVE=register" TIMEOUT 1800)

  # The composed attention (ruling A14), end to end against decode's own
  # reference. Same guard: it drives both the L0 list and sycl-tla's GEMM. No
  # checkpoint -- one synthetic FA layer over a synthetic cache. The host
  # reference is the expensive half (attn_ref::decode walks every block of a
  # 4352-position cache twice, once per q dtype), hence the timeout.
  add_executable(attn_chunk_test prefill/attn_chunk_test.cc)
  target_include_directories(attn_chunk_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(attn_chunk_test PRIVATE b70_prefill_host b70_runtime b70_l0)
  b70_link_prefill(attn_chunk_test)
  b70_target_kernel_dir(attn_chunk_test)
  add_dependencies(attn_chunk_test kernel_pf_attn kernel_pf_attn_prep_q16 kernel_pf_flash_attn
                   kernel_pf_gemm_T0 kernel_pf_gemm_T1)
  add_test(NAME attn_chunk_test COMMAND attn_chunk_test)
  set_tests_properties(attn_chunk_test PROPERTIES TIMEOUT 3600)
endif()

# D3's scan-only arithmetic discriminator is direct Level Zero/OpenCL C, so it
# must run in both configurations.  The fixtures make removal of either BF16
# low chain observable without re-implementing the scan.
add_executable(gdn_scan_split_test prefill/gdn_scan_split_test.cc)
target_include_directories(gdn_scan_split_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gdn_scan_split_test PRIVATE b70_prefill_host b70_l0)
if(B70_PREFILL_ENABLED)
  b70_link_prefill(gdn_scan_split_test)
endif()
b70_target_kernel_dir(gdn_scan_split_test)
add_dependencies(gdn_scan_split_test kernel_pf_gdn_scan)
add_test(NAME gdn_scan_split_test COMMAND gdn_scan_split_test)
set_tests_properties(gdn_scan_split_test PROPERTIES TIMEOUT 1800)


add_executable(int4_test kernels/int4_test.cc)
target_include_directories(int4_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME int4_test COMMAND int4_test)

add_executable(gemv_test kernels/gemv_test.cc)
target_include_directories(gemv_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_test)
add_dependencies(gemv_test
  kernel_gemv_control_M1_K6144_N5120_S4_L0
  kernel_gemv_M1_K6144_N5120_S4_L0 kernel_gemv_M1_K5120_N14336_S2_L0
  kernel_gemv_M1_K5120_N16384_S1_L1 kernel_gemv_M1_K5120_N34816_S8_L0
  kernel_gemv_M1_K17408_N5120_S4_L0 kernel_gemv_M2_K6144_N5120_S1_L0
  kernel_gemv_M1_K5120_N38912_S8_L0 kernel_gemv_M1_K19456_N5120_S4_L0)   # spec 14
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
add_test(NAME gemv_bf16_test COMMAND gemv_bf16_test)

# Spec 9 (plan 9b Task 2): the int8 lm_head GEMV at its real shape, M = 1 and 4, against
# an fp64 reference; prints the launch's GB/s.
add_executable(gemv_i8w_test kernels/gemv_i8w_test.cc)
target_include_directories(gemv_i8w_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_i8w_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_i8w_test)
add_dependencies(gemv_i8w_test kernel_gemv_i8w_M1_K5120_N248320 kernel_gemv_i8w_M4_K5120_N248320)
add_test(NAME gemv_i8w_test COMMAND gemv_i8w_test)
set_tests_properties(gemv_i8w_test PROPERTIES TIMEOUT 900)

# Spec 8 §11: the draft vocabulary's kernels at every compiled |V'|, int8 and bf16 heads -
# the compact GEMV bitwise equal to the full head's columns, the mapped argmax equal to the full
# output's argmax restricted to V', the scatter row, planted ties / -inf / NaN. The
# binaries are built with the MTP set (src/kernels/CMakeLists.txt, B70_MTP).
if(B70_MTP)
  add_executable(draft_vocab_kernels_test kernels/draft_vocab_kernels_test.cc)
  target_include_directories(draft_vocab_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
  target_link_libraries(draft_vocab_kernels_test PRIVATE b70_loader b70_l0)
  b70_target_kernel_dir(draft_vocab_kernels_test)
  add_dependencies(draft_vocab_kernels_test kernel_gemv_i8w_M1_K5120_N248320
    kernel_gemv_i8w_M1_K5120_N32768 kernel_gemv_i8w_M1_K5120_N65536
    kernel_gemv_i8w_M1_K5120_N131072 kernel_gemv_bf16_M1_K5120_N248320
    kernel_gemv_bf16_M1_K5120_N32768 kernel_gemv_bf16_M1_K5120_N65536
    kernel_gemv_bf16_M1_K5120_N131072 kernel_dv_argmax_N32768 kernel_dv_argmax_N65536
    kernel_dv_argmax_N131072)
  add_test(NAME draft_vocab_kernels_test COMMAND draft_vocab_kernels_test)
  set_tests_properties(draft_vocab_kernels_test PROPERTIES TIMEOUT 900)
endif()

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
  kernel_prep_silu_mul_M1 kernel_prep_gated_head_M1
  kernel_prep_silu_mul_M1_I19456)   # spec 14: Agnes's intermediate
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
      kernel_gemv_bf16_M${MM}_K5120_N14336 kernel_prep_norm_finish_M${MM}_K5120_G20_W20_X10240
      kernel_prep_res_fold_M${MM}_K5120_SP1_G20_Z kernel_attn_prep_M${MM}_S1)
  endforeach()
  foreach(KN K5120_N6144 K4096_N5120 K5120_N34816 K17408_N5120 K25600_N5120 K5120_N2048)
    list(APPEND B70_VERIFY_M8_KERNELS kernel_gemv_i8w_M8_${KN} kernel_gemv_bf16_M8_${KN})
  endforeach()
  foreach(KN K5120_N1280 K5120_N256)
    list(APPEND B70_VERIFY_M8_KERNELS kernel_gemv_i8w_M8_${KN} kernel_gemv_bf16_M8_${KN}_C16_S16)
  endforeach()
  list(APPEND B70_VERIFY_M8_KERNELS kernel_gemv_M8_K5120_N6144_S4_L0 kernel_gemv_M8_K4096_N5120_S4_L0
    kernel_gemv_M8_K5120_N1280_S16_L0 kernel_gemv_M8_K5120_N2048_S8_L0
    kernel_gemv_M8_K25600_N5120_S8_L0 kernel_gemv_M8_K5120_N256_S16_L0
    kernel_gemv_i8w_M7_K5120_N32768)
  set(_v19a_reused kernel_argmax_stage2)
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

# The decode-attention trio. Its reference (attn_ref.h) reproduces the kernels'
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
  kernel_attn_decode_M2_L4096_B${ATTN_BLOCK} kernel_attn_reduce_M2_L4096_B${ATTN_BLOCK}
  kernel_attn_decode_M1_L16384_B${ATTN_BLOCK} kernel_attn_reduce_M1_L16384_B${ATTN_BLOCK})
add_test(NAME attn_test COMMAND attn_test)

# Spec 10 A1 (plan 10b Task 1): decode attention v2 against v1. Part S (synthetic)
# always; part D on plan 10a's real capture when B70_ATTN_V2_DUMP names it (the box
# keeps it at ~/spec10a-dump).
set(B70_ATTN_V2_DUMP "$ENV{HOME}/spec10a-dump" CACHE PATH "plan 10a's decode-attention capture")
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

add_executable(json_test common/json_test.cc)
target_include_directories(json_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME json_test COMMAND json_test)

# Spec 12b: the int8 KV cache's rotkv arithmetic on the host (src/common/kv8.h) - the
# probe's signs, binary16 RNE, the rotation and the quantiser. No device.
add_executable(kv8_test common/kv8_test.cc)
target_include_directories(kv8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME kv8_test COMMAND kv8_test)

add_executable(safetensors_test loader/safetensors_test.cc)
target_include_directories(safetensors_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(safetensors_test PRIVATE b70_loader)
add_test(NAME safetensors_test COMMAND safetensors_test)

# Spec 9 (plan 9b Task 1): the int8 lm_head's host quantisation, on synthetic rows.
add_executable(lm_head_int8_test loader/lm_head_int8_test.cc)
target_include_directories(lm_head_int8_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(lm_head_int8_test PRIVATE b70_loader)
add_test(NAME lm_head_int8_test COMMAND lm_head_int8_test)

# Spec 8 §11: the draft vocabulary's host half - which ids V' holds, the ranked-id file,
# tokenizer.json's added tokens, and the compact head's gather from the int8 head.
add_executable(draft_vocab_test loader/draft_vocab_test.cc)
target_include_directories(draft_vocab_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(draft_vocab_test PRIVATE b70_loader)
add_test(NAME draft_vocab_test COMMAND draft_vocab_test)

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

# Spec 14 §2: the loader's packed-int4 fold against tools/oracle/agnes_fold.py on a
# shared synthetic fixture, byte for byte. Host only.
add_executable(agnes_fold_test loader/agnes_fold_test.cc)
target_include_directories(agnes_fold_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(agnes_fold_test PRIVATE b70_loader)
add_test(NAME agnes_fold_test COMMAND agnes_fold_test
  ${CMAKE_SOURCE_DIR}/tests/loader/agnes_fold_fixture.safetensors)

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
# Qwen3.8's and Agnes's shapes, suffixed ones at Ornith's. Host only; names only.
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

# The five checkpoint-labelled tests below (loader::load, argv[1] or [3])
# default with no argument to the HF repo id
# "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ", which loader::resolve_snapshot
# (src/loader/snapshot.cc) follows through the cache's refs/main pointer -
# whatever revision the operator downloaded last, not necessarily one this
# loader accepts: refs/main has pointed at a snapshot whose config.json carries
# a broken non-exclusion dynamic-quant rule that src/loader/quant.cc correctly
# refuses (see BENCHMARKS.md). A bare `ctest` must not depend on that pointer,
# so every add_test below for these five passes a snapshot directory
# explicitly - resolve_snapshot takes a directory verbatim, bypassing
# refs/main entirely. Same spirit as tools/probe/run_vllm_gemv_bench.sh's
# BENCH_SNAP (an overridable default rather than a hardcoded path): here a
# CMake cache variable (B70_TEST_SNAPSHOT), defaulted to the gate checkpoint's
# known-good snapshot, resolved against $HF_HOME (or ~/.cache/huggingface) the
# same way the loader itself does. Override with -DB70_TEST_SNAPSHOT=<dir> at
# configure time if the operator's cache moves the working snapshot; this never
# mutates the HF cache itself.
#
# 2026-09-09 (ruling A31): the gate checkpoint is
# urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ at snapshot 84575a1 (GPTQ g64
# sym, bf16 lm_head, exclusion-only dynamic rules). The previous default,
# Vishva007's 2a90776, was deleted from the box on 2026-09-09; its golden set in
# `oracle-out/` belongs to THAT checkpoint's weights and is not reused - the
# golden registrations below take B70_TEST_ORACLE_DIR, one golden set per
# checkpoint (tools/oracle/README.md), and SKIP (77) until the operator has
# dumped it.
if(DEFINED ENV{HF_HOME})
  set(_b70_hf_home "$ENV{HF_HOME}")
else()
  set(_b70_hf_home "$ENV{HOME}/.cache/huggingface")
endif()
set(B70_TEST_SNAPSHOT
  "${_b70_hf_home}/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d"
  CACHE PATH "Checkpoint snapshot directory the checkpoint-labelled tests load (bypasses refs/main)")
set(B70_TEST_ORACLE_DIR "${CMAKE_SOURCE_DIR}/oracle-out-primary"
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
  kernel_attn_prep_M1 kernel_attn_decode_M1_L16384_B${ATTN_BLOCK}
  kernel_attn_reduce_M1_L16384_B${ATTN_BLOCK} kernel_attn_v2_M1_T${ATTN_V2_TGT}
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
# that live only on the box - hence the second label and the SKIP_RETURN_CODE:
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
# Registered HERE rather than in the B70_PREFILL_ENABLED block above because
# these need B70_TEST_SNAPSHOT and B70_DECODE_LIST_KERNELS, both defined between
# there and here.
#
# Built in EVERY configuration since spec 2.1 (b70_prefill_host always exists). The sycl-tla
# registrations need libb70_prefill.so and stay under the guard; the l0 ones run everywhere,
# which is how bar 3's "in a build without the SYCL component" is checked.
#
# Every binary the prefill walk opens. The decode list is part of it because
# `step_head`'s lm_head and its two argmax stages ARE decode's own M = 1
# binaries -- the prefill path re-uses them rather than compiling a second
# copy, which is also why the module count below is not a new number. Since
# spec 2.1 S2/S3 the L0 backend's own binaries are in the list too: pf_gemm IS
# its linears' and its attention's GEMM, and the slab dequants feed them.
set(B70_PREFILL_KERNELS
  kernel_pf_embed_gather kernel_pf_res_fold_K5120_SP0_G20
  kernel_pf_res_fold_K5120_SP1_G20 kernel_pf_norm_finish_K5120_G20_W20
  kernel_pf_silu_mul kernel_pf_gated_head kernel_pf_ab_proj
  kernel_pf_attn_prep_q16 kernel_pf_attn kernel_pf_flash_attn
  kernel_pf_gdn_conv kernel_pf_gdn_wy kernel_pf_gdn_scan
  kernel_pf_dequant_tile_K5120_N16384_L1_T0
  kernel_pf_dequant_tile_K6144_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N34816_L0_T0
  kernel_pf_dequant_tile_K17408_N5120_L0_T0
  kernel_pf_dequant_tile_K5120_N14336_L0_T0
  kernel_pf_gemm_T0 kernel_pf_gemm_T1 kernel_pf_gemm_T0_SILU
  kernel_pf_dequant_slab_K5120_N16384_L1 kernel_pf_dequant_slab_K6144_N5120_L0
  kernel_pf_dequant_slab_K5120_N34816_L0 kernel_pf_dequant_slab_K17408_N5120_L0
  kernel_pf_dequant_slab_K5120_N14336_L0
  kernel_pf_dequant_slab_K5120_N38912_L0 kernel_pf_dequant_slab_K19456_N5120_L0
  kernel_pf_dequant_tile_K5120_N38912_L0_T0 kernel_pf_dequant_tile_K19456_N5120_L0_T0
  # Spec 5's h8 path, which `--prefill-backend l0-int8` runs (pf_colmax_rot is in the
  # pf_requant_rot binaries).
  kernel_pf_quant_had_K5120 kernel_pf_quant_had_K6144 kernel_pf_quant_had_K17408
  kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1 kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU
  ${B70_DECODE_LIST_KERNELS})

# The three spec 2 §6 gates. `tests/golden` is on the include path for
# golden_common.h -- the decode gate's machinery, extracted verbatim, so the
# prefill gate grades on the same code rather than on a copy of it. The three
# BINARIES are built in every configuration; only `b70_link_prefill` (the
# sycl-tla .so and its two rpaths) is conditional, which is plan 9b's ruling B4
# pattern and what `context_test` above already does.
foreach(t prefill_gate prefill_consistency prefill_determinism prefill_replay prefill_int8)
  add_executable(${t}_test prefill/${t}_test.cc)
  target_include_directories(${t}_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(${t}_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(${t}_test)
  endif()
  add_dependencies(${t}_test ${B70_PREFILL_KERNELS})
endforeach()
# Spec 6 K3a (plan 6b Task 4; redefined by the operator's 2026-09-26 ruling: flash no further
# from the CPU oracle than composed, per backend, flash >= composed - 1e-6).
#   flash_vs_oracle_test: THE K3a bar, on the golden set (prose, code, cjk; 96 decision rows
#     per mode), l0 and l0-int8, both modes; mean logit cosine against the CPU oracle.
#   flash_long_test: the 32704-id prefill of the committed prompt in both modes on both
#     backends, every logit finite; flash-vs-composed printed as information. A 32k CPU
#     oracle (tools/oracle/last_logits.py) measured 11359 s to reach 8192 ids, so it is not
#     part of the suite; given one, `flash_long_test ... <file>` applies the bar at depth.
#   Its `--128k <ids>` mode (K3b, determinism and replay at max_len 131072) is run by hand.
add_test(NAME flash_vs_oracle_test COMMAND ${CMAKE_SOURCE_DIR}/tools/probe/flash_vs_oracle.sh)
set_tests_properties(flash_vs_oracle_test PROPERTIES LABELS "checkpoint;golden;prefill"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
add_executable(flash_long_test prefill/flash_long_test.cc)
target_include_directories(flash_long_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(flash_long_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
if(B70_PREFILL_ENABLED)
  b70_link_prefill(flash_long_test)
endif()
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
if(B70_PREFILL_ENABLED)
  b70_link_prefill(prefill_split_test)
endif()
add_dependencies(prefill_split_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
foreach(b l0 l0-int8)
  add_test(NAME prefill_split_${b}_test COMMAND prefill_split_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long32k.ids ${b})
  set_tests_properties(prefill_split_${b}_test PROPERTIES LABELS "checkpoint;prefill"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
endforeach()
# Spec 7 C1 (plan 7b): state and KV snapshots restore bitwise; the block hook.
add_executable(snapshot_test runtime/snapshot_test.cc)
target_include_directories(snapshot_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(snapshot_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
if(B70_PREFILL_ENABLED)
  b70_link_prefill(snapshot_test)
endif()
add_dependencies(snapshot_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
add_test(NAME snapshot_test COMMAND snapshot_test ${B70_TEST_SNAPSHOT})
set_tests_properties(snapshot_test PROPERTIES LABELS checkpoint
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 900)
add_test(NAME prefill_replay_test COMMAND prefill_replay_test ${B70_TEST_SNAPSHOT})
set_tests_properties(prefill_replay_test PROPERTIES LABELS "checkpoint;prefill"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
# Spec 5 T3 (plan 5b Task 2): l0-int8 on the engine - a smoke cosine against l0, replay
# bitwise against immediate, and the l0 + 256 launch delta. The int8 kernels are in
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
# each backend's standing near-tie allowance (0 on l0, 1 on l0-int8), unchanged.
add_test(NAME prefill_gate_l0_i8head_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 0 l0 0 int8)
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
# is prefill_gate_long_int8_test below, which is registered only while the RTN checkpoint
# and oracle-out-long exist; neither does on the box (2026-09-24), so this pair stands in.
add_test(NAME prefill_gate_l0_c16_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 16 l0)
add_test(NAME prefill_gate_int8_c16_test COMMAND prefill_gate_test
  ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
  prose,code,cjk 16 l0-int8 1)
set_tests_properties(prefill_gate_l0_c16_test prefill_gate_int8_c16_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
add_test(NAME prefill_consistency_l0_test COMMAND prefill_consistency_test
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} l0)
add_test(NAME prefill_determinism_l0_test COMMAND prefill_determinism_test
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} l0)
set_tests_properties(prefill_consistency_l0_test prefill_determinism_l0_test PROPERTIES
  LABELS "checkpoint;prefill" TIMEOUT 3600)

if(B70_PREFILL_ENABLED)
  # sycl-tla, the full build: the registrations exactly as before, the backend now
  # explicit. `prose,code,cjk 0` are prefill_gate_test's own defaults for argv[4] and
  # argv[5], spelled out because argv[6] cannot be reached past them.
  #
  # The gate checkpoint (bf16 lm_head) and RTN (int4 lm_head) -- one golden set
  # per checkpoint (tools/oracle/README.md, "One golden set per checkpoint"), so
  # two registrations and never one that takes whichever directory is lying
  # around. The RTN registrations exist only while the RTN checkpoint does: it
  # was deleted from the box on 2026-09-09 (ruling A31) and a missing snapshot
  # is a load failure, not a golden-set SKIP.
  add_test(NAME prefill_gate_test COMMAND prefill_gate_test
    ${B70_TEST_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT}
    prose,code,cjk 0 sycl-tla)
  add_test(NAME prefill_consistency_test COMMAND prefill_consistency_test
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} sycl-tla)
  add_test(NAME prefill_determinism_test COMMAND prefill_determinism_test
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT} sycl-tla)
  set_tests_properties(prefill_gate_test PROPERTIES
    LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  set_tests_properties(prefill_consistency_test prefill_determinism_test PROPERTIES
    LABELS "checkpoint;prefill" TIMEOUT 3600)

  # `Engine::prefill` end to end: the chunk walk, the Control handoff, the
  # launch arithmetic and the rejections (plan 6b Task 10). Its sections 2-7 are
  # sycl-tla's arithmetic and pin the backend themselves; section 8 (ruling E1)
  # measures the L0 chunk's live launch delta. Both backends, so: under the guard.
  add_executable(prefill_smoke_test prefill/prefill_smoke_test.cc)
  target_include_directories(prefill_smoke_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(prefill_smoke_test PRIVATE b70_prefill_host b70_runtime b70_loader
                        b70_l0 b70_model)
  b70_link_prefill(prefill_smoke_test)
  add_dependencies(prefill_smoke_test ${B70_PREFILL_KERNELS})
  add_test(NAME prefill_smoke_test COMMAND prefill_smoke_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(prefill_smoke_test PROPERTIES LABELS "checkpoint;prefill" TIMEOUT 1800)

  # Spec 2.1 bar 2: the two backends leave identical state (sign of zero excepted).
  # Needs both backends, so it is registered only in the full build (spec 2.1 §4).
  add_executable(prefill_backend_equivalence_test prefill/prefill_backend_equivalence_test.cc)
  target_include_directories(prefill_backend_equivalence_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(prefill_backend_equivalence_test PRIVATE b70_prefill_host b70_runtime
                        b70_loader b70_l0 b70_model)
  b70_link_prefill(prefill_backend_equivalence_test)
  add_dependencies(prefill_backend_equivalence_test ${B70_PREFILL_KERNELS})
  add_test(NAME prefill_backend_equivalence_test COMMAND prefill_backend_equivalence_test
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT})
  set_tests_properties(prefill_backend_equivalence_test PROPERTIES LABELS "checkpoint;prefill" TIMEOUT 3600)

  # The int4-lm_head checkpoint (spec 1.6 §5.1, tools/quantize_qwen38_rtn.sh).
  # It never went through `hf download`, so it is a plain path and not a cache
  # lookup; the same override rule as B70_TEST_SNAPSHOT applies.
  set(B70_RTN_SNAPSHOT "$ENV{HOME}/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64"
    CACHE PATH "The int4-lm_head checkpoint the RTN-labelled gates load")

  if(EXISTS "${B70_RTN_SNAPSHOT}/config.json")
    add_test(NAME prefill_gate_rtn_test COMMAND prefill_gate_test
      ${CMAKE_SOURCE_DIR}/oracle-out-rtn ${CMAKE_SOURCE_DIR}/tests/golden/prompts
      ${B70_RTN_SNAPSHOT})
    set_tests_properties(prefill_gate_rtn_test PROPERTIES
      LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  else()
    message(STATUS "b70: RTN checkpoint absent (${B70_RTN_SNAPSHOT}) -- prefill_gate_rtn_test and prefill_gate_long_test not registered")
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
  # `oracle-out-long` does not exist yet: producing it is one CPU run inside the
  # reference container, and the L1-engine stage is barred from starting
  # containers (its brief's operational rules). The test SKIPs (77) until the
  # directory is there, which is the same contract the other three golden
  # registrations use, and tools/oracle/README.md carries the one command.
  if(EXISTS "${B70_RTN_SNAPSHOT}/config.json")
    add_test(NAME prefill_gate_long_test COMMAND prefill_gate_test
      ${CMAKE_SOURCE_DIR}/oracle-out-long ${CMAKE_SOURCE_DIR}/tests/golden/prompts
      ${B70_RTN_SNAPSHOT} long 1024)
    set_tests_properties(prefill_gate_long_test PROPERTIES
      LABELS "checkpoint;golden;prefill;long" SKIP_RETURN_CODE 77 TIMEOUT 7200)
    # Spec 5 (plan 5b Task 3): the same 2820-id, two-chunk gate on l0-int8.
    add_test(NAME prefill_gate_long_int8_test COMMAND prefill_gate_test
      ${CMAKE_SOURCE_DIR}/oracle-out-long ${CMAKE_SOURCE_DIR}/tests/golden/prompts
      ${B70_RTN_SNAPSHOT} long 1024 l0-int8 1)
    set_tests_properties(prefill_gate_long_int8_test PROPERTIES
      LABELS "checkpoint;golden;prefill;long" SKIP_RETURN_CODE 77 TIMEOUT 7200)
  endif()
endif()

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
# "--prefill-backend expects sycl-tla, l0 or l0-int8".
b70_cli_reject(cli_reject_prefill_backend_value ${B70_NO_MODEL} --bench --prefill-length 64
  --prefill-backend cuda EXPECT "--prefill-backend expects sycl-tla, l0 or l0-int8")
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
# These five are host-side argument validation and are compiled UNCONDITIONALLY
# - with -DB70_PREFILL=OFF the flags still parse and still reject for these
# reasons, so these cases grade the CLI's contract rather than the presence of
# the SYCL component. Each names its message: "exit 1 before the device" is
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
# The 2026-10-06 rename (cli/renamed_flags.h): `--pp` is vLLM's pipeline-parallel size now, so
# an old `--bench --pp 4096` must not request a 4096-way pipeline - any value but 1 or 2 is
# refused naming --prefill-length - and the old prefill flags are unknown options that name
# their new spelling (no aliases).
b70_cli_reject(cli_reject_pp_old_prefill_length ${B70_NO_MODEL} --bench --pp 4096
  EXPECT "--pp is the pipeline-parallel size, 1 or 2")
b70_cli_reject(cli_reject_pp_old_prefill_length_hint ${B70_NO_MODEL} --bench --pp 4096
  EXPECT "the bench's prefill length is --prefill-length N")
b70_cli_reject(cli_reject_pipeline_parallel_size_value ${B70_NO_MODEL} --bench
  --pipeline-parallel-size 512 EXPECT "--pipeline-parallel-size is the pipeline-parallel size, 1 or 2")
b70_cli_reject(cli_reject_renamed_pp_chunk ${B70_NO_MODEL} --bench --prefill-length 64 --pp-chunk 1024
  EXPECT "unknown option '--pp-chunk' (renamed: --prefill-chunk C)")
b70_cli_reject(cli_reject_renamed_pp_backend ${B70_NO_MODEL} --bench --prefill-length 64
  --pp-backend l0 EXPECT "unknown option '--pp-backend' (renamed: --prefill-backend B)")
b70_cli_reject(cli_reject_renamed_pipeline ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --pipeline 2
  EXPECT "unknown option '--pipeline' (renamed: --pp N, or --pipeline-parallel-size N)")
# `--prefill` is the ids-mode equivalent of `--prefill-length`; it must not be accepted as
# an unknown flag or leak as far as l0::Context.
b70_cli_reject(cli_reject_prefill_without_ids ${B70_NO_MODEL} --bench --prefill
  EXPECT "--prefill belongs to --ids")
# Spec 6 §10: --max-len is `auto` or a positive integer, --mem-reserve-gb a number of GB;
# both are parsed host-side, before the snapshot or the device is touched.
b70_cli_reject(cli_reject_max_len_zero ${B70_NO_MODEL} --bench --max-len 0
  EXPECT "--max-len 0 is not a model length")
b70_cli_reject(cli_reject_max_len_word ${B70_NO_MODEL} --bench --max-len big
  EXPECT "--max-len expects auto or a positive integer")
b70_cli_reject(cli_reject_mem_reserve ${B70_NO_MODEL} --bench --mem-reserve-gb -1
  EXPECT "--mem-reserve-gb expects a number of GB")
# Spec 8: `--mtp K|auto` is --ids mode's greedy speculative decoding (queue row 6's
# long-context checks without the server); parsed and placed before the device like the rest.
b70_cli_reject(cli_reject_mtp_with_bench ${B70_NO_MODEL} --bench --mtp 1
  EXPECT "--mtp belongs to --ids")
b70_cli_reject(cli_reject_mtp_value ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --mtp 4
  EXPECT "--mtp expects off, 0..3 or auto")
b70_cli_reject(cli_reject_mtp_k2 ${CMAKE_CURRENT_SOURCE_DIR}/model/k2 --ids ${B70_OK_IDS} --n 4
  --mtp auto EXPECT "K2-Horizon has no MTP head")

# Spec 6 (plan 6b Task 3): a max_len with no compiled decode attention. This one is NOT
# host-side -- it needs the device and the checkpoint, because the variant is resolved at
# capture -- so it is its own registration, labelled checkpoint: exit 1 (not a signal),
# and the message names the missing binary, attn_decode_M1_L65536_B64.
add_test(NAME cli_reject_maxlen_uncompiled COMMAND sh -c [==[
out=$("$@" 2>&1)
rc=$?
printf '%s\n' "$out" >&2
[ "$rc" -eq 1 ] || { echo "expected exit 1, got $rc" >&2; exit 1; }
printf '%s\n' "$out" | grep -qF attn_decode_M1_L65536 ||
  { echo "exit 1, but the missing variant is not named" >&2; exit 1; }
]==] cli_reject_maxlen_uncompiled $<TARGET_FILE:b70-decode> ${B70_TEST_SNAPSHOT}
  --bench --depth 64 --max-len 65536)
# Spec 10: decode attention v2 bakes no MAXLEN, so the missing binary is v1's alone and
# the test pins B70_DECODE_ATTN=v1 (v1 stays selectable; its guard must keep firing).
set_tests_properties(cli_reject_maxlen_uncompiled PROPERTIES LABELS "checkpoint" TIMEOUT 600
  ENVIRONMENT "B70_DECODE_ATTN=v1")

# --- spec 3 T1: tokenizer (host-only; needs the checkpoint's tokenizer.json) ---
if(B70_TOKENIZER_ENABLED)
  add_executable(tokenizer_smoke_test tokenizer/tokenizer_smoke_test.cc)
  target_include_directories(tokenizer_smoke_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(tokenizer_smoke_test PRIVATE b70_tokenizer)
  add_test(NAME tokenizer_smoke_test COMMAND tokenizer_smoke_test)
  add_executable(parity_test tokenizer/parity_test.cc)
  target_include_directories(parity_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(parity_test PRIVATE b70_tokenizer)
  add_test(NAME parity_test COMMAND parity_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
  add_executable(template_test tokenizer/template_test.cc)
  target_include_directories(template_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(template_test PRIVATE b70_tokenizer)
  add_test(NAME template_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
  # Spec 14: Agnes's template (byte-identical to Qwen3.8's) on five message lists - plain,
  # thinking, tools, a tool call in history, tool responses - against transformers' render.
  set(B70_AGNES_SNAPSHOT_DIR
    "${_b70_hf_home}/hub/models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ/snapshots/946012d0e0b1b6df685c910adf3d68e351931306"
    CACHE PATH "Agnes snapshot directory for the template test (tokenizer_config + chat_template.jinja)")
  add_test(NAME template_agnes_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${B70_AGNES_SNAPSHOT_DIR} agnes_template_cases.json)
  add_executable(streamer_test tokenizer/streamer_test.cc)
  target_include_directories(streamer_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(streamer_test PRIVATE b70_tokenizer)
  add_test(NAME streamer_test COMMAND streamer_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)

  add_executable(golden_server_test server/golden_server_test.cc)
  target_include_directories(golden_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(golden_server_test PRIVATE b70_server)
  add_dependencies(golden_server_test b70-serve b70-decode)
  add_test(NAME golden_server_test COMMAND golden_server_test
    $<TARGET_FILE:b70-serve> $<TARGET_FILE:b70-decode> ${CMAKE_SOURCE_DIR}/tests/golden/prompts
    ${B70_TEST_SNAPSHOT})
  set_tests_properties(golden_server_test PROPERTIES TIMEOUT 300)

  # Host sampling (spec 3.5, plan 7d Task 5): the sampler function alone, over
  # synthetic logits rows -- no device, no checkpoint, no Engine instantiated.
  # It includes serve_adapters.h (where `sample()` and `EngineAdapter` live),
  # so it links what b70-serve links -- b70_prefill_host unconditionally,
  # b70_link_prefill under the same guard -- even though this test never
  # constructs an EngineAdapter.
  add_executable(sampling_test server/sampling_test.cc)
  target_include_directories(sampling_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(sampling_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
    b70_model b70_l0 b70_prefill_host)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(sampling_test)
  endif()
  add_test(NAME sampling_test COMMAND sampling_test)

  # Spec 7 C2 (plan 7c Task 3): the prefix cache's request path on the card, cached against
  # cold under the tie-aware rule, on both L0 backends. serve_adapters.h again, so it links
  # what b70-serve links.
  add_executable(prefix_gpu_test server/prefix_gpu_test.cc)
  target_include_directories(prefix_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(prefix_gpu_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
    b70_model b70_l0 b70_prefill_host)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(prefix_gpu_test)
  endif()
  add_dependencies(prefix_gpu_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS})
  # l0 takes the near-tie allowance of 1 too (operator ruling, spec 7 §8): sequence e's
  # row 30 flip (0.108 bf16 ulp) is the GDN chunk-boundary rounding class of
  # docs/probe-prefill-continuation-2026-09-28.md, not a cache fault.
  add_test(NAME prefix_gpu_l0_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0 1)
  add_test(NAME prefix_gpu_int8_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0-int8)
  # The opt-in split_last path (b70-serve --prefix-split-last), same sequences.
  add_test(NAME prefix_gpu_split_l0_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0 0 1)
  add_test(NAME prefix_gpu_split_int8_test COMMAND prefix_gpu_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts l0-int8 1 1)
  set_tests_properties(prefix_gpu_l0_test prefix_gpu_int8_test prefix_gpu_split_l0_test
    prefix_gpu_split_int8_test PROPERTIES LABELS checkpoint TIMEOUT 1800)
  # Spec 8 M5 (plan 8c): C2 with the MTP head loaded and the adapter at --mtp 3.
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
    if(B70_PREFILL_ENABLED)
      b70_link_prefill(${t}_test)
    endif()
    add_dependencies(${t}_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
  endforeach()
  # M1: the head's draft logits against tools/oracle/mtp_ref.py --dump (SKIP without it).
  add_test(NAME mtp_head_test COMMAND mtp_head_test ${B70_TEST_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${CMAKE_SOURCE_DIR}/oracle-out-spec8b/m1
    ${CMAKE_SOURCE_DIR}/oracle-out-spec8b)
  set_tests_properties(mtp_head_test PROPERTIES LABELS "checkpoint;golden;mtp"
    SKIP_RETURN_CODE 77 TIMEOUT 1800)
  # Task 3 / Review Focus 2: the head's KV after prefill against M = 1 fills.
  add_test(NAME mtp_prefill_test COMMAND mtp_prefill_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(mtp_prefill_test PROPERTIES LABELS "checkpoint;prefill;mtp"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  # M2 + Review Focus 1-5: verify at M = k + 1, per-row GDN slots, commit, snapshots, replay.
  add_test(NAME mtp_verify_test COMMAND mtp_verify_test ${B70_TEST_SNAPSHOT})
  set_tests_properties(mtp_verify_test PROPERTIES LABELS "checkpoint;mtp"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  # Plan 8c: M3 (greedy MTP is identical to plain), RF1 on the card, M5, M4 end to end,
  # through EngineAdapter (the server's engine calls); and the D1/D2 bench (--bench K).
  if(B70_TOKENIZER_ENABLED)
    add_executable(mtp_gpu_test server/mtp_gpu_test.cc)
    target_include_directories(mtp_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
    target_link_libraries(mtp_gpu_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
      b70_model b70_l0 b70_prefill_host)
    if(B70_PREFILL_ENABLED)
      b70_link_prefill(mtp_gpu_test)
    endif()
    add_dependencies(mtp_gpu_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
    add_test(NAME mtp_gpu_test COMMAND mtp_gpu_test ${B70_TEST_SNAPSHOT} l0-int8)
    set_tests_properties(mtp_gpu_test PROPERTIES LABELS "checkpoint;mtp"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 5400)
    # Spec 8 §11's M3 gate: drafts over a 128k draft vocabulary, greedy output identical to
    # the plain loop (= --draft-vocab off) on the golden and A4 prompts at every K, and the
    # sampled / replay checks with q = -inf outside V'. 32k / 64k run the same way by hand.
    # Both head forms: the compact head is gathered from whichever one is loaded.
    add_test(NAME mtp_gpu_dv128k_test COMMAND mtp_gpu_test ${B70_TEST_SNAPSHOT} l0-int8
      --lm-head int8 --draft-vocab 128k)
    add_test(NAME mtp_gpu_dv128k_bf16_test COMMAND mtp_gpu_test ${B70_TEST_SNAPSHOT} l0-int8
      --lm-head bf16 --draft-vocab 128k)
    set_tests_properties(mtp_gpu_dv128k_test mtp_gpu_dv128k_bf16_test PROPERTIES
      LABELS "checkpoint;mtp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 5400)
  endif()
  # Spec 8 §12 (MTP at every context length; box queue row 6): mtp_verify_test (M2, RF3 at
  # max_len - 4, RF4 deep - spec 7 C1 with MTP at ~70k on the 131072 run) and mtp_gpu_test's
  # M3 gate at max_len 32768 and 131072. Label `longctx`. Each exits 77 (SKIP) at once unless
  # B70_LONGCTX_TESTS=1, so a plain `ctest` costs what it did; tools/box_validate (r6.mtp_long)
  # sets it. The variable is read here, by the wrapper, not by the tests.
  set(_b70_longctx_gate [==[
if [ "${B70_LONGCTX_TESTS:-0}" != 1 ]
then
  echo "SKIP: a long-context MTP variant (box queue row 6); B70_LONGCTX_TESTS=1 runs it"
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
      SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 7200)
    if(TARGET mtp_gpu_test)
      add_test(NAME mtp_gpu_${_tag}_test COMMAND sh -c "${_b70_longctx_gate}" b70_longctx
        $<TARGET_FILE:mtp_gpu_test> ${B70_TEST_SNAPSHOT} l0-int8 --max-len ${_len})
      set_tests_properties(mtp_gpu_${_tag}_test PROPERTIES LABELS "checkpoint;mtp;longctx"
        SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 7200)
    endif()
  endforeach()
endif()

# --- Spec 14: Agnes 3.0 Flash ---------------------------------------------------------
# The second model's gates (spec 14 §4): the loader and the fold on device (G1's loader
# half), the golden gates on decode and on both prefill backends (G2), replay determinism,
# prefill continuation, snapshots, prefix caching, MTP (G4). Same executables as Qwen3.8's,
# pointed at the Agnes checkpoint and its golden set, label `agnes` beside the usual ones:
# `ctest -L agnes` runs exactly these, `ctest -LE agnes` is a Qwen3.8 session as before.
# The golden set `oracle-out-agnes` is dumped on the box (tools/oracle/README.md, spec 14);
# the golden registrations SKIP (77) until it exists, as Qwen3.8's do.
# Unvalidated on the card (docs/superpowers/plans/2026-10-03-spec14-validation-checklist.md).
if(B70_AGNES)
  set(B70_AGNES_SNAPSHOT "urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ"
    CACHE STRING "Agnes checkpoint (repo id resolved in the HF cache, or a snapshot directory)")
  set(B70_AGNES_ORACLE_DIR "${CMAKE_SOURCE_DIR}/oracle-out-agnes"
    CACHE PATH "The CPU-oracle golden set dumped from B70_AGNES_SNAPSHOT")
  set(_ag_prompts ${CMAKE_SOURCE_DIR}/tests/golden/prompts)

  add_executable(load_agnes_test loader/load_agnes_test.cc)
  target_include_directories(load_agnes_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(load_agnes_test PRIVATE b70_loader b70_model)
  add_test(NAME load_agnes_test COMMAND load_agnes_test ${B70_AGNES_SNAPSHOT})
  set_tests_properties(load_agnes_test PROPERTIES LABELS "checkpoint;agnes" TIMEOUT 900)

  # G2: decode (bf16 and int8 head) and prefill (l0, l0-int8) against oracle-out-agnes.
  add_test(NAME golden_gate_agnes_test COMMAND golden_gate_test
    ${B70_AGNES_ORACLE_DIR} ${_ag_prompts} ${B70_AGNES_SNAPSHOT})
  add_test(NAME golden_gate_agnes_i8head_test COMMAND golden_gate_test
    ${B70_AGNES_ORACLE_DIR} ${_ag_prompts} ${B70_AGNES_SNAPSHOT} int8)
  add_test(NAME prefill_gate_agnes_l0_test COMMAND prefill_gate_test
    ${B70_AGNES_ORACLE_DIR} ${_ag_prompts} ${B70_AGNES_SNAPSHOT} prose,code,cjk 0 l0)
  add_test(NAME prefill_gate_agnes_int8_test COMMAND prefill_gate_test
    ${B70_AGNES_ORACLE_DIR} ${_ag_prompts} ${B70_AGNES_SNAPSHOT} prose,code,cjk 0 l0-int8 1)
  set_tests_properties(golden_gate_agnes_test golden_gate_agnes_i8head_test
    prefill_gate_agnes_l0_test prefill_gate_agnes_int8_test PROPERTIES
    LABELS "checkpoint;golden;agnes" SKIP_RETURN_CODE 77 TIMEOUT 3600)

  # Determinism, continuation, snapshots (G4 spec 7 C1), MTP verify (G4 spec 8 M2).
  add_test(NAME replay_determinism_agnes_test COMMAND replay_determinism_test ${B70_AGNES_SNAPSHOT})
  set_tests_properties(replay_determinism_agnes_test PROPERTIES LABELS "checkpoint;agnes" TIMEOUT 1200)
  foreach(b l0 l0-int8)
    add_test(NAME prefill_split_agnes_${b}_test COMMAND prefill_split_test ${B70_AGNES_SNAPSHOT}
      ${_ag_prompts}/long32k.ids ${b})
    set_tests_properties(prefill_split_agnes_${b}_test PROPERTIES LABELS "checkpoint;prefill;agnes"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
  endforeach()
  add_test(NAME snapshot_agnes_test COMMAND snapshot_test ${B70_AGNES_SNAPSHOT})
  set_tests_properties(snapshot_agnes_test PROPERTIES LABELS "checkpoint;agnes"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 1200)
  if(TARGET prefix_gpu_test)   # G4 spec 7 C2
    add_test(NAME prefix_gpu_agnes_int8_test COMMAND prefix_gpu_test ${B70_AGNES_SNAPSHOT}
      ${_ag_prompts} l0-int8)
    set_tests_properties(prefix_gpu_agnes_int8_test PROPERTIES LABELS "checkpoint;agnes" TIMEOUT 1800)
  endif()
  if(TARGET mtp_verify_test)
    add_test(NAME mtp_verify_agnes_test COMMAND mtp_verify_test ${B70_AGNES_SNAPSHOT})
    set_tests_properties(mtp_verify_agnes_test PROPERTIES LABELS "checkpoint;mtp;agnes"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
    # M1: the head's draft logits against tools/oracle/mtp_ref.py --dump on Agnes.
    add_test(NAME mtp_head_agnes_test COMMAND mtp_head_test ${B70_AGNES_SNAPSHOT} ${_ag_prompts}
      ${CMAKE_SOURCE_DIR}/oracle-out-agnes-mtp/m1 ${CMAKE_SOURCE_DIR}/oracle-out-agnes-mtp)
    set_tests_properties(mtp_head_agnes_test PROPERTIES LABELS "checkpoint;golden;mtp;agnes"
      SKIP_RETURN_CODE 77 TIMEOUT 1800)
  endif()
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
if(B70_PREFILL_ENABLED)
  b70_link_prefill(memory_plan_box_test)
endif()
add_dependencies(memory_plan_box_test ${_b70_plan_kernels} ${_b70_plan_pf_kernels})
add_test(NAME memory_plan_box_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT})
set(_b70_plan_box_tests memory_plan_box_test)
if(B70_MTP)
  add_test(NAME memory_plan_box_mtp_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT} mtp)
  # Spec 8 §11: the draft vocabulary's compact head + id table and compact logits are
  # planned and allocated alike, for both head forms (16384 and auto).
  add_test(NAME memory_plan_box_dv128k_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT} mtp
    --lm-head int8 --draft-vocab 128k)
  add_test(NAME memory_plan_box_dv128k_bf16_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT}
    mtp --lm-head bf16 --draft-vocab 128k)
  list(APPEND _b70_plan_box_tests memory_plan_box_mtp_test memory_plan_box_dv128k_test
    memory_plan_box_dv128k_bf16_test)
endif()
set_tests_properties(${_b70_plan_box_tests} PROPERTIES LABELS checkpoint
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 1800)
if(B70_AGNES)
  add_test(NAME memory_plan_box_agnes_test COMMAND memory_plan_box_test ${B70_AGNES_SNAPSHOT})
  set(_b70_plan_box_agnes memory_plan_box_agnes_test)
  if(B70_MTP)
    add_test(NAME memory_plan_box_agnes_mtp_test COMMAND memory_plan_box_test ${B70_AGNES_SNAPSHOT} mtp)
    list(APPEND _b70_plan_box_agnes memory_plan_box_agnes_mtp_test)
  endif()
  set_tests_properties(${_b70_plan_box_agnes} PROPERTIES LABELS "checkpoint;agnes"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 1800)
endif()

# --- Spec 15c: Ornith 1.5 35B-A3B decode on the card (label `ornith`) ----------------
# The int4 checkpoint is the operator's AutoRound W4A16 g64 export,
# urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ (spec 15 decision 1, settled in §13);
# B70_ORNITH_SNAPSHOT names it (resolved in the HF cache). Until it is on the machine
# ornith_decode_test SKIPs (77), and the golden registrations SKIP on the absent
# `oracle-out-ornith` (15a's golden set, plan 15a) before they load anything. Its
# tokenizer.json gives the committed tests/golden/prompts ids (checked, spec 15 §13), so
# B70_ORNITH_PROMPTS_DIR stays the committed directory.
#   R3 (decode): golden_gate_test, descriptor-driven since 15c (the tap is the layer
#       output when the FFN folds inside the layer), bf16 and int8 heads;
#   R2 + the list + replay bitwise + the plan: ornith_decode_test.
if(B70_ORNITH)
  set(B70_ORNITH_SNAPSHOT "urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ"
    CACHE STRING "Ornith int4 checkpoint (repo id resolved in the HF cache, or a snapshot directory)")
  set(B70_ORNITH_ORACLE_DIR "${CMAKE_SOURCE_DIR}/oracle-out-ornith"
    CACHE PATH "The CPU-oracle golden set of Ornith (spec 15a), with router_logits.L* for R2")
  set(B70_ORNITH_PROMPTS_DIR "${CMAKE_SOURCE_DIR}/tests/golden/prompts"
    CACHE PATH "Ornith's prompt ids (the committed ones: the int4 checkpoint's tokenizer gives them)")
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
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT})
  add_test(NAME golden_gate_ornith_i8head_test COMMAND golden_gate_test
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT} int8)
  set_tests_properties(golden_gate_ornith_test golden_gate_ornith_i8head_test PROPERTIES
    LABELS "checkpoint;golden;ornith" SKIP_RETURN_CODE 77 TIMEOUT 3600)

  add_executable(ornith_decode_test golden/ornith_decode_test.cc)
  target_include_directories(ornith_decode_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(ornith_decode_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
  add_dependencies(ornith_decode_test ${B70_ORNITH_DECODE_KERNELS})
  add_test(NAME ornith_decode_test COMMAND ornith_decode_test ${B70_ORNITH_SNAPSHOT}
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR})
  add_test(NAME ornith_decode_i8head_test COMMAND ornith_decode_test ${B70_ORNITH_SNAPSHOT}
    ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} int8)
  set_tests_properties(ornith_decode_test ornith_decode_i8head_test PROPERTIES
    LABELS "checkpoint;golden;ornith" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()

# --- Spec 12b: the int8 KV cache (`--kv-cache int8`, src/kernels/kv8.cl) ----------------
# Written blind on the Mac; docs/superpowers/plans/box-validation-queue.md row 11 is the order
# to run these in. `kv8` labels all of them: `ctest -L kv8`.
if(B70_KV8)
  set(B70_KV8_KERNELS
    kernel_attn_prep_M1_KV8 kernel_attn_prep_M2_KV8 kernel_attn_prep_M3_KV8 kernel_attn_prep_M4_KV8
    kernel_attn_v2_M1_T${ATTN_V2_TGT}_KV8 kernel_attn_v2_M2_T${ATTN_V2_TGT}_KV8
    kernel_attn_v2_M3_T${ATTN_V2_TGT}_KV8 kernel_attn_v2_M4_T${ATTN_V2_TGT}_KV8
    kernel_pf_attn_prep_kv8 kernel_pf_flash_attn_kv8 kernel_pf_attn_gate_kv8)
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
  add_test(NAME kv8_kernels_test COMMAND kv8_kernels_test)
  set_tests_properties(kv8_kernels_test PROPERTIES LABELS kv8 TIMEOUT 1800)

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

  foreach(exe replay_determinism_test engine_smoke_test golden_gate_test prefill_gate_test
              prefill_determinism_test prefill_consistency_test prefill_replay_test snapshot_test
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
  add_test(NAME kv8_vs_oracle_test COMMAND ${CMAKE_SOURCE_DIR}/tools/probe/kv8_vs_oracle.sh)
  set_tests_properties(kv8_vs_oracle_test PROPERTIES LABELS "checkpoint;golden;prefill;kv8"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 3600)
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
  add_test(NAME prefill_replay_kv8_test COMMAND prefill_replay_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(prefill_replay)
  add_test(NAME snapshot_kv8_test COMMAND snapshot_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(snapshot)
  add_test(NAME memory_plan_box_kv8_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(memory_plan_box)
  if(TEST mtp_verify_test)
    add_test(NAME mtp_verify_kv8_test COMMAND mtp_verify_test ${B70_TEST_SNAPSHOT})
    b70_kv8_twin(mtp_verify)
    add_test(NAME memory_plan_box_mtp_kv8_test COMMAND memory_plan_box_test ${B70_TEST_SNAPSHOT} mtp)
    b70_kv8_twin(memory_plan_box_mtp)
  endif()
  if(TEST mtp_gpu_test)
    add_test(NAME mtp_gpu_kv8_test COMMAND mtp_gpu_test ${B70_TEST_SNAPSHOT} l0-int8)
    b70_kv8_twin(mtp_gpu)
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
# Written blind on the Mac (docs/superpowers/plans/box-validation-queue.md row 13 is the
# order to run these in). The host reference test runs everywhere; the rest need a B70, and
# the checkpoint ones SKIP (77) until the Ornith int4 checkpoint (15a) and its golden set
# `oracle-out-ornith` are on the machine, as 15c's do.
#   the reference itself: pf_moe_ref_test (host) - the sort's invariants and tile bound,
#       row independence, the combine == decode's, the whole chunk against decode's chain;
#   the kernels: pf_moe_test (a B70, no checkpoint) - sort / gathers / combine exact,
#       grouped == dense GEMMs bitwise, row independence and replay on the card;
#   R3 (prefill): prefill_gate_test on oracle-out-ornith, l0 and l0-int8, one chunk and
#       chunks of 16;
#   continuation: prefill_split_test on Ornith (splits at multiples of 64 bitwise; the other
#       bars are Qwen3.8's, PROVISIONAL here until the box measures Ornith's);
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
    kernel_pf_gated_head_GK16V32 kernel_pf_attn_prep_q16_Q16KV2 kernel_pf_attn_Q16KV2
    kernel_pf_flash_attn_Q16KV2 kernel_pf_moe_router_K2048_N272 kernel_pf_moe_E256_T8_D2048_I512
    kernel_pf_moe_gemm_K2048_N1024_SILU kernel_pf_moe_gemm_i8_K2048_N1024_SILU
    kernel_pf_moe_gemm_K512_N2048
    kernel_pf_requant_rot_L0 kernel_pf_requant_rot_L1 kernel_pf_gemm_T0 kernel_pf_gemm_T0_SILU
    kernel_pf_gemm_i8 kernel_pf_gemm_i8_SILU
    ${B70_ORNITH_DECODE_KERNELS})

  add_executable(pf_moe_test kernels/pf_moe_test.cc)
  target_include_directories(pf_moe_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(pf_moe_test PRIVATE b70_l0)
  b70_target_kernel_dir(pf_moe_test)
  add_dependencies(pf_moe_test ${B70_ORNITH_PREFILL_KERNELS})
  add_test(NAME pf_moe_test COMMAND pf_moe_test)
  set_tests_properties(pf_moe_test PROPERTIES LABELS "ornith;prefill" TIMEOUT 1800)

  add_executable(ornith_prefill_test golden/ornith_prefill_test.cc)
  target_include_directories(ornith_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(ornith_prefill_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(ornith_prefill_test)
  endif()
  add_dependencies(ornith_prefill_test ${B70_ORNITH_PREFILL_KERNELS})
  add_dependencies(prefill_gate_test ${B70_ORNITH_PREFILL_KERNELS})
  add_dependencies(prefill_split_test ${B70_ORNITH_PREFILL_KERNELS})
  foreach(b l0 l0-int8)
    set(_near 0)
    if(b STREQUAL "l0-int8")
      set(_near 1)   # the l0-int8 gate's standing near-tie allowance (spec 5), as Qwen3.8's
    endif()
    add_test(NAME prefill_gate_ornith_${b}_test COMMAND prefill_gate_test
      ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT}
      prose,code,cjk 0 ${b} ${_near})
    add_test(NAME prefill_gate_ornith_${b}_c16_test COMMAND prefill_gate_test
      ${B70_ORNITH_ORACLE_DIR} ${B70_ORNITH_PROMPTS_DIR} ${B70_ORNITH_SNAPSHOT}
      prose,code,cjk 16 ${b} ${_near})
    add_test(NAME prefill_split_ornith_${b}_test COMMAND prefill_split_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR}/long32k.ids ${b})
    add_test(NAME ornith_prefill_${b}_test COMMAND ornith_prefill_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} ${b} ${B70_ORNITH_ORACLE_DIR})
    set_tests_properties(prefill_gate_ornith_${b}_test prefill_gate_ornith_${b}_c16_test
      PROPERTIES LABELS "checkpoint;golden;prefill;ornith" SKIP_RETURN_CODE 77 TIMEOUT 3600)
    set_tests_properties(prefill_split_ornith_${b}_test ornith_prefill_${b}_test PROPERTIES
      LABELS "checkpoint;prefill;ornith" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 3600)
  endforeach()

endif()

# ==== Spec 18b: K2-Horizon decode (label `k2`) ============================ (begin) ====
# One delimited block for every K2 registration, so the branches in flight (spec 15d's
# Ornith prefill beside it) never edit the same lines. Host tests first (they run on the
# Mac: tools/mac_check.sh), then the card's (box only; docs/superpowers/plans/
# box-validation-queue.md has the order).
set(B70_K2_SNAPSHOT "urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ"
  CACHE STRING "K2-Horizon int4 checkpoint (repo id resolved in the HF cache, or a snapshot directory)")

# Task 1: the model table, the RoPE table, the loader's host half on a synthetic fixture.
add_executable(k2_horizon_test model/k2_horizon_test.cc)
target_include_directories(k2_horizon_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_horizon_test PRIVATE b70_k2_model)
add_test(NAME k2_horizon_test COMMAND k2_horizon_test ${CMAKE_SOURCE_DIR}/tests/model/k2/config.json)
add_executable(k2_rope_test loader/k2_rope_test.cc)
target_include_directories(k2_rope_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_rope_test PRIVATE b70_k2_loader)
add_test(NAME k2_rope_test COMMAND k2_rope_test)
add_executable(k2_repack_test loader/k2_repack_test.cc)
target_include_directories(k2_repack_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_repack_test PRIVATE b70_k2_loader)
add_test(NAME k2_repack_test COMMAND k2_repack_test ${CMAKE_SOURCE_DIR}/tests/model/k2/config.json)
# ... and the real checkpoint on the card: every bucket, readbacks (both head forms).
add_executable(k2_load_checkpoint_test loader/k2_load_checkpoint_test.cc)
target_include_directories(k2_load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_load_checkpoint_test PRIVATE b70_k2_loader b70_l0)
add_test(NAME k2_load_checkpoint_test COMMAND k2_load_checkpoint_test ${B70_K2_SNAPSHOT})
add_test(NAME k2_load_checkpoint_i8head_test COMMAND k2_load_checkpoint_test ${B70_K2_SNAPSHOT} int8)
set_tests_properties(k2_load_checkpoint_test k2_load_checkpoint_i8head_test PROPERTIES
  LABELS "checkpoint;k2" SKIP_RETURN_CODE 77 TIMEOUT 1800)

# Task 2: the K2 kernels' host reference (tests/kernels/k2_ref.h) against spec 18 §3 -
# the norm traps, the softplus threshold, the router (padding never selected), the
# ascending-id combine, the RoPE chain, MoVA. Host only.
add_executable(k2_ref_test kernels/k2_ref_test.cc)
target_include_directories(k2_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_ref_test PRIVATE b70_k2_loader)
add_test(NAME k2_ref_test COMMAND k2_ref_test)
# Spec 18 §10: the eager attention's host chain (k2_ref.h attention_eager) against torch's own
# scores, probabilities and output (tests/kernels/k2_attn_eager_fixture.h, generated in the
# oracle container by tools/oracle/k2_attn_eager_fixture.py), and against flash. Host only.
add_executable(k2_attn_eager_ref_test kernels/k2_attn_eager_ref_test.cc)
target_include_directories(k2_attn_eager_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME k2_attn_eager_ref_test COMMAND k2_attn_eager_ref_test)
if(B70_K2)
  # Every binary K2's decode list binds (src/kernels/CMakeLists.txt's K2 block).
  set(B70_K2_DECODE_KERNELS
    kernel_k2_embed_gather_M1_D2560_V250624 kernel_k2_argmax_stage1_M1_N250624_V250624
    kernel_k2_argmax_stage2_N250624
    kernel_prep_res_fold_M1_K2560_SP0_G20 kernel_prep_res_fold_M1_K2560_SP4_G20
    kernel_k2_norm_M1_K2560_G20_W20_NG2 kernel_k2_silu_M1_I6144_S4
    kernel_k2_attn_prep_M1_N10240_S2_Q32KV8_V kernel_k2_attn_prep_M1_N9280_S2_Q32KV8
    kernel_k2_attn_M1_T${ATTN_V2_TGT}_Q32KV8 kernel_k2_attn_eager_M1_T${ATTN_V2_TGT}_Q32KV8
    kernel_k2_route_M1_E64_T4_N9280_O9216_S2 kernel_k2_route_M1_E100_T8_N128_O0_S1
    kernel_k2_moe_M1_E100_T8_D2560_I768 kernel_k2_mova_M1_E64_T4_D2560_N1024
    kernel_gemv_M1_K2560_N10240_S2_L0 kernel_gemv_M1_K2560_N9280_S2_L0
    kernel_gemv_M1_K4096_N2560_S4_L0 kernel_gemv_M1_K2560_N12288_S4_L0
    kernel_gemv_M1_K6144_N2560_S4_L0 kernel_gemv_bf16_M1_K2560_N128_C16_S16
    kernel_gemv_bf16_M1_K2560_N250624 kernel_gemv_i8w_M1_K2560_N250624)
  # Host: the names the capture binds (kernels::k2::decode_variants) are exactly this list.
  add_executable(k2_variant_names_test kernels/k2_variant_names_test.cc)
  target_include_directories(k2_variant_names_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(k2_variant_names_test PRIVATE b70_k2_model)
  b70_target_kernel_dir(k2_variant_names_test)
  add_test(NAME k2_variant_names_test COMMAND k2_variant_names_test ${B70_K2_DECODE_KERNELS})
  # K1 on the card: every K2 kernel at K2's real shapes against k2_ref.h (no checkpoint).
  add_executable(k2_kernels_test kernels/k2_kernels_test.cc)
  target_include_directories(k2_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(k2_kernels_test PRIVATE b70_l0 b70_k2_loader)
  b70_target_kernel_dir(k2_kernels_test)
  add_dependencies(k2_kernels_test ${B70_K2_DECODE_KERNELS})
  add_test(NAME k2_kernels_test COMMAND k2_kernels_test)
  set_tests_properties(k2_kernels_test PROPERTIES LABELS k2 TIMEOUT 900)
endif()

# Task 3: the runtime. The planner and the launch count on the host ...
add_executable(k2_plan_test runtime/k2_plan_test.cc)
target_include_directories(k2_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_plan_test PRIVATE b70_k2_plan)
add_test(NAME k2_plan_test COMMAND k2_plan_test)
# ... and on the card with the real checkpoint: the list (717 launches, Review Focus 5), the
# plan against the allocation, K3 replay determinism (k2_decode_test); K2, the golden gate
# with the routing diagnostic against 18a's oracle-out-k2 (k2_golden_test, SKIP 77 until it
# exists). Both head forms (`int8`: spec 9's head, b70-serve's default once 18d serves K2).
if(B70_K2)
  set(B70_K2_ORACLE_DIR "${CMAKE_SOURCE_DIR}/oracle-out-k2"
    CACHE PATH "The CPU-oracle golden set of K2-Horizon (spec 18a): <prompt>.ids + .golden.safetensors")
  foreach(t k2_decode_test k2_golden_test)
    if(t STREQUAL "k2_decode_test")
      add_executable(${t} runtime/${t}.cc)
    else()
      add_executable(${t} golden/${t}.cc)
    endif()
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(${t} PRIVATE b70_k2_runtime b70_l0)
    add_dependencies(${t} ${B70_K2_DECODE_KERNELS})
  endforeach()
  add_test(NAME k2_decode_test COMMAND k2_decode_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/prose.ids)
  add_test(NAME k2_decode_i8head_test COMMAND k2_decode_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/prose.ids int8)
  add_test(NAME k2_golden_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR})
  add_test(NAME k2_golden_i8head_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR} int8)
  # Spec 18 §10: the same gate through the EAGER attention (k2_attn_eager.cl, the reference's
  # bf16 chain) - the box's first session runs both pairs and compares (box-validation-queue
  # row 14).
  add_test(NAME k2_golden_eager_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR})
  add_test(NAME k2_golden_eager_i8head_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR} int8)
  set_tests_properties(k2_golden_eager_test k2_golden_eager_i8head_test PROPERTIES ENVIRONMENT "B70_K2_ATTN=eager")
  set_tests_properties(k2_decode_test k2_decode_i8head_test k2_golden_test k2_golden_i8head_test
    k2_golden_eager_test k2_golden_eager_i8head_test
    PROPERTIES LABELS "checkpoint;golden;k2" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# b70-decode's model_type dispatch, before the device: tests/model/k2/ holds only the
# checkpoint's config.json, which is all the dispatch reads, so these run on any box with the
# binary (b70_cli_reject's "exit 1, no `device:` line, the message").
# (cli_reject_k2_prefill / cli_reject_k2_pp lived here until spec 18c built K2's prefill; the
# 18c block below rejects the backends K2 does not prefill on instead. cli_reject_k2_kv8 lived
# here until spec 18e built K2's int8 KV cache: `--kv-cache int8` is accepted for K2 now.)
# ==== Spec 18b: K2-Horizon decode ========================================== (end) ====

# ==== Spec 18c: K2-Horizon prefill (labels `k2;prefill`) =================== (begin) ====
# Written blind on the Mac (docs/superpowers/plans/box-validation-queue.md, the K2 prefill row,
# has the order to run these in). Host tests first (the Mac runs them: tools/mac_check.sh),
# then the card's; the checkpoint ones SKIP (77) until the K2 int4 checkpoint and 18a's
# oracle-out-k2/ are on the machine, as 18b's do.
#   k2_pf_ref_test (host): the routing scatter with ties, the tile bound, grouped-GEMM row
#       independence, the combines == decode's chains, the slab tail, the eager reference;
#   k2_variant_names_test prefill (host): every name the walk binds is built, nothing dead;
#   k2_pf_kernels_test (a B70, no checkpoint): K1 for every prefill kernel incl. the flash
#       attention at depths 0 .. 62048 against fp64, grouped == dense, Review Focus 1-3;
#   k2_prefill_test (checkpoint): the walk, K3 determinism / replay / chunking, prefill KV and
#       routing against decode's fill, tokens; its `split` mode is prefill_split on K2;
#   k2_golden_test prefill (checkpoint + oracle): K2 on prefill, the routing diagnostic.
add_executable(k2_pf_ref_test kernels/k2_pf_ref_test.cc)
target_include_directories(k2_pf_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_pf_ref_test PRIVATE b70_k2_plan)
add_test(NAME k2_pf_ref_test COMMAND k2_pf_ref_test)
set_tests_properties(k2_pf_ref_test PROPERTIES LABELS "k2;prefill")
if(B70_K2)
  # Every binary the prefill walk binds and the K1 test's ungated attention builds
  # (src/kernels/CMakeLists.txt's spec 18c block; pf_gemm_T0 / _SILU are prefill/'s own).
  set(B70_K2_PREFILL_KERNELS
    kernel_k2_pf_embed_gather_D2560_V250624 kernel_pf_res_fold_K2560_SP0_G20
    kernel_pf_res_fold_K2560_SP1_G20 kernel_k2_norm_M2048_K2560_G20_W20_NG2
    kernel_k2_pf_dequant_slab_K2560_N10240 kernel_k2_pf_dequant_slab_K2560_N9280
    kernel_k2_pf_dequant_slab_K4096_N2560 kernel_k2_pf_dequant_slab_K2560_N12288
    kernel_k2_pf_dequant_slab_K6144_N2560 kernel_pf_gemm_T0 kernel_pf_gemm_T0_SILU
    kernel_k2_attn_prep_M2048_N10240_S1_Q32KV8_V kernel_k2_attn_prep_M2048_N9472_S1_Q32KV8
    kernel_k2_pf_flash_attn_Q32KV8 kernel_k2_pf_flash_attn_Q32KV8_EAGER
    kernel_k2_pf_flash_attn_Q32KV8_O kernel_k2_pf_flash_attn_Q32KV8_EAGER_O
    kernel_k2_route_M1_E64_T4_N9472_O9216_S1 kernel_k2_pf_mova_E64_T4_D2560_N1024
    kernel_pf_moe_gemm_K2560_N1024 kernel_pf_moe_router_K2560_N128
    kernel_k2_route_M1_E100_T8_N128_O0_S1 kernel_k2_pf_moe_E100_T8_D2560_I768
    kernel_pf_moe_gemm_K2560_N1536_SILU kernel_pf_moe_gemm_K768_N2560)
  add_test(NAME k2_pf_variant_names_test COMMAND k2_variant_names_test prefill ${B70_K2_PREFILL_KERNELS})
  set_tests_properties(k2_pf_variant_names_test PROPERTIES LABELS "k2;prefill")

  add_executable(k2_pf_kernels_test kernels/k2_pf_kernels_test.cc)
  target_include_directories(k2_pf_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(k2_pf_kernels_test PRIVATE b70_prefill_host b70_k2_loader b70_k2_plan b70_l0)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(k2_pf_kernels_test)
  endif()
  b70_target_kernel_dir(k2_pf_kernels_test)
  add_dependencies(k2_pf_kernels_test ${B70_K2_PREFILL_KERNELS} ${B70_K2_DECODE_KERNELS})
  add_test(NAME k2_pf_kernels_test COMMAND k2_pf_kernels_test)
  set_tests_properties(k2_pf_kernels_test PROPERTIES LABELS "k2;prefill" TIMEOUT 1800)

  add_executable(k2_prefill_test runtime/k2_prefill_test.cc)
  target_include_directories(k2_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(k2_prefill_test PRIVATE b70_k2_prefill b70_l0)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(k2_prefill_test)
  endif()
  add_dependencies(k2_prefill_test ${B70_K2_PREFILL_KERNELS} ${B70_K2_DECODE_KERNELS})
  add_test(NAME k2_prefill_test COMMAND k2_prefill_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long.ids all)
  add_test(NAME k2_prefill_i8head_test COMMAND k2_prefill_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long.ids all int8)
  add_test(NAME k2_prefill_eager_test COMMAND k2_prefill_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long.ids all)
  set_tests_properties(k2_prefill_eager_test PROPERTIES ENVIRONMENT "B70_K2_ATTN=eager")
  add_test(NAME prefill_split_k2_test COMMAND k2_prefill_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long32k.ids split)
  set_tests_properties(k2_prefill_test k2_prefill_i8head_test k2_prefill_eager_test prefill_split_k2_test
    PROPERTIES LABELS "checkpoint;prefill;k2" SKIP_RETURN_CODE 77 TIMEOUT 3600)

  # K2 on prefill: 18b's golden test in its prefill mode (one chunk, and chunks of 16).
  target_link_libraries(k2_golden_test PRIVATE b70_k2_prefill)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(k2_golden_test)
  endif()
  add_dependencies(k2_golden_test ${B70_K2_PREFILL_KERNELS})
  add_test(NAME k2_golden_prefill_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR} prefill)
  add_test(NAME k2_golden_prefill_c16_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR}
    prefill:16)
  add_test(NAME k2_golden_prefill_i8head_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR}
    int8 prefill)
  set_tests_properties(k2_golden_prefill_test k2_golden_prefill_c16_test k2_golden_prefill_i8head_test
    PROPERTIES LABELS "checkpoint;golden;prefill;k2" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# b70-decode's K2 prefill backends, before the device (tests/model/k2: config.json only).
b70_cli_reject(cli_reject_k2_prefill_length_int8 ${CMAKE_CURRENT_SOURCE_DIR}/model/k2 --bench
  --prefill-length 64 --prefill-backend l0-int8 EXPECT "K2-Horizon prefills on the l0 backend only")
b70_cli_reject(cli_reject_k2_prefill_sycl ${CMAKE_CURRENT_SOURCE_DIR}/model/k2 --ids ${B70_OK_IDS} --n 4
  --prefill --prefill-backend sycl-tla EXPECT "K2-Horizon prefills on the l0 backend only")
# ==== Spec 18c: K2-Horizon prefill ========================================= (end) ====

# ==== Spec 15e: Ornith 1.5 35B-A3B served, and its MoE MTP head (label `ornith`) == (begin) ==
# One delimited block for every 15e registration (the branches in flight beside it edit
# their own blocks). Host tests first - they run on the Mac (tools/mac_check.sh) - then the
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
      kernel_gdn_step_slots_M${MM}_G30_GK16V32 kernel_gemv_bf16_M${MM}_K4096_N2048
      kernel_gemv_bf16_M${MM}_K2048_N9216 kernel_prep_norm_finish_M${MM}_K2048_G20_W20_X4096
      kernel_prep_res_fold_M${MM}_K2048_SP1_G20_Z kernel_attn_prep_M${MM}_Q16KV2_S1)
  endforeach()
  list(APPEND B70_ORNITH_MTP_KERNELS kernel_prep_res_fold_M1_K2048_SP1_G20)
  foreach(NV 32768 65536 131072)
    list(APPEND B70_ORNITH_MTP_KERNELS kernel_gemv_i8w_M1_K2048_N${NV} kernel_gemv_bf16_M1_K2048_N${NV})
  endforeach()
  list(APPEND B70_ORNITH_MTP_KERNELS kernel_pf_res_fold_K2048_SP1_G20_Z
    kernel_pf_norm_finish_K2048_G20_W20_X4096 kernel_pf_bf16_slab_K4096 kernel_pf_bf16_slab_K2048)
  # What the lists reuse: 15c's decode list, 15d's prefill kernels, spec 8's shared ones.
  set(_o15e_reused ${B70_ORNITH_DECODE_KERNELS}
    kernel_argmax_stage1_M2 kernel_argmax_stage1_M3 kernel_argmax_stage1_M4
    kernel_dv_argmax_N32768 kernel_dv_argmax_N65536 kernel_dv_argmax_N131072
    kernel_pf_res_fold_K2048_SP0_G20 kernel_pf_res_fold_K2048_SP1_G20
    kernel_pf_norm_finish_K2048_G20_W20 kernel_pf_embed_gather_D2048
    kernel_pf_attn_prep_q16_Q16KV2)
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
  add_test(NAME moe_m_test COMMAND moe_m_test)
  set_tests_properties(moe_m_test PROPERTIES LABELS "ornith;mtp" TIMEOUT 900)

  # On the card with the checkpoint (SKIP 77 without it): the lists, M2 bitwise rows
  # (logits, GDN slots, every layer's routes), M3 greedy --mtp K == --mtp 0 against a
  # head-less engine, the head's route row; both lm_head forms. And M1 - the head's draft
  # logits against tools/oracle/mtp_ref.py's dump on Ornith (oracle-out-ornith-mtp, spec
  # 15a's; SKIP without it) - through spec 8's own mtp_head_test.
  if(TARGET mtp_head_test)
    get_property(_o15e_k DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
    get_property(_o15e_pf DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
    add_executable(ornith_mtp_test runtime/ornith_mtp_test.cc)
    target_include_directories(ornith_mtp_test PRIVATE ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
    target_link_libraries(ornith_mtp_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
    if(B70_PREFILL_ENABLED)
      b70_link_prefill(ornith_mtp_test)
    endif()
    add_dependencies(ornith_mtp_test ${_o15e_k} ${_o15e_pf})
    add_test(NAME ornith_mtp_test COMMAND ornith_mtp_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} int8)
    add_test(NAME ornith_mtp_bf16head_test COMMAND ornith_mtp_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} bf16)
    add_test(NAME mtp_head_ornith_test COMMAND mtp_head_test ${B70_ORNITH_SNAPSHOT}
      ${B70_ORNITH_PROMPTS_DIR} ${CMAKE_SOURCE_DIR}/oracle-out-ornith-mtp/m1
      ${CMAKE_SOURCE_DIR}/oracle-out-ornith-mtp)
    set_tests_properties(ornith_mtp_test ornith_mtp_bf16head_test mtp_head_ornith_test PROPERTIES
      LABELS "checkpoint;mtp;ornith" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 3600)
  endif()
endif()
# ==== Spec 15e ============================================================= (end) ====

# ==== Spec 18d: K2-Horizon served (label `k2`) ============================== (begin) ==
# One delimited block for every 18d registration. Host tests first - they run on the Mac
# (tools/mac_check.sh) - then the card's (box-validation-queue row 25), which SKIP (77)
# without the K2 int4 checkpoint. The host side as built: spec 18 §12; the engine side
# (K2Engine behind b70-serve, KV-only snapshots): spec 18 §14.
#
# The renderer extensions - third_party/minja's b70 patches (third_party/VERSIONS) - each
# against Jinja2's own render (tools/tokenizer/dump_minja_ext.py).
add_executable(minja_ext_test tokenizer/minja_ext_test.cc)
target_include_directories(minja_ext_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(minja_ext_test PRIVATE b70_third_party_minja)
add_test(NAME minja_ext_test COMMAND minja_ext_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
# K2's tool calls and reasoning (xml, xml_typed, json): whole text = byte at a time = random
# pieces, the HF renders' assistant turns parsed back to their messages, a fuzz; and the
# server's model dispatch (server::ChatFormat) over a mock engine.
add_executable(toolcall_k2_test server/toolcall_k2_test.cc)
target_include_directories(toolcall_k2_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(toolcall_k2_test PRIVATE b70_server)
add_test(NAME toolcall_k2_test COMMAND toolcall_k2_test ${CMAKE_SOURCE_DIR}/tests/tokenizer)
add_executable(k2_server_test server/k2_server_test.cc)
target_include_directories(k2_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_server_test PRIVATE b70_server)
add_test(NAME k2_server_test COMMAND k2_server_test)
set_tests_properties(toolcall_k2_test k2_server_test PROPERTIES LABELS k2)
if(B70_TOKENIZER_ENABLED)
  # K2's chat template - the int4 repo's, the one served, vendored with its tokenizer_config
  # and generation_config at tests/tokenizer/k2/ (Apache-2.0, urakozz/IFM-K2-Horizon-MoVA-
  # 36B-A4B-W4A16-AutoRound-GPTQ at 0e38c26c) - on 18 message lists (every tool_call_format
  # and tool_presentation_format, reasoning_effort, history with and without thinking, list
  # content, tool results, null defaults, $ref schemas) against transformers'
  # apply_chat_template (tools/tokenizer/dump_k2.py).
  add_test(NAME template_k2_test COMMAND template_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/k2 k2_template_cases.json k2)
  set_tests_properties(template_k2_test PROPERTIES LABELS k2 SKIP_RETURN_CODE 77)
  # ... and its tokenizer against HF's (a text set with every tag and the BOS, corpus.txt's
  # digests, three chat renders' ids). tokenizer.json (20.6 MB) is not vendored: the int4
  # snapshot's in the HF cache, or -DB70_K2_TOKENIZER_JSON / the environment variable of the
  # same name. Absent: disabled on the Mac (host_only), SKIP (77) elsewhere.
  set(B70_K2_TOKENIZER_JSON
    "${_b70_hf_home}/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/0e38c26ca3ce5899a0cb93ae8d085669d9565720/tokenizer.json"
    CACHE FILEPATH "K2-Horizon's tokenizer.json for k2_tokenizer_test")
  add_executable(k2_tokenizer_test tokenizer/k2_tokenizer_test.cc)
  target_include_directories(k2_tokenizer_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(k2_tokenizer_test PRIVATE b70_tokenizer)
  add_test(NAME k2_tokenizer_test COMMAND k2_tokenizer_test ${CMAKE_SOURCE_DIR}/tests/tokenizer
    ${CMAKE_SOURCE_DIR}/tests/tokenizer/k2 ${B70_K2_TOKENIZER_JSON})
  set_tests_properties(k2_tokenizer_test PROPERTIES LABELS k2 SKIP_RETURN_CODE 77)
endif()
# The engine side, on the host: cli::k2::K2EngineAdapterT (b70-serve's adapter) over a
# device-free engine with K2's real KV layout - the snapshot runs on K2's shapes (bf16 and
# int8), save / load on host buffers, Review Focus 3 through spec 7's request path (KV-only
# snapshots, every cached turn bitwise a cold run's, K2's own store key), and the server: a
# greedy chat's ids = the decode run on its prompt_token_ids, a cached second turn, seeded
# sampling reproducible.
add_executable(k2_serve_test server/k2_serve_test.cc)
target_include_directories(k2_serve_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_serve_test PRIVATE b70_server b70_k2_plan)
add_test(NAME k2_serve_test COMMAND k2_serve_test)
set_tests_properties(k2_serve_test PROPERTIES LABELS k2)
# On the card (row 25), checkpoint: K2Engine's snapshot calls byte-exact, C1 bitwise at a block
# end, C2 (prefix_gpu_test's sequences a-g through PrefixSession over the adapter) - bf16 KV,
# and the `_kv8` twin under B70_KV_CACHE=int8 (spec 18e's cache).
# serve_adapters.h (PinnedAlloc) and golden_common.h, as prefix_gpu_test: it links what
# b70-serve links.
if(B70_K2 AND B70_TOKENIZER_ENABLED AND TARGET b70_k2_prefill)
  add_executable(k2_prefix_gpu_test server/k2_prefix_gpu_test.cc)
  target_include_directories(k2_prefix_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(k2_prefix_gpu_test PRIVATE b70_server b70_tokenizer b70_k2_prefill b70_runtime
    b70_loader b70_model b70_l0 b70_prefill_host)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(k2_prefix_gpu_test)
  endif()
  add_dependencies(k2_prefix_gpu_test ${B70_K2_PREFILL_KERNELS} ${B70_K2_DECODE_KERNELS})
  add_test(NAME k2_prefix_gpu_test COMMAND k2_prefix_gpu_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts)
  add_test(NAME k2_prefix_gpu_split_test COMMAND k2_prefix_gpu_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts 0 1)
  set_tests_properties(k2_prefix_gpu_test k2_prefix_gpu_split_test PROPERTIES
    LABELS "checkpoint;k2" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  if(B70_KV8)
    add_test(NAME k2_prefix_gpu_kv8_test COMMAND k2_prefix_gpu_test ${B70_K2_SNAPSHOT}
      ${CMAKE_SOURCE_DIR}/tests/golden/prompts)
    set_tests_properties(k2_prefix_gpu_kv8_test PROPERTIES ENVIRONMENT "B70_KV_CACHE=int8"
      LABELS "checkpoint;k2;kv8" SKIP_RETURN_CODE 77 TIMEOUT 3600)
  endif()
endif()
# b70-serve's K2 refusals, before the device (tests/model/k2: config.json only) - what K2's
# engine does not have, by name - and the one-card refusal of --pp (any model).
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
    set_tests_properties(${name} PROPERTIES TIMEOUT 60 LABELS k2)
  endfunction()
  b70_serve_reject(cli_reject_serve_k2_mtp ${CMAKE_CURRENT_SOURCE_DIR}/model/k2 --mtp 1
    EXPECT "K2-Horizon has no MTP head")
  b70_serve_reject(cli_reject_serve_k2_spec_mtp ${CMAKE_CURRENT_SOURCE_DIR}/model/k2 --spec mtp
    EXPECT "K2-Horizon has no MTP head")
  b70_serve_reject(cli_reject_serve_k2_lookup ${CMAKE_CURRENT_SOURCE_DIR}/model/k2 --spec lookup
    EXPECT "K2-Horizon's engine does not capture")
  b70_serve_reject(cli_reject_serve_k2_prefill_int8 ${CMAKE_CURRENT_SOURCE_DIR}/model/k2
    --prefill-backend l0-int8 EXPECT "K2-Horizon prefills on the l0 backend only")
  # Spec 16d: --pp 2 serves the Qwen family on two cards; K2 stays refused, by name.
  b70_serve_reject(cli_reject_serve_pp ${CMAKE_CURRENT_SOURCE_DIR}/model/k2 --pp 2
    EXPECT "K2-Horizon runs its own engine")
endif()
# ==== Spec 18d ============================================================= (end) ====

# ==== Spec 18e: K2-Horizon's int8 KV cache (labels `k2;kv8`) =============== (begin) ====
# rotkv at head_dim 128 (spec 12 §8 carried to K2: src/common/kv8.h hd128, src/kernels/k2/
# k2_kv8.cl), written blind on the Mac (docs/superpowers/plans/box-validation-queue.md, the K2
# int8 KV row). Host tests first (the Mac runs them), then the card's; the checkpoint ones SKIP
# (77) without the K2 int4 checkpoint / 18a's oracle-out-k2, as 18b's and 18c's do.
#   kv8_test (host, spec 12b's, extended): the 128-point scheme - torch's hadamard(128, 0) signs
#       pinned, the rotations against fp64, the quantiser, rotkv beating per token at K2's shape;
#   k2_kv8_ref_test (host): the references - the flash epilogue's FWHT decomposition == unrotate
#       bitwise, the writer, Review Focus 2 (MoVA's mix quantised after the combine), the readers;
#   k2_kv8_variant_names_test (host): every int8 binary the walks bind is built, nothing dead;
#   k2_kv8_kernels_test (a B70, no checkpoint): K1 for every int8 kernel;
#   the `_kv8` twins (checkpoint): the SAME binaries and command lines as the bf16 registrations
#       of 18b / 18c with B70_KV_CACHE=int8 (K2Engine's default form) - K3 replay and the plan,
#       the golden gate with its routing diagnostic (flash, eager, int8 head, prefill), the
#       prefill walk / determinism / chunking / prefill-vs-decode and the split test.
add_executable(k2_kv8_ref_test kernels/k2_kv8_ref_test.cc)
target_include_directories(k2_kv8_ref_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_kv8_ref_test PRIVATE b70_k2_loader)
add_test(NAME k2_kv8_ref_test COMMAND k2_kv8_ref_test)
set_tests_properties(k2_kv8_ref_test PROPERTIES LABELS "k2;kv8")
if(B70_K2 AND B70_KV8)
  # Every binary the int8 cache adds (src/kernels/CMakeLists.txt's spec 18e block).
  set(B70_K2_KV8_KERNELS
    kernel_k2_attn_prep_M1_N10240_S2_Q32KV8_V_KV8 kernel_k2_attn_prep_M1_N9280_S2_Q32KV8_KV8
    kernel_k2_mova_M1_E64_T4_D2560_N1024_STAGE kernel_k2_attn_M1_T${ATTN_V2_TGT}_Q32KV8_KV8
    kernel_k2_attn_eager_M1_T${ATTN_V2_TGT}_Q32KV8_KV8
    kernel_k2_attn_prep_M2048_N10240_S1_Q32KV8_V_KV8 kernel_k2_attn_prep_M2048_N9472_S1_Q32KV8_KV8
    kernel_k2_pf_flash_attn_Q32KV8_KV8 kernel_k2_pf_flash_attn_Q32KV8_EAGER_KV8
    kernel_k2_pf_flash_attn_Q32KV8_O_KV8 kernel_k2_pf_flash_attn_Q32KV8_EAGER_O_KV8)
  add_test(NAME k2_kv8_variant_names_test COMMAND k2_variant_names_test kv8 ${B70_K2_KV8_KERNELS})
  set_tests_properties(k2_kv8_variant_names_test PROPERTIES LABELS "k2;kv8")

  # K1 on the card: the writers bitwise, MoVA's staged row bitwise the bf16 binary's, the decode
  # flash pair within 2 ulps of fp64, eager bitwise k2_kv8_ref, the prefill flash vs fp64.
  add_executable(k2_kv8_kernels_test kernels/k2_kv8_kernels_test.cc)
  target_include_directories(k2_kv8_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(k2_kv8_kernels_test PRIVATE b70_prefill_host b70_k2_loader b70_k2_plan b70_l0)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(k2_kv8_kernels_test)
  endif()
  b70_target_kernel_dir(k2_kv8_kernels_test)
  add_dependencies(k2_kv8_kernels_test ${B70_K2_KV8_KERNELS} ${B70_K2_DECODE_KERNELS})
  add_test(NAME k2_kv8_kernels_test COMMAND k2_kv8_kernels_test)
  set_tests_properties(k2_kv8_kernels_test PROPERTIES LABELS "k2;kv8" TIMEOUT 1800)

  # The twins: B70_KV_CACHE=int8 (B70_K2_ATTN=eager on the eager ones), the bf16 tests' argv.
  foreach(t k2_decode_test k2_golden_test k2_prefill_test)
    add_dependencies(${t} ${B70_K2_KV8_KERNELS})
  endforeach()
  add_test(NAME k2_decode_kv8_test COMMAND k2_decode_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/prose.ids)
  add_test(NAME k2_golden_kv8_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR})
  add_test(NAME k2_golden_i8head_kv8_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR} int8)
  add_test(NAME k2_golden_eager_kv8_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR})
  add_test(NAME k2_golden_prefill_kv8_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR} prefill)
  add_test(NAME k2_golden_prefill_c16_kv8_test COMMAND k2_golden_test ${B70_K2_SNAPSHOT} ${B70_K2_ORACLE_DIR}
    prefill:16)
  add_test(NAME k2_prefill_kv8_test COMMAND k2_prefill_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long.ids all)
  add_test(NAME k2_prefill_eager_kv8_test COMMAND k2_prefill_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long.ids all)
  add_test(NAME prefill_split_k2_kv8_test COMMAND k2_prefill_test ${B70_K2_SNAPSHOT}
    ${CMAKE_SOURCE_DIR}/tests/golden/prompts/long32k.ids split)
  set(_k2kv8_flash k2_decode_kv8_test k2_golden_kv8_test k2_golden_i8head_kv8_test k2_golden_prefill_kv8_test
    k2_golden_prefill_c16_kv8_test k2_prefill_kv8_test prefill_split_k2_kv8_test)
  set(_k2kv8_eager k2_golden_eager_kv8_test k2_prefill_eager_kv8_test)
  set_tests_properties(${_k2kv8_flash} PROPERTIES ENVIRONMENT "B70_KV_CACHE=int8")
  set_tests_properties(${_k2kv8_eager} PROPERTIES ENVIRONMENT "B70_KV_CACHE=int8;B70_K2_ATTN=eager")
  set_tests_properties(${_k2kv8_flash} ${_k2kv8_eager} PROPERTIES
    LABELS "checkpoint;golden;k2;kv8" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()
# Passkey near the one-card int8 ceiling (opt-in, box, by hand): tools/probe/k2_passkey.sh.
# ==== Spec 18e: K2-Horizon's int8 KV cache ================================ (end) ====

# ==== Spec 16b: pipeline parallel decode across two B70s ================================
# Written blind on the Mac (no box, and no multi-device Level Zero on a Mac at all);
# docs/superpowers/plans/box-validation-queue.md row 22 is the order to run these in.
# `pp` labels the device ones: `ctest -L pp`.
#
# Host, no device: the planner (stages, weights by device, the per-device plan, auto split
# and auto length for Qwen3.8 / Agnes / Ornith, K2-Horizon's bytes per card), the peer
# hand-off's protocol on two threads (runtime/pipeline_protocol.h, pp_handoff.cl's mirror),
# and b70-decode's --pp / --pipeline-parallel-size flags (and the 2026-10-06 rename's refusals).
add_executable(pipeline_plan_test runtime/pipeline_plan_test.cc)
target_include_directories(pipeline_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pipeline_plan_test PRIVATE b70_plan b70_k2_plan)
add_test(NAME pipeline_plan_test COMMAND pipeline_plan_test)
add_executable(pp_protocol_test runtime/pp_protocol_test.cc)
target_include_directories(pp_protocol_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
find_package(Threads REQUIRED)
target_link_libraries(pp_protocol_test PRIVATE Threads::Threads)
add_test(NAME pp_protocol_test COMMAND pp_protocol_test)
add_executable(pipeline_args_test cli/pipeline_args_test.cc)
target_include_directories(pipeline_args_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pipeline_args_test PRIVATE b70_plan)
add_test(NAME pipeline_args_test COMMAND pipeline_args_test)

# On the box, both cards and the checkpoint; SKIP (77) with fewer than two GPUs or without
# peer access. P1 (pp_decode_test): ingest + 64 greedy ids and a spec 7 restore at 4395,
# --pp 2 with copy and peer, the auto split and the cuts at 5 and 62 - ids, logits,
# every layer's GDN state / conv ring / KV and both Control blocks BITWISE the single card on
# device 0; a reset and the whole run again, bitwise. P4 (pp_fail_test): a lost hand-off under
# copy and under peer throws within its bound, never hangs, and reset() recovers.
add_executable(pp_decode_test runtime/pp_decode_test.cc)
target_include_directories(pp_decode_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(pp_decode_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
if(B70_PREFILL_ENABLED)
  b70_link_prefill(pp_decode_test)
endif()
add_dependencies(pp_decode_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS} kernel_pp_handoff)
add_test(NAME pp_decode_test COMMAND pp_decode_test ${B70_TEST_SNAPSHOT})
add_test(NAME pp_decode_i8head_test COMMAND pp_decode_test ${B70_TEST_SNAPSHOT} int8)
add_executable(pp_fail_test runtime/pp_fail_test.cc)
target_include_directories(pp_fail_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(pp_fail_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(pp_fail_test ${B70_DECODE_LIST_KERNELS} kernel_pp_handoff)
add_test(NAME pp_fail_test COMMAND pp_fail_test ${B70_TEST_SNAPSHOT})
set_tests_properties(pp_decode_test pp_decode_i8head_test pp_fail_test PROPERTIES
  LABELS "checkpoint;pp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 3600)
if(B70_KV8 AND COMMAND b70_kv8_twin)
  add_dependencies(pp_decode_test ${B70_KV8_KERNELS})
  add_test(NAME pp_decode_kv8_test COMMAND pp_decode_test ${B70_TEST_SNAPSHOT})
  b70_kv8_twin(pp_decode)
endif()

# b70-decode's --pp refusals, before any device (cli/pipeline_args.h), and P4's missing
# card: with ZE_AFFINITY_MASK=0 the driver shows one GPU and --pp 2 must stop with the
# count, before the load (the model path does not exist).
b70_cli_reject(cli_reject_pipeline_value ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --pp 3
  EXPECT "--pp is the pipeline-parallel size, 1 or 2")
# (Spec 16c lifted 16b's two prefill refusals; what it refuses is in its own block below.
# Spec 16d lifted the --mtp one, cli_reject_pipeline_mtp: --pp 2 --mtp runs.)
b70_cli_reject(cli_reject_pipeline_device ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --pp 2
  --device 1 EXPECT "--device names one card")
b70_cli_reject(cli_reject_pipeline_split_alone ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4
  --pipeline-split 32 EXPECT "--pipeline-split belongs to --pp 2")
b70_cli_reject(cli_reject_pipeline_handoff_value ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4
  --pp 2 --pipeline-handoff p2p EXPECT "--pipeline-handoff expects copy or peer")
b70_cli_reject(cli_reject_pipeline_k2 ${CMAKE_CURRENT_SOURCE_DIR}/model/k2 --ids ${B70_OK_IDS}
  --n 4 --pp 2 EXPECT "K2 across two cards is future work")
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
  --n 4 --pp 2)
set_tests_properties(cli_reject_pipeline_one_gpu PROPERTIES LABELS pp TIMEOUT 60)
# ==== Spec 16b ============================================================= (end) ====

# ==== Spec 16c: pipeline parallel PREFILL across two B70s (the chunk pipeline) ==========
# Written blind on the Mac, like 16b; box-validation-queue.md row 23 is the order to run the
# device ones in (`ctest -L pp`).
#
# Host, no device: the chunks (Engine::prefill's, exactly), the host's order and its rules
# (every mutation that drops one caught), the landing slots and the per-device plan with a
# prefill path (runtime/pipeline_prefill_plan.h); and that order run by its one executor over
# two host threads standing in for the devices - copy and peer, a slowed device 1
# (back-pressure), spec 7's hooks from the shadows, a throwing hook, a lost hand-off. Clean
# under ThreadSanitizer (by hand: tools/mac/pp_prefill_tsan.sh).
add_executable(pipeline_prefill_plan_test runtime/pipeline_prefill_plan_test.cc)
target_include_directories(pipeline_prefill_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pipeline_prefill_plan_test PRIVATE b70_plan)
add_test(NAME pipeline_prefill_plan_test COMMAND pipeline_prefill_plan_test)
add_executable(pp_prefill_protocol_test runtime/pp_prefill_protocol_test.cc)
target_include_directories(pp_prefill_protocol_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(pp_prefill_protocol_test PRIVATE b70_plan Threads::Threads)
add_test(NAME pp_prefill_protocol_test COMMAND pp_prefill_protocol_test)
set_tests_properties(pp_prefill_protocol_test PROPERTIES TIMEOUT 120)

# On the box, both cards and the checkpoint; SKIP (77) with fewer than two GPUs or without peer
# access. P1 / P2 (pp_prefill_test): 4103 ids (3 chunks), 32768 (16), 2, a prefill_split
# continuation at 2500 with chunk 1000, spec 7's hooks at 2048 / 4096 (from the shadows) and
# 5000, a hook that throws, a one-card snapshot continued on two cards - logits, every layer's
# GDN state / conv ring / KV, the Control blocks and 16 decoded ids BITWISE one card on device
# 0, under copy and peer at the auto split and the cuts at 5 and 62; l0-int8 (default), l0, the
# int8 head, the int8 KV cache. Back-pressure and P4 (pp_prefill_fail_test): device 1 held back
# artificially - bitwise the clean run; a lost hand-off throws within its bound and reset()
# recovers.
add_executable(pp_prefill_test runtime/pp_prefill_test.cc)
target_include_directories(pp_prefill_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(pp_prefill_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
if(B70_PREFILL_ENABLED)
  b70_link_prefill(pp_prefill_test)
endif()
add_dependencies(pp_prefill_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS} kernel_pp_handoff)
add_test(NAME pp_prefill_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0-int8)
add_test(NAME pp_prefill_l0_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0)
add_test(NAME pp_prefill_i8head_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0-int8 int8)
add_executable(pp_prefill_fail_test runtime/pp_prefill_fail_test.cc)
target_include_directories(pp_prefill_fail_test PRIVATE ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(pp_prefill_fail_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
if(B70_PREFILL_ENABLED)
  b70_link_prefill(pp_prefill_fail_test)
endif()
add_dependencies(pp_prefill_fail_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS} kernel_pp_handoff)
add_test(NAME pp_prefill_fail_test COMMAND pp_prefill_fail_test ${B70_TEST_SNAPSHOT})
set_tests_properties(pp_prefill_test pp_prefill_l0_test pp_prefill_i8head_test pp_prefill_fail_test
  PROPERTIES LABELS "checkpoint;pp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77
  TIMEOUT 5400)
if(B70_KV8 AND COMMAND b70_kv8_twin)
  add_dependencies(pp_prefill_test ${B70_KV8_KERNELS})
  add_test(NAME pp_prefill_kv8_test COMMAND pp_prefill_test ${B70_TEST_SNAPSHOT} l0-int8)
  b70_kv8_twin(pp_prefill)
endif()

# b70-decode --pp 2 runs --prefill and --bench --prefill-length through the pipeline; what it
# refuses before any device: a sycl-tla prefill (no two-card walk). B70_PREFILL_ATTN=composed
# is refused the same way (pipeline_args_test; the registrations here pass no environment).
b70_cli_reject(cli_reject_pipeline_prefill_sycl ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --pp 2
  --prefill --prefill-backend sycl-tla EXPECT "not sycl-tla")
b70_cli_reject(cli_reject_pipeline_bench_prefill_sycl ${B70_NO_MODEL} --bench --prefill-length 64
  --pp 2 --prefill-backend sycl-tla EXPECT "--prefill-backend l0 or l0-int8")
# ==== Spec 16c ============================================================= (end) ====

# ==== Spec 16d: pipeline parallel integration (b70-serve --pp 2, MTP across the split) ====
# Written blind on the Mac, like 16b / 16c; box-validation-queue.md row 27 is the order to run
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

# On the box, both cards and the checkpoint; SKIP (77) with fewer than two GPUs or without peer
# access. Review Focus 1-3 (pp_mtp_test): MTP across the split BITWISE one card's on device 0 -
# the prose prompt's prefill (the head's KV filled on device 1), 64 speculative iterations (K 3,
# 1, 2, 0: every draft and verify logits row, the ids, the session with the live verify slot,
# the head's hidden row and KV layer), M2 on both engines (4 plain rows == one M = 4 verify),
# a 4103-id hooked prefill (spec 7 at 2048 / 4096 from the shadows, with the head's row), a
# one-card snapshot continued on two cards; copy and peer at the auto split, the cuts at 5 and
# 62; both Control blocks equal after every call. The int8 head and the int8 KV twins.
if(B70_MTP)
  add_executable(pp_mtp_test runtime/pp_mtp_test.cc)
  target_include_directories(pp_mtp_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(pp_mtp_test PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(pp_mtp_test)
  endif()
  add_dependencies(pp_mtp_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels} kernel_pp_handoff)
  add_test(NAME pp_mtp_test COMMAND pp_mtp_test ${B70_TEST_SNAPSHOT})
  add_test(NAME pp_mtp_i8head_test COMMAND pp_mtp_test ${B70_TEST_SNAPSHOT} int8)
  set_tests_properties(pp_mtp_test pp_mtp_i8head_test PROPERTIES
    LABELS "checkpoint;pp;mtp" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" SKIP_RETURN_CODE 77 TIMEOUT 5400)
  if(B70_KV8 AND COMMAND b70_kv8_twin)
    add_test(NAME pp_mtp_kv8_test COMMAND pp_mtp_test ${B70_TEST_SNAPSHOT})
    b70_kv8_twin(pp_mtp)
  endif()
endif()
# Review Focus 2 / 5 and spec 16 P2 (spec 7's C2 with --pp 2): the prefix cache's request path
# through b70-serve's adapters, two cards BITWISE one card - the plans, the final turn's ids and
# rows, the session - on sequences a-f; copy, peer, the split_last path, MTP (K 3, step_many),
# the int8 head; the int8 KV twin.
if(B70_TOKENIZER_ENABLED)
  add_executable(pp_prefix_gpu_test server/pp_prefix_gpu_test.cc)
  target_include_directories(pp_prefix_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(pp_prefix_gpu_test PRIVATE b70_server b70_tokenizer b70_runtime b70_loader
    b70_model b70_l0 b70_prefill_host)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(pp_prefix_gpu_test)
  endif()
  add_dependencies(pp_prefix_gpu_test ${B70_DECODE_LIST_KERNELS} ${B70_PREFILL_KERNELS} kernel_pp_handoff)
  set(_pp_prompts ${CMAKE_SOURCE_DIR}/tests/golden/prompts)
  add_test(NAME pp_prefix_gpu_test COMMAND pp_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${_pp_prompts} 0 copy 0)
  add_test(NAME pp_prefix_gpu_peer_test COMMAND pp_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${_pp_prompts} 0 peer 0)
  add_test(NAME pp_prefix_gpu_split_test COMMAND pp_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${_pp_prompts} 0 copy 1 int8)
  set(_pp_prefix pp_prefix_gpu_test pp_prefix_gpu_peer_test pp_prefix_gpu_split_test)
  if(B70_MTP)
    add_dependencies(pp_prefix_gpu_test ${_b70_mtp_kernels} ${_b70_mtp_pf_kernels})
    add_test(NAME pp_prefix_gpu_mtp_test COMMAND pp_prefix_gpu_test ${B70_TEST_SNAPSHOT} ${_pp_prompts} 3 copy 1 int8)
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

# b70-serve's --pp refusals before the device (cli/pipeline_serve.h): the model's with
# tests/model/kolibri1 (config.json only; K2's is 18d's cli_reject_serve_pp, its expectation
# updated), the rest before the snapshot is resolved (the model path does not exist).
if(TARGET b70-serve AND COMMAND b70_serve_reject)
  b70_serve_reject(cli_reject_serve_pp_kolibri ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --pp 2 --device 1
    EXPECT "--device names one card")   # spec 20e: Kolibri-1 serves on --pp 2 (its default), refuses --device
  b70_serve_reject(cli_reject_serve_pp_value ${B70_NO_MODEL} --pp 4096
    EXPECT "--pp is the pipeline-parallel size, 1 or 2")
  b70_serve_reject(cli_reject_serve_pp_device ${B70_NO_MODEL} --pipeline-parallel-size 2 --device 1
    EXPECT "--device names one card")
  b70_serve_reject(cli_reject_serve_pp_sycl ${B70_NO_MODEL} --pp 2 --prefill-backend sycl-tla
    EXPECT "not sycl-tla")
  b70_serve_reject(cli_reject_serve_pp_split_alone ${B70_NO_MODEL} --pipeline-split 20
    EXPECT "--pipeline-split belongs to --pp 2")
  b70_serve_reject(cli_reject_serve_pp_handoff_value ${B70_NO_MODEL} --pp 2 --pipeline-handoff p2p
    EXPECT "--pipeline-handoff expects copy or peer")
  set_tests_properties(cli_reject_serve_pp_kolibri cli_reject_serve_pp_value cli_reject_serve_pp_device
    cli_reject_serve_pp_sycl cli_reject_serve_pp_split_alone cli_reject_serve_pp_handoff_value
    PROPERTIES LABELS pp)
endif()
# ==== Spec 16d ============================================================= (end) ====

# ==== Spec 20c: Kolibri-1 decode (label kolibri) ==== (begin) ====
# One delimited block for every 20c registration (spec 16c's and 20d's blocks stay apart). Host
# tests first (they run on the Mac: tools/mac_check.sh), then the card's (box only; box-validation
# queue row 24). Real-checkpoint tests SKIP (77) until spec 20b's checkpoint exists; the synthetic
# checkpoints (tools/box_validate/kolibri_oracle.sh synth) carry KL2 / KL3 before it.
# Task 2: the model table.
add_executable(kolibri1_test model/kolibri1_test.cc)
target_include_directories(kolibri1_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri1_test PRIVATE b70_kolibri_model)
add_test(NAME kolibri1_test COMMAND kolibri1_test ${CMAKE_SOURCE_DIR}/tests/model/kolibri1/config.json
  ${CMAKE_SOURCE_DIR}/tests/model/kolibri1/synth5_bf16attn.json)
set_tests_properties(kolibri1_test PROPERTIES LABELS kolibri)
# ==== Spec 21b: Qwen3.8-Flash-Next descriptor and loader ==== (end) ====
