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

# Spec 8 §3.6 (plan 8c): the server over a scripted speculative engine, host only.
add_executable(mtp_server_test server/mtp_server_test.cc)
target_include_directories(mtp_server_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(mtp_server_test PRIVATE b70_server)
add_test(NAME mtp_server_test COMMAND mtp_server_test)

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
  # Spec 5's h8 path, which `--pp-backend l0-int8` runs (pf_colmax_rot is in the
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
# "--pp-backend belongs to --pp and --prefill" (spec 2.1).
b70_cli_reject(cli_reject_pp_backend_with_ids ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --pp-backend l0
  EXPECT "--pp-backend belongs to --pp and --prefill")
# "--pp-backend expects sycl-tla, l0 or l0-int8".
b70_cli_reject(cli_reject_pp_backend_value ${B70_NO_MODEL} --bench --pp 64 --pp-backend cuda
  EXPECT "--pp-backend expects sycl-tla, l0 or l0-int8")
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
# `--pp` / `--pp-chunk` / `--prefill` (spec 2, interfaces.md's CLI contract and ruling A8).
# These five are host-side argument validation and are compiled UNCONDITIONALLY
# - with -DB70_PREFILL=OFF the flags still parse and still reject for these
# reasons, so these cases grade the CLI's contract rather than the presence of
# the SYCL component. Each names its message: "exit 1 before the device" is
# also what an unknown flag does, and without EXPECT these cases would pass on a
# binary that had never heard of the flag.
b70_cli_reject(cli_reject_pp_without_bench ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --pp 64
  EXPECT "--pp belongs to --bench")
b70_cli_reject(cli_reject_pp_and_depth ${B70_NO_MODEL} --bench --pp 64 --depth 64
  EXPECT "--pp and --depth are exclusive")
b70_cli_reject(cli_reject_pp_zero ${B70_NO_MODEL} --bench --pp 0
  EXPECT "--pp 0 would prefill nothing")
b70_cli_reject(cli_reject_pp_chunk_without_pp ${B70_NO_MODEL} --bench --pp-chunk 1024
  EXPECT "--pp-chunk belongs to --pp")
# `--prefill` is the ids-mode equivalent of `--pp`; it must not be accepted as
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
# No int4 Ornith checkpoint exists yet (spec 15 decision 1; 15a decides the source -
# AutoRound g64 proposed). B70_ORNITH_SNAPSHOT names where it will be; until it is on the
# machine ornith_decode_test SKIPs (77), and the golden registrations SKIP on the absent
# `oracle-out-ornith` (15a's golden set) before they load anything. If 15a writes Ornith
# prompt ids beside the committed ones (its tokenizer differs), point
# B70_ORNITH_PROMPTS_DIR at them.
#   R3 (decode): golden_gate_test, descriptor-driven since 15c (the tap is the layer
#       output when the FFN folds inside the layer), bf16 and int8 heads;
#   R2 + the list + replay bitwise + the plan: ornith_decode_test.
if(B70_ORNITH)
  set(B70_ORNITH_SNAPSHOT "urakozz/Ornith-1.5-35B-A3B-W4A16-g64-AutoRound-GPTQ"
    CACHE STRING "Ornith int4 checkpoint (repo id resolved in the HF cache, or a snapshot directory)")
  set(B70_ORNITH_ORACLE_DIR "${CMAKE_SOURCE_DIR}/oracle-out-ornith"
    CACHE PATH "The CPU-oracle golden set of Ornith (spec 15a), with router_logits.L* for R2")
  set(B70_ORNITH_PROMPTS_DIR "${CMAKE_SOURCE_DIR}/tests/golden/prompts"
    CACHE PATH "Ornith's prompt ids (the committed ones unless 15a writes Ornith's beside them)")
  set(B70_ORNITH_DECODE_KERNELS
    kernel_embed_gather_M1_D2048 kernel_argmax_stage1_M1 kernel_argmax_stage2
    kernel_prep_res_fold_M1_K2048_SP0_G20 kernel_prep_res_fold_M1_K2048_SP4_G20
    kernel_prep_norm_finish_M1_K2048_G20_W20 kernel_prep_gated_head_M1_GK16V32
    kernel_gdn_step_M1_GK16V32 kernel_attn_prep_M1_Q16KV2 kernel_attn_v2_M1_T${ATTN_V2_TGT}_Q16KV2
    kernel_gemv_M1_K2048_N12288_S1_L1 kernel_gemv_M1_K4096_N2048_S4_L0
    kernel_gemv_M1_K2048_N9216_S2_L0 kernel_gemv_M1_K2048_N248320_S1_L1
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
