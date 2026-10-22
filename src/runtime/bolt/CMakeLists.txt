# Spec 25b Task 5: Nemotron 3.5 Bolt's memory plan as device-free arithmetic (bolt_sizes.cc) - host only, so
# tests/runtime/bolt_plan_test.cc runs on the Mac. It reuses the loader's byte formulas (header-only loader/bolt_layout.h)
# and spec 16b's pp_balance / the max_len grid from b70_plan. No device buffers or engine yet (25c).
add_library(b70_bolt_plan STATIC bolt_sizes.cc)
target_include_directories(b70_bolt_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_bolt_plan PUBLIC b70_bolt_model b70_plan)

# Spec 25c: b70_bolt_runtime - the decode buffers, the capture and the decode loop on one or two devices (Level Zero). It
# links b70_runtime for spec 16b's pieces (runtime::PipelineLink, StageLink, ProfileEvents, the pp_handoff binary) and
# reads the kernels' names (kernels::bolt, src/kernels/bolt_kernels.h).
add_library(b70_bolt_runtime STATIC bolt_buffers.cc bolt_capture.cc bolt_engine.cc)
target_include_directories(b70_bolt_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_bolt_runtime PUBLIC b70_l0 b70_bolt_loader b70_bolt_plan b70_runtime)
b70_target_kernel_dir(b70_bolt_runtime)

# Spec 25d: Nemotron 3.5 Bolt's prefill (the chunk walk, its scratch, and BoltEngine::prefill's definitions) - its own
# archive, as b70_lfm2moe_prefill is beside b70_lfm2moe_runtime: a target that never prefills Bolt links what it always
# linked. It runs the prefill path's Level Zero half only (runtime::prefill::Context, KernelCache, gemm_l0), but Context's
# object names the SYCL side's entry points (libb70_prefill.so when the component is built), so an executable linking this
# calls b70_link_prefill when B70_PREFILL_ENABLED, as every b70_prefill_host user does.
add_library(b70_bolt_prefill STATIC bolt_prefill.cc bolt_prefill_engine.cc)
target_include_directories(b70_bolt_prefill PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_bolt_prefill PUBLIC b70_bolt_runtime b70_prefill_host)
b70_target_kernel_dir(b70_bolt_prefill)
