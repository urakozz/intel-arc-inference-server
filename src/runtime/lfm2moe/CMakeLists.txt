# Spec 24b Task 5: LFM2.5-24B-A2B's memory plan as device-free arithmetic (lfm2moe_sizes.cc) - host
# only, so tests/runtime/lfm2moe_plan_test.cc runs on the Mac. It reuses the loader's byte formulas
# (header-only loader/lfm2moe_layout.h) and spec 16b's pp_balance / the max_len grid from b70_plan.
# No device buffers or engine yet (24c).
add_library(b70_lfm2moe_plan STATIC lfm2moe_sizes.cc)
target_include_directories(b70_lfm2moe_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_lfm2moe_plan PUBLIC b70_lfm2moe_model b70_plan)

# Spec 24c: b70_lfm2moe_runtime - the decode buffers, the capture and the decode loop on one or two devices (Level
# Zero). It links b70_runtime for spec 16b's pieces (runtime::PipelineLink, StageLink, ProfileEvents, the pp_handoff
# binary) and reads the kernels' names (kernels::lfm2moe, src/kernels/lfm2moe_kernels.h).
add_library(b70_lfm2moe_runtime STATIC lfm2moe_buffers.cc lfm2moe_capture.cc lfm2moe_engine.cc)
target_include_directories(b70_lfm2moe_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_lfm2moe_runtime PUBLIC b70_l0 b70_lfm2moe_loader b70_lfm2moe_plan b70_runtime)
b70_target_kernel_dir(b70_lfm2moe_runtime)
