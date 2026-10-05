# Spec 18b: K2-Horizon's runtime, beside runtime::Engine (spec 18 §5.1: `K2Engine` and its
# capture; the qwen3_5 paths are untouched).
#
# b70_k2_plan: the allocations' sizes and the memory planner as device-free arithmetic
# (k2_sizes.cc) - host only, so tests/runtime/k2_plan_test.cc runs on the Mac. It reuses
# runtime::format_memory and the max_len grid from b70_plan.
add_library(b70_k2_plan STATIC k2_sizes.cc)
target_include_directories(b70_k2_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_k2_plan PUBLIC b70_k2_model b70_plan)

# b70_k2_runtime: the buffers, the capture and the decode loop - Level Zero.
add_library(b70_k2_runtime STATIC k2_buffers.cc k2_capture.cc k2_engine.cc)
target_include_directories(b70_k2_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_k2_runtime PUBLIC b70_l0 b70_k2_loader b70_k2_plan b70_runtime)
b70_target_kernel_dir(b70_k2_runtime)
