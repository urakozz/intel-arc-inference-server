# Spec 20c: Kolibri-1's runtime, beside runtime::Engine and runtime/k2 (spec 20 §4: KolibriEngine and
# its capture; the qwen3_5 and K2 paths are untouched).
#
# b70_kolibri_plan: the allocations' sizes, the planner and the split by bytes as device-free
# arithmetic (kolibri_sizes.cc) - host only, so tests/runtime/kolibri_plan_test.cc runs on the Mac.
# It reuses runtime::format_memory, the max_len grid and spec 16b's pp_balance / landing layout from
# b70_plan, and the loader's byte formulas (header-only loader/kolibri1_layout.h).
add_library(b70_kolibri_plan STATIC kolibri_sizes.cc)
target_include_directories(b70_kolibri_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_kolibri_plan PUBLIC b70_kolibri_model b70_plan)

# b70_kolibri_runtime: the buffers, the capture and the decode loop on one or two devices - Level Zero.
# It links b70_runtime for spec 16b's pieces (runtime::PipelineLink, StageLink, the pp_handoff binary).
add_library(b70_kolibri_runtime STATIC kolibri_buffers.cc kolibri_capture.cc kolibri_engine.cc)
target_include_directories(b70_kolibri_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_kolibri_runtime PUBLIC b70_l0 b70_kolibri_loader b70_kolibri_plan b70_runtime)
b70_target_kernel_dir(b70_kolibri_runtime)
