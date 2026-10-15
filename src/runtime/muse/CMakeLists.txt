# Spec 23b: Muse-Glimmer-30B's runtime, beside runtime::Engine, runtime/k2, runtime/kolibri and runtime/qwen4exp.
#
# b70_muse_plan: the planner (`--max-len auto`, the split by bytes) as device-free arithmetic (muse_sizes.cc) - host
# only, so tests/runtime/muse_plan_test.cc runs on the Mac. It reuses the max_len grid and spec 16b's pp_balance from
# b70_plan and the loader's byte formulas (header-only loader/muse_layout.h). 23c adds the engine beside it.
add_library(b70_muse_plan STATIC muse_sizes.cc)
target_include_directories(b70_muse_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_muse_plan PUBLIC b70_muse_model b70_plan)
