# Spec 23b: Muse-Glimmer-30B's runtime, beside runtime::Engine, runtime/k2, runtime/kolibri and runtime/qwen4exp.
#
# b70_muse_plan: the planner (`--max-len auto`, the split by bytes) as device-free arithmetic (muse_sizes.cc) - host
# only, so tests/runtime/muse_plan_test.cc runs on the Mac. It reuses the max_len grid and spec 16b's pp_balance from
# b70_plan and the loader's byte formulas (header-only loader/muse_layout.h). 23c adds the engine beside it.
# 23c: + muse_rope.cc, the sliding layers' RoPE table on the host (the engine uploads it; muse_ref_test holds it to
# transformers' rotary module).
add_library(b70_muse_plan STATIC muse_sizes.cc muse_rope.cc)
target_include_directories(b70_muse_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_muse_plan PUBLIC b70_muse_model b70_plan)

# Spec 23c: b70_muse_runtime - the decode buffers, the captured list and MuseEngine on one or two devices (Level Zero).
# It links b70_runtime for spec 16b's pieces (runtime::PipelineLink, StageLink, the pp_handoff binary).
add_library(b70_muse_runtime STATIC muse_buffers.cc muse_capture.cc muse_engine.cc)
target_include_directories(b70_muse_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_muse_runtime PUBLIC b70_l0 b70_muse_loader b70_muse_plan b70_runtime)
b70_target_kernel_dir(b70_muse_runtime)
