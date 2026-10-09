# Spec 21b: Qwen3.8-Flash-Next's runtime, beside runtime::Engine, runtime/k2 and runtime/kolibri (spec 21 §6).
#
# b70_qwen4exp_plan: the weights' and the persistent state's sizes, the planner (--layers auto, --max-len auto)
# and the split by bytes as device-free arithmetic (qwen4exp_sizes.cc) - host only, so
# tests/runtime/qwen4exp_plan_test.cc runs on the Mac. It reuses runtime::format_memory, the max_len grid and
# spec 16b's pp_balance from b70_plan, and the loader's byte formulas (header-only loader/qwen4exp_layout.h,
# loader/qwen4exp_ple_hash.h). 21c adds the engine (b70_qwen4exp_runtime) here.
add_library(b70_qwen4exp_plan STATIC qwen4exp_sizes.cc)
target_include_directories(b70_qwen4exp_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_qwen4exp_plan PUBLIC b70_qwen4exp_model b70_plan)

# Spec 21c: b70_qwen4exp_runtime - the buffers, the capture and the decode loop on one or two devices (Level
# Zero). It links b70_runtime for spec 16b's pieces (runtime::PipelineLink, StageLink, the pp_handoff binary) and
# the loader (load_qwen4exp's parts, the PLE table in host USM).
add_library(b70_qwen4exp_runtime STATIC qwen4exp_buffers.cc qwen4exp_capture.cc qwen4exp_engine.cc)
target_include_directories(b70_qwen4exp_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_qwen4exp_runtime PUBLIC b70_l0 b70_qwen4exp_loader b70_qwen4exp_plan b70_runtime)
b70_target_kernel_dir(b70_qwen4exp_runtime)
