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
