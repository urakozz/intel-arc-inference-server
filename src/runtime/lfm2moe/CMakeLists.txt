# Spec 24b Task 5: LFM2.5-24B-A2B's memory plan as device-free arithmetic (lfm2moe_sizes.cc) - host
# only, so tests/runtime/lfm2moe_plan_test.cc runs on the Mac. It reuses the loader's byte formulas
# (header-only loader/lfm2moe_layout.h) and spec 16b's pp_balance / the max_len grid from b70_plan.
# No device buffers or engine yet (24c).
add_library(b70_lfm2moe_plan STATIC lfm2moe_sizes.cc)
target_include_directories(b70_lfm2moe_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_lfm2moe_plan PUBLIC b70_lfm2moe_model b70_plan)
