# Spec 25b Task 5: Nemotron 3.5 Bolt's memory plan as device-free arithmetic (bolt_sizes.cc) - host only, so
# tests/runtime/bolt_plan_test.cc runs on the Mac. It reuses the loader's byte formulas (header-only loader/bolt_layout.h)
# and spec 16b's pp_balance / the max_len grid from b70_plan. No device buffers or engine yet (25c).
add_library(b70_bolt_plan STATIC bolt_sizes.cc)
target_include_directories(b70_bolt_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_bolt_plan PUBLIC b70_bolt_model b70_plan)
