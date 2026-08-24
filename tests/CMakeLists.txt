# Each test is add_executable + b70_target_kernel_dir + add_test. No framework:
# tests/check.h exits non-zero on the first failed CHECK.

add_executable(copy_test l0/copy_test.cc)
target_include_directories(copy_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(copy_test PRIVATE b70_l0)
add_test(NAME copy_test COMMAND copy_test)

add_executable(launch_test l0/launch_test.cc)
target_include_directories(launch_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(launch_test PRIVATE b70_l0)
b70_target_kernel_dir(launch_test)
add_dependencies(launch_test kernel_noop)
add_test(NAME launch_test COMMAND launch_test)

add_executable(arg_capture_test l0/arg_capture_test.cc)
target_include_directories(arg_capture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(arg_capture_test PRIVATE b70_l0)
b70_target_kernel_dir(arg_capture_test)
add_dependencies(arg_capture_test kernel_noop)
add_test(NAME arg_capture_test COMMAND arg_capture_test)

add_executable(int4_test kernels/int4_test.cc)
target_include_directories(int4_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME int4_test COMMAND int4_test)

add_executable(gemv_test kernels/gemv_test.cc)
target_include_directories(gemv_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_test)
add_dependencies(gemv_test kernel_gemv_M1_K6144_N5120_S1_L0 kernel_gemv_M1_K6144_N5120_S4_L1
  kernel_gemv_M1_K17408_N5120_S8_L0 kernel_gemv_M1_K5120_N34816_S2_L1 kernel_gemv_M2_K6144_N5120_S1_L0)
add_test(NAME gemv_test COMMAND gemv_test)

add_executable(gemv_bf16_test kernels/gemv_bf16_test.cc)
target_include_directories(gemv_bf16_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_bf16_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_bf16_test)
add_dependencies(gemv_bf16_test kernel_gemv_bf16_M1_K5120_N128 kernel_gemv_bf16_M1_K5120_N248320)
add_test(NAME gemv_bf16_test COMMAND gemv_bf16_test)

add_executable(json_test common/json_test.cc)
target_include_directories(json_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME json_test COMMAND json_test)
