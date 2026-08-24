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

add_executable(safetensors_test loader/safetensors_test.cc)
target_include_directories(safetensors_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(safetensors_test PRIVATE b70_loader)
add_test(NAME safetensors_test COMMAND safetensors_test)

add_executable(quant_test loader/quant_test.cc)
target_include_directories(quant_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(quant_test PRIVATE b70_loader)
add_test(NAME quant_test COMMAND quant_test)

add_executable(dequant_fixture_test loader/dequant_fixture_test.cc)
target_include_directories(dequant_fixture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dequant_fixture_test PRIVATE b70_loader)
add_test(NAME dequant_fixture_test COMMAND dequant_fixture_test ${CMAKE_SOURCE_DIR}/tests/golden/dequant_fixture.safetensors)

add_executable(qwen35_test model/qwen35_test.cc)
target_include_directories(qwen35_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen35_test PRIVATE b70_model)
add_test(NAME qwen35_test COMMAND qwen35_test)

# The model table and src/kernels/CMakeLists.txt are two lists that have to
# agree: every row's compiled variant must exist before plan 3 binds it. No
# device, no checkpoint - just the filenames.
add_executable(kernel_table_test kernels/kernel_table_test.cc)
target_include_directories(kernel_table_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kernel_table_test PRIVATE b70_model)
b70_target_kernel_dir(kernel_table_test)
add_dependencies(kernel_table_test
  kernel_gemv_M1_K5120_N16384_S1_L1 kernel_gemv_M1_K6144_N5120_S16_L1
  kernel_gemv_M1_K5120_N34816_S4_L1 kernel_gemv_M1_K17408_N5120_S16_L1
  kernel_gemv_M1_K5120_N14336_S1_L1
  kernel_gemv_bf16_M1_K5120_N128 kernel_gemv_bf16_M1_K5120_N248320)
add_test(NAME kernel_table_test COMMAND kernel_table_test)

# Needs the real 19 GB checkpoint and a B70: label `checkpoint` so it can be
# excluded (`ctest -LE checkpoint`) where the model or the card is absent.
add_executable(load_checkpoint_test loader/load_checkpoint_test.cc)
target_include_directories(load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(load_checkpoint_test PRIVATE b70_loader b70_model)
add_test(NAME load_checkpoint_test COMMAND load_checkpoint_test)
set_tests_properties(load_checkpoint_test PROPERTIES LABELS checkpoint TIMEOUT 600)
