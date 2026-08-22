add_executable(probe_bw probe_bw.cc)
target_include_directories(probe_bw PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_bw PRIVATE b70_l0)
b70_target_kernel_dir(probe_bw)
add_dependencies(probe_bw kernel_bw_sum)

add_executable(probe_replay probe_replay.cc)
target_include_directories(probe_replay PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_replay PRIVATE b70_l0)
b70_target_kernel_dir(probe_replay)
add_dependencies(probe_replay kernel_noop kernel_ctrl_read)

add_executable(probe_gemv probe_gemv.cc)
target_include_directories(probe_gemv PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_gemv PRIVATE b70_l0)
b70_target_kernel_dir(probe_gemv)
# depends on every gemv variant; the kernels target list is in src/kernels
get_property(ALL_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_gemv ${ALL_KERNELS})
