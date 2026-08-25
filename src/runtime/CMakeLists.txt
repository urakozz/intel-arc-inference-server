# The decode step's runtime state: the shared control block and every device
# allocation the captured list touches besides the weights (buffers.cc), plus
# the capture itself (capture.cc) - the one place every kernel of plan 3 is
# bound to those allocations and to plan 2's weights, in spec §9.1's order.
# It links b70_model because the buffer sizes and the capture's walk are the
# model table's shapes and constants, b70_loader because the capture binds
# `loader::LoadedModel`'s device weights, and b70_l0 because the allocations
# and the command list are Level Zero ones. b70_target_kernel_dir supplies
# B70_KERNEL_DIR, which kernels/kernels.h needs to name a variant's binary.
add_library(b70_runtime STATIC buffers.cc capture.cc)
target_include_directories(b70_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_runtime PUBLIC b70_l0 b70_model b70_loader)
b70_target_kernel_dir(b70_runtime)
