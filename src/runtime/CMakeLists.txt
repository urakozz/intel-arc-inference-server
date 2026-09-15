# The decode step's runtime state: the shared control block and every device
# allocation the captured list touches besides the weights (buffers.cc), plus
# the capture itself (capture.cc) - the one place every kernel of plan 3 is
# bound to those allocations and to plan 2's weights, in spec §9.1's order.
# It links b70_model because the buffer sizes and the capture's walk are the
# model table's shapes and constants, b70_loader because the capture binds
# `loader::LoadedModel`'s device weights, and b70_l0 because the allocations
# and the command list are Level Zero ones. b70_target_kernel_dir supplies
# B70_KERNEL_DIR, which kernels/kernels.h needs to name a variant's binary.
#
# engine.cc is the decode loop that owns a capture: ingestion, greedy
# generation, reset and the max_len precondition. It adds no dependency - the
# Engine is the three files above wired to a queue and a fence.
add_library(b70_runtime STATIC buffers.cc capture.cc engine.cc)
target_include_directories(b70_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_runtime PUBLIC b70_l0 b70_model b70_loader)
b70_target_kernel_dir(b70_runtime)

# The prefill path's host half - its own archive, so nothing that links
# b70_runtime acquires a dependency on libb70_prefill.so. Built in EVERY
# configuration (spec 2.1 §3.5): only libb70_prefill.so itself - the SYCL
# component - is optional (cmake/prefill.cmake).
add_subdirectory(prefill)
