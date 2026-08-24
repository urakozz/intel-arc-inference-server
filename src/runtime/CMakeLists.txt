# The decode step's runtime state: the shared control block and every device
# allocation the captured list touches besides the weights. It links b70_model
# because the buffer sizes are the model table's shapes and constants, and
# b70_l0 because the allocations are Level Zero ones. No kernels, no I/O.
add_library(b70_runtime STATIC buffers.cc)
target_include_directories(b70_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_runtime PUBLIC b70_l0 b70_model)
