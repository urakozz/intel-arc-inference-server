# The loader turns a checkpoint into the canonical device buffers the measured
# kernels expect. It links b70_model because the model description (what linears
# exist, their fusion and shape) is the table it walks - docs/13-loader.md.
add_library(b70_loader STATIC snapshot.cc safetensors.cc quant.cc loader.cc lm_head_int8.cc fold.cc
            trained_context.cc)
find_package(Threads REQUIRED)
target_include_directories(b70_loader PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_loader PUBLIC b70_l0 b70_model Threads::Threads)
