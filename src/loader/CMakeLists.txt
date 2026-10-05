# The loader turns a checkpoint into the canonical device buffers the measured
# kernels expect. It links b70_model because the model description (what linears
# exist, their fusion and shape) is the table it walks - docs/13-loader.md.
add_library(b70_loader STATIC snapshot.cc safetensors.cc quant.cc loader.cc lm_head_int8.cc fold.cc
            trained_context.cc draft_vocab.cc moe.cc)
find_package(Threads REQUIRED)
target_include_directories(b70_loader PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_loader PUBLIC b70_l0 b70_model Threads::Threads)

# --- Spec 18b: K2-Horizon ------------------------------------------------------------
# The K2 loader, beside loader.cc (which it does not touch): the host half (k2_repack.cc,
# k2_rope.cc - no Level Zero, run on any host by tests/loader/k2_repack_test.cc) and the
# upload (k2_loader.cc). It reuses this directory's public pieces through b70_loader.
add_library(b70_k2_loader STATIC k2_repack.cc k2_rope.cc k2_loader.cc)
target_include_directories(b70_k2_loader PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_k2_loader PUBLIC b70_loader b70_k2_model b70_l0)
