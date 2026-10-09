# The loader turns a checkpoint into the canonical device buffers the measured
# kernels expect. It links b70_model because the model description (what linears
# exist, their fusion and shape) is the table it walks - docs/13-loader.md.
add_library(b70_loader STATIC snapshot.cc safetensors.cc quant.cc loader.cc lm_head_int8.cc fold.cc
            trained_context.cc draft_vocab.cc moe.cc ab.cc)
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

# --- Spec 20c: Kolibri-1 -------------------------------------------------------------------
# The Kolibri-1 loader, beside loader.cc and the K2 loader (neither touched): the host half
# (kolibri1_repack.cc, kolibri1_rope.cc - no Level Zero, run on any host by
# tests/loader/kolibri1_repack_test.cc) and the upload onto one or two devices (kolibri1_loader.cc).
add_library(b70_kolibri_loader STATIC kolibri1_repack.cc kolibri1_rope.cc kolibri1_loader.cc)
target_include_directories(b70_kolibri_loader PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_kolibri_loader PUBLIC b70_loader b70_kolibri_model b70_l0)

# --- Spec 21b: Qwen3.8-Flash-Next (qwen4_exp) -----------------------------------------------------
# The qwen4_exp loader, beside loader.cc, the K2 and the Kolibri loaders (none touched): the host half
# (qwen4exp_repack.cc, qwen4exp_ple.cc - no Level Zero, run on any host by tests/loader/qwen4exp_repack_test.cc
# and qwen4exp_ple_test.cc), the PLE table's pinning into host USM (qwen4exp_ple_usm.cc, Level Zero: its own
# object so a host test that uses the hash or the file links no device call) and the upload onto one or two
# devices (qwen4exp_loader.cc). It links the host-only planner (b70_qwen4exp_plan): the load runs the plan and
# refuses what does not fit before the first allocation.
add_library(b70_qwen4exp_loader STATIC qwen4exp_repack.cc qwen4exp_ple.cc qwen4exp_ple_usm.cc qwen4exp_loader.cc)
target_include_directories(b70_qwen4exp_loader PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_qwen4exp_loader PUBLIC b70_loader b70_qwen4exp_model b70_qwen4exp_plan b70_l0)
