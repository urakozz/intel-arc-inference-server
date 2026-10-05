# The model description as data - no device, no I/O, no library dependency.
# Plan 3's capture code walks this table. The one header it borrows is
# `loader/small_layout.h`, a code-free set of block offsets shared with the
# loader and (plan 3) the kernels so the small-tensor table's destination
# offsets have exactly one definition; it adds no link dependency.
add_library(b70_model STATIC qwen35.cc model_desc.cc)
target_include_directories(b70_model PUBLIC ${CMAKE_SOURCE_DIR}/src)

# --- Spec 18b: K2-Horizon ------------------------------------------------------------
# K2-Horizon as data, BESIDE qwen3_5 (spec 18 §5.1): it shares GemvShape / WeightKind /
# Fuse from qwen35.h and nothing of the Qwen layer plan. Host only, like b70_model.
add_library(b70_k2_model STATIC k2_horizon.cc)
target_include_directories(b70_k2_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_k2_model PUBLIC b70_model)
