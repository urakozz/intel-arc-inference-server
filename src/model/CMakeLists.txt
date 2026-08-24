# The model description as data - no device, no I/O, no library dependency.
# Plan 3's capture code walks this table. The one header it borrows is
# `loader/small_layout.h`, a code-free set of block offsets shared with the
# loader and (plan 3) the kernels so the small-tensor table's destination
# offsets have exactly one definition; it adds no link dependency.
add_library(b70_model STATIC qwen35.cc)
target_include_directories(b70_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
