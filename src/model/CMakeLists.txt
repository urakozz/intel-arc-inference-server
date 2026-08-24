# The model description as data - no device, no loader, no I/O. Plan 3's
# capture code walks this table; nothing here may grow a dependency.
add_library(b70_model STATIC qwen35.cc)
target_include_directories(b70_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
