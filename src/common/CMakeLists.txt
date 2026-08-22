# Header-only: the bit-level meaning of bf16/f16 and the GPTQ int4 layout,
# shared by the kernel tests, the probes and (plan 2) the loader.
add_library(b70_common INTERFACE)
target_include_directories(b70_common INTERFACE ${CMAKE_SOURCE_DIR}/src)
