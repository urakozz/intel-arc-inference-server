# Host-side compute that runs on the CPU instead of the cards (PERF-124: Flash-Next's host-RAM experts).
# Built with the repo's -march=broadwell; the AVX-512 kernel carries its own `target` attribute and is chosen at runtime.
add_library(b70_cpu expert_int4.cc expert_q4.cc)
# expert_q4.cc is bit for bit the card's q4_moe: no product-sum contracted into an fma behind the code's back.
set_source_files_properties(expert_q4.cc PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
target_include_directories(b70_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_cpu PUBLIC b70_common pthread)
