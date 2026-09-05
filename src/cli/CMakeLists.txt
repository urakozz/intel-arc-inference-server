# The product binary: spec §12's `b70-decode`. It is runtime::Engine plus
# argument parsing - no logic of its own - so it links exactly what the engine
# needs and nothing more.
#
# Unlike the tests, which name the handful of variants they bind so a missing
# .bin fails fast and legibly, the CLI depends on every compiled kernel
# (the pattern tools/probe/CMakeLists.txt uses for probe_gemv): it is the whole
# product, it must be rebuilt whenever any kernel changes, and a list here
# would be a third copy of runtime::build's variant set to keep in step.
#
# `--pp` (spec 2) makes the CLI the one product binary that reaches
# `Engine::prefill`, which lives in b70_prefill_host and calls into
# libb70_prefill.so. That is guarded by B70_PREFILL_ENABLED and communicated to
# the source as `B70_HAVE_PREFILL`: with the component off the binary is
# byte-for-byte what it was, the two flags still PARSE (so their four
# rejections still fire, before the device, for the right reason), and only the
# prefill call itself is compiled out. The decode list is untouched either way
# - nothing here changes runtime::build's 774 launches or its 19 modules.
add_executable(b70-decode b70_decode.cc)
target_include_directories(b70-decode PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70-decode PRIVATE b70_runtime b70_loader b70_model b70_l0)
b70_target_kernel_dir(b70-decode)
if(B70_PREFILL_ENABLED)
  target_compile_definitions(b70-decode PRIVATE B70_HAVE_PREFILL=1)
  target_link_libraries(b70-decode PRIVATE b70_prefill_host)
  b70_link_prefill(b70-decode)
else()
  target_compile_definitions(b70-decode PRIVATE B70_HAVE_PREFILL=0)
endif()
get_property(ALL_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
# BUILDSYSTEM_TARGETS is per-directory and does not recurse, so the prefill
# kernels (a subdirectory since spec 2) need naming as well - same rule as
# above, one line rather than a list to keep in step.
get_property(PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
             PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(b70-decode ${ALL_KERNELS} ${PF_KERNELS})
