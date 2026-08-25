# The product binary: spec §12's `b70-decode`. It is runtime::Engine plus
# argument parsing - no logic of its own - so it links exactly what the engine
# needs and nothing more.
#
# Unlike the tests, which name the handful of variants they bind so a missing
# .bin fails fast and legibly, the CLI depends on every compiled kernel
# (the pattern tools/probe/CMakeLists.txt uses for probe_gemv): it is the whole
# product, it must be rebuilt whenever any kernel changes, and a list here
# would be a third copy of runtime::build's variant set to keep in step.
add_executable(b70-decode b70_decode.cc)
target_include_directories(b70-decode PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70-decode PRIVATE b70_runtime b70_loader b70_model b70_l0)
b70_target_kernel_dir(b70-decode)
get_property(ALL_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(b70-decode ${ALL_KERNELS})
