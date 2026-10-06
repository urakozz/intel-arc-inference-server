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
# `--prefill-length` (spec 2) makes the CLI the one product binary that reaches
# `Engine::prefill`, which lives in b70_prefill_host. Since spec 2.1, the L0
# backend's GEMMs are in b70_prefill_host itself, so that library is linked
# unconditionally; only the sycl-tla backend needs libb70_prefill.so, guarded
# by B70_PREFILL_ENABLED (`b70_link_prefill`). With the component off, sycl-tla
# is refused at runtime (runtime::prefill::sycl_available() is false) rather
# than compiled out - the decode list is untouched either way, nothing here
# changes runtime::build's 774 launches or its 19 modules.
add_executable(b70-decode b70_decode.cc)
target_include_directories(b70-decode PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70-decode PRIVATE b70_runtime b70_loader b70_model b70_l0 b70_prefill_host)
b70_target_kernel_dir(b70-decode)
if(B70_PREFILL_ENABLED)
  b70_link_prefill(b70-decode)
endif()
get_property(ALL_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
# BUILDSYSTEM_TARGETS is per-directory and does not recurse, so the prefill
# kernels (a subdirectory since spec 2) need naming as well - same rule as
# above, one line rather than a list to keep in step.
get_property(PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
             PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(b70-decode ${ALL_KERNELS} ${PF_KERNELS})

if(B70_TOKENIZER_ENABLED)
  add_executable(b70-serve b70_serve.cc)
  target_include_directories(b70-serve PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(b70-serve PRIVATE b70_runtime b70_loader b70_model b70_l0 b70_tokenizer b70_server b70_prefill_host)
  b70_target_kernel_dir(b70-serve)
  if(B70_PREFILL_ENABLED)
    b70_link_prefill(b70-serve)
  endif()
  add_dependencies(b70-serve ${ALL_KERNELS} ${PF_KERNELS})
endif()

# Spec 18b: b70-decode dispatches on config.json's model_type; K2-Horizon runs its own engine
# (runtime/k2, src/cli/k2_decode.h). Appended, so the lines above are unchanged.
target_link_libraries(b70-decode PRIVATE b70_k2_runtime)
# Spec 8: b70-decode --mtp auto runs b70-serve's per-request policy (server::AdaptiveK, host
# code in b70_server); appended, as the K2 line above.
target_link_libraries(b70-decode PRIVATE b70_server)
# Spec 18c: and K2's prefill (b70_k2_prefill: K2Engine::prefill, its walk and scratch).
# b70-decode already links the prefill component when it is built (b70_link_prefill above).
target_link_libraries(b70-decode PRIVATE b70_k2_prefill)
# Spec 20c: and Kolibri-1's engine (runtime/kolibri, src/cli/kolibri_decode.h), dispatched on model_type.
target_link_libraries(b70-decode PRIVATE b70_kolibri_runtime)
