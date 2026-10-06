# The prefill path's host half (spec 2.1 §3.5: in EVERY build). context.cc is the Level Zero
# list; the SYCL side lives in libb70_prefill.so and is reached through sycl_side.h. Exactly
# one of backend_sycl.cc / backend_sycl_absent.cc is compiled, so the walk never carries an
# #ifdef and a build without icpx still has the L0 backend.
set(B70_PREFILL_HOST_SRC context.cc kernels.cc gdn.cc dequant.cc attn.cc step.cc
    engine_prefill.cc profile.cc gemm_l0.cc linear_l0.cc int8.cc
    moe.cc      # spec 15d: a MoE layer's FFN on the prefill path
    pipeline_prefill.cc)   # spec 16c: PipelineEngine::prefill, the two-card chunk pipeline
if(B70_PREFILL_ENABLED)
  list(APPEND B70_PREFILL_HOST_SRC backend_sycl.cc)
else()
  list(APPEND B70_PREFILL_HOST_SRC backend_sycl_absent.cc)
endif()
add_library(b70_prefill_host STATIC ${B70_PREFILL_HOST_SRC})
target_include_directories(b70_prefill_host PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_prefill_host PUBLIC b70_l0 b70_model b70_loader b70_runtime PkgConfig::ZE)
b70_target_kernel_dir(b70_prefill_host)
