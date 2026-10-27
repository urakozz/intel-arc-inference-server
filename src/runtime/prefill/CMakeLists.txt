# The prefill path's host half (spec 2.1 §3.5): context.cc is the Level Zero list, and every
# prefill GEMM runs on it.
set(B70_PREFILL_HOST_SRC context.cc kernels.cc gdn.cc attn.cc step.cc
    engine_prefill.cc profile.cc gemm_l0.cc linear_l0.cc int8.cc
    moe.cc      # spec 15d: a MoE layer's FFN on the prefill path
    pipeline_prefill.cc)   # spec 16c: PipelineEngine::prefill, the two-card chunk pipeline
add_library(b70_prefill_host STATIC ${B70_PREFILL_HOST_SRC})
target_include_directories(b70_prefill_host PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_prefill_host PUBLIC b70_l0 b70_model b70_loader b70_runtime PkgConfig::ZE)
b70_target_kernel_dir(b70_prefill_host)
