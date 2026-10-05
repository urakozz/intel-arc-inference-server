cmake_minimum_required(VERSION 3.22)
# The icpx half of the build. Configured ONLY by cmake/prefill.cmake through
# ExternalProject_Add, never by the top-level project: it sets
# CMAKE_CXX_COMPILER to icpx, which one CMake project cannot do twice.
#
# C++20, not the decode tree's C++17: Intel's CuTe chunked-GDN kernel (P5) is
# C++20 (vllm-xpu-kernels/cmake/utils.cmake:580, CMakeLists.txt:498). sycl-tla
# itself asks only for C++17 (sycl-tla/CMakeLists.txt:233-237) and compiles at
# 20. Nothing the decode build compiles is in this project; the one header
# that crosses the boundary, src/runtime/prefill/context.h, is plain C++17 and
# SYCL-free.
#
# 3.22 is sycl-tla's own floor when SYCL is on (sycl-tla/CMakeLists.txt:29-30).
project(b70_prefill_sycl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra -Werror)

foreach(v B70_ROOT B70_KERNEL_DIR B70_OCLOC_DEVICE B70_ONEAPI_LIB B70_ONEAPI_UMF_LIB B70_L0_LIB
          B70_LOADER_LIB B70_MODEL_LIB B70_PREFILL_HOST_LIB B70_RUNTIME_LIB B70_PLAN_LIB B70_SYCL_TLA_SRC_DIR
          B70_SYCL_TLA_REVISION)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "${v} must be passed in by cmake/prefill.cmake")
  endif()
endforeach()

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZE REQUIRED IMPORTED_TARGET level-zero)

# AOT for the B70. These are sycl-tla's own flags for this part, read out of
# sycl-tla/cmake/FindDPCPP.cmake and spelled with the -Xs shorthand:
#   compile: -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen
#   link:    + -Xsycl-target-backend=spir64_gen "-device bmg-g31"
#            + -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,
#              +SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
# `-Xs` is the driver's shorthand for -Xsycl-target-backend. The two SPIR-V
# extensions beyond split_barrier are what the 2D block loads and the DPAS
# matrix-multiply-accumulate lower through; without them the GEMM and FMHA
# kernels fail to translate. bmg-g31 is the B70 (cmake/ocloc.cmake:4).
set(B70_SYCL_COMPILE -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen)
set(B70_SYCL_LINK ${B70_SYCL_COMPILE}
    -Xs "-device ${B70_OCLOC_DEVICE}"
    -Xspirv-translator
    -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate)
if(B70_SYCL_AOT_256_GRF)
  list(APPEND B70_SYCL_LINK
       -Xsycl-target-backend=spir64_gen "-options -cl-intel-256-GRF-per-thread")
endif()

# --- sycl-tla, pinned, HEADER-ONLY ----------------------------------------
# We deliberately do NOT add_subdirectory() it. Its README says the library
# "is header-only ... and does not need to be built to be used by other
# projects" (sycl-tla/README.md:179-181), and its top-level CMakeLists has a
# long list of effects an external consumer does not want: it FORCES
# CMAKE_INSTALL_PREFIX to `install` (:246-248) and CMAKE_BUILD_TYPE to Release
# (:376-380), sets CMAKE_POSITION_INDEPENDENT_CODE ON (:382), forces
# CMAKE_CXX_STANDARD 17 (:235-237), requires a Python3 interpreter (:255),
# FetchContents google/benchmark unless CUTLASS_ENABLE_BENCHMARKS is turned
# off explicitly -- it defaults ON unconditionally (:293, :1320-1323) -- and,
# through tools/util, ExternalProjects oneMKL when MKLROOT is unset
# (cmake/onemkl.cmake:38,54-77). We need include paths and two -D's.
#
# NOTE: `applications/` (the FMHA tree) and `benchmarks/` are on NO exported
# target and are NOT installed -- sycl-tla adds them ad hoc per example
# (examples/06_bmg_flash_attention/CMakeLists.txt:31). They are include roots
# here for exactly that reason.
if(NOT EXISTS "${B70_SYCL_TLA_SRC_DIR}/include/cute/tensor.hpp")
  message(FATAL_ERROR
    "sycl-tla not found at ${B70_SYCL_TLA_SRC_DIR}. It is a controller-provisioned, "
    "read-only rsync checkout; do not clone it from this build.")
endif()
# The controller-rsynced box copy intentionally excludes .git. Verify a file
# changed by the pinned commit instead of invoking git: the local checkout at
# 91e5bd7 has this SHA-256 for launch_policy.hpp.
set(_tla_sentinel "${B70_SYCL_TLA_SRC_DIR}/include/cute/util/compat/launch_policy.hpp")
set(_tla_sentinel_expected "5460b1e5119308896ab65ca99127179c2b47c2ca8173db4349b938d680fea661")
if(NOT EXISTS "${_tla_sentinel}")
  set(B70_TLA_SHA "unknown-missing-content-sentinel")
  message(WARNING "b70 prefill: sycl-tla content sentinel is absent; pin is "
                  "${B70_SYCL_TLA_REVISION} -- record this deviation in the probe report")
else()
  file(SHA256 "${_tla_sentinel}" _tla_sentinel_actual)
  if(_tla_sentinel_actual STREQUAL _tla_sentinel_expected)
    set(B70_TLA_SHA "${B70_SYCL_TLA_REVISION}-content-verified")
  else()
    set(B70_TLA_SHA "unknown-content-sentinel-${_tla_sentinel_actual}")
    message(WARNING "b70 prefill: sycl-tla content sentinel is ${_tla_sentinel_actual}, "
                    "expected ${_tla_sentinel_expected} for pin ${B70_SYCL_TLA_REVISION} "
                    "-- record this deviation in the probe report")
  endif()
endif()
# Stamped into a generated header so no probe can report a number without
# naming the revision that produced it.
configure_file(tla_pin.h.in "${CMAKE_CURRENT_BINARY_DIR}/tla_pin.h" @ONLY)

add_library(sycl_tla INTERFACE)
target_include_directories(sycl_tla SYSTEM INTERFACE
  ${B70_SYCL_TLA_SRC_DIR}/include
  ${B70_SYCL_TLA_SRC_DIR}/tools/util/include
  ${B70_SYCL_TLA_SRC_DIR}/applications
  ${B70_SYCL_TLA_SRC_DIR}/benchmarks
  ${B70_SYCL_TLA_SRC_DIR}/examples/common)
# The two definitions sycl-tla's own build adds for an Intel SYCL target
# (sycl-tla/CMakeLists.txt:214 and :187). A bare -D gives the macro the value
# 1, which is what the FMHA tree's `#if (SYCL_INTEL_TARGET == 35)` Xe3p branch
# tests against -- so 1 correctly selects the Xe2/BMG branch.
#
# We do NOT define CUTLASS_VERSIONS_GENERATED: it guards the include of the
# GENERATED header cutlass/version_extended.h (sycl-tla/CMakeLists.txt:884-887,
# :398), which only sycl-tla's own CMake produces and this build does not run.
# vllm-xpu-kernels does define it (CMakeLists.txt:389) along with
# CUTLASS_ENABLE_HEADERS_ONLY (:386, a CMake option, meaningless as a macro);
# if a build error ever names version_extended.h, that difference is the first
# thing to check.
target_compile_definitions(sycl_tla INTERFACE CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
# sycl-tla does NOT add this for SYCL builds -- :244 puts it in the CUDA-only
# flag set -- and its own docs tell you to pass it by hand
# (media/docs/cpp/build/building_with_sycl_support.md:41). Without it a
# template error in the mainloop prints thousands of lines.
target_compile_options(sycl_tla INTERFACE -ftemplate-backtrace-limit=0)

# --- the component ---------------------------------------------------------
add_library(b70_prefill SHARED sycl_side.cc gemm_batched.cc)
target_include_directories(b70_prefill PRIVATE ${B70_ROOT}/src)
target_compile_options(b70_prefill PRIVATE ${B70_SYCL_COMPILE})
target_link_options(b70_prefill PRIVATE ${B70_SYCL_LINK}
  -Wl,-rpath,${B70_ONEAPI_LIB} -Wl,-rpath,${B70_ONEAPI_UMF_LIB}
  -Wl,--disable-new-dtags -Wl,--no-as-needed)
# libsycl discovers its Level Zero adapter with dlopen; make libumf a direct
# dependency so that adapter can resolve it outside an interactive setvars.sh
# environment. This remains a oneAPI runtime dependency, not a project archive.
target_link_libraries(b70_prefill PRIVATE PkgConfig::ZE "${B70_ONEAPI_UMF_LIB}/libumf.so"
  sycl_tla)
set_property(TARGET b70_prefill APPEND PROPERTY
  INSTALL_RPATH "${B70_ONEAPI_LIB}" "${B70_ONEAPI_UMF_LIB}")
install(TARGETS b70_prefill LIBRARY DESTINATION lib)

# Compile-only proof that the pin resolves and its headers parse under our
# flags. It defines one symbol nobody calls; it exists so that a broken pin
# fails here in seconds rather than inside P2's mainloop instantiation.
add_library(b70_tla_smoke STATIC tla_smoke.cc)
target_include_directories(b70_tla_smoke PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(b70_tla_smoke PRIVATE ${B70_SYCL_COMPILE})
target_link_libraries(b70_tla_smoke PRIVATE sycl_tla)

# --- the SYCL probes -------------------------------------------------------
# Their sources live in tools/probe/ with every other probe; only their BUILD
# is here, because they contain SYCL kernels and so must be compiled AND
# linked by icpx (cmake/prefill.cmake states why). Each probe gets the test
# helpers' include dirs too, so gemv_harness.h / attn_ref.h / gdn_ref.h are
# reachable -- the probes reuse those references rather than restating them.
function(b70_sycl_probe NAME)
  add_executable(${NAME} ${B70_ROOT}/tools/probe/${NAME}.cc)
  target_include_directories(${NAME} PRIVATE
    ${B70_ROOT}/src ${B70_ROOT}/tests ${B70_ROOT}/tests/kernels
    ${B70_ROOT}/tools/probe ${CMAKE_CURRENT_BINARY_DIR})
  target_compile_definitions(${NAME} PRIVATE B70_KERNEL_DIR="${B70_KERNEL_DIR}")
  target_compile_options(${NAME} PRIVATE ${B70_SYCL_COMPILE})
  target_link_options(${NAME} PRIVATE ${B70_SYCL_LINK}
    -Wl,-rpath,${B70_ONEAPI_LIB} -Wl,-rpath,${B70_ONEAPI_UMF_LIB} -Wl,--disable-new-dtags)
  # The probes construct `Context`, which is now in the host archive, and the host archive
  # pulls the runtime (and its buffer sizes, b70_plan: spec 6 §10), loader and model
  # archives (spec 2.1 §3.5).
  target_link_libraries(${NAME} PRIVATE b70_prefill -Wl,--start-group "${B70_PREFILL_HOST_LIB}"
    "${B70_RUNTIME_LIB}" "${B70_PLAN_LIB}" "${B70_LOADER_LIB}" "${B70_MODEL_LIB}" "${B70_L0_LIB}" -Wl,--end-group
    PkgConfig::ZE ${ARGN})
  set_property(TARGET ${NAME} APPEND PROPERTY INSTALL_RPATH
    "$ORIGIN/../lib" "${B70_ONEAPI_LIB}" "${B70_ONEAPI_UMF_LIB}")
  install(TARGETS ${NAME} RUNTIME DESTINATION bin)
endfunction()
b70_sycl_probe(probe_interop)
b70_sycl_probe(probe_prefill_gemm sycl_tla)
b70_sycl_probe(probe_prefill_attn sycl_tla)
# Ruling A21's probe: int4-g64 through MainloopIntelXeXMX16MixedPrecision. It
# is the only SYCL probe that reads the checkpoint, so it is the only one that
# links the loader (and, through it, the model table). b70_sycl_probe puts
# B70_L0_LIB BEFORE anything in ARGN, and b70_loader references b70_l0 and
# b70_model, so the three archives go in an explicit --start-group rather than
# relying on a single-pass order that would leave loader's references dangling.
b70_sycl_probe(probe_gemm_int4_mixed sycl_tla)
target_link_libraries(probe_gemm_int4_mixed PRIVATE
  -Wl,--start-group "${B70_LOADER_LIB}" "${B70_MODEL_LIB}" "${B70_L0_LIB}" -Wl,--end-group)

# P-C (spec 2 addendum, docs/probe-prefill-vllm-parity-2026-09-14.md §A1/§A3.3):
# oneDNN's bf16_int4 W4A16 matmul, the exact primitive vLLM's own prefill GEMM
# path uses. REFERENCE TARGET, not a candidate (A0) -- this probe calls
# oneDNN's plain C++ primitive API directly (no vllm-xpu-kernels machinery),
# so it needs no sycl_tla, only the loader (like probe_gemm_int4_mixed above,
# for the one real correctness matrix) and oneDNN itself. oneDNN 2026.0 is
# installed on the box (docs/10-the-box.md); not part of this project's pin
# set, so its location is a plain CACHE PATH, not a content-verified checkout.
set(B70_ONEDNN_ROOT "/opt/intel/oneapi/dnnl/2026.0" CACHE PATH
    "oneDNN install used by P-C (reference target, not a project dependency)")
if(NOT EXISTS "${B70_ONEDNN_ROOT}/include/oneapi/dnnl/dnnl.hpp")
  message(FATAL_ERROR "B70_ONEDNN_ROOT=${B70_ONEDNN_ROOT} has no oneDNN headers")
endif()
b70_sycl_probe(probe_gemm_onednn_int4)
target_link_libraries(probe_gemm_onednn_int4 PRIVATE
  -Wl,--start-group "${B70_LOADER_LIB}" "${B70_MODEL_LIB}" "${B70_L0_LIB}" -Wl,--end-group
  "${B70_ONEDNN_ROOT}/lib/libdnnl.so")
target_include_directories(probe_gemm_onednn_int4 SYSTEM PRIVATE "${B70_ONEDNN_ROOT}/include")
set_property(TARGET probe_gemm_onednn_int4 APPEND PROPERTY
             INSTALL_RPATH "${B70_ONEDNN_ROOT}/lib")
target_link_options(probe_gemm_onednn_int4 PRIVATE "-Wl,-rpath,${B70_ONEDNN_ROOT}/lib")

# P5: Intel's CuTe chunked GDN header is consumed from the controller-provided,
# read-only vllm-xpu-kernels checkout. Its BMG path is C++20 and emits a large
# device image, hence the matching compiler definition and link option.
set(B70_VLLM_XPU_KERNELS_DIR "$ENV{HOME}/vllm-xpu-kernels"
    CACHE PATH "read-only checkout supplying the CuTe chunked-GDN kernel")
option(B70_P5_CUTE_GDN "Build the optional CuTe GDN probe (requires PyTorch headers)" OFF)
if(B70_P5_CUTE_GDN)
  b70_sycl_probe(probe_gdn_chunk sycl_tla)
  target_include_directories(probe_gdn_chunk SYSTEM PRIVATE
    ${B70_ROOT}/src/sycl/vllm_shim
    ${B70_VLLM_XPU_KERNELS_DIR}
    ${B70_VLLM_XPU_KERNELS_DIR}/csrc/xpu/gdn_attn
    ${B70_VLLM_XPU_KERNELS_DIR}/csrc/xpu/gdn_attn/xe_2)
  target_compile_definitions(probe_gdn_chunk PRIVATE VLLM_XPU_ENABLE_XE2)
  target_link_options(probe_gdn_chunk PRIVATE -flink-huge-device-code)
endif()

# P-D (docs/probe-prefill-vllm-parity-2026-09-14.md §A3.4): the torch-free
# shim Phase 1 §2.1 characterised as small. Unlike P5 above (gated OFF,
# "requires PyTorch headers"), this target calls `gdn::kernel_launcher<T,
# StateT>` directly -- the torch-free template -- and never executes the
# vendor header's torch-using OUTER wrapper, so it needs no real PyTorch: only
# `src/sycl/vllm_shim/torch/all.h`'s stand-in, just complete enough for that
# unused wrapper to typecheck (see the probe's own file header for the exact
# enumeration). Always built when the read-only vllm-xpu-kernels checkout is
# present, independent of B70_P5_CUTE_GDN.
if(EXISTS "${B70_VLLM_XPU_KERNELS_DIR}/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp")
  b70_sycl_probe(probe_pf_gdn_xe2 sycl_tla)
  target_include_directories(probe_pf_gdn_xe2 SYSTEM PRIVATE
    ${B70_ROOT}/src/sycl/vllm_shim
    ${B70_VLLM_XPU_KERNELS_DIR}
    ${B70_VLLM_XPU_KERNELS_DIR}/csrc/xpu/gdn_attn
    ${B70_VLLM_XPU_KERNELS_DIR}/csrc/xpu/gdn_attn/xe_2)
  target_compile_definitions(probe_pf_gdn_xe2 PRIVATE VLLM_XPU_ENABLE_XE2)
  target_link_options(probe_pf_gdn_xe2 PRIVATE -flink-huge-device-code)
endif()
