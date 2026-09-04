# The OPTIONAL SYCL prefill component (spec 2 §3.2, §3.6).
#
# It adds NOTHING to the decode build: no compile-flag change, no new
# dependency on any existing target, no change to the 774-launch list. Its one
# artifact is libb70_prefill.so, built by a SEPARATE CMake sub-build whose CXX
# compiler is icpx (src/sycl/CMakeLists.txt). Two reasons it must be that shape:
#
#   * `-fsycl` device code is embedded in the object file as a fat binary and
#     is only turned into a REGISTERED device image by the clang driver's LINK
#     step (sycl-post-link + clang-offload-wrapper). A g++ link of icpx objects
#     drops the device images silently -- the binary builds and then cannot
#     find its kernels at run time. So the SYCL objects must be LINKED by
#     icpx: a shared library, linked by icpx, whose static initialisers
#     register its images when a g++-linked binary loads it.
#   * CMake supports exactly one CXX compiler per project, so the icpx half
#     is a sub-project driven by ExternalProject_Add.
#
# The .so links NO project static library -- only libze_loader and the SYCL
# runtime -- so nothing in the decode build has to become -fPIC. The SYCL
# PROBES are executables and may link the g++-built b70_l0 archive directly,
# which needs no PIC either.
set(B70_ICPX "/opt/intel/oneapi/compiler/2026.1/bin/icpx"
    CACHE FILEPATH "oneAPI C++ compiler for the prefill component (docs/10-the-box.md:42)")
set(B70_PREFILL "AUTO" CACHE STRING "SYCL prefill component: AUTO, ON or OFF")
set_property(CACHE B70_PREFILL PROPERTY STRINGS AUTO ON OFF)
option(B70_SYCL_AOT_256_GRF
       "Pass -cl-intel-256-GRF-per-thread to the SYCL AOT backend" OFF)
set(B70_SYCL_TLA_SRC_DIR "$ENV{HOME}/sycl-tla"
    CACHE PATH "sycl-tla checkout (read-only); controller-rsynced on the box")
# 91e5bd7 includes 87f68506, "FIX BMG GEMM Performance regression (#846)".
# The previous plan pin, 2db1b7c, is one commit before that BMG GEMM performance
# fix and is therefore not a valid prefill baseline. The box copy is rsynced
# without .git; src/sycl/CMakeLists.txt verifies its pinned content sentinel.
set(B70_SYCL_TLA_REVISION "91e5bd735517d8e79591b41e0d0cd37a7bacdca7"
    CACHE STRING "sycl-tla pin; a mismatched content sentinel is a WARNING recorded in the probe report")

set(B70_PREFILL_ENABLED OFF)
if(NOT B70_PREFILL STREQUAL "OFF")
  if(EXISTS "${B70_ICPX}")
    execute_process(COMMAND "${B70_ICPX}" --version
                    RESULT_VARIABLE _icpx_rc
                    OUTPUT_VARIABLE _icpx_out ERROR_VARIABLE _icpx_out)
  else()
    set(_icpx_rc 1)
    set(_icpx_out "no such file: ${B70_ICPX}")
  endif()
  if(_icpx_rc EQUAL 0)
    set(B70_PREFILL_ENABLED ON)
    string(REGEX REPLACE "\n.*" "" _icpx_line "${_icpx_out}")
    message(STATUS "b70: prefill component ON  -- ${_icpx_line}")
  elseif(B70_PREFILL STREQUAL "ON")
    message(FATAL_ERROR "B70_PREFILL=ON but icpx is unusable:\n${_icpx_out}")
  else()
    message(STATUS "b70: prefill component OFF -- icpx unusable (${_icpx_out})")
  endif()
endif()

if(B70_PREFILL_ENABLED)
  include(ExternalProject)
  get_filename_component(_icpx_bin "${B70_ICPX}" DIRECTORY)
  get_filename_component(_oneapi_root "${_icpx_bin}" DIRECTORY)
  get_filename_component(_oneapi_base "${_oneapi_root}" DIRECTORY)
  get_filename_component(_oneapi_base "${_oneapi_base}" DIRECTORY)
  set(B70_ONEAPI_LIB "${_oneapi_root}/lib" CACHE INTERNAL "SYCL runtime lib dir")
  set(B70_ONEAPI_UMF_LIB "${_oneapi_base}/umf/1.1/lib" CACHE INTERNAL
      "oneAPI Unified Memory Framework runtime lib dir")
  set(B70_PREFILL_INSTALL "${CMAKE_BINARY_DIR}/sycl-install")
  set(B70_PREFILL_LIB "${B70_PREFILL_INSTALL}/lib/libb70_prefill.so")
  ExternalProject_Add(b70_prefill_ext
    SOURCE_DIR  "${CMAKE_SOURCE_DIR}/src/sycl"
    BINARY_DIR  "${CMAKE_BINARY_DIR}/sycl-build"
    INSTALL_DIR "${B70_PREFILL_INSTALL}"
    CMAKE_ARGS
      -DCMAKE_CXX_COMPILER=${B70_ICPX}
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DB70_ROOT=${CMAKE_SOURCE_DIR}
      -DB70_KERNEL_DIR=${B70_KERNEL_DIR}
      -DB70_OCLOC_DEVICE=${B70_OCLOC_DEVICE}
      -DB70_ONEAPI_LIB=${B70_ONEAPI_LIB}
      -DB70_ONEAPI_UMF_LIB=${B70_ONEAPI_UMF_LIB}
      -DB70_L0_LIB=$<TARGET_FILE:b70_l0>
      -DB70_SYCL_TLA_SRC_DIR=${B70_SYCL_TLA_SRC_DIR}
      -DB70_SYCL_TLA_REVISION=${B70_SYCL_TLA_REVISION}
      -DB70_SYCL_AOT_256_GRF=${B70_SYCL_AOT_256_GRF}
    BUILD_ALWAYS 1
    BUILD_BYPRODUCTS "${B70_PREFILL_LIB}")
  add_library(b70_prefill INTERFACE)
  target_link_libraries(b70_prefill INTERFACE "${B70_PREFILL_LIB}")
  target_include_directories(b70_prefill INTERFACE "${CMAKE_SOURCE_DIR}/src")
endif()

# Link a g++-built target against the prefill component. Also orders the
# sub-build ahead of it and records the two run-time library directories:
# ours, and the oneAPI SYCL runtime -- which is NOT on a non-interactive ssh
# LD_LIBRARY_PATH (docs/10-the-box.md:42), so without the rpath every probe
# and test dies at exec with "libsycl.so.8: cannot open shared object file".
function(b70_link_prefill TARGET)
  if(NOT B70_PREFILL_ENABLED)
    message(FATAL_ERROR "b70_link_prefill(${TARGET}) but the component is off")
  endif()
  target_link_libraries(${TARGET} PRIVATE b70_prefill)
  add_dependencies(${TARGET} b70_prefill_ext)
  set_property(TARGET ${TARGET} APPEND PROPERTY
               BUILD_RPATH "${B70_PREFILL_INSTALL}/lib" "${B70_ONEAPI_LIB}" "${B70_ONEAPI_UMF_LIB}")
  # libsycl loads the Level Zero adapter dynamically, and that adapter needs
  # libumf. DT_RPATH (rather than non-transitive DT_RUNPATH) makes this path
  # available to that second-level dependency in non-interactive ssh sessions.
  target_link_options(${TARGET} PRIVATE -Wl,--disable-new-dtags)
endfunction()
