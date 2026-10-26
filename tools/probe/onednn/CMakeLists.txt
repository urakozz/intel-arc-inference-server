cmake_minimum_required(VERSION 3.22)
# The stand-alone icpx build of P-C, oneDNN's bf16_int4 W4A16 matmul (probe_gemm_onednn_int4.cc,
# docs/probe-prefill-vllm-parity-2026-09-14.md §A3.3) - a REFERENCE TARGET, not part of the engine.
# The engine build has no SYCL / icpx (operator, 2026-10-27); this project is configured by hand,
# after an engine build, and is in neither tools/mac_check.sh nor the box rows:
#
#   cmake -S tools/probe/onednn -B build-onednn \
#     -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx -DB70_BUILD_DIR=$PWD/build
#   cmake --build build-onednn && build-onednn/probe_gemm_onednn_int4 [--ckpt=<snapshot>]
#
# The probe makes its own in-order sycl::queue; it reads the checkpoint through the engine build's
# g++ loader / model / l0 archives (B70_BUILD_DIR).
project(b70_probe_onednn LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_options(-Wall -Wextra -Werror)

set(B70_BUILD_DIR "" CACHE PATH "an engine build tree (its src/loader, src/model and src/l0 archives)")
set(B70_ONEDNN_ROOT "/opt/intel/oneapi/dnnl/latest" CACHE PATH "oneDNN install (docs/10-the-box.md)")
get_filename_component(B70_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
foreach(lib src/loader/libb70_loader.a src/model/libb70_model.a src/l0/libb70_l0.a)
  if(NOT EXISTS "${B70_BUILD_DIR}/${lib}")
    message(FATAL_ERROR "B70_BUILD_DIR=${B70_BUILD_DIR} has no ${lib}: build the engine first")
  endif()
endforeach()
if(NOT EXISTS "${B70_ONEDNN_ROOT}/include/oneapi/dnnl/dnnl.hpp")
  message(FATAL_ERROR "B70_ONEDNN_ROOT=${B70_ONEDNN_ROOT} has no oneDNN headers")
endif()

# The SYCL runtime's own directories, from the compiler's install: libsycl, and the Unified Memory
# Framework its Level Zero adapter loads. Neither is on a non-interactive ssh LD_LIBRARY_PATH, so they
# go in as DT_RPATH (--disable-new-dtags), which the adapter's second-level dlopen also sees.
get_filename_component(_icpx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_oneapi_root "${_icpx_bin}" DIRECTORY)
get_filename_component(_oneapi_base "${_oneapi_root}/../.." ABSOLUTE)
set(_oneapi_lib "${_oneapi_root}/lib")
set(_umf_lib "${_oneapi_base}/umf/latest/lib")

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZE REQUIRED IMPORTED_TARGET level-zero)
find_package(Threads REQUIRED)

add_executable(probe_gemm_onednn_int4 ${B70_ROOT}/tools/probe/probe_gemm_onednn_int4.cc)
target_include_directories(probe_gemm_onednn_int4 PRIVATE ${B70_ROOT}/src)
target_include_directories(probe_gemm_onednn_int4 SYSTEM PRIVATE "${B70_ONEDNN_ROOT}/include")
target_compile_options(probe_gemm_onednn_int4 PRIVATE -fsycl)
target_link_options(probe_gemm_onednn_int4 PRIVATE -fsycl
  "-Wl,-rpath,${B70_ONEDNN_ROOT}/lib" "-Wl,-rpath,${_oneapi_lib}" "-Wl,-rpath,${_umf_lib}" -Wl,--disable-new-dtags)
target_link_libraries(probe_gemm_onednn_int4 PRIVATE
  -Wl,--start-group "${B70_BUILD_DIR}/src/loader/libb70_loader.a" "${B70_BUILD_DIR}/src/model/libb70_model.a"
  "${B70_BUILD_DIR}/src/l0/libb70_l0.a" -Wl,--end-group
  "${B70_ONEDNN_ROOT}/lib/libdnnl.so" PkgConfig::ZE Threads::Threads)
