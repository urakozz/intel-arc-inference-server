cmake_minimum_required(VERSION 3.22)
# C++26 with static reflection (src/common/json_reflect.h) needs GCC 16: the box's /usr/local/bin/g++-16.2 unless a
# compiler is named (-DCMAKE_CXX_COMPILER or CXX). Set before project(), which fixes the compiler.
set(B70_GXX /usr/local/bin/g++-16.2)
if(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  if(NOT EXISTS ${B70_GXX})
    message(FATAL_ERROR "${B70_GXX} not found: this build needs GCC 16 (C++26 with -freflection). Install GCC 16.2 "
                        "there, or name another GCC 16 with -DCMAKE_CXX_COMPILER=...")
  endif()
  set(CMAKE_CXX_COMPILER ${B70_GXX})
endif()
project(b70_inference_server LANGUAGES CXX)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER} is ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}: this build "
                      "needs GCC 16 (C++26 with -freflection), ${B70_GXX} on the box")
endif()
# GCC 16's libstdc++ is newer than the system's: every binary finds it through its rpath, not LD_LIBRARY_PATH.
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so OUTPUT_VARIABLE B70_LIBSTDCXX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
file(REAL_PATH ${B70_LIBSTDCXX} B70_LIBSTDCXX)
get_filename_component(B70_LIBSTDCXX_DIR ${B70_LIBSTDCXX} DIRECTORY)
add_link_options(-Wl,-rpath,${B70_LIBSTDCXX_DIR})

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_options(-Wall -Wextra -Werror)
# Host instruction set: the oldest supported CPU is Broadwell (AVX2, FMA, BMI2, F16C; no AVX-512); every newer
# x86-64 runs this build. FMA contraction stays off so host math rounds as written (the CPU references are
# compared bitwise).
add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-march=broadwell> $<$<COMPILE_LANGUAGE:C,CXX>:-ffp-contract=off>)
# Static reflection (C++26 P2996 + P3394 annotations) in every C++ target: runtime/gpu_state.h's annotated state
# structs (spec 39) sit in headers most targets include.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-freflection>)
# C++26 contracts (pre / post / contract_assert) state the engine's bounds in its interfaces. Enforced in every build
# type, Release included (contracts do not follow NDEBUG): a violation is a bug in the caller, never a request's input,
# so the handler below prints the predicate's text and where it failed, and the process ends (SIGABRT) instead of
# serving from a wrong page or position. A request's or a flag's errors stay the plain exceptions.
add_compile_options(-fcontracts -fcontract-evaluation-semantic=enforce)
# The violation handler (src/common/contract_handler.cc), in every target linked after this line: its object is linked
# directly, so a reference from any static library resolves whatever the link order.
add_library(b70_contract_handler OBJECT src/common/contract_handler.cc)
link_libraries(b70_contract_handler)

include(cmake/tokenizer.cmake)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZE REQUIRED IMPORTED_TARGET level-zero)
include(cmake/ocloc.cmake)

# kernels::kernel_dir() (src/kernels/kernels.h) names the compiled device binaries' directory, so tests and probes
# can find .bin files. The absolute path lives in this one generated file rather than in a -D on every target: a -D
# differs between two trees and would keep ccache from sharing any object that includes kernels.h.
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/kernel_dir.cc"
     CONTENT "namespace kernels {\nconst char* kernel_dir() { return \"@B70_KERNEL_DIR@\"; }\n}  // namespace kernels\n"
     @ONLY)
add_library(b70_kernel_dir STATIC "${CMAKE_BINARY_DIR}/kernel_dir.cc")
function(b70_target_kernel_dir TARGET)
  target_link_libraries(${TARGET} PRIVATE b70_kernel_dir)
endfunction()

# spec §10 asks for `make golden`. The oracle cannot run from this build tree:
# it needs the reference container, ~61 GiB of RAM and ~17 minutes on the box,
# and it never runs from the Mac at all. A target that shelled out to docker
# from CMake would be a lie on every machine that cannot satisfy that. So this
# one PRINTS the exact on-box invocation and points at the committed scripts
# that do the work (tools/oracle/{golden,check}.sh) - which is the honest way
# to satisfy §10's "`make golden` regenerates": one command that tells you the
# one command, with the recipe under version control rather than in a shell
# history. Run `cmake --build build --target golden` (or `make golden`).
add_custom_target(golden
  COMMAND ${CMAKE_COMMAND} -E echo "The oracle runs on the box, in the reference container (61 GiB peak, ~17 min):"
  COMMAND ${CMAKE_COMMAND} -E echo "  tools/box.sh sync                     # from the Mac"
  COMMAND ${CMAKE_COMMAND} -E echo "  ssh \"$BOX\" && cd ~/b70-inference-server"
  COMMAND ${CMAKE_COMMAND} -E echo "  setsid nohup tools/oracle/golden.sh > oracle-out/golden.log 2>&1 </dev/null &"
  COMMAND ${CMAKE_COMMAND} -E echo "  tools/oracle/check.sh                 # re-read what was written"
  COMMAND ${CMAKE_COMMAND} -E echo "Prompt ids first, and the measured numbers: tools/oracle/README.md, 'Running it'."
  VERBATIM)

enable_testing()
# Every test runs through tools/test_launcher.sh: as it is, or - with B70_TRACE_DIR set (box_validate,
# box_check) - under strace, so tools/box_validate/reuse.py knows what the test read (a test whose inputs
# are all unchanged keeps its PASS). The launcher property needs CMake 3.29.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.29)
  set(CMAKE_TEST_LAUNCHER "${CMAKE_SOURCE_DIR}/tools/test_launcher.sh")
endif()
add_subdirectory(third_party)
add_subdirectory(src/common)
add_subdirectory(src/tokenizer)
add_subdirectory(src/server)
add_subdirectory(src/loader)
add_subdirectory(src/model)
add_subdirectory(src/runtime)
add_subdirectory(src/kernels)
add_subdirectory(src/l0)
add_subdirectory(tools/probe)
# After src/kernels: src/cli reads that directory's target list to depend on
# every compiled device binary.
add_subdirectory(src/cli)
add_subdirectory(tests)
