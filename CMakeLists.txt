cmake_minimum_required(VERSION 3.22)
project(b70_inference_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_options(-Wall -Wextra -Werror)

find_package(PkgConfig REQUIRED)
include(cmake/tokenizer.cmake)
pkg_check_modules(ZE REQUIRED IMPORTED_TARGET level-zero)

include(cmake/ocloc.cmake)
include(cmake/prefill.cmake)

# Adds -DB70_KERNEL_DIR="<abs path>" so tests and probes can find .bin files.
function(b70_target_kernel_dir TARGET)
  target_compile_definitions(${TARGET} PRIVATE B70_KERNEL_DIR="${B70_KERNEL_DIR}")
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
  COMMAND ${CMAKE_COMMAND} -E echo "  ssh user@box && cd ~/b70-inference-server"
  COMMAND ${CMAKE_COMMAND} -E echo "  setsid nohup tools/oracle/golden.sh > oracle-out/golden.log 2>&1 </dev/null &"
  COMMAND ${CMAKE_COMMAND} -E echo "  tools/oracle/check.sh                 # re-read what was written"
  COMMAND ${CMAKE_COMMAND} -E echo "Prompt ids first, and the measured numbers: tools/oracle/README.md, 'Running it'."
  VERBATIM)

enable_testing()
add_subdirectory(third_party)
add_subdirectory(src/common)
add_subdirectory(src/tokenizer)
add_subdirectory(src/server)
add_subdirectory(src/loader)
add_subdirectory(src/model)
add_subdirectory(src/runtime)
add_subdirectory(src/kernels)
add_subdirectory(src/l0)
# cmake/prefill.cmake is included before subdirectories so tests can consume
# its helper, while b70_l0 is declared here. Add the ExternalProject dependency
# only once both targets exist; ExternalProject_Add itself rejects a forward
# target reference during configure.
if(TARGET b70_prefill_ext)
  # P1 consumes these existing OpenCL binaries through kernels::path(); make
  # their generation precede the separate icpx build as well as b70_l0.
  # b70_loader/b70_model join the list for the int4-mixed probe (ruling A21),
  # the one icpx probe that reads the checkpoint. b70_prefill_host/b70_runtime
  # join it because the ExternalProject must not start before the archives its
  # probes link exist (spec 2.1 §3.5).
  add_dependencies(b70_prefill_ext b70_l0 b70_loader b70_model b70_prefill_host b70_runtime
                   kernel_noop kernel_ctrl_read kernel_bw_sum)
endif()
add_subdirectory(tools/probe)
# After src/kernels: src/cli reads that directory's target list to depend on
# every compiled device binary.
add_subdirectory(src/cli)
add_subdirectory(tests)
