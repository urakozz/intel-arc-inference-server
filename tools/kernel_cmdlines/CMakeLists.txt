# Lists every device binary src/kernels declares - name | source | -D defines | options -
# without ocloc, Level Zero or a GPU: add_ocloc_kernel is stubbed to append one line to
# <build>/kernels.txt. Two trees' lists, sorted and diffed, show whether a change moved
# any existing binary's command line (spec 14 G0's Mac-side pre-check; the box compares
# the binaries' checksums).
#
#   cmake -S tools/kernel_cmdlines -B /tmp/kc -DTREE=$PWD [-DB70_AGNES_SWEEP=ON] && sort /tmp/kc/kernels.txt
cmake_minimum_required(VERSION 3.22)
project(kernel_cmdlines NONE)
if(NOT TREE)
  message(FATAL_ERROR "pass -DTREE=<repo root>")
endif()
set(B70_KERNEL_DIR /kernels)
function(add_ocloc_kernel NAME)
  cmake_parse_arguments(K "" "SOURCE" "DEFINES;DEPENDS;OPTIONS" ${ARGN})
  get_filename_component(src "${K_SOURCE}" NAME)
  string(REPLACE ";" " " d "${K_DEFINES}")
  string(REPLACE ";" " " o "${K_OPTIONS}")
  file(APPEND ${CMAKE_BINARY_DIR}/kernels.txt "${NAME} | ${src} | ${d} | ${o}\n")
endfunction()
file(WRITE ${CMAKE_BINARY_DIR}/kernels.txt "")
add_subdirectory(${TREE}/src/kernels kernels)
