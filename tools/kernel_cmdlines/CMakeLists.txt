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
get_filename_component(_kc_tree "${TREE}" ABSOLUTE)
function(add_ocloc_kernel NAME)
  cmake_parse_arguments(K "" "SOURCE" "DEFINES;DEPENDS;OPTIONS" ${ARGN})
  get_filename_component(src "${K_SOURCE}" NAME)
  string(REPLACE ";" " " d "${K_DEFINES}")
  string(REPLACE ";" " " o "${K_OPTIONS}")
  # An option naming a path inside the tree (24c's -I<src/kernels/lfm2moe>) is the same command line in every
  # checkout: spell the tree's root as <tree>, so two trees compare by what they build, not where they sit.
  string(REPLACE "${_kc_tree}" "<tree>" o "${o}")
  file(APPEND ${CMAKE_BINARY_DIR}/kernels.txt "${NAME} | ${src} | ${d} | ${o}\n")
endfunction()
file(WRITE ${CMAKE_BINARY_DIR}/kernels.txt "")
# Spec 13a: the batching probe's variants are OFF in a default build (B70_BATCH_PROBE) but listed here, so the Mac's
# section 4 syntax-checks their command lines and section 3 shows them as additions. A tree without the option
# ignores the cache entry.
set(B70_BATCH_PROBE ON CACHE BOOL "list the batching probe's kernel variants (spec 13a)")
add_subdirectory(${TREE}/src/kernels kernels)
