# ocloc AOT rule. One call = one device binary for bmg-g31.
find_program(OCLOC_EXECUTABLE ocloc REQUIRED)
set(B70_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "directory of compiled device binaries")
set(B70_OCLOC_DEVICE "bmg-g31" CACHE STRING "ocloc -device target (B70 = Battlemage G31)")
file(MAKE_DIRECTORY "${B70_KERNEL_DIR}")
# src/kernels is on every kernel's include path: `#include "bf16.h"` (the shared bf16 <-> fp32 pair) from any source.
get_filename_component(B70_KERNEL_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../src/kernels" ABSOLUTE)
# The compiler's identity (cmake/ocloc_stamp.cmake), refreshed every build and rewritten only when it changes: every
# kernel depends on it, so a driver / IGC upgrade recompiles them all.
set(B70_OCLOC_STAMP "${B70_KERNEL_DIR}/.ocloc-toolchain")
execute_process(COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_LIST_DIR}/ocloc_stamp.cmake" "${B70_OCLOC_STAMP}" "${OCLOC_EXECUTABLE}")
# Every kernel compiles through tools/ocloc_cache.py, a content-addressed cache every source tree on the machine shares
# ($B70_DATA/cache/ocloc, else ~/.cache/b70-ocloc; B70_OCLOC_CACHE=0 in the build's environment bypasses it). It runs in the tree's root with
# tree-relative paths: the build options are kept inside each zebin, so an absolute -I would make two trees' binaries
# differ and their cache keys too. Without python3 ocloc runs directly.
get_filename_component(B70_TREE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(RELATIVE_PATH B70_KERNEL_SRC_REL "${B70_TREE_ROOT}" "${B70_KERNEL_SRC_DIR}")
find_program(B70_PYTHON3 python3)
if(B70_PYTHON3)
  set(B70_OCLOC_COMMAND "${B70_PYTHON3}" "${B70_TREE_ROOT}/tools/ocloc_cache.py" "${B70_OCLOC_STAMP}" "${OCLOC_EXECUTABLE}")
else()
  set(B70_OCLOC_COMMAND "${OCLOC_EXECUTABLE}")
endif()
add_custom_target(ocloc_toolchain ALL
  COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_LIST_DIR}/ocloc_stamp.cmake" "${B70_OCLOC_STAMP}" "${OCLOC_EXECUTABLE}"
  BYPRODUCTS "${B70_OCLOC_STAMP}"
  COMMENT "ocloc toolchain stamp"
  VERBATIM)

# add_ocloc_kernel(<name> SOURCE <file.cl> [DEFINES A=1 B=2 ...] [DEPENDS <files>]
#                  [OPTIONS -cl-... ...])
# Produces ${B70_KERNEL_DIR}/<name>.bin (ZE_MODULE_FORMAT_NATIVE). <name> is the
# variant name the runtime asks for, e.g. gemv_M1_K5120_N5120_S1_L0.
#
# **Every kernel is built with `-cl-fp32-correctly-rounded-divide-sqrt`.** It is
# not opt-in: OpenCL's default allows 2.5 ulp on `/` and `sqrt`, and this
# project's whole testing method is a host reference compared bit-exactly (or at
# a few-ulp bar) against the device. Measured 2026-08-25 while bringing up
# prep.cl, the device's `1/sqrt(mean + 1e-6)` sat 1-2 ulp under the host's and
# one x_out element in 5120 fell on the wrong side of a round-to-nearest tie
# (docs/12-kernels.md, "Rounding discipline"). Making it the default rather than
# a per-kernel OPTION removes the failure mode where a new kernel grows a divide
# and nobody remembers the flag; kernels with no divide and no sqrt (gemv,
# gemv_bf16, embed_gather, argmax) are unaffected, which the full suite proves
# every time it stays green (controller ruling 2026-08-25, plan 3 Task 4).
#
# OPTIONS appends further `-cl-` build options for one kernel. One rule:
# `-cl-denorms-are-zero` is FORBIDDEN project-wide -- the 27B's int4 scales go
# subnormal and flushing them makes the device disagree with both the host
# reference and the oracle (docs/13-loader.md, "The three data asserts, and what
# measured them").
function(add_ocloc_kernel NAME)
  cmake_parse_arguments(K "" "SOURCE" "DEFINES;DEPENDS;OPTIONS" ${ARGN})
  if(NOT K_SOURCE)
    message(FATAL_ERROR "add_ocloc_kernel(${NAME}): SOURCE is required")
  endif()
  # The project-wide options: the language version, the correctly rounded
  # divide/sqrt every host-comparable kernel needs (see above) and src/kernels
  # on the include path (bf16.h, tp_wire.h).
  set(opts "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -I${B70_KERNEL_SRC_REL}")
  foreach(d IN LISTS K_DEFINES)
    string(APPEND opts " -D${d}")
  endforeach()
  foreach(o IN LISTS K_OPTIONS)
    if(o STREQUAL "-cl-denorms-are-zero")
      message(FATAL_ERROR "add_ocloc_kernel(${NAME}): -cl-denorms-are-zero is forbidden (docs/13-loader.md)")
    endif()
    string(APPEND opts " ${o}")
  endforeach()
  # Paths inside the tree spelled from its root, where ocloc runs (see B70_OCLOC_COMMAND above).
  string(REPLACE "${B70_TREE_ROOT}/" "" opts "${opts}")
  get_filename_component(src "${K_SOURCE}" ABSOLUTE)
  file(RELATIVE_PATH src "${B70_TREE_ROOT}" "${src}")
  if(src MATCHES "(^|/)(moe|kol_moe|pf_moe_q4gemm)[.]cl$")
    list(APPEND K_DEPENDS "${B70_KERNEL_SRC_DIR}/expert_fp.h")
  endif()
  if(src MATCHES "(^|/)(moe|kol_moe)[.]cl$")
    list(APPEND K_DEPENDS "${B70_KERNEL_SRC_DIR}/expert_fp_gemv.h")
  endif()
  if(src MATCHES "(^|/)(kv8|kol_attn|kol_pf_attn|ink_prep)[.]cl$")
    list(APPEND K_DEPENDS "${B70_KERNEL_SRC_DIR}/kv8_scheme.h")
  endif()
  set(out "${B70_KERNEL_DIR}/${NAME}.bin")
  add_custom_command(
    OUTPUT "${out}"
    COMMAND ${B70_OCLOC_COMMAND} compile
            -file "${src}" -device "${B70_OCLOC_DEVICE}"
            -output "${NAME}" -output_no_suffix -out_dir "${B70_KERNEL_DIR}"
            -options "${opts}"
    WORKING_DIRECTORY "${B70_TREE_ROOT}"
    DEPENDS "${K_SOURCE}" ${K_DEPENDS} "${B70_KERNEL_SRC_DIR}/bf16.h" "${B70_KERNEL_SRC_DIR}/tp_wire.h"
            "${B70_OCLOC_STAMP}"
    COMMENT "ocloc ${NAME}"
    VERBATIM)
  add_custom_target("kernel_${NAME}" ALL DEPENDS "${out}")
  add_dependencies("kernel_${NAME}" ocloc_toolchain)
endfunction()
