# ocloc AOT rule. One call = one device binary for bmg-g31.
find_program(OCLOC_EXECUTABLE ocloc REQUIRED)
set(B70_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "directory of compiled device binaries")
set(B70_OCLOC_DEVICE "bmg-g31" CACHE STRING "ocloc -device target (B70 = Battlemage G31)")
file(MAKE_DIRECTORY "${B70_KERNEL_DIR}")

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
# reference and the oracle (docs/13-loader.md, "the no-denorm-flush build
# constraint").
function(add_ocloc_kernel NAME)
  cmake_parse_arguments(K "" "SOURCE" "DEFINES;DEPENDS;OPTIONS" ${ARGN})
  if(NOT K_SOURCE)
    message(FATAL_ERROR "add_ocloc_kernel(${NAME}): SOURCE is required")
  endif()
  # The two project-wide options: the language version and the correctly
  # rounded divide/sqrt every host-comparable kernel needs (see above).
  set(opts "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt")
  foreach(d IN LISTS K_DEFINES)
    string(APPEND opts " -D${d}")
  endforeach()
  foreach(o IN LISTS K_OPTIONS)
    if(o STREQUAL "-cl-denorms-are-zero")
      message(FATAL_ERROR "add_ocloc_kernel(${NAME}): -cl-denorms-are-zero is forbidden (docs/13-loader.md)")
    endif()
    string(APPEND opts " ${o}")
  endforeach()
  set(out "${B70_KERNEL_DIR}/${NAME}.bin")
  add_custom_command(
    OUTPUT "${out}"
    COMMAND "${OCLOC_EXECUTABLE}" compile
            -file "${K_SOURCE}" -device "${B70_OCLOC_DEVICE}"
            -output "${NAME}" -output_no_suffix -out_dir "${B70_KERNEL_DIR}"
            -options "${opts}"
    DEPENDS "${K_SOURCE}" ${K_DEPENDS}
    COMMENT "ocloc ${NAME}"
    VERBATIM)
  add_custom_target("kernel_${NAME}" ALL DEPENDS "${out}")
endfunction()
