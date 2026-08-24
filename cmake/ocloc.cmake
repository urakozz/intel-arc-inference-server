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
# OPTIONS appends extra `-cl-` build options for this kernel only. Two rules:
#   * `-cl-denorms-are-zero` is FORBIDDEN project-wide - the 27B's int4 scales go
#     subnormal and flushing them makes the device disagree with both the host
#     reference and the oracle (docs/13-loader.md, "the no-denorm-flush build
#     constraint");
#   * a kernel whose host reference is compared **bit-exactly** and that divides
#     or takes a square root must pass `-cl-fp32-correctly-rounded-divide-sqrt`:
#     without it OpenCL allows 2.5 ulp on both, and the device's `1/sqrt(x)`
#     really does land 1-2 ulp off the host's (measured 2026-08-25 while
#     bringing up prep.cl - docs/12-kernels.md, "Rounding discipline").
function(add_ocloc_kernel NAME)
  cmake_parse_arguments(K "" "SOURCE" "DEFINES;DEPENDS;OPTIONS" ${ARGN})
  if(NOT K_SOURCE)
    message(FATAL_ERROR "add_ocloc_kernel(${NAME}): SOURCE is required")
  endif()
  set(opts "-cl-std=CL3.0")
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
