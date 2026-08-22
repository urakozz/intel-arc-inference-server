# ocloc AOT rule. One call = one device binary for bmg-g31.
find_program(OCLOC_EXECUTABLE ocloc REQUIRED)
set(B70_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "directory of compiled device binaries")
set(B70_OCLOC_DEVICE "bmg-g31" CACHE STRING "ocloc -device target (B70 = Battlemage G31)")
file(MAKE_DIRECTORY "${B70_KERNEL_DIR}")

# add_ocloc_kernel(<name> SOURCE <file.cl> [DEFINES A=1 B=2 ...] [DEPENDS <files>])
# Produces ${B70_KERNEL_DIR}/<name>.bin (ZE_MODULE_FORMAT_NATIVE). <name> is the
# variant name the runtime asks for, e.g. gemv_M1_K5120_N5120_S1_L0.
function(add_ocloc_kernel NAME)
  cmake_parse_arguments(K "" "SOURCE" "DEFINES;DEPENDS" ${ARGN})
  if(NOT K_SOURCE)
    message(FATAL_ERROR "add_ocloc_kernel(${NAME}): SOURCE is required")
  endif()
  set(opts "-cl-std=CL3.0")
  foreach(d IN LISTS K_DEFINES)
    string(APPEND opts " -D${d}")
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
