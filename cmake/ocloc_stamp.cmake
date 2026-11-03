# cmake -P ocloc_stamp.cmake <stamp file> <ocloc>: writes the device compiler's identity (ocloc --version and the
# libigc / libigdfcl ldconfig resolves first, by real path) to the stamp - only when it changed, so its mtime moves
# only then. Every kernel depends on the stamp (ocloc.cmake): a driver / IGC upgrade recompiles them all, which their
# sources' timestamps alone never would (2026-11-04: compute-runtime 26.40 + IGC 2.42.6 landed beside the old
# 2.41.15 in another directory, and `make` kept every binary the old compiler had built).
set(stamp "${CMAKE_ARGV3}")
execute_process(COMMAND "${CMAKE_ARGV4}" --version OUTPUT_VARIABLE ver ERROR_VARIABLE ver)
set(id "ocloc: ${ver}")
execute_process(COMMAND ldconfig -p OUTPUT_VARIABLE cache ERROR_QUIET)
foreach(lib libigc.so.2 libigdfcl.so.2)
  string(REGEX MATCH "${lib} [^\n]*=> ([^\n]*)" m "${cache}")
  set(p "${CMAKE_MATCH_1}")
  if(p)
    file(REAL_PATH "${p}" p)
  endif()
  string(APPEND id "\n${lib}: ${p}")
endforeach()
set(old "")
if(EXISTS "${stamp}")
  file(READ "${stamp}" old)
endif()
if(NOT old STREQUAL id)
  file(WRITE "${stamp}" "${id}")
  message(STATUS "ocloc toolchain changed: every kernel recompiles")
endif()
