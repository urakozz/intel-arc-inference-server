# Host-only configuration (B70_HOST_ONLY=ON, preset `mac-host`): configure, build and
# ctest the tests that need no Level Zero device, on a machine without a Level Zero
# runtime, ocloc or icpx - the Mac. docs/18-mac-checks.md has the why; the short form:
#
# **The test list is derived, not listed.** tests/CMakeLists.txt is shared by every
# branch in flight, and a parallel list of "the host tests" would drift the day
# somebody adds a test. So this file changes no registration at all. It stands in for
# the three things the full build needs from the box (the Level Zero package, ocloc,
# icpx), records every add_test as it happens, DISABLES each test that needs a device
# or box-only data at the end of the directory that added it, and - last, from the
# top-level CMakeLists.txt (b70_host_only_finish) - writes the reasons to
# <build>/host_only_skipped.txt and takes everything else out of `all`. A plain `ctest`
# then runs exactly the host tests and lists the rest as "Not Run (Disabled)", which
# ctest does not count as a failure.
#
# **What makes a test a device test** (any one is enough):
#   - its target links b70_l0, b70_runtime, b70_prefill_host or b70_prefill directly
#     (b70_loader also links b70_l0 transitively, but the loader's host tests use only
#     its pure parts - static-archive linking pulls in no Level Zero object for them,
#     and a test that DID call into the device would fail to link here, which is the
#     honest signal);
#   - it depends on a kernel binary (kernel_*) or on b70-decode / b70-serve. NOT merely
#     carrying B70_KERNEL_DIR: kernels/kernels.h needs it to compile, and
#     variant_names_test uses it for names only;
#   - its LABELS contain `checkpoint` (the 19 GB model lives on the box);
#   - its command is not a test executable of this tree (the b70_cli_reject `sh -c`
#     blocks and tools/probe/*.sh drive the device binaries).
# And a host test is skipped for **missing data** when an absolute path on its command
# line does not exist here, or when it reads the Qwen3.8 tokenizer.json through
# tok::default_tokenizer_json() (B70_HOST_ONLY_NEEDS_QWEN_TOKENIZER below - the one
# list in this file, because that dependency is in the test's code, not its command).
#
# **Level Zero headers are still required.** b70_loader's sources include
# <level_zero/ze_api.h> and the loader host tests link it, so the headers (not the
# runtime) must be present: B70_L0_INCLUDE (default $ENV{L0_INCLUDE}) points at a
# level-zero checkout's include/ and a `level_zero` symlink to it is made in the build
# tree, because the checkout keeps ze_api.h at the top level while the code spells
# <level_zero/ze_api.h>. PkgConfig::ZE becomes an INTERFACE target carrying only that
# include directory; anything that really calls ze* fails to link, as it should.

set(B70_L0_INCLUDE "$ENV{L0_INCLUDE}" CACHE PATH
    "Level Zero headers (a level-zero checkout's include/, holding ze_api.h) for the host-only build")
if(NOT EXISTS "${B70_L0_INCLUDE}/ze_api.h")
  message(FATAL_ERROR "B70_HOST_ONLY: no ze_api.h under B70_L0_INCLUDE='${B70_L0_INCLUDE}'. "
                      "Set L0_INCLUDE (or -DB70_L0_INCLUDE) to a level-zero checkout's include/ "
                      "(git clone https://github.com/oneapi-src/level-zero).")
endif()
set(_ho_l0_root "${CMAKE_BINARY_DIR}/l0-include")
file(MAKE_DIRECTORY "${_ho_l0_root}")
if(NOT EXISTS "${_ho_l0_root}/level_zero/ze_api.h")
  file(REMOVE "${_ho_l0_root}/level_zero")
  file(CREATE_LINK "${B70_L0_INCLUDE}" "${_ho_l0_root}/level_zero" SYMBOLIC)
endif()
add_library(PkgConfig::ZE INTERFACE IMPORTED GLOBAL)
set_target_properties(PkgConfig::ZE PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_ho_l0_root}")

# The SYCL prefill component is icpx-only; AUTO already finds no icpx on a Mac, and
# ON cannot mean anything here.
if(B70_PREFILL STREQUAL "ON")
  message(FATAL_ERROR "B70_HOST_ONLY=ON with B70_PREFILL=ON: the SYCL component needs icpx")
endif()

# ocloc stand-in: no binary is produced, but kernel_<name> exists so every
# add_dependencies(... kernel_<name>) in the tree still resolves (and marks the
# depending target as a device test below).
set(B70_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "directory of compiled device binaries")
function(add_ocloc_kernel NAME)
  add_custom_target("kernel_${NAME}")
endfunction()

# Tests whose code (not command line) reads the Qwen3.8 snapshot's tokenizer.json via
# tok::default_tokenizer_json(): B70_TOKENIZER_JSON or HF_HOME's snapshot.
set(B70_HOST_ONLY_NEEDS_QWEN_TOKENIZER tokenizer_smoke_test parity_test template_test streamer_test)

# add_test is wrapped (once; the original stays reachable as _add_test) for two
# reasons: CMake has no property that returns a test's command, and a test's
# properties can only be set from the directory that added it before CMake 3.28 - so
# the first add_test of each directory defers _b70_ho_classify_dir to the end of that
# directory, when every set_tests_properties (LABELS) of it has run.
function(add_test)
  cmake_parse_arguments(T "" "NAME" "COMMAND" ${ARGN})
  if(T_NAME)
    set_property(DIRECTORY APPEND PROPERTY B70_HO_TESTS "${T_NAME}")
    set_property(GLOBAL PROPERTY "B70_HO_CMD_${T_NAME}" "${T_COMMAND}")
    get_property(deferred DIRECTORY PROPERTY B70_HO_DEFERRED)
    if(NOT deferred)
      set_property(DIRECTORY PROPERTY B70_HO_DEFERRED TRUE)
      cmake_language(DEFER CALL _b70_ho_classify_dir)
    endif()
  endif()
  _add_test(${ARGV})
endfunction()

function(_b70_ho_collect_targets dir out)
  get_property(t DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  get_property(subs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(s IN LISTS subs)
    _b70_ho_collect_targets("${s}" st)
    list(APPEND t ${st})
  endforeach()
  set(${out} ${t} PARENT_SCOPE)
endfunction()

# Why `tgt` (a test's executable) needs a device, or "" when it does not.
function(_b70_ho_target_reason tgt out)
  set(reason "")
  get_target_property(libs ${tgt} LINK_LIBRARIES)
  foreach(l IN ITEMS b70_l0 b70_runtime b70_prefill_host b70_prefill)
    if(libs AND l IN_LIST libs)
      set(reason "links ${l} (Level Zero device)")
      break()
    endif()
  endforeach()
  if(NOT reason)
    get_target_property(deps ${tgt} MANUALLY_ADDED_DEPENDENCIES)
    foreach(d IN LISTS deps)
      if(d MATCHES "^kernel_" OR d STREQUAL "b70-decode" OR d STREQUAL "b70-serve")
        set(reason "depends on ${d} (device binary)")
        break()
      endif()
    endforeach()
  endif()
  set(${out} "${reason}" PARENT_SCOPE)
endfunction()

# Runs at the end of each directory that added tests (see add_test above).
function(_b70_ho_classify_dir)
  # The Qwen3.8 tokenizer, where tok::default_tokenizer_json() will look for it.
  if(DEFINED ENV{B70_TOKENIZER_JSON})
    set(qwen_tok "$ENV{B70_TOKENIZER_JSON}")
  else()
    set(qwen_tok "${B70_TEST_SNAPSHOT}/tokenizer.json")
  endif()

  get_property(tests DIRECTORY PROPERTY B70_HO_TESTS)
  foreach(t IN LISTS tests)
    get_property(cmd GLOBAL PROPERTY "B70_HO_CMD_${t}")
    list(GET cmd 0 exe)
    set(reason "")
    if(NOT TARGET "${exe}")
      set(reason "not a test executable of this tree (drives device binaries)")
    else()
      _b70_ho_target_reason(${exe} reason)
    endif()
    if(NOT reason)
      get_test_property(${t} LABELS labels)
      if(labels AND "checkpoint" IN_LIST labels)
        set(reason "label checkpoint (needs the model on the box)")
      endif()
    endif()
    if(NOT reason)
      set(args ${cmd})
      list(REMOVE_AT args 0)
      foreach(a IN LISTS args)
        if(a MATCHES "^/" AND NOT EXISTS "${a}")
          set(reason "needs ${a} (box data)")
          break()
        endif()
      endforeach()
    endif()
    if(NOT reason AND exe IN_LIST B70_HOST_ONLY_NEEDS_QWEN_TOKENIZER AND NOT EXISTS "${qwen_tok}")
      list(LENGTH cmd ncmd)
      # template_test with a snapshot argument (template_agnes_test) renders that
      # snapshot's template and never opens the default tokenizer.
      if(NOT (exe STREQUAL "template_test" AND ncmd GREATER 3))
        set(reason "needs ${qwen_tok} (the Qwen3.8 tokenizer, or set B70_TOKENIZER_JSON)")
      endif()
    endif()
    if(reason)
      set_tests_properties(${t} PROPERTIES DISABLED TRUE)
      set_property(GLOBAL APPEND PROPERTY B70_HO_SKIPPED "${t} | ${reason}")
    else()
      set_property(GLOBAL APPEND PROPERTY B70_HO_RUN "${t}")
      set_property(GLOBAL APPEND PROPERTY B70_HO_KEEP "${exe}")
    endif()
  endforeach()
endfunction()

# Called last from the top-level CMakeLists.txt, after every directory's deferred
# classification has run (deferred calls fire at the end of their own directory,
# and tests/ is a subdirectory of the top level).
function(b70_host_only_finish)
  get_property(run GLOBAL PROPERTY B70_HO_RUN)
  get_property(skipped GLOBAL PROPERTY B70_HO_SKIPPED)
  get_property(keep_targets GLOBAL PROPERTY B70_HO_KEEP)
  string(REPLACE ";" "\n" report "${skipped}")
  file(WRITE "${CMAKE_BINARY_DIR}/host_only_skipped.txt" "${report}\n")
  string(REPLACE ";" "\n" ran "${run}")
  file(WRITE "${CMAKE_BINARY_DIR}/host_only_tests.txt" "${ran}\n")
  list(LENGTH run n_run)
  list(LENGTH skipped n_skip)

  # `all` = the enabled tests' executables and what they link, nothing else: the
  # device libraries, b70-decode/b70-serve and every disabled test stay out of the
  # default build (tools/mac_check.sh syntax-checks their sources instead).
  _b70_ho_collect_targets("${CMAKE_SOURCE_DIR}" all_targets)
  foreach(tg IN LISTS all_targets)
    get_target_property(type ${tg} TYPE)
    if(type STREQUAL "UTILITY" OR type STREQUAL "INTERFACE_LIBRARY")
      continue()
    endif()
    if(NOT tg IN_LIST keep_targets)
      set_target_properties(${tg} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
  endforeach()
  message(STATUS "b70: host-only - ${n_run} tests to run, ${n_skip} disabled "
                 "(${CMAKE_BINARY_DIR}/host_only_skipped.txt)")
endfunction()
