add_executable(probe_bw probe_bw.cc)
target_include_directories(probe_bw PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_bw PRIVATE b70_l0)
b70_target_kernel_dir(probe_bw)
add_dependencies(probe_bw kernel_bw_sum)

add_executable(probe_replay probe_replay.cc)
target_include_directories(probe_replay PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_replay PRIVATE b70_l0)
b70_target_kernel_dir(probe_replay)
add_dependencies(probe_replay kernel_noop kernel_ctrl_read)

add_executable(probe_gemv probe_gemv.cc)
target_include_directories(probe_gemv PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_gemv PRIVATE b70_l0)
b70_target_kernel_dir(probe_gemv)
# depends on every gemv variant; the kernels target list is in src/kernels
get_property(ALL_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_gemv ${ALL_KERNELS})

# probe_attn - the memo's §5.2 item: `attn_decode` with one term removed at a
# time, so what the 224 us/launch buys is a subtraction between measurements
# rather than a fifth cost model (four have died on that kernel already).
#
# One tag = one binary = one row of the probe's table. The tags below and the
# `kVariants` table in probe_attn.cc are two lists nothing checks against each
# other; a tag with a missing binary throws by name at `l0::Module`, which is
# the guard ATTN_BLOCK uses across its own three homes (src/kernels).
#
# These are PROBE kernels. They compute wrong answers on purpose, no runtime
# path binds them, and `src/kernels/attn.cl` is not touched by any of them.
set(PROBE_ATTN_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_attn.cl)
function(add_probe_attn TAG)
  add_ocloc_kernel(probe_attn_${TAG} SOURCE ${PROBE_ATTN_CL} DEFINES ${ARGN})
  list(APPEND PROBE_ATTN_TARGETS kernel_probe_attn_${TAG})
  set(PROBE_ATTN_TARGETS ${PROBE_ATTN_TARGETS} PARENT_SCOPE)
endfunction()
add_probe_attn(base)                                     # the control: attn_decode, verbatim
add_probe_attn(earlyout    PB_STAGE=1)                   # the grid floor
add_probe_attn(qstage      PB_STAGE=2)                   # + the 6 q-head stagings
add_probe_attn(nok         PB_NO_K=1)                    # - the K global load
add_probe_attn(nov         PB_NO_V=1)                    # - the V global load
add_probe_attn(noloads     PB_NO_K=1 PB_NO_V=1)          # - both
add_probe_attn(hotk        PB_HOT_K=1)                   # same K messages, all L1-resident
add_probe_attn(hotv        PB_HOT_V=1)
add_probe_attn(hotkv       PB_HOT_K=1 PB_HOT_V=1)
# The LICM control (review issue I2). `hotk`/`hotv`/`hotkv` hold the address
# constant across the wave loop, which IGC may legally hoist; these three keep
# the 8 KB L1-resident footprint but make the row a function of `w`, so the
# message COUNT is provably the base kernel's. hotkv2 == hotkv means no
# hoisting happened and the cache/issue split stands.
add_probe_attn(hotk2       PB_HOT_K=1 PB_HOT_W=1)
add_probe_attn(hotv2       PB_HOT_V=1 PB_HOT_W=1)
add_probe_attn(hotkv2      PB_HOT_K=1 PB_HOT_V=1 PB_HOT_W=1)
add_probe_attn(kt          PB_KT=1)                      # K from a [head][pos][dim] cache
add_probe_attn(vt          PB_VT=1)                      # V from a [head][pos][dim] cache
add_probe_attn(kvt         PB_KT=1 PB_VT=1)              # both
add_probe_attn(kvt_vec2    PB_KT=1 PB_VT=1 PB_VEC=2)     # both, plus 64 B K messages
add_probe_attn(nopart      PB_NO_PART=1)                 # - the attn_part stores
add_probe_attn(vec2        PB_VEC=2)                     # 64 B K messages instead of 32 B
add_probe_attn(vec4        PB_VEC=4)                     # 128 B K messages
add_probe_attn(natexp      PB_EXP=1)                     # exp -> native_exp
add_probe_attn(noexp       PB_EXP=2)                     # exp -> a multiply-add
add_probe_attn(notree      PB_NO_TREE=1)                 # - the 4 tree steps and 4 barriers
add_probe_attn(nobar       PB_NO_BARRIER=1)              # - all 6 barriers per wave
add_probe_attn(sgtree      PB_SG_TREE=1)                   # 3 of 4 tree barriers -> subgroup scope
add_probe_attn(nosoftmax   PB_NO_SOFTMAX=1)              # - the whole online update
add_probe_attn(depbreak    PB_DEP_BREAK=1)               # - the loop-carried chain
add_probe_attn(prefetch    PB_PREFETCH=1)                # K loads one wave ahead
add_probe_attn(mathonly    PB_NO_K=1 PB_NO_V=1 PB_EXP=2) # - memory AND transcendental
add_probe_attn(loadsonly   PB_NO_SOFTMAX=1 PB_NO_TREE=1) # loads + dot + V sum only
add_probe_attn(gqa1        PB_GQA=1)
add_probe_attn(gqa2        PB_GQA=2)
add_probe_attn(gqa3        PB_GQA=3)
add_probe_attn(b32         PB_BLOCK=32)                  # the L5 sweep, transplanted
add_probe_attn(b128        PB_BLOCK=128)
add_probe_attn(b256        PB_BLOCK=256)

add_executable(probe_attn probe_attn.cc)
target_include_directories(probe_attn PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_attn PRIVATE b70_l0)
b70_target_kernel_dir(probe_attn)
add_dependencies(probe_attn ${PROBE_ATTN_TARGETS})
