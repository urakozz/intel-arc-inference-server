# SYCL probes (including probe_interop) are built by the icpx-only sub-project
# src/sycl/CMakeLists.txt, not in this g++-compiled probe target list.
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

# P3 is a plain g++/Level Zero probe: no SYCL dependency and no prefill runtime
# dependency. It consumes every prefill dequant binary, including both store
# orientations, as an independent Stage-0 target.
add_executable(probe_dequant probe_dequant.cc)
target_include_directories(probe_dequant PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_dequant PRIVATE b70_l0)
b70_target_kernel_dir(probe_dequant)
get_property(PF_DEQUANT_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
             PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_dequant ${PF_DEQUANT_KERNELS})

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
# Correctness-candidate GQA packing: 2, 3 or all 6 Q heads share each staged
# K/V wave. Unlike the ablations above, all three must reproduce base's bytes.
add_probe_attn(gpack2      PB_GPACK=2)
add_probe_attn(gpack3      PB_GPACK=3)
add_probe_attn(gpack6      PB_GPACK=6)
# Same loop inversion with each work-item's K/V slice retained privately.
add_probe_attn(greg2       PB_GREG=2)
add_probe_attn(greg3       PB_GREG=3)
add_probe_attn(greg6       PB_GREG=6)
# P3-prep (spec 1.7 §3): the GEMV battery's 2D-block technique transferred to the
# KV walk. 16 messages of 32 B per K row become 2 (`b2d`) or 1 (`b2d1`), with
# the SAME values in the SAME fma order -- the only probe_attn variants that are
# bit-identical to `base` rather than deliberately wrong.
add_probe_attn(b2d         PB_B2D=1)
add_probe_attn(b2d1        PB_B2D=2)
add_probe_attn(kvt_b2d     PB_KT=1 PB_VT=1 PB_B2D=1)
add_probe_attn(b32         PB_BLOCK=32)                  # the L5 sweep, transplanted
add_probe_attn(b128        PB_BLOCK=128)
add_probe_attn(b256        PB_BLOCK=256)

add_executable(probe_attn probe_attn.cc)
target_include_directories(probe_attn PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_attn PRIVATE b70_l0)
b70_target_kernel_dir(probe_attn)
add_dependencies(probe_attn ${PROBE_ATTN_TARGETS})

# probe_gemv_loads - spec 1.7 §3's P1 battery: the SAME arithmetic as the
# production int4 GEMV with a different number of load messages under it, at
# the six production shapes. `tools/probe/probe_gemv_loads.cl` states the
# message-count model each tag moves; `probe_gemv --loads` runs them.
#
# One tag = one `-D` set = one binary = one row, and the tag list here and the
# `kVariants` table in probe_gemv.cc are two lists nothing checks against each
# other -- a tag whose binary is missing throws by name at `l0::Module`, which
# is the same guard probe_attn uses one directory up.
#
# These are PROBE kernels. `src/kernels/gemv.cl` is not touched by any of them,
# no runtime path binds them, and with every PGL_* knob at 0 the source IS
# gemv.cl -- which is what makes `base` a control rather than a paraphrase.
set(PROBE_GEMV_LOADS_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_gemv_loads.cl)
function(add_probe_gemv_loads TAG K N S L)
  add_ocloc_kernel(pgl_${TAG}_M1_K${K}_N${N}_S${S}_L${L} SOURCE ${PROBE_GEMV_LOADS_CL}
                   DEFINES M=1 K=${K} N=${N} S=${S} LAYOUT=${L} ${ARGN})
  list(APPEND PROBE_GEMV_LOADS_TARGETS kernel_pgl_${TAG}_M1_K${K}_N${N}_S${S}_L${L})
  set(PROBE_GEMV_LOADS_TARGETS ${PROBE_GEMV_LOADS_TARGETS} PARENT_SCOPE)
endfunction()

# The six production shapes at the S the engine binds (model/qwen35.cc), plus
# the S layout 0 picks for itself in docs/probe-gemv-2026-08-24.md -- the
# layout-0 rows are the battery's only apples-to-apples message-count contrast
# (17 messages per subgroup-step against layout 1's 10, byte-identical traffic).
#   K N S_prod S_l0best  blk2d16-able (G/S even)
set(PGL_SHAPES "6144 5120 16 4 1" "5120 14336 1 2 1" "5120 16384 1 16 1"
               "5120 34816 4 8 1" "17408 5120 16 4 0" "5120 248320 1 1 1")
foreach(row IN LISTS PGL_SHAPES)
  separate_arguments(f UNIX_COMMAND "${row}")
  list(GET f 0 K)
  list(GET f 1 N)
  list(GET f 2 SP)
  list(GET f 3 S0)
  list(GET f 4 B16)
  # layout 1 - the production load path and the four things that can be done to it
  add_probe_gemv_loads(base    ${K} ${N} ${SP} 1)
  add_probe_gemv_loads(xwide   ${K} ${N} ${SP} 1 PGL_XWIDE=1)
  add_probe_gemv_loads(deq     ${K} ${N} ${SP} 1 PGL_DEQ_SHIFT=1)
  add_probe_gemv_loads(pf1     ${K} ${N} ${SP} 1 PGL_PREFETCH=1)
  add_probe_gemv_loads(pf2     ${K} ${N} ${SP} 1 PGL_PREFETCH=2)  # bestla's production distance
  add_probe_gemv_loads(pf4     ${K} ${N} ${SP} 1 PGL_PREFETCH=4)
  add_probe_gemv_loads(pfbuf   ${K} ${N} ${SP} 1 PGL_PFBUF=1)
  add_probe_gemv_loads(ballast ${K} ${N} ${SP} 1 PGL_BALLAST=1)   # pfbuf's register control
  # No PGL_CACHECTL variants are built. The knob still exists and is documented
  # in probe_gemv_loads.cl, but it was MEASURED BROKEN 2026-08-26 -- the builtin
  # compiles and does not reproduce block_read8's data mapping under the
  # signature we declared, so its GB/s is meaningless. Dead binaries would only
  # invite someone to read them.
  # layout 0 - where a 2D block read has messages to remove, at both S values
  add_probe_gemv_loads(l0base  ${K} ${N} ${SP} 0)
  add_probe_gemv_loads(l0b2d   ${K} ${N} ${SP} 0 PGL_BLK2D=8)
  add_probe_gemv_loads(l0b2dx  ${K} ${N} ${SP} 0 PGL_BLK2D=8 PGL_XWIDE=1)
  if(B16)
    add_probe_gemv_loads(l0b2d16 ${K} ${N} ${SP} 0 PGL_BLK2D=16)
  endif()
  add_probe_gemv_loads(l0deq   ${K} ${N} ${SP} 0 PGL_DEQ_SHIFT=1)
  if(NOT S0 EQUAL SP)
    add_probe_gemv_loads(l0base ${K} ${N} ${S0} 0)
    add_probe_gemv_loads(l0b2d  ${K} ${N} ${S0} 0 PGL_BLK2D=8)
    # The composed cell (fix round I1): the retuned layout+S AND the dequant
    # change together, because `deq`'s gain shrinks as a shape approaches the
    # wall and the retuned cells sit at 94-97% of it. Composing them by
    # multiplication would over-count.
    add_probe_gemv_loads(l0deq    ${K} ${N} ${S0} 0 PGL_DEQ_SHIFT=1)
    add_probe_gemv_loads(l0b2ddeq ${K} ${N} ${S0} 0 PGL_BLK2D=8 PGL_DEQ_SHIFT=1)
    # Layout 1 at the SAME alternate S (fix round I2): layout and S were varied
    # together in the first battery, so this is the row that separates them --
    # an S-table one-liner inside layout 1 against the layout-0 rider.
    add_probe_gemv_loads(base   ${K} ${N} ${S0} 1)
    add_probe_gemv_loads(deq    ${K} ${N} ${S0} 1 PGL_DEQ_SHIFT=1)
  endif()
endforeach()
add_dependencies(probe_gemv ${PROBE_GEMV_LOADS_TARGETS})
