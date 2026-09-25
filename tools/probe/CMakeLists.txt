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

# The dequant-overlap lever (spec 2, plan 6c Task 4). Unlike P3 it needs BOTH
# queues -- the L0 immediate list for pf_dequant_tile and the SYCL queue for
# gemm_bf16 -- so it links libb70_prefill.so and lives under the same guard
# every other prefill target uses. It contains no SYCL of its own, so it is a
# plain g++ target here rather than one of src/sycl's icpx probes.
if(B70_PREFILL_ENABLED)
  # `runtime::prefill::Context` and `gemm_bf16*` now live in b70_prefill_host
  # (spec 2.1 §3.5, ruling B5).
  add_executable(probe_dequant_overlap probe_dequant_overlap.cc)
  target_include_directories(probe_dequant_overlap PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_dequant_overlap PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(probe_dequant_overlap)
  b70_target_kernel_dir(probe_dequant_overlap)
  add_dependencies(probe_dequant_overlap kernel_pf_dequant_tile_K5120_N34816_L0_T0)

  # The L2-resident slab lever. Same shape as the overlap probe (gate‖up,
  # K=5120, N=34816, M=2048) and the same two queues, so it links the same way.
  #
  # `pds_*` are PROBE kernels: `src/kernels/prefill/dequant.cl` is not touched,
  # no runtime path binds them, and at NS == N with n0 == 0 the source IS that
  # kernel's layout-0 / [K][N] path -- which is what makes the probe's C1'
  # equivalence cell a control rather than a paraphrase. The width list here and
  # `kWidths` in probe_dequant_slab.cc are two lists nothing checks against each
  # other; a width whose binary is missing is named and refused at `l0::Module`.
  set(PROBE_SLAB_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_dequant_slab.cl)
  set(PROBE_SLAB_TARGETS "")
  foreach(NS 1024 2048 4096 34816)
    add_ocloc_kernel(pds_K5120_N34816_NS${NS} SOURCE ${PROBE_SLAB_CL}
                     DEFINES K=5120 N=34816 NS=${NS})
    list(APPEND PROBE_SLAB_TARGETS kernel_pds_K5120_N34816_NS${NS})
  endforeach()

  add_executable(probe_dequant_slab probe_dequant_slab.cc)
  target_include_directories(probe_dequant_slab PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_dequant_slab PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(probe_dequant_slab)
  b70_target_kernel_dir(probe_dequant_slab)
  add_dependencies(probe_dequant_slab kernel_pf_dequant_tile_K5120_N34816_L0_T0
                   ${PROBE_SLAB_TARGETS})

  # P-A (spec 2 addendum, docs/probe-prefill-vllm-parity-2026-09-14.md §A4):
  # our own OpenCL C bf16 DPAS GEMM on the Level Zero list -- the go/no-go
  # candidate for dropping sycl-tla from the prefill path. AOT at 256 GRF is
  # MANDATORY (§A4.8): P2 measured a 33x loss without it, and zero spill does
  # not mean good code. `pf_gemm_bf16.cl` is a probe-only kernel: no runtime
  # path binds it and no existing kernel is touched. This target links
  # b70_prefill (like probe_dequant_slab above) purely to call `gemm_bf16` as
  # the C2 control beside every timed cell -- it contains no SYCL of its own.
  add_ocloc_kernel(pf_gemm_bf16 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemm_bf16.cl
                   OPTIONS -cl-intel-256-GRF-per-thread)

  add_executable(probe_pf_gemm probe_pf_gemm.cc)
  target_include_directories(probe_pf_gemm PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_pf_gemm PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(probe_pf_gemm)
  b70_target_kernel_dir(probe_pf_gemm)
  add_dependencies(probe_pf_gemm kernel_pf_gemm_bf16)

  # P-B (docs/probe-prefill-vllm-parity-2026-09-14.md §A3.2): slab dequant +
  # P-A's GEMM on ONE in-order L0 list, zero host waits. Runs only because
  # P-A measured >= 128 (committed: 156.53 TFLOP/s). `pf_gemm_bf16_slab` is a
  # second entry point in the SAME pf_gemm_bf16.cl / kernel_pf_gemm_bf16
  # binary as P-A's already-gated kernel -- pf_gemm_bf16 itself is untouched.
  add_executable(probe_pf_gemm_slab probe_pf_gemm_slab.cc)
  target_include_directories(probe_pf_gemm_slab PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_pf_gemm_slab PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(probe_pf_gemm_slab)
  b70_target_kernel_dir(probe_pf_gemm_slab)
  add_dependencies(probe_pf_gemm_slab kernel_pf_dequant_tile_K5120_N34816_L0_T0
                   kernel_pf_gemm_bf16 ${PROBE_SLAB_TARGETS})
endif()

# T6's M=2048 small-kernel check. These are probe-only ocloc binaries: no
# runtime variant helper names them, so the decode and future prefill bindings
# cannot select them by accident.
set(PREFILL_SMALL_PROBE_CL ${CMAKE_SOURCE_DIR}/src/kernels/prep.cl)
function(add_prefill_small_probe_kernel TAG)
  add_ocloc_kernel(pfs_${TAG}_M2048 SOURCE ${PREFILL_SMALL_PROBE_CL}
                   DEFINES M=2048 ${ARGN})
  list(APPEND PREFILL_SMALL_PROBE_TARGETS kernel_pfs_${TAG}_M2048)
  set(PREFILL_SMALL_PROBE_TARGETS ${PREFILL_SMALL_PROBE_TARGETS} PARENT_SCOPE)
endfunction()
add_prefill_small_probe_kernel(res_fold_sp4 S_PREV=4 FOLD_G=20)
add_prefill_small_probe_kernel(norm_finish NORM_G=20 NORM_WGS=20)
add_prefill_small_probe_kernel(silu_mul)
add_prefill_small_probe_kernel(gated_head)

add_executable(probe_prefill_small probe_prefill_small.cc)
target_include_directories(probe_prefill_small PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_prefill_small PRIVATE b70_l0)
b70_target_kernel_dir(probe_prefill_small)
# Both batteries: the probe-only decode-S binaries above AND the production
# prefill ones from src/kernels/prefill. The second list is deliberately spelled
# out rather than taken as that directory's whole BUILDSYSTEM_TARGETS (which
# would also drag in every pf_dequant variant probe_dequant owns).
add_dependencies(probe_prefill_small ${PREFILL_SMALL_PROBE_TARGETS}
  kernel_pf_res_fold_K5120_SP1_G20 kernel_pf_norm_finish_K5120_G20_W20
  kernel_pf_silu_mul kernel_pf_gated_head kernel_pf_attn_prep kernel_pf_ab_proj
  kernel_pf_embed_gather)

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

# --- S1: the fused int4 dequant probe (2026-09-22) --------------------------
# Pre-registration: docs/specs/2026-09-22-fused-dequant-gemm-probe-design.md.
# `probe_fused_dequant_gemm.cl` is a PROBE kernel: no runtime path binds
# `pfd_gemm`, and neither `src/kernels/prefill/pf_gemm.cl` nor
# `pf_dequant_slab.cl` is touched -- the probe's CONTROL runs those two
# production binaries themselves, which is what makes the memcmp a gate rather
# than a paraphrase. 256 GRF is mandatory for anything sharing pf_gemm's tile
# (P2 measured a 33x loss without it) and is carried on the compile line, the
# only place an AOT build honours it.
if(B70_PREFILL_ENABLED)
  set(PROBE_FUSED_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_fused_dequant_gemm.cl)
  add_ocloc_kernel(pfd_gemm_v1 SOURCE ${PROBE_FUSED_CL} DEFINES FUSED=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pfd_gemm_v2 SOURCE ${PROBE_FUSED_CL} DEFINES FUSED=2
                   OPTIONS -cl-intel-256-GRF-per-thread)

  add_executable(probe_fused_dequant_gemm probe_fused_dequant_gemm.cc)
  target_include_directories(probe_fused_dequant_gemm PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_fused_dequant_gemm PRIVATE b70_prefill_host b70_l0)
  b70_link_prefill(probe_fused_dequant_gemm)
  b70_target_kernel_dir(probe_fused_dequant_gemm)
  add_dependencies(probe_fused_dequant_gemm kernel_pfd_gemm_v1 kernel_pfd_gemm_v2
                   kernel_pf_gemm_T0 kernel_pf_dequant_slab_K5120_N34816_L0)
endif()

# --- probe_dpas_rate: the per-type DPAS issue rate (2026-09-22) -------------
# Record: docs/probe-dpas-rates-2026-09-22.md. `probe_dpas_rate.cl` is a PROBE
# kernel -- no runtime path binds `dpas_rate`, no production file is touched,
# and it computes a deliberately meaningless dot product. One binary per data
# type; `kVariants` in probe_dpas_rate.cc and the tag list below are two lists
# nothing checks against each other, so a tag whose binary is missing is named
# and skipped at `l0::Module` (the probe prints NO BINARY rather than dying).
#
# 256 GRF is mandatory here for the same reason it is for pf_gemm: eight live
# float8/int8 accumulators plus eight operand registers is ~112 GRF, which
# spills at the 128-GRF default and would measure the spill, not the pipe.
#
# THE TYPES THAT ARE NOT HERE ARE NOT AN OVERSIGHT. On bmg-g31 with ocloc
# 26.35.39758.10 the backend REFUSES them, and the record quotes the output:
#   * every FP4 form (`e2m1_e2m1_matrix_mad_k64`, scaled or not) --
#     "error: FP4 Dpas instruction is not supported on this device!";
#   * every `*_scaled_matrix_mad_*` form (bf16, f16, hf8, bf8, e2m1) --
#     "IGC: Internal Compiler Error: Abnormal termination", reproduced from a
#     single-call kernel, so it is the builtin and not this loop;
#   * `intel_sub_group_e4m3_e4m3_matrix_mad_k32` and every other unscaled FP8
#     spelling -- undeclared identifier; the family exists in libigc's strings
#     but this compiler declares no OpenCL C entry point for it.
# Adding any of them back turns the whole build red, which is the point.
set(PROBE_DPAS_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_dpas_rate.cl)
set(PROBE_DPAS_TARGETS "")
foreach(row IN ITEMS "bf16 1" "f16 2" "i8 3" "i4i8 4" "i4 5" "i2 6" "i8i4 11" "u4 12")
  separate_arguments(f UNIX_COMMAND "${row}")
  list(GET f 0 TAG)
  list(GET f 1 DT)
  add_ocloc_kernel(pdr_${TAG} SOURCE ${PROBE_DPAS_CL} DEFINES DT=${DT}
                   OPTIONS -cl-intel-256-GRF-per-thread)
  list(APPEND PROBE_DPAS_TARGETS kernel_pdr_${TAG})
endforeach()

add_executable(probe_dpas_rate probe_dpas_rate.cc)
target_include_directories(probe_dpas_rate PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_dpas_rate PRIVATE b70_l0)
b70_target_kernel_dir(probe_dpas_rate)
add_dependencies(probe_dpas_rate ${PROBE_DPAS_TARGETS})

# --- probe_w4a8: W4A8 on the mixed 4-bit x 8-bit k32 DPAS (2026-09-23) -------
# Pre-registration: docs/specs/2026-09-22-w4a8-dpas-probe-design.md.
# Record: docs/probe-w4a8-2026-09-23.md.
#
# `probe_w4a8.cl` is a PROBE kernel file with two entry points -- `pw4a8_gemm`
# and `pw4a8_quant`. No runtime path binds either, and no production file is
# touched: the probe's CONTROL runs the production `pf_dequant_slab` and
# `pf_gemm_T0` binaries themselves, and its inputs are the real gate||up
# weights and the real activations one prefill chunk fed them (the --capture
# mode reaches those through `loader::load` and `runtime::Engine::prefill`).
#
# 256 GRF is mandatory, as for everything else that shares this tile shape:
# `pw4a8_gemm` holds EIGHT int32 accumulators and EIGHT fp32 accumulators live
# at once (the int32 one is reset at every g64 boundary, the fp32 one is not),
# which is 128 GRF of accumulator before a single operand register. At the
# 128-GRF default it would measure the spill instead of the pipe.
if(B70_PREFILL_ENABLED)
  add_ocloc_kernel(pw4a8 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_w4a8.cl
                   OPTIONS -cl-intel-256-GRF-per-thread)

  add_executable(probe_w4a8 probe_w4a8.cc)
  target_include_directories(probe_w4a8 PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_w4a8 PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0
                        b70_model)
  b70_link_prefill(probe_w4a8)
  b70_target_kernel_dir(probe_w4a8)
  # --capture drives the whole engine, so it needs every kernel binary, not
  # just this probe's three. Taken as the two kernel directories' own target
  # lists (probe_dequant's pattern) because `B70_PREFILL_KERNELS` is defined in
  # tests/, which CMake adds after this directory.
  get_property(W4A8_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels
               PROPERTY BUILDSYSTEM_TARGETS)
  get_property(W4A8_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
               PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(probe_w4a8 kernel_pw4a8 ${W4A8_KERNELS} ${W4A8_PF_KERNELS})
endif()

# --- probe_w4a8_gsweep: the W4A8 group-size sweep (2026-09-23) ---------------
# Record: docs/probe-w4a8-2026-09-23.md, the appended sweep section.
#
# `probe_w4a8_gsweep.cl` is a PROBE kernel file with four entry points --
# `pw4a8s_g64`, `pw4a8s_g128`, `pw4a8s_g256`, `pw4a8s_gpc` -- one per scale
# stride. They differ ONLY in how many 64-k chunks accumulate in int32 before
# the fp32 rescale; the messages, the tile and the dpas are identical, and
# `pw4a8s_g64` is bit-identical in output to `probe_w4a8.cl`'s `pw4a8_gemm`
# (the harness checks all 71,303,168 words).
#
# No runtime path binds any of them, no production file is touched, and the
# production `GROUP 64` in the dequant kernels stays where it is: the sweep is
# a SPEED measurement made by synthesising coarser scale arrays on the host, not
# a re-quantisation of anything.
#
# 256 GRF for the same reason `pw4a8` needs it: eight int32 accumulators and
# eight fp32 accumulators live at once is 128 GRF of accumulator before a single
# operand register, and the per-channel arm keeps both live over all 80 chunks.
if(B70_PREFILL_ENABLED)
  add_ocloc_kernel(pw4a8s SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_w4a8_gsweep.cl
                   OPTIONS -cl-intel-256-GRF-per-thread)

  add_executable(probe_w4a8_gsweep probe_w4a8_gsweep.cc)
  target_include_directories(probe_w4a8_gsweep PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_w4a8_gsweep PRIVATE b70_prefill_host b70_runtime b70_loader
                        b70_l0 b70_model)
  b70_link_prefill(probe_w4a8_gsweep)
  b70_target_kernel_dir(probe_w4a8_gsweep)
  get_property(W4A8S_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels
               PROPERTY BUILDSYSTEM_TARGETS)
  get_property(W4A8S_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
               PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(probe_w4a8_gsweep kernel_pw4a8 kernel_pw4a8s ${W4A8S_KERNELS}
                   ${W4A8S_PF_KERNELS})
endif()

# --- probe_w8a8: i8 x i8 k32 DPAS, per-channel int8 weights (2026-09-23) -----
# Record: docs/probe-w4a8-2026-09-23.md, the appended i8 x i8 section.
#
# The one W4A8-family path the CPU accuracy study left standing: int4 g64
# weights requantised per 1024-column slab to per-channel int8 (`pw8_requant`),
# the i8 x i8 GEMM with the scales only in its epilogue (`pw8_gemm`), and the
# per-token activation quantiser with and without a 1024-block Hadamard
# (`pw8_quant`, `pw8_quant_had`). Speed probe; no runtime path binds any of it.
# 256 GRF: 64 int32 accumulator registers plus two 16-dword B fragments per lane.
if(B70_PREFILL_ENABLED)
  add_ocloc_kernel(pw8a8 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_w8a8.cl
                   OPTIONS -cl-intel-256-GRF-per-thread)
  # The same source again at the default 128 GRF: the helpers (requant,
  # quantisers) do not need 256, and 256 halves the threads per EU.
  add_ocloc_kernel(pw8a8_g128 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_w8a8.cl)

  add_executable(probe_w8a8 probe_w8a8.cc)
  target_include_directories(probe_w8a8 PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_w8a8 PRIVATE b70_prefill_host b70_runtime b70_loader
                        b70_l0 b70_model)
  b70_link_prefill(probe_w8a8)
  b70_target_kernel_dir(probe_w8a8)
  get_property(W8A8_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels
               PROPERTY BUILDSYSTEM_TARGETS)
  get_property(W8A8_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
               PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(probe_w8a8 kernel_pw8a8 kernel_pw8a8_g128 ${W8A8_KERNELS} ${W8A8_PF_KERNELS})
endif()

# --- probe_flash_attn: spec 6 P1, the flash-attention tile probe (2026-09-25) ---
# Plan: docs/superpowers/plans/2026-09-25-spec6a-flash-attn-baseline-and-probe.md.
# Record: docs/probe-flash-attn-2026-09-25.md.
#
# The control is the production composed path (`runtime::prefill::attn_chunk`, L0
# backend) on the same inputs; every arm is checked against an fp64 CPU reference
# before it is timed. Probe-only: no runtime path binds `pfa`.
if(B70_PREFILL_ENABLED)
  add_executable(probe_flash_attn probe_flash_attn.cc)
  target_include_directories(probe_flash_attn PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_flash_attn PRIVATE b70_prefill_host b70_runtime b70_loader
                        b70_l0 b70_model pthread)
  b70_link_prefill(probe_flash_attn)
  b70_target_kernel_dir(probe_flash_attn)
  get_property(PFA_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels
               PROPERTY BUILDSYSTEM_TARGETS)
  get_property(PFA_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
               PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(probe_flash_attn ${PFA_KERNELS} ${PFA_PF_KERNELS})
endif()
