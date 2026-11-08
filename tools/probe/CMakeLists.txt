# Every probe here is g++ / Level Zero. The one icpx probe (oneDNN's W4A16 matmul, a
# reference target) has its own stand-alone build: tools/probe/onednn/CMakeLists.txt.
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

# The dequant-overlap lever (spec 2, plan 6c Task 4): pf_dequant_tile on the L0
# immediate list. `runtime::prefill::Context` lives in b70_prefill_host.
add_executable(probe_dequant_overlap probe_dequant_overlap.cc)
target_include_directories(probe_dequant_overlap PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_dequant_overlap PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(probe_dequant_overlap)
add_dependencies(probe_dequant_overlap kernel_pf_dequant_tile_K5120_N34816_L0_T0)

# The L2-resident slab lever. Same shape as the overlap probe (gate‖up,
# K=5120, N=34816, M=2048), so it links the same way.
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
b70_target_kernel_dir(probe_dequant_slab)
add_dependencies(probe_dequant_slab kernel_pf_dequant_tile_K5120_N34816_L0_T0
                 ${PROBE_SLAB_TARGETS})

# P-A (spec 2 addendum, docs/08-decode-vs-prefill.md ("How prefill got here")):
# our own OpenCL C bf16 DPAS GEMM on the Level Zero list -- the go/no-go
# candidate for dropping sycl-tla from the prefill path. AOT at 256 GRF is
# MANDATORY (§A4.8): P2 measured a 33x loss without it, and zero spill does
# not mean good code. `pf_gemm_bf16.cl` is a probe-only kernel: no runtime
# path binds it and no existing kernel is touched.
add_ocloc_kernel(pf_gemm_bf16 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemm_bf16.cl
                 OPTIONS -cl-intel-256-GRF-per-thread)

add_executable(probe_pf_gemm probe_pf_gemm.cc)
target_include_directories(probe_pf_gemm PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_pf_gemm PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(probe_pf_gemm)
add_dependencies(probe_pf_gemm kernel_pf_gemm_bf16)

# P-B (docs/08-decode-vs-prefill.md ("How prefill got here")): slab dequant +
# P-A's GEMM on ONE in-order L0 list, zero host waits. Runs only because
# P-A measured >= 128 (committed: 156.53 TFLOP/s). `pf_gemm_bf16_slab` is a
# second entry point in the SAME pf_gemm_bf16.cl / kernel_pf_gemm_bf16
# binary as P-A's already-gated kernel -- pf_gemm_bf16 itself is untouched.
add_executable(probe_pf_gemm_slab probe_pf_gemm_slab.cc)
target_include_directories(probe_pf_gemm_slab PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_pf_gemm_slab PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(probe_pf_gemm_slab)
add_dependencies(probe_pf_gemm_slab kernel_pf_dequant_tile_K5120_N34816_L0_T0
                 kernel_pf_gemm_bf16 ${PROBE_SLAB_TARGETS})

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
# the S layout 0 picks for itself in docs/probes/probe-gemv-2026-08-24.md -- the
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
# Pre-registration: docs/superpowers/specs/2026-09-22-fused-dequant-gemm-probe-design.md.
# `probe_fused_dequant_gemm.cl` is a PROBE kernel: no runtime path binds
# `pfd_gemm`, and neither `src/kernels/prefill/pf_gemm.cl` nor
# `pf_dequant_slab.cl` is touched -- the probe's CONTROL runs those two
# production binaries themselves, which is what makes the memcmp a gate rather
# than a paraphrase. 256 GRF is mandatory for anything sharing pf_gemm's tile
# (P2 measured a 33x loss without it) and is carried on the compile line, the
# only place an AOT build honours it.
set(PROBE_FUSED_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_fused_dequant_gemm.cl)
add_ocloc_kernel(pfd_gemm_v1 SOURCE ${PROBE_FUSED_CL} DEFINES FUSED=1
                 OPTIONS -cl-intel-256-GRF-per-thread)
add_ocloc_kernel(pfd_gemm_v2 SOURCE ${PROBE_FUSED_CL} DEFINES FUSED=2
                 OPTIONS -cl-intel-256-GRF-per-thread)

add_executable(probe_fused_dequant_gemm probe_fused_dequant_gemm.cc)
target_include_directories(probe_fused_dequant_gemm PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_fused_dequant_gemm PRIVATE b70_prefill_host b70_l0)
b70_target_kernel_dir(probe_fused_dequant_gemm)
add_dependencies(probe_fused_dequant_gemm kernel_pfd_gemm_v1 kernel_pfd_gemm_v2
                 kernel_pf_gemm_T0 kernel_pf_dequant_slab_K5120_N34816_L0)

# --- probe_dpas_rate: the per-type DPAS issue rate (2026-09-22) -------------
# Record: docs/probes/probe-dpas-rates-2026-09-22.md. `probe_dpas_rate.cl` is a PROBE
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
# Pre-registration: docs/superpowers/specs/2026-09-22-w4a8-dpas-probe-design.md.
# Record: docs/probes/probe-w4a8-2026-09-23.md.
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
add_ocloc_kernel(pw4a8 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_w4a8.cl
                 OPTIONS -cl-intel-256-GRF-per-thread)

add_executable(probe_w4a8 probe_w4a8.cc)
target_include_directories(probe_w4a8 PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_w4a8 PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0
                      b70_model)
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

# --- probe_w4a8_gsweep: the W4A8 group-size sweep (2026-09-23) ---------------
# Record: docs/probes/probe-w4a8-2026-09-23.md, the appended sweep section.
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
add_ocloc_kernel(pw4a8s SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_w4a8_gsweep.cl
                 OPTIONS -cl-intel-256-GRF-per-thread)

add_executable(probe_w4a8_gsweep probe_w4a8_gsweep.cc)
target_include_directories(probe_w4a8_gsweep PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_w4a8_gsweep PRIVATE b70_prefill_host b70_runtime b70_loader
                      b70_l0 b70_model)
b70_target_kernel_dir(probe_w4a8_gsweep)
get_property(W4A8S_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels
             PROPERTY BUILDSYSTEM_TARGETS)
get_property(W4A8S_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
             PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_w4a8_gsweep kernel_pw4a8 kernel_pw4a8s ${W4A8S_KERNELS}
                 ${W4A8S_PF_KERNELS})

# --- probe_w8a8: i8 x i8 k32 DPAS, per-channel int8 weights (2026-09-23) -----
# Record: docs/probes/probe-w4a8-2026-09-23.md, the appended i8 x i8 section.
#
# The one W4A8-family path the CPU accuracy study left standing: int4 g64
# weights requantised per 1024-column slab to per-channel int8 (`pw8_requant`),
# the i8 x i8 GEMM with the scales only in its epilogue (`pw8_gemm`), and the
# per-token activation quantiser with and without a 1024-block Hadamard
# (`pw8_quant`, `pw8_quant_had`). Speed probe; no runtime path binds any of it.
# 256 GRF: 64 int32 accumulator registers plus two 16-dword B fragments per lane.
add_ocloc_kernel(pw8a8 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_w8a8.cl
                 OPTIONS -cl-intel-256-GRF-per-thread)
# The same source again at the default 128 GRF: the helpers (requant,
# quantisers) do not need 256, and 256 halves the threads per EU.
add_ocloc_kernel(pw8a8_g128 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_w8a8.cl)

add_executable(probe_w8a8 probe_w8a8.cc)
target_include_directories(probe_w8a8 PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_w8a8 PRIVATE b70_prefill_host b70_runtime b70_loader
                      b70_l0 b70_model)
b70_target_kernel_dir(probe_w8a8)
get_property(W8A8_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels
             PROPERTY BUILDSYSTEM_TARGETS)
get_property(W8A8_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
             PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_w8a8 kernel_pw8a8 kernel_pw8a8_g128 ${W8A8_KERNELS} ${W8A8_PF_KERNELS})

# --- probe_flash_attn: spec 6 P1, the flash-attention tile probe (2026-09-25) ---
# Plan: docs/superpowers/plans/2026-09-25-spec6a-flash-attn-baseline-and-probe.md.
# Record: docs/probes/probe-flash-attn-2026-09-25.md.
#
# The control is the production composed path (`runtime::prefill::attn_chunk`, L0
# backend) on the same inputs; every arm is checked against an fp64 CPU reference
# before it is timed. Probe-only: no runtime path binds `pfa`.
# One binary per arm: KT (KV positions per tile), RPW (query rows per work-group per
# head), HPW (q-heads per work-group), QREG (Q held in registers). 256 GRF: the O
# accumulator alone is 128 GRF per sub-group. The list and `kArms` in
# probe_flash_attn.cc are two lists nothing checks against each other; a missing
# binary is reported by name.
set(PFA_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_flash_attn.cl)
set(PFA_TARGETS "")
foreach(arm "32 16 6 1" "32 32 6 1" "64 16 6 1" "32 16 3 1" "32 16 1 1" "32 16 6 0" "64 16 6 0" "32 32 3 1")
  separate_arguments(f UNIX_COMMAND "${arm}")
  list(GET f 0 KT)
  list(GET f 1 R)
  list(GET f 2 H)
  list(GET f 3 QR)
  add_ocloc_kernel(pfa_KT${KT}_R${R}_H${H}_Q${QR} SOURCE ${PFA_CL}
                   DEFINES KT=${KT} RPW=${R} HPW=${H} QREG=${QR} OPTIONS -cl-intel-256-GRF-per-thread)
  list(APPEND PFA_TARGETS kernel_pfa_KT${KT}_R${R}_H${H}_Q${QR})
endforeach()

add_executable(probe_flash_attn probe_flash_attn.cc)
target_include_directories(probe_flash_attn PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_flash_attn PRIVATE b70_prefill_host b70_runtime b70_loader
                      b70_l0 b70_model pthread)
b70_target_kernel_dir(probe_flash_attn)
get_property(PFA_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels
             PROPERTY BUILDSYSTEM_TARGETS)
get_property(PFA_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill
             PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_flash_attn ${PFA_TARGETS} ${PFA_KERNELS} ${PFA_PF_KERNELS})

# --- spec 7 P0 (2026-09-27): pinned host copies and the tail-prefill floor ---
# Record: docs/probes/probe-prefix-cache-2026-09-27.md.
add_executable(probe_host_copy probe_host_copy.cc)
target_include_directories(probe_host_copy PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_host_copy PRIVATE b70_l0)
add_executable(probe_tail probe_tail.cc)
target_include_directories(probe_tail PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_tail PRIVATE b70_prefill_host b70_runtime b70_loader b70_model b70_l0 pthread)
b70_target_kernel_dir(probe_tail)
get_property(PT_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
get_property(PT_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_tail ${PT_KERNELS} ${PT_PF_KERNELS})
# --- spec 7 S3 (plan 7b Task 3): the write-through cost of block snapshots ---
# Record: docs/probes/probe-prefix-cache-2026-09-27.md, "Write-through cost, S3".
add_executable(probe_writethrough probe_writethrough.cc)
target_include_directories(probe_writethrough PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_writethrough PRIVATE b70_prefill_host b70_runtime b70_loader b70_model b70_l0 pthread)
b70_target_kernel_dir(probe_writethrough)
add_dependencies(probe_writethrough ${PT_KERNELS} ${PT_PF_KERNELS})
# --- spec 7 S1/S2 (plan 7c Task 4): time to first token at 60k through PrefixSession ---
# Record: docs/probes/probe-prefix-cache-2026-09-27.md, "S1 and S2, measured".
if(TARGET b70_tokenizer)
  add_executable(probe_prefix_s12 probe_prefix_s12.cc)
  target_include_directories(probe_prefix_s12 PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(probe_prefix_s12 PRIVATE b70_server b70_tokenizer b70_prefill_host b70_runtime
    b70_loader b70_model b70_l0 pthread)
  b70_target_kernel_dir(probe_prefix_s12)
  add_dependencies(probe_prefix_s12 ${PT_KERNELS} ${PT_PF_KERNELS})
endif()
# spec 7 C3: the tie-aware rule at a replay divergence (tools/prefix/replay_ab.sh)
add_executable(probe_tie_judge probe_tie_judge.cc)
target_include_directories(probe_tie_judge PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_tie_judge PRIVATE b70_prefill_host b70_runtime b70_loader b70_model b70_l0 pthread)
b70_target_kernel_dir(probe_tie_judge)
add_dependencies(probe_tie_judge ${PT_KERNELS} ${PT_PF_KERNELS})
# spec 28: the engine's plain decode row + per-layer taps at a forced-context position (tools/oracle/row_taps_ref.py)
add_executable(probe_row_taps probe_row_taps.cc)
target_include_directories(probe_row_taps PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_row_taps PRIVATE b70_prefill_host b70_runtime b70_loader b70_model b70_l0 pthread)
b70_target_kernel_dir(probe_row_taps)
add_dependencies(probe_row_taps ${PT_KERNELS} ${PT_PF_KERNELS})
# --- probe_verify_m: spec 8 P0, the MTP verify step at M = 1..4 (2026-09-27) ---
# Plan: docs/superpowers/plans/2026-09-27-spec8a-mtp-probe.md, Task 3.
# Record: docs/probes/probe-mtp-2026-09-27.md. Needs B70_MTP (the default: the M = 2..4
# decode-list variants and the head's bf16 GEMV shapes, src/kernels/CMakeLists.txt).
if(B70_MTP)
  add_executable(probe_verify_m probe_verify_m.cc)
  target_include_directories(probe_verify_m PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(probe_verify_m PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  b70_target_kernel_dir(probe_verify_m)
  get_property(VM_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
  get_property(VM_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(probe_verify_m ${VM_KERNELS} ${VM_PF_KERNELS})
endif()
# --- probe_mtp_steps: spec 8 plan 8b, the production MTP lists' wall time per call ---
# Record: docs/probes/probe-mtp-2026-09-27.md, "Engine". Needs B70_MTP (the default).
if(B70_MTP)
  add_executable(probe_mtp_steps probe_mtp_steps.cc)
  target_include_directories(probe_mtp_steps PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_mtp_steps PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  get_property(MS_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
  get_property(MS_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(probe_mtp_steps ${MS_KERNELS} ${MS_PF_KERNELS})
endif()
# --- probe_batch_step: spec 13a, rows of different sequences in one decode step ---
# Plan: docs/superpowers/plans/2026-09-28-spec13a-batching-probe.md, Task 2. Record: docs/probes/probe-batching-2026-10-23.md.
# Only with -DB70_BATCH_PROBE=ON (src/kernels' row builds; b70_runtime then compiles runtime::build_rows).
if(B70_BATCH_PROBE)
  add_executable(probe_batch_step probe_batch_step.cc)
  target_include_directories(probe_batch_step PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(probe_batch_step PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
  get_property(BS_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
  get_property(BS_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(probe_batch_step ${BS_KERNELS} ${BS_PF_KERNELS})
endif()
# --- probe_draft_cost: spec 19a Task 4, one DFlash2 block's GEMVs (int8 / bf16 / int4) ---
# Plan: docs/superpowers/plans/2026-10-05-spec19a-dflash-p0.md, Task 4. Random weights of
# the drafter's shapes (draft_cost_shapes.h); the binaries are B70_VERIFY_M8's
# (src/kernels/CMakeLists.txt), which also gives probe_mtp_steps its verify M = 5..8 arms.
if(B70_VERIFY_M8 AND B70_MTP)
  add_executable(probe_draft_cost probe_draft_cost.cc)
  target_include_directories(probe_draft_cost PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(probe_draft_cost PRIVATE b70_l0 b70_bolt_model)
  b70_target_kernel_dir(probe_draft_cost)
  get_property(DC_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(probe_draft_cost ${DC_KERNELS})
endif()

# --- probe_decode_attn: spec 10 P0/P1, decode attention at depth (2026-09-28) ---
# Plan: docs/superpowers/plans/2026-09-28-spec10a-decode-attn-probe.md.
# Record: docs/probes/probe-decode-attn-2026-09-28.md. Arm 0 (`pda_v1_M<M>`) is
# src/kernels/attn.cl ITSELF at max_len 131072 and the probe's M (no production build
# compiles M = 4 at 131072); every other arm is probe_decode_attn.cl at one tag
# P<ppw>_L<load>_D<dot>_F<prefetch>_R<reduce>. This list and the harness's --arms are two
# lists nothing checks against each other: a missing binary is named and skipped.
set(PDA_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_decode_attn.cl)
set(PDA_TAGS
    P64_L0_D0_F0_R0 P256_L0_D0_F0_R0 P512_L0_D0_F0_R0 P1024_L0_D0_F0_R0 P2048_L0_D0_F0_R0
    P0_L0_D0_F0_R0
    P64_L1_D0_F0_R0 P64_L2_D0_F0_R0 P64_L2_D1_F0_R0
    P256_L1_D0_F0_R0 P256_L2_D0_F0_R0 P256_L2_D1_F0_R0 P256_L2_D1_F1_R0 P256_L2_D1_F0_R1
    P512_L1_D0_F0_R0 P512_L2_D0_F0_R0 P512_L2_D1_F0_R0 P512_L2_D1_F1_R0 P512_L2_D1_F0_R1
    P1024_L2_D0_F0_R0 P1024_L2_D1_F0_R0 P1024_L2_D1_F1_R0 P1024_L2_D1_F0_R1
    P0_L1_D0_F0_R0 P0_L2_D0_F0_R0 P0_L2_D1_F0_R0 P0_L2_D1_F1_R0 P0_L2_D1_F0_R1
    P0_L2_D1_F1_R1
    P0_L2_D1_F0_R0_Q1_G0_E0 P0_L2_D1_F1_R0_Q1_G0_E0 P0_L2_D1_F0_R0_Q1_G1_E0
    P0_L2_D1_F1_R0_Q1_G1_E0 P0_L2_D1_F0_R0_Q1_G0_E1 P0_L2_D1_F1_R0_Q1_G1_E1
    P1024_L2_D1_F0_R0_Q1_G0_E0 P0_L2_D1_F1_R0_Q1_G0_E1)
set(PDA_TARGETS)
foreach(MM 1 4)
  add_ocloc_kernel(pda_v1_M${MM} SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/attn.cl
                   DEFINES CTRL_POS=0 CTRL_NACT=1 QNORM_OFF=0 KNORM_OFF=256 ATTN_BLOCK=64
                           M=${MM} MAXLEN=131072)
  list(APPEND PDA_TARGETS kernel_pda_v1_M${MM})
  foreach(tag IN LISTS PDA_TAGS)
    string(REGEX MATCH "^P([0-9]+)_L([0-9])_D([0-9])_F([0-9])_R([0-9])(_Q([0-9])_G([0-9])_E([0-9]))?$"
           _m ${tag})
    set(_q 0)
    set(_g 0)
    set(_e 0)
    if(CMAKE_MATCH_6)
      set(_q ${CMAKE_MATCH_7})
      set(_g ${CMAKE_MATCH_8})
      set(_e ${CMAKE_MATCH_9})
    endif()
    set(_opts)
    if(_g)
      set(_opts OPTIONS -cl-intel-256-GRF-per-thread)
    endif()
    add_ocloc_kernel(pda_${tag}_M${MM} SOURCE ${PDA_CL}
                     DEFINES M=${MM} MAXLEN=131072 PPW=${CMAKE_MATCH_1} LOAD=${CMAKE_MATCH_2}
                             DOT=${CMAKE_MATCH_3} PREFETCH=${CMAKE_MATCH_4} REDUCE=${CMAKE_MATCH_5}
                             QBLK=${_q} EXP2=${_e}
                     ${_opts})
    list(APPEND PDA_TARGETS kernel_pda_${tag}_M${MM})
  endforeach()
endforeach()
add_executable(probe_decode_attn probe_decode_attn.cc)
target_include_directories(probe_decode_attn PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_decode_attn PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
b70_target_kernel_dir(probe_decode_attn)
get_property(PDA_PROD_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_decode_attn ${PDA_TARGETS} ${PDA_PROD_KERNELS})
get_property(PDA_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_decode_attn ${PDA_PF_KERNELS})

# --- probe_lm_head_l1: spec 9 plan 9b, L1/L2 on the card (int8 head vs bf16 head, same
# hidden, golden prompts teacher-forced). Record: docs/BENCHMARKS.md "int8 lm_head (spec 9)".
add_executable(probe_lm_head_l1 probe_lm_head_l1.cc)
target_include_directories(probe_lm_head_l1 PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(probe_lm_head_l1 PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
b70_target_kernel_dir(probe_lm_head_l1)
get_property(LH_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_lm_head_l1 ${LH_KERNELS})

# --- spec 22 P0 (2026-10-10): the expert-offload probe ---
# probe_offload (plan 22a Task 2): copy-engine rates, zero-copy expert reads from pinned host USM mixed with VRAM,
# replayed visibility, host RAM under load, pinned capacity, the alias check. Its expert kernels are 21c's q4_moe.cl
# token for token with the two address macros as a pointer-table read (probe_offload_moe.cl, generated by
# mk_offload_kernel.py, drift-checked by probe_offload_kernel_drift_test) at the _SH4 binary's defines; arm A binds
# 21c's own q4_moe_M1_E512_T10_D2560_I640_SH4. Record: docs/probes/probe-offload-<date>.md.
if(B70_Q4EXP)
  add_ocloc_kernel(po_moe_M1_E512_T10_D2560_I640_SH4 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_offload_moe.cl
                   DEFINES M=1 MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=640 UP_KS=4 DN_KS=2 SHARED_BF16=0)
  add_ocloc_kernel(po_kernels SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_offload.cl
                   DEFINES GU_BYTES=1740800 DN_BYTES=870400 MOE_E=512 MOE_K=10 RW=32 PO_CHUNKS=32)
  add_executable(probe_offload probe_offload.cc)
  target_include_directories(probe_offload PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(probe_offload PRIVATE b70_l0 b70_qwen4exp_model pthread)
  b70_target_kernel_dir(probe_offload)
  add_dependencies(probe_offload kernel_po_moe_M1_E512_T10_D2560_I640_SH4 kernel_po_kernels
                   kernel_q4_moe_M1_E512_T10_D2560_I640_SH4)
endif()

# --- dflash_d1: spec 19 §6 D1 (plan 19b Task 3), the engine's half ---
# The DFlash drafter beside the loaded target drafts at teacher-forced anchors of tap dumps; tools/oracle/dflash_d1.py
# holds every draft to tools/oracle/dflash_ref.py on the same taps. Box queue row 54 (stage r54.d1).
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_executable(dflash_d1 dflash_d1.cc)
  target_include_directories(dflash_d1 PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(dflash_d1 PRIVATE b70_dflash_runtime b70_bolt_loader b70_loader b70_l0)   # 26b: + Bolt
  b70_target_kernel_dir(dflash_d1)
  get_property(D1_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
  add_dependencies(dflash_d1 ${D1_KERNELS})
endif()

# --- probe_bolt_step: spec 25c P0 (plan 25c Task 1) ---
# Nemotron 3.5 Bolt's decode step from EXISTING kernels at its shapes (tools/probe/bolt_step_shapes.h: 653 / 533 /
# GEMV-only / noop lists and the SSM state's read-modify-write floor, interleaved; --sweep-s times every int4 shape at
# every S and layout). Box queue row 60, stage r60.p0. Plain g++ / Level Zero; the binaries are src/kernels/
# CMakeLists.txt's spec 25c block (and gdn_step_M1, noop - built above).
if(B70_BOLT)
  add_executable(probe_bolt_step probe_bolt_step.cc)
  target_include_directories(probe_bolt_step PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(probe_bolt_step PRIVATE b70_l0 b70_bolt_model)
  b70_target_kernel_dir(probe_bolt_step)
  set(_nb_p0_kernels kernel_noop kernel_gdn_step_M1 kernel_moe_M1_E64_T8_D1024_I2048 kernel_mg_attn_M1_T64_Q32KV2_F
    kernel_lfm_res_norm_M1_K4096_W8_S1 kernel_prep_res_fold_M1_K4096_SP0_G16 kernel_kol_norm_M1_K4096_G16_W16
    kernel_gemv_i8w_M1_K4096_N131072 kernel_nb_argmax_stage1_M1_V131072 kernel_nb_argmax_stage2_V131072
    kernel_nb_state_rmw_F32 kernel_nb_state_rmw_B16)
  foreach(E 240 304 368 432 496)
    list(APPEND _nb_p0_kernels kernel_gemv_bf16_M1_K4096_N${E}_C16_S16)
  endforeach()
  foreach(L 0 1)
    foreach(S 1 2 4 8 16)
      list(APPEND _nb_p0_kernels kernel_gemv_M1_K4096_N18560_S${S}_L${L} kernel_gemv_M1_K8192_N4096_S${S}_L${L}
        kernel_gemv_M1_K4096_N4608_S${S}_L${L} kernel_gemv_M1_K4096_N4096_S${S}_L${L}
        kernel_gemv_M1_K4096_N6400_S${S}_L${L} kernel_gemv_M1_K1024_N4096_S${S}_L${L})
      if(S LESS_EQUAL 4)
        list(APPEND _nb_p0_kernels kernel_gemv_M1_K5376_N4096_S${S}_L${L})
      endif()
    endforeach()
  endforeach()
  add_dependencies(probe_bolt_step ${_nb_p0_kernels})
endif()

# --- probe_small_m: spec 28 P0 (plan 28a), small-M DPAS stand-ins against gemv ---------
# Plan: docs/superpowers/plans/2026-10-27-spec28a-small-m-probe.md. Record:
# docs/probes/probe-small-m-gemm-<date>.md. `probe_small_m.cl` is a PROBE kernel: no runtime path
# binds `psm` / `psm_quant`, no production file is touched. One binary per knob set, named
#   psm_<t><o>[_wg<NSGN>x<KSL>][_mg][_pf<G>][_g256]_M<M>_K<K>_N<N>_S<S>_L<L>
# t = b (bf16 DPAS) | i (int DPAS on int8 rows), o = B (weights on B) | A (weights on A);
# the kVariants logic in probe_small_m.cc builds the same names (a missing one is reported
# by name at l0::Module). The M = 16 controls are today's gemv / gemv_i8w / gemv_bf16
# sources at M = 16 with the production cells' defines (probe-only names, psm_ctl_*).
set(PSM_CL ${CMAKE_CURRENT_SOURCE_DIR}/probe_small_m.cl)
set(PSM_TARGETS "")
# K N S_prod L BLK2D(wB, layout 0, as the production cell)   - out_proj and o_proj share 6144x5120
set(PSM_SHAPES "5120 16384 1 1 0" "6144 5120 4 0 1" "5120 14336 2 0 1" "5120 34816 8 0 0"
               "17408 5120 4 0 1" "5120 248320 1 2 0")
function(psm_variant T O M K N S L B2D)   # ARGN: extra tag, then extra defines / OPTIONS
  cmake_parse_arguments(P "" "TAG" "DEFS;OPTS" ${ARGN})
  if(T STREQUAL "b")
    set(defs TYPE=1)
  else()
    set(defs TYPE=2)
  endif()
  if(O STREQUAL "B")
    list(APPEND defs ORIENT=0)
    if(B2D)
      list(APPEND defs BLK2D=1)
    endif()
  else()
    list(APPEND defs ORIENT=1)
  endif()
  set(name psm_${T}${O}${P_TAG}_M${M}_K${K}_N${N}_S${S}_L${L})
  if(NOT TARGET kernel_${name})
    if(P_OPTS)
      add_ocloc_kernel(${name} SOURCE ${PSM_CL} DEFINES M=${M} K=${K} N=${N} S=${S} LAYOUT=${L} ${defs} ${P_DEFS}
                       OPTIONS ${P_OPTS})
    else()
      add_ocloc_kernel(${name} SOURCE ${PSM_CL} DEFINES M=${M} K=${K} N=${N} S=${S} LAYOUT=${L} ${defs} ${P_DEFS})
    endif()
    list(APPEND PSM_TARGETS kernel_${name})
    set(PSM_TARGETS ${PSM_TARGETS} PARENT_SCOPE)
  endif()
endfunction()
foreach(row IN LISTS PSM_SHAPES)
  separate_arguments(f UNIX_COMMAND "${row}")
  list(GET f 0 K)
  list(GET f 1 N)
  list(GET f 2 SP)
  list(GET f 3 L)
  list(GET f 4 B2D)
  # the sweep: both types, both orientations, M = 1 / 2 / 4 / 8 / 16 (property R: + 3, 5, 6, 7 for bf16)
  foreach(O B A)
    foreach(M 1 2 4 8 16)
      psm_variant(b ${O} ${M} ${K} ${N} ${SP} ${L} ${B2D})
      psm_variant(i ${O} ${M} ${K} ${N} ${SP} ${L} ${B2D})
    endforeach()
    foreach(M 3 5 6 7)
      psm_variant(b ${O} ${M} ${K} ${N} ${SP} ${L} ${B2D})
    endforeach()
    # the oneDNN levers at M = 8, bf16, one at a time (spec 28 §2.1)
    psm_variant(b ${O} 8 ${K} ${N} 1 ${L} 0 TAG _wg2x8 DEFS SPLITK=1 NSGN=2 KSL=8)
    psm_variant(b ${O} 8 ${K} ${N} 1 ${L} 0 TAG _wg8x4 DEFS SPLITK=1 NSGN=8 KSL=4)
    psm_variant(b ${O} 8 ${K} ${N} ${SP} ${L} ${B2D} TAG _mg DEFS DEQ=1)
    foreach(PG 1 2 3)
      psm_variant(b ${O} 8 ${K} ${N} ${SP} ${L} ${B2D} TAG _pf${PG} DEFS PF_G=${PG})
    endforeach()
    psm_variant(b ${O} 8 ${K} ${N} ${SP} ${L} ${B2D} TAG _g256 OPTS -cl-intel-256-GRF-per-thread)
    # one in-work-group shape for every M (property R) - its cost at 2 / 4 / 16, and 8x4 at 16
    foreach(M 2 4 16)
      psm_variant(b ${O} ${M} ${K} ${N} 1 ${L} 0 TAG _wg2x8 DEFS SPLITK=1 NSGN=2 KSL=8)
    endforeach()
    psm_variant(b ${O} 16 ${K} ${N} 1 ${L} 0 TAG _wg8x4 DEFS SPLITK=1 NSGN=8 KSL=4)
    # the S sweep at M = 8 (grid split-K; the head has none)
    if(NOT L EQUAL 2)
      foreach(S 1 2 4 8 16)
        psm_variant(b ${O} 8 ${K} ${N} ${S} ${L} ${B2D})
      endforeach()
    endif()
  endforeach()
endforeach()
# The M = 16 controls: today's kernels, probe-only builds (gemv.cl / gemv_i8w.cl / gemv_bf16.cl
# are generic in M; production stops at 8). Defines copied from src/kernels' production cells.
set(PSM_GEMV_CL ${CMAKE_SOURCE_DIR}/src/kernels/gemv.cl)
add_ocloc_kernel(psm_ctl_gemv_M16_K5120_N16384_S1_L1 SOURCE ${PSM_GEMV_CL}
                 DEFINES M=16 K=5120 N=16384 S=1 LAYOUT=1)
add_ocloc_kernel(psm_ctl_gemv_M16_K6144_N5120_S4_L0 SOURCE ${PSM_GEMV_CL}
                 DEFINES M=16 K=6144 N=5120 S=4 LAYOUT=0 GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
add_ocloc_kernel(psm_ctl_gemv_M16_K5120_N14336_S2_L0 SOURCE ${PSM_GEMV_CL}
                 DEFINES M=16 K=5120 N=14336 S=2 LAYOUT=0 GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
add_ocloc_kernel(psm_ctl_gemv_M16_K5120_N34816_S8_L0 SOURCE ${PSM_GEMV_CL}
                 DEFINES M=16 K=5120 N=34816 S=8 LAYOUT=0 GEMV_DEQ_SHIFT=1)
add_ocloc_kernel(psm_ctl_gemv_M16_K17408_N5120_S4_L0 SOURCE ${PSM_GEMV_CL}
                 DEFINES M=16 K=17408 N=5120 S=4 LAYOUT=0 GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
add_ocloc_kernel(psm_ctl_gemv_i8w_M16_K5120_N248320 SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/gemv_i8w.cl
                 DEFINES M=16 K=5120 N=248320)
add_ocloc_kernel(psm_ctl_gemv_bf16_M16_K5120_N128_C16_S16 SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/gemv_bf16.cl
                 DEFINES M=16 K=5120 N=128 COLS_PER_WG=16 KSPLIT=16)
list(APPEND PSM_TARGETS kernel_psm_ctl_gemv_M16_K5120_N16384_S1_L1 kernel_psm_ctl_gemv_M16_K6144_N5120_S4_L0
     kernel_psm_ctl_gemv_M16_K5120_N14336_S2_L0 kernel_psm_ctl_gemv_M16_K5120_N34816_S8_L0
     kernel_psm_ctl_gemv_M16_K17408_N5120_S4_L0 kernel_psm_ctl_gemv_i8w_M16_K5120_N248320
     kernel_psm_ctl_gemv_bf16_M16_K5120_N128_C16_S16)
# Spec 28b (plan 28b): the lever alternates of the production gemm_sm rows (tests/kernels/gemm_sm_levers.h, the
# same order) at M = 2..8 and 16, from src/kernels/gemm_sm.cl (add_gemm_sm_variant / gemm_sm_rows, src/kernels/
# CMakeLists.txt) - probe_small_m prod times them against the production rows, gemm_sm_test --levers holds them to
# N2 / N3. Probe-only names: nothing binds one unless kernels::kSmallMTuning adopts it (src/kernels builds it then).
set(GEMM_SM_TARGETS "")
set(GEMM_SM_LEVERS "5120 16384 1 1" "5120 16384 1 1 MG G256" "6144 5120 4 0 B2D" "5120 14336 2 0 B2D" "5120 34816 8 0"
                   "5120 34816 8 0 MG" "17408 5120 4 0 B2D" "5120 248320 1 2")
set(_psm_lever_m 2 3 4 5 6 7 8 16)
gemm_sm_rows(GEMM_SM_LEVERS _psm_lever_m)
list(APPEND PSM_TARGETS ${GEMM_SM_TARGETS})
add_custom_target(psm_kernels DEPENDS ${PSM_TARGETS})

add_executable(probe_small_m probe_small_m.cc)
target_include_directories(probe_small_m PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_small_m PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model pthread)
b70_target_kernel_dir(probe_small_m)
get_property(PSM_SRC_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
get_property(PSM_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_small_m psm_kernels ${PSM_SRC_KERNELS} ${PSM_PF_KERNELS})

# --- probe_verify_attn: spec 28c (plan 28c Task 3), the verify attention - v2 against attn_verify_dpas ---------
# Plan: docs/superpowers/plans/2026-10-31-spec28c-verify-attn-build.md. Record: docs/probes/probe-verify-attn-<date>.md.
# `pvd_<tag>_M<M>[_KV8]` is src/kernels/attn_verify_dpas.cl itself at one lever tag
# K<KT>_P<g|h>_S<KS>_D<DSPLIT>_Q<QSPLIT>_G<128|256>[_P<PSPLIT>] (G: the ocloc 256-GRF option or not; PSPLIT 0 when the
# suffix is absent - the plan 28c sweep's kernel, P in bf16), M = 2..8 and 16, both KV
# forms; a (tag, M) whose work-group exceeds the GRF mode's lanes (512 at 256 GRF, 1024 at 128) or whose SLM exceeds
# 128 KB is not built (the kernel's own #error for the latter). `pvd_v2kv8_M<M>` (M = 5..8) is the int8 v2 control
# where production stops at M = 4: kv8.cl's KV8_DEC under BATCH_ROWS (spec 13a's row build, one slot = the plain
# decode), no source edit. The tag list and the harness's --arms are two lists nothing checks against each other: a
# missing binary is named and skipped. No production binary binds any of these.
set(PVD_CL ${CMAKE_SOURCE_DIR}/src/kernels/attn_verify_dpas.cl)
set(PVD_TAGS
    K32_Pg_S4_D256_Q1_G256                                               # the provisional setting
    K64_Pg_S4_D256_Q1_G256 K32_Ph_S4_D256_Q1_G256                        # KT, PACK
    K32_Pg_S1_D256_Q1_G256 K32_Pg_S2_D256_Q1_G256 K32_Pg_S8_D256_Q1_G128 # KS
    K32_Pg_S2_D128_Q1_G256 K32_Pg_S4_D128_Q1_G128                        # DSPLIT
    K32_Pg_S4_D256_Q0_G256                                               # QSPLIT
    K32_Pg_S4_D256_Q1_G128                                               # GRF
    K32_Pg_S4_D256_Q1_G256_P1)                                           # the production line since 2026-11-04
set(PVD_TARGETS)
foreach(tag IN LISTS PVD_TAGS)
  string(REGEX MATCH "^K([0-9]+)_P([gh])_S([0-9]+)_D([0-9]+)_Q([01])_G(128|256)(_P([01]))?$" _m ${tag})
  if(NOT _m)
    message(FATAL_ERROR "probe_verify_attn: bad lever tag ${tag}")
  endif()
  set(_kt ${CMAKE_MATCH_1})
  set(_ks ${CMAKE_MATCH_3})
  set(_dsplit ${CMAKE_MATCH_4})
  set(_q ${CMAKE_MATCH_5})
  set(_g ${CMAKE_MATCH_6})
  set(_p 0)
  if(CMAKE_MATCH_8)
    set(_p ${CMAKE_MATCH_8})
  endif()
  set(_ph 0)
  if(CMAKE_MATCH_2 STREQUAL "h")
    set(_ph 1)
  endif()
  set(_opts)
  set(_maxl 1024)
  if(_g EQUAL 256)
    set(_opts OPTIONS -cl-intel-256-GRF-per-thread)
    set(_maxl 512)
  endif()
  foreach(MM 2 3 4 5 6 7 8 16)
    if(_ph)
      math(EXPR _rb "6 * ((${MM} + 7) / 8)")
    else()
      math(EXPR _rb "(6 * ${MM} + 7) / 8")
    endif()
    math(EXPR _ds "256 / ${_dsplit}")
    math(EXPR _lanes "16 * ${_rb} * ${_ds} * ${_ks}")
    math(EXPR _slm "${_rb} * 16 * 128 * 2 * (1 + ${_q})")
    if(_ks GREATER 1)
      math(EXPR _slm "${_slm} + ${_rb} * 8 * 256 * 4 + ${_rb} * ${_ds} * 8 * 2 * 4")
    endif()
    if(_lanes GREATER _maxl OR _slm GREATER 131072)
      continue()
    endif()
    foreach(KV 0 1)
      set(_sfx "")
      if(KV)
        set(_sfx _KV8)
      endif()
      add_ocloc_kernel(pvd_${tag}_M${MM}${_sfx} SOURCE ${PVD_CL}
                       DEFINES CTRL_POS=0 CTRL_NACT=1 M=${MM} TGT=32 KT=${_kt} PACK_H=${_ph} KS=${_ks}
                               DSPLIT=${_dsplit} QSPLIT=${_q} PSPLIT=${_p} KV8=${KV}
                       ${_opts})
      list(APPEND PVD_TARGETS kernel_pvd_${tag}_M${MM}${_sfx})
    endforeach()
  endforeach()
endforeach()
foreach(MM 5 6 7 8)
  add_ocloc_kernel(pvd_v2kv8_M${MM} SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/kv8.cl
                   DEFINES CTRL_POS=0 CTRL_NACT=1 BATCH_ROWS=1 BROW_WORDS=8 BROW_POS=0 BROW_KVB=1 BROW_SRC=2
                           BROW_DST=3 BROW_SLOT=4 KV8_DEC=1 M=${MM} TGT=32)
  list(APPEND PVD_TARGETS kernel_pvd_v2kv8_M${MM})
endforeach()
add_custom_target(pvd_kernels DEPENDS ${PVD_TARGETS})
add_executable(probe_verify_attn probe_verify_attn.cc)
target_include_directories(probe_verify_attn PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_verify_attn PRIVATE b70_l0 b70_model pthread)
b70_target_kernel_dir(probe_verify_attn)
get_property(PVD_SRC_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
get_property(PVD_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_verify_attn pvd_kernels ${PVD_SRC_KERNELS} ${PVD_PF_KERNELS})

# --- probe_vattn_ab: spec 28c's open error (docs/07-open-questions.md §11), attn_verify_dpas against v2 on a real
# DFlash run - every verify twice (DPAS, then v2), the flagged iterations replayed layer by layer on v2's inputs
# (CaptureOptions::attn_dump) against fp64. `pvab_<arm>` are attn_verify_dpas.cl at the production bf16 lever line, M = 8,
# built differently for the compiler / precision A/B (build-time only; no production binary binds them):
#   P0_G256  P in bf16 alone (PSPLIT=0, the kernel before 2026-11-04), the production command line
#   P0_G128  the same without -cl-intel-256-GRF-per-thread (no spill)
#   P0_O0    the same with -cl-opt-disable
#   P0_EAGER the scalar walk (EAGER=1: same blocks, tiles, masks, P rounding and merge order, no DPAS / sub-groups)
#   P1_EAGER the scalar walk at PSPLIT=1 (the production line's P as hi + lo)
# Pass them with --bins (the production binary is always the first arm).
if(B70_MTP AND B70_VERIFY_M8 AND B70_VERIFY_DPAS)
  set(PVAB_DEFS CTRL_POS=0 CTRL_NACT=1 M=8 TGT=32 KT=32 PACK_H=0 KS=4 DSPLIT=256 QSPLIT=1 KV8=0)
  add_ocloc_kernel(pvab_P0_G256 SOURCE ${PVD_CL} DEFINES ${PVAB_DEFS} PSPLIT=0 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pvab_P0_G128 SOURCE ${PVD_CL} DEFINES ${PVAB_DEFS} PSPLIT=0)
  add_ocloc_kernel(pvab_P0_O0 SOURCE ${PVD_CL} DEFINES ${PVAB_DEFS} PSPLIT=0
                   OPTIONS -cl-intel-256-GRF-per-thread -cl-opt-disable)
  add_ocloc_kernel(pvab_P0_EAGER SOURCE ${PVD_CL} DEFINES ${PVAB_DEFS} PSPLIT=0 EAGER=1)
  add_ocloc_kernel(pvab_P1_EAGER SOURCE ${PVD_CL} DEFINES ${PVAB_DEFS} PSPLIT=1 EAGER=1)
  set(PVAB_ARMS P0_G256 P0_G128 P0_O0 P0_EAGER P1_EAGER)
  list(TRANSFORM PVAB_ARMS PREPEND kernel_pvab_)
  add_executable(probe_vattn_ab probe_vattn_ab.cc)
  target_include_directories(probe_vattn_ab PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(probe_vattn_ab PRIVATE b70_runtime b70_dflash_runtime b70_loader b70_model b70_l0 b70_prefill_host)
  b70_target_kernel_dir(probe_vattn_ab)
  add_dependencies(probe_vattn_ab ${PVAB_ARMS} ${PVD_SRC_KERNELS} ${PVD_PF_KERNELS})
  # probe_verify_rows (spec 28, dflash_gpu_test t5_test-box): an M = 8 verify row against the M = 1 plain row on the
  # same state - the plain history and the DFlash history, the verify at the rule / v2 / gemv arms.
  add_executable(probe_verify_rows probe_verify_rows.cc)
  target_include_directories(probe_verify_rows PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(probe_verify_rows PRIVATE b70_runtime b70_dflash_runtime b70_loader b70_model b70_l0
    b70_prefill_host)
  b70_target_kernel_dir(probe_verify_rows)
  add_dependencies(probe_verify_rows ${PVD_SRC_KERNELS} ${PVD_PF_KERNELS})
endif()

# --- probe_chunk_rows: plan 5d P0, the prefill linears at a 4096-row chunk ------------------
# Plan: docs/superpowers/plans/2026-10-27-spec5d-prefill-chunk-4096.md (Task 1). Record:
# docs/probe-prefill-chunk-4096-<date>.md. Production kernels only (pf_int8 / pf_dequant_slab /
# pf_gemm), M = 4096 against 2 x 2048; no production file is touched.
add_executable(probe_chunk_rows probe_chunk_rows.cc)
target_include_directories(probe_chunk_rows PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_chunk_rows PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
b70_target_kernel_dir(probe_chunk_rows)
get_property(PCR_PF_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/prefill PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_chunk_rows ${PCR_PF_KERNELS})

# --- probe_small_chunk: open-work 3a, the prefill linears at short chunks ------------------------
# The production walk (pf_int8) at M = 16 .. 256 itemised (requant / GEMM), against the split GEMM
# (pf_gemm_i8_part + pf_gemm_i8_fold, bit-checked) and gemm_sm (spec 28b) where built.
add_executable(probe_small_chunk probe_small_chunk.cc)
target_include_directories(probe_small_chunk PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_small_chunk PRIVATE b70_prefill_host b70_runtime b70_loader b70_l0 b70_model)
b70_target_kernel_dir(probe_small_chunk)
add_dependencies(probe_small_chunk ${PCR_PF_KERNELS} ${ALL_KERNELS})

# --- probe_tp: spec 17a (plan 17a), the tensor-parallel P0 on two cards, Qwen3.8 first -----------------
# Plan: docs/superpowers/plans/2026-10-05-spec17a-tp-probe.md. Record: docs/probes/probe-tp-2026-11-01.md.
# `probe_tp.cl` is a PROBE kernel (tp_send / tp_wait / tp_ping; the protocol is tp_protocol.h's): no runtime
# path binds it. The ptp_gemv_* binaries are today's gemv.cl / gemv_i8w.cl at the PER-CARD TP shapes of
# spec 17 §R2.5 (tp_shapes.h), each with its one-card production cell's defines (src/kernels/CMakeLists.txt):
# probe-only names, no production binary changes.
add_ocloc_kernel(ptp_kernels SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/probe_tp.cl)
set(PTP_GEMV_CL ${CMAKE_SOURCE_DIR}/src/kernels/gemv.cl)
set(PTP_TARGETS kernel_ptp_kernels)
# name-suffix K N S L extra-defines   (qkv‖z L1 S1; out_proj / o_proj 3072 x 5120 as the 6144 x 5120 S4 cell;
# q‖k‖v 7168 as the 14336 S2 cell; gate‖up as the 34816 S8 cell; down as the 17408 S4 cell)
foreach(row "5120 8192 1 1 -" "3072 5120 4 0 B" "5120 7168 2 0 B" "5120 17408 8 0 D" "8704 5120 4 0 B")
  separate_arguments(f UNIX_COMMAND "${row}")
  list(GET f 0 K)
  list(GET f 1 N)
  list(GET f 2 S)
  list(GET f 3 L)
  list(GET f 4 X)
  set(defs M=1 K=${K} N=${N} S=${S} LAYOUT=${L})
  if(X STREQUAL "B")
    list(APPEND defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
  elseif(X STREQUAL "D")
    list(APPEND defs GEMV_DEQ_SHIFT=1)
  endif()
  add_ocloc_kernel(ptp_gemv_M1_K${K}_N${N}_S${S}_L${L} SOURCE ${PTP_GEMV_CL} DEFINES ${defs})
  list(APPEND PTP_TARGETS kernel_ptp_gemv_M1_K${K}_N${N}_S${S}_L${L})
endforeach()
add_ocloc_kernel(ptp_gemv_i8w_M1_K5120_N124160 SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/gemv_i8w.cl
                 DEFINES M=1 K=5120 N=124160)
list(APPEND PTP_TARGETS kernel_ptp_gemv_i8w_M1_K5120_N124160)
add_custom_target(ptp_kernel_set DEPENDS ${PTP_TARGETS})

add_executable(probe_tp probe_tp.cc)
target_include_directories(probe_tp PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_tp PRIVATE b70_l0 b70_model pthread)
b70_target_kernel_dir(probe_tp)
add_dependencies(probe_tp ptp_kernel_set)
# the one-card production binaries `probe_tp step` times beside the per-card ones (only these: a new box tree
# builds probe_tp without the whole kernel set)
foreach(v gemv_M1_K5120_N16384_S1_L1 gemv_M1_K6144_N5120_S4_L0 gemv_M1_K5120_N14336_S2_L0
          gemv_M1_K5120_N34816_S8_L0 gemv_M1_K17408_N5120_S4_L0 gemv_i8w_M1_K5120_N248320)
  if(TARGET kernel_${v})
    add_dependencies(probe_tp kernel_${v})
  endif()
endforeach()
# --- probe_reset: what Engine::reset()'s persistent fills cost at max_len small vs auto (branch cheap-reset) ---
add_executable(probe_reset probe_reset.cc)
target_include_directories(probe_reset PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_reset PRIVATE b70_runtime b70_model b70_l0)
