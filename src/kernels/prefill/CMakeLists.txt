# Prefill-only kernels (spec 2). None is bound by src/runtime/capture.cc and
# none changes the decode list's 774 launches or 19 modules.
#
# **No `M` appears in any name or on any command line below**, and that is the
# rule rather than an accident: interfaces.md fixes the row count as a runtime
# kernel argument with grids of `ceil(M / tile)` and device-side masking, so one
# binary serves every chunk width the CLI's `--prefill-chunk` can ask for. `K`,
# `S_PREV` and the norm's group counts stay in the names because they are
# strides and grids the binary bakes -- the arrangement `prep_res_fold`'s `G`
# uses one directory up, for the same reason: a host that disagrees names a file
# that does not exist and throws instead of striding a buffer two ways.
#
# `S_PREV` is 0 or 1 and nothing else (plan 6b ruling R1: the prefill path runs
# every linear unsplit). The decode variants of these same kernels fold 8 and 4
# slices, which is what made their first M=2048 measurement 5x too expensive.
set(PF_DEQUANT_CL ${CMAKE_CURRENT_SOURCE_DIR}/dequant.cl)

function(add_pf_dequant K N L T)
  add_ocloc_kernel(pf_dequant_tile_K${K}_N${N}_L${L}_T${T} SOURCE ${PF_DEQUANT_CL}
                   DEFINES K=${K} N=${N} LAYOUT=${L} TRANSPOSED=${T})
endfunction()

# The unit-test shape, both layouts x both output orientations.
foreach(L 0 1)
  foreach(T 0 1)
    add_pf_dequant(256 64 ${L} ${T})
  endforeach()
endforeach()

# Production dimensions: both loader layouts and both store geometries are
# built so the probe measures rather than assumes either contrast.
set(PF_SHAPES "5120 16384" "6144 5120" "5120 34816"
              "17408 5120" "5120 14336" "5120 248320")
foreach(row IN LISTS PF_SHAPES)
  separate_arguments(f UNIX_COMMAND "${row}")
  list(GET f 0 K)
  list(GET f 1 N)
  foreach(L 0 1)
    foreach(T 0 1)
      add_pf_dequant(${K} ${N} ${L} ${T})
    endforeach()
  endforeach()
endforeach()

# --- The runtime-M small kernels (plan 6b Tasks 3-5) -------------------------
# Spec 15b: as in ../CMakeLists.txt, every binary here is at Qwen3.8's shapes and
# keeps its name; another model's pf_embed_gather / pf_ab_proj (`_D<hidden>`),
# pf_gated_head / pf_gdn_* (`_GK<k>V<v>`) and pf_attn_prep* / pf_attn /
# pf_flash_attn (`_Q<q>KV<kv>`) are named by kernels/prefill/pf_kernels.h and
# built from spec 15c on.
set(PF_PREP_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_prep.cl)
function(add_pf_res_fold K SP G)
  add_ocloc_kernel(pf_res_fold_K${K}_SP${SP}_G${G} SOURCE ${PF_PREP_CL}
                   DEFINES K=${K} S_PREV=${SP} FOLD_G=${G})
endfunction()
add_pf_res_fold(5120 0 20)   # layer 0's leading norm: nothing to fold
add_pf_res_fold(5120 1 20)   # every other norm (ruling R1: S = 1)
add_ocloc_kernel(pf_norm_finish_K5120_G20_W20 SOURCE ${PF_PREP_CL}
                 DEFINES K=5120 NORM_G=20 NORM_WGS=20)
add_ocloc_kernel(pf_silu_mul SOURCE ${PF_PREP_CL})
add_ocloc_kernel(pf_embed_gather SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_embed.cl
                 DEFINES HIDDEN=5120 VOCAB=248320)
add_ocloc_kernel(pf_gated_head SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gated_head.cl)
# CTRL_DEFINES is set in src/kernels/CMakeLists.txt, which adds this directory
# AFTER that line, so the variable is in scope (add_subdirectory inherits the
# parent scope). QNORM_OFF / KNORM_OFF are loader/small_layout.h's kFaOffQNorm
# (0 B) and kFaOffKNorm (1024 B) in floats, exactly as ATTN_DEFINES sets them.
add_ocloc_kernel(pf_attn_prep SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn_prep.cl
                 DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256)
# Ruling A9's bf16 `attn_q`, for the composed attention path (ruling A14). ONE
# `-D` apart from the binary above, which is the point: everything before the
# store is the same text, so pf_attn_test's bit-identity to `attn_prep_M1`
# covers the norm, the tree, the rstd and the RoPE of BOTH.
add_ocloc_kernel(pf_attn_prep_q16 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn_prep.cl
                 DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256 Q_BF16=1)
# The composed attention path's two own kernels: the causal row softmax between
# the inherited QK^T and PV GEMMs, and the output gate `attn_reduce` used to
# carry. No `C`, no `L`, no depth in the name -- every extent is a runtime
# argument or a grid dimension.
add_ocloc_kernel(pf_attn SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn.cl)
# a||b, zero-padded 96 -> 128, at the measured {16, 16} tiling
# (kernels::gemv_bf16_tiling) so that M = 1 is bit-identical to the binary
# capture.cc binds. `pf_gemv_int4_M` (plan 6b Task 4's temporary int4 GEMV) is
# deliberately NOT built: plan 6c lands sycl-tla's real GEMM in its place.
add_ocloc_kernel(pf_ab_proj SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemv_bf16.cl
                 DEFINES K=5120 N=128 COLS_PER_WG=16 KSPLIT=16)

# --- gdn_chunk's kernels (plan 6b Tasks 6-8) --------------------------------
# NEGA_OFF / DTBIAS_OFF are the layer's GDN small-block offsets **in floats**,
# the same two src/kernels/CMakeLists.txt:158 passes to gdn_step and owned by
# loader/small_layout.h. No `M` and no `C`: every one of these takes the chunk
# width as a runtime argument.
add_ocloc_kernel(pf_gdn_conv SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_conv.cl
                 DEFINES NEGA_OFF=40960 DTBIAS_OFF=41008)
add_ocloc_kernel(pf_gdn_wy SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_wy.cl)
add_ocloc_kernel(pf_gdn_scan SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_scan.cl)

# --- The execution-context probe (plan 6b Task 2) ---------------------------
# Bound only by tests/prefill/context_test.cc; it proves the immediate list's
# in-order execution and per-launch argument binding empirically.
add_ocloc_kernel(pf_probe_chain SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_probe_chain.cl)

# Spec 2.1 (plan 9a): the production DPAS GEMM, promoted from probe P-A. 256 GRF is
# mandatory (P2 measured a 33x loss without it) and is carried on the compile line, the
# only place an AOT build honours it. T0: B is [K][N]; T1: B is [N][K] (the KV cache).
set(PF_GEMM_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemm.cl)
foreach(T 0 1)
  add_ocloc_kernel(pf_gemm_T${T} SOURCE ${PF_GEMM_CL} DEFINES TRANSB=${T}
                   OPTIONS -cl-intel-256-GRF-per-thread)
endforeach()
# Parity program S2(a): the gate||up variant whose epilogue runs pf_silu_mul's
# chain out of the accumulators and stores bf16 x, so the fp32 [M][34816]
# `partials` rectangle is never written. A THIRD binary and not a replacement --
# the two above keep the exact command line and the exact source they had, which
# is what makes "every other linear is untouched" a build fact.
add_ocloc_kernel(pf_gemm_T0_SILU SOURCE ${PF_GEMM_CL} DEFINES TRANSB=0 SILU_EPI=1
                 OPTIONS -cl-intel-256-GRF-per-thread)

# Open-work 3a: pf_gemm for a short chunk (pf_gemm_small.cl), bit-identical to the two above:
# NA 8-row blocks x NB 16-column atoms a sub-group, SN sub-groups a work-group - the two tiles
# runtime::prefill::pf_small_tile binds (A2_B2_S4 to 896 rows, A4_B2_S4 to 1024;
# tools/probe/probe_small_chunk.cc). 256 GRF where the accumulators alone fill 128 (NA x NB >= 8).
# Additive: no existing binary changes. -DB70_PF_SMALL_SWEEP=ON builds the probe's whole sweep
# (NA 1/2/4 x NB 2/4 x SN 1/2/4) instead.
option(B70_PF_SMALL_SWEEP "build every pf_gemm_small tile probe_small_chunk sweeps" OFF)
set(_pf_small_tiles "2 2 4" "4 2 4")
if(B70_PF_SMALL_SWEEP)
  set(_pf_small_tiles "")
  foreach(NA 1 2 4)
    foreach(NB 2 4)
      foreach(SN 1 2 4)
        list(APPEND _pf_small_tiles "${NA} ${NB} ${SN}")
      endforeach()
    endforeach()
  endforeach()
endif()
foreach(tile IN LISTS _pf_small_tiles)
  separate_arguments(_t UNIX_COMMAND "${tile}")
  list(GET _t 0 NA)
  list(GET _t 1 NB)
  list(GET _t 2 SN)
  math(EXPR _acc "${NA} * ${NB}")
  set(_grf "")
  if(_acc GREATER_EQUAL 8)
    set(_grf OPTIONS -cl-intel-256-GRF-per-thread)
  endif()
  add_ocloc_kernel(pf_gemm_small_A${NA}_B${NB}_S${SN} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemm_small.cl
                   DEFINES NA=${NA} NB=${NB} SN=${SN} ${_grf})
  add_ocloc_kernel(pf_gemm_small_A${NA}_B${NB}_S${SN}_SILU SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemm_small.cl
                   DEFINES NA=${NA} NB=${NB} SN=${SN} SILU_EPI=1 ${_grf})
endforeach()

# Open-work 3b: pf_gemm_small reading the int4 weights directly (pf_gemm_q4.cl) - no bf16 slab,
# bit-identical to the slab walk. The tiles runtime::prefill::pf_q4_tile binds: plain (both
# layouts) A4_B1_S1 to 32 rows and A8_B1_S2 to 384; gate||up's SiLU form (layout 0) A1_B2_S4 to 8
# rows and A4_B2_S1 to 192 (tools/probe/probe_fused_small.cc). 256 GRF where the accumulators and
# the two groups of words fill the default 128 (NA x NB >= 4). Additive: no existing binary
# changes. -DB70_PF_Q4_SWEEP=ON builds the probe's sweep (NA 1/2/4 x NB 1/2/4 x SN 1/2/4 and NA 8
# x NB 1/2 x SN 1/2, every one plain at both layouts and, NB even, SiLU) instead.
option(B70_PF_Q4_SWEEP "build every pf_gemm_q4 tile probe_fused_small sweeps" OFF)
set(_pf_q4_plain "4 1 1" "8 1 2")
set(_pf_q4_silu "1 2 4" "4 2 1")
if(B70_PF_Q4_SWEEP)
  set(_pf_q4_plain "8 1 1" "8 2 1" "8 1 2" "8 2 2")
  foreach(NA 1 2 4)
    foreach(NB 1 2 4)
      foreach(SN 1 2 4)
        list(APPEND _pf_q4_plain "${NA} ${NB} ${SN}")
      endforeach()
    endforeach()
  endforeach()
  set(_pf_q4_silu "")
  foreach(tile IN LISTS _pf_q4_plain)
    separate_arguments(_t UNIX_COMMAND "${tile}")
    list(GET _t 1 NB)
    math(EXPR _odd "${NB} % 2")
    if(_odd EQUAL 0)
      list(APPEND _pf_q4_silu "${tile}")
    endif()
  endforeach()
endif()
function(add_pf_gemm_q4 tile suffix)   # suffix: _L0, _L1 or _L0_SILU
  separate_arguments(_t UNIX_COMMAND "${tile}")
  list(GET _t 0 NA)
  list(GET _t 1 NB)
  list(GET _t 2 SN)
  math(EXPR _acc "${NA} * ${NB}")
  set(_grf "")
  if(_acc GREATER_EQUAL 4)
    set(_grf OPTIONS -cl-intel-256-GRF-per-thread)
  endif()
  set(_defs LAYOUT=0)
  if(suffix STREQUAL "_L1")
    set(_defs LAYOUT=1)
  elseif(suffix STREQUAL "_L0_SILU")
    set(_defs LAYOUT=0 SILU_EPI=1)
  endif()
  add_ocloc_kernel(pf_gemm_q4_A${NA}_B${NB}_S${SN}${suffix} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemm_q4.cl
                   DEFINES NA=${NA} NB=${NB} SN=${SN} ${_defs} ${_grf})
endfunction()
foreach(tile IN LISTS _pf_q4_plain)
  add_pf_gemm_q4("${tile}" _L0)
  add_pf_gemm_q4("${tile}" _L1)
endforeach()
foreach(tile IN LISTS _pf_q4_silu)
  add_pf_gemm_q4("${tile}" _L0_SILU)
endforeach()

# Open-work 3d probe controls: the two-pass gate/up form loses on the real engine despite
# its isolated-linear gain. Keep these variants only for reproducing that comparison.
if(B70_PF_Q4_SWEEP)
  foreach(SN 1 2 4)
    foreach(pass GATE UP_SILU)
      if(pass STREQUAL "GATE")
        set(_split 1)
      else()
        set(_split 2)
      endif()
      add_ocloc_kernel(pf_gemm_q4_A8_B1_S${SN}_L0_${pass}
                       SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemm_q4.cl
                       DEFINES NA=8 NB=1 SN=${SN} LAYOUT=0 SILU_SPLIT=${_split}
                       OPTIONS -cl-intel-256-GRF-per-thread)
    endforeach()
  endforeach()
endif()

# Open-work 3d: paired NB1 sub-groups exchange gate bf16 through SLM in one launch.
# S4 clears the fixed bitwise and >=5% gates at every row from 193 to 320; S2 remains a probe control.
set(_pf_q4_slm_silu 4)
if(B70_PF_Q4_SWEEP)
  set(_pf_q4_slm_silu 2 4)
endif()
foreach(SN IN LISTS _pf_q4_slm_silu)
  add_ocloc_kernel(pf_gemm_q4_A8_B1_S${SN}_L0_SLM_SILU
                   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemm_q4.cl
                   DEFINES NA=8 NB=1 SN=${SN} LAYOUT=0 SILU_SLM=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
endforeach()

# Spec 2.1 (plan 9a): the 1024-column slab dequant, one variant per production (K, N, layout)
# -- the layouts the loader actually ships (qkv||z tiled, the rest GPTQ-native).
set(PF_SLAB_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_dequant_slab.cl)
set(PF_SLAB_ROWS "5120 16384 1" "6144 5120 0" "5120 34816 0" "17408 5120 0" "5120 14336 0")
foreach(row IN LISTS PF_SLAB_ROWS)
  separate_arguments(f UNIX_COMMAND "${row}")
  list(GET f 0 K)
  list(GET f 1 N)
  list(GET f 2 L)
  add_ocloc_kernel(pf_dequant_slab_K${K}_N${N}_L${L} SOURCE ${PF_SLAB_CL}
                   DEFINES K=${K} N=${N} LAYOUT=${L})
endforeach()

# Spec 5 (plan 5a): the int8 prefill linears, one source, per-family defines.
# The quantiser and requant run at the default 128 GRF (probe_w8a8 measured 256
# halving their occupancy); the GEMM needs 256 for its 64 int32 accumulators.
set(PF_INT8_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_int8.cl)
set(PF_HAD_K 3072 5120 6144 17408)
foreach(K IN LISTS PF_HAD_K)
  math(EXPR NBLK "${K} / 1024")
  add_ocloc_kernel(pf_quant_had_K${K} SOURCE ${PF_INT8_CL} DEFINES PF_QUANT_HAD=1 NBLK=${NBLK})
endforeach()
add_ocloc_kernel(pf_quant_had_K8704 SOURCE ${PF_INT8_CL}
                 DEFINES PF_QUANT_HAD=1 NBLK=9 PAD_K=1)
foreach(L 0 1)
  add_ocloc_kernel(pf_requant_rot_L${L}_PAD SOURCE ${PF_INT8_CL}
                   DEFINES PF_REQUANT_ROT=1 LAYOUT=${L} PAD_K=1)
  add_ocloc_kernel(pf_requant_rot_L${L} SOURCE ${PF_INT8_CL} DEFINES PF_REQUANT_ROT=1 LAYOUT=${L})
endforeach()
add_ocloc_kernel(pf_gemm_i8 SOURCE ${PF_INT8_CL} DEFINES PF_GEMM_I8=1
                 OPTIONS -cl-intel-256-GRF-per-thread)
add_ocloc_kernel(pf_gemm_i8_SILU SOURCE ${PF_INT8_CL} DEFINES PF_GEMM_I8=1 SILU_EPI=1
                 OPTIONS -cl-intel-256-GRF-per-thread)

# Spec 6 (plan 6b): the fused flash attention, plan 6a's winner pfa_KT64_R16_H6_Q0;
# spec 6c: RPW 8 (6 sub-groups per work-group) and EXP2=1 (exp2 with the scale folded):
# the operator enabled exp2 on 2026-09-27 accepting its 7.6e-6 K3a miss on l0 (spec 6 §9).
add_ocloc_kernel(pf_flash_attn SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_flash_attn.cl
                 DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 OPTIONS -cl-intel-256-GRF-per-thread)
# 3l (2): the split-K form for short chunks deep in the cache (SPLIT=1: spans of the key range into
# pf_o's spare rows) and the merge that folds the spans into pf_o. Additive: pf_flash_attn's
# command line and token stream are unchanged.
add_ocloc_kernel(pf_flash_attn_split SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_flash_attn.cl
                 DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 SPLIT=1 OPTIONS -cl-intel-256-GRF-per-thread)
add_ocloc_kernel(pf_flash_merge SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_flash_attn.cl
                 DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 FLASH_MERGE=1)

# Spec 8 (plan 8b Task 3): the MTP head's KV fill during prefill. The two pre-fc norms
# write the two halves of one [C][10240] row (X_STRIDE), fc's output starts the head's
# residual stream (ZERO_RESID), and the head's bf16 linears reach pf_gemm through an
# untiling slab kernel (pf_bf16_slab) instead of pf_dequant_slab. All additive: no
# existing binary's command line changes.
if(B70_MTP)
  add_ocloc_kernel(pf_res_fold_K5120_SP1_G20_Z SOURCE ${PF_PREP_CL}
                   DEFINES K=5120 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
  add_ocloc_kernel(pf_norm_finish_K5120_G20_W20_X10240 SOURCE ${PF_PREP_CL}
                   DEFINES K=5120 NORM_G=20 NORM_WGS=20 X_STRIDE=10240)
  foreach(K 10240 5120)
    add_ocloc_kernel(pf_bf16_slab_K${K} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_bf16_slab.cl
                     DEFINES K=${K})
  endforeach()
endif()

# Spec 12b (`--kv-cache int8`): the prefill path over the int8 KV cache, from the same
# kv8.cl as the decode variants (KV8_CL, set in the parent directory): the writer
# (pf_attn_prep_q16's chain, PF = 1, then rotate + quantise), the flash attention reading
# int8 + scales (pf_flash_attn's tiling and exp2, 256 GRF), and the gate that un-rotates
# each head first. Additive: a bf16 engine binds none of them; the composed path and
# sycl-tla have no int8 form and refuse --kv-cache int8 (runtime/prefill/attn.cc).
if(B70_KV8)
  add_ocloc_kernel(pf_attn_prep_kv8 SOURCE ${KV8_CL} DEFINES ${KV8_PREP_DEFINES} PF=1)
  add_ocloc_kernel(pf_flash_attn_kv8 SOURCE ${KV8_CL}
                   DEFINES KV8_FLASH=1 KT=64 RPW=8 HPW=6 QREG=0 EXP2=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_flash_attn_kv8_raw SOURCE ${KV8_CL}
                   DEFINES KV8_FLASH=1 KV8_RAW=1 KT=64 RPW=8 HPW=6 QREG=0 EXP2=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
  # 3l (2): the split forms over int8 (in place, and raw after kv8_widen); pf_flash_merge folds them.
  add_ocloc_kernel(pf_flash_attn_kv8_split SOURCE ${KV8_CL}
                   DEFINES KV8_FLASH=1 KV8_SPLIT=1 KT=64 RPW=8 HPW=6 QREG=0 EXP2=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_flash_attn_kv8_raw_split SOURCE ${KV8_CL}
                   DEFINES KV8_FLASH=1 KV8_RAW=1 KV8_SPLIT=1 KT=64 RPW=8 HPW=6 QREG=0 EXP2=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_attn_gate_kv8 SOURCE ${KV8_CL} DEFINES KV8_GATE=1)
endif()

# --- Spec 15d: Ornith 1.5 35B-A3B prefill (model::ornith()) -------------------------------
# Additive only: every binary above keeps its name and its command line (tools/kernel_cmdlines
# diff), and the .cl edits that made the prefill head counts overridable (GDN_K_HEADS /
# GDN_V_HEADS, FA_Q_HEADS / FA_KV_HEADS - the decode kernels' pair from spec 15c) leave the
# default token stream untouched: the defines below are the only way into the new branches.
# What Ornith's chunk binds (runtime/prefill/step.cc, moe.cc):
#   norms     pf_res_fold at K 2048: SP0 after layer 0's embedding and after every MoE
#             block (pf_moe_combine folds into the residual itself), SP1 after the mixer
#   embed     pf_embed_gather at hidden 2048; a||b pf_ab_proj at K 2048 ({16, 16})
#   linears   the slab dequant at the descriptor's provisional layouts - qkv||z 2048 x 12288
#             L1, out_proj = o_proj 4096 x 2048 L0, q||k||v 2048 x 9216 L0 - and the h8
#             quantiser at K 2048 / 4096 (pf_requant_rot / pf_gemm_i8 are shape-free)
#   GDN       the pf_gdn_* chain and pf_gated_head at 16 k / 32 v heads; negA / dt_bias at
#             32768 / 32800 floats (loader::make_small_layout(2048, 8192, 32))
#   FA        pf_attn_prep_q16, pf_attn (softmax + gate) and pf_flash_attn at 16 q / 2 kv
#             heads; flash at HPW 8 = the GQA group, so its grid.z stays 1
#   MoE       the router GEMV over the chunk (N 272, {16, 16}), decode's moe_route binary
#             (moe_M1_E256_T8_D2048_I512, grid (1, C)), pf_moe.cl (sort, gather, dequant,
#             combine) and the grouped GEMMs: gate||up bf16 and int8 (h8) with the SiLU
#             epilogue, down bf16 (its K 512 is not whole 1024-k rotation blocks)
# Not built for Ornith: the composed attention's identity prep (pf_attn_prep, fp32 q), the
# int8 KV cache's prefill (spec 12b, Qwen3.8's shapes), the MTP head's (spec 15e).
if(B70_ORNITH)
  set(PF_ORNITH_GDN GDN_K_HEADS=16 GDN_V_HEADS=32)
  set(PF_ORNITH_FA FA_Q_HEADS=16 FA_KV_HEADS=2)
  add_pf_res_fold(2048 0 20)
  add_pf_res_fold(2048 1 20)
  add_ocloc_kernel(pf_norm_finish_K2048_G20_W20 SOURCE ${PF_PREP_CL}
                   DEFINES K=2048 NORM_G=20 NORM_WGS=20)
  add_ocloc_kernel(pf_embed_gather_D2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_embed.cl
                   DEFINES HIDDEN=2048 VOCAB=248320)
  add_ocloc_kernel(pf_ab_proj_D2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemv_bf16.cl
                   DEFINES K=2048 N=128 COLS_PER_WG=16 KSPLIT=16)
  foreach(row "2048 12288 1" "4096 2048 0" "2048 9216 0")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    list(GET f 2 L)
    add_ocloc_kernel(pf_dequant_slab_K${K}_N${N}_L${L} SOURCE ${PF_SLAB_CL}
                     DEFINES K=${K} N=${N} LAYOUT=${L})
  endforeach()
  foreach(K 2048 4096)
    math(EXPR NBLK "${K} / 1024")
    add_ocloc_kernel(pf_quant_had_K${K} SOURCE ${PF_INT8_CL} DEFINES PF_QUANT_HAD=1 NBLK=${NBLK})
  endforeach()
  add_ocloc_kernel(pf_gdn_conv_GK16V32 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_conv.cl
                   DEFINES NEGA_OFF=32768 DTBIAS_OFF=32800 ${PF_ORNITH_GDN})
  add_ocloc_kernel(pf_gdn_wy_GK16V32 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_wy.cl
                   DEFINES ${PF_ORNITH_GDN})
  add_ocloc_kernel(pf_gdn_scan_GK16V32 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_scan.cl
                   DEFINES ${PF_ORNITH_GDN})
  add_ocloc_kernel(pf_gated_head_GK16V32 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gated_head.cl
                   DEFINES ${PF_ORNITH_GDN})
  # Phase 2 Task 5: the residual fold + RMSNorm in one launch, bitwise the pair above.
  foreach(SP 0 1)
    add_ocloc_kernel(pf_res_norm_K2048_SP${SP}_G20 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_norm_fused.cl
                     DEFINES K=2048 S_PREV=${SP} FOLD_G=20)
  endforeach()
  # Phase 2 Task 6: the a||b GEMV over a chunk at 16 rows x 2 column tiles a work-group, bitwise pf_ab_proj_D2048.
  add_ocloc_kernel(pf_route_ab_D2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_route.cl DEFINES K=2048 N=128 MT=16 TC=2)
  add_ocloc_kernel(pf_route_router_K2048_N272 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_route.cl
                   DEFINES K=2048 N=272 MT=16 TC=2)
  # Phase 2 Task 2: the GDN front's conv + SiLU and q / k l2norm in one launch, bitwise.
  add_ocloc_kernel(pf_gdn_front_GK16V32 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_front.cl
                   DEFINES ${PF_ORNITH_GDN})
  # Phase 2 Task 3: the GDN intra-chunk stages in one launch (A, T, W / U, A2), judged.
  add_ocloc_kernel(pf_gdn_intra_GK16V32 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_intra.cl
                   DEFINES ${PF_ORNITH_GDN})
  # ... and under operator decision D2 (Q1-Q4 in fp32): w / u as hi / lo planes and the scan that reads them.
  add_ocloc_kernel(pf_gdn_intra_hl_GK16V32 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_intra.cl
                   DEFINES WU_HILO=1 ${PF_ORNITH_GDN})
  add_ocloc_kernel(pf_gdn_scan_hl_GK16V32 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_scan_hl.cl
                   DEFINES ${PF_ORNITH_GDN})
  add_ocloc_kernel(pf_attn_prep_q16_Q16KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn_prep.cl
                   DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256 Q_BF16=1 ${PF_ORNITH_FA})
  add_ocloc_kernel(pf_attn_Q16KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn.cl
                   DEFINES ${PF_ORNITH_FA})
  add_ocloc_kernel(pf_flash_attn_Q16KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_flash_attn.cl
                   DEFINES KT=64 RPW=8 HPW=8 QREG=0 EXP2=1 ${PF_ORNITH_FA}
                   OPTIONS -cl-intel-256-GRF-per-thread)
  # The MoE block. TM (the tile rows) and KC (PrefillScratchDims::kCDefault, the sort's SLM
  # staging) are mirrored by kernels::pf_moe::kTileM and runtime::PrefillScratchDims::kCDefault
  # (plan 5d: a MoE model's chunk cap stays 2048 - check_prefill_chunk_cap).
  add_ocloc_kernel(pf_moe_router_K2048_N272 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemv_bf16.cl
                   DEFINES K=2048 N=272 COLS_PER_WG=16 KSPLIT=16)
  add_ocloc_kernel(pf_moe_E256_T8_D2048_I512 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_moe.cl
                   DEFINES EXPERTS=256 TOP_K=8 HIDDEN=2048 INTER=512 TM=32 KC=2048)
  set(PF_MOE_GEMM_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_moe_gemm.cl)
  add_ocloc_kernel(pf_moe_gemm_K2048_N1024_SILU SOURCE ${PF_MOE_GEMM_CL}
                   DEFINES K=2048 N=1024 TM=32 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_i8_K2048_N1024_SILU SOURCE ${PF_MOE_GEMM_CL}
                   DEFINES K=2048 N=1024 TM=32 I8=1 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_K512_N2048 SOURCE ${PF_MOE_GEMM_CL}
                   DEFINES K=512 N=2048 TM=32 OPTIONS -cl-intel-256-GRF-per-thread)
  # Plan 2026-11-12 Task 2: the multi-work-group sort (blocks of TB = 64 tokens; no chunk cap of its own).
  add_ocloc_kernel(pf_moe_sort_mw_E256_T8 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_moe_sort_mw.cl
                   DEFINES EXPERTS=256 TOP_K=8 TM=32 TB=64 MI=64)
  # Plan 2026-11-12 Task 3: the grouped GEMMs read straight from the int4 blocks (kernels::pf_moe::kQ4WgN columns a
  # work-group, kItemM rows a work item - the sort's MI).
  add_ocloc_kernel(pf_moe_q4gemm_K2048_N1024_SILU SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_moe_q4gemm.cl
                   DEFINES K=2048 N=1024 EXPERTS=256 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_q4gemm_K512_N2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_moe_q4gemm.cl
                   DEFINES K=512 N=2048 EXPERTS=256 OPTIONS -cl-intel-256-GRF-per-thread)
  # Plan 2026-11-12 Task 5: l0-int8's W4A8 gate||up (per-token-and-group int8 rows x the s4 blocks).
  add_ocloc_kernel(pf_moe_w4a8_K2048_N1024 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_moe_w4a8.cl
                   DEFINES K=2048 N=1024 EXPERTS=256 OPTIONS -cl-intel-256-GRF-per-thread)
endif()

# ==== Spec 15e: Ornith's MTP head KV fill during prefill (runtime/prefill/step.cc) == (begin) ==
# step_mtp_kv at Ornith's shapes - the head front only (its MoE FFN never runs in prefill):
# the two pre-fc norms into one [C][4096] row (X_STRIDE), fc's output starting the head's
# residual (ZERO_RESID), and the head's bf16 fc (K 4096) and q||k||v (K 2048) reaching
# pf_gemm through pf_bf16_slab. attn_prep is 15d's pf_attn_prep_q16_Q16KV2 (it reads the
# qkv at pitch N whatever produced it). Additive: no existing command line moves.
if(B70_ORNITH AND B70_MTP)
  add_ocloc_kernel(pf_res_fold_K2048_SP1_G20_Z SOURCE ${PF_PREP_CL}
                   DEFINES K=2048 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
  add_ocloc_kernel(pf_norm_finish_K2048_G20_W20_X4096 SOURCE ${PF_PREP_CL}
                   DEFINES K=2048 NORM_G=20 NORM_WGS=20 X_STRIDE=4096)
  foreach(K 4096 2048)
    add_ocloc_kernel(pf_bf16_slab_K${K} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_bf16_slab.cl
                     DEFINES K=${K})
  endforeach()
endif()
# ==== Spec 15e ================================================================ (end) ====

# ==== Spec 17c: tensor parallel PREFILL over two cards (`--parallel tp --prefill`, Qwen3.8) ======== (begin) ====
# Additive only: new names (every binary above keeps its name and command line) and one new source, pf_tp.cl (the
# fold of the two cards' halves). What a card's chunk binds (runtime/prefill/step.cc step_tp_unit / step_tp_head) at
# model::tp_card_desc's shapes - GDN 8 k / 24 v heads (NEGA_OFF / DTBIAS_OFF 20480 / 20504 floats, as 17b's
# gdn_step_M1_GK8V24), FA 12 q / 2 kv heads (GQA 6: pf_flash_attn's HPW 6 as one card's), MLP 8704:
#   linears  the slab dequant at the card's (K, N, layout): qkv||z 5120 x 8192 L1, q||k||v 5120 x 7168 L0, gate||up
#            5120 x 17408 L0 (column-parallel), out_proj / o_proj 3072 x 5120 and down 8704 x 5120 L0 (row-parallel);
#            pf_gemm / pf_gemm_small / pf_gemm_q4 are shape-free
#   GDN, FA  the pf_gdn_* chain and pf_gated_head at 8 / 24; pf_attn_prep_q16, pf_attn (the gate) and pf_flash_attn at
#            12 / 2; with B70_KV8 the int8 cache's writer, widening flash and gate at 12 / 2
#   fold     pf_tp_res_fold at K 5120, FOLD_G 20 (pf_res_fold's grid)
# Reused: pf_embed_gather, pf_res_fold SP0 (layer 0), pf_norm_finish, pf_ab_proj (a||b zero-padded to 128 columns, 48
# real), and the head's 17b binaries (gemv_i8w_M1_K5120_N124160, tp_argmax_M1_*). The l0-int8 backend adds
# the 3072/8704 quantisers and zero-padded requant variants below.
if(B70_TP)
  set(PF_TP_GDN GDN_K_HEADS=8 GDN_V_HEADS=24)
  set(PF_TP_FA FA_Q_HEADS=12 FA_KV_HEADS=2)
  set(B70_TP_PREFILL_KERNELS kernel_pf_quant_had_K3072 kernel_pf_quant_had_K8704
      kernel_pf_requant_rot_L0_PAD kernel_pf_requant_rot_L1_PAD)
  foreach(row "5120 8192 1" "5120 7168 0" "5120 17408 0" "3072 5120 0" "8704 5120 0")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    list(GET f 2 L)
    add_ocloc_kernel(pf_dequant_slab_K${K}_N${N}_L${L} SOURCE ${PF_SLAB_CL} DEFINES K=${K} N=${N} LAYOUT=${L})
    list(APPEND B70_TP_PREFILL_KERNELS kernel_pf_dequant_slab_K${K}_N${N}_L${L})
  endforeach()
  add_ocloc_kernel(pf_gdn_conv_GK8V24 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_conv.cl
                   DEFINES NEGA_OFF=20480 DTBIAS_OFF=20504 ${PF_TP_GDN})
  add_ocloc_kernel(pf_gdn_wy_GK8V24 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_wy.cl DEFINES ${PF_TP_GDN})
  add_ocloc_kernel(pf_gdn_scan_GK8V24 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_scan.cl DEFINES ${PF_TP_GDN})
  add_ocloc_kernel(pf_gated_head_GK8V24 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gated_head.cl DEFINES ${PF_TP_GDN})
  add_ocloc_kernel(pf_attn_prep_q16_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn_prep.cl
                   DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256 Q_BF16=1 ${PF_TP_FA})
  add_ocloc_kernel(pf_attn_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn.cl DEFINES ${PF_TP_FA})
  add_ocloc_kernel(pf_flash_attn_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_flash_attn.cl
                   DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 ${PF_TP_FA} OPTIONS -cl-intel-256-GRF-per-thread)
  # 3l (2): the short-chunk split and its merge at the card's heads.
  add_ocloc_kernel(pf_flash_attn_split_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_flash_attn.cl
                   DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 SPLIT=1 ${PF_TP_FA} OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_flash_merge_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_flash_attn.cl
                   DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 FLASH_MERGE=1 ${PF_TP_FA})
  add_ocloc_kernel(pf_tp_res_fold_K5120_G20 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_tp.cl DEFINES K=5120 FOLD_G=20)
  list(APPEND B70_TP_PREFILL_KERNELS kernel_pf_gdn_conv_GK8V24 kernel_pf_gdn_wy_GK8V24 kernel_pf_gdn_scan_GK8V24
       kernel_pf_gated_head_GK8V24 kernel_pf_attn_prep_q16_Q12KV2 kernel_pf_attn_Q12KV2 kernel_pf_flash_attn_Q12KV2
       kernel_pf_flash_attn_split_Q12KV2 kernel_pf_flash_merge_Q12KV2 kernel_pf_tp_res_fold_K5120_G20)
  if(B70_KV8)
    add_ocloc_kernel(pf_attn_prep_kv8_Q12KV2 SOURCE ${KV8_CL} DEFINES ${KV8_PREP_DEFINES} PF=1 ${PF_TP_FA})
    add_ocloc_kernel(pf_flash_attn_kv8_raw_Q12KV2 SOURCE ${KV8_CL}
                     DEFINES KV8_FLASH=1 KV8_RAW=1 KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 ${PF_TP_FA}
                     OPTIONS -cl-intel-256-GRF-per-thread)
    add_ocloc_kernel(pf_attn_gate_kv8_Q12KV2 SOURCE ${KV8_CL} DEFINES KV8_GATE=1 ${PF_TP_FA})
    add_ocloc_kernel(pf_flash_attn_kv8_split_Q12KV2 SOURCE ${KV8_CL}
                     DEFINES KV8_FLASH=1 KV8_SPLIT=1 KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 ${PF_TP_FA}
                     OPTIONS -cl-intel-256-GRF-per-thread)
    add_ocloc_kernel(pf_flash_attn_kv8_raw_split_Q12KV2 SOURCE ${KV8_CL}
                     DEFINES KV8_FLASH=1 KV8_RAW=1 KV8_SPLIT=1 KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 ${PF_TP_FA}
                     OPTIONS -cl-intel-256-GRF-per-thread)
    list(APPEND B70_TP_PREFILL_KERNELS kernel_pf_attn_prep_kv8_Q12KV2 kernel_pf_flash_attn_kv8_raw_Q12KV2
         kernel_pf_attn_gate_kv8_Q12KV2 kernel_pf_flash_attn_kv8_split_Q12KV2 kernel_pf_flash_attn_kv8_raw_split_Q12KV2)
  endif()
  set(B70_TP_PREFILL_KERNELS ${B70_TP_PREFILL_KERNELS} CACHE INTERNAL "spec 17c: the binaries a TP prefill chunk binds")
endif()
# ==== Spec 17c ============================================================= (end) ====
