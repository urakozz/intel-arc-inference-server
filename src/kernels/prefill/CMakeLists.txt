# Prefill-only kernels (spec 2). None is bound by src/runtime/capture.cc and
# none changes the decode list's 774 launches or 19 modules.
#
# **No `M` appears in any name or on any command line below**, and that is the
# rule rather than an accident: interfaces.md fixes the row count as a runtime
# kernel argument with grids of `ceil(M / tile)` and device-side masking, so one
# binary serves every chunk width the CLI's `--pp-chunk` can ask for. `K`,
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

# Spec 2.1 (plan 9a): the 1024-column slab dequant, one variant per production (K, N, layout)
# -- the layouts the loader actually ships (qkv||z tiled, the rest GPTQ-native).
set(PF_SLAB_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_dequant_slab.cl)
foreach(row "5120 16384 1" "6144 5120 0" "5120 34816 0" "17408 5120 0" "5120 14336 0")
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
foreach(K 5120 6144 17408)
  math(EXPR NBLK "${K} / 1024")
  add_ocloc_kernel(pf_quant_had_K${K} SOURCE ${PF_INT8_CL} DEFINES PF_QUANT_HAD=1 NBLK=${NBLK})
endforeach()
foreach(L 0 1)
  add_ocloc_kernel(pf_requant_rot_L${L} SOURCE ${PF_INT8_CL} DEFINES PF_REQUANT_ROT=1 LAYOUT=${L})
endforeach()
add_ocloc_kernel(pf_gemm_i8 SOURCE ${PF_INT8_CL} DEFINES PF_GEMM_I8=1
                 OPTIONS -cl-intel-256-GRF-per-thread)
add_ocloc_kernel(pf_gemm_i8_SILU SOURCE ${PF_INT8_CL} DEFINES PF_GEMM_I8=1 SILU_EPI=1
                 OPTIONS -cl-intel-256-GRF-per-thread)
