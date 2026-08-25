add_ocloc_kernel(noop SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/noop.cl)
add_ocloc_kernel(bw_sum SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bw_sum.cl)
add_ocloc_kernel(ctrl_read SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ctrl_read.cl)

set(GEMV_CL ${CMAKE_CURRENT_SOURCE_DIR}/gemv.cl)
function(add_gemv_variant M K N S L)
  add_ocloc_kernel(gemv_M${M}_K${K}_N${N}_S${S}_L${L} SOURCE ${GEMV_CL}
                   DEFINES M=${M} K=${K} N=${N} S=${S} LAYOUT=${L})
endfunction()
# Probe matrix (spec 1 §4.2): the five int4 shapes the 27B runs after load-time
# fusion, both layouts, S in {1,2,4,8,16}. All S divide K/64 (96, 80 and 272).
foreach(L 0 1)
  foreach(S 1 2 4 8 16)
    add_gemv_variant(1 6144 5120  ${S} ${L})
    add_gemv_variant(1 5120 14336 ${S} ${L})
    add_gemv_variant(1 5120 16384 ${S} ${L})
    add_gemv_variant(1 5120 34816 ${S} ${L})
    add_gemv_variant(1 17408 5120 ${S} ${L})
  endforeach()
endforeach()
add_gemv_variant(2 6144 5120 1 0)   # spec 1 §9: the M loop must compile at M=2

set(GEMV_BF16_CL ${CMAKE_CURRENT_SOURCE_DIR}/gemv_bf16.cl)
# `C` and `S` are the kernel's COLS_PER_WG (output columns per work-group) and
# KSPLIT (subgroups splitting one tile's K, merged in SLM). Both are suppressed
# from the name AND from the command line at their defaults, exactly as
# `kernels::gemv_bf16_variant` suppresses them from the name -- so the two
# cannot drift, and the pre-lever binaries (lm_head's above all) are still built
# by the identical ocloc invocation they always were.
function(add_gemv_bf16_variant M K N C S)
  set(name gemv_bf16_M${M}_K${K}_N${N})
  set(defs M=${M} K=${K} N=${N})
  if(NOT C EQUAL 64)
    set(name ${name}_C${C})
    list(APPEND defs COLS_PER_WG=${C})
  endif()
  if(NOT S EQUAL 1)
    set(name ${name}_S${S})
    list(APPEND defs KSPLIT=${S})
  endif()
  add_ocloc_kernel(${name} SOURCE ${GEMV_BF16_CL} DEFINES ${defs})
endfunction()
add_gemv_bf16_variant(1 5120 248320 64 1)  # lm_head: 3880 work-groups, device full
add_gemv_bf16_variant(1 5120 128 64 1)     # a‖b (zero-padded 96 -> 128) as it shipped
# a‖b as the runtime binds it since spec 1.5's lever L2 (kernels::gemv_bf16_tiling):
# 8 work-groups of 16 K-slice subgroups = 128 threads, against 8. `16` is
# measured, not chosen: 1 / 4 / 16 read 48.774 / 13.115 / 5.340 us in situ
# (docs/15 §L2). The `{64, 1}` build above is kept and still compiled --
# gemv_bf16_test runs both over one set of inputs and holds their outputs to the
# reference tolerance, which is the bar a reordered sum gets.
add_gemv_bf16_variant(1 5120 128 16 16)

# The between-GEMV kernels: three entry points in one file (prep.cl), so each
# binary carries all three and the runtime loads the one it asked for. Only
# prep_res_norm has a shape to vary; the other two bake the model's dimensions.
#
# These used to pass `-cl-fp32-correctly-rounded-divide-sqrt` by hand. It is now
# the project-wide default in cmake/ocloc.cmake -- prep's references are compared
# bit-exactly and OpenCL's default 2.5 ulp on `/` and `sqrt` really does break
# that (docs/12-kernels.md, "Rounding discipline"), and the next kernel with a
# divide should not have to remember (controller ruling 2026-08-25, Task 4).
set(PREP_CL ${CMAKE_CURRENT_SOURCE_DIR}/prep.cl)
function(add_prep_res_norm M K SP)
  add_ocloc_kernel(prep_res_norm_M${M}_K${K}_SP${SP} SOURCE ${PREP_CL}
                   DEFINES M=${M} K=${K} S_PREV=${SP})
endfunction()
add_prep_res_norm(1 5120 0)    # first norm of a layer: no mixer partials to fold in
add_prep_res_norm(1 5120 16)   # after out_proj / down (S = 16)
add_prep_res_norm(2 5120 16)   # spec 1 §9: the M loop must compile at M=2

# The two-stage split of prep_res_norm - spec 1.5 lever L1. The runtime binds
# THIS pair at all 129 sites; the single-WG `prep_res_norm` above stays compiled
# and stays run by prep_test, which grades the pair against it (`resid`
# bit-identical, `x` to the ulp bar for the reordered Sigma). That is the same
# arrangement lever L2 left behind for `gemv_bf16`'s two tilings.
#
# G is in the NAME on both sides: stage A's FOLD_G is how many fp32 partial
# sums-of-squares it writes and stage B's NORM_G is how many it folds, so a host
# that pairs mismatched grids names a binary that does not exist and throws at
# capture instead of reducing the wrong number of slices. `W` is stage B's own
# work-group count, and W=1 is the plan's literal single-work-group finish, kept
# compiled as the measurement control for W=20.
function(add_prep_res_fold M K SP G)
  add_ocloc_kernel(prep_res_fold_M${M}_K${K}_SP${SP}_G${G} SOURCE ${PREP_CL}
                   DEFINES M=${M} K=${K} S_PREV=${SP} FOLD_G=${G})
endfunction()
function(add_prep_norm_finish M K G W)
  add_ocloc_kernel(prep_norm_finish_M${M}_K${K}_G${G}_W${W} SOURCE ${PREP_CL}
                   DEFINES M=${M} K=${K} NORM_G=${G} NORM_WGS=${W})
endfunction()
# G = 20 at K = 5120 is one element per lane in a 256-lane work-group, which is
# 320 subgroups against the single-WG kernel's 16 (docs/15 §L2: the subgroup is
# the unit that priced out, not the work-group).
add_prep_res_fold(1 5120 0 20)
add_prep_res_fold(1 5120 16 20)
add_prep_res_fold(2 5120 16 20)    # spec 1 §9: the M loop must compile at M=2
add_prep_norm_finish(1 5120 20 20)
add_prep_norm_finish(1 5120 20 1)  # the single-work-group finish, for the record
add_prep_norm_finish(2 5120 20 20)
add_ocloc_kernel(prep_silu_mul_M1 SOURCE ${PREP_CL} DEFINES M=1)
add_ocloc_kernel(prep_gated_head_M1 SOURCE ${PREP_CL} DEFINES M=1)

# The control block (runtime::Control, src/runtime/control.h) seen as a flat
# `uint` array - the kernels that touch it index it with these. This ONE line is
# where the device side's field indices live: no `.cl` file spells a number.
# The host mirror is `kernels::ctrl_index` (src/kernels/kernels.h), which
# argmax_test static_asserts against `offsetof(runtime::Control, …)`; that the
# numbers below match is proved end to end by argmax_test / embed_gather_test,
# whose every assertion reads or writes a *named struct field*.
set(CTRL_DEFINES CTRL_POS=0 CTRL_NACT=1 CTRL_CUR=2 CTRL_OUT=10 CTRL_DEBUG=18)

# gdn_step: the GDN decode step (conv1d update + SiLU, l2norm, gated delta-rule
# recurrence) - 48 of the model's 64 layers. `M` is its whole variant space:
# every other dimension is the model's and is baked into the source.
#
# NEGA_OFF / DTBIAS_OFF are the layer's GDN small-block offsets **in floats**,
# i.e. loader/small_layout.h's kGdnOffNegA (163840 B) and kGdnOffDtBias
# (164032 B) divided by 4. The kernel takes the block base as a `const float*`
# and indexes with these, so the byte offsets stay in the one header that owns
# them; small_layout.h's static_asserts are what keeps this line honest.
set(GDN_DEFINES ${CTRL_DEFINES} NEGA_OFF=40960 DTBIAS_OFF=41008)
set(GDN_CL ${CMAKE_CURRENT_SOURCE_DIR}/gdn_step.cl)
add_ocloc_kernel(gdn_step_M1 SOURCE ${GDN_CL} DEFINES ${GDN_DEFINES} M=1)
# M=2 is compiled per spec 1 §9 (the M loop must compile); it is also RUN by
# gdn_step_test, as the only cover for the conv window's intra-step path.
add_ocloc_kernel(gdn_step_M2 SOURCE ${GDN_CL} DEFINES ${GDN_DEFINES} M=2)

# embed_gather and argmax are pure compare-and-copy: no divide, no sqrt, no
# rounding, so the project-wide correctly-rounded flag changes nothing in
# their code -- which is what keeps the flag safe as a default.
add_ocloc_kernel(embed_gather_M1 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                 DEFINES ${CTRL_DEFINES} M=1 HIDDEN=5120 VOCAB=248320)

# VOCAB_USED = model::Qwen35::kVocabUsed: `lm_head` has 248320 rows because the
# tiling wants a round number, but only the first 248077 are real tokens.
set(ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=248320 VOCAB_USED=248077)
set(ARGMAX_CL ${CMAKE_CURRENT_SOURCE_DIR}/argmax.cl)
add_ocloc_kernel(argmax_stage1_M1 SOURCE ${ARGMAX_CL} DEFINES ${ARGMAX_DEFINES} M=1)
# Stage 2 has no shape to vary: it loops m < ctrl.n_active internally, so its
# binary name is the entry-point name (both entry points are in both binaries).
add_ocloc_kernel(argmax_stage2 SOURCE ${ARGMAX_CL} DEFINES ${ARGMAX_DEFINES})

# attn: the decode-attention trio (attn_prep / attn_decode / attn_reduce) - the
# 16 full-attention layers. Three entry points in one file, so each binary
# carries all three and the runtime loads the one it asked for (prep.cl's
# pattern). Two things vary:
#
#   * `M`, as everywhere - the tokens in flight;
#   * `MAXLEN`, which `attn_decode` and `attn_reduce` need because
#     `attn_part` is strided `[24][MAXLEN/256][M][258]` (the *grid* comes from
#     `buffers.max_len` at capture, but the stride has to be a compile-time
#     constant). `attn_prep` touches no `max_len`-shaped array - it indexes the
#     KV caches by absolute position - so its variants carry no `L`.
#
# QNORM_OFF / KNORM_OFF are the FA small block's offsets **in floats**, i.e.
# loader/small_layout.h's kFaOffQNorm (0 B) and kFaOffKNorm (1024 B) divided by
# 4. The kernel takes the block base as a `const float*` and indexes with these,
# so the byte offsets stay in the one header that owns them; small_layout.h's
# static_asserts are what keeps this line honest - exactly as GDN_DEFINES does.
set(ATTN_DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256)
set(ATTN_CL ${CMAKE_CURRENT_SOURCE_DIR}/attn.cl)
function(add_attn_prep MM)
  add_ocloc_kernel(attn_prep_M${MM} SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM})
endfunction()
function(add_attn_decode MM LL)
  add_ocloc_kernel(attn_decode_M${MM}_L${LL} SOURCE ${ATTN_CL}
                   DEFINES ${ATTN_DEFINES} M=${MM} MAXLEN=${LL})
endfunction()
function(add_attn_reduce MM LL)
  add_ocloc_kernel(attn_reduce_M${MM}_L${LL} SOURCE ${ATTN_CL}
                   DEFINES ${ATTN_DEFINES} M=${MM} MAXLEN=${LL})
endfunction()
add_attn_prep(1)
add_attn_decode(1 16384)   # the loader's default max_len (docs/13-loader.md)
add_attn_reduce(1 16384)
add_attn_decode(1 4096)    # attn_test's buffers: 16 blocks, cheap to fill
add_attn_reduce(1 4096)
# M = 2 is compiled per spec 1 §9 (the M loop must compile); all three are also
# RUN by attn_test, as the only cover for per-m causal masking (a position valid
# for m = 1 and masked for m = 0, both inside one block and across a block edge).
#
# Note the asymmetry, which is deliberate while M = 2 is test-only: the M = 2
# decode/reduce pair exists **at L4096 only**, the length attn_test allocates.
# There is no `attn_decode_M2_L16384` / `attn_reduce_M2_L16384`, so
# `kernels::attn_{decode,reduce}_variant(2, 16384)` names no file and the
# runtime cannot bind M = 2 attention at the loader's default max_len. Whoever
# turns M > 1 on (plan 3's MTP work) adds the L16384 rows here first.
add_attn_prep(2)
add_attn_decode(2 4096)
add_attn_reduce(2 4096)
