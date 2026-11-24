add_ocloc_kernel(noop SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/noop.cl)
add_ocloc_kernel(bw_sum SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bw_sum.cl)
add_ocloc_kernel(bw_sum_sweep SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bw_sum.cl DEFINES BW_SWEEP=1)
add_ocloc_kernel(bw_sum_lsc SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bw_sum.cl DEFINES BW_LSC=1)
add_ocloc_kernel(ctrl_read SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ctrl_read.cl)
add_ocloc_kernel(host_read SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/host_read.cl)

set(GEMV_CL ${CMAKE_CURRENT_SOURCE_DIR}/gemv.cl)
function(add_gemv_variant M K N S L)
  set(extra_defs)
  # Every M, not just M = 1 (spec 8a): the cells are per shape, the M loop is
  # independent of the load geometry, and the only M > 1 variant built by
  # default (M2 K6144 N5120 S1) matches no cell, so no default binary changes.
  # Task 4's exact production cells, copied from probe_gemv_loads.cl. The
  # variant names stay shape-only, so capture's 19-module topology is
  # unchanged; only these binaries receive the measured arithmetic/geometry.
  if(L EQUAL 0 AND K EQUAL 6144 AND N EQUAL 5120 AND S EQUAL 4)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
  elseif(L EQUAL 0 AND K EQUAL 5120 AND N EQUAL 14336 AND S EQUAL 2)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
  elseif(L EQUAL 0 AND K EQUAL 5120 AND N EQUAL 34816 AND S EQUAL 8)
    list(APPEND extra_defs GEMV_DEQ_SHIFT=1)
  elseif(L EQUAL 0 AND K EQUAL 17408 AND N EQUAL 5120 AND S EQUAL 4)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
  elseif(L EQUAL 1 AND K EQUAL 5120 AND N EQUAL 248320 AND S EQUAL 1)
    list(APPEND extra_defs GEMV_DEQ_SHIFT=1)
  # Spec 17b: a TP card's halves take their one-card cell's defines (the probe_tp step's ptp_gemv_* cells).
  elseif(L EQUAL 0 AND N EQUAL 5120 AND (K EQUAL 3072 OR K EQUAL 8704) AND (S EQUAL 2 OR S EQUAL 4) AND M EQUAL 1)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
  elseif(L EQUAL 0 AND K EQUAL 5120 AND N EQUAL 7168 AND S EQUAL 2 AND M EQUAL 1)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)
  elseif(L EQUAL 0 AND K EQUAL 5120 AND N EQUAL 17408 AND S EQUAL 8 AND M EQUAL 1)
    list(APPEND extra_defs GEMV_DEQ_SHIFT=1)
  # Plan 21-QFN-v2b Task 6: Qwen3.8-Flash-Next's M = 1 linears, one card's (pp) and a tp card's halves - few work-groups
  # with long K chains, latency bound; two k-groups in flight (probe_gemv --variants' pgv_da2 cells, byte-identical:
  # qkv||z 75.1 -> 29.8 us at the tp half, 83.1 -> 41.1 at one card's width).
  elseif(L EQUAL 0 AND M EQUAL 1 AND ((K EQUAL 2560 AND (N EQUAL 8192 OR N EQUAL 16384) AND S EQUAL 1) OR
                                      (K EQUAL 2560 AND (N EQUAL 6656 OR N EQUAL 13312) AND S EQUAL 2) OR
                                      (K EQUAL 3072 AND N EQUAL 2560 AND (S EQUAL 2 OR S EQUAL 4)) OR
                                      (K EQUAL 6144 AND N EQUAL 2560 AND (S EQUAL 4 OR S EQUAL 8))))
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1 GEMV_AHEAD=2)
  endif()
  # PERF-44: an optional TP builds the shape's cell with tp_send folded in (GEMV_TP_SEND), named _TP.
  set(suffix)
  # PERF-108: a tensor parallel card's qkv||z at every M splits each tile's K over 4 sub-groups of the work-group
  # (GEMV_KSL; kernels::gemv_ksl names it _KS4)
  if(N EQUAL 8192 AND S EQUAL 1 AND ((K EQUAL 5120 AND L EQUAL 1) OR (K EQUAL 2560 AND L EQUAL 0)))
    set(suffix _KS4)
    list(APPEND extra_defs GEMV_KSL=4)
  endif()
  if(ARGN STREQUAL "TP")
    set(suffix _TP)
    list(APPEND extra_defs GEMV_TP_SEND=1)
  # PERF-44 on Qwen3.8 tp: an optional TF builds it with the folded crossing's tp_send as its tail (TP_FOLD_SEND), _TF.
  elseif(ARGN STREQUAL "TF")
    set(suffix _TF)
    list(APPEND extra_defs TP_FOLD_SEND=1 TP_FOLD_G=20)
  endif()
  add_ocloc_kernel(gemv_M${M}_K${K}_N${N}_S${S}_L${L}${suffix} SOURCE ${GEMV_CL}
                   DEFINES M=${M} K=${K} N=${N} S=${S} LAYOUT=${L} ${extra_defs})
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
# Test-only arithmetic/load control for Task 4's tuned out/o_proj cell. It is
# never loaded by capture and therefore does not affect the runtime module
# count; gemv_test compares all 16 signed nibbles against it bit-for-bit.
add_ocloc_kernel(gemv_control_M1_K6144_N5120_S4_L0 SOURCE ${GEMV_CL}
                 DEFINES M=1 K=6144 N=5120 S=4 LAYOUT=0 GEMV_BLOCK2D=0 GEMV_DEQ_SHIFT=0)
# The SIXTH int4 shape, and the only one that is not per-layer: `lm_head` at
# 5120x248320, when the checkpoint ships it packed (spec 1.6 §5.1). Exactly one
# variant is built, not a probe row of the matrix above, because exactly one is
# bindable: layout 1 remains the measured large-N winner, and S must be 1 for
# the capture to write straight into `logits` (model/qwen35.h argues both). A
# probe sweep over this shape would cost five more ocloc compiles of a
# 3880-work-group kernel to price splits the binding cannot use.
add_gemv_variant(1 5120 248320 1 1)  # lm_head int4: 3880 work-groups, device full
# PERF-111: probe_gemv --tiles' cells, never bound by the engine - the production layout-1 cells (qkv||z, the int4
# head) with the tiles k-group-major (GEMV_KMAJOR), alone and with two k-groups in flight. `_KM` / `_KM_A2`.
add_ocloc_kernel(gemv_M1_K5120_N16384_S1_L1_KM SOURCE ${GEMV_CL}
                 DEFINES M=1 K=5120 N=16384 S=1 LAYOUT=1 GEMV_KMAJOR=1)
add_ocloc_kernel(gemv_M1_K5120_N16384_S1_L1_KM_A2 SOURCE ${GEMV_CL}
                 DEFINES M=1 K=5120 N=16384 S=1 LAYOUT=1 GEMV_KMAJOR=1 GEMV_AHEAD=2)
add_ocloc_kernel(gemv_M1_K5120_N248320_S1_L1_KM SOURCE ${GEMV_CL}
                 DEFINES M=1 K=5120 N=248320 S=1 LAYOUT=1 GEMV_DEQ_SHIFT=1 GEMV_KMAJOR=1)
add_ocloc_kernel(gemv_M1_K5120_N248320_S1_L1_KM_A2 SOURCE ${GEMV_CL}
                 DEFINES M=1 K=5120 N=248320 S=1 LAYOUT=1 GEMV_DEQ_SHIFT=1 GEMV_KMAJOR=1 GEMV_AHEAD=2)

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
# Spec 9 §3: the int8 lm_head (W8A16, quantised at load; gemv_i8w.cl). gemv_bf16's
# lm_head tiling, fixed in the source; M is the whole variant space (M = 2..4 below).
set(GEMV_I8W_CL ${CMAKE_CURRENT_SOURCE_DIR}/gemv_i8w.cl)
function(add_gemv_i8w_variant M K N)
  add_ocloc_kernel(gemv_i8w_M${M}_K${K}_N${N} SOURCE ${GEMV_I8W_CL} DEFINES M=${M} K=${K} N=${N})
endfunction()
add_gemv_i8w_variant(1 5120 248320)
# PERF-111: probe_gemv --tiles' head cells, never bound by the engine: steps k-major (_KM), with four reads in flight
# (_KM_A4), and four in flight in today's order (_A4) to tell the two apart. The same three for gemv_bf16's head.
foreach(v "KM;GEMV_KMAJOR=1" "KM_A4;GEMV_KMAJOR=1;GEMV_AHEAD=4" "A4;GEMV_AHEAD=4")
  list(GET v 0 tag)
  list(SUBLIST v 1 -1 defs)
  add_ocloc_kernel(gemv_i8w_M1_K5120_N248320_${tag} SOURCE ${GEMV_I8W_CL} DEFINES M=1 K=5120 N=248320 ${defs})
  add_ocloc_kernel(gemv_bf16_M1_K5120_N248320_${tag} SOURCE ${GEMV_BF16_CL} DEFINES M=1 K=5120 N=248320 ${defs})
endforeach()
# a‖b as the runtime binds it since spec 1.5's lever L2 (kernels::gemv_bf16_tiling):
# 8 work-groups of 16 K-slice subgroups = 128 threads, against 8. `16` is
# measured, not chosen: 1 / 4 / 16 read 48.774 / 13.115 / 5.340 us in situ
# (docs/12 `gemv_bf16`). The `{64, 1}` build above is kept and still compiled --
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
add_prep_res_norm(1 5120 4)    # Task 4: after out_proj / o_proj / down (S = 4)

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
# 320 subgroups against the single-WG kernel's 16 (docs/15, "What the measurement says that nothing else did": the subgroup is
# the unit that priced out, not the work-group).
add_prep_res_fold(1 5120 0 20)
add_prep_res_fold(1 5120 4 20)     # Task 4 production mixer/down split
add_prep_res_fold(1 5120 16 20)
# Test-only: spec 2's prefill path folds ONE slice (plan 6b ruling R1), and
# tests/prefill/pf_prep_test.cc grades `pf_res_fold` bit-identical to a decode
# binary at M = 1. That control has to exist at S_PREV = 1, which no decode site
# binds. Additive, never named by capture.cc, so the decode list stays 774/19.
add_prep_res_fold(1 5120 1 20)
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
#     `attn_part` is strided `[24][MAXLEN/ATTN_BLOCK][M][258]` (the *grid* comes
#     from `buffers.max_len` at capture, but the stride has to be a compile-time
#     constant). `attn_prep` touches no `max_len`-shaped array - it indexes the
#     KV caches by absolute position - so its variants carry no `L`.
#
# ATTN_BLOCK - KV positions per `attn_decode` work-group - is the third, and it
# is here rather than as a literal in attn.cl because the HOST has to know it
# too: `runtime::DecodeBuffers::kAttnBlock` sizes `attn_part` and sets
# `attn_decode`'s grid, and nothing can check the two against each other. So it
# goes in the NAME on both sides (`attn_decode_M1_L16384_B64`,
# `kernels::attn_decode_variant`) exactly as `prep_res_fold`'s `G` does above:
# a host that disagrees names a binary that does not exist and throws at
# capture, instead of striding `attn_part` at one size while the kernel writes
# it at another. `attn_prep` does not block, so it carries no `B` either.
#
# **64, from 256, is spec 1.5's lever L5** and the number is MEASURED, not
# extrapolated. docs/15 part 2's fit over four low-depth points predicted 214.5 us
# per launch at a 128-position block; the retile measured **296.684**, +38%, so
# that model died the way its two predecessors did -- and so did the refit that
# replaced it, which called 260.0 us at a 64-position block and measured
# 224.046, -13.8% the other way. Four models dead on this kernel, and no fifth
# is offered. Four block sizes were then
# measured in situ at depth 4096 (docs/12 "Decode attention" -> Measured, docs/15, "The three changes of 2026-08-25"),
# attn_decode us/launch and the attn FAMILY ms/token:
#
#   B256  369.988  6.058 | B128  296.684  4.931 | B64  224.046  3.839
#   B32   203.837  3.769
#
# **64 is NOT a knee** -- B32's family total is 0.070 ms/token BETTER, and the
# step-Sigma difference that once looked like a crossover (+19.5 us) is inside
# the +0.283% drift the untouched launches showed between those two runs. 64 is
# chosen on three things that do not need a sign: the marginal gain has
# collapsed (-1.127, -1.092, -0.070 ms/token for the three halvings, the last
# about a third of one run's drift); `attn_reduce`'s merge is on a steep ramp
# (80 -> 127 -> 196 -> 450 us/step, taking back 254 of attn_decode's 323 at
# B32); and `attn_part` would double again to 101.4 MB of scratch to buy that
# 0.070 ms.
#
# Both compiled lengths move together; there is no mixed-block build, because
# `kAttnBlock` is one number for the whole engine.
#
# `CACHE INTERNAL` (which implies FORCE, so editing the line still takes effect
# on the next configure) rather than a plain `set`: tests/CMakeLists.txt is a
# SIBLING directory scope and would not otherwise see it, and it must, because
# attn_test's `add_dependencies` names the very binaries this number names.
#
# **This is one of THREE homes for the number** -- here,
# `runtime::DecodeBuffers::kAttnBlock` (src/runtime/buffers.h) and
# `attn_ref::kBlock` (tests/kernels/attn_ref.h) -- and nothing checks them
# against each other at compile time. The guard is the missing-binary throw at
# capture, which needs a device and a loaded model to fire, so a CPU-only build
# never sees a disagreement. Same arrangement for FOLD_G / NORM_G above and
# `runtime::DecodeBuffers::kNormGroups`. A device-free cross-check is a spec-1.6
# item and is deliberately not added here.
set(ATTN_BLOCK 64 CACHE INTERNAL
    "KV positions per attn_decode work-group; must equal runtime::DecodeBuffers::kAttnBlock")
#
# QNORM_OFF / KNORM_OFF are the FA small block's offsets **in floats**, i.e.
# loader/small_layout.h's kFaOffQNorm (0 B) and kFaOffKNorm (1024 B) divided by
# 4. The kernel takes the block base as a `const float*` and indexes with these,
# so the byte offsets stay in the one header that owns them; small_layout.h's
# static_asserts are what keeps this line honest - exactly as GDN_DEFINES does.
# ATTN_BLOCK is in the SHARED define list, not in add_attn_decode/_reduce
# alone, because all three entry points live in one .cl file: `attn_prep`'s
# compile would otherwise trip the `#ifndef ATTN_BLOCK / #error` guard that is
# there to stop a build from silently taking a default. It is inert in that
# binary - `attn_prep` expands neither ATTN_BLOCK nor NBLOCKS nor WAVES, so an
# unused `-D` cannot reach its codegen - which is why `attn_prep` carries no
# `_B` in its name while the two kernels whose behaviour and buffer stride DO
# depend on the value carry it in theirs.
set(ATTN_DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256 ATTN_BLOCK=${ATTN_BLOCK})
set(ATTN_CL ${CMAKE_CURRENT_SOURCE_DIR}/attn.cl)
function(add_attn_prep MM)
  add_ocloc_kernel(attn_prep_M${MM} SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM})
endfunction()
function(add_attn_decode MM LL)
  add_ocloc_kernel(attn_decode_M${MM}_L${LL}_B${ATTN_BLOCK} SOURCE ${ATTN_CL}
                   DEFINES ${ATTN_DEFINES} M=${MM} MAXLEN=${LL})
endfunction()
function(add_attn_reduce MM LL)
  add_ocloc_kernel(attn_reduce_M${MM}_L${LL}_B${ATTN_BLOCK} SOURCE ${ATTN_CL}
                   DEFINES ${ATTN_DEFINES} M=${MM} MAXLEN=${LL})
endfunction()
# Zero-config cleanup: decode attention v1 (attn_decode / attn_reduce)
# is no engine's any more - v2 below is the only decode attention. These three lengths stay,
# test-only: attn_v2_test grades v2 against v1 (L4096 synthetic, L131072 on the real capture)
# and attn_test grades v1 against its host reference (L4096, M = 1 and 2). attn_prep is shared.
add_attn_prep(1)
add_attn_decode(1 131072)  # attn_v2_test part D's reference
add_attn_reduce(1 131072)
add_attn_decode(1 4096)    # attn_test's buffers: 64 blocks, cheap to fill
add_attn_reduce(1 4096)
# M = 2 is compiled per spec 1 §9 (the M loop must compile); all three are also
# RUN by attn_test, as the only cover for per-m causal masking (a position valid
# for m = 1 and masked for m = 0, both inside one block and across a block edge).
#
add_attn_prep(2)
add_attn_decode(2 4096)
add_attn_reduce(2 4096)

# Spec 10 (plan 10b): decode attention v2 (attn_v2.cl), M = 1..4, one binary per M
# holding attn_decode_v2 + attn_reduce_v2. No MAXLEN (the grid is (4, TGT) and the
# partials [24][TGT][M][258] at every max_len); TGT is the host's
# runtime::DecodeScratch::kAttnV2Blocks and is in the name (`_T32`), as ATTN_BLOCK is.
set(ATTN_V2_TGT 32 CACHE INTERNAL
    "work-groups per kv head of attn_decode_v2; must equal runtime::DecodeScratch::kAttnV2Blocks")
# v2 runs attn.cl's `exp` and 1/16 scale: exp2 with the scale folded failed the golden gate
# (cjk row 9 flips, spec 10 §8) and its build option is gone (zero-config cleanup, 2026-11-06).
# EXP2=0 stays on the command lines, inert, so no existing binary's command line moves.
set(ATTN_V2_EXP2 0)
# Plan 19v3b Task 0: the positions a KV page of the sequence slots holds (the _PG<page> twins, below and in prefill/).
set(KV_PAGE 2048 CACHE INTERNAL "positions a KV page holds; must equal runtime::kKvPage and kernels::kKvPage")
foreach(MM 1 2 3 4)
  add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2})
endforeach()

# Spec 8 (plan 8b): MTP speculative decoding. ON by default - it only ADDS binaries
# (the M = 2..4 verify lists at max_len 16384, the per-row GDN slot variants, the MTP
# head's own); every existing binary's command line is unchanged, and an engine without
# `--spec mtp` binds none of them.
option(B70_MTP "build the MTP verify / draft variants (spec 8)" ON)
if(B70_MTP)
  foreach(MM 2 3 4)
    add_gemv_variant(${MM} 5120 16384 1 1)     # qkv||z
    add_gemv_variant(${MM} 6144 5120 4 0)      # out_proj, o_proj
    add_gemv_variant(${MM} 5120 34816 8 0)     # gate||up
    add_gemv_variant(${MM} 17408 5120 4 0)     # down
    add_gemv_variant(${MM} 5120 14336 2 0)     # qkv
    add_gemv_bf16_variant(${MM} 5120 248320 64 1)  # lm_head (bf16)
    add_gemv_i8w_variant(${MM} 5120 248320)        # lm_head (int8, spec 9)
    add_gemv_bf16_variant(${MM} 5120 128 16 16)    # a||b
    add_prep_res_fold(${MM} 5120 0 20)
    add_prep_res_fold(${MM} 5120 4 20)
    add_ocloc_kernel(prep_silu_mul_M${MM} SOURCE ${PREP_CL} DEFINES M=${MM})
    add_ocloc_kernel(prep_gated_head_M${MM} SOURCE ${PREP_CL} DEFINES M=${MM})
    add_ocloc_kernel(embed_gather_M${MM} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} HIDDEN=5120 VOCAB=248320)
    add_ocloc_kernel(argmax_stage1_M${MM} SOURCE ${ARGMAX_CL} DEFINES ${ARGMAX_DEFINES} M=${MM})
    if(NOT MM EQUAL 2)   # the M = 2 ones already exist above
      add_prep_norm_finish(${MM} 5120 20 20)
      add_ocloc_kernel(gdn_step_M${MM} SOURCE ${GDN_CL} DEFINES ${GDN_DEFINES} M=${MM})
      add_attn_prep(${MM})
    endif()
  endforeach()
  # The MTP head's own linears at M = 1, bf16 (draft-step cost, timed in
  # isolation by probe_verify_m): fc, q||k||v, o, gate||up, down. All N >= 2048,
  # so the default {64, 1} tiling (kernels::gemv_bf16_tiling).
  add_gemv_bf16_variant(1 10240 5120 64 1)
  add_gemv_bf16_variant(1 5120 14336 64 1)
  add_gemv_bf16_variant(1 6144 5120 64 1)
  add_gemv_bf16_variant(1 5120 34816 64 1)
  add_gemv_bf16_variant(1 17408 5120 64 1)
endif()

# Spec 8 (plan 8b): what the MTP lists bind beyond the M = 2..4 decode list above.
#   gdn_step_slots_M<M>   gdn_step with SPEC_SLOTS: the state after row m into slot
#                         (live + m) % 4 (gdn_step.cl), M = 1..4. SPEC_SLOT_STRIDE is one
#                         slot of MtpBuffers::gdn_spec in floats, 48 x 48 x 128 x 128.
#   the head (draft at M = 1, the verify list's KV fill at M = 1..4): fc and q||k||v at
#   every M; the two pre-fc norms into one [M][10240] row (X_STRIDE); the fold that starts
#   the head's residual at fc's output (ZERO_RESID); attn_prep over an S = 1 bf16 qkv;
#   and, at M = 1 only, the draft's o / gate||up (SILU_S = 1) / down.
if(B70_MTP)
  set(GDN_SLOT_DEFINES ${GDN_DEFINES} SPEC_SLOTS=1 CTRL_LIVE=19 SPEC_SLOT_STRIDE=37748736)
  foreach(MM 1 2 3 4)
    add_ocloc_kernel(gdn_step_slots_M${MM} SOURCE ${GDN_CL} DEFINES ${GDN_SLOT_DEFINES} M=${MM})
    if(NOT MM EQUAL 1)
      add_gemv_bf16_variant(${MM} 10240 5120 64 1)   # fc
      add_gemv_bf16_variant(${MM} 5120 14336 64 1)   # q||k||v
    endif()
    # PERF-84: the head's two pre-fc norms, one fold and one finish over both sources
    add_ocloc_kernel(prep_res_fold_M${MM}_K5120_SP0_G20_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 S_PREV=0 FOLD_G=20 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_norm_finish_M${MM}_K5120_G20_W20_X10240_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 NORM_G=20 NORM_WGS=20 X_STRIDE=10240 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_res_fold_M${MM}_K5120_SP1_G20_Z SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
    add_ocloc_kernel(attn_prep_M${MM}_S1 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1)
  endforeach()
  add_ocloc_kernel(prep_silu_mul_M1_S1 SOURCE ${PREP_CL} DEFINES M=1 SILU_S=1)

  # Spec 8 §11: the draft over a reduced vocabulary V' (`b70-serve --draft-vocab`), at
  # |V'| = 32768 / 65536 / 131072 - the M = 1 GEMV over the compact head in the head's
  # form (gemv_i8w.cl / gemv_bf16.cl unchanged, a new N only; bf16 at lm_head's {64, 1}
  # tiling, which kernels::gemv_bf16_tiling gives every N >= 2048) and the mapped argmax +
  # scatter (draft_vocab.cl), one binary per size holding both stages. Additive: no
  # existing binary's command line moves, and an engine without a draft vocabulary binds
  # none of them. The head's K is the model's hidden (5120 on Qwen3.8).
  set(DRAFT_VOCAB_CL ${CMAKE_CURRENT_SOURCE_DIR}/draft_vocab.cl)
  foreach(NV 32768 65536 131072)
    add_gemv_i8w_variant(1 5120 ${NV})
    add_gemv_bf16_variant(1 5120 ${NV} 64 1)
    add_ocloc_kernel(dv_argmax_N${NV} SOURCE ${DRAFT_VOCAB_CL} DEFINES ${CTRL_DEFINES} NV=${NV})
  endforeach()

  # Open-work 13: the draft's probability (draft_prob.cl, both stages in one binary) after the draft list's argmax -
  # draft_prob_N<row length>_U<real ids> (kernels::draft_prob_variant): the whole head's row here, a TP card's half in
  # the TP block. CTRL_DSTAT = runtime::Control::draft_stat (kernels::ctrl_index::kDraftStat). Additive only.
  set(DRAFT_PROB_CL ${CMAKE_CURRENT_SOURCE_DIR}/draft_prob.cl)
  add_ocloc_kernel(draft_prob_N248320_U248077 SOURCE ${DRAFT_PROB_CL} DEFINES CTRL_DSTAT=28 VOCAB=248320 VOCAB_USED=248077)
endif()

# ==== Spec 19a Task 4: the verify cost at M = 5..8 and DFlash's draft cost ==== (begin) ==
# Additive only: no
# existing variant's name or command line moves (tools/kernel_cmdlines' diff: additions only),
# and no engine binds any of these - only the two probes do:
#   tools/probe/probe_mtp_steps <snap> ... <max_m = 8>   Qwen3.8's verify list at M = 5..8
#       (runtime::build_verify over an 8-slot MtpBuffers the probe allocates; the engine's
#       own lists stay M = 1..4 over 4 slots), int8 lm_head, bf16 KV, attention v2.
#   tools/probe/probe_draft_cost                          DFlash2's drafter linears at M = 8
#       rows, int8 / bf16 / int4 g64, random weights of the checkpoint's shapes, and the
#       int8 head's K = 7 rows (full vocabulary and a 32k draft vocabulary).
# Two source edits make M = 5..8 compile, both invisible to every existing command line
# (their preprocessed sources are unchanged): gdn_step.cl's N_SLOTS became overridable
# (`_N8` here: the state after row m into slot (live + m) % 8, SPEC_SLOT_STRIDE one
# gdn_state as before), and attn_v2.cl's bound moved from M <= 4 to M <= 8.
# Untuned: the int4 GEMV cells at M = 5..8 take add_gemv_variant's per-shape cells (the
# M loop is independent of the load geometry); attention v2 at M = 8 holds 48 (q head,
# row) pairs per lane and may spill - the probe measures what these binaries cost.
option(B70_VERIFY_M8 "build the M = 5..8 verify and DFlash draft-cost probe variants (spec 19a)" ON)
if(B70_VERIFY_M8 AND B70_MTP)
  set(VM8_SLOT_DEFINES ${GDN_DEFINES} SPEC_SLOTS=1 CTRL_LIVE=19 SPEC_SLOT_STRIDE=37748736 N_SLOTS=8)
  # M = 16: open-work 28's wide list (its own ids, argmax and attention in its block below) shares the rest.
  foreach(MM 5 6 7 8 16)
    if(MM LESS 16)
      # the decode list at M rows (B70_MTP's M = 2..4 block's rows, int8 lm_head only)
      add_gemv_variant(${MM} 5120 16384 1 1)     # qkv||z
      add_gemv_variant(${MM} 6144 5120 4 0)      # out_proj, o_proj
      add_gemv_variant(${MM} 5120 34816 8 0)     # gate||up (and the drafter's, M = 8)
      add_gemv_variant(${MM} 17408 5120 4 0)     # down (and the drafter's, M = 8)
      add_gemv_variant(${MM} 5120 14336 2 0)     # qkv
      add_gemv_i8w_variant(${MM} 5120 248320)    # lm_head (int8; at M = 7 also the draft head)
      add_ocloc_kernel(embed_gather_M${MM} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} HIDDEN=5120 VOCAB=248320)
      add_ocloc_kernel(argmax_stage1_M${MM} SOURCE ${ARGMAX_CL} DEFINES ${ARGMAX_DEFINES} M=${MM})
      # the verify list's own 8-slot gdn_step
      add_ocloc_kernel(gdn_step_slots_M${MM}_N8 SOURCE ${GDN_CL} DEFINES ${VM8_SLOT_DEFINES} M=${MM})
    endif()
    add_gemv_bf16_variant(${MM} 5120 128 16 16)    # a||b
    add_prep_res_fold(${MM} 5120 0 20)
    add_prep_res_fold(${MM} 5120 4 20)
    add_prep_norm_finish(${MM} 5120 20 20)
    add_ocloc_kernel(prep_silu_mul_M${MM} SOURCE ${PREP_CL} DEFINES M=${MM})
    add_ocloc_kernel(prep_gated_head_M${MM} SOURCE ${PREP_CL} DEFINES M=${MM})
    add_attn_prep(${MM})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2})
    # the MTP head's KV fill of M rows
    add_gemv_bf16_variant(${MM} 10240 5120 64 1)   # fc
    add_gemv_bf16_variant(${MM} 5120 14336 64 1)   # q||k||v
    # PERF-84: the head's two pre-fc norms, one fold and one finish over both sources
    add_ocloc_kernel(prep_res_fold_M${MM}_K5120_SP0_G20_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 S_PREV=0 FOLD_G=20 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_norm_finish_M${MM}_K5120_G20_W20_X10240_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 NORM_G=20 NORM_WGS=20 X_STRIDE=10240 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_res_fold_M${MM}_K5120_SP1_G20_Z SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
    add_ocloc_kernel(attn_prep_M${MM}_S1 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1)
  endforeach()
  # DFlash2's drafter (z-lab/Qwen3.8-27B-DFlash2's config / safetensors header: 5 layers,
  # hidden 5120, 32 q / 8 kv heads x 128, intermediate 17408, fc 25600 -> 5120, two
  # 5120 -> 1280 conv kernel projections per layer, the selector's 5120 -> 256) at M = 8
  # rows (the block of 8), in the three formats; the shapes are probe_draft_cost's table
  # (tools/probe/draft_cost_shapes.h). int4 {S, layout}: gate||up and down are Qwen3.8's
  # production cells (built above at M = 8); the rest are PROVISIONAL picks, not swept.
  foreach(KN "5120 6144" "4096 5120" "5120 34816" "17408 5120" "5120 1280" "5120 2048"
             "25600 5120" "5120 256")
    separate_arguments(KN)
    list(GET KN 0 _k)
    list(GET KN 1 _n)
    add_gemv_i8w_variant(8 ${_k} ${_n})
    if(_n LESS 2048)   # kernels::gemv_bf16_tiling: N / 64 < 32 work-groups -> {16, 16}
      add_gemv_bf16_variant(8 ${_k} ${_n} 16 16)
    else()
      add_gemv_bf16_variant(8 ${_k} ${_n} 64 1)
    endif()
  endforeach()
  add_gemv_variant(8 5120 6144 4 0)     # q||k||v        PROVISIONAL S4 L0
  add_gemv_variant(8 4096 5120 4 0)     # o_proj         PROVISIONAL S4 L0
  add_gemv_variant(8 5120 1280 16 0)    # conv kernel projections  PROVISIONAL S16 L0
  add_gemv_variant(8 5120 2048 8 0)     # the commit's k||v of the context rows  PROVISIONAL S8 L0
  add_gemv_variant(8 25600 5120 8 0)    # fc             PROVISIONAL S8 L0
  add_gemv_variant(8 5120 256 16 0)     # the selector's hidden projection  PROVISIONAL S16 L0
  add_gemv_i8w_variant(7 5120 32768)    # the int8 head's K = 7 rows over a 32k draft vocabulary
endif()
# ==== Spec 19a Task 4 ======================================================== (end) ====

# ==== Spec 19b: the DFlash drafter on the card ==== (begin) ====
# Additive only: no existing variant's name or
# command line moves, and no engine binds any of these without a drafter (19c's --spec dflash). The draft list and
# the projection list (runtime/dflash/dflash_capture.cc) bind, at z-lab/Qwen3.8-27B-DFlash2's shapes beside Qwen3.8
# (kernels::dflash::kQwen38, src/kernels/dflash_kernels.h):
#   df_prep.cl    df_embed, df_norm, df_conv_prep / df_conv_add, df_add, df_silu_mul, df_qkv_prep, df_ctx_store
#   df_attn.cl    df_attn_part (grid (8 kv heads, 16 blocks of 128 ring offsets)) + df_attn_reduce
#   df_topk.cl    the 16 best head logits per draft row, two stages; the full head masks ids >= 248077, a draft
#                 vocabulary's (COMPACT) maps its indices through the id table
#   df_select.cl  DFlash 2's candidate-selector walk (V2), or v1's per-row pick
#   df_proj_wide.cl  the projection at prefill width: df_w_slab_i8 / df_w_slab_bf16, df_w_norm, df_w_ctx_store
# and the reused GEMVs: the drafter's linears at M = 8 (B70_VERIFY_M8's drafter shapes - built there - plus the
# context k||v at N 2048), the target's head at M = 7 rows (int8 from B70_VERIFY_M8; the bf16 head and the compact
# heads at 64k / 128k / bf16 added here). The block reads the ring (the projection's bf16 rows) and its own fp32 K / V.
option(B70_DFLASH "build the DFlash drafter variants (spec 19b)" ON)
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  set(DCTL_DEFINES DCTL_K=0 DCTL_PPOS=1 DCTL_PN=2 DCTL_TEMP=3 DCTL_SEED_LO=4 DCTL_SEED_HI=5)
  set(DF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dflash)
  add_ocloc_kernel(df_prep_M8_H5120_I17408_Q32KV8_D128_G16T2_R2048_V248320 SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 H=5120 WG=256 EPS=1e-06f VOCAB=248320 GROUP=16 CTAPS=2
                           INTER=17408 NH=32 NKV=8 HD=128 RING=2048)
  add_ocloc_kernel(df_prep_M8_H5120_I17408_Q32KV8_D128_G16T2_R2048_V248320_F8Q4R4U8C16H16K8 SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} FOLD=1 M=8 H=5120 WG=256 EPS=1e-06f VOCAB=248320 GROUP=16 CTAPS=2
                           INTER=17408 NH=32 NKV=8 HD=128 RING=2048)
  # The prompt's staged rows projected at prefill width (DFlashDrafter::project_wide): slabs for pf_gemm, the norm
  # and the ring store at any row count.
  add_ocloc_kernel(df_proj_wide_H5120_KV8_D128_R2048 SOURCE ${DF_DIR}/df_proj_wide.cl
                   DEFINES H=5120 WG=256 EPS=1e-06f NKV=8 HD=128 RING=2048)
  add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128 SOURCE ${DF_DIR}/df_attn.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 WGA=128
                           CAUSAL=0)
  foreach(RPW 1 2 4)
    add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128_H1_B${RPW} SOURCE ${DF_DIR}/df_attn_rows.cl
                     DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 WGA=128
                             CAUSAL=0 HPW=1 RPW=${RPW})
  endforeach()
  # Spec 19 v3d Task 6: df_attn_part on DPAS (df_attn_dpas.cl); its reduce is the _H1_B4 binary's.
  add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas SOURCE ${DF_DIR}/df_attn_dpas.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128)
  foreach(KSP 2 4)   # the occupancy forms: the block's keys split KSP ways a head
    add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas_k${KSP} SOURCE ${DF_DIR}/df_attn_dpas.cl
                     DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 KS=${KSP})
  endforeach()
  # Plan 19v3b: the KS 4 DPAS part at 256 GRF a thread (-cl-intel-256-GRF-per-thread: the same arithmetic, bitwise the
  # 128-GRF binary above, 0.535 -> 0.446 ms the five parts) - production; and the row-table forms of the batched draft
  # (SEQS: up to 4 blocks of rows, each its own slot's ring and position) - the part at KS 1 (fastest at 2 and 4
  # blocks), its one-ring twin (tests) and the H1 B4 reduce.
  add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas_k4_g256 SOURCE ${DF_DIR}/df_attn_dpas.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 KS=4
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas_g256 SOURCE ${DF_DIR}/df_attn_dpas.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 KS=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas_g256_SEQS SOURCE ${DF_DIR}/df_attn_dpas.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 KS=1 SEQS=4
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128_H1_B4_SEQS SOURCE ${DF_DIR}/df_attn_rows.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 WGA=128
                           CAUSAL=0 HPW=1 RPW=4 SEQS=4)
  add_ocloc_kernel(df_topk_R7_N248320_U248077 SOURCE ${DF_DIR}/df_topk.cl DEFINES R=7 N=248320 USED=248077 TOPK=16
                   CH=4096 COMPACT=0)
  foreach(NV 32768 65536 131072)
    add_ocloc_kernel(df_topk_R7_N${NV}_U${NV}_C SOURCE ${DF_DIR}/df_topk.cl DEFINES R=7 N=${NV} USED=${NV} TOPK=16
                     CH=4096 COMPACT=1)
    add_gemv_bf16_variant(7 5120 ${NV} 64 1)          # the compact bf16 head at the head's 7 rows
    if(NOT NV EQUAL 32768)                            # 32768 is B70_VERIFY_M8's
      add_gemv_i8w_variant(7 5120 ${NV})
    endif()
  endforeach()
  add_ocloc_kernel(df_candidate_R7_N248320_U248077_K64 SOURCE ${DF_DIR}/df_candidate.cl
                   DEFINES R=7 N=248320 USED=248077)
  add_ocloc_kernel(df_rerank_R7_K5120_U248077_C64_Q4 SOURCE ${DF_DIR}/df_rerank.cl DEFINES K=5120 USED=248077)
  # Spec 19 v3d Task 7: 80 candidates a row - the 2-bit head's default (64: the int4 head's)
  add_ocloc_kernel(df_candidate_R7_N248320_U248077_K80 SOURCE ${DF_DIR}/df_candidate.cl
                   DEFINES R=7 N=248320 USED=248077 C=80)
  add_ocloc_kernel(df_rerank_R7_K5120_U248077_C80_Q4 SOURCE ${DF_DIR}/df_rerank.cl DEFINES K=5120 USED=248077 C=80)
  add_ocloc_kernel(df_select_M8_K16_Q16_RK256_V248320 SOURCE ${DF_DIR}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} MAXR=8 TOPK=16 QS=16 V2=1 RANK=256 VROWS=248320)
  add_ocloc_kernel(df_select_M8_K16_Q16_RK256_V248320_H16 SOURCE ${DF_DIR}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} FOLD=1 MAXR=8 TOPK=16 QS=16 V2=1 RANK=256 VROWS=248320)
  # Plan 19v3b: the batched draft's row-table forms (SEQS) of the block-aware kernels - df_prep (embed, the
  # convolutions, qkv prep; both forms) and the DFlash 2 selector walk (both forms).
  add_ocloc_kernel(df_prep_M8_H5120_I17408_Q32KV8_D128_G16T2_R2048_V248320_SEQS SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 H=5120 WG=256 EPS=1e-06f VOCAB=248320 GROUP=16 CTAPS=2
                           INTER=17408 NH=32 NKV=8 HD=128 RING=2048 SEQS=4)
  add_ocloc_kernel(df_prep_M8_H5120_I17408_Q32KV8_D128_G16T2_R2048_V248320_F8Q4R4U8C16H16K8_SEQS
                   SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} FOLD=1 M=8 H=5120 WG=256 EPS=1e-06f VOCAB=248320 GROUP=16 CTAPS=2
                           INTER=17408 NH=32 NKV=8 HD=128 RING=2048 SEQS=4)
  add_ocloc_kernel(df_select_M8_K16_Q16_RK256_V248320_SEQS SOURCE ${DF_DIR}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} MAXR=8 TOPK=16 QS=16 V2=1 RANK=256 VROWS=248320 SEQS=4)
  add_ocloc_kernel(df_select_M8_K16_Q16_RK256_V248320_H16_SEQS SOURCE ${DF_DIR}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} FOLD=1 MAXR=8 TOPK=16 QS=16 V2=1 RANK=256 VROWS=248320 SEQS=4)
  # Plan 40a Task 4: the batched draft at four blocks of up to 8 rows (M 32, kernels::dflash::kSeqsRows) for the int4
  # drafter with the candidate head - df_prep, the DPAS part (a block's DPAS rows at its row0) and its reduce, the
  # selector walk (MAXR the readback's 8, ROWS the table's 32). The candidate select / rerank are row-wise (their R
  # is the grid's): the R 7 binaries at 31 rows.
  add_ocloc_kernel(df_prep_M32_H5120_I17408_Q32KV8_D128_G16T2_R2048_V248320_F8Q4R4U8C16H16K8_SEQS
                   SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} FOLD=1 M=32 H=5120 WG=256 EPS=1e-06f VOCAB=248320 GROUP=16
                           CTAPS=2 INTER=17408 NH=32 NKV=8 HD=128 RING=2048 SEQS=4)
  add_ocloc_kernel(df_attn_M32_Q32KV8_D128_R2048_W2048_P128_dpas_g256_SEQS SOURCE ${DF_DIR}/df_attn_dpas.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=32 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 KS=1 SEQS=4
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(df_attn_M32_Q32KV8_D128_R2048_W2048_P128_H1_B4_SEQS SOURCE ${DF_DIR}/df_attn_rows.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=32 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 WGA=128
                           CAUSAL=0 HPW=1 RPW=4 SEQS=4)
  add_ocloc_kernel(df_select_M32_K16_Q16_RK256_V248320_H16_SEQS SOURCE ${DF_DIR}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} FOLD=1 MAXR=8 ROWS=32 TOPK=16 QS=16 V2=1 RANK=256 VROWS=248320
                           SEQS=4)
  add_ocloc_kernel(df_select_v1_M8_K16 SOURCE ${DF_DIR}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} MAXR=8 TOPK=16 QS=16 V2=0)
  add_gemv_bf16_variant(7 5120 248320 64 1)           # the full bf16 head at 7 rows
endif()
# ==== Spec 19b ==== (end) ====

# ==== Spec 19c: DFlash in the decode loop ==== (begin) ====
# Additive only: no existing variant's name or
# command line moves, and no engine binds any of these without a drafter (`--spec dflash`). With a drafter the Engine's
# verify lists (runtime::build_verify_taps) run M = 1..8 rows over 8 GDN state slots, so:
#   gdn_step_slots_M<M>_N8, M = 1..4   the per-row slot build at the rows spec 8's lists had at 4 slots (B70_VERIFY_M8
#                                      built M = 5..8 at 8)
#   gemv_bf16_M<M>_K5120_N248320       the bf16 lm_head at M = 5, 6, 8 (7 is 19b's draft head) - `--lm-head bf16`
# Every other binary the lists bind is spec 8's (M = 1..4) or B70_VERIFY_M8's (M = 5..8).
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  foreach(MM 1 2 3 4)
    add_ocloc_kernel(gdn_step_slots_M${MM}_N8 SOURCE ${GDN_CL} DEFINES ${VM8_SLOT_DEFINES} M=${MM})
  endforeach()
  # verify-cost-m8: the one-card verify lists' DEFERRED build (gdn_step.cl): no per-row state, the committed rows
  # replayed by the next launch. CTRL_PEND = Control::gdn_pend; PEND_PAR = one parity of the pending rows, 48 layers x
  # (64 rows x (16 + 48) x 128 bf16 + 64 x 48 x 2 fp32) bytes (kernels::kGdnPendParityBytes; 64 rows since spec 32, 16 since open-work
  # 28). Qwen3.8's shapes only. M = 16: open-work 28's wide verify list.
  foreach(MM 1 2 3 4 5 6 7 8 16)
    add_ocloc_kernel(gdn_step_defer_M${MM} SOURCE ${GDN_CL}
                     DEFINES ${GDN_DEFINES} DEFERRED=1 CTRL_PEND=30 PEND_PAR=51511296 M=${MM})
  endforeach()
  # verify-cost-m8 probe: a||b at M = 2..8 with a wider in-work-group K split (CaptureOptions::probe_ab_ksplit)
  foreach(MM 2 3 4 5 6 7 8)
    foreach(SS 32 64)
      add_gemv_bf16_variant(${MM} 5120 128 16 ${SS})
    endforeach()
  endforeach()
  # verify-cost-m8 probe only: the slot build storing row 0's state alone (PROBE_ONE_STORE); no engine binds it
  foreach(MM 1 2 3 4 5 6 7 8)
    add_ocloc_kernel(gdn_step_slots_M${MM}_N8_P1 SOURCE ${GDN_CL} DEFINES ${VM8_SLOT_DEFINES} M=${MM} PROBE_ONE_STORE=1)
  endforeach()
  foreach(MM 5 6 8)
    add_gemv_bf16_variant(${MM} 5120 248320 64 1)
  endforeach()
endif()
# ==== Spec 19c ==== (end) ====

# --- Spec 15b: other models' shapes ------------------------------------------
# Every binary above is built at Qwen3.8's shapes - hidden 5120, FA 24 q / 4 kv
# heads, GDN 16 k / 48 v heads - which the .cl sources bake by default. The
# host names a binary from the model descriptor (src/kernels/shape_suffix.h):
# Qwen3.8's shapes carry no suffix, so every name and
# command line here is unchanged. Another model's kernels that bake a shape
# their name does not otherwise carry are suffixed `_D<hidden>` (embed_gather),
# `_GK<k>V<v>` (gdn_step*, prep_gated_head) and `_Q<q>KV<kv>` (attn_prep,
# attn_decode / attn_reduce, attn_v2); GEMV / prep_res_* names carry K and N
# already. Ornith's (model::ornith(), 2048 / 16 q 2 kv / 16 k 32 v) are built by
# the spec 15c block below, with the .cl defines that make those constants
# overridable (GDN_K_HEADS / GDN_V_HEADS, FA_Q_HEADS / FA_KV_HEADS); any other
# shape's missing binary throws at capture, as for any unbuilt variant.
#
# --- Spec 15c: Ornith 1.5 35B-A3B (model::ornith()), the M = 1 decode list ------
# Additive only: every binary above keeps its name and command
# line (tools/kernel_cmdlines diff), and the .cl edits that made the head counts
# overridable leave the default token stream untouched (the shape defines below are
# the only way into the new branches). What Ornith's decode list binds
# (runtime::build, capture.cc): 526 launches = embed + 40 x 13 + 5.
#   linears   q||k||v / qkv||z / out_proj = o_proj at hidden 2048 - PROVISIONAL cells,
#             model_desc.cc's copied {S, layout} (no extra GEMV_* defines: P0 measures)
#   a||b      gemv_bf16 at {16, 16} (64 real columns, padded to 128 as Qwen3.8's 96) for a
#             checkpoint that keeps in_proj_a / in_proj_b bf16; the published int4 checkpoint
#             (urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ) quantised them, so its a||b is
#             gemv.cl at K 2048, N 128 (the same zero padding), S 1 L1 - PROVISIONAL (spec 15
#             §13; model::ModelDesc::ab picks by the checkpoint's content)
#   router    gemv_bf16 over the 256 router rows || the shared gate, 272 columns
#   lm_head   bf16, int8 (spec 9) or a checkpoint's int4, at K 2048
#   the norm pair at K 2048: SP0 after layer 0's embedding and after every MoE block
#             (moe_down folds into the residual itself), SP4 after out_proj / o_proj
#   GDN       gdn_step / prep_gated_head at 16 k / 32 v heads; the small block's
#             negA / dt_bias at 131072 / 131200 B = 32768 / 32800 floats
#             (loader::make_small_layout(2048, 8192, 32), checked by model_desc_test)
#   FA        attn_prep and decode attention v2 at 16 q / 2 kv heads (GQA 8); v1
#             (removed) was never built for Ornith
#   MoE       moe.cl's three kernels: 256 experts, top-8, hidden 2048, expert 512
# argmax_stage1_M1_V248070: built for the base checkpoint's tokenizer.json (its last id is
# </think> = 248069). NOT BOUND since spec 15 §13: the int4 checkpoint's tokenizer.json
# defines Qwen3.8's 248077 ids, so Ornith's greedy argmax is Qwen3.8's argmax_stage1_M<M>
# (model::ornith().vocab_used). Kept so no command line disappears.
option(B70_ORNITH "build the Ornith 1.5 35B-A3B decode variants (spec 15c)" ON)
if(B70_ORNITH)
  add_gemv_variant(1 2048 12288 1 1)    # qkv||z      PROVISIONAL S1 L1
  add_gemv_variant(1 4096 2048 4 0)     # out_proj, o_proj  PROVISIONAL S4 L0
  add_gemv_variant(1 2048 9216 2 0)     # q||k||v     PROVISIONAL S2 L0
  add_gemv_variant(1 2048 248320 1 1)   # lm_head int4 (a --quant_lm_head checkpoint)
  add_gemv_variant(1 2048 128 1 1)      # a||b int4 (the int4 checkpoint's)  PROVISIONAL S1 L1
  add_gemv_bf16_variant(1 2048 128 16 16)
  add_gemv_bf16_variant(1 2048 272 16 16)
  add_gemv_bf16_variant(1 2048 248320 64 1)
  add_gemv_i8w_variant(1 2048 248320)
  add_prep_res_fold(1 2048 0 20)
  add_prep_res_fold(1 2048 4 20)
  add_prep_norm_finish(1 2048 20 20)
  add_ocloc_kernel(embed_gather_M1_D2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                   DEFINES ${CTRL_DEFINES} M=1 HIDDEN=2048 VOCAB=248320)
  set(ORNITH_GDN_HEADS GDN_K_HEADS=16 GDN_V_HEADS=32)
  add_ocloc_kernel(gdn_step_M1_GK16V32 SOURCE ${GDN_CL}
                   DEFINES ${CTRL_DEFINES} NEGA_OFF=32768 DTBIAS_OFF=32800 ${ORNITH_GDN_HEADS} M=1)
  add_ocloc_kernel(prep_gated_head_M1_GK16V32 SOURCE ${PREP_CL} DEFINES M=1 ${ORNITH_GDN_HEADS})
  set(ORNITH_FA_HEADS FA_Q_HEADS=16 FA_KV_HEADS=2)
  add_ocloc_kernel(attn_prep_M1_Q16KV2 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=1 ${ORNITH_FA_HEADS})
  add_ocloc_kernel(attn_v2_M1_T${ATTN_V2_TGT}_Q16KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} ${ORNITH_FA_HEADS})
  # UP_KS / DN_KS: the in-work-group K splits, mirrored by kernels::kMoeUpKs / kMoeDnKs
  # (they set the work-group sizes the capture binds). PROVISIONAL, for P0 to tune.
  set(MOE_DEFINES UP_KS=4 DN_KS=2)
  set(ORNITH_ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=248320 VOCAB_USED=248070)
  add_ocloc_kernel(argmax_stage1_M1_V248070 SOURCE ${ARGMAX_CL} DEFINES ${ORNITH_ARGMAX_DEFINES} M=1)
  add_ocloc_kernel(moe_M1_E256_T8_D2048_I512 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/moe.cl
                   DEFINES ${MOE_DEFINES} M=1 EXPERTS=256 TOP_K=8 HIDDEN=2048 INTER=512 ROUTER_N=272)
endif()

# --- Spec 12b: the int8 KV cache (`--kv-cache int8`) -------------------------
# The rotkv scheme (spec 12 §8): every binary that writes or reads the int8
# cache, from ONE new source (kv8.cl, per-family defines as pf_int8.cl). Additive only:
# no existing source changes, so every existing binary keeps its name, its command line
# and its bytes, and a `--kv-cache bf16` engine (the default) binds none of these.
#   attn_prep_M<M>_KV8        the decode writer at M = 1..8 (QKV_S = 2: the main FA layers)
#   attn_prep_M<M>_S1_KV8     the MTP head's writer (its bf16 qkv: QKV_S = 1), B70_MTP
#   attn_v2_M<M>_T<TGT>_KV8   decode attention v2 over int8 + scales, the un-rotating reduce
# The prefill three are in prefill/CMakeLists.txt. Shapes are Qwen3.8's;
# another model's FA heads name a `_Q<q>KV<kv>` binary nothing builds yet.
option(B70_KV8 "build the --kv-cache int8 variants (spec 12b)" ON)
if(B70_KV8)
  set(KV8_CL ${CMAKE_CURRENT_SOURCE_DIR}/kv8.cl)
  set(KV8_PREP_DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256 KV8_PREP=1)
  foreach(MM 1 2 3 4)
    add_ocloc_kernel(attn_prep_M${MM}_KV8 SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=2)
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_KV8 SOURCE ${KV8_CL}
                     DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT})
    if(B70_MTP)
      add_ocloc_kernel(attn_prep_M${MM}_S1_KV8 SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=1)
    endif()
  endforeach()
  if(B70_VERIFY_M8)
    foreach(MM 5 6 7 8)
      add_ocloc_kernel(attn_prep_M${MM}_KV8 SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=2)
      add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_KV8 SOURCE ${KV8_CL}
                       DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT})
    endforeach()
  endif()
endif()

# ==== Spec 17b: tensor parallel decode over two cards (`--parallel tp`, Qwen3.8) ============= (begin) ====
# Additive only: new names, and the .cl edits (kv8.cl's FA head override, prep.cl's INTERMEDIATE 8704) leave the
# default token streams untouched. What a card's M = 1 decode list binds (runtime::build_tp, capture.cc) at
# model::tp_card_desc's shapes - GDN 8 k / 24 v heads (conv 5120: NEGA_OFF / DTBIAS_OFF = make_small_layout(5120, 5120,
# 24)'s 81920 / 82016 B), FA 12 q / 2 kv heads, MLP 8704, the head's 124160 rows:
#   linears  qkv||z 5120 x 8192 S1 L1, q||k||v 5120 x 7168 S2 L0, gate||up 5120 x 17408 S8 L0 (column-parallel: the
#            one-card cells' S, layout and GEMV_* defines); out_proj / o_proj 3072 x 5120 and down 8704 x 5120 at S2
#            (slices: the one-card S / 2) and S4 (folded), L0 with the one-card cells' defines (add_gemv_variant);
#            the int8 head 5120 x 124160; a||b is the one-card binary (128 columns, 48 real)
#   prep     prep_silu_mul_M1_I8704, prep_gated_head_M1_GK8V24; the norm pair's SP0 fold and finish are one card's
#   GDN, FA  gdn_step_M1_GK8V24, attn_prep_M1_Q12KV2 / attn_v2_M1_T32_Q12KV2 and their _KV8 twins
#   tp.cl    tp_cross (tp_send + tp_res_fold) per rank and form, tp_argmax per vocabulary half
#   PERF-44  the folded form's lists bind out_proj / o_proj / down with tp_send as their tail (gemv _TF, gemm_sm _tf at
#            the verify rows: TP_FOLD_SEND, tp_fold_send.h - the last work-group under a chunk sends it); tp_send is
#            left to the slices form and the DFlash drafter
option(B70_TP "build the tensor-parallel decode variants (spec 17b)" ON)
if(B70_TP)
  set(TP_CL ${CMAKE_CURRENT_SOURCE_DIR}/tp.cl)
  add_gemv_variant(1 5120 8192 1 1)     # qkv||z, a card's heads
  add_gemv_variant(1 5120 7168 2 0)     # q||k||v
  add_gemv_variant(1 5120 17408 8 0)    # gate||up, the card's half
  foreach(S 2 4)                        # slices (S2) / folded (S4)
    add_gemv_variant(1 3072 5120 ${S} 0)    # out_proj, o_proj: the card's K rows
    add_gemv_variant(1 8704 5120 ${S} 0)    # down
  endforeach()
  # PERF-44: what the folded lists bind - the two with the folded tp_send as their tail (the plain S4 pair is
  # tp_fold_send_test's control)
  add_gemv_variant(1 3072 5120 4 0 TF)
  add_gemv_variant(1 8704 5120 4 0 TF)
  add_gemv_i8w_variant(1 5120 124160)   # the int8 head's vocabulary half
  add_ocloc_kernel(prep_silu_mul_M1_I8704 SOURCE ${PREP_CL} DEFINES M=1 INTERMEDIATE=8704)
  set(TP_GDN_HEADS GDN_K_HEADS=8 GDN_V_HEADS=24)
  add_ocloc_kernel(prep_gated_head_M1_GK8V24 SOURCE ${PREP_CL} DEFINES M=1 ${TP_GDN_HEADS})
  add_ocloc_kernel(gdn_step_M1_GK8V24 SOURCE ${GDN_CL}
                   DEFINES ${CTRL_DEFINES} NEGA_OFF=20480 DTBIAS_OFF=20504 ${TP_GDN_HEADS} M=1)
  set(TP_FA_HEADS FA_Q_HEADS=12 FA_KV_HEADS=2)
  add_ocloc_kernel(attn_prep_M1_Q12KV2 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=1 ${TP_FA_HEADS})
  add_ocloc_kernel(attn_v2_M1_T${ATTN_V2_TGT}_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} ${TP_FA_HEADS})
  if(B70_KV8)
    add_ocloc_kernel(attn_prep_M1_Q12KV2_KV8 SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} M=1 QKV_S=2 ${TP_FA_HEADS})
    add_ocloc_kernel(attn_v2_M1_T${ATTN_V2_TGT}_Q12KV2_KV8 SOURCE ${KV8_CL}
                     DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=1 TGT=${ATTN_V2_TGT} ${TP_FA_HEADS})
  endif()
  # tp_cross_M<M>_K<hidden>_G<fold groups>_S<local slices>_{X slices|F folded}_R<rank> (kernels::tp_cross_variant).
  foreach(R 0 1)
    add_ocloc_kernel(tp_cross_M1_K5120_G20_S2_X_R${R} SOURCE ${TP_CL}
                     DEFINES TP_CROSS=1 M=1 K=5120 FOLD_G=20 S_LOC=2 FOLDED=0 RANK=${R})
    add_ocloc_kernel(tp_cross_M1_K5120_G20_S4_F_R${R} SOURCE ${TP_CL}
                     DEFINES TP_CROSS=1 M=1 K=5120 FOLD_G=20 S_LOC=4 FOLDED=1 RANK=${R})
  endforeach()
  # tp_argmax_M<M>_N<rows>_U<real rows>_B<first id> (kernels::tp_argmax_variant): card 0 ids 0..124159 (all real),
  # card 1 ids 124160..248319 of which 248077 - 124160 = 123917 real (argmax.cl's VOCAB_USED mask, per half).
  set(TP_ARGMAX_DEFINES ${CTRL_DEFINES} CTRL_OUTV=20 TP_ARGMAX=1 M=1 VOCAB=124160)
  add_ocloc_kernel(tp_argmax_M1_N124160_U124160_B0 SOURCE ${TP_CL}
                   DEFINES ${TP_ARGMAX_DEFINES} VOCAB_USED=124160 ID_BASE=0)
  add_ocloc_kernel(tp_argmax_M1_N124160_U123917_B124160 SOURCE ${TP_CL}
                   DEFINES ${TP_ARGMAX_DEFINES} VOCAB_USED=123917 ID_BASE=124160)
  set(B70_TP_KERNELS
      kernel_gemv_M1_K5120_N8192_S1_L1_KS4 kernel_gemv_M1_K5120_N7168_S2_L0 kernel_gemv_M1_K5120_N17408_S8_L0
      kernel_gemv_M1_K3072_N5120_S2_L0 kernel_gemv_M1_K8704_N5120_S2_L0 kernel_gemv_M1_K3072_N5120_S4_L0
      kernel_gemv_M1_K8704_N5120_S4_L0 kernel_gemv_M1_K3072_N5120_S4_L0_TF kernel_gemv_M1_K8704_N5120_S4_L0_TF
      kernel_gemv_i8w_M1_K5120_N124160 kernel_prep_silu_mul_M1_I8704
      kernel_prep_gated_head_M1_GK8V24 kernel_gdn_step_M1_GK8V24 kernel_attn_prep_M1_Q12KV2
      kernel_attn_v2_M1_T${ATTN_V2_TGT}_Q12KV2
      kernel_tp_cross_M1_K5120_G20_S2_X_R0 kernel_tp_cross_M1_K5120_G20_S2_X_R1
      kernel_tp_cross_M1_K5120_G20_S4_F_R0 kernel_tp_cross_M1_K5120_G20_S4_F_R1
      kernel_tp_argmax_M1_N124160_U124160_B0 kernel_tp_argmax_M1_N124160_U123917_B124160)
  if(B70_KV8)
    list(APPEND B70_TP_KERNELS kernel_attn_prep_M1_Q12KV2_KV8 kernel_attn_v2_M1_T${ATTN_V2_TGT}_Q12KV2_KV8)
  endif()
  set(B70_TP_KERNELS ${B70_TP_KERNELS} CACHE INTERNAL "spec 17b: the binaries a TP decode list binds")
endif()
# ==== Spec 17b ============================================================= (end) ====

# Spec 28c (plan 28c Task 2): the verify step's attention on DPAS (attn_verify_dpas.cl), in v2's blocking
# and partials, so the reduce is the attn_v2_M<M>_T32[_KV8] binary's attn_reduce_v2[_kv8] (no new reduce).
#   attn_verify_dpas_M<M>_T<TGT>       bf16 KV at the verify lists' M = 2..4 (B70_MTP) and 5..8 (B70_VERIFY_M8)
#   attn_verify_dpas_M<M>_T<TGT>_KV8   int8 KV at M = 2..8 (5..8 with B70_VERIFY_M8)
# One value lever setting per KV form for every M (property R, the kernel's header). Additive only: a new source,
# new names; nothing binds them until the capture's rule (src/runtime/verify_attn.h) says so.
option(B70_VERIFY_DPAS "build the DPAS verify attention (spec 28c)" ON)
if(B70_VERIFY_DPAS AND B70_MTP)
  set(VDPAS_CL ${CMAKE_CURRENT_SOURCE_DIR}/attn_verify_dpas.cl)
  # Plan 28c Task 4's kept settings: KT=32 PACK=g KS=4 DSPLIT=256 QSPLIT=1,
  # 256 GRF for both KV forms - the fastest passing N1 / N2 at M = 8, depth 32768 (QSPLIT=0 fails N1 at int8).
  # PSPLIT=1 since 2026-11-04 (P as bf16 hi + lo): P in bf16 alone moved real verify rows off the f32 top
  # (attn_verify_dpas.cl's header, docs/07-open-questions.md, "Recently answered").
  set(VDPAS_LEVERS KT=32 PACK_H=0 KS=4 DSPLIT=256 QSPLIT=1 PSPLIT=1)
  set(VDPAS_OPTIONS -cl-intel-256-GRF-per-thread)
  # 3l: the MTP verify rows M = 2..4, both KV forms, split the same
  # values over the work-group differently - PAIR (S by key atom in each row block's two sub-groups, P V by dim
  # atom over every row block) and, at M = 3 / 4 (three row blocks), PVX = 2 P V-only sub-groups a slice: 32
  # sub-groups, a whole core. Bitwise the frozen kernel (tests/kernels/attn_verify_bitwise_test.cc). M = 5..8's
  # 4-6 row blocks exceed 32 sub-groups (and 128 KB of SLM) in the PAIR split: int8 KV takes SOLO there (PAIR at
  # DSPLIT 256: S by key atom over two row blocks a sub-group, one softmax owner a row block, P V as PAIR's; PVX to
  # 8 sub-groups a slice), bitwise the same frozen kernel; at 8 rows 0.78 / 0.68x the plain line at 64k / 128k keys,
  # below bf16's own 8-row time. bf16 KV keeps VDPAS_LEVERS (SOLO is 1.11x at 32k keys there).
  set(VDPAS_LEVERS_PAIR KT=32 PACK_H=0 KS=4 DSPLIT=128 QSPLIT=1 PSPLIT=1 PAIR=1)
  set(VDPAS_LEVERS_SOLO KT=32 PACK_H=0 KS=4 DSPLIT=256 QSPLIT=1 PSPLIT=1 PAIR=1 SRB=2)
  set(_vdpas_pvx_2 0)
  set(_vdpas_pvx_3 2)
  set(_vdpas_pvx_4 2)
  set(_vdpas_pvx_5 4)
  set(_vdpas_pvx_6 3)
  set(_vdpas_pvx_7 2)
  set(_vdpas_pvx_8 2)
  set(_vdpas_m 2 3 4)
  if(B70_VERIFY_M8)
    list(APPEND _vdpas_m 5 6 7 8)
  endif()
  foreach(MM ${_vdpas_m})
    if(MM LESS_EQUAL 4)
      set(_vdpas_lv ${VDPAS_LEVERS_PAIR} PVX=${_vdpas_pvx_${MM}})
    else()
      set(_vdpas_lv ${VDPAS_LEVERS})
    endif()
    add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT} SOURCE ${VDPAS_CL}
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} ${_vdpas_lv} KV8=0
                     OPTIONS ${VDPAS_OPTIONS})
  endforeach()
  if(B70_KV8)
    foreach(MM 2 3 4)
      add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_KV8 SOURCE ${VDPAS_CL}
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} ${VDPAS_LEVERS_PAIR} PVX=${_vdpas_pvx_${MM}}
                               KV8=1
                       OPTIONS ${VDPAS_OPTIONS})
    endforeach()
    if(B70_VERIFY_M8)
      foreach(MM 5 6 7 8)
        add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_KV8 SOURCE ${VDPAS_CL}
                         DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} ${VDPAS_LEVERS_SOLO} PVX=${_vdpas_pvx_${MM}}
                                 KV8=1
                         OPTIONS ${VDPAS_OPTIONS})
      endforeach()
    endif()
  endif()
  # Plan 19v3b Task 0: the slots' paged twins (block above) of the MTP verify lists' M = 2..4, both KV forms.
  if(B70_VERIFY_M8 AND B70_KV8)
    foreach(MM 2 3 4 5 6 7 8)   # M = 5..8 (plan 19v3b Tasks 3-7): slot 0's DFlash verify lists, the one-card lines' levers
      foreach(KV8 0 1)
        if(MM LESS_EQUAL 4)
          set(_pg_lv ${VDPAS_LEVERS_PAIR} PVX=${_vdpas_pvx_${MM}})
        elseif(KV8)
          set(_pg_lv ${VDPAS_LEVERS_SOLO} PVX=${_vdpas_pvx_${MM}})
        else()
          set(_pg_lv ${VDPAS_LEVERS})
        endif()
        set(_kv "")
        if(KV8)
          set(_kv _KV8)
        endif()
        add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}${_kv}_PG${KV_PAGE} SOURCE ${VDPAS_CL}
                         DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} ${_pg_lv} KV8=${KV8} KV_PAGED=${KV_PAGE}
                         OPTIONS ${VDPAS_OPTIONS})
      endforeach()
    endforeach()
  endif()
  if(B70_VERIFY_M8)
    # verify-cost-m8 probe: the DPAS verify attention (and its v2 reduce) at 8 / 16 key blocks a row
    # (CaptureOptions::probe_attn_tgt); M = 2..4 at the PAIR levers, 5..8 at the plain ones - the production lines' levers
    foreach(TT 2 4 8 16)   # PERF-120: 2 and 4 too
      foreach(MM 2 3 4 5 6 7 8)
        if(MM EQUAL 8 AND TT EQUAL 8)
          continue()   # PERF-120: the production 8-row lines below
        endif()
        add_ocloc_kernel(attn_v2_M${MM}_T${TT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                         DEFINES ${CTRL_DEFINES} M=${MM} TGT=${TT} EXP2=${ATTN_V2_EXP2})
        if(MM LESS_EQUAL 4)
          set(_pvx 2)
          if(MM EQUAL 2)
            set(_pvx 0)
          endif()
          set(_lv KT=32 PACK_H=0 KS=4 DSPLIT=128 QSPLIT=1 PSPLIT=1 PAIR=1 PVX=${_pvx})
        else()
          set(_lv KT=32 PACK_H=0 KS=4 DSPLIT=256 QSPLIT=1 PSPLIT=1)
        endif()
        add_ocloc_kernel(attn_verify_dpas_M${MM}_T${TT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_verify_dpas.cl
                         DEFINES ${CTRL_DEFINES} M=${MM} TGT=${TT} ${_lv} KV8=0 OPTIONS -cl-intel-256-GRF-per-thread)
        if(B70_KV8)
          add_ocloc_kernel(attn_v2_M${MM}_T${TT}_KV8 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kv8.cl
                           DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=${MM} TGT=${TT})
          add_ocloc_kernel(attn_verify_dpas_M${MM}_T${TT}_KV8 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_verify_dpas.cl
                           DEFINES ${CTRL_DEFINES} M=${MM} TGT=${TT} ${_lv} KV8=1 OPTIONS -cl-intel-256-GRF-per-thread)
        endif()
      endforeach()
    endforeach()
    # PERF-120 (plan 40a Task 7): the one-sequence 8-row verify (DFlash K 7) at 8 key blocks a row
    # (runtime::verify_attn_blocks): attn_verify_dpas_M8_T8[_KV8][_PG<page>] at the M = 8 lines' levers above (bf16
    # VDPAS_LEVERS, int8 SOLO) and their reduce attn_v2_M8_T8[_KV8][_PG<page>]; the per-card 12 / 2 twins are in spec
    # 19 v3c's TP block. -2.6 % at depth 4k, -5.6 % at 30k on the bf16 verify against 32 blocks.
    foreach(PG "" _PG${KV_PAGE})
      set(_pgd "")
      if(PG)
        set(_pgd KV_PAGED=${KV_PAGE})
      endif()
      add_ocloc_kernel(attn_v2_M8_T8${PG} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                       DEFINES ${CTRL_DEFINES} M=8 TGT=8 EXP2=${ATTN_V2_EXP2} ${_pgd})
      add_ocloc_kernel(attn_verify_dpas_M8_T8${PG} SOURCE ${VDPAS_CL}
                       DEFINES ${CTRL_DEFINES} M=8 TGT=8 ${VDPAS_LEVERS} KV8=0 ${_pgd}
                       OPTIONS ${VDPAS_OPTIONS})
      if(B70_KV8)
        add_ocloc_kernel(attn_v2_M8_T8_KV8${PG} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kv8.cl
                         DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=8 TGT=8 ${_pgd})
        add_ocloc_kernel(attn_verify_dpas_M8_T8_KV8${PG} SOURCE ${VDPAS_CL}
                         DEFINES ${CTRL_DEFINES} M=8 TGT=8 ${VDPAS_LEVERS_SOLO} PVX=${_vdpas_pvx_8} KV8=1 ${_pgd}
                         OPTIONS ${VDPAS_OPTIONS})
      endif()
    endforeach()
  endif()
endif()

# ==== Open-work 28: the one-card 16-row verify list (long prompt-lookup runs), bf16 KV ==== (begin) ====
# Additive only: new names; the one-card Engine binds them in its one wide list (runtime::Engine::verify at
# kWideK). Ids through Control::wide_cur / wide_out (CTRL_CUR 32 / CTRL_OUT 48: embed_gather, argmax and
# argmax_stage2_W16 only - no other kernel reads the ids). The linears are gemm_sm's M = 16 rows (built below),
# the GDN step gdn_step_defer_M16 (above), the attention attn_verify_dpas at 16 rows with the key split
# across work-groups (KS 1, DSPLIT 128: the production KS 4 needs 768 lanes at 16 rows) and attn_v2_M16's reduce. Spec 33 (plan 33b): the same values split as the
# 2-4-row lines split them - PAIR (each row block's scores once, by key atom) at KS 1, PVX 8 (32 sub-groups) and
# PVR 2 (P V by dim atom over half the row blocks: V read twice a work-group, not twelve times); bitwise the KS 1
# DSPLIT 128 line (attn_verify_bitwise_test), 1.43 / 1.63x at 32k / 98k keys.
if(B70_MTP AND B70_VERIFY_M8)
  set(WIDE_CTRL_DEFINES CTRL_POS=0 CTRL_NACT=1 CTRL_CUR=32 CTRL_OUT=48 CTRL_DEBUG=18)
  add_ocloc_kernel(embed_gather_M16 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                   DEFINES ${WIDE_CTRL_DEFINES} M=16 HIDDEN=5120 VOCAB=248320)
  add_ocloc_kernel(argmax_stage1_M16 SOURCE ${ARGMAX_CL} DEFINES ${WIDE_CTRL_DEFINES} VOCAB=248320 VOCAB_USED=248077 M=16)
  add_ocloc_kernel(argmax_stage2_W16 SOURCE ${ARGMAX_CL} DEFINES ${WIDE_CTRL_DEFINES} VOCAB=248320 VOCAB_USED=248077 M=16)
  # the rest of the list's per-row binaries (the norms, the MLP's, attn_prep / attn_v2, the head's KV fill) come
  # from the M = 5..8 loop above, at M = 16
  add_ocloc_kernel(attn_verify_dpas_M16_T${ATTN_V2_TGT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_verify_dpas.cl
                   DEFINES ${CTRL_DEFINES} M=16 TGT=${ATTN_V2_TGT} KT=32 PACK_H=0 KS=1 DSPLIT=128 QSPLIT=1 PSPLIT=1 PAIR=1
                           PVX=8 PVR=2 KV8=0
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_gemv_bf16_variant(16 5120 128 16 64)       # a||b, the K split of the lists at M >= 5
endif()
# ==== Open-work 28 ============================================================ (end) ====

# ==== Spec 32: the one-card 32- and 64-row verify lists (deep prompt-lookup runs), bf16 KV ==== (begin) ====
# Additive only: new names. Ids through Control::deep_cur / deep_out (CTRL_CUR 64 / CTRL_OUT 128). The attention is
# the 16-row line (spec 33's PAIR PVX 8 PVR 2 since spec 32e) in row chunks of 16 (RCHUNK: grid dimension 0, a block's
# chunks side by side) with attn_v2's reduce alone at the
# list's rows; the linears are gemm_sm_wide's rows at 32 / 64; the rest the 16-row list's per-row builds at W.
set(DEEP_W 32 64)
if(B70_MTP AND B70_VERIFY_M8)
  set(DEEP_CTRL_DEFINES CTRL_POS=0 CTRL_NACT=1 CTRL_CUR=64 CTRL_OUT=128 CTRL_DEBUG=18)
  foreach(MM ${DEEP_W})
    add_ocloc_kernel(embed_gather_M${MM} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                     DEFINES ${DEEP_CTRL_DEFINES} M=${MM} HIDDEN=5120 VOCAB=248320)
    add_ocloc_kernel(argmax_stage1_M${MM} SOURCE ${ARGMAX_CL} DEFINES ${DEEP_CTRL_DEFINES} VOCAB=248320 VOCAB_USED=248077 M=${MM})
    add_ocloc_kernel(argmax_stage2_W${MM} SOURCE ${ARGMAX_CL} DEFINES ${DEEP_CTRL_DEFINES} VOCAB=248320 VOCAB_USED=248077 M=${MM})
    add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_verify_dpas.cl
                     DEFINES ${CTRL_DEFINES} M=16 RCHUNK=${MM} TGT=${ATTN_V2_TGT} KT=32 PACK_H=0 KS=1 DSPLIT=128 QSPLIT=1
                             PSPLIT=1 PAIR=1 PVX=8 PVR=2 KV8=0
                     OPTIONS -cl-intel-256-GRF-per-thread)
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} REDUCE_ONLY=1)
    if(MM EQUAL 32)
      add_gemv_bf16_variant(${MM} 5120 128 16 16)  # a||b (at 64 rows: two 32-row launches)
    endif()
    add_prep_res_fold(${MM} 5120 0 20)
    add_prep_res_fold(${MM} 5120 4 20)
    add_prep_norm_finish(${MM} 5120 20 20)
    add_ocloc_kernel(prep_gated_head_M${MM} SOURCE ${PREP_CL} DEFINES M=${MM})
    add_attn_prep(${MM})
    # the MTP head's KV fill of W rows (its fc and q||k||v: the 16-row builds, in 16-row launches)
    # PERF-84: the head's two pre-fc norms, one fold and one finish over both sources
    add_ocloc_kernel(prep_res_fold_M${MM}_K5120_SP0_G20_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 S_PREV=0 FOLD_G=20 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_norm_finish_M${MM}_K5120_G20_W20_X10240_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 NORM_G=20 NORM_WGS=20 X_STRIDE=10240 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_res_fold_M${MM}_K5120_SP1_G20_Z SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
    add_ocloc_kernel(attn_prep_M${MM}_S1 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1)
  endforeach()
  # The delta-net step at W rows: W / 16 links of the 16-row deferred build (gdn_step.cl, ROW0), and the ring copy a
  # commit of rows 16.. runs (gdn_ring_commit.cl; CTRL_RC = Control::ring_commit).
  if(B70_DFLASH)
    foreach(R0 0 16 32 48)
      add_ocloc_kernel(gdn_step_defer_M16_R${R0} SOURCE ${GDN_CL}
                       DEFINES ${GDN_DEFINES} DEFERRED=1 CTRL_PEND=30 PEND_PAR=51511296 M=16 ROW0=${R0})
    endforeach()
  endif()
  add_ocloc_kernel(gdn_ring_commit SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gdn_ring_commit.cl DEFINES CTRL_POS=0 CTRL_RC=31)
  # Item 40: the 16-, 32- and 64-row lists' bf16 GEMVs (a||b, the MTP head's fc and q||k||v) in one launch each, rows
  # in groups of 8 on grid dimension 0 (ROWG) - every row bitwise the plain builds'.
  add_ocloc_kernel(gemv_bf16_M8_K5120_N128_C16_S16_RG SOURCE ${GEMV_BF16_CL}
                   DEFINES M=8 K=5120 N=128 COLS_PER_WG=16 KSPLIT=16 ROWG=1)
  add_ocloc_kernel(gemv_bf16_M8_K10240_N5120_RG SOURCE ${GEMV_BF16_CL} DEFINES M=8 K=10240 N=5120 ROWG=1)
  add_ocloc_kernel(gemv_bf16_M8_K5120_N14336_RG SOURCE ${GEMV_BF16_CL} DEFINES M=8 K=5120 N=14336 ROWG=1)
endif()
# ==== Spec 32 ================================================================== (end) ====

# ==== Spec 17d: tensor parallel MTP verify (`--parallel tp --spec mtp|auto`, Qwen3.8) ======= (begin) ====
# Additive only: new names (attn_verify_dpas.cl takes the per-card 12 / 2 heads by define; its default lines keep
# their preprocessed source). What a card's verify list at M = 1..4 binds (runtime::build_tp_verify, capture.cc) at
# model::tp_card_desc's shapes beyond 17b's M = 1 list - the folded crossing only (runtime::kTpDefaultCrossing):
#   GDN      gdn_step_slots_M<M>_GK8V24 (SPEC_SLOT_STRIDE one card's slot: 48 x 24 x 128 x 128 floats), M = 1..4;
#            prep_gated_head_M<M>_GK8V24, M = 2..4
#   FA       attn_prep / attn_v2 at M = 2..4, 12 q / 2 kv, and their _KV8 twins; attn_verify_dpas at 12 / 2 with the
#            one-card M = 2..4 lines' levers (3l's PAIR split, PVX by M)
#   MLP      prep_silu_mul_M<M>_I8704 (gate||up at S 8: the per-card gemm_sm row has no 2x8 work-group)
#   tp.cl    tp_cross at M = 2..4 (folded, S4) per rank, tp_argmax at M = 2..4 per vocabulary half
#   linears  the per-card gemm_sm rows (kernels::kSmallMTuning's Spec 17d rows, built in the 28b block below)
# The MTP head runs at one card's shapes on each card (replicated, its own KV layer at 24 / 4 heads): its draft list
# and its verify-tail KV fill bind spec 8's binaries, and the draft's lm_head is the card's int8 vocabulary half
# (17b's gemv_i8w_M1_K5120_N124160 and tp_argmax_M1).
if(B70_TP AND B70_MTP)
  set(TP_GDN_SLOT_DEFINES ${CTRL_DEFINES} NEGA_OFF=20480 DTBIAS_OFF=20504 ${TP_GDN_HEADS}
                          SPEC_SLOTS=1 CTRL_LIVE=19 SPEC_SLOT_STRIDE=18874368)
  set(B70_TP_VERIFY_KERNELS "")
  foreach(MM 1 2 3 4)
    add_ocloc_kernel(gdn_step_slots_M${MM}_GK8V24 SOURCE ${GDN_CL} DEFINES ${TP_GDN_SLOT_DEFINES} M=${MM})
    list(APPEND B70_TP_VERIFY_KERNELS kernel_gdn_step_slots_M${MM}_GK8V24)
  endforeach()
  foreach(MM 2 3 4)
    add_ocloc_kernel(prep_gated_head_M${MM}_GK8V24 SOURCE ${PREP_CL} DEFINES M=${MM} ${TP_GDN_HEADS})
    add_ocloc_kernel(prep_silu_mul_M${MM}_I8704 SOURCE ${PREP_CL} DEFINES M=${MM} INTERMEDIATE=8704)
    add_ocloc_kernel(attn_prep_M${MM}_Q12KV2 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} ${TP_FA_HEADS})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} ${TP_FA_HEADS})
    list(APPEND B70_TP_VERIFY_KERNELS kernel_prep_gated_head_M${MM}_GK8V24 kernel_prep_silu_mul_M${MM}_I8704
                kernel_attn_prep_M${MM}_Q12KV2 kernel_attn_v2_M${MM}_T${ATTN_V2_TGT}_Q12KV2)
    if(B70_KV8)
      add_ocloc_kernel(attn_prep_M${MM}_Q12KV2_KV8 SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=2 ${TP_FA_HEADS})
      add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q12KV2_KV8 SOURCE ${KV8_CL}
                       DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT} ${TP_FA_HEADS})
      list(APPEND B70_TP_VERIFY_KERNELS kernel_attn_prep_M${MM}_Q12KV2_KV8
                  kernel_attn_v2_M${MM}_T${ATTN_V2_TGT}_Q12KV2_KV8)
    endif()
    if(B70_VERIFY_DPAS)
      add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_Q12KV2 SOURCE ${VDPAS_CL}
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} ${VDPAS_LEVERS_PAIR} PVX=${_vdpas_pvx_${MM}}
                               KV8=0 ${TP_FA_HEADS}
                       OPTIONS ${VDPAS_OPTIONS})
      list(APPEND B70_TP_VERIFY_KERNELS kernel_attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_Q12KV2)
      if(B70_KV8)
        add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_Q12KV2_KV8 SOURCE ${VDPAS_CL}
                         DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} ${VDPAS_LEVERS_PAIR}
                                 PVX=${_vdpas_pvx_${MM}} KV8=1 ${TP_FA_HEADS}
                         OPTIONS ${VDPAS_OPTIONS})
        list(APPEND B70_TP_VERIFY_KERNELS kernel_attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_Q12KV2_KV8)
      endif()
    endif()
    foreach(R 0 1)
      add_ocloc_kernel(tp_cross_M${MM}_K5120_G20_S4_F_R${R} SOURCE ${TP_CL}
                       DEFINES TP_CROSS=1 M=${MM} K=5120 FOLD_G=20 S_LOC=4 FOLDED=1 RANK=${R})
      list(APPEND B70_TP_VERIFY_KERNELS kernel_tp_cross_M${MM}_K5120_G20_S4_F_R${R})
    endforeach()
    set(_tp_am ${CTRL_DEFINES} CTRL_OUTV=20 TP_ARGMAX=1 M=${MM} VOCAB=124160)
    add_ocloc_kernel(tp_argmax_M${MM}_N124160_U124160_B0 SOURCE ${TP_CL}
                     DEFINES ${_tp_am} VOCAB_USED=124160 ID_BASE=0)
    add_ocloc_kernel(tp_argmax_M${MM}_N124160_U123917_B124160 SOURCE ${TP_CL}
                     DEFINES ${_tp_am} VOCAB_USED=123917 ID_BASE=124160)
    list(APPEND B70_TP_VERIFY_KERNELS kernel_tp_argmax_M${MM}_N124160_U124160_B0
                kernel_tp_argmax_M${MM}_N124160_U123917_B124160)
  endforeach()
  # Open-work 13: each card's draft probability over its vocabulary half (draft_prob.cl; the B70_MTP block has the
  # whole row's).
  add_ocloc_kernel(draft_prob_N124160_U124160 SOURCE ${DRAFT_PROB_CL} DEFINES CTRL_DSTAT=28 VOCAB=124160 VOCAB_USED=124160)
  add_ocloc_kernel(draft_prob_N124160_U123917 SOURCE ${DRAFT_PROB_CL} DEFINES CTRL_DSTAT=28 VOCAB=124160 VOCAB_USED=123917)
  list(APPEND B70_TP_VERIFY_KERNELS kernel_draft_prob_N124160_U124160 kernel_draft_prob_N124160_U123917)
  set(B70_TP_VERIFY_KERNELS ${B70_TP_VERIFY_KERNELS} CACHE INTERNAL "spec 17d: the binaries a TP verify list binds")
endif()
# ==== Spec 17d ============================================================= (end) ====

# ==== Spec 19 v3c: DFlash under tensor parallel (`--parallel tp --spec dflash`, Qwen3.8) ===== (begin) ====
# Additive only: new names; no engine binds
# any of these without a drafter. What a card's tap verify list (runtime::build_tp_verify_taps) binds at M = 1..8 beyond
# spec 17d's M = 1..4 lists, at model::tp_card_desc's shapes (the folded crossing):
#   GDN      gdn_step_slots_M<M>_N8_GK8V24, M = 1..8 (the drafter's 8 state slots at the card's stride: 48 x 24 x
#            128 x 128 floats); prep_gated_head_M<M>_GK8V24, M = 5..8
#   FA       attn_prep / attn_v2 at M = 5..8, 12 q / 2 kv, and their _KV8 twins; attn_verify_dpas at 12 / 2 with the
#            one-card M = 5..8 lines' levers (VDPAS_LEVERS: the PAIR split of 2..4 rows does not fit 6+ row blocks)
#   MLP      prep_silu_mul_M<M>_I8704, M = 5..8
#   tp.cl    tp_cross at M = 5..8 (folded, S4) per rank, tp_argmax at M = 5..8 per vocabulary half
#   linears  the per-card gemm_sm rows (kernels::kSmallMTuning's spec 17d rows, now at M = 2..8: the 28b block below)
if(B70_TP AND B70_MTP AND B70_VERIFY_M8)
  set(TP_GDN_SLOT8_DEFINES ${CTRL_DEFINES} NEGA_OFF=20480 DTBIAS_OFF=20504 ${TP_GDN_HEADS}
                           SPEC_SLOTS=1 CTRL_LIVE=19 SPEC_SLOT_STRIDE=18874368 N_SLOTS=8)
  foreach(MM 1 2 3 4 5 6 7 8)
    add_ocloc_kernel(gdn_step_slots_M${MM}_N8_GK8V24 SOURCE ${GDN_CL} DEFINES ${TP_GDN_SLOT8_DEFINES} M=${MM})
  endforeach()
  foreach(MM 5 6 7 8)
    add_ocloc_kernel(prep_gated_head_M${MM}_GK8V24 SOURCE ${PREP_CL} DEFINES M=${MM} ${TP_GDN_HEADS})
    add_ocloc_kernel(prep_silu_mul_M${MM}_I8704 SOURCE ${PREP_CL} DEFINES M=${MM} INTERMEDIATE=8704)
    add_ocloc_kernel(attn_prep_M${MM}_Q12KV2 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} ${TP_FA_HEADS})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} ${TP_FA_HEADS})
    if(B70_KV8)
      add_ocloc_kernel(attn_prep_M${MM}_Q12KV2_KV8 SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=2 ${TP_FA_HEADS})
      add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q12KV2_KV8 SOURCE ${KV8_CL}
                       DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT} ${TP_FA_HEADS})
    endif()
    if(B70_VERIFY_DPAS)
      add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_Q12KV2 SOURCE ${VDPAS_CL}
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} ${VDPAS_LEVERS} KV8=0 ${TP_FA_HEADS}
                       OPTIONS ${VDPAS_OPTIONS})
      if(B70_KV8)
        add_ocloc_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_Q12KV2_KV8 SOURCE ${VDPAS_CL}
                         DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} ${VDPAS_LEVERS} KV8=1 ${TP_FA_HEADS}
                         OPTIONS ${VDPAS_OPTIONS})
      endif()
      if(MM EQUAL 8)   # PERF-120: the one-card 8-row verify's 8 key blocks a row (runtime::verify_attn_blocks)
        add_ocloc_kernel(attn_v2_M8_T8_Q12KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                         DEFINES ${CTRL_DEFINES} M=8 TGT=8 EXP2=${ATTN_V2_EXP2} ${TP_FA_HEADS})
        add_ocloc_kernel(attn_verify_dpas_M8_T8_Q12KV2 SOURCE ${VDPAS_CL}
                         DEFINES ${CTRL_DEFINES} M=8 TGT=8 ${VDPAS_LEVERS} KV8=0 ${TP_FA_HEADS}
                         OPTIONS ${VDPAS_OPTIONS})
        if(B70_KV8)
          add_ocloc_kernel(attn_v2_M8_T8_Q12KV2_KV8 SOURCE ${KV8_CL}
                           DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=8 TGT=8 ${TP_FA_HEADS})
          add_ocloc_kernel(attn_verify_dpas_M8_T8_Q12KV2_KV8 SOURCE ${VDPAS_CL}
                           DEFINES ${CTRL_DEFINES} M=8 TGT=8 ${VDPAS_LEVERS} KV8=1 ${TP_FA_HEADS}
                           OPTIONS ${VDPAS_OPTIONS})
        endif()
      endif()
    endif()
    foreach(R 0 1)
      add_ocloc_kernel(tp_cross_M${MM}_K5120_G20_S4_F_R${R} SOURCE ${TP_CL}
                       DEFINES TP_CROSS=1 M=${MM} K=5120 FOLD_G=20 S_LOC=4 FOLDED=1 RANK=${R})
    endforeach()
    set(_tp_am ${CTRL_DEFINES} CTRL_OUTV=20 TP_ARGMAX=1 M=${MM} VOCAB=124160)
    add_ocloc_kernel(tp_argmax_M${MM}_N124160_U124160_B0 SOURCE ${TP_CL}
                     DEFINES ${_tp_am} VOCAB_USED=124160 ID_BASE=0)
    add_ocloc_kernel(tp_argmax_M${MM}_N124160_U123917_B124160 SOURCE ${TP_CL}
                     DEFINES ${_tp_am} VOCAB_USED=123917 ID_BASE=124160)
  endforeach()
  # tp_kernels_test stress: the crossing with its trailers packed 16 to a line (TP_TRAILER_STRIDE 1, the layout before
  # 2026-11-14 that lost stamps), the A/B against the production binaries. Test-only names (_T1); nothing binds them.
  foreach(MM 1 2 3 4 5 6 7 8)
    foreach(R 0 1)
      add_ocloc_kernel(tp_cross_M${MM}_K5120_G20_S4_F_T1_R${R} SOURCE ${TP_CL}
                       DEFINES TP_CROSS=1 M=${MM} K=5120 FOLD_G=20 S_LOC=4 FOLDED=1 RANK=${R} TP_TRAILER_STRIDE=1)
      list(APPEND B70_TP_STRESS_KERNELS kernel_tp_cross_M${MM}_K5120_G20_S4_F_T1_R${R}
                                        kernel_tp_cross_M${MM}_K5120_G20_S4_F_R${R})
    endforeach()
  endforeach()
  set(B70_TP_STRESS_KERNELS ${B70_TP_STRESS_KERNELS} CACHE INTERNAL "tp_kernels_test stress: both trailer layouts")
endif()
# The drafter split over both cards (runtime::dflash::TpDFlashDrafter; kernels::dflash::kQwen38Tp): a card's draft and
# projection lists at its half shapes - 16 q / 4 kv heads, the MLP's 8704, its vocabulary half of the candidate head:
#   df_prep (plain and FOLD)   at I 8704, Q16KV4 (df_qkv_prep, df_silu_mul, df_ctx_store; the rest is the hidden's)
#   df_attn                   the DPAS part (KS 4) and its reduce (the H1 B4 binary) at 16 / 4 heads
#   df_candidate              a half's top C of its candidate rows (N 124160; card 1's real rows end at 123917)
#   df_rerank                 against the card's int8 head half, its ids offset by the half's first (ID_BASE)
#   tp.cl                     tp_send + tp_sum at M 8 per rank: the all-reduce after o and down, the sum in fp32 into
#                             the conv's finish side - S1 (gemm_sm_i8w's one slice) and S4 (the int4 grid split-K)
# The linears: kernels::kSmallMTuning's spec 19 v3c rows (the 28b block below).
if(B70_TP AND B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  set(DF_TP_SHAPE M=8 H=5120 WG=256 EPS=1e-06f VOCAB=248320 GROUP=16 CTAPS=2 INTER=8704 NH=16 NKV=4 HD=128 RING=2048)
  add_ocloc_kernel(df_prep_M8_H5120_I8704_Q16KV4_D128_G16T2_R2048_V248320 SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} ${DF_TP_SHAPE})
  add_ocloc_kernel(df_prep_M8_H5120_I8704_Q16KV4_D128_G16T2_R2048_V248320_F8Q4R4U8C16H16K8 SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} FOLD=1 ${DF_TP_SHAPE})
  add_ocloc_kernel(df_attn_M8_Q16KV4_D128_R2048_W2048_P128_H1_B4 SOURCE ${DF_DIR}/df_attn_rows.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=16 NKV=4 HD=128 RING=2048 WIN=2048 PB=128 WGA=128
                           CAUSAL=0 HPW=1 RPW=4)
  add_ocloc_kernel(df_attn_M8_Q16KV4_D128_R2048_W2048_P128_dpas_k4 SOURCE ${DF_DIR}/df_attn_dpas.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=16 NKV=4 HD=128 RING=2048 WIN=2048 PB=128 KS=4)
  foreach(CC 64 80)
    add_ocloc_kernel(df_candidate_R7_N124160_U124160_K${CC} SOURCE ${DF_DIR}/df_candidate.cl
                     DEFINES R=7 N=124160 USED=124160 C=${CC})
    add_ocloc_kernel(df_candidate_R7_N124160_U123917_K${CC} SOURCE ${DF_DIR}/df_candidate.cl
                     DEFINES R=7 N=124160 USED=123917 C=${CC})
    add_ocloc_kernel(df_rerank_R7_K5120_U124160_C${CC}_Q4_B0 SOURCE ${DF_DIR}/df_rerank.cl
                     DEFINES K=5120 USED=124160 C=${CC} ID_BASE=0u)
    add_ocloc_kernel(df_rerank_R7_K5120_U123917_C${CC}_Q4_B124160 SOURCE ${DF_DIR}/df_rerank.cl
                     DEFINES K=5120 USED=123917 C=${CC} ID_BASE=124160u)
  endforeach()
  foreach(S 1 4)
    foreach(R 0 1)
      add_ocloc_kernel(tp_cross_M8_K5120_G20_S${S}_F_R${R}_sum SOURCE ${TP_CL}
                       DEFINES TP_CROSS=1 M=8 K=5120 FOLD_G=20 S_LOC=${S} FOLDED=1 RANK=${R} TP_SUM=${S})
    endforeach()
  endforeach()
endif()
# ==== Spec 19 v3c ========================================================== (end) ====

# ==== Spec 20c: Kolibri-1 decode (model::kolibri1()) ======================= (begin) ====
# Additive only: no existing source is edited, so every binary above keeps its name, its command line
# and its bytes (tools/kernel_cmdlines' diff: additions only). Kolibri-only kernels live in
# src/kernels/kolibri/ and are named `kol_*` with every shape define in the name
# (src/kernels/kolibri_kernels.h is the host half); the reused sources run at Kolibri's shapes:
#   gemv.cl         q||k||v 2560x7168 (S2 L0), o_proj 6144x2560 (S4 L0) - the int4 attention arm,
#                   PROVISIONAL cells (model::Kolibri1Desc's {S, layout}) until Task 8's sweep
#   gemv_bf16.cl    the bf16 attention arm's two rows, the router 2560 -> 512 rows (384 + 128 zero;
#                   fp32 logits), lm_head 2560 -> 128000 (fp32 accumulation and logits: head_dtype float32)
#   gemv_i8w.cl     the int8 lm_head (spec 9)
#   prep.cl         prep_res_fold at K 2560: SP0 (after embed_gather, over the MoE output) and with
#                   ZERO_RESID over o_proj's S slices (the sandwich: the sub-block's own row and Σ²)
#   embed_gather.cl / argmax.cl at Kolibri's vocabulary 128000 (fully used)
# What a Kolibri decode list binds (runtime/kolibri/kolibri_capture.cc): 15 launches per layer under
# flash, 17 under B70_KOLIBRI_ATTN=eager, + embed + fold + 4 at the head: 756 / 856 at 50 layers.
option(B70_KOLIBRI "build the Kolibri-1 decode variants (spec 20c)" ON)
if(B70_KOLIBRI)
  set(KOL_CL ${CMAKE_CURRENT_SOURCE_DIR}/kolibri)
  add_gemv_variant(1 2560 7168 2 0)            # q||k||v int4        PROVISIONAL S2 L0
  add_gemv_variant(1 6144 2560 4 0)            # o_proj int4         PROVISIONAL S4 L0
  add_prep_res_fold(1 2560 0 20)
  add_gemv_bf16_variant(1 2560 7168 64 1)      # q||k||v bf16 (decision 2's bf16 arm)
  add_gemv_bf16_variant(1 6144 2560 64 1)      # o_proj bf16
  add_gemv_bf16_variant(1 2560 512 16 16)      # the router, rows 384..511 zero; fp32 logits
  add_gemv_bf16_variant(1 2560 128000 64 1)    # lm_head bf16 (fp32 accumulation and logits = head_dtype float32)
  add_gemv_i8w_variant(1 2560 128000)          # lm_head int8 (spec 9)
  # prep_res_fold ..._Z: o_proj's S slices (S4 int4, S1 bf16) into its own row `a` and Σa²
  foreach(SP 1 4)
    add_ocloc_kernel(prep_res_fold_M1_K2560_SP${SP}_G20_Z SOURCE ${PREP_CL}
                     DEFINES M=1 K=2560 S_PREV=${SP} FOLD_G=20 ZERO_RESID=1)
  endforeach()
  add_ocloc_kernel(kol_embed_gather_M1_D2560_V128000 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                   DEFINES ${CTRL_DEFINES} M=1 HIDDEN=2560 VOCAB=128000)
  set(KOL_ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=128000 VOCAB_USED=128000)
  add_ocloc_kernel(kol_argmax_stage1_M1_N128000_V128000 SOURCE ${ARGMAX_CL} DEFINES ${KOL_ARGMAX_DEFINES} M=1)
  add_ocloc_kernel(kol_argmax_stage2_N128000 SOURCE ${ARGMAX_CL} DEFINES ${KOL_ARGMAX_DEFINES})
  # kol_prep.cl: the norm's stage B (plain w, x̂ rounded first), the post norm + residual add, the
  # attention prep (q/k head norm, RoPE in the sliding layers only, the ring or the growing cache).
  add_ocloc_kernel(kol_norm_M1_K2560_G20_W20 SOURCE ${KOL_CL}/kol_prep.cl DEFINES M=1 K=2560 NORM_G=20 NORM_WGS=20)
  add_ocloc_kernel(kol_post_add_M1_K2560_G20 SOURCE ${KOL_CL}/kol_prep.cl DEFINES M=1 K=2560 POST_G=20)
  set(KOL_HEADS Q_HEADS=48 KV_HEADS=4 HD=128)
  foreach(S 1 2)          # S 2: the int4 arm's q||k||v split-K; S 1: the bf16 arm (gemv_bf16 writes [M][N])
    add_ocloc_kernel(kol_attn_prep_M1_N7168_S${S}_Q48KV4_R4096 SOURCE ${KOL_CL}/kol_prep.cl
                     DEFINES ${CTRL_DEFINES} M=1 QKV_N=7168 QKV_S=${S} ${KOL_HEADS} SLIDING=1 RING=4096)
    add_ocloc_kernel(kol_attn_prep_M1_N7168_S${S}_Q48KV4_F SOURCE ${KOL_CL}/kol_prep.cl
                     DEFINES ${CTRL_DEFINES} M=1 QKV_N=7168 QKV_S=${S} ${KOL_HEADS} SLIDING=0 RING=0)
  endforeach()
  # kol_moe.cl: the 384-expert router on 512 slots / 256 lanes, the MoE block with the bf16 shared slot.
  add_ocloc_kernel(kol_route_M1_E384_T6_N512_L256 SOURCE ${KOL_CL}/kol_moe.cl
                   DEFINES M=1 ROUTE_E=384 ROUTE_K=6 ROUTE_WG=256 ROUTE_EPL=2 ROUTE_LN=512)
  add_ocloc_kernel(kol_moe_M1_E384_T6_D2560_I512_SH SOURCE ${KOL_CL}/kol_moe.cl
                   DEFINES M=1 MOE_E=384 MOE_K=6 HIDDEN=2560 INTER=512 UP_KS=4 DN_KS=2)
  # kol_attn.cl (flash, the default) and kol_attn_eager.cl (B70_KOLIBRI_ATTN=eager), sliding and full.
  add_ocloc_kernel(kol_attn_M1_T${ATTN_V2_TGT}_Q48KV4_W513_R4096 SOURCE ${KOL_CL}/kol_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=513 RING=4096)
  add_ocloc_kernel(kol_attn_M1_T${ATTN_V2_TGT}_Q48KV4_F SOURCE ${KOL_CL}/kol_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=0 RING=0)
  add_ocloc_kernel(kol_attn_eager_M1_T${ATTN_V2_TGT}_Q48KV4_W513_R4096 SOURCE ${KOL_CL}/kol_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=513 RING=4096)
  add_ocloc_kernel(kol_attn_eager_M1_T${ATTN_V2_TGT}_Q48KV4_F SOURCE ${KOL_CL}/kol_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=0 RING=0)
endif()
# ==== Spec 20c: Kolibri-1 decode ============================================ (end) ====

# ==== Spec 20d: Kolibri-1 prefill ==== (begin) ====
# Additive only, as 20c's block: no existing source is edited, so every binary above (and in prefill/)
# keeps its name, command line and bytes. New Kolibri-only sources in src/kernels/kolibri/:
#   kol_pf_moe.cl     the 384-expert sort (ids as ushort, two experts a lane), gather, the dequant of a
#                     weight batch (block 384 = the bf16 shared expert, copied), the combine into `mo`
#   kol_pf_attn.cl    the windowed flash attention over the 4096-slot ring (sliding) and the growing
#                     cache (full), spec 6's DPAS flash walk at head_dim 128 and GQA 12, no gate; and
#                     its eager twin
#   (prefill/slab_bf16.cl: the bf16 attention arm's slabs with a zero-padded tail, o_proj's 512 columns)
# Reused at Kolibri's shapes: kol_prep.cl at M = 2048 (kernels::kolibri::kPfC: the norm, the post-add,
# the attention prep at S 1), prefill/pf_embed.cl, prefill/pf_prep.cl's pf_res_fold (SP0, and _Z over
# o_proj's one slice), prefill/pf_gemv_bf16.cl (the router at decode's {16, 16} tiling),
# prefill/pf_moe_gemm.cl (gate||up 2560 x 1024 SiLU, down 512 x 2560), prefill/slab_dequant_int4.cl (the int4
# arm's layout-0 slabs) and the shape-free pf_gemm_T0. The route is 20c's decode binary on grid (1, C).
# What a Kolibri prefill chunk binds: runtime/kolibri/kolibri_prefill.cc (45 launches a layer + 2,
# 2252 at 50 layers; derived in runtime/kolibri/kolibri_sizes.cc).
if(B70_KOLIBRI)
  set(KOL_PF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kolibri)
  set(KOL_PF_REUSE ${CMAKE_CURRENT_SOURCE_DIR}/prefill)
  add_ocloc_kernel(kol_pf_embed_gather_D2560_V128000 SOURCE ${KOL_PF_REUSE}/pf_embed.cl DEFINES HIDDEN=2560 VOCAB=128000)
  add_ocloc_kernel(pf_res_fold_K2560_SP0_G20 SOURCE ${KOL_PF_REUSE}/pf_prep.cl DEFINES K=2560 S_PREV=0 FOLD_G=20)
  add_ocloc_kernel(pf_int4_slab_K6144_N2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/slab_dequant_int4.cl DEFINES K=6144 N=2560)
  add_ocloc_kernel(pf_res_fold_K2560_SP1_G20_Z SOURCE ${KOL_PF_REUSE}/pf_prep.cl DEFINES K=2560 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
  add_ocloc_kernel(pf_int4_slab_K2560_N7168 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/slab_dequant_int4.cl DEFINES K=2560 N=7168)
  foreach(row "2560 7168" "6144 2560")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(kol_pf_bf16_slab_K${K}_N${N} SOURCE ${KOL_PF_REUSE}/slab_bf16.cl DEFINES K=${K} N=${N})
  endforeach()
  add_ocloc_kernel(kol_norm_M2048_K2560_G20_W20 SOURCE ${KOL_PF_SRC}/kol_prep.cl DEFINES M=2048 K=2560 NORM_G=20 NORM_WGS=20)
  add_ocloc_kernel(kol_post_add_M2048_K2560_G20 SOURCE ${KOL_PF_SRC}/kol_prep.cl DEFINES M=2048 K=2560 POST_G=20)
  set(KOL_PF_HEADS Q_HEADS=48 KV_HEADS=4 HD=128)
  add_ocloc_kernel(kol_attn_prep_M2048_N7168_S1_Q48KV4_R4096 SOURCE ${KOL_PF_SRC}/kol_prep.cl
                   DEFINES ${CTRL_DEFINES} M=2048 QKV_N=7168 QKV_S=1 ${KOL_PF_HEADS} SLIDING=1 RING=4096)
  add_ocloc_kernel(kol_attn_prep_M2048_N7168_S1_Q48KV4_F SOURCE ${KOL_PF_SRC}/kol_prep.cl
                   DEFINES ${CTRL_DEFINES} M=2048 QKV_N=7168 QKV_S=1 ${KOL_PF_HEADS} SLIDING=0 RING=0)
  # The flash attention: sliding (window 513 over the ring) and full, the default (fp32 scores and
  # probabilities, natural exp) and the eager-mirroring form (B70_KOLIBRI_ATTN=eager). 256 GRF as
  # pf_flash_attn.
  foreach(SL 1 0)
    foreach(EG 0 1)
      if(SL)
        set(_n kol_pf_flash_attn_Q48KV4_W513_R4096)
        set(_w WINDOW=513 RING=4096)
      else()
        set(_n kol_pf_flash_attn_Q48KV4_F)
        set(_w WINDOW=0 RING=0)
      endif()
      if(EG)
        string(APPEND _n _EAGER)
      endif()
      add_ocloc_kernel(${_n} SOURCE ${KOL_PF_SRC}/kol_pf_attn.cl
                       DEFINES KT=64 RPW=8 HPW=4 Q_HEADS=48 KV_HEADS=4 EXP2=0 EAGER=${EG} ${_w}
                       OPTIONS -cl-intel-256-GRF-per-thread)
    endforeach()
  endforeach()
  add_ocloc_kernel(kol_pf_moe_E384_T6_D2560_I512_L256 SOURCE ${KOL_PF_SRC}/kol_pf_moe.cl
                   DEFINES PF_E=384 PF_K=6 PF_WG=256 PF_EPL=2 PF_D=2560 PF_INTER=512 TM=32 KC=2048)
  add_ocloc_kernel(pf_moe_router_K2560_N512 SOURCE ${KOL_PF_REUSE}/pf_gemv_bf16.cl DEFINES K=2560 N=512 COLS_PER_WG=16 KSPLIT=16)
  add_ocloc_kernel(pf_moe_gemm_K2560_N1024_SILU SOURCE ${KOL_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=2560 N=1024 TM=32 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_K512_N2560 SOURCE ${KOL_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=512 N=2560 TM=32 OPTIONS -cl-intel-256-GRF-per-thread)
endif()
# ==== Spec 20d: Kolibri-1 prefill ==== (end) ====

# ==== Spec 21-QFN-v1c: Qwen3.8-Flash-Next decode (model::qwen4exp()) ==== (begin) ====
# Additive only but for ONE source: prep.cl gains -DGDN_GATE_SIGMOID (prep_gated_head's sigmoid gate, the `_SIG`
# binary); every existing prep.cl variant keeps its command line and its preprocessed source token for token (the
# 6499948 rule; G0's sha256 on the box). Family-only kernels live in src/kernels/qwen4exp/ and are named `q4_*` with
# every shape define in the name (src/kernels/qwen4exp_kernels.h is the host half); the reused sources run at this
# family's shapes:
#   gemv.cl         qkv||z 2560x16384 (S1 L0: gdn_step / prep_gated_head read S 1), q||gate||k||v 2560x13312 (S2 L0),
#                   out_proj / o_proj 6144x2560 (S8 L0, PERF-109) - ours (int4 g64); PROVISIONAL cells (model::Qwen4ExpDesc's
#                   {S, layout}) until Task 7's sweep
#   gemv_bf16.cl    Intel's interim bf16 dense rows (16384, 13312, 6144x2560), the HC down||inject {10240, 336} and the
#                   final mixer's down {10240, 320}, the indexer 2560 -> 640, the router + shared gate 2560 -> 528, a||b
#                   2560 -> 128, the PLE key||value 2560 -> 12800, lm_head 2560 -> 248320
#   gemv_i8w.cl     the int8 lm_head (spec 9)
#   gdn_step.cl / argmax.cl   Qwen3.8's own binaries (gdn_step_M1: 16 / 48 heads, 10240 conv rows; argmax_stage1_M1:
#                   vocab 248320 / 248077 used) - bound by name, nothing built here
#   attn.cl         attn_prep at 24 q / 2 kv heads (q / k (1 + w) norms, partial RoPE 64, the KV write, the gate),
#                   QKV_S 2 (int4) and 1 (bf16)
#   attn_v2.cl      at 24 / 2 for decision 10's bench only (qwen4exp_kernels_test --bench-attn); never bound
#   embed_gather.cl at hidden 2560, vocab 248320
# What a decode list binds (runtime/qwen4exp/qwen4exp_capture.cc): 15 launches a GDN layer, 19 a QSA layer (18
# under B70_Q4_ATTN=eager), + 4 on the PLE layer, + embed + 6 at the head: 779 at 48 layers (derived).
option(B70_Q4EXP "build the Qwen3.8-Flash-Next decode variants (spec 21-QFN-v1c)" ON)
# One row count's decode set: 21-QFN-v1c's at M = 1 and spec 21-QFN-v1e's verify lists at M = 2..4, by the same names and defines.
# At M = 1 the bf16 out_proj / o_proj 6144 x 2560 is Kolibri's o_proj binary: built here only without Kolibri (the
# int4 one is S8 here, Kolibri's S4).
function(add_q4_decode_rows MM)
  add_gemv_variant(${MM} 2560 16384 1 0)            # qkv||z int4                     S1 L0 (fixed)
  add_gemv_variant(${MM} 2560 13312 2 0)            # q||gate||k||v int4              PROVISIONAL S2 L0
  add_gemv_variant(${MM} 6144 2560 8 0)             # out_proj / o_proj int4          S8 L0 (PERF-109)
  if(MM GREATER 1 OR NOT B70_KOLIBRI)
    add_gemv_bf16_variant(${MM} 6144 2560 64 1)     # out_proj / o_proj bf16
  endif()
  add_gemv_bf16_variant(${MM} 2560 128 16 16)       # a||b {16, 16}
  add_gemv_bf16_variant(${MM} 2560 16384 64 1)      # qkv||z bf16 (Intel's)
  add_gemv_bf16_variant(${MM} 2560 13312 64 1)      # q||gate||k||v bf16 (Intel's, the MTP head's)
  add_gemv_bf16_variant(${MM} 10240 336 16 16)      # HC down || block_inject (324 rows used)
  add_gemv_bf16_variant(${MM} 10240 320 16 16)      # the final mixer's down
  add_gemv_bf16_variant(${MM} 2560 640 16 16)       # the indexer's index_qk_proj
  add_gemv_bf16_variant(${MM} 2560 528 16 16)       # the router (512) + shared_expert_gate (row 512), zero to 528
  add_gemv_bf16_variant(${MM} 2560 12800 64 1)      # the PLE key || value projections
  add_gemv_bf16_variant(${MM} 2560 248320 64 1)     # lm_head bf16
  add_gemv_i8w_variant(${MM} 2560 248320)           # lm_head int8 (spec 9)
  add_ocloc_kernel(embed_gather_M${MM}_D2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} HIDDEN=2560 VOCAB=248320)
  add_ocloc_kernel(prep_gated_head_M${MM}_SIG SOURCE ${PREP_CL} DEFINES M=${MM} GDN_GATE_SIGMOID=1)
  add_ocloc_kernel(attn_prep_M${MM}_Q24KV2 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} ${Q4_FA_HEADS})
  add_ocloc_kernel(attn_prep_M${MM}_Q24KV2_S1 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1 ${Q4_FA_HEADS})
  # q4_hc.cl: the combine + grouped norm by what it folds (_E embedding, _S<S> the mixer's slices, _Y the MoE's y,
  # _X nothing; _NN without the norm) and the up-mix (_I with the 4 inject rows; the final mixer without).
  add_ocloc_kernel(q4_hc_combine_norm_M${MM}_E SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=0 HC_NORM=1)
  foreach(S 1 8)
    add_ocloc_kernel(q4_hc_combine_norm_M${MM}_S${S} SOURCE ${Q4_CL}/q4_hc.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=1 SRC_S=${S} HC_NORM=1)
  endforeach()
  add_ocloc_kernel(q4_hc_combine_norm_M${MM}_Y SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=2 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M${MM}_Y_NN SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=2 HC_NORM=0)
  add_ocloc_kernel(q4_hc_combine_norm_M${MM}_X SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=3 HC_NORM=1)
  # The up-mix on sub-group block reads (_B, PERF-106), a work-group a row; PERF-66's int8 form
  # (Qwen4ExpDesc::hc_int8) - the up-mix over int8 up tiles (_Q8) and the down||inject / final mixer's down GEMV
  # (q4_hc_down_i8, in gemv_bf16's place; KS32: 32 sub-groups split a tile's K, kernels::qwen4exp::kHcDownI8Ks)
  add_ocloc_kernel(q4_hc_up_mix_M${MM}_I_B SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} UP_DOWN=336 UP_INJECT=1 UP_BLK=1)
  add_ocloc_kernel(q4_hc_up_mix_M${MM}_B SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} UP_DOWN=320 UP_INJECT=0 UP_BLK=1)
  add_ocloc_kernel(q4_hc_up_mix_M${MM}_I_Q8_B SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} UP_DOWN=336 UP_INJECT=1 UP_I8=1 UP_BLK=1)
  add_ocloc_kernel(q4_hc_up_mix_M${MM}_Q8_B SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} UP_DOWN=320 UP_INJECT=0 UP_I8=1 UP_BLK=1)
  foreach(N 336 320)
    add_ocloc_kernel(q4_hc_down_i8_M${MM}_N${N}_KS32 SOURCE ${Q4_CL}/q4_hc.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} HC_DOWN_I8=${N} DN_KS=32)
  endforeach()
  # PERF-106's bench only (qwen4exp_kernels_test --bench-hc8): lead 4's pair (the down writes the activations, _A;
  # the up-mix reads them, at M >= 2 in lead 2's _R<M>C16 form) against the lists' - removed with the box verdict
  if(MM EQUAL 1)
    set(_upr "")
    set(_upr_d "")
  else()
    set(_upr _R${MM}C16)
    set(_upr_d UP_ROWS=${MM} UP_COLS=16)
  endif()
  add_ocloc_kernel(q4_hc_up_mix_M${MM}_I${_upr}_Q8_B_A SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} UP_DOWN=336 UP_INJECT=1 UP_I8=1 UP_BLK=1 UP_ACT=1 ${_upr_d})
  add_ocloc_kernel(q4_hc_down_i8_M${MM}_N336_KS32_A SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} HC_DOWN_I8=336 DN_KS=32 DN_ACT=1)
  # q4_ple.cl: the gather from host USM by the table's scale dtype (+ q4_ple_check).
  add_ocloc_kernel(q4_ple_gather_M${MM}_BF16 SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=1)
  add_ocloc_kernel(q4_ple_gather_M${MM}_F32 SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=0)
  # q4_qsa.cl (prep, score, select), q4_qsa_attn.cl (flash: attn + reduce), q4_qsa_attn_eager.cl.
  # _I4H: the indexer scored in int4 x int4 on DPAS after the 128-point Hadamard - the engine's form
  add_ocloc_kernel(q4_qsa_M${MM}_T512_W1024_I4H SOURCE ${Q4_CL}/q4_qsa.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} TOPB=512 SEL_WG=1024 ${Q4_LIST_DEFINES} QSA_I4H=1)
  if(MM EQUAL 1)   # the bf16 form: the kernel test's and bench's reference
    add_ocloc_kernel(q4_qsa_M${MM}_T512_W1024 SOURCE ${Q4_CL}/q4_qsa.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TOPB=512 SEL_WG=1024 ${Q4_LIST_DEFINES})
  endif()
  add_ocloc_kernel(q4_qsa_attn_M${MM}_T32 SOURCE ${Q4_CL}/q4_qsa_attn.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} TGT=32 ${Q4_LIST_DEFINES})
  add_ocloc_kernel(q4_qsa_attn_eager_M${MM} SOURCE ${Q4_CL}/q4_qsa_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} ${Q4_LIST_DEFINES} LISTMAX=2052)
  # q4_moe.cl: the 512-expert softmax route (two experts a lane), the 11-slot block by the shared expert's form.
  # UP_KS / DN_KS PROVISIONAL (kernels::qwen4exp::kUpKs / kDnKs; Task 7 sweeps them).
  add_ocloc_kernel(q4_route_M${MM}_E512_T10_N528_L256 SOURCE ${Q4_CL}/q4_moe.cl
                   DEFINES M=${MM} ROUTE_E=512 ROUTE_K=10 ROUTE_WG=256 ROUTE_LN=528)
  foreach(SHB 0 1)
    if(SHB)
      set(_sh SHB)
    else()
      set(_sh SH4)
    endif()
    add_ocloc_kernel(q4_moe_M${MM}_E512_T10_D2560_I640_${_sh} SOURCE ${Q4_CL}/q4_moe.cl
                     DEFINES M=${MM} MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=640 UP_KS=4 DN_KS=2 SHARED_BF16=${SHB})
    # plan 21-QFN-v2a Task 8: the expert tier's form - a routed expert's blocks through the layer's u64 table row
    add_ocloc_kernel(q4_moe_M${MM}_E512_T10_D2560_I640_${_sh}_TBL SOURCE ${Q4_CL}/q4_moe.cl
                     DEFINES M=${MM} MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=640 UP_KS=4 DN_KS=2 SHARED_BF16=${SHB}
                             Q4_EXPERT_TABLE=1)
  endforeach()
endfunction()
if(B70_Q4EXP)
  set(Q4_CL ${CMAKE_CURRENT_SOURCE_DIR}/qwen4exp)
  set(Q4_FA_HEADS FA_Q_HEADS=24 FA_KV_HEADS=2)
  # LIST_ROW / COUNT_W / LISTMAX: kernels::qwen4exp::kListRow / kCountWord / kListMax; TGT: kAttnTgt.
  set(Q4_LIST_DEFINES LIST_ROW=2064 COUNT_W=2052)
  add_q4_decode_rows(1)
  add_ocloc_kernel(attn_v2_M1_T${ATTN_V2_TGT}_Q24KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} ${Q4_FA_HEADS})
  add_ocloc_kernel(q4_ple_block_M1 SOURCE ${Q4_CL}/q4_ple.cl DEFINES ${CTRL_DEFINES} M=1 PLE_BLOCK=1)   # the PLE block
  # plan 21-QFN-v2a Task 8: the expert tier's load check (sampled block words through the table, once at construction)
  add_ocloc_kernel(q4_expert_check SOURCE ${Q4_CL}/q4_expert_check.cl)
endif()
# ==== Spec 21-QFN-v1c: Qwen3.8-Flash-Next decode ==== (end) ====

# ==== Plan 21-QFN-v2b: Qwen3.8-Flash-Next tensor parallel decode (`--parallel tp`) ==== (begin) ====
# Additive only: new names; q4_qsa_attn.cl's Q4_Q_HEADS / Q4_KV_HEADS and q4_moe.cl's MOE_TP_PART are unset in every
# existing binary. A card's M = 1 decode list at model::q4_tp_card's shapes (runtime/qwen4exp/qwen4exp_capture.cc, tp):
#   linears  qkv||z 2560 x 8192 S1, q||gate||k||v 2560 x 6656 S2, out_proj / o_proj 3072 x 2560 S2 (the card's K rows),
#            all int4 L0; the int8 head's vocabulary half 2560 x 124160; a||b, the HCs, router, indexer, PLE one card's
#   GDN      gdn_step_M1_GK8V24 (Qwen3.8's card: the same 8 / 24 heads and conv 5120) and prep_gated_head _SIG at 8 / 24
#   QSA      attn_prep at 12 q / 1 kv heads (S2), q4_qsa_attn + _reduce at 12 / 1
#   MoE      q4_moe at the half width 320 (DN_KS 1: 5 groups), each slot's fp32 partial (MOE_TP_PART), direct and _TBL,
#            a card each (RANK): the down sends its slots to the peer, q4_moe_finish waits, sums and runs the epilogue
#   tp.cl    tp_send + tp_sum at hidden 2560 per rank, the slices form: out_proj / o_proj's 4 local slices summed in one
#            card's S8 order into one row (the combine's _S1; PERF-109). With these and the MoE's wired sums every value is one
#            card's, bit for bit. tp_argmax per vocabulary half is Qwen3.8's
#   PERF-44  the decode lists bind out_proj / o_proj's gemv with tp_send folded in (_TP, GEMV_TP_SEND: chunk = one
#            work-group's 64 columns of a slice) and tp_sum waiting on those chunks (_sum_W64, TP_WIRE_COLS 64): the
#            same sums, one launch and one pass fewer a layer. The plain gemv and _sum stay: qwen4exp_tp_wire_test's
#            control
if(B70_Q4EXP AND B70_TP)
  add_gemv_variant(1 2560 8192 1 0)
  add_gemv_variant(1 2560 6656 2 0)
  add_gemv_variant(1 3072 2560 4 0)   # one card's S8 slices 4r .. 4r + 3: the slices crossing is one card's chain
  add_gemv_variant(1 3072 2560 4 0 TP)
  add_gemv_variant(1 3072 2560 2 0)   # PERF-109's controls (qwen4exp_tp_wire_test): the S2 split as it was
  add_gemv_variant(1 3072 2560 2 0 TP)
  add_gemv_i8w_variant(1 2560 124160)
  add_ocloc_kernel(prep_gated_head_M1_GK8V24_SIG SOURCE ${PREP_CL}
                   DEFINES M=1 GDN_GATE_SIGMOID=1 GDN_K_HEADS=8 GDN_V_HEADS=24)
  add_ocloc_kernel(attn_prep_M1_Q12KV1 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=1 FA_Q_HEADS=12 FA_KV_HEADS=1)
  add_ocloc_kernel(q4_qsa_attn_M1_T32_Q12KV1 SOURCE ${Q4_CL}/q4_qsa_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=32 ${Q4_LIST_DEFINES} Q4_Q_HEADS=12 Q4_KV_HEADS=1)
  foreach(R 0 1)
    add_ocloc_kernel(q4_moe_M1_E512_T10_D2560_I320_SH4_TP_R${R} SOURCE ${Q4_CL}/q4_moe.cl
                     DEFINES M=1 MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=320 UP_KS=4 DN_KS=1 SHARED_BF16=0 MOE_TP_PART=1
                             RANK=${R})
    add_ocloc_kernel(q4_moe_M1_E512_T10_D2560_I320_SH4_TBL_TP_R${R} SOURCE ${Q4_CL}/q4_moe.cl
                     DEFINES M=1 MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=320 UP_KS=4 DN_KS=1 SHARED_BF16=0 MOE_TP_PART=1
                             Q4_EXPERT_TABLE=1 RANK=${R})
    foreach(S 4 2)   # the S2 pair: PERF-109's control
      add_ocloc_kernel(tp_cross_M1_K2560_G20_S${S}_X_R${R}_sum SOURCE ${TP_CL}
                       DEFINES TP_CROSS=1 M=1 K=2560 FOLD_G=20 S_LOC=${S} FOLDED=0 RANK=${R} TP_SUM=1)
      add_ocloc_kernel(tp_cross_M1_K2560_G20_S${S}_X_R${R}_sum_W64 SOURCE ${TP_CL}
                       DEFINES TP_CROSS=1 M=1 K=2560 FOLD_G=20 S_LOC=${S} FOLDED=0 RANK=${R} TP_SUM=1 TP_WIRE_COLS=64)
    endforeach()
  endforeach()
endif()
# ==== Plan 21-QFN-v2b: Qwen3.8-Flash-Next tensor parallel decode ==== (end) ====

# ==== Plan 21-QFN-v2b Task 5: Qwen3.8-Flash-Next MTP under tensor parallel ==== (begin) ====
# Additive only: new names. A card's verify list at M = 1..4 (runtime/qwen4exp/qwen4exp_capture.cc, tp + Verify) at
# model::q4_tp_card's shapes beyond the M = 1 decode list's: the GDN step's DEFERRED build at 8 / 24 heads (PERF-51; one
# layer's pending rows a parity, layer-major as gdn_step_defer_M<M>_G1), and at M = 2..4 the card's linears (gemv, every
# row the M = 1 step's arithmetic), prep_gated_head _SIG, attn_prep and q4_qsa_attn at 12 / 1 heads, the MoE's halves
# and the slices crossing. tp_argmax at M = 2..4 is Qwen3.8's (spec 17d, the same vocabulary halves); the MTP head
# runs whole on card 1 with spec 21-QFN-v1e's binaries.
if(B70_Q4EXP AND B70_TP AND B70_MTP)
  foreach(MM 1 2 3 4)
    # PERF-51: the verify lists' DEFERRED build at the card's 8 / 24 heads, layer-major as gdn_step_defer_M<M>_G1 (it
    # replaces gdn_step_slots_M<M>_GK8V24_G1): PEND_PAR = ONE layer's pending rows, 64 rows x (8 + 24) x 128 bf16 +
    # 64 x 24 x 2 fp32 (kernels::gdn_pend_layer_bytes)
    add_ocloc_kernel(gdn_step_defer_M${MM}_GK8V24_G1 SOURCE ${GDN_CL}
                     DEFINES ${CTRL_DEFINES} NEGA_OFF=20480 DTBIAS_OFF=20504 GDN_K_HEADS=8 GDN_V_HEADS=24 DEFERRED=1
                             CTRL_PEND=30 PEND_PAR=536576 M=${MM})
    if(MM GREATER 1)
      add_gemv_variant(${MM} 2560 8192 1 0)
      add_gemv_variant(${MM} 2560 6656 2 0)
      add_gemv_variant(${MM} 3072 2560 4 0)
      add_gemv_variant(${MM} 3072 2560 4 0 TP)
      add_gemv_variant(${MM} 3072 2560 2 0)   # PERF-109's control: the S2 split as it was
      add_gemv_i8w_variant(${MM} 2560 124160)
      add_ocloc_kernel(prep_gated_head_M${MM}_GK8V24_SIG SOURCE ${PREP_CL}
                       DEFINES M=${MM} GDN_GATE_SIGMOID=1 GDN_K_HEADS=8 GDN_V_HEADS=24)
      add_ocloc_kernel(attn_prep_M${MM}_Q12KV1 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} FA_Q_HEADS=12 FA_KV_HEADS=1)
      add_ocloc_kernel(q4_qsa_attn_M${MM}_T32_Q12KV1 SOURCE ${Q4_CL}/q4_qsa_attn.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=32 ${Q4_LIST_DEFINES} Q4_Q_HEADS=12 Q4_KV_HEADS=1)
      foreach(R 0 1)
        add_ocloc_kernel(q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TP_R${R} SOURCE ${Q4_CL}/q4_moe.cl
                         DEFINES M=${MM} MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=320 UP_KS=4 DN_KS=1 SHARED_BF16=0
                                 MOE_TP_PART=1 RANK=${R})
        add_ocloc_kernel(q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TBL_TP_R${R} SOURCE ${Q4_CL}/q4_moe.cl
                         DEFINES M=${MM} MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=320 UP_KS=4 DN_KS=1 SHARED_BF16=0
                                 MOE_TP_PART=1 Q4_EXPERT_TABLE=1 RANK=${R})
        add_ocloc_kernel(tp_cross_M${MM}_K2560_G20_S4_X_R${R}_sum SOURCE ${TP_CL}
                         DEFINES TP_CROSS=1 M=${MM} K=2560 FOLD_G=20 S_LOC=4 FOLDED=0 RANK=${R} TP_SUM=1)
        add_ocloc_kernel(tp_cross_M${MM}_K2560_G20_S4_X_R${R}_sum_W64 SOURCE ${TP_CL}
                         DEFINES TP_CROSS=1 M=${MM} K=2560 FOLD_G=20 S_LOC=4 FOLDED=0 RANK=${R} TP_SUM=1
                                 TP_WIRE_COLS=64)
        # open-work item 48: the union-read MoE on the card's halves (q4_moe_finish stays the per-row binary's)
        set(_ud M=${MM} MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=320 UP_KS=4 DN_KS=1 SHARED_BF16=0 MOE_TP_PART=1 RANK=${R})
        add_ocloc_kernel(q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TP_R${R}_U SOURCE ${Q4_CL}/q4_moe_union.cl DEFINES ${_ud})
        add_ocloc_kernel(q4_moe_M${MM}_E512_T10_D2560_I320_SH4_TBL_TP_R${R}_U SOURCE ${Q4_CL}/q4_moe_union.cl
                         DEFINES ${_ud} Q4_EXPERT_TABLE=1)
      endforeach()
    endif()
  endforeach()
  add_ocloc_kernel(q4_moe_M4_E512_T10_D2560_I320_SH4_TBL_TP_R0_U_CNT SOURCE ${Q4_CL}/q4_moe_union.cl
                   DEFINES M=4 MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=320 UP_KS=4 DN_KS=1 SHARED_BF16=0 MOE_TP_PART=1
                           RANK=0 Q4_EXPERT_TABLE=1 UNION_COUNT=1)
endif()
# ==== Plan 21-QFN-v2b Task 5: Qwen3.8-Flash-Next MTP under tensor parallel ==== (end) ====


# ==== Spec 21-QFN-v1d: Qwen3.8-Flash-Next prefill ==== (begin) ====
# Additive but for three sources, each gaining one define that every existing binary leaves unset - their
# preprocessed source token for token as before (the 6499948 rule; G0's sha256 on the box, r33.k0):
#   prefill/pf_gated_head.cl  GDN_GATE_SIGMOID (pf_gated_head_SIG: the sigmoid gate, prep.cl's _SIG twin)
#   qwen4exp/q4_qsa.cl        QSA_PF (the prep at M = 2048 without its in-launch ring write, + q4_qsa_ring; the indexer
#                             rows at the GEMM's pitch 768)
#   qwen4exp/q4_ple.cl        PLE_PF (the gather's tokens from the chunk's id buffer, no ring write)
# New family sources (src/kernels/qwen4exp/, kernels::qwen4exp's spec 21-QFN-v1d block is the host half):
#   q4_pf_moe.cl    the 512-expert ushort sort, gather, a weight batch's dequant (block 512 = the shared expert: int4
#                   dequantised or Intel's bf16 copied), the combine in decode's order
#   q4_pf_attn.cl   the sparse flash over each row's own list (rows past 2050; DPAS; flash and eager forms)
#   q4_pf_ple.cl    the PLE block over a chunk in three launches (gate, conv, the rings)
# Reused at this family's shapes: 21-QFN-v1c's q4_hc.cl at M = 2048 (combine _E / _S1 / _Y / _X / _Y_NN, up-mix _I at the
# GEMM's pitch 512 in the _R4C64 form), prefill/pf_embed.cl and pf_gemv_bf16.cl (a||b, the router at decode's {16, 16} tiling) at hidden
# 2560, pf_dequant_slab.cl (qkv||z, q||gate||k||v int4 layout 0), prefill/slab_bf16.cl's bf16 slab (q4_ prefix),
# pf_attn_prep.cl (q16), pf_attn.cl (the gate) and pf_flash_attn.cl at 24 q / 2 kv heads (HPW 6 of GQA 12, KT 64, RPW
# 8, EXP2 as the existing binary, 256 GRF), pf_moe_gemm.cl (gate||up 2560 x 1280 SiLU, down 640 x 2560); bound by
# name, nothing built here: Qwen3.8's pf_gdn_conv / _intra / _scan_OB / _front_PL (the GDN shape is Qwen3.8's), pf_gemm_T0, 21-QFN-v1c's
# q4_route_M1 (grid (1, C)); Kolibri's pf_int4_slab_K6144_N2560 (out_proj / o_proj int4) is built here
# only when Kolibri's block is off. What a chunk binds: runtime/qwen4exp/qwen4exp_prefill.cc (launch counts in
# runtime/qwen4exp/qwen4exp_sizes.cc: they depend on the chunk's dense / sparse rows).
if(B70_Q4EXP)
  set(Q4_PF_REUSE ${CMAKE_CURRENT_SOURCE_DIR}/prefill)
  # PERF-20 / PERF-21: the GDN walk on Qwen3.8's one-card bitwise forms (prefill/CMakeLists.txt: pf_gdn_front_PL, the
  # conv + l2norm with a lane's partials read first; pf_gdn_scan_OB, o written as bf16 words with q.S beside w.S and two
  # barriers a step) and the gated head one sub-group a (position, head) on that bf16 o (HEAD_SG HEAD_OBF16). The plain
  # pf_gated_head_SIG and pf_gdn_front stay, test-only: qwen4exp_pf_gdn_bundle_test's old walk.
  add_ocloc_kernel(pf_gated_head_SIG SOURCE ${Q4_PF_REUSE}/pf_gated_head.cl DEFINES GDN_GATE_SIGMOID=1)
  add_ocloc_kernel(pf_gated_head_SIG_OB SOURCE ${Q4_PF_REUSE}/pf_gated_head.cl
                   DEFINES GDN_GATE_SIGMOID=1 HEAD_SG=1 HEAD_OBF16=1)
  # MoE-model prefill phase 2 (Task 8): the GDN front at Qwen3.8's GDN shape, which this family shares.
  add_ocloc_kernel(pf_gdn_front SOURCE ${Q4_PF_REUSE}/pf_gdn_front.cl)
  # The qkv||z partials as the bf16 words the front and the head round them to (a chunk past kPfSmallMaxRows rows:
  # pf_gemm_T0_B16, prefill/CMakeLists.txt): the front / head forms reading them (QKVZ_BF16, Ornith's _PB) - bitwise.
  add_ocloc_kernel(pf_gated_head_SIG_OB_PB SOURCE ${Q4_PF_REUSE}/pf_gated_head.cl
                   DEFINES GDN_GATE_SIGMOID=1 HEAD_SG=1 HEAD_OBF16=1 QKVZ_BF16=1)
  add_ocloc_kernel(pf_gdn_front_PL_PB SOURCE ${Q4_PF_REUSE}/pf_gdn_front.cl DEFINES FRONT_PRELOAD=1 QKVZ_BF16=1)
  # MoE-model prefill phase 3 Task 8: the router and a||b by pf_gemm_T0, as Ornith's (prefill/CMakeLists.txt) - the
  # weights re-laid as zero-padded [K][P] slabs each chunk, and the two consumers at the GEMM's row pitch P (decode's
  # q4_route keeps its).
  add_ocloc_kernel(pf_route_slab_K2560_N528_P768 SOURCE ${Q4_PF_REUSE}/pf_route.cl DEFINES K=2560 N=528 ROUTE_SLAB=768)
  add_ocloc_kernel(pf_route_slab_K2560_N128_P256 SOURCE ${Q4_PF_REUSE}/pf_route.cl DEFINES K=2560 N=128 ROUTE_SLAB=256)
  # PERF-27: one sub-group a token (ROUTE_SG; the name kept, the walk launches it by its reqd_work_group_size)
  add_ocloc_kernel(q4_route_M1_E512_T10_N528_L256_P768 SOURCE ${Q4_CL}/q4_moe.cl
                   DEFINES M=1 ROUTE_E=512 ROUTE_K=10 ROUTE_WG=256 ROUTE_LN=768 ROUTE_SG=1)
  # pf_gdn_conv_AB256 (pf_gdn_gate at pitch 256) is Qwen3.8's binary: prefill/CMakeLists.txt builds it.
  add_ocloc_kernel(pf_embed_gather_D2560 SOURCE ${Q4_PF_REUSE}/pf_embed.cl DEFINES HIDDEN=2560 VOCAB=248320)
  add_ocloc_kernel(pf_ab_proj_D2560 SOURCE ${Q4_PF_REUSE}/pf_gemv_bf16.cl DEFINES K=2560 N=128 COLS_PER_WG=16 KSPLIT=16)
  add_ocloc_kernel(pf_moe_router_K2560_N528 SOURCE ${Q4_PF_REUSE}/pf_gemv_bf16.cl
                   DEFINES K=2560 N=528 COLS_PER_WG=16 KSPLIT=16)
  foreach(row "2560 16384" "2560 13312")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(pf_dequant_slab_K${K}_N${N}_L0 SOURCE ${Q4_PF_REUSE}/pf_dequant_slab.cl DEFINES K=${K} N=${N} LAYOUT=0)
  endforeach()
  if(NOT B70_KOLIBRI)
    add_ocloc_kernel(pf_int4_slab_K6144_N2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/slab_dequant_int4.cl DEFINES K=6144 N=2560)
  endif()
  # the bf16 slabs: Intel's dense rows (qkv||z, q||gate||k||v, out / o), the indexer, the HC down||inject (336 rows,
  # a 512-column slab), the PLE key||value
  foreach(row "2560 16384" "2560 13312" "6144 2560" "2560 640" "10240 336" "2560 12800")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(q4_pf_bf16_slab_K${K}_N${N} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/slab_bf16.cl
                     DEFINES K=${K} N=${N})
  endforeach()
  # 21-QFN-v1c's sources at M = 2048 (kernels::qwen4exp::kPfC)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_E SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=0 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_S1 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=1 SRC_S=1 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_Y SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=2 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_Y_NN SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=2 HC_NORM=0)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_X SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=3 HC_NORM=1)
  add_ocloc_kernel(q4_hc_up_mix_M2048_I_D512_R4C64 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 UP_DOWN=512 UP_INJECT=1 UP_ROWS=4 UP_COLS=64)
  # PERF-66: the int8 form - the up-mix over int8 up tiles and the down||inject slab dequantised from int8
  add_ocloc_kernel(q4_hc_up_mix_M2048_I_D512_R4C64_Q8 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 UP_DOWN=512 UP_INJECT=1 UP_ROWS=4 UP_COLS=64 UP_I8=1)
  add_ocloc_kernel(q4_hc_slab_i8_N336 SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} HC_SLAB_I8=336)
  add_ocloc_kernel(q4_qsa_M2048_T512_W1024_PF SOURCE ${Q4_CL}/q4_qsa.cl
                   DEFINES ${CTRL_DEFINES} M=2048 TOPB=512 SEL_WG=1024 ${Q4_LIST_DEFINES} QSA_PF=1 SCORE_R=4)
  add_ocloc_kernel(q4_qsa_M2048_T512_W1024_PF_I4H SOURCE ${Q4_CL}/q4_qsa.cl
                   DEFINES ${CTRL_DEFINES} M=2048 TOPB=512 SEL_WG=1024 ${Q4_LIST_DEFINES} QSA_PF=1 SCORE_R=4 QSA_I4H=1)
  add_ocloc_kernel(q4_ple_gather_M2048_BF16_PF SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=2048 PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=1 PLE_PF=1)
  add_ocloc_kernel(q4_ple_gather_M2048_F32_PF SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=2048 PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=0 PLE_PF=1)
  # PERF-36: the binaries above that bake M, again at M = 4096 (kernels::qwen4exp::kPfCWide, the 4096-row chunk) - M only
  # bounds a launch's rows, so a row's arithmetic is the M = 2048 build's - and the whole int4 slabs' block-store form
  # for the wide chunk's N-fastest walk (Qwen3.8's open-work 18 pair with pf_gemm_T0_R, bit-identical to the plain one)
  add_ocloc_kernel(q4_hc_combine_norm_M4096_E SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=4096 HC_SRC=0 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M4096_S1 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=4096 HC_SRC=1 SRC_S=1 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M4096_Y SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=4096 HC_SRC=2 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M4096_Y_NN SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=4096 HC_SRC=2 HC_NORM=0)
  add_ocloc_kernel(q4_hc_combine_norm_M4096_X SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=4096 HC_SRC=3 HC_NORM=1)
  add_ocloc_kernel(q4_hc_up_mix_M4096_I_D512_R4C64 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=4096 UP_DOWN=512 UP_INJECT=1 UP_ROWS=4 UP_COLS=64)
  add_ocloc_kernel(q4_hc_up_mix_M4096_I_D512_R4C64_Q8 SOURCE ${Q4_CL}/q4_hc.cl   # PERF-66's int8 form
                   DEFINES ${CTRL_DEFINES} M=4096 UP_DOWN=512 UP_INJECT=1 UP_ROWS=4 UP_COLS=64 UP_I8=1)
  add_ocloc_kernel(q4_qsa_M4096_T512_W1024_PF_I4H SOURCE ${Q4_CL}/q4_qsa.cl
                   DEFINES ${CTRL_DEFINES} M=4096 TOPB=512 SEL_WG=1024 ${Q4_LIST_DEFINES} QSA_PF=1 SCORE_R=4 QSA_I4H=1)
  add_ocloc_kernel(q4_ple_gather_M4096_BF16_PF SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=4096 PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=1 PLE_PF=1)
  add_ocloc_kernel(q4_ple_gather_M4096_F32_PF SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=4096 PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=0 PLE_PF=1)
  add_ocloc_kernel(q4_pf_ple_C4096 SOURCE ${Q4_CL}/q4_pf_ple.cl DEFINES ${CTRL_DEFINES} PF_C=4096)
  foreach(N 16384 13312)
    add_ocloc_kernel(pf_dequant_slab_K2560_N${N}_L0_BS SOURCE ${Q4_PF_REUSE}/pf_dequant_slab.cl
                     DEFINES K=2560 N=${N} LAYOUT=0 BLOCK_STORE=1)
  endforeach()
  # the attention: q16 prep, the gate, the dense flash (rows <= 2050) at 24 / 2 heads; the sparse flash and its eager
  # twin (B70_Q4_ATTN=eager: one switch for decode and prefill)
  set(Q4_PF_FA FA_Q_HEADS=24 FA_KV_HEADS=2)
  add_ocloc_kernel(pf_attn_prep_q16_Q24KV2 SOURCE ${Q4_PF_REUSE}/pf_attn_prep.cl
                   DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256 Q_BF16=1 ${Q4_PF_FA})
  add_ocloc_kernel(pf_attn_Q24KV2 SOURCE ${Q4_PF_REUSE}/pf_attn.cl DEFINES ${Q4_PF_FA})
  add_ocloc_kernel(pf_flash_attn_Q24KV2 SOURCE ${Q4_PF_REUSE}/pf_flash_attn.cl
                   DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 ${Q4_PF_FA} OPTIONS -cl-intel-256-GRF-per-thread)
  foreach(EG 0 1)
    set(_n q4_pf_sparse_attn_Q24KV2)
    if(EG)
      string(APPEND _n _EAGER)
    endif()
    add_ocloc_kernel(${_n} SOURCE ${Q4_CL}/q4_pf_attn.cl DEFINES KT=32 EAGER=${EG} ${Q4_LIST_DEFINES}
                     OPTIONS -cl-intel-256-GRF-per-thread)
  endforeach()
  # the MoE: the sort / gather / dequant / combine (TM and KC: kernels::qwen4exp::kPfTm / kPfC), the grouped GEMMs
  add_ocloc_kernel(q4_pf_moe_E512_T10_D2560_I640_L256 SOURCE ${Q4_CL}/q4_pf_moe.cl
                   DEFINES PF_E=512 PF_K=10 PF_WG=256 PF_EPL=2 PF_D=2560 PF_INTER=640 TM=32 KC=2048)
  # test-only (tests/kernels/qwen4exp_pf_glue_test.cc, PERF-27): the combine one column a lane, the comparison's form
  add_ocloc_kernel(q4_pf_moe_E512_T10_D2560_I640_L256_C1 SOURCE ${Q4_CL}/q4_pf_moe.cl
                   DEFINES PF_E=512 PF_K=10 PF_WG=256 PF_EPL=2 PF_D=2560 PF_INTER=640 TM=32 KC=2048 PF_COMBINE_C1=1)
  add_ocloc_kernel(pf_moe_gemm_K2560_N1280_SILU SOURCE ${Q4_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=2560 N=1280 TM=32 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_K640_N2560 SOURCE ${Q4_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=640 N=2560 TM=32 OPTIONS -cl-intel-256-GRF-per-thread)
  # plan 2026-11-12 Task 2: the multi-work-group sort at this family's 512 experts, top 10 (bound in Task 7)
  add_ocloc_kernel(pf_moe_sort_mw_E512_T10 SOURCE ${Q4_PF_REUSE}/pf_moe_sort_mw.cl
                   DEFINES EXPERTS=512 TOP_K=10 TM=32 TB=64 MI=64)
  # plan 2026-11-12 Task 3 / 7: the grouped GEMMs read straight from the int4 blocks, at this family's shapes
  add_ocloc_kernel(pf_moe_q4gemm_K2560_N1280_SILU SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                   DEFINES K=2560 N=1280 EXPERTS=512 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  # PERF-25: the direct expert GEMMs (the down here, the _SHX gate||up below; a tp card's further down) at Ornith's
  # spill-free register plan (prefill/CMakeLists.txt's E1: no register double buffer, the weights two groups and A
  # one step ahead prefetched into L1, A read per k16 half - the same DPAS chain, bitwise; 0.70-0.80x at C 2048).
  # Each one's _DB twin keeps the register double buffer it replaced, for qwen4exp_pf_glue_test's comparison only.
  # The _TBL binaries keep the double buffer: with a fifth of the experts in host memory the plan ran 1.19-2.08x, and
  # without the weight prefetch (PF_W 0) still 1.01x.
  set(Q4_PF_E1 NODB=1 PF_W=2 PF_A=1 AK16=1)
  set(Q4_PF_E1_0 ${Q4_PF_E1})   # by Q4_EXPERT_TABLE: direct on the plan, _TBL (Q4_PF_E1_1, unset) not
  add_ocloc_kernel(pf_moe_q4gemm_K640_N2560 SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                   DEFINES K=640 N=2560 EXPERTS=512 ${Q4_PF_E1} OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_q4gemm_K640_N2560_DB SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                   DEFINES K=640 N=2560 EXPERTS=512 OPTIONS -cl-intel-256-GRF-per-thread)
  # plan 21-QFN-v2a Task 8: the same through the expert tier's table rows (Q4_EXPERT_TABLE, _TBL)
  add_ocloc_kernel(pf_moe_q4gemm_K2560_N1280_SILU_TBL SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                   DEFINES K=2560 N=1280 EXPERTS=512 SILU_EPI=1 Q4_EXPERT_TABLE=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_q4gemm_K640_N2560_TBL SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                   DEFINES K=640 N=2560 EXPERTS=512 Q4_EXPERT_TABLE=1 OPTIONS -cl-intel-256-GRF-per-thread)
  # PERF-27: the walk's gate||up - the shared expert's rows read from x itself (SH_X), q4_pf_gather_routed skipping
  # them; direct and _TBL. The two plain gate||up above stay for the kernel test's comparison.
  foreach(TB 0 1)
    set(_t "")
    if(TB)
      set(_t _TBL)
    endif()
    add_ocloc_kernel(pf_moe_q4gemm_K2560_N1280_SILU${_t}_SHX SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                     DEFINES K=2560 N=1280 EXPERTS=512 SILU_EPI=1 Q4_EXPERT_TABLE=${TB} SH_X=1 ${Q4_PF_E1_${TB}}
                     OPTIONS -cl-intel-256-GRF-per-thread)
    if(NOT TB)
      add_ocloc_kernel(pf_moe_q4gemm_K2560_N1280_SILU_SHX_DB SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                       DEFINES K=2560 N=1280 EXPERTS=512 SILU_EPI=1 Q4_EXPERT_TABLE=0 SH_X=1
                       OPTIONS -cl-intel-256-GRF-per-thread)
    endif()
  endforeach()
  # the PLE block over a chunk
  add_ocloc_kernel(q4_pf_ple_C2048 SOURCE ${Q4_CL}/q4_pf_ple.cl DEFINES ${CTRL_DEFINES} PF_C=2048)
endif()
# ==== Spec 21-QFN-v1d: Qwen3.8-Flash-Next prefill ==== (end) ====

# ==== Plan 21-QFN-v2b Task 4: Qwen3.8-Flash-Next tensor parallel prefill ==== (begin) ====
# Additive only; three sources gain a define every existing binary leaves unset (q4_pf_attn.cl Q4_Q_HEADS /
# Q4_KV_HEADS, pf_moe_q4gemm.cl OUT_F32, q4_pf_moe.cl PF_COMBINE_F32). A card's prefill chunk at model::q4_tp_card's
# shapes (runtime/qwen4exp/qwen4exp_prefill.cc, tp):
#   linears  qkv||z 2560 x 8192 (whole slabs), q||gate||k||v 2560 x 6656 and out_proj / o_proj 3072 x 2560 (a tail):
#            int4 layout 0 and Intel's bf16 tiles
#   GDN      pf_gdn_front _PL, the gate at the GEMM's pitch 256 (the card's small block: negA / dt_bias at 20480 / 20504,
#            as Qwen3.8's card), pf_gdn_scan _OB, pf_gated_head _SIG_OB and pf_gdn_intra (PERF-19) at 8 / 24 heads;
#            pf_gdn_conv _GK8V24 is Qwen3.8's
#   QSA      pf_attn_prep_q16, pf_attn (the gate), pf_flash_attn and q4_pf_sparse_attn at 12 q / 1 kv heads
#   MoE      pf_moe_q4gemm gate||up 2560 x 640 (WGN 4: 128 columns a work-group; the walk's _SHX) and down 320 x 2560 with fp32 rows
#            (OUT_F32), direct and _TBL; q4_pf_moe_combine_f32: the card's fp32 partial of the block's output
#   the fold q4_hc_combine_norm _S2 at M = 2048 (and _NN) and M = 1 (the head's row): the two cards' fp32 partials,
#            card 0's first, one rounding - both cards fold the same two rows, so both hold the same H
if(B70_Q4EXP AND B70_TP)
  add_ocloc_kernel(pf_dequant_slab_K2560_N8192_L0 SOURCE ${Q4_PF_REUSE}/pf_dequant_slab.cl DEFINES K=2560 N=8192 LAYOUT=0)
  foreach(row "2560 6656" "3072 2560")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(pf_int4_slab_K${K}_N${N} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/slab_dequant_int4.cl
                     DEFINES K=${K} N=${N})
  endforeach()
  foreach(row "2560 8192" "2560 6656" "3072 2560")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(q4_pf_bf16_slab_K${K}_N${N} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/slab_bf16.cl
                     DEFINES K=${K} N=${N})
  endforeach()
  set(Q4_TP_GDN GDN_K_HEADS=8 GDN_V_HEADS=24)
  # PERF-20 / PERF-21: the card's walk on one card's bitwise forms at its heads (_PL / _OB / _SIG_OB); Qwen3.8's tp
  # card keeps the plain pf_gdn_scan_GK8V24. The plain front and head stay, test-only (qwen4exp_pf_gdn_bundle_test).
  add_ocloc_kernel(pf_gdn_front_GK8V24 SOURCE ${Q4_PF_REUSE}/pf_gdn_front.cl DEFINES ${Q4_TP_GDN})
  add_ocloc_kernel(pf_gdn_front_GK8V24_PL SOURCE ${Q4_PF_REUSE}/pf_gdn_front.cl DEFINES ${Q4_TP_GDN} FRONT_PRELOAD=1)
  # 256 GRF as prefill's pf_gdn_scan_OB (no spill, so no scratch on the prompt pass's list).
  add_ocloc_kernel(pf_gdn_scan_GK8V24_OB SOURCE ${Q4_PF_REUSE}/pf_gdn_scan.cl
                   DEFINES ${Q4_TP_GDN} SCAN_OBF16=1 SCAN_QS1=2 SCAN_FOLD=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_gdn_conv_GK8V24_AB256 SOURCE ${Q4_PF_REUSE}/pf_gdn_conv.cl
                   DEFINES NEGA_OFF=20480 DTBIAS_OFF=20504 AB_STRIDE=256 ${Q4_TP_GDN})
  add_ocloc_kernel(pf_gated_head_GK8V24_SIG SOURCE ${Q4_PF_REUSE}/pf_gated_head.cl
                   DEFINES GDN_GATE_SIGMOID=1 ${Q4_TP_GDN})
  add_ocloc_kernel(pf_gated_head_GK8V24_SIG_OB SOURCE ${Q4_PF_REUSE}/pf_gated_head.cl
                   DEFINES GDN_GATE_SIGMOID=1 HEAD_SG=1 HEAD_OBF16=1 ${Q4_TP_GDN})
  # PERF-19: the one-launch intra-chunk stages (A, T, W / U, A2) at the card's heads, as one card's pf_gdn_intra.
  add_ocloc_kernel(pf_gdn_intra_GK8V24 SOURCE ${Q4_PF_REUSE}/pf_gdn_intra.cl DEFINES ${Q4_TP_GDN} INTRA_WU2=1)
  add_ocloc_kernel(pf_gdn_front_GK8V24_PL_PB SOURCE ${Q4_PF_REUSE}/pf_gdn_front.cl
                   DEFINES ${Q4_TP_GDN} FRONT_PRELOAD=1 QKVZ_BF16=1)
  add_ocloc_kernel(pf_gated_head_GK8V24_SIG_OB_PB SOURCE ${Q4_PF_REUSE}/pf_gated_head.cl
                   DEFINES GDN_GATE_SIGMOID=1 HEAD_SG=1 HEAD_OBF16=1 QKVZ_BF16=1 ${Q4_TP_GDN})
  set(Q4_TP_PF_FA FA_Q_HEADS=12 FA_KV_HEADS=1)
  add_ocloc_kernel(pf_attn_prep_q16_Q12KV1 SOURCE ${Q4_PF_REUSE}/pf_attn_prep.cl
                   DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256 Q_BF16=1 ${Q4_TP_PF_FA})
  add_ocloc_kernel(pf_attn_Q12KV1 SOURCE ${Q4_PF_REUSE}/pf_attn.cl DEFINES ${Q4_TP_PF_FA})
  add_ocloc_kernel(pf_flash_attn_Q12KV1 SOURCE ${Q4_PF_REUSE}/pf_flash_attn.cl
                   DEFINES KT=64 RPW=8 HPW=6 QREG=0 EXP2=1 ${Q4_TP_PF_FA} OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(q4_pf_sparse_attn_Q12KV1 SOURCE ${Q4_CL}/q4_pf_attn.cl
                   DEFINES KT=32 EAGER=0 ${Q4_LIST_DEFINES} Q4_Q_HEADS=12 Q4_KV_HEADS=1
                   OPTIONS -cl-intel-256-GRF-per-thread)
  foreach(TB 0 1)
    set(_t "")
    if(TB)
      set(_t _TBL)
    endif()
    add_ocloc_kernel(pf_moe_q4gemm_K2560_N640_SILU${_t} SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                     DEFINES K=2560 N=640 EXPERTS=512 SILU_EPI=1 WGN=4 Q4_EXPERT_TABLE=${TB}
                     OPTIONS -cl-intel-256-GRF-per-thread)
    # PERF-27: the walk's (the shared expert's rows from x, SH_X); the one above stays for the kernel test
    # PERF-25: the two bound ones as one card's - direct on Q4_PF_E1 (their _DB twins test-only), _TBL as before
    add_ocloc_kernel(pf_moe_q4gemm_K2560_N640_SILU${_t}_SHX SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                     DEFINES K=2560 N=640 EXPERTS=512 SILU_EPI=1 WGN=4 Q4_EXPERT_TABLE=${TB} SH_X=1 ${Q4_PF_E1_${TB}}
                     OPTIONS -cl-intel-256-GRF-per-thread)
    add_ocloc_kernel(pf_moe_q4gemm_K320_N2560_F32${_t} SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                     DEFINES K=320 N=2560 EXPERTS=512 OUT_F32=1 Q4_EXPERT_TABLE=${TB} ${Q4_PF_E1_${TB}}
                     OPTIONS -cl-intel-256-GRF-per-thread)
    if(NOT TB)
      add_ocloc_kernel(pf_moe_q4gemm_K2560_N640_SILU_SHX_DB SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                       DEFINES K=2560 N=640 EXPERTS=512 SILU_EPI=1 WGN=4 Q4_EXPERT_TABLE=0 SH_X=1
                       OPTIONS -cl-intel-256-GRF-per-thread)
      add_ocloc_kernel(pf_moe_q4gemm_K320_N2560_F32_DB SOURCE ${Q4_PF_REUSE}/pf_moe_q4gemm.cl
                       DEFINES K=320 N=2560 EXPERTS=512 OUT_F32=1 Q4_EXPERT_TABLE=0
                       OPTIONS -cl-intel-256-GRF-per-thread)
    endif()
  endforeach()
  add_ocloc_kernel(q4_pf_moe_combine_f32_E512_T10_D2560 SOURCE ${Q4_CL}/q4_pf_moe.cl
                   DEFINES PF_E=512 PF_K=10 PF_WG=256 PF_EPL=2 PF_D=2560 PF_INTER=640 TM=32 KC=2048 PF_COMBINE_F32=1)
  # test-only (qwen4exp_pf_glue_test, PERF-27): the fp32 combine one column a lane
  add_ocloc_kernel(q4_pf_moe_combine_f32_E512_T10_D2560_C1 SOURCE ${Q4_CL}/q4_pf_moe.cl
                   DEFINES PF_E=512 PF_K=10 PF_WG=256 PF_EPL=2 PF_D=2560 PF_INTER=640 TM=32 KC=2048 PF_COMBINE_F32=1
                           PF_COMBINE_C1=1)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_S2 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=1 SRC_S=2 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_S2_NN SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=1 SRC_S=2 HC_NORM=0)
  add_ocloc_kernel(q4_hc_combine_norm_M1_S2 SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=1 HC_SRC=1 SRC_S=2 HC_NORM=1)
  # PERF-36: the fold at M = 4096 (its two slices sit M rows apart: the 4096-row chunk's slots) and the card's whole
  # qkv||z slabs' block-store form
  add_ocloc_kernel(q4_hc_combine_norm_M4096_S2 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=4096 HC_SRC=1 SRC_S=2 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M4096_S2_NN SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=4096 HC_SRC=1 SRC_S=2 HC_NORM=0)
  add_ocloc_kernel(pf_dequant_slab_K2560_N8192_L0_BS SOURCE ${Q4_PF_REUSE}/pf_dequant_slab.cl
                   DEFINES K=2560 N=8192 LAYOUT=0 BLOCK_STORE=1)
endif()
# ==== Plan 21-QFN-v2b Task 4: Qwen3.8-Flash-Next tensor parallel prefill ==== (end) ====

# ==== Spec 21-QFN-v1e: Qwen3.8-Flash-Next MTP (B70_Q4EXP and B70_MTP) ==== (begin) ====
# Additions only; no existing source changes. kernels::qwen4exp's spec 21-QFN-v1e block is the host half
# (verify_variants / draft_variants / mtp_prefill_variants). What the lists bind (runtime/qwen4exp/qwen4exp_capture.cc):
#   verify at M = k + 1 (spec 8's lists): every 21-QFN-v1c decode kernel at M = 2..4 - the q4_ sources (combine _E / _S1 /
#     _S4 / _Y / _Y_NN / _X, up-mix, the PLE gather, q4_qsa, the attention (flash, eager), the route and the 11-slot
#     MoE), attn_prep _Q24KV2 / _S1, prep_gated_head _SIG, embed_gather D2560, the GEMVs at this family's shapes (gemv
#     int4 S1 / S2 / S4 layout 0, gemv_bf16 for Intel's dense arms, the HC downs, the indexer, the router, a||b, the PLE
#     key||value, lm_head bf16 and int8); Qwen3.8's argmax_stage1_M<M> by name (vocab 248077 used);
#     gdn_step_defer_M<1..4>_G1 (PERF-51; gdn_step_slots_M<M>_G1 and its three state slots a layer before) -
#     gdn_step.cl's DEFERRED (verify-cost-m8's: no per-row state, the committed rows replayed by the next launch) with
#     PEND_PAR = ONE layer's pending rows, 64 rows x (16 + 48) x 128 bf16 + 64 x 48 x 2 fp32 (kernels::
#     gdn_pend_layer_bytes): the engine lays a layer's two parities side by side, layer-major, so no layer count is
#     baked (a device holds its part's GDN layers; Qwen3.8's gdn_step_defer_M<M> bakes 48 layers a parity);
#     q4_pf_ple_C4 - 21-QFN-v1d's three-launch PLE block at C = 4 (q4_ple_block is M = 1 only);
#   the MTP head (draft at M = 1, its KV pass in every verify at M = 1..4, the prefill's pass at M = 2048):
#     q4_mtp.cl (q4_mtp_norm: the gather of t, gemma_rms of the embedding and the SINGLE / per-STREAM RMS of the
#     4-stream R; q4_mtp_fuse: + e on every stream) at M = 1..4 and _PF at 2048; fc_embedding / fc_hidden gemv_bf16
#     {2560, 2560} at M = 1..4 (fc_hidden at M = 4 a row: its 4 streams) and the bf16 slab {2560, 2560} for prefill.
#     The head's own layer is bf16 dense / bf16 shared (loader::q4_mtp_desc) with int4 experts: 21-QFN-v1c's / 21-QFN-v1d's Intel-form
#     binaries serve it.
if(B70_Q4EXP AND B70_MTP)
  foreach(MM 1 2 3 4)
    # PERF-51: the verify lists' DEFERRED build (header above)
    add_ocloc_kernel(gdn_step_defer_M${MM}_G1 SOURCE ${GDN_CL}
                     DEFINES ${GDN_DEFINES} DEFERRED=1 CTRL_PEND=30 PEND_PAR=1073152 M=${MM})
    add_gemv_bf16_variant(${MM} 2560 2560 64 1)   # fc_embedding / fc_hidden
    foreach(NF SINGLE STREAM)
      if(NF STREQUAL SINGLE)
        set(_single 1)
      else()
        set(_single 0)
      endif()
      add_ocloc_kernel(q4_mtp_M${MM}_${NF} SOURCE ${Q4_CL}/q4_mtp.cl DEFINES ${CTRL_DEFINES} M=${MM} MTP_NORM_SINGLE=${_single})
    endforeach()
    if(MM GREATER 1)
      add_q4_decode_rows(${MM})
      # open-work item 48: the union-read MoE (q4_moe_union.cl) - each distinct expert read once a verify; the verify
      # lists bind it at M >= 2 (kernels::qwen4exp::moe_variant's _U), the per-row q4_moe_M<M> stays its reference.
      foreach(SHB 0 1)
        if(SHB)
          set(_sh SHB)
        else()
          set(_sh SH4)
        endif()
        set(_ud M=${MM} MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=640 UP_KS=4 DN_KS=2 SHARED_BF16=${SHB})
        add_ocloc_kernel(q4_moe_M${MM}_E512_T10_D2560_I640_${_sh}_U SOURCE ${Q4_CL}/q4_moe_union.cl DEFINES ${_ud})
        add_ocloc_kernel(q4_moe_M${MM}_E512_T10_D2560_I640_${_sh}_TBL_U SOURCE ${Q4_CL}/q4_moe_union.cl
                         DEFINES ${_ud} Q4_EXPERT_TABLE=1)
      endforeach()
    endif()
  endforeach()
  # the union's counting build (test-only: tests/kernels/qwen4exp_moe_union_test.cc)
  add_ocloc_kernel(q4_moe_M4_E512_T10_D2560_I640_SH4_TBL_U_CNT SOURCE ${Q4_CL}/q4_moe_union.cl
                   DEFINES M=4 MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=640 UP_KS=4 DN_KS=2 SHARED_BF16=0 Q4_EXPERT_TABLE=1
                           UNION_COUNT=1)
  add_ocloc_kernel(q4_pf_ple_C4 SOURCE ${Q4_CL}/q4_pf_ple.cl DEFINES ${CTRL_DEFINES} PF_C=4)
  # the prefill's head pass (kernels::qwen4exp::mtp_prefill_variants; 21-QFN-v1d's binaries otherwise)
  # PERF-85: the head's attn_prep over the kv heads only (pf_attn_prep.cl KV_ONLY, as Ornith's and Qwen3.8's heads)
  add_ocloc_kernel(pf_attn_prep_q16_Q24KV2_KVO SOURCE ${Q4_PF_REUSE}/pf_attn_prep.cl
                   DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256 Q_BF16=1 ${Q4_PF_FA} KV_ONLY=1)
  add_ocloc_kernel(q4_mtp_M2048_SINGLE_PF SOURCE ${Q4_CL}/q4_mtp.cl
                   DEFINES ${CTRL_DEFINES} M=2048 MTP_NORM_SINGLE=1 MTP_PF=1)
  add_ocloc_kernel(q4_mtp_M2048_STREAM_PF SOURCE ${Q4_CL}/q4_mtp.cl
                   DEFINES ${CTRL_DEFINES} M=2048 MTP_NORM_SINGLE=0 MTP_PF=1)
  # PERF-36: the head pass at the 4096-row chunk (M only bounds the rows)
  add_ocloc_kernel(q4_mtp_M4096_SINGLE_PF SOURCE ${Q4_CL}/q4_mtp.cl
                   DEFINES ${CTRL_DEFINES} M=4096 MTP_NORM_SINGLE=1 MTP_PF=1)
  add_ocloc_kernel(q4_mtp_M4096_STREAM_PF SOURCE ${Q4_CL}/q4_mtp.cl
                   DEFINES ${CTRL_DEFINES} M=4096 MTP_NORM_SINGLE=0 MTP_PF=1)
  add_ocloc_kernel(q4_pf_bf16_slab_K2560_N2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/slab_bf16.cl
                   DEFINES K=2560 N=2560)
endif()
# ==== Spec 21-QFN-v1e: Qwen3.8-Flash-Next MTP ==== (end) ====

# ==== PERF-53 (plan 21-QFN-v3b): Qwen3.8-Flash-Next's DFlash 2 drafter ==== (begin) ====
# Additive only: new names. incoai/Qwen3.8-Flash-Next-DFlash2 (kernels::dflash::kFlashNext): hidden 2560, 32 / 8 heads x
# 128, FF 7680, block 8, the 2048 window over a 2048-slot ring, conv group 16 x 2 taps, selector rank 256, the target's
# 248320-row head. Built here: df_prep and df_proj_wide at its hidden and MLP, the drafter's bf16 linears at M 8 and
# the target's head at the 7 draft rows; bound by name from spec 19b's block: df_attn (Qwen3.8's drafter has the same
# 32 / 8 x 128 attention), df_topk over the 248077 tokens, df_select's V2 walk at rank 256.
if(B70_Q4EXP AND B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  add_ocloc_kernel(df_prep_M8_H2560_I7680_Q32KV8_D128_G16T2_R2048_V248320 SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 H=2560 WG=256 EPS=1e-06f VOCAB=248320 GROUP=16 CTAPS=2
                           INTER=7680 NH=32 NKV=8 HD=128 RING=2048)
  add_ocloc_kernel(df_proj_wide_H2560_KV8_D128_R2048 SOURCE ${DF_DIR}/df_proj_wide.cl
                   DEFINES H=2560 WG=256 EPS=1e-06f NKV=8 HD=128 RING=2048)
  # q||k||v, o, gate||up, down, fc (5 taps of 2560), the context's k||v; then the conv kernel projections and the
  # selector's at the tiny tiling (kernels::gemv_bf16_tiling below N 2048)
  foreach(KN "2560 6144" "4096 2560" "2560 15360" "7680 2560" "12800 2560" "2560 2048")
    separate_arguments(KN)
    list(GET KN 0 _k)
    list(GET KN 1 _n)
    if(NOT TARGET kernel_gemv_bf16_M8_K${_k}_N${_n})
      add_gemv_bf16_variant(8 ${_k} ${_n} 64 1)
    endif()
  endforeach()
  foreach(_n 640 256)
    if(NOT TARGET kernel_gemv_bf16_M8_K2560_N${_n}_C16_S16)
      add_gemv_bf16_variant(8 2560 ${_n} 16 16)
    endif()
  endforeach()
  # the target's head at the 7 draft rows: int8 (ours, quantised at load) and bf16
  if(NOT TARGET kernel_gemv_i8w_M7_K2560_N248320)
    add_gemv_i8w_variant(7 2560 248320)
  endif()
  if(NOT TARGET kernel_gemv_bf16_M7_K2560_N248320)
    add_gemv_bf16_variant(7 2560 248320 64 1)
  endif()
endif()
# ==== PERF-53 (plan 21-QFN-v3b): Qwen3.8-Flash-Next's DFlash 2 drafter ==== (end) ====


# ==== Spec 15e: Ornith's MTP lists (model::ornith(), B70_ORNITH and B70_MTP) ===== (begin) ==
# Additive only: no source changes and no existing variant's name or command line moves
# (tools/kernel_cmdlines' diff: additions only). What the verify and draft lists bind on
# Ornith (runtime/capture.cc build_verify / build_draft), beyond its M = 1 decode list:
#   verify, M = 2..4   Ornith's decode list at M rows: the three int4 GEMVs (PROVISIONAL
#                      cells, 15c's {S, layout}), a||b and the router (gemv_bf16 {16, 16}),
#                      the norm pair at K 2048, embed, prep_gated_head, attn_prep and
#                      decode attention v2 at 16 q / 2 kv, the bf16 / int8 lm_head, and
#                      moe.cl at M rows - moe_route / moe_gate_up / moe_down index every
#                      buffer by the row's own group id (get_group_id(1)) and nothing in
#                      them depends on M but the grid, so row m of an M-row launch is the
#                      M = 1 code on row m's inputs (tests/kernels/moe_m_ref_test.cc,
#                      and moe_m_test on the card pin it).
#                      argmax_stage1_M<M>_V248070 (masks from 248070, the base checkpoint's
#                      tokenizer.json) is built beside moe_M<M> below but, since spec 15 §13,
#                      not bound: the int4 checkpoint's tokenizer.json defines 248077 ids, so
#                      the lists bind Qwen3.8's argmax_stage1_M<M>. a||b of the int4
#                      checkpoint is gemv.cl at K 2048 N 128 S1 L1 (spec 15 §13), beside the
#                      bf16 form a checkpoint that keeps it bf16 binds.
#   verify, M = 1..4   gdn_step_slots at 30 GDN layers (`_G30`, SPEC_SLOT_STRIDE = 30 x 32
#                      x 128 x 128 = 15728640 floats) and Ornith's heads; the head's KV fill:
#                      fc (bf16 4096 -> 2048), q||k||v (bf16 2048 -> 9216), the two pre-fc
#                      norms into one [M][4096] row (X_STRIDE), the ZERO_RESID fold and
#                      attn_prep over an S = 1 qkv (QKV_S = 1)
#   draft, M = 1       o (bf16 4096 -> 2048, fc's binary), the post norm's SP1 fold; the
#                      head's MoE layer runs 15c's moe_M1 and router binaries; with a draft
#                      vocabulary (spec 8 §11) the compact head at K 2048
# Not built for Ornith: an int4-checkpoint lm_head at M > 1 (as for Qwen3.8). The int8 KV cache's writers are
# PERF-64's block below.
if(B70_ORNITH AND B70_MTP)
  foreach(MM 2 3 4)
    add_gemv_variant(${MM} 2048 12288 1 1)       # qkv||z            PROVISIONAL S1 L1
    add_gemv_variant(${MM} 4096 2048 4 0)        # out_proj, o_proj  PROVISIONAL S4 L0
    add_gemv_variant(${MM} 2048 9216 2 0)        # q||k||v           PROVISIONAL S2 L0
    add_gemv_bf16_variant(${MM} 2048 128 16 16)  # a||b (a checkpoint that keeps it bf16)
    add_gemv_variant(${MM} 2048 128 1 1)         # a||b int4 (spec 15 §13)  PROVISIONAL S1 L1
    add_gemv_bf16_variant(${MM} 2048 272 16 16)  # the router || shared gate
    add_gemv_bf16_variant(${MM} 2048 248320 64 1)   # lm_head (bf16)
    add_gemv_i8w_variant(${MM} 2048 248320)         # lm_head (int8, spec 9)
    add_prep_res_fold(${MM} 2048 0 20)
    add_prep_res_fold(${MM} 2048 4 20)
    add_prep_norm_finish(${MM} 2048 20 20)
    add_ocloc_kernel(embed_gather_M${MM}_D2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} HIDDEN=2048 VOCAB=248320)
    add_ocloc_kernel(prep_gated_head_M${MM}_GK16V32 SOURCE ${PREP_CL}
                     DEFINES M=${MM} ${ORNITH_GDN_HEADS})
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2 SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} M=${MM} ${ORNITH_FA_HEADS})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q16KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2}
                             ${ORNITH_FA_HEADS})
    add_ocloc_kernel(moe_M${MM}_E256_T8_D2048_I512 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/moe.cl
                     DEFINES ${MOE_DEFINES} M=${MM} EXPERTS=256 TOP_K=8 HIDDEN=2048 INTER=512
                             ROUTER_N=272)
    add_ocloc_kernel(argmax_stage1_M${MM}_V248070 SOURCE ${ARGMAX_CL}
                     DEFINES ${ORNITH_ARGMAX_DEFINES} M=${MM})
  endforeach()
  foreach(MM 1 2 3 4)
    add_ocloc_kernel(gdn_step_slots_M${MM}_G30_GK16V32 SOURCE ${GDN_CL}
                     DEFINES ${CTRL_DEFINES} NEGA_OFF=32768 DTBIAS_OFF=32800 ${ORNITH_GDN_HEADS}
                             SPEC_SLOTS=1 CTRL_LIVE=19 SPEC_SLOT_STRIDE=15728640 M=${MM})
    # PERF-51: the one-card verify lists' DEFERRED build (verify-cost-m8's) at Ornith's shapes; PEND_PAR = 30 layers x
    # (64 rows x (16 + 32) x 128 bf16 + 64 x 32 x 2 fp32) bytes (kernels::gdn_pend_parity_bytes)
    add_ocloc_kernel(gdn_step_defer_M${MM}_G30_GK16V32 SOURCE ${GDN_CL}
                     DEFINES ${CTRL_DEFINES} NEGA_OFF=32768 DTBIAS_OFF=32800 ${ORNITH_GDN_HEADS}
                             DEFERRED=1 CTRL_PEND=30 PEND_PAR=24084480 M=${MM})
    add_gemv_bf16_variant(${MM} 4096 2048 64 1)    # fc (and, at M = 1, the draft's o)
    add_gemv_bf16_variant(${MM} 2048 9216 64 1)    # q||k||v
    # PERF-84: the head's two pre-fc norms, one fold and one finish over both sources
    add_ocloc_kernel(prep_res_fold_M${MM}_K2048_SP0_G20_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=2048 S_PREV=0 FOLD_G=20 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_norm_finish_M${MM}_K2048_G20_W20_X4096_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=2048 NORM_G=20 NORM_WGS=20 X_STRIDE=4096 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_res_fold_M${MM}_K2048_SP1_G20_Z SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=2048 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_S1 SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS})
  endforeach()
  add_prep_res_fold(1 2048 1 20)   # the draft's post-attention fold over o's S = 1 output
  foreach(NV 32768 65536 131072)   # spec 8 §11's compact draft head at K 2048
    add_gemv_i8w_variant(1 2048 ${NV})
    add_gemv_bf16_variant(1 2048 ${NV} 64 1)
  endforeach()
endif()
# ==== Spec 15e ================================================================ (end) ====

# ==== PERF-56: Ornith's 16-, 32- and 64-row lookup verify lists (one card, bf16 KV, int8 head) ==== (begin) ====
# Additive only: new names; one source edit each (attn_verify_dpas.cl's GQA 8 / chunks of 8, gdn_ring_commit.cl's
# CONV_ROWS), whose existing command lines preprocess to the tokens they had. Qwen3.8's 16- / 32- / 64-row lists'
# binaries at Ornith's shapes: ids through Control::wide_cur / deep_cur (embed_gather only; the argmax pair is Qwen3.8's
# own M16 / M32 / M64 builds - the same vocabulary), the norms, the gated head, attn_prep (and the head's S1), the
# attention as attn_verify_dpas at 16 q / 2 kv in row chunks of 8 (every width: 16 rows of 8 heads would need all of
# the SLM for q) with attn_v2's reduce alone, moe.cl per row at W rows (a row is bitwise its M = 1 result:
# moe_m_test), the deferred GDN step at 16 rows and the 32- / 64-row lists' chain links, the ring copy at 8192
# channels, and the ROWG builds of the bf16 router (|| shared gate), a||b (a bf16 checkpoint's) and the head's fc and
# q||k||v. The int4 linears and the int8 head are spec 28b's block below (kernels::kSmallMTuning, kSmAll).
if(B70_ORNITH AND B70_MTP AND B70_VERIFY_M8)
  set(ORNITH_WIDE_TARGETS "")
  macro(ornith_wide_kernel NAME)
    add_ocloc_kernel(${NAME} ${ARGN})
    list(APPEND ORNITH_WIDE_TARGETS kernel_${NAME})
  endmacro()
  set(_ow_ctrl_16 ${WIDE_CTRL_DEFINES})
  set(_ow_ctrl_32 ${DEEP_CTRL_DEFINES})
  set(_ow_ctrl_64 ${DEEP_CTRL_DEFINES})
  foreach(MM 16 32 64)
    ornith_wide_kernel(embed_gather_M${MM}_D2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                       DEFINES ${_ow_ctrl_${MM}} M=${MM} HIDDEN=2048 VOCAB=248320)
    foreach(SP 0 4)
      ornith_wide_kernel(prep_res_fold_M${MM}_K2048_SP${SP}_G20 SOURCE ${PREP_CL}
                         DEFINES M=${MM} K=2048 S_PREV=${SP} FOLD_G=20)
    endforeach()
    ornith_wide_kernel(prep_norm_finish_M${MM}_K2048_G20_W20 SOURCE ${PREP_CL}
                       DEFINES M=${MM} K=2048 NORM_G=20 NORM_WGS=20)
    ornith_wide_kernel(prep_gated_head_M${MM}_GK16V32 SOURCE ${PREP_CL} DEFINES M=${MM} ${ORNITH_GDN_HEADS})
    ornith_wide_kernel(attn_prep_M${MM}_Q16KV2 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} ${ORNITH_FA_HEADS})
    # PROVISIONAL levers: the 16-row line's (PAIR at KS 1, DSPLIT 128, PVR 2) with PVX 16, so a work-group is 32
    # sub-groups as Qwen3.8's (8 row blocks x 2 + 16)
    ornith_wide_kernel(attn_verify_dpas_M${MM}_T${ATTN_V2_TGT}_Q16KV2
                       SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_verify_dpas.cl
                       DEFINES ${CTRL_DEFINES} M=8 RCHUNK=${MM} TGT=${ATTN_V2_TGT} KT=32 PACK_H=0 KS=1 DSPLIT=128
                               QSPLIT=1 PSPLIT=1 PAIR=1 PVX=16 PVR=2 KV8=0 ${ORNITH_FA_HEADS}
                       OPTIONS -cl-intel-256-GRF-per-thread)
    ornith_wide_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q16KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} REDUCE_ONLY=1
                               ${ORNITH_FA_HEADS})
    ornith_wide_kernel(moe_M${MM}_E256_T8_D2048_I512 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/moe.cl
                       DEFINES ${MOE_DEFINES} M=${MM} EXPERTS=256 TOP_K=8 HIDDEN=2048 INTER=512 ROUTER_N=272)
    # the MTP head's KV fill of W rows (its fc and q||k||v: the ROWG builds below)
    # PERF-84: the head's two pre-fc norms, one fold and one finish over both sources
    ornith_wide_kernel(prep_res_fold_M${MM}_K2048_SP0_G20_P2 SOURCE ${PREP_CL}
                       DEFINES M=${MM} K=2048 S_PREV=0 FOLD_G=20 PRE_FC_PAIR=1)
    ornith_wide_kernel(prep_norm_finish_M${MM}_K2048_G20_W20_X4096_P2 SOURCE ${PREP_CL}
                       DEFINES M=${MM} K=2048 NORM_G=20 NORM_WGS=20 X_STRIDE=4096 PRE_FC_PAIR=1)
    ornith_wide_kernel(prep_res_fold_M${MM}_K2048_SP1_G20_Z SOURCE ${PREP_CL}
                       DEFINES M=${MM} K=2048 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
    ornith_wide_kernel(attn_prep_M${MM}_Q16KV2_S1 SOURCE ${ATTN_CL}
                       DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS})
  endforeach()
  # The delta-net step: the 16-row deferred build, and the 32- / 64-row lists' links of it (ROW0); PEND_PAR is
  # PERF-51's (64 rows a parity).
  set(_ow_gdn ${CTRL_DEFINES} NEGA_OFF=32768 DTBIAS_OFF=32800 ${ORNITH_GDN_HEADS} DEFERRED=1 CTRL_PEND=30
              PEND_PAR=24084480 M=16)
  ornith_wide_kernel(gdn_step_defer_M16_G30_GK16V32 SOURCE ${GDN_CL} DEFINES ${_ow_gdn})
  foreach(R0 0 16 32 48)
    ornith_wide_kernel(gdn_step_defer_M16_G30_GK16V32_R${R0} SOURCE ${GDN_CL} DEFINES ${_ow_gdn} ROW0=${R0})
  endforeach()
  ornith_wide_kernel(gdn_ring_commit_C8192 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gdn_ring_commit.cl
                     DEFINES CTRL_POS=0 CTRL_RC=31 CONV_ROWS=8192)
  ornith_wide_kernel(gemv_bf16_M8_K2048_N272_C16_S16_RG SOURCE ${GEMV_BF16_CL}
                     DEFINES M=8 K=2048 N=272 COLS_PER_WG=16 KSPLIT=16 ROWG=1)
  ornith_wide_kernel(gemv_bf16_M8_K2048_N128_C16_S16_RG SOURCE ${GEMV_BF16_CL}
                     DEFINES M=8 K=2048 N=128 COLS_PER_WG=16 KSPLIT=16 ROWG=1)
  ornith_wide_kernel(gemv_bf16_M8_K4096_N2048_RG SOURCE ${GEMV_BF16_CL} DEFINES M=8 K=4096 N=2048 ROWG=1)
  ornith_wide_kernel(gemv_bf16_M8_K2048_N9216_RG SOURCE ${GEMV_BF16_CL} DEFINES M=8 K=2048 N=9216 ROWG=1)
  set_property(GLOBAL PROPERTY B70_ORNITH_WIDE_KERNELS ${ORNITH_WIDE_TARGETS})
endif()
# ==== PERF-56 ================================================================== (end) ====

# ==== Spec 13a / 13c: rows of different sequences (the batched engine) ==== (begin) ====
# Plans 13a (Task 2, the probe) and 13c (Task 3: promoted - built by default since 13c; until then behind B70_BATCH_PROBE). runtime::build_rows
# (capture.cc) binds them for runtime::Engine's sequence slots; tools/probe/probe_batch_step too.
#   gdn_step_rows_M<M>                  gdn_step.cl BATCH_ROWS: per row its position, slot (conv ring) and GDN state
#                                       indices (src at a slot's first row, dst after every row); the strides of one
#                                       whole-model state and one whole-model ring baked (48 GDN layers)
#   attn_prep_M<M>[_S1][_KV8]_ROWS      attn.cl / kv8.cl's writer: RoPE at the row's position, K / V at its slot's
#                                       KV base + position (_S1: the MTP head's bf16 q||k||v, QKV_S 1)
#   attn_v2_M<M>_T32[_KV8]_ROWS         attention v2 / its kv8 twin: groups = a slot's rows at one stride, each over
#                                       its own slot's keys
# at M = 1..8 (the 8-row budget, spec 13 §9). Every other
# binary a row list binds is B70_MTP's (M = 1..4) or B70_VERIFY_M8's (M = 5..8). The source edits are BATCH_ROWS
# blocks only: no other command line defines it, so every other binary's preprocessed source is unchanged.
# BROW_DEFINES is the row table's word layout (runtime::BatchRow, kernels::batch_row - the host mirror).
if(B70_MTP AND B70_VERIFY_M8 AND B70_KV8)
  set(BROW_DEFINES BATCH_ROWS=1 BROW_WORDS=8 BROW_POS=0 BROW_KVB=1 BROW_SRC=2 BROW_DST=3 BROW_SLOT=4)
  # One whole-model GDN state (gdn_layers x 48 x 128 x 128 floats) and conv ring (gdn_layers x 32 x 10240 ushorts).
  set(BR_GDN_QWEN SPEC_SLOT_STRIDE=37748736 RING_SLOT_STRIDE=15728640)
  foreach(MM 1 2 3 4 5 6 7 8)
    add_ocloc_kernel(gdn_step_rows_M${MM} SOURCE ${GDN_CL} DEFINES ${GDN_DEFINES} ${BROW_DEFINES} ${BR_GDN_QWEN} M=${MM})
    add_ocloc_kernel(attn_prep_M${MM}_ROWS SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM})
    add_ocloc_kernel(attn_prep_M${MM}_S1_ROWS SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1)
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_ROWS SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2})
    add_ocloc_kernel(attn_prep_M${MM}_KV8_ROWS SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=2)
    add_ocloc_kernel(attn_prep_M${MM}_S1_KV8_ROWS SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1)
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_KV8_ROWS SOURCE ${KV8_CL}
                     DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT})
  endforeach()
  # Plan 40a Task 2: the row lists' delta-net step in the DEFERRED form (gdn_step.cl, BATCH_ROWS with DEFERRED) - each
  # run replays its sequence's committed rows and writes no per-row state. BROW_PEND: the row word with the sequence's
  # pending count and parity; PEND_SEQ: one sequence's pending rows, two parities (EngineSizes::gdn_pend / sequences).
  foreach(MM 2 3 4 5 6 7 8)
    add_ocloc_kernel(gdn_step_rows_defer_M${MM} SOURCE ${GDN_CL}
                     DEFINES ${GDN_DEFINES} ${BROW_DEFINES} ${BR_GDN_QWEN} BROW_PEND=5 DEFERRED=1 CTRL_PEND=30
                             PEND_PAR=51511296 PEND_SEQ=103022592 M=${MM})
  endforeach()
  # Plan 40a Task 3: the 16- and 32-row lists' delta-net step - the M 8 build over chunks of whole runs (WIDE_ROWS:
  # the list's rows at most; grid dimension 2 the chunk, a run a work-group).
  add_ocloc_kernel(gdn_step_rows_defer_M8_W32 SOURCE ${GDN_CL}
                   DEFINES ${GDN_DEFINES} ${BROW_DEFINES} ${BR_GDN_QWEN} BROW_PEND=5 DEFERRED=1 CTRL_PEND=30
                           PEND_PAR=51511296 PEND_SEQ=103022592 M=8 WIDE_ROWS=32)
  # Spec 13c: the MTP head's draft at M = 2..4 rows of different slots (runtime::build_draft_rows - every other binary
  # it binds is the verify lists' or the rows' above): the head's o, gate||up and down at M rows (bf16, the {64, 1}
  # tiling of their M = 1 twins) and the post-attention fold over o's S = 1 output.
  foreach(MM 2 3 4)
    add_gemv_bf16_variant(${MM} 6144 5120 64 1)
    add_gemv_bf16_variant(${MM} 5120 34816 64 1)
    add_gemv_bf16_variant(${MM} 17408 5120 64 1)
    add_prep_res_fold(${MM} 5120 1 20)
  endforeach()
endif()
# ==== Spec 13a / 13c ==== (end) ====

# ==== Plan 19v3b Task 0: the slots' KV in pages ==== (begin) ====
# (Task 0). An engine with sequence slots holds every
# slot's full-attention KV in pages of KV_PAGE positions behind a page table (runtime/buffer_sizes.h, SlotDims), and its
# lists - every slot's, slot 0's too, and the row lists - bind these paged twins of the binaries that read or write the
# KV: the name `_PG<page>` after the unpaged one's (kernels::kv_paged), -DKV_PAGED=<page>, and the table as a last
# argument. Additive only: the KV_PAGED blocks are undefined on every other command line, so every other binary's
# preprocessed source is unchanged; a one-slot engine binds none of these. The prefill twins are in
# prefill/CMakeLists.txt.
#   attn_prep_M<M>[_S1][_KV8]_PG<page>          the per-slot lists' writers (M = 1..4: plain, verify, the head's)
#   attn_v2_M<M>_T32[_KV8]_PG<page>             their decode attention (and the head's draft attention)
#   attn_prep_M<M>[_S1][_KV8]_ROWS_PG<page>     the row lists' (M = 2..8)
#   attn_v2_M<M>_T32[_KV8]_ROWS_PG<page>
#   attn_verify_dpas_M<M>_T32[_KV8]_PG<page>    the verify lists' DPAS attention (M = 2..4, VDPAS_LEVERS_PAIR)
if(B70_MTP AND B70_VERIFY_M8 AND B70_KV8)
  set(PG _PG${KV_PAGE})
  # M = 1..4; plan 19v3b Tasks 3-7: slot 0 beside the DFlash drafter verifies at M = 5..8 too (its tap verify lists).
  foreach(MM 1 2 3 4 5 6 7 8)
    add_ocloc_kernel(attn_prep_M${MM}${PG} SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_S1${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1 KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}${PG} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_KV8${PG} SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=2 KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_S1_KV8${PG} SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=1 KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_KV8${PG} SOURCE ${KV8_CL}
                     DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT} KV_PAGED=${KV_PAGE})
  endforeach()
  foreach(MM 2 3 4 5 6 7 8)
    add_ocloc_kernel(attn_prep_M${MM}_ROWS${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_S1_ROWS${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1 KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_ROWS${PG} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2}
                             KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_KV8_ROWS${PG} SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=2 KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_S1_KV8_ROWS${PG} SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1 KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_KV8_ROWS${PG} SOURCE ${KV8_CL}
                     DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT} KV_PAGED=${KV_PAGE})
  endforeach()
  # Plan 40a Task 3: the 16- and 32-row lists' KV writers, main and head (their attention runs the 8-row builds above
  # on groups of 8 rows; every other per-row build is the one-sequence 16- / 32-row lists')
  foreach(MM 16 32)
    add_ocloc_kernel(attn_prep_M${MM}_ROWS${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_S1_ROWS${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1 KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_KV8_ROWS${PG} SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=2 KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_S1_KV8_ROWS${PG} SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1 KV_PAGED=${KV_PAGE})
  endforeach()
  # Plan 40a Task 3: their attention - attn_verify_dpas at the one-sequence M 8 lines' levers (bf16 VDPAS_LEVERS, int8
  # SOLO), a run of up to 8 rows a work-group (BATCH_ROWS, grid dimension 2), the partials of attn_v2's 8-row groups
  # (ROWS_W: the widest list) for their reduce (attn_v2_M8_T32[_KV8]_ROWS_PG<page> above)
  if(B70_VERIFY_DPAS)
    foreach(KV8 0 1)
      if(KV8)
        set(_kv _KV8)
        set(_rw_lv ${VDPAS_LEVERS_SOLO} PVX=${_vdpas_pvx_8})
      else()
        set(_kv "")
        set(_rw_lv ${VDPAS_LEVERS})
      endif()
      # PERF-120 (plan 40a Task 7): at 4 key blocks a run (runtime::Engine::kRowsAttnBlocks: 16 to 18 % off the 32-row
      # step against 32 blocks at depth 4k / 12k / 30k; 2 blocks measured level, 64 / 128 slower), their reduce the v2
      # row build at TGT 4 below
      add_ocloc_kernel(attn_verify_dpas_M8_T4${_kv}_ROWS${PG} SOURCE ${VDPAS_CL}
                       DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} M=8 TGT=4 ${_rw_lv} KV8=${KV8}
                               KV_PAGED=${KV_PAGE} ROWS_W=32
                       OPTIONS ${VDPAS_OPTIONS})
    endforeach()
    add_ocloc_kernel(attn_v2_M8_T4_ROWS${PG} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} M=8 TGT=4 EXP2=${ATTN_V2_EXP2} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_v2_M8_T4_KV8_ROWS${PG} SOURCE ${KV8_CL}
                     DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} KV8_DEC=1 M=8 TGT=4 KV_PAGED=${KV_PAGE})
  endif()
endif()
# ==== Plan 19v3b Task 0 ==== (end; the DPAS twins follow spec 28c's block) ====

# ==== Plan 31 Task 12: Ornith's sequence slots on one card ==== (begin) ====
# Additive only: new names at Ornith's shapes (model::ornith(): hidden 2048, GDN 16 k / 32 v heads over 30 layers, FA
# 16 q / 2 kv heads, the MoE block), each the command line of its Qwen3.8 twin above plus Ornith's shape defines; no
# source is copied, and a one-slot Ornith engine binds none of these.
#   gdn_step_rows_M<M>_G30_GK16V32               the row lists' delta-net step (M = 2..8): BR_GDN_ORNITH's strides are
#                                                one whole-model state (30 x 32 x 128 x 128 floats) and ring (30 x 32 x
#                                                8192 ushorts)
#   attn_prep_M<M>_Q16KV2[_S1]_PG2048            the per-slot lists' KV writers (M = 1..4: plain, verify, the head's)
#   attn_v2_M<M>_T32_Q16KV2_PG2048               their decode attention (and the head's draft attention)
#   attn_prep_M<M>_Q16KV2[_S1]_ROWS_PG2048       the row lists' (M = 2..8)
#   attn_v2_M<M>_T32_Q16KV2_ROWS_PG2048
#   the decode list at M = 5..8 (spec 15e's block builds 2..4): the norm pairs, embed_gather, a||b and the router in bf16,
#   prep_gated_head, the MoE block, the bf16 lm_head, and the MTP head's KV fill (fc, q||k||v, the pre-fc norm pair into
#   one row, the zero-residual fold); the int4 linears and the int8 head are gemm_sm's rows (spec 28b's block, M = 2..8)
#   prep_res_fold_M<M>_K2048_SP1_G20            the row draft lists' post-attention fold (M = 2..4)
# Ornith's verify attention is v2 at every M (runtime/verify_attn.h), so no DPAS twin is built; the int8 KV cache's
# twins are PERF-64's block below.
if(B70_ORNITH AND B70_MTP AND B70_VERIFY_M8 AND B70_KV8)
  set(BR_GDN_ORNITH SPEC_SLOT_STRIDE=15728640 RING_SLOT_STRIDE=7864320)
  foreach(MM 1 2 3 4)
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} M=${MM} ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_S1${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q16KV2${PG} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} ${ORNITH_FA_HEADS}
                             KV_PAGED=${KV_PAGE})
  endforeach()
  foreach(MM 2 3 4 5 6 7 8)
    add_ocloc_kernel(gdn_step_rows_M${MM}_G30_GK16V32 SOURCE ${GDN_CL}
                     DEFINES ${CTRL_DEFINES} NEGA_OFF=32768 DTBIAS_OFF=32800 ${ORNITH_GDN_HEADS} ${BROW_DEFINES}
                             ${BR_GDN_ORNITH} M=${MM})
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_ROWS${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_S1_ROWS${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
    # Past 4 rows a lane's running max, sum and output of NPR = 8 x M (q head, row) pairs (attn_v2.cl) outgrow 128
    # GRF: 256 GRF there (who holds a value, not the value): 6 rows 2.62 -> 2.30 ms a step at depth 4096 (profiled).
    # 8 rows run as two 4-row launches (runtime/capture.cc, attn_pair_rows: its SLM).
    set(_orn_rows_grf "")
    if(MM GREATER 4)
      set(_orn_rows_grf OPTIONS -cl-intel-256-GRF-per-thread)
    endif()
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q16KV2_ROWS${PG} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2}
                             ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE} ${_orn_rows_grf})
  endforeach()
  foreach(MM 5 6 7 8)
    add_gemv_bf16_variant(${MM} 2048 128 16 16)     # a||b (a checkpoint that keeps it bf16)
    add_gemv_bf16_variant(${MM} 2048 272 16 16)     # the router || shared gate
    add_gemv_bf16_variant(${MM} 2048 248320 64 1)   # lm_head (bf16)
    add_prep_res_fold(${MM} 2048 0 20)
    add_prep_res_fold(${MM} 2048 4 20)
    add_prep_norm_finish(${MM} 2048 20 20)
    add_ocloc_kernel(embed_gather_M${MM}_D2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} HIDDEN=2048 VOCAB=248320)
    add_ocloc_kernel(prep_gated_head_M${MM}_GK16V32 SOURCE ${PREP_CL} DEFINES M=${MM} ${ORNITH_GDN_HEADS})
    add_ocloc_kernel(moe_M${MM}_E256_T8_D2048_I512 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/moe.cl
                     DEFINES ${MOE_DEFINES} M=${MM} EXPERTS=256 TOP_K=8 HIDDEN=2048 INTER=512 ROUTER_N=272)
    # the MTP head's KV fill of M rows
    add_gemv_bf16_variant(${MM} 4096 2048 64 1)     # fc
    add_gemv_bf16_variant(${MM} 2048 9216 64 1)     # q||k||v
    # PERF-84: the head's two pre-fc norms, one fold and one finish over both sources
    add_ocloc_kernel(prep_res_fold_M${MM}_K2048_SP0_G20_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=2048 S_PREV=0 FOLD_G=20 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_norm_finish_M${MM}_K2048_G20_W20_X4096_P2 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=2048 NORM_G=20 NORM_WGS=20 X_STRIDE=4096 PRE_FC_PAIR=1)
    add_ocloc_kernel(prep_res_fold_M${MM}_K2048_SP1_G20_Z SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=2048 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
  endforeach()
  foreach(MM 2 3 4)
    add_prep_res_fold(${MM} 2048 1 20)
  endforeach()
endif()
# ==== Plan 31 Task 12 ==== (end) ====

# ==== PERF-64: Ornith's int8 KV cache (`--kv-cache int8`) ==== (begin) ====
# Additive only: kv8.cl's writer and decode reader at Ornith's 16 q / 2 kv heads, each the command line of its Qwen3.8
# twin (spec 12b's block, the sequence slots' blocks above) plus ${ORNITH_FA_HEADS}; no source is copied. What an
# Ornith list binds over an int8 cache, as its bf16 twins above:
#   attn_prep_M<M>_Q16KV2[_S1]_KV8               one sequence's writers (M = 1..4: plain, MTP verify, the head's)
#   attn_v2_M<M>_T32_Q16KV2_KV8                  their decode attention (and the head's draft attention)
#   ..._PG2048, attn_prep_M<M>_Q16KV2[_S1]_KV8_ROWS_PG2048, attn_v2_M<M>_T32_Q16KV2_KV8_ROWS_PG2048
#                                                the sequence slots' / the pool's (per slot M = 1..4, rows M = 2..8;
#                                                past 4 rows 256 GRF, as the bf16 rows)
# The wide lookup verify lists (16 / 32 / 64 rows) stay bf16-only, as Qwen3.8's (runtime::DecodeScratchDims::rows).
if(B70_ORNITH AND B70_KV8)
  foreach(MM 1 2 3 4)
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_KV8 SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=2 ${ORNITH_FA_HEADS})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q16KV2_KV8 SOURCE ${KV8_CL}
                     DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT} ${ORNITH_FA_HEADS})
    if(B70_MTP)
      add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_S1_KV8 SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS})
    endif()
  endforeach()
  if(B70_MTP AND B70_VERIFY_M8)
    set(_opg _PG${KV_PAGE})
    foreach(MM 1 2 3 4)
      add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_KV8${_opg} SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=2 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
      add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_S1_KV8${_opg} SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
      add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q16KV2_KV8${_opg} SOURCE ${KV8_CL}
                       DEFINES ${CTRL_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT} ${ORNITH_FA_HEADS}
                               KV_PAGED=${KV_PAGE})
    endforeach()
    foreach(MM 2 3 4 5 6 7 8)
      set(_orn_grf "")
      if(MM GREATER 4)
        set(_orn_grf OPTIONS -cl-intel-256-GRF-per-thread)
      endif()
      add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_KV8_ROWS${_opg} SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=2 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
      add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_S1_KV8_ROWS${_opg} SOURCE ${KV8_CL}
                       DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
      add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT}_Q16KV2_KV8_ROWS${_opg} SOURCE ${KV8_CL}
                       DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} KV8_DEC=1 M=${MM} TGT=${ATTN_V2_TGT} ${ORNITH_FA_HEADS}
                               KV_PAGED=${KV_PAGE} ${_orn_grf})
    endforeach()
  endif()
endif()
# ==== PERF-64 ==== (end) ====

# ==== PERF-120: Ornith's row lists of 16 and 32 rows across sequences (plan 40a's builds at Ornith's shapes) ====
# Additive only: each the command line of its Qwen3.8 twin (plan 40a's lines in the row blocks above) plus Ornith's
# shape defines; no source is copied. What an Ornith slots engine with the MTP head binds:
#   gdn_step_rows_defer_M8_W32_G30_GK16V32       the 16- and 32-row lists' delta-net step in the DEFERRED form (WIDE_ROWS:
#                                                a run a work-group): each run replays its sequence's pending rows
#                                                (PEND_SEQ: two parities of PERF-51's PEND_PAR); BR_GDN_ORNITH's strides.
#                                                The 2..8-row lists keep the slot build (runtime::Engine: 4 % faster)
#   attn_prep_M<M>_Q16KV2[_S1][_KV8]_ROWS_PG2048 their KV writers, main and head (M = 16, 32)
#   attn_verify_dpas_M8_T4_Q16KV2[_KV8]_ROWS_PG2048
#                                                their attention, one launch a step (BATCH_ROWS, a run a work-group) at
#                                                4 key blocks a run; GQA 8 needs its own levers - bf16 PERF-56's Ornith
#                                                line (PAIR at KS 1), int8 SOLO at KS 2 (PAIR at KS 1 is bf16-only)
#   attn_v2_M8_T4_Q16KV2[_KV8]_ROWS_PG2048       their reduce, a group of 8 rows (256 GRF as the 8-row row builds)
# Ornith's lone verify is v2 (runtime/verify_attn.h), so a run is not bitwise its lone verify: the f32 judge.
if(B70_ORNITH AND B70_MTP AND B70_VERIFY_M8 AND B70_KV8 AND B70_VERIFY_DPAS)
  set(_orn_rdefer ${CTRL_DEFINES} NEGA_OFF=32768 DTBIAS_OFF=32800 ${ORNITH_GDN_HEADS} ${BROW_DEFINES} ${BR_GDN_ORNITH}
                  BROW_PEND=5 DEFERRED=1 CTRL_PEND=30 PEND_PAR=24084480 PEND_SEQ=48168960)
  add_ocloc_kernel(gdn_step_rows_defer_M8_W32_G30_GK16V32 SOURCE ${GDN_CL} DEFINES ${_orn_rdefer} M=8 WIDE_ROWS=32)
  foreach(MM 16 32)
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_ROWS${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_S1_ROWS${PG} SOURCE ${ATTN_CL}
                     DEFINES ${ATTN_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_KV8_ROWS${PG} SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=2 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
    add_ocloc_kernel(attn_prep_M${MM}_Q16KV2_S1_KV8_ROWS${PG} SOURCE ${KV8_CL}
                     DEFINES ${KV8_PREP_DEFINES} ${BROW_DEFINES} M=${MM} QKV_S=1 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE})
  endforeach()
  add_ocloc_kernel(attn_verify_dpas_M8_T4_Q16KV2_ROWS${PG} SOURCE ${VDPAS_CL}
                   DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} M=8 TGT=4 KT=32 PACK_H=0 KS=1 DSPLIT=128 QSPLIT=1 PSPLIT=1
                           PAIR=1 PVX=16 PVR=2 KV8=0 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE} ROWS_W=32
                   OPTIONS ${VDPAS_OPTIONS})
  add_ocloc_kernel(attn_verify_dpas_M8_T4_Q16KV2_KV8_ROWS${PG} SOURCE ${VDPAS_CL}
                   DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} M=8 TGT=4 KT=32 PACK_H=0 KS=2 DSPLIT=256 QSPLIT=1 PSPLIT=1
                           PAIR=1 SRB=2 PVX=8 KV8=1 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE} ROWS_W=32
                   OPTIONS ${VDPAS_OPTIONS})
  add_ocloc_kernel(attn_v2_M8_T4_Q16KV2_ROWS${PG} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                   DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} M=8 TGT=4 EXP2=${ATTN_V2_EXP2} ${ORNITH_FA_HEADS}
                           KV_PAGED=${KV_PAGE} OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(attn_v2_M8_T4_Q16KV2_KV8_ROWS${PG} SOURCE ${KV8_CL}
                   DEFINES ${CTRL_DEFINES} ${BROW_DEFINES} KV8_DEC=1 M=8 TGT=4 ${ORNITH_FA_HEADS} KV_PAGED=${KV_PAGE}
                   OPTIONS -cl-intel-256-GRF-per-thread)
  # their MoE block reading each routed expert once for the rows routed to it (moe.cl MOE_UNION; every pair bitwise
  # the per-row kernels' - moe_m_test)
  foreach(MM 16 32)
    add_ocloc_kernel(moe_M${MM}_E256_T8_D2048_I512_U SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/moe.cl
                     DEFINES ${MOE_DEFINES} ${CTRL_DEFINES} M=${MM} EXPERTS=256 TOP_K=8 HIDDEN=2048 INTER=512
                             ROUTER_N=272 MOE_UNION=1)
  endforeach()
endif()
# ==== PERF-120 (Ornith) ==== (end) ====

# ==== Spec 28b: the small-M GEMM (gemm_sm.cl) ==== (begin) ====
# Additive only: no existing variant's name or
# command line moves (tools/kernel_cmdlines: +N / -0 / ~0), every M = 1 list binds what it bound before. Every list
# a capture builds at M >= 2 binds these in place of gemv / gemv_i8w (runtime::gemv_kind, runtime/gemv_kind.h) for
# the shapes of kernels::kSmallMTuning (src/kernels/kernels.h) - ONE row per shape, whose knobs are spelled into the
# name, so a row and a line here that disagree are a missing file at capture (gemm_sm_names_test holds the two
# together on any host). Rows: M = 2..8 (the MTP / lookup verify lists 2..4, DFlash's tap verify 2..8, spec 13a's rows)
# and 16 (gemm_sm_test, the probe; no list binds 16 yet); Ornith's at 2..4, 16, 32 and 64 (PERF-56) and Flash-Next's
# at 2..4; the DFlash drafter's at
# its block's 8 / draft rows' 7 (spec 28d).
set(GEMM_SM_TARGETS "")
# add_gemm_sm_variant(M K N S L [WG] [MG] [G256] [B2D] [GATE 1|2]) - L 2 is the int8 form (gemm_sm_i8w_*), S 1 there;
# GATE (PERF-123, head_gate.h) the build with the head choice's extra argument. The
# source by its full path: tools/probe calls these helpers from its own directory scope.
function(add_gemm_sm_variant M K N S L)
  cmake_parse_arguments(G "WG;MG;G256;B2D;TF" "GATE" "" ${ARGN})
  set(defs M=${M} K=${K} N=${N} S=${S} LAYOUT=${L})
  set(tag "")
  set(opts "")
  if(G_WG)
    list(APPEND defs SPLITK=1)
    string(APPEND tag _wg)
  endif()
  if(G_MG)
    list(APPEND defs DEQ=1)
    string(APPEND tag _mg)
  endif()
  if(G_G256)
    set(opts OPTIONS -cl-intel-256-GRF-per-thread)
    string(APPEND tag _g256)
  endif()
  if(G_B2D)
    list(APPEND defs BLK2D=1)
    string(APPEND tag _b2d)
  endif()
  if(G_GATE)
    list(APPEND defs GATE=${G_GATE})
    string(APPEND tag _gate${G_GATE})
  endif()
  # PERF-44 on Qwen3.8 tp: the folded crossing's tp_send as the tail (TP_FOLD_SEND); not a kSmallMTuning row, so kept
  # out of GEMM_SM_TARGETS (gemm_sm_names_test) in GEMM_SM_TF_TARGETS
  if(G_TF)
    list(APPEND defs TP_FOLD_SEND=1 TP_FOLD_G=20)
    string(APPEND tag _tf)
  endif()
  if(L EQUAL 2)
    set(name gemm_sm_i8w_M${M}_K${K}_N${N}${tag})
  else()
    set(name gemm_sm_M${M}_K${K}_N${N}_S${S}_L${L}${tag})
  endif()
  if(NOT TARGET kernel_${name})
    add_ocloc_kernel(${name} SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/gemm_sm.cl DEFINES ${defs} ${opts})
    if(G_TF)
      set(GEMM_SM_TF_TARGETS ${GEMM_SM_TF_TARGETS} kernel_${name} PARENT_SCOPE)
    else()
      set(GEMM_SM_TARGETS ${GEMM_SM_TARGETS} kernel_${name} PARENT_SCOPE)
    endif()
  endif()
endfunction()
# Spec 32e: add_gemm_sm_wide_variant(M K N S L ARM [WG] [MG] [B2D]) - gemm_sm_wide.cl past 16 rows; ARM rc32 (row chunks
# of 32, prefetch 1, 256 GRF), sw (one sub-group for every row, prefetch 2, 128 GRF), rc32u (rc32, the K walk unrolled by
# 2, the scale row by a block read) or rc32h (rc32u at 128 GRF, half the A operands at a time; item 41):
# kernels::gemm_sm_variant's name.
function(add_gemm_sm_wide_variant M K N S L ARM)
  cmake_parse_arguments(G "WG;MG;B2D" "" "" ${ARGN})
  set(defs M=${M} K=${K} N=${N} S=${S} LAYOUT=${L})
  set(tag "")
  if(G_WG)
    list(APPEND defs SPLITK=1)
    string(APPEND tag _wg)
  endif()
  if(G_MG)
    list(APPEND defs DEQ=1)
    string(APPEND tag _mg)
  endif()
  if(G_B2D)
    list(APPEND defs BLK2D=1)
    string(APPEND tag _b2d)
  endif()
  set(opts "")
  if(ARM STREQUAL "rc32")
    list(APPEND defs RC=32 PF_G=1)
    set(opts OPTIONS -cl-intel-256-GRF-per-thread)
    string(APPEND tag _rc32_pf1_g256)
  elseif(ARM STREQUAL "rc32u")
    list(APPEND defs RC=32 PF_G=1 UNR=2 SCB=1)
    set(opts OPTIONS -cl-intel-256-GRF-per-thread)
    string(APPEND tag _rc32_u2_sb_pf1_g256)
  elseif(ARM STREQUAL "rc32h")
    list(APPEND defs RC=32 PF_G=1 UNR=2 SCB=1 AHALF=1)
    string(APPEND tag _rc32_ah_u2_sb_pf1)
  else()
    list(APPEND defs SWAP=1 PF_G=2)
    string(APPEND tag _sw_pf2)
  endif()
  set(name gemm_sm_wide_M${M}_K${K}_N${N}_S${S}_L${L}${tag})
  if(NOT TARGET kernel_${name})
    add_ocloc_kernel(${name} SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/gemm_sm_wide.cl DEFINES ${defs} ${opts})
    set(GEMM_SM_TARGETS ${GEMM_SM_TARGETS} kernel_${name} PARENT_SCOPE)
  endif()
endfunction()
# The production rows, kernels::kSmallMTuning's order: "K N S L flags...".
# (S is the OUTPUT's slices: gate||up's 2x8 work-group row compiles at S 1 - kernels::gemm_sm_out_s.)
set(GEMM_SM_QWEN38 "5120 16384 1 1 MG" "6144 5120 4 0 MG B2D" "5120 14336 2 0 MG B2D" "5120 34816 1 0 WG MG B2D"
                   "17408 5120 4 0 MG B2D" "5120 248320 1 2 WG")
macro(gemm_sm_rows ROWS MLIST)
  foreach(row IN LISTS ${ROWS})
    separate_arguments(f UNIX_COMMAND "${row}")
    foreach(MM IN LISTS ${MLIST})
      add_gemm_sm_variant(${MM} ${f})
    endforeach()
  endforeach()
endmacro()
set(GEMM_SM_M 2 3 4 5 6 7 8 16)
set(GEMM_SM_CONSUMERS "")
if(B70_MTP)
  gemm_sm_rows(GEMM_SM_QWEN38 GEMM_SM_M)
  # gate||up's SiLU after the one-slice 2x8 work-group (kernels::gemm_sm_silu_variant): SILU_S 1 at the lists' M
  foreach(MM 2 3 4 5 6 7 8 16)
    add_ocloc_kernel(prep_silu_mul_M${MM}_S1 SOURCE ${PREP_CL} DEFINES M=${MM} SILU_S=1)
    list(APPEND GEMM_SM_CONSUMERS kernel_prep_silu_mul_M${MM}_S1)
  endforeach()
  # Spec 32 / 32e: the deep lists' 32 and 64 rows. The int4 rows by gemm_sm_wide (gemm_sm spills there), the shape
  # kernels::SmallMTuning's wide_32 / wide_64 names (GEMM_SM_WIDE_32 / _64: "K N S L arm flags..."); the int8 head by
  # gemm_sm_i8w at 256 GRF (its g256_from 32); gate||up's SiLU at S 1.
  if(B70_VERIFY_M8)
    set(GEMM_SM_WIDE_32 "5120 16384 1 1 rc32 MG" "6144 5120 4 0 sw MG B2D" "5120 14336 2 0 rc32 MG B2D"
                      "5120 34816 1 0 rc32 WG MG B2D" "17408 5120 4 0 rc32 MG B2D")
    set(GEMM_SM_WIDE_64 "5120 16384 1 1 rc32h MG" "6144 5120 4 0 rc32u MG B2D" "5120 14336 2 0 rc32h MG B2D"
                      "5120 34816 1 0 rc32h WG MG B2D" "17408 5120 4 0 rc32u MG B2D")
    foreach(MM 32 64)
      foreach(row IN LISTS GEMM_SM_WIDE_${MM})
        separate_arguments(f UNIX_COMMAND "${row}")
        add_gemm_sm_wide_variant(${MM} ${f})
      endforeach()
      add_gemm_sm_variant(${MM} 5120 248320 1 2 WG G256)
      add_ocloc_kernel(prep_silu_mul_M${MM}_S1 SOURCE ${PREP_CL} DEFINES M=${MM} SILU_S=1)
      list(APPEND GEMM_SM_CONSUMERS kernel_prep_silu_mul_M${MM}_S1)
    endforeach()
  endif()
endif()
# PERF-123: Qwen3.8's one-card verify lists at M = 2..8 choose their head a step (head_gate.h): the full int8 head
# (GATE 1) or the 2-bit candidate pass (gemm_sm LAYOUT 3), df_candidate's top 64 a row, head_cand.cl's gather of those
# columns, the exact rescore (gemm_sm_i8w at N 512 with the full head's knobs) and its argmax (GATE 2).
if(B70_MTP AND B70_VERIFY_M8 AND B70_DFLASH)
  foreach(MM 2 3 4 5 6 7 8)
    add_gemm_sm_variant(${MM} 5120 248320 1 2 WG GATE 1)
    add_gemm_sm_variant(${MM} 5120 248320 1 3 GATE 2)
    add_gemm_sm_variant(${MM} 5120 512 1 2 WG GATE 2)
    add_gemm_sm_variant(${MM} 5120 512 1 2 WG)   # the table row's own build, which gemm_sm_test runs (no gate argument)
  endforeach()
  add_ocloc_kernel(df_candidate_R7_N248320_U248077_K64_gate2 SOURCE ${DF_DIR}/df_candidate.cl
                   DEFINES R=7 N=248320 USED=248077 GATE=2)
  add_ocloc_kernel(head_cand_K5120_C64_NC512 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/head_cand.cl
                   DEFINES K=5120 C=64 NC=512 GATE=2)
endif()
# PERF-123 on Ornith: the same head choice at its hidden 2048 - its int8 head's own knobs (no wg), the 2-bit pass, the
# rescore at N 512 with the full head's knobs; df_candidate's gate-2 build above is shared (the same vocabulary).
if(B70_ORNITH AND B70_MTP AND B70_VERIFY_M8 AND B70_DFLASH)
  foreach(MM 2 3 4 5 6 7 8)
    add_gemm_sm_variant(${MM} 2048 248320 1 2 GATE 1)
    add_gemm_sm_variant(${MM} 2048 248320 1 3 GATE 2)
    add_gemm_sm_variant(${MM} 2048 512 1 2 GATE 2)
    add_gemm_sm_variant(${MM} 2048 248320 1 3)   # the table rows' own builds, which gemm_sm_test runs
    add_gemm_sm_variant(${MM} 2048 512 1 2)
  endforeach()
  add_ocloc_kernel(head_cand_K2048_C64_NC512 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/head_cand.cl
                   DEFINES K=2048 C=64 NC=512 GATE=2)
endif()
# Ornith's dense projections at its verify lists' M = 2..4 (spec 15e's block builds the verify lists' gemv)
set(GEMM_SM_ORNITH "2048 12288 1 1 MG" "4096 2048 4 0 MG" "2048 9216 2 0 MG" "2048 128 1 1 MG" "2048 248320 1 2")
if(B70_ORNITH AND B70_MTP)
  # Plan 31 Task 12: 2..8 (the sequence slots' row lists); PERF-56: and the wide lookup verify lists' 16 / 32 / 64
  # rows - gemm_sm at 16, the int4 rows by gemm_sm_wide past it (the arms of Qwen3.8's row of the same form, "K N S L
  # arm flags..."), the head by gemm_sm_i8w at 256 GRF (g256_from 32); kSmAll
  set(_m_ornith 2 3 4)
  if(B70_VERIFY_M8)
    list(APPEND _m_ornith 5 6 7 8 16)
  endif()
  gemm_sm_rows(GEMM_SM_ORNITH _m_ornith)
  if(B70_VERIFY_M8)
    set(GEMM_SM_ORNITH_WIDE_32 "2048 12288 1 1 rc32 MG" "4096 2048 4 0 sw MG" "2048 9216 2 0 rc32 MG"
                               "2048 128 1 1 rc32 MG")
    set(GEMM_SM_ORNITH_WIDE_64 "2048 12288 1 1 rc32h MG" "4096 2048 4 0 rc32u MG" "2048 9216 2 0 rc32h MG"
                               "2048 128 1 1 rc32h MG")
    foreach(MM 32 64)
      foreach(row IN LISTS GEMM_SM_ORNITH_WIDE_${MM})
        separate_arguments(f UNIX_COMMAND "${row}")
        add_gemm_sm_wide_variant(${MM} ${f})
      endforeach()
      add_gemm_sm_variant(${MM} 2048 248320 1 2 G256)
    endforeach()
  endif()
endif()
# Spec 28d: the DFlash drafter's int8 linears at its block's M = 8 and its compact draft heads at the 7 draft rows
# (kSmDraft8 / kSmDraft7; the full int8 head is Qwen3.8's row, built at 7 above) - spec 19b's lists bind them.
set(GEMM_SM_DFLASH "5120 6144 1 2 WG" "4096 5120 1 2 WG" "5120 34816 1 2 WG" "17408 5120 1 2 WG" "25600 5120 1 2 WG"
                   "5120 2048 1 2 WG" "5120 1280 1 2 WG" "5120 256 1 2 WG")
set(GEMM_SM_DFLASH_HEAD "5120 32768 1 2 WG" "5120 65536 1 2 WG" "5120 131072 1 2 WG")
set(GEMM_SM_DFLASH_INT4 "5120 6144 1 1 MG" "4096 5120 1 1 MG" "5120 34816 1 1 MG" "17408 5120 1 1 MG"
                      "25600 5120 1 1 MG" "5120 2048 1 1 MG" "5120 1280 1 1 MG" "5120 256 1 1 MG"
                      "5120 6144 4 1 MG" "4096 5120 4 1 MG" "5120 34816 8 1 MG" "17408 5120 4 1 MG" "25600 5120 8 1 MG" "5120 2048 8 1 MG" "5120 1280 16 1 MG" "5120 256 16 1 MG")
if(B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  set(_m8 8)
  set(_m7 7)
  gemm_sm_rows(GEMM_SM_DFLASH _m8)
  gemm_sm_rows(GEMM_SM_DFLASH_INT4 _m8)
  gemm_sm_rows(GEMM_SM_DFLASH_HEAD _m7)
  set(GEMM_SM_DFLASH_CANDIDATE "5120 248320 1 1 MG" "5120 248320 1 3")   # 3: spec 19 v3d's 2-bit candidates
  gemm_sm_rows(GEMM_SM_DFLASH_CANDIDATE _m7)
  # Plan 40a Task 4: the batched pass at four blocks - the grid split-K int4 rows at 32 by gemm_sm_wide's row chunk
  # (kSmDraft8W), the 2-bit candidate head at its 31 rows at 256 GRF (kSmDraft7W, g256_from 31).
  foreach(row "5120 6144 4" "4096 5120 4" "5120 34816 8" "17408 5120 4" "25600 5120 8" "5120 2048 8" "5120 1280 16"
              "5120 256 16")
    separate_arguments(f UNIX_COMMAND "${row}")
    add_gemm_sm_wide_variant(32 ${f} 1 rc32 MG)
  endforeach()
  add_gemm_sm_variant(31 5120 248320 1 3 G256)
  add_gemv_variant(7 5120 248320 1 1)
  foreach(row "5120 6144 4" "4096 5120 4" "5120 34816 8" "17408 5120 4" "25600 5120 8"
              "5120 2048 8" "5120 1280 16" "5120 256 16")
    separate_arguments(parts UNIX_COMMAND "${row}")
    list(GET parts 0 K)
    list(GET parts 1 N)
    list(GET parts 2 S)
    add_gemv_variant(8 ${K} ${N} ${S} 1)
  endforeach()
  foreach(KN "5120 6144" "4096 5120" "5120 34816" "17408 5120" "25600 5120" "5120 2048" "5120 1280" "5120 256")
    separate_arguments(KN)
    add_gemv_variant(8 ${KN} 1 1)   # int4 g64 drafter, the forced-GEMV reference
  endforeach()
endif()
# Spec 28d: Qwen3.8-Flash-Next's dense int4 projections and int8 head at its verify lists' M = 2..4 (spec 21-QFN-v1e's block
# builds the lists' gemv; its own capture binds these at M >= 2).
set(GEMM_SM_Q4EXP "2560 16384 1 0 MG" "2560 13312 2 0 MG" "6144 2560 8 0 MG" "2560 248320 1 2 WG")
if(B70_Q4EXP AND B70_MTP)
  set(_m234 2 3 4)
  gemm_sm_rows(GEMM_SM_Q4EXP _m234)
endif()
# Spec 17d: a tensor-parallel card's shapes (model::tp_card_desc, the folded crossing) at its verify lists' M = 2..4 -
# the one-card rows' knobs for the per-card cell of each shape (gemv's S / layout, magic dequant, the 2D read where
# the one-card cell has it); gate||up keeps gemv's grid split-K (no 2x8 work-group: its SiLU is the S 8 build).
set(GEMM_SM_TP17D "5120 8192 1 1 WG MG" "5120 7168 2 0 MG B2D" "5120 17408 8 0 MG" "3072 5120 4 0 MG B2D"
                  "8704 5120 4 0 MG B2D" "5120 124160 1 2 WG")
if(B70_TP AND B70_MTP)
  set(_m2to8 2 3 4 5 6 7 8)   # spec 19 v3c: M = 5..8 for a DFlash drafter's verify rows (kSmVerify8)
  gemm_sm_rows(GEMM_SM_TP17D _m2to8)
  # PERF-44: out_proj / o_proj and down with the folded tp_send as their tail (capture binds these on a TP card)
  set(GEMM_SM_TP17D_TF "3072 5120 4 0 MG B2D TF" "8704 5120 4 0 MG B2D TF")
  gemm_sm_rows(GEMM_SM_TP17D_TF _m2to8)
  set(B70_TP_VERIFY_KERNELS ${B70_TP_VERIFY_KERNELS} ${GEMM_SM_TF_TARGETS} CACHE INTERNAL
      "spec 17d: the binaries a TP verify list binds")
endif()
# Spec 19 v3c: the DFlash drafter's half on a TP card at its block's 8 rows, the candidate heads' half at its 7.
set(GEMM_SM_DFLASH_TP "5120 3072 1 2 WG" "2048 5120 1 2 WG" "5120 17408 1 2 WG" "8704 5120 1 2 WG" "5120 1024 1 2 WG"
                      "5120 3072 4 1 MG" "2048 5120 4 1 MG" "5120 17408 8 1 MG" "8704 5120 4 1 MG" "5120 1024 8 1 MG")
set(GEMM_SM_DFLASH_TP_HEAD "5120 124160 1 1 MG" "5120 124160 1 3")
if(B70_TP AND B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  set(_m8 8)
  set(_m7 7)
  gemm_sm_rows(GEMM_SM_DFLASH_TP _m8)
  gemm_sm_rows(GEMM_SM_DFLASH_TP_HEAD _m7)
endif()
# tests/CMakeLists.txt reads the list (gemm_sm_names_test, gemm_sm_test).
set_property(GLOBAL PROPERTY B70_GEMM_SM_KERNELS ${GEMM_SM_TARGETS})
set_property(GLOBAL PROPERTY B70_GEMM_SM_CONSUMERS ${GEMM_SM_CONSUMERS})
# ==== Spec 28b ==== (end) ====

# ==== Kolibri DSpark: speculative decoding for Kolibri-1 ==== (begin) ====
# Additive only: no existing source is edited and every name below is new (a name some other block already builds is
# skipped, not rebuilt), so every binary above keeps its name, its command line and its bytes; nothing is bound without
# `--spec dflash` on Kolibri-1 (runtime::kolibri::KolibriEngine::enable_spec, runtime::dflash::DSparkDrafter).
#   The verify lists (src/kernels/kolibri_kernels.h verify_variants): Kolibri's decode kernels at M = 2..8 - kol_prep.cl,
#   kol_moe.cl, embed_gather.cl, argmax.cl stage 1, prep.cl's folds, gemv.cl (int4 S2 / S4), gemv_bf16.cl (the bf16 arm,
#   the router, the bf16 head), gemv_i8w.cl (the int8 head); kol_attn.cl at M 2..4 (a verify of more rows runs it twice).
#   The drafter (src/kernels/dflash_kernels.h kDSpark, dspark_draft_variants): df_prep.cl at H 2560, Q20 / KV4, FF 6144,
#   RING 2048, VOCAB 128000; df_attn.cl CAUSAL=1 (`_C`); df_topk.cl over 128000 rows of which 127998 are tokens;
#   df_select.cl's V2 walk at RANK 256 over 128000 codebook rows (the Markov head); gemv_bf16 at M 8 for its linears and
#   the head at the 7 draft rows (gemv_i8w / gemv_bf16).
if(B70_KOLIBRI AND B70_DFLASH AND B70_VERIFY_M8 AND B70_MTP)
  set(KDS_KOL ${CMAKE_CURRENT_SOURCE_DIR}/kolibri)
  set(KDS_DF ${CMAKE_CURRENT_SOURCE_DIR}/dflash)
  set(KDS_DCTL DCTL_K=0 DCTL_PPOS=1 DCTL_PN=2 DCTL_TEMP=3 DCTL_SEED_LO=4 DCTL_SEED_HI=5)
  # --- the verify lists, M = 2..8 ---
  foreach(M 2 3 4 5 6 7 8)
    if(NOT TARGET kernel_gemv_M${M}_K2560_N7168_S2_L0)
      add_gemv_variant(${M} 2560 7168 2 0)                 # q||k||v int4
    endif()
    if(NOT TARGET kernel_gemv_M${M}_K6144_N2560_S4_L0)
      add_gemv_variant(${M} 6144 2560 4 0)                 # o_proj int4
    endif()
    if(NOT TARGET kernel_gemv_bf16_M${M}_K2560_N7168)
      add_gemv_bf16_variant(${M} 2560 7168 64 1)           # q||k||v bf16
    endif()
    if(NOT TARGET kernel_gemv_bf16_M${M}_K6144_N2560)
      add_gemv_bf16_variant(${M} 6144 2560 64 1)           # o_proj bf16
    endif()
    if(NOT TARGET kernel_gemv_bf16_M${M}_K2560_N512_C16_S16)
      add_gemv_bf16_variant(${M} 2560 512 16 16)           # the router
    endif()
    if(NOT TARGET kernel_gemv_bf16_M${M}_K2560_N128000)
      add_gemv_bf16_variant(${M} 2560 128000 64 1)         # lm_head bf16
    endif()
    if(NOT TARGET kernel_gemv_i8w_M${M}_K2560_N128000)
      add_gemv_i8w_variant(${M} 2560 128000)               # lm_head int8
    endif()
    if(NOT TARGET kernel_prep_res_fold_M${M}_K2560_SP0_G20)
      add_prep_res_fold(${M} 2560 0 20)
    endif()
    foreach(SP 1 4)
      if(NOT TARGET kernel_prep_res_fold_M${M}_K2560_SP${SP}_G20_Z)
        add_ocloc_kernel(prep_res_fold_M${M}_K2560_SP${SP}_G20_Z SOURCE ${PREP_CL}
                         DEFINES M=${M} K=2560 S_PREV=${SP} FOLD_G=20 ZERO_RESID=1)
      endif()
    endforeach()
    add_ocloc_kernel(kol_embed_gather_M${M}_D2560_V128000 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                     DEFINES ${CTRL_DEFINES} M=${M} HIDDEN=2560 VOCAB=128000)
    add_ocloc_kernel(kol_argmax_stage1_M${M}_N128000_V128000 SOURCE ${ARGMAX_CL}
                     DEFINES ${CTRL_DEFINES} VOCAB=128000 VOCAB_USED=128000 M=${M})
    add_ocloc_kernel(kol_norm_M${M}_K2560_G20_W20 SOURCE ${KDS_KOL}/kol_prep.cl DEFINES M=${M} K=2560 NORM_G=20 NORM_WGS=20)
    add_ocloc_kernel(kol_post_add_M${M}_K2560_G20 SOURCE ${KDS_KOL}/kol_prep.cl DEFINES M=${M} K=2560 POST_G=20)
    foreach(S 1 2)
      add_ocloc_kernel(kol_attn_prep_M${M}_N7168_S${S}_Q48KV4_R4096 SOURCE ${KDS_KOL}/kol_prep.cl
                       DEFINES ${CTRL_DEFINES} M=${M} QKV_N=7168 QKV_S=${S} Q_HEADS=48 KV_HEADS=4 HD=128 SLIDING=1 RING=4096)
      add_ocloc_kernel(kol_attn_prep_M${M}_N7168_S${S}_Q48KV4_F SOURCE ${KDS_KOL}/kol_prep.cl
                       DEFINES ${CTRL_DEFINES} M=${M} QKV_N=7168 QKV_S=${S} Q_HEADS=48 KV_HEADS=4 HD=128 SLIDING=0 RING=0)
    endforeach()
    add_ocloc_kernel(kol_moe_M${M}_E384_T6_D2560_I512_SH SOURCE ${KDS_KOL}/kol_moe.cl
                     DEFINES M=${M} MOE_E=384 MOE_K=6 HIDDEN=2560 INTER=512 UP_KS=4 DN_KS=2)
  endforeach()
  foreach(M 2 3 4)   # kol_attn.cl's own bound (M <= 4): rows 4..7 of a deeper verify are a second launch
    add_ocloc_kernel(kol_attn_M${M}_T${ATTN_V2_TGT}_Q48KV4_W513_R4096 SOURCE ${KDS_KOL}/kol_attn.cl
                     DEFINES ${CTRL_DEFINES} M=${M} TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=513 RING=4096)
    add_ocloc_kernel(kol_attn_M${M}_T${ATTN_V2_TGT}_Q48KV4_F SOURCE ${KDS_KOL}/kol_attn.cl
                     DEFINES ${CTRL_DEFINES} M=${M} TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=0 RING=0)
  endforeach()
  # --- the drafter ---
  add_ocloc_kernel(df_prep_M8_H2560_I6144_Q20KV4_D128_G16T2_R2048_V128000 SOURCE ${KDS_DF}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${KDS_DCTL} M=8 H=2560 WG=256 EPS=1e-06f VOCAB=128000 GROUP=16 CTAPS=2
                           INTER=6144 NH=20 NKV=4 HD=128 RING=2048)
  add_ocloc_kernel(df_attn_M8_Q20KV4_D128_R2048_W2048_P128_C SOURCE ${KDS_DF}/df_attn.cl
                   DEFINES ${CTRL_DEFINES} ${KDS_DCTL} M=8 NH=20 NKV=4 HD=128 RING=2048 WIN=2048 PB=128 WGA=128 CAUSAL=1)
  add_ocloc_kernel(df_topk_R7_N128000_U127998 SOURCE ${KDS_DF}/df_topk.cl
                   DEFINES R=7 N=128000 USED=127998 TOPK=16 CH=4096 COMPACT=0)
  add_ocloc_kernel(df_select_M8_K16_Q16_RK256_V128000 SOURCE ${KDS_DF}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${KDS_DCTL} MAXR=8 TOPK=16 QS=16 V2=1 RANK=256 VROWS=128000)
  foreach(KN "2560 3584" "2560 2560" "2560 12288" "6144 2560" "12800 2560")   # q||k||v, o, gate||up, down, fc
    separate_arguments(KN)
    list(GET KN 0 _k)
    list(GET KN 1 _n)
    if(NOT TARGET kernel_gemv_bf16_M8_K${_k}_N${_n})
      add_gemv_bf16_variant(8 ${_k} ${_n} 64 1)
    endif()
  endforeach()
  if(NOT TARGET kernel_gemv_bf16_M8_K2560_N1024_C16_S16)
    add_gemv_bf16_variant(8 2560 1024 16 16)               # the context's k||v rows of q||k||v
  endif()
  # The head at the 7 draft rows: M 7 is also a verify M above (built there).
endif()
# ==== Kolibri DSpark ==== (end) ====

# prep_test's two-stage norm at the MoE hiddens (test-only where no model builds them): K 2048 is Ornith's, whose
# 20 fold work-groups run a chunk of 103 (not a multiple of the 256-lane tree), K 2560 a chunk of 128.
foreach(SP 0 4)
  if(NOT TARGET kernel_prep_res_fold_M1_K2048_SP${SP}_G20)
    add_prep_res_fold(1 2048 ${SP} 20)
  endif()
endforeach()
if(NOT TARGET kernel_prep_norm_finish_M1_K2048_G20_W20)
  add_prep_norm_finish(1 2048 20 20)
endif()
if(NOT TARGET kernel_prep_res_fold_M1_K2560_SP0_G20)
  add_prep_res_fold(1 2560 0 20)
endif()
add_prep_norm_finish(1 2560 20 20)

add_subdirectory(prefill)

# Release-format expert linears at Inkling's shapes. The route and activation
# variants are selected separately by the model runtime.
foreach(fmt NVFP4 MXFP8)
  if(fmt STREQUAL "NVFP4")
    set(wfmt 1)
  else()
    set(wfmt 2)
  endif()
  add_ocloc_kernel(kol_moe_E128_T8_D2304_I1152_${fmt} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_moe.cl
                   DEFINES WFMT=${wfmt} MOE_E=128 MOE_K=8 HIDDEN=2304 INTER=1152 UP_KS=4 DN_KS=1 MOE_SHARED=0)
  add_ocloc_kernel(pf_moe_q4gemm_K2304_N2304_SILU_${fmt} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/pf_moe_q4gemm.cl
                   DEFINES WFMT=${wfmt} K=2304 N=2304 EXPERTS=128 SILU_EPI=1 WGN=4
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_q4gemm_K1152_N2304_${fmt} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/pf_moe_q4gemm.cl
                   DEFINES WFMT=${wfmt} K=1152 N=2304 EXPERTS=128 WGN=4
                   OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(expert_fp_bytes_${fmt} SOURCE ${CMAKE_SOURCE_DIR}/tests/kernels/expert_fp_bytes.cl
                   DEFINES WFMT=${wfmt} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/expert_fp.h)
endforeach()

# Inkling decode and verify. The format-only variants above are also used by
# the expert byte test. The capped variants here implement Inkling's epilogue.
set(INKLING_KERNELS "")
set(INKLING_SMALL_M "")
foreach(M RANGE 1 8)
  foreach(ctrl "4608;64;1" "64;2304;2")
    list(GET ctrl 0 k)
    list(GET ctrl 1 n)
    list(GET ctrl 2 phase)
    set(name gemv_bf16_M${M}_K${k}_N${n}_INKCTRL)
    add_ocloc_kernel(${name} SOURCE ${GEMV_BF16_CL}
                     DEFINES M=${M} K=${k} N=${n} COLS_PER_WG=64 KSPLIT=1 INK_CONTROLLER=${phase})
    list(APPEND INKLING_KERNELS ${name})
  endforeach()
  foreach(kn "2304;3072" "2304;1024" "2304;384" "2304;64" "2304;200064" "3072;2304")
    list(GET kn 0 k)
    list(GET kn 1 n)
    if(M EQUAL 1)
      add_gemv_i8w_variant(${M} ${k} ${n})
      list(APPEND INKLING_KERNELS gemv_i8w_M${M}_K${k}_N${n})
    else()
      add_gemm_sm_variant(${M} ${k} ${n} 1 2)
      list(APPEND INKLING_KERNELS gemm_sm_i8w_M${M}_K${k}_N${n})
      list(APPEND INKLING_SMALL_M kernel_gemm_sm_i8w_M${M}_K${k}_N${n})
    endif()
  endforeach()
  foreach(kn "2304;128" "4608;64" "64;2304" "2304;1024")
    list(GET kn 0 k)
    list(GET kn 1 n)
    add_gemv_bf16_variant(${M} ${k} ${n} 64 1)
    list(APPEND INKLING_KERNELS gemv_bf16_M${M}_K${k}_N${n})
  endforeach()
  if(M EQUAL 1)
    set(name gemv_i8w_M1_K2304_N3072_QKV1024)
    add_ocloc_kernel(${name} SOURCE ${GEMV_I8W_CL} DEFINES M=1 K=2304 N=3072 QKV_NKV=1024)
  else()
    set(name gemm_sm_i8w_M${M}_K2304_N3072_QKV1024)
    add_ocloc_kernel(${name} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gemm_sm.cl
                     DEFINES M=${M} K=2304 N=3072 S=1 LAYOUT=2 QKV_NKV=1024)
  endif()
  list(APPEND INKLING_KERNELS ${name})
  if(M EQUAL 1)
    set(name gemv_i8w_M1_K2304_N3072_QKV1024_FULL)
    add_ocloc_kernel(${name} SOURCE ${GEMV_I8W_CL}
                     DEFINES M=1 K=2304 N=3072 QKV_NKV=1024 INK_ATTN_PROJECTIONS=1)
  else()
    set(name gemm_sm_i8w_M${M}_K2304_N3072_QKV1024_FULL)
    add_ocloc_kernel(${name} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gemm_sm.cl
                     DEFINES M=${M} K=2304 N=3072 S=1 LAYOUT=2 QKV_NKV=1024 INK_ATTN_PROJECTIONS=1)
  endif()
  list(APPEND INKLING_KERNELS ${name})
  set(name ink_embed_gather_M${M}_D2304_V201024)
  add_ocloc_kernel(${name} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                   DEFINES ${CTRL_DEFINES} M=${M} HIDDEN=2304 VOCAB=201024)
  list(APPEND INKLING_KERNELS ${name})
  set(name ink_argmax_stage1_M${M}_N200064_V200058)
  add_ocloc_kernel(${name} SOURCE ${ARGMAX_CL} DEFINES ${CTRL_DEFINES} M=${M} VOCAB=200064 VOCAB_USED=200058)
  list(APPEND INKLING_KERNELS ${name})

  set(ink_union)
  if(M GREATER 1)
    set(ink_union MOE_UNION=1)
  endif()
  foreach(row "NVFP4;1" "MXFP8;2")
    list(GET row 0 fmt)
    list(GET row 1 wfmt)
    set(name kol_moe_M${M}_E128_T8_D2304_I1152_CAP_${fmt})
    add_ocloc_kernel(${name} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_moe.cl
                    DEFINES M=${M} WFMT=${wfmt} MOE_E=128 MOE_K=8 HIDDEN=2304 INTER=1152 UP_KS=4 DN_KS=1
                    MOE_SHARED=0 MOE_OUT_F32=1 SWIGLU_CAP_G=4.0f SWIGLU_CAP_U=25.0f ${ink_union})
    list(APPEND INKLING_KERNELS ${name})
    if(M EQUAL 1)
      add_ocloc_kernel(${name}_POST SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_moe.cl
                       DEFINES M=1 WFMT=${wfmt} MOE_E=128 MOE_K=8 HIDDEN=2304 INTER=1152 UP_KS=4 DN_KS=1
                               MOE_SHARED=0 MOE_OUT_F32=1 SWIGLU_CAP_G=4.0f SWIGLU_CAP_U=25.0f INK_MOE_POST=1)
      list(APPEND INKLING_KERNELS ${name}_POST)
    endif()
  endforeach()
  foreach(window "W1024_R4096;1024;4096" "F;0;0")
    list(GET window 0 tag)
    list(GET window 1 w)
    list(GET window 2 ring)
    set(name kol_attn_M${M}_T32_Q24KV8_REL16E1024_SINK_GATE_${tag})
    add_ocloc_kernel(${name} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_attn.cl
                    DEFINES ${CTRL_DEFINES} M=${M} TGT=32 Q_HEADS=24 KV_HEADS=8 WINDOW=${w} RING=${ring}
                    REL_BIAS=1024 REL_DIM=16 SINK_TOK=1 SINK_PITCH=64 GATE=1)
    list(APPEND INKLING_KERNELS ${name})
  endforeach()
endforeach()

add_ocloc_kernel(kol_route_E128_T8_N128_L128_RENORM SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_moe.cl
                 DEFINES ROUTE_E=128 ROUTE_K=8 ROUTE_WG=128 ROUTE_EPL=1 ROUTE_LN=128 ROUTE_RENORM=1 ROUTE_SCALE=8.0f)
list(APPEND INKLING_KERNELS kol_route_E128_T8_N128_L128_RENORM)
add_ocloc_kernel(kol_route_E128_T8_N128_L128_RENORM_M1_FUSED SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_moe.cl
                 DEFINES ROUTE_E=128 ROUTE_K=8 ROUTE_WG=128 ROUTE_EPL=1 ROUTE_LN=128
                         ROUTE_RENORM=1 ROUTE_SCALE=8.0f INK_ROUTE_M1=1)
list(APPEND INKLING_KERNELS kol_route_E128_T8_N128_L128_RENORM_M1_FUSED)
foreach(kn "2304;3072" "3072;2304" "2304;9216" "9216;2304")
  list(GET kn 0 k)
  list(GET kn 1 n)
  add_gemv_bf16_variant(8 ${k} ${n} 64 1)
  list(APPEND INKLING_KERNELS gemv_bf16_M8_K${k}_N${n})
endforeach()
add_gemv_i8w_variant(1 256 200064)
add_gemv_bf16_variant(1 5120 768 64 1)
list(APPEND INKLING_KERNELS gemv_i8w_M1_K256_N200064 gemv_bf16_M1_K5120_N768)
set(INKLING_DCTL DCTL_K=0 DCTL_PPOS=1 DCTL_PN=2 DCTL_TEMP=3 DCTL_SEED_LO=4 DCTL_SEED_HI=5)
add_ocloc_kernel(df_prep_M8_H2304_I9216_Q24KV8_D128_R2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/dflash/df_prep.cl
                 DEFINES ${CTRL_DEFINES} ${INKLING_DCTL} M=8 H=2304 WG=256 EPS=1e-06f INTER=9216 NH=24 NKV=8 HD=128 RING=2048 INK_CONTEXT=1)
add_ocloc_kernel(df_attn_M8_Q24KV8_D128_R2048_W1024_P128 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/dflash/df_attn.cl
                 DEFINES ${CTRL_DEFINES} ${INKLING_DCTL} M=8 NH=24 NKV=8 HD=128 RING=2048 WIN=1024 PB=128 WGA=128 FIXED_DRAFT_WINDOW=1)
list(APPEND INKLING_KERNELS df_prep_M8_H2304_I9216_Q24KV8_D128_R2048 df_attn_M8_Q24KV8_D128_R2048_W1024_P128)

add_ocloc_kernel(ink_glue_H2304 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/ink_glue.cl DEFINES H=2304 ${CTRL_DEFINES})
add_ocloc_kernel(ink_glue_H2304_CTRL SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/ink_glue.cl
                 DEFINES H=2304 ${CTRL_DEFINES} INK_CONTROLLER_FUSION=1)
list(APPEND INKLING_KERNELS ink_glue_H2304_CTRL)
add_ocloc_kernel(ink_prep_Q24KV8_D128_REL16 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/ink_prep.cl DEFINES ${CTRL_DEFINES})
add_ocloc_kernel(ink_prep_Q24KV8_D128_REL16_FUSED SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/ink_prep.cl
                 DEFINES ${CTRL_DEFINES} INK_FUSED_PREP=1)
list(APPEND INKLING_KERNELS ink_prep_Q24KV8_D128_REL16_FUSED)
add_ocloc_kernel(df_attn_M8_Q24KV8_D128_FULL SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/dflash/df_attn.cl
                 DEFINES ${CTRL_DEFINES} ${INKLING_DCTL} M=8 NH=24 NKV=8 HD=128 RING=0 WIN=0 PB=128 WGA=128
                 FULL_TARGETS=32)
add_ocloc_kernel(df_inkling_markov_H2304_R256 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/ink_draft.cl
                 DEFINES ${CTRL_DEFINES})
list(APPEND INKLING_KERNELS ink_glue_H2304 ink_prep_Q24KV8_D128_REL16
     df_attn_M8_Q24KV8_D128_FULL df_inkling_markov_H2304_R256)
set_property(GLOBAL APPEND PROPERTY B70_GEMM_SM_KERNELS ${INKLING_SMALL_M})
set(B70_INKLING_KERNELS ${INKLING_KERNELS} CACHE INTERNAL "Inkling shared kernel variants")
