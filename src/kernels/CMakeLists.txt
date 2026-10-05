add_ocloc_kernel(noop SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/noop.cl)
add_ocloc_kernel(bw_sum SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bw_sum.cl)
add_ocloc_kernel(ctrl_read SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ctrl_read.cl)

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
  # Spec 14, Agnes 3.0 Flash: PROVISIONAL cells, COPIED from the nearest Qwen3.8
  # production cells above (gate||up 5120x34816 S8, down 17408x5120 S4), not
  # measured. The box sweep (B70_AGNES_SWEEP, docs/probe-gemv-2026-08-24.md's
  # method) replaces them and model::agnes()'s {S, layout} together:
  # docs/superpowers/plans/2026-10-03-spec14-validation-checklist.md, "GEMV tuning".
  elseif(L EQUAL 0 AND K EQUAL 5120 AND N EQUAL 38912 AND S EQUAL 8)
    list(APPEND extra_defs GEMV_DEQ_SHIFT=1)                    # PROVISIONAL
  elseif(L EQUAL 0 AND K EQUAL 19456 AND N EQUAL 5120 AND S EQUAL 4)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)     # PROVISIONAL
  endif()
  add_ocloc_kernel(gemv_M${M}_K${K}_N${N}_S${S}_L${L} SOURCE ${GEMV_CL}
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
add_prep_res_norm(1 5120 4)    # Task 4: after out_proj / o_proj / down (S = 4)
add_prep_res_norm(1 5120 16)   # pre-Task-4 control retained for prep_test/probes
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
# extrapolated. docs/15 §2's fit over four low-depth points predicted 214.5 us
# per launch at a 128-position block; the retile measured **296.684**, +38%, so
# that model died the way its two predecessors did -- and so did the refit that
# replaced it, which called 260.0 us at a 64-position block and measured
# 224.046, -13.8% the other way. Four models dead on this kernel, and no fifth
# is offered. Four block sizes were then
# measured in situ at depth 4096 (docs/12 `attn` -> Measured, docs/15 §L5),
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
add_attn_prep(1)
add_attn_decode(1 16384)   # the loader's default max_len (docs/13-loader.md)
add_attn_reduce(1 16384)
add_attn_decode(1 131072)  # spec 6: 128k context (--max-len 131072)
add_attn_reduce(1 131072)
add_attn_decode(1 32768)   # spec 6 K3a: flash_long_test's 32k engines (the composed path's
add_attn_reduce(1 32768)   # pf_s / pf_p do not fit beside a 131072 KV cache)
add_attn_decode(1 4096)    # attn_test's buffers: 64 blocks, cheap to fill
add_attn_reduce(1 4096)
# M = 2 is compiled per spec 1 §9 (the M loop must compile); all three are also
# RUN by attn_test, as the only cover for per-m causal masking (a position valid
# for m = 1 and masked for m = 0, both inside one block and across a block edge).
#
# Note the asymmetry, which is deliberate while M = 2 is test-only: the M = 2
# decode/reduce pair exists **at L4096 only**, the length attn_test allocates.
# There is no `attn_decode_M2_L16384_B*` / `attn_reduce_M2_L16384_B*`, so
# `kernels::attn_{decode,reduce}_variant(2, 16384, 64)` names no file and the
# runtime cannot bind M = 2 attention at the loader's default max_len. Whoever
# turns M > 1 on (plan 3's MTP work) adds the L16384 rows here first.
add_attn_prep(2)
add_attn_decode(2 4096)
add_attn_reduce(2 4096)

# Spec 10 (plan 10b): decode attention v2 (attn_v2.cl), M = 1..4, one binary per M
# holding attn_decode_v2 + attn_reduce_v2. No MAXLEN (the grid is (4, TGT) and the
# partials [24][TGT][M][258] at every max_len); TGT is the host's
# runtime::DecodeScratch::kAttnV2Blocks and is in the name (`_T32`), as ATTN_BLOCK is.
set(ATTN_V2_TGT 32 CACHE INTERNAL
    "work-groups per kv head of attn_decode_v2; must equal runtime::DecodeScratch::kAttnV2Blocks")
# exp2 with the scale folded (operator-approved 2026-09-28 subject to the gates) FAILS the
# golden gate: cjk row 9, a determined row with a 0.0625 golden margin, flips (spec 10 §8).
# So v2 ships with attn.cl's `exp`; the exp2 form stays buildable for the record.
option(B70_ATTN_V2_EXP2 "decode attention v2 with exp2 (fails the golden gate; spec 10 §8)" OFF)
if(B70_ATTN_V2_EXP2)
  set(ATTN_V2_EXP2 1)
else()
  set(ATTN_V2_EXP2 0)
endif()
foreach(MM 1 2 3 4)
  add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                   DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2})
endforeach()

# Spec 8a (MTP probe): the whole decode list at M = 2, 3, 4 - every variant
# runtime::build binds at max_len 16384 with the production checkpoint's bf16
# lm_head - so tools/probe/probe_verify_m can capture the real list at M rows.
# OFF by default: production builds compile exactly what they did before.
option(B70_DECODE_EXTRA_M "build the M = 2..4 decode-list variants (spec 8a probe)" OFF)
# Spec 8 (plan 8b): MTP speculative decoding. ON by default - it only ADDS binaries
# (the M = 2..4 verify lists at max_len 16384, the per-row GDN slot variants, the MTP
# head's own); every existing binary's command line is unchanged, and an engine without
# `--mtp` binds none of them.
option(B70_MTP "build the MTP verify / draft variants (spec 8)" ON)
if(B70_DECODE_EXTRA_M OR B70_MTP)
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
    add_attn_decode(${MM} 16384)
    add_attn_reduce(${MM} 16384)
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
    add_ocloc_kernel(prep_norm_finish_M${MM}_K5120_G20_W20_X10240 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 NORM_G=20 NORM_WGS=20 X_STRIDE=10240)
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
  # none of them. The head's K is the model's hidden (5120 on Qwen3.8 and Agnes), so these
  # serve both.
  set(DRAFT_VOCAB_CL ${CMAKE_CURRENT_SOURCE_DIR}/draft_vocab.cl)
  foreach(NV 32768 65536 131072)
    add_gemv_i8w_variant(1 5120 ${NV})
    add_gemv_bf16_variant(1 5120 ${NV} 64 1)
    add_ocloc_kernel(dv_argmax_N${NV} SOURCE ${DRAFT_VOCAB_CL} DEFINES ${CTRL_DEFINES} NV=${NV})
  endforeach()
endif()

# --- Spec 15b: other models' shapes ------------------------------------------
# Every binary above is built at Qwen3.8's shapes - hidden 5120, FA 24 q / 4 kv
# heads, GDN 16 k / 48 v heads - which the .cl sources bake by default. The
# host names a binary from the model descriptor (src/kernels/shape_suffix.h):
# Qwen3.8's shapes, and Agnes's (the same), carry no suffix, so every name and
# command line here is unchanged. Another model's kernels that bake a shape
# their name does not otherwise carry are suffixed `_D<hidden>` (embed_gather),
# `_GK<k>V<v>` (gdn_step*, prep_gated_head) and `_Q<q>KV<kv>` (attn_prep,
# attn_decode / attn_reduce, attn_v2); GEMV / prep_res_* names carry K and N
# already. Ornith's (model::ornith(), 2048 / 16 q 2 kv / 16 k 32 v) are built by
# the spec 15c block below, with the .cl defines that make those constants
# overridable (GDN_K_HEADS / GDN_V_HEADS, FA_Q_HEADS / FA_KV_HEADS); any other
# shape's missing binary throws at capture, as for any unbuilt variant.
#
# --- Spec 14: Agnes 3.0 Flash (model::agnes()) ------------------------------
# Additive only: every Qwen3.8 binary above keeps its name and its command line
# (G0 compares their checksums against a main build). Agnes differs from Qwen3.8
# in two baked numbers - the MLP intermediate (17408 + 2048 folded = 19456) and
# the GDN layer count (54, which sizes the MTP slot stride) - and in the two GEMV
# shapes that follow from the first.
option(B70_AGNES "build the Agnes 3.0 Flash variants (spec 14)" ON)
# The tuning sweep over the two new GEMV shapes, both layouts, S in {1,2,4,8,16}
# (all divide K/64 = 80 and 304) - for the box session that replaces the
# provisional rows. OFF by default: 20 ocloc compiles nobody binds.
option(B70_AGNES_SWEEP "build the Agnes GEMV tuning matrix (spec 14 validation)" OFF)
if(B70_AGNES)
  set(AGNES_I 19456)
  add_gemv_variant(1 5120 38912 8 0)    # gate'||up'  PROVISIONAL S8 L0
  add_gemv_variant(1 19456 5120 4 0)    # down'       PROVISIONAL S4 L0
  add_ocloc_kernel(prep_silu_mul_M1_I${AGNES_I} SOURCE ${PREP_CL} DEFINES M=1 INTERMEDIATE=${AGNES_I})
  # Agnes's tokenizer defines 12 more ids (248077..248088, its specials), so its greedy
  # argmax masks from 248089 (model::agnes().vocab_used, kernels::argmax_stage1_variant).
  set(AGNES_ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=248320 VOCAB_USED=248089)
  add_ocloc_kernel(argmax_stage1_M1_V248089 SOURCE ${ARGMAX_CL} DEFINES ${AGNES_ARGMAX_DEFINES} M=1)
  if(B70_AGNES_SWEEP)
    foreach(L 0 1)
      foreach(S 1 2 4 8 16)
        if(NOT (L EQUAL 0 AND S EQUAL 8))
          add_gemv_variant(1 5120 38912 ${S} ${L})
        endif()
        if(NOT (L EQUAL 0 AND S EQUAL 4))
          add_gemv_variant(1 19456 5120 ${S} ${L})
        endif()
      endforeach()
    endforeach()
  endif()
  if(B70_DECODE_EXTRA_M OR B70_MTP)
    foreach(MM 2 3 4)   # the verify lists' M rows (spec 8), at Agnes's shapes
      add_gemv_variant(${MM} 5120 38912 8 0)
      add_gemv_variant(${MM} 19456 5120 4 0)
      add_ocloc_kernel(prep_silu_mul_M${MM}_I${AGNES_I} SOURCE ${PREP_CL}
                       DEFINES M=${MM} INTERMEDIATE=${AGNES_I})
      add_ocloc_kernel(argmax_stage1_M${MM}_V248089 SOURCE ${ARGMAX_CL}
                       DEFINES ${AGNES_ARGMAX_DEFINES} M=${MM})
    endforeach()
  endif()
  if(B70_MTP)
    # gdn_step_slots bakes one MtpBuffers::gdn_spec slot: 54 GDN layers x 48 x 128 x 128
    # floats = 42467328 (kernels::gdn_step_slots_variant names it `_G54`).
    foreach(MM 1 2 3 4)
      add_ocloc_kernel(gdn_step_slots_M${MM}_G54 SOURCE ${GDN_CL}
                       DEFINES ${GDN_DEFINES} SPEC_SLOTS=1 CTRL_LIVE=19 SPEC_SLOT_STRIDE=42467328
                               M=${MM})
    endforeach()
  endif()
endif()

# --- Spec 15c: Ornith 1.5 35B-A3B (model::ornith()), the M = 1 decode list ------
# Additive only, like the Agnes block: every binary above keeps its name and command
# line (tools/kernel_cmdlines diff), and the .cl edits that made the head counts
# overridable leave the default token stream untouched (the shape defines below are
# the only way into the new branches). What Ornith's decode list binds
# (runtime::build, capture.cc): 526 launches = embed + 40 x 13 + 5.
#   linears   q||k||v / qkv||z / out_proj = o_proj at hidden 2048 - PROVISIONAL cells,
#             model_desc.cc's copied {S, layout} (no extra GEMV_* defines: P0 measures)
#   a||b      gemv_bf16 at {16, 16} (64 real columns, padded to 128 as Qwen3.8's 96)
#   router    gemv_bf16 over the 256 router rows || the shared gate, 272 columns
#   lm_head   bf16, int8 (spec 9) or a checkpoint's int4, at K 2048
#   the norm pair at K 2048: SP0 after layer 0's embedding and after every MoE block
#             (moe_down folds into the residual itself), SP4 after out_proj / o_proj
#   GDN       gdn_step / prep_gated_head at 16 k / 32 v heads; the small block's
#             negA / dt_bias at 131072 / 131200 B = 32768 / 32800 floats
#             (loader::make_small_layout(2048, 8192, 32), checked by model_desc_test)
#   FA        attn_prep and decode attention v2 at 16 q / 2 kv heads (GQA 8); v1
#             (B70_DECODE_ATTN=v1) is not built for Ornith
#   MoE       moe.cl's three kernels: 256 experts, top-8, hidden 2048, expert 512
# argmax_stage1_M1_V248070: Ornith masks from 248070 (tokenizer.json's last id is </think>
# = 248069), not Qwen3.8's 248077 - the seven specials above it decode to nothing.
option(B70_ORNITH "build the Ornith 1.5 35B-A3B decode variants (spec 15c)" ON)
if(B70_ORNITH)
  add_gemv_variant(1 2048 12288 1 1)    # qkv||z      PROVISIONAL S1 L1
  add_gemv_variant(1 4096 2048 4 0)     # out_proj, o_proj  PROVISIONAL S4 L0
  add_gemv_variant(1 2048 9216 2 0)     # q||k||v     PROVISIONAL S2 L0
  add_gemv_variant(1 2048 248320 1 1)   # lm_head int4 (a --quant_lm_head checkpoint)
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
# The operator's rotkv scheme (spec 12 §8): every binary that writes or reads the int8
# cache, from ONE new source (kv8.cl, per-family defines as pf_int8.cl). Additive only:
# no existing source changes, so every existing binary keeps its name, its command line
# and its bytes, and a `--kv-cache bf16` engine (the default) binds none of these.
#   attn_prep_M<M>_KV8        the decode writer at M = 1..4 (QKV_S = 2: the main FA layers)
#   attn_prep_M<M>_S1_KV8     the MTP head's writer (its bf16 qkv: QKV_S = 1), B70_MTP
#   attn_v2_M<M>_T<TGT>_KV8   decode attention v2 over int8 + scales, the un-rotating reduce
# The prefill three are in prefill/CMakeLists.txt. Shapes are Qwen3.8's (Agnes shares
# them); another model's FA heads name a `_Q<q>KV<kv>` binary nothing builds yet.
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
endif()

# ==== Spec 18b: K2-Horizon decode (model::k2()) ============================= (begin) ====
# Additive only: no existing source is edited, so every binary above keeps its name, its
# command line and its bytes (tools/kernel_cmdlines' diff: additions only). K2-only
# kernels live in src/kernels/k2/ and are named `k2_*` with every shape define in the name
# (src/kernels/k2_kernels.h is the host half); the reused sources run at K2's shapes:
#   gemv.cl         q||k||gate||v 2560x10240, q||k||gate||v_router 2560x9280 (S2 L0),
#                   o_proj 4096x2560, gate||up 2560x12288, down 6144x2560 (S4 L0) - all
#                   PROVISIONAL cells (model::K2Desc's {S, layout}, no GEMV_* tuning
#                   defines) until 18b Task 4's sweep
#   gemv_bf16.cl    the MoE router 2560 -> 128 rows ({16, 16}), lm_head 2560 -> 250624
#   gemv_i8w.cl     the int8 lm_head (spec 9)
#   prep.cl         prep_res_fold at K 2560 (stage A of the grouped norm, unchanged)
#   embed_gather.cl / argmax.cl at K2's vocabulary 250624 (fully used)
# What K2's decode list binds (runtime/k2/k2_capture.cc): 717 launches = embed + 3 dense
# layers x 12 + 45 MoVA/MoE layers x 15 + 5; 813 under B70_K2_ATTN=eager (4 attention launches
# a layer for flash's 2).
option(B70_K2 "build the K2-Horizon decode variants (spec 18b)" ON)
if(B70_K2)
  set(K2_CL ${CMAKE_CURRENT_SOURCE_DIR}/k2)
  add_gemv_variant(1 2560 10240 2 0)    # q||k||gate||v        PROVISIONAL S2 L0
  add_gemv_variant(1 2560 9280 2 0)     # q||k||gate||v_router PROVISIONAL S2 L0
  add_gemv_variant(1 4096 2560 4 0)     # o_proj               PROVISIONAL S4 L0
  add_gemv_variant(1 2560 12288 4 0)    # dense gate||up       PROVISIONAL S4 L0
  add_gemv_variant(1 6144 2560 4 0)     # dense down           PROVISIONAL S4 L0
  add_gemv_bf16_variant(1 2560 128 16 16)       # the MoE router, rows 100..127 zero
  add_gemv_bf16_variant(1 2560 250624 64 1)     # lm_head bf16
  add_gemv_i8w_variant(1 2560 250624)           # lm_head int8 (spec 9)
  add_prep_res_fold(1 2560 0 20)        # after embed_gather and after every MoE block
  add_prep_res_fold(1 2560 4 20)        # after o_proj and the dense down (S4)
  add_ocloc_kernel(k2_embed_gather_M1_D2560_V250624 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                   DEFINES ${CTRL_DEFINES} M=1 HIDDEN=2560 VOCAB=250624)
  set(K2_ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=250624 VOCAB_USED=250624)
  add_ocloc_kernel(k2_argmax_stage1_M1_N250624_V250624 SOURCE ${ARGMAX_CL}
                   DEFINES ${K2_ARGMAX_DEFINES} M=1)
  add_ocloc_kernel(k2_argmax_stage2_N250624 SOURCE ${ARGMAX_CL} DEFINES ${K2_ARGMAX_DEFINES})
  # k2_prep.cl: the grouped norm's stage B (2 groups of 1280, plain w), the dense SiLU x up,
  # and the attention prep (q / k RoPE over all 128 dims, the gate, the cache write).
  add_ocloc_kernel(k2_norm_M1_K2560_G20_W20_NG2 SOURCE ${K2_CL}/k2_prep.cl
                   DEFINES M=1 K=2560 NORM_G=20 NORM_WGS=20 NORM_GROUPS=2)
  add_ocloc_kernel(k2_silu_M1_I6144_S4 SOURCE ${K2_CL}/k2_prep.cl DEFINES M=1 SILU_I=6144 SILU_S=4)
  set(K2_PREP_DEFINES ${CTRL_DEFINES} M=1 QKV_S=2 Q_HEADS=32 KV_HEADS=8 HD=128)
  add_ocloc_kernel(k2_attn_prep_M1_N10240_S2_Q32KV8_V SOURCE ${K2_CL}/k2_prep.cl
                   DEFINES ${K2_PREP_DEFINES} QKV_N=10240 V_FROM_PARTIALS=1)
  add_ocloc_kernel(k2_attn_prep_M1_N9280_S2_Q32KV8 SOURCE ${K2_CL}/k2_prep.cl
                   DEFINES ${K2_PREP_DEFINES} QKV_N=9280 V_FROM_PARTIALS=0)
  # k2_attn.cl: decode attention v2 at head_dim 128 / GQA 4 with the softplus gate.
  add_ocloc_kernel(k2_attn_M1_T${ATTN_V2_TGT}_Q32KV8 SOURCE ${K2_CL}/k2_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=32 KV_HEADS=8)
  # k2_attn_eager.cl: the reference's bf16 eager chain (B70_K2_ATTN=eager, spec 18 §10) - the
  # same defines; a new source, so k2_attn.cl's binary keeps its bytes.
  add_ocloc_kernel(k2_attn_eager_M1_T${ATTN_V2_TGT}_Q32KV8 SOURCE ${K2_CL}/k2_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=32 KV_HEADS=8)
  # k2_moe.cl: the sigmoid router (MoVA's over the fused GEMV's v_router columns, the MoE's
  # over its bf16 GEMV), the MoE block (100 + 1 shared, top-8), MoVA's value experts.
  add_ocloc_kernel(k2_route_M1_E64_T4_N9280_O9216_S2 SOURCE ${K2_CL}/k2_moe.cl
                   DEFINES M=1 ROUTE_E=64 ROUTE_K=4 ROUTE_WG=64 ROUTE_LN=9280 ROUTE_LOFF=9216
                           ROUTE_LS=2 ROUTE_SCALE=2.5f)
  add_ocloc_kernel(k2_route_M1_E100_T8_N128_O0_S1 SOURCE ${K2_CL}/k2_moe.cl
                   DEFINES M=1 ROUTE_E=100 ROUTE_K=8 ROUTE_WG=128 ROUTE_LN=128 ROUTE_LOFF=0
                           ROUTE_LS=1 ROUTE_SCALE=2.5f)
  add_ocloc_kernel(k2_moe_M1_E100_T8_D2560_I768 SOURCE ${K2_CL}/k2_moe.cl
                   DEFINES M=1 MOE_E=100 MOE_K=8 HIDDEN=2560 INTER=768 UP_KS=4 DN_KS=2)
  add_ocloc_kernel(k2_mova_M1_E64_T4_D2560_N1024 SOURCE ${K2_CL}/k2_moe.cl
                   DEFINES ${CTRL_DEFINES} M=1 MOVA_E=64 MOVA_K=4 MOVA_D=2560 MOVA_N=1024 MOVA_KS=4)
endif()
# ==== Spec 18b: K2-Horizon decode ============================================ (end) ====

# ==== Spec 18c: K2-Horizon prefill (model::k2()) ============================ (begin) ====
# Additive only, as 18b's block: no existing source changes, so every binary above (and in
# prefill/) keeps its name and command line. New K2-only sources in src/kernels/k2/
# (k2_pf_linear.cl, k2_pf_moe.cl, k2_pf_attn.cl); reused sources at K2's shapes:
#   prefill/pf_embed.cl     the chunk's embedding gather (K2's vocabulary in the name)
#   prefill/pf_prep.cl      pf_res_fold at K 2560 (stage A of the grouped norm, runtime M)
#   k2/k2_prep.cl           k2_norm_finish and k2_attn_prep at M = 2048 (kernels::k2::kPfC),
#                           S 1: decode's chains, row for row, over the prefill GEMM's one slice
#   k2/k2_moe.cl            k2_route over [C][ln] logits, grid (1, C): MoVA's from the fused
#                           row's v_router columns at pitch 9472; the MoE's is decode's binary
#   prefill/pf_gemv_bf16.cl the MoE router GEMV over the chunk at decode's {16, 16} tiling
#   prefill/pf_moe_gemm.cl  the grouped expert GEMMs: gate||up 2560 x 1536 (SiLU epilogue),
#                           down 768 x 2560, MoVA's value experts 2560 x 1024 (plain)
#   prefill/pf_gemm.cl      the dense and attention linears over k2_pf_dequant_slab's slabs
#                           (pf_gemm_T0 / pf_gemm_T0_SILU, shape-free: the existing binaries)
# What K2's prefill chunk binds: runtime/k2/k2_prefill.cc (2392 launches at every C, derived in
# runtime/k2/k2_sizes.cc); the head is decode's binaries.
if(B70_K2)
  set(K2_PF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/k2)
  set(K2_PF_REUSE ${CMAKE_CURRENT_SOURCE_DIR}/prefill)
  add_ocloc_kernel(k2_pf_embed_gather_D2560_V250624 SOURCE ${K2_PF_REUSE}/pf_embed.cl
                   DEFINES HIDDEN=2560 VOCAB=250624)
  foreach(SP 0 1)
    add_ocloc_kernel(pf_res_fold_K2560_SP${SP}_G20 SOURCE ${K2_PF_REUSE}/pf_prep.cl
                     DEFINES K=2560 S_PREV=${SP} FOLD_G=20)
  endforeach()
  add_ocloc_kernel(k2_norm_M2048_K2560_G20_W20_NG2 SOURCE ${K2_PF_SRC}/k2_prep.cl
                   DEFINES M=2048 K=2560 NORM_G=20 NORM_WGS=20 NORM_GROUPS=2)
  # The int4 linears' slabs (layout 0): q||k||gate||v, q||k||gate||v_router (a 256-column
  # tail holding the router's 64), o_proj, dense gate||up, dense down.
  foreach(row "2560 10240" "2560 9280" "4096 2560" "2560 12288" "6144 2560")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(k2_pf_dequant_slab_K${K}_N${N} SOURCE ${K2_PF_SRC}/k2_pf_linear.cl
                     DEFINES K=${K} N=${N})
  endforeach()
  set(K2_PF_PREP_DEFINES ${CTRL_DEFINES} M=2048 QKV_S=1 Q_HEADS=32 KV_HEADS=8 HD=128)
  add_ocloc_kernel(k2_attn_prep_M2048_N10240_S1_Q32KV8_V SOURCE ${K2_PF_SRC}/k2_prep.cl
                   DEFINES ${K2_PF_PREP_DEFINES} QKV_N=10240 V_FROM_PARTIALS=1)
  add_ocloc_kernel(k2_attn_prep_M2048_N9472_S1_Q32KV8 SOURCE ${K2_PF_SRC}/k2_prep.cl
                   DEFINES ${K2_PF_PREP_DEFINES} QKV_N=9472 V_FROM_PARTIALS=0)
  # The flash attention at head_dim 128 / GQA 4: the default (fp32 scores and probabilities)
  # and the eager-mirroring variant (B70_K2_ATTN=eager), each gated (the walk's) and ungated
  # (_O, the K1 test's). Natural exp (EXP2 0) in both, as K2's decode attention and the
  # reference - PROVISIONAL for the default, a speed knob for plan 18c Task 3. 256 GRF as
  # pf_flash_attn.
  foreach(EG 0 1)
    foreach(GT 0 1)
      set(_n k2_pf_flash_attn_Q32KV8)
      set(_x2 0)
      if(EG)
        string(APPEND _n _EAGER)
      endif()
      if(NOT GT)
        string(APPEND _n _O)
      endif()
      add_ocloc_kernel(${_n} SOURCE ${K2_PF_SRC}/k2_pf_attn.cl
                       DEFINES KT=64 RPW=8 HPW=4 Q_HEADS=32 KV_HEADS=8 EXP2=${_x2} EAGER=${EG}
                               GATE_EPI=${GT}
                       OPTIONS -cl-intel-256-GRF-per-thread)
    endforeach()
  endforeach()
  add_ocloc_kernel(k2_route_M1_E64_T4_N9472_O9216_S1 SOURCE ${K2_PF_SRC}/k2_moe.cl
                   DEFINES M=1 ROUTE_E=64 ROUTE_K=4 ROUTE_WG=64 ROUTE_LN=9472 ROUTE_LOFF=9216
                           ROUTE_LS=1 ROUTE_SCALE=2.5f)
  add_ocloc_kernel(k2_pf_moe_E100_T8_D2560_I768 SOURCE ${K2_PF_SRC}/k2_pf_moe.cl
                   DEFINES PF_E=100 PF_K=8 PF_WG=128 PF_SHARED=1 PF_D=2560 PF_INTER=768 TM=32 KC=2048)
  add_ocloc_kernel(k2_pf_mova_E64_T4_D2560_N1024 SOURCE ${K2_PF_SRC}/k2_pf_moe.cl
                   DEFINES PF_E=64 PF_K=4 PF_WG=64 PF_SHARED=0 PF_D=2560 PF_VN=1024 TM=32 KC=2048)
  add_ocloc_kernel(pf_moe_router_K2560_N128 SOURCE ${K2_PF_REUSE}/pf_gemv_bf16.cl
                   DEFINES K=2560 N=128 COLS_PER_WG=16 KSPLIT=16)
  add_ocloc_kernel(pf_moe_gemm_K2560_N1536_SILU SOURCE ${K2_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=2560 N=1536 TM=32 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_K768_N2560 SOURCE ${K2_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=768 N=2560 TM=32 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_K2560_N1024 SOURCE ${K2_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=2560 N=1024 TM=32 OPTIONS -cl-intel-256-GRF-per-thread)
endif()
# ==== Spec 18c: K2-Horizon prefill ============================================ (end) ====

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
#                      tools/mac/clrun/moe_run.cc and moe_m_test on the card pin it).
#                      argmax_stage1_M<M> is Qwen3.8's (vocab_used 248077 on both).
#   verify, M = 1..4   gdn_step_slots at 30 GDN layers (`_G30`, SPEC_SLOT_STRIDE = 30 x 32
#                      x 128 x 128 = 15728640 floats) and Ornith's heads; the head's KV fill:
#                      fc (bf16 4096 -> 2048), q||k||v (bf16 2048 -> 9216), the two pre-fc
#                      norms into one [M][4096] row (X_STRIDE), the ZERO_RESID fold and
#                      attn_prep over an S = 1 qkv (QKV_S = 1)
#   draft, M = 1       o (bf16 4096 -> 2048, fc's binary), the post norm's SP1 fold; the
#                      head's MoE layer runs 15c's moe_M1 and router binaries; with a draft
#                      vocabulary (spec 8 §11) the compact head at K 2048
# Not built for Ornith: the int8 KV cache's MTP writer (spec 12b's kv8 binaries are at
# Qwen3.8's heads), an int4-checkpoint lm_head at M > 1 (as for Qwen3.8).
if(B70_ORNITH AND B70_MTP)
  foreach(MM 2 3 4)
    add_gemv_variant(${MM} 2048 12288 1 1)       # qkv||z            PROVISIONAL S1 L1
    add_gemv_variant(${MM} 4096 2048 4 0)        # out_proj, o_proj  PROVISIONAL S4 L0
    add_gemv_variant(${MM} 2048 9216 2 0)        # q||k||v           PROVISIONAL S2 L0
    add_gemv_bf16_variant(${MM} 2048 128 16 16)  # a||b
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
    add_gemv_bf16_variant(${MM} 4096 2048 64 1)    # fc (and, at M = 1, the draft's o)
    add_gemv_bf16_variant(${MM} 2048 9216 64 1)    # q||k||v
    add_ocloc_kernel(prep_norm_finish_M${MM}_K2048_G20_W20_X4096 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=2048 NORM_G=20 NORM_WGS=20 X_STRIDE=4096)
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

add_subdirectory(prefill)
