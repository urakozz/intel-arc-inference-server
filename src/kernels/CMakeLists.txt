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
# already. No such variant is built until a model can load: Ornith's
# (model::ornith(), 2048 / 16 q 2 kv / 16 k 32 v) arrive with spec 15c, together
# with the .cl defines that make those constants overridable; until then the
# missing binary throws at capture, as for any unbuilt variant.
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

add_subdirectory(prefill)
