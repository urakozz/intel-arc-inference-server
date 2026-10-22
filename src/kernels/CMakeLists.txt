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
  # Spec 23c, Muse-Glimmer-30B: PROVISIONAL cells, COPIED from the nearest Qwen3.8 production cells above
  # (q||k||v 5120x14336 S2, out_proj 6144x5120 S4, gate||up 5120x34816 S8, down 17408x5120 S4), not measured;
  # 23c Task 7's sweep (r44.sweeps) replaces them and model::MuseGlimmerDesc's S together.
  elseif(L EQUAL 0 AND K EQUAL 6656 AND N EQUAL 8704 AND S EQUAL 2)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)     # PROVISIONAL q||gate||k||v
  elseif(L EQUAL 0 AND K EQUAL 4096 AND N EQUAL 6656 AND S EQUAL 4)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)     # PROVISIONAL o_proj
  elseif(L EQUAL 0 AND K EQUAL 6656 AND N EQUAL 39936 AND S EQUAL 8)
    list(APPEND extra_defs GEMV_DEQ_SHIFT=1)                    # PROVISIONAL gate||up
  elseif(L EQUAL 0 AND K EQUAL 19968 AND N EQUAL 6656 AND S EQUAL 4)
    list(APPEND extra_defs GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1)     # PROVISIONAL down
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

# ==== Spec 19a Task 4: the verify cost at M = 5..8 and DFlash's draft cost ==== (begin) ==
# Plan: docs/superpowers/plans/2026-10-05-spec19a-dflash-p0.md, Task 4. Additive only: no
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
  foreach(MM 5 6 7 8)
    # the decode list at M rows (B70_MTP's M = 2..4 block's rows, int8 lm_head only)
    add_gemv_variant(${MM} 5120 16384 1 1)     # qkv||z
    add_gemv_variant(${MM} 6144 5120 4 0)      # out_proj, o_proj
    add_gemv_variant(${MM} 5120 34816 8 0)     # gate||up (and the drafter's, M = 8)
    add_gemv_variant(${MM} 17408 5120 4 0)     # down (and the drafter's, M = 8)
    add_gemv_variant(${MM} 5120 14336 2 0)     # qkv
    add_gemv_i8w_variant(${MM} 5120 248320)    # lm_head (int8; at M = 7 also the draft head)
    add_gemv_bf16_variant(${MM} 5120 128 16 16)    # a||b
    add_prep_res_fold(${MM} 5120 0 20)
    add_prep_res_fold(${MM} 5120 4 20)
    add_prep_norm_finish(${MM} 5120 20 20)
    add_ocloc_kernel(prep_silu_mul_M${MM} SOURCE ${PREP_CL} DEFINES M=${MM})
    add_ocloc_kernel(prep_gated_head_M${MM} SOURCE ${PREP_CL} DEFINES M=${MM})
    add_ocloc_kernel(embed_gather_M${MM} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} HIDDEN=5120 VOCAB=248320)
    add_ocloc_kernel(argmax_stage1_M${MM} SOURCE ${ARGMAX_CL} DEFINES ${ARGMAX_DEFINES} M=${MM})
    add_attn_prep(${MM})
    add_ocloc_kernel(attn_v2_M${MM}_T${ATTN_V2_TGT} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2})
    # the verify list's own: the 8-slot gdn_step and the MTP head's KV fill of M rows
    add_ocloc_kernel(gdn_step_slots_M${MM}_N8 SOURCE ${GDN_CL} DEFINES ${VM8_SLOT_DEFINES} M=${MM})
    add_gemv_bf16_variant(${MM} 10240 5120 64 1)   # fc
    add_gemv_bf16_variant(${MM} 5120 14336 64 1)   # q||k||v
    add_ocloc_kernel(prep_norm_finish_M${MM}_K5120_G20_W20_X10240 SOURCE ${PREP_CL}
                     DEFINES M=${MM} K=5120 NORM_G=20 NORM_WGS=20 X_STRIDE=10240)
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
# Plan: docs/superpowers/plans/2026-10-05-spec19b-dflash-drafter.md. Additive only: no existing variant's name or
# command line moves, and no engine binds any of these without a drafter (19c's --spec dflash). The draft list and
# the projection list (runtime/dflash/dflash_capture.cc) bind, at z-lab/Qwen3.8-27B-DFlash2's shapes beside Qwen3.8
# (kernels::dflash::kQwen38, src/kernels/dflash_kernels.h):
#   df_prep.cl    df_embed, df_norm, df_conv_prep / df_conv_add, df_add, df_silu_mul, df_qkv_prep, df_ctx_store
#   df_attn.cl    df_attn_part (grid (8 kv heads, 16 blocks of 128 ring offsets)) + df_attn_reduce
#   df_topk.cl    the 16 best head logits per draft row, two stages; the full head masks ids >= 248077, a draft
#                 vocabulary's (COMPACT) maps its indices through the id table
#   df_select.cl  DFlash 2's candidate-selector walk (V2), or v1's per-row pick
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
  add_ocloc_kernel(df_attn_M8_Q32KV8_D128_R2048_W2048_P128 SOURCE ${DF_DIR}/df_attn.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} M=8 NH=32 NKV=8 HD=128 RING=2048 WIN=2048 PB=128 WGA=128
                           CAUSAL=0)
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
  add_ocloc_kernel(df_select_M8_K16_Q16_RK256_V248320 SOURCE ${DF_DIR}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} MAXR=8 TOPK=16 QS=16 V2=1 RANK=256 VROWS=248320)
  add_ocloc_kernel(df_select_v1_M8_K16 SOURCE ${DF_DIR}/df_select.cl
                   DEFINES ${CTRL_DEFINES} ${DCTL_DEFINES} MAXR=8 TOPK=16 QS=16 V2=0)
  add_gemv_bf16_variant(7 5120 248320 64 1)           # the full bf16 head at 7 rows
endif()
# ==== Spec 19b ==== (end) ====

# ==== Spec 19c: DFlash in the decode loop ==== (begin) ====
# Plan: docs/superpowers/plans/2026-10-05-spec19c-dflash-engine-loop.md. Additive only: no existing variant's name or
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
  foreach(MM 5 6 8)
    add_gemv_bf16_variant(${MM} 5120 248320 64 1)
  endforeach()
endif()
# ==== Spec 19c ==== (end) ====

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
#             (B70_DECODE_ATTN=v1) is not built for Ornith
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

# ==== Spec 18e: K2-Horizon's int8 KV cache (`--kv-cache int8`) ============== (begin) ====
# rotkv at head_dim 128 (spec 12 §8; src/common/kv8.h hd128): every binary that writes or reads
# K2's int8 cache, from ONE new source (k2/k2_kv8.cl, per-family defines as kv8.cl) plus
# k2_moe.cl's k2_mova_value at MOVA_STAGE 1 (its routed mix into a staging row, which the
# writer quantises). Additive only: every existing binary keeps its name and command line, and
# a `--kv-cache bf16` K2 engine (the default) binds none of these. Suffix `_KV8` (spec 12b's) on
# the names that carry K2's `Q32KV8` heads already.
#   decode   k2_attn_prep_M1_N10240_S2_Q32KV8_V_KV8 / ..._N9280_S2_Q32KV8_KV8   the writer
#            k2_mova_M1_E64_T4_D2560_N1024_STAGE                             MoVA's v, staged
#            k2_attn_M1_T<TGT>_Q32KV8_KV8           flash decode + the un-rotating reduce
#            k2_attn_eager_M1_T<TGT>_Q32KV8_KV8     eager score / P·V / un-rotating reduce (the
#                                                   softmax is k2_attn_eager's binary, unchanged)
#   prefill  k2_attn_prep_M2048_N10240_S1_Q32KV8_V_KV8 / ..._N9472_S1_Q32KV8_KV8  the writer
#            k2_pf_flash_attn_Q32KV8[_EAGER][_O]_KV8  the flash over int8 (gated, and the K1
#                                                   test's ungated un-rotated o)
#            (MoVA's combine is k2_pf_mova's binary with pos 0 into the staging row)
if(B70_K2 AND B70_KV8)
  set(K2_KV8_CL ${CMAKE_CURRENT_SOURCE_DIR}/k2/k2_kv8.cl)
  set(K2_KV8_HEADS Q_HEADS=32 KV_HEADS=8)
  add_ocloc_kernel(k2_attn_prep_M1_N10240_S2_Q32KV8_V_KV8 SOURCE ${K2_KV8_CL}
                   DEFINES ${CTRL_DEFINES} K2KV8_PREP=1 M=1 QKV_S=2 ${K2_KV8_HEADS} QKV_N=10240
                           V_FROM_PARTIALS=1)
  add_ocloc_kernel(k2_attn_prep_M1_N9280_S2_Q32KV8_KV8 SOURCE ${K2_KV8_CL}
                   DEFINES ${CTRL_DEFINES} K2KV8_PREP=1 M=1 QKV_S=2 ${K2_KV8_HEADS} QKV_N=9280
                           V_FROM_PARTIALS=0)
  add_ocloc_kernel(k2_mova_M1_E64_T4_D2560_N1024_STAGE SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/k2/k2_moe.cl
                   DEFINES ${CTRL_DEFINES} M=1 MOVA_E=64 MOVA_K=4 MOVA_D=2560 MOVA_N=1024 MOVA_KS=4
                           MOVA_STAGE=1)
  add_ocloc_kernel(k2_attn_M1_T${ATTN_V2_TGT}_Q32KV8_KV8 SOURCE ${K2_KV8_CL}
                   DEFINES ${CTRL_DEFINES} K2KV8_DEC=1 M=1 TGT=${ATTN_V2_TGT} ${K2_KV8_HEADS})
  add_ocloc_kernel(k2_attn_eager_M1_T${ATTN_V2_TGT}_Q32KV8_KV8 SOURCE ${K2_KV8_CL}
                   DEFINES ${CTRL_DEFINES} K2KV8_EAGER=1 M=1 TGT=${ATTN_V2_TGT} ${K2_KV8_HEADS})
  add_ocloc_kernel(k2_attn_prep_M2048_N10240_S1_Q32KV8_V_KV8 SOURCE ${K2_KV8_CL}
                   DEFINES ${CTRL_DEFINES} K2KV8_PREP=1 M=2048 QKV_S=1 ${K2_KV8_HEADS} QKV_N=10240
                           V_FROM_PARTIALS=1)
  add_ocloc_kernel(k2_attn_prep_M2048_N9472_S1_Q32KV8_KV8 SOURCE ${K2_KV8_CL}
                   DEFINES ${CTRL_DEFINES} K2KV8_PREP=1 M=2048 QKV_S=1 ${K2_KV8_HEADS} QKV_N=9472
                           V_FROM_PARTIALS=0)
  foreach(EG 0 1)
    foreach(GT 0 1)
      set(_n k2_pf_flash_attn_Q32KV8)
      if(EG)
        string(APPEND _n _EAGER)
      endif()
      if(NOT GT)
        string(APPEND _n _O)
      endif()
      add_ocloc_kernel(${_n}_KV8 SOURCE ${K2_KV8_CL}
                       DEFINES K2KV8_FLASH=1 KT=64 RPW=8 HPW=4 ${K2_KV8_HEADS} EAGER=${EG} GATE_EPI=${GT}
                       OPTIONS -cl-intel-256-GRF-per-thread)
    endforeach()
  endforeach()
endif()
# ==== Spec 18e: K2-Horizon's int8 KV cache ==================================== (end) ====
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
  # o_proj int4 PROVISIONAL S4 L0 is gemv_M1_K6144_N2560_S4_L0, the binary K2's dense down already
  # builds (spec 18b block); prep_res_fold_M1_K2560_SP0_G20 likewise. Built here only without K2, so
  # no target is defined twice:
  if(NOT B70_K2)
    add_gemv_variant(1 6144 2560 4 0)
    add_prep_res_fold(1 2560 0 20)
  endif()
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
#                     cache (full), k2_pf_attn.cl's DPAS walk at GQA 12, no gate; and its eager twin
#   kol_pf_linear.cl  the bf16 attention arm's slabs with a zero-padded tail (o_proj's 512 columns)
# Reused at Kolibri's shapes: kol_prep.cl at M = 2048 (kernels::kolibri::kPfC: the norm, the post-add,
# the attention prep at S 1), prefill/pf_embed.cl, prefill/pf_prep.cl's pf_res_fold (SP0, and _Z over
# o_proj's one slice), prefill/pf_gemv_bf16.cl (the router at decode's {16, 16} tiling),
# prefill/pf_moe_gemm.cl (gate||up 2560 x 1024 SiLU, down 512 x 2560), k2/k2_pf_linear.cl (the int4
# arm's layout-0 slabs) and the shape-free pf_gemm_T0. The route is 20c's decode binary on grid (1, C).
# What a Kolibri prefill chunk binds: runtime/kolibri/kolibri_prefill.cc (45 launches a layer + 2,
# 2252 at 50 layers; derived in runtime/kolibri/kolibri_sizes.cc).
if(B70_KOLIBRI)
  set(KOL_PF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kolibri)
  set(KOL_PF_REUSE ${CMAKE_CURRENT_SOURCE_DIR}/prefill)
  add_ocloc_kernel(kol_pf_embed_gather_D2560_V128000 SOURCE ${KOL_PF_REUSE}/pf_embed.cl DEFINES HIDDEN=2560 VOCAB=128000)
  # K2's spec 18c block builds these two names (pf_res_fold SP0 at K 2560, o_proj's 6144 x 2560 slab):
  # built here only without K2, so no target is defined twice.
  if(NOT B70_K2)
    add_ocloc_kernel(pf_res_fold_K2560_SP0_G20 SOURCE ${KOL_PF_REUSE}/pf_prep.cl DEFINES K=2560 S_PREV=0 FOLD_G=20)
    add_ocloc_kernel(k2_pf_dequant_slab_K6144_N2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/k2/k2_pf_linear.cl DEFINES K=6144 N=2560)
  endif()
  add_ocloc_kernel(pf_res_fold_K2560_SP1_G20_Z SOURCE ${KOL_PF_REUSE}/pf_prep.cl DEFINES K=2560 S_PREV=1 FOLD_G=20 ZERO_RESID=1)
  add_ocloc_kernel(k2_pf_dequant_slab_K2560_N7168 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/k2/k2_pf_linear.cl DEFINES K=2560 N=7168)
  foreach(row "2560 7168" "6144 2560")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(kol_pf_bf16_slab_K${K}_N${N} SOURCE ${KOL_PF_SRC}/kol_pf_linear.cl DEFINES K=${K} N=${N})
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
  # k2_pf_attn / pf_flash_attn.
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

# ==== Spec 21c: Qwen3.8-Flash-Next decode (model::qwen4exp()) ==== (begin) ====
# Additive only but for ONE source: prep.cl gains -DGDN_GATE_SIGMOID (prep_gated_head's sigmoid gate, the `_SIG`
# binary); every existing prep.cl variant keeps its command line and its preprocessed source token for token (the
# 9a67889 rule; G0's sha256 on the box). Family-only kernels live in src/kernels/qwen4exp/ and are named `q4_*` with
# every shape define in the name (src/kernels/qwen4exp_kernels.h is the host half); the reused sources run at this
# family's shapes:
#   gemv.cl         qkv||z 2560x16384 (S1 L0: gdn_step / prep_gated_head read S 1), q||gate||k||v 2560x13312 (S2 L0),
#                   out_proj / o_proj 6144x2560 (S4 L0) - ours (int4 g64); PROVISIONAL cells (model::Qwen4ExpDesc's
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
option(B70_Q4EXP "build the Qwen3.8-Flash-Next decode variants (spec 21c)" ON)
if(B70_Q4EXP)
  set(Q4_CL ${CMAKE_CURRENT_SOURCE_DIR}/qwen4exp)
  add_gemv_variant(1 2560 16384 1 0)            # qkv||z int4                     S1 L0 (fixed)
  add_gemv_variant(1 2560 13312 2 0)            # q||gate||k||v int4              PROVISIONAL S2 L0
  # out_proj / o_proj int4 PROVISIONAL S4 L0 is gemv_M1_K6144_N2560_S4_L0 (K2's dense down, Kolibri's o_proj);
  # the bf16 6144 x 2560 is Kolibri's o_proj bf16; a||b {16, 16} is K2's router: built here only without them.
  if(NOT B70_K2 AND NOT B70_KOLIBRI)
    add_gemv_variant(1 6144 2560 4 0)
  endif()
  if(NOT B70_KOLIBRI)
    add_gemv_bf16_variant(1 6144 2560 64 1)
  endif()
  if(NOT B70_K2)
    add_gemv_bf16_variant(1 2560 128 16 16)
  endif()
  add_gemv_bf16_variant(1 2560 16384 64 1)      # qkv||z bf16 (Intel's)
  add_gemv_bf16_variant(1 2560 13312 64 1)      # q||gate||k||v bf16 (Intel's)
  add_gemv_bf16_variant(1 10240 336 16 16)      # HC down || block_inject (324 rows used)
  add_gemv_bf16_variant(1 10240 320 16 16)      # the final mixer's down
  add_gemv_bf16_variant(1 2560 640 16 16)       # the indexer's index_qk_proj
  add_gemv_bf16_variant(1 2560 528 16 16)       # the router (512) + shared_expert_gate (row 512), zero to 528
  add_gemv_bf16_variant(1 2560 12800 64 1)      # the PLE key || value projections
  add_gemv_bf16_variant(1 2560 248320 64 1)     # lm_head bf16
  add_gemv_i8w_variant(1 2560 248320)           # lm_head int8 (spec 9)
  add_ocloc_kernel(embed_gather_M1_D2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                   DEFINES ${CTRL_DEFINES} M=1 HIDDEN=2560 VOCAB=248320)
  add_ocloc_kernel(prep_gated_head_M1_SIG SOURCE ${PREP_CL} DEFINES M=1 GDN_GATE_SIGMOID=1)
  set(Q4_FA_HEADS FA_Q_HEADS=24 FA_KV_HEADS=2)
  add_ocloc_kernel(attn_prep_M1_Q24KV2 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=1 ${Q4_FA_HEADS})
  add_ocloc_kernel(attn_prep_M1_Q24KV2_S1 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=1 QKV_S=1 ${Q4_FA_HEADS})
  add_ocloc_kernel(attn_v2_M1_T${ATTN_V2_TGT}_Q24KV2 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/attn_v2.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} EXP2=${ATTN_V2_EXP2} ${Q4_FA_HEADS})
  # q4_hc.cl: the combine + grouped norm by what it folds (_E embedding, _S<S> the mixer's slices, _Y the MoE's y,
  # _X nothing; _NN without the norm) and the up-mix (_I with the 4 inject rows; the final mixer without).
  add_ocloc_kernel(q4_hc_combine_norm_M1_E SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=1 HC_SRC=0 HC_NORM=1)
  foreach(S 1 4)
    add_ocloc_kernel(q4_hc_combine_norm_M1_S${S} SOURCE ${Q4_CL}/q4_hc.cl
                     DEFINES ${CTRL_DEFINES} M=1 HC_SRC=1 SRC_S=${S} HC_NORM=1)
  endforeach()
  add_ocloc_kernel(q4_hc_combine_norm_M1_Y SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=1 HC_SRC=2 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M1_Y_NN SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=1 HC_SRC=2 HC_NORM=0)
  add_ocloc_kernel(q4_hc_combine_norm_M1_X SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=1 HC_SRC=3 HC_NORM=1)
  add_ocloc_kernel(q4_hc_up_mix_M1_I SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=1 UP_DOWN=336 UP_INJECT=1)
  add_ocloc_kernel(q4_hc_up_mix_M1 SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=1 UP_DOWN=320 UP_INJECT=0)
  # q4_ple.cl: the gather from host USM by the table's scale dtype (+ q4_ple_check), the block.
  add_ocloc_kernel(q4_ple_gather_M1_BF16 SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=1 PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=1)
  add_ocloc_kernel(q4_ple_gather_M1_F32 SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=1 PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=0)
  add_ocloc_kernel(q4_ple_block_M1 SOURCE ${Q4_CL}/q4_ple.cl DEFINES ${CTRL_DEFINES} M=1 PLE_BLOCK=1)
  # q4_qsa.cl (prep, score, select), q4_qsa_attn.cl (flash: attn + reduce), q4_qsa_attn_eager.cl. LIST_ROW / COUNT_W /
  # LISTMAX: kernels::qwen4exp::kListRow / kCountWord / kListMax; TGT: kAttnTgt.
  set(Q4_LIST_DEFINES LIST_ROW=2064 COUNT_W=2052)
  add_ocloc_kernel(q4_qsa_M1_T512_W1024 SOURCE ${Q4_CL}/q4_qsa.cl
                   DEFINES ${CTRL_DEFINES} M=1 TOPB=512 SEL_WG=1024 ${Q4_LIST_DEFINES})
  add_ocloc_kernel(q4_qsa_attn_M1_T32 SOURCE ${Q4_CL}/q4_qsa_attn.cl DEFINES ${CTRL_DEFINES} M=1 TGT=32 ${Q4_LIST_DEFINES})
  add_ocloc_kernel(q4_qsa_attn_eager_M1 SOURCE ${Q4_CL}/q4_qsa_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=1 ${Q4_LIST_DEFINES} LISTMAX=2052)
  # q4_moe.cl: the 512-expert softmax route (two experts a lane), the 11-slot block by the shared expert's form.
  # UP_KS / DN_KS PROVISIONAL (kernels::qwen4exp::kUpKs / kDnKs; Task 7 sweeps them).
  add_ocloc_kernel(q4_route_M1_E512_T10_N528_L256 SOURCE ${Q4_CL}/q4_moe.cl
                   DEFINES M=1 ROUTE_E=512 ROUTE_K=10 ROUTE_WG=256 ROUTE_LN=528)
  add_ocloc_kernel(q4_moe_M1_E512_T10_D2560_I640_SH4 SOURCE ${Q4_CL}/q4_moe.cl
                   DEFINES M=1 MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=640 UP_KS=4 DN_KS=2 SHARED_BF16=0)
  add_ocloc_kernel(q4_moe_M1_E512_T10_D2560_I640_SHB SOURCE ${Q4_CL}/q4_moe.cl
                   DEFINES M=1 MOE_E=512 MOE_K=10 HIDDEN=2560 INTER=640 UP_KS=4 DN_KS=2 SHARED_BF16=1)
endif()
# ==== Spec 21c: Qwen3.8-Flash-Next decode ==== (end) ====

# ==== Spec 21d: Qwen3.8-Flash-Next prefill ==== (begin) ====
# Additive but for three sources, each gaining one define that every existing binary leaves unset - their
# preprocessed source token for token as before (the 9a67889 rule; G0's sha256 on the box, r33.k0):
#   prefill/pf_gated_head.cl  GDN_GATE_SIGMOID (pf_gated_head_SIG: the sigmoid gate, prep.cl's _SIG twin)
#   qwen4exp/q4_qsa.cl        QSA_PF (the prep at M = 2048 without its in-launch ring write, + q4_qsa_ring; the indexer
#                             rows at the GEMM's pitch 768)
#   qwen4exp/q4_ple.cl        PLE_PF (the gather's tokens from the chunk's id buffer, no ring write)
# New family sources (src/kernels/qwen4exp/, kernels::qwen4exp's spec 21d block is the host half):
#   q4_pf_moe.cl    the 512-expert ushort sort, gather, a weight batch's dequant (block 512 = the shared expert: int4
#                   dequantised or Intel's bf16 copied), the combine in decode's order
#   q4_pf_attn.cl   the sparse flash over each row's own list (rows past 2050; DPAS; flash and eager forms)
#   q4_pf_ple.cl    the PLE block over a chunk in three launches (gate, conv, the rings)
# Reused at this family's shapes: 21c's q4_hc.cl at M = 2048 (combine _E / _S1 / _Y / _X / _Y_NN, up-mix _I at the
# GEMM's pitch 512), prefill/pf_embed.cl and pf_gemv_bf16.cl (a||b, the router at decode's {16, 16} tiling) at hidden
# 2560, pf_dequant_slab.cl (qkv||z, q||gate||k||v int4 layout 0), kolibri/kol_pf_linear.cl's bf16 slab (q4_ prefix),
# pf_attn_prep.cl (q16), pf_attn.cl (the gate) and pf_flash_attn.cl at 24 q / 2 kv heads (HPW 6 of GQA 12, KT 64, RPW
# 8, EXP2 as the existing binary, 256 GRF), pf_moe_gemm.cl (gate||up 2560 x 1280 SiLU, down 640 x 2560); bound by
# name, nothing built here: Qwen3.8's pf_gdn_conv / _wy / _scan (the GDN shape is Qwen3.8's), pf_gemm_T0, 21c's
# q4_route_M1 (grid (1, C)); K2's / Kolibri's k2_pf_dequant_slab_K6144_N2560 (out_proj / o_proj int4) is built here
# only when neither block is on. What a chunk binds: runtime/qwen4exp/qwen4exp_prefill.cc (launch counts in
# runtime/qwen4exp/qwen4exp_sizes.cc: they depend on the chunk's dense / sparse rows).
if(B70_Q4EXP)
  set(Q4_PF_REUSE ${CMAKE_CURRENT_SOURCE_DIR}/prefill)
  add_ocloc_kernel(pf_gated_head_SIG SOURCE ${Q4_PF_REUSE}/pf_gated_head.cl DEFINES GDN_GATE_SIGMOID=1)
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
  if(NOT B70_K2 AND NOT B70_KOLIBRI)
    add_ocloc_kernel(k2_pf_dequant_slab_K6144_N2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/k2/k2_pf_linear.cl DEFINES K=6144 N=2560)
  endif()
  # the bf16 slabs: Intel's dense rows (qkv||z, q||gate||k||v, out / o), the indexer, the HC down||inject (336 rows,
  # a 512-column slab), the PLE key||value
  foreach(row "2560 16384" "2560 13312" "6144 2560" "2560 640" "10240 336" "2560 12800")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(q4_pf_bf16_slab_K${K}_N${N} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_pf_linear.cl
                     DEFINES K=${K} N=${N})
  endforeach()
  # 21c's sources at M = 2048 (kernels::qwen4exp::kPfC)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_E SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=0 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_S1 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=1 SRC_S=1 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_Y SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=2 HC_NORM=1)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_Y_NN SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=2 HC_NORM=0)
  add_ocloc_kernel(q4_hc_combine_norm_M2048_X SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=2048 HC_SRC=3 HC_NORM=1)
  add_ocloc_kernel(q4_hc_up_mix_M2048_I_D512 SOURCE ${Q4_CL}/q4_hc.cl
                   DEFINES ${CTRL_DEFINES} M=2048 UP_DOWN=512 UP_INJECT=1)
  add_ocloc_kernel(q4_qsa_M2048_T512_W1024_PF SOURCE ${Q4_CL}/q4_qsa.cl
                   DEFINES ${CTRL_DEFINES} M=2048 TOPB=512 SEL_WG=1024 ${Q4_LIST_DEFINES} QSA_PF=1)
  add_ocloc_kernel(q4_ple_gather_M2048_BF16_PF SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=2048 PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=1 PLE_PF=1)
  add_ocloc_kernel(q4_ple_gather_M2048_F32_PF SOURCE ${Q4_CL}/q4_ple.cl
                   DEFINES ${CTRL_DEFINES} M=2048 PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=0 PLE_PF=1)
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
  add_ocloc_kernel(pf_moe_gemm_K2560_N1280_SILU SOURCE ${Q4_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=2560 N=1280 TM=32 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_K640_N2560 SOURCE ${Q4_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=640 N=2560 TM=32 OPTIONS -cl-intel-256-GRF-per-thread)
  # the PLE block over a chunk
  add_ocloc_kernel(q4_pf_ple_C2048 SOURCE ${Q4_CL}/q4_pf_ple.cl DEFINES ${CTRL_DEFINES} PF_C=2048)
endif()
# ==== Spec 21d: Qwen3.8-Flash-Next prefill ==== (end) ====

# ==== Spec 21e: Qwen3.8-Flash-Next MTP (B70_Q4EXP and B70_MTP) ==== (begin) ====
# Additions only; no existing source changes. kernels::qwen4exp's spec 21e block is the host half
# (verify_variants / draft_variants / mtp_prefill_variants). What the lists bind (runtime/qwen4exp/qwen4exp_capture.cc):
#   verify at M = k + 1 (spec 8's lists): every 21c decode kernel at M = 2..4 - the q4_ sources (combine _E / _S1 /
#     _S4 / _Y / _Y_NN / _X, up-mix, the PLE gather, q4_qsa, the attention (flash, eager), the route and the 11-slot
#     MoE), attn_prep _Q24KV2 / _S1, prep_gated_head _SIG, embed_gather D2560, the GEMVs at this family's shapes (gemv
#     int4 S1 / S2 / S4 layout 0, gemv_bf16 for Intel's dense arms, the HC downs, the indexer, the router, a||b, the PLE
#     key||value, lm_head bf16 and int8); Qwen3.8's argmax_stage1_M<M> by name (vocab 248077 used);
#     gdn_step_slots_M<1..4>_G1 - gdn_step.cl's SPEC_SLOTS at SPEC_SLOT_STRIDE = ONE layer's state (48 x 128 x 128
#     floats): the engine lays gdn_spec out layer-major, so no layer count is baked (Qwen3.8's _M<M> bakes 48 layers);
#     q4_pf_ple_C4 - 21d's three-launch PLE block at C = 4 (q4_ple_block is M = 1 only);
#   the MTP head (draft at M = 1, its KV pass in every verify at M = 1..4, the prefill's pass at M = 2048):
#     q4_mtp.cl (q4_mtp_norm: the gather of t, gemma_rms of the embedding and the SINGLE / per-STREAM RMS of the
#     4-stream R; q4_mtp_fuse: + e on every stream) at M = 1..4 and _PF at 2048; fc_embedding / fc_hidden gemv_bf16
#     {2560, 2560} at M = 1..4 (fc_hidden at M = 4 a row: its 4 streams) and the bf16 slab {2560, 2560} for prefill.
#     The head's own layer is bf16 dense / bf16 shared (loader::q4_mtp_desc) with int4 experts: 21c's / 21d's Intel-form
#     binaries serve it.
if(B70_Q4EXP AND B70_MTP)
  set(Q4_SLOT_DEFINES ${GDN_DEFINES} SPEC_SLOTS=1 CTRL_LIVE=19 SPEC_SLOT_STRIDE=786432)
  foreach(MM 1 2 3 4)
    add_ocloc_kernel(gdn_step_slots_M${MM}_G1 SOURCE ${GDN_CL} DEFINES ${Q4_SLOT_DEFINES} M=${MM})
    add_gemv_bf16_variant(${MM} 2560 2560 64 1)   # fc_embedding / fc_hidden
    foreach(NF SINGLE STREAM)
      if(NF STREQUAL SINGLE)
        set(_single 1)
      else()
        set(_single 0)
      endif()
      add_ocloc_kernel(q4_mtp_M${MM}_${NF} SOURCE ${Q4_CL}/q4_mtp.cl DEFINES ${CTRL_DEFINES} M=${MM} MTP_NORM_SINGLE=${_single})
    endforeach()
    if(MM EQUAL 1)
      continue()
    endif()
    add_gemv_variant(${MM} 2560 16384 1 0)            # qkv||z int4
    add_gemv_variant(${MM} 2560 13312 2 0)            # q||gate||k||v int4
    add_gemv_variant(${MM} 6144 2560 4 0)             # out_proj / o_proj int4
    add_gemv_bf16_variant(${MM} 2560 16384 64 1)      # qkv||z bf16 (Intel's)
    add_gemv_bf16_variant(${MM} 2560 13312 64 1)      # q||gate||k||v bf16 (Intel's, the head's)
    add_gemv_bf16_variant(${MM} 6144 2560 64 1)       # out_proj / o_proj bf16
    add_gemv_bf16_variant(${MM} 2560 128 16 16)       # a||b
    add_gemv_bf16_variant(${MM} 10240 336 16 16)      # HC down || block_inject
    add_gemv_bf16_variant(${MM} 10240 320 16 16)      # the final mixer's down
    add_gemv_bf16_variant(${MM} 2560 640 16 16)       # the indexer
    add_gemv_bf16_variant(${MM} 2560 528 16 16)       # the router + shared gate
    add_gemv_bf16_variant(${MM} 2560 12800 64 1)      # the PLE key || value
    add_gemv_bf16_variant(${MM} 2560 248320 64 1)     # lm_head bf16
    add_gemv_i8w_variant(${MM} 2560 248320)           # lm_head int8 (spec 9)
    add_ocloc_kernel(embed_gather_M${MM}_D2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} HIDDEN=2560 VOCAB=248320)
    add_ocloc_kernel(prep_gated_head_M${MM}_SIG SOURCE ${PREP_CL} DEFINES M=${MM} GDN_GATE_SIGMOID=1)
    add_ocloc_kernel(attn_prep_M${MM}_Q24KV2 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} ${Q4_FA_HEADS})
    add_ocloc_kernel(attn_prep_M${MM}_Q24KV2_S1 SOURCE ${ATTN_CL} DEFINES ${ATTN_DEFINES} M=${MM} QKV_S=1 ${Q4_FA_HEADS})
    add_ocloc_kernel(q4_hc_combine_norm_M${MM}_E SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=0 HC_NORM=1)
    foreach(S 1 4)
      add_ocloc_kernel(q4_hc_combine_norm_M${MM}_S${S} SOURCE ${Q4_CL}/q4_hc.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=1 SRC_S=${S} HC_NORM=1)
    endforeach()
    add_ocloc_kernel(q4_hc_combine_norm_M${MM}_Y SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=2 HC_NORM=1)
    add_ocloc_kernel(q4_hc_combine_norm_M${MM}_Y_NN SOURCE ${Q4_CL}/q4_hc.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=2 HC_NORM=0)
    add_ocloc_kernel(q4_hc_combine_norm_M${MM}_X SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=${MM} HC_SRC=3 HC_NORM=1)
    add_ocloc_kernel(q4_hc_up_mix_M${MM}_I SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=${MM} UP_DOWN=336 UP_INJECT=1)
    add_ocloc_kernel(q4_hc_up_mix_M${MM} SOURCE ${Q4_CL}/q4_hc.cl DEFINES ${CTRL_DEFINES} M=${MM} UP_DOWN=320 UP_INJECT=0)
    add_ocloc_kernel(q4_ple_gather_M${MM}_BF16 SOURCE ${Q4_CL}/q4_ple.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=1)
    add_ocloc_kernel(q4_ple_gather_M${MM}_F32 SOURCE ${Q4_CL}/q4_ple.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} PLE_GATHER=1 PLE_EOS=248044 PLE_SCALE_BF16=0)
    add_ocloc_kernel(q4_qsa_M${MM}_T512_W1024 SOURCE ${Q4_CL}/q4_qsa.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TOPB=512 SEL_WG=1024 ${Q4_LIST_DEFINES})
    add_ocloc_kernel(q4_qsa_attn_M${MM}_T32 SOURCE ${Q4_CL}/q4_qsa_attn.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} TGT=32 ${Q4_LIST_DEFINES})
    add_ocloc_kernel(q4_qsa_attn_eager_M${MM} SOURCE ${Q4_CL}/q4_qsa_attn_eager.cl
                     DEFINES ${CTRL_DEFINES} M=${MM} ${Q4_LIST_DEFINES} LISTMAX=2052)
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
    endforeach()
  endforeach()
  add_ocloc_kernel(q4_pf_ple_C4 SOURCE ${Q4_CL}/q4_pf_ple.cl DEFINES ${CTRL_DEFINES} PF_C=4)
  # the prefill's head pass (kernels::qwen4exp::mtp_prefill_variants; 21d's binaries otherwise)
  add_ocloc_kernel(q4_mtp_M2048_SINGLE_PF SOURCE ${Q4_CL}/q4_mtp.cl
                   DEFINES ${CTRL_DEFINES} M=2048 MTP_NORM_SINGLE=1 MTP_PF=1)
  add_ocloc_kernel(q4_mtp_M2048_STREAM_PF SOURCE ${Q4_CL}/q4_mtp.cl
                   DEFINES ${CTRL_DEFINES} M=2048 MTP_NORM_SINGLE=0 MTP_PF=1)
  add_ocloc_kernel(q4_pf_bf16_slab_K2560_N2560 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_pf_linear.cl
                   DEFINES K=2560 N=2560)
endif()
# ==== Spec 21e: Qwen3.8-Flash-Next MTP ==== (end) ====


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
# Not built for Ornith: the int8 KV cache's MTP writer (spec 12b's kv8 binaries are at
# Qwen3.8's heads), an int4-checkpoint lm_head at M > 1 (as for Qwen3.8).
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

# ==== Spec 23c: Muse-Glimmer-30B decode (model::muse_glimmer()) ==== (begin) ====
# Additive only: no existing source is edited, so every binary above keeps its name, its command line and its bytes
# (tools/kernel_cmdlines' diff: additions only - MG0). Muse-only kernels live in src/kernels/muse/ and are named
# `mg_*` with every shape define in the name (src/kernels/muse_kernels.h is the host half); the reused sources run at
# Muse's shapes:
#   gemv.cl         the four int4 layout-0 GEMVs a layer: q||gate||k||v 6656x8704 S2, o_proj 4096x6656 S4,
#                   gate||up 6656x39936 S8 (concatenated, up at column 19968), down 19968x6656 S4 - PROVISIONAL
#                   cells (add_gemv_variant's Spec 23c rows) until 23c Task 7's sweep
#   gemv_i8w.cl / gemv_bf16.cl   lm_head 6656 -> 202048 (int8 the default, spec 9; bf16)
#   prep.cl         prep_res_fold with ZERO_RESID at K 6656, G 26 over a sub-block's S4 slices (o_proj's and down's:
#                   the sandwich's own row and Σ²)
#   argmax.cl       at the vocabulary 202048 (every id real: VOCAB_USED = VOCAB)
# What a Muse decode list binds (runtime/muse/muse_capture.cc): 14 launches a layer under flash, 13 under
# B70_MUSE_ATTN=eager, + mg_embed on device 0 + 4 at the head: 733 / 681 at 52 layers.
option(B70_MUSE "build the Muse-Glimmer-30B decode variants (spec 23c)" ON)
if(B70_MUSE)
  set(MG_CL ${CMAKE_CURRENT_SOURCE_DIR}/muse)
  add_gemv_variant(1 6656 8704 2 0)            # q||gate||k||v int4   PROVISIONAL S2 L0
  add_gemv_variant(1 4096 6656 4 0)            # o_proj int4          PROVISIONAL S4 L0
  add_gemv_variant(1 6656 39936 8 0)           # gate||up int4        PROVISIONAL S8 L0
  add_gemv_variant(1 19968 6656 4 0)           # down int4            PROVISIONAL S4 L0
  add_gemv_i8w_variant(1 6656 202048)          # lm_head int8 (spec 9, the default)
  add_gemv_bf16_variant(1 6656 202048 64 1)    # lm_head bf16
  # prep_res_fold ..._Z: o_proj's / down's S4 slices into the sub-block's own row and its Σ² (26 chunks of 256)
  add_ocloc_kernel(prep_res_fold_M1_K6656_SP4_G26_Z SOURCE ${PREP_CL}
                   DEFINES M=1 K=6656 S_PREV=4 FOLD_G=26 ZERO_RESID=1)
  set(MG_ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=202048 VOCAB_USED=202048)
  add_ocloc_kernel(mg_argmax_stage1_M1_N202048_V202048 SOURCE ${ARGMAX_CL} DEFINES ${MG_ARGMAX_DEFINES} M=1)
  add_ocloc_kernel(mg_argmax_stage2_N202048 SOURCE ${ARGMAX_CL} DEFINES ${MG_ARGMAX_DEFINES})
  # mg_prep.cl: the embedding with its weight-less norm, the centered norm (and the plain-w final norm), the
  # sandwich's post norm + residual add, the attention prep (weight-less q / k norms, q x 3.87, RoPE on sliding
  # layers only, the gate's sigmoid, the ring or the NoPE cache), SiLU x up over the concatenated gate||up.
  add_ocloc_kernel(mg_embed_M1_K6656_G26_V202048 SOURCE ${MG_CL}/mg_prep.cl
                   DEFINES ${CTRL_DEFINES} M=1 K=6656 EMBED_G=26 VOCAB=202048 IMG_TOKEN=200092 VID_TOKEN=200091)
  add_ocloc_kernel(mg_norm_M1_K6656_G26_W26 SOURCE ${MG_CL}/mg_prep.cl
                   DEFINES M=1 K=6656 NORM_G=26 NORM_WGS=26 NORM_EPS=1e-5f)
  add_ocloc_kernel(mg_post_add_M1_K6656_G26 SOURCE ${MG_CL}/mg_prep.cl DEFINES M=1 K=6656 POST_G=26 POST_EPS=1e-8f)
  set(MG_HEADS Q_HEADS=32 KV_HEADS=2 HD=128 QK_SCALE=3.87f)
  add_ocloc_kernel(mg_attn_prep_M1_N8704_S2_Q32KV2_R4096 SOURCE ${MG_CL}/mg_prep.cl
                   DEFINES ${CTRL_DEFINES} M=1 QGKV_N=8704 QGKV_S=2 ${MG_HEADS} SLIDING=1 RING=4096)
  add_ocloc_kernel(mg_attn_prep_M1_N8704_S2_Q32KV2_F SOURCE ${MG_CL}/mg_prep.cl
                   DEFINES ${CTRL_DEFINES} M=1 QGKV_N=8704 QGKV_S=2 ${MG_HEADS} SLIDING=0 RING=0)
  add_ocloc_kernel(mg_silu_mul_M1_I19968_S8 SOURCE ${MG_CL}/mg_prep.cl DEFINES M=1 SILU_I=19968 SILU_S=8)
  # mg_attn.cl (flash, the default: kol_attn.cl's v2 structure at GQA 16 with the gate in the reduce, M <= 16 as a
  # define) and mg_attn_eager.cl (B70_MUSE_ATTN=eager: the reference's bf16 chain, one launch), sliding (the
  # 2048-key window through the 4096-slot ring) and full (NoPE). TGT 64: a window's 34 absolute 64-key blocks.
  set(MG_ATTN_HEADS TGT=64 Q_HEADS=32 KV_HEADS=2)
  add_ocloc_kernel(mg_attn_M1_T64_Q32KV2_W2048_R4096 SOURCE ${MG_CL}/mg_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 ${MG_ATTN_HEADS} WINDOW=2048 RING=4096)
  add_ocloc_kernel(mg_attn_M1_T64_Q32KV2_F SOURCE ${MG_CL}/mg_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 ${MG_ATTN_HEADS} WINDOW=0 RING=0)
  add_ocloc_kernel(mg_attn_eager_M1_T64_Q32KV2_W2048_R4096 SOURCE ${MG_CL}/mg_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=1 ${MG_ATTN_HEADS} WINDOW=2048 RING=4096)
  add_ocloc_kernel(mg_attn_eager_M1_T64_Q32KV2_F SOURCE ${MG_CL}/mg_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=1 ${MG_ATTN_HEADS} WINDOW=0 RING=0)
endif()
# ==== Spec 23c: Muse-Glimmer-30B decode ==== (end) ====

# ==== Spec 23d: Muse-Glimmer-30B prefill ==== (begin) ====
# Additive only, as 23c's block: no existing source is edited, so every binary above (and in prefill/) keeps its name,
# its command line and its bytes (MG0). New Muse-only sources in src/kernels/muse/:
#   mg_pf_attn.cl   the prefill attention over a chunk: the flash build (kol_pf_attn.cl's DPAS walk at GQA 16 - the
#                   2048-key window over the 4096-slot ring in absolute 64-key tiles, causal NoPE over the full cache)
#                   and the eager build (B70_MUSE_ATTN=eager: plain OpenCL, bitwise decode's eager row)
#   mg_pf_prep.cl   mg_pf_embed (mg_embed's chain over the chunk's ids) and mg_pf_attn_gate (decode's output gate)
# Reused at Muse's shapes: mg_prep.cl at M = 2048 (kernels::muse::kPfC: the norms, the sandwich's post-add, the attention
# prep and SiLU x up at S 1), prefill/pf_prep.cl's pf_res_fold _Z at K 6656 (o_proj's and down's own rows and Σ²),
# k2/k2_pf_linear.cl (the int4 layout-0 slabs, a zero-padded tail where N is not whole 1024-column slabs) and the
# shape-free pf_gemm_T0. What a Muse prefill chunk binds: runtime/muse/muse_prefill.cc (derived in muse_sizes.cc).
if(B70_MUSE)
  set(MG_CL ${CMAKE_CURRENT_SOURCE_DIR}/muse)
  add_ocloc_kernel(mg_pf_embed_M2048_K6656_G26_V202048 SOURCE ${MG_CL}/mg_pf_prep.cl
                   DEFINES M=2048 K=6656 EMBED_G=26 VOCAB=202048 IMG_TOKEN=200092 VID_TOKEN=200091)
  add_ocloc_kernel(mg_pf_attn_gate_M2048_N4096 SOURCE ${MG_CL}/mg_pf_prep.cl DEFINES M=2048 GATE_N=4096)
  add_ocloc_kernel(mg_norm_M2048_K6656_G26_W26 SOURCE ${MG_CL}/mg_prep.cl
                   DEFINES M=2048 K=6656 NORM_G=26 NORM_WGS=26 NORM_EPS=1e-5f)
  add_ocloc_kernel(mg_post_add_M2048_K6656_G26 SOURCE ${MG_CL}/mg_prep.cl DEFINES M=2048 K=6656 POST_G=26 POST_EPS=1e-8f)
  set(MG_PF_HEADS Q_HEADS=32 KV_HEADS=2 HD=128 QK_SCALE=3.87f)
  add_ocloc_kernel(mg_attn_prep_M2048_N8704_S1_Q32KV2_R4096 SOURCE ${MG_CL}/mg_prep.cl
                   DEFINES ${CTRL_DEFINES} M=2048 QGKV_N=8704 QGKV_S=1 ${MG_PF_HEADS} SLIDING=1 RING=4096)
  add_ocloc_kernel(mg_attn_prep_M2048_N8704_S1_Q32KV2_F SOURCE ${MG_CL}/mg_prep.cl
                   DEFINES ${CTRL_DEFINES} M=2048 QGKV_N=8704 QGKV_S=1 ${MG_PF_HEADS} SLIDING=0 RING=0)
  add_ocloc_kernel(mg_silu_mul_M2048_I19968_S1 SOURCE ${MG_CL}/mg_prep.cl DEFINES M=2048 SILU_I=19968 SILU_S=1)
  add_ocloc_kernel(pf_res_fold_K6656_SP1_G26_Z SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/pf_prep.cl
                   DEFINES K=6656 S_PREV=1 FOLD_G=26 ZERO_RESID=1)
  # The int4 linears' slabs (layout 0): q||gate||k||v (8 x 1024 + 512), o_proj (6 x 1024 + 512), gate||up (39 x 1024),
  # down (6 x 1024 + 512).
  foreach(row "6656 8704" "4096 6656" "6656 39936" "19968 6656")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_ocloc_kernel(k2_pf_dequant_slab_K${K}_N${N} SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/k2/k2_pf_linear.cl
                     DEFINES K=${K} N=${N})
  endforeach()
  # The attention: sliding (the 2048-key window over the ring) and full (NoPE), flash (256 GRF, as kol_pf_attn) and
  # eager (TGT 64: decode's eager_block target).
  foreach(SL 1 0)
    if(SL)
      set(_n mg_pf_flash_attn_Q32KV2_W2048_R4096)
      set(_w WINDOW=2048 RING=4096)
    else()
      set(_n mg_pf_flash_attn_Q32KV2_F)
      set(_w WINDOW=0 RING=0)
    endif()
    add_ocloc_kernel(${_n} SOURCE ${MG_CL}/mg_pf_attn.cl
                     DEFINES KT=64 RPW=8 HPW=4 Q_HEADS=32 KV_HEADS=2 EAGER=0 ${_w}
                     OPTIONS -cl-intel-256-GRF-per-thread)
    add_ocloc_kernel(${_n}_EAGER SOURCE ${MG_CL}/mg_pf_attn.cl DEFINES TGT=64 Q_HEADS=32 KV_HEADS=2 EAGER=1 ${_w})
  endforeach()
endif()
# ==== Spec 23d: Muse-Glimmer-30B prefill ==== (end) ====

# ==== Spec 23f: Muse-Glimmer-30B's verify lists and its DFlash drafter ==== (begin) ====
# Plan: docs/superpowers/plans/2026-10-15-spec23f-muse-dflash.md. Additive only, as 23c's / 23d's blocks: no existing
# source is edited (MG0 / D0), so every binary above keeps its name, its command line and its bytes; nothing here is
# bound without `--spec dflash` / `--spec lookup` on Muse-Glimmer (runtime::muse::MuseEngine's verify lists).
#
# B70_MUSE_VERIFY_M16 - the verify lists: 23c's decode walk at M = 1..16 rows (spec 23 decision 10: the anchor + up to
# 15 drafts) with the taps of every row, reused sources at new M:
#   gemv.cl         the four int4 layout-0 GEMVs at their PROVISIONAL S (add_gemv_variant applies the cells' defines at
#                   every M: one row's chain is M-independent - D3)
#   gemv_i8w.cl / gemv_bf16.cl   lm_head 6656 -> 202048
#   prep.cl         prep_res_fold _Z at K 6656, G 26 over S4 slices
#   mg_prep.cl      mg_norm, mg_post_add, mg_attn_prep (both kinds), mg_silu_mul
#   mg_attn.cl / mg_attn_eager.cl   both kinds (M <= 16 is 23c's define)
#   mg_pf_prep.cl   mg_pf_embed - the verify's ids [anchor, drafts] from a buffer (mg_embed reads Control::cur_token,
#                   8 slots), mg_embed's chain line for line, at M = 1..16
# and one new source, src/kernels/muse/mg_argmax.cl: argmax.cl's two stages at M <= 16 (argmax.cl caps M at 8 and
# advances Control::pos) - stage 1 line for line, stage 2 writing every row's id to its own buffer (the host commits).
option(B70_MUSE_VERIFY_M16 "build Muse-Glimmer's verify lists at M = 1..16 (spec 23f)" ON)
if(B70_MUSE AND B70_MUSE_VERIFY_M16)
  set(MG_CL ${CMAKE_CURRENT_SOURCE_DIR}/muse)
  set(MG_VARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=202048 VOCAB_USED=202048)
  foreach(MM RANGE 1 16)
    if(MM GREATER 1)
      add_gemv_variant(${MM} 6656 8704 2 0)
      add_gemv_variant(${MM} 4096 6656 4 0)
      add_gemv_variant(${MM} 6656 39936 8 0)
      add_gemv_variant(${MM} 19968 6656 4 0)
      add_gemv_i8w_variant(${MM} 6656 202048)
      add_gemv_bf16_variant(${MM} 6656 202048 64 1)
      add_ocloc_kernel(prep_res_fold_M${MM}_K6656_SP4_G26_Z SOURCE ${PREP_CL}
                       DEFINES M=${MM} K=6656 S_PREV=4 FOLD_G=26 ZERO_RESID=1)
      add_ocloc_kernel(mg_norm_M${MM}_K6656_G26_W26 SOURCE ${MG_CL}/mg_prep.cl
                       DEFINES M=${MM} K=6656 NORM_G=26 NORM_WGS=26 NORM_EPS=1e-5f)
      add_ocloc_kernel(mg_post_add_M${MM}_K6656_G26 SOURCE ${MG_CL}/mg_prep.cl
                       DEFINES M=${MM} K=6656 POST_G=26 POST_EPS=1e-8f)
      add_ocloc_kernel(mg_attn_prep_M${MM}_N8704_S2_Q32KV2_R4096 SOURCE ${MG_CL}/mg_prep.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} QGKV_N=8704 QGKV_S=2 Q_HEADS=32 KV_HEADS=2 HD=128
                               QK_SCALE=3.87f SLIDING=1 RING=4096)
      add_ocloc_kernel(mg_attn_prep_M${MM}_N8704_S2_Q32KV2_F SOURCE ${MG_CL}/mg_prep.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} QGKV_N=8704 QGKV_S=2 Q_HEADS=32 KV_HEADS=2 HD=128
                               QK_SCALE=3.87f SLIDING=0 RING=0)
      add_ocloc_kernel(mg_silu_mul_M${MM}_I19968_S8 SOURCE ${MG_CL}/mg_prep.cl DEFINES M=${MM} SILU_I=19968 SILU_S=8)
      add_ocloc_kernel(mg_attn_M${MM}_T64_Q32KV2_W2048_R4096 SOURCE ${MG_CL}/mg_attn.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=64 Q_HEADS=32 KV_HEADS=2 WINDOW=2048 RING=4096)
      add_ocloc_kernel(mg_attn_M${MM}_T64_Q32KV2_F SOURCE ${MG_CL}/mg_attn.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=64 Q_HEADS=32 KV_HEADS=2 WINDOW=0 RING=0)
      add_ocloc_kernel(mg_attn_eager_M${MM}_T64_Q32KV2_W2048_R4096 SOURCE ${MG_CL}/mg_attn_eager.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=64 Q_HEADS=32 KV_HEADS=2 WINDOW=2048 RING=4096)
      add_ocloc_kernel(mg_attn_eager_M${MM}_T64_Q32KV2_F SOURCE ${MG_CL}/mg_attn_eager.cl
                       DEFINES ${CTRL_DEFINES} M=${MM} TGT=64 Q_HEADS=32 KV_HEADS=2 WINDOW=0 RING=0)
    endif()
    add_ocloc_kernel(mg_pf_embed_M${MM}_K6656_G26_V202048 SOURCE ${MG_CL}/mg_pf_prep.cl
                     DEFINES M=${MM} K=6656 EMBED_G=26 VOCAB=202048 IMG_TOKEN=200092 VID_TOKEN=200091)
    add_ocloc_kernel(mg_vargmax_stage1_M${MM}_N202048 SOURCE ${MG_CL}/mg_argmax.cl
                     DEFINES ${MG_VARGMAX_DEFINES} M=${MM})
  endforeach()
  add_ocloc_kernel(mg_vargmax_stage2_N202048 SOURCE ${MG_CL}/mg_argmax.cl DEFINES ${MG_VARGMAX_DEFINES} M=16)
endif()

# B70_MUSE_DFLASH - the drafter (meta-models/Muse-Glimmer-30B-assistant: 5 sliding layers, hidden 6656, 32 q / 8 kv x
# 128, FF 19968, block 16, window |q - k| <= 2048, eps 1e-5; kernels::dflash::kMuse, src/kernels/dflash_kernels.h):
#   df_prep.cl   at its defines - df_embed (the target's RAW rows: in_scale 1), df_add, df_silu_mul, df_qkv_prep,
#                df_ctx_store (ring slot pos % 2112); no convolutions (DFlash v1)
#   df_attn.cl   the block's 16 rows over the ring + the block: WIN 2049 is the window |q - k| <= 2048 (df_attn's rule is
#                <= WIN - 1), RING 2112 = 33 blocks of PB 64 (RING >= WIN; PB 64 keeps the local memory of 64 q pairs
#                within 64 KB)
#   df_topk.cl   the 16 best of the target head's 202048 logits for each of the 15 draft rows
#   mg_dflash.cl mg_df_norm (the assistant's two-rounding RMSNorm) and mg_df_select (the per-row pick at 16 rows, the
#                verify's ids buffer)
# and the GEMVs at the block's 16 rows in the three formats (gemv_i8w / gemv_bf16 / gemv.cl int4 g64 layout 1 S 1:
# q||k||v 6656 x 6144, o 4096 x 6656, gate||up 6656 x 39936, down 19968 x 6656, fc 33280 x 6656, the context's k||v
# 6656 x 2048 - layout 1 is n-tile-major, so the k||v rows of q||k||v are a sub-range), the target's head at the 15
# draft rows (int8 / bf16).
option(B70_MUSE_DFLASH "build Muse-Glimmer's DFlash drafter variants (spec 23f)" ON)
if(B70_MUSE AND B70_MUSE_VERIFY_M16 AND B70_MUSE_DFLASH)
  set(MG_CL ${CMAKE_CURRENT_SOURCE_DIR}/muse)
  set(DF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dflash)
  set(MDCTL_DEFINES DCTL_K=0 DCTL_PPOS=1 DCTL_PN=2 DCTL_TEMP=3 DCTL_SEED_LO=4 DCTL_SEED_HI=5)
  add_ocloc_kernel(df_prep_M16_H6656_I19968_Q32KV8_D128_R2112_V202048_E5 SOURCE ${DF_DIR}/df_prep.cl
                   DEFINES ${CTRL_DEFINES} ${MDCTL_DEFINES} M=16 H=6656 WG=256 EPS=1e-05f VOCAB=202048
                           INTER=19968 NH=32 NKV=8 HD=128 RING=2112)
  add_ocloc_kernel(df_attn_M16_Q32KV8_D128_R2112_W2049_P64 SOURCE ${DF_DIR}/df_attn.cl
                   DEFINES ${CTRL_DEFINES} ${MDCTL_DEFINES} M=16 NH=32 NKV=8 HD=128 RING=2112 WIN=2049 PB=64 WGA=128
                           CAUSAL=0)
  add_ocloc_kernel(df_topk_R15_N202048_U202048 SOURCE ${DF_DIR}/df_topk.cl DEFINES R=15 N=202048 USED=202048 TOPK=16
                   CH=4096 COMPACT=0)
  add_ocloc_kernel(mg_dflash_M16_H6656_K16_E5 SOURCE ${MG_CL}/mg_dflash.cl
                   DEFINES ${CTRL_DEFINES} ${MDCTL_DEFINES} H=6656 WG=256 EPS=1e-05f SELECT=1 MAXR=16 TOPK=16)
  foreach(row "6656 6144" "4096 6656" "6656 39936" "19968 6656" "33280 6656" "6656 2048")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    add_gemv_i8w_variant(16 ${K} ${N})
    add_gemv_bf16_variant(16 ${K} ${N} 64 1)
    add_gemv_variant(16 ${K} ${N} 1 1)
  endforeach()
  # the target's head at the 15 draft rows (int8 / bf16) is the verify block's M = 15 lm_head
endif()
# ==== Spec 23f ==== (end) ====

# ==== Spec 24c: LFM2.5-24B-A2B decode (model::lfm2moe()) ==== (begin) ====
# Additive only: no existing source is edited, so every binary above keeps its name, its command line and its bytes
# (tools/kernel_cmdlines' diff: additions only - F0). LFM2-only kernels live in src/kernels/lfm2moe/ and are named
# `lfm_*` with every shape define in the name (src/kernels/lfm2moe_kernels.h is the host half). The reused sources at
# LFM2's shapes - also the engine's binaries:
#   gemv.cl        in_proj 2048x6144, out_proj / o_proj 2048x2048, q||k||v 2048x3072, dense gate||up 2048x23552,
#                  dense down 11776x2048 - every S in {1, 2, 4, 8, 16} dividing K / 64 (184 k-groups at 11776: S <= 8)
#                  and both layouts: P0's sweep (probe_lfm_step --sweep-s) times them all; the engine binds the
#                  descriptor's {S, layout} (model::Lfm2MoeDesc::linears, PROVISIONAL)
#   gemv_i8w.cl / gemv_bf16.cl   the tied head 2048 -> 128000 (int8 the default, spec 9; bf16 over the embedding tiles)
#   argmax.cl      at VOCAB 128000, VOCAB_USED 125017 (the tokenizer's ids; rows 125017.. never win)
# P0's stand-ins (plan 24c Task 1; no LFM2 kernel bound): moe.cl at E64 T4 D2048 I1536 (5 slots: the shared one is a
# 65th block), Kolibri's attention at Q32KV8 (head_dim 128), kol_norm_finish at K 2048 over 8 work-groups, the
# router as gemv_bf16 at N 64, and Ornith's prep_res_fold_M1_K2048_SP0_G20 (built here only without B70_ORNITH).
option(B70_LFM2MOE "build the LFM2.5-24B-A2B decode variants (spec 24c)" ON)
if(B70_LFM2MOE)
  set(LFM_CL ${CMAKE_CURRENT_SOURCE_DIR}/lfm2moe)
  foreach(L 0 1)
    foreach(S 1 2 4 8 16)
      add_gemv_variant(1 2048 6144 ${S} ${L})     # in_proj (B | C | x)
      add_gemv_variant(1 2048 2048 ${S} ${L})     # conv out_proj, attention out_proj
      add_gemv_variant(1 2048 3072 ${S} ${L})     # q||k||v
      add_gemv_variant(1 2048 23552 ${S} ${L})    # dense gate||up (interleave16)
      if(NOT S EQUAL 16)
        add_gemv_variant(1 11776 2048 ${S} ${L})  # dense down
      endif()
    endforeach()
  endforeach()
  add_gemv_i8w_variant(1 2048 128000)             # the int8 head (spec 9), built from the embedding
  add_gemv_bf16_variant(1 2048 128000 64 1)       # the bf16 head over the embedding tiles (--lm-head bf16)
  set(LFM_ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=128000 VOCAB_USED=125017)
  add_ocloc_kernel(lfm_argmax_stage1_M1_V128000_U125017 SOURCE ${ARGMAX_CL} DEFINES ${LFM_ARGMAX_DEFINES} M=1)
  add_ocloc_kernel(lfm_argmax_stage2_V128000 SOURCE ${ARGMAX_CL} DEFINES ${LFM_ARGMAX_DEFINES})
  # --- P0's stand-ins (tools/probe/probe_lfm_step.cc only) ---
  add_gemv_bf16_variant(1 2048 64 16 16)          # the router at gemv_bf16's tiny tiling
  add_ocloc_kernel(moe_M1_E64_T4_D2048_I1536 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/moe.cl
                   DEFINES UP_KS=4 DN_KS=2 M=1 EXPERTS=64 TOP_K=4 HIDDEN=2048 INTER=1536 ROUTER_N=80)
  add_ocloc_kernel(kol_attn_prep_M1_N6144_S1_Q32KV8_F SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_prep.cl
                   DEFINES ${CTRL_DEFINES} M=1 QKV_N=6144 QKV_S=1 Q_HEADS=32 KV_HEADS=8 HD=128 SLIDING=0 RING=0)
  add_ocloc_kernel(kol_attn_M1_T${ATTN_V2_TGT}_Q32KV8_F SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=32 KV_HEADS=8 WINDOW=0 RING=0)
  add_ocloc_kernel(kol_norm_M1_K2048_G20_W8 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_prep.cl
                   DEFINES M=1 K=2048 NORM_G=20 NORM_WGS=8)
  if(NOT B70_ORNITH)
    add_prep_res_fold(1 2048 0 20)
  endif()
  # --- the family kernels (src/kernels/lfm2moe/, kernels::lfm2moe's names; Task 3) ---
  # lfm_prep.cl: the one-launch norm (fold + Σ + the two-rounding x; _NF only normalises) at 8 work-groups
  # (kNormWgs, PROVISIONAL), over the split-K the descriptor gives out_proj / o_proj / the dense down (S 1 today:
  # a re-pick adds its S here - lfm2moe_variant_names_test names the missing binary); SiLU x up at 11776; the
  # attention prep at head_dim 64 (eps 1e-5, RoPE on every attention layer); the gather from the embedding tiles.
  set(LFM_NORM_DEFINES M=1 K=2048 NORM_WGS=8 NORM_EPS=1e-5f)
  add_ocloc_kernel(lfm_res_norm_M1_K2048_W8_NF SOURCE ${LFM_CL}/lfm_prep.cl DEFINES ${LFM_NORM_DEFINES} NORM_S=0)
  foreach(S 1)
    add_ocloc_kernel(lfm_res_norm_M1_K2048_W8_S${S} SOURCE ${LFM_CL}/lfm_prep.cl DEFINES ${LFM_NORM_DEFINES} NORM_S=${S})
  endforeach()
  add_ocloc_kernel(lfm_silu_mul_M1_I11776_S1 SOURCE ${LFM_CL}/lfm_prep.cl DEFINES M=1 SILU_I=11776 SILU_S=1)
  add_ocloc_kernel(lfm_attn_prep_M1_S1_Q32KV8 SOURCE ${LFM_CL}/lfm_prep.cl
                   DEFINES ${CTRL_DEFINES} M=1 QKV_S=1 Q_HEADS=32 KV_HEADS=8 HD=64 PREP_EPS=1e-5f)
  add_ocloc_kernel(lfm_embed_gather_M1_K2048_V128000 SOURCE ${LFM_CL}/lfm_prep.cl
                   DEFINES ${CTRL_DEFINES} M=1 K=2048 EMBED_V=128000)
  # lfm_conv.cl: the short conv over the Bx ring, after in_proj's S-slice GEMV.
  add_ocloc_kernel(lfm_conv_M1_K2048_S1 SOURCE ${LFM_CL}/lfm_conv.cl DEFINES ${CTRL_DEFINES} M=1 HIDDEN=2048 CONV_S=1)
  # Test-only (lfm2moe_kernels_test, Review Focus 1): M = 4 rows a launch - the M-row form 24d's chunk walk and a
  # future verify list use; no decode list binds it.
  add_ocloc_kernel(lfm_conv_M4_K2048_S1 SOURCE ${LFM_CL}/lfm_conv.cl DEFINES ${CTRL_DEFINES} M=4 HIDDEN=2048 CONV_S=1)
  # lfm_moe.cl: the router (64 rows, lfm_router's order), the sigmoid-bf16 route, the 4-slot MoE with the fold.
  # lfm_expert.h (the one expert address) sits beside it: -I the family's directory.
  add_ocloc_kernel(lfm_router_M1_K2048_E64 SOURCE ${LFM_CL}/lfm_moe.cl DEFINES M=1 ROUTER_E=64 ROUTER_K=2048)
  add_ocloc_kernel(lfm_route_M1_E64_T4 SOURCE ${LFM_CL}/lfm_moe.cl DEFINES M=1 ROUTE_E=64)
  add_ocloc_kernel(lfm_moe_M1_E64_T4_D2048_I1536 SOURCE ${LFM_CL}/lfm_moe.cl
                   DEFINES M=1 MOE_E=64 MOE_K=4 HIDDEN=2048 INTER=1536 UP_KS=4 DN_KS=2
                   DEPENDS ${LFM_CL}/lfm_expert.h OPTIONS -I${LFM_CL})
  # lfm_attn.cl (flash, the default) and lfm_attn_eager.cl (B70_LFM_ATTN=eager) at head_dim 64, GQA 4, TGT 32.
  set(LFM_ATTN_DEFINES ${CTRL_DEFINES} M=1 TGT=32 Q_HEADS=32 KV_HEADS=8)
  add_ocloc_kernel(lfm_attn_M1_T32_Q32KV8 SOURCE ${LFM_CL}/lfm_attn.cl DEFINES ${LFM_ATTN_DEFINES})
  add_ocloc_kernel(lfm_attn_eager_M1_T32_Q32KV8 SOURCE ${LFM_CL}/lfm_attn_eager.cl DEFINES ${LFM_ATTN_DEFINES})
  # --- the fused arms (B70_LFM_FUSE, spec 24 §4.5 / decision 6; Task 5) - each bitwise the as-built pair ---
  # router: the 64 router rows inside a MoE layer's ffn_norm launch (4 more work-groups), at the mixer's split-K;
  # route: the route recomputed in every gate||up work-group; conv: in_proj's GEMV (split-K 1, the conv-tile columns,
  # in_proj's layout - 24b's descriptor: 1) with the conv as its epilogue.
  foreach(S 1)
    add_ocloc_kernel(lfm_res_norm_M1_K2048_W8_S${S}_R SOURCE ${LFM_CL}/lfm_prep.cl
                     DEFINES ${LFM_NORM_DEFINES} NORM_S=${S} ROUTER_FUSED=1 ROUTER_E=64)
  endforeach()
  add_ocloc_kernel(lfm_moe_M1_E64_T4_D2048_I1536_R SOURCE ${LFM_CL}/lfm_moe.cl
                   DEFINES M=1 MOE_E=64 MOE_K=4 HIDDEN=2048 INTER=1536 UP_KS=4 DN_KS=2 ROUTE_FUSED=1
                   DEPENDS ${LFM_CL}/lfm_expert.h OPTIONS -I${LFM_CL})
  add_ocloc_kernel(lfm_conv_in_M1_K2048_L1 SOURCE ${LFM_CL}/lfm_conv.cl DEFINES ${CTRL_DEFINES} M=1 HIDDEN=2048 CONV_IN_L=1)
endif()
# ==== Spec 24c: LFM2.5-24B-A2B decode ==== (end) ====

# ==== Spec 24d: LFM2.5-24B-A2B prefill ==== (begin) ====
# Additive only, as 24c's block: no existing source is edited, so every binary above (and in prefill/) keeps its name,
# its command line and its bytes. New LFM2-only sources in src/kernels/lfm2moe/:
#   lfm_pf_conv.cl    the short conv over a chunk (lfm_pf_conv_seed: the ring's two slots before the chunk into a flat
#                     seed, a PRIOR launch; lfm_pf_conv: every row recomputes its predecessors' Bx, the chunk's last two
#                     Bx into the ring) - plain and conv-tile in_proj columns (_T: the arm conv's load)
#   lfm_pf_moe.cl     the 64-expert sort (uchar ids, one expert a lane, no shared block), gather, the dequant of a
#                     weight batch (lfm_expert.h's address), the rank-order combine with the residual fold
#   lfm_pf_attn.cl    flash attention at head_dim 64 over the growing cache (kol_pf_attn.cl's full path), and its eager
#                     twin (B70_LFM_ATTN=eager); 256 GRF as kol_pf_attn / pf_flash_attn
#   lfm_pf_prep.cl    the chunk's ids out of the tiled embedding
# Reused at LFM2's shapes: 24c's lfm_prep.cl / lfm_moe.cl at M = 2048 (kernels::lfm2moe::kPfC: the norm _NF / _S1,
# the router, the route, the attention prep at S 1 - decode's chains row for row), prefill/pf_dequant_slab.cl at each
# int4 linear's {K, N, layout} as model::Lfm2MoeDesc::linears gives it (24b's PROVISIONAL picks: layout 1 for in_proj,
# out_proj / o_proj and the dense down, 0 for q||k||v and the dense gate||up - a re-pick adds its line here and
# lfm2moe_pf_variant_names_test names the missing binary), prefill/pf_moe_gemm.cl (gate||up 2048 x 3072 SiLU, down
# 1536 x 2048) and the shape-free pf_gemm_T0 / pf_gemm_T0_SILU (prefill/CMakeLists.txt). What a chunk binds:
# runtime/lfm2moe/lfm2moe_prefill.cc (derived in runtime/lfm2moe/lfm2moe_sizes.cc: 1259 launches at 40 layers).
if(B70_LFM2MOE)
  set(LFM_CL ${CMAKE_CURRENT_SOURCE_DIR}/lfm2moe)
  set(LFM_PF_REUSE ${CMAKE_CURRENT_SOURCE_DIR}/prefill)
  add_ocloc_kernel(lfm_pf_embed_gather_K2048_V128000 SOURCE ${LFM_CL}/lfm_pf_prep.cl DEFINES K=2048 EMBED_V=128000)
  foreach(T 0 1)
    set(_n lfm_pf_conv_K2048_C2048)
    if(T)
      string(APPEND _n _T)
    endif()
    add_ocloc_kernel(${_n} SOURCE ${LFM_CL}/lfm_pf_conv.cl DEFINES ${CTRL_DEFINES} HIDDEN=2048 PF_C=2048 CONV_TILES=${T})
  endforeach()
  add_ocloc_kernel(lfm_pf_moe_E64_T4_D2048_I1536_TM32 SOURCE ${LFM_CL}/lfm_pf_moe.cl
                   DEFINES PF_E=64 PF_K=4 PF_WG=64 PF_D=2048 PF_INTER=1536 TM=32 KC=2048
                   DEPENDS ${LFM_CL}/lfm_expert.h OPTIONS -I${LFM_CL})
  foreach(EG 0 1)
    set(_n lfm_pf_attn_Q32KV8_HPW4_RPW16_KT64)
    if(EG)
      string(APPEND _n _EAGER)
    endif()
    add_ocloc_kernel(${_n} SOURCE ${LFM_CL}/lfm_pf_attn.cl
                     DEFINES KT=64 RPW=16 HPW=4 Q_HEADS=32 KV_HEADS=8 EXP2=0 EAGER=${EG}
                     OPTIONS -cl-intel-256-GRF-per-thread)
  endforeach()
  # 24c's per-row kernels at M = kPfC (2048): every prefill linear is one unsplit GEMM, so the folds are S 1.
  set(LFM_PF_NORM_DEFINES M=2048 K=2048 NORM_WGS=8 NORM_EPS=1e-5f)
  add_ocloc_kernel(lfm_res_norm_M2048_K2048_W8_NF SOURCE ${LFM_CL}/lfm_prep.cl DEFINES ${LFM_PF_NORM_DEFINES} NORM_S=0)
  add_ocloc_kernel(lfm_res_norm_M2048_K2048_W8_S1 SOURCE ${LFM_CL}/lfm_prep.cl DEFINES ${LFM_PF_NORM_DEFINES} NORM_S=1)
  add_ocloc_kernel(lfm_router_M2048_K2048_E64 SOURCE ${LFM_CL}/lfm_moe.cl DEFINES M=2048 ROUTER_E=64 ROUTER_K=2048)
  add_ocloc_kernel(lfm_route_M2048_E64_T4 SOURCE ${LFM_CL}/lfm_moe.cl DEFINES M=2048 ROUTE_E=64)
  add_ocloc_kernel(lfm_attn_prep_M2048_S1_Q32KV8 SOURCE ${LFM_CL}/lfm_prep.cl
                   DEFINES ${CTRL_DEFINES} M=2048 QKV_S=1 Q_HEADS=32 KV_HEADS=8 HD=64 PREP_EPS=1e-5f)
  # The reused prefill sources at LFM2's shapes (the descriptor's layouts).
  foreach(row "2048 6144 1" "2048 2048 1" "2048 3072 0" "2048 23552 0" "11776 2048 1")
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    list(GET f 2 L)
    add_ocloc_kernel(pf_dequant_slab_K${K}_N${N}_L${L} SOURCE ${LFM_PF_REUSE}/pf_dequant_slab.cl
                     DEFINES K=${K} N=${N} LAYOUT=${L})
  endforeach()
  add_ocloc_kernel(pf_moe_gemm_K2048_N3072_SILU SOURCE ${LFM_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=2048 N=3072 TM=32 SILU_EPI=1 OPTIONS -cl-intel-256-GRF-per-thread)
  add_ocloc_kernel(pf_moe_gemm_K1536_N2048 SOURCE ${LFM_PF_REUSE}/pf_moe_gemm.cl
                   DEFINES K=1536 N=2048 TM=32 OPTIONS -cl-intel-256-GRF-per-thread)
endif()
# ==== Spec 24d: LFM2.5-24B-A2B prefill ==== (end) ====

# ==== Spec 25c: Nemotron 3.5 Bolt decode ==== (begin) ====
# Additive only: no existing source is edited, so every binary above keeps its name, its command line and its bytes
# (tools/kernel_cmdlines' diff: additions only - B0). Bolt-only kernels live in src/kernels/bolt/ and are named `nb_*`
# with every shape define in the name (src/kernels/bolt_kernels.h is the host half). The reused sources at Bolt's shapes -
# also the engine's binaries:
#   gemv.cl        in_proj 4096x18560, out_proj 8192x4096, q||k||v 4096x4608, o_proj 4096x4096, fc1||shared_up 4096x6400,
#                  fc2 1024x4096, shared_down 5376x4096 - every S in {1, 2, 4, 8, 16} dividing K / 64 (84 k-groups at 5376:
#                  S <= 4) and both layouts: P0's sweep (probe_bolt_step --sweep-s) times them all; the engine binds the
#                  descriptor's {S, layout} (model::BoltDesc::linears, PROVISIONAL)
#   gemv_i8w.cl / gemv_bf16.cl   the untied head 4096 -> 131072 (int8 the default, Decision 18; bf16 over its own tiles)
#   argmax.cl      at VOCAB 131072 = VOCAB_USED (every row a token)
#   mg_attn.cl / mg_attn_eager.cl   Muse-Glimmer's full (NoPE) decode attention at Q 32 KV 2 head_dim 128 - Bolt's exact
#                  shape (Decision 15: bound by name with a ones gate; built here only when the Muse block is off)
# P0's stand-ins (plan 25c Task 1; tools/probe/probe_bolt_step.cc only): the routers as gemv_bf16 at each E, moe.cl at
# E64 T8 D1024 I2048 (gated; a pool of 64 blocks + the shared slot), lfm_prep.cl's one-launch norm at K 4096, the
# two-launch norm (prep_res_fold G16 + kol_norm_finish W16) for --pair, nb_probe.cl's state read-modify-write; the SSM
# step is gdn_step_M1 (Qwen3.8's shape, built above) and the light slots noop.
option(B70_BOLT "build the Nemotron 3.5 Bolt decode variants (spec 25c)" ON)
if(B70_BOLT)
  set(NB_CL ${CMAKE_CURRENT_SOURCE_DIR}/bolt)
  foreach(L 0 1)
    foreach(S 1 2 4 8 16)
      add_gemv_variant(1 4096 18560 ${S} ${L})    # in_proj (z | xBC | dt)
      add_gemv_variant(1 8192 4096 ${S} ${L})     # out_proj
      add_gemv_variant(1 4096 4608 ${S} ${L})     # q||k||v
      add_gemv_variant(1 4096 4096 ${S} ${L})     # o_proj
      add_gemv_variant(1 4096 6400 ${S} ${L})     # fc1_latent || shared_experts.up
      add_gemv_variant(1 1024 4096 ${S} ${L})     # fc2_latent
      if(S LESS_EQUAL 4)
        add_gemv_variant(1 5376 4096 ${S} ${L})   # shared_experts.down (84 k-groups)
      endif()
    endforeach()
  endforeach()
  add_gemv_i8w_variant(1 4096 131072)             # the int8 head (spec 9, Decision 18), built from lm_head
  add_gemv_bf16_variant(1 4096 131072 64 1)       # the bf16 head over lm_head's own tiles (--lm-head bf16)
  set(NB_ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=131072 VOCAB_USED=131072)
  add_ocloc_kernel(nb_argmax_stage1_M1_V131072 SOURCE ${ARGMAX_CL} DEFINES ${NB_ARGMAX_DEFINES} M=1)
  add_ocloc_kernel(nb_argmax_stage2_V131072 SOURCE ${ARGMAX_CL} DEFINES ${NB_ARGMAX_DEFINES})
  if(NOT B70_MUSE)
    set(NB_MG_ATTN_HEADS TGT=64 Q_HEADS=32 KV_HEADS=2)
    add_ocloc_kernel(mg_attn_M1_T64_Q32KV2_F SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/muse/mg_attn.cl
                     DEFINES ${CTRL_DEFINES} M=1 ${NB_MG_ATTN_HEADS} WINDOW=0 RING=0)
    add_ocloc_kernel(mg_attn_eager_M1_T64_Q32KV2_F SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/muse/mg_attn_eager.cl
                     DEFINES ${CTRL_DEFINES} M=1 ${NB_MG_ATTN_HEADS} WINDOW=0 RING=0)
  endif()
  # --- P0's stand-ins (tools/probe/probe_bolt_step.cc only) ---
  foreach(E 240 304 368 432 496)
    add_gemv_bf16_variant(1 4096 ${E} 16 16)      # the routers at gemv_bf16's tiny tiling (fp32 logits)
  endforeach()
  add_ocloc_kernel(moe_M1_E64_T8_D1024_I2048 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/moe.cl
                   DEFINES UP_KS=4 DN_KS=2 M=1 EXPERTS=64 TOP_K=8 HIDDEN=1024 INTER=2048 ROUTER_N=80)
  add_ocloc_kernel(lfm_res_norm_M1_K4096_W8_S1 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/lfm2moe/lfm_prep.cl
                   DEFINES M=1 K=4096 NORM_WGS=8 NORM_EPS=1e-5f NORM_S=1)
  add_prep_res_fold(1 4096 0 16)
  add_ocloc_kernel(kol_norm_M1_K4096_G16_W16 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kolibri/kol_prep.cl
                   DEFINES M=1 K=4096 NORM_G=16 NORM_WGS=16)
  foreach(B 0 1)
    if(B)
      set(_nb_rmw nb_state_rmw_B16)
    else()
      set(_nb_rmw nb_state_rmw_F32)
    endif()
    add_ocloc_kernel(${_nb_rmw} SOURCE ${NB_CL}/nb_probe.cl DEFINES STATE_BF16=${B})
  endforeach()
endif()
# ==== Spec 25c: Nemotron 3.5 Bolt decode ==== (end) ====

# Spec 16b (pipeline parallel, `--pipeline-handoff peer`): the peer-write hand-off between
# the two stages' decode lists - pp_send (device 0's last launch) and pp_recv (device 1's
# first). No defines: the word counts are kernel arguments, so one binary serves every model.
add_ocloc_kernel(pp_handoff SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pp_handoff.cl)

add_subdirectory(prefill)
