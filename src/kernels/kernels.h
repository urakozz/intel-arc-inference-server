#pragma once
#include <string>
#include <vector>

#include "kernels/linear_kind.h"
#include "kernels/xe_cores.h"
#include "kernels/shape_suffix.h"

#ifndef B70_KERNEL_DIR
#error "B70_KERNEL_DIR must be defined (use b70_target_kernel_dir in CMake)"
#endif

namespace kernels {
// Absolute path of a compiled device binary by variant name.
inline std::string path(const std::string& variant) {
  return std::string(B70_KERNEL_DIR) + "/" + variant + ".bin";
}
inline std::string gemv_variant(unsigned M, unsigned K, unsigned N, unsigned S, unsigned L) {
  return "gemv_M" + std::to_string(M) + "_K" + std::to_string(K) + "_N" + std::to_string(N) +
         "_S" + std::to_string(S) + "_L" + std::to_string(L);
}
// **`gemv_bf16`'s tiling - two compile-time knobs, chosen per shape.** They are
// the kernel's `COLS_PER_WG` and `KSPLIT` (src/kernels/gemv_bf16.cl); together
// they fix the compiled variant's name, the work-group size
// (`cols * ksplit` lanes) and the grid (`N / cols` work-groups). `{64, 1}` is
// what every bf16 GEMV was compiled at until spec 1.5's lever L2.
//
// **The choice is a function of `N` and lives here** so that the two callers
// that must agree - the runtime (src/runtime/capture.cc) and the table check
// (tests/kernels/kernel_table_test.cc) - cannot drift apart. It is not a
// project-wide invariant: tools/probe/probe_gemv.cc builds its own tiling and
// bypasses this function. That is harmless because the probe times `lm_head`
// only, at the default `{64, 1}`, which is what this function would hand it -
// but a probe row for any other shape must be routed through here first.
//
// `lm_head` (N = 248320) keeps `{64, 1}`: 3880 work-groups at 98.5% of measured
// bandwidth in situ, nothing to win, and measured unmoved in situ across the
// change - **4378.555 -> 4379.089 µs/launch, +0.012%** (before at `43bb720`,
// after at the committed `a1e2d3a`; docs/12 `gemv_bf16` carries both rollups).
//
// Its **binary is not** byte-identical, though its ocloc command line is:
// restructuring the K loop for `KSPLIT` changed IGC's output at `KSPLIT == 1`
// too. What covers lm_head's numerics is therefore not the defines but the
// acceptance that ran after the change - the golden gate 96/96 element-exact
// (every token passes through lm_head's logits and argmax) and
// replay_determinism_test green. `-cl-fp32-correctly-rounded-divide-sqrt` does
// not pin `FP_CONTRACT`, so "same defines ⇒ same numerics" was never airtight
// for any kernel here; the gate is the arbiter and always was.
// `a‖b` (N = 128) takes `{16, 16}` - 8 work-groups of 16 K-slice subgroups, so
// **128 hardware threads over the launch instead of 8**, and both halves of the
// pair earn their place: the `ksplit` is what creates the threads and the
// `cols` is what spreads them over eight Xe-cores rather than one.
//
// The width is measured, not chosen (docs/12 `gemv_bf16`, in situ, three points):
// ksplit 1 / 4 / 16 read **48.774 / 13.115 / 5.340 µs** per launch. 16 stops
// 2.4x above the shape's 2.22 µs traffic floor, so at most 0.15 ms/token is
// left in it and a 32-way split cannot repay its own gate run.
//
// **`ksplit > 1` changes the summation order** (the K slices are merged by a
// fixed SLM tree - gemv_bf16.cl states the order), so a variant with it is held
// to the reference tolerance and to the golden gate, not to bit-identity.
inline constexpr unsigned kGemvBf16Cols = 64;      // 4 subgroups per work-group
inline constexpr unsigned kGemvBf16TinyCols = 16;  // 1 column tile per work-group
// kBuiltXeCores (kernels/xe_cores.h): a shape whose default grid already puts at least
// one work-group on every core is not the tiny-N case.
struct GemvBf16Tiling {
  unsigned cols;    // COLS_PER_WG: output columns per work-group
  unsigned ksplit;  // KSPLIT: subgroups splitting one tile's K, merged in SLM
};
inline constexpr GemvBf16Tiling gemv_bf16_tiling(unsigned N) {
  return N / kGemvBf16Cols >= kBuiltXeCores ? GemvBf16Tiling{kGemvBf16Cols, 1}
                                             : GemvBf16Tiling{kGemvBf16TinyCols, 16};
}
// Both suffixes are empty at the default tiling, so `{64, 1}` names exactly the
// binaries that existed before this knob did. `_S` is `gemv.cl`'s letter for a
// split-K width and means the same thing here, one level down (in a work-group
// rather than across them).
inline std::string gemv_bf16_variant(unsigned M, unsigned K, unsigned N, GemvBf16Tiling t) {
  return "gemv_bf16_M" + std::to_string(M) + "_K" + std::to_string(K) + "_N" + std::to_string(N) +
         (t.cols == kGemvBf16Cols ? "" : "_C" + std::to_string(t.cols)) +
         (t.ksplit == 1 ? "" : "_S" + std::to_string(t.ksplit));
}

// Spec 9 §3: the int8 `lm_head` GEMV (src/kernels/gemv_i8w.cl). One tiling, gemv_bf16's
// lm_head one: 64 columns (four 16-lane subgroups) per work-group, no K split, grid
// N / 64. Its binaries exist for the lm_head shape only, at M = 1 (decode, prefill's
// step_head, the MTP draft) and M = 2..4 (the MTP verify lists).
inline constexpr unsigned kGemvI8wCols = 64;
inline std::string gemv_i8w_variant(unsigned M, unsigned K, unsigned N) {
  return "gemv_i8w_M" + std::to_string(M) + "_K" + std::to_string(K) + "_N" + std::to_string(N);
}

// ---- Spec 28b: the small-M GEMM (src/kernels/gemm_sm.cl) ------------------------------------
//
// Which kernel a linear at M rows binds is `runtime::gemv_kind(M)` (runtime/gemv_kind.h): `Gemv`
// at M = 1 - today's gemv / gemv_i8w binaries, untouched -, `SmallM` at M >= 2. `SmallM` binds
// gemm_sm / gemm_sm_i8w when the shape has a row in `kSmallMTuning` and gemv / gemv_i8w otherwise
// (the models 28b does not tune keep their M-row GEMVs; K2, Bolt and Kolibri have no M >= 2 list). Spec 28d: the
// DFlash drafter's int8 linears (block of 8) and its draft heads (7 rows), and Qwen3.8-Flash-Next's own capture
// (its verify lists' M = 2..4) bind gemm_sm too.
// LinearKind and the rule kernels::gemv_kind(M) are in kernels/linear_kind.h (no kernel-directory dependency).

// One row per weight shape: THE source of truth for its gemm_sm variant at every M (spec 28 §2:
// property R needs the same row block, group order, S and work-group shape at every M >= 2 - the
// table has no M column, so it cannot vary by M). `layout` 0 / 1 int4 (gemv's S and layout, the
// key), 2 int8 (gemv_i8w's: S 1). The knobs are gemm_sm.cl's:
//   wg      the in-work-group split-K (2 column x 8 K-slice sub-groups, SLM reduction). It writes
//           ONE slice (gemm_sm_out_s): where gemv's S is already 1 no consumer changes; gate||up's
//           consumer (prep_silu_mul, SILU_S baked) is bound at its _S1 build instead (capture.cc mlp(),
//           kernels::gemm_sm_silu_variant) - spec 28 §2's "another S with the consumer variant it needs",
//           taken because 28b measured it >= 5 % faster at M = 8. No other consumer may change S.
//   magic   the magic-exponent dequant (bit-identical output to the shift form; speed only).
//   grf256  the variant's ocloc line carries -cl-intel-256-GRF-per-thread.
//   blk2d   layout 0: the 2D block weight read, copied from gemv's production cell of the shape.
// The choices are plan 28b's lever measurement (docs/superpowers/specs/2026-10-27-spec28-small-m-
// verify-gemm-design.md, "28b as built"); src/kernels/CMakeLists.txt builds exactly these rows
// (gemm_sm_names_test holds the two together).
// The M a row is built at (`mset`): the CMake block builds exactly these (gemm_sm_built_m).
enum : unsigned char { kSmAll = 0, kSmVerify4 = 1, kSmDraft8 = 2, kSmDraft7 = 3 };
struct SmallMTuning {
  unsigned K, N, S, layout;
  bool wg, magic, grf256, blk2d;
  // kSmAll: M 2..8 and 16; kSmVerify4: 2..4; kSmDraft8: 8 (the DFlash drafter's block); kSmDraft7: 7 (its draft head)
  unsigned char mset = kSmAll;
};
inline constexpr SmallMTuning kSmallMTuning[] = {
    // Qwen3.8
    {5120, 16384, 1, 1, false, true, false, false},   // qkv||z      48 layers
    {6144, 5120, 4, 0, false, true, false, true},     // out_proj, o_proj
    {5120, 14336, 2, 0, false, true, false, true},    // q||k||v
    {5120, 34816, 8, 0, true, true, false, false},    // gate||up (2x8 work-group: one slice, SiLU at S 1)
    {17408, 5120, 4, 0, false, true, false, true},    // down
    {5120, 248320, 1, 2, true, false, false, false},  // the int8 lm_head
    // Ornith 1.5 35B-A3B's dense projections (spec 15; the operator's 28b addition, 2026-11-01) at its verify
    // lists' M = 2..4 - gemv's PROVISIONAL S / layout, magic dequant, no lever swept; the routed experts (moe.cl)
    // and the bf16 router stay as they are (spec 28m)
    {2048, 12288, 1, 1, false, true, false, false, kSmVerify4},      // qkv||z
    {4096, 2048, 4, 0, false, true, false, false, kSmVerify4},       // out_proj, o_proj
    {2048, 9216, 2, 0, false, true, false, false, kSmVerify4},       // q||k||v
    {2048, 128, 1, 1, false, true, false, false, kSmVerify4},        // a||b (the int4 checkpoint's)
    {2048, 248320, 1, 2, false, false, false, false, kSmVerify4},    // the int8 lm_head
    // Spec 28d: z-lab/Qwen3.8-27B-DFlash2's int8 linears at its block's M = 8 (the draft and projection lists,
    // runtime/dflash/dflash_capture.cc) and the compact draft heads at its 7 draft rows (the full head is Qwen3.8's
    // row above, built at 7). All in-work-group split-K: the int8 form has no grid split-K, and without it a column's
    // K walk is one sub-group's (K up to 25600) - 28b's int8-head measurement chose wg too. Every head shares the
    // knobs, so a column's output is the same whatever N (the draft vocabulary's head bitwise the full head's).
    {5120, 6144, 1, 2, true, false, false, false, kSmDraft8},        // q||k||v
    {4096, 5120, 1, 2, true, false, false, false, kSmDraft8},        // o_proj
    {5120, 34816, 1, 2, true, false, false, false, kSmDraft8},       // gate||up
    {17408, 5120, 1, 2, true, false, false, false, kSmDraft8},       // down
    {25600, 5120, 1, 2, true, false, false, false, kSmDraft8},       // fc (5 taps)
    {5120, 2048, 1, 2, true, false, false, false, kSmDraft8},        // the context's k||v (the projection list)
    {5120, 1280, 1, 2, true, false, false, false, kSmDraft8},        // the conv kernel projections (DFlash 2)
    {5120, 256, 1, 2, true, false, false, false, kSmDraft8},         // the selector's projection (DFlash 2)
    {5120, 32768, 1, 2, true, false, false, false, kSmDraft7},       // the draft vocabulary's heads
    {5120, 65536, 1, 2, true, false, false, false, kSmDraft7},
    {5120, 131072, 1, 2, true, false, false, false, kSmDraft7},
    // Spec 19v2: the RTN int4 g64 drafter, layout 1 and one output slice. Only the block's M 8; other lists keep
    // their existing rows. Groups stay ascending as in gemv; magic dequant is exact. Device cost gates still apply.
    {5120, 6144, 1, 1, false, true, false, false, kSmDraft8},
    {4096, 5120, 1, 1, false, true, false, false, kSmDraft8},
    {5120, 34816, 1, 1, false, true, false, false, kSmDraft8},
    {17408, 5120, 1, 1, false, true, false, false, kSmDraft8},
    {25600, 5120, 1, 1, false, true, false, false, kSmDraft8},
    {5120, 2048, 1, 1, false, true, false, false, kSmDraft8},
    {5120, 1280, 1, 1, false, true, false, false, kSmDraft8},
    {5120, 256, 1, 1, false, true, false, false, kSmDraft8},
    // Spec 19v2: grid split-K; the DF consumers fold the partials without another launch.
    {5120, 6144, 4, 1, false, true, false, false, kSmDraft8},
    {4096, 5120, 4, 1, false, true, false, false, kSmDraft8},
    {5120, 34816, 8, 1, false, true, false, false, kSmDraft8},
    {17408, 5120, 4, 1, false, true, false, false, kSmDraft8},
    {25600, 5120, 8, 1, false, true, false, false, kSmDraft8},
    {5120, 2048, 8, 1, false, true, false, false, kSmDraft8},
    {5120, 1280, 16, 1, false, true, false, false, kSmDraft8},
    {5120, 256, 16, 1, false, true, false, false, kSmDraft8},
    // Spec 28d: Qwen3.8-Flash-Next's dense int4 projections (GPTQ layout 0, model::qwen4exp's S) and its int8 head at
    // its verify lists' M = 2..4 (runtime/qwen4exp/qwen4exp_capture.cc) - magic dequant (bit-identical to the shift
    // form), no lever swept; the head the in-work-group form (28b's int8-head choice). Its bf16 linears (a||b, the
    // indexer, the router, the HC and the MTP head's) stay gemv_bf16.
    {2560, 16384, 1, 0, false, true, false, false, kSmVerify4},      // GDN qkv||z
    {2560, 13312, 2, 0, false, true, false, false, kSmVerify4},      // QSA q||gate||k||v
    {6144, 2560, 4, 0, false, true, false, false, kSmVerify4},       // GDN out_proj, QSA o_proj
    {2560, 248320, 1, 2, true, false, false, false, kSmVerify4},     // the int8 lm_head
    // Spec 17d: a tensor-parallel card's shapes (model::tp_card_desc, the folded crossing) at its verify lists'
    // M = 2..4 - each the one-card row's knobs for the per-card cell (gemv's S / layout, magic dequant, the 2D read
    // where the one-card cell has it); gate||up at gemv's grid S 8 (its SiLU the S 8 build), the int8 head's
    // vocabulary half in the in-work-group form, as the one-card head's.
    {5120, 8192, 1, 1, false, true, false, false, kSmVerify4},       // qkv||z, a card's heads
    {5120, 7168, 2, 0, false, true, false, true, kSmVerify4},        // q||k||v
    {5120, 17408, 8, 0, false, true, false, false, kSmVerify4},      // gate||up, the card's half
    {3072, 5120, 4, 0, false, true, false, true, kSmVerify4},        // out_proj, o_proj: the card's K rows
    {8704, 5120, 4, 0, false, true, false, true, kSmVerify4},        // down
    {5120, 124160, 1, 2, true, false, false, false, kSmVerify4},     // the int8 head's vocabulary half
};
// The row of a shape, or null (that shape keeps gemv / gemv_i8w at every M). int8: layout 2, S 1.
inline constexpr const SmallMTuning* small_m_tuning(unsigned K, unsigned N, unsigned S, unsigned layout) {
  for (const SmallMTuning& t : kSmallMTuning)
    if (t.K == K && t.N == N && t.S == S && t.layout == layout) return &t;
  return nullptr;
}
// The output's split-K slices (what the consumer folds): gemv's S, or 1 under wg.
inline constexpr unsigned gemm_sm_out_s(const SmallMTuning& t) { return t.wg ? 1 : t.S; }
// Work-group size and grid (N / cols, S): grid split-K 4 x 16 columns, wg 2 x 16 columns x 8 slices.
inline constexpr unsigned gemm_sm_wg_size(const SmallMTuning& t) { return t.wg ? 256 : 64; }
inline constexpr unsigned gemm_sm_cols(const SmallMTuning& t) { return t.wg ? 32 : 64; }
inline constexpr unsigned gemm_sm_grid_y(const SmallMTuning& t) { return t.wg ? 1 : t.S; }
inline constexpr bool small_m_table_ok() {
  for (const SmallMTuning& t : kSmallMTuning) {
    // wg writes one slice: only where gemv's S is 1, or Qwen3.8's gate||up (the SiLU has an _S1 build)
    if (t.wg && t.S != 1 && !(t.layout == 0 && t.K == 5120 && t.N == 2 * 17408)) return false;
    if (t.layout == 2 && (t.S != 1 || t.magic || t.blk2d)) return false;
    if (t.blk2d && t.layout != 0) return false;
    if (t.K % 64 != 0 || (t.K / 64) % t.S != 0 || (t.wg && (t.K / 64) % 8 != 0)) return false;
    if (t.N % gemm_sm_cols(t) != 0) return false;
  }
  return true;
}
static_assert(small_m_table_ok(), "kSmallMTuning: a row breaks gemm_sm.cl's #error guards or changes a consumer's S");
// gemm_sm_M<M>_K<K>_N<N>_S<S>_L<L>[_wg][_mg][_g256][_b2d] (int4; S the OUTPUT's slices, the binary's S define) /
// gemm_sm_i8w_M<M>_K<K>_N<N>[_wg] (int8): every knob is in the name, so a table row CMake does not build is a
// missing file at capture.
inline std::string gemm_sm_variant(unsigned M, const SmallMTuning& t) {
  const std::string mk = "_M" + std::to_string(M) + "_K" + std::to_string(t.K) + "_N" + std::to_string(t.N);
  const std::string knobs = std::string(t.wg ? "_wg" : "") + (t.magic ? "_mg" : "") + (t.grf256 ? "_g256" : "") +
                            (t.blk2d ? "_b2d" : "");
  if (t.layout == 2) return "gemm_sm_i8w" + mk + knobs;
  return "gemm_sm" + mk + "_S" + std::to_string(gemm_sm_out_s(t)) + "_L" + std::to_string(t.layout) + knobs;
}
// The M each row is built at (src/kernels/CMakeLists.txt's spec 28b block; gemm_sm_names_test holds the two
// together): 2..8 and 16 for Qwen3.8's linears and head; Ornith's and Flash-Next's at their verify lists'
// 2..4; the DFlash drafter's linears at its block's 8 and its draft heads at 7 (spec 28d).
inline std::vector<unsigned> gemm_sm_built_m(const SmallMTuning& t) {
  switch (t.mset) {
    case kSmVerify4: return {2, 3, 4};
    case kSmDraft8: return {8};
    case kSmDraft7: return {7};
    default: return {2, 3, 4, 5, 6, 7, 8, 16};
  }
}
inline constexpr const char* gemm_sm_entry(const SmallMTuning& t) { return t.layout == 2 ? "gemm_sm_i8w" : "gemm_sm"; }
// The int4 / int8 linear a capture binds at M rows of kind `k` - the one naming rule the captures and
// the names tests share.
inline std::string int4_linear_variant(LinearKind k, unsigned M, unsigned K, unsigned N, unsigned S, unsigned L) {
  const SmallMTuning* t = k == LinearKind::SmallM ? small_m_tuning(K, N, S, L) : nullptr;
  return t ? gemm_sm_variant(M, *t) : gemv_variant(M, K, N, S, L);
}
inline std::string int8_linear_variant(LinearKind k, unsigned M, unsigned K, unsigned N) {
  const SmallMTuning* t = k == LinearKind::SmallM ? small_m_tuning(K, N, 1, 2) : nullptr;
  return t ? gemm_sm_variant(M, *t) : gemv_i8w_variant(M, K, N);
}

// The between-GEMV kernels (src/kernels/prep.cl). Only `prep_res_norm` varies
// in shape: `prep_silu_mul` and `prep_gated_head` have the model's dimensions
// (17408 / gate||up S=4, 48x128 / qkv||z S=1) baked into the source, so `M` is
// their whole variant space.
inline std::string prep_res_norm_variant(unsigned M, unsigned K, unsigned S_PREV) {
  return "prep_res_norm_M" + std::to_string(M) + "_K" + std::to_string(K) + "_SP" +
         std::to_string(S_PREV);
}
// **The two-stage split of `prep_res_norm`** - spec 1.5 lever L1. The runtime
// binds this pair at every one of the 129 sites; `prep_res_norm_variant` above
// still names the single-work-group kernel, which stays compiled and stays run
// by tests/kernels/prep_test.cc as the pair's reference.
//
// `G` is stage A's work-group count and therefore the number of fp32 partial
// sums-of-squares it writes; stage B's `G` is how many it folds, and the two
// MUST be the same number - the grid `runtime::DecodeBuffers::kNormGroups`
// launches, and the size `norm_sumsq` is allocated at. It is in both names for
// that reason: pairing mismatched grids names a binary that does not exist and
// throws at capture rather than reducing the wrong number of slices.
//
// `W` is stage B's own work-group count. W = G spreads the rescale pass over
// the same grid the fold uses; W = 1 is the plan's literal single-work-group
// finish and is compiled as the measurement control (src/kernels/CMakeLists.txt).
inline std::string prep_res_fold_variant(unsigned M, unsigned K, unsigned S_PREV, unsigned G) {
  return "prep_res_fold_M" + std::to_string(M) + "_K" + std::to_string(K) + "_SP" +
         std::to_string(S_PREV) + "_G" + std::to_string(G);
}
inline std::string prep_norm_finish_variant(unsigned M, unsigned K, unsigned G, unsigned W) {
  return "prep_norm_finish_M" + std::to_string(M) + "_K" + std::to_string(K) + "_G" +
         std::to_string(G) + "_W" + std::to_string(W);
}
// Spec 14: the MLP intermediate size is baked into prep_silu_mul (prep.cl's
// SILU_N). Qwen3.8's 17408 keeps the historical name and command line; any other
// width is a separate binary named `_I<intermediate>` (model::ModelDesc::
// intermediate_suffix() - the same rule), built with -DINTERMEDIATE=<I>.
inline std::string intermediate_suffix(unsigned I) {
  return I == 17408 ? std::string() : "_I" + std::to_string(I);
}
inline std::string prep_silu_mul_variant(unsigned M, unsigned I) {
  return "prep_silu_mul_M" + std::to_string(M) + intermediate_suffix(I);
}
// Spec 15b: the kernels below that bake a MODEL shape their other parameters do
// not name take it as trailing arguments - hidden for embed_gather, the GDN heads
// for prep_gated_head / gdn_step*, the FA heads for the attention kernels - and
// append kernels/shape_suffix.h's suffix: empty at Qwen3.8's shapes, so no
// existing binary is renamed. The defaults are those
// reference shapes, for the kernel tests that bind Qwen3.8's binaries; the
// runtime always passes the model descriptor's.
inline std::string prep_gated_head_variant(unsigned M, unsigned gdn_k = kRefGdnKHeads,
                                           unsigned gdn_v = kRefGdnVHeads) {
  return "prep_gated_head_M" + std::to_string(M) + gdn_suffix(gdn_k, gdn_v);
}

// The GDN decode step (src/kernels/gdn_step.cl). Grid (48 heads, 4 state-column
// chunks), work-group 256; `M` is its whole variant space, every other
// dimension being the model's and baked into the source.
inline std::string gdn_step_variant(unsigned M, unsigned gdn_k = kRefGdnKHeads,
                                    unsigned gdn_v = kRefGdnVHeads) {
  return "gdn_step_M" + std::to_string(M) + gdn_suffix(gdn_k, gdn_v);
}

// The decode-attention trio (src/kernels/attn.cl), the 16 full-attention
// layers. `attn_prep` runs a grid of (28, M) - 24 q-heads then 4 kv-heads - and
// indexes the KV caches by absolute position, so `M` is its whole variant
// space. `attn_decode` (grid (4 kv-heads, max_len/BLOCK blocks)) and
// `attn_reduce` (grid (24, M)) additionally bake `MAXLEN` **and** `BLOCK`,
// because `attn_part` is strided `[24][MAXLEN/BLOCK][M][258]` and the stride
// has to be a compile-time constant; the grid itself is set at capture from
// `buffers.max_len`, which must therefore equal the `MAXLEN` of the variant
// bound to it.
//
// `BLOCK` is in the name - `attn_decode_M1_L16384_B64` - for the reason
// `prep_res_fold`'s `G` is: it is the ONE number the host and the device must
// agree on that neither can check against the other. The host's copy is
// `runtime::DecodeBuffers::kAttnBlock`, and it sizes `attn_part` and sets the
// grid; the device's is `-DATTN_BLOCK` in src/kernels/CMakeLists.txt. Putting
// it in the name turns a disagreement into "no such binary" at capture instead
// of a buffer strided one way and written another. Spec 1.5's lever L5 moved
// it from 256 to 64 and is why it is a parameter at all.
inline std::string attn_prep_variant(unsigned M, unsigned q = kRefFaQHeads,
                                     unsigned kv = kRefFaKvHeads) {
  return "attn_prep_M" + std::to_string(M) + fa_suffix(q, kv);
}
inline std::string attn_decode_variant(unsigned M, unsigned MAXLEN, unsigned BLOCK,
                                       unsigned q = kRefFaQHeads, unsigned kv = kRefFaKvHeads) {
  return "attn_decode_M" + std::to_string(M) + "_L" + std::to_string(MAXLEN) + "_B" +
         std::to_string(BLOCK) + fa_suffix(q, kv);
}
inline std::string attn_reduce_variant(unsigned M, unsigned MAXLEN, unsigned BLOCK,
                                       unsigned q = kRefFaQHeads, unsigned kv = kRefFaKvHeads) {
  return "attn_reduce_M" + std::to_string(M) + "_L" + std::to_string(MAXLEN) + "_B" +
         std::to_string(BLOCK) + fa_suffix(q, kv);
}

// Spec 10 (plan 10b): decode attention v2 (src/kernels/attn_v2.cl) - `attn_decode_v2`
// and `attn_reduce_v2` in ONE binary per M. No MAXLEN: the grid is (4, T) and
// `attn_part` is strided [24][T][M][258] whatever max_len is. `T` (work-groups per kv
// head, the stride target) is in the name for the reason `B` is above: the host's copy
// is `runtime::DecodeScratch::kAttnV2Blocks`, the device's `-DTGT`.
inline std::string attn_v2_variant(unsigned M, unsigned T, unsigned q = kRefFaQHeads,
                                   unsigned kv = kRefFaKvHeads) {
  return "attn_v2_M" + std::to_string(M) + "_T" + std::to_string(T) + fa_suffix(q, kv);
}

// Spec 8 (plan 8b): the MTP lists' variants (src/kernels/CMakeLists.txt, B70_MTP).
// `_Z`: prep_res_fold with ZERO_RESID (the residual starts at the fold's partials);
// `_X<stride>`: prep_norm_finish writing rows at that pitch; `_S1`: the kernel's baked
// split-K of its producer is 1 (the MTP head's bf16 linears write [M][N] directly).
inline std::string prep_res_fold_zero_variant(unsigned M, unsigned K, unsigned G) {
  return prep_res_fold_variant(M, K, 1, G) + "_Z";
}
inline std::string prep_norm_finish_strided_variant(unsigned M, unsigned K, unsigned G,
                                                    unsigned W, unsigned X) {
  return prep_norm_finish_variant(M, K, G, W) + "_X" + std::to_string(X);
}
// The MTP head's bf16 gate||up (SILU_S = 1), at the head's own intermediate (17408 on
// Qwen3.8).
inline std::string prep_silu_mul_s1_variant(unsigned M, unsigned I) {
  return prep_silu_mul_variant(M, I) + "_S1";
}
// Spec 28b: the SiLU a decode list binds after gate||up, by the slices gate||up wrote - gemv's S (SILU_S baked,
// the plain build) or 1 (gemm_sm's 2x8 work-group, the _S1 build; built at M 2..8 for I 17408).
inline std::string gemm_sm_silu_variant(unsigned M, unsigned I, unsigned gate_up_s, unsigned gemv_s) {
  return gate_up_s == gemv_s ? prep_silu_mul_variant(M, I) : prep_silu_mul_s1_variant(M, I);
}
inline std::string attn_prep_s1_variant(unsigned M, unsigned q = kRefFaQHeads,
                                        unsigned kv = kRefFaKvHeads) {
  return attn_prep_variant(M, q, kv) + "_S1";
}

// Spec 12b (`--kv-cache int8`, src/kernels/kv8.cl): the int8 KV cache's writer and
// decode reader, each the `_KV8` twin of the bf16 binary it replaces - same grid, two
// more arguments (the K and V scales). Additive: no existing binary's name or command
// line moves, and a bf16 engine binds none of these.
inline std::string attn_prep_kv8_variant(unsigned M, unsigned q = kRefFaQHeads,
                                         unsigned kv = kRefFaKvHeads) {
  return attn_prep_variant(M, q, kv) + "_KV8";
}
inline std::string attn_prep_s1_kv8_variant(unsigned M, unsigned q = kRefFaQHeads,
                                            unsigned kv = kRefFaKvHeads) {
  return attn_prep_s1_variant(M, q, kv) + "_KV8";
}
inline std::string attn_v2_kv8_variant(unsigned M, unsigned T, unsigned q = kRefFaQHeads,
                                       unsigned kv = kRefFaKvHeads) {
  return attn_v2_variant(M, T, q, kv) + "_KV8";
}
// Spec 28c (src/kernels/attn_verify_dpas.cl): the verify attention on DPAS - entry
// `attn_verify_dpas` (`attn_verify_dpas_kv8` in the `_KV8` binary) - writing v2's partials, so
// the reduce stays attn_v2_variant(M, T)'s (or attn_v2_kv8_variant's) attn_reduce_v2[_kv8].
// Built at Qwen3.8's heads only (fa_suffix is empty there); M = 2..8 bf16, 2..4 int8.
inline std::string attn_verify_dpas_variant(unsigned M, unsigned T, bool kv8, unsigned q = kRefFaQHeads,
                                            unsigned kv = kRefFaKvHeads) {
  return "attn_verify_dpas_M" + std::to_string(M) + "_T" + std::to_string(T) + fa_suffix(q, kv) +
         (kv8 ? "_KV8" : "");
}
// SPEC_SLOT_STRIDE (one slot of MtpBuffers::gdn_spec, gdn_layers x 48 x 128 x 128
// floats) is baked, so a model with another GDN layer count is another binary:
// Qwen3.8's 48 keeps the historical name, any other count is `_G<gdn_layers>`; the
// v-head count (in the stride too) and the k-heads follow as the GDN suffix (spec 15b).
// `slots` is the kernel's N_SLOTS (the ring of states row m writes into, (live + m) %
// slots): spec 8's 4 keeps every historical name; spec 19a's verify probe at M = 5..8
// binds the 8-slot build, `_N8` (runtime::MtpBuffers::slots).
inline std::string gdn_step_slots_variant(unsigned M, unsigned gdn_layers,
                                          unsigned gdn_k = kRefGdnKHeads,
                                          unsigned gdn_v = kRefGdnVHeads, unsigned slots = 4) {
  return "gdn_step_slots_M" + std::to_string(M) +
         (slots == 4 ? std::string() : "_N" + std::to_string(slots)) +
         (gdn_layers == 48 ? std::string() : "_G" + std::to_string(gdn_layers)) +
         gdn_suffix(gdn_k, gdn_v);
}

// The control block as the kernels see it: `runtime::Control`
// (src/runtime/control.h) indexed as a flat `uint` array. The device side gets
// these numbers from ONE place - the `CTRL_DEFINES` line in
// src/kernels/CMakeLists.txt - and this is the host mirror, checked against
// `offsetof(runtime::Control, …)` by tests/kernels/argmax_test.cc (and by the
// capture unit). Keep the three in step; a mismatch is a silently wrong token.
namespace ctrl_index {
inline constexpr unsigned kPos = 0;         // Control::pos
inline constexpr unsigned kNActive = 1;     // Control::n_active
inline constexpr unsigned kCurToken = 2;    // Control::cur_token[8]
inline constexpr unsigned kOutToken = 10;   // Control::out_token[8]
inline constexpr unsigned kDebugFlag = 18;  // Control::debug_flag
inline constexpr unsigned kGdnLive = 19;    // Control::gdn_live (spec 8, SPEC_SLOTS only)
inline constexpr unsigned kDraftStat = 28;  // Control::draft_stat[2] (open-work 13, draft_prob.cl's CTRL_DSTAT)
}  // namespace ctrl_index

// The token-boundary kernels (src/kernels/embed_gather.cl, argmax.cl). `M` is
// the launch grid's y extent, i.e. the tokens in flight; it appears in the name
// because the runtime asks for kernels by (kernel, M), even though these two
// generate M-independent code (M only bounds a compile-time assert).
inline std::string embed_gather_variant(unsigned M, unsigned hidden = kRefHidden) {
  return "embed_gather_M" + std::to_string(M) + hidden_suffix(hidden);
}
// VOCAB_USED (the ids the tokenizer defines) is baked: Qwen3.8's 248077 keeps the
// historical name, any other count is `_V<vocab_used>` (spec 14).
inline std::string argmax_stage1_variant(unsigned M, unsigned vocab_used) {
  return "argmax_stage1_M" + std::to_string(M) +
         (vocab_used == 248077 ? std::string() : "_V" + std::to_string(vocab_used));
}
// Stage 2 is a single work-group that loops m < ctrl.n_active internally: no
// variants at all, so the binary is named after the entry point.
inline std::string argmax_stage2_variant() { return "argmax_stage2"; }
// Open-work 13: the draft's probability over a logits row of `rows` entries, `used` of them real ids
// (src/kernels/draft_prob.cl: draft_prob_stage1 / _stage2 in one binary, WG 256, 1024 logits per stage-1 group).
inline std::string draft_prob_variant(unsigned rows, unsigned used) {
  return "draft_prob_N" + std::to_string(rows) + "_U" + std::to_string(used);
}
// Spec 16b: the pipeline stages' peer hand-off (pp_handoff.cl: pp_send, pp_recv), one binary.
inline std::string pp_handoff_variant() { return "pp_handoff"; }
inline constexpr unsigned kPpHandoffWg = 256;   // pp_handoff.cl WG

// Spec 15c: the mixture-of-experts block's three kernels (src/kernels/moe.cl) -
// moe_route, moe_gate_up, moe_down - in ONE binary per (M, shape), as prep.cl's entry
// points share theirs: `moe_M<M>_E<experts>_T<top_k>_D<hidden>_I<expert intermediate>`.
// Before them the decode list runs gemv_bf16 over the router || shared-gate rows at
// N = MoeDesc::router_n() (kernels::gemv_bf16_tiling: {16, 16} at 272).
inline std::string moe_variant(unsigned M, unsigned experts, unsigned top_k, unsigned hidden,
                               unsigned inter) {
  return "moe_M" + std::to_string(M) + "_E" + std::to_string(experts) + "_T" +
         std::to_string(top_k) + "_D" + std::to_string(hidden) + "_I" + std::to_string(inter);
}
// The two in-work-group K splits moe.cl is compiled with (its UP_KS / DN_KS, passed by
// src/kernels/CMakeLists.txt's MOE_DEFINES - the device side's one home). They fix the
// work-groups the capture sets: moe_gate_up 64 x UP_KS lanes, moe_down 16 x (top_k + 1)
// x DN_KS lanes; moe_route is one lane per expert. A host/device disagreement is a
// group-size error at capture, not a wrong sum.
inline constexpr unsigned kMoeUpKs = 4, kMoeDnKs = 2;
inline constexpr unsigned moe_gate_up_wg() { return 64 * kMoeUpKs; }
inline constexpr unsigned moe_down_wg(unsigned top_k) { return 16 * (top_k + 1) * kMoeDnKs; }
// moe_gate_up's grid x: (top_k + 1) slots x (2 I / 16 n-tiles / 4 per work-group).
inline constexpr unsigned moe_gate_up_groups(unsigned top_k, unsigned inter) {
  return (top_k + 1) * (2 * inter / 16 / 4);
}
// The route row moe_route writes and the two expert kernels read: u32 words per
// (layer, token), moe.cl's R_* defines. tests/kernels/moe_test.cc reads each by name.
namespace moe_route {
inline constexpr unsigned kWords = 32;
inline constexpr unsigned kIds = 0;          // u32 expert id of slot k, k < top_k
inline constexpr unsigned kWeights = 8;      // fp32 bits: w_k, a bf16 value
inline constexpr unsigned kSharedGate = 16;  // fp32 bits: s_b, a bf16 value
inline constexpr unsigned kProbs = 17;       // fp32 bits: p_k before renormalisation
inline constexpr unsigned kProb9 = 25;       // fp32 bits: p of rank top_k (first not taken)
}  // namespace moe_route

// Spec 8 §11: the draft's argmax over a reduced vocabulary of `nv` ids
// (src/kernels/draft_vocab.cl) - `dv_argmax_stage1` (with the scatter into the full
// row) and `dv_argmax_stage2` (the id-table mapping) in one binary per size, as prep.cl's
// entry points share theirs. The compact GEMV before it is `gemv_i8w_variant(1, K, nv)` or,
// for a bf16 head, `gemv_bf16_variant(1, K, nv, gemv_bf16_tiling(nv))` ({64, 1}).
inline constexpr unsigned kDraftVocabChunk = 1024;   // draft_vocab.cl CHUNK: stage 1's grid is nv / 1024
inline std::string dv_argmax_variant(unsigned nv) { return "dv_argmax_N" + std::to_string(nv); }
}  // namespace kernels
