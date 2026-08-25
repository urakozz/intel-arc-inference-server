#pragma once
#include <string>

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
// **The choice is a function of `N` and lives here** so that the runtime
// (src/runtime/capture.cc) and the table check (tests/kernels/kernel_table_test.cc)
// cannot drift apart. `lm_head` (N = 248320) keeps `{64, 1}`: 3880 work-groups
// at 98.5% of measured bandwidth in situ, nothing to win - and it is measured
// unmoved by this knob existing (4378.555 -> 4378.353 µs/launch in situ, 0.005%).
// `a‖b` (N = 128) takes `{16, 16}` - 8 work-groups of 16 K-slice subgroups, so
// **128 hardware threads over the launch instead of 8**, and both halves of the
// pair earn their place: the `ksplit` is what creates the threads and the
// `cols` is what spreads them over eight Xe-cores rather than one.
//
// The width is measured, not chosen (docs/15 §L2, in situ, three points):
// ksplit 1 / 4 / 16 read **48.774 / 13.115 / 5.340 µs** per launch. 16 stops
// 2.4x above the shape's 2.22 µs traffic floor, so at most 0.15 ms/token is
// left in it and a 32-way split cannot repay its own gate run.
//
// **`ksplit > 1` changes the summation order** (the K slices are merged by a
// fixed SLM tree - gemv_bf16.cl states the order), so a variant with it is held
// to the reference tolerance and to the golden gate, not to bit-identity.
inline constexpr unsigned kGemvBf16Cols = 64;      // 4 subgroups per work-group
inline constexpr unsigned kGemvBf16TinyCols = 16;  // 1 column tile per work-group
// The B70's Xe-core count (docs/01-hardware.md). A shape whose default grid
// already puts at least one work-group on every core is not the tiny-N case.
inline constexpr unsigned kGemvBf16Cores = 32;
struct GemvBf16Tiling {
  unsigned cols;    // COLS_PER_WG: output columns per work-group
  unsigned ksplit;  // KSPLIT: subgroups splitting one tile's K, merged in SLM
};
inline constexpr GemvBf16Tiling gemv_bf16_tiling(unsigned N) {
  return N / kGemvBf16Cols >= kGemvBf16Cores ? GemvBf16Tiling{kGemvBf16Cols, 1}
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

// The between-GEMV kernels (src/kernels/prep.cl). Only `prep_res_norm` varies
// in shape: `prep_silu_mul` and `prep_gated_head` have the model's dimensions
// (17408 / gate||up S=4, 48x128 / qkv||z S=1) baked into the source, so `M` is
// their whole variant space.
inline std::string prep_res_norm_variant(unsigned M, unsigned K, unsigned S_PREV) {
  return "prep_res_norm_M" + std::to_string(M) + "_K" + std::to_string(K) + "_SP" +
         std::to_string(S_PREV);
}
inline std::string prep_silu_mul_variant(unsigned M) { return "prep_silu_mul_M" + std::to_string(M); }
inline std::string prep_gated_head_variant(unsigned M) {
  return "prep_gated_head_M" + std::to_string(M);
}

// The GDN decode step (src/kernels/gdn_step.cl). Grid (48 heads, 4 state-column
// chunks), work-group 256; `M` is its whole variant space, every other
// dimension being the model's and baked into the source.
inline std::string gdn_step_variant(unsigned M) { return "gdn_step_M" + std::to_string(M); }

// The decode-attention trio (src/kernels/attn.cl), the 16 full-attention
// layers. `attn_prep` runs a grid of (28, M) - 24 q-heads then 4 kv-heads - and
// indexes the KV caches by absolute position, so `M` is its whole variant
// space. `attn_decode` (grid (4 kv-heads, max_len/256 blocks)) and
// `attn_reduce` (grid (24, M)) additionally bake `MAXLEN`, because `attn_part`
// is strided `[24][MAXLEN/256][M][258]` and the stride has to be a compile-time
// constant; the grid itself is set at capture from `buffers.max_len`, which
// must therefore equal the `MAXLEN` of the variant bound to it.
inline std::string attn_prep_variant(unsigned M) { return "attn_prep_M" + std::to_string(M); }
inline std::string attn_decode_variant(unsigned M, unsigned MAXLEN) {
  return "attn_decode_M" + std::to_string(M) + "_L" + std::to_string(MAXLEN);
}
inline std::string attn_reduce_variant(unsigned M, unsigned MAXLEN) {
  return "attn_reduce_M" + std::to_string(M) + "_L" + std::to_string(MAXLEN);
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
}  // namespace ctrl_index

// The token-boundary kernels (src/kernels/embed_gather.cl, argmax.cl). `M` is
// the launch grid's y extent, i.e. the tokens in flight; it appears in the name
// because the runtime asks for kernels by (kernel, M), even though these two
// generate M-independent code (M only bounds a compile-time assert).
inline std::string embed_gather_variant(unsigned M) {
  return "embed_gather_M" + std::to_string(M);
}
inline std::string argmax_stage1_variant(unsigned M) {
  return "argmax_stage1_M" + std::to_string(M);
}
// Stage 2 is a single work-group that loops m < ctrl.n_active internally: no
// variants at all, so the binary is named after the entry point.
inline std::string argmax_stage2_variant() { return "argmax_stage2"; }
}  // namespace kernels
