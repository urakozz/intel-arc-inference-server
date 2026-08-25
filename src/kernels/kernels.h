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
inline std::string gemv_bf16_variant(unsigned M, unsigned K, unsigned N) {
  return "gemv_bf16_M" + std::to_string(M) + "_K" + std::to_string(K) + "_N" + std::to_string(N);
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
