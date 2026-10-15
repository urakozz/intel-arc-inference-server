#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"

// Spec 23c: Muse-Glimmer-30B's device binaries by name - the host half of the "Spec 23c" block in
// src/kernels/CMakeLists.txt, in its own header so kernels.h (shared with every branch in flight) is not edited.
// Every Muse-only binary is prefixed `mg_` and carries in its name every define its code depends on, so a host /
// device disagreement names a binary that does not exist and throws at capture. The reused kernels keep their own
// names where those carry every shape (gemv_variant, gemv_bf16_variant, gemv_i8w_variant, prep_res_fold_variant -
// kernels.h); argmax takes `mg_` (its name elsewhere does not carry the vocabulary).
//
// The shapes are Muse-Glimmer-30B's (model::muse_glimmer(): hidden 6656, 32 / 2 heads x 128, FF 19968, vocab
// 202048, window 2048, ring 4096); a capture checks its descriptor against them (runtime/muse/muse_capture.cc) - a
// synthetic checkpoint differs only in its layer count.
//
// **M is a parameter of every name** (the rows of a step): 23c builds M = 1 only; 23f adds the verify widths
// M = 2..16 as new CMake lines without restructuring (spec 23 §5.2).
namespace kernels::muse {

inline constexpr unsigned kHidden = 6656, kQHeads = 32, kKvHeads = 2, kHd = 128, kQN = 4096, kKvN = 256;
inline constexpr unsigned kQgkvN = 8704, kInter = 19968, kGateUpN = 39936, kVocab = 202048;
inline constexpr unsigned kWindow = 2048, kRing = 4096;
inline constexpr unsigned kImageToken = 200092, kVideoToken = 200091;   // config.json image / video_token_id

inline constexpr unsigned kNormG = 26, kNormW = 26, kRowWg = 256;   // Σ² chunks of 256 at K 6656; norm grids
inline constexpr unsigned kAttnTgt = 64, kAttnWg = 128, kAttnPart = 130;   // 34 absolute 64-key blocks a window
inline constexpr unsigned kPrepWg = 128;                                     // mg_attn_prep: one head per work-group
inline constexpr unsigned kPrepGroups = 2 * kQHeads + 2 * kKvHeads;          // 68: q, gate, k, v heads
inline constexpr unsigned kArgmaxChunk = 1024, kArgmaxWg = 256;
inline constexpr unsigned kSiluGroups = (kInter + kRowWg - 1) / kRowWg;      // 78

// The four int4 GEMVs' split-K (layout 0), PROVISIONAL: Qwen3.8's measured cells at the nearest shapes
// (model::MuseGlimmerDesc's qgkv_s / oproj_s / gate_up_s / down_s) until 23c Task 7's sweep.
inline constexpr unsigned kQgkvS = 2, kOprojS = 4, kGateUpS = 8, kDownS = 4;

inline constexpr unsigned argmax_groups() { return (kVocab + kArgmaxChunk - 1) / kArgmaxChunk; }   // 198

inline std::string m_(unsigned M) { return "_M" + std::to_string(M); }
inline std::string embed_variant(unsigned M) { return "mg_embed" + m_(M) + "_K6656_G26_V202048"; }
// mg_norm and mg_final_norm share a binary (the NORM_G family).
inline std::string norm_variant(unsigned M) { return "mg_norm" + m_(M) + "_K6656_G26_W26"; }
inline std::string post_add_variant(unsigned M) { return "mg_post_add" + m_(M) + "_K6656_G26"; }
// S: the fused q||gate||k||v GEMV's split-K.
inline std::string attn_prep_variant(unsigned M, unsigned S, bool sliding) {
  return "mg_attn_prep" + m_(M) + "_N8704_S" + std::to_string(S) + "_Q32KV2" + (sliding ? "_R4096" : "_F");
}
inline std::string silu_variant(unsigned M, unsigned S) { return "mg_silu_mul" + m_(M) + "_I19968_S" + std::to_string(S); }
// mg_attn.cl (flash, the default) and mg_attn_eager.cl (B70_MUSE_ATTN=eager), sliding (window 2048 through the
// 4096-slot ring) or full (NoPE, the growing cache).
inline std::string attn_variant(unsigned M, bool sliding) {
  return "mg_attn" + m_(M) + "_T" + std::to_string(kAttnTgt) + "_Q32KV2" + (sliding ? "_W2048_R4096" : "_F");
}
inline std::string attn_eager_variant(unsigned M, bool sliding) {
  return "mg_attn_eager" + m_(M) + "_T" + std::to_string(kAttnTgt) + "_Q32KV2" + (sliding ? "_W2048_R4096" : "_F");
}
// prep.cl's prep_res_fold with ZERO_RESID at Muse's K: a sub-block's S slices (o_proj's, down's) -> its own bf16 row
// and Σ² (the sandwich: the post norm normalises the sub-block's own output).
inline std::string fold_zero_variant(unsigned M, unsigned S) {
  return prep_res_fold_variant(M, kHidden, S, kNormG) + "_Z";
}
inline std::string argmax1_variant(unsigned M) { return "mg_argmax_stage1" + m_(M) + "_N202048_V202048"; }
inline std::string argmax2_variant() { return "mg_argmax_stage2_N202048"; }

// The GEMVs (gemv.cl int4 layout 0 at the PROVISIONAL S; the head gemv_i8w / gemv_bf16).
inline std::string qgkv_variant(unsigned M = 1, unsigned S = kQgkvS) { return gemv_variant(M, kHidden, kQgkvN, S, 0); }
inline std::string oproj_variant(unsigned M = 1, unsigned S = kOprojS) { return gemv_variant(M, kQN, kHidden, S, 0); }
inline std::string gate_up_variant(unsigned M = 1, unsigned S = kGateUpS) { return gemv_variant(M, kHidden, kGateUpN, S, 0); }
inline std::string down_variant(unsigned M = 1, unsigned S = kDownS) { return gemv_variant(M, kInter, kHidden, S, 0); }
inline std::string lm_head_variant(bool int8) {
  return int8 ? gemv_i8w_variant(1, kHidden, kVocab) : gemv_bf16_variant(1, kHidden, kVocab, gemv_bf16_tiling(kVocab));
}

// Every binary a Muse M = 1 decode list binds (runtime/muse/muse_capture.cc forms the same names), for a head form
// and an attention form (`eager`: B70_MUSE_ATTN=eager binds mg_attn_eager.cl's one kernel for mg_attn.cl's two).
// The two-card lists add pp_handoff under --pipeline-handoff peer (16b's binary). Capture checks each exists before
// appending a command; tests/kernels/muse_variant_names_test.cc holds the union to the CMake block's list.
inline std::vector<std::string> decode_variants(bool int8_head, bool eager) {
  return {
      embed_variant(1),
      norm_variant(1),
      post_add_variant(1),
      qgkv_variant(),
      attn_prep_variant(1, kQgkvS, true),
      attn_prep_variant(1, kQgkvS, false),
      eager ? attn_eager_variant(1, true) : attn_variant(1, true),
      eager ? attn_eager_variant(1, false) : attn_variant(1, false),
      oproj_variant(),
      fold_zero_variant(1, kOprojS),
      gate_up_variant(),
      silu_variant(1, kGateUpS),
      down_variant(),
      fold_zero_variant(1, kDownS),
      lm_head_variant(int8_head),
      argmax1_variant(1),
      argmax2_variant(),
  };
}

}  // namespace kernels::muse
