#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"   // spec 23d: the reused prefill names

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

// ==== Spec 23d: the prefill chunk ===============================================================================
// One chunk of at most kPfC rows through one device's layers on the Level Zero list (runtime/muse/muse_prefill.cc has
// the walk). The Muse-only prefill binaries are mg_pf_* (src/kernels/muse/mg_pf_{attn,prep}.cl); 23c's mg_prep.cl is
// built again at M = kPfC (the norms, the sandwich, the attention prep and SiLU x up at S 1); the rest are reused
// sources at Muse's shapes (18c's k2_pf_dequant_slab for the int4 layout-0 slabs - widths that are not whole 1024-column
// slabs get one zero-padded tail -, prefill/pf_prep.cl's pf_res_fold _Z, the shape-free pf_gemm_T0). The head over the
// last row is decode's (mg_final_norm from the M kPfC norm binary on the last row's offsets, the lm_head, argmax).
inline constexpr unsigned kPfC = 2048, kPfSlab = 1024;
inline constexpr unsigned kPfKt = 64, kPfRpw = 8, kPfHpw = 4;   // mg_pf_flash_attn: key tile, rows, heads a work-group
inline constexpr unsigned kPfFlashWg = 16 * (kPfHpw * kPfRpw / 8);   // 64: HPW x RPW / 8 sub-groups of 16
inline constexpr unsigned kPfEagerWg = 128;                         // the eager build: one (row, head) a work-group
inline constexpr unsigned kPfGateN = kQN;                           // mg_pf_attn_gate's row (4096)
inline constexpr unsigned kPfGateGroups = kPfGateN / kRowWg;        // 16
static_assert(kRing >= kPfC + kWindow - 1 && kRing % kPfKt == 0,
              "a chunk's own keys and the window before it must fit the ring without colliding, in whole tiles");

inline std::string pf_embed_variant() { return "mg_pf_embed" + m_(kPfC) + "_K6656_G26_V202048"; }
inline std::string pf_norm_variant() { return norm_variant(kPfC); }           // mg_norm and mg_final_norm
inline std::string pf_post_add_variant() { return post_add_variant(kPfC); }
inline std::string pf_attn_prep_variant(bool sliding) { return attn_prep_variant(kPfC, 1, sliding); }
inline std::string pf_silu_variant() { return silu_variant(kPfC, 1); }
inline std::string pf_gate_variant() { return "mg_pf_attn_gate" + m_(kPfC) + "_N4096"; }
// mg_pf_attn.cl: the flash build (the default) or the eager one (B70_MUSE_ATTN=eager), sliding / full.
inline std::string pf_flash_variant(bool sliding, bool eager) {
  return std::string("mg_pf_flash_attn_Q32KV2") + (sliding ? "_W2048_R4096" : "_F") + (eager ? "_EAGER" : "");
}
// 18c's k2_pf_linear.cl at Muse's linears: q||gate||k||v 6656 x 8704, o_proj 4096 x 6656, gate||up 6656 x 39936,
// down 19968 x 6656.
inline std::string pf_slab_variant(unsigned K, unsigned N) {
  return "k2_pf_dequant_slab_K" + std::to_string(K) + "_N" + std::to_string(N);
}
inline std::string pf_fold_zero_variant() { return kernels::pf_res_fold_zero_variant(kHidden, kNormG); }
inline std::string pf_gemm_variant() { return kernels::pf_gemm_variant(false); }

// Every binary a Muse prefill chunk binds for an attention form (the head's lm_head and argmax are decode's, which
// the decode capture already checked). tests/kernels/muse_pf_variant_names_test.cc holds the union to the CMake
// block's list.
inline std::vector<std::string> prefill_variants(bool eager) {
  return {
      pf_embed_variant(),
      pf_norm_variant(),
      pf_post_add_variant(),
      pf_slab_variant(kHidden, kQgkvN),
      pf_slab_variant(kQN, kHidden),
      pf_slab_variant(kHidden, kGateUpN),
      pf_slab_variant(kInter, kHidden),
      pf_gemm_variant(),
      pf_attn_prep_variant(true),
      pf_attn_prep_variant(false),
      pf_flash_variant(true, eager),
      pf_flash_variant(false, eager),
      pf_gate_variant(),
      pf_fold_zero_variant(),
      pf_silu_variant(),
  };
}

}  // namespace kernels::muse
