#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"   // spec 20d: the reused prefill names
#include "model/kolibri1.h"

// Spec 20c: Kolibri-1's device binaries by name - the host half of the Kolibri block in
// src/kernels/CMakeLists.txt, in its own header so kernels.h (shared with every branch in flight) is
// not edited. Every Kolibri-only binary is prefixed `kol_` and carries in its name every define its
// code depends on, so a host / device disagreement names a binary that does not exist and throws at
// capture. The reused kernels keep their own names where those carry every shape (gemv_variant,
// gemv_bf16_variant, gemv_i8w_variant, prep_res_fold_variant - kernels.h) or take `kol_` where the
// existing name carries another model's vocabulary (embed_gather, argmax).
//
// The shapes are Kolibri-1's (model::kolibri1(): hidden 2560, 48 / 4 heads x 128, 384 experts top 6 x
// 512, router_n 512, vocab 128000, window 513, ring 4096); a capture checks its descriptor against
// them (runtime/kolibri/kolibri_capture.cc) - a synthetic checkpoint differs only in its layer count.
namespace kernels::kolibri {

inline constexpr unsigned kHidden = 2560, kQHeads = 48, kKvHeads = 4, kHd = 128, kQkvN = 7168;
inline constexpr unsigned kExperts = 384, kTopK = 6, kRouterN = 512, kInter = 512, kVocab = 128000;
inline constexpr unsigned kWindow = 513, kRing = 4096;

inline constexpr unsigned kNormG = 20, kNormW = 20, kNormWg = 256;   // prep_res_fold FOLD_G / stage-B grids at K 2560
inline constexpr unsigned kUpKs = 4, kDnKs = 2;                      // kol_moe_gate_up / kol_moe_down K splits
inline constexpr unsigned kAttnTgt = 32, kAttnWg = 128, kAttnPart = 130;
inline constexpr unsigned kPrepWg = 128;                             // kol_attn_prep: one head per work-group
inline constexpr unsigned kRouteWg = 256;                            // = model::Kolibri1Desc::kRouteLanes
inline constexpr unsigned kArgmaxChunk = 1024, kArgmaxWg = 256, kEmbedWg = 256;
inline constexpr unsigned kSlots = kTopK + 1;                        // 6 routed + the shared expert

inline constexpr unsigned moe_gate_up_wg() { return 64 * kUpKs; }                       // 256
inline constexpr unsigned moe_down_wg() { return 16 * kSlots * kDnKs; }                 // 224
inline constexpr unsigned moe_gate_up_groups() { return kSlots * (2 * kInter / 16 / 4); }   // 112
inline constexpr unsigned argmax_groups() { return (kVocab + kArgmaxChunk - 1) / kArgmaxChunk; }  // 125

// The route row kol_route writes and the MoE kernels read (kol_moe.cl R_*), u32 words per (layer, token).
namespace route {
inline constexpr unsigned kWords = 32, kIds = 0 /* 6 ids ascending */, kWeights = 8 /* fp32 sigmoid(logit) */,
                          kSel = 16 /* fp32 logit + bias */, kNext = 24 /* sel of rank 6 */,
                          kLogitMin = 25 /* the smallest selected logit, diagnostics */;
}

inline std::string m_(unsigned M) { return "_M" + std::to_string(M); }
inline std::string norm_variant(unsigned M) { return "kol_norm" + m_(M) + "_K2560_G20_W20"; }
inline std::string post_add_variant(unsigned M) { return "kol_post_add" + m_(M) + "_K2560_G20"; }
// S: the fused q||k||v GEMV's split-K (2 in the int4 arm, 1 in the bf16 arm: gemv_bf16 writes [M][N]).
inline std::string attn_prep_variant(unsigned M, unsigned S, bool sliding) {
  return "kol_attn_prep" + m_(M) + "_N7168_S" + std::to_string(S) + "_Q48KV4" + (sliding ? "_R4096" : "_F");
}
inline std::string route_variant(unsigned M) { return "kol_route" + m_(M) + "_E384_T6_N512_L256"; }
inline std::string moe_variant(unsigned M) { return "kol_moe" + m_(M) + "_E384_T6_D2560_I512_SH"; }
inline std::string attn_variant(unsigned M, bool sliding) {
  return "kol_attn" + m_(M) + "_T" + std::to_string(kAttnTgt) + "_Q48KV4" + (sliding ? "_W513_R4096" : "_F");
}
inline std::string attn_eager_variant(unsigned M, bool sliding) {
  return "kol_attn_eager" + m_(M) + "_T" + std::to_string(kAttnTgt) + "_Q48KV4" + (sliding ? "_W513_R4096" : "_F");
}
// prep.cl's prep_res_fold with ZERO_RESID at Kolibri's K: o_proj's S slices -> its own bf16 row and Σ².
inline std::string fold_zero_variant(unsigned M, unsigned S) {
  return prep_res_fold_variant(M, kHidden, S, kNormG) + "_Z";
}
inline std::string fold_variant(unsigned M) { return prep_res_fold_variant(M, kHidden, 0, kNormG); }
// embed_gather.cl / argmax.cl at Kolibri's vocabulary (their names elsewhere do not carry VOCAB).
inline std::string embed_variant(unsigned M) { return "kol_embed_gather" + m_(M) + "_D2560_V128000"; }
inline std::string argmax1_variant(unsigned M) { return "kol_argmax_stage1" + m_(M) + "_N128000_V128000"; }
inline std::string argmax2_variant() { return "kol_argmax_stage2_N128000"; }

// The attention arm's GEMV rows (the int4 arm's S from the descriptor: PROVISIONAL S2 / S4).
inline std::string qkv_variant(model::KolAttnForm a, unsigned qkv_s = 2) {
  return a == model::KolAttnForm::Int4 ? gemv_variant(1, kHidden, kQkvN, qkv_s, 0)
                                       : gemv_bf16_variant(1, kHidden, kQkvN, gemv_bf16_tiling(kQkvN));
}
inline std::string oproj_variant(model::KolAttnForm a, unsigned oproj_s = 4) {
  return a == model::KolAttnForm::Int4 ? gemv_variant(1, kQHeads * kHd, kHidden, oproj_s, 0)
                                       : gemv_bf16_variant(1, kQHeads * kHd, kHidden, gemv_bf16_tiling(kHidden));
}
inline std::string router_variant() { return gemv_bf16_variant(1, kHidden, kRouterN, gemv_bf16_tiling(kRouterN)); }
inline std::string lm_head_variant(bool int8) {
  return int8 ? gemv_i8w_variant(1, kHidden, kVocab) : gemv_bf16_variant(1, kHidden, kVocab, gemv_bf16_tiling(kVocab));
}

// Every binary a Kolibri M = 1 decode list binds (runtime/kolibri/kolibri_capture.cc forms the same
// names), for an attention arm, a head form and an attention form (`eager`: B70_KOLIBRI_ATTN=eager
// binds kol_attn_eager.cl's four kernels for kol_attn.cl's two). The two-card lists add pp_handoff
// under --pipeline-handoff peer (16b's binary). Capture checks each exists before appending a command;
// tests/kernels/kolibri_variant_names_test.cc holds the union to the CMake block's list.
inline std::vector<std::string> decode_variants(model::KolAttnForm a, bool int8_head, bool eager) {
  const bool i4 = a == model::KolAttnForm::Int4;
  const unsigned S = i4 ? 2u : 1u, OS = i4 ? 4u : 1u;
  return {
      embed_variant(1),
      fold_variant(1),
      fold_zero_variant(1, OS),
      norm_variant(1),
      post_add_variant(1),
      qkv_variant(a),
      oproj_variant(a),
      attn_prep_variant(1, S, true),
      attn_prep_variant(1, S, false),
      eager ? attn_eager_variant(1, true) : attn_variant(1, true),
      eager ? attn_eager_variant(1, false) : attn_variant(1, false),
      router_variant(),
      route_variant(1),
      moe_variant(1),
      lm_head_variant(int8_head),
      argmax1_variant(1),
      argmax2_variant(),
  };
}

// ==== Spec 20d: Kolibri-1 prefill (begin) ====================================================
// One chunk of at most kPfC positions through the whole model on the Level Zero list
// (runtime/kolibri/kolibri_prefill.cc has the walk). The Kolibri-only prefill binaries are kol_pf_*
// (src/kernels/kolibri/kol_pf_{moe,attn,linear}.cl); 20c's kol_prep.cl is built again at M = kPfC (the
// norm, the sandwich's post-add, the attention prep - decode's chains row for row); the route is 20c's
// decode binary on grid (1, C); the rest are reused sources at Kolibri's shapes (pf_embed, pf_prep's
// pf_res_fold, pf_gemv_bf16's router, pf_moe_gemm, pf_gemm_T0, 18c's k2_pf_dequant_slab for the int4
// attention arm). runtime/kolibri/kolibri_sizes.h spells the same constants device-free.
inline constexpr unsigned kPfC = 2048, kPfTm = 32, kPfSlab = 1024;
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;   // one weight batch of bf16 expert blocks
inline constexpr unsigned kPfKt = 64, kPfRpw = 8, kPfHpw = 4;  // kol_pf_flash_attn: key tile, rows, heads per WG
inline constexpr unsigned kPfSortWg = 256, kPfGatherWg = 64, kPfCombineWg = 256, kPfDequantWg = 16;
inline constexpr unsigned kPfSlabWg = 16;                    // kol_pf_bf16_slab / k2_pf_dequant_slab
// kol_pf_moe.cl's header words (H_*): [0] tiles used, [1] the shared expert's first row, [2] rows used,
// [3] C, [4 + e] expert e's rows (e < 384), [4 + 384] C.
namespace pf_hdr {
inline constexpr unsigned kTiles = 0, kSharedRow = 1, kRows = 2, kC = 3, kCount = 4;
inline constexpr unsigned words() { return (kCount + kExperts + 1 + 15) / 16 * 16; }   // 400
}  // namespace pf_hdr

inline std::string pf_moe_variant() { return "kol_pf_moe_E384_T6_D2560_I512_L256"; }
inline std::string pf_flash_variant(bool sliding, bool eager) {
  return std::string("kol_pf_flash_attn_Q48KV4") + (sliding ? "_W513_R4096" : "_F") + (eager ? "_EAGER" : "");
}
// kol_pf_linear.cl: one bf16-tiled linear's slab (the bf16 attention arm), any width (the tail zero-padded).
inline std::string pf_bf16_slab_variant(unsigned K, unsigned N) {
  return "kol_pf_bf16_slab_K" + std::to_string(K) + "_N" + std::to_string(N);
}
// 18c's k2_pf_linear.cl at Kolibri's widths: the int4 attention arm's layout-0 slabs.
inline std::string pf_int4_slab_variant(unsigned K, unsigned N) {
  return "k2_pf_dequant_slab_K" + std::to_string(K) + "_N" + std::to_string(N);
}
inline std::string pf_slab_variant(model::KolAttnForm a, unsigned K, unsigned N) {
  return a == model::KolAttnForm::Int4 ? pf_int4_slab_variant(K, N) : pf_bf16_slab_variant(K, N);
}
inline std::string pf_embed_variant() { return "kol_pf_embed_gather_D2560_V128000"; }
// pf_prep.cl's stage A at K 2560 (runtime M): SP0 after the embedding and over the MoE's `mo`, _Z over
// o_proj's one slice (the sandwich's own row and Σ²).
inline std::string pf_fold_variant() { return pf_res_fold_variant(kHidden, 0, kNormG); }
inline std::string pf_fold_zero_variant() { return pf_res_fold_zero_variant(kHidden, kNormG); }
inline std::string pf_norm_variant() { return norm_variant(kPfC); }
inline std::string pf_post_add_variant() { return post_add_variant(kPfC); }
inline std::string pf_attn_prep_variant(bool sliding) { return attn_prep_variant(kPfC, 1, sliding); }
inline std::string pf_router_variant() { return pf_moe_router_variant(kHidden, kRouterN); }
inline std::string pf_route_variant() { return route_variant(1); }   // decode's binary, grid (1, C)
inline std::string pf_gemm_gu_variant() { return pf_moe_gemm_variant(kHidden, 2 * kInter, false, true); }
inline std::string pf_gemm_dn_variant() { return pf_moe_gemm_variant(kInter, kHidden, false, false); }

// Every binary a Kolibri prefill chunk binds for an attention arm and an attention form (`eager`:
// B70_KOLIBRI_ATTN=eager binds the _EAGER flash builds). The head is decode's binaries (decode_variants);
// tests/kernels/kolibri_pf_variant_names_test.cc holds the union to the CMake block's list.
inline std::vector<std::string> prefill_variants(model::KolAttnForm a, bool eager) {
  return {
      pf_embed_variant(),
      pf_fold_variant(),
      pf_fold_zero_variant(),
      pf_norm_variant(),
      pf_post_add_variant(),
      pf_slab_variant(a, kHidden, kQkvN),
      pf_slab_variant(a, kQHeads * kHd, kHidden),
      pf_gemm_variant(false),
      pf_attn_prep_variant(true),
      pf_attn_prep_variant(false),
      pf_flash_variant(true, eager),
      pf_flash_variant(false, eager),
      pf_router_variant(),
      pf_route_variant(),
      pf_moe_variant(),
      pf_gemm_gu_variant(),
      pf_gemm_dn_variant(),
  };
}
// ==== Spec 20d: Kolibri-1 prefill (end) ======================================================

}  // namespace kernels::kolibri
