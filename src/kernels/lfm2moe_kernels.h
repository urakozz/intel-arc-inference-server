#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"   // spec 24d: the reused prefill names
#include "model/lfm2moe.h"

// Spec 24c: LFM2.5-24B-A2B's device binaries by name - the host half of the spec 24c block in
// src/kernels/CMakeLists.txt, in its own header so kernels.h (shared with every branch in flight) is not edited.
// Every LFM2-only binary is prefixed `lfm_` and carries in its name the defines its code depends on (a host / device
// disagreement names a binary that does not exist and throws at capture). The reused kernels keep their own names
// where those carry every shape (gemv_variant, gemv_bf16_variant, gemv_i8w_variant - kernels.h); argmax.cl takes
// `lfm_` (the existing names carry another vocabulary).
//
// The shapes are model::lfm2moe()'s (hidden 2048, 32 / 8 heads x 64, 64 experts top 4 x 1536, dense 11776, vocab
// 128000 of which 125017 are ids); a capture checks its descriptor against them (runtime/lfm2moe/lfm2moe_capture.cc)
// - a synthetic checkpoint differs only in its layer count. The int4 GEMVs' {S, layout} are the descriptor's
// (model::Lfm2MoeDesc::linears - 24b's PROVISIONAL picks, P0's sweep and Task 8 re-pick them there, never here).
namespace kernels::lfm2moe {

inline constexpr unsigned kHidden = 2048, kQHeads = 32, kKvHeads = 8, kHd = 64, kQkvN = 3072, kInProjN = 6144;
inline constexpr unsigned kExperts = 64, kTopK = 4, kInter = 1536, kDenseInter = 11776, kVocab = 128000,
                          kVocabUsed = 125017;
inline constexpr unsigned kConvRing = 16;                          // = loader::lfm::kConvRing
inline constexpr unsigned kNormWgs = 8, kNormWg = 256;             // lfm_res_norm: 8 work-groups of 256 (PROVISIONAL, Task 8)
inline constexpr unsigned kUpKs = 4, kDnKs = 2, kAttnTgt = 32;     // PROVISIONAL (Task 8 sweeps)
inline constexpr unsigned kAttnWg = 64, kAttnPart = kHd + 2;       // lfm_attn / lfm_attn_eager: a work-group per head dim
inline constexpr unsigned kPrepWg = kHd;                           // lfm_attn_prep: one head a work-group
inline constexpr unsigned kRouterWg = 128, kRouteWg = kExperts;    // lfm_router (8 spans x 16 columns), lfm_route
inline constexpr unsigned kConvWg = 256, kSiluWg = 256, kEmbedWg = 256;
inline constexpr unsigned kArgmaxChunk = 1024, kArgmaxWg = 256;
inline constexpr unsigned kNormRouterWgs = kExperts / 16;          // the arm `router`'s extra work-groups (4)
inline constexpr unsigned kConvInWg = 48;                          // the arm `conv`: B, C and x of one 16-channel tile

inline constexpr unsigned moe_gate_up_wg() { return 64 * kUpKs; }                          // 256
inline constexpr unsigned moe_down_wg() { return 16 * kTopK * kDnKs; }                     // 128
inline constexpr unsigned moe_gate_up_groups() { return kTopK * (2 * kInter / 16 / 4); }   // 192
inline constexpr unsigned argmax_groups() { return (kVocab + kArgmaxChunk - 1) / kArgmaxChunk; }   // 125

// The route row lfm_route writes and lfm_moe_down reads (lfm_moe.cl R_*), u32 words per (layer, token).
namespace route {
inline constexpr unsigned kWords = 16, kIds = 0 /* 4 ids, rank order */, kW = 4 /* f32 of bf16 */,
                          kSel = 8 /* 5 sel, rank 0..4 */, kSum = 13 /* f32 of bf16 Σ + eps */;
}

// The fused arms (spec 24 §4.5, decision 6) a list binds: runtime::lfm2moe::LfmFuse is the switch.
struct Arms {
  bool route = false, router = false, conv = false;
};

// The int4 GEMVs' cells, read off the descriptor (the ONE place their {S, layout} live).
struct Gemv {
  unsigned K, N, S, L;
};
inline Gemv gemv_of(const model::Lfm2MoeDesc& d, model::LfmLinearId id) {
  for (uint32_t l = 0; l < d.layers; ++l)
    for (const model::LfmLinear& ln : d.linears(l))
      if (ln.id == id) return {ln.shape.K, ln.shape.N, ln.shape.S, ln.shape.layout};
  return {0, 0, 0, 0};
}
inline Gemv in_proj() { return gemv_of(model::lfm2moe(), model::LfmLinearId::InProj); }
inline Gemv out_proj() { return gemv_of(model::lfm2moe(), model::LfmLinearId::OutProj); }
inline Gemv qkv() { return gemv_of(model::lfm2moe(), model::LfmLinearId::Qkv); }
inline Gemv o_proj() { return gemv_of(model::lfm2moe(), model::LfmLinearId::OProj); }
inline Gemv dense_gate_up() { return gemv_of(model::lfm2moe(), model::LfmLinearId::DenseGateUp); }
inline Gemv dense_down() { return gemv_of(model::lfm2moe(), model::LfmLinearId::DenseDown); }
inline std::string gemv_name(const Gemv& g) { return gemv_variant(1, g.K, g.N, g.S, g.L); }

inline std::string m_(unsigned M) { return "_M" + std::to_string(M); }
// S_fold: the preceding GEMV's split-K (0 = `_NF`: nothing to fold); router_fused: the arm `router` (ffn_norm of a
// MoE layer only).
inline std::string res_norm_variant(unsigned M, unsigned S_fold, bool router_fused) {
  return "lfm_res_norm" + m_(M) + "_K2048_W" + std::to_string(kNormWgs) +
         (S_fold ? "_S" + std::to_string(S_fold) : std::string("_NF")) + (router_fused ? "_R" : "");
}
inline std::string conv_variant(unsigned M, unsigned S) { return "lfm_conv" + m_(M) + "_K2048_S" + std::to_string(S); }
// The arm `conv`: in_proj's GEMV (split-K 1, the conv-tile column order, layout L) with the conv as its epilogue.
inline std::string conv_in_variant(unsigned M, unsigned L) { return "lfm_conv_in" + m_(M) + "_K2048_L" + std::to_string(L); }
inline std::string router_variant(unsigned M) { return "lfm_router" + m_(M) + "_K2048_E64"; }
inline std::string route_variant(unsigned M) { return "lfm_route" + m_(M) + "_E64_T4"; }
inline std::string moe_variant(unsigned M, bool route_fused) {
  return "lfm_moe" + m_(M) + "_E64_T4_D2048_I1536" + (route_fused ? "_R" : "");
}
inline std::string attn_prep_variant(unsigned M, unsigned S) {
  return "lfm_attn_prep" + m_(M) + "_S" + std::to_string(S) + "_Q32KV8";
}
inline std::string attn_variant(unsigned M, bool eager) {
  return std::string(eager ? "lfm_attn_eager" : "lfm_attn") + m_(M) + "_T" + std::to_string(kAttnTgt) + "_Q32KV8";
}
inline std::string silu_mul_variant(unsigned M, unsigned S) {
  return "lfm_silu_mul" + m_(M) + "_I11776_S" + std::to_string(S);
}
inline std::string embed_variant(unsigned M) { return "lfm_embed_gather" + m_(M) + "_K2048_V128000"; }
inline std::string argmax1_variant(unsigned M) { return "lfm_argmax_stage1" + m_(M) + "_V128000_U125017"; }
inline std::string argmax2_variant() { return "lfm_argmax_stage2_V128000"; }
inline std::string head_variant(bool int8) {
  return int8 ? gemv_i8w_variant(1, kHidden, kVocab) : gemv_bf16_variant(1, kHidden, kVocab, gemv_bf16_tiling(kVocab));
}

// Every binary an M = 1 decode list of `d` binds for a head form, an attention form, one or two cards and the fused
// arms (runtime/lfm2moe/lfm2moe_capture.cc forms the same names). `two_cards` under `--pipeline-handoff peer` adds
// spec 16b's pp_handoff (built outside this block). tests/kernels/lfm2moe_variant_names_test.cc holds the union to
// the CMake block's list.
inline std::vector<std::string> decode_variants(const model::Lfm2MoeDesc& d, bool int8_head, bool eager, bool two_cards,
                                                Arms arms = {}) {
  std::vector<std::string> v = {embed_variant(1), res_norm_variant(1, 0, false), head_variant(int8_head),
                                argmax1_variant(1), argmax2_variant()};
  auto add = [&](const std::string& s) {
    for (const std::string& x : v)
      if (x == s) return;
    v.push_back(s);
  };
  for (uint32_t l = 0; l < d.layers; ++l) {
    unsigned mixer_s = 0;
    for (const model::LfmLinear& ln : d.linears(l)) {
      const Gemv g{ln.shape.K, ln.shape.N, ln.shape.S, ln.shape.layout};
      switch (ln.id) {
        case model::LfmLinearId::InProj:
          if (arms.conv) {
            add(conv_in_variant(1, g.L));
          } else {
            add(gemv_name(g));
            add(conv_variant(1, g.S));
          }
          break;
        case model::LfmLinearId::OutProj: case model::LfmLinearId::OProj:
          add(gemv_name(g));
          mixer_s = g.S;
          break;
        case model::LfmLinearId::Qkv:
          add(gemv_name(g));
          add(attn_prep_variant(1, g.S));
          add(attn_variant(1, eager));
          break;
        case model::LfmLinearId::DenseGateUp:
          add(gemv_name(g));
          add(silu_mul_variant(1, g.S));
          break;
        case model::LfmLinearId::DenseDown:
          add(gemv_name(g));
          add(res_norm_variant(1, g.S, false));   // the next layer's operator norm (or the final norm) folds it
          break;
        default:
          break;
      }
    }
    // ffn_norm folds the mixer's out_proj / o_proj slices; under the arm `router` (MoE layers) it also runs the router.
    const bool moe = !d.is_dense(l);
    add(res_norm_variant(1, mixer_s, moe && arms.router));
    if (moe) {
      if (!arms.router) add(router_variant(1));
      if (!arms.route) add(route_variant(1));
      add(moe_variant(1, arms.route));
    }
  }
  if (two_cards) add(pp_handoff_variant());
  return v;
}

// ==== Spec 24d: LFM2.5-24B-A2B prefill (begin) =====================================================================
// One chunk of at most kPfC positions through a device's layers on the Level Zero list (runtime/lfm2moe/
// lfm2moe_prefill.cc has the walk). The LFM2-only prefill binaries are lfm_pf_* (src/kernels/lfm2moe/
// lfm_pf_{conv,moe,attn,prep}.cl); 24c's per-row kernels are built again at M = kPfC (the M in the name is the prefill
// line: the norm _NF / _S1 - every prefill linear is one unsplit GEMM, so S 1 -, the router, the route, the attention
// prep at S 1 - decode's chains row for row); the rest are reused sources at LFM2's shapes (prefill/pf_dequant_slab.cl
// at each linear's {K, N, layout} from model::Lfm2MoeDesc::linears - the layout follows the descriptor, never assumed
// -, pf_gemm_T0, pf_gemm_T0_SILU for the dense gate||up, pf_moe_gemm at K 2048 N 3072 SiLU and K 1536 N 2048). The
// head after the last chunk is decode's binaries over the last row (decode_variants). runtime/lfm2moe/lfm2moe_sizes.h
// spells the same constants device-free.
inline constexpr unsigned kPfC = 2048, kPfTm = 32, kPfSlab = 1024;
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;    // one weight batch of bf16 expert blocks (Kolibri's)
inline constexpr unsigned kPfKt = 64, kPfRpw = 16, kPfHpw = 4;  // lfm_pf_attn: key tile, rows, heads a WG (PROVISIONAL)
inline constexpr unsigned kPfSortWg = 64, kPfGatherWg = 64, kPfCombineWg = 256, kPfDequantWg = 16;
inline constexpr unsigned kPfConvWg = 256, kPfConvRows = 16, kPfEmbedWg = 256;
// lfm_pf_moe.cl's header words (H_*): [0] tiles used, [1] 0 (reserved), [2] rows used, [3] C, [4 + e] expert e's rows.
namespace pf_hdr {
inline constexpr unsigned kTiles = 0, kRows = 2, kC = 3, kCount = 4;
inline constexpr unsigned words() { return (kCount + kExperts + 15) / 16 * 16; }   // 80
}  // namespace pf_hdr

inline std::string pf_embed_variant() { return "lfm_pf_embed_gather_K2048_V128000"; }
// lfm_pf_conv.cl (lfm_pf_conv_seed + lfm_pf_conv): `tiles` - in_proj in the conv-tile column order (the arm conv's load).
inline std::string pf_conv_variant(bool tiles) { return std::string("lfm_pf_conv_K2048_C2048") + (tiles ? "_T" : ""); }
inline std::string pf_moe_variant() { return "lfm_pf_moe_E64_T4_D2048_I1536_TM32"; }
inline std::string pf_attn_variant(bool eager) {
  return "lfm_pf_attn_Q32KV8_HPW" + std::to_string(kPfHpw) + "_RPW" + std::to_string(kPfRpw) + "_KT" +
         std::to_string(kPfKt) + (eager ? "_EAGER" : "");
}
inline std::string pf_norm_variant(bool fold) { return res_norm_variant(kPfC, fold ? 1 : 0, false); }
inline std::string pf_router_variant() { return router_variant(kPfC); }
inline std::string pf_route_variant() { return route_variant(kPfC); }
inline std::string pf_attn_prep_variant() { return attn_prep_variant(kPfC, 1); }
// One int4 linear's 1024-column slab at its OWN layout (24b review I1: model::Lfm2MoeDesc::linears' {S, layout}).
inline std::string pf_slab_variant(const model::GemvShape& s) { return pf_dequant_slab_variant(s.K, s.N, s.layout); }
inline std::string pf_gemm_gu_variant() { return pf_moe_gemm_variant(kHidden, 2 * kInter, false, true); }
inline std::string pf_gemm_dn_variant() { return pf_moe_gemm_variant(kInter, kHidden, false, false); }

// Every binary a prefill chunk of `d` binds (the head is decode's: decode_variants) for an attention form and the
// in_proj layout (`conv_tiles`: the arm conv's load). tests/kernels/lfm2moe_pf_variant_names_test.cc holds the union
// to the CMake block's list.
inline std::vector<std::string> prefill_variants(const model::Lfm2MoeDesc& d, bool eager, bool conv_tiles) {
  std::vector<std::string> v = {pf_embed_variant(), pf_norm_variant(false), pf_norm_variant(true),
                                pf_gemm_variant(false)};
  auto add = [&](const std::string& s) {
    for (const std::string& x : v)
      if (x == s) return;
    v.push_back(s);
  };
  for (uint32_t l = 0; l < d.layers; ++l) {
    for (const model::LfmLinear& ln : d.linears(l)) {
      if (ln.kind != model::WeightKind::Int4) continue;   // the router: lfm_router at M kPfC
      add(pf_slab_variant(ln.shape));
      if (ln.id == model::LfmLinearId::DenseGateUp) add(pf_gemm_silu_variant());
    }
    if (d.is_attention(l)) {
      add(pf_attn_prep_variant());
      add(pf_attn_variant(eager));
    } else {
      add(pf_conv_variant(conv_tiles));
    }
    if (!d.is_dense(l)) {
      add(pf_router_variant());
      add(pf_route_variant());
      add(pf_moe_variant());
      add(pf_gemm_gu_variant());
      add(pf_gemm_dn_variant());
    }
  }
  return v;
}
// ==== Spec 24d: LFM2.5-24B-A2B prefill (end) =======================================================================

}  // namespace kernels::lfm2moe
