#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"   // spec 25d: the reused prefill names
#include "model/bolt.h"

// Spec 25c: Nemotron 3.5 Bolt's device binaries by name - the host half of the spec 25c block in src/kernels/
// CMakeLists.txt, in its own header so kernels.h (shared with every branch in flight) is not edited. Every Bolt-only
// binary is prefixed `nb_` and carries in its name the defines its code depends on (a host / device disagreement names a
// binary that does not exist and throws at capture). The reused kernels keep their own names where those carry every
// shape (gemv_variant, gemv_bf16_variant, gemv_i8w_variant - kernels.h); argmax.cl takes `nb_` (the existing names carry
// another vocabulary); the attention is Muse-Glimmer's binaries by name (Decision 15).
//
// The shapes are model::bolt()'s (hidden 4096, Mamba 128 heads x 64 / 8 groups / state 128 / conv 4, NoPE GQA 16 at
// head_dim 128, the 1024 latent, the 5376 shared expert, top-9, vocab 131072); a capture checks its descriptor against
// them (runtime/bolt/bolt_capture.cc) - a synthetic or --layers checkpoint differs only in its block count. The int4 GEMVs'
// {S, layout} are the descriptor's (model::BoltDesc::linears - 25b's PROVISIONAL picks; P0's sweep re-picks them there,
// never here). E (240 .. 496 by block) is a kernel ARGUMENT of the router / route / experts (one binary for every E); I
// (1536 .. 2560) is in the experts' names.
namespace kernels::bolt {

inline constexpr unsigned kHidden = 4096, kMHeads = 128, kMHeadDim = 64, kMGroups = 8, kMState = 128, kMConv = 4;
inline constexpr unsigned kInProjN = 18560, kXbc = 10240, kInner = 8192, kQHeads = 32, kKvHeads = 2, kHd = 128;
inline constexpr unsigned kLatent = 1024, kShared = 5376, kTopK = 9, kVocab = 131072, kConvRing = 16, kGnormGroup = 1024;
inline constexpr unsigned kQkvN = 4608, kFc1N = kLatent + kShared, kMaxExperts = 512;
inline constexpr unsigned kNormWgs = 8, kNormWg = 256;            // nb_res_norm: 8 work-groups of 256 (PROVISIONAL, Task 8)
inline constexpr unsigned kUpKs = 4, kDnKs = 2;                   // nb_moe_up / nb_moe_down's K splits (PROVISIONAL)
inline constexpr unsigned kRouterWg = 128, kRouteWg = 256;        // nb_router (8 spans x 16 columns), nb_route
inline constexpr unsigned kConvWg = 256, kSsmWg = 256, kSsmPw = 16, kGnormWg = 256, kActWg = 256, kPrepWg = 256,
                          kEmbedWg = 256;
inline constexpr unsigned kArgmaxChunk = 1024, kArgmaxWg = 256;
inline constexpr unsigned kAttnTgt = 64, kAttnWg = 128, kAttnPart = kHd + 2;   // Muse's mg_attn at T 64
inline constexpr float kEps = 1e-5f;

inline constexpr unsigned moe_up_wg() { return 64 * kUpKs; }                              // 256
inline constexpr unsigned moe_down_wg() { return 16 * kTopK * kDnKs; }                    // 288
inline constexpr unsigned moe_up_groups(unsigned I) { return kTopK * (I / 64); }          // 9 x I / 64
inline constexpr unsigned moe_down_groups() { return kLatent / 16; }                      // 64
inline constexpr unsigned ssm_groups() { return kMHeads * (kMHeadDim / kSsmPw); }         // 512
inline constexpr unsigned argmax_groups() { return (kVocab + kArgmaxChunk - 1) / kArgmaxChunk; }   // 128

// The route row nb_route writes and nb_moe_down reads (nb_moe.cl R_*), u32 words per (layer, token).
namespace route {
inline constexpr unsigned kWords = 48, kIds = 0 /* 9, ascending */, kW = 9 /* fp32 */, kSel = 18 /* 10: ranks 0..9 */,
                          kSum = 28, kSelMin = 29;
}

// The fused arms (spec 25 §6.1, Decision 16) a list binds: runtime::bolt::BoltFuse is the switch.
struct Arms {
  bool router = false, route = false, conv = false;
};

inline std::string m_(unsigned M) { return "_M" + std::to_string(M); }
inline std::string s_(unsigned S) { return "_S" + std::to_string(S); }
// S1: the pending slices' split-K (0 = `_NF`: nothing to fold); S2: a second source's (the MoE block's shared_down; 0 =
// one source); router_fused: the arm `router` (a MoE block's input norm only).
inline std::string res_norm_variant(unsigned M, unsigned S1, unsigned S2, bool router_fused) {
  return "nb_res_norm" + m_(M) + "_K4096_W" + std::to_string(kNormWgs) + (S1 ? s_(S1) : std::string("_NF")) +
         (S2 ? "_T" + std::to_string(S2) : std::string()) + (router_fused ? "_R" : "");
}
inline std::string conv_variant(unsigned M, unsigned S) { return "nb_conv" + m_(M) + "_C10240" + s_(S); }
inline std::string ssm_variant(unsigned M, unsigned S, bool conv_fused, bool bf16_state) {
  return "nb_ssm" + m_(M) + "_H128_P64_N128_G8" + s_(S) + (conv_fused ? "_C" : "") + (bf16_state ? "_B16" : "");
}
inline std::string gnorm_variant(unsigned M, unsigned S) { return "nb_gnorm" + m_(M) + "_D8192_G1024" + s_(S); }
inline std::string router_variant(unsigned M) { return "nb_router" + m_(M) + "_K4096"; }
inline std::string route_variant(unsigned M) { return "nb_route" + m_(M) + "_T9"; }
inline std::string moe_up_variant(unsigned M, unsigned I, bool route_fused) {
  return "nb_moe_up" + m_(M) + "_I" + std::to_string(I) + "_L1024_T9" + (route_fused ? "_R" : "");
}
inline std::string moe_down_variant(unsigned M, unsigned I) {
  return "nb_moe_down" + m_(M) + "_I" + std::to_string(I) + "_L1024_T9";
}
inline std::string sh_act_variant(unsigned M, unsigned S) { return "nb_sh_act" + m_(M) + "_L1024_SH5376" + s_(S); }
inline std::string attn_prep_variant(unsigned M, unsigned S) { return "nb_attn_prep" + m_(M) + "_KV2" + s_(S); }
// Decision 15: Muse-Glimmer's full (NoPE) decode attention at Q 32 KV 2 head_dim 128, TGT 64 - flash (mg_attn_decode +
// mg_attn_reduce, the gate a ones buffer) or eager (mg_attn_eager, ONE launch, its score row [M][32][max_len]).
inline std::string attn_variant(unsigned M, bool eager) {
  return std::string(eager ? "mg_attn_eager" : "mg_attn") + m_(M) + "_T64_Q32KV2_F";
}
inline std::string embed_variant(unsigned M) { return "nb_embed_gather" + m_(M) + "_K4096_V131072"; }
inline std::string argmax1_variant(unsigned M) { return "nb_argmax_stage1" + m_(M) + "_V131072"; }
inline std::string argmax2_variant() { return "nb_argmax_stage2_V131072"; }
inline std::string head_variant(bool int8) {
  return int8 ? gemv_i8w_variant(1, kHidden, kVocab) : gemv_bf16_variant(1, kHidden, kVocab, gemv_bf16_tiling(kVocab));
}
inline std::string gemv_name(const model::GemvShape& s) { return gemv_variant(1, s.K, s.N, s.S, s.layout); }

// The descriptor's shape of linear `id` in block l (the ONE place {S, layout} live).
inline model::GemvShape linear_shape(const model::BoltDesc& d, uint32_t l, model::BoltLinearId id) {
  for (const model::BoltLinear& ln : d.linears(l))
    if (ln.id == id) return ln.shape;
  return {0, 0, 0, 0};
}
// The split-K slices block l leaves for the next norm to fold: {S1, S2} (Mamba out_proj's, attention o_proj's; a MoE
// block's fc2 and shared_down).
struct Pending {
  unsigned s1 = 0, s2 = 0;
};
inline Pending pending_of(const model::BoltDesc& d, uint32_t l) {
  if (d.is_mamba(l)) return {linear_shape(d, l, model::BoltLinearId::OutProj).S, 0};
  if (d.is_attention(l)) return {linear_shape(d, l, model::BoltLinearId::OProj).S, 0};
  return {linear_shape(d, l, model::BoltLinearId::Fc2).S, linear_shape(d, l, model::BoltLinearId::SharedDown).S};
}

// Every binary an M = 1 decode list of `d` binds for a head form, an attention form, a peer hand-off (spec 16b's
// pp_handoff, built outside this block), the fused arms and the SSM state's precision (Decision 2)
// (runtime/bolt/bolt_capture.cc forms the same names). tests/kernels/bolt_variant_names_test.cc holds the union to the CMake
// block's list.
inline std::vector<std::string> decode_variants(const model::BoltDesc& d, bool int8_head, bool eager, bool peer,
                                                bool bf16_state, Arms arms = {}) {
  std::vector<std::string> v = {embed_variant(1), head_variant(int8_head), argmax1_variant(1), argmax2_variant()};
  auto add = [&](const std::string& s) {
    for (const std::string& x : v)
      if (x == s) return;
    v.push_back(s);
  };
  for (uint32_t l = 0; l < d.layers; ++l) {
    const Pending p = l == 0 ? Pending{} : pending_of(d, l - 1);
    add(res_norm_variant(1, p.s1, p.s2, d.is_moe(l) && arms.router));
    if (d.is_mamba(l)) {
      const model::GemvShape in = linear_shape(d, l, model::BoltLinearId::InProj);
      add(gemv_name(in));
      if (!arms.conv) add(conv_variant(1, in.S));
      add(ssm_variant(1, in.S, arms.conv, bf16_state));
      add(gnorm_variant(1, in.S));
      add(gemv_name(linear_shape(d, l, model::BoltLinearId::OutProj)));
    } else if (d.is_attention(l)) {
      const model::GemvShape qkv = linear_shape(d, l, model::BoltLinearId::Qkv);
      add(gemv_name(qkv));
      add(attn_prep_variant(1, qkv.S));
      add(attn_variant(1, eager));
      add(gemv_name(linear_shape(d, l, model::BoltLinearId::OProj)));
    } else {
      const unsigned I = d.inter.at(l);
      if (!arms.router) add(router_variant(1));
      if (!arms.route) add(route_variant(1));
      const model::GemvShape f1 = linear_shape(d, l, model::BoltLinearId::Fc1SharedUp);
      add(gemv_name(f1));
      add(sh_act_variant(1, f1.S));
      add(moe_up_variant(1, I, arms.route));
      add(moe_down_variant(1, I));
      add(gemv_name(linear_shape(d, l, model::BoltLinearId::Fc2)));
      add(gemv_name(linear_shape(d, l, model::BoltLinearId::SharedDown)));
    }
  }
  const Pending last = d.layers ? pending_of(d, d.layers - 1) : Pending{};
  add(res_norm_variant(1, last.s1, last.s2, false));   // norm_f folds the last block's slices
  if (peer) add(pp_handoff_variant());
  return v;
}

// ==== Spec 25d: Nemotron 3.5 Bolt prefill (begin) ======================================================================
// One chunk of at most kPfC rows through a device's blocks on the Level Zero list (runtime/bolt/bolt_prefill.cc has the
// walk). The Bolt-only prefill binaries are nb_pf_* (src/kernels/bolt/nb_pf_{mamba,moe,moe_gemm,prep}.cl); 25c's per-row
// kernels are built again at M = kPfC (the M in the name is the prefill line: the norm sites `_NF` / `_S1` / `_S1_T1` -
// every prefill linear is one unsplit GEMM, so S 1 whatever decode's S -, the router, the route, the shared expert's act,
// the attention prep at S 1 - decode's chains row for row); the rest are reused sources at Bolt's shapes: prefill/
// pf_dequant_slab.cl (a layout-1 linear of whole 1024-column slabs), 18c's k2/k2_pf_linear.cl (a layout-0 one, one
// zero-padded tail slab where N is not whole slabs: in_proj 18560, q||k||v 4608, fc1||shared_up 6400), the shape-free
// pf_gemm_T0 and Muse's mg_pf_flash_attn full (NoPE) build and its eager twin (Decision 15).
inline constexpr unsigned kPfC = 2048, kPfSub = 64, kPfTm = 32, kPfSlab = 1024;
inline constexpr unsigned kPfInLd = 18688;                        // pad256(18560): in_proj's prefill rows' pitch
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;        // one weight batch of bf16 expert blocks (Kolibri's)
inline constexpr unsigned kPfSsdW = 16, kPfConvRows = 16;         // nb_pf_ssd's dims a work-group, nb_pf_conv's rows a
                                                                  // work-item (PROVISIONAL: r61.speed's phase table)
inline constexpr unsigned kPfWg = 256, kPfDtWg = 128, kPfCbDpasWg = 64, kPfGatherWg = 64, kPfDequantWg = 16;
inline constexpr unsigned kPfFlashRpw = 8, kPfFlashHpw = 4;       // mg_pf_flash_attn's rows / heads a work-group (Muse's)
// nb_pf_moe.cl's header words (H_*): [0] tiles used, [1] 0, [2] rows used, [3] C, [4 + e] expert e's rows (e < 512).
namespace pf_hdr {
inline constexpr unsigned kTiles = 0, kRows = 2, kC = 3, kCount = 4;
inline constexpr unsigned words() { return (kCount + kMaxExperts + 15) / 16 * 16; }   // 528
}  // namespace pf_hdr
// The chunk's ABSOLUTE 64-row sub-chunks (nb_pf_mamba.cl): ceil((pos % 64 + C) / 64) - 33 at most.
inline constexpr unsigned pf_nsub(unsigned pos, unsigned C) { return (pos % kPfSub + C + kPfSub - 1) / kPfSub; }
inline constexpr unsigned kPfMaxSub = (kPfSub - 1 + kPfC + kPfSub - 1) / kPfSub;

// nb_pf_mamba.cl: seed, conv, dt, the plain C B^T, the scan, the gated norm - one binary per tiling and state precision.
inline std::string pf_mamba_variant(bool bf16_state, unsigned ssd_w = kPfSsdW, unsigned conv_rows = kPfConvRows) {
  return "nb_pf_mamba_L64_W" + std::to_string(ssd_w) + "_R" + std::to_string(conv_rows) + (bf16_state ? "_B16" : "");
}
// nb_pf_mamba.cl's CB_DPAS build: nb_pf_cb by DPAS (the walk's C B^T; its plain twin is the binary above).
inline std::string pf_cb_dpas_variant() { return "nb_pf_cb_L64_DPAS"; }
inline std::string pf_moe_variant(unsigned I) { return "nb_pf_moe_I" + std::to_string(I) + "_L1024_T9_TM32"; }
inline std::string pf_up_gemm_variant(unsigned I) { return "nb_pf_moe_gemm_K1024_N" + std::to_string(I) + "_RELU2"; }
inline std::string pf_down_gemm_variant(unsigned I) { return "nb_pf_moe_gemm_K" + std::to_string(I) + "_N1024"; }
inline std::string pf_embed_variant() { return "nb_pf_embed_gather_K4096_V131072"; }
inline std::string pf_norm_variant(unsigned S1, unsigned S2) { return res_norm_variant(kPfC, S1, S2, false); }
inline std::string pf_router_variant() { return router_variant(kPfC); }
inline std::string pf_route_variant() { return route_variant(kPfC); }
inline std::string pf_sh_act_variant() { return sh_act_variant(kPfC, 1); }
inline std::string pf_attn_prep_variant() { return attn_prep_variant(kPfC, 1); }
inline std::string pf_flash_variant(bool eager) { return std::string("mg_pf_flash_attn_Q32KV2_F") + (eager ? "_EAGER" : ""); }
inline std::string pf_gemm_t0_variant() { return pf_gemm_variant(false); }
// One int4 linear's slab binary at ITS layout (the descriptor's {K, N, layout}): whole 1024-column slabs -> pf_dequant_slab
// at that layout; a layout-0 linear with a tail -> 18c's k2_pf_dequant_slab (one zero-padded tail slab of pad256 columns).
// "" when neither covers it (layout 1 with a tail): the walk refuses it naming the linear (a P0 re-pick to that cell needs
// its kernel first).
inline std::string pf_slab_variant(const model::GemvShape& s) {
  if (s.N % kPfSlab == 0) return pf_dequant_slab_variant(s.K, s.N, s.layout);
  if (s.layout == 0) return "k2_pf_dequant_slab_K" + std::to_string(s.K) + "_N" + std::to_string(s.N);
  return "";
}
inline bool pf_slab_tail(const model::GemvShape& s) { return s.N % kPfSlab != 0; }
inline const char* pf_slab_entry(const model::GemvShape& s) {
  return pf_slab_tail(s) ? "k2_pf_dequant_slab" : "pf_dequant_slab";
}
// The prefill rows' pitch of a linear's GEMM output: pad256(N) (the tail slab's padding columns land there).
inline constexpr unsigned pf_ld(unsigned N) { return (N + 255) / 256 * 256; }

// Every binary a prefill chunk of `d` binds for an attention form and an SSM state precision (the head over the last row
// is decode's: decode_variants). tests/kernels/bolt_variant_names_test.cc holds the union to the CMake block's list.
inline std::vector<std::string> prefill_variants(const model::BoltDesc& d, bool eager, bool bf16_state) {
  std::vector<std::string> v = {pf_embed_variant(), pf_gemm_t0_variant()};
  auto add = [&](const std::string& s) {
    for (const std::string& x : v)
      if (x == s) return;
    v.push_back(s);
  };
  for (uint32_t l = 0; l < d.layers; ++l) {
    const Pending p = l == 0 ? Pending{} : Pending{1, d.is_moe(l - 1) ? 1u : 0u};   // prefill folds S 1
    add(pf_norm_variant(p.s1, p.s2));
    for (const model::BoltLinear& ln : d.linears(l))
      if (ln.kind == model::WeightKind::Int4 && ln.id != model::BoltLinearId::ExpertUp &&
          ln.id != model::BoltLinearId::ExpertDown)
        add(pf_slab_variant(ln.shape));
    if (d.is_mamba(l)) {
      add(pf_mamba_variant(bf16_state));
      add(pf_cb_dpas_variant());
    } else if (d.is_attention(l)) {
      add(pf_attn_prep_variant());
      add(pf_flash_variant(eager));
    } else {
      const unsigned I = d.inter.at(l);
      add(pf_router_variant());
      add(pf_route_variant());
      add(pf_sh_act_variant());
      add(pf_moe_variant(I));
      add(pf_up_gemm_variant(I));
      add(pf_down_gemm_variant(I));
    }
  }
  return v;
}
// The prefill head over the last row (prefill_head): decode's norm at M 1 folding the last block's S 1 rows, the head, the
// two argmax stages.
inline std::vector<std::string> prefill_head_variants(const model::BoltDesc& d, bool int8_head) {
  const bool moe = d.layers && d.is_moe(d.layers - 1);
  return {res_norm_variant(1, 1, moe ? 1u : 0u, false), head_variant(int8_head), argmax1_variant(1), argmax2_variant()};
}
// ==== Spec 25d: Nemotron 3.5 Bolt prefill (end) ========================================================================

}  // namespace kernels::bolt
