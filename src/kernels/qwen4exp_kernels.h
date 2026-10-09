#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "model/qwen4exp.h"

// Spec 21c: Qwen3.8-Flash-Next's decode binaries by name - the host half of the spec 21c block in
// src/kernels/CMakeLists.txt, in its own header so kernels.h (shared with every branch in flight) is not
// edited. Every family-only binary is prefixed `q4_` and carries in its name every define its code depends
// on, so a host / device disagreement names a binary that does not exist and throws at capture. The reused
// kernels keep their own names (gemv_variant, gemv_bf16_variant, gemv_i8w_variant, attn_prep_variant,
// gdn_step_variant, embed_gather_variant, argmax_*_variant - kernels.h).
//
// The shapes are Qwen3.8-Flash-Next's (model::qwen4exp(): hidden 2560 in 4 hyper-connection streams, low
// rank 320; GDN 16 / 48 heads x 128 (Qwen3.8's gdn_step); QSA 24 q / 2 kv heads x 256 with a 4 x 128 + 1 x
// 128 indexer, blocks of 4, the top 512; 512 experts top 10 x 640; PLE 16 heads x 160; vocab 248320). A
// capture checks its descriptor against them (runtime/qwen4exp/qwen4exp_capture.cc) - a synthetic
// checkpoint or --layers N differs only in its layer count.
//
// The source files (src/kernels/qwen4exp/):
//   q4_hc.cl              q4_hc_combine_norm, q4_hc_up_mix                    plain OpenCL C (the Mac runs it)
//   q4_ple.cl             q4_ple_gather, q4_ple_block, q4_ple_check           plain OpenCL C
//   q4_qsa.cl             q4_qsa_prep, q4_qsa_score, q4_qsa_select            plain OpenCL C
//   q4_qsa_attn.cl        q4_qsa_attn, q4_qsa_reduce (flash, sub-groups)       card only
//   q4_qsa_attn_eager.cl  q4_qsa_attn_eager (the reference's bf16 chain)      plain OpenCL C
//   q4_moe.cl             q4_route, q4_moe_gate_up, q4_moe_down               block reads under
//                                                                             cl_intel_subgroups, plain otherwise
// and prep.cl's prep_gated_head at -DGDN_GATE_SIGMOID=1 (the `_SIG` binary; every existing prep.cl binary's
// command line and preprocessed source unchanged - commit 9a67889's rule).
//
// Every sigmoid / SiLU in the q4_ kernels (and in prep_gated_head _SIG) is 1 / (1 + exp_torch(-x)) /
// x / (1 + exp_torch(-x)): exp_torch is Sleef's expf u10 step for step (torch's Vectorized<float>::exp,
// k2_attn_eager.cl's), so a kernel and its host twin (tests/kernels/qwen4exp_ref.h) agree bit for bit and
// both sit on torch's own value.
namespace kernels::qwen4exp {

// --- the family's shapes (model::qwen4exp(); the capture checks the descriptor against them) ------------------
inline constexpr unsigned kHidden = 2560, kHc = 4, kHcN = 10240, kHcLow = 320, kHcDownN = 336;
inline constexpr unsigned kQHeads = 24, kKvHeads = 2, kHd = 256, kQkvgN = 13312, kQN = 6144;
inline constexpr unsigned kIdxHeads = 4, kIdxDim = 128, kIdxN = 640, kBlock = 4, kTopBlocks = 512;
inline constexpr unsigned kExperts = 512, kTopK = 10, kInter = 640, kRouterN = 528;
inline constexpr unsigned kPleHeads = 16, kPleDim = 160, kPleKvN = 12800;
inline constexpr unsigned kVocab = 248320, kVocabUsed = 248077, kPleEos = 248044;
inline constexpr unsigned kGdnHeads = 48, kGdnHd = 128, kQkvzN = 16384, kGdnZN = 6144, kAbN = 128;

// --- the device-side constants -----------------------------------------------------------------------------
inline constexpr unsigned kTailSlots = 8;     // the open block's raw keys, slot p % 8 (Review Focus 3: 4 would let
                                              // a 4-row launch overwrite a completing row's keys; spec 21 §4.2 said 4)
inline constexpr unsigned kListMax = 2052;    // 2048 + 3 tail + 1: the positions a row may attend (<= 2051)
inline constexpr unsigned kListRow = 2064;    // u32 words of one row's list: positions [0, count), the count at
                                              // word kCountWord, padding to 64 B (8256 B)
inline constexpr unsigned kCountWord = kListMax;
inline constexpr unsigned kPleRing = 16;      // the PLE id ring and conv ring: slot p % 16 (rows p - 3, p - 6, p - 9)
inline constexpr unsigned kScoreWg = 256, kSelectWg = 1024, kAttnTgt = 32, kRouteLanes = 256;
inline constexpr unsigned kHcWg = 256, kUpMixWg = 64, kPrepWg = 128, kPleWg = 160, kPleBlockWg = 256;
inline constexpr unsigned kAttnWg = 256, kEagerWg = 256, kPleCheckWg = 64;
inline constexpr unsigned kUpKs = 4, kDnKs = 2;   // q4_moe K splits, PROVISIONAL (Task 7 sweeps)
inline constexpr unsigned kArgmaxChunk = 1024, kArgmaxWg = 256, kEmbedWg = 256;
inline constexpr unsigned kSlots = kTopK + 1;     // 10 routed + the shared expert (slot 10)
inline constexpr unsigned kAttnPart = kHd + 2;    // {mx, sm, acc[256]} per (q head, slice, row)
inline constexpr unsigned kPleConsts = 3 + 2 * kPleHeads;   // u64: multipliers [3], sizes [16], offsets [16]

inline constexpr unsigned moe_gate_up_wg() { return 64 * kUpKs; }                         // 256
inline constexpr unsigned moe_down_wg() { return 16 * kSlots * kDnKs; }                   // 352
inline constexpr unsigned moe_gate_up_groups() { return kSlots * (2 * kInter / 16 / 4); }   // 220
inline constexpr unsigned argmax_groups() { return (kVocab + kArgmaxChunk - 1) / kArgmaxChunk; }   // 243

// The route row q4_route writes and the MoE kernels read (q4_moe.cl R_*), u32 words per (layer, row).
namespace route {
inline constexpr unsigned kWords = 32;
inline constexpr unsigned kIds = 0;         // 10 ids in RANK order (descending p, ties to the lower id): the
                                            // combine's order (grouped_mm's topk slot order, spec 21 §12)
inline constexpr unsigned kWeights = 16;    // fp32 bits of the bf16 renormalised weight, slot order
inline constexpr unsigned kSharedGate = 26; // fp32 bits of rne(sigmoid(rne(shared gate logit)))
inline constexpr unsigned kP10 = 27, kP11 = 28;   // diagnostics: p of rank 9 and of rank 10 (the first not taken)
}  // namespace route

// What a q4_hc_combine_norm folds into H before the norm (spec 21 §4.1: the pending H = H0 + y (x) inj of
// the block before, vLLM's fusion).
//   Embed   layer 0: H = the embedding row repeated into the 4 streams (nothing pending)
//   Slices  the mixer's GEMV split-K slices: y = rne(sum_s slices[s]) (S 4 int4, S 1 bf16)
//   Y       the MoE block's bf16 output
//   None    nothing pending (H materialised: after the PLE block, or the hand-off's landing)
enum class HcSrc { Embed, Slices, Y, None };
inline std::string m_(unsigned M) { return "_M" + std::to_string(M); }
// "q4_hc_combine_norm_M1_E" | "_S4" | "_S1" | "_Y" | "_X", + "_NN" without the norm (combine only).
inline std::string hc_combine_norm_variant(unsigned M, HcSrc src, unsigned S, bool norm) {
  std::string v = "q4_hc_combine_norm" + m_(M);
  switch (src) {
    case HcSrc::Embed: v += "_E"; break;
    case HcSrc::Slices: v += "_S" + std::to_string(S); break;
    case HcSrc::Y: v += "_Y"; break;
    case HcSrc::None: v += "_X"; break;
  }
  return norm ? v : v + "_NN";
}
// "q4_hc_up_mix_M1_I" (a gated residual: 320 + 4 inject rows) | "q4_hc_up_mix_M1" (the final mixer).
inline std::string hc_up_mix_variant(unsigned M, bool inject) { return "q4_hc_up_mix" + m_(M) + (inject ? "_I" : ""); }
// q4_ple.cl: the gather by the table's scale dtype (decision 7: bf16 the default file), the block, the check.
inline std::string ple_gather_variant(unsigned M, bool bf16_scale) {
  return "q4_ple_gather" + m_(M) + (bf16_scale ? "_BF16" : "_F32");
}
inline std::string ple_block_variant(unsigned M) { return "q4_ple_block" + m_(M); }
// q4_ple_check lives in every q4_ple binary; the engine binds the gather's.
inline std::string ple_check_variant(bool bf16_scale) { return ple_gather_variant(1, bf16_scale); }
// q4_qsa.cl's three kernels are one binary per M (prep, score, select: attn.cl's arrangement).
inline std::string qsa_variant(unsigned M) {
  return "q4_qsa" + m_(M) + "_T" + std::to_string(kTopBlocks) + "_W" + std::to_string(kSelectWg);
}
inline std::string qsa_prep_variant(unsigned M) { return qsa_variant(M); }
inline std::string qsa_score_variant(unsigned M) { return qsa_variant(M); }
inline std::string qsa_select_variant(unsigned M) { return qsa_variant(M); }
// "q4_qsa_attn_M1_T32" (q4_qsa_attn + q4_qsa_reduce) | "q4_qsa_attn_eager_M1" (one kernel, B70_Q4_ATTN=eager).
inline std::string qsa_attn_variant(unsigned M, bool eager) {
  return eager ? "q4_qsa_attn_eager" + m_(M) : "q4_qsa_attn" + m_(M) + "_T" + std::to_string(kAttnTgt);
}
inline std::string route_variant(unsigned M) { return "q4_route" + m_(M) + "_E512_T10_N528_L256"; }
// The shared expert's form: int4 layout-1 blocks (ours: _SH4) or bf16 gemv_bf16 tiles (Intel's: _SHB).
inline std::string moe_variant(unsigned M, bool shared_bf16) {
  return "q4_moe" + m_(M) + "_E512_T10_D2560_I640" + (shared_bf16 ? "_SHB" : "_SH4");
}
inline std::string gated_head_sig_variant(unsigned M) { return "prep_gated_head" + m_(M) + "_SIG"; }

// The reused kernels at this family's shapes.
inline std::string embed_variant(unsigned M) { return embed_gather_variant(M, kHidden); }   // embed_gather_M1_D2560
inline std::string argmax1_variant(unsigned M) { return argmax_stage1_variant(M, kVocabUsed); }   // argmax_stage1_M1
inline std::string argmax2_variant() { return argmax_stage2_variant(); }
inline std::string gdn_variant(unsigned M) { return gdn_step_variant(M); }   // Qwen3.8's gdn_step_M1 (16 / 48 heads)
// S: the fused q||gate||k||v GEMV's split-K (2 in the int4 arm, 1 in the bf16 arm: gemv_bf16 writes [M][N]).
inline std::string attn_prep_q4_variant(unsigned M, unsigned S) {
  return S == 1 ? attn_prep_s1_variant(M, kQHeads, kKvHeads) : attn_prep_variant(M, kQHeads, kKvHeads);
}
inline std::string bf16_variant(unsigned K, unsigned N) { return gemv_bf16_variant(1, K, N, gemv_bf16_tiling(N)); }
inline std::string hc_down_variant(bool inject) { return bf16_variant(kHcN, inject ? kHcDownN : kHcLow); }
inline std::string idx_variant() { return bf16_variant(kHidden, kIdxN); }
inline std::string router_variant() { return bf16_variant(kHidden, kRouterN); }
inline std::string ab_variant() { return bf16_variant(kHidden, kAbN); }
inline std::string ple_kv_variant() { return bf16_variant(kHidden, kPleKvN); }
inline std::string lm_head_variant(bool int8) {
  return int8 ? gemv_i8w_variant(1, kHidden, kVocab) : bf16_variant(kHidden, kVocab);
}
// The dense arms (model::Qwen4ExpDesc's {S, layout}: int4 layout 0 qkv||z S 1, q||gate||k||v S 2, out / o S 4
// PROVISIONAL; or Intel's bf16 tiles).
inline std::string qkvz_variant(bool int4) { return int4 ? gemv_variant(1, kHidden, kQkvzN, 1, 0) : bf16_variant(kHidden, kQkvzN); }
inline std::string qkvg_variant(bool int4, unsigned S = 2) {
  return int4 ? gemv_variant(1, kHidden, kQkvgN, S, 0) : bf16_variant(kHidden, kQkvgN);
}
inline std::string out_variant(bool int4, unsigned S = 4) {   // GDN out_proj and QSA o_proj: both 6144 x 2560
  return int4 ? gemv_variant(1, kGdnZN, kHidden, S, 0) : bf16_variant(kGdnZN, kHidden);
}
// Decision 10's evidence only (qwen4exp_kernels_test --bench-attn): attention v2 at the QSA shape, dense.
inline std::string attn_v2_q4_variant(unsigned M) { return attn_v2_variant(M, kAttnTgt, kQHeads, kKvHeads); }

// Every binary an M = 1 decode list binds, for a dense arm (int4 / bf16), a shared-expert form, a head
// form, an attention form (`eager`: B70_Q4_ATTN=eager binds q4_qsa_attn_eager for q4_qsa_attn + reduce) and
// the PLE scale form. Two-card lists add nothing new (the materialising _Y_NN is in every list: the PLE
// layer's prologue) but pp_handoff under --pipeline-handoff peer (16b's binary). Capture checks each exists
// before appending a command; tests/kernels/qwen4exp_variant_names_test.cc holds the union to the CMake
// block's list.
inline std::vector<std::string> decode_variants(bool int4_dense, bool shared_bf16, bool int8_head, bool eager,
                                                bool ple_bf16_scale = true) {
  const unsigned S = int4_dense ? 4u : 1u;
  std::vector<std::string> v = {
      embed_variant(1),
      hc_combine_norm_variant(1, HcSrc::Embed, 0, true),
      hc_combine_norm_variant(1, HcSrc::Slices, S, true),
      hc_combine_norm_variant(1, HcSrc::Y, 0, true),
      hc_combine_norm_variant(1, HcSrc::None, 0, true),
      hc_combine_norm_variant(1, HcSrc::Y, 0, false),
      hc_down_variant(true),
      hc_down_variant(false),
      hc_up_mix_variant(1, true),
      hc_up_mix_variant(1, false),
      qkvz_variant(int4_dense),
      ab_variant(),
      gdn_variant(1),
      gated_head_sig_variant(1),
      out_variant(int4_dense),
      qkvg_variant(int4_dense),
      idx_variant(),
      attn_prep_q4_variant(1, int4_dense ? 2u : 1u),
      qsa_variant(1),
      qsa_attn_variant(1, eager),
      router_variant(),
      route_variant(1),
      moe_variant(1, shared_bf16),
      ple_gather_variant(1, ple_bf16_scale),
      ple_block_variant(1),
      ple_kv_variant(),
      lm_head_variant(int8_head),
      argmax1_variant(1),
      argmax2_variant(),
  };
  return v;
}
// The descriptor's form (ours: int4 dense and shared; Intel's: bf16 both - 21b's q4_forms reads them all-or-
// nothing). `two_cards` binds nothing new (the materialising _Y_NN is every list's: the PLE prologue); peer adds
// 16b's pp_handoff, which is not this block's.
inline std::vector<std::string> decode_variants(const model::Qwen4ExpDesc& d, bool int8_head, bool eager, bool two_cards,
                                                bool ple_bf16_scale = true) {
  (void)two_cards;
  return decode_variants(d.forms.dense == model::Q4Form::Int4, d.forms.shared == model::Q4Form::Bf16, int8_head, eager,
                         ple_bf16_scale);
}

}  // namespace kernels::qwen4exp
