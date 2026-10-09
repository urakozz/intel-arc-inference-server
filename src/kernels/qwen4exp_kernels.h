#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
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

// =================================================================================================================
// Spec 21d: Qwen3.8-Flash-Next's PREFILL binaries by name - the host half of src/kernels/CMakeLists.txt's spec 21d
// block. A chunk of at most kPfC positions on the `l0` backend (runtime/qwen4exp/qwen4exp_prefill.h has the walk):
// the linears as slab GEMMs (prefill/pf_gemm.cl's pf_gemm_T0 over int4 dequant slabs or bf16 slabs), 21c's HC /
// indexer / PLE-gather sources rebuilt at M = kPfC, Qwen3.8's GDN chain (gdn_chunk_q4) with the sigmoid gated head,
// spec 6's flash at 24 / 2 heads for the rows with <= 2051 visible positions and q4_pf_sparse_attn over each row's
// own list past them, the MoE through a 512-expert ushort sort and spec 15d's grouped GEMMs. New sources
// (src/kernels/qwen4exp/):
//   q4_pf_moe.cl    q4_pf_sort, q4_pf_gather, q4_pf_dequant_{gu,dn}[_shb], q4_pf_moe_combine     the Mac runs it
//   q4_pf_attn.cl   q4_pf_sparse_attn (DPAS over a gathered list)                               card only
//   q4_pf_ple.cl    q4_pf_ple_gate, q4_pf_ple_conv, q4_pf_ple_ring                               plain OpenCL C
// and three existing sources gain a define each, every existing binary's preprocessed source token for token as it
// was (commit 9a67889's rule): prefill/pf_gated_head.cl GDN_GATE_SIGMOID (`pf_gated_head_SIG`), q4_qsa.cl QSA_PF
// (the M = kPfC prep without the in-launch ring write + q4_qsa_ring; `_PF`), q4_ple.cl PLE_PF (the gather's tokens
// from the chunk's id buffer, no ring write; `_PF`).
inline constexpr unsigned kPfC = 2048, kPfTm = 32, kPfSlab = 1024;
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;   // gate||up: 513 blocks of 6,553,600 B, 81 a batch, 7
                                                             // batches; down: 3,276,800 B, 163 a batch, 4 (derived)
inline constexpr unsigned kPfSortWg = 256, kPfGatherWg = 64, kPfCombineWg = 256, kPfDequantWg = 16;
inline constexpr unsigned kPfKt = 32, kPfSparseWg = 32;      // q4_pf_sparse_attn: list entries a tile, 2 sub-groups
inline constexpr unsigned kPfPleWg = 256, kPfRingWg = 128;   // q4_pf_ple_*; q4_qsa_ring (= kPrepWg)
inline constexpr unsigned kPfIdxLd = 768;                    // the indexer GEMM's row pitch: pf_ld(640)
inline constexpr unsigned kPfHcDownLd = 512;                 // the HC down||inject GEMM's: pf_ld(336)
inline constexpr unsigned kPfFlashRpw = 8, kPfFlashHpw = 6;  // pf_flash_attn_Q24KV2: rows a work-group, heads (GQA 12)
// The sort's header (q4_pf_moe.cl H_*): [0] tiles used, [1] the shared expert's first row, [2] rows used, [3] C,
// [4 + e] expert e's rows, [4 + 512] C; padded to 16 words.
namespace pf_hdr {
inline constexpr unsigned kTiles = 0, kSharedRow = 1, kRows = 2, kC = 3, kCount = 4;
inline constexpr unsigned words() { return (kCount + kExperts + 1 + 15) / 16 * 16; }   // 528
}  // namespace pf_hdr

inline std::string pf_moe_variant() {   // "q4_pf_moe_E512_T10_D2560_I640_L256"
  return "q4_pf_moe_E" + std::to_string(kExperts) + "_T" + std::to_string(kTopK) + "_D" + std::to_string(kHidden) +
         "_I" + std::to_string(kInter) + "_L" + std::to_string(kPfSortWg);
}
inline std::string pf_sparse_attn_variant(bool eager) {   // "q4_pf_sparse_attn_Q24KV2" | "_EAGER"
  return std::string("q4_pf_sparse_attn_Q") + std::to_string(kQHeads) + "KV" + std::to_string(kKvHeads) +
         (eager ? "_EAGER" : "");
}
inline std::string pf_ple_variant() { return "q4_pf_ple_C" + std::to_string(kPfC); }   // "q4_pf_ple_C2048"
// kolibri/kol_pf_linear.cl's bf16 slab at this family's shapes (the source's own name carries Kolibri's prefix).
inline std::string pf_bf16_slab_variant(unsigned K, unsigned N) {
  return "q4_pf_bf16_slab_K" + std::to_string(K) + "_N" + std::to_string(N);
}
// The int4 (layout 0) slabs: whole 1024-column slabs through prefill/pf_dequant_slab.cl (qkv||z, q||gate||k||v), a
// zero-padded tail through k2/k2_pf_linear.cl (out_proj / o_proj: 2560 = 1024 + 1024 + 512).
inline bool pf_int4_slab_tail(unsigned N) { return N % kPfSlab != 0; }
inline std::string pf_int4_slab_variant(unsigned K, unsigned N) {
  return pf_int4_slab_tail(N) ? "k2_pf_dequant_slab_K" + std::to_string(K) + "_N" + std::to_string(N)
                              : pf_dequant_slab_variant(K, N, 0);
}
// 21c's sources at M = kPfC.
inline std::string pf_hc_combine_norm_variant(HcSrc src, bool norm) { return hc_combine_norm_variant(kPfC, src, 1, norm); }
inline std::string pf_hc_up_mix_variant() {   // "q4_hc_up_mix_M2048_I_D512": UP_DOWN = the GEMM's pitch
  return hc_up_mix_variant(kPfC, true) + "_D" + std::to_string(kPfHcDownLd);
}
inline std::string pf_qsa_variant() { return qsa_variant(kPfC) + "_PF"; }   // "q4_qsa_M2048_T512_W1024_PF"
inline std::string pf_ple_gather_variant(bool bf16_scale) { return ple_gather_variant(kPfC, bf16_scale) + "_PF"; }
// The reused prefill binaries at this family's shapes.
inline std::string pf_embed_variant() { return pf_embed_gather_variant(kHidden); }          // pf_embed_gather_D2560
inline std::string pf_ab_variant() { return pf_ab_proj_variant(kHidden); }                  // pf_ab_proj_D2560
inline std::string pf_router_variant() { return pf_moe_router_variant(kHidden, kRouterN); }  // pf_moe_router_K2560_N528
inline std::string pf_gemm_gu_variant() { return pf_moe_gemm_variant(kHidden, 2 * kInter, false, true); }
inline std::string pf_gemm_dn_variant() { return pf_moe_gemm_variant(kInter, kHidden, false, false); }
inline std::string pf_gated_head_sig_variant() { return pf_gated_head_variant() + "_SIG"; }   // pf_gated_head_SIG
inline std::string pf_attn_prep_q4_variant() { return pf_attn_prep_q16_variant(kQHeads, kKvHeads); }
inline std::string pf_flash_q4_variant() { return pf_flash_attn_variant(kQHeads, kKvHeads); }
inline std::string pf_gate_q4_variant() { return pf_attn_variant(kQHeads, kKvHeads); }

// Every binary a prefill chunk binds (runtime/qwen4exp/qwen4exp_prefill.cc), for a dense arm (int4 / bf16), an
// attention form and the PLE scale form (the shared expert's form is an entry point of q4_pf_moe, not a binary). The
// last row's head is decode's binaries (decode_variants: the capture checks them). Two cards add nothing (the chunk
// crosses by copy).
inline std::vector<std::string> prefill_variants(bool int4_dense, bool eager, bool ple_bf16_scale = true) {
  std::vector<std::string> v = {
      pf_embed_variant(),
      pf_hc_combine_norm_variant(HcSrc::Embed, true),
      pf_hc_combine_norm_variant(HcSrc::Slices, true),
      pf_hc_combine_norm_variant(HcSrc::Y, true),
      pf_hc_combine_norm_variant(HcSrc::None, true),
      pf_hc_combine_norm_variant(HcSrc::Y, false),
      pf_bf16_slab_variant(kHcN, kHcDownN),
      pf_gemm_variant(false),
      pf_hc_up_mix_variant(),
      pf_ab_variant(),
      pf_gdn_conv_variant(),
      pf_gdn_wy_variant(),
      pf_gdn_scan_variant(),
      pf_gated_head_sig_variant(),
      pf_bf16_slab_variant(kHidden, kIdxN),
      pf_attn_prep_q4_variant(),
      pf_qsa_variant(),
      pf_flash_q4_variant(),
      pf_sparse_attn_variant(eager),
      pf_gate_q4_variant(),
      pf_router_variant(),
      route_variant(1),
      pf_moe_variant(),
      pf_gemm_gu_variant(),
      pf_gemm_dn_variant(),
      pf_ple_gather_variant(ple_bf16_scale),
      pf_bf16_slab_variant(kHidden, kPleKvN),
      pf_ple_variant(),
  };
  if (int4_dense) {
    v.push_back(pf_int4_slab_variant(kHidden, kQkvzN));
    v.push_back(pf_int4_slab_variant(kHidden, kQkvgN));
    v.push_back(pf_int4_slab_variant(kGdnZN, kHidden));
  } else {
    v.push_back(pf_bf16_slab_variant(kHidden, kQkvzN));
    v.push_back(pf_bf16_slab_variant(kHidden, kQkvgN));
    v.push_back(pf_bf16_slab_variant(kGdnZN, kHidden));
  }
  return v;
}
inline std::vector<std::string> prefill_variants(const model::Qwen4ExpDesc& d, bool eager, bool ple_bf16_scale = true) {
  return prefill_variants(d.forms.dense == model::Q4Form::Int4, eager, ple_bf16_scale);
}

// =================================================================================================================
// Spec 21e: Qwen3.8-Flash-Next's MTP head and spec 8's verify lists by name - the host half of src/kernels/
// CMakeLists.txt's spec 21e block (B70_Q4EXP and B70_MTP). What the lists bind (runtime/qwen4exp/qwen4exp_capture.cc):
//   verify at M = k + 1 (1..4), every device: 21c's decode list at M rows - every 21c kernel rebuilt at M = 2..4 (the
//     q4_ sources, attn_prep _Q24KV2 / _S1, prep_gated_head _SIG, embed_gather D2560, the GEMVs at this family's
//     shapes) - with the GDN step `gdn_step_slots_M<M>_G1` (spec 8's state slots, SPEC_SLOT_STRIDE = ONE layer's
//     state: the engine lays gdn_spec out layer-major, so the stride does not bake a layer count - Qwen3.8's
//     `gdn_step_slots_M<M>` bakes its 48-layer slot) and, on the PLE layer at M > 1, q4_pf_ple (21d's three-launch
//     cut of q4_ple_block, which is M = 1 only) at C = 4; argmax_stage1_M<M> is Qwen3.8's (vocab 248077 used);
//   the head's KV pass in every verify list (last device): q4_mtp (norm, fuse) at M, fc_embedding gemv_bf16 {2560,
//     2560} at M and fc_hidden at M = 4 a row, the attn side's gated residual (_X: H materialised by the fuse), the
//     head's bf16 q||gate||k||v and indexer at M, attn_prep _Q24KV2_S1 and q4_qsa at M;
//   a draft step (M = 1): q4_mtp_M1, the fc's, the head's QSA layer (bf16 dense, int4 experts, bf16 shared: _SHB) and
//     its final mixer, lm_head, argmax - every 21c M = 1 binary but q4_mtp and the fc's;
//   the prefill's head pass (M = kPfC): q4_mtp _PF (the chunk's ids), the fc's as bf16 slabs {2560, 2560}, then 21d's
//     binaries (combine _X / _Y_NN, the HC down slab, up_mix, the k||v slab of q||gate||k||v, the indexer slab,
//     pf_attn_prep_q16, q4_qsa _PF).
// The norm form of pre_fc_norm_hidden (decision 4) is a binary: `_SINGLE` (one RMS over the row's 10240, vLLM's) or
// `_STREAM` (per stream), B70_Q4_MTP_NORM.
inline constexpr unsigned kMtpVerifyRows = 4;   // = runtime::qwen4exp::kVerifyRows
inline constexpr unsigned kMtpWg = 256;
// "q4_mtp_M1_SINGLE" | "_STREAM", + "_PF" at M = kPfC (the prefill's head pass: tokens from the chunk's ids).
inline std::string mtp_variant(unsigned M, bool single, bool pf = false) {
  return "q4_mtp" + m_(M) + (single ? "_SINGLE" : "_STREAM") + (pf ? "_PF" : "");
}
inline std::string gdn_slots_variant(unsigned M) { return gdn_step_slots_variant(M, 1); }   // gdn_step_slots_M<M>_G1
inline std::string pf_ple_verify_variant() { return "q4_pf_ple_C" + std::to_string(kMtpVerifyRows); }   // q4_pf_ple_C4
inline std::string bf16_m_variant(unsigned M, unsigned K, unsigned N) { return gemv_bf16_variant(M, K, N, gemv_bf16_tiling(N)); }
inline std::string fc_variant(unsigned M) { return bf16_m_variant(M, kHidden, kHidden); }   // gemv_bf16_M<M>_K2560_N2560
// The dense / head linears at M rows (21c's M = 1 helpers above are these at M = 1).
inline std::string qkvz_m_variant(unsigned M, bool int4) {
  return int4 ? gemv_variant(M, kHidden, kQkvzN, 1, 0) : bf16_m_variant(M, kHidden, kQkvzN);
}
inline std::string qkvg_m_variant(unsigned M, bool int4) {
  return int4 ? gemv_variant(M, kHidden, kQkvgN, 2, 0) : bf16_m_variant(M, kHidden, kQkvgN);
}
inline std::string out_m_variant(unsigned M, bool int4) {
  return int4 ? gemv_variant(M, kGdnZN, kHidden, 4, 0) : bf16_m_variant(M, kGdnZN, kHidden);
}
inline std::string lm_head_m_variant(unsigned M, bool int8) {
  return int8 ? gemv_i8w_variant(M, kHidden, kVocab) : bf16_m_variant(M, kHidden, kVocab);
}

// Every binary a verify list at M binds (one device or the last of two: the union), for the dense arm, the shared
// form, the head form, the attention form, the PLE scale form and the norm form. Two cards add nothing new (the
// materialising _Y_NN is every list's) but pp_handoff under peer (16b's binary).
inline std::vector<std::string> verify_variants(unsigned M, bool int4_dense, bool shared_bf16, bool int8_head, bool eager,
                                                bool ple_bf16_scale, bool single) {
  const unsigned S = int4_dense ? 4u : 1u;
  std::vector<std::string> v = {
      embed_variant(M),
      hc_combine_norm_variant(M, HcSrc::Embed, 0, true),
      hc_combine_norm_variant(M, HcSrc::Slices, S, true),
      hc_combine_norm_variant(M, HcSrc::Y, 0, true),
      hc_combine_norm_variant(M, HcSrc::None, 0, true),
      hc_combine_norm_variant(M, HcSrc::Y, 0, false),
      bf16_m_variant(M, kHcN, kHcDownN),
      bf16_m_variant(M, kHcN, kHcLow),
      hc_up_mix_variant(M, true),
      hc_up_mix_variant(M, false),
      qkvz_m_variant(M, int4_dense),
      bf16_m_variant(M, kHidden, kAbN),
      gdn_slots_variant(M),
      gated_head_sig_variant(M),
      out_m_variant(M, int4_dense),
      qkvg_m_variant(M, int4_dense),
      bf16_m_variant(M, kHidden, kIdxN),
      attn_prep_q4_variant(M, int4_dense ? 2u : 1u),
      qsa_variant(M),
      qsa_attn_variant(M, eager),
      bf16_m_variant(M, kHidden, kRouterN),
      route_variant(M),
      moe_variant(M, shared_bf16),
      ple_gather_variant(M, ple_bf16_scale),
      bf16_m_variant(M, kHidden, kPleKvN),
      M == 1 ? ple_block_variant(1) : pf_ple_verify_variant(),
      lm_head_m_variant(M, int8_head),
      argmax1_variant(M),
      argmax2_variant(),
      // the head's KV pass
      mtp_variant(M, single),
      fc_variant(M),
      fc_variant(kMtpVerifyRows),
      qkvg_m_variant(M, false),
      attn_prep_q4_variant(M, 1u),
  };
  return v;
}
inline std::vector<std::string> verify_variants(const model::Qwen4ExpDesc& d, unsigned M, bool int8_head, bool eager,
                                                bool ple_bf16_scale, bool single) {
  return verify_variants(M, d.forms.dense == model::Q4Form::Int4, d.forms.shared == model::Q4Form::Bf16, int8_head, eager,
                         ple_bf16_scale, single);
}
// Every binary a draft step binds (the last device, M = 1): the head's layer is bf16 dense and bf16 shared
// (loader::q4_mtp_desc) with int4 experts, whatever the main model's form.
inline std::vector<std::string> draft_variants(bool int8_head, bool eager, bool single) {
  return {mtp_variant(1, single),
          fc_variant(1),
          fc_variant(kMtpVerifyRows),
          hc_combine_norm_variant(1, HcSrc::None, 0, true),
          hc_combine_norm_variant(1, HcSrc::Slices, 1, true),
          hc_combine_norm_variant(1, HcSrc::Y, 0, true),
          hc_down_variant(true),
          hc_down_variant(false),
          hc_up_mix_variant(1, true),
          hc_up_mix_variant(1, false),
          qkvg_variant(false),
          idx_variant(),
          attn_prep_q4_variant(1, 1u),
          qsa_variant(1),
          qsa_attn_variant(1, eager),
          out_variant(false),
          router_variant(),
          route_variant(1),
          moe_variant(1, true),
          lm_head_variant(int8_head),
          argmax1_variant(1),
          argmax2_variant()};
}
// Every binary the prefill's head pass binds (the last device, a chunk).
inline std::vector<std::string> mtp_prefill_variants(bool single) {
  return {mtp_variant(kPfC, single, true),
          pf_bf16_slab_variant(kHidden, kHidden),
          pf_gemm_variant(false),
          pf_hc_combine_norm_variant(HcSrc::None, true),
          pf_hc_combine_norm_variant(HcSrc::Y, false),
          pf_bf16_slab_variant(kHcN, kHcDownN),
          pf_hc_up_mix_variant(),
          pf_bf16_slab_variant(kHidden, kQkvgN),
          pf_bf16_slab_variant(kHidden, kIdxN),
          pf_attn_prep_q4_variant(),
          pf_qsa_variant(),
          hc_combine_norm_variant(1, HcSrc::None, 0, true)};   // prefill_head's final mixer over R (H materialised)
}

}  // namespace kernels::qwen4exp
