#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"   // spec 18c: the reused prefill names
#include "model/k2_horizon.h"

// Spec 18b: K2-Horizon's device binaries by name - the host half of the K2 block in
// src/kernels/CMakeLists.txt, kept in its own header so kernels.h (shared with every other
// branch in flight) is not edited. Every K2-only binary is prefixed `k2_` and carries in
// its name every define its code depends on, so a host / device disagreement names a
// binary that does not exist and throws at capture. The reused kernels keep their own
// names: gemv_variant / gemv_bf16_variant / gemv_i8w_variant / prep_res_fold_variant
// (kernels.h) at K2's shapes - they bake K and N into the name already.
namespace kernels::k2 {

// The K2 block's work-group geometry (src/kernels/CMakeLists.txt passes the same numbers).
inline constexpr unsigned kNormG = 20;      // prep_res_fold's FOLD_G / k2_norm_finish's NORM_G
inline constexpr unsigned kNormW = 20;      // k2_norm_finish's work-groups
inline constexpr unsigned kNormWg = 256;
inline constexpr unsigned kSiluWg = 256, kSiluChunk = 4096;
inline constexpr unsigned kUpKs = 4, kDnKs = 2;   // k2_moe_gate_up / k2_moe_down K splits
inline constexpr unsigned kMovaKs = 4;            // k2_mova_value's K split
inline constexpr unsigned kAttnTgt = 32;          // k2_attn_decode's work-groups per kv head
inline constexpr unsigned kAttnWg = 128;          // = head_dim
inline constexpr unsigned kAttnPart = 130;        // {mx, sm, acc[128]}
inline constexpr unsigned kArgmaxChunk = 1024, kArgmaxWg = 256;
inline constexpr unsigned kEmbedWg = 256;

inline constexpr unsigned moe_gate_up_wg() { return 64 * kUpKs; }
inline constexpr unsigned moe_down_wg(unsigned top_k) { return 16 * (top_k + 1) * kDnKs; }
inline constexpr unsigned moe_gate_up_groups(unsigned top_k, unsigned inter) {
  return (top_k + 1) * (2 * inter / 16 / 4);
}
inline constexpr unsigned mova_wg(unsigned top_k) { return 16 * top_k * kMovaKs; }

// The route row k2_route writes and the expert kernels read (k2_moe.cl R_*).
namespace route {
inline constexpr unsigned kWords = 32;
inline constexpr unsigned kIds = 0;       // u32 expert id of slot j, ASCENDING id
inline constexpr unsigned kWeights = 8;   // fp32 bits: w_j, a bf16 value
inline constexpr unsigned kSel = 16;      // fp32 bits: sel_j = s_j + bias
inline constexpr unsigned kNext = 24;     // fp32 bits: sel of rank top_k (first not taken)
inline constexpr unsigned kSum = 25;      // fp32 bits: Σ s of the top-k, rank order
}  // namespace route

inline std::string norm_variant(unsigned M, unsigned K, unsigned G, unsigned W, unsigned groups) {
  return "k2_norm_M" + std::to_string(M) + "_K" + std::to_string(K) + "_G" + std::to_string(G) +
         "_W" + std::to_string(W) + "_NG" + std::to_string(groups);
}
inline std::string silu_variant(unsigned M, unsigned I, unsigned S) {
  return "k2_silu_M" + std::to_string(M) + "_I" + std::to_string(I) + "_S" + std::to_string(S);
}
// `v`: the dense layers' build, which also writes v from the fused row (V_FROM_PARTIALS).
inline std::string attn_prep_variant(unsigned M, unsigned n, unsigned S, unsigned q, unsigned kv,
                                     bool v) {
  return "k2_attn_prep_M" + std::to_string(M) + "_N" + std::to_string(n) + "_S" +
         std::to_string(S) + "_Q" + std::to_string(q) + "KV" + std::to_string(kv) + (v ? "_V" : "");
}
inline std::string attn_variant(unsigned M, unsigned T, unsigned q, unsigned kv) {
  return "k2_attn_M" + std::to_string(M) + "_T" + std::to_string(T) + "_Q" + std::to_string(q) +
         "KV" + std::to_string(kv);
}
// k2_attn_eager.cl at the same defines (B70_K2_ATTN=eager; runtime::k2::K2Attn).
inline std::string attn_eager_variant(unsigned M, unsigned T, unsigned q, unsigned kv) {
  return "k2_attn_eager_M" + std::to_string(M) + "_T" + std::to_string(T) + "_Q" +
         std::to_string(q) + "KV" + std::to_string(kv);
}
// The router over `experts` of `top_k`, reading `ls` slices of a [ls][M][ln] fp32 output
// from column `loff`.
inline std::string route_variant(unsigned M, unsigned experts, unsigned top_k, unsigned ln,
                                 unsigned loff, unsigned ls) {
  return "k2_route_M" + std::to_string(M) + "_E" + std::to_string(experts) + "_T" +
         std::to_string(top_k) + "_N" + std::to_string(ln) + "_O" + std::to_string(loff) + "_S" +
         std::to_string(ls);
}
inline std::string moe_variant(unsigned M, unsigned experts, unsigned top_k, unsigned hidden,
                               unsigned inter) {
  return "k2_moe_M" + std::to_string(M) + "_E" + std::to_string(experts) + "_T" +
         std::to_string(top_k) + "_D" + std::to_string(hidden) + "_I" + std::to_string(inter);
}
inline std::string mova_variant(unsigned M, unsigned experts, unsigned top_k, unsigned hidden,
                                unsigned n) {
  return "k2_mova_M" + std::to_string(M) + "_E" + std::to_string(experts) + "_T" +
         std::to_string(top_k) + "_D" + std::to_string(hidden) + "_N" + std::to_string(n);
}
// embed_gather.cl / argmax.cl at K2's vocabulary (their names elsewhere do not carry VOCAB).
inline std::string embed_variant(unsigned M, unsigned hidden, unsigned vocab) {
  return "k2_embed_gather_M" + std::to_string(M) + "_D" + std::to_string(hidden) + "_V" +
         std::to_string(vocab);
}
inline std::string argmax1_variant(unsigned M, unsigned vocab, unsigned used) {
  return "k2_argmax_stage1_M" + std::to_string(M) + "_N" + std::to_string(vocab) + "_V" +
         std::to_string(used);
}
inline std::string argmax2_variant(unsigned vocab) {
  return "k2_argmax_stage2_N" + std::to_string(vocab);
}
inline constexpr unsigned argmax_groups(unsigned vocab) { return (vocab + kArgmaxChunk - 1) / kArgmaxChunk; }

// Every binary K2's M = 1 decode list binds (runtime/k2/k2_capture.cc forms the same names
// at its binding sites), for a head form and an attention form (`eager_attn`:
// B70_K2_ATTN=eager binds k2_attn_eager.cl's four kernels for k2_attn.cl's two): capture
// checks each exists before appending a command, and tests/kernels/k2_variant_names_test.cc
// holds the list to the CMake block's.
inline std::vector<std::string> decode_variants(const model::K2Desc& d, bool int8_head,
                                                bool eager_attn = false) {
  const auto lin = [&](model::K2LinearId id) {
    const model::GemvShape s = d.linear(id).shape;
    return gemv_variant(1, s.K, s.N, s.S, s.layout);
  };
  const auto bf16 = [&](unsigned K, unsigned N) {
    return gemv_bf16_variant(1, K, N, gemv_bf16_tiling(N));
  };
  return {
      embed_variant(1, d.hidden, d.vocab),
      prep_res_fold_variant(1, d.hidden, 0, kNormG),
      prep_res_fold_variant(1, d.hidden, d.down_s, kNormG),
      prep_res_fold_variant(1, d.hidden, d.oproj_s, kNormG),
      norm_variant(1, d.hidden, kNormG, kNormW, d.norm_groups),
      lin(model::K2LinearId::AttnDense),
      lin(model::K2LinearId::AttnSparse),
      lin(model::K2LinearId::OProj),
      lin(model::K2LinearId::DenseGateUp),
      lin(model::K2LinearId::DenseDown),
      silu_variant(1, d.dense_inter, d.gate_up_s),
      attn_prep_variant(1, d.attn_dense_n(), d.attn_s, d.q_heads, d.kv_heads, true),
      attn_prep_variant(1, d.attn_sparse_n(), d.attn_s, d.q_heads, d.kv_heads, false),
      eager_attn ? attn_eager_variant(1, kAttnTgt, d.q_heads, d.kv_heads)
                 : attn_variant(1, kAttnTgt, d.q_heads, d.kv_heads),
      route_variant(1, d.value_experts, d.value_top_k, d.attn_sparse_n(), d.v_off(), d.attn_s),
      mova_variant(1, d.value_experts, d.value_top_k, d.hidden, d.kv_n()),
      bf16(d.hidden, d.router_n()),
      route_variant(1, d.experts, d.top_k, d.router_n(), 0, 1),
      moe_variant(1, d.experts, d.top_k, d.hidden, d.moe_inter),
      int8_head ? gemv_i8w_variant(1, d.hidden, d.vocab) : bf16(d.hidden, d.vocab),
      argmax1_variant(1, d.vocab, d.vocab_used),
      argmax2_variant(d.vocab),
  };
}

// ==== Spec 18c: K2-Horizon prefill ==========================================================
// The prefill chunk's binaries (runtime/k2/k2_prefill.cc binds them; src/kernels/CMakeLists.txt's
// spec 18c block builds them). The reused sources keep their own names where those already
// carry every shape define (pf_res_fold_variant, pf_moe_router_variant, pf_moe_gemm_variant,
// pf_gemm_variant / pf_gemm_silu_variant, kernels.h / prefill/pf_kernels.h); the K2-only
// kernels and the reused ones whose names would not carry K2's shape are `k2_pf_*`.
inline constexpr unsigned kPfC = 2048;     // the chunk's rows (k2_pf_moe.cl KC; = PrefillScratchDims::kC)
inline constexpr unsigned kPfTm = 32;      // the grouped GEMM's tile rows (pf_moe_gemm.cl TM)
inline constexpr unsigned kPfSlab = 1024;  // a whole slab's columns (k2_pf_dequant_slab's ns)
inline constexpr unsigned kPfFlashKt = 64, kPfFlashRpw = 8;   // k2_pf_attn.cl's KT / RPW
inline constexpr unsigned kPfGatherWg = 64, kPfCombineWg = 256;
inline constexpr unsigned kPfFoldWg = 256;   // pf_prep.cl's pf_res_fold

// pf_embed.cl at K2's hidden and vocabulary (pf_embed_gather_variant's name carries no vocab).
inline std::string pf_embed_variant(unsigned hidden, unsigned vocab) {
  return "k2_pf_embed_gather_D" + std::to_string(hidden) + "_V" + std::to_string(vocab);
}
// k2_pf_linear.cl: one layout-0 linear's bf16 slabs, any width (the tail zero-padded).
inline std::string pf_slab_variant(unsigned K, unsigned N) {
  return "k2_pf_dequant_slab_K" + std::to_string(K) + "_N" + std::to_string(N);
}
// k2_prep.cl's stage B at M = kPfC: pf_res_fold (runtime M) writes sumsq [G][kPfC] when the
// walk passes m_count = kPfC, which is the layout this build reads.
inline std::string pf_norm_variant(unsigned K, unsigned G, unsigned W, unsigned groups) {
  return norm_variant(kPfC, K, G, W, groups);
}
// k2_prep.cl's attention prep at M = kPfC over the prefill GEMM's ONE slice (S 1), the
// partials row at pitch `ld` (pad256 of the fused row: 10240 dense, 9472 MoVA).
inline std::string pf_attn_prep_variant(unsigned ld, unsigned q, unsigned kv, bool v) {
  return attn_prep_variant(kPfC, ld, 1, q, kv, v);
}
// k2_moe.cl's router over [C][ln] fp32 logits from column `loff`, one slice: launched on
// grid (1, C), row m reads logits row m (LS 1 makes the binary's M irrelevant). The MoE one
// (ln = router_n, loff 0) IS decode's binary.
inline std::string pf_route_variant(unsigned experts, unsigned top_k, unsigned ln, unsigned loff) {
  return route_variant(1, experts, top_k, ln, loff, 1);
}
// k2_pf_moe.cl's two binaries: the MoE block's (sort, gather, gate||up / down dequant, the
// ascending-id combine + shared + residual) and MoVA's (sort, gather, value dequant, the
// SiLU combine into V).
inline std::string pf_moe_variant(unsigned experts, unsigned top_k, unsigned hidden, unsigned inter) {
  return "k2_pf_moe_E" + std::to_string(experts) + "_T" + std::to_string(top_k) + "_D" +
         std::to_string(hidden) + "_I" + std::to_string(inter);
}
inline std::string pf_mova_variant(unsigned experts, unsigned top_k, unsigned hidden, unsigned n) {
  return "k2_pf_mova_E" + std::to_string(experts) + "_T" + std::to_string(top_k) + "_D" +
         std::to_string(hidden) + "_N" + std::to_string(n);
}
// k2_pf_attn.cl: the flash attention with the softplus gate fused (`gated`), or the ungated
// o / l in fp32 (the K1 test's form); `eager` the reference-rounding variant (B70_K2_ATTN).
inline std::string pf_flash_variant(unsigned q, unsigned kv, bool eager, bool gated) {
  return "k2_pf_flash_attn_Q" + std::to_string(q) + "KV" + std::to_string(kv) +
         (eager ? "_EAGER" : "") + (gated ? "" : "_O");
}
inline constexpr unsigned pf_flash_wg(unsigned q, unsigned kv) {
  return 16 * (q / kv) * kPfFlashRpw / 8;   // HPW = the GQA group
}
// The sort's lanes: one per expert, a power of two (K2Desc::route_wg's rule).
inline unsigned pf_sort_wg(unsigned experts) { return model::K2Desc::route_wg(experts); }
// Header words (k2_pf_moe.cl H_*): 4 + experts + 1, padded to 16.
inline constexpr unsigned pf_hdr_words(unsigned experts) { return (4 + experts + 1 + 15) / 16 * 16; }
inline constexpr unsigned pad256(unsigned n) { return (n + 255) / 256 * 256; }

// Every binary the prefill walk binds (runtime/k2/k2_prefill.cc forms the same names at its
// binding sites) plus the head's (decode's own binaries, decode_variants), in one list for
// the variant-names test and the walk's pre-check. Both attention variants are listed.
inline std::vector<std::string> prefill_variants(const model::K2Desc& d) {
  const auto slab = [&](model::K2LinearId id) {
    const model::GemvShape s = d.linear(id).shape;
    return pf_slab_variant(s.K, s.N);
  };
  return {
      pf_embed_variant(d.hidden, d.vocab),
      pf_res_fold_variant(d.hidden, 0, kNormG),
      pf_res_fold_variant(d.hidden, 1, kNormG),
      pf_norm_variant(d.hidden, kNormG, kNormW, d.norm_groups),
      slab(model::K2LinearId::AttnDense),
      slab(model::K2LinearId::AttnSparse),
      slab(model::K2LinearId::OProj),
      slab(model::K2LinearId::DenseGateUp),
      slab(model::K2LinearId::DenseDown),
      pf_gemm_variant(false),
      pf_gemm_silu_variant(),
      pf_attn_prep_variant(pad256(d.attn_dense_n()), d.q_heads, d.kv_heads, true),
      pf_attn_prep_variant(pad256(d.attn_sparse_n()), d.q_heads, d.kv_heads, false),
      pf_flash_variant(d.q_heads, d.kv_heads, false, true),
      pf_flash_variant(d.q_heads, d.kv_heads, true, true),
      pf_route_variant(d.value_experts, d.value_top_k, pad256(d.attn_sparse_n()), d.v_off()),
      pf_mova_variant(d.value_experts, d.value_top_k, d.hidden, d.kv_n()),
      pf_moe_gemm_variant(d.hidden, d.kv_n(), false, false),
      pf_moe_router_variant(d.hidden, d.router_n()),
      pf_route_variant(d.experts, d.top_k, d.router_n(), 0),
      pf_moe_variant(d.experts, d.top_k, d.hidden, d.moe_inter),
      pf_moe_gemm_variant(d.hidden, 2 * d.moe_inter, false, true),
      pf_moe_gemm_variant(d.moe_inter, d.hidden, false, false),
  };
}
// The K1 test's extra builds (not bound by the walk): the ungated attention, both variants.
inline std::vector<std::string> prefill_test_variants(const model::K2Desc& d) {
  return {pf_flash_variant(d.q_heads, d.kv_heads, false, false),
          pf_flash_variant(d.q_heads, d.kv_heads, true, false)};
}
// ==== Spec 18c: K2-Horizon prefill (end) ====================================================

}  // namespace kernels::k2
