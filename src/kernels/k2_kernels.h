#pragma once
#include <string>

#include "kernels/kernels.h"

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

}  // namespace kernels::k2
