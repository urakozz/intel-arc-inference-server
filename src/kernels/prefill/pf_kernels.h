#pragma once
#include <string>

#include "kernels/shape_suffix.h"

// Every prefill variant name (spec 2, plan 6b Tasks 3-5). `M` is NEVER in a
// name and never a `-D`: the prefill path takes the row count as a **runtime**
// kernel argument, grids are `ceil(M / tile)` and the tail tile masks itself
// on the device (interfaces.md, "Layout conventions"). What stays in a name is
// what the binary bakes as a stride or a grid: `K`, the previous GEMV's
// split-K width `S_PREV`, and the norm's group counts.
//
// **S_PREV is 0 or 1 here and nothing else.** Plan 6b ruling R1: the prefill
// path runs every linear at S = 1 -- decode's split-K exists to buy hardware
// threads at M = 1, and at M = C the M tile axis already saturates the grid.
// The consequence is that every consumer folds ONE slice, which is the whole
// reason these variants are cheaper than decode's widened naively (the
// small-kernel term was measured at decode's S and was 5x overstated;
// docs/probe-prefill-small-2026-09-05.md).
//
// Deliberately absent: `pf_gemv_variant` / `pf_gemv_int4_M`. Plan 6b Task 4's
// temporary int4 GEMV is SKIPPED by ruling -- plan 6c (L2) lands sycl-tla's
// real `gemm_bf16` + `dequant_to_bf16`, which is what will feed these kernels
// their fp32 `[M][N]` partials at S = 1 by construction.
//
// Spec 15b: the kernels that bake a MODEL shape their parameters do not name -
// hidden (pf_embed_gather, pf_ab_proj), the GDN heads (pf_gated_head and the
// pf_gdn_* chain), the FA heads (pf_attn_prep*, pf_attn, pf_flash_attn) - take
// it as arguments and append kernels/shape_suffix.h's suffix, which is empty
// at Qwen3.8's shapes (Agnes's too), so no existing binary is renamed. The
// defaults are those reference shapes; the runtime passes the descriptor's.
namespace kernels {

// One int4 linear expanded into the bf16 [K][N] scratch (P3's kernel).
// `TRANSPOSED` is 0 on the production path: `gemm_bf16` takes B row-major
// [K][N] with ldb = N (ruling A23/A24 keep the two-pass dequant->GEMM as THE
// GEMM path, so this is the only orientation the walk ever asks for; T1 stays
// built because the probe measures the contrast).
inline std::string pf_dequant_variant(unsigned K, unsigned N, unsigned layout,
                                      unsigned transposed = 0) {
  return "pf_dequant_tile_K" + std::to_string(K) + "_N" + std::to_string(N) + "_L" +
         std::to_string(layout) + "_T" + std::to_string(transposed);
}

inline std::string pf_embed_gather_variant(unsigned hidden = kRefHidden) {
  return "pf_embed_gather" + hidden_suffix(hidden);
}
inline std::string pf_res_fold_variant(unsigned K, unsigned SP, unsigned G) {
  return "pf_res_fold_K" + std::to_string(K) + "_SP" + std::to_string(SP) + "_G" +
         std::to_string(G);
}
inline std::string pf_norm_finish_variant(unsigned K, unsigned G, unsigned W) {
  return "pf_norm_finish_K" + std::to_string(K) + "_G" + std::to_string(G) + "_W" +
         std::to_string(W);
}
// Spec 14: SILU_N is baked (pf_prep.cl); `_I<intermediate>` for any width but
// Qwen3.8's 17408, exactly as kernels::prep_silu_mul_variant.
inline std::string pf_silu_mul_variant(unsigned I) {
  return std::string("pf_silu_mul") + (I == 17408 ? "" : "_I" + std::to_string(I));
}
// Spec 8 (plan 8b Task 3): the MTP head's prefill KV fill (src/kernels/prefill/CMakeLists.txt).
inline std::string pf_res_fold_zero_variant(unsigned K, unsigned G) {
  return pf_res_fold_variant(K, 1, G) + "_Z";
}
inline std::string pf_norm_finish_strided_variant(unsigned K, unsigned G, unsigned W, unsigned X) {
  return pf_norm_finish_variant(K, G, W) + "_X" + std::to_string(X);
}
inline std::string pf_bf16_slab_variant(unsigned K) { return "pf_bf16_slab_K" + std::to_string(K); }
inline std::string pf_gated_head_variant(unsigned gdn_k = kRefGdnKHeads,
                                         unsigned gdn_v = kRefGdnVHeads) {
  return "pf_gated_head" + gdn_suffix(gdn_k, gdn_v);
}
inline std::string pf_attn_prep_variant(unsigned q = kRefFaQHeads, unsigned kv = kRefFaKvHeads) {
  return "pf_attn_prep" + fa_suffix(q, kv);
}
// Ruling A9's bf16-`attn_q` build of the SAME source, bound by the composed
// attention path (ruling A14). It writes no `attn_gate`: `pf_attn_gate` reads
// the gate columns straight out of `qkv_partials`.
inline std::string pf_attn_prep_q16_variant(unsigned q = kRefFaQHeads,
                                            unsigned kv = kRefFaKvHeads) {
  return "pf_attn_prep_q16" + fa_suffix(q, kv);
}
// The composed path's own two kernels, one binary: `pf_softmax_causal` (the
// causal row softmax between the QK^T and PV GEMMs) and `pf_attn_gate` (the
// output gate that lived in decode's `attn_reduce`, ruling A16).
inline std::string pf_attn_variant(unsigned q = kRefFaQHeads, unsigned kv = kRefFaKvHeads) {
  return "pf_attn" + fa_suffix(q, kv);
}
// Spec 6: the fused flash attention (plan 6a's pfa_KT64_R16_H6_Q0), the default prefill
// attention on the L0 backends; the composed path above is B70_PREFILL_ATTN=composed.
// It bakes HPW = 6, the GQA group (q-heads per kv-head), so it carries the FA suffix.
inline std::string pf_flash_attn_variant(unsigned q = kRefFaQHeads, unsigned kv = kRefFaKvHeads) {
  return "pf_flash_attn" + fa_suffix(q, kv);
}
// Spec 12b (`--kv-cache int8`, src/kernels/kv8.cl): the prefill path over the int8 KV
// cache - the writer (pf_attn_prep_q16's chain, then rotate and quantise), the flash
// attention reading int8 + scales, and the gate that un-rotates first. Additive.
inline std::string pf_attn_prep_kv8_variant(unsigned q = kRefFaQHeads, unsigned kv = kRefFaKvHeads) {
  return "pf_attn_prep_kv8" + fa_suffix(q, kv);
}
inline std::string pf_flash_attn_kv8_variant(unsigned q = kRefFaQHeads,
                                             unsigned kv = kRefFaKvHeads) {
  return "pf_flash_attn_kv8" + fa_suffix(q, kv);
}
inline std::string pf_attn_gate_kv8_variant(unsigned q = kRefFaQHeads, unsigned kv = kRefFaKvHeads) {
  return "pf_attn_gate_kv8" + fa_suffix(q, kv);
}
// The a||b projection, mirroring gemv_bf16's {COLS_PER_WG 16, KSPLIT 16}
// tiling so that at M = 1 it is BIT-IDENTICAL to the binary capture.cc binds.
inline std::string pf_ab_proj_variant(unsigned hidden = kRefHidden) {
  return "pf_ab_proj" + hidden_suffix(hidden);
}

// --- gdn_chunk's nine kernels, three files (plan 6b Tasks 6-8) --------------
// One variant per .cl file, several entry points each, because they share the
// same literals (CONV_ROWS, HEADS, DIM, Q_OFF/K_OFF/V_OFF, RING) and a reader
// checking `pf_gdn_A`'s `i > j` mask against `pf_gdn_A2`'s `j <= i` should see
// both on one screen. `gdn_chunk`'s tenth launch is `pf_gated_head` above
// (ruling R3: the gated head is the GDN mixer's, not the caller's).
inline std::string pf_gdn_conv_variant(unsigned gdn_k = kRefGdnKHeads,
                                       unsigned gdn_v = kRefGdnVHeads) {
  return "pf_gdn_conv" + gdn_suffix(gdn_k, gdn_v);
}
inline std::string pf_gdn_wy_variant(unsigned gdn_k = kRefGdnKHeads,
                                     unsigned gdn_v = kRefGdnVHeads) {
  return "pf_gdn_wy" + gdn_suffix(gdn_k, gdn_v);
}
inline std::string pf_gdn_scan_variant(unsigned gdn_k = kRefGdnKHeads,
                                       unsigned gdn_v = kRefGdnVHeads) {
  return "pf_gdn_scan" + gdn_suffix(gdn_k, gdn_v);
}

// The execution-context probe (Task 2); bound by context_test.cc alone.
inline std::string pf_probe_chain_variant() { return "pf_probe_chain"; }

// Spec 2.1: the Level Zero prefill GEMM (pf_gemm.cl) and the slab dequant it pairs with.
inline std::string pf_gemm_variant(bool transB) { return transB ? "pf_gemm_T1" : "pf_gemm_T0"; }
// Parity program S2(a): the same GEMM with pf_silu_mul fused into its epilogue.
// Only gate||up is ever launched with it -- the fusion is a statement about that
// linear's INTERLEAVED gate/up column layout, not about the GEMM.
inline std::string pf_gemm_silu_variant() { return "pf_gemm_T0_SILU"; }
inline constexpr unsigned kPfSlabWidth = 1024;   // columns per pf_dequant_slab / slab GEMM
inline std::string pf_dequant_slab_variant(unsigned K, unsigned N, unsigned layout) {
  return "pf_dequant_slab_K" + std::to_string(K) + "_N" + std::to_string(N) + "_L" +
         std::to_string(layout);
}

// Spec 5: the int8 prefill linears (pf_int8.cl).
inline std::string pf_quant_had_variant(unsigned K) { return "pf_quant_had_K" + std::to_string(K); }
inline std::string pf_requant_rot_variant(unsigned layout) { return "pf_requant_rot_L" + std::to_string(layout); }
inline std::string pf_gemm_i8_variant(bool silu) { return silu ? "pf_gemm_i8_SILU" : "pf_gemm_i8"; }

}  // namespace kernels
