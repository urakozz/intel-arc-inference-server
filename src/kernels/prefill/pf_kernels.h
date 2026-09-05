#pragma once
#include <string>

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
namespace kernels {

inline std::string pf_embed_gather_variant() { return "pf_embed_gather"; }
inline std::string pf_res_fold_variant(unsigned K, unsigned SP, unsigned G) {
  return "pf_res_fold_K" + std::to_string(K) + "_SP" + std::to_string(SP) + "_G" +
         std::to_string(G);
}
inline std::string pf_norm_finish_variant(unsigned K, unsigned G, unsigned W) {
  return "pf_norm_finish_K" + std::to_string(K) + "_G" + std::to_string(G) + "_W" +
         std::to_string(W);
}
inline std::string pf_silu_mul_variant() { return "pf_silu_mul"; }
inline std::string pf_gated_head_variant() { return "pf_gated_head"; }
inline std::string pf_attn_prep_variant() { return "pf_attn_prep"; }
// The a||b projection, mirroring gemv_bf16's {COLS_PER_WG 16, KSPLIT 16}
// tiling so that at M = 1 it is BIT-IDENTICAL to the binary capture.cc binds.
inline std::string pf_ab_proj_variant() { return "pf_ab_proj"; }

}  // namespace kernels
