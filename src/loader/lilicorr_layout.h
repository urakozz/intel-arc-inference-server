#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/dflash_layout.h"
#include "model/dflash.h"

// Spec 26b Task 2: the LiLiCorr drafter's device bytes as one header-only formula, shared by the loader (which asserts
// its allocations add up to it) and the planner (runtime/dflash/lilicorr_sizes.h: the drafter's term in Bolt's last-device
// reserve, runtime::bolt::BoltReserve::drafter).
//
// **The backbone** is 19b's linears (loader/dflash_layout.h: fc, per layer q||k||v, o, gate||up, down, the two conv kernel
// projections - no selector) in the chosen format (Decision 2): Exact (W' bf16 tiles; an NVFP4 linear also carries its
// fp32 output scale per column, 4 N bytes), Int8 (spec 9's tiles + fp32 row scales fp32(g x s'_r)) or Int4G64 (rtn from
// g x W' in fp64, gemv.cl layout 1). Its small tensors (fp32): hidden_norm, the final norm, per layer input / post norms,
// q_norm || k_norm, the conv bases [layers][attn, mlp][side][tap][hidden], the sinks [layers][q heads], the mask row.
//
// **The LiLiCorr head** is exact in every arm (Decision 3): each of its linears as W'^T bf16 [K][N] (k-major - ll_gemm's
// columns are lanes) + an fp32 output scale [N] (its g, 1 for a BF16 linear) + an fp32 bias [N] (0 where none):
//   token_proj         lil_hidden x hidden            (absent when lil_hidden == hidden: the embedding rows as they are)
//   pass_hidden_proj   lil_hidden x hidden
//   feature_up         lil_hidden x 5;  feature_down lil_hidden x lil_hidden
//   context_proj       lil_hidden x hidden
//   per layer          in_proj 3 lil_hidden x lil_hidden, out_proj lil_hidden x lil_hidden, mlp up lil_mlp x lil_hidden,
//                      mlp down lil_hidden x lil_mlp
//   factor_input_proj  lil_hidden x 3 lil_hidden      (k rows [0, Hl) self, [Hl, 2 Hl) anchor, [2 Hl, 3 Hl) cross)
//   out_in_head        2 factor_dim x lil_hidden      (out_head's rows, then in_head's: one launch)
//   anchor_out_head    factor_dim x lil_hidden
// and its small tensors (one fp32 block, lili_head_small's offsets): feature_norm weight / bias [5] each, the slot and rank
// embeddings [slots][Hl] / [k][Hl], relative_slot_bias [heads][2 block - 1], same_slot_bias [heads], per layer attn_norm
// / mlp_norm [Hl] each, output_norm, anchor_norm.
namespace loader {

enum class LiliHead : uint32_t {
  TokenProj, PassHidden, FeatUp, FeatDown, ContextProj, InProj, OutProj, MlpUp, MlpDown, Factor, OutIn, AnchorOut
};
inline constexpr uint32_t kLiliHeadPerLayer = 4;   // InProj, OutProj, MlpUp, MlpDown

struct LiliShape {
  uint32_t K = 0, N = 0;
};
// One head linear's K x N (in_features x out_features).
inline LiliShape lili_head_shape(const model::DFlashDesc& d, LiliHead w) {
  const uint32_t H = d.hidden, Hl = d.lil_hidden, F = d.factor_dim;
  switch (w) {
    case LiliHead::TokenProj: return {H, Hl};
    case LiliHead::PassHidden: return {H, Hl};
    case LiliHead::FeatUp: return {5, Hl};
    case LiliHead::FeatDown: return {Hl, Hl};
    case LiliHead::ContextProj: return {H, Hl};
    case LiliHead::InProj: return {Hl, 3 * Hl};
    case LiliHead::OutProj: return {Hl, Hl};
    case LiliHead::MlpUp: return {Hl, d.lil_mlp};
    case LiliHead::MlpDown: return {d.lil_mlp, Hl};
    case LiliHead::Factor: return {3 * Hl, Hl};
    case LiliHead::OutIn: return {Hl, 2 * F};
    case LiliHead::AnchorOut: return {Hl, F};
  }
  return {};
}
inline bool lili_has_token_proj(const model::DFlashDesc& d) { return d.lil_hidden != d.hidden; }
// A head linear's device bytes: W'^T bf16 + the output scale + the bias.
inline size_t lili_head_linear_bytes(LiliShape s) { return size_t(s.K) * s.N * 2 + size_t(s.N) * 8; }

// The head's small fp32 block, offsets in floats.
struct LiliHeadSmall {
  size_t fn_w = 0, fn_b = 0, slot = 0, rank = 0, rel = 0, same = 0, norms = 0, out_norm = 0, anchor_norm = 0, total = 0;
  size_t attn_norm(uint32_t layer, uint32_t hl) const { return norms + size_t(layer) * 2 * hl; }
  size_t mlp_norm(uint32_t layer, uint32_t hl) const { return norms + size_t(layer) * 2 * hl + hl; }
};
inline LiliHeadSmall lili_head_small(const model::DFlashDesc& d) {
  LiliHeadSmall s;
  size_t at = 0;
  auto take = [&](size_t n) {
    const size_t o = at;
    at += n;
    return o;
  };
  s.fn_w = take(5);
  s.fn_b = take(5);
  s.slot = take(size_t(d.lil_slots()) * d.lil_hidden);
  s.rank = take(size_t(d.cand_topk) * d.lil_hidden);
  s.rel = take(size_t(d.lil_heads) * (2 * d.block - 1));
  s.same = take(d.lil_heads);
  s.norms = take(size_t(d.lil_layers) * 2 * d.lil_hidden);
  s.out_norm = take(d.lil_hidden);
  s.anchor_norm = take(d.lil_hidden);
  s.total = at;
  return s;
}

struct LiliBytes {
  size_t linears = 0;   // the backbone's linears (with an Exact NVFP4 linear's output scales)
  size_t small = 0;     // the backbone's norms, q/k norms, conv bases, sinks, the mask row
  size_t head = 0;      // the LiLiCorr head: its linears and its small block
  size_t total() const { return linears + small + head; }
};

// `nvfp4_rows`: the output rows of the backbone's NVFP4 linears (each carries 4 B of output scale a row in Exact) - the
// published checkpoint's fc + every layer's gate / up / down: hidden + layers x (2 inter + hidden).
inline uint64_t lili_nvfp4_rows_published(const model::DFlashDesc& d) {
  return uint64_t(d.hidden) + uint64_t(d.layers) * (2ull * d.inter + d.hidden);
}
inline LiliBytes lilicorr_weight_bytes(const model::DFlashDesc& d, uint32_t target_hidden, DraftFormat f,
                                       uint64_t nvfp4_rows) {
  LiliBytes b;
  const uint32_t H = d.hidden;
  b.linears += dflash_linear_bytes(d.fc_k(target_hidden), H, f);
  const size_t per_layer = dflash_linear_bytes(H, d.qkv_n(), f) + dflash_linear_bytes(d.q_n(), H, f) +
                           dflash_linear_bytes(H, d.gate_up_n(), f) + dflash_linear_bytes(d.inter, H, f) +
                           2 * dflash_linear_bytes(H, d.kproj_n(), f);
  b.linears += per_layer * d.layers;
  if (f == DraftFormat::Exact) b.linears += size_t(nvfp4_rows) * 4;
  b.small = size_t(H) * 4 * 2                                     // hidden_norm, final norm
            + size_t(d.layers) * 2 * H * 4                        // input / post norms
            + size_t(d.layers) * 2 * d.head_dim * 4               // q_norm || k_norm
            + size_t(d.layers) * 2 * 2 * d.conv_taps * H * 4      // conv bases
            + size_t(d.layers) * d.q_heads * 4                    // sinks
            + size_t(H) * 4;                                      // the mask row
  for (LiliHead w : {LiliHead::PassHidden, LiliHead::FeatUp, LiliHead::FeatDown, LiliHead::ContextProj, LiliHead::Factor,
                     LiliHead::OutIn, LiliHead::AnchorOut})
    b.head += lili_head_linear_bytes(lili_head_shape(d, w));
  if (lili_has_token_proj(d)) b.head += lili_head_linear_bytes(lili_head_shape(d, LiliHead::TokenProj));
  for (LiliHead w : {LiliHead::InProj, LiliHead::OutProj, LiliHead::MlpUp, LiliHead::MlpDown})
    b.head += size_t(d.lil_layers) * lili_head_linear_bytes(lili_head_shape(d, w));
  b.head += lili_head_small(d).total * 4;
  return b;
}

// Offsets in the sinks block [layers][q heads] (floats).
inline size_t lili_sink_off(const model::DFlashDesc& d, uint32_t layer) { return size_t(layer) * d.q_heads; }

}  // namespace loader
