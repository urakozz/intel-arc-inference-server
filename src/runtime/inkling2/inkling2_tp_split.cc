#include "runtime/inkling2/inkling2_tp_split.h"

#include <string>
#include <utility>

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;
using L = loader::Inkling2WeightLayout;

size_t aligned(size_t bytes) { return (bytes + 63) & ~size_t(63); }   // the loader's

// The rows a card keeps of a row-cut projection (its heads' outputs), 0 for a tensor that is not cut by rows.
uint32_t card_rows(const std::string& part) {
  if (part == "attn.wq_du.weight" || part == "attn.gate_attn.weight") return D::q_n / 2;
  if (part == "attn.wk_dv.weight" || part == "attn.wv_dv.weight") return D::kv_n / 2;
  if (part == "attn.wr_du.weight") return D::q_heads * D::relative_dim / 2;
  return 0;
}

// The drafter's: its layers' row-cut linears (q / k / v, the context k / v, gate / up) at the card's rows, 0 else.
uint32_t draft_rows(const std::string& name) {
  if (!name.starts_with("layers.")) return 0;
  if (name.ends_with(".q_proj.weight")) return D::q_n / 2;
  for (const char* n : {".k_proj.weight", ".v_proj.weight", ".k_proj_ctx.weight", ".v_proj_ctx.weight"})
    if (name.ends_with(n)) return D::kv_n / 2;
  if (name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight")) return D::draft_inter / 2;
  return 0;
}
}  // namespace

TpWeights tp_weights(const D& d, const L& one, uint32_t rank) {
  TpWeights out;
  std::vector<std::pair<const std::string*, const loader::Inkling2Weight*>> order;
  for (const auto& [name, w] : one.weights) order.push_back({&name, &w});
  std::ranges::sort(order, [](const auto& a, const auto& b) {
    return std::pair(a.second->arena, a.second->offset) < std::pair(b.second->arena, b.second->offset);
  });
  std::array<size_t, L::kDraft + 1> at{};
  for (const auto& [name, w] : order) {
    auto& c = at[w->arena];
    auto& pieces = out.pieces[w->arena];
    loader::Inkling2Weight cw = *w;
    cw.offset = c;
    const bool layer = w->arena >= L::kFirstLayer && w->arena < L::kDraft;
    const uint32_t l = w->arena - L::kFirstLayer;
    const std::string part = layer ? name->substr(model::inkling2_layer_prefix(l).size()) : "";
    const uint32_t rows = layer ? card_rows(part) : 0;
    // DSpark's drafter is split over the cards; the DFlash 2 drafter (plan 41k) is whole on each, as are its weights
    const bool draft = w->arena == L::kDraft && d.draft.form == model::DFlashForm::InklingDSpark;
    const uint32_t bf16_rows = draft ? draft_rows(*name) : 0;
    if (bf16_rows) {   // whole 16-row tiles: the card's rows one run
      contract_assert(w->N == 2 * bf16_rows && bf16_rows % 16 == 0 && !w->scales);
      cw.N = bf16_rows;
      cw.bytes = size_t(bf16_rows) * w->K * 2;
      pieces.push_back({w->offset + rank * cw.bytes, c, cw.bytes});
    } else if (draft && (name->ends_with(".o_proj.weight") || name->ends_with(".down_proj.weight"))) {
      // a 16-row tile is K / 8 octets of 256 bytes, the card's its [rank K / 16, + K / 16)
      const size_t k8 = w->K / 8, tile = 256;
      contract_assert(name->starts_with("layers.") && k8 % 2 == 0 && !w->scales);
      cw.K = w->K / 2;
      cw.bytes = w->bytes / 2;
      for (size_t b = 0; b < w->N / 16; ++b)
        pieces.push_back({w->offset + (b * k8 + rank * k8 / 2) * tile, c + b * k8 / 2 * tile, k8 / 2 * tile});
    } else if (rows) {
      contract_assert(w->N == 2 * rows && rows % 16 == 0 && w->scales);
      cw.N = rows;
      cw.bytes = size_t(rows) * w->K;
      pieces.push_back({w->offset + rank * cw.bytes, c, cw.bytes});
    } else if (part == "attn.wo_ud.weight") {
      // n-major 16 x 16 tiles: a 16-row tile row is K / 16 tiles, the card's its [rank K / 32, + K / 32)
      const size_t k16 = w->K / 16, tile = 256;
      cw.K = w->K / 2;
      cw.bytes = w->bytes / 2;
      for (size_t b = 0; b < w->N / 16; ++b)
        pieces.push_back({w->offset + (b * k16 + rank * k16 / 2) * tile, c + b * k16 / 2 * tile, k16 / 2 * tile});
    } else if (part == "mlp.experts.w13_weight" || part == "mlp.experts.w2_weight") {
      const auto f = d.expert_format[l];
      const size_t tw = common::expert_fp_tile_words(f) * 4, outer = f == common::ExpertFpFormat::Nvfp4 ? 4 : 0;
      const size_t nt = w->N / 16, g = w->K / 64, block = nt * g * tw + outer;
      const bool up = part == "mlp.experts.w13_weight";
      // gate || up: the card's n-tiles (its intermediate columns) are one run; down: under each n-tile its k-groups
      const size_t card_block = nt * g * tw / 2 + outer;
      if (up) cw.N = w->N / 2;
      else cw.K = w->K / 2;
      cw.bytes = D::experts * card_block;
      for (size_t e = 0; e < D::experts; ++e) {
        const size_t src = w->offset + e * block, dst = c + e * card_block;
        if (up) {
          pieces.push_back({src + rank * nt / 2 * g * tw, dst, nt / 2 * g * tw});
        } else {
          for (size_t n = 0; n < nt; ++n)
            pieces.push_back({src + (n * g + rank * g / 2) * tw, dst + n * g / 2 * tw, g / 2 * tw});
        }
        if (outer) pieces.push_back({src + nt * g * tw, dst + nt * g * tw / 2, outer});
      }
    } else {
      pieces.push_back({w->offset, c, w->bytes});
    }
    c = aligned(c + cw.bytes);
    if (w->scales) {
      cw.scales = c;
      const size_t bytes = size_t(cw.N) * sizeof(float);
      pieces.push_back({w->scales + (rows ? rank * bytes : 0), c, bytes});
      c = aligned(c + bytes);
    }
    out.layout.weights.emplace(*name, cw);
  }
  out.layout.bytes = at;
  for (auto& p : out.pieces) std::ranges::sort(p, {}, &TpWeights::Piece::src);
  return out;
}

}  // namespace runtime::inkling2
