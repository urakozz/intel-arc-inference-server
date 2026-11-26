#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "loader/inkling2_loader.h"

namespace runtime::inkling2 {

// `--parallel tp` (plan 41j): card `rank`'s weights. The one-card layout with each visit's q / gate / k / v / relative
// projections cut by rows (the card's 12 q and 4 kv heads), o_proj by columns (those heads' inputs: K 1,536 of each
// row's 3,072, the row's scale whole) and every expert by its width (576 of 1,152 intermediate columns: the gate /
// up block's n-tiles [72 rank, + 72), the down block's k-groups [9 rank, + 9) under every n-tile, the outer scale
// word whole). The drafter's layers the same way on their bf16 tiles ([n tile][k octet][8 k][16 n]): q / k / v, the
// context k / v and gate / up by rows (the card's 12 q / 4 kv heads, its 4,608 MLP columns), o_proj and down by K (a
// 16-row tile's k octets [rank K / 16, + K / 16)). Everything else - the embedding, the head and its 2-bit copy's
// source, the router, the controllers, the norms, the per-head scales and taps, the sinks' projection, the drafter's
// norms and Markov head (the drafter reads its kTpHeadHalf rows of the head and fold) - is whole on both cards; the
// engine offsets its pointers into the per-head tensors by the card's first head.
//
// The one-card checkpoint stream lands on the card through `pieces`: a byte range of a one-card arena and where it
// goes in the card's arena (whole tensors one piece, a row cut one piece a region, o_proj one a 16-row tile row, an
// expert block one a range of tiles). The card's codes and scales are one card's, byte for byte.
struct TpWeights {
  struct Piece {
    size_t src, dst, bytes;
  };
  loader::Inkling2WeightLayout layout;   // the card's: offsets, bytes, K and N at its widths
  std::array<std::vector<Piece>, loader::Inkling2WeightLayout::kDraft + 1> pieces;   // per arena, ascending src
};

TpWeights tp_weights(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& one, uint32_t rank)
    pre(rank < 2);

// A one-card write of `bytes` at (arena, offset): each piece it covers, as (the card's offset, its bytes).
template <class F>
void tp_write(const TpWeights& w, uint32_t arena, size_t offset, std::span<const std::byte> bytes, F f) {
  const auto& pieces = w.pieces[arena];
  const size_t end = offset + bytes.size();
  auto it = std::upper_bound(pieces.begin(), pieces.end(), offset,
                             [](size_t at, const TpWeights::Piece& p) { return at < p.src; });
  if (it != pieces.begin()) --it;
  for (; it != pieces.end() && it->src < end; ++it) {
    const size_t lo = std::max(offset, it->src), hi = std::min(end, it->src + it->bytes);
    if (lo < hi) f(it->dst + (lo - it->src), bytes.subspan(lo - offset, hi - lo));
  }
}

}  // namespace runtime::inkling2
