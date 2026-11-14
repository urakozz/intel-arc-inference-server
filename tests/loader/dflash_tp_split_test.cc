// Spec 19 v3c Task 2 Step 1: the DFlash drafter split over the two `--parallel tp` cards, on the host
// (loader/dflash_tp_split.h), and a card's plan with its half (runtime::tp_card_plan's drafter term).
//
//   split   every device form (bf16 tiles, int8 tiles + row scales, int4 g64 layout 1): a card's output rows are the
//           one-card weight's 16-row tiles of those rows in order (q||k||v by head, gate||up), its input rows the
//           k-tiles of every row tile (o, down) - so the column halves concatenate, tile for tile, to the one-card
//           weight and the row halves' tiles are disjoint and cover it; int8's row scales whole on both row halves; the
//           conv kernel projections whole on both cards.
//   plan    at 262,144 positions a card with the drafter's half fits a 32.53 GB card with the 1.5 GB reserve
//           (`--max-len auto` = 262,144), printed; a TP plan without a drafter is byte for byte what it was.
// Host only.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "loader/dflash_layout.h"
#include "loader/dflash_tp_split.h"
#include "model/dflash.h"
#include "model/model_desc.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/tp_plan.h"

namespace {

using loader::DFlashHostLinear;

struct Form {
  model::WeightKind kind;
  uint32_t kt, elems, bytes;   // a tile's k rows, its elements and an element's bytes
};
constexpr Form kForms[] = {{model::WeightKind::Bf16, 8, 128, 2},
                           {model::WeightKind::Int8, 16, 256, 1},
                           {model::WeightKind::Int4, 64, 136, 4}};

uint8_t* data(DFlashHostLinear& h) {
  return h.kind == model::WeightKind::Bf16   ? reinterpret_cast<uint8_t*>(h.bf16.data())
         : h.kind == model::WeightKind::Int8 ? reinterpret_cast<uint8_t*>(h.i8.data())
                                             : reinterpret_cast<uint8_t*>(h.int4.data());
}

// A linear whose every tile's bytes name it: tile (n, k) byte b = (n * 31 + k * 7 + b) mod 251.
DFlashHostLinear make(const Form& f, uint32_t K, uint32_t N) {
  DFlashHostLinear h;
  h.kind = f.kind;
  h.K = K;
  h.N = N;
  const size_t n = size_t(N / 16) * (K / f.kt) * f.elems;
  if (f.kind == model::WeightKind::Bf16) h.bf16.resize(n);
  if (f.kind == model::WeightKind::Int8) {
    h.i8.resize(n);
    for (uint32_t i = 0; i < N; ++i) h.scales.push_back(float(i) + 0.5f);
  }
  if (f.kind == model::WeightKind::Int4) h.int4.resize(n);
  const size_t tile = size_t(f.elems) * f.bytes;
  for (uint32_t nt = 0; nt < N / 16; ++nt)
    for (uint32_t kt = 0; kt < K / f.kt; ++kt)
      for (size_t b = 0; b < tile; ++b)
        data(h)[(size_t(nt) * (K / f.kt) + kt) * tile + b] = uint8_t((nt * 31 + kt * 7 + b) % 251);
  return h;
}
const uint8_t* tile(DFlashHostLinear& h, const Form& f, uint32_t nt, uint32_t kt) {
  return data(h) + (size_t(nt) * (h.K / f.kt) + kt) * f.elems * f.bytes;
}

void check_split() {
  // A small drafter: 4 q / 2 kv heads of 32, an MLP of 64, hidden 128 - every half whole 16-row tiles.
  model::DFlashDesc d;
  d.hidden = 128;
  d.q_heads = 4;
  d.kv_heads = 2;
  d.head_dim = 32;
  d.inter = 64;
  const model::DFlashDesc c = model::dflash_tp_card_desc(d);
  CHECK_EQ(c.q_heads, 2u);
  CHECK_EQ(c.kv_heads, 1u);
  CHECK_EQ(c.inter, 32u);
  for (const Form& f : kForms) {
    // q||k||v (column-parallel by head): card r's rows are q heads r, k head r, v head r.
    {
      DFlashHostLinear w = make(f, 128, d.qkv_n());   // q 128 | k 64 | v 64
      DFlashHostLinear h[2] = {loader::dflash_tp_layer_linear(d, loader::DFlashLinear::Qkv, 0, w),
                               loader::dflash_tp_layer_linear(d, loader::DFlashLinear::Qkv, 1, w)};
      for (uint32_t r = 0; r < 2; ++r) {
        CHECK_EQ(h[r].N, c.qkv_n());
        CHECK_EQ(h[r].K, 128u);
        const uint32_t rows[] = {r * 64, r * 64 + 16, r * 64 + 32, r * 64 + 48, 128 + r * 32, 128 + r * 32 + 16,
                                 192 + r * 32, 192 + r * 32 + 16};
        for (uint32_t i = 0; i < 8; ++i)
          for (uint32_t kt = 0; kt < 128 / f.kt; ++kt)
            CHECK(std::memcmp(tile(h[r], f, i, kt), tile(w, f, rows[i] / 16, kt), f.elems * f.bytes) == 0);
        if (f.kind == model::WeightKind::Int8)
          for (uint32_t i = 0; i < 8; ++i)
            for (uint32_t j = 0; j < 16; ++j) CHECK_EQ(h[r].scales[i * 16 + j], w.scales[rows[i] + j]);
      }
    }
    // gate||up: card r's gate rows then its up rows.
    {
      DFlashHostLinear w = make(f, 128, d.gate_up_n());
      for (uint32_t r = 0; r < 2; ++r) {
        DFlashHostLinear h = loader::dflash_tp_layer_linear(d, loader::DFlashLinear::GateUp, r, w);
        CHECK_EQ(h.N, c.gate_up_n());
        const uint32_t rows[] = {r * 32, r * 32 + 16, 64 + r * 32, 64 + r * 32 + 16};
        for (uint32_t i = 0; i < 4; ++i)
          for (uint32_t kt = 0; kt < 128 / f.kt; ++kt)
            CHECK(std::memcmp(tile(h, f, i, kt), tile(w, f, rows[i] / 16, kt), f.elems * f.bytes) == 0);
      }
    }
    // o and down (row-parallel): card r's k-tiles of every row tile, the whole row scales; the halves cover the
    // weight once.
    for (const auto which : {loader::DFlashLinear::O, loader::DFlashLinear::Down}) {
      const uint32_t K = which == loader::DFlashLinear::O ? 256 : 128;   // q_n 128 is below int4's 2 x 64: use 256
      DFlashHostLinear w = make(f, K, 128);
      model::DFlashDesc dd = d;
      if (which == loader::DFlashLinear::O) dd.q_heads = 8;   // q_n 256
      else dd.inter = 128;
      for (uint32_t r = 0; r < 2; ++r) {
        DFlashHostLinear h = loader::dflash_tp_layer_linear(dd, which, r, w);
        CHECK_EQ(h.K, K / 2);
        CHECK_EQ(h.N, 128u);
        for (uint32_t nt = 0; nt < 8; ++nt)
          for (uint32_t kt = 0; kt < K / 2 / f.kt; ++kt)
            CHECK(std::memcmp(tile(h, f, nt, kt), tile(w, f, nt, r * (K / 2 / f.kt) + kt), f.elems * f.bytes) == 0);
        CHECK(h.scales == w.scales);
        if (f.kind == model::WeightKind::Int4) CHECK(h.slices >= 1);
      }
    }
    // The conv kernel projections: whole on both cards.
    {
      DFlashHostLinear w = make(f, 128, 64);
      DFlashHostLinear h = loader::dflash_tp_layer_linear(d, loader::DFlashLinear::AttnKproj, 1, w);
      CHECK(h.bytes() == w.bytes() && std::memcmp(data(h), data(w), h.bytes() - h.scales.size() * 4) == 0);
    }
    std::printf("split: %s - q||k||v and gate||up by output tiles, o and down by input tiles, whole the rest\n",
                f.kind == model::WeightKind::Bf16 ? "bf16" : f.kind == model::WeightKind::Int8 ? "int8" : "int4 g64");
  }
}

void check_plan() {
  // z-lab/Qwen3.8-27B-DFlash2 (tests/model/dflash/qwen38_dflash2_config.json): 5 layers, 32 / 8 heads x 128,
  // intermediate 17408, the selector's rank 256 - one card's half of it beside one TP card at 262,144 positions.
  model::DFlashDesc d;
  d.version = 2;
  d.layers = 5;
  d.hidden = 5120;
  d.q_heads = 32;
  d.kv_heads = 8;
  d.head_dim = 128;
  d.inter = 17408;
  d.vocab = 248320;
  d.block = 8;
  d.taps = {1, 16, 31, 46, 61};
  d.window = {2048, 2048, 2048, 2048, 2048};
  d.causal = {false, false, false, false, false};
  d.conv_group = 16;
  d.conv_taps = 2;
  d.sel_rank = 256;
  d.sel_top_k = 16;
  const model::DFlashDesc half = model::dflash_tp_card_desc(d);
  const loader::DFlashBytes one = loader::dflash_weight_bytes(d, 5120, loader::DraftFormat::Int8);
  const loader::DFlashBytes card = loader::dflash_weight_bytes(half, 5120, loader::DraftFormat::Int8);
  std::printf("drafter weights, int8: one card %.3f GB, a TP card's half %.3f GB (linears %.3f, small %.3f, codebooks "
              "%.3f)\n",
              one.total() / 1e9, card.total() / 1e9, card.linears / 1e9, card.small / 1e9, card.codebooks / 1e9);
  CHECK(card.linears < one.linears && card.linears > one.linears / 2);   // fc, the conv projections and the selector whole
  const runtime::DrafterPlan dp = runtime::dflash::drafter_plan(half, 5120, 124160, loader::DraftFormat::Int8, true);
  const model::ModelDesc& tc = model::tp_card_desc(model::qwen38(), model::TpCrossing::Folded);
  runtime::PrefillPath pf;
  const size_t weights = size_t(11e9);   // tp_desc_test's card (above the measured slice: a margin)
  const runtime::MemoryPlan without = runtime::tp_card_plan(tc, model::TpCrossing::Folded, 262144, weights,
                                                            runtime::KvCache::Bf16, pf);
  const runtime::MemoryPlan none = runtime::tp_card_plan(tc, model::TpCrossing::Folded, 262144, weights,
                                                         runtime::KvCache::Bf16, pf, false, {});
  CHECK(without.total() == none.total() && without.decode_state == none.decode_state &&
        without.prefill_scratch == none.prefill_scratch && without.drafter == 0);
  const runtime::MemoryPlan with = runtime::tp_card_plan(tc, model::TpCrossing::Folded, 262144, weights,
                                                         runtime::KvCache::Bf16, pf, false, dp);
  std::printf("a TP card at 262144, bf16 KV, prefill on: without the drafter %.3f GB, with its half %.3f GB "
              "(drafter %.3f, slots %.3f, prefill taps %.3f)\n",
              without.total() / 1e9, with.total() / 1e9, with.drafter / 1e9, with.drafter_slots / 1e9,
              with.drafter_taps / 1e9);
  const size_t dev = size_t(32.53e9), reserve = size_t(1.5e9);
  CHECK(with.total() + reserve <= dev);
  const uint32_t len = runtime::tp_max_len_that_fits(tc, model::TpCrossing::Folded, {weights, weights}, {dev, dev},
                                                     reserve, 262144, runtime::KvCache::Bf16, pf, false, dp);
  std::printf("--max-len auto with the drafter's half: %u\n", len);
  CHECK_EQ(len, 262144u);
}

}  // namespace

int main() {
  check_split();
  check_plan();
  std::printf("dflash_tp_split_test: OK\n");
  return 0;
}
