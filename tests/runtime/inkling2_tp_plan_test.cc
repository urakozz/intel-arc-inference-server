// `--parallel tp`'s placement and plan (runtime/inkling2/inkling2_tp_split.h, inkling2_sizes.h): what each card holds
// of every tensor, where the one-card stream's bytes land on it (sampled elements of the cut tensors, by their
// layouts), each card's bytes, the all-reduces a token, and the context both cards hold.
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "loader/lm_head_int8.h"
#include "runtime/inkling2/inkling2_sizes.h"
#include "runtime/inkling2/inkling2_tp_split.h"

namespace {
namespace ink = runtime::inkling2;
using D = model::Inkling2Desc;
using L = loader::Inkling2WeightLayout;

// Where the one-card byte at (arena, offset) lands on the card: its offset, or SIZE_MAX when the card drops it.
size_t landing(const ink::TpWeights& w, uint32_t arena, size_t offset) {
  const std::byte b{};
  size_t at = SIZE_MAX;
  ink::tp_write(w, arena, offset, std::span(&b, 1), [&](size_t dst, std::span<const std::byte>) { at = dst; });
  return at;
}
}  // namespace

int main(int argc, char** argv) {
  CHECK_EQ(argc, 2);
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  const auto& d = checkpoint.desc();
  const auto& one = checkpoint.layout();
  const std::array<ink::TpWeights, 2> card{ink::tp_weights(d, one, 0), ink::tp_weights(d, one, 1)};

  // Every tensor on both cards; the cut ones at half their rows (q / gate / k / v / relative; the drafter's q / k / v,
  // context k / v, gate / up), K (o_proj; the drafter's o_proj and down) or expert width, the rest whole.
  size_t whole = 0, cut = 0;
  for (const auto& [name, w] : one.weights) {
    for (uint32_t r = 0; r < 2; ++r) {
      const auto& c = card[r].layout.weights.at(name);
      CHECK_EQ(c.arena, w.arena);
      const bool rows = name.ends_with("attn.wq_du.weight") || name.ends_with("attn.gate_attn.weight") ||
                        name.ends_with("attn.wk_dv.weight") || name.ends_with("attn.wv_dv.weight") ||
                        name.ends_with("attn.wr_du.weight");
      const bool layer = w.arena == L::kDraft && name.starts_with("layers.");
      const bool draft_rows = layer && (name.ends_with("_proj.weight") || name.ends_with("_proj_ctx.weight")) &&
                              !name.ends_with("o_proj.weight") && !name.ends_with("down_proj.weight");
      const bool draft_k = layer && (name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight"));
      if (rows) {
        CHECK(c.N * 2 == w.N && c.K == w.K && c.bytes * 2 == w.bytes && c.scales != 0);
      } else if (draft_rows) {
        CHECK(c.N * 2 == w.N && c.K == w.K && c.bytes * 2 == w.bytes && c.scales == 0);
      } else if (draft_k) {
        CHECK(c.K * 2 == w.K && c.N == w.N && c.bytes * 2 == w.bytes);
      } else if (name.ends_with("attn.wo_ud.weight")) {
        CHECK(c.K * 2 == w.K && c.N == w.N && c.bytes * 2 == w.bytes);
      } else if (name.ends_with("mlp.experts.w13_weight")) {
        CHECK(c.N * 2 == w.N && c.K == w.K && c.bytes < w.bytes);
      } else if (name.ends_with("mlp.experts.w2_weight")) {
        CHECK(c.K * 2 == w.K && c.N == w.N && c.bytes < w.bytes);
      } else {
        CHECK(c.N == w.N && c.K == w.K && c.bytes == w.bytes);
      }
      if (r == 0) (rows || draft_rows || draft_k || name.find("attn.wo_ud") != std::string::npos ||
                           name.find("mlp.experts") != std::string::npos
                       ? cut : whole) += w.bytes;
    }
  }
  CHECK(whole > 0 && cut > 0);
  CHECK_EQ(card[0].layout.bytes[L::kModel], one.bytes[L::kModel]);   // the embedding, head: whole on both cards
  CHECK_EQ(card[1].layout.bytes[L::kModel], one.bytes[L::kModel]);
  CHECK(card[0].layout.bytes[L::kDraft] == card[1].layout.bytes[L::kDraft] &&
        card[0].layout.bytes[L::kDraft] < one.bytes[L::kDraft]);
  std::printf("drafter: one card %.1f MB, a tp card %.1f MB\n", one.bytes[L::kDraft] / 1e6,
              card[0].layout.bytes[L::kDraft] / 1e6);
  for (uint32_t l = 0; l < D::layers; ++l) {
    const uint32_t a = L::kFirstLayer + l;
    CHECK(card[0].layout.bytes[a] == card[1].layout.bytes[a] && card[0].layout.bytes[a] < one.bytes[a]);
    std::printf("layer %2u: one card %.1f MB, a tp card %.1f MB\n", l, one.bytes[a] / 1e6, card[0].layout.bytes[a] / 1e6);
  }

  // Sampled elements of layer 3's cut tensors land where the card's layout reads them.
  const auto layer = model::inkling2_layer_prefix(3);
  for (uint32_t r = 0; r < 2; ++r) {
    const auto& w = card[r];
    // q: card row n is one-card row 1536 r + n (n-major 16 x 16 tiles: a row block is contiguous)
    const auto& q1 = one.weights.at(layer + "attn.wq_du.weight");
    const auto& qc = w.layout.weights.at(layer + "attn.wq_du.weight");
    for (const auto& [k, n] : {std::pair{0u, 0u}, {17u, 5u}, {2303u, 1535u}, {1000u, 777u}}) {
      const size_t src = q1.offset + loader::int8_tiled_index(q1.K, q1.N, k, n + 1536 * r);
      CHECK_EQ(landing(w, q1.arena, src), qc.offset + loader::int8_tiled_index(qc.K, qc.N, k, n));
      CHECK_EQ(landing(w, q1.arena, q1.scales + (n + 1536 * r) * 4), qc.scales + n * 4);
    }
    // o_proj: card element (k, n) is one-card (1536 r + k, n); every row's scale whole
    const auto& o1 = one.weights.at(layer + "attn.wo_ud.weight");
    const auto& oc = w.layout.weights.at(layer + "attn.wo_ud.weight");
    for (const auto& [k, n] : {std::pair{0u, 0u}, {15u, 16u}, {1535u, 2303u}, {700u, 1100u}}) {
      const size_t src = o1.offset + loader::int8_tiled_index(o1.K, o1.N, k + 1536 * r, n);
      CHECK_EQ(landing(w, o1.arena, src), oc.offset + loader::int8_tiled_index(oc.K, oc.N, k, n));
    }
    CHECK_EQ(landing(w, o1.arena, o1.scales + 2303 * 4), oc.scales + 2303 * 4);
    // the other card's rows / columns are not this card's
    const size_t other_q = q1.offset + loader::int8_tiled_index(q1.K, q1.N, 0, 1536 * (1 - r));
    CHECK_EQ(landing(w, q1.arena, other_q), SIZE_MAX);
    // experts: card n-tile t of the gate || up block is one-card n-tile 72 r + t; down's k-group g under n-tile t is
    // one-card k-group 9 r + g; the outer scale word follows the card's tiles
    const auto f = d.expert_format[3];
    const size_t tw = common::expert_fp_tile_words(f) * 4, outer = f == common::ExpertFpFormat::Nvfp4 ? 4 : 0;
    const auto& u1 = one.weights.at(layer + "mlp.experts.w13_weight");
    const auto& uc = w.layout.weights.at(layer + "mlp.experts.w13_weight");
    const size_t ub1 = 144 * 36 * tw + outer, ubc = 72 * 36 * tw + outer;
    for (const auto& [e, t, g] : {std::tuple{0u, 0u, 0u}, {5u, 71u, 35u}, {127u, 13u, 7u}}) {
      const size_t src = u1.offset + e * ub1 + ((72 * r + t) * 36 + g) * tw + 3;
      CHECK_EQ(landing(w, u1.arena, src), uc.offset + e * ubc + (t * 36 + g) * tw + 3);
    }
    if (outer) CHECK_EQ(landing(w, u1.arena, u1.offset + 9 * ub1 + 144 * 36 * tw), uc.offset + 9 * ubc + 72 * 36 * tw);
    const auto& d1 = one.weights.at(layer + "mlp.experts.w2_weight");
    const auto& dc = w.layout.weights.at(layer + "mlp.experts.w2_weight");
    const size_t db1 = 144 * 18 * tw + outer, dbc = 144 * 9 * tw + outer;
    for (const auto& [e, t, g] : {std::tuple{0u, 0u, 0u}, {64u, 143u, 8u}, {1u, 70u, 4u}}) {
      const size_t src = d1.offset + e * db1 + (t * 18 + 9 * r + g) * tw + 1;
      CHECK_EQ(landing(w, d1.arena, src), dc.offset + e * dbc + (t * 9 + g) * tw + 1);
    }
    // the drafter's layer 1: q card row n is one-card row 1536 r + n (bf16 [n tile][k octet][8][16]: a row block is
    // contiguous); o_proj card element (k, n) is one-card (1536 r + k, n)
    const auto bf16_at = [](uint32_t K, uint32_t k, uint32_t n) {
      return ((size_t(n / 16) * (K / 8) + k / 8) * 128 + (k % 8) * 16 + n % 16) * 2;
    };
    const auto& dq1 = one.weights.at("layers.1.self_attn.q_proj.weight");
    const auto& dqc = w.layout.weights.at("layers.1.self_attn.q_proj.weight");
    for (const auto& [k, n] : {std::pair{0u, 0u}, {17u, 5u}, {2303u, 1535u}})
      CHECK_EQ(landing(w, dq1.arena, dq1.offset + bf16_at(dq1.K, k, n + 1536 * r)), dqc.offset + bf16_at(dqc.K, k, n));
    const auto& do1 = one.weights.at("layers.1.mlp.down_proj.weight");
    const auto& doc = w.layout.weights.at("layers.1.mlp.down_proj.weight");
    for (const auto& [k, n] : {std::pair{0u, 0u}, {9u, 16u}, {4607u, 2303u}})
      CHECK_EQ(landing(w, do1.arena, do1.offset + bf16_at(do1.K, k + 4608 * r, n)), doc.offset + bf16_at(doc.K, k, n));
    // a whole tensor lands at its card offset unchanged
    const auto& n1 = one.weights.at(layer + "mlp_norm.weight");
    CHECK_EQ(landing(w, n1.arena, n1.offset + 8), w.layout.weights.at(layer + "mlp_norm.weight").offset + 8);
  }

  // Each card: half of every visit's caches, rings and k / v histories, the replicated rest whole; two all-reduces a
  // visit, 96 a token.
  constexpr uint32_t len = 32768, rows = 2048;
  const auto tp = ink::tp_sizes(d, one, len, true, rows, false, false);
  const auto solo = ink::sizes(d, one, len, true, rows);
  for (uint32_t r = 0; r < 2; ++r) {
    CHECK_EQ(tp[r].persistent.full_k * 2, solo.persistent.full_k);
    CHECK_EQ(tp[r].persistent.ring_k * 2, solo.persistent.ring_k);
    CHECK_EQ(tp[r].persistent.conv_k * 2, solo.persistent.conv_k);
    CHECK_EQ(tp[r].persistent.conv_attn, solo.persistent.conv_attn);
    CHECK_EQ(tp[r].draft.draft_full_k * 2, solo.draft.draft_full_k);
    CHECK_EQ(tp[r].draft.draft_ring_k * 2, solo.draft.draft_ring_k);
    CHECK_EQ(tp[r].draft.draft_tap, solo.draft.draft_tap);
    CHECK(tp[r].tp.tp_landing > 0 && tp[r].tp.tp_part == 8 * size_t(D::hidden) * 4);
    CHECK(tp[r].prefill_b.tp_landing == 2 * size_t(rows) * D::hidden * 4);
    CHECK(tp[r].memory().model < solo.memory().model);
    std::printf("card %u: weights %.2f GB (one card %.2f), total at %u positions %.2f GB (one card %.2f)\n", r,
                tp[r].memory().model / 1e9, solo.memory().model / 1e9, len, tp[r].memory().total() / 1e9,
                solo.memory().total() / 1e9);
  }
  CHECK_EQ(2 * D::visit_count, 96u);

  // The context: both cards hold more than one card does.
  constexpr size_t capacity = 32'000'000'000ull, reserve = 1'500'000'000ull;
  for (const bool kv8 : {false, true}) {
    const uint32_t solo_len = ink::max_len_that_fits(d, one, capacity, reserve, true, rows, kv8);
    const uint32_t tp_len = ink::tp_max_len_that_fits(d, one, capacity, reserve, true, rows, kv8);
    std::printf("max_len with the drafter, %s cache: one card %u, tp %u\n", kv8 ? "int8" : "bf16", solo_len, tp_len);
    CHECK(tp_len > solo_len);
  }
  std::printf("inkling2_tp_plan_test: PASS\n");
}
