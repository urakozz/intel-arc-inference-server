// Spec 17b (plan 17b's review, Task 1): how loader::load_tp cuts the checkpoint per card, and how TpEngine puts the two
// cards' state back into one card's host layout - host only.
//
//   columns  loader::tp_columns for every column-parallel linear: each card takes exactly its N columns, the two cards'
//            sets are disjoint and together every real one-card column; a card's columns are its own heads' (q, k, v,
//            z of v-heads 24 r .. 24 r + 23 / k-heads 8 r ..; q||gate of q-heads 12 r ..; kv-heads 2 r ..; its 16-column
//            gate / up blocks; its vocabulary half), in the card's fused order.
//   repack   the column map cut that way and the row-parallel pointer offset give tiles that ARE one-card tiles: a
//            synthetic GPTQ part repacked whole (layout 1 and layout 0) and per card - every card tile / word equals
//            the one-card tile / word it came from.
//   small    loader::tp_small_elems: the conv taps of the card's q / k / v channels, its v-heads' A_log / dt_bias,
//            everything else whole.
//   gather   runtime::tp::gather_state / gather_kv: a one-card state and KV split by the card rules, gathered back,
//            (spec 17d) scattered again, and the MTP head's KV layer spliced in / split out,
//            byte-identical (bf16 and int8 KV).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

#include "check.h"
#include "common/repack.h"
#include "loader/loader.h"
#include "model/model_desc.h"
#include "runtime/tp_gather.h"

namespace {
using model::LinearId;
constexpr uint32_t kG = model::Qwen35::kGdnHeadDim, kF = model::Qwen35::kFaHeadDim;

void check_columns(const model::ModelDesc& one, const model::ModelDesc& card, LinearId id, uint32_t real_n) {
  std::set<uint32_t> all;
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    const std::vector<uint32_t> c = loader::tp_columns(one, id, r);
    CHECK_EQ(uint32_t(c.size()), id == LinearId::AB ? card.linear(id).pad_n : card.shape(id).N);
    for (uint32_t x : c) CHECK(all.insert(x).second);   // disjoint across the cards
  }
  CHECK_EQ(uint32_t(all.size()), real_n);
  CHECK(*all.rbegin() == real_n - 1);
}

uint32_t word(uint32_t k8, uint32_t n) { return k8 * 2654435761u ^ n * 40503u; }
uint16_t scale(uint32_t g, uint32_t n) { return uint16_t(0x3C00 + ((g * 31 + n * 7) & 0x3FF)); }
}  // namespace

int main() {
  const model::ModelDesc& one = model::qwen38();
  const model::ModelDesc& card = model::tp_card_desc(one, model::TpCrossing::Slices);

  // --- columns -------------------------------------------------------------------------------------------------
  check_columns(one, card, LinearId::QkvZ, one.gdn_qkvz_n());
  check_columns(one, card, LinearId::Qkv, one.fa_qkv_n());
  check_columns(one, card, LinearId::GateUp, 2 * one.intermediate);
  check_columns(one, card, LinearId::LmHead, model::Qwen35::kVocab);
  check_columns(one, card, LinearId::AB, one.gdn_ab_n());
  {  // the heads behind the columns (card 1)
    const std::vector<uint32_t> q = loader::tp_columns(one, LinearId::QkvZ, 1);
    CHECK_EQ(q[0], 8 * kG);                        // q of k-head 8
    CHECK_EQ(q[1024], 2048 + 8 * kG);              // k of k-head 8
    CHECK_EQ(q[2048], 4096 + 24 * kG);             // v of v-head 24
    CHECK_EQ(q[5120], 10240 + 24 * kG);            // z of v-head 24
    const std::vector<uint32_t> a = loader::tp_columns(one, LinearId::Qkv, 1);
    CHECK_EQ(a[0], 12 * 2 * kF);                   // q||gate of q-head 12
    CHECK_EQ(a[6144], 12288 + 2 * kF);             // k of kv-head 2
    CHECK_EQ(a[6656], 13312 + 2 * kF);             // v of kv-head 2
    const std::vector<uint32_t> gu = loader::tp_columns(one, LinearId::GateUp, 1);
    CHECK_EQ(gu[0], 17408u);                       // gate[8704] sits at 2 x 8704 in the 16-column interleave
    CHECK_EQ(gu[16], 17424u);                      // and up[8704] 16 columns on
  }
  for (LinearId id : {LinearId::OutProj, LinearId::OProj, LinearId::Down}) CHECK(loader::tp_row_parallel(id));
  for (LinearId id : {LinearId::QkvZ, LinearId::Qkv, LinearId::GateUp, LinearId::AB, LinearId::LmHead})
    CHECK(!loader::tp_row_parallel(id));

  // --- repack: card tiles are one-card tiles -------------------------------------------------------------------
  {
    constexpr uint32_t K = 512, N = 128, Kc = K / 2;
    std::vector<uint32_t> qw(size_t(K / 8) * N);
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    for (uint32_t r = 0; r < K / 8; ++r)
      for (uint32_t n = 0; n < N; ++n) qw[size_t(r) * N + n] = word(r, n);
    for (uint32_t g = 0; g < K / 64; ++g)
      for (uint32_t n = 0; n < N; ++n) sc[size_t(g) * N + n] = scale(g, n);
    const std::vector<common::ColSource> full = common::cols_concat({{qw.data(), sc.data(), N}});
    std::vector<uint32_t> one_l1(size_t(N / 16) * (K / 64) * 136);
    common::repack_int4_layout1_cols(K, N, full, one_l1.data());
    for (uint32_t r = 0; r < 2; ++r) {
      // row-parallel: the card's K rows by pointer offset (load_linear_tp's arithmetic)
      std::vector<common::ColSource> rows = full;
      for (common::ColSource& c : rows) {
        c.qweight += size_t(r) * (Kc / 8) * c.n_part;
        c.scales += size_t(r) * (Kc / 64) * c.n_part;
      }
      std::vector<uint32_t> l1(size_t(N / 16) * (Kc / 64) * 136);
      common::repack_int4_layout1_cols(Kc, N, rows, l1.data());
      for (uint32_t nt = 0; nt < N / 16; ++nt)
        for (uint32_t g = 0; g < Kc / 64; ++g)
          CHECK(std::memcmp(&l1[(size_t(nt) * (Kc / 64) + g) * 136],
                            &one_l1[(size_t(nt) * (K / 64) + g + r * (Kc / 64)) * 136], 136 * 4) == 0);
      std::vector<uint32_t> l0(size_t(Kc / 8) * N);
      std::vector<uint16_t> s0(size_t(Kc / 64) * N);
      common::repack_int4_layout0_cols(Kc, N, rows, l0.data(), s0.data());
      CHECK(std::memcmp(l0.data(), &qw[size_t(r) * (Kc / 8) * N], l0.size() * 4) == 0);
      CHECK(std::memcmp(s0.data(), &sc[size_t(r) * (Kc / 64) * N], s0.size() * 2) == 0);
      // column-parallel: 16-aligned column ranges (each card's half) - whole one-card n-tiles
      std::vector<common::ColSource> cols(full.begin() + r * (N / 2), full.begin() + (r + 1) * (N / 2));
      std::vector<uint32_t> c1(size_t(N / 32) * (K / 64) * 136);
      common::repack_int4_layout1_cols(K, N / 2, cols, c1.data());
      CHECK(std::memcmp(c1.data(), &one_l1[size_t(r) * (N / 32) * (K / 64) * 136], c1.size() * 4) == 0);
    }
    std::printf("repack: row-parallel (layout 0 / 1) and column-parallel card tiles are one-card tiles\n");
  }

  // --- small tensors ---------------------------------------------------------------------------------------------
  for (uint32_t r = 0; r < 2; ++r) {
    const std::vector<uint32_t> conv = loader::tp_small_elems(one, "linear_attn.conv1d.weight", r);
    CHECK_EQ(conv.size(), size_t(card.gdn_conv_dim()) * loader::kConvTaps);
    CHECK_EQ(conv[0], r * 1024 * 4);                         // q channel 1024 r, tap 0
    CHECK_EQ(conv[1024 * 4], (2048 + r * 1024) * 4);         // k
    CHECK_EQ(conv[2048 * 4], (4096 + r * 3072) * 4);         // v
    const std::vector<uint32_t> a = loader::tp_small_elems(one, "linear_attn.A_log", r);
    CHECK_EQ(a.size(), size_t(24));
    CHECK_EQ(a[0], 24 * r);
    CHECK(loader::tp_small_elems(one, "linear_attn.norm.weight", r).empty());
    CHECK(loader::tp_small_elems(one, "input_layernorm.weight", r).empty());
    CHECK(loader::tp_small_elems(one, "self_attn.q_norm.weight", r).empty());
  }

  // --- gather: split one card's state / KV by the card rules, gather back ------------------------------------------
  {
    const size_t head = size_t(kG) * kG * 4;
    std::vector<uint8_t> gdn(size_t(one.gdn_layers) * one.gdn_v_heads * head);
    for (size_t i = 0; i < gdn.size(); ++i) gdn[i] = uint8_t(i * 131 + (i >> 9));
    const size_t ring_row = size_t(one.gdn_conv_dim()) * 2, rows = size_t(one.gdn_layers) * 16;
    std::vector<uint8_t> conv(rows * ring_row);
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = uint8_t(i * 37 + (i >> 11));
    std::vector<uint8_t> cg[2], cc[2];
    const size_t card_row = size_t(card.gdn_conv_dim()) * 2;
    for (uint32_t r = 0; r < 2; ++r) {
      for (uint32_t l = 0; l < one.gdn_layers; ++l) {   // the card's v-heads of every layer
        const uint8_t* src = &gdn[(size_t(l) * one.gdn_v_heads + r * card.gdn_v_heads) * head];
        cg[r].insert(cg[r].end(), src, src + card.gdn_v_heads * head);
      }
      for (size_t i = 0; i < rows; ++i)   // the card's q / k / v channels of every ring row (bf16)
        for (const auto& [base, w] : {std::pair<size_t, size_t>{0, 1024}, {2048, 1024}, {4096, 3072}}) {
          const uint8_t* src = &conv[i * ring_row + (base + r * w) * 2];
          cc[r].insert(cc[r].end(), src, src + w * 2);
        }
      CHECK_EQ(cc[r].size(), rows * card_row);
    }
    const uint8_t* gp[2] = {cg[0].data(), cg[1].data()};
    const uint8_t* cp[2] = {cc[0].data(), cc[1].data()};
    std::vector<uint8_t> out(gdn.size() + conv.size());
    runtime::tp::gather_state(card, gp, cp, out.data());
    CHECK(std::memcmp(out.data(), gdn.data(), gdn.size()) == 0);
    CHECK(std::memcmp(out.data() + gdn.size(), conv.data(), conv.size()) == 0);
    {   // spec 17d: the scatter is the gather's inverse, byte for byte
      std::vector<uint8_t> sg[2], sc2[2];
      uint8_t* sgp[2];
      uint8_t* scp[2];
      for (uint32_t r = 0; r < 2; ++r) {
        sg[r].assign(cg[r].size(), 0xA5);
        sc2[r].assign(cc[r].size(), 0x5A);
        sgp[r] = sg[r].data();
        scp[r] = sc2[r].data();
      }
      runtime::tp::scatter_state(card, out.data(), sgp, scp);
      for (uint32_t r = 0; r < 2; ++r) CHECK(sg[r] == cg[r] && sc2[r] == cc[r]);
    }

    for (runtime::KvCache kv : {runtime::KvCache::Bf16, runtime::KvCache::Int8}) {
      constexpr uint32_t n = 37;
      const size_t e = kv == runtime::KvCache::Int8 ? 1 : 2, row = size_t(4) * kF * e,
                   sc = kv == runtime::KvCache::Int8 ? 8 : 0, L = one.fa_layers;
      const size_t tensor = L * n * (row + sc);
      std::vector<uint8_t> kvb(2 * tensor);
      for (size_t i = 0; i < kvb.size(); ++i) kvb[i] = uint8_t(i * 29 + (i >> 10));
      std::vector<uint8_t> cards[2];
      for (uint32_t r = 0; r < 2; ++r)
        for (uint32_t t = 0; t < 2; ++t) {
          for (size_t i = 0; i < L * n; ++i) {   // rows: the card's two kv-heads of every (layer, position)
            const uint8_t* src = &kvb[t * tensor + i * row + r * row / 2];
            cards[r].insert(cards[r].end(), src, src + row / 2);
          }
          for (size_t i = 0; sc && i < L * n; ++i) {   // scales: its two heads' fp16
            const uint8_t* src = &kvb[t * tensor + L * n * row + i * sc + r * sc / 2];
            cards[r].insert(cards[r].end(), src, src + sc / 2);
          }
        }
      const uint8_t* kp[2] = {cards[0].data(), cards[1].data()};
      std::vector<uint8_t> got(kvb.size());
      runtime::tp::gather_kv(card, kv, n, kp, got.data());
      CHECK(got == kvb);
      // Spec 17d: the scatter back into the cards' layouts, byte for byte.
      std::vector<uint8_t> back[2] = {std::vector<uint8_t>(cards[0].size(), 0xEE),
                                      std::vector<uint8_t>(cards[1].size(), 0xEE)};
      uint8_t* bp[2] = {back[0].data(), back[1].data()};
      runtime::tp::scatter_kv(card, kv, n, got.data(), bp);
      CHECK(back[0] == cards[0] && back[1] == cards[1]);
      // Spec 17d: the MTP head's layer spliced in after the FA layers (Engine::save_kv with MTP on: per tensor the
      // FA rows, the head's rows, the FA scales, the head's scales) and split back out.
      const runtime::tp::KvSplit ks = runtime::tp::kv_split(one.fa_layers, 4, 4, kv, n);
      CHECK_EQ(2 * ks.main_tensor(), kvb.size());
      std::vector<uint8_t> headb(2 * ks.head_tensor());
      for (size_t i = 0; i < headb.size(); ++i) headb[i] = uint8_t(i * 7 + 3);
      std::vector<uint8_t> spliced(kvb.size() + headb.size());
      runtime::tp::splice_head_kv(ks, kvb.data(), headb.data(), spliced.data());
      size_t at = 0;
      for (uint32_t t = 0; t < 2; ++t) {   // the expected order, written out
        CHECK(std::memcmp(&spliced[at], &kvb[t * tensor], L * n * row) == 0);
        at += L * n * row;
        CHECK(std::memcmp(&spliced[at], &headb[t * ks.head_tensor()], ks.rows_head) == 0);
        at += ks.rows_head;
        if (sc) {
          CHECK(std::memcmp(&spliced[at], &kvb[t * tensor + L * n * row], L * n * sc) == 0);
          at += L * n * sc;
          CHECK(std::memcmp(&spliced[at], &headb[t * ks.head_tensor() + ks.rows_head], ks.scales_head) == 0);
          at += ks.scales_head;
        }
      }
      CHECK_EQ(at, spliced.size());
      std::vector<uint8_t> m2(kvb.size()), h2(headb.size());
      runtime::tp::split_head_kv(ks, spliced.data(), m2.data(), h2.data());
      CHECK(m2 == kvb && h2 == headb);
    }
    std::printf("gather: state, bf16 and int8 KV byte-identical to the one-card layouts; scatter and the MTP head's "
                "splice their inverses (spec 17d)\n");
  }
  std::printf("tp_slice_test: OK\n");
  return 0;
}
