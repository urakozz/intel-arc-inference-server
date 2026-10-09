// Spec 21d Task 1: the Qwen3.8-Flash-Next prefill references (tests/kernels/qwen4exp_pf_ref.h) against
// independent formulas and against 21c's decode references (qwen4exp_ref.h) - host only, so the Mac runs it.
//
//   (a) the sort (Review Focus 3): random routes from q4ref::route at C 1 / 37 / 300 / 2048, routes with exact
//       ties at the 10th (coarse logits), and the adversaries - every token to the same 10 experts (0, 1, 127,
//       254, 255, 256, 257, 383, 510, 511: the 256-lane boundaries and the last expert; 704 tiles), a chunk that
//       reaches tmax(C) EXACTLY (112 experts with 65 rows, 400 with 33: 1200 tiles at C = 2048), an expert with
//       exactly 32 rows (one tile, no padding) and one with 0 - every (token, slot) once, in a tile of its expert,
//       ascending token, padding NONE, the shared block 512 last with the tokens in order, the header; and the
//       whole result equal to spec 15d's pf_moe_ref::sort at this family's shape (one layout, two homes);
//   (b) row independence: the chunk REVERSED, and the rows permuted inside every tile, leave every token's combine
//       output bitwise unchanged;
//   (c) the combine == 21c's decode chain bit for bit: y rows formed as the decode kernel's own down sums
//       (q4ref::split_dot over real-width layout-1 blocks, rounded once - what the grouped GEMM's plain epilogue
//       stores) give exactly q4ref::down's y on every token and column, both shared forms;
//   (d) dense_rows at Review Focus 1's chunk starts (0, 1000, 2000 straddling, 2048, 4096, a 300-row tail), each
//       row's side held to q4ref::qsa_count (dense <=> the identity list);
//   (e) sparse_attn_fp64 (the tiled walk) == q4ref::qsa_attn_fp64 (the direct softmax) over q4ref::qsa_select's
//       lists at rows 2051 / 2052 / 2055 / 4095 / 4096 / 9000; sparse_row_eager + the gate == q4ref::qsa_attn_eager
//       bitwise;
//   (f) qsa_prep_chunk (the M = C prep + the ring launch) == decode's M = 1 chain row by row - the query heads,
//       every compressed key and the 8-slot tail ring - over chunk sizes 1 / 3 / 7 / 8 / 13 / 64 / 300 from starts
//       that are and are not multiples of 4 (Review Focus 2);
//   (g) ple_ids_chunk == loader::q4_ple_history over the whole sequence (EOS inside chunks and at a chunk edge),
//       ple_chunk (gate, conv, ring: three launches) == q4ref::ple_block applied row by row with decode's rings -
//       H, the conv ring and the id ring bitwise (Review Focus 4).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/pf_moe_ref.h"
#include "kernels/qwen4exp_pf_ref.h"
#include "kernels/qwen4exp_ref.h"
#include "loader/qwen4exp_ple_hash.h"

namespace {

namespace qr = q4ref;
namespace qp = q4pf;
using qr::f32;
using qr::rf;
using qr::rne;
constexpr uint32_t W = qr::kWords, K = qr::kTopK, E = qr::kExperts;

// A route row with given ids (rank order) and weights; the shared gate sg.
void put_ids(uint32_t* row, const uint32_t* ids, const float* w, float sg = 0.5f) {
  std::fill(row, row + W, 0u);
  for (uint32_t j = 0; j < K; ++j) {
    row[qr::kIds + j] = ids[j];
    row[qr::kWeights + j] = qr::as_u32(w[j]);
  }
  row[qr::kSharedGate] = qr::as_u32(sg);
}

// C tokens routed by q4ref::route; `coarse` puts the logits on a small grid (exact ties at the 10th).
std::vector<uint32_t> routed(uint32_t C, uint32_t seed, bool coarse, uint32_t* ties) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-3.f, 3.f);
  std::uniform_int_distribution<int> lq(-3, 3);
  std::vector<float> lg(qr::kRouterN, 0.0f);
  std::vector<uint32_t> rows(size_t(C) * W);
  for (uint32_t t = 0; t < C; ++t) {
    for (uint32_t e = 0; e <= E; ++e) lg[e] = coarse ? 0.5f * float(lq(rng)) : u(rng);
    const qr::Route r = qr::route(lg.data());
    if (r.p10 == r.p11) ++*ties;   // a tie at the cut: the taken one has the lower id
    const std::vector<uint32_t> row = qr::route_row(r);
    std::copy(row.begin(), row.end(), rows.begin() + size_t(t) * W);
  }
  return rows;
}

// Route rows from an expert multiset laid out column-major: token t's slot j takes list[j x C + t] - distinct per
// token as long as no expert has more than C rows.
std::vector<uint32_t> from_list(const std::vector<uint32_t>& list, uint32_t C) {
  CHECK_EQ(list.size(), size_t(C) * K);
  std::vector<uint32_t> rows(size_t(C) * W);
  for (uint32_t t = 0; t < C; ++t) {
    uint32_t ids[K];
    float w[K];
    for (uint32_t j = 0; j < K; ++j) {
      ids[j] = list[size_t(j) * C + t];
      w[j] = rf(0.05f + 0.01f * float(j));
    }
    for (uint32_t a = 0; a < K; ++a)
      for (uint32_t b = a + 1; b < K; ++b) CHECK(ids[a] != ids[b]);
    put_ids(rows.data() + size_t(t) * W, ids, w);
  }
  return rows;
}

// Every invariant of the sorted layout (q4_pf_moe.cl's "no pair is lost" argument).
void check_sorted(const qp::Sorted& st, const uint32_t* route, uint32_t C) {
  const uint32_t T = qp::tmax(C);
  CHECK(st.hdr[qp::kHdrTiles] <= T);
  CHECK_EQ(st.hdr[qp::kHdrTiles], st.tiles_used);
  CHECK_EQ(st.hdr[qp::kHdrRows], st.hdr[qp::kHdrTiles] * qp::kTm);
  CHECK_EQ(st.hdr[qp::kHdrC], C);
  CHECK_EQ(st.hdr[qp::kHdrCount + E], C);
  std::vector<uint32_t> cnt(E, 0), seen(size_t(T) * qp::kTm, 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < K; ++j) {
      const uint32_t e = route[size_t(t) * W + qr::kIds + j];
      const uint32_t r = st.pair_row[size_t(t) * K + j];
      CHECK(r < st.rows_used());
      CHECK_EQ(st.row_tok[r], t);
      CHECK_EQ(st.tiles[2 * (r / qp::kTm)], e);
      CHECK_EQ(seen[r], 0u);
      seen[r] = 1;
      ++cnt[e];
    }
  for (uint32_t e = 0; e < E; ++e) CHECK_EQ(st.hdr[qp::kHdrCount + e], cnt[e]);
  for (uint32_t tile = 0; tile < st.hdr[qp::kHdrTiles]; ++tile) {
    const uint32_t b = st.tiles[2 * tile], r0 = st.tiles[2 * tile + 1];
    CHECK_EQ(r0, tile * qp::kTm);
    CHECK(b <= qp::kShared);
    for (uint32_t i = 1; i < qp::kTm; ++i) {
      const uint32_t a = st.row_tok[r0 + i - 1], c = st.row_tok[r0 + i];
      if (c != qp::kNone) CHECK(a != qp::kNone && (b == qp::kShared ? c == a + 1 : c > a));
    }
    if (tile > 0) CHECK(st.tiles[2 * (tile - 1)] <= b);   // expert-major, the shared block last
  }
  for (uint32_t tile = st.hdr[qp::kHdrTiles]; tile < T; ++tile) CHECK_EQ(st.tiles[2 * tile], qp::kNone);
  for (uint32_t t = 0; t < C; ++t) CHECK_EQ(st.row_tok[st.shared_row() + t], t);
  for (uint32_t r = st.shared_row() + C; r < st.rows_used(); ++r) CHECK_EQ(st.row_tok[r], qp::kNone);
  // one layout, two homes: spec 15d's sort at this family's shape (its route row reads the ids at words 0..9 too)
  const pf_moe_ref::Sorted o = pf_moe_ref::sort(route, C, moe_ref::Shape{E, K, qr::kHidden, qr::kInter, qr::kRouterN});
  CHECK(o.hdr.size() <= st.hdr.size());
  CHECK(std::equal(o.hdr.begin(), o.hdr.end(), st.hdr.begin()));
  CHECK(o.tiles == st.tiles && o.row_tok == st.row_tok && o.pair_row == st.pair_row);
}

void test_sort() {
  CHECK_EQ(qp::tmax(2048), 1200u);
  CHECK_EQ(qp::tmax(1), 496u + 1u);   // (10 + 15872) / 32 = 496, + 1 shared tile
  CHECK_EQ(qp::hdr_words(), 528u);
  uint32_t ties = 0, tied_rows = 0;
  for (uint32_t C : {1u, 37u, 300u, 2048u}) {
    const std::vector<uint32_t> r = routed(C, C, false, &ties);
    check_sorted(qp::sort(r.data(), C), r.data(), C);
    const std::vector<uint32_t> rt = routed(C, C + 7, true, &tied_rows);
    check_sorted(qp::sort(rt.data(), C), rt.data(), C);
  }
  CHECK(tied_rows > 0);
  const uint32_t C = 2048;
  // every token to the same ten experts - the lane boundaries 255 / 256, 511 / 0 and the last expert
  {
    const uint32_t ten[K] = {511, 0, 256, 255, 1, 127, 254, 257, 383, 510};
    std::vector<uint32_t> list;
    for (uint32_t j = 0; j < K; ++j) list.insert(list.end(), C, ten[j]);
    const std::vector<uint32_t> r = from_list(list, C);
    const qp::Sorted st = qp::sort(r.data(), C);
    check_sorted(st, r.data(), C);
    CHECK_EQ(st.tiles_used, 10u * 64u + 64u);
    for (uint32_t e : ten) CHECK_EQ(st.hdr[qp::kHdrCount + e], C);
  }
  // the bound reached exactly: 112 experts with 65 rows (three tiles), 400 with 33 (two): 1136 + 64 tiles
  {
    std::vector<uint32_t> list;
    for (uint32_t e = 0; e < E; ++e) list.insert(list.end(), e < 112 ? 65u : 33u, e);
    const std::vector<uint32_t> r = from_list(list, C);
    const qp::Sorted st = qp::sort(r.data(), C);
    check_sorted(st, r.data(), C);
    CHECK_EQ(st.tiles_used, qp::tmax(C));
    CHECK_EQ(st.tiles_used, 1200u);
  }
  // an expert with exactly 32 rows (one full tile), one with 0, the rest in runs <= 41
  {
    std::vector<uint32_t> list;
    list.insert(list.end(), 32, 5u);
    uint32_t left = C * K - 32, e = 0;
    while (left) {
      if (e == 5 || e == 6) {
        ++e;
        continue;
      }
      const uint32_t n = std::min(left, 41u);
      list.insert(list.end(), n, e % E);
      left -= n;
      e = (e + 1) % E;
    }
    std::sort(list.begin(), list.end());
    const std::vector<uint32_t> r = from_list(list, C);
    const qp::Sorted st = qp::sort(r.data(), C);
    check_sorted(st, r.data(), C);
    CHECK_EQ(st.hdr[qp::kHdrCount + 5], 32u);
    CHECK_EQ(st.hdr[qp::kHdrCount + 6], 0u);
    uint32_t tiles5 = 0, tiles6 = 0;
    for (uint32_t t = 0; t < st.tiles_used; ++t) {
      tiles5 += st.tiles[2 * t] == 5;
      tiles6 += st.tiles[2 * t] == 6;
    }
    CHECK(tiles5 == 1 && tiles6 == 0);
  }
  std::printf("  sort: random / tied (%u rows with an exact tie at the 10th) at C 1 / 37 / 300 / 2048, all-to-10 (lanes "
              "255 / 256 / 511), the bound reached (1200 tiles), a full tile and an empty expert - every invariant; == "
              "pf_moe_ref::sort\n", tied_rows);
}

// y rows from the A rows: a function of the row's own A row and its expert only (as the grouped GEMM's).
std::vector<uint16_t> fake_y(const qp::Sorted& st, const std::vector<uint16_t>& x) {
  std::vector<uint16_t> y(size_t(st.rows_used()) * qr::kHidden, 0x7FC0);
  for (uint32_t tile = 0; tile < st.tiles_used; ++tile) {
    const uint32_t e = st.tiles[2 * tile];
    for (uint32_t i = 0; i < qp::kTm; ++i) {
      const uint32_t r = st.tiles[2 * tile + 1] + i, t = st.row_tok[r];
      if (t == qp::kNone) continue;
      for (uint32_t n = 0; n < qr::kHidden; ++n) {
        float a = 0.f;
        for (uint32_t k = 0; k < 16; ++k)
          a += f32(x[size_t(t) * qr::kHidden + k]) * (float(int((e * 131 + n * 7 + k * 3) % 17) - 8) * 0.0625f);
        y[size_t(r) * qr::kHidden + n] = rne(a);
      }
    }
  }
  return y;
}

void test_independence() {
  const uint32_t C = 300, H = qr::kHidden;
  uint32_t ties = 0;
  const std::vector<uint32_t> rt = routed(C, 41, false, &ties);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<uint16_t> x(size_t(C) * H);
  for (uint16_t& v : x) v = rne(u(rng));
  const qp::Sorted st = qp::sort(rt.data(), C);
  const std::vector<uint16_t> y = fake_y(st, x);
  std::vector<uint16_t> mo(size_t(C) * H);
  qp::combine(rt.data(), st, y.data(), mo.data(), C);
  {   // the chunk reversed
    std::vector<uint16_t> xr(x.size());
    std::vector<uint32_t> rr(rt.size());
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t v = C - 1 - t;
      std::copy(x.begin() + size_t(t) * H, x.begin() + size_t(t + 1) * H, xr.begin() + size_t(v) * H);
      std::copy(rt.begin() + size_t(t) * W, rt.begin() + size_t(t + 1) * W, rr.begin() + size_t(v) * W);
    }
    const qp::Sorted sr = qp::sort(rr.data(), C);
    const std::vector<uint16_t> yr = fake_y(sr, xr);
    std::vector<uint16_t> mr(size_t(C) * H);
    qp::combine(rr.data(), sr, yr.data(), mr.data(), C);
    for (uint32_t t = 0; t < C; ++t)
      CHECK(std::equal(mo.begin() + size_t(t) * H, mo.begin() + size_t(t + 1) * H, mr.begin() + size_t(C - 1 - t) * H));
  }
  {   // the rows permuted inside every routed tile (row_tok and pair_row moved with them)
    qp::Sorted sp = st;
    std::mt19937 g(43);
    for (uint32_t tile = 0; tile < sp.tiles_used; ++tile) {
      if (sp.tiles[2 * tile] == qp::kShared) continue;   // combine reads row shared_row + t: keep those in order
      const uint32_t r0 = sp.tiles[2 * tile + 1];
      std::vector<uint32_t> perm(qp::kTm);
      for (uint32_t i = 0; i < qp::kTm; ++i) perm[i] = i;
      std::shuffle(perm.begin(), perm.end(), g);
      for (uint32_t i = 0; i < qp::kTm; ++i) sp.row_tok[r0 + perm[i]] = st.row_tok[r0 + i];
    }
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < K; ++j) {
        const uint32_t r = st.pair_row[size_t(t) * K + j], tile = r / qp::kTm;
        for (uint32_t i = 0; i < qp::kTm; ++i)
          if (sp.row_tok[tile * qp::kTm + i] == t) sp.pair_row[size_t(t) * K + j] = tile * qp::kTm + i;
      }
    const std::vector<uint16_t> yp = fake_y(sp, x);
    std::vector<uint16_t> mp(size_t(C) * H);
    qp::combine(rt.data(), sp, yp.data(), mp.data(), C);
    CHECK(mp == mo);
  }
  std::puts("  row independence: the chunk reversed and the rows permuted inside every tile leave every token's combine "
            "bitwise");
}

// (c) the combine against decode's whole down chain on real-width blocks: experts 0..15, both shared forms.
void test_combine_vs_decode() {
  const uint32_t C = 12, H = qr::kHidden, I = qr::kInter, NE = 16;
  std::mt19937 rng(77);
  std::uniform_int_distribution<uint32_t> word;
  std::uniform_real_distribution<float> sc(0.002f, 0.02f), u(-1.f, 1.f);
  const size_t dw = qr::down_block_words();
  auto block = [&](size_t words) {
    std::vector<uint32_t> b(words);
    const size_t tiles = words / 136;
    for (size_t t = 0; t < tiles; ++t) {
      for (uint32_t i = 0; i < 128; ++i) b[t * 136 + i] = word(rng);
      for (uint32_t i = 0; i < 8; ++i)
        b[t * 136 + 128 + i] = uint32_t(common::f32_to_f16(sc(rng))) | (uint32_t(common::f32_to_f16(sc(rng))) << 16);
    }
    return b;
  };
  std::vector<uint32_t> dn = block(dw * NE), sh4 = block(dw);
  std::vector<uint16_t> shb(size_t(I) * H);
  for (uint16_t& v : shb) v = rne(0.05f * u(rng));
  std::vector<uint32_t> route(size_t(C) * W);
  std::vector<std::vector<uint16_t>> hs(C);
  for (uint32_t t = 0; t < C; ++t) {
    std::vector<uint32_t> ids(NE);
    for (uint32_t e = 0; e < NE; ++e) ids[e] = e;
    std::shuffle(ids.begin(), ids.end(), rng);
    float w[K];
    for (uint32_t j = 0; j < K; ++j) w[j] = rf(0.02f + 0.1f * std::fabs(u(rng)));
    put_ids(route.data() + size_t(t) * W, ids.data(), w, rf(0.3f + 0.2f * u(rng)));
    hs[t].resize(size_t(qr::kSlots) * I);
    for (uint16_t& v : hs[t]) v = rne(u(rng));
  }
  const qp::Sorted st = qp::sort(route.data(), C);
  for (int form = 0; form < 2; ++form) {
    const uint32_t* s4 = form == 0 ? sh4.data() : nullptr;
    const uint16_t* sb = form == 0 ? nullptr : shb.data();
    // the grouped GEMM's plain epilogue: y = rne(the row's down sum) - here decode's own sums
    std::vector<uint16_t> y(size_t(st.rows_used()) * H, 0x7FC0);
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t* rr = route.data() + size_t(t) * W;
      for (uint32_t j = 0; j <= K; ++j) {
        const uint16_t* h = hs[t].data() + size_t(j) * I;
        const uint32_t row = j < K ? st.pair_row[size_t(t) * K + j] : st.shared_row() + t;
        for (uint32_t n = 0; n < H; ++n) {
          const float d = j < K ? qr::split_dot(dn.data() + rr[qr::kIds + j] * dw, I, n, h, 2)
                                : (s4 ? qr::split_dot(s4, I, n, h, 2) : qr::bf16_split_dot(sb, I, n, h, 2));
          y[size_t(row) * H + n] = rne(d);
        }
      }
    }
    std::vector<uint16_t> out(size_t(C) * H);
    qp::combine(route.data(), st, y.data(), out.data(), C);
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t* rr = route.data() + size_t(t) * W;
      float w[K];
      for (uint32_t j = 0; j < K; ++j) w[j] = qr::as_f32(rr[qr::kWeights + j]);
      const std::vector<uint16_t> want =
          qr::down(rr + qr::kIds, w, qr::as_f32(rr[qr::kSharedGate]), hs[t], dn.data(), s4, sb, 2);
      CHECK(std::equal(want.begin(), want.end(), out.begin() + size_t(t) * H));
    }
  }
  std::puts("  combine: == decode's down chain (q4ref::down over real-width layout-1 blocks, int4 and bf16 shared) on 12 "
            "tokens x 2560, bitwise");
}

void test_dense_rows() {
  struct Case { uint32_t pos, C, want; };
  for (const Case c : {Case{0, 2048, 2048}, Case{1000, 2048, 1051}, Case{2000, 2048, 51}, Case{2048, 2048, 3},
                       Case{2051, 2048, 0}, Case{4096, 2048, 0}, Case{2048 + 2048, 300, 0}, Case{1900, 300, 151},
                       Case{0, 300, 300}}) {
    const uint32_t nd = qp::dense_rows(c.pos, c.C);
    CHECK_EQ(nd, c.want);
    for (uint32_t m = 0; m < c.C; ++m) {
      const uint32_t p = c.pos + m;
      CHECK_EQ(m < nd, qr::qsa_count(p) == p + 1);   // dense <=> the identity list (every visible position)
    }
  }
  std::puts("  dense_rows: chunk starts 0 / 1000 / 2000 (51 dense rows) / 2048 (3) / 4096 (0) and 300-row tails, each "
            "row's side == q4ref::qsa_count's");
}

uint16_t kv_val(uint32_t p, uint32_t j, uint32_t d, uint32_t which) {
  return rne(qr::fixture_val(false, 30 + which, (p * 2 + j) * 256 + d, 11, 0.125f, 1.0f));
}

void test_sparse() {
  const uint32_t HD = qr::kHd, QH = qr::kQHeads, KVN = qr::kKvHeads * HD, P = 9001;
  std::vector<uint16_t> kc(size_t(P) * KVN), vc(size_t(P) * KVN);
  for (uint32_t p = 0; p < P; ++p)
    for (uint32_t j = 0; j < qr::kKvHeads; ++j)
      for (uint32_t d = 0; d < HD; ++d) {
        kc[(size_t(p) * 2 + j) * HD + d] = kv_val(p, j, d, 0);
        vc[(size_t(p) * 2 + j) * HD + d] = kv_val(p, j, d, 1);
      }
  std::mt19937 rng(91);
  std::uniform_real_distribution<float> u(-2.f, 2.f), us(0.f, 4.f);
  const std::vector<uint32_t> rows = {2051, 2052, 2055, 4095, 4096, 9000};
  const uint32_t C = uint32_t(rows.size());
  std::vector<float> q(size_t(C) * QH * HD), gate(size_t(QH) * HD);
  for (float& v : q) v = rf(u(rng));
  for (float& v : gate) v = rf(u(rng));
  std::vector<uint32_t> lists(size_t(C) * qp::kListRow, 0), counts(C);
  for (uint32_t r = 0; r < C; ++r) {
    const uint32_t p = rows[r], n = (p + 1) / 4;
    std::vector<float> scores(n);
    for (float& s : scores) s = rf(us(rng));
    const qr::Selection sel = qr::qsa_select(scores.data(), p);
    CHECK_EQ(sel.count(), qr::qsa_count(p));
    std::copy(sel.list.begin(), sel.list.end(), lists.begin() + size_t(r) * qp::kListRow);
    counts[r] = sel.count();
  }
  std::vector<double> o(size_t(QH) * C * HD);
  qp::sparse_attn_fp64(q.data(), kc.data(), vc.data(), lists.data(), counts.data(), 0, C, o.data());
  double worst = 0;
  uint32_t eager_rows = 0;
  for (uint32_t r = 0; r < C; ++r)
    for (uint32_t h = 0; h < QH; ++h) {
      const float* qh = q.data() + size_t(r) * QH * HD;
      const std::vector<double> want =
          qr::qsa_attn_fp64(qh, kc.data(), vc.data(), lists.data() + size_t(r) * qp::kListRow, counts[r], h);
      for (uint32_t d = 0; d < HD; ++d) worst = std::max(worst, std::fabs(o[(size_t(h) * C + r) * HD + d] - want[d]));
      if (h % 7 == 0) {   // the eager chain + the gate == q4ref::qsa_attn_eager, bitwise
        const std::vector<uint16_t> e =
            qp::sparse_row_eager(qh, kc.data(), vc.data(), lists.data() + size_t(r) * qp::kListRow, counts[r], h);
        const qr::EagerHead ref = qr::qsa_attn_eager(qh, gate.data(), kc.data(), vc.data(),
                                                     lists.data() + size_t(r) * qp::kListRow, counts[r], h);
        for (uint32_t d = 0; d < HD; ++d)
          CHECK_EQ(rne(f32(e[d]) * rf(qr::sigmoid_t(gate[size_t(h) * HD + d]))), ref.out[d]);
        ++eager_rows;
      }
    }
  CHECK(worst < 1e-12);
  std::printf("  sparse attention: the tiled fp64 walk (tiles of %u list entries) == the direct softmax within %.1e on "
              "rows 2051..9000 x 24 heads; the eager chain + the gate == q4ref::qsa_attn_eager bitwise (%u heads)\n",
              qp::kKt, worst, eager_rows);
}

// Decode's M = 1 indexer chain for one row (q4_qsa_prep at M = 1): q, the completing block's key, ring[p % 8].
void prep_m1(const float* idx, uint32_t p, const float* wq, const float* wk, const float* rope, uint16_t* tail,
             uint16_t* keys, float* q) {
  const uint32_t D = qr::kIdxDim;
  qr::qsa_q(idx, wq, rope + size_t(p) * 2 * qr::kRotHalf, q);
  uint16_t raw[qr::kIdxDim];
  qr::qsa_raw_key(idx, raw);
  if ((p + 1) % 4 == 0) {
    const uint32_t b = (p + 1) / 4 - 1;
    const uint16_t* r4[4];
    for (uint32_t j = 0; j < 4; ++j) r4[j] = b * 4 + j == p ? raw : tail + size_t((b * 4 + j) % qr::kTailSlots) * D;
    qr::qsa_block_key(r4, wk, rope + size_t(b * 4) * 2 * qr::kRotHalf, keys + size_t(b) * D);
  }
  std::copy(raw, raw + D, tail + size_t(p % qr::kTailSlots) * D);
}

void test_prep_chunk() {
  const uint32_t D = qr::kIdxDim, N = 900, LD = 768;   // pf_ld(640): the prefill GEMM's row pitch
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(-2.f, 2.f), c(-1.f, 1.f), w(0.5f, 1.5f);
  std::vector<float> idx(size_t(N) * LD), rope(size_t(N) * 2 * qr::kRotHalf), wq(D), wk(D);
  for (float& v : idx) v = u(rng);
  for (float& v : rope) v = c(rng);
  for (float& v : wq) v = w(rng);
  for (float& v : wk) v = w(rng);
  uint32_t runs = 0;
  for (const std::vector<uint32_t>& sizes : std::vector<std::vector<uint32_t>>{
           {1, 3, 7, 8, 13, 64, 300, 1}, {6, 300, 2, 9, 64}, {2, 1, 1, 13, 8, 300, 7}, {600, 3}}) {
    std::vector<uint16_t> tail_a(size_t(qr::kTailSlots) * D, 0), tail_b = tail_a, keys_a(size_t(N / 4) * D, 0),
        keys_b = keys_a;
    std::vector<float> q_a(size_t(N) * 4 * D), q_b = q_a;
    uint32_t pos = 0;
    for (uint32_t C : sizes) {
      qp::qsa_prep_chunk(idx.data() + size_t(pos) * LD, LD, pos, C, wq.data(), wk.data(), rope.data(), tail_a.data(),
                         keys_a.data(), q_a.data() + size_t(pos) * 4 * D);
      for (uint32_t m = 0; m < C; ++m)
        prep_m1(idx.data() + size_t(pos + m) * LD, pos + m, wq.data(), wk.data(), rope.data(), tail_b.data(),
                keys_b.data(), q_b.data() + size_t(pos + m) * 4 * D);
      pos += C;
      CHECK(tail_a == tail_b);   // the ring after the chunk == decode's after the same ids
      CHECK(keys_a == keys_b);
      ++runs;
    }
    CHECK(q_a == q_b);
  }
  std::printf("  indexer over a chunk: query heads, compressed keys and the 8-slot tail ring == decode's M = 1 chain "
              "bitwise after each of %u chunks (sizes 1..600, starts on and off multiples of 4)\n", runs);
}

void test_ple_chunk() {
  const uint32_t HN = qr::kHcN, KVN = HN + qr::kHidden, N = 70, R = qr::kPleRing, EOS = qr::kPleEos;
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> u(-2.f, 2.f), w(0.5f, 1.5f), tp(-0.5f, 0.5f);
  std::vector<float> kv(size_t(N) * KVN), wk(HN), wq(HN), wc(HN), taps(size_t(HN) * 4);
  for (float& v : kv) v = u(rng);
  for (float& v : wk) v = w(rng);
  for (float& v : wq) v = w(rng);
  for (float& v : wc) v = w(rng);
  for (float& v : taps) v = rf(tp(rng));
  std::vector<uint16_t> H0(size_t(N) * HN);
  for (uint16_t& v : H0) v = rne(u(rng));
  std::vector<uint32_t> ids(N);
  for (uint32_t i = 0; i < N; ++i) ids[i] = (i * 2654435761u) % 248000u;
  for (uint32_t i : {5u, 16u, 17u, 40u}) ids[i] = EOS;   // EOS inside a chunk and at a chunk edge
  uint32_t chunks = 0;
  for (const std::vector<uint32_t>& sizes :
       std::vector<std::vector<uint32_t>>{{1, 5, 16, 17, 31}, {40, 30}, {3, 1, 1, 9, 16, 40}}) {
    std::vector<uint16_t> Ha = H0, Hb = H0, ring_a(size_t(R) * HN, 0), ring_b = ring_a;
    std::vector<uint32_t> idr_a(R, 0), idr_b(R, 0);
    uint32_t pos = 0;
    for (uint32_t C : sizes) {
      // the ids: every row's history from the chunk or the ring == the loader's rule over the whole sequence
      const std::vector<loader::Q4PleHistory> hs = qp::ple_ids_chunk(ids.data() + pos, pos, C, idr_a.data());
      for (uint32_t m = 0; m < C; ++m) {
        const loader::Q4PleHistory want = loader::q4_ple_history(ids.data(), pos + m, EOS);
        CHECK(hs[m].t0 == want.t0 && hs[m].t1 == want.t1 && hs[m].t2 == want.t2);
      }
      qp::ple_chunk(Ha.data() + size_t(pos) * HN, kv.data() + size_t(pos) * KVN, ids.data() + pos, pos, C, wk.data(),
                    wq.data(), wc.data(), taps.data(), ring_a.data(), idr_a.data());
      for (uint32_t m = 0; m < C; ++m) {   // decode: one row a step, the rings written by the row itself
        const uint32_t p = pos + m;
        const uint16_t* hist[3];
        for (uint32_t t = 0; t < 3; ++t) {
          const uint32_t back = 9 - 3 * t;
          hist[t] = p >= back ? ring_b.data() + size_t((p - back) % R) * HN : nullptr;
        }
        std::vector<uint16_t> ring_out(HN);
        qr::ple_block(Hb.data() + size_t(p) * HN, kv.data() + size_t(p) * KVN, wk.data(), wq.data(), wc.data(),
                      taps.data(), hist, ring_out.data());
        std::copy(ring_out.begin(), ring_out.end(), ring_b.begin() + size_t(p % R) * HN);
        idr_b[p % R] = ids[p];
      }
      pos += C;
      CHECK(ring_a == ring_b);
      CHECK(idr_a == idr_b);
      ++chunks;
    }
    CHECK(Ha == Hb);
  }
  std::printf("  PLE over a chunk: histories == the loader's EOS rule; gate / conv / ring (three launches) == decode's "
              "block row by row - H, the conv ring and the id ring bitwise after each of %u chunks\n", chunks);
}

}  // namespace

int main() {
  test_sort();
  test_independence();
  test_combine_vs_decode();
  test_dense_rows();
  test_sparse();
  test_prep_chunk();
  test_ple_chunk();
  std::puts("qwen4exp_pf_ref_test: PASS");
  return 0;
}
