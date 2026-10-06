// Spec 20d Task 1: the Kolibri-1 prefill references (tests/kernels/kolibri_pf_ref.h) against independent
// formulas, against 20c's decode references (kolibri_ref.h) and against kolibri_ref.py's own outputs
// (tests/kernels/kolibri_fixture.h) - host only, so the Mac runs it.
//
//   (a) the sort (Review Focus 2): random routes from kolibri_ref::route at C 1 / 37 / 300 / 2048, routes
//       with exact selection TIES at the cut (coarse logits and biases), and the adversaries - every token
//       to the same 6 experts (0, 127, 255, 256, 382, 383: the 256-lane boundaries; one lane's serial
//       worst case), a chunk that reaches tmax(C) EXACTLY (372 experts with 33 rows, 12 with 1: the most
//       padding; 820 tiles at C = 2048), an expert with exactly 32 rows (one tile, no padding) and one with
//       0 - every (token, slot) once, in a tile of its expert, ascending token, padding NONE, the shared
//       block 384 last with the tokens in order, the header; and the whole result equal to spec 15d's
//       pf_moe_ref::sort at Kolibri's shape (one layout, two homes);
//   (b) row independence: the chunk REVERSED, and the rows permuted inside every tile, leave every token's
//       combine output bitwise unchanged (a row's y is a function of its own A row and its expert only);
//   (c) the combine == 20c's decode chain (kolibri_ref::combine) bit for bit on every token and column, and
//       == torch's (the fixture's kCombine row: route row 0, y_j fine) bit for bit;
//   (d) the windowed flash walk in fp64 == the direct softmax over the visible keys (kolibri_ref::attention)
//       on Review Focus 1's chunks - c0 = 0, 1000, 1001 (a row group whose first tile is WHOLLY masked for
//       its last row), 4396 (= 4096 + 300: the window wrapped in the ring) with rows 0, 1, 7, 23, 511, 512,
//       513 and C - 1 of C = 2048 - the keys read through a ring the previous chunks and this chunk wrote
//       in position order (a chunk of 2048 never overwrites a key a row of it still needs), and a full
//       layer's linear cache; the tiles start at absolute multiples of 64;
//       eager_window against the fixture's eager rows (positions 5, 512, 513, 700 sliding, 700 full):
//       scores, probabilities AND outputs bitwise torch's (P·V as one ascending fp32 chain over the row's
//       keys reproduces torch's bf16 matmul on these rows; 20c's blocked decode order is 1 ulp off);
//   (e) the bf16 slab (the o_proj tail: 2560 = 1024 + 1024 + 512, its 512 padding columns zero) and the
//       dequants' block 384 (the shared expert's bf16 tiles copied, not dequantised).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/kolibri_fixture.h"
#include "kernels/kolibri_pf_ref.h"
#include "kernels/kolibri_ref.h"
#include "kernels/pf_moe_ref.h"

namespace {

namespace kr = kolibri_ref;
namespace kp = kolibri_pf_ref;
namespace fx = kolibri_fixture;
using kr::f32;
using kr::rf;
using kr::rne;
constexpr uint32_t W = kr::kWords;

int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

// A kol_route row from the reference's Route.
void put_row(uint32_t* row, const kr::Route& r) {
  std::fill(row, row + W, 0u);
  for (uint32_t j = 0; j < kr::kTopK; ++j) {
    row[kr::kIds + j] = r.ids[j];
    row[kr::kWeights + j] = kr::as_u32(r.w[j]);
    row[kr::kSel + j] = kr::as_u32(r.sel[j]);
  }
  row[kr::kNext] = kr::as_u32(r.next);
}
// A row with given ids (ascending) and weights.
void put_ids(uint32_t* row, const uint32_t* ids, const float* w) {
  std::fill(row, row + W, 0u);
  for (uint32_t j = 0; j < kr::kTopK; ++j) {
    row[kr::kIds + j] = ids[j];
    row[kr::kWeights + j] = kr::as_u32(w[j]);
  }
}

// C tokens routed by kolibri_ref::route; `coarse` puts logits and biases on small grids (exact ties).
std::vector<uint32_t> routed(uint32_t C, uint32_t seed, bool coarse, uint32_t* ties) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-3.f, 3.f);
  std::uniform_int_distribution<int> lq(-4, 4), bq(0, 3);
  std::vector<float> bias(kr::kRouterN, 0.0f), lg(kr::kRouterN, 1e30f);
  for (uint32_t e = 0; e < kr::kExperts; ++e) bias[e] = coarse ? 0.25f * float(bq(rng)) : 0.1f * u(rng);
  std::vector<uint32_t> rows(size_t(C) * W);
  for (uint32_t t = 0; t < C; ++t) {
    for (uint32_t e = 0; e < kr::kExperts; ++e) lg[e] = coarse ? 0.5f * float(lq(rng)) : u(rng);
    const kr::Route r = kr::route(lg.data(), bias.data());
    for (uint32_t j = 0; j < kr::kTopK; ++j)
      if (r.sel[j] == r.next) ++*ties;   // a tie at the cut: the selected one has the lower id
    put_row(rows.data() + size_t(t) * W, r);
  }
  return rows;
}

// Route rows from an expert multiset laid out column-major: token t's slot j takes list[j x C + t], so the
// six of a token come from six different stretches - distinct as long as no expert has more than C rows.
std::vector<uint32_t> from_list(const std::vector<uint32_t>& list, uint32_t C) {
  CHECK_EQ(list.size(), size_t(C) * kr::kTopK);
  std::vector<uint32_t> rows(size_t(C) * W);
  for (uint32_t t = 0; t < C; ++t) {
    uint32_t ids[6];
    float w[6];
    for (uint32_t j = 0; j < 6; ++j) {
      ids[j] = list[size_t(j) * C + t];
      w[j] = 0.1f + 0.1f * float(j);
    }
    std::sort(ids, ids + 6);
    for (uint32_t j = 1; j < 6; ++j) CHECK(ids[j] != ids[j - 1]);
    put_ids(rows.data() + size_t(t) * W, ids, w);
  }
  return rows;
}

// Every invariant of the sorted layout (kol_pf_moe.cl's "no pair is lost" argument).
void check_sorted(const kp::Sorted& st, const uint32_t* route, uint32_t C) {
  const uint32_t T = kp::tmax(C), E = kr::kExperts, K = kr::kTopK;
  CHECK(st.hdr[kp::kHdrTiles] <= T);
  CHECK_EQ(st.hdr[kp::kHdrTiles], st.tiles_used);
  CHECK_EQ(st.hdr[kp::kHdrRows], st.hdr[kp::kHdrTiles] * kp::kTm);
  CHECK_EQ(st.hdr[kp::kHdrC], C);
  CHECK_EQ(st.hdr[kp::kHdrCount + E], C);
  std::vector<uint32_t> cnt(E, 0), seen(size_t(T) * kp::kTm, 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < K; ++j) {
      const uint32_t e = route[size_t(t) * W + j];
      const uint32_t r = st.pair_row[size_t(t) * K + j];
      CHECK(r < st.rows_used());
      CHECK_EQ(st.row_tok[r], t);
      CHECK_EQ(st.tiles[2 * (r / kp::kTm)], e);
      CHECK_EQ(seen[r], 0u);
      seen[r] = 1;
      ++cnt[e];
    }
  for (uint32_t e = 0; e < E; ++e) CHECK_EQ(st.hdr[kp::kHdrCount + e], cnt[e]);
  for (uint32_t tile = 0; tile < st.hdr[kp::kHdrTiles]; ++tile) {
    const uint32_t b = st.tiles[2 * tile], r0 = st.tiles[2 * tile + 1];
    CHECK_EQ(r0, tile * kp::kTm);
    CHECK(b <= kp::kShared);
    for (uint32_t i = 1; i < kp::kTm; ++i) {
      const uint32_t a = st.row_tok[r0 + i - 1], c = st.row_tok[r0 + i];
      if (c != kp::kNone) CHECK(a != kp::kNone && (b == kp::kShared ? c == a + 1 : c > a));
    }
    if (tile > 0) CHECK(st.tiles[2 * (tile - 1)] <= b);   // expert-major, the shared block last
  }
  for (uint32_t tile = st.hdr[kp::kHdrTiles]; tile < T; ++tile) CHECK_EQ(st.tiles[2 * tile], kp::kNone);
  for (uint32_t t = 0; t < C; ++t) CHECK_EQ(st.row_tok[st.shared_row() + t], t);
  for (uint32_t r = st.shared_row() + C; r < st.rows_used(); ++r) CHECK_EQ(st.row_tok[r], kp::kNone);
  // one layout, two homes: spec 15d's sort at Kolibri's shape
  const pf_moe_ref::Sorted o = pf_moe_ref::sort(route, C, moe_ref::Shape{E, K, kr::kHidden, kr::kInter, kr::kRouterN});
  CHECK(o.hdr.size() <= st.hdr.size());
  CHECK(std::equal(o.hdr.begin(), o.hdr.end(), st.hdr.begin()));
  CHECK(o.tiles == st.tiles && o.row_tok == st.row_tok && o.pair_row == st.pair_row);
}

void test_sort() {
  CHECK_EQ(kp::tmax(2048), 820u);
  CHECK_EQ(kp::tmax(1), 372u + 1u);   // (6 + 11904) / 32 = 372, + 1 shared tile
  CHECK_EQ(kp::hdr_words(), 400u);
  uint32_t ties = 0, tied_rows = 0;
  for (uint32_t C : {1u, 37u, 300u, 2048u}) {
    const std::vector<uint32_t> r = routed(C, C, false, &ties);
    check_sorted(kp::sort(r.data(), C), r.data(), C);
    const std::vector<uint32_t> rt = routed(C, C + 7, true, &tied_rows);
    check_sorted(kp::sort(rt.data(), C), rt.data(), C);
  }
  CHECK(tied_rows > 0);
  const uint32_t C = 2048;
  // every token to the same six experts - the lane boundaries 255 / 256 and the last expert 383
  {
    const uint32_t six[6] = {0, 127, 255, 256, 382, 383};
    std::vector<uint32_t> list;
    for (uint32_t j = 0; j < 6; ++j) list.insert(list.end(), C, six[j]);
    const std::vector<uint32_t> r = from_list(list, C);
    const kp::Sorted st = kp::sort(r.data(), C);
    check_sorted(st, r.data(), C);
    CHECK_EQ(st.tiles_used, 6u * 64u + 64u);
    for (uint32_t e : six) CHECK_EQ(st.hdr[kp::kHdrCount + e], C);
  }
  // the bound reached exactly: 372 experts with 33 rows (two tiles, 31 padding rows each), 12 with 1
  {
    std::vector<uint32_t> list;
    for (uint32_t e = 0; e < kr::kExperts; ++e) list.insert(list.end(), e < 372 ? 33u : 1u, e);
    const std::vector<uint32_t> r = from_list(list, C);
    const kp::Sorted st = kp::sort(r.data(), C);
    check_sorted(st, r.data(), C);
    CHECK_EQ(st.tiles_used, kp::tmax(C));
    CHECK_EQ(st.tiles_used, 820u);
  }
  // an expert with exactly 32 rows (one full tile), one with 0, the rest uniform-ish
  {
    std::vector<uint32_t> list;
    list.insert(list.end(), 32, 5u);   // expert 5: one whole tile
    // expert 6: none; 12288 - 32 = 12256 rows over the 382 others, in runs <= 33
    uint32_t left = 12256, e = 0;
    while (left) {
      if (e == 5 || e == 6) {
        ++e;
        continue;
      }
      const uint32_t n = std::min(left, 33u);
      list.insert(list.end(), n, e % kr::kExperts);
      left -= n;
      e = (e + 1) % kr::kExperts;
    }
    std::sort(list.begin(), list.end());
    const std::vector<uint32_t> r = from_list(list, C);
    const kp::Sorted st = kp::sort(r.data(), C);
    check_sorted(st, r.data(), C);
    CHECK_EQ(st.hdr[kp::kHdrCount + 5], 32u);
    CHECK_EQ(st.hdr[kp::kHdrCount + 6], 0u);
    uint32_t tiles5 = 0, tiles6 = 0;
    for (uint32_t t = 0; t < st.tiles_used; ++t) {
      tiles5 += st.tiles[2 * t] == 5;
      tiles6 += st.tiles[2 * t] == 6;
    }
    CHECK(tiles5 == 1 && tiles6 == 0);
    for (uint32_t t = 0; t < st.tiles_used; ++t)
      if (st.tiles[2 * t] == 5)
        for (uint32_t i = 0; i < kp::kTm; ++i) CHECK(st.row_tok[st.tiles[2 * t + 1] + i] != kp::kNone);
  }
  std::printf("  sort: random / tied (%u exact ties at the cut) at C 1 / 37 / 300 / 2048, all-to-6 (lanes 255 / "
              "256 / 383), the bound reached (820 tiles), a full tile and an empty expert - every invariant; == "
              "pf_moe_ref::sort\n", tied_rows);
}

// y rows from the A rows: row r = rne(Σ_{k < 16} x[row_tok[r]][k] w_e[k][n]) for an expert-dependent w
// (a function of the row's own A row and its expert only, as the grouped GEMM's), padding rows garbage.
std::vector<uint16_t> fake_y(const kp::Sorted& st, const std::vector<uint16_t>& x) {
  std::vector<uint16_t> y(size_t(st.rows_used()) * kr::kHidden, 0x7FC0);   // NaN-ish padding: never read
  for (uint32_t tile = 0; tile < st.tiles_used; ++tile) {
    const uint32_t e = st.tiles[2 * tile];
    for (uint32_t i = 0; i < kp::kTm; ++i) {
      const uint32_t r = st.tiles[2 * tile + 1] + i, t = st.row_tok[r];
      if (t == kp::kNone) continue;
      for (uint32_t n = 0; n < kr::kHidden; ++n) {
        float a = 0.f;
        for (uint32_t k = 0; k < 16; ++k)
          a += f32(x[size_t(t) * kr::kHidden + k]) * (float(int((e * 131 + n * 7 + k * 3) % 17) - 8) * 0.0625f);
        y[size_t(r) * kr::kHidden + n] = rne(a);
      }
    }
  }
  return y;
}

void test_independence_and_combine() {
  const uint32_t C = 300, H = kr::kHidden;
  uint32_t ties = 0;
  const std::vector<uint32_t> rt = routed(C, 41, false, &ties);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<uint16_t> x(size_t(C) * H);
  for (uint16_t& v : x) v = rne(u(rng));
  const kp::Sorted st = kp::sort(rt.data(), C);
  const std::vector<uint16_t> y = fake_y(st, x);
  std::vector<uint16_t> mo(size_t(C) * H);
  kp::combine(rt.data(), st, y.data(), mo.data(), C);
  // (c) == decode's chain on every token and column
  for (uint32_t t = 0; t < C; ++t) {
    float w[6];
    for (uint32_t j = 0; j < 6; ++j) w[j] = kr::as_f32(rt[size_t(t) * W + kr::kWeights + j]);
    for (uint32_t n = 0; n < H; ++n) {
      float d[7];
      for (uint32_t j = 0; j < 6; ++j) d[j] = f32(y[size_t(st.pair_row[size_t(t) * 6 + j]) * H + n]);
      d[6] = f32(y[size_t(st.shared_row() + t) * H + n]);
      CHECK_EQ(mo[size_t(t) * H + n], kr::combine(d, w));
    }
  }
  // (c) == torch's: the fixture's route row 0 weights, y_j fine (t 6 + j, seed 7), the shared (t 12)
  {
    const std::vector<uint32_t> fids = kr::hex32(fx::kRouteIds), fw = kr::hex32(fx::kRouteW);
    std::vector<uint32_t> row(W, 0);
    float w[6];
    uint32_t ids[6];
    for (uint32_t j = 0; j < 6; ++j) {
      ids[j] = fids[j];
      w[j] = kr::as_f32(fw[j]);
    }
    put_ids(row.data(), ids, w);
    const kp::Sorted s1 = kp::sort(row.data(), 1);
    std::vector<uint16_t> y1(size_t(s1.rows_used()) * H, 0);
    for (uint32_t j = 0; j < 6; ++j)
      for (uint32_t n = 0; n < H; ++n)
        y1[size_t(s1.pair_row[j]) * H + n] = rne(kr::fixture_val(false, 6 + j, n, 7, 0.125f, 1.f));
    for (uint32_t n = 0; n < H; ++n) y1[size_t(s1.shared_row()) * H + n] = rne(kr::fixture_val(false, 12, n, 7, 0.125f, 1.f));
    std::vector<uint16_t> m1(H);
    kp::combine(row.data(), s1, y1.data(), m1.data(), 1);
    CHECK(m1 == kr::hex16(fx::kCombine));
  }
  // (b) the chunk reversed
  {
    std::vector<uint16_t> xr(x.size());
    std::vector<uint32_t> rr(rt.size());
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t v = C - 1 - t;
      std::copy(x.begin() + size_t(t) * H, x.begin() + size_t(t + 1) * H, xr.begin() + size_t(v) * H);
      std::copy(rt.begin() + size_t(t) * W, rt.begin() + size_t(t + 1) * W, rr.begin() + size_t(v) * W);
    }
    const kp::Sorted sr = kp::sort(rr.data(), C);
    const std::vector<uint16_t> yr = fake_y(sr, xr);
    std::vector<uint16_t> mr(size_t(C) * H);
    kp::combine(rr.data(), sr, yr.data(), mr.data(), C);
    for (uint32_t t = 0; t < C; ++t)
      CHECK(std::equal(mo.begin() + size_t(t) * H, mo.begin() + size_t(t + 1) * H, mr.begin() + size_t(C - 1 - t) * H));
  }
  // (b) the rows permuted inside every tile (row_tok and pair_row moved with them)
  {
    kp::Sorted sp = st;
    std::mt19937 g(43);
    for (uint32_t tile = 0; tile < sp.tiles_used; ++tile) {
      const uint32_t r0 = sp.tiles[2 * tile + 1];
      std::vector<uint32_t> perm(kp::kTm);
      for (uint32_t i = 0; i < kp::kTm; ++i) perm[i] = i;
      std::shuffle(perm.begin(), perm.end(), g);
      std::vector<uint32_t> tok(kp::kTm);
      for (uint32_t i = 0; i < kp::kTm; ++i) tok[perm[i]] = st.row_tok[r0 + i];
      for (uint32_t i = 0; i < kp::kTm; ++i) sp.row_tok[r0 + i] = tok[i];
    }
    for (uint32_t t = 0; t < C; ++t)   // pair_row follows its row
      for (uint32_t j = 0; j < 6; ++j) {
        const uint32_t e = rt[size_t(t) * W + j], r = st.pair_row[size_t(t) * 6 + j], tile = r / kp::kTm;
        for (uint32_t i = 0; i < kp::kTm; ++i)
          if (sp.row_tok[tile * kp::kTm + i] == t && sp.tiles[2 * tile] == e) sp.pair_row[size_t(t) * 6 + j] = tile * kp::kTm + i;
      }
    // the shared rows were permuted too, but combine reads row shared_row + t: keep its tokens in order
    for (uint32_t i = 0; i < (C + kp::kTm - 1) / kp::kTm * kp::kTm; ++i) sp.row_tok[st.shared_row() + i] = st.row_tok[st.shared_row() + i];
    const std::vector<uint16_t> yp = fake_y(sp, x);
    std::vector<uint16_t> mp(size_t(C) * H);
    kp::combine(rt.data(), sp, yp.data(), mp.data(), C);
    CHECK(mp == mo);
  }
  std::puts("  combine: == decode's chain (kolibri_ref::combine) on 300 tokens x 2560 and == torch's fixture row, "
            "bitwise; the chunk reversed and the rows permuted inside every tile leave every token bitwise");
}

// A deterministic key / value of absolute position p, kv head j, dim d.
uint16_t kv_val(uint32_t p, uint32_t j, uint32_t d, uint32_t which) {
  return rne(kr::fixture_val(false, 30 + which, (p * 4 + j) * 128 + d, 11, 0.125f, 1.0f));
}

void test_flash() {
  const uint32_t C = 2048, H = kr::kQHeads, HD = kr::kHd, KVN = kr::kKvHeads * HD;
  std::mt19937 rng(50);
  std::uniform_real_distribution<float> u(-2.f, 2.f);
  std::vector<float> q(size_t(C) * H * HD);
  for (float& v : q) v = rf(u(rng));
  double worst = 0;
  uint32_t masked_first = 0, checked = 0;
  struct Case { uint32_t c0; bool sliding; };
  for (const Case c : {Case{0, true}, Case{1000, true}, Case{1001, true}, Case{4396, true}, Case{0, false},
                       Case{1001, false}, Case{4396, false}}) {
    // the ring as the chunks wrote it, in position order through this chunk's last key; a linear cache
    const uint32_t end = c.c0 + C, rows = c.sliding ? kr::kRing : end;
    std::vector<uint16_t> kc(size_t(rows) * KVN), vc(size_t(rows) * KVN), kl(size_t(end) * KVN), vl(size_t(end) * KVN);
    for (uint32_t p = 0; p < end; ++p)
      for (uint32_t j = 0; j < kr::kKvHeads; ++j)
        for (uint32_t d = 0; d < HD; ++d) {
          const size_t at = (size_t(kr::ring_row(p, c.sliding)) * kr::kKvHeads + j) * HD + d,
                       li = (size_t(p) * kr::kKvHeads + j) * HD + d;
          kc[at] = kl[li] = kv_val(p, j, d, 0);
          vc[at] = vl[li] = kv_val(p, j, d, 1);
        }
    std::vector<uint32_t> trows = {0, 1, 7, 23, 511, 512, 513, C - 1};
    for (uint32_t t : trows) {
      const uint32_t p = c.c0 + t, r0 = t / 8 * 8, lo = kr::key_lo(p, c.sliding);
      const uint32_t t0 = kp::first_tile(c.c0 + r0, c.sliding);
      CHECK(t0 % kp::kKt == 0 && t0 <= lo);
      CHECK((t0 & (kr::kRing - 1)) + kp::kKt <= kr::kRing);   // no 2D read crosses the ring's end
      if (t0 + kp::kKt <= lo) ++masked_first;                  // this row's first tile is wholly masked
      for (uint32_t h = 0; h < H; h += 5) {
        const std::vector<double> f = kp::flash_row(q.data(), kc.data(), vc.data(), c.c0, C, t, c.sliding, h);
        const std::vector<double> want = kr::attention(q.data() + (size_t(t) * H + h) * HD, kl.data(), vl.data(), lo, p,
                                                       false, h / 12);
        for (uint32_t d = 0; d < HD; ++d) worst = std::max(worst, std::fabs(f[d] - want[d]));
        ++checked;
      }
    }
  }
  CHECK(worst < 1e-12);
  CHECK(masked_first > 0);
  std::printf("  flash walk (fp64): %u (row, head) pairs at c0 0 / 1000 / 1001 / 4396, sliding through the ring and "
              "full, == the direct softmax within %.1e; %u rows start on a wholly masked tile\n", checked, worst,
              masked_first);

  // eager_window against torch's eager rows (the fixture's cases; kv head 0, 12 q heads)
  for (uint32_t ci = 0; ci < fx::kAttnCases; ++ci) {
    const fx::AttnCase& fc = fx::kAttn[ci];
    const bool sliding = fc.sliding != 0;
    const uint32_t pos = fc.query, lo = kr::key_lo(pos, sliding), L = pos - lo + 1, rows = sliding ? kr::kRing : pos + 1;
    std::vector<uint16_t> kc(size_t(rows) * KVN, 0), vc(size_t(rows) * KVN, 0);
    for (uint32_t p = lo; p <= pos; ++p)
      for (uint32_t d = 0; d < HD; ++d) {
        const size_t at = size_t(kr::ring_row(p, sliding)) * KVN + d;
        kc[at] = rne(kr::fixture_val(true, 21, p * 128 + d, 9, 0.125f));
        vc[at] = rne(kr::fixture_val(true, 22, p * 128 + d, 9, 0.125f));
      }
    std::vector<float> qh(size_t(H) * HD, 0.0f);   // the row's 48 heads; the fixture's 12 are kv head 0's
    for (uint32_t i = 0; i < 12 * HD; ++i) qh[i] = kr::fixture_val(true, 20, i, ci, 0.5f);
    const std::vector<uint16_t> s = kr::hex16(fc.s), pr = kr::hex16(fc.p), o = kr::hex16(fc.o);
    std::vector<uint16_t> ow(size_t(H) * HD);
    kp::eager_window(qh.data(), kc.data(), vc.data(), pos, 1, sliding, ow.data());
    const kp::EagerRow e0 = kp::eager_row(qh.data(), kc.data(), vc.data(), pos, sliding, 0);
    for (uint32_t i = 0; i < L; ++i) {
      CHECK_EQ(rne(e0.s[i]), s[i]);
      CHECK_EQ(rne(e0.p[i]), pr[i]);
    }
    size_t same = 0;
    int worst_o = 0;
    for (uint32_t i = 0; i < 12 * HD; ++i) {
      worst_o = std::max(worst_o, ulps16(ow[i], o[i]));
      same += ow[i] == o[i];
    }
    CHECK_EQ(worst_o, 0);
    std::printf("  eager_window %s at %u (keys %u..%u): scores, probabilities and outputs (%zu / 1536) bitwise "
                "torch's\n", sliding ? "sliding" : "full", pos, lo, pos, same);
  }
}

void test_slabs() {
  const uint32_t H = kr::kHidden;
  // o_proj's widths: N = 2560 = 1024 + 1024 + 512 (pad256(512) = 512, no padding), and a synthetic
  // N = 2400 whose 352-column tail is a 512-wide slab with 160 zero columns (K kept small)
  for (uint32_t N : {2560u, 2400u}) {
    const uint32_t K = 256;
    std::mt19937 rng(N);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<uint16_t> wr(size_t(N) * K);
    for (uint16_t& v : wr) v = rne(u(rng));
    std::vector<uint16_t> tiles(((N + 15) / 16 * 16) * size_t(K), 0);
    common::repack_bf16_tiled(wr.data(), K, N, tiles.data());
    for (uint32_t n0 = 0; n0 < N; n0 += 1024) {
      const uint32_t ns = N - n0 >= 1024 ? 1024 : (N - n0 + 255) / 256 * 256;
      const std::vector<uint16_t> s = kp::bf16_slab(tiles.data(), K, N, n0, ns);
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t c = 0; c < ns; ++c)
          CHECK_EQ(s[size_t(k) * ns + c], n0 + c < N ? wr[size_t(n0 + c) * K + k] : uint16_t(0));
    }
  }
  // the shared expert's block 384: the bf16 tiles copied, gate||up interleave16 kept
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-0.05f, 0.05f);
  std::vector<uint16_t> gr(size_t(2 * kr::kInter) * H), dr(size_t(H) * kr::kInter);
  for (uint16_t& v : gr) v = rne(u(rng));
  for (uint16_t& v : dr) v = rne(u(rng));
  std::vector<uint16_t> gt(gr.size()), dt(dr.size());
  common::repack_bf16_tiled(gr.data(), H, 2 * kr::kInter, gt.data());
  common::repack_bf16_tiled(dr.data(), kr::kInter, H, dt.data());
  const std::vector<uint16_t> g = kp::dequant_gu(nullptr, gt.data(), kp::kShared), d = kp::dequant_dn(nullptr, dt.data(), kp::kShared);
  for (uint32_t k = 0; k < H; k += 37)
    for (uint32_t n = 0; n < 2 * kr::kInter; ++n) CHECK_EQ(g[size_t(k) * 2 * kr::kInter + n], gr[size_t(n) * H + k]);
  for (uint32_t k = 0; k < kr::kInter; k += 13)
    for (uint32_t n = 0; n < H; ++n) CHECK_EQ(d[size_t(k) * H + n], dr[size_t(n) * kr::kInter + k]);
  std::puts("  slabs: the bf16 slab's columns and its zero tail; the dequants' block 384 = the shared expert's bf16 "
            "tiles, row-major");
}

}  // namespace

int main() {
  test_sort();
  test_independence_and_combine();
  test_flash();
  test_slabs();
  std::puts("kolibri_pf_ref_test OK");
  return 0;
}
