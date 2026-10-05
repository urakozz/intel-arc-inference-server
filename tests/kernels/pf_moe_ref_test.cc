// Spec 15d: the prefill MoE host reference (tests/kernels/pf_moe_ref.h) on the host - the
// reference tests/kernels/pf_moe_test.cc holds the kernels to must itself route, sort,
// combine and multiply as the design says. Host only, no device.
//
//   1. the tile bound: tmax(C) covers every count distribution - the adversarial ones
//      (one row for as many experts as the pairs allow, one expert holding every token,
//      C = 1) included;
//   2. the sort: every (token, slot) pair gets exactly one row, in its expert's tiles, in
//      ascending token order within the expert; padding rows are NONE and only at an
//      expert's end; the shared expert's row Rs + t is token t; empty experts own no tile;
//      the table past the used tiles is NONE; the header's counts are the pairs';
//   3. row independence (spec 15d Review Focus 1): permuting the rows inside every tile -
//      and moving rows across tiles of the same expert - leaves every row's grouped-GEMM
//      output (gate||up + SiLU, and down) bitwise unchanged;
//   4. the combine is decode's: combine_token over the sorted rows equals moe_ref::combine
//      (moe_down's epilogue) bit for bit, on random terms and on terms whose bf16 sum
//      depends on the order (fixed slot order);
//   5. the whole chunk (route, sort, gather, gate||up, SiLU, down, combine) against the
//      decode reference token by token (moe_ref's int4 GEMV chain): the residual within a
//      few bf16 ulps (the two GEMM orders round differently) - the prefill data flow's
//      indexing (pair_row, the shared rows, the interleave) end to end;
//   6. chunk boundaries: the same tokens as one chunk or as two give bitwise the same
//      results (rows are independent of the chunk they ride in).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "pf_moe_ref.h"

namespace {

using moe_ref::f32;
using moe_ref::rne;
using pf_moe_ref::kNone;
using pf_moe_ref::kTm;
namespace hk = kernels::pf_moe;


// Route rows [C][kWords] with the given expert ids per token (weights / gate unset).
std::vector<uint32_t> route_rows(const std::vector<std::vector<uint32_t>>& ids) {
  std::vector<uint32_t> r(ids.size() * moe_ref::kWords, 0);
  for (size_t t = 0; t < ids.size(); ++t)
    for (size_t k = 0; k < ids[t].size(); ++k) r[t * moe_ref::kWords + moe_ref::kIds + k] = ids[t][k];
  return r;
}

// top_k distinct random experts per token.
std::vector<std::vector<uint32_t>> random_ids(uint32_t C, const moe_ref::Shape& s, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint32_t> all(s.experts);
  std::iota(all.begin(), all.end(), 0u);
  std::vector<std::vector<uint32_t>> ids(C);
  for (auto& v : ids) {
    std::shuffle(all.begin(), all.end(), rng);
    v.assign(all.begin(), all.begin() + s.top_k);
  }
  return ids;
}

// --- 1, 2: the sort's invariants and the tile bound -----------------------------------
void check_sorted(const pf_moe_ref::Sorted& st, const std::vector<std::vector<uint32_t>>& ids,
                  const moe_ref::Shape& s, const char* what) {
  const uint32_t C = uint32_t(ids.size()), T = pf_moe_ref::tmax(s, C);
  CHECK_EQ(st.tiles.size(), size_t(T) * 2);
  CHECK(st.tiles_used() <= T);
  CHECK_EQ(st.rows_used(), st.tiles_used() * kTm);
  std::vector<uint32_t> cnt(s.experts, 0);
  for (const auto& v : ids)
    for (uint32_t e : v) ++cnt[e];
  for (uint32_t e = 0; e < s.experts; ++e) CHECK_EQ(st.hdr[hk::kHdrCount + e], cnt[e]);
  CHECK_EQ(st.hdr[hk::kHdrCount + s.experts], C);
  CHECK_EQ(st.hdr[hk::kHdrC], C);
  // the block of every used row
  std::vector<uint32_t> row_block(st.rows_used(), kNone);
  uint32_t prev_block = 0;
  for (uint32_t t = 0; t < T; ++t) {
    const uint32_t b = st.tiles[2 * t];
    if (t >= st.tiles_used()) {
      CHECK_EQ(b, kNone);
      continue;
    }
    CHECK(b != kNone && b <= s.shared_block());
    CHECK(b >= prev_block);                       // expert-major, the shared expert last
    prev_block = b;
    CHECK_EQ(st.tiles[2 * t + 1], t * kTm);       // tiles are consecutive TM-row blocks
    if (b < s.experts) CHECK(cnt[b] > 0);         // an empty expert owns no tile
    for (uint32_t i = 0; i < kTm; ++i) row_block[t * kTm + i] = b;
  }
  // every pair exactly once, in its expert's rows
  std::vector<int> seen(st.rows_used(), 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t k = 0; k < s.top_k; ++k) {
      const uint32_t r = st.pair_row[size_t(t) * s.top_k + k];
      CHECK(r < st.rows_used());
      CHECK_EQ(row_block[r], ids[t][k]);
      CHECK_EQ(st.row_tok[r], t);
      ++seen[r];
    }
  // ascending tokens within an expert, padding only at its end, the shared rows in order
  std::vector<uint32_t> next(s.experts, 0);       // smallest token the next row may hold
  std::vector<bool> in_pad(s.experts, false);
  for (uint32_t r = 0; r < st.rows_used(); ++r) {
    const uint32_t b = row_block[r], tok = st.row_tok[r];
    if (b == s.shared_block()) {
      const uint32_t i = r - st.shared_row();
      CHECK_EQ(tok, i < C ? i : kNone);
      continue;
    }
    if (tok == kNone) {
      in_pad[b] = true;
      continue;
    }
    CHECK(!in_pad[b]);
    CHECK_EQ(seen[r], 1);
    CHECK(tok >= next[b]);
    next[b] = tok + 1;
  }
  CHECK_EQ(st.shared_row() % kTm, 0u);
  for (uint32_t e = 0; e < s.experts; ++e) {   // the padding is less than one tile
    uint32_t rows = 0;
    for (uint32_t r = 0; r < st.rows_used(); ++r) rows += row_block[r] == e;
    CHECK_EQ(rows, (cnt[e] + kTm - 1) / kTm * kTm);
  }
  std::printf("  sort %-24s C %4u: %4u of %4u tiles, %5u rows, %3u empty experts, max %4u rows\n",
              what, C, st.tiles_used(), T, st.rows_used(),
              uint32_t(std::count(cnt.begin(), cnt.end(), 0u)),
              *std::max_element(cnt.begin(), cnt.end()));
}

void test_sort() {
  const moe_ref::Shape s = moe_ref::ornith_shape();
  for (uint32_t C : {1u, 7u, 31u, 32u, 33u, 64u, 100u, 777u, 2048u}) {
    const auto ids = random_ids(C, s, 100 + C);
    const auto rr = route_rows(ids);
    check_sorted(pf_moe_ref::sort(rr.data(), C, s), ids, s, "random");
  }
  // every token to experts 0..7 in reverse slot order: one expert per slot holds all C
  // rows, 248 experts are empty
  for (uint32_t C : {1u, 2048u}) {
    std::vector<std::vector<uint32_t>> ids(C);
    for (auto& v : ids) v = {7, 6, 5, 4, 3, 2, 1, 0};
    const auto rr = route_rows(ids);
    check_sorted(pf_moe_ref::sort(rr.data(), C, s), ids, s, "all to experts 0..7");
  }
  // the tile bound's adversary: one row for as many experts as possible (every expert at
  // 1 mod TM wastes TM - 1 rows), the rest of the pairs on a fixed eight
  for (uint32_t C : {1u, 5u, 32u, 33u, 2048u}) {
    std::vector<std::vector<uint32_t>> ids(C);
    uint32_t single = 8;   // experts 8.. get one row each, while they last
    for (uint32_t t = 0; t < C; ++t) {
      for (uint32_t k = 0; k < s.top_k; ++k) {
        if (k == 0 && single < s.experts) ids[t].push_back(single++);
        else ids[t].push_back(k);
      }
      std::sort(ids[t].begin(), ids[t].end());
      ids[t].erase(std::unique(ids[t].begin(), ids[t].end()), ids[t].end());
      for (uint32_t e = 0; ids[t].size() < s.top_k; ++e)
        if (std::find(ids[t].begin(), ids[t].end(), e) == ids[t].end()) ids[t].push_back(e);
    }
    const auto rr = route_rows(ids);
    check_sorted(pf_moe_ref::sort(rr.data(), C, s), ids, s, "one-row experts");
  }
  // an id past the last expert clamps to it (the kernel's byte staging), never off the table
  {
    std::vector<std::vector<uint32_t>> ids = {{999, 1, 2, 3, 4, 5, 6, 7}};
    const auto rr = route_rows(ids);
    const pf_moe_ref::Sorted st = pf_moe_ref::sort(rr.data(), 1, s);
    CHECK_EQ(st.hdr[hk::kHdrCount + s.experts - 1], 1u);
    CHECK(st.pair_row[0] < st.rows_used());
  }
}

// Random layout-1 blocks, as tests/kernels/moe_test.cc generates them.
std::vector<uint32_t> random_blocks(size_t words, uint32_t seed) {
  std::vector<uint32_t> v(words);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> sd(0.01f, 0.05f);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = rng();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(sd(rng))) | (uint32_t(common::f32_to_f16(sd(rng))) << 16);
  }
  return v;
}
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(d(rng));
  return v;
}

// A small MoE shape so the whole-chunk reference runs in a second.
const moe_ref::Shape kSmall{16, 4, 128, 64, 32};

// --- 3: row independence --------------------------------------------------------------
struct Experts {
  std::vector<std::vector<uint16_t>> gu, dn;   // per block, bf16 [K][N]
};
Experts dequant_all(const std::vector<uint32_t>& gu, const std::vector<uint32_t>& dn,
                    const moe_ref::Shape& s) {
  Experts x;
  for (uint32_t b = 0; b <= s.experts; ++b) {
    x.gu.push_back(pf_moe_ref::dequant_block(gu.data() + b * s.gate_up_words(), s.hidden, 2 * s.inter));
    x.dn.push_back(pf_moe_ref::dequant_block(dn.data() + b * s.down_words(), s.inter, s.hidden));
  }
  return x;
}
// The grouped GEMMs over the reference's tile walk: h [rows][inter] (gate||up + SiLU),
// y [rows][hidden] (down, rne'd).
void grouped_both(const pf_moe_ref::Sorted& st, const std::vector<uint16_t>& xg, const Experts& w,
                  const moe_ref::Shape& s, std::vector<uint16_t>& h, std::vector<uint16_t>& y) {
  h.assign(size_t(st.rows_used()) * s.inter, 0);
  y.assign(size_t(st.rows_used()) * s.hidden, 0);
  pf_moe_ref::grouped(st, 0, s.experts + 1, [&](uint32_t b, uint32_t r) {
    const std::vector<uint16_t> hr = pf_moe_ref::silu_row(
        pf_moe_ref::gemm_rows(xg.data() + size_t(r) * s.hidden, w.gu[b].data(), s.hidden, 2 * s.inter),
        s.inter);
    std::copy(hr.begin(), hr.end(), h.begin() + size_t(r) * s.inter);
  });
  pf_moe_ref::grouped(st, 0, s.experts + 1, [&](uint32_t b, uint32_t r) {
    const std::vector<float> d =
        pf_moe_ref::gemm_rows(h.data() + size_t(r) * s.inter, w.dn[b].data(), s.inter, s.hidden);
    for (uint32_t n = 0; n < s.hidden; ++n) y[size_t(r) * s.hidden + n] = rne(d[n]);
  });
}

void test_row_independence() {
  const moe_ref::Shape& s = kSmall;
  const uint32_t C = 90;
  const auto ids = random_ids(C, s, 7);
  const auto rr = route_rows(ids);
  const pf_moe_ref::Sorted st = pf_moe_ref::sort(rr.data(), C, s);
  const Experts w = dequant_all(random_blocks(s.gate_up_words() * (s.experts + 1), 8),
                                random_blocks(s.down_words() * (s.experts + 1), 9), s);
  const std::vector<uint16_t> x = random_bf16(size_t(C) * s.hidden, -1.0f, 1.0f, 10);
  const std::vector<uint16_t> xg = pf_moe_ref::gather(st, x.data(), s.hidden);
  std::vector<uint16_t> h, y;
  grouped_both(st, xg, w, s, h, y);
  // (a) a random permutation of the rows inside every tile; (b) rows exchanged between
  // two tiles of the same block. dest[r] = where row r's operand goes.
  std::mt19937 rng(11);
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<uint32_t> dest(st.rows_used());
    std::iota(dest.begin(), dest.end(), 0u);
    if (mode == 0) {
      for (uint32_t t = 0; t < st.tiles_used(); ++t)
        std::shuffle(dest.begin() + t * kTm, dest.begin() + (t + 1) * kTm, rng);
    } else {
      uint32_t moved = 0;
      for (uint32_t t = 0; t + 1 < st.tiles_used(); ++t)
        if (st.tiles[2 * t] == st.tiles[2 * t + 2]) {   // two tiles of one block
          for (uint32_t i = 0; i < kTm; i += 3) std::swap(dest[t * kTm + i], dest[(t + 1) * kTm + i]);
          ++moved;
        }
      CHECK(moved > 0);
    }
    std::vector<uint16_t> xg2(xg.size());
    for (uint32_t r = 0; r < st.rows_used(); ++r)
      std::copy(xg.begin() + size_t(r) * s.hidden, xg.begin() + size_t(r + 1) * s.hidden,
                xg2.begin() + size_t(dest[r]) * s.hidden);
    std::vector<uint16_t> h2, y2;
    grouped_both(st, xg2, w, s, h2, y2);
    for (uint32_t r = 0; r < st.rows_used(); ++r) {
      CHECK(std::memcmp(&h[size_t(r) * s.inter], &h2[size_t(dest[r]) * s.inter], s.inter * 2) == 0);
      CHECK(std::memcmp(&y[size_t(r) * s.hidden], &y2[size_t(dest[r]) * s.hidden], s.hidden * 2) == 0);
    }
  }
  std::printf("  row independence: %u rows permuted inside their tiles and across tiles of one "
              "block, every row's h and y bitwise unchanged\n", st.rows_used());
}

// --- 4: the combine is decode's -------------------------------------------------------
void test_combine_is_decode() {
  const moe_ref::Shape s = moe_ref::ornith_shape();
  // One token, one column: rows 0..top_k-1 its routed terms, row top_k its shared one.
  pf_moe_ref::Sorted st;
  st.hdr.assign(hk::hdr_words(s.experts), 0);
  st.hdr[hk::kHdrSharedRow] = s.top_k;
  st.pair_row.resize(s.top_k);
  std::iota(st.pair_row.begin(), st.pair_row.end(), 0u);
  std::mt19937 rng(12);
  std::uniform_real_distribution<float> mag(-6.0f, 3.0f), sgn(-1.0f, 1.0f), wd(0.0f, 1.0f);
  uint64_t cases = 0, order_sensitive = 0;
  for (int it = 0; it < 200000; ++it) {
    std::vector<float> d(s.slots());
    for (float& v : d) v = std::copysign(std::exp2(mag(rng)), sgn(rng));
    if (it % 4 == 0) d[it % s.top_k] = std::copysign(256.0f, sgn(rng));   // one large term
    moe_ref::Route r;
    float wsum = 0;
    for (uint32_t k = 0; k < s.top_k; ++k) wsum += (r.w[k] = wd(rng) + 1e-3f);
    for (uint32_t k = 0; k < s.top_k; ++k) r.w[k] = moe_ref::rf(r.w[k] / wsum);
    r.sg = moe_ref::rf(wd(rng));
    std::vector<uint32_t> row(moe_ref::kWords, 0);
    for (uint32_t k = 0; k < s.top_k; ++k) std::memcpy(&row[moe_ref::kWeights + k], &r.w[k], 4);
    std::memcpy(&row[moe_ref::kSharedGate], &r.sg, 4);
    std::vector<uint16_t> y(s.slots());
    for (uint32_t k = 0; k < s.slots(); ++k) y[k] = rne(d[k]);
    const uint16_t resid = rne(sgn(rng) * 4.0f);
    const uint16_t want = moe_ref::combine(d.data(), r, s, resid);
    const uint16_t got = pf_moe_ref::combine_token(st, row.data(), y.data(), 1, s, 0, 0, resid);
    CHECK_EQ(got, want);
    // the reverse slot order differs on some terms - which is what makes the order a rule
    float rev = 0, fwd = 0;
    for (uint32_t k = s.top_k; k-- > 0;) rev += f32(rne(f32(y[k]) * r.w[k]));
    for (uint32_t k = 0; k < s.top_k; ++k) fwd += f32(rne(f32(y[k]) * r.w[k]));
    order_sensitive += rne(rev) != rne(fwd);
    ++cases;
  }
  // A crafted order-sensitive token: terms 2^24, 2^16 and six 1.0s (weights 1). Ascending
  // slots: 2^24 + 2^16 is exact in fp32 and every 1.0 is absorbed (ties to even), so the
  // fp32 sum sits on the bf16 midpoint and rounds DOWN to 2^24; descending: the 1.0s add
  // to 6 first and the sum lands above the midpoint, rounding UP. combine_token must give
  // the ascending result, as moe_ref::combine does.
  {
    std::vector<float> d(s.slots(), 1.0f);
    d[0] = 16777216.0f;
    d[1] = 65536.0f;
    d[s.top_k] = 0.0f;
    moe_ref::Route r;
    for (uint32_t k = 0; k < s.top_k; ++k) r.w[k] = 1.0f;
    r.sg = 0.0f;
    std::vector<uint32_t> row(moe_ref::kWords, 0);
    for (uint32_t k = 0; k < s.top_k; ++k) std::memcpy(&row[moe_ref::kWeights + k], &r.w[k], 4);
    std::vector<uint16_t> y(s.slots());
    for (uint32_t k = 0; k < s.slots(); ++k) y[k] = rne(d[k]);
    float rev = 0;
    for (uint32_t k = s.top_k; k-- > 0;) rev += f32(y[k]);
    const uint16_t got = pf_moe_ref::combine_token(st, row.data(), y.data(), 1, s, 0, 0, 0);
    CHECK_EQ(got, moe_ref::combine(d.data(), r, s, 0));
    CHECK_EQ(got, rne(16777216.0f));
    CHECK(rne(rev) != got);   // the other order really is a different bf16 value
  }
  std::printf("  combine: %llu random cases bitwise equal to moe_ref::combine (moe_down's "
              "epilogue; %llu of them order-sensitive) and the crafted order-sensitive token "
              "summed in ascending slot order\n", (unsigned long long)cases,
              (unsigned long long)order_sensitive);
}

// --- 5, 6: the whole chunk against decode, and chunk boundaries -------------------------
struct Chunk {
  std::vector<uint32_t> route;   // [C][kWords]
  std::vector<uint16_t> x, resid;
};
std::vector<uint16_t> prefill_chunk(const Chunk& c, uint32_t t0, uint32_t C, const Experts& w,
                                    const moe_ref::Shape& s) {
  const pf_moe_ref::Sorted st =
      pf_moe_ref::sort(c.route.data() + size_t(t0) * moe_ref::kWords, C, s);
  const std::vector<uint16_t> xg = pf_moe_ref::gather(st, c.x.data() + size_t(t0) * s.hidden, s.hidden);
  std::vector<uint16_t> h, y;
  grouped_both(st, xg, w, s, h, y);
  std::vector<uint16_t> out(size_t(C) * s.hidden);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t n = 0; n < s.hidden; ++n)
      out[size_t(t) * s.hidden + n] = pf_moe_ref::combine_token(
          st, c.route.data() + size_t(t0 + t) * moe_ref::kWords, y.data(), s.hidden, s, t, n,
          c.resid[size_t(t0 + t) * s.hidden + n]);
  return out;
}

void test_chunk_vs_decode() {
  const moe_ref::Shape& s = kSmall;
  const uint32_t C = 70;
  const std::vector<uint32_t> gu = random_blocks(s.gate_up_words() * (s.experts + 1), 20);
  const std::vector<uint32_t> dn = random_blocks(s.down_words() * (s.experts + 1), 21);
  const Experts w = dequant_all(gu, dn, s);
  Chunk c;
  c.x = random_bf16(size_t(C) * s.hidden, -1.0f, 1.0f, 22);
  c.resid = random_bf16(size_t(C) * s.hidden, -2.0f, 2.0f, 23);
  c.route.assign(size_t(C) * moe_ref::kWords, 0);
  std::vector<moe_ref::Route> routes;
  std::mt19937 rng(24);
  std::uniform_real_distribution<float> ld(-3.0f, 3.0f);
  for (uint32_t t = 0; t < C; ++t) {
    std::vector<float> lg(s.router_n, 0.0f);
    for (uint32_t e = 0; e <= s.experts; ++e) lg[e] = ld(rng);
    const moe_ref::Route r = moe_ref::route(lg.data(), s);
    routes.push_back(r);
    uint32_t* row = &c.route[size_t(t) * moe_ref::kWords];
    for (uint32_t k = 0; k < s.top_k; ++k) {
      row[moe_ref::kIds + k] = r.ids[k];
      std::memcpy(&row[moe_ref::kWeights + k], &r.w[k], 4);
    }
    std::memcpy(&row[moe_ref::kSharedGate], &r.sg, 4);
  }
  const std::vector<uint16_t> one = prefill_chunk(c, 0, C, w, s);
  // 5: token by token against decode's chain (moe_ref::gate_up / down, the int4 GEMV).
  // The two differ only by GEMM rounding (prefill: bf16-dequantised weights, fp32 sums;
  // decode: the per-group int4 sums), so the error is measured in bf16 ulps of the
  // largest term that went into the value - |resid|, each |w_k d_k|, |s_b d_s| - not of
  // the value itself: the block's terms cancel to near zero in places, and a near-zero
  // result's own ulps would measure the cancellation, not the arithmetic.
  double worst = 0;
  size_t exact = 0, total = 0;
  for (uint32_t t = 0; t < C; ++t) {
    const moe_ref::Route& r = routes[t];
    const std::vector<uint16_t> hd = moe_ref::gate_up(r, &c.x[size_t(t) * s.hidden], gu.data(), s, 1);
    const std::vector<uint16_t> od =
        moe_ref::down(r, hd, dn.data(), s, &c.resid[size_t(t) * s.hidden], 1);
    for (uint32_t n = 0; n < s.hidden; ++n) {
      float scale = std::fabs(f32(c.resid[size_t(t) * s.hidden + n]));
      for (uint32_t k = 0; k < s.slots(); ++k) {
        const uint32_t e = k < s.top_k ? r.ids[k] : s.shared_block();
        const float d = moe_ref::split_dot(dn.data() + e * s.down_words(), s.inter, n,
                                           hd.data() + size_t(k) * s.inter, 1);
        scale = std::max(scale, std::fabs(d * (k < s.top_k ? r.w[k] : r.sg)));
      }
      const float a = f32(one[size_t(t) * s.hidden + n]), b = f32(od[n]);
      int e2 = 0;
      std::frexp(std::max(scale, 1e-6f), &e2);
      worst = std::max(worst, std::fabs(double(a) - b) / std::ldexp(1.0, e2 - 8));
      exact += a == b;
      ++total;
    }
  }
  std::printf("  whole chunk vs decode's chain: %zu / %zu residual values exact, worst %.2f bf16 "
              "ulps of the largest term\n", exact, total, worst);
  // Measured 2026-10-05: 5745 / 8960 exact, worst 14.0 ulps of the largest term - the
  // bf16 dequantisation of the weights (2^-9 relative each) carried through gate||up, the
  // SiLU and down. A routing or indexing error (a wrong expert, slot, token or shared row)
  // is of the order of the term itself, ~256 ulps of it; the bar sits between the two.
  CHECK(worst <= 32.0);
  CHECK(exact * 10 >= total * 6);
  // 6: the same tokens as two chunks (37 + 33): every token's result bitwise the same.
  const std::vector<uint16_t> a = prefill_chunk(c, 0, 37, w, s);
  const std::vector<uint16_t> b = prefill_chunk(c, 37, C - 37, w, s);
  CHECK(std::memcmp(a.data(), one.data(), a.size() * 2) == 0);
  CHECK(std::memcmp(b.data(), one.data() + a.size(), b.size() * 2) == 0);
  std::puts("  chunk boundary: 70 tokens as one chunk and as 37 + 33, bitwise equal");
}

}  // namespace

int main() {
  test_sort();
  test_row_independence();
  test_combine_is_decode();
  test_chunk_vs_decode();
  std::puts("pf_moe_ref_test OK");
  return 0;
}
