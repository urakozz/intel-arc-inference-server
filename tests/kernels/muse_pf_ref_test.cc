// Spec 23d Task 1: the Muse-Glimmer prefill attention's host twins (tests/kernels/muse_pf_ref.h) against independent
// formulas and against 23c's decode references (tests/kernels/muse_ref.h) - host only, so the Mac runs it.
//
//   (a) Review Focus 1, the window over the ring: visible() is (p - 2048, p] sliding, [0, p] full; a chunk of 2048 at
//       c0 = 2048, 4096, 6143 needs keys [c0 - 2047, c0 + 2047] - 4095 positions whose ring slots are pairwise
//       distinct (no row's live key is overwritten by another row's write of the chunk), the first and the last row
//       of the chunk named; the first tile is an absolute multiple of 64 at most 63 keys below the group's lo;
//   (b) the windowed flash walk in fp64 == the direct softmax over the visible keys (keys read from per-POSITION
//       arrays, so the ring addressing is proved too) within 1e-12, at c0 = 0, 1000, 2048, 4096, 6143 (the ring
//       wrapped) on rows 0, 1, 7, 8, 63, 64, 1000, 2047 of C = 2048, and a full layer at depth 3000 / 20000;
//   (c) a row whose first tile holds NONE of its keys (the group's first row reaches further back): the walk from
//       that tile == the walk from the next one, BITWISE (m_safe: a wholly masked tile leaves (m, l, o) unchanged);
//       and the wrong form (m without the guard) is NaN there - the guard is load-bearing;
//   (d) row independence: the chunk walked in REVERSED row order, and with the rows permuted inside every 8-row
//       group, gives every row bitwise the same output; a row's output is the same whatever chunk it rides in -
//       splits at multiples of 64 BITWISE (Review Focus 2), a split at 2049 within cosine 0.99999 (printed: bitwise
//       or not - the twin's tiles are absolute, so it is expected bitwise here; the card's GEMMs are not the twin);
//   (e) flash's fp32 arithmetic (flash_window_sim: bf16 P, natural exp) against fp64 at cosine >= 0.99999 - sliding
//       through the ring and full to depth 30000 (MG2's bar for the card's kernel);
//   (f) eager_window (the EAGER build's 128-key tiles, lane sums and blocks) == museref::attention_eager_head's rne(o)
//       BITWISE on the same rows - decode's eager row - at 5, 127, 128, 129, 2047, 2048, 4097, 6143 sliding and
//       3000 / 9000 full; the wrong form (P·V as one fma chain, no blocks) differs somewhere - the structure is
//       load-bearing;
//   (g) gate_window == museref::gate_product (decode's mg_attn_reduce chain rne(rne(o) x gate)) per element.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "check.h"
#include "kernels/muse_pf_ref.h"
#include "kernels/muse_ref.h"

namespace {

namespace mp = musepf;
namespace mr = museref;
using mr::bf;
using mr::rne;
constexpr uint32_t HD = mp::kHd, QH = mp::kQH, KVH = mp::kKVH, KVN = KVH * HD, QN = QH * HD;

uint32_t hash32(uint32_t x) { return k2_ref::lowbias32(x); }
// A bf16 value per (tensor, position, element): coarse enough that scores spread, never NaN.
uint16_t val(uint32_t t, uint32_t p, uint32_t i, float amp) {
  const uint32_t h = hash32(hash32(t * 0x9E3779B9u + p) ^ i);
  return rne((float(h % 2001) / 1000.0f - 1.0f) * amp);
}

// The keys / values of positions [0, n) (per position) and the ring / cache the walk reads: written in position
// order, so slot s of the ring holds the LAST position mapping to it - what decode and the previous chunks leave.
struct Caches {
  std::vector<uint16_t> kpos, vpos, kc, vc;
};
Caches caches(uint32_t n, bool sliding, uint32_t seed) {
  Caches c;
  c.kpos.resize(size_t(n) * KVN);
  c.vpos.resize(c.kpos.size());
  for (uint32_t p = 0; p < n; ++p)
    for (uint32_t i = 0; i < KVN; ++i) {
      c.kpos[size_t(p) * KVN + i] = val(seed, p, i, 1.0f);
      c.vpos[size_t(p) * KVN + i] = val(seed + 1, p, i, 1.0f);
    }
  if (!sliding) {
    c.kc = c.kpos;
    c.vc = c.vpos;
    return c;
  }
  c.kc.assign(size_t(mp::kRing) * KVN, 0);
  c.vc.assign(c.kc.size(), 0);
  for (uint32_t p = 0; p < n; ++p) {
    const size_t s = size_t(p & (mp::kRing - 1)) * KVN;
    std::copy_n(c.kpos.begin() + size_t(p) * KVN, KVN, c.kc.begin() + s);
    std::copy_n(c.vpos.begin() + size_t(p) * KVN, KVN, c.vc.begin() + s);
  }
  return c;
}
// q of rows [0, rows) of a chunk at c0 (bf16 [rows][32][128]); larger amplitude so the softmax is not flat.
std::vector<uint16_t> queries(uint32_t c0, uint32_t rows, uint32_t seed) {
  std::vector<uint16_t> q(size_t(rows) * QN);
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t i = 0; i < QN; ++i) q[size_t(t) * QN + i] = val(seed, c0 + t, i, 4.0f);
  return q;
}
double max_abs(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
double cosine(const double* a, const double* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return ab / std::sqrt(aa * bb);
}

void test_visible() {   // (a)
  CHECK(mp::visible(0, true) == std::make_pair(0u, 0u));
  CHECK(mp::visible(2047, true) == std::make_pair(0u, 2047u));
  CHECK(mp::visible(2048, true) == std::make_pair(1u, 2048u));   // 2048 keys including the query
  CHECK(mp::visible(9000, true) == std::make_pair(6953u, 9000u));
  CHECK(mp::visible(9000, false) == std::make_pair(0u, 9000u));
  for (uint32_t c0 : {2048u, 4096u, 6143u}) {
    const uint32_t C = mp::kC;
    const uint32_t lo = mp::visible(c0, true).first, hi = c0 + C - 1;
    CHECK_EQ(lo, c0 - 2047);
    CHECK_EQ(hi - lo + 1, 4095u);   // 4095 positions in 4096 slots
    std::set<uint32_t> slots;
    for (uint32_t k = lo; k <= hi; ++k) slots.insert(k & (mp::kRing - 1));
    CHECK_EQ(slots.size(), size_t(4095));   // pairwise distinct: no live key overwritten within the chunk
    // the first and last row of the chunk
    CHECK(mp::visible(c0, true) == std::make_pair(c0 - 2047, c0));
    CHECK(mp::visible(c0 + C - 1, true) == std::make_pair(c0, c0 + C - 1));
    for (uint32_t r0 = 0; r0 < C; r0 += 8) {   // every group's first tile: absolute, at most 63 keys below its lo
      const uint32_t ft = mp::first_tile(c0 + r0, true), glo = mp::visible(c0 + r0, true).first;
      CHECK(ft % 64 == 0 && ft <= glo && glo - ft < 64);
    }
  }
  CHECK_EQ(mp::first_tile(5000, false), 0u);
  std::printf("(a) visible sets and the ring: a chunk of 2048 at 2048 / 4096 / 6143 reads 4095 distinct slots\n");
}

void test_flash_direct() {   // (b)
  double worst = 0;
  for (uint32_t c0 : {0u, 1000u, 2048u, 4096u, 6143u}) {
    const uint32_t C = mp::kC;
    const Caches c = caches(c0 + C, true, 7);
    const std::vector<uint16_t> q = queries(c0, C, 9);
    for (uint32_t t : {0u, 1u, 7u, 8u, 63u, 64u, 1000u, 2047u})
      for (uint32_t h : {0u, 15u, 16u, 31u}) {
        const std::vector<double> f = mp::flash_row_fp64(q.data(), c.kc.data(), c.vc.data(), c0, C, t, true, h);
        const std::vector<double> d =
            mp::direct_fp64(q.data() + (size_t(t) * QH + h) * HD, c.kpos.data(), c.vpos.data(), c0 + t, true, h);
        worst = std::max(worst, max_abs(f, d));
      }
  }
  for (uint32_t depth : {3000u, 20000u}) {   // full: the growing cache, causal, no window
    const uint32_t C = 512, c0 = depth - C;
    const Caches c = caches(depth, false, 11);
    const std::vector<uint16_t> q = queries(c0, C, 13);
    for (uint32_t t : {0u, 7u, 8u, 511u})
      for (uint32_t h : {0u, 31u}) {
        const std::vector<double> f = mp::flash_row_fp64(q.data(), c.kc.data(), c.vc.data(), c0, C, t, false, h);
        const std::vector<double> d =
            mp::direct_fp64(q.data() + (size_t(t) * QH + h) * HD, c.kpos.data(), c.vpos.data(), c0 + t, false, h);
        worst = std::max(worst, max_abs(f, d));
      }
  }
  std::printf("(b) flash walk (fp64, 64-key absolute tiles, through the ring) vs the direct softmax: max |diff| %.3e\n",
              worst);
  CHECK(worst <= 1e-12);
}

void test_masked_first_tile() {   // (c)
  // Find a chunk start where the 8-row group at r0 = 8 has a row whose lo is >= first_tile + 64.
  uint32_t c0 = 0, t = 0;
  for (uint32_t c = 2048; c < 2048 + 64 && !t; ++c)
    for (uint32_t r = 9; r < 16 && !t; ++r)
      if (mp::visible(c + r, true).first >= mp::first_tile(c + 8, true) + 64) {
        c0 = c;
        t = r;
      }
  CHECK(t != 0);
  const uint32_t C = 64;
  const Caches c = caches(c0 + C, true, 21);
  const std::vector<uint16_t> q = queries(c0, C, 23);
  const uint32_t ft = mp::first_tile(c0 + 8, true);
  for (uint32_t h : {0u, 20u}) {
    const std::vector<double> a = mp::flash_row_fp64(q.data(), c.kc.data(), c.vc.data(), c0, C, t, true, h);
    const std::vector<double> b = mp::flash_row_fp64(q.data(), c.kc.data(), c.vc.data(), c0, C, t, true, h, ft + 64);
    CHECK(a == b);
  }
  // the unguarded form: exp(-INF - (-INF)) = NaN poisons the row
  const double m = -INFINITY;
  CHECK(std::isnan(std::exp(m - m)));
  std::printf("(c) row %u of the chunk at %u: its first tile [%u, %u) holds none of its keys - the walk from it == the "
              "walk from the next tile, bitwise (m_safe)\n", t, c0, ft, ft + 64);
}

void test_independence() {   // (d)
  const uint32_t C = 64, c0 = 4000;
  const Caches cc = caches(c0 + C, true, 31);   // the ring after positions [0, c0 + C)
  const std::vector<uint16_t> q = queries(c0, C, 33);
  std::vector<double> fwd(size_t(C) * QN), rev(fwd.size()), perm(fwd.size());
  mp::flash_window_fp64(q.data(), cc.kc.data(), cc.vc.data(), c0, C, true, fwd.data());
  for (uint32_t tt = C; tt-- > 0;)
    for (uint32_t h = 0; h < QH; ++h) {
      const std::vector<double> r = mp::flash_row_fp64(q.data(), cc.kc.data(), cc.vc.data(), c0, C, tt, true, h);
      std::copy(r.begin(), r.end(), rev.begin() + (size_t(tt) * QH + h) * HD);
    }
  std::mt19937 rng(5);
  for (uint32_t g = 0; g < C; g += 8) {
    std::vector<uint32_t> order(8);
    std::iota(order.begin(), order.end(), g);
    std::shuffle(order.begin(), order.end(), rng);
    for (uint32_t tt : order)
      for (uint32_t h = 0; h < QH; ++h) {
        const std::vector<double> r = mp::flash_row_fp64(q.data(), cc.kc.data(), cc.vc.data(), c0, C, tt, true, h);
        std::copy(r.begin(), r.end(), perm.begin() + (size_t(tt) * QH + h) * HD);
      }
  }
  CHECK(fwd == rev && fwd == perm);
  // splits: the row at absolute p = 4100 inside chunks starting at different c0 (the ring holds [0, chunk end))
  const uint32_t p = 4100;
  const Caches pc = caches(p + 1, true, 41);
  std::vector<uint16_t> qrow(QN);
  for (uint32_t i = 0; i < QN; ++i) qrow[i] = val(43, p, i, 4.0f);
  auto row_in = [&](uint32_t start) {   // the row walked as row p - start of a chunk at `start` ending at p
    const uint32_t C2 = p + 1 - start;
    std::vector<uint16_t> qq(size_t(C2) * QN, 0);
    std::copy(qrow.begin(), qrow.end(), qq.begin() + size_t(C2 - 1) * QN);
    std::vector<double> o(QN);
    for (uint32_t h = 0; h < QH; ++h) {
      const std::vector<double> r = mp::flash_row_fp64(qq.data(), pc.kc.data(), pc.vc.data(), start, C2, C2 - 1, true, h);
      std::copy(r.begin(), r.end(), o.begin() + size_t(h) * HD);
    }
    return o;
  };
  const std::vector<double> ref = row_in(4096 - 2048);
  for (uint32_t start : {4096u - 64, 4096u, 4096u - 2048 + 64}) CHECK(row_in(start) == ref);   // multiples of 64
  const std::vector<double> odd = row_in(2049);
  const double cs = cosine(odd.data(), ref.data(), QN);
  std::printf("(d) row independence: reversed and in-group permuted chunks bitwise; the row at 4100 in chunks from "
              "2048 / 4032 / 4096 / 2112 bitwise; from 2049 %s (cosine %.12f)\n",
              odd == ref ? "bitwise" : "NOT bitwise", cs);
  CHECK(cs >= 0.99999);
}

void test_flash_sim() {   // (e)
  double worst = 1;
  for (uint32_t c0 : {0u, 2048u, 6143u}) {
    const uint32_t C = 64;
    const Caches c = caches(c0 + C, true, 51);
    const std::vector<uint16_t> q = queries(c0, C, 53);
    std::vector<uint16_t> sim(size_t(C) * QN);
    std::vector<double> ref(sim.size());
    mp::flash_window_sim(q.data(), c.kc.data(), c.vc.data(), c0, C, true, sim.data());
    mp::flash_window_fp64(q.data(), c.kc.data(), c.vc.data(), c0, C, true, ref.data());
    for (size_t r = 0; r < size_t(C) * QH; ++r) {
      std::vector<double> s(HD);
      for (uint32_t d = 0; d < HD; ++d) s[d] = bf(sim[r * HD + d]);
      worst = std::min(worst, cosine(s.data(), ref.data() + r * HD, HD));
    }
  }
  {
    const uint32_t depth = 30000, C = 16, c0 = depth - C;
    const Caches c = caches(depth, false, 55);
    const std::vector<uint16_t> q = queries(c0, C, 57);
    std::vector<uint16_t> sim(size_t(C) * QN);
    std::vector<double> ref(sim.size());
    mp::flash_window_sim(q.data(), c.kc.data(), c.vc.data(), c0, C, false, sim.data());
    mp::flash_window_fp64(q.data(), c.kc.data(), c.vc.data(), c0, C, false, ref.data());
    for (size_t r = 0; r < size_t(C) * QH; ++r) {
      std::vector<double> s(HD);
      for (uint32_t d = 0; d < HD; ++d) s[d] = bf(sim[r * HD + d]);
      worst = std::min(worst, cosine(s.data(), ref.data() + r * HD, HD));
    }
  }
  std::printf("(e) flash in fp32 (bf16 P, natural exp) vs fp64: worst per-(row, head) cosine %.7f (bar 0.99999)\n", worst);
  CHECK(worst >= 0.99999);
}

void test_eager() {   // (f)
  size_t rows = 0, single_chain_diff = 0;
  for (int sliding = 1; sliding >= 0; --sliding) {
    const std::vector<uint32_t> pos = sliding ? std::vector<uint32_t>{5, 127, 128, 129, 2047, 2048, 4097, 6143}
                                              : std::vector<uint32_t>{3000, 9000};
    const Caches c = caches(*std::max_element(pos.begin(), pos.end()) + 1, sliding != 0, 61);
    for (uint32_t p : pos) {
      std::vector<uint16_t> q(QN);
      for (uint32_t i = 0; i < QN; ++i) q[i] = val(63, p, i, 4.0f);
      std::vector<uint16_t> got(QN), want(QN);
      mp::eager_window(q.data(), c.kc.data(), c.vc.data(), p, 1, sliding != 0, got.data());
      mr::attention_eager(q.data(), c.kc.data(), c.vc.data(), p, sliding != 0, mr::kRing, nullptr, want.data());
      CHECK(got == want);
      ++rows;
      // the wrong form: P·V as ONE fma chain (no blocks) differs somewhere once the row has more than one block
      const uint32_t lo = mp::visible(p, sliding != 0).first, L = p - lo + 1;
      if (L > mr::eager_block(L)) {
        for (uint32_t h = 0; h < QH; ++h) {
          const mr::EagerHead e = mr::attention_eager_head(q.data(), c.kc.data(), c.vc.data(), p, sliding != 0, h);
          for (uint32_t d = 0; d < HD; ++d) {
            float one = 0;
            for (uint32_t i = 0; i < L; ++i)
              one = std::fma(e.p[i], bf(c.vc[mp::key_at(lo + i, sliding != 0, h / mp::kGqa) + d]), one);
            single_chain_diff += rne(one) != got[size_t(h) * HD + d];
          }
        }
      }
    }
  }
  std::printf("(f) eager_window (128-key tiles, 8 lanes, eager_block P·V) == decode's eager row bitwise on %zu rows x 32 "
              "heads; one P·V chain would differ in %zu values\n", rows, single_chain_diff);
  CHECK(single_chain_diff > 0);
}

void test_gate() {   // (g)
  const uint32_t rows = 3;
  std::vector<uint16_t> a(size_t(rows) * QN), g(a.size()), want(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = val(71, 0, uint32_t(i), 2.0f);
    g[i] = mr::gate_bf16(val(73, 0, uint32_t(i), 6.0f));
    want[i] = rne(bf(a[i]) * bf(g[i]));
  }
  mp::gate_window(a.data(), g.data(), rows);
  CHECK(a == want);
  std::printf("(g) the gate: rne(rne(o) x bf16(sigmoid(g))) per element, decode's chain\n");
}

}  // namespace

int main() {
  test_visible();
  test_flash_direct();
  test_masked_first_tile();
  test_independence();
  test_flash_sim();
  test_eager();
  test_gate();
  std::puts("muse_pf_ref_test OK");
  return 0;
}
