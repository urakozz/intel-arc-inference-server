// Spec 24d Task 1: tests/kernels/lfm2moe_pf_ref.h (the LFM2.5-24B-A2B prefill kernels' host twin) against 24c's
// decode twin (tests/kernels/lfm2moe_ref.h) and torch's own conv over a chunked prompt (tests/kernels/
// lfm2moe_fixture.h case (h)). Host only.
//
//   (1) the conv's chunk boundary (plan 24d Review Focus 1): a 2100-position prompt at the real width through
//       conv_chunk under every schedule - one chunk of 2048 + the rest, chunks of 64, a split at every offset 1..5 and
//       at 64, 2047, 2048, 2049, chunks of 1 and of 2 - equals lfmref::conv_step one position at a time, bitwise (y
//       and the ring slots decode reads next); the first chunk at position 0 reads zeros from the seed although the
//       ring holds garbage; the conv-tile column order (the arm conv's in_proj) gives the same y and ring
//   (2) torch's conv over a prompt in chunks (fixture (h): chunk 0 from the zero history, chunk 1 seeded with the
//       state): y and the final state bitwise
//   (3) the sort at 64 experts, top-4 (Review Focus 3): every (token, slot) pair placed once, expert-major, ascending
//       token, padded to whole 32-row tiles, the NONE tail, the header; tmax(C) = floor((4C + 64 x 31) / 32) at
//       C 1, 31, 32, 33, 2048; the bound reached (63 experts at 129 rows, one at 65: 318 tiles) and one expert in
//       every token's slots (crafted) - neither overflows
//   (4) the combine: sort + combine equals lfmref::combine + the fold per token (random routes), and torch's
//       grouped_mm combine on the fixture's expert outputs (case (d)) bitwise; the gather's rows
//   (5) the flash walk in fp64 against lfmref::attention (fp64 softmax over [0, p]) within 1e-12 at chunks (0, 1),
//       (0, 63), (0, 64), (100, 37) and (4096, 70) with earlier keys in the cache (rows attend across the chunk
//       boundary); eager_row against fp64 (cosine >= 0.999)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/lfm2moe_fixture.h"
#include "kernels/lfm2moe_pf_ref.h"
#include "kernels/lfm2moe_ref.h"

namespace {

namespace fx = lfm2moe_fixture;
namespace lr = lfmref;
namespace lp = lfmpfref;
using lr::f32;
using lr::rf;
using lr::rne;

constexpr uint32_t H = lr::kHidden;

std::vector<float> rf32(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = d(g);
  return v;
}
std::vector<uint16_t> rbf16(size_t n, float lo, float hi, uint32_t seed) {
  const std::vector<float> f = rf32(n, lo, hi, seed);
  std::vector<uint16_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = rne(f[i]);
  return v;
}

// --- (1) the chunk boundary --------------------------------------------------------------------------------------
struct ConvRun {
  std::vector<uint16_t> y, ring;
};
// A schedule: the chunk sizes, in order (their sum is the prompt).
ConvRun run_chunks(const std::vector<float>& inproj, const std::vector<float>& taps, const std::vector<uint16_t>& garbage,
                   const std::vector<uint32_t>& sizes, bool tiled = false) {
  ConvRun r;
  r.ring = garbage;
  uint32_t p0 = 0;
  for (uint32_t C : sizes) {
    const std::vector<uint16_t> seed = lp::conv_seed(r.ring.data(), p0, H);
    const std::vector<uint16_t> y =
        lp::conv_chunk(inproj.data() + size_t(p0) * 3 * H, seed.data(), taps.data(), r.ring.data(), p0, C, H, tiled);
    r.y.insert(r.y.end(), y.begin(), y.end());
    p0 += C;
  }
  return r;
}
std::vector<uint32_t> uniform(uint32_t T, uint32_t c) {
  std::vector<uint32_t> s;
  for (uint32_t p = 0; p < T; p += c) s.push_back(std::min(c, T - p));
  return s;
}

void test_conv_chunks() {
  constexpr uint32_t T = 2100;
  const std::vector<float> inproj = rf32(size_t(T) * 3 * H, -3.0f, 3.0f, 11);   // fp32 GEMM rows (rne'd by the conv)
  std::vector<float> taps(size_t(H) * 3);
  {
    const std::vector<uint16_t> t = rbf16(taps.size(), -0.6f, 0.6f, 12);
    for (size_t i = 0; i < taps.size(); ++i) taps[i] = f32(t[i]);
  }
  std::vector<uint16_t> garbage = rbf16(size_t(lr::kRing) * H, -50.0f, 50.0f, 13);   // never read below position 0
  // the reference: decode's conv, one position a launch (M 1, S 1), from the same garbage ring
  std::vector<uint16_t> ring = garbage, yref;
  for (uint32_t p = 0; p < T; ++p) {
    const std::vector<uint16_t> y = lr::conv_step(inproj.data() + size_t(p) * 3 * H, 1, ring.data(), taps.data(), p, 1);
    yref.insert(yref.end(), y.begin(), y.end());
  }
  auto last_two = [&](const std::vector<uint16_t>& rg) {
    std::vector<uint16_t> v;
    for (uint32_t p : {T - 2, T - 1}) v.insert(v.end(), rg.begin() + size_t(p % lr::kRing) * H, rg.begin() + size_t(p % lr::kRing + 1) * H);
    return v;
  };
  const std::vector<uint16_t> want_ring = last_two(ring);
  std::vector<std::pair<std::string, std::vector<uint32_t>>> scheds = {
      {"2048 + 52", {2048, 52}}, {"chunks of 64", uniform(T, 64)}, {"chunks of 1", uniform(T, 1)},
      {"chunks of 2", uniform(T, 2)}};
  for (uint32_t s : {1u, 2u, 3u, 4u, 5u, 64u, 2047u, 2048u, 2049u}) {
    std::vector<uint32_t> v = {s};
    for (uint32_t c : uniform(T - s, 2048)) v.push_back(c);
    scheds.push_back({"split at " + std::to_string(s), v});
  }
  for (const auto& [name, sizes] : scheds) {
    const ConvRun r = run_chunks(inproj, taps, garbage, sizes);
    CHECK(r.y == yref);
    CHECK(last_two(r.ring) == want_ring);
    std::printf("(1) conv %-16s: y [%u][%u] and the two ring slots decode reads next bitwise conv_step's\n", name.c_str(),
                T, H);
  }
  {   // the first chunk at position 0 reads zeros, not the garbage ring
    const std::vector<uint16_t> seed = lp::conv_seed(garbage.data(), 0, H);
    CHECK(std::all_of(seed.begin(), seed.end(), [](uint16_t v) { return v == 0; }));
    const std::vector<uint16_t> s1 = lp::conv_seed(garbage.data(), 1, H);
    CHECK(std::all_of(s1.begin(), s1.begin() + H, [](uint16_t v) { return v == 0; }));
    CHECK(std::equal(s1.begin() + H, s1.end(), garbage.begin() + size_t(0) * H));   // slot 0 = position 0
  }
  {   // the conv-tile column order (in_proj loaded for the arm conv): the same y and ring
    std::vector<float> tiled(inproj.size());
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t c = 0; c < H; ++c)
        for (uint32_t part = 0; part < 3; ++part)
          tiled[size_t(t) * 3 * H + lp::conv_col(c, part, H, true)] = inproj[size_t(t) * 3 * H + lp::conv_col(c, part, H, false)];
    const ConvRun r = run_chunks(tiled, taps, garbage, {2048, 52}, true);
    CHECK(r.y == yref);
    CHECK(last_two(r.ring) == want_ring);
    std::printf("(1) conv in the conv-tile column order: bitwise the plain order's\n");
  }
}

// --- (2) torch's chunked conv --------------------------------------------------------------------------------------
void test_conv_fixture() {
  const uint32_t T = fx::kConvChunkT, S = fx::kConvChunkSplit, W = fx::kConvChunkW;
  std::vector<float> inproj(size_t(T) * 3 * W), taps(size_t(W) * 3);
  for (uint32_t p = 0; p < T; ++p)
    for (uint32_t c = 0; c < W; ++c)
      for (uint32_t part = 0; part < 3; ++part)
        inproj[size_t(p) * 3 * W + size_t(part) * W + c] = lr::fixture_val(false, 30 + part, p * W + c, 6, 0.125f, 2.0f);
  for (uint32_t i = 0; i < W * 3; ++i) taps[i] = lr::fixture_val(false, 33, i, 6, 0.125f, 0.5f);
  std::vector<uint16_t> ring(size_t(lr::kRing) * W, 0x7F7F);   // garbage: position < 0 never reads it
  std::vector<uint16_t> y;
  uint32_t p0 = 0;
  for (uint32_t C : {S, T - S}) {
    const std::vector<uint16_t> seed = lp::conv_seed(ring.data(), p0, W);
    const std::vector<uint16_t> yc = lp::conv_chunk(inproj.data() + size_t(p0) * 3 * W, seed.data(), taps.data(),
                                                    ring.data(), p0, C, W);
    y.insert(y.end(), yc.begin(), yc.end());
    p0 += C;
  }
  CHECK(y == lr::hex16(fx::kConvChunkY));
  const std::vector<uint16_t> st = lr::hex16(fx::kConvChunkState);
  for (uint32_t s = 0; s < 2; ++s)
    CHECK(std::equal(st.begin() + size_t(s) * W, st.begin() + size_t(s + 1) * W,
                     ring.begin() + size_t((T - 2 + s) % lr::kRing) * W));
  std::printf("(2) torch's conv over a prompt in chunks of %u + %u (fixture (h)): y [%u][%u] and the final state bitwise\n",
              S, T - S, T, W);
}

// --- (3) the sort --------------------------------------------------------------------------------------------------
std::vector<uint32_t> routes_from(const std::vector<std::array<uint32_t, 4>>& ids) {
  std::vector<uint32_t> r(ids.size() * lr::kWords, 0);
  for (size_t t = 0; t < ids.size(); ++t)
    for (uint32_t j = 0; j < 4; ++j) r[t * lr::kWords + lr::kIds + j] = ids[t][j];
  return r;
}
std::vector<std::array<uint32_t, 4>> random_ids(uint32_t C, uint32_t seed) {
  std::mt19937 g(seed);
  std::vector<std::array<uint32_t, 4>> v(C);
  std::vector<uint32_t> e(64);
  std::iota(e.begin(), e.end(), 0u);
  for (auto& a : v) {
    std::shuffle(e.begin(), e.end(), g);
    for (uint32_t j = 0; j < 4; ++j) a[j] = e[j];
  }
  return v;
}
// Token t's slot j is element j x C + t of the expert-major list with the given counts (distinct per token while
// every count <= C).
std::vector<std::array<uint32_t, 4>> ids_with_counts(const std::vector<uint32_t>& counts, uint32_t C) {
  std::vector<uint32_t> list;
  for (uint32_t e = 0; e < counts.size(); ++e) list.insert(list.end(), counts[e], e);
  CHECK_EQ(list.size(), size_t(4) * C);
  std::vector<std::array<uint32_t, 4>> v(C);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < 4; ++j) v[t][j] = list[size_t(j) * C + t];
  return v;
}
uint32_t check_sorted(const std::vector<std::array<uint32_t, 4>>& ids, const char* what) {
  const uint32_t C = uint32_t(ids.size());
  for (const auto& a : ids) CHECK_EQ(std::set<uint32_t>(a.begin(), a.end()).size(), size_t(4));
  const std::vector<uint32_t> route = routes_from(ids);
  const lp::Sorted s = lp::sort(route.data(), C);
  const uint32_t T = lp::tmax(C);
  CHECK(s.tiles_used <= T);
  CHECK_EQ(s.rows_used(), s.tiles_used * lp::kTm);
  CHECK_EQ(s.hdr[lp::kHdrC], C);
  CHECK_EQ(s.hdr[1], 0u);
  // every pair placed exactly once, in its expert's tile
  std::vector<int> seen(s.rows_used(), 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t r = s.pair_row[size_t(t) * 4 + j];
      CHECK(r < s.rows_used());
      CHECK_EQ(s.row_tok[r], t);
      CHECK_EQ(s.tiles[2 * (r / lp::kTm)], ids[t][j]);
      CHECK_EQ(s.tiles[2 * (r / lp::kTm) + 1], r / lp::kTm * lp::kTm);
      ++seen[r];
    }
  uint32_t pad = 0, prev_e = 0;
  std::vector<int> last_tok(lp::kE, -1);
  std::vector<char> padded(lp::kE, 0);
  for (uint32_t tile = 0; tile < s.tiles_used; ++tile) {
    const uint32_t e = s.tiles[2 * tile];
    CHECK(e < lp::kE);
    CHECK(e >= prev_e);   // expert-major
    prev_e = e;
    for (uint32_t r = tile * lp::kTm; r < (tile + 1) * lp::kTm; ++r) {
      if (s.row_tok[r] == lp::kNone) {   // padding only after the expert's last pair
        CHECK_EQ(seen[r], 0);
        padded[e] = 1;
        ++pad;
        continue;
      }
      CHECK_EQ(seen[r], 1);
      CHECK(!padded[e]);
      CHECK(int(s.row_tok[r]) > last_tok[e]);   // ascending token (a token names an expert once)
      last_tok[e] = int(s.row_tok[r]);
    }
  }
  for (uint32_t tile = s.tiles_used; tile < T; ++tile) CHECK_EQ(s.tiles[2 * tile], lp::kNone);
  uint32_t total = 0;
  for (uint32_t e = 0; e < lp::kE; ++e) total += s.hdr[lp::kHdrCount + e];
  CHECK_EQ(total, 4 * C);
  CHECK_EQ(pad + 4 * C, s.rows_used());
  std::printf("(3) sort %-38s C %4u: %3u / %3u tiles, %5u padding rows - every pair once, expert-major, ascending token\n",
              what, C, s.tiles_used, T, pad);
  return s.tiles_used;
}
void test_sort() {
  CHECK_EQ(lp::tmax(1), 62u);
  CHECK_EQ(lp::tmax(31), 65u);
  CHECK_EQ(lp::tmax(32), 66u);
  CHECK_EQ(lp::tmax(33), 66u);
  CHECK_EQ(lp::tmax(2048), 318u);
  for (uint32_t C : {1u, 2u, 31u, 32u, 33u, 64u, 2047u, 2048u}) check_sorted(random_ids(C, 100 + C), "random routes");
  {   // the bound reached: 63 experts at 129 rows (5 tiles each), one at 65 (3 tiles)
    std::vector<uint32_t> counts(64, 129);
    counts[63] = 8192 - 63 * 129;
    CHECK_EQ(check_sorted(ids_with_counts(counts, 2048), "the bound (63 x 129 + 65)"), lp::tmax(2048));
  }
  {   // one expert in every token's slots (a crafted router), the rest spread
    std::vector<std::array<uint32_t, 4>> ids = random_ids(2048, 7);
    for (auto& a : ids) {
      if (std::find(a.begin(), a.end(), 9u) == a.end()) a[0] = 9;
      std::swap(*std::find(a.begin(), a.end(), 9u), a[0]);
    }
    check_sorted(ids, "expert 9 in every token");
    std::vector<uint32_t> counts(64, 0);   // every pair into four experts
    for (uint32_t e = 0; e < 4; ++e) counts[e] = 2048;
    CHECK_EQ(check_sorted(ids_with_counts(counts, 2048), "four experts take every row"), 256u);
  }
  {   // C = 1: one token, four experts, four tiles of 1 row + 31 padding
    CHECK_EQ(check_sorted({{3, 0, 63, 17}}, "one token"), 4u);
  }
}

// --- (4) the combine ---------------------------------------------------------------------------------------------
void test_combine() {
  constexpr uint32_t C = 96;
  std::mt19937 g(5);
  std::uniform_real_distribution<float> dl(-4.f, 4.f), db(-0.05f, 0.05f);
  std::vector<float> bias(64);
  for (float& b : bias) b = db(g);
  std::vector<uint32_t> route(size_t(C) * lr::kWords);
  for (uint32_t t = 0; t < C; ++t) {
    std::vector<float> lg(64);
    for (float& v : lg) v = dl(g);
    const lr::Route r = lr::route(lg.data(), bias.data());
    std::copy(r.row, r.row + lr::kWords, route.begin() + size_t(t) * lr::kWords);
  }
  const lp::Sorted s = lp::sort(route.data(), C);
  // the down GEMM's sorted rows: fp32 sums d per (row, n), y = rne(d)
  const std::vector<float> d = rf32(size_t(s.rows_used()) * H, -2.f, 2.f, 6);
  std::vector<uint16_t> y(d.size());
  for (size_t i = 0; i < d.size(); ++i) y[i] = rne(d[i]);
  const std::vector<uint16_t> h0 = rbf16(size_t(C) * H, -3.f, 3.f, 7);
  std::vector<uint16_t> h = h0;
  lp::combine(route.data(), s, y.data(), h.data(), C);
  for (uint32_t t = 0; t < C; ++t) {
    float w[4];
    for (uint32_t j = 0; j < 4; ++j) w[j] = lr::as_f32(route[size_t(t) * lr::kWords + lr::kW + j]);
    for (uint32_t n = 0; n < H; ++n) {
      float dd[4];
      for (uint32_t j = 0; j < 4; ++j) dd[j] = d[size_t(s.pair_row[size_t(t) * 4 + j]) * H + n];
      const uint16_t want = rne(f32(h0[size_t(t) * H + n]) + f32(lr::combine(dd, w)));
      CHECK_EQ(h[size_t(t) * H + n], want);
    }
  }
  std::printf("(4) combine over %u sorted tokens: bitwise lfmref::combine (decode's rank-order chain) + the fold\n", C);
  {   // torch's grouped_mm combine on the fixture's expert outputs (case (d)): ids rank order, weights, proj rows
    const std::vector<uint16_t> proj = lr::hex16(fx::kMoeProj), want_y = lr::hex16(fx::kMoeY);
    const std::vector<uint32_t> wbits = lr::hex32(fx::kMoeW);
    std::vector<std::array<uint32_t, 4>> ids(4);
    for (uint32_t t = 0; t < 4; ++t)
      for (uint32_t j = 0; j < 4; ++j) ids[t][j] = fx::kMoeIds[t][j];
    std::vector<uint32_t> rt = routes_from(ids);
    for (uint32_t t = 0; t < 4; ++t)
      for (uint32_t j = 0; j < 4; ++j) rt[size_t(t) * lr::kWords + lr::kW + j] = wbits[t * 4 + j];
    const lp::Sorted st = lp::sort(rt.data(), 4);
    std::vector<uint16_t> yy(size_t(st.rows_used()) * H, 0);
    for (uint32_t t = 0; t < 4; ++t)
      for (uint32_t j = 0; j < 4; ++j)
        std::copy(proj.begin() + (size_t(t) * 4 + j) * H, proj.begin() + (size_t(t) * 4 + j + 1) * H,
                  yy.begin() + size_t(st.pair_row[t * 4 + j]) * H);
    const std::vector<uint16_t> hb = rbf16(size_t(4) * H, -1.f, 1.f, 8);
    std::vector<uint16_t> hh = hb;
    lp::combine(rt.data(), st, yy.data(), hh.data(), 4);
    for (size_t i = 0; i < hh.size(); ++i) CHECK_EQ(hh[i], rne(f32(hb[i]) + f32(want_y[i])));
    std::printf("(4) combine on torch's expert outputs (fixture (d)): bitwise grouped_mm's y + the fold\n");
  }
  {   // the gather: every row's token, zeros for padding
    const std::vector<uint16_t> x = rbf16(size_t(C) * H, -1.f, 1.f, 9);
    const std::vector<uint16_t> xg = lp::gather(s, x.data());
    for (uint32_t r = 0; r < s.rows_used(); ++r) {
      const uint32_t t = s.row_tok[r];
      for (uint32_t k = 0; k < H; k += 97)
        CHECK_EQ(xg[size_t(r) * H + k], t == lp::kNone ? uint16_t(0) : x[size_t(t) * H + k]);
    }
  }
}

// --- (5) the flash walk ------------------------------------------------------------------------------------------
void test_flash() {
  constexpr uint32_t L = 4166;
  const std::vector<uint16_t> kc = rbf16(size_t(L) * lr::kKvHeads * lr::kHd, -1.f, 1.f, 21);
  const std::vector<uint16_t> vc = rbf16(size_t(L) * lr::kKvHeads * lr::kHd, -1.f, 1.f, 22);
  double worst = 0, ecos = 1;
  for (const auto& [c0, C] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {0, 63}, {0, 64}, {100, 37}, {4096, 70}}) {
    std::vector<float> q(size_t(C) * lr::kQHeads * lr::kHd);
    {
      const std::vector<uint16_t> qb = rbf16(q.size(), -2.f, 2.f, 30 + c0 + C);
      for (size_t i = 0; i < q.size(); ++i) q[i] = f32(qb[i]);
    }
    for (uint32_t t = 0; t < C; t += (C > 40 ? 7 : 1))
      for (uint32_t h : {0u, 5u, 31u}) {
        const std::vector<double> f = lp::flash_row(q.data(), kc.data(), vc.data(), c0, C, t, h);
        const std::vector<double> want =
            lr::attention(q.data() + (size_t(t) * lr::kQHeads + h) * lr::kHd, kc.data(), vc.data(), c0 + t + 1, h / 4);
        for (uint32_t dd = 0; dd < lr::kHd; ++dd) worst = std::max(worst, std::fabs(f[dd] - want[dd]));
        const std::vector<uint16_t> e = lp::eager_row(q.data() + (size_t(t) * lr::kQHeads + h) * lr::kHd, kc.data(),
                                                      vc.data(), c0 + t, h);
        double ab = 0, aa = 0, bb = 0;
        for (uint32_t dd = 0; dd < lr::kHd; ++dd) {
          ab += want[dd] * f32(e[dd]);
          aa += want[dd] * want[dd];
          bb += double(f32(e[dd])) * f32(e[dd]);
        }
        ecos = std::min(ecos, (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb));
      }
  }
  CHECK(worst < 1e-12);
  CHECK(ecos >= 0.999);
  std::printf("(5) flash walk (64-key tiles from key 0, causal across the chunk boundary) vs fp64 softmax: worst |d| "
              "%.2e; eager_row vs fp64: worst cosine %.6f\n", worst, ecos);
}

}  // namespace

int main() {
  test_conv_chunks();
  test_conv_fixture();
  test_sort();
  test_combine();
  test_flash();
  std::puts("lfm2moe_pf_ref_test OK");
  return 0;
}
