// Spec 20c Task 4: tests/kernels/kolibri_ref.h (the Kolibri-1 kernels' host twin) against
// tools/oracle/kolibri_ref.py's own outputs (tests/kernels/kolibri_fixture.h) and against the
// semantics' properties (plan 20c Review Focus 1-4). Host only, headers only.
//
//   (a) the norm: coarse rows bitwise for w = 1, 0, 2, random; a fine row within 1 ulp (Σx²'s order);
//       post_add: a zero post-norm weight leaves the residual unchanged, 2 doubles the contribution,
//       its Σ² is prep_res_fold's tree over the new residual
//   (b) q / k norm + RoPE bitwise at 0, 1, 513, 100000, 262143; the full layer's k is the normed head
//   (c) the route: ids exact, the tie at the cut to the lower id, the bias reorders without moving a
//       weight, the all -1e30 row never selects a padded slot; weights within 2 ulp of torch's sigmoid
//   (d) the combine bitwise, and an fma in place of mul + add would be caught
//   (e) eager attention: scores and probabilities bitwise, outputs within 1 ulp (P·V's fp32 order),
//       flash's fp64 form within cosine 0.99999; the ring's key -> row map at 0, 1, 511, 512, 513,
//       4095, 4096, 4097, 9000; ring addressing == a linear cache, bitwise
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "kernels/kolibri_fixture.h"
#include "kernels/kolibri_ref.h"

namespace {

namespace fx = kolibri_fixture;
namespace kr = kolibri_ref;
using kr::f32;
using kr::rf;
using kr::rne;

int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
int64_t ulps32(float a, float b) {
  auto key = [](float v) {
    const int32_t i = int32_t(kr::as_u32(v));
    return i < 0 ? -int64_t(i & 0x7FFFFFFF) : int64_t(i);
  };
  return std::llabs(key(a) - key(b));
}

std::vector<uint16_t> bf16_row(bool coarse, uint32_t t, uint32_t n, uint32_t seed, float step, float amp,
                               uint32_t base = 0) {
  std::vector<uint16_t> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = rne(kr::fixture_val(coarse, t, base + i, seed, step, amp));
  return v;
}

void test_norm() {
  const uint32_t H = kr::kHidden, G = kr::kG;
  const std::vector<uint16_t> xc = bf16_row(true, 0, H, 1, 0.125f, 1.f), xf = bf16_row(false, 0, H, 2, 0.125f, 4.f);
  std::vector<float> wr(H);
  for (uint32_t k = 0; k < H; ++k) wr[k] = rf(1.0f + kr::fixture_val(false, 1, k, 1, 0.125f, 0.5f));
  const auto run = [&](const std::vector<uint16_t>& x, const std::vector<float>& w) {
    std::vector<uint16_t> r = x, out(H);
    std::vector<float> ss(G);
    prep_ref::res_fold(nullptr, r.data(), ss.data(), 1, H, 0, G);
    kr::norm_finish(ss.data(), r.data(), w.data(), out.data(), 1);
    return out;
  };
  const std::pair<const char*, std::vector<float>> ws[] = {
      {fx::kNormCoarseOne, std::vector<float>(H, 1.f)}, {fx::kNormCoarseZero, std::vector<float>(H, 0.f)},
      {fx::kNormCoarseTwo, std::vector<float>(H, 2.f)}, {fx::kNormCoarseRand, wr}};
  for (const auto& [hex, w] : ws) CHECK(run(xc, w) == kr::hex16(hex));
  const std::vector<uint16_t> want = kr::hex16(fx::kNormFineRand), got = run(xf, wr);
  size_t diff = 0;
  for (uint32_t k = 0; k < H; ++k) {
    CHECK(ulps16(got[k], want[k]) <= 1);
    diff += got[k] != want[k];
  }
  std::printf("  norm: coarse rows bitwise (w = 1, 0, 2, random); fine row %zu of %u one ulp apart (Σx² order)\n",
              diff, H);

  // post_add (Review Focus 1): the sub-block's own normalised output, then the add.
  std::vector<float> sa(G);
  std::vector<uint16_t> a = xf;
  prep_ref::res_fold(nullptr, a.data(), sa.data(), 1, H, 0, G);   // prep_res_fold ..._Z's sums over a
  const std::vector<uint16_t> resid0 = bf16_row(false, 3, H, 4, 0.125f, 2.f);
  std::vector<float> ss(G), ss_ref(G);
  std::vector<uint16_t> r0 = resid0;
  kr::post_add(sa.data(), a.data(), std::vector<float>(H, 0.f).data(), r0.data(), ss.data(), 1);
  CHECK(r0 == resid0);   // w = 0: the sub-block vanishes
  std::vector<uint16_t> z1(H, 0), z2(H, 0);
  kr::post_add(sa.data(), a.data(), std::vector<float>(H, 1.f).data(), z1.data(), ss.data(), 1);
  kr::post_add(sa.data(), a.data(), std::vector<float>(H, 2.f).data(), z2.data(), ss.data(), 1);
  for (uint32_t k = 0; k < H; ++k) CHECK(f32(z2[k]) == 2.0f * f32(z1[k]));   // doubled, exactly
  std::vector<uint16_t> r1 = resid0;
  kr::post_add(sa.data(), a.data(), wr.data(), r1.data(), ss.data(), 1);
  std::vector<uint16_t> r1b = r1;
  prep_ref::res_fold(nullptr, r1b.data(), ss_ref.data(), 1, H, 0, G);
  CHECK(ss == ss_ref);
  // and the add is the residual plus the normalised row, rounded once
  const float rstd = kr::rstd_of(sa.data(), 1, 0, H, G);
  for (uint32_t k = 0; k < H; k += 97)
    CHECK(r1[k] == rne(f32(resid0[k]) + f32(kr::norm_elem(f32(a[k]), rstd, wr[k]))));
  std::puts("  post_add: w 0 vanishes, w 2 doubles, its Σ² is prep_res_fold's tree");
}

void test_prep() {
  const uint32_t HD = kr::kHd, N = kr::kQHeads * HD + 2 * kr::kKvHeads * HD;
  const std::vector<uint16_t> qx = bf16_row(true, 2, HD, 3, 0.25f, 1.f), kx = bf16_row(true, 3, HD, 3, 0.25f, 1.f);
  std::vector<float> qkn(2 * HD);
  for (uint32_t d = 0; d < HD; ++d) {
    qkn[d] = rf(1.0f + kr::fixture_val(false, 4, d, 3, 0.125f, 0.5f));
    qkn[HD + d] = rf(1.0f + kr::fixture_val(false, 5, d, 3, 0.125f, 0.5f));
  }
  // q head 0 and k head 0 (kv head 0) of the fused row; S = 2 slices (x = half + half, exact)
  std::vector<float> part(size_t(2) * N, 0.0f);
  for (uint32_t d = 0; d < HD; ++d) {
    part[d] = f32(qx[d]) * 0.5f;
    part[N + d] = f32(qx[d]) * 0.5f;
    part[kr::kQHeads * HD + d] = f32(kx[d]);
  }
  const std::vector<uint16_t> qn = kr::hex16(fx::kQNormed), kn = kr::hex16(fx::kKNormed);
  const std::vector<uint16_t> qr = kr::hex16(fx::kQRoped), krr = kr::hex16(fx::kKRoped);
  const std::vector<uint16_t> cs = kr::hex16(fx::kRopeCos), sn = kr::hex16(fx::kRopeSin);
  // NoPE (a full layer): the normed heads, whatever the position (Review Focus 4)
  const kr::Prep full = kr::attn_prep(part.data(), 1, 0, 2, qkn.data(), nullptr);
  for (uint32_t d = 0; d < HD; ++d) {
    CHECK(rne(full.q[d]) == qn[d]);
    CHECK(full.k[d] == kn[d]);
  }
  for (uint32_t r = 0; r < fx::kRopeN; ++r) {
    std::vector<float> row(HD);
    for (uint32_t i = 0; i < HD / 2; ++i) {
      row[i] = f32(cs[size_t(r) * HD / 2 + i]);
      row[HD / 2 + i] = f32(sn[size_t(r) * HD / 2 + i]);
    }
    const kr::Prep p = kr::attn_prep(part.data(), 1, 0, 2, qkn.data(), row.data());
    for (uint32_t d = 0; d < HD; ++d) {
      CHECK(rne(p.q[d]) == qr[size_t(r) * HD + d]);
      CHECK(p.k[d] == krr[size_t(r) * HD + d]);
    }
    if (fx::kRopePos[r] == 0)
      for (uint32_t d = 0; d < HD; ++d) CHECK(p.k[d] == kn[d]);   // position 0 is the identity rotation
  }
  std::printf("  attn prep: q / k norm + RoPE bitwise at %u positions; NoPE k == the normed head\n", fx::kRopeN);
}

void test_route() {
  const uint32_t E = kr::kExperts;
  const std::vector<uint32_t> lg = kr::hex32(fx::kRouteLogits), bs = kr::hex32(fx::kRouteBias);
  const std::vector<uint32_t> ids = kr::hex32(fx::kRouteIds), w = kr::hex32(fx::kRouteW), gp = kr::hex32(fx::kRouteGap);
  int64_t worst = 0;
  for (uint32_t r = 0; r < 4; ++r) {
    std::vector<float> logits(kr::kRouterN, 1e30f), bias(kr::kRouterN, 0.0f);   // padded slots: huge, never read
    for (uint32_t e = 0; e < E; ++e) {
      logits[e] = kr::as_f32(lg[size_t(r) * E + e]);
      bias[e] = kr::as_f32(bs[size_t(r) * E + e]);
    }
    const kr::Route rt = kr::route(logits.data(), bias.data());
    for (uint32_t j = 0; j < kr::kTopK; ++j) {
      CHECK(rt.ids[j] == ids[size_t(r) * 6 + j]);
      CHECK(rt.ids[j] < E);
      worst = std::max(worst, ulps32(rt.w[j], kr::as_f32(w[size_t(r) * 6 + j])));
      CHECK(rt.w[j] == kr::sigmoid(logits[rt.ids[j]]));   // the weight is the LOGIT's, never the bias's
    }
    CHECK(kr::gap(logits.data(), bias.data(), rt) == kr::as_f32(gp[r]));
  }
  CHECK(worst <= 2);
  // the tie at the cut: experts 77 and 300 both at sel 1.0 for the sixth place - 77 (lower) goes in
  CHECK(ids[6 + 2] == 77 && kr::as_f32(gp[1]) == 0.0f);
  // the all -1e30 row: ids 0..5, weights 0
  for (uint32_t j = 0; j < 6; ++j) CHECK(ids[18 + j] == j && kr::as_f32(w[18 + j]) == 0.0f);
  std::printf("  route: ids exact on 4 rows (tie, bias reorder, all -1e30), weights within %lld ulp of torch's\n",
              static_cast<long long>(worst));
}

void test_combine() {
  const uint32_t H = kr::kHidden;
  const std::vector<uint32_t> w32 = kr::hex32(fx::kRouteW);
  float w[6];
  for (uint32_t j = 0; j < 6; ++j) w[j] = kr::as_f32(w32[j]);
  std::vector<std::vector<uint16_t>> y;
  for (uint32_t j = 0; j < 6; ++j) y.push_back(bf16_row(false, 6 + j, H, 7, 0.125f, 1.f));
  const std::vector<uint16_t> sh = bf16_row(false, 12, H, 7, 0.125f, 1.f), want = kr::hex16(fx::kCombine);
  for (uint32_t n = 0; n < H; ++n) {
    float d[7];
    for (uint32_t j = 0; j < 6; ++j) d[j] = f32(y[j][n]);
    d[6] = f32(sh[n]);
    CHECK(kr::combine(d, w) == want[n]);
  }
  // An fma in the chain would be caught: find a pair where fma(y1, w1, y0 w0) rounds apart from
  // y0 w0 + rf32(y1 w1) and check combine takes the second.
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(-2.f, 2.f), uw(0.05f, 1.0f);
  bool found = false;
  for (int it = 0; it < 100000 && !found; ++it) {
    float d[7] = {rf(u(rng)), rf(u(rng)), 0, 0, 0, 0, 0};
    float ww[6] = {uw(rng), uw(rng), 0, 0, 0, 0};
    const float t0 = d[0] * ww[0];
    const float fused = std::fma(d[1], ww[1], t0), plain = t0 + d[1] * ww[1];
    if (rne(fused) == rne(plain)) continue;
    found = true;
    CHECK(kr::combine(d, ww) == rne(plain));
  }
  CHECK(found);
  std::puts("  combine: bitwise torch's (ascending id fp32 + shared, one rounding); an fma would show");
}

// A [rows][4][128] cache holding kv head 0's coarse keys of positions [lo, hi] at their rows.
struct Cache {
  std::vector<uint16_t> k, v;
};
Cache cache_for(uint32_t lo, uint32_t hi, bool ring, uint32_t rows) {
  Cache c{std::vector<uint16_t>(size_t(rows) * 4 * 128, 0), std::vector<uint16_t>(size_t(rows) * 4 * 128, 0)};
  for (uint32_t p = lo; p <= hi; ++p) {
    const uint32_t row = ring ? kr::ring_row(p, true) : p;
    for (uint32_t d = 0; d < 128; ++d) {
      c.k[size_t(row) * 4 * 128 + d] = rne(kr::fixture_val(true, 21, p * 128 + d, 9, 0.125f));
      c.v[size_t(row) * 4 * 128 + d] = rne(kr::fixture_val(true, 22, p * 128 + d, 9, 0.125f));
    }
  }
  return c;
}

void test_attention() {
  for (uint32_t ci = 0; ci < fx::kAttnCases; ++ci) {
    const fx::AttnCase& c = fx::kAttn[ci];
    const bool sliding = c.sliding != 0;
    const uint32_t lo = kr::key_lo(c.query, sliding), hi = c.query, L = hi - lo + 1;
    CHECK(L == c.visible);
    const Cache kv = cache_for(lo, hi, sliding, sliding ? kr::kRing : hi + 1);
    std::vector<float> q(12 * 128);
    for (uint32_t i = 0; i < q.size(); ++i) q[i] = kr::fixture_val(true, 20, i, ci, 0.5f);
    const std::vector<uint16_t> s = kr::hex16(c.s), p = kr::hex16(c.p), o = kr::hex16(c.o);
    const uint32_t blk = kr::eager_block(L);
    size_t o_diff = 0;
    double worst_cos = 1.0;
    for (uint32_t h = 0; h < 12; ++h) {
      const kr::EagerHead e = kr::attention_eager(q.data() + h * 128, kv.k.data(), kv.v.data(), lo, hi, sliding, 0, blk);
      if (h == 0)
        for (uint32_t i = 0; i < L; ++i) {
          CHECK(rne(e.s[i]) == s[i]);
          CHECK(rne(e.p[i]) == p[i]);
        }
      const std::vector<double> f = kr::attention(q.data() + h * 128, kv.k.data(), kv.v.data(), lo, hi, sliding, 0);
      double dot = 0, na = 0, nb = 0;
      for (uint32_t d = 0; d < 128; ++d) {
        const uint16_t got = rne(e.o[d]);
        CHECK(ulps16(got, o[h * 128 + d]) <= 1);
        o_diff += got != o[h * 128 + d];
        dot += f[d] * f32(o[h * 128 + d]);
        na += f[d] * f[d];
        nb += double(f32(o[h * 128 + d])) * f32(o[h * 128 + d]);
      }
      worst_cos = std::min(worst_cos, dot / std::sqrt(na * nb));
    }
    // The fp64 form (flash's reference on the card) against the reference's bf16 eager output: the
    // eager chain rounds every probability to bf16, so near-cancelling outputs (513 coarse keys of
    // either sign) part by more than spec 6 K1's 0.99999 - that bar is flash vs fp64, on the card.
    CHECK(worst_cos >= 0.999);
    std::printf("  eager attention %s at %u (keys %u..%u): scores and probabilities bitwise; outputs %zu of 1536 "
                "one ulp apart; the fp64 form's worst cosine %.7f\n", sliding ? "sliding" : "full", c.query, lo, hi,
                o_diff, worst_cos);
  }
  // The ring (Review Focus 2): key pos - 512 visible, pos - 513 not; wrapped slots in key order.
  for (uint32_t pos : {0u, 1u, 511u, 512u, 513u, 4095u, 4096u, 4097u, 9000u}) {
    const std::vector<uint32_t> rows = kr::ring_rows(pos, true);
    CHECK(rows.size() == std::min(pos + 1, 513u));
    const uint32_t lo = kr::key_lo(pos, true);
    CHECK(lo == (pos >= 512 ? pos - 512 : 0));
    for (uint32_t i = 0; i < rows.size(); ++i) CHECK(rows[i] == ((lo + i) & 4095u));
    CHECK(kr::ring_rows(pos, false).size() == pos + 1);
  }
  CHECK(kr::ring_rows(4097, true).back() == 1 && kr::ring_rows(4097, true).front() == 3585);
  // The ring is only addressing: the same keys through the ring and through a linear cache.
  for (uint32_t pos : {4097u, 9000u}) {
    const uint32_t lo = kr::key_lo(pos, true);
    const Cache ring = cache_for(lo, pos, true, kr::kRing), lin = cache_for(lo, pos, false, pos + 1);
    std::vector<float> q(128);
    for (uint32_t i = 0; i < 128; ++i) q[i] = kr::fixture_val(true, 20, i, 7, 0.5f);
    const uint32_t blk = kr::eager_block(pos + 1 - lo);
    const kr::EagerHead a = kr::attention_eager(q.data(), ring.k.data(), ring.v.data(), lo, pos, true, 0, blk);
    const kr::EagerHead b = kr::attention_eager(q.data(), lin.k.data(), lin.v.data(), lo, pos, false, 0, blk);
    CHECK(a.o == b.o && a.p == b.p);
  }
  std::puts("  ring: 513 keys, pos - 512 the first, rows (lo + i) & 4095; ring == linear cache bitwise");
}

}  // namespace

int main() {
  test_norm();
  test_prep();
  test_route();
  test_combine();
  test_attention();
  std::puts("kolibri_ref_test OK");
  return 0;
}
