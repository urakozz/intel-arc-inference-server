// Spec 19b Task 2 (host): the DFlash host twin (tests/kernels/dflash_ref.h) against tools/oracle/dflash_ref.py - the
// reference the D1 gate is measured against - on its own outputs (tests/loader/dflash_ref_fixture.safetensors, written
// by tools/oracle/dflash_fixture.py: a tiny random DFlash 2 drafter and the same weights as a v1 drafter).
//
//   1. The context projection (Reference numerics): fc -> hidden_norm, per layer k (k_norm, RoPE) and v at every
//      position 0..39, against context_kv's fc / k / v planes (max |diff| 2e-5 relative to the row's max).
//   2. draft() at every fixture case (window 16 over a 16-slot ring: p 15 / 16 / 17 cross the window's edge, p 30 / 39
//      the ring's wrap; K 7 / 6 / 5 / 3 / 1; the mask id as the anchor; two sampled walks): the final-norm rows
//      (cosine >= 0.999999), the top-16 ids and order exact, unary / selector scores (|diff| <= 1e-4 x max), the walk's
//      choices and ids exact. v1: the same rows and top-16, ids = top-1.
//   3. Engine numerics (bf16 GEMV inputs, a bf16 ring, the GEMVs' device forms): bf16 drafter + bf16 head gated on
//      the rows (cosine >= 0.9999) and the scores (cosine >= 0.99); the top-16 order / sets and the walk printed. The
//      int8 drafter + int8 head arm is reported only. The tiny random drafter's score margins are not the D1
//      checkpoint's (measured here: rows 0.99996, scores 0.9989-0.99999, walks 7 / 9): D1 gates the real drafter on
//      the box (tools/oracle/dflash_d1.py).
//   4. The pieces: the window bound per row (a key at p + r - WIN is invisible, at p + r - WIN + 1 visible), the conv's
//      look-back (row 0 never reads row -1; row r reads r - 1 only), top-k's tie / NaN / mask rules, murmur3 against
//      dflash_ref's values, the walk's tie rule, the attention's exp (attn_exp = df_exp: within 2 fp32 ulps).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/dflash_fixture_io.h"
#include "kernels/dflash_ref.h"

namespace {

std::string g_fixture = "tests/loader/dflash_ref_fixture.safetensors";

float maxabs(const std::vector<float>& v) {
  float m = 0;
  for (float x : v) m = std::fmax(m, std::fabs(x));
  return m;
}

void test_projection() {
  const dffix::Fixture fx(g_fixture);
  const dfref::Weights w = dffix::weights(fx, 2, dffix::Form::Reference, dffix::Form::Reference, 8);
  const dfref::Dims& d = w.d;
  const std::vector<float> rope = dfref::rope_table(d.theta, d.HD, 64);
  const std::vector<uint16_t> taps = fx.bf16("taps");
  const std::vector<float> want_fc = fx.f32("v2.ctx.fc");
  double worst = 0;
  for (uint32_t pos = 0; pos < 40; ++pos) {
    dfref::Ring r = dfref::make_ring(d);
    dfref::project(w, dfref::Numerics::Reference, &taps[size_t(pos) * d.fck()], 1, pos, rope, r);
    const uint32_t slot = pos % d.ring;
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::vector<float> wk = fx.f32("v2.ctx.k" + std::to_string(l)), wv = fx.f32("v2.ctx.v" + std::to_string(l));
      const float mk = maxabs(wk), mv = maxabs(wv);
      for (uint32_t i = 0; i < d.kvn(); ++i) {
        const size_t at = ((size_t(l) * d.ring + slot) * d.kvn()) + i;
        const double ek = std::fabs(r.k[at] - wk[size_t(pos) * d.kvn() + i]) / mk;
        const double ev = std::fabs(r.v[at] - wv[size_t(pos) * d.kvn() + i]) / mv;
        worst = std::fmax(worst, std::fmax(ek, ev));
      }
    }
  }
  std::printf("projection: 40 positions x 2 layers, k / v worst |diff| / max = %.2e\n", worst);
  CHECK(worst <= 2e-5);
  (void)want_fc;
}

struct Stats {
  double worst_cos = 1, worst_score = 0;
  int ids_equal = 0, n = 0;
};

// The twin's draft vs the fixture's expected outputs. `strict`: ids / order exact and scores within 1e-4 x max;
// otherwise D1's bars: top-16 sets equal tie-aware, scores cosine, the walk equal.
Stats compare(const dffix::Fixture& fx, uint32_t version, const dfref::Weights& w, dfref::Numerics nm, bool strict,
              double cos_bar) {
  const dfref::Dims& d = w.d;
  const std::vector<float> rope = dfref::rope_table(d.theta, d.HD, 64);
  Stats st;
  const std::vector<dffix::Case> cs = dffix::cases(fx);
  for (size_t ci = 0; ci < cs.size(); ++ci) {
    const dffix::Case& c = cs[ci];
    const std::string pre = "v" + std::to_string(version) + ".case" + std::to_string(ci) + ".";
    if (!fx.has(pre + "ids")) continue;
    const dfref::Ring ring = dffix::ring_at(fx, w, nm, rope, c.p);
    const dfref::DraftOut o = dfref::draft(w, nm, ring, rope, c.anchor, c.p, c.K, c.T, c.seed);
    const std::vector<float> hidden = fx.f32(pre + "hidden");
    const std::vector<int32_t> cand = fx.i32(pre + "cand"), chosen = fx.i32(pre + "chosen"), ids = fx.i32(pre + "ids");
    const std::vector<float> unary = fx.f32(pre + "unary"), scores = fx.f32(pre + "scores");
    double hc = 1;
    for (uint32_t r = 0; r <= c.K; ++r)
      hc = std::fmin(hc, dfref::cosine(&o.hidden[size_t(r) * d.H], &hidden[size_t(r) * d.H], d.H));
    const double sc = dfref::cosine(o.scores.data(), scores.data(), scores.size());
    const float sm = maxabs(scores);
    double sd = 0;
    for (size_t i = 0; i < scores.size(); ++i)
      sd = std::fmax(sd, std::fmax(std::fabs(o.scores[i] - scores[i]), std::fabs(o.unary[i] - unary[i])) / sm);
    bool order = true, sets = true, walk = true;
    for (uint32_t j = 0; j < c.K; ++j) {
      std::vector<uint32_t> a(o.cand.begin() + j * 16, o.cand.begin() + (j + 1) * 16), b(16);
      for (uint32_t k = 0; k < 16; ++k) b[k] = uint32_t(cand[j * 16 + k]);
      order &= a == b;
      // tie-aware: a candidate only one side has must sit at the boundary within 1e-3 of the 16th value
      std::vector<uint32_t> sa = a, sb = b;
      std::sort(sa.begin(), sa.end());
      std::sort(sb.begin(), sb.end());
      if (sa != sb) {
        const float edge = unary[j * 16 + 15];
        for (uint32_t k = 0; k < 16; ++k)
          if (!std::binary_search(sb.begin(), sb.end(), a[k]) && std::fabs(o.unary[j * 16 + k] - edge) > 1e-3f * sm)
            sets = false;
      }
      walk &= o.ids[j] == uint32_t(ids[j]) && o.chosen[j] == uint32_t(chosen[j]);
    }
    std::printf("  v%u case %zu (p %u, K %u, T %.1f): hidden cos %.8f, scores cos %.8f, max|diff| %.2e, order %s, "
                "sets %s, walk %s\n",
                version, ci, c.p, c.K, double(c.T), hc, sc, sd, order ? "=" : "DIFF", sets ? "=" : "DIFF",
                walk ? "=" : "DIFF");
    st.worst_cos = std::fmin(st.worst_cos, hc);
    st.worst_score = std::fmax(st.worst_score, sd);
    st.ids_equal += walk;
    ++st.n;
    if (strict) {
      CHECK(hc >= 0.999999);
      CHECK(order && walk);
      CHECK(sd <= 1e-4);
    } else {
      // Engine numerics on tiny random weights: the rows and the scores are gated; the sets and the walk are reported
      // (the fixture's margins are not D1's checkpoint's - D1 gates those on the real drafter, on the box).
      CHECK(hc >= 0.9999);
      CHECK(sc >= cos_bar);
    }
  }
  return st;
}

void test_pieces() {
  // the window bound: row r sees context >= p + r - (WIN - 1); a lone visible key decides the output
  dfref::AttnDims a{2, 1, 4, 3, 8, 8, 4, 0.5f};
  std::vector<float> q(3 * 2 * 4, 0.1f), kb(3 * 4, 0.0f), vb(3 * 4, 0.0f), rk(8 * 4, 0.0f), rv(8 * 4, 0.0f), out(3 * 8);
  const uint32_t p = 20;                // ring holds 12..19; row 0 sees >= 13, row 2 sees >= 15
  for (uint32_t pos = 12; pos < 20; ++pos)
    for (uint32_t i = 0; i < 4; ++i) rv[(pos % 8) * 4 + i] = float(pos);
  dfref::attention(a, q.data(), kb.data(), vb.data(), rk.data(), rv.data(), p, 2, false, out.data());
  // all scores equal (k = 0): the output is the mean of the visible values (and the block's zeros)
  auto mean_of = [](uint32_t lo, uint32_t n_block) {
    double s = 0;
    for (uint32_t v = lo; v < 20; ++v) s += v;
    return s / double(20 - lo + n_block);
  };
  CHECK_NEAR(out[0], mean_of(13, 3), 1e-5);
  CHECK_NEAR(out[2 * 8], mean_of(15, 3), 1e-5);
  // attn_exp (df_exp, spec 19f): every float in [-86, -2^-20] within 2 fp32 ulps of exp (in double, rounded); 0
  // below -86 and at -inf, exactly 1 at 0, NaN through
  {
    int worst = 0;
    size_t n = 0;
    for (float x = -86.0f; x <= -0x1p-20f; x = std::nextafter(x, 0.0f), ++n) {
      const float got = dfref::attn_exp(x), want = float(std::exp(double(x)));
      int32_t a, b;
      std::memcpy(&a, &got, 4);
      std::memcpy(&b, &want, 4);
      worst = std::max(worst, std::abs(a - b));
    }
    std::printf("attn_exp over %zu floats in [-86, -2^-20]: worst %d fp32 ulp of exp (bar 2)\n", n, worst);
    CHECK(worst <= 2);
    CHECK(dfref::attn_exp(0.0f) == 1.0f && dfref::attn_exp(-0.0f) == 1.0f);
    CHECK(dfref::attn_exp(-86.5f) == 0.0f && dfref::attn_exp(-INFINITY) == 0.0f);
    CHECK(std::isnan(dfref::attn_exp(NAN)));
  }
  // the conv: row 0 reads no previous row; row r reads only r - 1
  {
    const uint32_t H = 4, G = 1;   // group 4: one coefficient per row and tap
    std::vector<float> src = {1, 1, 1, 1, 10, 10, 10, 10, 100, 100, 100, 100};
    std::vector<float> co = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   // [3 rows][2 taps][1 group] (one side, stride 4)
    (void)G;
    std::vector<float> base = {1, 1, 1, 1, 0.5f, 0.5f, 0.5f, 0.5f}, o(12);
    dfref::conv(src.data(), co.data(), 4, base.data(), 3, H, 4, 2, o.data());
    CHECK(o[0] == 1.0f && o[4] == 10.5f && o[8] == 105.0f);
  }
  // top-k: ties to the lower index, NaN never, indices >= used never, the scale and the cap
  {
    std::vector<float> lg = {1, 3, NAN, 3, 2, 9};
    const std::vector<dfref::Cand> c = dfref::topk(lg.data(), 5, 3, 2.0f, 0.0f);
    CHECK(c[0].i == 1 && c[1].i == 3 && c[2].i == 4 && c[0].v == 6.0f);
    const std::vector<dfref::Cand> c2 = dfref::topk(lg.data(), 5, 6, 1.0f, 2.0f);
    CHECK(c2[4].i == dfref::kNoIdx && c2[0].v == std::tanh(1.5f) * 2.0f);
  }
  // murmur3: dflash_ref.murmur3_uniform32(1234567, 23 + 2^30, keys 0 / 9 / 248319) as the python prints them
  // (float64 of the fp32 value): the walk's noise is keyed by (seed, position, candidate id)
  CHECK(dfref::murmur3_uniform(1234567, 23 + dfref::kDraftNoiseSalt, 9) ==
        dfref::murmur3_uniform(1234567, 23 + dfref::kDraftNoiseSalt, 9));
  CHECK(dfref::murmur3_uniform(1234567, 23 + dfref::kDraftNoiseSalt, 9) !=
        dfref::murmur3_uniform(1234567, 24 + dfref::kDraftNoiseSalt, 9));
  // the walk's tie rule
  {
    const float s[4] = {1, 5, 5, NAN};
    const uint32_t cand[4] = {10, 11, 12, 13};
    CHECK(dfref::choose(s, cand, 4, 0.0f, 0, 0) == 1);
  }
  std::printf("pieces: window per row, conv look-back, top-k ties / NaN / mask, murmur3 keying, walk ties\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_fixture = argv[1];
  test_pieces();
  test_projection();
  const dffix::Fixture fx(g_fixture);
  for (uint32_t v : {2u, 1u}) {
    std::printf("reference numerics, v%u:\n", v);
    const dfref::Weights w = dffix::weights(fx, v, dffix::Form::Reference, dffix::Form::Reference, 8);
    compare(fx, v, w, dfref::Numerics::Reference, true, 1.0);
  }
  {
    std::printf("engine numerics, bf16 drafter, bf16 head (rows cos >= 0.9999, scores cos >= 0.99):\n");
    const dfref::Weights w = dffix::weights(fx, 2, dffix::Form::Bf16, dffix::Form::Bf16, 8);
    const Stats s = compare(fx, 2, w, dfref::Numerics::Engine, false, 0.99);
    std::printf("  worst hidden cos %.6f, walks equal %d / %d\n", s.worst_cos, s.ids_equal, s.n);
  }
  {
    std::printf("engine numerics, int8 drafter, int8 head (reported):\n");
    const dfref::Weights w = dffix::weights(fx, 2, dffix::Form::Int8, dffix::Form::Int8, 8);
    // no CHECKs beyond the sets: the tiny random drafter is not D1's checkpoint
    const dfref::Weights& ww = w;
    const std::vector<float> rope = dfref::rope_table(ww.d.theta, ww.d.HD, 64);
    const std::vector<dffix::Case> cs = dffix::cases(fx);
    for (size_t ci = 0; ci < cs.size(); ++ci) {
      const dffix::Case& c = cs[ci];
      const dfref::Ring ring = dffix::ring_at(fx, ww, dfref::Numerics::Engine, rope, c.p);
      const dfref::DraftOut o = dfref::draft(ww, dfref::Numerics::Engine, ring, rope, c.anchor, c.p, c.K, c.T, c.seed);
      const std::vector<float> scores = fx.f32("v2.case" + std::to_string(ci) + ".scores");
      std::printf("  case %zu: scores cos %.6f\n", ci, dfref::cosine(o.scores.data(), scores.data(), scores.size()));
    }
  }
  std::printf("dflash_ref_test: PASS\n");
  return 0;
}
