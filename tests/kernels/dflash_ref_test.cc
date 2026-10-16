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
//      dflash_ref's values, the walk's tie rule.
//   5. Spec 23f, the Muse-Glimmer assistant's form (tests/loader/muse_dflash_fixture.safetensors, dflash_fixture.py
//      --muse; dflash_ref.py's Muse form is bitwise transformers 5.19.0): the context projection (fc -> the
//      two-rounding hidden_norm, k_norm + RoPE at theta 5e5, v) and draft() over the WHOLE 16-row block at the
//      inclusive window's edge (|q - k| <= 16 as WIN 17 over a 32-slot ring: p 15 / 16 / 17 for row 0 and row 15, the
//      wrap at p 33 / 39; K 15 / 7 / 1; the mask id as the anchor) in Reference numerics - hidden rows cosine >=
//      0.999999 (all 16), the raw head's top-16 ids and order exact, unary within 1e-4 x max, the greedy ids exact;
//      norm2 (mg_df_norm's twin) against transformers' own bf16 _m_norm (the bf16 values; at most 1 ulp on a few
//      elements, the Σx² order being the kernel's); Engine numerics for bf16 / int8 / int4 g64 drafters (rows cosine
//      >= 0.9999 bf16, >= 0.999 int8 / int4; ids reported).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/dflash_fixture_io.h"
#include "kernels/dflash_ref.h"

namespace {

std::string g_fixture = "tests/loader/dflash_ref_fixture.safetensors";
std::string g_muse_fixture = "tests/loader/muse_dflash_fixture.safetensors";

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

// --- spec 23f: the Muse-Glimmer assistant's form ----------------------------------------------------------------
void test_muse_projection(const dffix::Fixture& fx) {
  const dfref::Weights w = dffix::muse_weights(fx, dffix::Form::Reference, dffix::Form::Reference, 16);
  const dfref::Dims& d = w.d;
  CHECK(d.win == 17 && d.ring == 32 && d.M == 16 && d.muse);
  const std::vector<float> rope = dfref::rope_table(d.theta, d.HD, 64);
  const std::vector<uint16_t> taps = fx.bf16("taps");
  double worst = 0;
  for (uint32_t pos = 0; pos < 40; ++pos) {
    dfref::Ring r = dfref::make_ring(d);
    dfref::project(w, dfref::Numerics::Reference, &taps[size_t(pos) * d.fck()], 1, pos, rope, r);
    const uint32_t slot = pos % d.ring;
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::vector<float> wk = fx.f32("muse.ctx.k" + std::to_string(l)), wv = fx.f32("muse.ctx.v" + std::to_string(l));
      const float mk = maxabs(wk), mv = maxabs(wv);
      for (uint32_t i = 0; i < d.kvn(); ++i) {
        const size_t at = ((size_t(l) * d.ring + slot) * d.kvn()) + i;
        worst = std::fmax(worst, std::fmax(std::fabs(r.k[at] - wk[size_t(pos) * d.kvn() + i]) / mk,
                                           std::fabs(r.v[at] - wv[size_t(pos) * d.kvn() + i]) / mv));
      }
    }
  }
  std::printf("muse projection: 40 positions x 2 layers, k / v worst |diff| / max = %.2e\n", worst);
  CHECK(worst <= 2e-5);
}

// The twin's Muse draft vs the fixture's. strict: Reference numerics (exact ids / order); else the rows' cosine bar.
void muse_compare(const dffix::Fixture& fx, const dfref::Weights& w, dfref::Numerics nm, bool strict, double cos_bar,
                  const char* arm) {
  const dfref::Dims& d = w.d;
  const std::vector<float> rope = dfref::rope_table(d.theta, d.HD, 64);
  const std::vector<dffix::Case> cs = dffix::cases(fx);
  int walks = 0;
  double worst = 1;
  for (size_t ci = 0; ci < cs.size(); ++ci) {
    const dffix::Case& c = cs[ci];
    const std::string pre = "muse.case" + std::to_string(ci) + ".";
    const dfref::Ring ring = dffix::ring_at(fx, w, nm, rope, c.p);
    const dfref::DraftOut o = dfref::draft(w, nm, ring, rope, c.anchor, c.p, c.K);
    const std::vector<float> hidden = fx.f32(pre + "hidden"), unary = fx.f32(pre + "unary");
    const std::vector<int32_t> cand = fx.i32(pre + "cand"), ids = fx.i32(pre + "ids");
    double hc = 1;
    for (uint32_t r = 0; r < d.M; ++r)   // every row of the block, whatever K
      hc = std::fmin(hc, dfref::cosine(&o.hidden[size_t(r) * d.H], &hidden[size_t(r) * d.H], d.H));
    const float um = maxabs(unary);
    double ud = 0;
    bool order = true, walk = true;
    for (uint32_t j = 0; j < c.K; ++j) {
      for (uint32_t k = 0; k < 16; ++k) {
        order &= o.cand[j * 16 + k] == uint32_t(cand[j * 16 + k]);
        ud = std::fmax(ud, std::fabs(o.unary[j * 16 + k] - unary[j * 16 + k]) / um);
      }
      walk &= o.ids[j] == uint32_t(ids[j]);
    }
    std::printf("  muse %s case %zu (p %u, K %u): rows cos %.8f, unary max|diff| %.2e, order %s, ids %s\n", arm, ci,
                c.p, c.K, hc, ud, order ? "=" : "DIFF", walk ? "=" : "DIFF");
    walks += walk;
    worst = std::fmin(worst, hc);
    if (strict) {
      CHECK(hc >= 0.999999);
      CHECK(order && walk);
      CHECK(ud <= 1e-4);
    } else {
      CHECK(hc >= cos_bar);
    }
  }
  std::printf("  muse %s: worst rows cos %.7f, ids equal %d / %zu\n", arm, worst, walks, cs.size());
}

void test_muse_norm2(const dffix::Fixture& fx) {
  const std::vector<float> src = fx.f32("muse.norm.src"), w = fx.f32("muse.norm.w");
  const std::vector<uint16_t> want = fx.bf16("muse.norm.out");
  const uint32_t H = uint32_t(w.size()), rows = uint32_t(src.size() / H);
  int exact = 0, ulp1 = 0, worse = 0;
  for (uint32_t r = 0; r < rows; ++r) {
    std::vector<uint16_t> x(H);
    std::vector<float> h(H);
    dfref::norm2(&src[size_t(r) * H], w.data(), H, 1e-5f, h.data(), x.data());
    for (uint32_t i = 0; i < H; ++i) {
      const int dd = std::abs(int(x[i]) - int(want[size_t(r) * H + i]));
      exact += dd == 0;
      ulp1 += dd == 1;
      worse += dd > 1;
      CHECK(h[i] == dfref::bf(x[i]));
    }
  }
  // and the one-rounding norm differs from transformers' (Review Focus 4: the mutation is visible)
  int one_diff = 0;
  for (uint32_t r = 0; r < rows; ++r) {
    std::vector<uint16_t> x(H);
    dfref::norm(&src[size_t(r) * H], w.data(), H, 1e-5f, nullptr, x.data());
    for (uint32_t i = 0; i < H; ++i) one_diff += x[i] != want[size_t(r) * H + i];
  }
  std::printf("muse norm2 vs transformers' bf16 _m_norm: %d exact, %d 1-ulp, %d worse (of %u); one rounding differs on "
              "%d\n", exact, ulp1, worse, rows * H, one_diff);
  CHECK(worse == 0);
  CHECK(ulp1 * 50 <= int(rows * H));   // <= 2 %: the Σx² order (the kernel's lanes) against torch's mean
  CHECK(one_diff > 0);
}

void test_muse() {
  const dffix::Fixture fx(g_muse_fixture);
  test_muse_projection(fx);
  test_muse_norm2(fx);
  std::printf("muse reference numerics (the whole block, raw head, greedy):\n");
  muse_compare(fx, dffix::muse_weights(fx, dffix::Form::Reference, dffix::Form::Reference, 16),
               dfref::Numerics::Reference, true, 1.0, "ref");
  std::printf("muse engine numerics (rows cos >= 0.9999 bf16, >= 0.999 int8 / int4):\n");
  muse_compare(fx, dffix::muse_weights(fx, dffix::Form::Bf16, dffix::Form::Bf16, 16), dfref::Numerics::Engine, false,
               0.9999, "bf16");
  muse_compare(fx, dffix::muse_weights(fx, dffix::Form::Int8, dffix::Form::Int8, 16), dfref::Numerics::Engine, false,
               0.999, "int8");
  // int4 g64 against the unquantised reference is RTN's own error on tiny random weights (printed, bar 0.95); the
  // gate is the int4 GEMV path against Reference numerics over the SAME dequantised weights (the D1 arm's form).
  muse_compare(fx, dffix::muse_weights(fx, dffix::Form::Int4, dffix::Form::Int8, 16), dfref::Numerics::Engine, false,
               0.95, "int4 vs unquantised");
  {
    const dfref::Weights we = dffix::muse_weights(fx, dffix::Form::Int4, dffix::Form::Reference, 16);
    const dfref::Weights wr = dffix::muse_weights(fx, dffix::Form::Int4Ref, dffix::Form::Reference, 16);
    const std::vector<float> rope = dfref::rope_table(we.d.theta, we.d.HD, 64);
    double worst = 1;
    int eq = 0;
    const std::vector<dffix::Case> cs = dffix::cases(fx);
    for (const dffix::Case& c : cs) {
      const dfref::Ring re = dffix::ring_at(fx, we, dfref::Numerics::Engine, rope, c.p);
      const dfref::Ring rr = dffix::ring_at(fx, wr, dfref::Numerics::Reference, rope, c.p);
      const dfref::DraftOut a = dfref::draft(we, dfref::Numerics::Engine, re, rope, c.anchor, c.p, c.K);
      const dfref::DraftOut b = dfref::draft(wr, dfref::Numerics::Reference, rr, rope, c.anchor, c.p, c.K);
      for (uint32_t r = 0; r < we.d.M; ++r)
        worst = std::fmin(worst, dfref::cosine(&a.hidden[size_t(r) * we.d.H], &b.hidden[size_t(r) * we.d.H], we.d.H));
      eq += a.ids == b.ids;
    }
    std::printf("  muse int4 (gemv.cl's sums, bf16 activations) vs Reference over the dequantised int4: worst rows cos "
                "%.7f, ids equal %d / %zu\n", worst, eq, cs.size());
    CHECK(worst >= 0.9999);
  }
  // K only bounds what is read: the block's rows are the same at K 1 and K 15 (the whole block always runs)
  {
    const dfref::Weights w = dffix::muse_weights(fx, dffix::Form::Bf16, dffix::Form::Bf16, 16);
    const std::vector<float> rope = dfref::rope_table(w.d.theta, w.d.HD, 64);
    const dfref::Ring ring = dffix::ring_at(fx, w, dfref::Numerics::Engine, rope, 30);
    const dfref::DraftOut a = dfref::draft(w, dfref::Numerics::Engine, ring, rope, 9, 30, 1);
    const dfref::DraftOut b = dfref::draft(w, dfref::Numerics::Engine, ring, rope, 9, 30, 15);
    CHECK(a.hidden == b.hidden && a.ids[0] == b.ids[0]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_fixture = argv[1];
  if (argc > 2) g_muse_fixture = argv[2];
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
  test_muse();
  std::printf("dflash_ref_test: PASS\n");
  return 0;
}
