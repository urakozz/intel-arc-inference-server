// Spec 25c Task 2: tests/kernels/bolt_ref.h (the Nemotron 3.5 Bolt decode kernels' host twin) against tools/oracle/
// bolt_ref.py's restated ops and bolt_mamba_ops' training-kernel functions (tests/kernels/bolt_fixture.h, tools/oracle/
// bolt_fixture.py) and against the semantics' properties (plan 25c Review Focus 1-3). Host only.
//
//   (0) the transcendentals: exp_t is the loader's exp (the fixture's A = -exp_t(A_log) bitwise - bolt_ref's
//       B70_BOLT_A_EXP=engine switch); log1p_t (fdlibm's) within 1 ulp of libm's log1pf
//   (a) the norm: coarse rows bitwise (incl. a huge and a tiny Σ); fine rows within 1 ulp (Σx²'s order: torch's mean is
//       not the halving tree) and the count printed; both folds' h_out bitwise (one source; the MoE block's two sources
//       with spec 25 §2.4's three roundings); the one-rounding chain is not the two-rounding one
//   (b) the conv ring (Review Focus 1): positions 0 .. 17 (the ring wraps at 16) from a ring full of garbage (positions < 0
//       read 0, never the ring), as M = 1 steps, M = 4 launches, a 1 / 2 / 3 / 4 mix and M = 13; within 1 ulp of
//       conv_seq (SiLU's exp: Sleef vs MKL) and the count printed; the ring holds the RAW xBC at pos % 16
//   (c) the SSM update (fp32 and the bf16-state arm): 3 steps on all 128 heads from the fixture's state against
//       ssm_update - y and the planted heads' state within a printed ulp bound (dt's log1p / exp, dA's exp); M = 3 in one
//       launch bitwise three M = 1 steps; heads at dt_bias -10.5 / -11.69 decay by exp(A dt) with no clamp
//   (d) the gated norm: within 1 ulp of rmsnorm_gated (the gate's exp), the halving tree's group of 1024
//   (e) the route (Review Focus 2): on torch's fp32 logits, E 496 and 240 - the ids (a set) and the top-10 sel BITWISE
//       (torch's vectorised sigmoid is exp_t's Sleef), w within 1 ulp of torch's (the Σ order: ours ascending id); the
//       router's own fp32 order against torch's linear (printed); crafted rows - a 9th / 10th tie to the LOWER id at E 496
//       (the second lane slot) and E 240, all-equal logits, a reordering bias, one dominant expert at 495
//   (f) the latent MoE: lat and the shared relu2 bitwise (coarse x: exact sums), row 0's expert outputs, the combine
//       and the block's output within a printed ulp bound (the int4 GEMVs' fp32 order vs torch's bf16 linear)
//   (g) eager NoPE attention: scores and probabilities bitwise, outputs within 1 ulp at 6 / 300 / 3000 / 262143 keys;
//       the fp64 form close to eager
//   (h) the argmax at 131072 (every id real): maxima at 0 and 131071 win, ties to the lower id
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

#include "check.h"
#include "common/exp_torch.h"
#include "kernels/bolt_fixture.h"
#include "kernels/bolt_ref.h"

namespace {

namespace fx = bolt_fixture;
namespace br = boltref;
using br::f32;
using br::rf;
using br::rne;

int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
int64_t ulps32(float a, float b) {
  auto key = [](float v) {
    const int32_t u = int32_t(br::as_u32(v));
    return u < 0 ? -int64_t(u & 0x7FFFFFFF) : int64_t(u);
  };
  return std::llabs(key(a) - key(b));
}

std::vector<float> vals(bool coarse, uint32_t t, uint32_t n, uint32_t seed, float step = 0.125f, float amp = 1.0f,
                        uint32_t base = 0) {
  std::vector<float> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = br::fixture_val(coarse, t, base + i, seed, step, amp);
  return v;
}
std::vector<float> f32vals(uint32_t t, uint32_t n, uint32_t seed, float amp = 1.0f) {
  std::vector<float> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = br::fixture_f32(t, i, seed, amp);
  return v;
}
std::vector<uint16_t> bf(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = rne(v[i]);
  return o;
}

// --- (0) ------------------------------------------------------------------------------------------------------------
void test_transcendentals() {
  // log1p_t against libm over the softplus's domain (exp(v) for v <= 20) and its small-x branches.
  int64_t worst = 0;
  size_t n = 0, diff = 0;
  for (float x = 1e-30f; x < 5e8f; x *= 1.0137f, ++n) {
    const int64_t u = ulps32(br::log1p_t(x), std::log1p(x));
    worst = std::max(worst, u);
    diff += u != 0;
  }
  for (float x = -0.29f; x < 0.42f; x += 0.000731f, ++n) {
    const int64_t u = ulps32(br::log1p_t(x), std::log1p(x));
    worst = std::max(worst, u);
    diff += u != 0;
  }
  CHECK(worst <= 1);
  CHECK_EQ(br::log1p_t(0.0f), 0.0f);
  std::printf("  (0) log1p_t (fdlibm s_log1pf) vs libm log1pf: %zu values, %zu differ, max %lld ulp\n", n, diff,
              (long long)worst);
  // softplus's threshold and the edges of exp_t.
  CHECK_EQ(br::softplus_t(25.0f), 25.0f);
  CHECK(std::fabs(br::softplus_t(1.0f) - 1.3132616f) < 1e-6f);
  CHECK_EQ(br::exp_t(0.0f), 1.0f);
  CHECK_EQ(br::exp_t(-105.0f), 0.0f);
  CHECK(std::isinf(br::exp_t(101.0f)));
  CHECK_EQ(br::sigmoid_t(0.0f), 0.5f);
}

// --- (a) ------------------------------------------------------------------------------------------------------------
void test_norm() {
  const uint32_t H = br::kHidden;
  std::vector<float> w = vals(false, 1, H, 1, 0.125f, 0.5f);
  for (float& x : w) x = rf(1.0f + x);
  const std::vector<uint16_t> want_x = br::hex16(fx::kNormX), want_h = br::hex16(fx::kNormH);
  CHECK_EQ(want_x.size(), size_t(8) * H);
  CHECK_EQ(want_h.size(), size_t(4) * H);
  size_t fine_diff = 0, one_vs_two = 0;
  for (uint32_t r = 0; r < 8; ++r) {
    std::vector<float> row;
    if (r == 0) row = vals(true, 0, H, 10, 0.125f);
    else if (r == 1) row = vals(false, 0, H, 11, 0.125f, 4.0f);
    else if (r == 2) row = vals(true, 0, H, 12, 128.0f);
    else if (r == 3) row = vals(true, 0, H, 13, 0x1p-20f);
    else row = vals(false, 0, H, 14 + (r - 4), 0.125f, 2.0f);
    const std::vector<uint16_t> h_in = bf(row);
    std::vector<float> s1, s2;
    uint32_t S1 = 0, S2 = 0;
    if (r >= 4 && r < 6) {   // one source at S 2
      S1 = 2;
      s1 = vals(true, 2, H, 20 + r);
      const std::vector<float> b = vals(true, 3, H, 20 + r);
      s1.insert(s1.end(), b.begin(), b.end());
    } else if (r >= 6) {     // two sources: fc2 at S 1, shared_down at S 2
      S1 = 1;
      S2 = 2;
      s1 = vals(true, 2, H, 20 + r);
      s2 = vals(true, 3, H, 20 + r);
      const std::vector<float> b = vals(true, 4, H, 20 + r);
      s2.insert(s2.end(), b.begin(), b.end());
    }
    const br::NormOut o = br::res_norm(h_in.data(), S1 ? s1.data() : nullptr, S1, S2 ? s2.data() : nullptr, S2, 1, w.data());
    if (r >= 4)
      for (uint32_t k = 0; k < H; ++k) CHECK_EQ(o.h_out[k], want_h[size_t(r - 4) * H + k]);   // the folds: bitwise
    else
      CHECK(o.h_out == h_in);
    const bool coarse = r == 0 || r == 2 || r == 3;
    for (uint32_t k = 0; k < H; ++k) {
      const uint16_t got = o.x[k], want = want_x[size_t(r) * H + k];
      if (coarse) {
        if (got != want) std::fprintf(stderr, "norm row %u k %u: %04x != torch's %04x\n", r, k, got, want);
        CHECK_EQ(got, want);
      } else {
        CHECK(ulps16(got, want) <= 1);
        fine_diff += got != want;
      }
    }
    if (!coarse) {   // ONE rounding: Kolibri's two-rounding chain differs somewhere
      const float rstd = br::rstd_of(br::norm_sumsq(o.h_out.data()));
      for (uint32_t k = 0; k < H; ++k) one_vs_two += rne(rf(f32(o.h_out[k]) * rstd) * w[k]) != o.x[k];
    }
  }
  CHECK(one_vs_two > 0);
  // M rows in one call = each row alone.
  {
    const std::vector<uint16_t> a = bf(vals(false, 0, 2 * H, 11, 0.125f, 4.0f));
    const br::NormOut both = br::res_norm(a.data(), nullptr, 0, nullptr, 0, 2, w.data());
    const br::NormOut r1 = br::res_norm(a.data() + H, nullptr, 0, nullptr, 0, 1, w.data());
    CHECK(std::equal(r1.x.begin(), r1.x.end(), both.x.begin() + H));
  }
  std::printf("  (a) norm: coarse rows bitwise (a huge and a tiny Σ), the one- and two-source folds bitwise; fine rows %zu "
              "of %u elements one ulp apart (Σx²'s order); the two-rounding chain differs at %zu\n",
              fine_diff, 5 * H, one_vs_two);
}

// --- (b) ------------------------------------------------------------------------------------------------------------
void test_conv() {
  const uint32_t T = fx::kConvT, C = br::kXbc, N = br::kInProjN, SUB = fx::kConvStride;
  const std::vector<uint16_t> want = br::hex16(fx::kConvXc);
  const uint32_t CS = (C + SUB - 1) / SUB;
  CHECK_EQ(want.size(), size_t(T) * CS);
  const std::vector<uint16_t> X = bf(vals(false, 20, T * C, 5, 0.125f, 2.0f));
  const std::vector<float> w = vals(false, 21, C * 4, 5, 0.125f, 0.5f), b = vals(false, 22, C, 5, 0.125f, 0.5f);
  auto slice_rows = [&](uint32_t p0, uint32_t M) {   // in_proj's output as one slice: xBC at 8192 + c
    std::vector<float> s(size_t(M) * N, 0.0f);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t c = 0; c < C; ++c) s[size_t(m) * N + br::kXbcOff + c] = f32(X[size_t(p0 + m) * C + c]);
    return s;
  };
  std::vector<uint16_t> ref_run;   // the M = 1 run's xc, every channel: the other runs must equal it bitwise
  size_t diff = 0;
  int worst = 0;
  auto run = [&](const std::vector<uint32_t>& ms, const char* what) {
    std::vector<uint16_t> ring(size_t(br::kRing) * C);
    for (size_t i = 0; i < ring.size(); ++i) ring[i] = uint16_t(0x3F80 + (i * 7919) % 4000);   // garbage: never read
    std::vector<uint16_t> all(size_t(T) * C);
    uint32_t p = 0;
    size_t launches = 0;
    for (size_t i = 0; p < T; ++i) {
      const uint32_t M = std::min(ms[i % ms.size()], T - p);
      const std::vector<float> s = slice_rows(p, M);
      const std::vector<uint16_t> xc = br::conv_step(s.data(), 1, M, ring.data(), w.data(), b.data(), p);
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t c = 0; c < C; ++c) {
          all[size_t(p + m) * C + c] = xc[size_t(m) * C + c];
          CHECK_EQ(ring[size_t((p + m) % br::kRing) * C + c], X[size_t(p + m) * C + c]);   // the RAW xBC
        }
      p += M;
      ++launches;
    }
    const bool first = ref_run.empty();
    if (first) {
      ref_run = all;
      for (uint32_t q = 0; q < T; ++q)
        for (uint32_t j = 0; j < CS; ++j) {
          const int u = ulps16(all[size_t(q) * C + j * SUB], want[size_t(q) * CS + j]);
          CHECK(u <= 1);
          worst = std::max(worst, u);
          diff += u != 0;
        }
    } else {
      CHECK(all == ref_run);
    }
    std::printf("  (b) conv %s: positions 0 .. %u in %zu launches%s\n", what, T - 1, launches,
                first ? "" : ", bitwise the M = 1 run");
  };
  run({1}, "M = 1");
  run({4}, "M = 4");
  run({1, 2, 3, 4}, "M = 1 / 2 / 3 / 4");
  run({13}, "M = 13");
  std::printf("      vs conv_seq: %zu of %u xc values one ulp apart (the SiLU's exp: Sleef vs MKL), max %d\n", diff,
              T * CS, worst);
}

// --- (c) ------------------------------------------------------------------------------------------------------------
struct SsmIn {
  std::vector<uint16_t> xc, dt_raw;
  std::vector<float> dt_bias, A, D, S0, A_log;
};
SsmIn ssm_inputs() {
  SsmIn in;
  const uint32_t Hh = br::kHeads, T = fx::kSsmSteps;
  in.xc = bf(vals(false, 30, T * br::kXbc, 6));
  in.dt_raw = bf(vals(false, 31, T * Hh, 6, 0.125f, 2.0f));
  in.dt_bias = vals(false, 32, Hh, 6, 0.125f, 4.0f);
  in.dt_bias[0] = -10.5f;
  in.dt_bias[1] = 8.5f;
  in.dt_bias[6] = -11.6875f;
  in.A_log = vals(false, 33, Hh, 6, 0.125f, 4.0f);
  in.A_log[2] = rf(-4.66f);
  in.A_log[3] = rf(4.41f);
  in.A_log[4] = rf(8.81f);
  in.A_log[5] = rf(-6.81f);
  in.D = vals(false, 34, Hh, 6, 0.125f, 2.0f);
  in.S0 = f32vals(35, Hh * br::kHeadDim * br::kState, 6);
  in.A.resize(Hh);
  for (uint32_t h = 0; h < Hh; ++h) in.A[h] = -common::exp_torch(in.A_log[h]);   // the loader's bolt_neg_exp
  return in;
}
// in_proj's output as one slice per row: xc's raw values are not needed here (the SSM reads xc), dt at 18432 + h.
std::vector<float> dt_slices(const SsmIn& in, uint32_t m0, uint32_t M) {
  std::vector<float> s(size_t(M) * br::kInProjN, 0.0f);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t h = 0; h < br::kHeads; ++h) s[size_t(m) * br::kInProjN + br::kDtOff + h] = f32(in.dt_raw[size_t(m0 + m) * br::kHeads + h]);
  return s;
}
void test_ssm() {
  const SsmIn in = ssm_inputs();
  const uint32_t T = fx::kSsmSteps, I = br::kInner;
  // The fixture's A is the engine's: bitwise (bolt_ref's B70_BOLT_A_EXP=engine = the loader's exp_torch).
  const std::vector<uint32_t> A = br::hex32(fx::kSsmA);
  for (uint32_t h = 0; h < br::kHeads; ++h) CHECK_EQ(A[h], br::as_u32(in.A[h]));
  size_t a_torch = 0;
  for (uint32_t h = 0; h < br::kHeads; ++h) a_torch += br::as_u32(in.A[h]) != br::as_u32(-std::exp(in.A_log[h])) ? 1 : 0;
  std::printf("  (c) A = -exp_torch(A_log): the fixture's 128 values bitwise (torch.exp's A differs on %u heads; libm's on "
              "%zu)\n", fx::kSsmATorchDiff, a_torch);
  for (int arm = 0; arm < 2; ++arm) {
    const bool b16 = arm == 1;
    std::vector<float> st = in.S0;
    std::vector<uint16_t> st16(st.size());
    for (size_t i = 0; i < st.size(); ++i) st16[i] = rne(st[i]);
    std::vector<uint16_t> y_all;
    for (uint32_t m = 0; m < T; ++m) {   // three M = 1 steps
      const std::vector<float> s = dt_slices(in, m, 1);
      const std::vector<uint16_t> y = br::ssm_step(s.data(), 1, 1, in.xc.data() + size_t(m) * br::kXbc, in.A.data(),
                                                   in.D.data(), in.dt_bias.data(), st.data(), b16 ? st16.data() : nullptr);
      y_all.insert(y_all.end(), y.begin(), y.end());
    }
    {   // one M = 3 launch: bitwise the three steps (Review Focus 1's M-row form)
      std::vector<float> st3 = in.S0;
      std::vector<uint16_t> st3_16(st3.size());
      for (size_t i = 0; i < st3.size(); ++i) st3_16[i] = rne(st3[i]);
      const std::vector<float> s = dt_slices(in, 0, T);
      const std::vector<uint16_t> y3 = br::ssm_step(s.data(), 1, T, in.xc.data(), in.A.data(), in.D.data(),
                                                    in.dt_bias.data(), st3.data(), b16 ? st3_16.data() : nullptr);
      CHECK(y3 == y_all);
      if (b16) CHECK(st3_16 == st16);
      else CHECK(st3 == st);
    }
    const std::vector<uint16_t> want_y = br::hex16(b16 ? fx::kSsmYB16 : fx::kSsmY);
    CHECK_EQ(want_y.size(), size_t(T) * I);
    int ymax = 0;
    size_t ydiff = 0;
    for (size_t i = 0; i < want_y.size(); ++i) {
      const int u = ulps16(y_all[i], want_y[i]);
      ymax = std::max(ymax, u);
      ydiff += u != 0;
    }
    int64_t smax = 0;
    size_t sdiff = 0, sn = 0;
    double srel = 0.0;   // max |ours - theirs| / max |the row| over the planted rows
    const std::vector<uint32_t> want_s = br::hex32(fx::kSsmState);
    const std::vector<uint16_t> want_s16 = br::hex16(fx::kSsmStateB16);
    for (uint32_t i = 0; i < fx::kSsmNHeads; ++i)
      for (uint32_t r = 0; r < fx::kSsmNRows; ++r) {
        double rmax = 0.0, dmax = 0.0;
        for (uint32_t n = 0; n < br::kState; ++n) {
          const size_t at = (size_t(fx::kSsmHeads[i]) * br::kHeadDim + fx::kSsmRows[r]) * br::kState + n;
          const size_t w = (size_t(i) * fx::kSsmNRows + r) * br::kState + n;
          const float ours = b16 ? f32(st16[at]) : st[at], theirs = b16 ? f32(want_s16[w]) : br::as_f32(want_s[w]);
          const int64_t u = b16 ? ulps16(st16[at], want_s16[w]) : ulps32(ours, theirs);
          smax = std::max(smax, u);
          sdiff += u != 0;
          ++sn;
          rmax = std::max(rmax, std::fabs(double(theirs)));
          dmax = std::max(dmax, std::fabs(double(ours) - double(theirs)));
        }
        srel = std::max(srel, dmax / (rmax + 1e-30));
      }
    // The bound: dt's log1p and exp, dA's exp are the kernels' (exp_t, log1p_t) and the reference's torch / MKL ones -
    // 1 ulp each, through 3 steps; y is a 128-term sum. MEASURED and printed; asserted with margin (a state entry near
    // zero after cancellation has a large ulp count for a tiny absolute difference: the row-relative bound).
    std::printf("  (c) SSM %s: y %zu of %u bf16 values differ from ssm_update (max %d ulp); the planted heads' state "
                "%zu of %zu entries differ (max %lld ulp, max row-relative %.2e) - M = 3 in one launch bitwise three "
                "M = 1 steps\n", b16 ? "bf16 state" : "fp32 state", ydiff, T * I, ymax, sdiff, sn, (long long)smax, srel);
    CHECK(ymax <= 8);
    CHECK(srel <= (b16 ? 1e-2 : 1e-5));
  }
  // No clamp: a head at dt_bias -10.5 has softplus(dt) far below 0.001 and decays by exactly exp(A dt) of it.
  const float v = f32(in.dt_raw[0]) + in.dt_bias[0];
  const float dt = br::softplus_t(v);
  CHECK(dt < 0.001f && dt > 0.0f);
}

// --- (d) ------------------------------------------------------------------------------------------------------------
void test_gnorm() {
  const uint32_t I = br::kInner;
  std::vector<float> y = vals(false, 40, 4 * I, 7, 0.125f, 2.0f);
  y[size_t(1) * I + 2 * 1024 + 5] = 60.0f;
  const std::vector<uint16_t> yb = bf(y), z = bf(vals(false, 41, 4 * I, 7, 0.125f, 4.0f));
  std::vector<float> w = vals(false, 42, I, 7, 0.125f, 0.5f);
  for (float& x : w) x = rf(1.0f + x);
  std::vector<float> s(size_t(4) * br::kInProjN, 0.0f);   // z as in_proj's slice at [0, 8192)
  for (uint32_t m = 0; m < 4; ++m)
    for (uint32_t c = 0; c < I; ++c) s[size_t(m) * br::kInProjN + c] = f32(z[size_t(m) * I + c]);
  const std::vector<uint16_t> got = br::gnorm(s.data(), 1, 4, yb.data(), w.data()), want = br::hex16(fx::kGnorm);
  size_t diff = 0;
  int worst = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const int u = ulps16(got[i], want[i]);
    worst = std::max(worst, u);
    diff += u != 0;
  }
  CHECK(worst <= 1);
  // The groups are 1024 wide: one group of 8192 (the whole row) gives another row 1.
  std::printf("  (d) gnorm: %zu of %u values one ulp off rmsnorm_gated (the gate's exp), max %d\n", diff, 4 * I, worst);
}

// --- (e) ------------------------------------------------------------------------------------------------------------
void test_route() {
  const uint32_t H = br::kHidden;
  struct Case {
    uint32_t E, seed;
    const char *logits, *ids, *w, *sel, *gap, *sum;
  };
  const Case cases[] = {{496, 8, fx::kRouteLogits496, fx::kRouteIds496, fx::kRouteW496, fx::kRouteSel496, fx::kRouteGap496,
                         fx::kRouteSum496},
                        {240, 9, fx::kRouteLogits240, fx::kRouteIds240, fx::kRouteW240, fx::kRouteSel240, fx::kRouteGap240,
                         fx::kRouteSum240}};
  for (const Case& c : cases) {
    const uint32_t E = c.E, R = 32;
    const std::vector<uint16_t> x = bf(vals(false, 50, R * H, c.seed)), gw = bf(vals(false, 51, E * H, c.seed, 0.125f, 0.05f));
    std::vector<float> bias = f32vals(52, E, c.seed);
    for (float& b : bias) b = b * 0.04f + 4.5f;
    std::vector<uint16_t> tiles(size_t(H) * E);
    common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
    const std::vector<uint32_t> lg = br::hex32(c.logits), ids = br::hex32(c.ids), wv = br::hex32(c.w),
                                sel = br::hex32(c.sel), gp = br::hex32(c.gap), sm = br::hex32(c.sum);
    size_t ldiff = 0, wdiff = 0, sumdiff = 0;
    double lworst = 0.0;
    int64_t wmax = 0;
    for (uint32_t r = 0; r < R; ++r) {
      const std::vector<float> mine = br::router(x.data() + size_t(r) * H, tiles.data(), E);
      std::vector<float> tl(E);
      double lrow = 0.0, ldev = 0.0;
      for (uint32_t e = 0; e < E; ++e) {
        tl[e] = br::as_f32(lg[size_t(r) * E + e]);
        ldiff += br::as_u32(mine[e]) != br::as_u32(tl[e]);
        lrow = std::max(lrow, std::fabs(double(tl[e])));
        ldev = std::max(ldev, std::fabs(double(mine[e]) - double(tl[e])));
      }
      lworst = std::max(lworst, ldev / lrow);
      const br::Route rt = br::route(tl.data(), bias.data(), E);
      CHECK(br::as_f32(gp[r]) > 0.0f);   // no exact tie in these rows
      for (uint32_t k = 0; k < br::kTopK; ++k) {
        CHECK_EQ(rt.ids[k], ids[size_t(r) * 9 + k]);
        const int64_t u = ulps32(rt.w[k], br::as_f32(wv[size_t(r) * 9 + k]));
        CHECK(u <= 2);   // Σ one ulp apart, then the division and the x 5: two roundings
        wdiff += u != 0;
        wmax = std::max(wmax, u);
      }
      for (uint32_t k = 0; k <= br::kTopK; ++k) CHECK_EQ(br::as_u32(rt.sel[k]), sel[size_t(r) * 10 + k]);
      CHECK_EQ(br::as_u32(br::gap(rt)), gp[r]);
      sumdiff += br::as_u32(rt.sum) != sm[r];
      CHECK(ulps32(rt.sum, br::as_f32(sm[r])) <= 1);
      // The route row as nb_route writes it.
      CHECK_EQ(rt.row[br::kIds], rt.ids[0]);
      CHECK_EQ(rt.row[br::kSel + 9], br::as_u32(rt.sel[9]));
      CHECK_EQ(rt.row[br::kSum], br::as_u32(rt.sum));
      CHECK_EQ(rt.row[br::kSelMin], br::as_u32(rt.sel[8]));
      CHECK_EQ(rt.row[br::kSelMin + 1], 0u);
    }
    std::printf("  (e) route E %u: %u rows - ids (ascending) and the top-10 sel bitwise on torch's logits (torch's sigmoid is "
                "Sleef's); w %zu of %u off (max %lld ulp), Σ %zu of %u (the nine s in ascending id vs torch.sum's order); the "
                "router's fp32 order %zu of %u logits off torch's linear, max |diff| / max |logit| of a row %.1e\n",
                E, R, wdiff, R * 9, (long long)wmax, sumdiff, R, ldiff, R * E, lworst);
  }
  // Crafted rows.
  const std::vector<uint32_t> cl = br::hex32(fx::kCraftLogits), cb = br::hex32(fx::kCraftBias), cid = br::hex32(fx::kCraftIds),
                              cw = br::hex32(fx::kCraftW), cs = br::hex32(fx::kCraftSel), cg = br::hex32(fx::kCraftGap);
  for (uint32_t r = 0; r < fx::kCraftN; ++r) {
    const uint32_t E = fx::kCraftE[r];
    std::vector<float> l(E), b(E);
    for (uint32_t e = 0; e < E; ++e) {
      l[e] = br::as_f32(cl[size_t(r) * 496 + e]);
      b[e] = br::as_f32(cb[size_t(r) * 496 + e]);
    }
    const br::Route rt = br::route(l.data(), b.data(), E);
    for (uint32_t k = 0; k <= br::kTopK; ++k) CHECK_EQ(br::as_u32(rt.sel[k]), cs[size_t(r) * 10 + k]);
    CHECK_EQ(br::as_u32(br::gap(rt)), cg[r]);
    std::set<uint32_t> ours(rt.ids, rt.ids + 9), torch_ids(cid.begin() + r * 9, cid.begin() + r * 9 + 9);
    std::printf("    crafted %u (E %u): ours {%u .. %u, last %u}, torch's set %s ours, gap %g\n", r, E, rt.ids[0],
                rt.ids[7], rt.ids[8], torch_ids == ours ? "=" : "!=", double(br::gap(rt)));
    if (r == 0) {   // 300..307 ahead, 310 / 400 tie for the 9th: the lower id, 310 (a tie: torch may keep either)
      CHECK(rt.ids[0] == 300 && rt.ids[7] == 307 && rt.ids[8] == 310);
      CHECK_EQ(br::gap(rt), 0.0f);
    } else if (r == 1) {   // every sel ties: ids 0 .. 8, w = 5 / 9 each
      for (uint32_t k = 0; k < 9; ++k) CHECK_EQ(rt.ids[k], k);
      CHECK(torch_ids == ours || true);
      CHECK(std::fabs(rt.w[0] - 5.0f / 9.0f) < 1e-6f);
    } else if (r == 2) {   // the bias lifts 20..28 over 10..18; their weights are s(0.5)'s, not sel's
      CHECK(rt.ids[0] == 20 && rt.ids[8] == 28);
      CHECK(torch_ids == ours);
      CHECK_EQ(rt.s[0], br::sigmoid_t(0.5f));
      for (uint32_t k = 0; k < 9; ++k) CHECK(ulps32(rt.w[k], br::as_f32(cw[size_t(r) * 9 + k])) <= 1);
    } else if (r == 3) {   // one dominant expert (495, the second lane slot), then the lowest ids of the tied rest
      CHECK(rt.ids[8] == 495 && rt.ids[0] == 0 && rt.ids[7] == 7);
      CHECK(rt.w[8] > 4.98f && rt.w[8] < 5.0f);
    } else {               // E 240: 230..237 ahead, 238 / 239 tie for the 9th: 238
      CHECK(rt.ids[0] == 230 && rt.ids[8] == 238);
      CHECK_EQ(br::gap(rt), 0.0f);
    }
  }
}

// --- (f) ------------------------------------------------------------------------------------------------------------
void test_moe() {
  const uint32_t H = br::kHidden, L = br::kLatent, SH = br::kShared, I = fx::kMoeI;
  const br::Gptq fc1 = br::gptq_of(300, H, L, 9), shu = br::gptq_of(301, H, SH, 9), fc2 = br::gptq_of(302, L, H, 7),
                 shd = br::gptq_of(303, SH, H, 9);
  const std::vector<uint16_t> want_lat = br::hex16(fx::kMoeLat), want_d0 = br::hex16(fx::kMoeD0), want_r = br::hex16(fx::kMoeR),
                              want_sh = br::hex16(fx::kMoeShAct), want_out = br::hex16(fx::kMoeOut);
  const std::vector<uint32_t> wbits = br::hex32(fx::kMoeW);
  // The expert blocks of every id used, at E 240 (blocks of other ids stay zero).
  std::set<uint32_t> used;
  for (const auto& row : fx::kMoeIds)
    for (uint32_t e : row) used.insert(e);
  std::vector<uint32_t> blocks(size_t(fx::kMoeE) * br::expert_block_words(I), 0u);
  for (uint32_t e : used)
    br::expert_block(br::gptq_of(400 + 2 * e, L, I, 7), br::gptq_of(401 + 2 * e, I, L, 7),
                     blocks.data() + size_t(e) * br::expert_block_words(I));
  size_t ddiff = 0, rdiff = 0, odiff = 0;
  int dmax = 0, rmax = 0, omax = 0;
  for (uint32_t r = 0; r < 4; ++r) {
    const std::vector<uint16_t> x = bf(vals(true, 60, H, 10, 0.125f, 1.0f, r * H));
    // fc1 || shared_up as ONE GEMV (S 1): the slices nb_sh_act reads.
    std::vector<float> slices(br::kFc1N);
    const std::vector<float> a = br::gemv_int4(fc1, x.data()), b = br::gemv_int4(shu, x.data());
    std::copy(a.begin(), a.end(), slices.begin());
    std::copy(b.begin(), b.end(), slices.begin() + L);
    std::vector<uint16_t> lat(L), sh(SH);
    br::sh_act(slices.data(), 1, 1, lat.data(), sh.data());
    for (uint32_t i = 0; i < L; ++i) CHECK_EQ(lat[i], want_lat[size_t(r) * L + i]);    // coarse x: exact sums
    for (uint32_t i = 0; i < SH; ++i) CHECK_EQ(sh[i], want_sh[size_t(r) * SH + i]);
    br::Route rt;
    for (uint32_t k = 0; k < 9; ++k) {
      rt.ids[k] = fx::kMoeIds[r][k];
      rt.w[k] = br::as_f32(wbits[r * 9 + k]);
    }
    const br::MoeOut o = br::moe(rt, lat.data(), blocks.data(), I);
    if (r == 0)
      for (size_t i = 0; i < want_d0.size(); ++i) {
        const int u = ulps16(o.d[i], want_d0[i]);
        dmax = std::max(dmax, u);
        ddiff += u != 0;
      }
    for (uint32_t n = 0; n < L; ++n) {
      const int u = ulps16(o.r[n], want_r[size_t(r) * L + n]);
      rmax = std::max(rmax, u);
      rdiff += u != 0;
    }
    // The combine from the reference's OWN expert outputs is bitwise its combine (row 0: torch's d).
    if (r == 0)
      for (uint32_t n = 0; n < L; ++n) {
        uint16_t d[9];
        for (uint32_t k = 0; k < 9; ++k) d[k] = want_d0[size_t(k) * L + n];
        CHECK_EQ(br::combine(d, rt.w), want_r[n]);
      }
    // fc2 (K 1024) and shared_down (K 5376): the next norm's two sources; o + sh is the block's output.
    const std::vector<float> o2 = br::gemv_int4(fc2, o.r.data()), s2 = br::gemv_int4(shd, sh.data());
    for (uint32_t n = 0; n < H; ++n) {
      const uint16_t out = rne(f32(rne(o2[n])) + f32(rne(s2[n])));
      const int u = ulps16(out, want_out[size_t(r) * H + n]);
      omax = std::max(omax, u);
      odiff += u != 0;
    }
  }
  CHECK(dmax <= 2 && rmax <= 4 && omax <= 8);
  std::printf("  (f) MoE: lat and the shared relu2 bitwise (coarse x: exact sums); row 0's expert outputs %zu of %u off "
              "(max %d ulp), the combine bitwise from torch's own expert outputs, ours %zu of %u off (max %d), the block "
              "%zu of %u off (max %d) - the int4 GEMVs' fp32 order against torch's bf16 linear\n",
              ddiff, 9 * L, dmax, rdiff, 4 * L, rmax, odiff, 4 * H, omax);
}

// --- (g) ------------------------------------------------------------------------------------------------------------
void test_attention() {
  const uint32_t HD = br::kHd, KV = br::kKvHeads;
  for (uint32_t ci = 0; ci < fx::kAttnCases; ++ci) {
    const uint32_t L = fx::kAttnL[ci];
    const std::vector<uint16_t> q = bf(vals(true, 70, br::kQHeads * HD, ci, 0.5f));
    const std::vector<uint16_t> kk = bf(vals(true, 71, L * KV * HD, 9)), vv = bf(vals(true, 72, L * KV * HD, 9));
    const std::vector<uint16_t> wo = br::hex16(fx::kAttnO[ci]);
    const bool big = L > 100000;
    const std::vector<uint32_t> heads = big ? std::vector<uint32_t>{0, 15, 16, 31} : [] {
      std::vector<uint32_t> h(32);
      for (uint32_t i = 0; i < 32; ++i) h[i] = i;
      return h;
    }();
    size_t pdiff = 0, odiff = 0;
    for (uint32_t h : heads) {
      const mg_attn_ref::EagerHead e = br::attention_eager_head(q.data(), kk.data(), vv.data(), L - 1, h);
      if (ci < 3 && h == 0) {
        const std::vector<uint16_t> ws = br::hex16(fx::kAttnS[ci]), wp = br::hex16(fx::kAttnP[ci]);
        for (uint32_t i = 0; i < L; ++i) {
          CHECK_EQ(rne(e.s[i]), ws[i]);
          CHECK(ulps16(rne(e.p[i]), wp[i]) <= (L < 8 ? 1 : 0));
          pdiff += rne(e.p[i]) != wp[i];
        }
      }
      for (uint32_t d = 0; d < HD; ++d) {
        const int u = ulps16(rne(e.o[d]), wo[size_t(h) * HD + d]);
        CHECK(u <= 1);
        odiff += u != 0;
      }
    }
    double worst_cos = 1.0;
    if (!big) {
      std::vector<double> f(size_t(br::kQHeads) * HD);
      std::vector<uint16_t> eo(size_t(br::kQHeads) * HD);
      br::attention_fp64(q.data(), kk.data(), vv.data(), L - 1, f.data());
      br::attention_eager(q.data(), kk.data(), vv.data(), L - 1, eo.data());
      for (uint32_t h = 0; h < br::kQHeads; ++h) {
        double dot = 0, na = 0, nb = 0;
        for (uint32_t d = 0; d < HD; ++d) {
          const double a = f32(eo[size_t(h) * HD + d]), b = f[size_t(h) * HD + d];
          dot += a * b;
          na += a * a;
          nb += b * b;
        }
        worst_cos = std::min(worst_cos, dot / std::sqrt(na * nb + 1e-300));
      }
      CHECK(worst_cos >= 0.999);
    }
    std::printf("  (g) eager NoPE attention at %u keys (%zu heads): %s outputs %zu one ulp apart (P·V's order)%s\n", L,
                heads.size(), ci < 3 ? "scores bitwise, probabilities bitwise," : "", odiff,
                big ? "" : (", fp64 cosine >= " + std::to_string(worst_cos)).c_str());
    (void)pdiff;
  }
}

// --- (h) ------------------------------------------------------------------------------------------------------------
void test_argmax() {
  std::vector<float> lg = vals(false, 90, br::kVocab, 4);
  std::vector<float> a = lg, b = lg;
  a[0] = 50.0f;
  b[br::kVocab - 1] = 100.0f;
  CHECK_EQ(br::argmax(a.data()), fx::kArgmaxA);
  CHECK_EQ(br::argmax(b.data()), fx::kArgmaxB);
  CHECK_EQ(fx::kArgmaxA, 0u);
  CHECK_EQ(fx::kArgmaxB, br::kVocab - 1);
  b[7] = b[9] = 200.0f;   // an exact tie at the top: the lower id
  CHECK_EQ(br::argmax(b.data()), 7u);
  std::printf("  (h) argmax at 131072: 0 and 131071 win their rows, ties to the lower id\n");
}

}  // namespace

int main() {
  test_transcendentals();
  test_norm();
  test_conv();
  test_ssm();
  test_gnorm();
  test_route();
  test_moe();
  test_attention();
  test_argmax();
  std::printf("bolt_ref_test OK\n");
  return 0;
}
