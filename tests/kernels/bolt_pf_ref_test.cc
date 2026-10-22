// Spec 25d Task 1: tests/kernels/bolt_pf_ref.h (the Nemotron 3.5 Bolt prefill kernels' host twin) against 25c's decode twin
// (tests/kernels/bolt_ref.h - the per-row chains the prefill must apply), the algebra (the SSD form IS the recurrence), the
// reference's own chunk functions (tests/kernels/bolt_pf_fixture.h, tools/oracle/bolt_pf_fixture.py) and plan 25d's Review
// Focus 1-3. Host only; every measured bound is printed (the bars of the card tests and of spec 25 §16 are set from them).
//
//   (0) the twin's recurrence() is boltref::ssm_step BITWISE (all 128 heads, 5 rows): the decode chain the SSD is held to
//   (a) the conv over a chunk (Review Focus 1's seed / writeback): decode's conv_step row by row BITWISE at the real width
//       from a garbage ring - positions 0 .. 3 at a sequence start (C 1 / 2 / 3 / 4 at pos 0), a 181-row chunk at 0, 1, 2,
//       3 and 4097, a chunk split at 117 (the seed carried) - xc and the ring slots decode reads next; the fixture's torch
//       conv_seq (whole = [0, 117) + [117, 181)) within 1 ulp (SiLU's exp: Sleef vs MKL), the count printed
//   (b) the SSD algebra (Review Focus 2): in double the SSD form equals the sequential recurrence to 1e-12 (row-relative)
//       for sub-chunk lengths 1, 2, 63, 64, a partial 53 (181 = 64 + 64 + 53), two one-row sub-chunks (63, 2) and an
//       off-grid start (37, 300); the dt_bias -10.5 / -11.69 heads and A_log +4.41 / +8.81 (decay e^-82 / e^-6700 per unit
//       dt) included; in fp32 the twin's SSD against the twin's recurrence over 2048 rows (pos 0 and 37): y row-relative
//       (before the rounding), the bf16 flips, the state's cosine - every value finite
//   (c) the fixture: the twin's SSD at 64-row sub-chunks against ssm_chunk (torch's einsums) at 64 - y within a printed ulp
//       bound, the state row-relative; ssm_chunk at 128 (the training chunk) printed beside it
//   (d) absolute sub-chunks (Review Focus 1): the Mamba chain (seed, conv, dt, C B^T, SSD, gnorm) over [2085, 2385) as one
//       chunk = split at 2112 (2048 + 64) = split at 2176 / 2304 = the chunk rule at 64 / 128 / 2048 - out, y, the state
//       and the three ring columns decode reads next BITWISE; a one-row sub-chunk's state is decode's step BITWISE; a split
//       off the grid (2100) differs (printed - the walk never cuts there)
//   (e) the grouped MoE (Review Focus 3): nb_pf_sort's layout (every pair once, expert-major, ascending row within an
//       expert, padding NONE, the tile bound reached exactly by an adversarial route at C 2048 / E 496), gather, the twin's
//       grouped up / down and the ascending-id combine = boltref::moe per row BITWISE at E 240 / 496 over 181 rows
//   (f) the chunk rule: every chunk ends on the absolute 64-grid or at the prompt end, never past the block hook's end
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/bolt_pf_fixture.h"
#include "kernels/bolt_pf_ref.h"

namespace {

namespace br = boltref;
namespace bp = boltpf;
namespace pfx = bolt_pf_fixture;
using br::f32;
using br::rne;

constexpr uint32_t X = bp::kX, LD = bp::kInLd, IN = bp::kInner, NH = bp::kHeads;
const std::vector<uint32_t> kHeadsSel = {0, 1, 2, 3, 4, 5, 6, 77, 127};

int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
float fval(bool coarse, uint32_t t, uint32_t i, uint32_t seed, float amp = 1.0f) {
  return br::fixture_val(coarse, t, i, seed, 0.125f, amp);
}

// The SSD inputs of fixture case (b) (tools/oracle/bolt_pf_fixture.py), at `rows` rows (index-based: row t of any case
// is the same value): xc, in_proj's dt columns, dt_bias / A_log with the planted heads, A = the engine's -exp(A_log), D,
// the initial state.
struct SsdIn {
  uint32_t rows;
  std::vector<uint16_t> xc;
  std::vector<float> P, dt_bias, A_log, A, D, S0;
};
SsdIn ssd_inputs(uint32_t rows) {
  SsdIn s;
  s.rows = rows;
  s.xc.resize(size_t(rows) * X);
  for (size_t i = 0; i < s.xc.size(); ++i) s.xc[i] = rne(fval(false, 61, uint32_t(i), 12));
  s.P.assign(size_t(rows) * LD, 0.0f);
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t h = 0; h < NH; ++h) s.P[size_t(t) * LD + br::kDtOff + h] = fval(false, 62, t * NH + h, 12, 2.0f);
  s.dt_bias.resize(NH);
  s.A_log.resize(NH);
  s.D.resize(NH);
  s.A.resize(NH);
  for (uint32_t h = 0; h < NH; ++h) {
    s.dt_bias[h] = fval(false, 63, h, 12, 4.0f);
    s.A_log[h] = fval(false, 64, h, 12, 4.0f);
    s.D[h] = fval(false, 65, h, 12, 2.0f);
  }
  s.dt_bias[0] = -10.5f;
  s.dt_bias[1] = 8.5f;
  s.dt_bias[6] = -11.6875f;
  const float planted[4] = {-4.66f, 4.41f, 8.81f, -6.81f};
  for (uint32_t i = 0; i < 4; ++i) s.A_log[2 + i] = br::rf(planted[i]);
  for (uint32_t h = 0; h < NH; ++h) s.A[h] = -br::exp_t(s.A_log[h]);
  s.S0.resize(size_t(NH) * bp::kP * bp::kN);
  for (size_t i = 0; i < s.S0.size(); ++i) s.S0[i] = br::fixture_f32(66, uint32_t(i), 12);
  return s;
}

// --- (0) ----------------------------------------------------------------------------------------------------------
void test_recurrence_is_decode() {
  const uint32_t C = 5;
  SsdIn in = ssd_inputs(C);
  std::vector<float> slices(size_t(C) * br::kInProjN, 0.0f);
  for (uint32_t t = 0; t < C; ++t)
    std::copy(in.P.begin() + size_t(t) * LD, in.P.begin() + size_t(t) * LD + br::kInProjN, slices.begin() + size_t(t) * br::kInProjN);
  std::vector<float> s1 = in.S0, s2 = in.S0;
  std::vector<uint16_t> y1(size_t(C) * IN), y2(size_t(C) * IN);
  for (uint32_t t = 0; t < C; ++t) {   // decode: one M = 1 step a row
    const std::vector<uint16_t> yt = br::ssm_step(slices.data() + size_t(t) * br::kInProjN, 1, 1, in.xc.data() + size_t(t) * X,
                                                  in.A.data(), in.D.data(), in.dt_bias.data(), s1.data());
    std::copy(yt.begin(), yt.end(), y1.begin() + size_t(t) * IN);
  }
  bp::recurrence(in.P.data(), in.xc.data(), C, in.A.data(), in.D.data(), in.dt_bias.data(), s2.data(), y2.data());
  CHECK(y1 == y2);
  CHECK(std::memcmp(s1.data(), s2.data(), s1.size() * 4) == 0);
  std::printf("(0) recurrence(): boltref::ssm_step bitwise on all 128 heads over %u rows (y and the state)\n", C);
}

// --- (a) ----------------------------------------------------------------------------------------------------------
struct ConvIn {
  std::vector<float> P, cw, cb;
};
ConvIn conv_inputs(uint32_t rows, uint32_t raw_t, uint32_t seed) {
  ConvIn c;
  c.P.assign(size_t(rows) * LD, 0.0f);
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t ch = 0; ch < X; ++ch) c.P[size_t(t) * LD + br::kXbcOff + ch] = fval(false, raw_t, t * X + ch, seed, 2.0f);
  c.cw.resize(size_t(X) * 4);
  c.cb.resize(X);
  for (uint32_t i = 0; i < X * 4; ++i) c.cw[i] = fval(false, 21, i, seed, 0.5f);
  for (uint32_t i = 0; i < X; ++i) c.cb[i] = fval(false, 22, i, seed, 0.5f);
  return c;
}
std::vector<uint16_t> garbage_ring(uint32_t seed) {
  std::vector<uint16_t> r(size_t(br::kRing) * X);
  for (size_t i = 0; i < r.size(); ++i) r[i] = rne(fval(false, 77, uint32_t(i), seed, 3.0f));
  return r;
}
// Decode's conv_step one row at a time over in_proj's prefill rows [t0, t0 + C).
std::vector<uint16_t> conv_decode(const ConvIn& c, uint32_t t0, uint32_t C, std::vector<uint16_t>& ring, uint32_t pos) {
  std::vector<uint16_t> xc(size_t(C) * X);
  std::vector<float> slice(br::kInProjN, 0.0f);
  for (uint32_t t = 0; t < C; ++t) {
    std::copy(c.P.begin() + size_t(t0 + t) * LD, c.P.begin() + size_t(t0 + t) * LD + br::kInProjN, slice.begin());
    const std::vector<uint16_t> r = br::conv_step(slice.data(), 1, 1, ring.data(), c.cw.data(), c.cb.data(), pos + t);
    std::copy(r.begin(), r.end(), xc.begin() + size_t(t) * X);
  }
  return xc;
}
// The ring slots decode reads after position end: end - 3 .. end - 1 (those >= 0).
bool next_slots_equal(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, uint32_t end) {
  for (int p = int(end) - 3; p < int(end); ++p) {
    if (p < 0) continue;
    const size_t s = size_t(uint32_t(p) % br::kRing) * X;
    if (!std::equal(a.begin() + s, a.begin() + s + X, b.begin() + s)) return false;
  }
  return true;
}
void test_conv() {
  const ConvIn c = conv_inputs(181, 60, 11);
  struct Case {
    uint32_t pos, C;
  };
  for (const Case k : {Case{0, 1}, Case{0, 2}, Case{0, 3}, Case{0, 4}, Case{0, 181}, Case{1, 181}, Case{2, 181}, Case{3, 181},
                       Case{4097, 181}}) {
    std::vector<uint16_t> rd = garbage_ring(k.pos + 1), rp = rd;
    const std::vector<uint16_t> want = conv_decode(c, 0, k.C, rd, k.pos);
    const std::vector<uint16_t> sd = bp::seed(rp.data(), k.pos);
    const std::vector<uint16_t> got = bp::conv_chunk(c.P.data(), k.C, sd.data(), c.cw.data(), c.cb.data(), rp.data(), k.pos);
    CHECK(got == want);
    CHECK(next_slots_equal(rp, rd, k.pos + k.C));
  }
  // a split at 117: the second chunk's seed is the first's writeback
  std::vector<uint16_t> ra = garbage_ring(99), rb = ra;
  const std::vector<uint16_t> whole = bp::conv_chunk(c.P.data(), 181, bp::seed(ra.data(), 0).data(), c.cw.data(), c.cb.data(),
                                                     ra.data(), 0);
  std::vector<uint16_t> parts = bp::conv_chunk(c.P.data(), 117, bp::seed(rb.data(), 0).data(), c.cw.data(), c.cb.data(),
                                               rb.data(), 0);
  const std::vector<uint16_t> sd2 = bp::seed(rb.data(), 117);
  const std::vector<uint16_t> p2 = bp::conv_chunk(c.P.data() + size_t(117) * LD, 64, sd2.data(), c.cw.data(), c.cb.data(),
                                                  rb.data(), 117);
  parts.insert(parts.end(), p2.begin(), p2.end());
  CHECK(parts == whole);
  CHECK(next_slots_equal(ra, rb, 181));
  // against torch's conv_seq (the fixture: whole = [0, 117) + [117, 181) carried)
  const std::vector<uint16_t> fxc = br::hex16(pfx::kConvXc);
  const uint32_t nr = sizeof(pfx::kConvRows) / sizeof(pfx::kConvRows[0]), nc = (X + pfx::kConvStride - 1) / pfx::kConvStride;
  CHECK(fxc.size() == size_t(nr) * nc);
  size_t diff = 0;
  int worst = 0;
  for (uint32_t r = 0; r < nr; ++r)
    for (uint32_t j = 0; j < nc; ++j) {
      const int u = ulps(whole[size_t(pfx::kConvRows[r]) * X + j * pfx::kConvStride], fxc[size_t(r) * nc + j]);
      diff += u != 0;
      worst = std::max(worst, u);
    }
  CHECK(worst <= 1);
  std::printf("(a) conv over a chunk: decode's conv_step bitwise row by row (C 1 / 2 / 3 / 4 at pos 0; 181 rows at 0 / 1 / 2 / "
              "3 / 4097 from a garbage ring; the next three ring columns); split at 117 = whole bitwise; torch's conv_seq: "
              "%zu of %zu values differ, worst %d ulp (SiLU's exp)\n", diff, size_t(nr) * nc, worst);
}

// --- (b) ----------------------------------------------------------------------------------------------------------
// Two [C][heads][64] double y's: {head-relative (per head: max over its rows of |a - b|, over max |b| of the head's whole
// chunk), row-relative (per (t, h): max |a - b| / max |b| of that row)}, the worst of each. A row whose y is far below its
// head's scale (a decayed state: y ~ C B dt x + D x) loses digits to C . B's own cancellation in BOTH evaluations, so the
// row-relative figure is printed and held to 1e-10; the head-relative one is the 1e-12 bar.
std::pair<double, double> yrel(const std::vector<double>& a, const std::vector<double>& b, uint32_t C, uint32_t nh) {
  double worst_h = 0.0, worst_r = 0.0;
  for (uint32_t h = 0; h < nh; ++h) {
    double mh = 0.0, dh = 0.0;
    for (uint32_t t = 0; t < C; ++t) {
      double m = 0.0, d = 0.0;
      for (uint32_t p = 0; p < bp::kP; ++p) {
        const size_t i = (size_t(t) * nh + h) * bp::kP + p;
        CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
        m = std::max(m, std::fabs(b[i]));
        d = std::max(d, std::fabs(a[i] - b[i]));
      }
      if (m > 0.0) worst_r = std::max(worst_r, d / m);
      else CHECK(d == 0.0);
      mh = std::max(mh, m);
      dh = std::max(dh, d);
    }
    if (mh > 0.0) worst_h = std::max(worst_h, dh / mh);
  }
  return {worst_h, worst_r};
}
void test_algebra() {
  const SsdIn in = ssd_inputs(2048 + 64);
  // (b.1) double, every sub-chunk shape
  struct Case {
    uint32_t pos, C;
    const char* what;
  };
  double worst_y = 0.0, worst_s = 0.0, worst_cy = 0.0, worst_r = 0.0;
  for (const Case k : {Case{0, 1, "1"}, Case{0, 2, "2"}, Case{0, 63, "63"}, Case{0, 64, "64"}, Case{0, 181, "64+64+53"},
                       Case{63, 2, "1+1"}, Case{37, 300, "27+64x4+17"}}) {
    const bp::DtCum dc = bp::dt_cum(in.P.data(), k.C, k.pos, in.A.data(), in.dt_bias.data());
    const uint32_t nh = k.C <= 64 ? NH : 16;   // every head of the short cases; group 0 (the planted heads 0..6) after
    std::vector<double> s1(size_t(nh) * bp::kP * bp::kN), s2;
    for (size_t i = 0; i < s1.size(); ++i) s1[i] = in.S0[i];
    s2 = s1;
    std::vector<double> s3 = s1;
    const std::vector<double> yr = bp::recurrence_f64(in.xc.data(), k.C, dc.dt, in.A.data(), in.D.data(), s2, 0, nh);
    const std::vector<double> ys = bp::ssd_f64(in.xc.data(), k.C, k.pos, dc.dt, in.A.data(), in.D.data(), s1, 0, nh, true);
    const std::vector<double> yc = bp::ssd_f64(in.xc.data(), k.C, k.pos, dc.dt, in.A.data(), in.D.data(), s3, 0, nh, false);
    auto srel = [&](const std::vector<double>& sa) {
      double rs = 0.0;
      for (uint32_t h = 0; h < nh; ++h) {
        double m = 0.0, d = 0.0;
        for (size_t i = size_t(h) * bp::kP * bp::kN; i < size_t(h + 1) * bp::kP * bp::kN; ++i) {
          CHECK(std::isfinite(sa[i]));
          m = std::max(m, std::fabs(s2[i]));
          d = std::max(d, std::fabs(sa[i] - s2[i]));
        }
        if (m > 0.0) rs = std::max(rs, d / m);
      }
      return rs;
    };
    const std::pair<double, double> ry = yrel(ys, yr, k.C, nh), cy = yrel(yc, yr, k.C, nh);
    const double rs = srel(s1), cs = srel(s3);
    std::printf("(b) double, sub-chunks %-11s (pos %u, C %u): SSD vs the recurrence - y %.3g head-relative (%.3g row-relative), "
                "state %.3g; through exp(cum_t - cum_s): y %.3g (%.3g), state %.3g\n", k.what, k.pos, k.C, ry.first,
                ry.second, rs, cy.first, cy.second, cs);
    CHECK(ry.first <= 1e-12 && rs <= 1e-12 && ry.second <= 1e-10);
    CHECK(cy.first <= 1e-12 && cs <= 1e-12 && cy.second <= 1e-10);
    worst_y = std::max(worst_y, ry.first);
    worst_r = std::max(worst_r, std::max(ry.second, cy.second));
    worst_s = std::max(worst_s, std::max(rs, cs));
    worst_cy = std::max(worst_cy, cy.first);
  }
  std::printf("(b) double: the SSD form IS the recurrence - worst y %.3g (segment sums direct) / %.3g (exp of cum_t - cum_s, "
              "the kernels' form) head-relative (bar 1e-12), %.3g row-relative (bar 1e-10), state %.3g (bar 1e-12)\n",
              worst_y, worst_cy, worst_r, worst_s);
  // (b.2) fp32: the twin's SSD (C B^T, M, the scan) against the twin's recurrence (decode's chain), 2048 rows
  for (const uint32_t pos : {0u, 37u}) {
    const uint32_t C = 2048;
    const bp::DtCum dc = bp::dt_cum(in.P.data(), C, pos, in.A.data(), in.dt_bias.data());
    const std::vector<float> cbm = bp::cb(in.xc.data(), C, pos);
    std::vector<float> sa = in.S0, sb = in.S0, ya(size_t(C) * IN, 0.0f), yb(size_t(C) * IN, 0.0f);
    std::vector<uint16_t> y16a(size_t(C) * IN, 0), y16b(size_t(C) * IN, 0);
    double hrel = 0.0, rrel = 0.0, scos = 1.0, srel = 0.0;
    size_t flips = 0, vals = 0;
    for (uint32_t h : kHeadsSel) {
      bp::ssd(in.xc.data(), C, pos, dc, cbm.data(), in.D.data(), sa.data(), y16a.data(), h, h + 1, nullptr, ya.data());
      bp::recurrence(in.P.data(), in.xc.data(), C, in.A.data(), in.D.data(), in.dt_bias.data(), sb.data(), y16b.data(), h,
                     h + 1, LD, yb.data());
      double mh = 0.0, dh = 0.0;
      for (uint32_t t = 0; t < C; ++t) {
        double m = 0.0, d = 0.0;
        for (uint32_t p = 0; p < bp::kP; ++p) {
          const size_t i = size_t(t) * IN + size_t(h) * bp::kP + p;
          CHECK(std::isfinite(ya[i]) && std::isfinite(yb[i]));
          m = std::max(m, double(std::fabs(yb[i])));
          d = std::max(d, double(std::fabs(ya[i] - yb[i])));
          flips += y16a[i] != y16b[i];
          ++vals;
        }
        if (m > 0.0) rrel = std::max(rrel, d / m);
        mh = std::max(mh, m);
        dh = std::max(dh, d);
      }
      if (mh > 0.0) hrel = std::max(hrel, dh / mh);
      double dot = 0.0, na = 0.0, nb = 0.0, m = 0.0, d = 0.0;
      for (size_t i = size_t(h) * bp::kP * bp::kN; i < size_t(h + 1) * bp::kP * bp::kN; ++i) {
        CHECK(std::isfinite(sa[i]));
        dot += double(sa[i]) * sb[i];
        na += double(sa[i]) * sa[i];
        nb += double(sb[i]) * sb[i];
        m = std::max(m, double(std::fabs(sb[i])));
        d = std::max(d, double(std::fabs(sa[i] - sb[i])));
      }
      scos = std::min(scos, na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 1.0);
      if (m > 0.0) srel = std::max(srel, d / m);
    }
    std::printf("(b) fp32, 2048 rows at pos %u (%u sub-chunks), heads 0-6 / 77 / 127: y %.3g head-relative before the "
                "rounding (bar 1e-4, PROPOSED; %.3g row-relative: rows far below their head's scale lose digits to C . "
                "B), %zu of %zu bf16 values differ; the state after 2048 rows: cosine %.9f (bar 0.99999), row-relative %.3g\n",
                pos, bp::n_sub(pos, C), hrel, rrel, flips, vals, scos, srel);
    CHECK(hrel <= 1e-4);
    CHECK(scos >= 0.99999);
  }
}

// --- (c) ----------------------------------------------------------------------------------------------------------
void test_fixture() {
  const uint32_t C = pfx::kT;
  const SsdIn in = ssd_inputs(C);
  CHECK(br::hex32(pfx::kSsdA) == [&] {
    std::vector<uint32_t> v;
    for (float a : in.A) v.push_back(br::as_u32(a));
    return v;
  }());
  const bp::DtCum dc = bp::dt_cum(in.P.data(), C, 0, in.A.data(), in.dt_bias.data());
  const std::vector<float> cbm = bp::cb(in.xc.data(), C, 0);
  std::vector<float> S = in.S0;
  std::vector<uint16_t> y(size_t(C) * IN, 0);
  for (uint32_t h : pfx::kSsdHeads) bp::ssd(in.xc.data(), C, 0, dc, cbm.data(), in.D.data(), S.data(), y.data(), h, h + 1);
  const uint32_t nr = sizeof(pfx::kSsdRows) / 4, nh = sizeof(pfx::kSsdHeads) / 4, np = sizeof(pfx::kSsdP) / 4;
  for (const uint32_t chunk : {64u, 128u}) {
    const std::vector<uint16_t> fy = br::hex16(chunk == 64 ? pfx::kSsdY64 : pfx::kSsdY128);
    const std::vector<uint32_t> fs = br::hex32(chunk == 64 ? pfx::kSsdState64 : pfx::kSsdState128);
    CHECK(fy.size() == size_t(nr) * nh * bp::kP && fs.size() == size_t(nh) * np * bp::kN);
    size_t diff = 0;
    int worst = 0;
    for (uint32_t r = 0; r < nr; ++r)
      for (uint32_t j = 0; j < nh; ++j)
        for (uint32_t p = 0; p < bp::kP; ++p) {
          const int u = ulps(y[size_t(pfx::kSsdRows[r]) * IN + size_t(pfx::kSsdHeads[j]) * bp::kP + p],
                             fy[(size_t(r) * nh + j) * bp::kP + p]);
          diff += u != 0;
          worst = std::max(worst, u);
        }
    double srel = 0.0;
    for (uint32_t j = 0; j < nh; ++j)
      for (uint32_t q = 0; q < np; ++q) {
        double m = 0.0, d = 0.0;
        for (uint32_t n = 0; n < bp::kN; ++n) {
          const float want = br::as_f32(fs[(size_t(j) * np + q) * bp::kN + n]);
          const float got = S[(size_t(pfx::kSsdHeads[j]) * bp::kP + pfx::kSsdP[q]) * bp::kN + n];
          m = std::max(m, double(std::fabs(want)));
          d = std::max(d, double(std::fabs(got - want)));
        }
        if (m > 0) srel = std::max(srel, d / m);
      }
    std::printf("(c) the twin's SSD (64-row absolute sub-chunks) vs ssm_chunk at %u: y %zu of %zu differ, worst %d ulp; the "
                "state row-relative %.3g\n", chunk, diff, size_t(nr) * nh * bp::kP, worst, srel);
    if (chunk == 64) {   // the same sub-chunks: only the sums' orders and the transcendentals differ
      CHECK(worst <= 2);
      CHECK(srel <= 1e-5);
    }
  }
}

// --- (d) ----------------------------------------------------------------------------------------------------------
// The Mamba chain's inputs over a span of positions (row t of P is position base + t) and its weights.
struct Chain {
  uint32_t base;
  std::vector<float> P, cw, cb, A, D, dt_bias, gw;
};
Chain chain_inputs(uint32_t base, uint32_t rows) {
  Chain c;
  c.base = base;
  c.P.assign(size_t(rows) * LD, 0.0f);
  for (uint32_t t = 0; t < rows; ++t) {
    float* r = c.P.data() + size_t(t) * LD;
    for (uint32_t i = 0; i < IN; ++i) r[i] = fval(false, 80, t * IN + i, 13, 4.0f);                     // z
    for (uint32_t i = 0; i < X; ++i) r[br::kXbcOff + i] = fval(false, 81, t * X + i, 13, 2.0f);         // xBC
    for (uint32_t h = 0; h < NH; ++h) r[br::kDtOff + h] = fval(false, 82, t * NH + h, 13, 2.0f);        // dt
  }
  c.cw.resize(size_t(X) * 4);
  c.cb.resize(X);
  for (uint32_t i = 0; i < X * 4; ++i) c.cw[i] = fval(false, 83, i, 13, 0.5f);
  for (uint32_t i = 0; i < X; ++i) c.cb[i] = fval(false, 84, i, 13, 0.5f);
  const SsdIn s = ssd_inputs(1);
  c.A = s.A;
  c.D = s.D;
  c.dt_bias = s.dt_bias;
  c.gw.resize(IN);
  for (uint32_t i = 0; i < IN; ++i) c.gw[i] = br::rf(1.0f + fval(false, 85, i, 13, 0.5f));
  return c;
}
struct ChainState {
  std::vector<uint16_t> ring;
  std::vector<float> S;
  std::vector<uint16_t> y, out;   // [rows][8192] each, rows of every chunk in order
};
ChainState chain_state0() {
  ChainState s;
  s.ring = garbage_ring(5);
  s.S.resize(size_t(NH) * bp::kP * bp::kN);
  for (size_t i = 0; i < s.S.size(); ++i) s.S[i] = br::fixture_f32(86, uint32_t(i), 13, 0.5f);
  return s;
}
// One chunk of C rows at position pos through the chain (heads 0..7 and 77 in the SSD; the others' y stay 0).
void chain_chunk(const Chain& c, ChainState& s, uint32_t pos, uint32_t C) {
  const float* P = c.P.data() + size_t(pos - c.base) * LD;
  const std::vector<uint16_t> sd = bp::seed(s.ring.data(), pos);
  const std::vector<uint16_t> xc = bp::conv_chunk(P, C, sd.data(), c.cw.data(), c.cb.data(), s.ring.data(), pos);
  const bp::DtCum dc = bp::dt_cum(P, C, pos, c.A.data(), c.dt_bias.data());
  const std::vector<float> cbm = bp::cb(xc.data(), C, pos);
  std::vector<uint16_t> y(size_t(C) * IN, 0);
  bp::ssd(xc.data(), C, pos, dc, cbm.data(), c.D.data(), s.S.data(), y.data(), 0, 8);
  bp::ssd(xc.data(), C, pos, dc, cbm.data(), c.D.data(), s.S.data(), y.data(), 77, 78);
  const std::vector<uint16_t> o = bp::gnorm_chunk(P, C, y.data(), c.gw.data());
  s.y.insert(s.y.end(), y.begin(), y.end());
  s.out.insert(s.out.end(), o.begin(), o.end());
}
void test_absolute() {
  const uint32_t base = 2048 + 37, L = 300;
  const Chain c = chain_inputs(base, L);
  auto run = [&](const std::vector<uint32_t>& cuts) {   // cut positions (absolute), the prompt [base, base + L)
    ChainState s = chain_state0();
    uint32_t p = base;
    for (uint32_t e : cuts) {
      chain_chunk(c, s, p, e - p);
      p = e;
    }
    chain_chunk(c, s, p, base + L - p);
    return s;
  };
  auto same = [&](const ChainState& a, const ChainState& b) {
    return a.y == b.y && a.out == b.out && std::memcmp(a.S.data(), b.S.data(), a.S.size() * 4) == 0 &&
           next_slots_equal(a.ring, b.ring, base + L);
  };
  const ChainState whole = run({});
  CHECK(same(whole, run({2048 + 64})));
  CHECK(same(whole, run({2048 + 128, 2048 + 256})));
  for (const uint32_t chunk : {64u, 128u, 2048u}) {   // the walk's own cuts
    std::vector<uint32_t> cuts;
    for (uint32_t p = base; p < base + L;) {
      p += bp::prefill_chunk_rows(p, base + L - p, chunk, false);
      if (p < base + L) cuts.push_back(p);
    }
    CHECK(same(whole, run(cuts)));
  }
  const ChainState off = run({2048 + 52});
  size_t diff = 0;
  for (size_t i = 0; i < whole.out.size(); ++i) diff += whole.out[i] != off.out[i];
  std::printf("(d) absolute sub-chunks: [2085, 2385) whole = split at 2112 = at 2176 / 2304 = the chunk rule at 64 / 128 / "
              "2048 - out, y, the state and the next ring columns bitwise; a split off the grid at 2100 changes %zu of %zu "
              "outputs (the walk never cuts there)\n", diff, whole.out.size());
  // a one-row sub-chunk's state update is decode's step, bitwise
  const SsdIn in = ssd_inputs(2);
  const bp::DtCum dc = bp::dt_cum(in.P.data(), 1, 63, in.A.data(), in.dt_bias.data());
  const std::vector<float> cbm = bp::cb(in.xc.data(), 1, 63);
  std::vector<float> sa = in.S0, sb = in.S0;
  std::vector<uint16_t> ya(IN, 0), yb(IN, 0);
  bp::ssd(in.xc.data(), 1, 63, dc, cbm.data(), in.D.data(), sa.data(), ya.data());
  bp::recurrence(in.P.data(), in.xc.data(), 1, in.A.data(), in.D.data(), in.dt_bias.data(), sb.data(), yb.data());
  CHECK(std::memcmp(sa.data(), sb.data(), sa.size() * 4) == 0);
  size_t yd = 0;
  for (uint32_t i = 0; i < IN; ++i) yd += ya[i] != yb[i];
  std::printf("(d) a one-row sub-chunk (pos 63): the state bitwise decode's step on all 128 heads; y %zu of %u differ by its "
              "other order (Yd + Yo)\n", yd, IN);
}

// --- (e) ----------------------------------------------------------------------------------------------------------
std::vector<uint32_t> blocks_of(uint32_t E, uint32_t I, uint32_t seed) {
  const size_t bw = br::expert_block_words(I);
  std::vector<uint32_t> b(size_t(E) * bw);
  for (size_t i = 0; i < b.size(); ++i) {
    const size_t within = (i % bw) % 136;   // a tile: 128 nibble words, then 8 words of 16 f16 scales
    if (within < 128) {
      b[i] = br::fixture_hash(90, uint32_t(i), seed);
    } else {
      auto sc = [&](uint32_t h) {
        const float s = (1.0f + float((h >> 4) % 8) / 8.0f) * std::ldexp(1.0f, -6) * ((h & 1) ? -1.0f : 1.0f);
        return uint32_t(common::f32_to_f16(s));
      };
      b[i] = sc(br::fixture_hash(91, uint32_t(i), seed)) | (sc(br::fixture_hash(92, uint32_t(i), seed)) << 16);
    }
  }
  return b;
}
std::vector<uint32_t> routes_of(uint32_t C, uint32_t E, uint32_t seed) {
  std::vector<uint32_t> rt(size_t(C) * br::kWords);
  std::vector<float> lg(E), bias(E);
  for (uint32_t e = 0; e < E; ++e) bias[e] = 4.5f + br::fixture_f32(93, e, seed) * 0.04f;
  for (uint32_t t = 0; t < C; ++t) {
    for (uint32_t e = 0; e < E; ++e) lg[e] = br::fixture_f32(94, t * E + e, seed, 4.0f);
    const br::Route r = br::route(lg.data(), bias.data(), E);
    std::copy(r.row, r.row + br::kWords, rt.begin() + size_t(t) * br::kWords);
  }
  return rt;
}
void check_sorted(const bp::Sorted& s, const uint32_t* route, uint32_t C, uint32_t E) {
  CHECK(s.tiles_used() <= bp::tmax(C, E));
  std::vector<uint32_t> seen(size_t(s.rows_used()), 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t k = 0; k < br::kTopK; ++k) {
      const uint32_t r = s.pair_row[size_t(t) * br::kTopK + k];
      CHECK(r < s.rows_used() && s.row_tok[r] == t && seen[r]++ == 0);
      CHECK(bp::row_expert(s, r) == std::min(route[size_t(t) * br::kWords + br::kIds + k], E - 1));
    }
  uint32_t used = 0;
  for (uint32_t r = 0; r < s.rows_used(); ++r) {
    if (s.row_tok[r] == bp::kNone) continue;
    ++used;
    if (r % bp::kTm && s.row_tok[r - 1] != bp::kNone && bp::row_expert(s, r - 1) == bp::row_expert(s, r))
      CHECK(s.row_tok[r - 1] < s.row_tok[r]);   // stable: ascending row inside an expert
  }
  CHECK(used == C * br::kTopK);
  for (uint32_t i = s.tiles_used(); i < bp::tmax(C, E); ++i) CHECK(s.tiles[2 * i] == bp::kNone);
}
void test_moe() {
  const uint32_t I = 128, C = 181;
  for (const uint32_t E : {240u, 496u}) {
    const std::vector<uint32_t> rt = routes_of(C, E, E);
    const std::vector<uint32_t> blocks = blocks_of(E, I, E);
    std::vector<uint16_t> lat(size_t(C) * br::kLatent);
    for (size_t i = 0; i < lat.size(); ++i) lat[i] = rne(fval(false, 95, uint32_t(i), E));
    const bp::Sorted s = bp::sort(rt.data(), C, E);
    check_sorted(s, rt.data(), C, E);
    const std::vector<uint16_t> xg = bp::gather(s, lat.data());
    const std::vector<uint16_t> a = bp::grouped_up(s, xg.data(), blocks.data(), I);
    const std::vector<uint16_t> yd = bp::grouped_down(s, a.data(), blocks.data(), I);
    const std::vector<uint16_t> r = bp::combine(rt.data(), s, yd.data(), C);
    for (uint32_t t = 0; t < C; ++t) {
      br::Route ro;
      for (uint32_t k = 0; k < br::kTopK; ++k) {
        ro.ids[k] = rt[size_t(t) * br::kWords + br::kIds + k];
        ro.w[k] = br::as_f32(rt[size_t(t) * br::kWords + br::kW + k]);
      }
      const br::MoeOut m = br::moe(ro, lat.data() + size_t(t) * br::kLatent, blocks.data(), I);
      CHECK(std::equal(m.r.begin(), m.r.end(), r.begin() + size_t(t) * br::kLatent));
    }
    std::printf("(e) grouped MoE at E %u (I %u in the twin), %u rows: %u tiles of %u, every pair once, stable; the latent r "
                "bitwise boltref::moe on every row\n", E, I, C, s.tiles_used(), bp::tmax(C, E));
  }
  // the tile bound reached: 494 experts at 33 rows, two at 1065 (slot-major runs never repeat an expert in a row)
  const uint32_t CB = 2048, EB = 496;
  std::vector<uint32_t> list;
  for (uint32_t e = 0; e < EB; ++e) list.insert(list.end(), e < 494 ? 33u : 1065u, e);
  CHECK(list.size() == size_t(CB) * br::kTopK);
  std::vector<uint32_t> rt(size_t(CB) * br::kWords, 0);
  for (uint32_t j = 0; j < br::kTopK; ++j)
    for (uint32_t t = 0; t < CB; ++t) rt[size_t(t) * br::kWords + j] = list[size_t(j) * CB + t];
  const bp::Sorted s = bp::sort(rt.data(), CB, EB);
  check_sorted(s, rt.data(), CB, EB);
  CHECK(s.tiles_used() == bp::tmax(CB, EB));
  std::printf("(e) the tile bound at C 2048, E 496: %u tiles used = floor((9 C + 31 E) / 32) = %u\n", s.tiles_used(),
              bp::tmax(CB, EB));
}

// --- (f) ----------------------------------------------------------------------------------------------------------
void test_chunk_rule() {
  CHECK(bp::prefill_chunk_rows(37, 5000, 2048, false) == 2048 - 37);
  CHECK(bp::prefill_chunk_rows(0, 5000, 2048, false) == 2048);
  CHECK(bp::prefill_chunk_rows(37, 100, 2048, false) == 100);
  CHECK(bp::prefill_chunk_rows(63, 5000, 64, false) == 1);
  CHECK(bp::prefill_chunk_rows(2085, 5000, 2048, true) == 4096 - 2085);
  size_t n = 0;
  for (uint32_t pos = 0; pos < 300; pos += 7)
    for (uint32_t chunk = 64; chunk <= 2048; chunk += 64)
      for (const size_t rem : {size_t(1), size_t(63), size_t(64), size_t(65), size_t(2047), size_t(5000)})
        for (const bool hooked : {false, true}) {
          const uint32_t c = bp::prefill_chunk_rows(pos, rem, chunk, hooked);
          CHECK(c > 0 && c <= chunk && c <= rem);
          CHECK(c == rem || (pos + c) % bp::kSub == 0);
          if (hooked) CHECK((pos % 2048) + c <= 2048);
          ++n;
        }
  std::printf("(f) the chunk rule: %zu (pos, chunk, remaining, hook) cases end on the 64-grid or at the prompt end\n", n);
}

}  // namespace

int main() {
  test_recurrence_is_decode();
  test_conv();
  test_algebra();
  test_fixture();
  test_absolute();
  test_moe();
  test_chunk_rule();
  std::printf("bolt_pf_ref_test OK\n");
  return 0;
}
