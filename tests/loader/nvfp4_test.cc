// Spec 26b Task 2 (host): NVFP4 decoded exactly (loader/nvfp4.h) and the two quantisers from (W', g). No device.
//
//   1. Every (e2m1 code, e4m3 code) product: all 16 x 256 - the two NaN scale codes (0x7F, 0xFF) aside, 16 x 254 - is
//      exact in bf16 and in f16 (the fp64 product round-trips both), |W'| in [2^-10, 2688]; the e4m3 table against the
//      format's formula (bias 7, subnormals m x 2^-9) and e2m1's eight magnitudes.
//   2. decode_nvfp4: the nibble order (element 2i low, 2i + 1 high), one scale per 16 k; a NaN scale refused naming the
//      tensor and the element ([row][block]); K not a multiple of 16 refused.
//   3. f64_to_f16_rne: ONE rounding - equal to the fp32 path on every double that is a float, and right where the two
//      roundings of double -> float -> half differ (1 + 2^-11 + 2^-40 -> 1 + 2^-10); ties to even, subnormals, overflow.
//   4. rtn_int4_g64_g: rtn.h's rule over g x W' in fp64 (checked against a direct fp64 evaluation); at g = 1 on these bf16
//      rows it equals rtn_int4_g64 (26a ruling 9's agreement, here on 64 x 256 random rows).
//   5. quantise_int8_tiled_g: the codes are spec 9's of W' whatever g (Review Focus 1: the int8 arm's integer codes do not
//      depend on g), the scales fp32(g x s'_r).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "loader/lm_head_int8.h"
#include "loader/nvfp4.h"
#include "loader/rtn.h"

namespace {

bool throws_with(const std::function<void()>& f, const std::string& needle) {
  try {
    f();
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) {
      std::printf("  refused: %s\n", e.what());
      return true;
    }
    std::fprintf(stderr, "  threw, but without '%s': %s\n", needle.c_str(), e.what());
    return false;
  }
  std::fprintf(stderr, "  did not throw (wanted '%s')\n", needle.c_str());
  return false;
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 7) {}
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return uint32_t(s >> 16);
  }
};

}  // namespace

int main() {
  // --- 1. the 16 x 256 products -------------------------------------------------------------------------------------
  {
    const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint32_t c = 0; c < 16; ++c) CHECK(loader::e2m1_value(uint8_t(c)) == ((c & 8) ? -mags[c & 7] : mags[c & 7]));
    size_t finite = 0, nan = 0;
    double lo = INFINITY, hi = 0;
    for (uint32_t sc = 0; sc < 256; ++sc) {
      const uint8_t s = uint8_t(sc);
      if (loader::e4m3_is_nan(s)) {
        CHECK(sc == 0x7F || sc == 0xFF);
        CHECK(std::isnan(loader::e4m3_value(s)));
        ++nan;
        continue;
      }
      const uint32_t e = (sc >> 3) & 0xF, m = sc & 7;
      const double want = ((sc & 0x80) ? -1.0 : 1.0) * (e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, int(e) - 7));
      CHECK(double(loader::e4m3_value(s)) == want);
      for (uint32_t c = 0; c < 16; ++c) {
        const double p = want * double(loader::e2m1_value(uint8_t(c)));
        CHECK(double(common::bf16_to_f32(common::f32_to_bf16(float(p)))) == p);   // exact in bf16
        CHECK(double(common::f16_to_f32(loader::f64_to_f16_rne(p))) == p);        // and in f16
        if (p != 0.0) {
          lo = std::fmin(lo, std::fabs(p));
          hi = std::fmax(hi, std::fabs(p));
        }
        ++finite;
      }
    }
    CHECK(nan == 2 && finite == 16 * 254);
    CHECK(lo == std::ldexp(1.0, -10) && hi == 2688.0);
    std::printf("1. 16 x 254 finite (code, scale) products exact in bf16 and f16, |W'| in [2^-10, %g]; the 2 NaN scale "
                "codes recognised\n", hi);
  }

  // --- 2. decode ------------------------------------------------------------------------------------------------------
  {
    const uint32_t N = 2, K = 32;
    std::vector<uint8_t> w(N * K / 2), s(N * K / 16);
    for (uint32_t i = 0; i < w.size(); ++i) w[i] = uint8_t(i * 37 + 11);
    for (uint32_t i = 0; i < s.size(); ++i) s[i] = uint8_t(0x30 + i * 5);
    const std::vector<uint16_t> wp = loader::decode_nvfp4(w.data(), s.data(), K, N, "t");
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k) {
        const uint8_t b = w[n * K / 2 + k / 2];
        const uint8_t code = (k % 2) ? uint8_t(b >> 4) : uint8_t(b & 0xF);
        const float want = loader::e4m3_value(s[n * (K / 16) + k / 16]) * loader::e2m1_value(code);
        CHECK(common::bf16_to_f32(wp[n * K + k]) == want);
      }
    s[1 * 2 + 1] = 0x7F;
    CHECK(throws_with([&] { (void)loader::decode_nvfp4(w.data(), s.data(), K, N, "layers.3.mlp.up_proj"); },
                      "layers.3.mlp.up_proj: weight_scale[1][1] is an e4m3 NaN code (0x7f)"));
    s[1 * 2 + 1] = 0xFF;
    CHECK(throws_with([&] { (void)loader::decode_nvfp4(w.data(), s.data(), K, N, "fc"); }, "weight_scale[1][1]"));
    CHECK(throws_with([&] { (void)loader::decode_nvfp4(w.data(), s.data(), 24, 1, "fc"); }, "not a multiple of 16"));
    std::printf("2. decode: element 2i in the low nibble, one e4m3 scale per 16 k; NaN scales refused by tensor and element\n");
  }

  // --- 3. f64 -> f16 in one rounding ----------------------------------------------------------------------------------
  {
    Rng r(3);
    for (int i = 0; i < 200000; ++i) {
      uint32_t u = r.next() ^ (r.next() << 16);
      float f;
      std::memcpy(&f, &u, 4);
      if (!std::isfinite(f) || std::fabs(f) >= 65504.0f) continue;
      CHECK(loader::f64_to_f16_rne(double(f)) == common::kv8::f32_to_f16_rne(f));
    }
    const double x = 1.0 + std::ldexp(1.0, -11) + std::ldexp(1.0, -40);   // just above a half-ulp tie
    CHECK(common::kv8::f32_to_f16_rne(float(x)) == 0x3C00);                 // via fp32: the tie, to even (1.0)
    CHECK(loader::f64_to_f16_rne(x) == 0x3C01);                             // once: up
    CHECK(loader::f64_to_f16_rne(1.0 + std::ldexp(1.0, -11)) == 0x3C00);    // an exact tie: to even
    CHECK(loader::f64_to_f16_rne(1.0 + 3 * std::ldexp(1.0, -11)) == 0x3C02);
    CHECK(loader::f64_to_f16_rne(std::ldexp(1.0, -24)) == 0x0001 && loader::f64_to_f16_rne(std::ldexp(1.0, -26)) == 0);
    CHECK(loader::f64_to_f16_rne(3 * std::ldexp(1.0, -25)) == 0x0002);      // 1.5 x 2^-24: tie to even 2
    CHECK(loader::f64_to_f16_rne(65504.0) == 0x7BFF && loader::f64_to_f16_rne(65520.0) == 0x7C00);
    CHECK(loader::f64_to_f16_rne(-2.0) == 0xC000);
    std::printf("3. f64 -> f16: the fp32 path on 200k floats; one rounding where double -> float -> half takes two\n");
  }

  // --- 4. int4 g64 from g x W' in fp64 --------------------------------------------------------------------------------
  {
    const uint32_t N = 64, K = 256;
    Rng r(5);
    std::vector<uint16_t> wp(size_t(N) * K);
    for (uint16_t& v : wp) {
      const uint8_t sc = uint8_t(0x28 + r.next() % 0x20), code = uint8_t(r.next() & 0xF);
      v = common::f32_to_bf16(loader::e4m3_value(sc) * loader::e2m1_value(code));
    }
    const double g = 0.0023719f;
    std::vector<uint32_t> qw(size_t(K / 8) * N), qw1(qw.size()), qr(qw.size());
    std::vector<uint16_t> sc(size_t(K / 64) * N), sc1(sc.size()), sr(sc.size());
    loader::rtn_int4_g64_g(wp.data(), g, K, N, qw.data(), sc.data());
    size_t moved = 0;
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t grp = 0; grp < K / 64; ++grp) {
        double amax = 0;
        for (uint32_t k = grp * 64; k < grp * 64 + 64; ++k)
          amax = std::fmax(amax, std::fabs(g * double(common::bf16_to_f32(wp[size_t(n) * K + k]))));
        CHECK(sc[size_t(grp) * N + n] == loader::f64_to_f16_rne(2.0 * amax / 15.0));
        const double s = common::f16_to_f32(sc[size_t(grp) * N + n]);
        for (uint32_t k = grp * 64; k < grp * 64 + 64; ++k) {
          const double v = g * double(common::bf16_to_f32(wp[size_t(n) * K + k]));
          const uint32_t q = uint32_t(std::fmin(std::fmax(s > 0 ? std::nearbyint(v / s) + 8.0 : 8.0, 0.0), 15.0));
          CHECK(((qw[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 0xF) == q);
        }
      }
    // folding g first into bf16 (the shortcut spec 26 §3 rules out) moves codes
    std::vector<uint16_t> folded(wp.size());
    for (size_t i = 0; i < wp.size(); ++i) folded[i] = common::f32_to_bf16(float(g) * common::bf16_to_f32(wp[i]));
    loader::rtn_int4_g64(folded.data(), K, N, qr.data(), sr.data());
    for (size_t i = 0; i < qw.size(); ++i) moved += qw[i] != qr[i];
    // at g = 1 the fp64 rule is rtn.h's on these rows
    loader::rtn_int4_g64_g(wp.data(), 1.0, K, N, qw1.data(), sc1.data());
    loader::rtn_int4_g64(wp.data(), K, N, qr.data(), sr.data());
    CHECK(qw1 == qr && sc1 == sr);
    std::printf("4. int4 g64 from g x W' in fp64 = the rule evaluated directly; at g = 1 = rtn.h's; folding g into bf16 "
                "first moves %zu of %zu words\n", moved, qw.size());
  }

  // --- 5. int8 codes independent of g ---------------------------------------------------------------------------------
  {
    const uint32_t N = 32, K = 64;
    Rng r(9);
    std::vector<uint16_t> wp(size_t(N) * K);
    for (uint16_t& v : wp)
      v = common::f32_to_bf16(loader::e4m3_value(uint8_t(0x28 + r.next() % 0x20)) * loader::e2m1_value(uint8_t(r.next() & 0xF)));
    std::vector<int8_t> q0(wp.size()), q1(wp.size()), q2(wp.size());
    std::vector<float> s0(N), s1(N), s2(N);
    loader::quantise_int8_tiled(wp.data(), K, N, q0.data(), s0.data(), 1);
    loader::quantise_int8_tiled_g(wp.data(), 0.00379f, K, N, q1.data(), s1.data(), 1);
    loader::quantise_int8_tiled_g(wp.data(), 0.0007911f, K, N, q2.data(), s2.data(), 2);
    CHECK(q0 == q1 && q0 == q2);
    for (uint32_t n = 0; n < N; ++n) CHECK(s1[n] == 0.00379f * s0[n] && s2[n] == 0.0007911f * s0[n]);
    std::printf("5. int8: the codes of W' whatever g; the row scales fp32(g x s'_r)\n");
  }
  std::printf("nvfp4_test OK\n");
  return 0;
}
