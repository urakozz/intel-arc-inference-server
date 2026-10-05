// kv8_test - spec 12b: the rotkv scheme's host arithmetic (src/common/kv8.h), no device.
//
//   1. the signs are the 12a probe's (hadamard(256, 0) in tools/oracle/kv_int8_probe.py);
//   2. f16f is exact on every finite binary16; f32_to_f16_rne is IEEE round-to-nearest-even
//      with subnormals (against the compiler's _Float16 where it has one, and by the
//      round trip on every finite binary16 always);
//   3. rotate / unrotate are x R and y R^T for R = H diag(s) / 16 (fp64 matrix, a few
//      fp32 ulp), inverse to each other, and preserve q.k (the reason q is rotated too);
//   4. quantise: |y - deq| <= scale / 2 off the clamp, the row's amax lands on +-127, a
//      zero row is scale 0 / all zeros, tiny rows keep their (subnormal) scale;
//   5. FWHT known answers (e_0 -> ones; ones -> 256 e_0).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"

namespace kv8 = common::kv8;

namespace {

// What the probe printed for its signs (1 = -1), 2026-10-05, torch 2.14.1+cpu:
//   g = torch.Generator().manual_seed(0); torch.randint(0, 2, (256,), generator=g) * 2 - 1
const char* kProbeSigns =
    "1001000000011011111010011000010101001001101000001010000101100101"
    "0111110011100101101000000100110110010110111001011111010100000100"
    "0010011011110011010000111010001011010011010101010111010101111101"
    "1011101101011001110011111010111000110000111001011010000111000100";

double h_entry(uint32_t i, uint32_t j) { return (__builtin_popcount(i & j) & 1) ? -1.0 : 1.0; }

void check_signs() {
  CHECK_EQ(std::strlen(kProbeSigns), size_t(256));
  uint32_t neg = 0;
  for (uint32_t j = 0; j < 256; ++j) {
    CHECK_EQ(kv8::sign_neg(j), kProbeSigns[j] == '1');
    neg += kv8::sign_neg(j);
  }
  CHECK_EQ(neg, 127u);
  std::printf("signs: the probe's hadamard(256, 0), %u of 256 negative\n", neg);
}

void check_f16() {
  // Every finite binary16: f16f exact (common::f16_to_f32 is the loader's exact converter),
  // and the round trip through f32_to_f16_rne is the identity.
  uint32_t n = 0;
  for (uint32_t h = 0; h < 0x10000u; ++h) {
    if (((h >> 10) & 0x1Fu) == 0x1Fu) continue;   // inf / NaN: never a scale
    const float f = kv8::f16f(uint16_t(h));
    CHECK_EQ(kv8::f32_bits(f), kv8::f32_bits(common::f16_to_f32(uint16_t(h))));
    CHECK_EQ(kv8::f32_to_f16_rne(f), uint16_t(h));
    ++n;
  }
  // Halfway cases between neighbours round to even; a hair above rounds up.
  for (uint32_t h = 1; h < 0x7BFFu; ++h) {
    const float lo = kv8::f16f(uint16_t(h)), hi = kv8::f16f(uint16_t(h + 1));
    const float mid = 0.5f * (lo + hi);   // exact: one more bit than binary16 has
    CHECK_EQ(kv8::f32_to_f16_rne(mid), uint16_t((h & 1u) ? h + 1 : h));
    CHECK_EQ(kv8::f32_to_f16_rne(std::nextafter(mid, 1e9f)), uint16_t(h + 1));
    CHECK_EQ(kv8::f32_to_f16_rne(std::nextafter(mid, -1e9f)), uint16_t(h));
  }
  CHECK_EQ(kv8::f32_to_f16_rne(70000.0f), uint16_t(0x7BFFu));   // saturates, never inf
  CHECK_EQ(kv8::f32_to_f16_rne(0.0f), uint16_t(0));
  CHECK_EQ(kv8::f32_to_f16_rne(1e-30f), uint16_t(0));            // below half's range
#ifdef __FLT16_MAX__
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> ex(-30.0f, 16.0f);
  for (int i = 0; i < 2000000; ++i) {
    const float f = std::ldexp(1.0f + float(rng() & 0xFFFFFFu) / 16777216.0f, int(ex(rng)));
    if (f >= 65504.0f) continue;
    const _Float16 c = static_cast<_Float16>(f);
    uint16_t b;
    std::memcpy(&b, &c, 2);
    CHECK_EQ(kv8::f32_to_f16_rne(f), b);
  }
  std::printf("f16: %u finite halves exact, round trip identity, 2e6 random against _Float16\n", n);
#else
  std::printf("f16: %u finite halves exact, round trip identity (no _Float16 here)\n", n);
#endif
}

void check_rotation() {
  // R in fp64 from its definition, R[i][j] = H[i][j] s[j] / 16.
  std::vector<double> R(256 * 256);
  for (uint32_t i = 0; i < 256; ++i)
    for (uint32_t j = 0; j < 256; ++j)
      R[i * 256 + j] = h_entry(i, j) * (kv8::sign_neg(j) ? -1.0 : 1.0) / 16.0;

  // Known answers.
  {
    float e0[256] = {1.0f}, y[256];
    kv8::rotate(e0, y);
    for (uint32_t j = 0; j < 256; ++j) CHECK_EQ(y[j], kv8::sign_neg(j) ? -0.0625f : 0.0625f);
    float ones[256];
    for (float& v : ones) v = 1.0f;
    kv8::fwht256(ones);
    CHECK_EQ(ones[0], 256.0f);
    for (uint32_t j = 1; j < 256; ++j) CHECK_EQ(ones[j], 0.0f);
  }

  std::mt19937 rng(12);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  double worst_fwd = 0, worst_inv = 0, worst_dot = 0;
  for (int t = 0; t < 200; ++t) {
    float x[256], q[256], y[256], back[256], qy[256];
    for (uint32_t i = 0; i < 256; ++i) {
      // bf16-valued, with an outlier channel or two like K's (the reason for rotating)
      x[i] = common::bf16_to_f32(common::f32_to_bf16(nd(rng) * (i % 61 == 7 ? 40.0f : 1.0f)));
      q[i] = nd(rng);
    }
    kv8::rotate(x, y);
    double nx = 0;
    for (uint32_t i = 0; i < 256; ++i) nx += double(x[i]) * x[i];
    nx = std::sqrt(nx);
    for (uint32_t j = 0; j < 256; ++j) {
      double r = 0;
      for (uint32_t i = 0; i < 256; ++i) r += double(x[i]) * R[i * 256 + j];
      worst_fwd = std::max(worst_fwd, std::fabs(r - y[j]) / nx);
    }
    kv8::unrotate(y, back);
    for (uint32_t i = 0; i < 256; ++i) worst_inv = std::max(worst_inv, std::fabs(double(back[i]) - x[i]) / nx);
    // unrotate is y R^T: against fp64 too.
    for (uint32_t i = 0; i < 256; ++i) {
      double r = 0;
      for (uint32_t j = 0; j < 256; ++j) r += double(y[j]) * R[i * 256 + j];
      CHECK(std::fabs(r - back[i]) / nx < 1e-6);
    }
    kv8::rotate(q, qy);
    double d0 = 0, d1 = 0, nq = 0;
    for (uint32_t i = 0; i < 256; ++i) {
      d0 += double(q[i]) * x[i];
      d1 += double(qy[i]) * y[i];
      nq += double(q[i]) * q[i];
    }
    worst_dot = std::max(worst_dot, std::fabs(d0 - d1) / (std::sqrt(nq) * nx));
  }
  std::printf("rotation: |xR - fp64| / |x| <= %.2e, |unrotate(rotate(x)) - x| / |x| <= %.2e,"
              " |qR.kR - q.k| / (|q||k|) <= %.2e\n", worst_fwd, worst_inv, worst_dot);
  CHECK(worst_fwd < 1e-6);
  CHECK(worst_inv < 1e-6);
  CHECK(worst_dot < 1e-6);
}

void check_quantise() {
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  int8_t q[256];
  // A zero row.
  {
    float y[256] = {};
    CHECK_EQ(kv8::quantise(y, q), uint16_t(0));
    for (int8_t v : q) CHECK_EQ(int(v), 0);
  }
  double worst_rel = 0;
  for (int t = 0; t < 3000; ++t) {
    // Row magnitudes from 1e-5 (subnormal half scales) to 1e3.
    const float mag = std::pow(10.0f, -5.0f + 8.0f * float(t) / 3000.0f);
    float y[256];
    for (float& v : y) v = nd(rng) * mag;
    const uint16_t s16 = kv8::quantise(y, q);
    const float sf = kv8::f16f(s16);
    float amax = 0;
    uint32_t at = 0;
    for (uint32_t i = 0; i < 256; ++i)
      if (std::fabs(y[i]) > amax) amax = std::fabs(y[i]), at = i;
    CHECK(sf > 0.0f);
    const bool normal = sf >= 6.103515625e-05f;   // 2^-14: the scale has all 11 bits
    if (normal) CHECK_EQ(std::abs(int(q[at])), 127);
    double e2 = 0, n2 = 0;
    for (uint32_t i = 0; i < 256; ++i) {
      CHECK(q[i] >= -127 && q[i] <= 127);
      const float d = kv8::dequant(q[i], s16);
      CHECK_EQ(d, float(q[i]) * sf);                       // the product is exact
      const float lim = std::fabs(y[i] / sf) <= 127.0f ? 0.5f * sf : std::fabs(y[i]) - 127.0f * sf;
      CHECK(std::fabs(d - y[i]) <= lim * (1.0f + 1e-6f) + 1e-30f);
      e2 += double(d - y[i]) * (d - y[i]);
      n2 += double(y[i]) * y[i];
    }
    if (normal) worst_rel = std::max(worst_rel, std::sqrt(e2 / n2));
  }
  std::printf("quantise: 3000 rows over 1e-5 .. 1e3, rel L2 <= %.2e, amax on +-127\n", worst_rel);
  CHECK(worst_rel < 1.5e-2);

  // encode = rotate then quantise; a K row with outlier channels rotates to a flat row,
  // so its int8 error is the flat row's (the reason for rotkv).
  float x[256], y[256];
  for (uint32_t i = 0; i < 256; ++i) x[i] = nd(rng) * (i == 3 || i == 77 ? 50.0f : 1.0f);
  const uint16_t s_rot = kv8::encode(x, q);
  kv8::rotate(x, y);
  double e_rot = 0, e_pt = 0, n2 = 0;
  for (uint32_t i = 0; i < 256; ++i) {
    e_rot += std::pow(double(kv8::dequant(q[i], s_rot)) - y[i], 2);
    n2 += double(x[i]) * x[i];
  }
  const uint16_t s_pt = kv8::quantise(x, q);   // per token without the rotation
  for (uint32_t i = 0; i < 256; ++i) e_pt += std::pow(double(kv8::dequant(q[i], s_pt)) - x[i], 2);
  std::printf("outlier row: rel L2 rotated %.2e vs per token %.2e\n", std::sqrt(e_rot / n2),
              std::sqrt(e_pt / n2));
  CHECK(e_rot < e_pt);
}

}  // namespace

int main() {
  check_signs();
  check_f16();
  check_rotation();
  check_quantise();
  std::printf("kv8_test: OK\n");
  return 0;
}
