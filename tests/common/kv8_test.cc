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
//   6. spec 18e: the head_dim-128 scheme (hd128) - its signs are torch's hadamard(128, 0)
//      (pinned), rotate_kv / rotate_q / unrotate against the fp64 orthonormal R (x sqrt(2),
//      / sqrt(2), / sqrt(2)), q.k preserved, the inverse, the quantiser at 128 (the 256
//      quantiser's statements), and K2's GQA attention shape where rotkv beats per token.
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

// The scheme end to end on one synthetic head, fp64 attention: bf16 K / V against rotkv
// (q rotated, K and V rotated + int8, the output un-rotated) and against plain per-token
// int8. K has outlier channels (x 20 on 4 of 256), as the real K does. rotkv must be the
// closer of the two (the 12a ranking), and close: rel L2 < 2e-2 over 64 queries at depth 512.
void check_attention() {
  constexpr uint32_t D = 256, T = 512, Q = 64;
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf = [](float x) { return common::bf16_to_f32(common::f32_to_bf16(x)); };
  std::vector<float> K(size_t(T) * D), V(size_t(T) * D), q(size_t(Q) * D);
  for (uint32_t t = 0; t < T; ++t)
    for (uint32_t i = 0; i < D; ++i) {
      K[size_t(t) * D + i] = bf(nd(rng) * (i % 64 == 5 ? 20.0f : 1.0f));
      V[size_t(t) * D + i] = bf(nd(rng));
    }
  for (float& x : q) x = nd(rng) * 0.5f;
  // The three caches as fp32 rows: bf16, rotkv (rotated basis), per token (plain basis).
  std::vector<float> Kr(K.size()), Vr(V.size()), Kp(K.size()), Vp(V.size());
  int8_t q8[D];
  for (uint32_t t = 0; t < T; ++t) {
    const uint16_t sk = kv8::encode(&K[size_t(t) * D], q8);
    for (uint32_t i = 0; i < D; ++i) Kr[size_t(t) * D + i] = kv8::dequant(q8[i], sk);
    const uint16_t sv = kv8::encode(&V[size_t(t) * D], q8);
    for (uint32_t i = 0; i < D; ++i) Vr[size_t(t) * D + i] = kv8::dequant(q8[i], sv);
    const uint16_t pk = kv8::quantise(&K[size_t(t) * D], q8);
    for (uint32_t i = 0; i < D; ++i) Kp[size_t(t) * D + i] = kv8::dequant(q8[i], pk);
    const uint16_t pv = kv8::quantise(&V[size_t(t) * D], q8);
    for (uint32_t i = 0; i < D; ++i) Vp[size_t(t) * D + i] = kv8::dequant(q8[i], pv);
  }
  auto attend = [&](const float* qq, const std::vector<float>& Kc, const std::vector<float>& Vc,
                    double* o) {
    std::vector<double> s(T);
    double mx = -1e300, sum = 0;
    for (uint32_t t = 0; t < T; ++t) {
      double a = 0;
      for (uint32_t i = 0; i < D; ++i) a += double(qq[i]) * Kc[size_t(t) * D + i];
      s[t] = a / 16.0;
      mx = std::max(mx, s[t]);
    }
    for (uint32_t i = 0; i < D; ++i) o[i] = 0;
    for (uint32_t t = 0; t < T; ++t) {
      const double w = std::exp(s[t] - mx);
      sum += w;
      for (uint32_t i = 0; i < D; ++i) o[i] += w * Vc[size_t(t) * D + i];
    }
    for (uint32_t i = 0; i < D; ++i) o[i] /= sum;
  };
  double e_rot = 0, e_pt = 0, n2 = 0;
  for (uint32_t k = 0; k < Q; ++k) {
    const float* qq = &q[size_t(k) * D];
    double ref[D], orot[D], opt[D];
    attend(qq, K, V, ref);
    float qr[D], o32[D], back[D];
    kv8::rotate(qq, qr);
    attend(qr, Kr, Vr, orot);
    for (uint32_t i = 0; i < D; ++i) o32[i] = float(orot[i]);
    kv8::unrotate(o32, back);
    attend(qq, Kp, Vp, opt);
    for (uint32_t i = 0; i < D; ++i) {
      e_rot += (back[i] - ref[i]) * (back[i] - ref[i]);
      e_pt += (opt[i] - ref[i]) * (opt[i] - ref[i]);
      n2 += ref[i] * ref[i];
    }
  }
  const double r_rot = std::sqrt(e_rot / n2), r_pt = std::sqrt(e_pt / n2);
  std::printf("attention (synthetic, K outliers, depth %u): rel L2 vs bf16 KV - rotkv %.2e,"
              " per token %.2e\n", T, r_rot, r_pt);
  CHECK(r_rot < r_pt);
  CHECK(r_rot < 2e-2);
}

// ---- spec 18e: head_dim 128 ----------------------------------------------------------------
namespace h128 = common::kv8::hd128;

// torch 2.2.2 (the Mac's cached wheel), printed 2026-10-06 by
//   g = torch.Generator().manual_seed(0); torch.randint(0, 2, (128,), generator=g)
// - the RANDINT values (1 is +1 after `* 2 - 1`, 0 is -1); the same draw for (256,) begins
// with these 128 (one 32-bit word per element, in order): hadamard(256, 0)'s first 128.
const char* kTorchRandint128 =
    "0110111111100100000101100111101010110110010111110101111010011010"
    "1000001100011010010111111011001001101001000110100000101011111011";

void check_hd128_signs() {
  CHECK_EQ(std::strlen(kTorchRandint128), size_t(128));
  uint32_t neg = 0;
  for (uint32_t j = 0; j < 128; ++j) {
    CHECK_EQ(h128::sign_neg(j), kTorchRandint128[j] == '0');
    CHECK_EQ(h128::sign_neg(j), kProbeSigns[j] == '1');   // the 256 probe string's prefix
    neg += h128::sign_neg(j);
  }
  CHECK_EQ(neg, 58u);
  std::printf("hd128 signs: torch's hadamard(128, 0) = hadamard(256, 0)'s first 128, %u negative\n", neg);
}

void check_hd128_rotation() {
  constexpr uint32_t N = 128;
  std::vector<double> R(N * N);   // orthonormal: H[i][j] s[j] / sqrt(128)
  for (uint32_t i = 0; i < N; ++i)
    for (uint32_t j = 0; j < N; ++j)
      R[i * N + j] = h_entry(i, j) * (h128::sign_neg(j) ? -1.0 : 1.0) / std::sqrt(128.0);
  const double r2 = std::sqrt(2.0);
  {
    float e0[N] = {1.0f}, y[N];
    h128::rotate_kv(e0, y);
    for (uint32_t j = 0; j < N; ++j) CHECK_EQ(y[j], h128::sign_neg(j) ? -0.125f : 0.125f);
    h128::rotate_q(e0, y);
    for (uint32_t j = 0; j < N; ++j) CHECK_EQ(y[j], h128::sign_neg(j) ? -0.0625f : 0.0625f);
    float ones[N];
    for (float& v : ones) v = 1.0f;
    h128::fwht128(ones);
    CHECK_EQ(ones[0], 128.0f);
    for (uint32_t j = 1; j < N; ++j) CHECK_EQ(ones[j], 0.0f);
  }
  std::mt19937 rng(18);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  double w_kv = 0, w_q = 0, w_un = 0, w_inv = 0, w_dot = 0;
  for (int t = 0; t < 200; ++t) {
    float x[N], q[N], yk[N], yq[N], back[N], un[N];
    for (uint32_t i = 0; i < N; ++i) {
      x[i] = common::bf16_to_f32(common::f32_to_bf16(nd(rng) * (i % 37 == 5 ? 30.0f : 1.0f)));
      q[i] = nd(rng);
    }
    h128::rotate_kv(x, yk);
    h128::rotate_q(q, yq);
    double nx = 0, nq = 0, d0 = 0, d1 = 0;
    for (uint32_t i = 0; i < N; ++i) {
      nx += double(x[i]) * x[i];
      nq += double(q[i]) * q[i];
      d0 += double(q[i]) * x[i];
      d1 += double(yq[i]) * yk[i];
    }
    nx = std::sqrt(nx);
    nq = std::sqrt(nq);
    for (uint32_t j = 0; j < N; ++j) {
      double xr = 0, qr = 0;
      for (uint32_t i = 0; i < N; ++i) {
        xr += double(x[i]) * R[i * N + j];
        qr += double(q[i]) * R[i * N + j];
      }
      w_kv = std::max(w_kv, std::fabs(r2 * xr - yk[j]) / nx);   // rotate_kv = sqrt(2) x R
      w_q = std::max(w_q, std::fabs(qr / r2 - yq[j]) / nq);     // rotate_q = x R / sqrt(2)
    }
    h128::unrotate(yk, back);   // the inverse of rotate_kv
    for (uint32_t i = 0; i < N; ++i) w_inv = std::max(w_inv, std::fabs(double(back[i]) - x[i]) / nx);
    h128::unrotate(yq, un);     // = y R^T / sqrt(2) on any y
    for (uint32_t i = 0; i < N; ++i) {
      double r = 0;
      for (uint32_t j = 0; j < N; ++j) r += double(yq[j]) * R[i * N + j];
      w_un = std::max(w_un, std::fabs(r / r2 - un[i]) / nq);
    }
    w_dot = std::max(w_dot, std::fabs(d0 - d1) / (nq * nx));
  }
  std::printf("hd128 rotation: |rotate_kv - sqrt2 xR| %.2e, |rotate_q - xR/sqrt2| %.2e, |unrotate - "
              "yR^T/sqrt2| %.2e, |unrotate(rotate_kv(x)) - x| %.2e, |q'.k' - q.k| / (|q||k|) %.2e\n",
              w_kv, w_q, w_un, w_inv, w_dot);
  CHECK(w_kv < 1e-6 && w_q < 1e-6 && w_un < 1e-6 && w_inv < 1e-6 && w_dot < 1e-6);
  // The 256 path is the template at 256: fwht256 is fwht<256> bit for bit.
  float a[256], b[256];
  for (uint32_t i = 0; i < 256; ++i) a[i] = b[i] = nd(rng);
  kv8::fwht256(a);
  kv8::fwht<256>(b);
  CHECK(std::memcmp(a, b, sizeof a) == 0);
}

void check_hd128_quantise() {
  constexpr uint32_t N = 128;
  std::mt19937 rng(28);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  int8_t q[N];
  const float z[N] = {};
  CHECK_EQ(h128::quantise(z, q), uint16_t(0));
  for (int8_t v : q) CHECK_EQ(int(v), 0);
  for (int t = 0; t < 2000; ++t) {
    const float mag = std::pow(10.0f, -5.0f + 8.0f * float(t) / 2000.0f);
    float y[N];
    for (float& v : y) v = nd(rng) * mag;
    const uint16_t s16 = h128::quantise(y, q);
    // The 256 quantiser on the same 128 values padded with zeros: the same scale and codes.
    float y256[256] = {};
    std::memcpy(y256, y, sizeof y);
    int8_t q256[256];
    CHECK_EQ(kv8::quantise(y256, q256), s16);
    CHECK(std::memcmp(q, q256, N) == 0);
    const float sf = kv8::f16f(s16);
    for (uint32_t i = 0; i < N; ++i) {
      CHECK(q[i] >= -127 && q[i] <= 127);
      const float lim = std::fabs(y[i] / sf) <= 127.0f ? 0.5f * sf : std::fabs(y[i]) - 127.0f * sf;
      CHECK(std::fabs(kv8::dequant(q[i], s16) - y[i]) <= lim * (1.0f + 1e-6f) + 1e-30f);
    }
  }
  std::printf("hd128 quantise: 2000 rows over 1e-5 .. 1e3 within half a scale, equal to the 256 "
              "quantiser on the same values\n");
}

// K2's attention shape on synthetic rows: depth 512, K with outlier channels (x 25 on 2 of
// 128), scale 1/sqrt(128). bf16 KV (fp64 attention) against rotkv at 128 (q rotate_q, K and V
// rotate_kv + int8, the output unrotated) and plain per token: rotkv the closer, and close.
void check_hd128_attention() {
  constexpr uint32_t D = 128, T = 512, Q = 64;
  std::mt19937 rng(38);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf = [](float x) { return common::bf16_to_f32(common::f32_to_bf16(x)); };
  std::vector<float> K(size_t(T) * D), V(size_t(T) * D), q(size_t(Q) * D);
  for (uint32_t t = 0; t < T; ++t)
    for (uint32_t i = 0; i < D; ++i) {
      K[size_t(t) * D + i] = bf(nd(rng) * (i == 9 || i == 73 ? 25.0f : 1.0f));
      V[size_t(t) * D + i] = bf(nd(rng));
    }
  for (float& x : q) x = bf(nd(rng) * 0.7f);
  std::vector<float> Kr(K.size()), Vr(V.size()), Kp(K.size()), Vp(V.size());
  int8_t q8[D];
  for (uint32_t t = 0; t < T; ++t) {
    const uint16_t sk = h128::encode(&K[size_t(t) * D], q8);
    for (uint32_t i = 0; i < D; ++i) Kr[size_t(t) * D + i] = kv8::dequant(q8[i], sk);
    const uint16_t sv = h128::encode(&V[size_t(t) * D], q8);
    for (uint32_t i = 0; i < D; ++i) Vr[size_t(t) * D + i] = kv8::dequant(q8[i], sv);
    const uint16_t pk = h128::quantise(&K[size_t(t) * D], q8);
    for (uint32_t i = 0; i < D; ++i) Kp[size_t(t) * D + i] = kv8::dequant(q8[i], pk);
    const uint16_t pv = h128::quantise(&V[size_t(t) * D], q8);
    for (uint32_t i = 0; i < D; ++i) Vp[size_t(t) * D + i] = kv8::dequant(q8[i], pv);
  }
  const double scale = 1.0 / std::sqrt(128.0);
  auto attend = [&](const float* qq, const std::vector<float>& Kc, const std::vector<float>& Vc, double* o) {
    std::vector<double> s(T);
    double mx = -1e300, sum = 0;
    for (uint32_t t = 0; t < T; ++t) {
      double a = 0;
      for (uint32_t i = 0; i < D; ++i) a += double(qq[i]) * Kc[size_t(t) * D + i];
      s[t] = a * scale;
      mx = std::max(mx, s[t]);
    }
    for (uint32_t i = 0; i < D; ++i) o[i] = 0;
    for (uint32_t t = 0; t < T; ++t) {
      const double w = std::exp(s[t] - mx);
      sum += w;
      for (uint32_t i = 0; i < D; ++i) o[i] += w * Vc[size_t(t) * D + i];
    }
    for (uint32_t i = 0; i < D; ++i) o[i] /= sum;
  };
  double e_rot = 0, e_pt = 0, n2 = 0;
  for (uint32_t k = 0; k < Q; ++k) {
    const float* qq = &q[size_t(k) * D];
    double ref[D], orot[D], opt[D];
    attend(qq, K, V, ref);
    float qr[D], o32[D], back[D];
    h128::rotate_q(qq, qr);
    attend(qr, Kr, Vr, orot);
    for (uint32_t i = 0; i < D; ++i) o32[i] = float(orot[i]);
    h128::unrotate(o32, back);
    attend(qq, Kp, Vp, opt);
    for (uint32_t i = 0; i < D; ++i) {
      e_rot += (back[i] - ref[i]) * (back[i] - ref[i]);
      e_pt += (opt[i] - ref[i]) * (opt[i] - ref[i]);
      n2 += ref[i] * ref[i];
    }
  }
  const double r_rot = std::sqrt(e_rot / n2), r_pt = std::sqrt(e_pt / n2);
  std::printf("hd128 attention (synthetic, K outliers, depth %u): rel L2 vs bf16 KV - rotkv %.2e, "
              "per token %.2e\n", T, r_rot, r_pt);
  CHECK(r_rot < r_pt);
  CHECK(r_rot < 2e-2);
}

}  // namespace

int main() {
  check_signs();
  check_f16();
  check_rotation();
  check_quantise();
  check_attention();
  check_hd128_signs();
  check_hd128_rotation();
  check_hd128_quantise();
  check_hd128_attention();
  std::printf("kv8_test: OK\n");
  return 0;
}
