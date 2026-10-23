#include "loader/nvfp4.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "common/bf16.h"
#include "loader/lm_head_int8.h"

// nvfp4.cc - the NVFP4 decode and the two quantisers from (W', g) (nvfp4.h has the contract).
namespace loader {

float e2m1_value(uint8_t code) {
  static const float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = kMag[code & 0x7];
  return (code & 0x8) ? -m : m;
}

bool e4m3_is_nan(uint8_t code) { return (code & 0x7F) == 0x7F; }

float e4m3_value(uint8_t code) {
  if (e4m3_is_nan(code)) return NAN;
  const uint32_t e = (code >> 3) & 0xF, m = code & 0x7;
  // bias 7; exponent 0 is subnormal m x 2^-9 (dflash_ref.e4m3_value)
  const float v = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + float(m) / 8.0f, int(e) - 7);
  return (code & 0x80) ? -v : v;
}

std::vector<uint16_t> decode_nvfp4(const uint8_t* weight, const uint8_t* scale, uint32_t K, uint32_t N,
                                   const std::string& name) {
  if (K % 16) throw std::runtime_error("NVFP4 " + name + ": K " + std::to_string(K) + " is not a multiple of 16");
  const uint32_t B = K / 16;
  std::vector<uint16_t> out(size_t(N) * K);
  float e2m1[16];
  for (uint32_t c = 0; c < 16; ++c) e2m1[c] = e2m1_value(uint8_t(c));
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t b = 0; b < B; ++b) {
      const uint8_t sc = scale[size_t(n) * B + b];
      if (e4m3_is_nan(sc)) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "0x%02x", unsigned(sc));
        throw std::runtime_error("NVFP4 " + name + ": weight_scale[" + std::to_string(n) + "][" + std::to_string(b) +
                                 "] is an e4m3 NaN code (" + buf + ")");
      }
      const float s = e4m3_value(sc);
      for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t k = b * 16 + j;
        const uint8_t byte = weight[size_t(n) * (K / 2) + k / 2];
        const uint8_t code = (k & 1) ? uint8_t(byte >> 4) : uint8_t(byte & 0xF);
        // exact: <= 6 significant bits in [2^-10, 2688], so the fp32 product and its bf16 are W' itself
        out[size_t(n) * K + k] = common::f32_to_bf16(s * e2m1[code]);
      }
    }
  return out;
}

uint16_t f64_to_f16_rne(double x) {
  const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
  if (std::isnan(x)) return 0x7E00;
  double a = std::fabs(x);
  if (a == 0.0) return sign;
  int e2 = 0;
  (void)std::frexp(a, &e2);                         // a = f x 2^e2, f in [0.5, 1): the unbiased exponent is e2 - 1
  const int ue = std::max(e2 - 1, -14);             // below 2^-14 the f16 quantum is 2^-24 (subnormals)
  const double q = std::ldexp(1.0, ue - 10);
  a = std::nearbyint(a / q) * q;                    // one rounding, ties to even (the quotient is exact: q is 2^n)
  if (a >= 65520.0 || std::isinf(a)) return uint16_t(sign | 0x7C00);
  if (a == 0.0) return sign;
  (void)std::frexp(a, &e2);
  const int ue2 = e2 - 1;
  if (ue2 < -14) return uint16_t(sign | uint16_t(a / std::ldexp(1.0, -24)));   // subnormal: mantissa x 2^-24
  const uint32_t mant = uint32_t(a / std::ldexp(1.0, ue2 - 10)) - 1024u;
  return uint16_t(sign | uint16_t((ue2 + 15) << 10) | uint16_t(mant));
}

void rtn_int4_g64_g(const uint16_t* wp, double g, uint32_t K, uint32_t N, uint32_t* qweight, uint16_t* scales) {
  if (K % 64) throw std::runtime_error("rtn_int4_g64_g: K " + std::to_string(K) + " is not a multiple of 64");
  for (uint32_t n = 0; n < N; ++n) {
    const uint16_t* row = wp + size_t(n) * K;
    for (uint32_t grp = 0; grp < K / 64; ++grp) {
      double amax = 0.0;
      for (uint32_t k = grp * 64; k < grp * 64 + 64; ++k)
        amax = std::fmax(amax, std::fabs(g * double(common::bf16_to_f32(row[k]))));   // exact: 24 x 6 bits
      const uint16_t s16 = f64_to_f16_rne(2.0 * amax / 15.0);
      scales[size_t(grp) * N + n] = s16;
      const double s = double(common::f16_to_f32(s16));
      for (uint32_t r = grp * 8; r < grp * 8 + 8; ++r) {
        uint32_t word = 0;
        for (uint32_t j = 0; j < 8; ++j) {
          const double v = g * double(common::bf16_to_f32(row[r * 8 + j]));
          double qv = s > 0.0 ? std::nearbyint(v / s) + 8.0 : 8.0;
          qv = std::fmin(std::fmax(qv, 0.0), 15.0);
          word |= uint32_t(qv) << (4 * j);
        }
        qweight[size_t(r) * N + n] = word;
      }
    }
  }
}

void quantise_int8_tiled_g(const uint16_t* wp, float g, uint32_t K, uint32_t N, int8_t* q_tiled, float* scales,
                           unsigned threads) {
  quantise_int8_tiled(wp, K, N, q_tiled, scales, threads);   // the codes and s'_r of W' (independent of g)
  for (uint32_t n = 0; n < N; ++n) scales[n] = g * scales[n];
}

}  // namespace loader
