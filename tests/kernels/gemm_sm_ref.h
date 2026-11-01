#pragma once
// gemm_sm_ref.h - spec 28's host side of the small-M GEMM (plan 28a; shared with 28b):
//
//   - weight views over the bytes the device reads: int4 g64 symmetric in layout 0
//     (GPTQ qweight [K/8][N] + f16 scales [K/64][N]) or layout 1 (544-byte tiles,
//     common/repack.h), and the int8 lm_head ([n_tile][k16][16 k][16 n] + fp32 [N]
//     column scales, loader/lm_head_int8.h);
//   - `ref`: the fp64 reference out[m][n] = sum_k w(k, n) x[m][k] and, beside it, the
//     N2 bound's denominator sum_k |w(k, n) x[m][k]| (spec 28 §3);
//   - `twin_gemv` / `twin_gemv_i8w`: gemv.cl's / gemv_i8w.cl's fp32 order on the host
//     (per split-K slice, per 64-k group, eight dot8 chains, the group's scale, the
//     slices summed in s order) - what the M = 1 path computes, up to the device's
//     multiply-add contraction;
//   - the i8 arm's per-row activation quantiser (pw4a8_quant's rule: scale =
//     max|x| / 127, RNE, saturate) and the reference of what the i8 kernel computes
//     (fp64 over the quantised rows), which is the arm's kernel-correctness bar - the
//     quantisation error itself is the type rule's question, measured against `ref`;
//   - `n2`: spec 28 N2 per output (|err| <= 2^-16 x sum|w x|) and the per-row cosine.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "common/bf16.h"

namespace gemm_sm {

constexpr uint32_t kGroup = 64;
constexpr uint32_t kTileU32 = 136;

// The weights as the device holds them. kind: 0 = int4 layout 0, 1 = int4 layout 1,
// 2 = int8 head (tiled, column scale).
struct WeightView {
  uint32_t K = 0, N = 0, kind = 0;
  const uint32_t* words = nullptr;   // int4: qweight (L0) or tiles (L1)
  const uint16_t* scales = nullptr;  // int4 L0: f16 [K/64][N]
  const int8_t* q8 = nullptr;        // int8 head: tiled bytes
  const float* colscale = nullptr;   // int8 head: fp32 [N]

  // The integer weight (q - 8 for int4, q for int8) and its multiplier.
  int q(uint32_t k, uint32_t n) const {
    if (kind == 0) return int((words[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 0xFu) - 8;
    if (kind == 1) {
      const uint32_t* tile = words + (size_t(n / 16) * (K / kGroup) + k / kGroup) * kTileU32;
      return int((tile[((k % kGroup) / 8) * 16 + n % 16] >> (4 * (k % 8))) & 0xFu) - 8;
    }
    return q8[(size_t(n / 16) * (K / 16) + k / 16) * 256 + size_t(k % 16) * 16 + n % 16];
  }
  // int4: the f16 group scale of (k, n); int8: the column scale.
  float scale(uint32_t k, uint32_t n) const {
    if (kind == 0) return common::f16_to_f32(scales[size_t(k / kGroup) * N + n]);
    if (kind == 1) {
      const uint32_t* tile = words + (size_t(n / 16) * (K / kGroup) + k / kGroup) * kTileU32;
      uint16_t h;
      std::memcpy(&h, reinterpret_cast<const uint16_t*>(tile + 128) + n % 16, 2);
      return common::f16_to_f32(h);
    }
    return colscale[n];
  }
  double w(uint32_t k, uint32_t n) const { return double(q(k, n)) * double(scale(k, n)); }
};

inline std::vector<float> bf16_rows(const std::vector<uint16_t>& x) {
  std::vector<float> f(x.size());
  for (size_t i = 0; i < x.size(); ++i) f[i] = common::bf16_to_f32(x[i]);
  return f;
}

// fp64 reference over columns [n0, n1) (all by default): ref[m][n], bound[m][n] = sum |w x|.
// `xf` is [M][K] in fp32 (bf16 values, or dequantised int8 rows for the i8 reference). The
// outputs are [M][N]; columns outside [n0, n1) are zero, or left as they were when the
// vectors already have that size (so several column ranges can fill one pair).
inline void ref(const WeightView& wv, const std::vector<float>& xf, uint32_t M,
                std::vector<double>& out, std::vector<double>& bound, uint32_t n0 = 0,
                uint32_t n1 = 0) {
  if (n1 == 0) n1 = wv.N;
  if (out.size() != size_t(M) * wv.N) out.assign(size_t(M) * wv.N, 0.0);
  if (bound.size() != size_t(M) * wv.N) bound.assign(size_t(M) * wv.N, 0.0);
  std::vector<double> col(wv.K);
  for (uint32_t n = n0; n < n1; ++n) {
    for (uint32_t k = 0; k < wv.K; ++k) col[k] = wv.w(k, n);
    for (uint32_t m = 0; m < M; ++m) {
      const float* x = xf.data() + size_t(m) * wv.K;
      double a = 0, b = 0;
      for (uint32_t k = 0; k < wv.K; ++k) {
        const double p = col[k] * double(x[k]);
        a += p;
        b += std::fabs(p);
      }
      out[size_t(m) * wv.N + n] = a;
      bound[size_t(m) * wv.N + n] = b;
    }
  }
}

// gemv.cl's fp32 order (int4, either layout): slice s of S owns groups [s G/S, (s+1) G/S);
// per group eight dot8 chains (a = 0; a += q * x, k ascending), gacc += dot8, then
// acc += gacc * scale; the S partials summed by the host from 0.f in s order (run_gemv).
inline std::vector<float> twin_gemv(const WeightView& wv, const std::vector<float>& xf,
                                    uint32_t M, uint32_t S) {
  const uint32_t G = wv.K / kGroup, GS = G / S;
  std::vector<float> out(size_t(M) * wv.N, 0.f);
  for (uint32_t n = 0; n < wv.N; ++n)
    for (uint32_t m = 0; m < M; ++m) {
      const float* x = xf.data() + size_t(m) * wv.K;
      float total = 0.f;
      for (uint32_t s = 0; s < S; ++s) {
        float acc = 0.f;
        for (uint32_t g = s * GS; g < (s + 1) * GS; ++g) {
          float gacc = 0.f;
          for (uint32_t j = 0; j < 8; ++j) {
            float a = 0.f;
            for (uint32_t i = 0; i < 8; ++i) {
              const uint32_t k = g * kGroup + j * 8 + i;
              a += float(wv.q(k, n)) * x[k];
            }
            gacc += a;
          }
          acc += gacc * wv.scale(g * kGroup, n);
        }
        total += acc;
      }
      out[size_t(m) * wv.N + n] = total;
    }
  return out;
}

// gemv_i8w.cl's order: one ascending fp32 chain over k, then x the column scale.
inline std::vector<float> twin_gemv_i8w(const WeightView& wv, const std::vector<float>& xf,
                                        uint32_t M) {
  std::vector<float> out(size_t(M) * wv.N, 0.f);
  for (uint32_t n = 0; n < wv.N; ++n)
    for (uint32_t m = 0; m < M; ++m) {
      const float* x = xf.data() + size_t(m) * wv.K;
      float a = 0.f;
      for (uint32_t k = 0; k < wv.K; ++k) a += float(wv.q(k, n)) * x[k];
      out[size_t(m) * wv.N + n] = a * wv.colscale[n];
    }
  return out;
}

// pw4a8_quant's rule per row: scale = max|x| / 127 (fp32, correctly rounded), q = RNE(x / scale)
// saturated to [-127, 127] (|x| / scale <= 127 by construction); an all-zero row -> scale 0, q 0.
inline void quant_rows_i8(const std::vector<float>& xf, uint32_t M, uint32_t K,
                          std::vector<int8_t>& q, std::vector<float>& xs) {
  q.assign(size_t(M) * K, 0);
  xs.assign(M, 0.f);
  for (uint32_t m = 0; m < M; ++m) {
    const float* x = xf.data() + size_t(m) * K;
    float amax = 0.f;
    for (uint32_t k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x[k]));
    const float s = amax / 127.0f;
    xs[m] = s;
    if (s == 0.f) continue;
    for (uint32_t k = 0; k < K; ++k) {
      const float r = std::nearbyint(x[k] / s);   // default rounding mode: to nearest even
      q[size_t(m) * K + k] = int8_t(std::max(-127.f, std::min(127.f, r)));
    }
  }
}
// The dequantised rows q x scale in fp32 - the i8 arm's reference input.
inline std::vector<float> dequant_rows_i8(const std::vector<int8_t>& q, const std::vector<float>& xs,
                                          uint32_t M, uint32_t K) {
  std::vector<float> f(size_t(M) * K);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < K; ++k) f[size_t(m) * K + k] = float(q[size_t(m) * K + k]) * xs[m];
  return f;
}

// Spec 28 N2 over rows [0, M) and columns [n0, n1): |got - ref| <= 2^-16 x bound per output,
// and the per-row cosine >= 0.9999999. `ratio` is |err| / (2^-16 bound): N2 holds iff
// max_ratio <= 1. Outputs whose bound is 0 (an all-zero row) must be exactly 0.
struct N2 {
  double max_ratio = 0, median_ratio = 0, max_abs = 0, min_cos = 1;
  size_t outputs = 0, over = 0;
  bool pass() const { return over == 0 && min_cos >= 0.9999999; }
};
inline N2 n2(const std::vector<float>& got, const std::vector<double>& refv,
             const std::vector<double>& bound, uint32_t M, uint32_t N, uint32_t n0 = 0,
             uint32_t n1 = 0) {
  if (n1 == 0) n1 = N;
  N2 r;
  std::vector<double> ratios;
  ratios.reserve(size_t(M) * (n1 - n0));
  for (uint32_t m = 0; m < M; ++m) {
    double dot = 0, gg = 0, rr = 0;
    for (uint32_t n = n0; n < n1; ++n) {
      const size_t i = size_t(m) * N + n;
      const double e = std::fabs(double(got[i]) - refv[i]);
      const double lim = std::ldexp(bound[i], -16);
      const double q = lim > 0 ? e / lim : (e == 0 ? 0.0 : 1e30);
      ratios.push_back(q);
      r.max_ratio = std::max(r.max_ratio, q);
      r.max_abs = std::max(r.max_abs, e);
      if (q > 1.0) ++r.over;
      dot += double(got[i]) * refv[i];
      gg += double(got[i]) * double(got[i]);
      rr += refv[i] * refv[i];
    }
    if (rr > 0) r.min_cos = std::min(r.min_cos, dot / std::sqrt(gg * rr));
  }
  r.outputs = ratios.size();
  if (!ratios.empty()) {
    std::nth_element(ratios.begin(), ratios.begin() + ratios.size() / 2, ratios.end());
    r.median_ratio = ratios[ratios.size() / 2];
  }
  return r;
}

}  // namespace gemm_sm
