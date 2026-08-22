#pragma once
#include <cstdint>
#include <random>
#include <vector>
#include "common/bf16.h"
#include "common/int4.h"

// Reference y[m][n] = sum_k x[m][k] * W[k][n] with double accumulation.
inline void gemv_ref(const common::Int4Gptq& w, const std::vector<uint16_t>& x_bf16,
                     uint32_t M, std::vector<float>& out) {
  out.assign(size_t(M) * w.N, 0.f);
  std::vector<float> xf(size_t(M) * w.K);
  for (size_t i = 0; i < xf.size(); ++i) xf[i] = common::bf16_to_f32(x_bf16[i]);
  for (uint32_t n = 0; n < w.N; ++n) {
    for (uint32_t m = 0; m < M; ++m) {
      double acc = 0.0;
      for (uint32_t k = 0; k < w.K; ++k) acc += double(xf[size_t(m) * w.K + k]) * double(w.at(k, n));
      out[size_t(m) * w.N + n] = float(acc);
    }
  }
}

inline std::vector<uint16_t> random_bf16(size_t n, uint32_t seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = common::f32_to_bf16(d(rng));
  return v;
}
