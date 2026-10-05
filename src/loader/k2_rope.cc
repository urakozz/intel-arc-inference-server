#include <cmath>

#include "common/bf16.h"
#include "loader/k2_repack.h"

namespace loader {

// K2HorizonRotaryEmbedding, step for step in fp32 (modeling_k2_horizon.py:782-800):
//   inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))    fp32 pow, fp32 divide
//   freqs    = inv_freq @ position                                    one fp32 product
//   cos, sin = emb.cos(), emb.sin()   (x attention_scaling = 1.0)     fp32
//   .to(hidden dtype)                                                 bf16
std::vector<float> k2_rope_table(const model::K2Desc& d, uint32_t max_len) {
  const uint32_t half = d.head_dim / 2;
  const float base = float(d.rope_theta);   // 1e7 is exact in fp32
  std::vector<float> inv(half);
  for (uint32_t i = 0; i < half; ++i)
    inv[i] = 1.0f / std::pow(base, float(2 * i) / float(d.head_dim));
  std::vector<float> t(size_t(max_len) * 2 * half);
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t i = 0; i < half; ++i) {
      const float ang = float(p) * inv[i];
      t[(size_t(p) * 2 + 0) * half + i] =
          common::bf16_to_f32(common::f32_to_bf16(float(std::cos(double(ang)))));
      t[(size_t(p) * 2 + 1) * half + i] =
          common::bf16_to_f32(common::f32_to_bf16(float(std::sin(double(ang)))));
    }
  return t;
}

}  // namespace loader
