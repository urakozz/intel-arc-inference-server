#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "common/bf16.h"

namespace loader {

// A family loader's RoPE table fp32 [max_len][2][dims / 2], cos at [p][0][i], sin at [p][1][i], each a bf16 value -
// the references' fp32 steps (kolibri_ref.rope_cos_sin):
//   inv[i] = 1 / powf(theta, float(2i) / dims); ang = float(p) x inv[i]; bf16(cos / sin(ang))
// Kolibri-1's (kol_rope_table) and Flash-Next's (q4_rope_table); Qwen3.8's is loader.cc's, in double, unrounded.
inline std::vector<float> bf16_rope_table(uint32_t dims, float theta, uint32_t max_len) {
  const uint32_t half = dims / 2;
  std::vector<float> inv(half);
  for (uint32_t i = 0; i < half; ++i) inv[i] = 1.0f / std::pow(theta, float(2 * i) / float(dims));
  std::vector<float> t(size_t(max_len) * 2 * half);
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t i = 0; i < half; ++i) {
      const float ang = float(p) * inv[i];
      t[(size_t(p) * 2 + 0) * half + i] = common::bf16_to_f32(common::f32_to_bf16(float(std::cos(double(ang)))));
      t[(size_t(p) * 2 + 1) * half + i] = common::bf16_to_f32(common::f32_to_bf16(float(std::sin(double(ang)))));
    }
  return t;
}

}  // namespace loader
