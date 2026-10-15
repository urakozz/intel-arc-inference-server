#include "runtime/muse/muse_rope.h"

#include <cmath>

#include "common/bf16.h"

namespace runtime::muse {

std::vector<float> rope_table(uint32_t head_dim, float theta, uint32_t max_len) {
  const uint32_t half = head_dim / 2;
  std::vector<float> inv(half);
  for (uint32_t i = 0; i < half; ++i) inv[i] = 1.0f / std::pow(theta, float(2 * i) / float(head_dim));
  std::vector<float> t(size_t(max_len) * 2 * half);
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t i = 0; i < half; ++i) {
      const float ang = float(p) * inv[i];
      t[(size_t(p) * 2 + 0) * half + i] = common::bf16_to_f32(common::f32_to_bf16(float(std::cos(double(ang)))));
      t[(size_t(p) * 2 + 1) * half + i] = common::bf16_to_f32(common::f32_to_bf16(float(std::sin(double(ang)))));
    }
  return t;
}

}  // namespace runtime::muse
