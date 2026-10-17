#include <cmath>

#include "common/bf16.h"
#include "loader/lfm2moe_repack.h"

namespace loader {

// transformers 5.19.0's Lfm2MoeRotaryEmbedding.forward (rope_type "default"), step for step in fp32
// (kolibri1_rope.cc's method, theta 1e7, head_dim 64):
//   inv  = 1.0 / (theta ** (arange(0, 64, 2).float() / 64))   fp32 pow, fp32 divide
//   freqs = position x inv                                      one fp32 product
//   cos, sin of cat(freqs, freqs) -> .to(bf16)
// Half a head of angles per position (dims i and i + 32 share one). tests/loader/lfm2moe_rope_test.cc
// holds it to transformers' own values (tests/loader/lfm2moe_rope_fixture.h) bit for bit.
std::vector<float> lfm_rope_table(const model::Lfm2MoeDesc& d, uint32_t max_len) {
  const uint32_t half = d.head_dim / 2;
  const float base = float(d.rope_theta);   // 1e7 is exact in fp32
  std::vector<float> inv(half);
  for (uint32_t i = 0; i < half; ++i) inv[i] = 1.0f / std::pow(base, float(2 * i) / float(d.head_dim));
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
