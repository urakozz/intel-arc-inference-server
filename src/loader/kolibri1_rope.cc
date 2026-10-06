#include <cmath>

#include "common/bf16.h"
#include "loader/kolibri1_repack.h"

namespace loader {

// kolibri_ref.rope_cos_sin, step for step in fp32 (the sliding layers' RoPE; the full layers
// have none):
//   inv  = 1.0 / (theta ** (arange(0, 128, 2).float() / 128))   fp32 pow, fp32 divide
//   freqs = inv @ position                                        one fp32 product
//   cos, sin of cat(freqs, freqs)  -> .to(bf16)
// Half a head of angles per position (dims i and i + 64 share one). tests/loader/
// kolibri1_rope_test.cc holds it to torch's values (tests/kernels/kolibri_fixture.h) bit for bit.
std::vector<float> kol_rope_table(const model::Kolibri1Desc& d, uint32_t max_len) {
  const uint32_t half = d.head_dim / 2;
  const float base = float(d.rope_theta);   // 1e4 is exact in fp32
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
