// Spec 18b Task 1: loader::k2_rope_table - layout, the fp32 construction, bf16-valued
// entries. Host only.
#include <cmath>
#include <cstdio>

#include "check.h"
#include "common/bf16.h"
#include "loader/k2_repack.h"
#include "model/k2_horizon.h"

int main() {
  const model::K2Desc& d = model::k2();
  const uint32_t L = 32768, H = d.head_dim / 2;   // 64 angles: cat(freqs, freqs) over 128
  const std::vector<float> t = loader::k2_rope_table(d, L);
  CHECK_EQ(t.size() * 4, d.rope_table_bytes(L));
  auto cosv = [&](uint32_t p, uint32_t i) { return t[(size_t(p) * 2 + 0) * H + i]; };
  auto sinv = [&](uint32_t p, uint32_t i) { return t[(size_t(p) * 2 + 1) * H + i]; };
  for (uint32_t i = 0; i < H; ++i) {   // position 0: the identity rotation
    CHECK(cosv(0, i) == 1.0f);
    CHECK(sinv(0, i) == 0.0f);
  }
  // Every entry is a bf16 value (the reference casts cos / sin to the hidden dtype).
  for (size_t j = 0; j < t.size(); j += 7) CHECK(common::bf16_to_f32(common::f32_to_bf16(t[j])) == t[j]);
  // The fp32 construction, re-spelled at probes: inv = 1 / powf(1e7, 2i / 128), ang =
  // float(p) * inv, both fp32 - NOT the double angle the 27B's table uses.
  for (uint32_t p : {1u, 4097u, 16383u, 32767u})
    for (uint32_t i : {0u, 1u, 31u, 63u}) {
      const float inv = 1.0f / std::pow(1e7f, float(2 * i) / 128.0f);
      const float ang = float(p) * inv;
      CHECK(cosv(p, i) == common::bf16_to_f32(common::f32_to_bf16(float(std::cos(double(ang))))));
      CHECK(sinv(p, i) == common::bf16_to_f32(common::f32_to_bf16(float(std::sin(double(ang))))));
    }
  // i = 0 has inv = 1 exactly: the angle is the position itself.
  CHECK(cosv(1, 0) == common::bf16_to_f32(common::f32_to_bf16(float(std::cos(1.0)))));
  // The last angle (i = 63) is ~1.29e-7 rad per position: sin(32767 x that) is small but
  // non-zero, and the fp32 product is what decides its bf16 value.
  CHECK(sinv(32767, 63) > 0.0f && sinv(32767, 63) < 0.01f);
  std::puts("k2_rope_test OK");
  return 0;
}
