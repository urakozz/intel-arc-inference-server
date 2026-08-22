#include <cstdint>
#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"

int main() {
  // bf16 round trip on representable values, and RNE on a tie.
  CHECK_EQ(common::bf16_to_f32(common::f32_to_bf16(1.5f)), 1.5f);
  CHECK_EQ(common::f32_to_bf16(1.0f), uint16_t(0x3F80));
  CHECK_EQ(common::f16_to_f32(uint16_t(0x3C00)), 1.0f);
  CHECK_EQ(common::f16_to_f32(common::f32_to_f16(0.015625f)), 0.015625f);

  // GPTQ nibble semantics: word 0 holds k = 0..7 for column n, low nibble first.
  common::Int4Gptq w;
  w.K = 64; w.N = 16;
  w.qweight.assign(w.K / 8 * w.N, 0u);
  w.scales.assign(w.K / 64 * w.N, common::f32_to_f16(0.5f));
  w.qweight[0 * w.N + 3] = 0x0000000Fu;   // k=0, n=3 -> nibble 15 -> (15-8)*0.5 = 3.5
  w.qweight[1 * w.N + 3] = 0x00000080u;   // k=9 (row 1, nibble 1 -> k=8+1), n=3 -> nibble 8 -> 0
  CHECK_NEAR(w.at(0, 3), 3.5, 1e-6);
  CHECK_NEAR(w.at(1, 3), (0 - 8) * 0.5, 1e-6);
  CHECK_NEAR(w.at(9, 3), 0.0, 1e-6);
  CHECK_EQ(w.bytes(), size_t(w.K / 8 * w.N * 4 + w.K / 64 * w.N * 2));

  common::Int4Gptq r = common::Int4Gptq::random(128, 64, 1);
  CHECK_EQ(r.qweight.size(), size_t(128 / 8 * 64));
  CHECK(r.at(5, 7) >= -8.0f * 0.08f && r.at(5, 7) <= 7.0f * 0.08f);   // scales are in [0.02, 0.08]
  std::puts("int4_test OK");
  return 0;
}
