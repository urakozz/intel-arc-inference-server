#include <cstdint>
#include <cstring>
#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"

static float f32_from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

int main() {
  // bf16 round trip on representable values.
  CHECK_EQ(common::bf16_to_f32(common::f32_to_bf16(1.5f)), 1.5f);
  CHECK_EQ(common::f32_to_bf16(1.0f), uint16_t(0x3F80));
  CHECK_EQ(common::f16_to_f32(uint16_t(0x3C00)), 1.0f);
  CHECK_EQ(common::f16_to_f32(common::f32_to_f16(0.015625f)), 0.015625f);

  // RNE on an exact tie: the 16 dropped bits are exactly 0x8000, so the result
  // is the neighbour with an even low bit -- round up only when the kept half is odd.
  CHECK_EQ(common::f32_to_bf16(f32_from_bits(0x3F808000u)), uint16_t(0x3F80));  // kept half even -> stays
  CHECK_EQ(common::f32_to_bf16(f32_from_bits(0x3F818000u)), uint16_t(0x3F82));  // kept half odd  -> up

  // Same for f16, where the 13 dropped mantissa bits are exactly 0x1000.
  CHECK_EQ(common::f32_to_f16(f32_from_bits((127u << 23) | (1u << 13) | 0x1000u)),
           uint16_t(0x3C02));   // kept mantissa LSB 1 -> rounds up to mantissa 0x002
  CHECK_EQ(common::f32_to_f16(f32_from_bits((127u << 23) | 0x1000u)),
           uint16_t(0x3C00));   // kept mantissa LSB 0 -> stays at mantissa 0x000

  // GPTQ nibble semantics: word 0 holds k = 0..7 for column n, low nibble first.
  // K = 128 gives two scale groups (kGroup = 64), so the group index is pinned too.
  common::Int4Gptq w;
  w.K = 128; w.N = 16;
  w.qweight.assign(w.K / 8 * w.N, 0u);
  w.scales.assign(w.K / 64 * w.N, common::f32_to_f16(0.5f));            // group 0 (k = 0..63)
  for (uint32_t n = 0; n < w.N; ++n)
    w.scales[1 * w.N + n] = common::f32_to_f16(0.25f);                  // group 1 (k = 64..127)
  w.qweight[0 * w.N + 3] = 0x0000000Fu;   // k=0, n=3 -> nibble 15 -> (15-8)*0.5 = 3.5
  w.qweight[1 * w.N + 3] = 0x00000080u;   // k=9 (row 1, nibble 1 -> k=8+1), n=3 -> nibble 8 -> 0
  w.qweight[9 * w.N + 3] = 0xFu << (4 * 3);  // k=75 (row 9, nibble 3 -> k=72+3), n=3 -> nibble 15
  CHECK_NEAR(w.at(0, 3), 3.5, 1e-6);          // group 0 scale 0.5
  CHECK_NEAR(w.at(1, 3), (0 - 8) * 0.5, 1e-6);
  CHECK_NEAR(w.at(9, 3), 0.0, 1e-6);
  CHECK_NEAR(w.at(75, 3), (15 - 8) * 0.25, 1e-6);   // group 1 scale 0.25 -> 1.75
  CHECK_NEAR(w.at(74, 3), (0 - 8) * 0.25, 1e-6);    // neighbouring nibble, same group
  CHECK_EQ(w.bytes(), size_t(w.K / 8 * w.N * 4 + w.K / 64 * w.N * 2));

  common::Int4Gptq r = common::Int4Gptq::random(128, 64, 1);
  CHECK_EQ(r.qweight.size(), size_t(128 / 8 * 64));
  CHECK(r.at(5, 7) >= -8.0f * 0.08f && r.at(5, 7) <= 7.0f * 0.08f);   // scales are in [0.02, 0.08]
  std::puts("int4_test OK");
  return 0;
}
