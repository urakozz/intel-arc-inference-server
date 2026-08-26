#include <cstdint>
#include <cstring>
#include <vector>
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

  // Layout 0 keeps qweight and scales in separate row-major arrays. Exercise
  // all three fusion maps with hand-derived tagged values; expected values do
  // not call the repacker or its column-map helpers.
  constexpr uint32_t RK = 64, PN = 32, RN = 64;
  std::vector<uint32_t> aq(size_t(RK / 8) * PN), bq(size_t(RK / 8) * PN);
  std::vector<uint16_t> as(size_t(RK / 64) * PN), bs(size_t(RK / 64) * PN);
  for (uint32_t row = 0; row < RK / 8; ++row)
    for (uint32_t n = 0; n < PN; ++n) {
      aq[size_t(row) * PN + n] = 0xA0000000u | (row << 8) | n;
      bq[size_t(row) * PN + n] = 0xB0000000u | (row << 8) | n;
    }
  for (uint32_t n = 0; n < PN; ++n) {
    as[n] = uint16_t(0x1000u + n);
    bs[n] = uint16_t(0x2000u + n);
  }
  const common::Part ap{aq.data(), as.data(), PN}, bp{bq.data(), bs.data(), PN};
  auto check_layout0 = [&](const std::vector<common::ColSource>& cols, bool interleave) {
    std::vector<uint32_t> oq(size_t(RK / 8) * RN);
    std::vector<uint16_t> os(size_t(RK / 64) * RN);
    common::repack_int4_layout0_cols(RK, RN, cols, oq.data(), os.data());
    for (uint32_t n = 0; n < RN; ++n) {
      const bool from_b = interleave ? (n % 32 >= 16) : (n >= PN);
      const uint32_t src_n = interleave ? (n / 32) * 16 + (n % 16) : (n % PN);
      CHECK_EQ(os[n], uint16_t((from_b ? 0x2000u : 0x1000u) + src_n));
      for (uint32_t row = 0; row < RK / 8; ++row)
        CHECK_EQ(oq[size_t(row) * RN + n],
                 (from_b ? 0xB0000000u : 0xA0000000u) | (row << 8) | src_n);
    }
  };
  check_layout0(common::cols_concat({ap, bp}), false);              // Concat
  check_layout0(common::cols_interleave16(ap, bp), true);           // Interleave16
  std::vector<uint32_t> single_q(size_t(RK / 8) * PN);
  std::vector<uint16_t> single_s(PN);
  common::repack_int4_layout0_cols(RK, PN, common::cols_concat({ap}),
                                   single_q.data(), single_s.data());
  CHECK_EQ(single_q, aq);                                           // Single
  CHECK_EQ(single_s, as);
  std::puts("int4_test OK");
  return 0;
}
