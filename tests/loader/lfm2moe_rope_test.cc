// Spec 24b Task 3: loader::lfm_rope_table against transformers 5.19.0's own Lfm2MoeRotaryEmbedding
// (tools/oracle/lfm2moe_rope_fixture.py -> lfm2moe_rope_fixture.h) bit for bit at 5 fixed positions
// (0, 1, 2, 4095, 131071) and 1000 random ones, plus the table's size and layout. Host only.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "loader/lfm2moe_repack.h"
#include "loader/lfm2moe_rope_fixture.h"
#include "model/lfm2moe.h"

namespace {
std::vector<uint16_t> hex16(const char* s) {
  std::vector<uint16_t> v;
  const std::string h(s);
  for (size_t i = 0; i + 4 <= h.size(); i += 4) v.push_back(uint16_t(std::stoul(h.substr(i, 4), nullptr, 16)));
  return v;
}
}  // namespace

int main() {
  namespace fx = lfm2moe_rope_fixture;
  const model::Lfm2MoeDesc& d = model::lfm2moe();
  const uint32_t H = d.head_dim / 2;
  CHECK_EQ(H, fx::kHalf);
  const std::vector<uint16_t> cs = hex16(fx::kCos), sn = hex16(fx::kSin);
  CHECK_EQ(cs.size(), size_t(fx::kN) * H);

  // The whole table at a small length: size, position 0 the identity.
  {
    const std::vector<float> t = loader::lfm_rope_table(d, 1024);
    CHECK_EQ(t.size() * 4, size_t(1024) * 2 * H * 4);
    for (uint32_t i = 0; i < H; ++i) CHECK(t[i] == 1.0f && t[H + i] == 0.0f);
    for (size_t j = 0; j < t.size(); j += 7) CHECK(common::bf16_to_f32(common::f32_to_bf16(t[j])) == t[j]);
  }

  // transformers' values, every position of the fixture (one table to the largest).
  uint32_t max_pos = 0;
  for (uint32_t r = 0; r < fx::kN; ++r) max_pos = std::max(max_pos, fx::kPos[r]);
  const std::vector<float> t = loader::lfm_rope_table(d, max_pos + 1);
  size_t bad = 0;
  for (uint32_t r = 0; r < fx::kN; ++r) {
    const uint32_t p = fx::kPos[r];
    for (uint32_t i = 0; i < H; ++i) {
      const uint16_t c = common::f32_to_bf16(t[(size_t(p) * 2 + 0) * H + i]);
      const uint16_t s = common::f32_to_bf16(t[(size_t(p) * 2 + 1) * H + i]);
      if (c != cs[size_t(r) * H + i] || s != sn[size_t(r) * H + i]) {
        if (bad < 8)
          std::fprintf(stderr, "pos %u dim %u: cos %04x sin %04x, transformers %04x %04x\n", p, i, c, s,
                       cs[size_t(r) * H + i], sn[size_t(r) * H + i]);
        ++bad;
      }
    }
  }
  CHECK_EQ(bad, size_t(0));
  std::printf("lfm2moe_rope_test OK: %u positions x %u angles bitwise transformers 5.19.0's\n", fx::kN, H);
  return 0;
}
