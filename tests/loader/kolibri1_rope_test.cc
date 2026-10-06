// Spec 20c Task 3: loader::kol_rope_table against torch's own cos / sin
// (tools/oracle/kolibri_fixture.py: kolibri_ref.rope_cos_sin, tests/kernels/kolibri_fixture.h)
// bit for bit at positions 0, 1, 513, 100000, 262143 - and the table's size and layout. Host only.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kolibri_fixture.h"
#include "loader/kolibri1_repack.h"
#include "model/kolibri1.h"

namespace {
std::vector<uint16_t> hex16(const char* s) {
  std::vector<uint16_t> v;
  const std::string h(s);
  for (size_t i = 0; i + 4 <= h.size(); i += 4) v.push_back(uint16_t(std::stoul(h.substr(i, 4), nullptr, 16)));
  return v;
}
}  // namespace

int main() {
  namespace fx = kolibri_fixture;
  const model::Kolibri1Desc& d = model::kolibri1();
  const uint32_t H = d.head_dim / 2;
  const std::vector<uint16_t> cs = hex16(fx::kRopeCos), sn = hex16(fx::kRopeSin);
  CHECK_EQ(cs.size(), size_t(fx::kRopeN) * H);
  // The whole table at a small length: size, position 0 the identity.
  {
    const std::vector<float> t = loader::kol_rope_table(d, 1024);
    CHECK_EQ(t.size() * 4, d.rope_table_bytes(1024));
    CHECK_EQ(t.size() * 4, size_t(1024) * 512);
    for (uint32_t i = 0; i < H; ++i) CHECK(t[i] == 1.0f && t[H + i] == 0.0f);
    for (size_t j = 0; j < t.size(); j += 7) CHECK(common::bf16_to_f32(common::f32_to_bf16(t[j])) == t[j]);
  }
  // Torch's values, every position of the fixture (one table to the largest).
  const std::vector<float> t = loader::kol_rope_table(d, fx::kRopePos[fx::kRopeN - 1] + 1);
  size_t bad = 0;
  for (uint32_t r = 0; r < fx::kRopeN; ++r) {
    const uint32_t p = fx::kRopePos[r];
    for (uint32_t i = 0; i < H; ++i) {
      const uint16_t c = common::f32_to_bf16(t[(size_t(p) * 2 + 0) * H + i]);
      const uint16_t s = common::f32_to_bf16(t[(size_t(p) * 2 + 1) * H + i]);
      if (c != cs[size_t(r) * H + i] || s != sn[size_t(r) * H + i]) {
        if (bad < 8)
          std::fprintf(stderr, "pos %u dim %u: cos %04x sin %04x, torch %04x %04x\n", p, i, c, s,
                       cs[size_t(r) * H + i], sn[size_t(r) * H + i]);
        ++bad;
      }
    }
  }
  CHECK_EQ(bad, size_t(0));
  std::printf("kolibri1_rope_test OK: %u positions x %u angles bitwise torch's\n", fx::kRopeN, H);
  return 0;
}
