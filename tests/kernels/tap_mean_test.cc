// Plan 21-QFN-v3e Task 1 (host): the taps kernel's host twin (tests/kernels/tap_mean_ref.h, the same arithmetic as
// kernels/tap_mean.cl's tap_mean) on random data - the mean of a row's HC hyper-connection streams, an fp32 sum then
// one bf16 rounding. The card test later runs the kernel against tap_mean_ref bitwise; this checks the twin against
// a double-precision reference (exact bitwise on small-integer data, within one bf16 ulp on random data) and the
// Flash-Next shape's constants (H 2560, HC 4, the five tap layers 5 / 15 / 25 / 35 / 45).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/tap_mean_ref.h"

int main() {
  std::mt19937 rng(42);
  const uint32_t H = 2560, HC = 4;
  // the tap layers the drafter's config names (model::kFlashNextDflashTaps)
  const uint32_t taps[5] = {5, 15, 25, 35, 45};
  for (uint32_t t : taps) CHECK(t < 48);

  // small integers: the fp32 sum is exact, the twin is bitwise the double reference
  {
    const uint32_t rows = 8;
    std::vector<uint16_t> in(size_t(rows) * HC * H), out(size_t(rows) * H);
    for (size_t i = 0; i < in.size(); ++i) in[i] = common::f32_to_bf16(float(int(i % 17) - 8));
    tap_mean_ref::tap_mean(in.data(), out.data(), rows, H, HC);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < H; ++c) {
        double sum = 0;
        for (uint32_t k = 0; k < HC; ++k)
          sum += common::bf16_to_f32(in[(size_t)r * HC * H + size_t(k) * H + c]);
        CHECK_EQ(out[(size_t)r * H + c], common::f32_to_bf16(float(sum / HC)));
      }
    std::puts("small integers: bitwise the double reference");
  }

  // random bf16: within one bf16 ulp of the double reference (the fp32 sum's rounding is the only difference)
  {
    const uint32_t rows = 64;
    std::vector<uint16_t> in(size_t(rows) * HC * H), out(size_t(rows) * H);
    std::uniform_real_distribution<float> u(-30.f, 30.f);
    for (size_t i = 0; i < in.size(); ++i) in[i] = common::f32_to_bf16(u(rng));
    tap_mean_ref::tap_mean(in.data(), out.data(), rows, H, HC);
    uint32_t checked = 0, ulp1 = 0;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < H; ++c) {
        double sum = 0;
        for (uint32_t k = 0; k < HC; ++k)
          sum += common::bf16_to_f32(in[(size_t)r * HC * H + size_t(k) * H + c]);
        const uint16_t w = common::f32_to_bf16(float(sum / HC));
        const uint16_t g = out[(size_t)r * H + c];
        if (g != w) {
          const int64_t d = int64_t(g) - int64_t(w);   // one bf16 ulp apart at most (the fp32 sum's rounding)
          CHECK(d == 1 || d == -1);
          ++ulp1;
        }
        ++checked;
      }
    std::printf("random: %u values, %u one-ulp from the double reference (the fp32 sum's rounding)\n", checked, ulp1);
  }

  // a degenerate row: all four streams equal -> the tap is the value itself
  {
    std::vector<uint16_t> in(size_t(HC) * H), out(H);
    for (uint32_t k = 0; k < HC; ++k)
      for (uint32_t c = 0; c < H; ++c) in[size_t(k) * H + c] = common::f32_to_bf16(1.5f + 0.001f * c);
    tap_mean_ref::tap_mean(in.data(), out.data(), 1, H, HC);
    for (uint32_t c = 0; c < H; ++c) CHECK_EQ(out[c], in[c]);
    std::puts("constant streams: the tap equals the stream");
  }
  std::puts("tap_mean_test OK");
  return 0;
}
