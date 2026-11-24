#include <array>
#include <cstdio>
#include <vector>

#include "check.h"
#include "kernels/inkling2_ref.h"

int main() {
  namespace ref = inkling2_ref;
  constexpr size_t H = 2304;
  std::vector<float> bank(2 * H, -1.f), zero(H);
  std::fill_n(bank.begin(), H, 1.f);
  CHECK(ref::depth_mix(bank, zero, 1e-6f) == zero);
  const auto biased = ref::depth_mix(bank, std::vector<float>(H, 1.f), 1e-6f);
  CHECK_NEAR(biased[0], 1.f, 1e-6f);
  CHECK_EQ(ref::controller(2.f, 3.f, 0.f, 0.f), 2.f);

  // Chronological static taps and backward-distance dynamic taps agree on
  // the selected source positions before their different output scales.
  const std::array history{1.f, 2.f, 3.f};
  const std::array current{4.f, 5.f, 6.f, 7.f, 8.f};
  const std::array tap{1.f, 2.f, 3.f, 4.f};
  const std::array reverse{4.f, 3.f, 2.f, 1.f}, dynamic{0.f, 0.f, 0.f, 0.f};
  CHECK_EQ(ref::convolve(current, history, 1, 1, 0, tap), 45.f / std::sqrt(5.f));
  CHECK_EQ(ref::convolve(current, history, 1, 1, 0, reverse, dynamic), 15.f);
  auto committed = history;
  ref::commit_conv(current, committed, 1, 2);
  CHECK_EQ(committed, (std::array{3.f, 4.f, 5.f}));
  ref::commit_conv(current, committed, 1, 5);
  CHECK_EQ(committed, (std::array{6.f, 7.f, 8.f}));

  std::array<float, 128> v{};
  v.fill(2.f);
  ref::head_scale(v, 0.f, 0.f, true);
  for (const float f : v) CHECK_EQ(f, 1.f);
  std::array<float, 16> q{};
  q[3] = 2.f;
  std::vector<float> table(16 * 1024);
  table[3 * 1024 + 1023] = 3.f;
  CHECK_EQ(ref::relative_bias(q, table, 1024, 1023), 6.f);
  CHECK_EQ(ref::relative_bias(q, table, 1024, 1024), 0.f);
  CHECK_EQ(ref::gated_value(12.f, 3.f, 0.f), common::f32_to_bf16(2.f));
  std::puts("Inkling depth mix, controller, convolution, head scale, relative bias and gate checked");
}
