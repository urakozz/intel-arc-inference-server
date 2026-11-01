// greedy_tie_test - tests/server/greedy_tie.h (spec 28 N6's tie rule for the speculative greedy gates). Host only.
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"
#include "greedy_tie.h"

using greedy_tie::Verdict;

int main() {
  // ref_step: top-1, runner-up, margin in bf16 ulps of the top logit (2^(e-8) for |top| in [2^(e-1), 2^e)).
  std::vector<float> row = {1.0f, 3.0f, 2.99f, -1.0f, 3.5f};
  greedy_tie::RefStep r = greedy_tie::ref_step(row.data(), 4);   // index 4 is past `used`
  CHECK_EQ(r.top1, 1u);
  CHECK_EQ(r.second, 2u);
  CHECK_NEAR(r.margin_ulps, 0.01 / std::ldexp(1.0, 2 - 8), 1e-4);   // |3| in [2, 4): e = 2, ulp 2^-6
  // ties of equal value: the lower index wins (argmax_stage's order)
  std::vector<float> eq = {5.f, 5.f, 1.f};
  r = greedy_tie::ref_step(eq.data(), 3);
  CHECK_EQ(r.top1, 0u);
  CHECK_EQ(r.second, 1u);
  CHECK_EQ(r.margin_ulps, 0.0);

  // classify
  const std::vector<uint32_t> ref = {7, 8, 9, 10};
  std::vector<greedy_tie::RefStep> steps(4);
  steps[0] = {UINT32_MAX, UINT32_MAX, 1e30};   // the prefill's id: never a tie
  steps[1] = {8, 11, 0.5};
  steps[2] = {9, 12, 3.0};
  steps[3] = {10, 13, 1.0};
  CHECK(greedy_tie::classify(ref, steps, {7, 8, 9, 10}).v == Verdict::Same);
  greedy_tie::Result t = greedy_tie::classify(ref, steps, {7, 11, 4, 4});   // runner-up within 1 ulp: tie at 1
  CHECK(t.v == Verdict::Tie && t.at == 1);
  t = greedy_tie::classify(ref, steps, {7, 12, 9, 10});   // not the runner-up
  CHECK(t.v == Verdict::Diverged && t.at == 1);
  t = greedy_tie::classify(ref, steps, {7, 8, 12, 10});   // the runner-up, but 3 ulps apart
  CHECK(t.v == Verdict::Diverged && t.at == 2);
  t = greedy_tie::classify(ref, steps, {7, 8, 9, 13});   // exactly 1 ulp: still a near-tie
  CHECK(t.v == Verdict::Tie && t.at == 3);
  t = greedy_tie::classify(ref, steps, {6, 8, 9, 10});   // the prefill's id differs: never a tie
  CHECK(t.v == Verdict::Diverged && t.at == 0);
  t = greedy_tie::classify(ref, steps, {7, 8, 9});   // a shorter run
  CHECK(t.v == Verdict::Diverged && t.at == 3);
  std::puts("greedy_tie_test OK");
  return 0;
}
