// Open-work 13: the per-step draft choice (server::StepK, server::DraftCalib), host only - the calibration's prior and
// learning, the choice from given probabilities and costs (draft on, stop, verify fewer than drafted), the K = 0 run
// and its probe, the call-order checks, and on a synthetic stream whose drafts are accepted with the head's stated
// probability: more ids per plain-step cost than the per-request policy (server::AdaptiveK) on the same stream.
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "server/step_k.h"

namespace {

using server::AdaptiveK;
using server::AdaptiveKOptions;
using server::DraftCalib;
using server::MtpCost;
using server::StepK;
using server::StepKOptions;

StepKOptions opts(const MtpCost& c = MtpCost::int8_head()) {
  StepKOptions o;
  o.cost = c;
  return o;
}

void test_calib() {
  CHECK_EQ(DraftCalib::bin(0.0f), 0u);
  CHECK_EQ(DraftCalib::bin(0.3f), 1u);
  CHECK_EQ(DraftCalib::bin(0.97f), 6u);
  CHECK_EQ(DraftCalib::bin(1.0f), 8u);
  DraftCalib c;
  CHECK_NEAR(c.rate(0, 0.6f), 0.6, 1e-6);   // no data: the stated probability
  for (int n = 0; n < 400; ++n) c.observe(0, 0.6f, n % 10 != 0);   // 90 % accepted at p 0.6
  CHECK(c.rate(0, 0.65f) > 0.85);
  CHECK_NEAR(c.rate(1, 0.6f), 0.6, 1e-6);   // positions are separate
  CHECK_NEAR(c.rate(0, 0.95f), 0.95, 1e-6);   // so are bins
}

void test_choice() {
  DraftCalib c;
  StepK s(opts(), c);
  CHECK_EQ(s.max_k(), 3u);
  // Fresh (the prior 0.8 at every position): drafting pays, up to 3.
  CHECK_EQ(s.begin(), 3u);
  CHECK_EQ(s.choose(1, 0.99f), -1);
  CHECK_EQ(s.choose(2, 0.99f), -1);
  CHECK_EQ(s.choose(3, 0.99f), 3);   // the cap always answers
  s.observe(3);
  // An unsure first draft stops the chain: verify it (verify[1] is nearly free), draft no more.
  CHECK_EQ(s.begin(), 3u);
  CHECK_EQ(s.choose(1, 0.05f), 1);
  s.observe(0);
  // A sure first draft and a hopeless second: verify only the first of the two made.
  CHECK_EQ(s.begin(), 3u);
  CHECK_EQ(s.choose(1, 0.99f), -1);
  CHECK_EQ(s.choose(2, 0.02f), 1);
  CHECK_EQ(s.verified(), 1u);
  s.observe(1);
  // The engine stopped drafting before an answer (max_len): every draft made counts as verified.
  CHECK_EQ(s.begin(), 3u);
  CHECK_EQ(s.choose(1, 0.99f), -1);
  s.observe(1);
  CHECK_EQ(s.verified(), 1u);
}

void test_low_acceptance_and_probe() {
  DraftCalib c;
  StepKOptions o = opts();
  StepK s(o, c);
  // Every first draft rejected: the request's rate at position 0 falls until drafting does not pay.
  uint32_t plain_at = 0;
  for (uint32_t it = 1; it <= 40 && plain_at == 0; ++it) {
    const uint32_t cap = s.begin();
    if (cap == 0) {
      plain_at = it;
      break;
    }
    const int v = s.choose(1, 0.5f);
    if (v < 0) {
      for (uint32_t j = 2; j <= cap; ++j)
        if (s.choose(j, 0.5f) >= 0) break;
    }
    s.observe(0);
  }
  CHECK(plain_at > 0);
  CHECK(s.position_rate(0) < 0.2);
  s.observe(0);   // the plain iteration
  // probe_every plain iterations, then one that drafts one.
  for (uint32_t n = 1; n < o.probe_every; ++n) {
    CHECK_EQ(s.begin(), 0u);
    s.observe(0);
  }
  CHECK_EQ(s.begin(), 1u);
  CHECK(s.choose(1, 0.5f) >= 0);
  s.observe(0);
  CHECK_EQ(s.begin(), 0u);
}

void test_order_checks() {
  DraftCalib c;
  StepK s(opts(), c);
  s.begin();
  bool threw = false;
  try {
    s.choose(2, 0.5f);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
  CHECK_EQ(s.choose(1, 0.5f) >= -1, true);
  threw = false;
  try {
    s.observe(3);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

// A text whose every position carries the head's probability for drafting it, which is also the chance the main
// model accepts that draft: stretches of sure positions (code-like, p in [0.85, 1]) and unsure ones (p in [0, 0.6]),
// ~20 positions each. An iteration at position t drafts positions t, t + 1, ... and moves t on by the ids it keeps.
// Returns ids per plain-step cost.
struct Text {
  std::vector<float> p;
  std::mt19937_64 rng;
  Text(uint64_t seed, size_t n) : rng(seed + 1) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    bool sure = true;
    for (size_t i = 0; i < n; ++i) {
      if (u(g) < 0.05) sure = !sure;
      p.push_back(static_cast<float>(sure ? 0.85 + 0.15 * u(g) : 0.6 * u(g)));
    }
  }
  bool accept(float q) { return std::uniform_real_distribution<double>(0.0, 1.0)(rng) < q; }
};

double run_step_k(const MtpCost& cost, uint64_t seed, size_t n) {
  DraftCalib c;
  StepK s(opts(cost), c);
  Text text(seed, n + 8);
  double ids = 0, spent = 0;
  for (size_t t = 0; t < n;) {
    const uint32_t cap = s.begin();
    uint32_t drafted = 0, v = 0;
    for (uint32_t j = 1; j <= cap; ++j) {
      drafted = j;
      const int r = s.choose(j, text.p[t + j - 1]);
      if (r >= 0) {
        v = static_cast<uint32_t>(r);
        break;
      }
    }
    uint32_t acc = 0;
    while (acc < v && text.accept(text.p[t + acc])) ++acc;
    s.observe(acc);
    ids += acc + 1;
    spent += cost.verify[v] + (drafted > 0 ? cost.draft[drafted - 1] : 0.0);
    t += acc + 1;
  }
  return ids / spent;
}

double run_adaptive(const MtpCost& cost, uint64_t seed, size_t n) {
  AdaptiveKOptions o;
  o.cost = cost;
  AdaptiveK a(o);
  Text text(seed, n + 8);
  double ids = 0, spent = 0;
  for (size_t t = 0; t < n;) {
    const uint32_t k = a.next();
    uint32_t acc = 0;
    while (acc < k && text.accept(text.p[t + acc])) ++acc;
    a.observe(k, acc);
    ids += acc + 1;
    spent += cost.cost(k);
    t += acc + 1;
  }
  return ids / spent;
}

void test_beats_per_request() {
  for (const MtpCost& cost : {MtpCost::int8_head(), MtpCost::bf16_head(), MtpCost::tp_int8_head()}) {
    const double step = run_step_k(cost, 7, 60000), req = run_adaptive(cost, 7, 60000);
    std::printf("synthetic stream: per step %.3f ids per plain step, per request %.3f (%+.1f %%)\n", step, req,
                100.0 * (step / req - 1.0));
    CHECK(step > req);
  }
}

}  // namespace

int main() {
  test_calib();
  test_choice();
  test_low_acceptance_and_probe();
  test_order_checks();
  test_beats_per_request();
  std::printf("step_k_test: all cases passed\n");
  return 0;
}
