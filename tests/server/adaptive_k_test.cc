// Spec 8 §10: the --mtp auto policy (server::AdaptiveK) on synthetic acceptance sequences,
// host only: the cost model, the cost table's parser, K up on high acceptance, down to
// 0 / 1 on low, the K = 0 probe, hysteresis against flapping, determinism.
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "server/adaptive_k.h"

namespace {

using server::AdaptiveK;
using server::AdaptiveKOptions;
using server::MtpCost;

// Drives a policy for `n` iterations; `accept(k)` says how many of k drafts were kept.
// Records the K run, the policy's K after each observation, and ids / cost in the model.
struct Trace {
  std::vector<uint32_t> ks, policy;
  double ids = 0, cost = 0;
  double efficiency() const { return ids / cost; }
};
template <class Accept>
Trace drive_trace(AdaptiveK& p, size_t n, Accept accept, const MtpCost& cost = MtpCost{}) {
  Trace t;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t k = p.next();
    CHECK(k <= p.max_k());
    const uint32_t j = accept(k);
    t.ks.push_back(k);
    t.ids += j + 1;
    t.cost += cost.cost(k);
    p.observe(k, j);
    t.policy.push_back(p.current());
  }
  return t;
}
template <class Accept>
std::vector<uint32_t> drive(AdaptiveK& p, size_t n, Accept accept) {
  return drive_trace(p, n, accept).ks;
}

// Per-draft Bernoulli(alpha), drafts tried in order until the first rejection.
struct Geometric {
  std::mt19937_64 rng;
  double alpha;
  Geometric(uint64_t seed, double a) : rng(seed), alpha(a) {}
  uint32_t operator()(uint32_t k) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    uint32_t j = 0;
    while (j < k && u(rng) < alpha) ++j;
    return j;
  }
};

size_t switches(const std::vector<uint32_t>& ks, size_t from) {
  size_t n = 0;
  for (size_t i = from + 1; i < ks.size(); ++i) n += ks[i] != ks[i - 1];
  return n;
}

void test_model() {
  CHECK_NEAR(AdaptiveK::expected_ids(0.5, 1), 1.5, 1e-12);
  CHECK_NEAR(AdaptiveK::expected_ids(0.5, 3), 1.875, 1e-12);
  CHECK_NEAR(AdaptiveK::expected_ids(1.0, 3), 4.0, 1e-12);
  CHECK_NEAR(AdaptiveK::expected_ids(0.0, 3), 1.0, 1e-12);
  AdaptiveKOptions o;
  o.cost = MtpCost::bf16_head();
  const AdaptiveK bf(o);
  CHECK_NEAR(bf.rate(0.8, 0), 1.0, 1e-12);
  CHECK_NEAR(bf.rate(0.8, 1), 1.8 / 1.36, 1e-12);
  CHECK_NEAR(bf.rate(0.97, 3), AdaptiveK::expected_ids(0.97, 3) / 2.29, 1e-12);
  // The boundaries of spec 8 §10: bf16 0 | 0.36 | 1 | 0.83 | 3.
  CHECK_EQ(bf.best(0.20), 0U);
  CHECK_EQ(bf.best(0.35), 0U);
  CHECK_EQ(bf.best(0.37), 1U);
  CHECK_EQ(bf.best(0.44), 1U);   // prose
  CHECK_EQ(bf.best(0.80), 1U);
  CHECK_EQ(bf.best(0.84), 3U);
  CHECK_EQ(bf.best(0.97), 3U);   // tool-call output
  o.cost = MtpCost::int8_head();
  const AdaptiveK i8(o);
  CHECK_EQ(i8.best(0.30), 0U);
  CHECK_EQ(i8.best(0.32), 1U);
  CHECK_EQ(i8.best(0.80), 1U);
  CHECK_EQ(i8.best(0.82), 3U);
  for (int i = 0; i <= 1000; ++i) {   // K = 2 is never the best on either table
    CHECK(bf.best(i / 1000.0) != 2U);
    CHECK(i8.best(i / 1000.0) != 2U);
  }
  std::printf("model: E(K), rates and the K boundaries per table\n");
}

void test_cost_parse() {
  const MtpCost base = MtpCost::bf16_head();
  const MtpCost c = MtpCost::parse("verify=1,1.2,1.5,1.8;draft=0.1,0.2,0.3", base);
  CHECK_EQ(c.verify.size(), size_t(4));
  CHECK_NEAR(c.verify[3], 1.8, 1e-12);
  CHECK_NEAR(c.draft[1], 0.2, 1e-12);
  CHECK_NEAR(c.cost(0), 1.0, 1e-12);
  CHECK_NEAR(c.cost(2), 1.5 + 0.2, 1e-12);
  CHECK_EQ(c.max_k(), 3U);
  const MtpCost d = MtpCost::parse("draft=0.12,0.25", base);   // verify kept from base
  CHECK_NEAR(d.verify[1], 1.17, 1e-12);
  CHECK_EQ(d.max_k(), 2U);
  AdaptiveKOptions o;
  o.cost = d;
  CHECK_EQ(AdaptiveK(o).max_k(), 2U);   // bounded by the table
  o.max_k = 1;
  CHECK_EQ(AdaptiveK(o).max_k(), 1U);
  for (const char* bad : {"", "verify=", "foo=1", "verify=1,x", "draft=0.1,-0.2", "verify=1;",
                          "draft=0.1;draft=0.2", "verify=0,1.2"}) {
    bool threw = false;
    try {
      MtpCost::parse(bad, base);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) {
      std::fprintf(stderr, "MtpCost::parse accepted '%s'\n", bad);
      std::exit(1);
    }
  }
  bool threw = false;
  try {
    AdaptiveK p{AdaptiveKOptions{}};
    p.observe(2, 3);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("cost table: parse, partial override, max K bounded by the table\n");
}

void test_up_on_high_acceptance() {
  AdaptiveK p{AdaptiveKOptions{}};
  const auto ks = drive(p, 60, [](uint32_t k) { return k; });   // every draft kept
  for (uint32_t k : ks) CHECK_EQ(k, 3U);
  CHECK(p.alpha() > 0.99);
  // Tool-call-like output, alpha 0.97 per draft: K = 3 after the warm-up, every iteration.
  AdaptiveK q{AdaptiveKOptions{}};
  Geometric g(1, 0.97);
  const auto kq = drive(q, 300, g);
  size_t at3 = 0;
  for (uint32_t k : kq) at3 += k == 3;
  CHECK(at3 >= 295);
  std::printf("high acceptance: K = 3 throughout (alpha 0.97: %zu/300 at K = 3)\n", at3);
}

void test_down_on_low_acceptance() {
  AdaptiveK p{AdaptiveKOptions{}};
  const auto ks = drive(p, 120, [](uint32_t) { return 0U; });   // every first draft rejected
  CHECK_EQ(ks[0], 3U);
  CHECK_EQ(ks[1], 3U);   // the warm-up
  size_t first0 = ks.size();
  for (size_t i = 0; i < ks.size(); ++i) {
    if (ks[i] == 0) {
      first0 = i;
      break;
    }
  }
  CHECK(first0 <= 12);
  // At K = 0: one K = 1 probe after every 4 plain iterations, nothing deeper.
  for (size_t i = first0; i < ks.size(); ++i) CHECK(ks[i] <= 1);
  for (size_t i = first0; i + 4 < ks.size(); ++i) {
    if (ks[i] != 1) continue;
    CHECK(ks[i + 1] == 0 && ks[i + 2] == 0 && ks[i + 3] == 0 && ks[i + 4] == 0);
  }
  // Prose-like output, alpha 0.44 (the K = 0 / 1 boundary is 0.36): only K 0 and 1 once
  // settled, and ids per cost within 5 % of the best fixed K's.
  AdaptiveKOptions o;
  for (uint64_t seed = 1; seed <= 5; ++seed) {
    AdaptiveK q{o};
    Geometric g(seed, 0.44);
    const Trace t = drive_trace(q, 400, g);
    for (size_t i = 20; i < t.ks.size(); ++i) CHECK(t.ks[i] <= 1);
    CHECK(t.efficiency() >= 0.95 * q.rate(0.44, 1));
  }
  std::printf("low acceptance: K = 0 by iteration %zu with a K = 1 probe every 5th; "
              "alpha 0.44: K in {0, 1}, within 5 %% of the best rate\n", first0);
}

void test_recovers() {
  AdaptiveK p{AdaptiveKOptions{}};
  drive(p, 40, [](uint32_t) { return 0U; });
  CHECK_EQ(p.current(), 0U);
  const auto ks = drive(p, 60, [](uint32_t k) { return k; });   // acceptance comes back
  size_t back = ks.size();
  for (size_t i = 0; i < ks.size(); ++i) {
    if (ks[i] == 3) {
      back = i;
      break;
    }
  }
  CHECK(back < 40);
  for (size_t i = back; i < ks.size(); ++i) CHECK_EQ(ks[i], 3U);
  std::printf("recovery: from K = 0 back to K = 3 in %zu iterations once drafts are kept\n", back);
}

void test_hysteresis() {
  // alpha at the K = 1 / 3 boundary (0.83 on bf16): the default policy must not flap.
  AdaptiveKOptions jumpy;
  jumpy.hysteresis = 0;
  jumpy.decay = 0.3;
  size_t calm_total = 0, jumpy_total = 0;
  for (uint64_t seed = 1; seed <= 5; ++seed) {
    AdaptiveK calm{AdaptiveKOptions{}}, flappy{jumpy};
    Geometric g1(seed, 0.83), g2(seed, 0.83);
    const Trace a = drive_trace(calm, 400, g1);
    const Trace b = drive_trace(flappy, 400, g2);
    calm_total += switches(a.policy, 2);
    jumpy_total += switches(b.policy, 2);
    CHECK(switches(a.policy, 2) <= 40);   // at most one switch per 10 iterations
    CHECK(a.efficiency() >= 0.97 * calm.rate(0.83, calm.best(0.83)));
  }
  std::printf("hysteresis at alpha 0.83: %zu switches in 5 x 400 iterations "
              "(no hysteresis, short window: %zu)\n", calm_total, jumpy_total);
  CHECK(calm_total * 3 <= jumpy_total);
}

void test_deterministic() {
  AdaptiveK a{AdaptiveKOptions{}}, b{AdaptiveKOptions{}};
  Geometric g1(9, 0.7), g2(9, 0.7);
  CHECK(drive(a, 500, g1) == drive(b, 500, g2));
  std::printf("determinism: the same history gives the same K sequence\n");
}

}  // namespace

int main() {
  test_model();
  test_cost_parse();
  test_up_on_high_acceptance();
  test_down_on_low_acceptance();
  test_recovers();
  test_hysteresis();
  test_deterministic();
  std::printf("adaptive_k_test: all cases passed\n");
  return 0;
}
