// Spec 8 §10: the --mtp auto policy (server::AdaptiveK) on synthetic acceptance sequences,
// host only: the cost model, the cost table's parser, K up on high acceptance, down to
// 0 / 1 on low, the K = 0 probe, hysteresis against flapping, determinism; and the cost
// table under a draft vocabulary (spec 8 §11, MtpCost::with_draft_vocab).
#include <cstdio>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "runtime/inkling2/inkling2_window.h"
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
  CHECK_NEAR(bf.rate(0.8, 1), 1.8 / (1.04 + 0.19), 1e-12);
  CHECK_NEAR(bf.rate(0.97, 3), AdaptiveK::expected_ids(0.97, 3) / (1.13 + 0.55), 1e-12);
  // The boundaries on spec 28b's measured verify rows (flat: 1.04 / 1.09 / 1.13): bf16 0 | 0.231 | 1 | 0.536 | 2 |
  // 0.689 | 3 (spec 8 §10's, on the old rows, were 0 | 0.36 | 1 | 0.83 | 3 with K = 2 never the best).
  CHECK_EQ(bf.best(0.20), 0U);
  CHECK_EQ(bf.best(0.24), 1U);
  CHECK_EQ(bf.best(0.44), 1U);   // prose
  CHECK_EQ(bf.best(0.53), 1U);
  CHECK_EQ(bf.best(0.55), 2U);
  CHECK_EQ(bf.best(0.68), 2U);
  CHECK_EQ(bf.best(0.70), 3U);
  CHECK_EQ(bf.best(0.97), 3U);   // tool-call output
  o.cost = MtpCost::int8_head();
  // int8 on spec 19 v3d's rows (1.04 / 1.07 / 1.10): 0 | 0.170 | 1 | 0.445 | 2 | 0.607 | 3 (on 28b's 1.04 / 1.09 /
  // 1.13: 0 | 0.171 | 1 | 0.477 | 2 | 0.620 | 3)
  const AdaptiveK i8(o);
  CHECK_EQ(i8.best(0.16), 0U);
  CHECK_EQ(i8.best(0.18), 1U);
  CHECK_EQ(i8.best(0.43), 1U);
  CHECK_EQ(i8.best(0.46), 2U);
  CHECK_EQ(i8.best(0.60), 2U);
  CHECK_EQ(i8.best(0.62), 3U);
  CHECK_EQ(i8.best(0.97), 3U);
  std::print("model: E(K), rates and the K boundaries per table\n");
}

void test_dflash_measured_cost() {
  const MtpCost c = MtpCost::dflash(true, true);
  const double verify[] = {1.000091, 1.048509, 1.102680, 1.123232, 1.170806, 1.199997, 1.228823, 1.246703};
  CHECK(c.verify.size() == 8 && c.draft.size() == 7);
  for (size_t i = 0; i < 8; ++i) CHECK_NEAR(c.verify[i], verify[i], 1e-12);
  for (double d : c.draft) CHECK_NEAR(d, 0.298136, 1e-12);
  CHECK_NEAR(c.draft[0] * MtpCost::dflash_head_share(true, true), 2.166354 / 29.222498, 1e-12);
  const auto candidate = MtpCost::dflash(true, 1u, true);
  const double verify_candidate[] = {1.000230, 1.041410, 1.073907, 1.105789, 1.151472, 1.180781, 1.205189, 1.221471};
  for (size_t i = 0; i < 8; ++i) CHECK_NEAR(candidate.verify[i], verify_candidate[i], 1e-12);
  for (double d : candidate.draft) CHECK_NEAR(d, 0.217462, 1e-12);
  const auto int4 = MtpCost::dflash(true, 2u);
  CHECK(int4.draft[0] < c.draft[0]);
  CHECK(int4.draft[0] != MtpCost::dflash(true, 0u).draft[0]);
  AdaptiveKOptions o;
  o.cost = c;
  o.max_k = 7;
  const AdaptiveK p(o);
  CHECK(p.best(0.5) == 3);
  CHECK(p.best(0.8) == 7);
  CHECK(p.best(0.95) == 7);
  std::print("DFlash measured costs: K 3 / 7 / 7 at alpha 0.5 / 0.8 / 0.95\n");
}

void test_flash_next_cost() {
  AdaptiveKOptions o;
  o.cost = MtpCost::flash_next();
  const AdaptiveK p(o);
  CHECK_EQ(p.max_k(), 3U);
  CHECK_EQ(p.best(0.66), 0U);   // a short story's acceptance: plain steps
  CHECK_EQ(p.best(0.88), 3U);   // code
  CHECK_EQ(p.best(0.97), 3U);   // tool calls
  std::print("Flash-Next costs: K 0 / 3 / 3 at alpha 0.66 / 0.88 / 0.97\n");
  o.cost = MtpCost::tp_flash_next();   // tp: cheaper verify rows - MTP pays from lower acceptance
  const AdaptiveK t(o);
  std::print("Flash-Next tp costs: K {} / {} / {} at alpha 0.5 / 0.66 / 0.88\n", t.best(0.5), t.best(0.66), t.best(0.88));
  CHECK(t.best(0.66) >= p.best(0.66));
  CHECK_EQ(t.best(0.88), 3U);
}

// The request's own costs replace the table's once it has run both K and plain steps, and a probe that leaves K at 0
// waits twice as long for the next: prose whose K = 1 costs 1.9 plain steps at acceptance 0.62 stays plain, its probes
// spreading out (the table said 1.58 - K 1 would pay).
void test_measured_cost() {
  AdaptiveKOptions o;
  o.cost = MtpCost::tp_flash_next();
  o.max_k = 1;
  AdaptiveK p(o);
  CHECK_EQ(p.best(0.62), 1U);   // the table's choice
  Geometric g(7, 0.62);
  std::vector<uint32_t> ks;
  for (int i = 0; i < 400; ++i) {
    const uint32_t k = p.next();
    p.observe(k, g(k), k ? 0.019 * 1.9 : 0.019);   // K 1 measured at 1.9 plain steps
    ks.push_back(k);
  }
  CHECK_NEAR(p.cost(1), 1.9, 1e-9);
  CHECK_EQ(p.best(0.62), 0U);
  size_t probes = 0;
  for (size_t i = 200; i < ks.size(); ++i) probes += ks[i];
  CHECK(probes <= 200 / 32 + 2);   // the gap at its cap
  std::print("measured costs: K 1 at 1.9 plain steps, acceptance 0.62: plain, {} probes in the last 200 iterations\n",
             probes);
}

// Plan 41f: Inkling's served policy (b70-serve's options) at its measured table - an untimed request's rejected
// probes spread out toward the cap at low acceptance, and high acceptance still takes K up.
void test_inkling_policy() {
  AdaptiveKOptions o;
  o.cost = MtpCost::inkling_dspark();
  o.max_k = 7;
  o.probe_max = 128;
  o.probe_backoff = true;
  AdaptiveK low(o);
  Geometric g(11, 0.16);
  size_t probes = 0;
  for (int i = 0; i < 256; ++i) {
    const uint32_t k = low.next();
    low.observe(k, g(k));
    probes += i >= 2 && k > 0;
  }
  CHECK_EQ(low.current(), 0U);
  CHECK(probes <= 14);   // a rejected probe doubles the gap (4 .. 128), a kept one resets it; 50 without the backoff
  AdaptiveK high(o);
  Geometric h(13, 0.9);
  for (int i = 0; i < 128; ++i) {
    const uint32_t k = high.next();
    high.observe(k, h(k));
  }
  CHECK(high.current() >= 3U);   // at 0.9 the table's best is K 7 (2.51 ids a plain step)
  std::print("Inkling: acceptance 0.16 plain with {} probes in 254 iterations; at 0.9 K {}\n", probes, high.current());
}

// b70-serve's Inkling table and b70-decode's draft window read the same measured costs (written in both places).
void test_inkling_tables_equal() {
  // DSpark's and (plan 41k) the DFlash 2 drafter's
  for (const auto& [c, w] : {std::pair{MtpCost::inkling_dspark(), runtime::inkling2::kOneCardCost},
                             std::pair{MtpCost::inkling_dflash2(), runtime::inkling2::kOneCardDflashCost},
                             std::pair{MtpCost::inkling_dflash2_tp(), runtime::inkling2::kTpDflashCost}}) {
    CHECK_EQ(c.verify.size(), w.verify.size());
    for (size_t i = 0; i < w.verify.size(); ++i) CHECK_EQ(c.verify[i], w.verify[i]);
    CHECK_EQ(c.draft.size(), size_t(runtime::inkling2::DraftWindow::kMaxDrafts));
    for (const double d : c.draft) CHECK_EQ(d, w.draft);
  }
}

// Costs by depth: the draft window's table interpolated between its depths (the ends held), and AdaptiveK's K from the
// table at the iteration's position - drafting where drafts are cheap, plain where the drafter's pass costs 5 steps.
void test_depth_costs() {
  namespace ink = runtime::inkling2;
  const auto& t = ink::kPpCosts;
  CHECK_EQ(ink::cost_at(t, 0).draft, t.front().cost.draft);
  CHECK_EQ(ink::cost_at(t, 1u << 30).draft, t.back().cost.draft);
  const uint32_t mid = (t[0].depth + t[1].depth) / 2;
  CHECK_NEAR(ink::cost_at(t, mid).draft, (t[0].cost.draft + t[1].cost.draft) / 2, 1e-12);
  CHECK_NEAR(ink::cost_at(t, mid).verify[7], (t[0].cost.verify[7] + t[1].cost.verify[7]) / 2, 1e-12);

  AdaptiveKOptions o;
  o.max_k = 3;
  o.warmup = 0;
  o.probe_every = 0;
  MtpCost cheap, dear;
  cheap.verify = dear.verify = {1.0, 1.1, 1.2, 1.3};
  cheap.draft = {0.1, 0.1, 0.1};
  dear.draft = {5.0, 5.0, 5.0};
  o.cost = cheap;
  o.by_depth = {{1000, cheap}, {100000, dear}};
  AdaptiveK k(o);
  (void)k.next(200000);
  k.observe(3, 3);
  CHECK_EQ(k.current(), 0u);
  CHECK_EQ(k.next(500), 0u);
  k.observe(0, 0);
  CHECK(k.current() > 0);
}

void test_cost_table() {
  MtpCost c;
  c.verify = {1, 1.2, 1.5, 1.8};
  c.draft = {0.1, 0.2, 0.3};
  CHECK_NEAR(c.cost(0), 1.0, 1e-12);
  CHECK_NEAR(c.cost(2), 1.5 + 0.2, 1e-12);
  CHECK_EQ(c.max_k(), 3U);
  MtpCost d = MtpCost::bf16_head();
  d.draft = {0.12, 0.25};   // a shorter draft row bounds K
  CHECK_NEAR(d.verify[1], 1.04, 1e-12);
  CHECK_EQ(d.max_k(), 2U);
  AdaptiveKOptions o;
  o.cost = d;
  CHECK_EQ(AdaptiveK(o).max_k(), 2U);   // bounded by the table
  o.max_k = 1;
  CHECK_EQ(AdaptiveK(o).max_k(), 1U);
  bool threw = false;
  try {
    AdaptiveK p{AdaptiveKOptions{}};
    p.observe(2, 3);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::print("cost table: parse, partial override, max K bounded by the table\n");
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
  std::print("high acceptance: K = 3 throughout (alpha 0.97: {:d}/300 at K = 3)\n", at3);
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
  // Prose-like output, alpha 0.44 (K = 1 is the best from 0.231 to 0.536 on spec 28b's flat bf16 rows, K = 2 within
  // 5 % of it): never K = 3 once settled, and ids per cost within 5 % of the best fixed K's.
  AdaptiveKOptions o;
  for (uint64_t seed = 1; seed <= 5; ++seed) {
    AdaptiveK q{o};
    Geometric g(seed, 0.44);
    const Trace t = drive_trace(q, 400, g);
    for (size_t i = 20; i < t.ks.size(); ++i) CHECK(t.ks[i] <= 2);
    CHECK(t.efficiency() >= 0.95 * q.rate(0.44, 1));
  }
  std::print("low acceptance: K = 0 by iteration {:d} with a K = 1 probe every 5th; "
             "alpha 0.44: K in {{0, 1, 2}}, within 5 % of the best rate\n", first0);
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
  std::print("recovery: from K = 0 back to K = 3 in {:d} iterations once drafts are kept\n", back);
}

void test_hysteresis() {
  // alpha at the K = 2 / 3 boundary (0.689 on spec 28b's bf16 rows; 0.83, K = 1 / 3, before): the default policy
  // must not flap.
  AdaptiveKOptions jumpy;
  jumpy.hysteresis = 0;
  jumpy.decay = 0.3;
  size_t calm_total = 0, jumpy_total = 0;
  for (uint64_t seed = 1; seed <= 5; ++seed) {
    AdaptiveK calm{AdaptiveKOptions{}}, flappy{jumpy};
    Geometric g1(seed, 0.689), g2(seed, 0.689);
    const Trace a = drive_trace(calm, 400, g1);
    const Trace b = drive_trace(flappy, 400, g2);
    calm_total += switches(a.policy, 2);
    jumpy_total += switches(b.policy, 2);
    CHECK(switches(a.policy, 2) <= 40);   // at most one switch per 10 iterations
    CHECK(a.efficiency() >= 0.97 * calm.rate(0.689, calm.best(0.689)));
  }
  std::print("hysteresis at alpha 0.689: {:d} switches in 5 x 400 iterations "
             "(no hysteresis, short window: {:d})\n", calm_total, jumpy_total);
  CHECK(calm_total * 3 <= jumpy_total);
}

void test_deterministic() {
  AdaptiveK a{AdaptiveKOptions{}}, b{AdaptiveKOptions{}};
  Geometric g1(9, 0.7), g2(9, 0.7);
  CHECK(drive(a, 500, g1) == drive(b, 500, g2));
  std::print("determinism: the same history gives the same K sequence\n");
}

// Spec 8 §11: the cost table under a draft vocabulary - each draft's lm_head share scales
// with |V'| / 248320, the rest of a draft and every verify stay (derived numbers).
void test_draft_vocab_cost() {
  const MtpCost i8 = MtpCost::int8_head(), bf = MtpCost::bf16_head();
  const double h8 = MtpCost::kInt8DraftHeadShare, h16 = MtpCost::kBf16DraftHeadShare;
  CHECK_NEAR(h8 * i8.draft[0], 0.067, 1e-12);   // the head's part of one int8 draft
  CHECK_NEAR(h16 * bf.draft[0], 0.129, 1e-12);
  const double want8[3] = {0.0718, 0.0807, 0.0984};   // 32k / 64k / 128k, one draft
  const uint32_t sizes[3] = {32768, 65536, 131072};
  for (int s = 0; s < 3; ++s) {
    const double f = sizes[s] / 248320.0;
    const MtpCost c = i8.with_draft_vocab(h8, f), b = bf.with_draft_vocab(h16, f);
    CHECK_NEAR(c.draft[0], want8[s], 5e-4);
    for (size_t k = 0; k < 3; ++k) {
      CHECK_NEAR(c.draft[k] / i8.draft[k], c.draft[0] / i8.draft[0], 1e-12);   // one factor
      CHECK(c.draft[k] < i8.draft[k] && b.draft[k] < bf.draft[k]);
    }
    CHECK(c.verify == i8.verify && b.verify == bf.verify);   // verify reads the full head
    std::print("draft vocab {:6d}: int8 drafts {:.4f} {:.4f} {:.4f}, bf16 drafts {:.4f} {:.4f} {:.4f}\n",
               sizes[s], c.draft[0], c.draft[1], c.draft[2], b.draft[0], b.draft[1], b.draft[2]);
  }
  // Off (fraction 1) is the table itself; a share of 0 changes nothing either.
  CHECK(i8.with_draft_vocab(h8, 1.0).draft == i8.draft);
  CHECK(bf.with_draft_vocab(0.0, 0.5).draft == bf.draft);
  for (double bad_f : {0.0, -0.1, 1.5}) {
    bool threw = false;
    try {
      (void)i8.with_draft_vocab(h8, bad_f);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
}

}  // namespace

// Plan 40a Task 5: the scheduler's deal over the row-step costs (server::deal_by_cost, server::RowCost).
void test_deal_by_cost() {
  const server::RowCost df{true}, mtp{false};
  const auto cost = [](const server::RowCost& c) { return [&c](const std::vector<uint32_t>& k) { return c.step(k); }; };
  const auto rows = [](const std::vector<uint32_t>& k) {
    uint32_t n = 0;
    for (uint32_t x : k) n += x + 1;
    return n;
  };
  // four requests at high acceptance, K 7 asked: the 32-row step, K 7 each
  CHECK((server::deal_by_cost({0.85, 0.85, 0.85, 0.85}, {7, 7, 7, 7}, 32, cost(df)) == std::vector<uint32_t>{7, 7, 7, 7}));
  // one at high acceptance keeps 7 while three at low get fewer - a 16-row step; every row low: plain rows
  const std::vector<uint32_t> low = server::deal_by_cost({0.85, 0.2, 0.2, 0.2}, {7, 7, 7, 7}, 32, cost(df));
  CHECK(low[0] == 7 && low[1] < 7 && low[2] < 7 && low[3] < 7 && rows(low) <= 16);
  CHECK((server::deal_by_cost({0.3, 0.3, 0.3, 0.3}, {7, 7, 7, 7}, 32, cost(df)) == std::vector<uint32_t>{0, 0, 0, 0}));
  // three at K 7: 24 rows (the 32-row list) where their acceptance pays for it, else at most the 16-row list's
  CHECK((server::deal_by_cost({0.85, 0.85, 0.85}, {7, 7, 7}, 32, cost(df)) == std::vector<uint32_t>{7, 7, 7}));
  const std::vector<uint32_t> mid = server::deal_by_cost({0.6, 0.6, 0.6}, {7, 7, 7}, 32, cost(df));
  CHECK(rows(mid) <= 16 && rows(mid) > 8);
  // the 8-row budget: the drafts stay inside it
  CHECK(rows(server::deal_by_cost({0.85, 0.85, 0.85, 0.85}, {7, 7, 7, 7}, 8, cost(df))) <= 8);
  // MTP (a draft index a head step): four at 0.8 asking K 3 - the 16-row list, K 3 each
  CHECK((server::deal_by_cost({0.8, 0.8, 0.8, 0.8}, {3, 3, 3, 3}, 32, cost(mtp)) == std::vector<uint32_t>{3, 3, 3, 3}));
  // nothing asked: plain rows
  CHECK((server::deal_by_cost({0.9, 0.9}, {0, 0}, 32, cost(df)) == std::vector<uint32_t>{0, 0}));
  // PERF-120, Ornith's table: four at 0.8 asking K 3 still the 16-row list; at 0.5 the 8-row step (K 1 each) pays more
  const server::RowCost orn = server::RowCost::ornith();
  CHECK((server::deal_by_cost({0.8, 0.8, 0.8, 0.8}, {3, 3, 3, 3}, 32, cost(orn)) == std::vector<uint32_t>{3, 3, 3, 3}));
  CHECK(rows(server::deal_by_cost({0.5, 0.5, 0.5, 0.5}, {3, 3, 3, 3}, 32, cost(orn))) <= 8);
  std::printf("deal_by_cost: K 7 x 4 at 0.85 (32 rows), three low-acceptance rows cut to %u rows, three at 0.6 in %u rows OK\n",
              rows(low), rows(mid));
}

int main() {
  test_deal_by_cost();
  test_model();
  test_dflash_measured_cost();
  test_flash_next_cost();
  test_cost_table();
  test_depth_costs();
  test_inkling_policy();
  test_inkling_tables_equal();
  test_measured_cost();
  test_draft_vocab_cost();
  test_up_on_high_acceptance();
  test_down_on_low_acceptance();
  test_recovers();
  test_hysteresis();
  test_deterministic();
  std::print("adaptive_k_test: all cases passed\n");
  return 0;
}
