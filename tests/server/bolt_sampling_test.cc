// Spec 25e review I1: cli::bolt::filter_probs_top_p (the top-p filter without a top-k, no full-vocabulary sort) against the shared
// server::filter_probs on randomized rows - distinct logits (the same ids in the same order, the same kept count, probabilities
// within 1e-9 relative), rows with ties (the same kept count and the same probability values; tied ids may differ in order:
// filter_probs leaves it unspecified, the fast path takes ascending id), many top_p / temperatures, top_k > 0 and softcap
// (delegated: bitwise the shared path) - and the measured time of each at 131072 rows.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "cli/bolt_serve.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

}  // namespace

int main() {
  std::mt19937_64 rng(2510);
  const uint32_t V = 131072;
  std::vector<float> row(V);
  size_t distinct_rows = 0, tie_rows = 0, delegated = 0, kept_total = 0, boundary = 0;
  const float top_ps[] = {0.0f, 0.1f, 0.5f, 0.9f, 0.95f, 0.999f, 1.0f};
  const float temps[] = {0.0f, 0.3f, 1.0f, 2.5f};
  for (int r = 0; r < 120; ++r) {
    const bool ties = r % 3 == 1;
    const uint32_t vocab = r % 5 == 0 ? 50 + uint32_t(rng() % 300) : V;
    std::normal_distribution<float> n(0.0f, r % 2 ? 1.5f : 6.0f);
    for (uint32_t i = 0; i < vocab; ++i) {
      row[i] = n(rng);
      if (ties) row[i] = std::round(row[i] * 4.0f) / 4.0f;   // many equal logits
    }
    if (!ties) {   // distinct: break any accidental equality
      std::vector<float> s(row.begin(), row.begin() + vocab);
      std::sort(s.begin(), s.end());
      if (std::adjacent_find(s.begin(), s.end()) != s.end()) continue;
    }
    for (float tp : top_ps)
      for (float t : temps) {
        server::Sampling s;
        s.greedy = false;
        s.top_k = 0;
        s.top_p = tp;
        s.temperature = t;
        const server::FilteredProbs a = server::filter_probs(row.data(), vocab, s);
        const server::FilteredProbs b = cli::bolt::filter_probs_top_p(row.data(), vocab, s);
        // The kept count is the same unless the cumulative mass sits within rounding (1e-9) of top_p at the cut - the normaliser is
        // summed in another order, so the last bits differ (top_p 1.0 makes every row such a case).
        const size_t common = std::min(a.ids.size(), b.ids.size());
        if (a.ids.size() != b.ids.size()) {
          double cum = 0.0;
          for (size_t i = 0; i < common; ++i) cum += a.probs[i];
          CHECK(std::fabs(cum - std::clamp(static_cast<double>(tp), 0.0, 1.0)) < 1e-9);
          ++boundary;
        }
        kept_total += common;
        for (size_t i = 0; i < common; ++i) {
          CHECK_NEAR(a.probs[i], b.probs[i], 1e-9 * a.probs[i] + 1e-300);
          if (!ties) CHECK_EQ(a.ids[i], b.ids[i]);
        }
        if (a.ids.size() == b.ids.size()) CHECK_NEAR(a.mass, b.mass, 1e-9);
        if (ties)   // the same logits at the same ranks: ids differ only among equal logits
          for (size_t i = 0; i < common; ++i) CHECK_EQ(row[a.ids[i]], row[b.ids[i]]);
        // top-k > 0 and a softcap go through the shared path: bitwise.
        server::Sampling k = s;
        k.top_k = 40;
        const server::FilteredProbs ka = server::filter_probs(row.data(), vocab, k), kb = cli::bolt::filter_probs_top_p(row.data(), vocab, k);
        CHECK(ka.ids == kb.ids && ka.probs == kb.probs && ka.mass == kb.mass);
        server::Sampling c = s;
        c.softcap = 20.0f;
        const server::FilteredProbs ca = server::filter_probs(row.data(), vocab, c), cb = cli::bolt::filter_probs_top_p(row.data(), vocab, c);
        CHECK(ca.ids == cb.ids && ca.probs == cb.probs && ca.mass == cb.mass);
        delegated += 2;
      }
    (ties ? tie_rows : distinct_rows) += 1;
  }
  // A draw from either is a valid id of the nucleus; with the same RNG state the distinct-row draws agree.
  {
    std::normal_distribution<float> n(0.0f, 4.0f);
    for (uint32_t i = 0; i < V; ++i) row[i] = n(rng);
    server::Sampling s;
    s.greedy = false;
    s.top_p = 0.95f;
    s.temperature = 1.0f;
    s.top_k = 0;
    size_t same = 0;
    for (int i = 0; i < 400; ++i) {
      std::mt19937_64 r1(i), r2(i);
      same += server::draw(server::filter_probs(row.data(), V, s), r1) == server::draw(cli::bolt::filter_probs_top_p(row.data(), V, s), r2);
    }
    std::printf("draws: %zu of 400 seeded draws equal between the two filters\n", same);
    CHECK(same >= 396);   // the last bits of the probabilities may flip a draw at a boundary, rarely
  }
  // The timing the review measured: top_k 0 at 131072 rows, T 1.0, top-p 0.95 (Bolt's defaults) and top_k 50.
  {
    std::normal_distribution<float> n(0.0f, 3.0f);
    for (uint32_t i = 0; i < V; ++i) row[i] = n(rng);
    server::Sampling s;
    s.greedy = false;
    s.top_p = 0.95f;
    s.temperature = 1.0f;
    s.top_k = 0;
    // A language model's row: a bulk far under a few peaks (here N(0, 2) under 12 rows near +25) - the second timing.
    std::vector<float> peaked(V);
    std::normal_distribution<float> m(0.0f, 2.0f);
    for (uint32_t i = 0; i < V; ++i) peaked[i] = m(rng);
    for (int i = 0; i < 12; ++i) peaked[rng() % V] = 25.0f + float(i) * 0.37f;
    {
      const server::FilteredProbs a = server::filter_probs(peaked.data(), V, s), b = cli::bolt::filter_probs_top_p(peaked.data(), V, s);
      CHECK_EQ(a.ids.size(), b.ids.size());
      CHECK(a.ids == b.ids);
      for (size_t i = 0; i < a.probs.size(); ++i) CHECK_NEAR(a.probs[i], b.probs[i], 1e-9 * a.probs[i]);
      const int pr = 20;
      double po = 0, pf = 0;
      for (int i = 0; i < pr; ++i) {
        auto t0 = Clock::now();
        const size_t x = server::filter_probs(peaked.data(), V, s).ids.size();
        po += ms(t0);
        t0 = Clock::now();
        const size_t y = cli::bolt::filter_probs_top_p(peaked.data(), V, s).ids.size();
        pf += ms(t0);
        CHECK_EQ(x, y);
      }
      std::printf("timing, a peaked row (MEASURED, this host, %d reps): filter_probs top_k 0 %.2f ms, top_p path %.2f ms (kept %zu)\n", pr,
                  po / pr, pf / pr, a.ids.size());
    }
    const int reps = 20;
    double old_ms = 0, fast_ms = 0, k50_ms = 0;
    size_t sink = 0;
    for (int i = 0; i < reps; ++i) {
      auto t0 = Clock::now();
      sink += server::filter_probs(row.data(), V, s).ids.size();
      old_ms += ms(t0);
      t0 = Clock::now();
      sink += cli::bolt::filter_probs_top_p(row.data(), V, s).ids.size();
      fast_ms += ms(t0);
      server::Sampling k = s;
      k.top_k = 50;
      t0 = Clock::now();
      sink += server::filter_probs(row.data(), V, k).ids.size();
      k50_ms += ms(t0);
    }
    std::printf("timing at %u rows (MEASURED, this host, %d reps): filter_probs top_k 0 %.2f ms, top_p path %.2f ms, filter_probs top_k 50 "
                "%.2f ms (sink %zu)\n", V, reps, old_ms / reps, fast_ms / reps, k50_ms / reps, sink);
  }
  std::printf("bolt_sampling_test OK: %zu distinct rows, %zu tie rows, %zu kept ids compared, %zu cuts within rounding of top_p, %zu delegated "
              "comparisons bitwise\n", distinct_rows, tie_rows, kept_total, boundary, delegated);
  return 0;
}
