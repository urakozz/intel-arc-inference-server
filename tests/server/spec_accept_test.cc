// Host-only (spec 8 M4, plan 8c Task 1): the speculative acceptance rule over
// synthetic logits rows. No device.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "server/spec_accept.h"

namespace {

using server::Sampling;

// Regularised upper incomplete gamma Q(a, x) (Numerical Recipes gammq).
double gammq(double a, double x) {
  if (x <= 0) return 1.0;
  const double gln = std::lgamma(a);
  if (x < a + 1) {
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; ++n) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-15) break;
    }
    return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
  }
  double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (int i = 1; i < 1000; ++i) {
    const double an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (std::fabs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (std::fabs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1) < 1e-15) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

// Chi-square p-value of `counts` against the filtered distribution `f`; ids the
// filter dropped must have count 0.
double chi2_pvalue(const std::vector<uint64_t>& counts, const server::FilteredProbs& f,
                   const char* what) {
  uint64_t n = 0;
  for (uint64_t c : counts) n += c;
  double chi2 = 0;
  int bins = 0;
  for (uint32_t id = 0; id < counts.size(); ++id) {
    const double p = f.prob(id);
    if (p == 0) {
      if (counts[id] != 0) {
        std::fprintf(stderr, "%s: id %u has p = 0 but %llu draws\n", what, id,
                     static_cast<unsigned long long>(counts[id]));
        std::exit(1);
      }
      continue;
    }
    const double e = p * double(n);
    chi2 += (double(counts[id]) - e) * (double(counts[id]) - e) / e;
    ++bins;
  }
  const double pv = bins > 1 ? gammq(0.5 * (bins - 1), 0.5 * chi2) : 1.0;
  std::printf("%s: n %llu, bins %d, chi2 %.2f, p-value %.4f\n", what,
              static_cast<unsigned long long>(n), bins, chi2, pv);
  return pv;
}

std::vector<float> random_rows(uint32_t rows, uint32_t v, uint64_t seed, float scale) {
  std::mt19937_64 g(seed);
  std::normal_distribution<float> nd(0.0f, scale);
  std::vector<float> out(size_t(rows) * v);
  for (float& x : out) x = nd(g);
  return out;
}

// M4: the full procedure (sample drafts from q, accept against p) 1e6 times; the
// first emitted token must be distributed as filtered p_0, and the second emitted
// token, given d_1 was kept, as filtered p_1 (the synthetic rows are fixed, so
// p_1 does not depend on the first token).
void chi_square_case(const Sampling& s, const char* name) {
  constexpr uint32_t V = 50, K = 2;
  constexpr int kDraws = 1000000;
  const std::vector<float> p = random_rows(K + 1, V, 11, 1.5f);
  // The draft head approximates the main model: q_i = p_i + noise.
  std::vector<float> q = random_rows(K, V, 23, 0.8f);
  for (size_t i = 0; i < q.size(); ++i) q[i] += p[i];
  std::mt19937_64 rng(2026);
  std::vector<uint64_t> first(V, 0), second(V, 0);
  for (int it = 0; it < kDraws; ++it) {
    uint32_t d[K];
    for (uint32_t i = 0; i < K; ++i) d[i] = server::sample_draft(q.data() + i * V, V, s, rng);
    const server::AcceptResult r = server::accept_sampled(p.data(), q.data(), d, K, V, V, s, rng);
    CHECK(r.accepted <= K);
    ++first[r.accepted >= 1 ? d[0] : r.next_token];
    if (r.accepted >= 1) ++second[r.accepted >= 2 ? d[1] : r.next_token];
  }
  const std::string n0 = std::string(name) + " first token vs p_0";
  const std::string n1 = std::string(name) + " second token | d_1 kept vs p_1";
  CHECK(chi2_pvalue(first, server::filter_probs(p.data(), V, s), n0.c_str()) >= 0.01);
  CHECK(chi2_pvalue(second, server::filter_probs(p.data() + V, V, s), n1.c_str()) >= 0.01);
}

// Spec 8 §11: a draft vocabulary V'. The draft list writes q_i only at V''s ids; every
// other entry of the row holds -inf (MtpBuffers::zero), i.e. q = 0 there. The rows here
// are that: q = p + noise on V', -inf elsewhere, and p puts real mass outside V'.
void draft_vocab_case(const Sampling& s, const char* name) {
  constexpr uint32_t V = 50, K = 2;
  constexpr int kDraws = 1000000;
  std::vector<bool> in_v(V, false);
  for (uint32_t id = 0; id < V; ++id) in_v[id] = id % 5 != 3 && id < 44;   // 35 of 50
  const std::vector<float> p = random_rows(K + 1, V, 31, 1.5f);
  std::vector<float> q = random_rows(K, V, 37, 0.8f);
  for (uint32_t i = 0; i < K; ++i)
    for (uint32_t id = 0; id < V; ++id)
      q[i * V + id] = in_v[id] ? q[i * V + id] + p[i * V + id] : -INFINITY;
  // The filter over a -inf row: finite probabilities, and exactly 0 outside V'.
  for (uint32_t i = 0; i < K; ++i) {
    const server::FilteredProbs f = server::filter_probs(q.data() + i * V, V, s);
    CHECK(std::isfinite(f.mass) && f.mass > 0.0);
    for (double pr : f.probs) CHECK(std::isfinite(pr) && pr >= 0.0);
    for (uint32_t id = 0; id < V; ++id) {
      const double pr = f.prob(id);
      CHECK(std::isfinite(pr));
      if (!in_v[id]) CHECK_EQ(pr, 0.0);
    }
  }
  std::mt19937_64 rng(2027);
  std::vector<uint64_t> first(V, 0), second(V, 0);
  uint64_t outside_emitted = 0;
  for (int it = 0; it < kDraws; ++it) {
    uint32_t d[K];
    for (uint32_t i = 0; i < K; ++i) {
      d[i] = server::sample_draft(q.data() + i * V, V, s, rng);
      CHECK(in_v[d[i]]);   // a draft is never an id outside V'
    }
    const server::AcceptResult r = server::accept_sampled(p.data(), q.data(), d, K, V, V, s, rng);
    CHECK(r.accepted <= K && r.next_token < V);
    const uint32_t t0 = r.accepted >= 1 ? d[0] : r.next_token;
    ++first[t0];
    outside_emitted += !in_v[t0];
    if (r.accepted >= 1) ++second[r.accepted >= 2 ? d[1] : r.next_token];
  }
  // The output keeps p's distribution, mass outside V' included: those ids are reached
  // through the residual after a rejection, never as drafts.
  const std::string n0 = std::string(name) + " first token vs p_0";
  const std::string n1 = std::string(name) + " second token | d_1 kept vs p_1";
  CHECK(chi2_pvalue(first, server::filter_probs(p.data(), V, s), n0.c_str()) >= 0.01);
  CHECK(chi2_pvalue(second, server::filter_probs(p.data() + V, V, s), n1.c_str()) >= 0.01);
  const server::FilteredProbs f0 = server::filter_probs(p.data(), V, s);
  double p_out = 0;
  for (uint32_t id = 0; id < V; ++id)
    if (!in_v[id]) p_out += f0.prob(id);
  std::printf("%s: p_0 mass outside V' %.4f, emitted %.4f\n", name, p_out,
              double(outside_emitted) / kDraws);
  if (p_out > 0.01) CHECK(outside_emitted > 0);

  // The math is the one a finite zero-probability logit gives: -1e30 instead of -inf
  // filters to the same ids and probabilities, so the same seed accepts the same drafts
  // and draws the same corrections, iteration for iteration.
  std::vector<float> q_fin = q;
  for (float& x : q_fin)
    if (std::isinf(x)) x = -1e30f;
  std::mt19937_64 ra(99), rb(99);
  for (int it = 0; it < 20000; ++it) {
    uint32_t da[K], db[K];
    for (uint32_t i = 0; i < K; ++i) {
      da[i] = server::sample_draft(q.data() + i * V, V, s, ra);
      db[i] = server::sample_draft(q_fin.data() + i * V, V, s, rb);
      CHECK_EQ(da[i], db[i]);
    }
    const server::AcceptResult a = server::accept_sampled(p.data(), q.data(), da, K, V, V, s, ra);
    const server::AcceptResult b = server::accept_sampled(p.data(), q_fin.data(), db, K, V, V, s, rb);
    CHECK(a.accepted == b.accepted && a.next_token == b.next_token);
  }
  // An id outside V' handed in as a draft (an engine bug) is refused, not divided by.
  uint32_t outside = 0;
  while (in_v[outside]) ++outside;
  bool threw = false;
  try {
    const uint32_t d[1] = {outside};
    (void)server::accept_sampled(p.data(), q.data(), d, 1, V, V, s, rng);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
}

}  // namespace

int main() {
  int cases = 0;
  // 1. Greedy.
  {
    const uint32_t argmax[4] = {7, 8, 9, 10};
    const uint32_t all[3] = {7, 8, 9};
    server::AcceptResult r = server::accept_greedy(argmax, all, 3);
    CHECK_EQ(r.accepted, 3u);
    CHECK_EQ(r.next_token, 10u);
    const uint32_t first_bad[3] = {1, 8, 9};
    r = server::accept_greedy(argmax, first_bad, 3);
    CHECK_EQ(r.accepted, 0u);
    CHECK_EQ(r.next_token, 7u);
    const uint32_t mid_bad[3] = {7, 2, 9};
    r = server::accept_greedy(argmax, mid_bad, 3);
    CHECK_EQ(r.accepted, 1u);
    CHECK_EQ(r.next_token, 8u);
    r = server::accept_greedy(argmax, nullptr, 0);   // k = 0: the plain step
    CHECK_EQ(r.accepted, 0u);
    CHECK_EQ(r.next_token, 7u);
    ++cases;
  }
  // 2-3. M4 chi-square, without and with the filter.
  {
    Sampling s;
    s.greedy = false;
    s.temperature = 1.0f;
    s.top_k = 0;
    s.top_p = 1.0f;
    chi_square_case(s, "unfiltered");
    ++cases;
    s.temperature = 0.7f;
    s.top_k = 12;
    s.top_p = 0.8f;
    chi_square_case(s, "top_k 12, top_p 0.8, T 0.7");
    ++cases;
  }
  // 4. Review Focus 3: one candidate left after filtering, and q(d) = 0.
  {
    constexpr uint32_t V = 20;
    Sampling s;
    s.greedy = false;
    s.top_k = 1;
    s.top_p = 0.95f;
    std::vector<float> p(3 * V, 0.0f), q(2 * V, 0.0f);
    for (uint32_t r = 0; r < 3; ++r) p[r * V + 5] = 3.0f;   // p one-hot on 5
    for (uint32_t r = 0; r < 2; ++r) q[r * V + 9] = 3.0f;   // q one-hot on 9
    std::mt19937_64 rng(1);
    for (int it = 0; it < 1000; ++it) {
      uint32_t d[2] = {server::sample_draft(q.data(), V, s, rng),
                       server::sample_draft(q.data() + V, V, s, rng)};
      CHECK_EQ(d[0], 9u);
      const server::AcceptResult r = server::accept_sampled(p.data(), q.data(), d, 2, V, V, s, rng);
      CHECK_EQ(r.accepted, 0u);        // p(9) = 0: always rejected
      CHECK_EQ(r.next_token, 5u);      // residual = p
    }
    // p == q, one-hot: always kept, bonus from p_2.
    for (uint32_t r = 0; r < 2; ++r) q[r * V + 9] = 0.0f, q[r * V + 5] = 3.0f;
    for (int it = 0; it < 1000; ++it) {
      uint32_t d[2] = {5, 5};
      const server::AcceptResult r = server::accept_sampled(p.data(), q.data(), d, 2, V, V, s, rng);
      CHECK_EQ(r.accepted, 2u);
      CHECK_EQ(r.next_token, 5u);
    }
    // A draft q never proposes: a logic error, not a division by zero.
    bool threw = false;
    try {
      uint32_t d[1] = {9};
      (void)server::accept_sampled(p.data(), q.data(), d, 1, V, V, s, rng);
    } catch (const std::logic_error&) {
      threw = true;
    }
    CHECK(threw);
    // top_p leaving one candidate (a dominant logit), unfiltered by top_k.
    s.top_k = 0;
    s.top_p = 0.5f;
    std::vector<float> p2 = random_rows(2, V, 5, 0.1f), q2 = random_rows(1, V, 6, 0.1f);
    p2[3] = 20.0f;
    p2[V + 4] = 20.0f;
    q2[3] = 20.0f;
    for (int it = 0; it < 1000; ++it) {
      uint32_t d[1] = {server::sample_draft(q2.data(), V, s, rng)};
      CHECK_EQ(d[0], 3u);
      const server::AcceptResult r =
          server::accept_sampled(p2.data(), q2.data(), d, 1, V, V, s, rng);
      CHECK_EQ(r.accepted, 1u);
      CHECK_EQ(r.next_token, 4u);
    }
    ++cases;
  }
  // 5. Review Focus 4: seeded reproducibility of whole iterations (drafts, then
  // uniforms, then the correction/bonus), over varying k.
  {
    constexpr uint32_t V = 40;
    Sampling s;
    s.greedy = false;
    s.temperature = 0.9f;
    s.top_k = 20;
    s.top_p = 0.95f;
    const std::vector<float> p = random_rows(4, V, 77, 1.0f), q = random_rows(3, V, 78, 1.0f);
    const auto run = [&](uint64_t seed) {
      std::mt19937_64 rng(seed);
      std::vector<uint32_t> out;
      for (int it = 0; it < 2000; ++it) {
        const uint32_t k = 1 + it % 3;
        uint32_t d[3];
        for (uint32_t i = 0; i < k; ++i) d[i] = server::sample_draft(q.data() + i * V, V, s, rng);
        const server::AcceptResult r = server::accept_sampled(p.data(), q.data(), d, k, V, V, s, rng);
        for (uint32_t i = 0; i < r.accepted; ++i) out.push_back(d[i]);
        out.push_back(r.next_token);
      }
      return out;
    };
    const std::vector<uint32_t> a = run(4242), b = run(4242), c = run(4243);
    CHECK(a == b);
    CHECK(a != c);
    ++cases;
  }
  // 6. Spec 8 §11: q = 0 (-inf logits) outside a draft vocabulary, without and with the
  // filter: no draft outside V', the output still distributed as p, the same accept and
  // residual arithmetic as a finite zero-probability logit, everything finite.
  {
    Sampling s;
    s.greedy = false;
    s.temperature = 1.0f;
    s.top_k = 0;
    s.top_p = 1.0f;
    draft_vocab_case(s, "draft vocab, unfiltered");
    s.temperature = 0.7f;
    s.top_k = 12;
    s.top_p = 0.8f;
    draft_vocab_case(s, "draft vocab, top_k 12, top_p 0.8, T 0.7");
    ++cases;
  }
  std::printf("spec_accept_test: %d/6 cases passed\n", cases);
  return cases == 6 ? 0 : 1;
}
