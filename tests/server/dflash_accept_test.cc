// Host-only (spec 19c Task 3, spec 19 §6 D4): lossless sampling with the DFlash drafter's SPARSE draft
// distribution. The drafter's walk samples each draft from q = softmax(scores / T) over its row's 16 candidates
// (df_select.cl's Gumbel-argmax); every other id has q = 0. server::accept_sparse is spec 8's rule over that q.
//
//   1. dflash_q: the softmax at T over the candidates (a -inf or NaN score is q = 0, never drawn), sums to 1;
//      dflash_temperature mirrors filter_probs' T (<= 0 is 1.0).
//   2. D4: 10^6 seeded iterations (drafts sampled from q, accepted against p) - the first emitted token is
//      distributed as the filtered p_0 and the second, given d_1 was kept, as p_1 (chi-square p-value >= 0.01, spec 8
//      M4's bar), unfiltered and at top-k 12 / top-p 0.8 / T 0.7; p puts real mass outside the candidates.
//   3. q = 0 outside the candidates: no accepted draft is ever outside them (over all draws), and a draft outside
//      them (not sampled from q) is refused as a logic error.
//   4. The residual stays finite: p equal to q on the candidates (a rejection then leaves max(0, p - q) = 0: the
//      guard draws from p) and p with no mass on any candidate (every draft rejected: the output is p's).
//   5. RNG order: the same seed gives the same accept / next sequence.
#include <algorithm>
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

double gammq(double a, double x) {   // Numerical Recipes gammq, as spec_accept_test's
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

double chi2_pvalue(const std::vector<uint64_t>& counts, const server::FilteredProbs& f, const std::string& what) {
  uint64_t n = 0;
  for (uint64_t c : counts) n += c;
  double chi2 = 0;
  int bins = 0;
  for (uint32_t id = 0; id < counts.size(); ++id) {
    const double p = f.prob(id);
    if (p == 0) {
      CHECK(counts[id] == 0);
      continue;
    }
    const double e = p * double(n);
    chi2 += (double(counts[id]) - e) * (double(counts[id]) - e) / e;
    ++bins;
  }
  const double pv = bins > 1 ? gammq(0.5 * (bins - 1), 0.5 * chi2) : 1.0;
  std::printf("%s: n %llu, bins %d, chi2 %.2f, p-value %.4f\n", what.c_str(), (unsigned long long)n, bins, chi2, pv);
  return pv;
}

std::vector<float> random_rows(uint32_t rows, uint32_t v, uint64_t seed, float scale) {
  std::mt19937_64 g(seed);
  std::normal_distribution<float> nd(0.0f, scale);
  std::vector<float> out(size_t(rows) * v);
  for (float& x : out) x = nd(g);
  return out;
}

// A draft row as the drafter writes it: 16 distinct candidates (the drafter's top-16 of a noisy copy of p's
// logits, so they overlap p's mass without covering it) and their selector scores.
struct DraftRow {
  uint32_t cand[16];
  float scores[16];
};
DraftRow draft_row(const float* p_row, uint32_t V, uint64_t seed) {
  std::mt19937_64 g(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::pair<float, uint32_t>> s(V);
  for (uint32_t id = 0; id < V; ++id) s[id] = {p_row[id] + nd(g), id};
  std::sort(s.begin(), s.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
  DraftRow r{};
  for (uint32_t c = 0; c < 16; ++c) {
    r.cand[c] = s[c].second;
    r.scores[c] = s[c].first;
  }
  return r;
}
uint32_t draw_q(const server::SparseQ& q, std::mt19937_64& g) {
  std::discrete_distribution<uint32_t> d(q.probs, q.probs + q.n);
  return q.ids[d(g)];
}

void q_case() {
  const uint32_t cand[16] = {9, 3, 7, 1, 0, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15};
  float sc[16];
  for (int i = 0; i < 16; ++i) sc[i] = 2.0f - 0.25f * float(i);
  sc[5] = -INFINITY;
  sc[6] = NAN;
  const server::SparseQ q = server::dflash_q(cand, sc, 16, 0.7f);
  double sum = 0, z = 0;
  const double T = double(0.7f);   // the float temperature, as the walk and dflash_q see it
  for (int i = 0; i < 16; ++i)
    if (i != 5 && i != 6) z += std::exp(double(sc[i]) / T);
  for (uint32_t i = 0; i < q.n; ++i) sum += q.probs[i];
  CHECK(q.n == 16 && std::fabs(sum - 1.0) < 1e-12);
  CHECK(std::fabs(q.prob(9) - std::exp(2.0 / T) / z) < 1e-12);
  CHECK(q.prob(cand[5]) == 0.0 && q.prob(cand[6]) == 0.0 && q.prob(99) == 0.0);
  Sampling s;
  s.greedy = false;
  s.temperature = 0.0f;
  CHECK(server::dflash_temperature(s) == 1.0f);
  s.temperature = 0.6f;
  CHECK(server::dflash_temperature(s) == 0.6f);
  std::printf("dflash_q: softmax at T over 16 candidates, -inf / NaN scores q = 0, sum 1\n");
}

// D4 (and 3: q = 0 outside the candidates) over 1e6 seeded iterations.
void chi_square_case(const Sampling& s, const char* name) {
  constexpr uint32_t V = 50, K = 2;
  constexpr int kDraws = 1000000;
  const std::vector<float> p = random_rows(K + 1, V, 11, 1.5f);
  const float T = server::dflash_temperature(s);
  server::SparseQ q[K];
  DraftRow rows[K];
  for (uint32_t i = 0; i < K; ++i) {
    rows[i] = draft_row(p.data() + i * V, V, 100 + i);
    q[i] = server::dflash_q(rows[i].cand, rows[i].scores, 16, T);
  }
  // p puts real mass outside the candidates (else the case would not test the residual's tail)
  for (uint32_t i = 0; i < K; ++i) {
    const server::FilteredProbs f = server::filter_probs(p.data() + i * V, V, s);
    double outside = 0;
    for (size_t r = 0; r < f.ids.size(); ++r)
      if (q[i].prob(f.ids[r]) == 0) outside += f.probs[r] / f.mass;
    std::printf("%s: row %u p mass outside the 16 candidates %.3f\n", name, i, outside);
    if (!s.greedy && s.top_k == 0) CHECK(outside > 0.05);
  }
  std::mt19937_64 rng(2026), dg(77);
  std::vector<uint64_t> first(V, 0), second(V, 0);
  uint64_t outside_drafts = 0, accepted = 0;
  for (int it = 0; it < kDraws; ++it) {
    uint32_t d[K];
    for (uint32_t i = 0; i < K; ++i) d[i] = draw_q(q[i], dg);   // the device's walk, on the host
    const server::AcceptResult r = server::accept_sparse(p.data(), q, d, K, V, V, s, rng);
    CHECK(r.accepted <= K && r.next_token < V);
    for (uint32_t i = 0; i < r.accepted; ++i) outside_drafts += q[i].prob(d[i]) == 0;
    accepted += r.accepted;
    ++first[r.accepted >= 1 ? d[0] : r.next_token];
    if (r.accepted >= 1) ++second[r.accepted >= 2 ? d[1] : r.next_token];
  }
  std::printf("%s: %.4f drafts accepted, %llu accepted drafts outside the candidates\n", name,
              double(accepted) / (double(kDraws) * K), (unsigned long long)outside_drafts);
  CHECK(outside_drafts == 0);
  CHECK(chi2_pvalue(first, server::filter_probs(p.data(), V, s), std::string(name) + " first token vs p_0") >= 0.01);
  CHECK(chi2_pvalue(second, server::filter_probs(p.data() + V, V, s), std::string(name) + " second | d_1 kept vs p_1") >=
        0.01);
}

void refusal_and_residual_case() {
  constexpr uint32_t V = 50;
  Sampling s;
  s.greedy = false;
  s.top_k = 0;
  s.top_p = 1.0f;
  s.temperature = 1.0f;
  const std::vector<float> p = random_rows(2, V, 5, 1.0f);
  const DraftRow row = draft_row(p.data(), V, 9);
  const server::SparseQ q = server::dflash_q(row.cand, row.scores, 16, 1.0f);
  std::mt19937_64 rng(1);
  // a draft that q could not have drawn
  uint32_t outside = 0;
  while (q.prob(outside) != 0) ++outside;
  bool threw = false;
  try {
    (void)server::accept_sparse(p.data(), &q, &outside, 1, V, V, s, rng);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
  // p == q on the candidates: p's logits ARE the scores there and -inf elsewhere
  std::vector<float> pe(2 * V, -INFINITY);
  for (int c = 0; c < 16; ++c) pe[row.cand[c]] = row.scores[c];
  for (uint32_t id = 0; id < V; ++id) pe[V + id] = p[V + id];
  uint64_t rej = 0;
  for (int it = 0; it < 20000; ++it) {
    const uint32_t d = draw_q(q, rng);
    const server::AcceptResult r = server::accept_sparse(pe.data(), &q, &d, 1, V, V, s, rng);
    CHECK(r.next_token < V && std::isfinite(double(r.next_token)));
    rej += r.accepted == 0;
  }
  CHECK(rej < 200);   // p / q = 1 up to fp32 rounding: accepted almost always, the guard covers the rest
  // p with no mass on any candidate: every draft rejected, the output p's
  std::vector<float> pz(2 * V);
  for (uint32_t id = 0; id < V; ++id) pz[id] = q.prob(id) > 0 ? -INFINITY : p[id];
  for (uint32_t id = 0; id < V; ++id) pz[V + id] = p[V + id];
  std::vector<uint64_t> cnt(V, 0);
  for (int it = 0; it < 200000; ++it) {
    const uint32_t d = draw_q(q, rng);
    const server::AcceptResult r = server::accept_sparse(pz.data(), &q, &d, 1, V, V, s, rng);
    CHECK(r.accepted == 0 && q.prob(r.next_token) == 0);
    ++cnt[r.next_token];
  }
  CHECK(chi2_pvalue(cnt, server::filter_probs(pz.data(), V, s), "no candidate in p's support: output vs p") >= 0.01);
}

void rng_order_case() {
  constexpr uint32_t V = 50, K = 3;
  Sampling s;
  s.greedy = false;
  const std::vector<float> p = random_rows(K + 1, V, 41, 1.2f);
  server::SparseQ q[K];
  for (uint32_t i = 0; i < K; ++i) {
    const DraftRow row = draft_row(p.data() + i * V, V, 50 + i);
    q[i] = server::dflash_q(row.cand, row.scores, 16, server::dflash_temperature(s));
  }
  auto run = [&]() {
    std::mt19937_64 rng(99), dg(98);
    std::vector<uint32_t> out;
    for (int it = 0; it < 1000; ++it) {
      uint32_t d[K];
      for (uint32_t i = 0; i < K; ++i) d[i] = draw_q(q[i], dg);
      const server::AcceptResult r = server::accept_sparse(p.data(), q, d, K, V, V, s, rng);
      out.push_back(r.accepted);
      out.push_back(r.next_token);
    }
    return out;
  };
  CHECK(run() == run());
}

}  // namespace

int main() {
  q_case();
  Sampling s;
  s.greedy = false;
  s.top_k = 0;
  s.top_p = 1.0f;
  s.temperature = 1.0f;
  chi_square_case(s, "unfiltered");
  s.top_k = 12;
  s.top_p = 0.8f;
  s.temperature = 0.7f;
  chi_square_case(s, "top_k 12, top_p 0.8, T 0.7");
  refusal_and_residual_case();
  rng_order_case();
  std::printf("dflash_accept_test OK\n");
  return 0;
}
