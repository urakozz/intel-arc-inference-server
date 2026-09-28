#include "server/spec_accept.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace server {

double FilteredProbs::prob(uint32_t id) const {
  for (size_t i = 0; i < ids.size(); ++i)
    if (ids[i] == id) return probs[i] / mass;
  return 0.0;
}

FilteredProbs filter_probs(const float* logits, uint32_t vocab_used, const Sampling& s) {
  const uint32_t k = std::min(s.top_k == 0 ? vocab_used : s.top_k, vocab_used);
  FilteredProbs f;
  std::vector<uint32_t>& idx = f.ids;
  idx.resize(vocab_used);
  for (uint32_t i = 0; i < vocab_used; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [logits](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
  idx.resize(k);

  const float temperature = s.temperature > 0.0f ? s.temperature : 1.0f;
  const float top_logit = logits[idx[0]];
  std::vector<double>& probs = f.probs;
  probs.resize(k);
  double sum = 0.0;
  for (uint32_t i = 0; i < k; ++i) {
    probs[i] = std::exp(static_cast<double>((logits[idx[i]] - top_logit) / temperature));
    sum += probs[i];
  }
  for (double& p : probs) p /= sum;

  double cumulative = 0.0;
  uint32_t kept = k;
  const double top_p = std::clamp(static_cast<double>(s.top_p), 0.0, 1.0);
  for (uint32_t i = 0; i < k; ++i) {
    cumulative += probs[i];
    if (cumulative >= top_p) {
      kept = i + 1;
      break;
    }
  }
  kept = std::max<uint32_t>(kept, 1);
  probs.resize(kept);
  idx.resize(kept);
  f.mass = 0.0;
  for (double p : probs) f.mass += p;
  return f;
}

uint32_t draw(const FilteredProbs& f, std::mt19937_64& rng) {
  std::discrete_distribution<uint32_t> d(f.probs.begin(), f.probs.end());
  return f.ids[d(rng)];
}

AcceptResult accept_greedy(const uint32_t* verify_argmax, const uint32_t* drafts, uint32_t k) {
  uint32_t j = 0;
  while (j < k && drafts[j] == verify_argmax[j]) ++j;
  return {j, verify_argmax[j]};
}

uint32_t sample_draft(const float* q_row, uint32_t vocab_used, const Sampling& s,
                      std::mt19937_64& rng) {
  return draw(filter_probs(q_row, vocab_used, s), rng);
}

AcceptResult accept_sampled(const float* p_rows, const float* q_rows, const uint32_t* drafts,
                            uint32_t k, uint32_t vocab_used, size_t row_stride,
                            const Sampling& s, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (uint32_t i = 0; i < k; ++i) {
    const FilteredProbs p = filter_probs(p_rows + i * row_stride, vocab_used, s);
    const FilteredProbs q = filter_probs(q_rows + i * row_stride, vocab_used, s);
    const double qd = q.prob(drafts[i]);
    if (!(qd > 0.0))
      throw std::logic_error("server::accept_sampled: draft " + std::to_string(i) + " (id " +
                             std::to_string(drafts[i]) +
                             ") has q = 0 - drafts must be sampled from q");
    const double pd = p.prob(drafts[i]);
    const double u = uniform(rng);
    if (u < std::min(1.0, pd / qd)) continue;   // accepted with probability min(1, p/q)
    // Rejected: the correction from normalise(max(0, p - q)). Only p's support can
    // carry residual mass.
    std::vector<double> residual(p.ids.size());
    double total = 0.0;
    for (size_t r = 0; r < p.ids.size(); ++r) {
      residual[r] = std::max(0.0, p.probs[r] / p.mass - q.prob(p.ids[r]));
      total += residual[r];
    }
    if (!(total > 0.0)) return {i, draw(p, rng)};   // p == q: cannot follow a rejection
    std::discrete_distribution<uint32_t> d(residual.begin(), residual.end());
    return {i, p.ids[d(rng)]};
  }
  return {k, draw(filter_probs(p_rows + k * row_stride, vocab_used, s), rng)};
}

}  // namespace server
