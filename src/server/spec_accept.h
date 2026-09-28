#pragma once
// Spec 8 §3.3 step 3 (plan 8c Task 1): the speculative acceptance rule, on the host.
// Pure functions over logits rows: no device, no Engine, unit-tested by
// tests/server/spec_accept_test.cc.
#include <cstdint>
#include <random>
#include <vector>

#include "server/deps.h"

namespace server {

// The filter the host sampler applies (spec 7 §3.5, plan 7d Task 5), shared by
// `sample()` (src/cli/serve_adapters.h) and the acceptance rule so that p and q
// are filtered exactly as a plain sampled step filters its row:
//
// - top-k: `std::partial_sort` picks the k highest-logit ids among the first
//   `vocab_used` (k = min(top_k, vocab_used); top_k == 0 means no cap). Ids >=
//   vocab_used are never candidates -- that is the vocabulary mask.
// - softmax over exactly those k, at `temperature` (<= 0 treated as 1.0).
// - top-p: walk the k in probability order, cut after the first prefix whose
//   cumulative mass >= top_p; at least one candidate is always kept.
//
// `probs` are the softmax over the top-k, truncated at the top-p cut and NOT
// renormalised -- exactly the weights `sample()` has always handed to
// `std::discrete_distribution` (so `--mtp 0` sampling is bit-for-bit what it was);
// `mass` is their sum. The filtered distribution is probs[i] / mass on ids[i]
// (ids in descending logit order).
struct FilteredProbs {
  std::vector<uint32_t> ids;
  std::vector<double> probs;
  double mass = 0.0;
  // Probability of `id` under the filtered, renormalised distribution; 0 when
  // the filter dropped it.
  double prob(uint32_t id) const;
};
FilteredProbs filter_probs(const float* logits, uint32_t vocab_used, const Sampling& s);

// One draw from the filtered distribution (what `sample()` returns).
uint32_t draw(const FilteredProbs& f, std::mt19937_64& rng);

// j drafts kept, then one token from p (the correction on a rejection, the bonus
// when all k are kept). The iteration emits the pending token, d_1 .. d_j, and
// `next_token` becomes the pending token.
struct AcceptResult {
  uint32_t accepted;
  uint32_t next_token;
};

// Greedy: accept d_i while d_i == argmax p_{i-1}; next_token = argmax p_j.
AcceptResult accept_greedy(const uint32_t* verify_argmax, const uint32_t* drafts, uint32_t k);

// Lossless sampling (Leviathan et al. 2023) over the filtered p and q: accept d_i
// with probability min(1, p(d_i) / q(d_i)); on the first rejection sample from
// normalise(max(0, p - q)) (from p if that is all zero, which cannot happen after
// a rejection but is guarded); all k kept: the bonus from p_k.
//
// RNG order (Review Focus 4): one uniform per draft considered, in order, then
// exactly one draw for the correction/bonus. Throws std::logic_error when q(d_i)
// is 0 (d_i was not sampled from q).
AcceptResult accept_sampled(const float* p_rows /*[k+1][row_stride]*/,
                            const float* q_rows /*[k][row_stride]*/, const uint32_t* drafts,
                            uint32_t k, uint32_t vocab_used, size_t row_stride,
                            const Sampling& s, std::mt19937_64& rng);

// A draft sampled from the head's row q_i, filtered like p.
uint32_t sample_draft(const float* q_row, uint32_t vocab_used, const Sampling& s,
                      std::mt19937_64& rng);

}  // namespace server
