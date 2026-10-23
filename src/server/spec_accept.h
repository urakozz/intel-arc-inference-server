#pragma once
// Spec 8 §3.3 step 3 (plan 8c Task 1): the speculative acceptance rule, on the host.
// Pure functions over logits rows: no device, no Engine, unit-tested by
// tests/server/spec_accept_test.cc.
#include <cstdint>
#include <functional>
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
// - softmax over exactly those k, at `temperature` (<= 0 treated as 1.0) - of the model's transformed
//   logits when Sampling::logit_mult / softcap say so (spec 23e: Muse-Glimmer's
//   20 tanh(z x 0.196... / 20), fp32, on the k candidates; monotone, so the raw order is kept).
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

// Spec 24e (decision 1): Hugging Face's repetition penalty (logits_process.py, RepetitionPenaltyLogitsProcessor) on the host
// logits row, applied BEFORE filter_probs - before temperature, top-k and top-p, HF's order, so the penalty can demote a
// leader out of the top-k: each listed id's logit l becomes `l < 0 ? l * penalty : l / penalty`. The ids are gathered
// first and scattered after, as HF does, so an id listed twice is penalised once. Ids at or above `vocab_used` are
// ignored. A penalty of 1.0 returns without touching the row.
void apply_repetition_penalty(float* logits, uint32_t vocab_used, const std::vector<uint32_t>& ids, float penalty);

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

// Spec 19e (plan 19e Review Focus 1): lossless sampling when the drafts are DETERMINISTIC
// (prompt lookup: the proposal is a point mass, q_i = 1 on d_i and 0 elsewhere). The rule
// above with that q: accept d_i with probability min(1, p(d_i) / 1) = p(d_i) (the filtered,
// renormalised p, so 0 when the filter dropped d_i); on the first rejection sample from
// normalise(max(0, p - q)), which is p with d_i removed and renormalised; all k kept: the
// bonus from p_k. The output is distributed exactly as p, whatever the drafts were.
//
// RNG order as accept_sampled's: one uniform per draft considered, in order, then exactly
// one draw for the correction/bonus. No q rows: nothing was sampled to propose.
AcceptResult accept_point_mass(const float* p_rows /*[k+1][row_stride]*/, const uint32_t* drafts,
                               uint32_t k, uint32_t vocab_used, size_t row_stride,
                               const Sampling& s, std::mt19937_64& rng);

// A draft sampled from the head's row q_i, filtered like p.
uint32_t sample_draft(const float* q_row, uint32_t vocab_used, const Sampling& s,
                      std::mt19937_64& rng);

// --- spec 19c: the DFlash drafter's sparse q (spec 19 §2 "The walk", §6 D4) --------------
//
// The drafter's walk (df_select.cl) samples draft row j by Gumbel-argmax of scores[j][c] / T over the row's 16
// candidates, i.e. from q_j = softmax(scores[j] / T) on cand[j] and q_j = 0 on every other id. SparseQ is that row;
// a candidate whose score is -inf or NaN has q = 0 (the walk never picks it).
struct SparseQ {
  static constexpr uint32_t kMax = 16;
  uint32_t n = 0;
  uint32_t ids[kMax] = {};
  double probs[kMax] = {};
  double prob(uint32_t id) const;   // q(id): 0 off the candidates
};
SparseQ dflash_q(const uint32_t* cand, const float* scores, uint32_t n, float temperature);
// The temperature the drafter's walk samples at for a sampled request: filter_probs' (T <= 0 is 1.0), so q's
// sharpness follows the request's p.
float dflash_temperature(const Sampling& s);

// Spec 8's lossless rule (accept_sampled) with a sparse q: accept d_i with probability min(1, p(d_i) / q_i(d_i));
// on the first rejection sample from normalise(max(0, p - q_i)) - p's filtered support only (q = 0 off the
// candidates leaves p there whole), from p itself if that is all zero (p == q on the candidates: a rejection then
// came from fp rounding); all k kept: the bonus from p_k. The output is distributed as p whatever q was, as long as
// each d_i was drawn from q_i. Throws std::logic_error when q_i(d_i) = 0 (d_i was not drawn from q_i). RNG order:
// one uniform per draft considered, then exactly one draw for the correction / bonus (the caller's device seed,
// drawn before the drafts, comes first).
AcceptResult accept_sparse(const float* p_rows /*[k+1][row_stride]*/, const SparseQ* q /*[k]*/,
                           const uint32_t* drafts, uint32_t k, uint32_t vocab_used, size_t row_stride,
                           const Sampling& s, std::mt19937_64& rng);

// Spec 26c: accept_sparse and accept_point_mass with the caller's row filter in place of filter_probs - the filter a
// model's own sampler applies (Nemotron 3.5 Bolt's cli::bolt::filter_probs_top_p: filter_probs' distribution without the
// full-vocabulary sort, 17 ms a row at top-k 0). The rule, the RNG order and the throws are the functions above, unchanged;
// `filter(row)` must return the filtered p of a [vocab] logits row (ids in descending order, probs unnormalised, mass).
using RowFilter = std::function<FilteredProbs(const float* row)>;
AcceptResult accept_sparse_rows(const float* p_rows /*[k+1][row_stride]*/, const SparseQ* q /*[k]*/,
                                const uint32_t* drafts, uint32_t k, size_t row_stride, const RowFilter& filter,
                                std::mt19937_64& rng);
AcceptResult accept_point_mass_rows(const float* p_rows /*[k+1][row_stride]*/, const uint32_t* drafts, uint32_t k,
                                    size_t row_stride, const RowFilter& filter, std::mt19937_64& rng);

}  // namespace server
