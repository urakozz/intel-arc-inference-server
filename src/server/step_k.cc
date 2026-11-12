#include "server/step_k.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace server {

namespace {
// The bins' upper edges: finest near 1, where the deep drafts are decided (basalt's draft_policy).
constexpr float kEdge[DraftCalib::kBins - 1] = {0.3f, 0.5f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f, 0.995f};
}  // namespace

uint32_t DraftCalib::bin(float p) {
  uint32_t b = 0;
  while (b < kBins - 1 && p >= kEdge[b]) ++b;
  return b;
}

double DraftCalib::rate(uint32_t i, float p) const {
  const uint32_t b = bin(p);
  const double c = std::clamp(static_cast<double>(p), 0.0, 1.0);
  return (ok_.at(i)[b] + kPriorN * c) / (n_.at(i)[b] + kPriorN);
}

void DraftCalib::observe(uint32_t i, float p, bool accepted) {
  const uint32_t b = bin(p);
  ok_.at(i)[b] = kDecay * ok_[i][b] + (accepted ? 1.0 : 0.0);
  n_.at(i)[b] = kDecay * n_[i][b] + 1.0;
}

StepK::StepK(const StepKOptions& opts, DraftCalib& calib)
    : o_(opts), calib_(calib), max_k_(std::min({opts.max_k, opts.cost.max_k(), DraftCalib::kPositions})) {
  if (!(opts.prior_weight > 0) || !(opts.prior > 0 && opts.prior <= 1))
    throw std::invalid_argument("StepK: the prior needs 0 < prior <= 1 and a weight > 0");
  s_.fill(opts.prior * opts.prior_weight);
  t_.fill(opts.prior_weight);
}

double StepK::best(const double* q, uint32_t j, uint32_t lo, uint32_t hi, uint32_t drafted, uint32_t* k_out) const {
  double e = 1.0, all = 1.0, best_r = -1.0;
  uint32_t best_k = lo;
  for (uint32_t k = 0; k <= hi; ++k) {
    if (k > 0) {
      // A draft not yet made: like the last one made (a sure draft is followed by sure ones in a stretch of code),
      // or before any, the request's recent acceptance at that position.
      all *= k - 1 < j ? q[k - 1] : j > 0 ? q[j - 1] : position_rate(k - 1);
      e += all;
    }
    if (k < lo) continue;
    const uint32_t d = std::max(drafted, k);
    const double r = e / (o_.cost.verify.at(k) + (d > 0 ? o_.cost.draft.at(d - 1) : 0.0));
    if (r > best_r) {   // a tie keeps the smaller K
      best_r = r;
      best_k = k;
    }
  }
  *k_out = best_k;
  return best_r;
}

uint32_t StepK::begin() {
  drafted_ = verified_ = 0;
  answered_ = false;
  uint32_t k = 0;
  best(q_.data(), 0, 0, max_k_, 0, &k);
  if (k > 0)
    cap_ = max_k_;
  else
    cap_ = max_k_ > 0 && o_.probe_every > 0 && plain_run_ >= o_.probe_every ? 1 : 0;
  return cap_;
}

int StepK::choose(uint32_t j, float p) {
  if (j != drafted_ + 1 || j > cap_)
    throw std::logic_error("StepK::choose: draft " + std::to_string(j) + " after " + std::to_string(drafted_) +
                           " of at most " + std::to_string(cap_));
  p_[j - 1] = p;
  q_[j - 1] = calib_.rate(j - 1, p);
  drafted_ = j;
  uint32_t stop = 0;
  const double r_stop = best(q_.data(), j, 0, j, j, &stop);
  if (j < cap_) {
    uint32_t more = 0;
    if (best(q_.data(), j, j + 1, cap_, j, &more) > r_stop) return -1;
  }
  verified_ = stop;
  answered_ = true;
  return static_cast<int>(stop);
}

void StepK::observe(uint32_t accepted) {
  if (!answered_) verified_ = drafted_;
  if (accepted > verified_)
    throw std::invalid_argument("StepK::observe: " + std::to_string(accepted) + " drafts kept of " +
                                std::to_string(verified_));
  // The drafts after the first rejection were never tested.
  const uint32_t tested = verified_ == 0 ? 0 : std::min(verified_, accepted + 1);
  for (uint32_t i = 0; i < tested; ++i) {
    const bool ok = i < accepted;
    calib_.observe(i, p_[i], ok);
    s_[i] = o_.decay * s_[i] + (ok ? 1.0 : 0.0);
    t_[i] = o_.decay * t_[i] + 1.0;
  }
  plain_run_ = cap_ == 0 ? plain_run_ + 1 : 0;
}

}  // namespace server
