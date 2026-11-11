#pragma once
// Open-work 13: `--spec mtp` with auto K (Qwen3.8, Ornith) - the drafts of every iteration chosen as they are made,
// from the MTP head's own confidence in each, instead of one K per request (server::AdaptiveK). Pure host code,
// unit-tested by tests/server/step_k_test.cc.
//
// After draft j (1-based) the head has said how sure it is of it - its softmax probability p_j at the drafted id
// (runtime::Engine::draft's `more`, draft_prob.cl). DraftCalib turns p_j into q_j, the odds that the main model
// accepts that draft: decayed counts of accepted / tested drafts per draft position and probability bin, the stated
// probability as the prior (basalt's draft_policy). The expected ids of an iteration that verifies K drafts are
//   E(K) = 1 + q_1 + q_1 q_2 + ... + q_1 ... q_K,
// its cost (in plain steps, MtpCost) verify[K] + draft[d - 1] when d drafts were made. After each draft StepK either
// stops - verifying the K <= j with the most E(K) per cost - or drafts another, when some deeper K (the drafts not
// yet made counted at the request's own recent acceptance at their position) promises more ids per cost. So a
// confident stretch (code) verifies all three, an unsure draft stops the chain at the previous one, and before any
// draft the request's recent acceptance decides whether to draft at all (a K = 0 run probes K = 1 now and then, as
// AdaptiveK does). Only which drafts are verified changes: the main model checks every one, so the output does not.
#include <array>
#include <cstdint>

#include "server/adaptive_k.h"

namespace server {

// How often a draft at position i (0-based) whose head stated probability p was accepted. One per server: it is the
// head's calibration, a property of the model, not of a request.
class DraftCalib {
 public:
  static constexpr uint32_t kPositions = 3;   // runtime::Engine::kMaxDraft
  static constexpr uint32_t kBins = 9;
  static constexpr double kDecay = 0.98;      // per tested draft of that bin
  static constexpr double kPriorN = 10;       // the stated probability is worth this many drafts

  static uint32_t bin(float p);
  double rate(uint32_t i, float p) const;     // q: (accepted + kPriorN p) / (tested + kPriorN)
  void observe(uint32_t i, float p, bool accepted);

 private:
  std::array<std::array<double, kBins>, kPositions> ok_{}, n_{};
};

struct StepKOptions {
  uint32_t max_k = 3;
  // The per-request acceptance at each draft position (conditional on the drafts before it accepted) - what a draft
  // not yet made is counted at: AdaptiveK's prior, weight and decay.
  double prior = 0.8;
  double prior_weight = 4;
  double decay = 0.9;
  uint32_t probe_every = 4;   // after this many plain iterations in a row, one iteration drafts
  MtpCost cost;
};

class StepK {
 public:
  StepK(const StepKOptions& opts, DraftCalib& calib);

  // Before an iteration: the most drafts it may make (0 = a plain step; the request's acceptance says drafting
  // does not pay, except every probe_every-th plain iteration, which drafts one).
  uint32_t begin();
  // After draft j (1 <= j <= begin()'s cap) with the head's probability p: < 0 to draft another, otherwise how
  // many of the j drafts to verify (0..j). At the cap it always answers.
  int choose(uint32_t j, float p);
  // After the verify: drafts kept of the ones verified (the last answer of choose(), or every draft made when the
  // engine stopped drafting before it answered - the last positions of max_len).
  void observe(uint32_t accepted);

  uint32_t verified() const { return verified_; }
  uint32_t max_k() const { return max_k_; }
  double position_rate(uint32_t i) const { return s_[i] / t_[i]; }

  // E(K) / cost for the known q's q[0..j) and the request's rates past them: the best K in [lo, hi] and its rate.
  // `drafted` drafts already made (their cost counted whatever K); deeper K draft up to K.
  double best(const double* q, uint32_t j, uint32_t lo, uint32_t hi, uint32_t drafted, uint32_t* k_out) const;

 private:
  StepKOptions o_;
  DraftCalib& calib_;
  uint32_t max_k_;
  std::array<double, DraftCalib::kPositions> s_{}, t_{};   // weighted accepted / tested per position (prior in)
  std::array<float, DraftCalib::kPositions> p_{};          // this iteration's stated probabilities
  std::array<double, DraftCalib::kPositions> q_{};         // ... calibrated
  uint32_t cap_ = 0, drafted_ = 0, verified_ = 0;
  bool answered_ = false;
  uint32_t plain_run_ = 0;
};

}  // namespace server
