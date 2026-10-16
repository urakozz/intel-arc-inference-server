#pragma once
// Spec 8 §10: `b70-serve --mtp auto` - per request, choose the iteration's draft count K
// from the request's own acceptance so far. Pure host code, unit-tested by
// tests/server/adaptive_k_test.cc; the server keeps one AdaptiveK per request.
#include <cstdint>
#include <string>
#include <vector>

namespace server {

// Step costs in units of one plain decode step: verify[m - 1] for a verify of M = m rows
// (verify[0] = the plain step), draft[k - 1] for k drafts. Data, not code: defaults per
// head form, overridable with `--mtp-cost` so the box can calibrate.
struct MtpCost {
  std::vector<double> verify{1.00, 1.17, 1.52, 1.74};
  std::vector<double> draft{0.19, 0.37, 0.55};

  static MtpCost bf16_head();   // measured (spec 8 P0 / §8)
  static MtpCost int8_head();   // derived from spec 9 H2 (drafts -34 %, verify -~2 ms)

  // Spec 8 §11: the share of ONE draft's cost that is its lm_head read, per head form -
  // what a reduced draft vocabulary scales. **Derived, not measured:** the head's bytes at
  // ~570 GB/s in units of that form's plain step, over the table's one-draft cost.
  //   int8: 1.27 GB -> ~2.2 ms of the 32.65 ms int8 plain step = 0.067 of draft[0] 0.13
  //   bf16: 2.54 GB -> ~4.4 ms of the ~34 ms bf16 plain step   = 0.129 of draft[0] 0.19
  // The rest of a draft (the MTP head's own layer, its norms, the argmax) does not move.
  // `probe_mtp_steps ... <form> <size>` on the box replaces the result (via --mtp-cost).
  static constexpr double kInt8DraftHeadShare = 0.067 / 0.13;
  static constexpr double kBf16DraftHeadShare = 0.129 / 0.19;
  // This table with every draft[k - 1] scaled to draft[k - 1] * ((1 - head_share) +
  // head_share * fraction), `fraction` = |V'| / 248320 (the rows a draft's head reads);
  // verify is unchanged (it always reads the full head). fraction 1 returns the table.
  MtpCost with_draft_vocab(double head_share, double fraction) const;
  // "verify=1,1.17,1.52,1.74;draft=0.19,0.37,0.55"; either part may be left out (it then
  // keeps `base`'s). Throws std::invalid_argument on anything else.
  static MtpCost parse(const std::string& text, const MtpCost& base);

  // Spec 19e: the same verify rows with a free draft - prompt lookup proposes on the host in
  // microseconds (prompt_lookup.h), ~0 of a 32 ms plain step. draft[k - 1] = 0 for every K
  // the verify list covers.
  MtpCost with_free_drafts() const;

  // Spec 19c (`--spec dflash`, spec 19 decision 4): the verify rows to M = 8 and the DFlash drafter's draft row,
  // FLAT in K - one pass of the drafter drafts the whole block. DERIVED until the box measures them (queue row 29's
  // probe_mtp_steps at M = 5..8 and probe_draft_cost; `--spec-cost` replaces either part):
  //   verify M = 1..4   the head form's rows above; M = 5..8: M = 3 -> 4's slope, 0.22 a row
  //                     (docs/probe-dflash-2026-10-08.md §3);
  //   draft             the drafter's weights read once (int8 ~1.8 GB -> 0.10, bf16 ~3.6 GB -> 0.19 of the 32.65 ms
  //                     int8 plain step) plus the target's head over the K mask rows, read once (int8 1.27 GB ->
  //                     0.07, bf16 2.54 GB -> 0.14), whatever K (spec 19 §3).
  static MtpCost dflash(bool i8_head, bool int8_drafter);
  static constexpr double kDFlashDrafterInt8 = 0.10, kDFlashDrafterBf16 = 0.19;
  static constexpr double kDFlashHeadInt8 = 0.07, kDFlashHeadBf16 = 0.14;
  // The head's share of that draft row - what a draft vocabulary scales (with_draft_vocab).
  static double dflash_head_share(bool i8_head, bool int8_drafter);

  // Spec 23f: Muse-Glimmer-30B's rows, DERIVED (tools/spec/muse_p0_project.py; box row 47's r47.cost measures them).
  // The verify at M = 1..16: the int8 head's M = 1..4 (1.00 1.18 1.56 1.79) extended by its M = 2..4 slope, 0.305 a
  // row. The draft row flat in K (one 16-row block drafts all 15): (the drafter's bytes + the head's) over the plain
  // step's 14.86 GB - int4 g64 0.182, int8 0.263, bf16 0.435 with the int8 head (1.346 GB); the bf16 head (2.692 GB)
  // adds its 1.346 GB to both. `drafter_format`: 0 bf16, 1 int8, 2 int4 g64 (loader::DraftFormat's order).
  static MtpCost muse_verify(bool i8_head);
  static MtpCost muse_dflash(bool i8_head, unsigned drafter_format);
  static constexpr double kMusePlainGB = 14.86, kMuseHeadInt8GB = 1.346;
  static constexpr double kMuseDrafterGB[3] = {5.112, 2.556, 1.358};

  uint32_t max_k() const;              // the deepest K both lists cover
  double cost(uint32_t k) const;       // K = 0: 1 (a plain step)
};

struct AdaptiveKOptions {
  uint32_t max_k = 3;
  double prior = 0.8;          // alpha before any observation
  double prior_weight = 4;     // ... worth this many drafts
  double decay = 0.9;          // per-iteration weight of the history (EMA, ~10 iterations)
  uint32_t warmup = 2;         // iterations at K = max before the policy chooses
  double hysteresis = 0.03;    // a new K must beat the current K's rate by this fraction
  uint32_t probe_every = 4;    // at K = 0: one K = 1 iteration after this many plain ones
  MtpCost cost;
};

class AdaptiveK {
 public:
  explicit AdaptiveK(const AdaptiveKOptions& opts);

  uint32_t next();                                // K for the coming iteration
  void observe(uint32_t k, uint32_t accepted);    // what that iteration did (j of k kept)

  double alpha() const { return s_ / t_; }
  uint32_t current() const { return cur_; }
  uint32_t max_k() const { return max_k_; }

  // E[ids](K) = (1 - alpha^(K+1)) / (1 - alpha); ids per plain-step cost.
  static double expected_ids(double alpha, uint32_t k);
  double rate(double alpha, uint32_t k) const;
  uint32_t best(double alpha) const;

 private:
  AdaptiveKOptions o_;
  uint32_t max_k_;
  double s_, t_;               // weighted accepted and tried drafts (prior included)
  uint32_t iters_ = 0;         // iterations observed
  uint32_t cur_;               // the policy's K
  uint32_t plain_run_ = 0;     // plain iterations since the last draft at K = 0
};

}  // namespace server
