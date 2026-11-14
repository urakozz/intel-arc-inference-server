#pragma once
// Spec 8 §10: `b70-serve --spec mtp` - per request, choose the iteration's draft count K
// from the request's own acceptance so far. Pure host code, unit-tested by
// tests/server/adaptive_k_test.cc; the server keeps one AdaptiveK per request.
#include <cstdint>
#include <string>
#include <vector>

namespace server {

// Step costs in units of one plain decode step: verify[m - 1] for a verify of M = m rows
// (verify[0] = the plain step), draft[k - 1] for k drafts. Data, not code: defaults per
// head form, overridable with `the cost table` so the box can calibrate.
struct MtpCost {
  // Spec 28b: the verify rows on gemm_sm (the small-M GEMM) - measured 2026-11-01 by probe_mtp_steps at depth 4096,
  // bf16 head (busy-box timing on one card, to be re-checked on a quiet box; before 28b: 1.17 / 1.52 / 1.74).
  std::vector<double> verify{1.00, 1.04, 1.09, 1.13};
  std::vector<double> draft{0.19, 0.37, 0.55};

  static MtpCost bf16_head();   // measured (verify: spec 28b; drafts: spec 8 P0 / §8)
  static MtpCost int8_head();   // verify measured (spec 28b, interleaved pairs); drafts derived from spec 9 H2
  // Spec 17d: `--parallel tp`'s table (the int8 head's halves, the head's layer whole on each card), in units of the
  // TP plain step - its verify rows cross at M rows and its drafts read half the head, so the ratios move.
  static MtpCost tp_int8_head();

  // Spec 8 §11: the share of ONE draft's cost that is its lm_head read, per head form -
  // what a reduced draft vocabulary scales. **Derived, not measured:** the head's bytes at
  // ~570 GB/s in units of that form's plain step, over the table's one-draft cost.
  //   int8: 1.27 GB -> ~2.2 ms of the 32.65 ms int8 plain step = 0.067 of draft[0] 0.13
  //   bf16: 2.54 GB -> ~4.4 ms of the ~34 ms bf16 plain step   = 0.129 of draft[0] 0.19
  // The rest of a draft (the MTP head's own layer, its norms, the argmax) does not move.
  // `probe_mtp_steps ... <form> <size>` on the box replaces the result (via the cost table).
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

  // Spec 19v2: one block drafts all K rows. The int8 H1/B4 full/candidate totals and per-mode verify rows are
  // measured paired medians at depth4096 (2026-11-12, busy box); docs/BENCHMARKS.md. The full-head decomposition
  // uses that profile's head GEMM, and BF16 doubles its head/body GEMM shares, retaining non-linear/host cost.
  // drafter_format: 0 bf16, 1 int8, 2 int4 g64; candidate_head is C3's top64/exact-rerank path.
  static MtpCost dflash(bool i8_head, unsigned drafter_format, bool candidate_head = false);
  static constexpr double kDFlashHeadInt8 = 2.166354 / 29.222498, kDFlashHeadBf16 = 2 * kDFlashHeadInt8;
  static constexpr double kDFlashDrafterInt8 = 0.298136 - kDFlashHeadInt8;
  static constexpr double kDFlashDrafterBf16 = kDFlashDrafterInt8 + 3.185105 / 29.222498;
  // Spec 19 v3d (2026-11-14, card 0, busy box): the DPAS attention at KS 4, the radix candidate selection and the
  // 2-bit head with 80 candidates - the draft pass 0.217462 of a plain step at depth 4096 (was 0.287036).
  static constexpr double kDFlashDraftCandidateInt8 = 0.217462;
  // The full-head int4 row: the experimental PRE-H1/B4 RTN estimate (the draft-vocabulary path only).
  static constexpr double kDFlashDrafterInt4 = 0.269526 - kDFlashHeadInt8;
  // The packed AutoRound drafter (urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound, the default; 2026-11-14, card 0,
  // busy box): the draft pass 0.151481 of a plain step at depth 4096 (4.41 ms; int8 0.210086 in the same run).
  static constexpr double kDFlashDraftCandidateInt4 = 0.151481;
  // The head's share of that draft row - what a draft vocabulary scales (with_draft_vocab).
  static double dflash_head_share(bool i8_head, unsigned drafter_format);

  // Spec 26c: Nemotron 3.5 Bolt's rows, DERIVED (spec 26 §11's table over spec 25 §14's layout bytes, depth 4096; box row 63's
  // r63.cost measures them, `--spec-cost` replaces either part). The verify at M = 1..8 with the plain M-row MoE: 1 + 0.1933 a
  // row (the routed experts' 9 blocks, the slot writes, the head's row - 1.000 .. 2.353); with the union-aware MoE
  // (Decision 10, the default) the experts NOT read again subtracted: (9 M - u(M)) x 40 x 2.5 MB over the 5.524 GB plain step,
  // u(M) ESTIMATED linear from 9 at M 1 to 30 at M 8 (spec 26 §4's u = 30 - r63.union measures it) - M 8 1.593, §11's figure.
  // The draft row flat in K (one 8-row block drafts all 7): exact 0.500, int8 0.306, int4 g64 0.214 plain steps (§11, the
  // int8 head at the 7 mask rows included). The bf16 head adds its 0.537 GB to every verify (read once) and to the draft row.
  // `drafter_format`: 0 exact, 1 int8, 2 int4 g64.
  static MtpCost bolt_verify(bool i8_head, bool union_moe);
  static MtpCost bolt_lilicorr(bool i8_head, unsigned drafter_format, bool union_moe);
  static constexpr double kBoltPlainGB = 5.524, kBoltHeadInt8GB = 0.537, kBoltExpertGB = 0.0025;
  static constexpr double kBoltPlainRow = 1.353 / 7;   // a verify row with the plain M-row MoE (§11: 2.353 at M 8)
  static constexpr double kBoltDraft[3] = {0.500, 0.306, 0.214};
  static constexpr double kBoltUnionM1 = 9.0, kBoltUnionM8 = 30.0;

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
