#pragma once
// Spec 8 §10: `b70-serve --spec mtp` - per request, choose the iteration's draft count K
// from the request's own acceptance so far. Pure host code, unit-tested by
// tests/server/adaptive_k_test.cc; the server keeps one AdaptiveK per request.
#include <cstdint>
#include <inplace_vector>
#include <string>
#include <vector>

namespace server {

// Step costs in units of one plain decode step: verify[m - 1] for a verify of M = m rows
// (verify[0] = the plain step), draft[k - 1] for k drafts. Data, not code: defaults per
// head form, overridable with `the cost table` so the box can calibrate.
struct MtpCost {
  static constexpr uint32_t kMaxRows = 8;   // a verify step's rows at most: runtime::DecodeScratchDims::kM
  // Spec 28b: the verify rows on gemm_sm (the small-M GEMM) - measured 2026-11-01 by probe_mtp_steps at depth 4096,
  // bf16 head (busy-box timing on one card, to be re-checked on a quiet box; before 28b: 1.17 / 1.52 / 1.74).
  std::inplace_vector<double, kMaxRows> verify{1.00, 1.04, 1.09, 1.13};
  std::inplace_vector<double, kMaxRows - 1> draft{0.19, 0.37, 0.55};

  static MtpCost bf16_head();   // measured (verify: spec 28b; drafts: spec 8 P0 / §8)
  static MtpCost int8_head();   // verify measured (spec 28b, interleaved pairs); drafts derived from spec 9 H2
  // Spec 17d: `--parallel tp`'s table (the int8 head's halves, the head's layer whole on each card), in units of the
  // TP plain step - its verify rows cross at M rows and its drafts read half the head, so the ratios move.
  static MtpCost tp_int8_head();
  // Qwen3.8-Flash-Next served on two cards (--parallel pp, the expert tier): every verify row reads its own 10
  // experts a layer, some from host memory, so a row costs far more than on the dense models. In units of the plain
  // step with the head loaded.
  static MtpCost flash_next();
  // Qwen3.8-Flash-Next under --parallel tp: its verify rows read each card's halves of their experts (and cross twice a
  // layer), the head drafts whole on the second card. In units of tp's plain step with the head loaded.
  static MtpCost tp_flash_next();
  // Spec 19 v3c: DFlash under tensor parallel (the packed W4A16 drafter split over both cards, the 2-bit candidate
  // head, fp32 crossings) - tp_dflash_test cost at depth 4096.
  static MtpCost tp_dflash();

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

  // Kolibri DSpark: Kolibri-1's rows with the bf16 DSpark drafter, DERIVED from the bytes a step reads (loader/kolibri1_layout.h
  // at 50 layers, int4 attention; TODO(card): the first card run measures them): a plain step reads 1.7696 GB of layers
  // (attention, router, the top-6 + shared experts) + the head; a verify row adds its own 6 experts a layer (0.6267 GB - the
  // M-row MoE reads each row's experts; the weights of every GEMV are read once for the M rows); a draft pass reads the bf16
  // drafter's 0.6946 GB (fc, 5 layers) + the head once at its 7 rows - flat in K (one block drafts all 7).
  static MtpCost kolibri_dspark(bool i8_head);
  static constexpr double kKolPlainGB = 1.7696, kKolRowGB = 0.6267, kKolDrafterGB = 0.6946;
  static constexpr double kKolHeadInt8GB = 0.3282, kKolHeadBf16GB = 0.6554;

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
  uint32_t probe_every = 4;    // at K = 0: one K = 1 iteration after this many plain ones ...
  uint32_t probe_max = 32;     // ... doubling after each probe that leaves K at 0 at measured costs, up to this many
  // The server times each iteration for observe(): only for an engine whose output does not depend on K (Flash-Next's
  // verify rows are its plain step's arithmetic) - elsewhere a timed K would make a request's output irreproducible.
  bool measure = false;
  MtpCost cost;
};

class AdaptiveK {
 public:
  explicit AdaptiveK(const AdaptiveKOptions& opts);

  uint32_t next();                                // K for the coming iteration
  // What that iteration did (j of k kept) and, when the caller timed it, its seconds: the request's own iterations then
  // replace the table's cost of K (its seconds at K over its plain steps', once it has run both) - an iteration's cost
  // depends on the text (a verify row reads its own experts, some from host memory).
  void observe(uint32_t k, uint32_t accepted, double seconds = 0);

  double alpha() const { return s_ / t_; }
  uint32_t current() const { return cur_; }
  uint32_t max_k() const { return max_k_; }

  // E[ids](K) = (1 - alpha^(K+1)) / (1 - alpha); ids per plain-step cost.
  static double expected_ids(double alpha, uint32_t k);
  double cost(uint32_t k) const;                  // the measured cost of K in plain steps, else the table's
  double rate(double alpha, uint32_t k) const;
  uint32_t best(double alpha) const;

 private:
  AdaptiveKOptions o_;
  uint32_t max_k_;
  double s_, t_;               // weighted accepted and tried drafts (prior included)
  uint32_t iters_ = 0;         // iterations observed
  uint32_t cur_;               // the policy's K
  uint32_t plain_run_ = 0;     // plain iterations since the last draft at K = 0
  uint32_t probe_gap_;         // the plain iterations before the next K = 1 probe
  std::vector<double> sec_;    // [k]: this request's seconds an iteration at K (an EMA, decay); 0 = not run
  bool timed_ = false;         // an iteration came with its seconds
};

}  // namespace server
