#pragma once
// Spec 8 §10: `b70-serve --spec mtp` - per request, choose the iteration's draft count K
// from the request's own acceptance so far. Pure host code, unit-tested by
// tests/server/adaptive_k_test.cc; the server keeps one AdaptiveK per request.
#include <array>
#include <cstdint>
#include <functional>
#include <inplace_vector>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace server {

// Step costs in units of one plain decode step: verify[m - 1] for a verify of M = m rows
// (verify[0] = the plain step), draft[k - 1] for k drafts. Data, not code: defaults per
// head form, overridable with `the cost table` so the test machine can calibrate.
struct MtpCost {
  static constexpr uint32_t kMaxRows = 8;   // a verify step's rows at most: runtime::DecodeScratchDims::kM
  // Spec 28b: the verify rows on gemm_sm (the small-M GEMM) - measured 2026-11-01 by probe_mtp_steps at depth 4096,
  // bf16 head (busy-machine timing on one card, to be re-checked on a quiet machine; before 28b: 1.17 / 1.52 / 1.74).
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

  // Spec 19e: the same verify rows with a free draft - prompt lookup proposes on the host in
  // microseconds (prompt_lookup.h), ~0 of a 32 ms plain step. draft[k - 1] = 0 for every K
  // the verify list covers.
  MtpCost with_free_drafts() const;

  // Spec 19v2: one block drafts all K rows. The int8 H1/B4 full/candidate totals and per-mode verify rows are
  // measured paired medians at depth4096 (2026-11-12, busy machine); the benchmark record. The full-head decomposition
  // uses that profile's head GEMM, and BF16 doubles its head/body GEMM shares, retaining non-linear/host cost.
  // drafter_format: 0 bf16, 1 int8, 2 int4 g64; candidate_head is C3's top64/exact-rerank path.
  static MtpCost dflash(bool i8_head, unsigned drafter_format, bool candidate_head = false);
  static constexpr double kDFlashHeadInt8 = 2.166354 / 29.222498, kDFlashHeadBf16 = 2 * kDFlashHeadInt8;
  static constexpr double kDFlashDrafterInt8 = 0.298136 - kDFlashHeadInt8;
  static constexpr double kDFlashDrafterBf16 = kDFlashDrafterInt8 + 3.185105 / 29.222498;
  // Spec 19 v3d (2026-11-14, card 0, busy machine): the DPAS attention at KS 4, the radix candidate selection and the
  // 2-bit head with 80 candidates - the draft pass 0.217462 of a plain step at depth 4096 (was 0.287036).
  static constexpr double kDFlashDraftCandidateInt8 = 0.217462;
  // The packed AutoRound drafter (urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound, the default; 2026-11-14, card 0,
  // busy machine): the draft pass 0.151481 of a plain step at depth 4096 (4.41 ms; int8 0.210086 in the same run).
  static constexpr double kDFlashDraftCandidateInt4 = 0.151481;

  // Kolibri DSpark: Kolibri-1's rows with the bf16 DSpark drafter, DERIVED from the bytes a step reads (loader/kolibri1_layout.h
  // at 50 layers, int4 attention; TODO(card): the first card run measures them): a plain step reads 1.7696 GB of layers
  // (attention, router, the top-6 + shared experts) + the head; a verify row adds its own 6 experts a layer (0.6267 GB - the
  // M-row MoE reads each row's experts; the weights of every GEMV are read once for the M rows); a draft pass reads the bf16
  // drafter's 0.6946 GB (fc, 5 layers) + the head once at its 7 rows - flat in K (one block drafts all 7).
  static MtpCost kolibri_dspark(bool i8_head);
  static constexpr double kKolPlainGB = 1.7696, kKolRowGB = 0.6267, kKolDrafterGB = 0.6946;
  static constexpr double kKolHeadInt8GB = 0.3282, kKolHeadBf16GB = 0.6554;

  // Inkling 2 Mini's DSpark drafter on one card - b70-decode --profile at depth 4096 (card 1, 2026-11-26), the numbers
  // of runtime::inkling2::kOneCardCost (b70-decode's draft window): a verify of 2..8 rows costs 1.143 .. 1.991 plain
  // steps (15.09 ms), the drafter's pass 4.21 ms = 0.279, flat in K (one pass drafts all 7 rows).
  static MtpCost inkling_dspark();
  // Plan 41k: with the DFlash 2 drafter (runtime::inkling2::kOneCardDflashCost, depth 4096): a verify of 2..8
  // rows 1.170 .. 1.960 plain steps (11.7 ms), the drafter's pass 1.70 ms = 0.145 (the packed int4 drafter, the 2-bit
  // candidate head), flat in K. b70-serve's prior only: it serves on its own measured costs (LiveCost).
  static MtpCost inkling_dflash2();
  // ... under `--parallel tp`, the drafter's head rows split over the cards (runtime::inkling2::kTpDflashCost): 1.097 ..
  // 1.778, the drafter's pass 0.213 (2.13 ms against tp's 10.0 ms plain step).
  static MtpCost inkling_dflash2_tp();
  // Plan 41j: the same under `--parallel tp` (both cards, depth 4096, 2026-11-26, the drafter's layers split): 1.102 ..
  // 1.835 plain steps for 2..8 rows, the drafter's pass 0.310 (served tp's table by depth: kTpCosts).
  static MtpCost inkling_dspark_tp();

  uint32_t max_k() const;              // the deepest K both lists cover
  double cost(uint32_t k) const;       // K = 0: 1 (a plain step)
};

// Plan 40a Task 5: one step of several sequences' rows (runtime::Engine::verify_rows and its drafts) in plain steps,
// for the scheduler's deal (BatchEngineIface::row_cost). The verify by its rows: the 2..8-row lists, the 16- and 32-row
// ones (padded) - measured 2026-11-22 at depth 4096, four sequences, int8 KV and head (probe_pool_speed --wide, A B A B:
// 1.44 / 1.74 / 2.13 at 8 / 16 / 32 rows; at 2..7 rows the line to the plain step's 1.00). The drafts: the MTP head one
// step a draft index over the rows drafting that far (`mtp_draft`, the int8 head's draft[0]); the DFlash drafter one
// batched pass of its 8 rows or its four blocks (`dflash8`, `dflash32`: the packed int4 drafter, 4.55 / 7.20 ms against
// the lone K 7 draft's 4.49 ms = 0.217 plain - dflash_pool_test T, card 0, the other card busy, 2026-11-22).
// PERF-120: `v8` / `v16` / `v32` the verify at 8 / 16 / 32 rows (Qwen3.8's above); ornith() Ornith's table.
struct RowCost {
  bool dflash = false;
  double mtp_draft = 0.13, dflash8 = 0.22, dflash32 = 0.35;
  double v8 = 1.44, v16 = 1.74, v32 = 2.13;
  double verify(uint32_t rows) const;
  double step(const std::vector<uint32_t>& ks) const;   // one entry a row: its drafts
  // Ornith (MTP, its MoE head): measured 2026-11-23 at depth 4096, four sequences, int8 KV and head, card 0 beside the
  // other card's quantisation (probe_pool_speed --wide, the union-read MoE): 2.28 / 3.03 / 4.49 plain steps at 8 / 16
  // / 32 rows; the head's draft over four rows 1.81 ms against the 7.85 ms plain step.
  static RowCost ornith() { return {.mtp_draft = 0.23, .v8 = 2.28, .v16 = 3.03, .v32 = 4.49}; }
};

// Plan 40a Task 5, the scheduler's deal over a cost table: the drafts go one at a time to the row whose next draft is
// likeliest kept (alpha^(k+1) - its request's acceptance; the earlier row on a tie), each up to its `want`, so every
// count of drafts has its most expected ids; the deal keeps the count with the most expected ids per `cost` of the
// step, its rows (each draft one, each row's pending id one) within `budget`. Rows in age order; returns their drafts.
std::vector<uint32_t> deal_by_cost(const std::vector<double>& alpha, const std::vector<uint32_t>& want, uint32_t budget,
                                   const std::function<double(const std::vector<uint32_t>&)>& cost);

// A (depth, table) list ascending (AdaptiveKOptions::by_depth): the table at pos, interpolated between the depths, the
// ends held. Every table has the first's rows.
MtpCost cost_at_depth(const std::vector<std::pair<uint32_t, MtpCost>>& by_depth, uint32_t pos);

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
  // Plan 41f (Inkling): an untimed request's probe whose draft is rejected doubles the gap to the next (a kept one
  // keeps probe_every), not only a timed request's that did not pay - otherwise it probes every probe_every plain
  // steps for good; and with no warm-up the first iteration's K is the prior's best, not the deepest.
  bool probe_backoff = false;
  MtpCost cost;
  // Costs that move with the context (Inkling under pp: its drafter's full layer and the full visits read every
  // position): (depth, table) ascending; next(pos) uses the table interpolated at pos (the ends held) in place of cost.
  // Each table has cost's rows.
  std::vector<std::pair<uint32_t, MtpCost>> by_depth;
  // The costs the server measures as it serves (LiveCost::at), in place of cost and by_depth when set.
  std::function<MtpCost(uint32_t pos)> live;
};

// A speculative round's costs as the server measures them (Inkling 2 Mini): the wall time of each verify width (1..8
// rows, 1 the plain step) and of the draft pass at each K, a running average over the last ~kWindow rounds at depths
// 4096 x 2^i (4,096 .. 1,048,576; a round counts at the nearest), in plain steps of the same depth, interpolated
// between the depths at a round's position. `prior` (a (depth, table) list ascending, plain-step units, the rows
// MtpCost::kMaxRows) stands in for what no round has timed yet - a width, a K's draft or a depth - and for the plain
// step's own time at a depth only drafted rounds reached (there: each timed width's time over its prior ratio,
// averaged). seed() enters a load-time measurement at its depth and scales the prior by its ratios to it. A time over
// 4x its width's average (a stall) is not counted. Locked: replica's two cards report into one.
class LiveCost {
 public:
  static constexpr uint32_t kRows = MtpCost::kMaxRows, kFirstDepth = 4096, kDepths = 9, kWindow = 32;
  explicit LiveCost(std::vector<std::pair<uint32_t, MtpCost>> prior);
  // verify_ms[m - 1]: an m-row verify; draft_ms: the draft pass at K kRows - 1.
  void seed(uint32_t pos, const std::array<double, kRows>& verify_ms, double draft_ms);
  void verify(uint32_t pos, uint32_t rows, double ms);
  void draft(uint32_t pos, uint32_t k, double ms);
  MtpCost at(uint32_t pos) const;

 private:
  struct Mean {
    double ms = 0;
    uint32_t n = 0;
    void add(double x);
  };
  struct Depth {
    std::array<Mean, kRows> verify;
    std::array<Mean, kRows - 1> draft;
  };
  MtpCost cost(uint32_t d) const;   // depth d's, under the lock
  std::vector<std::pair<uint32_t, MtpCost>> prior_;
  std::array<Depth, kDepths> d_{};
  mutable std::mutex mu_;
};

class AdaptiveK {
 public:
  explicit AdaptiveK(const AdaptiveKOptions& opts);

  uint32_t next(uint32_t pos = 0);                // K for the coming iteration at the request's position
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
