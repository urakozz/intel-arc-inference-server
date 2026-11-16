#include "server/adaptive_k.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace server {

MtpCost MtpCost::bf16_head() { return MtpCost{}; }

// The int8 head, in units of the int8 plain step. Verify: spec 28b's gemm_sm rows, measured 2026-11-01 by
// probe_mtp_steps' interleaved pairs at depth 4096 (1.037 / 1.086 / 1.131; before 28b 1.18 / 1.56 / 1.79, derived from
// spec 9 H2); spec 19 v3d's GDN step at M >= 2 (2026-11-13, the same probe, busy box): 1.039 / 1.073 / 1.100. Drafts: spec 9 H2 (34 % cheaper than bf16's),
// which the same run measured at 0.133 / 0.256 / 0.369.
MtpCost MtpCost::int8_head() {
  MtpCost c;
  c.verify = {1.00, 1.04, 1.07, 1.10};
  c.draft = {0.13, 0.26, 0.38};
  return c;
}

// Spec 17d: tensor parallel (`--parallel tp`), in units of the K = 0 iteration (verify(0) + commit: 18.93 ms at depth
// 4096). Measured 2026-11-10 by tp_mtp_test `cost` (medians of 5 interleaved rounds of 16, box shared with a CPU
// job): verify M = 1..4 18.93 / 20.60 / 21.43 / 22.66 ms, drafts k = 1..3 3.01 / 6.02 / 9.00 ms at 4096 (at 32768:
// 1.010 / 1.070 / 1.128 and 0.151 a draft). The M-row crossings make TP's verify rows dearer than one card's
// (1.04 / 1.09 / 1.13); the draft (the whole head's layer plus the card's half of the int8 head) is 0.16 of a TP step
// against one card's 0.13.
MtpCost MtpCost::tp_int8_head() {
  MtpCost c;
  c.verify = {1.00, 1.09, 1.13, 1.20};
  c.draft = {0.16, 0.32, 0.48};
  return c;
}

// PROVISIONAL (replaced by the measured table below this task's sessions).
MtpCost MtpCost::flash_next() {
  MtpCost c;
  c.verify = {1.00, 1.45, 1.90, 2.35};
  c.draft = {0.12, 0.24, 0.36};
  return c;
}

MtpCost MtpCost::tp_dflash() {
  MtpCost c;
  // tp_dflash_test cost 4096, 2026-11-14 (both cards held, the CPU reference job running; medians of 5 interleaved
  // rounds of 16): verify(k) + commit at M = k + 1 in K = 0 iterations (19.083 ms), and the draft (the deferred
  // projection, both cards' draft lists, the host merge) 3.697 ms whatever K - the block runs whole.
  c.verify = {1.000, 1.081, 1.104, 1.156, 1.238, 1.311, 1.367, 1.417};
  c.draft.assign(7, 3.697 / 19.083);
  return c;
}

MtpCost MtpCost::dflash(bool i8_head, unsigned drafter_format, bool candidate_head) {
  MtpCost c;
  // H1/B4 paired medians at depth4096, busy box, 2026-11-12 (docs/BENCHMARKS.md); the candidate head's rows re-measured
  // with spec 19 v3d's final form (2026-11-14, BENCHMARKS "Round cost (spec 19 v3d)").
  // The bf16 and int4 verify rows are derived equal to int8 (the packed drafter's measured 1.224 at 8 rows, 2026-11-14).
  c.verify = {1.000091, 1.048509, 1.102680, 1.123232, 1.170806, 1.199997, 1.228823, 1.246703};
  if (candidate_head)
    c.verify = {1.000230, 1.041410, 1.073907, 1.105789, 1.151472, 1.180781, 1.205189, 1.221471};
  const double body = drafter_format == 2 ? kDFlashDrafterInt4 :
                       drafter_format == 1 ? kDFlashDrafterInt8 : kDFlashDrafterBf16;
  double d = body + (i8_head ? kDFlashHeadInt8 : kDFlashHeadBf16);
  if (candidate_head) {
    d = drafter_format == 2 ? kDFlashDraftCandidateInt4 : kDFlashDraftCandidateInt8;
    if (drafter_format == 0) d += kDFlashDrafterBf16 - kDFlashDrafterInt8;
  }
  c.draft.assign(7, d);
  return c;
}

double MtpCost::dflash_head_share(bool i8_head, unsigned drafter_format) {
  const double h = i8_head ? kDFlashHeadInt8 : kDFlashHeadBf16;
  const double body = drafter_format == 2 ? kDFlashDrafterInt4 :
                       drafter_format == 1 ? kDFlashDrafterInt8 : kDFlashDrafterBf16;
  return h / (h + body);
}

MtpCost MtpCost::with_draft_vocab(double head_share, double fraction) const {
  if (!(head_share >= 0 && head_share <= 1) || !(fraction > 0 && fraction <= 1))
    throw std::invalid_argument("MtpCost::with_draft_vocab: head_share in [0, 1], fraction in (0, 1]");
  MtpCost c = *this;
  const double scale = (1.0 - head_share) + head_share * fraction;
  for (double& d : c.draft) d *= scale;
  return c;
}

MtpCost MtpCost::bolt_verify(bool i8_head, bool union_moe) {
  MtpCost c;
  c.verify.clear();
  for (uint32_t M = 1; M <= 8; ++M) {
    double v = 1.0 + kBoltPlainRow * double(M - 1);
    if (union_moe) {   // the experts the union does not read again (40 MoE blocks of 2.5 MB)
      const double u = kBoltUnionM1 + (kBoltUnionM8 - kBoltUnionM1) * double(M - 1) / 7.0;
      v -= (9.0 * M - u) * 40.0 * kBoltExpertGB / kBoltPlainGB;
    }
    if (!i8_head) v = (v * kBoltPlainGB + kBoltHeadInt8GB) / (kBoltPlainGB + kBoltHeadInt8GB);
    c.verify.push_back(std::round(v * 1000.0) / 1000.0);
  }
  c.draft.assign(7, 0.0);
  return c;
}

MtpCost MtpCost::bolt_lilicorr(bool i8_head, unsigned drafter_format, bool union_moe) {
  if (drafter_format > 2) throw std::invalid_argument("MtpCost::bolt_lilicorr: drafter format 0..2");
  MtpCost c = bolt_verify(i8_head, union_moe);
  double d = kBoltDraft[drafter_format];
  if (!i8_head) d = (d * kBoltPlainGB + kBoltHeadInt8GB) / (kBoltPlainGB + kBoltHeadInt8GB);
  c.draft.assign(7, std::round(d * 1000.0) / 1000.0);
  return c;
}

MtpCost MtpCost::kolibri_dspark(bool i8_head) {
  const double plain = kKolPlainGB + (i8_head ? kKolHeadInt8GB : kKolHeadBf16GB);
  MtpCost c;
  c.verify.clear();
  for (uint32_t M = 1; M <= 8; ++M) c.verify.push_back(std::round((1.0 + double(M - 1) * kKolRowGB / plain) * 1000.0) / 1000.0);
  const double d = (kKolDrafterGB + (i8_head ? kKolHeadInt8GB : kKolHeadBf16GB)) / plain;
  c.draft.assign(7, std::round(d * 1000.0) / 1000.0);
  return c;
}

MtpCost MtpCost::with_free_drafts() const {
  MtpCost c = *this;
  c.draft.assign(verify.empty() ? 0 : verify.size() - 1, 0.0);
  return c;
}

uint32_t MtpCost::max_k() const {
  if (verify.empty()) return 0;
  return static_cast<uint32_t>(std::min(verify.size() - 1, draft.size()));
}

double MtpCost::cost(uint32_t k) const {
  if (k == 0) return 1.0;
  if (k > max_k()) throw std::out_of_range("MtpCost: no cost for K = " + std::to_string(k));
  return verify[k] + draft[k - 1];
}

AdaptiveK::AdaptiveK(const AdaptiveKOptions& opts)
    : o_(opts),
      max_k_(std::min(opts.max_k, opts.cost.max_k())),
      s_(opts.prior * opts.prior_weight),
      t_(opts.prior_weight),
      cur_(max_k_) {
  if (!(opts.prior_weight > 0) || !(opts.prior > 0 && opts.prior <= 1))
    throw std::invalid_argument("AdaptiveK: the prior needs 0 < prior <= 1 and a weight > 0");
}

double AdaptiveK::expected_ids(double alpha, uint32_t k) {
  if (alpha >= 1.0) return k + 1.0;
  return (1.0 - std::pow(alpha, static_cast<double>(k + 1))) / (1.0 - alpha);
}

double AdaptiveK::rate(double alpha, uint32_t k) const {
  return expected_ids(alpha, k) / o_.cost.cost(k);
}

uint32_t AdaptiveK::best(double alpha) const {
  uint32_t b = 0;
  double r = rate(alpha, 0);
  for (uint32_t k = 1; k <= max_k_; ++k) {
    const double rk = rate(alpha, k);
    if (rk > r) {
      b = k;
      r = rk;
    }
  }
  return b;
}

uint32_t AdaptiveK::next() {
  if (iters_ < o_.warmup) return max_k_;
  if (cur_ == 0 && max_k_ > 0 && o_.probe_every > 0 && plain_run_ >= o_.probe_every) return 1;
  return cur_;
}

void AdaptiveK::observe(uint32_t k, uint32_t accepted) {
  if (accepted > k)
    throw std::invalid_argument("AdaptiveK::observe: " + std::to_string(accepted) +
                                " drafts kept of " + std::to_string(k));
  ++iters_;
  if (k == 0) {
    ++plain_run_;   // no drafts, nothing learned
  } else {
    plain_run_ = 0;
    // The drafts after the first rejection were never tried.
    const double tried = accepted < k ? accepted + 1.0 : static_cast<double>(k);
    s_ = o_.decay * s_ + accepted;
    t_ = o_.decay * t_ + tried;
  }
  if (iters_ < o_.warmup) return;
  const double a = alpha();
  const uint32_t b = best(a);
  if (b != cur_ && rate(a, b) > rate(a, cur_) * (1.0 + o_.hysteresis)) cur_ = b;
}

}  // namespace server
