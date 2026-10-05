#include "server/adaptive_k.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace server {

MtpCost MtpCost::bf16_head() { return MtpCost{}; }

// Spec 9 H2 (docs/BENCHMARKS.md, "int8 lm_head"): the int8 head makes drafts 34 % cheaper
// and each verify ~2 ms cheaper; in units of the int8 plain step (32.65 ms). Derived, to be
// replaced by the box's own measurement (spec 8 §10).
MtpCost MtpCost::int8_head() {
  MtpCost c;
  c.verify = {1.00, 1.18, 1.56, 1.79};
  c.draft = {0.13, 0.26, 0.38};
  return c;
}

MtpCost MtpCost::with_draft_vocab(double head_share, double fraction) const {
  if (!(head_share >= 0 && head_share <= 1) || !(fraction > 0 && fraction <= 1))
    throw std::invalid_argument("MtpCost::with_draft_vocab: head_share in [0, 1], fraction in (0, 1]");
  MtpCost c = *this;
  const double scale = (1.0 - head_share) + head_share * fraction;
  for (double& d : c.draft) d *= scale;
  return c;
}

MtpCost MtpCost::with_free_drafts() const {
  MtpCost c = *this;
  c.draft.assign(verify.empty() ? 0 : verify.size() - 1, 0.0);
  return c;
}

namespace {

std::vector<double> parse_list(const std::string& key, const std::string& text) {
  std::vector<double> out;
  size_t at = 0;
  while (at <= text.size()) {
    const size_t comma = std::min(text.find(',', at), text.size());
    const std::string item = text.substr(at, comma - at);
    if (item.empty()) throw std::invalid_argument("--mtp-cost: empty value in '" + key + "'");
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(item.c_str(), &end);
    if (errno != 0 || end != item.c_str() + item.size() || !std::isfinite(v) || v <= 0)
      throw std::invalid_argument("--mtp-cost: '" + item + "' is not a positive cost");
    out.push_back(v);
    at = comma + 1;
  }
  return out;
}

}  // namespace

MtpCost MtpCost::parse(const std::string& text, const MtpCost& base) {
  MtpCost c = base;
  bool seen_verify = false, seen_draft = false;
  size_t at = 0;
  if (text.empty()) throw std::invalid_argument("--mtp-cost: empty");
  while (at <= text.size()) {
    const size_t semi = std::min(text.find(';', at), text.size());
    const std::string part = text.substr(at, semi - at);
    const size_t eq = part.find('=');
    if (eq == std::string::npos)
      throw std::invalid_argument("--mtp-cost: expected verify=... or draft=..., got '" + part + "'");
    const std::string key = part.substr(0, eq);
    std::vector<double> v = parse_list(key, part.substr(eq + 1));
    if (key == "verify" && !seen_verify) {
      seen_verify = true;
      c.verify = std::move(v);
    } else if (key == "draft" && !seen_draft) {
      seen_draft = true;
      c.draft = std::move(v);
    } else {
      throw std::invalid_argument("--mtp-cost: unknown or repeated key '" + key + "'");
    }
    at = semi + 1;
  }
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
