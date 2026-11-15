#include "server/lookup_in_dflash.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace server {

LookupInDFlash::LookupInDFlash(const LookupInDFlashOptions& o) : o_(o) {
  if (o_.max_k == 0) throw std::invalid_argument("LookupInDFlash: max_k must be >= 1");
}

std::vector<uint32_t> LookupInDFlash::propose(const PromptLookup& m, uint32_t cap_k) {
  // The pending id sits at pos = size() - 1; a source must lie before pos - window.
  const uint32_t pos = m.size() > 0 ? static_cast<uint32_t>(m.size()) - 1 : 0;
  const uint32_t source_before = pos > o_.window ? pos - o_.window : 0;
  uint32_t k = o_.max_k;
  if (cap_k > 0) k = std::min(k, cap_k);
  const uint32_t min_n = std::min(lookup_ ? kStayMatch : kEnterMatch, m.options().max_match);
  std::vector<uint32_t> d;
  if (lookup_ && hot_ && o_.wide_k > k && source_before > 0) {   // open-work 28: a hot run goes wide where it can
    d = m.propose(o_.wide_k, min_n, nullptr, source_before);
    if (d.size() < kWideMin && d.size() > k) d.resize(k);
  } else if (source_before > 0) {
    d = m.propose(k, min_n, nullptr, source_before);
  }
  if (d.empty()) {
    if (lookup_) ++exits;   // the match is gone: back to DFlash
    lookup_ = false;
    return d;
  }
  if (!lookup_) ++enters;
  lookup_ = true;
  return d;
}

void LookupInDFlash::observe(uint32_t proposed, uint32_t kept) {
  if (proposed == 0 || kept > proposed)
    throw std::invalid_argument("LookupInDFlash::observe: " + std::to_string(kept) + " kept of " +
                                std::to_string(proposed) + " lookup drafts");
  ++iterations;
  drafted += proposed;
  accepted += kept;
  const bool was_wide = proposed > o_.max_k;
  wide += was_wide;
  hot_ = was_wide ? kept >= kWideMin : kept == proposed;
  if (kept < std::min(kStayAccepted, proposed)) {
    lookup_ = hot_ = false;
    ++exits;
  }
}

std::vector<uint32_t> LookupInDFlash::candidate(const PromptLookup& m) const {
  const uint32_t pos = m.size() > 0 ? static_cast<uint32_t>(m.size()) - 1 : 0;
  const uint32_t source_before = pos > o_.window ? pos - o_.window : 0;
  if (lookup_ || o_.wide_k == 0 || source_before == 0) return {};
  std::vector<uint32_t> d = m.propose(o_.wide_k, kConfirmMatch, nullptr, source_before);
  if (d.size() < kWideMin) d.clear();
  return d;
}

void LookupInDFlash::confirmed() {
  if (lookup_) throw std::logic_error("LookupInDFlash::confirmed: already in lookup");
  lookup_ = hot_ = true;
  ++enters;
  ++confirms;
}

}  // namespace server
