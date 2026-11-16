#include "server/lookup_in_dflash.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace server {

LookupInDFlash::LookupInDFlash(const LookupInDFlashOptions& o) : o_(o), width_(o.wide_k) {
  if (o_.max_k == 0) throw std::invalid_argument("LookupInDFlash: max_k must be >= 1");
  if (o_.deep_k != 0 && o_.deep_k <= o_.wide_k)
    throw std::invalid_argument("LookupInDFlash: deep_k must exceed wide_k");
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
    d = m.propose(width_, min_n, nullptr, source_before);   // spec 32: the width the run has earned
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
  deep += o_.deep_k != 0 && proposed > o_.wide_k;
  hot_ = was_wide ? kept >= kWideMin : kept == proposed;
  // Spec 32: a wide verify kept whole lets the next one ask for deep_k (the adapter checks the drafts in the narrowest
  // list that holds them: the source's remaining continuation decides the width), one kept at least half drops a
  // width (63 -> 31 -> 15), any other verify (or a cold run) goes back to wide_k. Spec 32e: two wide misses in a row
  // keep the request at wide_k.
  if (was_wide) {
    misses_ = kept < proposed ? misses_ + 1 : 0;
    if (misses_ >= 2 && !narrow_ && o_.deep_k != 0) {
      narrow_ = true;
      ++narrowed;
    }
  }
  uint32_t w = o_.wide_k;   // the list this verify ran on: the narrowest width holding its drafts
  while (w < proposed && w < o_.deep_k) w = 2 * w + 1;
  width_ = !was_wide || narrow_ || o_.deep_k == 0 ? o_.wide_k
           : kept == proposed                     ? o_.deep_k
           : 2 * kept >= proposed                 ? std::max((w - 1) / 2, o_.wide_k)
                                                  : o_.wide_k;
  if (kept < std::min(kStayAccepted, proposed)) {
    lookup_ = hot_ = false;
    width_ = o_.wide_k;
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
  width_ = o_.wide_k;
  ++enters;
  ++confirms;
}

}  // namespace server
