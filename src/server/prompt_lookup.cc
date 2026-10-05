#include "server/prompt_lookup.h"

#include <stdexcept>
#include <string>

namespace server {
namespace {

constexpr uint64_t kBase = 0x9E3779B97F4A7C15ull | 1;   // odd: invertible mod 2^64
constexpr uint32_t kFirstBits = 12;

}  // namespace

PromptLookup::PromptLookup(const PromptLookupOptions& o) : o_(o), levels_(0) {
  if (o_.max_match < kMinMatch || o_.max_match > kLevels[kNumLevels - 1])
    throw std::invalid_argument("PromptLookup: max_match must be in [2, 64], got " +
                                std::to_string(o_.max_match));
  if (o_.max_candidates == 0 || o_.max_steps == 0)
    throw std::invalid_argument("PromptLookup: max_candidates and max_steps must be > 0");
  while (levels_ < kNumLevels && kLevels[levels_] <= o_.max_match) ++levels_;
  for (uint32_t l = 0; l < kNumLevels; ++l) {
    uint64_t p = 1;
    for (uint32_t i = 0; i < kLevels[l]; ++i) p *= kBase;
    pow_[l] = p;
  }
  clear();
}

void PromptLookup::clear() {
  ids_.clear();
  prefix_.assign(1, 0);
  for (uint32_t l = 0; l < kNumLevels; ++l) prev_[l].clear();
  next_boundary_ = 0xFFFFFFFFu;
  rebuild(kFirstBits);
}

uint64_t PromptLookup::hash(uint32_t level, uint32_t end) const {
  const uint32_t k = kLevels[level];
  return prefix_[end + 1] - prefix_[end + 1 - k] * pow_[level];
}

uint32_t PromptLookup::bucket(uint32_t level, uint64_t h) const {
  // A multiplicative mix with the level folded in; the top bits_ bits.
  const uint64_t x = (h ^ (uint64_t(level) << 59)) * 0xD6E8FEB86659FD93ull;
  return static_cast<uint32_t>(x >> (64 - bits_));
}

void PromptLookup::insert(uint32_t level, uint32_t end) {
  const uint32_t b = bucket(level, hash(level, end));
  prev_[level][end] = head_[level][b];
  head_[level][b] = end;
}

void PromptLookup::rebuild(uint32_t bits) {
  bits_ = bits;
  for (uint32_t l = 0; l < levels_; ++l) {
    head_[l].assign(size_t(1) << bits_, kNone);
    prev_[l].assign(ids_.size(), kNone);
    for (uint32_t p = kLevels[l] - 1; p < ids_.size(); ++p) insert(l, p);
  }
}

void PromptLookup::push(uint32_t id) {
  if (ids_.size() >= kBoundaryFloor) throw std::length_error("PromptLookup: too many ids");
  ids_.push_back(id);
  prefix_.push_back(prefix_.back() * kBase + uint64_t(id) + 1);
  const uint32_t end = static_cast<uint32_t>(ids_.size() - 1);
  for (uint32_t l = 0; l < levels_; ++l) prev_[l].push_back(kNone);
  // Twice as many buckets as ids: chains hold few strangers.
  if ((ids_.size() << 1) > (size_t(1) << bits_)) {
    rebuild(bits_ + 1);   // inserts every position, this one included
    return;
  }
  for (uint32_t l = 0; l < levels_; ++l)
    if (end + 1 >= kLevels[l]) insert(l, end);
}

void PromptLookup::append(uint32_t id) {
  if (id >= kBoundaryFloor)
    throw std::invalid_argument("PromptLookup::append: id " + std::to_string(id) +
                                " is in the boundary range");
  push(id);
}

void PromptLookup::append(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) append(id);
}

void PromptLookup::boundary() {
  if (next_boundary_ < kBoundaryFloor) throw std::length_error("PromptLookup: too many boundaries");
  push(next_boundary_--);
}

void PromptLookup::truncate(size_t n) {
  while (ids_.size() > n) {
    const uint32_t end = static_cast<uint32_t>(ids_.size() - 1);
    for (uint32_t l = 0; l < levels_; ++l) {
      if (end + 1 >= kLevels[l]) {
        uint32_t& head = head_[l][bucket(l, hash(l, end))];
        if (head != end) throw std::logic_error("PromptLookup::truncate: chain order broken");
        head = prev_[l][end];
      }
      prev_[l].pop_back();
    }
    ids_.pop_back();
    prefix_.pop_back();
  }
}

PromptLookup::Match PromptLookup::match(uint32_t min_n) const {
  if (min_n < kMinMatch || min_n > o_.max_match)
    throw std::invalid_argument("PromptLookup::match: n must be in [2, max_match " +
                                std::to_string(o_.max_match) + "], got " + std::to_string(min_n));
  Match r;
  const uint32_t t = static_cast<uint32_t>(ids_.size());
  if (t < 2) return r;
  const uint32_t last = t - 1;
  const uint32_t* s = ids_.data();
  for (int l = static_cast<int>(levels_) - 1; l >= 0; --l) {
    const uint32_t k = kLevels[l];
    if (k > last) continue;   // a candidate needs k <= e + 1 <= t - 1
    // Below the largest level <= n every candidate's m is < n.
    if (static_cast<uint32_t>(l) + 1 < levels_ && kLevels[l + 1] <= min_n) break;
    const uint64_t h = hash(static_cast<uint32_t>(l), last);
    uint32_t best = kNone, best_m = 0, found = 0, steps = 0;
    for (uint32_t c = prev_[l][last]; c != kNone; c = prev_[l][c]) {
      if (steps == o_.max_steps || found == o_.max_candidates) {
        r.truncated = true;
        break;
      }
      ++steps;
      if (hash(static_cast<uint32_t>(l), c) != h) continue;
      bool equal = true;
      for (uint32_t i = 0; i < k && equal; ++i) equal = s[c - i] == s[last - i];
      if (!equal) continue;
      ++found;
      uint32_t m = k;
      while (m < o_.max_match && m <= c && s[c - m] == s[last - m]) ++m;
      if (m > best_m) {
        best = c;
        best_m = m;
        if (m == o_.max_match) break;
      }
    }
    r.steps += steps;
    if (found == 0) continue;
    r.length = best_m;
    if (best_m >= min_n) r.end = best;
    return r;
  }
  return r;
}

std::vector<uint32_t> PromptLookup::propose(uint32_t max_k, uint32_t min_n, Match* info) const {
  const Match m = match(min_n);
  if (info != nullptr) *info = m;
  std::vector<uint32_t> out;
  if (m.end == kNone || max_k == 0) return out;
  const size_t t = ids_.size();
  out.reserve(max_k);
  for (uint32_t j = 1; j <= max_k; ++j) {
    const size_t at = size_t(m.end) + j;
    const uint32_t id = at < t ? ids_[at] : out[at - t];
    if (id >= kBoundaryFloor) break;
    out.push_back(id);
  }
  return out;
}

}  // namespace server
