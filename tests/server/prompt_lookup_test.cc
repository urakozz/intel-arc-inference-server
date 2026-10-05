// Host-only (spec 19e, plan 19e Task 2, Review Focus 2-3): the prompt-lookup matcher against a
// brute-force reference, its edge cases, and its cost on a 262144-id context. No device.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "server/prompt_lookup.h"

namespace {

using server::PromptLookup;
using server::PromptLookupOptions;

struct Ref {
  uint32_t end = PromptLookup::kNone;
  uint32_t length = 0;   // the best m (also when it is < n)
};

// The definition in prompt_lookup.h, by scanning every earlier end: the longest common
// suffix with the whole sequence, capped; the most recent on ties; boundaries never match
// (they are unique ids, so a plain comparison handles them).
Ref brute(const std::vector<uint32_t>& s, uint32_t n, uint32_t max_match) {
  Ref r;
  const size_t t = s.size();
  if (t < 2) return r;
  uint32_t best = 0;
  size_t best_e = 0;
  for (size_t e = t - 1; e-- > 0;) {
    uint32_t m = 0;
    while (m < max_match && m <= e && s[e - m] == s[t - 1 - m]) ++m;
    if (m > best) {
      best = m;
      best_e = e;
    }
  }
  r.length = best;
  if (best >= n) r.end = static_cast<uint32_t>(best_e);
  return r;
}

std::vector<uint32_t> brute_drafts(const std::vector<uint32_t>& s, uint32_t e, uint32_t max_k) {
  std::vector<uint32_t> out;
  for (uint32_t j = 1; j <= max_k; ++j) {
    const size_t at = size_t(e) + j;
    const uint32_t id = at < s.size() ? s[at] : out[at - s.size()];
    if (id >= PromptLookup::kBoundaryFloor) break;
    out.push_back(id);
  }
  return out;
}

PromptLookupOptions uncapped(uint32_t max_match) {
  PromptLookupOptions o;
  o.max_match = max_match;
  o.max_candidates = 0xFFFFFFFFu;
  o.max_steps = 0xFFFFFFFFu;
  return o;
}

// Every prefix of `s`, every n, every max_match: the uncapped matcher equals the brute force
// exactly; the default-capped one equals it whenever no walk was truncated, and is otherwise
// a true match no longer than the best.
void against_brute(const std::vector<uint32_t>& s, uint64_t& checks, uint64_t& truncated,
                   size_t stride = 1) {
  for (uint32_t max_match : {2u, 3u, 9u, 16u, 64u}) {
    PromptLookup exact(uncapped(max_match));
    PromptLookupOptions capped_o;
    capped_o.max_match = max_match;
    capped_o.max_candidates = 4;   // small, so truncation happens and is exercised
    capped_o.max_steps = 16;
    PromptLookup capped(capped_o);
    std::vector<uint32_t> prefix;
    for (size_t i = 0; i < s.size(); ++i) {
      exact.append(s[i]);
      capped.append(s[i]);
      prefix.push_back(s[i]);
      if (i % stride != 0 && i + 1 != s.size()) continue;
      for (uint32_t n : {2u, 3u, 5u, 8u}) {
        if (n > max_match) continue;
        const Ref want = brute(prefix, n, max_match);
        PromptLookup::Match got;
        const std::vector<uint32_t> d = exact.propose(7, n, &got);
        CHECK(!got.truncated);
        if (got.end != want.end) {
          std::fprintf(stderr, "prefix %zu n %u max_match %u: end %u want %u (len %u want %u)\n",
                       prefix.size(), n, max_match, got.end, want.end, got.length, want.length);
          std::exit(1);
        }
        if (got.end != PromptLookup::kNone) {
          CHECK_EQ(got.length, want.length);
          CHECK(d == brute_drafts(prefix, got.end, 7));
          CHECK(!d.empty() || prefix[got.end + 1] >= PromptLookup::kBoundaryFloor);
        } else {
          CHECK(d.empty());
          CHECK(got.length < n);
        }
        PromptLookup::Match c;
        (void)capped.propose(7, n, &c);
        if (!c.truncated) {
          CHECK_EQ(c.end, want.end);
        } else {
          ++truncated;
          if (c.end != PromptLookup::kNone) {
            // A real match of the length it claims, at least n, at most the best.
            uint32_t m = 0;
            while (m < max_match && m <= c.end && prefix[c.end - m] == prefix[prefix.size() - 1 - m])
              ++m;
            CHECK_EQ(m, c.length);
            CHECK(c.length >= n && c.length <= want.length);
          }
        }
        ++checks;
      }
    }
  }
}

void random_and_repetitive() {
  std::mt19937_64 g(19);
  uint64_t checks = 0, truncated = 0;
  for (uint32_t alphabet : {2u, 3u, 5u, 50u, 1000u}) {
    for (int rep = 0; rep < 12; ++rep) {
      std::uniform_int_distribution<uint32_t> tok(0, alphabet - 1);
      std::vector<uint32_t> s(1 + g() % 220);
      for (uint32_t& x : s) x = tok(g);
      // Repetitive shapes: copied spans (agentic echoes), a run, a short period.
      if (rep % 3 == 1 && s.size() > 10) {
        const size_t a = g() % (s.size() - 5), len = 3 + g() % 80;
        for (size_t i = 0; i < len; ++i) s.push_back(s[a + i % (s.size() - a)]);
      }
      if (rep % 3 == 2) {
        for (int i = 0; i < 30; ++i) s.push_back(7);
        for (int i = 0; i < 40; ++i) s.push_back(uint32_t(i % 3));
      }
      against_brute(s, checks, truncated);
    }
  }
  // A context past the first table size (rebuilds at 2049, 4097 ids), checked sparsely.
  std::vector<uint32_t> big(9000);
  std::uniform_int_distribution<uint32_t> tok(0, 400);
  for (size_t i = 0; i < big.size(); ++i)
    big[i] = (i > 3000 && i % 700 < 300) ? big[i - 2500] : tok(g);   // echoes 2500 back
  against_brute(big, checks, truncated, 97);
  std::printf("against brute force: %llu checks, %llu with a truncated walk (capped matcher)\n",
              static_cast<unsigned long long>(checks), static_cast<unsigned long long>(truncated));
  CHECK(checks > 50000);
  CHECK(truncated > 100);
}

std::vector<uint32_t> ids(std::initializer_list<uint32_t> l) { return std::vector<uint32_t>(l); }

void edge_cases() {
  PromptLookup lk;
  CHECK(lk.propose(3, 2).empty());   // empty
  lk.append(5);
  CHECK(lk.propose(3, 2).empty());   // one id
  lk.clear();
  lk.append(ids({1, 2, 3, 4, 5}));
  CHECK(lk.propose(3, 2).empty());   // no repeat
  // The most recent of two equal-length matches.
  lk.clear();
  lk.append(ids({1, 2, 7, 1, 2, 8, 1, 2}));
  CHECK(lk.propose(1, 2) == ids({8}));
  // A longer, older match beats a shorter, recent one.
  lk.clear();
  lk.append(ids({3, 1, 2, 7, 9, 1, 2, 8, 9, 3, 1, 2}));
  PromptLookup::Match m;
  CHECK(lk.propose(1, 2, &m) == ids({7}));
  CHECK_EQ(m.length, 3u);
  // n above the match: nothing (n = 4 is a key length, so the 2-level is not walked).
  CHECK(lk.propose(3, 4, &m).empty());
  CHECK_EQ(m.end, PromptLookup::kNone);
  CHECK_EQ(m.length, 0u);
  // n = 3 (not a key length) walks the 2-level and finds the 3-long match.
  CHECK(lk.propose(3, 3, &m) == ids({7, 9, 1}));
  // The match at the very end (e = t - 2): a run drafts itself on.
  lk.clear();
  lk.append(ids({9, 4, 4, 4}));
  CHECK(lk.propose(5, 2) == ids({4, 4, 4, 4, 4}));
  // Overlapping repeats: a period of 3 continues through the drafts.
  lk.clear();
  lk.append(ids({1, 2, 3, 1, 2, 3, 1}));
  CHECK(lk.propose(7, 2) == ids({2, 3, 1, 2, 3, 1, 2}));
  // max_k 0.
  CHECK(lk.propose(0, 2).empty());
  // Boundaries: no draft crosses one, no match spans one.
  lk.clear();
  lk.append(ids({1, 2, 3}));
  lk.boundary();
  lk.append(ids({1, 2}));
  CHECK(lk.propose(4, 2) == ids({3}));
  lk.clear();
  lk.append(ids({4, 1, 2}));
  lk.boundary();
  lk.append(ids({2}));   // [boundary, 2]
  CHECK(lk.propose(4, 2).empty());
  // Two boundaries are distinct ids: [b1, x] and [b2, x] never match each other.
  lk.clear();
  lk.boundary();
  lk.append(ids({6, 1}));
  lk.boundary();
  lk.append(ids({6}));
  CHECK(lk.propose(2, 2).empty());
  // Arguments.
  bool threw = false;
  try {
    (void)lk.propose(3, 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    lk.append(PromptLookup::kBoundaryFloor);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    PromptLookupOptions o;
    o.max_match = 65;
    PromptLookup bad(o);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  // max_match below n is refused; max_match 3 keeps only the 2-level and caps at 3.
  PromptLookupOptions o3;
  o3.max_match = 3;
  PromptLookup short3(o3);
  short3.append(ids({1, 2, 3, 4, 9, 2, 3, 4, 8, 1, 2, 3, 4}));
  CHECK(short3.propose(1, 3, &m) == ids({8}));   // capped at 3: both tie, most recent wins
  CHECK_EQ(m.length, 3u);
  std::printf("edge cases: ok\n");
}

// truncate(n) then appending gives the matcher a fresh build of the same ids would: a
// session's next request reuses the index from the common prefix.
void truncate_equals_rebuild() {
  std::mt19937_64 g(7);
  std::uniform_int_distribution<uint32_t> tok(0, 30);
  std::vector<uint32_t> s(6000);
  for (size_t i = 0; i < s.size(); ++i) s[i] = (i > 500 && i % 400 < 150) ? s[i - 333] : tok(g);
  PromptLookup reused;
  reused.append(s);
  for (size_t cut : {5999u, 4000u, 2048u, 2047u, 100u, 1u, 0u}) {
    reused.truncate(cut);
    CHECK_EQ(reused.size(), cut);
    std::vector<uint32_t> tail(s.begin() + long(cut), s.begin() + long(std::min<size_t>(cut + 700, s.size())));
    for (uint32_t& x : tail) x = (x * 7 + 3) % 31;   // a different continuation
    PromptLookup fresh;
    fresh.append(std::vector<uint32_t>(s.begin(), s.begin() + long(cut)));
    for (uint32_t x : tail) {
      reused.append(x);
      fresh.append(x);
      for (uint32_t n : {2u, 3u, 6u}) {
        PromptLookup::Match a, b;
        CHECK(reused.propose(5, n, &a) == fresh.propose(5, n, &b));
        CHECK(a.end == b.end && a.length == b.length);
      }
    }
    reused.truncate(cut);
    reused.append(std::vector<uint32_t>(s.begin() + long(cut), s.end()));   // back to s
  }
  std::printf("truncate + append equals a fresh build: ok\n");
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Review Focus 2: O(1) amortised per appended id and a bounded walk per proposal, on a
// 262144-id context (Qwen3.8's trained length). Three shapes: code-like (a skewed vocabulary
// with long echoes of earlier spans), one id repeated (every chain is the whole context),
// and two ids at random (long chains at every key length).
void cost_262144() {
  constexpr size_t kN = 262144, kSteps = 4096;
  std::mt19937_64 g(262144);
  const auto shape = [&](int which) {
    std::vector<uint32_t> s(kN + kSteps);
    std::geometric_distribution<uint32_t> zipfish(0.002);
    for (size_t i = 0; i < s.size(); ++i) {
      if (which == 1) s[i] = 11;
      else if (which == 2) s[i] = uint32_t(g() & 1);
      else if (i > 20000 && (i / 1500) % 3 == 0) s[i] = s[i - 17000];   // an echo
      else s[i] = zipfish(g) % 248000;
    }
    return s;
  };
  const char* names[3] = {"code-like", "one id repeated", "two ids at random"};
  PromptLookupOptions o;   // the defaults the server runs
  const uint32_t step_bound = PromptLookup::kNumLevels * o.max_steps;
  for (int which = 0; which < 3; ++which) {
    const std::vector<uint32_t> s = shape(which);
    PromptLookup lk(o);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < kN; ++i) lk.append(s[i]);
    const double append_ms = ms_since(t0);
    uint64_t steps = 0, worst = 0, drafted = 0;
    t0 = std::chrono::steady_clock::now();
    for (size_t i = kN; i < kN + kSteps; ++i) {   // decode: propose, then commit one id
      PromptLookup::Match m;
      drafted += lk.propose(7, 3, &m).size();
      steps += m.steps;
      worst = std::max<uint64_t>(worst, m.steps);
      lk.append(s[i]);
    }
    const double step_ms = ms_since(t0);
    std::printf("262144 ids, %s: append %.1f ms (%.3f us/id), %zu proposals %.1f ms (%.2f us "
                "each, incl. one append), chain steps mean %.1f worst %llu (bound %u), "
                "%llu drafts, table 2^%u\n",
                names[which], append_ms, 1e3 * append_ms / kN, kSteps, step_ms,
                1e3 * step_ms / kSteps, double(steps) / kSteps,
                static_cast<unsigned long long>(worst), step_bound,
                static_cast<unsigned long long>(drafted), lk.table_bits());
    CHECK(worst <= step_bound);
    // Generous wall-clock bars (Release; a loaded laptop or the box): a 262144-id prompt
    // indexes in well under a second, a proposal costs far below a 32 ms decode step.
    CHECK(append_ms < 1500.0);
    CHECK(1e3 * step_ms / kSteps < 500.0);
    if (which == 1) CHECK(worst <= 2);   // the 64-level's first candidate is capped: done
  }
}

}  // namespace

int main() {
  edge_cases();
  random_and_repetitive();
  truncate_equals_rebuild();
  cost_262144();
  std::printf("prompt_lookup_test: ok\n");
  return 0;
}
