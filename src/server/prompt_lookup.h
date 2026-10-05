#pragma once
// Spec 19e (spec 19 §3 C, plan 19e Task 2, Review Focus 2-3): the prompt-lookup proposer's
// matcher. Pure host code over a request's committed ids, no device; unit-tested by
// tests/server/prompt_lookup_test.cc against a brute-force reference.
//
// **What it proposes.** For the indexed sequence s[0 .. t-1] (the request's prompt and the ids
// committed after it, optionally behind earlier requests of the session), an earlier end
// position e < t - 1 matches with length m(e) = the longest common suffix of s[0 .. e] and
// s[0 .. t-1], capped at `max_match`. propose(K, n) picks the e with the largest m(e) and,
// among those, the MOST RECENT (the largest e): in an agentic session the latest copy of a
// file or a tool result is the one an edit quotes. It drafts only when m(e) >= n, and the
// drafts are the continuation s[e + 1], s[e + 2], ... copied forward; when the copy reaches
// the end of s it continues from its own drafts (an LZ77 overlapping copy), so a run of one
// id, or a short period, drafts the run on. A draft never crosses a boundary() (below).
//
// **How (Review Focus 2: O(1) amortised per appended id, never a device stall).** Hash chains
// as zlib keeps them, at six key lengths k in kLevels = 2, 4, 8, 16, 32, 64: for every k, a
// bucket table over a 64-bit rolling hash of the last k ids gives the most recent position
// whose last k ids hash there, and prev[k][p] the one before it in the same bucket.
// - append(id): one prefix hash and six bucket insertions; the tables double (and rebuild)
//   when the ids outgrow half the buckets, so the cost is O(1) amortised. Memory 36 B per id
//   plus the tables (2 to 4 buckets per id, 4 B each, per key length): ~34 MB at 262144 ids.
//   Measured on the Mac (prompt_lookup_test): ~0.23 us per appended id, a proposal with its
//   append ~8 us, at 262144 ids.
// - match(n): from the longest key length down, walk the bucket chain of the context's last k
//   ids, most recent first; a candidate is a position whose last k ids EQUAL the context's
//   (hash, then the ids compared - no false positive survives), and its m is extended past k
//   by comparing ids backwards up to max_match. The first key length that has a candidate
//   holds the answer: every e with m(e) >= k is in that chain, and none reaches the next key
//   length (it would be in that chain instead). Within it the longest m wins, the most recent
//   on ties; a candidate at max_match ends the walk (nothing beats it, nothing more recent is
//   left). Levels below the largest one <= n are never walked (their m < n).
// - Bounded: a level's walk stops after `max_candidates` verified candidates or `max_steps`
//   chain steps (bucket collisions included); Match::truncated then says the answer is the
//   best of the most recent max_candidates, not of all. With the caps unreached the answer
//   equals the brute-force definition above exactly (the test's reference).
//
// **Boundaries.** boundary() appends a unique id (counting down from 0xFFFFFFFF), so no match
// spans two segments and no continuation runs into the next: seeding a request with earlier
// requests of the session is seed ids, boundary(), seed ids, boundary(), ..., the prompt.
// Real ids must be below kBoundaryFloor.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace server {

struct PromptLookupOptions {
  uint32_t max_match = 64;        // m(e) is capped here (2 .. 64; the longest key length)
  uint32_t max_candidates = 64;   // verified candidates per key length's walk
  uint32_t max_steps = 1024;      // chain steps per key length's walk, collisions included
};

class PromptLookup {
 public:
  static constexpr uint32_t kNumLevels = 6;
  static constexpr uint32_t kLevels[kNumLevels] = {2, 4, 8, 16, 32, 64};
  static constexpr uint32_t kMinMatch = 2;                  // the smallest key length
  static constexpr uint32_t kBoundaryFloor = 0xFFF00000u;   // ids >= this are boundaries
  static constexpr uint32_t kNone = 0xFFFFFFFFu;

  explicit PromptLookup(const PromptLookupOptions& o = {});

  void clear();                                   // empty, tables back to their first size
  void append(uint32_t id);                       // throws std::invalid_argument on a boundary id
  void append(const std::vector<uint32_t>& ids);
  void boundary();                                // a segment end (see above)
  // Keep the first n ids (n >= size(): nothing). O(removed ids): the removed positions are
  // the most recent of their buckets, so each unlinks from its chain heads. The next request
  // of a session reuses the index this way - truncate to the common prefix with its prompt,
  // append the rest - instead of re-indexing a long prompt.
  void truncate(size_t n);

  struct Match {
    uint32_t end = kNone;      // the chosen e, kNone when nothing matched >= n
    uint32_t length = 0;       // m(e); with end == kNone, the best m a walked level saw
                               // (< n), 0 when none (informative only)
    uint32_t steps = 0;        // chain steps walked, every level
    bool truncated = false;    // a walk stopped at max_candidates or max_steps
  };
  // n in [kMinMatch, max_match]; throws std::invalid_argument otherwise.
  Match match(uint32_t min_n) const;
  // Up to max_k drafts continuing the match (fewer only at a boundary); empty when no match
  // reaches min_n or max_k is 0.
  std::vector<uint32_t> propose(uint32_t max_k, uint32_t min_n, Match* info = nullptr) const;

  size_t size() const { return ids_.size(); }
  const std::vector<uint32_t>& ids() const { return ids_; }
  const PromptLookupOptions& options() const { return o_; }
  uint32_t table_bits() const { return bits_; }

 private:
  uint64_t hash(uint32_t level, uint32_t end) const;   // the last kLevels[level] ids at end
  uint32_t bucket(uint32_t level, uint64_t h) const;
  void insert(uint32_t level, uint32_t end);
  void push(uint32_t id);
  void rebuild(uint32_t bits);

  PromptLookupOptions o_;
  uint32_t levels_;                  // key lengths <= max_match
  std::vector<uint32_t> ids_;
  std::vector<uint64_t> prefix_;     // prefix_[i] = rolling hash of ids_[0 .. i)
  uint64_t pow_[kNumLevels];         // B^k
  uint32_t bits_ = 0;
  std::vector<uint32_t> head_[kNumLevels];   // [1 << bits_], kNone = empty
  std::vector<uint32_t> prev_[kNumLevels];   // [size()], kNone = end of chain
  uint32_t next_boundary_ = 0xFFFFFFFFu;
};

}  // namespace server
