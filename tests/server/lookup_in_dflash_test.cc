// Spec 19f (spec 19 §13.1): server::LookupInDFlash - the ENTER / STAY / EXIT transitions on scripted acceptances,
// the window filter (a match whose source is inside the drafter's window never enters), the full-depth K and its caps
// (max_k, the engine's cap, a boundary). Host only.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "check.h"
#include "server/lookup_in_dflash.h"
#include "server/prompt_lookup.h"

namespace {

using server::LookupInDFlash;
using server::LookupInDFlashOptions;
using server::PromptLookup;

// `far` fresh ids, then a gap of `gap` other fresh ids, then the first `copied` ids of `far` again: the context's
// tail matches the far span with length `copied` (source at 0 .. copied - 1, continuation at `copied`).
PromptLookup context(uint32_t far, uint32_t gap, uint32_t copied) {
  PromptLookup m;
  for (uint32_t i = 0; i < far; ++i) m.append(1000 + i);
  for (uint32_t i = 0; i < gap; ++i) m.append(100000 + i);
  for (uint32_t i = 0; i < copied; ++i) m.append(1000 + i);
  return m;
}

LookupInDFlashOptions opts(uint32_t window, uint32_t max_k) {
  LookupInDFlashOptions o;
  o.window = window;
  o.max_k = max_k;
  return o;
}

void enter_needs_eight_beyond_the_window() {
  // 8 copied ids, the source 64 + 8 back: beyond a window of 16 - enters, K 7 drafts = the far span's continuation.
  {
    LookupInDFlash p(opts(16, 7));
    const PromptLookup m = context(64, 64, 8);
    const std::vector<uint32_t> d = p.propose(m);
    CHECK_EQ(d.size(), size_t(7));
    for (uint32_t i = 0; i < 7; ++i) CHECK_EQ(d[i], 1008 + i);
    CHECK(p.in_lookup());
    CHECK_EQ(p.enters, uint64_t(1));
  }
  // 7 copied ids: below ENTER's 8.
  {
    LookupInDFlash p(opts(16, 7));
    CHECK(p.propose(context(64, 64, 7)).empty());
    CHECK(!p.in_lookup());
  }
  // The source inside the window: the far span starts at 0, the context is 64 + 4 + 8 = 76 long, a window of 2048
  // covers it all - DFlash sees the text itself, no lookup.
  {
    LookupInDFlash p(opts(2048, 7));
    CHECK(p.propose(context(64, 4, 8)).empty());
    CHECK(!p.in_lookup());
  }
  // The boundary of the window: the continuation's source is position 8 (e = 7); the pending id is the last one, at
  // pos = size() - 1. It enters iff 8 < pos - window.
  for (uint32_t gap : {0u, 1u, 2u, 40u}) {
    const PromptLookup m = context(64, gap, 8);
    const uint32_t window = uint32_t(m.size()) - 1 - 9;   // pos - window = 9: 8 < 9 enters
    LookupInDFlash in(opts(window, 7)), out(opts(window + 1, 7));
    CHECK(!in.propose(m).empty());
    CHECK(out.propose(m).empty());
  }
  std::printf("ENTER: >= 8 ids, the source beyond the window (boundary exact)\n");
}

void stay_and_exit() {
  LookupInDFlash p(opts(16, 7));
  const PromptLookup m = context(200, 64, 8);
  CHECK(!p.propose(m).empty());
  p.observe(7, 7);
  CHECK(p.in_lookup());
  p.observe(7, 3);   // >= 3 stays
  CHECK(p.in_lookup());
  p.observe(2, 2);   // fewer than 3 verified, all kept: stays
  CHECK(p.in_lookup());
  p.observe(7, 2);   // below 3: exit
  CHECK(!p.in_lookup());
  CHECK_EQ(p.exits, uint64_t(1));
  // Re-entry on the next proposal when the match still holds.
  CHECK(!p.propose(m).empty());
  CHECK_EQ(p.enters, uint64_t(2));
  p.observe(1, 0);   // the only draft rejected: exit
  CHECK(!p.in_lookup());
  // In lookup but the match is gone (the context diverged): exit at the proposal, no drafts.
  LookupInDFlash q(opts(16, 7));
  PromptLookup mm = context(200, 64, 8);
  CHECK(!q.propose(mm).empty());
  q.observe(7, 7);
  mm.append(424242);   // a fresh id: no match at all
  CHECK(q.propose(mm).empty());
  CHECK(!q.in_lookup());
  CHECK_EQ(q.exits, uint64_t(1));
  CHECK_EQ(q.iterations, uint64_t(1));
  CHECK_EQ(q.drafted, uint64_t(7));
  CHECK_EQ(q.accepted, uint64_t(7));
  // STAY's match bar is 3, ENTER's 8: a 4-id far match continues a hot lookup but does not start one.
  {
    LookupInDFlash hot(opts(16, 7)), cold(opts(16, 7));
    CHECK(!hot.propose(context(200, 64, 8)).empty());
    hot.observe(7, 7);
    const PromptLookup four = context(200, 64, 4);
    CHECK_EQ(hot.propose(four).size(), size_t(7));
    CHECK(hot.in_lookup());
    CHECK(cold.propose(four).empty());
    hot.observe(7, 1);   // exit
    CHECK(hot.propose(four).empty());   // re-entry needs 8 again
    CHECK(!hot.in_lookup());
    const PromptLookup two = context(200, 64, 2);
    LookupInDFlash hot2(opts(16, 7));
    CHECK(!hot2.propose(context(200, 64, 8)).empty());
    hot2.observe(7, 7);
    CHECK(hot2.propose(two).empty());   // below STAY's 3: exit
    CHECK(!hot2.in_lookup());
  }
  std::printf("STAY >= 3 kept (or every draft of fewer) and a match >= 3, EXIT otherwise; re-entry needs 8\n");
}

void k_caps() {
  // max_k 3, then the engine's cap 2, then a boundary 4 ids after the source.
  {
    LookupInDFlash p(opts(16, 3));
    CHECK_EQ(p.propose(context(200, 64, 8)).size(), size_t(3));
  }
  {
    LookupInDFlash p(opts(16, 7));
    CHECK_EQ(p.propose(context(200, 64, 8), 2).size(), size_t(2));
  }
  {
    LookupInDFlash p(opts(16, 7));
    PromptLookup m;
    for (uint32_t i = 0; i < 12; ++i) m.append(1000 + i);
    m.boundary();
    for (uint32_t i = 0; i < 64; ++i) m.append(100000 + i);
    for (uint32_t i = 0; i < 8; ++i) m.append(1000 + i);
    const std::vector<uint32_t> d = p.propose(m);
    CHECK_EQ(d.size(), size_t(4));   // 1008..1011, then the boundary
    CHECK_EQ(d[3], 1011u);
  }
  // A short context (size() <= window): nothing is beyond the window.
  {
    LookupInDFlash p(opts(2048, 7));
    CHECK(p.propose(context(8, 0, 8)).empty());
  }
  std::printf("K: max_k, the engine's cap, a boundary; a context inside the window never enters\n");
}

}  // namespace

int main() {
  enter_needs_eight_beyond_the_window();
  stay_and_exit();
  k_caps();
  std::printf("lookup_in_dflash_test: ok\n");
  return 0;
}
