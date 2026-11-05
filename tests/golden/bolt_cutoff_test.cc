// bolt_cutoff_test - host only: the first-flip cut-off of Bolt's synthetic gates (tests/golden/bolt_cutoff.h; operator
// ruling 2026-11-05, spec 25 §9.1): the classes, the flip's place, upstream / row_clean, Beyond before and after the flip,
// the disabled (real checkpoint) rule.
#include <cstdint>
#include <cstdio>

#include "bolt_cutoff.h"
#include "check.h"

using bolt_cutoff::Cut;
using bolt_cutoff::Diff;
using bolt_cutoff::classify;

int main() {
  constexpr float kTol = 1e-3f;
  const uint32_t a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int32_t same[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};      // the same set, another order
  const int32_t one[9] = {1, 2, 3, 4, 5, 6, 7, 8, 10};      // one expert differs
  const int32_t two[9] = {1, 2, 3, 4, 5, 6, 7, 11, 10};     // two differ

  // 1. the classes
  CHECK(classify(a, same, 9, 5e-4f, kTol) == Diff::Exact);
  CHECK(classify(a, same, 9, 0.0f, kTol) == Diff::Exact);        // equal sets are exact whatever the gap
  CHECK(classify(a, one, 9, 0.0f, kTol) == Diff::Undetermined);  // an exact reference tie
  CHECK(classify(a, two, 9, 0.0f, kTol) == Diff::Undetermined);
  CHECK(classify(a, one, 9, 1e-3f, kTol) == Diff::NearTie);      // at the tolerance: a near-tie
  CHECK(classify(a, one, 9, 2.7e-5f, kTol) == Diff::NearTie);
  CHECK(classify(a, one, 9, 1.0001e-3f, kTol) == Diff::Beyond);  // above it: a defect
  CHECK(classify(a, two, 9, 5e-4f, kTol) == Diff::Beyond);       // two experts: a defect at any positive gap

  // 2. no flip: everything upstream, a Beyond fails
  {
    Cut c;
    for (uint32_t r = 0; r < 4; ++r)
      for (uint32_t l : {1u, 3u, 5u}) c.add(r, l, Diff::Exact);
    CHECK(!c.found);
    CHECK(c.upstream(1000, 0) && c.row_clean(1000));
    c.add(4, 1, Diff::Beyond);
    CHECK_EQ(c.bad_before, uint64_t(1));
    CHECK(!c.found);   // a Beyond is not a flip
  }

  // 3. the flip at (row 2, block 5): upstream is lexicographic, row_clean needs the whole row
  {
    Cut c;
    c.add(0, 1, Diff::Exact);
    c.add(0, 3, Diff::Exact);
    c.add(1, 1, Diff::Exact);
    c.add(2, 1, Diff::Exact);
    c.add(2, 3, Diff::Exact);
    CHECK(c.add(2, 5, Diff::NearTie) == Diff::NearTie);
    CHECK(c.found && c.row == 2 && c.block == 5);
    CHECK(c.upstream(1, 11));     // an earlier row, any block
    CHECK(c.upstream(2, 4));      // the flip's row, a block before it
    CHECK(!c.upstream(2, 5));     // the flipped block's own output
    CHECK(!c.upstream(2, 8));
    CHECK(!c.upstream(3, 0));     // every later row
    CHECK(c.row_clean(1) && !c.row_clean(2) && !c.row_clean(3));
    // after the flip: a Beyond is printed, a later near-tie does not move the flip
    c.add(2, 8, Diff::Beyond);
    c.add(3, 1, Diff::NearTie);
    c.add(3, 3, Diff::Undetermined);
    CHECK_EQ(c.bad_before, uint64_t(0));
    CHECK_EQ(c.bad_after, uint64_t(1));
    CHECK(c.row == 2 && c.block == 5);
    CHECK_EQ(c.ties, uint64_t(2));
    CHECK_EQ(c.undetermined, uint64_t(1));
    CHECK_EQ(c.pairs, uint64_t(9));
  }

  // 4. a Beyond BEFORE the flip still fails; an undetermined pair is a flip too
  {
    Cut c;
    c.add(0, 1, Diff::Beyond);
    c.add(0, 3, Diff::Undetermined);
    CHECK(c.found && c.row == 0 && c.block == 3);
    CHECK_EQ(c.bad_before, uint64_t(1));
    CHECK(c.upstream(0, 1) && !c.upstream(0, 3) && !c.row_clean(0));
  }

  // 5. disabled (a real checkpoint): the old rule - no cut-off, every Beyond binds, near-ties allowed
  {
    Cut c;
    c.enabled = false;
    c.add(0, 1, Diff::NearTie);
    c.add(0, 3, Diff::Undetermined);
    c.add(5, 1, Diff::Beyond);
    CHECK(!c.found);
    CHECK(c.upstream(5, 1) && c.row_clean(99));
    CHECK_EQ(c.bad_before, uint64_t(1));
    CHECK_EQ(c.bad_after, uint64_t(0));
  }
  // 6. the h bars on few gated rows (ruling 2026-11-05, the second)
  using bolt_cutoff::h_block_ok;
  CHECK(bolt_cutoff::kMinMedianRows == 8);
  CHECK(h_block_ok(true, 1, 0.99979, 0.99979, 0.9998, 0.99));     // one row: the median printed, the min binds
  CHECK(h_block_ok(true, 7, 0.99979, 0.995, 0.9998, 0.99));
  CHECK(!h_block_ok(true, 7, 0.99999, 0.98, 0.9998, 0.99));       // the min still binds on one row
  CHECK(!h_block_ok(true, 8, 0.99979, 0.995, 0.9998, 0.99));      // 8 rows: the median binds
  CHECK(h_block_ok(true, 8, 0.99981, 0.995, 0.9998, 0.99));
  CHECK(h_block_ok(true, 0, 1.0, 1.0, 0.9998, 0.99));             // nothing gated
  CHECK(!h_block_ok(false, 1, 0.99979, 0.99979, 0.9998, 0.99));   // a real checkpoint: the median always binds
  // 7. section 3 on the bf16-state arm (ruling 2026-11-05, the third)
  using bolt_cutoff::section3_binds;
  CHECK(section3_binds(true, 2048, 2048) && section3_binds(true, 4096, 2048));
  CHECK(!section3_binds(true, 512, 2048) && !section3_binds(true, 64, 2048) && !section3_binds(true, 2112, 2048));
  CHECK(!section3_binds(true, 2085, 2048));
  CHECK(section3_binds(false, 64, 2048) && section3_binds(false, 2112, 2048) && section3_binds(false, 512, 2048));
  std::puts("bolt_cutoff_test OK");
  return 0;
}
