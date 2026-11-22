// Spec 31 §5.4 (plan 31 Task 3): runtime::KvPool on the host - the slots' KV page allocator. No device.
//
// Pinned: take and release in every order lose no page and double none; a page sits in one table only (its count
// is 1; Task 8's shared pages raise it on purpose); a take past the free pages fails and changes nothing; a take past
// the per-request cap, a release past the sequence's pages and an empty pool are contract violations (the process
// ends: contract_death.h); 10,000 random takes and releases keep free + sum(owned) == total; a release names the
// rows to zero - every row below the sequence's written bound on the pages it frees, none above (PERF-90's written
// range, per page).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "check.h"
#include "contract_death.h"
#include "runtime/kv_pool.h"

namespace {
using runtime::KvPool;
using runtime::kKvPage;

// Every page is free xor in exactly one table, with the count that says so.
void check_consistent(const KvPool& pool) {
  std::vector<uint32_t> seen(pool.pages_total(), 0);
  size_t owned = 0;
  for (uint32_t s = 0; s < pool.seqs(); ++s) {
    CHECK(pool.pages_of(s).size() <= pool.max_pages());
    for (uint32_t p : pool.pages_of(s)) {
      CHECK(p < pool.pages_total());
      ++seen[p];
      ++owned;
    }
  }
  CHECK_EQ(owned + pool.pages_free(), size_t{pool.pages_total()});
  for (uint32_t p = 0; p < pool.pages_total(); ++p) {
    CHECK(seen[p] <= 1);
    CHECK_EQ(pool.refs(p), seen[p]);
  }
}

// With shared pages: every count equals the tables holding the page, and free + distinct pages held == total.
void check_consistent_shared(const KvPool& pool) {
  std::vector<uint32_t> seen(pool.pages_total(), 0);
  for (uint32_t s = 0; s < pool.seqs(); ++s)
    for (uint32_t p : pool.pages_of(s)) ++seen[p];
  uint32_t held = 0;
  for (uint32_t p = 0; p < pool.pages_total(); ++p) {
    CHECK_EQ(pool.refs(p), seen[p]);
    held += seen[p] != 0;
  }
  CHECK_EQ(held + pool.pages_free(), pool.pages_total());
}

void check_take_release() {
  KvPool pool(10, 3, 8);
  CHECK_EQ(pool.pages_free(), 10u);
  CHECK(pool.take(0, 1));                 // position 0 needs one page
  CHECK_EQ(pool.pages_of(0).size(), size_t{1});
  CHECK_EQ(pool.pages_of(0)[0], 0u);      // a fresh pool hands out pages in order
  CHECK(pool.take(0, kKvPage));           // still one page
  CHECK_EQ(pool.pages_of(0).size(), size_t{1});
  CHECK(pool.take(0, kKvPage + 1));       // the next position starts page 2
  CHECK_EQ((pool.pages_of(0)), (std::vector<uint32_t>{0, 1}));
  CHECK(pool.take(1, 3 * kKvPage));
  CHECK_EQ((pool.pages_of(1)), (std::vector<uint32_t>{2, 3, 4}));
  CHECK(pool.take(0, 100));               // never shrinks
  CHECK_EQ(pool.pages_of(0).size(), size_t{2});
  CHECK(pool.take(2, 0));                 // nothing asked, nothing taken
  CHECK(pool.pages_of(2).empty());
  check_consistent(pool);

  pool.release(0, 0);
  CHECK(pool.pages_of(0).empty());
  CHECK_EQ(pool.pages_free(), 7u);
  check_consistent(pool);
  // Released pages are taken next: sequence 2 gets sequence 0's pages and then fresh ones.
  CHECK(pool.take(2, 4 * kKvPage));
  CHECK_EQ((pool.pages_of(2)), (std::vector<uint32_t>{0, 1, 5, 6}));
  pool.release(1, 0);
  CHECK(pool.release(1, 0).empty());      // an empty table releases nothing
  pool.release(2, 0);
  CHECK_EQ(pool.pages_free(), 10u);
  check_consistent(pool);
}

void check_take_fails_unchanged() {
  KvPool pool(5, 2, 5);
  CHECK(pool.take(0, 3 * kKvPage));
  CHECK(pool.take(1, kKvPage));
  const std::vector<uint32_t> t0 = pool.pages_of(0), t1 = pool.pages_of(1);
  CHECK(!pool.take(1, 3 * kKvPage));      // two more pages, one free
  CHECK_EQ(pool.pages_free(), 1u);
  CHECK(pool.pages_of(0) == t0);
  CHECK(pool.pages_of(1) == t1);
  check_consistent(pool);
  CHECK(pool.take(1, 2 * kKvPage));       // the one free page is still there
  CHECK_EQ(pool.pages_free(), 0u);
  CHECK(!pool.take(0, 3 * kKvPage + 1));
  check_consistent(pool);

  // Past the cap is a caller's error (a contract violation), not a full pool.
  KvPool capped(8, 2, 2);
  CHECK(dies([&] { capped.take(0, 2 * kKvPage + 1); }));
  CHECK(dies([] { (void)KvPool(0, 1, 1); }));   // no pages
  CHECK_EQ(capped.pages_free(), 8u);
  CHECK(capped.pages_of(0).empty());
}

void check_every_order() {
  // Three sequences take 1-3 pages each in all six orders and release in all six: the pool always comes back whole.
  const uint32_t want[3] = {1, 2, 3};
  uint32_t order[3] = {0, 1, 2};
  do {
    uint32_t back[3] = {0, 1, 2};
    do {
      KvPool pool(6, 3, 6);
      for (uint32_t s : order) CHECK(pool.take(s, want[s] * kKvPage));
      CHECK_EQ(pool.pages_free(), 0u);
      check_consistent(pool);
      for (uint32_t s : back) {
        pool.release(s, 0);
        check_consistent(pool);
      }
      CHECK_EQ(pool.pages_free(), 6u);
    } while (std::next_permutation(back, back + 3));
  } while (std::next_permutation(order, order + 3));
}

void check_random() {
  KvPool pool(64, 4, 40);
  std::mt19937 rng(31);
  uint32_t fails = 0;
  for (int i = 0; i < 10000; ++i) {
    const uint32_t s = rng() % 4;
    if (rng() % 16 == 0) {   // rare enough that the four tables outgrow the pool
      const std::vector<uint32_t> held = pool.pages_of(s);
      const uint32_t written = held.empty() ? 0 : rng() % (uint32_t(held.size()) * kKvPage + 1);
      uint64_t rows = 0;
      for (const KvPool::Zero& z : pool.release(s, written)) {
        CHECK(std::find(held.begin(), held.end(), z.page) != held.end());
        rows += z.rows;
      }
      CHECK_EQ(rows, uint64_t{written});
    } else {
      const uint32_t have = static_cast<uint32_t>(pool.pages_of(s).size());
      const uint32_t upto = std::min<uint32_t>(have * kKvPage + rng() % (3 * kKvPage), 40 * kKvPage);
      const uint32_t free_before = pool.pages_free();
      const std::vector<uint32_t> before = pool.pages_of(s);
      if (!pool.take(s, upto)) {
        ++fails;
        CHECK_EQ(pool.pages_free(), free_before);
        CHECK(pool.pages_of(s) == before);
      }
    }
    check_consistent(pool);
  }
  std::printf("random: 10000 takes and releases, %u takes refused for want of pages\n", fails);
  CHECK(fails > 0);   // the run reached a full pool
  for (uint32_t s = 0; s < 4; ++s) pool.release(s, 0);
  CHECK_EQ(pool.pages_free(), 64u);
}

void check_release_zeros() {
  KvPool pool(8, 2, 8);
  CHECK(pool.take(0, 3 * kKvPage));
  CHECK(pool.take(1, 2 * kKvPage));
  const std::vector<uint32_t> t0 = pool.pages_of(0);
  // Written to 2 pages and 952 rows: the first two pages whole, the third up to row 952.
  const std::vector<KvPool::Zero> z = pool.release(0, 2 * kKvPage + 952);
  CHECK_EQ(z.size(), size_t{3});
  for (size_t i = 0; i < z.size(); ++i) CHECK_EQ(z[i].page, t0[i]);
  CHECK_EQ(z[0].rows, kKvPage);
  CHECK_EQ(z[1].rows, kKvPage);
  CHECK_EQ(z[2].rows, 952u);
  // A bound on a page boundary names nothing above it; nothing written, nothing to zero.
  CHECK(pool.take(0, 3 * kKvPage));
  const std::vector<KvPool::Zero> w = pool.release(0, kKvPage);
  CHECK_EQ(w.size(), size_t{1});
  CHECK_EQ(w[0].rows, kKvPage);
  CHECK(pool.release(1, 0).empty());
  CHECK_EQ(pool.pages_free(), 8u);
  // A bound past the sequence's pages is a caller's error (a contract violation).
  CHECK(pool.take(1, kKvPage));
  CHECK(dies([&] { pool.release(1, kKvPage + 1); }));
  CHECK_EQ(pool.pages_of(1).size(), size_t{1});
  CHECK_EQ(pool.pages_free(), 7u);
  check_consistent(pool);
  std::printf("release: the rows below the written bound on the freed pages, none above\n");
}

// Spec 31 §9: shared pages. A share raises the count, a release lowers it; the page goes back (zeroed whole) only with
// its last holder. A write into a shared page gets a private one; with no free page it is refused unchanged.
void check_share() {
  KvPool pool(8, 3, 8);
  CHECK(pool.take(0, 3 * kKvPage));                      // A: pages 0, 1, 2
  const std::vector<uint32_t> a = pool.pages_of(0);
  CHECK_EQ(pool.share(1, 0, 0), a[0]);                   // B shares A's first two pages
  CHECK_EQ(pool.share(1, 1, 0), a[1]);
  CHECK(pool.take(1, 3 * kKvPage));                      // and has a third of its own
  CHECK_EQ(pool.refs(a[0]), 2u);
  CHECK(pool.pages_of(1)[2] != a[2]);
  CHECK_EQ(pool.pages_free(), 4u);
  bool threw = false;
  try {
    pool.share(1, 1, 0);                                 // not B's next entry
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  // B writes into page 1: a private copy; A keeps the old one.
  uint32_t old = 0, fresh = 0;
  CHECK(pool.make_private(1, 1, &old, &fresh));
  CHECK(old == a[1] && fresh != a[1]);
  CHECK_EQ(pool.refs(a[1]), 1u);
  CHECK_EQ(pool.pages_of(1)[1], fresh);
  CHECK(pool.make_private(1, 2, &old, &fresh) && old == fresh);   // its own page: nothing to do
  // A ends: its page 0 stays (B holds it); pages 1 and 2 go back - page 1 whole (it was shared), page 2 to A's bound.
  const std::vector<KvPool::Zero> z = pool.release(0, 3 * kKvPage - 10);
  CHECK_EQ(z.size(), size_t{2});
  CHECK(z[0].page == a[1] && z[0].rows == kKvPage);
  CHECK(z[1].page == a[2] && z[1].rows == kKvPage - 10);
  CHECK_EQ(pool.refs(a[0]), 1u);
  // B ends: page 0 goes back whole even though B wrote none of it.
  const std::vector<KvPool::Zero> zb = pool.release(1, 10);
  bool page0_whole = false;
  for (const KvPool::Zero& x : zb) page0_whole = page0_whole || (x.page == a[0] && x.rows == kKvPage);
  CHECK(page0_whole);
  CHECK_EQ(pool.pages_free(), 8u);
  check_consistent_shared(pool);
  // No free page for a private copy: refused, nothing changed.
  KvPool full(2, 2, 2);
  CHECK(full.take(0, kKvPage));
  full.share(1, 0, 0);
  CHECK(full.take(1, 2 * kKvPage));                      // the last page
  CHECK(!full.make_private(1, 0, &old, &fresh));
  CHECK_EQ(full.refs(full.pages_of(0)[0]), 2u);
  std::printf("share: counts rise and fall, a shared page freed whole with its last holder, writes go private\n");
}

}  // namespace

int main() {
  check_take_release();
  check_take_fails_unchanged();
  check_every_order();
  check_random();
  check_release_zeros();
  check_share();
  std::printf("kv_pool_test: ok\n");
  return 0;
}
