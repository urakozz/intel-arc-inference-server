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

}  // namespace

int main() {
  check_take_release();
  check_take_fails_unchanged();
  check_every_order();
  check_random();
  check_release_zeros();
  std::printf("kv_pool_test: ok\n");
  return 0;
}
