// Spec 7 C4 (plan 7c Task 1): the prefix cache store and the restart plan, host only.
// block = 4 positions, state 64 B, KV 8 B per position, over malloc. Every stored byte is
// a function of the ids it covers (KV of position p: the ids' hash up to p; the state at
// e: the hash up to e), so a plan's runs and state can be checked against the prompt: a
// wrong hit, a snapshot without its blocks, or a freed entry shows as a mismatch.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>

#include "check.h"
#include "server/prefix_cache.h"

namespace {

using server::PrefixCache;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kB = 4;
constexpr size_t kState = 64, kKv = 8;

struct MallocAlloc : server::HostAlloc {
  std::map<void*, size_t> live;
  size_t bytes = 0;
  void* alloc(size_t n) override {
    void* p = std::malloc(n);
    live[p] = n;
    bytes += n;
    return p;
  }
  void free(void* p, size_t n) override {
    CHECK(live.count(p) == 1);
    CHECK_EQ(live[p], n);
    live.erase(p);
    bytes -= n;
    std::free(p);
  }
};

uint64_t prefix_hash(const Ids& ids, uint32_t n) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = 0; i < n; ++i) h = (h ^ ids[i]) * 1099511628211ull;
  return h;
}

// What the engine would copy for a session holding `ids`: the store's call, filled.
void store(PrefixCache& c, const Ids& ids, uint32_t end) {
  const PrefixCache::Slot s = c.reserve(ids, end, end % kB == 0);
  if (s.state) {
    std::memset(s.state, 0, kState);
    const uint64_t h = prefix_hash(ids, end);
    std::memcpy(s.state, &h, 8);
  }
  for (const auto& p : s.kv) {
    for (uint32_t q = p.begin; q < p.end; ++q) {
      const uint64_t h = prefix_hash(ids, q + 1);
      std::memcpy(static_cast<uint8_t*>(p.host) + size_t(q - p.begin) * kKv, &h, kKv);
    }
  }
  c.commit(s);
}

// A prefill of ids[from:] with the block hook: a call at every block end, then the end.
void prefill(PrefixCache& c, const Ids& ids, uint32_t from) {
  const uint32_t len = uint32_t(ids.size());
  for (uint32_t e = (from / kB + 1) * kB; e < len; e += kB) store(c, ids, e);
  store(c, ids, len);
  c.release();
}

// The plan's state and runs hold exactly the prompt's data, and the runs cover
// [kv_from, restart) contiguously.
void check_plan(const PrefixCache::Plan& p, const Ids& prompt) {
  CHECK(p.restart + 1 <= prompt.size());   // at least one id is always fed
  if (p.kind != PrefixCache::Plan::Restore) {
    CHECK(p.kv.empty());
    return;
  }
  uint64_t h = 0;
  std::memcpy(&h, p.state, 8);
  CHECK_EQ(h, prefix_hash(prompt, p.restart));
  uint32_t at = p.kv_from;
  for (const auto& r : p.kv) {
    CHECK_EQ(r.begin, at);
    for (uint32_t q = r.begin; q < r.end; ++q) {
      std::memcpy(&h, static_cast<const uint8_t*>(r.host) + size_t(q - r.begin) * kKv, 8);
      CHECK_EQ(h, prefix_hash(prompt, q + 1));
    }
    at = r.end;
  }
  CHECK_EQ(at, p.restart);
}

Ids seq(uint32_t n, uint32_t base) {
  Ids v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = base + i;
  return v;
}

Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

Ids head(const Ids& a, size_t n) { return Ids(a.begin(), a.begin() + n); }

void test_plans() {
  MallocAlloc m;
  PrefixCache c(1 << 20, kState, kKv, kB, m);
  const Ids a = seq(10, 100);
  const Ids gen = seq(4, 500);

  // Nothing stored: cold; a valid resident prefix: continue.
  auto p = c.plan(a, {}, false);
  CHECK(p.kind == PrefixCache::Plan::Cold && p.restart == 0);
  c.release();
  p = c.plan(a, head(a, 6), true);
  CHECK(p.kind == PrefixCache::Plan::Continue && p.restart == 6 && p.kv.empty());
  c.release();
  p = c.plan(a, head(a, 6), false);   // after a failure: never continue
  CHECK(p.kind == PrefixCache::Plan::Cold);
  c.release();

  // Request A (10 ids): blocks [0,4) [4,8), snapshots at 4, 8 and the prompt end 10.
  prefill(c, a, 0);
  CHECK_EQ(c.bytes_used(), 2 * 4 * kKv + 3 * kState + 2 * kKv);

  // Restore at the prompt end, resident unknown: every KV run from 0.
  const Ids b = cat(a, seq(5, 900));
  p = c.plan(b, {}, false);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 10 && p.kv_from == 0);
  CHECK_EQ(p.kv.size(), size_t(3));
  check_plan(p, b);
  c.release();

  // C4: a prompt equal to a snapshot's position restores below it (here the block end
  // at 8) and prefills; the server's own prompt-end snapshot sits at len - 1 for this.
  p = c.plan(a, cat(a, gen), true);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 8);
  CHECK(p.kv.empty() && p.kv_from == 8);   // the card holds A's KV already
  check_plan(p, a);
  c.release();

  // Divergence mid-block (at 9): the block-end snapshot at 8.
  Ids d = a;
  d[9] = 4242;
  d = cat(d, seq(3, 700));
  p = c.plan(d, {}, false);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 8 && p.kv_from == 0);
  check_plan(p, d);
  c.release();

  // kv_from follows the common prefix with the resident, rounded down to the entry start:
  // resident = a[0:6] + other; the card's KV [0,6) is right, the block [4,8) is loaded.
  p = c.plan(d, cat(head(a, 6), seq(8, 3000)), true);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 8 && p.kv_from == 4);
  CHECK_EQ(p.kv.size(), size_t(1));
  check_plan(p, d);
  c.release();
  p = c.plan(d, a, true);   // resident agrees past the restart point: the state alone
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 8 && p.kv.empty());
  c.release();

  // Review Focus 1: a request shorter than the resident, a prefix of it (a retry of an
  // edited last message): never continue past the prompt; a snapshot <= len - 1.
  const Ids resident = cat(a, gen);
  for (uint32_t n : {7u, 9u, 11u, 13u}) {
    const Ids q = head(resident, n);
    p = c.plan(q, resident, true);
    CHECK(p.kind != PrefixCache::Plan::Continue);
    CHECK(p.restart <= n - 1);
    check_plan(p, q);
    c.release();
  }
  p = c.plan(head(a, 7), resident, true);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 4);
  c.release();

  // A deeper snapshot beats a shallower continuation: resident a[0:5], snapshot at 10.
  p = c.plan(b, head(a, 5), true);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 10 && p.kv_from == 4);
  check_plan(p, b);
  c.release();

  // The request-end snapshot (generated ids included) serves the next turn.
  store(c, resident, 14);
  c.release();
  // The request-end snapshot at 14 is the prompt's own length: restore below it.
  p = c.plan(resident, {}, false);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 10);
  check_plan(p, resident);
  c.release();
  const Ids turn2 = cat(resident, seq(6, 1200));
  p = c.plan(turn2, {}, false);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 14);
  check_plan(p, turn2);
  c.release();
  store(c, a, 9);                          // what the server stores: the snapshot at len - 1
  c.release();
  p = c.plan(a, cat(a, gen), true);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 9);   // Review Focus 5
  c.release();
  CHECK_EQ(m.bytes, c.bytes_used());
  std::printf("plans: cold, continue, restore at a prompt end / block end / request end,"
              " kv_from, Review Focus 1 OK\n");
}

void test_budget() {
  // Review Focus 2: a budget smaller than one session.
  MallocAlloc m;
  const size_t block_bytes = kB * kKv, snap_bytes = kState;
  PrefixCache c(3 * block_bytes + 2 * snap_bytes, kState, kKv, kB, m);
  const Ids s1 = seq(24, 10);
  prefill(c, s1, 0);
  CHECK(c.bytes_used() <= c.budget());
  CHECK_EQ(m.bytes, c.bytes_used());
  auto p = c.plan(cat(s1, {1}), {}, false);
  CHECK(p.kind == PrefixCache::Plan::Restore);   // what did fit is usable
  check_plan(p, cat(s1, {1}));
  c.release();

  // A second session evicts the first one's leaves once nothing pins them; the plan's own
  // entries stay while the request runs.
  const Ids s2 = cat(head(s1, 4), seq(20, 5000));
  p = c.plan(s2, s1, true);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 4);
  const void* used_state = p.state;
  for (uint32_t e = 8; e < s2.size(); e += kB) {
    store(c, s2, e);
    CHECK(c.bytes_used() <= c.budget());
    CHECK(m.live.count(const_cast<void*>(used_state)) == 1);   // never the pinned entry
  }
  store(c, s2, uint32_t(s2.size()));
  c.release();
  CHECK(c.bytes_used() <= c.budget());
  p = c.plan(cat(s2, {2}), {}, false);
  check_plan(p, cat(s2, {2}));
  CHECK(p.restart >= 8);
  c.release();

  // A store that cannot fit anything skips it; the request goes on.
  PrefixCache tiny(snap_bytes - 1, kState, kKv, kB, m);
  const PrefixCache::Slot s = tiny.reserve(s1, 3, false);
  CHECK(s.empty());
  tiny.release();
  CHECK_EQ(tiny.bytes_used(), size_t(0));

  // release() unpins: with everything pinned nothing can be evicted, after it the LRU
  // leaves go.
  PrefixCache one(block_bytes + snap_bytes, kState, kKv, kB, m);
  prefill(one, head(s1, 5), 0);                      // block 0 + snapshot 4; 5 does not fit
  CHECK_EQ(one.bytes_used(), one.budget());
  p = one.plan(head(s1, 5), {}, false);
  CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 4);
  const PrefixCache::Slot blocked = one.reserve(seq(8, 7000), 4, true);
  CHECK(blocked.empty());                            // everything is pinned
  one.release();
  const PrefixCache::Slot freed = one.reserve(seq(8, 7000), 4, true);
  CHECK(!freed.empty());
  one.commit(freed);
  one.release();
  CHECK(one.bytes_used() <= one.budget());
  std::printf("budget: bytes_used <= budget, pinned entries kept, skip when nothing fits,"
              " release() unpins (Review Focus 2) OK\n");
}

void test_collision() {
  MallocAlloc m;
  PrefixCache c(1 << 20, kState, kKv, kB, m,
                [](uint64_t, const uint32_t*, size_t) { return uint64_t(42); });
  const Ids a = seq(10, 100);
  Ids a2 = a;
  a2[1] = 9999;   // same hash (forced), different ids
  prefill(c, a, 0);
  auto p = c.plan(cat(a2, {5}), {}, false);
  CHECK(p.kind == PrefixCache::Plan::Cold);
  c.release();
  prefill(c, a2, 0);
  for (const Ids& q : {cat(a, {5}), cat(a2, {5})}) {
    p = c.plan(q, {}, false);
    CHECK(p.kind == PrefixCache::Plan::Restore && p.restart == 10);
    check_plan(p, q);
    c.release();
  }
  std::printf("collision: a forced hash collision never matches different ids OK\n");
}

void test_stress() {
  std::mt19937 rng(7);
  for (int round = 0; round < 40; ++round) {
    MallocAlloc m;
    const size_t budget = 64 + rng() % 1500;
    PrefixCache c(budget, kState, kKv, kB, m);
    Ids resident;
    bool valid = false;
    for (int r = 0; r < 60; ++r) {
      // Prompts over a small alphabet so that prefixes are shared often.
      Ids q;
      if (valid && !resident.empty() && rng() % 2) q = head(resident, rng() % resident.size() + 1);
      const uint32_t extra = 1 + rng() % 14;
      for (uint32_t i = 0; i < extra; ++i) q.push_back(rng() % 3);
      auto p = c.plan(q, resident, valid);
      check_plan(p, q);
      if (p.kind == PrefixCache::Plan::Continue) CHECK(p.restart == resident.size());
      if (p.restart < q.size()) {
        for (uint32_t e = (p.restart / kB + 1) * kB; e < q.size(); e += kB) {
          store(c, q, e);
          CHECK(c.bytes_used() <= budget);
          check_plan(p, q);   // the pinned entries survive every store
        }
        store(c, q, uint32_t(q.size()));
      }
      c.release();
      CHECK(c.bytes_used() <= budget);
      CHECK_EQ(m.bytes, c.bytes_used());
      resident = q;
      for (uint32_t i = 0, n = rng() % 5; i < n; ++i) resident.push_back(rng() % 3);
      valid = rng() % 8 != 0;
      store(c, resident, uint32_t(resident.size()));
      c.release();
      CHECK(c.bytes_used() <= budget);
    }
  }
  std::printf("stress: 40 budgets x 60 requests, every plan's data equals the prompt's,"
              " bytes_used <= budget OK\n");
}

}  // namespace

int main() {
  test_plans();
  test_budget();
  test_collision();
  test_stress();
  std::printf("prefix_cache_test OK\n");
  return 0;
}
