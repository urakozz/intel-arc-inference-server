// Spec 19d Task 1 (spec 19 decision 5), host only: the drafter's ring across a prefix-cache restore, through the
// request path (server::PrefixSession) over a mock engine whose drafter ring fills exactly as runtime::Engine's
// does (tests/server/mock.h DrafterMockEngine: a prefill call projects the window of its end, every decode step its
// own position) and rides in save_state / load_state as the spec 19d engine's does.
//
//   K   the store's key carries the drafter: server::store_root(kv_form, 0) is kv_form (every key without a drafter
//       is what it was), with a drafter it differs from kv_form and between two drafters and two KV forms; a
//       PrefixSession's chain root is store_root(engine.kv_form(), engine.drafter_key()).
//   R1  restore-and-continue (19c's deferred minor 8): turn 1 (prompt A, 6 ids generated, the request-end snapshot),
//       a side request that overwrites the ring, then turn 2 = turn 1 + its ids + B - restored from turn 1's
//       request-end snapshot - leaves the ring EQUAL to that of the uninterrupted session (turn 1 then turn 2 by
//       Continue), after the tail and after 6 more ids. The mutant without the ring in the snapshot (19c's engine)
//       is caught: its ring after the same restore differs.
//   R2  a restore at a block end followed by a tail at least the window long: the ring equals a cold prefill's
//       (the tail's own window covers every slot). The mid-call caveat, recorded: restored at a block end the call's
//       last window did not cover, a SHORT tail leaves ring slots the cold run filled - counted, not gated (spec 19
//       §12: lower acceptance there, never a different output).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#include "check.h"
#include "mock.h"
#include "server/prefix_cache.h"

namespace {

using Ids = std::vector<uint32_t>;

struct MallocAlloc : server::HostAlloc {
  std::map<void*, size_t> live;
  void* alloc(size_t n) override {
    void* p = std::malloc(n);
    live[p] = n;
    return p;
  }
  void free(void* p, size_t n) override {
    CHECK(live.count(p) == 1);
    CHECK_EQ(live[p], n);
    live.erase(p);
    std::free(p);
  }
};

Ids seq(uint32_t n, uint32_t from) {
  Ids v;
  for (uint32_t i = 0; i < n; ++i) v.push_back(from + (i * 7) % 23);
  return v;
}
Ids cat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

DrafterMockEngine make_engine(bool ring_in_state) {
  DrafterMockEngine e;
  for (uint32_t i = 0; i < 16; ++i) e.words.push_back(500 + i);
  e.ring_in_state = ring_in_state;
  return e;
}

// One request through the request path: begin, `n` greedy steps (each fed), end. Returns the generated ids.
Ids turn(server::PrefixSession& s, DrafterMockEngine& e, const Ids& prompt, uint32_t n,
         server::PrefixCache::Plan::Kind* kind = nullptr) {
  const server::PrefixSession::Report r = s.begin(prompt);
  if (kind) *kind = r.kind;
  Ids out;
  server::Sampling greedy;
  for (uint32_t i = 0; i < n; ++i) {
    out.push_back(e.step(greedy));
    s.fed(out.back());
  }
  s.end();
  return out;
}

void test_key() {
  for (uint64_t form : {uint64_t(0), uint64_t(1), uint64_t(0x5EED)}) CHECK_EQ(server::store_root(form, 0), form);
  const uint64_t a = server::store_root(0, 11), b = server::store_root(0, 12), c = server::store_root(1, 11);
  CHECK(a != 0 && a != b && a != c && b != c);
  MallocAlloc m;
  DrafterMockEngine e = make_engine(true);
  {
    server::PrefixSession s(e, 1 << 20, &m);
    CHECK_EQ(s.cache()->kv_form(), server::store_root(0, e.key));
    CHECK(s.cache()->kv_form() != 0);
  }
  StateMockEngine plain;
  for (uint32_t i = 0; i < 16; ++i) plain.words.push_back(500 + i);
  {
    server::PrefixSession s(plain, 1 << 20, &m);
    CHECK_EQ(s.cache()->kv_form(), uint64_t(0));   // no drafter: the chain's root is the KV form, as before
  }
  std::printf("K: store_root(form, 0) = form; a drafter's key changes the root; PrefixSession roots its chains "
              "there OK\n");
}

// The ring of the uninterrupted session and of the restored one, after turn 2 and after 6 more ids.
struct Rings {
  std::vector<uint64_t> after_tail, after_gen;
  Ids gen;
};

Rings restored(bool ring_in_state, const Ids& a, const Ids& side, const Ids& b) {
  MallocAlloc m;
  DrafterMockEngine e = make_engine(ring_in_state);
  server::PrefixSession s(e, 1 << 20, &m);
  const Ids g1 = turn(s, e, a, 6);
  (void)turn(s, e, side, 3);   // another session: cold, its own ring
  server::PrefixCache::Plan::Kind kind{};
  Rings r;
  const server::PrefixSession::Report rep = s.begin(cat(cat(a, g1), b));
  kind = rep.kind;
  CHECK(kind == server::PrefixCache::Plan::Restore);
  CHECK_EQ(rep.restart, uint32_t(a.size() + g1.size()));   // turn 1's request-end snapshot
  r.after_tail = e.ring;
  server::Sampling greedy;
  for (int i = 0; i < 6; ++i) {
    r.gen.push_back(e.step(greedy));
    s.fed(r.gen.back());
  }
  s.end();
  r.after_gen = e.ring;
  return r;
}

Rings uninterrupted(const Ids& a, const Ids& b) {
  MallocAlloc m;
  DrafterMockEngine e = make_engine(true);
  server::PrefixSession s(e, 1 << 20, &m);
  const Ids g1 = turn(s, e, a, 6);
  Rings r;
  const server::PrefixSession::Report rep = s.begin(cat(cat(a, g1), b));
  CHECK(rep.kind == server::PrefixCache::Plan::Continue);
  r.after_tail = e.ring;
  server::Sampling greedy;
  for (int i = 0; i < 6; ++i) {
    r.gen.push_back(e.step(greedy));
    s.fed(r.gen.back());
  }
  s.end();
  r.after_gen = e.ring;
  return r;
}

void test_restore_and_continue() {
  // A: 11 ids (block 4, window 4: block-end snapshots at 4 and 8, the prompt end at 11); B: a 3-id tail - shorter
  // than the window, so the restored ring's older slots are the snapshot's own.
  const Ids a = seq(11, 100), side = seq(9, 300), b = seq(3, 200);
  const Rings ref = uninterrupted(a, b);
  const Rings got = restored(true, a, side, b);
  CHECK(got.gen == ref.gen);
  CHECK(got.after_tail == ref.after_tail);
  CHECK(got.after_gen == ref.after_gen);
  const Rings mutant = restored(false, a, side, b);
  CHECK(mutant.gen == ref.gen);                 // the output never depends on the ring
  CHECK(mutant.after_tail != ref.after_tail);   // ... but 19c's restore leaves another session's ring
  std::printf("R1: restored from the request-end snapshot, the ring equals the uninterrupted session's after the "
              "tail and after 6 ids; the ring-less snapshot (19c) is caught OK\n");
}

void test_block_end_restore() {
  // Turn 1: 14 ids in one call (block-end snapshots at 4, 8, 12; the call's window projects 10..13 only).
  const Ids a = seq(14, 100), c = seq(9, 200);
  MallocAlloc m;
  // R2: a restore at 8 followed by a tail >= the window: every ring slot is the tail's own projection.
  const Ids long_prompt = cat(Ids(a.begin(), a.begin() + 8), c);
  std::vector<uint64_t> cold_ring;
  {
    DrafterMockEngine e = make_engine(true);
    e.reset();
    e.prefill(long_prompt);
    cold_ring = e.ring;
  }
  DrafterMockEngine e = make_engine(true);
  server::PrefixSession s(e, 1 << 20, &m);
  (void)turn(s, e, a, 2);
  (void)turn(s, e, seq(7, 400), 1);   // another session
  const server::PrefixSession::Report rep = s.begin(long_prompt);
  CHECK(rep.kind == server::PrefixCache::Plan::Restore);
  CHECK_EQ(rep.restart, 8u);
  CHECK(e.ring == cold_ring);
  s.end();
  // The caveat: the same block end, a 1-id tail - slots of positions 5..7 were never projected in turn 1's call
  // (outside its last window), so they hold what the ring held then (zeros: turn 1 started from reset()).
  const Ids short_prompt = cat(Ids(a.begin(), a.begin() + 8), Ids{777});
  {
    DrafterMockEngine cold = make_engine(true);
    cold.reset();
    cold.prefill(short_prompt);
    (void)turn(s, e, seq(7, 400), 1);
    const server::PrefixSession::Report r2 = s.begin(short_prompt);
    CHECK(r2.kind == server::PrefixCache::Plan::Restore && r2.restart == 8u);
    uint32_t differ = 0;
    for (uint32_t i = 0; i < e.window; ++i) differ += e.ring[i] != cold.ring[i];
    s.end();
    std::printf("R2: a restore at a block end with a tail >= the window equals a cold prefill's ring OK; with a "
                "1-id tail %u of %u slots differ from the cold run's (the mid-call caveat, not gated)\n",
                differ, e.window);
  }
}

}  // namespace

int main() {
  test_key();
  test_restore_and_continue();
  test_block_end_restore();
  std::printf("dflash_prefix_test OK\n");
  return 0;
}
