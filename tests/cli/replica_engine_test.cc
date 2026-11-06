// Plan 13b Task 3: cli::ReplicaEngine (two single-sequence engines, one per card, behind server::BatchEngineIface)
// over two host mock engines (tests/server/mock.h StateMockEngine), driven by the real scheduler and the prefix
// cache per slot. Host only. Checks: every request's ids equal the same request alone on a fresh engine (the
// serial server's call sequence); each card's calls run on that card's own thread, the two cards' steps at once; a
// failure on one card ends that card's request only; a snapshot saved on card 0 restores on card 1 (one shared host
// store); the prompt cut where one runtime::Engine::prefill call cuts it; --prefix-split-last's replay; a lookup
// row's proposer reaches its card; cards that differ are refused.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "cli/replica_engine.h"
#include "server/mock.h"
#include "server/scheduler.h"

namespace {

using server::Finish;
using server::SchedRequest;
using server::Scheduler;
using server::SchedulerOptions;

struct MallocAlloc : server::HostAlloc {
  void* alloc(size_t n) override { return std::malloc(n); }
  void free(void* p, size_t) override { std::free(p); }
};

std::atomic<int> g_in_step{0}, g_max_in_step{0};

// StateMockEngine that records the thread of every call, the size of every prefill call, and can sleep in a
// step (to see the two cards step at once) or fail at its n-th step.
struct CardMock : StateMockEngine {
  std::mutex mu;
  std::set<std::thread::id> threads;
  std::vector<size_t> prefill_calls;
  int step_ms = 0;
  uint32_t fail_step = 0, steps = 0;
  size_t proposals = 0;
  void seen() {
    std::lock_guard<std::mutex> lock(mu);
    threads.insert(std::this_thread::get_id());
  }
  void reset() override {
    seen();
    StateMockEngine::reset();
  }
  void prefill(const std::vector<uint32_t>& ids) override {
    seen();
    prefill_calls.push_back(ids.size());
    StateMockEngine::prefill(ids);
  }
  uint32_t step(const server::Sampling& s) override {
    seen();
    if (fail_step != 0 && ++steps == fail_step) throw std::runtime_error("mock card: device lost");
    const int now = ++g_in_step;
    int m = g_max_in_step.load();
    while (now > m && !g_max_in_step.compare_exchange_weak(m, now)) {
    }
    if (step_ms != 0) std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
    --g_in_step;
    return StateMockEngine::step(s);
  }
  uint32_t pos() override {
    seen();
    return StateMockEngine::pos();
  }
  void save_state(void* h) override {
    seen();
    StateMockEngine::save_state(h);
  }
  void load_state(const void* h, uint32_t at) override {
    seen();
    StateMockEngine::load_state(h, at);
  }
  void ingest(const std::vector<uint32_t>& ids) override {
    seen();
    StateMockEngine::ingest(ids);
  }
  // A lookup "verify": takes the proposal, keeps nothing (one plain step).
  uint32_t lookup_k = 0;
  uint32_t verify_k() override { return lookup_k; }
  std::vector<uint32_t> step_drafts(const server::Sampling& s, const Proposer& propose) override {
    (void)propose(cur);
    ++proposals;
    return {step(s)};
  }
};

std::vector<uint32_t> words() {
  std::vector<uint32_t> w;
  for (uint32_t i = 0; i < 24; ++i) w.push_back(1000 + i);
  return w;
}

void setup(CardMock& c, uint32_t maxlen = 1024) {
  c.words = words();
  c.maxlen = maxlen;
  c.kv.assign(maxlen, 0);
}

std::vector<uint32_t> P(uint32_t key, uint32_t n) {
  std::vector<uint32_t> p{key};
  for (uint32_t i = 1; i < n; ++i) p.push_back(5000 + key * 131 + i);
  return p;
}

// The request alone on a fresh card, as the serial server drives it (reset, one prefill, steps).
std::vector<uint32_t> alone(const std::vector<uint32_t>& prompt, uint32_t n, const server::Sampling& s = {}) {
  CardMock c;
  setup(c);
  c.reset();
  c.prefill(prompt);
  std::vector<uint32_t> out;
  for (uint32_t i = 0; i < n; ++i) out.push_back(c.step(s));
  return out;
}

struct Out {
  std::vector<uint32_t> ids;
  Finish fin;
  std::atomic<bool> done{false};
};

SchedRequest req(const std::vector<uint32_t>& prompt, uint32_t max_tokens, Out& out, const server::Sampling& s = {}) {
  SchedRequest r;
  r.prompt = prompt;
  r.max_tokens = max_tokens;
  r.sampling = s;
  r.on_token = [&out](uint32_t id) {
    out.ids.push_back(id);
    return true;
  };
  r.on_done = [&out](const Finish& f) {
    out.fin = f;
    out.done = true;
  };
  return r;
}

void drain(Scheduler& s) {
  long n = 0;
  while ((s.waiting() != 0 || s.active() != 0) && n++ < 1000000) s.tick();
  CHECK(s.waiting() == 0 && s.active() == 0);
}

// Fan-out and join, bitwise per card, threads.
void test_two_at_once() {
  CardMock c0, c1;
  setup(c0);
  setup(c1);
  c0.step_ms = c1.step_ms = 2;
  cli::ReplicaEngine eng({{&c0, 8}, {&c1, 8}}, /*hooked=*/true, /*wide_stride=*/true);
  MallocAlloc alloc;
  server::PrefixSlots cache(eng, 1 << 20, alloc);
  Scheduler s(eng, SchedulerOptions{}, &cache);
  server::Sampling seeded;
  seeded.greedy = false;
  seeded.seed = 11;
  seeded.has_seed = true;
  Out a, b;
  g_max_in_step = 0;
  CHECK(s.submit(req(P(1, 21), 30, a)) != nullptr);
  CHECK(s.submit(req(P(2, 13), 30, b, seeded)) != nullptr);
  drain(s);
  CHECK_EQ(a.fin.slot, 0);
  CHECK_EQ(b.fin.slot, 1);
  CHECK(a.ids == alone(P(1, 21), 30));
  CHECK(b.ids == alone(P(2, 13), 30, seeded));
  CHECK(b.ids != alone(P(2, 13), 30));
  // Each card's engine saw one thread, its own; never the scheduler's.
  CHECK_EQ(c0.threads.size(), size_t(1));
  CHECK_EQ(c1.threads.size(), size_t(1));
  CHECK(*c0.threads.begin() == eng.thread_of(0));
  CHECK(*c1.threads.begin() == eng.thread_of(1));
  CHECK(eng.thread_of(0) != eng.thread_of(1));
  CHECK(c0.threads.count(std::this_thread::get_id()) == 0);
  CHECK_EQ(g_max_in_step.load(), 2);   // the two cards stepped at once
  std::printf("two at once: each request's ids equal it alone on a fresh card; one thread per card; the cards "
              "step concurrently\n");
}

// Review Focus 2: card 1 fails mid-request; card 0's request finishes as alone.
void test_card_failure() {
  CardMock c0, c1;
  setup(c0);
  setup(c1);
  c1.fail_step = 4;
  cli::ReplicaEngine eng({{&c0, 8}, {&c1, 8}}, false, true);
  Scheduler s(eng, SchedulerOptions{});
  Out a, b;
  CHECK(s.submit(req(P(3, 9), 20, a)) != nullptr);
  CHECK(s.submit(req(P(4, 9), 20, b)) != nullptr);
  drain(s);
  CHECK_EQ(b.fin.reason, Finish::Error);
  CHECK(b.fin.error.find("device lost") != std::string::npos);
  CHECK_EQ(b.ids.size(), size_t(3));
  CHECK_EQ(a.fin.reason, Finish::Length);
  CHECK(a.ids == alone(P(3, 9), 20));
  // Card 1 serves again after a reset.
  c1.fail_step = 0;
  Out c;
  CHECK(s.submit(req(P(5, 9), 5, c)) != nullptr);
  drain(s);
  CHECK_EQ(c.fin.reason, Finish::Length);
  std::printf("card failure: card 1's error ended its request only (after 3 ids); card 0's ran to the end\n");
}

// B3 on the host: A on card 0; X (sharing A's start) then holds card 0; A's continuation goes to card 1 and is
// restored there from the snapshot card 0 saved in the one host store.
void test_restore_across_cards() {
  CardMock c0, c1;
  setup(c0);
  setup(c1);
  cli::ReplicaEngine eng({{&c0, 8}, {&c1, 8}}, true, true);
  MallocAlloc alloc;
  server::PrefixSlots cache(eng, 1 << 20, alloc);
  Scheduler s(eng, SchedulerOptions{}, &cache);
  const std::vector<uint32_t> A = P(6, 19);
  Out a;
  CHECK(s.submit(req(A, 9, a)) != nullptr);
  drain(s);
  CHECK_EQ(a.fin.slot, 0);
  std::vector<uint32_t> xp(A.begin(), A.begin() + 6);
  xp.push_back(7777);
  Out x;
  CHECK(s.submit(req(xp, 500, x)) != nullptr);
  s.tick();
  std::vector<uint32_t> a2 = A;
  a2.insert(a2.end(), a.ids.begin(), a.ids.end());
  for (uint32_t t : {9001u, 9002u, 9003u}) a2.push_back(t);
  Out b;
  CHECK(s.submit(req(a2, 12, b)) != nullptr);
  drain(s);
  CHECK_EQ(x.fin.slot, 0);
  CHECK_EQ(b.fin.slot, 1);
  CHECK_EQ(cache.last_kind(1), server::PrefixCache::Plan::Restore);
  CHECK_EQ(b.fin.cached_tokens, uint32_t(A.size() + a.ids.size()));
  CHECK(c1.state_loads > 0);
  CHECK(b.ids == alone(a2, 12));
  std::printf("restore across cards: A's snapshot from card 0 restored on card 1 at %u, ids as a cold run\n",
              b.fin.cached_tokens);
}

// The prompt is cut where one Engine::prefill call cuts it: hooked (the cache on) at every stride end from the
// absolute position; without the hook every chunk from the call's start.
void test_prefill_cut() {
  {
    CardMock c0, c1;
    setup(c0);
    setup(c1);
    cli::ReplicaEngine eng({{&c0, 8}, {&c1, 8}}, true, true);
    MallocAlloc alloc;
    server::PrefixSlots cache(eng, 1 << 20, alloc);
    Scheduler s(eng, SchedulerOptions{}, &cache);
    Out a;
    CHECK(s.submit(req(P(7, 30), 2, a)) != nullptr);
    drain(s);
    CHECK(c0.prefill_calls == std::vector<size_t>({8, 8, 8, 6}));
    // A continuation from pos 32 (30 + 2): the resident session continues; the tail is cut at the stride end 40.
    std::vector<uint32_t> p2 = P(7, 30);
    p2.insert(p2.end(), a.ids.begin(), a.ids.end());
    for (uint32_t i = 0; i < 12; ++i) p2.push_back(8000 + i);
    Out b;
    c0.prefill_calls.clear();
    CHECK(s.submit(req(p2, 2, b)) != nullptr);
    drain(s);
    CHECK_EQ(cache.last_kind(0), server::PrefixCache::Plan::Continue);
    CHECK(c0.prefill_calls == std::vector<size_t>({8, 4}));
  }
  {
    CardMock c0, c1;
    setup(c0);
    setup(c1);
    cli::ReplicaEngine eng({{&c0, 16}, {&c1, 16}}, false, true);
    CHECK_EQ(eng.prefill_cut(0, 0, 40), 16U);
    CHECK_EQ(eng.prefill_cut(0, 16, 40 - 16), 16U);
    CHECK_EQ(eng.prefill_cut(0, 32, 8), 8U);
    // Hooked, wide stride at cap 4096 over 2048-blocks (plan 5d): every 4096 from the absolute position.
    cli::ReplicaEngine wide({{&c0, 4096}, {&c1, 4096}}, true, true);
    CHECK_EQ(wide.prefill_cut(0, 5000, 100000), 8192U - 5000U);
  }
  std::printf("prefill cut: 8/8/8/6 hooked, a continuation from 32 cut at 40; unhooked from the call's start\n");
}

// --prefix-split-last through the replica engine: the last prompt id goes through the card's ingest.
void test_split_last_and_lookup() {
  CardMock c0, c1;
  setup(c0);
  setup(c1);
  cli::ReplicaEngine eng({{&c0, 8}, {&c1, 8}}, true, true);
  MallocAlloc alloc;
  server::PrefixSlots cache(eng, 1 << 20, alloc);
  Scheduler s(eng, SchedulerOptions{}, &cache);
  Out a;
  SchedRequest r = req(P(8, 12), 6, a);
  r.split_last = true;
  CHECK(s.submit(std::move(r)) != nullptr);
  drain(s);
  CHECK_EQ(c0.ingested, size_t(1));
  CHECK(c0.prefill_calls == std::vector<size_t>({8, 3}));
  CHECK(a.ids == alone(P(8, 12), 6));
  // A lookup row's proposer reaches its card's step_drafts.
  CardMock l0, l1;
  setup(l0);
  setup(l1);
  l0.lookup_k = l1.lookup_k = 3;
  cli::ReplicaEngine leng({{&l0, 8}, {&l1, 8}}, false, true);
  SchedulerOptions o;
  o.lookup = true;
  o.lookup_adaptive.max_k = 3;
  o.lookup_adaptive.cost = server::MtpCost::int8_head().with_free_drafts();
  Scheduler ls(leng, o);
  Out la, lb;
  CHECK(ls.submit(req(P(9, 10), 7, la)) != nullptr);
  CHECK(ls.submit(req(P(10, 10), 7, lb)) != nullptr);
  drain(ls);
  CHECK(l0.proposals > 0 && l1.proposals > 0);
  CHECK(la.ids == alone(P(9, 10), 7));
  std::printf("split-last: prefill 8 + 3, the last id ingested; lookup rows reach each card's step_drafts\n");
}

void test_cards_must_agree() {
  CardMock c0, c1;
  setup(c0);
  setup(c1);
  c1.blk = 8;
  bool threw = false;
  try {
    cli::ReplicaEngine eng({{&c0, 8}, {&c1, 8}}, true, true);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  // Per-card lengths: the scheduler sees each card's own max_len.
  CardMock d0, d1;
  setup(d0, 1024);
  setup(d1, 512);
  cli::ReplicaEngine eng({{&d0, 8}, {&d1, 8}}, false, true);
  CHECK_EQ(eng.max_len(), 1024U);
  CHECK_EQ(eng.slot_max_len(1), 512U);
  std::printf("cards that differ in block are refused; per-card max_len reported\n");
}

}  // namespace

int main() {
  test_two_at_once();
  test_card_failure();
  test_restore_across_cards();
  test_prefill_cut();
  test_split_last_and_lookup();
  test_cards_must_agree();
  std::printf("replica_engine_test: all cases passed\n");
  return 0;
}
