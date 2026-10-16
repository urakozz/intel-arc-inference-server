// Spec 19d Task 2 (spec 19 §5 "Batching", plan 19d Review Focus 3), host only: speculative decoding under the
// batching scheduler, against MockBatchEngine's speculative mode (tests/server/mock.h: one drafter ring per slot,
// a row's acceptance a function of its own slot's ring).
//
//   S1  lossless and isolated: four greedy requests together through a speculative engine (K by each request's own
//       AdaptiveK over MtpCost::dflash) emit exactly the ids the plain batched scheduler emits, and each request's
//       speculative record (iterations, drafts asked, drafts kept) equals the same request served ALONE - so no row
//       read another slot's ring and no request's K followed another's acceptance; the steps are batched (a step
//       carries several rows) and speculation happened (fewer iterations than ids).
//   S2  a request that ends inside a run - EOS (not emitted), max_tokens, a stop by on_token - keeps exactly the ids
//       consumed: the ids and the slot's final pos equal the plain scheduler's, the run's tail was handed back with
//       truncate_to, and the request-end snapshot sits at that pos.
//   S3  sampled rows draw from their own generator: a seeded request among greedy ones emits what it emits on the
//       plain scheduler.
//   S4  the ring follows a prefix-cache restore onto ANOTHER slot: request A on slot 0, then (slot 0 busy) its
//       continuation A' restored on slot 1 from A's request-end snapshot - its ids and speculative record equal the
//       uninterrupted session's (A then A' by Continue on one slot). The 19c mutant (the ring not in the snapshot)
//       is caught: A''s record differs.
//   S5  a fixed K (SchedulerOptions::spec_fixed_k) asks every row for that K.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "server/adaptive_k.h"
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

struct Out {
  std::vector<uint32_t> ids;
  Finish fin;
  int done = 0;
};
struct Req {
  std::vector<uint32_t> prompt;
  uint32_t max_tokens = 40;
  server::Sampling s;
  uint32_t stop_after = 0;
};

Req R(std::vector<uint32_t> prompt, uint32_t max_tokens = 40, server::Sampling s = {}, uint32_t stop_after = 0) {
  Req r;
  r.prompt = std::move(prompt);
  r.max_tokens = max_tokens;
  r.s = s;
  r.stop_after = stop_after;
  return r;
}

std::vector<uint32_t> P(uint32_t key, uint32_t n) {
  std::vector<uint32_t> p{key};
  for (uint32_t i = 1; i < n; ++i) p.push_back(5000 + key * 131 + i);
  return p;
}

SchedRequest make(const Req& q, Out& out) {
  SchedRequest r;
  r.prompt = q.prompt;
  r.max_tokens = q.max_tokens;
  r.sampling = q.s;
  const uint32_t stop_after = q.stop_after;
  r.on_token = [&out, stop_after](uint32_t id) {
    out.ids.push_back(id);
    return stop_after == 0 || out.ids.size() < stop_after;
  };
  r.on_done = [&out](const Finish& f) {
    out.fin = f;
    ++out.done;
  };
  return r;
}

SchedulerOptions opts(uint32_t fixed_k = 0) {
  SchedulerOptions o;
  o.eos_ids = {248046};
  o.spec.max_k = 7;
  o.spec.cost = server::MtpCost::dflash(true, true);
  o.spec_fixed_k = fixed_k;
  return o;
}

void drain(Scheduler& s) {
  long n = 0;
  while ((s.waiting() != 0 || s.active() != 0) && n++ < 1000000) s.tick();
  CHECK(s.waiting() == 0 && s.active() == 0);
}

void configure(MockBatchEngine& e, uint32_t spec) {
  e.spec = spec;
  e.eos_at[11] = 3;   // EOS as the 3rd generated id
  e.eos_at[12] = 9;
}

std::vector<Out> serve(const std::vector<Req>& reqs, uint32_t slots, uint32_t spec, MockBatchEngine** keep = nullptr,
                       uint32_t fixed_k = 0) {
  static std::vector<std::unique_ptr<MockBatchEngine>> engines;   // outlive the call for inspection
  engines.push_back(std::make_unique<MockBatchEngine>(slots));
  MockBatchEngine& e = *engines.back();
  configure(e, spec);
  if (keep) *keep = &e;
  std::vector<Out> out(reqs.size());
  Scheduler s(e, opts(fixed_k));
  for (size_t i = 0; i < reqs.size(); ++i) CHECK(s.submit(make(reqs[i], out[i])) != nullptr);
  drain(s);
  for (const Out& o : out) CHECK_EQ(o.done, 1);
  return out;
}

void test_lossless_isolated() {
  const std::vector<Req> reqs = {R(P(1, 9)), R(P(2, 13)), R(P(3, 7)), R(P(4, 21))};
  const std::vector<Out> plain = serve(reqs, 4, 0);
  MockBatchEngine* eng = nullptr;
  const std::vector<Out> spec = serve(reqs, 4, 7, &eng);
  uint32_t ids = 0, iters = 0, multi = 0;
  for (size_t i = 0; i < reqs.size(); ++i) {
    CHECK(spec[i].ids == plain[i].ids);
    CHECK(spec[i].fin.reason == plain[i].fin.reason);
    const std::vector<Out> alone = serve({reqs[i]}, 1, 7);
    CHECK(alone[0].ids == spec[i].ids);
    CHECK_EQ(alone[0].fin.spec_iters, spec[i].fin.spec_iters);
    CHECK_EQ(alone[0].fin.spec_drafted, spec[i].fin.spec_drafted);
    CHECK_EQ(alone[0].fin.spec_accepted, spec[i].fin.spec_accepted);
    ids += uint32_t(spec[i].ids.size());
    iters += spec[i].fin.spec_iters;
  }
  for (const std::string& e : eng->events)
    if (e.rfind("spec", 0) == 0 && std::count(e.begin(), e.end(), ':') >= 2) ++multi;
  CHECK(iters < ids);
  CHECK(multi > 0);
  std::printf("S1: 4 requests, %u ids in %u speculative iterations (%u batched steps); ids = the plain scheduler's, "
              "every request's record = its record alone OK\n",
              ids, iters, multi);
}

void test_end_inside_run() {
  // EOS at the 3rd generated id; max_tokens 5; a stop after 4 ids by on_token; EOS at the 9th with min ids.
  std::vector<Req> reqs = {R(P(11, 9)), R(P(5, 10), 5), R(P(6, 8), 40, {}, 4), R(P(12, 6))};
  MockBatchEngine* pe = nullptr;
  MockBatchEngine* se = nullptr;
  const std::vector<Out> plain = serve(reqs, 4, 0, &pe);
  const std::vector<Out> spec = serve(reqs, 4, 7, &se);
  for (size_t i = 0; i < reqs.size(); ++i) {
    CHECK(spec[i].ids == plain[i].ids);
    CHECK(spec[i].fin.reason == plain[i].fin.reason);
    CHECK_EQ(spec[i].fin.completion_tokens, plain[i].fin.completion_tokens);
    const uint32_t slot = uint32_t(spec[i].fin.slot);
    CHECK_EQ(se->pos(slot), pe->pos(uint32_t(plain[i].fin.slot)));
  }
  CHECK(plain[0].fin.reason == Finish::Stop && plain[0].ids.size() == 2);   // EOS consumed, not emitted
  CHECK(plain[1].fin.reason == Finish::Length && plain[1].ids.size() == 5);
  CHECK(plain[2].fin.reason == Finish::Stop && plain[2].ids.size() == 4);
  CHECK(!se->truncations.empty());
  std::printf("S2: EOS / max_tokens / a stop inside a run keep exactly the consumed ids (pos = the plain run's), "
              "%zu runs handed back by truncate_to OK\n",
              se->truncations.size());
}

void test_sampled() {
  server::Sampling seeded;
  seeded.greedy = false;
  seeded.seed = 42;
  seeded.has_seed = true;
  const std::vector<Req> reqs = {R(P(1, 9)), R(P(7, 11), 30, seeded), R(P(3, 7))};
  const std::vector<Out> plain = serve(reqs, 3, 0);
  const std::vector<Out> spec = serve(reqs, 3, 7);
  for (size_t i = 0; i < reqs.size(); ++i) CHECK(spec[i].ids == plain[i].ids);
  std::printf("S3: a seeded sampled row among greedy ones draws what it draws on the plain scheduler OK\n");
}

// A, then its continuation A' = A + A's ids + a tail. `other_slot`: a long request X holds slot 0 when A' arrives,
// so A' is restored on slot 1 from the store; otherwise A' continues on slot 0. Returns A''s (anchor, kept) per
// iteration, from the engine's log of the slot A' ran on.
std::vector<std::pair<uint32_t, uint32_t>> continuation(bool other_slot, bool ring_in_state, Out* b_out) {
  MockBatchEngine e(2);
  configure(e, 7);
  e.ring_in_state = ring_in_state;
  MallocAlloc alloc;
  server::PrefixSlots cache(e, 1 << 20, alloc);
  Scheduler s(e, opts(), &cache);
  const std::vector<uint32_t> A = P(21, 10);
  Out a;
  CHECK(s.submit(make(R(A, 12), a)) != nullptr);
  drain(s);
  CHECK_EQ(a.fin.slot, 0);
  Out x;
  if (other_slot) {
    CHECK(s.submit(make(R(P(22, 9), 300), x)) != nullptr);
    s.tick();   // X granted slot 0
  }
  std::vector<uint32_t> a2 = A;
  a2.insert(a2.end(), a.ids.begin(), a.ids.end());
  for (uint32_t t : {9001u, 9002u, 9003u}) a2.push_back(t);
  Out& b = *b_out;
  const size_t before = e.spec_log[other_slot ? 1 : 0].size();
  CHECK(s.submit(make(R(a2, 24), b)) != nullptr);
  drain(s);
  CHECK_EQ(b.fin.slot, other_slot ? 1 : 0);
  CHECK_EQ(cache.last_kind(uint32_t(b.fin.slot)),
           other_slot ? server::PrefixCache::Plan::Restore : server::PrefixCache::Plan::Continue);
  CHECK_EQ(b.fin.cached_tokens, uint32_t(A.size() + a.ids.size()));
  const auto& log = e.spec_log[uint32_t(b.fin.slot)];
  return std::vector<std::pair<uint32_t, uint32_t>>(log.begin() + long(before), log.end());
}

void test_restore_other_slot() {
  Out ref, got, mutant;
  const auto lr = continuation(false, true, &ref), lg = continuation(true, true, &got),
             lm = continuation(true, false, &mutant);
  CHECK(got.ids == ref.ids && mutant.ids == ref.ids);   // the output never depends on the ring
  CHECK(lg == lr);
  CHECK(!lm.empty() && !lr.empty() && lm.front().first == lr.front().first && lm.front().second == 0 &&
        lr.front().second > 0);
  std::printf("S4: A' restored on slot 1 from A's request-end snapshot drafts exactly as the uninterrupted session "
              "(%zu iterations, the first at pos %u keeping %u); the ring-less snapshot keeps %u there OK\n",
              lg.size(), lg.front().first, lg.front().second, lm.front().second);
}

void test_fixed_k() {
  MockBatchEngine* e = nullptr;
  const std::vector<Out> out = serve({R(P(1, 9)), R(P(2, 13))}, 2, 7, &e, 3);
  CHECK(out[0].ids == serve({R(P(1, 9))}, 1, 0)[0].ids);
  bool all3 = true;
  for (const std::string& ev : e->events)
    if (ev.rfind("spec", 0) == 0)
      for (size_t at = ev.find(':'); at != std::string::npos; at = ev.find(':', at + 1)) all3 &= ev[at + 1] == '3';
  CHECK(all3);
  CHECK_EQ(out[0].fin.spec_drafted, 3 * out[0].fin.spec_iters);
  std::printf("S5: spec_fixed_k 3 asks every row for 3 drafts OK\n");
}

}  // namespace

int main() {
  test_lossless_isolated();
  test_end_inside_run();
  test_sampled();
  test_restore_other_slot();
  test_fixed_k();
  std::printf("dflash_scheduler_test OK\n");
  return 0;
}
