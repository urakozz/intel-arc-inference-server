// Spec 13 B4 (plan 13b Task 1, Review Focus 1-5): the batching scheduler against a mock
// batched engine (tests/server/mock.h, MockBatchEngine), host only. Every request's ids
// must equal the same request served alone; the cases cover streaming order, per-row
// stops, disconnects, per-request seeds, the bounded queue, the prefix plan taken at the
// slot grant, both admission policies, and prefill fairness in mock ticks.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "server/mock.h"
#include "server/scheduler.h"

namespace {

using server::Admission;
using server::Finish;
using server::SchedRequest;
using server::Scheduler;
using server::SchedulerOptions;

struct MallocAlloc : server::HostAlloc {
  void* alloc(size_t n) override { return std::malloc(n); }
  void free(void* p, size_t) override { std::free(p); }
};

// What one request saw.
struct Out {
  std::vector<uint32_t> ids;
  std::vector<double> t;   // engine clock at each id
  Finish fin;
  int done_calls = 0;
  bool token_after_done = false;
};

struct Spec {
  std::vector<uint32_t> prompt;
  uint32_t max_tokens = 16;
  server::Sampling s;
  uint32_t stop_after = 0;   // on_token returns false at this many ids (0: never)
  uint32_t min_tokens = 0;
  bool ignore_eos = false;
};

std::vector<uint32_t> P(uint32_t key, uint32_t n) {
  std::vector<uint32_t> p{key};
  for (uint32_t i = 1; i < n; ++i) p.push_back(5000 + key * 131 + i);
  return p;
}

server::Sampling seeded(uint64_t seed) {
  server::Sampling s;
  s.greedy = false;
  s.seed = seed;
  s.has_seed = true;
  return s;
}

SchedRequest make(const Spec& spec, Out& out, const MockBatchEngine& eng) {
  SchedRequest r;
  r.prompt = spec.prompt;
  r.max_tokens = spec.max_tokens;
  r.sampling = spec.s;
  r.min_tokens = spec.min_tokens;
  r.ignore_eos = spec.ignore_eos;
  const uint32_t stop_after = spec.stop_after;
  r.on_token = [&out, &eng, stop_after](uint32_t id) {
    if (out.done_calls != 0) out.token_after_done = true;
    out.ids.push_back(id);
    out.t.push_back(eng.clock);
    return stop_after == 0 || out.ids.size() < stop_after;
  };
  r.on_done = [&out](const Finish& f) {
    out.fin = f;
    ++out.done_calls;
  };
  return r;
}

SchedulerOptions opts(uint32_t chunk = 512, Admission a = Admission::Interleave) {
  SchedulerOptions o;
  o.chunk = chunk;
  o.admission = a;
  o.eos_ids = {248046};
  return o;
}

void drain(Scheduler& s) {
  long n = 0;
  while ((s.waiting() != 0 || s.active() != 0) && n++ < 10000000) s.tick();
  CHECK(s.waiting() == 0 && s.active() == 0);
}

MockBatchEngine& configure(MockBatchEngine& e) {
  e.eos_at[11] = 3;   // A: EOS as its 3rd generated id
  e.eos_at[41] = 2;
  e.eos_at[42] = 2;
  return e;
}

// The reference: the request alone on a one-slot engine, no cache.
Out alone(const Spec& spec, uint32_t maxlen = 1024) {
  MockBatchEngine eng(1, maxlen);
  configure(eng);
  Scheduler s(eng, opts());
  Out out;
  CHECK(s.submit(make(spec, out, eng)) != nullptr);
  drain(s);
  CHECK_EQ(out.done_calls, 1);
  return out;
}

void same_as_alone(const char* what, const Spec& spec, const Out& got, uint32_t maxlen = 1024) {
  const Out ref = alone(spec, maxlen);
  if (got.ids != ref.ids) {
    std::fprintf(stderr, "%s: ids differ from the request served alone (%zu vs %zu)\n", what,
                 got.ids.size(), ref.ids.size());
    std::exit(1);
  }
  CHECK_EQ(got.fin.reason, ref.fin.reason);
  CHECK_EQ(got.done_calls, 1);
  CHECK(!got.token_after_done);
  CHECK_EQ(got.fin.completion_tokens, static_cast<uint32_t>(got.ids.size()));
}

// B1 at the scheduler, streaming order, Review Focus 4 (per-request seeds).
void test_batch_independence() {
  const std::vector<Spec> specs = {
      {P(1, 10), 30, {}},
      {P(2, 6), 30, seeded(7)},
      {P(3, 13), 25, seeded(7)},
      {P(4, 3), 40, {}},
  };
  MockBatchEngine eng(4);
  configure(eng);
  Scheduler s(eng, opts(4));   // chunks of 4: every prompt but the last is chunked
  std::vector<Out> outs(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) CHECK(s.submit(make(specs[i], outs[i], eng)) != nullptr);
  drain(s);
  bool four_rows = false;
  for (const auto& e : eng.events) four_rows = four_rows || e == "step 0 1 2 3";
  CHECK(four_rows);
  for (size_t i = 0; i < specs.size(); ++i) {
    same_as_alone("batch independence", specs[i], outs[i]);
    CHECK_EQ(outs[i].fin.reason, Finish::Length);
  }
  CHECK(outs[1].ids != outs[2].ids);
  // The seeded request with other company (a different batch, other slots): same ids.
  MockBatchEngine eng2(2);
  configure(eng2);
  Scheduler s2(eng2, opts(512));
  Out other, again;
  CHECK(s2.submit(make({P(9, 20), 50, seeded(3)}, other, eng2)) != nullptr);
  s2.tick();
  s2.tick();
  CHECK(s2.submit(make(specs[1], again, eng2)) != nullptr);
  drain(s2);
  CHECK(again.ids == outs[1].ids);
  CHECK_EQ(again.fin.slot, 1);
  Spec other_seed = specs[1];
  other_seed.s = seeded(8);
  CHECK(alone(other_seed).ids != outs[1].ids);
  std::printf("batch independence and seeds: 4 rows in one step, each equal to its run alone\n");
}

// Review Focus 2: EOS, max_tokens and a stop string each end only their own row; every
// finished slot's final state is snapshotted before the slot is reused.
void test_per_row_stops() {
  MockBatchEngine eng(3);
  configure(eng);
  MallocAlloc alloc;
  server::PrefixSlots cache(eng, 1 << 20, alloc);
  Scheduler s(eng, opts(512), &cache);
  const std::vector<Spec> specs = {
      {P(11, 7), 50, {}},           // EOS as its 3rd id: 2 ids, Stop
      {P(12, 5), 5, {}},            // max_tokens: 5 ids, Length
      {P(13, 9), 50, {}, 4},        // a stop string at its 4th id: Stop
      {P(14, 6), 20, {}},           // waits for a slot
      {P(15, 4), 12, seeded(5)},    // waits for a slot
  };
  std::vector<Out> outs(specs.size());
  std::vector<std::string> last_event_at_done(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    SchedRequest r = make(specs[i], outs[i], eng);
    auto inner = r.on_done;
    r.on_done = [&, i, inner](const Finish& f) {
      last_event_at_done[i] = eng.events.back();
      inner(f);
    };
    CHECK(s.submit(std::move(r)) != nullptr);
  }
  drain(s);
  for (size_t i = 0; i < specs.size(); ++i) same_as_alone("per-row stops", specs[i], outs[i]);
  CHECK_EQ(outs[0].fin.reason, Finish::Stop);
  CHECK_EQ(outs[0].ids.size(), size_t(2));
  CHECK_EQ(outs[1].fin.reason, Finish::Length);
  CHECK_EQ(outs[1].ids.size(), size_t(5));
  CHECK_EQ(outs[2].fin.reason, Finish::Stop);
  CHECK_EQ(outs[2].ids.size(), size_t(4));
  CHECK_EQ(outs[3].fin.reason, Finish::Length);
  CHECK_EQ(outs[3].ids.size(), size_t(20));
  // The snapshot is the last engine call before on_done: at the slot's final pos, which is
  // the prompt plus every id a step consumed (the EOS that ended A included).
  const uint32_t consumed[] = {3, 5, 4, 20, 12};
  for (size_t i = 0; i < specs.size(); ++i) {
    const std::string want = MockBatchEngine::ev("save_state", uint32_t(outs[i].fin.slot),
                                                 uint32_t(specs[i].prompt.size()) + consumed[i]);
    if (last_event_at_done[i] != want) {
      std::fprintf(stderr, "request %zu: last event at done '%s', want '%s'\n", i,
                   last_event_at_done[i].c_str(), want.c_str());
      std::exit(1);
    }
  }
  // The two late requests took slots the first three freed.
  CHECK(outs[3].fin.slot >= 0 && outs[3].fin.slot < 3);
  CHECK(outs[4].fin.slot >= 0 && outs[4].fin.slot < 3);
  std::printf("per-row stops: EOS / max_tokens / stop string end one row each; "
              "final state snapshotted before reuse\n");
}

// Review Focus 1: a disconnect frees the slot at the next tick; the others keep going.
void test_disconnect() {
  MockBatchEngine eng(2);
  configure(eng);
  Scheduler s(eng, opts());
  Out a, b, c, d;
  auto ja = s.submit(make({P(21, 5), 500, {}}, a, eng));
  auto jb = s.submit(make({P(22, 5), 40, {}}, b, eng));
  auto jc = s.submit(make({P(23, 5), 30, {}}, c, eng));
  CHECK(ja && jb && jc);
  while (a.ids.size() < 5) s.tick();
  CHECK_EQ(s.active(), size_t(2));
  CHECK_EQ(s.waiting(), size_t(1));
  s.cancel(ja);
  const size_t b_before = b.ids.size();
  s.tick();
  CHECK_EQ(a.done_calls, 1);
  CHECK_EQ(a.fin.reason, Finish::Cancelled);
  CHECK_EQ(a.fin.completion_tokens, 5U);
  CHECK(ja->finished());
  CHECK_EQ(b.ids.size(), b_before + 1);   // B's row did not stall
  CHECK_EQ(s.waiting(), size_t(0));       // C took A's slot at the same tick
  CHECK_EQ(c.ids.size(), size_t(1));
  // A waiting request cancelled before its grant never touches the engine.
  auto jd = s.submit(make({P(24, 5), 30, {}}, d, eng));
  s.cancel(jd);
  s.tick();
  CHECK_EQ(d.done_calls, 1);
  CHECK_EQ(d.fin.reason, Finish::Cancelled);
  CHECK_EQ(d.fin.slot, -1);
  drain(s);
  same_as_alone("disconnect, the neighbour", {P(22, 5), 40, {}}, b);
  same_as_alone("disconnect, the late request", {P(23, 5), 30, {}}, c);
  CHECK_EQ(a.ids.size(), size_t(5));

  // The same on the scheduler's own thread, the cancel from another thread.
  MockBatchEngine teng(2, 1 << 16);
  teng.step_us = 200;
  Scheduler ts(teng, opts());
  ts.start();
  std::atomic<size_t> e_ids{0};
  std::atomic<bool> e_done{false}, f_done{false};
  Finish e_fin, f_fin;
  SchedRequest e;
  e.prompt = P(25, 8);
  e.max_tokens = 60000;
  e.on_token = [&](uint32_t) { ++e_ids; return true; };
  e.on_done = [&](const Finish& f) { e_fin = f; e_done = true; };
  SchedRequest f;
  f.prompt = P(26, 8);
  f.max_tokens = 300;
  f.on_token = [](uint32_t) { return true; };
  f.on_done = [&](const Finish& fin) { f_fin = fin; f_done = true; };
  auto je = ts.submit(std::move(e));
  CHECK(ts.submit(std::move(f)) != nullptr);
  while (e_ids < 3) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ts.cancel(je);
  while (!e_done || !f_done) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ts.stop();
  CHECK_EQ(e_fin.reason, Finish::Cancelled);
  CHECK(e_fin.completion_tokens < 60000U);
  CHECK_EQ(f_fin.reason, Finish::Length);
  CHECK_EQ(f_fin.completion_tokens, 300U);
  std::printf("disconnect: slot freed at the next tick, neighbour emitted that tick; "
              "threaded cancel ends after %u ids\n", e_fin.completion_tokens);
}

// Review Focus 5: the bounded queue, and the prefix plan taken at the slot grant.
void test_queue_and_plan_at_grant() {
  {
    MockBatchEngine eng(2);
    SchedulerOptions o = opts();
    o.queue_depth = 1;
    Scheduler s(eng, o);
    Out x[5];
    CHECK(s.submit(make({P(31, 4), 8, {}}, x[0], eng)) != nullptr);
    CHECK(s.submit(make({P(32, 4), 8, {}}, x[1], eng)) != nullptr);
    CHECK(s.submit(make({P(33, 4), 8, {}}, x[2], eng)) != nullptr);
    CHECK(s.submit(make({P(34, 4), 8, {}}, x[3], eng)) == nullptr);   // 2 slots + 1 queued
    s.tick();
    CHECK_EQ(s.active(), size_t(2));
    CHECK_EQ(s.waiting(), size_t(1));
    CHECK(s.submit(make({P(35, 4), 8, {}}, x[4], eng)) == nullptr);
    drain(s);
    CHECK_EQ(x[2].fin.reason, Finish::Length);
    CHECK_EQ(x[3].done_calls, 0);   // refused requests are never called back
  }
  {
    // One slot: B's prompt continues A's conversation (A's prompt and output, then a new
    // turn). B arrives while A decodes; at its arrival the slot holds only part of what
    // B needs. Its plan is taken when A's slot is granted to it: Continue on A's whole
    // session.
    const Spec a{P(36, 9), 6, {}};
    const Out a_alone = alone(a);
    Spec b{a.prompt, 10, {}};
    b.prompt.insert(b.prompt.end(), a_alone.ids.begin(), a_alone.ids.end());
    for (uint32_t i = 0; i < 5; ++i) b.prompt.push_back(7000 + i);
    MockBatchEngine eng(1);
    MallocAlloc alloc;
    server::PrefixSlots cache(eng, 1 << 20, alloc);
    Scheduler s(eng, opts(), &cache);
    Out oa, ob;
    CHECK(s.submit(make(a, oa, eng)) != nullptr);
    s.tick();
    s.tick();
    CHECK(s.submit(make(b, ob, eng)) != nullptr);
    drain(s);
    same_as_alone("plan at grant, A", a, oa);
    same_as_alone("plan at grant, B", b, ob);
    CHECK_EQ(ob.fin.cached_tokens, uint32_t(a.prompt.size() + a_alone.ids.size()));
    CHECK_EQ(cache.last_kind(0), server::PrefixCache::Plan::Continue);
    // C, a new conversation, takes the slot; then D continues A's again: a host restore.
    Spec c{P(37, 7), 4, {}};
    Spec d = b;
    d.prompt.push_back(7100);
    Out oc, od;
    CHECK(s.submit(make(c, oc, eng)) != nullptr);
    CHECK(s.submit(make(d, od, eng)) != nullptr);
    drain(s);
    same_as_alone("plan at grant, C", c, oc);
    same_as_alone("plan at grant, D (restored)", d, od);
    CHECK_EQ(cache.last_kind(0), server::PrefixCache::Plan::Restore);
    CHECK(od.fin.cached_tokens > 0);
    std::printf("queue full refused at slots + depth; plan at grant: B cached %u (Continue), "
                "D cached %u (Restore)\n", ob.fin.cached_tokens, od.fin.cached_tokens);
  }
}

// Spec 13 §8: batched admission holds a burst, then prefills it in one call. Spec 31: the same with the slots' lengths
// as one page pool (`pool`).
void test_batched_admission(bool pool = false) {
  double now = 0;
  MockBatchEngine eng(4);
  if (pool) eng.pool = 4 * eng.maxlen / eng.blk;
  SchedulerOptions o = opts(512, Admission::Batched);
  o.burst_hold_s = 0.005;
  o.clock = [&now] { return now; };
  Scheduler s(eng, o);
  Out x[7];
  CHECK(s.submit(make({P(51, 6), 5, {}}, x[0], eng)) != nullptr);
  CHECK(!s.tick());   // held
  CHECK_NEAR(s.hold_remaining(), 0.005, 1e-12);
  now = 0.002;
  CHECK(s.submit(make({P(52, 9), 5, {}}, x[1], eng)) != nullptr);
  now = 0.003;
  CHECK(s.submit(make({P(53, 3), 5, {}}, x[2], eng)) != nullptr);
  CHECK(!s.tick());
  CHECK(eng.events.empty());
  now = 0.0051;
  CHECK(s.tick());
  // The grants reset slots 0..2, then one prefill call for the burst and one step.
  CHECK_EQ(eng.events.size(), size_t(5));
  CHECK_EQ(eng.events[3], std::string("prefill 0:6 1:9 2:3"));
  CHECK_EQ(eng.events[4], std::string("step 0 1 2"));
  drain(s);
  // A burst that fills the free slots is admitted at once.
  const size_t before = eng.events.size();
  for (int i = 3; i < 7; ++i) CHECK(s.submit(make({P(54 + i, 4), 3, {}}, x[i], eng)) != nullptr);
  CHECK(s.tick());
  // A pool grants the slot holding the fewest pages first, and a grant that gives up an idle session evicts it (one
  // more reset): the same four chunks in one call, in another slot order.
  if (pool) {
    const auto it = std::find_if(eng.events.begin() + before, eng.events.end(),
                                 [](const std::string& e) { return e.rfind("prefill ", 0) == 0; });
    CHECK(it != eng.events.end() && std::count(it->begin(), it->end(), ':') == 4);
  } else {
    CHECK_EQ(eng.events[before + 4], std::string("prefill 0:4 1:4 2:4 3:4"));
  }
  drain(s);
  for (int i = 0; i < 7; ++i) CHECK_EQ(x[i].fin.reason, Finish::Length);
  same_as_alone("batched admission", {P(52, 9), 5, {}}, x[1]);
  std::printf("batched admission%s: a 3-request burst held 5 ms then one prefill call\n", pool ? " (a page pool)" : "");
}

// Review Focus 3 / B4 fairness: a running request's worst inter-token gap (mock ticks: a
// decode step costs 1, prefill 32 ids per tick) while a 32k prompt prefills.
double worst_gap(Admission a, uint32_t chunk, uint32_t* ttft_ticks, bool pool = false) {
  MockBatchEngine eng(4, 40000);
  if (pool) eng.pool = 4 * eng.maxlen / eng.blk;   // spec 31: the four slots' positions as one pool
  SchedulerOptions o = opts(chunk, a);
  o.burst_hold_s = 0;
  Scheduler s(eng, o);
  const Spec r{P(61, 16), 3000, {}};
  const Spec l{P(62, 32768), 4, {}};
  Out orr, ol;
  CHECK(s.submit(make(r, orr, eng)) != nullptr);
  while (orr.ids.size() < 5) s.tick();
  const double t_arrive = eng.clock;
  CHECK(s.submit(make(l, ol, eng)) != nullptr);
  while (ol.done_calls == 0) s.tick();
  double gap = 0;
  for (size_t i = 1; i < orr.t.size(); ++i) gap = std::max(gap, orr.t[i] - orr.t[i - 1]);
  *ttft_ticks = static_cast<uint32_t>(ol.t.at(0) - t_arrive);
  same_as_alone("fairness, the long prompt", l, ol, 40000);
  return gap;
}

void test_fairness() {
  uint32_t ttft[3];
  const double g512 = worst_gap(Admission::Interleave, 512, &ttft[0]);
  const double g2048 = worst_gap(Admission::Interleave, 2048, &ttft[1]);
  const double gb = worst_gap(Admission::Batched, 512, &ttft[2]);
  std::printf("fairness (32768-id prompt while a request decodes; mock ticks, step = 1, "
              "prefill 32 ids/tick):\n"
              "  interleave chunk 512:  worst gap %.0f, long prompt's first id after %u\n"
              "  interleave chunk 2048: worst gap %.0f, first id after %u\n"
              "  batched admission:     worst gap %.0f, first id after %u\n",
              g512, ttft[0], g2048, ttft[1], gb, ttft[2]);
  CHECK(g512 <= 512.0 / 32 + 1 + 1e-9);
  CHECK(g2048 <= 2048.0 / 32 + 1 + 1e-9);
  CHECK(gb >= 32768.0 / 32);
  // Spec 31: a page pool changes nothing here.
  uint32_t t;
  CHECK_EQ(worst_gap(Admission::Interleave, 512, &t, true), g512);
  CHECK_EQ(t, ttft[0]);
  CHECK_EQ(worst_gap(Admission::Batched, 512, &t, true), gb);
  CHECK_EQ(t, ttft[2]);
}

// EOS rules as server.cc's: min_tokens and ignore_eos keep the EOS as an id; max_tokens 0.
void test_eos_rules_and_errors() {
  MockBatchEngine eng(2);
  configure(eng);
  Scheduler s(eng, opts());
  Out a, b, c, d, e;
  Spec sa{P(41, 4), 6, {}};
  sa.min_tokens = 3;     // the EOS at id 2 is below min_tokens: kept, generation goes on
  Spec sb{P(42, 4), 5, {}};
  sb.ignore_eos = true;
  CHECK(s.submit(make(sa, a, eng)) != nullptr);
  CHECK(s.submit(make(sb, b, eng)) != nullptr);
  CHECK(s.submit(make({P(43, 4), 0, {}}, c, eng)) != nullptr);
  drain(s);
  CHECK_EQ(a.ids.size(), size_t(6));
  CHECK_EQ(a.ids[1], 248046U);
  CHECK_EQ(b.ids.size(), size_t(5));
  CHECK_EQ(b.ids[1], 248046U);
  CHECK_EQ(c.ids.size(), size_t(0));
  CHECK_EQ(c.fin.reason, Finish::Length);
  same_as_alone("min_tokens", sa, a);
  same_as_alone("ignore_eos", sb, b);
  // A prefill that throws fails that request only.
  eng.bad_id = 9999;
  std::vector<uint32_t> bad = P(44, 6);
  bad[3] = 9999;
  CHECK(s.submit(make({bad, 5, {}}, d, eng)) != nullptr);
  CHECK(s.submit(make({P(45, 6), 9, {}}, e, eng)) != nullptr);
  drain(s);
  CHECK_EQ(d.fin.reason, Finish::Error);
  CHECK(d.fin.error.find("vocabulary") != std::string::npos);
  CHECK_EQ(e.fin.reason, Finish::Length);
  same_as_alone("after an error", {P(45, 6), 9, {}}, e);
  // Validation at submit.
  bool threw = false;
  try {
    s.submit(make({{}, 5, {}}, d, eng));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    s.submit(make({P(46, 1024), 5, {}}, d, eng));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  // max_tokens is clamped to the slot's room.
  Out f;
  CHECK(s.submit(make({P(47, 1020), 100, {}}, f, eng)) != nullptr);
  drain(s);
  CHECK_EQ(f.ids.size(), size_t(4));
  std::printf("EOS rules, max_tokens 0 and clamping, a failing prefill: as server.cc\n");
}


// ---- Plan 13b Task 2: the scheduler catches up with the server ---------------------------

// Per-slot lengths: a request goes to a free slot it fits, its max_tokens clamped to that
// slot's room; a prompt no slot fits is refused at submit.
void test_slot_lengths() {
  MockBatchEngine eng(2);
  eng.lens = {64, 1024};
  configure(eng);
  Scheduler s(eng, opts());
  Out a, b, c;
  CHECK(s.submit(make({P(71, 100), 8, {}}, a, eng)) != nullptr);   // fits slot 1 only
  CHECK(s.submit(make({P(72, 60), 100, {}}, b, eng)) != nullptr);  // slot 0: 4 ids of room
  drain(s);
  CHECK_EQ(a.fin.slot, 1);
  CHECK_EQ(a.ids.size(), size_t(8));
  CHECK_EQ(b.fin.slot, 0);
  CHECK_EQ(b.ids.size(), size_t(4));
  CHECK_EQ(b.fin.reason, Finish::Length);
  // Two long prompts and one long slot: the second waits for slot 1 (never slot 0).
  Out d, e;
  CHECK(s.submit(make({P(73, 200), 6, {}}, d, eng)) != nullptr);
  CHECK(s.submit(make({P(74, 300), 6, {}}, e, eng)) != nullptr);
  drain(s);
  CHECK_EQ(d.fin.slot, 1);
  CHECK_EQ(e.fin.slot, 1);
  same_as_alone("slot lengths, the waiting one", {P(74, 300), 6, {}}, e);
  bool threw = false;
  try {
    s.submit(make({P(75, 1024), 5, {}}, c, eng));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("slot lengths: a 100-id prompt skips the 64-position slot; max_tokens clamped to the slot's room\n");
}

// --prefix-split-last (Review Focus 3): the prompt is prefilled to len - 1, the prompt-end
// snapshot taken there, the last id replayed; a later request whose prompt diverges at the
// last id restores at len - 1 on another slot.
void test_split_last() {
  MockBatchEngine eng(2);
  configure(eng);
  MallocAlloc alloc;
  server::PrefixSlots cache(eng, 1 << 20, alloc);
  Scheduler s(eng, opts(), &cache);
  const std::vector<uint32_t> prompt = P(81, 11);
  Out a;
  SchedRequest ra = make({prompt, 6, {}}, a, eng);
  ra.split_last = true;
  CHECK(s.submit(std::move(ra)) != nullptr);
  drain(s);
  same_as_alone("split-last", {prompt, 6, {}}, a);
  bool prefill10 = false, replay = false, snap10 = false;
  for (const auto& e : eng.events) {
    prefill10 = prefill10 || e == "prefill 0:10";
    replay = replay || (prefill10 && e == "replay 0:1");
    snap10 = snap10 || (prefill10 && !replay && e == MockBatchEngine::ev("save_state", 0, 10));
  }
  CHECK(prefill10 && replay && snap10);
  // B: the same conversation re-rendered (the last prompt id differs), on slot 1 while slot
  // 0 is busy: a host restore at len - 1.
  Out x, b;
  CHECK(s.submit(make({prompt, 400, {}}, x, eng)) != nullptr);   // slot 0 (its match)
  s.tick();
  std::vector<uint32_t> pb(prompt.begin(), prompt.end() - 1);
  pb.push_back(9999);
  pb.push_back(9998);
  SchedRequest rb = make({pb, 5, {}}, b, eng);
  rb.split_last = true;
  CHECK(s.submit(std::move(rb)) != nullptr);
  drain(s);
  CHECK_EQ(x.fin.slot, 0);
  CHECK_EQ(b.fin.slot, 1);
  CHECK_EQ(b.fin.cached_tokens, uint32_t(prompt.size() - 1));
  CHECK_EQ(cache.last_kind(1), server::PrefixCache::Plan::Restore);
  same_as_alone("split-last, restored at len - 1", {pb, 5, {}}, b);
  // Without a cache split_last is ignored (PrefixSession's rule): one prefill of the whole prompt.
  MockBatchEngine e2(1);
  Scheduler s2(e2, opts());
  Out c;
  SchedRequest rc = make({prompt, 3, {}}, c, e2);
  rc.split_last = true;
  CHECK(s2.submit(std::move(rc)) != nullptr);
  drain(s2);
  for (const auto& e : e2.events) CHECK(e != "replay 0:1");
  std::printf("split-last: prefill to len - 1, snapshot there, the last id replayed; B restored at %u\n",
              b.fin.cached_tokens);
}

SchedulerOptions lookup_opts(uint32_t history = 0) {
  SchedulerOptions o = opts();
  o.lookup = true;
  o.lookup_history = history;
  o.lookup_adaptive.max_k = 3;
  o.lookup_adaptive.cost = server::MtpCost::int8_head().with_free_drafts();
  return o;
}

// A prompt that repeats a phrase: prompt lookup finds matches.
std::vector<uint32_t> repetitive(uint32_t key, uint32_t n) {
  std::vector<uint32_t> p{key};
  for (uint32_t i = 1; i < n; ++i) p.push_back(1000 + (i * 7 + key) % 24);
  return p;
}

// Prompt lookup per request (Review Focus 4): the ids equal the plain run's; each slot's
// drafts come from its own request's ids only; a request's K sequence and acceptance are the
// same alone and in company.
void test_lookup_per_request() {
  const std::vector<Spec> specs = {{repetitive(91, 60), 40, {}}, {repetitive(92, 45), 40, seeded(5)}};
  // Alone, with lookup, on a one-slot engine; and the plain reference.
  std::vector<Out> alone_lk(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    MockBatchEngine e(1);
    e.lookup = 3;
    e.echo = 24;
    Scheduler s(e, lookup_opts());
    CHECK(s.submit(make(specs[i], alone_lk[i], e)) != nullptr);
    drain(s);
    if (specs[i].s.greedy) CHECK(alone_lk[i].fin.spec_accepted > 0);
  }
  MockBatchEngine eng(2);
  eng.lookup = 3;
  eng.echo = 24;
  Scheduler s(eng, lookup_opts());
  std::vector<Out> outs(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) CHECK(s.submit(make(specs[i], outs[i], eng)) != nullptr);
  drain(s);
  for (size_t i = 0; i < specs.size(); ++i) {
    // The plain reference: the same echo engine, no lookup.
    MockBatchEngine pe(1);
    pe.echo = 24;
    Scheduler ps(pe, opts());
    Out plain;
    CHECK(ps.submit(make(specs[i], plain, pe)) != nullptr);
    drain(ps);
    CHECK(outs[i].ids == plain.ids && alone_lk[i].ids == plain.ids);
    CHECK_EQ(outs[i].fin.spec_iters, alone_lk[i].fin.spec_iters);
    CHECK_EQ(outs[i].fin.spec_drafted, alone_lk[i].fin.spec_drafted);
    CHECK_EQ(outs[i].fin.spec_accepted, alone_lk[i].fin.spec_accepted);
    // Every draft is an id of this request's own prompt or output.
    const uint32_t slot = uint32_t(outs[i].fin.slot);
    std::set<uint32_t> own(specs[i].prompt.begin(), specs[i].prompt.end());
    own.insert(outs[i].ids.begin(), outs[i].ids.end());
    for (const auto& d : eng.proposals_of[slot])
      for (uint32_t id : d) CHECK(own.count(id) != 0);
  }
  // The key ids 91 / 92 are each in one prompt only: never drafted for the other.
  for (const auto& d : eng.proposals_of[uint32_t(outs[1].fin.slot)])
    for (uint32_t id : d) CHECK(id != 91);
  // lookup_history 1: the slot's earlier request's output seeds the next one's matcher; ids unchanged.
  MockBatchEngine eh(1);
  eh.lookup = 3;
  Scheduler sh(eh, lookup_opts(1));
  Out h1, h2;
  CHECK(sh.submit(make(specs[0], h1, eh)) != nullptr);
  drain(sh);
  CHECK(sh.submit(make({P(93, 9), 30, {}}, h2, eh)) != nullptr);
  drain(sh);
  same_as_alone("lookup history", {P(93, 9), 30, {}}, h2);
  // A lookup scheduler needs a verifying engine without its own proposer.
  MockBatchEngine plain(1);
  bool threw = false;
  try {
    Scheduler bad(plain, lookup_opts());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("lookup per request: ids as plain decode, drafts from the request's own ids, K and acceptance as "
              "alone (%u / %u drafts kept)\n", outs[0].fin.spec_accepted, outs[0].fin.spec_drafted);
}

// Review Focus 2: a failure on one card (SlotsFailed) ends that slot's request only.
void test_slot_failure() {
  {
    MockBatchEngine eng(2);
    configure(eng);
    eng.fail_at[1] = 3;   // slot 1's third step fails
    Scheduler s(eng, opts());
    Out a, b;
    CHECK(s.submit(make({P(101, 6), 20, {}}, a, eng)) != nullptr);
    CHECK(s.submit(make({P(102, 6), 20, {}}, b, eng)) != nullptr);
    drain(s);
    CHECK_EQ(b.fin.slot, 1);
    CHECK_EQ(b.fin.reason, Finish::Error);
    CHECK(b.fin.error.find("slot 1") != std::string::npos);
    CHECK_EQ(b.ids.size(), size_t(2));
    CHECK_EQ(a.fin.reason, Finish::Length);
    same_as_alone("a neighbour's card failed", {P(101, 6), 20, {}}, a);
  }
  {
    MockBatchEngine eng(2);
    configure(eng);
    eng.fail_prefill = true;
    eng.fail_at[0] = 1;   // slot 0's first prefill fails, in a batched admission call
    Scheduler s(eng, opts(512, Admission::Batched));
    Out a, b;
    CHECK(s.submit(make({P(103, 6), 9, {}}, a, eng)) != nullptr);
    CHECK(s.submit(make({P(104, 6), 9, {}}, b, eng)) != nullptr);
    drain(s);
    CHECK_EQ(a.fin.reason, Finish::Error);
    CHECK_EQ(b.fin.reason, Finish::Length);
    same_as_alone("a neighbour's prefill failed", {P(104, 6), 9, {}}, b);
  }
  std::printf("slot failure: a step or prefill failing on one card ends that request only\n");
}

// The engine's prefill cut and the overlapped prefill: a replica engine's chunks end where
// its card's one call would cut, and each prefill call runs beside the other card's step.
void test_prefill_cut_and_overlap() {
  // Review (plan 13b): a card's prefill must not slow the other card's decode. Each chunk keeps its card busy for
  // 5 polls (6 ticks); A, decoding, still gets one id EVERY tick while B's 30 ids prefill in 8 / 8 / 8 / 6.
  MockBatchEngine eng(2, 4096);
  configure(eng);
  eng.cut = 8;
  eng.overlap = true;
  eng.prefill_polls = 5;
  Scheduler s(eng, opts());
  Out a, b;
  CHECK(s.submit(make({P(111, 13), 400, {}}, a, eng)) != nullptr);   // 8 + 5, nothing else decoding
  while (a.ids.size() < 3) s.tick();
  CHECK(s.submit(make({P(112, 30), 4, {}}, b, eng)) != nullptr);
  size_t ticks = 0, b_prefill_ticks = 0;
  while (b.done_calls == 0) {
    const size_t before = a.ids.size();
    const bool b_prefilling = b.ids.empty();
    s.tick();
    ++ticks;
    if (b_prefilling) ++b_prefill_ticks;
    CHECK_EQ(a.ids.size(), before + 1);   // the idle rate: one id a tick
  }
  CHECK(b_prefill_ticks >= 4 * 6);   // B's four chunks really took their polls
  std::vector<std::string> pf;
  for (const auto& e : eng.events)
    if (e.rfind("prefill", 0) == 0) pf.push_back(e);
  const std::vector<std::string> want = {"prefill 0:8", "prefill 0:5", "prefill 1:8", "prefill 1:8", "prefill 1:8",
                                         "prefill 1:6"};
  CHECK(pf == want);
  s.cancel(nullptr);
  while (s.active() != 0) {
    const size_t before = a.ids.size();
    s.tick();
    CHECK(a.ids.size() == before + 1 || a.done_calls == 1);
  }
  same_as_alone("cut and overlap, A", {P(111, 13), 400, {}}, a, 4096);
  same_as_alone("cut and overlap, B", {P(112, 30), 4, {}}, b, 4096);
  std::printf("overlap: A got one id every one of %zu ticks while B's 8/8/8/6 chunks prefilled (%zu ticks)\n", ticks,
              b_prefill_ticks);
}

// Review (plan 13b): two prompts arriving together prefill at the same time, one on each card.
void test_parallel_prefill() {
  MockBatchEngine eng(2, 4096);
  configure(eng);
  eng.cut = 8;
  eng.overlap = true;
  eng.prefill_polls = 3;
  Scheduler s(eng, opts());
  Out a, b;
  CHECK(s.submit(make({P(113, 20), 6, {}}, a, eng)) != nullptr);
  CHECK(s.submit(make({P(114, 20), 6, {}}, b, eng)) != nullptr);
  s.tick();
  // The first tick started both slots' first chunks in one prefill_begin; nothing joined yet.
  std::vector<std::string> first;
  for (const auto& e : eng.events)
    if (e == "begin" || e.rfind("prefill", 0) == 0 || e.rfind("end", 0) == 0) first.push_back(e);
  CHECK(first == std::vector<std::string>({"begin", "prefill 0:8", "prefill 1:8"}));
  drain(s);
  // Every later begin also carried both slots' chunks (8 / 8 / 4 each), side by side.
  size_t both = 0;
  for (size_t i = 0; i + 2 < eng.events.size(); ++i)
    if (eng.events[i] == "begin" && eng.events[i + 1].rfind("prefill 0:", 0) == 0 &&
        eng.events[i + 2].rfind("prefill 1:", 0) == 0)
      ++both;
  CHECK_EQ(both, size_t(3));
  same_as_alone("parallel prefill, A", {P(113, 20), 6, {}}, a, 4096);
  same_as_alone("parallel prefill, B", {P(114, 20), 6, {}}, b, 4096);
  std::printf("parallel prefill: two prompts prefilled side by side, 3 chunks each\n");
}

// Review (plan 13b): a callback that throws ends its own request with Error and nothing else - mid-step beside the
// other slot's prefill in flight, and beside another row of the same step; on_done throwing changes nothing either.
void test_callback_throws() {
  {
    MockBatchEngine eng(2, 4096);
    configure(eng);
    eng.cut = 8;
    eng.overlap = true;
    eng.prefill_polls = 4;
    Scheduler s(eng, opts());
    Out a, b;
    SchedRequest ra = make({P(121, 9), 50, {}}, a, eng);
    auto inner = ra.on_token;
    ra.on_token = [inner, &a](uint32_t id) -> bool {
      if (a.ids.size() == 2) throw std::runtime_error("the streamer broke");
      return inner(id);
    };
    CHECK(s.submit(std::move(ra)) != nullptr);
    while (a.ids.size() < 1) s.tick();
    CHECK(s.submit(make({P(122, 40), 7, {}}, b, eng)) != nullptr);   // prefilling on slot 1 when A throws
    drain(s);
    CHECK_EQ(a.fin.reason, Finish::Error);
    CHECK(a.fin.error.find("streamer broke") != std::string::npos);
    CHECK_EQ(a.done_calls, 1);
    CHECK_EQ(b.fin.reason, Finish::Length);
    same_as_alone("a neighbour's callback threw", {P(122, 40), 7, {}}, b, 4096);
    // The engine recovered: both slots serve again, nothing left in flight.
    Out c, d;
    CHECK(s.submit(make({P(123, 17), 5, {}}, c, eng)) != nullptr);
    CHECK(s.submit(make({P(124, 23), 5, {}}, d, eng)) != nullptr);
    drain(s);
    same_as_alone("after the throw, C", {P(123, 17), 5, {}}, c, 4096);
    same_as_alone("after the throw, D", {P(124, 23), 5, {}}, d, 4096);
  }
  {
    MockBatchEngine eng(2);
    configure(eng);
    Scheduler s(eng, opts());
    Out a, b, c;
    SchedRequest ra = make({P(125, 6), 30, {}}, a, eng);
    auto inner = ra.on_token;
    ra.on_token = [inner, &a](uint32_t id) -> bool {
      if (a.ids.size() == 3) throw std::runtime_error("the parser broke");
      return inner(id);
    };
    CHECK(s.submit(std::move(ra)) != nullptr);
    CHECK(s.submit(make({P(126, 6), 30, {}}, b, eng)) != nullptr);
    drain(s);
    CHECK_EQ(a.fin.reason, Finish::Error);
    same_as_alone("the other row of the throwing step", {P(126, 6), 30, {}}, b);
    // on_done throwing: the request is over, the scheduler goes on.
    SchedRequest rc = make({P(127, 6), 4, {}}, c, eng);
    auto done = rc.on_done;
    rc.on_done = [done](const Finish& f) {
      done(f);
      throw std::runtime_error("on_done broke");
    };
    CHECK(s.submit(std::move(rc)) != nullptr);
    Out e;
    CHECK(s.submit(make({P(128, 6), 4, {}}, e, eng)) != nullptr);
    drain(s);
    CHECK_EQ(c.done_calls, 1);
    same_as_alone("after on_done threw", {P(128, 6), 4, {}}, e);
  }
  std::printf("callback throws: its own request ends with Error; the neighbour, the next requests unaffected\n");
}

}  // namespace

// Spec 12b: PrefixSlots keys its cache by the batched engine's KV form, as PrefixSession
// does for EngineIface, so a bf16 entry is never restored into an int8 engine.
void test_kv_form() {
  struct Int8Engine : MockBatchEngine {
    using MockBatchEngine::MockBatchEngine;
    uint64_t kv_form() override { return 1; }
  };
  MallocAlloc alloc;
  MockBatchEngine bf16(2);
  configure(bf16);
  CHECK_EQ(server::PrefixSlots(bf16, 1 << 20, alloc).cache().kv_form(), uint64_t(0));
  Int8Engine int8(2);
  configure(int8);
  CHECK_EQ(server::PrefixSlots(int8, 1 << 20, alloc).cache().kv_form(), uint64_t(1));
  std::printf("kv form: PrefixSlots keys by the engine's KV form OK\n");
}

// PERF-91: with an engine that peeks (pending_id), each request's first id goes out after the prompt's last chunk,
// before the first step - plain and speculative - and every request ends exactly as without the peek: the same ids,
// reasons and engine calls, also when the first id ends it (EOS, max_tokens 1, a stop string at once). The chunk that
// ends the prompt carries the request's sampling: a sampled request's first id is drawn there.
// Spec 31 (`pool`): the same with the slots' lengths as one page pool.
void test_first_id_peek(bool pool = false) {
  const std::vector<Spec> specs = {
      {P(1, 10), 12, {}},
      {P(2, 6), 12, seeded(7)},
      {P(21, 5), 12, {}},      // EOS as its first id
      {P(22, 5), 12, {}, 0, 1},   // EOS first but min_tokens 1
      {P(3, 7), 1, {}},        // max_tokens 1
      {P(4, 8), 12, {}, 1},    // a stop string at its first id
      {P(5, 9), 12, seeded(9)},
  };
  for (const uint32_t spec : {0u, 2u}) {
    std::vector<Out> outs[2];
    std::vector<std::string> events[2];
    std::vector<std::string> at_first(specs.size());
    for (const bool peek : {false, true}) {
      MockBatchEngine eng(3);
      configure(eng);
      if (pool) eng.pool = 3 * eng.maxlen / eng.blk;
      eng.eos_at[21] = 1;
      eng.eos_at[22] = 1;
      eng.spec = spec;
      eng.peek = peek;
      eng.sample_first = true;
      Scheduler s(eng, opts(4));
      std::vector<Out>& out = outs[peek];
      out.resize(specs.size());
      for (size_t i = 0; i < specs.size(); ++i) {
        SchedRequest r = make(specs[i], out[i], eng);
        auto inner = r.on_token;
        r.on_token = [&, i, inner, peek](uint32_t id) {
          if (peek && out[i].ids.empty()) at_first[i] = eng.events.back();
          return inner(id);
        };
        CHECK(s.submit(std::move(r)) != nullptr);
      }
      drain(s);
      events[peek] = eng.events;
    }
    CHECK(events[0] == events[1]);
    for (size_t i = 0; i < specs.size(); ++i) {
      const Out &a = outs[0][i], &b = outs[1][i];
      CHECK(a.ids == b.ids);
      CHECK_EQ(a.fin.reason, b.fin.reason);
      CHECK_EQ(a.fin.completion_tokens, b.fin.completion_tokens);
      CHECK_EQ(b.done_calls, 1);
      CHECK(!b.token_after_done);
      if (!b.ids.empty()) CHECK(at_first[i].rfind("prefill", 0) == 0);   // sent before the first step
    }
    MockBatchEngine ref(1);
    CHECK(outs[1][2].ids.empty() && outs[1][2].fin.reason == Finish::Stop);
    CHECK(outs[1][3].ids.size() > 1 && outs[1][3].ids[0] == ref.eos);
    CHECK(outs[1][4].ids.size() == 1 && outs[1][4].fin.reason == Finish::Length);
    CHECK(outs[1][5].ids.size() == 1 && outs[1][5].fin.reason == Finish::Stop);
    CHECK_EQ(outs[1][1].ids.at(0), ref.first_draw(specs[1].s));
    CHECK_EQ(outs[1][6].ids.at(0), ref.first_draw(specs[6].s));
  }
  std::printf("first id peeked%s: sent after the prompt's last chunk, the same ids and ends as without, plain and "
              "speculative; a sampled first id drawn at the prompt's end\n", pool ? " (a page pool)" : "");
}

// Plan 31 Task 13: the per-step draft choice without lookup inside MTP (SchedulerOptions::step_k alone) - greedy rows
// of a speculative engine run StepK, batched and alone, and every request gives the ids it gives alone.
void test_step_k_without_lookup() {
  const std::vector<Spec> specs = {{P(1, 10), 30, {}}, {P(2, 6), 20, {}}};
  MockBatchEngine eng(2);
  configure(eng);
  eng.spec = 2;
  SchedulerOptions o = opts(512);
  o.step_k = true;
  Scheduler s(eng, o);
  std::vector<Out> outs(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) CHECK(s.submit(make(specs[i], outs[i], eng)) != nullptr);
  drain(s);
  for (size_t i = 0; i < specs.size(); ++i) same_as_alone("step K without lookup", specs[i], outs[i]);
  std::printf("the per-step draft choice without lookup inside MTP: each request as alone\n");
}

// ---- Spec 31 (plan 31 Task 6): admission by free KV pages, idle sessions evicted to the prefix cache ------------

// The fixed slots' lengths as one pool: a request may grow past any one slot's length; a prompt past the cap is
// refused at submit as before.
void test_slot_lengths_pool() {
  MockBatchEngine eng(2);
  eng.pool = (64 + 1024) / eng.blk;
  configure(eng);
  Scheduler s(eng, opts());
  Out a, b, c, d, e;
  CHECK(s.submit(make({P(71, 100), 8, {}}, a, eng)) != nullptr);
  CHECK(s.submit(make({P(72, 60), 100, {}}, b, eng)) != nullptr);   // no 64-position slot: all 100 ids
  drain(s);
  CHECK_EQ(a.ids.size(), size_t(8));
  CHECK_EQ(b.ids.size(), size_t(100));
  same_as_alone("pool lengths, past a fixed slot's end", {P(72, 60), 100, {}}, b);
  // Two long prompts run side by side: no long slot to wait for.
  CHECK(s.submit(make({P(73, 200), 6, {}}, d, eng)) != nullptr);
  CHECK(s.submit(make({P(74, 300), 6, {}}, e, eng)) != nullptr);
  s.tick();
  CHECK_EQ(s.active(), size_t(2));
  drain(s);
  CHECK(d.fin.slot != e.fin.slot);
  same_as_alone("pool lengths, the second long prompt", {P(74, 300), 6, {}}, e);
  bool threw = false;
  try {
    s.submit(make({P(75, 1024), 5, {}}, c, eng));
  } catch (const std::invalid_argument& error) {
    threw = std::string(error.what()).find("max_model_len is 1024") != std::string::npos;
  }
  CHECK(threw);
  std::printf("pool lengths: a request grows past a fixed slot's end; two long prompts at once; past the cap: 400\n");
}

// A 100k-token request beside two short ones: slot 0 of the fixed layout (46,080 at bf16) refuses it, a pool of the
// same card's 70 pages serves all three at once.
void test_pool_long_beside_short() {
  MockBatchEngine fixed(4, 46080);
  fixed.lens = {46080, 32768, 32768, 32768};
  Scheduler sf(fixed, opts());
  Out refused;
  bool threw = false;
  try {
    sf.submit(make({P(81, 100000), 40, {}}, refused, fixed));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  MockBatchEngine eng(4, 70 * 2048);
  eng.blk = 2048;
  eng.pool = 70;
  Scheduler s(eng, opts());
  const std::vector<Spec> specs = {{P(81, 100000), 40, {}}, {P(82, 3000), 40, {}}, {P(83, 3000), 40, seeded(4)}};
  std::vector<Out> outs(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) CHECK(s.submit(make(specs[i], outs[i], eng)) != nullptr);
  drain(s);
  bool three = false;
  for (const auto& ev : eng.events) three = three || (ev.rfind("step ", 0) == 0 && ev.size() == std::string("step 0 1 2").size());
  CHECK(three);
  for (size_t i = 0; i < specs.size(); ++i) {
    CHECK_EQ(outs[i].fin.reason, Finish::Length);
    same_as_alone("pool, long beside short", specs[i], outs[i], 70 * 2048);
  }
  std::printf("pool: a 100000-id prompt the fixed slots refuse runs beside two short ones (70 pages of 2048)\n");
}

// A request short of pages evicts the least recently used idle session; that session's next turn is a prefix cache
// hit restored from host memory, its ids those of the session served alone. Its entries stay pinned until then: with
// a store just big enough for A's and B's sessions, C's entries push out B's (idle on the card, not pinned), never A's.
void test_pool_evicts_idle() {
  size_t after_ab = 0;
  for (const bool tight : {false, true}) {
    MockBatchEngine eng(3, 8192);
    eng.blk = 256;
    eng.pool = 48;
    configure(eng);
    MallocAlloc alloc;
    server::PrefixSlots cache(eng, tight ? after_ab : size_t(1) << 26, alloc);
    Scheduler s(eng, opts(), &cache);
    const Spec a{P(91, 5000), 20, {}}, b{P(92, 5000), 20, {}}, c{P(93, 6000), 20, {}};
    Out oa, ob, oc;
    CHECK(s.submit(make(a, oa, eng)) != nullptr);
    drain(s);
    CHECK(s.submit(make(b, ob, eng)) != nullptr);
    drain(s);
    if (!tight) after_ab = cache.cache().bytes_used();
    const uint32_t sa = uint32_t(oa.fin.slot), sb = uint32_t(ob.fin.slot);
    CHECK(sa != sb);
    CHECK_EQ(eng.pages_held(sa), 21u);   // prompt + a page of output + a row: 5257 positions
    CHECK_EQ(eng.pages_held(sb), 21u);
    CHECK(s.submit(make(c, oc, eng)) != nullptr);   // 25 pages, 6 free: A's session (the older) leaves the card
    drain(s);
    CHECK(uint32_t(oc.fin.slot) != sa && uint32_t(oc.fin.slot) != sb);
    CHECK_EQ(eng.pages_held(sa), 0u);
    CHECK_EQ(eng.pages_held(sb), 21u);
    same_as_alone("pool eviction, C", c, oc, 8192);
    // A's next turn: its whole conversation so far, then new ids. B's session (now the older) makes room.
    Spec a2{a.prompt, 10, {}};
    a2.prompt.insert(a2.prompt.end(), oa.ids.begin(), oa.ids.end());
    for (uint32_t i = 0; i < 5; ++i) a2.prompt.push_back(7200 + i);
    Out oa2;
    CHECK(s.submit(make(a2, oa2, eng)) != nullptr);
    drain(s);
    CHECK_EQ(cache.last_kind(uint32_t(oa2.fin.slot)), server::PrefixCache::Plan::Restore);
    CHECK_EQ(oa2.fin.cached_tokens, uint32_t(a.prompt.size() + oa.ids.size()));
    CHECK_EQ(eng.pages_held(sb), 0u);
    same_as_alone("pool eviction, A's next turn", a2, oa2, 8192);
    std::printf("pool: an idle session evicted for a new request; its next turn restored %u positions from the "
                "prefix cache%s\n", oa2.fin.cached_tokens,
                tight ? " (a store of exactly A's and B's sessions: A's entries held while C's were stored)" : "");
  }
}

// Every slot holds an idle session, so new requests take a slot whose session they do not continue. Those sessions
// hold no entries in the host store past what the new request continues: with a store of exactly two sessions, X's
// sessions given up for Y's must not push Y's snapshots out, and Y's prompts sent again restore as with room to spare.
void test_pool_given_up_session_holds_nothing() {
  size_t after_x = 0;
  uint32_t roomy = 0;
  for (const bool tight : {false, true}) {
    MockBatchEngine eng(2, 8192);
    eng.blk = 256;
    eng.pool = 64;
    eng.cut = 512;   // prefill calls end at 512 and 1024: a snapshot at each, as the engine's 2048 chunks
    configure(eng);
    MallocAlloc alloc;
    server::PrefixSlots cache(eng, tight ? after_x : size_t(1) << 26, alloc);
    Scheduler s(eng, opts(), &cache);
    auto round = [&](const std::vector<Spec>& specs) {
      std::vector<Out> outs(specs.size());
      for (size_t i = 0; i < specs.size(); ++i) CHECK(s.submit(make(specs[i], outs[i], eng)) != nullptr);
      drain(s);
      return outs;
    };
    round({{P(81, 1100), 20, {}}, {P(82, 1100), 20, {}}});
    if (!tight) after_x = cache.cache().bytes_used();
    const std::vector<Spec> y{{P(83, 1100), 20, {}}, {P(84, 1100), 20, {}}};
    round(y);
    const std::vector<Out> again = round(y);
    for (size_t i = 0; i < y.size(); ++i) {
      CHECK_EQ(cache.last_kind(uint32_t(again[i].fin.slot)), server::PrefixCache::Plan::Restore);
      if (!tight && i == 0) roomy = again[i].fin.cached_tokens;
      CHECK_EQ(again[i].fin.cached_tokens, roomy);
      same_as_alone("pool, a prompt sent again", y[i], again[i], 8192);
    }
  }
  std::printf("pool: sessions given up for new requests hold nothing; prompts sent again restore %u positions with a "
              "store of exactly two sessions\n", roomy);
}

// A request that shares only its first id with an idle long session takes a free slot, not its slot; when the
// requests need its pages it leaves the card as an eviction (its entries held), and its next turn restores it whole.
void test_pool_shared_first_id() {
  MockBatchEngine eng(4, 8192);
  eng.blk = 256;
  eng.pool = 40;
  configure(eng);
  MallocAlloc alloc;
  server::PrefixSlots cache(eng, size_t(1) << 26, alloc);
  Scheduler s(eng, opts(), &cache);
  const Spec l{P(7, 6000), 20, {}};
  Out ol;
  CHECK(s.submit(make(l, ol, eng)) != nullptr);
  drain(s);
  const uint32_t sl = uint32_t(ol.fin.slot);
  std::vector<Spec> shorts;
  for (uint32_t k = 0; k < 3; ++k) {
    Spec x{P(60 + k, 2000), 8, {}};
    x.prompt[0] = l.prompt[0];   // the same first id as the long session
    shorts.push_back(x);
  }
  std::vector<Out> os(3);
  for (uint32_t k = 0; k < 3; ++k) CHECK(s.submit(make(shorts[k], os[k], eng)) != nullptr);
  drain(s);
  // The first is granted while the long session still holds its pages: a free slot, not its slot. (Once it has left
  // the card its slot is as free as any other.)
  CHECK(uint32_t(os[0].fin.slot) != sl);
  for (uint32_t k = 0; k < 3; ++k) same_as_alone("pool, a short request beside an idle long session", shorts[k], os[k], 8192);
  CHECK(eng.pages_held(sl) < 25u);   // 25 + 3 x 9 pages > 40: the long session left the card (a short one may hold its slot)
  Spec l2{l.prompt, 10, {}};
  l2.prompt.insert(l2.prompt.end(), ol.ids.begin(), ol.ids.end());
  for (uint32_t i = 0; i < 5; ++i) l2.prompt.push_back(7300 + i);
  Out ol2;
  CHECK(s.submit(make(l2, ol2, eng)) != nullptr);
  drain(s);
  CHECK_EQ(cache.last_kind(uint32_t(ol2.fin.slot)), server::PrefixCache::Plan::Restore);
  CHECK_EQ(ol2.fin.cached_tokens, uint32_t(l.prompt.size() + ol.ids.size()));
  same_as_alone("pool, the long session's next turn", l2, ol2, 8192);
  std::printf("pool: requests sharing a long idle session's first id took a free slot; the session left the "
              "card for their pages and its next turn restored %u positions\n", ol2.fin.cached_tokens);
}

// Spec 31 §9 (plan 31 Task 8): requests with the same start share read-only pages. B starts as A's prompt does: its
// restore maps A's live pages instead of copying the blocks in, and B's ids are those of B alone. A's next turn
// rewrites its history inside the shared pages - its writes go to pages of its own (the mock refuses a write into a
// shared page) - and B continues unchanged; a third conversation takes A's slot while B still reads the pages; when
// every session ends the pool is whole again.
void test_pool_shared_pages() {
  MockBatchEngine eng(3, 8192);
  eng.blk = 256;
  eng.pool = 40;
  eng.cut = 512;   // chunks end every 512 ids, the cache's snapshots with them (the server's hooked cut)
  configure(eng);
  MallocAlloc alloc;
  server::PrefixSlots cache(eng, size_t(1) << 26, alloc);
  Scheduler s(eng, opts(), &cache);
  const Spec a{P(201, 2000), 20, {}};
  Out oa;
  CHECK(s.submit(make(a, oa, eng)) != nullptr);
  drain(s);
  Spec b{std::vector<uint32_t>(a.prompt.begin(), a.prompt.begin() + 1536), 20, {}};
  for (uint32_t i = 0; i < 500; ++i) b.prompt.push_back(9100 + i);
  Out ob;
  CHECK(s.submit(make(b, ob, eng)) != nullptr);
  drain(s);
  CHECK_EQ(cache.last_kind(uint32_t(ob.fin.slot)), server::PrefixCache::Plan::Restore);
  CHECK_EQ(ob.fin.cached_tokens, 1536u);
  CHECK_EQ(eng.shares, 6u);   // the blocks below the snapshot at 1536 B restarts from
  const uint32_t sa = uint32_t(oa.fin.slot), sb = uint32_t(ob.fin.slot);
  CHECK(sa != sb);
  for (uint32_t i = 0; i < 6; ++i) CHECK_EQ(eng.tab[sb][i], eng.tab[sa][i]);
  same_as_alone("pool, a request on shared pages", b, ob, 8192);
  // A's next turn rewrites its history at 1300, inside the shared pages.
  Spec a2{std::vector<uint32_t>(a.prompt.begin(), a.prompt.begin() + 1300), 12, {}};
  for (uint32_t i = 0; i < 300; ++i) a2.prompt.push_back(9600 + i);
  Out oa2;
  CHECK(s.submit(make(a2, oa2, eng)) != nullptr);
  drain(s);
  same_as_alone("pool, a turn rewriting history inside shared pages", a2, oa2, 8192);
  // A third conversation takes A's slot; B's next turn still reads the shared pages.
  const Spec c{P(202, 3000), 8, {}};
  Out oc;
  CHECK(s.submit(make(c, oc, eng)) != nullptr);
  drain(s);
  Spec b2{b.prompt, 10, {}};
  b2.prompt.insert(b2.prompt.end(), ob.ids.begin(), ob.ids.end());
  for (uint32_t i = 0; i < 5; ++i) b2.prompt.push_back(9900 + i);
  Out ob2;
  CHECK(s.submit(make(b2, ob2, eng)) != nullptr);
  drain(s);
  same_as_alone("pool, B's next turn on the shared pages", b2, ob2, 8192);
  for (uint32_t q = 0; q < 3; ++q) eng.reset(q);
  CHECK_EQ(eng.pages_free(), eng.pool);
  std::printf("pool: a request shared %u pages of an idle session; history rewritten in them went to private pages; "
              "the pool whole after every session\n", 6u);
}

// Every sequence running and no page free: the growing request ends with "length" where its pages end; the other
// finishes unchanged, plain and with the first id peeked. A prompt that fits the cap but not the free pages waits, and is served when pages return.
void test_pool_full() {
  // With the first id peeked too (PERF-91): the reservation covers the first step, so no request ends for want of a
  // page right after its first id went out.
  for (const bool peek : {false, true}) {
    MockBatchEngine eng(2, 4096);
    eng.blk = 256;
    eng.pool = 13;
    eng.peek = peek;
    Scheduler s(eng, opts());
    const Spec a{P(101, 1000), 2000, {}};   // 5 pages reserved (to 1280), then it grows
    const Spec b{P(102, 1700), 340, {}};    // 8 pages, ends at 2040: never grows
    Out oa, ob;
    CHECK(s.submit(make(a, oa, eng)) != nullptr);
    CHECK(s.submit(make(b, ob, eng)) != nullptr);
    drain(s);
    CHECK_EQ(oa.fin.reason, Finish::Length);
    CHECK_EQ(oa.ids.size(), size_t(280));   // position 1280 is on no page
    const Out ra = alone(a, 4096);
    CHECK(std::equal(oa.ids.begin(), oa.ids.end(), ra.ids.begin()));
    same_as_alone("pool full, the other request", b, ob, 4096);
  }
  {
    MockBatchEngine eng(2, 4096);
    eng.blk = 256;
    eng.pool = 13;
    Scheduler s(eng, opts());
    const Spec a{P(103, 2500), 50, {}};   // 11 pages
    const Spec b{P(104, 1000), 10, {}};   // 5 pages, 2 free: waits
    Out oa, ob;
    CHECK(s.submit(make(a, oa, eng)) != nullptr);
    CHECK(s.submit(make(b, ob, eng)) != nullptr);
    s.tick();
    CHECK_EQ(s.active(), size_t(1));
    CHECK_EQ(s.waiting(), size_t(1));
    drain(s);
    CHECK_EQ(ob.fin.reason, Finish::Length);
    CHECK(ob.t.at(0) > oa.t.back());   // served once A's pages came back
    same_as_alone("pool, the request that waited", b, ob, 4096);
  }
  std::printf("pool full: the growing request ends with length, the other unchanged; a request waits for pages\n");
}

// Replica x slots with a pool per card (8 slots, 4 on each of two cards): new requests sent together split evenly
// across the cards - a slot granted in the same tick has taken no pages yet, so its card must not look as free as
// the idle one.
void test_replica_spread() {
  struct TwoCards : MockBatchEngine {
    using MockBatchEngine::MockBatchEngine;
    uint32_t card_of(uint32_t s) override { return s / 4; }
    uint32_t card_pool_pages(uint32_t) override { return pool / 2; }
    uint32_t card_pages_free(uint32_t c) override {
      uint32_t held = 0;
      for (uint32_t s = 4 * c; s < 4 * c + 4; ++s) held += pages_held(s);
      return pool / 2 - held;
    }
  };
  for (const uint32_t n : {2u, 4u}) {
    TwoCards eng(8);
    eng.overlap = true;
    eng.pool = 8 * eng.maxlen / eng.blk;
    Scheduler s(eng, opts());
    std::vector<Out> outs(n);
    std::vector<Spec> specs;
    for (uint32_t i = 0; i < n; ++i) {
      specs.push_back({P(130 + i, 20), 6, {}});
      CHECK(s.submit(make(specs[i], outs[i], eng)) != nullptr);
    }
    drain(s);
    uint32_t on[2] = {0, 0};
    for (uint32_t i = 0; i < n; ++i) {
      ++on[eng.card_of(uint32_t(outs[i].fin.slot))];
      same_as_alone("replica spread", specs[i], outs[i]);
    }
    CHECK_EQ(on[0], n / 2);
    CHECK_EQ(on[1], n / 2);
  }
  std::printf("replica spread: 2 and 4 requests sent together split evenly over two cards of 4 slots\n");
}

// Replica x slots: a request that arrives while the other card runs a chunk starts on the idle card in that tick,
// without asking the busy card for its pages (the real engine asks on the card's thread, so the tick would wait
// behind the chunk and the two prompts would prefill one after the other).
void test_replica_busy_card() {
  struct TwoCards : MockBatchEngine {
    using MockBatchEngine::MockBatchEngine;
    uint32_t asked_busy = 0;
    bool running(uint32_t c) const {
      for (uint32_t s = 4 * c; s < 4 * c + 4; ++s)
        if (const auto it = busy.find(s); it != busy.end() && it->second != 0) return true;
      return false;
    }
    uint32_t card_of(uint32_t s) override { return s / 4; }
    uint32_t card_pool_pages(uint32_t) override { return pool / 2; }
    uint32_t pages_held(uint32_t s) override {
      asked_busy += running(s / 4);
      return MockBatchEngine::pages_held(s);
    }
    uint32_t card_pages_free(uint32_t c) override {
      asked_busy += running(c);
      uint32_t held = 0;
      for (uint32_t s = 4 * c; s < 4 * c + 4; ++s) held += MockBatchEngine::pages_held(s);
      return pool / 2 - held;
    }
  };
  TwoCards eng(8);
  eng.overlap = true;
  eng.prefill_polls = 1000;
  eng.pool = 8 * eng.maxlen / eng.blk;
  Scheduler s(eng, opts());
  const Spec a{P(140, 300), 6, {}}, b{P(150, 300), 6, {}};
  Out oa, ob;
  CHECK(s.submit(make(a, oa, eng)) != nullptr);
  s.tick();   // A's chunk starts on its card and runs
  CHECK(s.submit(make(b, ob, eng)) != nullptr);
  s.tick();
  CHECK_EQ(s.waiting(), size_t(0));   // B granted while A's chunk runs
  CHECK_EQ(eng.asked_busy, 0u);
  eng.prefill_polls = 0;
  drain(s);
  CHECK(eng.card_of(uint32_t(oa.fin.slot)) != eng.card_of(uint32_t(ob.fin.slot)));
  same_as_alone("replica busy card", a, oa);
  same_as_alone("replica busy card", b, ob);
  std::printf("replica busy card: a request arriving during the other card's chunk starts at once, the busy card "
              "not asked\n");
}

int main() {
  test_batch_independence();
  test_per_row_stops();
  test_disconnect();
  test_queue_and_plan_at_grant();
  test_batched_admission();
  test_fairness();
  test_eos_rules_and_errors();
  test_kv_form();
  test_slot_lengths();
  test_split_last();
  test_lookup_per_request();
  test_slot_failure();
  test_prefill_cut_and_overlap();
  test_parallel_prefill();
  test_callback_throws();
  test_first_id_peek();
  test_first_id_peek(true);
  test_step_k_without_lookup();
  test_batched_admission(true);
  test_slot_lengths_pool();
  test_pool_long_beside_short();
  test_pool_evicts_idle();
  test_pool_shared_first_id();
  test_pool_given_up_session_holds_nothing();
  test_pool_shared_pages();
  test_pool_full();
  test_replica_spread();
  test_replica_busy_card();
  std::printf("scheduler_test: all cases passed\n");
  return 0;
}
