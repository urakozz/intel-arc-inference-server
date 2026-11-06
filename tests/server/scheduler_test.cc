// Spec 13 B4 (plan 13b Task 1, Review Focus 1-5): the batching scheduler against a mock
// batched engine (tests/server/mock.h, MockBatchEngine), host only. Every request's ids
// must equal the same request served alone; the cases cover streaming order, per-row
// stops, disconnects, per-request seeds, the bounded queue, the prefix plan taken at the
// slot grant, both admission policies, and prefill fairness in mock ticks.
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

// Spec 13 §8: batched admission holds a burst, then prefills it in one call.
void test_batched_admission() {
  double now = 0;
  MockBatchEngine eng(4);
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
  CHECK_EQ(eng.events[before + 4], std::string("prefill 0:4 1:4 2:4 3:4"));
  drain(s);
  for (int i = 0; i < 7; ++i) CHECK_EQ(x[i].fin.reason, Finish::Length);
  same_as_alone("batched admission", {P(52, 9), 5, {}}, x[1]);
  std::printf("batched admission: a 3-request burst held 5 ms then one prefill call\n");
}

// Review Focus 3 / B4 fairness: a running request's worst inter-token gap (mock ticks: a
// decode step costs 1, prefill 32 ids per tick) while a 32k prompt prefills.
double worst_gap(Admission a, uint32_t chunk, uint32_t* ttft_ticks) {
  MockBatchEngine eng(4, 40000);
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
  MockBatchEngine eng(2, 4096);
  configure(eng);
  eng.cut = 8;
  eng.overlap = true;
  Scheduler s(eng, opts());
  Out a, b;
  CHECK(s.submit(make({P(111, 13), 200, {}}, a, eng)) != nullptr);   // 8 + 5, nothing else decoding
  while (a.ids.size() < 3) s.tick();
  CHECK(s.submit(make({P(112, 30), 4, {}}, b, eng)) != nullptr);     // 8, 8, 8, 6 beside A's steps
  while (b.done_calls == 0) s.tick();
  std::vector<std::string> pf;
  for (const auto& e : eng.events)
    if (e.rfind("prefill", 0) == 0) pf.push_back(e);
  const std::vector<std::string> want = {"prefill 0:8", "prefill 0:5", "prefill 1:8", "prefill 1:8", "prefill 1:8",
                                         "prefill 1:6"};
  CHECK(pf == want);
  // Every one of B's chunks: begin, its prefill, A's step, end - in one tick.
  size_t overlapped = 0;
  for (size_t i = 0; i + 3 < eng.events.size(); ++i)
    if (eng.events[i] == "begin" && eng.events[i + 1].rfind("prefill 1:", 0) == 0 && eng.events[i + 2] == "step 0" &&
        eng.events[i + 3] == "end")
      ++overlapped;
  CHECK_EQ(overlapped, size_t(4));
  drain(s);
  same_as_alone("cut and overlap, A", {P(111, 13), 200, {}}, a, 4096);
  same_as_alone("cut and overlap, B", {P(112, 30), 4, {}}, b, 4096);
  std::printf("prefill cut and overlap: chunks 8/8/8/6 at the engine's cut, each beside the other slot's step\n");
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
  std::printf("scheduler_test: all cases passed\n");
  return 0;
}
