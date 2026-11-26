// Plan 41h (both cards): where Inkling's `--parallel pp` plain decode step spends its time over one card's.
//   probe_ink_pp_step SNAPSHOT [depth=4096] [steps=200] [one]
// After a `depth`-id prompt (synthetic ids), drafter off, bf16 cache, each a mean over `steps` replays:
//   one card        card 0's own engine: append()
//   pp              PpEngine::append() (both lists, the hand-off event)
//   card 0 alone    card 0's list, nothing waits on its event
//   card 1 alone    card 1's list with the hand-off event signalled by the host first (no wait)
//   pp, host side   the pair as PpEngine::run submits it, with the host time spent in each execute and fence wait
//   one card, list  card 0's own target list replayed alone, the host touching no Control
//   pp + Controls   the pair as above, the host also writing both cards' Control blocks (shared memory) and reading
//                   card 1's id, as PpEngine::verify does
// `one`: the one-card rows only (one card is enough). The lists replay at one position (the caches' row is
// rewritten); the timings only, never the ids.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <vector>

#include "runtime/inkling2/inkling2_pp_engine.h"

namespace runtime::inkling2 {
struct PpProbeAccess {
  static l0::CmdList& list(PpEngine& pp, uint32_t card) { return *pp.target_[1][0][card][0]; }   // append's (candidate head)
  static l0::SyncEvent& handoff(PpEngine& pp) { return *pp.handoff_; }
  static l0::Fence& fence(PpEngine& pp, uint32_t card) { return *pp.fences_[card]; }
  static l0::Queue& queue(PpEngine& pp, uint32_t card) { return pp.cards_[card]->queue_; }
  static Control* control(PpEngine& pp, uint32_t card) { return pp.cards_[card]->ctrl_; }
  static l0::CmdList& list(Engine& e) { return *e.plain_; }
  static l0::Queue& queue(Engine& e) { return e.queue_; }
  static Control* control(Engine& e) { return e.ctrl_; }
};
}  // namespace runtime::inkling2

namespace {
namespace ink = runtime::inkling2;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

std::vector<uint32_t> prompt(uint32_t n) {
  std::vector<uint32_t> ids(n);
  for (uint32_t i = 0; i < n; ++i) ids[i] = 1000 + (i * 7919u) % 100000u;
  return ids;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: probe_ink_pp_step SNAPSHOT [depth] [steps]\n");
    return 2;
  }
  const uint32_t depth = argc > 2 ? uint32_t(std::atoi(argv[2])) : 4096, steps = argc > 3 ? uint32_t(std::atoi(argv[3])) : 200;
  const uint32_t max_len = (depth + 2 * steps + 4096) / 4096 * 4096, chunk = 2048;
  const bool one_only = argc > 4 && std::string_view(argv[4]) == "one";
  if (!one_only && l0::Context::gpu_count() < 2) {
    std::fprintf(stderr, "probe_ink_pp_step: needs both cards\n");
    return 77;
  }
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  l0::Context ctx0(0);
  const auto ids = prompt(depth);
  const auto time_appends = [&](auto& engine) {
    uint32_t next = engine.prefill(ids, chunk);
    for (uint32_t i = 0; i < 20; ++i) next = engine.append(next);
    const auto t0 = Clock::now();
    for (uint32_t i = 0; i < steps; ++i) next = engine.append(next);
    return ms(t0, Clock::now()) / steps;
  };
  using A = ink::PpProbeAccess;
  {
    ink::Engine one(ctx0, checkpoint, max_len, false, chunk);
    std::printf("one card      %.3f ms a step (depth %u)\n", time_appends(one), depth);
    double t = 0, tc = 0;
    for (uint32_t i = 0; i < steps; ++i) {
      auto a = Clock::now();
      A::queue(one).execute(A::list(one));
      A::queue(one).synchronize();
      t += ms(a, Clock::now());
      a = Clock::now();
      auto* c = A::control(one);
      c->cur_token[0] = 1000 + i;
      c->pos = c->pos;
      c->n_active = 1;
      A::queue(one).execute(A::list(one));
      A::queue(one).synchronize();
      volatile uint32_t id = c->out_token[0];
      (void)id;
      tc += ms(a, Clock::now());
    }
    std::printf("one card, list %.3f ms; with the host's Control writes and id read %.3f ms\n", t / steps, tc / steps);
  }
  if (one_only) return 0;
  l0::Context ctx1(ctx0, 1);
  ink::PpEngine pp(ctx0, ctx1, checkpoint, max_len, false, chunk);
  std::printf("pp            %.3f ms a step\n", time_appends(pp));
  auto& event = A::handoff(pp);
  const uint64_t bound = 10'000'000'000ull;
  double t = 0;
  for (uint32_t i = 0; i < steps; ++i) {
    event.host_reset();
    const auto a = Clock::now();
    A::queue(pp, 0).execute(A::list(pp, 0), &A::fence(pp, 0));
    A::fence(pp, 0).wait_for(bound);
    t += ms(a, Clock::now());
  }
  std::printf("card 0 alone  %.3f ms\n", t / steps);
  t = 0;
  for (uint32_t i = 0; i < steps; ++i) {
    event.host_signal();
    const auto a = Clock::now();
    A::queue(pp, 1).execute(A::list(pp, 1), &A::fence(pp, 1));
    A::fence(pp, 1).wait_for(bound);
    t += ms(a, Clock::now());
  }
  std::printf("card 1 alone  %.3f ms\n", t / steps);
  double exec0 = 0, exec1 = 0, wait1 = 0, wait0 = 0, total = 0;
  for (uint32_t i = 0; i < steps; ++i) {
    event.host_reset();
    const auto a = Clock::now();
    A::queue(pp, 0).execute(A::list(pp, 0), &A::fence(pp, 0));
    const auto b = Clock::now();
    A::queue(pp, 1).execute(A::list(pp, 1), &A::fence(pp, 1));
    const auto c = Clock::now();
    A::fence(pp, 1).wait_for(bound);
    const auto d = Clock::now();
    A::fence(pp, 0).wait_for(bound);
    const auto e = Clock::now();
    exec0 += ms(a, b), exec1 += ms(b, c), wait1 += ms(c, d), wait0 += ms(d, e), total += ms(a, e);
  }
  std::printf("pp, host side %.3f ms: execute card 0 %.3f, execute card 1 %.3f, card 1's fence %.3f, card 0's %.3f\n",
              total / steps, exec0 / steps, exec1 / steps, wait1 / steps, wait0 / steps);
  total = 0;
  for (uint32_t i = 0; i < steps; ++i) {
    const auto a = Clock::now();
    for (uint32_t c = 0; c < 2; ++c) {
      auto* ctl = A::control(pp, c);
      ctl->cur_token[0] = 1000 + i;
      ctl->pos = ctl->pos;
      ctl->n_active = 1;
    }
    event.host_reset();
    A::queue(pp, 0).execute(A::list(pp, 0), &A::fence(pp, 0));
    A::queue(pp, 1).execute(A::list(pp, 1), &A::fence(pp, 1));
    A::fence(pp, 1).wait_for(bound);
    A::fence(pp, 0).wait_for(bound);
    volatile uint32_t id = A::control(pp, 1)->out_token[0];
    (void)id;
    total += ms(a, Clock::now());
  }
  std::printf("pp + Controls %.3f ms\n", total / steps);
  return 0;
}
