// Spec 16c (plan 16c Review Focus 2, spec 16 §4 P4): the two-card prefill's back-pressure
// and failures, on the cards.
//
//   back-pressure  device 1 slowed artificially (PipelineOptions::prefill_hold_ms: each of
//                  its chunks first waits on an event a host thread signals 150 ms after the
//                  chunk is appended), 9000 ids in 5 chunks of 2048: device 0 must wait for its
//                  two landing slots instead of overwriting one device 1 has not taken - the
//                  result is bitwise the unslowed run's (an overwritten slot would hand device
//                  1 another chunk's rows), and the wall time shows the hold (>= 5 x 150 ms).
//   lost hand-off  drop_next_handoff(): device 0 never hands chunk 0 over. Device 1's list
//                  waits on the ready event without a bound (a list's wait); the host's wait
//                  gives up after prefill_timeout_ms (here 5 s), releases device 1, and the
//                  prefill throws naming the chunk and the event, within the bound plus slack.
//                  The next prefill throws "reset() first"; after reset() the prefill is
//                  bitwise the clean run's again.
// Each under copy and peer. The other P4 refusals (a missing card, no peer access) are 16b's.
//
// argv: [1] the checkpoint. Exits 77 (SKIP) with fewer than two GPUs or without peer access.
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_plan.h"

namespace {
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kN = 9000;
constexpr uint32_t kHoldMs = 150;
constexpr uint32_t kBoundMs = 5000;

std::vector<uint32_t> repeat_to(const std::vector<uint32_t>& src, size_t n) {
  std::vector<uint32_t> out;
  while (out.size() < n) out.insert(out.end(), src.begin(), src.end());
  out.resize(n);
  return out;
}

// What a prefill leaves: the state, the KV, the first id and 8 decoded after it.
struct Result {
  std::vector<uint8_t> state, kv;
  std::vector<uint32_t> ids;
  bool operator==(const Result& o) const { return state == o.state && kv == o.kv && ids == o.ids; }
};
Result result(runtime::PipelineEngine& e) {
  Result r;
  r.state.resize(e.state_bytes());
  e.save_state(r.state.data());
  r.kv.resize(e.kv_bytes(e.pos()));
  e.save_kv(0, e.pos(), r.kv.data());
  r.ids = e.generate(8);
  return r;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string checkpoint = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const uint32_t gpus = l0::Context::gpu_count();
  if (gpus < 2) {
    std::printf("SKIP: --pp 2 needs two GPUs, Level Zero shows %u\n", gpus);
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1)) {
    std::printf("SKIP: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer)\n");
    return 77;
  }
  const std::vector<uint32_t> prompt = repeat_to(golden::read_ids("tests/golden/prompts/long32k.ids"), kN);
  using Clock = std::chrono::steady_clock;

  for (const runtime::PpHandoff h : {runtime::PpHandoff::Copy, runtime::PpHandoff::Peer}) {
    loader::LoadedModel full = loader::load(d0, checkpoint, kMaxLen);
    const uint32_t split = runtime::pp_auto_split(*full.desc, runtime::pp_weights(full), kMaxLen);
    runtime::PipelineOptions opt;
    opt.handoff = h;
    opt.prefill_timeout_ms = kBoundMs;
    runtime::PipelineEngine e(d0, d1, runtime::place_stages(d0, d1, std::move(full), split), kMaxLen, opt);
    const char* name = runtime::pp_handoff_name(h);
    e.prepare_prefill();

    e.prefill(prompt);
    const double clean_ms = e.last_prefill().wall_ms;
    const Result clean = result(e);

    // Back-pressure: device 1 held back before every chunk; device 0 waits for its slots.
    e.reset();
    e.set_prefill_hold_ms(kHoldMs);
    e.prefill(prompt);
    e.set_prefill_hold_ms(0);
    const runtime::PipelineEngine::PrefillStats st = e.last_prefill();
    std::printf("%s, device 1 held %u ms a chunk: %u chunks in %.1f ms (clean %.1f ms); busy %.1f / "
                "%.1f ms\n", name, kHoldMs, st.chunks, st.wall_ms, clean_ms, st.busy_ms[0], st.busy_ms[1]);
    CHECK(st.wall_ms >= double(st.chunks) * kHoldMs);
    CHECK(result(e) == clean);
    std::printf("%s: bitwise the clean run's - no slot was overwritten before device 1 took it\n", name);

    // A lost hand-off.
    e.reset();
    e.drop_next_handoff();
    const Clock::time_point t0 = Clock::now();
    std::string err;
    try {
      e.prefill(prompt);
    } catch (const std::runtime_error& x) {
      err = x.what();
    }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("%s, lost hand-off: threw after %.2f s: %s\n", name, s, err.c_str());
    CHECK(!err.empty());
    CHECK(err.find("chunk 0") != std::string::npos);
    CHECK(err.find("rows never arrived") != std::string::npos);
    CHECK(s < 3.0 * kBoundMs / 1000.0 + 10.0);   // the wait, the drain, + slack
    std::string again;
    try {
      e.prefill(prompt);
    } catch (const std::runtime_error& x) {
      again = x.what();
    }
    CHECK(again.find("reset() first") != std::string::npos);
    e.reset();
    e.prefill(prompt);
    CHECK(result(e) == clean);
    std::printf("%s: recovered after reset(), bitwise the clean run's\n", name);
  }
  std::printf("pp_prefill_fail_test: OK\n");
  return 0;
}
