// Spec 16b P4 (plan 16b Review Focus 4): a hand-off that never arrives is a clear error
// within its bound, never a hang - and the engine recovers after reset().
//
//   copy  device 1's list waits on the cross-device event device 0 never signals
//         (drop_next_handoff: device 0's list is not submitted). The fence wait gives up
//         after the timeout, the host signals the event so device 1's queue drains, and the
//         step throws naming the event. The next step throws "reset() first".
//   peer  the same lost step: pp_recv gives up after its spin bound and records the flag it
//         read; the step throws naming pp_recv.
// Each: the throw arrives within the bound plus slack; reset(); the prompt and 16 greedy ids
// again equal a clean run's (a failed step leaves nothing behind that reset() does not clear).
// The other two P4 refusals: a missing second card is `cli_reject_pipeline_one_gpu`
// (ZE_AFFINITY_MASK=0, the shipped binary); no peer access cannot be produced on the box -
// PipelineEngine and b70-decode refuse it before the load (zeDeviceCanAccessPeer).
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
constexpr uint32_t kMaxLen = 4096;

std::string step_error(runtime::PipelineEngine& e, double* seconds) {
  const auto t0 = std::chrono::steady_clock::now();
  try {
    e.generate(1);
  } catch (const std::runtime_error& x) {
    *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return x.what();
  }
  *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return "";
}
}  // namespace

int main(int argc, char** argv) {
  const std::string checkpoint =
      argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const uint32_t gpus = l0::Context::gpu_count();
  if (gpus < 2) {
    std::printf("SKIP: --pipeline 2 needs two GPUs, Level Zero shows %u\n", gpus);
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1)) {
    std::printf("SKIP: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer)\n");
    return 77;
  }
  const std::vector<uint32_t> prompt = golden::read_ids("tests/golden/prompts/prose.ids");

  for (const runtime::PpHandoff h : {runtime::PpHandoff::Copy, runtime::PpHandoff::Peer}) {
    loader::LoadedModel full = loader::load(d0, checkpoint, kMaxLen);
    const runtime::PpWeights w = runtime::pp_weights(full);
    const uint32_t split = runtime::pp_auto_split(*full.desc, w, kMaxLen);
    runtime::PipelineOptions opt;
    opt.handoff = h;
    // copy: the host's fence bound is the only one. peer: the device's spin bound must give
    // up first - 2^20 system-scope loads (unmeasured: 16a would have; ~1 s at 1 us a load)
    // against a 30 s fence bound, and still far above a clean step's wait for device 0.
    opt.timeout_ms = h == runtime::PpHandoff::Copy ? 3000 : 30000;
    opt.spin_limit = 1u << 20;
    runtime::PipelineEngine e(d0, d1, runtime::place_stages(d0, d1, std::move(full), split),
                              kMaxLen, opt);
    const char* name = runtime::pp_handoff_name(h);

    e.ingest(prompt);
    const std::vector<uint32_t> clean = e.generate(16);

    e.reset();
    e.ingest(prompt);
    e.drop_next_handoff();
    double s = 0;
    const std::string err = step_error(e, &s);
    std::printf("%s, lost hand-off: threw after %.2f s: %s\n", name, s, err.c_str());
    CHECK(!err.empty());
    CHECK(s < 3 * opt.timeout_ms / 1000.0 + 5.0);   // bounded: two fence waits at most, + slack
    CHECK(err.find(h == runtime::PpHandoff::Copy ? "cross-device event was not signalled"
                                                 : "pp_recv gave up") != std::string::npos);
    // Marked: the next step refuses until reset().
    const std::string again = step_error(e, &s);
    CHECK(again.find("reset() first") != std::string::npos);

    e.reset();
    e.ingest(prompt);
    CHECK(e.generate(16) == clean);
    std::printf("%s: recovered after reset(), 16 ids equal the clean run's\n", name);
  }
  std::printf("pp_fail_test: OK\n");
  return 0;
}
