// Spec 17b T4 (plan 17b's review, Task 2): a crossing that never arrives is a clear error within the bounds, never a
// hang, and reset() recovers.
//
//   for each card r: a clean step, then drop_next(r) - the next step submits only the other card's list, so the
//   other card's first fold waits for card r's part that never comes. The step must throw within its bound (the
//   in-list spin is bounded and every later wait on that card skips its spin), name the timeout, and leave the engine
//   failed (the next step throws without submitting); reset() then runs the clean session again and it is bitwise the
//   first clean run (the ids of a short generation).
//
// argv: [1] the checkpoint. Exits 77 (SKIP) with fewer than two GPUs or without peer access both ways.
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "runtime/tp_engine.h"

int main(int argc, char** argv) {
  const std::string ckpt = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  constexpr uint32_t kMaxLen = 4096;
  const uint32_t gpus = l0::Context::gpu_count();
  if (gpus < 2) {
    std::printf("SKIP: --parallel tp needs two GPUs, Level Zero shows %u\n", gpus);
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0)) {
    std::printf("SKIP: the cards cannot access each other's memory (zeDeviceCanAccessPeer)\n");
    return 77;
  }
  const model::TpCrossing c = runtime::kTpDefaultCrossing;
  runtime::TpOptions opt;
  opt.spin_limit = 1u << 18;   // a short bound: the test is about the failure, not the wait
  opt.timeout_ms = 20000;
  runtime::TpEngine e(d0, d1,
                      {loader::load_tp(d0, ckpt, {0, c}, kMaxLen, loader::LmHeadForm::Int8),
                       loader::load_tp(d1, ckpt, {1, c}, kMaxLen, loader::LmHeadForm::Int8)},
                      kMaxLen, opt);
  const std::vector<uint32_t> prompt = golden::read_ids("tests/golden/prompts/prose.ids");
  e.ingest(prompt);
  const std::vector<uint32_t> clean = e.generate(8);

  for (uint32_t r = 0; r < 2; ++r) {
    e.reset();
    e.ingest(prompt);
    e.drop_next(r);
    const auto t0 = std::chrono::steady_clock::now();
    std::string msg;
    try {
      e.generate(1);
    } catch (const std::exception& ex) {
      msg = ex.what();
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("card %u dropped: threw after %.1f ms: %s\n", r, ms, msg.c_str());
    CHECK(!msg.empty());
    CHECK(msg.find("timed out") != std::string::npos);
    CHECK(msg.find("card " + std::to_string(1 - r) + "'s wait for card " + std::to_string(r)) != std::string::npos);
    CHECK(ms < double(opt.timeout_ms));
    // Failed: the next step throws without running.
    bool threw = false;
    try {
      e.generate(1);
    } catch (const std::exception& ex) {
      threw = std::string(ex.what()).find("reset() first") != std::string::npos;
    }
    CHECK(threw);
    // reset() recovers: the clean session again, bitwise.
    e.reset();
    e.ingest(prompt);
    CHECK(e.generate(8) == clean);
    std::printf("card %u dropped: reset() recovered, the clean run's 8 ids again\n", r);
  }
  std::printf("tp_fail_test: OK\n");
  return 0;
}
