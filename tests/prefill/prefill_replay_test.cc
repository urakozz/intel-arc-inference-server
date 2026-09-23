// Whole-chunk replay must preserve every persistent byte and the last logits,
// including changing inputs, positions, ragged tails and reset with dirty scratch.
// --bench skips readbacks and measures first-use + three warm requests per mode.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "loader/loader.h"
#include "runtime/engine.h"
#include "runtime/prefill/gdn.h"
#include "runtime/prefill/step.h"

namespace {
constexpr uint32_t kMaxLen = 16384;
using Snapshot = std::vector<std::vector<uint8_t>>;
Snapshot snapshot(runtime::Engine& engine, l0::CmdList& copy) {
  Snapshot result;
  auto grab = [&](const l0::Mem& mem) {
    result.emplace_back(mem.size());
    copy.copy(result.back().data(), mem.ptr(), mem.size());
  };
  grab(engine.buffers().gdn_state);
  grab(engine.buffers().conv_ring);
  grab(engine.buffers().kv_k);
  grab(engine.buffers().kv_v);
  grab(engine.buffers().control);
  grab(engine.prefill_scratch()->logits);
  return result;
}
void equal(const Snapshot& expected, const Snapshot& actual) {
  const char* names[] = {"gdn_state", "conv_ring", "kv_k", "kv_v", "control", "logits"};
  CHECK_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) std::fprintf(stderr, "replay mismatch: %s\n", names[i]);
    CHECK(expected[i] == actual[i]);
  }
}
}

int main(int argc, char** argv) {
  const std::string checkpoint = argc > 1 ? argv[1] :
      "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const bool bench = argc > 2 && std::string(argv[2]) == "--bench";
  l0::Context context(0);
  auto model = loader::load(context, checkpoint, kMaxLen);
  runtime::Engine engine(context, std::move(model), kMaxLen);
  engine.set_prefill_backend(runtime::PrefillBackend::L0);
  // The scan selector's DISPATCH PROOF (parity-program design §11). A gate run
  // under `B70_PREFILL_GDN_SCAN` is not evidence unless it says which kernel it
  // launched: the 2026-09-21 "green" split record was executing `pf_gdn_scan`.
  {
    const char* const sel = std::getenv("B70_PREFILL_GDN_SCAN");
    std::printf("gdn scan selector: B70_PREFILL_GDN_SCAN=%s -> entry %s\n",
                sel && *sel ? sel : "(unset)", runtime::prefill::gdn_scan_entry_name());
  }
  l0::CmdList copy = l0::CmdList::immediate(context);
  const auto seed = golden::read_ids("tests/golden/prompts/prose.ids");
  std::vector<uint32_t> ids(bench ? 4096 : 4097);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = seed[i % seed.size()];
  if (bench) {
    uint32_t first_token = 0;
    for (bool replay : {false, true}) {
      engine.set_prefill_replay(replay);
      std::vector<double> warm;
      for (int run = 0; run < 4; ++run) {
        engine.reset();
        const auto count = engine.prefill_launches();
        const auto start = std::chrono::steady_clock::now();
        engine.prefill(ids, 2048);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        const auto token = engine.buffers().control.as<runtime::Control>()->cur_token[0];
        if (!replay && run == 0) first_token = token;
        CHECK_EQ(token, first_token);
        CHECK_EQ(engine.pos(), 4096u);
        CHECK_EQ(engine.prefill_launches() - count,
                 2 * runtime::prefill::step_chunk_launches(runtime::PrefillBackend::L0, 2048) + 5);
        std::printf("mode=%s run=%d wall_ms=%.3f tokens_per_second=%.3f first_token=%u\n",
                    replay ? "recorded" : "immediate", run, ms, 4096000.0 / ms, token);
        std::fflush(stdout);
        if (run) warm.push_back(ms);
      }
      std::sort(warm.begin(), warm.end());
      std::printf("mode=%s warm_median_ms=%.3f tokens_per_second=%.3f\n",
                  replay ? "recorded" : "immediate", warm[1], 4096000.0 / warm[1]);
    }
  } else {
    struct Case { std::vector<uint32_t> ids; uint32_t chunk; bool incremental; };
    auto changed = seed;
    std::reverse(changed.begin(), changed.end());
    const std::vector<Case> cases = {{seed, 16, false}, {changed, 16, false},
                                   {seed, 16, true}, {ids, 2048, false},
                                   {seed, 17, false}, {seed, 18, false},
                                   {seed, 19, false}, {seed, 16, false}};
    for (const auto& test : cases) {
      auto run = [&] {
        if (test.incremental) {
          engine.prefill(std::vector<uint32_t>(test.ids.begin(), test.ids.begin() + 16), 16);
          engine.prefill(std::vector<uint32_t>(test.ids.begin() + 16, test.ids.end()), 16);
        } else engine.prefill(test.ids, test.chunk);
        CHECK_EQ(engine.pos(), uint32_t(test.ids.size()));
      };
      engine.set_prefill_replay(false);
      engine.reset();
      run();
      const auto expected = snapshot(engine, copy);
      engine.set_prefill_replay(true);
      for (int repeat = 0; repeat < 2; ++repeat) {
        engine.reset();
        run();
        equal(expected, snapshot(engine, copy));
      }
      std::printf("replay exact: ids=%zu chunk=%u incremental=%d\n",
                  test.ids.size(), test.chunk, test.incremental);
      std::fflush(stdout);
    }
  }
  CHECK(!engine.prefill_sycl_side_created());
  // The other half of the dispatch proof: the entry a scan launch was really
  // built with, read back after the walk rather than before it.
  CHECK(runtime::prefill::gdn_scan_launched_entry() != nullptr);
  CHECK_EQ(std::string(runtime::prefill::gdn_scan_launched_entry()),
           std::string(runtime::prefill::gdn_scan_entry_name()));
  std::printf("gdn scan entry LAUNCHED: %s\n", runtime::prefill::gdn_scan_launched_entry());
  std::puts("prefill_replay_test OK");
}
