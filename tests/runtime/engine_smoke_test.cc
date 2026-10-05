// engine_smoke_test - the Engine wrapper around the captured decode list
// (spec §8.4/§8.5, plan 3 Task 7). Task 6's replay_determinism_test proves the
// *list* is a pure function of the state it reads, driving the raw loop by
// hand. This test proves the class that now owns that loop does not lose the
// property while adding ingestion, greedy generation, timing and reset.
//
// Phase 1, debug_resid off (the shape everything but the golden gate runs):
//   1. **It generates.** Ingest a fixed 16-id prompt, generate 8 with a
//      counting callback: 8 ids back, every one below kVocabUsed (an id the
//      tokenizer cannot spell is an engine bug - argmax masks the tail), the
//      callback fired exactly 8 times with exactly those ids, and
//      last_tok_per_s() positive.
//   2. **reset() is a fresh process.** Zero the persistent group, re-ingest
//      the same prompt, generate 8 again: identical ids. Note what is NOT
//      zeroed - scratch keeps run 1's bytes, so equality here also says no
//      step read scratch it had not written. That is deliberately the same
//      coverage run C of the determinism test buys, at the Engine's API.
//   3. **The wall is the Engine's to enforce** (Task 5/7 ruling): with `pos`
//      poked to max_len, a step must throw rather than replay past the KV
//      cache and the RoPE table.
//   4. **read_debug_resid() refuses** when the tap was never captured.
//
// Phase 2, debug_resid on - Task 8's gate builds on this accessor, so it is
// exercised here rather than discovered there. A second load (the first
// Engine owns its LoadedModel, and two 18 GB copies do not fit on the card)
// is the price; it also prices the tap's 64 device copies per token.
//
// It does NOT judge output quality - that is Task 8's golden gate, against the
// oracle. Nor does it re-document what tap[L] *means*: runtime/capture.h does.
//
// Label `checkpoint`: needs the real 19 GB checkpoint and a B70.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
using model::Qwen35;

// The same 16 ids as tests/runtime/replay_determinism_test.cc - the first 16
// of tests/golden/prompts/prose.txt under the checkpoint's own tokenizer. Kept
// literal for the same reason: Task 8 is what commits the .ids files.
constexpr uint32_t kPrompt[] = {760, 72103, 506, 37119, 557, 11012, 3213, 310,
                                6512, 279, 61789, 272, 1072, 2272, 279, 197616};
constexpr uint32_t kGen = 8;
constexpr uint32_t kMaxLen = 16384;
// The tap as Engine::read_debug_resid() hands it back: bf16 [64][M=1][5120].
// Spec 14: the layer count is the loaded model's (set before the debug engine is built).
uint32_t kLayers = 0;
uint32_t kHidden = 0;   // the loaded descriptor's (spec 15b), set beside kLayers
size_t kTapElems = 0;

std::vector<uint32_t> prompt() {
  return std::vector<uint32_t>(kPrompt, kPrompt + sizeof(kPrompt) / sizeof(kPrompt[0]));
}
// bf16 carries fp32's exponent field: all ones is Inf or NaN either way.
bool bf16_finite(uint16_t w) { return (w & 0x7F80u) != 0x7F80u; }

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  l0::Context ctx(0);
  std::vector<uint32_t> a;  // phase 1's generated ids, phase 2 must reproduce them
  double plain_ms_per_token = 0.0;

  // --- phase 1: debug_resid off ---------------------------------------------
  {
    loader::LoadedModel m = loader::load(ctx, arg, kMaxLen);
    const auto t_build = std::chrono::steady_clock::now();
    runtime::Engine eng(ctx, std::move(m), kMaxLen);
    std::printf("engine: %zu kernels, %zu modules, max_len %u, captured+reset in %.1f ms\n",
                eng.step().kernel_count, eng.step().modules.size(), eng.max_len(),
                ms_since(t_build));
    CHECK_EQ(eng.step().kernel_count, size_t(774));   // 645 + lever L1's 129
    CHECK_EQ(eng.pos(), uint32_t(0));      // the constructor reset it
    CHECK_EQ(eng.last_tok_per_s(), 0.0);   // nothing generated yet
    CHECK(!eng.debug_resid());

    eng.ingest(prompt());
    CHECK_EQ(eng.pos(), uint32_t(prompt().size()));

    std::vector<uint32_t> seen;
    a = eng.generate(kGen, [&](uint32_t id) { seen.push_back(id); });

    CHECK_EQ(a.size(), size_t(kGen));
    CHECK_EQ(seen.size(), size_t(kGen));                          // the callback fired 8x
    for (uint32_t i = 0; i < kGen; ++i) CHECK_EQ(seen[i], a[i]);  // ...with the returned ids
    for (uint32_t id : a) CHECK(id < Qwen35::kVocabUsed);
    CHECK_EQ(eng.pos(), uint32_t(prompt().size() + kGen));
    CHECK(eng.last_tok_per_s() > 0.0);
    // The two halves of the wall clock Task 9's breakdown reports.
    CHECK(eng.last_gen_ms() > 0.0);
    CHECK(eng.last_fence_ms() > 0.0);
    CHECK(eng.last_fence_ms() <= eng.last_gen_ms());
    plain_ms_per_token = eng.last_gen_ms() / kGen;

    std::printf("run 1 ids:");
    for (uint32_t id : a) std::printf(" %u", id);
    std::printf("   (%.2f tok/s, %.2f ms/token, %.1f%% of it inside the fence)\n",
                eng.last_tok_per_s(), plain_ms_per_token,
                100.0 * eng.last_fence_ms() / eng.last_gen_ms());

    // reset, same prompt, same ids.
    eng.reset();
    CHECK_EQ(eng.pos(), uint32_t(0));
    eng.ingest(prompt());
    const std::vector<uint32_t> b = eng.generate(kGen);
    CHECK_EQ(b.size(), size_t(kGen));
    for (uint32_t i = 0; i < kGen; ++i) {
      if (a[i] != b[i]) std::fprintf(stderr, "token %u: run 1 %u, run 2 %u\n", i, a[i], b[i]);
      CHECK_EQ(a[i], b[i]);
    }

    // The max_len wall. Reached without generating 16 k tokens: the control
    // block is shared memory, so the host can put `pos` on the last slot
    // directly. The step that would write KV row max_len must throw.
    eng.buffers().control.as<runtime::Control>()->pos = eng.max_len();
    bool threw = false;
    try {
      eng.generate(1);
    } catch (const std::exception& e) {
      threw = true;
      std::printf("max_len wall: %s\n", e.what());
    }
    CHECK(threw);

    // ...and the tap that was never captured.
    threw = false;
    try {
      eng.read_debug_resid();
    } catch (const std::exception& e) {
      threw = true;
      std::printf("tap refusal: %s\n", e.what());
    }
    CHECK(threw);
  }

  // --- phase 2: debug_resid on ----------------------------------------------
  {
    loader::LoadedModel m = loader::load(ctx, arg, kMaxLen);
    kLayers = m.desc->layers;
    kHidden = m.desc->hidden;
    kTapElems = size_t(kLayers) * kHidden;
    runtime::Engine eng(ctx, std::move(m), kMaxLen, /*debug_resid=*/true);
    CHECK(eng.debug_resid());
    eng.ingest(prompt());

    // The tap is re-written every replay, so reading it after each of two
    // steps must give two different pictures - that is what says the accessor
    // reads the live buffer and not a stale copy or a zero page.
    const std::vector<uint32_t> t0 = eng.generate(1);
    const std::vector<uint16_t> tap0 = eng.read_debug_resid();
    const std::vector<uint32_t> t1 = eng.generate(1);
    const std::vector<uint16_t> tap1 = eng.read_debug_resid();
    CHECK_EQ(tap0.size(), kTapElems);
    CHECK_EQ(tap1.size(), kTapElems);
    CHECK(tap0 != tap1);

    // Every layer's row is populated and finite. A layer that came back all
    // zero would be a tap bound to the wrong slice, not a residual.
    for (uint32_t l = 0; l < kLayers; ++l) {
      bool nonzero = false;
      for (uint32_t k = 0; k < kHidden; ++k) {
        const uint16_t w = tap0[size_t(l) * kHidden + k];
        if (!bf16_finite(w)) {
          std::fprintf(stderr, "non-finite resid: layer %u element %u = 0x%04X\n", l, k, w);
          CHECK(false);
        }
        nonzero = nonzero || w != 0;
      }
      CHECK(nonzero);
    }

    // The tap costs 64 device copies a token and changes nothing else: the
    // same prompt must still produce the same ids.
    CHECK_EQ(t0[0], a[0]);
    CHECK_EQ(t1[0], a[1]);
    const std::vector<uint32_t> rest = eng.generate(kGen - 2);
    for (uint32_t i = 0; i < kGen - 2; ++i) CHECK_EQ(rest[i], a[i + 2]);
    std::printf("tap on: %.2f ms/token vs %.2f off (%+.2f ms for 64 device copies)\n",
                eng.last_gen_ms() / (kGen - 2), plain_ms_per_token,
                eng.last_gen_ms() / (kGen - 2) - plain_ms_per_token);
  }

  std::printf("engine_smoke_test OK (%u tokens, tap off and on, ids identical)\n", kGen);
  return 0;
}
