// prefill_int8_test - plan 5b Task 2: `--prefill-backend l0-int8` on the real engine.
//
// For `prose` (one short chunk) and the first 2048 ids of `long` (one full chunk):
//
//   1. prefill on l0: the first generated id and the last-row logits;
//   2. reset, prefill on l0-int8: finite logits and the same first id on both, and on
//      `prose` a last-row cosine >= 0.99 against l0. A SMOKE bar: the acceptance bars
//      (A2, A3) are the golden gates, `prefill_gate_int8*_test` and
//      tools/rotate/gate_compare.py. The plan also put the 0.99 bar on long[:2048], and
//      that is not a property of ONE row: `--sweep` below measured the l0 / l0-int8
//      last-row cosine over 30 prefixes of `long` at 0.998 to 0.9999 on most and
//      0.88 to 0.95 on a scattered few (257, 258, 800, 1100, 1200, 1536, 2048), with
//      the first id equal on all 30 and neighbours of a bad length good (258 at 0.943,
//      260 at 0.9994). `--history` measured the row independent of the scratch's padded
//      rows and of splitting the chunk at 256, and moved to 0.89 by splitting at 16,
//      where l0 moves to 0.9987: the int8 rounding redraws its noise on any perturbation,
//      and some positions are sensitive to it. So on `long` the cosine is printed, not
//      gated (2026-09-24, plan 5b Task 2);
//   3. reset, l0-int8 with replay twice: the replayed state (logits, Control, GDN state,
//      conv ring, KV) is BITWISE the non-replay int8 run's. This is the plan's Review
//      Focus "scales built mid-recording": a recording holding one-time work (the host
//      finish of pf_colmax_rot) would either fail to replay or diverge;
//   4. the launch delta of the 2048-id int8 prefill is step_chunk_launches(model::qwen38(), L0Int8, 2048)
//      + the head's 5, i.e. l0's + 256, which is the proof the walk really ran the h8
//      linears (on the bf16 walk it would be 256 short).
//
// `--first-cost <backend>`: a FRESH engine, three 4096-id prefills at chunk 2048, each
// timed around Engine::prefill (host wall). The first call's excess over the later ones
// is the one-time cost a user's first request pays: scratch allocation and module loads on
// both backends, plus the column-scale pass over every int4 linear on l0-int8.
//
// argv[1]: the snapshot (B70_TEST_SNAPSHOT). Runs from the source root (the prompt ids).
// A snapshot that cannot be resolved is a SKIP (77).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "model/model_desc.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill/step.h"
#include "runtime/prefill_backend.h"

namespace {
using model::Qwen35;
using runtime::PrefillBackend;
constexpr uint32_t kMaxLen = 16384;

struct Run {
  uint32_t first_id = 0;
  std::vector<float> logits;                   // the last row, [kVocab]
  std::vector<std::vector<uint8_t>> state;     // persistent buffers + Control, for bitwise
};

Run run(runtime::Engine& eng, l0::CmdList& copy, PrefillBackend b, bool replay,
        const std::vector<uint32_t>& ids, bool grab_state, uint32_t chunk = 2048) {
  eng.set_prefill_backend(b);
  eng.set_prefill_replay(replay);
  eng.reset();
  eng.prefill(ids, chunk);
  CHECK_EQ(eng.pos(), uint32_t(ids.size()));
  Run r;
  r.first_id = eng.buffers().control.as<runtime::Control>()->cur_token[0];
  r.logits.resize(Qwen35::kVocab);
  copy.copy(r.logits.data(), eng.prefill_scratch()->logits.ptr(), r.logits.size() * 4);
  if (grab_state) {
    auto grab = [&](const l0::Mem& m) {
      r.state.emplace_back(m.size());
      copy.copy(r.state.back().data(), m.ptr(), m.size());
    };
    grab(eng.buffers().gdn_state);
    grab(eng.buffers().conv_ring);
    grab(eng.buffers().kv_k);
    grab(eng.buffers().kv_v);
    grab(eng.buffers().control);
  }
  return r;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb);
}

bool finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

int first_cost(runtime::Engine& eng, const std::string& backend_name) {
  PrefillBackend b{};
  CHECK(runtime::parse_prefill_backend(backend_name, b));
  eng.set_prefill_backend(b);
  eng.set_prefill_replay(false);
  const auto seed = golden::read_ids("tests/golden/prompts/prose.ids");
  std::vector<uint32_t> ids(4096);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = seed[i % seed.size()];
  std::vector<double> ms;
  for (int call = 0; call < 3; ++call) {
    eng.reset();
    const auto t0 = std::chrono::steady_clock::now();
    eng.prefill(ids, 2048);
    ms.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    std::printf("first-cost backend=%s call=%d wall_ms=%.3f\n", backend_name.c_str(), call,
                ms.back());
    std::fflush(stdout);
  }
  std::printf("first-cost backend=%s first_minus_warm_ms=%.3f (warm = mean of calls 1, 2)\n",
              backend_name.c_str(), ms[0] - 0.5 * (ms[1] + ms[2]));
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap_arg = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  std::string snap;
  try {
    snap = loader::resolve_snapshot(snap_arg);
  } catch (const std::exception& e) {
    std::printf("SKIP: the checkpoint is not here (%s)\n", e.what());
    return 77;
  }
  l0::Context ctx(0);
  runtime::Engine eng(ctx, loader::load(ctx, snap, kMaxLen), kMaxLen);
  if (argc > 3 && std::string(argv[2]) == "--first-cost") return first_cost(eng, argv[3]);

  l0::CmdList copy = l0::CmdList::immediate(ctx);
  if (argc > 2 && std::string(argv[2]) == "--sweep") {
    // Diagnostic: the l0 / l0-int8 last-row cosine over prefixes of `long`, so a drift
    // that grows smoothly with depth (quantisation noise carried by the KV and GDN
    // state) can be told from a cliff at a shape boundary (a bug).
    // argv[3], optional: a comma-separated list of prefix lengths.
    const auto all = golden::read_ids("tests/golden/prompts/long.ids");
    std::vector<uint32_t> lens = {64, 255, 256, 257, 512, 1000, 1024, 1025, 1536, 2048};
    if (argc > 3) {
      lens.clear();
      for (const char* p = argv[3]; *p;) {
        char* e = nullptr;
        lens.push_back(uint32_t(std::strtoul(p, &e, 10)));
        p = *e == ',' ? e + 1 : e;
      }
    }
    for (uint32_t n : lens) {
      CHECK(n > 0 && n <= all.size());
      const std::vector<uint32_t> ids(all.begin(), all.begin() + n);
      const Run a = run(eng, copy, PrefillBackend::L0, false, ids, false);
      const Run b = run(eng, copy, PrefillBackend::L0Int8, false, ids, false);
      std::printf("sweep long[:%u]: first id l0 %u / l0-int8 %u, last-row cosine %.9f\n", n,
                  a.first_id, b.first_id, cosine(a.logits, b.logits));
      std::fflush(stdout);
    }
    return 0;
  }
  if (argc > 2 && std::string(argv[2]) == "--history") {
    // Diagnostic: long[:257] (255 padded rows) after two different previous prefills.
    // Rows are independent in every linear, so the result must not depend on what the
    // scratch's padded rows held before.
    const auto all = golden::read_ids("tests/golden/prompts/long.ids");
    const std::vector<uint32_t> ids(all.begin(), all.begin() + 257);
    const auto pr = golden::read_ids("tests/golden/prompts/prose.ids");
    const std::vector<uint32_t> big(all.begin() + 100, all.begin() + 2148);
    for (PrefillBackend b : {PrefillBackend::L0, PrefillBackend::L0Int8}) {
      run(eng, copy, b, false, big, false);
      const Run x = run(eng, copy, b, false, ids, false);
      run(eng, copy, b, false, pr, false);
      const Run y = run(eng, copy, b, false, ids, false);
      std::printf("history %s: long[:257] after 2048 others vs after prose: %s (cos %.9f)\n",
                  runtime::prefill_backend_name(b),
                  std::memcmp(x.logits.data(), y.logits.data(), x.logits.size() * 4) == 0
                      ? "BITWISE EQUAL" : "DIFFER",
                  cosine(x.logits, y.logits));
      std::fflush(stdout);
      for (uint32_t c : {256u, 16u}) {
        const Run z = run(eng, copy, b, false, ids, false, c);
        std::printf("history %s: long[:257] one chunk vs chunk %u: cos %.9f, first id %u / %u\n",
                    runtime::prefill_backend_name(b), c, cosine(x.logits, z.logits), x.first_id,
                    z.first_id);
        std::fflush(stdout);
      }
    }
    return 0;
  }
  const auto prose = golden::read_ids("tests/golden/prompts/prose.ids");
  auto long_ids = golden::read_ids("tests/golden/prompts/long.ids");
  CHECK(long_ids.size() >= 2048);
  long_ids.resize(2048);

  struct Case { const char* name; const std::vector<uint32_t>* ids; bool cos_bar; };
  for (const Case& c : {Case{"prose", &prose, true}, Case{"long[:2048]", &long_ids, false}}) {
    const Run l0 = run(eng, copy, PrefillBackend::L0, false, *c.ids, false);
    const size_t before = eng.prefill_launches();
    const Run i8 = run(eng, copy, PrefillBackend::L0Int8, false, *c.ids, true);
    const size_t delta = eng.prefill_launches() - before;
    const double cos = cosine(l0.logits, i8.logits);
    std::printf("%s: %zu ids, first id l0 %u / l0-int8 %u, last-row logits cosine %.9f,"
                " int8 launches %zu\n",
                c.name, c.ids->size(), l0.first_id, i8.first_id, cos, delta);
    std::fflush(stdout);
    CHECK(finite(i8.logits));
    CHECK_EQ(i8.first_id, l0.first_id);
    if (c.cos_bar) CHECK(cos >= 0.99);
    if (c.ids->size() == 2048) {
      // One chunk, the scales already built by the prose case: nothing but the walk.
      const size_t want = runtime::prefill::step_chunk_launches(model::qwen38(), PrefillBackend::L0Int8, 2048) +
                          runtime::prefill::kStepHeadLaunches;
      std::printf("  launch delta %zu, predicted %zu (l0 + 256 + head)\n", delta, want);
      CHECK_EQ(delta, want);
    }

    Run rp;
    for (int repeat = 0; repeat < 2; ++repeat)
      rp = run(eng, copy, PrefillBackend::L0Int8, true, *c.ids, true);
    const bool same_logits =
        std::memcmp(rp.logits.data(), i8.logits.data(), i8.logits.size() * 4) == 0;
    const char* names[] = {"gdn_state", "conv_ring", "kv_k", "kv_v", "control"};
    bool same_state = rp.state.size() == i8.state.size();
    for (size_t k = 0; same_state && k < i8.state.size(); ++k)
      if (rp.state[k] != i8.state[k]) {
        std::fprintf(stderr, "int8 replay mismatch: %s\n", names[k]);
        same_state = false;
      }
    std::printf("  l0-int8 replay (second recorded run) vs immediate: logits %s, state %s\n",
                same_logits ? "BITWISE EQUAL" : "DIFFER", same_state ? "BITWISE EQUAL" : "DIFFER");
    std::fflush(stdout);
    CHECK(same_logits);
    CHECK(same_state);
  }
  CHECK(!eng.prefill_sycl_side_created());
  std::puts("prefill_int8_test OK");
  return 0;
}
