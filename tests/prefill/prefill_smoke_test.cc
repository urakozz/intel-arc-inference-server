// `Engine::prefill` - the chunk walk, the `Control` handoff, and the
// rejections (plan 6b Task 10 Step 1).
//
// This is the plumbing test: it proves that a prefill runs at all, that it
// leaves the engine in a state decode continues from, that the walk is the
// shape the launch arithmetic says it is, and that every precondition throws by
// name rather than striding a buffer. **What it does NOT do is grade the
// numbers** - `prefill_gate_test`, `prefill_consistency_test` and
// `prefill_determinism_test` are the three gates and they are Task 12.
//
// argv[1] is the snapshot, per the `B70_TEST_SNAPSHOT` convention (see
// tests/CMakeLists.txt's comment on why every checkpoint test takes it
// explicitly rather than depending on `refs/main`).
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "model/model_desc.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "l0/cmdlist.h"
#include "runtime/engine.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/step.h"

namespace {
using model::Qwen35;

constexpr uint32_t kMaxLen = 16384;

// `b70_decode.cc`'s `kBenchPrompt`, cycled - the 42 ids of the committed
// golden prompt `tests/golden/prompts/prose.ids`. Copied rather than included
// because this test must not depend on the CLI's translation unit; using the
// same ids means the smoke test and `--bench --pp` drive the same tokens.
constexpr uint32_t kSeed[] = {
    760,   72103,  506,  37119, 557,   11012, 3213,  310,  6512, 279, 61789, 272, 1072, 2272,
    279,   197616, 2271, 13,    469,   68042, 29123, 7247, 383,  279, 1387,  12615, 1345, 279,
    49813, 78911,  1141, 20459, 13,    3113,  7840,  279,  2981, 1000, 381,  16850, 1495, 13};
constexpr size_t kSeedLen = sizeof(kSeed) / sizeof(kSeed[0]);

std::vector<uint32_t> ids(size_t n) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = kSeed[i % kSeedLen];
  return v;
}

// Every rejection is asserted to throw AND to name the thing it is about, so a
// future change that throws for a different reason does not pass silently.
template <class F>
void throws_naming(F&& f, const char* needle, const char* what) {
  try {
    f();
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (msg.find(needle) == std::string::npos) {
      std::fprintf(stderr, "%s: threw, but the message does not contain '%s': %s\n", what,
                   needle, msg.c_str());
      CHECK(false);
    }
    std::printf("  rejects %-28s -> %s\n", what, msg.substr(0, 96).c_str());
    return;
  }
  std::fprintf(stderr, "%s: did NOT throw\n", what);
  CHECK(false);
}

// Spec 6 Review Focus: what one prefill leaves behind, for bitwise comparison.
struct State {
  std::vector<uint8_t> logits, gdn, control;
  bool operator==(const State& o) const {
    return logits == o.logits && gdn == o.gdn && control == o.control;
  }
};
State grab(runtime::Engine& e, l0::Context& ctx) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  State st;
  auto one = [&](std::vector<uint8_t>& v, const l0::Mem& m) {
    v.resize(m.size());
    imm.copy(v.data(), m.ptr(), m.size());
  };
  one(st.logits, e.prefill_scratch()->logits);
  one(st.gdn, e.buffers().gdn_state);
  one(st.control, e.buffers().control);
  return st;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snap =
      argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";

  using runtime::prefill::AttnMode;
  // The mode the environment selected (B70_PREFILL_ATTN); the sections below that toggle it
  // put it back, so the live measurements run in whatever mode the test was started in.
  const AttnMode env_mode = runtime::prefill::attn_mode();
  std::printf("attention mode: %s\n", runtime::prefill::attn_mode_name(env_mode));

  l0::Context ctx(0);
  {
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  runtime::Engine eng(ctx, std::move(model), kMaxLen);

  // ---- 1. ruling R7: nothing prefill-shaped is allocated yet ---------------
  CHECK(eng.prefill_scratch() == nullptr);
  CHECK_EQ(eng.prefill_launches(), size_t(0));
  CHECK_EQ(eng.buffers().persistent_bytes(), size_t(1240465536));
  CHECK_EQ(eng.step().kernel_count, size_t(774));
  std::printf("R7: persistent %zu B, decode list %zu launches / %zu modules, PrefillScratch"
              " not allocated\n",
              eng.buffers().persistent_bytes(), eng.step().kernel_count,
              eng.step().modules.size());

  // ---- 2. one chunk ------------------------------------------------------
  const uint32_t kShort = 64;
  // its launch arithmetic is sycl-tla's; the default flips to l0 at spec 2.1's close
  eng.set_prefill_backend(runtime::PrefillBackend::SyclTla);
  eng.reset();
  eng.prefill(ids(kShort));
  CHECK_EQ(eng.pos(), kShort);
  CHECK(eng.prefill_scratch() != nullptr);

  // The launch arithmetic, pinned. `step_chunk_launches(model::qwen38(), )` is derived from the
  // walk itself (src/runtime/prefill/step.cc's closing block) and
  // docs/prefill-l1-engine-preregistration-2026-09-05.md §1 pre-registered its
  // value at 1201 per chunk + 5 for the head, independent of C. If this ever
  // differs, print both and re-derive the arithmetic from the walk - the number
  // in the pre-registration is a prediction about the walk, not a target.
  const size_t per_chunk =
      runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::SyclTla, kShort);
  const size_t head = runtime::prefill::kStepHeadLaunches;
  CHECK_EQ(per_chunk, size_t(1201));
  CHECK_EQ(head, size_t(5));
  CHECK_EQ(eng.prefill_launches(), per_chunk + head);
  std::printf("launch arithmetic: %zu L0 per chunk + %zu head = %zu for one chunk;"
              " %zu SYCL GEMMs and %zu host waits per chunk (neither on the L0 counter)\n",
              per_chunk, head, eng.prefill_launches(),
              runtime::prefill::step_chunk_gemms(model::qwen38(), runtime::PrefillBackend::SyclTla),
              runtime::prefill::step_chunk_waits(model::qwen38(), runtime::PrefillBackend::SyclTla));
  // S3: the L0 backend's linears are the slab walk (2 x N/1024 launches each) and its
  // attention's two GEMMs per kv group are pf_gemm launches -- a chunk calls no SYCL and
  // waits on the host nowhere (spec 2.1 §3.3 / §3.4).
  //
  // The parity program's S3 splits the QK^T GEMM into ROW BLOCKS of 256, so the count is
  // a function of C for the first time; S2a then fuses `pf_silu_mul` into the gate||up slab
  // GEMM's epilogue and every layer loses one launch. 8625 while one block covers the chunk
  // (C <= 256, this test's own shape) and 9073 at the C = 2048 the bench and the server run
  // (1 + 48 x 135 + 16 x (130 + 4 x 8), derived from src/runtime/prefill/step.cc's closing
  // block).
  runtime::prefill::set_attn_mode_for_test(AttnMode::Composed);
  CHECK_EQ(runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, kShort),
           size_t(8625));
  CHECK_EQ(runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, 2048),
           size_t(9073));
  // Spec 6: in Flash mode attention is ONE launch per FA layer, so a chunk drops by
  // 16 x (attn_chunk_launches_composed(C) - 1) and stops depending on C: 8449 on L0 and
  // 8705 on l0-int8 at C = 64 (drop 176), 1000 (drop 368) and 2048 (drop 624). Derived:
  // 1 + 48 x 135 + 16 x 123.
  for (runtime::PrefillBackend b : {runtime::PrefillBackend::L0, runtime::PrefillBackend::L0Int8})
    for (uint32_t C : {kShort, 1000u, 2048u}) {
      runtime::prefill::set_attn_mode_for_test(AttnMode::Composed);
      const size_t composed = runtime::prefill::step_chunk_launches(model::qwen38(), b, C);
      runtime::prefill::set_attn_mode_for_test(AttnMode::Flash);
      const size_t flash = runtime::prefill::step_chunk_launches(model::qwen38(), b, C);
      const size_t drop =
          16 * (runtime::prefill::attn_chunk_launches_composed(model::qwen38(), C, b) - 1);
      std::printf("  %s C = %4u: composed %zu, flash %zu (drop %zu)\n",
                  runtime::prefill_backend_name(b), C, composed, flash, drop);
      CHECK_EQ(composed - flash, drop);
      CHECK_EQ(flash, size_t(b == runtime::PrefillBackend::L0 ? 8449 : 8705));
      CHECK_EQ(drop, size_t(C == kShort ? 176 : C == 1000u ? 368 : 624));
    }
  runtime::prefill::set_attn_mode_for_test(env_mode);
  CHECK_EQ(runtime::prefill::step_chunk_gemms(model::qwen38(), runtime::PrefillBackend::L0), size_t(0));
  CHECK_EQ(runtime::prefill::step_chunk_waits(model::qwen38(), runtime::PrefillBackend::L0), size_t(0));
  // Spec 5 (plan 5b Task 1): l0-int8 is L0 but for the int4 linears, and each of those
  // adds exactly one launch (the activation quantiser): 4 per layer, +256 per chunk.
  {
    runtime::PrefillBackend b{};
    CHECK(runtime::parse_prefill_backend("l0-int8", b) && b == runtime::PrefillBackend::L0Int8);
    CHECK(std::string(runtime::prefill_backend_name(runtime::PrefillBackend::L0Int8)) ==
          "l0-int8");
    CHECK(runtime::is_l0(runtime::PrefillBackend::L0Int8) &&
          runtime::is_l0(runtime::PrefillBackend::L0));
    CHECK(!runtime::is_l0(runtime::PrefillBackend::SyclTla));
    for (uint32_t C : {kShort, 1000u, 2048u})
      CHECK_EQ(runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0Int8, C),
               runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, C) + 256);
    std::printf("l0-int8 backend: %zu launches at C = %u and %zu at C = 2048 (L0 + 256)\n",
                runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0Int8, kShort),
                kShort, runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0Int8, 2048));
  }
  std::printf("L0 backend: %zu launches at C = %u and %zu at C = 2048 (8 QK^T row blocks),"
              " %zu SYCL GEMMs, %zu host waits\n",
              runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, kShort), kShort,
              runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, 2048),
              runtime::prefill::step_chunk_gemms(model::qwen38(), runtime::PrefillBackend::L0),
              runtime::prefill::step_chunk_waits(model::qwen38(), runtime::PrefillBackend::L0));

  // ---- 3. the Control handoff, exactly as argmax_stage2 leaves it ---------
  const runtime::Control* c = eng.buffers().control.as<runtime::Control>();
  CHECK(c->cur_token[0] < Qwen35::kVocabUsed);
  CHECK_EQ(c->out_token[0], c->cur_token[0]);
  std::printf("handoff: pos %u, cur_token %u == out_token %u (< kVocabUsed %u)\n", eng.pos(),
              c->cur_token[0], c->out_token[0], Qwen35::kVocabUsed);

  // ---- 4. decode continues from prefill's state --------------------------
  const std::vector<uint32_t> gen = eng.generate(8);
  CHECK_EQ(gen.size(), size_t(8));
  for (uint32_t id : gen) CHECK(id < Qwen35::kVocabUsed);
  CHECK_EQ(eng.pos(), kShort + 8);
  std::printf("generate(8) after prefill:");
  for (uint32_t id : gen) std::printf(" %u", id);
  std::printf("   (pos %u)\n", eng.pos());

  // ---- 5. multi-chunk with a ragged tail ---------------------------------
  eng.reset();
  eng.prefill(ids(200), /*chunk=*/64);           // 64 + 64 + 64 + 8
  CHECK_EQ(eng.pos(), uint32_t(200));
  std::printf("multi-chunk: 200 ids at chunk 64 (3 full + a ragged 8) -> pos %u,"
              " cumulative launches %zu\n", eng.pos(), eng.prefill_launches());

  // ---- 6. incremental prefill: two calls extend one session --------------
  eng.reset();
  eng.prefill(ids(32));
  CHECK_EQ(eng.pos(), uint32_t(32));
  eng.prefill(ids(16));
  CHECK_EQ(eng.pos(), uint32_t(48));
  std::printf("incremental: prefill(32) then prefill(16) -> pos %u\n", eng.pos());

  // ---- 7. the rejections -------------------------------------------------
  throws_naming([&] { eng.prefill({}); }, "no ids", "empty ids");
  throws_naming([&] { eng.prefill(ids(8), 5000); }, "PrefillScratch::kC", "chunk > kC");
  throws_naming([&] { eng.prefill({Qwen35::kVocab}); }, "vocabulary", "id == kVocab");
  eng.reset();
  throws_naming([&] { eng.prefill(ids(size_t(kMaxLen) + 1)); }, "max_len",
                "more ids than max_len");

  // ---- 8. the L0 backend's launch arithmetic, MEASURED (ruling E1) --------
  // Sections 2-7 assert sycl-tla's 1201 + 5 against the live counter and state the
  // L0 formula; nothing ran it. One chunk on L0 from a reset must advance the SAME
  // counter by `step_chunk_launches(model::qwen38(), L0, C) + kStepHeadLaunches` (8625 + 5 = 8630
  // at this C, derived from src/runtime/prefill/step.cc's closing block). The delta
  // is taken rather than the absolute, because `prefill_launches()` is cumulative
  // over the engine's whole life and sections 2-7 already spent some of it.
  const size_t l0_before = eng.prefill_launches();
  eng.set_prefill_backend(runtime::PrefillBackend::L0);
  eng.reset();
  eng.prefill(ids(kShort));
  CHECK_EQ(eng.pos(), kShort);
  const size_t l0_delta = eng.prefill_launches() - l0_before;
  const size_t l0_want =
      runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, kShort) +
      runtime::prefill::kStepHeadLaunches;
  // Printed BEFORE the assertion so a mismatch shows both numbers rather than only
  // the macro's text: the formula is a prediction about the walk, not a target.
  std::printf("L0 live: one chunk of %u ids advanced the L0 counter by %zu launches;"
              " the arithmetic says %zu (%zu per chunk + %zu head)\n",
              kShort, l0_delta, l0_want,
              runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, kShort),
              runtime::prefill::kStepHeadLaunches);
  CHECK_EQ(l0_delta, l0_want);
  CHECK_EQ(l0_want, size_t(env_mode == AttnMode::Flash ? 8454 : 8630));

  // The shape that S3 actually changed: C = 2048 is eight QK^T row blocks, so the
  // same counter must advance by 9073 + 5. One chunk, so the ragged-tail arithmetic is
  // not in the way -- exactly the C = 64 measurement above at the bench's width.
  const size_t wide_before = eng.prefill_launches();
  eng.reset();
  eng.prefill(ids(2048), /*chunk=*/2048);
  CHECK_EQ(eng.pos(), uint32_t(2048));
  const size_t wide_delta = eng.prefill_launches() - wide_before;
  const size_t wide_want =
      runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, 2048) +
      runtime::prefill::kStepHeadLaunches;
  std::printf("L0 live: one chunk of 2048 ids advanced the L0 counter by %zu launches;"
              " the arithmetic says %zu (%zu per chunk + %zu head)\n",
              wide_delta, wide_want,
              runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, 2048),
              runtime::prefill::kStepHeadLaunches);
  CHECK_EQ(wide_delta, wide_want);
  CHECK_EQ(wide_want, size_t(env_mode == AttnMode::Flash ? 8454 : 9078));
  }   // the first engine is gone: the Review Focus below needs fresh ones

  // ---- 9. spec 6 Review Focus: a composed session after a flash session ----
  // The lazy pf_s / pf_p must be built on first composed use, not assumed. Engine 1, fresh:
  // flash, then reset() and composed. Engine 2, fresh: composed, then reset() and flash.
  // Each mode's two outputs (last logits, GDN state, Control) must be bitwise equal: the
  // first-in-a-fresh-engine run against the run that follows the other mode.
  {
    const std::vector<uint32_t> prose(kSeed, kSeed + kSeedLen);
    State flash1, comp1, comp2, flash2;
    {
      runtime::Engine e(ctx, loader::load(ctx, snap, kMaxLen), kMaxLen);
      runtime::prefill::set_attn_mode_for_test(AttnMode::Flash);
      e.reset();
      e.prefill(prose);
      CHECK_EQ(e.prefill_scratch()->pf_s_bytes(), size_t(0));
      flash1 = grab(e, ctx);
      runtime::prefill::set_attn_mode_for_test(AttnMode::Composed);
      e.reset();
      e.prefill(prose);
      CHECK(e.prefill_scratch()->pf_s_bytes() > 0);
      comp1 = grab(e, ctx);
      std::printf("review focus, engine 1 (%s): flash then composed; pf_s built on first"
                  " composed use (%zu B)\n", runtime::prefill_backend_name(e.prefill_backend()),
                  e.prefill_scratch()->pf_s_bytes());
    }
    {
      runtime::Engine e(ctx, loader::load(ctx, snap, kMaxLen), kMaxLen);
      runtime::prefill::set_attn_mode_for_test(AttnMode::Composed);
      e.reset();
      e.prefill(prose);
      comp2 = grab(e, ctx);
      runtime::prefill::set_attn_mode_for_test(AttnMode::Flash);
      e.reset();
      e.prefill(prose);
      flash2 = grab(e, ctx);
    }
    runtime::prefill::set_attn_mode_for_test(env_mode);
    const bool fl = flash1 == flash2, co = comp1 == comp2;
    std::printf("review focus: flash fresh == flash after composed: %s; composed fresh =="
                " composed after flash: %s; flash != composed logits: %s\n",
                fl ? "bitwise" : "DIFFER", co ? "bitwise" : "DIFFER",
                flash1.logits != comp1.logits ? "yes (two paths)" : "no");
    CHECK(fl);
    CHECK(co);
  }

  std::puts("prefill_smoke_test OK");
  return 0;
}
