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
#include "chunk_cap_arg.h"
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
// same ids means the smoke test and `--bench --prefill-length` drive the same tokens.
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
  b70test::take_chunk_cap_arg(argc, argv);   // plan 5d: --chunk-cap 4096 (the test seam)
  const std::string snap =
      argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";

  using runtime::prefill::AttnMode;
  // The process's mode (flash: B70_PREFILL_ATTN is gone since the zero-config cleanup); the
  // sections below that set the composed reference put it back.
  const AttnMode env_mode = runtime::prefill::attn_mode();
  std::printf("attention mode: %s\n", runtime::prefill::attn_mode_name(env_mode));

  // 3j: the engines' KV form (B70_KV_CACHE, the Engine's default) goes into every LIVE launch count, as the
  // production counts take it (step_chunk_launches' `kv`): int8 widens each FA layer's raw K/V first, one launch a
  // layer (+16 on Qwen3.8).
  const runtime::KvCache kv = runtime::default_kv_cache();
  const size_t widen = kv == runtime::KvCache::Int8 ? model::qwen38().fa_layers : 0;
  std::printf("KV cache: %s (+%zu launches a chunk)\n", runtime::kv_cache_name(kv), widen);

  l0::Context ctx(0);
  {
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  CHECK(eng.kv_cache() == kv);

  // ---- 1. ruling R7: nothing prefill-shaped is allocated yet ---------------
  CHECK(eng.prefill_scratch() == nullptr);
  CHECK_EQ(eng.prefill_launches(), size_t(0));
  // int8 KV: 32 KiB of rows + 256 B of scales a position instead of 64 KiB (PersistentDims::sizes).
  CHECK_EQ(eng.buffers().persistent_bytes(), size_t(kv == runtime::KvCache::Int8 ? 707788928 : 1240465536));
  CHECK_EQ(eng.step().kernel_count, size_t(774));
  std::printf("R7: persistent %zu B, decode list %zu launches / %zu modules, PrefillScratch"
              " not allocated\n",
              eng.buffers().persistent_bytes(), eng.step().kernel_count,
              eng.step().modules.size());

  // ---- 2. one chunk ------------------------------------------------------
  const uint32_t kShort = 64;
  eng.set_prefill_backend(runtime::PrefillBackend::L0);
  eng.reset();
  eng.prefill(ids(kShort));
  CHECK_EQ(eng.pos(), kShort);
  CHECK(eng.prefill_scratch() != nullptr);

  // The launch arithmetic, pinned. `step_chunk_launches(model::qwen38(), )` is derived from the
  // walk itself (src/runtime/prefill/step.cc's closing block); one chunk from a fresh engine
  // advances the counter by it + 5 for the head (1089 + 5 in Flash mode, 1265 + 5 composed, at
  // this C: open-work 3b runs each int4 linear of a chunk this short as ONE pf_gemm_q4 launch).
  // If this ever differs, print both and re-derive the arithmetic from the walk.
  const size_t per_chunk =
      runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, kShort, kv);
  const size_t head = runtime::prefill::kStepHeadLaunches;
  CHECK_EQ(per_chunk, size_t(env_mode == AttnMode::Flash ? 1089 : 1265) + widen);
  CHECK_EQ(head, size_t(5));
  CHECK_EQ(eng.prefill_launches(), per_chunk + head);
  std::printf("launch arithmetic: %zu L0 per chunk + %zu head = %zu for one chunk\n", per_chunk, head,
              eng.prefill_launches());
  // S3: the L0 backend's linears are the slab walk (2 x N/1024 launches each) and its
  // attention's two GEMMs per kv group are pf_gemm launches -- a chunk waits on the host
  // nowhere (spec 2.1 §3.3 / §3.4).
  //
  // The parity program's S3 splits the QK^T GEMM into ROW BLOCKS of 256, so the count is
  // a function of C for the first time; S2a then fuses `pf_silu_mul` into the gate||up slab
  // GEMM's epilogue and every layer loses one launch. 8625 while one block covers the chunk
  // (C <= 256, this test's own shape) and 9073 at the C = 2048 the bench and the server run
  // (1 + 48 x 135 + 16 x (130 + 4 x 8), derived from src/runtime/prefill/step.cc's closing
  // block). Open-work 3b: at C <= 192 every int4 linear is one launch (1 + 48 x 19 + 16 x 22 =
  // 1265 composed at this C). Open-work 3d's SLM gate/up epilogue keeps one launch through
  // 320 rows; above that, through 384 rows, every linear but gate||up remains fused.
  runtime::prefill::set_attn_mode_for_test(AttnMode::Composed);
  CHECK_EQ(runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, kShort),
           size_t(1265));
  CHECK_EQ(runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, 2048),
           size_t(9073));
  // Spec 6: in Flash mode attention is ONE launch per FA layer, so a chunk drops by
  // 16 x (attn_chunk_launches_composed(C) - 1): 8449 on L0 and 8705 on l0-int8 at C = 1000
  // (drop 368) and 2048 (drop 624), 1 + 48 x 135 + 16 x 123 (derived); at C = 64 (drop 176)
  // l0-int8 is 8705 and L0 1089 (open-work 3b: 1 + 48 x 19 + 16 x 11, one pf_gemm_q4 launch
  // a linear). Open-work 3d keeps L0 at 1089 through C = 320 (the SLM gate/up epilogue is
  // one launch); at C = 321 it is 5377 (gate||up's 68 slab launches: 1 + 48 x 86 + 16 x 78).
  for (runtime::PrefillBackend b : {runtime::PrefillBackend::L0, runtime::PrefillBackend::L0Int8})
    for (uint32_t C : {kShort, 250u, 300u, 320u, 321u, 1000u, 2048u}) {
      runtime::prefill::set_attn_mode_for_test(AttnMode::Composed);
      const size_t composed = runtime::prefill::step_chunk_launches(model::qwen38(), b, C);
      runtime::prefill::set_attn_mode_for_test(AttnMode::Flash);
      const size_t flash = runtime::prefill::step_chunk_launches(model::qwen38(), b, C);
      const size_t drop =
          16 * (runtime::prefill::attn_chunk_launches_composed(model::qwen38(), C, b) - 1);
      std::printf("  %s C = %4u: composed %zu, flash %zu (drop %zu)\n",
                  runtime::prefill_backend_name(b), C, composed, flash, drop);
      CHECK_EQ(composed - flash, drop);
      const bool l0 = b == runtime::PrefillBackend::L0;
      CHECK_EQ(flash, size_t(!l0 ? 8705 : C <= 320u ? 1089 : C == 321u ? 5377 : 8449));
      CHECK_EQ(drop, size_t(C <= 256u ? 176 : C <= 321u ? 240 : C == 1000u ? 368 : 624));
    }
  runtime::prefill::set_attn_mode_for_test(env_mode);
  // Spec 5 (plan 5b Task 1): l0-int8 is L0 but for the int4 linears, and each of those
  // adds exactly one launch (the activation quantiser): 4 per layer, +256 per chunk - above
  // kPfQ4MaxRows, where L0 walks the same slabs (open-work 3b: below, L0 is one launch a linear).
  {
    runtime::PrefillBackend b{};
    CHECK(runtime::parse_prefill_backend("l0-int8", b) && b == runtime::PrefillBackend::L0Int8);
    CHECK(std::string(runtime::prefill_backend_name(runtime::PrefillBackend::L0Int8)) ==
          "l0-int8");
    CHECK(!runtime::parse_prefill_backend("sycl-tla", b));
    for (uint32_t C : {1000u, 2048u})
      CHECK_EQ(runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0Int8, C),
               runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, C) + 256);
    std::printf("l0-int8 backend: %zu launches at C = %u and %zu at C = 2048 (L0 + 256)\n",
                runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0Int8, kShort),
                kShort, runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0Int8, 2048));
  }
  std::printf("L0 backend: %zu launches at C = %u and %zu at C = 2048 (8 QK^T row blocks)\n",
              runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, kShort), kShort,
              runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, 2048));

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
  throws_naming([&] { eng.prefill(ids(8), 5000); }, "chunk cap", "chunk > the chunk cap");
  // Plan 5d: the cap is 2048 or 4096, and sized once - after the scratch exists another cap
  // is refused, the same one accepted.
  throws_naming([&] { eng.set_prefill_chunk_cap(3072); }, "neither", "chunk cap 3072");
  if (eng.prefill_chunk_cap() == 2048u) {
    throws_naming([&] { eng.set_prefill_chunk_cap(4096); }, "before prepare_prefill", "cap after the scratch");
    eng.set_prefill_chunk_cap(2048);
    throws_naming([&] { eng.prefill(ids(8), 4096); }, "chunk cap 2048", "chunk 4096 at cap 2048");
  }
  throws_naming([&] { eng.prefill({Qwen35::kVocab}); }, "vocabulary", "id == kVocab");
  eng.reset();
  throws_naming([&] { eng.prefill(ids(size_t(kMaxLen) + 1)); }, "max_len",
                "more ids than max_len");

  // ---- 8. the L0 backend's launch arithmetic, MEASURED (ruling E1) --------
  // Section 2 measured one chunk from a fresh engine; this one measures it again after
  // sections 3-7 have used the engine. One chunk on L0 from a reset must advance the SAME
  // counter by `step_chunk_launches(model::qwen38(), L0, C) + kStepHeadLaunches` (1089 + 5 = 1094
  // at this C in Flash mode, 1265 + 5 composed, derived from src/runtime/prefill/step.cc's closing block). The delta
  // is taken rather than the absolute, because `prefill_launches()` is cumulative
  // over the engine's whole life and sections 2-7 already spent some of it.
  for (uint32_t C : {kShort, 250u, 300u}) {
    const size_t l0_before = eng.prefill_launches();
    eng.set_prefill_backend(runtime::PrefillBackend::L0);
    eng.reset();
    eng.prefill(ids(C));
    CHECK_EQ(eng.pos(), C);
    const size_t l0_delta = eng.prefill_launches() - l0_before;
    const size_t l0_want =
        runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, C, kv) +
        runtime::prefill::kStepHeadLaunches;
    // Print the prediction and live count before asserting so a wrong launch count is visible.
    std::printf("L0 live: one chunk of %u ids advanced the L0 counter by %zu launches;"
                " the arithmetic says %zu (%zu per chunk + %zu head)\n",
                C, l0_delta, l0_want,
                runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, C, kv),
                runtime::prefill::kStepHeadLaunches);
    CHECK_EQ(l0_delta, l0_want);
    const size_t flash_count = 1094 + widen;
    CHECK_EQ(l0_want, env_mode == AttnMode::Flash ? flash_count : flash_count + (C <= 256u ? 176 : 240));
  }

  // The shape that S3 actually changed: C = 2048 is eight QK^T row blocks, so the
  // same counter must advance by 9073 + 5. One chunk, so the ragged-tail arithmetic is
  // not in the way -- exactly the C = 64 measurement above at the bench's width.
  const size_t wide_before = eng.prefill_launches();
  eng.reset();
  eng.prefill(ids(2048), /*chunk=*/2048);
  CHECK_EQ(eng.pos(), uint32_t(2048));
  const size_t wide_delta = eng.prefill_launches() - wide_before;
  const size_t wide_want =
      runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, 2048, kv) +
      runtime::prefill::kStepHeadLaunches;
  std::printf("L0 live: one chunk of 2048 ids advanced the L0 counter by %zu launches;"
              " the arithmetic says %zu (%zu per chunk + %zu head)\n",
              wide_delta, wide_want,
              runtime::prefill::step_chunk_launches(model::qwen38(), runtime::PrefillBackend::L0, 2048, kv),
              runtime::prefill::kStepHeadLaunches);
  CHECK_EQ(wide_delta, wide_want);
  CHECK_EQ(wide_want, size_t(env_mode == AttnMode::Flash ? 8454 : 9078) + widen);
  }   // the first engine is gone: the Review Focus below needs fresh ones

  // ---- 9. spec 6 Review Focus: a composed session after a flash session ----
  // The lazy pf_s / pf_p must be built on first composed use, not assumed. Engine 1, fresh:
  // flash, then reset() and composed. Engine 2, fresh: composed, then reset() and flash.
  // Each mode's two outputs (last logits, GDN state, Control) must be bitwise equal: the
  // first-in-a-fresh-engine run against the run that follows the other mode.
  // 3j: int8 KV prefills through the flash attention only (attn.cc, require_kv8_path) - the composed
  // attention is the bf16-KV reference, so this section is bf16's.
  if (kv == runtime::KvCache::Int8) {
    std::printf("review focus: skipped at int8 KV (the composed attention is the bf16-KV reference)\n");
  } else {
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
