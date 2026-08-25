// profile_capture_test - spec §3.5: the captured decode step, instrumented.
//
// Task 1 proved the event machinery on a two-noop list. This proves it on the
// real 774-launch step, and it proves the one property that lets the profiler
// be trusted at all: **an instrumented list computes the same tokens as a
// plain one.** A profiler that changed what it measures would price a fiction.
//
// Three assertions, spec §3.5 in order:
//
//   A. **Semantics.** Two `runtime::build` calls against the same model and
//      the same `DecodeBuffers` - one plain, one profiled. Zero the persistent
//      state, ingest a fixed 16-id prompt, generate 8 tokens with the PLAIN
//      list; re-zero, repeat with the PROFILED list. The 8 ids must be
//      identical. The signal events observe only - nothing in the list waits
//      on one - so this is what turns "should not matter" into a measurement.
//      The labels are compared too: they are host-side and must not depend on
//      whether profiling is on.
//   B. **Monotone, non-overlapping.** After a profiled replay, every launch's
//      `duration_us()` is > 0, and on the raw timestamp query the global start
//      of launch i+1 is at or after the global end of launch i. The list is
//      in-order, so its 774 kernels tile the step end to end in walk order;
//      an overlap would mean the driver had reordered them and every per-kernel
//      share in docs/15 would be wrong.
//   C. **Σ ≤ wall, and the gap.** The per-kernel durations sum to no more than
//      1.001 x the host-measured fence wall of that same replay, and the
//      difference - the first in-situ dispatch+gap measurement this project
//      has - is printed. Plus the structure Task 3's CLI indexes by:
//      `labels.size() == kernel_count == 774` and `labels[0]` naming the
//      token-boundary `embed_gather`.
//
// Deliberately NOT asserted: any absolute microsecond number. This test is the
// instrument's calibration, not a benchmark - and a profiled list is never a
// bench list (spec §3.3), because each signal carries a host-scope flush an
// unprofiled list never pays.
//
// Label `checkpoint`: needs the real 19 GB checkpoint and a B70.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"

namespace {
using model::Qwen35;

// The same 16-id prompt tests/runtime/replay_determinism_test.cc ingests - the
// first 16 ids of tests/golden/prompts/prose.txt under the checkpoint's own
// tokenizer. Literal here for the same reason it is literal there: this test
// must not depend on a generated .ids file. If one of the two copies changes,
// both change.
constexpr uint32_t kPrompt[] = {760, 72103, 506, 37119, 557, 11012, 3213, 310,
                                6512, 279, 61789, 272, 1072, 2272, 279, 197616};
constexpr size_t kPromptLen = sizeof(kPrompt) / sizeof(kPrompt[0]);

constexpr int kGen = 8;  // generated tokens per run

// The walk-clock pair straight from the driver, masked to the device's valid
// bits (tests/l0/event_test.cc uses the same helper). Absolute values mean
// nothing across a step; only wrap-safe differences do.
struct Raw {
  uint64_t start;
  uint64_t end;
};
Raw raw_global(const l0::Event& e, uint64_t mask) {
  ze_kernel_timestamp_result_t ts{};
  CHECK_EQ(zeEventQueryKernelTimestamp(e.handle(), &ts), ZE_RESULT_SUCCESS);
  return Raw{ts.global.kernelStart & mask, ts.global.kernelEnd & mask};
}
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ";
  l0::Context ctx(0);
  loader::LoadedModel m = loader::load(ctx, arg);
  runtime::DecodeBuffers b(ctx, m.max_len);

  // Two lists over ONE set of buffers: the same allocations are baked into
  // both, so whichever one is replayed advances the same state. (The golden
  // gate and the smoke test already build twice against one DecodeBuffers.)
  // No `debug_resid` on either - the tap is a command, and this test compares
  // tokens, not residuals.
  runtime::CapturedStep plain = runtime::build(ctx, m, b);
  runtime::ProfileEvents prof(ctx);
  runtime::CapturedStep instr = runtime::build(ctx, m, b, nullptr, &prof);

  std::printf("captured: plain %zu kernels, profiled %zu kernels, %zu events, max_len %u\n",
              plain.kernel_count, instr.kernel_count, prof.events.size(), b.max_len);

  // --- C (structure half): what Task 3's CLI indexes by --------------------
  // Spec §9.1 as amended: 48 GDN layers x 12 + 16 FA layers x 12 + 6
  // token-boundary kernels (576 + 192 + 6) - 645 plus the 129 launches L1
  // added by splitting every `prep_res_norm` site into `prep_res_fold` +
  // `prep_norm_finish`. replay_determinism_test pins the same 774 (and the
  // module count, which this test does not); if a lever changes the launch
  // count, BOTH move together.
  CHECK_EQ(plain.kernel_count, size_t(774));
  CHECK_EQ(instr.kernel_count, size_t(774));
  CHECK_EQ(plain.labels.size(), plain.kernel_count);
  CHECK_EQ(instr.labels.size(), instr.kernel_count);
  CHECK_EQ(prof.events.size(), instr.kernel_count);
  // Labels are host-side and always filled: the plain list has them too, and
  // they are the same strings. A label that appeared only under --profile
  // would make the profiled walk a different walk on paper.
  for (size_t i = 0; i < plain.labels.size(); ++i) {
    if (plain.labels[i] != instr.labels[i])
      std::fprintf(stderr, "label %zu: plain \"%s\", profiled \"%s\"\n", i,
                   plain.labels[i].c_str(), instr.labels[i].c_str());
    CHECK(plain.labels[i] == instr.labels[i]);
  }
  // Launch 0 is the token boundary's embed_gather - no layer, hence "--".
  CHECK(instr.labels[0].rfind("-- embed_gather", 0) == 0);
  // The last label is printed by INDEX-FROM-THE-END, not by a literal: 644 was
  // the last index of the 645-launch walk and is now an arbitrary mid-walk
  // launch, and a hard-coded index would silently stop showing the tail the
  // next time a lever moves the count.
  std::printf("labels[0]   = \"%s\"\nlabels[1]   = \"%s\"\nlabels[%zu] = \"%s\"\n",
              instr.labels[0].c_str(), instr.labels[1].c_str(), instr.labels.size() - 1,
              instr.labels.back().c_str());

  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  runtime::Control* c = b.control.as<runtime::Control>();

  // --- the raw decode loop (spec §8.4/§8.5), as replay_determinism_test -----
  // Every event is reset before EVERY replay: re-signalling an un-reset event
  // is undefined in Level Zero, and this test replays the profiled list 25
  // times. The reset is host-side and happens before `execute`, so it is not
  // part of what the fence wall measures.
  auto reset_events = [&]() {
    for (l0::Event& e : prof.events) e.reset();
  };
  auto step = [&](runtime::CapturedStep& s, bool profiled) {
    // The engine-layer precondition every attention kernel's bound assumes;
    // here this loop is the caller, so the check is here (Task 5 ruling).
    CHECK(size_t(c->pos) + size_t(c->n_active) <= size_t(b.max_len));
    if (profiled) reset_events();
    q.execute(s.list, &fence);
    fence.wait();
  };
  auto ingest = [&](runtime::CapturedStep& s, bool profiled) {
    c->n_active = 1;
    for (uint32_t id : kPrompt) {
      c->cur_token[0] = id;
      step(s, profiled);
    }
    CHECK_EQ(c->pos, uint32_t(kPromptLen));
  };
  // Step g embeds the token the previous step sampled, so `cur_token[0]` read
  // before the step IS generated token g.
  auto generate = [&](runtime::CapturedStep& s, bool profiled, uint32_t* ids) {
    for (int g = 0; g < kGen; ++g) {
      ids[g] = c->cur_token[0];
      step(s, profiled);
    }
  };
  // Only the persistent five. Scratch is left dirty on purpose - the profiled
  // run therefore starts from the plain run's leftover bytes and still has to
  // produce the same tokens (replay_determinism_test's run-C bar).
  auto zero_state = [&]() {
    for (l0::Mem* mm : {&b.control, &b.gdn_state, &b.conv_ring, &b.kv_k, &b.kv_v})
      imm.fill(mm->ptr(), 0u, mm->size());
  };

  // --- A: the plain list, then the profiled list ---------------------------
  uint32_t ids_plain[kGen] = {};
  zero_state();
  ingest(plain, false);
  generate(plain, false, ids_plain);
  CHECK_EQ(c->pos, uint32_t(kPromptLen + kGen));

  uint32_t ids_prof[kGen] = {};
  zero_state();
  ingest(instr, true);
  generate(instr, true, ids_prof);
  CHECK_EQ(c->pos, uint32_t(kPromptLen + kGen));

  std::printf("plain ids:   ");
  for (uint32_t id : ids_plain) std::printf(" %u", id);
  std::printf("\nprofiled ids:");
  for (uint32_t id : ids_prof) std::printf(" %u", id);
  std::printf("\n");
  for (int g = 0; g < kGen; ++g) {
    if (ids_plain[g] != ids_prof[g])
      std::fprintf(stderr, "token %d: plain %u, profiled %u\n", g, ids_plain[g], ids_prof[g]);
    CHECK_EQ(ids_plain[g], ids_prof[g]);
    CHECK(ids_plain[g] < Qwen35::kVocabUsed);
  }

  // --- the measured replay -------------------------------------------------
  // One more profiled step, with the host clock around exactly the submission
  // and the fence wait. Nothing is queried before the fence: `duration_us()`
  // throws on an unsignalled event (ZE_RESULT_NOT_READY), by design.
  reset_events();
  const auto t0 = std::chrono::steady_clock::now();
  q.execute(instr.list, &fence);
  fence.wait();
  const double wall_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

  // --- B: every launch ran, and they tile the step in walk order -----------
  const uint64_t mask = prof.pool.calib().mask();
  double sum_us = 0.0;
  size_t slowest = 0;
  double slowest_us = 0.0;
  for (size_t i = 0; i < prof.events.size(); ++i) {
    const double d = prof.events[i].duration_us();
    if (!(d > 0.0)) std::fprintf(stderr, "launch %zu (%s): %g us\n", i, instr.labels[i].c_str(), d);
    CHECK(d > 0.0);
    sum_us += d;
    if (d > slowest_us) {
      slowest_us = d;
      slowest = i;
    }
  }
  // In-order list => kernel i ends before kernel i+1 starts. Compared wrap-
  // safely: the forward distance between two kernels microseconds apart is
  // tiny, while an overlap of even one tick underflows the mask and lands near
  // the top of the counter's range (tests/l0/event_test.cc, comparison 4).
  Raw prev = raw_global(prof.events[0], mask);
  double max_gap_us = 0.0, total_gap_us = 0.0;
  for (size_t i = 1; i < prof.events.size(); ++i) {
    const Raw cur = raw_global(prof.events[i], mask);
    const double gap_us =
        static_cast<double>((cur.start - prev.end) & mask) / prof.pool.calib().cycles_per_us;
    if (!(gap_us < 1e6)) {
      std::fprintf(stderr, "launch %zu (%s) starts before launch %zu (%s) ends: gap %g us\n", i,
                   instr.labels[i].c_str(), i - 1, instr.labels[i - 1].c_str(), gap_us);
      CHECK(false);
    }
    if (gap_us > max_gap_us) max_gap_us = gap_us;
    total_gap_us += gap_us;
    prev = cur;
  }

  // --- C: the sum against the wall, and the gap ----------------------------
  std::printf("sum of %zu kernel durations: %.3f us   fence wall: %.3f us   ratio %.4f\n",
              prof.events.size(), sum_us, wall_us, sum_us / wall_us);
  std::printf("slowest launch: %s  %.3f us\n", instr.labels[slowest].c_str(), slowest_us);
  CHECK(sum_us <= 1.001 * wall_us);
  // The first in-situ dispatch measurement this project has. It is NOT a
  // dispatch cost an unprofiled list pays: every signal here carries
  // ZE_EVENT_SCOPE_FLAG_HOST, a per-launch flush after kernelEnd, measured at
  // 1.771 us of inter-kernel gap on a two-noop list against ~1.15 us for the
  // same list unprofiled (Task 1). Read this number as an upper bound.
  std::printf("dispatch+gap: %.3f us\n", wall_us - sum_us);
  std::printf("  (upper bound: each profiled launch signals with "
              "ZE_EVENT_SCOPE_FLAG_HOST, a per-launch flush an unprofiled list never pays - "
              "1.771 us vs ~1.15 us of gap on the two-noop list of Task 1)\n");
  std::printf("  inter-kernel gaps: %.3f us total over %zu boundaries, largest %.3f us\n",
              total_gap_us, prof.events.size() - 1, max_gap_us);

  std::puts("profile_capture_test OK");
  return 0;
}
