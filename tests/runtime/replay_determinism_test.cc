// replay_determinism_test - spec §8.7, and the acceptance for every kernel in
// plan 3: the captured decode list, replayed, is a pure function of the state
// it reads. A kernel that breaks this is rejected whatever else it does
// (doc 04's capture-safety rule).
//
// Three comparisons, in order of strength:
//
//   1. **Same list, same state, twice.** Ingest a fixed 16-id prompt, snapshot
//      every persistent buffer, generate 8 tokens recording the ids and the
//      per-layer residual tap, restore the snapshot, generate 8 again. The two
//      token sequences, the two residual traces, the two *final* states and the
//      last step's `logits` and `x` must be bitwise identical. This is what
//      rules out a data-dependent reduction order, an atomic, or a work-group
//      count that varies with arrival order. The `logits`/`x` pair is what
//      carries the bar past *argmax resolution*: the residual tap ends at layer
//      63's input, so without it the only witness to layer 63's MLP, the final
//      norm and lm_head is an integer arg-max that a low-bit difference in all
//      248320 logits need not move.
//   2. **Fresh-process equivalent.** Re-zero the persistent state, re-ingest
//      the same prompt from pos = 0 and generate again: same ids, same taps,
//      and the same five persistent buffers at the end - the state token 9
//      would read. Deliberately, only control/gdn_state/conv_ring/kv are
//      re-zeroed - this run starts from run 2's leftover *scratch*, different
//      bytes than run 1's fresh allocations. Identical ids and taps therefore
//      prove no step reads scratch it has not first written. Do NOT "clean this
//      up" by zeroing scratch too: that would delete exactly the coverage this
//      run adds.
//   3. **Structure.** kernel_count == 774 (spec §9.1 as amended, plus spec
//      1.5's lever L1), one module per distinct variant, every generated id <
//      kVocabUsed, and no NaN or Inf anywhere in the residual trace.
//
// It does NOT judge output quality - whether the continuation reads like
// English is Task 8's golden gate, which compares against the oracle. What is
// asserted here is that whatever the engine computes, it computes it the same
// way every time.
//
// Label `checkpoint`: needs the real 19 GB checkpoint and a B70.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
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

// The first 16 ids of tests/golden/prompts/prose.txt under the checkpoint's own
// tokenizer - "The harbour at dawn was quiet enough to hear the ropes creak
// against the bollards" (tools/oracle/tokenize.py, oracle-out/prose.ids,
// 2026-08-25). Literal here on purpose: Task 8 is what commits the .ids files,
// and this test must not wait on it. If the committed prose.ids ever disagrees
// with this prefix, this list is the one to change.
constexpr uint32_t kPrompt[] = {760, 72103, 506, 37119, 557, 11012, 3213, 310,
                                6512, 279, 61789, 272, 1072, 2272, 279, 197616};
constexpr size_t kPromptLen = sizeof(kPrompt) / sizeof(kPrompt[0]);

constexpr int kGen = 8;             // generated tokens per run
constexpr uint32_t kCapM = 1;       // the captured list's M (only M = 1 compiles)
// The residual tap: bf16 [64 layers][M][5120], one buffer per step.
// Spec 14: the layer count is the loaded model's, set in main() before anything reads it.
uint32_t kLayers = 0;
uint32_t kHidden = 0;   // the loaded descriptor's (spec 15b), set beside kLayers
size_t kTapElems = 0, kTapBytes = 0;

// Everything that survives a token boundary (runtime::DecodeBuffers' first
// group) - the whole of what a replay is allowed to depend on - plus the two
// scratch buffers at the *end* of a step, which are snapshotted but never
// restored.
//
// The scratch pair is what raises the run-B bar past argmax resolution.
// Comparing the generated ids compares `argmax(logits)`, an integer: two runs
// whose layer-63 MLP, final norm and lm_head differed in the low bits of every
// logit would still agree on the arg of the max and this test would report
// bitwise determinism. `logits` is fp32 [M][248320] and `x` is the last thing
// prep_res_norm wrote before it, so comparing both compares that tail of the
// step at full resolution. (The per-layer tap ends at layer 63's *input*
// residual, so it does not reach either.)
struct State {
  std::vector<uint8_t> gdn_state, conv_ring, kv_k, kv_v, control;
  std::vector<uint8_t> logits, x;   // scratch: snapshotted, never restored
};

struct Run {
  uint32_t ids[kGen] = {};
  std::vector<uint16_t> tap = std::vector<uint16_t>(size_t(kGen) * kTapElems);
};

void same_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const char* what) {
  CHECK_EQ(a.size(), b.size());
  size_t diff = 0, first = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) {
      if (diff == 0) first = i;
      ++diff;
    }
  if (diff != 0)
    std::fprintf(stderr, "%s: %zu of %zu bytes differ, first at byte %zu\n", what, diff, a.size(),
                 first);
  CHECK_EQ(diff, size_t(0));
}

// bf16 carries fp32's exponent field: all ones is Inf or NaN either way.
bool bf16_finite(uint16_t w) { return (w & 0x7F80u) != 0x7F80u; }
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  // argv[2] (spec 9): the lm_head form, bf16 (the checkpoint's; default) or int8.
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  if (argc > 2) CHECK(loader::parse_lm_head_form(argv[2], lm_head));
  std::printf("lm_head: %s\n", loader::lm_head_form_name(lm_head));
  l0::Context ctx(0);
  loader::LoadedModel m = loader::load(ctx, arg, 16384, /*mtp=*/false, lm_head);
  kLayers = m.desc->layers;
  kHidden = m.desc->hidden;
  kTapElems = size_t(kLayers) * kCapM * kHidden;
  kTapBytes = kTapElems * 2;
  runtime::DecodeBuffers b(ctx, m.max_len, *m.desc);
  l0::Mem tapmem(ctx, l0::MemKind::Device, kTapBytes);

  runtime::CapturedStep cap = runtime::build(ctx, m, b, &tapmem);
  std::printf("captured: %zu kernels, %zu modules, max_len %u, persistent %.2f GB\n",
              cap.kernel_count, cap.modules.size(), b.max_len,
              b.persistent_bytes() / 1e9);
  // 48 GDN layers x 12 + 16 FA layers x 12 + 6 token-boundary kernels
  // (embed_gather, the final norm's two launches, lm_head, argmax x2) - spec
  // §9.1's 645 (x 10, x 10, 5) plus the 129 launches spec 1.5's lever L1 added
  // by splitting every `prep_res_norm` site into `prep_res_fold` +
  // `prep_norm_finish`. A layer holds two of those sites (the input/pre-mixer
  // norm and the MLP's post norm) so it gains 2, and the boundary holds one
  // (the final norm) so it gains 1: 129 sites = 2 x 64 + 1.
  // 576 + 192 + 6 = 774 = 645 + 129.
  CHECK_EQ(cap.kernel_count, size_t(12) * kLayers + 6);   // 774 Qwen3.8, 870 Agnes (spec 14)
  // One Module per distinct variant: embed_gather, 2 prep_res_fold (SP0/SP16),
  // 1 prep_norm_finish (it does not read `partials`, so the two prep modes
  // share it), silu_mul, gated_head, gdn_step, 3 attn, 5 int4 gemv, 2 bf16
  // gemv, 2 argmax. That is 19 - the 18 of plan 3 with the 2 `prep_res_norm`
  // modules replaced by the pair's 3.
  //
  // **19 on BOTH checkpoints, and that is arithmetic rather than luck**
  // (spec 1.6 §5.1). An int4 `lm_head` swaps one module for another -
  // `gemv_bf16_M1_K5120_N248320` out, `gemv_M1_K5120_N248320_S1_L1` in - so
  // the bf16 gemv count falls 2 -> 1 and the int4 gemv count rises 5 -> 6.
  // Nothing else in the walk depends on the checkpoint, so if this ever fires
  // on one checkpoint and not the other, the swap is not one-for-one any more
  // and the reason belongs in the message, not in a widened number.
  // Spec 10 (plan 10b): decode attention v2 is ONE module (attn_decode_v2 and
  // attn_reduce_v2 share attn_v2_M1_T32), so the v2 walk has 18; the kernel count is
  // unchanged (two launches per FA layer either way).
  // Spec 9: the int8 head (`--lm-head int8`) is the same one-for-one swap, to
  // `gemv_i8w_M1_K5120_N248320`, so the count stays 19 (v1) / 18 (v2) on every head form.
  const bool attn_v2 = runtime::decode_attn() == runtime::DecodeAttn::V2;
  std::printf("decode attention: %s\n", runtime::decode_attn_name(runtime::decode_attn()));
  CHECK_EQ(cap.modules.size(), size_t(attn_v2 ? 18 : 19));
  CHECK_EQ(cap.modules.count(attn_v2 ? "attn_v2_M1_T32" : "attn_decode_M1_L16384_B64"), size_t(1));
  {
    const model::WeightKind kind =
        m.linears.at({loader::kTopLevel, model::LinearId::LmHead}).kind;
    const char* const all[] = {"gemv_M1_K5120_N248320_S1_L1", "gemv_bf16_M1_K5120_N248320",
                               "gemv_i8w_M1_K5120_N248320"};
    const size_t pick = kind == model::WeightKind::Int4 ? 0 : kind == model::WeightKind::Bf16 ? 1 : 2;
    std::printf("lm_head: %s -> module %s\n",
                pick == 0 ? "int4 g64" : pick == 1 ? "bf16" : "int8", all[pick]);
    for (size_t i = 0; i < 3; ++i) CHECK_EQ(cap.modules.count(all[i]), size_t(i == pick ? 1 : 0));
  }

  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  runtime::Control* c = b.control.as<runtime::Control>();

  // --- the raw decode loop (spec §8.4/§8.5) ---------------------------------
  // Task 7's Engine wraps exactly this; the test drives it itself so this task
  // stands alone. Note where the engine-layer precondition lives: every
  // attention kernel's bound assumes `pos + n_active <= max_len`, and the
  // *caller* of the list is what guarantees it (Task 5 ruling). Here the caller
  // is this loop, so the check is here.
  auto step = [&]() {
    CHECK(size_t(c->pos) + size_t(c->n_active) <= size_t(b.max_len));
    q.execute(cap.list, &fence);
    fence.wait();
  };
  // Prompt ingestion: one step per id, the host writing four bytes of shared
  // memory between replays and nothing else. After the last one `cur_token[0]`
  // holds the first generated token - argmax_stage2 put it there.
  auto ingest = [&]() {
    c->n_active = 1;
    for (uint32_t id : kPrompt) {
      c->cur_token[0] = id;
      step();
    }
    CHECK_EQ(c->pos, uint32_t(kPromptLen));
  };
  // Generation: the host writes nothing at all. Step g embeds the token the
  // previous step sampled, so `cur_token[0]` read before the step IS generated
  // token g; the tap it leaves behind is that token's residual trace.
  auto generate = [&](Run& r) {
    for (int g = 0; g < kGen; ++g) {
      r.ids[g] = c->cur_token[0];
      step();
      imm.copy(r.tap.data() + size_t(g) * kTapElems, tapmem.ptr(), kTapBytes);
    }
  };
  auto snapshot = [&]() {
    State s;
    auto rd = [&](std::vector<uint8_t>& d, const l0::Mem& src) {
      d.resize(src.size());
      imm.copy(d.data(), src.ptr(), src.size());
    };
    rd(s.gdn_state, b.gdn_state);
    rd(s.conv_ring, b.conv_ring);
    rd(s.kv_k, b.kv_k);
    rd(s.kv_v, b.kv_v);
    rd(s.control, b.control);
    rd(s.logits, b.logits);
    rd(s.x, b.x);
    return s;
  };
  // Only the persistent five are written back. `logits` and `x` are deliberately
  // left as run A left them - restoring scratch would delete run B's other job,
  // which is to start from *different* scratch bytes than run A's fresh
  // allocations and still produce the same tokens.
  auto restore = [&](const State& s) {
    auto wr = [&](const std::vector<uint8_t>& src, l0::Mem& d) {
      CHECK_EQ(src.size(), d.size());
      imm.copy(d.ptr(), src.data(), src.size());
    };
    wr(s.gdn_state, b.gdn_state);
    wr(s.conv_ring, b.conv_ring);
    wr(s.kv_k, b.kv_k);
    wr(s.kv_v, b.kv_v);
    wr(s.control, b.control);
  };
  auto zero_state = [&]() {
    for (l0::Mem* mm : {&b.control, &b.gdn_state, &b.conv_ring, &b.kv_k, &b.kv_v})
      imm.fill(mm->ptr(), 0u, mm->size());
  };

  // --- run A: ingest, snapshot, generate ------------------------------------
  ingest();
  const State after_ingest = snapshot();
  Run a;
  const auto t0 = std::chrono::steady_clock::now();
  generate(a);
  const double gen_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const State final_a = snapshot();
  CHECK_EQ(c->pos, uint32_t(kPromptLen + kGen));

  std::printf("run A ids:");
  for (uint32_t id : a.ids) std::printf(" %u", id);
  std::printf("   (%.1f ms/token over %d generated tokens)\n", gen_ms / kGen, kGen);

  // Structure: an id the tokenizer cannot spell is an engine bug, not a rare
  // token (argmax masks at kVocabUsed - src/kernels/argmax.cl).
  for (uint32_t id : a.ids) CHECK(id < Qwen35::kVocabUsed);
  // And nothing in the residual stream may be NaN or Inf: the kernels' rounding
  // helpers do not handle either, so one would poison every later layer.
  for (int g = 0; g < kGen; ++g)
    for (uint32_t l = 0; l < kLayers; ++l)
      for (uint32_t k = 0; k < kHidden; ++k) {
        const uint16_t w = a.tap[(size_t(g) * kLayers + l) * kHidden + k];
        if (!bf16_finite(w)) {
          std::fprintf(stderr, "non-finite resid: token %d layer %u element %u = 0x%04X\n", g, l, k,
                       w);
          CHECK(false);
        }
      }

  // --- run B: restore the post-ingest state, generate again ------------------
  restore(after_ingest);
  Run bb;
  generate(bb);
  const State final_b = snapshot();

  for (int g = 0; g < kGen; ++g) {
    if (a.ids[g] != bb.ids[g])
      std::fprintf(stderr, "token %d: run A %u, run B %u\n", g, a.ids[g], bb.ids[g]);
    CHECK_EQ(a.ids[g], bb.ids[g]);
  }
  // Bitwise on the tap, compared per (token, layer) so a failure names the
  // first layer that diverged - the diagnostic this tap exists for.
  for (int g = 0; g < kGen; ++g)
    for (uint32_t l = 0; l < kLayers; ++l) {
      const size_t off = (size_t(g) * kLayers + l) * kHidden;
      if (std::memcmp(a.tap.data() + off, bb.tap.data() + off, kHidden * 2) != 0) {
        std::fprintf(stderr, "resid tap differs at generated token %d, layer %u\n", g, l);
        CHECK(false);
      }
    }
  same_bytes(final_a.gdn_state, final_b.gdn_state, "gdn_state");
  same_bytes(final_a.conv_ring, final_b.conv_ring, "conv_ring");
  same_bytes(final_a.kv_k, final_b.kv_k, "kv_k");
  same_bytes(final_a.kv_v, final_b.kv_v, "kv_v");
  same_bytes(final_a.control, final_b.control, "control");
  // The step's tail at full resolution rather than at argmax resolution: the
  // 248320 fp32 logits the last token produced, and the normalised row lm_head
  // read to produce them. Nothing else in this test can see a difference in
  // layer 63's MLP, the final norm or lm_head that does not move the arg of the
  // max (see State).
  same_bytes(final_a.logits, final_b.logits, "logits");
  same_bytes(final_a.x, final_b.x, "x");

  // --- run C: fresh-process equivalent --------------------------------------
  zero_state();
  ingest();
  Run cc;
  generate(cc);
  const State final_c = snapshot();
  for (int g = 0; g < kGen; ++g) {
    if (a.ids[g] != cc.ids[g])
      std::fprintf(stderr, "token %d: run A %u, fresh run %u\n", g, a.ids[g], cc.ids[g]);
    CHECK_EQ(a.ids[g], cc.ids[g]);
  }
  for (int g = 0; g < kGen; ++g)
    for (uint32_t l = 0; l < kLayers; ++l) {
      const size_t off = (size_t(g) * kLayers + l) * kHidden;
      if (std::memcmp(a.tap.data() + off, cc.tap.data() + off, kHidden * 2) != 0) {
        std::fprintf(stderr, "fresh-ingest resid differs at generated token %d, layer %u\n", g, l);
        CHECK(false);
      }
    }
  // And the state the fresh run *ends* in, which run C did not check at all
  // before: identical ids and taps say the visible outputs agree, but the KV
  // cache, the 48 GDN recurrent states, the conv rings and the control block are
  // what the *next* token would read. A fresh-ingest path that reproduced 8
  // tokens and left one of those different would be a bug that only appeared at
  // token 9. All five persistent buffers, byte for byte, against run A's final
  // state.
  same_bytes(final_a.gdn_state, final_c.gdn_state, "fresh-ingest gdn_state");
  same_bytes(final_a.conv_ring, final_c.conv_ring, "fresh-ingest conv_ring");
  same_bytes(final_a.kv_k, final_c.kv_k, "fresh-ingest kv_k");
  same_bytes(final_a.kv_v, final_c.kv_v, "fresh-ingest kv_v");
  same_bytes(final_a.control, final_c.control, "fresh-ingest control");

  std::printf("replay_determinism_test OK (%d tokens x 3 runs bitwise identical)\n", kGen);
  return 0;
}
