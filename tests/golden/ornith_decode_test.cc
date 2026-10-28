// ornith_decode_test - spec 15c's decode gates on the card that are not the golden gate
// itself (R3 is golden_gate_test on oracle-out-ornith):
//
//   1. the list: Ornith's captured decode step is decode_launches() = 526 launches
//      (embed + 40 x 13 + 5), every MoE layer's four launches in place of the dense MLP;
//   2. replay determinism, bitwise: one prompt ingested and 16 tokens generated twice,
//      reset between - the generated ids, the final logits row, every layer's router
//      logits and route rows (ids, weights, gate) and the GDN state all identical. The
//      expert sum is in fixed slot order and no sum anywhere is atomic (spec 15 §4.2);
//   3. R2, routing (spec 15 §5): on every prompt position (and the teacher-forced
//      generated ones the dump covers), every layer's engine top-8 expert SET equals
//      the reference's, computed by the engine's own formula (tests/kernels/moe_ref.h)
//      from the oracle's router logits - except where the reference's 8th and 9th
//      probabilities are within kR2TieRel of each other (a near-tie: the boundary
//      expert may be either); the weights of the matched experts within kR2WeightRel.
//      Needs `router_logits.L{l}` [rows][256] (and `shared_gate_logits.L{l}` [rows][1]
//      for the shared gate) in the golden dumps - spec 15a's oracle writes them
//      beside the usual tensors; without them R2 is skipped with a message.
//      With `--judge=<dir>` (operator ruling 2026-10-23: Ornith's gates are judged by the
//      f32 teacher-forced reference, tools/box_validate/f32_oracle.sh) the reference route
//      is the f32 dump's: its router logits (F32) through the same formula, teacher-forced on
//      the same tokens, and a near-tie is an 8th / 9th LOGIT gap within one bf16 ulp of the
//      8th (golden_common.h's bf16_ulp_at: the engine rounds router logits to bf16 before the
//      softmax, so a smaller gap may round either way). Beside it the bf16 golden's own
//      routes are judged by the same rule - the reference's distance from exact arithmetic.
//
// argv: <snapshot> <oracle dir> <prompt ids dir> [lm_head form] [--judge=<dir>]. Exit 77 (SKIP) when
// the Ornith int4 checkpoint (urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ, spec 15 §13)
// is not on this machine - the Mac and a box without it skip, they do not fail.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/capture.h"
#include "runtime/engine.h"
#include "golden_common.h"
#include "kernels/moe_ref.h"

namespace {

// R2's tolerances (spec 15 §5): PROPOSED here, to be set from 15a's measured 8th / 9th
// router-probability gaps (its Task 3) before the gate is binding.
constexpr float kR2TieRel = 1e-3f;      // (p8 - p9) / p8 below this: a near-tie
constexpr float kR2WeightRel = 1e-3f;   // matched experts' weights, relative
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGenDeterminism = 16;
const char* const kPrompts[] = {"prose", "code", "cjk"};

struct Snapshot {   // what one replay leaves on the card
  std::vector<uint32_t> tokens;
  std::vector<float> logits;
  std::vector<uint8_t> moe, gdn;
  bool operator==(const Snapshot& o) const {
    return tokens == o.tokens && logits == o.logits && moe == o.moe && gdn == o.gdn;
  }
};

Snapshot run_once(runtime::Engine& eng, l0::CmdList& imm, const std::vector<uint32_t>& ids) {
  eng.reset();
  eng.ingest(ids);
  Snapshot s;
  s.tokens = eng.generate(kGenDeterminism);
  s.logits.resize(model::Qwen35::kVocab);
  imm.copy(s.logits.data(), eng.buffers().logits.ptr(), s.logits.size() * 4);
  s.moe.resize(eng.buffers().moe->size());
  imm.copy(s.moe.data(), eng.buffers().moe->ptr(), s.moe.size());
  s.gdn.resize(eng.buffers().gdn_state.size());
  imm.copy(s.gdn.data(), eng.buffers().gdn_state.ptr(), s.gdn.size());
  return s;
}

struct R2Count {
  uint64_t rows = 0, exact_sets = 0, near_ties = 0, bad_sets = 0, bad_weights = 0;
  float worst_w = 0;
  uint64_t ref_bad = 0;   // --judge: the bf16 golden's own sets that differ from the judge's beyond the rule
};

// Logits row `r` of layer `l` (the experts, then the shared gate): the f32 judge's (F32) or the golden's (BF16).
std::vector<float> router_row(const golden::Golden& g, bool f32, const model::ModelDesc& d, uint32_t l, uint32_t r,
                              uint32_t rows) {
  const model::MoeDesc& md = d.moe;
  const std::string ls = std::to_string(l);
  std::vector<float> lg(md.router_n(), 0.0f);
  const bool have_sg = g.has("shared_gate_logits.L" + ls);
  if (f32) {
    const float* gl = g.f32("router_logits.L" + ls, size_t(rows) * md.experts);
    std::copy(gl + size_t(r) * md.experts, gl + size_t(r + 1) * md.experts, lg.begin());
    if (have_sg) lg[md.experts] = g.f32("shared_gate_logits.L" + ls, rows)[r];
  } else {
    const uint16_t* gl = g.bf16("router_logits.L" + ls, size_t(rows) * md.experts);
    for (uint32_t e = 0; e < md.experts; ++e) lg[e] = common::bf16_to_f32(gl[size_t(r) * md.experts + e]);
    if (have_sg) lg[md.experts] = common::bf16_to_f32(g.bf16("shared_gate_logits.L" + ls, rows)[r]);
  }
  return lg;
}

// The judge's near-tie: the top_k-th and (top_k + 1)-th f32 logits within one bf16 ulp of the former.
bool ulp_near_tie(const std::vector<float>& lg, uint32_t experts, uint32_t top_k) {
  std::vector<float> s(lg.begin(), lg.begin() + experts);
  std::nth_element(s.begin(), s.begin() + top_k, s.end(), std::greater<float>());
  const float l9 = s[top_k];
  const float l8 = *std::min_element(s.begin(), s.begin() + top_k);
  return double(l8) - double(l9) <= golden::bf16_ulp_at(l8);
}

// Two top_k sets (ids by rank): equal, or at a near-tie all but the boundary expert equal, or not.
enum class SetCmp { Exact, Tie, Bad };
SetCmp compare_sets(const uint32_t* got, const uint32_t* ref, uint32_t top_k, bool near_tie) {
  std::vector<uint32_t> gs(got, got + top_k), rs(ref, ref + top_k);
  std::sort(gs.begin(), gs.end());
  std::sort(rs.begin(), rs.end());
  if (gs == rs) return SetCmp::Exact;
  if (!near_tie) return SetCmp::Bad;
  std::vector<uint32_t> g7(got, got + top_k - 1), r7(ref, ref + top_k - 1);
  std::sort(g7.begin(), g7.end());
  std::sort(r7.begin(), r7.end());
  return g7 == r7 ? SetCmp::Tie : SetCmp::Bad;
}

// One position: every layer's engine route row against the reference route of the
// oracle's logits row `r` of that layer (with a judge: the f32 dump's, and the bf16
// golden `g` judged beside the engine).
void r2_position(const golden::Golden& g, const golden::Golden* judge, const std::vector<uint8_t>& moe,
                 const runtime::MoeScratchLayout& ml, const model::ModelDesc& d, uint32_t r,
                 uint32_t rows, R2Count& c, const char* pname, uint32_t pos) {
  const model::MoeDesc& md = d.moe;
  const moe_ref::Shape s{md.experts, md.top_k, d.hidden, md.expert_intermediate, md.router_n()};
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::vector<float> lg = router_row(judge ? *judge : g, judge != nullptr, d, l, r, rows);
    const moe_ref::Route want = moe_ref::route(lg.data(), s);
    const uint32_t* row = reinterpret_cast<const uint32_t*>(moe.data() + ml.route_at(l));   // m = 0
    ++c.rows;
    const float pk = want.p[md.top_k - 1];
    const bool near_tie = judge ? ulp_near_tie(lg, md.experts, md.top_k) : (pk - want.p9) <= kR2TieRel * pk;
    if (judge) {
      const moe_ref::Route b = moe_ref::route(router_row(g, false, d, l, r, rows).data(), s);
      if (compare_sets(b.ids, want.ids, md.top_k, near_tie) == SetCmp::Bad) ++c.ref_bad;
    }
    std::vector<uint32_t> got(row + moe_ref::kIds, row + moe_ref::kIds + md.top_k);
    std::vector<uint32_t> ref(want.ids, want.ids + md.top_k);
    const SetCmp cmp = compare_sets(got.data(), ref.data(), md.top_k, near_tie);
    if (cmp == SetCmp::Exact) {
      ++c.exact_sets;
    } else if (near_tie) {
      ++c.near_ties;   // the boundary expert may be either: the other 7 must agree
      if (cmp == SetCmp::Bad) {
        ++c.bad_sets;
        if (c.bad_sets <= 10)
          std::printf("  R2 %s pos %u L%u: top-%u differs beyond the near-tie\n", pname, pos, l,
                      md.top_k - 1);
      }
      continue;
    } else {
      ++c.bad_sets;
      if (c.bad_sets <= 10)
        std::printf("  R2 %s pos %u L%u: expert set differs (8th/9th gap %.3e relative)\n", pname,
                    pos, l, double((pk - want.p9) / pk));
      continue;
    }
    for (uint32_t k = 0; k < md.top_k; ++k) {   // weights of the same expert, by id
      const uint32_t e = got[k];
      const uint32_t j = uint32_t(std::find(ref.begin(), ref.end(), e) - ref.begin());
      float w;
      std::memcpy(&w, &row[moe_ref::kWeights + k], 4);
      const float rel = std::fabs(w - want.w[j]) / std::max(want.w[j], 1e-6f);
      c.worst_w = std::max(c.worst_w, rel);
      if (rel > kR2WeightRel) ++c.bad_weights;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string judge_dir;   // --judge=<dir>, anywhere after the positional arguments
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind("--judge=", 0) == 0) {
      judge_dir = argv[i] + 8;
      for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
      --argc;
      break;
    }
  const std::string snap = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-ornith";
  const std::string pdir = argc > 3 ? argv[3] : "tests/golden/prompts";
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  if (argc > 4) CHECK(loader::parse_lm_head_form(argv[4], lm_head));
  try {
    (void)loader::resolve_snapshot(snap);
  } catch (const std::exception& e) {
    std::printf("SKIP: no Ornith int4 checkpoint at '%s' (%s). None is published; spec 15a "
                "decides its source (AutoRound g64 proposed) - see the box validation queue.\n",
                snap.c_str(), e.what());
    return 77;
  }

  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, lm_head);
  CHECK(model.desc->is_moe());
  const model::ModelDesc& d = *model.desc;
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // ---- 1. the list ----------------------------------------------------------------
  CHECK_EQ(runtime::decode_launches(d), size_t(526));
  CHECK_EQ(eng.step().kernel_count, size_t(526));
  CHECK(eng.buffers().moe != nullptr);
  const runtime::MoeScratchLayout ml = runtime::moe_scratch_layout(d);
  CHECK(eng.buffers().moe->size() == ml.total);
  std::printf("list: %zu launches, %zu modules (%u layers x 13 + 6)\n", eng.step().kernel_count,
              eng.step().modules.size(), d.layers);
  // The planner (decode only - Ornith's prefill is 15d) against what this engine holds:
  // the loader's MoE bytes are loader::moe_bytes, the MoE scratch moe_scratch_layout's.
  {
    runtime::PrefillPath decode_only;
    decode_only.prefill = false;
    const loader::LoadReport& rep = eng.model().report;
    const runtime::MemoryPlan p =
        runtime::plan(d, kMaxLen, false, rep.total() - rep.rope_bytes, decode_only);
    const runtime::MemoryComponents u = eng.memory_use();
    std::printf("%s\nplan: %s\n", eng.memory_line().c_str(),
                runtime::describe(p, ctx.memory_bytes(), 0).c_str());
    CHECK_EQ(u.model, p.model);
    CHECK_EQ(u.kv, p.kv);
    CHECK_EQ(u.decode_state, p.decode_state);
    CHECK_EQ(rep.moe_bytes, p.moe_weights);
    CHECK_EQ(u.prefill_scratch, size_t(0));
  }

  // ---- 2. replay determinism ------------------------------------------------------
  const std::vector<uint32_t> ids = golden::read_ids(pdir + "/prose.ids");
  const Snapshot a = run_once(eng, imm, ids), b = run_once(eng, imm, ids);
  CHECK(a.tokens.size() == kGenDeterminism);
  if (!(a == b)) {
    std::fprintf(stderr, "replay determinism FAILED: tokens %s, logits %s, MoE scratch %s, GDN %s\n",
                 a.tokens == b.tokens ? "same" : "DIFFER", a.logits == b.logits ? "same" : "DIFFER",
                 a.moe == b.moe ? "same" : "DIFFER", a.gdn == b.gdn ? "same" : "DIFFER");
    return 1;
  }
  std::printf("replay: two runs of %zu ids + %u tokens bitwise identical (ids, logits, every "
              "layer's router logits and route rows, GDN state)\n",
              ids.size(), kGenDeterminism);

  // ---- 3. R2 ------------------------------------------------------------------------
  R2Count c;
  bool any = false;
  for (const char* pname : kPrompts) {
    const std::string gpath = gdir + "/" + pname + ".golden.safetensors";
    if (!golden::exists(gpath)) {
      std::printf("R2 SKIPPED for %s: %s is absent (spec 15a's golden set, oracle-out-ornith)\n",
                  pname, gpath.c_str());
      continue;
    }
    golden::Golden g(gpath);
    if (!g.has("router_logits.L0")) {
      std::printf("R2 SKIPPED for %s: the dump has no router_logits.L* (15a's oracle must write "
                  "them, [rows][%u] bf16, beside shared_gate_logits.L* [rows][1])\n",
                  pname, d.moe.experts);
      continue;
    }
    any = true;
    const std::vector<uint32_t> pids = golden::read_ids(pdir + "/" + pname + ".ids");
    const uint32_t rows = uint32_t(g.dim("router_logits.L0", 2, 0));
    CHECK(rows >= pids.size());
    std::vector<uint32_t> feed = pids;   // the prompt, then the oracle's tokens teacher-forced
    if (g.has("tokens")) {
      const uint32_t ng = uint32_t(g.dim("tokens", 1, 0));
      const int32_t* gt = g.i32("tokens", ng);
      for (uint32_t i = 0; i < ng && feed.size() < rows; ++i) feed.push_back(uint32_t(gt[i]));
    }
    // The judge decides the same contexts: teacher-forced on exactly the golden's tokens.
    std::unique_ptr<golden::Golden> J;
    if (!judge_dir.empty()) {
      const std::string jpath = judge_dir + "/" + pname + ".golden.safetensors";
      if (!golden::exists(jpath)) {
        std::printf("SKIP: the f32 judge %s is absent (tools/box_validate/f32_oracle.sh)\n", jpath.c_str());
        return 77;
      }
      J = std::make_unique<golden::Golden>(jpath);
      CHECK(J->has("router_logits.L0"));
      CHECK_EQ(J->dim("router_logits.L0", 2, 0), uint64_t(rows));
      const uint32_t ng = uint32_t(g.dim("tokens", 1, 0));
      CHECK_EQ(J->dim("tokens", 1, 0), uint64_t(ng));
      CHECK(std::memcmp(J->i32("tokens", ng), g.i32("tokens", ng), size_t(ng) * 4) == 0);
    }
    eng.reset();
    std::vector<uint8_t> moe(eng.buffers().moe->size());
    for (uint32_t p = 0; p < feed.size(); ++p) {
      eng.ingest({feed[p]});
      imm.copy(moe.data(), eng.buffers().moe->ptr(), moe.size());
      r2_position(g, J.get(), moe, ml, d, p, rows, c, pname, p);
    }
  }
  if (any) {
    if (judge_dir.empty())
      std::printf("R2: %llu (position, layer) routes - %llu exact sets, %llu near-ties (8th/9th "
                  "within %.0e), %llu differing sets; weights worst %.3e relative, %llu beyond %.0e\n",
                  (unsigned long long)c.rows, (unsigned long long)c.exact_sets,
                  (unsigned long long)c.near_ties, kR2TieRel, (unsigned long long)c.bad_sets,
                  double(c.worst_w), (unsigned long long)c.bad_weights, kR2WeightRel);
    else
      std::printf("R2 judged by the f32 reference %s: %llu (position, layer) routes - %llu exact sets, %llu "
                  "near-ties (8th/9th logits within 1 bf16 ulp), %llu differing sets; weights worst %.3e relative, "
                  "%llu beyond %.0e; the bf16 golden against the same judge: %llu differing sets\n",
                  judge_dir.c_str(), (unsigned long long)c.rows, (unsigned long long)c.exact_sets,
                  (unsigned long long)c.near_ties, (unsigned long long)c.bad_sets, double(c.worst_w),
                  (unsigned long long)c.bad_weights, kR2WeightRel, (unsigned long long)c.ref_bad);
    CHECK_EQ(c.bad_sets, uint64_t(0));
    CHECK_EQ(c.bad_weights, uint64_t(0));
  }
  std::puts("ornith_decode_test OK");
  return 0;
}
