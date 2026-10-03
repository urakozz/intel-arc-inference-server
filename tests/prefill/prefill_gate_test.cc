// prefill_gate_test - spec 2 §6.1: the tie-aware golden gate, after
// `Engine::prefill` instead of after `T` × `ingest({id})`.
//
// It is `golden_gate_test`'s section 4 with **two changes and nothing else**:
//
//   * ingestion is ONE `prefill(ids)` call rather than one replay per id;
//   * there is no per-layer tap (ruling R6: the prefill walk writes no
//     `debug_resid`), so `golden_gate_test`'s sections 2 and 3 - the tap tables
//     and the per-layer `gdn_state` cosines - collapse to ONE `gdn_state`
//     comparison after the prefill, per GDN layer, printed with `**LOW**` under
//     `kBar` and **not gating**, exactly as the 2026-08-25 ruling that
//     `kBar`'s comment records requires.
//
// Everything else is the machinery itself, not a copy of it:
// `tests/golden/golden_common.h` is the extracted half of the decode gate, so
// the decision rule, the masking, the metrics and the golden-file validation
// are literally the same code. The gate's three clauses are unchanged -
// determined rows element-exact, undetermined rows set-membership,
// teacher-forced after the first divergence of either kind with `forced` set
// BEFORE the advance (the ordering `golden_gate_test.cc` measured and pinned).
//
// argv: [1] golden dir, [2] prompt dir, [3] snapshot,
//       [4] comma-separated prompt names (default "prose,code,cjk"),
//       [5] prefill chunk width (default 0 = PrefillScratch::kC),
//       [6] prefill backend, sycl-tla, l0 or l0-int8 (default: the build's),
//       [7] bf16 near-tie flips accepted per prompt (default 0; see main()).
// Arguments 4 and 5 are what make the >= 2048-id multi-chunk gate of spec §6.2
// a registration rather than a second binary (plan 6b Task 13 Step 6); argument
// 6 is what makes spec 2.1 §2 bar 3 a second registration rather than a second
// binary (plan 9e Task 1).
//
// Label `checkpoint golden prefill`: it needs the 19 GB checkpoint, a B70, and
// the ~1 GB of oracle output that lives only on the box. Absent goldens are a
// SKIP (exit 77), not a failure.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill/gdn.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill_backend.h"

namespace {
using model::Qwen35;
using golden::argmax_full;
using golden::argmax_masked;
using golden::compare_f32;
using golden::exists;
using golden::Golden;
using golden::GoldenDecision;
using golden::golden_decision;
using golden::kBar;
using golden::Metric;
using golden::print_top5;
using golden::read_ids;

constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 32;
constexpr uint32_t kHid = Qwen35::kHidden;
constexpr size_t kGdnElems =
    size_t(Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim;

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i <= s.size()) {
    const size_t j = s.find(',', i);
    const std::string part = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!part.empty()) out.push_back(part);
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return out;
}

// golden_gate_test's Verdict, minus the per-layer tap fields R6 removes.
struct Verdict {
  std::string name;
  uint32_t n_prompt = 0, exact = 0;
  uint32_t n_determined = 0, det_exact = 0;
  uint32_t near_tie = 0;   // determined rows accepted under argv[7]'s near-tie rule
  uint32_t n_tie = 0, tie_agree = 0, tie_member = 0;
  int first_bad = -1, first_diverge = -1;
  std::vector<uint32_t> tie_steps;
  double gdn_min_cos = 1.0, logit_min_cos = 1.0;
  uint32_t gdn_min_layer = 0;
};

// The oracle's masked runner-up and whether (top1 - top2) is within one bf16 ulp
// of top1: the margin a single bf16 rounding of the logits can erase. Used only
// by argv[7] (spec 5's accepted near-tie, 2026-09-24).
bool bf16_near_tie(const float* row, uint32_t used, uint32_t top1, uint32_t* second) {
  uint32_t s2 = top1 == 0 ? 1u : 0u;
  for (uint32_t i = 0; i < used; ++i)
    if (i != top1 && row[i] > row[s2]) s2 = i;
  *second = s2;
  int e = 0;
  std::frexp(std::fabs(row[top1]), &e);             // |v| = m 2^e, m in [0.5, 1)
  const double ulp = std::ldexp(1.0, e - 8);         // bf16: 8 significand bits
  return double(row[top1]) - double(row[s2]) <= ulp;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string gdir = argc > 1 ? argv[1] : "oracle-out";
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  const std::string snap = argc > 3 ? argv[3] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const std::vector<std::string> prompts = split_commas(argc > 4 ? argv[4] : "prose,code,cjk");
  const uint32_t chunk = argc > 5 ? uint32_t(std::atoi(argv[5])) : 0u;
  // argv[6]: the prefill backend (spec 2.1) -- sycl-tla or l0; default = the build's.
  runtime::PrefillBackend backend = runtime::prefill::default_prefill_backend();
  if (argc > 6) CHECK(runtime::parse_prefill_backend(argv[6], backend));
  // argv[7]: near-tie flips allowed per prompt (default 0: the gate as it always
  // was). A DETERMINED row the engine gets wrong is accepted instead of failing
  // only if the engine picked the oracle's runner-up AND the oracle's own top-2
  // margin is at most one bf16 ulp of its top logit. Operator ruling 2026-09-24
  // (spec 5 amendment): the l0-int8 path's one such flip, cjk row 9, margin
  // 0.0625 at 13.94, is accepted. Absent argv[7], the allowance is 1 on l0-int8
  // (the operator's ruling travels with the backend, so a registration that runs
  // the build's default gets it) and 0 on every other backend, l0 included.
  const uint32_t near_tie_allowed =
      argc > 7 ? uint32_t(std::atoi(argv[7]))
               : (backend == runtime::PrefillBackend::L0Int8 ? 1u : 0u);
  // argv[8] (spec 9): the lm_head form, bf16 (the checkpoint's; default) or int8. The
  // oracle is the bf16 CPU model either way, so an int8 run grades the head's error on
  // top of the engine's (spec 9 §4 L3).
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  if (argc > 8) CHECK(loader::parse_lm_head_form(argv[8], lm_head));
  std::printf("lm_head: %s\n", loader::lm_head_form_name(lm_head));
  CHECK(!prompts.empty());

  for (const std::string& p : prompts) {
    const std::string g = gdir + "/" + p + ".golden.safetensors";
    if (!exists(g)) {
      std::printf(
          "SKIP: %s is absent. The oracle dumps live only on the box "
          "(~1 GB, .gitignore'd) - see tools/oracle/README.md, 'Running it'.\n",
          g.c_str());
      return 77;
    }
  }

  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, lm_head);
  // debug_resid=false: ruling R6 gives the prefill walk no per-layer tap, so
  // there is nothing for it to fill and the 64 device copies a token would be
  // paid for nothing.
  // spec 14: 64 / 48 on Qwen3.8, 72 / 54 on Agnes
  const uint32_t kLayers = model.desc->layers, kGdnLayers = model.desc->gdn_layers;
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  eng.set_prefill_backend(backend);
  std::printf("prefill backend: %s\n", runtime::prefill_backend_name(backend));
  // The scan selector's DISPATCH PROOF (parity-program design §11). A gate run
  // under `B70_PREFILL_GDN_SCAN` is not evidence unless it says which kernel it
  // launched: the 2026-09-21 "green" split record was executing `pf_gdn_scan`.
  {
    const char* const sel = std::getenv("B70_PREFILL_GDN_SCAN");
    std::printf("gdn scan selector: B70_PREFILL_GDN_SCAN=%s -> entry %s\n",
                sel && *sel ? sel : "(unset)", runtime::prefill::gdn_scan_entry_name());
  }
  // The decode list is 12 launches per layer + 6 at the boundary: 774 on Qwen3.8
  // (645 + lever L1's 129), 870 on Agnes's 72 layers (spec 14, derived).
  CHECK_EQ(eng.step().kernel_count, size_t(12) * kLayers + 6);   // decode is untouched
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const size_t gdn_stride = kGdnElems * sizeof(float);
  CHECK_EQ(eng.buffers().gdn_state.size(), gdn_stride * kGdnLayers);

  std::vector<Verdict> verdicts;
  std::vector<double> sa, sb;

  for (const std::string& pname : prompts) {
    Verdict v;
    v.name = pname;
    const std::vector<uint32_t> ids = read_ids(pdir + "/" + pname + ".ids");
    Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size());
    v.n_prompt = T;
    CHECK(T > 0);
    CHECK_EQ(g.dim("resid.L0", 2, 0), uint64_t(T));
    CHECK_EQ(g.dim("resid.L0", 2, 1), uint64_t(kHid));
    const uint32_t V = uint32_t(g.dim("logits", 2, 1));
    CHECK_EQ(uint32_t(g.dim("logits", 2, 0)), T + kGen);
    const uint32_t Vcmp = std::min(V, Qwen35::kVocab);
    const uint32_t C = chunk ? chunk : runtime::PrefillScratch::kC;
    std::printf(
        "\n================ %s: %u prompt ids PREFILLED at chunk %u (%u chunk(s)),"
        " %u generated, golden vocab %u ================\n",
        pname.c_str(), T, C, (T + C - 1) / C, kGen, V);

    // ---- 1. ONE prefill call. This is the whole difference. -----------------
    eng.reset();
    CHECK_EQ(eng.pos(), uint32_t(0));
    eng.prefill(ids, chunk);
    CHECK_EQ(eng.pos(), T);
    // The other half of the dispatch proof: the entry a scan launch was really
    // built with, read back after the walk rather than before it.
    CHECK(runtime::prefill::gdn_scan_launched_entry() != nullptr);
    CHECK_EQ(std::string(runtime::prefill::gdn_scan_launched_entry()),
             std::string(runtime::prefill::gdn_scan_entry_name()));
    std::printf("gdn scan entry LAUNCHED: %s\n",
                runtime::prefill::gdn_scan_launched_entry());

    // ---- 2. the ONE gdn_state comparison R6 leaves (diagnostic, not gating) -
    std::printf("  gdn_state after the prefill, per GDN layer (DIAGNOSTIC - the 2026-08-25\n"
                "  ruling in golden_common.h's kBar comment: tokens gate, tensors diagnose)\n"
                "    layer      cosine      relL2     |oracle|      |err|\n");
    {
      std::vector<float> got(kGdnElems);
      for (uint32_t l = 0, gl = 0; l < kLayers; ++l) {
        if (::model::ModelDesc::is_fa(l)) continue;
        const float* ref = g.f32("gdn_state.L" + std::to_string(l), kGdnElems);
        imm.copy(got.data(), static_cast<const uint8_t*>(eng.buffers().gdn_state.ptr()) +
                                 size_t(gl) * gdn_stride,
                 gdn_stride);
        const Metric m = compare_f32(got.data(), ref, kGdnElems, sa, sb);
        if (m.cos < v.gdn_min_cos) {
          v.gdn_min_cos = m.cos;
          v.gdn_min_layer = l;
        }
        if (m.cos < kBar || l % 8 == 0)
          std::printf("      %2u     %.9f   %.3e   %9.3f  %9.3f%s\n", l, m.cos, m.rel, m.nb,
                      m.err, m.cos < kBar ? "   **LOW**" : "");
        ++gl;
      }
      std::printf("    worst: L%u at %.9f\n", v.gdn_min_layer, v.gdn_min_cos);
    }

    // ---- 3. the gate, unchanged in semantics --------------------------------
    const float* glog = g.f32("logits", size_t(T + kGen) * V);
    const int32_t* gtok = g.i32("tokens", kGen);
    std::vector<float> dec(Qwen35::kVocab);
    // **The decision row for step 0 is in `PrefillScratch::logits`, not in
    // `DecodeBuffers::logits`.** `step_head`'s lm_head writes the prefill
    // scratch's own [1][248320] row and `argmax_stage2` samples from it; decode's
    // logits buffer still holds whatever the last replay left, which for a
    // freshly reset engine is the zeros. Every LATER row comes from a replay and
    // is decode's. Getting this wrong does not change a token -- the device has
    // already sampled -- but it makes the host-vs-device argmax cross-check
    // below fail on step 0, which is exactly what it is there for.
    auto read_logits = [&](bool from_prefill) {
      const l0::Mem& src =
          from_prefill ? eng.prefill_scratch()->logits : eng.buffers().logits;
      imm.copy(dec.data(), src.ptr(), size_t(Qwen35::kVocab) * 4);
    };
    read_logits(/*from_prefill=*/true);
    runtime::Control* ctrl = eng.buffers().control.as<runtime::Control>();
    std::printf("  greedy %u:  step  engine  golden  det?  logit-cos   engine-argmax "
                "golden-argmax (masked / full)\n", kGen);
    std::vector<uint32_t> etok;
    etok.reserve(kGen);
    bool forced = false;
    for (uint32_t p = 0; p < kGen; ++p) {
      const float* grow = glog + size_t(p == 0 ? T - 1 : T + p - 1) * V;
      const Metric lm = compare_f32(dec.data(), grow, Vcmp, sa, sb);
      v.logit_min_cos = std::min(v.logit_min_cos, lm.cos);
      const uint32_t ea = argmax_masked(dec.data(), Vcmp, Qwen35::kVocabUsed);
      const uint32_t gam = argmax_masked(grow, Vcmp, Qwen35::kVocabUsed);
      const uint32_t ea_dev = argmax_masked(dec.data(), Qwen35::kVocab, Qwen35::kVocabUsed);
      const uint32_t gaf = argmax_full(grow, V);
      const GoldenDecision gd = golden_decision(grow, Vcmp, Qwen35::kVocabUsed);
      CHECK_EQ(gd.set[0], gam);
      CHECK_EQ(gam, uint32_t(gtok[p]));

      const uint32_t id = ctrl->cur_token[0];
      CHECK_EQ(id, ea_dev);
      etok.push_back(id);

      const bool ok = id == uint32_t(gtok[p]);
      if (ok) ++v.exact;
      bool accepted_near_tie = false;
      if (gd.determined()) {
        ++v.n_determined;
        uint32_t second = 0;
        if (ok) {
          ++v.det_exact;
        } else if (v.near_tie < near_tie_allowed &&
                   bf16_near_tie(grow, Qwen35::kVocabUsed, gam, &second) && id == second) {
          ++v.near_tie;
          accepted_near_tie = true;
        } else if (v.first_bad < 0) {
          v.first_bad = int(p);
        }
      } else {
        ++v.n_tie;
        v.tie_steps.push_back(p);
        if (ok) ++v.tie_agree;
        else if (gd.contains(id)) ++v.tie_member;
        else if (v.first_bad < 0) v.first_bad = int(p);
      }
      if (!ok && v.first_diverge < 0) v.first_diverge = int(p);

      const char* mark =
          ok ? ""
             : accepted_near_tie
                   ? "  <== near-tie ACCEPTED (engine = golden runner-up, margin <= 1 bf16 ulp)"
                   : (gd.determined() ? "  <== MISMATCH (determined - GATE)"
                                      : "  <== differs, but the golden row is a TIE");
      std::printf("            %4u  %6u  %6u  %-4s  %.9f   %6u %6u %6u %s%s%s\n", p, id,
                  uint32_t(gtok[p]), gd.determined() ? "yes" : "TIE", lm.cos, ea, gam, gaf, mark,
                  gam != gaf ? "  [golden argmax is a padding id]" : "",
                  forced ? "  [teacher-forced]" : "");
      if (!ok && int(p) == v.first_diverge) {
        std::printf("      first divergence at generated position %u: engine %u, golden %u"
                    "  (decision row = golden logits[%u], cos %.9f, relL2 %.3e)\n",
                    p, id, uint32_t(gtok[p]), p == 0 ? T - 1 : T + p - 1, lm.cos, lm.rel);
        print_top5("engine", dec.data(), Qwen35::kVocab, Qwen35::kVocabUsed);
        print_top5("golden", grow, V, Qwen35::kVocabUsed);
      }
      // `forced` BEFORE the advance - golden_gate_test measured and pinned this
      // ordering (its comment carries the numbers); the replay at the divergence
      // row is the one that writes the diverging token into the KV cache, the
      // conv ring and the GDN state.
      if (!ok) forced = true;
      if (forced) {
        CHECK(uint32_t(gtok[p]) < Qwen35::kVocabUsed);
        eng.ingest({uint32_t(gtok[p])});
      } else {
        CHECK_EQ(eng.generate(1)[0], id);
      }
      read_logits(/*from_prefill=*/false);
    }
    std::printf("  undetermined rows in this golden file: %u of %u", v.n_tie, kGen);
    if (v.n_tie == 0) {
      std::printf(" (none)\n");
    } else {
      std::printf(" -");
      for (uint32_t t : v.tie_steps) std::printf(" step %u", t);
      std::printf("\n");
    }
    std::printf("  %s: %u determined-exact / %u tie-agreements / %u tie-set-members"
                "   (%u determined + %u undetermined = %u)%s\n",
                pname.c_str(), v.det_exact, v.tie_agree, v.tie_member, v.n_determined, v.n_tie,
                kGen, v.first_bad < 0 ? "" : "   ** GATE FAILURE **");
    if (v.near_tie)
      std::printf("  %s: %u determined row(s) accepted as bf16 near-ties (argv[7] allows %u)\n",
                  pname.c_str(), v.near_tie, near_tie_allowed);
    std::printf("  engine:");
    for (uint32_t id : etok) std::printf(" %u", id);
    std::printf("\n  golden:");
    for (uint32_t p = 0; p < kGen; ++p) std::printf(" %u", uint32_t(gtok[p]));
    std::printf("\n");
    CHECK_EQ(v.n_determined + v.n_tie, kGen);
    verdicts.push_back(v);
  }

  std::printf("\n================ prefill golden gate ================\n");
  std::printf("  the gate is the token columns alone; every cosine below is a diagnostic.\n"
              "  prompt   ids   det-exact  tie-agree  tie-member   gdn min cos   logit min cos\n");
  for (const Verdict& v : verdicts)
    std::printf("  %-7s %4u    %2u/%-2u        %2u         %2u        %.9f (L%u)  %.9f\n",
                v.name.c_str(), v.n_prompt, v.det_exact, v.n_determined, v.tie_agree,
                v.tie_member, v.gdn_min_cos, v.gdn_min_layer, v.logit_min_cos);

  uint32_t tot_det = 0, tot_det_ok = 0, tot_tie = 0, tot_agree = 0, tot_member = 0;
  bool bad = false;
  for (const Verdict& v : verdicts) {
    tot_det += v.n_determined;
    tot_det_ok += v.det_exact;
    tot_tie += v.n_tie;
    tot_agree += v.tie_agree;
    tot_member += v.tie_member;
    if (v.det_exact + v.near_tie != v.n_determined) {
      std::fprintf(stderr,
                   "GATE FAILED: %s got %u of %u DETERMINED rows exact (first bad row %d) - a row"
                   " whose golden argmax is unique is not a judgement call\n",
                   v.name.c_str(), v.det_exact, v.n_determined, v.first_bad);
      bad = true;
    }
    if (v.tie_agree + v.tie_member != v.n_tie) {
      std::fprintf(stderr,
                   "GATE FAILED: %s produced an id outside the golden argmax set on an"
                   " undetermined row (%u agree + %u member != %u ties, first bad row %d)\n",
                   v.name.c_str(), v.tie_agree, v.tie_member, v.n_tie, v.first_bad);
      bad = true;
    }
  }
  std::printf("  TOTAL: %u/%u determined rows exact, %u undetermined (%u agree + %u other"
              " member)\n", tot_det_ok, tot_det, tot_tie, tot_agree, tot_member);
  if (bad) return 1;
  std::printf("prefill_gate_test OK: %zu prompt(s) x %u greedy tokens after ONE"
              " Engine::prefill each - %u/%u determined rows element-exact against the CPU"
              " oracle, %u undetermined rows all inside the golden argmax set\n",
              verdicts.size(), kGen, tot_det_ok, tot_det, tot_tie);
  return 0;
}
