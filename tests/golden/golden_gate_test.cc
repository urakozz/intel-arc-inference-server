// golden_gate_test - the trust chain closed (spec §11, plan 3 Task 8).
//
// Everything before this test compares a kernel against a host reference that
// *this project wrote*. This one compares the whole engine against a reference
// it did not write: `transformers` 5.15 on CPU, the pure-torch
// gated-delta-rule, run over the same checkpoint dequantised by the same rule
// the C++ loader is bit-compared against (tools/oracle/README.md). If the two
// agree on 3 x 32 greedy tokens, element-exact, then 774 kernels, the loader,
// the repack, the capture and the decode loop are all right together.
//
// The gate, per prompt:
//   * ingest the committed prompt ids ONE AT A TIME, reading the per-layer
//     tap after each (Engine::ingest has no per-token callback; one id per
//     call is byte-identical to one call with all of them, because each call
//     is just a replay per id);
//   * generate 32 ids as 32 x generate(1), reading `buffers().logits` before
//     each step - that fp32 row is exactly the row whose argmax the step is
//     about to consume, so a mismatch can be reported with both top-5s;
//   * **GATE**, under the controller's ruling of 2026-08-26 (docs/14, "The
//     RTN-checkpoint gate"): a golden decision row whose top-1 is **not unique**
//     is UNDETERMINED, and the three clauses are
//       (i)   determined rows - the engine's id == the golden id, element-exact,
//             every one of them. Unchanged strictness; this is the gate.
//       (ii)  undetermined rows - the engine's id must be a MEMBER of the golden
//             argmax set. The oracle's logits are bf16 widened to fp32, so two
//             candidates can be the same word; which one `torch.argmax` returns
//             is its lowest-index tie-break, not an output of the model.
//       (iii) after any divergence the walk continues TEACHER-FORCED on the
//             oracle's token, so clause (i) still binds the tail on the
//             reference's own context instead of grading a diverged sequence.
//     The tie census is derived here from the golden `logits`, never pasted in.
//
// Diagnostics - printed for every prompt whether or not the gate passes, and
// evaluated only after all three prompts have printed, so a failure still
// leaves the complete picture in the log:
//   * per layer, min-over-t cosine (fp64) of the tap against the oracle;
//   * the tail the tap cannot see, `b.resid` vs golden `resid.L63`;
//   * the GDN recurrent state after the prompt, per GDN layer;
//   * per generated step, the cosine of the two logit rows.
//
// **What tap[i] is.** NOT layer i's output. `runtime/capture.h` is the single
// authority: the tap copies after layer i's last kernel, and the residual
// stream is advanced only by `prep_res_norm`, so tap[i] holds the hidden state
// with layer i's *mixer* folded in and layer i's MLP still un-folded in
// `partials` (layer i+1's leading prep folds it). The oracle's `resid.L{i}` is
// the layer *output* - post-MLP - so the comparator is built, not read:
//
//     expected_tap[i][t] = bf16( resid.L{i-1}[t] + mixer.L{i}[t] )   i >= 1
//     expected_tap[0][t] = bf16( embed[ids[t]]   + mixer.L0[t]  )
//
// summed in fp32 from the two golden bf16 tensors and rounded RNE, because
// that is the arithmetic the engine's `prep_res_norm` does. Layer 0 has no
// `resid.L-1`: the oracle dumps no embedding tensor, so the rows are gathered
// host-side out of `LoadedModel::embed`, which is the same bf16 table the
// device gathers from. Layer 63's post-MLP residual has NO tap at all (there
// is no tap 64) - `b.resid` after the fence covers it instead, and the logits
// and the tokens cover it again.
//
// Label `checkpoint golden`: needs the real 19 GB checkpoint, a B70, and the
// ~1 GB of oracle output that lives only on the box. Absent goldens are a SKIP
// (exit 77), not a failure - the Mac never has them.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "loader/safetensors.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"

// Everything this test and prefill_gate_test share: the golden-file reader, the
// fp64 metrics, the masked argmax, the tie-aware decision rule and kBar. Moved
// out verbatim (plan 6b Task 12 Step 1); this file keeps its Verdict and main.
#include "golden_common.h"

namespace {
using model::Qwen35;
using golden::argmax_full;
using golden::argmax_masked;
using golden::compare;
using golden::compare_bf16;
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
constexpr uint32_t kHid = Qwen35::kHidden;  // 5120
constexpr size_t kGdnElems =
    size_t(Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim;  // 48*128*128
const char* const kPrompts[] = {"prose", "code", "cjk"};






// One prompt's verdict, printed again in the summary at the end.
struct Verdict {
  std::string name;
  uint32_t n_prompt = 0, exact = 0;
  // The gate under the 2026-08-26 ruling. `n_determined + n_tie == kGen`
  // always; the gate passes iff `det_exact == n_determined` AND
  // `tie_agree + tie_member == n_tie`.
  uint32_t n_determined = 0, det_exact = 0;   // clause (i): unique golden argmax
  uint32_t n_tie = 0, tie_agree = 0;          // clause (ii), engine took torch's pick
  uint32_t tie_member = 0;                    // clause (ii), engine took another member
  int first_bad = -1;        // first DETERMINED row the engine got wrong - the failure
  int first_diverge = -1;    // first row where engine != golden, tie or not
  std::vector<uint32_t> tie_steps;   // the undetermined rows, for the census
  double tap_min_cos = 1.0, tail_min_cos = 1.0, gdn_min_cos = 1.0, logit_min_cos = 1.0;
  uint32_t tap_min_layer = 0, tap_min_t = 0, gdn_min_layer = 0;
};
}  // namespace

int main(int argc, char** argv) {
  const std::string gdir = argc > 1 ? argv[1] : "oracle-out";
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  const std::string snap = argc > 3 ? argv[3] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  // argv[4] (spec 9): the lm_head form, bf16 (the checkpoint's; default) or int8.
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  if (argc > 4) CHECK(loader::parse_lm_head_form(argv[4], lm_head));
  std::printf("lm_head: %s\n", loader::lm_head_form_name(lm_head));

  for (const char* p : kPrompts) {
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
  // One load, one Engine, three prompts: reset() zeroes exactly the persistent
  // group between them, which is what starting a fresh session means. Loading
  // 19 GB three times would cost ~6 minutes and prove nothing extra.
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, lm_head);
  CHECK(model.embed.size() >= size_t(Qwen35::kVocab) * kHid * 2);
  runtime::Engine eng(ctx, std::move(model), kMaxLen, /*debug_resid=*/true);
  CHECK(eng.debug_resid());
  CHECK_EQ(eng.step().kernel_count, size_t(774));   // 645 + lever L1's 129
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  const size_t gdn_stride = kGdnElems * sizeof(float);
  CHECK_EQ(eng.buffers().gdn_state.size(), gdn_stride * 48);

  std::vector<Verdict> verdicts;
  std::vector<double> sa, sb;  // fp64 scratch, reused

  for (const char* pname : kPrompts) {
    Verdict v;
    v.name = pname;
    const std::vector<uint32_t> ids = read_ids(pdir + "/" + pname + ".ids");
    Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size());
    v.n_prompt = T;
    CHECK(T > 0 && T <= 64);
    // The one standing consistency check between a committed .ids file and the
    // golden dump it is the prompt of: same number of positions. (The ids were
    // additionally verified byte-equal to each file's `prompt_ids` metadata by
    // hand when they were committed; `parse_header` drops `__metadata__`, so
    // the test cannot re-check that, and plumbing it through is not worth it.)
    CHECK_EQ(g.dim("resid.L0", 2, 0), uint64_t(T));
    CHECK_EQ(g.dim("resid.L0", 2, 1), uint64_t(kHid));
    const uint32_t V = uint32_t(g.dim("logits", 2, 1));
    const uint32_t Lrows = uint32_t(g.dim("logits", 2, 0));
    CHECK_EQ(Lrows, T + kGen);
    const uint32_t Vcmp = std::min(V, Qwen35::kVocab);
    std::printf(
        "\n================ %s: %u prompt ids, %u generated, golden vocab %u "
        "(engine %u padded / %u usable) ================\n",
        pname, T, kGen, V, Qwen35::kVocab, Qwen35::kVocabUsed);

    // ---- 1. ingest, one id per replay, tapping after each -------------------
    eng.reset();
    CHECK_EQ(eng.pos(), uint32_t(0));
    std::vector<std::vector<uint16_t>> tap(T);
    std::vector<std::vector<uint16_t>> tail(T);   // b.resid after the fence
    std::vector<std::vector<uint16_t>> emb(T);    // the embedding row the device gathered
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      tap[t] = eng.read_debug_resid();
      CHECK_EQ(tap[t].size(), size_t(Qwen35::kLayers) * kHid);
      tail[t].resize(kHid);
      imm.copy(tail[t].data(), eng.buffers().resid.ptr(), size_t(kHid) * 2);
      emb[t].resize(kHid);
      imm.copy(emb[t].data(),
               static_cast<const uint8_t*>(eng.model().embed.ptr()) + size_t(ids[t]) * kHid * 2,
               size_t(kHid) * 2);
    }
    CHECK_EQ(eng.pos(), T);

    // ---- 2. the per-layer tap vs the built comparator -----------------------
    std::printf("  layer diagnostics (min over the %u prompt positions; expected tap =\n"
                "  bf16(resid.L{i-1}[t] + mixer.L{i}[t]), embed[id] standing in for resid.L-1).\n"
                "  |oracle| and |err| are at the argmin t: a cosine is only as meaningful as\n"
                "  the magnitude it divides by, so both are printed next to it.\n"
                "    layer kind    min cos      at t   median cos   max relL2   |oracle|      |err|\n", T);
    std::vector<uint16_t> expect(kHid);
    std::vector<double> cos_lt(size_t(Qwen35::kLayers) * T), nb_lt(size_t(Qwen35::kLayers) * T);
    uint32_t n_low_pairs = 0;
    std::vector<uint32_t> low_t;   // the distinct positions that go below the bar
    std::vector<double> med_l(Qwen35::kLayers, 1.0);   // per-layer upper-median cosine
    for (uint32_t l = 0; l < Qwen35::kLayers; ++l) {
      const std::string ls = std::to_string(l);
      const uint16_t* mix = g.bf16("mixer.L" + ls, size_t(T) * kHid);
      const uint16_t* prev =
          l == 0 ? nullptr : g.bf16("resid.L" + std::to_string(l - 1), size_t(T) * kHid);
      double lmin = 2.0, lrel = 0.0, lnb = 0.0, lerr = 0.0;
      uint32_t at = 0;
      for (uint32_t t = 0; t < T; ++t) {
        const uint16_t* base = prev ? prev + size_t(t) * kHid : emb[t].data();
        for (uint32_t k = 0; k < kHid; ++k)
          expect[k] = common::f32_to_bf16(common::bf16_to_f32(base[k]) +
                                          common::bf16_to_f32(mix[size_t(t) * kHid + k]));
        const Metric m = compare_bf16(tap[t].data() + size_t(l) * kHid, expect.data(), kHid, sa, sb);
        cos_lt[size_t(l) * T + t] = m.cos;
        nb_lt[size_t(l) * T + t] = m.nb;
        if (m.cos < kBar) {
          ++n_low_pairs;
          if (std::find(low_t.begin(), low_t.end(), t) == low_t.end()) low_t.push_back(t);
        }
        if (m.cos < lmin) { lmin = m.cos; at = t; lnb = m.nb; lerr = m.err; }
        lrel = std::max(lrel, m.rel);
      }
      if (lmin < v.tap_min_cos) { v.tap_min_cos = lmin; v.tap_min_layer = l; v.tap_min_t = at; }
      // Upper median: element T/2 of the sorted row, which for even T is the
      // upper of the two central values rather than their mean. Named that way
      // in the header so nobody averages two of these and calls it a median.
      std::vector<double> row(cos_lt.begin() + size_t(l) * T, cos_lt.begin() + size_t(l + 1) * T);
      std::nth_element(row.begin(), row.begin() + T / 2, row.end());
      med_l[l] = row[T / 2];
      std::printf("      %2u  %-4s  %.9f   %4u   %.9f   %.3e   %9.3f  %9.3f%s\n", l,
                  Qwen35::is_fa(l) ? "FA" : "GDN", lmin, at, med_l[l], lrel, lnb, lerr,
                  lmin < kBar ? "   **LOW**" : "");
    }
    // Printed rather than derived by hand afterwards: this is the number the
    // docs quote as "the typical comparison", and the split says whether the
    // softmax path (FA) diverges more than the GEMV path alone (GDN).
    {
      double all = 0, gdn = 0, fa = 0;
      uint32_t ng = 0, nf = 0;
      for (uint32_t l = 0; l < Qwen35::kLayers; ++l) {
        all += med_l[l];
        if (Qwen35::is_fa(l)) { fa += med_l[l]; ++nf; } else { gdn += med_l[l]; ++ng; }
      }
      std::printf("  mean of the 64 per-layer upper-medians: %.9f   (GDN %u: %.9f, FA %u: %.9f)\n",
                  all / Qwen35::kLayers, ng, gdn / ng, nf, fa / nf);
    }
    std::printf("  census: %u of the %u (layer, t) tap comparisons are below %.3f, on %zu of the"
                " %u positions:", n_low_pairs, uint32_t(Qwen35::kLayers) * T, kBar, low_t.size(), T);
    std::sort(low_t.begin(), low_t.end());
    for (uint32_t t : low_t) std::printf(" %u", t);
    std::printf("%s\n", low_t.empty() ? " (none)" : "");

    // Attribution at the worst position. The tap error alone cannot say whether
    // a layer produced a wrong output or merely carried an earlier one, so the
    // layer's own contribution is compared too: in the engine it is
    // tap[i] - tap[i-1] (mixer_i + mlp_{i-1}, the two things folded between the
    // two copies), and the oracle dumps both halves of that as mixer.L{i} and
    // mlp.L{i-1}. Layer 0's contribution is tap[0] - embed = mixer.L0.
    {
      const uint32_t tw = v.tap_min_t;
      std::printf("  attribution at the worst position t=%u (contribution = mixer.L{i} +"
                  " mlp.L{i-1}):\n"
                  "    layer kind   tap cos     |tap err|   contrib cos  |contrib|  |contrib err|\n",
                  tw);
      std::vector<double> ce(kHid), co(kHid);
      for (uint32_t l = 0; l < Qwen35::kLayers; ++l) {
        const std::string ls = std::to_string(l);
        const uint16_t* mix = g.bf16("mixer.L" + ls, size_t(T) * kHid) + size_t(tw) * kHid;
        const uint16_t* prev =
            l == 0 ? nullptr : g.bf16("resid.L" + std::to_string(l - 1), size_t(T) * kHid) +
                                   size_t(tw) * kHid;
        const uint16_t* base = prev ? prev : emb[tw].data();
        for (uint32_t k = 0; k < kHid; ++k)
          expect[k] = common::f32_to_bf16(common::bf16_to_f32(base[k]) + common::bf16_to_f32(mix[k]));
        const Metric mt =
            compare_bf16(tap[tw].data() + size_t(l) * kHid, expect.data(), kHid, sa, sb);
        const uint16_t* pmlp =
            l == 0 ? nullptr : g.bf16("mlp.L" + std::to_string(l - 1), size_t(T) * kHid) +
                                   size_t(tw) * kHid;
        const uint16_t* eprev = l == 0 ? emb[tw].data() : tap[tw].data() + size_t(l - 1) * kHid;
        for (uint32_t k = 0; k < kHid; ++k) {
          ce[k] = common::bf16_to_f32(tap[tw][size_t(l) * kHid + k]) - common::bf16_to_f32(eprev[k]);
          co[k] = common::bf16_to_f32(mix[k]) + (pmlp ? common::bf16_to_f32(pmlp[k]) : 0.0f);
        }
        const Metric mc = compare(ce.data(), co.data(), kHid);
        std::printf("      %2u  %-4s  %.9f  %9.3f   %.9f  %9.3f  %9.3f\n", l,
                    Qwen35::is_fa(l) ? "FA" : "GDN", mt.cos, mt.err, mc.cos, mc.nb, mc.err);
      }
    }
    // The worst layer, position by position. A cosine that dips at two or three
    // positions and recovers is a different animal from one that decays with
    // depth, and this line is what tells the two apart.
    std::printf("  worst tap layer L%u, every position (t: cos / |oracle|):\n",
                v.tap_min_layer);
    for (uint32_t t = 0; t < T; ++t)
      std::printf("      %2u: %.6f %8.2f%s", t, cos_lt[size_t(v.tap_min_layer) * T + t],
                  nb_lt[size_t(v.tap_min_layer) * T + t], (t % 4 == 3 || t + 1 == T) ? "\n" : "");
    // The tail the tap cannot reach: layer 63 post-MLP, which `b.resid` holds
    // after every fence (capture.h). There is no tap 64 and none is invented.
    {
      const uint16_t* r63 = g.bf16("resid.L63", size_t(T) * kHid);
      double lmin = 2.0, lrel = 0.0, lnb = 0.0, lerr = 0.0;
      uint32_t at = 0;
      for (uint32_t t = 0; t < T; ++t) {
        const Metric m = compare_bf16(tail[t].data(), r63 + size_t(t) * kHid, kHid, sa, sb);
        if (m.cos < lmin) { lmin = m.cos; at = t; lnb = m.nb; lerr = m.err; }
        lrel = std::max(lrel, m.rel);
      }
      v.tail_min_cos = lmin;
      std::printf("      63+ MLP (b.resid vs resid.L63)  min cos %.9f at t=%u, max relL2 %.3e,"
                  " |oracle| %.3f, |err| %.3f%s\n",
                  lmin, at, lrel, lnb, lerr, lmin < kBar ? "   **LOW**" : "");
    }

    // ---- 3. the GDN recurrent state after the prompt ------------------------
    std::printf("  gdn_state after the prompt (%zu fp32 per layer, 48 GDN layers):\n", kGdnElems);
    {
      std::vector<float> st(kGdnElems);
      uint32_t gi = 0, n_low = 0;
      double gmin = 2.0, gmax = -2.0, relmax = 0.0;
      for (uint32_t l = 0; l < Qwen35::kLayers; ++l) {
        if (Qwen35::is_fa(l)) continue;
        imm.copy(st.data(),
                 static_cast<const uint8_t*>(eng.buffers().gdn_state.ptr()) + size_t(gi) * gdn_stride,
                 gdn_stride);
        const Metric m = compare_f32(st.data(), g.f32("gdn_state.L" + std::to_string(l), kGdnElems),
                                     kGdnElems, sa, sb);
        if (m.cos < gmin) { gmin = m.cos; v.gdn_min_layer = l; }
        gmax = std::max(gmax, m.cos);
        relmax = std::max(relmax, m.rel);
        if (m.cos < kBar) {
          ++n_low;
          std::printf("      L%-2u cos %.9f relL2 %.3e   **LOW**\n", l, m.cos, m.rel);
        }
        ++gi;
      }
      CHECK_EQ(gi, uint32_t(48));
      v.gdn_min_cos = gmin;
      std::printf("      min cos %.9f (L%u), max cos %.9f, max relL2 %.3e, %u/48 below %.3f\n",
                  gmin, v.gdn_min_layer, gmax, relmax, n_low, kBar);
    }

    // ---- 4. THE GATE: 32 greedy ids, element-exact --------------------------
    CHECK_EQ(g.dim("tokens", 1, 0), uint64_t(kGen));
    const int32_t* gtok = g.i32("tokens", kGen);
    const float* glog = g.f32("logits", size_t(Lrows) * V);
    std::vector<float> dec(Qwen35::kVocab);
    auto read_logits = [&] {
      imm.copy(dec.data(), eng.buffers().logits.ptr(), size_t(Qwen35::kVocab) * sizeof(float));
    };
    read_logits();  // the row whose argmax is generated token 0: golden logits[T-1]

    // **Clause (iii) of the ruling: after a divergence the walk continues
    // TEACHER-FORCED.** A greedy sequence never recovers from one different
    // token, so a free-running tail compares two different contexts and says
    // nothing about the engine - every row after a divergence would be a
    // consequence of it, not an independent test. Feeding the oracle's own
    // token instead puts the engine back on the reference's context, and
    // clause (i) then binds the remaining DETERMINED rows at full strictness.
    // `ingest({id})` is the same single replay of the same captured list that
    // `generate(1)` runs; the only difference is which token `cur_token`
    // carries into it (runtime/engine.cc).
    //
    // Teacher-forcing starts at the first divergence of EITHER kind. On a tie
    // row that is the ruling; on a determined row the gate has already failed
    // and the tail is then a diagnostic, which is worth more than 16 rows of
    // noise.
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
      // Both compared argmaxes see the SAME width, so a future dump narrower
      // than the engine's padded row cannot silently desymmetrize the pair.
      // The device-consistency check below is separate and must keep the full
      // engine width, because the device argmaxes the whole padded row.
      const uint32_t ea = argmax_masked(dec.data(), Vcmp, Qwen35::kVocabUsed);
      const uint32_t gam = argmax_masked(grow, Vcmp, Qwen35::kVocabUsed);
      const uint32_t ea_dev = argmax_masked(dec.data(), Qwen35::kVocab, Qwen35::kVocabUsed);
      const uint32_t gaf = argmax_full(grow, V);   // the pad-tail check wants all of V
      const GoldenDecision gd = golden_decision(grow, Vcmp, Qwen35::kVocabUsed);
      // torch's argmax and this one break a tie the same way, so set[0] is
      // what the dump recorded. If that ever stops holding, the golden file and
      // this test disagree about the reference itself.
      CHECK_EQ(gd.set[0], gam);
      CHECK_EQ(gam, uint32_t(gtok[p]));

      // The engine's token p: produced by the previous replay and sitting in
      // `cur_token`, which is what the NEXT replay will consume. Read here
      // rather than from generate()'s return value so that the device-vs-host
      // argmax cross-check runs on every step, teacher-forced ones included.
      const uint32_t id = ctrl->cur_token[0];
      CHECK_EQ(id, ea_dev);
      etok.push_back(id);

      const bool ok = id == uint32_t(gtok[p]);
      if (ok) ++v.exact;
      if (gd.determined()) {
        ++v.n_determined;
        if (ok) ++v.det_exact;
        else if (v.first_bad < 0) v.first_bad = int(p);
      } else {
        ++v.n_tie;
        v.tie_steps.push_back(p);
        if (ok) ++v.tie_agree;
        else if (gd.contains(id)) ++v.tie_member;
        else if (v.first_bad < 0) v.first_bad = int(p);   // not even a member: a real failure
      }
      if (!ok && v.first_diverge < 0) v.first_diverge = int(p);

      const char* mark = ok ? ""
                            : (gd.determined() ? "  <== MISMATCH (determined - GATE)"
                                               : "  <== differs, but the golden row is a TIE");
      std::printf("            %4u  %6u  %6u  %-4s  %.9f   %6u %6u %6u %s%s%s\n", p, id,
                  uint32_t(gtok[p]), gd.determined() ? "yes" : "TIE", lm.cos, ea, gam, gaf,
                  mark, gam != gaf ? "  [golden argmax is a padding id]" : "",
                  forced ? "  [teacher-forced]" : "");
      if (!ok && int(p) == v.first_diverge) {
        std::printf("      first divergence at generated position %u: engine %u, golden %u"
                    "  (decision row = golden logits[%u], cos %.9f, relL2 %.3e)\n",
                    p, id, uint32_t(gtok[p]), p == 0 ? T - 1 : T + p - 1, lm.cos, lm.rel);
        if (!gd.determined()) {
          std::printf("      the golden row is UNDETERMINED: %zu ids attain the maximum"
                      " %.9f -", gd.set.size(), double(gd.value));
          for (uint32_t i : gd.set) std::printf(" %u", i);
          std::printf("\n      the engine's %u is a member, so clause (ii) is satisfied; the"
                      " walk continues teacher-forced on the golden %u\n", id, uint32_t(gtok[p]));
        }
        print_top5("engine", dec.data(), Qwen35::kVocab, Qwen35::kVocabUsed);
        print_top5("golden", grow, V, Qwen35::kVocabUsed);
      }

      // Advance one replay. Free-running until the first divergence, then the
      // oracle's token - see the clause-(iii) note above.
      //
      // **`forced` is set BEFORE the advance, and that ordering is the whole
      // point.** The replay at the divergence row is the one that writes the
      // diverging token into the KV cache, the conv ring and the GDN recurrent
      // state; letting it consume the engine's own id and only teacher-forcing
      // from the NEXT row leaves every later row reading a contaminated state,
      // which is the free-running tail this clause exists to avoid. Measured
      // 2026-08-26: with the flag set after the advance, prose's teacher-forced
      // rows read logit cosines of 0.63 … 0.99 and three determined rows
      // "failed"; setting it before, they read 0.9999 and pass. Same code, one
      // statement moved.
      if (!ok) forced = true;
      if (forced) {
        CHECK(uint32_t(gtok[p]) < Qwen35::kVocabUsed);
        eng.ingest({uint32_t(gtok[p])});
      } else {
        CHECK_EQ(eng.generate(1)[0], id);
      }
      read_logits();
    }
    // The census the ruling requires be DERIVED here rather than pasted from a
    // console log: which rows this golden file leaves undetermined.
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
                pname, v.det_exact, v.tie_agree, v.tie_member, v.n_determined, v.n_tie, kGen,
                v.first_bad < 0 ? "" : "   ** GATE FAILURE **");
    std::printf("  legacy strict count (every row, tie-break included): %u/%u\n", v.exact, kGen);
    std::printf("  engine:");
    for (uint32_t id : etok) std::printf(" %u", id);
    std::printf("\n  golden:");
    for (uint32_t p = 0; p < kGen; ++p) std::printf(" %u", uint32_t(gtok[p]));
    std::printf("\n");
    CHECK_EQ(v.n_determined + v.n_tie, kGen);
    verdicts.push_back(v);
  }

  // ---- 5. the verdict, once every prompt has printed its diagnostics -------
  std::printf("\n================ golden gate ================\n");
  std::printf("  the gate is the token columns alone; every cosine below is a diagnostic.\n"
              "  det-exact: rows whose golden argmax is UNIQUE, engine element-exact - full\n"
              "             strictness, and every one of them must pass.\n"
              "  tie-agree / tie-member: rows where the golden top-1 is NOT unique. The\n"
              "             engine's id must be a member of the golden argmax set; -agree\n"
              "             is the member torch's lowest-index tie-break happened to pick.\n"
              "  After any divergence the walk is teacher-forced on the oracle's token, so\n"
              "  every determined row is judged on the reference's own context.\n"
              "  prompt   ids   det-exact  tie-agree  tie-member   tap min cos (layer,t)   "
              "L63 tail   gdn min cos   logit min cos\n");
  for (const Verdict& v : verdicts)
    std::printf("  %-7s %4u    %2u/%-2u        %2u         %2u        %.9f (%2u,%2u)   %.9f  "
                "%.9f (L%u)  %.9f\n",
                v.name.c_str(), v.n_prompt, v.det_exact, v.n_determined, v.tie_agree,
                v.tie_member, v.tap_min_cos, v.tap_min_layer, v.tap_min_t, v.tail_min_cos,
                v.gdn_min_cos, v.gdn_min_layer, v.logit_min_cos);

  uint32_t tot_det = 0, tot_det_ok = 0, tot_tie = 0, tot_agree = 0, tot_member = 0;
  bool bad = false;
  for (const Verdict& v : verdicts) {
    tot_det += v.n_determined;
    tot_det_ok += v.det_exact;
    tot_tie += v.n_tie;
    tot_agree += v.tie_agree;
    tot_member += v.tie_member;
    if (v.det_exact != v.n_determined) {
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
    // gdn_state is NOT gated - see kBar's comment. It prints **LOW** per layer
    // like the tap does, and the token ids are what decide this test.
  }
  std::printf("  TOTAL: %u/%u determined rows exact, %u undetermined (%u agree + %u other"
              " member)\n", tot_det_ok, tot_det, tot_tie, tot_agree, tot_member);
  if (bad) return 1;
  std::printf("golden_gate_test OK: 3 prompts x %u greedy tokens - %u/%u determined rows"
              " element-exact against the CPU oracle, %u undetermined rows all inside the"
              " golden argmax set\n", kGen, tot_det_ok, tot_det, tot_tie);
  return 0;
}
