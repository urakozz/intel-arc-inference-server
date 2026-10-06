// kolibri_golden_test - spec 20 §5 KL2 (decode): Kolibri-1's engine against tools/oracle/kolibri_ref.py
// (spec 20a) - each golden prompt x 32 greedy tokens - with the per-layer ROUTING DIAGNOSTIC.
// k2_golden_test's copy (tests/golden/k2_golden_test.cc), Kolibri's semantics.
//
// **The golden files** (kolibri_ref.py `run`; tools/box_validate/kolibri_oracle.sh synth|real):
//   <oracle dir>/<prompt>.ids                 Kolibri's tokenisation, no BOS
//   <oracle dir>/<prompt>.golden.safetensors
//     logits             F32  [T + gen][128000]   row t < T: prompt position t; row T + j: the forward
//                                                 that consumed tokens[j]
//     tokens             I32  [gen]               the greedy continuation (gen = 32)
//     route.moe.ids.L{l} I32  [T + gen][6]        the experts, ascending id
//     route.moe.w.L{l}   F32  [T + gen][6]        their sigmoid(logit) weights (fp32, never rounded)
//     route.moe.gap.L{l} F32  [T + gen]           6th minus 7th selection score (0 = a tie at the cut)
//     resid.L{l}         BF16 [T][2560]           layer l's output (the tap diagnostic)
// The prompts read: prose, de_prose (the synthetic sets) and code, de_chat (the real set) - each whose
// files exist; prose must.
//
// **The gate** (golden_common.h's tie-aware rule): ingest the ids one replay at a time, then for
// j = 0..31 the engine's sampled id against the golden decision row T - 1 + j (determined rows exact;
// a row whose top-1 is not unique accepts any member of the argmax set), then feed the GOLDEN token.
//
// **The routing diagnostic**: at every (row, layer) the engine's route row (ids ascending) against
// route.moe.ids. A set difference is a NEAR-TIE when the reference's gap at the cut is within
// tie_tol() (B70_KOL_TIE_TOL, 1e-2 PROPOSED - kolibri_oracle.sh prints the gap distribution) and the
// sets differ by one expert; any other difference FAILS. Printed per layer: the first row whose ids
// differ, the near-tie count, the worst weight difference in fp32 ulps over the matching sets (the
// weights are sigmoid of the fp32 logit: OpenCL's exp against torch's).
//
// **KL2 on prefill (spec 20d)**: with `prefill` (or `prefill:<chunk>`) the prompt goes through
// KolibriEngine::prefill instead of one replay per id; the gate is the same rule from row T - 1 on (the
// prefill's head chooses the first token), the routing diagnostic reads the prefill's route rows of the
// last chunk (all of the prompt's in one chunk; the last <chunk> rows otherwise), and the tap is not read
// (prefill writes no tap). The 32 teacher-forced steps after it are decode's, as before.
//
// argv: <checkpoint> <oracle dir> [int8] [pp2 | pp2:<split>] [copy | peer] [prefill | prefill:<chunk>]
// (`pp2`: spec 20c Task 6 - the engine on two cards). Exit 77 (SKIP) when the checkpoint or the golden
// set is absent, or the model does not fit the asked placement.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "kernels/kolibri_kernels.h"
#include "runtime/kolibri_rig.h"
#include "runtime/control.h"
#include "runtime/kolibri/kolibri_sizes.h"

namespace {

constexpr uint32_t kMaxLen = 16384, kGen = 32;
const char* const kPrompts[] = {"prose", "code", "de_prose", "de_chat"};
float tie_tol() {
  const char* e = std::getenv("B70_KOL_TIE_TOL");
  return e && *e ? std::strtof(e, nullptr) : 1e-2f;   // PROPOSED (header)
}

struct RouteCount {
  uint64_t rows = 0, exact = 0, ties = 0, bad = 0;
  int64_t worst_w = 0;
};

int64_t ulps32(float a, float b) {
  auto key = [](float v) {
    int32_t i;
    std::memcpy(&i, &v, 4);
    return i < 0 ? -int64_t(i & 0x7FFFFFFF) : int64_t(i);
  };
  return std::llabs(key(a) - key(b));
}

void compare_route(const uint32_t* row, const int32_t* want_ids, const float* want_w, float gap, uint32_t K,
                   RouteCount& c, char& first_diff, const char* pname, uint32_t r, uint32_t layer) {
  ++c.rows;
  std::vector<uint32_t> g(row + kernels::kolibri::route::kIds, row + kernels::kolibri::route::kIds + K), w(K);
  for (uint32_t k = 0; k < K; ++k) w[k] = uint32_t(want_ids[k]);
  if (g == w) {
    ++c.exact;
    for (uint32_t k = 0; k < K; ++k) {
      float gw;
      std::memcpy(&gw, &row[kernels::kolibri::route::kWeights + k], 4);
      c.worst_w = std::max(c.worst_w, ulps32(gw, want_w[k]));
    }
    return;
  }
  if (!first_diff) {
    std::printf("    %s L%u: first differing row %u (reference gap %.3e)\n", pname, layer, r, double(gap));
    first_diff = 1;
  }
  std::vector<uint32_t> gs = g, ws = w, only_g, only_w;
  std::sort(gs.begin(), gs.end());
  std::sort(ws.begin(), ws.end());
  std::set_difference(gs.begin(), gs.end(), ws.begin(), ws.end(), std::back_inserter(only_g));
  std::set_difference(ws.begin(), ws.end(), gs.begin(), gs.end(), std::back_inserter(only_w));
  if (gap <= tie_tol() && only_g.size() == 1 && only_w.size() == 1) {
    ++c.ties;
    return;
  }
  ++c.bad;
  if (c.bad <= 10)
    std::printf("    %s L%u row %u: expert set differs beyond a near-tie (reference gap %.3e, %zu experts differ)\n",
                pname, layer, r, double(gap), only_g.size());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-kolibri";
  kolibri_rig::Options o;
  o.max_len = kMaxLen;
  o.debug_tap = true;
  bool prefill = false;   // spec 20d: the prompt through KolibriEngine::prefill
  uint32_t chunk = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "prefill") {
      prefill = true;
    } else if (a.rfind("prefill:", 0) == 0) {
      prefill = true;
      chunk = uint32_t(std::strtoul(a.c_str() + 8, nullptr, 10));
    } else if (a == "int8") {
      o.int8_head = true;
    } else if (a == "pp2") {
      o.devices = 2;
    } else if (a.rfind("pp2:", 0) == 0) {
      o.devices = 2;
      o.split = uint32_t(std::strtoul(a.c_str() + 4, nullptr, 10));
    } else if (a == "copy" || a == "peer") {
      o.handoff = a == "peer" ? runtime::PpHandoff::Peer : runtime::PpHandoff::Copy;
    } else {
      std::fprintf(stderr, "kolibri_golden_test: unknown flag %s (int8, pp2[:split], copy, peer, prefill[:chunk])\n",
                   a.c_str());
      return 2;
    }
  }
  const std::string snap = kolibri_rig::kolibri_snapshot(arg);
  if (snap.empty()) {
    std::printf("SKIP: no Kolibri-1 checkpoint at '%s' (the synthetic sets: kolibri_oracle.sh synth; the real "
                "one: spec 20b)\n", arg.c_str());
    return 77;
  }
  std::vector<std::string> prompts;
  for (const char* p : kPrompts)
    if (golden::exists(gdir + "/" + p + ".golden.safetensors") && golden::exists(gdir + "/" + p + ".ids"))
      prompts.push_back(p);
  if (prompts.empty() || prompts[0] != "prose") {
    std::printf("SKIP: %s/prose.{ids,golden.safetensors} is absent (tools/box_validate/kolibri_oracle.sh)\n", gdir.c_str());
    return 77;
  }
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(snap);
  if (o.devices == 1 &&
      !runtime::kolibri::fits_one_card(pre, o.int8_head, size_t(32530000000ull), size_t(1.5e9), prefill)) {
    std::printf("SKIP: %s does not fit one card - run with pp2\n", snap.c_str());
    return 77;
  }
  if (o.devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: pp2 needs two GPUs, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  kolibri_rig::Rig rig;
  kolibri_rig::build(rig, snap, o);
  runtime::kolibri::KolibriEngine& eng = *rig.eng;
  const model::Kolibri1Desc& d = eng.model().desc;
  std::printf("%s: %u layers, %s attention arm, lm_head %s, %s attention (B70_KOLIBRI_ATTN), %u device(s)%s, %zu "
              "launches a token, near-tie tolerance %.1e, the prompt by %s\n", snap.c_str(), d.layers,
              model::kol_attn_form_name(d.attn),
              o.int8_head ? "int8" : "bf16", runtime::kolibri::kol_attn_name(eng.attention()), eng.devices(),
              eng.devices() == 2 ? (std::string(", split ") + std::to_string(eng.split()) + " " +
                                    runtime::pp_handoff_name(eng.handoff())).c_str() : "",
              eng.launches(), double(tie_tol()), prefill ? "KolibriEngine::prefill" : "decode replays");

  bool gate_ok = true;
  RouteCount rc;
  std::vector<double> sa, sb;
  for (const std::string& pname : prompts) {
    const std::vector<uint32_t> ids = kolibri_rig::read_ids(gdir + "/" + pname + ".ids", d.vocab);
    CHECK(!ids.empty());
    golden::Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size());
    const uint32_t rows = uint32_t(g.dim("logits", 2, 0));
    CHECK_EQ(g.dim("logits", 2, 1), uint64_t(d.vocab));
    CHECK_EQ(rows, T + kGen);
    const float* glog = g.f32("logits", size_t(rows) * d.vocab);
    const int32_t* gtok = g.i32("tokens", kGen);
    std::printf("\n================ %s: %u prompt ids, %u generated ================\n", pname.c_str(), T, kGen);
    std::vector<char> first(d.layers, 0);
    std::vector<double> tap_min(d.layers, 1.0);
    auto check_routes = [&](uint32_t row) {
      const std::vector<uint32_t> r = eng.read_routes();
      for (uint32_t l = 0; l < d.layers; ++l) {
        const std::string ls = std::to_string(l);
        if (!g.has("route.moe.ids.L" + ls)) continue;
        const int32_t* gi = g.i32("route.moe.ids.L" + ls, size_t(rows) * d.top_k) + size_t(row) * d.top_k;
        const float* gw = g.f32("route.moe.w.L" + ls, size_t(rows) * d.top_k) + size_t(row) * d.top_k;
        const float gap = g.f32("route.moe.gap.L" + ls, rows)[row];
        compare_route(r.data() + size_t(l) * runtime::kolibri::kRouteWords, gi, gw, gap, d.top_k, rc, first[l],
                      pname.c_str(), row, l);
      }
    };
    eng.reset();
    // Spec 20d, KL2 on prefill: the prompt in one KolibriEngine::prefill (chunks of `chunk`), the routing
    // diagnostic on the rows the last chunk's route buffer holds, then the 32 teacher-forced decode steps.
    if (prefill) {
      eng.prefill(ids, chunk);
      const uint32_t c = chunk ? chunk : runtime::kolibri::kPfC;
      const uint32_t row0 = ((T - 1) / c) * c;   // the last chunk's first position
      const std::vector<uint32_t> r = eng.read_prefill_routes();
      for (uint32_t t = row0; t < T; ++t)
        for (uint32_t l = 0; l < d.layers; ++l) {
          const std::string ls = std::to_string(l);
          if (!g.has("route.moe.ids.L" + ls)) continue;
          const int32_t* gi = g.i32("route.moe.ids.L" + ls, size_t(rows) * d.top_k) + size_t(t) * d.top_k;
          const float* gw = g.f32("route.moe.w.L" + ls, size_t(rows) * d.top_k) + size_t(t) * d.top_k;
          const float gap = g.f32("route.moe.gap.L" + ls, rows)[t];
          compare_route(r.data() + runtime::kolibri::pf_route_at(l) / 4 + size_t(t - row0) * runtime::kolibri::kRouteWords,
                        gi, gw, gap, d.top_k, rc, first[l], pname.c_str(), t, l);
        }
      std::printf("  prefill: %u ids in chunks of %u (%zu launches), routing diagnostic on rows %u..%u\n", T, c,
                  eng.prefill_launches(), row0, T - 1);
    }
    for (uint32_t t = 0; t < T && !prefill; ++t) {
      eng.ingest({ids[t]});
      check_routes(t);
      const std::vector<uint16_t> tap = eng.read_debug_resid();
      for (uint32_t l = 0; l < d.layers; ++l) {
        if (!g.has("resid.L" + std::to_string(l))) continue;
        const uint16_t* want = g.bf16("resid.L" + std::to_string(l), size_t(T) * d.hidden) + size_t(t) * d.hidden;
        tap_min[l] = std::min(tap_min[l], golden::compare_bf16(tap.data() + size_t(l) * d.hidden, want, d.hidden, sa, sb).cos);
      }
    }
    uint32_t det = 0, det_ok = 0, ties = 0, tie_ok = 0;
    int first_bad = -1;
    double logit_min = 1.0;
    for (uint32_t j = 0; j < kGen; ++j) {
      const uint32_t row = T - 1 + j;
      const uint32_t got = eng.control(eng.devices() - 1).cur_token[0];
      const std::vector<float> lg = eng.read_logits();
      logit_min = std::min(logit_min, golden::compare_f32(lg.data(), glog + size_t(row) * d.vocab, d.vocab, sa, sb).cos);
      const golden::GoldenDecision dec = golden::golden_decision(glog + size_t(row) * d.vocab, d.vocab, d.vocab_used);
      if (dec.determined()) {
        ++det;
        if (got == dec.set[0]) ++det_ok;
        else if (first_bad < 0) first_bad = int(j);
      } else {
        ++ties;
        if (dec.contains(got)) ++tie_ok;
        else if (first_bad < 0) first_bad = int(j);
      }
      if (got != uint32_t(gtok[j])) {
        std::printf("  step %u: engine %u, golden %u (%s)\n", j, got, uint32_t(gtok[j]),
                    dec.determined() ? "determined" : "tie row");
        golden::print_top5("engine", lg.data(), d.vocab, d.vocab_used);
        golden::print_top5("golden", glog + size_t(row) * d.vocab, d.vocab, d.vocab_used);
      }
      eng.ingest({uint32_t(gtok[j])});   // teacher-forced: row T + j
      check_routes(T + j);
    }
    const bool ok = det_ok == det && tie_ok == ties;
    gate_ok = gate_ok && ok;
    std::printf("  gate: %u / %u determined rows exact, %u / %u tie rows in the argmax set%s; logits worst cosine %.6f\n",
                det_ok, det, tie_ok, ties, ok ? "" : (" - FIRST BAD STEP " + std::to_string(first_bad)).c_str(),
                logit_min);
    if (!prefill) {   // prefill writes no tap
      std::printf("  tap (min cosine over the prompt):");
      for (uint32_t l = 0; l < d.layers; ++l) std::printf(" %u:%.4f", l, tap_min[l]);
      std::printf("\n");
    }
  }
  std::printf("\nrouting: %llu rows (%llu exact, %llu near-ties, %llu differing; weights worst %lld fp32 ulps)\n",
              (unsigned long long)rc.rows, (unsigned long long)rc.exact, (unsigned long long)rc.ties,
              (unsigned long long)rc.bad, (long long)rc.worst_w);
  CHECK(gate_ok);
  CHECK_EQ(rc.bad, uint64_t(0));
  std::puts("kolibri_golden_test OK");
  return 0;
}
