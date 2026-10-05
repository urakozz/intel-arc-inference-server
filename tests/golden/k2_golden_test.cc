// k2_golden_test - spec 18 §6 K2 (decode): K2-Horizon's engine against the CPU reference
// of 18a (modeling_k2_horizon.py, layer-streamed on the dequantised checkpoint), three prompts
// x 32 greedy tokens, with the per-layer ROUTING DIAGNOSTIC (plan 18b Review Focus 3).
//
// **The golden files** (spec 18a's tools/oracle/k2_ref.py `run`, merged; tools/oracle/README.md
// "The K2-Horizon reference" has the commands, docs/probe-k2-2026-10-05.md the facts):
//
//   <oracle dir>/<prompt>.ids                 K2's tokenisation, BOS 0 first (tokenize.py --bos)
//   <oracle dir>/<prompt>.golden.safetensors  dump.py's convention plus the routing dumps:
//     logits             F32  [T + gen][250624]  row t < T: prompt position t; row T + j: the
//                                                forward that consumed tokens[j] (teacher-forced)
//     tokens             I32  [gen]              the greedy continuation (gen = 32)
//     route.moe.ids.L{l} I32  [T + gen][8]       the MoE experts, ascending id (layers 3..47)
//     route.moe.w.L{l}   BF16 [T + gen][8]       their weights, rne((s / sum s) x 2.5)
//     route.moe.gap.L{l} F32  [T + gen]          8th minus 9th selection score (0 = a tie at the cut)
//     route.mova.*       the same for MoVA's 4 of 64 value experts (gap = 4th minus 5th)
//     resid.L{l}         BF16 [T][2560]          layer l's output (the tap diagnostic)
//   The reference runs in bf16 (k2_ref.py's default mode, bit-identical to the vendored HF model
//   with eager attention) - the int4 checkpoint's config.json says "dtype": "float16", AutoRound's
//   export setting, which is NOT what spec 18 §3's rounding points describe - and widens the F16
//   v_router.bias to fp32 (as the engine's loader does). Ties at the cut go to the LOWER id in
//   both (a stable descending sort there, the rank count here).
//
// **The gate** (golden_gate_test's tie-aware rule, golden_common.h): per prompt, ingest the
// ids one replay at a time, then for j = 0..31: the engine's sampled id against the golden
// decision row T - 1 + j (determined rows exact; a row whose top-1 is not unique accepts any
// member of the argmax set), then feed the GOLDEN token (teacher-forced throughout, so the
// routing rows of the generated positions are the reference's own contexts).
//
// **The routing diagnostic** (Review Focus 3): at every (row, MoVA/MoE layer) the engine's
// route rows (K2Engine::read_routes; ids ascending) against route.*.ids. A difference is a
// NEAR-TIE when the reference's gap at the cut is within tie_tol() (1e-3, PROPOSED: 18a's real-
// weight run prints the gap distribution and sets it; B70_K2_TIE_TOL overrides) and the sets
// differ by one expert. Any other difference FAILS. Printed: per layer, the first row where the
// ids differ (tie or not), and the worst weight difference in bf16 ulps over the matching sets
// (a diagnostic - the weights' fp32 sum order is torch's reduction against our rank order).
//
// **K2 on prefill (spec 18c)**: with `prefill` (or `prefill:<chunk>`) the prompt goes through
// K2Engine::prefill instead of one replay per id; the gate is the same rule from row T - 1 on
// (the prefill's head chooses the first token), and the routing diagnostic reads the prefill's
// own route rows for the prompt positions its last chunk holds (all of them in one chunk;
// plan 18c Review Focus 5). The tap diagnostic is decode's and is skipped there.
//
// argv: <snapshot> <oracle dir> [int8] [prefill | prefill:<chunk>]. Exit 77 (SKIP) when the checkpoint or the golden set
// (18a's oracle-out-k2/) is not on this machine - the Mac never has them.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "kernels/k2_kernels.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/k2/k2_engine.h"
#include "runtime/k2/k2_sizes.h"

namespace {

constexpr uint32_t kMaxLen = 16384, kGen = 32;
const char* const kPrompts[] = {"prose", "code", "cjk"};
// PROPOSED (spec 18 §6 K2): selection scores (sigmoid + bias, ~0..1.1) within this of each
// other at the top-k boundary are a near-tie. 18a measures the 8th / 9th and 4th / 5th gaps
// and sets it before the gate binds.
float tie_tol() {
  const char* e = std::getenv("B70_K2_TIE_TOL");
  return e && *e ? std::strtof(e, nullptr) : 1e-3f;
}

std::vector<uint32_t> read_ids(const std::string& p, uint32_t vocab) {
  std::ifstream f(p);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", p.c_str());
    std::exit(1);
  }
  std::vector<uint32_t> ids;
  for (long long v; f >> v;) {
    CHECK(v >= 0 && v < (long long)vocab);
    ids.push_back(uint32_t(v));
  }
  CHECK(!ids.empty());
  return ids;
}

struct RouteCount {
  uint64_t rows = 0, exact = 0, ties = 0, bad = 0;
  int worst_w = 0;
};

int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

// One (row, layer, router) comparison: the engine's K ids (ascending) and weights against the
// golden row's ids, weights and gap.
void compare_route(const uint32_t* got_ids, const uint32_t* got_w_bits, const int32_t* want_ids,
                   const uint16_t* want_w, float gap, uint32_t K, RouteCount& c, char& first_diff,
                   const char* what, const char* pname, uint32_t row, uint32_t layer) {
  ++c.rows;
  std::vector<uint32_t> g(got_ids, got_ids + K), w(K);
  for (uint32_t k = 0; k < K; ++k) w[k] = uint32_t(want_ids[k]);
  if (g == w) {
    ++c.exact;
    for (uint32_t k = 0; k < K; ++k) {
      float gw;
      std::memcpy(&gw, &got_w_bits[k], 4);
      c.worst_w = std::max(c.worst_w, ulps(common::f32_to_bf16(gw), want_w[k]));
    }
    return;
  }
  if (!first_diff) {
    std::printf("    %s %s L%u: first differing row %u (reference gap %.3e)\n", pname, what, layer,
                row, double(gap));
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
    std::printf("    %s %s L%u row %u: expert set differs beyond a near-tie (reference gap %.3e, "
                "%zu experts differ)\n", pname, what, layer, row, double(gap), only_g.size());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-k2";
  // Flags after the oracle dir, any order: `int8` (spec 9's head), `prefill` or
  // `prefill:<chunk>` (spec 18c: the prompt through K2Engine::prefill).
  bool int8 = false, prefill = false;
  uint32_t chunk = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "int8") {
      int8 = true;
    } else if (a == "prefill") {
      prefill = true;
    } else if (a.rfind("prefill:", 0) == 0) {
      prefill = true;
      chunk = uint32_t(std::strtoul(a.c_str() + 8, nullptr, 10));
    } else {
      std::fprintf(stderr, "k2_golden_test: unknown flag %s (int8, prefill, prefill:<chunk>)\n", a.c_str());
      return 2;
    }
  }
  try {
    (void)loader::resolve_snapshot(snap);
  } catch (const std::exception& e) {
    std::printf("SKIP: no K2-Horizon checkpoint at '%s' (%s)\n", snap.c_str(), e.what());
    return 77;
  }
  for (const char* p : kPrompts)
    if (!golden::exists(gdir + "/" + p + ".golden.safetensors") || !golden::exists(gdir + "/" + p + ".ids")) {
      std::printf("SKIP: %s/%s.{ids,golden.safetensors} is absent - spec 18a's golden set "
                  "(oracle-out-k2/, box only; the format is this file's header)\n", gdir.c_str(), p);
      return 77;
    }

  l0::Context ctx(0);
  loader::K2LoadedModel model = loader::load_k2(
      ctx, snap, kMaxLen, int8 ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint);
  const model::K2Desc& d = *model.desc;
  runtime::k2::K2Engine eng(ctx, std::move(model), kMaxLen, /*debug_tap=*/true);
  CHECK_EQ(eng.step().kernel_count, runtime::k2::decode_launches(d));
  runtime::Control* ctl = eng.buffers().control.as<runtime::Control>();
  std::printf("lm_head %s, %s attention (B70_K2_ATTN), %zu launches per token, near-tie tolerance "
              "%.1e, prompt by %s\n", int8 ? "int8" : "bf16", runtime::k2::k2_attn_name(runtime::k2::k2_attn()),
              eng.step().kernel_count, double(tie_tol()), prefill ? "K2Engine::prefill" : "decode replays");

  bool gate_ok = true;
  RouteCount moe_c, mova_c;
  std::vector<double> sa, sb;
  for (const char* pname : kPrompts) {
    const std::vector<uint32_t> ids = read_ids(gdir + "/" + pname + ".ids", d.vocab);
    golden::Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size());
    const uint32_t rows = uint32_t(g.dim("logits", 2, 0));
    CHECK_EQ(g.dim("logits", 2, 1), uint64_t(d.vocab));
    CHECK_EQ(rows, T + kGen);
    CHECK_EQ(g.dim("tokens", 1, 0), uint64_t(kGen));
    const float* glog = g.f32("logits", size_t(rows) * d.vocab);
    const int32_t* gtok = g.i32("tokens", kGen);
    const bool have_routes = g.has("route.moe.ids.L" + std::to_string(d.dense_layers));
    std::printf("\n================ %s: %u prompt ids, %u generated%s ================\n", pname, T,
                kGen, have_routes ? "" : " (NO routing dumps: the diagnostic is skipped)");

    std::vector<char> moe_first(d.layers, 0), mova_first(d.layers, 0);
    std::vector<double> tap_min(d.layers, 1.0);
    // `route_row(l, which)`: the engine's route row of golden row `row` for (layer, router).
    auto check_route_rows = [&](uint32_t row, const std::function<const uint32_t*(uint32_t, int)>& route_row) {
      if (!have_routes) return;
      for (uint32_t l = d.dense_layers; l < d.layers; ++l) {
        const std::string ls = std::to_string(l);
        for (int which = 0; which < 2; ++which) {   // 0 MoVA, 1 MoE
          const uint32_t K = which ? d.top_k : d.value_top_k;
          const std::string pre = which ? "route.moe." : "route.mova.";
          const int32_t* gi = g.i32(pre + "ids.L" + ls, size_t(rows) * K) + size_t(row) * K;
          const uint16_t* gw = g.bf16(pre + "w.L" + ls, size_t(rows) * K) + size_t(row) * K;
          const float gap = g.f32(pre + "gap.L" + ls, rows)[row];
          const uint32_t* er = route_row(l, which);
          compare_route(er + kernels::k2::route::kIds, er + kernels::k2::route::kWeights, gi, gw, gap, K,
                        which ? moe_c : mova_c, which ? moe_first[l] : mova_first[l],
                        which ? "MoE" : "MoVA", pname, row, l);
        }
      }
    };
    // Decode's route rows of the last replay (K2Engine::read_routes).
    auto check_routes = [&](uint32_t row) {
      if (!have_routes) return;
      const std::vector<uint32_t> r = eng.read_routes();
      check_route_rows(row, [&](uint32_t l, int which) {
        return r.data() + (which ? runtime::k2::moe_route_at(l) : runtime::k2::mova_route_at(l)) / 4;
      });
    };

    eng.reset();
    // Spec 18c, K2 on prefill: the prompt in one K2Engine::prefill (chunks of `chunk`), the
    // routing diagnostic on the rows the last chunk's route buffer holds (Review Focus 5: all
    // of them in one chunk), then the 32 teacher-forced decode steps below exactly as decode.
    if (prefill) {
      eng.prefill(ids, chunk);
      const uint32_t c = chunk ? chunk : runtime::k2::kPfC;
      const uint32_t first = ((T - 1) / c) * c;   // the last chunk's first position
      const std::vector<uint32_t> r = eng.read_prefill_routes();
      for (uint32_t t = first; t < T; ++t)
        check_route_rows(t, [&](uint32_t l, int which) {
          return r.data() + runtime::k2::pf_route_at(l, uint32_t(which)) / 4 + size_t(t - first) * 32;
        });
      std::printf("  prefill: %u ids in chunks of %u, routing diagnostic on rows %u..%u\n", T, c, first, T - 1);
    }
    for (uint32_t t = 0; t < T && !prefill; ++t) {
      eng.ingest({ids[t]});
      check_routes(t);
      // The tap diagnostic on the MoE layers (their tap IS the layer output, k2_capture.h).
      if (g.has("resid.L" + std::to_string(d.layers - 1))) {
        const std::vector<uint16_t> tap = eng.read_debug_resid();
        for (uint32_t l = d.dense_layers; l < d.layers; ++l) {
          const uint16_t* want = g.bf16("resid.L" + std::to_string(l), size_t(T) * d.hidden) + size_t(t) * d.hidden;
          const golden::Metric mt = golden::compare_bf16(tap.data() + size_t(l) * d.hidden, want, d.hidden, sa, sb);
          tap_min[l] = std::min(tap_min[l], mt.cos);
        }
      }
    }
    uint32_t det = 0, det_ok = 0, ties = 0, tie_ok = 0;
    int first_bad = -1;
    double logit_min = 1.0;
    for (uint32_t j = 0; j < kGen; ++j) {
      const uint32_t row = T - 1 + j;
      const uint32_t got = ctl->cur_token[0];
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
    std::printf("  gate: %u / %u determined rows exact, %u / %u tie rows in the argmax set%s; "
                "logits worst cosine %.6f\n", det_ok, det, tie_ok, ties,
                ok ? "" : (" - FIRST BAD STEP " + std::to_string(first_bad)).c_str(), logit_min);
    if (!prefill && g.has("resid.L" + std::to_string(d.layers - 1))) {
      std::printf("  tap (MoE layers, min cosine over the prompt):");
      for (uint32_t l = d.dense_layers; l < d.layers; ++l) std::printf(" %u:%.4f", l, tap_min[l]);
      std::printf("\n");
    }
  }
  std::printf("\nrouting: MoE %llu rows (%llu exact, %llu near-ties, %llu differing; weights worst %d "
              "bf16 ulps); MoVA %llu rows (%llu exact, %llu near-ties, %llu differing; weights worst %d)\n",
              (unsigned long long)moe_c.rows, (unsigned long long)moe_c.exact,
              (unsigned long long)moe_c.ties, (unsigned long long)moe_c.bad, moe_c.worst_w,
              (unsigned long long)mova_c.rows, (unsigned long long)mova_c.exact,
              (unsigned long long)mova_c.ties, (unsigned long long)mova_c.bad, mova_c.worst_w);
  CHECK(gate_ok);
  CHECK_EQ(moe_c.bad, uint64_t(0));
  CHECK_EQ(mova_c.bad, uint64_t(0));
  std::puts("k2_golden_test OK");
  return 0;
}
