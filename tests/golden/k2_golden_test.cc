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
// differ by one expert. Printed: per layer, the first row where the ids differ (tie or not), and
// the worst weight difference in bf16 ulps over the matching sets (a diagnostic - the weights'
// fp32 sum order is torch's reduction against our rank order).
//   The GATE is a rate (box 2026-10-23, replacing "any non-tie difference fails"): the share of
// (row, layer) route rows whose set differs from the reference's may not exceed the share by
// which the bf16 reference ITSELF differs from exact arithmetic on the same rows - k2_ref.py
// --mode f32 --force <the golden's tokens>, the same weights in float32, teacher-forced so every
// row is the same context: MoE 2144 / 10665 (0.2010), MoVA 1169 / 10665 (0.1096) over the three
// prompts, at reference gaps up to 1.2e-1 (median 4.5e-3). No gap tolerance below that is one
// the reference meets, so the old rule (1e-3) failed the reference against itself. The engine
// measured: flash 2051 / 1126 (0.192 / 0.106), eager 2173 / 1130 (0.204 / 0.106).
// B70_K2_ROUTE_RATE=<moe>,<mova> overrides.
//
// **K2 on prefill (spec 18c)**: with `prefill` (or `prefill:<chunk>`) the prompt goes through
// K2Engine::prefill instead of one replay per id; the gate is the same rule from row T - 1 on
// (the prefill's head chooses the first token), and the routing diagnostic reads the prefill's
// own route rows for the prompt positions its last chunk holds (all of them in one chunk;
// plan 18c Review Focus 5). The tap diagnostic is decode's and is skipped there.
//
// **Spec 18e (the `_kv8` twins)**: under B70_KV_CACHE=int8 the engine holds the int8 rotkv cache
// (K2Engine's default form) and the same gate, the same tie rule and the same routing diagnostic
// run unchanged - spec 12 Q2's rule; the logits' worst cosine is printed per prompt beside bf16's.
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
#include <utility>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "kernels/k2_kernels.h"
#include "l0/cmdlist.h"
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
// The routing gate's rates (the header's measurement): {MoE, MoVA}.
std::pair<double, double> route_rate() {
  const char* e = std::getenv("B70_K2_ROUTE_RATE");
  if (e && *e) {
    const std::string s = e;
    const size_t c = s.find(',');
    CHECK(c != std::string::npos);
    return {std::stod(s.substr(0, c)), std::stod(s.substr(c + 1))};
  }
  return {2144.0 / 10665, 1169.0 / 10665};
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
  // `--judge=<f32 dir>` (operator ruling 2026-10-23, golden_common.h): the token gate judges
  // against the f32 teacher-forced reference; the bf16 golden is printed beside it.
  const std::string judge_dir = golden::take_judge(argc, argv);
  const golden::AcceptList accept = golden::take_accept(argc, argv);   // known rows (2026-10-23)
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
  std::printf("lm_head %s, %s attention (B70_K2_ATTN), %s KV (B70_KV_CACHE), %zu launches per token, "
              "near-tie tolerance %.1e, prompt by %s\n", int8 ? "int8" : "bf16",
              runtime::k2::k2_attn_name(runtime::k2::k2_attn()), runtime::kv_cache_name(eng.kv_cache()),
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
    const golden::Judge J =
        judge_dir.empty() ? golden::Judge{} : golden::open_judge(judge_dir, pname, gtok, kGen, rows);
    uint32_t bf16_vs_f32 = 0, bf16_det_bad = 0;   // the bf16 golden as a diagnostic beside the judge
    uint32_t det = 0, det_ok = 0, ties = 0, tie_ok = 0;
    int first_bad = -1;
    double logit_min = 1.0;
    for (uint32_t j = 0; j < kGen; ++j) {
      const uint32_t row = T - 1 + j;
      const uint32_t got = ctl->cur_token[0];
      const std::vector<float> lg = eng.read_logits();
      // Diagnostic: B70_K2_HEAD_CHECK=<prompt>:<step> - the int8 head (gemv_i8w) of that decision
      // row recomputed on the host from the device's own inputs: the final-normed hidden (x, bf16)
      // and the head's int8 rows + fp32 scales as loaded (tiled [N/16][K/16][16 k][16 n]),
      // (x . q) s in double, for the engine's top 2 and the judge's top.
      if (const char* hc = std::getenv("B70_K2_HEAD_CHECK"); hc && int8) {
        const std::string s = hc;
        const size_t c1 = s.find(':');
        if (c1 != std::string::npos && s.substr(0, c1) == pname && std::stoul(s.substr(c1 + 1)) == j) {
          l0::CmdList im = l0::CmdList::immediate(ctx);
          const loader::DeviceWeight& lm = *eng.model().lm_head;
          const uint32_t K = d.hidden;
          std::vector<uint16_t> x(K);
          im.copy(x.data(), eng.buffers().x.ptr(), size_t(K) * 2);
          std::vector<uint32_t> ids;
          {
            std::vector<uint32_t> o(d.vocab_used);
            for (uint32_t i = 0; i < d.vocab_used; ++i) o[i] = i;
            std::partial_sort(o.begin(), o.begin() + 2, o.end(), [&](uint32_t a, uint32_t b) { return lg[a] > lg[b]; });
            ids = {o[0], o[1]};
            if (J) ids.push_back(golden::judge_row(J.row(row), std::min(J.V, d.vocab), d.vocab_used).top);
          }
          for (uint32_t n : ids) {
            std::vector<int8_t> tile(size_t(K) * 16);
            float sc = 0;
            im.copy(tile.data(), static_cast<const int8_t*>(lm.mem.ptr()) + size_t(n / 16) * K * 16, tile.size());
            im.copy(&sc, static_cast<const float*>(lm.scales->ptr()) + n, 4);
            double acc = 0;
            for (uint32_t k = 0; k < K; ++k) acc += double(common::bf16_to_f32(x[k])) * tile[size_t(k) * 16 + n % 16];
            std::printf("  head check step %u id %u: device %.6f, host (x . q) s %.6f (scale %.6g)\n", j, n,
                        double(lg[n]), acc * sc, double(sc));
          }
        }
      }
      const golden::Metric lm = golden::compare_f32(lg.data(), glog + size_t(row) * d.vocab, d.vocab, sa, sb);
      logit_min = std::min(logit_min, lm.cos);
      if (lm.cos < 0.99) {   // a low row: where it is, and the largest single difference
        uint32_t at = 0;
        float worst = 0;
        for (uint32_t v = 0; v < d.vocab; ++v) {
          const float e = std::fabs(lg[v] - glog[size_t(row) * d.vocab + v]);
          if (e > worst) {
            worst = e;
            at = v;
          }
        }
        std::printf("  step %u (row %u): logits cosine %.6f, |golden| %.1f, |err| %.1f; worst |diff| %.3f at id %u "
                    "(engine %.4f, golden %.4f)\n", j, row, lm.cos, lm.nb, lm.err, double(worst), at,
                    double(lg[at]), double(glog[size_t(row) * d.vocab + at]));
      }
      // The verdict: the f32 judge's decision row when one is given (operator ruling 2026-10-23,
      // golden_common.h), else the bf16 golden's. The bf16 golden's own verdict is printed.
      const golden::GoldenDecision gdec = golden::golden_decision(glog + size_t(row) * d.vocab, d.vocab, d.vocab_used);
      // The judge's row with the one-bf16-ulp tie rule (golden_common.h judge_row, the operator's
      // second ruling of 2026-10-23); without a judge the bf16 golden's exact rule.
      golden::JudgedRow jr;
      if (J) jr = golden::judge_row(J.row(row), std::min(J.V, d.vocab), d.vocab_used);
      else jr.d = gdec, jr.top = gdec.set[0];
      const golden::GoldenDecision& dec = jr.d;
      if (J && jr.top != gdec.set[0]) ++bf16_vs_f32;
      if (J && gdec.determined() && got != gdec.set[0]) ++bf16_det_bad;
      const bool row_fails = dec.determined() ? got != jr.top : !dec.contains(got);
      const bool acc = accept.has(pname, j);
      if (dec.determined()) {
        ++det;
        if (got == jr.top || (row_fails && acc)) ++det_ok;   // an accepted known row counts as kept
        else if (first_bad < 0) first_bad = int(j);
      } else {
        ++ties;
        if (dec.contains(got) || (row_fails && acc)) ++tie_ok;
        else if (first_bad < 0) first_bad = int(j);
      }
      if (acc)
        std::printf("  step %u: %s\n", j,
                    row_fails ? (std::string(golden::kAcceptNote) + ", gap " + std::to_string(jr.gap)).c_str()
                              : "accepted row now passes - remove it");
      if (got != uint32_t(gtok[j]) || got != jr.top || jr.near) {
        std::printf("  step %u: engine %u, golden %u%s (%s%s)", j, got, uint32_t(gtok[j]),
                    J ? (", f32 judge " + std::to_string(jr.top)).c_str() : "",
                    dec.determined() ? "determined" : "tie row",
                    dec.contains(got) ? (J ? "; the f32 judge accepts the engine's" : "; in the argmax set")
                                      : "; GATE");
        if (jr.near)
          std::printf("  f32 near-tie: gap %.4f <= 1 bf16 ulp %.4f, %zu ids%s", jr.gap, jr.ulp, dec.set.size(),
                      got != jr.top && dec.contains(got) ? " - PASSES ONLY BY THE NEAR-TIE RULE" : "");
        std::printf("\n");
        golden::print_top5("engine", lg.data(), d.vocab, d.vocab_used);
        golden::print_top5("golden", glog + size_t(row) * d.vocab, d.vocab, d.vocab_used);
        if (J) golden::print_top5("f32", J.row(row), std::min(J.V, d.vocab), d.vocab_used);
      }
      eng.ingest({uint32_t(gtok[j])});   // teacher-forced: row T + j
      check_routes(T + j);
      // Diagnostic: B70_K2_ATTN_CHECK=<prompt>:<row> - the LAST layer's decode attention of that
      // row (its scratch still holds q, the gate and attn_out; the cache every row) against a
      // double-precision softmax(q·k / sqrt(128)) · v and the same gate, on the device's inputs.
      if (const char* chk = std::getenv("B70_K2_ATTN_CHECK"); chk && !prefill) {
        const std::string s = chk;
        const size_t c1 = s.find(':');
        if (c1 != std::string::npos && s.substr(0, c1) == pname && std::stoul(s.substr(c1 + 1)) == T + j) {
          l0::CmdList im = l0::CmdList::immediate(ctx);
          const runtime::k2::K2Buffers& b = eng.buffers();
          const uint32_t L = d.layers - 1, n = T + j + 1, HD = d.head_dim, KVN = d.kv_n();
          std::vector<float> q(d.q_n()), g(d.q_n());
          std::vector<uint16_t> out(d.q_n()), kk(size_t(n) * KVN), vv(size_t(n) * KVN);
          im.copy(q.data(), b.attn_q.ptr(), q.size() * 4);
          im.copy(g.data(), b.attn_gate.ptr(), g.size() * 4);
          im.copy(out.data(), b.attn_out.ptr(), out.size() * 2);
          im.copy(kk.data(), b.kv_k_layer(L), kk.size() * 2);
          im.copy(vv.data(), b.kv_v_layer(L), vv.size() * 2);
          double worst = 0, ab = 0, aa = 0, bb = 0, pmax = 0;
          for (uint32_t h = 0; h < d.q_heads; ++h) {
            const uint32_t kh = h / d.gqa();
            std::vector<double> sc(n);
            double mx = -1e300;
            for (uint32_t p = 0; p < n; ++p) {
              double a = 0;
              for (uint32_t t = 0; t < HD; ++t)
                a += double(q[h * HD + t]) * common::bf16_to_f32(kk[size_t(p) * KVN + kh * HD + t]);
              sc[p] = a / std::sqrt(double(HD));
              mx = std::max(mx, sc[p]);
            }
            double sum = 0;
            for (double& v : sc) sum += (v = std::exp(v - mx));
            pmax = std::max(pmax, *std::max_element(sc.begin(), sc.end()) / sum);
            for (uint32_t t = 0; t < HD; ++t) {
              double o = 0;
              for (uint32_t p = 0; p < n; ++p) o += sc[p] / sum * common::bf16_to_f32(vv[size_t(p) * KVN + kh * HD + t]);
              const double x = g[h * HD + t] * 0.6931471805599453;
              const double sp = x > 20 ? g[h * HD + t] : std::log1p(std::exp(x)) / 0.6931471805599453;
              const double want = o * sp, got = common::bf16_to_f32(out[h * HD + t]);
              worst = std::max(worst, std::fabs(want - got) / std::max(std::fabs(want), 1e-3));
              ab += want * got;
              aa += want * want;
              bb += got * got;
            }
          }
          std::printf("  attention check, row %u layer %u (%s): attn_out vs double cos %.6f, worst rel %.3e; "
                      "the most peaked head's max p %.6f\n", T + j, L,
                      runtime::k2::k2_attn_name(runtime::k2::k2_attn()), ab / std::sqrt(aa * bb), worst, pmax);
        }
      }
      // Diagnostic: B70_K2_DUMP=<prompt>:<row>:<file> writes that generated row's per-layer
      // tap (bf16 [layers][hidden]) - two runs (B70_K2_ATTN=flash / eager) compared layer by layer.
      if (const char* dump = std::getenv("B70_K2_DUMP"); dump && !prefill) {
        const std::string s = dump;
        const size_t c1 = s.find(':'), c2 = s.find(':', c1 + 1);
        if (c1 != std::string::npos && c2 != std::string::npos && s.substr(0, c1) == pname &&
            std::stoul(s.substr(c1 + 1, c2 - c1 - 1)) == T + j) {
          const std::vector<uint16_t> tap = eng.read_debug_resid();
          std::ofstream(s.substr(c2 + 1), std::ios::binary)
              .write(reinterpret_cast<const char*>(tap.data()), std::streamsize(tap.size() * 2));
          std::printf("  dumped row %u's per-layer tap to %s\n", T + j, s.substr(c2 + 1).c_str());
        }
      }
    }
    const bool ok = det_ok == det && tie_ok == ties;
    gate_ok = gate_ok && ok;
    std::printf("  gate: %u / %u determined rows exact, %u / %u tie rows in the argmax set%s; "
                "logits worst cosine %.6f\n", det_ok, det, tie_ok, ties,
                ok ? "" : (" - FIRST BAD STEP " + std::to_string(first_bad)).c_str(), logit_min);
    if (J)
      std::printf("  judged by the f32 reference %s; bf16 golden (diagnostic, not the gate): its argmax differs "
                  "from the judge on %u of %u decision rows, against it the engine misses %u determined row(s)\n",
                  judge_dir.c_str(), bf16_vs_f32, kGen, bf16_det_bad);
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
  const auto [moe_rate, mova_rate] = route_rate();
  const double moe_got = moe_c.rows ? double(moe_c.ties + moe_c.bad) / double(moe_c.rows) : 0.0;
  const double mova_got = mova_c.rows ? double(mova_c.ties + mova_c.bad) / double(mova_c.rows) : 0.0;
  // Operator ruling 2026-10-23: K2's default decode attention stays flash, and the eager
  // variant is RECORDED, not gated, on the routing rate - the number and the bar are printed,
  // the test does not fail on them (its ctest label `diagnostic` says so). The token gate binds
  // both.
  const bool eager = runtime::k2::k2_attn() == runtime::k2::K2Attn::Eager;
  const bool route_ok = moe_got <= moe_rate && mova_got <= mova_rate;
  std::printf("routing %s: differing sets MoE %.4f (bar %.4f), MoVA %.4f (bar %.4f) - the bf16 reference's own "
              "rate against its f32 control%s\n", eager ? "(eager: RECORDED, not gated)" : "gate", moe_got,
              moe_rate, mova_got, mova_rate, route_ok ? "" : (eager ? "  (above the bar)" : "  ** FAIL **"));
  CHECK(gate_ok);
  if (!eager) {
    CHECK(moe_got <= moe_rate);
    CHECK(mova_got <= mova_rate);
  }
  std::puts("k2_golden_test OK");
  return 0;
}
