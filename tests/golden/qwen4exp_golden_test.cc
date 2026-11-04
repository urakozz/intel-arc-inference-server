// qwen4exp_golden_test - spec 21 F3 (decode): Qwen3.8-Flash-Next's engine against tools/oracle/qwen4exp_ref.py
// (spec 21a: transformers 5.19.0, the engine-format reference - int4 experts and the int8 PLE rows dequantised) -
// each golden prompt x 32 greedy tokens - with the per-layer ROUTING diagnostic, the selection gate S per QSA layer
// and the PLE ids. kolibri_golden_test's copy (tests/golden/kolibri_golden_test.cc), this family's semantics.
//
// **The golden files** (qwen4exp_ref.py `run`; tools/box_validate/qwen4exp_oracle.sh synth | intel | intel-layers N):
//   <oracle dir>/<prompt>.ids, <prompt>.golden.safetensors
//     logits         F32  [Lr][248320]     rows T + gen - Lr .. T + gen - 1 (the tail; row t's forward consumed id t)
//     tokens         I32  [gen]            the greedy continuation (gen = 32)
//     route.ids.L{l} I32  [T + gen][10]    the router's top-10 in ITS order (descending p)
//     route.w.L{l}   F32  [T + gen][10]    their renormalised bf16 weights
//     route.gap.L{l} F32  [T + gen]        p(10) - p(11) of the fp32 softmax (0 = an exact tie at the cut)
//     qsa.sel.L{l}   I32  [R][512]         QSA layers: rows T + gen - R .. (p >= 2051), the selected blocks ascending
//     qsa.gap.L{l}   F32  [T + gen]        the 512th - 513th block score (+inf where <= 512 blocks are complete)
//     qsa.score.L{l} F32  [R][W]           the rows of qsa.sel: every complete block's score (n = (p + 1) / 4; -1 past)
//     route.top.ids.L{l} I32, route.top.p.L{l} F32  [T + gen][16]   the fp32 softmax's top 16 (descending p)
//                                          (qsa.score / route.top: since 2026-11-04 - older sets judge by the gaps)
//     ple.ids        I64  [T + gen][16]    the 16 PLE rows each position read
//     H.L{l}         BF16 [A][10240]       the 4-stream H after layer l, rows T + gen - A .. (the tap diagnostic)
// The prompts read: q4exp_short (must exist), q4exp_4k, q4exp_agentic - each whose files exist.
//
// **The gates** (golden_common.h's tie-aware rule; spec 21 §7 F3):
//   tokens   ingest the ids one replay at a time; for j = 0..31 the engine's id against the golden decision row
//            T - 1 + j (determined rows exact; a row whose top-1 is not unique accepts any member of the argmax
//            set), then feed the GOLDEN token
//   routing  at every (row, layer) the engine's top-10 SET against route.ids. A reference gap of exactly 0 is an
//            exact tie, which torch's topk breaks its own way - undetermined (spec 21 §12 finding 2). Otherwise, with
//            the reference's router scores (route.top.*), a difference is a ROUNDING SWAP when the weights of the
//            experts both took agree (cosine >= 0.999) and every expert the engine took instead is in the
//            reference's top 16 with its p within 2 x the row's score difference (the engine's p of rank 9 / 10 and
//            its weights in p, against the reference's p) below the reference's p of rank 9. The route row keeps no
//            other p, so that difference is a lower bound: a swap it cannot show still passes the gap rule (a
//            NEAR-TIE: one expert, the gap within B70_Q4_TIE_TOL 1e-3 - the rule before 2026-11-04, and the only one
//            for an old golden set without route.top.*); anything else FAILS
//   gate S   at every (row >= 2051, QSA layer) the engine's selected block set against qsa.sel. With the reference's
//            block scores (qsa.score) on the LAST QSA layer (the engine keeps that layer's scores:
//            read_last_qsa_scores / read_prefill_last_qsa_scores): a ROUNDING SWAP when the row's scores agree
//            (cosine >= 0.999) and every block the engine took instead lies within 2 x the row's largest score
//            difference below the reference's cut - qwen4exp_prefill_test's rule (2026-11-04: rounding-level
//            differences accepted, the bar unchanged, no tolerance widened); other QSA layers, and old golden sets
//            without qsa.score, by the gap rule (qsa.gap within B70_Q4_SEL_TOL 1e-3; 0 = undetermined); the judge used
//            is printed per prompt, swaps / near-tie rows counted
//   PLE      the 16 row ids of every row bitwise (integers)
//   `inject` the reference's selections fed in (Qwen4ExpEngine's injected capture: the identity list below 2052
//            visible positions, the reference's blocks + the tail after) - isolates the attention arithmetic: the
//            determined-row token gate must pass on EVERY row
// The free run passes S and the token gate on rows with no near-tie in any layer, and reports the others (a near
// tie upstream can move a determined row's logits: counted, not failed).
//
// **F3 on prefill (spec 21d)**: with `prefill` (or `prefill:<chunk>`) the prompt goes through Qwen4ExpEngine::prefill
// instead of one replay per id; the token gate is the same rule from row T - 1 on (the prefill's head chooses the
// first token), the routing diagnostic and gate S read the prefill's rows of the LAST chunk (its route rows and
// selection rows), with `inject` the reference's selections reach the prefill through set_prefill_injector; no tap
// (prefill writes none) and no PLE ids on the prompt rows (the decode rows after it check them). The 32 teacher-forced
// steps after it are decode's, as before.
//
// argv: <checkpoint> <oracle dir> [int8] [inject] [pp2 | pp2:<split>] [copy | peer] [layers:<N>] [prefill[:<chunk>]]
// Exit 77 (SKIP) when the checkpoint, its PLE file or the golden set is absent.
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
#include "golden_common.h"
#include "kernels/qwen4exp_kernels.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp_rig.h"

namespace {

namespace rq = runtime::qwen4exp;
namespace kr = kernels::qwen4exp::route;
constexpr uint32_t kGen = 32, kDenseRows = 2051;
const char* const kPrompts[] = {"q4exp_short", "q4exp_4k", "q4exp_agentic"};
float env_tol(const char* name, float dflt) {
  const char* e = std::getenv(name);
  return e && *e ? std::strtof(e, nullptr) : dflt;
}

struct Counts {
  uint64_t rows = 0, exact = 0, near = 0, swaps = 0, undetermined = 0, bad = 0;
  double cos_min = 1, deficit_ratio_max = 0;   // over the swaps judged by scores
};

constexpr double kScoreCos = 0.999;   // the prefill test's kScoreCos: a row's scores, engine vs reference

float bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
double cos_f32(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  if (aa == 0 || bb == 0) return aa == bb ? 1.0 : 0.0;   // two zero rows agree; one does not (never NaN)
  return ab / std::sqrt(aa * bb);
}

// The score judge (qwen4exp_prefill_test's rule, the operator's of 2026-11-04): a differing selection / expert set is
// a ROUNDING SWAP when the row's scores agree (cosine >= kScoreCos) and every item the engine took instead lies within
// twice the row's score difference ("noise") below the reference's cut - both sides run one top-k rule over their own
// scores, so rounding moves the cut's neighbours and nothing else; a larger deficit is a wrong row, not rounding.
struct Swap {
  double cos = 1, noise = 0, deficit = 0;
  size_t swapped = 0;
  bool ok() const { return cos >= kScoreCos && deficit <= 2 * noise; }
};
void count_swap(const Swap& s, Counts& c) {
  ++c.swaps;
  c.cos_min = std::min(c.cos_min, s.cos);
  if (s.noise > 0) c.deficit_ratio_max = std::max(c.deficit_ratio_max, s.deficit / s.noise);
}

// Selection: the engine's block scores `e` and the reference's `r` over the row's n complete blocks; `eb` / `rb` the
// selected block sets, ascending. The reference's cut is its 512th score (the lowest it took).
Swap judge_selection(const float* e, const float* r, uint32_t n, const std::vector<uint32_t>& eb,
                     const std::vector<uint32_t>& rb) {
  Swap s;
  for (uint32_t b = 0; b < n; ++b) s.noise = std::max(s.noise, std::fabs(double(e[b]) - r[b]));
  s.cos = cos_f32(e, r, n);
  double cut = INFINITY;
  for (uint32_t b : rb)
    if (b < n) cut = std::min(cut, double(r[b]));
  std::vector<uint32_t> in;
  std::set_difference(eb.begin(), eb.end(), rb.begin(), rb.end(), std::back_inserter(in));
  s.swapped = in.size();
  for (uint32_t b : in) s.deficit = std::max(s.deficit, b < n ? cut - r[b] : INFINITY);
  return s;
}

// The reference's router scores of one row (route.top.*: the fp32 softmax's top kTop, descending; route.w its weights
// in route.ids' order) - null pointers: an old golden set (the gap rule).
struct RouteRef {
  const int32_t* top_ids = nullptr;
  const float* top_p = nullptr;
  const float* w = nullptr;
  uint32_t top = 0;
};
// Routing: the row's score difference ("noise", in p) is the larger of the cut's - the engine's p of rank 9 / 10
// (route row kP10 / kP11) against the reference's - and the experts both took: the engine's renormalised weight in p
// (x the reference's top-10 sum) against the reference's p (the route row keeps no other p; the weight is bf16, so
// this term is known to half a bf16 ulp of the weight). The agreement is the cosine of the renormalised weights over
// the experts both took; every expert the engine took instead must be in the reference's top kTop, its p within
// 2 x noise below the reference's p of rank 9.
Swap judge_route(const uint32_t* row, const int32_t* want, const RouteRef& ref) {
  Swap s;
  const double e10 = bits(row[kr::kP10]), e11 = bits(row[kr::kP11]), r10 = ref.top_p[9], r11 = ref.top_p[10];
  s.noise = std::max(std::fabs(e10 - r10), std::fabs(e11 - r11));
  double sum10 = 0;
  for (uint32_t k = 0; k < 10; ++k) sum10 += ref.top_p[k];
  std::vector<float> ew, rw;
  for (uint32_t i = 0; i < 10; ++i) {
    const uint32_t id = row[kr::kIds + i];
    const int32_t* at = std::find(want, want + 10, int32_t(id));
    if (at != want + 10) {
      ew.push_back(bits(row[kr::kWeights + i]));
      rw.push_back(ref.w[at - want]);
      const int32_t* t = std::find(ref.top_ids, ref.top_ids + ref.top, int32_t(id));
      if (t != ref.top_ids + ref.top)
        s.noise = std::max(s.noise, std::fabs(double(ew.back()) * sum10 - ref.top_p[t - ref.top_ids]));
      continue;
    }
    ++s.swapped;
    const int32_t* t = std::find(ref.top_ids, ref.top_ids + ref.top, int32_t(id));
    s.deficit = std::max(s.deficit, t == ref.top_ids + ref.top ? INFINITY : r10 - ref.top_p[t - ref.top_ids]);
  }
  s.cos = cos_f32(ew.data(), rw.data(), ew.size());
  return s;
}

// The engine's route row (rank order) against the reference's top-10 (its topk order), as sets. With the reference's
// router scores (`ref.top_p`) a difference is judged by scores; without them (an old golden set) by the gap rule.
void compare_route(const uint32_t* row, const int32_t* want, float gap, float tol, const RouteRef& ref, Counts& c,
                   const char* p, uint32_t r, uint32_t l) {
  ++c.rows;
  std::vector<uint32_t> g(row + kr::kIds, row + kr::kIds + 10), w(want, want + 10);
  std::sort(g.begin(), g.end());
  std::sort(w.begin(), w.end());
  if (g == w) {
    ++c.exact;
    return;
  }
  std::vector<uint32_t> og, ow;
  std::set_difference(g.begin(), g.end(), w.begin(), w.end(), std::back_inserter(og));
  std::set_difference(w.begin(), w.end(), g.begin(), g.end(), std::back_inserter(ow));
  if (gap == 0.0f) {
    ++c.undetermined;
    return;
  }
  // The route row keeps no per-expert p past the engine's top 10, so the score noise below is a LOWER bound of the
  // row's (a pairwise swap is always within 2 x the true sup-norm difference): a difference the scores cannot show
  // to be rounding still passes the gap rule it passed before (one expert, gap <= B70_Q4_TIE_TOL) - counted apart.
  Swap s;
  if (ref.top_p) {
    s = judge_route(row, want, ref);
    if (s.ok()) {
      count_swap(s, c);
      return;
    }
  }
  if (gap <= tol && og.size() == 1 && ow.size() == 1) {
    ++c.near;
    return;
  }
  if (ref.top_p) {
    ++c.bad;
    if (c.bad <= 10)
      std::printf("    %s L%u row %u: the expert set differs beyond rounding (%zu experts, reference gap %.3e, weights "
                  "cosine %.6f, deficit %.3e, score noise %.3e)\n", p, l, r, og.size(), double(gap), s.cos, s.deficit,
                  s.noise);
    return;
  }
  ++c.bad;
  if (c.bad <= 10)
    std::printf("    %s L%u row %u: the expert set differs beyond a near-tie (gap %.3e, %zu experts)\n", p, l, r, double(gap),
                og.size());
}

// Gate S for one (row, QSA layer): `eb` / `rb` the engine's / reference's blocks (ascending); `esc` the engine's scores
// of the row (null: none readable for this layer) and `rsc` the reference's (null: an old golden set). Returns true
// for a near-tie / rounding row (the token gate then reports instead of failing downstream).
bool compare_selection(const std::vector<uint32_t>& eb, const std::vector<uint32_t>& rb, float gap, float tol,
                       const float* esc, const float* rsc, uint32_t n, Counts& c, const char* p, uint32_t row, uint32_t l,
                       const char* what) {
  if (eb == rb) {
    ++c.exact;
    return false;
  }
  if (gap == 0.0f) {
    ++c.undetermined;
    return true;
  }
  if (esc && rsc) {
    const Swap s = judge_selection(esc, rsc, n, eb, rb);
    if (s.ok()) {
      count_swap(s, c);
      return true;
    }
    if (++c.bad <= 10)
      std::printf("    %s QSA L%u row %u: %s differs beyond rounding (%zu blocks swapped, reference gap %.3e, scores "
                  "cosine %.6f, deficit %.3e, row noise %.3e)\n", p, l, row, what, s.swapped, double(gap), s.cos, s.deficit,
                  s.noise);
    return false;
  }
  if (gap <= tol) {
    ++c.near;
    return true;
  }
  if (++c.bad <= 10)
    std::printf("    %s QSA L%u row %u: %s differs beyond a near-tie (gap %.3e)\n", p, l, row, what, double(gap));
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-q4exp";
  qwen4exp_rig::Options o;
  o.max_len = 16384;
  o.debug_tap = true;
  bool inject = false, prefill = false;
  uint32_t chunk = 0;   // spec 21d: 0 = kPfC
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "int8") o.int8_head = true;
    else if (a == "inject") inject = true;
    else if (a == "prefill") prefill = true;
    else if (a.rfind("prefill:", 0) == 0) { prefill = true; chunk = uint32_t(std::strtoul(a.c_str() + 8, nullptr, 10)); }
    else if (a == "pp2") o.devices = 2;
    else if (a.rfind("pp2:", 0) == 0) { o.devices = 2; o.split = uint32_t(std::strtoul(a.c_str() + 4, nullptr, 10)); }
    else if (a == "copy" || a == "peer") o.handoff = a == "peer" ? runtime::PpHandoff::Peer : runtime::PpHandoff::Copy;
    else if (a.rfind("layers:", 0) == 0) o.layers = uint32_t(std::strtoul(a.c_str() + 7, nullptr, 10));
    else {
      std::fprintf(stderr, "qwen4exp_golden_test: unknown flag %s (int8, inject, pp2[:split], copy, peer, layers:N, "
                   "prefill[:chunk])\n", a.c_str());
      return 2;
    }
  }
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(arg, &why);
  if (snap.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  std::vector<std::string> prompts;
  for (const char* p : kPrompts)
    if (golden::exists(gdir + "/" + p + ".golden.safetensors") && golden::exists(gdir + "/" + p + ".ids")) prompts.push_back(p);
  if (prompts.empty() || prompts[0] != "q4exp_short") {
    std::printf("SKIP: %s/q4exp_short.{ids,golden.safetensors} is absent (tools/box_validate/qwen4exp_oracle.sh synth)\n",
                gdir.c_str());
    return 77;
  }
  if (o.devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: pp2 needs two GPUs, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const float tie_tol = env_tol("B70_Q4_TIE_TOL", 1e-3f), sel_tol = env_tol("B70_Q4_SEL_TOL", 1e-3f);
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::build(rig, snap, o);
  rq::Qwen4ExpEngine& eng = *rig.eng;
  const model::Qwen4ExpDesc& d = eng.model().desc;
  const uint32_t nq = d.qsa_before(d.layers);
  uint32_t last_qsa = d.layers;   // the QSA layer whose block scores the engine keeps (read_last_qsa_scores)
  for (uint32_t l = 0; l < d.layers; ++l)
    if (d.is_qsa(l)) last_qsa = l;
  eng.set_injected_selection(inject);
  std::printf("%s: %u layers, %s dense, lm_head %s, %s attention (B70_Q4_ATTN), %u device(s)%s, %zu launches a token%s; "
              "tolerances: routing %.1e (B70_Q4_TIE_TOL), selection %.1e (B70_Q4_SEL_TOL)\n", snap.c_str(), d.layers,
              model::q4_form_name(d.forms.dense), o.int8_head ? "int8" : "bf16", rq::q4_attn_name(eng.attention()),
              eng.devices(),
              eng.devices() == 2 ? (std::string(", split ") + std::to_string(eng.split()) + " " +
                                    runtime::pp_handoff_name(eng.handoff())).c_str() : "",
              eng.launches(), inject ? " (INJECTED selections)" : "", double(tie_tol), double(sel_tol));
  if (prefill) std::printf("the prompt through Qwen4ExpEngine::prefill (chunks of %u)\n", chunk ? chunk : rq::kPfC);

  bool gate_ok = true;
  Counts route_c, sel_c;
  uint64_t ple_rows = 0, ple_bad = 0, reported = 0;
  std::vector<double> sa, sb;
  for (const std::string& pname : prompts) {
    const std::vector<uint32_t> ids = qwen4exp_rig::read_ids(gdir + "/" + pname + ".ids", d.vocab);
    CHECK(!ids.empty());
    golden::Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size()), rows = T + kGen;
    if (rows > eng.max_len()) {
      std::printf("  %s: %u rows exceed max_len %u - skipped\n", pname.c_str(), rows, eng.max_len());
      continue;
    }
    const uint32_t Lr = uint32_t(g.dim("logits", 2, 0));
    CHECK_EQ(g.dim("logits", 2, 1), uint64_t(d.vocab));
    CHECK(Lr >= kGen + 1 && Lr <= rows);
    const uint32_t lrow0 = rows - Lr;
    const float* glog = g.f32("logits", size_t(Lr) * d.vocab);
    const int32_t* gtok = g.i32("tokens", kGen);
    const int64_t* gple = g.has("ple.ids") ? g.i64("ple.ids", size_t(rows) * 16) : nullptr;
    std::printf("\n================ %s: %u prompt ids, %u generated ================\n", pname.c_str(), T, kGen);
    // per QSA layer: the reference's selection rows and gaps
    struct Sel {
      uint32_t layer = 0, row0 = 0, R = 0, W = 0;
      const int32_t* blocks = nullptr;
      const float* gap = nullptr;
      const float* score = nullptr;   // qsa.score: [R][W], the rows of qsa.sel (null: an old golden set)
    };
    std::vector<Sel> sels;
    bool have_sel_scores = false, have_route_scores = true;
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (!d.is_qsa(l)) continue;
      Sel s;
      s.layer = l;
      const std::string ls = std::to_string(l);
      if (g.has("qsa.gap.L" + ls)) s.gap = g.f32("qsa.gap.L" + ls, rows);
      if (g.has("qsa.sel.L" + ls)) {
        s.R = uint32_t(g.dim("qsa.sel.L" + ls, 2, 0));
        s.row0 = rows - s.R;
        s.blocks = g.i32("qsa.sel.L" + ls, size_t(s.R) * 512);
      }
      if (s.blocks && s.R && g.has("qsa.score.L" + ls)) {
        CHECK_EQ(g.dim("qsa.score.L" + ls, 2, 0), uint64_t(s.R));
        s.W = uint32_t(g.dim("qsa.score.L" + ls, 2, 1));
        CHECK(s.W >= rows / 4);   // row rows - 1's n = rows / 4 blocks
        s.score = g.f32("qsa.score.L" + ls, size_t(s.R) * s.W);
        have_sel_scores = true;
      }
      sels.push_back(s);
    }
    // the reference's router scores per layer (route.top.*) - absent in an old golden set
    std::vector<const int32_t*> top_ids(d.layers, nullptr);
    std::vector<const float*> top_p(d.layers, nullptr), route_w(d.layers, nullptr);
    uint32_t top_k = 0;
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::string ls = std::to_string(l);
      if (!g.has("route.ids.L" + ls)) continue;
      if (!g.has("route.top.p.L" + ls) || !g.has("route.top.ids.L" + ls)) {
        have_route_scores = false;
        continue;
      }
      top_k = uint32_t(g.dim("route.top.p.L" + ls, 2, 1));
      CHECK(top_k >= 11);
      top_ids[l] = g.i32("route.top.ids.L" + ls, size_t(rows) * top_k);
      top_p[l] = g.f32("route.top.p.L" + ls, size_t(rows) * top_k);
      route_w[l] = g.f32("route.w.L" + ls, size_t(rows) * 10);
    }
    const auto route_ref = [&](uint32_t l, uint32_t row) {
      RouteRef r;
      if (!top_p[l]) return r;
      r.top = top_k;
      r.top_ids = top_ids[l] + size_t(row) * top_k;
      r.top_p = top_p[l] + size_t(row) * top_k;
      r.w = route_w[l] + size_t(row) * 10;
      return r;
    };
    std::printf("  judge: routing %s; gate S %s\n",
                have_route_scores ? "by the reference's router scores (route.top.*)"
                                  : "by the gap rule - this golden set has no router scores (route.top.*: regenerate "
                                    "it with qwen4exp_oracle.sh)",
                rows <= kDenseRows ? "- no rows past the cut (2051)"
                : have_sel_scores  ? ("by block scores on the last QSA layer (L" + std::to_string(last_qsa) +
                                      "), the gap rule on the others").c_str()
                                   : "by the gap rule - this golden set has no block scores (qsa.score.*: regenerate it "
                                     "with qwen4exp_oracle.sh)");
    std::vector<uint8_t> near_row(rows, 0);
    std::vector<double> tap_min(d.layers, 1.0);
    uint32_t act0 = rows;
    if (g.has("H.L0")) act0 = rows - uint32_t(g.dim("H.L0", 2, 0));
    // the reference's list of row p in QSA layer q (inject): identity below a cut, else its blocks + the tail
    const auto feed = [&](uint32_t p) {
      for (uint32_t q = 0; q < nq; ++q) {
        uint32_t* list = eng.injected_list(q);
        const uint32_t n = (p + 1) / 4;
        const Sel& s = sels[q];
        if (n <= 512 || !s.blocks || p < s.row0) {
          if (n > 512) {
            std::printf("  inject: %s has no selection row for position %u of QSA layer %u\n", pname.c_str(), p, s.layer);
            CHECK(false);
          }
          for (uint32_t i = 0; i <= p; ++i) list[i] = i;
          list[rq::kCountWord] = p + 1;
          continue;
        }
        const int32_t* b = s.blocks + size_t(p - s.row0) * 512;
        uint32_t at = 0;
        for (uint32_t k = 0; k < 512; ++k)
          for (uint32_t j = 0; j < 4; ++j) list[at++] = uint32_t(b[k]) * 4 + j;
        for (uint32_t q2 = n * 4; q2 <= p; ++q2) list[at++] = q2;
        list[rq::kCountWord] = at;
      }
    };
    const auto check_row = [&](uint32_t row) {
      const std::vector<uint32_t> r = eng.read_routes();
      for (uint32_t l = 0; l < d.layers; ++l) {
        const std::string ls = std::to_string(l);
        if (!g.has("route.ids.L" + ls)) continue;
        const int32_t* gi = g.i32("route.ids.L" + ls, size_t(rows) * 10) + size_t(row) * 10;
        const float gap = g.f32("route.gap.L" + ls, rows)[row];
        const uint64_t near0 = route_c.near + route_c.swaps + route_c.undetermined;
        compare_route(r.data() + size_t(l) * rq::kRouteWords, gi, gap, tie_tol, route_ref(l, row), route_c, pname.c_str(),
                      row, l);
        if (route_c.near + route_c.swaps + route_c.undetermined != near0) near_row[row] = 1;
      }
      for (const Sel& s : sels) {   // gate S
        if (row < kDenseRows || !s.blocks || row < s.row0) continue;
        ++sel_c.rows;
        const std::vector<uint32_t> got = eng.read_selection(s.layer);
        std::vector<uint32_t> gb, wb(s.blocks + size_t(row - s.row0) * 512, s.blocks + size_t(row - s.row0) * 512 + 512);
        for (uint32_t k = 0; k < 512 && 4 * k < got.size(); ++k) gb.push_back(got[4 * k] / 4);
        std::sort(wb.begin(), wb.end());
        const float gap = s.gap ? s.gap[row] : 1.0f;
        // the engine's scores of this row: the last QSA layer's survive the replay (read_last_qsa_scores), not injected
        std::vector<float> esc;
        if (gb != wb && s.score && s.layer == last_qsa && !inject) esc = eng.read_last_qsa_scores();
        if (compare_selection(gb, wb, gap, sel_tol, esc.empty() ? nullptr : esc.data(),
                              s.score ? s.score + size_t(row - s.row0) * s.W : nullptr, (row + 1) / 4, sel_c,
                              pname.c_str(), row, s.layer, "the selection"))
          near_row[row] = 1;
      }
      if (gple) {   // the PLE ids, exact
        ++ple_rows;
        const std::vector<uint64_t> got = eng.read_ple_ids();
        for (uint32_t h = 0; h < 16; ++h)
          if (got[h] != uint64_t(gple[size_t(row) * 16 + h])) {
            if (ple_bad++ < 5) std::printf("    %s row %u: PLE head %u id %llu, reference %lld\n", pname.c_str(), row, h,
                                           (unsigned long long)got[h], (long long)gple[size_t(row) * 16 + h]);
            break;
          }
      }
      if (row >= act0) {   // the tap
        const std::vector<uint16_t> tap = eng.read_debug_H();
        for (uint32_t l = 0; l < d.layers; ++l) {
          if (!g.has("H.L" + std::to_string(l))) continue;
          const uint16_t* want = g.bf16("H.L" + std::to_string(l), size_t(rows - act0) * d.hc_n()) +
                                 size_t(row - act0) * d.hc_n();
          tap_min[l] = std::min(tap_min[l], golden::compare_bf16(tap.data() + size_t(l) * d.hc_n(), want, d.hc_n(), sa, sb).cos);
        }
      }
    };
    eng.reset();
    if (prefill) {
      // Spec 21d: the prompt in one Qwen4ExpEngine::prefill; with `inject` the reference's lists per chunk
      if (inject)
        eng.set_prefill_injector([&](uint32_t q, uint32_t pos, uint32_t n_rows, uint32_t* lists) {
          for (uint32_t m = 0; m < n_rows; ++m) {
            uint32_t* list = lists + size_t(m) * rq::kListRow;
            const uint32_t p = pos + m, n = (p + 1) / 4;
            const Sel& s = sels[q];
            uint32_t at = 0;
            if (n <= 512 || !s.blocks || p < s.row0) {
              CHECK(n <= 512);   // a row past the cut needs the reference's selection row
              for (uint32_t i = 0; i <= p; ++i) list[at++] = i;
            } else {
              const int32_t* b = s.blocks + size_t(p - s.row0) * 512;
              std::vector<uint32_t> blocks(b, b + 512);
              std::sort(blocks.begin(), blocks.end());
              for (uint32_t k = 0; k < 512; ++k)
                for (uint32_t j = 0; j < 4; ++j) list[at++] = blocks[k] * 4 + j;
              for (uint32_t q2 = n * 4; q2 <= p; ++q2) list[at++] = q2;
            }
            list[rq::kCountWord] = at;
          }
        });
      eng.prefill(ids, chunk);
      const uint32_t c = chunk ? chunk : rq::kPfC, row0 = (T - 1) / c * c;
      const std::vector<uint32_t> r = eng.read_prefill_routes();
      for (uint32_t row = row0; row < T; ++row)
        for (uint32_t l = 0; l < d.layers; ++l) {
          const std::string ls = std::to_string(l);
          if (!g.has("route.ids.L" + ls)) continue;
          const int32_t* gi = g.i32("route.ids.L" + ls, size_t(rows) * 10) + size_t(row) * 10;
          const float gap = g.f32("route.gap.L" + ls, rows)[row];
          const uint64_t near0 = route_c.near + route_c.swaps + route_c.undetermined;
          compare_route(r.data() + rq::pf_route_at(l) / 4 + size_t(row - row0) * rq::kRouteWords, gi, gap, tie_tol,
                        route_ref(l, row), route_c, pname.c_str(), row, l);
          if (route_c.near + route_c.swaps + route_c.undetermined != near0) near_row[row] = 1;
        }
      // the engine's block scores of the last chunk's rows in the last QSA layer (rows [0, C) written when the chunk
      // had sparse rows; none under inject: the selection is the reference's)
      const bool sparse = T - 1 >= kDenseRows;
      const std::vector<float> pf_scores = have_sel_scores && sparse && !inject ? eng.read_prefill_last_qsa_scores()
                                                                                 : std::vector<float>();
      const uint32_t stride = eng.max_len() / d.idx_compress;
      for (const Sel& s : sels) {   // gate S on the last chunk's rows
        if (!s.blocks) continue;
        const std::vector<uint32_t> lists = eng.read_prefill_selection(s.layer);
        for (uint32_t row = std::max(row0, std::max(kDenseRows, s.row0)); row < T; ++row) {
          ++sel_c.rows;
          const uint32_t* got = lists.data() + size_t(row - row0) * rq::kListRow;
          std::vector<uint32_t> gb, wb(s.blocks + size_t(row - s.row0) * 512, s.blocks + size_t(row - s.row0) * 512 + 512);
          for (uint32_t k = 0; k < 512 && 4 * k < got[rq::kCountWord]; ++k) gb.push_back(got[4 * k] / 4);
          std::sort(wb.begin(), wb.end());
          const float gap = s.gap ? s.gap[row] : 1.0f;
          const float* esc = !pf_scores.empty() && s.layer == last_qsa ? pf_scores.data() + size_t(row - row0) * stride
                                                                       : nullptr;
          if (compare_selection(gb, wb, gap, sel_tol, esc, s.score ? s.score + size_t(row - s.row0) * s.W : nullptr,
                                (row + 1) / 4, sel_c, pname.c_str(), row, s.layer, "the prefill's selection"))
            near_row[row] = 1;
        }
      }
      std::printf("  prefill: %u ids in chunks of %u (%zu launches), routing and gate S on rows %u..%u\n", T, c,
                  eng.prefill_launches(), row0, T - 1);
    }
    for (uint32_t t = 0; t < T && !prefill; ++t) {
      if (inject) feed(t);
      eng.ingest({ids[t]});
      check_row(t);
    }
    uint32_t det = 0, det_ok = 0, ties = 0, tie_ok = 0;
    int first_bad = -1;
    double logit_min = 1.0;
    for (uint32_t j = 0; j < kGen; ++j) {
      const uint32_t row = T - 1 + j;
      const uint32_t got = eng.control(eng.devices() - 1).cur_token[0];
      const std::vector<float> lg = eng.read_logits();
      const float* want = glog + size_t(row - lrow0) * d.vocab;
      logit_min = std::min(logit_min, golden::compare_f32(lg.data(), want, d.vocab, sa, sb).cos);
      const golden::GoldenDecision dec = golden::golden_decision(want, d.vocab, d.vocab_used);
      bool upstream_near = false;
      for (uint32_t r = 0; r <= row; ++r) upstream_near = upstream_near || near_row[r];
      bool ok;
      if (dec.determined()) {
        ++det;
        ok = got == dec.set[0];
        det_ok += ok;
      } else {
        ++ties;
        ok = dec.contains(got);
        tie_ok += ok;
      }
      if (!ok) {
        if (!inject && upstream_near) {
          ++reported;   // a near tie upstream: reported, not failed (the free run's rule)
          std::printf("  step %u: engine %u, golden %u - after a near-tie row (reported)\n", j, got, uint32_t(gtok[j]));
        } else if (first_bad < 0) {
          first_bad = int(j);
        }
      }
      if (got != uint32_t(gtok[j])) {
        golden::print_top5("engine", lg.data(), d.vocab, d.vocab_used);
        golden::print_top5("golden", want, d.vocab, d.vocab_used);
      }
      if (inject) feed(T + j);
      eng.ingest({uint32_t(gtok[j])});   // teacher-forced: row T + j
      check_row(T + j);
    }
    const bool ok = first_bad < 0;
    gate_ok = gate_ok && ok;
    std::printf("  gate: %u / %u determined rows exact, %u / %u tie rows in the argmax set%s; logits worst cosine %.6f\n",
                det_ok, det, tie_ok, ties, ok ? "" : (" - FIRST BAD STEP " + std::to_string(first_bad)).c_str(), logit_min);
    std::printf("  tap (min cosine over the activation tail):");
    for (uint32_t l = 0; l < d.layers; ++l) std::printf(" %u:%.4f", l, tap_min[l]);
    std::printf("\n");
  }
  std::printf("\nrouting: %llu (row, layer) - %llu exact, %llu near-ties (gap rule), %llu rounding swaps (scores: weights "
              "cosine >= %.6f, deficit <= %.2f x the score noise), %llu exact ties (undetermined), %llu differing\n",
              (unsigned long long)route_c.rows, (unsigned long long)route_c.exact, (unsigned long long)route_c.near,
              (unsigned long long)route_c.swaps, route_c.cos_min, route_c.deficit_ratio_max,
              (unsigned long long)route_c.undetermined, (unsigned long long)route_c.bad);
  std::printf("gate S: %llu (row >= 2051, QSA layer) - %llu exact, %llu near-ties (gap rule), %llu rounding swaps (scores: "
              "cosine >= %.6f, deficit <= %.2f x the row noise), %llu exact ties (undetermined), %llu differing; PLE ids: "
              "%llu rows, %llu differing; %llu token rows reported after a near tie\n",
              (unsigned long long)sel_c.rows, (unsigned long long)sel_c.exact, (unsigned long long)sel_c.near,
              (unsigned long long)sel_c.swaps, sel_c.cos_min, sel_c.deficit_ratio_max,
              (unsigned long long)sel_c.undetermined, (unsigned long long)sel_c.bad, (unsigned long long)ple_rows,
              (unsigned long long)ple_bad, (unsigned long long)reported);
  CHECK(gate_ok);
  CHECK_EQ(route_c.bad, uint64_t(0));
  if (!inject) CHECK_EQ(sel_c.bad, uint64_t(0));
  CHECK_EQ(ple_bad, uint64_t(0));
  std::puts("qwen4exp_golden_test OK");
  return 0;
}
