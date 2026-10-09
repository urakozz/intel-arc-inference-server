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
//     ple.ids        I64  [T + gen][16]    the 16 PLE rows each position read
//     H.L{l}         BF16 [A][10240]       the 4-stream H after layer l, rows T + gen - A .. (the tap diagnostic)
// The prompts read: q4exp_short (must exist), q4exp_4k, q4exp_agentic - each whose files exist.
//
// **The gates** (golden_common.h's tie-aware rule; spec 21 §7 F3):
//   tokens   ingest the ids one replay at a time; for j = 0..31 the engine's id against the golden decision row
//            T - 1 + j (determined rows exact; a row whose top-1 is not unique accepts any member of the argmax
//            set), then feed the GOLDEN token
//   routing  at every (row, layer) the engine's top-10 SET against route.ids: a difference is a NEAR-TIE when the
//            reference's gap is within B70_Q4_TIE_TOL (1e-3 PROPOSED; 0 = an exact tie, which torch's topk breaks
//            its own way - undetermined, spec 21 §12 finding 2) and the sets differ by one expert; anything else FAILS
//   gate S   at every (row >= 2051, QSA layer) the engine's selected block set against qsa.sel: a difference FAILS
//            unless qsa.gap is within B70_Q4_SEL_TOL (1e-3 PROPOSED; 0 = undetermined, ties of zeros included);
//            near-tie rows are counted and printed
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
  uint64_t rows = 0, exact = 0, near = 0, undetermined = 0, bad = 0;
};

// The engine's route row (rank order) against the reference's top-10 (its topk order), as sets.
void compare_route(const uint32_t* row, const int32_t* want, float gap, float tol, Counts& c, const char* p, uint32_t r,
                   uint32_t l) {
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
  if (gap <= tol && og.size() == 1 && ow.size() == 1) {
    ++c.near;
    return;
  }
  ++c.bad;
  if (c.bad <= 10)
    std::printf("    %s L%u row %u: the expert set differs beyond a near-tie (gap %.3e, %zu experts)\n", p, l, r, double(gap),
                og.size());
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
      uint32_t layer = 0, row0 = 0, R = 0;
      const int32_t* blocks = nullptr;
      const float* gap = nullptr;
    };
    std::vector<Sel> sels;
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
      sels.push_back(s);
    }
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
        const uint64_t bad0 = route_c.bad, near0 = route_c.near + route_c.undetermined;
        compare_route(r.data() + size_t(l) * rq::kRouteWords, gi, gap, tie_tol, route_c, pname.c_str(), row, l);
        if (route_c.near + route_c.undetermined != near0) near_row[row] = 1;
        (void)bad0;
      }
      for (const Sel& s : sels) {   // gate S
        if (row < kDenseRows || !s.blocks || row < s.row0) continue;
        ++sel_c.rows;
        const std::vector<uint32_t> got = eng.read_selection(s.layer);
        std::vector<uint32_t> gb, wb(s.blocks + size_t(row - s.row0) * 512, s.blocks + size_t(row - s.row0) * 512 + 512);
        for (uint32_t k = 0; k < 512 && 4 * k < got.size(); ++k) gb.push_back(got[4 * k] / 4);
        std::sort(wb.begin(), wb.end());
        const float gap = s.gap ? s.gap[row] : 1.0f;
        if (gb == wb) {
          ++sel_c.exact;
        } else if (gap == 0.0f) {
          ++sel_c.undetermined;
          near_row[row] = 1;
        } else if (gap <= sel_tol) {
          ++sel_c.near;
          near_row[row] = 1;
        } else {
          ++sel_c.bad;
          if (sel_c.bad <= 10)
            std::printf("    %s QSA L%u row %u: the selection differs beyond a near-tie (gap %.3e)\n", pname.c_str(),
                        s.layer, row, double(gap));
        }
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
          const uint64_t near0 = route_c.near + route_c.undetermined;
          compare_route(r.data() + rq::pf_route_at(l) / 4 + size_t(row - row0) * rq::kRouteWords, gi, gap, tie_tol, route_c,
                        pname.c_str(), row, l);
          if (route_c.near + route_c.undetermined != near0) near_row[row] = 1;
        }
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
          if (gb == wb) {
            ++sel_c.exact;
          } else if (gap == 0.0f) {
            ++sel_c.undetermined;
            near_row[row] = 1;
          } else if (gap <= sel_tol) {
            ++sel_c.near;
            near_row[row] = 1;
          } else if (++sel_c.bad <= 10) {
            std::printf("    %s QSA L%u row %u: the prefill's selection differs beyond a near-tie (gap %.3e)\n",
                        pname.c_str(), s.layer, row, double(gap));
          }
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
  std::printf("\nrouting: %llu (row, layer) - %llu exact, %llu near-ties, %llu exact ties (undetermined), %llu differing\n",
              (unsigned long long)route_c.rows, (unsigned long long)route_c.exact, (unsigned long long)route_c.near,
              (unsigned long long)route_c.undetermined, (unsigned long long)route_c.bad);
  std::printf("gate S: %llu (row >= 2051, QSA layer) - %llu exact, %llu near-ties, %llu exact ties (undetermined), %llu "
              "differing; PLE ids: %llu rows, %llu differing; %llu token rows reported after a near tie\n",
              (unsigned long long)sel_c.rows, (unsigned long long)sel_c.exact, (unsigned long long)sel_c.near,
              (unsigned long long)sel_c.undetermined, (unsigned long long)sel_c.bad, (unsigned long long)ple_rows,
              (unsigned long long)ple_bad, (unsigned long long)reported);
  CHECK(gate_ok);
  CHECK_EQ(route_c.bad, uint64_t(0));
  if (!inject) CHECK_EQ(sel_c.bad, uint64_t(0));
  CHECK_EQ(ple_bad, uint64_t(0));
  std::puts("qwen4exp_golden_test OK");
  return 0;
}
