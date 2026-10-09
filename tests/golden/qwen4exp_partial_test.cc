// qwen4exp_partial_test - spec 21c Task 3: the development mode on the REAL checkpoint, one card (Intel's
// checkpoint at --layers N - 18 by default: what fits one B70 with its bf16 dense layers). kolibri_partial_test's
// arrangement:
//
//   * the tap: the engine's H after each layer l < N against the FULL reference's H.L<l> (qwen4exp_oracle.sh intel:
//     the reference's own forward through all 48 layers, whose first N layers are exactly what the truncated engine
//     computes) - cosine per row over the activation tail: median >= 0.9998 and min >= 0.99 (PROPOSED), printed per
//     layer
//   * the routing diagnostic and gate S for the layers < N against the full reference (B70_Q4_TIE_TOL /
//     B70_Q4_SEL_TOL, as qwen4exp_golden_test)
//   * the token gate (tie-aware) against the TRUNCATED reference (qwen4exp_oracle.sh intel-layers N: transformers
//     cut to N layers then the final mixer and lm_head, as the engine runs)
//
// argv: <checkpoint> <full oracle dir> <truncated oracle dir> [N] [int8]
// Exit 77 (SKIP) when the checkpoint, its PLE file or either golden set is absent, or N layers do not fit one card.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "kernels/qwen4exp_kernels.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp_rig.h"

namespace {
namespace rq = runtime::qwen4exp;
constexpr uint32_t kGen = 32;
const char* const kPrompts[] = {"q4exp_short", "q4exp_4k", "q4exp_agentic"};
float env_tol(const char* name, float dflt) {
  const char* e = std::getenv(name);
  return e && *e ? std::strtof(e, nullptr) : dflt;
}
double median(std::vector<double> v) {
  if (v.empty()) return 1.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <checkpoint> <full oracle dir> <truncated oracle dir> [N] [int8]\n", argv[0]);
    return 2;
  }
  qwen4exp_rig::Options o;
  o.max_len = 16384;
  o.debug_tap = true;
  o.layers = 18;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "int8") o.int8_head = true;
    else o.layers = uint32_t(std::strtoul(a.c_str(), nullptr, 10));
  }
  const std::string full_dir = argv[2], cut_dir = argv[3];
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  std::vector<std::string> prompts;
  for (const char* p : kPrompts)
    if (golden::exists(full_dir + "/" + p + ".golden.safetensors") && golden::exists(cut_dir + "/" + p + ".golden.safetensors") &&
        golden::exists(cut_dir + "/" + p + ".ids"))
      prompts.push_back(p);
  if (prompts.empty()) {
    std::printf("SKIP: no prompt has both %s/<p>.golden.safetensors and %s/<p>.{ids,golden.safetensors} (21a's real "
                "sets: qwen4exp_oracle.sh intel and intel-layers %u)\n", full_dir.c_str(), cut_dir.c_str(), o.layers);
    return 77;
  }
  const model::Qwen4ExpDesc pre = loader::qwen4exp_checkpoint_desc(snap, o.layers);
  const std::array<size_t, runtime::kPpDevices> caps = {size_t(32530000000ull), size_t(32530000000ull)};
  if (!rq::fits(rq::plan(pre, model::Q4Placement::one(pre), o.max_len, o.int8_head, false, true), caps, size_t(1.5e9))) {
    std::printf("SKIP: %u layers do not fit one card at max_len %u\n", o.layers, o.max_len);
    return 77;
  }
  const float tie_tol = env_tol("B70_Q4_TIE_TOL", 1e-3f), sel_tol = env_tol("B70_Q4_SEL_TOL", 1e-3f);
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::build(rig, snap, o);
  rq::Qwen4ExpEngine& eng = *rig.eng;
  const model::Qwen4ExpDesc& d = eng.model().desc;
  std::printf("%s: layers [0, %u) of %u, %s dense, lm_head %s, %zu launches a token\n", snap.c_str(), d.layers,
              eng.model().checkpoint_layers, model::q4_form_name(d.forms.dense), o.int8_head ? "int8" : "bf16",
              eng.launches());
  bool ok_all = true;
  std::vector<double> sa, sb;
  uint64_t route_bad = 0, sel_bad = 0, route_near = 0, sel_near = 0;
  for (const std::string& pname : prompts) {
    const std::vector<uint32_t> ids = qwen4exp_rig::read_ids(cut_dir + "/" + pname + ".ids", d.vocab);
    golden::Golden full(full_dir + "/" + pname + ".golden.safetensors"), cut(cut_dir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size()), rows = T + kGen;
    if (rows > eng.max_len()) continue;
    // the full reference's rows: its prompt rows are this run's (the same ids, the first N layers identical)
    const uint32_t frows = T + kGen;
    const uint32_t act0 = full.has("H.L0") ? frows - uint32_t(full.dim("H.L0", 2, 0)) : frows;
    std::vector<std::vector<double>> cos(d.layers);
    const auto check_row = [&](uint32_t row) {
      const std::vector<uint32_t> r = eng.read_routes();
      for (uint32_t l = 0; l < d.layers; ++l) {
        const std::string ls = std::to_string(l);
        if (!full.has("route.ids.L" + ls)) continue;
        const int32_t* gi = full.i32("route.ids.L" + ls, size_t(rows) * 10) + size_t(row) * 10;
        const float gap = full.f32("route.gap.L" + ls, rows)[row];
        std::vector<uint32_t> g(r.begin() + size_t(l) * rq::kRouteWords, r.begin() + size_t(l) * rq::kRouteWords + 10),
            w(gi, gi + 10);
        std::sort(g.begin(), g.end());
        std::sort(w.begin(), w.end());
        if (g != w) (gap <= tie_tol ? route_near : route_bad)++;
        if (d.is_qsa(l) && row >= 2051 && full.has("qsa.sel.L" + ls)) {
          const uint32_t R = uint32_t(full.dim("qsa.sel.L" + ls, 2, 0)), row0 = rows - R;
          if (row < row0) continue;
          const int32_t* b = full.i32("qsa.sel.L" + ls, size_t(R) * 512) + size_t(row - row0) * 512;
          const std::vector<uint32_t> got = eng.read_selection(l);
          std::vector<uint32_t> gb, wb(b, b + 512);
          for (uint32_t k = 0; k < 512 && 4 * k < got.size(); ++k) gb.push_back(got[4 * k] / 4);
          std::sort(wb.begin(), wb.end());
          const float sgap = full.f32("qsa.gap.L" + ls, rows)[row];
          if (gb != wb) (sgap <= sel_tol ? sel_near : sel_bad)++;
        }
      }
      if (row >= act0) {
        const std::vector<uint16_t> tap = eng.read_debug_H();
        for (uint32_t l = 0; l < d.layers; ++l) {
          if (!full.has("H.L" + std::to_string(l))) continue;
          const uint16_t* want = full.bf16("H.L" + std::to_string(l), size_t(rows - act0) * d.hc_n()) +
                                 size_t(row - act0) * d.hc_n();
          cos[l].push_back(golden::compare_bf16(tap.data() + size_t(l) * d.hc_n(), want, d.hc_n(), sa, sb).cos);
        }
      }
    };
    eng.reset();
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      check_row(t);
    }
    const uint32_t Lr = uint32_t(cut.dim("logits", 2, 0)), lrow0 = rows - Lr;
    const float* glog = cut.f32("logits", size_t(Lr) * d.vocab);
    const int32_t* gtok = cut.i32("tokens", kGen);
    uint32_t det = 0, det_ok = 0, ties = 0, tie_ok = 0;
    for (uint32_t j = 0; j < kGen; ++j) {
      const uint32_t row = T - 1 + j, got = eng.control(0).cur_token[0];
      const golden::GoldenDecision dec = golden::golden_decision(glog + size_t(row - lrow0) * d.vocab, d.vocab, d.vocab_used);
      if (dec.determined()) {
        ++det;
        det_ok += got == dec.set[0];
      } else {
        ++ties;
        tie_ok += dec.contains(got);
      }
      // teacher-forced with the TRUNCATED reference's tokens: the full reference generated its own, so its rows past
      // the prompt are not this run's - the tap / routing / gate S comparisons stop at the prompt
      eng.ingest({uint32_t(gtok[j])});
    }
    std::printf("\n%s (%u ids): tokens %u / %u determined exact, %u / %u ties in the set (the truncated reference)\n",
                pname.c_str(), T, det_ok, det, tie_ok, ties);
    std::printf("  tap cosine per layer (median / min over %zu rows):", cos.empty() ? size_t(0) : cos[0].size());
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (cos[l].empty()) continue;
      const double med = median(cos[l]), mn = *std::min_element(cos[l].begin(), cos[l].end());
      std::printf(" L%u %.5f / %.5f", l, med, mn);
      ok_all = ok_all && med >= 0.9998 && mn >= 0.99;   // PROPOSED bars
    }
    std::printf("\n");
    ok_all = ok_all && det_ok == det && tie_ok == ties;
  }
  std::printf("\nrouting: %llu near-ties, %llu beyond; gate S: %llu near-ties, %llu beyond (layers < %u, the full "
              "reference)\n", (unsigned long long)route_near, (unsigned long long)route_bad, (unsigned long long)sel_near,
              (unsigned long long)sel_bad, d.layers);
  CHECK(ok_all);
  CHECK_EQ(route_bad, uint64_t(0));
  CHECK_EQ(sel_bad, uint64_t(0));
  std::puts("qwen4exp_partial_test OK");
  return 0;
}
