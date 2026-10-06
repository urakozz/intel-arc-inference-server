// kolibri_partial_test - spec 20 §4's development mode on the REAL checkpoint, one card: load only its
// first N layers (loader::load_kolibri1 layers_limit; default 30 - fits one card at int4), ingest each
// golden prompt one replay at a time, and compare the residual tap of layers 0..N-1 with the reference's
// resid.L* (kolibri_ref.py run on the WHOLE model: its layer l's output does not depend on layers > l),
// plus the routing diagnostic for layers < N. The head is not graded (it reads layer N-1's residual).
//
// Bars per layer, over every prompt row: cosine median >= 0.9998 and min >= 0.99 (PROPOSED - set from the
// first box run, printed per layer); routing: kolibri_golden_test's rule (a set difference beyond a
// near-tie at B70_KOL_TIE_TOL fails).
//
// argv: <real checkpoint> <oracle-out-kolibri> [N]. Exit 77 (SKIP) without the checkpoint (spec 20b) or
// the golden set (kolibri_oracle.sh real).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "kernels/kolibri_kernels.h"
#include "runtime/kolibri_rig.h"
#include "runtime/kolibri/kolibri_sizes.h"

namespace {
const char* const kPrompts[] = {"prose", "code", "de_prose", "de_chat"};
float tie_tol() {
  const char* e = std::getenv("B70_KOL_TIE_TOL");
  return e && *e ? std::strtof(e, nullptr) : 1e-2f;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-kolibri";
  const uint32_t N = argc > 3 ? uint32_t(std::strtoul(argv[3], nullptr, 10)) : 30;
  const std::string snap = kolibri_rig::kolibri_snapshot(arg);
  if (snap.empty()) {
    std::printf("SKIP: no Kolibri-1 checkpoint at '%s' (spec 20b's)\n", arg.c_str());
    return 77;
  }
  std::vector<std::string> prompts;
  for (const char* p : kPrompts)
    if (golden::exists(gdir + "/" + p + ".golden.safetensors") && golden::exists(gdir + "/" + p + ".ids"))
      prompts.push_back(p);
  if (prompts.empty()) {
    std::printf("SKIP: no golden set in %s (tools/box_validate/kolibri_oracle.sh real)\n", gdir.c_str());
    return 77;
  }
  kolibri_rig::Rig rig;
  kolibri_rig::Options o;
  o.max_len = 8192;
  o.layers = N;
  o.debug_tap = true;
  kolibri_rig::build(rig, snap, o);
  runtime::kolibri::KolibriEngine& eng = *rig.eng;
  const model::Kolibri1Desc& d = eng.model().desc;
  CHECK_EQ(d.layers, N);
  std::printf("%s: layers [0, %u) of %u, %s attention, %s attention arm\n", snap.c_str(), N,
              eng.model().checkpoint_layers, runtime::kolibri::kol_attn_name(eng.attention()),
              model::kol_attn_form_name(d.attn));
  std::vector<std::vector<double>> cos(N);
  uint64_t rows = 0, ties = 0, bad = 0;
  std::vector<double> sa, sb;
  for (const std::string& pname : prompts) {
    const std::vector<uint32_t> ids = kolibri_rig::read_ids(gdir + "/" + pname + ".ids", d.vocab);
    golden::Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size());
    const uint32_t grows = uint32_t(g.dim("logits", 2, 0));
    eng.reset();
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      const std::vector<uint16_t> tap = eng.read_debug_resid();
      const std::vector<uint32_t> r = eng.read_routes();
      for (uint32_t l = 0; l < N; ++l) {
        const std::string ls = std::to_string(l);
        const uint16_t* want = g.bf16("resid.L" + ls, size_t(T) * d.hidden) + size_t(t) * d.hidden;
        cos[l].push_back(golden::compare_bf16(tap.data() + size_t(l) * d.hidden, want, d.hidden, sa, sb).cos);
        const int32_t* gi = g.i32("route.moe.ids.L" + ls, size_t(grows) * d.top_k) + size_t(t) * d.top_k;
        const float gap = g.f32("route.moe.gap.L" + ls, grows)[t];
        const uint32_t* er = r.data() + size_t(l) * runtime::kolibri::kRouteWords + kernels::kolibri::route::kIds;
        ++rows;
        std::vector<uint32_t> ge(er, er + d.top_k), wa(gi, gi + d.top_k);
        if (ge == wa) continue;
        uint32_t common = 0;
        for (uint32_t x : ge) common += std::count(wa.begin(), wa.end(), x) ? 1u : 0u;
        if (gap <= tie_tol() && common == d.top_k - 1)
          ++ties;
        else if (++bad <= 10)
          std::printf("  %s L%u row %u: expert set differs beyond a near-tie (gap %.3e)\n", pname.c_str(), l, t,
                      double(gap));
      }
    }
  }
  bool ok = true;
  std::printf("layer  rows  median cos   min cos\n");
  for (uint32_t l = 0; l < N; ++l) {
    std::vector<double> c = cos[l];
    std::sort(c.begin(), c.end());
    const double med = c[c.size() / 2], mn = c.front();
    const bool lok = med >= 0.9998 && mn >= 0.99;
    ok = ok && lok;
    std::printf("  %3u  %4zu  %.6f    %.6f%s\n", l, c.size(), med, mn, lok ? "" : "  BELOW (proposed bars)");
  }
  std::printf("routing: %llu rows, %llu near-ties, %llu differing\n", (unsigned long long)rows,
              (unsigned long long)ties, (unsigned long long)bad);
  CHECK(ok);
  CHECK_EQ(bad, uint64_t(0));
  std::puts("kolibri_partial_test OK");
  return 0;
}
