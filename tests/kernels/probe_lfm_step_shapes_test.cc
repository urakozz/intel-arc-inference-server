// Spec 24c P0 (plan 24c Task 1): the skeleton lists probe_lfm_step replays (tools/probe/lfm_step_shapes.h)
// hold spec 24 §4.5's counts - 383 launches a token (2 x 8 + 28 x 9 + 10 x 11 + 5), 277 with every fused arm,
// the per-family counts, and the synthetic N 4's 41 / 31. Host only.
#include <cstdio>
#include <map>

#include "check.h"
#include "lfm_step_shapes.h"
#include "model/lfm2moe.h"

namespace {

using probe_lfm::Fam;
using probe_lfm::List;

std::map<Fam, size_t> count(const std::vector<probe_lfm::Launch>& v) {
  std::map<Fam, size_t> c;
  for (const probe_lfm::Launch& x : v) ++c[x.fam];
  return c;
}

}  // namespace

int main() {
  const model::Lfm2MoeDesc& d = model::lfm2moe();
  CHECK_EQ(d.layers, 40u);
  CHECK_EQ(d.attn_layers(), 10u);
  CHECK_EQ(d.conv_layers(), 30u);
  CHECK_EQ(d.moe_layers(), 38u);

  const std::vector<probe_lfm::Launch> a = probe_lfm::build_list(d, List::A);
  const std::vector<probe_lfm::Launch> f = probe_lfm::build_list(d, List::Fused);
  const std::vector<probe_lfm::Launch> g = probe_lfm::build_list(d, List::Gemv);
  const std::vector<probe_lfm::Launch> n = probe_lfm::build_list(d, List::Noop);
  // The formula, spelled out: 2 dense conv layers x 8, 28 conv + MoE x 9, 10 attention + MoE x 11, the head's 5.
  CHECK_EQ(a.size(), size_t(2 * 8 + 28 * 9 + 10 * 11 + 5));
  CHECK_EQ(a.size(), size_t(383));
  CHECK_EQ(a.size(), probe_lfm::derived_a(d));
  CHECK_EQ(f.size(), size_t(383 - 38 - 38 - 30));
  CHECK_EQ(f.size(), size_t(277));
  CHECK_EQ(f.size(), probe_lfm::derived_fused(d));
  CHECK_EQ(n.size(), size_t(383));
  for (const probe_lfm::Launch& x : n) CHECK(x.fam == Fam::Noop);

  const std::map<Fam, size_t> ca = count(a);
  auto at = [](const std::map<Fam, size_t>& c, Fam k) { auto it = c.find(k); return it == c.end() ? size_t(0) : it->second; };
  CHECK_EQ(at(ca, Fam::Embed), size_t(1));
  CHECK_EQ(at(ca, Fam::Norm), size_t(2 * 40 + 1));
  CHECK_EQ(at(ca, Fam::InProj), size_t(30));
  CHECK_EQ(at(ca, Fam::Conv), size_t(30));
  CHECK_EQ(at(ca, Fam::OutProj), size_t(30));
  CHECK_EQ(at(ca, Fam::Qkv), size_t(10));
  CHECK_EQ(at(ca, Fam::AttnPrep), size_t(10));
  CHECK_EQ(at(ca, Fam::Attn), size_t(10));
  CHECK_EQ(at(ca, Fam::AttnReduce), size_t(10));
  CHECK_EQ(at(ca, Fam::OProj), size_t(10));
  CHECK_EQ(at(ca, Fam::Router), size_t(38));
  CHECK_EQ(at(ca, Fam::Route), size_t(38));
  CHECK_EQ(at(ca, Fam::GateUp), size_t(38));
  CHECK_EQ(at(ca, Fam::Down), size_t(38));
  CHECK_EQ(at(ca, Fam::DenseGateUp), size_t(2));
  CHECK_EQ(at(ca, Fam::Silu), size_t(2));
  CHECK_EQ(at(ca, Fam::DenseDown), size_t(2));
  CHECK_EQ(at(ca, Fam::Head), size_t(1));
  CHECK_EQ(at(ca, Fam::Argmax1), size_t(1));
  CHECK_EQ(at(ca, Fam::Argmax2), size_t(1));
  // The order: the embedding first, the two argmax stages last, the layers in between.
  CHECK(a.front().fam == Fam::Embed && a.back().fam == Fam::Argmax2);

  const std::map<Fam, size_t> cf = count(f);
  CHECK_EQ(at(cf, Fam::Conv), size_t(0));
  CHECK_EQ(at(cf, Fam::Route), size_t(0));
  CHECK_EQ(at(cf, Fam::Router), size_t(0));
  CHECK_EQ(at(cf, Fam::InProj), size_t(30));
  CHECK_EQ(at(cf, Fam::GateUp), size_t(38));

  // gemv: the 4 GEMV / expert launches of every layer, the 38 routers, the head.
  CHECK_EQ(g.size(), size_t(40 * 4 + 38 + 1));
  for (const probe_lfm::Launch& x : g) CHECK(probe_lfm::is_gemv(x.fam));

  // The synthetic N 4 (make_synth.py --layers 4: dense conv 0, 1; attention + MoE 2; conv + MoE 3): 41, and
  // 34 fused (3 conv layers, 2 MoE layers).
  model::Lfm2MoeDesc d4 = d;
  d4.layers = 4;
  d4.kinds.assign(d.kinds.begin(), d.kinds.begin() + 4);
  CHECK_EQ(probe_lfm::build_list(d4, List::A).size(), size_t(8 + 8 + 11 + 9 + 5));
  CHECK_EQ(probe_lfm::build_list(d4, List::Fused).size(), size_t(41 - 3 - 2 * 2));

  // The int4 shapes the sweep times: every one a whole number of 64-k groups and 64-column work-groups.
  for (const probe_lfm::Int4Shape& s : probe_lfm::int4_shapes(d)) CHECK(s.K % 64 == 0 && s.N % 64 == 0);

  std::printf("probe_lfm_step_shapes_test: a %zu, fused %zu, gemv %zu, noop %zu - ok\n", a.size(), f.size(), g.size(),
              n.size());
  return 0;
}
