// Spec 25c P0 (plan 25c Task 1): the skeleton lists probe_bolt_step replays (tools/probe/bolt_step_shapes.h) hold
// spec 25 §6.1's counts - 653 launches a token (40 x 6 + 8 x 6 + 40 x 9 + 5), 533 with every fused arm, the
// per-family counts, the GEMV-only and state lists, and the synthetic N 12's 92 / 76. Host only.
#include <cstdio>
#include <map>

#include "bolt_step_shapes.h"
#include "check.h"
#include "model/bolt.h"

namespace {

using probe_bolt::Fam;
using probe_bolt::List;

std::map<Fam, size_t> count(const std::vector<probe_bolt::Launch>& v) {
  std::map<Fam, size_t> c;
  for (const probe_bolt::Launch& x : v) ++c[x.fam];
  return c;
}
size_t at(const std::map<Fam, size_t>& c, Fam k) {
  auto it = c.find(k);
  return it == c.end() ? size_t(0) : it->second;
}

}  // namespace

int main() {
  const model::BoltDesc& d = model::bolt();
  CHECK_EQ(d.layers, 88u);
  CHECK_EQ(d.mamba_layers(), 40u);
  CHECK_EQ(d.attn_layers(), 8u);
  CHECK_EQ(d.moe_layers(), 40u);

  const std::vector<probe_bolt::Launch> a = probe_bolt::build_list(d, List::A);
  const std::vector<probe_bolt::Launch> f = probe_bolt::build_list(d, List::Fused);
  const std::vector<probe_bolt::Launch> g = probe_bolt::build_list(d, List::Gemv);
  const std::vector<probe_bolt::Launch> n = probe_bolt::build_list(d, List::Noop);
  const std::vector<probe_bolt::Launch> s = probe_bolt::build_list(d, List::State);
  // The formula, spelled out: 40 Mamba x 6, 8 attention x 6, 40 MoE x 9, the head's 5.
  CHECK_EQ(a.size(), size_t(40 * 6 + 8 * 6 + 40 * 9 + 5));
  CHECK_EQ(a.size(), size_t(653));
  CHECK_EQ(a.size(), probe_bolt::derived_a(d));
  CHECK_EQ(f.size(), size_t(653 - 40 - 40 - 40));
  CHECK_EQ(f.size(), size_t(533));
  CHECK_EQ(f.size(), probe_bolt::derived_fused(d));
  CHECK_EQ(n.size(), size_t(653));
  for (const probe_bolt::Launch& x : n) CHECK(x.fam == Fam::Noop);
  CHECK_EQ(s.size(), size_t(40));
  for (const probe_bolt::Launch& x : s) CHECK(x.fam == Fam::State && d.is_mamba(uint32_t(x.layer)));

  const std::map<Fam, size_t> ca = count(a);
  CHECK_EQ(at(ca, Fam::Embed), size_t(1));
  CHECK_EQ(at(ca, Fam::Norm), size_t(88 + 1));
  for (Fam k : {Fam::InProj, Fam::Conv, Fam::Ssm, Fam::Gnorm, Fam::OutProj}) CHECK_EQ(at(ca, k), size_t(40));
  for (Fam k : {Fam::Qkv, Fam::AttnPrep, Fam::Attn, Fam::AttnReduce, Fam::OProj}) CHECK_EQ(at(ca, k), size_t(8));
  for (Fam k : {Fam::Router, Fam::Route, Fam::Fc1Up, Fam::ShAct, Fam::MoeUp, Fam::MoeDown, Fam::Fc2, Fam::ShDown})
    CHECK_EQ(at(ca, k), size_t(40));
  CHECK_EQ(at(ca, Fam::Head), size_t(1));
  CHECK_EQ(at(ca, Fam::Argmax1), size_t(1));
  CHECK_EQ(at(ca, Fam::Argmax2), size_t(1));
  // The order: the embedding first, the two argmax stages last, the blocks in between.
  CHECK(a.front().fam == Fam::Embed && a.back().fam == Fam::Argmax2);

  const std::map<Fam, size_t> cf = count(f);
  CHECK_EQ(at(cf, Fam::Conv), size_t(0));
  CHECK_EQ(at(cf, Fam::Route), size_t(0));
  CHECK_EQ(at(cf, Fam::Router), size_t(0));
  CHECK_EQ(at(cf, Fam::Ssm), size_t(40));
  CHECK_EQ(at(cf, Fam::MoeUp), size_t(40));
  CHECK_EQ(at(cf, Fam::Norm), size_t(89));

  // gemv: Mamba's 2, attention's 2, MoE's 6 (router, fc1||shared_up, up, down, fc2, shared_down), the head.
  CHECK_EQ(g.size(), size_t(40 * 2 + 8 * 2 + 40 * 6 + 1));
  for (const probe_bolt::Launch& x : g) CHECK(probe_bolt::is_gemv(x.fam));

  // The synthetic N 12 (make_synth.py --layers 12: M E M E M E M * E M E M - 6 Mamba, 5 MoE, 1 attention): 92, and 76
  // fused (6 conv slots, 5 router and 5 route slots).
  const model::BoltDesc d12 = model::bolt_truncated(d, 12);
  CHECK_EQ(d12.mamba_layers(), 6u);
  CHECK_EQ(d12.moe_layers(), 5u);
  CHECK_EQ(d12.attn_layers(), 1u);
  CHECK_EQ(probe_bolt::build_list(d12, List::A).size(), size_t(5 + 6 * 6 + 1 * 6 + 5 * 9));
  CHECK_EQ(probe_bolt::build_list(d12, List::A).size(), size_t(92));
  CHECK_EQ(probe_bolt::build_list(d12, List::Fused).size(), size_t(92 - 6 - 2 * 5));

  // The int4 shapes the sweep times: whole 64-k groups and 64-column work-groups; each slot's descriptor cell exists.
  for (const probe_bolt::Int4Shape& sh : probe_bolt::int4_shapes(d)) {
    CHECK(sh.K % 64 == 0 && sh.N % 64 == 0);
    bool found = false;
    for (uint32_t l = 0; l < d.layers && !found; ++l)
      for (const model::BoltLinear& ln : d.linears(l))
        if (ln.id == probe_bolt::linear_of(sh.fam)) {
          CHECK_EQ(ln.shape.K, sh.K);
          CHECK_EQ(ln.shape.N, sh.N);
          found = true;
        }
    CHECK(found);
  }

  std::printf("probe_bolt_step_shapes_test: a %zu, fused %zu, gemv %zu, noop %zu, state %zu; N 12: %zu / %zu - ok\n",
              a.size(), f.size(), g.size(), n.size(), s.size(), probe_bolt::build_list(d12, List::A).size(),
              probe_bolt::build_list(d12, List::Fused).size());
  return 0;
}
