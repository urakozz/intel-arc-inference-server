// Spec 26a (plan 26a Task 4): every binary `probe_draft_cost --shapes bolt-lilicorr` binds - Bolt's verify at M = 1..8
// (model::BoltDesc's int4 GEMVs at their {S, layout}, nb_router, the int8 head, argmax stage 1), the routed-expert
// stand-in, the state's nb_state_rmw and one LiLiCorr block in three formats with the bf16 LiLiCorr head
// (tools/probe/draft_cost_shapes.h's bolt_lilicorr namespace) - formed here as the probe forms them, against the binaries
// CMake builds (passed as arguments, `kernel_<name>`). Host only, no device: a name the probe would ask for that no CMake
// line builds fails on the Mac instead of on the box.
//
// usage: lilicorr_cost_names_test <the B70_LILICORR_P0 block's kernel_*...> -- <reused kernel_*>
// The first list must be exactly bound (nothing in the block is dead); the union must cover every bound name.
#include <cstdio>
#include <set>
#include <string>

#include "check.h"
#include "draft_cost_shapes.h"
#include "kernels/bolt_kernels.h"
#include "kernels/kernels.h"
#include "model/bolt.h"

namespace {

namespace dl = draft_cost::bolt_lilicorr;

std::string bf16(unsigned M, unsigned K, unsigned N) {
  return kernels::gemv_bf16_variant(M, K, N, kernels::gemv_bf16_tiling(N));
}

void verify_names(const model::BoltDesc& d, unsigned M, std::set<std::string>& out) {
  for (uint32_t l = 0; l < d.layers; ++l)
    for (const model::BoltLinear& L : d.linears(l)) {
      if (L.id == model::BoltLinearId::ExpertUp || L.id == model::BoltLinearId::ExpertDown) continue;
      if (L.id == model::BoltLinearId::Router) {
        out.insert(kernels::bolt::router_variant(M));
        continue;
      }
      out.insert(kernels::gemv_variant(M, L.shape.K, L.shape.N, L.shape.S, L.shape.layout));
    }
  out.insert(kernels::gemv_i8w_variant(M, dl::kHidden, dl::kVocab));
  out.insert(kernels::bolt::argmax1_variant(M));
}

void block_names(std::set<std::string>& out) {
  for (const draft_cost::Linear& L : dl::kDrafter) {
    out.insert(kernels::gemv_variant(dl::kDraftRows, L.K, L.N, 1, 1));
    out.insert(kernels::gemv_i8w_variant(dl::kDraftRows, L.K, L.N));
    out.insert(bf16(dl::kDraftRows, L.K, L.N));
  }
  out.insert(kernels::gemv_i8w_variant(dl::kDraftHeadRows, dl::kHidden, dl::kVocab));
  out.insert(kernels::bolt::argmax1_variant(dl::kDraftHeadRows));
  for (const dl::HeadGemv& g : dl::kHeadGemvs) out.insert(bf16(g.M, g.K, g.N));
}

}  // namespace

int main(int argc, char** argv) {
  std::set<std::string> block, all;
  bool other = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") {
      other = true;
      continue;
    }
    CHECK(a.rfind("kernel_", 0) == 0);
    (other ? all : block).insert(a.substr(7));
  }
  CHECK(!block.empty());
  all.insert(block.begin(), block.end());

  const model::BoltDesc& d = model::bolt();
  std::set<std::string> bound;
  for (unsigned M = 1; M <= dl::kMaxM; ++M) verify_names(d, M, bound);
  bound.insert(kernels::gemv_variant(1, dl::kLatent, dl::kStandInI, 1, 1));
  bound.insert(kernels::gemv_variant(1, dl::kStandInI, dl::kLatent, 1, 1));
  bound.insert("nb_state_rmw_F32");
  block_names(bound);

  int bad = 0;
  for (const std::string& n : bound)
    if (!all.count(n)) {
      std::fprintf(stderr, "probe_draft_cost --shapes bolt-lilicorr binds %s, which no CMake line given here builds\n",
                   n.c_str());
      ++bad;
    }
  for (const std::string& n : block)
    if (!bound.count(n)) {
      std::fprintf(stderr, "%s is built by the B70_LILICORR_P0 block but the probe never binds it\n", n.c_str());
      ++bad;
    }
  CHECK_EQ(bad, 0);
  // The shapes the names carry, spelled out (a helper change shows here, not as a missing file on the box).
  CHECK_EQ(kernels::gemv_variant(8, 4096, 18560, 1, 0), std::string("gemv_M8_K4096_N18560_S1_L0"));
  CHECK_EQ(kernels::bolt::router_variant(5), std::string("nb_router_M5_K4096"));
  CHECK_EQ(kernels::bolt::argmax1_variant(7), std::string("nb_argmax_stage1_M7_V131072"));
  CHECK_EQ(bf16(8, 4096, 512), std::string("gemv_bf16_M8_K4096_N512_C16_S16"));
  CHECK_EQ(bf16(8, 1024, 3072), std::string("gemv_bf16_M8_K1024_N3072"));
  // The expert stand-in's block is within 0.3 % of the 40 MoE blocks' average (2 x I x 544 B at latent 1024).
  double sum = 0;
  for (uint32_t l = 0; l < d.layers; ++l)
    if (d.is_moe(l)) sum += 2.0 * d.inter.at(l) * 544;
  const double avg = sum / d.moe_layers(), stand = 2.0 * dl::kStandInI * 544;
  CHECK(stand / avg > 0.997 && stand / avg < 1.003);
  // The drafter's parameters at these shapes: 5 layers + fc = spec 26 §1's 1.076 B (derived from the header).
  // (ctx k||v IS k_proj / v_proj - counted in q||k||v: the commit reads them again, they are not more parameters.)
  double params = 0;
  for (const draft_cost::Linear& L : dl::kDrafter)
    if (std::string(L.name) != "ctx k||v") params += double(L.K) * L.N * (L.per_layer ? dl::kDraftLayers : 1);
  std::printf("lilicorr_cost_names_test OK: %zu names bound (verify M = 1..8 + the stand-in + one block), %zu in the "
              "block, all built; a stand-in expert block %.3f MB vs the average %.3f MB; the drafter's linears %.4f B "
              "parameters\n",
              bound.size(), block.size(), stand / 1e6, avg / 1e6, params / 1e9);
  CHECK(params > 1.07e9 && params < 1.08e9);
  return 0;
}
