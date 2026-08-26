#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include "check.h"
#include "loader/small_layout.h"
#include "model/qwen35.h"

int main() {
  using model::LayerKind;
  using model::LinearId;
  using model::Qwen35;
  auto layers = Qwen35::layers();
  CHECK_EQ(layers.size(), size_t(64));
  size_t gdn = 0, fa = 0;
  for (const auto& l : layers) (l.kind == LayerKind::GDN ? gdn : fa)++;
  CHECK_EQ(gdn, size_t(48));
  CHECK_EQ(fa, size_t(16));
  CHECK(layers[3].kind == LayerKind::FA);
  CHECK(layers[62].kind == LayerKind::GDN);

  // Every GEMV shape is kernel-legal: N%64==0, K%64==0, S divides K/64.
  for (LinearId id : {LinearId::QkvZ, LinearId::AB, LinearId::OutProj, LinearId::GateUp,
                      LinearId::Down, LinearId::Qkv, LinearId::OProj, LinearId::LmHead}) {
    const auto& s = Qwen35::shape(id);
    CHECK_EQ(s.N % 64, uint32_t(0));
    CHECK_EQ(s.K % 64, uint32_t(0));
    CHECK_EQ((s.K / 64) % s.S, uint32_t(0));
  }
  // Task 4's production map. Every cell is literal so a wrong layout or split
  // cannot hide behind a helper shared with the table under test.
  struct ShapeWant { LinearId id; uint32_t K, N, S, layout; };
  for (const ShapeWant& w : {
           ShapeWant{LinearId::OutProj, 6144, 5120, 4, 0},
           ShapeWant{LinearId::OProj, 6144, 5120, 4, 0},
           ShapeWant{LinearId::Qkv, 5120, 14336, 2, 0},
           ShapeWant{LinearId::QkvZ, 5120, 16384, 1, 1},
           ShapeWant{LinearId::GateUp, 5120, 34816, 8, 0},
           ShapeWant{LinearId::Down, 17408, 5120, 4, 0},
       }) {
    const auto& s = Qwen35::shape(w.id);
    CHECK_EQ(s.K, w.K);
    CHECK_EQ(s.N, w.N);
    CHECK_EQ(s.S, w.S);
    CHECK_EQ(s.layout, w.layout);
  }
  const auto& int4_head = Qwen35::lm_head(model::WeightKind::Int4).shape;
  CHECK_EQ(int4_head.K, uint32_t(5120));
  CHECK_EQ(int4_head.N, uint32_t(248320));
  CHECK_EQ(int4_head.S, uint32_t(1));
  CHECK_EQ(int4_head.layout, uint32_t(1));

  // Fusion wiring for layer 0 (GDN) and 3 (FA).
  CHECK_EQ(layers[0].linears.size(), size_t(5));
  CHECK(layers[0].linears[0].id == LinearId::QkvZ);
  CHECK_EQ(layers[0].linears[0].parts.size(), size_t(2));
  CHECK_EQ(layers[0].linears[1].pad_n, uint32_t(96));   // a||b padded 96 -> 128
  CHECK_EQ(layers[3].linears.size(), size_t(4));
  CHECK(layers[3].linears[0].id == LinearId::Qkv);
  CHECK_EQ(Qwen35::layer_prefix(5), std::string("layers.5."));

  // T5: the table is exactly kCount rows and shape() refuses anything else.
  CHECK_EQ(size_t(LinearId::kCount), size_t(8));
  CHECK(Qwen35::linear(LinearId::LmHead).kind == model::WeightKind::Bf16);
  bool threw = false;
  try { Qwen35::shape(LinearId::kCount); } catch (const std::out_of_range&) { threw = true; }
  CHECK(threw);

  // I3: LayerDesc::small_tensors is the loader's single source of truth, so it
  // is asserted here rather than only inside the checkpoint test. Six GDN
  // entries, four FA; the offsets are loader/small_layout.h's constants and the
  // ruling of 2026-08-25 (RMSNorm fp32 (1+w), gated norm plain bf16) is what
  // the bake column has to say.
  using model::SmallBake;
  using model::SmallBlock;
  const auto& g = layers[0].small_tensors;
  const auto& f = layers[3].small_tensors;
  CHECK_EQ(g.size(), size_t(6));
  CHECK_EQ(f.size(), size_t(4));
  CHECK_EQ(g[0].name, std::string("input_layernorm.weight"));
  CHECK_EQ(g[0].elems, Qwen35::kHidden);
  CHECK_EQ(g[0].offset, uint32_t(loader::kNormsOffInput));
  CHECK(g[0].block == SmallBlock::Norms && g[0].bake == SmallBake::OnePlusWFp32);
  CHECK_EQ(g[1].name, std::string("post_attention_layernorm.weight"));
  CHECK_EQ(g[1].offset, uint32_t(loader::kNormsOffPost));      // 20480: fp32 now
  CHECK(g[1].bake == SmallBake::OnePlusWFp32);
  CHECK_EQ(g[2].name, std::string("linear_attn.conv1d.weight"));
  CHECK_EQ(g[2].elems, uint32_t(10240 * 4));
  CHECK_EQ(g[2].offset, uint32_t(loader::kGdnOffConv));
  CHECK(g[2].block == SmallBlock::Kind && g[2].bake == SmallBake::RawFp32Widen);
  CHECK_EQ(g[3].name, std::string("linear_attn.A_log"));
  CHECK_EQ(g[3].offset, uint32_t(loader::kGdnOffNegA));
  CHECK(g[3].bake == SmallBake::NegExpFp32);
  CHECK_EQ(g[4].name, std::string("linear_attn.dt_bias"));
  CHECK_EQ(g[4].offset, uint32_t(loader::kGdnOffDtBias));
  CHECK(g[4].bake == SmallBake::RawFp32Widen);
  CHECK_EQ(g[5].name, std::string("linear_attn.norm.weight"));
  CHECK_EQ(g[5].elems, Qwen35::kGdnHeadDim);
  CHECK_EQ(g[5].offset, uint32_t(loader::kGdnOffGatedNorm));
  CHECK(g[5].bake == SmallBake::PlainBf16);   // RMSNormGated: no +1, stays bf16
  CHECK_EQ(f[2].name, std::string("self_attn.q_norm.weight"));
  CHECK_EQ(f[2].offset, uint32_t(loader::kFaOffQNorm));
  CHECK(f[2].bake == SmallBake::OnePlusWFp32);
  CHECK_EQ(f[3].name, std::string("self_attn.k_norm.weight"));
  CHECK_EQ(f[3].elems, Qwen35::kFaHeadDim);
  CHECK_EQ(f[3].offset, uint32_t(loader::kFaOffKNorm));        // 1024: fp32 now
  CHECK(f[3].bake == SmallBake::OnePlusWFp32);
  // Every entry lands inside its block and the entries tile it exactly.
  for (const auto& [ld, kind_bytes] : {std::make_pair(layers[0], loader::kGdnBlockBytes),
                                       std::make_pair(layers[3], loader::kFaBlockBytes)}) {
    size_t norms_b = 0, kind_b = 0;
    for (const auto& t : ld.small_tensors)
      (t.block == SmallBlock::Norms ? norms_b : kind_b) +=
          size_t(t.elems) * (t.bake == SmallBake::PlainBf16 ? 2 : 4);
    CHECK_EQ(norms_b, loader::kNormsBlockBytes);
    CHECK_EQ(kind_b, kind_bytes);
  }

  std::puts("qwen35_test OK");
  return 0;
}
