#include <cstdio>
#include "check.h"
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
  // The measured table (docs/probe-gemv-2026-08-24.md): spot-check the two
  // that differ from naive expectations.
  CHECK_EQ(Qwen35::shape(LinearId::OutProj).K, uint32_t(6144));
  CHECK_EQ(Qwen35::shape(LinearId::OutProj).S, uint32_t(16));
  CHECK_EQ(Qwen35::shape(LinearId::GateUp).S, uint32_t(4));

  // Fusion wiring for layer 0 (GDN) and 3 (FA).
  CHECK_EQ(layers[0].linears.size(), size_t(5));
  CHECK(layers[0].linears[0].id == LinearId::QkvZ);
  CHECK_EQ(layers[0].linears[0].parts.size(), size_t(2));
  CHECK_EQ(layers[0].linears[1].pad_n, uint32_t(96));   // a||b padded 96 -> 128
  CHECK_EQ(layers[3].linears.size(), size_t(4));
  CHECK(layers[3].linears[0].id == LinearId::Qkv);
  CHECK_EQ(Qwen35::layer_prefix(5), std::string("layers.5."));
  std::puts("qwen35_test OK");
  return 0;
}
