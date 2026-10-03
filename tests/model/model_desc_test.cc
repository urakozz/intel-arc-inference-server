// Spec 14 §3.1: the model descriptor. Host-only (no device, no checkpoint).
#include <cstdio>
#include <stdexcept>
#include <string>
#include "check.h"
#include "model/model_desc.h"

namespace {

using model::LayerKind;
using model::LinearId;
using model::ModelDesc;

// Layer-kind counts and the FA pattern, from the descriptor's own layer list.
void check_layers(const ModelDesc& d) {
  const auto layers = d.layer_descs();
  CHECK_EQ(layers.size(), size_t(d.layers));
  CHECK_EQ(d.gdn_layers + d.fa_layers, d.layers);
  uint32_t gdn = 0, fa = 0;
  for (uint32_t i = 0; i < layers.size(); ++i) {
    CHECK_EQ(layers[i].index, i);
    const bool is_fa = i % 4 == 3;
    CHECK(layers[i].kind == (is_fa ? LayerKind::FA : LayerKind::GDN));
    CHECK_EQ(ModelDesc::is_fa(i), is_fa);
    (is_fa ? fa : gdn)++;
    CHECK_EQ(layers[i].linears.size(), size_t(is_fa ? 4 : 5));
    CHECK(layers[i].linears[layers[i].linears.size() - 2].id == LinearId::GateUp);
    CHECK(layers[i].linears.back().id == LinearId::Down);
  }
  CHECK_EQ(gdn, d.gdn_layers);
  CHECK_EQ(fa, d.fa_layers);
  // The MLP rows follow the intermediate size: gate||up is 2 x I wide, down reads I.
  CHECK_EQ(d.shape(LinearId::GateUp).N, 2 * d.intermediate);
  CHECK_EQ(d.shape(LinearId::Down).K, d.intermediate);
  // Every shape is kernel-legal: N%64, K%64, S | K/64; and the spec 5
  // 1024-column slab / 1024-blocked Hadamard alignment holds (spec 14 §2).
  for (size_t i = 0; i < model::kLinearCount; ++i) {
    const auto& s = d.shape(static_cast<LinearId>(i));
    CHECK_EQ(s.N % 64, uint32_t(0));
    CHECK_EQ(s.K % 64, uint32_t(0));
    CHECK_EQ((s.K / 64) % s.S, uint32_t(0));
  }
  CHECK_EQ(d.intermediate % 1024, uint32_t(0));
}

}  // namespace

int main() {
  const ModelDesc& q = model::qwen38();
  CHECK_EQ(q.architecture, std::string("Qwen3_5ForConditionalGeneration"));
  CHECK_EQ(q.layers, uint32_t(64));
  CHECK_EQ(q.gdn_layers, uint32_t(48));
  CHECK_EQ(q.fa_layers, uint32_t(16));
  CHECK_EQ(q.intermediate, uint32_t(17408));
  CHECK(!q.has_parallel_ffn());
  CHECK_EQ(q.max_len_ceiling, uint32_t(0));
  CHECK_EQ(q.intermediate_suffix(), std::string());
  CHECK_EQ(q.to_engine("layers.0.linear_attn.A_log"), std::string("layers.0.linear_attn.A_log"));
  CHECK_EQ(q.to_checkpoint("layers.3.self_attn.q_proj"), std::string("layers.3.self_attn.q_proj"));
  check_layers(q);
  // Qwen3.8's MLP rows are exactly the measured production map, no fold.
  CHECK_EQ(q.shape(LinearId::GateUp).N, uint32_t(34816));
  CHECK_EQ(q.shape(LinearId::GateUp).S, uint32_t(8));
  CHECK_EQ(q.shape(LinearId::Down).K, uint32_t(17408));
  CHECK_EQ(q.shape(LinearId::Down).S, uint32_t(4));
  for (size_t i = 0; i < model::kLinearCount; ++i)
    CHECK(q.linear(static_cast<LinearId>(i)).fold == model::Fold::None);
  CHECK(&model::desc_for_architecture("Qwen3_5ForConditionalGeneration") == &q);

  // An unknown architecture throws, naming it.
  bool threw = false;
  try {
    model::desc_for_architecture("LlamaForCausalLM");
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("LlamaForCausalLM") != std::string::npos;
  }
  CHECK(threw);

  std::puts("model_desc_test OK");
  return 0;
}
