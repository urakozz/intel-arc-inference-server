// Spec 14 §3.1: the model descriptor. Host-only (no device, no checkpoint).
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
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

  CHECK_EQ(q.doc_w, 15.519e9);
  CHECK_EQ(q.vocab_used, model::Qwen35::kVocabUsed);
  CHECK(!q.provisional_tuning);

  // Spec 14: Agnes 3.0 Flash.
  const ModelDesc& a = model::agnes();
  CHECK(&model::desc_for_architecture("AgnesForConditionalGeneration") == &a);
  CHECK_EQ(a.layers, uint32_t(72));
  CHECK_EQ(a.gdn_layers, uint32_t(54));
  CHECK_EQ(a.fa_layers, uint32_t(18));
  CHECK_EQ(a.parallel_ffn, uint32_t(2048));
  CHECK_EQ(a.intermediate, uint32_t(19456));
  CHECK_EQ(a.max_len_ceiling, uint32_t(65536));
  CHECK_EQ(a.vocab_used, uint32_t(248089));   // 248077 + the 12 Agnes specials
  CHECK_EQ(a.intermediate_suffix(), std::string("_I19456"));
  CHECK(a.provisional_tuning);
  check_layers(a);
  // The fold's alignment (spec 14 §2): g64 groups and 1024-column slabs / blocks.
  CHECK_EQ(17408u % 64, 0u);
  CHECK_EQ(a.intermediate % 1024, 0u);
  CHECK_EQ(a.shape(LinearId::GateUp).N % 1024, 0u);
  // The two folded rows and nothing else differ from Qwen3.8's table.
  for (size_t i = 0; i < model::kLinearCount; ++i) {
    const auto id = static_cast<LinearId>(i);
    const model::FusedLinear& fa = a.linear(id);
    const model::FusedLinear& fq = q.linear(id);
    CHECK(fa.parts == fq.parts);
    if (id == LinearId::GateUp || id == LinearId::Down) continue;
    CHECK(fa.fold == model::Fold::None);
    CHECK_EQ(fa.shape.K, fq.shape.K);
    CHECK_EQ(fa.shape.N, fq.shape.N);
    CHECK_EQ(fa.shape.S, fq.shape.S);
    CHECK_EQ(fa.shape.layout, fq.shape.layout);
  }
  const model::FusedLinear& gu = a.linear(LinearId::GateUp);
  CHECK(gu.fold == model::Fold::N);
  CHECK_EQ(gu.shape.K, uint32_t(5120));
  CHECK_EQ(gu.shape.N, uint32_t(38912));
  CHECK_EQ(gu.shape.S, uint32_t(8));     // PROVISIONAL (copied from Qwen3.8's gate||up)
  CHECK_EQ(gu.shape.layout, uint32_t(0));
  CHECK(gu.fold_parts == std::vector<std::string>({"mlp.parallel_ffn.gate_proj",
                                                   "mlp.parallel_ffn.up_proj"}));
  const model::FusedLinear& dn = a.linear(LinearId::Down);
  CHECK(dn.fold == model::Fold::K);
  CHECK_EQ(dn.shape.K, uint32_t(19456));
  CHECK_EQ(dn.shape.N, uint32_t(5120));
  CHECK_EQ(dn.shape.S, uint32_t(4));     // PROVISIONAL (copied from Qwen3.8's down)
  CHECK(dn.fold_parts == std::vector<std::string>({"mlp.parallel_ffn.down_proj"}));
  // The name map, both directions, on real checkpoint spellings.
  CHECK_EQ(a.to_engine("layers.0.delta_attn.in_proj_qkv.qweight"),
           std::string("layers.0.linear_attn.in_proj_qkv.qweight"));
  CHECK_EQ(a.to_engine("mtp.layers.0.global_attn.q_proj.weight"),
           std::string("mtp.layers.0.self_attn.q_proj.weight"));
  CHECK_EQ(a.to_engine("layers.0.mlp.parallel_ffn.down_proj.scales"),
           std::string("layers.0.mlp.parallel_ffn.down_proj.scales"));
  CHECK_EQ(a.to_checkpoint("layers.7.self_attn.o_proj"), std::string("layers.7.global_attn.o_proj"));
  CHECK_EQ(a.to_checkpoint("layers.4.linear_attn.A_log"), std::string("layers.4.delta_attn.A_log"));

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
