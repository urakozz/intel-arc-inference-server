// Spec 14 §3.1 / spec 15b: the model descriptor. Host-only (no device, no checkpoint).
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "check.h"
#include "loader/small_layout.h"
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
  // Spec 15b: every other row follows the descriptor's shapes (Review Focus 2).
  CHECK_EQ(d.gdn_conv_dim(), (2 * d.gdn_k_heads + d.gdn_v_heads) * 128u);
  CHECK_EQ(d.gdn_value_dim(), d.gdn_v_heads * 128u);
  CHECK_EQ(d.gdn_qkvz_n(), d.gdn_conv_dim() + d.gdn_value_dim());
  CHECK_EQ(d.gdn_ab_n(), 2 * d.gdn_v_heads);
  CHECK_EQ(d.fa_q_proj_n(), 2 * d.fa_q_heads * 256u);   // q || the output gate, per head
  CHECK_EQ(d.fa_kv_n(), d.fa_kv_heads * 256u);
  CHECK_EQ(d.fa_qkv_n(), d.fa_q_proj_n() + 2 * d.fa_kv_n());
  CHECK_EQ(d.fa_value_dim(), d.fa_q_heads * 256u);
  CHECK_EQ(d.fa_gqa() * d.fa_kv_heads, d.fa_q_heads);
  CHECK_EQ(d.shape(LinearId::QkvZ).K, d.hidden);
  CHECK_EQ(d.shape(LinearId::QkvZ).N, d.gdn_qkvz_n());
  CHECK_EQ(d.shape(LinearId::AB).K, d.hidden);
  CHECK_EQ(d.linear(LinearId::AB).pad_n, d.gdn_ab_n());
  CHECK(d.shape(LinearId::AB).N >= d.gdn_ab_n());
  CHECK_EQ(d.shape(LinearId::OutProj).K, d.gdn_value_dim());
  CHECK_EQ(d.shape(LinearId::OutProj).N, d.hidden);
  CHECK_EQ(d.shape(LinearId::GateUp).K, d.hidden);
  CHECK_EQ(d.shape(LinearId::Down).N, d.hidden);
  CHECK_EQ(d.shape(LinearId::Qkv).K, d.hidden);
  CHECK_EQ(d.shape(LinearId::Qkv).N, d.fa_qkv_n());
  CHECK_EQ(d.shape(LinearId::OProj).K, d.fa_value_dim());
  CHECK_EQ(d.shape(LinearId::OProj).N, d.hidden);
  CHECK_EQ(d.shape(LinearId::LmHead).K, d.hidden);
  CHECK_EQ(d.shape(LinearId::LmHead).N, model::Qwen35::kVocab);
  for (model::WeightKind k :
       {model::WeightKind::Bf16, model::WeightKind::Int4, model::WeightKind::Int8})
    CHECK_EQ(d.lm_head(k).shape.K, d.hidden);
  // The small-tensor tables follow the shapes and tile the descriptor's blocks.
  const loader::SmallLayout sl = d.small_layout();
  CHECK_EQ(sl.norms_off_post, size_t(d.hidden) * 4);
  CHECK_EQ(sl.gdn_off_nega, size_t(d.gdn_conv_dim()) * 4 * 4);
  CHECK_EQ(sl.final_norm_bytes, size_t(d.hidden) * 4);
  for (const model::LayerDesc& ld : {layers[0], layers[3]}) {
    size_t norms_b = 0, kind_b = 0;
    for (const auto& t : ld.small_tensors)
      (t.block == model::SmallBlock::Norms ? norms_b : kind_b) +=
          size_t(t.elems) * (t.bake == model::SmallBake::PlainBf16 ? 2 : 4);
    CHECK_EQ(norms_b, sl.norms_block_bytes);
    CHECK_EQ(kind_b, ld.kind == LayerKind::FA ? loader::kFaBlockBytes : sl.gdn_block_bytes);
  }
  CHECK_EQ(layers[0].small_tensors[0].elems, d.hidden);
  CHECK_EQ(layers[0].small_tensors[1].offset, uint32_t(sl.norms_off_post));
  CHECK_EQ(layers[0].small_tensors[2].elems, d.gdn_conv_dim() * 4u);   // conv1d [C][1][4]
  CHECK_EQ(layers[0].small_tensors[3].elems, d.gdn_v_heads);            // A_log
  CHECK_EQ(layers[0].small_tensors[3].offset, uint32_t(sl.gdn_off_nega));
  CHECK_EQ(layers[0].small_tensors[4].elems, d.gdn_v_heads);            // dt_bias
  CHECK_EQ(layers[0].small_tensors[4].offset, uint32_t(sl.gdn_off_dtbias));
  CHECK_EQ(layers[0].small_tensors[5].offset, uint32_t(sl.gdn_off_gated_norm));
  CHECK_EQ(layers[3].small_tensors[0].elems, d.hidden);
  // Every shape is kernel-legal: N%64, K%64, S | K/64; and the spec 5
  // 1024-column slab / 1024-blocked Hadamard alignment holds (spec 14 §2).
  for (size_t i = 0; i < model::kLinearCount; ++i) {
    const auto& s = d.shape(static_cast<LinearId>(i));
    CHECK_EQ(s.N % 64, uint32_t(0));
    CHECK_EQ(s.K % 64, uint32_t(0));
    CHECK_EQ((s.K / 64) % s.S, uint32_t(0));
  }
  // The prefill slab / 1024-blocked Hadamard alignment is a dense-MLP property.
  if (d.ffn == model::FfnKind::Dense) CHECK_EQ(d.intermediate % 1024, uint32_t(0));
}

// Qwen3.8's shapes, which Agnes shares (spec 15b R0: both unchanged).
void check_qwen38_shapes(const ModelDesc& d) {
  CHECK_EQ(d.hidden, uint32_t(5120));
  CHECK_EQ(d.fa_q_heads, uint32_t(24));
  CHECK_EQ(d.fa_kv_heads, uint32_t(4));
  CHECK_EQ(d.gdn_k_heads, uint32_t(16));
  CHECK_EQ(d.gdn_v_heads, uint32_t(48));
  CHECK_EQ(d.gdn_conv_dim(), uint32_t(10240));
  CHECK_EQ(d.gdn_value_dim(), uint32_t(6144));
  CHECK_EQ(d.gdn_qkvz_n(), uint32_t(16384));
  CHECK_EQ(d.gdn_ab_n(), uint32_t(96));
  CHECK_EQ(d.fa_q_proj_n(), uint32_t(12288));
  CHECK_EQ(d.fa_qkv_n(), uint32_t(14336));
  CHECK_EQ(d.fa_value_dim(), uint32_t(6144));
  CHECK_EQ(d.fa_gqa(), uint32_t(6));
  CHECK(d.ffn == model::FfnKind::Dense);
  CHECK(!d.is_moe());
  CHECK(!d.tied_embeddings);
  CHECK_EQ(d.mtp_intermediate, uint32_t(17408));            // no parallel FFN in Agnes's head
  CHECK_EQ(d.mtp_checkpoint_bytes(), size_t(849398784));    // docs/03: 0.849 GB
  CHECK(!d.mtp_head_moe());                                 // spec 15e: dense head
  CHECK_EQ(d.mtp_checkpoint_tensors(), size_t(15));
  const loader::SmallLayout sl = d.small_layout();
  const loader::SmallLayout& q = loader::kQwen38Small;
  CHECK_EQ(sl.norms_off_post, q.norms_off_post);
  CHECK_EQ(sl.norms_block_bytes, q.norms_block_bytes);
  CHECK_EQ(sl.gdn_off_nega, q.gdn_off_nega);
  CHECK_EQ(sl.gdn_off_dtbias, q.gdn_off_dtbias);
  CHECK_EQ(sl.gdn_off_gated_norm, q.gdn_off_gated_norm);
  CHECK_EQ(sl.gdn_block_bytes, q.gdn_block_bytes);
  CHECK_EQ(sl.final_norm_bytes, q.final_norm_bytes);
  // The kernel-name suffixes are empty: every Qwen3.8 / Agnes binary keeps its name.
  CHECK_EQ(d.hidden_suffix(), std::string());
  CHECK_EQ(d.gdn_suffix(), std::string());
  CHECK_EQ(d.fa_suffix(), std::string());
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
  CHECK_EQ(q.intermediate_suffix(), std::string());
  CHECK_EQ(q.to_engine("layers.0.linear_attn.A_log"), std::string("layers.0.linear_attn.A_log"));
  CHECK_EQ(q.to_checkpoint("layers.3.self_attn.q_proj"), std::string("layers.3.self_attn.q_proj"));
  check_layers(q);
  check_qwen38_shapes(q);
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
  CHECK_EQ(a.vocab_used, uint32_t(248089));   // 248077 + the 12 Agnes specials
  CHECK_EQ(a.intermediate_suffix(), std::string("_I19456"));
  CHECK(a.provisional_tuning);
  check_layers(a);
  check_qwen38_shapes(a);
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

  // Spec 15b: Ornith 1.5 35B-A3B (spec 15 §1; its config.json and index).
  const ModelDesc& o = model::ornith();
  CHECK(&model::desc_for_architecture("Qwen3_5MoeForConditionalGeneration") == &o);
  CHECK_EQ(o.name, std::string("ornith-1.5-35b-a3b"));
  CHECK_EQ(o.layers, uint32_t(40));
  CHECK_EQ(o.gdn_layers, uint32_t(30));
  CHECK_EQ(o.fa_layers, uint32_t(10));
  CHECK_EQ(o.hidden, uint32_t(2048));
  CHECK_EQ(o.fa_q_heads, uint32_t(16));
  CHECK_EQ(o.fa_kv_heads, uint32_t(2));
  CHECK_EQ(o.fa_q_proj_n(), uint32_t(8192));   // 16 x 256 x (q || gate)
  CHECK_EQ(o.fa_qkv_n(), uint32_t(9216));      // 8192 + 512 + 512
  CHECK_EQ(o.fa_value_dim(), uint32_t(4096));
  CHECK_EQ(o.fa_gqa(), uint32_t(8));
  CHECK_EQ(o.gdn_k_heads, uint32_t(16));
  CHECK_EQ(o.gdn_v_heads, uint32_t(32));
  CHECK_EQ(o.gdn_conv_dim(), uint32_t(8192));
  CHECK_EQ(o.gdn_value_dim(), uint32_t(4096));
  CHECK_EQ(o.gdn_qkvz_n(), uint32_t(12288));
  CHECK_EQ(o.gdn_ab_n(), uint32_t(64));
  CHECK(o.ffn == model::FfnKind::Moe);
  CHECK(o.is_moe());
  CHECK_EQ(o.moe.experts, uint32_t(256));
  CHECK_EQ(o.moe.top_k, uint32_t(8));
  CHECK_EQ(o.moe.expert_intermediate, uint32_t(512));
  CHECK_EQ(o.moe.shared_intermediate, uint32_t(512));
  CHECK(o.moe.has_shared_gate);
  // A MoE model's dense MLP rows describe its shared expert (one expert's shape).
  CHECK_EQ(o.intermediate, o.moe.shared_intermediate);
  CHECK(o.linear(LinearId::GateUp).parts ==
        std::vector<std::string>({"mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"}));
  CHECK(o.linear(LinearId::Down).parts ==
        std::vector<std::string>({"mlp.shared_expert.down_proj"}));
  CHECK_EQ(o.vocab_used, uint32_t(248070));   // tokenizer.json's ids end at </think> = 248069
  CHECK(!o.tied_embeddings);
  CHECK(o.name_map.empty());                  // linear_attn. / self_attn., as Qwen3.8
  CHECK_EQ(o.mtp_intermediate, uint32_t(0));  // its MTP head is one MoE layer (15e)
  // Spec 15e: the head's 785 bf16 tensors, 1,689,281,536 B - the sum over the published
  // checkpoint's shard 16 header (2026-10-05): fc, q/k/v/o, the five norms, q/k_norm,
  // the router, the shared gate and 257 SwiGLU experts of 512.
  CHECK(o.mtp_head_moe());
  CHECK_EQ(o.mtp_checkpoint_bytes(), size_t(1689281536));
  CHECK_EQ(o.mtp_checkpoint_tensors(), size_t(785));
  CHECK(o.provisional_tuning);
  CHECK_EQ(o.doc_w, 0.0);                     // no int4 checkpoint exists yet (15a)
  check_layers(o);
  // Ornith's small blocks: norms 2 x 2048 fp32; GDN conv 8192 x 4 fp32, 32 + 32 fp32, 128 bf16.
  const loader::SmallLayout os = o.small_layout();
  CHECK_EQ(os.norms_off_post, size_t(8192));
  CHECK_EQ(os.norms_block_bytes, size_t(16384));
  CHECK_EQ(os.gdn_off_nega, size_t(131072));
  CHECK_EQ(os.gdn_off_dtbias, size_t(131200));
  CHECK_EQ(os.gdn_off_gated_norm, size_t(131328));
  CHECK_EQ(os.gdn_block_bytes, size_t(131584));
  CHECK_EQ(os.final_norm_bytes, size_t(8192));
  // Its kernels carry shape suffixes (no binary is built for them until 15c).
  CHECK_EQ(o.hidden_suffix(), std::string("_D2048"));
  CHECK_EQ(o.gdn_suffix(), std::string("_GK16V32"));
  CHECK_EQ(o.fa_suffix(), std::string("_Q16KV2"));
  CHECK_EQ(o.intermediate_suffix(), std::string("_I512"));
  // Spec 15c: the device form of the MoE block (MoeDesc's derived numbers): 257
  // weight blocks (the shared expert last), 9 slots per token, the router || shared
  // gate GEMV at 256 + 1 rows padded to 272.
  CHECK_EQ(o.moe.shared_block(), uint32_t(256));
  CHECK_EQ(o.moe.blocks(), uint32_t(257));
  CHECK_EQ(o.moe.slots(), uint32_t(9));
  CHECK_EQ(o.moe.router_n(), uint32_t(272));
  // The FFN's partials for the next residual fold: the dense down's S, none on a MoE
  // model (moe_down folds into the residual stream itself).
  CHECK_EQ(q.ffn_fold_s(), uint32_t(4));
  CHECK_EQ(a.ffn_fold_s(), uint32_t(4));
  CHECK_EQ(o.ffn_fold_s(), uint32_t(0));
  // The loader's gate (loader::load calls it after picking the descriptor): from spec
  // 15c Ornith loads (decode), and so do the dense models.
  model::require_loadable(o);
  model::require_loadable(q);
  model::require_loadable(a);
  // A MoE shape the kernels are not written for is refused by name (here: no shared
  // expert gate; top-k 9, past the 8-slot route row; and - spec 18b - 96 and 100 experts:
  // a multiple of 16 and not, but neither the power of two moe.cl's route work-group is).
  for (int variant = 0; variant < 4; ++variant) {
    ModelDesc bad = o;
    if (variant == 0) bad.moe.has_shared_gate = false;
    else if (variant == 1) bad.moe.top_k = 9;
    else bad.moe.experts = variant == 2 ? 96 : 100;
    bool threw = false;
    try {
      model::require_loadable(bad);
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()).find("is not implemented") != std::string::npos;
    }
    CHECK(threw);
  }
  // Prefill: the dense models pass, and from spec 15d Ornith does too (the grouped
  // experts); a MoE shape the prefill kernels are not written for is refused by name
  // (here: a hidden size that is not whole 1024-k rotation blocks).
  model::require_prefill(q);
  model::require_prefill(a);
  model::require_prefill(o);
  bool pf_threw = false;
  try {
    ModelDesc bad = o;
    bad.hidden = 1536;
    model::require_prefill(bad);
  } catch (const std::runtime_error& e) {
    pf_threw = std::string(e.what()).find("not implemented on the prefill path") != std::string::npos;
  }
  CHECK(pf_threw);

  // An unknown architecture throws, naming it and the three supported ones.
  bool threw = false;
  try {
    model::desc_for_architecture("LlamaForCausalLM");
  } catch (const std::runtime_error& e) {
    const std::string w = e.what();
    threw = w.find("LlamaForCausalLM") != std::string::npos &&
            w.find("Qwen3_5MoeForConditionalGeneration") != std::string::npos;
  }
  CHECK(threw);

  std::puts("model_desc_test OK");
  return 0;
}
