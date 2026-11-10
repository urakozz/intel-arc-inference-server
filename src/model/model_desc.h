#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "loader/small_layout.h"
#include "model/qwen35.h"

// Spec 14 §3.1 / spec 15b: the per-model shape as data. What the supported
// checkpoints share - head dims, vocab, rotary, the per-layer linear order and
// the small tensors' names - stays `constexpr` in `model::Qwen35`. What differs
// is here, chosen by the loader from `config.json`'s `architectures[0]`:
//
//                     Qwen3.8-27B            Ornith 1.5 35B-A3B
//   architecture      Qwen3_5ForCondi-       Qwen3_5MoeForCondi-
//                     tionalGeneration       tionalGeneration
//   layers            64 = 48 GDN + 16 FA    40 = 30 GDN + 10 FA
//   FA layers         l % 4 == 3             l % 4 == 3
//   hidden            5120                   2048
//   FA q / kv heads   24 / 4 (GQA 6)         16 / 2 (GQA 8)
//   GDN k / v heads   16 / 48 (conv 10240)   16 / 32 (conv 8192)
//   FFN               dense 17408            MoE: 256 experts, top-8,
//                                            expert 512, shared
//                                            expert 512 + sigmoid gate
//   tensor names      linear_attn. /         linear_attn. /
//                     self_attn.             self_attn.
//   max_len           any multiple of 256 up to config.json's max_position_embeddings
//                     (262144 for both) that the memory plan fits on the card (spec 6 §10)
//   loadable          yes                    decode (spec 15c), prefill (15d), MTP and
//                                            b70-serve (15e)
//
// (Agnes 3.0 Flash, spec 14's second dense descriptor, was dropped on 2026-11-07;
// desc_for_architecture refuses its architecture by name.)
//
// Descriptors are process-lifetime singletons: hold them by reference/pointer.
namespace model {

// Spec 15b: the feed-forward block's kind. Dense is the SwiGLU MLP the table's
// GateUp / Down rows run; Moe adds routed experts (`MoeDesc`), run by spec 15c's
// decode kernels (src/kernels/moe.cl).
enum class FfnKind { Dense, Moe };

// The routed-expert shape of a FfnKind::Moe model (spec 15 §1). Unused (zero) on
// a dense model.
//
// Spec 15c: the device form, derived here so the loader, the buffer sizes, the
// capture and the kernels' CMake defines read ONE set of numbers:
//   * the expert weights of a layer are `blocks()` contiguous int4 g64 layout-1
//     blocks (common/repack.h) addressed by block index - the routed experts at
//     0..experts-1, the shared expert at `shared_block()` (= experts), so one
//     kernel serves all `slots()` = top_k + 1 slots of a token (spec 15 §9);
//   * the router and the shared expert's sigmoid gate are one bf16 GEMV of
//     `router_n()` rows: the experts' rows, the gate's row at `experts`, zero rows
//     to the next multiple of 16 (Ornith 256 + 1 -> 272).
struct MoeDesc {
  uint32_t experts = 0;               // routed experts per layer
  uint32_t top_k = 0;                 // experts per token
  uint32_t expert_intermediate = 0;   // one routed expert's SwiGLU width
  uint32_t shared_intermediate = 0;   // the always-on shared expert's width; 0 = none
  bool has_shared_gate = false;       // the shared expert is scaled by sigmoid(x . g)

  uint32_t shared_block() const { return experts; }
  uint32_t blocks() const { return experts + 1; }       // routed + the shared expert
  uint32_t slots() const { return top_k + 1; }          // per token: top_k routed + shared
  uint32_t router_n() const { return (experts + 1 + 15) / 16 * 16; }
};

struct ModelDesc {
  std::string name;           // a short label for logs: "qwen3.8", "ornith-1.5-35b-a3b"
  std::string architecture;   // config.json's architectures[0]
  uint32_t layers = 0, gdn_layers = 0, fa_layers = 0;
  // Spec 15b: the per-model widths (config.json hidden_size, num_attention_heads,
  // num_key_value_heads, linear_num_key_heads, linear_num_value_heads). The head
  // dims (256 FA, 128 GDN) are shared: model::Qwen35.
  uint32_t hidden = 0;
  uint32_t fa_q_heads = 0, fa_kv_heads = 0;
  uint32_t gdn_k_heads = 0, gdn_v_heads = 0;
  FfnKind ffn = FfnKind::Dense;
  MoeDesc moe{};                      // FfnKind::Moe only
  bool tied_embeddings = false;       // lm_head shares embed_tokens (none supported does)
  // The MTP head's dense MLP width: 17408 on Qwen3.8. 0 = the head's FFN is not dense (Ornith: one MoE
  // layer of the main model's shape, spec 15e - mtp_head_moe()).
  uint32_t mtp_intermediate = 0;
  // The MLP width the engine runs. For a MoE model, its shared expert's width - the dense GateUp / Down rows of
  // the table describe the shared expert (spec 15b; the routed experts are `moe`).
  uint32_t intermediate = 0;
  std::array<FusedLinear, kLinearCount> table{};   // indexed by LinearId ordinal
  // lm_head's int4 and int8 rows (the bf16 one is table[LmHead]); `lm_head(kind)`
  // picks. Per descriptor because K is the model's hidden size (spec 15b).
  FusedLinear lm_int4{}, lm_int8{};
  // The GDN a||b projection's int4 row (the bf16 one is table[AB]); `ab(kind)` picks.
  // Like lm_head, a||b's kind is a property of the CHECKPOINT, not of the model: Qwen3.8's
  // exports keep in_proj_a / in_proj_b bf16, the published Ornith int4
  // checkpoint quantises them (int4 g64 sym, spec 15 §13). The loader classifies by
  // content (loader/ab.h) and the capture binds by the loaded weight's kind: gemv_bf16 for
  // the bf16 row, gemv.cl at S 1 for this one - both write fp32 [M][128] at ab_out, the
  // real 2 x v-heads columns zero-padded to 128 (gdn_step's AB_STRIDE). A separate member,
  // not a ninth table entry, for lm_int4's reason: LinearId::AB stays one ordinal and every
  // table walk (buffer sizes, the h8 scales, the variant names) sees the bf16 row.
  FusedLinear ab_int4{};
  // `W` with a bf16 lm_head: the bytes a decode step streams (int4 qweight +
  // scales + bf16 per-layer tensors + lm_head), the loader's 2% cross-check
  // (docs/13). Qwen3.8: 15.519 GB, measured (docs/03).
  // Ornith: 2.345 GB, summed from the int4 checkpoint's headers the same way, with a
  // MoE layer's per-token share - the router and shared gate, the shared expert and
  // top_k routed experts (all the same size) - in place of a dense MLP (spec 15 §13).
  double doc_w = 0;
  // The ids the tokenizer defines - the greedy argmax masks every lm_head row at or
  // above it (argmax.cl's VOCAB_USED). Qwen3.8: 248077 (Qwen35::kVocabUsed). Ornith's int4 checkpoint's
  // tokenizer.json defines Qwen3.8's 248077 (spec 15 §13).
  uint32_t vocab_used = 0;
  // True for a descriptor whose GEMV tuning rows were copied, not measured (Ornith's,
  // copied from Qwen3.8's map).
  bool provisional_tuning = false;
  // The per-layer non-GEMV tensors of each layer kind, in engine names - the
  // loader's single source of truth (fix I3). Element counts and offsets follow
  // this descriptor's shapes (`small_layout()`); built with the descriptor.
  std::vector<SmallTensor> small_gdn, small_fa;

  bool is_moe() const { return ffn == FfnKind::Moe; }
  // Spec 15c: the split-K width of the partials the FFN leaves for the next
  // residual fold (the next layer's input norm, or the final norm): the dense
  // down GEMV's S, or 0 on a MoE model - its moe_down kernel folds the block's
  // output into the residual stream itself (src/kernels/moe.cl), so the next
  // prep_res_fold has nothing to fold (its SP0 variant).
  uint32_t ffn_fold_s() const;
  // [GDN, GDN, GDN, FA] repeating - every supported model (`full_attention_interval`
  // / `global_attention_interval` 4).
  static bool is_fa(uint32_t layer) { return layer % 4 == 3; }

  // --- derived widths (spec 15b Review Focus 2) -------------------------------
  // GDN conv channels: q and k (k-heads each) plus v (v-heads), x 128. The z half
  // of the qkv||z GEMV does not go through the conv. Qwen3.8 10240, Ornith 8192.
  uint32_t gdn_conv_dim() const { return (2 * gdn_k_heads + gdn_v_heads) * Qwen35::kGdnHeadDim; }
  // v-heads x 128: z's width, out_proj's K, one token's gdn_o row. Qwen3.8 6144.
  uint32_t gdn_value_dim() const { return gdn_v_heads * Qwen35::kGdnHeadDim; }
  uint32_t gdn_qkvz_n() const { return gdn_conv_dim() + gdn_value_dim(); }   // 16384
  // a || b before the kernel's zero-pad: one decay and one beta per v-head. 96.
  uint32_t gdn_ab_n() const { return 2 * gdn_v_heads; }
  // q_proj with the output gate: q || gate per head, 2 x q-heads x 256. 12288.
  uint32_t fa_q_proj_n() const { return 2 * fa_q_heads * Qwen35::kFaHeadDim; }
  uint32_t fa_kv_n() const { return fa_kv_heads * Qwen35::kFaHeadDim; }      // k (or v): 1024
  uint32_t fa_qkv_n() const { return fa_q_proj_n() + 2 * fa_kv_n(); }       // 14336
  // q-heads x 256: o_proj's K, attn_out's row. Qwen3.8 6144, Ornith 4096.
  uint32_t fa_value_dim() const { return fa_q_heads * Qwen35::kFaHeadDim; }
  uint32_t fa_gqa() const { return fa_q_heads / fa_kv_heads; }   // q-heads per kv-head: 6 / 8
  // The per-layer small blocks' byte layout (loader/small_layout.h).
  loader::SmallLayout small_layout() const {
    return loader::make_small_layout(hidden, gdn_conv_dim(), gdn_v_heads);
  }
  const std::vector<SmallTensor>& small_tensors(LayerKind kind) const {
    return kind == LayerKind::FA ? small_fa : small_gdn;
  }
  // Spec 15e: the MTP head's FFN is the main model's MoE block (one routed layer of
  // `moe`'s shape: router, routed experts, the gated shared expert) - Ornith's
  // `mtp.layers.0.mlp.experts.N.*` - rather than a dense MLP at mtp_intermediate.
  bool mtp_head_moe() const { return is_moe() && mtp_intermediate == 0; }
  // The MTP head's bf16 checkpoint bytes, from the shapes: fc [hidden][2 hidden],
  // q/k/v/o, the FFN, five hidden-wide norms and q_norm / k_norm. The FFN is the dense
  // MLP at mtp_intermediate (Qwen3.8: 15 tensors, 849,398,784 B, docs/03) or,
  // on a MoE head (spec 15e), the router [experts][hidden], the shared gate [1][hidden]
  // and experts + 1 SwiGLU experts of expert_intermediate (Ornith: 785 tensors,
  // 1,689,281,536 B - the sum over its shard 16 header, 2026-10-05). An int4 export of
  // the head's experts ships fewer bytes; the loader checks this figure only when every
  // head tensor is bf16.
  size_t mtp_checkpoint_bytes() const;
  // The head's checkpoint tensor count when every tensor is bf16: 15 dense, 14 + 3 x
  // (experts + 1) on a MoE head (Ornith 785).
  size_t mtp_checkpoint_tensors() const;

  // The table row. Throws std::out_of_range on kCount or a bad cast.
  // `linear(LmHead)` is the bf16 row (see Qwen35's history note); callers that
  // must honour the checkpoint use `lm_head(kind)`.
  const FusedLinear& linear(LinearId id) const;
  const GemvShape& shape(LinearId id) const { return linear(id).shape; }
  const FusedLinear& lm_head(WeightKind kind) const;
  // a||b's row for a checkpoint that ships in_proj_a / in_proj_b in `kind`: Bf16 is
  // table[AB] (the same object as linear(AB)), Int4 is ab_int4. Int8 throws.
  const FusedLinear& ab(WeightKind kind) const;
  // All `layers` layer descriptors, fully populated, in index order.
  std::vector<LayerDesc> layer_descs() const;

  // The suffix that kernels baking the intermediate size carry in their variant
  // names: "" at Qwen3.8's 17408 (so every Qwen3.8 binary keeps its name and its
  // command line), "_I<intermediate>" otherwise.
  std::string intermediate_suffix() const;
  // Spec 15b: the suffixes of kernels that bake a model shape their name does not
  // otherwise carry (kernels.h, kernels::hidden_suffix / gdn_suffix / fa_suffix).
  // "" at Qwen3.8's shapes, so no existing binary is renamed.
  std::string hidden_suffix() const;   // "_D<hidden>"
  std::string gdn_suffix() const;      // "_GK<k-heads>V<v-heads>"
  std::string fa_suffix() const;       // "_Q<q-heads>KV<kv-heads>"
};

const ModelDesc& qwen38();
// Spec 17b: tensor parallel over two cards (`--parallel tp`). How the two cards add their halves of a row-parallel
// linear (out_proj, o_proj, down) into the residual - plan 17b's review, operator decision D1:
//   Slices  each card runs the linear at the one-card S / 2 over the one-card slice boundaries (card 0 slices
//           0 .. S/2 - 1, card 1 the rest) and sends its raw slices; both cards fold all S in one-card order, so
//           every output is BITWISE one card's;
//   Folded  each card runs the one-card S over its K half and sends one folded fp32 row (spec 17a's form); both
//           cards add card 0's row then card 1's - bitwise between the cards, judged against one card.
enum class TpCrossing { Slices, Folded };
constexpr uint32_t kTpSize = 2;   // spec 17 builds and gates two cards
const char* tp_crossing_name(TpCrossing c);
// The per-card descriptor of `one` under TP (both ranks share it; the rank only decides which slice the loader
// takes): half the GDN and FA heads, half the MLP width, the head's vocabulary half (124160 rows), the row-parallel
// rows' S per `c`. doc_w 0 (the loader's W check is per model; tests sum the two cards instead), no MTP head (17d).
// Qwen3.8 only; anything else throws naming the model.
const ModelDesc& tp_card_desc(const ModelDesc& one, TpCrossing c);
const ModelDesc& ornith();   // Ornith 1.5 35B-A3B (spec 15): decode from spec 15c
// The descriptor for config.json's `architectures[0]`; throws std::runtime_error
// naming the architecture and the supported ones (a dropped model - Agnes 3.0 Flash's
// AgnesForConditionalGeneration - by name).
const ModelDesc& desc_for_architecture(const std::string& architecture);
// Throws std::runtime_error when the engine cannot run this descriptor: tied
// embeddings, or a FfnKind::Moe shape src/kernels/moe.cl is not written for (a
// shared expert with a sigmoid gate and the routed experts' width, top_k <= 8,
// experts a power of two in [16, 256] - moe.cl's route work-group; Ornith's 256 is). The
// loader calls it right after desc_for_architecture.
void require_loadable(const ModelDesc& desc);
// Throws std::runtime_error naming the reason when the prefill path cannot run this
// model: from spec 15d a FfnKind::Moe model prefills through the grouped experts, so
// only a MoE shape those kernels are not written for is refused (Ornith's is not).
// Called by Engine::prefill / prepare_prefill and by the CLIs before they plan a prefill.
void require_prefill(const ModelDesc& desc);

}  // namespace model
