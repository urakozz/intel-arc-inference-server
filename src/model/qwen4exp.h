#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/json.h"
#include "model/qwen35.h"   // GemvShape, WeightKind, Fuse - the shared vocabulary, nothing else

// Spec 21-QFN-v1b: Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, config.json model_type `qwen4_exp`)
// as data - a fourth model BESIDE qwen3_5 and Kolibri-1 (spec 21-QFN-v1 §6: its own table, loader,
// kernels and engine; their files untouched). Every number of `qwen4exp()` is the checkpoint's
// (config.json text_config, the index and every shard header):
//
//   layers 48: GDN (`linear_attention`) at l % 4 != 3 (36), QSA sparse attention at l % 4 == 3 (12) -
//          the checkpoint spells it `full_attention`, the tiny model `qwen_sparse_attention`
//   hidden 2560; the residual is 4 hyper-connection streams (hc 4, low rank 320): per layer an attention-
//          side and an MLP-side gated residual, after the last layer the final mixer; NO input /
//          post-attention / final norm - the HC norm replaces them. Every RMSNorm (1 + w) in fp32, eps 1e-6,
//          but the GDN gated norm (plain w)
//   GDN    16 k / 48 v heads x 128, conv 4 over 10240 channels; v head h reads k head h // 3; output gate
//          sigmoid(z) (`output_gate_type: sigmoid` - Ornith / Qwen3.8 have silu)
//   QSA    24 q / 2 kv heads x 256 (GQA 12); q_proj gives [q 256 | gate 256] per head; partial NEOX RoPE on
//          the first 64 dims, theta 1e7; the indexer: 4 query heads + 1 raw key of 128 (index_qk_proj 640),
//          blocks of 4 positions, the top 512 blocks (2048 positions) + the open block's tail: <= 2051
//   MoE    every layer: 512 routed SwiGLU experts (640), top-10, softmax renormalised (norm_topk_prob
//          true), no bias; + sigmoid(shared_expert_gate x) x the shared expert (640)
//   PLE    one per-layer embedding at ple_layer_ids [2] - ONE-indexed: layer_idx 1, a GDN layer; 16 n-gram
//          heads (8 bigram + 8 trigram) x 160 over a hashed table of distinct primes just above
//          ngram_vocab_size_base (20,000,000), padded to a multiple of 128; key / value projections
//          2560 -> 10240 / 2560, a dilated (3) depthwise conv of 4 taps over a 9-row history
//   MTP    one head (`mtp.*`): its own QSA layer and 512 experts, both HCs, a final mixer, fc_embedding /
//          fc_hidden, pre_fc_norm_embedding [2560], pre_fc_norm_hidden [10240]
//   vocab  248320 rows (248077 ids in the original's tokenizer.json); untied embed_tokens / lm_head;
//          generation EOS {248046, 248044}; the PLE's EOS 248044; trained context 262144
//
// A plain struct so a test can build a small one (8 experts, tests/loader/qwen4exp_repack_test.cc); every
// width the loader, the planner and (21-QFN-v1c) the kernels use is DERIVED here. A synthetic checkpoint
// (tools/quantize/qwen4exp/make_synth.py) or `--layers N` holds the first N layers of the real pattern
// (N >= 2: the PLE layer is layer 1).
namespace model {

// A tensor group's form, read from the tensor names (`.qweight` / `.scales` / `.qzeros`: Int4, `.weight`:
// Bf16), all-or-nothing per group (loader::q4_forms).
enum class Q4Form { Int4, Bf16 };
const char* q4_form_name(Q4Form f);
// The checkpoint's forms: (a) the dense projections (GDN in_proj_qkv / in_proj_z / out_proj, QSA
// q / k / v / o_proj), (b) the shared experts, (c) the MTP head's routed experts. The main model's routed
// experts are always int4 - g64 (ours) or g128 (Intel's, expanded exactly to g64 at load).
//   ours   (21q)   {Int4, Int4, decision 6}, expert_group 64
//   Intel's export {Bf16, Bf16, Bf16},       expert_group 128
struct Q4Forms {
  Q4Form dense = Q4Form::Int4, shared = Q4Form::Int4, mtp_experts = Q4Form::Int4;
  uint32_t expert_group = 64;   // 64 (ours) | 128 (Intel's, expanded at load)
};

// The family's GEMV rows that are not routed experts (the routed experts are the loader's blocks).
//   GdnQkvz  in_proj_qkv || in_proj_z (Fuse::Concat, Qwen3.8's qkv||z)     2560 x 16384
//   GdnAb    in_proj_a || in_proj_b, bf16 always, padded to 128            2560 x 128 (96 used)
//   GdnOut   out_proj                                                       6144 x 2560
//   QsaQkvg  q_proj (q||gate per head) || k_proj || v_proj                  2560 x 13312
//   QsaIdx   indexer.index_qk_proj, bf16 always                             2560 x 640
//   QsaO     o_proj                                                         6144 x 2560
//   Router   mlp.gate (512 rows) || shared_expert_gate (row 512), zero to 528, bf16 always
//   PleKv    ple.key_proj || ple.value_proj, bf16 always                    2560 x 12800
//   LmHead   lm_head, bf16 (the loader builds spec 9's int8 head from it)   2560 x 248320
enum class Q4LinearId { GdnQkvz, GdnAb, GdnOut, QsaQkvg, QsaIdx, QsaO, Router, PleKv, LmHead, kCount };
struct Q4Linear {
  Q4LinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // layer-relative checkpoint prefixes, in column order
};

struct Qwen4ExpDesc;

// Plan 21-QFN-v2b: where the tensor-parallel form keeps the two parts read once a token, not a layer - the embedding (one
// row) and the MTP head (a draft step). Both: whole on each card (Qwen3.8's TP). Split: the embedding on card 0 (card
// 1 gathers its row from card 0's table over peer access, as pp's head already reads device 0's), the head on card 1
// (it drafts there; both cards hold the same residual) - ~1.3 GB more experts resident a card.
enum class Q4TpEnds { Split, Both };
const char* q4_tp_ends_name(Q4TpEnds e);

// Layers [0, split) on device 0, [split, layers) on device 1 (spec 16b's cut). One device: devices == 1
// and split == layers (Kolibri's KolPlacement, the same rules). Plan 21-QFN-v2b's tensor-parallel form (`tp`): two
// devices, each holding its half of EVERY layer (q4_tp_card's shapes; split == layers), the final mixer and its
// vocabulary half of lm_head on both, the embedding and the head as `ends` places them.
struct Q4Placement {
  uint32_t devices = 1, split = 0, layers = 0;
  bool tp = false;
  Q4TpEnds ends = Q4TpEnds::Split;
  uint32_t first(uint32_t dev) const { return tp || dev == 0 ? 0 : split; }
  uint32_t end(uint32_t dev) const { return tp || dev != 0 ? layers : split; }
  uint32_t device_of(uint32_t layer) const { return !tp && devices > 1 && layer >= split ? 1 : 0; }
  uint32_t count(uint32_t dev) const { return end(dev) - first(dev); }
  bool holds_embed(uint32_t dev) const { return dev == 0 || (tp && ends == Q4TpEnds::Both); }
  bool holds_mtp(uint32_t dev) const { return dev + 1 == devices || (tp && ends == Q4TpEnds::Both); }
  static Q4Placement one(const Qwen4ExpDesc& d);
  static Q4Placement two(const Qwen4ExpDesc& d, uint32_t split);
  static Q4Placement tp2(const Qwen4ExpDesc& d, Q4TpEnds ends);
};
// Throws std::invalid_argument unless devices is 1 (split == layers) or 2 (1 <= split <= layers - 1; tp: split ==
// layers).
void validate(const Q4Placement& p, const Qwen4ExpDesc& d);

// Plan 21-QFN-v2b: card `rank`'s descriptor under tp (Qwen3.8's model::tp_card_desc rule): half the QSA q / kv heads (12 /
// 1), half the GDN k / v heads (8 / 24), half of every expert's and the shared expert's width (320), half the
// vocabulary (124160 rows; vocab_used the rank's real ids), half out_proj's and o_proj's split-K (4: its slices are
// one card's slices 4r .. 4r + 3). Every width derived from these is the card's (qkv||z 8192,
// q||gate||k||v 6656, out_proj / o_proj K 3072, the conv 5120, the KV and GDN state halves); the hyper-connections,
// the router, the indexer, the PLE and the RoPE table keep one card's. The kernels bake one card's shapes; their
// card builds name these.
inline constexpr uint32_t kQ4TpCards = 2;
Qwen4ExpDesc q4_tp_card(const Qwen4ExpDesc& one, uint32_t rank);

struct Qwen4ExpDesc {
  std::string name, architecture, model_type, prefix;   // "qwen3.8-flash-next", "Qwen4ExpForConditionalGeneration",
                                                        // "qwen4_exp", "model.language_model."
  uint32_t layers = 0, hidden = 0, hc = 0, hc_low = 0;  // 48, 2560, 4, 320
  uint32_t vocab = 0, vocab_used = 0, trained_max_len = 0;   // 248320, 248077, 262144
  uint32_t gdn_k_heads = 0, gdn_v_heads = 0, gdn_head = 0, conv_taps = 0;   // 16, 48, 128, 4
  uint32_t q_heads = 0, kv_heads = 0, head_dim = 0, rope_dims = 0;          // 24, 2, 256, 64
  double rope_theta = 0;                                                    // 1e7
  uint32_t idx_heads = 0, idx_dim = 0, idx_budget = 0, idx_compress = 0;    // 4, 128, 2048, 4
  uint32_t experts = 0, top_k = 0, moe_inter = 0, shared_inter = 0;         // 512, 10, 640, 640
  uint32_t ple_layer = 0, ngram = 0, ple_heads = 0, ple_dim = 0, ple_conv_taps = 0;   // 1 (zero-indexed), 3, 16, 160, 4
  uint32_t ple_dilation = 0;                            // 3 (= ngram: M:1208-1225's dilated conv)
  uint64_t ple_base = 0, ple_seed = 0;                  // 20,000,000 (a synthetic: config's value), 1234
  uint32_t ple_eos = 0, ple_pad = 0;                    // 248044, 128 (make_ngram_vocab_size_divisible_by)
  uint32_t mtp_layers = 0;                              // 1
  std::vector<uint32_t> eos;                            // {248046, 248044} (generation_config.json)
  float rms_eps = 0;
  Q4Forms forms;
  // Not the checkpoint's: the hyper-connection mixers' device form, chosen at load (B70_Q4_HC, loader::q4_hc_int8;
  // spec 21-QFN-v2 §7 lever 5, PERF-66). Both checkpoints ship them bf16; true = int8 with one fp16 scale per 32 k
  // (loader::Q4HcOffsets).
  bool hc_int8 = false;
  // Not the checkpoint's: the QSA layers' KV form, chosen at load (`--kv-cache int8`, plan 21-QFN-v3a): false = bf16
  // rows; true = kv8_scheme.h's rotated int8 codes with one fp16 scale a position and kv head (the MTP head's too).
  bool kv8 = false;
  // int4 split-K (gemv.cl S, layout 0), PROVISIONAL until 21-QFN-v1c's sweep - copied from Qwen3.8's measured map
  // at the nearest shapes (q||k||v S2); out_proj / o_proj S8 (PERF-109: a tp card's half at S4 launches 640 sub-groups
  // instead of 320, and a tp slice stays one card's). qkv||z is S 1, fixed: gdn_step and prep_gated_head read S 1
  // (prep.cl:118-123).
  uint32_t gdn_out_s = 0, qkvg_s = 0, o_s = 0;

  bool is_qsa(uint32_t l) const { return l % 4 == 3; }
  uint32_t qsa_before(uint32_t l) const;               // QSA layers in [0, l)
  uint32_t gdn_before(uint32_t l) const;               // GDN layers in [0, l)
  uint32_t hc_n() const { return hc * hidden; }                        // 10240
  uint32_t hc_down_n() const { return hc_low + hc; }                   // 324 (down's 320 + block_inject's 4)
  uint32_t hc_down_rows() const { return (hc_down_n() + 15) / 16 * 16; }   // 336: gemv_bf16's 16-column tiles
  uint32_t conv_rows() const;                                          // 10240 = (16 + 16 + 48) x 128
  uint32_t gdn_z_n() const { return gdn_v_heads * gdn_head; }          // 6144
  uint32_t qkvz_n() const;                                             // 16384: qkv 10240 | z 6144
  uint32_t ab_n() const { return 2 * gdn_v_heads; }                    // 96, padded to 128 on the device
  static constexpr uint32_t kAbPaddedN = 128;
  uint32_t q_n() const { return q_heads * head_dim; }                  // 6144
  uint32_t kv_n() const { return kv_heads * head_dim; }                // 512
  uint32_t qkvg_n() const { return 2 * q_n() + 2 * kv_n(); }           // 13312: q||gate per head | k | v
  uint32_t idx_n() const { return (idx_heads + 1) * idx_dim; }         // 640: 4 query heads + 1 raw key
  uint32_t gqa() const { return q_heads / kv_heads; }                  // 12
  uint32_t block_topk() const { return idx_budget / idx_compress; }    // 512
  uint32_t max_visible() const { return idx_budget + idx_compress - 1; }   // 2051
  uint32_t router_n() const;                                           // 528: 512 + the shared gate's row, to 16
  static constexpr uint32_t kRouteLanes = 256;
  uint32_t ple_e() const { return ple_heads * ple_dim; }               // 2560
  uint32_t ple_kv_n() const { return hc_n() + hidden; }                // 12800: key 10240 | value 2560
  uint32_t ple_ring() const { return (ple_conv_taps - 1) * ple_dilation; }   // 9
  static std::string layer_prefix(uint32_t l);                         // "model.language_model.layers.<l>."
  Q4Linear linear(Q4LinearId id) const;                                // throws on kCount
  // fp32 [max_len][2][rope_dims / 2] (loader::q4_rope_table, attn.cl's form): 256 B a position.
  size_t rope_table_bytes(uint32_t max_len) const { return size_t(max_len) * 2 * (rope_dims / 2) * 4; }
};

// The published model, forms = ours (int4 g64 everywhere AutoRound quantises). A process-lifetime singleton.
const Qwen4ExpDesc& qwen4exp();
// config.json -> the descriptor: num_hidden_layers / layer_types (a prefix of the real pattern, 2..48),
// ngram_vocab_size_base and seed read; the forms the loader found in the tensor names; everything else held
// to qwen4exp() (check_qwen4exp_config). The text config is `text_config`, or the top level of a text-only
// export (model_type qwen4_exp_text). Throws naming the key.
Qwen4ExpDesc qwen4exp_desc(const common::json::Value& config, const Q4Forms& forms);
// Throws std::runtime_error naming the key when config.json describes a qwen4_exp the engine is not
// written for: every structural key of `d` (layers, layer_types, widths, heads, the indexer, the experts,
// the PLE, HC, RoPE, the gate, norm_topk_prob, untied, no bias). Host only.
void check_qwen4exp_config(const Qwen4ExpDesc& d, const common::json::Value& config);
// "qwen4_exp" (the checkpoints) or "qwen4_exp_text" (a text-only export: the tiny model's spelling).
bool is_qwen4exp_model_type(const std::string& model_type);

}  // namespace model
