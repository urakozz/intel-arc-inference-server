#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/json.h"
#include "model/qwen35.h"   // GemvShape, WeightKind, Fuse - the shared vocabulary, nothing else

// Spec 20c: Aleph Alpha's Kolibri-1 (`Kolibri1ForCausalLM`, config.json model_type `kolibri1`)
// as data - a third model BESIDE qwen3_5 and K2-Horizon (spec 20 §4: its own table, loader,
// kernels and engine, K2's pattern). Every number of `kolibri1()` is the checkpoint's
// (Aleph-Alpha/Kolibri-1-BF16 config.json, index and shard headers; docs/probe-kolibri-2026-10-05.md):
//
//   layers 50, every one MoE; `layer_types` full attention at 4, 9, ..., 49 (every 5th), the other
//          40 sliding with a 513-key window INCLUDING the query (key j visible to query i iff
//          i - 513 < j <= i)
//   hidden 2560, RMSNorm eps 1e-6, PLAIN w (`w * x_hat`, no 1 + w), sandwich norms:
//          x += post_attn_norm(attn(input_layernorm(x))); x += post_ffn_norm(moe(post_attention_layernorm(x)))
//   attention 48 q / 4 kv heads x 128 (GQA 12), q/k RMSNorm per head; RoPE (neox rotate_half,
//          theta 1e4, all 128 dims) in the sliding layers only, the full layers NoPE; no bias, no gate
//   MoE    384 routed experts top 6 (intermediate 512, SiLU) + one UNGATED shared expert (512);
//          router mlp.gate bf16 [384][2560] with fp32 logits, selection on logit + expert_bias,
//          weights sigmoid(logit) NOT renormalised, ties to the lower id; the combine in ascending
//          id in fp32, + the shared expert, one rounding
//   vocab  128000, untied embed_tokens / lm_head bf16; head_dtype float32 (fp32 logits); EOS
//          {127906, 127901}; no BOS; trained context 262144
//
// The descriptor is a struct of plain fields so a test can build a small one (an expert count
// of 8, tests/loader/kolibri1_repack_test.cc); every width the loader, the buffers, the planner
// and the capture use is DERIVED here. A synthetic checkpoint (tools/quantize/kolibri/
// make_synth.py) or `--layers N` holds the first `layers` of the real pattern.
namespace model {

// Spec 20 decision 2 (open): the checkpoint's attention arm. The loader reads it from the
// tensor names (q_proj.qweight: Int4; q_proj.weight: Bf16) - both arms are built and gated.
enum class KolAttnForm { Int4, Bf16 };
const char* kol_attn_form_name(KolAttnForm a);

// Kolibri's linears that are not routed experts (the router, the shared expert and the experts
// are the loader's blocks, not GEMV rows).
enum class KolLinearId { Qkv, OProj, LmHead, kCount };
struct KolLinear {
  KolLinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // layer-relative checkpoint prefixes, in column order
};

struct Kolibri1Desc;

// Layers [0, split) on device 0, [split, layers) on device 1 (spec 16b's cut). One device:
// devices == 1 and split == layers. `layers` is the descriptor's (the plan's two-field form
// {devices, split} cannot say where device 1 ends, so the count travels with the split).
struct KolPlacement {
  uint32_t devices = 1, split = 0, layers = 0;
  uint32_t first(uint32_t dev) const { return dev == 0 ? 0 : split; }
  uint32_t end(uint32_t dev) const { return dev == 0 ? split : layers; }
  uint32_t device_of(uint32_t layer) const { return devices > 1 && layer >= split ? 1 : 0; }
  uint32_t count(uint32_t dev) const { return end(dev) - first(dev); }
  static KolPlacement one(const Kolibri1Desc& d);
  static KolPlacement two(const Kolibri1Desc& d, uint32_t split);
};
// Throws std::invalid_argument unless devices is 1 (split == layers) or 2 (1 <= split <=
// layers - 1). Every user of a placement calls it.
void validate(const KolPlacement& p, const Kolibri1Desc& d);

struct Kolibri1Desc {
  std::string name, architecture, model_type;     // "kolibri-1", "Kolibri1ForCausalLM", "kolibri1"
  uint32_t layers = 0;                             // 50; a synthetic checkpoint: 1..50
  uint32_t hidden = 0, q_heads = 0, kv_heads = 0, head_dim = 0;
  uint32_t experts = 0, top_k = 0, moe_inter = 0, shared_inter = 0;
  uint32_t vocab = 0, vocab_used = 0;              // 128000, 128000 (argmax over every row)
  uint32_t window = 0, full_every = 0;             // 513, 5
  uint32_t trained_max_len = 0;                    // 262144
  std::vector<uint32_t> eos;                       // {127906, 127901}
  double rope_theta = 0;
  float rms_eps = 0;
  KolAttnForm attn = KolAttnForm::Int4;
  // The int4 arm's split-K (gemv.cl S; layout 0). PROVISIONAL - q||k||v S2 and o_proj S4 are
  // copied from K2's measured-nearest cells (spec 18b) until Task 8's sweep.
  uint32_t qkv_s = 0, oproj_s = 0;

  bool is_sliding(uint32_t l) const { return l % full_every != full_every - 1; }
  uint32_t full_before(uint32_t l) const;          // full layers in [0, l)
  uint32_t sliding_before(uint32_t l) const;       // sliding layers in [0, l)
  uint32_t q_n() const { return q_heads * head_dim; }        // 6144
  uint32_t kv_n() const { return kv_heads * head_dim; }      // 512
  uint32_t qkv_n() const { return q_n() + 2 * kv_n(); }      // 7168: q | k | v
  uint32_t gqa() const { return q_heads / kv_heads; }        // 12
  // The router's rows on the device: the next power of two >= experts (gemv_bf16's 16-column
  // tiles), rows >= experts zero; the route never reads them (sel -INF).
  uint32_t router_n() const;                       // 512
  static constexpr uint32_t kRouteLanes = 256;     // the route work-group (the Mac's cap too)
  uint32_t route_epl() const { return router_n() / kRouteLanes; }   // 2 experts per lane
  // Sliding-ring slots per layer: a power of two >= a prefill chunk (2048, spec 20d) + window - 1,
  // so a chunk's own keys and the window before it never collide.
  static constexpr uint32_t kRing = 4096;
  KolLinear linear(KolLinearId id) const;          // throws on kCount
  std::vector<KolLinearId> layer_linears() const;  // {Qkv, OProj}
  static std::string layer_prefix(uint32_t l);     // "model.layers.<l>."
  // fp32 [max_len][2][head_dim / 2] (loader::kol_rope_table).
  size_t rope_table_bytes(uint32_t max_len) const { return size_t(max_len) * 2 * (head_dim / 2) * 4; }
  // The per-layer fp32 norm block (loader::KolLayer::norms): input | post_attn | post_attention |
  // post_ffn [hidden] each, then q_norm | k_norm [head_dim] each. Offsets in floats.
  uint32_t norm_off_input() const { return 0; }
  uint32_t norm_off_post_attn() const { return hidden; }
  uint32_t norm_off_post_attention() const { return 2 * hidden; }
  uint32_t norm_off_post_ffn() const { return 3 * hidden; }
  uint32_t norm_off_q() const { return 4 * hidden; }
  uint32_t norm_off_k() const { return 4 * hidden + head_dim; }
  uint32_t norm_floats() const { return 4 * hidden + 2 * head_dim; }
};

// The published model, int4 attention (decision 2's proposal). A process-lifetime singleton.
const Kolibri1Desc& kolibri1();
// config.json -> the descriptor: layers and layer_types read (a prefix of the real pattern), the
// attention form the loader found in the tensor names, everything else held to kolibri1()
// (check_kolibri1_config). Throws naming the key.
Kolibri1Desc kolibri1_desc(const common::json::Value& config, KolAttnForm attn);
// Throws std::runtime_error naming the key when config.json describes a Kolibri-1 the engine is
// not written for: every structural key of `d` (num_hidden_layers == d.layers, widths, heads,
// experts, the window, layer_types = the real pattern's prefix, rope default, theta, silu, no
// renormalisation, untied, vocab, no attention bias). Host only.
void check_kolibri1_config(const Kolibri1Desc& d, const common::json::Value& config);
bool is_kolibri1_model_type(const std::string& model_type);   // "kolibri1"

}  // namespace model
