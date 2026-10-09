#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "loader/qwen4exp_layout.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/qwen4exp.h"

// Spec 21b: the host half of the Qwen3.8-Flash-Next loader - a checkpoint's tensors repacked into the device
// layouts loader/qwen4exp_layout.h sizes, with NO Level Zero, so tests/loader/qwen4exp_repack_test.cc runs it
// on any host over a synthetic checkpoint. loader::load_qwen4exp (qwen4exp_loader.h) uploads what this
// produces, one layer at a time, straight onto the layer's device.
//
// **Two checkpoint forms, one weight format** (spec 21 §5, the operator's rule): AutoRound W4A16 int4 g64
// symmetric in the auto_round:auto_gptq packing (`.qweight` I32 [K/8, N], `.scales` F16 [K/64, N],
// `.qzeros` 0x77777777 - proved by assert_quant_invariants, never read - no g_idx), with ONE exact
// conversion: Intel's export ships its routed experts at g128, which LinearSrc::classify expands to g64
// (groups 2j and 2j + 1 both take g128 group j's scale - every weight keeps its own scale, so the
// dequantised words equal (q - 8) x scale of the source bit for bit). Everything AutoRound leaves bf16 stays
// bf16 (`.weight`). The forms are read from the names, all-or-nothing per group (q4_forms).
//
// **The MTP head's bf16 experts** (both checkpoints ship `mtp.*` bf16) are quantised at load to int4 g64 by
// loader::rtn_int4_g64 - spec 15e's precedent for Ornith's head: the head only drafts, the verify decides
// every token, so it moves acceptance, never output. The interim until decision 6.
//
// **Refusals are by name.** check_names holds the checkpoint's tensor list to q4_expected_names both ways;
// the skipped-by-design groups are named, never guessed: `model.visual.*` (the tower), the PLE table's bf16
// shards (`...ple.ple_embedding.ngram_embedding.shard_K.weight` - the engine reads the int8 file instead,
// loader/qwen4exp_ple.h), `mtp.*` when the head is not loaded, and with --layers N the later layers.
namespace loader {

// The forms from the names (model::Q4Forms): the dense group from layer 0's linear_attn.in_proj_qkv, the
// shared group from layer 0's mlp.shared_expert.gate_proj, the MTP head's experts from expert 0's gate_proj
// (Bf16 when the checkpoint has no head), the expert group from layer 0 expert 0's scales rows. Every member
// of every group in every layer of `d` must agree, else throws naming the first tensor that does not. A bf16
// or fused routed expert in the main model is refused by name (the bf16 original is the reference's input,
// never the engine's).
model::Q4Forms q4_forms(const SafetensorsSet& st, const model::Qwen4ExpDesc& d);

// Every text-model tensor the checkpoint of `d` in forms `f` ships, by name (21a's qwen4exp_ref.expected_names
// with the int4 linears' `.weight` replaced by `.qweight` / `.scales` / `.qzeros`), WITHOUT the PLE table's
// shards (skipped by design: their count is the export's split, not the model's). `mtp` adds the head.
std::vector<std::string> q4_expected_names(const model::Qwen4ExpDesc& d, const model::Q4Forms& f, bool mtp);

// The RoPE table fp32 [max_len][2][rope_dims / 2], cos at [p][0][i], sin at [p][1][i], each a bf16 value -
// the reference's fp32 steps (kol_rope_table's construction at theta 1e7 over the 64 rotary dims):
//   inv[i] = 1 / pow(theta, float(2i) / 64); ang = float(p) x inv[i]; bf16(cos / sin(ang))
std::vector<float> q4_rope_table(const model::Qwen4ExpDesc& d, uint32_t max_len);

// One GEMV weight in its device form: int4 layout 0 (words + f16 scales) or bf16 tiles.
struct Q4HostWeight {
  model::GemvShape shape{};
  model::WeightKind kind = model::WeightKind::Bf16;
  std::vector<uint32_t> words;    // int4: [K/8][N]
  std::vector<uint16_t> scales;   // int4: f16 [K/64][N]
  std::vector<uint16_t> tiles;    // bf16: gemv_bf16 tiles
  size_t bytes() const { return words.size() * 4 + scales.size() * 2 + tiles.size() * 2; }
  void clear() { words.clear(); scales.clear(); tiles.clear(); }
};

// One layer's device bytes (qwen4exp_layout.h's sizes); the fields of the other layer kind stay empty.
struct Q4HostLayer {
  std::vector<uint8_t> hc_attn, hc_mlp;           // Q4HcOffsets blocks
  Q4HostWeight gdn_qkvz, gdn_ab, gdn_out;         // GDN layers
  std::vector<uint8_t> gdn_small;                 // make_small_layout's block
  Q4HostWeight qsa_qkvg, qsa_idx, qsa_o;          // QSA layers
  std::vector<uint8_t> qsa_small;                 // Q4QsaSmall
  Q4HostWeight router;                            // bf16 tiles {hidden, router_n}
  std::vector<uint32_t> gate_up, down;            // layout-1 blocks, block e at q4_gate_up_offset / q4_down_offset
  std::vector<uint8_t> shared;                    // gate||up then down (int4 layout-1 blocks or bf16 tiles)
  size_t shared_gate_up = 0;                      // where down starts in `shared` (q4_shared_gate_up_bytes)
  std::vector<uint8_t> ple;                       // the PLE layer only (Q4PleOffsets)
  size_t src_int4_bytes = 0, src_bf16_bytes = 0;  // checkpoint bytes read
  // Each field's bytes, in Q4LayerBytes' terms (the loader and the test hold them to q4_layer_bytes).
  Q4LayerBytes bytes() const;
};

// The PLE layer's three I64 tensors as the checkpoint ships them.
struct Q4PleConstants {
  std::array<uint64_t, 3> multipliers{};
  std::vector<uint64_t> sizes, offsets;
};

class Q4Checkpoint {
 public:
  // `set` must outlive this object (its mmaps are the repack's source). `d` is the checkpoint's descriptor
  // (its layers: the checkpoint's count; its forms: q4_forms').
  Q4Checkpoint(const model::Qwen4ExpDesc& d, const SafetensorsSet& set);

  // Throws naming up to five unexpected tensors, else the first missing one. Before any repack.
  void check_names(bool mtp) const;
  void repack_layer(uint32_t layer, Q4HostLayer& out);
  void repack_mtp(Q4HostLayer& out);                 // bf16 experts -> rtn_int4_g64 (spec 15e's rule)
  const uint16_t* embed();                           // bf16 [vocab][hidden], marks it consumed
  const uint16_t* lm_head();                         // bf16 [vocab][hidden] row-major
  std::vector<uint8_t> final_mixer();                // Q4HcOffsets(inject = false)
  std::vector<uint8_t> mtp_fc();                     // Q4MtpFcOffsets
  std::vector<uint8_t> mtp_mixer();                  // the head's own final mixer
  Q4PleConstants ple_constants();                    // the PLE layer's I64 tensors, consumed
  size_t skip_layers_from(uint32_t from);            // --layers N: every tensor of layers >= from
  size_t skip_prefix(const std::string& p);          // model.visual., mtp., the PLE shards
  size_t skip_ple_shards();                          // the PLE layer's bf16 shards (the int8 file replaces them)
  // Tensors nothing read (qzeros / g_idx excepted), up to five names in `names`.
  size_t unconsumed(std::string* names = nullptr) const;
  const model::Qwen4ExpDesc& desc() const { return d_; }

 private:
  LinearSrc int4(const std::string& prefix, uint32_t K, uint32_t N, uint32_t group);
  LinearSrc bf16(const std::string& prefix, uint32_t K, uint32_t N);
  TensorInfo take(const std::string& name, uint64_t elems, const char* dtype = "BF16");
  void widen(const TensorInfo& t, float* dst) const;
  void one_plus_w(const std::string& name, uint32_t n, uint8_t* dst);
  void dense(const model::Qwen4ExpDesc& d, const std::string& lp, model::Q4LinearId id, Q4HostWeight& out,
             Q4HostLayer& L);
  void hc(const std::string& base, bool inject, std::vector<uint8_t>& out, Q4HostLayer& L);
  void moe(const model::Qwen4ExpDesc& d, const std::string& lp, bool mtp, Q4HostLayer& L);
  void qsa(const model::Qwen4ExpDesc& d, const std::string& lp, Q4HostLayer& L);

  model::Qwen4ExpDesc d_;
  const SafetensorsSet& set_;
  std::set<std::string> consumed_;
};

// The checks load_qwen4exp runs on config.json before any tensor is read: QuantConfig::parse.
QuantConfig check_qwen4exp_checkpoint_config(const common::json::Value& config);

}  // namespace loader
