#pragma once
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/kolibri1.h"

// Spec 20c: the host half of the Kolibri-1 loader - a checkpoint's tensors repacked into the
// device layouts loader/kolibri1_layout.h sizes, with NO Level Zero, so tests/loader/
// kolibri1_repack_test.cc runs it on any host over a synthetic checkpoint in Kolibri's naming.
// loader::load_kolibri1 (kolibri1_loader.h) uploads what this produces, one layer at a time,
// straight onto the layer's device (no load-then-place: the model is ~42 GB).
//
// **One weight format** (spec 20 §3.1): AutoRound int4 g64 symmetric in the auto_round:auto_gptq
// packing - `.qweight` I32 [K/8, N], `.scales` F16 [K/64, N], `.qzeros` (0x77777777, proved by
// assert_quant_invariants, never read), no `.g_idx` (allowed when the identity) - for every routed
// expert and, in decision 2's int4 arm, the attention projections; bf16 `.weight` for everything
// else. No compressed-tensors names, no asymmetric zero points.
//
// **Refusals are by name.** check_names holds the checkpoint's tensor list to kol_expected_names
// both ways: a tensor the descriptor does not know and one it needs that is absent both throw
// naming it. Every expected tensor is read exactly once; unconsumed() must be 0 at the end.
namespace loader {

// The attention arm from the tensor names: layer 0's `self_attn.q_proj.qweight` (Int4) or
// `.weight` (Bf16); every q/k/v/o projection of every layer of `d` must agree, else throws
// naming the first tensor that does not (spec 20c Task 3).
model::KolAttnForm kol_attn_form(const SafetensorsSet& st, const model::Kolibri1Desc& d);

// Every tensor the AutoRound export of `d` in arm `a` ships, by name - kolibri_ref.expected_names
// with the int4 linears' `.weight` replaced by `.qweight` / `.scales` / `.qzeros`.
std::vector<std::string> kol_expected_names(const model::Kolibri1Desc& d, model::KolAttnForm a);

// RoPE table fp32 [max_len][2][head_dim / 2], cos at [p][0][i], sin at [p][1][i], each a bf16
// value - kolibri_ref.rope_cos_sin's fp32 steps (k2_rope_table's construction, theta 1e4):
//   inv[i] = 1 / powf(theta, float(2i) / 128); ang = float(p) x inv[i]; bf16(cos / sin(ang))
std::vector<float> kol_rope_table(const model::Kolibri1Desc& d, uint32_t max_len);

// One layer's device bytes (kolibri1_layout.h's sizes).
struct KolHostLayer {
  std::vector<uint32_t> qkv_words, oproj_words;    // int4 arm: layout 0 words
  std::vector<uint16_t> qkv_scales, oproj_scales;  // int4 arm: f16 scales
  std::vector<uint16_t> qkv_bf16, oproj_bf16;      // bf16 arm: gemv_bf16 tiles
  std::vector<float> norms;                        // fp32 Kolibri1Desc::norm_floats(), plain w
  std::vector<uint16_t> router;                    // bf16 tiled {hidden, router_n}, rows >= experts zero
  std::vector<float> bias;                         // fp32 [router_n], 0 past the experts
  std::vector<uint32_t> gate_up, down;             // layout-1 blocks, block e at e x block
  std::vector<uint16_t> shared_gate_up, shared_down;   // bf16 tiled
  size_t src_int4_bytes = 0, src_bf16_bytes = 0;   // checkpoint bytes read
};

class KolCheckpoint {
 public:
  // `set` must outlive this object (its mmaps are the repack's source). `d` is the checkpoint's
  // descriptor (its layers: the checkpoint's count); `d.attn` the arm kol_attn_form found.
  KolCheckpoint(const model::Kolibri1Desc& d, const SafetensorsSet& set);

  // Throws naming up to five unexpected tensors, else the first missing one. Before any repack.
  void check_names() const;
  void repack_layer(uint32_t layer, KolHostLayer& out);
  const uint16_t* embed();               // bf16 [vocab][hidden], marks it consumed
  std::vector<float> final_norm();       // fp32 plain w [hidden]
  const uint16_t* lm_head();             // bf16 [vocab][hidden] row-major
  // Marks every tensor of layers >= `from` consumed (layers_limit, development mode).
  size_t skip_layers_from(uint32_t from);
  // Tensors nothing read (qzeros / g_idx excepted), up to five names in `names`.
  size_t unconsumed(std::string* names = nullptr) const;
  const model::Kolibri1Desc& desc() const { return d_; }

 private:
  LinearSrc int4(const std::string& prefix, uint32_t K, uint32_t N);
  LinearSrc bf16(const std::string& prefix, uint32_t K, uint32_t N);
  TensorInfo take(const std::string& name, uint64_t elems);
  void widen(const TensorInfo& t, float* dst) const;
  void attn_linear(const std::string& lp, model::KolLinearId id, std::vector<uint32_t>& words,
                   std::vector<uint16_t>& scales, std::vector<uint16_t>& tiles, KolHostLayer& out);

  model::Kolibri1Desc d_;
  const SafetensorsSet& set_;
  std::set<std::string> consumed_;
};

// The checks load_kolibri1 runs on config.json before any tensor is read: model::kolibri1_desc
// (the structure, at the checkpoint's layer count) and QuantConfig::parse.
QuantConfig check_kolibri1_checkpoint_config(const common::json::Value& config);

}  // namespace loader
