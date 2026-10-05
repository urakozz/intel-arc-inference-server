#pragma once
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/k2_horizon.h"

// Spec 18b: the host half of the K2-Horizon loader - a K2 checkpoint's tensors repacked
// into the device layouts loader/k2_layout.h sizes, with NO Level Zero, so
// tests/loader/k2_repack_test.cc runs it on any host over a synthetic checkpoint in
// K2's naming. loader::load_k2 (k2_loader.h) uploads what this produces, one layer at a
// time, reusing one K2HostLayer.
//
// **Refusals are by name.** `check_names` holds the checkpoint's tensor list to the one
// the descriptor implies, both ways: a tensor the descriptor does not know (a q_norm, an
// mtp.* head, an extra expert, a quantised lm_head) and a tensor it needs that is absent
// both throw naming it. `.qzeros` / `.g_idx` of an expected int4 linear are allowed and
// never read (assert_quant_invariants proves them the symmetric zero point and the
// identity); every other tensor is read exactly once, and `unconsumed()` must be 0 at the
// end of a load.
//
// **Two packings, one per checkpoint.** If any tensor is a compressed-tensors
// `.weight_packed`, every int4 linear is expected in that form ({weight_packed,
// weight_scale, weight_shape} required, weight_g_idx allowed) and converted by
// LinearSrc::classify - the same exact repack the main loader uses (docs/13,
// "compressed-tensors symmetric checkpoints"); otherwise the GPTQ names above.
namespace loader {

// One non-expert int4 linear as the device holds it: GPTQ layout 0.
struct K2HostLinear {
  std::vector<uint32_t> words;    // [K/8][N]
  std::vector<uint16_t> scales;   // [K/64][N] f16
};

// One layer's device bytes. The expert vectors and the router / route block are empty on
// a dense layer.
struct K2HostLayer {
  std::vector<K2HostLinear> linears;   // in K2Desc::layer_linears(layer) order
  std::vector<uint32_t> value;         // layout-1 blocks [value_experts]
  std::vector<uint32_t> gate_up;       // layout-1 blocks [experts + 1], shared last
  std::vector<uint32_t> down;          // layout-1 blocks [experts + 1], shared last
  std::vector<uint16_t> router;        // bf16 tiled [router_n][hidden], zero rows past experts
  std::vector<float> norms;            // fp32 plain w: input [hidden] || post [hidden]
  std::vector<float> route;            // fp32: MoE bias [router_n] || MoVA bias [value_experts]
  size_t src_int4_bytes = 0, src_bf16_bytes = 0;   // checkpoint bytes read, for the report
};

class K2Checkpoint {
 public:
  // `set` must outlive this object (its mmaps are the repack's source).
  K2Checkpoint(const model::K2Desc& d, const SafetensorsSet& set);

  // Throws std::runtime_error naming up to five unexpected tensors, else the first
  // missing one. Call before any repack.
  void check_names() const;
  // The expected tensor names (full checkpoint names); `required` = must be present
  // (everything but .qzeros / .g_idx, or .weight_g_idx for compressed-tensors).
  std::set<std::string> expected_names(bool required_only) const;

  void repack_layer(uint32_t layer, K2HostLayer& out);
  // Top level. embed_tokens is uploaded verbatim (bf16 [vocab][hidden], gathered).
  const uint16_t* embed();                        // marks it consumed
  std::vector<float> final_norm();                // fp32 plain w [hidden]
  const uint16_t* lm_head();                      // bf16 [vocab][hidden] row-major

  // Tensors nothing read (qzeros / g_idx excepted), with up to five names in `names`.
  // compressed-tensors: the checkpoint's int4 linears are `.weight_packed` (constructor).
  bool compressed_tensors() const { return ct_; }
  size_t unconsumed(std::string* names = nullptr) const;
  const SafetensorsSet& set() const { return set_; }
  const model::K2Desc& desc() const { return d_; }

 private:
  LinearSrc int4(const std::string& prefix, uint32_t K, uint32_t N);
  TensorInfo take(const std::string& name, const char* dtype, uint64_t elems);
  void widen(const TensorInfo& t, const std::string& name, float* dst) const;

  const model::K2Desc& d_;
  const SafetensorsSet& set_;
  std::set<std::string> consumed_;
  bool ct_ = false;
};

// K2's RoPE table: fp32 [max_len][2][head_dim / 2], cos at [p][0][i], sin at [p][1][i] -
// built with the reference's fp32 steps and rounded to bf16, because
// K2HorizonRotaryEmbedding (modeling_k2_horizon.py:782-800) computes in fp32 and casts
// cos / sin to the hidden dtype before apply_rotary_pos_emb multiplies:
//   inv[i] = 1 / powf(theta, float(2i) / head_dim)       fp32 pow, fp32 divide
//   ang    = float(p) * inv[i]                           one fp32 product
//   cos, sin of that fp32 angle, rounded to bf16
// cat(freqs, freqs) gives dims i and i + head_dim/2 the same angle, so half a head of
// angles per position suffices. The 27B loader computes its angles in double; at 32k
// positions an fp32 angle is ~1e-3 rad away from the double one, so K2 mirrors fp32.
std::vector<float> k2_rope_table(const model::K2Desc& d, uint32_t max_len);

// The checks load_k2 runs on config.json before any tensor is read: model::check_k2_config
// (the structure) and QuantConfig::parse (int4 g64 sym, desc_act false). Returns the
// parsed quantisation config.
QuantConfig check_k2_checkpoint_config(const model::K2Desc& d, const common::json::Value& config);

}  // namespace loader
