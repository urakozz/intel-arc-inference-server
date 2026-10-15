#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/json.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/muse_glimmer.h"

// Spec 23b: the host half of the Muse-Glimmer loader - a checkpoint's tensors repacked into the device layouts
// loader/muse_layout.h sizes, with NO Level Zero, so tests/loader/muse_repack_test.cc and muse_synth_host_test.cc
// run it on any host. loader::load_muse_glimmer (muse_loader.h) uploads what this produces, one layer at a time,
// straight onto the layer's device.
//
// **One weight format** (spec 23 §3): int4 g64 symmetric GPTQ - `.qweight` I32 [K/8, N], `.scales` F16 [K/64, N],
// `.qzeros` I32 all 0x77777777, `.g_idx` I32 i // 64 when shipped - in EITHER Vishva007 spelling of the one export
// (23a's facts sheet): `-AutoRound` (quant_method auto-round, packing_format auto_round:auto_gptq, no g_idx: 1459
// text names) or `-AutoRound-GPTQ` (quant_method gptq, g_idx shipped: 1875 text names). Every linear goes through
// LinearSrc::classify (g128 would be expanded exactly; no other form exists here); the norms, embed_tokens,
// lm_head and the final norm are bf16. QuantConfig::parse refuses another arithmetic by name (desc_act, a group
// other than 64 / 128, bits other than 4, asymmetric); assert_quant_invariants proves every qzeros word and every
// g_idx (docs/13's three data asserts) before a byte is repacked. compressed-tensors is refused.
//
// **Refusals are by name.** The checkpoint's names are held to muse_expected_names both ways: an unknown text
// tensor and a missing one both throw naming it. The vision tower / adapter / projection (809 tensors on the real
// checkpoint, `model.vision_{tower,adapter,projection}.*`) are skipped BY NAME and counted; any other non-text
// name throws. Every text tensor is read exactly once; unconsumed() must be empty at the end.
namespace loader {

// The two spellings of the int4 export (the names differ by `.g_idx` alone).
enum class MuseQuantForm { AutoRound, Gptq };
const char* muse_quant_form_name(MuseQuantForm f);   // "autoround" | "gptq" (muse_ref.expected_names' forms)

// Every text tensor of a checkpoint of `d` in form `f` - muse_ref.expected_names(tc, form): the embedding, the
// final norm, lm_head, and per layer the four norms and the eight linears' `.qweight` / `.scales` / `.qzeros`
// (+ `.g_idx` in the gptq form). Sorted.
std::vector<std::string> muse_expected_names(const model::MuseGlimmerDesc& d, MuseQuantForm f);
// muse_ref.is_text: under `model.language_model.` or `lm_head.weight`.
bool muse_is_text(const std::string& name);
// The vision side, skipped by name: `model.vision_tower.` / `model.vision_adapter.` / `model.vision_projection.`.
bool muse_is_vision(const std::string& name);

// One fused linear in its row's layout: layout 0 = words [K/8][N] + scales f16 [K/64][N]; layout 1 = tiles in
// `words`, `scales` empty.
struct MuseHostLinear {
  std::vector<uint32_t> words;
  std::vector<uint16_t> scales;
  size_t bytes() const { return words.size() * 4 + scales.size() * 2; }
};

// One layer's device-form bytes, host side (the tests read them).
struct MuseHostLayer {
  MuseHostLinear qgkv, oproj, gate_up, down;               // columns fused in the descriptor's order
  std::vector<float> in_norm, post_attn, pre_ff, post_ff;  // fp32 widenings of the bf16 w (bit-exact)
  size_t src_bytes = 0;                                    // checkpoint bytes read (qweight + scales + norms)
  // The device's norm block: in | post_attn | pre_ff | post_ff (MuseGlimmerDesc::norm_off_*).
  std::vector<float> norm_block() const;
  size_t bytes() const;                                    // = muse_layer_bytes(d).total()
};

class MuseCheckpoint {
 public:
  // The served path: config.json held to model::check_muse_config (the checkpoint's own layer count), the
  // quantisation config parsed, the shards opened, the data asserts, every name accounted for both ways.
  // `layers_override` > 0 keeps layers [0, layers_override) (development mode): the later layers' tensors are
  // consumed by design (skipped_layer_tensors()).
  explicit MuseCheckpoint(const std::string& snapshot, uint32_t layers_override = 0);
  // A hand-built descriptor (tests: the tiny model's widths, which check_muse_config refuses by design): the
  // config's structure is not checked; the quantisation, the names, the data asserts and the shapes are, against
  // `d` (d.layers = the checkpoint's layer count).
  MuseCheckpoint(const model::MuseGlimmerDesc& d, const std::string& snapshot);

  const model::MuseGlimmerDesc& desc() const { return d_; }
  const QuantConfig& quant() const { return qc_; }
  MuseQuantForm form() const { return form_; }
  uint32_t checkpoint_layers() const { return ckpt_layers_; }
  size_t subnormal_scales() const { return scan_.subnormal_scales; }
  size_t tensors() const { return set_->tensors().size(); }
  size_t text_tensors() const { return text_; }
  size_t vision_skipped() const { return vision_; }        // 809 on the real checkpoint
  size_t skipped_layer_tensors() const { return skipped_; }
  const std::string& snapshot() const { return snap_; }

  MuseHostLayer layer(uint32_t l) const;
  void layer(uint32_t l, MuseHostLayer& out) const;        // reuses out's buffers
  const uint16_t* embed() const;                           // bf16 [vocab][hidden] row-major (the mmap)
  const uint16_t* lm_head() const;                         // bf16 [vocab][hidden] row-major (the mmap)
  std::vector<float> final_norm() const;                   // fp32 plain w [hidden]
  // Text tensors nothing read (vision excepted: skipped by name). The load throws unless empty.
  std::vector<std::string> unconsumed() const;

 private:
  void open(const common::json::Value& cfg);
  void check_names();
  void linear(const std::string& lp, model::MuseLinearId id, MuseHostLinear& out, size_t& src) const;
  const uint16_t* bf16(const std::string& name, uint64_t elems) const;

  std::string snap_;
  model::MuseGlimmerDesc d_;
  uint32_t ckpt_layers_ = 0;
  QuantConfig qc_;
  QuantScan scan_;
  MuseQuantForm form_ = MuseQuantForm::AutoRound;
  std::unique_ptr<SafetensorsSet> set_;
  size_t text_ = 0, vision_ = 0, skipped_ = 0;
  mutable std::set<std::string> consumed_;
};

}  // namespace loader
