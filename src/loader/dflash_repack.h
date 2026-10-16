#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "loader/dflash_layout.h"
#include "loader/safetensors.h"
#include "model/dflash.h"
#include "model/qwen35.h"

// Spec 19b Task 1: the host half of the DFlash drafter loader - the checkpoint opened and held to its config and
// its names, and every device-form piece built on demand, with NO Level Zero (tests/loader/dflash_loader_test.cc
// runs it on any host). loader::load_drafter (dflash_loader.h) uploads each piece as it is built, so the host holds
// one linear at a time beside the mmapped shards.
//
// **Names both ways.** The checkpoint's tensors (after dflash_ref.py's `_canon`: a leading `model.`, `midlayer.` ->
// `layers.0.`, `encoder.output_norm_enc.` -> `hidden_norm.`, `encoder.fc.` -> `fc.`) must be exactly
// dflash_expected_names(desc): a missing one and an unknown one both throw naming it. Refused by name besides:
// `embed_tokens.weight` / `lm_head.weight` (Review Focus 3: the drafter never owns a copy - it shares the target's),
// `d2t` / `t2d` (a draft vocabulary of its own), `mask_hidden`, and every tensor that is not bf16 or has another
// shape than the descriptor gives it.
//
// **The mask embedding** (Review Focus 4): the target's embedding row `mask_id` (the device loader copies it), or -
// when the checkpoint ships one - `mask_embedding.safetensors` (one tensor `embedding`, [hidden] or [1, hidden], bf16
// or fp32; optional metadata `mask_token_id`, which must equal the config's): it replaces the row. vLLM's
// `mask_embedding.pt` is a torch pickle and is refused naming the converter (tools/oracle/dflash_mask_embedding.py).
namespace loader {

// Every tensor name of a drafter of `d` (canonical spelling), sorted.
std::vector<std::string> dflash_expected_names(const model::DFlashDesc& d);
// dflash_ref.py's `_canon`.
std::string dflash_canon(const std::string& name);

// One linear in its device form: Bf16 - gemv_bf16's tiles [N/16][K/8][8][16] as uint16; Int8 - gemv_i8w's tiles
// [N/16][K/16][16][16] as int8 + fp32 row scales [N].
struct DFlashHostLinear {
  model::WeightKind kind = model::WeightKind::Bf16;
  uint32_t K = 0, N = 0;
  std::vector<uint16_t> bf16;   // Bf16
  std::vector<int8_t> i8;       // Int8
  std::vector<float> scales;    // Int8
  size_t bytes() const { return bf16.size() * 2 + i8.size() + scales.size() * 4; }
  model::GemvShape shape() const { return {K, N, 1, 0}; }
};

enum class DFlashLinear { Qkv, O, GateUp, Down, AttnKproj, MlpKproj };

class DFlashCheckpoint {
 public:
  // `snapshot_or_repo`: resolve_snapshot's rules (an HF cache repo id or a directory). Reads config.json
  // (model::dflash_desc_from_config), opens the shards (a single model.safetensors is accepted: every published
  // drafter is one file), checks names / dtypes / shapes against the descriptor (fc's K is the checkpoint's, a
  // multiple of the tap count: the target's hidden is checked by check_target).
  explicit DFlashCheckpoint(const std::string& snapshot_or_repo);
  // A hand-built descriptor (tests): the config's structure is not re-read; names, dtypes and shapes are.
  DFlashCheckpoint(const model::DFlashDesc& d, const std::string& snapshot_dir);

  const model::DFlashDesc& desc() const { return d_; }
  const std::string& dir() const { return dir_; }
  uint32_t fc_k() const { return fc_k_; }                     // taps x the target's hidden, as shipped
  uint32_t target_hidden() const { return fc_k_ / uint32_t(d_.taps.size()); }
  size_t tensors() const { return set_->tensors().size(); }

  // model::check_dflash_target, plus fc's K = taps x target.hidden.
  void check_target(const model::ModelDesc& target, uint32_t target_vocab_rows) const;

  // The device-form pieces (each built from the mmapped bf16 tensors when called). `threads` = 0: all cores; the
  // result does not depend on it.
  DFlashHostLinear fc(DraftFormat f, unsigned threads = 0) const;
  DFlashHostLinear layer_linear(uint32_t layer, DFlashLinear which, DraftFormat f, unsigned threads = 0) const;
  DFlashHostLinear selector_projection(DraftFormat f, unsigned threads = 0) const;   // v2
  std::vector<float> hidden_norm() const;
  std::vector<float> final_norm() const;
  std::vector<float> layer_norms(uint32_t layer) const;      // [input_layernorm | post_attention_layernorm]
  std::vector<float> qk_norms() const;                       // [layers][q_norm | k_norm][head_dim]
  std::vector<float> conv_bases() const;                     // v2: [layers][attn, mlp][side][tap][hidden]
  // v2: the codebook's bf16 rows [vocab][sel_rank], straight from the mmapped shard.
  const uint16_t* codebook(bool successor) const;
  // The shipped mask embedding (fp32 [hidden]), empty when none ships.
  std::vector<float> mask_embedding() const;
  bool ships_mask_embedding() const { return !mask_path_.empty(); }

 private:
  void open_and_check();
  const uint16_t* bf16(const std::string& canonical) const;
  DFlashHostLinear make_linear(const std::vector<std::string>& parts, uint32_t K, DraftFormat f,
                               unsigned threads) const;

  model::DFlashDesc d_;
  std::string dir_;
  std::unique_ptr<SafetensorsSet> set_;
  std::vector<std::pair<std::string, std::string>> canon_;   // canonical -> the checkpoint's name (sorted)
  uint32_t fc_k_ = 0;
  std::string mask_path_;
};

}  // namespace loader
