#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "loader/dflash_repack.h"   // DFlashHostLinear, DFlashLinear
#include "loader/lilicorr_layout.h"
#include "loader/safetensors.h"
#include "model/dflash.h"

// Spec 26b Task 2: the host half of the LiLiCorr drafter loader (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr,
// model::DFlashForm::LiLiCorr) - the checkpoint opened and held to its config and names, every device-form piece built on
// demand from the decoded (W', g) of each linear, NO Level Zero (tests/loader/lilicorr_repack_test.cc runs it on any host).
// Beside DFlashCheckpoint (19b / 23f's bf16 drafters), not inside it: this checkpoint's dtypes, names and head differ
// wholesale; the device-form pieces are the same structs (DFlashHostLinear) and layouts.
//
// **Names both ways** (after dflash_canon): exactly lilicorr_expected_names(desc) - every linear (`<base>.weight`) either
// BF16 as shipped or ModelOpt NVFP4 (`<base>.weight` U8 [N][K/2] + `<base>.weight_scale` F8_E4M3 [N][K/16] +
// `<base>.weight_scale_2` F32 [] - an `<base>.input_scale`, W4A16's unused activation scale, is accepted and ignored),
// every other tensor BF16. Refused by name: a missing or unknown tensor, a scale without its U8 weight (or the reverse),
// another dtype or shape, `embed_tokens` / `lm_head` (the drafter shares the target's), a draft vocabulary (d2t / t2d),
// mask_hidden, a shipped mask embedding (the mask row is the target's embedding row 990), NVFP4 tensors in a checkpoint
// whose config has no quantization_config.
//
// **The formats** (loader/nvfp4.h, spec 26 §3): Exact - W' in gemv_bf16's tiles and, for an NVFP4 linear, its g as the
// per-column output scale (DFlashHostLinear::gs); Int8 - spec 9's tiles of W' with the row scales fp32(g x s'_r) (a BF16
// linear: 19b's quantiser as is); Int4G64 - rtn from g x W' in fp64 (a BF16 linear: rtn.h's rule as is), gemv.cl layout 1.
// A multi-part linear (q||k||v, gate||up) is its parts' device forms concatenated (every layout is n-tile-major). The
// LiLiCorr head is always exact (Decision 3): LiliHostHead.
namespace loader {

// A LiLiCorr head linear's device form: W'^T bf16 [K][N] (k-major), the fp32 output scale [N] (g; 1 for BF16) and the
// fp32 bias [N] (0 where the checkpoint has none).
struct LiliHostHead {
  uint32_t K = 0, N = 0;
  std::vector<uint16_t> wt;
  std::vector<float> gs, bias;
  size_t bytes() const { return wt.size() * 2 + gs.size() * 4 + bias.size() * 4; }
};

// One linear's decoded form: W' bf16 row-major [N][K] (exact) and its global g (1 for a BF16 tensor).
struct LiliDecoded {
  uint32_t K = 0, N = 0;
  std::vector<uint16_t> wp;
  float g = 1.0f;
  bool nvfp4 = false;
};

// Every tensor name a LiLiCorr drafter of `d` holds (canonical spelling; the linears' `.weight` only - their NVFP4 parts
// are checked beside it), sorted; and the linears' bases (`<base>.weight` is a linear).
std::vector<std::string> lilicorr_expected_names(const model::DFlashDesc& d);
bool lilicorr_is_linear(const model::DFlashDesc& d, const std::string& canonical_weight);

class LiliCheckpoint {
 public:
  // `snapshot_or_repo`: resolve_snapshot's rules. Reads config.json (model::dflash_desc_from_config: must be the LiLiCorr
  // form), opens the one model.safetensors, checks names / dtypes / shapes (fc's K is the checkpoint's, a multiple of the
  // tap count: the target's hidden is checked by check_target).
  explicit LiliCheckpoint(const std::string& snapshot_or_repo);

  const model::DFlashDesc& desc() const { return d_; }
  const std::string& dir() const { return dir_; }
  uint32_t fc_k() const { return fc_k_; }
  uint32_t target_hidden() const { return fc_k_ / uint32_t(d_.taps.size()); }
  size_t tensors() const { return set_->tensors().size(); }
  size_t nvfp4_linears() const { return nvfp4_.size(); }
  // The backbone's NVFP4 output rows (lilicorr_weight_bytes' `nvfp4_rows`).
  uint64_t backbone_nvfp4_rows() const;
  bool is_nvfp4(const std::string& base) const;

  // model::check_dflash_target, plus fc's K = taps x target.hidden.
  void check_target(const model::DFlashTargetShape& target) const;

  // A linear by base name (`fc`, `layers.0.mlp.gate_proj`, `lilicorr.token_proj`, `lilicorr.layers.0.attn.in_proj`).
  LiliDecoded decoded(const std::string& base) const;

  // The backbone's device-form pieces in `f` (Exact / Int8 / Int4G64; Bf16 is refused - exact is its spelling here).
  DFlashHostLinear fc(DraftFormat f, unsigned threads = 0) const;
  DFlashHostLinear layer_linear(uint32_t layer, DFlashLinear which, DraftFormat f, unsigned threads = 0) const;
  std::vector<float> hidden_norm() const;
  std::vector<float> final_norm() const;
  std::vector<float> layer_norms(uint32_t layer) const;      // [input_layernorm | post_attention_layernorm]
  std::vector<float> qk_norms() const;                       // [layers][q_norm | k_norm][head_dim]
  std::vector<float> conv_bases() const;                     // [layers][attn, mlp][side][tap][hidden]
  std::vector<float> sinks() const;                          // [layers][q heads]

  // The LiLiCorr head (always exact): one linear (`layer` for the per-layer ones) and the small block (lili_head_small).
  LiliHostHead head_linear(LiliHead which, uint32_t layer = 0) const;
  std::vector<float> head_small() const;

 private:
  void open_and_check();
  const uint8_t* raw(const std::string& canonical) const;
  std::vector<float> widen(const std::string& canonical, size_t n) const;
  DFlashHostLinear make_linear(const std::vector<std::string>& bases, DraftFormat f, unsigned threads) const;

  model::DFlashDesc d_;
  std::string dir_;
  std::unique_ptr<SafetensorsSet> set_;
  std::vector<std::pair<std::string, std::string>> canon_;   // canonical -> the checkpoint's name (sorted)
  std::vector<std::string> nvfp4_;                           // the NVFP4 linears' bases (sorted)
  uint32_t fc_k_ = 0;
};

}  // namespace loader
