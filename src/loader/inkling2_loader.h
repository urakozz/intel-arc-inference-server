#pragma once
#include <functional>
#include <memory>
#include <span>

#include "loader/dflash_repack.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "model/inkling2.h"

namespace loader {

struct Inkling2Weight {
  uint32_t arena = 0;
  size_t offset = 0, bytes = 0;
  // An int8 linear has one fp32 scale per padded output row; both locations
  // are absolute byte offsets into the same arena.
  size_t scales = 0;
  uint32_t K = 0, N = 0;
};

struct Inkling2WeightLayout {
  static constexpr uint32_t kModel = 0, kFirstLayer = 1, kDraft = kFirstLayer + model::Inkling2Desc::layers;
  std::array<size_t, kDraft + 1> bytes{};
  std::map<std::string, Inkling2Weight> weights;
  DraftFormat draft_format = DraftFormat::Int8;   // DFlash 2 linears
  uint64_t packed_draft_key = 0;
  size_t layer_bytes() const;
  size_t total(bool draft) const;
};

// Opens and validates every text and draft tensor's header. Payload is only
// touched by stream(), which emits one tensor or one expert matrix at a time.
// The sink consumes or copies each span before returning.
class Inkling2Checkpoint {
 public:
  using Sink = std::function<void(uint32_t arena, size_t offset, std::span<const std::byte>)>;
  // `dflash`: a DFlash 2 drafter's folder (incoai/Inkling-2-Mini-DFlash2, plan 41k) loaded in place of the release's
  // DSpark head (mtp.safetensors) - one drafter, never both; desc().draft is then that drafter's (dflash()). Its 81
  // tensors in the draft arena by their checkpoint names, in df_prep's forms (runtime/dflash/dflash_capture.cc): the
  // linears int8 at load (one fp32 scale a row), or packed int4 g64 for a GPTQ export; q || k || v as
  // layers.<l>.self_attn.qkv_proj.weight and gate || up as
  // layers.<l>.mlp.gate_up_proj.weight (rows concatenated, so a context projection reads the k || v rows in place),
  // layers.<l>.self_attn.kv_proj.weight the k || v rows of the first in place (no bytes of its own), q_norm || k_norm
  // as layers.<l>.self_attn.qk_norm (fp32), the other norms and the conv bases fp32, the codebooks
  // raw bf16, and "df2.mask_embed": the target's embed_norm(embed[mask_token_id]) rounded to bf16, computed here once
  // (the block's mask rows).
  explicit Inkling2Checkpoint(const std::string& snapshot_or_repo, const std::string& dflash = "");
  const std::string& directory() const { return dir_; }
  const model::Inkling2Desc& desc() const { return desc_; }
  const Inkling2WeightLayout& layout() const { return layout_; }
  bool has_drafter() const { return draft_ || dflash_; }
  bool dflash() const { return desc_.draft.form == model::DFlashForm::Vllm; }
  size_t tensor_count() const { return text_.tensors().size() - vision_ + (dflash_ ? dflash_->tensors() : draft_tensors_.size()); }
  void stream(const Sink& write, bool draft, unsigned threads = 0) const;

 private:
  enum class Encoding { Raw, Float, Bf16Tile, Int8Tile, Expert, FoldSource };
  struct Source {
    Encoding encoding;
    std::vector<uint64_t> shape;
    std::string dtype;
    uint32_t arena;
  };
  void add(std::string name, Source source, uint32_t padded_rows = 0);
  void open_draft(const std::string& file);   // the draft file's header into draft_tensors_
  void add_dflash(const std::string& dir);
  const uint8_t* data(const std::string& name, const Source& s) const;

  std::string dir_;
  model::Inkling2Desc desc_;
  SafetensorsSet text_;
  std::unique_ptr<MappedFile> draft_;
  std::unique_ptr<DFlashCheckpoint> dflash_;
  std::map<std::string, TensorInfo> draft_tensors_;
  size_t draft_data_start_ = 0, vision_ = 0;
  std::map<std::string, Source> sources_;
  Inkling2WeightLayout layout_;
};

}  // namespace loader
