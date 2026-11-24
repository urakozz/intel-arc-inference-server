#pragma once
#include <functional>
#include <memory>
#include <span>

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
  size_t layer_bytes() const;
  size_t total(bool draft) const;
};

// Opens and validates every text and draft tensor's header. Payload is only
// touched by stream(), which emits one tensor or one expert matrix at a time.
// The sink consumes or copies each span before returning.
class Inkling2Checkpoint {
 public:
  using Sink = std::function<void(uint32_t arena, size_t offset, std::span<const std::byte>)>;
  explicit Inkling2Checkpoint(const std::string& snapshot_or_repo);
  const std::string& directory() const { return dir_; }
  const model::Inkling2Desc& desc() const { return desc_; }
  const Inkling2WeightLayout& layout() const { return layout_; }
  bool has_drafter() const { return bool(draft_); }
  size_t tensor_count() const { return text_.tensors().size() - vision_ + draft_tensors_.size(); }
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
  const uint8_t* data(const std::string& name, const Source& s) const;

  std::string dir_;
  model::Inkling2Desc desc_;
  SafetensorsSet text_;
  std::unique_ptr<MappedFile> draft_;
  std::map<std::string, TensorInfo> draft_tensors_;
  size_t draft_data_start_ = 0, vision_ = 0;
  std::map<std::string, Source> sources_;
  Inkling2WeightLayout layout_;
};

}  // namespace loader
