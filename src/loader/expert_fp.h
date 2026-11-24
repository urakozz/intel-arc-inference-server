#pragma once

#include "common/expert_fp.h"
#include "loader/safetensors.h"

namespace loader {

// One expert matrix, in layout-1 tile order [N/16][K/64]. Within each tile,
// consecutive words hold one K word per lane, then scale bytes [K group][lane].
// NVFP4 appends one FP32 outer scale word to the expert's matrix. The block
// decode is the GEMM's BF16 operand, and the outer scale is its FP32 epilogue.
struct ExpertFpMatrix {
  common::ExpertFpFormat format;
  unsigned K, N;
  std::vector<uint32_t> words;

  float outer_scale() const;
  float block_f32(unsigned n, unsigned k) const;
  uint16_t block_bf16(unsigned n, unsigned k) const { return common::f32_to_bf16(block_f32(n, k)); }
  float f32(unsigned n, unsigned k) const { return block_f32(n, k) * outer_scale(); }
  uint16_t bf16(unsigned n, unsigned k) const { return common::f32_to_bf16(f32(n, k)); }
};

struct ExpertFpWeights {
  ExpertFpMatrix gate_up, down;
};

class ExpertFpCheckpoint {
 public:
  explicit ExpertFpCheckpoint(const std::string& snapshot_dir);
  ExpertFpWeights load(unsigned layer, unsigned expert) const;

 private:
  ExpertFpMatrix load_matrix(const std::string& name, unsigned expert, bool fused) const;

  SafetensorsSet tensors_;
  std::map<std::string, common::ExpertFpFormat> formats_;
};

}  // namespace loader
