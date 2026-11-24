#pragma once
#include <map>
#include <stdexcept>
#include <string>
#include "common/expert_fp.h"
#include "common/json.h"

namespace common {

inline std::map<std::string, ExpertFpFormat> expert_fp_formats(const json::Value& config) {
  if (config.at("quant_method").str() != "inkling2")
    throw std::runtime_error("hf_quant_config.json: expected the inkling2 quantization method");
  std::map<std::string, ExpertFpFormat> formats;
  for (const auto& group : config.at("model").at("parameter_groups").arr()) {
    const auto& transform = group.at("weight_transform").str();
    for (const auto& value : group.at("tensors").arr()) {
      const auto& name = value.str();
      if (!name.ends_with(".mlp.experts.w13_weight") && !name.ends_with(".mlp.experts.w2_weight")) continue;
      const bool nv = transform == "nvfp4_nearest";
      if (!nv && transform != "mxfp8_nearest")
        throw std::runtime_error(name + ": unsupported expert weight transform " + transform);
      const auto& shape = group.at("weight_block_shape").arr();
      if (shape.size() != 2 || shape[0].num() != (nv ? 1 : 32) || shape[1].num() != (nv ? 16 : 32))
        throw std::runtime_error(name + ": unexpected expert weight block shape");
      if (nv && group.at("weight_scale_format").str() != "amax_denominator")
        throw std::runtime_error(name + ": expected amax_denominator NVFP4 scales");
      if (!formats.emplace(name, nv ? ExpertFpFormat::Nvfp4 : ExpertFpFormat::Mxfp8).second)
        throw std::runtime_error(name + ": duplicate expert quantization entry");
    }
  }
  return formats;
}

}  // namespace common
