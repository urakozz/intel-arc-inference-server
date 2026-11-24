#pragma once

#include <cmath>
#include "common/bf16.h"

namespace common {

enum class ExpertFpFormat { Nvfp4 = 1, Mxfp8 = 2 };

inline float e2m1_to_f32(uint8_t code) {
  return f16_to_f32(uint16_t(((code & 8u) << 12) | ((code & 7u) << 9))) * 0x1p14f;
}

// The loader rejects the two E4M3 NaN encodings.
inline float e4m3_to_f32(uint8_t code) {
  return f16_to_f32(uint16_t(((code & 128u) << 8) | ((code & 127u) << 7))) * 0x1p8f;
}

inline float expert_fp_block_f32(ExpertFpFormat format, uint8_t code, uint8_t scale) {
  if (format == ExpertFpFormat::Nvfp4) return e2m1_to_f32(code) * e4m3_to_f32(scale);
  return std::ldexp(e4m3_to_f32(code), int(scale) - 127);
}

inline unsigned expert_fp_code_bits(ExpertFpFormat format) {
  return format == ExpertFpFormat::Nvfp4 ? 4 : 8;
}

inline unsigned expert_fp_group(ExpertFpFormat format) {
  return format == ExpertFpFormat::Nvfp4 ? 16 : 32;
}

inline unsigned expert_fp_tile_words(ExpertFpFormat format) {
  return 32 * expert_fp_code_bits(format) + (64 / expert_fp_group(format)) * 4;
}

}  // namespace common
