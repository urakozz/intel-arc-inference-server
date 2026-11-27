#pragma once
// The host twin of kernels/tap_mean.cl (plan 21-QFN-v3e Task 1): tap row r is the mean of the row's HC
// hyper-connection streams - an fp32 sum in stream order, then one rounding to bf16 (the division by HC is exact
// in fp32, HC a power of two). The card test compares the kernel against this, bitwise. `in` bf16 [rows][HC*H],
// stream k at the row's offset k*H; `out` bf16 [rows][H].
#include <cstdint>

#include "common/bf16.h"

namespace tap_mean_ref {

inline void tap_mean(const uint16_t* in, uint16_t* out, uint32_t rows, uint32_t H, uint32_t HC) {
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < H; ++c) {
      float sum = 0.f;
      for (uint32_t k = 0; k < HC; ++k)
        sum += common::bf16_to_f32(in[(size_t)r * HC * H + size_t(k) * H + c]);
      out[(size_t)r * H + c] = common::f32_to_bf16(sum / float(HC));
    }
}

}  // namespace tap_mean_ref
