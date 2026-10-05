#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "common/bf16.h"
#include "common/kv8.h"

// Spec 15e: round-to-nearest int4 g64 SYMMETRIC quantisation of one bf16 linear on the
// host, into the GPTQ v1 form the loader reads from a checkpoint (loader::LinearSrc's
// int4 path): qweight [K/8][N] u32 - word (r, n) holds k = 8r..8r+7 of column n, k = 8r
// in the lowest nibble - and f16 scales [K/64][N], zero point 8 (qzeros 0x77777777), so
// the dequant is the kernels' w = scale x (q - 8) (docs/02, common/int4.h).
//
// Used for ONE thing: Ornith's MTP head ships its 257 SwiGLU experts in bf16 (the
// published checkpoint is all bf16, and an AutoRound export of the main model leaves
// `mtp.*` as it found it), while the MoE kernels (src/kernels/moe.cl) read int4 g64
// layout-1 blocks. The head only DRAFTS - the verify list decides every token - so its
// quantisation moves acceptance, never output (spec 8 M3). AutoRound is not run at load.
//
// The quantiser is GPTQ's symmetric one (maxq 15, zero (maxq + 1) / 2 = 8), per output
// column n and group of 64 k:
//
//   amax  = max_k |w[n][k]|                     (bf16 values, exact in fp32)
//   s16   = f16(2 amax / 15)                    round to nearest even, subnormals kept
//   q     = clamp(rint(w / f32(s16)) + 8, 0, 15)    rint: ties to even, as torch.round
//
// The STORED scale divides (not GPTQ's fp32 one), so q is the nearest grid point of the
// dequant the kernels compute. An all-zero group has s16 = 0 and q = 8 (w = 0 exactly).
namespace loader {

// `w` is the checkpoint's row-major bf16 [N][K] (nn.Linear's weight: N outputs). K is a
// multiple of 64. `qweight` holds K/8 x N words, `scales` K/64 x N halves.
inline void rtn_int4_g64(const uint16_t* w, uint32_t K, uint32_t N, uint32_t* qweight,
                         uint16_t* scales) {
  for (uint32_t n = 0; n < N; ++n) {
    const uint16_t* row = w + size_t(n) * K;
    for (uint32_t g = 0; g < K / 64; ++g) {
      float amax = 0.f;
      for (uint32_t k = g * 64; k < g * 64 + 64; ++k)
        amax = std::fmax(amax, std::fabs(common::bf16_to_f32(row[k])));
      const uint16_t s16 = common::kv8::f32_to_f16_rne(2.0f * amax / 15.0f);
      scales[size_t(g) * N + n] = s16;
      const float s = common::f16_to_f32(s16);
      for (uint32_t r = g * 8; r < g * 8 + 8; ++r) {
        uint32_t word = 0;
        for (uint32_t j = 0; j < 8; ++j) {
          const float v = common::bf16_to_f32(row[r * 8 + j]);
          float q = s > 0.f ? std::rint(v / s) + 8.0f : 8.0f;
          q = std::fmin(std::fmax(q, 0.0f), 15.0f);
          word |= uint32_t(q) << (4 * j);
        }
        qweight[size_t(r) * N + n] = word;
      }
    }
  }
}

}  // namespace loader
