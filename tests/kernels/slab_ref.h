#pragma once
// CPU references for the shared prefill slab kernels, each with a twin on the device; edit the two together.
//   dequant_slab()   src/kernels/prefill/slab_dequant_int4.cl (entry k2_pf_dequant_slab, binaries pf_int4_slab_K*_N*):
//                    a layout-0 int4 linear's [K][ns] bf16 slab from column n0, the columns past N zero
//   bf16_slab()      src/kernels/prefill/slab_bf16.cl (entry kol_pf_bf16_slab): a gemv_bf16-tiled bf16 linear's
//                    [K][ns] slab from column n0, the columns past N zero
// Used by the Flash-Next and Kolibri-1 prefill kernel tests. Moved out of K2-Horizon's k2_pf_ref.h
// (removed, tag archive/k2-horizon) and kolibri_pf_ref.h unchanged.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bf16.h"
#include "kernels/moe_ref.h"   // rne

namespace slab_ref {

using moe_ref::rne;

// k2_pf_dequant_slab: columns [n0, n0 + ns) of a layout-0 K x N int4 linear (words
// [K/8][N] u32, scales [K/64][N] f16 bits) as bf16 [K][ns]; columns >= N are zero.
inline std::vector<uint16_t> dequant_slab(const uint32_t* words, const uint16_t* scales, uint32_t K,
                                          uint32_t N, uint32_t n0, uint32_t ns) {
  std::vector<uint16_t> out(size_t(K) * ns, 0);
  for (uint32_t c = 0; c < ns; ++c) {
    const uint32_t n = n0 + c;
    if (n >= N) continue;
    for (uint32_t g = 0; g < K / 64; ++g) {
      const float scale = common::f16_to_f32(scales[size_t(g) * N + n]);
      for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t word = words[size_t(g * 8 + j) * N + n];
        for (uint32_t i = 0; i < 8; ++i) {
          const int q = int((word >> (4 * i)) & 0xFu) - 8;
          out[size_t(g * 64 + j * 8 + i) * ns + c] = rne(float(q) * scale);
        }
      }
    }
  }
  return out;
}

// kol_pf_bf16_slab: columns [n0, n0 + ns) of a gemv_bf16-tiled K x N linear as bf16 [K][ns]; c with
// n0 + c >= N exact zeros (the slab / tail contract of k2_pf_dequant_slab).
inline std::vector<uint16_t> bf16_slab(const uint16_t* tiles, uint32_t K, uint32_t N, uint32_t n0, uint32_t ns) {
  std::vector<uint16_t> out(size_t(K) * ns, 0);
  const uint32_t K8 = K / 8;
  for (uint32_t c = 0; c < ns; ++c) {
    const uint32_t n = n0 + c;
    if (n >= N) continue;
    for (uint32_t k = 0; k < K; ++k) out[size_t(k) * ns + c] = tiles[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16];
  }
  return out;
}

}  // namespace slab_ref
