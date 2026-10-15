#pragma once
#include <cstddef>
#include <cstdint>

// The int4 g64 GEMV layouts' byte counts (common/repack.h, src/kernels/gemv.cl) - ONE formula each, shared by
// every model's layout header (moe_layout.h, kolibri1_layout.h, muse_layout.h; spec 23b moved them here from
// moe_layout.h / kolibri1_layout.h unchanged - kolibri1_layout.h static_asserts its aliases byte-identical).
//
//   layout 0  GPTQ-native: words u32 [K/8][N] (8 nibbles of k along a word) + a separate f16 scales [K/64][N]
//   layout 1  subgroup tiles: (N / 16) x (K / 64) tiles of 136 u32 (128 nibble words + 8 scale words)
//
// Both are 4.25 bits a weight: at N % 16 == 0 and K % 64 == 0 the two counts are equal.
namespace loader {

inline constexpr uint32_t kInt4TileU32 = 136;   // common/repack.h layout 1: 128 nibble + 8 scale u32

// Layout-1 bytes of a K x N int4 g64 linear: (N / 16) x (K / 64) tiles of 136 u32.
inline constexpr size_t int4_layout1_bytes(uint32_t K, uint32_t N) {
  return size_t(N / 16) * (K / 64) * kInt4TileU32 * 4;
}
// Layout-0 bytes: the words and the f16 scales (two allocations, DeviceWeight::mem and ::scales).
inline constexpr size_t int4_layout0_words_bytes(uint32_t K, uint32_t N) { return size_t(K / 8) * N * 4; }
inline constexpr size_t int4_layout0_scales_bytes(uint32_t K, uint32_t N) { return size_t(K / 64) * N * 2; }
inline constexpr size_t int4_layout0_bytes(uint32_t K, uint32_t N) {
  return int4_layout0_words_bytes(K, N) + int4_layout0_scales_bytes(K, N);
}

}  // namespace loader
