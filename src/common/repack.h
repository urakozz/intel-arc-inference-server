#pragma once
// The canonical on-device layouts as free functions - the single source of
// truth shared by the test helpers (Int4Gptq::tiled, Bf16Tiled) and the
// loader, which repacks per-linear straight from the mmapped checkpoint.
// Layout definitions: docs/12-kernels.md.
#include <cstdint>
#include <vector>

namespace common {

// int4 layout 1: per (n_tile of 16, k_group of 64) one 136-u32 tile:
// 128 u32 nibbles [k_octet j][lane l] = qweight word (g*8+j, n_tile*16+l),
// then 16 f16 scales packed two per u32 (even lane in the low half).
// Tiles are ordered g-inner, n_tile-outer, so one subgroup streams contiguous
// memory along K.
inline void repack_int4_layout1(const uint32_t* qweight, const uint16_t* scales, uint32_t K,
                                uint32_t N, uint32_t* out) {
  const uint32_t G = K / 64, NT = N / 16;
  for (uint32_t nt = 0; nt < NT; ++nt)
    for (uint32_t g = 0; g < G; ++g) {
      uint32_t* tile = out + (size_t(nt) * G + g) * 136;
      for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t l = 0; l < 16; ++l)
          tile[j * 16 + l] = qweight[size_t(g * 8 + j) * N + nt * 16 + l];
      for (uint32_t l = 0; l < 16; l += 2)
        tile[128 + l / 2] = uint32_t(scales[size_t(g) * N + nt * 16 + l]) |
                            (uint32_t(scales[size_t(g) * N + nt * 16 + l + 1]) << 16);
    }
}

// A fused linear's output column n sourced from one part's column.
struct ColSource {
  const uint32_t* qweight;  // the part's [K/8][N_part] words
  const uint16_t* scales;   // the part's [K/64][N_part] f16
  uint32_t n;               // column within the part
  uint32_t n_part = 0;      // the part's N (stride for row indexing)
};

struct Part {
  const uint32_t* qweight;
  const uint16_t* scales;
  uint32_t N;
};

inline std::vector<ColSource> cols_concat(const std::vector<Part>& parts) {
  std::vector<ColSource> cols;
  for (const Part& p : parts)
    for (uint32_t n = 0; n < p.N; ++n) cols.push_back({p.qweight, p.scales, n, p.N});
  return cols;
}

// gate||up interleaved in 16-column blocks: a[0:16] b[0:16] a[16:32] b[16:32]...
// so the lane holding gate[n] finds up[n] a fixed +16 output columns away.
inline std::vector<ColSource> cols_interleave16(const Part& a, const Part& b) {
  std::vector<ColSource> cols;
  for (uint32_t base = 0; base < a.N; base += 16) {
    for (uint32_t l = 0; l < 16; ++l) cols.push_back({a.qweight, a.scales, base + l, a.N});
    for (uint32_t l = 0; l < 16; ++l) cols.push_back({b.qweight, b.scales, base + l, b.N});
  }
  return cols;
}

// Same tiles as repack_int4_layout1, but output column n reads cols[n] - the
// fused case, where one linear's columns come from several checkpoint tensors.
// cols.size() must be N_total; every part must share K.
inline void repack_int4_layout1_cols(uint32_t K, uint32_t N_total,
                                     const std::vector<ColSource>& cols, uint32_t* out) {
  const uint32_t G = K / 64, NT = N_total / 16;
  for (uint32_t nt = 0; nt < NT; ++nt)
    for (uint32_t g = 0; g < G; ++g) {
      uint32_t* tile = out + (size_t(nt) * G + g) * 136;
      for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t l = 0; l < 16; ++l) {
          const ColSource& c = cols[nt * 16 + l];
          tile[j * 16 + l] = c.qweight[size_t(g * 8 + j) * c.n_part + c.n];
        }
      for (uint32_t l = 0; l < 16; l += 2) {
        const ColSource& c0 = cols[nt * 16 + l];
        const ColSource& c1 = cols[nt * 16 + l + 1];
        tile[128 + l / 2] = uint32_t(c0.scales[size_t(g) * c0.n_part + c0.n]) |
                            (uint32_t(c1.scales[size_t(g) * c1.n_part + c1.n]) << 16);
      }
    }
}

// bf16 tiles [n_tile][k_octet][8 k][16 n] from HF row-major [N][K].
inline void repack_bf16_tiled(const uint16_t* w, uint32_t K, uint32_t N, uint16_t* out) {
  const uint32_t K8 = K / 8;
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t k = 0; k < K; ++k)
      out[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16] = w[size_t(n) * K + k];
}

}  // namespace common
