#pragma once
#include <cstdint>
#include <cstring>

// Open-work 3n: the packed A2 that `pf_gdn_A2_hilo` (pf_gdn_wy.cl entry (2c)) writes and
// `pf_gdn_scan_dpas_split` (pf_gdn_scan.cl entry (3)) reads - the host's statement of the layout,
// for the tests and probes that feed the scan without running the A2 kernel, or that check the
// kernel's words against `pf_gdn_A2`'s fp32 tile. The two .cl files carry the same numbers.
//
// Per (sub-chunk t, head h), at `(t * heads + h) * kA2HlWords` uints: the hi words of the
// kA2HlBlocks blocks on or below the diagonal, then the lo words at +kA2HlLo. Block (bnt, bkb),
// bkb <= bnt, sits at `(bnt (bnt + 1) / 2 + bkb) * 128`; inside it row i (16 bnt + ln) owns 8 uints
// at `ln * 8`, uint r holding A2[i][16 bkb + 2r] in its low half and A2[i][16 bkb + 2r + 1] in its
// high half - lane ln's DPAS B operand. hi = rne(v), lo = rne(v - f32(hi)), the scan's old split.
namespace runtime::prefill {

inline constexpr uint32_t kA2HlBlocks = 10;                    // (nt, kb) with kb <= nt, nt < 4
inline constexpr uint32_t kA2HlLo = kA2HlBlocks * 16 * 8;      // 1,280
inline constexpr uint32_t kA2HlWords = 2 * kA2HlLo;            // 2,560 uints = 10,240 B

namespace a2_pack_detail {
inline uint16_t rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
inline float widen(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
}  // namespace a2_pack_detail

// One 64 x 64 fp32 tile (row-major, `pf_gdn_A2`'s) into its kA2HlWords words.
inline void pack_a2_hilo_tile(const float* tile, uint32_t* out) {
  using a2_pack_detail::rne;
  using a2_pack_detail::widen;
  for (uint32_t bnt = 0; bnt < 4; ++bnt)
    for (uint32_t bkb = 0; bkb <= bnt; ++bkb)
      for (uint32_t ln = 0; ln < 16; ++ln)
        for (uint32_t r = 0; r < 8; ++r) {
          const uint32_t p = ((bnt * (bnt + 1)) / 2 + bkb) * 128 + ln * 8 + r;
          const float* v = tile + size_t(bnt * 16 + ln) * 64 + bkb * 16 + 2 * r;
          const uint16_t h0 = rne(v[0]), h1 = rne(v[1]);
          out[p] = (uint32_t(h1) << 16) | h0;
          out[kA2HlLo + p] = (uint32_t(rne(v[1] - widen(h1))) << 16) | rne(v[0] - widen(h0));
        }
}

// `tiles` fp32 tiles of 64 x 64 (sub-chunk-major, then head: `pf_gdn_A2`'s buffer) into
// `tiles * kA2HlWords` words.
inline void pack_a2_hilo(const float* a2, size_t tiles, uint32_t* out) {
  for (size_t t = 0; t < tiles; ++t) pack_a2_hilo_tile(a2 + t * 64 * 64, out + t * kA2HlWords);
}

}  // namespace runtime::prefill
