#pragma once
// CPU references for spec 24d's LFM2.5-24B-A2B prefill kernels (src/kernels/lfm2moe/lfm_pf_{conv,moe,attn,prep}.cl)
// - each function with a twin on the device; EDIT THE TWO TOGETHER. Built on 24c's tests/kernels/lfm2moe_ref.h (the
// decode chains: conv_step, route, combine, attention, attention_eager) and spec 15d's pf_moe_ref.h (dequant_block),
// which it includes and does not change - so "prefill applies decode's per-row chain" is a comparison between two
// functions of this directory (tests/kernels/lfm2moe_pf_ref_test.cc), not a claim.
//
//   conv_seed()    lfm_pf_conv_seed: Bx of positions p0 - 2, p0 - 1 out of the ring into seed [2][H] (0 below position
//                  0 - the ring is never cleared: a fresh sequence never reads it)
//   conv_chunk()   lfm_pf_conv: a causal conv has no recurrence - row t needs Bx[t - 2 .. t] and Bx is a function of
//                  in_proj's row alone - so every row recomputes its two predecessors' Bx (the same rounding) instead
//                  of carrying it; rows 0 / 1 read the seed. B, C, x = rne(0 + in_proj[t][col]) (decode's split-K sum
//                  at S 1); Bx = rne(B · x); acc = fma(w2, Bx[t], fma(w1, Bx[t-1], fma(w0, Bx[t-2], 0))) (taps oldest
//                  first); y = rne(C · rne(acc)); the chunk's last min(C, 2) Bx into their ring slots (p0 + t) % 16 -
//                  decode then continues from the ring exactly as after decode steps. `tiled`: in_proj in the conv-tile
//                  column order (the arm conv's load, loader::repack_in_proj conv_tiles: tile t = [B 16 | C 16 | x 16])
//   tmax()         the padded tile count of a C-row chunk, NO shared block: floor((4 C + 64 x 31) / 32) (318 at 2048;
//                  runtime::lfm2moe::pf_tiles is the host's home of it)
//   sort()         lfm_pf_sort: every (token, slot) pair one row, expert-major, ascending token inside an expert, each
//                  expert padded to whole 32-row tiles; the tile table padded to tmax with NONE; pair_row; the header
//   gather()       lfm_pf_gather: row r of the sorted A operand (zeros for a padding row)
//   dequant_gu/_dn lfm_pf_dequant_gu / _dn: one expert's int4 layout-1 block as bf16 [K][N] (pf_moe_ref's)
//   combine()      lfm_pf_combine: lfm_moe_down's epilogue over the sorted rows - t_k = rne(f32(y_k) · w_k) in RANK
//                  order, acc = ((t0 + t1) + t2) + t3 (fp32, no fma), h = rne(h + rne(acc)) (the residual fold, in place)
//   flash_row()    lfm_pf_attn's walk in fp64: per 8-row group, 64-key tiles from key 0 up to the group's last row,
//                  the keys of the growing cache [rows][8][64] (earlier chunks' and this chunk's), the online softmax
//                  with the -INF guard (a wholly masked tile adds exact zeros)
//   eager_row()    the reference's rounding points over the same keys (lfmref::attention_eager over [0, p], o rounded
//                  once) - the EAGER build's tolerance reference
//
// What is NOT bit-exact between these and the device: the grouped GEMMs' sums (DPAS order), the flash softmax's exp /
// sum order (a cosine bar) and `exp` itself. The seed, the conv, the sort, the gather, the dequant and the combine are
// exact.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kernels/lfm2moe_ref.h"
#include "kernels/pf_moe_ref.h"   // dequant_block (layout 1)

namespace lfmpfref {

using lfmref::f32;
using lfmref::rf;
using lfmref::rne;
constexpr uint32_t kC = 2048, kTm = 32, kNone = 0xFFFFFFFFu;
constexpr uint32_t kE = lfmref::kExperts, kK = lfmref::kTopK, kH = lfmref::kHidden, kI = lfmref::kInter,
                   kHd = lfmref::kHd, kQH = lfmref::kQHeads, kKVH = lfmref::kKvHeads, kRing = lfmref::kRing;
constexpr uint32_t kRows = 16;   // lfm_pf_conv: rows per work-item
// The header (lfm_pf_moe.cl H_*, kernels::lfm2moe::pf_hdr): [0] tiles used, [1] 0 (reserved: Kolibri's shared-row
// word - this model has no shared expert), [2] rows used (32 x tiles), [3] C, [4 + e] expert e's rows (unpadded).
constexpr uint32_t kHdrTiles = 0, kHdrRows = 2, kHdrC = 3, kHdrCount = 4;
inline constexpr uint32_t hdr_words() { return (kHdrCount + kE + 15) / 16 * 16; }   // 80
constexpr uint32_t kKt = 64, kRpg = 8;   // the flash walk's key tile and rows per sub-group

// --- the conv over a chunk ----------------------------------------------------------------------------------------
// seed [2][H]: Bx of positions p0 - 2 (row 0) and p0 - 1 (row 1) from the ring [16][H]; 0 below position 0.
inline std::vector<uint16_t> conv_seed(const uint16_t* ring, uint32_t p0, uint32_t H = kH) {
  std::vector<uint16_t> s(size_t(2) * H, 0);
  for (uint32_t j = 0; j < 2; ++j) {
    const int p = int(p0) - 2 + int(j);
    if (p < 0) continue;
    for (uint32_t c = 0; c < H; ++c) s[size_t(j) * H + c] = ring[size_t(uint32_t(p) % kRing) * H + c];
  }
  return s;
}
// in_proj's column of channel c's B (part 0), C (1) or x (2): plain [B | C | x], or the conv-tile order.
inline size_t conv_col(uint32_t c, uint32_t part, uint32_t H, bool tiled) {
  return tiled ? (size_t(3) * (c / 16) + part) * 16 + c % 16 : size_t(part) * H + c;
}
// Bx of chunk row t (t >= 0) at channel c: rne(B · x) of in_proj's row.
inline uint16_t conv_bx(const float* inproj, uint32_t C, uint32_t t, uint32_t c, uint32_t H, bool tiled) {
  const uint16_t B = rne(lfmref::slice_sum(inproj, 1, C, t, 3 * H, conv_col(c, 0, H, tiled)));
  const uint16_t X = rne(lfmref::slice_sum(inproj, 1, C, t, 3 * H, conv_col(c, 2, H, tiled)));
  return rne(f32(B) * f32(X));
}
// `inproj` fp32 [C][3H] (the in_proj GEMM's rows, pitch 3H), `seed` [2][H] (conv_seed's), `taps` fp32 [H][3]. Returns
// y [C][H]; writes the Bx of rows max(0, C - 2) .. C - 1 into `ring` slots (p0 + t) % 16.
inline std::vector<uint16_t> conv_chunk(const float* inproj, const uint16_t* seed, const float* taps, uint16_t* ring,
                                        uint32_t p0, uint32_t C, uint32_t H = kH, bool tiled = false) {
  std::vector<uint16_t> y(size_t(C) * H);
  for (uint32_t c = 0; c < H; ++c) {
    const float w0 = taps[size_t(c) * 3 + 0], w1 = taps[size_t(c) * 3 + 1], w2 = taps[size_t(c) * 3 + 2];
    auto bx = [&](int t) -> float {
      return t < 0 ? f32(seed[size_t(t + 2) * H + c]) : f32(conv_bx(inproj, C, uint32_t(t), c, H, tiled));
    };
    for (uint32_t t = 0; t < C; ++t) {
      const float b0 = bx(int(t)), b1 = bx(int(t) - 1), b2 = bx(int(t) - 2);
      const uint16_t Cv = rne(lfmref::slice_sum(inproj, 1, C, t, 3 * H, conv_col(c, 1, H, tiled)));
      const float acc = std::fma(w2, b0, std::fma(w1, b1, std::fma(w0, b2, 0.0f)));
      y[size_t(t) * H + c] = rne(f32(Cv) * f32(rne(acc)));
    }
    for (uint32_t t = C >= 2 ? C - 2 : 0; t < C; ++t)
      ring[size_t((p0 + t) % kRing) * H + c] = conv_bx(inproj, C, t, c, H, tiled);
  }
  return y;
}

// --- the grouped MoE ---------------------------------------------------------------------------------------------
inline uint32_t tmax(uint32_t C) { return (C * kK + kE * (kTm - 1)) / kTm; }

struct Sorted {
  std::vector<uint32_t> hdr, tiles, row_tok, pair_row;
  uint32_t tiles_used = 0;
  uint32_t rows_used() const { return hdr[kHdrRows]; }
};
// lfm_pf_sort over route rows [C][16] (ids at lfmref::kIds; an id past 63 clamps to it, as the kernel's uchar
// staging does - lfm_route never writes one).
inline Sorted sort(const uint32_t* route, uint32_t C) {
  const uint32_t T = tmax(C);
  std::vector<uint32_t> id(size_t(C) * kK);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < kK; ++j)
      id[size_t(t) * kK + j] = std::min(route[size_t(t) * lfmref::kWords + lfmref::kIds + j], kE - 1);
  std::vector<uint32_t> cnt(kE, 0), toff(kE + 1, 0);
  for (uint32_t v : id) ++cnt[v];
  for (uint32_t e = 0; e < kE; ++e) toff[e + 1] = toff[e] + (cnt[e] + kTm - 1) / kTm;
  const uint32_t ntiles = toff[kE];
  Sorted o;
  o.hdr.assign(hdr_words(), 0);
  o.tiles.assign(size_t(T) * 2, 0);
  o.row_tok.assign(size_t(T) * kTm, kNone);
  o.pair_row.assign(size_t(C) * kK, kNone);
  for (uint32_t t = 0; t < T; ++t) o.tiles[2 * t] = kNone;
  for (uint32_t e = 0; e < kE; ++e) {
    const uint32_t t0 = toff[e], r0 = t0 * kTm;
    for (uint32_t i = 0; i < toff[e + 1] - t0; ++i) {
      o.tiles[2 * (t0 + i)] = e;
      o.tiles[2 * (t0 + i) + 1] = r0 + i * kTm;
    }
    uint32_t p = 0;
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < kK; ++j)
        if (id[size_t(t) * kK + j] == e) {
          o.row_tok[r0 + p] = t;
          o.pair_row[size_t(t) * kK + j] = r0 + p;
          ++p;
        }
    o.hdr[kHdrCount + e] = cnt[e];
  }
  o.tiles_used = ntiles;
  o.hdr[kHdrTiles] = ntiles;
  o.hdr[kHdrRows] = ntiles * kTm;
  o.hdr[kHdrC] = C;
  return o;
}

// Row r of the sorted A operand: x row row_tok[r] (bf16 [C][width]), zeros for a padding row.
inline std::vector<uint16_t> gather(const Sorted& st, const uint16_t* x, uint32_t width = kH) {
  std::vector<uint16_t> g(size_t(st.rows_used()) * width, 0);
  for (uint32_t r = 0; r < st.rows_used(); ++r)
    if (st.row_tok[r] != kNone)
      std::copy(x + size_t(st.row_tok[r]) * width, x + size_t(st.row_tok[r] + 1) * width, g.begin() + size_t(r) * width);
  return g;
}

// lfm_pf_dequant_gu: expert e's gate||up block as bf16 [hidden][2I] (interleave16 kept); _dn: [I][hidden].
inline std::vector<uint16_t> dequant_gu(const uint32_t* gu, uint32_t e) {
  return pf_moe_ref::dequant_block(gu + size_t(e) * lfmref::gate_up_block_words(), kH, 2 * kI);
}
inline std::vector<uint16_t> dequant_dn(const uint32_t* dn, uint32_t e) {
  return pf_moe_ref::dequant_block(dn + size_t(e) * lfmref::down_block_words(), kI, kH);
}

// lfm_pf_combine for every token of the chunk, folding into h [C][hidden] in place: `y` the down GEMM's sorted rows
// (rne(Σ), pitch hidden), the route rows [C][16] (ids and weights in rank order).
inline void combine(const uint32_t* route, const Sorted& s, const uint16_t* y, uint16_t* h, uint32_t C) {
  for (uint32_t t = 0; t < C; ++t) {
    const uint32_t* rr = route + size_t(t) * lfmref::kWords;
    for (uint32_t n = 0; n < kH; ++n) {
      float acc = 0.0f;
      for (uint32_t j = 0; j < kK; ++j) {
        const float tj = f32(rne(f32(y[size_t(s.pair_row[size_t(t) * kK + j]) * kH + n]) * lfmref::as_f32(rr[lfmref::kW + j])));
        acc = j == 0 ? tj : acc + tj;
      }
      uint16_t& hv = h[size_t(t) * kH + n];
      hv = rne(f32(hv) + f32(rne(acc)));
    }
  }
}

// --- the attention -----------------------------------------------------------------------------------------------
// The cache [rows][8][64] bf16 (key k at row k), q fp32 [C][32][64] (bf16 values: lfm_attn_prep's), the chunk at c0.
// One (row t, head h) by lfm_pf_attn's walk in fp64: the row's 8-row group r0 = t & ~7 walks 64-key tiles from key 0
// up to min(c0 + C, c0 + r0 + 8); a row sees keys k <= c0 + t; the online softmax with m_safe = (m == -INF ? 0 : m).
inline std::vector<double> flash_row(const float* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0, uint32_t C,
                                     uint32_t t, uint32_t h) {
  const uint32_t j = h / (kQH / kKVH), r0 = t / kRpg * kRpg, p = c0 + t;
  const uint32_t last = std::min(c0 + C, c0 + r0 + kRpg);
  const float* qr = q + (size_t(t) * kQH + h) * kHd;
  double m = -INFINITY, l = 0.0;
  std::vector<double> o(kHd, 0.0);
  for (uint32_t t0 = 0; t0 < last; t0 += kKt) {
    double s[kKt];
    double tm = -INFINITY;
    for (uint32_t i = 0; i < kKt; ++i) {
      const uint32_t k = t0 + i;
      if (k > p) {
        s[i] = -INFINITY;
        continue;
      }
      double a = 0;
      const size_t row = (size_t(k) * kKVH + j) * kHd;
      for (uint32_t d = 0; d < kHd; ++d) a += double(qr[d]) * f32(kc[row + d]);
      s[i] = a * 0.125;
      tm = std::max(tm, s[i]);
    }
    const double mnew = std::max(m, tm), msafe = mnew == -INFINITY ? 0.0 : mnew;
    const double corr = std::exp(m - msafe);
    double psum = 0;
    for (uint32_t d = 0; d < kHd; ++d) o[d] *= corr;
    for (uint32_t i = 0; i < kKt; ++i) {
      const double pi = std::exp(s[i] - msafe);
      if (pi == 0.0) continue;
      psum += pi;
      const size_t row = (size_t(t0 + i) * kKVH + j) * kHd;
      for (uint32_t d = 0; d < kHd; ++d) o[d] += pi * f32(vc[row + d]);
    }
    l = l * corr + psum;
    m = mnew;
  }
  for (double& v : o) v /= l;
  return o;
}
// The reference's rounding points for (row t, head h) at absolute position p = c0 + t: lfmref::attention_eager over
// keys [0, p] (scores rne(f32(rne(q.k)) x 0.125), torch's fp32 softmax rounded once, P.V in fp32), o rounded once.
inline std::vector<uint16_t> eager_row(const float* q, const uint16_t* kc, const uint16_t* vc, uint32_t p, uint32_t h) {
  const uint32_t j = h / (kQH / kKVH), L = p + 1;
  const lfmref::EagerHead e = lfmref::attention_eager(q, kc, vc, L, j, std::max(L, 1u));
  std::vector<uint16_t> o(kHd);
  for (uint32_t d = 0; d < kHd; ++d) o[d] = rne(e.o[d]);
  return o;
}

}  // namespace lfmpfref
