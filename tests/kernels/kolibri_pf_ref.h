#pragma once
// CPU references for spec 20d's Kolibri-1 prefill kernels (src/kernels/kolibri/kol_pf_moe.cl,
// kol_pf_attn.cl, kol_pf_linear.cl) - each function with a twin on the device; edit the two together.
// Built on 20c's tests/kernels/kolibri_ref.h (the decode chains: route, combine, eager attention,
// ring_row), spec 15d's pf_moe_ref.h (the grouped layout, dequant_block) and 18c's k2_pf_ref.h (the
// layout-0 slab), which it includes and does not change - so "prefill combines by decode's rule" is a
// comparison between two functions of this directory (kolibri_pf_ref_test.cc), not a claim.
//
//   tmax()          the padded tile count of a C-row chunk: floor((C x 6 + 384 x 31) / 32) + ceil(C / 32)
//                   (820 at C = 2048; runtime::kolibri::pf_tiles is the host's home of it)
//   sort()          kol_pf_sort: expert-major rows, ascending token inside an expert, tiles of 32, the
//                   bf16 shared expert's block 384 last with its C tokens in order, pair_row, the header
//   gather()        kol_pf_gather: row r of the sorted A operand (zeros for padding)
//   dequant_gu/_dn  kol_pf_dequant_gu / _dn: block e < 384 the int4 layout-1 dequant (pf_moe_ref's),
//                   block 384 the shared expert's bf16 tiles copied into row-major [K][N]
//   bf16_slab()     kol_pf_bf16_slab: columns [n0, n0 + ns) of a gemv_bf16-tiled linear as bf16 [K][ns],
//                   the columns >= N exact zeros (the o_proj tail: 2560 = 2 x 1024 + 512)
//   combine()       kol_pf_moe_combine: kol_moe_down's epilogue over sorted rows - acc = acc + y_j x w_j in
//                   ascending id (product rounded to fp32, THEN added: no fma), + the shared row, one
//                   rounding; into `mo`, NOT the residual (the sandwich's post_ffn_norm comes first)
//   flash_window()  kol_pf_flash_attn's walk in fp64: per 8-row group, 64-key tiles from an ABSOLUTE
//                   multiple of 64 (the window's first tile for a sliding layer, 0 for a full one), the
//                   keys read through the ring (row k & 4095) or the linear cache, the online softmax with
//                   the -INF guard (a wholly masked tile adds exact zeros, even as a row's first tile)
//   eager_window()  the reference's rounding points over the same keys: s = rne(f32(rne(q.k)) x 128^-0.5),
//                   p = torch's fp32 softmax rounded once (k2_ref::eager_softmax), o = rne(Σ p v)
//
// What is NOT bit-exact between these and the device: the grouped GEMMs' sums (DPAS order), the flash
// softmax's exp / sum order (a cosine bar), and `exp` itself. The sort, gather, dequants, slab copy and
// the combine are exact.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/bf16.h"
#include "kernels/k2_pf_ref.h"   // dequant_slab (layout 0, the int4 attention arm's slabs)
#include "kernels/k2_ref.h"      // eager_score / eager_softmax (torch's), the fp64 attention
#include "kernels/kolibri_ref.h"
#include "kernels/pf_moe_ref.h"  // dequant_block (layout 1)

namespace kolibri_pf_ref {

using kolibri_ref::f32;
using kolibri_ref::rf;
using kolibri_ref::rne;
constexpr uint32_t kTm = 32, kNone = 0xFFFFFFFFu, kC = 2048;
constexpr uint32_t kE = kolibri_ref::kExperts, kK = kolibri_ref::kTopK, kH = kolibri_ref::kHidden,
                   kI = kolibri_ref::kInter, kHd = kolibri_ref::kHd, kQH = kolibri_ref::kQHeads,
                   kKVH = kolibri_ref::kKvHeads, kRing = kolibri_ref::kRing;
constexpr uint32_t kShared = kE;   // the shared expert's block in a weight batch
// The header (kol_pf_moe.cl H_*, kernels::kolibri::pf_hdr): [0] tiles used, [1] the shared expert's
// first row, [2] rows used (32 x tiles), [3] C, [4 + e] expert e's rows (e < 384), [4 + 384] C.
constexpr uint32_t kHdrTiles = 0, kHdrSharedRow = 1, kHdrRows = 2, kHdrC = 3, kHdrCount = 4;
inline constexpr uint32_t hdr_words() { return (kHdrCount + kE + 1 + 15) / 16 * 16; }   // 400
constexpr uint32_t kKt = 64, kRpw = 8;   // the flash walk's key tile and rows per sub-group

inline uint32_t tmax(uint32_t C) { return (C * kK + kE * (kTm - 1)) / kTm + (C + kTm - 1) / kTm; }

struct Sorted {
  std::vector<uint32_t> hdr, tiles, row_tok, pair_row;
  uint32_t tiles_used = 0;
  uint32_t rows_used() const { return hdr[kHdrRows]; }
  uint32_t shared_row() const { return hdr[kHdrSharedRow]; }
};

// kol_pf_sort over route rows [C][kolibri_ref::kWords] (ids at kIds; an id past 383 clamps to it, as the
// kernel's ushort staging does - kol_route never writes one).
inline Sorted sort(const uint32_t* route, uint32_t C) {
  const uint32_t T = tmax(C);
  std::vector<uint32_t> id(size_t(C) * kK);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < kK; ++j)
      id[size_t(t) * kK + j] = std::min(route[size_t(t) * kolibri_ref::kWords + kolibri_ref::kIds + j], kE - 1);
  std::vector<uint32_t> cnt(kE, 0), toff(kE + 1, 0);
  for (uint32_t v : id) ++cnt[v];
  for (uint32_t e = 0; e < kE; ++e) toff[e + 1] = toff[e] + (cnt[e] + kTm - 1) / kTm;
  const uint32_t ts = toff[kE], st = (C + kTm - 1) / kTm, ntiles = ts + st, rs = ts * kTm;
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
  }
  for (uint32_t i = 0; i < st; ++i) {
    o.tiles[2 * (ts + i)] = kShared;
    o.tiles[2 * (ts + i) + 1] = rs + i * kTm;
  }
  for (uint32_t i = 0; i < st * kTm; ++i) o.row_tok[rs + i] = i < C ? i : kNone;
  o.tiles_used = ntiles;
  o.hdr[kHdrTiles] = ntiles;
  o.hdr[kHdrSharedRow] = rs;
  o.hdr[kHdrRows] = ntiles * kTm;
  o.hdr[kHdrC] = C;
  for (uint32_t e = 0; e < kE; ++e) o.hdr[kHdrCount + e] = cnt[e];
  o.hdr[kHdrCount + kE] = C;
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

// A gemv_bf16-tiled {K, N} weight (common::repack_bf16_tiled: W[n][k] at ((n/16) K/8 + k/8) 128 +
// (k%8) 16 + n%16) as row-major bf16 [K][N] - the shared expert's block 384 of a weight batch.
inline std::vector<uint16_t> untile_bf16(const uint16_t* tiles, uint32_t K, uint32_t N) {
  std::vector<uint16_t> out(size_t(K) * N);
  const uint32_t K8 = K / 8;
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      out[size_t(k) * N + n] = tiles[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16];
  return out;
}
// kol_pf_dequant_gu: block e of the gate||up batch as bf16 [hidden][2I] (interleave16 kept).
inline std::vector<uint16_t> dequant_gu(const uint32_t* gu, const uint16_t* sh_gu, uint32_t e) {
  return e < kE ? pf_moe_ref::dequant_block(gu + size_t(e) * kolibri_ref::gate_up_block_words(), kH, 2 * kI)
                : untile_bf16(sh_gu, kH, 2 * kI);
}
// kol_pf_dequant_dn: block e of the down batch as bf16 [I][hidden].
inline std::vector<uint16_t> dequant_dn(const uint32_t* dn, const uint16_t* sh_dn, uint32_t e) {
  return e < kE ? pf_moe_ref::dequant_block(dn + size_t(e) * kolibri_ref::down_block_words(), kI, kH)
                : untile_bf16(sh_dn, kI, kH);
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

// kol_pf_moe_combine for every token of the chunk into mo [C][hidden]: `y` the down GEMM's sorted rows
// (rne(Σ), pitch hidden), the route rows [C][32].
inline void combine(const uint32_t* route, const Sorted& s, const uint16_t* y, uint16_t* mo, uint32_t C) {
  for (uint32_t t = 0; t < C; ++t) {
    const uint32_t* rr = route + size_t(t) * kolibri_ref::kWords;
    for (uint32_t n = 0; n < kH; ++n) {
      float acc = 0.0f;
      for (uint32_t j = 0; j < kK; ++j) {
        const float t_j = f32(y[size_t(s.pair_row[size_t(t) * kK + j]) * kH + n]) *
                          kolibri_ref::as_f32(rr[kolibri_ref::kWeights + j]);   // rounded to fp32
        acc = acc + t_j;                                                       // then added (no fma)
      }
      mo[size_t(t) * kH + n] = rne(acc + f32(y[size_t(s.shared_row() + t) * kH + n]));
    }
  }
}

// --- the attention ---------------------------------------------------------------------------------
// The caches are [rows][kv_heads][128] bf16: a sliding layer's ring (rows = 4096, key k at k & 4095) or a
// full layer's linear rows. q is fp32 [C][q_heads][128] (bf16 values, kol_attn_prep's), o [C][q_heads][128].
inline size_t key_at(uint32_t k, bool sliding, uint32_t j) {
  return (size_t(kolibri_ref::ring_row(k, sliding)) * kKVH + j) * kHd;
}
// The first key tile of the 8-row group whose first row sits at absolute position p0: an absolute
// multiple of KT - the window's (p0 - 512 rounded down) in a sliding layer, 0 in a full one.
inline uint32_t first_tile(uint32_t p0, bool sliding) {
  return sliding && p0 > kolibri_ref::kWindow - 1 ? (p0 - (kolibri_ref::kWindow - 1)) / kKt * kKt : 0;
}

// One (row t, head h) of the chunk at c0 by kol_pf_flash_attn's walk, in fp64: the row's group
// r0 = t & ~7 starts at first_tile(c0 + r0) and ends below min(c0 + C, c0 + r0 + 8); visible keys
// lo(p) <= k <= p; the online softmax with m_safe = (m == -INF ? 0 : m), so a wholly masked tile - even
// the row's first - leaves (m, l, o) exactly as they were.
inline std::vector<double> flash_row(const float* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0, uint32_t C,
                                     uint32_t t, bool sliding, uint32_t h) {
  const uint32_t j = h / (kQH / kKVH), r0 = t / kRpw * kRpw, p = c0 + t, lo = kolibri_ref::key_lo(p, sliding);
  const uint32_t last = std::min(c0 + C, c0 + r0 + kRpw);
  const float* qr = q + (size_t(t) * kQH + h) * kHd;
  double m = -INFINITY, l = 0.0;
  std::vector<double> o(kHd, 0.0);
  const double scale = 1.0 / std::sqrt(double(kHd));
  for (uint32_t t0 = first_tile(c0 + r0, sliding); t0 < last; t0 += kKt) {
    double s[kKt];
    double tmax = -INFINITY;
    for (uint32_t i = 0; i < kKt; ++i) {
      const uint32_t k = t0 + i;
      if (k < lo || k > p) {
        s[i] = -INFINITY;
        continue;
      }
      double a = 0;
      const size_t row = key_at(k, sliding, j);
      for (uint32_t d = 0; d < kHd; ++d) a += double(qr[d]) * f32(kc[row + d]);
      s[i] = a * scale;
      tmax = std::max(tmax, s[i]);
    }
    const double mnew = std::max(m, tmax), msafe = mnew == -INFINITY ? 0.0 : mnew;
    const double corr = std::exp(m - msafe);
    double psum = 0;
    for (uint32_t d = 0; d < kHd; ++d) o[d] *= corr;
    for (uint32_t i = 0; i < kKt; ++i) {
      const double pi = std::exp(s[i] - msafe);
      if (pi == 0.0) continue;
      psum += pi;
      const size_t row = key_at(t0 + i, sliding, j);
      for (uint32_t d = 0; d < kHd; ++d) o[d] += pi * f32(vc[row + d]);
    }
    l = l * corr + psum;
    m = mnew;
  }
  for (double& v : o) v /= l;
  return o;
}
inline void flash_window(const float* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0, uint32_t C, bool sliding,
                         double* o) {
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t h = 0; h < kQH; ++h) {
      const std::vector<double> r = flash_row(q, kc, vc, c0, C, t, sliding, h);
      std::copy(r.begin(), r.end(), o + (size_t(t) * kQH + h) * kHd);
    }
}

// The reference's rounding points for one (row t, head h): the visible keys lo..p in key order (the
// cached decode pass's row, kolibri_ref::attention_eager's - spec 20 §11's noted deviation for prompt
// rows past 512), scores and probabilities exactly kolibri_ref.h's, o = rne(Σ p v) as ONE ascending fp32
// chain (fma).
struct EagerRow {
  std::vector<float> s, p;
  std::vector<uint16_t> o;
};
inline EagerRow eager_row(const float* q, const uint16_t* kc, const uint16_t* vc, uint32_t p, bool sliding, uint32_t h) {
  const uint32_t j = h / (kQH / kKVH), lo = kolibri_ref::key_lo(p, sliding), L = p - lo + 1;
  const kolibri_ref::EagerHead e = kolibri_ref::attention_eager(q, kc, vc, lo, p, sliding, j, L);
  EagerRow r{e.s, e.p, std::vector<uint16_t>(kHd)};
  for (uint32_t d = 0; d < kHd; ++d) r.o[d] = rne(e.o[d]);
  return r;
}
inline void eager_window(const float* q, const uint16_t* kc, const uint16_t* vc, uint32_t c0, uint32_t C, bool sliding,
                         uint16_t* o) {
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t h = 0; h < kQH; ++h) {
      const EagerRow r = eager_row(q + (size_t(t) * kQH + h) * kHd, kc, vc, c0 + t, sliding, h);
      std::copy(r.o.begin(), r.o.end(), o + (size_t(t) * kQH + h) * kHd);
    }
}

}  // namespace kolibri_pf_ref
