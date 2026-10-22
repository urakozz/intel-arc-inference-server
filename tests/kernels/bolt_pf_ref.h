#pragma once
// CPU references for spec 25d's Nemotron 3.5 Bolt prefill kernels (src/kernels/bolt/nb_pf_{mamba,moe,moe_gemm,prep}.cl) -
// the same op chains in the same orders, every line with a twin in a kernel; EDIT THE TWO TOGETHER. Built on 25c's
// tests/kernels/bolt_ref.h (namespace boltref: the decode chains, the transcendentals exp_t / log1p_t, the halving tree),
// which it includes and does not change - so "the prefill applies decode's per-row chain" (the conv, the route, the MoE)
// is a comparison between two functions of this directory (tests/kernels/bolt_pf_ref_test.cc), not a claim.
//
// **The chunk.** C <= 2048 rows at absolute positions pos .. pos + C - 1. The linears are ONE unsplit GEMM each (S 1): their
// fp32 rows are the consumers' input, summed from 0.0f exactly as decode's S = 1 slices (0.0f + v - it turns a -0.0 into
// +0.0, as decode does). in_proj's rows sit at pitch kInLd = pad256(18560) = 18688 (pf_gemm writes a zero-padded tail slab,
// runtime/bolt/bolt_prefill.cc).
//
// **The SSD sub-chunks are ABSOLUTE** (plan 25d Review Focus 1): sub-chunk k of a chunk covers the chunk's rows whose
// absolute position lies in one 64-aligned window - the first one is partial when pos % 64 != 0, the last when the chunk
// ends inside a window. A chunk boundary at a multiple of 64 is therefore a sub-chunk boundary, and a split there changes
// no sub-chunk's composition: the state after each sub-chunk and every row's y are the same bits however a prompt is cut at
// multiples of 64 (runtime::bolt::prefill_chunk_rows makes every chunk end on that grid or at the prompt end).
//
// The SSD form per head h (group g = h / 16), sub-chunk rows s, t (absolute-local indices i, j in [0, 64)):
//   dt_t   = softplus(f32(rne(0 + dt_raw_t)) + dt_bias)            decode's chain (threshold 20, no clamp)
//   a_t    = A · dt_t;  cum_t = a_{t0} + ... + a_t                 sequential fp32 from the sub-chunk's first row IN THIS
//                                                                 chunk (a partial first sub-chunk starts at the chunk)
//   CB     = C_t · B_s (n ascending from 0.0f; bf16 x bf16 products are exact) - nb_pf_cb; its DPAS build sums in the
//            hardware's order (a bar, never bitwise)
//   M[t][s]= (CB[t][s] · exp(cum_t - cum_s)) · dt_s   for s <= t - the MASK BEFORE THE EXP (s > t is 0, never exp'd)
//   Yd     = M[t][s0] x_s0 + M[t][s0+1] x_s0+1 + ... (sequential, from the first term)
//   Yo     = tree_n(S_start[p][n] · C_t[n]) · exp(cum_t)          the halving tree of decode's y; S_start fp32
//   y      = rne((Yd + Yo) + x_t · D)
//   S_end  = S_start · exp(cum_L) + acc,  acc = (B_s0[n] w_s0) x_s0[p] + ... (sequential from the first term),
//            w_s = exp(cum_L - cum_s) · dt_s,  L the sub-chunk's last row in this chunk
// so a ONE-row sub-chunk updates the state bitwise as decode's step does (exp(0) = 1: w = dt, S · exp(A dt) + (B dt) x).
// The fp32 state never meets DPAS (the GDN record, pf_gdn_scan.cl): every product with it is on the vector path.
//
//   seed()          nb_pf_seed: the ring's raw xBC of positions pos - 3 .. pos - 1 into seed [3][10240] (0 below 0)
//   conv_chunk()    nb_pf_conv: decode's conv_step row by row - the history from the seed for the chunk's first three
//                   rows, recomputed from in_proj's rows after (a causal conv has no recurrence); the chunk's last
//                   min(C, 3) raw columns into their ring slots
//   dt_cum()        nb_pf_dt: dt and the per-sub-chunk cumulative sums, [C][128] each
//   cb()            nb_pf_cb (plain build): C B^T per (sub-chunk, group), [nsub][8][64][64], 0 where a row is outside
//   ssd()           nb_pf_ssd over a head range (the twin's speed knob; the kernel runs all 128)
//   gnorm_chunk()   nb_pf_gnorm: decode's gnorm at in_proj's prefill pitch
//   recurrence()    decode's nb_ssm chain row by row (boltref::ssm_step's arithmetic) over a head range - what the SSD
//                   form must equal (Review Focus 2); ssd_f64() / recurrence_f64(): the same two in double (std::exp)
//   sort() / gather() / grouped_up() / grouped_down() / combine()   nb_pf_moe / nb_pf_moe_gemm: every (row, slot) pair one
//                   row, expert-major, ascending row inside an expert (stable), each expert padded to whole 32-row tiles;
//                   the grouped GEMMs' twin uses DECODE's dot order (moe_ref::split_dot over the int4 tiles) so the twin
//                   prefill MoE equals boltref::moe per row BITWISE (Review Focus 3) - on the card the GEMMs are DPAS
//                   over bf16-dequantised blocks (a cosine bar against this, bitwise against pf_gemm on the same rows)
//   prefill_chunk_rows()   runtime::bolt's chunk rule (mirrored in runtime/bolt/bolt_sizes.h; bolt_plan_test holds them equal)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kernels/bolt_ref.h"
#include "kernels/pf_moe_ref.h"   // dequant_block (layout 1)

namespace boltpf {

using boltref::f32;
using boltref::rf;
using boltref::rne;
namespace br = boltref;

constexpr uint32_t kPfC = 2048, kSub = 64, kTm = 32, kNone = 0xFFFFFFFFu;
constexpr uint32_t kInLd = 18688;                  // pad256(18560): in_proj's prefill rows' pitch
constexpr uint32_t kX = br::kXbc, kInner = br::kInner, kHeads = br::kHeads, kP = br::kHeadDim, kN = br::kState;
constexpr uint32_t kG = br::kGroups, kHpg = kHeads / kG;
constexpr uint32_t kMaxSub = (kSub - 1 + kPfC + kSub - 1) / kSub;   // 33: a chunk of 2048 starting off the grid
// The sort's header (nb_pf_moe.cl H_*, kernels::bolt::pf_hdr): [0] tiles used, [1] 0, [2] rows used = 32 x tiles, [3] C,
// [4 + e] expert e's rows (unpadded), e < 512.
constexpr uint32_t kHdrTiles = 0, kHdrRows = 2, kHdrC = 3, kHdrCount = 4, kMaxE = 512;
constexpr uint32_t kHdrWords = (kHdrCount + kMaxE + 15) / 16 * 16;   // 528

// --- the chunk's sub-chunks --------------------------------------------------------------------------------------
inline uint32_t n_sub(uint32_t pos, uint32_t C) { return (pos % kSub + C + kSub - 1) / kSub; }
struct Span {
  uint32_t r0, r1;   // the chunk's rows [r0, r1)
  uint32_t u0;       // r0's absolute-local index (pos + r0) % 64
};
inline Span sub_span(uint32_t pos, uint32_t C, uint32_t k) {
  const uint32_t off = pos % kSub;
  const uint32_t r0 = k == 0 ? 0u : k * kSub - off;
  const uint32_t r1 = std::min(C, (k + 1) * kSub - off);
  return {r0, r1, k == 0 ? off : 0u};
}
// Column `col` of in_proj's prefill row t, summed as decode sums one slice (0.0f + v).
inline float col1(const float* P, uint32_t t, uint32_t col, uint32_t ld = kInLd) {
  float v = 0.0f;
  v += P[size_t(t) * ld + col];
  return v;
}

// --- the conv over a chunk -----------------------------------------------------------------------------------------
inline std::vector<uint16_t> seed(const uint16_t* ring, uint32_t pos) {
  std::vector<uint16_t> s(size_t(3) * kX, 0);
  for (uint32_t j = 0; j < 3; ++j) {
    const int p = int(pos) - 3 + int(j);
    if (p < 0) continue;
    for (uint32_t c = 0; c < kX; ++c) s[size_t(j) * kX + c] = ring[size_t(uint32_t(p) % br::kRing) * kX + c];
  }
  return s;
}
// P [C][ld] fp32 (in_proj's rows), seed [3][10240], cw [10240][4] (tap 0 the OLDEST), cb [10240]. Returns xc [C][10240];
// writes the raw xBC of rows max(0, C - 3) .. C - 1 into `ring` slots (pos + t) % 16.
inline std::vector<uint16_t> conv_chunk(const float* P, uint32_t C, const uint16_t* sd, const float* cw, const float* cb,
                                        uint16_t* ring, uint32_t pos, uint32_t ld = kInLd) {
  std::vector<uint16_t> xc(size_t(C) * kX);
  for (uint32_t c = 0; c < kX; ++c) {
    auto raw = [&](int t) -> float {
      return t < 0 ? f32(sd[size_t(t + 3) * kX + c]) : f32(rne(col1(P, uint32_t(t), br::kXbcOff + c, ld)));
    };
    for (uint32_t t = 0; t < C; ++t) {
      float acc = cb[c];
      acc = acc + cw[size_t(c) * 4 + 0] * raw(int(t) - 3);
      acc = acc + cw[size_t(c) * 4 + 1] * raw(int(t) - 2);
      acc = acc + cw[size_t(c) * 4 + 2] * raw(int(t) - 1);
      acc = acc + cw[size_t(c) * 4 + 3] * raw(int(t));
      xc[size_t(t) * kX + c] = rne(br::silu_conv(acc));
    }
    for (uint32_t t = C >= 3 ? C - 3 : 0; t < C; ++t)
      ring[size_t((pos + t) % br::kRing) * kX + c] = rne(col1(P, t, br::kXbcOff + c, ld));
  }
  return xc;
}

// --- dt and the cumulative sums ------------------------------------------------------------------------------------
struct DtCum {
  std::vector<float> dt, cum;   // [C][128]
};
inline DtCum dt_cum(const float* P, uint32_t C, uint32_t pos, const float* A, const float* dt_bias, uint32_t ld = kInLd) {
  DtCum o;
  o.dt.assign(size_t(C) * kHeads, 0.0f);
  o.cum.assign(size_t(C) * kHeads, 0.0f);
  for (uint32_t k = 0; k < n_sub(pos, C); ++k) {
    const Span sp = sub_span(pos, C, k);
    for (uint32_t h = 0; h < kHeads; ++h) {
      float cum = 0.0f;
      for (uint32_t t = sp.r0; t < sp.r1; ++t) {
        const float dt = br::softplus_t(f32(rne(col1(P, t, br::kDtOff + h, ld))) + dt_bias[h]);
        const float a = A[h] * dt;
        cum = t == sp.r0 ? a : cum + a;
        o.dt[size_t(t) * kHeads + h] = dt;
        o.cum[size_t(t) * kHeads + h] = cum;
      }
    }
  }
  return o;
}

// --- C B^T per (sub-chunk, group): the plain build's order ---------------------------------------------------------
inline size_t cb_at(uint32_t k, uint32_t g, uint32_t i, uint32_t j) { return ((size_t(k) * kG + g) * kSub + i) * kSub + j; }
inline std::vector<float> cb(const uint16_t* xc, uint32_t C, uint32_t pos) {
  const uint32_t ns = n_sub(pos, C), off = pos % kSub;
  std::vector<float> o(size_t(ns) * kG * kSub * kSub, 0.0f);
  for (uint32_t k = 0; k < ns; ++k)
    for (uint32_t g = 0; g < kG; ++g)
      for (uint32_t i = 0; i < kSub; ++i) {
        const int ti = int(k * kSub + i) - int(off);
        if (ti < 0 || ti >= int(C)) continue;
        for (uint32_t j = 0; j < kSub; ++j) {
          const int tj = int(k * kSub + j) - int(off);
          if (tj < 0 || tj >= int(C)) continue;
          const uint16_t* Cr = xc + size_t(ti) * kX + br::kCOff + g * kN;
          const uint16_t* Br = xc + size_t(tj) * kX + br::kBOff + g * kN;
          float acc = 0.0f;
          for (uint32_t n = 0; n < kN; ++n) acc = acc + f32(Cr[n]) * f32(Br[n]);
          o[cb_at(k, g, i, j)] = acc;
        }
      }
  return o;
}

// --- the SSD scan (nb_pf_ssd) --------------------------------------------------------------------------------------
// Heads [h0, h1): y [C][8192] (other heads' columns untouched), `state` [128][64][128] fp32 updated in place; `state16`
// non-null: the bf16-state arm - the state read widened and rounded ONCE at the end of the chunk (Decision 2's rule for
// prefill: not inside the scan). `yraw` (tests): y before its rounding.
inline void ssd(const uint16_t* xc, uint32_t C, uint32_t pos, const DtCum& dc, const float* cbm, const float* D,
                float* state, uint16_t* y, uint32_t h0 = 0, uint32_t h1 = kHeads, uint16_t* state16 = nullptr,
                float* yraw = nullptr) {
  const uint32_t ns = n_sub(pos, C);
  std::vector<float> S(size_t(kP) * kN), Snew(size_t(kP) * kN), M(size_t(kSub) * kSub), w(kSub), prod(kN);
  for (uint32_t h = h0; h < h1; ++h) {
    const uint32_t g = h / kHpg;
    for (size_t i = 0; i < S.size(); ++i)
      S[i] = state16 ? f32(state16[size_t(h) * kP * kN + i]) : state[size_t(h) * kP * kN + i];
    for (uint32_t k = 0; k < ns; ++k) {
      const Span sp = sub_span(pos, C, k);
      auto cum = [&](uint32_t t) { return dc.cum[size_t(t) * kHeads + h]; };
      auto dtv = [&](uint32_t t) { return dc.dt[size_t(t) * kHeads + h]; };
      auto xv = [&](uint32_t t, uint32_t p) { return f32(xc[size_t(t) * kX + size_t(h) * kP + p]); };
      // M over the sub-chunk's rows (absolute-local i = u0 + (t - r0)); s > t is 0 and never exp'd
      for (uint32_t t = sp.r0; t < sp.r1; ++t)
        for (uint32_t s = sp.r0; s <= t; ++s) {
          const uint32_t i = sp.u0 + (t - sp.r0), j = sp.u0 + (s - sp.r0);
          M[size_t(i) * kSub + j] = (cbm[cb_at(k, g, i, j)] * br::exp_t(cum(t) - cum(s))) * dtv(s);
        }
      for (uint32_t t = sp.r0; t < sp.r1; ++t) {
        const uint32_t i = sp.u0 + (t - sp.r0);
        const uint16_t* Ct = xc + size_t(t) * kX + br::kCOff + g * kN;
        const float et = br::exp_t(cum(t));
        for (uint32_t p = 0; p < kP; ++p) {
          float yd = M[size_t(i) * kSub + sp.u0] * xv(sp.r0, p);
          for (uint32_t s = sp.r0 + 1; s <= t; ++s) yd = yd + M[size_t(i) * kSub + sp.u0 + (s - sp.r0)] * xv(s, p);
          for (uint32_t n = 0; n < kN; ++n) prod[n] = S[size_t(p) * kN + n] * f32(Ct[n]);
          const float yo = br::tree_halving(prod) * et;
          const float v = (yd + yo) + xv(t, p) * D[h];
          y[size_t(t) * kInner + size_t(h) * kP + p] = rne(v);
          if (yraw) yraw[size_t(t) * kInner + size_t(h) * kP + p] = v;
        }
      }
      const uint32_t L = sp.r1 - 1;
      const float decay = br::exp_t(cum(L));
      for (uint32_t s = sp.r0; s <= L; ++s) w[s - sp.r0] = br::exp_t(cum(L) - cum(s)) * dtv(s);
      for (uint32_t p = 0; p < kP; ++p)
        for (uint32_t n = 0; n < kN; ++n) {
          float acc = (f32(xc[size_t(sp.r0) * kX + br::kBOff + g * kN + n]) * w[0]) * xv(sp.r0, p);
          for (uint32_t s = sp.r0 + 1; s <= L; ++s)
            acc = acc + (f32(xc[size_t(s) * kX + br::kBOff + g * kN + n]) * w[s - sp.r0]) * xv(s, p);
          Snew[size_t(p) * kN + n] = S[size_t(p) * kN + n] * decay + acc;
        }
      S.swap(Snew);
    }
    for (size_t i = 0; i < S.size(); ++i) {
      if (state16)
        state16[size_t(h) * kP * kN + i] = rne(S[i]);
      else
        state[size_t(h) * kP * kN + i] = S[i];
    }
  }
}

// --- the gated norm at the prefill pitch (nb_pf_gnorm) --------------------------------------------------------------
inline std::vector<uint16_t> gnorm_chunk(const float* P, uint32_t C, const uint16_t* y, const float* w, uint32_t ld = kInLd) {
  std::vector<uint16_t> out(size_t(C) * kInner);
  std::vector<float> v(br::kGnormGroup), sq(br::kGnormGroup);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t g = 0; g < kInner / br::kGnormGroup; ++g) {
      for (uint32_t i = 0; i < br::kGnormGroup; ++i) {
        const uint32_t c = g * br::kGnormGroup + i;
        v[i] = f32(y[size_t(t) * kInner + c]) * br::silu_gate(f32(rne(col1(P, t, c, ld))));
        sq[i] = v[i] * v[i];
      }
      const float rstd = 1.0f / std::sqrt(br::tree_halving(sq) / float(br::kGnormGroup) + br::kEps);
      for (uint32_t i = 0; i < br::kGnormGroup; ++i)
        out[size_t(t) * kInner + g * br::kGnormGroup + i] = rne((v[i] * rstd) * w[g * br::kGnormGroup + i]);
    }
  return out;
}

// --- decode's recurrence over the same rows (Review Focus 2) -------------------------------------------------------
// boltref::ssm_step's chain one row at a time over heads [h0, h1): dt from in_proj's rows, dA = exp(A dt), dB = B dt,
// S = S dA + dB x, y = rne(tree(S C) + x D). bolt_pf_ref_test holds it to boltref::ssm_step bitwise on every head.
inline void recurrence(const float* P, const uint16_t* xc, uint32_t C, const float* A, const float* D,
                       const float* dt_bias, float* state, uint16_t* y, uint32_t h0 = 0, uint32_t h1 = kHeads,
                       uint32_t ld = kInLd, float* yraw = nullptr) {
  std::vector<float> st(kN), prod(kN);
  for (uint32_t h = h0; h < h1; ++h) {
    const uint32_t g = h / kHpg;
    for (uint32_t p = 0; p < kP; ++p) {
      float* row = state + (size_t(h) * kP + p) * kN;
      for (uint32_t n = 0; n < kN; ++n) st[n] = row[n];
      for (uint32_t t = 0; t < C; ++t) {
        const float dt = br::softplus_t(f32(rne(col1(P, t, br::kDtOff + h, ld))) + dt_bias[h]);
        const float dA = br::exp_t(A[h] * dt);
        const uint16_t* xr = xc + size_t(t) * kX;
        const float x = f32(xr[size_t(h) * kP + p]);
        for (uint32_t n = 0; n < kN; ++n) {
          const float dB = f32(xr[br::kBOff + g * kN + n]) * dt;
          st[n] = st[n] * dA + dB * x;
          prod[n] = st[n] * f32(xr[br::kCOff + g * kN + n]);
        }
        const float v = br::tree_halving(prod) + x * D[h];
        y[size_t(t) * kInner + size_t(h) * kP + p] = rne(v);
        if (yraw) yraw[size_t(t) * kInner + size_t(h) * kP + p] = v;
      }
      for (uint32_t n = 0; n < kN; ++n) row[n] = st[n];
    }
  }
}

// --- the two in double: the algebra (Review Focus 2: equal to 1e-12) ------------------------------------------------
// Inputs as the fp32 forms see them (x, B, C bf16 values; dt the fp32 twin's dt widened; A), every operation in double,
// y UNROUNDED [C][heads h1 - h0][64], the state [heads][64][128] in double, updated.
inline std::vector<double> recurrence_f64(const uint16_t* xc, uint32_t C, const std::vector<float>& dt, const float* A,
                                          const float* D, std::vector<double>& state, uint32_t h0, uint32_t h1) {
  const uint32_t nh = h1 - h0;
  std::vector<double> y(size_t(C) * nh * kP);
  for (uint32_t h = h0; h < h1; ++h) {
    const uint32_t g = h / kHpg;
    for (uint32_t p = 0; p < kP; ++p) {
      double* st = state.data() + (size_t(h - h0) * kP + p) * kN;
      for (uint32_t t = 0; t < C; ++t) {
        const double d = dt[size_t(t) * kHeads + h], dA = std::exp(double(A[h]) * d);
        const uint16_t* xr = xc + size_t(t) * kX;
        const double x = f32(xr[size_t(h) * kP + p]);
        double acc = 0.0;
        for (uint32_t n = 0; n < kN; ++n) {
          st[n] = st[n] * dA + double(f32(xr[br::kBOff + g * kN + n])) * d * x;
          acc += st[n] * double(f32(xr[br::kCOff + g * kN + n]));
        }
        y[(size_t(t) * nh + (h - h0)) * kP + p] = acc + x * double(D[h]);
      }
    }
  }
  return y;
}
// `direct`: exp's argument for a pair (t, s) is the segment sum a_{s+1} + ... + a_t itself; otherwise cum_t - cum_s (the
// kernels' and the reference's form: cs[t] - cs[s]), whose cancellation costs |cum| x 2^-52 of the exponent in double
// (|cum| x 2^-24 in fp32) - bolt_pf_ref_test prints both.
inline std::vector<double> ssd_f64(const uint16_t* xc, uint32_t C, uint32_t pos, const std::vector<float>& dt,
                                   const float* A, const float* D, std::vector<double>& state, uint32_t h0, uint32_t h1,
                                   bool direct = false) {
  const uint32_t nh = h1 - h0, ns = n_sub(pos, C);
  std::vector<double> y(size_t(C) * nh * kP), cum(C), a(C);
  auto seg = [&](uint32_t t, uint32_t s) {
    if (!direct) return cum[t] - cum[s];
    double v = 0.0;
    for (uint32_t u = s + 1; u <= t; ++u) v += a[u];
    return v;
  };
  for (uint32_t h = h0; h < h1; ++h) {
    const uint32_t g = h / kHpg;
    double* S = state.data() + size_t(h - h0) * kP * kN;
    for (uint32_t k = 0; k < ns; ++k) {
      const Span sp = sub_span(pos, C, k);
      for (uint32_t t = sp.r0; t < sp.r1; ++t) {
        a[t] = double(A[h]) * double(dt[size_t(t) * kHeads + h]);
        cum[t] = t == sp.r0 ? a[t] : cum[t - 1] + a[t];
      }
      for (uint32_t t = sp.r0; t < sp.r1; ++t) {
        const uint16_t* xt = xc + size_t(t) * kX;
        for (uint32_t p = 0; p < kP; ++p) {
          double yd = 0.0, yo = 0.0;
          for (uint32_t s = sp.r0; s <= t; ++s) {
            const uint16_t* xs = xc + size_t(s) * kX;
            double cbv = 0.0;
            for (uint32_t n = 0; n < kN; ++n) cbv += double(f32(xt[br::kCOff + g * kN + n])) * f32(xs[br::kBOff + g * kN + n]);
            yd += cbv * std::exp(seg(t, s)) * double(dt[size_t(s) * kHeads + h]) * f32(xs[size_t(h) * kP + p]);
          }
          for (uint32_t n = 0; n < kN; ++n) yo += S[size_t(p) * kN + n] * double(f32(xt[br::kCOff + g * kN + n]));
          y[(size_t(t) * nh + (h - h0)) * kP + p] = yd + yo * std::exp(cum[t]) + double(f32(xt[size_t(h) * kP + p])) * D[h];
        }
      }
      const uint32_t L = sp.r1 - 1;
      for (uint32_t p = 0; p < kP; ++p)
        for (uint32_t n = 0; n < kN; ++n) {
          double acc = 0.0;
          for (uint32_t s = sp.r0; s <= L; ++s) {
            const uint16_t* xs = xc + size_t(s) * kX;
            acc += std::exp(seg(L, s)) * double(dt[size_t(s) * kHeads + h]) * f32(xs[size_t(h) * kP + p]) *
                   f32(xs[br::kBOff + g * kN + n]);
          }
          S[size_t(p) * kN + n] = S[size_t(p) * kN + n] * std::exp(cum[L]) + acc;
        }
    }
  }
  return y;
}

// --- the grouped MoE ---------------------------------------------------------------------------------------------
// The tile table's bound (nb_pf_moe.cl; runtime::bolt::pf_tiles is its host home): Σ_e ceil(c_e / 32) <= floor((9 C +
// E x 31) / 32) - 1056 at C 2048, E 496.
inline uint32_t tmax(uint32_t C, uint32_t E) { return (C * br::kTopK + E * (kTm - 1)) / kTm; }
struct Sorted {
  std::vector<uint32_t> hdr, tiles, row_tok, pair_row;
  uint32_t rows_used() const { return hdr[kHdrRows]; }
  uint32_t tiles_used() const { return hdr[kHdrTiles]; }
};
// nb_pf_sort over route rows [C][48] (ids at kIds; an id >= E clamps to E - 1, as the kernel's staging does - nb_route
// never writes one).
inline Sorted sort(const uint32_t* route, uint32_t C, uint32_t E) {
  const uint32_t T = tmax(C, E);
  std::vector<uint32_t> id(size_t(C) * br::kTopK);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < br::kTopK; ++j)
      id[size_t(t) * br::kTopK + j] = std::min(route[size_t(t) * br::kWords + br::kIds + j], E - 1);
  std::vector<uint32_t> cnt(E, 0), toff(E + 1, 0);
  for (uint32_t v : id) ++cnt[v];
  for (uint32_t e = 0; e < E; ++e) toff[e + 1] = toff[e] + (cnt[e] + kTm - 1) / kTm;
  const uint32_t ntiles = toff[E];
  Sorted s;
  s.hdr.assign(kHdrWords, 0);
  s.tiles.assign(size_t(2) * T, 0);
  s.row_tok.assign(size_t(T) * kTm, kNone);
  s.pair_row.assign(size_t(C) * br::kTopK, 0);
  for (uint32_t e = 0; e < E; ++e) {
    for (uint32_t i = toff[e]; i < toff[e + 1]; ++i)
      if (i < T) {
        s.tiles[2 * i] = e;
        s.tiles[2 * i + 1] = i * kTm;
      }
    uint32_t r = toff[e] * kTm;
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < br::kTopK; ++j)
        if (id[size_t(t) * br::kTopK + j] == e) {
          s.row_tok[r] = t;
          s.pair_row[size_t(t) * br::kTopK + j] = r;
          ++r;
        }
    s.hdr[kHdrCount + e] = cnt[e];
  }
  for (uint32_t i = ntiles; i < T; ++i) {
    s.tiles[2 * i] = kNone;
    s.tiles[2 * i + 1] = 0;
  }
  s.hdr[kHdrTiles] = ntiles;
  s.hdr[kHdrRows] = ntiles * kTm;
  s.hdr[kHdrC] = C;
  return s;
}
// nb_pf_gather: row r of the sorted A operand = lat[row_tok[r]] (1024 bf16), zeros for a padding row.
inline std::vector<uint16_t> gather(const Sorted& s, const uint16_t* lat) {
  std::vector<uint16_t> xg(size_t(s.rows_used()) * br::kLatent, 0);
  for (uint32_t r = 0; r < s.rows_used(); ++r)
    if (s.row_tok[r] != kNone)
      std::copy(lat + size_t(s.row_tok[r]) * br::kLatent, lat + size_t(s.row_tok[r] + 1) * br::kLatent,
                xg.begin() + size_t(r) * br::kLatent);
  return xg;
}
// The expert of sorted row r (its tile's block).
inline uint32_t row_expert(const Sorted& s, uint32_t r) { return s.tiles[2 * (r / kTm)]; }
// The twin's grouped up: a[r][i] = relu2(rne(up_e(xg_r))) in DECODE's dot order (split_dot over the int4 tiles); rows of
// padding hold zeros in, zeros out.
inline std::vector<uint16_t> grouped_up(const Sorted& s, const uint16_t* xg, const uint32_t* blocks, uint32_t I) {
  std::vector<uint16_t> a(size_t(s.rows_used()) * I, 0);
  for (uint32_t r = 0; r < s.rows_used(); ++r) {
    const uint32_t* up = blocks + size_t(row_expert(s, r)) * br::expert_block_words(I);
    for (uint32_t i = 0; i < I; ++i)
      a[size_t(r) * I + i] = br::relu2(rne(moe_ref::split_dot(up, br::kLatent, i, xg + size_t(r) * br::kLatent, br::kMoeUpKs)));
  }
  return a;
}
// The twin's grouped down: y[r][n] = rne(down_e(a_r)) in decode's dot order.
inline std::vector<uint16_t> grouped_down(const Sorted& s, const uint16_t* a, const uint32_t* blocks, uint32_t I) {
  std::vector<uint16_t> y(size_t(s.rows_used()) * br::kLatent, 0);
  for (uint32_t r = 0; r < s.rows_used(); ++r) {
    const uint32_t* dn = blocks + size_t(row_expert(s, r)) * br::expert_block_words(I) + br::expert_up_words(I);
    for (uint32_t n = 0; n < br::kLatent; ++n)
      y[size_t(r) * br::kLatent + n] = rne(moe_ref::split_dot(dn, I, n, a + size_t(r) * I, br::kMoeDnKs));
  }
  return y;
}
// nb_pf_combine: token t's latent r[t][n] = rne(Σ over the nine slots in ASCENDING id (the route row's order) of
// f32(y[pair_row[t][k]][n]) · w_k), fp32 from 0.0f, product then add - decode's nb_moe_down epilogue.
inline std::vector<uint16_t> combine(const uint32_t* route, const Sorted& s, const uint16_t* y, uint32_t C) {
  std::vector<uint16_t> r(size_t(C) * br::kLatent);
  for (uint32_t t = 0; t < C; ++t) {
    const uint32_t* rr = route + size_t(t) * br::kWords;
    for (uint32_t n = 0; n < br::kLatent; ++n) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < br::kTopK; ++k) {
        const float v = f32(y[size_t(s.pair_row[size_t(t) * br::kTopK + k]) * br::kLatent + n]) * br::as_f32(rr[br::kW + k]);
        acc = acc + v;
      }
      r[size_t(t) * br::kLatent + n] = rne(acc);
    }
  }
  return r;
}
// The expert blocks of nb_pf_dequant_up / _dn: expert e's up as bf16 [1024][I], its down as bf16 [I][1024].
inline std::vector<uint16_t> dequant_up(const uint32_t* blocks, uint32_t e, uint32_t I) {
  return pf_moe_ref::dequant_block(blocks + size_t(e) * br::expert_block_words(I), br::kLatent, I);
}
inline std::vector<uint16_t> dequant_dn(const uint32_t* blocks, uint32_t e, uint32_t I) {
  return pf_moe_ref::dequant_block(blocks + size_t(e) * br::expert_block_words(I) + br::expert_up_words(I), I, br::kLatent);
}

// --- the chunk rule (runtime::bolt::prefill_chunk_rows' mirror) ------------------------------------------------------
// At most `chunk` rows (a multiple of 64), ending on the absolute 64-grid unless the prompt ends first; with a block hook
// never past the next multiple of `block` (2048: on the grid too).
inline uint32_t prefill_chunk_rows(uint32_t pos, size_t remaining, uint32_t chunk, bool hooked, uint32_t block = 2048) {
  uint32_t c = uint32_t(std::min<size_t>(chunk, remaining));
  if (c < remaining && (pos + c) % kSub != 0) c -= (pos + c) % kSub;
  if (hooked) c = std::min(c, block - pos % block);
  return c;
}

}  // namespace boltpf
