#pragma once
// CPU twin of spec 19b's DFlash drafter kernels (src/kernels/dflash/df_*.cl) - the same op chains in the same orders,
// every function with a twin in a kernel; edit the two together. The semantics are tools/oracle/dflash_ref.py's (the
// D1 reference); tests/kernels/dflash_ref_test.cc holds this twin to that reference's own outputs
// (tests/loader/dflash_ref_fixture.safetensors, tools/oracle/dflash_fixture.py) in fp32 mode, and the kernels to
// this twin (tests/kernels/dflash_kernels_test.cc on the card, tools/mac/clrun/dflash_run.cc on the Mac's GPU).
//
//   norm()          df_norm: Σx² as WG lanes (i = lane, lane + WG, ... ascending) then a pairwise tree, rstd =
//                   1 / sqrt(Σ / H + eps), h = (x · rstd) · w in fp32, x = bf16(h)
//   norm2()         spec 23f, mg_df_norm (src/kernels/muse/mg_dflash.cl): the same Σ and rstd, then the Muse-Glimmer
//                   assistant's two roundings - x = bf16(w · bf16(x · rstd)), h = f32(x)
//   conv()          df_conv_prep / df_conv_add: out[r] = (base[0] + co[r][0][g]) x[r] + (base[1] + co[r][1][g]) x[r-1]
//                   (the second term only for r >= 1: one row of look-back inside the block), g = i / group
//   head_prep()     df_qkv_prep / df_ctx_store: a head's RMSNorm over head_dim (Σ² by a pairwise tree over the
//                   head_dim lanes) then neox RoPE from the table: y·cos - y'·sin (first half), y·cos + y'·sin
//   attention()     df_attn_part + df_attn_reduce: context keys in TGT blocks of PB ring offsets (offset t of block
//                   b is position p - RING + b·PB + t, visible to row r iff >= 0 and >= p + r - (WIN - 1)), each
//                   block's (m, l, acc) in ascending t; the reduce merges blocks ascending, then the block's own
//                   rows 0..K (non-causal), and divides once
//   silu_mul()      df_silu_mul: g / (1 + exp(-g)) · u
//   topk()          df_topk_stage1/2: the TOPK largest (value desc, then index asc) over indices < used, NaN never
//                   selected; values then · out_mult and, with a softcap, tanh(v / cap) · cap
//   select()        df_select: the walk - S[c] = unary[c] + Σ_r (pred[prev][r] · hid[r]) · succ[cand c][r] with the
//                   rank summed in TOPK x QS lane slices then the QS partials ascending; argmax (ties: lower c) or
//                   Gumbel-argmax: x = S / T - log(-log1p(-u)), u = murmur3(seed, p + j + 2^30, cand id) in fp32
//   gemv_*()        the reused GEMVs' arithmetic: gemv_i8w (Σ q·x ascending, then · s), gemv_bf16 (Σ w·x
//                   ascending; KSPLIT slices merged by the fixed pairwise tree), gemv_int4 (spec 23f: gemv.cl at
//                   S 1 - per 64-k group eight 8-k dots summed in order, the group's sum x its f16 scale, groups
//                   ascending)
//
// Spec 23f: Dims::muse is the Muse-Glimmer assistant's form (DFlashForm::Muse) - norm2 in Engine numerics, and the
// block is always all 16 rows (transformers' DFlashTokenCandidateGenerator: every row and every block key whatever K
// the caller reads; the engine's draft list runs at dctl.k = 15), the head raw (out_mult 1, no softcap), greedy.
//
// draft() chains them in the capture's order (runtime/dflash/dflash_capture.cc) over host copies of the weights in
// their device forms; with Numerics::Reference every activation stays fp32 and the GEMVs are fp32 dot products over
// the widened bf16 weights - dflash_ref.py's arithmetic up to summation order.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "common/bf16.h"

#pragma STDC FP_CONTRACT OFF

namespace dfref {

inline uint16_t rne(float f) {
  if (std::isnan(f)) return 0x7FC0;
  return common::f32_to_bf16(f);
}
inline float bf(uint16_t h) { return common::bf16_to_f32(h); }
inline float rf(float f) { return bf(rne(f)); }

constexpr uint32_t kNoIdx = 0x7FFFFFFFu;
constexpr uint32_t kDraftNoiseSalt = 1u << 30;

// The work-group sizes the kernels reduce with (src/kernels/dflash/*.cl), which fix the summation trees.
constexpr uint32_t kNormWg = 256;

// Σ x² as df_norm forms it: lane l sums x[i]^2 for i = l, l + wg, ... ascending, then strides wg/2 .. 1.
inline float wg_sumsq(const float* x, uint32_t n, uint32_t wg) {
  std::vector<float> s(wg, 0.0f);
  for (uint32_t l = 0; l < wg; ++l) {
    float a = 0.0f;
    for (uint32_t i = l; i < n; i += wg) a = a + x[i] * x[i];
    s[l] = a;
  }
  for (uint32_t stride = wg / 2; stride > 0; stride >>= 1)
    for (uint32_t l = 0; l < stride; ++l) s[l] = s[l] + s[l + stride];
  return s[0];
}

// df_norm, one row.
inline void norm(const float* src, const float* w, uint32_t H, float eps, float* h, uint16_t* x) {
  const float ss = wg_sumsq(src, H, kNormWg);
  const float rstd = 1.0f / std::sqrt(ss / float(H) + eps);
  for (uint32_t i = 0; i < H; ++i) {
    const float v = (src[i] * rstd) * w[i];
    if (h) h[i] = v;
    if (x) x[i] = rne(v);
  }
}

// mg_df_norm, one row (spec 23f): MuseGlimmerAssistantRMSNorm in bf16 - x-hat rounded, then w x it rounded.
inline void norm2(const float* src, const float* w, uint32_t H, float eps, float* h, uint16_t* x) {
  const float ss = wg_sumsq(src, H, kNormWg);
  const float rstd = 1.0f / std::sqrt(ss / float(H) + eps);
  for (uint32_t i = 0; i < H; ++i) {
    const uint16_t y = rne(w[i] * rf(src[i] * rstd));
    if (h) h[i] = bf(y);
    if (x) x[i] = y;
  }
}

// df_conv_prep / df_conv_add on rows [0, rows): co row r holds [side][tap][groups]; `co_side` points at side s of row
// 0 (co + s · taps · groups) and rows are `co_stride` floats apart; base is [tap][H] for that side.
inline void conv(const float* src, const float* co_side, uint32_t co_stride, const float* base, uint32_t rows,
                 uint32_t H, uint32_t group, uint32_t taps, float* out) {
  const uint32_t G = H / group;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t i = 0; i < H; ++i) {
      const uint32_t g = i / group;
      float o = (base[i] + co_side[size_t(r) * co_stride + g]) * src[size_t(r) * H + i];
      for (uint32_t t = 1; t < taps; ++t)
        if (r >= t)
          o = o + (base[size_t(t) * H + i] + co_side[size_t(r) * co_stride + size_t(t) * G + g]) *
                      src[size_t(r - t) * H + i];
      out[size_t(r) * H + i] = o;
    }
}

// RoPE table [pos][2][hd/2] (cos, sin): angle = pos · theta^(-2j/hd) in double, rounded to fp32 - dflash_ref.rope().
inline std::vector<float> rope_table(double theta, uint32_t hd, uint32_t rows) {
  const uint32_t half = hd / 2;
  std::vector<float> t(size_t(rows) * hd);
  for (uint32_t p = 0; p < rows; ++p)
    for (uint32_t j = 0; j < half; ++j) {
      const double inv = 1.0 / std::pow(theta, double(2 * j) / double(hd));
      const double a = double(p) * inv;
      t[size_t(p) * hd + j] = float(std::cos(a));
      t[size_t(p) * hd + half + j] = float(std::sin(a));
    }
  return t;
}

// One head: RMSNorm over hd (Σ² by the hd-lane tree), then neox RoPE at `pos` from the table (rope == nullptr: none).
inline void head_prep(const float* x, const float* w, uint32_t hd, float eps, const float* rope, uint32_t pos,
                      float* out) {
  std::vector<float> s(hd), y(hd);
  for (uint32_t i = 0; i < hd; ++i) s[i] = x[i] * x[i];
  for (uint32_t stride = hd / 2; stride > 0; stride >>= 1)
    for (uint32_t l = 0; l < stride; ++l) s[l] = s[l] + s[l + stride];
  const float rstd = 1.0f / std::sqrt(s[0] / float(hd) + eps);
  for (uint32_t i = 0; i < hd; ++i) y[i] = (x[i] * rstd) * w[i];
  if (!rope) {
    std::copy(y.begin(), y.end(), out);
    return;
  }
  const uint32_t half = hd / 2;
  const float* c = rope + size_t(pos) * hd;
  for (uint32_t i = 0; i < hd; ++i) {
    const uint32_t j = i % half;
    out[i] = i < half ? y[i] * c[j] - y[i + half] * c[half + j] : y[i] * c[j] + y[i - half] * c[half + j];
  }
}

// The attention's geometry.
struct AttnDims {
  uint32_t NH, NKV, HD, M, RING, WIN, PB;   // PB: ring offsets per context block (TGT = RING / PB)
  float scale;
  uint32_t tgt() const { return RING / PB; }
  uint32_t g() const { return NH / NKV; }
};

// df_attn_part + df_attn_reduce for one layer. q [M][NH][HD], kb / vb [M][NKV][HD] (fp32), ring_k / ring_v
// [RING][NKV][HD] (floats - the device's bf16 rows widened), anchor position p, K drafts (rows 0..K valid; rows past K
// are zero). Output out [M][NH·HD] fp32 (the kernel rounds it to bf16).
inline void attention(const AttnDims& a, const float* q, const float* kb, const float* vb, const float* ring_k,
                      const float* ring_v, uint32_t p, uint32_t K, bool causal, float* out,
                      const std::function<float(float)>& expf_ = [](float x) { return std::exp(x); }) {
  const uint32_t HD = a.HD, G = a.g();
  std::fill(out, out + size_t(a.M) * a.NH * HD, 0.0f);
  for (uint32_t h = 0; h < a.NH; ++h) {
    const uint32_t kvh = h / G;
    for (uint32_t r = 0; r < a.M && r <= K; ++r) {
      const float* qr = q + (size_t(r) * a.NH + h) * HD;
      // the blocks' partials
      std::vector<float> bm(a.tgt()), bl(a.tgt()), bacc(size_t(a.tgt()) * HD, 0.0f);
      for (uint32_t b = 0; b < a.tgt(); ++b) {
        std::vector<float> s(a.PB, -INFINITY);
        for (uint32_t t = 0; t < a.PB; ++t) {
          const uint32_t off = b * a.PB + t;
          const int64_t pos = int64_t(p) - int64_t(a.RING) + off;
          const int64_t lo = int64_t(p) + r - int64_t(a.WIN - 1);
          if (pos < 0 || pos < lo) continue;
          const uint32_t slot = uint32_t((uint64_t(p) + off) % a.RING);
          const float* k = ring_k + (size_t(slot) * a.NKV + kvh) * HD;
          float d = 0.0f;
          for (uint32_t i = 0; i < HD; ++i) d = d + qr[i] * k[i];
          s[t] = d * a.scale;
        }
        float m = -INFINITY;
        for (uint32_t t = 0; t < a.PB; ++t) m = std::fmax(m, s[t]);
        float l = 0.0f;
        std::vector<float> e(a.PB, 0.0f);
        if (m != -INFINITY)
          for (uint32_t t = 0; t < a.PB; ++t) {
            e[t] = s[t] == -INFINITY ? 0.0f : expf_(s[t] - m);
            l = l + e[t];
          }
        for (uint32_t d = 0; d < HD; ++d) {
          float acc = 0.0f;
          for (uint32_t t = 0; t < a.PB; ++t) {
            if (e[t] == 0.0f) continue;
            const uint32_t slot = uint32_t((uint64_t(p) + b * a.PB + t) % a.RING);
            acc = acc + e[t] * ring_v[(size_t(slot) * a.NKV + kvh) * HD + d];
          }
          bacc[size_t(b) * HD + d] = acc;
        }
        bm[b] = m;
        bl[b] = l;
      }
      // the block's own rows
      std::vector<float> sb(a.M, -INFINITY);
      for (uint32_t j = 0; j < a.M && j <= K; ++j) {
        if (causal && j > r) continue;
        const float* k = kb + (size_t(j) * a.NKV + kvh) * HD;
        float d = 0.0f;
        for (uint32_t i = 0; i < HD; ++i) d = d + qr[i] * k[i];
        sb[j] = d * a.scale;
      }
      float m = -INFINITY;
      for (uint32_t b = 0; b < a.tgt(); ++b) m = std::fmax(m, bm[b]);
      for (uint32_t j = 0; j < a.M; ++j) m = std::fmax(m, sb[j]);
      float l = 0.0f;
      std::vector<float> wb(a.tgt(), 0.0f), wj(a.M, 0.0f);
      for (uint32_t b = 0; b < a.tgt(); ++b) {
        wb[b] = bm[b] == -INFINITY ? 0.0f : expf_(bm[b] - m);
        l = l + bl[b] * wb[b];
      }
      for (uint32_t j = 0; j < a.M; ++j) {
        wj[j] = sb[j] == -INFINITY ? 0.0f : expf_(sb[j] - m);
        l = l + wj[j];
      }
      for (uint32_t d = 0; d < HD; ++d) {
        float acc = 0.0f;
        for (uint32_t b = 0; b < a.tgt(); ++b) acc = acc + bacc[size_t(b) * HD + d] * wb[b];
        for (uint32_t j = 0; j < a.M; ++j)
          if (wj[j] != 0.0f) acc = acc + wj[j] * vb[(size_t(j) * a.NKV + kvh) * HD + d];
        out[size_t(r) * a.NH * HD + size_t(h) * HD + d] = acc / l;
      }
    }
  }
}

// The exp the twin's SiLU uses (a test may swap it to price another implementation's ulps; default libm's expf).
inline float (*silu_exp)(float) = [](float x) { return std::exp(x); };
inline float silu_mul(float g, float u) { return g / (1.0f + silu_exp(-g)) * u; }

struct Cand {
  float v;
  uint32_t i;
};
inline bool better(const Cand& a, const Cand& b) { return a.v > b.v || (a.v == b.v && a.i < b.i); }

// The TOPK best of logits[0, used) (NaN never selected), then scaled / capped. `ids` maps an index to its id (the
// draft vocabulary's table; null: identity).
inline std::vector<Cand> topk(const float* logits, uint32_t used, uint32_t k, float mult, float softcap,
                              const uint32_t* ids = nullptr) {
  std::vector<Cand> c;
  for (uint32_t i = 0; i < used; ++i)
    if (!std::isnan(logits[i])) c.push_back({logits[i], i});
  std::sort(c.begin(), c.end(), better);
  c.resize(std::min<size_t>(c.size(), k));
  for (Cand& x : c) {
    float v = x.v * mult;
    if (softcap > 0.0f) v = std::tanh(v / softcap) * softcap;
    x.v = v;
    if (ids) x.i = ids[x.i];
  }
  while (c.size() < k) c.push_back({-INFINITY, kNoIdx});
  return c;
}

// murmur3 as vLLM's gumbel.py (dflash_ref.murmur3_uniform32): the uniform for (seed, pos, key) in fp32.
inline uint32_t rotl32(uint32_t v, int s) { return (v << s) | (v >> (32 - s)); }
inline uint32_t mm_mix(uint32_t h, uint32_t key) {
  key *= 0xCC9E2D51u;
  key = rotl32(key, 15);
  key *= 0x1B873593u;
  h ^= key;
  h = rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}
inline float murmur3_uniform(uint64_t seed, uint32_t pos, uint32_t key) {
  uint32_t h = 0;
  h = mm_mix(h, uint32_t(seed));
  h = mm_mix(h, uint32_t(seed >> 32));
  h = mm_mix(h, pos);
  h = mm_mix(h, key);
  h ^= 16u;
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return float(h >> 16) * 0x1p-16f + (float(h & 0xFFFFu) + 0.5f) * 0x1p-32f;
}

// One step of the walk's choice: argmax at T = 0 (ties: lower index; NaN never wins), else Gumbel-argmax.
inline uint32_t choose(const float* s, const uint32_t* cand, uint32_t k, float T, uint64_t seed, uint32_t pos) {
  uint32_t best = 0;
  float bv = -INFINITY;
  bool any = false;
  for (uint32_t c = 0; c < k; ++c) {
    float x = s[c];
    if (T != 0.0f) {
      const float u = murmur3_uniform(seed, pos + kDraftNoiseSalt, cand[c]);
      x = x / T - std::log(-std::log1p(-u));
    }
    if (!any ? !std::isnan(x) : x > bv) {
      bv = x;
      best = c;
      any = true;
    }
  }
  return best;
}

// df_select's edge score for one (row, predecessor id) over the TOPK candidates: P[r] = pred[prev][r] · hid[r],
// S[c] = unary[c] + Σ_q (Σ_{r in slice q} P[r] · succ[cand c][r]) with QS slices of rank / QS, ascending.
inline void edge_scores(const uint16_t* pred, const uint16_t* succ, uint32_t rank, uint32_t qs, uint32_t prev,
                        const float* hid, const uint32_t* cand, const float* unary, uint32_t k, float* s) {
  const uint32_t per = rank / qs;
  for (uint32_t c = 0; c < k; ++c) {
    float tot = 0.0f;
    for (uint32_t q = 0; q < qs; ++q) {
      float part = 0.0f;
      for (uint32_t r = q * per; r < (q + 1) * per; ++r) {
        const float P = bf(pred[size_t(prev) * rank + r]) * hid[r];
        part = part + P * (cand[c] == kNoIdx ? 0.0f : bf(succ[size_t(cand[c]) * rank + r]));
      }
      tot = tot + part;
    }
    s[c] = unary[c] + tot;
  }
}

// --- the reused GEMVs' arithmetic (x bf16 [M][K] row-major, weights in the device forms) -----------------------------
inline size_t i8_index(uint32_t K, uint32_t k, uint32_t n) {
  return (size_t(n / 16) * (K / 16) + k / 16) * 256 + size_t(k % 16) * 16 + n % 16;
}
inline size_t bf16_tiled_index(uint32_t K, uint32_t k, uint32_t n) {
  return ((size_t(n / 16) * (K / 8) + k / 8) * 8 + k % 8) * 16 + n % 16;
}
// The twin's GEMV threads (a test at the real shapes raises it; every output column is one thread's, so the result
// does not depend on it).
inline unsigned threads = 1;
template <class F>
inline void parallel_cols(uint32_t N, F f) {
  const unsigned T = std::max(1u, std::min(threads, N));
  if (T == 1) {
    f(0u, N);
    return;
  }
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < T; ++t)
    pool.emplace_back([&, t] { f(uint32_t(uint64_t(N) * t / T), uint32_t(uint64_t(N) * (t + 1) / T)); });
  for (std::thread& th : pool) th.join();
}

inline void gemv_i8w(const int8_t* q, const float* scales, uint32_t K, uint32_t N, const uint16_t* x, uint32_t M,
                     float* out) {
  std::vector<float> xf(size_t(M) * K);
  for (size_t i = 0; i < xf.size(); ++i) xf[i] = bf(x[i]);
  parallel_cols(N, [&](uint32_t n0, uint32_t n1) {
    for (uint32_t n = n0; n < n1; ++n) {
      const int8_t* col = q + size_t(n / 16) * (K / 16) * 256 + n % 16;   // + (k / 16) 256 + (k % 16) 16
      for (uint32_t m = 0; m < M; ++m) {
        const float* xr = &xf[size_t(m) * K];
        float a = 0.0f;
        for (uint32_t k = 0; k < K; ++k) a = a + float(col[size_t(k / 16) * 256 + (k % 16) * 16]) * xr[k];
        out[size_t(m) * N + n] = a * scales[n];
      }
    }
  });
}
inline void gemv_bf16(const uint16_t* w, uint32_t K, uint32_t N, uint32_t ksplit, const uint16_t* x, uint32_t M,
                      float* out) {
  const uint32_t k8_per = K / 8 / ksplit;
  std::vector<float> xf(size_t(M) * K);
  for (size_t i = 0; i < xf.size(); ++i) xf[i] = bf(x[i]);
  parallel_cols(N, [&](uint32_t n0, uint32_t n1) {
    std::vector<float> red(ksplit);
    for (uint32_t n = n0; n < n1; ++n) {
      const uint16_t* col = w + size_t(n / 16) * (K / 8) * 128 + n % 16;   // + (k / 8) 128 + (k % 8) 16
      for (uint32_t m = 0; m < M; ++m) {
        const float* xr = &xf[size_t(m) * K];
        for (uint32_t qq = 0; qq < ksplit; ++qq) {
          float a = 0.0f;
          for (uint32_t k = qq * k8_per * 8; k < (qq + 1) * k8_per * 8; ++k)
            a = a + bf(col[size_t(k / 8) * 128 + (k % 8) * 16]) * xr[k];
          red[qq] = a;
        }
        for (uint32_t stride = ksplit / 2; stride > 0; stride >>= 1)
          for (uint32_t qq = 0; qq < stride; ++qq) red[qq] = red[qq] + red[qq + stride];
        out[size_t(m) * N + n] = red[0];
      }
    }
  });
}

// gemv.cl (int4 g64, LAYOUT 1 or 0 - the same sums) at S 1 from the GPTQ-native form: qweight [K/8][N] (word (r, n)
// holds k = 8r .. 8r + 7, k = 8r lowest) and f16 scales [K/64][N] - loader::rtn_int4_g64's output.
inline void gemv_int4(const uint32_t* qw, const uint16_t* scales, uint32_t K, uint32_t N, const uint16_t* x,
                      uint32_t M, float* out) {
  std::vector<float> xf(size_t(M) * K);
  for (size_t i = 0; i < xf.size(); ++i) xf[i] = bf(x[i]);
  parallel_cols(N, [&](uint32_t n0, uint32_t n1) {
    for (uint32_t n = n0; n < n1; ++n)
      for (uint32_t m = 0; m < M; ++m) {
        const float* xr = &xf[size_t(m) * K];
        float acc = 0.0f;
        for (uint32_t g = 0; g < K / 64; ++g) {
          float gacc = 0.0f;
          for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t word = qw[size_t(g * 8 + j) * N + n];
            float a = 0.0f;
            for (uint32_t t = 0; t < 8; ++t)
              a = a + float(int((word >> (4 * t)) & 0xFu) - 8) * xr[g * 64 + j * 8 + t];
            gacc = gacc + a;
          }
          acc = acc + gacc * common::f16_to_f32(scales[size_t(g) * N + n]);
        }
        out[size_t(m) * N + n] = acc;
      }
  });
}

// --- the whole drafter --------------------------------------------------------------------------------------------
enum class Numerics { Engine, Reference };

struct Dims {
  uint32_t version = 2, layers = 0, H = 0, NH = 0, NKV = 0, HD = 0, I = 0;
  uint32_t V = 0, used = 0, M = 8, ring = 0, win = 0, pb = 0, taps = 0, th = 0;
  uint32_t group = 0, ctaps = 0, rank = 0, topk = 16, qs = 16;
  float eps = 1e-6f, in_scale = 1.0f, out_mult = 1.0f, softcap = 0.0f, scale = 0.0f;
  double theta = 1e6;
  bool causal = false;
  bool muse = false;   // spec 23f: the Muse-Glimmer assistant's form (the header's note)
  uint32_t qn() const { return NH * HD; }
  uint32_t kvn() const { return NKV * HD; }
  uint32_t qkvn() const { return qn() + 2 * kvn(); }
  uint32_t groups() const { return group ? H / group : 0; }
  uint32_t kpn() const { return 2 * ctaps * groups(); }
  uint32_t fck() const { return taps * th; }
  AttnDims attn() const { return {NH, NKV, HD, M, ring, win, pb, scale}; }
};

// A linear in either device form (one of bf16 / i8 non-empty) or, for Numerics::Reference, row-major fp32 [N][K].
struct Lin {
  uint32_t K = 0, N = 0, ksplit = 1;
  std::vector<uint16_t> bf16;
  std::vector<int8_t> i8;
  std::vector<float> scales;
  std::vector<float> f32;   // row-major [N][K] (Reference)
  std::vector<uint32_t> q4; // spec 23f: int4 g64 qweight [K/8][N] + f16 scales [K/64][N] (gemv_int4)
  std::vector<uint16_t> s4;
  // y [M][N] = W x for M rows; xb the bf16 rows (Engine), xf the fp32 rows (Reference).
  void apply(const uint16_t* xb, const float* xf, uint32_t M, float* y) const {
    if (!f32.empty()) {
      parallel_cols(N, [&](uint32_t n0, uint32_t n1) {
        for (uint32_t m = 0; m < M; ++m)
          for (uint32_t n = n0; n < n1; ++n) {
            float a = 0.0f;
            for (uint32_t k = 0; k < K; ++k) a = a + f32[size_t(n) * K + k] * xf[size_t(m) * K + k];
            y[size_t(m) * N + n] = a;
          }
      });
    } else if (!q4.empty()) {
      gemv_int4(q4.data(), s4.data(), K, N, xb, M, y);
    } else if (!i8.empty()) {
      gemv_i8w(i8.data(), scales.data(), K, N, xb, M, y);
    } else {
      gemv_bf16(bf16.data(), K, N, ksplit, xb, M, y);
    }
  }
};

struct Layer {
  Lin qkv, o, gate_up, down, akp, mkp;
  std::vector<float> in_norm, post_norm, q_norm, k_norm;
  std::vector<float> conv_base;   // [attn, mlp][side][tap][H]
};

struct Weights {
  Dims d;
  Lin fc, sel, head;              // head: the target's lm_head ([V][H], or the compact one)
  std::vector<Layer> layers;
  std::vector<float> hidden_norm, final_norm, mask_row;
  std::vector<uint16_t> embed;    // the target's bf16 [V][H]
  std::vector<uint16_t> pred, succ;
  std::vector<uint32_t> head_ids; // compact head: index -> id (empty: identity)
};

// The drafter's per-layer context ring, positions at slot pos % ring (floats: bf16-widened under Engine).
struct Ring {
  std::vector<float> k, v;   // [layers][ring][NKV][HD]
};

inline Ring make_ring(const Dims& d) {
  Ring r;
  r.k.assign(size_t(d.layers) * d.ring * d.kvn(), 0.0f);
  r.v.assign(r.k.size(), 0.0f);
  return r;
}

// The context projection of `n` tap rows (bf16 [n][taps·th]) at positions pos0 .. pos0 + n - 1 into the ring.
inline void project(const Weights& w, Numerics nm, const uint16_t* taps, uint32_t n, uint32_t pos0,
                    const std::vector<float>& rope, Ring& ring) {
  const Dims& d = w.d;
  std::vector<float> tf(size_t(n) * d.fck());
  for (size_t i = 0; i < tf.size(); ++i) tf[i] = bf(taps[i]);
  std::vector<float> f(size_t(n) * d.H), h(size_t(n) * d.H);
  std::vector<uint16_t> x(size_t(n) * d.H);
  w.fc.apply(taps, tf.data(), n, f.data());
  const bool eng = nm == Numerics::Engine;
  for (uint32_t r = 0; r < n; ++r)
    (d.muse && eng ? norm2 : norm)(&f[size_t(r) * d.H], w.hidden_norm.data(), d.H, d.eps, &h[size_t(r) * d.H],
                                   &x[size_t(r) * d.H]);
  for (uint32_t l = 0; l < d.layers; ++l) {
    std::vector<float> qkv(size_t(n) * d.qkvn());
    w.layers[l].qkv.apply(x.data(), h.data(), n, qkv.data());   // the k||v rows are the q||k||v rows [qn, qkvn)
    for (uint32_t r = 0; r < n; ++r) {
      const uint32_t pos = pos0 + r, slot = pos % d.ring;
      for (uint32_t kh = 0; kh < d.NKV; ++kh) {
        float* kd = &ring.k[((size_t(l) * d.ring + slot) * d.NKV + kh) * d.HD];
        float* vd = &ring.v[((size_t(l) * d.ring + slot) * d.NKV + kh) * d.HD];
        head_prep(&qkv[size_t(r) * d.qkvn() + d.qn() + kh * d.HD], w.layers[l].k_norm.data(), d.HD, d.eps,
                  rope.data(), pos, kd);
        const float* vs = &qkv[size_t(r) * d.qkvn() + d.qn() + d.kvn() + kh * d.HD];
        for (uint32_t i = 0; i < d.HD; ++i) {
          vd[i] = eng ? rf(vs[i]) : vs[i];
          if (eng) kd[i] = rf(kd[i]);
        }
      }
    }
  }
}

struct DraftOut {
  std::vector<float> hidden;            // [M][H] the final-norm rows (fp32)
  std::vector<uint32_t> cand;           // [K][topk]
  std::vector<float> unary, scores;     // [K][topk]
  std::vector<uint32_t> chosen, ids;    // [K]
};

// One draft block at anchor `anchor_id`, position p, K drafts, over the ring.
inline DraftOut draft(const Weights& w, Numerics nm, const Ring& ring, const std::vector<float>& rope,
                      uint32_t anchor_id, uint32_t p, uint32_t K, float T = 0.0f, uint64_t seed = 0) {
  const Dims& d = w.d;
  const bool eng = nm == Numerics::Engine, v2 = d.version == 2;
  const uint32_t M = d.M, H = d.H;
  std::vector<float> resid(size_t(M) * H), h(size_t(M) * H), xf(size_t(M) * H);
  std::vector<uint16_t> x(size_t(M) * H);
  for (uint32_t r = 0; r < M; ++r)
    for (uint32_t i = 0; i < H; ++i)
      resid[size_t(r) * H + i] = (r == 0 ? bf(w.embed[size_t(anchor_id) * H + i]) : w.mask_row[i]) * d.in_scale;
  // A GEMV input: the rows as bf16 (Engine) / fp32 (Reference).
  auto as_input = [&](const std::vector<float>& src, uint32_t width, std::vector<uint16_t>& xb) {
    xb.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) xb[i] = rne(src[i]);
    (void)width;
  };
  auto norm_rows = [&](const std::vector<float>& src, const std::vector<float>& wt) {
    for (uint32_t r = 0; r < M; ++r)
      (d.muse && eng ? norm2 : norm)(&src[size_t(r) * H], wt.data(), H, d.eps, &h[size_t(r) * H], &x[size_t(r) * H]);
  };
  const uint32_t Ka = d.muse ? M - 1 : K;   // the rows the attention runs (Muse: the whole block)
  const AttnDims ad = d.attn();
  std::vector<float> co, tmp, qkv(size_t(M) * d.qkvn()), q(size_t(M) * d.qn()), kb(size_t(M) * d.kvn()),
      vb(size_t(M) * d.kvn()), ao(size_t(M) * d.qn()), oo(size_t(M) * H), gu(size_t(M) * 2 * d.I),
      a(size_t(M) * d.I), dn(size_t(M) * H);
  std::vector<uint16_t> xb;
  std::vector<float> ring_k_l, ring_v_l;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const Layer& L = w.layers[l];
    // attention sub-block
    norm_rows(resid, L.in_norm);
    std::vector<float> cin = h;   // the conv's / q||k||v's input (fp32)
    if (v2) {
      co.assign(size_t(M) * d.kpn(), 0.0f);
      L.akp.apply(x.data(), h.data(), M, co.data());
      tmp.assign(size_t(M) * H, 0.0f);
      conv(h.data(), co.data(), d.kpn(), &L.conv_base[(0 * 2 + 0) * size_t(d.ctaps) * H], M, H, d.group, d.ctaps,
           tmp.data());
      cin = tmp;
    }
    as_input(cin, H, xb);
    L.qkv.apply(xb.data(), cin.data(), M, qkv.data());
    for (uint32_t r = 0; r < M; ++r) {
      for (uint32_t hh = 0; hh < d.NH; ++hh)
        head_prep(&qkv[size_t(r) * d.qkvn() + hh * d.HD], L.q_norm.data(), d.HD, d.eps, rope.data(), p + r,
                  &q[(size_t(r) * d.NH + hh) * d.HD]);
      for (uint32_t kh = 0; kh < d.NKV; ++kh) {
        head_prep(&qkv[size_t(r) * d.qkvn() + d.qn() + kh * d.HD], L.k_norm.data(), d.HD, d.eps, rope.data(), p + r,
                  &kb[(size_t(r) * d.NKV + kh) * d.HD]);
        std::memcpy(&vb[(size_t(r) * d.NKV + kh) * d.HD], &qkv[size_t(r) * d.qkvn() + d.qn() + d.kvn() + kh * d.HD],
                    d.HD * 4);
      }
    }
    const size_t lay = size_t(d.ring) * d.kvn();
    attention(ad, q.data(), kb.data(), vb.data(), &ring.k[l * lay], &ring.v[l * lay], p, Ka, d.causal, ao.data());
    as_input(ao, d.qn(), xb);
    L.o.apply(xb.data(), ao.data(), M, oo.data());
    if (v2) {
      tmp.assign(size_t(M) * H, 0.0f);
      conv(oo.data(), co.data() + size_t(d.ctaps) * d.groups(), d.kpn(), &L.conv_base[(0 * 2 + 1) * size_t(d.ctaps) * H],
           M, H, d.group, d.ctaps, tmp.data());
      for (size_t i = 0; i < resid.size(); ++i) resid[i] = resid[i] + tmp[i];
    } else {
      for (size_t i = 0; i < resid.size(); ++i) resid[i] = resid[i] + oo[i];
    }
    // MLP sub-block
    norm_rows(resid, L.post_norm);
    cin = h;
    if (v2) {
      co.assign(size_t(M) * d.kpn(), 0.0f);
      L.mkp.apply(x.data(), h.data(), M, co.data());
      tmp.assign(size_t(M) * H, 0.0f);
      conv(h.data(), co.data(), d.kpn(), &L.conv_base[(1 * 2 + 0) * size_t(d.ctaps) * H], M, H, d.group, d.ctaps,
           tmp.data());
      cin = tmp;
    }
    as_input(cin, H, xb);
    L.gate_up.apply(xb.data(), cin.data(), M, gu.data());
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t i = 0; i < d.I; ++i)
        a[size_t(r) * d.I + i] = silu_mul(gu[size_t(r) * 2 * d.I + i], gu[size_t(r) * 2 * d.I + d.I + i]);
    as_input(a, d.I, xb);
    L.down.apply(xb.data(), a.data(), M, dn.data());
    if (v2) {
      tmp.assign(size_t(M) * H, 0.0f);
      conv(dn.data(), co.data() + size_t(d.ctaps) * d.groups(), d.kpn(), &L.conv_base[(1 * 2 + 1) * size_t(d.ctaps) * H],
           M, H, d.group, d.ctaps, tmp.data());
      for (size_t i = 0; i < resid.size(); ++i) resid[i] = resid[i] + tmp[i];
    } else {
      for (size_t i = 0; i < resid.size(); ++i) resid[i] = resid[i] + dn[i];
    }
  }
  norm_rows(resid, w.final_norm);
  DraftOut out;
  out.hidden = h;
  // the head over rows 1..K (the engine runs rows 1..M-1; rows past K are never read)
  const uint32_t R = M - 1;
  std::vector<float> logits(size_t(R) * w.head.N);
  std::vector<float> hrows(h.begin() + H, h.end());
  w.head.apply(x.data() + H, hrows.data(), R, logits.data());
  std::vector<float> hid;
  if (v2) {
    hid.assign(size_t(M) * d.rank, 0.0f);
    w.sel.apply(x.data(), h.data(), M, hid.data());
  }
  const uint32_t used = w.head_ids.empty() ? d.used : w.head.N;
  uint32_t prev = anchor_id;
  for (uint32_t j = 0; j < K; ++j) {
    const std::vector<Cand> c = topk(&logits[size_t(j) * w.head.N], used, d.topk, d.out_mult, d.softcap,
                                     w.head_ids.empty() ? nullptr : w.head_ids.data());
    std::vector<uint32_t> cid(d.topk);
    std::vector<float> un(d.topk), s(d.topk);
    for (uint32_t i = 0; i < d.topk; ++i) {
      cid[i] = c[i].i;
      un[i] = c[i].v;
    }
    if (v2)
      edge_scores(w.pred.data(), w.succ.data(), d.rank, d.qs, prev, &hid[size_t(1 + j) * d.rank], cid.data(),
                  un.data(), d.topk, s.data());
    else
      s = un;
    const uint32_t ch = choose(s.data(), cid.data(), d.topk, v2 ? T : 0.0f, seed, p + j);
    out.cand.insert(out.cand.end(), cid.begin(), cid.end());
    out.unary.insert(out.unary.end(), un.begin(), un.end());
    out.scores.insert(out.scores.end(), s.begin(), s.end());
    out.chosen.push_back(ch);
    out.ids.push_back(cid[ch]);
    prev = cid[ch];
  }
  return out;
}

// Cosine of two equal-length vectors (fp64).
inline double cosine(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return aa == 0 && bb == 0 ? 1.0 : ab / std::sqrt(aa * bb);
}

}  // namespace dfref
