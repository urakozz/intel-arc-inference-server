#pragma once
// CPU references for spec 21c's Qwen3.8-Flash-Next kernels (src/kernels/qwen4exp/*.cl, prep.cl's
// prep_gated_head _SIG) - the same op chains in the same orders, every line with a twin in a kernel; edit
// the two together. The chains are tools/oracle/qwen4exp_ref.py's restated ops (spec 21a, transformers
// 5.19.0's qwen4_exp; docs/probe-qwen4exp-2026-10-09.md "The rounding chain"), op for op, at every rounding
// point; tests/kernels/qwen4exp_ref_test.cc holds them to that reference's own outputs
// (tests/kernels/qwen4exp_fixture.h) and to the semantics' properties (plan 21c Review Focus 1-6).
//
//   hc_combine_norm()  q4_hc_combine_norm: the pending H = H0 + rne(y x inj) of the block before (product
//                      rounded, then the add), or the embedding repeated (layer 0), then the grouped (1 + w)
//                      norm per stream (sum of squares: 256 lanes, k = i + 256 j ascending with fma, the
//                      pairwise tree; 1 / sqrt, never rsqrt) -> xn
//   hc_up_mix()        q4_hc_up_mix: down's bf16 rows, / 4, silu -> a [320]; the up linear (fma chain over k
//                      ascending), sigmoid -> g; mean over the 4 streams of rne(g x xn) ((s0 + s1) + s2) + s3,
//                      / 4, one rounding -> the block input x; inj = 2 rne(sigmoid(rne(inject / 4)))
//   ple_ids() / ple_row() / ple_block()   q4_ple_gather / q4_ple_block (spec 21 §2.4)
//   qsa_q() / qsa_block_key() / qsa_score() / qsa_select()   q4_qsa_prep / _score / _select (§2.3)
//   qsa_attn_eager()   q4_qsa_attn_eager: the reference's bf16 chain over the list
//   qsa_attn_fp64()    decode attention over the list in fp64 (flash's tolerance reference)
//   route() / gate_up() / down() / combine()   q4_route / q4_moe_gate_up / q4_moe_down (§2.1, grouped_mm)
//   gated_head_sig()   prep_gated_head _SIG: Qwen4ExpTextRMSNormGated with the sigmoid gate
//
// **exp:** every sigmoid / SiLU is 1 / (1 + exp_torch(-x)) / x / (1 + exp_torch(-x)) on both sides
// (k2_ref::exp_torch = Sleef's expf u10, torch's Vectorized<float>::exp), so the kernels and this file agree
// bit for bit; the GEMV-fed values are exact where the GEMV is (the tests feed the kernels' own GEMV
// outputs). Not bit-exact against torch: the fp32 ACCUMULATION orders of torch's reductions (its matmul,
// its sums, its softmax's sum), which are its own - the fixture's coarse inputs make every such sum exact in
// any order, so there the comparison is of the rounding points alone.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "kernels/k2_ref.h"    // exp_torch, eager_softmax (torch's fp32 softmax), lowbias32
#include "kernels/moe_ref.h"   // f32 / rne / rf, the layout-1 tile_dot / split_dot, gate_col / up_col
#include "loader/qwen4exp_ple_hash.h"

namespace q4ref {

using moe_ref::f32;
using moe_ref::rf;
using moe_ref::rne;

constexpr float kEps = 1e-6f;
constexpr uint32_t kHidden = 2560, kHc = 4, kHcN = 10240, kHcLow = 320, kHcDownRows = 336, kInj = 4;
constexpr uint32_t kQHeads = 24, kKvHeads = 2, kHd = 256, kGqa = 12, kRotHalf = 32, kQkvgN = 13312;
constexpr uint32_t kIdxHeads = 4, kIdxDim = 128, kIdxN = 640, kBlock = 4, kTopBlocks = 512;
constexpr uint32_t kListMax = 2052, kTailSlots = 8;
constexpr uint32_t kExperts = 512, kTopK = 10, kInter = 640, kRouterN = 528, kSlots = 11;
constexpr uint32_t kPleHeads = 16, kPleDim = 160, kPleRing = 16, kPleEos = 248044;
constexpr uint32_t kGdnHd = 128, kGdnHeads = 48, kQkvzN = 16384, kZOff = 10240;
// The route row (q4_moe.cl R_*, kernels::qwen4exp::route).
constexpr uint32_t kWords = 32, kIds = 0, kWeights = 16, kSharedGate = 26, kP10 = 27, kP11 = 28;
// fp32 constants torch divides by: (float) math.sqrt(2560) (the PLE gate), (float) math.sqrt(128) (the indexer).
constexpr float kSqrtHidden = 50.596442562694070f, kSqrtIdx = 11.313708498984761f;
constexpr float kAttnScale = 0.0625f;   // 256 ** -0.5

inline float exp_torch(float x) { return k2_ref::exp_torch(x); }
inline float sigmoid_t(float x) { return 1.0f / (1.0f + exp_torch(-x)); }
inline float silu_t(float x) { return x / (1.0f + exp_torch(-x)); }

inline float as_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint32_t as_u32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// --- the fixture's inputs and its hex arrays (tools/oracle/qwen4exp_fixture.py val()) ---------------------------
// Element i of input tensor t. coarse: (h % 17 - 8) x step; fine: ((h >> 8) x 2^-23 - 1) x amp, rounded to bf16.
inline float fixture_val(bool coarse, uint32_t t, uint32_t i, uint32_t seed, float step = 0.125f, float amp = 1.0f) {
  const uint32_t h = k2_ref::lowbias32(k2_ref::lowbias32(seed * 3 + t) ^ i);
  if (coarse) return float(int(h % 17) - 8) * step;
  const float f = (float(h >> 8) * 0x1p-23f - 1.0f) * amp;
  return rf(f);
}
inline std::vector<uint16_t> fixture_bf16(bool coarse, uint32_t t, uint32_t n, uint32_t seed, float step = 0.125f,
                                          float amp = 1.0f, uint32_t base = 0) {
  std::vector<uint16_t> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = rne(fixture_val(coarse, t, base + i, seed, step, amp));
  return v;
}
inline std::vector<uint16_t> hex16(const char* s) {
  std::vector<uint16_t> v;
  const std::string h(s);
  for (size_t i = 0; i + 4 <= h.size(); i += 4) v.push_back(uint16_t(std::stoul(h.substr(i, 4), nullptr, 16)));
  return v;
}
inline std::vector<uint32_t> hex32(const char* s) {
  std::vector<uint32_t> v;
  const std::string h(s);
  for (size_t i = 0; i + 8 <= h.size(); i += 8) v.push_back(uint32_t(std::stoul(h.substr(i, 8), nullptr, 16)));
  return v;
}
inline std::vector<uint64_t> hex64(const char* s) {
  std::vector<uint64_t> v;
  const std::string h(s);
  for (size_t i = 0; i + 16 <= h.size(); i += 16) v.push_back(std::stoull(h.substr(i, 16), nullptr, 16));
  return v;
}

// --- the norms ------------------------------------------------------------------------------------------------
// rstd of n bf16-valued floats over a `wg`-lane work-group: lane i accumulates fma(v, v, acc) over k = i, i + wg,
// ... ascending, then the pairwise tree wg/2 .. 1; 1 / sqrt(sum / n + eps) (correctly rounded on the device).
inline float rstd_tree(const float* v, uint32_t n, uint32_t wg) {
  std::vector<float> red(wg, 0.0f);
  for (uint32_t i = 0; i < wg; ++i) {
    float acc = 0.0f;
    for (uint32_t k = i; k < n; k += wg) acc = std::fma(v[k], v[k], acc);
    red[i] = acc;
  }
  for (uint32_t stride = wg / 2; stride > 0; stride >>= 1)
    for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
  return 1.0f / std::sqrt(red[0] / float(n) + kEps);
}
// One (1 + w) RMSNorm element: rne(x · rstd · w) - (x · rstd) first, then · w (fp32), one rounding.
inline uint16_t norm_elem(float x, float rstd, float w) { return rne(x * rstd * w); }
// A grouped (per-stream) norm over [4][2560] bf16: rstd per stream (256 lanes), w [10240] fp32 (1 + w).
inline void grouped_norm(const uint16_t* in, const float* w, uint16_t* out) {
  std::vector<float> v(kHidden);
  for (uint32_t s = 0; s < kHc; ++s) {
    for (uint32_t k = 0; k < kHidden; ++k) v[k] = f32(in[s * kHidden + k]);
    const float rstd = rstd_tree(v.data(), kHidden, 256);
    for (uint32_t k = 0; k < kHidden; ++k) out[s * kHidden + k] = norm_elem(v[k], rstd, w[s * kHidden + k]);
  }
}

// --- hyper-connections (spec 21 §2.2, §4.1) -------------------------------------------------------------------
enum class HcSrc { Embed, Y, None };   // kernels::qwen4exp::HcSrc (Slices: the caller forms y, y_of_slices)
// y = rne(sum_s slices[s][m][k]) ascending s: a GEMV's split-K slices -> its bf16 output.
inline uint16_t y_of_slices(const float* slices, uint32_t S, uint32_t M, uint32_t m, uint32_t N, uint32_t k) {
  float v = 0.0f;
  for (uint32_t s = 0; s < S; ++s) v += slices[(size_t(s) * M + m) * N + k];
  return rne(v);
}
// q4_hc_combine_norm for one row. H [10240] bf16 in / out. Embed: H[s][k] = src[k]; Y: H[s][k] =
// rne(f32(H) + rf(f32(src[k]) · inj[s])); None: H unchanged. Then (xn non-null) xn = grouped norm(H).
inline void hc_combine_norm(uint16_t* H, HcSrc src, const uint16_t* y, const float* inj, const float* w, uint16_t* xn) {
  for (uint32_t s = 0; s < kHc; ++s)
    for (uint32_t k = 0; k < kHidden; ++k) {
      uint16_t& h = H[s * kHidden + k];
      if (src == HcSrc::Embed) h = y[k];
      else if (src == HcSrc::Y) h = rne(f32(h) + rf(f32(y[k]) * inj[s]));
    }
  if (xn) grouped_norm(H, w, xn);
}
// gemv_bf16 tiles [n_tile][k_octet][8 k][16 n]: element (k, n) of a {K, N} weight.
inline size_t tile_index(uint32_t K, uint32_t k, uint32_t n) {
  return ((size_t(n / 16) * (K / 8) + k / 8) * 8 + k % 8) * 16 + n % 16;
}
// The activation of down's 320 rows: a_j = rne(silu(f32(rne(f32(rne(d_j)) / 4)))).
inline float hc_act(float down_f32) { return rf(silu_t(rf(rf(down_f32) / 4.0f))); }
// inj_s = rne(2 · rne(sigmoid(f32(rne(f32(rne(d_{320+s})) / 4))))).
inline float hc_inj(float down_f32) { return rf(2.0f * rf(sigmoid_t(rf(rf(down_f32) / 4.0f)))); }
// q4_hc_up_mix for one row: down_f32 [324] (inject) or [320], up tiles {K 320, N 10240}, xn [10240];
// x [2560] out; inj [4] out (inject only).
inline void hc_up_mix(const float* down_f32, bool inject, const uint16_t* up, const uint16_t* xn, uint16_t* x, float* inj) {
  float a[kHcLow];
  for (uint32_t j = 0; j < kHcLow; ++j) a[j] = hc_act(down_f32[j]);
  if (inject)
    for (uint32_t s = 0; s < kInj; ++s) inj[s] = hc_inj(down_f32[kHcLow + s]);
  for (uint32_t c = 0; c < kHidden; ++c) {
    float p[kHc];
    for (uint32_t s = 0; s < kHc; ++s) {
      const uint32_t n = s * kHidden + c;
      float acc = 0.0f;
      for (uint32_t k = 0; k < kHcLow; ++k) acc = std::fma(a[k], f32(up[tile_index(kHcLow, k, n)]), acc);
      const float g = rf(sigmoid_t(rf(acc)));
      p[s] = rf(g * f32(xn[n]));
    }
    x[c] = rne((((p[0] + p[1]) + p[2]) + p[3]) / 4.0f);
  }
}

// --- PLE (spec 21 §2.4) -------------------------------------------------------------------------------------------
// The id-ring rule (21a's reading of M:1107-1121): t1 = id[p - 1] (EOS if p < 1); t2 = t1 == EOS ? EOS :
// id[p - 2] (EOS if p < 2). `id(q)` is the token at position q <= p.
template <class Id>
inline loader::Q4PleHistory ple_history(uint32_t p, Id id, uint32_t eos = kPleEos) {
  loader::Q4PleHistory h;
  h.t0 = id(p);
  h.t1 = p >= 1 ? id(p - 1) : eos;
  h.t2 = (h.t1 == eos || p < 2) ? eos : id(p - 2);
  return h;
}
// Row r of a head as the kernel dequantises it: rne(float(q) · float(s)).
inline void ple_row(const int8_t* q, float scale, uint16_t* e) {
  for (uint32_t i = 0; i < kPleDim; ++i) e[i] = rne(float(q[i]) * scale);
}
// q4_ple_block for one row. H [10240] in / out; kv [12800] the key||value GEMV's fp32 row; wk / wq / wc
// [10240] fp32 (1 + w) (norm_key, norm_query, norm_conv); taps [10240][4] fp32 (the dilated conv's weights);
// hist[3] the conv ring's rows of positions p - 9, p - 6, p - 3 (bf16 [10240], zero before position 0);
// ring_out [10240] gets this row's normed gated value (the ring's slot p).
struct PleTrace {   // the intermediates, for the tests
  float s[kHc] = {}, gate[kHc] = {}, sg[kHc] = {};
};
inline PleTrace ple_block(uint16_t* H, const float* kv, const float* wk, const float* wq, const float* wc, const float* taps,
                          const uint16_t* const hist[3], uint16_t* ring_out) {
  PleTrace t;
  std::vector<uint16_t> key(kHcN), keyn(kHcN), qn(kHcN), gated(kHcN), gatedn(kHcN);
  for (uint32_t c = 0; c < kHcN; ++c) key[c] = rne(kv[c]);
  grouped_norm(key.data(), wk, keyn.data());
  grouped_norm(H, wq, qn.data());
  const uint16_t eps_b = rne(1e-6f);
  for (uint32_t s = 0; s < kHc; ++s) {
    // (key_normed x query_normed).sum(-1): each product rounded, the sum fp32 (256 lanes ascending, the tree)
    std::vector<float> red(256, 0.0f);
    for (uint32_t i = 0; i < 256; ++i) {
      float acc = 0.0f;
      for (uint32_t k = i; k < kHidden; k += 256) acc += rf(f32(keyn[s * kHidden + k]) * f32(qn[s * kHidden + k]));
      red[i] = acc;
    }
    for (uint32_t stride = 128; stride > 0; stride >>= 1)
      for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
    const float sb = rf(rf(red[0]) / kSqrtHidden);
    const float a = std::fabs(sb);
    const float c = a < f32(eps_b) ? f32(eps_b) : a;        // clamp_min(1e-6) in bf16
    const float r = rf(std::sqrt(c));
    const float g = sb > 0.0f ? r : (sb < 0.0f ? -r : 0.0f);   // gate.sign() x sqrt: exact
    const float sg = rf(sigmoid_t(g));
    t.s[s] = sb;
    t.gate[s] = g;
    t.sg[s] = sg;
    for (uint32_t k = 0; k < kHidden; ++k) gated[s * kHidden + k] = rne(sg * rf(kv[kHcN + k]));
  }
  grouped_norm(gated.data(), wc, gatedn.data());
  for (uint32_t c = 0; c < kHcN; ++c) {
    const float xs[4] = {hist[0] ? f32(hist[0][c]) : 0.0f, hist[1] ? f32(hist[1][c]) : 0.0f,
                         hist[2] ? f32(hist[2][c]) : 0.0f, f32(gatedn[c])};
    float acc = 0.0f;
    for (uint32_t j = 0; j < 4; ++j) acc = std::fma(xs[j], taps[size_t(c) * 4 + j], acc);
    const float conv = rf(silu_t(rf(acc)));
    const float out = rf(f32(gated[c]) + conv);
    H[c] = rne(f32(H[c]) + out);
    ring_out[c] = gatedn[c];
  }
  return t;
}

// --- QSA (spec 21 §2.3) -------------------------------------------------------------------------------------------
// The bf16 RoPE torch applies on the first 64 dims (cos / sin cast to bf16 first; q·cos and rot(q)·sin each
// rounded, their sum rounded): `cs` = the loader's fp32 table row [2][32] (cos then sin) at the position.
inline void rope64_bf16(const uint16_t* n, const float* cs, uint32_t dim, uint16_t* out) {
  for (uint32_t i = 0; i < dim; ++i) {
    if (i >= 2 * kRotHalf) {
      out[i] = n[i];
      continue;
    }
    const uint32_t ii = i % kRotHalf;
    const float c = rf(cs[ii]), s = rf(cs[kRotHalf + ii]);
    const float rot = i < kRotHalf ? -f32(n[i + kRotHalf]) : f32(n[i - kRotHalf]);
    out[i] = rne(rf(f32(n[i]) * c) + rf(rot * s));
  }
}
// The 128-dim head norm of q4_qsa_prep: 128 lanes each x_i^2 (a plain multiply), the tree 64 .. 1.
inline float rstd128(const float* x) {
  float red[kIdxDim];
  for (uint32_t i = 0; i < kIdxDim; ++i) red[i] = x[i] * x[i];
  for (uint32_t stride = kIdxDim / 2; stride > 0; stride >>= 1)
    for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
  return 1.0f / std::sqrt(red[0] / float(kIdxDim) + kEps);
}
// The indexer's query heads at the row's position: q [4][128] bf16 values (q4_qsa_prep heads 0..3).
inline void qsa_q(const float* idx_f32, const float* wq, const float* cs, float* q) {
  for (uint32_t h = 0; h < kIdxHeads; ++h) {
    float x[kIdxDim];
    uint16_t n[kIdxDim], o[kIdxDim];
    for (uint32_t i = 0; i < kIdxDim; ++i) x[i] = rf(idx_f32[h * kIdxDim + i]);
    const float rstd = rstd128(x);
    for (uint32_t i = 0; i < kIdxDim; ++i) n[i] = norm_elem(x[i], rstd, wq[i]);
    rope64_bf16(n, cs, kIdxDim, o);
    for (uint32_t i = 0; i < kIdxDim; ++i) q[h * kIdxDim + i] = f32(o[i]);
  }
}
// The raw key of a row: rne(idx_f32[512 .. 640)).
inline void qsa_raw_key(const float* idx_f32, uint16_t* raw) {
  for (uint32_t i = 0; i < kIdxDim; ++i) raw[i] = rne(idx_f32[kIdxHeads * kIdxDim + i]);
}
// A complete block's compressed key from its 4 raw keys (positions 4b .. 4b + 3): the fp32 mean
// (((k0 + k1) + k2) + k3) / 4 -> bf16 -> k_layernorm -> RoPE at 4b (`cs` its table row).
inline void qsa_block_key(const uint16_t* const raw[4], const float* wk, const float* cs, uint16_t* key) {
  float x[kIdxDim];
  uint16_t n[kIdxDim];
  for (uint32_t i = 0; i < kIdxDim; ++i)
    x[i] = rf((((f32(raw[0][i]) + f32(raw[1][i])) + f32(raw[2][i])) + f32(raw[3][i])) / 4.0f);
  const float rstd = rstd128(x);
  for (uint32_t i = 0; i < kIdxDim; ++i) n[i] = norm_elem(x[i], rstd, wk[i]);
  rope64_bf16(n, cs, kIdxDim, key);
}
// A block's score: sum_h relu(sum_i q[h][i] k[i]) / sqrt(128): each head's dot one fma chain (i ascending -
// bf16 x bf16 products are exact, so contraction-proof), relu (+0 for any non-positive), the heads
// ((r0 + r1) + r2) + r3, then the division.
inline float qsa_score(const float* q, const uint16_t* key) {
  float r[kIdxHeads];
  for (uint32_t h = 0; h < kIdxHeads; ++h) {
    float a = 0.0f;
    for (uint32_t i = 0; i < kIdxDim; ++i) a = std::fma(q[h * kIdxDim + i], f32(key[i]), a);
    r[h] = a > 0.0f ? a : 0.0f;
  }
  return (((r[0] + r[1]) + r[2]) + r[3]) / kSqrtIdx;
}
// The selection of row p over its n = (p + 1) / 4 complete blocks' scores (spec 21 §2.3, decision 3): n <= 512
// selects everything - the list 0 .. p; else the top 512 by (score desc, block asc) - EXACT ties to the lower
// block - written ascending, each block expanded to its 4 positions, then the open block's tail 4n .. p.
// s512 / s513: the 512th / 513th scores in that order (+INF / -INF when n <= 512: no cut).
struct Selection {
  std::vector<uint32_t> list;      // positions ascending
  std::vector<uint32_t> blocks;    // the selected blocks ascending (n > 512)
  float s512 = INFINITY, s513 = -INFINITY;
  uint32_t count() const { return uint32_t(list.size()); }
};
inline Selection qsa_select(const float* scores, uint32_t p) {
  Selection r;
  const uint32_t n = (p + 1) / kBlock;
  if (n <= kTopBlocks) {
    for (uint32_t q = 0; q <= p; ++q) r.list.push_back(q);
    return r;
  }
  std::vector<uint32_t> idx(n);
  for (uint32_t b = 0; b < n; ++b) idx[b] = b;
  const auto before = [&](uint32_t a, uint32_t b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); };
  std::partial_sort(idx.begin(), idx.begin() + kTopBlocks + 1, idx.end(), before);
  r.s512 = scores[idx[kTopBlocks - 1]];
  r.s513 = scores[idx[kTopBlocks]];
  r.blocks.assign(idx.begin(), idx.begin() + kTopBlocks);
  std::sort(r.blocks.begin(), r.blocks.end());
  for (uint32_t b : r.blocks)
    for (uint32_t j = 0; j < kBlock; ++j) r.list.push_back(b * kBlock + j);
  for (uint32_t q = n * kBlock; q <= p; ++q) r.list.push_back(q);
  return r;
}
// The list's count for row p: 2048 + the tail when more than 512 blocks are complete, else p + 1.
inline uint32_t qsa_count(uint32_t p) {
  const uint32_t n = (p + 1) / kBlock;
  return n <= kTopBlocks ? p + 1 : kTopBlocks * kBlock + (p + 1 - n * kBlock);
}

// q4_qsa_attn_eager, one q head h of one row over the list (kv caches [rows][2][256] bf16, rows by position):
//   q_b = rne(attn_q) (attn_prep keeps the RoPE'd q in fp32; the reference's q is bf16);
//   s_i = rne(rne(sum_d q_b k_i) / 16) (one fma chain, d ascending; the /16 exact);
//   p = torch's fp32 softmax over the list (k2_ref::eager_softmax: exp_torch, 8 lane sums, the tree), bf16;
//   o_d = rne(sum_i p_i v_i[d]) (one fma chain, i ascending); out = rne(f32(o) x rne(sigmoid(gate))).
struct EagerHead {
  std::vector<float> s, p;
  std::vector<uint16_t> out;   // [256]
};
inline EagerHead qsa_attn_eager(const float* attn_q, const float* attn_gate, const uint16_t* kv_k, const uint16_t* kv_v,
                                const uint32_t* list, uint32_t count, uint32_t h) {
  const uint32_t j = h / kGqa;
  EagerHead r;
  r.s.resize(count);
  r.p.resize(count);
  r.out.resize(kHd);
  float qb[kHd];
  for (uint32_t d = 0; d < kHd; ++d) qb[d] = rf(attn_q[h * kHd + d]);
  for (uint32_t i = 0; i < count; ++i) {
    const uint16_t* k = kv_k + (size_t(list[i]) * kKvHeads + j) * kHd;
    float a = 0.0f;
    for (uint32_t d = 0; d < kHd; ++d) a = std::fma(qb[d], f32(k[d]), a);
    r.s[i] = rf(rf(a) * kAttnScale);
  }
  k2_ref::eager_softmax(r.s.data(), count, r.p.data());
  for (uint32_t d = 0; d < kHd; ++d) {
    float o = 0.0f;
    for (uint32_t i = 0; i < count; ++i) o = std::fma(r.p[i], f32(kv_v[(size_t(list[i]) * kKvHeads + j) * kHd + d]), o);
    r.out[d] = rne(rf(o) * rf(sigmoid_t(attn_gate[h * kHd + d])));
  }
  return r;
}
// The same attention in fp64 without the gate (flash's tolerance reference: o before rounding).
inline std::vector<double> qsa_attn_fp64(const float* attn_q, const uint16_t* kv_k, const uint16_t* kv_v,
                                         const uint32_t* list, uint32_t count, uint32_t h) {
  const uint32_t j = h / kGqa;
  std::vector<double> sc(count);
  double mx = -INFINITY;
  for (uint32_t i = 0; i < count; ++i) {
    const uint16_t* k = kv_k + (size_t(list[i]) * kKvHeads + j) * kHd;
    double a = 0;
    for (uint32_t d = 0; d < kHd; ++d) a += double(attn_q[h * kHd + d]) * f32(k[d]);
    sc[i] = a * kAttnScale;
    mx = std::max(mx, sc[i]);
  }
  double z = 0;
  for (double& s : sc) z += (s = std::exp(s - mx));
  std::vector<double> o(kHd, 0.0);
  for (uint32_t i = 0; i < count; ++i)
    for (uint32_t d = 0; d < kHd; ++d) o[d] += sc[i] / z * f32(kv_v[(size_t(list[i]) * kKvHeads + j) * kHd + d]);
  return o;
}

// --- the MoE (spec 21 §2.1, §4.4; 21a: grouped_mm) -----------------------------------------------------------------
struct Route {
  uint32_t ids[kTopK] = {};   // rank order: descending p, exact ties to the lower id
  float w[kTopK] = {};        // rne(p_k / sum_k p_k), bf16 values
  float p[kTopK] = {};
  float p10 = 0, p11 = 0;     // p of rank 9 and rank 10 (the first not taken)
  float sg = 0;               // rne(sigmoid(rne(logit[512])))
};
// q4_route over one row of the router GEMV's fp32 output [528]: l = rf(logits) (the linear's bf16 output);
// mx exact; ex = exp_torch(l - mx); the sum a pairwise tree over 512 (stride 256 .. 1); p = ex / sum; rank by
// (p desc, id asc); s10 = sum_{k < 10} p_k ascending k; w_k = rf(p_k / s10); the shared gate.
inline Route route(const float* logits) {
  std::vector<float> l(kExperts), ex(kExperts), red(kExperts), p(kExperts);
  for (uint32_t e = 0; e < kExperts; ++e) l[e] = rf(logits[e]);
  float mx = l[0];
  for (uint32_t e = 1; e < kExperts; ++e) mx = std::fmax(mx, l[e]);
  for (uint32_t e = 0; e < kExperts; ++e) red[e] = ex[e] = exp_torch(l[e] - mx);
  for (uint32_t stride = kExperts / 2; stride > 0; stride >>= 1)
    for (uint32_t e = 0; e < stride; ++e) red[e] += red[e + stride];
  for (uint32_t e = 0; e < kExperts; ++e) p[e] = ex[e] / red[0];
  std::vector<uint32_t> top(kTopK + 1);
  std::vector<float> tp(kTopK + 1, 0.0f);
  for (uint32_t k = 0; k <= kTopK; ++k) top[k] = k;
  for (uint32_t e = 0; e < kExperts; ++e) {
    uint32_t rank = 0;
    for (uint32_t j = 0; j < kExperts; ++j) rank += (p[j] > p[e] || (p[j] == p[e] && j < e)) ? 1u : 0u;
    if (rank <= kTopK) {
      top[rank] = e;
      tp[rank] = p[e];
    }
  }
  Route r;
  float s10 = 0.0f;
  for (uint32_t k = 0; k < kTopK; ++k) s10 += tp[k];
  for (uint32_t k = 0; k < kTopK; ++k) {
    r.ids[k] = top[k];
    r.p[k] = tp[k];
    r.w[k] = rf(tp[k] / s10);
  }
  r.p10 = tp[kTopK - 1];
  r.p11 = tp[kTopK];
  r.sg = rf(sigmoid_t(rf(logits[kExperts])));
  return r;
}
// The route row q4_route writes.
inline std::vector<uint32_t> route_row(const Route& r) {
  std::vector<uint32_t> w(kWords, 0u);
  for (uint32_t k = 0; k < kTopK; ++k) {
    w[kIds + k] = r.ids[k];
    w[kWeights + k] = as_u32(r.w[k]);
  }
  w[kSharedGate] = as_u32(r.sg);
  w[kP10] = as_u32(r.p10);
  w[kP11] = as_u32(r.p11);
  return w;
}

// u32 words of one expert's layout-1 blocks (loader/qwen4exp_layout.h: 1,740,800 B and 870,400 B).
inline size_t gate_up_block_words() { return size_t(2 * kInter / 16) * (kHidden / 64) * 136; }
inline size_t down_block_words() { return size_t(kHidden / 16) * (kInter / 64) * 136; }
// One bf16-tiled column n ({K, N} gemv_bf16 tiles) over `ks` K slices, each ascending, the pairwise tree.
inline float bf16_split_dot(const uint16_t* tiles, uint32_t K, uint32_t n, const uint16_t* x, uint32_t ks) {
  const uint32_t K8 = K / 8, per = K8 / ks;
  std::vector<float> r(ks);
  for (uint32_t q = 0; q < ks; ++q) {
    float a = 0.f;
    for (uint32_t k8 = q * per; k8 < (q + 1) * per; ++k8)
      for (uint32_t i = 0; i < 8; ++i) a += f32(tiles[((size_t(n / 16) * K8 + k8) * 8 + i) * 16 + n % 16]) * f32(x[k8 * 8 + i]);
    r[q] = a;
  }
  for (uint32_t stride = ks / 2; stride > 0; stride >>= 1)
    for (uint32_t q = 0; q < stride; ++q) r[q] += r[q + stride];
  return r[0];
}
// h = rne(f32(rne(silu(f32(g_b)))) · f32(u_b)).
inline uint16_t silu_up(float gate_sum, float up_sum) {
  const uint16_t g_b = rne(gate_sum), u_b = rne(up_sum);
  return rne(rf(silu_t(f32(g_b))) * f32(u_b));
}
// h [11][640] bf16: slot k < 10 the routed expert ids[k] (layout-1 blocks at gu + id x the block), slot 10 the
// shared expert: int4 layout-1 (sh_gu4) or bf16 tiles {2560, 1280} gate||up interleave16 (sh_gub).
inline std::vector<uint16_t> gate_up(const uint32_t* ids, const uint16_t* x, const uint32_t* gu, const uint32_t* sh_gu4,
                                     const uint16_t* sh_gub, uint32_t up_ks) {
  std::vector<uint16_t> h(size_t(kSlots) * kInter);
  for (uint32_t slot = 0; slot < kSlots; ++slot)
    for (uint32_t i = 0; i < kInter; ++i) {
      const uint32_t gc = moe_ref::gate_col(i), uc = moe_ref::up_col(i);
      float g, u;
      if (slot < kTopK || sh_gu4) {
        const uint32_t* blk = slot < kTopK ? gu + ids[slot] * gate_up_block_words() : sh_gu4;
        g = moe_ref::split_dot(blk, kHidden, gc, x, up_ks);
        u = moe_ref::split_dot(blk, kHidden, uc, x, up_ks);
      } else {
        g = bf16_split_dot(sh_gub, kHidden, gc, x, up_ks);
        u = bf16_split_dot(sh_gub, kHidden, uc, x, up_ks);
      }
      h[size_t(slot) * kInter + i] = silu_up(g, u);
    }
  return h;
}
// q4_moe_down's epilogue for one hidden column: down[11] the fp32 down sums (slot order, the shared last):
//   r = sum_k f32(rne(f32(rne(down_k)) · w_k)) in fp32, k ascending (the rank order), r_b = rne(r);
//   sh_b = rne(f32(rne(down_10)) · sg); y = rne(f32(r_b) + f32(sh_b))       NOT folded into H
inline uint16_t combine(const float* down, const float* w, float sg) {
#pragma STDC FP_CONTRACT OFF
  float sum = 0.0f;
  for (uint32_t k = 0; k < kTopK; ++k) {
    const float t = rf(rf(down[k]) * w[k]);
    sum = sum + t;
  }
  const uint16_t r_b = rne(sum);
  const uint16_t sh_b = rne(rf(down[kTopK]) * sg);
  return rne(f32(r_b) + f32(sh_b));
}
// y [2560] bf16.
inline std::vector<uint16_t> down(const uint32_t* ids, const float* w, float sg, const std::vector<uint16_t>& h,
                                  const uint32_t* dn, const uint32_t* sh_dn4, const uint16_t* sh_dnb, uint32_t dn_ks) {
  std::vector<uint16_t> out(kHidden);
  float d[kSlots];
  for (uint32_t n = 0; n < kHidden; ++n) {
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
      const uint16_t* hs = h.data() + size_t(slot) * kInter;
      if (slot < kTopK)
        d[slot] = moe_ref::split_dot(dn + ids[slot] * down_block_words(), kInter, n, hs, dn_ks);
      else
        d[slot] = sh_dn4 ? moe_ref::split_dot(sh_dn4, kInter, n, hs, dn_ks) : bf16_split_dot(sh_dnb, kInter, n, hs, dn_ks);
    }
    out[n] = combine(d, w, sg);
  }
  return out;
}

// --- the GDN gated head with the sigmoid gate (prep_gated_head _SIG; spec 21 §4.4) ----------------------------------
// prep_ref::gated_head's chain with the last factor sigmoid(z) = 1 / (1 + exp_torch(-z)):
//   o_b = rne(gdn_o); z_b = rne(qkvz[Z_OFF + h·128 + i]) (S 1); var: 128 lanes o_f^2 (plain), the tree;
//   n_b = rne(o_f · (1 / sqrt(var + eps))); t_b = rne(f32(w) · f32(n_b)); out = rne(f32(t_b) · sigmoid(f32(z_b)))
inline void gated_head_sig(const float* qkvz, const float* gdn_o, const uint16_t* gated_w, uint16_t* out) {
  for (uint32_t h = 0; h < kGdnHeads; ++h) {
    float red[kGdnHd];
    uint16_t o_b[kGdnHd], z_b[kGdnHd];
    for (uint32_t i = 0; i < kGdnHd; ++i) {
      o_b[i] = rne(gdn_o[h * kGdnHd + i]);
      z_b[i] = rne(qkvz[kZOff + h * kGdnHd + i]);
      red[i] = f32(o_b[i]) * f32(o_b[i]);
    }
    for (uint32_t stride = kGdnHd / 2; stride > 0; stride >>= 1)
      for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
    const float rstd = 1.0f / std::sqrt(red[0] / float(kGdnHd) + kEps);
    for (uint32_t i = 0; i < kGdnHd; ++i) {
      const uint16_t n_b = rne(f32(o_b[i]) * rstd);
      const uint16_t t_b = rne(f32(gated_w[i]) * f32(n_b));
      out[h * kGdnHd + i] = rne(f32(t_b) * sigmoid_t(f32(z_b[i])));
    }
  }
}

}  // namespace q4ref
