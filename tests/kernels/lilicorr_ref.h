#pragma once
// CPU twin of spec 26b's LiLiCorr drafter kernels (src/kernels/dflash/ll_*.cl) and the reused df_prep / GEMV chain at
// LiLiCorr's shapes - the same op chains in the same orders, every function with a twin in a kernel; edit the two
// together. The semantics are tools/oracle/dflash_ref.py's LiLiCorr form (spec 26 §2, 26a's port of vLLM's
// models/lilicorr.py); tests/kernels/lilicorr_ref_test.cc holds this twin to that reference's own outputs
// (tests/loader/lilicorr_dflash_fixture.safetensors, tools/oracle/dflash_fixture.py --lilicorr) in Reference numerics, and
// the kernels to this twin (tools/mac/clrun/lilicorr_run.cc on the Mac's GPU, tests/kernels/lilicorr_kernels_test.cc on
// the card). Built on tests/kernels/dflash_ref.h (dfref: norm, conv, head_prep, the GEMVs' arithmetic, murmur3, choose).
//
//   attention_sink()  ll_attn_part + ll_attn_reduce: df_attn's chain (context blocks of PB ring offsets, each block's
//                     (m, l, acc) ascending; the reduce merges blocks ascending, then the block's rows 0..K) and THE SINK:
//                     l += exp(sink_h - m) once per (row, head), after the blocks and the keys merged; the numerator
//                     untouched (Review Focus 2) - dflash_ref's softmax(cat([s, sink]))[..., :-1]
//   topk_lse()        ll_topk_stage1/2: the TOPK (8) largest raw logits (value desc, index asc; NaN never), and the row's
//                     logsumexp from per-chunk (max, Σ exp(v - max)) - CH-logit chunks, lane partials (i = lane, lane +
//                     WG, ... ascending) then the pairwise tree - merged in chunk order: lse = M + log(Σ_c s_c exp(m_c -
//                     M)); lp = v - lse (Review Focus 3). A chunk of -inf only (m = -inf) adds 0; NaN is skipped.
//   head_lattice()    ll_lattice.cl's chain (the LiLiCorr head, vLLM's LiLiCorrHead, fp32): every projection is ll_gemm -
//                     y[r][n] = (Σ_k x[r][k] w[k][n], k ascending) x gs[n] + bias[n] over fp32 inputs, W'^T bf16
//                     (Decision 3: exact) -; the features and LayerNorm(5) (ll_feat), the sum tok + pass + feat + slot +
//                     rank (ll_sum), RMSNorm by df_norm's 256-lane tree (ll_rmsnorm), the 56 x 56 attention per head with
//                     the slot bias (ll_lattn: scores ascending d, + bias, m, e, l ascending j, Σ e v / l), SiLU, the factor
//                     projection's three sums ((self + b) + anchor) + cross (ll_factor), the L2 normalisations x / max(||x||,
//                     eps) (ll_l2norm), start / pair (ll_scores) x logit_scale (Review Focus 4)
//   walk()            ll_select: previous = 0; slot s scores lattice[s][previous] - argmax (ties lower) or Gumbel-argmax
//                     keyed by (seed ^ (s << 32), p + s, candidate id) in fp32 (dfref::choose)
//   project()         the projection list: fc (+ its g) -> df_norm (hidden_norm) -> the anchor (the last row's fp32 h, Review
//                     Focus 5) -> per layer the k||v rows -> df_ctx_store (k_norm, RoPE, ring slot pos % RING)
//   draft()           the draft list in capture order: the 8-row block always (anchor + 7 masks; K only bounds the walk -
//                     spec 26 §12 ruling 1), the 5 layers with the convolutions and the sinks, the final norm, the target
//                     head over rows 1..7, topk_lse, head_lattice, walk
//
// Numerics::Engine is the device's (bf16 GEMV inputs, a bf16 ring, the backbone in its device form); Numerics::Reference keeps
// every backbone activation fp32 over fp32 weights (W' widened, g on the output) - dflash_ref.py's arithmetic up to
// summation order. The head runs on fp32 inputs in both (a departure from 19b's bf16-input GEMVs that costs nothing in
// bytes: the 74 MB of head weights dominate, spec 26 §5.2).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kernels/dflash_ref.h"

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace llref {

using dfref::bf;
using dfref::Numerics;
using dfref::rf;
using dfref::rne;

constexpr uint32_t kTopkChunk = 4096, kTopkWg = 256;   // ll_topk's CH and WG
constexpr uint32_t kNormWg = 256;                      // ll_rmsnorm's / ll_l2norm's lanes

struct Dims : dfref::Dims {
  uint32_t S = 7, k = 8, Hl = 0, lheads = 0, lmlp = 0, F = 0, llayers = 0;
  float veps = 1e-4f, lscale = 8.0f, feps = 1e-5f;
  bool token_proj = true;
  uint32_t R() const { return S * k; }
  uint32_t lhd() const { return Hl / lheads; }
};

// A backbone linear in its device form and, for an NVFP4 linear in Exact (or any linear in Reference), its per-column
// output scale - ll_gemv_bf16_g: y = (W' x) x gs[n] (empty: none).
struct GLin {
  dfref::Lin lin;
  std::vector<float> gs;
  void apply(const uint16_t* xb, const float* xf, uint32_t M, float* y) const {
    lin.apply(xb, xf, M, y);
    if (!gs.empty())
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < lin.N; ++n) y[size_t(m) * lin.N + n] = y[size_t(m) * lin.N + n] * gs[n];
  }
};

struct Layer {
  GLin qkv, o, gate_up, down, akp, mkp;
  std::vector<float> in_norm, post_norm, q_norm, k_norm, conv_base, sink;   // conv_base [attn, mlp][side][tap][H]
};

// A head linear: W'^T (bf16 values, k-major [K][N]) widened, the output scale and the bias.
struct HeadLin {
  uint32_t K = 0, N = 0;
  std::vector<float> wt, gs, bias;
  void apply(const float* x, uint32_t rows, float* y) const {
    dfref::parallel_cols(N, [&](uint32_t n0, uint32_t n1) {
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t n = n0; n < n1; ++n) {
          float a = 0.0f;
          for (uint32_t kk = 0; kk < K; ++kk) a = a + x[size_t(r) * K + kk] * wt[size_t(kk) * N + n];
          y[size_t(r) * N + n] = a * gs[n] + bias[n];
        }
    });
  }
};

struct Head {
  HeadLin tok, pass, fup, fdown, ctx, factor, outin, aout;
  std::vector<HeadLin> inproj, outproj, up, down;
  std::vector<float> fn_w, fn_b, slot, rank, rel, same, attn_norm, mlp_norm, out_norm, anchor_norm;   // per layer: [l][Hl]
};

struct Weights {
  Dims d;
  GLin fc;
  std::vector<Layer> layers;
  std::vector<float> hidden_norm, final_norm, mask_row;
  std::vector<uint16_t> embed;   // the target's bf16 [V][H]
  dfref::Lin head;               // the target's lm_head
  Head lil;
};

// --- the backbone's pieces ------------------------------------------------------------------------------------------

// ll_attn: df_attn's chain + the sink. `sink` [NH] fp32 (this layer's).
inline void attention_sink(const dfref::AttnDims& a, const float* q, const float* kb, const float* vb, const float* ring_k,
                           const float* ring_v, uint32_t p, uint32_t K, const float* sink, float* out) {
  const uint32_t HD = a.HD, G = a.g();
  std::fill(out, out + size_t(a.M) * a.NH * HD, 0.0f);
  for (uint32_t h = 0; h < a.NH; ++h) {
    const uint32_t kvh = h / G;
    for (uint32_t r = 0; r < a.M && r <= K; ++r) {
      const float* qr = q + (size_t(r) * a.NH + h) * HD;
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
            e[t] = s[t] == -INFINITY ? 0.0f : std::exp(s[t] - m);
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
      std::vector<float> sb(a.M, -INFINITY);
      for (uint32_t j = 0; j < a.M && j <= K; ++j) {
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
        wb[b] = bm[b] == -INFINITY ? 0.0f : std::exp(bm[b] - m);
        l = l + bl[b] * wb[b];
      }
      for (uint32_t j = 0; j < a.M; ++j) {
        wj[j] = sb[j] == -INFINITY ? 0.0f : std::exp(sb[j] - m);
        l = l + wj[j];
      }
      l = l + std::exp(sink[h] - m);   // the sink: the denominator only, once, after the blocks and the keys
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

struct TopK {
  std::vector<uint32_t> id;   // [k]
  std::vector<float> v, lp;   // the raw logits and v - lse
  float lse = 0.0f;
};

// The lane-and-tree Σ of one chunk's exp(v - m) (ll_topk_stage1's): lane l over e = 0 .. CH / WG - 1 of i = c CH + e WG +
// l (valid: i < used), then strides WG / 2 .. 1.
inline TopK topk_lse(const float* logits, uint32_t used, uint32_t k) {
  TopK out;
  std::vector<dfref::Cand> c;
  for (uint32_t i = 0; i < used; ++i)
    if (!std::isnan(logits[i])) c.push_back({logits[i], i});
  std::sort(c.begin(), c.end(), dfref::better);
  c.resize(std::min<size_t>(c.size(), k));
  while (c.size() < k) c.push_back({-INFINITY, dfref::kNoIdx});
  const uint32_t chunks = (used + kTopkChunk - 1) / kTopkChunk;
  std::vector<float> cm(chunks), cs(chunks);
  for (uint32_t ch = 0; ch < chunks; ++ch) {
    float m = -INFINITY;
    for (uint32_t i = ch * kTopkChunk; i < std::min(used, (ch + 1) * kTopkChunk); ++i)
      if (!std::isnan(logits[i])) m = std::fmax(m, logits[i]);
    std::vector<float> lane(kTopkWg, 0.0f);
    if (m != -INFINITY)
      for (uint32_t l = 0; l < kTopkWg; ++l) {
        float a = 0.0f;
        for (uint32_t e = 0; e < kTopkChunk / kTopkWg; ++e) {
          const uint32_t i = ch * kTopkChunk + e * kTopkWg + l;
          if (i >= used) continue;
          const float x = logits[i];
          if (std::isnan(x) || x == -INFINITY) continue;
          a = a + (x == m ? 1.0f : std::exp(x - m));
        }
        lane[l] = a;
      }
    for (uint32_t stride = kTopkWg / 2; stride > 0; stride >>= 1)
      for (uint32_t l = 0; l < stride; ++l) lane[l] = lane[l] + lane[l + stride];
    cm[ch] = m;
    cs[ch] = lane[0];
  }
  float M = -INFINITY;
  for (uint32_t ch = 0; ch < chunks; ++ch) M = std::fmax(M, cm[ch]);
  float S = 0.0f;
  for (uint32_t ch = 0; ch < chunks; ++ch)
    if (cm[ch] != -INFINITY) S = S + cs[ch] * (cm[ch] == M ? 1.0f : std::exp(cm[ch] - M));
  out.lse = M == -INFINITY ? -INFINITY : M + std::log(S);
  for (const dfref::Cand& x : c) {
    out.id.push_back(x.i);
    out.v.push_back(x.v);
    out.lp.push_back(x.i == dfref::kNoIdx ? -INFINITY : x.v - out.lse);
  }
  return out;
}

// ll_rmsnorm, one row of n: Σx² by the 256-lane tree (dfref::wg_sumsq), y = (x · rstd) · w.
inline void rmsnorm(const float* x, const float* w, uint32_t n, float eps, float* y) {
  const float ss = dfref::wg_sumsq(x, n, kNormWg);
  const float rstd = 1.0f / std::sqrt(ss / float(n) + eps);
  for (uint32_t i = 0; i < n; ++i) y[i] = (x[i] * rstd) * w[i];
}
// ll_l2norm, one row of n: x / max(sqrt(Σx²), eps) (F.normalize).
inline void l2norm(float* x, uint32_t n, float eps) {
  const float nr = std::sqrt(dfref::wg_sumsq(x, n, kNormWg));
  const float den = std::fmax(nr, eps);
  for (uint32_t i = 0; i < n; ++i) x[i] = x[i] / den;
}
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// The LiLiCorr head over one block: cand [S][k] ids, lp [S][k], rows [S][H] (the final-normed mask rows, fp32), the anchor
// [H] fp32 and its validity -> the lattice [S][k (previous)][k (next)] x logit_scale.
inline std::vector<float> head_lattice(const Weights& w, const uint32_t* cand, const float* lp, const float* rows,
                                       const float* anchor, bool valid) {
  const Dims& d = w.d;
  const Head& h = w.lil;
  const uint32_t S = d.S, k = d.k, R = d.R(), Hl = d.Hl, H = d.H, F = d.F, nh = d.lheads, hd = d.lhd();
  // tok: the candidates' embedding rows (fp32 widenings of the bf16 rows) through token_proj
  std::vector<float> emb(size_t(R) * H), tok(size_t(R) * Hl);
  for (uint32_t r = 0; r < R; ++r)
    for (uint32_t i = 0; i < H; ++i) emb[size_t(r) * H + i] = bf(w.embed[size_t(cand[r]) * H + i]);
  if (d.token_proj)
    h.tok.apply(emb.data(), R, tok.data());
  else
    tok = emb;
  std::vector<float> pas(size_t(S) * Hl);
  h.pass.apply(rows, S, pas.data());
  // the features (ll_feat): [lp, exp(lp), lp - max lp, rank / (k - 1), rank == 0], then LayerNorm(5)
  std::vector<float> f(size_t(R) * 5);
  for (uint32_t s = 0; s < S; ++s) {
    float mx = -INFINITY;
    for (uint32_t i = 0; i < k; ++i) mx = std::fmax(mx, lp[s * k + i]);
    for (uint32_t i = 0; i < k; ++i) {
      const float x = lp[s * k + i];
      float* o = &f[(size_t(s) * k + i) * 5];
      o[0] = x;
      o[1] = std::exp(x);
      o[2] = x - mx;
      o[3] = k > 1 ? float(i) / float(k - 1) : 0.0f;
      o[4] = i == 0 ? 1.0f : 0.0f;
      const float mean = ((((o[0] + o[1]) + o[2]) + o[3]) + o[4]) / 5.0f;
      float var = 0.0f;
      for (uint32_t j = 0; j < 5; ++j) var = var + (o[j] - mean) * (o[j] - mean);
      var = var / 5.0f;
      const float rstd = 1.0f / std::sqrt(var + d.feps);
      for (uint32_t j = 0; j < 5; ++j) o[j] = ((o[j] - mean) * rstd) * h.fn_w[j] + h.fn_b[j];
    }
  }
  std::vector<float> fu(size_t(R) * Hl), fm(size_t(R) * Hl);
  h.fup.apply(f.data(), R, fu.data());
  for (float& x : fu) x = silu(x);
  h.fdown.apply(fu.data(), R, fm.data());
  // ll_sum: (((tok + pass) + feat) + slot) + rank
  std::vector<float> x(size_t(R) * Hl);
  for (uint32_t r = 0; r < R; ++r) {
    const uint32_t s = r / k, i = r % k;
    for (uint32_t j = 0; j < Hl; ++j) {
      const size_t at = size_t(r) * Hl + j;
      x[at] = (((tok[at] + pas[size_t(s) * Hl + j]) + fm[at]) + h.slot[size_t(s) * Hl + j]) + h.rank[size_t(i) * Hl + j];
    }
  }
  // the anchor: context_proj(anchor) x valid
  std::vector<float> a(Hl);
  h.ctx.apply(anchor, 1, a.data());
  for (float& v : a) v = v * (valid ? 1.0f : 0.0f);
  // the layers
  const float scale = float(1.0 / std::sqrt(double(hd)));
  std::vector<float> xn(size_t(R) * Hl), qkv(size_t(R) * 3 * Hl), o(size_t(R) * Hl), y(size_t(R) * Hl),
      u(size_t(R) * d.lmlp);
  for (uint32_t l = 0; l < d.llayers; ++l) {
    for (uint32_t r = 0; r < R; ++r) rmsnorm(&x[size_t(r) * Hl], &h.attn_norm[size_t(l) * Hl], Hl, d.eps, &xn[size_t(r) * Hl]);
    h.inproj[l].apply(xn.data(), R, qkv.data());
    for (uint32_t hh = 0; hh < nh; ++hh)
      for (uint32_t i = 0; i < R; ++i) {
        std::vector<float> sc(R);
        float m = -INFINITY;
        for (uint32_t j = 0; j < R; ++j) {
          float dd = 0.0f;
          for (uint32_t t = 0; t < hd; ++t)
            dd = dd + qkv[size_t(i) * 3 * Hl + hh * hd + t] * qkv[size_t(j) * 3 * Hl + Hl + hh * hd + t];
          const uint32_t si = i / k, sj = j / k;
          const float bias = h.rel[size_t(hh) * (2 * d.M - 1) + (si + d.M - 1 - sj)] + (si == sj ? h.same[hh] : 0.0f);
          sc[j] = dd * scale + bias;
          m = std::fmax(m, sc[j]);
        }
        float lsum = 0.0f;
        for (uint32_t j = 0; j < R; ++j) {
          sc[j] = std::exp(sc[j] - m);
          lsum = lsum + sc[j];
        }
        for (uint32_t t = 0; t < hd; ++t) {
          float acc = 0.0f;
          for (uint32_t j = 0; j < R; ++j) acc = acc + sc[j] * qkv[size_t(j) * 3 * Hl + 2 * Hl + hh * hd + t];
          o[size_t(i) * Hl + hh * hd + t] = acc / lsum;
        }
      }
    h.outproj[l].apply(o.data(), R, y.data());
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] + y[i];
    for (uint32_t r = 0; r < R; ++r) rmsnorm(&x[size_t(r) * Hl], &h.mlp_norm[size_t(l) * Hl], Hl, d.eps, &xn[size_t(r) * Hl]);
    h.up[l].apply(xn.data(), R, u.data());
    for (float& v : u) v = silu(v);
    h.down[l].apply(u.data(), R, y.data());
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] + y[i];
  }
  for (uint32_t r = 0; r < R; ++r) rmsnorm(&x[size_t(r) * Hl], h.out_norm.data(), Hl, d.eps, &xn[size_t(r) * Hl]);
  std::vector<float> an(Hl);
  rmsnorm(a.data(), h.anchor_norm.data(), Hl, d.eps, an.data());
  // ll_factor: pre = ((h W_self + b) + a W_anchor) + (h ⊙ a) W_cross, then SiLU
  std::vector<float> fh(size_t(R) * Hl);
  {
    const HeadLin& W = h.factor;   // k-major [3 Hl][Hl]
    for (uint32_t r = 0; r < R; ++r)
      for (uint32_t n = 0; n < Hl; ++n) {
        float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (uint32_t t = 0; t < Hl; ++t) s1 = s1 + xn[size_t(r) * Hl + t] * W.wt[size_t(t) * Hl + n];
        for (uint32_t t = 0; t < Hl; ++t) s2 = s2 + an[t] * W.wt[size_t(Hl + t) * Hl + n];
        for (uint32_t t = 0; t < Hl; ++t) s3 = s3 + (xn[size_t(r) * Hl + t] * an[t]) * W.wt[size_t(2 * Hl + t) * Hl + n];
        fh[size_t(r) * Hl + n] = silu(((s1 * W.gs[n] + W.bias[n]) + s2 * W.gs[n]) + s3 * W.gs[n]);
      }
  }
  // out_v / in_v, a_out
  std::vector<float> ov(size_t(R) * 2 * F), ao(F);
  h.outin.apply(fh.data(), R, ov.data());
  for (uint32_t r = 0; r < R; ++r) {
    l2norm(&ov[size_t(r) * 2 * F], F, d.veps);
    l2norm(&ov[size_t(r) * 2 * F + F], F, d.veps);
  }
  h.aout.apply(an.data(), 1, ao.data());
  l2norm(ao.data(), F, d.veps);
  // ll_scores: slot 0 = start broadcast over the previous index; slot s >= 1 = out_v[s - 1][i] · in_v[s][j]
  std::vector<float> lat(size_t(S) * k * k);
  for (uint32_t j = 0; j < k; ++j) {
    float st = 0.0f;
    for (uint32_t t = 0; t < F; ++t) st = st + ao[t] * ov[size_t(j) * 2 * F + F + t];
    for (uint32_t i = 0; i < k; ++i) lat[size_t(i) * k + j] = d.lscale * st;
  }
  for (uint32_t s = 1; s < S; ++s)
    for (uint32_t i = 0; i < k; ++i)
      for (uint32_t j = 0; j < k; ++j) {
        float pr = 0.0f;
        for (uint32_t t = 0; t < F; ++t)
          pr = pr + ov[(size_t(s - 1) * k + i) * 2 * F + t] * ov[(size_t(s) * k + j) * 2 * F + F + t];
        lat[(size_t(s) * k + i) * k + j] = d.lscale * pr;
      }
  return lat;
}

struct Walk {
  std::vector<uint32_t> chosen, ids;
  std::vector<float> scores;   // [K][k] the realized rows
};
// ll_select's walk over the lattice.
inline Walk walk(const std::vector<float>& lat, const uint32_t* cand, uint32_t k, uint32_t p, uint32_t K, float T,
                 uint64_t seed) {
  Walk w;
  uint32_t prev = 0;
  for (uint32_t s = 0; s < K; ++s) {
    const float* row = &lat[(size_t(s) * k + prev) * k];
    const uint32_t c = dfref::choose(row, cand + s * k, k, T, seed ^ (uint64_t(s) << 32), p + s);
    w.chosen.push_back(c);
    w.ids.push_back(cand[s * k + c]);
    w.scores.insert(w.scores.end(), row, row + k);
    prev = c;
  }
  return w;
}

// The projection of n tap rows at pos0 .. into the ring; returns the anchor (the last row's hidden_norm output, fp32).
inline std::vector<float> project(const Weights& w, Numerics nm, const uint16_t* taps, uint32_t n, uint32_t pos0,
                                  const std::vector<float>& rope, dfref::Ring& ring) {
  const Dims& d = w.d;
  std::vector<float> tf(size_t(n) * d.fck());
  for (size_t i = 0; i < tf.size(); ++i) tf[i] = bf(taps[i]);
  std::vector<float> f(size_t(n) * d.H), h(size_t(n) * d.H);
  std::vector<uint16_t> x(size_t(n) * d.H);
  w.fc.apply(taps, tf.data(), n, f.data());
  const bool eng = nm == Numerics::Engine;
  for (uint32_t r = 0; r < n; ++r)
    dfref::norm(&f[size_t(r) * d.H], w.hidden_norm.data(), d.H, d.eps, &h[size_t(r) * d.H], &x[size_t(r) * d.H]);
  for (uint32_t l = 0; l < d.layers; ++l) {
    std::vector<float> qkv(size_t(n) * d.qkvn());
    w.layers[l].qkv.apply(x.data(), h.data(), n, qkv.data());
    for (uint32_t r = 0; r < n; ++r) {
      const uint32_t pos = pos0 + r, slot = pos % d.ring;
      for (uint32_t kh = 0; kh < d.NKV; ++kh) {
        float* kd = &ring.k[((size_t(l) * d.ring + slot) * d.NKV + kh) * d.HD];
        float* vd = &ring.v[((size_t(l) * d.ring + slot) * d.NKV + kh) * d.HD];
        dfref::head_prep(&qkv[size_t(r) * d.qkvn() + d.qn() + kh * d.HD], w.layers[l].k_norm.data(), d.HD, d.eps,
                         rope.data(), pos, kd);
        const float* vs = &qkv[size_t(r) * d.qkvn() + d.qn() + d.kvn() + kh * d.HD];
        for (uint32_t i = 0; i < d.HD; ++i) {
          vd[i] = eng ? rf(vs[i]) : vs[i];
          if (eng) kd[i] = rf(kd[i]);
        }
      }
    }
  }
  return std::vector<float>(h.end() - long(d.H), h.end());
}

struct DraftOut {
  std::vector<float> hidden;      // [M][H] the final-norm rows
  std::vector<uint32_t> cand;     // [S][k]
  std::vector<float> logit, lp;   // [S][k] the candidates' raw logits and log-probs
  std::vector<float> lattice;     // [S][k][k]
  Walk walk;                      // the first K slots
};

// One draft block at anchor id `anchor_id`, position p (the anchor row's context row is `anchor` - the projection's last),
// K drafts read from one 8-row block.
inline DraftOut draft(const Weights& w, Numerics nm, const dfref::Ring& ring, const std::vector<float>& rope,
                      const std::vector<float>& anchor, bool valid, uint32_t anchor_id, uint32_t p, uint32_t K, float T = 0.0f,
                      uint64_t seed = 0) {
  const Dims& d = w.d;
  const bool eng = nm == Numerics::Engine;
  const uint32_t M = d.M, H = d.H, Ka = M - 1;
  std::vector<float> resid(size_t(M) * H), h(size_t(M) * H);
  std::vector<uint16_t> x(size_t(M) * H);
  for (uint32_t r = 0; r < M; ++r)
    for (uint32_t i = 0; i < H; ++i)
      resid[size_t(r) * H + i] = (r == 0 ? bf(w.embed[size_t(anchor_id) * H + i]) : w.mask_row[i]) * d.in_scale;
  auto as_input = [&](const std::vector<float>& src, std::vector<uint16_t>& xb) {
    xb.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) xb[i] = rne(src[i]);
  };
  auto norm_rows = [&](const std::vector<float>& src, const std::vector<float>& wt) {
    for (uint32_t r = 0; r < M; ++r) dfref::norm(&src[size_t(r) * H], wt.data(), H, d.eps, &h[size_t(r) * H], &x[size_t(r) * H]);
  };
  const dfref::AttnDims ad = d.attn();
  std::vector<float> co, tmp, qkv(size_t(M) * d.qkvn()), q(size_t(M) * d.qn()), kb(size_t(M) * d.kvn()),
      vb(size_t(M) * d.kvn()), ao(size_t(M) * d.qn()), oo(size_t(M) * H), gu(size_t(M) * 2 * d.I), a(size_t(M) * d.I),
      dn(size_t(M) * H);
  std::vector<uint16_t> xb;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const Layer& L = w.layers[l];
    norm_rows(resid, L.in_norm);
    co.assign(size_t(M) * d.kpn(), 0.0f);
    L.akp.apply(x.data(), h.data(), M, co.data());
    tmp.assign(size_t(M) * H, 0.0f);
    dfref::conv(h.data(), co.data(), d.kpn(), &L.conv_base[0], M, H, d.group, d.ctaps, tmp.data());
    as_input(tmp, xb);
    L.qkv.apply(xb.data(), tmp.data(), M, qkv.data());
    for (uint32_t r = 0; r < M; ++r) {
      for (uint32_t hh = 0; hh < d.NH; ++hh)
        dfref::head_prep(&qkv[size_t(r) * d.qkvn() + hh * d.HD], L.q_norm.data(), d.HD, d.eps, rope.data(), p + r,
                         &q[(size_t(r) * d.NH + hh) * d.HD]);
      for (uint32_t kh = 0; kh < d.NKV; ++kh) {
        dfref::head_prep(&qkv[size_t(r) * d.qkvn() + d.qn() + kh * d.HD], L.k_norm.data(), d.HD, d.eps, rope.data(),
                         p + r, &kb[(size_t(r) * d.NKV + kh) * d.HD]);
        std::memcpy(&vb[(size_t(r) * d.NKV + kh) * d.HD], &qkv[size_t(r) * d.qkvn() + d.qn() + d.kvn() + kh * d.HD],
                    d.HD * 4);
      }
    }
    const size_t lay = size_t(d.ring) * d.kvn();
    attention_sink(ad, q.data(), kb.data(), vb.data(), &ring.k[l * lay], &ring.v[l * lay], p, Ka, L.sink.data(), ao.data());
    if (eng)
      for (float& v : ao) v = rf(v);   // ll_attn_reduce writes bf16
    as_input(ao, xb);
    L.o.apply(xb.data(), ao.data(), M, oo.data());
    tmp.assign(size_t(M) * H, 0.0f);
    dfref::conv(oo.data(), co.data() + size_t(d.ctaps) * d.groups(), d.kpn(), &L.conv_base[size_t(d.ctaps) * H], M, H,
                d.group, d.ctaps, tmp.data());
    for (size_t i = 0; i < resid.size(); ++i) resid[i] = resid[i] + tmp[i];
    norm_rows(resid, L.post_norm);
    co.assign(size_t(M) * d.kpn(), 0.0f);
    L.mkp.apply(x.data(), h.data(), M, co.data());
    tmp.assign(size_t(M) * H, 0.0f);
    dfref::conv(h.data(), co.data(), d.kpn(), &L.conv_base[size_t(2) * d.ctaps * H], M, H, d.group, d.ctaps, tmp.data());
    as_input(tmp, xb);
    L.gate_up.apply(xb.data(), tmp.data(), M, gu.data());
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t i = 0; i < d.I; ++i) {
        const float s = dfref::silu_mul(gu[size_t(r) * 2 * d.I + i], gu[size_t(r) * 2 * d.I + d.I + i]);
        a[size_t(r) * d.I + i] = eng ? rf(s) : s;
      }
    as_input(a, xb);
    L.down.apply(xb.data(), a.data(), M, dn.data());
    tmp.assign(size_t(M) * H, 0.0f);
    dfref::conv(dn.data(), co.data() + size_t(d.ctaps) * d.groups(), d.kpn(), &L.conv_base[size_t(3) * d.ctaps * H], M,
                H, d.group, d.ctaps, tmp.data());
    for (size_t i = 0; i < resid.size(); ++i) resid[i] = resid[i] + tmp[i];
  }
  norm_rows(resid, w.final_norm);
  DraftOut out;
  out.hidden = h;
  const uint32_t S = d.S, k = d.k;
  std::vector<float> logits(size_t(S) * w.head.N);
  std::vector<float> hrows(h.begin() + H, h.end());
  w.head.apply(x.data() + H, hrows.data(), S, logits.data());
  for (uint32_t s = 0; s < S; ++s) {
    const TopK t = topk_lse(&logits[size_t(s) * w.head.N], d.used, k);
    out.cand.insert(out.cand.end(), t.id.begin(), t.id.end());
    out.logit.insert(out.logit.end(), t.v.begin(), t.v.end());
    out.lp.insert(out.lp.end(), t.lp.begin(), t.lp.end());
  }
  out.lattice = head_lattice(w, out.cand.data(), out.lp.data(), hrows.data(), anchor.data(), valid);
  out.walk = walk(out.lattice, out.cand.data(), k, p, K, T, seed);
  return out;
}

}  // namespace llref
