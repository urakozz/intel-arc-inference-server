// Spec 26b Task 4 on the card: every ll_* kernel at nvidia/...-24GB-LiLiCorr's shapes (the binaries the LiLiCorr draft and
// projection lists bind, kernels::dflash::kLiLiCorr) against the host twin tests/kernels/lilicorr_ref.h, on random data -
// no checkpoint.
//
//   ll_gemv_bf16_g  at fc (24576 -> 4096), gate||up (4096 -> 24576), down (12288 -> 4096), M 8: BITWISE gemv_bf16's
//                   output x gs[n] (the same tiles and order on the same compiler - Review Focus 1's device half), and
//                   within the twin's GLin by cosine
//   ll_attn         the sinks at p 5, 1023, 1024, 1025, 2049, 131000 and K 7 / 3: within 1 bf16 ulp of the twin (exp),
//                   rows past K zero
//   ll_topk         131072 logits a row, 7 rows with planted ties / NaN / a -inf chunk / a huge value: ids and raw logits
//                   exact, log-probs within 1e-6 x max(1, |lse|) of the twin (exp / log), no NaN
//   ll_lattice      the whole head chain (gather + token_proj, pass_hidden_proj, the features, the sum, context_proj and
//                   validity, 2 layers of RMSNorm / in_proj / slot-biased attention / out_proj / MLP, the norms, the factor
//                   projection, out / in heads, the L2 norms, the scores) over random head weights of the real widths:
//                   the lattice's cosine >= 0.99999 and max |diff| printed; an invalid anchor equals the twin's too
//   ll_select       the walk over a random lattice, K 7: greedy - choices, ids, scores bitwise and ctrl.cur_token[1..7];
//                   sampled (T 0.9, a 64-bit seed) - the choices equal wherever the twin's top two noised scores are
//                   1e-5 apart (log / log1p)
//   ll_anchor       the projection's row proj_n - 1, bitwise
//
// Exit 0 on success; every comparison prints its count.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "common/repack.h"
#include "kernels/dflash_kernels.h"
#include "kernels/lilicorr_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/lilicorr_layout.h"
#include "loader/lilicorr_repack.h"   // LiliHostHead
#include "runtime/control.h"
#include "runtime/dflash/dflash_control.h"

namespace {

namespace kd = kernels::dflash;
using dfref::bf;
using dfref::rne;

const kd::Shape S = kd::kLiLiCorr;
const uint32_t M = kd::kM, H = S.H, HD = S.HD, NH = S.NH, NKV = S.NKV, QN = NH * HD, KVN = NKV * HD, RING = S.ring;
const uint32_t SL = kd::kLiliSlots, KC = kd::kLiliTopK, R = SL * KC, HL = kd::kLiliHidden, LH = kd::kLiliHeads, F = 1024,
               LMLP = 2048, VE = 2048;   // VE: the embedding stand-in's rows (the candidates are ids below it)

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem upload(const void* p, size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, std::max<size_t>(bytes, 4));
    if (bytes) imm.copy(m.ptr(), p, bytes);
    return m;
  }
  template <class T>
  l0::Mem up(const std::vector<T>& v) {
    return upload(v.data(), v.size() * sizeof(T));
  }
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.fill(m.ptr(), 0u, bytes);
    return m;
  }
  template <class T>
  std::vector<T> read(const l0::Mem& m, size_t n, size_t off = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, n * sizeof(T));
    return v;
  }
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy = 1) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy);
    list.close();
    q.execute(list, &f);
    f.wait();
  }
};

std::vector<float> rnd_f32(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = d(g);
  return v;
}
std::vector<uint16_t> rnd_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::vector<float> f = rnd_f32(n, lo, hi, seed);
  std::vector<uint16_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = rne(f[i]);
  return v;
}
int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

// --- ll_gemv_bf16_g -------------------------------------------------------------------------------------------------
int check_gemv_g(Dev& d) {
  int bad = 0;
  for (auto [K, N] : {std::pair<uint32_t, uint32_t>{24576, 4096}, {4096, 24576}, {12288, 4096}}) {
    const std::vector<uint16_t> rows = rnd_bf16(size_t(N) * K, -0.05f, 0.05f, K + N);
    std::vector<uint16_t> tiles(rows.size());
    common::repack_bf16_tiled(rows.data(), K, N, tiles.data());
    const std::vector<uint16_t> x = rnd_bf16(size_t(M) * K, -1, 1, K * 3 + N);
    std::vector<float> gs(N);
    for (uint32_t n = 0; n < N; ++n) gs[n] = n < N / 2 ? 0.0023719f : 0.00145903f;   // two parts' g (gate||up)
    l0::Mem bw = d.up(tiles), bx = d.up(x), bg = d.up(gs), y0 = d.zeros(size_t(M) * N * 4), y1 = d.zeros(size_t(M) * N * 4);
    const kernels::GemvBf16Tiling tl = kernels::gemv_bf16_tiling(N);
    l0::Module m0(d.ctx, kernels::path(kernels::gemv_bf16_variant(M, K, N, tl))),
        m1(d.ctx, kernels::path(kd::ll_gemv_g_variant(M, K, N)));
    l0::Kernel k0(m0, "gemv_bf16"), k1(m1, "ll_gemv_bf16_g");
    k0.group_size(tl.cols * tl.ksplit);
    k0.arg_ptr(0, bw.ptr());
    k0.arg_ptr(1, bx.ptr());
    k0.arg_ptr(2, y0.ptr());
    d.run(k0, N / tl.cols);
    k1.group_size(tl.cols * tl.ksplit);
    k1.arg_ptr(0, bw.ptr());
    k1.arg_ptr(1, bx.ptr());
    k1.arg_ptr(2, bg.ptr());
    k1.arg_ptr(3, y1.ptr());
    d.run(k1, N / tl.cols);
    const std::vector<float> a = d.read<float>(y0, size_t(M) * N), b = d.read<float>(y1, size_t(M) * N);
    int diff = 0;
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t n = 0; n < N; ++n) diff += b[size_t(m) * N + n] != a[size_t(m) * N + n] * gs[n];
    llref::GLin gl;
    gl.lin.K = K;
    gl.lin.N = N;
    gl.lin.bf16 = tiles;
    gl.lin.ksplit = tl.ksplit;
    gl.gs = gs;
    std::vector<float> want(size_t(M) * N);
    gl.apply(x.data(), nullptr, M, want.data());
    const double cs = dfref::cosine(b.data(), want.data(), want.size());
    std::printf("ll_gemv_bf16_g K %u N %u: %d of %u outputs differ from gemv_bf16 x gs (want 0); cosine to the twin %.9f\n",
                K, N, diff, M * N, cs);
    bad += diff != 0 || cs < 0.999999;
  }
  return bad;
}

// --- ll_attn --------------------------------------------------------------------------------------------------------
int check_attn(Dev& d) {
  l0::Module mod(d.ctx, kernels::path(kd::ll_attn_variant()));
  const uint32_t TGT = RING / S.pb, HS = (NH / NKV) / kd::kLiliHpw;
  int bad = 0;
  for (auto [p, K] : {std::pair<uint32_t, uint32_t>{5, 7}, {1023, 7}, {1024, 7}, {1025, 3}, {2049, 7}, {131000, 7}}) {
    const std::vector<float> q = rnd_f32(size_t(M) * QN, -2, 2, p), kb = rnd_f32(size_t(M) * KVN, -1, 1, p + 1),
                             vb = rnd_f32(size_t(M) * KVN, -1, 1, p + 2), sink = rnd_f32(NH, -2, 2, p + 3);
    const std::vector<uint16_t> rk = rnd_bf16(size_t(RING) * KVN, -1, 1, p + 4), rv = rnd_bf16(size_t(RING) * KVN, -1, 1, p + 5);
    std::vector<float> fk(rk.size()), fv(rv.size());
    for (size_t i = 0; i < rk.size(); ++i) {
      fk[i] = bf(rk[i]);
      fv[i] = bf(rv[i]);
    }
    runtime::Control c{};
    c.pos = p;
    runtime::dflash::LiliCtl dc{};
    dc.k = K;
    l0::Mem bc = d.upload(&c, sizeof c), bd = d.upload(&dc, sizeof dc), bq = d.up(q), bkb = d.up(kb), bvb = d.up(vb),
            brk = d.up(rk), brv = d.up(rv), bs = d.up(sink), part = d.zeros(size_t(NH) * TGT * M * (HD + 2) * 4),
            out = d.zeros(size_t(M) * QN * 2);
    const float scale = float(1.0 / std::sqrt(double(HD)));
    l0::Kernel kp(mod, "ll_attn_part"), kr(mod, "ll_attn_reduce");
    kp.group_size(128);
    kp.arg_ptr(0, bc.ptr());
    kp.arg_ptr(1, bd.ptr());
    kp.arg_ptr(2, bq.ptr());
    kp.arg_ptr(3, brk.ptr());
    kp.arg_ptr(4, brv.ptr());
    kp.arg(5, scale);
    kp.arg_ptr(6, part.ptr());
    d.run(kp, NKV * HS, TGT);
    kr.group_size(HD);
    kr.arg_ptr(0, bc.ptr());
    kr.arg_ptr(1, bd.ptr());
    kr.arg_ptr(2, part.ptr());
    kr.arg_ptr(3, bq.ptr());
    kr.arg_ptr(4, bkb.ptr());
    kr.arg_ptr(5, bvb.ptr());
    kr.arg_ptr(6, bs.ptr());
    kr.arg(7, scale);
    kr.arg_ptr(8, out.ptr());
    d.run(kr, NH, M);
    std::vector<float> want(size_t(M) * QN);
    llref::attention_sink({NH, NKV, HD, M, RING, S.win, S.pb, scale}, q.data(), kb.data(), vb.data(), fk.data(), fv.data(),
                          p, K, sink.data(), want.data());
    const std::vector<uint16_t> got = d.read<uint16_t>(out, size_t(M) * QN);
    int over = 0, zero = 0;
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t i = 0; i < QN; ++i) {
        const size_t j = size_t(r) * QN + i;
        if (r > K) zero += got[j] != 0;
        else over += ulps(got[j], rne(want[j])) > 1;
      }
    std::printf("ll_attn at p %u, K %u: %d values over 1 bf16 ulp, %d nonzero past K\n", p, K, over, zero);
    bad += over + zero;
  }
  return bad;
}

// --- ll_topk --------------------------------------------------------------------------------------------------------
int check_topk(Dev& d) {
  const uint32_t V = S.vocab, CHUNKS = V / 4096;
  std::vector<float> lg = rnd_f32(size_t(SL) * V, -8, 8, 77);
  lg[5] = lg[7] = lg[70000] = 30.0f;                                                   // row 0: a three-way tie
  lg[size_t(1) * V + 11] = NAN;                                                        // row 1: NaN
  for (uint32_t i = 4096; i < 8192; ++i) lg[size_t(2) * V + i] = -INFINITY;            // row 2: a -inf chunk
  lg[size_t(3) * V + 3] = 3.0e38f;                                                     // row 3: huge
  l0::Module mod(d.ctx, kernels::path(kd::ll_topk_variant()));
  l0::Mem bl = d.up(lg), part = d.zeros(size_t(SL) * CHUNKS * (2 * KC + 2) * 4), cand = d.zeros(size_t(SL) * KC * 4),
          lo = d.zeros(size_t(SL) * KC * 4), lp = d.zeros(size_t(SL) * KC * 4);
  l0::Kernel k1(mod, "ll_topk_stage1"), k2(mod, "ll_topk_stage2");
  k1.group_size(256);
  k1.arg_ptr(0, bl.ptr());
  k1.arg_ptr(1, part.ptr());
  d.run(k1, CHUNKS, SL);
  k2.group_size(256);
  k2.arg_ptr(0, part.ptr());
  k2.arg_ptr(1, cand.ptr());
  k2.arg_ptr(2, lo.ptr());
  k2.arg_ptr(3, lp.ptr());
  d.run(k2, SL);
  const std::vector<uint32_t> c = d.read<uint32_t>(cand, size_t(SL) * KC);
  const std::vector<float> v = d.read<float>(lo, size_t(SL) * KC), l = d.read<float>(lp, size_t(SL) * KC);
  int bad = 0;
  double worst = 0;
  for (uint32_t j = 0; j < SL; ++j) {
    const llref::TopK t = llref::topk_lse(&lg[size_t(j) * V], V, KC);
    for (uint32_t k = 0; k < KC; ++k) {
      bad += c[j * KC + k] != t.id[k] || v[j * KC + k] != t.v[k] || std::isnan(l[j * KC + k]);
      worst = std::fmax(worst, std::fabs(double(l[j * KC + k]) - t.lp[k]) / std::fmax(1.0, std::fabs(double(t.lse))));
    }
  }
  std::printf("ll_topk over %u logits x %u rows: %d ids / logits differ or NaN, log-probs within %.2e\n", V, SL, bad, worst);
  return bad + (worst > 1e-6);
}

// --- the head chain -------------------------------------------------------------------------------------------------
struct HeadMem {
  l0::Mem wt, gb;
  uint32_t K, N;
};
HeadMem head_mem(Dev& d, const loader::LiliHostHead& h) {
  std::vector<float> gb(h.gs);
  gb.insert(gb.end(), h.bias.begin(), h.bias.end());
  return {d.up(h.wt), d.up(gb), h.K, h.N};
}
loader::LiliHostHead rnd_head(uint32_t K, uint32_t N, float g, uint32_t seed) {
  loader::LiliHostHead h;
  h.K = K;
  h.N = N;
  h.wt = rnd_bf16(size_t(K) * N, -1.0f / std::sqrt(float(K)) / g, 1.0f / std::sqrt(float(K)) / g, seed);
  h.gs.assign(N, g);
  h.bias = rnd_f32(N, -0.1f, 0.1f, seed + 1);
  return h;
}
llref::HeadLin twin_of(const loader::LiliHostHead& h) {
  llref::HeadLin o;
  o.K = h.K;
  o.N = h.N;
  for (uint16_t v : h.wt) o.wt.push_back(bf(v));
  o.gs = h.gs;
  o.bias = h.bias;
  return o;
}

int check_lattice(Dev& d) {
  int bad = 0;
  // the weights: the real widths (g of an NVFP4 linear ~ 1e-3: W' then sits near the e2m1 x e4m3 range)
  llref::Weights w;
  llref::Dims& D = w.d;
  D.H = H;
  D.M = M;
  D.S = SL;
  D.k = KC;
  D.Hl = HL;
  D.lheads = LH;
  D.lmlp = LMLP;
  D.F = F;
  D.llayers = 2;
  D.eps = 1e-6f;
  w.embed = rnd_bf16(size_t(VE) * H, -1, 1, 5);
  std::vector<loader::LiliHostHead> host;
  auto add = [&](uint32_t K, uint32_t N, float g, uint32_t seed) {
    host.push_back(rnd_head(K, N, g, seed));
    return twin_of(host.back());
  };
  w.lil.tok = add(H, HL, 0.0021f, 11);
  w.lil.pass = add(H, HL, 0.0033f, 12);
  w.lil.fup = add(5, HL, 1.0f, 13);
  w.lil.fdown = add(HL, HL, 1.0f, 14);
  w.lil.ctx = add(H, HL, 0.0045f, 15);
  for (uint32_t l = 0; l < 2; ++l) {
    w.lil.inproj.push_back(add(HL, 3 * HL, 1.0f, 20 + l * 4));
    w.lil.outproj.push_back(add(HL, HL, 0.0017f, 21 + l * 4));
    w.lil.up.push_back(add(HL, LMLP, 0.0019f, 22 + l * 4));
    w.lil.down.push_back(add(LMLP, HL, 0.0013f, 23 + l * 4));
  }
  w.lil.factor = add(3 * HL, HL, 1.0f, 30);
  w.lil.outin = add(HL, 2 * F, 1.0f, 31);
  w.lil.aout = add(HL, F, 1.0f, 32);
  model::DFlashDesc desc;
  desc.block = M;
  desc.cand_topk = KC;
  desc.lil_hidden = HL;
  desc.lil_heads = LH;
  desc.lil_layers = 2;
  const loader::LiliHeadSmall so = loader::lili_head_small(desc);
  std::vector<float> small = rnd_f32(so.total, -0.5f, 0.5f, 40);
  for (size_t i = so.norms; i < so.total; ++i) small[i] = 1.0f + small[i] * 0.2f;   // the norms' weights near 1
  auto take = [&](size_t a, size_t n) { return std::vector<float>(small.begin() + long(a), small.begin() + long(a + n)); };
  w.lil.fn_w = take(so.fn_w, 5);
  w.lil.fn_b = take(so.fn_b, 5);
  w.lil.slot = take(so.slot, size_t(SL) * HL);
  w.lil.rank = take(so.rank, size_t(KC) * HL);
  w.lil.rel = take(so.rel, size_t(LH) * (2 * M - 1));
  w.lil.same = take(so.same, LH);
  for (uint32_t l = 0; l < 2; ++l) {
    const std::vector<float> a = take(so.attn_norm(l, HL), HL), m = take(so.mlp_norm(l, HL), HL);
    w.lil.attn_norm.insert(w.lil.attn_norm.end(), a.begin(), a.end());
    w.lil.mlp_norm.insert(w.lil.mlp_norm.end(), m.begin(), m.end());
  }
  w.lil.out_norm = take(so.out_norm, HL);
  w.lil.anchor_norm = take(so.anchor_norm, HL);
  // the inputs
  std::mt19937 gen(9);
  std::vector<uint32_t> cand(R);
  for (uint32_t r = 0; r < R; ++r) cand[r] = gen() % VE;
  std::vector<float> lp(R);
  for (uint32_t s = 0; s < SL; ++s) {
    float x = -0.2f;
    for (uint32_t i = 0; i < KC; ++i) lp[s * KC + i] = (x -= 0.3f + 0.2f * float(gen() % 7));
  }
  const std::vector<float> rows = rnd_f32(size_t(M) * H, -2, 2, 50), anchor = rnd_f32(H, -2, 2, 51);
  // the device chain (runtime/dflash/lilicorr_drafter.cc's order)
  l0::Module mod(d.ctx, kernels::path(kd::ll_lattice_variant()));
  std::vector<HeadMem> hm;
  for (const loader::LiliHostHead& h : host) hm.push_back(head_mem(d, h));
  l0::Mem bemb = d.up(w.embed), bc = d.up(cand), blp = d.up(lp), bh = d.up(rows), banc = d.up(anchor), bsm = d.up(small);
  l0::Mem tok = d.zeros(size_t(R) * HL * 4), pas = d.zeros(size_t(SL) * HL * 4), f5 = d.zeros(size_t(R) * 5 * 4),
          fu = d.zeros(size_t(R) * HL * 4), fm = d.zeros(size_t(R) * HL * 4), xh = d.zeros(size_t(R) * HL * 4),
          xn = d.zeros(size_t(R) * HL * 4), qkv = d.zeros(size_t(R) * 3 * HL * 4), o = d.zeros(size_t(R) * HL * 4),
          u = d.zeros(size_t(R) * LMLP * 4), a = d.zeros(size_t(HL) * 4), an = d.zeros(size_t(HL) * 4),
          fh = d.zeros(size_t(R) * HL * 4), ov = d.zeros(size_t(R) * 2 * F * 4), ao = d.zeros(size_t(F) * 4),
          lat = d.zeros(size_t(SL) * KC * KC * 4);
  auto gemm = [&](const l0::Mem& x, uint32_t xoff, uint32_t ldx, const HeadMem& W, const l0::Mem& y, uint32_t rows_,
                  uint32_t mode, uint32_t act) {
    l0::Kernel k(mod, "ll_gemm");
    k.group_size(kd::kLiliGemmWg);
    k.arg_ptr(0, x.ptr());
    k.arg(1, xoff);
    k.arg(2, ldx);
    k.arg_ptr(3, W.wt.ptr());
    k.arg_ptr(4, W.gb.ptr());
    k.arg_ptr(5, y.ptr());
    k.arg(6, rows_);
    k.arg(7, W.K);
    k.arg(8, W.N);
    k.arg(9, mode);
    k.arg(10, act);
    d.run(k, W.N / kd::kLiliGemmWg);
  };
  auto rmsnorm = [&](const l0::Mem& x, size_t woff, const l0::Mem& y, uint32_t rows_) {
    l0::Kernel k(mod, "ll_rmsnorm");
    k.group_size(256);
    k.arg_ptr(0, x.ptr());
    k.arg_ptr(1, bsm.ptr());
    k.arg(2, uint32_t(woff));
    k.arg_ptr(3, y.ptr());
    k.arg(4, HL);
    d.run(k, 1, rows_);
  };
  for (const bool valid : {true, false}) {
    runtime::dflash::LiliCtl dc{};
    dc.anchor_valid = valid ? 1 : 0;
    l0::Mem bd = d.upload(&dc, sizeof dc);
    {
      l0::Kernel k(mod, "ll_gemm_gather");
      k.group_size(kd::kLiliGemmWg);
      k.arg_ptr(0, bemb.ptr());
      k.arg_ptr(1, bc.ptr());
      k.arg_ptr(2, hm[0].wt.ptr());
      k.arg_ptr(3, hm[0].gb.ptr());
      k.arg_ptr(4, tok.ptr());
      k.arg(5, R);
      k.arg(6, H);
      k.arg(7, HL);
      k.arg(8, VE);
      d.run(k, HL / kd::kLiliGemmWg);
    }
    gemm(bh, H, H, hm[1], pas, SL, 0, 0);
    {
      l0::Kernel k(mod, "ll_feat");
      k.group_size(kd::kLiliRmax);
      k.arg_ptr(0, blp.ptr());
      k.arg_ptr(1, bsm.ptr());
      k.arg(2, uint32_t(so.fn_w));
      k.arg_ptr(3, f5.ptr());
      d.run(k, 1);
    }
    gemm(f5, 0, 5, hm[2], fu, R, 0, 1);
    gemm(fu, 0, HL, hm[3], fm, R, 0, 0);
    {
      l0::Kernel k(mod, "ll_sum");
      k.group_size(256);
      k.arg_ptr(0, tok.ptr());
      k.arg_ptr(1, pas.ptr());
      k.arg_ptr(2, fm.ptr());
      k.arg_ptr(3, bsm.ptr());
      k.arg(4, uint32_t(so.slot));
      k.arg(5, uint32_t(so.rank));
      k.arg_ptr(6, xh.ptr());
      d.run(k, (HL + 255) / 256, R);
    }
    gemm(banc, 0, H, hm[4], a, 1, 0, 0);
    {
      l0::Kernel k(mod, "ll_valid");
      k.group_size(256);
      k.arg_ptr(0, bd.ptr());
      k.arg_ptr(1, a.ptr());
      d.run(k, (HL + 255) / 256);
    }
    for (uint32_t l = 0; l < 2; ++l) {
      rmsnorm(xh, so.attn_norm(l, HL), xn, R);
      gemm(xn, 0, HL, hm[5 + l * 4], qkv, R, 0, 0);
      {
        l0::Kernel k(mod, "ll_lattn");
        k.group_size(kd::kLiliRmax);
        k.arg_ptr(0, qkv.ptr());
        k.arg_ptr(1, bsm.ptr());
        k.arg(2, uint32_t(so.rel));
        k.arg(3, uint32_t(so.same));
        k.arg(4, float(1.0 / std::sqrt(double(HL / LH))));
        k.arg_ptr(5, o.ptr());
        d.run(k, LH, R);
      }
      gemm(o, 0, HL, hm[6 + l * 4], xh, R, 1, 0);
      rmsnorm(xh, so.mlp_norm(l, HL), xn, R);
      gemm(xn, 0, HL, hm[7 + l * 4], u, R, 0, 1);
      gemm(u, 0, LMLP, hm[8 + l * 4], xh, R, 1, 0);
    }
    rmsnorm(xh, so.out_norm, xn, R);
    rmsnorm(a, so.anchor_norm, an, 1);
    {
      l0::Kernel k(mod, "ll_factor");
      k.group_size(kd::kLiliGemmWg);
      k.arg_ptr(0, xn.ptr());
      k.arg_ptr(1, an.ptr());
      k.arg_ptr(2, hm[13].wt.ptr());
      k.arg_ptr(3, hm[13].gb.ptr());
      k.arg_ptr(4, fh.ptr());
      d.run(k, HL / kd::kLiliGemmWg);
    }
    gemm(fh, 0, HL, hm[14], ov, R, 0, 0);
    auto l2 = [&](const l0::Mem& x, uint32_t halves, uint32_t rows_, uint32_t ld) {
      l0::Kernel k(mod, "ll_l2norm");
      k.group_size(256);
      k.arg_ptr(0, x.ptr());
      k.arg(1, F);
      k.arg(2, ld);
      k.arg(3, 1e-4f);
      d.run(k, halves, rows_);
    };
    l2(ov, 2, R, 2 * F);
    gemm(an, 0, HL, hm[15], ao, 1, 0, 0);
    l2(ao, 1, 1, F);
    {
      l0::Kernel k(mod, "ll_scores");
      k.group_size(KC);
      k.arg_ptr(0, ov.ptr());
      k.arg_ptr(1, ao.ptr());
      k.arg(2, F);
      k.arg(3, 8.0f);
      k.arg_ptr(4, lat.ptr());
      d.run(k, SL, KC);
    }
    const std::vector<float> got = d.read<float>(lat, size_t(SL) * KC * KC);
    D.lscale = 8.0f;
    D.veps = 1e-4f;
    const std::vector<float> want =
        llref::head_lattice(w, cand.data(), lp.data(), rows.data() + H, anchor.data(), valid);
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::fmax(worst, std::fabs(double(got[i]) - want[i]));
    const double cs = dfref::cosine(got.data(), want.data(), got.size());
    std::printf("ll_lattice chain (anchor %s): the lattice's cosine to the twin %.9f, max |diff| %.2e\n",
                valid ? "valid" : "invalid", cs, worst);
    bad += cs < 0.99999;
  }
  return bad;
}

// --- ll_select ------------------------------------------------------------------------------------------------------
int check_select(Dev& d) {
  l0::Module mod(d.ctx, kernels::path(kd::ll_select_variant()));
  std::mt19937 gen(3);
  std::vector<uint32_t> cand(R);
  for (uint32_t& c : cand) c = gen() % S.vocab;
  const std::vector<float> lp = rnd_f32(R, -9, -0.1f, 4), lo = rnd_f32(R, -5, 20, 5),
                           lat = rnd_f32(size_t(SL) * KC * KC, -8, 8, 6);
  int bad = 0;
  for (const float T : {0.0f, 0.9f}) {
    const uint64_t seed = 0x0123456789ABCDEFull;
    runtime::Control c{};
    c.pos = 4321;
    c.cur_token[0] = 777;
    runtime::dflash::LiliCtl dc{};
    dc.k = 7;
    dc.temperature = T;
    dc.seed_lo = uint32_t(seed);
    dc.seed_hi = uint32_t(seed >> 32);
    dc.anchor_valid = 1;
    l0::Mem bc = d.upload(&c, sizeof c), bd = d.upload(&dc, sizeof dc), bca = d.up(cand), blp = d.up(lp), blo = d.up(lo),
            blat = d.up(lat), out = d.zeros(sizeof(runtime::dflash::LiliDraftOut));
    l0::Kernel k(mod, "ll_select");
    k.group_size(kd::kLiliSelectWg);
    k.arg_ptr(0, bc.ptr());
    k.arg_ptr(1, bd.ptr());
    k.arg_ptr(2, bca.ptr());
    k.arg_ptr(3, blp.ptr());
    k.arg_ptr(4, blo.ptr());
    k.arg_ptr(5, blat.ptr());
    k.arg_ptr(6, out.ptr());
    d.run(k, 1);
    runtime::dflash::LiliDraftOut o;
    d.imm.copy(&o, out.ptr(), sizeof o);
    runtime::Control cc;
    d.imm.copy(&cc, bc.ptr(), sizeof cc);
    const llref::Walk wk = llref::walk(lat, cand.data(), KC, c.pos, 7, T, seed);
    int diff = 0, excused = 0;
    for (uint32_t s = 0; s < 7; ++s) {
      if (o.chosen[s] != wk.chosen[s] || o.ids[s] != wk.ids[s] || cc.cur_token[1 + s] != o.ids[s]) {
        if (T != 0.0f) {   // a near tie of the noised scores (log / log1p not correctly rounded) is excused
          std::vector<double> xs;
          const uint32_t prev = s == 0 ? 0 : wk.chosen[s - 1];
          for (uint32_t cI = 0; cI < KC; ++cI) {
            const float u = dfref::murmur3_uniform(seed ^ (uint64_t(s) << 32), c.pos + s + dfref::kDraftNoiseSalt,
                                                   cand[s * KC + cI]);
            xs.push_back(double(lat[(size_t(s) * KC + prev) * KC + cI]) / T - std::log(-std::log1p(-double(u))));
          }
          std::sort(xs.rbegin(), xs.rend());
          if (xs[0] - xs[1] < 1e-5) {
            ++excused;
            break;
          }
        }
        ++diff;
        break;
      }
      for (uint32_t cI = 0; cI < KC; ++cI) diff += o.scores[s][cI] != wk.scores[s * KC + cI];
    }
    for (uint32_t i = 0; i < R; ++i) diff += o.cand[i / KC][i % KC] != cand[i] || o.lp[i / KC][i % KC] != lp[i];
    diff += o.k != 7 || o.anchor != 777 || o.pos != 4321 || o.anchor_valid != 1;
    std::printf("ll_select at T %.1f: %d differences from the twin (%d near-ties excused)\n", double(T), diff, excused);
    bad += diff;
  }
  return bad;
}

// --- ll_anchor ------------------------------------------------------------------------------------------------------
int check_anchor(Dev& d) {
  l0::Module mod(d.ctx, kernels::path(kd::ll_lattice_variant()));
  const std::vector<float> h = rnd_f32(size_t(M) * H, -3, 3, 60);
  int bad = 0;
  for (uint32_t pn : {1u, 5u, 8u}) {
    runtime::dflash::LiliCtl dc{};
    dc.proj_n = pn;
    l0::Mem bd = d.upload(&dc, sizeof dc), bh = d.up(h), an = d.zeros(size_t(H) * 4);
    l0::Kernel k(mod, "ll_anchor");
    k.group_size(256);
    k.arg_ptr(0, bd.ptr());
    k.arg_ptr(1, bh.ptr());
    k.arg(2, H);
    k.arg_ptr(3, an.ptr());
    d.run(k, (H + 255) / 256);
    const std::vector<float> got = d.read<float>(an, H);
    bad += !std::equal(got.begin(), got.end(), h.begin() + size_t(pn - 1) * H);
  }
  std::printf("ll_anchor: the projection's last row at proj_n 1 / 5 / 8: %s\n", bad ? "DIFFERS" : "bitwise");
  return bad;
}

}  // namespace

int main() {
  dfref::threads = std::max(1u, std::thread::hardware_concurrency());
  Dev d;
  int bad = check_gemv_g(d);
  bad += check_attn(d);
  bad += check_topk(d);
  bad += check_lattice(d);
  bad += check_select(d);
  bad += check_anchor(d);
  if (bad) {
    std::printf("lilicorr_kernels_test: %d FAILURES\n", bad);
    return 1;
  }
  std::printf("lilicorr_kernels_test: PASS\n");
  return 0;
}
