// Spec 19b Task 2 on the card: every df_* kernel at z-lab/Qwen3.8-27B-DFlash2's shapes (the binaries the draft and
// projection lists bind, kernels::dflash::kQwen38) against the host twin tests/kernels/dflash_ref.h, on random data -
// no checkpoint.
//
//   df_embed       the anchor's row and the mask rows, x in_scale: bitwise
//   df_norm        bitwise (h and its bf16 copy): the B70 build's divide / sqrt are correctly rounded
//   df_conv_prep / df_conv_add   bitwise (no transcendental)
//   df_silu_mul    within 1 bf16 ulp (exp)
//   df_qkv_prep    q / k / v at positions 2041..2048 and 131065..131072: bitwise
//   df_ctx_store   5 rows at proj_pos 2045 (slots 2045, 2046, 2047, 0, 1: the ring's wrap) bitwise; the 3 rows past
//                  proj_n and every other slot untouched (Review Focus 1)
//   df_attn        at p 5, 2046, 2047, 2048, 2049, 4100, 131000 and K 7 / 3 / 1: bitwise the twin (both use df_exp /
//                  dfref::attn_exp, not the device's exp - spec 19f: outputs that cancel toward 0 turned the device
//                  exp's 1-2 fp32 ulps into up to 45 bf16 ulps), rows past K zero
//   df_topk        the full head (248320, the mask at 248077, planted ties / NaN / -inf rows) and a 32768-id draft
//                  vocabulary (indices mapped through an ascending id table): ids and values exact
//   df_select      the walk at rank 256 over 248320-row codebooks, K 7: greedy - ids, choices, scores bitwise and
//                  ctrl.cur_token[1..7] the ids; sampled (T 0.7, a 64-bit seed) - the choices equal wherever the
//                  twin's top two Gumbel scores are 1e-5 apart (log / log1p are not correctly rounded)
//
// Exit 0 on success; every comparison prints its count.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/dflash_kernels.h"
#include "kernels/dflash_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_control.h"

namespace {

namespace kd = kernels::dflash;
using dfref::bf;
using dfref::rne;

const kd::Shape S = kd::kQwen38;
const uint32_t M = kd::kM, H = S.H, HD = S.HD, NH = S.NH, NKV = S.NKV, QN = NH * HD, KVN = NKV * HD,
               QKVN = QN + 2 * KVN, I = S.I, RING = S.ring, G = H / S.group, KPN = 2 * S.ctaps * G;

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem upload(const void* p, size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.copy(m.ptr(), p, bytes);
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

struct Bins {
  l0::Module prep, attn, topk, topk_c, sel;
  Bins(Dev& d)
      : prep(d.ctx, kernels::path(kd::prep_variant())), attn(d.ctx, kernels::path(kd::attn_variant())),
        topk(d.ctx, kernels::path(kd::topk_variant(S.vocab, S.used, false))),
        topk_c(d.ctx, kernels::path(kd::topk_variant(32768, 32768, true))),
        sel(d.ctx, kernels::path(kd::select_variant(true))) {}
};

int check_prep(Dev& d, Bins& b) {
  int bad = 0;
  // df_embed
  {
    const std::vector<uint16_t> emb = rnd_bf16(size_t(4) * H, -1, 1, 1);   // a 4-row stand-in is enough to gather
    const std::vector<float> mask = rnd_f32(H, -1, 1, 2);
    runtime::Control c{};
    c.cur_token[0] = 3;
    l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof c);
    std::memcpy(ctrl.ptr(), &c, sizeof c);
    l0::Mem be = d.up(emb), bm = d.up(mask), resid = d.zeros(size_t(M) * H * 4);
    l0::Kernel k(b.prep, "df_embed");
    k.group_size(kd::kWg);
    k.arg_ptr(0, ctrl.ptr());
    k.arg_ptr(1, be.ptr());
    k.arg_ptr(2, bm.ptr());
    k.arg(3, 0.9f);
    k.arg_ptr(4, resid.ptr());
    d.run(k, 1, M);
    const std::vector<float> got = d.read<float>(resid, size_t(M) * H);
    int e = 0;
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t i = 0; i < H; ++i)
        e += got[size_t(r) * H + i] != (r == 0 ? bf(emb[size_t(3) * H + i]) : mask[i]) * 0.9f;
    std::printf("df_embed: %d of %u values differ\n", e, M * H);
    bad += e;
  }
  // df_norm
  {
    const std::vector<float> src = rnd_f32(size_t(M) * H, -3, 3, 3), w = rnd_f32(H, 0.5f, 1.5f, 4);
    l0::Mem bs = d.up(src), bw = d.up(w), bh = d.zeros(size_t(M) * H * 4), bx = d.zeros(size_t(M) * H * 2);
    l0::Kernel k(b.prep, "df_norm");
    k.group_size(kd::kWg);
    k.arg_ptr(0, bs.ptr());
    k.arg_ptr(1, bw.ptr());
    k.arg_ptr(2, bh.ptr());
    k.arg_ptr(3, bx.ptr());
    d.run(k, 1, M);
    const std::vector<float> h = d.read<float>(bh, size_t(M) * H);
    const std::vector<uint16_t> x = d.read<uint16_t>(bx, size_t(M) * H);
    std::vector<float> wh(H);
    std::vector<uint16_t> wx(H);
    int e = 0;
    for (uint32_t r = 0; r < M; ++r) {
      dfref::norm(&src[size_t(r) * H], w.data(), H, 1e-6f, wh.data(), wx.data());
      for (uint32_t i = 0; i < H; ++i) e += h[size_t(r) * H + i] != wh[i] || x[size_t(r) * H + i] != wx[i];
    }
    std::printf("df_norm: %d of %u values differ (bitwise bar)\n", e, M * H);
    bad += e;
  }
  // df_conv_prep / df_conv_add
  {
    const std::vector<float> src = rnd_f32(size_t(M) * H, -2, 2, 5), co = rnd_f32(size_t(M) * KPN, -0.3f, 0.3f, 6),
                             base = rnd_f32(size_t(2) * S.ctaps * H, 0, 1, 7),
                             r0 = rnd_f32(size_t(M) * H, -1, 1, 8);
    l0::Mem bs = d.up(src), bc = d.up(co), bb = d.up(base), bo = d.zeros(size_t(M) * H * 2), br = d.up(r0);
    l0::Kernel kp(b.prep, "df_conv_prep"), ka(b.prep, "df_conv_add");
    for (l0::Kernel* k : {&kp, &ka}) {
      k->group_size(kd::kWg);
      k->arg_ptr(0, bs.ptr());
      k->arg_ptr(1, bc.ptr());
      k->arg_ptr(2, bb.ptr());
    }
    kp.arg_ptr(3, bo.ptr());
    ka.arg_ptr(3, br.ptr());
    d.run(kp, (H + kd::kWg - 1) / kd::kWg, M);
    d.run(ka, (H + kd::kWg - 1) / kd::kWg, M);
    std::vector<float> w0(size_t(M) * H), w1(size_t(M) * H);
    dfref::conv(src.data(), co.data(), KPN, base.data(), M, H, S.group, S.ctaps, w0.data());
    dfref::conv(src.data(), co.data() + size_t(S.ctaps) * G, KPN, base.data() + size_t(S.ctaps) * H, M, H, S.group,
                S.ctaps, w1.data());
    const std::vector<uint16_t> o = d.read<uint16_t>(bo, size_t(M) * H);
    const std::vector<float> r = d.read<float>(br, size_t(M) * H);
    int e = 0;
    for (size_t i = 0; i < o.size(); ++i) e += o[i] != rne(w0[i]) || r[i] != r0[i] + w1[i];
    std::printf("df_conv_prep / df_conv_add: %d of %u values differ (bitwise bar)\n", e, 2 * M * H);
    bad += e;
  }
  // df_silu_mul
  {
    const std::vector<float> gu = rnd_f32(size_t(M) * 2 * I, -6, 6, 9);
    l0::Mem bg = d.up(gu), ba = d.zeros(size_t(M) * I * 2);
    l0::Kernel k(b.prep, "df_silu_mul");
    k.group_size(kd::kWg);
    k.arg_ptr(0, bg.ptr());
    k.arg_ptr(1, ba.ptr());
    d.run(k, (I + kd::kWg - 1) / kd::kWg, M);
    const std::vector<uint16_t> a = d.read<uint16_t>(ba, size_t(M) * I);
    int e = 0, off = 0;
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t i = 0; i < I; ++i) {
        const int u = ulps(a[size_t(r) * I + i], rne(dfref::silu_mul(gu[size_t(r) * 2 * I + i], gu[size_t(r) * 2 * I + I + i])));
        off += u != 0;
        e += u > 1;
      }
    std::printf("df_silu_mul: %d values 1 bf16 ulp off (exp), %d further (bar 0)\n", off, e);
    bad += e;
  }
  return bad;
}

int check_heads(Dev& d, Bins& b, const std::vector<float>& rope, const l0::Mem& brope) {
  int bad = 0;
  const std::vector<float> qkn = rnd_f32(size_t(2) * HD, 0.5f, 1.5f, 10);
  l0::Mem bq = d.up(qkn);
  for (uint32_t p : {2041u, 131065u}) {
    const std::vector<float> qkv = rnd_f32(size_t(M) * QKVN, -2, 2, 11 + p);
    runtime::Control c{};
    c.pos = p;
    l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof c);
    std::memcpy(ctrl.ptr(), &c, sizeof c);
    l0::Mem bin = d.up(qkv), q = d.zeros(size_t(M) * QN * 4), kb = d.zeros(size_t(M) * KVN * 4),
            vb = d.zeros(size_t(M) * KVN * 4);
    l0::Kernel k(b.prep, "df_qkv_prep");
    k.group_size(HD);
    k.arg_ptr(0, ctrl.ptr());
    k.arg_ptr(1, bin.ptr());
    k.arg_ptr(2, bq.ptr());
    k.arg_ptr(3, brope.ptr());
    k.arg_ptr(4, q.ptr());
    k.arg_ptr(5, kb.ptr());
    k.arg_ptr(6, vb.ptr());
    d.run(k, NH + 2 * NKV, M);
    const std::vector<float> gq = d.read<float>(q, size_t(M) * QN), gk = d.read<float>(kb, size_t(M) * KVN),
                             gv = d.read<float>(vb, size_t(M) * KVN);
    std::vector<float> want(HD);
    int e = 0;
    for (uint32_t r = 0; r < M; ++r) {
      for (uint32_t h = 0; h < NH; ++h) {
        dfref::head_prep(&qkv[size_t(r) * QKVN + h * HD], qkn.data(), HD, 1e-6f, rope.data(), p + r, want.data());
        for (uint32_t i = 0; i < HD; ++i) e += gq[(size_t(r) * NH + h) * HD + i] != want[i];
      }
      for (uint32_t h = 0; h < NKV; ++h) {
        dfref::head_prep(&qkv[size_t(r) * QKVN + QN + h * HD], qkn.data() + HD, HD, 1e-6f, rope.data(), p + r,
                         want.data());
        for (uint32_t i = 0; i < HD; ++i) {
          e += gk[(size_t(r) * NKV + h) * HD + i] != want[i];
          e += gv[(size_t(r) * NKV + h) * HD + i] != qkv[size_t(r) * QKVN + QN + KVN + h * HD + i];
        }
      }
    }
    std::printf("df_qkv_prep at positions %u..%u: %d values differ (bitwise bar)\n", p, p + M - 1, e);
    bad += e;
  }
  // df_ctx_store: 5 rows from 2045 across the wrap; the projection's k||v rows [L][M][2 KVN]
  {
    const uint32_t L = 5, pos0 = 2045, n = 5;
    const std::vector<float> kv = rnd_f32(size_t(L) * M * 2 * KVN, -2, 2, 12), qk = rnd_f32(size_t(L) * 2 * HD, 0.5f, 1.5f, 13);
    runtime::dflash::DFlashCtl dc{};
    dc.proj_pos = pos0;
    dc.proj_n = n;
    l0::Mem dctl(d.ctx, l0::MemKind::Shared, sizeof dc);
    std::memcpy(dctl.ptr(), &dc, sizeof dc);
    const size_t ring_elems = size_t(L) * RING * KVN;
    std::vector<uint16_t> init(ring_elems, 0x1234);
    l0::Mem bkv = d.up(kv), bqk = d.up(qk), rk = d.up(init), rv = d.up(init);
    l0::Kernel k(b.prep, "df_ctx_store");
    k.group_size(HD);
    k.arg_ptr(0, dctl.ptr());
    k.arg_ptr(1, bkv.ptr());
    k.arg_ptr(2, bqk.ptr());
    k.arg_ptr(3, brope.ptr());
    k.arg_ptr(4, rk.ptr());
    k.arg_ptr(5, rv.ptr());
    d.run(k, L * 2 * NKV, M);
    const std::vector<uint16_t> gk = d.read<uint16_t>(rk, ring_elems), gv = d.read<uint16_t>(rv, ring_elems);
    std::vector<uint16_t> wk = init, wv = init;
    std::vector<float> tmp(HD);
    for (uint32_t l = 0; l < L; ++l)
      for (uint32_t r = 0; r < n; ++r) {
        const uint32_t pos = pos0 + r, slot = pos % RING;
        for (uint32_t h = 0; h < NKV; ++h) {
          const float* src = &kv[((size_t(l) * M + r) * 2 * KVN) + h * HD];
          dfref::head_prep(src, &qk[(size_t(l) * 2 + 1) * HD], HD, 1e-6f, rope.data(), pos, tmp.data());
          for (uint32_t i = 0; i < HD; ++i) {
            wk[((size_t(l) * RING + slot) * NKV + h) * HD + i] = rne(tmp[i]);
            wv[((size_t(l) * RING + slot) * NKV + h) * HD + i] = rne(src[KVN + i]);
          }
        }
      }
    size_t e = 0;
    for (size_t i = 0; i < ring_elems; ++i) e += gk[i] != wk[i] || gv[i] != wv[i];
    std::printf("df_ctx_store: rows 2045..2049 into slots 2045..2047, 0, 1 - %zu of %zu ring values differ "
                "(bitwise; rows past proj_n and every other slot untouched)\n", e, 2 * ring_elems);
    bad += int(e != 0);
  }
  return bad;
}

int check_attn(Dev& d, Bins& b) {
  int bad = 0;
  const float scale = float(1.0 / std::sqrt(double(HD)));
  const size_t ring_elems = size_t(RING) * KVN;
  const std::vector<uint16_t> rk = rnd_bf16(ring_elems, -1, 1, 20), rv = rnd_bf16(ring_elems, -1, 1, 21);
  std::vector<float> fk(ring_elems), fv(ring_elems);
  for (size_t i = 0; i < ring_elems; ++i) {
    fk[i] = bf(rk[i]);
    fv[i] = bf(rv[i]);
  }
  l0::Mem brk = d.up(rk), brv = d.up(rv);
  const uint32_t tgt = RING / kd::kAttnPb;
  for (auto [p, K] : {std::pair<uint32_t, uint32_t>{5, 7}, {2046, 7}, {2047, 3}, {2048, 7}, {2049, 1}, {4100, 7},
                      {131000, 5}}) {
    const std::vector<float> q = rnd_f32(size_t(M) * QN, -1.5f, 1.5f, 22 + p), kb = rnd_f32(size_t(M) * KVN, -1, 1, 23 + p),
                             vb = rnd_f32(size_t(M) * KVN, -1, 1, 24 + p);
    runtime::Control c{};
    c.pos = p;
    runtime::dflash::DFlashCtl dc{};
    dc.k = K;
    l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof c), dctl(d.ctx, l0::MemKind::Shared, sizeof dc);
    std::memcpy(ctrl.ptr(), &c, sizeof c);
    std::memcpy(dctl.ptr(), &dc, sizeof dc);
    l0::Mem bq = d.up(q), bkb = d.up(kb), bvb = d.up(vb), part = d.zeros(size_t(NH) * tgt * M * (HD + 2) * 4),
            out = d.zeros(size_t(M) * QN * 2);
    l0::Kernel k1(b.attn, "df_attn_part"), k2(b.attn, "df_attn_reduce");
    k1.group_size(kd::kAttnWg);
    k1.arg_ptr(0, ctrl.ptr());
    k1.arg_ptr(1, dctl.ptr());
    k1.arg_ptr(2, bq.ptr());
    k1.arg_ptr(3, brk.ptr());
    k1.arg_ptr(4, brv.ptr());
    k1.arg(5, scale);
    k1.arg_ptr(6, part.ptr());
    k2.group_size(HD);
    k2.arg_ptr(0, ctrl.ptr());
    k2.arg_ptr(1, dctl.ptr());
    k2.arg_ptr(2, part.ptr());
    k2.arg_ptr(3, bq.ptr());
    k2.arg_ptr(4, bkb.ptr());
    k2.arg_ptr(5, bvb.ptr());
    k2.arg(6, scale);
    k2.arg_ptr(7, out.ptr());
    d.run(k1, NKV, tgt);
    d.run(k2, NH, M);
    std::vector<float> want(size_t(M) * QN);
    dfref::attention({NH, NKV, HD, M, RING, S.win, kd::kAttnPb, scale}, q.data(), kb.data(), vb.data(), fk.data(),
                     fv.data(), p, K, false, want.data());
    const std::vector<uint16_t> got = d.read<uint16_t>(out, size_t(M) * QN);
    int e = 0, worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const int u = ulps(got[i], rne(want[i]));
      e += u != 0;
      worst = std::max(worst, u);
    }
    std::printf("df_attn at p %u, K %u: %d of %zu values differ, worst %d bf16 ulp (bitwise bar)\n", p, K, e,
                got.size(), worst);
    bad += e;
  }
  return bad;
}

int check_topk(Dev& d, Bins& b) {
  int bad = 0;
  const uint32_t R = kd::kR;
  for (bool compact : {false, true}) {
    const uint32_t N = compact ? 32768 : S.vocab, used = compact ? N : S.used;
    std::vector<float> lg = rnd_f32(size_t(R) * N, -10, 10, compact ? 30 : 31);
    // planted: a top-value tie across chunks (row 0), the masked tail larger than everything (row 1), a NaN at the
    // top (row 2), a row of -inf with three finite values (row 3), many equal values (row 4)
    lg[5] = lg[100000 % N] = lg[N - 300] = 50.0f;
    if (!compact)
      for (uint32_t i = used; i < N; ++i) lg[size_t(1) * N + i] = 99.0f;
    lg[size_t(2) * N + 7] = NAN;
    for (uint32_t i = 0; i < N; ++i) lg[size_t(3) * N + i] = -INFINITY;
    lg[size_t(3) * N + 9] = 1.0f;
    lg[size_t(3) * N + 19] = 2.0f;
    lg[size_t(3) * N + N - 1] = 1.0f;
    for (uint32_t i = 0; i < N; i += 3) lg[size_t(4) * N + i] = 9.5f;
    std::vector<uint32_t> ids(N);
    for (uint32_t i = 0; i < N; ++i) ids[i] = i * 7 + 3;   // ascending, distinct
    const uint32_t chunks = (N + kd::kTopkChunk - 1) / kd::kTopkChunk;
    l0::Mem bl = d.up(lg), bid = d.up(ids), part = d.zeros(size_t(R) * chunks * kd::kTopK * 8),
            cand = d.zeros(size_t(R) * kd::kTopK * 4), un = d.zeros(size_t(R) * kd::kTopK * 4);
    l0::Module& m = compact ? b.topk_c : b.topk;
    l0::Kernel k1(m, "df_topk_stage1"), k2(m, "df_topk_stage2");
    k1.group_size(kd::kWg);
    k1.arg_ptr(0, bl.ptr());
    k1.arg_ptr(1, part.ptr());
    k2.group_size(kd::kWg);
    k2.arg_ptr(0, part.ptr());
    k2.arg_ptr(1, bid.ptr());
    k2.arg(2, 1.5f);
    k2.arg(3, 30.0f);
    k2.arg_ptr(4, cand.ptr());
    k2.arg_ptr(5, un.ptr());
    d.run(k1, chunks, R);
    d.run(k2, R);
    const std::vector<uint32_t> gc = d.read<uint32_t>(cand, size_t(R) * kd::kTopK);
    const std::vector<float> gu = d.read<float>(un, size_t(R) * kd::kTopK);
    int e = 0;
    for (uint32_t j = 0; j < R; ++j) {
      const std::vector<dfref::Cand> w =
          dfref::topk(&lg[size_t(j) * N], used, kd::kTopK, 1.5f, 30.0f, compact ? ids.data() : nullptr);
      for (uint32_t k = 0; k < kd::kTopK; ++k) {
        const bool same_v = gu[j * kd::kTopK + k] == w[k].v ||
                            std::fabs(gu[j * kd::kTopK + k] - w[k].v) <= 4e-7f * std::fabs(w[k].v);   // tanh
        e += gc[j * kd::kTopK + k] != w[k].i || !same_v;
      }
    }
    std::printf("df_topk (%s, N %u): %d of %u (id, value) pairs differ\n", compact ? "a draft vocabulary" : "the full head",
                N, e, R * kd::kTopK);
    bad += e;
  }
  return bad;
}

int check_select(Dev& d, Bins& b) {
  int bad = 0;
  const uint32_t R = kd::kR, V = S.vocab, RK = S.rank, K = 7;
  const std::vector<uint16_t> pred = rnd_bf16(size_t(V) * RK, -0.5f, 0.5f, 40), succ = rnd_bf16(size_t(V) * RK, -0.5f, 0.5f, 41);
  l0::Mem bp = d.up(pred), bs = d.up(succ);
  for (float T : {0.0f, 0.7f}) {
    std::mt19937 g(42);
    std::vector<uint32_t> cand(size_t(R) * kd::kTopK);
    for (uint32_t& c : cand) c = g() % V;
    const std::vector<float> unary = rnd_f32(cand.size(), 5, 15, 43), hid = rnd_f32(size_t(M) * RK, -1, 1, 44);
    runtime::Control c{};
    c.pos = 3000;
    c.cur_token[0] = 123456;
    runtime::dflash::DFlashCtl dc{};
    dc.k = K;
    dc.temperature = T;
    dc.seed_lo = 0x9E3779B9u;
    dc.seed_hi = 7;
    l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof c), dctl(d.ctx, l0::MemKind::Shared, sizeof dc),
        out(d.ctx, l0::MemKind::Shared, sizeof(runtime::dflash::DFlashDraftOut));
    std::memcpy(ctrl.ptr(), &c, sizeof c);
    std::memcpy(dctl.ptr(), &dc, sizeof dc);
    l0::Mem bc = d.up(cand), bu = d.up(unary), bh = d.up(hid);
    l0::Kernel k(b.sel, "df_select");
    k.group_size(kd::kWg);
    k.arg_ptr(0, ctrl.ptr());
    k.arg_ptr(1, dctl.ptr());
    k.arg_ptr(2, bc.ptr());
    k.arg_ptr(3, bu.ptr());
    k.arg_ptr(4, bh.ptr());
    k.arg_ptr(5, bp.ptr());
    k.arg_ptr(6, bs.ptr());
    k.arg_ptr(7, out.ptr());
    d.run(k, 1);
    const runtime::dflash::DFlashDraftOut* o = out.as<runtime::dflash::DFlashDraftOut>();
    const runtime::Control* co = ctrl.as<runtime::Control>();
    uint32_t prev = 123456;
    int e = 0, near = 0;
    const uint64_t seed = (uint64_t(7) << 32) | 0x9E3779B9u;
    for (uint32_t j = 0; j < K; ++j) {
      float s[16];
      dfref::edge_scores(pred.data(), succ.data(), RK, kd::kQs, prev, &hid[size_t(j + 1) * RK], &cand[j * 16],
                         &unary[j * 16], 16, s);
      const uint32_t ch = dfref::choose(s, &cand[j * 16], 16, T, seed, 3000 + j);
      for (uint32_t i = 0; i < 16; ++i) e += o->scores[j][i] != s[i] || o->cand[j][i] != cand[j * 16 + i];
      if (o->chosen[j] != ch) {
        // sampled: the device's log / log1p may differ by ulps - a difference is excused only at a near tie
        float x[16];
        for (uint32_t i = 0; i < 16; ++i)
          x[i] = s[i] / T - std::log(-std::log1p(-dfref::murmur3_uniform(seed, 3000 + j + dfref::kDraftNoiseSalt,
                                                                          cand[j * 16 + i])));
        if (T > 0 && std::fabs(x[ch] - x[o->chosen[j]]) <= 1e-5f * std::fabs(x[ch])) {
          ++near;
        } else {
          ++e;
        }
      }
      e += co->cur_token[1 + j] != o->ids[j] || o->ids[j] != cand[j * 16 + o->chosen[j]];
      prev = o->ids[j];
    }
    e += o->k != K || o->anchor != 123456 || o->pos != 3000;
    std::printf("df_select (T %.1f, K %u, rank 256, 248320-row codebooks): %d mismatches, %d near-tie choices\n",
                double(T), K, e, near);
    bad += e;
  }
  return bad;
}

}  // namespace

int main() {
  Dev d;
  Bins b(d);
  const std::vector<float> rope = dfref::rope_table(1e7, HD, 131072 + 16);
  l0::Mem brope = d.up(rope);
  int bad = check_prep(d, b);
  bad += check_heads(d, b, rope, brope);
  bad += check_attn(d, b);
  bad += check_topk(d, b);
  bad += check_select(d, b);
  if (bad) {
    std::printf("dflash_kernels_test: FAIL (%d)\n", bad);
    return 1;
  }
  std::printf("dflash_kernels_test: PASS\n");
  return 0;
}
