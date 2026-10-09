// Spec 21c Task 1: tests/kernels/qwen4exp_ref.h (the Qwen3.8-Flash-Next kernels' host twin) against
// tools/oracle/qwen4exp_fixture.py's outputs (tests/kernels/qwen4exp_fixture.h: qwen4exp_ref.py's restated ops and
// transformers 5.19.0's qwen4_exp modules) and against the semantics' properties (plan 21c Review Focus 1-5).
// Host only. The fixture's inputs come from the hash both sides share (q4ref::fixture_val).
//
//   (a) HC: the grouped norm of a coarse H bitwise; every elementwise step bitwise from torch's own previous
//       output (silu(down / 4), the inject, sigmoid(up), the mean of the 4 streams); the two linears through
//       this file's sums within one bf16 ulp (counted: torch's GEMM order is its own); the combine bitwise; inj
//       0 makes a block vanish and 2 doubles it; Review Focus 1 - three layers in the reference's order and in the
//       fused kernel order (the pending combine folded by the NEXT combine_norm, the PLE layer's _NN, the final
//       mixer's), H bitwise after every layer
//   (b) PLE: the ids of 64 positions bitwise (the EOS rule, positions 0 / 1, the id 248319); the block over 10
//       positions through the conv ring (rows p - 9, p - 6, p - 3, zero before 0) within 2 bf16 ulps of torch
//       (counted: the norms and the gate's dot are fine-valued sums)
//   (c) the indexer: q heads normed + roped bitwise at 0, 2047, 2050, 2051, 9000; compressed keys bitwise; the
//       loader's RoPE table against transformers' bf16 cos / sin (counted); Review Focus 3 - the 8-slot tail
//       ring at 1 and 4 rows a launch gives every block's key from its own raw keys, once
//   (d) the scores bitwise; the selection = the ruled one (score desc, block asc) on every crafted row, torch's
//       topk apart only inside exact ties; Review Focus 2's rows - 2049, 2050 (identity), 2051 (the first cut),
//       (p + 1) % 4 == 0, tails 0 and 3, the count 2048 + tail, the list ascending
//   (e) the route: ids = the ruled order exactly (torch's topk differs only inside exact ties of p), weights
//       within one bf16 ulp, the shared gate bitwise; the -1e30 row selects its ten
//   (f) the combine bitwise (grouped_mm's sum in the rank order + the gated shared expert)
//   (g) the gated GDN norm with the sigmoid gate bitwise
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/qwen4exp_fixture.h"
#include "kernels/qwen4exp_ref.h"
#include "loader/qwen4exp_ple_hash.h"
#include "loader/qwen4exp_repack.h"
#include "model/qwen4exp.h"

namespace {

namespace fx = qwen4exp_fixture;
namespace qr = q4ref;
using qr::f32;
using qr::rf;
using qr::rne;

int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

// Element-wise comparison: the count of differing elements and the worst ulp distance.
struct Diff {
  size_t n = 0, differ = 0;
  int worst = 0;
};
Diff diff16(const uint16_t* got, const uint16_t* want, size_t n) {
  Diff d;
  d.n = n;
  for (size_t i = 0; i < n; ++i) {
    const int u = ulps16(got[i], want[i]);
    d.differ += u != 0;
    d.worst = std::max(d.worst, u);
  }
  return d;
}

std::vector<float> one_plus(const std::vector<uint16_t>& w) {
  std::vector<float> o(w.size());
  for (size_t i = 0; i < w.size(); ++i) o[i] = 1.0f + f32(w[i]);
  return o;
}

// --- (a) ------------------------------------------------------------------------------------------------------------
// The fixture's weights: down||inject [324][10240] and up [10240][320] (torch's [out][in]), in gemv_bf16 tiles
// for this file's GEMVs ({K, N}: element (k, n) = W[n][k]).
struct HcWeights {
  std::vector<uint16_t> down_tiles, up_tiles;   // {10240, 336} (rows 324.. zero), {320, 10240}
  std::vector<float> w;                         // (1 + w) [10240]
};
HcWeights hc_weights() {
  HcWeights h;
  const std::vector<uint16_t> down = qr::fixture_bf16(false, 2, 324 * qr::kHcN, 1, 0.125f, 1.0f / 32);
  const std::vector<uint16_t> up = qr::fixture_bf16(false, 3, qr::kHcN * qr::kHcLow, 1, 0.125f, 1.0f / 16);
  h.down_tiles.assign(size_t(qr::kHcN) * qr::kHcDownRows, 0);
  for (uint32_t n = 0; n < 324; ++n)
    for (uint32_t k = 0; k < qr::kHcN; ++k) h.down_tiles[qr::tile_index(qr::kHcN, k, n)] = down[size_t(n) * qr::kHcN + k];
  h.up_tiles.assign(size_t(qr::kHcLow) * qr::kHcN, 0);
  for (uint32_t n = 0; n < qr::kHcN; ++n)
    for (uint32_t k = 0; k < qr::kHcLow; ++k) h.up_tiles[qr::tile_index(qr::kHcLow, k, n)] = up[size_t(n) * qr::kHcLow + k];
  h.w = one_plus(qr::fixture_bf16(false, 1, qr::kHcN, 1, 0.125f, 0.5f));
  return h;
}
// A {K, N} tiled GEMV row, one ascending fp32 chain per column (gemv_bf16's KSPLIT 1 order).
std::vector<float> gemv(const std::vector<uint16_t>& tiles, uint32_t K, uint32_t N, const uint16_t* x) {
  std::vector<float> o(N);
  for (uint32_t n = 0; n < N; ++n) {
    float a = 0.0f;
    for (uint32_t k = 0; k < K; ++k) a += f32(tiles[qr::tile_index(K, k, n)]) * f32(x[k]);
    o[n] = a;
  }
  return o;
}

void test_hc() {
  const HcWeights W = hc_weights();
  const std::vector<uint16_t> H = qr::fixture_bf16(true, 0, qr::kHcN, 1);
  const std::vector<uint16_t> y = qr::fixture_bf16(true, 4, qr::kHidden, 1, 0.25f);
  const std::vector<uint16_t> xn_w = qr::hex16(fx::kHcXn), dn_w = qr::hex16(fx::kHcDown), a_w = qr::hex16(fx::kHcA),
                              up_w = qr::hex16(fx::kHcUp), g_w = qr::hex16(fx::kHcG), x_w = qr::hex16(fx::kHcX),
                              inj_w = qr::hex16(fx::kHcInj), h1_w = qr::hex16(fx::kHcH1);
  CHECK(xn_w.size() == qr::kHcN && dn_w.size() == 324 && up_w.size() == qr::kHcN && x_w.size() == qr::kHidden);
  // the norm: coarse H, every sum of squares exact
  std::vector<uint16_t> Hc = H, xn(qr::kHcN);
  qr::hc_combine_norm(Hc.data(), qr::HcSrc::None, nullptr, nullptr, W.w.data(), xn.data());
  CHECK(Hc == H);
  CHECK(xn == xn_w);
  // the elementwise steps from torch's own outputs
  for (uint32_t j = 0; j < qr::kHcLow; ++j) CHECK(rne(qr::hc_act(f32(dn_w[j]))) == a_w[j]);
  float inj[4];
  for (uint32_t s = 0; s < 4; ++s) {
    inj[s] = qr::hc_inj(f32(dn_w[qr::kHcLow + s]));
    CHECK(rne(inj[s]) == inj_w[s]);
  }
  for (uint32_t c = 0; c < qr::kHcN; ++c) CHECK(rne(qr::sigmoid_t(f32(up_w[c]))) == g_w[c]);
  for (uint32_t c = 0; c < qr::kHidden; ++c) {
    float p[4];
    for (uint32_t s = 0; s < 4; ++s) p[s] = rf(f32(g_w[s * qr::kHidden + c]) * f32(xn_w[s * qr::kHidden + c]));
    CHECK(rne((((p[0] + p[1]) + p[2]) + p[3]) / 4.0f) == x_w[c]);
  }
  // the linears through this file's sums: within one ulp of torch's GEMM, counted
  const std::vector<float> dn = gemv(W.down_tiles, qr::kHcN, 324, xn.data());
  std::vector<uint16_t> dn_b(324);
  for (uint32_t j = 0; j < 324; ++j) dn_b[j] = rne(dn[j]);
  const Diff dd = diff16(dn_b.data(), dn_w.data(), 324);
  CHECK(dd.worst <= 1);
  std::vector<float> dn_torch(324);
  for (uint32_t j = 0; j < 324; ++j) dn_torch[j] = f32(dn_w[j]);
  std::vector<uint16_t> x(qr::kHidden);
  float inj2[4];
  qr::hc_up_mix(dn_torch.data(), true, W.up_tiles.data(), xn_w.data(), x.data(), inj2);
  const Diff dx = diff16(x.data(), x_w.data(), qr::kHidden);
  CHECK(dx.worst <= 1);
  for (uint32_t s = 0; s < 4; ++s) CHECK(inj2[s] == inj[s]);
  // the combine, bitwise; inj 0 vanishes, 2 doubles (H and y coarse: every sum exact)
  std::vector<uint16_t> H1 = H;
  qr::hc_combine_norm(H1.data(), qr::HcSrc::Y, y.data(), inj, W.w.data(), nullptr);
  CHECK(H1 == h1_w);
  const float z[4] = {0, 0, 0, 0}, one[4] = {1, 1, 1, 1}, two[4] = {2, 2, 2, 2};
  std::vector<uint16_t> H0 = H, Hone = H, Htwo = H;
  qr::hc_combine_norm(H0.data(), qr::HcSrc::Y, y.data(), z, W.w.data(), nullptr);
  qr::hc_combine_norm(Hone.data(), qr::HcSrc::Y, y.data(), one, W.w.data(), nullptr);
  qr::hc_combine_norm(Htwo.data(), qr::HcSrc::Y, y.data(), two, W.w.data(), nullptr);
  CHECK(H0 == H);
  for (uint32_t c = 0; c < qr::kHcN; ++c) CHECK(f32(Htwo[c]) - f32(H[c]) == 2.0f * (f32(Hone[c]) - f32(H[c])));
  std::printf("  HC: xn, silu, inject, sigmoid, the mean, the combine bitwise; down %zu / 324 and x %zu / 2560 one ulp "
              "from torch's GEMM order; inj 0 vanishes, 2 doubles\n", dd.differ, dx.differ);
}

// Review Focus 1: three layers (layer 1 the PLE layer) in the reference's order and the fused order. The mixer
// and the MoE are stand-ins (y = rne(0.75 x) and rne(-0.5 x)); the PLE block a stand-in that adds rne(H / 8).
void test_fused_order() {
  const HcWeights W = hc_weights();
  const std::vector<uint16_t> emb = qr::fixture_bf16(false, 5, qr::kHidden, 1, 0.125f, 1.0f);
  const auto mixer = [](const std::vector<uint16_t>& x) {
    std::vector<uint16_t> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = rne(0.75f * f32(x[i]));
    return y;
  };
  const auto moe = [](const std::vector<uint16_t>& x) {
    std::vector<uint16_t> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = rne(-0.5f * f32(x[i]));
    return y;
  };
  const auto ple = [](std::vector<uint16_t>& H) {
    for (uint16_t& h : H) h = rne(f32(h) + rf(f32(h) / 8.0f));
  };
  const auto hc = [&](const std::vector<uint16_t>& Hin, std::vector<uint16_t>& x, float* inj) {   // HC(H) -> x, inj
    std::vector<uint16_t> H = Hin, xn(qr::kHcN);
    qr::hc_combine_norm(H.data(), qr::HcSrc::None, nullptr, nullptr, W.w.data(), xn.data());
    const std::vector<float> d = gemv(W.down_tiles, qr::kHcN, qr::kHcDownRows, xn.data());
    x.assign(qr::kHidden, 0);
    qr::hc_up_mix(d.data(), true, W.up_tiles.data(), xn.data(), x.data(), inj);
  };
  const uint32_t L = 3, ple_layer = 1;
  for (float inj_force : {-1.0f, 0.0f, 2.0f}) {   // -1: the computed inj; 0 / 2: crafted
    // the reference: per layer [PLE], attn HC -> mixer -> H0 + y (x) inj, MLP HC -> MoE -> combine
    std::vector<std::vector<uint16_t>> ref_H;
    std::vector<uint16_t> H(qr::kHcN);
    for (uint32_t s = 0; s < 4; ++s) std::copy(emb.begin(), emb.end(), H.begin() + s * qr::kHidden);
    for (uint32_t l = 0; l < L; ++l) {
      if (l == ple_layer) ple(H);
      for (int side = 0; side < 2; ++side) {
        std::vector<uint16_t> x;
        float inj[4];
        hc(H, x, inj);
        if (inj_force >= 0) std::fill(inj, inj + 4, inj_force);
        const std::vector<uint16_t> y = side == 0 ? mixer(x) : moe(x);
        qr::hc_combine_norm(H.data(), qr::HcSrc::Y, y.data(), inj, W.w.data(), nullptr);
      }
      ref_H.push_back(H);
    }
    // the fused order (runtime/qwen4exp/qwen4exp_capture.cc's walk): each combine_norm folds the PENDING y of the
    // block before, then norms; the PLE layer materialises (_Y_NN) first and its attn side folds nothing (_X)
    std::vector<uint16_t> G(qr::kHcN), xn(qr::kHcN), pend_y;
    float pend_inj[4] = {};
    qr::HcSrc src = qr::HcSrc::Embed;
    std::vector<std::vector<uint16_t>> got_H;
    for (uint32_t l = 0; l < L; ++l) {
      if (l == ple_layer) {
        qr::hc_combine_norm(G.data(), qr::HcSrc::Y, pend_y.data(), pend_inj, W.w.data(), nullptr);   // _Y_NN
        got_H.push_back(G);   // layer l - 1's output, materialised
        ple(G);
        src = qr::HcSrc::None;
      }
      for (int side = 0; side < 2; ++side) {
        qr::hc_combine_norm(G.data(), src, src == qr::HcSrc::Embed ? emb.data() : pend_y.data(), pend_inj, W.w.data(),
                            xn.data());
        if (side == 0 && src == qr::HcSrc::Y) got_H.push_back(G);   // layer l - 1's output, folded here
        const std::vector<float> d = gemv(W.down_tiles, qr::kHcN, qr::kHcDownRows, xn.data());
        std::vector<uint16_t> x(qr::kHidden);
        float inj[4];
        qr::hc_up_mix(d.data(), true, W.up_tiles.data(), xn.data(), x.data(), inj);
        if (inj_force >= 0) std::fill(inj, inj + 4, inj_force);
        pend_y = side == 0 ? mixer(x) : moe(x);
        std::copy(inj, inj + 4, pend_inj);
        src = qr::HcSrc::Y;
      }
    }
    qr::hc_combine_norm(G.data(), qr::HcSrc::Y, pend_y.data(), pend_inj, W.w.data(), xn.data());   // the final mixer's
    got_H.push_back(G);
    CHECK_EQ(got_H.size(), size_t(L));
    for (uint32_t l = 0; l < L; ++l) CHECK(got_H[l] == ref_H[l]);
    if (inj_force == 0.0f) {   // every block vanished: H is the embedding (+ the PLE stand-in on layer 1)
      std::vector<uint16_t> e(qr::kHcN);
      for (uint32_t s = 0; s < 4; ++s) std::copy(emb.begin(), emb.end(), e.begin() + s * qr::kHidden);
      CHECK(ref_H[0] == e);
      ple(e);
      CHECK(ref_H[2] == e);
    }
  }
  std::puts("  Review Focus 1: three layers, the reference's order = the fused order, H bitwise after every layer "
            "(computed inj, 0: every block vanishes, 2)");
}

// --- (b) ------------------------------------------------------------------------------------------------------------
void test_ple() {
  const std::array<uint64_t, 3> mult = {fx::kPleMult[0], fx::kPleMult[1], fx::kPleMult[2]};
  const std::vector<uint64_t> sizes(std::begin(fx::kPleSizes), std::end(fx::kPleSizes)),
      offs(std::begin(fx::kPleOffsets), std::end(fx::kPleOffsets));
  CHECK(sizes == loader::q4_ple_primes(20000000, 16, 0));
  CHECK(offs == loader::q4_ple_offsets(sizes));
  CHECK(mult == loader::q4_ple_multipliers(248320, 3, 0, 1234));
  const std::vector<uint64_t> want = qr::hex64(fx::kPleIds);
  CHECK_EQ(want.size(), size_t(fx::kPleSeqN) * 16);
  for (uint32_t p = 0; p < fx::kPleSeqN; ++p) {
    const loader::Q4PleHistory h = qr::ple_history(p, [](uint32_t q) { return fx::kPleSeq[q]; });
    const loader::Q4PleHistory l = loader::q4_ple_history(fx::kPleSeq, p, qr::kPleEos);
    CHECK(h.t0 == l.t0 && h.t1 == l.t1 && h.t2 == l.t2);
    const std::array<uint64_t, 16> ids = loader::q4_ple_ids(h.t0, h.t1, h.t2, mult, sizes, offs);
    for (uint32_t k = 0; k < 16; ++k) CHECK_EQ(ids[k], want[size_t(p) * 16 + k]);
  }
  // the block, position by position through the 16-row ring
  const uint32_t T = fx::kPleBlockPos, N = qr::kHcN;
  const std::vector<float> wk = one_plus(qr::fixture_bf16(false, 20, N, 3, 0.125f, 0.5f)),
                           wq = one_plus(qr::fixture_bf16(false, 21, N, 3, 0.125f, 0.5f)),
                           wc = one_plus(qr::fixture_bf16(false, 22, N, 3, 0.125f, 0.5f));
  const std::vector<uint16_t> taps_b = qr::fixture_bf16(false, 23, N * 4, 3, 0.125f, 0.5f);
  std::vector<float> taps(taps_b.size());
  for (size_t i = 0; i < taps.size(); ++i) taps[i] = f32(taps_b[i]);
  const std::vector<uint16_t> want_H = qr::hex16(fx::kPleH);
  CHECK_EQ(want_H.size(), size_t(T) * N);
  std::vector<std::vector<uint16_t>> ring(qr::kPleRing, std::vector<uint16_t>(N, 0));
  Diff all;
  for (uint32_t p = 0; p < T; ++p) {
    std::vector<uint16_t> H = qr::fixture_bf16(true, 25, N, 3, 0.125f, 1.0f, p * N);
    const std::vector<uint16_t> kv_b = qr::fixture_bf16(false, 24, N + qr::kHidden, 3, 0.125f, 2.0f, p * (N + qr::kHidden));
    std::vector<float> kv(kv_b.size());
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = f32(kv_b[i]);
    const uint16_t* hist[3];
    for (uint32_t j = 0; j < 3; ++j) {
      const int q = int(p) - 9 + 3 * int(j);
      hist[j] = q >= 0 ? ring[q % qr::kPleRing].data() : nullptr;
    }
    std::vector<uint16_t> out(N);
    qr::ple_block(H.data(), kv.data(), wk.data(), wq.data(), wc.data(), taps.data(), hist, out.data());
    ring[p % qr::kPleRing] = out;
    const Diff d = diff16(H.data(), want_H.data() + size_t(p) * N, N);
    all.n += d.n;
    all.differ += d.differ;
    all.worst = std::max(all.worst, d.worst);
  }
  CHECK(all.worst <= 2);
  std::printf("  PLE: 64 positions' ids bitwise (EOS at 5, 6, 20; 248319; p 0 / 1); the block over %u positions "
              "through the ring: %zu / %zu elements differ from torch, worst %d ulp (its norms' / gate's sum order)\n",
              T, all.differ, all.n, all.worst);
}

// --- (c) ------------------------------------------------------------------------------------------------------------
std::vector<float> cs_row(const std::vector<uint16_t>& cos, const std::vector<uint16_t>& sin, uint32_t i) {
  std::vector<float> r(64);
  for (uint32_t k = 0; k < 32; ++k) {
    r[k] = f32(cos[size_t(i) * 64 + k]);
    r[32 + k] = f32(sin[size_t(i) * 64 + k]);
  }
  return r;
}

void test_indexer() {
  const std::vector<float> wq = one_plus(qr::fixture_bf16(false, 30, 128, 4, 0.125f, 0.5f)),
                           wk = one_plus(qr::fixture_bf16(false, 31, 128, 4, 0.125f, 0.5f));
  const std::vector<uint16_t> cos = qr::hex16(fx::kIdxCos), sin = qr::hex16(fx::kIdxSin), qw = qr::hex16(fx::kIdxQ);
  const model::Qwen4ExpDesc& d = model::qwen4exp();
  const std::vector<float> table = loader::q4_rope_table(d, 9216);
  uint32_t rope_diff = 0;
  for (uint32_t i = 0; i < fx::kIdxPosN; ++i) {
    std::vector<float> row(640, 0.0f);
    const std::vector<uint16_t> qin = qr::fixture_bf16(true, 32, 512, 4, 0.25f, 1.0f, i * 512);
    for (uint32_t k = 0; k < 512; ++k) row[k] = f32(qin[k]);
    const std::vector<float> cs = cs_row(cos, sin, i);
    float q[512];
    qr::qsa_q(row.data(), wq.data(), cs.data(), q);
    for (uint32_t k = 0; k < 512; ++k) CHECK(rne(q[k]) == qw[size_t(i) * 512 + k]);
    for (uint32_t k = 0; k < 64; ++k) rope_diff += table[size_t(fx::kIdxPos[i]) * 64 + k] != cs[k];
  }
  const std::vector<uint16_t> kc = qr::hex16(fx::kKeyCos), ks = qr::hex16(fx::kKeySin), kw = qr::hex16(fx::kKeys);
  for (uint32_t i = 0; i < 3; ++i) {
    const std::vector<uint16_t> raw = qr::fixture_bf16(true, 33, 512, 4, 0.25f, 1.0f, i * 512);
    const uint16_t* r4[4] = {raw.data(), raw.data() + 128, raw.data() + 256, raw.data() + 384};
    const std::vector<float> cs = cs_row(kc, ks, i);
    uint16_t key[128];
    qr::qsa_block_key(r4, wk.data(), cs.data(), key);
    for (uint32_t k = 0; k < 128; ++k) CHECK(key[k] == kw[size_t(i) * 128 + k]);
    for (uint32_t k = 0; k < 64; ++k) rope_diff += table[size_t(4 * fx::kKeyBlocks[i]) * 64 + k] != cs[k];
  }
  CHECK(rope_diff == 0);
  // Review Focus 3: the 8-slot tail ring - rows in launches of 1 and of 4; a completing row forms its block's key
  // from this launch's rows directly and the older rows from the ring (q4_qsa_prep's rule) - every key equals the
  // one formed from all raw keys.
  const uint32_t P = 41;
  std::vector<std::vector<uint16_t>> raw(P);
  for (uint32_t p = 0; p < P; ++p) raw[p] = qr::fixture_bf16(false, 34, 128, 4, 0.125f, 2.0f, p * 128);
  const std::vector<float> table0 = loader::q4_rope_table(d, 256);
  std::vector<std::vector<uint16_t>> want((P + 1) / 4, std::vector<uint16_t>(128));
  for (uint32_t b = 0; b < want.size(); ++b) {
    const uint16_t* r4[4] = {raw[4 * b].data(), raw[4 * b + 1].data(), raw[4 * b + 2].data(), raw[4 * b + 3].data()};
    qr::qsa_block_key(r4, wk.data(), table0.data() + size_t(4 * b) * 64, want[b].data());
  }
  for (uint32_t M : {1u, 4u}) {
    std::vector<std::vector<uint16_t>> tail(qr::kTailSlots, std::vector<uint16_t>(128, 0));
    std::vector<int> formed(want.size(), 0);
    for (uint32_t pos = 0; pos < P; pos += M) {
      const uint32_t n = std::min(M, P - pos);
      std::vector<std::vector<uint16_t>> next = tail;   // the launch's writes land at its end (no row reads them)
      for (uint32_t m = 0; m < n; ++m) {
        const uint32_t p = pos + m;
        next[p % qr::kTailSlots] = raw[p];
        if ((p + 1) % 4 != 0) continue;
        const uint32_t b = (p + 1) / 4 - 1;
        const uint16_t* r4[4];
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t q = 4 * b + j;
          r4[j] = q >= pos ? raw[q].data() : tail[q % qr::kTailSlots].data();   // this launch's rows, or the ring
        }
        uint16_t key[128];
        qr::qsa_block_key(r4, wk.data(), table0.data() + size_t(4 * b) * 64, key);
        CHECK(std::equal(key, key + 128, want[b].begin()));
        ++formed[b];
      }
      tail = next;
    }
    for (int f : formed) CHECK_EQ(f, 1);
  }
  std::printf("  indexer: q at 0 / 2047 / 2050 / 2051 / 9000 and the keys of blocks 0 / 511 / 2250 bitwise; the "
              "loader's RoPE table = transformers' bf16 cos / sin; the 8-slot tail ring forms every key once at 1 and 4 "
              "rows a launch\n");
}

// --- (d) ------------------------------------------------------------------------------------------------------------
void test_select() {
  const std::vector<uint16_t> qb = qr::fixture_bf16(true, 40, 512, 5, 0.25f);
  const std::vector<uint16_t> keys = qr::fixture_bf16(true, 41, fx::kSelN * 128, 5, 0.25f);
  std::vector<float> q(512);
  for (uint32_t i = 0; i < 512; ++i) q[i] = f32(qb[i]);
  const std::vector<uint32_t> sw = qr::hex32(fx::kSelScores);
  for (uint32_t b = 0; b < fx::kSelN; ++b) CHECK(qr::as_u32(qr::qsa_score(q.data(), keys.data() + size_t(b) * 128)) == sw[b]);
  const std::vector<uint32_t> rows = qr::hex32(fx::kSelRows), tsel = qr::hex32(fx::kSelTorch), rsel = qr::hex32(fx::kSelRuled);
  uint32_t torch_apart = 0;
  for (uint32_t r = 0; r < 4; ++r) {
    const uint32_t n = fx::kSelRowsN[r];
    std::vector<float> s(n);
    for (uint32_t b = 0; b < n; ++b) s[b] = qr::as_f32(rows[size_t(r) * fx::kSelN + b]);
    for (uint32_t tail = 0; tail < 4; ++tail) {
      const uint32_t p = 4 * n - 1 + tail;   // n complete blocks, `tail` positions of the open one
      const qr::Selection sel = qr::qsa_select(s.data(), p);
      CHECK_EQ(sel.count(), qr::qsa_count(p));
      CHECK(std::is_sorted(sel.list.begin(), sel.list.end()));
      CHECK(std::adjacent_find(sel.list.begin(), sel.list.end()) == sel.list.end());
      if (tail > 0 || n <= 512) CHECK_EQ(sel.list.back(), p);   // the open block's tail is always attended
      if (n <= 512) {
        CHECK_EQ(sel.count(), p + 1);
        CHECK(std::isinf(sel.s512) && sel.s512 > 0 && std::isinf(sel.s513) && sel.s513 < 0);
        continue;
      }
      CHECK_EQ(sel.count(), 2048 + tail);
      for (uint32_t k = 0; k < 512; ++k) CHECK_EQ(sel.blocks[k], rsel[size_t(r) * 512 + k]);
      if (tail == 0) {   // torch's topk: apart only inside the exact tie at the cut
        std::vector<uint32_t> tb(tsel.begin() + size_t(r) * 512, tsel.begin() + size_t(r) * 512 + 512), only;
        std::set_difference(tb.begin(), tb.end(), sel.blocks.begin(), sel.blocks.end(), std::back_inserter(only));
        for (uint32_t b : only) CHECK(s[b] == sel.s512);
        CHECK(sel.s512 == sel.s513 || only.empty());
        torch_apart += uint32_t(only.size());
      }
    }
  }
  // Review Focus 2's rows on a strictly decreasing score row (no ties): 2049, 2050 identity; 2051 the first cut
  // (513 complete blocks, p's own block a candidate - it loses: the lowest score); a row whose own block wins.
  std::vector<float> dec(700);
  for (uint32_t b = 0; b < dec.size(); ++b) dec[b] = 1000.0f - float(b);
  for (uint32_t p : {2049u, 2050u}) {
    const qr::Selection sel = qr::qsa_select(dec.data(), p);
    CHECK_EQ(sel.count(), p + 1);
    for (uint32_t i = 0; i <= p; ++i) CHECK_EQ(sel.list[i], i);
  }
  {
    const qr::Selection sel = qr::qsa_select(dec.data(), 2051);   // blocks 0..512 complete, 512 drops
    CHECK_EQ(sel.count(), 2048u);
    CHECK_EQ(sel.blocks.back(), 511u);
    CHECK_EQ(sel.list.back(), 2047u);   // p = 2051 itself is NOT attended: its block lost
    std::vector<float> win = dec;
    win[512] = 2000.0f;                 // p's own block wins: p is attended
    const qr::Selection w = qr::qsa_select(win.data(), 2051);
    CHECK_EQ(w.list.back(), 2051u);
    CHECK_EQ(w.blocks.back(), 512u);
    CHECK(std::find(w.blocks.begin(), w.blocks.end(), 511u) == w.blocks.end());
  }
  std::printf("  select: the scores bitwise; every crafted row = the ruled selection (ties to the lower block, the "
              "cut inside zeros), tails 0..3, count 2048 + tail; torch's topk apart on %u tied blocks only; 2049 / 2050 "
              "identity, 2051 the first cut\n", torch_apart);
}

// --- (e)-(g) --------------------------------------------------------------------------------------------------------
void test_route_combine_gated() {
  const std::vector<uint32_t> L = qr::hex32(fx::kRouteLogits), W = qr::hex32(fx::kRouteW);
  const std::vector<uint16_t> sg = qr::hex16(fx::kRouteSg);
  uint32_t w_differ = 0, torch_apart = 0;
  for (uint32_t r = 0; r < 4; ++r) {
    std::vector<float> lg(qr::kRouterN);
    for (uint32_t e = 0; e < qr::kRouterN; ++e) lg[e] = qr::as_f32(L[size_t(r) * qr::kRouterN + e]);
    const qr::Route rt = qr::route(lg.data());
    for (uint32_t k = 0; k < 10; ++k) {
      CHECK_EQ(rt.ids[k], fx::kRouteRuledIds[r * 10 + k]);
      const uint16_t want = rne(qr::as_f32(W[size_t(r) * 10 + k]));
      CHECK(ulps16(rne(rt.w[k]), want) <= 1);
      w_differ += rne(rt.w[k]) != want;
      torch_apart += rt.ids[k] != fx::kRouteTorchIds[r * 10 + k];
    }
    CHECK(rne(rt.sg) == sg[r]);
    if (r == 2)   // every expert -1e30 but 5, 55, ..., 455: those ten, the largest first; the 11th p is 0
      for (uint32_t k = 0; k < 10; ++k) CHECK_EQ(rt.ids[k], 455u - 50u * k);
    if (r == 2) CHECK(rt.p11 == 0.0f);
    if (r == 1) CHECK_EQ(rt.ids[9], 99u);   // the tie for the 10th goes to the lower id
  }
  // (f) the combine
  const std::vector<uint16_t> w16 = qr::hex16(fx::kCombineW), yw = qr::hex16(fx::kCombineY);
  float w[10];
  for (uint32_t k = 0; k < 10; ++k) w[k] = f32(w16[k]);
  std::vector<std::vector<uint16_t>> down(11);
  for (uint32_t j = 0; j < 11; ++j) down[j] = qr::fixture_bf16(false, 51 + j, qr::kHidden, 6, 0.125f, 4.0f);
  for (uint32_t n = 0; n < qr::kHidden; ++n) {
    float d[11];
    for (uint32_t j = 0; j < 11; ++j) d[j] = f32(down[j][n]);
    CHECK(qr::combine(d, w, 0.40625f) == yw[n]);
  }
  // (g) the gated norm
  const std::vector<uint16_t> x = qr::fixture_bf16(true, 60, 48 * 128, 7, 0.25f), z = qr::fixture_bf16(false, 61, 48 * 128, 7, 0.125f, 6.0f),
                              gw = qr::fixture_bf16(false, 62, 128, 7, 0.125f, 1.5f), want = qr::hex16(fx::kGated);
  std::vector<float> o(48 * 128), qkvz(qr::kQkvzN, 0.0f);
  for (uint32_t i = 0; i < 48 * 128; ++i) {
    o[i] = f32(x[i]);
    qkvz[qr::kZOff + i] = f32(z[i]);
  }
  std::vector<uint16_t> out(48 * 128);
  qr::gated_head_sig(qkvz.data(), o.data(), gw.data(), out.data());
  CHECK(out == want);
  std::printf("  route: ids = the ruled order on 4 rows (torch's topk order apart on %u slots, exact ties of p), %u / 40 "
              "weights one ulp apart, the shared gate bitwise; the combine and the sigmoid gated norm bitwise\n",
              torch_apart, w_differ);
}

}  // namespace

int main() {
  test_hc();
  test_fused_order();
  test_ple();
  test_indexer();
  test_select();
  test_route_combine_gated();
  std::puts("qwen4exp_ref_test OK");
  return 0;
}
