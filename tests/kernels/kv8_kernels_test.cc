// kv8_kernels_test - spec 12b: src/kernels/kv8.cl on the card against tests/kernels/kv8_ref.h.
// Written blind on the Mac (2026-10-05); the box runs it first (box-validation-queue).
//
//   W   the writers, BITWISE: attn_prep_kv8 at M = 1..4 (QKV_S = 2; n_active < M too), the
//       MTP head's S1 build (B70_MTP), and the prefill build (PF = 1) at C = 37 - attn_q
//       (rotated, fp32 / bf16), attn_gate, the int8 K and V rows and their fp16 scales;
//       rows outside [pos, pos + rows) keep their canary.
//   D   decode attention v2 over int8 (attn_decode_v2_kv8 + attn_reduce_v2_kv8): M = 1 at
//       pos in {0, 1, 63, 64, 2047, 2048, 2049, 4095} against the fp64 reference (dequantised
//       K / V, un-rotated, gated) - per q head cosine >= 0.99999, finite; two runs bitwise;
//       M = 2..4 rows BITWISE the M = 1 step at pos + m (spec 8's M2 with the int8 cache).
//   F   prefill over int8 (pf_flash_attn_kv8 then pf_attn_gate_kv8): sampled rows x 24 heads
//       against the same reference - pf_o (still rotated) cosine >= 0.99999 as
//       pf_flash_attn_test's bar, the gated rows >= 0.9999 (PROVISIONAL: the flash rounds
//       P x scale to bf16 where the bf16 kernel rounds P); finite, padding rows finite.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "kv8_ref.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "prefill/pf_harness.h"
#include "runtime/buffers.h"
#include "runtime/control.h"

namespace {

using kv8_ref::kHD;
using kv8_ref::kKvH;
using kv8_ref::kQH;
constexpr uint32_t kQkvN = attn_ref::kQkvN, kRow = kQH * kHD, kKvRow = kKvH * kHD;
constexpr uint32_t kTgt = runtime::DecodeScratch::kAttnV2Blocks;
constexpr int8_t kCanary = 0x5A;
constexpr uint16_t kCanaryScale = 0xBEEF;

bool have(const std::string& variant) { return std::ifstream(kernels::path(variant)).good(); }

template <class T>
void download_all(pf_harness::Dev& d, std::vector<T>& v, const l0::Mem& m) {
  v.resize(m.size() / sizeof(T));
  pf_harness::download(d.imm, v, m);
}

// The inputs every writer case draws from: a qkv rectangle, the FA small block, a RoPE
// table. K's channel 70 of every kv head is an outlier (x 30), as the real K's are.
struct PrepInputs {
  std::vector<float> partials, fa_small, rope;
  PrepInputs(uint32_t S, uint32_t rows, uint32_t max_len, uint32_t seed) {
    partials = pf_harness::random_f32(size_t(S) * rows * kQkvN, seed, 0.0f, 0.7f);
    for (float& v : partials)
      if (v == 0.0f) v = 1e-3f;   // no -0.0f: the +0.0f slice padding stays exact
    for (uint32_t s = 0; s < S; ++s)
      for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t j = 0; j < kKvH; ++j)
          partials[(size_t(s) * rows + m) * kQkvN + attn_ref::kKOff + j * kHD + 70] *= 30.0f;
    fa_small = pf_harness::random_f32(512, seed + 1, 1.0f, 0.3f);
    rope.resize(size_t(max_len) * 64);
    for (uint32_t p = 0; p < max_len; ++p)
      for (uint32_t i = 0; i < 32; ++i) {
        const double th = p * std::pow(1e7, -double(i) / 32.0);
        rope[size_t(p) * 64 + i] = float(std::cos(th));
        rope[size_t(p) * 64 + 32 + i] = float(std::sin(th));
      }
  }
};

// One writer launch: `variant` / `entry`, grid (28, rows), the int8 cache of `max_len` rows
// pre-filled with the canary. `S` slices of `rows` rows; `pf` = the bf16-q prefill build.
struct WriterOut {
  std::vector<float> q, gate;
  std::vector<uint16_t> q16, ks, vs;
  std::vector<int8_t> k, v;
};
WriterOut run_writer(pf_harness::Dev& d, const std::string& variant, uint32_t S, uint32_t M,
                     uint32_t n_act, uint32_t pos, uint32_t max_len, bool pf,
                     const PrepInputs& in) {
  l0::Module mod(d.ctx, kernels::path(variant));
  l0::Kernel k = mod.kernel("attn_prep_kv8");
  k.group_size(256);
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  auto* c = ctrl.as<runtime::Control>();
  *c = runtime::Control{};
  c->pos = pos;
  c->n_active = n_act;
  l0::Mem dp = pf_harness::upload(d.ctx, d.imm, in.partials);
  l0::Mem dfa = pf_harness::upload(d.ctx, d.imm, in.fa_small);
  l0::Mem drope = pf_harness::upload(d.ctx, d.imm, in.rope);
  l0::Mem dq(d.ctx, l0::MemKind::Device, size_t(M) * kRow * 4);
  l0::Mem dg(d.ctx, l0::MemKind::Device, size_t(M) * kRow * 4);
  l0::Mem dk(d.ctx, l0::MemKind::Device, size_t(max_len) * kKvRow);
  l0::Mem dv(d.ctx, l0::MemKind::Device, size_t(max_len) * kKvRow);
  l0::Mem dks(d.ctx, l0::MemKind::Device, size_t(max_len) * kKvH * 2);
  l0::Mem dvs(d.ctx, l0::MemKind::Device, size_t(max_len) * kKvH * 2);
  const uint32_t canary4 = 0x5A5A5A5Au, scale4 = 0xBEEFBEEFu;
  d.imm.fill(dk.ptr(), canary4, dk.size());
  d.imm.fill(dv.ptr(), canary4, dv.size());
  d.imm.fill(dks.ptr(), scale4, dks.size());
  d.imm.fill(dvs.ptr(), scale4, dvs.size());
  d.imm.fill(dq.ptr(), 0u, dq.size());
  d.imm.fill(dg.ptr(), 0u, dg.size());
  k.arg_ptr(0, ctrl.ptr());
  k.arg_ptr(1, dp.ptr());
  k.arg_ptr(2, dfa.ptr());
  k.arg_ptr(3, drope.ptr());
  k.arg_ptr(4, dq.ptr());
  k.arg_ptr(5, pf ? nullptr : dg.ptr());
  k.arg_ptr(6, dk.ptr());
  k.arg_ptr(7, dv.ptr());
  k.arg_ptr(8, dks.ptr());
  k.arg_ptr(9, dvs.ptr());
  d.run(k, kQH + kKvH, M);
  (void)S;
  WriterOut o;
  if (pf) {
    o.q16.resize(size_t(M) * kRow);
    pf_harness::download(d.imm, o.q16, dq);
  } else {
    download_all(d, o.q, dq);
    download_all(d, o.gate, dg);
  }
  download_all(d, o.k, dk);
  download_all(d, o.v, dv);
  download_all(d, o.ks, dks);
  download_all(d, o.vs, dvs);
  return o;
}

// Compare a writer's output with kv8_ref::prep; returns mismatches (0 = bitwise).
size_t check_writer(const WriterOut& o, uint32_t M, uint32_t n_act, uint32_t pos, uint32_t max_len,
                    bool pf, const PrepInputs& in, uint32_t S_ref_from, const char* what) {
  // The reference reads [2][M][14336]; a one-slice input is slice 0 with slice 1 = +0.0f.
  std::vector<float> p2(size_t(2) * M * kQkvN, 0.0f);
  if (S_ref_from == 2)
    p2 = in.partials;
  else
    std::memcpy(p2.data(), in.partials.data(), size_t(M) * kQkvN * 4);
  kv8_ref::Cache c(max_len);
  std::vector<float> q(size_t(M) * kRow), g(size_t(M) * kRow);
  std::vector<uint16_t> q16(size_t(M) * kRow);
  kv8_ref::prep(pos, n_act, M, p2.data(), in.fa_small.data(), in.rope.data(), max_len, q.data(),
                q16.data(), g.data(), c);
  size_t bad = 0, untouched_bad = 0;
  for (uint32_t m = 0; m < n_act; ++m)
    for (uint32_t i = 0; i < kRow; ++i) {
      const size_t e = size_t(m) * kRow + i;
      if (pf) {
        bad += o.q16[e] != q16[e];
      } else {
        bad += common::kv8::f32_bits(o.q[e]) != common::kv8::f32_bits(q[e]);
        bad += common::kv8::f32_bits(o.gate[e]) != common::kv8::f32_bits(g[e]);
      }
    }
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t j = 0; j < kKvH; ++j) {
      const size_t row = size_t(p) * kKvH + j;
      const bool written = p >= pos && p < pos + n_act;
      for (uint32_t i = 0; i < kHD; ++i) {
        if (written) {
          bad += o.k[row * kHD + i] != c.k[row * kHD + i];
          bad += o.v[row * kHD + i] != c.v[row * kHD + i];
        } else {
          untouched_bad += o.k[row * kHD + i] != kCanary;
          untouched_bad += o.v[row * kHD + i] != kCanary;
        }
      }
      if (written) {
        bad += o.ks[row] != c.ks[row];
        bad += o.vs[row] != c.vs[row];
      } else {
        untouched_bad += o.ks[row] != kCanaryScale;
        untouched_bad += o.vs[row] != kCanaryScale;
      }
    }
  std::printf("W %-28s M %u n_active %u pos %u: %zu mismatches against kv8_ref, %zu canary"
              " bytes overwritten -- %s\n",
              what, M, n_act, pos, bad, untouched_bad, bad + untouched_bad ? "FAIL" : "PASS");
  return bad + untouched_bad;
}

bool part_w(pf_harness::Dev& d) {
  size_t bad = 0;
  constexpr uint32_t kL = 128;
  for (uint32_t M = 1; M <= 4; ++M) {
    const PrepInputs in(2, M, kL, 100 + M);
    for (uint32_t n_act : {M, M > 1 ? M - 1 : M}) {
      const uint32_t pos = 37 + M;
      const WriterOut o = run_writer(d, kernels::attn_prep_kv8_variant(M), 2, M, n_act, pos, kL,
                                     false, in);
      bad += check_writer(o, M, n_act, pos, kL, false, in, 2, kernels::attn_prep_kv8_variant(M).c_str());
    }
    const std::string s1 = kernels::attn_prep_s1_kv8_variant(M);
    if (have(s1)) {   // B70_MTP: the head's bf16 qkv, one slice
      const PrepInputs in1(1, M, kL, 200 + M);
      const WriterOut o = run_writer(d, s1, 1, M, M, 5, kL, false, in1);
      bad += check_writer(o, M, M, 5, kL, false, in1, 1, s1.c_str());
    } else {
      std::printf("W %s: not built (B70_MTP off), skipped\n", s1.c_str());
    }
  }
  // The prefill build: C rows of one slice, no gate, bf16 q, at pos 200.
  {
    constexpr uint32_t C = 37, pos = 200, L = 256;
    const PrepInputs in(1, C, L, 300);
    const WriterOut o = run_writer(d, kernels::pf_attn_prep_kv8_variant(), 1, C, C, pos, L, true, in);
    bad += check_writer(o, C, C, pos, L, true, in, 1, "pf_attn_prep_kv8");
  }
  return bad == 0;
}

// A random int8 cache of L rows (encoded on the host from bf16 rows with K outliers).
kv8_ref::Cache random_cache(uint32_t L, uint32_t seed) {
  kv8_ref::Cache c(L);
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  uint16_t kb[kHD], vb[kHD];
  for (size_t row = 0; row < size_t(L) * kKvH; ++row) {
    for (uint32_t i = 0; i < kHD; ++i) {
      kb[i] = common::f32_to_bf16(nd(rng) * (i == 70 || i == 133 ? 20.0f : 1.0f));
      vb[i] = common::f32_to_bf16(nd(rng));
    }
    c.put(row, kb, vb);
  }
  return c;
}

struct DevCache {
  l0::Mem k, v, ks, vs;
  DevCache(pf_harness::Dev& d, const kv8_ref::Cache& c)
      : k(pf_harness::upload(d.ctx, d.imm, c.k)), v(pf_harness::upload(d.ctx, d.imm, c.v)),
        ks(pf_harness::upload(d.ctx, d.imm, c.ks)), vs(pf_harness::upload(d.ctx, d.imm, c.vs)) {}
};

// attn_decode_v2_kv8 + attn_reduce_v2_kv8 at M rows from pos; q / gate rows [row0, row0 + M).
std::vector<uint16_t> run_decode(pf_harness::Dev& d, l0::Module& mod, const DevCache& dc,
                                 uint32_t M, uint32_t pos, const std::vector<float>& q,
                                 const std::vector<float>& g, uint32_t row0) {
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  auto* c = ctrl.as<runtime::Control>();
  *c = runtime::Control{};
  c->pos = pos;
  c->n_active = M;
  l0::Mem aq(d.ctx, l0::MemKind::Device, size_t(M) * kRow * 4);
  l0::Mem ag(d.ctx, l0::MemKind::Device, size_t(M) * kRow * 4);
  d.imm.copy(aq.ptr(), q.data() + size_t(row0) * kRow, size_t(M) * kRow * 4);
  d.imm.copy(ag.ptr(), g.data() + size_t(row0) * kRow, size_t(M) * kRow * 4);
  l0::Mem part(d.ctx, l0::MemKind::Device, size_t(kQH) * kTgt * M * 258 * 4);
  l0::Mem out(d.ctx, l0::MemKind::Device, size_t(M) * kRow * 2);
  d.imm.fill(part.ptr(), 0x7FC00000u, part.size());   // NaN canary: an unwritten partial shows
  d.imm.fill(out.ptr(), 0u, out.size());
  l0::Kernel dec = mod.kernel("attn_decode_v2_kv8");
  l0::Kernel red = mod.kernel("attn_reduce_v2_kv8");
  dec.group_size(256);
  red.group_size(256);
  dec.arg_ptr(0, ctrl.ptr());
  dec.arg_ptr(1, aq.ptr());
  dec.arg_ptr(2, dc.k.ptr());
  dec.arg_ptr(3, dc.ks.ptr());
  dec.arg_ptr(4, dc.v.ptr());
  dec.arg_ptr(5, dc.vs.ptr());
  dec.arg_ptr(6, part.ptr());
  red.arg_ptr(0, ctrl.ptr());
  red.arg_ptr(1, part.ptr());
  red.arg_ptr(2, ag.ptr());
  red.arg_ptr(3, out.ptr());
  d.run2(dec, kKvH, kTgt, red, kQH, M);
  std::vector<uint16_t> o(size_t(M) * kRow);
  pf_harness::download(d.imm, o, out);
  return o;
}

bool part_d(pf_harness::Dev& d) {
  constexpr uint32_t L = 4096;
  const kv8_ref::Cache c = random_cache(L, 7);
  DevCache dc(d, c);
  std::vector<std::unique_ptr<l0::Module>> mods;
  for (uint32_t M = 1; M <= 4; ++M)
    mods.push_back(
        std::make_unique<l0::Module>(d.ctx, kernels::path(kernels::attn_v2_kv8_variant(M, kTgt))));
  // q (already rotated: any fp32 row is a valid rotated q) at a scale that makes the
  // softmax peaked as well as flat; the gate bf16-valued, as attn_prep writes it.
  std::vector<float> q = pf_harness::random_f32(size_t(4) * kRow, 11, 0.0f, 1.0f);
  for (size_t i = 0; i < q.size(); ++i) q[i] *= (i / kHD) % 3 == 0 ? 3.0f : 1.0f;
  std::vector<float> g = pf_harness::random_f32(size_t(4) * kRow, 12, 0.0f, 1.5f);
  for (float& v : g) v = common::bf16_to_f32(common::f32_to_bf16(v));
  bool ok = true;
  for (uint32_t pos : {0u, 1u, 63u, 64u, 2047u, 2048u, 2049u, 4092u, 4095u}) {
    const std::vector<uint16_t> a = run_decode(d, *mods[0], dc, 1, pos, q, g, 0);
    const std::vector<uint16_t> b = run_decode(d, *mods[0], dc, 1, pos, q, g, 0);
    const bool repeat = a == b;
    double worst = 2.0, maxabs = 0;
    bool finite = true;
    for (uint32_t h = 0; h < kQH; ++h) {
      double o[kHD];
      kv8_ref::attend(&q[size_t(h) * kHD], c, h / kv8_ref::kGqa, pos + 1, o);
      uint16_t ref[kHD];
      for (uint32_t i = 0; i < kHD; ++i) {
        ref[i] = kv8_ref::gated(o[i], g[size_t(h) * kHD + i]);
        const double x = common::bf16_to_f32(a[size_t(h) * kHD + i]);
        if (!std::isfinite(x)) finite = false;
        maxabs = std::max(maxabs, std::fabs(x - common::bf16_to_f32(ref[i])));
      }
      worst = std::min(worst, kv8_ref::cos_bf16(&a[size_t(h) * kHD], ref));
    }
    const bool pass = worst >= 0.99999 && finite && repeat;
    std::printf("D M=1 pos %4u: worst head cosine %.9f, max abs %.3e, finite %d, repeat bitwise %d"
                " -- %s\n", pos, worst, maxabs, finite, repeat, pass ? "PASS" : "FAIL");
    ok &= pass;
    // Verify rows: M = 2..4 from pos, row m bitwise the M = 1 step at pos + m.
    for (uint32_t M = 2; M <= 4; ++M) {
      if (pos + M > L) continue;
      const std::vector<uint16_t> r = run_decode(d, *mods[M - 1], dc, M, pos, q, g, 0);
      bool same = true;
      for (uint32_t m = 0; m < M; ++m) {
        const std::vector<uint16_t> one = run_decode(d, *mods[0], dc, 1, pos + m, q, g, m);
        same &= std::equal(one.begin(), one.end(), r.begin() + size_t(m) * kRow);
      }
      std::printf("D M=%u pos %4u: rows bitwise the M = 1 steps: %d -- %s\n", M, pos, same,
                  same ? "PASS" : "FAIL");
      ok &= same;
    }
  }
  return ok;
}

bool flash_case(pf_harness::Dev& d, l0::Kernel& fk, l0::Kernel& gk, uint32_t pos, uint32_t C) {
  const uint32_t depth = pos + C, rows = (C + 255u) & ~255u, L = (depth + 255u) & ~255u;
  const kv8_ref::Cache c = random_cache(L, 1000 + pos + C);
  DevCache dc(d, c);
  // bf16 rotated q (what pf_attn_prep_kv8 writes), the qkv rectangle's gate columns.
  std::mt19937 rng(pos * 3 + C);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> q(size_t(C) * kRow);
  for (auto& e : q) e = common::f32_to_bf16(nd(rng));
  std::vector<float> part = pf_harness::random_f32(size_t(C) * kQkvN, pos + 17, 0.0f, 1.5f);
  l0::Mem dq = pf_harness::upload(d.ctx, d.imm, q);
  l0::Mem dpart = pf_harness::upload(d.ctx, d.imm, part);
  const size_t o_elems = size_t(kQH) * rows * kHD;
  l0::Mem dO(d.ctx, l0::MemKind::Device, o_elems * 4);
  l0::Mem dout(d.ctx, l0::MemKind::Device, size_t(C) * kRow * 2);
  d.imm.fill(dO.ptr(), 0u, dO.size());
  fk.arg_ptr(0, dq.ptr());
  fk.arg_ptr(1, dc.k.ptr());
  fk.arg_ptr(2, dc.ks.ptr());
  fk.arg_ptr(3, dc.v.ptr());
  fk.arg_ptr(4, dc.vs.ptr());
  fk.arg_ptr(5, dO.ptr());
  fk.arg(6, pos);
  fk.arg(7, C);
  fk.arg(8, rows);
  const uint32_t stride_h = rows * kHD;
  gk.arg_ptr(0, dO.ptr());
  gk.arg_ptr(1, dpart.ptr());
  gk.arg_ptr(2, dout.ptr());
  gk.arg(3, stride_h);
  d.run2(fk, (C + 7u) / 8u, kKvH, gk, kQH, C);
  std::vector<float> O(o_elems);
  std::vector<uint16_t> out(size_t(C) * kRow);
  pf_harness::download(d.imm, O, dO);
  pf_harness::download(d.imm, out, dout);

  std::vector<uint32_t> srows = {0, 1, 7, 8, 63, 64, C / 2, C - 2, C - 1};
  for (uint32_t& m : srows) m = std::min(m, C - 1);
  if (C < 2) srows.assign(1, 0);
  std::sort(srows.begin(), srows.end());
  srows.erase(std::unique(srows.begin(), srows.end()), srows.end());
  double worst_o = 2.0, worst_g = 2.0;
  size_t nonfinite = 0, pad_nonfinite = 0;
  for (uint32_t m : srows)
    for (uint32_t h = 0; h < kQH; ++h) {
      float qf[kHD];
      for (uint32_t i = 0; i < kHD; ++i) qf[i] = common::bf16_to_f32(q[size_t(m) * kRow + h * kHD + i]);
      double o[kHD], orot[kHD];
      kv8_ref::attend(qf, c, h / kv8_ref::kGqa, pos + m + 1, o, orot);
      const float* got = &O[(size_t(h) * rows + m) * kHD];
      double xy = 0, xx = 0, yy = 0;
      uint16_t ref[kHD];
      for (uint32_t i = 0; i < kHD; ++i) {
        if (!std::isfinite(got[i])) ++nonfinite;
        xy += got[i] * orot[i];
        xx += double(got[i]) * got[i];
        yy += orot[i] * orot[i];
        const float gate = attn_ref::f32(attn_ref::rne(part[size_t(m) * kQkvN + h * 512 + 256 + i]));
        ref[i] = kv8_ref::gated(o[i], gate);
      }
      worst_o = std::min(worst_o, xx > 0 && yy > 0 ? xy / std::sqrt(xx * yy) : -2.0);
      worst_g = std::min(worst_g, kv8_ref::cos_bf16(&out[size_t(m) * kRow + h * kHD], ref));
    }
  for (uint32_t h = 0; h < kQH; ++h)
    for (uint32_t m = C; m < rows; ++m)
      for (uint32_t i = 0; i < kHD; ++i)
        if (!std::isfinite(O[(size_t(h) * rows + m) * kHD + i])) ++pad_nonfinite;
  const bool ok = worst_o >= 0.99999 && worst_g >= 0.9999 && nonfinite == 0 && pad_nonfinite == 0;
  std::printf("F pos %5u C %4u: pf_o (rotated) worst cos %.9f, gated rows worst cos %.9f,"
              " non-finite %zu, padding non-finite %zu -- %s\n",
              pos, C, worst_o, worst_g, nonfinite, pad_nonfinite, ok ? "PASS" : "FAIL");
  return ok;
}

bool part_f(pf_harness::Dev& d) {
  l0::Module fm(d.ctx, kernels::path(kernels::pf_flash_attn_kv8_variant()));
  l0::Module gm(d.ctx, kernels::path(kernels::pf_attn_gate_kv8_variant()));
  l0::Kernel fk = fm.kernel("pf_flash_attn_kv8");
  l0::Kernel gk = gm.kernel("pf_attn_gate_kv8");
  fk.group_size(16 * 6);   // 6 sub-groups of 16, as pf_flash_attn
  gk.group_size(256);
  bool ok = true;
  ok &= flash_case(d, fk, gk, 0, 64);
  ok &= flash_case(d, fk, gk, 777, 300);
  ok &= flash_case(d, fk, gk, 0, 1);
  ok &= flash_case(d, fk, gk, 4096, 1);
  ok &= flash_case(d, fk, gk, 0, 9);
  ok &= flash_case(d, fk, gk, 2000, 2048);
  return ok;
}

}  // namespace

int main() {
  pf_harness::Dev d;
  bool ok = true;
  ok &= part_w(d);
  ok &= part_d(d);
  ok &= part_f(d);
  CHECK(ok);
  std::printf("kv8_kernels_test: PASS\n");
  return 0;
}
