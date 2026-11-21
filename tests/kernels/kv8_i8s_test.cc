// kv8_i8s_test - PERF-114 (the int8 prompt pass): kv8.cl's KV8_I8S binaries, int8 attention scores on the int8 KV cache, no
// checkpoint.
//
//   W   the writer pf_attn_prep_kv8_I8S, BITWISE: q8 and its fp32 scale against common::kv8::quantise_q over
//       kv8_ref::prep's bf16 rotated q; the int8 K / V rows and fp16 scales against kv8_ref (what the bf16-q
//       writer stores, kv8_kernels_test), rows outside the chunk keep their canary.
//   S   the pair pf_flash_attn_kv8_raw_pair_I8S (with its kv8_widen) BITWISE pf_flash_attn_kv8_raw_pair, every pf_o
//       word, on q scales that are powers of two: q8 * 2^-e is exact in bf16 and every partial dot is an exact
//       integer times 2^-e, so the bf16 DPAS sum and (int32 dot * 2^-e) are one value - only the score's product
//       order (dot * s_q, then * s_k, then the 1/16 log2 e) and the q scale's row and head can differ. The I8S
//       widen: K's int8 bytes and V's bf16 words bitwise. Shapes as kv8_flash_bitwise_test's (one 2048-row launch at
//       most, the engine's).
//   P   the paged pair (_PG): the cache in pages behind a reversed page table, BITWISE the contiguous I8S pair.
//   T   the tensor-parallel card's 12 / 2 pair BITWISE its raw flash on power-of-two q scales (part_tp).
//   D   real q scales (common::kv8::quantise_q of random rotated-like q) against the CPU twin: the scores exact
//       (common::kv8::score_i8), softmax and P V in fp64 over the cache's dequantised V - pf_o (rotated) cosine
//       >= 0.99999 per sampled row and head, kv8_kernels_test's F bar (the kernel rounds p s_v to bf16 as the
//       bf16 pair does: its only rounding past the scores).
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
#include "common/kv8.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "kv8_ref.h"
#include "prefill/pf_harness.h"
#include "runtime/control.h"

namespace {

namespace kv8 = common::kv8;
using kv8_ref::kHD;
using kv8_ref::kKvH;
using kv8_ref::kQH;
constexpr uint32_t kQkvN = attn_ref::kQkvN, kRow = kQH * kHD, kKvRow = kKvH * kHD, kPage = kernels::kKvPage;
constexpr float kScoreScale = (1.0f / 16.0f) * 1.44269504088896340736f;   // kv8.cl SCORE_SCALE (EXP2 = 1)

bool have(const std::string& variant) { return std::ifstream(kernels::path(variant)).good(); }

void run3(pf_harness::Dev& d, l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz) {
  auto list = l0::CmdList::regular(d.ctx);
  list.launch(k, gx, gy, gz);
  list.close();
  d.q.execute(list, &d.fence);
  d.fence.wait();
}

// --- W ---------------------------------------------------------------------------------------------------------
bool part_w(pf_harness::Dev& d) {
  constexpr uint32_t C = 37, pos = 200, L = 256;
  // kv8_kernels_test's inputs: a qkv rectangle (K channel 70 an outlier), the FA small block, a RoPE table.
  std::vector<float> partials = pf_harness::random_f32(size_t(C) * kQkvN, 300, 0.0f, 0.7f);
  for (float& v : partials)
    if (v == 0.0f) v = 1e-3f;
  for (uint32_t m = 0; m < C; ++m)
    for (uint32_t j = 0; j < kKvH; ++j) partials[size_t(m) * kQkvN + attn_ref::kKOff + j * kHD + 70] *= 30.0f;
  const std::vector<float> fa_small = pf_harness::random_f32(512, 301, 1.0f, 0.3f);
  std::vector<float> rope(size_t(L) * 64);
  for (uint32_t p = 0; p < L; ++p)
    for (uint32_t i = 0; i < 32; ++i) {
      const double th = p * std::pow(1e7, -double(i) / 32.0);
      rope[size_t(p) * 64 + i] = float(std::cos(th));
      rope[size_t(p) * 64 + 32 + i] = float(std::sin(th));
    }
  // The reference: kv8_ref::prep (slice 1 = +0.0f), then quantise_q of each bf16 row.
  std::vector<float> p2(size_t(2) * C * kQkvN, 0.0f);
  std::memcpy(p2.data(), partials.data(), partials.size() * 4);
  kv8_ref::Cache c(L);
  std::vector<uint16_t> q16(size_t(C) * kRow);
  std::vector<float> gate(size_t(C) * kRow);
  kv8_ref::prep(pos, C, C, p2.data(), fa_small.data(), rope.data(), L, nullptr, q16.data(), gate.data(), c);
  std::vector<int8_t> want_q8(size_t(C) * kRow);
  std::vector<float> want_qs(size_t(C) * kQH);
  for (size_t r = 0; r < size_t(C) * kQH; ++r) {
    float y[kHD];
    for (uint32_t i = 0; i < kHD; ++i) y[i] = common::bf16_to_f32(q16[r * kHD + i]);
    want_qs[r] = kv8::quantise_q(y, &want_q8[r * kHD]);
  }

  l0::Module mod(d.ctx, kernels::path(kernels::pf_attn_prep_kv8_i8s_variant()));
  l0::Kernel k = mod.kernel("attn_prep_kv8");
  k.group_size(256);
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  *ctrl.as<runtime::Control>() = runtime::Control{};
  ctrl.as<runtime::Control>()->pos = pos;
  ctrl.as<runtime::Control>()->n_active = C;
  l0::Mem dp = pf_harness::upload(d.ctx, d.imm, partials);
  l0::Mem dfa = pf_harness::upload(d.ctx, d.imm, fa_small);
  l0::Mem drope = pf_harness::upload(d.ctx, d.imm, rope);
  l0::Mem dq(d.ctx, l0::MemKind::Device, size_t(C) * kRow * 2);   // pf_q's bf16 size; int8 fills the first half
  l0::Mem dqs(d.ctx, l0::MemKind::Device, size_t(C) * kQH * 4);
  l0::Mem dk(d.ctx, l0::MemKind::Device, size_t(L) * kKvRow), dv(d.ctx, l0::MemKind::Device, size_t(L) * kKvRow);
  l0::Mem dks(d.ctx, l0::MemKind::Device, size_t(L) * kKvH * 2), dvs(d.ctx, l0::MemKind::Device, size_t(L) * kKvH * 2);
  d.imm.fill(dk.ptr(), 0x5A5A5A5Au, dk.size());
  d.imm.fill(dv.ptr(), 0x5A5A5A5Au, dv.size());
  d.imm.fill(dks.ptr(), 0xBEEFBEEFu, dks.size());
  d.imm.fill(dvs.ptr(), 0xBEEFBEEFu, dvs.size());
  d.imm.fill(dq.ptr(), 0xA5A5A5A5u, dq.size());
  d.imm.fill(dqs.ptr(), 0xA5A5A5A5u, dqs.size());
  void* ptrs[] = {ctrl.ptr(), dp.ptr(), dfa.ptr(), drope.ptr(), dq.ptr(), nullptr,
                  dk.ptr(),   dv.ptr(), dks.ptr(), dvs.ptr(),   dqs.ptr()};
  for (uint32_t i = 0; i < 11; ++i) k.arg_ptr(i, ptrs[i]);
  d.run(k, kQH + kKvH, C);
  std::vector<int8_t> q8(size_t(C) * kRow), gk(size_t(L) * kKvRow), gv(gk.size());
  std::vector<float> qs(size_t(C) * kQH);
  std::vector<uint16_t> gks(size_t(L) * kKvH), gvs(gks.size());
  d.imm.copy(q8.data(), dq.ptr(), q8.size());
  pf_harness::download(d.imm, qs, dqs);
  pf_harness::download(d.imm, gk, dk);
  pf_harness::download(d.imm, gv, dv);
  pf_harness::download(d.imm, gks, dks);
  pf_harness::download(d.imm, gvs, dvs);
  size_t bad = 0;
  for (size_t i = 0; i < q8.size(); ++i) bad += q8[i] != want_q8[i];
  for (size_t i = 0; i < qs.size(); ++i) bad += kv8::f32_bits(qs[i]) != kv8::f32_bits(want_qs[i]);
  for (uint32_t p = 0; p < L; ++p)
    for (uint32_t j = 0; j < kKvH; ++j) {
      const size_t row = size_t(p) * kKvH + j;
      const bool written = p >= pos && p < pos + C;
      for (uint32_t i = 0; i < kHD; ++i) {
        bad += gk[row * kHD + i] != (written ? c.k[row * kHD + i] : int8_t(0x5A));
        bad += gv[row * kHD + i] != (written ? c.v[row * kHD + i] : int8_t(0x5A));
      }
      bad += gks[row] != (written ? c.ks[row] : uint16_t(0xBEEF));
      bad += gvs[row] != (written ? c.vs[row] : uint16_t(0xBEEF));
    }
  std::printf("W pf_attn_prep_kv8_I8S C %u pos %u: %zu mismatches (q8, q scales, K / V rows and scales, canaries) "
              "-- %s\n", C, pos, bad, bad ? "FAIL" : "PASS");
  return bad == 0;
}

// --- S / P / D: one shape's cache, q and the launches ------------------------------------------------------------
struct Shape {
  uint32_t pos, C;
};

struct Cache {
  std::vector<int8_t> k, v;
  std::vector<uint16_t> ks, vs;
};

Cache random_cache(uint32_t depth, uint32_t seed) {
  std::mt19937 rng(seed);
  Cache c;
  c.k.resize(size_t(depth) * kKvRow);
  c.v.resize(c.k.size());
  for (size_t i = 0; i < c.k.size(); ++i) {
    c.k[i] = int8_t(int(rng() % 255) - 127);
    c.v[i] = int8_t(int(rng() % 255) - 127);
  }
  c.ks.resize(size_t(depth) * kKvH);
  c.vs.resize(c.ks.size());
  for (size_t i = 0; i < c.ks.size(); ++i) {
    c.ks[i] = kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
    c.vs[i] = kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
  }
  return c;
}

struct Kernels {
  l0::Module def, i8s;
  std::unique_ptr<l0::Module> i8s_pg;
  l0::Kernel def_widen, def_pair, widen, pair;
  std::unique_ptr<l0::Kernel> widen_pg, pair_pg;
  explicit Kernels(pf_harness::Dev& d)
      : def(d.ctx, kernels::path(kernels::pf_flash_attn_kv8_raw_pair_variant())),
        i8s(d.ctx, kernels::path(kernels::pf_flash_attn_kv8_raw_pair_i8s_variant())),
        def_widen(def.kernel("kv8_widen")), def_pair(def.kernel("pf_flash_attn_kv8")),
        widen(i8s.kernel("kv8_widen")), pair(i8s.kernel("pf_flash_attn_kv8")) {
    const std::string pg = kernels::kv_paged(kernels::pf_flash_attn_kv8_raw_pair_i8s_variant());
    if (have(pg)) {
      i8s_pg = std::make_unique<l0::Module>(d.ctx, kernels::path(pg));
      widen_pg = std::make_unique<l0::Kernel>(i8s_pg->kernel("kv8_widen"));
      pair_pg = std::make_unique<l0::Kernel>(i8s_pg->kernel("pf_flash_attn_kv8"));
    }
    for (l0::Kernel* w : {&def_widen, &widen, widen_pg.get()})
      if (w) w->group_size(256);
    for (l0::Kernel* p : {&def_pair, &pair, pair_pg.get()})
      if (p) p->group_size(kernels::kPfFlashPairHeads * 32);
  }
};

// The widened scratch (K's region, then V's at `plane` bf16 words) and the pair's pf_o for one launch. `q` is the
// default pair's bf16 q or the I8S int8 q (then `qs` its scales); `pages` null: contiguous.
std::vector<float> attend(pf_harness::Dev& d, l0::Kernel& widen, l0::Kernel& pair, bool i8s, const Shape& s,
                          const l0::Mem& q, const l0::Mem* qs, const l0::Mem& k, const l0::Mem& v, const l0::Mem& ks,
                          const l0::Mem& vs, const l0::Mem* pages, std::vector<uint8_t>* scratch_out = nullptr) {
  const uint32_t depth = s.pos + s.C, rows = (s.C + 255u) & ~255u, count = depth * kKvRow;
  const size_t plane = size_t(depth + 256u) * kKvRow;   // a gap before V, as the engine's longer kv8_len
  l0::Mem raw(d.ctx, l0::MemKind::Device, 2 * plane * 2);
  d.imm.fill(raw.ptr(), 0xA55AA55Au, raw.size());
  uint16_t* kr = raw.as<uint16_t>();
  uint16_t* vr = kr + plane;
  widen.arg_ptr(0, k.ptr());
  widen.arg_ptr(1, v.ptr());
  widen.arg_ptr(2, kr);
  widen.arg_ptr(3, vr);
  widen.arg(4, count);
  if (pages) widen.arg_ptr(5, pages->ptr());
  d.run(widen, (count + 4095u) / 4096u, 1);
  l0::Mem o(d.ctx, l0::MemKind::Device, size_t(kQH) * rows * kHD * 4);
  d.imm.fill(o.ptr(), 0x3F123456u, o.size());
  uint32_t a = 0;
  pair.arg_ptr(a++, q.ptr());
  if (i8s) pair.arg_ptr(a++, qs->ptr());
  pair.arg_ptr(a++, kr);
  pair.arg_ptr(a++, ks.ptr());
  pair.arg_ptr(a++, vr);
  pair.arg_ptr(a++, vs.ptr());
  pair.arg_ptr(a++, o.ptr());
  pair.arg(a++, s.pos);
  pair.arg(a++, s.C);
  pair.arg(a++, rows);
  if (pages) pair.arg_ptr(a++, pages->ptr());
  run3(d, pair, (s.C + kernels::kPfFlashPairRows - 1u) / kernels::kPfFlashPairRows, kKvH,
       kQH / kKvH / kernels::kPfFlashPairHeads);
  std::vector<float> out(o.size() / 4);
  pf_harness::download(d.imm, out, o);
  if (scratch_out) {
    scratch_out->resize(raw.size());
    d.imm.copy(scratch_out->data(), raw.ptr(), raw.size());
  }
  return out;
}

size_t words_differ(const std::vector<float>& a, const std::vector<float>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
  return n;
}

// S and P on one shape.
size_t part_sp(pf_harness::Dev& d, Kernels& kn, const Shape& s) {
  const uint32_t depth = s.pos + s.C;
  const Cache c = random_cache(depth, s.pos + s.C);
  std::mt19937 rng(s.pos * 3 + s.C);
  std::vector<int8_t> q8(size_t(s.C) * kRow);
  std::vector<float> qs(size_t(s.C) * kQH);
  std::vector<uint16_t> qb(q8.size());
  for (uint32_t r = 0; r < s.C; ++r)
    for (uint32_t h = 0; h < kQH; ++h) {
      const size_t rh = size_t(r) * kQH + h;
      qs[rh] = std::ldexp(1.0f, -3 - int((r * 7u + h * 3u) % 9u));   // a power of two per row and head
      for (uint32_t i = 0; i < kHD; ++i) {
        q8[rh * kHD + i] = int8_t(int(rng() % 255) - 127);
        qb[rh * kHD + i] = common::f32_to_bf16(float(q8[rh * kHD + i]) * qs[rh]);   // exact
      }
    }
  auto dk = pf_harness::upload(d.ctx, d.imm, c.k);
  auto dv = pf_harness::upload(d.ctx, d.imm, c.v);
  auto dks = pf_harness::upload(d.ctx, d.imm, c.ks);
  auto dvs = pf_harness::upload(d.ctx, d.imm, c.vs);
  auto dq8 = pf_harness::upload(d.ctx, d.imm, q8);
  auto dqs = pf_harness::upload(d.ctx, d.imm, qs);
  auto dqb = pf_harness::upload(d.ctx, d.imm, qb);
  std::vector<uint8_t> scr_def, scr_i8s;
  const auto ref = attend(d, kn.def_widen, kn.def_pair, false, s, dqb, nullptr, dk, dv, dks, dvs, nullptr, &scr_def);
  const auto got = attend(d, kn.widen, kn.pair, true, s, dq8, &dqs, dk, dv, dks, dvs, nullptr, &scr_i8s);
  // The I8S widen: K's int8 bytes at the scratch's start, V's bf16 words where the default widen put them.
  const size_t plane_bytes = size_t(depth + 256u) * kKvRow * 2;
  size_t widen_bad = std::memcmp(scr_i8s.data(), c.k.data(), c.k.size()) != 0;
  widen_bad += std::memcmp(scr_i8s.data() + plane_bytes, scr_def.data() + plane_bytes, size_t(depth) * kKvRow * 2) != 0;
  const size_t diff = words_differ(got, ref);
  for (float x : got) CHECK(!std::isnan(x));
  std::printf("S pos %6u C %4u: %zu of %zu pf_o words differ from the bf16 pair; widen %s -- %s\n", s.pos, s.C, diff,
              got.size(), widen_bad ? "DIFFERS" : "bitwise", diff + widen_bad ? "FAIL" : "PASS");
  size_t bad = diff + widen_bad;
  if (!kn.pair_pg) return bad;
  // P: the same rows in pages, the table reversed (page n - 1 - p holds the sequence's page p).
  const uint32_t np = (depth + kPage - 1) / kPage;
  std::vector<uint32_t> table(np);
  for (uint32_t p = 0; p < np; ++p) table[p] = np - 1 - p;
  Cache pc;
  pc.k.assign(size_t(np) * kPage * kKvRow, 0);
  pc.v.assign(pc.k.size(), 0);
  pc.ks.assign(size_t(np) * kPage * kKvH, 0);
  pc.vs.assign(pc.ks.size(), 0);
  for (uint32_t p = 0; p < depth; ++p) {
    const size_t row = size_t(table[p / kPage]) * kPage + p % kPage;
    std::memcpy(&pc.k[row * kKvRow], &c.k[size_t(p) * kKvRow], kKvRow);
    std::memcpy(&pc.v[row * kKvRow], &c.v[size_t(p) * kKvRow], kKvRow);
    for (uint32_t j = 0; j < kKvH; ++j) {
      pc.ks[row * kKvH + j] = c.ks[size_t(p) * kKvH + j];
      pc.vs[row * kKvH + j] = c.vs[size_t(p) * kKvH + j];
    }
  }
  auto pk = pf_harness::upload(d.ctx, d.imm, pc.k);
  auto pv = pf_harness::upload(d.ctx, d.imm, pc.v);
  auto pks = pf_harness::upload(d.ctx, d.imm, pc.ks);
  auto pvs = pf_harness::upload(d.ctx, d.imm, pc.vs);
  auto dpages = pf_harness::upload(d.ctx, d.imm, table);
  const auto paged = attend(d, *kn.widen_pg, *kn.pair_pg, true, s, dq8, &dqs, pk, pv, pks, pvs, &dpages);
  const size_t pdiff = words_differ(paged, got);
  std::printf("P pos %6u C %4u: %u pages reversed, %zu pf_o words differ from the contiguous pair -- %s\n", s.pos, s.C,
              np, pdiff, pdiff ? "FAIL" : "PASS");
  return bad + pdiff;
}

// D on one shape: rotated-like q (normal, bf16), quantised as the writer does, against the CPU twin.
size_t part_d(pf_harness::Dev& d, Kernels& kn, const Shape& s) {
  const uint32_t depth = s.pos + s.C, rows = (s.C + 255u) & ~255u;
  const Cache c = random_cache(depth, 7 * s.pos + s.C);
  std::mt19937 rng(s.pos + 11 * s.C);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<int8_t> q8(size_t(s.C) * kRow);
  std::vector<float> qs(size_t(s.C) * kQH);
  for (size_t rh = 0; rh < qs.size(); ++rh) {
    float y[kHD];
    for (uint32_t i = 0; i < kHD; ++i) y[i] = common::bf16_to_f32(common::f32_to_bf16(2.5f * nd(rng)));
    qs[rh] = kv8::quantise_q(y, &q8[rh * kHD]);
  }
  auto dk = pf_harness::upload(d.ctx, d.imm, c.k);
  auto dv = pf_harness::upload(d.ctx, d.imm, c.v);
  auto dks = pf_harness::upload(d.ctx, d.imm, c.ks);
  auto dvs = pf_harness::upload(d.ctx, d.imm, c.vs);
  auto dq8 = pf_harness::upload(d.ctx, d.imm, q8);
  auto dqs = pf_harness::upload(d.ctx, d.imm, qs);
  const auto got = attend(d, kn.widen, kn.pair, true, s, dq8, &dqs, dk, dv, dks, dvs, nullptr);
  double worst = 1.0;
  size_t bad = 0;
  for (uint32_t r : {0u, s.C / 2, s.C - 1})
    for (uint32_t h = 0; h < kQH; ++h) {
      const uint32_t j = h / (kQH / kKvH), n = s.pos + r + 1;   // causal: keys 0 .. pos + r
      const size_t rh = size_t(r) * kQH + h;
      std::vector<double> sc(n);
      double m = -INFINITY;
      for (uint32_t t = 0; t < n; ++t) {
        const size_t kr = size_t(t) * kKvH + j;
        sc[t] = kv8::score_i8(&q8[rh * kHD], qs[rh], &c.k[kr * kHD], c.ks[kr], kScoreScale);
        m = std::max(m, sc[t]);
      }
      std::vector<double> o(kHD, 0.0);
      double l = 0;
      for (uint32_t t = 0; t < n; ++t) {
        const double p = std::exp2(sc[t] - m);
        l += p;
        const size_t vr = size_t(t) * kKvH + j;
        for (uint32_t i = 0; i < kHD; ++i) o[i] += p * double(kv8::dequant(c.v[vr * kHD + i], c.vs[vr]));
      }
      double dot = 0, na = 0, nb = 0;
      for (uint32_t i = 0; i < kHD; ++i) {
        const double a = got[(size_t(h) * rows + r) * kHD + i], b = o[i] / l;
        dot += a * b;
        na += a * a;
        nb += b * b;
      }
      const double cosv = dot / std::sqrt(na * nb);
      worst = std::min(worst, cosv);
      bad += !(cosv >= 0.99999);
    }
  std::printf("D pos %6u C %4u: worst pf_o cosine against the CPU twin %.7f (bar 0.99999) -- %s\n", s.pos, s.C, worst,
              bad ? "FAIL" : "PASS");
  return bad;
}

// T: the tensor-parallel card's 12 / 2 heads - the I8S pair (pf_flash_attn_kv8_raw_pair_I8S_Q12KV2) BITWISE the card's
// raw flash (pf_flash_attn_kv8_raw_Q12KV2, the 8-row form the pair is bitwise) on power-of-two q scales, as S.
size_t part_tp(pf_harness::Dev& d, const Shape& s) {
  constexpr uint32_t qh = 12, kvh = 2, kvrow = kvh * kHD;
  const uint32_t depth = s.pos + s.C, rows = (s.C + 255u) & ~255u, count = depth * kvrow;
  l0::Module rm(d.ctx, kernels::path(kernels::pf_flash_attn_kv8_raw_variant(qh, kvh)));
  l0::Module im(d.ctx, kernels::path(kernels::pf_flash_attn_kv8_raw_pair_i8s_variant(qh, kvh)));
  std::mt19937 rng(s.pos * 5 + s.C);
  std::vector<int8_t> k(size_t(depth) * kvrow), v(k.size()), q8(size_t(s.C) * qh * kHD);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = int8_t(int(rng() % 255) - 127);
    v[i] = int8_t(int(rng() % 255) - 127);
  }
  std::vector<uint16_t> ks(size_t(depth) * kvh), vs(ks.size()), qb(q8.size());
  for (size_t i = 0; i < ks.size(); ++i) {
    ks[i] = kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
    vs[i] = kv8::f32_to_f16_rne(float(1 + rng() % 100) / 4096.0f);
  }
  std::vector<float> qs(size_t(s.C) * qh);
  for (size_t rh = 0; rh < qs.size(); ++rh) {
    qs[rh] = std::ldexp(1.0f, -3 - int((rh * 5u) % 9u));
    for (uint32_t i = 0; i < kHD; ++i) {
      q8[rh * kHD + i] = int8_t(int(rng() % 255) - 127);
      qb[rh * kHD + i] = common::f32_to_bf16(float(q8[rh * kHD + i]) * qs[rh]);   // exact
    }
  }
  auto dk = pf_harness::upload(d.ctx, d.imm, k), dv = pf_harness::upload(d.ctx, d.imm, v);
  auto dks = pf_harness::upload(d.ctx, d.imm, ks), dvs = pf_harness::upload(d.ctx, d.imm, vs);
  auto dq8 = pf_harness::upload(d.ctx, d.imm, q8), dqs = pf_harness::upload(d.ctx, d.imm, qs);
  auto dqb = pf_harness::upload(d.ctx, d.imm, qb);
  const auto run = [&](l0::Module& m, bool i8s) {
    const size_t plane = size_t(depth + 256u) * kvrow;
    l0::Mem raw(d.ctx, l0::MemKind::Device, 2 * plane * 2);
    uint16_t* kr = raw.as<uint16_t>();
    uint16_t* vr = kr + plane;
    l0::Kernel w = m.kernel("kv8_widen");
    w.group_size(256);
    w.arg_ptr(0, dk.ptr());
    w.arg_ptr(1, dv.ptr());
    w.arg_ptr(2, kr);
    w.arg_ptr(3, vr);
    w.arg(4, count);
    d.run(w, (count + 4095u) / 4096u, 1);
    l0::Mem o(d.ctx, l0::MemKind::Device, size_t(qh) * rows * kHD * 4);
    d.imm.fill(o.ptr(), 0x3F123456u, o.size());
    l0::Kernel f = m.kernel("pf_flash_attn_kv8");
    uint32_t a = 0;
    f.arg_ptr(a++, i8s ? dq8.ptr() : dqb.ptr());
    if (i8s) f.arg_ptr(a++, dqs.ptr());
    f.arg_ptr(a++, kr);
    f.arg_ptr(a++, dks.ptr());
    f.arg_ptr(a++, vr);
    f.arg_ptr(a++, dvs.ptr());
    f.arg_ptr(a++, o.ptr());
    f.arg(a++, s.pos);
    f.arg(a++, s.C);
    f.arg(a++, rows);
    if (i8s) {
      f.group_size(kernels::kPfFlashPairHeads * 32);
      run3(d, f, (s.C + kernels::kPfFlashPairRows - 1u) / kernels::kPfFlashPairRows, kvh,
           qh / kvh / kernels::kPfFlashPairHeads);
    } else {
      f.group_size(96);
      run3(d, f, (s.C + 7u) / 8u, kvh, 1);
    }
    std::vector<float> out(o.size() / 4);
    pf_harness::download(d.imm, out, o);
    return out;
  };
  const auto ref = run(rm, false), got = run(im, true);
  const size_t diff = words_differ(got, ref);
  std::printf("T 12 / 2 pos %6u C %4u: %zu of %zu pf_o words differ from the card's raw flash -- %s\n", s.pos, s.C,
              diff, got.size(), diff ? "FAIL" : "PASS");
  return diff;
}

}  // namespace

int main() {
  if (!have(kernels::pf_flash_attn_kv8_raw_pair_i8s_variant())) {
    std::puts("SKIP: the KV8_I8S binaries are not built (B70_KV8 off)");
    return 77;
  }
  pf_harness::Dev d;
  size_t bad = part_w(d) ? 0 : 1;
  Kernels kn(d);
  if (!kn.pair_pg) std::puts("P: the paged binary is not built (B70_MTP / B70_VERIFY_M8 off), skipped");
  for (const Shape& s : {Shape{0, 1}, Shape{0, 9}, Shape{0, 64}, Shape{777, 300}, Shape{2000, 2048},
                         Shape{30963, 37}, Shape{62952, 2048}})
    bad += part_sp(d, kn, s);
  for (const Shape& s : {Shape{0, 64}, Shape{777, 300}, Shape{4000, 129}}) bad += part_d(d, kn, s);
  if (have(kernels::pf_flash_attn_kv8_raw_pair_i8s_variant(12, 2)))
    for (const Shape& s : {Shape{0, 9}, Shape{777, 300}, Shape{30963, 2048}}) bad += part_tp(d, s);
  else
    std::puts("T: the tensor-parallel binaries are not built (B70_TP off), skipped");
  std::printf("kv8_i8s_test: %s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
