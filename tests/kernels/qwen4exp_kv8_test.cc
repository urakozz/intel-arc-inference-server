// Plan 21-QFN-v3a Task 1: Qwen3.8-Flash-Next's int8 KV cache kernels on the card against src/common/kv8.h (the rotkv
// scheme at head_dim 256) - no checkpoint, synthetic inputs; exits 77 (SKIP) without a GPU.
//   1. the writers: kv8.cl's attn_prep_kv8 at 24 / 2 heads (decode, M 1) stores encode() of exactly the bf16 K / V rows
//      attn_prep_M1_Q24KV2 stores and q = rotate() of its fp32 q, the gate the same bits; the prompt pass's
//      (pf_attn_prep_kv8_Q24KV2 against pf_attn_prep_q16_Q24KV2, 37 rows) the same K / V codes and scales, its q at
//      cosine >= 0.99999 against rotate(rne(q)) (the bf16 writer keeps only rne(q)); the _KVO build's K / V the full
//      build's;
//   2. the decode reader: q4_qsa_attn_M1_T32_KV8 + its reduce over a cache encoded by kv8.h and a rotated q, against a
//      double twin on the same codes (scores, softmax, P V in the rotated basis, un-rotation, x rne(sigmoid(gate))) at
//      cosine >= 0.99999 (spec 6 K1's bar), at p 1, 2050, 2051, 9000, 131072; the cosine against the bf16 cache's fp64
//      result printed (what the codes cost);
//   3. the prompt pass's readers: pf_flash_attn_kv8_Q24KV2 (the dense rows) and q4_pf_sparse_attn_Q24KV2_KV8 against the
//      same twin in the rotated basis (cosine >= 0.99999); pf_attn_gate_kv8_Q24KV2 within one bf16 ulp of
//      rne(rne(unrotate(o)) x sigmoid(gate)) (kv8.h's unrotate: the device's op order).
// qwen4exp_kv8_test --bench: the readers' device time, int8 against bf16 on the same lists (decode: attention + reduce
// at p 131072, 2049 listed; prompt: the sparse flash over 64 rows of 2048..2051 listed at pos 131072) - lists of 50
// launches, a warm-up, then 3 rounds A / B interleaved; the median round in us a launch.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "kernels/qwen4exp_pf_ref.h"
#include "kernels/qwen4exp_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"

namespace {
namespace qk = kernels::qwen4exp;
namespace qr = q4ref;
namespace qp = q4pf;
namespace kv8 = common::kv8;
using qr::f32;
using qr::rf;
using qr::rne;
constexpr uint32_t HD = qk::kHd, QH = qk::kQHeads, KVH = qk::kKvHeads, QN = QH * HD, KVN = KVH * HD, QKVN = qk::kQkvgN;
int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}

std::vector<float> rnd(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = u(g);
  return v;
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  template <class T>
  l0::Mem up(const std::vector<T>& v) {
    l0::Mem m(ctx, l0::MemKind::Device, std::max<size_t>(v.size() * sizeof(T), 64));
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
    return m;
  }
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, std::max<size_t>(bytes, 64));
    imm.fill(m.ptr(), 0u, std::max<size_t>(bytes, 64));
    return m;
  }
  template <class T>
  std::vector<T> rd(const l0::Mem& m, size_t n, size_t off_bytes = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off_bytes, n * sizeof(T));
    return v;
  }
  // one launch of `entry` from binary `variant`, arguments in order (pointers as void*, values as uint32_t)
  struct A {
    bool ptr;
    const void* p;
    uint32_t u;
    A(const l0::Mem& m) : ptr(true), p(m.ptr()), u(0) {}
    A(const void* q) : ptr(true), p(q), u(0) {}
    A(uint32_t v) : ptr(false), p(nullptr), u(v) {}
  };
  void go(const std::string& variant, const char* entry, uint32_t wg, uint32_t gx, uint32_t gy, uint32_t gz,
          std::initializer_list<A> args) {
    l0::Module m(ctx, kernels::path(variant));
    l0::Kernel k = m.kernel(entry);
    k.group_size(wg);
    uint32_t i = 0;
    for (const A& a : args) {
      if (a.ptr) k.arg_ptr(i, a.p);
      else k.arg<uint32_t>(i, a.u);
      ++i;
    }
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy, gz);
    list.close();
    q.execute(list, &f);
    f.wait();
  }
};

struct Ctl {
  l0::Mem m;
  explicit Ctl(l0::Context& ctx) : m(ctx, l0::MemKind::Shared, sizeof(runtime::Control)) {}
  void at(uint32_t pos, uint32_t n) {
    std::memset(m.ptr(), 0, m.size());
    m.as<runtime::Control>()->pos = pos;
    m.as<runtime::Control>()->n_active = n;
  }
};

double cosine(const double* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += double(b[i]) * b[i];
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}
int ulps16(uint16_t a, uint16_t b) {
  const auto key = [](uint16_t x) { return (x & 0x8000) ? -int(x & 0x7FFF) : int(x); };
  return std::abs(key(a) - key(b));
}
std::vector<float> rope_table(uint32_t rows) {   // fp32 [rows][2][32] (the loader's form; any table serves)
  std::vector<float> t(size_t(rows) * 64);
  for (uint32_t p = 0; p < rows; ++p)
    for (uint32_t i = 0; i < 32; ++i) {
      const double a = double(p) * std::pow(1e7, -double(2 * i) / 64.0);
      t[size_t(p) * 64 + i] = float(std::cos(a));
      t[size_t(p) * 64 + 32 + i] = float(std::sin(a));
    }
  return t;
}

// The int8 cache of bf16 rows (what the writer stores): codes, scales, and the de-quantised rotated values (exact in
// fp32: code x f16(scale)).
struct Cache8 {
  std::vector<int8_t> k, v;
  std::vector<uint16_t> ks, vs;
  std::vector<float> kd, vd;
  Cache8(const std::vector<uint16_t>& kb, const std::vector<uint16_t>& vb)
      : k(kb.size()), v(vb.size()), ks(kb.size() / HD), vs(vb.size() / HD), kd(kb.size()), vd(vb.size()) {
    for (size_t r = 0; r < ks.size(); ++r) {
      float x[HD];
      for (uint32_t i = 0; i < HD; ++i) x[i] = f32(kb[r * HD + i]);
      ks[r] = kv8::encode(x, &k[r * HD]);
      for (uint32_t i = 0; i < HD; ++i) x[i] = f32(vb[r * HD + i]);
      vs[r] = kv8::encode(x, &v[r * HD]);
      for (uint32_t i = 0; i < HD; ++i) {
        kd[r * HD + i] = kv8::dequant(k[r * HD + i], ks[r]);
        vd[r * HD + i] = kv8::dequant(v[r * HD + i], vs[r]);
      }
    }
  }
};

// Attention of head h of one row (q: that row's QN values, rotated) over `count` listed positions of a de-quantised
// rotated cache, fp64: the rotated output.
std::vector<double> attn64(const float* q, const std::vector<float>& kd, const std::vector<float>& vd,
                           const uint32_t* list, uint32_t count, uint32_t h) {
  const uint32_t j = h / (QH / KVH);
  std::vector<double> sc(count), o(HD, 0.0);
  double mx = -INFINITY, z = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const float* kr = &kd[(size_t(list[i]) * KVH + j) * HD];
    double a = 0;
    for (uint32_t d = 0; d < HD; ++d) a += double(q[h * HD + d]) * kr[d];
    sc[i] = a / 16.0;
    mx = std::max(mx, sc[i]);
  }
  for (uint32_t i = 0; i < count; ++i) {
    const double p = std::exp(sc[i] - mx);
    z += p;
    const float* vr = &vd[(size_t(list[i]) * KVH + j) * HD];
    for (uint32_t d = 0; d < HD; ++d) o[d] += p * vr[d];
  }
  for (double& x : o) x /= z;
  return o;
}
// kv8.h's un-rotation in fp64: x = FWHT(s (.) y) / 16
std::vector<double> unrotate64(const std::vector<double>& y) {
  std::vector<double> v(HD);
  for (uint32_t j = 0; j < HD; ++j) v[j] = kv8::sign_neg(j) ? -y[j] : y[j];
  for (uint32_t h = 1; h < HD; h <<= 1)
    for (uint32_t i = 0; i < HD; ++i)
      if ((i & h) == 0) {
        const double a = v[i], c = v[i + h];
        v[i] = a + c;
        v[i + h] = a - c;
      }
  for (double& x : v) x /= 16.0;
  return v;
}

// ---- 1. the writers ------------------------------------------------------------------------------------------------
void writers(Dev& dv) {
  const uint32_t pos = 777, max_len = 1024, C = 37;
  const std::vector<float> rope = rope_table(max_len);
  std::vector<float> small = rnd(512, 0.5f, 1.5f, 11);
  l0::Mem rb = dv.up(rope), sb = dv.up(small);
  Ctl ctl(dv.ctx);
  // decode, M 1: the partials of the two split-K slices
  {
    const std::vector<float> part = rnd(size_t(2) * QKVN, -2.f, 2.f, 12);
    l0::Mem pb = dv.up(part), q0 = dv.zeros(QN * 4), g0 = dv.zeros(QN * 4), q1 = dv.zeros(QN * 4), g1 = dv.zeros(QN * 4),
            k0 = dv.zeros(size_t(max_len) * KVN * 2), v0 = dv.zeros(size_t(max_len) * KVN * 2),
            k1 = dv.zeros(size_t(max_len) * KVN), v1 = dv.zeros(size_t(max_len) * KVN), ks = dv.zeros(size_t(max_len) * KVH * 2),
            vs = dv.zeros(size_t(max_len) * KVH * 2);
    ctl.at(pos, 1);
    const std::string bv = qk::attn_prep_q4_variant(1, 2);
    dv.go(bv, "attn_prep", 256, QH + KVH, 1, 1, {ctl.m, pb, sb, rb, q0, g0, k0, v0});
    dv.go(qk::kv8_variant(bv), "attn_prep_kv8", 256, QH + KVH, 1, 1, {ctl.m, pb, sb, rb, q1, g1, k1, v1, ks, vs});
    const std::vector<float> qa = dv.rd<float>(q0, QN), qb = dv.rd<float>(q1, QN);
    std::vector<float> want(QN);
    for (uint32_t h = 0; h < QH; ++h) kv8::rotate(&qa[size_t(h) * HD], &want[size_t(h) * HD]);
    check(std::memcmp(want.data(), qb.data(), QN * 4) == 0, "decode writer: q is not rotate(the bf16 writer's q)");
    check(dv.rd<float>(g0, QN) == dv.rd<float>(g1, QN), "decode writer: the gate differs");
    bool kv_ok = true;
    for (int which = 0; which < 2; ++which) {
      const std::vector<uint16_t> rows = dv.rd<uint16_t>(which ? v0 : k0, KVN, size_t(pos) * KVN * 2);
      const std::vector<int8_t> codes = dv.rd<int8_t>(which ? v1 : k1, KVN, size_t(pos) * KVN);
      const std::vector<uint16_t> sc = dv.rd<uint16_t>(which ? vs : ks, KVH, size_t(pos) * KVH * 2);
      for (uint32_t j = 0; j < KVH; ++j) {
        float x[HD];
        int8_t c[HD];
        for (uint32_t i = 0; i < HD; ++i) x[i] = f32(rows[j * HD + i]);
        kv_ok &= kv8::encode(x, c) == sc[j] && std::memcmp(c, &codes[j * HD], HD) == 0;
      }
    }
    check(kv_ok, "decode writer: K / V codes or scales are not encode(the bf16 writer's rows)");
    std::printf("1. decode writer (attn_prep_M1_Q24KV2_KV8): q = rotate(q), the gate, K / V = encode(bf16 rows) - %s\n",
                kv_ok ? "bitwise" : "DIFFERS");
  }
  // the prompt pass, C rows at pos
  {
    const std::vector<float> part = rnd(size_t(C) * QKVN, -2.f, 2.f, 13);
    l0::Mem pb = dv.up(part), q0 = dv.zeros(size_t(C) * QN * 2), q1 = dv.zeros(size_t(C) * QN * 2),
            k0 = dv.zeros(size_t(max_len) * KVN * 2), v0 = dv.zeros(size_t(max_len) * KVN * 2),
            k1 = dv.zeros(size_t(max_len) * KVN), v1 = dv.zeros(size_t(max_len) * KVN), ks = dv.zeros(size_t(max_len) * KVH * 2),
            vs = dv.zeros(size_t(max_len) * KVH * 2), k2 = dv.zeros(size_t(max_len) * KVN), v2 = dv.zeros(size_t(max_len) * KVN),
            ks2 = dv.zeros(size_t(max_len) * KVH * 2), vs2 = dv.zeros(size_t(max_len) * KVH * 2);
    ctl.at(pos, C);
    const std::string bv = qk::pf_attn_prep_q4_variant(), kv = qk::kv8_variant(bv);
    dv.go(bv, "pf_attn_prep", 256, QH + KVH, C, 1, {ctl.m, pb, sb, rb, q0, nullptr, k0, v0});
    dv.go(kv, "attn_prep_kv8", 256, QH + KVH, C, 1, {ctl.m, pb, sb, rb, q1, nullptr, k1, v1, ks, vs});
    dv.go(qk::kv8_variant(kernels::kv_only(bv)), "attn_prep_kv8", 256, KVH, C, 1, {ctl.m, pb, sb, rb, q1, nullptr, k2, v2, ks2, vs2});
    bool kv_ok = true;
    for (int which = 0; which < 2; ++which) {
      const std::vector<uint16_t> rows = dv.rd<uint16_t>(which ? v0 : k0, size_t(C) * KVN, size_t(pos) * KVN * 2);
      const std::vector<int8_t> codes = dv.rd<int8_t>(which ? v1 : k1, size_t(C) * KVN, size_t(pos) * KVN);
      const std::vector<uint16_t> sc = dv.rd<uint16_t>(which ? vs : ks, size_t(C) * KVH, size_t(pos) * KVH * 2);
      for (uint32_t r = 0; r < C * KVH; ++r) {
        float x[HD];
        int8_t c[HD];
        for (uint32_t i = 0; i < HD; ++i) x[i] = f32(rows[size_t(r) * HD + i]);
        kv_ok &= kv8::encode(x, c) == sc[r] && std::memcmp(c, &codes[size_t(r) * HD], HD) == 0;
      }
      kv_ok &= dv.rd<int8_t>(which ? v2 : k2, size_t(C) * KVN, size_t(pos) * KVN) == codes &&
               dv.rd<uint16_t>(which ? vs2 : ks2, size_t(C) * KVH, size_t(pos) * KVH * 2) == sc;
    }
    check(kv_ok, "prompt writer: K / V codes or scales are not encode(the bf16 writer's rows), or _KVO's differ");
    const std::vector<uint16_t> qa = dv.rd<uint16_t>(q0, size_t(C) * QN), qb = dv.rd<uint16_t>(q1, size_t(C) * QN);
    double worst = 1.0;
    for (size_t r = 0; r < size_t(C) * QH; ++r) {
      float x[HD], y[HD], got[HD];
      double want[HD];
      for (uint32_t i = 0; i < HD; ++i) x[i] = f32(qa[r * HD + i]);
      kv8::rotate(x, y);
      for (uint32_t i = 0; i < HD; ++i) {
        want[i] = y[i];
        got[i] = f32(qb[r * HD + i]);
      }
      worst = std::min(worst, cosine(want, got, HD));
    }
    check(worst >= 0.99999, "prompt writer: q below cosine 0.99999 against rotate(rne(q))");
    std::printf("1. prompt writer (pf_attn_prep_kv8_Q24KV2, %u rows; _KVO): K / V = encode(bf16 rows) %s; q against "
                "rotate(rne(q)) worst cosine %.7f (the bf16 writer keeps rne(q), this one rne(rotate(q)))\n", C,
                kv_ok ? "bitwise" : "DIFFERS", worst);
  }
}

// ---- 2. the decode reader ------------------------------------------------------------------------------------------
void decode_reader(Dev& dv) {
  const uint32_t rows = 131073;
  const std::vector<uint16_t> kb = [&] {
    std::vector<uint16_t> v(size_t(rows) * KVN);
    const std::vector<float> f = rnd(v.size(), -1.f, 1.f, 21);
    for (size_t i = 0; i < v.size(); ++i) v[i] = rne(f[i]);
    return v;
  }();
  const std::vector<uint16_t> vb = [&] {
    std::vector<uint16_t> v(size_t(rows) * KVN);
    const std::vector<float> f = rnd(v.size(), -1.f, 1.f, 22);
    for (size_t i = 0; i < v.size(); ++i) v[i] = rne(f[i]);
    return v;
  }();
  const Cache8 c(kb, vb);
  const std::vector<float> q = rnd(QN, -2.f, 2.f, 23), gate = rnd(QN, -4.f, 4.f, 24);
  std::vector<float> qrot(QN);
  for (uint32_t h = 0; h < QH; ++h) kv8::rotate(&q[size_t(h) * HD], &qrot[size_t(h) * HD]);
  l0::Mem k8 = dv.up(c.k), v8 = dv.up(c.v), ks = dv.up(c.ks), vs = dv.up(c.vs), qb = dv.up(qrot), gb = dv.up(gate),
          part = dv.zeros(size_t(QH) * qk::kAttnTgt * qk::kAttnPart * 4), ob = dv.zeros(QN * 2);
  const std::vector<float> scores = rnd(32769, 0.f, 4.f, 25);
  const std::string v = qk::kv8_variant(qk::qsa_attn_variant(1, false));
  Ctl ctl(dv.ctx);
  double worst = 1.0, vs_bf16 = 1.0;
  for (uint32_t p : {1u, 2050u, 2051u, 9000u, 131072u}) {
    const qr::Selection sel = qr::qsa_select(scores.data(), p);
    std::vector<uint32_t> row(qk::kListRow, 0);
    std::copy(sel.list.begin(), sel.list.end(), row.begin());
    row[qk::kCountWord] = sel.count();
    l0::Mem lb = dv.up(row);
    ctl.at(p, 1);
    dv.go(v, "q4_qsa_attn", qk::kAttnWg, KVH, qk::kAttnTgt, 1, {ctl.m, qb, k8, v8, lb, part, ks, vs});
    dv.go(v, "q4_qsa_reduce", qk::kAttnWg, QH, 1, 1, {ctl.m, part, gb, lb, ob});
    const std::vector<uint16_t> o = dv.rd<uint16_t>(ob, QN);
    double w = 1.0;
    for (uint32_t h = 0; h < QH; ++h) {
      const std::vector<double> x = unrotate64(attn64(qrot.data(), c.kd, c.vd, sel.list.data(), sel.count(), h));
      const std::vector<double> ref = qr::qsa_attn_fp64(q.data(), kb.data(), vb.data(), sel.list.data(), sel.count(), h);
      std::vector<double> want(HD), plain(HD);
      std::vector<float> got(HD);
      for (uint32_t d = 0; d < HD; ++d) {
        const double sg = rf(qr::sigmoid_t(gate[size_t(h) * HD + d]));
        want[d] = x[d] * sg;
        plain[d] = ref[d] * sg;
        got[d] = f32(o[size_t(h) * HD + d]);
      }
      w = std::min(w, cosine(want.data(), got.data(), HD));
      vs_bf16 = std::min(vs_bf16, cosine(plain.data(), got.data(), HD));
    }
    worst = std::min(worst, w);
    std::printf("2. %s at p %6u (%u listed): worst cosine %.7f against the int8 twin\n", v.c_str(), p, sel.count(), w);
  }
  check(worst >= 0.99999, "decode reader below cosine 0.99999 against its fp64 twin");
  std::printf("2. decode reader: worst %.7f (bar 0.99999); against the bf16 cache's fp64 result %.6f (the codes' cost)\n",
              worst, vs_bf16);
}

// ---- 3. the prompt pass's readers ----------------------------------------------------------------------------------
void prompt_readers(Dev& dv) {
  constexpr uint32_t kC = 64;
  const uint32_t maxd = 131072 + 2048;
  std::vector<uint16_t> kb(size_t(maxd) * KVN), vb(kb.size());
  {
    const std::vector<float> a = rnd(kb.size(), -1.f, 1.f, 31), b = rnd(vb.size(), -1.f, 1.f, 32);
    for (size_t i = 0; i < kb.size(); ++i) {
      kb[i] = rne(a[i]);
      vb[i] = rne(b[i]);
    }
  }
  const Cache8 c(kb, vb);
  l0::Mem k8 = dv.up(c.k), v8 = dv.up(c.v), ks = dv.up(c.ks), vs = dv.up(c.vs);
  // the rotated bf16 q (what the writer leaves in q16) and its values
  const std::vector<float> qf0 = rnd(size_t(kC) * QN, -1.f, 1.f, 33);
  std::vector<uint16_t> qb(qf0.size());
  std::vector<float> qf(qf0.size());
  for (size_t r = 0; r < size_t(kC) * QH; ++r) {
    float y[HD];
    kv8::rotate(&qf0[r * HD], y);
    for (uint32_t i = 0; i < HD; ++i) {
      qb[r * HD + i] = rne(y[i]);
      qf[r * HD + i] = f32(qb[r * HD + i]);
    }
  }
  const std::vector<float> gate = rnd(size_t(kC) * QKVN, -4.f, 4.f, 34);
  l0::Mem dq = dv.up(qb), O = dv.zeros(size_t(QH) * kC * HD * 4), lists = dv.zeros(size_t(kC) * qk::kListRow * 4),
          dgate = dv.up(gate), out = dv.zeros(size_t(kC) * QN * 2);
  double worst_dense = 1.0, worst_sparse = 1.0;
  int gate_ulps = 0;
  for (const uint32_t pos : {0u, 2000u}) {   // the dense rows
    const uint32_t nd = qp::dense_rows(pos, kC);
    dv.go(qk::kv8_variant(qk::pf_flash_q4_variant()), "pf_flash_attn_kv8", 16 * (qk::kPfFlashRpw / 8) * qk::kPfFlashHpw,
          (nd + qk::kPfFlashRpw - 1) / qk::kPfFlashRpw, KVH, QH / KVH / qk::kPfFlashHpw,
          {dq, k8, ks, v8, vs, O, pos, nd, kC});
    const std::vector<float> o = dv.rd<float>(O, size_t(QH) * kC * HD);
    for (uint32_t r : {0u, 1u, 7u, 23u, nd / 2, nd - 1}) {
      if (r >= nd) continue;
      std::vector<uint32_t> ident(pos + r + 1);
      for (uint32_t i = 0; i <= pos + r; ++i) ident[i] = i;
      for (uint32_t h = 0; h < QH; h += 5) {
        const std::vector<double> want = attn64(qf.data() + size_t(r) * QN, c.kd, c.vd, ident.data(), pos + r + 1, h);
        worst_dense = std::min(worst_dense, cosine(want.data(), o.data() + (size_t(h) * kC + r) * HD, HD));
      }
    }
  }
  struct Case { uint32_t pos, C; };
  for (const Case cs : {Case{2048, 64}, Case{4096, 64}, Case{30000, 64}, Case{131072, 64}}) {   // the sparse rows
    const uint32_t r0 = qp::dense_rows(cs.pos, cs.C);
    std::vector<uint32_t> lh(size_t(cs.C) * qk::kListRow, 0), counts(cs.C, 0);
    std::mt19937 g(110 + cs.pos);
    std::uniform_real_distribution<float> u(0.f, 4.f);
    for (uint32_t m = r0; m < cs.C; ++m) {
      const uint32_t p = cs.pos + m, n = (p + 1) / 4;
      std::vector<float> s(n);
      for (float& x : s) x = rf(u(g));
      const qr::Selection sel = qr::qsa_select(s.data(), p);
      std::copy(sel.list.begin(), sel.list.end(), lh.begin() + size_t(m) * qk::kListRow);
      lh[size_t(m) * qk::kListRow + qk::kCountWord] = counts[m] = sel.count();
    }
    dv.imm.copy(lists.ptr(), lh.data(), lh.size() * 4);
    dv.go(qk::kv8_variant(qk::pf_sparse_attn_variant(false)), "q4_pf_sparse_attn", qk::kPfSparseWg, cs.C - r0, KVH, 1,
          {dq, k8, v8, lists, O, r0, kC, ks, vs});
    const std::vector<float> o = dv.rd<float>(O, size_t(QH) * kC * HD);
    for (uint32_t m = r0; m < cs.C; m += 3)
      for (uint32_t h = 0; h < QH; ++h) {
        const std::vector<double> want =
            attn64(qf.data() + size_t(m) * QN, c.kd, c.vd, lh.data() + size_t(m) * qk::kListRow, counts[m], h);
        worst_sparse = std::min(worst_sparse, cosine(want.data(), o.data() + (size_t(h) * kC + m) * HD, HD));
      }
    if (cs.pos == 4096) {   // the gate over the sparse O: un-rotation first
      dv.go(qk::kv8_variant(qk::pf_gate_q4_variant()), "pf_attn_gate_kv8", 256, QH, cs.C, 1,
            {O, dgate, out, uint32_t(kC * HD)});
      const std::vector<uint16_t> gt = dv.rd<uint16_t>(out, size_t(cs.C) * QN);
      for (uint32_t m = r0; m < cs.C; ++m)
        for (uint32_t h = 0; h < QH; ++h) {
          float x[HD];
          kv8::unrotate(o.data() + (size_t(h) * kC + m) * HD, x);
          for (uint32_t d = 0; d < HD; ++d) {
            const float gv = rf(gate[size_t(m) * QKVN + size_t(h) * 2 * HD + HD + d]);
            const uint16_t want = rne(rf(x[d]) * (1.0f / (1.0f + std::exp(-gv))));
            gate_ulps = std::max(gate_ulps, ulps16(gt[size_t(m) * QN + h * HD + d], want));
          }
        }
    }
    std::printf("3. chunk (%6u, %u): sparse rows worst %.7f\n", cs.pos, cs.C, worst_sparse);
  }
  check(worst_dense >= 0.99999, "the dense rows' int8 flash below cosine 0.99999");
  check(worst_sparse >= 0.99999, "the sparse rows' int8 flash below cosine 0.99999");
  check(gate_ulps <= 1, "pf_attn_gate_kv8 further than one bf16 ulp");
  std::printf("3. prompt pass: dense flash (pf_flash_attn_kv8_Q24KV2) worst %.7f, sparse (q4_pf_sparse_attn_Q24KV2_KV8) "
              "worst %.7f (bar 0.99999); the gate within %d bf16 ulp\n", worst_dense, worst_sparse, gate_ulps);
}

double timed(Dev& dv, const std::vector<std::pair<l0::Kernel*, std::array<uint32_t, 3>>>& ks, uint32_t reps) {
  l0::CmdList list = l0::CmdList::regular(dv.ctx);
  for (uint32_t r = 0; r < reps; ++r)
    for (const auto& [k, g] : ks) list.launch(*k, g[0], g[1], g[2]);
  list.close();
  const auto t0 = std::chrono::steady_clock::now();
  dv.q.execute(list, &dv.f);
  dv.f.wait();
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
}
l0::Kernel bound(l0::Module& m, const char* entry, uint32_t wg, std::initializer_list<Dev::A> args) {
  l0::Kernel k = m.kernel(entry);
  k.group_size(wg);
  uint32_t i = 0;
  for (const Dev::A& a : args) {
    if (a.ptr) k.arg_ptr(i, a.p);
    else k.arg<uint32_t>(i, a.u);
    ++i;
  }
  return k;
}
int bench(Dev& dv) {
  const uint32_t rows = 133120, pos = 131072;
  std::vector<uint16_t> kb(size_t(rows) * KVN), vb(kb.size());
  {
    const std::vector<float> a = rnd(kb.size(), -1.f, 1.f, 41), b = rnd(vb.size(), -1.f, 1.f, 42);
    for (size_t i = 0; i < kb.size(); ++i) {
      kb[i] = rne(a[i]);
      vb[i] = rne(b[i]);
    }
  }
  const Cache8 c(kb, vb);
  l0::Mem K = dv.up(kb), V = dv.up(vb), k8 = dv.up(c.k), v8 = dv.up(c.v), ks = dv.up(c.ks), vs = dv.up(c.vs);
  const std::vector<float> scores = rnd(pos / 4 + 1024, 0.f, 4.f, 43);
  Ctl ctl(dv.ctx);
  // decode
  const qr::Selection sel = qr::qsa_select(scores.data(), pos);
  std::vector<uint32_t> row(qk::kListRow, 0);
  std::copy(sel.list.begin(), sel.list.end(), row.begin());
  row[qk::kCountWord] = sel.count();
  l0::Mem lb = dv.up(row), q = dv.up(rnd(QN, -2.f, 2.f, 44)), g = dv.up(rnd(QN, -4.f, 4.f, 45)),
          part = dv.zeros(size_t(QH) * qk::kAttnTgt * qk::kAttnPart * 4), ob = dv.zeros(QN * 2);
  ctl.at(pos, 1);
  l0::Module mb(dv.ctx, kernels::path(qk::qsa_attn_variant(1, false))),
      m8(dv.ctx, kernels::path(qk::kv8_variant(qk::qsa_attn_variant(1, false))));
  l0::Kernel ab = bound(mb, "q4_qsa_attn", qk::kAttnWg, {ctl.m, q, K, V, lb, part}),
             rb = bound(mb, "q4_qsa_reduce", qk::kAttnWg, {ctl.m, part, g, lb, ob}),
             a8 = bound(m8, "q4_qsa_attn", qk::kAttnWg, {ctl.m, q, k8, v8, lb, part, ks, vs}),
             r8 = bound(m8, "q4_qsa_reduce", qk::kAttnWg, {ctl.m, part, g, lb, ob});
  const std::array<uint32_t, 3> ga{KVH, qk::kAttnTgt, 1}, gr{QH, 1, 1};
  // prompt: 64 sparse rows
  constexpr uint32_t kC = 64;
  std::vector<uint32_t> lh(size_t(kC) * qk::kListRow, 0);
  for (uint32_t m = 0; m < kC; ++m) {
    const qr::Selection s2 = qr::qsa_select(scores.data(), pos + m);
    std::copy(s2.list.begin(), s2.list.end(), lh.begin() + size_t(m) * qk::kListRow);
    lh[size_t(m) * qk::kListRow + qk::kCountWord] = s2.count();
  }
  std::vector<uint16_t> qh(size_t(kC) * QN);
  for (size_t i = 0; i < qh.size(); ++i) qh[i] = rne(float(int(i % 37) - 18) / 16.f);
  l0::Mem lists = dv.up(lh), Q = dv.up(qh), O = dv.zeros(size_t(QH) * kC * HD * 4);
  l0::Module sb(dv.ctx, kernels::path(qk::pf_sparse_attn_variant(false))),
      s8(dv.ctx, kernels::path(qk::kv8_variant(qk::pf_sparse_attn_variant(false))));
  l0::Kernel pb = bound(sb, "q4_pf_sparse_attn", qk::kPfSparseWg, {Q, K, V, lists, O, 0u, kC}),
             p8 = bound(s8, "q4_pf_sparse_attn", qk::kPfSparseWg, {Q, k8, v8, lists, O, 0u, kC, ks, vs});
  const std::array<uint32_t, 3> gp{kC, KVH, 1};
  for (int warm = 0; warm < 2; ++warm) {
    timed(dv, {{&ab, ga}, {&rb, gr}}, 50);
    timed(dv, {{&a8, ga}, {&r8, gr}}, 50);
    timed(dv, {{&pb, gp}}, 20);
    timed(dv, {{&p8, gp}}, 20);
  }
  std::vector<double> d0, d1, p0, p1;
  for (int r = 0; r < 3; ++r) {
    d0.push_back(timed(dv, {{&ab, ga}, {&rb, gr}}, 50));
    d1.push_back(timed(dv, {{&a8, ga}, {&r8, gr}}, 50));
    p0.push_back(timed(dv, {{&pb, gp}}, 20));
    p1.push_back(timed(dv, {{&p8, gp}}, 20));
  }
  for (auto* v : {&d0, &d1, &p0, &p1}) std::sort(v->begin(), v->end());
  std::printf("bench: decode attention + reduce (%u listed): bf16 %.1f us, int8 %.1f us (%.3fx)\n", sel.count(), d0[1], d1[1],
              d1[1] / d0[1]);
  std::printf("bench: prompt sparse flash (64 rows, ~2050 listed): bf16 %.1f us, int8 %.1f us (%.3fx)\n", p0[1], p1[1],
              p1[1] / p0[1]);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (l0::Context::gpu_count() == 0) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  Dev dv;
  if (argc > 1 && std::string(argv[1]) == "--bench") return bench(dv);
  writers(dv);
  decode_reader(dv);
  prompt_readers(dv);
  if (failures) {
    std::printf("qwen4exp_kv8_test: %d FAIL(s)\n", failures);
    return 1;
  }
  std::printf("qwen4exp_kv8_test OK\n");
  return 0;
}
