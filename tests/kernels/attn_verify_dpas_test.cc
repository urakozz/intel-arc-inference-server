// attn_verify_dpas_test - spec 28c's gates N1 / N2a / N2b on the card for the PRODUCTION binaries
// attn_verify_dpas_M<M>_T32[_KV8] (src/kernels/attn_verify_dpas.cl at the CMake block's lever line) followed by
// the attn_v2_M<M>_T32[_KV8] binary's reduce - exactly the pair the capture binds (plan 28c Task 6).
//
//   attn_verify_dpas_test [<dump dir>]
//
// Part S (synthetic, always): random bf16 KV of 4096 rows (and its int8 form through src/common/kv8.h), random q /
// gate, at the positions attn_v2_test walks (depth 0 -> 1, tiny pos, the 64-block edges, the stride's first step
// 2048 -> 2049 keys inside one launch, the end of the cache), M = 2..8 bf16 and M = 2..4 int8:
//   N1   every gated row against tests/kernels/attn_verify_ref.h (fp64): cosine >= kN1 = 0.999997 per (q head, row),
//        finite (spec 28c §4's 0.99999, tightened 2026-11-04: see kN1);
//        v2's own worst cosine on the same inputs printed beside it;
//   N2b  every row against v2 M = 1 at pos + m (cosine >= kN1, max abs printed);
//   rep  two runs bitwise;
//   N2a  property R: per (KV form, position) 1000 trials - a probe row (q, gate) at a random slot of a random M launch
//        positioned so that it keeps its key count, the other rows random: its gated row bitwise the same every time.
// Part D (real, when <dump dir> is plan 28c Task 4's capture, probe_decode_attn capture --rows 8): FA layer 15's KV and
//   the q / gate of 8 plain steps from each depth in {512, 4096, 32768, 130816}, KV at 131072 rows: N1, N2b and rep at
//   every M, N2a with 100 trials a depth. Without a dump part D is skipped (printed) and part S decides.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/attn_verify_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/verify_attn.h"

namespace {

namespace avr = attn_verify_ref;
// N1's bar. Spec 28c §4 says 0.99999; the kernel with P in bf16 alone met it here (0.999991 - 0.999994 on random data,
// 0.9999880 on the int8 dump's row at 4096, which the operator ruling of 2026-11-03 accepted) and still moved real
// DFlash verify rows off the f32 top (docs/07-open-questions.md §11): random rows do not show the cancellation that
// amplifies P's rounding on real heads. P as hi + lo (PSPLIT=1) measures 0.9999984 - 0.99999999 on both parts, so the
// bar sits between the two kernels: a return to P in bf16 fails it. That also retires the int8 4096 exception.
constexpr double kN1 = 0.999997;
constexpr uint32_t kQH = 24, kKvH = 4, kHD = 256, kRow = kQH * kHD;
constexpr uint32_t kTgt = runtime::DecodeScratch::kAttnV2Blocks, kPart = 258, kMaxM = 8;

// The production lever line's work-group (attn_verify_dpas.cl: 16 RB DS KS). The test reads the levers from the
// binary's own reqd_work_group_size through the kernel properties instead of repeating CMake's line.
uint32_t wg_of(l0::Kernel& k) {
  ze_kernel_properties_t p{};
  p.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  zeKernelGetProperties(k.handle(), &p);
  return p.requiredGroupSizeX;
}

double cos_rows(const uint16_t* a, const uint16_t* b, double* maxabs) {
  double xy = 0, xx = 0, yy = 0;
  for (uint32_t i = 0; i < kHD; ++i) {
    const double x = common::bf16_to_f32(a[i]), y = common::bf16_to_f32(b[i]);
    if (!std::isfinite(x)) return -2.0;
    xy += x * y;
    xx += x * x;
    yy += y * y;
    *maxabs = std::max(*maxabs, std::fabs(x - y));
  }
  if (xx == 0 || yy == 0) return xx == yy ? 1.0 : 0.0;
  return xy / std::sqrt(xx * yy);
}

struct Rig {
  l0::Context& ctx;
  uint32_t L;
  bool kv8;
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem ctrl{ctx, l0::MemKind::Shared, sizeof(runtime::Control)};
  std::unique_ptr<l0::Mem> k, v, ks, vs;
  l0::Mem aq{ctx, l0::MemKind::Device, size_t(kMaxM) * kRow * 4};
  l0::Mem ag{ctx, l0::MemKind::Device, size_t(kMaxM) * kRow * 4};
  l0::Mem part{ctx, l0::MemKind::Device, size_t(kQH) * kTgt * kMaxM * kPart * 4};
  l0::Mem out{ctx, l0::MemKind::Device, size_t(kMaxM) * kRow * 2};
  std::map<std::string, std::unique_ptr<l0::Module>> mods;
  // host copies
  std::vector<uint16_t> k16, v16;
  std::unique_ptr<kv8_ref::Cache> c8;

  Rig(l0::Context& c, uint32_t max_len, bool int8, std::vector<uint16_t> hk, std::vector<uint16_t> hv)
      : ctx(c), L(max_len), kv8(int8), k16(std::move(hk)), v16(std::move(hv)) {
    const size_t rows = k16.size() / kHD;   // (position, kv head) rows given
    k16.resize(size_t(L) * kKvH * kHD, 0);
    v16.resize(size_t(L) * kKvH * kHD, 0);
    if (kv8) {
      c8 = std::make_unique<kv8_ref::Cache>(L);
      for (size_t r = 0; r < rows; ++r) c8->put(r, &k16[r * kHD], &v16[r * kHD]);
      k = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, c8->k.size());
      v = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, c8->v.size());
      ks = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, c8->ks.size() * 2);
      vs = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, c8->vs.size() * 2);
      put(k->ptr(), c8->k.data(), c8->k.size());
      put(v->ptr(), c8->v.data(), c8->v.size());
      put(ks->ptr(), c8->ks.data(), c8->ks.size() * 2);
      put(vs->ptr(), c8->vs.data(), c8->vs.size() * 2);
    } else {
      k = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, k16.size() * 2);
      v = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, v16.size() * 2);
      put(k->ptr(), k16.data(), k16.size() * 2);
      put(v->ptr(), v16.data(), v16.size() * 2);
    }
  }
  void put(void* dev, const void* host, size_t bytes) {
    constexpr size_t kChunk = 64ul << 20;
    for (size_t off = 0; off < bytes; off += kChunk)
      imm.copy(static_cast<char*>(dev) + off, static_cast<const char*>(host) + off, std::min(kChunk, bytes - off));
  }
  avr::Kv hkv() const { return kv8 ? avr::Kv::int8(*c8) : avr::Kv::bf16(k16.data(), v16.data()); }
  l0::Module& mod(const std::string& variant) {
    auto it = mods.find(variant);
    if (it != mods.end()) return *it->second;
    auto m = std::make_unique<l0::Module>(ctx, kernels::path(variant));
    l0::Module& r = *m;
    mods[variant] = std::move(m);
    return r;
  }
  // One launch at (pos, M) of the DPAS pair (dpas) or v2's pair; q / gate already uploaded; returns [M][6144].
  std::vector<uint16_t> run(uint32_t pos, uint32_t M, bool dpas) {
    runtime::Control* c = ctrl.as<runtime::Control>();
    c->pos = pos;
    c->n_active = M;
    const std::string v2 = kv8 ? kernels::attn_v2_kv8_variant(M, kTgt) : kernels::attn_v2_variant(M, kTgt);
    l0::Module& rm = mod(v2);
    l0::Module& dm = dpas ? mod(kernels::attn_verify_dpas_variant(M, kTgt, kv8)) : rm;
    const char* dn = dpas ? (kv8 ? "attn_verify_dpas_kv8" : "attn_verify_dpas")
                          : (kv8 ? "attn_decode_v2_kv8" : "attn_decode_v2");
    l0::Kernel dec(dm, dn);
    dec.group_size(dpas ? wg_of(dec) : 256);
    uint32_t a = 0;
    dec.arg_ptr(a++, ctrl.ptr());
    dec.arg_ptr(a++, aq.ptr());
    dec.arg_ptr(a++, k->ptr());
    if (kv8) dec.arg_ptr(a++, ks->ptr());
    dec.arg_ptr(a++, v->ptr());
    if (kv8) dec.arg_ptr(a++, vs->ptr());
    dec.arg_ptr(a++, part.ptr());
    l0::Kernel red(rm, kv8 ? "attn_reduce_v2_kv8" : "attn_reduce_v2");
    red.group_size(256);
    red.arg_ptr(0, ctrl.ptr());
    red.arg_ptr(1, part.ptr());
    red.arg_ptr(2, ag.ptr());
    red.arg_ptr(3, out.ptr());
    // A partial the decode kernel does not write must not be read: the buffer starts as NaN for every run.
    imm.fill(part.ptr(), 0xFFFFFFFFu, part.size());
    l0::CmdList l = l0::CmdList::regular(ctx);
    l.launch(dec, kKvH, kTgt);
    l.launch(red, kQH, M);
    l.close();
    imm.fill(out.ptr(), 0u, size_t(M) * kRow * 2);
    q.execute(l, &fence);
    fence.wait();
    std::vector<uint16_t> o(size_t(M) * kRow);
    imm.copy(o.data(), out.ptr(), o.size() * 2);
    return o;
  }
};

struct Worst {
  double n1 = 2, n1_v2 = 2, n2b = 2, abs_n2b = 0;
  bool rep = true;
};

// N1 / N2b / rep at (pos, M) with q / gate (M rows; host copies).
void check(Rig& r, uint32_t pos, uint32_t M, const std::vector<float>& q, const std::vector<float>& g, Worst& w) {
  r.put(r.aq.ptr(), q.data(), size_t(M) * kRow * 4);
  r.put(r.ag.ptr(), g.data(), size_t(M) * kRow * 4);
  const std::vector<uint16_t> o1 = r.run(pos, M, true), o2 = r.run(pos, M, true), ov2 = r.run(pos, M, false);
  w.rep = w.rep && o1 == o2;
  const avr::Result ref = avr::run(pos, M, q.data(), g.data(), r.hkv());
  double ab = 0;
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t h = 0; h < kQH; ++h) {
      const size_t k = size_t(m) * kRow + h * kHD;
      w.n1 = std::min(w.n1, cos_rows(&o1[k], &ref.out[k], &ab));
      w.n1_v2 = std::min(w.n1_v2, cos_rows(&ov2[k], &ref.out[k], &ab));
    }
  // v2 M = 1 at pos + m
  for (uint32_t m = 0; m < M; ++m) {
    r.put(r.aq.ptr(), &q[size_t(m) * kRow], kRow * 4);
    r.put(r.ag.ptr(), &g[size_t(m) * kRow], kRow * 4);
    const std::vector<uint16_t> one = r.run(pos + m, 1, false);
    for (uint32_t h = 0; h < kQH; ++h)
      w.n2b = std::min(w.n2b, cos_rows(&o1[size_t(m) * kRow + h * kHD], &one[h * kHD], &w.abs_n2b));
  }
}

// N2a at a probe row with key count pos0 + 1: trials over the given M list.
uint32_t prop_r(Rig& r, uint32_t pos0, const std::vector<uint32_t>& Ms, const float* q0, const float* g0, uint32_t trials,
                std::mt19937& rng, uint32_t* done) {
  std::normal_distribution<float> nd(0.f, 1.f);
  double rms = 0;
  for (uint32_t i = 0; i < kRow; ++i) rms += double(q0[i]) * q0[i];
  const float sig = float(std::sqrt(rms / kRow));
  std::vector<uint16_t> first;
  uint32_t same = 0;
  *done = 0;
  for (uint32_t t = 0; t < trials; ++t) {
    const uint32_t M = Ms[rng() % Ms.size()], s = rng() % M;
    if (pos0 < s || pos0 - s + M > r.L) continue;
    std::vector<float> q(size_t(M) * kRow), g(size_t(M) * kRow);
    for (auto& x : q) x = sig * nd(rng);
    for (auto& x : g) x = nd(rng);
    std::memcpy(&q[size_t(s) * kRow], q0, kRow * 4);
    std::memcpy(&g[size_t(s) * kRow], g0, kRow * 4);
    r.put(r.aq.ptr(), q.data(), q.size() * 4);
    r.put(r.ag.ptr(), g.data(), g.size() * 4);
    const std::vector<uint16_t> o = r.run(pos0 - s, M, true);
    std::vector<uint16_t> row(o.begin() + size_t(s) * kRow, o.begin() + size_t(s + 1) * kRow);
    if (first.empty()) first = row;
    same += row == first;
    ++*done;
  }
  return same;
}

template <class T>
std::vector<T> read_file(const std::string& p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
  if (!f) {
    v.clear();
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  l0::Context ctx;
  std::printf("attn_verify_dpas_test on %s\n", ctx.name().c_str());
  std::mt19937 rng(28);
  bool ok = true;
  // ---- part S ----
  {
    constexpr uint32_t L = 4096;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> hk(size_t(L) * kKvH * kHD), hv(hk.size());
    for (auto& x : hk) x = common::f32_to_bf16(nd(rng));
    for (auto& x : hv) x = common::f32_to_bf16(nd(rng));
    for (bool kv8 : {false, true}) {
      Rig r(ctx, L, kv8, hk, hv);
      const std::vector<uint32_t> Ms = kv8 ? std::vector<uint32_t>{2, 3, 4} : std::vector<uint32_t>{2, 3, 4, 5, 6, 7, 8};
      for (uint32_t pos : {0u, 1u, 5u, 63u, 64u, 127u, 2041u, 2044u, 2047u, 3000u, 4088u}) {
        Worst w;
        for (uint32_t M : Ms) {
          if (pos + M > L) continue;
          std::vector<float> q(size_t(M) * kRow), g(q.size());
          for (auto& x : q) x = nd(rng);
          for (auto& x : g) x = nd(rng);
          if (kv8)
            for (uint32_t i = 0; i < M * kQH; ++i) {
              float y[kHD];
              common::kv8::rotate(&q[size_t(i) * kHD], y);
              std::memcpy(&q[size_t(i) * kHD], y, sizeof y);
            }
          check(r, pos, M, q, g, w);
        }
        std::vector<float> q0(kRow), g0(kRow);
        for (auto& x : q0) x = nd(rng);
        for (auto& x : g0) x = nd(rng);
        uint32_t done = 0;
        const uint32_t same = prop_r(r, pos, Ms, q0.data(), g0.data(), 1000, rng, &done);
        const bool pass = w.n1 >= kN1 && w.n2b >= kN1 && w.rep && same == done;
        ok = ok && pass;
        std::printf("S %s pos %4u: N1 worst %.8f (v2 %.8f)  N2b worst %.8f (max abs %.3g)  rep %s  N2a %u/%u bitwise%s\n",
                    kv8 ? "int8" : "bf16", pos, w.n1, w.n1_v2, w.n2b, w.abs_n2b, w.rep ? "yes" : "NO", same, done,
                    pass ? "" : "  <-- FAIL");
      }
    }
  }
  // ---- sweep (spec 28c diagnostic, `--sweep` as the only argument): every position 0 .. 4088 at M = 2, 3, 4, 8 -
  // each DPAS row against v2's M = 1 step at pos + m (cosine), the worst per M with where it is ----
  if (argc > 1 && std::string(argv[1]) == "--sweep") {
    constexpr uint32_t L = 4096;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> hk(size_t(L) * kKvH * kHD), hv(hk.size());
    for (auto& x : hk) x = common::f32_to_bf16(nd(rng));
    for (auto& x : hv) x = common::f32_to_bf16(nd(rng));
    Rig r(ctx, L, false, hk, hv);
    for (uint32_t M : {2u, 3u, 4u, 8u}) {
      double worst = 2;
      uint32_t wp = 0, wh = 0, wm = 0, bad = 0;
      for (uint32_t pos = 0; pos + M <= L - 8; ++pos) {
        std::vector<float> q(size_t(M) * kRow), g(q.size());
        for (auto& x : q) x = nd(rng);
        for (auto& x : g) x = nd(rng);
        r.put(r.aq.ptr(), q.data(), q.size() * 4);
        r.put(r.ag.ptr(), g.data(), g.size() * 4);
        const std::vector<uint16_t> o = r.run(pos, M, true);
        for (uint32_t m = 0; m < M; ++m) {
          r.put(r.aq.ptr(), &q[size_t(m) * kRow], kRow * 4);
          r.put(r.ag.ptr(), &g[size_t(m) * kRow], kRow * 4);
          const std::vector<uint16_t> one = r.run(pos + m, 1, false);
          for (uint32_t h = 0; h < kQH; ++h) {
            double ab = 0;
            const double c = cos_rows(&o[size_t(m) * kRow + h * kHD], &one[h * kHD], &ab);
            bad += c < 0.9999;
            if (c < worst) worst = c, wp = pos, wh = h, wm = m;
          }
        }
      }
      ok = ok && worst >= 0.9999;   // against v2's M = 1 step, which carries its own rounding (N1 is the fp64 bar)
      std::printf("SWEEP M=%u: worst cos %.8f at pos %u head %u row %u; %u (pos, head, row) below 0.9999\n", M, worst,
                  wp, wh, wm, bad);
    }
    std::printf("attn_verify_dpas_test --sweep: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }
  // ---- part D ----
  if (argc < 2) {
    std::printf("D skipped: no dump dir given\n");
  } else {
    const std::string dir = argv[1];
    std::ifstream kf(dir + "/kv_k.bin", std::ios::binary | std::ios::ate);
    if (!kf) {
      std::printf("D skipped: no %s/kv_k.bin\n", dir.c_str());
    } else {
      const size_t rows = size_t(kf.tellg()) / (kKvH * kHD * 2);
      const std::vector<uint16_t> hk = read_file<uint16_t>(dir + "/kv_k.bin", rows * kKvH * kHD);
      const std::vector<uint16_t> hv = read_file<uint16_t>(dir + "/kv_v.bin", rows * kKvH * kHD);
      for (bool kv8 : {false, true}) {
        Rig r(ctx, 131072, kv8, hk, hv);
        const std::vector<uint32_t> Ms = kv8 ? std::vector<uint32_t>{2, 3, 4} : std::vector<uint32_t>{2, 3, 4, 5, 6, 7, 8};
        for (uint32_t d : {512u, 4096u, 32768u, 130816u}) {
          std::vector<float> q = read_file<float>(dir + "/q_" + std::to_string(d) + ".bin", size_t(8) * kRow);
          const std::vector<float> g = read_file<float>(dir + "/g_" + std::to_string(d) + ".bin", size_t(8) * kRow);
          if (q.empty() || g.empty() || rows < size_t(d) + 8) {
            std::printf("D %u: not in the dump, skipped\n", d);
            continue;
          }
          if (kv8)
            for (uint32_t i = 0; i < 8 * kQH; ++i) {
              float y[kHD];
              common::kv8::rotate(&q[size_t(i) * kHD], y);
              std::memcpy(&q[size_t(i) * kHD], y, sizeof y);
            }
          Worst w;
          for (uint32_t M : Ms) check(r, d, M, q, g, w);
          uint32_t done = 0;
          const uint32_t same = prop_r(r, d, Ms, q.data(), g.data(), 100, rng, &done);
          const bool pass = w.n1 >= kN1 && w.n2b >= kN1 && w.rep && same == done;
          ok = ok && pass;
          std::printf("D %s depth %6u: N1 worst %.8f (v2 %.8f)  N2b worst %.8f (max abs %.3g)  rep %s  N2a %u/%u  "
                      "bar %.8f%s\n",
                      kv8 ? "int8" : "bf16", d, w.n1, w.n1_v2, w.n2b, w.abs_n2b, w.rep ? "yes" : "NO", same, done,
                      kN1, pass ? "" : "  <-- FAIL");
        }
      }
    }
  }
  std::printf("attn_verify_dpas_test: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
