// attn_v2_test - spec 10 A1 for decode attention v2 (src/kernels/attn_v2.cl), plan 10b
// Task 1, against v1 (attn.cl's attn_decode + attn_reduce, M = 1).
//
//   attn_v2_test [<dump dir>]
//
// Part S (synthetic, always): random bf16 KV of exactly 4096 rows and random q / gate.
//   For every pos in a list that covers depth 0 -> 1, tiny pos, the 64-block edges, the
//   stride's first step (2048 -> 2049 keys, Review Focus 1) and the end of the cache
//   (pos + M = 4096, Review Focus 2):
//     S1  v2 M = 1 against v1 M = 1: per (q head) cosine >= 0.99999, finite, max abs
//         recorded; two runs bitwise identical;
//     S2  v2 at M = 2..4 from pos (when pos + M <= 4096): row m BITWISE equal to v2 M = 1
//         at pos + m with q / gate row m (spec 8's M2 property: the stride is a function
//         of the row's own key count, never of n_active).
// Part D (real, when <dump dir> is given and holds plan 10a's capture): FA layer 15's KV
//   and the q / gate / production attn_out of 4 plain steps from each depth in
//   {4096, 32768, 65536, 130816}; KV allocated at 131072 rows.
//     D0  v1 M = 1 at d + m (q row m) equals production's dumped attn_out row m bitwise
//         (the harness is the production kernel's);
//     D1  v2 M = 1 at d + m against it: cosine >= 0.99999 per (q head, m), max abs;
//         repeatable;
//     D2  v2 at M = 2..4 from d: row m bitwise equal to D1's row m.
// With no dump dir, part D is skipped (printed), and the test still passes on part S.
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

namespace {
constexpr uint32_t kQH = 24, kKvH = 4, kHD = 256, kRow = kQH * kHD;   // 6144
constexpr uint32_t kWG = 256, kPart = 258;
constexpr uint32_t kTgt = runtime::DecodeScratch::kAttnV2Blocks;
constexpr uint32_t kBlock = runtime::DecodeScratch::kAttnBlock;

float bf16f(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// One engine-shaped rig: KV of `max_len` rows, q / gate [4][6144], v1 at M = 1 for this
// max_len, v2 at M = 1..4.
struct Rig {
  l0::Context& ctx;
  uint32_t max_len;
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem ctrl{ctx, l0::MemKind::Shared, sizeof(runtime::Control)};
  l0::Mem kk, vv, aq, ag, part1, part2, out;
  l0::Module m1dec, m1red;
  std::vector<std::unique_ptr<l0::Module>> m2;
  Rig(l0::Context& c, uint32_t L)
      : ctx(c), max_len(L),
        kk(c, l0::MemKind::Device, size_t(L) * kKvH * kHD * 2),
        vv(c, l0::MemKind::Device, size_t(L) * kKvH * kHD * 2),
        aq(c, l0::MemKind::Device, size_t(4) * kRow * 4),
        ag(c, l0::MemKind::Device, size_t(4) * kRow * 4),
        part1(c, l0::MemKind::Device, size_t(kQH) * (L / kBlock) * kPart * 4),
        part2(c, l0::MemKind::Device, size_t(kQH) * kTgt * 4 * kPart * 4),
        out(c, l0::MemKind::Device, size_t(4) * kRow * 2),
        m1dec(c, kernels::path(kernels::attn_decode_variant(1, L, kBlock))),
        m1red(c, kernels::path(kernels::attn_reduce_variant(1, L, kBlock))) {
    for (uint32_t M = 1; M <= 4; ++M)
      m2.push_back(std::make_unique<l0::Module>(c, kernels::path(kernels::attn_v2_variant(M, kTgt))));
  }
  void upload(void* dev, const void* host, size_t bytes) {
    constexpr size_t kChunk = 64ul << 20;
    for (size_t off = 0; off < bytes; off += kChunk)
      imm.copy(static_cast<char*>(dev) + off, static_cast<const char*>(host) + off,
               std::min(kChunk, bytes - off));
  }
  // q / gate rows [row0, row0 + M) of the host arrays into device rows [0, M).
  void set_q(const std::vector<float>& qh, const std::vector<float>& gh, uint32_t row0, uint32_t M) {
    imm.copy(aq.ptr(), qh.data() + size_t(row0) * kRow, size_t(M) * kRow * 4);
    imm.copy(ag.ptr(), gh.data() + size_t(row0) * kRow, size_t(M) * kRow * 4);
  }
  // v2 == false: v1 at M = 1. Returns attn_out [M][6144].
  std::vector<uint16_t> run(bool v2, uint32_t M, uint32_t pos) {
    auto* c = ctrl.as<runtime::Control>();
    *c = runtime::Control{};
    c->pos = pos;
    c->n_active = M;
    l0::Kernel dec(v2 ? *m2[M - 1] : m1dec, v2 ? "attn_decode_v2" : "attn_decode");
    l0::Kernel red(v2 ? *m2[M - 1] : m1red, v2 ? "attn_reduce_v2" : "attn_reduce");
    l0::Mem& part = v2 ? part2 : part1;
    dec.group_size(kWG);
    dec.arg_ptr(0, ctrl.ptr());
    dec.arg_ptr(1, aq.ptr());
    dec.arg_ptr(2, kk.ptr());
    dec.arg_ptr(3, vv.ptr());
    dec.arg_ptr(4, part.ptr());
    red.group_size(kWG);
    red.arg_ptr(0, ctrl.ptr());
    red.arg_ptr(1, part.ptr());
    red.arg_ptr(2, ag.ptr());
    red.arg_ptr(3, out.ptr());
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.fill(part.ptr(), 0x7FC00000u, part.size());   // NaN canary: a partial read unwritten shows
    list.fill(out.ptr(), 0u, out.size());
    list.launch(dec, kKvH, v2 ? kTgt : max_len / kBlock);
    list.launch(red, kQH, M);
    list.close();
    q.execute(list, &fence);
    fence.wait();
    std::vector<uint16_t> o(size_t(M) * kRow);
    imm.copy(o.data(), out.ptr(), o.size() * 2);
    return o;
  }
};

struct Cmp {
  double worst = 2.0, maxabs = 0.0;
  bool finite = true;
};
Cmp compare(const uint16_t* a, const uint16_t* b) {   // one row [6144], per q head
  Cmp r;
  for (uint32_t h = 0; h < kQH; ++h) {
    double xy = 0, xx = 0, yy = 0;
    for (uint32_t i = 0; i < kHD; ++i) {
      const double x = bf16f(a[h * kHD + i]), y = bf16f(b[h * kHD + i]);
      if (!std::isfinite(x)) r.finite = false;
      xy += x * y;
      xx += x * x;
      yy += y * y;
      r.maxabs = std::max(r.maxabs, std::fabs(x - y));
    }
    const double cs = (xx == 0 && yy == 0) ? 1.0 : xy / std::sqrt(xx * yy);
    r.worst = std::min(r.worst, cs);
  }
  return r;
}

// v2 at M = 1 at pos + m (q row m) for m < 4 (where it fits), then the M2 property.
bool check_depth(Rig& r, const std::vector<float>& qh, const std::vector<float>& gh, uint32_t pos,
                 const std::vector<uint16_t>* prod, const char* tag) {
  bool ok = true;
  std::vector<std::vector<uint16_t>> one(4);
  double worst = 2.0, maxabs = 0.0;
  bool rep = true, finite = true, prod_ok = true;
  const uint32_t rows = std::min<uint32_t>(4, r.max_len - pos);
  for (uint32_t m = 0; m < rows; ++m) {
    r.set_q(qh, gh, m, 1);
    const auto v1 = r.run(false, 1, pos + m);
    if (prod) prod_ok &= std::memcmp(v1.data(), prod->data() + size_t(m) * kRow, kRow * 2) == 0;
    one[m] = r.run(true, 1, pos + m);
    rep &= r.run(true, 1, pos + m) == one[m];
    const Cmp c = compare(one[m].data(), v1.data());
    worst = std::min(worst, c.worst);
    maxabs = std::max(maxabs, c.maxabs);
    finite &= c.finite;
  }
  std::string m2s;
  bool m2ok = true;
  for (uint32_t M = 2; M <= 4 && M <= rows; ++M) {
    r.set_q(qh, gh, 0, M);
    const auto o = r.run(true, M, pos);
    bool same = true;
    for (uint32_t m = 0; m < M; ++m)
      same &= std::memcmp(o.data() + size_t(m) * kRow, one[m].data(), kRow * 2) == 0;
    m2s += " M" + std::to_string(M) + (same ? " =" : " DIFF");
    m2ok &= same;
  }
  const bool pass = finite && worst >= 0.99999 && rep && m2ok && prod_ok;
  std::printf("  %s pos %6u: v2 vs v1 worst cos %.9f max abs %.4g%s, repeatable %s, rows vs "
              "M=1:%s%s%s\n",
              tag, pos, worst, maxabs, finite ? "" : " NONFINITE", rep ? "yes" : "NO", m2s.c_str(),
              prod ? (prod_ok ? ", v1 = production dump" : ", v1 != PRODUCTION DUMP") : "",
              pass ? "" : "  <-- FAIL");
  ok &= pass;
  return ok;
}

template <class T>
bool read_file(const std::string& p, std::vector<T>& v) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
  return bool(f);
}
}  // namespace

int main(int argc, char** argv) {
  l0::Context ctx(0);
  bool ok = true;

  // ---- Part S -------------------------------------------------------------------
  {
    constexpr uint32_t L = 4096;
    Rig r(ctx, L);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-2.f, 2.f);
    std::normal_distribution<float> n(0.f, 1.f);
    std::vector<uint16_t> kv(size_t(L) * kKvH * kHD);
    for (auto& x : kv) x = to_bf16(u(rng));
    r.upload(r.kk.ptr(), kv.data(), kv.size() * 2);
    for (auto& x : kv) x = to_bf16(u(rng));
    r.upload(r.vv.ptr(), kv.data(), kv.size() * 2);
    std::vector<float> qh(size_t(4) * kRow), gh(size_t(4) * kRow);
    for (auto& x : qh) x = n(rng);
    for (auto& x : gh) x = n(rng);
    std::printf("Part S: random KV, max_len %u, ppw TGT %u\n", L, kTgt);
    for (uint32_t pos : {0u, 1u, 2u, 13u, 15u, 16u, 17u, 60u, 62u, 63u, 64u, 65u, 127u, 128u,
                         1000u, 2043u, 2044u, 2045u, 2046u, 2047u, 2048u, 2049u, 3000u, 4092u,
                         4093u, 4094u, 4095u})
      ok &= check_depth(r, qh, gh, pos, nullptr, "S");
  }

  // ---- Part D -------------------------------------------------------------------
  const std::string dir = argc > 1 ? argv[1] : "";
  std::vector<uint32_t> depths = {4096, 32768, 65536, 130816};
  if (dir.empty() || !std::ifstream(dir + "/kv_k.bin").good()) {
    std::printf("Part D: SKIPPED (no dump dir; plan 10a's capture, e.g. ~/spec10a-dump)\n");
  } else {
    constexpr uint32_t L = 131072;
    Rig r(ctx, L);
    const size_t rows = 130816 + 4, elems = rows * kKvH * kHD;
    std::vector<uint16_t> kv(elems);
    CHECK(read_file(dir + "/kv_k.bin", kv));
    std::vector<uint16_t> zero(size_t(L) * kKvH * kHD - elems, 0);
    r.upload(r.kk.ptr(), kv.data(), elems * 2);
    r.upload(static_cast<uint16_t*>(r.kk.ptr()) + elems, zero.data(), zero.size() * 2);
    CHECK(read_file(dir + "/kv_v.bin", kv));
    r.upload(r.vv.ptr(), kv.data(), elems * 2);
    r.upload(static_cast<uint16_t*>(r.vv.ptr()) + elems, zero.data(), zero.size() * 2);
    std::printf("Part D: plan 10a's capture (%s), FA layer 15, max_len %u\n", dir.c_str(), L);
    for (uint32_t d : depths) {
      std::vector<float> qh(size_t(4) * kRow), gh(size_t(4) * kRow);
      std::vector<uint16_t> prod(size_t(4) * kRow);
      const std::string s = "_" + std::to_string(d) + ".bin";
      CHECK(read_file(dir + "/q" + s, qh));
      CHECK(read_file(dir + "/g" + s, gh));
      CHECK(read_file(dir + "/o" + s, prod));
      ok &= check_depth(r, qh, gh, d, &prod, "D");
    }
  }

  CHECK(ok);
  std::puts("attn_v2_test OK");
  return 0;
}
