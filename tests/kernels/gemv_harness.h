#pragma once
// Shared by gemv_test and probe_gemv: upload, launch, time, read back.
// Timing: `timed_launches` launches are recorded in ONE regular list cycling
// through NB identical copies of the weights at different addresses, so that
// consecutive launches miss the 24 MB L2 (doc 01) and the number is DRAM
// bandwidth, not cache bandwidth. us_per_launch = list time / launches.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include <chrono>

struct GemvCase { uint32_t M, K, N, S, L; };
struct GemvResult { std::vector<float> out; double us_per_launch = 0; size_t weight_bytes = 0; };

inline double max_abs_err(const std::vector<float>& got, const std::vector<float>& ref) {
  double e = 0;
  for (size_t i = 0; i < ref.size(); ++i) e = std::max(e, double(std::fabs(got[i] - ref[i])));
  return e;
}
inline double tol_for(const std::vector<float>& ref) {
  double mx = 0;
  for (float v : ref) mx = std::max(mx, double(std::fabs(v)));
  return 1e-4 * mx + 1e-5;
}

inline GemvResult run_gemv(l0::Context& ctx, l0::Queue& q, l0::Fence& fence,
                           const common::Int4Gptq& w, const std::vector<uint16_t>& x,
                           GemvCase c, int timed_launches) {
  GemvResult r;
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  // Weight copies: enough that one cycle through them exceeds 3x L2 (72 MB).
  std::vector<uint32_t> tiled;
  const uint32_t* wsrc = w.qweight.data();
  size_t wbytes = w.qweight.size() * 4;
  if (c.L == 1) { tiled = w.tiled(); wsrc = tiled.data(); wbytes = tiled.size() * 4; }
  r.weight_bytes = w.bytes();
  const int NB = timed_launches ? std::max(2, int((72u << 20) / wbytes) + 1) : 1;

  std::vector<l0::Mem> wbufs, sbufs;
  wbufs.reserve(NB);
  sbufs.reserve(NB);
  for (int i = 0; i < NB; ++i) {
    wbufs.emplace_back(ctx, l0::MemKind::Device, wbytes);
    imm.copy(wbufs.back().ptr(), wsrc, wbytes);
    sbufs.emplace_back(ctx, l0::MemKind::Device, w.scales.size() * 2);
    imm.copy(sbufs.back().ptr(), w.scales.data(), w.scales.size() * 2);
  }
  l0::Mem xbuf(ctx, l0::MemKind::Device, x.size() * 2);
  imm.copy(xbuf.ptr(), x.data(), x.size() * 2);
  const size_t out_n = size_t(c.S) * c.M * c.N;
  l0::Mem obuf(ctx, l0::MemKind::Device, out_n * 4);

  l0::Module mod(ctx, kernels::path(kernels::gemv_variant(c.M, c.K, c.N, c.S, c.L)));
  l0::Kernel k = mod.kernel("gemv");
  k.group_size(64);
  auto bind = [&](int i) {
    k.arg_ptr(0, wbufs[i].ptr());
    k.arg_ptr(1, sbufs[i].ptr());
    k.arg_ptr(2, xbuf.ptr());
    k.arg_ptr(3, obuf.ptr());
  };

  // Correctness launch.
  {
    l0::CmdList list = l0::CmdList::regular(ctx);
    bind(0);
    list.launch(k, c.N / 64, c.S);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }
  if (timed_launches > 0) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    for (int i = 0; i < timed_launches; ++i) { bind(i % NB); list.launch(k, c.N / 64, c.S); }
    list.close();
    std::vector<double> us;
    for (int rep = 0; rep < 8; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      q.execute(list, &fence);
      fence.wait();
      double total = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
      if (rep >= 3) us.push_back(total / timed_launches);
    }
    std::sort(us.begin(), us.end());
    r.us_per_launch = us[us.size() / 2];
  }
  std::vector<float> partials(out_n);
  imm.copy(partials.data(), obuf.ptr(), out_n * 4);
  r.out.assign(size_t(c.M) * c.N, 0.f);
  for (uint32_t s = 0; s < c.S; ++s)
    for (size_t i = 0; i < size_t(c.M) * c.N; ++i) r.out[i] += partials[size_t(s) * c.M * c.N + i];
  return r;
}
