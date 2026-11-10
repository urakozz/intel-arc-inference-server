// Spec 17b (plan 17b's review, Task 2): tp.cl's crossing on the two cards against prep_res_fold - no checkpoint.
//
//   slices  one-card partials P [4][1][5120] (fp32, random) and a residual row; card r holds slices 2 r, 2 r + 1. Each
//           card's list: the residual restored, tp_send (its two slices into the other card's landing), tp_res_fold.
//           Both cards' residual and sums of squares must be BITWISE prep_res_fold_M1_K5120_SP4_G20's on card 0 over
//           all of P - after every one of 300 replays (the counters advance in the kernels; slots alternate).
//   folded  card r holds four slices of its own; both cards' residuals bitwise equal and equal to the host's
//           rne(r + rne((0 + a) + b)), a / b the cards' slice sums in slice order.
// Exits 77 (SKIP) with fewer than two GPUs or without peer access both ways.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/tp_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/tp_protocol.h"

namespace {
using runtime::tp::Layout;
constexpr uint32_t K = 5120, G = 20, kReplays = 300;

struct Side {
  Side(l0::Context& c, const Layout& L, uint32_t slices)
      : ctx(c),
        partials(c, l0::MemKind::Device, size_t(slices) * K * 4),
        resid0(c, l0::MemKind::Device, K * 2),
        resid(c, l0::MemKind::Device, K * 2),
        sumsq(c, l0::MemKind::Device, G * 4),
        land(c, l0::MemKind::Device, L.total_bytes(), 4096),
        send_ctr(c, l0::MemKind::Device, L.counter_bytes()),
        fold_ctr(c, l0::MemKind::Device, L.counter_bytes()),
        state(c, l0::MemKind::Shared, runtime::tp::kStateWords * 4),
        own(c, l0::MemKind::Device, size_t(runtime::tp::kMRows) * K * 4),
        queue(c),
        fence(queue),
        imm(l0::CmdList::immediate(c)) {
    imm.fill(land.ptr(), 0u, land.size());
    imm.fill(send_ctr.ptr(), 0u, send_ctr.size());
    imm.fill(fold_ctr.ptr(), 0u, fold_ctr.size());
    std::memset(state.ptr(), 0, state.size());
  }
  l0::Context& ctx;
  l0::Mem partials, resid0, resid, sumsq, land, send_ctr, fold_ctr, state, own;
  l0::Queue queue;
  l0::Fence fence;
  l0::CmdList imm;
  std::unique_ptr<l0::Module> mod;
  std::unique_ptr<l0::Kernel> send, fold;
  std::unique_ptr<l0::CmdList> list;
};

void build(Side& s, Side& peer, uint32_t rank, uint32_t s_loc, bool folded) {
  s.mod = std::make_unique<l0::Module>(s.ctx, kernels::path(kernels::tp_cross_variant(1, K, G, s_loc, folded, rank)));
  s.send = std::make_unique<l0::Kernel>(*s.mod, "tp_send");
  s.fold = std::make_unique<l0::Kernel>(*s.mod, "tp_res_fold");
  s.send->group_size(256);
  s.fold->group_size(256);
  s.send->arg_ptr(0, s.partials.ptr());
  s.send->arg_ptr(1, peer.land.ptr());
  s.send->arg_ptr(2, s.send_ctr.ptr());
  s.send->arg_ptr(3, s.own.ptr());
  s.fold->arg_ptr(0, folded ? s.own.ptr() : s.partials.ptr());
  s.fold->arg_ptr(1, s.land.ptr());
  s.fold->arg_ptr(2, s.resid.ptr());
  s.fold->arg_ptr(3, s.sumsq.ptr());
  s.fold->arg_ptr(4, s.fold_ctr.ptr());
  s.fold->arg_ptr(5, s.state.ptr());
  s.fold->arg(6, uint32_t(1u << 22));
  s.list = std::make_unique<l0::CmdList>(l0::CmdList::regular(s.ctx));
  s.list->copy(s.resid.ptr(), s.resid0.ptr(), K * 2);
  s.list->launch(*s.send, G, 1);
  s.list->launch(*s.fold, G, 1);
  s.list->close();
}

template <class T>
std::vector<T> read(Side& s, const l0::Mem& m, size_t n) {
  std::vector<T> v(n);
  s.imm.copy(v.data(), m.ptr(), n * sizeof(T));
  return v;
}

void replay(Side& a, Side& b) {
  for (uint32_t i = 0; i < kReplays; ++i) {
    a.queue.execute(*a.list, &a.fence);
    b.queue.execute(*b.list, &b.fence);
    CHECK(a.fence.wait_for(uint64_t(20) * 1000000000ull));
    CHECK(b.fence.wait_for(uint64_t(20) * 1000000000ull));
  }
  for (Side* s : {&a, &b}) CHECK_EQ(s->state.as<uint32_t>()[runtime::tp::kStatus], 0u);
}
}  // namespace

int main() {
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two GPUs needed\n");
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0)) {
    std::printf("SKIP: no peer access both ways\n");
    return 77;
  }
  std::mt19937 rng(17);
  std::normal_distribution<float> nd(0.f, 0.05f);
  std::vector<uint16_t> resid(K);
  for (uint16_t& r : resid) r = common::f32_to_bf16(nd(rng) * 20.f);

  // --- slices: bitwise prep_res_fold over the one-card slices --------------------------------------------------------
  {
    std::vector<float> P(size_t(4) * K);
    for (float& p : P) p = nd(rng);
    Side ref(d0, Layout{K, 2, G}, 4);
    ref.imm.copy(ref.partials.ptr(), P.data(), P.size() * 4);
    ref.imm.copy(ref.resid.ptr(), resid.data(), K * 2);
    {
      l0::Module m(d0, kernels::path(kernels::prep_res_fold_variant(1, K, 4, G)));
      l0::Kernel k(m, "prep_res_fold");
      k.group_size(256);
      k.arg_ptr(0, ref.partials.ptr());
      k.arg_ptr(1, ref.resid.ptr());
      k.arg_ptr(2, ref.sumsq.ptr());
      l0::CmdList l = l0::CmdList::regular(d0);
      l.launch(k, G, 1);
      l.close();
      ref.queue.execute(l, &ref.fence);
      ref.fence.wait();
    }
    const std::vector<uint16_t> want_r = read<uint16_t>(ref, ref.resid, K);
    const std::vector<float> want_s = read<float>(ref, ref.sumsq, G);
    const Layout L{K, 2, G};
    Side a(d0, L, 2), b(d1, L, 2);
    a.imm.copy(a.partials.ptr(), P.data(), size_t(2) * K * 4);
    b.imm.copy(b.partials.ptr(), P.data() + size_t(2) * K, size_t(2) * K * 4);
    for (Side* s : {&a, &b}) s->imm.copy(s->resid0.ptr(), resid.data(), K * 2);
    build(a, b, 0, 2, false);
    build(b, a, 1, 2, false);
    replay(a, b);
    for (Side* s : {&a, &b}) {
      CHECK(read<uint16_t>(*s, s->resid, K) == want_r);
      const std::vector<float> got = read<float>(*s, s->sumsq, G);
      CHECK(std::memcmp(got.data(), want_s.data(), G * 4) == 0);
    }
    std::printf("slices: %u replays, both cards' residual and sums of squares bitwise prep_res_fold SP4's\n", kReplays);
  }
  // --- folded: both cards equal, the host's two-row sum ------------------------------------------------------------
  {
    std::vector<float> P(size_t(8) * K);
    for (float& p : P) p = nd(rng);
    const Layout L{K, 1, G};
    Side a(d0, L, 4), b(d1, L, 4);
    a.imm.copy(a.partials.ptr(), P.data(), size_t(4) * K * 4);
    b.imm.copy(b.partials.ptr(), P.data() + size_t(4) * K, size_t(4) * K * 4);
    for (Side* s : {&a, &b}) s->imm.copy(s->resid0.ptr(), resid.data(), K * 2);
    build(a, b, 0, 4, true);
    build(b, a, 1, 4, true);
    replay(a, b);
    const std::vector<uint16_t> ra = read<uint16_t>(a, a.resid, K), rb = read<uint16_t>(b, b.resid, K);
    CHECK(ra == rb);
    CHECK(read<float>(a, a.sumsq, G) == read<float>(b, b.sumsq, G));
    for (uint32_t k = 0; k < K; ++k) {
      float x = 0.f, y = 0.f;
      for (uint32_t s = 0; s < 4; ++s) x += P[size_t(s) * K + k];
      for (uint32_t s = 4; s < 8; ++s) y += P[size_t(s) * K + k];
      const float acc = (0.f + x) + y;
      const uint16_t want = common::f32_to_bf16(common::bf16_to_f32(resid[k]) +
                                                common::bf16_to_f32(common::f32_to_bf16(acc)));
      CHECK_EQ(ra[k], want);
    }
    std::printf("folded: %u replays, both cards bitwise each other and the host's card-0-then-card-1 sum\n", kReplays);
  }
  std::printf("tp_kernels_test: OK\n");
  return 0;
}
