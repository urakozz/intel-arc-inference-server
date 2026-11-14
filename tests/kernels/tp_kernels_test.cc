// Spec 17b (plan 17b's review, Task 2): tp.cl's crossing on the two cards against prep_res_fold - no checkpoint.
//
//   slices  one-card partials P [4][1][5120] (fp32, random) and a residual row; card r holds slices 2 r, 2 r + 1. Each
//           card's list: the residual restored, tp_send (its two slices into the other card's landing), tp_res_fold.
//           Both cards' residual and sums of squares must be BITWISE prep_res_fold_M1_K5120_SP4_G20's on card 0 over
//           all of P - after every one of 300 replays (the counters advance in the kernels; slots alternate).
//   folded  card r holds four slices of its own; both cards' residuals bitwise equal and equal to the host's
//           rne(r + rne((0 + a) + b)), a / b the cards' slice sums in slice order.
// Exits 77 (SKIP) with fewer than two GPUs or without peer access both ways.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// `stress <seconds> [pairs]`: how often the 8-row folded crossing goes stale. Each arm is a list a card of `pairs`
// tp_send + tp_res_fold at M = 8 (the verify lists' crossing), replayed on both cards; arms take turns, a list each,
// until `seconds` pass. Arms: the trailers packed (the production binary) or each on its own 64-byte line (_T16),
// back to back or with a 32 MiB copy between pairs (memory traffic, as the linears between real crossings). A failed
// list is printed with its state words and the arm reset (landing, counters, state on both cards).
struct StressArm {
  StressArm(l0::Context& c0, l0::Context& c1, uint32_t stride, bool gap, uint32_t pairs)
      : stride(stride), gap(gap), sides{std::make_unique<SSide>(c0, stride, 0u), std::make_unique<SSide>(c1, stride, 1u)} {
    for (uint32_t r = 0; r < 2; ++r) {
      SSide& s = *sides[r];
      SSide& peer = *sides[1 - r];
      const std::string v = stride == 1 ? kernels::tp_cross_variant(8, K, G, 4, true, r)
                                        : "tp_cross_M8_K5120_G20_S4_F_T16_R" + std::to_string(r);
      s.mod = std::make_unique<l0::Module>(s.ctx, kernels::path(v));
      s.send = std::make_unique<l0::Kernel>(*s.mod, "tp_send");
      s.fold = std::make_unique<l0::Kernel>(*s.mod, "tp_res_fold");
      s.send->group_size(256);
      s.fold->group_size(256);
      s.send->arg_ptr(0, s.partials.ptr());
      s.send->arg_ptr(1, peer.land.ptr());
      s.send->arg_ptr(2, s.send_ctr.ptr());
      s.send->arg_ptr(3, s.own.ptr());
      s.fold->arg_ptr(0, s.own.ptr());
      s.fold->arg_ptr(1, s.land.ptr());
      s.fold->arg_ptr(2, s.resid.ptr());
      s.fold->arg_ptr(3, s.sumsq.ptr());
      s.fold->arg_ptr(4, s.fold_ctr.ptr());
      s.fold->arg_ptr(5, s.state.ptr());
      s.fold->arg(6, uint32_t(1u << 22));
      s.list = std::make_unique<l0::CmdList>(l0::CmdList::regular(s.ctx));
      for (uint32_t i = 0; i < pairs; ++i) {
        s.list->launch(*s.send, G, 8);
        s.list->launch(*s.fold, G, 8);
        if (gap) s.list->copy(s.gap_dst.ptr(), s.gap_src.ptr(), s.gap_src.size());
      }
      s.list->close();
    }
    reset();
  }
  // The landing at the trailer stride: data, trailers, page, flags, page (runtime::tp::Layout's at stride 1).
  static size_t land_bytes(uint32_t stride) {
    const Layout L{K, 1, G};
    const size_t flag_base = runtime::tp::round_up(runtime::tp::kSlots * L.data_words() +
                                                   size_t(runtime::tp::kSlots) * L.chunks() * stride,
                                                   runtime::tp::kPageWords);
    return runtime::tp::round_up(flag_base + size_t(runtime::tp::kSlots) * L.chunks() * runtime::tp::kFlagStride,
                                 runtime::tp::kPageWords) * 4;
  }
  struct SSide {
    SSide(l0::Context& c, uint32_t stride, uint32_t rank)
        : ctx(c),
          partials(c, l0::MemKind::Device, size_t(4) * 8 * K * 4),
          resid(c, l0::MemKind::Device, size_t(8) * K * 2),
          sumsq(c, l0::MemKind::Device, size_t(G) * 8 * 4),
          land(c, l0::MemKind::Device, land_bytes(stride), 4096),
          send_ctr(c, l0::MemKind::Device, size_t(G) * 8 * 4),
          fold_ctr(c, l0::MemKind::Device, size_t(G) * 8 * 4),
          state(c, l0::MemKind::Shared, runtime::tp::kStateWords * 4),
          own(c, l0::MemKind::Device, size_t(8) * K * 4),
          gap_src(c, l0::MemKind::Device, size_t(32) << 20),
          gap_dst(c, l0::MemKind::Device, size_t(32) << 20),
          queue(c),
          fence(queue),
          imm(l0::CmdList::immediate(c)) {
      std::mt19937 rng(rank + 5u);
      std::normal_distribution<float> nd(0.f, 0.05f);
      std::vector<float> p(size_t(4) * 8 * K);
      for (float& x : p) x = nd(rng);
      imm.copy(partials.ptr(), p.data(), p.size() * 4);
      imm.fill(resid.ptr(), 0u, resid.size());
    }
    l0::Context& ctx;
    l0::Mem partials, resid, sumsq, land, send_ctr, fold_ctr, state, own, gap_src, gap_dst;
    l0::Queue queue;
    l0::Fence fence;
    l0::CmdList imm;
    std::unique_ptr<l0::Module> mod;
    std::unique_ptr<l0::Kernel> send, fold;
    std::unique_ptr<l0::CmdList> list;
  };
  void reset() {
    for (auto& s : sides) {
      s->imm.fill(s->land.ptr(), 0u, s->land.size());
      s->imm.fill(s->send_ctr.ptr(), 0u, s->send_ctr.size());
      s->imm.fill(s->fold_ctr.ptr(), 0u, s->fold_ctr.size());
      std::memset(s->state.ptr(), 0, s->state.size());
    }
  }
  // One list on both cards; false (printed, the arm reset) when either card recorded a failure.
  bool run(const char* name, uint64_t list_no) {
    for (auto& s : sides) s->queue.execute(*s->list, &s->fence);
    for (auto& s : sides) CHECK(s->fence.wait_for(uint64_t(60) * 1000000000ull));
    bool ok = true;
    for (uint32_t r = 0; r < 2; ++r) {
      const uint32_t* w = sides[r]->state.as<uint32_t>();
      if (w[runtime::tp::kStatus] == 0u) continue;
      ok = false;
      std::printf("FAIL %s list %llu card %u: status %u (1 timeout, 2 stale), call %u, chunk %u (g %u, m %u), flag %u, "
                  "trailer %u, %u failed chunk(s), late %s\n",
                  name, (unsigned long long)list_no, r, w[0], w[3], w[4], w[4] / 8, w[4] % 8, w[1], w[2], w[5],
                  w[6] == 0u ? "-" : w[6] == ~0u ? "never (lost)" : std::to_string(w[6] - 1u).append(" loads").c_str());
    }
    if (!ok) reset();
    return ok;
  }
  uint32_t stride;
  bool gap;
  std::unique_ptr<SSide> sides[2];
  uint64_t lists = 0, fails = 0;
};

int stress(l0::Context& d0, l0::Context& d1, double seconds, uint32_t pairs) {
  std::vector<std::unique_ptr<StressArm>> arms;
  const char* names[] = {"packed", "packed+gap", "padded", "padded+gap"};
  for (uint32_t stride : {1u, 16u})
    for (bool gap : {false, true}) arms.push_back(std::make_unique<StressArm>(d0, d1, stride, gap, pairs));
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds)
    for (size_t i = 0; i < arms.size(); ++i) {
      StressArm& a = *arms[i];
      if (!a.run(names[i], a.lists)) ++a.fails;
      ++a.lists;
    }
  uint64_t total = 0;
  for (size_t i = 0; i < arms.size(); ++i) {
    std::printf("STRESS %-11s %llu of %llu lists failed (%llu crossings at M = 8, %u a list)\n", names[i],
                (unsigned long long)arms[i]->fails, (unsigned long long)arms[i]->lists,
                (unsigned long long)(arms[i]->lists * pairs), pairs);
    total += arms[i]->fails;
  }
  return total ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
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
  if (argc > 1 && std::string(argv[1]) == "stress")
    return stress(d0, d1, argc > 2 ? std::atof(argv[2]) : 60.0, argc > 3 ? uint32_t(std::atoi(argv[3])) : 64);
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
