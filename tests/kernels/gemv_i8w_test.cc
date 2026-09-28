// gemv_i8w (spec 9 §3, plan 9b Task 2): the int8 `lm_head` GEMV at its real shape,
// K 5120 x N 248320, against an fp64 host reference on random int8 weights, random
// fp32 row scales and random bf16 activations - at M = 1 (decode, prefill's step_head,
// the MTP draft) and M = 4 (the MTP verify list at K = 3).
//
// **The bar is a relative error of 1e-5 per output**, relative to the column's own
// magnitude sum `s * sum_k |q_k x_k|` rather than to |out|: an fp32 chain of 5120
// products can cancel to near zero, and a bar on |out| would then grade the
// cancellation, not the kernel. fp32 accumulation over K = 5120 sits near
// sqrt(K) * 2^-24 = 4e-6 of that sum.
//
// It also times the launch (weights cycled through two device copies, so L2 holds
// none of it) and prints GB/s against the bytes read: 1 271 398 400 weight + 993 280
// scale bytes (Review Focus 1). The in-situ number is the engine profile's.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "check.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "kernels/kernels.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/lm_head_int8.h"

namespace {
constexpr uint32_t K = 5120, N = 248320;

struct Case {
  std::vector<int8_t> q_rm;      // [N][K]
  std::vector<int8_t> q_tiled;   // loader::int8_tiled_index
  std::vector<float> s;          // [N]
};

Case make_weights() {
  Case c;
  c.q_rm.resize(size_t(N) * K);
  c.q_tiled.resize(c.q_rm.size());
  c.s.resize(N);
  std::mt19937 g(99);
  std::uniform_int_distribution<int> qd(-127, 127);
  std::uniform_real_distribution<float> sd(1e-4f, 2e-3f);
  for (uint32_t n = 0; n < N; ++n) {
    c.s[n] = sd(g);
    for (uint32_t k = 0; k < K; ++k) {
      const int8_t v = int8_t(qd(g));
      c.q_rm[size_t(n) * K + k] = v;
      c.q_tiled[loader::int8_tiled_index(K, k, n)] = v;
    }
  }
  return c;
}

// Returns the worst relative error over all M x N outputs.
double check_against_fp64(const Case& c, const std::vector<uint16_t>& x, uint32_t M,
                          const std::vector<float>& out) {
  double worst = 0;
  for (uint32_t m = 0; m < M; ++m) {
    std::vector<double> xf(K);
    for (uint32_t k = 0; k < K; ++k) xf[k] = common::bf16_to_f32(x[size_t(m) * K + k]);
    for (uint32_t n = 0; n < N; ++n) {
      const int8_t* q = c.q_rm.data() + size_t(n) * K;
      double dot = 0, mag = 0;
      for (uint32_t k = 0; k < K; ++k) {
        const double t = double(q[k]) * xf[k];
        dot += t;
        mag += std::fabs(t);
      }
      const double ref = dot * c.s[n], scale = mag * c.s[n];
      const double e = scale > 0 ? std::fabs(double(out[size_t(m) * N + n]) - ref) / scale : 0.0;
      worst = std::max(worst, e);
    }
  }
  return worst;
}
}  // namespace

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const Case c = make_weights();
  const size_t wbytes = c.q_tiled.size();
  // Two weight copies for the timing loop (each 1.27 GB, far past L2).
  std::vector<l0::Mem> wbufs;
  for (int i = 0; i < 2; ++i) {
    wbufs.emplace_back(ctx, l0::MemKind::Device, wbytes);
    imm.copy(wbufs.back().ptr(), c.q_tiled.data(), wbytes);
  }
  l0::Mem sbuf(ctx, l0::MemKind::Device, size_t(N) * 4);
  imm.copy(sbuf.ptr(), c.s.data(), size_t(N) * 4);

  for (uint32_t M : {1u, 4u}) {
    std::vector<uint16_t> x = random_bf16(size_t(M) * K, 11 + M);
    l0::Mem xbuf(ctx, l0::MemKind::Device, x.size() * 2);
    imm.copy(xbuf.ptr(), x.data(), x.size() * 2);
    l0::Mem obuf(ctx, l0::MemKind::Device, size_t(M) * N * 4);
    l0::Module mod(ctx, kernels::path(kernels::gemv_i8w_variant(M, K, N)));
    l0::Kernel k = mod.kernel("gemv_i8w");
    k.group_size(kernels::kGemvI8wCols);
    auto bind = [&](int i) {
      k.arg_ptr(0, wbufs[i].ptr());
      k.arg_ptr(1, sbuf.ptr());
      k.arg_ptr(2, xbuf.ptr());
      k.arg_ptr(3, obuf.ptr());
    };
    std::vector<float> out(size_t(M) * N);
    {
      l0::CmdList list = l0::CmdList::regular(ctx);
      bind(0);
      list.launch(k, N / kernels::kGemvI8wCols);
      list.close();
      q.execute(list, &fence);
      fence.wait();
      imm.copy(out.data(), obuf.ptr(), out.size() * 4);
    }
    const double err = check_against_fp64(c, x, M, out);
    double us = 0;
    {
      const int launches = 20;
      l0::CmdList list = l0::CmdList::regular(ctx);
      for (int i = 0; i < launches; ++i) {
        bind(i % 2);
        list.launch(k, N / kernels::kGemvI8wCols);
      }
      list.close();
      us = time_list(q, fence, list, launches);
    }
    const double bytes = double(wbytes) + double(N) * 4 + double(M) * K * 2;
    std::printf("gemv_i8w M=%u K=%u N=%u: max rel err %.3g (bar 1e-5); %.1f us/launch, %.1f GB/s\n",
                M, K, N, err, us, bytes / us / 1e3);
    CHECK(err <= 1e-5);
  }
  std::puts("gemv_i8w_test OK");
  return 0;
}
