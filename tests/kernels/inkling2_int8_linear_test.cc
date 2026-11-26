// Plan 41e: the int8 prompt pass's linears (ink_int8.cl + pf_gemm_i8) against the fp32 product of the same int8
// weights and bf16 rows: every row's cosine >= 0.9995 at both Ks, rows past the chunk not stored; wr's N 384 stored
// as 512 zero-padded columns (the GEMM's 256-column work-groups), columns past N not stored.
#include <bit>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"

int main() {
  l0::Context ctx(0);
  auto list = l0::CmdList::immediate(ctx);
  const auto upload = [&](const auto& v) {
    l0::Mem m(ctx, l0::MemKind::Device, v.size() * sizeof(v[0]));
    list.copy(m.ptr(), v.data(), m.size());
    return m;
  };
  std::mt19937 rng(7);
  constexpr uint32_t M = 300, Mp = 512;
  for (const auto& [K, N] : {std::pair<uint32_t, uint32_t>{2304, 1024}, {3072, 2304}, {2304, 384}}) {
    const uint32_t columns = (N + 255) / 256 * 256;
    std::vector<int8_t> w(size_t(N) * K);   // tile layout [N / 16][K][16]
    std::vector<float> sc(N);
    std::vector<uint16_t> x(size_t(M) * K);
    std::normal_distribution<float> g(0.f, 1.f);
    for (auto& v : w) v = int8_t(int(rng() % 255) - 127);
    for (auto& v : sc) v = std::ldexp(1.f + float(rng() % 64) / 64.f, -9);
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t k = 0; k < K; ++k)   // a few outlier channels, as normed rows have
        x[size_t(r) * K + k] = common::f32_to_bf16(g(rng) * (k % 97 == 3 ? 20.f : 1.f));
    auto wd = upload(w), sd = upload(sc), xd = upload(x);
    l0::Mem w8(ctx, l0::MemKind::Device, size_t(K) * columns), ws(ctx, l0::MemKind::Device, columns * 4);
    list.fill(w8.ptr(), 0, w8.size());
    list.fill(ws.ptr(), 0, ws.size());
    l0::Mem xq(ctx, l0::MemKind::Device, size_t(Mp) * K), xs(ctx, l0::MemKind::Device, Mp * 4);
    std::vector<float> y(size_t(Mp) * N, 1234.f);
    auto yd = upload(y);
    l0::Module m(ctx, kernels::path("ink_int8_K" + std::to_string(K)));
    auto rot = m.kernel("ink_rot_weight");
    rot.group_size(16);
    rot.arg_ptr(0, wd.ptr()); rot.arg_ptr(1, sd.ptr()); rot.arg_ptr(2, w8.ptr()); rot.arg_ptr(3, ws.ptr()); rot.arg(4, columns);
    list.launch(rot, N);
    auto quant = m.kernel("ink_quant_rot");
    quant.group_size(16 * K / 256);
    quant.arg_ptr(0, xd.ptr()); quant.arg_ptr(1, xq.ptr()); quant.arg_ptr(2, xs.ptr());
    list.launch(quant, M);
    l0::Module gm(ctx, kernels::path(kernels::pf_gemm_i8_variant(false)));
    auto gemm = gm.kernel("pf_gemm_i8");
    gemm.group_size(512);
    gemm.arg_ptr(0, xq.ptr()); gemm.arg_ptr(1, xs.ptr()); gemm.arg_ptr(2, w8.ptr()); gemm.arg_ptr(3, ws.ptr());
    gemm.arg_ptr(4, yd.ptr());
    gemm.arg(5, M); gemm.arg(6, K); gemm.arg(7, N); gemm.arg(8, K / 2); gemm.arg(9, columns); gemm.arg(10, N);
    list.launch(gemm, Mp / 256, columns / kernels::kPfGemmI8WgN);
    list.copy(y.data(), yd.ptr(), yd.size());
    double worst = 1;
    for (uint32_t r = 0; r < M; ++r) {
      double cross = 0, ng = 0, nw = 0;
      for (uint32_t n = 0; n < N; ++n) {
        double want = 0;
        for (uint32_t k = 0; k < K; ++k)
          want += double(common::bf16_to_f32(x[size_t(r) * K + k])) *
                  double(w[(size_t(n / 16) * K + k) * 16 + n % 16]);
        want *= sc[n];
        const double got = y[size_t(r) * N + n];
        cross += got * want, ng += got * got, nw += want * want;
      }
      worst = std::min(worst, cross / std::sqrt(ng * nw));
    }
    for (size_t i = size_t(M) * N; i < y.size(); ++i) CHECK(y[i] == 1234.f);
    std::printf("K %u N %u: worst row cosine %.6f over %u rows\n", K, N, worst, M);
    CHECK(worst >= .9995);
  }
  std::puts("Inkling int8 prompt linears match the fp32 product");
}
