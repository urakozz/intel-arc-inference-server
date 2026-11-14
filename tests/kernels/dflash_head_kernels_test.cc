// Spec 19v2 C3: full-vocabulary top-64 candidates, then exact int8-row reranking to the selector's top 16.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "kernels/dflash_kernels.h"
#include "kernels/dflash_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace kd = kernels::dflash;

struct Env {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  template<class T> l0::Mem up(const std::vector<T>& v) {
    l0::Mem b(ctx, l0::MemKind::Device, v.size() * sizeof(T));
    imm.copy(b.ptr(), v.data(), b.size());
    return b;
  }
  template<class T> std::vector<T> read(const l0::Mem& b, size_t n) {
    std::vector<T> v(n);
    imm.copy(v.data(), b.ptr(), n * sizeof(T));
    return v;
  }
  void run(l0::Kernel& k, unsigned x, unsigned y = 1) {
    auto list = l0::CmdList::regular(ctx);
    list.launch(k, x, y);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }
};

// C: 64 (the int4 head's) or 80 (spec 19 v3d's 2-bit head's default)
void check_candidates(Env& e, unsigned C) {
  constexpr unsigned N = 248320, used = 248077, R = 7, chunks = (N + 4095) / 4096;
  std::mt19937 g(91);
  std::uniform_real_distribution<float> rand(-2, 2);
  std::vector<float> logits(size_t(R) * N);
  for (auto& v : logits) v = rand(g);
  for (unsigned r = 0; r < R; ++r) {
    for (unsigned i = 0; i < (r ? 8u : 80u); ++i) logits[size_t(r) * N + (4093 * i + 7) % used] = 10.0f;   // ties across lanes/chunks
    logits[size_t(r) * N + 5] = NAN;
    for (unsigned i = used; i < N; ++i) logits[size_t(r) * N + i] = INFINITY;
  }
  // Spec 19 v3d: a chunk with more ties at the 64th place than places (row 2, chunk 5: 300 x 3.0, row 2's best after
  // its 8 x 10.0) - the lowest ids among the ties win.
  for (unsigned i = 0; i < 300; ++i) logits[size_t(2) * N + 5 * 4096 + 13 * i] = 3.0f;
  auto in = e.up(logits);
  auto part = e.up(std::vector<float>(size_t(R) * chunks * C * 2));
  auto cand = e.up(std::vector<uint32_t>(size_t(R) * C));
  l0::Module mod(e.ctx, kernels::path(kd::candidate_topk_variant(C)));
  l0::Kernel k1(mod, "df_candidate_stage1"), k2(mod, "df_candidate_stage2");
  k1.group_size(256);
  k1.arg_ptr(0, in.ptr());
  k1.arg_ptr(1, part.ptr());
  k2.group_size(256);
  k2.arg_ptr(0, part.ptr());
  k2.arg_ptr(1, cand.ptr());
  l0::Kernel k1r(mod, "df_candidate_select1");   // spec 19 v3d Task 7: stage 1 by radix select, the same cand
  k1r.group_size(256);
  k1r.arg_ptr(0, in.ptr());
  k1r.arg_ptr(1, part.ptr());
  for (unsigned repeat = 0; repeat < 4; ++repeat) {
    e.run(repeat < 2 ? k1 : k1r, chunks, R);
    e.run(k2, R);
    const auto got = e.read<uint32_t>(cand, size_t(R) * C);
    for (unsigned r = 0; r < R; ++r) {
      const auto want = dfref::topk(logits.data() + size_t(r) * N, used, C, 1, 0);
      for (unsigned c = 0; c < C; ++c) CHECK_EQ(got[r * C + c], want[c].i);
    }
  }
  std::printf("candidate top%u OK: real shape, padding mask, NaN, ties and deterministic replay; the radix-select stage 1 "
              "the same\n", C);
}

void check_rerank(Env& e, unsigned C) {
  constexpr unsigned K = 5120, N = 128, R = 7;
  std::mt19937 g(92);
  std::vector<int8_t> w(size_t(N) * K);
  for (auto& v : w) v = int8_t(int(g() % 255) - 127);
  std::vector<float> scale(N);
  for (auto& v : scale) v = 0.0007f + float(g() % 100) * 0.00001f;
  std::vector<uint16_t> x(size_t(R) * K);
  for (auto& v : x) v = dfref::rne(float(int(g() % 10001) - 5000) / 10000.0f);
  std::fill(x.begin(), x.begin() + K, uint16_t(0));   // all scores tie: lower real id wins
  std::vector<uint32_t> ids(size_t(R) * C);
  for (unsigned r = 0; r < R; ++r) {
    for (unsigned c = 0; c < C; ++c) ids[r * C + c] = (c * 37 + r * 11) % N;
    ids[r * C] = 0x7FFFFFFFu;
    ids[r * C + 1] = 248077;   // padded/masked vocabulary id, never gather it
  }
  auto bw = e.up(w), bs = e.up(scale), bx = e.up(x), bi = e.up(ids);
  auto out = e.up(std::vector<uint32_t>(R * 16));
  auto unary = e.up(std::vector<float>(R * 16));
  l0::Module mod(e.ctx, kernels::path(kd::rerank_variant(C)));
  l0::Kernel k(mod, "df_rerank");
  k.group_size(C * 4);
  k.arg_ptr(0, bw.ptr());
  k.arg_ptr(1, bs.ptr());
  k.arg_ptr(2, bx.ptr());
  k.arg_ptr(3, bi.ptr());
  k.arg_ptr(6, out.ptr());
  k.arg_ptr(7, unary.ptr());
  for (float cap : {0.0f, 30.0f}) {
    k.arg(4, 1.5f);
    k.arg(5, cap);
    e.run(k, R);
    const auto got = e.read<uint32_t>(out, R * 16);
    const auto vals = e.read<float>(unary, R * 16);
    for (unsigned r = 0; r < R; ++r) {
      const auto want = dfref::rerank_candidates(w.data(), scale.data(), K, N, x.data() + size_t(r) * K,
                                                ids.data() + r * C, 1.5f, cap, C);
      for (unsigned c = 0; c < 16; ++c) {
        CHECK_EQ(got[r * 16 + c], want[c].i);
        CHECK(vals[r * 16 + c] == want[c].v ||
              std::fabs(vals[r * 16 + c] - want[c].v) <= 4e-7f * std::fabs(want[c].v));
      }
    }
  }
  std::printf("exact int8 rerank OK: all %u candidates, real K, masked ids, ties, scaling and softcap\n", C);
}

int main() {
  Env e;
  for (unsigned C : {64u, 80u}) {
    check_candidates(e, C);
    check_rerank(e, C);
  }
  std::puts("dflash_head_kernels_test: PASS");
}
