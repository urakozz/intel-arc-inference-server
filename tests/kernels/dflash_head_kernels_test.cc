// Spec 19v2 C3: full-vocabulary top-64 candidates, then exact int8-row reranking to the selector's top 16.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
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
  // plan 41k: stage 2 by radix select - the same ids as a set (in no order), ties at the C-th place included
  l0::Kernel k2s(mod, "df_candidate_select2");
  k2s.group_size(256);
  k2s.arg_ptr(0, part.ptr());
  k2s.arg_ptr(1, cand.ptr());
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    e.run(k1r, chunks, R);
    e.run(k2s, R);
    const auto got = e.read<uint32_t>(cand, size_t(R) * C);
    for (unsigned r = 0; r < R; ++r) {
      const auto want = dfref::topk(logits.data() + size_t(r) * N, used, C, 1, 0);
      std::vector<uint32_t> a(got.begin() + r * C, got.begin() + (r + 1) * C), b;
      for (unsigned c = 0; c < C; ++c) b.push_back(want[c].i);
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      CHECK(a == b);
    }
  }
  std::printf("candidate top%u OK: real shape, padding mask, NaN, ties and deterministic replay; the radix-select stage 1 "
              "the same, the radix-select stage 2 the same set\n", C);
}

// The rerank reads the real head's layout (N 248320 rows: k-major tiles at that width, PERF-111, loader::
// int8_tiled_index); rows 0..127 hold random weights, the rest zero. Against the host reference, and byte for byte
// against today's n-major order (the test's own df_rerank_*_NM build over the same rows tiled n-major).
void check_rerank(Env& e, unsigned C) {
  constexpr unsigned K = 5120, N = 248320, ROWS = 128, USED = 248077, R = 7;
  std::mt19937 g(92);
  std::vector<int8_t> w(size_t(N) * K, 0), wn(w.size(), 0);
  for (unsigned n = 0; n < ROWS; ++n)
    for (unsigned k = 0; k < K; ++k) {
      const int8_t v = int8_t(int(g() % 255) - 127);
      w[dfref::i8_index(K, N, k, n)] = v;
      wn[(size_t(n / 16) * (K / 16) + k / 16) * 256 + size_t(k % 16) * 16 + n % 16] = v;
    }
  std::vector<float> scale(N, 0.0f);
  for (unsigned n = 0; n < ROWS; ++n) scale[n] = 0.0007f + float(g() % 100) * 0.00001f;
  std::vector<uint16_t> x(size_t(R) * K);
  for (auto& v : x) v = dfref::rne(float(int(g() % 10001) - 5000) / 10000.0f);
  std::fill(x.begin(), x.begin() + K, uint16_t(0));   // all scores tie: lower real id wins
  std::vector<uint32_t> ids(size_t(R) * C);
  for (unsigned r = 0; r < R; ++r) {
    for (unsigned c = 0; c < C; ++c) ids[r * C + c] = (c * 37 + r * 11) % ROWS;
    ids[r * C] = 0x7FFFFFFFu;
    ids[r * C + 1] = USED;   // padded/masked vocabulary id, never gather it
  }
  auto bs = e.up(scale), bx = e.up(x), bi = e.up(ids);
  auto out = e.up(std::vector<uint32_t>(R * 16));
  auto unary = e.up(std::vector<float>(R * 16));
  // One rerank of build `v` over weights `w`: the candidates and their values.
  auto rerank = [&](const std::string& v, const std::vector<int8_t>& wt, float cap, std::vector<uint32_t>& got,
                    std::vector<float>& vals) {
    l0::Mem bw = e.up(wt);
    l0::Module mod(e.ctx, kernels::path(v));
    l0::Kernel k(mod, "df_rerank");
    k.group_size(C * 4);
    k.arg_ptr(0, bw.ptr());
    k.arg_ptr(1, bs.ptr());
    k.arg_ptr(2, bx.ptr());
    k.arg_ptr(3, bi.ptr());
    k.arg(4, 1.5f);
    k.arg(5, cap);
    k.arg_ptr(6, out.ptr());
    k.arg_ptr(7, unary.ptr());
    e.run(k, R);
    got = e.read<uint32_t>(out, R * 16);
    vals = e.read<float>(unary, R * 16);
  };
  for (float cap : {0.0f, 30.0f}) {
    std::vector<uint32_t> got, got_nm;
    std::vector<float> vals, vals_nm;
    rerank(kd::rerank_variant(C), w, cap, got, vals);
    rerank(kd::rerank_variant(C) + "_NM", wn, cap, got_nm, vals_nm);
    CHECK(got == got_nm);
    CHECK(std::memcmp(vals.data(), vals_nm.data(), vals.size() * 4) == 0);   // bitwise today's n-major order
    for (unsigned r = 0; r < R; ++r) {
      const auto want = dfref::rerank_candidates(w.data(), scale.data(), K, N, USED, x.data() + size_t(r) * K,
                                                ids.data() + r * C, 1.5f, cap, C);
      for (unsigned c = 0; c < 16; ++c) {
        CHECK_EQ(got[r * 16 + c], want[c].i);
        CHECK(vals[r * 16 + c] == want[c].v ||
              std::fabs(vals[r * 16 + c] - want[c].v) <= 4e-7f * std::fabs(want[c].v));
      }
    }
  }
  std::printf("exact int8 rerank OK: all %u candidates, real K, masked ids, ties, scaling and softcap; bitwise "
              "today's n-major order\n", C);
}

int main() {
  Env e;
  for (unsigned C : {64u, 80u}) {
    check_candidates(e, C);
    check_rerank(e, C);
  }
  std::puts("dflash_head_kernels_test: PASS");
}
