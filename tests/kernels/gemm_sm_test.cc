// gemm_sm_test - spec 28 N2 and N3 for the production small-M GEMM (plan 28b Task 1; box, one card).
//
// For every row of kernels::kSmallMTuning (src/kernels/kernels.h) at every M src/kernels/CMakeLists.txt builds
// for it (M = 2..8 and 16; Ornith's rows at 2..4), on random weights
// of the shape and random bf16 rows:
//
//   N2  the output (the S slices summed in s order, as the consumers fold them) against the fp64 reference
//       (tests/kernels/gemm_sm_ref.h): |err| <= 2^-16 x sum_k |w_k x_k| per output, cosine >= 0.9999999 per
//       row; and against gemv / gemv_i8w at the same M (M <= 8, where that binary is built), the same bound.
//       Columns: all of them up to N = 65536; above, one column of every 64-column work-group (a different
//       lane each) plus the first and last 512 - every work-group is checked, not three windows (28a's
//       deferred review item).
//   (spec 19 v3d) a row with the 2D block read is bitwise its plain-read twin at every M, where that twin is built.
//   N3  property R: every row of every M bitwise equal (every slice) to the same row at the largest built M;
//       then `trials` trials (default 1000 a row) of a probe row at a random index among fresh random rows,
//       at a random built M - row bitwise equal to its reference. The output buffer is filled with a NaN
//       pattern before EVERY run (28a's deferred item: a stale result can never pass as a fresh one).
//
// usage: gemm_sm_test [--trials T] [--only <substring of the variant name>] [--levers]
//   --levers  also the lever alternates (tests/kernels/gemm_sm_levers.h; tools/probe/CMakeLists.txt builds them):
//             the base form of each Qwen3.8 row, qkv||z without 256 GRF, gate||up with the in-work-group split-K.
// Exit 1 on any failure; every config prints one line.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/kv8.h"
#include "common/bf16.h"
#include "common/int4.h"
#include "gemm_sm_levers.h"
#include "gemm_sm_ref.h"
#include "gemv_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace {

constexpr uint32_t kNaN = 0x7fc00bad;   // the fill: a quiet NaN no kernel writes

bool exists(const std::string& v) { return bool(std::ifstream(kernels::path(v), std::ios::binary)); }

struct Weights {
  kernels::SmallMTuning t{};
  common::Int4Gptq i4;
  std::vector<uint32_t> words;   // L0 qweight / L1 tiles
  std::vector<int8_t> q8;        // tiled
  std::vector<float> cs;
  gemm_sm::WeightView view() const {
    gemm_sm::WeightView v;
    v.K = t.K;
    v.N = t.N;
    v.kind = t.layout;
    if (t.layout == 2) {
      v.q8 = q8.data();
      v.colscale = cs.data();
    } else {
      v.words = words.data();
      v.scales = i4.scales.data();
    }
    return v;
  }
};

Weights make_weights(const kernels::SmallMTuning& t, uint32_t seed) {
  Weights w;
  w.t = t;
  if (t.layout == 2) {
    w.q8.resize(size_t(t.K) * t.N);
    w.cs.resize(t.N);
    std::mt19937 g(seed);
    for (auto& q : w.q8) q = int8_t(int(g() % 255) - 127);
    std::uniform_real_distribution<float> sd(1e-4f, 2e-3f);
    for (auto& s : w.cs) s = sd(g);
  } else if (t.layout == 3) {   // random 2-bit codes, f16 scales of an int4 group's size
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> sd(2e-3f, 2e-2f);
    w.words.resize(size_t(t.N / 16) * (t.K / 64) * 72);
    for (size_t i = 0; i < w.words.size(); ++i)
      w.words[i] = i % 72 < 64 ? uint32_t(g()) : (uint32_t(common::kv8::f32_to_f16_rne(sd(g))) |
                                                  uint32_t(common::kv8::f32_to_f16_rne(sd(g))) << 16);
  } else {
    w.i4 = common::Int4Gptq::random(t.K, t.N, seed);
    w.words = t.layout == 1 ? w.i4.tiled() : w.i4.qweight;
  }
  return w;
}

std::vector<uint32_t> checked_columns(uint32_t N) {
  std::vector<uint32_t> c;
  if (N <= 65536) {
    for (uint32_t n = 0; n < N; ++n) c.push_back(n);
    return c;
  }
  for (uint32_t n = 0; n < 512; ++n) c.push_back(n);
  for (uint32_t g = 8; g < N / 64 - 8; ++g) c.push_back(g * 64 + (g * 37u) % 64u);
  for (uint32_t n = N - 512; n < N; ++n) c.push_back(n);
  return c;
}

struct Dev {
  l0::Context& ctx;
  l0::Queue& q;
  l0::Fence& f;
};

// One bound kernel over the shape's device buffers.
struct Bound {
  std::unique_ptr<l0::Module> mod;
  std::unique_ptr<l0::Kernel> k;
  uint32_t gx = 1, gy = 1, S = 1, M = 1;
};
Bound bind_gemm_sm(Dev& d, const kernels::SmallMTuning& t, uint32_t M) {
  Bound b;
  b.mod = std::make_unique<l0::Module>(d.ctx, kernels::path(kernels::gemm_sm_variant(M, t)));
  b.k = std::make_unique<l0::Kernel>(*b.mod, kernels::gemm_sm_entry(t));
  b.k->group_size(kernels::gemm_sm_wg_size(t));
  b.gx = t.N / kernels::gemm_sm_cols(t);
  b.gy = kernels::gemm_sm_grid_y(t);
  b.S = kernels::gemm_sm_out_s(t);
  b.M = M;
  return b;
}
Bound bind_gemv(Dev& d, const kernels::SmallMTuning& t, uint32_t M) {
  Bound b;
  const std::string v = t.layout == 2 ? kernels::gemv_i8w_variant(M, t.K, t.N)
                                      : kernels::gemv_variant(M, t.K, t.N, t.S, t.layout);
  b.mod = std::make_unique<l0::Module>(d.ctx, kernels::path(v));
  b.k = std::make_unique<l0::Kernel>(*b.mod, t.layout == 2 ? "gemv_i8w" : "gemv");
  b.k->group_size(64);
  b.gx = t.N / 64;
  b.gy = t.layout == 2 ? 1 : t.S;
  b.S = t.layout == 2 ? 1 : t.S;
  b.M = M;
  return b;
}

struct Buffers {
  std::unique_ptr<l0::Mem> w, sc, x, out;
  size_t out_bytes = 0;
};

// Run `b` over rows `x` (M rows) and return the raw partials [S][M][N]; the output is NaN-filled first.
std::vector<float> run(Dev& d, Buffers& bf, Bound& b, const std::vector<uint16_t>& x, uint32_t N) {
  l0::CmdList imm = l0::CmdList::immediate(d.ctx);
  imm.copy(bf.x->ptr(), x.data(), x.size() * 2);
  imm.fill(bf.out->ptr(), kNaN, bf.out_bytes);
  b.k->arg_ptr(0, bf.w->ptr());
  b.k->arg_ptr(1, bf.sc->ptr());
  b.k->arg_ptr(2, bf.x->ptr());
  b.k->arg_ptr(3, bf.out->ptr());
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(*b.k, b.gx, b.gy);
  list.close();
  d.q.execute(list, &d.f);
  d.f.wait();
  std::vector<float> p(size_t(b.S) * b.M * N);
  imm.copy(p.data(), bf.out->ptr(), p.size() * 4);
  return p;
}
// Only row i's slices, [S][N] (the trials read one row).
std::vector<float> run_row(Dev& d, Buffers& bf, Bound& b, const std::vector<uint16_t>& x, uint32_t N,
                           uint32_t i) {
  l0::CmdList imm = l0::CmdList::immediate(d.ctx);
  imm.copy(bf.x->ptr(), x.data(), x.size() * 2);
  imm.fill(bf.out->ptr(), kNaN, bf.out_bytes);
  b.k->arg_ptr(0, bf.w->ptr());
  b.k->arg_ptr(1, bf.sc->ptr());
  b.k->arg_ptr(2, bf.x->ptr());
  b.k->arg_ptr(3, bf.out->ptr());
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(*b.k, b.gx, b.gy);
  list.close();
  d.q.execute(list, &d.f);
  d.f.wait();
  std::vector<float> r(size_t(b.S) * N);
  for (uint32_t s = 0; s < b.S; ++s)
    imm.copy(r.data() + size_t(s) * N, static_cast<const float*>(bf.out->ptr()) + (size_t(s) * b.M + i) * N,
             size_t(N) * 4);
  return r;
}

std::vector<float> fold(const std::vector<float>& p, uint32_t S, uint32_t M, uint32_t N) {
  const size_t mn = size_t(M) * N;
  std::vector<float> o(mn, 0.f);
  for (uint32_t s = 0; s < S; ++s)
    for (size_t i = 0; i < mn; ++i) o[i] += p[size_t(s) * mn + i];
  return o;
}
std::vector<float> row_slices(const std::vector<float>& p, uint32_t S, uint32_t M, uint32_t N, uint32_t m) {
  std::vector<float> r(size_t(S) * N);
  for (uint32_t s = 0; s < S; ++s)
    std::memcpy(r.data() + size_t(s) * N, p.data() + (size_t(s) * M + m) * N, size_t(N) * 4);
  return r;
}
bool bitwise(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// N2 over the checked columns only: compact both sides to those columns.
gemm_sm::N2 n2_cols(const std::vector<float>& got, const std::vector<double>& ref, const std::vector<double>& bound,
                    uint32_t M, uint32_t N, const std::vector<uint32_t>& cols) {
  const uint32_t C = uint32_t(cols.size());
  std::vector<float> g(size_t(M) * C);
  std::vector<double> r(g.size()), b(g.size());
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t c = 0; c < C; ++c) {
      g[size_t(m) * C + c] = got[size_t(m) * N + cols[c]];
      r[size_t(m) * C + c] = ref[size_t(m) * N + cols[c]];
      b[size_t(m) * C + c] = bound[size_t(m) * N + cols[c]];
    }
  return gemm_sm::n2(g, r, b, M, C);
}

int test_row(Dev& d, const kernels::SmallMTuning& t, uint32_t trials, uint32_t seed) {
  const std::vector<unsigned> ms = kernels::gemm_sm_built_m(t);
  const uint32_t Mmax = ms.back();
  const std::string name0 = kernels::gemm_sm_variant(Mmax, t);
  for (uint32_t M : ms)
    if (!exists(kernels::gemm_sm_variant(M, t))) {
      std::printf("FAIL %s: %s is not built\n", name0.c_str(), kernels::gemm_sm_variant(M, t).c_str());
      return 1;
    }
  const Weights w = make_weights(t, seed);
  const gemm_sm::WeightView view = w.view();
  Buffers bf;
  l0::CmdList imm = l0::CmdList::immediate(d.ctx);
  const size_t wb = t.layout == 2 ? w.q8.size() : w.words.size() * 4;
  bf.w = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Device, wb);
  imm.copy(bf.w->ptr(), t.layout == 2 ? static_cast<const void*>(w.q8.data()) : w.words.data(), wb);
  const size_t sb = t.layout == 2 ? w.cs.size() * 4 : t.layout == 0 ? w.i4.scales.size() * 2 : 64;
  bf.sc = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Device, sb);
  if (t.layout == 2) imm.copy(bf.sc->ptr(), w.cs.data(), sb);
  if (t.layout == 0) imm.copy(bf.sc->ptr(), w.i4.scales.data(), sb);
  bf.x = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Device, size_t(16) * t.K * 2);
  bf.out_bytes = size_t(std::max(t.S, 1u)) * 16 * t.N * 4;
  bf.out = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Device, bf.out_bytes);

  // 16 random rows (bf16 in [-1, 1)), the fp64 reference over the checked columns.
  const std::vector<uint16_t> x = random_bf16(size_t(16) * t.K, seed * 7 + 1);
  const std::vector<float> xf = gemm_sm::bf16_rows(x);
  const std::vector<uint32_t> cols = checked_columns(t.N);
  std::vector<double> ref, bound;
  if (cols.size() == t.N) {
    gemm_sm::ref(view, xf, Mmax, ref, bound);
  } else {
    // contiguous runs of the checked columns, one ref call each (ref keeps columns outside a run)
    size_t i = 0;
    while (i < cols.size()) {
      size_t j = i + 1;
      while (j < cols.size() && cols[j] == cols[j - 1] + 1) ++j;
      gemm_sm::ref(view, xf, Mmax, ref, bound, cols[i], cols[j - 1] + 1);
      i = j;
    }
  }

  int fails = 0;
  double worst_ref = 0, worst_gemv = 0, min_cos = 1;
  // Reference partials of every row at Mmax (property R's reference).
  Bound bmax = bind_gemm_sm(d, t, Mmax);
  const std::vector<float> pmax = run(d, bf, bmax, std::vector<uint16_t>(x.begin(), x.begin() + size_t(Mmax) * t.K), t.N);
  size_t r_rows = 0, r_diff = 0;
  for (uint32_t M : ms) {
    Bound b = bind_gemm_sm(d, t, M);
    const std::vector<uint16_t> xm(x.begin(), x.begin() + size_t(M) * t.K);
    const std::vector<float> p = run(d, bf, b, xm, t.N);
    const std::vector<float> o = fold(p, b.S, M, t.N);
    // the reference was computed over Mmax rows; rows 0..M-1 are a prefix of it
    std::vector<double> rm(ref.begin(), ref.begin() + size_t(M) * t.N), bm(bound.begin(), bound.begin() + size_t(M) * t.N);
    const gemm_sm::N2 e = n2_cols(o, rm, bm, M, t.N, cols);
    worst_ref = std::max(worst_ref, e.max_ratio);
    min_cos = std::min(min_cos, e.min_cos);
    if (!e.pass()) {
      std::printf("FAIL %s M=%u: N2 vs fp64 max ratio %.4g (%zu over), min cos %.9f\n", name0.c_str(), M,
                  e.max_ratio, e.over, e.min_cos);
      ++fails;
    }
    const std::string gv = t.layout == 2 ? kernels::gemv_i8w_variant(M, t.K, t.N)
                                         : kernels::gemv_variant(M, t.K, t.N, t.S, t.layout);
    if (M <= 8 && exists(gv)) {
      Bound g = bind_gemv(d, t, M);
      const std::vector<float> og = fold(run(d, bf, g, xm, t.N), g.S, M, t.N);
      const std::vector<double> ogd(og.begin(), og.end());
      const gemm_sm::N2 eg = n2_cols(o, ogd, bm, M, t.N, cols);
      worst_gemv = std::max(worst_gemv, eg.max_ratio);
      if (eg.over) {
        std::printf("FAIL %s M=%u: N2 vs %s max ratio %.4g (%zu over)\n", name0.c_str(), M, gv.c_str(), eg.max_ratio,
                    eg.over);
        ++fails;
      }
    }
    // Spec 19 v3d: the 2D block read is a load path only - bitwise its plain-read twin where that is built
    kernels::SmallMTuning plain = t;
    plain.blk2d = false;
    if (t.blk2d && exists(kernels::gemm_sm_variant(M, plain))) {
      Bound b2 = bind_gemm_sm(d, plain, M);
      if (!bitwise(p, run(d, bf, b2, xm, t.N))) {
        std::printf("FAIL %s M=%u: not bitwise %s\n", name0.c_str(), M, kernels::gemm_sm_variant(M, plain).c_str());
        ++fails;
      } else if (M == Mmax) {
        std::printf("  %s: bitwise %s at every M\n", name0.c_str(), kernels::gemm_sm_variant(M, plain).c_str());
      }
    }
    // R, part 1: each row equals the same row at Mmax, every slice
    for (uint32_t m = 0; m < M; ++m) {
      ++r_rows;
      if (!bitwise(row_slices(p, b.S, M, t.N, m), row_slices(pmax, bmax.S, Mmax, t.N, m))) ++r_diff;
    }
  }
  // R, part 2: probe rows at random indices among fresh random rows, at random built M.
  std::mt19937 g(seed * 31 + 5);
  std::vector<std::vector<float>> probe_ref;   // probe j = row j of x, reference = pmax's row j
  for (uint32_t j = 0; j < 4; ++j) probe_ref.push_back(row_slices(pmax, bmax.S, Mmax, t.N, j));
  std::vector<Bound> bs;
  for (uint32_t M : ms) bs.push_back(bind_gemm_sm(d, t, M));
  size_t t_diff = 0;
  std::uniform_real_distribution<float> ud(-1.f, 1.f);
  for (uint32_t tr = 0; tr < trials; ++tr) {
    const size_t mi = g() % ms.size();
    const uint32_t M = ms[mi];
    const uint32_t i = g() % M, j = g() % 4;
    std::vector<uint16_t> xt(size_t(M) * t.K);
    for (auto& e : xt) e = common::f32_to_bf16(ud(g));
    std::memcpy(xt.data() + size_t(i) * t.K, x.data() + size_t(j) * t.K, size_t(t.K) * 2);
    if (!bitwise(run_row(d, bf, bs[mi], xt, t.N, i), probe_ref[j])) ++t_diff;
  }
  if (r_diff || t_diff) {
    std::printf("FAIL %s: property R - %zu of %zu rows differ across M, %zu of %u trials differ\n", name0.c_str(),
                r_diff, r_rows, t_diff, trials);
    ++fails;
  }
  std::printf("%s %s (M", fails ? "FAIL" : "PASS", name0.c_str());
  for (uint32_t M : ms) std::printf(" %u", M);
  std::printf("): N2 vs fp64 max %.4f of the bound (%zu columns), min cos %.9f; vs gemv max %.4f; R %zu rows, %u "
              "trials, 0 must differ: %zu / %zu\n",
              worst_ref, cols.size(), min_cos, worst_gemv, r_rows, trials, r_diff, t_diff);
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t trials = 1000;
  std::string only;
  bool levers = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--trials" && i + 1 < argc) trials = uint32_t(std::stoul(argv[++i]));
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--levers") levers = true;
    else {
      std::fprintf(stderr, "usage: gemm_sm_test [--trials T] [--only <substring>] [--levers]\n");
      return 2;
    }
  }
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  Dev d{ctx, q, f};
  std::vector<kernels::SmallMTuning> rows(std::begin(kernels::kSmallMTuning), std::end(kernels::kSmallMTuning));
  if (levers)
    for (const auto& r : gemm_sm::lever_rows()) rows.push_back(r);
  int fails = 0, ran = 0;
  uint32_t seed = 1;
  for (const kernels::SmallMTuning& t : rows) {
    ++seed;
    if (!only.empty() && kernels::gemm_sm_variant(8, t).find(only) == std::string::npos) continue;
    ++ran;
    fails += test_row(d, t, trials, seed);
  }
  CHECK(ran > 0);
  if (fails) {
    std::printf("gemm_sm_test FAILED: %d failure(s) over %d rows\n", fails, ran);
    return 1;
  }
  std::printf("gemm_sm_test OK: %d rows\n", ran);
  return 0;
}
