// `runtime::prefill::gemm_bf16_batched` -- the batched Xe bf16 GEMM behind
// composed attention (rulings A14/A15, plan 6d-composed Task 1).
//
// Eight bars, in the plan's order:
//
//   1. Correctness at the two attention shapes, packed, at L in {1, 4, 24},
//      against a `double` reference over exactly-widened bf16 inputs, sampled.
//   2. Batch independence: entry `l` bitwise identical to the same GEMM run
//      alone at L = 1. A failure means the batch mode changes the reduction.
//   3. `strideB = 0` (the GQA form): every entry bitwise identical to each
//      other and to the L = 1 run.
//   4. Non-packed pitches -- the stride-plumbing test. `ldb`, `lda` and `ldc`
//      each widened over a padded buffer whose live window holds the packed
//      operand; each must be bitwise identical to the packed run. This is the
//      case that catches an (N,K,L) vs (K,N,L) mode swap, silent otherwise.
//   5. `transB`: B as [N][K] row-major, bitwise identical to the packed [K][N]
//      run of the same logical matrix.
//   6. The M-stacking identity: L = 6 with strideB = 0 over one buffer must be
//      bitwise identical to L = 1 at M = 6*2048. Task 4 relies on it, and it
//      holds because 2048 is a multiple of the 256-row M-tile.
//   7. Determinism (spec §6.4): every case runs twice into two distinct output
//      allocations and the full fp32 result is memcmp'd.
//   8. The timed rows for P-1, P-2 and P-3 (docs/probe-gemm-batched-2026-09-05.md),
//      labelled **iterate grade, not a bench row** -- this is not the P2 harness.
//
// The reference accumulates in `double` over `common::bf16_to_f32` (exact), so
// the bar covers only the device's fp32 accumulation. Bar and its derivation
// are plan 6c Task 1 Step 5's, quoted: for each checked element compute
// S = sum_k |A[m][k]*B[k][n]| in double and require
//     |C_dev - C_ref| <= 64 * 2^-24 * S + 2^-24
// with 2^-24 fp32's unit roundoff. The random-walk model puts the expected
// ratio at ~0.93 independently of K, so 64 is a ~69x margin; it is a
// probabilistic bar, not a proof, which is why the measured max ratio is
// PRINTED at every case.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "gemv_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"

namespace {

using runtime::prefill::GemmBatch;

constexpr uint32_t kC = 2048;      // PrefillScratch::kC (ruling A13)
constexpr uint32_t kD = 256;       // head dim
constexpr uint32_t kDepth = 4096;  // pos + C at the operating point
constexpr double kUlp = 5.9604644775390625e-08;  // 2^-24
constexpr double kBar = 64.0;
constexpr size_t kSamples = 4096;
constexpr size_t kStageBytes = 64ul << 20;
constexpr int kReplays = 8;
constexpr int kDropped = 3;
constexpr int kEnqueues = 4;  // P2's, so the rates are comparable to its matrix

l0::Mem device_copy(l0::Context& ctx, const std::vector<uint16_t>& host) {
  l0::Mem mem(ctx, l0::MemKind::Device, host.size() * sizeof(uint16_t));
  l0::CmdList::immediate(ctx).copy(mem.ptr(), host.data(), host.size() * sizeof(uint16_t));
  return mem;
}

// Two device buffers hold the same fp32 result iff every byte matches. Compared
// through a 64 MB host staging pair so an 805 MB output costs no host image.
bool bitwise_equal(l0::Context& ctx, const void* a, const void* b, size_t bytes) {
  const size_t chunk = std::min(kStageBytes, bytes);
  l0::Mem ha(ctx, l0::MemKind::Host, chunk);
  l0::Mem hb(ctx, l0::MemKind::Host, chunk);
  for (size_t off = 0; off < bytes; off += chunk) {
    const size_t now = std::min(chunk, bytes - off);
    l0::CmdList copy = l0::CmdList::immediate(ctx);
    copy.copy(ha.ptr(), static_cast<const char*>(a) + off, now);
    copy.copy(hb.ptr(), static_cast<const char*>(b) + off, now);
    if (std::memcmp(ha.ptr(), hb.ptr(), now) != 0) return false;
  }
  return true;
}

std::vector<float> download(l0::Context& ctx, const void* src, size_t elements) {
  std::vector<float> out(elements);
  l0::CmdList::immediate(ctx).copy(out.data(), src, elements * sizeof(float));
  return out;
}

// One problem's host-side addressing, shared by the reference and the device
// arguments so the two cannot drift.
struct Problem {
  GemmBatch b;
  bool transB = false;
  size_t a_elements = 0, b_elements = 0, c_elements = 0;

  size_t a_index(uint32_t l, uint32_t m, uint32_t k) const {
    return l * b.strideA + size_t(m) * b.lda + k;
  }
  size_t b_index(uint32_t l, uint32_t k, uint32_t n) const {
    return l * b.strideB + (transB ? size_t(n) * b.ldb + k : size_t(k) * b.ldb + n);
  }
  size_t c_index(uint32_t l, uint32_t m, uint32_t n) const {
    return l * b.strideC + size_t(m) * b.ldc + n;
  }
};

Problem packed(uint32_t M, uint32_t K, uint32_t N, uint32_t L) {
  Problem p;
  p.b = GemmBatch{M, K, N, L, K, N, N, size_t(M) * K, size_t(K) * N, size_t(M) * N};
  p.a_elements = size_t(L) * M * K;
  p.b_elements = size_t(L) * K * N;
  p.c_elements = size_t(L) * M * N;
  return p;
}

struct Ratio {
  double max_ratio = 0.0;
  double max_abs_err = 0.0;
};

// The sampled `double` reference and the bar, at 4096 fixed (l, m, n) triples.
// The device output is walked in 64 MB chunks rather than downloaded whole: at
// QK^T L = 24 the result is 805,306,368 B and a host image of it is a needless
// allocation, not a check.
Ratio check_reference(l0::Context& ctx, const Problem& p, const std::vector<uint16_t>& a,
                      const std::vector<uint16_t>& b, const void* c_dev) {
  struct Sample {
    size_t index;
    uint32_t l, m, n;
  };
  std::mt19937 rng(0x6e6d31u);
  std::vector<Sample> samples;
  samples.reserve(kSamples);
  for (size_t s = 0; s < kSamples; ++s) {
    const uint32_t l = rng() % p.b.L, m = rng() % p.b.M, n = rng() % p.b.N;
    samples.push_back({p.c_index(l, m, n), l, m, n});
  }
  std::sort(samples.begin(), samples.end(),
            [](const Sample& x, const Sample& y) { return x.index < y.index; });

  Ratio r;
  const size_t bytes = p.c_elements * sizeof(float);
  const size_t chunk = std::min(kStageBytes, bytes);
  l0::Mem stage(ctx, l0::MemKind::Host, chunk);
  size_t next = 0;
  for (size_t off = 0; off < bytes && next < samples.size(); off += chunk) {
    const size_t now = std::min(chunk, bytes - off);
    l0::CmdList::immediate(ctx).copy(stage.ptr(), static_cast<const char*>(c_dev) + off, now);
    const size_t first = off / sizeof(float), end = first + now / sizeof(float);
    const float* got = stage.as<float>();
    while (next < samples.size() && samples[next].index < end) {
      const Sample& s = samples[next++];
      double ref = 0.0, mag = 0.0;
      for (uint32_t k = 0; k < p.b.K; ++k) {
        const double av = common::bf16_to_f32(a[p.a_index(s.l, s.m, k)]);
        const double bv = common::bf16_to_f32(b[p.b_index(s.l, k, s.n)]);
        ref += av * bv;
        mag += std::fabs(av * bv);
      }
      const double err = std::fabs(double(got[s.index - first]) - ref);
      r.max_abs_err = std::max(r.max_abs_err, err);
      r.max_ratio = std::max(r.max_ratio, err / (kUlp * mag + kUlp));
      CHECK(err <= kBar * kUlp * mag + kUlp);
    }
  }
  CHECK_EQ(next, samples.size());
  return r;
}

double time_case(runtime::prefill::Context& cx, const Problem& p, const uint16_t* A,
                 const uint16_t* B, float* C) {
  runtime::prefill::gemm_bf16_batched(cx, p.b, A, B, C, p.transB);  // warm-up, discarded
  cx.wait();
  std::array<double, kReplays - kDropped> samples{};
  for (int replay = 0; replay < kReplays; ++replay) {
    const auto begin = std::chrono::steady_clock::now();
    for (int e = 0; e < kEnqueues; ++e)
      runtime::prefill::gemm_bf16_batched(cx, p.b, A, B, C, p.transB);
    cx.wait();
    if (replay >= kDropped)
      samples[size_t(replay - kDropped)] =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
              .count() /
          kEnqueues;
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double tflops(const GemmBatch& b, double ms) {
  return 2.0 * double(b.M) * b.K * b.N * b.L / (ms * 1e9);
}

void print_grid(const Problem& p) {
  uint32_t g[3] = {0, 0, 0};
  runtime::prefill::gemm_bf16_batched_grid(p.b, p.transB, g);
  std::printf("grid %ux%ux%u = %llu WGs", g[0], g[1], g[2],
              static_cast<unsigned long long>(uint64_t(g[0]) * g[1] * g[2]));
}

// Case 1 + case 7 at one (shape, L): correctness against the sampled reference,
// and two runs into two allocations compared bitwise.
void correctness_and_determinism(l0::Context& ctx, runtime::prefill::Context& cx, const char* name,
                                 uint32_t M, uint32_t K, uint32_t N, uint32_t L) {
  const Problem p = packed(M, K, N, L);
  const std::vector<uint16_t> a = random_bf16(p.a_elements, 0x9e3779b9u ^ (K * 31u + L));
  const std::vector<uint16_t> b = random_bf16(p.b_elements, 0x85ebca6bu ^ (N * 17u + L));
  l0::Mem da = device_copy(ctx, a);
  l0::Mem db = device_copy(ctx, b);
  l0::Mem c1(ctx, l0::MemKind::Device, p.c_elements * sizeof(float));
  l0::Mem c2(ctx, l0::MemKind::Device, p.c_elements * sizeof(float));

  runtime::prefill::gemm_bf16_batched(cx, p.b, da.as<uint16_t>(), db.as<uint16_t>(),
                                      c1.as<float>());
  runtime::prefill::gemm_bf16_batched(cx, p.b, da.as<uint16_t>(), db.as<uint16_t>(),
                                      c2.as<float>());
  cx.wait();
  CHECK(bitwise_equal(ctx, c1.ptr(), c2.ptr(), p.c_elements * sizeof(float)));

  const Ratio r = check_reference(ctx, p, a, b, c1.ptr());
  std::printf("  %-22s L=%2u  max|err| %.6g  max ratio %.3f (bar %.0f)  determinism bitwise\n",
              name, L, r.max_abs_err, r.max_ratio, kBar);
}

}  // namespace

int main() {
  l0::Context ctx(0);
  runtime::prefill::Context cx(ctx);
  std::printf("gemm_batched_test on %s\n", ctx.name().c_str());
  std::printf("transB (ColumnMajor-B chain) instantiated: %s\n",
              runtime::prefill::gemm_bf16_supports_transb() ? "YES" : "NO");

  // --- 1 + 7: correctness and determinism, packed, both attention shapes -----
  std::printf("[1,7] correctness vs double reference + two-run determinism\n");
  for (uint32_t L : {1u, 4u, 24u}) {
    correctness_and_determinism(ctx, cx, "QK^T 2048x256x4096", kC, kD, kDepth, L);
    correctness_and_determinism(ctx, cx, "PV   2048x4096x256", kC, kDepth, kD, L);
  }

  // --- 2: batch independence -------------------------------------------------
  // Entry l of an L = 4 run must be bitwise identical to the same operands run
  // alone at L = 1.
  {
    const Problem p4 = packed(kC, kDepth, kD, 4);
    const Problem p1 = packed(kC, kDepth, kD, 1);
    const std::vector<uint16_t> a = random_bf16(p4.a_elements, 0x1234567u);
    const std::vector<uint16_t> b = random_bf16(p4.b_elements, 0x89abcdeu);
    l0::Mem da = device_copy(ctx, a), db = device_copy(ctx, b);
    l0::Mem c4(ctx, l0::MemKind::Device, p4.c_elements * sizeof(float));
    l0::Mem c1(ctx, l0::MemKind::Device, p1.c_elements * sizeof(float));
    runtime::prefill::gemm_bf16_batched(cx, p4.b, da.as<uint16_t>(), db.as<uint16_t>(),
                                        c4.as<float>());
    cx.wait();
    for (uint32_t l = 0; l < 4; ++l) {
      runtime::prefill::gemm_bf16_batched(cx, p1.b, da.as<uint16_t>() + l * p1.a_elements,
                                          db.as<uint16_t>() + l * p1.b_elements, c1.as<float>());
      cx.wait();
      CHECK(bitwise_equal(ctx, c4.as<float>() + l * p1.c_elements, c1.ptr(),
                          p1.c_elements * sizeof(float)));
    }
    std::printf("[2] batch independence: all 4 entries bitwise == the L=1 run\n");
  }

  // --- 3: strideB = 0, the GQA form -----------------------------------------
  // Six q-heads sharing one kv-head: one B pointer, six batch entries whose only
  // difference is strideA / strideC. Every entry must equal the L = 1 run.
  {
    const uint32_t L = 6;
    Problem p = packed(kC, kDepth, kD, L);
    p.b.strideB = 0;
    p.b_elements = size_t(kDepth) * kD;
    const std::vector<uint16_t> a = random_bf16(p.a_elements, 0x2468aceu);
    const std::vector<uint16_t> b = random_bf16(p.b_elements, 0x13579bdu);
    l0::Mem da = device_copy(ctx, a), db = device_copy(ctx, b);
    l0::Mem c6(ctx, l0::MemKind::Device, p.c_elements * sizeof(float));
    runtime::prefill::gemm_bf16_batched(cx, p.b, da.as<uint16_t>(), db.as<uint16_t>(),
                                        c6.as<float>());
    cx.wait();
    const Problem p1 = packed(kC, kDepth, kD, 1);
    l0::Mem c1(ctx, l0::MemKind::Device, p1.c_elements * sizeof(float));
    for (uint32_t l = 0; l < L; ++l) {
      runtime::prefill::gemm_bf16_batched(cx, p1.b, da.as<uint16_t>() + l * p1.a_elements,
                                          db.as<uint16_t>(), c1.as<float>());
      cx.wait();
      CHECK(bitwise_equal(ctx, c6.as<float>() + l * p1.c_elements, c1.ptr(),
                          p1.c_elements * sizeof(float)));
    }
    const Ratio r = check_reference(ctx, p, a, b, c6.ptr());
    std::printf("[3] strideB=0 L=6: every entry bitwise == the L=1 run; max ratio %.3f\n",
                r.max_ratio);
  }

  // --- 4: non-packed pitches -------------------------------------------------
  // Each of ldb, lda and ldc widened in turn over a padded buffer whose live
  // window holds the packed operand; the result must not move a bit.
  {
    const Problem ref = packed(kC, kDepth, kD, 1);  // PV shape
    const std::vector<uint16_t> a = random_bf16(ref.a_elements, 0xfeedfaceu);
    const std::vector<uint16_t> b = random_bf16(ref.b_elements, 0xdeadbeefu);
    l0::Mem da = device_copy(ctx, a), db = device_copy(ctx, b);
    l0::Mem cref(ctx, l0::MemKind::Device, ref.c_elements * sizeof(float));
    runtime::prefill::gemm_bf16_batched(cx, ref.b, da.as<uint16_t>(), db.as<uint16_t>(),
                                        cref.as<float>());
    cx.wait();

    // ldb = 1024: B lives at columns [256, 512) of a [K][1024] slab -- the K/V
    // cache's own geometry ([pos][4][256], one kv-head's rows pitched 1024).
    {
      const size_t ldb = 1024, head = 1;
      std::vector<uint16_t> wide(size_t(kDepth) * ldb, 0);
      for (uint32_t k = 0; k < kDepth; ++k)
        for (uint32_t n = 0; n < kD; ++n) wide[size_t(k) * ldb + head * kD + n] = b[size_t(k) * kD + n];
      l0::Mem dwide = device_copy(ctx, wide);
      GemmBatch g = ref.b;
      g.ldb = ldb;
      l0::Mem c(ctx, l0::MemKind::Device, ref.c_elements * sizeof(float));
      runtime::prefill::gemm_bf16_batched(cx, g, da.as<uint16_t>(),
                                          dwide.as<uint16_t>() + head * kD, c.as<float>());
      cx.wait();
      CHECK(bitwise_equal(ctx, c.ptr(), cref.ptr(), ref.c_elements * sizeof(float)));
    }
    // lda = 6144: A is head 1 of a [M][24][256]-shaped q_out... at the PV shape
    // A's K is 4096, so the padded pitch is 4096+2048 and the live window is
    // columns [2048, 6144). Same mechanism, a pitch the GEMM must honour.
    {
      const size_t lda = 6144, off = 2048;
      std::vector<uint16_t> wide(size_t(kC) * lda, 0);
      for (uint32_t m = 0; m < kC; ++m)
        for (uint32_t k = 0; k < kDepth; ++k)
          wide[size_t(m) * lda + off + k] = a[size_t(m) * kDepth + k];
      l0::Mem dwide = device_copy(ctx, wide);
      GemmBatch g = ref.b;
      g.lda = lda;
      l0::Mem c(ctx, l0::MemKind::Device, ref.c_elements * sizeof(float));
      runtime::prefill::gemm_bf16_batched(cx, g, dwide.as<uint16_t>() + off, db.as<uint16_t>(),
                                          c.as<float>());
      cx.wait();
      CHECK(bitwise_equal(ctx, c.ptr(), cref.ptr(), ref.c_elements * sizeof(float)));
    }
    // ldc = 1024: the output written into one head's window of a [C][4][256]
    // slab. Compared row by row against the packed result.
    {
      const size_t ldc = 1024, head = 2;
      GemmBatch g = ref.b;
      g.ldc = ldc;
      l0::Mem c(ctx, l0::MemKind::Device, size_t(kC) * ldc * sizeof(float));
      l0::CmdList::immediate(ctx).fill(c.ptr(), 0u, size_t(kC) * ldc * sizeof(float));
      runtime::prefill::gemm_bf16_batched(cx, g, da.as<uint16_t>(), db.as<uint16_t>(),
                                          c.as<float>() + head * kD);
      cx.wait();
      const std::vector<float> got = download(ctx, c.ptr(), size_t(kC) * ldc);
      const std::vector<float> want = download(ctx, cref.ptr(), ref.c_elements);
      for (uint32_t m = 0; m < kC; ++m)
        CHECK(std::memcmp(&got[size_t(m) * ldc + head * kD], &want[size_t(m) * kD],
                          kD * sizeof(float)) == 0);
    }
    std::printf("[4] non-packed pitches: ldb=1024, lda=6144, ldc=1024 all bitwise == packed\n");
  }

  // --- 5: transB -------------------------------------------------------------
  if (runtime::prefill::gemm_bf16_supports_transb()) {
    // QK^T: B is the K cache, [N=Dp][K=256] row-major with pitch 1024.
    const Problem ref = packed(kC, kD, kDepth, 1);
    const std::vector<uint16_t> a = random_bf16(ref.a_elements, 0xc0ffeeu);
    const std::vector<uint16_t> bkn = random_bf16(ref.b_elements, 0xbadf00du);  // [K][N]
    const size_t ldb = 1024, head = 3;
    std::vector<uint16_t> bnk(size_t(kDepth) * ldb, 0);                          // [N][K], pitched
    for (uint32_t k = 0; k < kD; ++k)
      for (uint32_t n = 0; n < kDepth; ++n)
        bnk[size_t(n) * ldb + head * kD + k] = bkn[size_t(k) * kDepth + n];
    l0::Mem da = device_copy(ctx, a), dkn = device_copy(ctx, bkn), dnk = device_copy(ctx, bnk);
    l0::Mem c1(ctx, l0::MemKind::Device, ref.c_elements * sizeof(float));
    l0::Mem c2(ctx, l0::MemKind::Device, ref.c_elements * sizeof(float));
    runtime::prefill::gemm_bf16_batched(cx, ref.b, da.as<uint16_t>(), dkn.as<uint16_t>(),
                                        c1.as<float>());
    GemmBatch g = ref.b;
    g.ldb = ldb;
    runtime::prefill::gemm_bf16_batched(cx, g, da.as<uint16_t>(), dnk.as<uint16_t>() + head * kD,
                                        c2.as<float>(), /*transB=*/true);
    cx.wait();
    CHECK(bitwise_equal(ctx, c1.ptr(), c2.ptr(), ref.c_elements * sizeof(float)));
    std::printf("[5] transB: [N][K] ldb=1024 bitwise == the packed [K][N] run\n");
  } else {
    std::printf("[5] transB: SKIPPED -- no ColumnMajor-B chain in this build\n");
  }

  // --- 6: the M-stacking identity -------------------------------------------
  {
    const uint32_t L = 6;
    Problem p = packed(kC, kDepth, kD, L);
    p.b.strideB = 0;
    p.b_elements = size_t(kDepth) * kD;
    const std::vector<uint16_t> a = random_bf16(p.a_elements, 0x5a5a5a5au);
    const std::vector<uint16_t> b = random_bf16(p.b_elements, 0xa5a5a5a5u);
    l0::Mem da = device_copy(ctx, a), db = device_copy(ctx, b);
    l0::Mem cb(ctx, l0::MemKind::Device, p.c_elements * sizeof(float));
    l0::Mem cm(ctx, l0::MemKind::Device, p.c_elements * sizeof(float));
    runtime::prefill::gemm_bf16_batched(cx, p.b, da.as<uint16_t>(), db.as<uint16_t>(),
                                        cb.as<float>());
    const Problem stacked = packed(kC * L, kDepth, kD, 1);
    runtime::prefill::gemm_bf16_batched(cx, stacked.b, da.as<uint16_t>(), db.as<uint16_t>(),
                                        cm.as<float>());
    cx.wait();
    CHECK(bitwise_equal(ctx, cb.ptr(), cm.ptr(), p.c_elements * sizeof(float)));
    std::printf("[6] M-stacking: batched L=6 strideB=0 bitwise == one GEMM at M=%u\n", kC * L);
  }

  // --- 8: the timed rows (iterate grade, NOT a bench row) --------------------
  std::printf("\n[8] rates (measured, iterate-grade; %d replays, drop %d, median of %d, "
              "%d enqueues/replay)\n",
              kReplays, kDropped, kReplays - kDropped, kEnqueues);
  std::printf("| cell | M | K | N | L | strideB | grid | ms | TFLOP/s |\n");
  std::printf("|---|---:|---:|---:|---:|---|---:|---:|---:|\n");
  struct TimedCase {
    const char* name;
    uint32_t M, K, N, L;
    bool share_b;
  };
  const TimedCase timed[] = {
      {"P-1 PV packed", kC, kDepth, kD, 24, false},
      {"P-2 PV strideB=0", kC, kDepth, kD, 6, true},
      {"P-3 QK^T strideB=0", kC, kD, kDepth, 6, true},
      {"QK^T packed", kC, kD, kDepth, 24, false},
      {"PV strideB=0 L=24", kC, kDepth, kD, 24, true},
      {"PV packed L=6", kC, kDepth, kD, 6, false},
      {"QK^T packed L=6", kC, kD, kDepth, 6, false},
      {"PV single head", kC, kDepth, kD, 1, false},
      {"QK^T single head", kC, kD, kDepth, 1, false},
  };
  for (const TimedCase& t : timed) {
    Problem p = packed(t.M, t.K, t.N, t.L);
    if (t.share_b) {
      p.b.strideB = 0;
      p.b_elements = size_t(t.K) * t.N;
    }
    const std::vector<uint16_t> a = random_bf16(p.a_elements, 0x77777777u);
    const std::vector<uint16_t> b = random_bf16(p.b_elements, 0x33333333u);
    l0::Mem da = device_copy(ctx, a), db = device_copy(ctx, b);
    l0::Mem c(ctx, l0::MemKind::Device, p.c_elements * sizeof(float));
    const double ms = time_case(cx, p, da.as<uint16_t>(), db.as<uint16_t>(), c.as<float>());
    std::printf("| %s | %u | %u | %u | %u | %s | ", t.name, t.M, t.K, t.N, t.L,
                t.share_b ? "0" : "packed");
    print_grid(p);
    std::printf(" | %.3f | %.2f |\n", ms, tflops(p.b, ms));
    std::fflush(stdout);
  }

  std::printf("\ngemm_batched_test OK\n");
  return 0;
}
