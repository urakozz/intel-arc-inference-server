// pf_gemm (our DPAS GEMM on the L0 list) against a HOST fp64 reference at every shape the
// prefill walk will hand it (spec 2.1 §2 bar 1; until 2026-10-27 the reference was sycl-tla's
// gemm_bf16_batched, matched bitwise). Only the LOGICAL rectangle (rows < M, cols < N) is
// compared: pf_gemm computes 256-padded rows and columns that the walk never reads. P·V's
// device K is the 256-padded depth; the extra P columns are exact +0.0 and add nothing.
//
// **The tolerance, derived.** bf16 x bf16 products are exact in fp32 (8-bit significands), so
// the device's only error is the fp32 accumulation of K exact products, in whatever order the
// DPAS walk takes. For ANY summation order that error is at most gamma_(K-1) x S, S = sum |a b|
// over the element's K terms (Higham, Accuracy and Stability §4.2), i.e. at most about K u S
// with u = 2^-24, and u S <= ulp32(S). The reference's own error (fp64 sums of the same exact
// products) is below 2^-40 S and is ignored. So, per element:
//   max-ulp  |got - ref| / ulp32(S) <= K                              (the worst case; gated)
// and over each case's compared rectangle:
//   cosine(got, ref) >= 0.999999                                       (gated)
// The cosine bar sits far outside the error's expected size - a random-walk accumulation
// error is ~sqrt(K) u relative, so 1 - cos ~ K u^2 < 1e-10 here - and far inside what a
// layout, offset or stride bug produces (cos near 0). The observed max-ulp is printed for the
// record. The host reference runs on a row sample of the big shapes (`sample_rows`: both
// ends of every 64-row block, so every 256-row tile boundary is in it) and every column.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"

namespace {
using runtime::prefill::GemmBatch;
using runtime::prefill::pad256;
constexpr uint32_t kSlab = 1024, kHd = 256, kMaxLen = 16384, kGroup = 6;
constexpr size_t kStage = 64ul << 20;

std::vector<uint16_t> random_bf16(size_t n, uint32_t seed) {   // incompressible, magnitude in [1, 2), random sign
  std::mt19937 rng(seed);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = uint16_t(0x3F80u | (rng() & 0x007Fu)) ^ uint16_t((rng() & 1u) << 15);
  return v;
}
l0::Mem upload(l0::Context& ctx, const std::vector<uint16_t>& h) {
  l0::Mem m(ctx, l0::MemKind::Device, h.size() * 2);
  l0::CmdList::immediate(ctx).copy(m.ptr(), h.data(), h.size() * 2);
  return m;
}
std::vector<uint16_t> download16(l0::Context& ctx, const void* src, size_t elems) {
  std::vector<uint16_t> out(elems);
  l0::Mem stage(ctx, l0::MemKind::Host, std::min(kStage, elems * 2));
  for (size_t off = 0; off < elems * 2; off += kStage) {
    const size_t now = std::min(kStage, elems * 2 - off);
    l0::CmdList::immediate(ctx).copy(stage.ptr(), static_cast<const char*>(src) + off, now);
    std::memcpy(reinterpret_cast<char*>(out.data()) + off, stage.ptr(), now);
  }
  return out;
}
std::vector<float> download(l0::Context& ctx, const void* src, size_t elems) {
  std::vector<float> out(elems);
  l0::Mem stage(ctx, l0::MemKind::Host, std::min(kStage, elems * 4));
  for (size_t off = 0; off < elems * 4; off += kStage) {
    const size_t now = std::min(kStage, elems * 4 - off);
    l0::CmdList::immediate(ctx).copy(stage.ptr(), static_cast<const char*>(src) + off, now);
    std::memcpy(reinterpret_cast<char*>(out.data()) + off, stage.ptr(), now);
  }
  return out;
}
bool zero_either_sign(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  return (u & 0x7FFFFFFFu) == 0;
}
// got[l][m][n] at got_index vs ref at ref_index over the logical rectangle.
struct Diff { size_t words = 0, sign_zero = 0, first = size_t(-1); };
Diff compare(const std::vector<float>& got, const std::vector<float>& ref, uint32_t L, uint32_t M,
             uint32_t N, size_t got_l, size_t got_m, size_t ref_l, size_t ref_m) {
  Diff d;
  for (uint32_t l = 0; l < L; ++l)
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t n = 0; n < N; ++n) {
        const size_t gi = l * got_l + size_t(m) * got_m + n, ri = l * ref_l + size_t(m) * ref_m + n;
        if (std::memcmp(&got[gi], &ref[ri], 4) == 0) continue;
        if (zero_either_sign(got[gi]) && zero_either_sign(ref[ri])) { ++d.sign_zero; continue; }
        if (d.words++ == 0) d.first = gi;
      }
  return d;
}
// The host reference's verdict for one case (see the file header's tolerance).
constexpr double kCosBar = 0.999999;
struct Tol {
  double dot = 0, g2 = 0, r2 = 0, max_ulp = 0;
  size_t n = 0;
  void add(float got, double ref, double S) {
    const double ulp = S > 0 ? std::ldexp(1.0, std::ilogb(S) - 23) : std::ldexp(1.0, -149);
    max_ulp = std::max(max_ulp, std::fabs(double(got) - ref) / ulp);
    dot += double(got) * ref;
    g2 += double(got) * double(got);
    r2 += ref * ref;
    ++n;
  }
};
void verdict(const char* what, const Tol& t, uint32_t K) {
  const double cos = t.dot / std::sqrt(t.g2 * t.r2);
  const bool ok = std::isfinite(cos) && cos >= kCosBar && t.max_ulp <= double(K);
  std::printf("  %-44s %s (cos %.9f, max %.1f ulp of sum|ab|, bar %u; %zu words)\n", what,
              ok ? "within" : "**OUTSIDE**", cos, t.max_ulp, K, t.n);
  CHECK(ok);
}
float f32(uint16_t h) { return common::bf16_to_f32(h); }
// Both ends of every 64-row block, and the last row: every 256-row tile boundary of pf_gemm is
// sampled. All rows when there are at most 64.
std::vector<uint32_t> sample_rows(uint32_t M) {
  std::vector<uint32_t> r;
  for (uint32_t m = 0; m < M; ++m)
    if (M <= 64 || m % 64 == 0 || m % 64 == 63 || m == M - 1) r.push_back(m);
  return r;
}

struct Dev {
  l0::Context ctx{0};
  runtime::prefill::Context cx{ctx};
  runtime::prefill::KernelCache kc{ctx};
};

// (a) one 1024-column slab of an int4 linear, written into a full-width C at column n0.
void slab_case(Dev& d, uint32_t M, uint32_t K, uint32_t Nfull) {
  const uint32_t Mp = pad256(M), n0 = kSlab;   // the SECOND slab: exercises the C offset
  const std::vector<uint16_t> a = random_bf16(size_t(Mp) * K, 11 + M + K);
  const std::vector<uint16_t> slab = random_bf16(size_t(K) * kSlab, 13 + Nfull);
  l0::Mem A = upload(d.ctx, a), B = upload(d.ctx, slab);
  l0::Mem Cpf(d.ctx, l0::MemKind::Device, size_t(Mp) * Nfull * 4);
  const GemmBatch pf{M, K, kSlab, 1, K, kSlab, Nfull, 0, 0, 0};
  runtime::prefill::gemm_l0(d.cx, d.kc, pf, A.as<uint16_t>(), B.as<uint16_t>(),
                            Cpf.as<float>() + n0, false);
  d.cx.wait();
  const std::vector<float> got = download(d.ctx, Cpf.ptr(), size_t(Mp) * Nfull);
  Tol t;
  std::vector<double> ref(kSlab), S(kSlab);
  for (uint32_t m : sample_rows(M)) {
    std::fill(ref.begin(), ref.end(), 0.0);
    std::fill(S.begin(), S.end(), 0.0);
    for (uint32_t k = 0; k < K; ++k) {
      const double av = f32(a[size_t(m) * K + k]);
      const uint16_t* b = &slab[size_t(k) * kSlab];
      for (uint32_t n = 0; n < kSlab; ++n) {
        const double p = av * f32(b[n]);
        ref[n] += p;
        S[n] += std::fabs(p);
      }
    }
    for (uint32_t n = 0; n < kSlab; ++n) t.add(got[size_t(m) * Nfull + n0 + n], ref[n], S[n]);
  }
  char name[96];
  std::snprintf(name, sizeof name, "slab M=%u K=%u N=%u @n0=%u", M, K, Nfull, n0);
  verdict(name, t, K);
}

// (d) Parity program S2(a): the gate||up slab walk with `pf_silu_mul` fused into each slab
// GEMM's epilogue, against the unfused pair -- the same GEMM into fp32 `partials` followed
// by `pf_silu_mul` -- on the SAME operands. Device against device, and bitwise: a host
// reference would be grading the device's `exp` as well, which is not what is in question.
void silu_case(Dev& d, uint32_t M) {
  constexpr uint32_t kK = 5120, kN = 34816, kX = kN / 2;   // gate||up's own shape
  constexpr uint32_t kSiluChunk = 4096;
  const uint32_t Mp = pad256(M);
  const std::vector<uint16_t> a = random_bf16(size_t(Mp) * kK, 41 + M);
  // TWO slabs behind one [K][2048] buffer, alternated by slab index: B must not repeat
  // with a period the x-column mapping could hide an offset error inside.
  const std::vector<uint16_t> bb = random_bf16(size_t(kK) * 2 * kSlab, 43 + M);
  l0::Mem A = upload(d.ctx, a), B = upload(d.ctx, bb);
  l0::Mem Part(d.ctx, l0::MemKind::Device, size_t(Mp) * kN * 4);
  l0::Mem Xfused(d.ctx, l0::MemKind::Device, size_t(Mp) * kX * 2);
  l0::Mem Xref(d.ctx, l0::MemKind::Device, size_t(Mp) * kX * 2);
  for (uint32_t n0 = 0; n0 < kN; n0 += kSlab) {
    const uint16_t* bp = B.as<uint16_t>() + ((n0 / kSlab) & 1u) * kSlab;
    const GemmBatch plain{M, kK, kSlab, 1, kK, 2 * kSlab, kN, 0, 0, 0};
    runtime::prefill::gemm_l0(d.cx, d.kc, plain, A.as<uint16_t>(), bp,
                              Part.as<float>() + n0, false);
    const GemmBatch fused{M, kK, kSlab, 1, kK, 2 * kSlab, 0, 0, 0, 0};
    runtime::prefill::gemm_l0_silu(d.cx, d.kc, fused, A.as<uint16_t>(), bp,
                                   Xfused.as<uint16_t>() + n0 / 2, kX);
  }
  d.cx.launch(d.kc(kernels::pf_silu_mul_variant(kX), "pf_silu_mul"),
              (kX + kSiluChunk - 1) / kSiluChunk, M, 1,
              {runtime::prefill::PtrArg(Part.ptr()), runtime::prefill::PtrArg(Xref.ptr()),
               runtime::prefill::arg_val(M)});
  d.cx.wait();

  // Rows [M, pad256(M)) only exist on the fused side -- pf_silu_mul's grid is M rows --
  // and the walk never reads them, so the comparison is the logical rectangle.
  const std::vector<uint16_t> got = download16(d.ctx, Xfused.ptr(), size_t(Mp) * kX);
  const std::vector<uint16_t> want = download16(d.ctx, Xref.ptr(), size_t(Mp) * kX);
  size_t words = 0, first = size_t(-1);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < kX; ++k) {
      const size_t i = size_t(m) * kX + k;
      if (got[i] != want[i] && words++ == 0) first = i;
    }
  std::printf("  %-44s %s (%zu words differ, first %zu)\n",
              ("silu-fused gate||up M=" + std::to_string(M)).c_str(),
              words == 0 ? "bitwise" : "**DIFFERS**", words, first);
  CHECK_EQ(words, size_t(0));
}

// (b) Q·K^T: 6 q-heads of kv group 1 against the K cache in place (transB), N = depth.
// (c) P·V: the same group's P against the V cache, K = depth (pf: padded to 256).
void attention_case(Dev& d, uint32_t C, uint32_t depth) {
  const uint32_t Mp = pad256(C), npad256 = pad256(depth);
  const uint32_t j = 1;      // kv group 1: a non-zero base offset on every operand
  const std::vector<uint16_t> q = random_bf16(size_t(Mp) * 24 * kHd, 21 + C + depth);
  const std::vector<uint16_t> kc = random_bf16(size_t(kMaxLen) * 4 * kHd, 23 + depth);
  const std::vector<uint16_t> vc = random_bf16(size_t(kMaxLen) * 4 * kHd, 29 + depth);
  std::vector<uint16_t> p = random_bf16(size_t(kGroup) * Mp * kMaxLen, 31 + C + depth);
  for (uint32_t l = 0; l < kGroup; ++l)          // P beyond the depth is exact +0.0
    for (uint32_t m = 0; m < Mp; ++m)
      std::fill(p.begin() + (size_t(l) * Mp + m) * kMaxLen + depth,
                p.begin() + (size_t(l) * Mp + m) * kMaxLen + kMaxLen, uint16_t(0));
  l0::Mem Q = upload(d.ctx, q), KC = upload(d.ctx, kc), VC = upload(d.ctx, vc), P = upload(d.ctx, p);
  const size_t s_pf = size_t(Mp) * kMaxLen;
  l0::Mem Spf(d.ctx, l0::MemKind::Device, size_t(kGroup) * s_pf * 4);
  const size_t o_pf = size_t(Mp) * kHd;
  l0::Mem Opf(d.ctx, l0::MemKind::Device, size_t(kGroup) * o_pf * 4);
  const uint16_t* qb = Q.as<uint16_t>() + size_t(j) * kGroup * kHd;
  const uint16_t* kb = KC.as<uint16_t>() + size_t(j) * kHd;
  const uint16_t* vb = VC.as<uint16_t>() + size_t(j) * kHd;

  GemmBatch qk_pf{C, kHd, depth, kGroup, 24 * kHd, 4 * kHd, kMaxLen, kHd, 0, s_pf};
  runtime::prefill::gemm_l0(d.cx, d.kc, qk_pf, qb, kb, Spf.as<float>(), true);
  GemmBatch pv_pf{C, npad256, kHd, kGroup, kMaxLen, 4 * kHd, kHd, s_pf, 0, o_pf};
  runtime::prefill::gemm_l0(d.cx, d.kc, pv_pf, P.as<uint16_t>(), vb, Opf.as<float>(), false);
  d.cx.wait();

  // The host reference over the same host operands: q row m of head l is
  // q[m][j*6 + l][:], the K / V cache row n of kv-head j is cache[n][j][:].
  const std::vector<float> s_got = download(d.ctx, Spf.ptr(), size_t(kGroup) * s_pf);
  const std::vector<float> o_got = download(d.ctx, Opf.ptr(), size_t(kGroup) * o_pf);
  Tol tq, tp;
  std::vector<double> ref(kHd), S(kHd);
  for (uint32_t l = 0; l < kGroup; ++l)
    for (uint32_t m : sample_rows(C)) {
      const uint16_t* qr = &q[size_t(m) * 24 * kHd + (size_t(j) * kGroup + l) * kHd];
      for (uint32_t n = 0; n < depth; ++n) {
        const uint16_t* kr = &kc[(size_t(n) * 4 + j) * kHd];
        double r = 0, s = 0;
        for (uint32_t e = 0; e < kHd; ++e) {
          const double x = double(f32(qr[e])) * f32(kr[e]);
          r += x;
          s += std::fabs(x);
        }
        tq.add(s_got[l * s_pf + size_t(m) * kMaxLen + n], r, s);
      }
      std::fill(ref.begin(), ref.end(), 0.0);
      std::fill(S.begin(), S.end(), 0.0);
      const uint16_t* pr = &p[(size_t(l) * Mp + m) * kMaxLen];
      for (uint32_t n = 0; n < depth; ++n) {
        const double pv = f32(pr[n]);
        const uint16_t* vr = &vc[(size_t(n) * 4 + j) * kHd];
        for (uint32_t e = 0; e < kHd; ++e) {
          const double x = pv * f32(vr[e]);
          ref[e] += x;
          S[e] += std::fabs(x);
        }
      }
      for (uint32_t e = 0; e < kHd; ++e) tp.add(o_got[l * o_pf + size_t(m) * kHd + e], ref[e], S[e]);
    }
  char name[96];
  std::snprintf(name, sizeof name, "QK^T C=%u depth=%u (transB)", C, depth);
  verdict(name, tq, kHd);
  std::snprintf(name, sizeof name, "PV   C=%u depth=%u (K padded %u)", C, depth, npad256);
  verdict(name, tp, npad256);

  // Determinism: the same launch twice is bitwise, for BOTH variants -- pf_gemm_T1
  // (transB, QK^T above) and pf_gemm_T0 (untransposed, every linear and this P·V call).
  // Each check is restricted to the rectangle gemm_l0 actually writes, never to the
  // full pitched buffer: an unwritten tail holds each allocation's own unrelated
  // leftover device memory, not kernel output, and comparing past what either launch
  // wrote would memcmp that leftover instead.

  // T1: Mp rows x npad256 columns per batch entry (qk_pf's N is padded to npad256, same
  // as PV's K padding above); ldc = kMaxLen leaves every column beyond npad256 untouched.
  l0::Mem S2(d.ctx, l0::MemKind::Device, size_t(kGroup) * s_pf * 4);
  runtime::prefill::gemm_l0(d.cx, d.kc, qk_pf, qb, kb, S2.as<float>(), true);
  d.cx.wait();
  const std::vector<float> s1 = download(d.ctx, Spf.ptr(), size_t(kGroup) * s_pf);
  const std::vector<float> s2 = download(d.ctx, S2.ptr(), size_t(kGroup) * s_pf);
  const Diff dd = compare(s1, s2, kGroup, Mp, npad256, s_pf, kMaxLen, s_pf, kMaxLen);
  CHECK_EQ(dd.words, size_t(0));
  CHECK_EQ(dd.sign_zero, size_t(0));

  // T0: Mp rows x kHd columns per batch entry, pitch kHd -- pv_pf's N = kHd is already a
  // multiple of 256, so gemm_l0 does not pad it and this is exactly Opf's full extent.
  l0::Mem O2(d.ctx, l0::MemKind::Device, size_t(kGroup) * o_pf * 4);
  runtime::prefill::gemm_l0(d.cx, d.kc, pv_pf, P.as<uint16_t>(), vb, O2.as<float>(), false);
  d.cx.wait();
  const std::vector<float> o1 = download(d.ctx, Opf.ptr(), size_t(kGroup) * o_pf);
  const std::vector<float> o2 = download(d.ctx, O2.ptr(), size_t(kGroup) * o_pf);
  const Diff od = compare(o1, o2, kGroup, Mp, kHd, o_pf, kHd, o_pf, kHd);
  CHECK_EQ(od.words, size_t(0));
  CHECK_EQ(od.sign_zero, size_t(0));
}
}  // namespace

int main() {
  Dev d;
  std::puts("pf_gemm vs a host fp64 reference over the logical rectangle (cos and max-ulp bars):");
  // Controller ruling: the brief's heterogeneous braced list of pairs will not deduce
  // under -Werror, so this is a typed array of the same five cells instead.
  const std::pair<uint32_t, uint32_t> shapes[] = {{5120, 16384}, {6144, 5120}, {5120, 34816},
                                                   {17408, 5120}, {5120, 14336}};
  for (const auto& [K, N] : shapes)
    for (uint32_t M : {2048u, 772u, 40u}) slab_case(d, M, K, N);
  for (uint32_t depth : {64u, 1000u, 4096u, 16384u})
    for (uint32_t C : {2048u, 40u}) attention_case(d, C, depth);
  std::puts("pf_gemm's fused SiLU epilogue vs the GEMM + pf_silu_mul pair, bitwise:");
  for (uint32_t M : {2048u, 772u}) silu_case(d, M);
  std::puts("pf_gemm_test OK");
  return 0;
}
