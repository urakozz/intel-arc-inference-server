// `pf_ab_proj` (src/kernels/prefill/pf_gemv_bf16.cl) -- the GDN a||b projection
// at runtime `M` -- plan 6b Task 4.
//
// Four bars:
//
//   1. **M = 1, bit-identical to `gemv_bf16_M1_K5120_N128_C16_S16`**, the
//      binary `src/runtime/capture.cc` actually binds. The register tile
//      `acc[MT]` and the SLM tree `red[sg][i][lane]` are per-slot independent,
//      so slot 0 reduces exactly the terms decode's `M = 1` build reduces, in
//      exactly that order -- `memcmp`, no tolerance.
//   2. **M = 64 against `gemv_bf16_ref`**, at `tol_for(ref)` from
//      `tests/kernels/gemv_harness.h` (`1e-4 * max|ref| + 1e-5`, quoted rather
//      than re-invented).
//   3. **The ragged tile.** `M = 65` is one full `MT = 8` tile plus one live
//      lane out of eight. Rows 0..63 must be bit-identical to the `M = 64` run
//      and row 64 correct. This is the masking test: an unmasked `acc[]`
//      writeback scribbles seven rows past the end of `out` here and nowhere
//      else.
//   4. **Row independence.** `M = 64` with row 40's activations zeroed must
//      give an exactly-zero output row 40 -- the dead-lane activation load
//      reads row `m0` rather than out of bounds, and this is what catches it
//      reaching the wrong row.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "pf_harness.h"

namespace {
using pf_harness::Dev;
using pf_harness::download;
using pf_harness::upload;
// `random_bf16`, `max_abs_err` and `tol_for` are gemv_ref.h's / gemv_harness.h's
// -- the same generators and the same tolerance the decode GEMV tests use, so
// this test's numbers sit on their scale. (pf_harness has its own
// `random_bf16`; it is deliberately NOT pulled in here, because two identically
// named generators in one translation unit is exactly the ambiguity that makes
// a reader guess which bar a number was measured against.)

constexpr uint32_t kK = 5120;   // a||b's K (kHidden)
constexpr uint32_t kN = 128;    // a||b, zero-padded from 96
constexpr uint32_t kMt = 8;     // pf_gemv_bf16.cl's register tile

std::vector<float> run_pf_ab(Dev& d, const common::Bf16Tiled& w, const std::vector<uint16_t>& x,
                             uint32_t M) {
  const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(kN);   // {16, 16}
  l0::Mem wbuf = upload(d.ctx, d.imm, w.data);
  l0::Mem xbuf = upload(d.ctx, d.imm, x);
  l0::Mem obuf(d.ctx, l0::MemKind::Device, size_t(M) * kN * sizeof(float));
  // The output is filled with a sentinel first: bar 3 needs "row 64 was
  // written" to be distinguishable from "row 64 was left alone".
  d.imm.fill(obuf.ptr(), 0x7FC00000u, obuf.size());   // fp32 qNaN
  l0::Module mod(d.ctx, kernels::path(kernels::pf_ab_proj_variant()));
  l0::Kernel k = mod.kernel("pf_ab_proj");
  k.group_size(t.cols * t.ksplit);
  k.arg_ptr(0, wbuf.ptr());
  k.arg_ptr(1, xbuf.ptr());
  k.arg_ptr(2, obuf.ptr());
  k.arg<uint32_t>(3, M);
  d.run(k, kN / t.cols, (M + kMt - 1) / kMt);
  std::vector<float> out(size_t(M) * kN);
  download(d.imm, out, obuf);
  return out;
}

std::vector<float> run_decode_ab(Dev& d, const common::Bf16Tiled& w,
                                 const std::vector<uint16_t>& x) {
  const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(kN);
  GemvResult r = run_gemv_bf16(d.ctx, d.q, d.fence, w, x, 1, 0, t);
  return r.out;
}

}  // namespace

int main() {
  Dev d;
  const std::vector<uint16_t> w_rm = random_bf16(size_t(kN) * kK, 7 + kN, -0.05f, 0.05f);  // [N][K]
  const common::Bf16Tiled w = common::Bf16Tiled::from_rowmajor(w_rm.data(), kK, kN);

  // 1. M = 1 vs the binary the runtime binds.
  {
    const std::vector<uint16_t> x = random_bf16(kK, 11, -1.f, 1.f);
    const std::vector<float> pre = run_pf_ab(d, w, x, 1);
    const std::vector<float> dec = run_decode_ab(d, w, x);
    CHECK_EQ(pre.size(), dec.size());
    CHECK(std::memcmp(pre.data(), dec.data(), pre.size() * sizeof(float)) == 0);
    std::printf("pf_ab_proj M=1: %zu fp32 outputs bit-identical to "
                "gemv_bf16_M1_K%u_N%u_C16_S16\n", pre.size(), kK, kN);
  }

  // 2/4. M = 64 against the reference, and row independence.
  constexpr uint32_t M = 64, kZeroRow = 40;
  std::vector<uint16_t> x = random_bf16(size_t(M) * kK, 13, -1.f, 1.f);
  for (uint32_t k = 0; k < kK; ++k) x[size_t(kZeroRow) * kK + k] = 0;
  std::vector<float> ref;
  gemv_bf16_ref(w_rm.data(), kK, kN, x, M, ref);
  const std::vector<float> got = run_pf_ab(d, w, x, M);
  const double err = max_abs_err(got, ref), tol = tol_for(ref);
  std::printf("pf_ab_proj M=%u K=%u N=%u: max_abs_err=%.3g tol=%.3g (%u work-groups)\n", M, kK, kN,
              err, tol, (kN / 16) * ((M + kMt - 1) / kMt));
  CHECK(err <= tol);
  for (uint32_t n = 0; n < kN; ++n) CHECK_EQ(got[size_t(kZeroRow) * kN + n], 0.0f);
  std::printf("pf_ab_proj row independence: row %u (zeroed activations) is exactly zero\n",
              kZeroRow);

  // 3. The ragged tile: one full MT tile plus one live lane out of eight.
  {
    constexpr uint32_t M65 = 65;
    std::vector<uint16_t> x65(size_t(M65) * kK);
    std::memcpy(x65.data(), x.data(), x.size() * sizeof(uint16_t));
    const std::vector<uint16_t> tail = random_bf16(kK, 17, -1.f, 1.f);
    std::memcpy(x65.data() + size_t(M) * kK, tail.data(), tail.size() * sizeof(uint16_t));
    const std::vector<float> got65 = run_pf_ab(d, w, x65, M65);
    CHECK(std::memcmp(got65.data(), got.data(), got.size() * sizeof(float)) == 0);
    std::vector<float> ref65;
    gemv_bf16_ref(w_rm.data(), kK, kN, x65, M65, ref65);
    double tail_err = 0;
    for (uint32_t n = 0; n < kN; ++n) {
      const double e = double(got65[size_t(M) * kN + n]) - double(ref65[size_t(M) * kN + n]);
      tail_err = e < 0 ? (tail_err > -e ? tail_err : -e) : (tail_err > e ? tail_err : e);
    }
    std::printf("pf_ab_proj M=65 ragged tile: rows 0..63 bit-identical to the M=64 run, "
                "row 64 max_abs_err=%.3g tol=%.3g\n", tail_err, tol_for(ref65));
    CHECK(tail_err <= tol_for(ref65));
  }

  std::puts("pf_gemv_test OK");
  return 0;
}
