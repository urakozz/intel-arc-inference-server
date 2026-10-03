// The prefill path's runtime-`M` embedding gather and between-linear kernels
// (src/kernels/prefill/pf_embed.cl, pf_prep.cl) -- plan 6b Task 3.
//
// Three bars, and they are deliberately different bars:
//
//   1. **M = 1, bit-identical to the decode binary.** These kernels are decode's
//      with `M` moved from a `-D` to an argument, so at one row they must
//      produce the same bits, not merely close ones. `memcmp`, no tolerance.
//      `pf_silu_mul` folds ONE slice (ruling R1) where `prep_silu_mul_M1` folds
//      eight, so the decode side is driven with an eight-slice rectangle whose
//      slices 1..7 are `+0.0f`: `Sigma_s` then reduces to slice 0 exactly.
//      That is exact only while nothing in slice 0 is `-0.0f`
//      (`+0.0f + -0.0f == +0.0f` would flip a sign of zero and therefore a bf16
//      word), which `require_no_negative_zero` asserts before the comparison.
//   2. **M = 64 against the CPU reference.** `tests/kernels/prep_ref.h` is
//      already parametric in `M` and `tests/prefill/pf_ref.h` supplies the two
//      chains that genuinely differ (the S = 1 silu, the gather). Bit-exact for
//      everything that does not carry `exp`; the established 2 bf16 ulp bar for
//      `pf_silu_mul`'s random case, plus a `silu(30) = 30` case that is held
//      bit-exact and pins the rest of that chain.
//   3. **Row independence.** One chunk's rows must not leak into one another.
//      Run at M = 64, then re-run row 17's inputs alone at M = 1, and require
//      row 17's outputs identical. This is the property that makes a chunk a
//      batch of independent positions, and it is what an unmasked write or a
//      stride computed from the wrong `m_count` breaks.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "check.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "pf_harness.h"
#include "pf_ref.h"
#include "prep_ref.h"

namespace {
using pf_harness::Cmp;
using pf_harness::compare;
using pf_harness::Dev;
using pf_harness::download;
using pf_harness::random_bf16;
using pf_harness::random_f32;
using pf_harness::require;
using pf_harness::require_no_negative_zero;
using pf_harness::upload;

constexpr uint32_t kK = 5120;       // the residual row length (kHidden)
constexpr uint32_t kG = 20;         // FOLD_G / NORM_G
constexpr uint32_t kW = 20;         // NORM_WGS
constexpr uint32_t kEmbedRows = 64; // stand-in embedding table depth
constexpr uint32_t kCtrlWords = 32;

// One (fold, finish) pipeline run, in ONE in-order list -- the shape the
// runtime appends the pair in. Returns `resid` (rewritten in place), `sumsq`
// and `x_out`.
struct PrepOut {
  std::vector<uint16_t> resid, x;
  std::vector<float> sumsq;
};

// `pf` selects the prefill binaries (runtime M, `m_count` argument) over the
// decode ones (M in the binary, three arguments).
PrepOut run_prep_pair(Dev& d, bool pf, uint32_t M, uint32_t SP, const std::vector<float>& partials,
                      const std::vector<uint16_t>& resid_in, const std::vector<float>& norm_w) {
  l0::Mem pbuf = upload(d.ctx, d.imm, partials);
  l0::Mem rbuf = upload(d.ctx, d.imm, resid_in);
  l0::Mem wbuf = upload(d.ctx, d.imm, norm_w);
  l0::Mem sbuf(d.ctx, l0::MemKind::Device, size_t(kG) * M * sizeof(float));
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * kK * sizeof(uint16_t));

  l0::Module fold_mod(d.ctx, kernels::path(pf ? kernels::pf_res_fold_variant(kK, SP, kG)
                                              : kernels::prep_res_fold_variant(M, kK, SP, kG)));
  l0::Module fin_mod(d.ctx, kernels::path(pf ? kernels::pf_norm_finish_variant(kK, kG, kW)
                                             : kernels::prep_norm_finish_variant(M, kK, kG, kW)));
  l0::Kernel fold = fold_mod.kernel(pf ? "pf_res_fold" : "prep_res_fold");
  l0::Kernel fin = fin_mod.kernel(pf ? "pf_norm_finish" : "prep_norm_finish");
  fold.group_size(256);
  fin.group_size(256);
  fold.arg_ptr(0, pbuf.ptr());
  fold.arg_ptr(1, rbuf.ptr());
  fold.arg_ptr(2, sbuf.ptr());
  fin.arg_ptr(0, sbuf.ptr());
  fin.arg_ptr(1, rbuf.ptr());
  fin.arg_ptr(2, wbuf.ptr());
  fin.arg_ptr(3, xbuf.ptr());
  if (pf) {
    fold.arg<uint32_t>(3, M);
    fin.arg<uint32_t>(4, M);
  }
  d.run2(fold, kG, M, fin, kW, M);

  PrepOut out;
  out.resid.resize(size_t(M) * kK);
  out.x.resize(size_t(M) * kK);
  out.sumsq.resize(size_t(kG) * M);
  download(d.imm, out.resid, rbuf);
  download(d.imm, out.x, xbuf);
  download(d.imm, out.sumsq, sbuf);
  return out;
}

std::vector<uint16_t> run_silu(Dev& d, bool pf, uint32_t M, const std::vector<float>& partials) {
  const uint32_t N = prep_ref::kSiluN;
  l0::Mem pbuf = upload(d.ctx, d.imm, partials);
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * N * sizeof(uint16_t));
  l0::Module mod(d.ctx, kernels::path(pf ? kernels::pf_silu_mul_variant(17408)
                                         : kernels::prep_silu_mul_variant(M, 17408)));
  l0::Kernel k = mod.kernel(pf ? "pf_silu_mul" : "prep_silu_mul");
  k.group_size(256);
  k.arg_ptr(0, pbuf.ptr());
  k.arg_ptr(1, xbuf.ptr());
  if (pf) k.arg<uint32_t>(2, M);
  d.run(k, (N + 4095) / 4096, M);
  std::vector<uint16_t> x(size_t(M) * N);
  download(d.imm, x, xbuf);
  return x;
}

std::vector<uint16_t> run_embed(Dev& d, bool pf, uint32_t M, const std::vector<uint32_t>& ids,
                                const std::vector<uint16_t>& table) {
  l0::Mem tbuf = upload(d.ctx, d.imm, table);
  l0::Mem rbuf(d.ctx, l0::MemKind::Device, size_t(M) * kK * sizeof(uint16_t));
  l0::Module mod(d.ctx, kernels::path(pf ? kernels::pf_embed_gather_variant()
                                         : kernels::embed_gather_variant(M)));
  l0::Kernel k = mod.kernel(pf ? "pf_embed_gather" : "embed_gather");
  k.group_size(256);
  // Decode reads the id from the shared control block (its list is captured
  // once and replayed); prefill reads it from a device buffer, because a chunk
  // holds up to 2048 ids and Control::cur_token holds eight.
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, kCtrlWords * sizeof(uint32_t));
  l0::Mem ibuf = upload(d.ctx, d.imm, ids);
  if (pf) {
    k.arg_ptr(0, ibuf.ptr());
  } else {
    uint32_t* c = ctrl.as<uint32_t>();
    for (uint32_t i = 0; i < kCtrlWords; ++i) c[i] = 0;
    for (uint32_t m = 0; m < M; ++m) c[kernels::ctrl_index::kCurToken + m] = ids[m];
    k.arg_ptr(0, ctrl.ptr());
  }
  k.arg_ptr(1, tbuf.ptr());
  k.arg_ptr(2, rbuf.ptr());
  if (pf) k.arg<uint32_t>(3, M);
  d.run(k, 1, M);
  std::vector<uint16_t> resid(size_t(M) * kK);
  download(d.imm, resid, rbuf);
  return resid;
}

// ---------------------------------------------------------------------------

void case_prep_pair_identity(Dev& d, uint32_t SP) {
  constexpr uint32_t M = 1;
  const size_t np = SP ? size_t(SP) * M * kK : 1;
  std::vector<float> partials = random_f32(np, 100 + SP, 0.f, 1.f);
  std::vector<uint16_t> resid = random_bf16(size_t(M) * kK, 200 + SP, -1.f, 1.f);
  std::vector<float> norm_w = random_f32(kK, 300 + SP, 1.0f, 0.03f);

  const PrepOut dec = run_prep_pair(d, false, M, SP, partials, resid, norm_w);
  const PrepOut pre = run_prep_pair(d, true, M, SP, partials, resid, norm_w);
  CHECK(pre.resid == dec.resid);
  CHECK(pre.x == dec.x);
  CHECK(pre.sumsq == dec.sumsq);
  std::printf("pf_res_fold/pf_norm_finish M=1 SP=%u: resid, sumsq and x_out bit-identical to "
              "prep_res_fold_M1_K%u_SP%u_G%u + prep_norm_finish_M1_K%u_G%u_W%u\n",
              SP, kK, SP, kG, kK, kG, kW);
}

void case_prep_pair_ref(Dev& d, uint32_t M) {
  constexpr uint32_t SP = 1;   // the prefill path folds one slice (R1)
  std::vector<float> partials = random_f32(size_t(SP) * M * kK, 110, 0.f, 1.f);
  std::vector<uint16_t> resid = random_bf16(size_t(M) * kK, 210, -1.f, 1.f);
  std::vector<float> norm_w = random_f32(kK, 310, 1.0f, 0.03f);

  std::vector<uint16_t> resid_ref = resid, x_ref(size_t(M) * kK, 0);
  std::vector<float> sumsq_ref(size_t(kG) * M, 0.f);
  prep_ref::res_fold(partials.data(), resid_ref.data(), sumsq_ref.data(), M, kK, SP, kG);
  prep_ref::norm_finish(sumsq_ref.data(), resid_ref.data(), norm_w.data(), x_ref.data(), M, kK, kG);

  const PrepOut got = run_prep_pair(d, true, M, SP, partials, resid, norm_w);
  Cmp cr = compare(got.resid, resid_ref), cx = compare(got.x, x_ref);
  size_t sum_exact = 0;
  for (size_t i = 0; i < sumsq_ref.size(); ++i) sum_exact += got.sumsq[i] == sumsq_ref[i] ? 1 : 0;
  std::printf("pf_res_fold/pf_norm_finish M=%u SP=1 vs pf_ref: resid %zu/%zu exact (max %u ulp), "
              "sumsq %zu/%zu exact, x_out %zu/%zu exact (max %u ulp)  [bar: bit-exact]\n",
              M, cr.exact, cr.n, cr.max_ulp, sum_exact, sumsq_ref.size(), cx.exact, cx.n,
              cx.max_ulp);
  require(cr, 0, "pf_res_fold resid");
  require(cx, 0, "pf_norm_finish x_out");
  CHECK_EQ(sum_exact, sumsq_ref.size());

  // Row independence: row 17's inputs alone, at M = 1, must give row 17's
  // outputs. The `sumsq` rectangle is `[G][m_count]`, so a wrong `m_count`
  // would read another row's partial sums here and nowhere else.
  constexpr uint32_t kRow = 17;
  CHECK(kRow < M);
  std::vector<float> p1(partials.begin() + size_t(kRow) * kK,
                        partials.begin() + size_t(kRow + 1) * kK);
  std::vector<uint16_t> r1(resid.begin() + size_t(kRow) * kK,
                           resid.begin() + size_t(kRow + 1) * kK);
  const PrepOut one = run_prep_pair(d, true, 1, SP, p1, r1, norm_w);
  for (uint32_t k = 0; k < kK; ++k) {
    CHECK_EQ(one.resid[k], got.resid[size_t(kRow) * kK + k]);
    CHECK_EQ(one.x[k], got.x[size_t(kRow) * kK + k]);
  }
  for (uint32_t g = 0; g < kG; ++g) CHECK_EQ(one.sumsq[g], got.sumsq[size_t(g) * M + kRow]);
  std::printf("pf_res_fold/pf_norm_finish row independence: row %u of M=%u is bit-identical to "
              "the same row run alone\n", kRow, M);
}

void case_silu_identity(Dev& d) {
  constexpr uint32_t M = 1;
  const uint32_t S = prep_ref::kSiluS, FN = prep_ref::kSiluFusedN;
  // The prefill kernel's rectangle is one slice; the decode kernel's is eight,
  // with slices 1..7 exactly +0.0f so its ascending Sigma reduces to slice 0.
  std::vector<float> one = random_f32(size_t(M) * FN, 400, 0.f, 1.f);
  require_no_negative_zero(one, "pf_silu_mul slice 0");
  std::vector<float> eight(size_t(S) * M * FN, +0.0f);
  for (size_t i = 0; i < one.size(); ++i) eight[i] = one[i];

  const std::vector<uint16_t> pre = run_silu(d, true, M, one);
  const std::vector<uint16_t> dec = run_silu(d, false, M, eight);
  CHECK(pre == dec);
  std::printf("pf_silu_mul M=1: bit-identical to prep_silu_mul_M1 over an 8-slice +0.0f-padded "
              "rectangle (%zu elements)\n", pre.size());
}

// `exact_silu`: force every gate column to 30.0f, making silu(gate) provably
// 30.0f on host and device alike, so the whole chain is bit-exact.
void case_silu_ref(Dev& d, uint32_t M, bool exact_silu) {
  const uint32_t FN = prep_ref::kSiluFusedN, N = prep_ref::kSiluN;
  std::vector<float> partials = random_f32(size_t(M) * FN, exact_silu ? 411 : 410, 0.f, 1.f);
  if (exact_silu)
    for (size_t m = 0; m < M; ++m)
      for (size_t j = 0; j < FN; ++j)
        if ((j / 16) % 2 == 0) partials[m * FN + j] = 30.0f;

  std::vector<uint16_t> x_ref(size_t(M) * N, 0);
  pf_ref::silu_mul_s1(partials.data(), x_ref.data(), M);
  const std::vector<uint16_t> x_got = run_silu(d, true, M, partials);
  Cmp c = compare(x_got, x_ref);
  const uint32_t tol = exact_silu ? 0 : 2;
  std::printf("pf_silu_mul M=%u %s: %zu/%zu exact (max %u ulp, tol %u)\n", M,
              exact_silu ? "silu(30)=30 exact-case" : "random", c.exact, c.n, c.max_ulp, tol);
  require(c, tol, "pf_silu_mul x_out");

  if (!exact_silu) {
    constexpr uint32_t kRow = 17;
    CHECK(kRow < M);
    std::vector<float> p1(partials.begin() + size_t(kRow) * FN,
                          partials.begin() + size_t(kRow + 1) * FN);
    const std::vector<uint16_t> one = run_silu(d, true, 1, p1);
    for (uint32_t k = 0; k < N; ++k) CHECK_EQ(one[k], x_got[size_t(kRow) * N + k]);
    std::printf("pf_silu_mul row independence: row %u of M=%u is bit-identical to the same row "
                "run alone\n", kRow, M);
  }
}

void case_embed(Dev& d) {
  // A 64-row stand-in for the real [248320][5120] bf16 table (2.54 GB). That is
  // faithful because the kernel's only use of the vocabulary size is the range
  // guard: for an in-range id it touches exactly embed[row*5120 .. +5120).
  const std::vector<uint16_t> table = random_bf16(size_t(kEmbedRows) * kK, 800, -1.f, 1.f);

  const std::vector<uint32_t> id1{7};
  CHECK(run_embed(d, true, 1, id1, table) == run_embed(d, false, 1, id1, table));
  std::printf("pf_embed_gather M=1: bit-identical to embed_gather_M1 (id 7)\n");

  constexpr uint32_t M = 64;
  std::vector<uint32_t> ids(M);
  for (uint32_t m = 0; m < M; ++m) ids[m] = (m * 37u + 5u) % kEmbedRows;
  std::vector<uint16_t> ref(size_t(M) * kK, 0);
  pf_ref::embed_gather(ids.data(), table.data(), ref.data(), M, kK, kEmbedRows);
  const std::vector<uint16_t> got = run_embed(d, true, M, ids, table);
  CHECK(got == ref);
  std::printf("pf_embed_gather M=%u: %zu elements bit-exact vs pf_ref\n", M, ref.size());

  constexpr uint32_t kRow = 17;
  const std::vector<uint32_t> one_id{ids[kRow]};
  const std::vector<uint16_t> one = run_embed(d, true, 1, one_id, table);
  for (uint32_t k = 0; k < kK; ++k) CHECK_EQ(one[k], got[size_t(kRow) * kK + k]);
  std::printf("pf_embed_gather row independence: row %u of M=%u is bit-identical to the same row "
              "run alone\n", kRow, M);
}

}  // namespace

int main() {
  Dev d;
  case_embed(d);
  case_prep_pair_identity(d, 0);
  case_prep_pair_identity(d, 1);
  case_prep_pair_ref(d, 64);
  case_silu_identity(d);
  case_silu_ref(d, 64, false);
  case_silu_ref(d, 64, true);
  std::puts("pf_prep_test OK");
  return 0;
}
