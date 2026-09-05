// `pf_gated_head` and `pf_attn_prep` (src/kernels/prefill/) -- plan 6b Task 5's
// two kernels, at runtime `M` and S = 1.
//
// What is NOT here, and why: plan 6b Task 5 also builds an L1 attention route
// (`attn_prep_l1` / `attn_l1` over decode's unmodified `attn_decode` /
// `attn_reduce` recompiled at `M = 64`, ruling R5). Controller ruling **A14**
// (interfaces.md, 2026-09-05) pivots L3 to **composed** attention -- QK^T and
// PV through the inherited sycl-tla GEMM with one bandwidth-bound softmax of
// ours -- which retires that route before it is written, and its host half
// needs `runtime::prefill::Context` (plan 6b Task 2) which does not exist yet.
// The two kernels below are the half of Task 5 that survives the pivot
// unchanged: `pf_attn_prep` writes the KV cache and the normed, RoPE'd q that
// EITHER attention consumes.
//
// The bars are attn.cl's own (tests/kernels/attn_test.cc): `kv_k`, `kv_v`,
// `attn_gate` and `attn_q` are held **bit-exact**, because the only op in this
// chain that OpenCL does not require to be correctly rounded is `exp` and there
// is none here. `pf_gated_head` ends in a `silu`, so its random case takes
// prep_test's established **2 bf16 ulp** bar and a `silu(30) = 30` case pins
// the rest of that chain bit-exactly.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "attn_ref.h"
#include "check.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "pf_harness.h"
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

constexpr uint32_t kMaxLen = 4160;     // >= 4033 + 64, the high-pos case
constexpr uint32_t kCtrlWords = 32;

// --------------------------------------------------------------------------
// pf_gated_head

std::vector<uint16_t> run_gated(Dev& d, bool pf, uint32_t M, const std::vector<float>& qkvz,
                                const std::vector<float>& gdn_o,
                                const std::vector<uint16_t>& gated_w) {
  const uint32_t H = prep_ref::kGatedHeads, ON = prep_ref::kGatedOutN;
  l0::Mem qbuf = upload(d.ctx, d.imm, qkvz);
  l0::Mem obuf = upload(d.ctx, d.imm, gdn_o);
  l0::Mem wbuf = upload(d.ctx, d.imm, gated_w);
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * ON * sizeof(uint16_t));
  l0::Module mod(d.ctx, kernels::path(pf ? kernels::pf_gated_head_variant()
                                         : kernels::prep_gated_head_variant(M)));
  l0::Kernel k = mod.kernel(pf ? "pf_gated_head" : "prep_gated_head");
  k.group_size(128);
  k.arg_ptr(0, qbuf.ptr());
  k.arg_ptr(1, obuf.ptr());
  k.arg_ptr(2, wbuf.ptr());
  k.arg_ptr(3, xbuf.ptr());
  if (pf) k.arg<uint32_t>(4, M);
  d.run(k, H, M);
  std::vector<uint16_t> x(size_t(M) * ON);
  download(d.imm, x, xbuf);
  return x;
}

// `exact_silu`: z = 30.0f everywhere, which makes silu(z) provably 30.0f on
// host and device alike and the whole chain (o_b -> var tree -> n_b -> t_b)
// observable exactly.
void case_gated_head(Dev& d, uint32_t M, bool exact_silu, bool identity) {
  const uint32_t QN = prep_ref::kQkvzN, H = prep_ref::kGatedHeads;
  const uint32_t D = prep_ref::kHeadDim, ON = prep_ref::kGatedOutN;
  std::vector<float> qkvz = random_f32(size_t(M) * QN, exact_silu ? 501 : 500, 0.f, 1.f);
  if (exact_silu)
    for (size_t i = 0; i < qkvz.size(); ++i) qkvz[i] = 30.0f;
  const std::vector<float> gdn_o = random_f32(size_t(M) * H * D, 600, 0.f, 1.f);
  const std::vector<uint16_t> gated_w = random_bf16(D, 700, -0.5f, 0.5f);   // plain w, bf16

  const std::vector<uint16_t> got = run_gated(d, true, M, qkvz, gdn_o, gated_w);

  if (identity) {
    // GATED_S is 1 on BOTH sides -- qkv||z runs unsplit in decode too -- so no
    // slice padding is needed for this comparison; the rectangles are the same
    // shape and the only difference is where `M` came from.
    const std::vector<uint16_t> dec = run_gated(d, false, M, qkvz, gdn_o, gated_w);
    CHECK(got == dec);
    std::printf("pf_gated_head M=%u: %zu bf16 outputs bit-identical to prep_gated_head_M%u\n", M,
                got.size(), M);
    return;
  }

  std::vector<uint16_t> ref(size_t(M) * ON, 0);
  prep_ref::gated_head(qkvz.data(), gdn_o.data(), gated_w.data(), ref.data(), M);
  Cmp c = compare(got, ref);
  const uint32_t tol = exact_silu ? 0 : 2;
  std::printf("pf_gated_head M=%u %s: %zu/%zu exact (max %u ulp, tol %u)\n", M,
              exact_silu ? "silu(30)=30 exact-case" : "random", c.exact, c.n, c.max_ulp, tol);
  require(c, tol, "pf_gated_head x_out");

  if (!exact_silu) {
    constexpr uint32_t kRow = 17;
    CHECK(kRow < M);
    const std::vector<float> q1(qkvz.begin() + size_t(kRow) * QN,
                                qkvz.begin() + size_t(kRow + 1) * QN);
    const std::vector<float> o1(gdn_o.begin() + size_t(kRow) * H * D,
                                gdn_o.begin() + size_t(kRow + 1) * H * D);
    const std::vector<uint16_t> one = run_gated(d, true, 1, q1, o1, gated_w);
    for (uint32_t i = 0; i < ON; ++i) CHECK_EQ(one[i], got[size_t(kRow) * ON + i]);
    std::printf("pf_gated_head row independence: row %u of M=%u is bit-identical to the same row "
                "run alone\n", kRow, M);
  }
}

// --------------------------------------------------------------------------
// pf_attn_prep

struct PrepOut {
  std::vector<float> q, gate;
  std::vector<uint16_t> kv_k, kv_v;
};

PrepOut run_attn_prep(Dev& d, bool pf, uint32_t pos, uint32_t n_act, uint32_t M,
                      const std::vector<float>& partials, const std::vector<float>& fa_small,
                      const std::vector<float>& rope) {
  const uint32_t QH = attn_ref::kQHeads, KVH = attn_ref::kKvHeads, HD = attn_ref::kHeadDim;
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, kCtrlWords * sizeof(uint32_t));
  uint32_t* c = ctrl.as<uint32_t>();
  for (uint32_t i = 0; i < kCtrlWords; ++i) c[i] = 0;
  c[kernels::ctrl_index::kPos] = pos;
  c[kernels::ctrl_index::kNActive] = n_act;

  l0::Mem pbuf = upload(d.ctx, d.imm, partials);
  l0::Mem sbuf = upload(d.ctx, d.imm, fa_small);
  l0::Mem rbuf = upload(d.ctx, d.imm, rope);
  l0::Mem qbuf(d.ctx, l0::MemKind::Device, size_t(M) * QH * HD * sizeof(float));
  l0::Mem gbuf(d.ctx, l0::MemKind::Device, size_t(M) * QH * HD * sizeof(float));
  l0::Mem kbuf(d.ctx, l0::MemKind::Device, size_t(kMaxLen) * KVH * HD * sizeof(uint16_t));
  l0::Mem vbuf(d.ctx, l0::MemKind::Device, size_t(kMaxLen) * KVH * HD * sizeof(uint16_t));
  d.imm.fill(qbuf.ptr(), 0u, qbuf.size());
  d.imm.fill(gbuf.ptr(), 0u, gbuf.size());
  d.imm.fill(kbuf.ptr(), 0u, kbuf.size());
  d.imm.fill(vbuf.ptr(), 0u, vbuf.size());

  l0::Module mod(d.ctx, kernels::path(pf ? kernels::pf_attn_prep_variant()
                                         : kernels::attn_prep_variant(M)));
  l0::Kernel k = mod.kernel(pf ? "pf_attn_prep" : "attn_prep");
  k.group_size(256);
  k.arg_ptr(0, ctrl.ptr());
  k.arg_ptr(1, pbuf.ptr());
  k.arg_ptr(2, sbuf.ptr());
  k.arg_ptr(3, rbuf.ptr());
  k.arg_ptr(4, qbuf.ptr());
  k.arg_ptr(5, gbuf.ptr());
  k.arg_ptr(6, kbuf.ptr());
  k.arg_ptr(7, vbuf.ptr());
  d.run(k, QH + KVH, M);

  PrepOut out;
  out.q.resize(size_t(M) * QH * HD);
  out.gate.resize(size_t(M) * QH * HD);
  out.kv_k.resize(size_t(kMaxLen) * KVH * HD);
  out.kv_v.resize(size_t(kMaxLen) * KVH * HD);
  download(d.imm, out.q, qbuf);
  download(d.imm, out.gate, gbuf);
  download(d.imm, out.kv_k, kbuf);
  download(d.imm, out.kv_v, vbuf);
  return out;
}

// The RoPE table the loader builds: cos at [p][0][i], sin at [p][1][i], angles
// computed in double (docs/13-loader.md). The values here are a stand-in with
// the same shape and the same |cos|,|sin| <= 1 range; nothing in this kernel
// depends on them being the model's.
std::vector<float> rope_table(uint32_t max_len) {
  std::vector<float> t(size_t(max_len) * 2 * attn_ref::kRotHalf);
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t i = 0; i < attn_ref::kRotHalf; ++i) {
      const double theta = double(p) / std::pow(10000.0, double(2 * i) / 64.0);
      t[(size_t(p) * 2 + 0) * attn_ref::kRotHalf + i] = float(std::cos(theta));
      t[(size_t(p) * 2 + 1) * attn_ref::kRotHalf + i] = float(std::sin(theta));
    }
  return t;
}

void case_attn_prep(Dev& d, uint32_t pos, uint32_t M, bool identity) {
  const uint32_t QN = attn_ref::kQkvN, S = attn_ref::kQkvS;
  // The prefill kernel folds ONE slice (ruling R1) and the decode binary and
  // the reference fold two, so the rectangle is built at S = 2 with slice 1
  // exactly +0.0f: `Sigma_s` then reduces to slice 0 exactly. That is exact
  // only while nothing in slice 0 is -0.0f (`+0.0f + -0.0f == +0.0f` would flip
  // a sign of zero and therefore a bf16 word), which is asserted below.
  std::vector<float> slice0 = random_f32(size_t(M) * QN, 900 + pos, 0.f, 1.f);
  require_no_negative_zero(slice0, "pf_attn_prep slice 0");
  std::vector<float> partials(size_t(S) * M * QN, +0.0f);
  for (size_t i = 0; i < slice0.size(); ++i) partials[i] = slice0[i];

  // The FA small block, flat fp32: q_norm's (1 + w) at kQNormOff, k_norm's at
  // kKNormOff. The loader bakes the +1 in, so these are values around 1.0.
  const std::vector<float> fa_small =
      random_f32(attn_ref::kKNormOff + attn_ref::kHeadDim, 901, 1.0f, 0.03f);
  const std::vector<float> rope = rope_table(kMaxLen);

  const PrepOut got = run_attn_prep(d, true, pos, M, M, partials, fa_small, rope);

  if (identity) {
    const PrepOut dec = run_attn_prep(d, false, pos, M, M, partials, fa_small, rope);
    CHECK(got.q == dec.q);
    CHECK(got.gate == dec.gate);
    CHECK(got.kv_k == dec.kv_k);
    CHECK(got.kv_v == dec.kv_v);
    std::printf("pf_attn_prep pos=%u n_act=%u: attn_q, attn_gate, kv_k and kv_v bit-identical to "
                "attn_prep_M%u over a 2-slice +0.0f-padded rectangle\n", pos, M, M);
    return;
  }

  PrepOut ref;
  ref.q.assign(got.q.size(), 0.f);
  ref.gate.assign(got.gate.size(), 0.f);
  ref.kv_k.assign(got.kv_k.size(), 0);
  ref.kv_v.assign(got.kv_v.size(), 0);
  attn_ref::prep(pos, M, M, partials.data(), fa_small.data(), rope.data(), ref.q.data(),
                 ref.gate.data(), ref.kv_k.data(), ref.kv_v.data());
  CHECK(got.q == ref.q);
  CHECK(got.gate == ref.gate);
  CHECK(got.kv_k == ref.kv_k);
  CHECK(got.kv_v == ref.kv_v);
  std::printf("pf_attn_prep pos=%u n_act=%u vs attn_ref::prep: attn_q (%zu), attn_gate (%zu) and "
              "the KV cache (%zu each) bit-exact\n", pos, M, ref.q.size(), ref.gate.size(),
              ref.kv_k.size());

  // Row independence: position `pos + 17`'s row, run alone at that absolute
  // position, must give the same q, gate and cache rows.
  constexpr uint32_t kRow = 17;
  CHECK(kRow < M);
  const std::vector<float> one_slice(slice0.begin() + size_t(kRow) * QN,
                                     slice0.begin() + size_t(kRow + 1) * QN);
  std::vector<float> one_p(size_t(S) * QN, +0.0f);
  for (size_t i = 0; i < one_slice.size(); ++i) one_p[i] = one_slice[i];
  const PrepOut one = run_attn_prep(d, true, pos + kRow, 1, 1, one_p, fa_small, rope);
  const uint32_t QH = attn_ref::kQHeads, KVH = attn_ref::kKvHeads, HD = attn_ref::kHeadDim;
  for (uint32_t i = 0; i < QH * HD; ++i) {
    CHECK_EQ(one.q[i], got.q[size_t(kRow) * QH * HD + i]);
    CHECK_EQ(one.gate[i], got.gate[size_t(kRow) * QH * HD + i]);
  }
  for (uint32_t i = 0; i < KVH * HD; ++i) {
    const size_t slot = size_t(pos + kRow) * KVH * HD + i;
    CHECK_EQ(one.kv_k[slot], got.kv_k[slot]);
    CHECK_EQ(one.kv_v[slot], got.kv_v[slot]);
  }
  std::printf("pf_attn_prep row independence: position %u of a %u-row chunk is bit-identical to "
              "the same row run alone\n", pos + kRow, M);
}

}  // namespace

int main() {
  Dev d;
  case_gated_head(d, 1, false, /*identity=*/true);
  case_gated_head(d, 64, false, /*identity=*/false);
  case_gated_head(d, 64, true, /*identity=*/false);
  case_attn_prep(d, 0, 1, /*identity=*/true);
  case_attn_prep(d, 0, 64, /*identity=*/false);
  // pos 4033: the RoPE table is read at a high index and the chunk straddles a
  // 64-position block edge, which is where an absolute-position bug hides.
  case_attn_prep(d, 4033, 64, /*identity=*/false);
  std::puts("pf_attn_test OK");
  return 0;
}
