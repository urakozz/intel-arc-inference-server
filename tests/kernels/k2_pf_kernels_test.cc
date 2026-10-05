// Spec 18c (K1 for the prefill): K2-Horizon's prefill kernels on the card, at K2's real shapes,
// against tests/kernels/k2_pf_ref.h / k2_ref.h and against the dense prefill GEMMs the grouped
// ones are built from. Needs a B70, no checkpoint (synthetic weights). Every binary is the one
// runtime/k2/k2_prefill.cc binds (kernels::k2 names). Written blind on the Mac (box queue row).
//
//   1. the slab dequant at N 9280 (a whole slab and the 256-column tail: exact, its padding
//      zero) and the linear through pf_gemm (rows within cosine 0.99999 of fp32 host sums; the
//      partials row's padding columns exactly 0);
//   2. the grouped norm: pf_res_fold (SP 1, m_count kPfC) + k2_norm_finish (M kPfC) - resid
//      and xn bit-exact against prep_ref + k2_ref (no exp; correctly rounded sqrt / divide);
//   3. k2_attn_prep at M kPfC / S 1, both builds (dense with v; MoVA without) at pos 37 over a
//      C = 37 chunk - q, gate, k, v bit-exact;
//   4. the routers over the chunk: MoVA's k2_route at pitch 9472 from column 9216, the MoE
//      router GEMV (each sampled row bitwise decode's gemv_bf16 of that row) and decode's
//      k2_route binary on grid (1, C) - ids exact, weights within 1 bf16 ulp;
//   5. the MoE block over C = 64: sort / gather exact; the dequantised blocks exact; GROUPED ==
//      DENSE, bitwise, per (token, slot): gate||up (both batches) against pf_gemm_T0_SILU over
//      the chunk, down against rne(pf_gemm_T0) over the sorted h, for the most loaded expert,
//      expert 0 when routed, the shared expert; the combine exactly k2_pf_ref's; Review Focus
//      2: the chunk REVERSED leaves every (token, slot)'s h and y and every token's residual
//      bitwise unchanged; replay bitwise;
//   6. MoVA over C = 64: the same, the value experts plain (no SiLU epilogue), the combine into
//      V at pos + t exactly k2_pf_ref's; Review Focus 1 at kernel level: decode's k2_mova_value
//      on the same x and route row gives a V row within cosine 0.9999 (the GEMMs' sum orders
//      differ, not the rule);
//   7. the flash attention, Review Focus 3 (K1): the ungated _O builds against fp64 (default)
//      and against the eager reference's rounding points (EAGER) at (pos, C) = (0, 1), (0, 37),
//      (0, 2048), (2011, 37) (a tail chunk), (2048, 2048), (30000, 2048), (60000, 2048) and
//      (4095, 1) (one row at depth 4096) - sampled rows x every head, cosine >= 0.99999
//      (default vs fp64) / 0.9999 (eager vs its reference) / 0.999 (eager vs fp64); the gated
//      production build within 1 bf16 ulp of k2_ref::attn_gate over the _O output, gates on
//      both sides of softplus's threshold;
//   8. replay of the flash attention bitwise.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/k2_kernels.h"
#include "kernels/k2_pf_ref.h"
#include "kernels/k2_ref.h"
#include "kernels/kernels.h"
#include "kernels/pf_moe_ref.h"
#include "kernels/prefill/pf_kernels.h"
#include "kernels/prep_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/k2_repack.h"
#include "model/k2_horizon.h"
#include "runtime/control.h"
#include "runtime/k2/k2_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"

namespace {

namespace kk = kernels::k2;
using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;
using runtime::prefill::arg_val;
using runtime::prefill::KernelArg;
using runtime::prefill::PtrArg;

constexpr uint32_t kC = kk::kPfC;
const model::K2Desc& D = model::k2();

int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

// A fast deterministic generator (the 60k-position caches are 127M values).
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint32_t u32() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return uint32_t(s >> 16);
  }
  float uni(float lo, float hi) { return lo + (hi - lo) * float(u32() & 0xFFFFFF) / float(0x1000000); }
};
std::vector<uint16_t> rbf16(size_t n, float lo, float hi, uint64_t seed) {
  Rng r(seed);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(r.uni(lo, hi));
  return v;
}
std::vector<float> rf32(size_t n, float lo, float hi, uint64_t seed) {
  Rng r(seed);
  std::vector<float> v(n);
  for (float& e : v) e = r.uni(lo, hi);
  return v;
}
std::vector<uint32_t> rblocks(size_t words, uint64_t seed) {
  std::vector<uint32_t> v(words);
  Rng r(seed);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = r.u32();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(r.uni(0.01f, 0.05f))) |
                      (uint32_t(common::f32_to_f16(r.uni(0.01f, 0.05f))) << 16);
  }
  return v;
}
double cosine(const double* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += double(b[i]) * b[i];
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}
double cosine_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = f32(a[i]), y = f32(b[i]);
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}

struct Dev {
  l0::Context ctx{0};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  runtime::prefill::Context cx{ctx};
  runtime::prefill::KernelCache kc{ctx};
  template <class T>
  l0::Mem up(const std::vector<T>& v) {
    l0::Mem m(ctx, l0::MemKind::Device, std::max<size_t>(v.size() * sizeof(T), 4));
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
    return m;
  }
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.fill(m.ptr(), 0u, bytes);
    return m;
  }
  template <class T>
  std::vector<T> rd(const l0::Mem& m, size_t n, size_t off_elems = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const T*>(m.ptr()) + off_elems, n * sizeof(T));
    return v;
  }
  void go(const std::string& v, const char* entry, uint32_t gx, uint32_t gy, uint32_t gz,
          std::initializer_list<KernelArg> args) {
    cx.launch(kc(v, entry), gx, gy, gz, args);
    cx.wait();
  }
};

// ---- 1. the slab dequant and the linear ---------------------------------------------------
void slab_and_linear(Dev& dv) {
  const uint32_t K = D.hidden, N = D.attn_sparse_n(), C = 37, ld = runtime::k2::pf_ld(N);
  Rng r(1);
  std::vector<uint32_t> words(size_t(K / 8) * N);
  for (uint32_t& w : words) w = r.u32();
  std::vector<uint16_t> sc(size_t(K / 64) * N);
  for (uint16_t& s : sc) s = common::f32_to_f16(r.uni(0.01f, 0.05f));
  l0::Mem dw = dv.up(words), ds = dv.up(sc), slab = dv.zeros(size_t(K) * kk::kPfSlab * 2);
  const std::string v = kk::pf_slab_variant(K, N);
  for (uint32_t n0 : {0u, 9216u}) {
    const uint32_t ns = runtime::k2::pf_slab_width(N, n0);
    dv.go(v, "k2_pf_dequant_slab", ns / 16, K / 64, 1,
          {PtrArg(dw.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0), arg_val(ns)});
    CHECK(dv.rd<uint16_t>(slab, size_t(K) * ns) == k2_pf_ref::dequant_slab(words.data(), sc.data(), K, N, n0, ns));
  }
  // The linear: x [C][K] through every slab into partials [kC][ld].
  const std::vector<uint16_t> x = rbf16(size_t(kC) * K, -1.f, 1.f, 2);
  l0::Mem dx = dv.up(x), part = dv.up(std::vector<float>(size_t(kC) * ld, 7.0f));
  for (uint32_t n0 = 0; n0 < N;) {
    const uint32_t ns = runtime::k2::pf_slab_width(N, n0);
    dv.cx.launch(dv.kc(v, "k2_pf_dequant_slab"), ns / 16, K / 64, 1,
                 {PtrArg(dw.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0), arg_val(ns)});
    const runtime::prefill::GemmBatch gb{C, K, ns, 1, K, ns, ld, 0, 0, 0};
    runtime::prefill::gemm_l0(dv.cx, dv.kc, gb, dx.as<uint16_t>(), slab.as<uint16_t>(), part.as<float>() + n0, false);
    n0 += ns;
  }
  dv.cx.wait();
  const std::vector<float> got = dv.rd<float>(part, size_t(C) * ld);
  const std::vector<uint16_t> full = k2_pf_ref::dequant_slab(words.data(), sc.data(), K, N, 0, ld);
  double worst = 1.0;
  for (uint32_t m = 0; m < C; ++m) {
    std::vector<double> want(N, 0.0);
    for (uint32_t k = 0; k < K; ++k) {
      const double xv = f32(x[size_t(m) * K + k]);
      for (uint32_t n = 0; n < N; ++n) want[n] += xv * f32(full[size_t(k) * ld + n]);
    }
    worst = std::min(worst, cosine(want.data(), got.data() + size_t(m) * ld, N));
    for (uint32_t n = N; n < ld; ++n) CHECK_EQ(got[size_t(m) * ld + n], 0.0f);
  }
  CHECK(worst >= 0.99999);
  std::printf("1. slab dequant (whole + 256-column tail) exact; linear 2560 x 9280 rows worst cosine %.7f, "
              "padding columns 0\n", worst);
}

// ---- 2. the grouped norm -------------------------------------------------------------------
void norm(Dev& dv) {
  const uint32_t K = D.hidden, C = 37, G = kk::kNormG;
  std::vector<uint16_t> resid = rbf16(size_t(kC) * K, -1.f, 1.f, 3);
  for (uint32_t k = 0; k < K / 2; ++k) resid[k] = rne(f32(resid[k]) * 0.01f);   // unequal groups
  const std::vector<float> part = rf32(size_t(kC) * K, -0.5f, 0.5f, 4);
  std::vector<float> w(K);
  for (uint32_t k = 0; k < K; ++k) w[k] = rf(0.3f + float(k % 11) * 0.1f);
  l0::Mem dr = dv.up(resid), dp = dv.up(part), dw = dv.up(w), ss = dv.zeros(size_t(G) * kC * 4),
          xn = dv.zeros(size_t(kC) * K * 2);
  dv.go(kernels::pf_res_fold_variant(K, 1, G), "pf_res_fold", G, C, 1,
        {PtrArg(dp.ptr()), PtrArg(dr.ptr()), PtrArg(ss.ptr()), arg_val(kC)});
  dv.go(kk::pf_norm_variant(K, G, kk::kNormW, D.norm_groups), "k2_norm_finish", kk::kNormW, C, 1,
        {PtrArg(ss.ptr()), PtrArg(dr.ptr()), PtrArg(dw.ptr()), PtrArg(xn.ptr())});
  std::vector<uint16_t> r_ref(resid.begin(), resid.begin() + size_t(C) * K), x_ref(size_t(C) * K);
  std::vector<float> ss_ref(size_t(G) * C);
  prep_ref::res_fold(part.data(), r_ref.data(), ss_ref.data(), C, K, 1, G);
  k2_ref::norm_finish(ss_ref.data(), r_ref.data(), w.data(), x_ref.data(), C, K, G, D.norm_groups);
  CHECK(dv.rd<uint16_t>(dr, size_t(C) * K) == r_ref);
  CHECK(dv.rd<uint16_t>(xn, size_t(C) * K) == x_ref);
  std::printf("2. grouped norm over %u rows: resid and xn bit-exact\n", C);
}

// ---- 3. the attention prep -----------------------------------------------------------------
void attn_prep(Dev& dv) {
  const uint32_t C = 37, pos = 37, hd = 128;
  const std::vector<float> rope = loader::k2_rope_table(D, 256);
  l0::Mem drope = dv.up(rope);
  l0::Mem ctrl(dv.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  std::memset(ctrl.ptr(), 0, ctrl.size());
  ctrl.as<runtime::Control>()->pos = pos;
  ctrl.as<runtime::Control>()->n_active = C;
  for (int dense = 1; dense >= 0; --dense) {
    const uint32_t ld = runtime::k2::pf_ld(dense ? D.attn_dense_n() : D.attn_sparse_n());
    const std::vector<float> part = rf32(size_t(kC) * ld, -1.f, 1.f, 10 + dense);
    l0::Mem dp = dv.up(part), aq = dv.zeros(size_t(kC) * D.q_n() * 4), ag = dv.zeros(size_t(kC) * D.q_n() * 4);
    l0::Mem kc = dv.zeros(size_t(256) * D.kv_n() * 2), vc = dv.zeros(size_t(256) * D.kv_n() * 2);
    dv.go(kk::pf_attn_prep_variant(ld, D.q_heads, D.kv_heads, dense), "k2_attn_prep", D.q_heads + D.kv_heads, C, 1,
          {PtrArg(ctrl.ptr()), PtrArg(dp.ptr()), PtrArg(drope.ptr()), PtrArg(aq.ptr()), PtrArg(ag.ptr()),
           PtrArg(kc.ptr()), PtrArg(vc.ptr())});
    const std::vector<float> q = dv.rd<float>(aq, size_t(C) * D.q_n()), g = dv.rd<float>(ag, size_t(C) * D.q_n());
    const std::vector<uint16_t> kr = dv.rd<uint16_t>(kc, size_t(256) * D.kv_n()), vr = dv.rd<uint16_t>(vc, size_t(256) * D.kv_n());
    for (uint32_t m = 0; m < C; ++m) {
      const k2_ref::Prep want = k2_ref::attn_prep(part.data(), kC, m, ld, 1, rope.data() + size_t(pos + m) * hd,
                                                  D.q_heads, D.kv_heads, hd, dense);
      CHECK(std::equal(want.q.begin(), want.q.end(), q.begin() + size_t(m) * D.q_n()));
      CHECK(std::equal(want.gate.begin(), want.gate.end(), g.begin() + size_t(m) * D.q_n()));
      CHECK(std::equal(want.k.begin(), want.k.end(), kr.begin() + size_t(pos + m) * D.kv_n()));
      if (dense) CHECK(std::equal(want.v.begin(), want.v.end(), vr.begin() + size_t(pos + m) * D.kv_n()));
    }
  }
  std::printf("3. attention prep (dense with v, MoVA without), %u rows at pos %u: q, gate, k, v bit-exact\n", C, pos);
}

// The routers' rows of one chunk, device side (section 4 makes them; 5 and 6 consume them).
struct Routes {
  std::vector<uint32_t> mova, moe;   // [C][32]
};

// ---- 4. the routers --------------------------------------------------------------------------
Routes routers(Dev& dv, uint32_t C, const std::vector<uint16_t>& x) {
  Routes out;
  {   // MoVA: from the fused row's v_router columns at pitch 9472
    const uint32_t ld = runtime::k2::pf_ld(D.attn_sparse_n());
    const std::vector<float> part = rf32(size_t(kC) * ld, -3.f, 3.f, 20);
    const std::vector<float> bias = rf32(D.value_experts, -0.05f, 0.05f, 21);
    l0::Mem dp = dv.up(part), db = dv.up(bias), rt = dv.zeros(size_t(kC) * 32 * 4);
    dv.go(kk::pf_route_variant(D.value_experts, D.value_top_k, ld, D.v_off()), "k2_route", 1, C, 1,
          {PtrArg(dp.ptr()), PtrArg(db.ptr()), PtrArg(rt.ptr())});
    out.mova = dv.rd<uint32_t>(rt, size_t(C) * 32);
    for (uint32_t m = 0; m < C; ++m) {
      const k2_ref::Route want = k2_ref::route(part.data(), kC, m, ld, D.v_off(), 1, bias.data(),
                                               D.value_experts, D.value_top_k, 2.5f);
      for (uint32_t j = 0; j < D.value_top_k; ++j) {
        CHECK_EQ(out.mova[size_t(m) * 32 + j], want.ids[j]);
        float w;
        std::memcpy(&w, &out.mova[size_t(m) * 32 + 8 + j], 4);
        CHECK(ulps(rne(w), rne(want.w[j])) <= 1);
      }
    }
  }
  {   // MoE: the router GEMV over the chunk (row m bitwise decode's), decode's k2_route
    const uint32_t H = D.hidden, RN = D.router_n();
    std::vector<uint16_t> rows = rbf16(size_t(RN) * H, -0.05f, 0.05f, 22);
    std::fill(rows.begin() + size_t(D.experts) * H, rows.end(), uint16_t(0));
    std::vector<uint16_t> tiled(rows.size());
    common::repack_bf16_tiled(rows.data(), H, RN, tiled.data());
    const std::vector<float> bias = rf32(RN, -0.05f, 0.05f, 23);
    l0::Mem dwr = dv.up(tiled), dx = dv.up(x), lg = dv.zeros(size_t(kC) * RN * 4), db = dv.up(bias),
            rt = dv.zeros(size_t(kC) * 32 * 4), one = dv.zeros(size_t(RN) * 4);
    dv.go(kernels::pf_moe_router_variant(H, RN), "pf_ab_proj", RN / 16, (C + 7) / 8, 1,
          {PtrArg(dwr.ptr()), PtrArg(dx.ptr()), PtrArg(lg.ptr()), arg_val(C)});
    const std::vector<float> logits = dv.rd<float>(lg, size_t(C) * RN);
    for (uint32_t m : {0u, C / 2, C - 1}) {
      l0::Mem xr = dv.up(std::vector<uint16_t>(x.begin() + size_t(m) * H, x.begin() + size_t(m + 1) * H));
      dv.go(kernels::gemv_bf16_variant(1, H, RN, kernels::gemv_bf16_tiling(RN)), "gemv_bf16",
            RN / kernels::gemv_bf16_tiling(RN).cols, 1, 1, {PtrArg(dwr.ptr()), PtrArg(xr.ptr()), PtrArg(one.ptr())});
      const std::vector<float> dec = dv.rd<float>(one, RN);
      CHECK(std::equal(dec.begin(), dec.end(), logits.begin() + size_t(m) * RN));
    }
    dv.go(kk::pf_route_variant(D.experts, D.top_k, RN, 0), "k2_route", 1, C, 1,
          {PtrArg(lg.ptr()), PtrArg(db.ptr()), PtrArg(rt.ptr())});
    out.moe = dv.rd<uint32_t>(rt, size_t(C) * 32);
    for (uint32_t m = 0; m < C; ++m) {
      const k2_ref::Route want = k2_ref::route(logits.data(), C, m, RN, 0, 1, bias.data(), D.experts, D.top_k, 2.5f);
      for (uint32_t j = 0; j < D.top_k; ++j) {
        CHECK_EQ(out.moe[size_t(m) * 32 + j], want.ids[j]);
        CHECK(out.moe[size_t(m) * 32 + j] < D.experts);   // the padded lanes are never ranked
      }
    }
  }
  std::printf("4. routers over %u rows: MoVA from column 9216 at pitch 9472, the MoE router GEMV "
              "(rows bitwise decode's) and decode's k2_route on grid (1, C) - ids exact\n", C);
  return out;
}

// One router family's walk over a chunk: sort, gather, every batch's dequant + grouped GEMM,
// and (MoE) down; the buffers the checks read back.
struct Walked {
  std::vector<uint32_t> hdr, tiles, row_tok, pair_row;
  std::vector<uint16_t> h, y, out;   // out: resid (MoE) or the V rows (MoVA)
};

struct MoeBufs {
  l0::Mem x, rt, hdr, tiles, row_tok, pair_row, xg, h, w, resid, gu, dn;
};

// ---- 5. the MoE block ----------------------------------------------------------------------
Walked moe_walk(Dev& dv, MoeBufs& b, uint32_t C) {
  const uint32_t H = D.hidden, I = D.moe_inter, B = D.moe_blocks();
  const uint32_t tmax = runtime::k2::pf_moe_tiles(D, C);
  const std::string v = kk::pf_moe_variant(D.experts, D.top_k, H, I);
  dv.go(v, "k2_pf_sort", 1, 1, 1,
        {PtrArg(b.rt.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.tiles.ptr()), PtrArg(b.row_tok.ptr()),
         PtrArg(b.pair_row.ptr()), arg_val(C), arg_val(tmax)});
  dv.go(v, "k2_pf_gather", tmax * kk::kPfTm, 1, 1,
        {PtrArg(b.x.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.row_tok.ptr()), PtrArg(b.xg.ptr())});
  Walked w;
  w.hdr = dv.rd<uint32_t>(b.hdr, kk::pf_hdr_words(D.experts));
  w.tiles = dv.rd<uint32_t>(b.tiles, size_t(tmax) * 2);
  w.row_tok = dv.rd<uint32_t>(b.row_tok, size_t(tmax) * kk::kPfTm);
  w.pair_row = dv.rd<uint32_t>(b.pair_row, size_t(C) * D.top_k);
  const uint32_t per_gu = runtime::k2::pf_batch_blocks(D, runtime::k2::PfGroup::GateUp);
  for (uint32_t b0 = 0; b0 < B; b0 += per_gu) {
    const uint32_t nb = std::min(per_gu, B - b0);
    dv.go(v, "k2_pf_dequant_gu", 2 * I / 16, H / 64, nb, {PtrArg(b.gu.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.w.ptr()), arg_val(b0)});
    dv.go(kernels::pf_moe_gemm_variant(H, 2 * I, false, true), "pf_moe_gemm", tmax, 2 * I / 256, 1,
          {PtrArg(b.tiles.ptr()), PtrArg(b.xg.ptr()), PtrArg(nullptr), PtrArg(b.w.ptr()), PtrArg(nullptr),
           PtrArg(b.h.ptr()), arg_val(b0), arg_val(b0 + nb), arg_val(0u)});
  }
  w.h = dv.rd<uint16_t>(b.h, size_t(tmax) * kk::kPfTm * I);
  const uint32_t per_dn = runtime::k2::pf_batch_blocks(D, runtime::k2::PfGroup::Down);
  for (uint32_t b0 = 0; b0 < B; b0 += per_dn) {
    const uint32_t nb = std::min(per_dn, B - b0);
    dv.go(v, "k2_pf_dequant_dn", H / 16, I / 64, nb, {PtrArg(b.dn.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.w.ptr()), arg_val(b0)});
    dv.go(kernels::pf_moe_gemm_variant(I, H, false, false), "pf_moe_gemm", tmax, H / 256, 1,
          {PtrArg(b.tiles.ptr()), PtrArg(b.h.ptr()), PtrArg(nullptr), PtrArg(b.w.ptr()), PtrArg(nullptr),
           PtrArg(b.xg.ptr()), arg_val(b0), arg_val(b0 + nb), arg_val(0u)});
  }
  w.y = dv.rd<uint16_t>(b.xg, size_t(tmax) * kk::kPfTm * H);
  dv.go(v, "k2_pf_moe_combine", H / 256, C, 1,
        {PtrArg(b.rt.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.pair_row.ptr()), PtrArg(b.xg.ptr()),
         PtrArg(b.resid.ptr()), arg_val(C)});
  w.out = dv.rd<uint16_t>(b.resid, size_t(C) * H);
  return w;
}

void moe_block(Dev& dv, const std::vector<uint16_t>& x, const std::vector<uint32_t>& routes, uint32_t C) {
  const uint32_t H = D.hidden, I = D.moe_inter, B = D.moe_blocks();
  const moe_ref::Shape ms{D.experts, D.top_k, H, I, D.router_n()};
  const std::vector<uint32_t> gu = rblocks(ms.gate_up_words() * B, 30), dn = rblocks(ms.down_words() * B, 31);
  const std::vector<uint16_t> resid0 = rbf16(size_t(C) * H, -2.f, 2.f, 32);
  const uint32_t T = runtime::k2::pf_moe_tiles(D, kC), R = T * kk::kPfTm + 256;
  MoeBufs b{dv.up(x), dv.up(routes), dv.zeros(kk::pf_hdr_words(D.experts) * 4), dv.zeros(size_t(T) * 8),
            dv.zeros(size_t(R) * 4), dv.zeros(size_t(kC) * 8 * 4), dv.zeros(size_t(R) * H * 2),
            dv.zeros(size_t(R) * I * 2), dv.zeros(runtime::k2::pf_weight_batch_bytes(D)), dv.up(resid0),
            dv.up(gu), dv.up(dn)};
  const Walked w = moe_walk(dv, b, C);
  // sort exact
  const k2_pf_ref::Sorted st = k2_pf_ref::sort(routes.data(), C, D.experts, D.top_k, true);
  CHECK(w.hdr == st.hdr);
  CHECK(std::equal(st.tiles.begin(), st.tiles.end(), w.tiles.begin()));
  CHECK(std::equal(st.row_tok.begin(), st.row_tok.begin() + st.rows_used(), w.row_tok.begin()));
  CHECK(w.pair_row == st.pair_row);
  // grouped == dense, per (token, slot), for the most loaded expert, expert 0, the shared expert
  std::vector<uint32_t> load(D.experts, 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < D.top_k; ++j) ++load[routes[size_t(t) * 32 + j]];
  const uint32_t busiest = uint32_t(std::max_element(load.begin(), load.end()) - load.begin());
  l0::Mem dx = dv.up(x);
  for (uint32_t e : {busiest, 0u, D.shared_block()}) {
    if (e < D.experts && load[e] == 0) continue;
    const std::vector<uint16_t> bgu = pf_moe_ref::dequant_block(gu.data() + e * ms.gate_up_words(), H, 2 * I);
    l0::Mem db = dv.up(bgu), xs = dv.zeros(size_t(kC) * I * 2);
    const runtime::prefill::GemmBatch gb{C, H, 2 * I, 1, H, 2 * I, 0, 0, 0, 0};
    runtime::prefill::gemm_l0_silu(dv.cx, dv.kc, gb, dx.as<uint16_t>(), db.as<uint16_t>(), xs.as<uint16_t>(), I);
    dv.cx.wait();
    const std::vector<uint16_t> dense = dv.rd<uint16_t>(xs, size_t(C) * I);
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j <= D.top_k; ++j) {
        const bool hit = j < D.top_k ? routes[size_t(t) * 32 + j] == e : e == D.shared_block();
        if (!hit) continue;
        const uint32_t r = j < D.top_k ? st.pair_row[size_t(t) * D.top_k + j] : st.shared_row() + t;
        CHECK(std::equal(dense.begin() + size_t(t) * I, dense.begin() + size_t(t + 1) * I, w.h.begin() + size_t(r) * I));
      }
    // down: the dense GEMM over every sorted h row against this expert's block; the rows of
    // its tiles must match rne of it.
    const std::vector<uint16_t> bdn = pf_moe_ref::dequant_block(dn.data() + e * ms.down_words(), I, H);
    l0::Mem ddn = dv.up(bdn), yd = dv.zeros(size_t(R) * H * 4);
    const uint32_t rows = st.rows_used();
    const runtime::prefill::GemmBatch gd{rows, I, H, 1, I, H, H, 0, 0, 0};
    runtime::prefill::gemm_l0(dv.cx, dv.kc, gd, b.h.as<uint16_t>(), ddn.as<uint16_t>(), yd.as<float>(), false);
    dv.cx.wait();
    const std::vector<float> yf = dv.rd<float>(yd, size_t(rows) * H);
    for (uint32_t tile = 0; tile < st.hdr[k2_pf_ref::kHdrTiles]; ++tile) {
      if (st.tiles[2 * tile] != e) continue;
      for (uint32_t i = 0; i < kk::kPfTm; ++i) {
        const uint32_t r = st.tiles[2 * tile + 1] + i;
        if (st.row_tok[r] == k2_pf_ref::kNone) continue;
        for (uint32_t n = 0; n < H; ++n) CHECK_EQ(w.y[size_t(r) * H + n], rne(yf[size_t(r) * H + n]));
      }
    }
  }
  // the combine exactly the reference's over the device's y
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t n = 0; n < H; ++n)
      CHECK_EQ(w.out[size_t(t) * H + n],
               k2_pf_ref::moe_combine(st, routes.data() + size_t(t) * 32, w.y.data(), H, D.top_k, t, n,
                                      resid0[size_t(t) * H + n]));
  // Review Focus 2: the chunk reversed
  {
    std::vector<uint16_t> xr(x.size()), rr0(resid0.size());
    std::vector<uint32_t> rtr(routes.size());
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t u = C - 1 - t;
      std::copy(x.begin() + size_t(t) * H, x.begin() + size_t(t + 1) * H, xr.begin() + size_t(u) * H);
      std::copy(resid0.begin() + size_t(t) * H, resid0.begin() + size_t(t + 1) * H, rr0.begin() + size_t(u) * H);
      std::copy(routes.begin() + size_t(t) * 32, routes.begin() + size_t(t + 1) * 32, rtr.begin() + size_t(u) * 32);
    }
    dv.imm.copy(b.x.ptr(), xr.data(), xr.size() * 2);
    dv.imm.copy(b.rt.ptr(), rtr.data(), rtr.size() * 4);
    dv.imm.copy(b.resid.ptr(), rr0.data(), rr0.size() * 2);
    const Walked v = moe_walk(dv, b, C);
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t u = C - 1 - t;
      CHECK(std::equal(w.out.begin() + size_t(t) * H, w.out.begin() + size_t(t + 1) * H, v.out.begin() + size_t(u) * H));
      for (uint32_t j = 0; j < D.top_k; ++j) {
        const uint32_t ra = w.pair_row[size_t(t) * D.top_k + j], rb = v.pair_row[size_t(u) * D.top_k + j];
        CHECK(std::equal(w.h.begin() + size_t(ra) * I, w.h.begin() + size_t(ra + 1) * I, v.h.begin() + size_t(rb) * I));
        CHECK(std::equal(w.y.begin() + size_t(ra) * H, w.y.begin() + size_t(ra + 1) * H, v.y.begin() + size_t(rb) * H));
      }
    }
    // replay: the forward chunk again, bitwise
    dv.imm.copy(b.x.ptr(), x.data(), x.size() * 2);
    dv.imm.copy(b.rt.ptr(), routes.data(), routes.size() * 4);
    dv.imm.copy(b.resid.ptr(), resid0.data(), resid0.size() * 2);
    const Walked again = moe_walk(dv, b, C);
    CHECK(again.out == w.out && again.h == w.h && again.y == w.y);
  }
  std::printf("5. MoE block over %u rows: sort / gather exact; grouped == dense bitwise (experts %u, 0, "
              "shared); combine exact; reversed chunk and replay bitwise\n", C, busiest);
}

// ---- 6. MoVA -------------------------------------------------------------------------------
void mova_block(Dev& dv, const std::vector<uint16_t>& x, const std::vector<uint32_t>& routes, uint32_t C) {
  const uint32_t H = D.hidden, VN = D.kv_n(), E = D.value_experts, K = D.value_top_k, pos = 100;
  const size_t vblk = size_t(VN / 16) * (H / 64) * 136;
  const std::vector<uint32_t> vw = rblocks(vblk * E, 40);
  const uint32_t T = runtime::k2::pf_mova_tiles(D, kC), R = T * kk::kPfTm + 256;
  l0::Mem dx = dv.up(x), rt = dv.up(routes), hdr = dv.zeros(kk::pf_hdr_words(E) * 4), tiles = dv.zeros(size_t(T) * 8),
          row_tok = dv.zeros(size_t(R) * 4), pair_row = dv.zeros(size_t(kC) * 8 * 4), xg = dv.zeros(size_t(R) * H * 2),
          y = dv.zeros(size_t(R) * VN * 2), wb = dv.zeros(runtime::k2::pf_weight_batch_bytes(D)), dw = dv.up(vw),
          kv = dv.zeros(size_t(pos + kC) * VN * 2);
  const uint32_t tmax = runtime::k2::pf_mova_tiles(D, C);
  const std::string v = kk::pf_mova_variant(E, K, H, VN);
  auto walk = [&](std::vector<uint32_t>& pr, std::vector<uint16_t>& yv) {
    dv.go(v, "k2_pf_sort", 1, 1, 1, {PtrArg(rt.ptr()), PtrArg(hdr.ptr()), PtrArg(tiles.ptr()), PtrArg(row_tok.ptr()),
                                     PtrArg(pair_row.ptr()), arg_val(C), arg_val(tmax)});
    dv.go(v, "k2_pf_gather", tmax * kk::kPfTm, 1, 1, {PtrArg(dx.ptr()), PtrArg(hdr.ptr()), PtrArg(row_tok.ptr()), PtrArg(xg.ptr())});
    const uint32_t per = runtime::k2::pf_batch_blocks(D, runtime::k2::PfGroup::Value);
    for (uint32_t b0 = 0; b0 < E; b0 += per) {
      const uint32_t nb = std::min(per, E - b0);
      dv.go(v, "k2_pf_dequant_v", VN / 16, H / 64, nb, {PtrArg(dw.ptr()), PtrArg(hdr.ptr()), PtrArg(wb.ptr()), arg_val(b0)});
      dv.go(kernels::pf_moe_gemm_variant(H, VN, false, false), "pf_moe_gemm", tmax, VN / 256, 1,
            {PtrArg(tiles.ptr()), PtrArg(xg.ptr()), PtrArg(nullptr), PtrArg(wb.ptr()), PtrArg(nullptr), PtrArg(y.ptr()),
             arg_val(b0), arg_val(b0 + nb), arg_val(0u)});
    }
    dv.go(v, "k2_pf_mova_combine", VN / 256, C, 1,
          {PtrArg(rt.ptr()), PtrArg(pair_row.ptr()), PtrArg(y.ptr()), PtrArg(kv.ptr()), arg_val(pos), arg_val(C)});
    pr = dv.rd<uint32_t>(pair_row, size_t(C) * K);
    yv = dv.rd<uint16_t>(y, size_t(tmax) * kk::kPfTm * VN);
    return dv.rd<uint16_t>(kv, size_t(C) * VN, size_t(pos) * VN);
  };
  std::vector<uint32_t> pr;
  std::vector<uint16_t> yv;
  const std::vector<uint16_t> vrows = walk(pr, yv);
  const k2_pf_ref::Sorted st = k2_pf_ref::sort(routes.data(), C, E, K, false);
  CHECK(pr == st.pair_row);
  // grouped == dense for the busiest value expert
  std::vector<uint32_t> load(E, 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < K; ++j) ++load[routes[size_t(t) * 32 + j]];
  const uint32_t e = uint32_t(std::max_element(load.begin(), load.end()) - load.begin());
  {
    const std::vector<uint16_t> bb = pf_moe_ref::dequant_block(vw.data() + e * vblk, H, VN);
    l0::Mem db = dv.up(bb), od = dv.zeros(size_t(kC) * VN * 4);
    const runtime::prefill::GemmBatch gb{C, H, VN, 1, H, VN, VN, 0, 0, 0};
    runtime::prefill::gemm_l0(dv.cx, dv.kc, gb, dx.as<uint16_t>(), db.as<uint16_t>(), od.as<float>(), false);
    dv.cx.wait();
    const std::vector<float> o = dv.rd<float>(od, size_t(C) * VN);
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < K; ++j)
        if (routes[size_t(t) * 32 + j] == e)
          for (uint32_t n = 0; n < VN; ++n)
            CHECK_EQ(yv[size_t(st.pair_row[size_t(t) * K + j]) * VN + n], rne(o[size_t(t) * VN + n]));
  }
  // the combine into V exactly the reference's
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t n = 0; n < VN; ++n)
      CHECK_EQ(vrows[size_t(t) * VN + n], k2_pf_ref::mova_combine(st, routes.data() + size_t(t) * 32, yv.data(), VN, K, t, n));
  // Review Focus 1 at kernel level: decode's k2_mova_value on token t's x and route row
  double worst = 1.0;
  l0::Mem ctrl(dv.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  l0::Mem kvd = dv.zeros(size_t(4) * VN * 2), xr = dv.zeros(size_t(H) * 2), r1 = dv.zeros(32 * 4);
  for (uint32_t t : {0u, C / 2, C - 1}) {
    std::memset(ctrl.ptr(), 0, ctrl.size());
    ctrl.as<runtime::Control>()->pos = 1;
    ctrl.as<runtime::Control>()->n_active = 1;
    dv.imm.copy(xr.ptr(), x.data() + size_t(t) * H, size_t(H) * 2);
    dv.imm.copy(r1.ptr(), routes.data() + size_t(t) * 32, 32 * 4);
    dv.go(kk::mova_variant(1, E, K, H, VN), "k2_mova_value", VN / 16, 1, 1,
          {PtrArg(ctrl.ptr()), PtrArg(r1.ptr()), PtrArg(xr.ptr()), PtrArg(dw.ptr()), PtrArg(kvd.ptr())});
    const std::vector<uint16_t> dec = dv.rd<uint16_t>(kvd, VN, VN);
    worst = std::min(worst, cosine_bf16(dec.data(), vrows.data() + size_t(t) * VN, VN));
  }
  CHECK(worst >= 0.9999);
  std::printf("6. MoVA over %u rows: sort exact, grouped == dense bitwise (expert %u), the combine into V "
              "exact; decode's k2_mova_value on the same rows: worst cosine %.6f\n", C, e, worst);
}

// ---- 7 / 8. the flash attention ---------------------------------------------------------------
void flash(Dev& dv) {
  const uint32_t QH = D.q_heads, KVH = D.kv_heads, HD = 128, QN = QH * HD, KVN = KVH * HD;
  struct Case { uint32_t pos, C; };
  const Case cases[] = {{0, 1}, {0, 37}, {0, 2048}, {2011, 37}, {2048, 2048}, {30000, 2048}, {60000, 2048}, {4095, 1}};
  const uint32_t maxd = 62048;
  // One cache for every case: rows [0, depth) of it are read.
  const std::vector<uint16_t> kc = rbf16(size_t(maxd) * KVN, -1.f, 1.f, 50);
  const std::vector<uint16_t> vc = rbf16(size_t(maxd) * KVN, -1.f, 1.f, 51);
  l0::Mem dk = dv.up(kc), dvv = dv.up(vc);
  std::vector<float> q = rf32(size_t(kC) * QN, -1.f, 1.f, 52);
  for (float& v : q) v = rf(v);   // k2_attn_prep's q are bf16 values
  std::vector<float> g = rf32(size_t(kC) * QN, -6.f, 6.f, 53);
  for (size_t i = 0; i < g.size(); i += 7) g[i] = 25.0f + float(i % 17);   // beta x > 20 for x > 28.85
  for (float& v : g) v = rf(v);
  l0::Mem dq = dv.up(q), dg = dv.up(g), o = dv.zeros(size_t(QH) * kC * HD * 4), out = dv.zeros(size_t(kC) * QN * 2);
  double worst_flash = 1.0, worst_eager = 1.0, worst_eager64 = 1.0;
  int worst_gate = 0;
  for (const Case& c : cases) {
    std::vector<uint32_t> rows = {0, c.C - 1, c.C / 2};
    if (c.C > 9) rows.insert(rows.end(), {7, 8});
    for (int eager = 0; eager <= 1; ++eager) {
      dv.go(kk::pf_flash_variant(QH, KVH, eager, false), "k2_pf_flash_attn", (c.C + 7) / 8, KVH, 1,
            {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(dg.ptr()), PtrArg(o.ptr()),
             arg_val(c.pos), arg_val(c.C)});
      const std::vector<float> of = dv.rd<float>(o, size_t(QH) * c.C * HD);
      for (uint32_t r : rows)
        for (uint32_t h = 0; h < QH; ++h) {
          const float* qr = q.data() + (size_t(r) * QH + h) * HD;
          const float* got = of.data() + (size_t(h) * c.C + r) * HD;
          const std::vector<double> want = k2_pf_ref::attention(qr, kc.data(), vc.data(), c.pos + r, h / (QH / KVH), KVH, HD);
          if (!eager) {
            worst_flash = std::min(worst_flash, cosine(want.data(), got, HD));
          } else {
            worst_eager64 = std::min(worst_eager64, cosine(want.data(), got, HD));
            const std::vector<float> e =
                k2_pf_ref::attention_eager(qr, kc.data(), vc.data(), c.pos + r, h / (QH / KVH), KVH, HD);
            std::vector<double> ed(e.begin(), e.end());
            worst_eager = std::min(worst_eager, cosine(ed.data(), got, HD));
          }
        }
      // the gated production build against k2_ref::attn_gate over this _O output
      dv.go(kk::pf_flash_variant(QH, KVH, eager, true), "k2_pf_flash_attn", (c.C + 7) / 8, KVH, 1,
            {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(dg.ptr()), PtrArg(out.ptr()),
             arg_val(c.pos), arg_val(c.C)});
      const std::vector<uint16_t> ob = dv.rd<uint16_t>(out, size_t(c.C) * QN);
      for (uint32_t r : rows)
        for (uint32_t h = 0; h < QH; ++h)
          for (uint32_t dd = 0; dd < HD; ++dd) {
            const size_t idx = (size_t(r) * QH + h) * HD + dd;
            worst_gate = std::max(worst_gate, ulps(ob[idx], k2_ref::attn_gate(of[(size_t(h) * c.C + r) * HD + dd], g[idx])));
          }
      if (c.C == 2048 && c.pos == 2048) {   // 8. replay bitwise
        dv.go(kk::pf_flash_variant(QH, KVH, eager, true), "k2_pf_flash_attn", (c.C + 7) / 8, KVH, 1,
              {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(dg.ptr()), PtrArg(out.ptr()),
               arg_val(c.pos), arg_val(c.C)});
        CHECK(dv.rd<uint16_t>(out, size_t(c.C) * QN) == ob);
      }
    }
    std::printf("   (pos %5u, C %4u): flash worst %.7f, eager worst %.7f vs its reference / %.6f vs fp64\n",
                c.pos, c.C, worst_flash, worst_eager, worst_eager64);
  }
  CHECK(worst_flash >= 0.99999);
  CHECK(worst_eager >= 0.9999);
  CHECK(worst_eager64 >= 0.999);
  CHECK(worst_gate <= 1);
  std::printf("7. flash attention at head_dim 128 / GQA 4, depths 0 .. 62048: default %.7f (bar 0.99999), "
              "eager %.6f vs its reference (0.9999) / %.6f vs fp64 (0.999); gated build within %d bf16 ulp; "
              "8. replay bitwise\n", worst_flash, worst_eager, worst_eager64, worst_gate);
}

}  // namespace

int main() {
  Dev dv;
  std::printf("k2_pf_kernels_test on %s\n", dv.ctx.name().c_str());
  slab_and_linear(dv);
  norm(dv);
  attn_prep(dv);
  const uint32_t C = 64;
  const std::vector<uint16_t> x = rbf16(size_t(kC) * D.hidden, -0.2f, 0.2f, 60);
  const Routes r = routers(dv, C, x);
  moe_block(dv, x, r.moe, C);
  mova_block(dv, x, r.mova, C);
  flash(dv);
  std::printf("k2_pf_kernels_test OK\n");
  return 0;
}
