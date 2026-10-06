// Spec 20d (K1 for the prefill): Kolibri-1's prefill kernels on the card, at the real shapes, against
// tests/kernels/kolibri_pf_ref.h / kolibri_ref.h and against the dense prefill GEMMs the grouped ones are
// built from. Needs a B70, no checkpoint (synthetic weights). Every binary is one
// runtime/kolibri/kolibri_prefill.cc binds (kernels::kolibri names), plus pf_gemm_T0_SILU for the dense
// comparison. Written blind on the Mac (box queue row 26).
//
//   1. the slabs: k2_pf_dequant_slab at q||k||v 2560 x 7168 (the int4 arm) and kol_pf_bf16_slab at o_proj
//      6144 x 2560 (1024 + 1024 + 512) exact; the bf16 o_proj through pf_gemm_T0 rows within cosine
//      0.99999 of fp64 host sums;
//   2. the grouped norm: pf_res_fold SP0 (m_count kPfC) + kol_norm_finish (M kPfC), and the sandwich:
//      pf_res_fold _Z over o_proj's one slice + kol_post_add (M kPfC) - every value and sum bit-exact
//      against kolibri_ref.h (no exp; sqrt / divide correctly rounded on ocloc's command line);
//   3. kol_attn_prep at M kPfC / S 1: sliding at pos 4090 over 37 rows (the chunk wraps the ring: slots
//      4090..4095, 0..30) and full (NoPE) at 37 - q, k, v bit-exact;
//   4. the router over the chunk: pf_ab_proj 2560 -> 512 (each sampled row bitwise decode's gemv_bf16 of
//      that row; rows 384.. zero) and decode's kol_route on grid (1, C) - ids exact, weights within 2 ulp;
//   5. the MoE block over C = 64: sort / gather exact; the dequantised blocks exact (block 384 = the bf16
//      shared expert's tiles); GROUPED == DENSE, bitwise, per (token, slot): gate||up (every batch)
//      against pf_gemm_T0_SILU over the chunk, down against rne(pf_gemm_T0) over the sorted h, for the
//      busiest expert, expert 0 when routed and the shared expert; the combine exactly kolibri_pf_ref's;
//      Review Focus 2: the chunk REVERSED leaves every (token, slot)'s h and y and every token's mo
//      bitwise unchanged; replay bitwise; Review Focus 3: the shared expert alone (every routed weight 0):
//      mo = the shared expert's down row, bitwise;
//      and the sort at C = 2048 on the adversaries (all to 6 experts; the tile bound reached, 820 tiles);
//   6. the flash attention (Review Focus 1), against flash_row (fp64; cosine >= 0.99999, spec 6 K1):
//      full at (pos, C) = (0, 2048), (2048, 2048), (30000, 2048), (60000, 2048); sliding at c0 = 0, 1000,
//      4396 (wrapped), a tail chunk (1001, 300) and single rows (5000, 1), rows 0, 1, 7, 23, 511, 512, 513
//      and C - 1; the EAGER build against eager_window (0.9999; the rounding points, not the sum order:
//      spec 18 §11) and fp64 (0.999); a chunk split at a multiple of 64 bitwise (rows of (1000, 2048) ==
//      rows of (1064, 1984)), and replay bitwise.
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
#include "kernels/kernels.h"
#include "kernels/kolibri_kernels.h"
#include "kernels/kolibri_pf_ref.h"
#include "kernels/kolibri_ref.h"
#include "kernels/prefill/pf_kernels.h"
#include "kernels/prep_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/kolibri1_repack.h"
#include "model/kolibri1.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"

namespace {

namespace kk = kernels::kolibri;
namespace kr = kolibri_ref;
namespace kp = kolibri_pf_ref;
using kr::f32;
using kr::rf;
using kr::rne;
using runtime::prefill::arg_val;
using runtime::prefill::KernelArg;
using runtime::prefill::PtrArg;

constexpr uint32_t kC = kk::kPfC;
const uint32_t H = kk::kHidden, HD = kk::kHd, QH = kk::kQHeads, KVH = kk::kKvHeads, QN = QH * HD, KVN = KVH * HD,
               I = kk::kInter;

static_assert(kk::kPfC == kp::kC && kk::kPfTm == kp::kTm && kk::kPfKt == kp::kKt && kk::kPfRpw == kp::kRpw,
              "kernels::kolibri's prefill constants and kolibri_pf_ref's disagree");
static_assert(kk::pf_hdr::words() == kp::hdr_words(), "the sort's header width moved");

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
double cosine16(const double* a, const uint16_t* b, size_t n) {
  std::vector<float> f(n);
  for (size_t i = 0; i < n; ++i) f[i] = f32(b[i]);
  return cosine(a, f.data(), n);
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

uint32_t slab_width(uint32_t N, uint32_t n0) { return N - n0 >= kk::kPfSlab ? kk::kPfSlab : (N - n0 + 255) / 256 * 256; }

// ---- 1. the slabs and the linear ---------------------------------------------------------
void slabs(Dev& dv) {
  {   // int4 q||k||v (layout 0)
    const uint32_t K = H, N = kk::kQkvN;
    Rng r(1);
    std::vector<uint32_t> words(size_t(K / 8) * N);
    for (uint32_t& w : words) w = r.u32();
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    for (uint16_t& s : sc) s = common::f32_to_f16(r.uni(0.01f, 0.05f));
    l0::Mem dw = dv.up(words), ds = dv.up(sc), slab = dv.zeros(size_t(K) * kk::kPfSlab * 2);
    for (uint32_t n0 : {0u, 6144u}) {
      dv.go(kk::pf_int4_slab_variant(K, N), "k2_pf_dequant_slab", kk::kPfSlab / 16, K / 64, 1,
            {PtrArg(dw.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0), arg_val(kk::kPfSlab)});
      CHECK(dv.rd<uint16_t>(slab, size_t(K) * kk::kPfSlab) ==
            k2_pf_ref::dequant_slab(words.data(), sc.data(), K, N, n0, kk::kPfSlab));
    }
  }
  // bf16 o_proj: the slabs, then the linear through pf_gemm_T0 into partials [kC][2560]
  const uint32_t K = QN, N = H, C = 37;
  const std::vector<uint16_t> wr = rbf16(size_t(N) * K, -0.05f, 0.05f, 2);
  std::vector<uint16_t> wt(wr.size());
  common::repack_bf16_tiled(wr.data(), K, N, wt.data());
  const std::vector<uint16_t> x = rbf16(size_t(kC) * K, -1.f, 1.f, 3);
  l0::Mem dw = dv.up(wt), dx = dv.up(x), slab = dv.zeros(size_t(K) * kk::kPfSlab * 2),
          part = dv.up(std::vector<float>(size_t(kC) * N, 7.0f));
  for (uint32_t n0 = 0; n0 < N;) {
    const uint32_t ns = slab_width(N, n0);
    dv.go(kk::pf_bf16_slab_variant(K, N), "kol_pf_bf16_slab", ns / 16, K / 8, 1,
          {PtrArg(dw.ptr()), PtrArg(slab.ptr()), arg_val(n0), arg_val(ns)});
    CHECK(dv.rd<uint16_t>(slab, size_t(K) * ns) == kp::bf16_slab(wt.data(), K, N, n0, ns));
    const runtime::prefill::GemmBatch gb{C, K, ns, 1, K, ns, N, 0, 0, 0};
    runtime::prefill::gemm_l0(dv.cx, dv.kc, gb, dx.as<uint16_t>(), slab.as<uint16_t>(), part.as<float>() + n0, false);
    dv.cx.wait();
    n0 += ns;
  }
  const std::vector<float> got = dv.rd<float>(part, size_t(C) * N);
  double worst = 1.0;
  for (uint32_t m = 0; m < C; m += 6) {
    std::vector<double> want(N, 0.0);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k) want[n] += double(f32(x[size_t(m) * K + k])) * f32(wr[size_t(n) * K + k]);
    worst = std::min(worst, cosine(want.data(), got.data() + size_t(m) * N, N));
  }
  CHECK(worst >= 0.99999);
  std::printf("1. slabs exact (int4 q||k||v 2560 x 7168, bf16 o_proj 6144 x 2560 = 1024 + 1024 + 512); the bf16 "
              "o_proj through pf_gemm_T0: rows worst cosine %.7f\n", worst);
}

// ---- 2. the grouped norm and the sandwich -----------------------------------------------------
void norms(Dev& dv) {
  const uint32_t C = 37, G = kk::kNormG;
  std::vector<uint16_t> resid = rbf16(size_t(kC) * H, -2.f, 2.f, 10);
  for (uint32_t k = 0; k < H / 2; ++k) resid[k] = rne(f32(resid[k]) * 0.01f);   // unequal groups
  const std::vector<float> part = rf32(size_t(kC) * H, -0.5f, 0.5f, 11);
  std::vector<float> w(H), w2(H);
  for (uint32_t k = 0; k < H; ++k) {
    w[k] = rf(0.3f + float(k % 11) * 0.1f);
    w2[k] = rf(1.5f + float(k % 7) * 0.2f);
  }
  l0::Mem dr = dv.up(resid), dp = dv.up(part), dw = dv.up(w), dw2 = dv.up(w2), ss = dv.zeros(size_t(G) * kC * 4),
          xn = dv.zeros(size_t(kC) * H * 2), a = dv.zeros(size_t(kC) * H * 2), sa = dv.zeros(size_t(G) * kC * 4);
  dv.go(kk::pf_fold_variant(), "pf_res_fold", G, C, 1, {PtrArg(dp.ptr()), PtrArg(dr.ptr()), PtrArg(ss.ptr()), arg_val(kC)});
  dv.go(kk::pf_norm_variant(), "kol_norm_finish", kk::kNormW, C, 1,
        {PtrArg(ss.ptr()), PtrArg(dr.ptr()), PtrArg(dw.ptr()), PtrArg(xn.ptr())});
  // host: the C rows at pitch kC (the sums' rectangle is [G][kC]; rows past C are not touched)
  std::vector<uint16_t> r_ref(resid.begin(), resid.begin() + size_t(C) * H), x_ref(size_t(C) * H);
  std::vector<float> s_c(size_t(G) * C), s_ref(size_t(G) * kC, 0.0f);
  prep_ref::res_fold(nullptr, r_ref.data(), s_c.data(), C, H, 0, G);
  kr::norm_finish(s_c.data(), r_ref.data(), w.data(), x_ref.data(), C);
  for (uint32_t g = 0; g < G; ++g)
    for (uint32_t m = 0; m < C; ++m) s_ref[size_t(g) * kC + m] = s_c[size_t(g) * C + m];
  CHECK(dv.rd<uint16_t>(dr, size_t(C) * H) == r_ref);
  CHECK(dv.rd<uint16_t>(xn, size_t(C) * H) == x_ref);
  const std::vector<float> ssd = dv.rd<float>(ss, size_t(G) * kC);
  for (uint32_t g = 0; g < G; ++g)
    for (uint32_t m = 0; m < C; ++m) CHECK_EQ(ssd[size_t(g) * kC + m], s_ref[size_t(g) * kC + m]);
  // the sandwich: o_proj's slice -> a, Σa²; kol_post_add -> resid, Σ resid²
  dv.go(kk::pf_fold_zero_variant(), "pf_res_fold", G, C, 1, {PtrArg(dp.ptr()), PtrArg(a.ptr()), PtrArg(sa.ptr()), arg_val(kC)});
  dv.go(kk::pf_post_add_variant(), "kol_post_add", G, C, 1,
        {PtrArg(sa.ptr()), PtrArg(a.ptr()), PtrArg(dw2.ptr()), PtrArg(dr.ptr()), PtrArg(ss.ptr())});
  std::vector<uint16_t> a_ref(size_t(C) * H);
  for (size_t i = 0; i < a_ref.size(); ++i) a_ref[i] = rne(part[i]);
  std::vector<float> sa_c(size_t(G) * C), so_c(size_t(G) * C);
  {
    std::vector<uint16_t> t = a_ref;
    prep_ref::res_fold(nullptr, t.data(), sa_c.data(), C, H, 0, G);
  }
  kr::post_add(sa_c.data(), a_ref.data(), w2.data(), r_ref.data(), so_c.data(), C);
  CHECK(dv.rd<uint16_t>(a, size_t(C) * H) == a_ref);
  CHECK(dv.rd<uint16_t>(dr, size_t(C) * H) == r_ref);
  const std::vector<float> sod = dv.rd<float>(ss, size_t(G) * kC);
  for (uint32_t g = 0; g < G; ++g)
    for (uint32_t m = 0; m < C; ++m) CHECK_EQ(sod[size_t(g) * kC + m], so_c[size_t(g) * C + m]);
  std::printf("2. grouped norm and the sandwich over %u rows at pitch %u: resid, xn, a and every sum bit-exact\n", C, kC);
}

// ---- 3. the attention prep --------------------------------------------------------------------
void attn_prep(Dev& dv) {
  const uint32_t C = 37;
  const std::vector<float> rope = loader::kol_rope_table(model::kolibri1(), 4200);
  l0::Mem drope = dv.up(rope);
  l0::Mem ctrl(dv.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  std::vector<float> qkn(2 * HD);
  for (uint32_t i = 0; i < 2 * HD; ++i) qkn[i] = rf(0.8f + float(i % 13) * 0.03f);
  l0::Mem dq = dv.up(qkn);
  for (int sliding = 1; sliding >= 0; --sliding) {
    const uint32_t pos = sliding ? 4090 : 37, rows = sliding ? kk::kRing : 128;
    std::memset(ctrl.ptr(), 0, ctrl.size());
    ctrl.as<runtime::Control>()->pos = pos;
    ctrl.as<runtime::Control>()->n_active = C;
    const std::vector<float> part = rf32(size_t(kC) * kk::kQkvN, -1.f, 1.f, 20 + sliding);
    l0::Mem dp = dv.up(part), aq = dv.zeros(size_t(kC) * QN * 4), kc = dv.zeros(size_t(rows) * KVN * 2),
            vc = dv.zeros(size_t(rows) * KVN * 2);
    dv.go(kk::pf_attn_prep_variant(sliding), "kol_attn_prep", QH + 2 * KVH, C, 1,
          {PtrArg(ctrl.ptr()), PtrArg(dp.ptr()), PtrArg(dq.ptr()), PtrArg(drope.ptr()), PtrArg(aq.ptr()),
           PtrArg(kc.ptr()), PtrArg(vc.ptr())});
    const std::vector<float> q = dv.rd<float>(aq, size_t(C) * QN);
    const std::vector<uint16_t> k = dv.rd<uint16_t>(kc, size_t(rows) * KVN), v = dv.rd<uint16_t>(vc, size_t(rows) * KVN);
    for (uint32_t m = 0; m < C; ++m) {
      const kr::Prep want = kr::attn_prep(part.data(), kC, m, 1, qkn.data(), sliding ? rope.data() + size_t(pos + m) * HD : nullptr);
      const size_t row = kr::ring_row(pos + m, sliding);
      CHECK(std::equal(want.q.begin(), want.q.end(), q.begin() + size_t(m) * QN));
      CHECK(std::equal(want.k.begin(), want.k.end(), k.begin() + row * KVN));
      CHECK(std::equal(want.v.begin(), want.v.end(), v.begin() + row * KVN));
    }
  }
  std::printf("3. attention prep at M %u / S 1: sliding at 4090 over %u rows (the ring wraps after slot 4095), full "
              "at 37: q, k, v bit-exact\n", kC, C);
}

// ---- 4. the router -------------------------------------------------------------------------------
std::vector<uint32_t> routers(Dev& dv, uint32_t C, const std::vector<uint16_t>& x) {
  const uint32_t RN = kk::kRouterN;
  std::vector<uint16_t> rows = rbf16(size_t(RN) * H, -0.05f, 0.05f, 30);
  std::fill(rows.begin() + size_t(kk::kExperts) * H, rows.end(), uint16_t(0));
  std::vector<uint16_t> tiled(rows.size());
  common::repack_bf16_tiled(rows.data(), H, RN, tiled.data());
  std::vector<float> bias = rf32(RN, -0.05f, 0.05f, 31);
  std::fill(bias.begin() + kk::kExperts, bias.end(), 0.0f);
  l0::Mem dwr = dv.up(tiled), dx = dv.up(x), lg = dv.zeros(size_t(kC) * RN * 4), db = dv.up(bias),
          rt = dv.zeros(size_t(kC) * 32 * 4), one = dv.zeros(size_t(RN) * 4);
  dv.go(kk::pf_router_variant(), "pf_ab_proj", RN / 16, (C + 7) / 8, 1,
        {PtrArg(dwr.ptr()), PtrArg(dx.ptr()), PtrArg(lg.ptr()), arg_val(C)});
  const std::vector<float> logits = dv.rd<float>(lg, size_t(C) * RN);
  for (uint32_t m : {0u, C / 2, C - 1}) {
    l0::Mem xr = dv.up(std::vector<uint16_t>(x.begin() + size_t(m) * H, x.begin() + size_t(m + 1) * H));
    dv.go(kk::router_variant(), "gemv_bf16", RN / kernels::gemv_bf16_tiling(RN).cols, 1, 1,
          {PtrArg(dwr.ptr()), PtrArg(xr.ptr()), PtrArg(one.ptr())});
    const std::vector<float> dec = dv.rd<float>(one, RN);
    CHECK(std::equal(dec.begin(), dec.end(), logits.begin() + size_t(m) * RN));
  }
  dv.go(kk::pf_route_variant(), "kol_route", 1, C, 1, {PtrArg(lg.ptr()), PtrArg(db.ptr()), PtrArg(rt.ptr())});
  const std::vector<uint32_t> out = dv.rd<uint32_t>(rt, size_t(C) * 32);
  int64_t worst = 0;
  for (uint32_t m = 0; m < C; ++m) {
    const kr::Route want = kr::route(logits.data() + size_t(m) * RN, bias.data());
    for (uint32_t j = 0; j < kk::kTopK; ++j) {
      CHECK_EQ(out[size_t(m) * 32 + j], want.ids[j]);
      worst = std::max(worst, std::llabs(int64_t(out[size_t(m) * 32 + 8 + j]) - int64_t(kr::as_u32(want.w[j]))));
    }
  }
  CHECK(worst <= 2);
  std::printf("4. router over %u rows: pf_ab_proj rows bitwise decode's gemv_bf16, decode's kol_route on grid (1, C): "
              "ids exact, weights within %lld fp32 ulp\n", C, (long long)worst);
  return out;
}

// ---- 5. the MoE block ----------------------------------------------------------------------------
struct Walked {
  std::vector<uint32_t> hdr, tiles, row_tok, pair_row;
  std::vector<uint16_t> xg, h, y, mo;
};
struct MoeBufs {
  l0::Mem x, rt, hdr, tiles, row_tok, pair_row, xg, h, w, mo, gu, dn, sgu, sdn;
};

Walked moe_walk(Dev& dv, MoeBufs& b, uint32_t C) {
  const uint32_t tmax = kp::tmax(C), B = kk::kExperts + 1;
  const uint32_t per_gu = uint32_t(kk::kPfBatchBytes / (size_t(H) * 2 * I * 2)), per_dn = uint32_t(kk::kPfBatchBytes / (size_t(I) * H * 2));
  const std::string v = kk::pf_moe_variant();
  dv.go(v, "kol_pf_sort", 1, 1, 1,
        {PtrArg(b.rt.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.tiles.ptr()), PtrArg(b.row_tok.ptr()),
         PtrArg(b.pair_row.ptr()), arg_val(C), arg_val(tmax)});
  dv.go(v, "kol_pf_gather", tmax * kk::kPfTm, 1, 1,
        {PtrArg(b.x.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.row_tok.ptr()), PtrArg(b.xg.ptr())});
  Walked w;
  w.hdr = dv.rd<uint32_t>(b.hdr, kp::hdr_words());
  w.tiles = dv.rd<uint32_t>(b.tiles, size_t(tmax) * 2);
  w.row_tok = dv.rd<uint32_t>(b.row_tok, size_t(tmax) * kk::kPfTm);
  w.pair_row = dv.rd<uint32_t>(b.pair_row, size_t(C) * kk::kTopK);
  w.xg = dv.rd<uint16_t>(b.xg, size_t(w.hdr[kp::kHdrRows]) * H);
  for (uint32_t b0 = 0; b0 < B; b0 += per_gu) {
    const uint32_t b1 = std::min(B, b0 + per_gu);
    dv.go(v, "kol_pf_dequant_gu", 2 * I / 16, H / 64, b1 - b0,
          {PtrArg(b.gu.ptr()), PtrArg(b.sgu.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.w.ptr()), arg_val(b0), arg_val(b1)});
    dv.go(kk::pf_gemm_gu_variant(), "pf_moe_gemm", tmax, 2 * I / 256, 1,
          {PtrArg(b.tiles.ptr()), PtrArg(b.xg.ptr()), PtrArg(nullptr), PtrArg(b.w.ptr()), PtrArg(nullptr),
           PtrArg(b.h.ptr()), arg_val(b0), arg_val(b1), arg_val(0u)});
  }
  w.h = dv.rd<uint16_t>(b.h, size_t(tmax) * kk::kPfTm * I);
  for (uint32_t b0 = 0; b0 < B; b0 += per_dn) {
    const uint32_t b1 = std::min(B, b0 + per_dn);
    dv.go(v, "kol_pf_dequant_dn", H / 16, I / 64, b1 - b0,
          {PtrArg(b.dn.ptr()), PtrArg(b.sdn.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.w.ptr()), arg_val(b0), arg_val(b1)});
    dv.go(kk::pf_gemm_dn_variant(), "pf_moe_gemm", tmax, H / 256, 1,
          {PtrArg(b.tiles.ptr()), PtrArg(b.h.ptr()), PtrArg(nullptr), PtrArg(b.w.ptr()), PtrArg(nullptr),
           PtrArg(b.xg.ptr()), arg_val(b0), arg_val(b1), arg_val(0u)});
  }
  w.y = dv.rd<uint16_t>(b.xg, size_t(tmax) * kk::kPfTm * H);
  dv.go(v, "kol_pf_moe_combine", H / 256, C, 1,
        {PtrArg(b.rt.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.pair_row.ptr()), PtrArg(b.xg.ptr()), PtrArg(b.mo.ptr()),
         arg_val(C)});
  w.mo = dv.rd<uint16_t>(b.mo, size_t(C) * H);
  return w;
}

void moe_block(Dev& dv, const std::vector<uint16_t>& x, const std::vector<uint32_t>& routes, uint32_t C) {
  const std::vector<uint32_t> gu = rblocks(kr::gate_up_block_words() * kk::kExperts, 40),
                              dn = rblocks(kr::down_block_words() * kk::kExperts, 41);
  const std::vector<uint16_t> sgr = rbf16(size_t(2 * I) * H, -0.03f, 0.03f, 42), sdr = rbf16(size_t(H) * I, -0.05f, 0.05f, 43);
  std::vector<uint16_t> sgu(sgr.size()), sdn(sdr.size());
  common::repack_bf16_tiled(sgr.data(), H, 2 * I, sgu.data());
  common::repack_bf16_tiled(sdr.data(), I, H, sdn.data());
  const uint32_t T = kp::tmax(kC), R = T * kk::kPfTm;
  MoeBufs b{dv.up(x), dv.up(routes), dv.zeros(kp::hdr_words() * 4), dv.zeros(size_t(T) * 8), dv.zeros(size_t(R) * 4),
            dv.zeros(size_t(kC) * 6 * 4), dv.zeros(size_t(R) * H * 2), dv.zeros(size_t(R) * I * 2),
            dv.zeros(kk::kPfBatchBytes), dv.zeros(size_t(kC) * H * 2), dv.up(gu), dv.up(dn), dv.up(sgu), dv.up(sdn)};
  const Walked w = moe_walk(dv, b, C);
  const kp::Sorted st = kp::sort(routes.data(), C);
  CHECK(w.hdr == st.hdr);
  CHECK(std::equal(st.tiles.begin(), st.tiles.end(), w.tiles.begin()));
  CHECK(std::equal(st.row_tok.begin(), st.row_tok.begin() + st.rows_used(), w.row_tok.begin()));
  CHECK(w.pair_row == st.pair_row);
  CHECK(w.xg == kp::gather(st, x.data()));
  // grouped == dense, per (token, slot): the busiest expert, expert 0 when routed, the shared expert
  std::vector<uint32_t> load(kk::kExperts, 0);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < kk::kTopK; ++j) ++load[routes[size_t(t) * 32 + j]];
  const uint32_t busiest = uint32_t(std::max_element(load.begin(), load.end()) - load.begin());
  l0::Mem dx = dv.up(x), hd = dv.up(w.h);
  for (uint32_t e : {busiest, 0u, kp::kShared}) {
    if (e < kk::kExperts && load[e] == 0) continue;
    const std::vector<uint16_t> bgu = kp::dequant_gu(gu.data(), sgu.data(), e);
    l0::Mem db = dv.up(bgu), xs = dv.zeros(size_t(kC) * I * 2);
    const runtime::prefill::GemmBatch gb{C, H, 2 * I, 1, H, 2 * I, 0, 0, 0, 0};
    runtime::prefill::gemm_l0_silu(dv.cx, dv.kc, gb, dx.as<uint16_t>(), db.as<uint16_t>(), xs.as<uint16_t>(), I);
    dv.cx.wait();
    const std::vector<uint16_t> dense = dv.rd<uint16_t>(xs, size_t(C) * I);
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j <= kk::kTopK; ++j) {
        const bool hit = j < kk::kTopK ? routes[size_t(t) * 32 + j] == e : e == kp::kShared;
        if (!hit) continue;
        const uint32_t r = j < kk::kTopK ? st.pair_row[size_t(t) * kk::kTopK + j] : st.shared_row() + t;
        CHECK(std::equal(dense.begin() + size_t(t) * I, dense.begin() + size_t(t + 1) * I, w.h.begin() + size_t(r) * I));
      }
    const std::vector<uint16_t> bdn = kp::dequant_dn(dn.data(), sdn.data(), e);
    l0::Mem ddn = dv.up(bdn), yd = dv.zeros(size_t(R) * H * 4);
    const uint32_t rows = st.rows_used();
    const runtime::prefill::GemmBatch gd{rows, I, H, 1, I, H, H, 0, 0, 0};
    runtime::prefill::gemm_l0(dv.cx, dv.kc, gd, hd.as<uint16_t>(), ddn.as<uint16_t>(), yd.as<float>(), false);
    dv.cx.wait();
    const std::vector<float> yf = dv.rd<float>(yd, size_t(rows) * H);
    for (uint32_t tile = 0; tile < st.tiles_used; ++tile) {
      if (st.tiles[2 * tile] != e) continue;
      for (uint32_t i = 0; i < kk::kPfTm; ++i) {
        const uint32_t r = st.tiles[2 * tile + 1] + i;
        if (st.row_tok[r] == kp::kNone) continue;
        for (uint32_t n = 0; n < H; ++n) CHECK_EQ(w.y[size_t(r) * H + n], rne(yf[size_t(r) * H + n]));
      }
    }
  }
  // the combine exactly the reference's over the device's y
  std::vector<uint16_t> mo_ref(size_t(C) * H);
  kp::combine(routes.data(), st, w.y.data(), mo_ref.data(), C);
  CHECK(w.mo == mo_ref);
  // Review Focus 2: the chunk reversed
  std::vector<uint16_t> xr(x.size(), 0);
  std::vector<uint32_t> rtr(routes.size(), 0);
  for (uint32_t t = 0; t < C; ++t) {
    const uint32_t u = C - 1 - t;
    std::copy(x.begin() + size_t(t) * H, x.begin() + size_t(t + 1) * H, xr.begin() + size_t(u) * H);
    std::copy(routes.begin() + size_t(t) * 32, routes.begin() + size_t(t + 1) * 32, rtr.begin() + size_t(u) * 32);
  }
  dv.imm.copy(b.x.ptr(), xr.data(), xr.size() * 2);
  dv.imm.copy(b.rt.ptr(), rtr.data(), rtr.size() * 4);
  const Walked v = moe_walk(dv, b, C);
  for (uint32_t t = 0; t < C; ++t) {
    const uint32_t u = C - 1 - t;
    CHECK(std::equal(w.mo.begin() + size_t(t) * H, w.mo.begin() + size_t(t + 1) * H, v.mo.begin() + size_t(u) * H));
    for (uint32_t j = 0; j < kk::kTopK; ++j) {
      const uint32_t ra = w.pair_row[size_t(t) * 6 + j], rb = v.pair_row[size_t(u) * 6 + j];
      CHECK(std::equal(w.h.begin() + size_t(ra) * I, w.h.begin() + size_t(ra + 1) * I, v.h.begin() + size_t(rb) * I));
      CHECK(std::equal(w.y.begin() + size_t(ra) * H, w.y.begin() + size_t(ra + 1) * H, v.y.begin() + size_t(rb) * H));
    }
  }
  // replay: the forward chunk again, bitwise
  dv.imm.copy(b.x.ptr(), x.data(), x.size() * 2);
  dv.imm.copy(b.rt.ptr(), routes.data(), routes.size() * 4);
  const Walked again = moe_walk(dv, b, C);
  CHECK(again.mo == w.mo && again.h == w.h && again.y == w.y);
  // Review Focus 3: the shared expert alone - every routed weight 0, so mo = rne(0 + shared row) = that row
  std::vector<uint32_t> zw = routes;
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < kk::kTopK; ++j) zw[size_t(t) * 32 + 8 + j] = 0u;
  dv.imm.copy(b.rt.ptr(), zw.data(), zw.size() * 4);
  const Walked sh = moe_walk(dv, b, C);
  for (uint32_t t = 0; t < C; ++t)
    CHECK(std::equal(sh.mo.begin() + size_t(t) * H, sh.mo.begin() + size_t(t + 1) * H,
                     sh.y.begin() + size_t(st.shared_row() + t) * H));
  std::printf("5. MoE block over %u rows: sort / gather exact; grouped == dense bitwise (experts %u, 0, the shared "
              "block 384); combine exact; reversed chunk and replay bitwise; the shared expert alone = its row\n",
              C, busiest);
}

// The sort at C = 2048 on the adversaries (kolibri_pf_ref_test's): all to 6 experts, the tile bound reached.
void sort_adversaries(Dev& dv) {
  const uint32_t T = kp::tmax(kC), R = T * kk::kPfTm;
  for (int kind = 0; kind < 2; ++kind) {
    std::vector<uint32_t> list;
    if (kind == 0)
      for (uint32_t e : {0u, 127u, 255u, 256u, 382u, 383u}) list.insert(list.end(), kC, e);
    else
      for (uint32_t e = 0; e < kk::kExperts; ++e) list.insert(list.end(), e < 372 ? 33u : 1u, e);
    std::vector<uint32_t> rows(size_t(kC) * 32, 0);
    for (uint32_t t = 0; t < kC; ++t) {
      uint32_t ids[6];
      for (uint32_t j = 0; j < 6; ++j) ids[j] = list[size_t(j) * kC + t];
      std::sort(ids, ids + 6);
      for (uint32_t j = 0; j < 6; ++j) rows[size_t(t) * 32 + j] = ids[j];
    }
    l0::Mem rt = dv.up(rows), hdr = dv.zeros(kp::hdr_words() * 4), tiles = dv.zeros(size_t(T) * 8),
            row_tok = dv.zeros(size_t(R) * 4), pair_row = dv.zeros(size_t(kC) * 6 * 4);
    dv.go(kk::pf_moe_variant(), "kol_pf_sort", 1, 1, 1,
          {PtrArg(rt.ptr()), PtrArg(hdr.ptr()), PtrArg(tiles.ptr()), PtrArg(row_tok.ptr()), PtrArg(pair_row.ptr()),
           arg_val(kC), arg_val(T)});
    const kp::Sorted st = kp::sort(rows.data(), kC);
    CHECK(dv.rd<uint32_t>(hdr, kp::hdr_words()) == st.hdr);
    CHECK(dv.rd<uint32_t>(tiles, size_t(T) * 2) == st.tiles);
    const std::vector<uint32_t> rtok = dv.rd<uint32_t>(row_tok, size_t(R));
    CHECK(std::equal(st.row_tok.begin(), st.row_tok.begin() + st.rows_used(), rtok.begin()));
    CHECK(dv.rd<uint32_t>(pair_row, size_t(kC) * 6) == st.pair_row);
    if (kind == 1) CHECK_EQ(st.tiles_used, T);
  }
  std::printf("   sort at C %u: all to 6 experts (lanes 255 / 256 / 383) and the tile bound reached (%u tiles) exact\n",
              kC, T);
}

// ---- 6. the flash attention -------------------------------------------------------------------------
void flash(Dev& dv) {
  struct Case { uint32_t pos, C; bool sliding; };
  const Case cases[] = {{0, 2048, false}, {2048, 2048, false}, {30000, 2048, false}, {60000, 2048, false},
                        {0, 2048, true},  {1000, 2048, true},  {4396, 2048, true},   {1001, 300, true},
                        {5000, 1, true},  {5000, 1, false}};
  const uint32_t maxd = 62048;
  const std::vector<uint16_t> kf = rbf16(size_t(maxd) * KVN, -1.f, 1.f, 50), vf = rbf16(size_t(maxd) * KVN, -1.f, 1.f, 51);
  const std::vector<uint16_t> kring = rbf16(size_t(kk::kRing) * KVN, -1.f, 1.f, 52),
                              vring = rbf16(size_t(kk::kRing) * KVN, -1.f, 1.f, 53);
  l0::Mem dkf = dv.up(kf), dvf = dv.up(vf), dkr = dv.up(kring), dvr = dv.up(vring);
  std::vector<float> q = rf32(size_t(kC) * QN, -1.f, 1.f, 54);
  for (float& v : q) v = rf(v);   // kol_attn_prep's q are bf16 values
  l0::Mem dq = dv.up(q), out = dv.zeros(size_t(kC) * QN * 2), out2 = dv.zeros(size_t(kC) * QN * 2);
  double worst_flash = 1.0, worst_eager = 1.0, worst_eager64 = 1.0;
  for (const Case& c : cases) {
    const uint16_t* hk = c.sliding ? kring.data() : kf.data();
    const uint16_t* hv = c.sliding ? vring.data() : vf.data();
    const void* K = c.sliding ? dkr.ptr() : dkf.ptr();
    const void* V = c.sliding ? dvr.ptr() : dvf.ptr();
    std::vector<uint32_t> rows = {0, c.C - 1};
    for (uint32_t r : {1u, 7u, 23u, 511u, 512u, 513u})
      if (r < c.C) rows.push_back(r);
    const uint32_t hstep = c.sliding ? 1 : 3;
    for (int eager = 0; eager <= 1; ++eager) {
      dv.go(kk::pf_flash_variant(c.sliding, eager), "kol_pf_flash_attn", (c.C + 7) / 8, KVH, QH / KVH / kk::kPfHpw,
            {PtrArg(dq.ptr()), PtrArg(K), PtrArg(V), PtrArg(out.ptr()), arg_val(c.pos), arg_val(c.C)});
      const std::vector<uint16_t> o = dv.rd<uint16_t>(out, size_t(c.C) * QN);
      for (uint32_t r : rows)
        for (uint32_t h = 0; h < QH; h += hstep) {
          const uint16_t* got = o.data() + (size_t(r) * QH + h) * HD;
          const std::vector<double> want = kp::flash_row(q.data(), hk, hv, c.pos, c.C, r, c.sliding, h);
          if (!eager) {
            worst_flash = std::min(worst_flash, cosine16(want.data(), got, HD));
          } else {
            worst_eager64 = std::min(worst_eager64, cosine16(want.data(), got, HD));
            const kp::EagerRow e = kp::eager_row(q.data() + (size_t(r) * QH + h) * HD, hk, hv, c.pos + r, c.sliding, h);
            std::vector<double> ed(HD);
            for (uint32_t d = 0; d < HD; ++d) ed[d] = f32(e.o[d]);
            worst_eager = std::min(worst_eager, cosine16(ed.data(), got, HD));
          }
        }
      if (c.pos == 1000 && c.sliding) {
        // replay bitwise; a chunk split at a multiple of 64: rows 64.. of (1000, 2048) == rows of (1064, 1984)
        dv.go(kk::pf_flash_variant(c.sliding, eager), "kol_pf_flash_attn", (c.C + 7) / 8, KVH, QH / KVH / kk::kPfHpw,
              {PtrArg(dq.ptr()), PtrArg(K), PtrArg(V), PtrArg(out2.ptr()), arg_val(c.pos), arg_val(c.C)});
        CHECK(dv.rd<uint16_t>(out2, size_t(c.C) * QN) == o);
        std::vector<float> q2(q.begin() + size_t(64) * QN, q.end());
        q2.resize(q.size(), 0.0f);
        l0::Mem dq2 = dv.up(q2);
        dv.go(kk::pf_flash_variant(c.sliding, eager), "kol_pf_flash_attn", (c.C - 64 + 7) / 8, KVH, QH / KVH / kk::kPfHpw,
              {PtrArg(dq2.ptr()), PtrArg(K), PtrArg(V), PtrArg(out2.ptr()), arg_val(c.pos + 64), arg_val(c.C - 64)});
        const std::vector<uint16_t> o2 = dv.rd<uint16_t>(out2, size_t(c.C - 64) * QN);
        CHECK(std::equal(o2.begin(), o2.end(), o.begin() + size_t(64) * QN));
      }
    }
    std::printf("   (%s, pos %5u, C %4u): flash worst %.7f, eager worst %.7f vs eager_window / %.6f vs fp64\n",
                c.sliding ? "sliding" : "full", c.pos, c.C, worst_flash, worst_eager, worst_eager64);
  }
  CHECK(worst_flash >= 0.99999);
  CHECK(worst_eager >= 0.9999);
  CHECK(worst_eager64 >= 0.999);
  std::printf("6. flash attention at GQA 12: sliding over the ring (window 513) and full to depth 62048: default %.7f "
              "(bar 0.99999), eager %.6f vs eager_window (0.9999) / %.6f vs fp64 (0.999); a split at 64 and replay "
              "bitwise\n", worst_flash, worst_eager, worst_eager64);
}

}  // namespace

int main() {
  Dev dv;
  std::printf("kolibri_pf_kernels_test on %s\n", dv.ctx.name().c_str());
  slabs(dv);
  norms(dv);
  attn_prep(dv);
  const uint32_t C = 64;
  const std::vector<uint16_t> x = rbf16(size_t(kC) * H, -0.2f, 0.2f, 60);
  const std::vector<uint32_t> routes = routers(dv, C, x);
  moe_block(dv, x, routes, C);
  sort_adversaries(dv);
  flash(dv);
  std::printf("kolibri_pf_kernels_test OK\n");
  return 0;
}
