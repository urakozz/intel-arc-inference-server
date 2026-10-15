// Spec 23d (MG2 for the prefill): Muse-Glimmer's prefill kernels on the card, at the real shapes, against
// tests/kernels/muse_pf_ref.h / muse_ref.h / k2_pf_ref.h. Needs a B70, no checkpoint (synthetic data). Every binary
// is one runtime/muse/muse_prefill.cc binds (kernels::muse names). Written blind on the Mac (box queue row 45,
// r45.k1).
//
//   1. the slabs: k2_pf_dequant_slab at Muse's four linears (the first and the last slab; q||gate||k||v's, o_proj's and
//      down's last is 512 wide) exact; q||gate||k||v through pf_gemm_T0 over C = 37 rows within cosine 0.99999 of
//      fp64 host sums;
//   2. the between-GEMM kernels at M 2048: mg_pf_embed (3 ids, the placeholder, sums at the pitch), mg_norm,
//      pf_res_fold _Z K 6656 + mg_post_add, mg_attn_prep S 1 at 4090 over 37 rows (the ring's wrap: slots 4090..4095,
//      0..30) and full at 37, mg_silu_mul S 1, mg_pf_attn_gate - bit-exact against museref (sqrt / divide correctly
//      rounded on ocloc's command line; exp_torch on both sides);
//   3. the attention (Review Focus 1 / 2): the flash build against musepf::flash_row_fp64 at cosine >= 0.99999 -
//      sliding at c0 = 0, 1000, 4396 (wrapped), 6143 and a tail chunk (1001, 300) and single rows (5000, 1); full at
//      (0, 2048), (2048, 2048), (28000, 2048) (depth 30048) and (5000, 1); rows 0, 1, 7, 8, 23, 63, 64, 511, 1000 and
//      C - 1. The EAGER build against musepf::eager_window BITWISE (decode's eager row; the chunks with depth x C
//      small enough for the three-pass walk). A chunk split at a multiple of 64 bitwise (rows 64.. of (1000, 2048) ==
//      the rows of (1064, 1984)), a split at 2049 (rows of (1000, 1049) and (2049, 999) against (1000, 2048)) printed
//      bitwise or not and within the cosine bar, replay bitwise.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/k2_pf_ref.h"
#include "kernels/kernels.h"
#include "kernels/muse_kernels.h"
#include "kernels/muse_pf_ref.h"
#include "kernels/muse_ref.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/control.h"
#include "runtime/muse/muse_rope.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"

namespace {

namespace km = kernels::muse;
namespace mr = museref;
namespace mp = musepf;
using mr::bf;
using mr::rf;
using mr::rne;
using runtime::prefill::arg_val;
using runtime::prefill::KernelArg;
using runtime::prefill::PtrArg;

constexpr uint32_t kC = km::kPfC;
const uint32_t H = km::kHidden, HD = km::kHd, QH = km::kQHeads, KVH = km::kKvHeads, QN = km::kQN, KVN = km::kKvN,
               G = km::kNormG, I = km::kInter, QGKV = km::kQgkvN;
static_assert(km::kPfC == mp::kC && km::kPfKt == mp::kKt && km::kPfRpw == mp::kRpw && km::kPfEagerWg == mp::kEagerTile,
              "kernels::muse's prefill constants and musepf's disagree");

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
double cosine16(const double* a, const uint16_t* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    const double y = bf(b[i]);
    ab += a[i] * y;
    aa += a[i] * a[i];
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

uint32_t slab_width(uint32_t N, uint32_t n0) { return N - n0 >= km::kPfSlab ? km::kPfSlab : (N - n0 + 255) / 256 * 256; }

// ---- 1. the slabs and one linear ---------------------------------------------------------------------------------
void slabs(Dev& dv) {
  struct L { uint32_t K, N; };
  for (const L& l : {L{H, QGKV}, L{QN, H}, L{H, km::kGateUpN}, L{I, H}}) {
    Rng r(l.K + l.N);
    std::vector<uint32_t> words(size_t(l.K / 8) * l.N);
    for (uint32_t& w : words) w = r.u32();
    std::vector<uint16_t> sc(size_t(l.K / 64) * l.N);
    for (uint16_t& s : sc) s = common::f32_to_f16(r.uni(0.01f, 0.05f));
    l0::Mem dw = dv.up(words), ds = dv.up(sc), slab = dv.zeros(size_t(l.K) * km::kPfSlab * 2);
    const uint32_t last = (l.N - 1) / km::kPfSlab * km::kPfSlab;
    for (uint32_t n0 : {0u, last}) {
      const uint32_t ns = slab_width(l.N, n0);
      dv.go(km::pf_slab_variant(l.K, l.N), "k2_pf_dequant_slab", ns / 16, l.K / 64, 1,
            {PtrArg(dw.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0), arg_val(ns)});
      CHECK(dv.rd<uint16_t>(slab, size_t(l.K) * ns) == k2_pf_ref::dequant_slab(words.data(), sc.data(), l.K, l.N, n0, ns));
    }
    if (l.N != QGKV) continue;
    // q||gate||k||v through pf_gemm_T0 over 37 rows into partials [kC][8704]
    const uint32_t C = 37;
    const std::vector<uint16_t> x = rbf16(size_t(kC) * l.K, -1.f, 1.f, 3);
    l0::Mem dx = dv.up(x), part = dv.up(std::vector<float>(size_t(kC) * l.N, 7.0f));
    for (uint32_t n0 = 0; n0 < l.N;) {
      const uint32_t ns = slab_width(l.N, n0);
      dv.go(km::pf_slab_variant(l.K, l.N), "k2_pf_dequant_slab", ns / 16, l.K / 64, 1,
            {PtrArg(dw.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0), arg_val(ns)});
      const runtime::prefill::GemmBatch gb{C, l.K, ns, 1, l.K, ns, l.N, 0, 0, 0};
      runtime::prefill::gemm_l0(dv.cx, dv.kc, gb, dx.as<uint16_t>(), slab.as<uint16_t>(), part.as<float>() + n0, false);
      dv.cx.wait();
      n0 += ns;
    }
    const std::vector<float> got = dv.rd<float>(part, size_t(C) * l.N);
    double worst = 1.0;
    for (uint32_t m = 0; m < C; m += 6) {
      std::vector<double> want(l.N, 0.0);
      for (uint32_t n0 = 0; n0 < l.N; n0 += km::kPfSlab) {
        const uint32_t ns = slab_width(l.N, n0);
        const std::vector<uint16_t> s = k2_pf_ref::dequant_slab(words.data(), sc.data(), l.K, l.N, n0, ns);
        for (uint32_t k = 0; k < l.K; ++k)
          for (uint32_t c = 0; c < ns && n0 + c < l.N; ++c)
            want[n0 + c] += double(bf(x[size_t(m) * l.K + k])) * bf(s[size_t(k) * ns + c]);
      }
      double ab = 0, aa = 0, bb = 0;
      for (uint32_t n = 0; n < l.N; ++n) {
        ab += want[n] * got[size_t(m) * l.N + n];
        aa += want[n] * want[n];
        bb += double(got[size_t(m) * l.N + n]) * got[size_t(m) * l.N + n];
      }
      worst = std::min(worst, ab / std::sqrt(aa * bb));
    }
    std::printf("1. q||gate||k||v 6656 x 8704 through 9 slabs + pf_gemm_T0 (37 rows): worst row cosine %.7f\n", worst);
    CHECK(worst >= 0.99999);
  }
  std::printf("1. k2_pf_dequant_slab at 6656x8704, 4096x6656, 6656x39936, 19968x6656: first and last slabs exact\n");
}

// ---- 2. the between-GEMM kernels at M 2048 -----------------------------------------------------------------------
void preps(Dev& dv) {
  {   // mg_pf_embed: 3 ids into a 202048-row table's first rows (only those are read)
    const std::vector<uint16_t> table = rbf16(size_t(8) * H, -3.f, 3.f, 10);
    std::vector<uint16_t> big(size_t(km::kVocab) * H, 0);
    std::copy(table.begin(), table.end(), big.begin());
    const std::vector<uint32_t> ids = {3, km::kImageToken, 7};
    l0::Mem di = dv.up(ids), dt = dv.up(big), dr = dv.zeros(size_t(kC) * H * 2), ds = dv.zeros(size_t(G) * kC * 4);
    dv.go(km::pf_embed_variant(), "mg_pf_embed", 1, 3, 1, {PtrArg(di.ptr()), PtrArg(dt.ptr()), PtrArg(dr.ptr()), PtrArg(ds.ptr())});
    const std::vector<uint16_t> r = dv.rd<uint16_t>(dr, size_t(3) * H);
    const std::vector<float> s = dv.rd<float>(ds, size_t(G) * kC);
    for (uint32_t m = 0; m < 3; ++m) {
      std::vector<uint16_t> want(H);
      std::vector<float> ws(G);
      mr::embed_norm_sums(table.data() + size_t(m == 1 ? 0 : ids[m]) * H, H, want.data(), ws.data());
      CHECK(std::equal(want.begin(), want.end(), r.begin() + size_t(m) * H));
      for (uint32_t g = 0; g < G; ++g) CHECK(s[size_t(g) * kC + m] == ws[g]);
    }
  }
  const uint32_t C = 37;
  {   // mg_norm; pf_res_fold _Z + mg_post_add
    const std::vector<uint16_t> resid = rbf16(size_t(kC) * H, -2.f, 2.f, 11);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(-0.4f + float(k % 11) * 0.08f);
    std::vector<float> ss(size_t(G) * kC, 0.f);
    for (uint32_t m = 0; m < C; ++m) {
      std::vector<float> s1(G);
      mr::chunk_sumsq(resid.data() + size_t(m) * H, H, G, s1.data());
      for (uint32_t g = 0; g < G; ++g) ss[size_t(g) * kC + m] = s1[g];
    }
    l0::Mem ds = dv.up(ss), dr = dv.up(resid), dw = dv.up(w), dx = dv.zeros(size_t(kC) * H * 2);
    dv.go(km::pf_norm_variant(), "mg_norm", km::kNormW, C, 1, {PtrArg(ds.ptr()), PtrArg(dr.ptr()), PtrArg(dw.ptr()), PtrArg(dx.ptr())});
    const std::vector<uint16_t> x = dv.rd<uint16_t>(dx, size_t(C) * H);
    for (uint32_t m = 0; m < C; ++m) {
      std::vector<float> s1(G);
      for (uint32_t g = 0; g < G; ++g) s1[g] = ss[size_t(g) * kC + m];
      std::vector<uint16_t> want(H);
      mr::centered_norm_from(s1.data(), G, resid.data() + size_t(m) * H, w.data(), mr::kEps, H, want.data());
      CHECK(std::equal(want.begin(), want.end(), x.begin() + size_t(m) * H));
    }
    const std::vector<float> part = rf32(size_t(kC) * H, -0.5f, 0.5f, 12);
    l0::Mem dp = dv.up(part), da = dv.zeros(size_t(kC) * H * 2), dsa = dv.zeros(size_t(G) * kC * 4),
            dso = dv.zeros(size_t(G) * kC * 4);
    dv.go(km::pf_fold_zero_variant(), "pf_res_fold", G, C, 1,
          {PtrArg(dp.ptr()), PtrArg(da.ptr()), PtrArg(dsa.ptr()), arg_val(kC)});
    dv.go(km::pf_post_add_variant(), "mg_post_add", G, C, 1,
          {PtrArg(dsa.ptr()), PtrArg(da.ptr()), PtrArg(dw.ptr()), PtrArg(dr.ptr()), PtrArg(dso.ptr())});
    const std::vector<uint16_t> r2 = dv.rd<uint16_t>(dr, size_t(C) * H);
    const std::vector<float> so = dv.rd<float>(dso, size_t(G) * kC);
    for (uint32_t m = 0; m < C; ++m) {
      std::vector<uint16_t> a(H), r(resid.begin() + size_t(m) * H, resid.begin() + size_t(m + 1) * H);
      for (uint32_t k = 0; k < H; ++k) a[k] = rne(part[size_t(m) * H + k]);
      std::vector<float> sa(G), so_ref(G);
      mr::chunk_sumsq(a.data(), H, G, sa.data());
      mr::post_add_sums(sa.data(), G, a.data(), w.data(), r.data(), so_ref.data(), H);
      CHECK(std::equal(r.begin(), r.end(), r2.begin() + size_t(m) * H));
      for (uint32_t g = 0; g < G; ++g) CHECK(so[size_t(g) * kC + m] == so_ref[g]);
    }
  }
  {   // mg_attn_prep S 1: sliding at 4090 over 37 rows (slots 4090..4095, 0..30), full at 37
    const std::vector<float> rope = runtime::muse::rope_table(HD, 5e5f, 4200);
    l0::Mem drope = dv.up(rope);
    for (int sliding = 1; sliding >= 0; --sliding) {
      const uint32_t pos = sliding ? 4090 : 37, rows = sliding ? km::kRing : 128;
      const std::vector<float> part = rf32(size_t(kC) * QGKV, -1.f, 1.f, 20 + sliding);
      runtime::Control ctl{};
      ctl.pos = pos;
      ctl.n_active = C;
      l0::Mem dctl(dv.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
      std::memcpy(dctl.ptr(), &ctl, sizeof ctl);
      l0::Mem dp = dv.up(part), dq = dv.zeros(size_t(kC) * QN * 4), dg = dv.zeros(size_t(kC) * QN * 2),
              dk = dv.zeros(size_t(rows) * KVN * 2), dvv = dv.zeros(size_t(rows) * KVN * 2);
      dv.go(km::pf_attn_prep_variant(sliding != 0), "mg_attn_prep", km::kPrepGroups, C, 1,
            {PtrArg(dctl.ptr()), PtrArg(dp.ptr()), PtrArg(drope.ptr()), PtrArg(dq.ptr()), PtrArg(dg.ptr()),
             PtrArg(dk.ptr()), PtrArg(dvv.ptr())});
      const std::vector<float> q = dv.rd<float>(dq, size_t(C) * QN);
      const std::vector<uint16_t> g = dv.rd<uint16_t>(dg, size_t(C) * QN), k = dv.rd<uint16_t>(dk, size_t(rows) * KVN),
                                  v = dv.rd<uint16_t>(dvv, size_t(rows) * KVN);
      for (uint32_t m = 0; m < C; ++m) {
        std::vector<uint16_t> qi(QN), ki(KVN), qo(QN), ko(KVN);
        for (uint32_t i = 0; i < QN; ++i) {
          qi[i] = mr::partials_col(part.data(), 1, kC, m, QGKV, i);
          CHECK(g[size_t(m) * QN + i] == mr::gate_bf16(mr::partials_col(part.data(), 1, kC, m, QGKV, mr::kOffGate + i)));
        }
        for (uint32_t i = 0; i < KVN; ++i) ki[i] = mr::partials_col(part.data(), 1, kC, m, QGKV, mr::kOffK + i);
        mr::qk_prep(qi.data(), ki.data(), pos + m, sliding != 0, rope.data(), qo.data(), ko.data());
        const size_t row = mr::ring_row(pos + m, sliding != 0);
        for (uint32_t i = 0; i < QN; ++i) CHECK(rne(q[size_t(m) * QN + i]) == qo[i]);
        for (uint32_t i = 0; i < KVN; ++i) {
          CHECK(k[row * KVN + i] == ko[i]);
          CHECK(v[row * KVN + i] == mr::partials_col(part.data(), 1, kC, m, QGKV, mr::kOffV + i));
        }
      }
    }
  }
  {   // mg_silu_mul S 1 and mg_pf_attn_gate
    const std::vector<float> part = rf32(size_t(kC) * 2 * I, -1.f, 1.f, 30);
    l0::Mem dp = dv.up(part), dh = dv.zeros(size_t(kC) * I * 2);
    dv.go(km::pf_silu_variant(), "mg_silu_mul", km::kSiluGroups, C, 1, {PtrArg(dp.ptr()), PtrArg(dh.ptr())});
    const std::vector<uint16_t> h = dv.rd<uint16_t>(dh, size_t(C) * I);
    for (uint32_t m = 0; m < C; m += 5) {
      std::vector<uint16_t> gg(I), u(I), want(I);
      for (uint32_t i = 0; i < I; ++i) {
        gg[i] = mr::partials_col(part.data(), 1, kC, m, 2 * I, i);
        u[i] = mr::partials_col(part.data(), 1, kC, m, 2 * I, I + i);
      }
      mr::silu_mul(gg.data(), u.data(), I, want.data());
      CHECK(std::equal(want.begin(), want.end(), h.begin() + size_t(m) * I));
    }
    std::vector<uint16_t> a = rbf16(size_t(C) * QN, -2.f, 2.f, 31);
    const std::vector<uint16_t> g = rbf16(size_t(C) * QN, 0.f, 1.f, 32);
    l0::Mem dg = dv.up(g), da = dv.up(a);
    dv.go(km::pf_gate_variant(), "mg_pf_attn_gate", km::kPfGateGroups, C, 1, {PtrArg(dg.ptr()), PtrArg(da.ptr())});
    mp::gate_window(a.data(), g.data(), C);
    CHECK(dv.rd<uint16_t>(da, size_t(C) * QN) == a);
  }
  std::printf("2. mg_pf_embed, mg_norm, pf_res_fold _Z + mg_post_add, mg_attn_prep (ring wrap / NoPE), mg_silu_mul, "
              "mg_pf_attn_gate at M 2048: bit-exact\n");
}

// ---- 3. the attention -------------------------------------------------------------------------------------------------
struct Attn {
  Dev& dv;
  const void* K;
  const void* V;
  void run(bool sliding, bool eager, const l0::Mem& q, const l0::Mem& out, uint32_t pos, uint32_t C) {
    if (eager)
      dv.go(km::pf_flash_variant(sliding, true), "mg_pf_flash_attn", QH, C, 1,
            {PtrArg(q.ptr()), PtrArg(K), PtrArg(V), PtrArg(out.ptr()), arg_val(pos), arg_val(C)});
    else
      dv.go(km::pf_flash_variant(sliding, false), "mg_pf_flash_attn", (C + km::kPfRpw - 1) / km::kPfRpw, KVH,
            QH / KVH / km::kPfHpw, {PtrArg(q.ptr()), PtrArg(K), PtrArg(V), PtrArg(out.ptr()), arg_val(pos), arg_val(C)});
  }
};

void attention(Dev& dv) {
  struct Case { uint32_t pos, C; bool sliding; };
  const Case cases[] = {{0, 2048, true},    {1000, 2048, true},  {4396, 2048, true},   {6143, 2048, true},
                        {1001, 300, true},  {5000, 1, true},     {0, 2048, false},     {2048, 2048, false},
                        {28000, 2048, false}, {5000, 1, false}};
  const uint32_t maxd = 30048;
  const std::vector<uint16_t> kf = rbf16(size_t(maxd) * KVN, -1.f, 1.f, 50), vf = rbf16(size_t(maxd) * KVN, -1.f, 1.f, 51);
  const std::vector<uint16_t> kring = rbf16(size_t(km::kRing) * KVN, -1.f, 1.f, 52),
                              vring = rbf16(size_t(km::kRing) * KVN, -1.f, 1.f, 53);
  l0::Mem dkf = dv.up(kf), dvf = dv.up(vf), dkr = dv.up(kring), dvr = dv.up(vring);
  const std::vector<uint16_t> q16 = rbf16(size_t(kC) * QN, -4.f, 4.f, 54);
  std::vector<float> q(q16.size());
  for (size_t i = 0; i < q.size(); ++i) q[i] = bf(q16[i]);
  l0::Mem dq = dv.up(q), out = dv.zeros(size_t(kC) * QN * 2), out2 = dv.zeros(size_t(kC) * QN * 2);
  double worst_flash = 1.0;
  size_t eager_rows = 0;
  for (const Case& c : cases) {
    const uint16_t* hk = c.sliding ? kring.data() : kf.data();
    const uint16_t* hv = c.sliding ? vring.data() : vf.data();
    Attn at{dv, c.sliding ? dkr.ptr() : dkf.ptr(), c.sliding ? dvr.ptr() : dvf.ptr()};
    std::vector<uint32_t> rows = {0, c.C - 1};
    for (uint32_t r : {1u, 7u, 8u, 23u, 63u, 64u, 511u, 1000u})
      if (r < c.C) rows.push_back(r);
    const uint32_t hstep = c.sliding ? 1 : 3;
    at.run(c.sliding, false, dq, out, c.pos, c.C);
    const std::vector<uint16_t> o = dv.rd<uint16_t>(out, size_t(c.C) * QN);
    double case_worst = 1.0;
    for (uint32_t r : rows)
      for (uint32_t h = 0; h < QH; h += hstep) {
        const std::vector<double> want = mp::flash_row_fp64(q16.data(), hk, hv, c.pos, c.C, r, c.sliding, h);
        case_worst = std::min(case_worst, cosine16(want.data(), o.data() + (size_t(r) * QH + h) * HD, HD));
      }
    worst_flash = std::min(worst_flash, case_worst);
    // EAGER: bitwise eager_window on the sampled rows (the three-pass walk at depth x C a test can afford)
    const bool do_eager = size_t(c.sliding ? 2048 : c.pos + c.C) * c.C <= size_t(4096) * 2048;
    if (do_eager) {
      at.run(c.sliding, true, dq, out2, c.pos, c.C);
      const std::vector<uint16_t> e = dv.rd<uint16_t>(out2, size_t(c.C) * QN);
      for (uint32_t r : rows) {
        std::vector<uint16_t> want(QN);
        mp::eager_window(q16.data() + size_t(r) * QN, hk, hv, c.pos + r, 1, c.sliding, want.data());
        CHECK(std::equal(want.begin(), want.end(), e.begin() + size_t(r) * QN));
        ++eager_rows;
      }
    }
    std::printf("   (%s, pos %5u, C %4u): flash worst cosine %.7f%s\n", c.sliding ? "sliding" : "full", c.pos, c.C,
                case_worst, do_eager ? ", eager rows bitwise" : "");
    if (c.pos == 1000 && c.sliding) {
      // replay bitwise; a split at 64: rows 64.. of (1000, 2048) == rows of (1064, 1984); a split at 2049
      at.run(true, false, dq, out2, c.pos, c.C);
      CHECK(dv.rd<uint16_t>(out2, size_t(c.C) * QN) == o);
      std::vector<float> q2(q.begin() + size_t(64) * QN, q.end());
      q2.resize(q.size(), 0.0f);
      l0::Mem dq2 = dv.up(q2);
      at.run(true, false, dq2, out2, c.pos + 64, c.C - 64);
      const std::vector<uint16_t> o2 = dv.rd<uint16_t>(out2, size_t(c.C - 64) * QN);
      CHECK(std::equal(o2.begin(), o2.end(), o.begin() + size_t(64) * QN));
      std::vector<float> q3(q.begin() + size_t(1049) * QN, q.end());
      q3.resize(q.size(), 0.0f);
      l0::Mem dq3 = dv.up(q3);
      at.run(true, false, dq3, out2, 2049, c.C - 1049);
      const std::vector<uint16_t> o3 = dv.rd<uint16_t>(out2, size_t(c.C - 1049) * QN);
      const bool bitwise = std::equal(o3.begin(), o3.end(), o.begin() + size_t(1049) * QN);
      double w = 1.0;
      for (uint32_t r = 0; r < c.C - 1049; r += 37)
        for (uint32_t h = 0; h < QH; ++h) {
          std::vector<double> a(HD);
          for (uint32_t d = 0; d < HD; ++d) a[d] = bf(o[(size_t(1049 + r) * QH + h) * HD + d]);
          w = std::min(w, cosine16(a.data(), o3.data() + (size_t(r) * QH + h) * HD, HD));
        }
      std::printf("   splits: at 64 bitwise; at 2049 %s (worst cosine %.7f); replay bitwise\n",
                  bitwise ? "bitwise" : "not bitwise", w);
      CHECK(w >= 0.99999);
    }
  }
  CHECK(worst_flash >= 0.99999);
  std::printf("3. attention at GQA 16: flash over the ring (window 2048) and NoPE to depth 30048 worst %.7f (bar 0.99999); "
              "eager %zu rows bitwise decode's eager rows\n", worst_flash, eager_rows);
}

}  // namespace

int main() {
  Dev dv;
  std::printf("muse_pf_kernels_test on %s\n", dv.ctx.name().c_str());
  slabs(dv);
  preps(dv);
  attention(dv);
  std::printf("muse_pf_kernels_test OK\n");
  return 0;
}
