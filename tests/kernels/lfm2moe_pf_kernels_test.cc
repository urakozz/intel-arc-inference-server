// Spec 24d Task 2 (F2 for the prefill): LFM2.5-24B-A2B's prefill kernels on the card, at the real shapes, against
// tests/kernels/lfm2moe_pf_ref.h / lfm2moe_ref.h and against the dense prefill GEMMs the grouped ones are built from.
// Needs a B70, no checkpoint (synthetic weights). Every binary is one runtime/lfm2moe/lfm2moe_prefill.cc binds
// (kernels::lfm2moe names), plus pf_gemm_T0 / pf_gemm_T0_SILU for the dense comparisons. Written blind on the Mac (box
// queue row 52).
//
//   1. the slabs follow each linear's layout (24b review I1): pf_dequant_slab at in_proj 2048 x 6144 (layout 1, from
//      loader::repack_in_proj), q||k||v 2048 x 3072 (layout 0, repack_fused_linear) and the dense down 11776 x 2048
//      (layout 1) - every slab exactly bf16((q - 8) x scale) of the GPTQ source column; one slab through pf_gemm_T0
//      within cosine 0.99999 of fp64 host sums
//   2. lfm_pf_embed_gather: the chunk's rows out of the tiles, bitwise
//   3. the conv over a chunk (Review Focus 1): lfm_pf_conv_seed + lfm_pf_conv at (p0, C) = (0, 1), (0, 2), (5, 17),
//      (1, 2048), (30, 64), (4097, 2048) from a garbage ring - y, the seed and the whole ring bitwise conv_chunk's; the
//      conv-tile build (_T) on the permuted in_proj bitwise the plain one's
//   4. 24c's per-row kernels at M = kPfC: lfm_res_norm _NF / _S1 over 2048 rows (h_out and x bitwise), lfm_router over
//      2048 rows bitwise the host's order, lfm_route ids exact (the other words counted: sigmoid's exp), lfm_attn_prep
//      S 1 at pos 4090 over 37 rows (q, k, v bitwise)
//   5. the MoE block over C = 64 (Review Focus 3): sort / gather exact; the dequantised blocks exact; GROUPED ==
//      DENSE, bitwise per (token, slot), for the busiest expert: gate||up (two weight batches, b0 = 0 and 42) against
//      pf_gemm_T0_SILU over that expert's gathered rows, down against rne(pf_gemm_T0) over its h rows; the combine
//      and fold exactly lfm2moe_pf_ref's; the sort at C = 2048 on the bound (318 tiles) and one expert in every row,
//      exactly the host's
//   6. flash at head_dim 64 (Review Focus 4), against flash_row (fp64; cosine >= 0.99999, spec 6 K1) at (pos, C) =
//      (0, 1), (0, 63), (0, 64), (0, 2048), (4096, 64), (131071, 1); rows 0, 1, 7, 8, 63, C - 1; the EAGER build
//      against eager_row (0.9999) and fp64 (0.999); a chunk split at a multiple of 64 bitwise (rows of (1000, 2048)
//      == rows of (1064, 1984)); a replay bitwise
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
#include "kernels/lfm2moe_kernels.h"
#include "kernels/lfm2moe_pf_ref.h"
#include "kernels/lfm2moe_ref.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/lfm2moe_repack.h"
#include "model/lfm2moe.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"

namespace {

namespace kl = kernels::lfm2moe;
namespace lr = lfmref;
namespace lp = lfmpfref;
using lr::f32;
using lr::rf;
using lr::rne;
using runtime::prefill::arg_val;
using runtime::prefill::KernelArg;
using runtime::prefill::PtrArg;

constexpr uint32_t kC = kl::kPfC;
const uint32_t H = kl::kHidden, HD = kl::kHd, QH = kl::kQHeads, KVH = kl::kKvHeads, QN = QH * HD, KVN = KVH * HD,
               I = kl::kInter, E = kl::kExperts;

static_assert(kl::kPfC == lp::kC && kl::kPfTm == lp::kTm && kl::kPfKt == lp::kKt && kl::kPfConvRows == lp::kRows,
              "kernels::lfm2moe's prefill constants and lfm2moe_pf_ref's disagree");
static_assert(kl::pf_hdr::words() == lp::hdr_words() && kl::pf_hdr::kRows == lp::kHdrRows &&
                  kl::pf_hdr::kCount == lp::kHdrCount,
              "the sort's header moved");

std::vector<uint16_t> rbf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(d(g));
  return v;
}
std::vector<float> rf32(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& e : v) e = d(g);
  return v;
}
// Random layout-1 blocks: nibbles uniform, f16 scales in [0.01, 0.05].
std::vector<uint32_t> rblocks(size_t words, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(0.01f, 0.05f);
  std::vector<uint32_t> v(words);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = g();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(d(g))) | (uint32_t(common::f32_to_f16(d(g))) << 16);
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
    l0::Mem m(ctx, l0::MemKind::Device, std::max<size_t>(v.size() * sizeof(T), 64));
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
    return m;
  }
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, std::max<size_t>(bytes, 64));
    imm.fill(m.ptr(), 0u, std::max<size_t>(bytes, 64));
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
struct Ctrl {
  l0::Mem m;
  explicit Ctrl(Dev& d) : m(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control)) {
    std::memset(m.ptr(), 0, sizeof(runtime::Control));
  }
  void set(uint32_t pos, uint32_t n) {
    m.as<runtime::Control>()->pos = pos;
    m.as<runtime::Control>()->n_active = n;
  }
};

// A GPTQ source [K/8][N] + f16 scales [K/64][N] and its column n's bf16 dequant at k.
struct Gptq {
  uint32_t K, N;
  std::vector<uint32_t> q;
  std::vector<uint16_t> s;
  uint16_t at(uint32_t k, uint32_t n) const {
    const int v = int((q[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 0xF) - 8;
    return rne(float(v) * common::f16_to_f32(s[size_t(k / 64) * N + n]));
  }
};
Gptq gptq(uint32_t K, uint32_t N, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(0.01f, 0.05f);
  Gptq w{K, N, std::vector<uint32_t>(size_t(K / 8) * N), std::vector<uint16_t>(size_t(K / 64) * N)};
  for (uint32_t& v : w.q) v = g();
  for (uint16_t& v : w.s) v = common::f32_to_f16(d(g));
  return w;
}

// ---- 1. the slabs at each linear's layout ---------------------------------------------------------------------
void slabs(Dev& dv) {
  const model::Lfm2MoeDesc& d = model::lfm2moe();
  struct Case {
    model::LfmLinearId id;
    uint32_t layer;
  };
  for (const Case& c : {Case{model::LfmLinearId::InProj, 0}, Case{model::LfmLinearId::Qkv, 2},
                        Case{model::LfmLinearId::DenseDown, 0}}) {
    model::LfmLinear ln{};
    for (const model::LfmLinear& x : d.linears(c.layer))
      if (x.id == c.id) ln = x;
    const uint32_t K = ln.shape.K, N = ln.shape.N;
    const Gptq src = gptq(K, N, 100 + uint32_t(c.id));
    loader::HostLinear hl;
    if (c.id == model::LfmLinearId::InProj) {
      hl = loader::repack_in_proj(src.q.data(), src.s.data(), K, N, false, ln.shape.layout);
    } else {   // one GPTQ part holding the fused columns (the column order is the part's)
      loader::LinearSrc ls;
      ls.kind = loader::WKind::Int4;
      ls.K = K;
      ls.N = N;
      ls.qweight = src.q.data();
      ls.scales = src.s.data();
      model::LfmLinear one = ln;
      one.fuse = model::Fuse::Single;
      one.parts = {ln.parts[0]};
      one.part_n = {N};
      hl = loader::repack_fused_linear(one, {ls});
    }
    l0::Mem w = dv.up(hl.words), sc = dv.up(hl.scales.empty() ? std::vector<uint16_t>(32, 0) : hl.scales),
            slab = dv.zeros(size_t(K) * kl::kPfSlab * 2);
    for (uint32_t n0 : {0u, N - kl::kPfSlab}) {
      // pf_dequant_slab(w, scales, out, n0), grid (1024 / 16, K / 64)
      dv.go(kl::pf_slab_variant(ln.shape), "pf_dequant_slab", kl::kPfSlab / 16, K / 64, 1,
            {PtrArg(w.ptr()), PtrArg(sc.ptr()), PtrArg(slab.ptr()), arg_val(n0)});
      const std::vector<uint16_t> got = dv.rd<uint16_t>(slab, size_t(K) * kl::kPfSlab);
      for (uint32_t k = 0; k < K; k += (K > 4096 ? 37 : 13))
        for (uint32_t cc = 0; cc < kl::kPfSlab; ++cc) CHECK_EQ(got[size_t(k) * kl::kPfSlab + cc], src.at(k, n0 + cc));
    }
    if (c.id == model::LfmLinearId::InProj) {   // one slab through pf_gemm_T0
      const uint32_t M = 37;
      const std::vector<uint16_t> x = rbf16(size_t(kC) * K, -1.f, 1.f, 3);
      l0::Mem dx = dv.up(x), part = dv.zeros(size_t(kC) * kl::kPfSlab * 4);
      dv.go(kl::pf_slab_variant(ln.shape), "pf_dequant_slab", kl::kPfSlab / 16, K / 64, 1,
            {PtrArg(w.ptr()), PtrArg(sc.ptr()), PtrArg(slab.ptr()), arg_val(0u)});
      const runtime::prefill::GemmBatch gb{M, K, kl::kPfSlab, 1, K, kl::kPfSlab, kl::kPfSlab, 0, 0, 0};
      runtime::prefill::gemm_l0(dv.cx, dv.kc, gb, dx.as<uint16_t>(), slab.as<uint16_t>(), part.as<float>(), false);
      dv.cx.wait();
      const std::vector<float> got = dv.rd<float>(part, size_t(M) * kl::kPfSlab);
      double worst = 1.0;
      for (uint32_t m = 0; m < M; m += 6) {
        std::vector<double> want(kl::kPfSlab, 0.0);
        for (uint32_t n = 0; n < kl::kPfSlab; ++n)
          for (uint32_t k = 0; k < K; ++k) want[n] += double(f32(x[size_t(m) * K + k])) * f32(src.at(k, n));
        worst = std::min(worst, cosine(want.data(), got.data() + size_t(m) * kl::kPfSlab, kl::kPfSlab));
      }
      CHECK(worst >= 0.99999);
      std::printf("1. in_proj's first slab through pf_gemm_T0: rows worst cosine %.7f\n", worst);
    }
    std::printf("1. pf_dequant_slab %s: layout %u (the descriptor's), slabs at n0 0 and %u exactly the GPTQ columns\n",
                kl::pf_slab_variant(ln.shape).c_str(), ln.shape.layout, N - kl::kPfSlab);
  }
}

// ---- 2. the embedding gather -----------------------------------------------------------------------------------
void embed(Dev& dv) {
  const uint32_t C = 40;
  const std::vector<uint16_t> tiles = rbf16(size_t(H) * 32, -1.f, 1.f, 5);   // the tiles of ids 0 .. 31 (2 n-tiles)
  std::vector<uint32_t> ids(C);
  for (uint32_t m = 0; m < C; ++m) ids[m] = (m * 7 + 3) % 32;
  l0::Mem di = dv.up(ids), dt = dv.up(tiles), h = dv.zeros(size_t(C) * H * 2);
  dv.go(kl::pf_embed_variant(), "lfm_pf_embed_gather", 1, C, 1,
        {PtrArg(di.ptr()), PtrArg(dt.ptr()), PtrArg(h.ptr()), arg_val(C)});
  const std::vector<uint16_t> got = dv.rd<uint16_t>(h, size_t(C) * H);
  for (uint32_t m = 0; m < C; ++m) {
    const std::vector<uint16_t> want = lr::embed_row(tiles.data(), ids[m]);
    CHECK(std::equal(want.begin(), want.end(), got.begin() + size_t(m) * H));
  }
  std::printf("2. lfm_pf_embed_gather: %u rows out of the tiles bitwise\n", C);
}

// ---- 3. the conv over a chunk ----------------------------------------------------------------------------------
void conv(Dev& dv) {
  const std::vector<float> taps = [] {
    std::vector<float> t(size_t(2048) * 3);
    const std::vector<uint16_t> b = rbf16(t.size(), -0.6f, 0.6f, 12);
    for (size_t i = 0; i < t.size(); ++i) t[i] = f32(b[i]);
    return t;
  }();
  const std::vector<uint16_t> garbage = rbf16(size_t(lp::kRing) * H, -50.f, 50.f, 13);
  l0::Mem dtaps = dv.up(taps), ring = dv.zeros(size_t(lp::kRing) * H * 2), seed = dv.zeros(size_t(2) * H * 2),
          y = dv.zeros(size_t(kC) * H * 2);
  Ctrl ctl(dv);
  for (bool tiled : {false, true})
    for (const auto& [p0, C] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {0, 2}, {5, 17}, {1, 2048}, {30, 64},
                                                                           {4097, 2048}}) {
      const std::vector<float> plain = rf32(size_t(C) * 3 * H, -3.f, 3.f, 20 + p0 + C);
      std::vector<float> in = plain;
      if (tiled)
        for (uint32_t t = 0; t < C; ++t)
          for (uint32_t c = 0; c < H; ++c)
            for (uint32_t part = 0; part < 3; ++part)
              in[size_t(t) * 3 * H + lp::conv_col(c, part, H, true)] = plain[size_t(t) * 3 * H + lp::conv_col(c, part, H, false)];
      l0::Mem din = dv.up(in);
      dv.imm.copy(ring.ptr(), garbage.data(), garbage.size() * 2);
      ctl.set(p0, C);
      const std::string v = kl::pf_conv_variant(tiled);
      // lfm_pf_conv_seed(ctrl, ring, seed), grid (H / 256); lfm_pf_conv(ctrl, inproj, seed, taps, ring, y),
      // grid (H / 256, ceil(C / 16))
      dv.go(v, "lfm_pf_conv_seed", H / kl::kPfConvWg, 1, 1, {PtrArg(ctl.m.ptr()), PtrArg(ring.ptr()), PtrArg(seed.ptr())});
      dv.go(v, "lfm_pf_conv", H / kl::kPfConvWg, (C + kl::kPfConvRows - 1) / kl::kPfConvRows, 1,
            {PtrArg(ctl.m.ptr()), PtrArg(din.ptr()), PtrArg(seed.ptr()), PtrArg(dtaps.ptr()), PtrArg(ring.ptr()),
             PtrArg(y.ptr())});
      std::vector<uint16_t> hring = garbage;
      const std::vector<uint16_t> hseed = lp::conv_seed(hring.data(), p0);
      const std::vector<uint16_t> hy = lp::conv_chunk(in.data(), hseed.data(), taps.data(), hring.data(), p0, C, H, tiled);
      CHECK(dv.rd<uint16_t>(seed, size_t(2) * H) == hseed);
      CHECK(dv.rd<uint16_t>(y, size_t(C) * H) == hy);
      CHECK(dv.rd<uint16_t>(ring, size_t(lp::kRing) * H) == hring);
      std::printf("3. lfm_pf_conv%s at (p0 %u, C %u): seed, y and the ring bitwise conv_chunk's\n", tiled ? " _T" : "",
                  p0, C);
    }
}

// ---- 4. 24c's per-row kernels at M = kPfC ----------------------------------------------------------------------
void per_row(Dev& dv) {
  {   // the norm, _NF and _S1, over 2048 rows
    const std::vector<uint16_t> h = rbf16(size_t(kC) * H, -2.f, 2.f, 30);
    const std::vector<float> sl = rf32(size_t(kC) * H, -0.5f, 0.5f, 31);
    std::vector<float> w(H);
    for (uint32_t k = 0; k < H; ++k) w[k] = rf(0.3f + float(k % 11) * 0.1f);
    l0::Mem dh = dv.up(h), ds = dv.up(sl), dw = dv.up(w), ho = dv.zeros(size_t(kC) * H * 2), x = dv.zeros(size_t(kC) * H * 2);
    for (bool fold : {false, true}) {
      // lfm_res_norm(h_in, slices, w, h_out, x), grid (8, C)
      dv.go(kl::pf_norm_variant(fold), "lfm_res_norm", kl::kNormWgs, kC, 1,
            {PtrArg(dh.ptr()), PtrArg(ds.ptr()), PtrArg(dw.ptr()), PtrArg(ho.ptr()), PtrArg(x.ptr())});
      const lr::NormOut want = lr::res_norm(h.data(), sl.data(), fold ? 1 : 0, kC, w.data());
      CHECK(dv.rd<uint16_t>(x, size_t(kC) * H) == want.x);
      if (fold) CHECK(dv.rd<uint16_t>(ho, size_t(kC) * H) == want.h_out);
    }
    std::printf("4. lfm_res_norm _NF / _S1 at M 2048: x and the fold bitwise over 2048 rows\n");
  }
  {   // the router and the route over 2048 rows
    const std::vector<uint16_t> x = rbf16(size_t(kC) * H, -1.f, 1.f, 32), gw = rbf16(size_t(E) * H, -0.05f, 0.05f, 33);
    std::vector<uint16_t> tiles(size_t(H) * E);
    common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
    std::vector<float> bias = rf32(E, -0.05f, 0.05f, 34);
    l0::Mem dx = dv.up(x), dt = dv.up(tiles), lg = dv.zeros(size_t(kC) * E * 4), db = dv.up(bias),
            rt = dv.zeros(size_t(kC) * lr::kWords * 4);
    dv.go(kl::pf_router_variant(), "lfm_router", E / 16, kC, 1, {PtrArg(dx.ptr()), PtrArg(dt.ptr()), PtrArg(lg.ptr())});
    dv.go(kl::pf_route_variant(), "lfm_route", 1, kC, 1, {PtrArg(lg.ptr()), PtrArg(db.ptr()), PtrArg(rt.ptr())});
    const std::vector<float> got = dv.rd<float>(lg, size_t(kC) * E);
    const std::vector<uint32_t> routes = dv.rd<uint32_t>(rt, size_t(kC) * lr::kWords);
    size_t other = 0;
    for (uint32_t m = 0; m < kC; m += 3) {
      const std::vector<float> want = lr::router(x.data() + size_t(m) * H, tiles.data());
      for (uint32_t e = 0; e < E; ++e) CHECK_EQ(lr::as_u32(got[size_t(m) * E + e]), lr::as_u32(want[e]));
      const lr::Route r = lr::route(got.data() + size_t(m) * E, bias.data());
      for (uint32_t j = 0; j < 4; ++j) CHECK_EQ(routes[size_t(m) * lr::kWords + lr::kIds + j], r.ids[j]);
      for (uint32_t w = 4; w < lr::kWords; ++w) other += routes[size_t(m) * lr::kWords + w] != r.row[w];
    }
    std::printf("4. lfm_router at M 2048 bitwise the host order; lfm_route ids exact, %zu other words off (sigmoid's "
                "exp)\n", other);
  }
  {   // the attention prep at S 1 over 37 rows at pos 4090
    const uint32_t C = 37, pos = 4090, L = 8192;
    const std::vector<float> table = loader::lfm_rope_table(model::lfm2moe(), L);
    const std::vector<float> sl = rf32(size_t(kC) * kl::kQkvN, -1.f, 1.f, 35);
    std::vector<float> qkn(2 * HD);
    for (uint32_t i = 0; i < 2 * HD; ++i) qkn[i] = rf(0.8f + float(i % 13) * 0.03f);
    l0::Mem rb = dv.up(table), sb = dv.up(sl), qb = dv.up(qkn), aq = dv.zeros(size_t(kC) * QN * 4),
            kb = dv.zeros(size_t(L) * KVN * 2), vb = dv.zeros(size_t(L) * KVN * 2);
    Ctrl ctl(dv);
    ctl.set(pos, C);
    // lfm_attn_prep(ctrl, slices, qkn, rope, attn_q, kv_k, kv_v), grid (q + 2 kv heads, C)
    dv.go(kl::pf_attn_prep_variant(), "lfm_attn_prep", QH + 2 * KVH, C, 1,
          {PtrArg(ctl.m.ptr()), PtrArg(sb.ptr()), PtrArg(qb.ptr()), PtrArg(rb.ptr()), PtrArg(aq.ptr()), PtrArg(kb.ptr()),
           PtrArg(vb.ptr())});
    for (uint32_t m = 0; m < C; ++m) {
      const lr::Prep want = lr::attn_prep(sl.data(), 1, kC, m, qkn.data(), table.data() + size_t(pos + m) * HD);
      CHECK(dv.rd<float>(aq, QN, size_t(m) * QN) == want.q);
      CHECK(dv.rd<uint16_t>(kb, KVN, size_t(pos + m) * KVN) == want.k);
      CHECK(dv.rd<uint16_t>(vb, KVN, size_t(pos + m) * KVN) == want.v);
    }
    std::printf("4. lfm_attn_prep at M 2048 (S 1): 37 rows at pos 4090 - q, k, v bitwise\n");
  }
}

// ---- 5. the MoE block ------------------------------------------------------------------------------------------
std::vector<uint32_t> route_rows(uint32_t C, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> dl(-4.f, 4.f), db(-0.05f, 0.05f);
  std::vector<float> bias(E);
  for (float& b : bias) b = db(g);
  std::vector<uint32_t> route(size_t(C) * lr::kWords);
  for (uint32_t t = 0; t < C; ++t) {
    std::vector<float> lg(E);
    for (float& v : lg) v = dl(g);
    const lr::Route r = lr::route(lg.data(), bias.data());
    std::copy(r.row, r.row + lr::kWords, route.begin() + size_t(t) * lr::kWords);
  }
  return route;
}
struct SortBufs {
  l0::Mem hdr, tiles, row_tok, pair_row;
  SortBufs(Dev& dv)
      : hdr(dv.zeros(size_t(kl::pf_hdr::words()) * 4)), tiles(dv.zeros(size_t(lp::tmax(kC)) * 2 * 4)),
        row_tok(dv.zeros(size_t(lp::tmax(kC)) * lp::kTm * 4)), pair_row(dv.zeros(size_t(kC) * 4 * 4)) {}
};
void sort_on(Dev& dv, SortBufs& s, const l0::Mem& rt, uint32_t C) {
  // lfm_pf_sort(route, hdr, tiles, row_tok, pair_row, C, tmax), grid (1)
  dv.go(kl::pf_moe_variant(), "lfm_pf_sort", 1, 1, 1,
        {PtrArg(rt.ptr()), PtrArg(s.hdr.ptr()), PtrArg(s.tiles.ptr()), PtrArg(s.row_tok.ptr()), PtrArg(s.pair_row.ptr()),
         arg_val(C), arg_val(lp::tmax(C))});
}
void check_sort(Dev& dv, SortBufs& s, const lp::Sorted& want, uint32_t C) {
  CHECK(dv.rd<uint32_t>(s.hdr, lp::kHdrCount + E) ==
        std::vector<uint32_t>(want.hdr.begin(), want.hdr.begin() + lp::kHdrCount + E));
  CHECK(dv.rd<uint32_t>(s.tiles, size_t(lp::tmax(C)) * 2) == want.tiles);
  CHECK(dv.rd<uint32_t>(s.row_tok, want.rows_used()) ==
        std::vector<uint32_t>(want.row_tok.begin(), want.row_tok.begin() + want.rows_used()));
  CHECK(dv.rd<uint32_t>(s.pair_row, size_t(C) * 4) == want.pair_row);
}

void moe(Dev& dv) {
  const uint32_t C = 64;
  const std::vector<uint32_t> route = route_rows(C, 40);
  const lp::Sorted st = lp::sort(route.data(), C);
  const uint32_t T = lp::tmax(kC), R = T * lp::kTm + 256;   // + 256: the dense GEMM's padded rows stay inside
  const std::vector<uint16_t> x = rbf16(size_t(kC) * H, -1.f, 1.f, 41);
  const std::vector<uint32_t> gu = rblocks(size_t(E) * lr::gate_up_block_words(), 42),
                              dn = rblocks(size_t(E) * lr::down_block_words(), 43);
  l0::Mem drt = dv.up(route), dx = dv.up(x), dgu = dv.up(gu), ddn = dv.up(dn), xg = dv.zeros(size_t(R) * H * 2),
          hm = dv.zeros(size_t(R) * I * 2), yg = dv.zeros(size_t(R) * H * 2);
  SortBufs sb(dv);
  sort_on(dv, sb, drt, C);
  check_sort(dv, sb, st, C);
  // lfm_pf_gather(x, hdr, row_tok, xg), grid (tmax x TM)
  dv.go(kl::pf_moe_variant(), "lfm_pf_gather", T * lp::kTm, 1, 1,
        {PtrArg(dx.ptr()), PtrArg(sb.hdr.ptr()), PtrArg(sb.row_tok.ptr()), PtrArg(xg.ptr())});
  CHECK(dv.rd<uint16_t>(xg, size_t(st.rows_used()) * H) == lp::gather(st, x.data()));
  // gate||up in two weight batches ([0, 42), [42, 64)) and down in one, the walk's arrangement
  const uint32_t per_gu = 42;
  l0::Mem wgu = dv.zeros(size_t(per_gu) * H * 2 * I * 2), wdn = dv.zeros(size_t(E) * I * H * 2);
  for (uint32_t b0 = 0; b0 < E; b0 += per_gu) {
    const uint32_t b1 = std::min(E, b0 + per_gu);
    // lfm_pf_dequant_gu(w, hdr, out, b0, b1), grid (2I / 16, H / 64, b1 - b0)
    dv.go(kl::pf_moe_variant(), "lfm_pf_dequant_gu", 2 * I / 16, H / 64, b1 - b0,
          {PtrArg(dgu.ptr()), PtrArg(sb.hdr.ptr()), PtrArg(wgu.ptr()), arg_val(b0), arg_val(b1)});
    if (b0 == 0)   // every routed expert's block exact (an unrouted one is skipped: its slot holds the zeros)
      for (uint32_t e = b0; e < b1; ++e)
        if (st.hdr[lp::kHdrCount + e])
          CHECK(dv.rd<uint16_t>(wgu, size_t(H) * 2 * I, size_t(e - b0) * H * 2 * I) == lp::dequant_gu(gu.data(), e));
    // pf_moe_gemm(tiles, A, -, B, -, out, b0, b1, 0), grid (tmax, N / 256)
    dv.go(kl::pf_gemm_gu_variant(), "pf_moe_gemm", T, 2 * I / kernels::pf_moe::kGemmWgN, 1,
          {PtrArg(sb.tiles.ptr()), PtrArg(xg.ptr()), PtrArg(nullptr), PtrArg(wgu.ptr()), PtrArg(nullptr), PtrArg(hm.ptr()),
           arg_val(b0), arg_val(b1), arg_val(0u)});
  }
  dv.go(kl::pf_moe_variant(), "lfm_pf_dequant_dn", H / 16, I / 64, E,
        {PtrArg(ddn.ptr()), PtrArg(sb.hdr.ptr()), PtrArg(wdn.ptr()), arg_val(0u), arg_val(E)});
  dv.go(kl::pf_gemm_dn_variant(), "pf_moe_gemm", T, H / kernels::pf_moe::kGemmWgN, 1,
        {PtrArg(sb.tiles.ptr()), PtrArg(hm.ptr()), PtrArg(nullptr), PtrArg(wdn.ptr()), PtrArg(nullptr), PtrArg(yg.ptr()),
         arg_val(0u), arg_val(E), arg_val(0u)});
  // GROUPED == DENSE for the busiest expert: its rows [r0, r0 + n) through pf_gemm_T0_SILU / pf_gemm_T0
  uint32_t eb = 0;
  for (uint32_t e = 1; e < E; ++e)
    if (st.hdr[lp::kHdrCount + e] > st.hdr[lp::kHdrCount + eb]) eb = e;
  uint32_t r0 = 0;
  while (st.tiles[2 * (r0 / lp::kTm)] != eb) r0 += lp::kTm;
  const uint32_t n = st.hdr[lp::kHdrCount + eb];
  {
    l0::Mem blk = dv.zeros(size_t(H) * 2 * I * 2), xd = dv.zeros(size_t(256) * I * 2);
    dv.go(kl::pf_moe_variant(), "lfm_pf_dequant_gu", 2 * I / 16, H / 64, 1,
          {PtrArg(dgu.ptr()), PtrArg(sb.hdr.ptr()), PtrArg(blk.ptr()), arg_val(eb), arg_val(eb + 1)});
    const runtime::prefill::GemmBatch gb{n, H, 2 * I, 1, H, 2 * I, 0, 0, 0, 0};
    runtime::prefill::gemm_l0_silu(dv.cx, dv.kc, gb, xg.as<uint16_t>() + size_t(r0) * H, blk.as<uint16_t>(),
                                   xd.as<uint16_t>(), I);
    dv.cx.wait();
    CHECK(dv.rd<uint16_t>(xd, size_t(n) * I) == dv.rd<uint16_t>(hm, size_t(n) * I, size_t(r0) * I));
    l0::Mem bd = dv.zeros(size_t(I) * H * 2), cd = dv.zeros(size_t(256) * H * 4);
    dv.go(kl::pf_moe_variant(), "lfm_pf_dequant_dn", H / 16, I / 64, 1,
          {PtrArg(ddn.ptr()), PtrArg(sb.hdr.ptr()), PtrArg(bd.ptr()), arg_val(eb), arg_val(eb + 1)});
    const runtime::prefill::GemmBatch gd{n, I, H, 1, I, H, H, 0, 0, 0};
    runtime::prefill::gemm_l0(dv.cx, dv.kc, gd, hm.as<uint16_t>() + size_t(r0) * I, bd.as<uint16_t>(), cd.as<float>(),
                              false);
    dv.cx.wait();
    const std::vector<float> c = dv.rd<float>(cd, size_t(n) * H);
    const std::vector<uint16_t> yy = dv.rd<uint16_t>(yg, size_t(n) * H, size_t(r0) * H);
    for (size_t i = 0; i < c.size(); ++i) CHECK_EQ(yy[i], rne(c[i]));
  }
  // the combine and the fold
  const std::vector<uint16_t> h0 = rbf16(size_t(C) * H, -2.f, 2.f, 44);
  l0::Mem dh = dv.up(h0);
  // lfm_pf_combine(route, pair_row, y, h, C), grid (H / 256, C)
  dv.go(kl::pf_moe_variant(), "lfm_pf_combine", H / kl::kPfCombineWg, C, 1,
        {PtrArg(drt.ptr()), PtrArg(sb.pair_row.ptr()), PtrArg(yg.ptr()), PtrArg(dh.ptr()), arg_val(C)});
  std::vector<uint16_t> hh = h0;
  const std::vector<uint16_t> ydev = dv.rd<uint16_t>(yg, size_t(st.rows_used()) * H);
  lp::combine(route.data(), st, ydev.data(), hh.data(), C);
  CHECK(dv.rd<uint16_t>(dh, size_t(C) * H) == hh);
  std::printf("5. MoE over C = 64: sort, gather and the dequantised blocks exact; grouped == dense for expert %u (%u "
              "rows) gate||up and down bitwise; the rank-order combine + fold bitwise the host's\n", eb, n);
  // the sort at C = 2048: the bound (63 experts at 129 rows, one at 65) and one expert in every row
  for (int adv = 0; adv < 2; ++adv) {
    std::vector<uint32_t> rt(size_t(kC) * lr::kWords, 0);
    if (adv == 0) {
      std::vector<uint32_t> list;
      for (uint32_t e = 0; e < E; ++e) list.insert(list.end(), e == 63 ? 8192 - 63 * 129 : 129, e);
      for (uint32_t t = 0; t < kC; ++t)
        for (uint32_t j = 0; j < 4; ++j) rt[size_t(t) * lr::kWords + j] = list[size_t(j) * kC + t];
    } else {
      for (uint32_t t = 0; t < kC; ++t) {
        rt[size_t(t) * lr::kWords + 0] = 9;
        for (uint32_t j = 1; j < 4; ++j) rt[size_t(t) * lr::kWords + j] = (9 + j * (1 + t % 20)) % E;
      }
    }
    const lp::Sorted want = lp::sort(rt.data(), kC);
    l0::Mem d2 = dv.up(rt);
    sort_on(dv, sb, d2, kC);
    check_sort(dv, sb, want, kC);
    std::printf("5. lfm_pf_sort at C 2048, %s: %u / %u tiles, exactly the host's\n",
                adv == 0 ? "the bound" : "expert 9 in every row", want.tiles_used, lp::tmax(kC));
  }
}

// ---- 6. flash at head_dim 64 -----------------------------------------------------------------------------------
void flash(Dev& dv) {
  const uint32_t L = 131072;
  const std::vector<uint16_t> kc = rbf16(size_t(L) * KVN, -1.f, 1.f, 50), vc = rbf16(size_t(L) * KVN, -1.f, 1.f, 51);
  l0::Mem dk = dv.up(kc), dvv = dv.up(vc), out = dv.zeros(size_t(kC) * QN * 2);
  auto run = [&](bool eager, const std::vector<float>& q, uint32_t pos, uint32_t C) {
    l0::Mem dq = dv.up(q);
    // lfm_pf_attn(q, kv_k, kv_v, out, pos, C), grid (ceil(C / RPW), kv heads, GQA / HPW)
    dv.go(kl::pf_attn_variant(eager), "lfm_pf_attn", (C + kl::kPfRpw - 1) / kl::kPfRpw, KVH, QH / KVH / kl::kPfHpw,
          {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(out.ptr()), arg_val(pos), arg_val(C)});
    return dv.rd<uint16_t>(out, size_t(C) * QN);
  };
  auto qrows = [](uint32_t C, uint32_t seed) {
    const std::vector<uint16_t> b = rbf16(size_t(C) * QN, -2.f, 2.f, seed);
    std::vector<float> q(b.size());
    for (size_t i = 0; i < b.size(); ++i) q[i] = f32(b[i]);
    return q;
  };
  double fworst = 1, eworst = 1, e64 = 1;
  for (const auto& [pos, C] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {0, 63}, {0, 64}, {0, 2048}, {4096, 64},
                                                                          {131071, 1}}) {
    const std::vector<float> q = qrows(C, 60 + C + pos % 997);
    const std::vector<uint16_t> fo = run(false, q, pos, C), eo = run(true, q, pos, C);
    for (uint32_t t : {0u, 1u, 7u, 8u, 63u, C - 1})
      if (t < C)
        for (uint32_t h : {0u, 3u, 4u, 31u}) {
          const std::vector<double> want = lp::flash_row(q.data(), kc.data(), vc.data(), pos, C, t, h);
          fworst = std::min(fworst, cosine16(want.data(), fo.data() + (size_t(t) * QH + h) * HD, HD));
          e64 = std::min(e64, cosine16(want.data(), eo.data() + (size_t(t) * QH + h) * HD, HD));
          const std::vector<uint16_t> er = lp::eager_row(q.data() + (size_t(t) * QH + h) * HD, kc.data(), vc.data(),
                                                         pos + t, h);
          std::vector<double> erd(HD);
          for (uint32_t dd = 0; dd < HD; ++dd) erd[dd] = f32(er[dd]);
          eworst = std::min(eworst, cosine16(erd.data(), eo.data() + (size_t(t) * QH + h) * HD, HD));
        }
  }
  CHECK(fworst >= 0.99999);
  CHECK(eworst >= 0.9999);
  CHECK(e64 >= 0.999);
  // a split at a multiple of 64: rows of (1000, 2048) == rows of (1064, 1984); a replay bitwise
  const std::vector<float> q = qrows(2048, 70);
  const std::vector<uint16_t> a = run(false, q, 1000, 2048), a2 = run(false, q, 1000, 2048);
  const std::vector<float> qb(q.begin() + size_t(64) * QN, q.end());
  const std::vector<uint16_t> b = run(false, qb, 1064, 1984);
  CHECK(a == a2);
  CHECK(std::equal(b.begin(), b.end(), a.begin() + size_t(64) * QN));
  std::printf("6. lfm_pf_attn at head_dim 64: flash vs fp64 worst cosine %.7f (bar 0.99999); EAGER vs the reference's "
              "rounding points %.6f, vs fp64 %.6f; a split at 64 and a replay bitwise\n", fworst, eworst, e64);
}

}  // namespace

int main() {
  Dev dv;
  std::printf("lfm2moe_pf_kernels_test on %s\n", dv.ctx.name().c_str());
  slabs(dv);
  embed(dv);
  conv(dv);
  per_row(dv);
  moe(dv);
  flash(dv);
  std::puts("lfm2moe_pf_kernels_test OK");
  return 0;
}
