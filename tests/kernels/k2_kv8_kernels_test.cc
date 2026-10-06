// Spec 18e Task 1 (K1): K2-Horizon's int8 KV cache kernels (src/kernels/k2/k2_kv8.cl and k2_moe.cl's
// MOVA_STAGE build) on the card at K2's real shapes, against tests/kernels/k2_kv8_ref.h. Needs a
// B70, no checkpoint (synthetic rows). Every binary here is one the decode list or the prefill walk
// binds under --kv-cache int8 (kernels::k2 names).
//
//   1. the decode writer k2_attn_prep_kv8 (M 1, S 2), both builds - dense (V from the fused row)
//      and MoVA (V from the staged row) - at pos 37: q (rotated, fp32), gate, the int8 K / V rows
//      and their fp16 scales BITWISE (the chain is bit-defined: kv8.h hd128), other rows untouched;
//   2. MoVA's staged value experts (MOVA_STAGE): the staging row is bitwise the bf16 binary's V row
//      (the routed mix after the combine, Review Focus 2);
//   3. the prefill writer (M 2048 over a chunk of 37 rows at pos 37, S 1 at the pad256 pitch):
//      bitwise, both builds;
//   4. the decode flash pair k2_attn_decode_kv8 / k2_attn_reduce_kv8 at 6, 300 and 3000 keys against
//      the fp64 attention over the dequantised cache, un-rotated, gated: within 2 bf16 ulps (k2_attn's
//      own bar against its fp64 reference); a replay bitwise;
//   5. the eager kv8 kernels (score / k2_attn_eager.cl's softmax / P·V / reduce) at 6, 300, 3000 and
//      4096 keys against k2_kv8_ref::eager_head: scores and probabilities BITWISE, the output bitwise
//      through gates past softplus's threshold, within 1 ulp below it (OpenCL's exp / log1p);
//   6. the prefill flash k2_pf_flash_attn_kv8, default and EAGER, at depths 1 .. 32048: the ungated
//      _O build's un-rotated o against the fp64 attention (q as the kernel reads it: rne(q_rot)):
//      default >= 0.9999, eager >= 0.999 (PROPOSED: the int8 V scale is folded into P, one bf16
//      rounding more than bf16 KV's 0.99999 / 0.999); the gated build against k2_ref::attn_gate over
//      the _O output within 1 ulp; a replay bitwise.
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
#include "common/kv8.h"
#include "kernels/k2_kernels.h"
#include "kernels/k2_kv8_ref.h"
#include "kernels/k2_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/k2_repack.h"
#include "model/k2_horizon.h"
#include "runtime/control.h"
#include "runtime/k2/k2_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace {

namespace kk = kernels::k2;
namespace kr = k2_kv8_ref;
namespace h128 = common::kv8::hd128;
using k2_ref::f32;
using k2_ref::rf;
using k2_ref::rne;
using runtime::prefill::arg_val;
using runtime::prefill::KernelArg;
using runtime::prefill::PtrArg;

constexpr uint32_t kC = kk::kPfC, kHD = 128;
const model::K2Desc& D = model::k2();

int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
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

l0::Mem control(Dev& dv, uint32_t pos, uint32_t n) {
  l0::Mem c(dv.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  std::memset(c.ptr(), 0, c.size());
  c.as<runtime::Control>()->pos = pos;
  c.as<runtime::Control>()->n_active = n;
  return c;
}

// A device cache of one layer: int8 rows [L][kvh][128] and fp16 scales [L][kvh], K and V.
struct DevCache {
  l0::Mem k, v, ks, vs;
  DevCache(Dev& dv, uint32_t L)
      : k(dv.zeros(size_t(L) * D.kv_n())), v(dv.zeros(size_t(L) * D.kv_n())),
        ks(dv.zeros(size_t(L) * D.kv_heads * 2)), vs(dv.zeros(size_t(L) * D.kv_heads * 2)) {}
  DevCache(Dev& dv, const kr::Cache& c) : k(dv.up(c.k)), v(dv.up(c.v)), ks(dv.up(c.ks)), vs(dv.up(c.vs)) {}
};

// ---- 1-3. the writers ----------------------------------------------------------------------
void writers(Dev& dv) {
  const uint32_t pos = 37, L = 256;
  const std::vector<float> rope = loader::k2_rope_table(D, L);
  l0::Mem drope = dv.up(rope);
  for (int dense = 1; dense >= 0; --dense) {
    // 1. decode, M 1, S 2
    const uint32_t N = dense ? D.attn_dense_n() : D.attn_sparse_n(), S = D.attn_s;
    std::vector<float> part = rf32(size_t(S) * N, -1.f, 1.f, 10 + dense);
    for (uint32_t s = 0; s < S; ++s)   // K's outlier channels
      for (uint32_t j = 0; j < D.kv_heads; ++j) part[size_t(s) * N + D.q_n() + j * kHD + 9] *= 30.0f;
    const std::vector<uint16_t> staged = rbf16(D.kv_n(), -0.5f, 0.5f, 12);
    l0::Mem ctrl = control(dv, pos, 1), dp = dv.up(part), stg = dv.up(staged);
    l0::Mem aq = dv.zeros(size_t(D.q_n()) * 4), ag = dv.zeros(size_t(D.q_n()) * 4);
    DevCache c(dv, L);
    dv.go(kk::attn_prep_kv8_variant(1, N, S, D.q_heads, D.kv_heads, dense), "k2_attn_prep_kv8",
          D.q_heads + D.kv_heads, 1, 1,
          {PtrArg(ctrl.ptr()), PtrArg(dp.ptr()), PtrArg(drope.ptr()), PtrArg(aq.ptr()), PtrArg(ag.ptr()),
           PtrArg(c.k.ptr()), PtrArg(c.v.ptr()), PtrArg(c.ks.ptr()), PtrArg(c.vs.ptr()), PtrArg(stg.ptr())});
    kr::Cache want(L, D.kv_heads);
    const kr::Prep p = kr::prep(part.data(), 1, 0, N, S, rope.data() + size_t(pos) * kHD, D.q_heads, D.kv_heads,
                                pos, dense ? nullptr : staged.data(), want);
    CHECK(dv.rd<float>(aq, D.q_n()) == p.q);
    CHECK(dv.rd<float>(ag, D.q_n()) == p.gate);
    CHECK(dv.rd<int8_t>(c.k, size_t(L) * D.kv_n()) == want.k);   // row pos written, the rest 0
    CHECK(dv.rd<int8_t>(c.v, size_t(L) * D.kv_n()) == want.v);
    CHECK(dv.rd<uint16_t>(c.ks, size_t(L) * D.kv_heads) == want.ks);
    CHECK(dv.rd<uint16_t>(c.vs, size_t(L) * D.kv_heads) == want.vs);
    std::printf("1. decode writer (%s): q, gate, int8 K / V rows and fp16 scales at pos %u bitwise\n",
                dense ? "dense, V from the fused row" : "MoVA, V from the staged row", pos);
  }
  {
    // 2. MoVA staged against the bf16 binary
    const uint32_t H = D.hidden, VN = D.kv_n();
    const size_t vblk = size_t(VN / 16) * (H / 64) * 136;
    const std::vector<uint32_t> vw = rblocks(vblk * D.value_experts, 40);
    const std::vector<uint16_t> x = rbf16(H, -0.1f, 0.1f, 41);
    std::vector<uint32_t> row(32, 0);
    const uint32_t ids[4] = {2, 31, 32, 63};
    const float ws[4] = {0.625f, 0.75f, 0.5f, 0.625f};
    for (uint32_t j = 0; j < 4; ++j) {
      row[j] = ids[j];
      std::memcpy(&row[8 + j], &ws[j], 4);
    }
    l0::Mem ctrl = control(dv, 211, 1), dw = dv.up(vw), dx = dv.up(x), dr = dv.up(row);
    l0::Mem cache = dv.zeros(size_t(256) * VN * 2), stage = dv.zeros(size_t(VN) * 2);
    dv.go(kk::mova_variant(1, D.value_experts, D.value_top_k, H, VN), "k2_mova_value", VN / 16, 1, 1,
          {PtrArg(ctrl.ptr()), PtrArg(dr.ptr()), PtrArg(dx.ptr()), PtrArg(dw.ptr()), PtrArg(cache.ptr())});
    dv.go(kk::mova_stage_variant(1, D.value_experts, D.value_top_k, H, VN), "k2_mova_value", VN / 16, 1, 1,
          {PtrArg(ctrl.ptr()), PtrArg(dr.ptr()), PtrArg(dx.ptr()), PtrArg(dw.ptr()), PtrArg(stage.ptr())});
    CHECK(dv.rd<uint16_t>(stage, VN) == dv.rd<uint16_t>(cache, VN, size_t(211) * VN));
    std::puts("2. MoVA staged (MOVA_STAGE): the staging row is the bf16 binary's V[pos] row bitwise");
  }
  {
    // 3. prefill, M 2048 over C rows, S 1, the pad256 pitch
    const uint32_t C = 37;
    for (int dense = 1; dense >= 0; --dense) {
      const uint32_t ld = runtime::k2::pf_ld(dense ? D.attn_dense_n() : D.attn_sparse_n());
      const std::vector<float> part = rf32(size_t(kC) * ld, -1.f, 1.f, 20 + dense);
      const std::vector<uint16_t> staged = rbf16(size_t(kC) * D.kv_n(), -0.5f, 0.5f, 22);
      l0::Mem ctrl = control(dv, pos, C), dp = dv.up(part), stg = dv.up(staged);
      l0::Mem aq = dv.zeros(size_t(kC) * D.q_n() * 4), ag = dv.zeros(size_t(kC) * D.q_n() * 4);
      DevCache c(dv, L);
      dv.go(kk::pf_attn_prep_kv8_variant(ld, D.q_heads, D.kv_heads, dense), "k2_attn_prep_kv8",
            D.q_heads + D.kv_heads, C, 1,
            {PtrArg(ctrl.ptr()), PtrArg(dp.ptr()), PtrArg(drope.ptr()), PtrArg(aq.ptr()), PtrArg(ag.ptr()),
             PtrArg(c.k.ptr()), PtrArg(c.v.ptr()), PtrArg(c.ks.ptr()), PtrArg(c.vs.ptr()), PtrArg(stg.ptr())});
      kr::Cache want(L, D.kv_heads);
      const std::vector<float> q = dv.rd<float>(aq, size_t(C) * D.q_n()), g = dv.rd<float>(ag, size_t(C) * D.q_n());
      for (uint32_t m = 0; m < C; ++m) {
        const kr::Prep p = kr::prep(part.data(), kC, m, ld, 1, rope.data() + size_t(pos + m) * kHD, D.q_heads,
                                    D.kv_heads, pos, dense ? nullptr : staged.data() + size_t(m) * D.kv_n(), want);
        CHECK(std::equal(p.q.begin(), p.q.end(), q.begin() + size_t(m) * D.q_n()));
        CHECK(std::equal(p.gate.begin(), p.gate.end(), g.begin() + size_t(m) * D.q_n()));
      }
      CHECK(dv.rd<int8_t>(c.k, size_t(L) * D.kv_n()) == want.k);
      CHECK(dv.rd<int8_t>(c.v, size_t(L) * D.kv_n()) == want.v);
      CHECK(dv.rd<uint16_t>(c.ks, size_t(L) * D.kv_heads) == want.ks);
      CHECK(dv.rd<uint16_t>(c.vs, size_t(L) * D.kv_heads) == want.vs);
    }
    std::printf("3. prefill writer (M %u over %u rows at pos %u, both builds): bitwise\n", kC, C, pos);
  }
}

// A random int8 cache of L positions (K with outlier channels), and q / gate rows.
kr::Cache make_cache(uint32_t L, uint64_t seed) {
  kr::Cache c(L, D.kv_heads);
  Rng r(seed);
  uint16_t kb[kHD], vb[kHD];
  for (uint32_t p = 0; p < L; ++p)
    for (uint32_t j = 0; j < D.kv_heads; ++j) {
      for (uint32_t i = 0; i < kHD; ++i) {
        kb[i] = rne(r.uni(-1.f, 1.f) * (i == 9 || i == 73 ? 12.0f : 1.0f));
        vb[i] = rne(r.uni(-1.f, 1.f));
      }
      c.put_k(c.row(p, j), kb);
      c.put_v(c.row(p, j), vb);
    }
  return c;
}

// ---- 4. decode flash ----------------------------------------------------------------------------
void decode_flash(Dev& dv) {
  const uint32_t ML = 4096, QN = D.q_n();
  const kr::Cache c = make_cache(ML, 50);
  DevCache dc(dv, c);
  std::vector<float> q(QN), g(QN);
  {
    Rng r(52);
    for (uint32_t h = 0; h < D.q_heads; ++h) {
      float qb[kHD];
      for (float& e : qb) e = rf(r.uni(-2.f, 2.f));
      h128::rotate_q(qb, &q[size_t(h) * kHD]);   // what k2_attn_prep_kv8 leaves in attn_q
    }
    for (uint32_t i = 0; i < QN; ++i) g[i] = rf(i % 5 == 0 ? 40.0f + float(i % 7) : (i % 5 == 1 ? 28.0f : r.uni(-8.f, 8.f)));
  }
  l0::Mem dq = dv.up(q), dg = dv.up(g), part = dv.zeros(size_t(D.q_heads) * kk::kAttnTgt * kk::kAttnPart * 4);
  l0::Mem out = dv.zeros(size_t(QN) * 2);
  const std::string v = kk::attn_kv8_variant(1, kk::kAttnTgt, D.q_heads, D.kv_heads);
  for (uint32_t pos : {5u, 299u, 2999u}) {
    l0::Mem ctrl = control(dv, pos, 1);
    auto run = [&] {
      dv.go(v, "k2_attn_decode_kv8", D.kv_heads, kk::kAttnTgt, 1,
            {PtrArg(ctrl.ptr()), PtrArg(dq.ptr()), PtrArg(dc.k.ptr()), PtrArg(dc.ks.ptr()), PtrArg(dc.v.ptr()),
             PtrArg(dc.vs.ptr()), PtrArg(part.ptr())});
      dv.go(v, "k2_attn_reduce_kv8", D.q_heads, 1, 1,
            {PtrArg(ctrl.ptr()), PtrArg(part.ptr()), PtrArg(dg.ptr()), PtrArg(out.ptr())});
      return dv.rd<uint16_t>(out, QN);
    };
    const std::vector<uint16_t> o = run();
    int worst = 0;
    for (uint32_t h = 0; h < D.q_heads; ++h) {
      double ref[kHD];
      kr::attend(q.data() + size_t(h) * kHD, c, h / D.gqa(), pos + 1, ref);
      for (uint32_t d = 0; d < kHD; ++d)
        worst = std::max(worst, ulps(o[size_t(h) * kHD + d], k2_ref::attn_gate(float(ref[d]), g[size_t(h) * kHD + d])));
    }
    CHECK(run() == o);   // replay
    std::printf("4. decode flash over int8 at %4u keys: worst %d bf16 ulps against the fp64 attention, "
                "un-rotated and gated; replay bitwise\n", pos + 1, worst);
    CHECK(worst <= 2);
  }
}

// ---- 5. decode eager ----------------------------------------------------------------------------
void decode_eager(Dev& dv) {
  const uint32_t ML = 4096, QN = D.q_n(), QH = D.q_heads;
  const kr::Cache c = make_cache(ML, 60);
  DevCache dc(dv, c);
  std::vector<float> q(QN), g(QN);
  {
    Rng r(62);
    for (uint32_t h = 0; h < QH; ++h) {
      float qb[kHD];
      for (float& e : qb) e = rf(r.uni(-4.f, 4.f));
      h128::rotate_q(qb, &q[size_t(h) * kHD]);
    }
    for (uint32_t i = 0; i < QN; ++i) g[i] = rf((i / 128) % 2 == 0 ? 30.0f + float(i % 11) : r.uni(-4.f, 4.f));
  }
  l0::Mem dq = dv.up(q), dg = dv.up(g), sc = dv.zeros(size_t(QH) * ML * 4);
  l0::Mem part = dv.zeros(size_t(QH) * kk::kAttnTgt * kk::kAttnPart * 4), out = dv.zeros(size_t(QN) * 2);
  const std::string v8 = kk::attn_eager_kv8_variant(1, kk::kAttnTgt, QH, D.kv_heads);
  const std::string vs = kk::attn_eager_variant(1, kk::kAttnTgt, QH, D.kv_heads);
  for (uint32_t pos : {5u, 299u, 2999u, ML - 1}) {
    l0::Mem ctrl = control(dv, pos, 1);
    const uint32_t len = pos + 1, blk = k2_ref::eager_block(pos, 1, kk::kAttnTgt);
    dv.go(v8, "k2_attn_eager_score_kv8", D.kv_heads, kk::kAttnTgt, 1,
          {PtrArg(ctrl.ptr()), PtrArg(dq.ptr()), PtrArg(dc.k.ptr()), PtrArg(dc.ks.ptr()), PtrArg(sc.ptr()), arg_val(ML)});
    const std::vector<float> s = dv.rd<float>(sc, size_t(QH) * ML);
    dv.go(vs, "k2_attn_eager_softmax", QH, 1, 1, {PtrArg(ctrl.ptr()), PtrArg(sc.ptr()), arg_val(ML)});
    const std::vector<float> pr = dv.rd<float>(sc, size_t(QH) * ML);
    dv.go(v8, "k2_attn_eager_pv_kv8", D.kv_heads, kk::kAttnTgt, 1,
          {PtrArg(ctrl.ptr()), PtrArg(sc.ptr()), PtrArg(dc.v.ptr()), PtrArg(dc.vs.ptr()), PtrArg(part.ptr()), arg_val(ML)});
    dv.go(v8, "k2_attn_eager_reduce_kv8", QH, 1, 1,
          {PtrArg(ctrl.ptr()), PtrArg(part.ptr()), PtrArg(dg.ptr()), PtrArg(out.ptr())});
    const std::vector<uint16_t> o = dv.rd<uint16_t>(out, QN);
    size_t s_bad = 0, p_bad = 0, o_exact_bad = 0, o_sp = 0;
    int sp_worst = 0;
    for (uint32_t h = 0; h < QH; ++h) {
      const kr::EagerHead e = kr::eager_head(q.data() + size_t(h) * kHD, c, len, h / D.gqa(), blk);
      for (uint32_t i = 0; i < len; ++i) {
        s_bad += s[size_t(h) * ML + i] != e.s[i];
        p_bad += pr[size_t(h) * ML + i] != e.p[i];
      }
      for (uint32_t d = 0; d < kHD; ++d) {
        const float gate = g[size_t(h) * kHD + d];
        const uint16_t got = o[size_t(h) * kHD + d], want = k2_ref::attn_gate(e.o[d], gate);
        if (gate * k2_ref::kSpBeta > k2_ref::kSpThreshold) {
          o_exact_bad += got != want;
        } else {
          o_sp += got != want;
          sp_worst = std::max(sp_worst, ulps(got, want));
        }
      }
    }
    std::printf("5. eager over int8 at %4u keys (%u-key blocks): scores %zu, probabilities %zu, outputs "
                "(gate past the threshold) %zu differ from k2_kv8_ref; below it %zu within %d ulp\n",
                len, blk, s_bad, p_bad, o_exact_bad, o_sp, sp_worst);
    CHECK_EQ(s_bad, size_t(0));
    CHECK_EQ(p_bad, size_t(0));
    CHECK_EQ(o_exact_bad, size_t(0));
    CHECK(sp_worst <= 1);
  }
}

// ---- 6. prefill flash ---------------------------------------------------------------------------
void prefill_flash(Dev& dv) {
  const uint32_t QH = D.q_heads, KVH = D.kv_heads, QN = QH * kHD;
  struct Case { uint32_t pos, C; };
  const Case cases[] = {{0, 1}, {0, 37}, {0, 2048}, {2011, 37}, {2048, 2048}, {30000, 2048}, {4095, 1}};
  const uint32_t maxd = 32048;
  const kr::Cache c = make_cache(maxd, 70);
  DevCache dc(dv, c);
  std::vector<float> q(size_t(kC) * QN), qk(size_t(kC) * QN);   // rotated (fp32), and as the kernel reads it
  {
    Rng r(72);
    for (uint32_t t = 0; t < kC * QH; ++t) {
      float qb[kHD];
      for (float& e : qb) e = rf(r.uni(-1.f, 1.f));
      h128::rotate_q(qb, &q[size_t(t) * kHD]);
    }
    for (size_t i = 0; i < q.size(); ++i) qk[i] = rf(q[i]);
  }
  std::vector<float> g = rf32(size_t(kC) * QN, -6.f, 6.f, 73);
  for (size_t i = 0; i < g.size(); i += 7) g[i] = 25.0f + float(i % 17);
  for (float& v : g) v = rf(v);
  l0::Mem dq = dv.up(q), dg = dv.up(g), o = dv.zeros(size_t(QH) * kC * kHD * 4), out = dv.zeros(size_t(kC) * QN * 2);
  double worst[2] = {1.0, 1.0};
  int worst_gate = 0;
  for (const Case& cs : cases) {
    std::vector<uint32_t> rows = {0, cs.C - 1, cs.C / 2};
    if (cs.C > 9) rows.insert(rows.end(), {7, 8});
    for (int eager = 0; eager <= 1; ++eager) {
      auto launch = [&](bool gated) {
        dv.go(kk::pf_flash_kv8_variant(QH, KVH, eager, gated), "k2_pf_flash_attn_kv8", (cs.C + 7) / 8, KVH, 1,
              {PtrArg(dq.ptr()), PtrArg(dc.k.ptr()), PtrArg(dc.ks.ptr()), PtrArg(dc.v.ptr()), PtrArg(dc.vs.ptr()),
               PtrArg(dg.ptr()), PtrArg(gated ? out.ptr() : o.ptr()), arg_val(cs.pos), arg_val(cs.C)});
      };
      launch(false);
      const std::vector<float> of = dv.rd<float>(o, size_t(QH) * cs.C * kHD);
      for (uint32_t r : rows)
        for (uint32_t h = 0; h < QH; ++h) {
          double want[kHD];
          kr::attend(qk.data() + (size_t(r) * QH + h) * kHD, c, h / D.gqa(), cs.pos + r + 1, want);
          worst[eager] = std::min(worst[eager], cosine(want, of.data() + (size_t(h) * cs.C + r) * kHD, kHD));
        }
      launch(true);
      const std::vector<uint16_t> ob = dv.rd<uint16_t>(out, size_t(cs.C) * QN);
      for (uint32_t r : rows)
        for (uint32_t h = 0; h < QH; ++h)
          for (uint32_t d = 0; d < kHD; ++d) {
            const size_t idx = (size_t(r) * QH + h) * kHD + d;
            worst_gate = std::max(worst_gate, ulps(ob[idx], k2_ref::attn_gate(of[(size_t(h) * cs.C + r) * kHD + d], g[idx])));
          }
      if (cs.C == 2048 && cs.pos == 2048) {
        launch(true);
        CHECK(dv.rd<uint16_t>(out, size_t(cs.C) * QN) == ob);
      }
    }
    std::printf("   (pos %5u, C %4u): default worst %.7f, eager worst %.7f vs fp64 over the int8 cache\n",
                cs.pos, cs.C, worst[0], worst[1]);
  }
  CHECK(worst[0] >= 0.9999);
  CHECK(worst[1] >= 0.999);
  CHECK(worst_gate <= 1);
  std::printf("6. prefill flash over int8, depths 1 .. 32048: default %.7f (bar 0.9999), eager %.6f (0.999), "
              "un-rotated in the epilogue; gated build within %d bf16 ulp; replay bitwise\n",
              worst[0], worst[1], worst_gate);
}

}  // namespace

int main() {
  Dev dv;
  std::printf("k2_kv8_kernels_test on %s\n", dv.ctx.name().c_str());
  writers(dv);
  decode_flash(dv);
  decode_eager(dv);
  prefill_flash(dv);
  std::printf("k2_kv8_kernels_test OK\n");
  return 0;
}
