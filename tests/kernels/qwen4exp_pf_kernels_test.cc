// Spec 21d (F2 for the prefill): Qwen3.8-Flash-Next's prefill kernels on the card, at the real shapes, against
// tests/kernels/qwen4exp_pf_ref.h / qwen4exp_ref.h and against the dense prefill GEMMs the grouped ones are built
// from. Needs a B70, no checkpoint (synthetic weights). Every binary is one runtime/qwen4exp/qwen4exp_prefill.cc
// binds (kernels::qwen4exp names), plus pf_gemm_T0_SILU for the dense comparison. Written blind on the Mac (box
// queue row 33).
//
//   1. the slabs: pf_dequant_slab at qkv||z 2560 x 16384 (whole slabs), k2_pf_dequant_slab at out / o 6144 x 2560
//      (the 512-column tail), q4_pf_bf16_slab at the HC down||inject 10240 x 336 (one 512-column slab, 176 zero
//      columns) exact; the HC down through pf_gemm_T0 at pitch 512 within cosine 0.99999 of fp64;
//   2. 21c's HC at M = 2048: combine _E / _S1 / _Y / _X / _Y_NN and the up-mix at pitch 512 bitwise q4ref's;
//   3. the indexer over chunks (q4_qsa_prep _PF + q4_qsa_ring: q, the compressed keys, the tail ring == decode's) and
//      score + select at M = 2048 over a sparse chunk - bitwise;
//   4. the PLE over a chunk: q4_ple_gather _PF (the tables in device memory behind the pointer table) ids / rows,
//      and gate / conv / ring == q4pf::ple_chunk bitwise, through chunks 5, 16, 17, 40;
//   5. the router over the chunk (pf_ab_proj rows bitwise decode's gemv_bf16) and decode's q4_route on grid (1, C):
//      ids exact (q4ref::route over the device's logits);
//   6. the MoE block over C = 64 (Review Focus 3): sort / gather exact; the dequantised blocks exact (block 512 the
//      shared expert in both forms); GROUPED == DENSE, bitwise, per (token, slot): gate||up (every batch) against
//      pf_gemm_T0_SILU, down against rne(pf_gemm_T0), for the busiest expert, expert 0 when routed and the shared
//      expert; the combine exactly q4pf's; the chunk REVERSED leaves every (token, slot)'s h and y and every token's
//      y bitwise; replay bitwise; and the sort at C = 2048 on the adversaries (all to 10 experts over 255 / 256 / 511,
//      the 1200-tile bound);
//   7. the attention (Review Focus 1): pf_flash_attn_Q24KV2 against fp64 causal on the dense rows of chunks (0,
//      2048) and (2000, 2048) (its 51 dense rows); q4_pf_sparse_attn against sparse_attn_fp64 (cosine >= 0.99999,
//      spec 6 K1) at chunk starts 2048 (rows 3..), 4096, 30000, 131072 and a 300-row tail; the EAGER build against
//      sparse_row_eager (0.9999: the rounding points, not the sum order - spec 18 §11) and fp64 (0.999); replay
//      bitwise; pf_attn_gate over the sparse O within 1 bf16 ulp of rne(rne(o) x sigmoid(gate)).
// (gdn_chunk_q4 against decode's gdn_step + prep_gated_head_M1_SIG lives in tests/runtime/qwen4exp_prefill_test.cc's
// `gdn` mode: it needs the runtime's gdn_chunk_q4.)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/k2_pf_ref.h"
#include "kernels/kernels.h"
#include "kernels/kolibri_pf_ref.h"
#include "kernels/prefill/pf_kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "kernels/qwen4exp_pf_ref.h"
#include "kernels/qwen4exp_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/qwen4exp_ple_hash.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"

namespace {

namespace kq = kernels::qwen4exp;
namespace qr = q4ref;
namespace qp = q4pf;
using qr::f32;
using qr::rf;
using qr::rne;
using runtime::prefill::arg_val;
using runtime::prefill::KernelArg;
using runtime::prefill::PtrArg;

constexpr uint32_t kC = kq::kPfC;
const uint32_t H = kq::kHidden, HCN = kq::kHcN, HD = kq::kHd, QH = kq::kQHeads, KVH = kq::kKvHeads, QN = QH * HD,
               KVN = KVH * HD, I = kq::kInter;
constexpr uint32_t kListRow = kq::kListRow, kCountW = kq::kCountWord;

static_assert(kq::kPfC == qp::kC && kq::kPfTm == qp::kTm && kq::kPfKt == qp::kKt, "the prefill constants moved");
static_assert(kq::pf_hdr::words() == qp::hdr_words(), "the sort's header width moved");

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
std::vector<float> rbv(size_t n, float lo, float hi, uint64_t seed) {   // fp32 holding bf16 values
  std::vector<float> v = rf32(n, lo, hi, seed);
  for (float& e : v) e = rf(e);
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
int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

struct Dev {
  l0::Context ctx{0};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  runtime::prefill::Context cx{ctx};
  runtime::prefill::KernelCache kc{ctx};
  l0::Mem ctrl{ctx, l0::MemKind::Shared, sizeof(runtime::Control)};
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
  void set_ctl(uint32_t pos, uint32_t n) {
    std::memset(ctrl.ptr(), 0, ctrl.size());
    ctrl.as<runtime::Control>()->pos = pos;
    ctrl.as<runtime::Control>()->n_active = n;
  }
  void go(const std::string& v, const char* entry, uint32_t gx, uint32_t gy, uint32_t gz,
          std::initializer_list<KernelArg> args) {
    cx.launch(kc(v, entry), gx, gy, gz, args);
    cx.wait();
  }
};

std::vector<float> rope_table(uint32_t rows) {   // fp32 [rows][2][32] (the loader's form; any table serves)
  std::vector<float> t(size_t(rows) * 64);
  for (uint32_t p = 0; p < rows; ++p)
    for (uint32_t i = 0; i < 32; ++i) {
      const double a = double(p) * std::pow(1e7, -double(2 * i) / 64.0);
      t[size_t(p) * 64 + i] = float(std::cos(a));
      t[size_t(p) * 64 + 32 + i] = float(std::sin(a));
    }
  return t;
}

// ---- 1. the slabs ------------------------------------------------------------------------------------------------
uint32_t slab_width(uint32_t N, uint32_t n0) { return N - n0 >= kq::kPfSlab ? kq::kPfSlab : (N - n0 + 255) / 256 * 256; }

void slabs(Dev& dv) {
  for (int tail = 0; tail <= 1; ++tail) {   // int4 layout 0: qkv||z whole slabs; out / o with the 512 tail
    const uint32_t K = tail ? kq::kGdnZN : H, N = tail ? H : kq::kQkvzN;
    Rng r(1 + tail);
    std::vector<uint32_t> words(size_t(K / 8) * N);
    for (uint32_t& w : words) w = r.u32();
    std::vector<uint16_t> sc(size_t(K / 64) * N);
    for (uint16_t& s : sc) s = common::f32_to_f16(r.uni(0.01f, 0.05f));
    l0::Mem dw = dv.up(words), ds = dv.up(sc), slab = dv.zeros(size_t(K) * kq::kPfSlab * 2);
    for (uint32_t n0 = 0; n0 < N;) {
      const uint32_t ns = slab_width(N, n0);
      if (tail)
        dv.go(kq::pf_int4_slab_variant(K, N), "k2_pf_dequant_slab", ns / 16, K / 64, 1,
              {PtrArg(dw.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0), arg_val(ns)});
      else
        dv.go(kq::pf_int4_slab_variant(K, N), "pf_dequant_slab", ns / 16, K / 64, 1,
              {PtrArg(dw.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0)});
      if (tail || n0 == 0 || n0 + ns == N)
        CHECK(dv.rd<uint16_t>(slab, size_t(K) * ns) == k2_pf_ref::dequant_slab(words.data(), sc.data(), K, N, n0, ns));
      n0 += ns;
    }
  }
  // the HC down||inject: one 512-wide bf16 slab (336 used), through pf_gemm_T0 at pitch 512
  const uint32_t K = HCN, N = kq::kHcDownN, C = 37, LD = kq::kPfHcDownLd;
  const std::vector<uint16_t> wr = rbf16(size_t(N) * K, -0.02f, 0.02f, 3);
  std::vector<uint16_t> wt(wr.size());
  common::repack_bf16_tiled(wr.data(), K, N, wt.data());
  const std::vector<uint16_t> x = rbf16(size_t(kC) * K, -1.f, 1.f, 4);
  l0::Mem dw = dv.up(wt), dx = dv.up(x), slab = dv.zeros(size_t(K) * LD * 2), part = dv.up(std::vector<float>(size_t(kC) * LD, 7.0f));
  dv.go(kq::pf_bf16_slab_variant(K, N), "kol_pf_bf16_slab", LD / 16, K / 8, 1,
        {PtrArg(dw.ptr()), PtrArg(slab.ptr()), arg_val(0u), arg_val(LD)});
  CHECK(dv.rd<uint16_t>(slab, size_t(K) * LD) == kolibri_pf_ref::bf16_slab(wt.data(), K, N, 0, LD));
  const runtime::prefill::GemmBatch gb{C, K, LD, 1, K, LD, LD, 0, 0, 0};
  runtime::prefill::gemm_l0(dv.cx, dv.kc, gb, dx.as<uint16_t>(), slab.as<uint16_t>(), part.as<float>(), false);
  dv.cx.wait();
  const std::vector<float> got = dv.rd<float>(part, size_t(C) * LD);
  double worst = 1.0;
  for (uint32_t m = 0; m < C; m += 6) {
    std::vector<double> want(N, 0.0);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k) want[n] += double(f32(x[size_t(m) * K + k])) * f32(wr[size_t(n) * K + k]);
    worst = std::min(worst, cosine(want.data(), got.data() + size_t(m) * LD, N));
    for (uint32_t n = N; n < LD; ++n) CHECK_EQ(got[size_t(m) * LD + n], 0.0f);   // the padding columns: exact zeros
  }
  CHECK(worst >= 0.99999);
  std::printf("1. slabs exact (int4 qkv||z 2560 x 16384, int4 out / o 6144 x 2560 = 1024 + 1024 + 512, bf16 HC down "
              "10240 x 336 in one 512 slab); the HC down through pf_gemm_T0 at pitch 512: rows worst cosine %.7f, the "
              "padding exact zeros\n", worst);
}

// ---- 2. the HC at M = 2048 ---------------------------------------------------------------------------------------
void hc(Dev& dv) {
  const uint32_t C = 37;
  const std::vector<uint16_t> H0 = rbf16(size_t(kC) * HCN, -2.f, 2.f, 10);
  const std::vector<float> w = rbv(HCN, 0.5f, 1.5f, 11);
  std::vector<float> inj(size_t(kC) * 4);
  for (size_t i = 0; i < inj.size(); ++i) inj[i] = rf(0.5f + 0.125f * float(i % 13));
  const std::vector<float> slices = rf32(size_t(kC) * H, -0.5f, 0.5f, 12);
  const std::vector<uint16_t> y = rbf16(size_t(kC) * H, -1.f, 1.f, 13);
  l0::Mem dw = dv.up(w), di = dv.up(inj), ds = dv.up(slices), dy = dv.up(y);
  dv.set_ctl(4000, C);
  struct V { kq::HcSrc src; bool norm; };
  for (const V v : {V{kq::HcSrc::Embed, true}, V{kq::HcSrc::Slices, true}, V{kq::HcSrc::Y, true}, V{kq::HcSrc::None, true},
                    V{kq::HcSrc::Y, false}}) {
    l0::Mem dh = dv.up(H0), dxn = dv.zeros(size_t(kC) * HCN * 2);
    const void* src = v.src == kq::HcSrc::Slices ? ds.ptr() : dy.ptr();
    dv.go(kq::pf_hc_combine_norm_variant(v.src, v.norm), "q4_hc_combine_norm", 4, C, 1,
          {PtrArg(dv.ctrl.ptr()), PtrArg(dh.ptr()), PtrArg(src), PtrArg(di.ptr()), PtrArg(dw.ptr()), PtrArg(dxn.ptr())});
    const std::vector<uint16_t> Hd = dv.rd<uint16_t>(dh, size_t(C) * HCN), xd = dv.rd<uint16_t>(dxn, size_t(C) * HCN);
    for (uint32_t m = 0; m < C; ++m) {
      std::vector<uint16_t> Hh(H0.begin() + size_t(m) * HCN, H0.begin() + size_t(m + 1) * HCN), xh(HCN, 0), ym(H);
      for (uint32_t k = 0; k < H; ++k) ym[k] = v.src == kq::HcSrc::Slices ? rne(slices[size_t(m) * H + k]) : y[size_t(m) * H + k];
      const qr::HcSrc hs = v.src == kq::HcSrc::Embed ? qr::HcSrc::Embed : v.src == kq::HcSrc::None ? qr::HcSrc::None : qr::HcSrc::Y;
      qr::hc_combine_norm(Hh.data(), hs, ym.data(), inj.data() + size_t(m) * 4, w.data(), v.norm ? xh.data() : nullptr);
      CHECK(std::equal(Hh.begin(), Hh.end(), Hd.begin() + size_t(m) * HCN));
      if (v.norm) CHECK(std::equal(xh.begin(), xh.end(), xd.begin() + size_t(m) * HCN));
    }
  }
  const std::vector<float> down = rf32(size_t(kC) * kq::kPfHcDownLd, -6.f, 6.f, 14);
  const std::vector<uint16_t> upw = rbf16(size_t(kq::kHcLow) * HCN, -0.15f, 0.15f, 15), xn = rbf16(size_t(kC) * HCN, -2.f, 2.f, 16);
  l0::Mem dd = dv.up(down), du = dv.up(upw), dxn = dv.up(xn), dx = dv.zeros(size_t(kC) * H * 2), dinj = dv.zeros(size_t(kC) * 16);
  dv.go(kq::pf_hc_up_mix_variant(), "q4_hc_up_mix", H / 16, C, 1,
        {PtrArg(dv.ctrl.ptr()), PtrArg(dd.ptr()), PtrArg(du.ptr()), PtrArg(dxn.ptr()), PtrArg(dx.ptr()), PtrArg(dinj.ptr())});
  const std::vector<uint16_t> xd = dv.rd<uint16_t>(dx, size_t(C) * H);
  const std::vector<float> id = dv.rd<float>(dinj, size_t(C) * 4);
  for (uint32_t m = 0; m < C; ++m) {
    std::vector<uint16_t> xh(H);
    float ih[4];
    qr::hc_up_mix(down.data() + size_t(m) * kq::kPfHcDownLd, true, upw.data(), xn.data() + size_t(m) * HCN, xh.data(), ih);
    CHECK(std::equal(xh.begin(), xh.end(), xd.begin() + size_t(m) * H));
    for (uint32_t s = 0; s < 4; ++s) CHECK_EQ(qr::as_u32(id[m * 4 + s]), qr::as_u32(ih[s]));
  }
  std::printf("2. the HC at M = %u over %u rows: combine _E / _S1 / _Y / _X / _Y_NN (H, xn) and the up-mix at pitch 512 "
              "(x, inj) bitwise q4ref's\n", kC, C);
}

// ---- 3. the indexer over chunks ------------------------------------------------------------------------------------
void indexer(Dev& dv) {
  const uint32_t LD = kq::kPfIdxLd, ML = 8192, stride = ML / 4;
  const std::vector<float> rope = rope_table(ML), small = rbv(768, 0.5f, 1.5f, 20);
  l0::Mem drope = dv.up(rope), dsmall = dv.up(small), tail = dv.zeros(8 * 128 * 2), keys = dv.zeros(size_t(stride) * 128 * 2),
          idx = dv.zeros(size_t(kC) * LD * 4), q = dv.zeros(size_t(kC) * 512 * 4);
  std::vector<uint16_t> tail_h(8 * 128, 0), keys_h(size_t(stride) * 128, 0);
  std::vector<float> q_last;
  uint32_t pos = 0;
  for (uint32_t C : {9u, 2048u, 1u, 2u, 300u, 2048u}) {   // ... 2060, 2360, 4408
    const std::vector<float> rows = rf32(size_t(C) * LD, -3.f, 3.f, 30 + pos);
    dv.imm.copy(idx.ptr(), rows.data(), rows.size() * 4);
    dv.set_ctl(pos, C);
    dv.go(kq::pf_qsa_variant(), "q4_qsa_prep", 5, C, 1,
          {PtrArg(dv.ctrl.ptr()), PtrArg(idx.ptr()), PtrArg(dsmall.ptr()), PtrArg(drope.ptr()), PtrArg(q.ptr()),
           PtrArg(tail.ptr()), PtrArg(keys.ptr())});
    dv.go(kq::pf_qsa_variant(), "q4_qsa_ring", 1, 8, 1, {PtrArg(dv.ctrl.ptr()), PtrArg(idx.ptr()), PtrArg(tail.ptr())});
    std::vector<float> qh(size_t(C) * 512);
    qp::qsa_prep_chunk(rows.data(), LD, pos, C, small.data() + 512, small.data() + 640, rope.data(), tail_h.data(), keys_h.data(),
                       qh.data());
    const std::vector<float> qd = dv.rd<float>(q, size_t(C) * 512);
    for (size_t i = 0; i < qh.size(); ++i) CHECK_EQ(qr::as_u32(qd[i]), qr::as_u32(qh[i]));
    CHECK(dv.rd<uint16_t>(tail, 8 * 128) == tail_h);
    CHECK(dv.rd<uint16_t>(keys, size_t(stride) * 128) == keys_h);
    q_last = qd;
    pos += C;
  }
  // score + select over the last chunk (positions 2360..4407: every row sparse) at M = 2048
  const uint32_t C = 2048, p0 = pos - C;
  l0::Mem sc = dv.zeros(size_t(C) * stride * 4), lists = dv.zeros(size_t(C) * kListRow * 4), diag = dv.zeros(size_t(C) * 8);
  dv.set_ctl(p0, C);
  dv.go(kq::pf_qsa_variant(), "q4_qsa_score", (stride + 255) / 256, C, 1,
        {PtrArg(dv.ctrl.ptr()), PtrArg(q.ptr()), PtrArg(keys.ptr()), PtrArg(sc.ptr()), arg_val(stride)});
  dv.go(kq::pf_qsa_variant(), "q4_qsa_select", 1, C, 1,
        {PtrArg(dv.ctrl.ptr()), PtrArg(sc.ptr()), arg_val(stride), PtrArg(lists.ptr()), PtrArg(diag.ptr())});
  const std::vector<float> sd = dv.rd<float>(sc, size_t(C) * stride);
  const std::vector<uint32_t> ld = dv.rd<uint32_t>(lists, size_t(C) * kListRow);
  for (uint32_t m = 0; m < C; m += 7) {
    const uint32_t p = p0 + m, n = (p + 1) / 4;
    for (uint32_t b = 0; b < n; ++b)
      CHECK_EQ(qr::as_u32(sd[size_t(m) * stride + b]), qr::as_u32(qr::qsa_score(q_last.data() + size_t(m) * 512, keys_h.data() + size_t(b) * 128)));
    const qr::Selection sel = qr::qsa_select(sd.data() + size_t(m) * stride, p);
    CHECK_EQ(ld[size_t(m) * kListRow + kCountW], sel.count());
    CHECK(std::equal(sel.list.begin(), sel.list.end(), ld.begin() + size_t(m) * kListRow));
  }
  std::printf("3. the indexer over chunks 9, 2048, 1, 2, 300, 2048 (q, keys, the tail ring == decode's) and score + "
              "select at M = %u over 2360..4407: bitwise\n", kC);
}

// ---- 4. the PLE over a chunk -----------------------------------------------------------------------------------------
void ple(Dev& dv) {
  const uint32_t KVN_ = HCN + H;
  // the gather: 16 heads of a reduced-prime table (base 1000) in device memory behind the pointer table
  const std::vector<uint64_t> sizes = loader::q4_ple_primes(1000, 16, 0), offs = loader::q4_ple_offsets(sizes);
  const std::array<uint64_t, 3> mult = loader::q4_ple_multipliers(kq::kVocab, 3, 0, 1234);
  std::vector<std::unique_ptr<l0::Mem>> qm, sm;
  std::vector<uint64_t> ptrs(32);
  std::vector<std::vector<int8_t>> qh(16);
  std::vector<std::vector<uint16_t>> sh(16);
  for (uint32_t h = 0; h < 16; ++h) {
    Rng r(40 + h);
    qh[h].resize(size_t(sizes[h]) * 160);
    for (int8_t& v : qh[h]) v = int8_t(int(r.u32() % 255) - 127);
    sh[h].resize(sizes[h]);
    for (uint16_t& v : sh[h]) v = rne(r.uni(0.001f, 0.05f));
    qm.push_back(std::make_unique<l0::Mem>(dv.up(qh[h])));
    sm.push_back(std::make_unique<l0::Mem>(dv.up(sh[h])));
    ptrs[h] = reinterpret_cast<uint64_t>(qm.back()->ptr());
    ptrs[16 + h] = reinterpret_cast<uint64_t>(sm.back()->ptr());
  }
  std::vector<uint64_t> consts(35);
  for (uint32_t i = 0; i < 3; ++i) consts[i] = mult[i];
  for (uint32_t h = 0; h < 16; ++h) {
    consts[3 + h] = sizes[h];
    consts[19 + h] = offs[h];
  }
  l0::Mem dptrs = dv.up(ptrs), dconsts = dv.up(consts), e = dv.zeros(size_t(kC) * H * 2), ids_out = dv.zeros(size_t(kC) * 16 * 8),
          idring = dv.zeros(16 * 4), ring = dv.zeros(size_t(16) * HCN * 2), gated = dv.zeros(size_t(kC) * HCN * 2),
          gn = dv.zeros(size_t(kC) * HCN * 2), ids_buf = dv.zeros(size_t(kC) * 4);
  std::vector<float> pw = rbv(size_t(3) * HCN, 0.5f, 1.5f, 50);
  const std::vector<float> taps = rbv(size_t(HCN) * 4, -0.5f, 0.5f, 51);
  pw.insert(pw.end(), taps.begin(), taps.end());
  l0::Mem dpw = dv.up(pw);
  std::vector<uint32_t> seq(78);
  {
    Rng r(52);
    for (uint32_t& t : seq) t = r.u32() % kq::kVocab;
    seq[4] = seq[21] = seq[22] = kq::kPleEos;
  }
  std::vector<uint16_t> ring_h(size_t(16) * HCN, 0);
  std::vector<uint32_t> idr_h(16, 0);
  uint32_t pos = 0;
  for (uint32_t C : {5u, 16u, 17u, 40u}) {
    const std::vector<float> kv = rf32(size_t(C) * KVN_, -2.f, 2.f, 60 + pos);
    const std::vector<uint16_t> H0 = rbf16(size_t(C) * HCN, -2.f, 2.f, 70 + pos);
    l0::Mem dkv = dv.up(kv), dh = dv.up(H0);
    dv.imm.copy(ids_buf.ptr(), seq.data() + pos, size_t(C) * 4);
    dv.set_ctl(pos, C);
    dv.go(kq::pf_ple_gather_variant(true), "q4_ple_gather", 16, C, 1,
          {PtrArg(dv.ctrl.ptr()), PtrArg(idring.ptr()), PtrArg(dptrs.ptr()), PtrArg(dconsts.ptr()), PtrArg(e.ptr()),
           PtrArg(ids_out.ptr()), PtrArg(ids_buf.ptr())});
    const std::vector<loader::Q4PleHistory> hs = qp::ple_ids_chunk(seq.data() + pos, pos, C, idr_h.data());
    const std::vector<uint64_t> got = dv.rd<uint64_t>(ids_out, size_t(C) * 16);
    const std::vector<uint16_t> ed = dv.rd<uint16_t>(e, size_t(C) * H);
    for (uint32_t m = 0; m < C; ++m) {
      const std::array<uint64_t, 16> want = loader::q4_ple_ids(hs[m].t0, hs[m].t1, hs[m].t2, mult, sizes, offs);
      for (uint32_t h = 0; h < 16; ++h) {
        CHECK_EQ(got[size_t(m) * 16 + h], want[h]);
        const uint64_t r = want[h] - offs[h];
        uint16_t row[160];
        qr::ple_row(qh[h].data() + r * 160, f32(sh[h][r]), row);
        CHECK(std::equal(row, row + 160, ed.begin() + size_t(m) * H + h * 160));
      }
    }
    dv.go(kq::pf_ple_variant(), "q4_pf_ple_gate", 4, C, 1,
          {PtrArg(dv.ctrl.ptr()), PtrArg(dkv.ptr()), PtrArg(dh.ptr()), PtrArg(dpw.ptr()), PtrArg(gated.ptr()), PtrArg(gn.ptr())});
    dv.go(kq::pf_ple_variant(), "q4_pf_ple_conv", HCN / 256, C, 1,
          {PtrArg(dv.ctrl.ptr()), PtrArg(gated.ptr()), PtrArg(gn.ptr()), PtrArg(ring.ptr()), PtrArg(dpw.ptr()), PtrArg(dh.ptr())});
    dv.go(kq::pf_ple_variant(), "q4_pf_ple_ring", HCN / 256, 16, 1,
          {PtrArg(dv.ctrl.ptr()), PtrArg(gn.ptr()), PtrArg(ids_buf.ptr()), PtrArg(ring.ptr()), PtrArg(idring.ptr())});
    std::vector<uint16_t> Hh = H0;
    qp::ple_chunk(Hh.data(), kv.data(), seq.data() + pos, pos, C, pw.data(), pw.data() + HCN, pw.data() + 2 * HCN, taps.data(),
                  ring_h.data(), idr_h.data());
    CHECK(dv.rd<uint16_t>(dh, size_t(C) * HCN) == Hh);
    CHECK(dv.rd<uint16_t>(ring, size_t(16) * HCN) == ring_h);
    CHECK(dv.rd<uint32_t>(idring, 16) == idr_h);
    pos += C;
  }
  std::printf("4. the PLE over chunks 5, 16, 17, 40: the gather's ids and rows (the pointer table), gate / conv / ring "
              "== q4pf::ple_chunk - H and both rings bitwise\n");
}

// ---- 5. the router -------------------------------------------------------------------------------------------------
std::vector<uint32_t> routers(Dev& dv, uint32_t C, const std::vector<uint16_t>& x) {
  const uint32_t RN = kq::kRouterN;
  std::vector<uint16_t> rows = rbf16(size_t(RN) * H, -0.05f, 0.05f, 80);
  std::fill(rows.begin() + size_t(kq::kExperts + 1) * H, rows.end(), uint16_t(0));
  std::vector<uint16_t> tiled(rows.size());
  common::repack_bf16_tiled(rows.data(), H, RN, tiled.data());
  l0::Mem dwr = dv.up(tiled), dx = dv.up(x), lg = dv.zeros(size_t(kC) * RN * 4), rt = dv.zeros(size_t(kC) * 32 * 4),
          one = dv.zeros(size_t(RN) * 4);
  dv.go(kq::pf_router_variant(), "pf_ab_proj", RN / 16, (C + 7) / 8, 1,
        {PtrArg(dwr.ptr()), PtrArg(dx.ptr()), PtrArg(lg.ptr()), arg_val(C)});
  const std::vector<float> logits = dv.rd<float>(lg, size_t(C) * RN);
  for (uint32_t m : {0u, C / 2, C - 1}) {
    l0::Mem xr = dv.up(std::vector<uint16_t>(x.begin() + size_t(m) * H, x.begin() + size_t(m + 1) * H));
    dv.go(kq::router_variant(), "gemv_bf16", RN / kernels::gemv_bf16_tiling(RN).cols, 1, 1,
          {PtrArg(dwr.ptr()), PtrArg(xr.ptr()), PtrArg(one.ptr())});
    const std::vector<float> dec = dv.rd<float>(one, RN);
    CHECK(std::equal(dec.begin(), dec.end(), logits.begin() + size_t(m) * RN));
  }
  dv.go(kq::route_variant(1), "q4_route", 1, C, 1, {PtrArg(lg.ptr()), PtrArg(rt.ptr())});
  const std::vector<uint32_t> out = dv.rd<uint32_t>(rt, size_t(C) * 32);
  for (uint32_t m = 0; m < C; ++m) {
    const std::vector<uint32_t> want = qr::route_row(qr::route(logits.data() + size_t(m) * RN));
    for (uint32_t j = 0; j < kq::kTopK; ++j) CHECK_EQ(out[size_t(m) * 32 + j], want[j]);
  }
  std::printf("5. router over %u rows: pf_ab_proj rows bitwise decode's gemv_bf16, decode's q4_route on grid (1, C): ids "
              "exact\n", C);
  return out;
}

// ---- 6. the MoE block ----------------------------------------------------------------------------------------------
struct Walked {
  std::vector<uint32_t> hdr, tiles, row_tok, pair_row;
  std::vector<uint16_t> xg, h, y, out;
};
struct MoeBufs {
  l0::Mem x, rt, hdr, tiles, row_tok, pair_row, xg, h, w, out, gu, dn, sgu, sdn;
};

Walked moe_walk(Dev& dv, MoeBufs& b, uint32_t C, bool shb) {
  const uint32_t tmax = qp::tmax(C), B = kq::kExperts + 1;
  const uint32_t per_gu = uint32_t(kq::kPfBatchBytes / (size_t(H) * 2 * I * 2)), per_dn = uint32_t(kq::kPfBatchBytes / (size_t(I) * H * 2));
  const std::string v = kq::pf_moe_variant();
  dv.go(v, "q4_pf_sort", 1, 1, 1,
        {PtrArg(b.rt.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.tiles.ptr()), PtrArg(b.row_tok.ptr()), PtrArg(b.pair_row.ptr()),
         arg_val(C), arg_val(tmax)});
  dv.go(v, "q4_pf_gather", tmax * kq::kPfTm, 1, 1,
        {PtrArg(b.x.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.row_tok.ptr()), PtrArg(b.xg.ptr())});
  Walked w;
  w.hdr = dv.rd<uint32_t>(b.hdr, qp::hdr_words());
  w.tiles = dv.rd<uint32_t>(b.tiles, size_t(tmax) * 2);
  w.row_tok = dv.rd<uint32_t>(b.row_tok, size_t(tmax) * kq::kPfTm);
  w.pair_row = dv.rd<uint32_t>(b.pair_row, size_t(C) * kq::kTopK);
  w.xg = dv.rd<uint16_t>(b.xg, size_t(w.hdr[qp::kHdrRows]) * H);
  for (uint32_t b0 = 0; b0 < B; b0 += per_gu) {
    const uint32_t b1 = std::min(B, b0 + per_gu);
    dv.go(v, shb ? "q4_pf_dequant_gu_shb" : "q4_pf_dequant_gu", 2 * I / 16, H / 64, b1 - b0,
          {PtrArg(b.gu.ptr()), PtrArg(b.sgu.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.w.ptr()), arg_val(b0), arg_val(b1)});
    dv.go(kq::pf_gemm_gu_variant(), "pf_moe_gemm", tmax, 2 * I / 256, 1,
          {PtrArg(b.tiles.ptr()), PtrArg(b.xg.ptr()), PtrArg(nullptr), PtrArg(b.w.ptr()), PtrArg(nullptr), PtrArg(b.h.ptr()),
           arg_val(b0), arg_val(b1), arg_val(0u)});
  }
  w.h = dv.rd<uint16_t>(b.h, size_t(tmax) * kq::kPfTm * I);
  for (uint32_t b0 = 0; b0 < B; b0 += per_dn) {
    const uint32_t b1 = std::min(B, b0 + per_dn);
    dv.go(v, shb ? "q4_pf_dequant_dn_shb" : "q4_pf_dequant_dn", H / 16, I / 64, b1 - b0,
          {PtrArg(b.dn.ptr()), PtrArg(b.sdn.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.w.ptr()), arg_val(b0), arg_val(b1)});
    dv.go(kq::pf_gemm_dn_variant(), "pf_moe_gemm", tmax, H / 256, 1,
          {PtrArg(b.tiles.ptr()), PtrArg(b.h.ptr()), PtrArg(nullptr), PtrArg(b.w.ptr()), PtrArg(nullptr), PtrArg(b.xg.ptr()),
           arg_val(b0), arg_val(b1), arg_val(0u)});
  }
  w.y = dv.rd<uint16_t>(b.xg, size_t(tmax) * kq::kPfTm * H);
  dv.go(v, "q4_pf_moe_combine", H / 256, C, 1,
        {PtrArg(b.rt.ptr()), PtrArg(b.hdr.ptr()), PtrArg(b.pair_row.ptr()), PtrArg(b.xg.ptr()), PtrArg(b.out.ptr()), arg_val(C)});
  w.out = dv.rd<uint16_t>(b.out, size_t(C) * H);
  return w;
}

void moe_block(Dev& dv, const std::vector<uint16_t>& x, const std::vector<uint32_t>& routes, uint32_t C) {
  const std::vector<uint32_t> gu = rblocks(qr::gate_up_block_words() * kq::kExperts, 90),
                              dn = rblocks(qr::down_block_words() * kq::kExperts, 91);
  const std::vector<uint32_t> sgu4 = rblocks(qr::gate_up_block_words(), 92), sdn4 = rblocks(qr::down_block_words(), 93);
  const std::vector<uint16_t> sgr = rbf16(size_t(2 * I) * H, -0.03f, 0.03f, 94), sdr = rbf16(size_t(H) * I, -0.05f, 0.05f, 95);
  std::vector<uint16_t> sgub(sgr.size()), sdnb(sdr.size());
  common::repack_bf16_tiled(sgr.data(), H, 2 * I, sgub.data());
  common::repack_bf16_tiled(sdr.data(), I, H, sdnb.data());
  const uint32_t T = qp::tmax(kC), R = T * kq::kPfTm;
  for (int shb = 0; shb <= 1; ++shb) {
    MoeBufs b{dv.up(x), dv.up(routes), dv.zeros(qp::hdr_words() * 4), dv.zeros(size_t(T) * 8), dv.zeros(size_t(R) * 4),
              dv.zeros(size_t(kC) * kq::kTopK * 4), dv.zeros(size_t(R) * H * 2), dv.zeros(size_t(R) * I * 2),
              dv.zeros(kq::kPfBatchBytes), dv.zeros(size_t(kC) * H * 2), dv.up(gu), dv.up(dn),
              shb ? dv.up(sgub) : dv.up(sgu4), shb ? dv.up(sdnb) : dv.up(sdn4)};
    const uint32_t* s4g = shb ? nullptr : sgu4.data();
    const uint32_t* s4d = shb ? nullptr : sdn4.data();
    const uint16_t* sbg = shb ? sgub.data() : nullptr;
    const uint16_t* sbd = shb ? sdnb.data() : nullptr;
    const Walked w = moe_walk(dv, b, C, shb);
    const qp::Sorted st = qp::sort(routes.data(), C);
    CHECK(w.hdr == st.hdr);
    CHECK(std::equal(st.tiles.begin(), st.tiles.end(), w.tiles.begin()));
    CHECK(std::equal(st.row_tok.begin(), st.row_tok.begin() + st.rows_used(), w.row_tok.begin()));
    CHECK(w.pair_row == st.pair_row);
    CHECK(w.xg == qp::gather(st, x.data()));
    std::vector<uint32_t> load(kq::kExperts, 0);
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < kq::kTopK; ++j) ++load[routes[size_t(t) * 32 + j]];
    const uint32_t busiest = uint32_t(std::max_element(load.begin(), load.end()) - load.begin());
    l0::Mem dx = dv.up(x), hd = dv.up(w.h);
    for (uint32_t e : {busiest, 0u, qp::kShared}) {
      if (e < kq::kExperts && load[e] == 0) continue;
      const std::vector<uint16_t> bgu = qp::dequant_gu(gu.data(), s4g, sbg, e);
      l0::Mem db = dv.up(bgu), xs = dv.zeros(size_t(kC) * I * 2);
      const runtime::prefill::GemmBatch gb{C, H, 2 * I, 1, H, 2 * I, 0, 0, 0, 0};
      runtime::prefill::gemm_l0_silu(dv.cx, dv.kc, gb, dx.as<uint16_t>(), db.as<uint16_t>(), xs.as<uint16_t>(), I);
      dv.cx.wait();
      const std::vector<uint16_t> dense = dv.rd<uint16_t>(xs, size_t(C) * I);
      for (uint32_t t = 0; t < C; ++t)
        for (uint32_t j = 0; j <= kq::kTopK; ++j) {
          const bool hit = j < kq::kTopK ? routes[size_t(t) * 32 + j] == e : e == qp::kShared;
          if (!hit) continue;
          const uint32_t r = j < kq::kTopK ? st.pair_row[size_t(t) * kq::kTopK + j] : st.shared_row() + t;
          CHECK(std::equal(dense.begin() + size_t(t) * I, dense.begin() + size_t(t + 1) * I, w.h.begin() + size_t(r) * I));
        }
      const std::vector<uint16_t> bdn = qp::dequant_dn(dn.data(), s4d, sbd, e);
      l0::Mem ddn = dv.up(bdn), yd = dv.zeros(size_t(R) * H * 4);
      const uint32_t rows = st.rows_used();
      const runtime::prefill::GemmBatch gd{rows, I, H, 1, I, H, H, 0, 0, 0};
      runtime::prefill::gemm_l0(dv.cx, dv.kc, gd, hd.as<uint16_t>(), ddn.as<uint16_t>(), yd.as<float>(), false);
      dv.cx.wait();
      const std::vector<float> yf = dv.rd<float>(yd, size_t(rows) * H);
      for (uint32_t tile = 0; tile < st.tiles_used; ++tile) {
        if (st.tiles[2 * tile] != e) continue;
        for (uint32_t i = 0; i < kq::kPfTm; ++i) {
          const uint32_t r = st.tiles[2 * tile + 1] + i;
          if (st.row_tok[r] == qp::kNone) continue;
          for (uint32_t n = 0; n < H; ++n) CHECK_EQ(w.y[size_t(r) * H + n], rne(yf[size_t(r) * H + n]));
        }
      }
    }
    std::vector<uint16_t> out_ref(size_t(C) * H);
    qp::combine(routes.data(), st, w.y.data(), out_ref.data(), C);
    CHECK(w.out == out_ref);
    // the chunk reversed
    std::vector<uint16_t> xr(x.size(), 0);
    std::vector<uint32_t> rtr(routes.size(), 0);
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t u = C - 1 - t;
      std::copy(x.begin() + size_t(t) * H, x.begin() + size_t(t + 1) * H, xr.begin() + size_t(u) * H);
      std::copy(routes.begin() + size_t(t) * 32, routes.begin() + size_t(t + 1) * 32, rtr.begin() + size_t(u) * 32);
    }
    dv.imm.copy(b.x.ptr(), xr.data(), xr.size() * 2);
    dv.imm.copy(b.rt.ptr(), rtr.data(), rtr.size() * 4);
    const Walked v = moe_walk(dv, b, C, shb);
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t u = C - 1 - t;
      CHECK(std::equal(w.out.begin() + size_t(t) * H, w.out.begin() + size_t(t + 1) * H, v.out.begin() + size_t(u) * H));
      for (uint32_t j = 0; j < kq::kTopK; ++j) {
        const uint32_t ra = w.pair_row[size_t(t) * kq::kTopK + j], rb = v.pair_row[size_t(u) * kq::kTopK + j];
        CHECK(std::equal(w.h.begin() + size_t(ra) * I, w.h.begin() + size_t(ra + 1) * I, v.h.begin() + size_t(rb) * I));
        CHECK(std::equal(w.y.begin() + size_t(ra) * H, w.y.begin() + size_t(ra + 1) * H, v.y.begin() + size_t(rb) * H));
      }
    }
    // replay: the forward chunk again, bitwise
    dv.imm.copy(b.x.ptr(), x.data(), x.size() * 2);
    dv.imm.copy(b.rt.ptr(), routes.data(), routes.size() * 4);
    const Walked again = moe_walk(dv, b, C, shb);
    CHECK(again.out == w.out && again.h == w.h && again.y == w.y);
    std::printf("6. MoE block over %u rows (%s shared expert): sort / gather exact; grouped == dense bitwise (experts %u, 0, "
                "the shared block 512); combine exact; reversed chunk and replay bitwise\n",
                C, shb ? "Intel's bf16" : "ours int4", busiest);
  }
}

// The sort at C = 2048 on the adversaries: all to 10 experts (255 / 256 / 511), the 1200-tile bound.
void sort_adversaries(Dev& dv) {
  const uint32_t T = qp::tmax(kC), R = T * kq::kPfTm;
  for (int kind = 0; kind < 2; ++kind) {
    std::vector<uint32_t> list;
    if (kind == 0)
      for (uint32_t e : {511u, 0u, 256u, 255u, 1u, 127u, 254u, 257u, 383u, 510u}) list.insert(list.end(), kC, e);
    else
      for (uint32_t e = 0; e < kq::kExperts; ++e) list.insert(list.end(), e < 112 ? 65u : 33u, e);
    std::vector<uint32_t> rows(size_t(kC) * 32, 0);
    for (uint32_t t = 0; t < kC; ++t)
      for (uint32_t j = 0; j < kq::kTopK; ++j) rows[size_t(t) * 32 + j] = list[size_t(j) * kC + t];
    l0::Mem rt = dv.up(rows), hdr = dv.zeros(qp::hdr_words() * 4), tiles = dv.zeros(size_t(T) * 8), row_tok = dv.zeros(size_t(R) * 4),
            pair_row = dv.zeros(size_t(kC) * kq::kTopK * 4);
    dv.go(kq::pf_moe_variant(), "q4_pf_sort", 1, 1, 1,
          {PtrArg(rt.ptr()), PtrArg(hdr.ptr()), PtrArg(tiles.ptr()), PtrArg(row_tok.ptr()), PtrArg(pair_row.ptr()), arg_val(kC),
           arg_val(T)});
    const qp::Sorted st = qp::sort(rows.data(), kC);
    CHECK(dv.rd<uint32_t>(hdr, qp::hdr_words()) == st.hdr);
    CHECK(dv.rd<uint32_t>(tiles, size_t(T) * 2) == st.tiles);
    const std::vector<uint32_t> rtok = dv.rd<uint32_t>(row_tok, size_t(R));
    CHECK(std::equal(st.row_tok.begin(), st.row_tok.begin() + st.rows_used(), rtok.begin()));
    CHECK(dv.rd<uint32_t>(pair_row, size_t(kC) * kq::kTopK) == st.pair_row);
    if (kind == 1) CHECK_EQ(st.tiles_used, T);
  }
  std::printf("   sort at C %u: all to 10 experts (lanes 255 / 256 / 511) and the tile bound reached (%u tiles) exact\n", kC, T);
}

// ---- 7. the attention ----------------------------------------------------------------------------------------------
void attention(Dev& dv) {
  const uint32_t maxd = 131072 + 2048;
  const std::vector<uint16_t> kf = rbf16(size_t(maxd) * KVN, -1.f, 1.f, 100), vf = rbf16(size_t(maxd) * KVN, -1.f, 1.f, 101);
  l0::Mem dk = dv.up(kf), dvv = dv.up(vf);
  std::vector<float> qf = rbv(size_t(kC) * QN, -1.f, 1.f, 102);
  std::vector<uint16_t> qb(qf.size());
  for (size_t i = 0; i < qf.size(); ++i) qb[i] = rne(qf[i]);
  const std::vector<float> gate = rf32(size_t(kC) * kq::kQkvgN, -4.f, 4.f, 103);   // the qkv partials' gate columns
  l0::Mem dq = dv.up(qb), O = dv.zeros(size_t(QH) * kC * HD * 4), O2 = dv.zeros(size_t(QH) * kC * HD * 4),
          lists = dv.zeros(size_t(kC) * kListRow * 4), dgate = dv.up(gate), out = dv.zeros(size_t(kC) * QN * 2);
  double worst_dense = 1.0, worst_sparse = 1.0, worst_eager = 1.0, worst_eager64 = 1.0;
  int gate_ulps = 0;
  // dense rows: chunks (0, 2048) and (2000, 2048) (its 51 dense rows)
  for (const uint32_t pos : {0u, 2000u}) {
    const uint32_t nd = qp::dense_rows(pos, kC);
    dv.go(kq::pf_flash_q4_variant(), "pf_flash_attn", (nd + kq::kPfFlashRpw - 1) / kq::kPfFlashRpw, KVH,
          QH / KVH / kq::kPfFlashHpw, {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(O.ptr()), arg_val(pos),
                                       arg_val(nd), arg_val(kC)});
    const std::vector<float> o = dv.rd<float>(O, size_t(QH) * kC * HD);
    for (uint32_t r : {0u, 1u, 7u, 23u, nd / 2, nd - 1}) {
      if (r >= nd) continue;
      std::vector<uint32_t> ident(pos + r + 1);
      for (uint32_t i = 0; i <= pos + r; ++i) ident[i] = i;
      for (uint32_t h = 0; h < QH; h += 5) {
        const std::vector<double> want = qr::qsa_attn_fp64(qf.data() + size_t(r) * QN, kf.data(), vf.data(), ident.data(),
                                                           pos + r + 1, h);
        worst_dense = std::min(worst_dense, cosine(want.data(), o.data() + (size_t(h) * kC + r) * HD, HD));
      }
    }
  }
  // sparse rows: chunk starts 2048 (rows 3..), 4096, 30000, 131072 and a 300-row tail
  struct Case { uint32_t pos, C; };
  for (const Case c : {Case{2048, 64}, Case{4096, 64}, Case{30000, 64}, Case{131072, 64}, Case{2048 + 2048, 300}}) {
    const uint32_t r0 = qp::dense_rows(c.pos, c.C);
    std::vector<uint32_t> lh(size_t(c.C) * kListRow, 0), counts(c.C, 0);
    Rng rr(110 + c.pos);
    for (uint32_t m = r0; m < c.C; ++m) {
      const uint32_t p = c.pos + m, n = (p + 1) / 4;
      std::vector<float> s(n);
      for (float& v : s) v = rf(rr.uni(0.f, 4.f));
      const qr::Selection sel = qr::qsa_select(s.data(), p);
      std::copy(sel.list.begin(), sel.list.end(), lh.begin() + size_t(m) * kListRow);
      lh[size_t(m) * kListRow + kCountW] = counts[m] = sel.count();
    }
    dv.imm.copy(lists.ptr(), lh.data(), lh.size() * 4);
    std::vector<double> o64(size_t(QH) * c.C * HD, 0.0);
    qp::sparse_attn_fp64(qf.data(), kf.data(), vf.data(), lh.data(), counts.data(), r0, c.C, o64.data());
    for (int eager = 0; eager <= 1; ++eager) {
      dv.go(kq::pf_sparse_attn_variant(eager), "q4_pf_sparse_attn", c.C - r0, KVH, 1,
            {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(lists.ptr()), PtrArg(O.ptr()), arg_val(r0),
             arg_val(kC)});
      const std::vector<float> o = dv.rd<float>(O, size_t(QH) * kC * HD);
      for (uint32_t m = r0; m < c.C; m += (c.C > 64 ? 13 : 3))
        for (uint32_t h = 0; h < QH; ++h) {
          std::vector<double> want(HD);
          for (uint32_t d = 0; d < HD; ++d) want[d] = o64[(size_t(h) * c.C + m) * HD + d];
          const double cs = cosine(want.data(), o.data() + (size_t(h) * kC + m) * HD, HD);
          if (!eager) {
            worst_sparse = std::min(worst_sparse, cs);
          } else {
            worst_eager64 = std::min(worst_eager64, cs);
            const std::vector<uint16_t> e = qp::sparse_row_eager(qf.data() + size_t(m) * QN, kf.data(), vf.data(),
                                                                  lh.data() + size_t(m) * kListRow, counts[m], h);
            std::vector<double> ed(HD);
            for (uint32_t d = 0; d < HD; ++d) ed[d] = f32(e[d]);
            worst_eager = std::min(worst_eager, cosine(ed.data(), o.data() + (size_t(h) * kC + m) * HD, HD));
          }
        }
      if (!eager && c.pos == 4096) {   // replay bitwise; the gate over the sparse O
        dv.go(kq::pf_sparse_attn_variant(false), "q4_pf_sparse_attn", c.C - r0, KVH, 1,
              {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(lists.ptr()), PtrArg(O2.ptr()), arg_val(r0),
               arg_val(kC)});
        const std::vector<float> o2 = dv.rd<float>(O2, size_t(QH) * kC * HD);
        for (uint32_t h = 0; h < QH; ++h)
          for (uint32_t m = r0; m < c.C; ++m)
            CHECK(std::memcmp(o2.data() + (size_t(h) * kC + m) * HD, o.data() + (size_t(h) * kC + m) * HD, HD * 4) == 0);
        dv.go(kq::pf_gate_q4_variant(), "pf_attn_gate", QH, c.C, 1,
              {PtrArg(O.ptr()), PtrArg(dgate.ptr()), PtrArg(out.ptr()), arg_val(uint32_t(kC * HD))});
        const std::vector<uint16_t> g = dv.rd<uint16_t>(out, size_t(c.C) * QN);
        for (uint32_t m = r0; m < c.C; ++m)
          for (uint32_t h = 0; h < QH; ++h)
            for (uint32_t d = 0; d < HD; ++d) {
              const float gv = rf(gate[size_t(m) * kq::kQkvgN + size_t(h) * 2 * HD + HD + d]);
              const uint16_t want = rne(rf(o[(size_t(h) * kC + m) * HD + d]) * qr::sigmoid_t(gv));
              gate_ulps = std::max(gate_ulps, ulps16(g[size_t(m) * QN + h * HD + d], want));
            }
      }
    }
    std::printf("   chunk (%6u, %3u): %u dense row(s), sparse rows: flash worst %.7f, eager %.6f vs the eager chain / %.6f "
                "vs fp64\n", c.pos, c.C, r0, worst_sparse, worst_eager, worst_eager64);
  }
  CHECK(worst_dense >= 0.99999);
  CHECK(worst_sparse >= 0.99999);
  CHECK(worst_eager >= 0.9999);
  CHECK(worst_eager64 >= 0.999);
  CHECK(gate_ulps <= 1);
  std::printf("7. attention: dense flash at 24 / 2 (rows <= 2050) worst cosine %.7f; the sparse flash over row lists at "
              "2048..133119 worst %.7f (bar 0.99999), eager %.6f vs its chain (0.9999) / %.6f vs fp64 (0.999); replay "
              "bitwise; pf_attn_gate within %d bf16 ulp of rne(rne(o) x sigmoid(gate))\n",
              worst_dense, worst_sparse, worst_eager, worst_eager64, gate_ulps);
}

}  // namespace

int main() {
  Dev dv;
  std::printf("qwen4exp_pf_kernels_test on %s\n", dv.ctx.name().c_str());
  slabs(dv);
  hc(dv);
  indexer(dv);
  ple(dv);
  const uint32_t C = 64;
  const std::vector<uint16_t> x = rbf16(size_t(kC) * H, -0.2f, 0.2f, 120);
  const std::vector<uint32_t> routes = routers(dv, C, x);
  moe_block(dv, x, routes, C);
  sort_adversaries(dv);
  attention(dv);
  std::printf("qwen4exp_pf_kernels_test OK\n");
  return 0;
}
