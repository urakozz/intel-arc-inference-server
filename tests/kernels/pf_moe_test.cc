// Spec 15d: the prefill MoE kernels on the card at Ornith's shape (256 experts, top-8,
// hidden 2048, expert 512) - src/kernels/prefill/pf_moe.cl and pf_moe_gemm.cl - against
// tests/kernels/pf_moe_ref.h and against the DENSE prefill GEMMs the grouped ones are
// built from. Needs a B70; no checkpoint. Written blind on the Mac (box queue row 12).
//
// One chunk is walked exactly as runtime/prefill/moe.cc walks it (router GEMV, decode's
// moe_route, sort, gather, weights, grouped GEMMs, combine), on random weights, then:
//   1. the router GEMV over the chunk: each sampled row bitwise decode's gemv_bf16 of that
//      row (so prefill and decode route from the same logits for the same x), and the
//      route rows' ids the reference's (near-ties excepted, as moe_test);
//   2. the sort: header, the whole padded tile table, pair_row and the used rows' tokens
//      exactly the host walk's - on the device routes and on crafted ones (all tokens to
//      experts 0..7; one-row experts, the tile bound's adversary; C = 1);
//   3. the gathers: bf16 rows and int8 rows + scales exactly the host gather of the
//      device's own inputs;
//   4. GROUPED == DENSE, bitwise, per (token, expert), for a sample of experts (the most
//      and least loaded, ids 0 / 17 / 128 / 255 when routed, the shared expert):
//        bf16 gate||up (in the engine's two batches) vs pf_gemm_T0_SILU over the chunk,
//        int8 gate||up vs pf_gemm_i8_SILU over the chunk's quantised rows,
//        bf16 down vs rne(pf_gemm_T0) over the expert's sorted h rows -
//      the same dpas sequence per row, so equality is the expectation, and a 1-ulp
//      difference is a finding (not a tolerance);
//   5. row independence (Review Focus 1): the chunk with its tokens in REVERSE order puts
//      every row at another position of its tiles (and in other tiles); every (token,
//      slot)'s h (both paths), y and the combined residual are bitwise the forward run's;
//   6. the combine: exactly pf_moe_ref::combine_token over the device's y and route rows;
//   7. replay: the whole walk again from the same inputs, every buffer bitwise;
//   8. a ragged chunk (C = 37: most experts empty or one row) through 4 and 6.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "pf_moe_ref.h"
#include "runtime/prefill/int8_signs.h"

namespace {

using moe_ref::f32;
using moe_ref::rne;
using pf_moe_ref::kNone;
using pf_moe_ref::kTm;

const moe_ref::Shape kS = moe_ref::ornith_shape();
constexpr uint32_t kH = 2048, kI = 512, kNG = 2 * kI, kB = 257, kKc = 2048;
constexpr uint32_t kNall = kB * kNG;   // the layer's gate||up array as one weight's columns
static_assert(kNall % 16 == 0, "");

uint32_t pad256(uint32_t m) { return (m + 255) / 256 * 256; }

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  template <class T>
  l0::Mem upload(const std::vector<T>& v) {
    l0::Mem m(ctx, l0::MemKind::Device, v.size() * sizeof(T));
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
    return m;
  }
  l0::Mem zeros(size_t bytes) {
    std::vector<uint8_t> z(bytes, 0);
    return upload(z);
  }
  template <class T>
  std::vector<T> read(const l0::Mem& m, size_t n, size_t off_elems = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const T*>(m.ptr()) + off_elems, n * sizeof(T));
    return v;
  }
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy, gz);
    list.close();
    q.execute(list, &f);
    f.wait();
  }
};

// Random layout-1 blocks with real-checkpoint scales (moe_test.cc's generator).
std::vector<uint32_t> random_blocks(size_t words, uint32_t seed) {
  std::vector<uint32_t> v(words);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> sd(0.01f, 0.05f);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = rng();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(sd(rng))) | (uint32_t(common::f32_to_f16(sd(rng))) << 16);
  }
  return v;
}
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(d(rng));
  return v;
}

// Every binary the walk and its dense references bind.
struct Kernels {
  l0::Module pf, dec, router, gemv, g_gu, g_gu8, g_dn, quant, requant, dense, dense_silu, dense_i8;
  l0::Kernel route, sort, gather, gather_i8, dq_gu, dq_dn, combine, rgemv, dgemv, kg_gu, kg_gu8,
      kg_dn, kquant, krequant, kcolmax, kdense, kdense_silu, kdense_i8;
  explicit Kernels(Dev& d)
      : pf(d.ctx, kernels::path(kernels::pf_moe_variant(256, 8, kH, kI))),
        dec(d.ctx, kernels::path(kernels::moe_variant(1, 256, 8, kH, kI))),
        router(d.ctx, kernels::path(kernels::pf_moe_router_variant(kH, 272))),
        gemv(d.ctx, kernels::path(kernels::gemv_bf16_variant(1, kH, 272, kernels::gemv_bf16_tiling(272)))),
        g_gu(d.ctx, kernels::path(kernels::pf_moe_gemm_variant(kH, kNG, false, true))),
        g_gu8(d.ctx, kernels::path(kernels::pf_moe_gemm_variant(kH, kNG, true, true))),
        g_dn(d.ctx, kernels::path(kernels::pf_moe_gemm_variant(kI, kH, false, false))),
        quant(d.ctx, kernels::path(kernels::pf_quant_had_variant(kH))),
        requant(d.ctx, kernels::path(kernels::pf_requant_rot_variant(1))),
        dense(d.ctx, kernels::path(kernels::pf_gemm_variant(false))),
        dense_silu(d.ctx, kernels::path(kernels::pf_gemm_silu_variant())),
        dense_i8(d.ctx, kernels::path(kernels::pf_gemm_i8_variant(true))),
        route(dec, "moe_route"), sort(pf, "pf_moe_sort"), gather(pf, "pf_moe_gather"),
        gather_i8(pf, "pf_moe_gather_i8"), dq_gu(pf, "pf_moe_dequant_gu"),
        dq_dn(pf, "pf_moe_dequant_dn"), combine(pf, "pf_moe_combine"), rgemv(router, "pf_ab_proj"),
        dgemv(gemv, "gemv_bf16"), kg_gu(g_gu, "pf_moe_gemm"), kg_gu8(g_gu8, "pf_moe_gemm"),
        kg_dn(g_dn, "pf_moe_gemm"), kquant(quant, "pf_quant_had"),
        krequant(requant, "pf_requant_rot"), kcolmax(requant, "pf_colmax_rot"),
        kdense(dense, "pf_gemm"), kdense_silu(dense_silu, "pf_gemm"), kdense_i8(dense_i8, "pf_gemm_i8") {
    route.group_size(256);
    sort.group_size(256);
    gather.group_size(64);
    gather_i8.group_size(64);
    dq_gu.group_size(16);
    dq_dn.group_size(16);
    combine.group_size(kernels::pf_moe::kCombineWg);
    rgemv.group_size(16 * 16);
    dgemv.group_size(16 * 16);
    kg_gu.group_size(64);
    kg_gu8.group_size(128);
    kg_dn.group_size(64);
    kquant.group_size(16 * (kH / 1024));
    krequant.group_size(256);
    kcolmax.group_size(256);
    kdense.group_size(512);
    kdense_silu.group_size(512);
    kdense_i8.group_size(512);
  }
};

// The layer's weights on the card, and h8's per-column scales of the gate||up array
// (pf_colmax_rot + Int8State's host finish).
std::vector<uint16_t> router_rows() {
  std::vector<uint16_t> r = random_bf16(size_t(272) * kH, -0.05f, 0.05f, 33);
  std::fill(r.begin() + size_t(257) * kH, r.end(), uint16_t(0));   // rows 257..271 are padding
  return r;
}
std::vector<uint16_t> tiled(const std::vector<uint16_t>& rows) {
  std::vector<uint16_t> t(rows.size());
  common::repack_bf16_tiled(rows.data(), kH, 272, t.data());
  return t;
}
struct Layer {
  std::vector<uint32_t> gu_h = random_blocks(kS.gate_up_words() * kB, 31);
  std::vector<uint32_t> dn_h = random_blocks(kS.down_words() * kB, 32);
  std::vector<uint16_t> rows = router_rows();   // [272][2048] row-major
  l0::Mem router, gu, dn, sbits, signs;
  std::unique_ptr<l0::Mem> ws, inv;
  Layer(Dev& d, Kernels& k)
      : router(d.upload(tiled(rows))),
        gu(d.upload(gu_h)),
        dn(d.upload(dn_h)),
        sbits(d.upload(runtime::prefill::int8_sign_bits(kH))),
        signs(d.upload(runtime::prefill::int8_signs(kH))) {
    l0::Mem colmax = d.zeros(size_t(kNall) * 4);
    k.kcolmax.arg_ptr(0, gu.ptr());
    k.kcolmax.arg_ptr(1, nullptr);
    k.kcolmax.arg_ptr(2, sbits.ptr());
    k.kcolmax.arg_ptr(3, colmax.ptr());
    k.kcolmax.arg(4, 0u);
    k.kcolmax.arg(5, kNall);
    k.kcolmax.arg(6, kH);
    d.run(k.kcolmax, kNall / 16, kH / 1024);
    const std::vector<uint32_t> mx = d.read<uint32_t>(colmax, kNall);
    std::vector<float> w(kNall), iv(kNall);
    for (uint32_t n = 0; n < kNall; ++n) {
      float m;
      std::memcpy(&m, &mx[n], 4);
      w[n] = m > 0.0f ? m / 127.0f : 1.0f;
      iv[n] = 1.0f / w[n];
    }
    ws = std::make_unique<l0::Mem>(d.upload(w));
    inv = std::make_unique<l0::Mem>(d.upload(iv));
  }
};

// One chunk through the walk of runtime/prefill/moe.cc, both gate||up forms.
struct Run {
  uint32_t C = 0, tmax = 0;
  std::vector<float> logits;
  std::vector<uint32_t> route, hdr, tiles, row_tok, pair_row;
  std::vector<uint8_t> xq_rows;           // the quantiser's int8 rows [C][kH]
  std::vector<float> xs;                  // and scales [C]
  std::vector<uint16_t> xg, h_bf, h_i8, y, resid;
  std::vector<uint8_t> xg8;
  std::vector<float> xsg;
};

struct Bufs {   // device scratch for one walk, sized for kKc (+256 rows for the dense refs)
  l0::Mem x, logits, route, hdr, tiles, row_tok, pair_row, xq, xs, xg, xg8, xsg, h, h8, y, w, resid;
};
Bufs alloc(Dev& d) {
  const uint32_t T = pf_moe_ref::tmax(kS, kKc), R = T * kTm + 256;
  auto z = [&](size_t b) { return d.zeros(b); };
  return Bufs{z(size_t(kKc) * kH * 2), z(size_t(kKc) * 272 * 4), z(size_t(kKc) * 32 * 4),
              z(size_t(kernels::pf_moe::hdr_words(256)) * 4), z(size_t(T) * 8), z(size_t(R) * 4),
              z(size_t(kKc) * 8 * 4), z(size_t(pad256(kKc)) * kH), z(size_t(pad256(kKc)) * 4),
              z(size_t(R) * kH * 2), z(size_t(R) * kH), z(size_t(R) * 4), z(size_t(R) * kI * 2),
              z(size_t(R) * kI * 2), z(size_t(R) * kH * 2), z(size_t(129) * kH * kNG * 2),
              z(size_t(kKc) * kH * 2)};
}

Run walk(Dev& d, Kernels& k, Layer& L, Bufs& b, const std::vector<uint16_t>& x,
         const std::vector<uint16_t>& resid0, uint32_t C) {
  Run r;
  r.C = C;
  r.tmax = pf_moe_ref::tmax(kS, C);
  const uint32_t rows = r.tmax * kTm;
  d.imm.copy(b.x.ptr(), x.data(), x.size() * 2);
  d.imm.copy(b.resid.ptr(), resid0.data(), resid0.size() * 2);
  // router GEMV over the chunk + decode's moe_route, grid (1, C)
  k.rgemv.arg_ptr(0, L.router.ptr());
  k.rgemv.arg_ptr(1, b.x.ptr());
  k.rgemv.arg_ptr(2, b.logits.ptr());
  k.rgemv.arg(3, C);
  d.run(k.rgemv, 272 / 16, (C + 7) / 8);
  k.route.arg_ptr(0, b.logits.ptr());
  k.route.arg_ptr(1, b.route.ptr());
  d.run(k.route, 1, C);
  // sort
  k.sort.arg_ptr(0, b.route.ptr());
  k.sort.arg_ptr(1, b.hdr.ptr());
  k.sort.arg_ptr(2, b.tiles.ptr());
  k.sort.arg_ptr(3, b.row_tok.ptr());
  k.sort.arg_ptr(4, b.pair_row.ptr());
  k.sort.arg(5, C);
  k.sort.arg(6, r.tmax);
  d.run(k.sort, 1);
  // bf16: gather, then gate||up in batches of 129 blocks (the engine's: 257 = 129 + 128)
  k.gather.arg_ptr(0, b.x.ptr());
  k.gather.arg_ptr(1, b.hdr.ptr());
  k.gather.arg_ptr(2, b.row_tok.ptr());
  k.gather.arg_ptr(3, b.xg.ptr());
  d.run(k.gather, rows);
  r.xg = d.read<uint16_t>(b.xg, size_t(rows) * kH);   // before y overwrites... (y has its own buffer here)
  for (uint32_t b0 = 0; b0 < kB; b0 += 129) {
    const uint32_t nb = std::min(129u, kB - b0);
    k.dq_gu.arg_ptr(0, L.gu.ptr());
    k.dq_gu.arg_ptr(1, b.hdr.ptr());
    k.dq_gu.arg_ptr(2, b.w.ptr());
    k.dq_gu.arg(3, b0);
    d.run(k.dq_gu, kNG / 16, kH / 64, nb);
    k.kg_gu.arg_ptr(0, b.tiles.ptr());
    k.kg_gu.arg_ptr(1, b.xg.ptr());
    k.kg_gu.arg_ptr(2, nullptr);
    k.kg_gu.arg_ptr(3, b.w.ptr());
    k.kg_gu.arg_ptr(4, nullptr);
    k.kg_gu.arg_ptr(5, b.h.ptr());
    k.kg_gu.arg(6, b0);
    k.kg_gu.arg(7, b0 + nb);
    k.kg_gu.arg(8, 0u);
    d.run(k.kg_gu, r.tmax, kNG / 256);
  }
  // int8 (h8): quantise the chunk once, gather rows + scales, requant every block, one GEMM
  k.kquant.arg_ptr(0, b.x.ptr());
  k.kquant.arg_ptr(1, L.signs.ptr());
  k.kquant.arg_ptr(2, b.xq.ptr());
  k.kquant.arg_ptr(3, b.xs.ptr());
  k.kquant.arg(4, kH);
  d.run(k.kquant, C);
  k.gather_i8.arg_ptr(0, b.xq.ptr());
  k.gather_i8.arg_ptr(1, b.xs.ptr());
  k.gather_i8.arg_ptr(2, b.hdr.ptr());
  k.gather_i8.arg_ptr(3, b.row_tok.ptr());
  k.gather_i8.arg_ptr(4, b.xg8.ptr());
  k.gather_i8.arg_ptr(5, b.xsg.ptr());
  d.run(k.gather_i8, rows);
  k.krequant.arg_ptr(0, L.gu.ptr());
  k.krequant.arg_ptr(1, nullptr);
  k.krequant.arg_ptr(2, L.sbits.ptr());
  k.krequant.arg_ptr(3, L.inv->ptr());
  k.krequant.arg_ptr(4, b.w.ptr());
  k.krequant.arg(5, 0u);
  k.krequant.arg(6, kNall);
  k.krequant.arg(7, kH);
  k.krequant.arg(8, kNall);
  d.run(k.krequant, kNall / 16, kH / 1024);
  k.kg_gu8.arg_ptr(0, b.tiles.ptr());
  k.kg_gu8.arg_ptr(1, b.xg8.ptr());
  k.kg_gu8.arg_ptr(2, b.xsg.ptr());
  k.kg_gu8.arg_ptr(3, b.w.ptr());
  k.kg_gu8.arg_ptr(4, L.ws->ptr());
  k.kg_gu8.arg_ptr(5, b.h8.ptr());
  k.kg_gu8.arg(6, 0u);
  k.kg_gu8.arg(7, kB);
  k.kg_gu8.arg(8, kNall);
  d.run(k.kg_gu8, r.tmax, kNG / 256);
  // down (bf16) on the bf16 path's h, one batch of all 257 blocks
  k.dq_dn.arg_ptr(0, L.dn.ptr());
  k.dq_dn.arg_ptr(1, b.hdr.ptr());
  k.dq_dn.arg_ptr(2, b.w.ptr());
  k.dq_dn.arg(3, 0u);
  d.run(k.dq_dn, kH / 16, kI / 64, kB);
  k.kg_dn.arg_ptr(0, b.tiles.ptr());
  k.kg_dn.arg_ptr(1, b.h.ptr());
  k.kg_dn.arg_ptr(2, nullptr);
  k.kg_dn.arg_ptr(3, b.w.ptr());
  k.kg_dn.arg_ptr(4, nullptr);
  k.kg_dn.arg_ptr(5, b.y.ptr());
  k.kg_dn.arg(6, 0u);
  k.kg_dn.arg(7, kB);
  k.kg_dn.arg(8, 0u);
  d.run(k.kg_dn, r.tmax, kH / 256);
  // combine
  k.combine.arg_ptr(0, b.route.ptr());
  k.combine.arg_ptr(1, b.hdr.ptr());
  k.combine.arg_ptr(2, b.pair_row.ptr());
  k.combine.arg_ptr(3, b.y.ptr());
  k.combine.arg_ptr(4, b.resid.ptr());
  k.combine.arg(5, C);
  d.run(k.combine, kH / kernels::pf_moe::kCombineWg, C);

  r.logits = d.read<float>(b.logits, size_t(C) * 272);
  r.route = d.read<uint32_t>(b.route, size_t(C) * 32);
  r.hdr = d.read<uint32_t>(b.hdr, kernels::pf_moe::hdr_words(256));
  r.tiles = d.read<uint32_t>(b.tiles, size_t(r.tmax) * 2);
  r.row_tok = d.read<uint32_t>(b.row_tok, rows);
  r.pair_row = d.read<uint32_t>(b.pair_row, size_t(C) * 8);
  r.xq_rows = d.read<uint8_t>(b.xq, size_t(C) * kH);
  r.xs = d.read<float>(b.xs, C);
  r.xg8 = d.read<uint8_t>(b.xg8, size_t(rows) * kH);
  r.xsg = d.read<float>(b.xsg, rows);
  r.h_bf = d.read<uint16_t>(b.h, size_t(rows) * kI);
  r.h_i8 = d.read<uint16_t>(b.h8, size_t(rows) * kI);
  r.y = d.read<uint16_t>(b.y, size_t(rows) * kH);
  r.resid = d.read<uint16_t>(b.resid, size_t(C) * kH);
  return r;
}

// The sorted row of (token t, slot k), k == top_k meaning the shared expert.
uint32_t row_of(const Run& r, uint32_t t, uint32_t k) {
  return k < kS.top_k ? r.pair_row[size_t(t) * kS.top_k + k] : r.hdr[kernels::pf_moe::kHdrSharedRow] + t;
}
uint32_t expert_of(const Run& r, uint32_t t, uint32_t k) {
  return k < kS.top_k ? r.route[size_t(t) * 32 + k] : kS.shared_block();
}

// 4b. plan 2026-11-12 Task 3: pf_moe_q4gemm (B straight from the int4 blocks, over the sort's work items) ==
// today's grouped GEMM over pf_moe_dequant_*'s bf16 blocks, every used row bitwise - gate||up + SiLU on the walk's
// gathered rows, down on the walk's bf16 h.
void fused_vs_grouped(Dev& d, Layer& L, Bufs& b, const Run& r) {
  l0::Module mgu(d.ctx, kernels::path(kernels::pf_moe_q4gemm_variant(kH, kNG, true))),
      mdn(d.ctx, kernels::path(kernels::pf_moe_q4gemm_variant(kI, kH, false)));
  l0::Kernel gu(mgu, "pf_moe_q4gemm"), dn(mdn, "pf_moe_q4gemm");
  gu.group_size(16 * (kernels::pf_moe::kQ4WgN / 32));
  dn.group_size(16 * (kernels::pf_moe::kQ4WgN / 32));
  pf_moe_ref::Sorted st;
  st.hdr = r.hdr;
  st.tiles = r.tiles;
  const std::vector<uint32_t> it = pf_moe_ref::items(st, kS, r.C, kernels::pf_moe::kItemM);
  const uint32_t nit = uint32_t(it.size() / 4), used = r.hdr[kernels::pf_moe::kHdrRows];
  l0::Mem itd = d.upload(it), hf = d.zeros(size_t(used) * kI * 2), yf = d.zeros(size_t(used) * kH * 2);
  d.imm.copy(b.xg.ptr(), r.xg.data(), size_t(used) * kH * 2);
  d.imm.copy(b.h.ptr(), r.h_bf.data(), size_t(used) * kI * 2);
  gu.arg_ptr(0, itd.ptr());
  gu.arg_ptr(1, b.xg.ptr());
  gu.arg_ptr(2, L.gu.ptr());
  gu.arg_ptr(3, L.gu.as<uint32_t>() + kS.gate_up_words() * kS.experts);
  gu.arg_ptr(4, hf.ptr());
  d.run(gu, kNG / kernels::pf_moe::kQ4WgN, nit);
  dn.arg_ptr(0, itd.ptr());
  dn.arg_ptr(1, b.h.ptr());
  dn.arg_ptr(2, L.dn.ptr());
  dn.arg_ptr(3, L.dn.as<uint32_t>() + kS.down_words() * kS.experts);
  dn.arg_ptr(4, yf.ptr());
  d.run(dn, kH / kernels::pf_moe::kQ4WgN, nit);
  const std::vector<uint16_t> h = d.read<uint16_t>(hf, size_t(used) * kI), y = d.read<uint16_t>(yf, size_t(used) * kH);
  size_t bh = 0, by = 0;
  for (size_t i = 0; i < h.size(); ++i) bh += h[i] != r.h_bf[i];
  for (size_t i = 0; i < y.size(); ++i) by += y[i] != r.y[i];
  std::printf("  4b. fused int4 GEMM == grouped, C %u (%u items): h %zu of %zu, y %zu of %zu values differ\n", r.C,
              nit, bh, h.size(), by, y.size());
  CHECK_EQ(bh, size_t(0));
  CHECK_EQ(by, size_t(0));
}

// 4c. plan 2026-11-12 Task 5: l0-int8's W4A8 gate||up (pf_moe_w4a8.cl). The quantiser's codes and scales exactly
// the host's (s = max|x| / 127 a token and 64-k group, rint(x / s)); the gather exactly the host gather; the GEMM's
// h on every row of the first and the last item (a routed expert, the shared one) against the host's integer sums
// rescaled in the kernel's fma order and the SiLU chain - within one bf16 ulp (std::exp against the device's exp),
// the count of 1-ulp rows printed.
void w4a8_check(Dev& d, Layer& L, Bufs& b, const Run& r) {
  constexpr uint32_t G = kH / 64;
  l0::Module mw(d.ctx, kernels::path(kernels::pf_moe_w4a8_variant(kH, kNG)));
  l0::Kernel quant(mw, "pf_quant_g64"), gather(mw, "pf_moe_gather_i8g"), gemm(mw, "pf_moe_w4a8");
  quant.group_size(16 * G);
  gather.group_size(64);
  gemm.group_size(16 * 2 * 8);
  const uint32_t C = r.C, used = r.hdr[kernels::pf_moe::kHdrRows];
  const std::vector<uint16_t> x = d.read<uint16_t>(b.x, size_t(C) * kH);
  l0::Mem xq = d.zeros(size_t(C) * kH), xs = d.zeros(size_t(C) * G * 4), xg = d.zeros(size_t(used) * kH),
          xsg = d.zeros(size_t(used) * G * 4), h = d.zeros(size_t(used) * kI * 2);
  quant.arg_ptr(0, b.x.ptr());
  quant.arg_ptr(1, xq.ptr());
  quant.arg_ptr(2, xs.ptr());
  quant.arg(3, C);
  d.run(quant, C);
  const std::vector<int8_t> q = d.read<int8_t>(xq, size_t(C) * kH);
  const std::vector<float> s = d.read<float>(xs, size_t(C) * G);
  size_t bad = 0;
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t g = 0; g < G; ++g) {
      float m = 0;
      for (uint32_t k = 0; k < 64; ++k) m = std::max(m, std::fabs(f32(x[size_t(t) * kH + g * 64 + k])));
      const float sc = m > 0 ? m / 127.0f : 1.0f;
      bad += std::memcmp(&sc, &s[size_t(t) * G + g], 4) != 0;
      for (uint32_t k = 0; k < 64; ++k) {
        const float v = std::nearbyint(f32(x[size_t(t) * kH + g * 64 + k]) / sc);
        bad += q[size_t(t) * kH + g * 64 + k] != int8_t(std::max(-127.0f, std::min(127.0f, v)));
      }
    }
  CHECK_EQ(bad, size_t(0));
  gather.arg_ptr(0, xq.ptr());
  gather.arg_ptr(1, xs.ptr());
  gather.arg_ptr(2, b.hdr.ptr());
  gather.arg_ptr(3, b.row_tok.ptr());
  gather.arg_ptr(4, xg.ptr());
  gather.arg_ptr(5, xsg.ptr());
  d.imm.copy(b.hdr.ptr(), r.hdr.data(), r.hdr.size() * 4);
  d.imm.copy(b.row_tok.ptr(), r.row_tok.data(), r.row_tok.size() * 4);
  d.run(gather, used);
  const std::vector<int8_t> ag = d.read<int8_t>(xg, size_t(used) * kH);
  const std::vector<float> sg = d.read<float>(xsg, size_t(used) * G);
  for (uint32_t row = 0; row < used; ++row) {
    const uint32_t t = r.row_tok[row];
    for (uint32_t k = 0; k < kH; ++k) bad += ag[size_t(row) * kH + k] != (t == kNone ? 0 : q[size_t(t) * kH + k]);
    for (uint32_t g = 0; g < G; ++g) bad += sg[size_t(row) * G + g] != (t == kNone ? 0.0f : s[size_t(t) * G + g]);
  }
  CHECK_EQ(bad, size_t(0));
  pf_moe_ref::Sorted st;
  st.hdr = r.hdr;
  st.tiles = r.tiles;
  const std::vector<uint32_t> it = pf_moe_ref::items(st, kS, C, kernels::pf_moe::kItemM);
  const uint32_t nit = uint32_t(it.size() / 4);
  l0::Mem itd = d.upload(it);
  gemm.arg_ptr(0, itd.ptr());
  gemm.arg_ptr(1, xg.ptr());
  gemm.arg_ptr(2, xsg.ptr());
  gemm.arg_ptr(3, L.gu.ptr());
  gemm.arg_ptr(4, L.gu.as<uint32_t>() + kS.gate_up_words() * kS.experts);
  gemm.arg_ptr(5, h.ptr());
  d.run(gemm, kNG / kernels::pf_moe::kQ4WgN, nit);
  const std::vector<uint16_t> hd = d.read<uint16_t>(h, size_t(used) * kI);
  size_t n = 0, ulp1 = 0;
  uint32_t last = 0;
  while (last + 1 < nit && it[4 * (last + 1)] != kNone) ++last;
  for (uint32_t i : {0u, last}) {
    const uint32_t e = it[4 * i], r0 = it[4 * i + 1], nr = it[4 * i + 2];
    const uint32_t* blk = L.gu_h.data() + kS.gate_up_words() * e;
    for (uint32_t row = r0; row < r0 + nr; ++row)
      for (uint32_t col = 0; col < kNG; col += 32)
        for (uint32_t l = 0; l < 16; ++l) {
          float acc[2];
          for (uint32_t half = 0; half < 2; ++half) {   // gate atom (col / 16 even), its up atom
            const uint32_t nn = col + 16 * half + l;
            float a = 0.0f;
            for (uint32_t g = 0; g < G; ++g) {
              const uint32_t* tile = blk + (size_t(nn / 16) * G + g) * 136;
              int32_t tsum = 0;
              for (uint32_t j = 0; j < 8; ++j)
                for (uint32_t ii = 0; ii < 8; ++ii)
                  tsum += int32_t(ag[size_t(row) * kH + g * 64 + j * 8 + ii]) *
                          (int32_t((tile[j * 16 + nn % 16] >> (4 * ii)) & 0xF) - 8);
              const float ws = common::f16_to_f32(reinterpret_cast<const uint16_t*>(tile + 128)[nn % 16]);
              a = std::fma(float(tsum), ws * sg[size_t(row) * G + g], a);
            }
            acc[half] = a;
          }
          const uint16_t g_b = rne(acc[0]), u_b = rne(acc[1]);
          const uint16_t s_b = rne(f32(g_b) / (1.0f + std::exp(-f32(g_b))));
          const uint16_t want = rne(f32(s_b) * f32(u_b));
          const uint16_t got = hd[size_t(row) * kI + col / 2 + l];
          ++n;
          if (got != want) {
            const int du = std::abs(int(got) - int(want));
            if (du == 1) ++ulp1;
            else ++bad;
          }
        }
  }
  std::printf("  4c. W4A8 gate||up C %u: quantiser and gather exact; h %zu values of two items against the host's "
              "integer sums: %zu one bf16 ulp off (exp), %zu further\n", C, n, ulp1, bad);
  CHECK_EQ(bad, size_t(0));
}

// 4: grouped == dense, bitwise, for expert e (all three GEMMs).
void grouped_vs_dense(Dev& d, Kernels& k, Layer& L, Bufs& b, const Run& r, uint32_t e) {
  const uint32_t C = r.C, Mp = pad256(C);
  // the chunk's x, zero-padded to Mp rows (the dense GEMM's tile), already in b.x rows < C
  std::vector<uint16_t> xp(size_t(Mp) * kH, 0);
  {
    const std::vector<uint16_t> x = d.read<uint16_t>(b.x, size_t(C) * kH);
    std::copy(x.begin(), x.end(), xp.begin());
  }
  l0::Mem A = d.upload(xp);
  // expert e's bf16 gate||up and down blocks (pf_moe_dequant_*: b0 = e, one block); the
  // header says e is routed (or e is the shared block), so neither is skipped
  l0::Mem wgu = d.zeros(size_t(kH) * kNG * 2), wdn = d.zeros(size_t(kI) * kH * 2);
  k.dq_gu.arg_ptr(0, L.gu.ptr());
  k.dq_gu.arg_ptr(1, b.hdr.ptr());
  k.dq_gu.arg_ptr(2, wgu.ptr());
  k.dq_gu.arg(3, e);
  d.run(k.dq_gu, kNG / 16, kH / 64, 1);
  k.dq_dn.arg_ptr(0, L.dn.ptr());
  k.dq_dn.arg_ptr(1, b.hdr.ptr());
  k.dq_dn.arg_ptr(2, wdn.ptr());
  k.dq_dn.arg(3, e);
  d.run(k.dq_dn, kH / 16, kI / 64, 1);
  CHECK(d.read<uint16_t>(wgu, size_t(kH) * kNG) ==
        pf_moe_ref::dequant_block(L.gu_h.data() + size_t(e) * kS.gate_up_words(), kH, kNG));
  // bf16 gate||up, dense: X [Mp][512] = silu(x W_e)
  l0::Mem X = d.zeros(size_t(Mp) * kI * 2);
  k.kdense_silu.arg_ptr(0, A.ptr());
  k.kdense_silu.arg_ptr(1, wgu.ptr());
  k.kdense_silu.arg_ptr(2, X.ptr());
  k.kdense_silu.arg(3, Mp);
  k.kdense_silu.arg(4, kH);
  k.kdense_silu.arg(5, kNG);
  k.kdense_silu.arg(6, kH);
  k.kdense_silu.arg(7, kNG);
  k.kdense_silu.arg(8, kI);
  k.kdense_silu.arg(9, uint64_t(0));
  k.kdense_silu.arg(10, uint64_t(0));
  k.kdense_silu.arg(11, uint64_t(0));
  d.run(k.kdense_silu, Mp / 256, kNG / 256, 1);
  const std::vector<uint16_t> xd = d.read<uint16_t>(X, size_t(C) * kI);
  // int8 gate||up, dense over the quantised chunk: pf_gemm_i8_SILU at expert e's columns of
  // the requantised array (rebuilt into b.w: the walk's last use of it was down's bf16)
  k.krequant.arg_ptr(4, b.w.ptr());
  d.run(k.krequant, kNall / 16, kH / 1024);
  l0::Mem X8 = d.zeros(size_t(Mp) * kI * 2);
  k.kdense_i8.arg_ptr(0, b.xq.ptr());
  k.kdense_i8.arg_ptr(1, b.xs.ptr());
  k.kdense_i8.arg_ptr(2, static_cast<const uint32_t*>(b.w.ptr()) + size_t(e) * kNG);
  k.kdense_i8.arg_ptr(3, static_cast<const float*>(L.ws->ptr()) + size_t(e) * kNG);
  k.kdense_i8.arg_ptr(4, X8.ptr());
  k.kdense_i8.arg(5, Mp);
  k.kdense_i8.arg(6, kH);
  k.kdense_i8.arg(7, kNG);
  k.kdense_i8.arg(8, kH / 2);
  k.kdense_i8.arg(9, kNall);
  k.kdense_i8.arg(10, kI);
  d.run(k.kdense_i8, Mp / 256, kNG / 128);
  const std::vector<uint16_t> xd8 = d.read<uint16_t>(X8, size_t(C) * kI);
  // down, dense over expert e's sorted h rows: [r0, r0 + cnt), padded to 256 rows
  uint32_t r0 = kNone, cnt = 0;
  for (uint32_t t = 0; t < r.tmax; ++t)
    if (r.tiles[2 * t] == e) {
      if (r0 == kNone) r0 = r.tiles[2 * t + 1];
    }
  cnt = e < kS.experts ? r.hdr[kernels::pf_moe::kHdrCount + e] : C;
  CHECK(r0 != kNone && cnt > 0);
  const uint32_t Mc = pad256(cnt);
  l0::Mem Y = d.zeros(size_t(Mc) * kH * 4);
  k.kdense.arg_ptr(0, static_cast<const uint16_t*>(b.h.ptr()) + size_t(r0) * kI);
  k.kdense.arg_ptr(1, wdn.ptr());
  k.kdense.arg_ptr(2, Y.ptr());
  k.kdense.arg(3, Mc);
  k.kdense.arg(4, kI);
  k.kdense.arg(5, kH);
  k.kdense.arg(6, kI);
  k.kdense.arg(7, kH);
  k.kdense.arg(8, kH);
  k.kdense.arg(9, uint64_t(0));
  k.kdense.arg(10, uint64_t(0));
  k.kdense.arg(11, uint64_t(0));
  d.run(k.kdense, Mc / 256, kH / 256, 1);
  const std::vector<float> yd = d.read<float>(Y, size_t(cnt) * kH);
  // compare every (token, slot) routed to e
  size_t rows_gu = 0, bad_gu = 0, bad_gu8 = 0;
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t s = 0; s <= kS.top_k; ++s) {
      if (expert_of(r, t, s) != e) continue;
      const uint32_t row = row_of(r, t, s);
      ++rows_gu;
      bad_gu += std::memcmp(&r.h_bf[size_t(row) * kI], &xd[size_t(t) * kI], kI * 2) != 0;
      bad_gu8 += std::memcmp(&r.h_i8[size_t(row) * kI], &xd8[size_t(t) * kI], kI * 2) != 0;
    }
  size_t bad_dn = 0;
  for (uint32_t i = 0; i < cnt; ++i)
    for (uint32_t n = 0; n < kH; ++n) bad_dn += r.y[size_t(r0 + i) * kH + n] != rne(yd[size_t(i) * kH + n]);
  std::printf("    expert %3u: %5zu rows; grouped vs dense - gate||up bf16 %zu rows differ, int8 %zu, "
              "down %zu values\n", e, rows_gu, bad_gu, bad_gu8, bad_dn);
  CHECK_EQ(rows_gu, size_t(cnt));
  CHECK_EQ(bad_gu, size_t(0));
  CHECK_EQ(bad_gu8, size_t(0));
  CHECK_EQ(bad_dn, size_t(0));
}

// 6: the combine, exactly, over the device's own y and route rows.
void check_combine(const Run& r, const std::vector<uint16_t>& resid0) {
  pf_moe_ref::Sorted st;
  st.hdr = r.hdr;
  st.pair_row = r.pair_row;
  size_t bad = 0;
  for (uint32_t t = 0; t < r.C; ++t)
    for (uint32_t n = 0; n < kH; ++n)
      bad += r.resid[size_t(t) * kH + n] !=
             pf_moe_ref::combine_token(st, &r.route[size_t(t) * 32], r.y.data(), kH, kS, t, n,
                                       resid0[size_t(t) * kH + n]);
  std::printf("    combine C %u: %zu of %zu residual values differ from the host chain\n", r.C, bad,
              size_t(r.C) * kH);
  CHECK_EQ(bad, size_t(0));
}

void check_sort(const std::vector<uint32_t>& route, const Run& r) {
  const pf_moe_ref::Sorted want = pf_moe_ref::sort(route.data(), r.C, kS);
  CHECK(r.hdr == want.hdr);
  CHECK(r.tiles == want.tiles);
  CHECK(r.pair_row == want.pair_row);
  CHECK(std::equal(want.row_tok.begin(), want.row_tok.begin() + want.rows_used(), r.row_tok.begin()));
}

// The experts section 4 checks for a run: the most and the least loaded routed expert,
// 0 / 17 / 128 / 255 when routed, and the shared expert.
std::vector<uint32_t> sample_experts(const Run& r) {
  std::vector<uint32_t> out;
  uint32_t hi = 0, lo = 0;
  for (uint32_t e = 0; e < kS.experts; ++e) {
    const uint32_t c = r.hdr[kernels::pf_moe::kHdrCount + e];
    if (c == 0) continue;
    if (c > r.hdr[kernels::pf_moe::kHdrCount + hi]) hi = e;
    if (r.hdr[kernels::pf_moe::kHdrCount + lo] == 0 || c < r.hdr[kernels::pf_moe::kHdrCount + lo]) lo = e;
  }
  for (uint32_t e : {hi, lo, 0u, 17u, 128u, 255u})
    if (r.hdr[kernels::pf_moe::kHdrCount + e] > 0 &&
        std::find(out.begin(), out.end(), e) == out.end())
      out.push_back(e);
  out.push_back(kS.shared_block());
  return out;
}

}  // namespace

int main() {
  Dev d;
  Kernels k(d);
  Layer L(d, k);
  Bufs b = alloc(d);

  // ---- the full chunk ------------------------------------------------------------------
  const uint32_t C = kKc;
  const std::vector<uint16_t> x = random_bf16(size_t(C) * kH, -1.0f, 1.0f, 41);
  const std::vector<uint16_t> resid0 = random_bf16(size_t(C) * kH, -2.0f, 2.0f, 42);
  const Run a = walk(d, k, L, b, x, resid0, C);

  // 1. the router GEMV is decode's, row for row; the route rows the reference's
  {
    l0::Mem xr(d.ctx, l0::MemKind::Device, size_t(kH) * 2), lg(d.ctx, l0::MemKind::Device, 272 * 4);
    size_t bad = 0;
    for (uint32_t t : {0u, 1u, 7u, 8u, 1000u, C - 1}) {
      d.imm.copy(xr.ptr(), x.data() + size_t(t) * kH, size_t(kH) * 2);
      k.dgemv.arg_ptr(0, L.router.ptr());
      k.dgemv.arg_ptr(1, xr.ptr());
      k.dgemv.arg_ptr(2, lg.ptr());
      d.run(k.dgemv, 272 / 16);
      bad += std::memcmp(d.read<float>(lg, 272).data(), &a.logits[size_t(t) * 272], 272 * 4) != 0;
    }
    CHECK_EQ(bad, size_t(0));
    size_t ties = 0;
    for (uint32_t t = 0; t < C; ++t) {
      const moe_ref::Route want = moe_ref::route(&a.logits[size_t(t) * 272], kS);
      const float pk = want.p[kS.top_k - 1];
      const bool near_tie = pk != want.p9 && pk - want.p9 <= 1e-5f * want.p9;
      ties += near_tie;
      for (uint32_t s = 0; s < kS.top_k - (near_tie ? 1 : 0); ++s) CHECK_EQ(a.route[size_t(t) * 32 + s], want.ids[s]);
    }
    std::printf("  1. router GEMV rows bitwise decode's gemv_bf16; %u route rows' ids the "
                "reference's (%zu near-ties at the boundary)\n", C, ties);
  }
  // 2. the sort on the device routes, and crafted routes through the sort alone
  check_sort(a.route, a);
  {
    auto sort_only = [&](const std::vector<uint32_t>& route, uint32_t Cc, const char* what) {
      d.imm.copy(b.route.ptr(), route.data(), route.size() * 4);
      Run r;
      r.C = Cc;
      r.tmax = pf_moe_ref::tmax(kS, Cc);
      k.sort.arg_ptr(0, b.route.ptr());
      k.sort.arg(5, Cc);
      k.sort.arg(6, r.tmax);
      d.run(k.sort, 1);
      r.hdr = d.read<uint32_t>(b.hdr, kernels::pf_moe::hdr_words(256));
      r.tiles = d.read<uint32_t>(b.tiles, size_t(r.tmax) * 2);
      r.row_tok = d.read<uint32_t>(b.row_tok, size_t(r.tmax) * kTm);
      r.pair_row = d.read<uint32_t>(b.pair_row, size_t(Cc) * 8);
      check_sort(route, r);
      std::printf("     sort %-22s C %4u: %3u tiles of %3u - exact\n", what, Cc,
                  r.hdr[kernels::pf_moe::kHdrTiles], r.tmax);
    };
    std::vector<uint32_t> rt(size_t(C) * 32, 0);
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t s = 0; s < 8; ++s) rt[size_t(t) * 32 + s] = 7 - s;
    sort_only(rt, C, "all to experts 0..7");
    uint32_t single = 8;
    for (uint32_t t = 0; t < C; ++t) {
      for (uint32_t s = 0; s < 8; ++s) rt[size_t(t) * 32 + s] = s;
      if (single < 256) rt[size_t(t) * 32] = single++;
    }
    sort_only(rt, C, "one-row experts");
    sort_only(rt, 1, "C = 1");
  }
  std::puts("  2. pf_moe_sort: header, padded tile table, pair_row, rows exactly the host walk's");
  // 2b. plan 2026-11-12 Task 2: pf_moe_sort_mw (hist + scan + scatter) - every buffer exactly the host walk's (so
  // pf_moe_sort's), and its work items the host's, on the device routes, the crafted ones, C = 1 / 37, and random routes past the old KC (4096,
  // 16384), in its own buffers
  {
    l0::Module mw(d.ctx, kernels::path(kernels::pf_moe_sort_mw_variant(256, 8)));
    l0::Kernel hist(mw, "pf_moe_hist"), scan(mw, "pf_moe_scan"), scat(mw, "pf_moe_scatter");
    hist.group_size(256);
    scan.group_size(256);
    scat.group_size(256);
    constexpr uint32_t kCx = 16384;
    const uint32_t Tx = pf_moe_ref::tmax(kS, kCx);
    l0::Mem rt_d = d.zeros(size_t(kCx) * 32 * 4), hdr_d = d.zeros(size_t(kernels::pf_moe::hdr_words(256)) * 4),
            tl_d = d.zeros(size_t(Tx) * 8), rtk_d = d.zeros(size_t(Tx) * kTm * 4),
            pr_d = d.zeros(size_t(kCx) * 8 * 4), cnt_d = d.zeros(size_t(kCx / 64) * 256 * 4),
            it_d = d.zeros(size_t(pf_moe_ref::items_bound(kS, kCx, kernels::pf_moe::kItemM)) * 16);
    auto mw_sort = [&](const std::vector<uint32_t>& route, uint32_t Cc, const char* what) {
      d.imm.copy(rt_d.ptr(), route.data(), size_t(Cc) * 32 * 4);
      Run r;
      r.C = Cc;
      r.tmax = pf_moe_ref::tmax(kS, Cc);
      const uint32_t nb = (Cc + 63) / 64;
      hist.arg_ptr(0, rt_d.ptr());
      hist.arg_ptr(1, cnt_d.ptr());
      hist.arg(2, Cc);
      d.run(hist, nb);
      scan.arg_ptr(0, cnt_d.ptr());
      scan.arg_ptr(1, hdr_d.ptr());
      scan.arg_ptr(2, tl_d.ptr());
      scan.arg_ptr(3, rtk_d.ptr());
      scan.arg_ptr(4, it_d.ptr());
      scan.arg(5, Cc);
      scan.arg(6, r.tmax);
      const uint32_t ib = pf_moe_ref::items_bound(kS, Cc, kernels::pf_moe::kItemM);
      scan.arg(7, ib);
      d.run(scan, 1);
      scat.arg_ptr(0, rt_d.ptr());
      scat.arg_ptr(1, cnt_d.ptr());
      scat.arg_ptr(2, rtk_d.ptr());
      scat.arg_ptr(3, pr_d.ptr());
      scat.arg(4, Cc);
      d.run(scat, nb);
      r.hdr = d.read<uint32_t>(hdr_d, kernels::pf_moe::hdr_words(256));
      r.tiles = d.read<uint32_t>(tl_d, size_t(r.tmax) * 2);
      r.row_tok = d.read<uint32_t>(rtk_d, size_t(r.tmax) * kTm);
      r.pair_row = d.read<uint32_t>(pr_d, size_t(Cc) * 8);
      check_sort(route, r);
      CHECK(d.read<uint32_t>(it_d, size_t(ib) * 4) ==
            pf_moe_ref::items(pf_moe_ref::sort(route.data(), Cc, kS), kS, Cc, kernels::pf_moe::kItemM));
      std::printf("     mw sort %-22s C %5u: %4u tiles of %4u - exact\n", what, Cc, r.hdr[kernels::pf_moe::kHdrTiles],
                  r.tmax);
    };
    mw_sort(a.route, C, "the device routes");
    std::vector<uint32_t> rt(size_t(C) * 32, 0);
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t s = 0; s < 8; ++s) rt[size_t(t) * 32 + s] = 7 - s;
    mw_sort(rt, C, "all to experts 0..7");
    uint32_t single = 8;
    for (uint32_t t = 0; t < C; ++t) {
      for (uint32_t s = 0; s < 8; ++s) rt[size_t(t) * 32 + s] = s;
      if (single < 256) rt[size_t(t) * 32] = single++;
    }
    mw_sort(rt, C, "one-row experts");
    mw_sort(rt, 1, "C = 1");
    mw_sort(a.route, 37, "the device routes");
    for (uint32_t Cc : {4096u, kCx}) {
      std::vector<uint32_t> big(size_t(Cc) * 32, 0);
      std::mt19937 rng(Cc);
      std::vector<uint32_t> ids(256);
      for (uint32_t t = 0; t < Cc; ++t) {
        std::iota(ids.begin(), ids.end(), 0u);
        std::shuffle(ids.begin(), ids.end(), rng);
        for (uint32_t s = 0; s < 8; ++s) big[size_t(t) * 32 + s] = ids[s];
      }
      mw_sort(big, Cc, "random routes");
    }
  }
  // 3. the gathers of the device's own inputs
  {
    // The walk reads the WHOLE buffers (tmax x TM rows); the gathers write rows
    // [0, hdr[kHdrRows]) only and the reference holds exactly those, so the comparison is
    // over the used prefix - as check_sort's row_tok. Comparing the vectors whole compared
    // two sizes (the box's first run: 824 x 16 rows read, fewer used) and failed on that alone.
    pf_moe_ref::Sorted st = pf_moe_ref::sort(a.route.data(), C, kS);
    const size_t used = st.rows_used();
    CHECK_EQ(a.hdr[kernels::pf_moe::kHdrRows], uint32_t(used));
    CHECK(used <= size_t(a.tmax) * kTm);
    const std::vector<uint16_t> want = pf_moe_ref::gather(st, x.data(), kH);
    const std::vector<uint8_t> want8 = pf_moe_ref::gather(st, a.xq_rows.data(), kH);
    CHECK_EQ(want.size(), used * kH);
    CHECK_EQ(want8.size(), used * kH);
    size_t bad = 0, bad8 = 0, bads = 0;
    for (size_t r = 0; r < used; ++r) {
      bad += std::memcmp(&a.xg[r * kH], &want[r * kH], kH * 2) != 0;
      bad8 += std::memcmp(&a.xg8[r * kH], &want8[r * kH], kH) != 0;
      bads += a.xsg[r] != (st.row_tok[r] == kNone ? 0.0f : a.xs[st.row_tok[r]]);
    }
    std::printf("  3. pf_moe_gather / _i8: %zu used rows of %u read (%u tiles of %u); %zu bf16, %zu int8 "
                "rows and %zu scales differ\n", used, a.tmax * kTm, st.tiles_used(), a.tmax, bad, bad8, bads);
    CHECK_EQ(bad, size_t(0));
    CHECK_EQ(bad8, size_t(0));
    CHECK_EQ(bads, size_t(0));
  }
  // 4. grouped == dense. Section 2's crafted sorts rewrote the device header, tile table,
  // row_tok and pair_row (its last: C = 1, experts 1..8 only), and pf_moe_dequant_* skips a
  // routed block whose header count is 0 - so the chunk's own sort goes back first, or the
  // dequant of an expert the C = 1 header does not route is skipped and reads as zeros.
  d.imm.copy(b.hdr.ptr(), a.hdr.data(), a.hdr.size() * 4);
  d.imm.copy(b.tiles.ptr(), a.tiles.data(), a.tiles.size() * 4);
  d.imm.copy(b.row_tok.ptr(), a.row_tok.data(), a.row_tok.size() * 4);
  d.imm.copy(b.pair_row.ptr(), a.pair_row.data(), a.pair_row.size() * 4);
  std::puts("  4. grouped == dense, bitwise, per (token, expert):");
  for (uint32_t e : sample_experts(a)) grouped_vs_dense(d, k, L, b, a, e);
  fused_vs_grouped(d, L, b, a);
  w4a8_check(d, L, b, a);
  // 6. the combine
  check_combine(a, resid0);
  // 5. row independence: the reversed chunk
  {
    std::vector<uint16_t> xr(x.size()), rr(resid0.size());
    for (uint32_t t = 0; t < C; ++t) {
      std::copy(x.begin() + size_t(t) * kH, x.begin() + size_t(t + 1) * kH, xr.begin() + size_t(C - 1 - t) * kH);
      std::copy(resid0.begin() + size_t(t) * kH, resid0.begin() + size_t(t + 1) * kH,
                rr.begin() + size_t(C - 1 - t) * kH);
    }
    const Run rv = walk(d, k, L, b, xr, rr, C);
    size_t moved = 0, bad = 0;
    for (uint32_t t = 0; t < C; ++t) {
      const uint32_t u = C - 1 - t;
      CHECK(std::memcmp(&a.route[size_t(t) * 32], &rv.route[size_t(u) * 32], 32 * 4) == 0);
      for (uint32_t s = 0; s <= kS.top_k; ++s) {
        const uint32_t ra = row_of(a, t, s), rb = row_of(rv, u, s);
        moved += (ra % kTm) != (rb % kTm);
        bad += std::memcmp(&a.h_bf[size_t(ra) * kI], &rv.h_bf[size_t(rb) * kI], kI * 2) != 0;
        bad += std::memcmp(&a.h_i8[size_t(ra) * kI], &rv.h_i8[size_t(rb) * kI], kI * 2) != 0;
        bad += std::memcmp(&a.y[size_t(ra) * kH], &rv.y[size_t(rb) * kH], kH * 2) != 0;
      }
      bad += std::memcmp(&a.resid[size_t(t) * kH], &rv.resid[size_t(u) * kH], kH * 2) != 0;
    }
    std::printf("  5. row independence: the reversed chunk moved %zu of %u (token, slot) rows to "
                "another place in their tile; %zu rows differ\n", moved, C * kS.slots(), bad);
    CHECK(moved > 0);
    CHECK_EQ(bad, size_t(0));
  }
  // 7. replay
  {
    const Run a2 = walk(d, k, L, b, x, resid0, C);
    CHECK(a2.route == a.route && a2.hdr == a.hdr && a2.tiles == a.tiles && a2.pair_row == a.pair_row);
    CHECK(a2.h_bf == a.h_bf && a2.h_i8 == a.h_i8 && a2.y == a.y && a2.resid == a.resid);
    std::puts("  7. replay: every buffer bitwise");
  }
  // 8. a ragged chunk
  {
    const uint32_t Cr = 37;
    const std::vector<uint16_t> xs(x.begin(), x.begin() + size_t(Cr) * kH);
    const std::vector<uint16_t> r0(resid0.begin(), resid0.begin() + size_t(Cr) * kH);
    const Run c = walk(d, k, L, b, xs, r0, Cr);
    check_sort(c.route, c);
    std::printf("  8. ragged C = %u: %u tiles\n", Cr, c.hdr[kernels::pf_moe::kHdrTiles]);
    for (uint32_t e : sample_experts(c)) grouped_vs_dense(d, k, L, b, c, e);
    fused_vs_grouped(d, L, b, c);
    w4a8_check(d, L, b, c);
    check_combine(c, r0);
    // the same 37 tokens inside the full chunk: every (token, slot) result bitwise
    size_t bad = 0;
    for (uint32_t t = 0; t < Cr; ++t)
      for (uint32_t s = 0; s <= kS.top_k; ++s) {
        bad += std::memcmp(&a.h_i8[size_t(row_of(a, t, s)) * kI], &c.h_i8[size_t(row_of(c, t, s)) * kI], kI * 2) != 0;
        bad += std::memcmp(&a.y[size_t(row_of(a, t, s)) * kH], &c.y[size_t(row_of(c, t, s)) * kH], kH * 2) != 0;
      }
    CHECK_EQ(bad, size_t(0));
    CHECK(std::memcmp(a.resid.data(), c.resid.data(), size_t(Cr) * kH * 2) == 0);
    std::puts("     the 37 tokens as their own chunk and inside the 2048-row one: bitwise equal");
  }
  std::puts("pf_moe_test OK");
  return 0;
}
