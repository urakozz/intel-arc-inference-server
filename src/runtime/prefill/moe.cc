#include "runtime/prefill/moe.h"

#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "runtime/buffer_sizes.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill/int8.h"
#include "runtime/prefill/profile.h"

namespace runtime::prefill {
namespace {

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::moe_chunk: " + what);
}

template <class T>
T* at(l0::Mem& m, size_t off) {
  return reinterpret_cast<T*>(static_cast<uint8_t*>(m.ptr()) + off);
}

}  // namespace

size_t moe_chunk_launches(const model::ModelDesc& d, PrefillBackend b) {
  (void)d;
  // router GEMV + moe_route + combine, the sort's three launches, the down GEMM; gate||up: the gather and one GEMM
  // (l0), or the W4A8 quantiser, gather and GEMM (l0-int8).
  return 3 + 3 + 1 + (b == PrefillBackend::L0Int8 ? 3 : 2);
}

void moe_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::MoeLayer& w,
               uint32_t layer, uint32_t C, PrefillBackend backend, Int8State* q, const void* next_norm) {
  const model::ModelDesc& d = s.desc();
  require(d.is_moe() && s.moe != nullptr, "the prefill scratch has no MoE region (a dense model?)");
  require((q != nullptr) == (backend == PrefillBackend::L0Int8),
          "the int8 state must be given iff the backend is l0-int8");
  require(C > 0 && C <= s.chunk_cap(), "C = " + std::to_string(C) + " is outside (0, " + std::to_string(s.chunk_cap()) + "], the scratch's chunk cap");
  require(layer < d.layers, "layer " + std::to_string(layer) + " is past the model's layers");
  const model::MoeDesc& m = d.moe;
  const MoePrefillLayout L = moe_prefill_layout(d, s.chunk_cap());
  require(s.moe->size() == L.total, "the MoE scratch is not moe_prefill_layout's size");
  const uint32_t H = d.hidden, I = m.expert_intermediate, E = m.experts;
  const uint32_t NG = 2 * I;                  // one block's gate||up columns
  const uint32_t tmax = moe_prefill_tiles(d, C);
  const uint32_t rows = tmax * kernels::pf_moe::kTileM;
  require(tmax <= L.tmax, "tmax(C) exceeds the scratch's tile table");
  const uint32_t ibound = moe_prefill_items(d, C);
  require(ibound <= L.ibound, "the work items' bound at C exceeds the scratch's item table");
  uint32_t* items = at<uint32_t>(*s.moe, L.items_off);
  // one layout-1 int4 block's u32 words (136 a 16-column x 64-k tile): the shared expert is block E
  const size_t gu_words = size_t(NG / 16) * (H / 64) * 136, dn_words = size_t(H / 16) * (I / 64) * 136;
  require(NG % kernels::pf_moe::kQ4WgN == 0 && H % kernels::pf_moe::kQ4WgN == 0, "the expert shape is not whole fused-GEMM work-groups");
  require(H % kernels::pf_moe::kCombineWg == 0, "the hidden size is not whole 256-column combine groups");

  float* logits = at<float>(*s.moe, L.logits_off);
  uint32_t* route = at<uint32_t>(*s.moe, L.route_at(layer));
  uint32_t* hdr = at<uint32_t>(*s.moe, L.hdr_off);
  uint32_t* tiles = at<uint32_t>(*s.moe, L.tiles_off);
  uint32_t* row_tok = at<uint32_t>(*s.moe, L.row_tok_off);
  uint32_t* pair_row = at<uint32_t>(*s.moe, L.pair_row_off);
  uint16_t* h = at<uint16_t>(*s.moe, L.h_off);
  void* xg = at<uint8_t>(*s.moe, L.xg_off);   // the gathered A, then (down) y
  const std::string pv = kernels::pf_moe_variant(E, m.top_k, H, I);

  // 1. Routing, the decode formula op for op: the router || shared-gate GEMV over the chunk
  //    (decode's {16, 16} tiling, so each row is decode's GEMV of that row), then decode's
  //    moe_route binary - grid (1, C) - writing this layer's route rows.
  const loader::DeviceWeight& rw = w.router;
  require(rw.kind == model::WeightKind::Bf16 && rw.shape.N == m.router_n() && rw.shape.K == H,
          "the router is not bf16 [router_n][hidden]");
  // Phase 2 Task 6: pf_route.cl's tiling (16 rows x 2 column tiles a work-group), bitwise pf_ab_proj.
  cx.launch(kc(kernels::pf_route_router_variant(H, m.router_n()), "pf_gemv_rows"),
            (m.router_n() / 16 + kernels::kPfRouteTc - 1) / kernels::kPfRouteTc,
            (C + kernels::kPfRouteMt - 1) / kernels::kPfRouteMt, 1,
            {PtrArg(rw.mem.ptr()), PtrArg(s.x.ptr()), PtrArg(logits), arg_val(C)});
  cx.launch(kc(kernels::moe_variant(1, E, m.top_k, H, I), "moe_route"), 1, C, 1,
            {PtrArg(logits), PtrArg(route)});
  profile_wait(cx, Phase::kMoeRoute);
  // 2. The sort: the header, the tile table padded to tmax, the rows and the work items - pf_moe_sort_mw's
  //    histogram, scan and scatter over 64-token blocks (plan 2026-11-12 Task 2: bitwise the one-work-group sort).
  {
    const std::string sv = kernels::pf_moe_sort_mw_variant(E, m.top_k);
    uint32_t* cnt = at<uint32_t>(*s.moe, L.cnt_off);
    const uint32_t nblk = (C + 63) / 64;
    cx.launch(kc(sv, "pf_moe_hist"), nblk, 1, 1, {PtrArg(route), PtrArg(cnt), arg_val(C)});
    cx.launch(kc(sv, "pf_moe_scan"), 1, 1, 1,
              {PtrArg(cnt), PtrArg(hdr), PtrArg(tiles), PtrArg(row_tok), PtrArg(items), arg_val(C), arg_val(tmax),
               arg_val(ibound)});
    cx.launch(kc(sv, "pf_moe_scatter"), nblk, 1, 1,
              {PtrArg(route), PtrArg(cnt), PtrArg(row_tok), PtrArg(pair_row), arg_val(C)});
  }
  profile_wait(cx, Phase::kMoeSort);

  // 3. gate||up, SiLU fused -> h [rows][I], one launch over the work items, B read from the int4 blocks (plan
  //    2026-11-12 Tasks 3 and 5).
  if (backend == PrefillBackend::L0Int8) {
    // W4A8 (operator 2026-11-12, D1 revised): the chunk's rows to int8 per token and 64-k group (pf_quant_g64, no
    // rotation), gathered with their group scales, x the int4 blocks as the s4 DPAS operand, int32 a group, fp32
    // rescaled by the block's group scale x the row's.
    require(H <= q->max_k() && C <= q->chunk_cap(), "the int8 state's rows do not hold the chunk");
    float* xs_tok = at<float>(*s.moe, L.xsg_tok_off);
    float* xs_row = at<float>(*s.moe, L.xsg_off);
    const std::string wv = kernels::pf_moe_w4a8_variant(H, NG);
    cx.launch(kc(wv, "pf_quant_g64"), C, 1, 1, {PtrArg(s.x.ptr()), PtrArg(q->xq().ptr()), PtrArg(xs_tok), arg_val(C)});
    cx.launch(kc(wv, "pf_moe_gather_i8g"), rows, 1, 1,
              {PtrArg(q->xq().ptr()), PtrArg(xs_tok), PtrArg(hdr), PtrArg(row_tok), PtrArg(xg), PtrArg(xs_row)});
    profile_wait(cx, Phase::kMoeGather);
    cx.launch(kc(wv, "pf_moe_w4a8"), NG / kernels::pf_moe::kQ4WgN, ibound, 1,
              {PtrArg(items), PtrArg(xg), PtrArg(xs_row), PtrArg(w.gate_up.ptr()),
               PtrArg(w.gate_up.as<uint32_t>() + size_t(E) * gu_words), PtrArg(h)});
  } else {
    cx.launch(kc(pv, "pf_moe_gather"), rows, 1, 1,
              {PtrArg(s.x.ptr()), PtrArg(hdr), PtrArg(row_tok), PtrArg(xg)});
    profile_wait(cx, Phase::kMoeGather);
    cx.launch(kc(kernels::pf_moe_q4gemm_variant(H, NG, true), "pf_moe_q4gemm"), NG / kernels::pf_moe::kQ4WgN,
              ibound, 1,
              {PtrArg(items), PtrArg(xg), PtrArg(w.gate_up.ptr()),
               PtrArg(w.gate_up.as<uint32_t>() + size_t(E) * gu_words), PtrArg(h)});
  }
  profile_wait(cx, Phase::kMoeGemm);

  // 4. down (bf16 on both backends) -> y, rne'd, into xg's region (gate||up has consumed the gathered A).
  cx.launch(kc(kernels::pf_moe_q4gemm_variant(I, H, false), "pf_moe_q4gemm"), H / kernels::pf_moe::kQ4WgN, ibound, 1,
            {PtrArg(items), PtrArg(h), PtrArg(w.down.ptr()), PtrArg(w.down.as<uint32_t>() + size_t(E) * dn_words),
             PtrArg(xg)});
  profile_wait(cx, Phase::kMoeGemm);

  // 5. The weighted sum in fixed slot order, the shared expert by its gate, the residual
  //    fold - moe_down's epilogue over the sorted rows. Phase 3 Task 5: with the next layer's input-norm weight,
  //    one work-group a row also runs that layer's pf_res_norm (S_PREV 0) into x, bitwise the two launches.
  if (next_norm)
    cx.launch(kc(kernels::pf_moe_combine_norm_variant(E, m.top_k, H, I, PrefillScratch::kNormGroups),
                 "pf_moe_combine_norm"),
              1, C, 1,
              {PtrArg(route), PtrArg(hdr), PtrArg(pair_row), PtrArg(xg), PtrArg(s.resid.ptr()), PtrArg(next_norm),
               PtrArg(s.x.ptr()), arg_val(C)});
  else
    cx.launch(kc(pv, "pf_moe_combine"), H / kernels::pf_moe::kCombineWg, C, 1,
              {PtrArg(route), PtrArg(hdr), PtrArg(pair_row), PtrArg(xg), PtrArg(s.resid.ptr()),
               arg_val(C)});
  profile_wait(cx, Phase::kMoeCombine);
}

}  // namespace runtime::prefill
