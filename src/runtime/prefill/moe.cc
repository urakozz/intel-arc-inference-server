#include "runtime/prefill/moe.h"

#include <algorithm>
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

constexpr uint32_t kRotBlock = 1024;   // pf_int8.cl's Hadamard block (int8.cc)

// ceil(blocks / per-batch blocks) for one weight form.
uint32_t batches(const model::ModelDesc& d, MoeWeightForm f) {
  const uint32_t per = moe_prefill_batch_blocks(d, f);
  return (d.moe.blocks() + per - 1) / per;
}

template <class T>
T* at(l0::Mem& m, size_t off) {
  return reinterpret_cast<T*>(static_cast<uint8_t*>(m.ptr()) + off);
}

}  // namespace

size_t moe_chunk_launches(const model::ModelDesc& d, PrefillBackend b) {
  // router GEMV + moe_route + sort + combine, and the down batches' dequant + GEMM pairs.
  size_t n = 4 + 2 * size_t(batches(d, MoeWeightForm::DownBf16));
  if (b == PrefillBackend::L0Int8)
    n += 4;   // quantiser, gather_i8, requant (the whole array), the i8 GEMM
  else
    n += 1 + 2 * size_t(batches(d, MoeWeightForm::GateUpBf16));   // gather, dequant + GEMM pairs
  return n;
}

void moe_prepare_int8(Context& cx, KernelCache& kc, Int8State& q, const loader::LoadedModel& m) {
  const model::ModelDesc& d = *m.desc;
  if (!d.is_moe()) return;
  for (const loader::MoeLayer& w : m.moe)
    q.scales_layout1(cx, kc, w.gate_up.ptr(), d.hidden, d.moe.blocks() * 2 * d.moe.expert_intermediate);
}

void moe_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::MoeLayer& w,
               uint32_t layer, uint32_t C, PrefillBackend backend, Int8State* q) {
  const model::ModelDesc& d = s.desc();
  require(d.is_moe() && s.moe != nullptr, "the prefill scratch has no MoE region (a dense model?)");
  require(is_l0(backend), "a mixture-of-experts layer prefills on the L0 backends only (l0, l0-int8)");
  require((q != nullptr) == (backend == PrefillBackend::L0Int8),
          "the int8 state must be given iff the backend is l0-int8");
  require(C > 0 && C <= PrefillScratch::kC, "C = " + std::to_string(C) + " is outside (0, kC]");
  require(layer < d.layers, "layer " + std::to_string(layer) + " is past the model's layers");
  const model::MoeDesc& m = d.moe;
  const MoePrefillLayout L = moe_prefill_layout(d);
  require(s.moe->size() == L.total, "the MoE scratch is not moe_prefill_layout's size");
  const uint32_t H = d.hidden, I = m.expert_intermediate, E = m.experts, B = m.blocks();
  const uint32_t NG = 2 * I;                  // one block's gate||up columns
  const uint32_t tmax = moe_prefill_tiles(d, C);
  const uint32_t rows = tmax * kernels::pf_moe::kTileM;
  require(tmax <= L.tmax, "tmax(C) exceeds the scratch's tile table");
  require(NG % kernels::pf_moe::kGemmWgN == 0 && H % kernels::pf_moe::kGemmWgN == 0 &&
              H % kernels::pf_moe::kCombineWg == 0,
          "the expert shape is not whole 256-column work-groups");

  float* logits = at<float>(*s.moe, L.logits_off);
  uint32_t* route = at<uint32_t>(*s.moe, L.route_at(layer));
  uint32_t* hdr = at<uint32_t>(*s.moe, L.hdr_off);
  uint32_t* tiles = at<uint32_t>(*s.moe, L.tiles_off);
  uint32_t* row_tok = at<uint32_t>(*s.moe, L.row_tok_off);
  uint32_t* pair_row = at<uint32_t>(*s.moe, L.pair_row_off);
  float* xs = at<float>(*s.moe, L.xs_off);
  uint16_t* h = at<uint16_t>(*s.moe, L.h_off);
  void* xg = at<uint8_t>(*s.moe, L.xg_off);   // the gathered A, then (down) y
  void* wb = at<uint8_t>(*s.moe, L.w_off);    // the weight batch
  const std::string pv = kernels::pf_moe_variant(E, m.top_k, H, I);

  // 1. Routing, the decode formula op for op: the router || shared-gate GEMV over the chunk
  //    (decode's {16, 16} tiling, so each row is decode's GEMV of that row), then decode's
  //    moe_route binary - grid (1, C) - writing this layer's route rows.
  const loader::DeviceWeight& rw = w.router;
  require(rw.kind == model::WeightKind::Bf16 && rw.shape.N == m.router_n() && rw.shape.K == H,
          "the router is not bf16 [router_n][hidden]");
  cx.launch(kc(kernels::pf_moe_router_variant(H, m.router_n()), "pf_ab_proj"), m.router_n() / 16,
            (C + 7) / 8, 1, {PtrArg(rw.mem.ptr()), PtrArg(s.x.ptr()), PtrArg(logits), arg_val(C)});
  cx.launch(kc(kernels::moe_variant(1, E, m.top_k, H, I), "moe_route"), 1, C, 1,
            {PtrArg(logits), PtrArg(route)});
  profile_wait(cx, Phase::kMoeRoute);
  // 2. The sort: one work-group, the header, the tile table padded to tmax, the rows.
  cx.launch(kc(pv, "pf_moe_sort"), 1, 1, 1,
            {PtrArg(route), PtrArg(hdr), PtrArg(tiles), PtrArg(row_tok), PtrArg(pair_row),
             arg_val(C), arg_val(tmax)});
  profile_wait(cx, Phase::kMoeSort);

  // 3. gate||up, SiLU fused -> h [rows][I].
  if (backend == PrefillBackend::L0Int8) {
    // spec 5's h8: the chunk's x rotated and quantised ONCE per token (pf_quant_had at the
    // hidden size), the int8 rows and their scales gathered into sorted order, the layer's
    // whole gate||up array requantised into the VNNI-4 batch (one launch over every block:
    // the array is one layout-1 weight of K = hidden, N = blocks x 2 I), one grouped GEMM.
    require(H % kRotBlock == 0 && H <= q->max_k(), "hidden is not whole 1024-k rotation blocks");
    const uint32_t nall = B * NG;
    const std::pair<l0::Mem, l0::Mem>& sc = q->scales_layout1(cx, kc, w.gate_up.ptr(), H, nall);
    cx.launch(kc(kernels::pf_quant_had_variant(H), "pf_quant_had"), C, 1, 1,
              {PtrArg(s.x.ptr()), PtrArg(q->signs_f32(H).ptr()), PtrArg(q->xq().ptr()),
               PtrArg(q->xs().ptr()), arg_val(H)});
    cx.launch(kc(pv, "pf_moe_gather_i8"), rows, 1, 1,
              {PtrArg(q->xq().ptr()), PtrArg(q->xs().ptr()), PtrArg(hdr), PtrArg(row_tok),
               PtrArg(xg), PtrArg(xs)});
    profile_wait(cx, Phase::kMoeGather);
    require(moe_prefill_batch_blocks(d, MoeWeightForm::GateUpInt8) == B,
            "the weight batch does not hold every int8 gate||up block");
    cx.launch(kc(kernels::pf_requant_rot_variant(1), "pf_requant_rot"), nall / 16, H / kRotBlock, 1,
              {PtrArg(w.gate_up.ptr()), PtrArg(nullptr), PtrArg(q->sign_bits(H).ptr()),
               PtrArg(sc.second.ptr()), PtrArg(wb), arg_val(0u), arg_val(nall), arg_val(H),
               arg_val(nall)});
    profile_wait(cx, Phase::kMoeWeights);
    cx.launch(kc(kernels::pf_moe_gemm_variant(H, NG, true, true), "pf_moe_gemm"), tmax,
              NG / kernels::pf_moe::kGemmWgN, 1,
              {PtrArg(tiles), PtrArg(xg), PtrArg(xs), PtrArg(wb), PtrArg(sc.first.ptr()), PtrArg(h),
               arg_val(0u), arg_val(B), arg_val(nall)});
    profile_wait(cx, Phase::kMoeGemm);
  } else {
    cx.launch(kc(pv, "pf_moe_gather"), rows, 1, 1,
              {PtrArg(s.x.ptr()), PtrArg(hdr), PtrArg(row_tok), PtrArg(xg)});
    profile_wait(cx, Phase::kMoeGather);
    const uint32_t per = moe_prefill_batch_blocks(d, MoeWeightForm::GateUpBf16);
    l0::Kernel& gemm = kc(kernels::pf_moe_gemm_variant(H, NG, false, true), "pf_moe_gemm");
    for (uint32_t b0 = 0; b0 < B; b0 += per) {
      const uint32_t nb = std::min(per, B - b0);
      cx.launch(kc(pv, "pf_moe_dequant_gu"), NG / 16, H / 64, nb,
                {PtrArg(w.gate_up.ptr()), PtrArg(hdr), PtrArg(wb), arg_val(b0)});
      profile_wait(cx, Phase::kMoeWeights);
      cx.launch(gemm, tmax, NG / kernels::pf_moe::kGemmWgN, 1,
                {PtrArg(tiles), PtrArg(xg), PtrArg(nullptr), PtrArg(wb), PtrArg(nullptr), PtrArg(h),
                 arg_val(b0), arg_val(b0 + nb), arg_val(0u)});
      profile_wait(cx, Phase::kMoeGemm);
    }
  }

  // 4. down (bf16 on both backends: its K = I is not whole 1024-k rotation blocks) -> y,
  //    rne'd, into xg's region (gate||up has consumed the gathered A).
  {
    const uint32_t per = moe_prefill_batch_blocks(d, MoeWeightForm::DownBf16);
    l0::Kernel& gemm = kc(kernels::pf_moe_gemm_variant(I, H, false, false), "pf_moe_gemm");
    for (uint32_t b0 = 0; b0 < B; b0 += per) {
      const uint32_t nb = std::min(per, B - b0);
      cx.launch(kc(pv, "pf_moe_dequant_dn"), H / 16, I / 64, nb,
                {PtrArg(w.down.ptr()), PtrArg(hdr), PtrArg(wb), arg_val(b0)});
      profile_wait(cx, Phase::kMoeWeights);
      cx.launch(gemm, tmax, H / kernels::pf_moe::kGemmWgN, 1,
                {PtrArg(tiles), PtrArg(h), PtrArg(nullptr), PtrArg(wb), PtrArg(nullptr), PtrArg(xg),
                 arg_val(b0), arg_val(b0 + nb), arg_val(0u)});
      profile_wait(cx, Phase::kMoeGemm);
    }
  }

  // 5. The weighted sum in fixed slot order, the shared expert by its gate, the residual
  //    fold - moe_down's epilogue over the sorted rows.
  cx.launch(kc(pv, "pf_moe_combine"), H / kernels::pf_moe::kCombineWg, C, 1,
            {PtrArg(route), PtrArg(hdr), PtrArg(pair_row), PtrArg(xg), PtrArg(s.resid.ptr()),
             arg_val(C)});
  profile_wait(cx, Phase::kMoeCombine);
}

}  // namespace runtime::prefill
