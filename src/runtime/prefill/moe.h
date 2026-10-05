#pragma once
#include <cstddef>
#include <cstdint>

#include "loader/loader.h"
#include "model/model_desc.h"
#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_backend.h"

// Spec 15d: a mixture-of-experts layer's FFN over one prefill chunk (spec 15 §4.4, §9) -
// the device routing, the grouped expert GEMMs and the combine, with no host read of any
// count. The kernels are src/kernels/prefill/pf_moe.cl and pf_moe_gemm.cl (their headers
// carry the layout and the determinism argument); the scratch is PrefillScratch::moe at
// runtime::moe_prefill_layout's offsets.
namespace runtime::prefill {

class Int8State;

// x = PrefillScratch::x, bf16 [C][hidden]: the post-attention norm's output. The block's
// output is FOLDED into PrefillScratch::resid (pf_moe_combine, as decode's moe_down), so
// the next norm folds nothing (S_PREV 0). Launches, in order:
//
//   router GEMV (pf_moe_router, entry pf_ab_proj)    logits [C][router_n]
//   moe_route (decode's moe.cl binary, grid (1, C))  route rows, this layer's slice
//   pf_moe_sort                                      header, tile table, sorted rows
//   l0-int8: pf_quant_had (the chunk's x, once), pf_moe_gather_i8, pf_requant_rot over the
//            layer's whole gate||up array, pf_moe_gemm_i8 (SiLU)               -> h
//   l0:      pf_moe_gather, then per batch of blocks pf_moe_dequant_gu +
//            pf_moe_gemm (SiLU)                                                 -> h
//   per batch of blocks pf_moe_dequant_dn + pf_moe_gemm (down)                  -> y
//   pf_moe_combine                                   resid += the block's output
//
// `layer` picks the route rows' slice (kept per layer for R2). The l0-int8 form needs the
// layer's expert scales already in `q` (moe_prepare_int8, outside any recording).
void moe_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::MoeLayer& w,
               uint32_t layer, uint32_t C, PrefillBackend backend, Int8State* q);

// What one moe_chunk appends (a function of the shape and the backend, never of C):
// Ornith 10 on l0-int8, 11 on l0.
size_t moe_chunk_launches(const model::ModelDesc& d, PrefillBackend b);

// Engine::prepare_prefill on l0-int8: every layer's expert gate||up array's rotated column
// scales (Int8State::scales_layout1, cached; one pf_colmax_rot and a host finish each).
void moe_prepare_int8(Context& cx, KernelCache& kc, Int8State& q, const loader::LoadedModel& m);

}  // namespace runtime::prefill
