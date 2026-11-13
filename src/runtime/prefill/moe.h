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
//   l0-int8: router GEMV (pf_route_router, entry pf_gemv_rows)   logits [C][router_n]
//            moe_route (decode's moe.cl binary, grid (1, C))      route rows, this layer's slice
//   l0:      pf_gemm on the router's B slab (tiled_gemm, phase 3 Task 2)  logits [C][512] in xg's region
//            moe_route (moe.cl at ROUTER_N 512, grid (1, C))       route rows, this layer's slice
//   pf_moe_hist, pf_moe_scan, pf_moe_scatter         header, tile table, sorted rows, work items
//   l0-int8: pf_quant_g64 (the chunk's x, once), pf_moe_gather_i8g, pf_moe_w4a8 (SiLU) -> h
//   l0:      pf_moe_gather, pf_moe_q4gemm (SiLU)                               -> h
//   pf_moe_q4gemm (down)                                                         -> y
//   pf_moe_combine                                   resid += the block's output
//   (or, `next_norm` given, pf_moe_combine_norm: that and the next layer's leading norm into s.x - phase 3 Task 5)
//
// (plan 2026-11-12: the GEMMs read the int4 blocks in place - no per-chunk weight pass.) `layer` picks the route
// rows' slice (kept per layer for R2).
void moe_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::MoeLayer& w,
               uint32_t layer, uint32_t C, PrefillBackend backend, Int8State* q,
               const void* next_norm = nullptr);

// What one moe_chunk appends (a function of the backend, never of C): 10 on l0-int8, 9 on l0.
size_t moe_chunk_launches(const model::ModelDesc& d, PrefillBackend b);

// Phase 3 Task 2 (l0): the pf_gemm B slabs of layers [first, last) - each router || shared gate and each GDN layer's
// a||b (tiled_slab) - made on cx's list before the first chunk, so no chunk makes one. `m` is the walk's model (the
// stage's: its expert blocks indexed by layer - first). Nothing on a dense model.
void make_route_slabs(Context& cx, KernelCache& kc, PrefillScratch& s, const loader::LoadedModel& m, uint32_t first,
                      uint32_t last);

}  // namespace runtime::prefill
