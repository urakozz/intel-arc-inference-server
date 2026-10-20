#pragma once
#include <cstddef>
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/lfm2moe_loader.h"
#include "model/lfm2moe.h"
#include "runtime/lfm2moe/lfm2moe_buffers.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

// Spec 24d: LFM2.5-24B-A2B's prefill - one chunk of at most kPfC positions through one device's layers on the Level
// Zero list (spec 24 §4.1, §4.2, §4.7; Kolibri's arrangement, runtime/kolibri/kolibri_prefill.h), no host wait inside
// it. The walk, per chunk on a device (lfm2moe_sizes.h counts it):
//
//   device 0       lfm_pf_embed_gather (the chunk's ids -> h0)                                                       1
//   conv layer     lfm_res_norm (M kPfC: _NF, or _S1 after a dense layer) · in_proj: 6 x (pf_dequant_slab +
//                  pf_gemm_T0) · lfm_pf_conv_seed · lfm_pf_conv · out_proj 2 x 2 · lfm_res_norm _S1                 20
//   attention      lfm_res_norm _NF · q||k||v 3 x 2 · lfm_attn_prep (M kPfC, S 1: norms, RoPE, KV rows pos ..) ·
//                  lfm_pf_attn (flash or B70_LFM_ATTN=eager, causal over the cache) · o_proj 2 x 2 · lfm_res_norm _S1 14
//   MoE FFN        lfm_router (M kPfC) · lfm_route (grid (1, C)) · lfm_pf_sort · lfm_pf_gather · 2 x
//                  (lfm_pf_dequant_gu + pf_moe_gemm SiLU) · 1 x (lfm_pf_dequant_dn + pf_moe_gemm) · lfm_pf_combine
//                  (rank order, one rounding, + the residual fold)                                                 + 11
//   dense FFN      gate||up 23 x (pf_dequant_slab + pf_gemm_T0_SILU) · down 2 x 2 (the next norm folds it)         + 50
//
// 1259 launches per chunk at 40 layers, the same at every C and on one card or two (the cut is 16b's: device 0 ends
// after layer s-1 - its h rows, and at a dense cut the down's pending rows, cross: the engine's copies, outside this
// walk - and device 1 starts with layer s's norm). The tail (prefill_head) is decode's binaries over the last row: 4.
//
// **Each linear follows its own layout** (24b review I1): the slab binary is pf_dequant_slab at the weight's
// {K, N, layout} - model::Lfm2MoeDesc::linears' (layout 1 for in_proj, out_proj, o_proj and the dense down, 0 for
// q||k||v and the dense gate||up today), the same source 24c's decode GEMVs read; never an assumed layout 1.
// **The rounding chain**, everywhere a decode kernel's twin exists, IS that kernel's: the norms, the router, the route
// and the attention prep are 24c's binaries at M = kPfC; the conv is lfm_conv's chain per row (lfm_pf_conv); the
// combine is lfm_moe_down's epilogue over sorted rows (lfm_pf_combine). What differs from decode is where the linears'
// sums are formed (DPAS over bf16-dequantised weights against the int4 GEMVs' order) and the attention's softmax walk
// (64-key flash tiles against decode's blocks) - so prefill's conv rings, KV, routes and h are decode's to a cosine
// bar and near-ties, not bitwise (plan 24d Review Focus 2).
// **The conv ring** is carried: lfm_pf_conv_seed reads the ring slots of p0 - 2 and p0 - 1 in a launch BEFORE the
// conv, and lfm_pf_conv writes only the chunk's last two Bx; decode continues from the ring as after decode steps.
namespace runtime::lfm2moe {

// Every allocation runtime::lfm2moe::prefill_sizes names, exactly those (allocated by the first prefill: a decode-only
// Lfm2MoeEngine holds none). One per device.
struct Lfm2MoePrefillScratch {
  Lfm2MoePrefillScratch(l0::Context& ctx, const model::Lfm2MoeDesc& d);
  l0::Mem ids;   // host memory: the chunk's ids, written by the host before the chunk (device 0 reads it)
  l0::Mem h0, h1, x, y, partials, slab, seed, attn_q, logits, routes, hdr, tiles, row_tok, pair_row, xg, moe_h, s_mlp, w;
  const model::Lfm2MoeDesc& desc;
  size_t bytes() const;
  l0::Mem& h(int i) { return i ? h1 : h0; }
};

// Where a device's walk leaves the residual: h[cur] holds every row, and `pending` (0 or 1) says whether a dense
// down's rows still wait in `partials` for the next norm to fold (the walk's input on device 1 of two: cur 0 - the
// landing goes to h0 - and the cut's pending; on device 0: the gather's h0, nothing pending).
struct PrefillResid {
  int cur = 0;
  uint32_t pending = 0;
  bool operator==(const PrefillResid& o) const { return cur == o.cur && pending == o.pending; }
};
// Both are functions of the placement alone (the same layers on every chunk), so a replayed recording needs no walk to
// know them: device `dev`'s input (device 0: h0, nothing pending - the gather; device 1: h0 - the landing - and the
// cut's pending down rows when layer s - 1 is dense) and its output (every fold flips cur: the operator norm when a
// dense layer's rows are pending, every ffn_norm; a dense layer leaves its down's rows pending).
PrefillResid prefill_resid_in(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev);
PrefillResid prefill_resid_out(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev);

// ONE chunk at absolute position `pos`, C rows, through device `part.device`'s layers. The caller has uploaded the ids
// into device 0's s.ids, set every device's Control::{pos = pos, n_active = C} (lfm_attn_prep and the conv read both),
// and - device 1 of two - landed device 0's h rows in s.h0 (and the pending down rows in s.partials) first. Appends
// exactly prefill_device_launches(d, b.placement, part.device) launches (and, with `tap`, one copy a layer - never
// under a recording); waits only when B70_PREFILL_PROFILE=1. `eager`: B70_LFM_ATTN=eager's attention build;
// `conv_tiles`: in_proj in the conv-tile column order (the arm conv's load). `tap` (debug, the last chunk only): row
// C - 1 of each layer's output into tap row l (decode's tap layout, [layers][hidden]). Leaves: the K / V rows
// [pos, pos + C) of the device's attention layers, the conv rings' slots of the chunk's last two positions, the h
// rows after its last layer (the returned PrefillResid), its MoE layers' route rows of this chunk in s.routes.
PrefillResid prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, Lfm2MoePrefillScratch& s,
                           const loader::LfmDevicePart& part, const model::Lfm2MoeDesc& d, Lfm2MoeBuffers& b,
                           uint32_t pos, uint32_t C, bool eager, bool conv_tiles, PrefillResid in, l0::Mem* tap);

// The tail only the LAST chunk runs, on the last device: decode's lfm_res_norm (M 1: _NF, or _S1 folding a dense last
// layer's pending row) over row `last_row` with embedding_norm into the decode buffers' x, the head (int8 or bf16
// tiles), the two argmax stages - decode's binaries, so the first generated id is chosen exactly as every later one.
// The caller sets the device's Control::{pos = base + L - 1, n_active = 1}; argmax_stage2 leaves pos = base + L and
// the id in cur_token[0]. `tap`: a dense last layer's folded row into tap row layers - 1.
void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, Lfm2MoePrefillScratch& s,
                  const loader::LfmDevicePart& part, const model::Lfm2MoeDesc& d, Lfm2MoeBuffers& b, uint32_t last_row,
                  PrefillResid r, l0::Mem* tap);

}  // namespace runtime::lfm2moe
