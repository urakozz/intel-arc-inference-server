#pragma once
#include <cstddef>
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/k2_loader.h"
#include "model/k2_horizon.h"
#include "runtime/k2/k2_buffers.h"
#include "runtime/k2/k2_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

// Spec 18c: K2-Horizon's prefill - one chunk of at most kPfC positions through the whole
// model on the Level Zero list (spec 18 §5.3), no host wait inside it. The walk, per chunk:
//
//   embed        k2_pf_embed_gather (pf_embed.cl at K2's shapes)                       1
//   dense layer  pf_res_fold + k2_norm_finish (M kPfC), q||k||gate||v slabs, k2_attn_prep
//   (0-2)        (k, v -> cache; q, gate), k2_pf_flash_attn (gate fused), o_proj slabs,
//                fold + norm, gate||up slabs (SiLU fused), down slabs                 62
//   MoVA/MoE     fold + norm, q||k||gate||v_router slabs (9 + a 256-column tail),
//   layer        MoVA: k2_route (grid (1, C)), sort, gather, value dequant + grouped GEMM,
//   (3-47)       combine (SiLU, ascending id, -> V cache at pos + t); k2_attn_prep (k ->
//                cache), flash, o_proj slabs, fold + norm; MoE: router GEMV, decode's
//                k2_route, sort, gather, gate||up (2 batches) and down (1) dequant +
//                grouped GEMM, combine (ascending id + shared + residual)              49
//
// 2392 launches per chunk on K2 at every C (runtime::k2::prefill_chunk_launches, asserted
// by the walk). The tail (prefill_head) is decode's binaries over the last row: 5.
//
// The rounding chain, everywhere a decode kernel's twin exists, IS that kernel's (the
// norm's stage B, the attention prep, the router and the routing kernel are decode's
// sources; the combines are their epilogues over sorted rows); what differs from decode is
// only where the GEMMs' sums are formed (the slab GEMM over bf16-dequantised weights in
// DPAS order against the int4 GEMV's split-K order) and the attention's softmax walk (flash
// tiles of 64 keys against decode's 32 blocks) - so prefill's KV rows are decode's to a
// cosine bar, not bitwise (plan 18c Review Focus 1).
namespace runtime::k2 {

// Every allocation runtime::k2::prefill_sizes names, exactly those (allocated on the first
// prefill: a decode-only K2Engine holds none).
struct K2PrefillScratch {
  K2PrefillScratch(l0::Context& ctx, const model::K2Desc& d);
  l0::Mem ids;   // host memory: the chunk's ids, written by the host before the chunk
  l0::Mem resid, xn, xi, partials, slab, sumsq, attn_q, attn_gate, attn_out, logits, routes, hdr,
      tiles, row_tok, pair_row, xg, h, w;
  const model::K2Desc& desc;
  size_t bytes() const;
};

// B70_K2_ATTN=eager selects the reference-rounding attention (k2_pf_attn.cl EAGER), read
// once; unset / `flash` is the default. Spec 18c's half of the switch 18b's review asked for
// (the decode half is branch k2-attn-eager's).
bool prefill_attn_eager();

// ONE chunk at absolute position `pos`, C rows. The caller has uploaded the ids into s.ids
// and set Control::{pos = pos, n_active = C} (k2_attn_prep reads both), after a wait.
// Appends exactly prefill_chunk_launches(desc) launches; waits only when
// B70_PREFILL_PROFILE=1 (runtime/prefill/profile.h). Leaves: the K / V cache rows
// [pos, pos + C) of every layer, the residual rows (resid) after the last layer, and every
// layer's route rows of this chunk in s.routes.
void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, K2PrefillScratch& s,
                   const loader::K2LoadedModel& m, K2Buffers& b, uint32_t pos, uint32_t C,
                   bool eager);

// The tail only the LAST chunk runs: the final grouped norm of row `last_row` into the decode
// buffers' x, the lm_head, the two argmax stages - decode's binaries, so the first generated
// id is chosen exactly as every later one. The caller sets Control::{pos = base + L - 1,
// n_active = 1}; argmax_stage2 leaves pos = base + L and the id in cur_token[0].
void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, K2PrefillScratch& s,
                  const loader::K2LoadedModel& m, K2Buffers& b, uint32_t last_row);

}  // namespace runtime::k2
