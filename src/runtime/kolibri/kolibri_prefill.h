#pragma once
#include <cstddef>
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/kolibri1_loader.h"
#include "model/kolibri1.h"
#include "runtime/kolibri/kolibri_buffers.h"
#include "runtime/kolibri/kolibri_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

// Spec 20d: Kolibri-1's prefill - one chunk of at most kPfC positions through one device's layers on
// the Level Zero list (spec 20 §4; 18c's arrangement, runtime/k2/k2_prefill.h), no host wait inside it.
// The walk, per chunk on a device (kolibri_sizes.h counts it):
//
//   device 0     pf_embed_gather (the chunk's ids), pf_res_fold SP0 (Σ resid² for layer 0's norm)       2
//   every layer  kol_norm_finish (input_layernorm, M kPfC) · q||k||v: 7 slabs x (k2_pf_dequant_slab |
//                kol_pf_bf16_slab, pf_gemm_T0) · kol_attn_prep (q/k head norm, RoPE in sliding layers;
//                every row's k / v into the ring or the full cache) · kol_pf_flash_attn (window 513 over
//                the ring / causal over the full cache) · o_proj: 3 slabs x 2 · pf_res_fold _Z (o_proj's
//                own row a, Σa²) + kol_post_add (post_attn_norm, resid +=, Σ resid²) · kol_norm_finish
//                (post_attention_layernorm) · the router (pf_ab_proj, fp32 logits) · kol_route (decode's
//                binary, grid (1, C)) · kol_pf_sort · kol_pf_gather · gate||up: 4 weight batches x
//                (kol_pf_dequant_gu + pf_moe_gemm SiLU) · down: 2 x (kol_pf_dequant_dn + pf_moe_gemm)
//                · kol_pf_moe_combine (into mo) · pf_res_fold SP0 (Σmo²) + kol_post_add (post_ffn_norm) 45
//
// 2252 launches per chunk at 50 layers, the same at every C and on one card or two (the cut is 16b's:
// device 0 ends with layer s-1's kol_post_add, whose rows and sums cross - the engine's copies, outside
// this walk - and device 1 starts with layer s's norm). The tail (prefill_head) is decode's binaries over
// the last row: 5.
//
// **The rounding chain**, everywhere a decode kernel's twin exists, IS that kernel's: the norms, the
// sandwich and the attention prep are kol_prep.cl at M = kPfC; the route is decode's binary (its logits
// are the router GEMV's at decode's {16, 16} tiling: row m bitwise decode's); the combine is
// kol_moe_down's epilogue over sorted rows. What differs from decode is where the GEMMs' sums are formed
// (DPAS over bf16-dequantised weights against the int4 GEMVs' split-K order) and the attention's softmax
// walk (64-key flash tiles against decode's blocks) - so prefill's KV and routes are decode's to a
// cosine bar and near-ties, not bitwise (plan 20d Review Focus 4).
namespace runtime::kolibri {

// Every allocation runtime::kolibri::prefill_sizes names, exactly those (allocated by the first
// prefill: a decode-only KolibriEngine holds none). One per device.
struct KolibriPrefillScratch {
  KolibriPrefillScratch(l0::Context& ctx, const model::Kolibri1Desc& d);
  l0::Mem ids;   // host memory: the chunk's ids, written by the host before the chunk (device 0 reads it)
  l0::Mem resid, x, a, mo, partials, slab, sumsq_a, sumsq_r, attn_q, attn_out, logits, routes, hdr, tiles, row_tok,
      pair_row, xg, h, w;
  const model::Kolibri1Desc& desc;
  size_t bytes() const;
};

// B70_KOLIBRI_ATTN=eager selects the reference-rounding attention in prefill as in decode - one parser,
// runtime::kolibri::kolibri_attn() (kolibri_sizes.h); KolibriEngine reads it once, at construction.
bool prefill_attn_eager();

// ONE chunk at absolute position `pos`, C rows, through device `part.device`'s layers. The caller has
// uploaded the ids into device 0's s.ids, set every device's Control::{pos = pos, n_active = C}
// (kol_attn_prep reads both), and - device 1 of two - landed device 0's residual rows and sums in s.resid /
// s.sumsq_r first. Appends exactly prefill_device_launches(d, b.placement, part.device) launches; waits
// only when B70_PREFILL_PROFILE=1 (runtime/prefill/profile.h). Leaves: the K / V rows [pos, pos + C) of
// the device's layers (the ring slots (pos + t) & 4095 of a sliding one), the residual rows and their sums
// after its last layer, and its layers' route rows of this chunk in s.routes.
void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, KolibriPrefillScratch& s,
                   const loader::KolDevicePart& part, const model::Kolibri1Desc& d, KolibriBuffers& b, uint32_t pos,
                   uint32_t C, bool eager);

// The tail only the LAST chunk runs, on the last device: the final norm of row `last_row` into the decode
// buffers' x, the lm_head, the two argmax stages - decode's binaries, so the first generated id is chosen
// exactly as every later one. The caller sets the device's Control::{pos = base + L - 1, n_active = 1};
// argmax_stage2 leaves pos = base + L and the id in cur_token[0].
void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, KolibriPrefillScratch& s,
                  const loader::KolDevicePart& part, const model::Kolibri1Desc& d, KolibriBuffers& b,
                  uint32_t last_row);

}  // namespace runtime::kolibri
