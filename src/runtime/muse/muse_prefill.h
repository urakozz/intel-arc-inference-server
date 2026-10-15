#pragma once
#include <cstddef>
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/muse_loader.h"
#include "model/muse_glimmer.h"
#include "runtime/muse/muse_buffers.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

// Spec 23d: Muse-Glimmer-30B's prefill - one chunk of at most kPfC positions through one device's layers on the Level
// Zero list (spec 23 §5.3; Kolibri's arrangement, runtime/kolibri/kolibri_prefill.h, without the MoE), no host wait
// inside it. The walk, per chunk on a device (muse_sizes.h counts it):
//
//   device 0     mg_pf_embed (the chunk's ids: the gathered row through the weight-less norm, layer 0's Σ²)         1
//   every layer  mg_norm (input_layernorm, M kPfC) · q||gate||k||v: 9 slabs x (k2_pf_dequant_slab, pf_gemm_T0) ·
//                mg_attn_prep (S 1: weight-less q / k norms, q x 3.87, RoPE on sliding layers, the gate's sigmoid;
//                every row's K / V into the ring slot p & 4095 or the NoPE cache row p) · mg_pf_flash_attn (the
//                2048-key window over the ring / causal over the full cache; flash or B70_MUSE_ATTN=eager) ·
//                mg_pf_attn_gate · o_proj 7 x 2 · pf_res_fold _Z (o_proj's own row a, Σa²) + mg_post_add
//                (post_attention_layernorm eps 1e-8, resid +=, Σ resid²) · mg_norm (pre_feedforward_layernorm) ·
//                gate||up 39 x 2 · mg_silu_mul · down 7 x 2 · pf_res_fold _Z + mg_post_add (post_feedforward)     134
//
// 6969 launches per chunk at 52 layers, the same at every C and on one card or two (the cut is 16b's: device 0 ends
// with layer s-1's mg_post_add, whose rows and sums cross - the engine's copies, outside this walk - and device 1
// starts with layer s's norm). The tail (prefill_head) is decode's head over the last row: 4.
//
// **The rounding chain**, everywhere a decode kernel's twin exists, IS that kernel's: the norms, the sandwich, the
// attention prep, SiLU x up and the gate are mg_prep.cl / decode's gate chain at M = kPfC; the eager attention is
// decode's eager row bitwise. What differs from decode is where the linears' sums are formed (DPAS over bf16-
// dequantised weights against the int4 GEMVs' split-K order) and the flash attention's softmax walk (64-key tiles
// against decode's blocks) - so prefill's KV, rings and logits are decode's to a cosine bar and near-ties, not bitwise
// (plan 23d Review Focus 5).
namespace runtime::muse {

// Every allocation runtime::muse::prefill_sizes names, exactly those (allocated by the first prefill: a decode-only
// MuseEngine holds none). One per device.
struct MusePrefillScratch {
  MusePrefillScratch(l0::Context& ctx, const model::MuseGlimmerDesc& d);
  l0::Mem ids;   // host memory: the chunk's ids, written by the host before the chunk (device 0 reads it)
  l0::Mem resid, x, a, partials, slab, sumsq_a, sumsq_r, attn_q, gate, attn_out, h;
  const model::MuseGlimmerDesc& desc;
  size_t bytes() const;
};

// ONE chunk at absolute position `pos`, C rows, through device `part.device`'s layers. The caller has uploaded the ids
// into device 0's s.ids, set every device's Control::{pos = pos, n_active = C} (mg_attn_prep reads both), and - device
// 1 of two - landed device 0's residual rows and sums in s.resid / s.sumsq_r first. Appends exactly
// prefill_device_launches(d, b.placement, part.device) launches; waits only when B70_PREFILL_PROFILE=1. Leaves: the
// K / V rows [pos, pos + C) of the device's layers (the ring slots (pos + t) & 4095 of a sliding one), the residual
// rows and their sums after its last layer. `eager`: B70_MUSE_ATTN=eager's attention build.
void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, MusePrefillScratch& s,
                   const loader::MuseDevicePart& part, const model::MuseGlimmerDesc& d, MuseBuffers& b, uint32_t pos,
                   uint32_t C, bool eager);

// The tail only the LAST chunk runs, on the last device: mg_final_norm over row `last_row` (the M kPfC norm binary at
// grid (26, 1) on that row's offsets: decode's chain) into the decode buffers' x, the lm_head, the two argmax stages -
// decode's binaries, so the first generated id is chosen exactly as every later one. The caller sets the device's
// Control::{pos = base + L - 1, n_active = 1}; argmax_stage2 leaves pos = base + L and the id in cur_token[0].
void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, MusePrefillScratch& s,
                  const loader::MuseDevicePart& part, const model::MuseGlimmerDesc& d, MuseBuffers& b, uint32_t last_row);

}  // namespace runtime::muse
