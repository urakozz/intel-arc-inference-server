#pragma once
#include <cstddef>
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/bolt_loader.h"
#include "model/bolt.h"
#include "runtime/bolt/bolt_buffers.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

// Spec 25d: Nemotron 3.5 Bolt's prefill - one chunk of at most kPfC positions through one device's blocks on the Level Zero
// list (spec 25 §6.2; the Kolibri arrangement, runtime/kolibri/kolibri_prefill.h), no host wait inside it. The walk,
// per chunk on a device (bolt_sizes.h counts it: prefill_device_launches):
//
//   device 0     nb_pf_embed_gather (the chunk's ids -> h0)                                                           1
//   every block  nb_res_norm at M 2048 (_NF over the gathered rows; _S1 after a Mamba / attention block; _S1_T1 after a
//                MoE block - the next site folds a block's GEMM rows, S 1: every prefill linear is one unsplit GEMM)
//   Mamba        in_proj 19 x (k2_pf_dequant_slab + pf_gemm_T0) into pa (pitch 18688) · nb_pf_seed · nb_pf_conv ·
//                nb_pf_dt · nb_pf_cb (DPAS) · nb_pf_ssd (the state read and written in the decode buffers) · nb_pf_gnorm ·
//                out_proj 4 x 2                                                                                      53
//   attention    q||k||v 5 x 2 · nb_attn_prep (M 2048: K / V rows pos .. pos + C - 1) · mg_pf_flash_attn (flash or
//                B70_BOLT_ATTN=eager's build) · o_proj 4 x 2                                                         21
//   MoE          nb_router · nb_route (M 2048, the chunk's C REAL rows) · fc1||shared_up 7 x 2 · nb_sh_act · nb_pf_sort ·
//                nb_pf_gather · per weight batch (nb_pf_dequant_up + nb_pf_moe_gemm relu2) · per batch (nb_pf_dequant_dn
//                + nb_pf_moe_gemm) · nb_pf_combine · fc2 4 x 2 (into pa) · shared_down 4 x 2 (into pb)   37 + 4 x batches
//
// The residual ping-pongs between h0 / h1 (every fold flips it), and where each device's walk takes and leaves it
// (prefill_resid_in / _out: a function of the placement alone) is asserted, so a replayed recording needs no walk to know
// it. **The rounding chain**, everywhere a decode kernel's twin exists, IS that kernel's: the norms, the router, the route,
// the shared act and the attention prep are 25c's binaries at M = 2048; the conv is decode's conv_step per row; the
// combine is nb_moe_down's epilogue. What differs from decode is where the linears' sums are formed (DPAS over bf16-
// dequantised weights against the int4 GEMVs' order), the SSM's evaluation (the SSD form over absolute 64-row sub-chunks
// against the per-row recurrence) and the attention's softmax walk - so prefill's states, rings, KV, routes and h are
// decode's to a cosine bar, not bitwise (spec 25 §6.2), while a prompt cut at multiples of 64 is bitwise one chunk
// (Review Focus 1: every kernel is row-local or keyed to absolute positions - the GEMM rows, the conv's per-row chain with
// its seed, the 64-aligned sub-chunks, the sorted MoE rows, the flash tiles from key 0).
// **The ring and the state** are carried: nb_pf_seed reads the ring's three columns before the chunk in a launch BEFORE the
// conv, nb_pf_conv writes back only the chunk's last min(C, 3) raw columns; nb_pf_ssd reads the state at the chunk's first
// row and writes the state after its last (fp32, or Decision 2's bf16 arm rounded once at the chunk's end) - decode then
// continues from both exactly as after decode steps.
namespace runtime::bolt {

// Every allocation runtime::bolt::prefill_sizes names, exactly those (made by the first prefill: a decode-only BoltEngine
// holds none). One per device.
struct BoltPrefillScratch {
  BoltPrefillScratch(l0::Context& ctx, const model::BoltDesc& d);
  l0::Mem ids;   // host memory: the chunk's ids, written by the host before the chunk (device 0 reads it)
  l0::Mem h0, h1, x, pa, pb, slab, seed, xc, dt, cum, cb, y, g, attn_q, logits, routes, lat, sh, r, hdr, tiles, row_tok,
      pair_row, xg, moe_h, w;
  const model::BoltDesc& desc;
  size_t bytes() const;
  l0::Mem& h(int i) { return i ? h1 : h0; }
};

// Where a device's walk takes / leaves the residual: h[cur] holds every row, `pending` the GEMM rows waiting for the next
// norm site to fold - 0 (nothing: the gather), 1 (pa: out_proj's / o_proj's) or 2 (pa and pb: a MoE block's fc2 and
// shared_down).
struct PrefillResid {
  int cur = 0;
  uint32_t pending = 0;
  bool operator==(const PrefillResid& o) const { return cur == o.cur && pending == o.pending; }
};
PrefillResid prefill_resid_in(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev);
PrefillResid prefill_resid_out(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev);

// ONE chunk at absolute position `pos`, C rows, through device `part.device`'s blocks. The caller has uploaded the ids into
// device 0's s.ids, set every device's Control::{pos = pos, n_active = C} (nb_attn_prep reads both) and - device 1 of two -
// landed device 0's h rows in s.h0 and the pending GEMM rows in s.pa (and s.pb) first. Appends exactly
// prefill_device_launches(d, b.placement, part.device) launches (and, with `tap`, one copy a block - never under a
// recording); waits only when B70_PREFILL_PROFILE=1. `eager`: B70_BOLT_ATTN=eager's attention build. `tap` (debug, the
// last chunk only): row C - 1 of each block's output into tap row l. Leaves: the K / V rows [pos, pos + C) of the device's
// attention blocks, the rings' slots of the chunk's last three positions, the SSM states after its last row, the h rows
// after its last block (the returned PrefillResid), its MoE blocks' route rows of this chunk in s.routes.
PrefillResid prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, BoltPrefillScratch& s,
                           const loader::BoltDevicePart& part, const model::BoltDesc& d, BoltBuffers& b, uint32_t pos,
                           uint32_t C, bool eager, PrefillResid in, l0::Mem* tap);

// The tail only the LAST chunk runs, on the last device: decode's nb_res_norm (M 1: _S1 / _S1_T1 folding the last block's
// GEMM rows of row `last_row`) with norm_f into the decode buffers' h1 / x, the head (int8 or bf16), the two argmax stages -
// decode's binaries, so the first generated id is chosen exactly as every later one. The caller sets the device's
// Control::{pos = base + L - 1, n_active = 1}; argmax_stage2 leaves pos = base + L and the id in cur_token[0]. `tap`: the
// folded row into tap row layers - 1.
void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, BoltPrefillScratch& s, const loader::BoltDevicePart& part,
                  const model::BoltDesc& d, BoltBuffers& b, uint32_t last_row, PrefillResid r, bool int8_head, l0::Mem* tap);

}  // namespace runtime::bolt
