#pragma once
#include <cstddef>
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/qwen4exp_loader.h"
#include "model/qwen4exp.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/qwen4exp/qwen4exp_buffers.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

// Spec 21d: Qwen3.8-Flash-Next's prefill - one chunk of at most kPfC positions through one device's layers on the
// Level Zero list (spec 21 §4.2 "Prefill"; Kolibri's 20d arrangement, runtime/kolibri/kolibri_prefill.h), no host
// wait inside it. The walk, per chunk on a device (qwen4exp_sizes.h counts it):
//
//   device 0       pf_embed_gather (the chunk's ids; layer 0's combine _E repeats the rows into the 4 streams)
//   a gated res.   q4_hc_combine_norm (M = C: folds what is pending - _E / _S1 / _Y / _X) · the down||inject bf16 slab
//                  {10240, 336} (one 512-column slab) + pf_gemm · q4_hc_up_mix _D512 (M = C)
//   GDN layer      HC · qkv||z: 16 slabs (pf_dequant_slab int4 layout 0 | q4_pf_bf16_slab) + pf_gemm · a||b
//                  pf_ab_proj (hidden 2560) · gdn_chunk_q4 (Qwen3.8's ten binaries, the tenth pf_gated_head_SIG) ·
//                  out_proj 3 slabs (k2_pf_dequant_slab | q4_pf_bf16_slab) + pf_gemm · HC · the MoE
//   QSA layer      HC · q||gate||k||v 13 slabs + pf_gemm · the indexer bf16 slab {2560, 640} + pf_gemm (pitch 768) ·
//                  pf_attn_prep_q16 _Q24KV2 (q / k (1 + w) norms, partial RoPE, every row's K / V at its position) ·
//                  q4_qsa_prep _PF (M = C: the query heads, every block the chunk completes - its older raw keys from
//                  the tail ring) · q4_qsa_ring (the last 8 raw keys) · [sparse rows: q4_qsa_score + q4_qsa_select
//                  (decode's code at M = C), unless the selection is injected] · [rows <= 2050: pf_flash_attn _Q24KV2]
//                  · [rows >= 2051: q4_pf_sparse_attn over each row's own list] · pf_attn_gate (pf_attn _Q24KV2) ·
//                  o_proj 3 slabs + pf_gemm · HC · the MoE
//   the MoE        the router pf_gemv_bf16 {2560, 528} · q4_route (decode's binary on grid (1, C)) · q4_pf_sort ·
//                  q4_pf_gather · gate||up 7 weight batches x (q4_pf_dequant_gu[_shb] + pf_moe_gemm SiLU) · down 4 x
//                  (q4_pf_dequant_dn[_shb] + pf_moe_gemm) · q4_pf_moe_combine (y: the next combine folds it)
//   the PLE layer  (before its attn side) combine _Y_NN (materialise H; not when H landed) · q4_ple_gather _PF (the
//                  chunk's ids, the id ring before pos) · key||value 13 bf16 slabs + pf_gemm · q4_pf_ple_gate ·
//                  q4_pf_ple_conv · q4_pf_ple_ring; its attn side's combine is _X
//   two cards      device 0 ends with combine _Y_NN over the chunk's rows: the materialised H crosses (the engine's
//                  copies, outside this walk); device 1 starts with nothing pending (_X)
//
// **The rounding chains**, everywhere a decode kernel's twin exists, ARE that kernel's: the HC, the indexer prep /
// score / select, the route, the PLE gather and block (cut in three launches), the combine (q4_moe_down's epilogue
// over sorted rows), the GDN gate (pf_gated_head _SIG = prep_gated_head _SIG). What differs from decode is where the
// linears' sums are formed (DPAS over bf16 slabs against the GEMVs' split-K orders), the GDN's chunked recurrence
// against the step, and the attention's walks (flash tiles against decode's waves) - so prefill's state is decode's
// to a cosine bar, its routes and selections decode's except near-ties (plan 21d Review Focus 5).
namespace runtime::qwen4exp {

// Every allocation runtime::qwen4exp::prefill_sizes names, exactly those (allocated by the first prefill: a
// decode-only engine holds none). One per device.
struct Qwen4ExpPrefillScratch {
  Qwen4ExpPrefillScratch(l0::Context& ctx, const model::Qwen4ExpDesc& d, uint32_t max_len);
  l0::Mem ids;   // host memory: the chunk's ids (the embedding on device 0, the PLE gather and ring on its device)
  l0::Mem H, xn, x, down_f32, inj, partials, slab, idx_f32, idx_q, scores, lists, diag, q16, o, attn_out, ab, gdn, logits,
      routes, hdr, tiles, row_tok, pair_row, xg, h, y, w, ple_ids;
  const model::Qwen4ExpDesc& desc;
  uint32_t max_len;
  size_t bytes() const;
};

// ONE chunk at absolute position `pos`, C rows, through device `part.device`'s layers. The caller has written the ids
// into every device's s.ids, set every device's Control::{pos = pos, n_active = C} (the HC, the prep, the indexer,
// the PLE kernels read both), and - device 1 of two - landed device 0's H rows in s.H. `injected` (spec 21 F3's
// debug input): host-USM rows [qsa layers][kPfC][kListRow] the caller filled for this chunk - the sparse rows read
// their lists there and q4_qsa_score / _select are not launched. `two_cards`: device 0 ends with the materialising
// _Y_NN. Appends exactly prefill_device_launches(...) launches; waits only when B70_PREFILL_PROFILE=1. Leaves: the
// device's layers' KV rows [pos, pos + C), compressed keys and tail rings, GDN state and conv rings, the PLE rings,
// H (pending the last MoE's y) and its layers' route rows / selections of this chunk.
void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, Qwen4ExpPrefillScratch& s, const loader::Q4LoadedModel& m,
                   const loader::Q4DevicePart& part, Qwen4ExpBuffers& b, uint32_t pos, uint32_t C, bool eager,
                   const l0::Mem* injected, bool two_cards);

// The tail only the LAST chunk runs, on the last device: decode's final mixer over row `last_row` (its H, the pending
// y and inject weights), lm_head and the two argmax stages - decode's binaries, so the first generated id is chosen
// exactly as every later one. The caller sets the device's Control::{pos = base + L - 1, n_active = 1};
// argmax_stage2 leaves pos = base + L and the id in cur_token[0]. kPrefillHeadLaunches launches.
void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, Qwen4ExpPrefillScratch& s, const loader::Q4LoadedModel& m,
                  const loader::Q4DevicePart& part, Qwen4ExpBuffers& b, uint32_t last_row);

}  // namespace runtime::qwen4exp
