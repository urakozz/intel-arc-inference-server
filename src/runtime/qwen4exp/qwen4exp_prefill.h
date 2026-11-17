#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"
#include "l0/sync_event.h"
#include "loader/qwen4exp_loader.h"
#include "model/qwen4exp.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/qwen4exp/qwen4exp_buffers.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

// Spec 21-QFN-v1d: Qwen3.8-Flash-Next's prefill - one chunk of at most kPfC positions through one device's layers on the
// Level Zero list (spec 21-QFN-v1 §4.2 "Prefill"; Kolibri's 20d arrangement, runtime/kolibri/kolibri_prefill.h), no host
// wait inside it. The walk, per chunk on a device (qwen4exp_sizes.h counts it):
//
//   device 0       pf_embed_gather (the chunk's ids; layer 0's combine _E repeats the rows into the 4 streams)
//   a gated res.   q4_hc_combine_norm (M = C: folds what is pending - _E / _S1 / _Y / _X) · the down||inject bf16 slab
//                  {10240, 336} (one 512-column slab) + pf_gemm · q4_hc_up_mix _D512 (M = C)
//   GDN layer      HC · qkv||z: 16 slabs (pf_dequant_slab int4 layout 0 | q4_pf_bf16_slab) + pf_gemm · a||b
//                  pf_ab_proj (hidden 2560) · gdn_chunk_q4 (Qwen3.8's ten binaries, the tenth pf_gated_head_SIG) ·
//                  out_proj 3 slabs (pf_int4_slab_dequant | q4_pf_bf16_slab) + pf_gemm · HC · the MoE
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
// to a cosine bar, its routes and selections decode's except near-ties (plan 21-QFN-v1d Review Focus 5).
namespace runtime::qwen4exp {

// Every allocation runtime::qwen4exp::prefill_sizes names, exactly those (allocated by the first prefill: a
// decode-only engine holds none). One per device.
struct Qwen4ExpPrefillScratch {
  Qwen4ExpPrefillScratch(l0::Context& ctx, const model::Qwen4ExpDesc& d, uint32_t max_len);
  l0::Mem ids;   // host memory: the chunk's ids (the embedding on device 0, the PLE gather and ring on its device)
  l0::Mem H, xn, x, down_f32, inj, partials, slab, idx_f32, idx_q, scores, lists, diag, q16, o, attn_out, ab, gdn, logits,
      routes, hdr, tiles, row_tok, pair_row, xg, h, y, w, ple_ids, cnt, items;
  const model::Qwen4ExpDesc& desc;
  uint32_t max_len;
  size_t bytes() const;
};

// Spec 21-QFN-v1e: the MTP head's pass over a chunk (plan 21-QFN-v1e Review Focus 4: the head's KV is filled for every position the
// main model has), on the last device after its layers: combine _Y_NN over the chunk (its rows of R, the main model's
// pre-mixer H, materialised), R into `R` rows 1..C (row 0 is R_{pos-1}: the head's hh row 0, copied in first), then
// the head on (R_q, t_{q+1}) for the head's rows - positions pos - 1 .. pos + C - 2 (pos 0: 0 .. C - 2, C - 1 rows) -
// under the head's Control (the caller sets pos / n_active = mtp_prefill_rows): q4_mtp _PF (norm), fc_embedding /
// fc_hidden as bf16 slabs + pf_gemm, q4_mtp_fuse, the attn side's gated residual (_X), the k||v slab of
// q||gate||k||v, the indexer slab, pf_attn_prep_q16 (the head's K / V), q4_qsa_prep _PF and q4_qsa_ring (its compressed
// keys and tail ring); last, `R` row C into hh row 0 (R_{pos+C-1}: the next chunk's row 0, the first draft's R).
struct Qwen4ExpMtpPrefill {
  Qwen4ExpMtpBuffers* bufs = nullptr;
  l0::Mem* R = nullptr;            // bf16 [kPfC + 1][10240] (mtp_prefill_R_bytes)
  const void* embed = nullptr;     // device 0's embedding table (peer-read on two cards)
  bool single = true;              // decision 4's norm form
  const Control* hctl = nullptr;   // the head's Control for this chunk (its slot: the next chunk's may be written)
  // Plan 21-QFN-v2b Task 5 (a tensor-parallel card's pass: its scratch is at the card's widths, the head's layer one card's):
  // the head's own rows for q||gate||k||v's k||v columns (fp32 [kPfC][pf_ld(13312)], also fh) and the attn prep's
  // q16 (bf16 [kPfC][24 x 256]); null: the chunk scratch's
  l0::Mem* partials = nullptr;
  l0::Mem* q16 = nullptr;
};

// Plan 21v2b Task 4: a tensor-parallel card's chunk (`--parallel tp`). The walk is one card's over every layer at the
// card's shapes (model::q4_tp_card: its halves of the GDN and QSA heads, of every expert, its K half of out_proj /
// o_proj), with the replicated parts (the hyper-connections, the router and route, the indexer and selection, the PLE)
// run on both cards on the same bits - the selection split by rows between them (`rctl` below). After each
// row-parallel unit - out_proj / o_proj (the GEMM over the card's K
// half), and the MoE (the down over the card's K half of every expert, fp32 rows: pf_moe_q4gemm OUT_F32, then
// q4_pf_moe_combine_f32 - the card's fp32 partial of the block's output) - the card's fp32 partial [C][2560] goes into
// slice `rank` of the unit's fold slot (slot u & 1, u counting the units from 0 each chunk), the card copies it into the
// same slice of the peer's slot, signals sent(rank, u) and waits sent(1 - rank, u); the next combine folds both slices,
// card 0's first, with one rounding (q4_hc_combine_norm _S2). Both cards fold the same two rows, so both hold the same
// H. Two slots: the peer's copy of unit u can land only after its own unit u, which waited for this card's copy of unit
// u - 1, which follows this card's fold of unit u - 2 (the slot's last read) on its in-order list. Each card gathers
// the embedding rows itself (card 1 from card 0's table over peer access) and the PLE rows through its own pointer
// table. Not bitwise pp's prefill: the K halves are summed apart and the MoE's combine runs in fp32 (judged, as
// prefill against decode is).
struct Q4TpPrefill {
  uint32_t rank = 0;
  const model::Qwen4ExpDesc* card = nullptr;   // model::q4_tp_card(m.desc, rank)
  const void* embed = nullptr;                 // the embedding table this card gathers from
  const void* ple_ptrs = nullptr;              // this card's PLE pointer table
  float* fold[2] = {nullptr, nullptr};         // fp32 [2][kPfC][hidden] a slot (slice r: card r's partial)
  float* peer_fold[2] = {nullptr, nullptr};    // the peer's slots (this card writes their slice `rank`)
  float* yf = nullptr;                         // fp32 [tmax x 32][hidden]: the down GEMM's rows
  const l0::SyncEventPool* ev = nullptr;       // sent(card, u) = ev[ev_off + card x units + u]
  uint32_t units = 0;                          // 2 x layers
  // Plan 21-QFN-v2b Task 4 Step 2, two chunks in flight (prefill_pair_tp): this chunk's events from ev_off (sent as
  // above, ready(card, u) = ev[ev_off + (2 + card) x units + u]); its residual rows and inject weights when not the
  // chunk scratch's (the pair's second chunk: bf16 [kPfC][10240], fp32 [kPfC][4]); `bcs` the card's copy-engine list
  // - the unit's slice is copied there after ready(rank, u), beside the next unit's compute (null: on the compute list)
  uint32_t ev_off = 0;
  void* H = nullptr;
  void* inj = nullptr;
  ze_command_list_handle_t bcs = nullptr;
  // The indexer split by rows (tp_split_rows(pos, C)). Both cards ran the scoring and selection of every row; here
  // each runs it for its rows - card 0 [0, tp_rows_cut(C)), card 1 the rest - under `rctl` (host: the chunk's Control
  // with pos = its first row's, n_active = its rows), after the indexer's keys and queries of every row. Then it
  // copies its selection rows into the same rows of the peer's (on `bcs` after lists_ready(rank, u), signalling
  // lists_sent(rank, u): ev[ev_off + (6 + card) x units + u] and ev[ev_off + (4 + card) x units + u]), runs
  // q||gate||k||v and the attention prep beside the copy, and - in a second part, once the peer's copy is appended
  // (prefill_pair_tp) - waits for the peer's rows before the attention. Selection rows: the scratch's per-layer rows,
  // or (`lists`, the pair's second chunk) one layer's [kPfC][kListRow] that every QSA layer reuses: the peer copies
  // into them only after its wait for this card's crossing of a unit between, which follows this card's attention.
  Control* rctl = nullptr;
  void* lists = nullptr;
  void* peer_lists = nullptr;   // the peer's selection rows of this chunk (its scratch's lists, or its `lists`)
};

// ONE chunk at absolute position `pos`, C rows, through device `part.device`'s layers. The caller has written the ids
// into `ids` (the chunk's slot of s.ids), set `ctl` - this chunk's Control for this device, host memory, a copy of the
// device's with pos = pos, n_active = C (the HC, the prep, the indexer, the PLE kernels read both) - and - device 1 of
// two - landed device 0's H rows in s.H. The slots let the host write chunk j + 1's while chunk j runs. `injected` (spec 21-QFN-v1 F3's
// debug input): host-USM rows [qsa layers][kPfC][kListRow] the caller filled for this chunk - the sparse rows read
// their lists there and q4_qsa_score / _select are not launched. `two_cards`: device 0 ends with the materialising
// _Y_NN. Appends exactly prefill_device_launches(...) launches; waits only when B70_PREFILL_PROFILE=1. Leaves: the
// device's layers' KV rows [pos, pos + C), compressed keys and tail rings, GDN state and conv rings, the PLE rings,
// H (pending the last MoE's y) and its layers' route rows / selections of this chunk.
void prefill_chunk(prefill::Context& cx, prefill::KernelCache& kc, Qwen4ExpPrefillScratch& s, const loader::Q4LoadedModel& m,
                   const loader::Q4DevicePart& part, Qwen4ExpBuffers& b, uint32_t pos, uint32_t C, bool eager,
                   const l0::Mem* injected, bool two_cards, const Qwen4ExpMtpPrefill* mtp, const Control* ctl,
                   const uint32_t* ids);

// Plan 21v2b Task 4: a chunk on both tensor-parallel cards (Q4TpPrefill), prefill_chunk's contract per card at the
// card's shapes (each appends prefill_device_launches at its descriptor and one placement). From one host thread, a
// unit at a time, each card's wait for the other's sent(u) appended only after the signal it waits for, and neither
// list more than a unit ahead of its appends (a list blocked on a wait with thousands of commands behind it stalls the
// appends: the driver's command buffer fills). Leaves the last MoE's partials in the last unit's fold slot (crossed) on
// both cards.
struct Q4TpChunk {
  prefill::Context* cx = nullptr;
  prefill::KernelCache* kc = nullptr;
  Qwen4ExpPrefillScratch* s = nullptr;
  const loader::Q4DevicePart* part = nullptr;
  Qwen4ExpBuffers* b = nullptr;
  const Control* ctl = nullptr;
  const uint32_t* ids = nullptr;
  const Q4TpPrefill* tp = nullptr;
  const Qwen4ExpMtpPrefill* mtp = nullptr;   // Task 5: the MTP head's pass after the card's layers (card 1)
};
// Plan 21-QFN-v2b Task 4 Step 2: `n` (1 or 2) consecutive chunks in flight on both cards - Qwen3.8's TP prefill
// wavefront (plan 17c): each card appends, for unit u, chunk A's unit then chunk B's, each preceded by the wait for that
// chunk's crossing of unit u - 1; a crossing runs on the card's copy-engine list (Q4TpPrefill::bcs) after the unit's
// ready, so A's crossing of unit u rides beside B's compute of unit u and the reverse. A QSA unit under the indexer's
// row split is appended in two parts: both cards up to their selection rows' copy, then both from the wait for the
// peer's (each wait after the signal it waits for). Unit u of B follows unit u of A
// on the in-order compute list, so the state (KV, GDN, rings) of a layer is A's when B reads it, as two consecutive
// chunks on one card. Leaves each chunk as Q4TpChunk says (and the head's pass on its card, per chunk in order).
void prefill_pair_tp(const loader::Q4LoadedModel& m, const std::array<std::array<Q4TpChunk, 2>, 2>& chunks,
                     const uint32_t* pos, const uint32_t* C, uint32_t n);

// The tail only the LAST chunk runs, on the last device: decode's final mixer over row `last_row` (its H, the pending
// y and inject weights), lm_head and the two argmax stages - decode's binaries, so the first generated id is chosen
// exactly as every later one. The caller sets the device's Control::{pos = base + L - 1, n_active = 1};
// argmax_stage2 leaves pos = base + L and the id in cur_token[0]. kPrefillHeadLaunches launches. Spec 21-QFN-v1e: with the
// MTP head the chunk's rows were materialised for the head's pass, so the final mixer reads `R_last` (that row of R,
// nothing pending: combine _X) instead.
void prefill_head(prefill::Context& cx, prefill::KernelCache& kc, Qwen4ExpPrefillScratch& s, const loader::Q4LoadedModel& m,
                  const loader::Q4DevicePart& part, Qwen4ExpBuffers& b, uint32_t last_row, const void* R_last = nullptr);
// Plan 21v2b Task 4: a tensor-parallel card's tail - the last row's two partials (`fold`: the last unit's slot,
// fp32 [2][kPfC][hidden]) copied side by side into b.partials and folded by combine _S2 at M = 1 with the final mixer's
// norm, the mixer, the card's vocabulary half of lm_head and tp_argmax's two stages (the card's best id and its value
// into b.control's out_token / out_value; the host merges the two cards'). kPrefillHeadLaunches launches.
// `H` / `inj`: the last chunk's rows when not the chunk scratch's (Q4TpPrefill::H / inj). `R_last` (the MTP head's
// card): the head's pass materialised the rows and then wrote the head's input over H, so the final mixer reads that
// row of R with nothing pending (combine _X), as prefill_head's R_last.
void prefill_head_tp(prefill::Context& cx, prefill::KernelCache& kc, Qwen4ExpPrefillScratch& s, const loader::Q4LoadedModel& m,
                     const loader::Q4DevicePart& part, Qwen4ExpBuffers& b, uint32_t last_row, const float* fold,
                     uint32_t rank, const model::Qwen4ExpDesc& card, const void* H = nullptr, const void* inj = nullptr,
                     const void* R_last = nullptr);

}  // namespace runtime::qwen4exp
