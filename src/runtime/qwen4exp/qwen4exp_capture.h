#pragma once
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/qwen4exp_loader.h"
#include "runtime/capture.h"   // CapturedStep, StageLink - the same records
#include "runtime/qwen4exp/qwen4exp_buffers.h"

// Spec 21-QFNc: one device's Qwen3.8-Flash-Next decode step, captured once and replayed per token - Kolibri's
// arrangement (runtime/kolibri/kolibri_capture.h: a straight walk, every per-token value read from
// runtime::Control at execution, the expert ids and the QSA selections produced and consumed on the device,
// the PLE rows read from host USM by an index the device computes) over the device's layers [first, end).
//
// **The pending combine.** The residual H (4 streams x 2560 bf16) is materialised only by the combines: a
// block's output y and its inject weights are folded into H by the NEXT q4_hc_combine_norm (vLLM's fusion), so
// the walk carries what is pending - the embedding (layer 0: _E), a mixer's GEMV slices (_S<S>), the MoE's y
// (_Y) or nothing (_X) - into the next combine; the PLE layer and device 0's end on two cards materialise it
// with a combine-only _Y_NN. One gated residual = combine_norm, the down||inject gemv_bf16, up_mix:
//
//   device 0     embed_gather (into x; layer 0's combine _E repeats it into the 4 streams)
//   GDN layer    hc(attn) [3] - qkv||z GEMV, a||b gemv_bf16, gdn_step, prep_gated_head _SIG, out_proj GEMV -
//                hc(mlp) [3] - router gemv_bf16, q4_route, q4_moe_gate_up, q4_moe_down                     15
//   QSA layer    hc(attn) [3] - q||gate||k||v GEMV, indexer gemv_bf16, attn_prep _Q24KV2, q4_qsa_prep,
//                q4_qsa_score, q4_qsa_select, q4_qsa_attn + q4_qsa_reduce (eager: q4_qsa_attn_eager), o_proj GEMV -
//                hc(mlp) [3] - the MoE's 4                                                                19 (18)
//   PLE layer    + combine_norm _Y_NN, q4_ple_gather, key||value gemv_bf16, q4_ple_block; its attn side's
//                combine is _X                                                                            + 4
//   last device  the final mixer (combine_norm _Y, gemv_bf16 {10240, 320}, up_mix), lm_head, argmax x 2       6
// 779 launches at 48 layers on one card (runtime::qwen4exp::decode_launches), asserted per device at capture.
//
// **Injected selections** (`injected` non-null; spec 21-QFN F3's debug run): every QSA layer reads its list from the
// host-USM rows [qsa layers][M][kListRow] the caller writes before each replay, and q4_qsa_score / _select are not
// in the list (2 launches fewer a QSA layer).
//
// **Two cards** (`link` non-null; spec 16b's cut and hand-offs): device 0's list ends with a combine-only _Y_NN
// (the MATERIALISED H, as vLLM's PP carries it) and hands H off: a device-to-device copy into the landing buffer
// and a barrier signalling the cross-device event. Device 1's list starts with the wait, the copy into its own H,
// then its first layer with
// nothing pending (_X; the PLE layer's prologue skips its _Y_NN when it is device 1's first layer).
//
// tap (debug): H copied to tap row l right after the combine that materialises layer l's output (the next
// layer's first combine, the PLE prologue's, device 0's last, or the final mixer's) - the reference's H.L<l>.
//
// **Spec 21-QFNe: spec 8's lists** (`spec.kind`; runtime/engine.h's draft / verify / commit contracts):
//   Verify at M = k + 1 rows, every device: the decode list at M rows (Control::cur_token[0..M) the ids, positions
//     pos .. pos + M - 1; every kernel row-independent - the verify row r IS the M = 1 step at its position, plan 21-QFNe
//     Review Focus 2) with gdn_step_slots_M<M>_G1 (row r's GDN state into slot (gdn_live + r) % 4) and, on the PLE
//     layer at M > 1, q4_pf_ple's gate / conv / ring at C = 4 for q4_ple_block; the hand-off carries M rows; on the
//     last device the final mixer, lm_head and argmax at M (out_token[0..M), pos += M), then a copy of the M
//     pre-mixer rows of H into the head's hh rows 1..M, then the head's KV pass over the M rows - positions pos - 1 ..
//     pos + M - 2 (the head's Control), inputs (hh[r], cur_token[r]) = (R_{pos-1+r}, t_{pos+r}): q4_mtp_norm,
//     fc_embedding, fc_hidden (one M = 4 GEMV a row), q4_mtp_fuse, the attn side's gated residual (_X), q||gate||k||v,
//     the indexer, attn_prep (the head's K / V) and q4_qsa_prep (its tail ring, its compressed keys) - Review Focus 4:
//     the head's KV is written for every position the main model has;
//   Draft step i (the last device, M = 1, the head's Control: position pos - 1 + i): q4_mtp_norm over R = hh row 0
//     (step 0) or the previous step's pre-mixer H (b.H, which its final mixer's combine materialised) and the head
//     Control's cur_token[0], the fc's, q4_mtp_fuse, the head's QSA layer - q4_qsa_score / _select into the head's
//     list on step 0, steps 1.. attend step 0's list as it is
//     (decision 5) - and MoE (bf16 dense and shared, int4 experts), its own final mixer (b.H = the head's pre-mixer H:
//     the next step's R), lm_head into the head's logits row i, argmax into the head's Control (out_token[0],
//     cur_token[0], pos + 1).
// The launch counts are runtime::qwen4exp's verify_device_launches / draft_launches, asserted at capture.
namespace runtime::qwen4exp {

enum class ListKind { Decode, Verify, Draft };
struct ListSpec {
  ListKind kind = ListKind::Decode;
  uint32_t M = 1;        // Verify: 1..kVerifyRows; Decode / Draft: 1
  uint32_t draft = 0;    // Draft: the step index (0 reads hh row 0, later steps the previous step's H)
  bool select = true;    // Draft: whether the step scores and selects (step 0; every step under `fresh`)
  // Spec 28d: the dense int4 projections and the int8 head bind gemm_sm at M >= 2 (runtime::gemv_kind, as Qwen3.8's
  // capture; kSmallMTuning's Flash-Next rows), gemv / gemv_i8w at M = 1. `opt.force_linear` is the tests' / probes'
  // override only (Gemv: the verify rows bitwise the plain steps, the pre-28d binding).
  CaptureOptions opt;
  // Probes only: launch i signals prof->events[i] (runtime/capture.h's ProfileEvents; observes only).
  ProfileEvents* prof = nullptr;
};
// The head's pieces the Verify / Draft lists bind (the last device).
struct MtpBinding {
  Qwen4ExpMtpBuffers* bufs = nullptr;
  const void* embed = nullptr;   // the embedding table: device 0's (read by the last device over peer access)
  bool single = true;            // decision 4: q4_mtp's _SINGLE (one RMS over 10240) or _STREAM
};

// Plan 21-QFN-v2b: a tensor-parallel card's decode list (`tp` non-null; Decode, M = 1, flash attention). The buffers are
// the card's (Qwen4ExpBuffers over `card` = model::q4_tp_card at one card's placement, device 0); the walk is one
// card's over every layer, at the card's widths, with Qwen3.8's crossing (runtime::TpLink, src/kernels/tp.cl) after
// each row-parallel linear, in the slices form (tp_send pushes the card's raw slices into the peer's landing buffer,
// tp_sum waits for the peer's): out_proj / o_proj run at split-K 2 over the card's K half - one card's slices 2r,
// 2r + 1 - and tp_sum adds card 0's two then card 1's into one row (one card's S4 chain), which the next combine folds
// as one slice (_S1); the MoE's down writes each slot's dot over the card's K half (MOE_TP_PART: one card's two DN_KS
// partials) into `slots` and sends it to the peer's landing (`moe`, tp.cl's protocol inside the down), and
// q4_moe_finish waits for the peer's, adds them slot by slot and runs the down's epilogue into y. So every
// value is one card's, bit for bit. Everything
// replicated (the hyper-connections, router and route, the indexer and selection, the PLE) runs on both cards on the
// same bits, so both cards hold the same H. `embed`: the table card 1 gathers from (card 0's, over peer access) or
// card 0's own. The head: the card's vocabulary half and tp_argmax's two stages (Control::out_token / out_value; the
// host merges). Launches: tp_device_launches.
// Task 5 Step 0, MTP under tp (`spec.kind` Verify / Draft with `mtp`): the verify list at M rows is the decode list's
// walk at M (the GDN slots build, the PLE layer's q4_pf_ple at M > 1, the crossings at M rows, the gemv linears: every
// row the M = 1 step's arithmetic), the head's vocabulary half and tp_argmax at M (the host merges each row); the head's
// card (part.mtp: card 1) adds the head's KV pass. A draft step on the head's card is spec 21-QFNe's at one card's shapes
// (its buffers' scratch is one card's: Qwen4ExpBuffers' `scratch`) up to the head's final mixer, then `draft_ready`
// signalled (its x is the step's normed row), the card's vocabulary half and tp_argmax into the head's Control. On
// the other card a draft step is the head's other half only: wait for `draft_ready`, the peer's x copied in, gemv_i8w
// and tp_argmax into `draft_ctl` (kTpDraftHeadLaunches).
struct Q4TpBinding {
  const runtime::TpLink* link = nullptr;   // out_proj / o_proj's crossing (and the head's rank, vocabulary half)
  const runtime::TpLink* moe = nullptr;    // the MoE's crossing (its own landing buffer and counters)
  const model::Qwen4ExpDesc* card = nullptr;
  const void* embed = nullptr;
  void* slots = nullptr;                   // fp32 [11][rows][2560]: the MoE's slots
  ze_event_handle_t draft_ready = nullptr; // the head's card signals, the other card waits (draft steps)
  const void* peer_x = nullptr;            // the head card's x (the other card's draft step reads it over peer access)
  void* draft_ctl = nullptr;               // the other card's draft Control (its half's winner)
};

CapturedStep build(l0::Context& ctx, const loader::Q4LoadedModel& m, const loader::Q4DevicePart& part,
                   Qwen4ExpBuffers& b, const runtime::StageLink* link, l0::Mem* tap = nullptr,
                   const l0::Mem* injected = nullptr, const ListSpec& spec = {}, const MtpBinding* mtp = nullptr,
                   const Q4TpBinding* tp = nullptr);

}  // namespace runtime::qwen4exp
