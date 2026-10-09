#pragma once
#include <cstdint>
#include <memory>

#include "l0/context.h"
#include "l0/memory.h"
#include "model/qwen4exp.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace l0 {
class CmdList;
}

// Spec 21c: one device's Qwen3.8-Flash-Next decode allocations (runtime/qwen4exp/qwen4exp_sizes.h sizes every
// one; the constructor allocates exactly those): the persistent group of the device's layers - the QSA layers'
// KV, compressed indexer keys and tail rings, the GDN layers' state and conv rings, the PLE layer's rings - and
// the whole decode scratch. Kolibri's KolibriBuffers arrangement.
//
// Spec 21e (`mtp`): the scratch holds kVerifyRows rows (the verify lists run at M = k + 1 <= 4; the per-layer rows
// - routes, selections, diagnostics - are strided by `rows`), gdn_spec holds the verify's state slots 1..3 of every
// GDN layer of the device (LAYER-major: layer i's slot s at gdn_spec + (3 i + s - 1) x one state - the stride
// gdn_step_slots_M<M>_G1 bakes), and the PLE layer's device holds the verify's PLE rows (q4_pf_ple at C = 4). The
// head's own buffers are Qwen4ExpMtpBuffers (the last device).
namespace runtime::qwen4exp {

struct Qwen4ExpBuffers {
  Qwen4ExpBuffers(l0::Context& ctx, const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t device,
                  uint32_t max_len, Q4Attn attn = q4_attn(), bool mtp = false);

  // --- persistent: zeroed by zero() (Qwen4ExpEngine::reset) -------------------------------------------------
  l0::Mem control;      // runtime::Control, shared memory
  l0::Mem kv;           // bf16 [its QSA layers][K | V][max_len][2][256]
  l0::Mem idx_keys;     // bf16 [its QSA layers][max_len / 4][128]
  l0::Mem idx_tail;     // bf16 [its QSA layers][8][128]
  l0::Mem gdn_state;    // fp32 [its GDN layers][48][128][128]
  l0::Mem conv_ring;    // bf16 [its GDN layers][16][10240]
  l0::Mem ple;          // the PLE layer's device: u32 [16] id ring at 0, bf16 [16][10240] conv ring at kPleConvOff
  // --- decode scratch: never zeroed (no step reads scratch it has not written first; the hash constants are
  //     written once at construction) -------------------------------------------------------------------------
  l0::Mem H, xn, x, partials, down_f32, inj, ab, gdn_o, idx_f32, idx_q, scores, list, diag, attn_q, attn_gate,
      attn_part, attn_out, logits_r, routes, moe_h, y, ple_e, ple_kv, ple_ids, ple_consts, logits, argmax_part;
  // --- spec 21e, with the MTP head (null without): the verify's GDN state slots 1..3 (never zeroed: a slot is read
  //     only after a verify wrote it - Control::gdn_live is 0 after reset / load_state) and the PLE rows ----------
  std::unique_ptr<l0::Mem> gdn_spec;        // fp32 [its GDN layers][3][48][128][128]
  std::unique_ptr<l0::Mem> ple_gated, ple_gn;   // bf16 [kVerifyRows][10240] each (the PLE layer's device)

  const model::Qwen4ExpDesc& desc;
  model::Q4Placement placement;
  uint32_t device, max_len;
  Q4Attn attn;
  uint32_t rows;        // kM, or kVerifyRows with the MTP head

  bool holds(uint32_t layer) const { return layer >= placement.first(device) && layer < placement.end(device); }
  bool holds_ple() const { return holds(desc.ple_layer); }
  // Layer `layer`'s rows on this device (it must live here and be of the kind asked).
  void* k_layer(uint32_t layer) const;        // [max_len][2][256]
  void* v_layer(uint32_t layer) const;
  void* idx_keys_layer(uint32_t layer) const;  // [max_len / 4][128]
  void* tail_layer(uint32_t layer) const;      // [8][128]
  void* gdn_state_layer(uint32_t layer) const;
  void* conv_ring_layer(uint32_t layer) const;
  // Spec 21e: GDN layer `layer`'s state slot s (0: gdn_state_layer; 1..3: gdn_spec's) and its gdn_spec base (the
  // slots 1..3 gdn_step_slots_M<M>_G1 addresses from).
  void* gdn_slot(uint32_t layer, uint32_t s) const;
  void* gdn_spec_layer(uint32_t layer) const;
  void* ple_ids_ring() const { return ple.ptr(); }
  void* ple_conv_ring() const { return static_cast<uint8_t*>(ple.ptr()) + kPleConvOff; }
  // The per-layer scratch rows at this buffer's stride.
  size_t route_off(uint32_t layer) const { return route_at_r(layer, rows); }
  size_t list_off(uint32_t qsa_index) const { return list_at_r(qsa_index, rows); }
  size_t diag_off(uint32_t qsa_index) const { return diag_at_r(qsa_index, rows); }
  size_t persistent_bytes() const;
  size_t scratch_bytes() const;
  size_t mtp_bytes() const;   // gdn_spec + the PLE rows (0 without the head)
  // control (then n_active = 1 is the caller's), the KV, keys, tails, GDN state, rings - in construction order.
  void zero(l0::CmdList& imm);
};

// Spec 21e: the MTP head's buffers on the last device (runtime/qwen4exp/qwen4exp_sizes.h MtpSizes).
struct Qwen4ExpMtpBuffers {
  Qwen4ExpMtpBuffers(l0::Context& ctx, const model::Qwen4ExpDesc& d, uint32_t max_len);
  // persistent (zeroed by zero()): the head's Control, its KV, compressed keys, tail ring, the R rows
  l0::Mem hctl;       // runtime::Control, shared memory: the head runs one position behind the main model
  l0::Mem kv;         // bf16 [K | V][max_len][2][256]
  l0::Mem idx_keys;   // bf16 [max_len / 4][128]
  l0::Mem idx_tail;   // bf16 [8][128]
  l0::Mem hh;         // bf16 [1 + kVerifyRows][10240]: row 0 R_{pos-1}, rows 1..M the last verify's pre-mixer H
  // scratch
  l0::Mem list, diag;   // draft step 0's selection row (u32 [kListRow]) and its 512th / 513th scores
  l0::Mem xe, xh, fe, fh;   // the fusion's rows: bf16 [4][2560], bf16 [4][10240], fp32 [4][2560], fp32 [16][2560]
  l0::Mem logits;     // fp32 [kMaxDraft][vocab]: draft step i's head row (q_i, the sampled path)
  l0::Mem routes;     // u32 [kMaxDraft][32]: draft step i's route row (the head's MoE)
  const model::Qwen4ExpDesc& desc;
  uint32_t max_len;
  void* k() const { return kv.ptr(); }
  void* v() const { return static_cast<uint8_t*>(kv.ptr()) + size_t(max_len) * desc.kv_n() * 2; }
  void* hh_row(uint32_t r) const { return static_cast<uint8_t*>(hh.ptr()) + size_t(r) * desc.hc_n() * 2; }
  size_t bytes() const;
  void zero(l0::CmdList& imm);
};

}  // namespace runtime::qwen4exp
