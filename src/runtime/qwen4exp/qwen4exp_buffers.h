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
namespace runtime::qwen4exp {

struct Qwen4ExpBuffers {
  Qwen4ExpBuffers(l0::Context& ctx, const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t device,
                  uint32_t max_len, Q4Attn attn = q4_attn());

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

  const model::Qwen4ExpDesc& desc;
  model::Q4Placement placement;
  uint32_t device, max_len;
  Q4Attn attn;

  bool holds(uint32_t layer) const { return layer >= placement.first(device) && layer < placement.end(device); }
  bool holds_ple() const { return holds(desc.ple_layer); }
  // Layer `layer`'s rows on this device (it must live here and be of the kind asked).
  void* k_layer(uint32_t layer) const;        // [max_len][2][256]
  void* v_layer(uint32_t layer) const;
  void* idx_keys_layer(uint32_t layer) const;  // [max_len / 4][128]
  void* tail_layer(uint32_t layer) const;      // [8][128]
  void* gdn_state_layer(uint32_t layer) const;
  void* conv_ring_layer(uint32_t layer) const;
  void* ple_ids_ring() const { return ple.ptr(); }
  void* ple_conv_ring() const { return static_cast<uint8_t*>(ple.ptr()) + kPleConvOff; }
  size_t persistent_bytes() const;
  size_t scratch_bytes() const;
  // control (then n_active = 1 is the caller's), the KV, keys, tails, GDN state, rings - in construction order.
  void zero(l0::CmdList& imm);
};

}  // namespace runtime::qwen4exp
