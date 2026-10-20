#pragma once
#include <cstdint>
#include <memory>

#include "l0/context.h"
#include "l0/memory.h"
#include "model/lfm2moe.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

namespace l0 {
class CmdList;
}

// Spec 24c: one device's LFM2.5-24B-A2B decode allocations (runtime/lfm2moe/lfm2moe_sizes.h sizes every one; the
// constructor allocates exactly those): the persistent group of the device's layers - the attention layers' growing
// KV and the conv layers' Bx rings - and the whole decode scratch.
namespace runtime::lfm2moe {

struct Lfm2MoeBuffers {
  // `attn`: the decode attention the capture will bind (eager allocates the score row).
  Lfm2MoeBuffers(l0::Context& ctx, const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t device,
                 uint32_t max_len, LfmAttn attn);

  // --- persistent: zeroed by zero() (Lfm2MoeEngine::reset) ----------------------------------------------------------
  l0::Mem control;                 // runtime::Control, shared memory
  l0::Mem kv_k, kv_v;              // bf16 [its attention layers][max_len][kv_heads][64]
  l0::Mem conv_ring;               // bf16 [its conv layers][16][hidden] (Bx; never read before written: positions < 0
                                   // read 0 - zeroed anyway, so a snapshot of a fresh session is all zeros)
  // --- decode scratch: never zeroed (no step reads scratch it has not written first) -------------------------------
  l0::Mem h0, h1, x, slices, conv_y, logits_r, routes, moe_h, attn_q, attn_part, attn_out, s_mlp, logits, argmax_part;
  std::unique_ptr<l0::Mem> attn_scores;   // eager only: fp32 [M][q_heads][max_len]

  const model::Lfm2MoeDesc& desc;
  model::LfmPlacement placement;
  uint32_t device, max_len;
  LfmAttn attn;

  // Layer `layer`'s K / V rows / conv ring on this device (it must live here and be of that kind).
  void* k_layer(uint32_t layer) const;
  void* v_layer(uint32_t layer) const;
  void* ring_layer(uint32_t layer) const;
  bool holds(uint32_t layer) const { return layer >= placement.first(device) && layer < placement.end(device); }
  size_t persistent_bytes() const { return control.size() + kv_k.size() + kv_v.size() + conv_ring.size(); }
  size_t scratch_bytes() const;
  // control (then n_active = 1), the KV and the rings - in construction order.
  void zero(l0::CmdList& imm);
};

}  // namespace runtime::lfm2moe
