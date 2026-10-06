#pragma once
#include <cstdint>
#include <memory>

#include "l0/context.h"
#include "l0/memory.h"
#include "model/kolibri1.h"
#include "runtime/kolibri/kolibri_sizes.h"

namespace l0 {
class CmdList;
}

// Spec 20c: one device's Kolibri-1 decode allocations (runtime/kolibri/kolibri_sizes.h sizes every
// one; the constructor allocates exactly those): the persistent group of the device's layers - the
// full layers' growing KV and the sliding layers' rings - and the whole decode scratch.
namespace runtime::kolibri {

struct KolibriBuffers {
  // `attn`: the decode attention the capture will bind; by default the one B70_KOLIBRI_ATTN selects.
  KolibriBuffers(l0::Context& ctx, const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t device,
                 uint32_t max_len, KolAttn attn = kolibri_attn());

  // --- persistent: zeroed by zero() (KolibriEngine::reset) -------------------------------------------
  l0::Mem control;                 // runtime::Control, shared memory
  l0::Mem full_k, full_v;          // bf16 [its full layers][max_len][kv_heads][128]
  l0::Mem ring_k, ring_v;          // bf16 [its sliding layers][kRing][kv_heads][128]
  // --- decode scratch: never zeroed (no step reads scratch it has not written first) -------------------
  l0::Mem resid, x, a, mo, partials, sumsq_a, sumsq_r, attn_q, attn_part, attn_out, logits_r, routes, moe_h,
      logits, argmax_part;
  std::unique_ptr<l0::Mem> attn_scores;   // B70_KOLIBRI_ATTN=eager only: fp32 [M][q_heads][max_len]

  const model::Kolibri1Desc& desc;
  model::KolPlacement placement;
  uint32_t device, max_len;
  KolAttn attn;

  // Layer `layer`'s K / V rows on this device (it must live here): a full layer's
  // [max_len][kv_heads][128] or a sliding layer's ring [kRing][kv_heads][128].
  void* k_layer(uint32_t layer) const;
  void* v_layer(uint32_t layer) const;
  bool holds(uint32_t layer) const { return layer >= placement.first(device) && layer < placement.end(device); }
  size_t persistent_bytes() const { return control.size() + full_k.size() + full_v.size() + ring_k.size() + ring_v.size(); }
  size_t scratch_bytes() const;
  // control (then n_active = 1), the full KV and the rings - in construction order.
  void zero(l0::CmdList& imm);
};

}  // namespace runtime::kolibri
