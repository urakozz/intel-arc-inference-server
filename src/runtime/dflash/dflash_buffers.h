#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "model/dflash.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/dflash/dflash_sizes.h"

// Spec 19b Task 3: the DFlash drafter's device buffers, allocated exactly as runtime::dflash::sizes() says (the
// planner's formula; dflash_sizes.h lists every buffer and its layout). Owned beside the drafter's weights; the draft
// and projection lists (dflash_capture.h) bind them once.
namespace runtime::dflash {

struct DFlashBuffers {
  // head_rows: the target's lm_head rows, or a draft vocabulary's |V'| (the head GEMV's N).
  DFlashBuffers(l0::Context& ctx, const model::DFlashDesc& d, uint32_t target_hidden, uint32_t head_rows,
                uint32_t max_len);

  model::DFlashDesc desc;
  uint32_t target_hidden, head_rows, max_len;
  DFlashSizes sz;
  l0::Mem dctl, out;   // shared
  l0::Mem rope, ring_k, ring_v, taps, resid, h, x, xa, gout, q, kb, vb, part, kv, logits, tpart, cand, unary;
  std::unique_ptr<l0::Mem> co, hid;   // DFlash 2 only

  DFlashCtl* ctl() const { return dctl.as<DFlashCtl>(); }
  const DFlashDraftOut* readback() const { return out.as<DFlashDraftOut>(); }
  size_t bytes() const;   // == sz.total()
  // Zero the ring and every scratch, reset the control block and the readback, and upload the RoPE table: what
  // construction leaves and what a session reset restores (the ring then holds no position - a cold drafter).
  void zero(l0::CmdList& imm);
};

}  // namespace runtime::dflash
