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

// Plan 19v3b: one sequence's drafter state - its control block and readback (shared memory), its context ring and
// its tap buffer - sized as DFlashSizes' terms of those names. DFlashBuffers' own members are the first sequence's;
// every further sequence slot of the engine holds one of these (DFlashBuffers::add_sessions).
struct DFlashSession {
  DFlashSession(l0::Context& ctx, const DFlashSizes& sz);
  l0::Mem dctl, out, ring_k, ring_v, taps;
  DFlashCtl* ctl() const { return dctl.as<DFlashCtl>(); }
  const DFlashDraftOut* readback() const { return out.as<DFlashDraftOut>(); }
  size_t bytes() const;
  void zero(l0::CmdList& imm);   // ring and taps zeroed, control and readback reset
};

struct DFlashBuffers {
  // head_rows: the target's lm_head rows, or a draft vocabulary's |V'| (the head GEMV's N).
  // `host_candidates` (spec 19 v3c, a TP card's half): cand and unary in shared memory - the host merges the two
  // cards' top-16 lists between the draft list and the tail.
  DFlashBuffers(l0::Context& ctx, const model::DFlashDesc& d, uint32_t target_hidden, uint32_t head_rows,
                uint32_t max_len, loader::DraftFormat format = loader::DraftFormat::Int8, bool candidate_head = false,
                bool host_candidates = false);

  model::DFlashDesc desc;
  uint32_t target_hidden, head_rows, max_len;
  DFlashSizes sz;
  l0::Mem dctl, out;   // shared
  l0::Mem rope, ring_k, ring_v, taps, resid, h, x, xa, gout, q, kb, vb, part, kv, logits, tpart, cand, unary;
  std::unique_ptr<l0::Mem> co, hid, candidate;   // DFlash 2 only (hid also Kolibri DSpark's: ones, its Markov walk)

  // Plan 19v3b: sessions 1..n-1 for an engine of n sequence slots (session 0 is the members above).
  std::vector<std::unique_ptr<DFlashSession>> sessions;
  void add_sessions(l0::Context& ctx, uint32_t n);
  // Session s's buffers (0: the members above) and its reset - ring and taps zeroed, control and readback reset.
  struct Session {
    l0::Mem *dctl, *out, *ring_k, *ring_v, *taps;
    DFlashCtl* ctl() const { return dctl->as<DFlashCtl>(); }
    const DFlashDraftOut* readback() const { return out->as<DFlashDraftOut>(); }
  };
  Session session(uint32_t s);
  void zero_session(l0::CmdList& imm, uint32_t s);
  // Plan 19v3b: the batched draft's row table (shared, uint [kM + 8 kSeqs]: per row its slot or 0xFFFFFFFF, then per
  // slot row0, rows, pos, K, anchor, temperature bits, seed lo, seed hi - df_prep.cl's SEQS) - with sessions only.
  std::unique_ptr<l0::Mem> seqs;
  // Plan 19v3b: the row lists' taps (every row's, [kM][taps][hidden] bf16 as `taps`) and the rows projection's table
  // (`seqs`' layout: per slot row0, its committed rows, its first position) - with sessions only.
  std::unique_ptr<l0::Mem> rows_taps, proj_seqs;

  DFlashCtl* ctl() const { return dctl.as<DFlashCtl>(); }
  const DFlashDraftOut* readback() const { return out.as<DFlashDraftOut>(); }
  size_t bytes() const;   // == sz.total() + the further sessions'
  // Zero the ring and every scratch, reset the control block and the readback. The immutable RoPE table is
  // uploaded once at construction; a session reset leaves it alone (the ring then holds no position).
  void zero(l0::CmdList& imm);
};

}  // namespace runtime::dflash
