#pragma once
// Spec 17c: TpEngine::Card's definition, shared by runtime/tp_engine.cc (the decode half, b70_runtime) and
// runtime/prefill/tp_prefill.cc (the prefill half, b70_prefill_host). Internal to the two; nothing else includes it.
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "runtime/tp_engine.h"
#include "runtime/tp_plan.h"   // TpCardSizes

namespace runtime {

// One card's half (Engine's members for one device, PipelineEngine::Stage's shape) and its crossing buffers.
struct TpEngine::Card {
  Card(l0::Context& c, loader::LoadedModel m, uint32_t max_len, KvCache kv, const tp::Layout& L, bool folded,
       bool tap_on)
      : ctx(c),
        model(std::move(m)),
        persist(c, max_len, *model.desc, kv),
        scratch(c, max_len, *model.desc),
        buffers(persist, scratch),
        queue(c),
        fence(queue),
        imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()),
        land(c, l0::MemKind::Device, sizes(L, folded, tap_on).land, 4096),
        send_ctr(c, l0::MemKind::Device, sizes(L, folded, tap_on).send_ctr),
        fold_ctr(c, l0::MemKind::Device, sizes(L, folded, tap_on).fold_ctr),
        state(c, l0::MemKind::Shared, sizes(L, folded, tap_on).state),
        own(folded ? std::make_unique<l0::Mem>(c, l0::MemKind::Device, sizes(L, folded, tap_on).own) : nullptr),
        tap(tap_on ? std::make_unique<l0::Mem>(c, l0::MemKind::Device, sizes(L, folded, tap_on).tap) : nullptr) {}
  // Spec 39: the card's own allocations' sizes (TpCardSizes, runtime/tp_plan.h); df_slots is made by TpEngine.
  TpCardSizes sizes(const tp::Layout& L, bool folded, bool tap_on) const {
    return tp_card_sizes(*model.desc, L, folded, tap_on, false);
  }
  l0::Context& ctx;
  loader::LoadedModel model;
  // Declaration order is construction order: `buffers` binds references into the two above it (Engine's rule).
  PersistentBuffers persist;
  DecodeScratch scratch;
  DecodeBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  l0::Mem land;                    // written by the OTHER card's tp_send, read by this card's tp_res_fold
  l0::Mem send_ctr, fold_ctr;      // this card's call counters (device, advanced by its kernels)
  l0::Mem state;                   // shared: the first failure, read after the fence
  std::unique_ptr<l0::Mem> own;    // folded: this card's folded row
  std::unique_ptr<l0::Mem> tap;    // debug_resid
  std::optional<CapturedStep> step;
  // Spec 17d: the MTP head's buffers and lists (null / empty without the head), and the link the lists were captured
  // with (kept for spec 28c's deep lists).
  TpLink link;
  std::unique_ptr<MtpBuffers> mtp;
  Control* hctl = nullptr;
  std::vector<CapturedStep> drafts;                          // [kMaxDraft]
  std::vector<CapturedStep> draft_probs;                     // [kMaxDraft], open-work 13: drafts ending in draft_prob
  std::vector<CapturedStep> verifies;                        // [M - 1], M = 1..kSlots, by the rule at row 0's key 1
  std::vector<std::unique_ptr<CapturedStep>> verify_deep;    // [M - 1]: DPAS bound where the rule switches by depth
  std::vector<uint32_t> verify_deep_from;
  // Spec 19 v3c: with a drafter, its GDN state slots 1..7 at the card's shapes and what the tap verify lists (in
  // `verifies` / `verify_deep`, M = 1..max_k + 1) bind.
  std::unique_ptr<l0::Mem> df_slots;
  TapVerify tv;
  // The live GDN state (slot 0, or a speculative slot after a commit) and the hh rows (row 0 = h_{pos-1}).
  uint8_t* live_gdn() const {
    const uint32_t live = (mtp || df_slots) ? ctl->gdn_live : 0;
    if (live == 0) return persist.gdn_state.as<uint8_t>();
    const uint8_t* spec = df_slots ? df_slots->as<uint8_t>() : mtp->gdn_spec.as<uint8_t>();
    return const_cast<uint8_t*>(spec) + size_t(live - 1) * persist.gdn_state.size();
  }

  // Spec 39: every allocation above of TpCardSizes' names (the crossing, the tap, the drafter's slots).
  size_t bytes() const { return gs::buffer_bytes<l0::Mem, TpCardSizes>(*this); }
  // Every zero_on_reset one: the crossing's landing, counters and state words, and (spec 39: reset left them before)
  // the drafter's slots.
  void zero() {
    gs::zero<l0::Mem, TpCardSizes>(*this, [&](l0::Mem& m) {
      if (m.kind() == l0::MemKind::Shared)
        std::memset(m.ptr(), 0, m.size());
      else
        imm.fill(m.ptr(), 0u, m.size());
    });
  }
};


}  // namespace runtime
