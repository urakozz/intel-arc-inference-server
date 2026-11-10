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
        land(c, l0::MemKind::Device, L.total_bytes(), 4096),
        send_ctr(c, l0::MemKind::Device, L.counter_bytes()),
        fold_ctr(c, l0::MemKind::Device, L.counter_bytes()),
        state(c, l0::MemKind::Shared, tp::kStateWords * 4),
        own(folded ? std::make_unique<l0::Mem>(c, l0::MemKind::Device,
                                               size_t(tp::kMRows) * model.desc->hidden * 4)
                   : nullptr),
        tap(tap_on ? std::make_unique<l0::Mem>(c, l0::MemKind::Device,
                                                size_t(model.desc->layers) * model.desc->hidden * 2)
                   : nullptr) {}
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
  std::vector<CapturedStep> verifies;                        // [M - 1], M = 1..kSlots, by the rule at row 0's key 1
  std::vector<std::unique_ptr<CapturedStep>> verify_deep;    // [M - 1]: DPAS bound where the rule switches by depth
  std::vector<uint32_t> verify_deep_from;
  // The live GDN state (slot 0, or a speculative slot after a commit) and the hh rows (row 0 = h_{pos-1}).
  uint8_t* live_gdn() const {
    const uint32_t live = mtp ? ctl->gdn_live : 0;
    return live == 0 ? persist.gdn_state.as<uint8_t>()
                     : mtp->gdn_spec.as<uint8_t>() + size_t(live - 1) * persist.gdn_state.size();
  }

  size_t bytes_crossing() const {
    return land.size() + send_ctr.size() + fold_ctr.size() + state.size() + (own ? own->size() : 0);
  }
  void zero_crossing() {
    imm.fill(land.ptr(), 0u, land.size());
    imm.fill(send_ctr.ptr(), 0u, send_ctr.size());
    imm.fill(fold_ctr.ptr(), 0u, fold_ctr.size());
    std::memset(state.ptr(), 0, state.size());
  }
};

}  // namespace runtime
