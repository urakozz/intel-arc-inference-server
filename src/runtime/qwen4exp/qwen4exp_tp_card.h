#pragma once
#include <optional>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "kernels/qwen4exp_kernels.h"
#include "l0/queue.h"
#include "model/qwen4exp.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/qwen4exp/qwen4exp_buffers.h"
#include "runtime/qwen4exp/qwen4exp_capture.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp/qwen4exp_tp_engine.h"
#include "runtime/tp_protocol.h"

// Qwen4ExpTpEngine's card (qwen4exp_tp_engine.cc), in a header of its own so the prefill half
// (qwen4exp_tp_prefill.cc, b70_qwen4exp_prefill) reaches the card's buffers, queue and Control.
namespace runtime::qwen4exp {

// One card: its descriptor (the buffers keep a reference to it, so it is declared first), buffers, queue, crossing.
// With the MTP head (`mtp`): the verify rows and GDN slots, the MoE's slots at kVerifyRows rows; the head's card
// (`head`) sizes its decode scratch at one card's widths (the head's layer runs whole there).
struct Qwen4ExpTpEngine::Card {
  Card(l0::Context& c, const model::Qwen4ExpDesc& one, uint32_t rank, uint32_t max_len, bool mtp = false,
       bool head = false)
      : ctx(c),
        desc(model::q4_tp_card(one, rank)),
        buffers(c, desc, model::Q4Placement::one(desc), 0, max_len, Q4Attn::Flash, mtp, head ? &one : nullptr),
        queue(c),
        fence(queue),
        imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()),
        lo{one.hidden, kTpOutSlices, tp::kFoldG},
        lm{one.hidden, one.top_k + 1, kernels::qwen4exp::kTpMoeFoldG},
        land_o(c, l0::MemKind::Device, lo.total_bytes(), 4096),
        land_m(c, l0::MemKind::Device, lm.total_bytes(), 4096),
        ctr_so(c, l0::MemKind::Device, lo.counter_bytes()),
        ctr_fo(c, l0::MemKind::Device, lo.counter_bytes()),
        ctr_sm(c, l0::MemKind::Device, lm.counter_bytes()),
        ctr_fm(c, l0::MemKind::Device, lm.counter_bytes()),
        slots(c, l0::MemKind::Device, size_t(one.top_k + 1) * (mtp ? kVerifyRows : 1) * one.hidden * 4),
        state(c, l0::MemKind::Shared, tp::kStateWords * 4) {}
  l0::Context& ctx;
  const model::Qwen4ExpDesc desc;
  Qwen4ExpBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  tp::Layout lo, lm;   // out_proj / o_proj's crossing, the MoE's
  l0::Mem land_o, land_m, ctr_so, ctr_fo, ctr_sm, ctr_fm, slots, state;
  TpLink link, moe;
  Q4TpBinding binding;
  std::optional<CapturedStep> step;
  bool pending = false;
  // Task 5 Step 0 (the MTP head): the verify lists at M = 1..kVerifyRows and the draft steps (the head's card: spec
  // 21e's step on its half of the head; the other card: its half of the head over the peer's row, into dctl)
  std::vector<CapturedStep> verify, drafts;
  std::unique_ptr<l0::Mem> dctl;   // runtime::Control, shared: the card without the head's draft winner
};

}  // namespace runtime::qwen4exp
