#pragma once
// Kolibri DSpark: speculative decoding for Kolibri-1 - b70-serve on Kolibri's cards (two by default), selected by the one
// --spec surface (cli/spec_args.h, SpecFamily::Kolibri):
//
//   --spec dflash --draft-model PATH   the proposer is the DSpark drafter (intuit/Kolibri-1-dspark: a snapshot directory or
//                          a repo id in the HF cache, never downloaded here) - its config.json's DSparkDraftModel picks the
//                          DSpark form; on the LAST card beside the target's lm_head, with a 655 MB embedding replica there
//                          (Bolt's arrangement for LiLiCorr): one 8-row block drafts up to 7 ids, verified at M = K + 1 by
//                          Kolibri's verify lists (runtime::kolibri::KolibriEngine)
//   --spec-max K           the deepest block, 1..7 (default 7); AdaptiveK picks K per iteration
//
// What the engine decides (no flag, no environment variable): the drafter runs bf16 (an int8 / int4 drafter is a later step);
// the Markov walk takes the head's top 16 a row; the AdaptiveK cost table is server::MtpCost::kolibri_dspark (DERIVED -
// TODO(card): the first card run measures it). --spec dflash is NOT Kolibri's `auto`.
//
// Refused before the device, each by name: by cli::resolve_spec --spec mtp / lookup / dflash-lookup and --spec-max outside
// 1..7; by cli::dflash::check_flags --spec dflash without --draft-model; here a drafter that is not DSpark, not DSpark's
// kernels' shapes (model::check_dspark) or does not fit Kolibri-1 (taps past its layers, another hidden or vocabulary).
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "cli/kolibri_serve.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_layout.h"
#include "loader/dflash_loader.h"
#include "loader/dflash_repack.h"
#include "model/dflash.h"
#include "model/kolibri1.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/dflash/dspark_drafter.h"
#include "runtime/kolibri/kolibri_engine.h"
#include "runtime/kolibri/kolibri_sizes.h"
#include "server/adaptive_k.h"

namespace cli::kolibri {

struct SpecArgs {
  bool on = false;          // --spec dflash
  std::string draft_model;  // --draft-model
  uint32_t max_k = 7;       // --spec-max
};

inline model::DFlashTargetShape target_shape(const model::Kolibri1Desc& d) {
  return {"Kolibri-1", d.layers, d.hidden, d.vocab};
}

// The drafter's descriptor before the device - its checkpoint alone (config.json, the safetensors header), held to the DSpark
// kernels' shapes and to the target `d` (the descriptor the engine will load).
inline model::DFlashDesc drafter_desc(const SpecArgs& s, const model::Kolibri1Desc& d) {
  const loader::DFlashCheckpoint ck(s.draft_model);
  if (!ck.desc().dspark())
    throw std::runtime_error("--draft-model " + s.draft_model + " is " + ck.desc().architecture +
                             ", not a DSpark drafter (DSparkDraftModel): Kolibri-1's --spec dflash runs DSpark "
                             "(intuit/Kolibri-1-dspark)");
  model::check_dspark(ck.desc());
  ck.check_target(target_shape(d));
  if (s.max_k > ck.desc().max_k())
    throw std::runtime_error("--spec-max " + std::to_string(s.max_k) + " exceeds the drafter's block (" +
                             std::to_string(ck.desc().max_k()) + " drafts)");
  return ck.desc();
}

// The plan's spec terms (DecodeArgs::spec - settle() adds them): the verify lists on every card and, on the last card, the
// drafter at the length bound (bf16 weights + buffers + its prefill staging) and the embedding replica. `bound` sizes the
// drafter's RoPE rows: `--max-len auto` passes the cap it sizes to.
inline runtime::kolibri::SpecReserve spec_reserve(const SpecArgs& s, const model::Kolibri1Desc& d, uint32_t bound) {
  runtime::kolibri::SpecReserve r;
  if (!s.on) return r;
  const model::DFlashDesc dd = drafter_desc(s, d);
  r.verify = true;
  r.drafter = loader::dflash_weight_bytes(dd, d.hidden, loader::DraftFormat::Bf16).total() +
              runtime::dflash::sizes(dd, d.hidden, d.vocab, bound, loader::DraftFormat::Bf16).total() +
              runtime::kolibri::prefill_taps_bytes(dd.window.at(0), uint32_t(dd.taps.size()), d.hidden);
  r.replica = size_t(d.vocab) * d.hidden * 2;
  return r;
}

// The drafter beside the loaded engine (whose verify lists are on): the embedding replica on the last card, copied from device
// 0's embedding by device 0 (the hand-off's direction of peer access), the bf16 weights (loader::load_drafter's target-shape
// overload: its own checks; the mask row from the replica), its buffers at the engine's max_len, attached to the engine
// (verify taps, commit projection, prefill tail, reset, snapshots) and its draft list captured over the replica, the engine's
// lm_head and the last card's Control block.
struct KolibriDrafter {
  std::unique_ptr<l0::Mem> replica;
  std::unique_ptr<runtime::dflash::DSparkDrafter> drafter;
};
inline KolibriDrafter attach_drafter(runtime::kolibri::KolibriEngine& e, const SpecArgs& s) {
  const model::Kolibri1Desc& d = e.model().desc;
  const uint32_t last = e.devices() - 1;
  l0::Context& ctx = e.context(last);
  KolibriDrafter out;
  const size_t embed_bytes = size_t(d.vocab) * d.hidden * 2;
  out.replica = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, embed_bytes);
  {   // TODO(card): 655 MB peer-to-peer at load - if the first run stalls here, upload the replica from the host instead
    l0::CmdList imm = l0::CmdList::immediate(e.context(0));
    imm.copy(out.replica->ptr(), e.embedding().ptr(), embed_bytes);
  }
  out.drafter = std::make_unique<runtime::dflash::DSparkDrafter>(
      ctx, loader::load_drafter(ctx, s.draft_model, target_shape(d), *out.replica, loader::DraftFormat::Bf16), e.max_len(),
      s.max_k);
  e.attach_drafter(*out.drafter);
  out.drafter->attach(ctx, *out.replica, e.lm_head(), e.control_mem(last));
  return out;
}

// The adapter's view of it: one block at the engine's pos, replayed on the last card's queue.
inline KolibriDraftHooks draft_hooks(runtime::kolibri::KolibriEngine& e, runtime::dflash::DSparkDrafter& d) {
  KolibriDraftHooks h;
  h.draft = [&e, &d](uint32_t k, float temperature, uint64_t seed) -> const runtime::dflash::DFlashDraftOut& {
    return d.draft([&e](l0::CmdList& list) { e.replay_aux(list); }, k, temperature, seed);
  };
  return h;
}

}  // namespace cli::kolibri
