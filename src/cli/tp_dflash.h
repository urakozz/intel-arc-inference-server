#pragma once
#include <array>
#include <stdexcept>
#include <string>

#include "cli/dflash_args.h"
#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/memory_plan.h"

// Spec 19 v3c: b70-serve --parallel tp --spec dflash - the drafter's halves beside the two TP cards
// (runtime::dflash::TpDFlashDrafter). Each card loads its half beside its own slice of the target (loader::load_drafter's
// tp_rank: its heads and MLP half, its candidate head made from its vocabulary half of the int8 head), and plans it
// as a drafter of the card's half shapes (the engine's 7 more GDN slots and the prefill's tap staging at the card's
// shapes come with runtime::plan's drafter term).
namespace cli::dflash {

struct TpLoaded {
  std::array<loader::DFlashWeights, 2> halves;
  runtime::DrafterPlan plan;   // one card's (both are the same shapes)
};

inline TpLoaded load_tp(std::array<l0::Context*, 2> ctx, const Args& a, const std::array<loader::LoadedModel, 2>& cards) {
  for (const loader::LoadedModel& m : cards)
    if (m.linears.at({loader::kTopLevel, model::LinearId::LmHead}).kind != model::WeightKind::Int8)
      throw std::runtime_error("--parallel tp --spec dflash: the drafter re-ranks against each card's int8 head half "
                               "(--lm-head int8, the default)");
  TpLoaded l{{loader::load_drafter(*ctx[0], a.model, cards[0], a.format, 0, true, 2, 80, 0),
              loader::load_drafter(*ctx[1], a.model, cards[1], a.format, 0, true, 2, 80, 1)},
             {}};
  if (a.max_k > l.halves[0].desc.max_k())
    throw std::runtime_error("--spec-max " + std::to_string(a.max_k) + " exceeds the drafter's block (" +
                             std::to_string(l.halves[0].desc.max_k()) + " drafts)");
  const loader::DFlashWeights& w = l.halves[0];
  l.plan = runtime::dflash::drafter_plan(w.desc, w.target_hidden, w.candidate_head->shape.N, a.format, true);
  return l;
}

}  // namespace cli::dflash
