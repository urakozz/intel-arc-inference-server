#pragma once
// Spec 19c: `--spec dflash` - speculative decoding with a DFlash drafter (spec 19 §5) - for b70-serve and b70-decode
// (--ids), the Qwen3.5 family's engine (runtime::Engine); spec 19d: and b70-serve --parallel pp (runtime::PipelineEngine,
// the drafter on device 1 beside lm_head). The flags (cli/spec_args.h has --spec itself):
//
//   --spec dflash          the proposer is the drafter (spec 19 decision 3: one per server; the MTP head is not
//                          loaded)
//   --draft-model PATH     the drafter: a snapshot directory or a repo id in the HF cache (default
//                          z-lab/Qwen3.8-27B-DFlash2, the published drafter for Qwen3.8); never downloaded here
//   --spec-max K           the deepest block, 1..7 (default 7: DFlash 2's block of 8 - 1)
//   --draft-vocab N        (b70-serve) the drafter's head over a draft vocabulary (spec 8 §11's, built without the
//                          MTP head)
// The drafter's linears are int8 (spec 19 decision 2: per-row RTN at load; --draft-format went in the zero-config
// cleanup - Args::format stays the tests' seam) and its AdaptiveK cost table is MtpCost::dflash's (--spec-cost went
// with it).
//
// What is refused here, by name, before the device (each is a later stage's, or has no binaries):
//   pipeline parallel      b70-decode: one card (b70-serve --parallel pp serves it); b70-serve --parallel pp: the
//                          copy hand-off only (--pipeline-handoff peer: device 0's taps are copies into device 1
//                          that only the copy hand-off's event orders)
//   --kv-cache int8        the verify lists at M = 5..8 have no int8-KV attention binaries (kv8.cl builds M <= 4)
// and by the loader / the capture (with the reason): a target the drafter is not trained for
// (model::check_dflash_target), an int4 lm_head (the draft head reads int8 or bf16 rows).
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "l0/context.h"
#include "loader/dflash_layout.h"
#include "loader/dflash_loader.h"
#include "loader/dflash_repack.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffer_sizes.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/memory_plan.h"
#include "runtime/prefill_backend.h"
#include "server/adaptive_k.h"

namespace cli::dflash {

struct Args {
  bool on = false;
  std::string model = "z-lab/Qwen3.8-27B-DFlash2";
  loader::DraftFormat format = loader::DraftFormat::Int8;
  uint32_t max_k = 7;
  bool have_model = false;
};

// --draft-model belongs to --spec dflash.
inline void check_flags(const Args& a) {
  if (!a.on && a.have_model) throw std::runtime_error("--draft-model needs --spec dflash");
  if (a.on && (a.max_k == 0 || a.max_k > 7))
    throw std::runtime_error("--spec-max with --spec dflash expects 1..7 (the drafter's block of 8)");
}

// The engine's refusals (see the header). Spec 19d: `pipeline_served` - the caller serves two cards with the drafter
// (b70-serve); `peer` - its hand-off is --pipeline-handoff peer.
inline void check_engine(const Args& a, bool pipeline, runtime::KvCache kv, bool pipeline_served = false,
                         bool peer = false) {
  if (!a.on) return;
  if (pipeline && !pipeline_served)
    throw std::runtime_error("b70-decode --spec dflash runs on one card (b70-serve --parallel pp --spec dflash serves "
                             "the drafter beside a split target): --parallel 1");
  if (pipeline && peer)
    throw std::runtime_error("--spec dflash --parallel pp needs --pipeline-handoff copy: device 0's taps are copies into "
                             "device 1, which only the copy hand-off's event orders before device 1's rows");
  if (kv == runtime::KvCache::Int8)
    throw std::runtime_error("--spec dflash verifies up to 8 rows, and --kv-cache int8's attention is built for "
                             "M <= 4 (kv8.cl): use --kv-cache bf16 with --spec dflash");
}

// Spec 19v2 C3: ranked32k failed quality; full int8 heads use the candidate64/exact-rerank fallback.
inline bool candidate_head(bool int8_head, uint32_t draft_vocab) {
  return int8_head && draft_vocab == 0;
}

// The AdaptiveK cost table: MtpCost::dflash for the head and drafter forms, a draft vocabulary scaling the head's
// share of the draft row (derived, as the MTP head's).
inline server::MtpCost cost(const Args& a, loader::LmHeadForm head, uint32_t draft_vocab) {
  const bool i8h = head == loader::LmHeadForm::Int8, i8d = a.format == loader::DraftFormat::Int8;
  server::MtpCost c = server::MtpCost::dflash(i8h, i8d);
  if (draft_vocab != 0)
    c = c.with_draft_vocab(server::MtpCost::dflash_head_share(i8h, i8d), double(draft_vocab) / model::Qwen35::kVocab);
  if (c.max_k() < a.max_k)
    throw std::runtime_error("the drafter's cost table covers K up to " + std::to_string(c.max_k()) +
                             ", below --spec-max " + std::to_string(a.max_k));
  return c;
}

// The drafter's weights beside the loaded target, the head rows its draft list reads (the full head's, or the
// draft vocabulary's) and its planner term (runtime::dflash::drafter_plan: weights, buffers; the plan adds the
// engine's verify slots and prefill taps).
struct Loaded {
  loader::DFlashWeights w;
  uint32_t head_rows = 0;
  runtime::DrafterPlan plan;
};
// Spec 19d (two cards): the planner's term before the drafter is loaded - read off its checkpoint (config.json, the
// shards' shapes) against the WHOLE target as loaded on device 0, so the split and max_len account for it; the
// weights are then loaded onto device 1 beside the stage model (load(), below).
inline runtime::DrafterPlan plan_only(const Args& a, const loader::LoadedModel& target) {
  const loader::DFlashCheckpoint ck(a.model);
  const uint32_t head_rows = target.draft_vocab ? target.draft_vocab->size()
                                                : target.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  ck.check_target(*target.desc, target.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N);
  if (a.max_k > ck.desc().max_k())
    throw std::runtime_error("--spec-max " + std::to_string(a.max_k) + " exceeds the drafter's block (" +
                             std::to_string(ck.desc().max_k()) + " drafts)");
  const bool candidate = candidate_head(target.linears.at({loader::kTopLevel, model::LinearId::LmHead}).kind ==
                                           model::WeightKind::Int8, target.draft_vocab ? target.draft_vocab->size() : 0);
  return runtime::dflash::drafter_plan(ck.desc(), ck.target_hidden(), head_rows, a.format, candidate);
}

inline Loaded load(l0::Context& ctx, const Args& a, const loader::LoadedModel& target) {
  const bool candidate = candidate_head(target.linears.at({loader::kTopLevel, model::LinearId::LmHead}).kind ==
                                           model::WeightKind::Int8, target.draft_vocab ? target.draft_vocab->size() : 0);
  Loaded l{loader::load_drafter(ctx, a.model, target, a.format, 0, candidate), 0, {}};
  l.head_rows = target.draft_vocab ? target.draft_vocab->size()
                                   : target.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  l.plan = runtime::dflash::drafter_plan(l.w.desc, l.w.target_hidden, l.head_rows, a.format, candidate);
  if (a.max_k > l.w.desc.max_k())
    throw std::runtime_error("--spec-max " + std::to_string(a.max_k) + " exceeds the drafter's block (" +
                             std::to_string(l.w.desc.max_k()) + " drafts)");
  return l;
}

}  // namespace cli::dflash
