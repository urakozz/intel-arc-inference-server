#pragma once
// Spec 23f: speculative decoding for Muse-Glimmer-30B - b70-serve (and b70-decode --ids, greedy) on ONE card:
//
//   --spec dflash          the proposer is Muse-Glimmer-30B's DFlash assistant (runtime::dflash::MuseDrafter beside
//                          runtime::muse::MuseEngine): one 16-row block drafts up to 15 ids, verified at M = K + 1
//   --draft-model PATH     the assistant: a snapshot directory or a repo id in the HF cache (default
//                          meta-models/Muse-Glimmer-30B-assistant, bf16 as published); never downloaded here
//   --draft-format F       int4 (the default: g64 RTN at load - spec 23 decision 9, provisional on box P0), int8
//                          (per-row RTN) or bf16 (as shipped) - our formats only
//   --draft-q raw|capped   the draft distribution q a sampled request draws its drafts from: softmax(z / T) over the
//                          16 candidates' RAW head logits (the default - transformers' form; P0 measured it far ahead)
//                          or the target's capped 20 tanh(z x 0.196 / 20) (spec 23 decision 11's other form)
//   --spec lookup          the proposer is the server's prompt lookup (spec 19e), verified through the same lists
//   --spec-max K           the deepest block, 1..15 (default 15; AdaptiveK picks K per iteration in b70-serve)
//   --spec-cost SPEC       b70-serve: the AdaptiveK cost table (--mtp-cost's syntax, verify to M = 16, draft to 15);
//                          default server::MtpCost::muse_dflash / muse_verify (derived)
//
// Refused before the device, each by name: --pp 2 (the verify lists and the drafter run beside one engine: two cards
// serve Muse plainly), --spec-max outside 1..15, --draft-vocab (no compact head for this drafter), --draft-model /
// --draft-format / --draft-q without --spec dflash, B70_PREFILL_REPLAY=1 with --spec dflash (the prefill taps are
// copies a recorded chunk cannot hold); by the loader: a drafter that is not the assistant's shapes
// (model::check_muse_assistant) or does not fit the target (check_dflash_target).
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "cli/muse_serve.h"
#include "l0/cmdlist.h"
#include "loader/dflash_layout.h"
#include "loader/dflash_loader.h"
#include "loader/dflash_repack.h"
#include "model/dflash.h"
#include "model/muse_glimmer.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/dflash/muse_drafter.h"
#include "runtime/muse/muse_engine.h"
#include "runtime/muse/muse_sizes.h"
#include "server/adaptive_k.h"

namespace cli::muse {

inline constexpr const char* kDefaultAssistant = "meta-models/Muse-Glimmer-30B-assistant";
inline constexpr uint32_t kSpecMaxK = 15;

struct SpecArgs {
  std::string spec = "off";   // off | dflash | lookup (mtp: check_args refuses it)
  std::string draft_model = kDefaultAssistant;
  loader::DraftFormat format = loader::DraftFormat::Int4G64;
  runtime::dflash::MuseQ q = runtime::dflash::MuseQ::Raw;
  uint32_t max_k = kSpecMaxK;
  std::string cost;            // --spec-cost
  bool have_model = false, have_format = false, have_q = false, have_max_k = false, draft_vocab = false;
  bool dflash() const { return spec == "dflash"; }
  bool lookup() const { return spec == "lookup"; }
  bool on() const { return dflash() || lookup(); }
};

// Every refusal that needs no device (the header's list), in order.
inline void check_spec(const SpecArgs& s, uint32_t devices, bool prefills) {
  if (!s.dflash() && (s.have_model || s.have_format || s.have_q))
    throw std::runtime_error("--draft-model, --draft-format and --draft-q need --spec dflash");
  if (s.on() && devices != 1)
    throw std::runtime_error("--spec " + s.spec + " on Muse-Glimmer-30B runs on one card (spec 23f: the verify lists "
                             "and the drafter beside one engine; --pp 2 serves it plainly) - drop --pp 2");
  if ((s.have_max_k || !s.cost.empty()) && !s.on())
    throw std::runtime_error("--spec-max and --spec-cost need --spec dflash or --spec lookup");
  if (s.on() && (s.max_k == 0 || s.max_k > kSpecMaxK))
    throw std::runtime_error("--spec-max with Muse-Glimmer-30B expects 1..15 (the assistant's block of 16 - 1)");
  if (s.draft_vocab)
    throw std::runtime_error("--draft-vocab is not built for Muse-Glimmer-30B's drafter (spec 23f: no compact head)");
  if (s.dflash() && prefills) {
    const char* env = std::getenv("B70_PREFILL_REPLAY");
    if (env && std::strcmp(env, "1") == 0)
      throw std::runtime_error("--spec dflash: the drafter's prefill taps are copies, which a recorded chunk cannot "
                               "hold - unset B70_PREFILL_REPLAY");
  }
}

inline model::DFlashTargetShape target_shape(const model::MuseGlimmerDesc& d, uint32_t head_rows) {
  return {"Muse-Glimmer-30B", d.layers, d.hidden, head_rows};
}

// The drafter's planner term before the device - from its checkpoint alone (config.json, the shards' shapes), held to
// the assistant's shapes and to the target (`d`: the descriptor the engine will load).
inline runtime::DrafterPlan drafter_plan(const SpecArgs& s, const model::MuseGlimmerDesc& d) {
  const loader::DFlashCheckpoint ck(s.draft_model);
  model::check_muse_assistant(ck.desc());
  ck.check_target(target_shape(d, d.vocab));
  if (s.max_k > ck.desc().max_k())
    throw std::runtime_error("--spec-max " + std::to_string(s.max_k) + " exceeds the drafter's block (" +
                             std::to_string(ck.desc().max_k()) + " drafts)");
  return runtime::dflash::drafter_plan(ck.desc(), ck.target_hidden(), d.vocab, s.format);
}

// MuseOptions::spec for the run: the verify lists at K_max + 1 rows, and with --spec dflash the drafter's term.
inline runtime::muse::MuseSpec engine_spec(const SpecArgs& s, const model::MuseGlimmerDesc& d) {
  runtime::muse::MuseSpec m;
  if (!s.on()) return m;
  m.rows = s.max_k + 1;
  if (s.dflash()) m.drafter = drafter_plan(s, d);
  return m;
}

// The AdaptiveK cost table (b70-serve): the derived Muse rows for the drafter's format and the head, or lookup's free
// drafts over the same verify rows, then --spec-cost over it.
inline server::MtpCost spec_cost(const SpecArgs& s, bool int8_head) {
  const unsigned fmt = s.format == loader::DraftFormat::Int4G64 ? 2u : s.format == loader::DraftFormat::Int8 ? 1u : 0u;
  server::MtpCost c = s.dflash() ? server::MtpCost::muse_dflash(int8_head, fmt) : server::MtpCost::muse_verify(int8_head);
  if (!s.cost.empty()) {
    try {
      c = server::MtpCost::parse(s.cost, c);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(std::string(error.what()) + " (--spec-cost)");
    }
  }
  if (c.max_k() < s.max_k)
    throw std::runtime_error("--spec-cost covers K up to " + std::to_string(c.max_k()) + ", below --spec-max " +
                             std::to_string(s.max_k));
  return c;
}

// The drafter beside the loaded engine: its weights (load_drafter by the target's shape, over the engine's own
// embedding), its buffers at the engine's max_len, attached to the engine (verify taps, commit projection, prefill
// tail, reset, snapshots) and its draft list captured over the engine's embedding, head and Control block.
inline std::unique_ptr<runtime::dflash::MuseDrafter> attach_drafter(runtime::muse::MuseEngine& e, const SpecArgs& s) {
  l0::Context& ctx = e.context(0);
  const model::DFlashTargetShape shape = target_shape(e.weights().desc, e.lm_head().shape.N);
  auto d = std::make_unique<runtime::dflash::MuseDrafter>(
      ctx, loader::load_drafter(ctx, s.draft_model, shape, e.embed(), s.format), e.max_len(), s.max_k, s.q);
  e.attach_drafter(*d);
  d->attach(ctx, e.embed(), e.lm_head(), e.control_mem());
  return d;
}

// The adapter's view of it: one block at the engine's pos, replayed on the engine's queue.
inline MuseDraftHooks draft_hooks(runtime::muse::MuseEngine& e, runtime::dflash::MuseDrafter& d) {
  MuseDraftHooks h;
  h.draft = [&e, &d](uint32_t k) -> const runtime::dflash::MuseDraftOut& {
    return d.draft([&e](l0::CmdList& list) { e.replay_aux(list); }, k);
  };
  return h;
}

}  // namespace cli::muse
