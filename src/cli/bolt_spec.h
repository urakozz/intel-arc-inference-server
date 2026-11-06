#pragma once
// Spec 26c: speculative decoding for Nemotron 3.5 Bolt - b70-serve on Bolt's cards (two by default), selected by the one
// --spec surface (cli/spec_args.h, SpecFamily::Bolt; the zero-config cleanup of 2026-11-06 left no other knob):
//
//   --spec dflash          the proposer is LiLiCorr (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr, spec 26) on the LAST
//                          card beside the target's lm_head (Decision 8: with a 1.07 GB embedding replica there): one 8-row
//                          block drafts up to 7 ids, verified at M = K + 1 by Bolt's verify lists (runtime::bolt::BoltEngine)
//   --draft-model PATH     the drafter: a snapshot directory or a repo id in the HF cache (default the published one); never
//                          downloaded here
//   --spec lookup          the proposer is the server's prompt lookup (spec 19e), verified through the same lists
//   --spec-max K           the deepest block, 1..7 (default 7, Decision 5); AdaptiveK picks K per iteration
//
// What the engine decides (no flag, no environment variable): the drafter's linears are int8 per row (spec 26 §3; PROVISIONAL
// until box row 65 - r65.d2's acceptance per format with r63.cost's cost; `exact` and int4 g64 stay SpecArgs::format, the
// tests' seam), the head exact in every format (Decision 3); the sampled draft distribution is vLLM's lattice (softmax(realized
// lattice / T), Decision 6; lattice+lp stays SpecArgs::q, the tests' seam); the verify's routed experts are the union-aware
// form (Decision 10 - the plain M-row form, its bitwise reference, is the tests' `plain` argument and bolt_moe_union_test
// --time's arm); the AdaptiveK cost table is the derived Bolt rows (server::MtpCost::bolt_lilicorr / bolt_verify, spec 26 §11 -
// r63.cost measures them). `--spec dflash` is NOT Bolt's `auto`: it becomes so only on D5's rule (spec 26 §7: >= 10 % over both
// off and lookup on the agentic set, nowhere worse than 3 %) - box row 65's r65.benchy decides.
//
// Refused before the device, each by name (tests/server/bolt_serve_args_test.cc, tests/runtime/bolt_spec_plan_test.cc; through
// the binary the box's cli_reject_serve_bolt_*): by cli::resolve_spec --spec mtp, --spec dflash-lookup, --spec-max outside 1..7,
// --spec-max with --spec off, --draft-model without --spec dflash; by check_serve_flags --draft-vocab (LiLiCorr reads the whole
// lm_head); here --pipeline-handoff peer with a proposer (the taps ride on the copy hand-off's barrier, 19d). By the loader: a
// drafter that is not LiLiCorr's shapes or does not fit Bolt (model::check_lilicorr, check_target).
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "cli/bolt_decode.h"
#include "cli/bolt_serve.h"
#include "cli/spec_args.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/bolt_repack.h"
#include "loader/dflash_layout.h"
#include "loader/lilicorr_loader.h"
#include "loader/lilicorr_repack.h"
#include "loader/safetensors.h"
#include "model/bolt.h"
#include "model/dflash.h"
#include "runtime/bolt/bolt_engine.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/dflash/lilicorr_drafter.h"
#include "runtime/dflash/lilicorr_sizes.h"
#include "runtime/pipeline_plan.h"   // PpHandoff
#include "server/adaptive_k.h"

namespace cli::bolt {

inline constexpr const char* kDefaultDrafter = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr";
inline constexpr uint32_t kSpecMaxK = 7;

struct SpecArgs {
  std::string spec = "off";        // off | dflash | lookup (cli::resolve_spec's SpecKind for SpecFamily::Bolt)
  std::string draft_model = kDefaultDrafter;
  uint32_t max_k = kSpecMaxK;
  bool have_model = false;
  // The engine's choices (the header's list) - not user-facing: the tests' seams for the other arms.
  loader::DraftFormat format = loader::DraftFormat::Int8;
  BoltDraftQ q = BoltDraftQ::Lattice;
  bool dflash() const { return spec == "dflash"; }
  bool lookup() const { return spec == "lookup"; }
  bool on() const { return dflash() || lookup(); }
};

// b70-serve's --spec / --spec-max / --draft-model for Bolt: cli::resolve_spec's refusals and K (SpecFamily::Bolt).
inline SpecArgs spec_args(const std::string& spec_arg, bool have_max, uint32_t max, bool have_model, const std::string& model) {
  const Spec r = resolve_spec(spec_arg, SpecFamily::Bolt, "Nemotron 3.5 Bolt", have_max, max, have_model);
  SpecArgs s;
  s.spec = r.kind == SpecKind::DFlash ? "dflash" : r.kind == SpecKind::Lookup ? "lookup" : "off";
  if (s.on()) s.max_k = r.max_k;
  s.have_model = have_model;
  if (have_model) s.draft_model = model;
  return s;
}

// The refusals that need the placement (the header's list), after spec_args.
inline void check_spec(const SpecArgs& s, uint32_t devices, runtime::PpHandoff handoff) {
  if (s.on() && (s.max_k == 0 || s.max_k > kSpecMaxK))
    throw std::runtime_error("--spec-max with Nemotron 3.5 Bolt expects 1..7 (LiLiCorr's block of 8 - 1, spec 26 Decision 5)");
  if (s.on() && devices == 2 && handoff != runtime::PpHandoff::Copy)
    throw std::runtime_error("--spec " + s.spec + " on Nemotron 3.5 Bolt crosses the cards by the copy hand-off only (19d: "
                             "the drafter's taps ride on its barrier) - drop --pipeline-handoff peer");
}

inline unsigned format_index(loader::DraftFormat f) {   // MtpCost::bolt_lilicorr's: 0 exact, 1 int8, 2 int4 g64
  return f == loader::DraftFormat::Exact ? 0u : f == loader::DraftFormat::Int4G64 ? 2u : 1u;
}

// The AdaptiveK cost table (b70-serve): the derived Bolt rows for the drafter's format, the head and the MoE form, or lookup's
// free drafts over the same verify rows.
inline server::MtpCost spec_cost(const SpecArgs& s, bool int8_head, runtime::bolt::VerifyMoe moe) {
  const bool un = moe == runtime::bolt::VerifyMoe::Union;
  const server::MtpCost c = s.dflash() ? server::MtpCost::bolt_lilicorr(int8_head, format_index(s.format), un)
                                       : server::MtpCost::bolt_verify(int8_head, un);
  if (c.max_k() < s.max_k)
    throw std::runtime_error("the Bolt cost table covers K up to " + std::to_string(c.max_k()) + ", below --spec-max " +
                             std::to_string(s.max_k));
  return s.lookup() ? c.with_free_drafts() : c;
}

inline model::DFlashTargetShape target_shape(const model::BoltDesc& d) { return {"Nemotron 3.5 Bolt", d.layers, d.hidden, d.vocab}; }

// The drafter's descriptor before the device - its checkpoint alone (config.json, the safetensors header), held to LiLiCorr's
// shapes and to the target `d` (the descriptor the engine will load).
inline model::DFlashDesc drafter_desc(const SpecArgs& s, const model::BoltDesc& d) {
  const loader::LiliCheckpoint ck(s.draft_model);
  ck.check_target(target_shape(d));
  if (s.max_k > ck.desc().max_k())
    throw std::runtime_error("--spec-max " + std::to_string(s.max_k) + " exceeds the drafter's block (" +
                             std::to_string(ck.desc().max_k()) + " drafts)");
  return ck.desc();
}

// The plan's spec terms (cli::bolt::DecodeArgs' - settle() adds them to reserve_for): the verify lists on every card (their
// scratch, the union's rows, the verify link - and the 7 SSM slots through BoltReserve::verify_slots), and with --spec dflash on
// the last card the drafter at the length bound (weights + buffers: runtime::dflash::lili_drafter_bytes), its prefill staging
// and the embedding replica (1,073,741,824 B for Bolt).
inline void plan_terms(const SpecArgs& s, const model::BoltDesc& d, uint32_t bound, DecodeArgs& a) {
  a.verify_lists = s.on();
  if (!s.dflash()) return;
  const model::DFlashDesc dd = drafter_desc(s, d);
  const runtime::bolt::BoltReserve r = runtime::dflash::lili_bolt_reserve(dd, d.hidden, d.vocab, s.format, bound);
  a.drafter = r.drafter + runtime::bolt::prefill_taps_bytes(dd.window.at(0), uint32_t(dd.taps.size()), d.hidden);
  a.drafter_replica = r.drafter_embed_replica;
}

// The drafter beside the loaded engine (whose verify lists are on): the embedding replica on the last card from the Bolt
// checkpoint's own rows (mmapped), the weights in SpecArgs::format (load_lilicorr: its own checks), its buffers at the engine's
// max_len, attached to the engine (verify taps, commit projection, prefill tail, reset, snapshots) and its draft list captured
// over the replica, the engine's lm_head and the last card's Control block.
struct BoltDrafter {
  std::unique_ptr<l0::Mem> replica;
  std::unique_ptr<runtime::dflash::LiliDrafter> drafter;
};
inline BoltDrafter attach_drafter(runtime::bolt::BoltEngine& e, const SpecArgs& s, const std::string& snapshot_dir) {
  const model::BoltDesc& d = e.model().desc;
  const uint32_t last = e.devices() - 1;
  l0::Context& ctx = e.context(last);
  BoltDrafter out;
  {
    const loader::SafetensorsSet set(snapshot_dir.empty() || snapshot_dir.back() == '/' ? snapshot_dir : snapshot_dir + "/", true);
    const std::string trunk = loader::bolt_trunk_prefix(set);
    const loader::TensorInfo& t = set.tensors().at(trunk + "embeddings.weight");
    if (t.dtype != "BF16" || t.shape != std::vector<uint64_t>{d.vocab, d.hidden})
      throw std::runtime_error(trunk + "embeddings.weight is not bf16 [vocab][hidden] - the drafter's replica reads it");
    out.replica = std::make_unique<l0::Mem>(
        loader::lilicorr_embed_replica(ctx, reinterpret_cast<const uint16_t*>(set.data(t)), d.vocab, d.hidden));
  }
  out.drafter = std::make_unique<runtime::dflash::LiliDrafter>(
      ctx, loader::load_lilicorr(ctx, s.draft_model, target_shape(d), *out.replica, s.format), e.max_len(), s.max_k);
  e.attach_drafter(*out.drafter);
  out.drafter->attach(ctx, *out.replica, e.lm_head(), e.control_mem(last));
  return out;
}

// The adapter's view of it: one block at the engine's pos through the device's greedy walk, replayed on the last card's queue.
inline BoltDraftHooks draft_hooks(runtime::bolt::BoltEngine& e, runtime::dflash::LiliDrafter& d) {
  BoltDraftHooks h;
  h.draft = [&e, &d](uint32_t k) -> const runtime::dflash::LiliDraftOut& {
    return d.draft([&e](l0::CmdList& list) { e.replay_aux(list); }, k, 0.0f, 0);
  };
  return h;
}

}  // namespace cli::bolt
