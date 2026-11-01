#pragma once
// Spec 26c: speculative decoding for Nemotron 3.5 Bolt - b70-serve on Bolt's cards (two by default):
//
//   --spec dflash          the proposer is LiLiCorr (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr, spec 26) on the LAST
//                          card beside the target's lm_head (Decision 8: with a 1.07 GB embedding replica there): one 8-row
//                          block drafts up to 7 ids, verified at M = K + 1 by Bolt's verify lists (runtime::bolt::BoltEngine)
//   --draft-model PATH     the drafter: a snapshot directory or a repo id in the HF cache (default the published one); never
//                          downloaded here
//   --draft-format F       exact (W' bf16 + g: spec 26 §3), int8 (per row - the default, PROVISIONAL until r63.accept, spec 26
//                          §11) or int4 (g64 RTN); the LiLiCorr head is exact in every format (Decision 3)
//   --draft-q Q            the sampled draft distribution: lattice (softmax(realized lattice / T) - vLLM's, the default,
//                          PROVISIONAL until r63.accept) or lattice+lp (+ the candidates' log-probs, the alternative 26a
//                          records); lossless either way (D4)
//   --spec lookup          the proposer is the server's prompt lookup (spec 19e), verified through the same lists
//   --spec-max K           the deepest block, 1..7 (default 7, Decision 5); AdaptiveK picks K per iteration
//   --spec-cost SPEC       the AdaptiveK cost table (--mtp-cost's syntax, verify to M = 8, draft to 7); default
//                          server::MtpCost::bolt_lilicorr / bolt_verify (DERIVED, spec 26 §11 - r63.cost measures it)
// and B70_BOLT_VERIFY_MOE=union|plain (the verify's routed experts: the union-aware form, Decision 10, or its bitwise reference).
// `--spec dflash` is NOT the default: it becomes Bolt's only on D5's rule (spec 26 §7: >= 10 % over both off and lookup on the
// agentic set, nowhere worse than 3 %) - box row 65's r65.benchy decides.
//
// Refused before the device, each by name (tests/server/bolt_serve_args_test.cc; through the binary the box's
// cli_reject_serve_bolt_*): --draft-model / --draft-format / --draft-q without --spec dflash; --spec-max / --spec-cost without a
// proposer; --spec-max outside 1..7; --draft-format bf16 (LiLiCorr's NVFP4 linears have no exact bf16 form); --draft-vocab (no
// compact head); --mtp-max / --mtp-cost (an MTP head's); --pipeline-handoff peer with --spec (the taps ride on the copy
// hand-off's barrier, 19d); B70_PREFILL_REPLAY=1 with --spec dflash (the prefill taps are copies a recorded chunk cannot hold).
// By the loader: a drafter that is not LiLiCorr's shapes or does not fit Bolt (model::check_lilicorr, check_target).
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "cli/bolt_decode.h"
#include "cli/bolt_serve.h"
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
  std::string spec = "off";        // off | dflash | lookup (mtp: check_serve_flags refuses it)
  std::string draft_model = kDefaultDrafter;
  std::string format_arg;          // --draft-format as given ("" = the default, int8)
  std::string q_arg;               // --draft-q as given ("" = lattice)
  uint32_t max_k = kSpecMaxK;
  std::string cost;                // --spec-cost
  bool have_model = false, have_max_k = false, draft_vocab = false, mtp_tuning = false;
  // resolved by check_spec
  loader::DraftFormat format = loader::DraftFormat::Int8;
  BoltDraftQ q = BoltDraftQ::Lattice;
  bool dflash() const { return spec == "dflash"; }
  bool lookup() const { return spec == "lookup"; }
  bool on() const { return dflash() || lookup(); }
};

// Every refusal that needs no device (the header's list), in order; resolves --draft-format and --draft-q.
inline void check_spec(SpecArgs& s, uint32_t devices, runtime::PpHandoff handoff, bool prefills) {
  if (!s.dflash() && (s.have_model || !s.format_arg.empty() || !s.q_arg.empty()))
    throw std::runtime_error("--draft-model, --draft-format and --draft-q need --spec dflash (Nemotron 3.5 Bolt's drafter is "
                             "LiLiCorr, spec 26)");
  if ((s.have_max_k || !s.cost.empty()) && !s.on())
    throw std::runtime_error("--spec-max and --spec-cost need --spec dflash or --spec lookup");
  if (s.on() && (s.max_k == 0 || s.max_k > kSpecMaxK))
    throw std::runtime_error("--spec-max with Nemotron 3.5 Bolt expects 1..7 (LiLiCorr's block of 8 - 1, spec 26 Decision 5)");
  if (s.draft_vocab)
    throw std::runtime_error("--draft-vocab and --draft-vocab-ids narrow an MTP head's or a DFlash drafter's head; LiLiCorr reads "
                             "Bolt's whole lm_head (no compact head, spec 26)");
  if (s.mtp_tuning)
    throw std::runtime_error("--mtp-max and --mtp-cost belong to an MTP head; Nemotron 3.5 Bolt has none (its proposers: --spec "
                             "dflash / lookup, tuned by --spec-max and --spec-cost)");
  if (!s.format_arg.empty() && !loader::parse_lilicorr_draft_format(s.format_arg, s.format))
    throw std::runtime_error("--draft-format " + s.format_arg + ": LiLiCorr's formats are exact, int8 and int4 (spec 26 §3: its "
                             "NVFP4 linears have no exact bf16 form - `exact` is W' bf16 with g on the output)");
  if (!s.q_arg.empty() && !parse_bolt_draft_q(s.q_arg, s.q))
    throw std::runtime_error("--draft-q " + s.q_arg + ": Nemotron 3.5 Bolt's drafter samples from lattice or lattice+lp "
                             "(spec 26 Decision 6)");
  if (s.on() && devices == 2 && handoff != runtime::PpHandoff::Copy)
    throw std::runtime_error("--spec " + s.spec + " on Nemotron 3.5 Bolt crosses the cards by the copy hand-off only (19d: "
                             "the drafter's taps ride on its barrier) - drop --pipeline-handoff peer");
  if (s.dflash() && prefills) {
    const char* env = std::getenv("B70_PREFILL_REPLAY");
    if (env && std::strcmp(env, "1") == 0)
      throw std::runtime_error("--spec dflash: the drafter's prefill taps are copies, which a recorded chunk cannot hold - "
                               "unset B70_PREFILL_REPLAY");
  }
}

inline unsigned format_index(loader::DraftFormat f) {   // MtpCost::bolt_lilicorr's: 0 exact, 1 int8, 2 int4 g64
  return f == loader::DraftFormat::Exact ? 0u : f == loader::DraftFormat::Int4G64 ? 2u : 1u;
}

// The AdaptiveK cost table (b70-serve): the derived Bolt rows for the drafter's format, the head and the MoE form, or lookup's
// free drafts over the same verify rows, then --spec-cost over it.
inline server::MtpCost spec_cost(const SpecArgs& s, bool int8_head, runtime::bolt::VerifyMoe moe) {
  const bool un = moe == runtime::bolt::VerifyMoe::Union;
  server::MtpCost c = s.dflash() ? server::MtpCost::bolt_lilicorr(int8_head, format_index(s.format), un)
                                 : server::MtpCost::bolt_verify(int8_head, un);
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
// checkpoint's own rows (mmapped), the weights in --draft-format (load_lilicorr: its own checks), its buffers at the engine's
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
