#pragma once
// The parallel mode and the pipeline's override flags, shared by b70-serve and b70-decode, as
// device-free parsing and refusals (tests/cli/pipeline_args_test.cc runs them on any host; the
// device flows are cli/pipeline_decode.h and b70_serve.cc's serve_pipeline):
//
//   --parallel auto|1|pp        auto (default): the faster choice by fit - one card when the model
//                               and the asked context fit it, else pipeline parallel (each model
//                               family's rule: cli/pipeline_settle.h parallel_auto, Kolibri-1's and
//                               Nemotron 3.5 Bolt's two cards for the whole model); 1: one card;
//                               pp: the model's layers split over GPUs 0 and 1 of what Level Zero
//                               shows (spec 16).
//                               tp (Qwen3.8 - spec 17b-d): tensor parallel, every layer's linears
//                               split over GPUs 0 and 1 (half the heads, half the MLP, half the
//                               head's vocabulary a card), the two halves added after each layer
//                               by P2P writes; needs peer access both ways. Prefill by the
//                               wavefront (17c), --spec mtp / auto / lookup with the MTP head whole
//                               on each card, b70-serve with the prefix cache (17d); no DFlash, no
//                               draft vocabulary, no l0-int8 prefill (refused by name).
//                               replica (b70-serve, Qwen3.8 only - spec 13 §11, plan 13b): the model
//                               on both cards, one engine each, two requests served at once (each
//                               bitwise what it is alone on its card); every other model and
//                               b70-decode refuse it by name, one visible card is refused.
//   --max-num-seqs N            b70-serve: requests at once (check_max_num_seqs / resolve_max_num_seqs below).
//   --pipeline-split auto|N     the override, for experiments: device 0 runs layers [0, N), device 1
//                               [N, layers); auto (default) balances the bytes each card holds
//                               (runtime::pp_auto_split). Given under --parallel auto, it means pp.
//   --pipeline-handoff copy|peer  how the residual crosses (runtime::PpHandoff); copy (default) or
//                               peer. Given under --parallel auto, it means pp.
//
// (The zero-config cleanup, operator 2026-11-06, replaced `--pp N` / `--pipeline-parallel-size N`
// with --parallel and dropped the refusals of the 2026-10-06 rename's old spellings.)
#include <cstdint>
#include <stdexcept>
#include <string>

#include "runtime/buffer_sizes.h"
#include "runtime/pipeline_plan.h"
#include "runtime/prefill_backend.h"

namespace cli {

// The parallel mode as given; `devices` is the resolved device count (1 or 2).
enum class Parallel { Auto, One, Pp, Replica, Tp };

struct PipelineArgs {
  Parallel mode = Parallel::Auto;
  uint32_t devices = 1;
  bool split_auto = true;
  uint32_t split = 0;   // the explicit N; 0 with split_auto
  runtime::PpHandoff handoff = runtime::kDefaultPpHandoff;
  bool have_split = false, have_handoff = false;
  bool on() const { return devices > 1; }   // pipeline parallel (replica keeps devices 1: one engine per card)
  bool replica() const { return mode == Parallel::Replica; }
  bool tp() const { return mode == Parallel::Tp; }   // spec 17b (devices stays 1: on() is pipeline parallel's)
  // The mode was named (1 or pp) - each family's default applies otherwise.
  bool given() const { return mode != Parallel::Auto; }
};

inline void parse_parallel(const std::string& v, PipelineArgs& a) {
  if (v == "auto") {
    a.mode = Parallel::Auto;
    a.devices = 1;
  } else if (v == "1") {
    a.mode = Parallel::One;
    a.devices = 1;
  } else if (v == "pp") {
    a.mode = Parallel::Pp;
    a.devices = 2;
  } else if (v == "replica") {   // plan 13b: one engine per card (b70-serve, Qwen3.8)
    a.mode = Parallel::Replica;
    a.devices = 1;
  } else if (v == "tp") {   // spec 17b: tensor parallel (b70-decode, Qwen3.8)
    a.mode = Parallel::Tp;
    a.devices = 1;
  } else {
    throw std::runtime_error("--parallel expects auto, 1, pp, replica or tp, got '" + v + "'");
  }
}

// Spec 17b: --parallel tp's refusals that need no device. What 17b / 17c do not run under tensor parallel is refused
// by name, with the plan that builds it (or why not).
struct TpContext {
  bool prefill_int8 = false;   // --prefill-backend l0-int8 (spec 17c prefills on l0)
  bool dflash = false;         // --spec dflash / dflash-lookup (spec 17d: not built under TP)
  bool profile = false;        // --profile
  bool other_family = false;   // Bolt / Kolibri-1 / Qwen3.8-Flash-Next / K2-Horizon
  bool draft_vocab = false;    // --draft-vocab / --draft-vocab-ids (spec 17d: the draft head is the card's half)
};
inline void check_tp(const PipelineArgs& a, const TpContext& c) {
  if (!a.tp()) return;
  if (a.have_split)
    throw std::runtime_error("--pipeline-split belongs to --parallel pp (the layer device 1 starts at); --parallel tp "
                             "splits every layer");
  if (a.have_handoff)
    throw std::runtime_error("--pipeline-handoff belongs to --parallel pp; --parallel tp's cards add their halves by "
                             "P2P writes");
  if (c.other_family)
    throw std::runtime_error("--parallel tp is built for Qwen3.8 (spec 17b); this model runs --parallel auto, 1 or pp");
  if (c.prefill_int8)
    throw std::runtime_error("--parallel tp prefills on the l0 backend only (spec 17c): l0-int8's activation quantiser "
                             "rotates 1024-k blocks and a card's down has K 8704; drop --prefill-backend l0-int8");
  if (c.dflash)
    throw std::runtime_error("--parallel tp drafts with the MTP head (--spec auto, mtp or lookup - spec 17d); --spec "
                             "dflash / dflash-lookup under tensor parallel are not built - one card or --parallel pp");
  if (c.draft_vocab)
    throw std::runtime_error("--parallel tp's draft head is each card's vocabulary half of lm_head (spec 17d); "
                             "--draft-vocab is not built under tensor parallel - drop it");
  if (c.profile) throw std::runtime_error("--profile instruments one device's list; it has no --parallel tp");
}
// Spec 17d: b70-serve --parallel tp's model, by config.json's architecture before any device (Qwen3.8 only: its per-card
// descriptor is the one built, model::tp_card_desc).
inline void check_tp_model(const std::string& architecture, const std::string& model) {
  if (architecture == "Qwen3_5ForConditionalGeneration") return;
  throw std::runtime_error("--parallel tp is built for Qwen3.8 (spec 17b-d); " +
                           (model.empty() ? std::string("this checkpoint (no readable config.json)") : model) +
                           " runs --parallel auto, 1 or pp");
}
// Two cards visible, before anything is loaded (peer access both ways is checked on the opened devices).
inline void require_two_devices_for_tp(uint32_t gpus) {
  if (gpus < 2)
    throw std::runtime_error("--parallel tp needs two GPUs and Level Zero shows " + std::to_string(gpus) +
                             " - is ZE_AFFINITY_MASK hiding the second card (docs/10-the-box.md)?");
}

// The override flags under --parallel auto mean pipeline parallel (they have no meaning on one card).
inline void resolve_overrides(PipelineArgs& a) {
  if (a.mode == Parallel::Auto && (a.have_split || a.have_handoff)) {
    a.mode = Parallel::Pp;
    a.devices = 2;
  }
}

inline void parse_pipeline_split(const std::string& v, PipelineArgs& a) {
  a.have_split = true;
  if (v == "auto") {
    a.split_auto = true;
    a.split = 0;
    return;
  }
  if (v.empty() || v.size() > 5 || v.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("--pipeline-split expects auto or a layer index, got '" + v + "'");
  const unsigned long n = std::stoul(v);
  if (n == 0)
    throw std::runtime_error("--pipeline-split 0 would leave device 0 no layer; the split is the"
                             " first layer of device 1");
  a.split_auto = false;
  a.split = static_cast<uint32_t>(n);
}

inline runtime::PpHandoff parse_pipeline_handoff(const std::string& v, PipelineArgs& a) {
  a.have_handoff = true;
  if (!runtime::parse_pp_handoff(v, a.handoff))
    throw std::runtime_error("--pipeline-handoff expects copy or peer, got '" + v + "'");
  return a.handoff;
}

// What else b70-decode was asked for, for the refusals below.
struct PipelineContext {
  bool prefill = false;      // --prefill
  bool bench_prefill = false;   // --bench --prefill-length N
  bool mtp = false;          // --spec mtp (spec 16d: allowed with pipeline parallel)
  bool profile = false;      // --profile
};

// Every refusal that needs no device. One card refuses nothing; the two override flags belong
// to --parallel pp.
inline void check_pipeline(const PipelineArgs& a, const PipelineContext& c) {
  if (!a.on()) {
    if (a.have_split)
      throw std::runtime_error("--pipeline-split belongs to --parallel pp (the layer device 1 starts at)");
    if (a.have_handoff)
      throw std::runtime_error("--pipeline-handoff belongs to --parallel pp (how the residual crosses)");
    return;
  }
  // Spec 16c: --prefill and --bench --prefill-length run the two-card chunk pipeline (the flash
  // attention, the only prefill attention).
  // (Spec 16d lifted 16b's --spec mtp refusal: the head, its embedding replica and the draft lists
  // run on device 1, the verify lists on both.)
  if (c.profile)
    throw std::runtime_error("--profile instruments one device's list; it has no --parallel pp");
}

// Plan 13b: --parallel replica's refusals that need no device.
// The architecture (config.json's architectures[0]) of the one model with a replica engine.
inline bool replica_architecture(const std::string& architecture) {
  return architecture == "Qwen3_5ForConditionalGeneration";   // Qwen3.8 (model::qwen38)
}
// `model` names the checkpoint in the refusal ("" when its config.json could not be read).
inline void check_replica_model(const std::string& architecture, const std::string& model) {
  if (replica_architecture(architecture)) return;
  throw std::runtime_error("--parallel replica serves Qwen3.8 only (spec 13b: one engine per card, two requests at "
                           "once); " + (model.empty() ? std::string("this checkpoint (no readable config.json)") : model) +
                           " has no replica engine - --parallel auto, 1 or pp");
}
// The override flags belong to pp; DFlash per card is spec 19's (not built for replica).
inline void check_replica_flags(const PipelineArgs& a, bool dflash) {
  if (!a.replica()) return;
  if (a.have_split)
    throw std::runtime_error("--pipeline-split belongs to --parallel pp (the layer device 1 starts at); --parallel "
                             "replica loads the whole model on each card");
  if (a.have_handoff)
    throw std::runtime_error("--pipeline-handoff belongs to --parallel pp (how the hidden state crosses); --parallel "
                             "replica's cards exchange nothing");
  if (dflash)
    throw std::runtime_error("--parallel replica serves --spec off, mtp or lookup (spec 13b); --spec dflash / "
                             "dflash-lookup run on one card or --parallel pp");
}
// b70-decode runs one sequence: two requests at once is the server's.
inline void refuse_replica_in_decode(const PipelineArgs& a) {
  if (a.replica())
    throw std::runtime_error("--parallel replica is b70-serve's (two requests at once, one engine per card); "
                             "b70-decode runs one sequence - --parallel auto, 1 or pp");
}
// Two cards visible (no peer access needed: the engines share nothing on the devices).
inline void require_two_devices_for_replica(uint32_t gpus) {
  if (gpus < 2)
    throw std::runtime_error("--parallel replica needs two GPUs (one engine per card) and Level Zero shows " +
                             std::to_string(gpus) +
                             " - is ZE_AFFINITY_MASK hiding the second card (docs/10-the-box.md)?");
}

// --max-num-seqs N (operator 2026-11-11; vLLM's flag, the same meaning): how many requests run at once - the user's
// choice, and the context follows (slot 0 the largest that fits beside N - 1 slots of 32768 on its card,
// cli::settle_slots). N = 1 (the default): one slot. N > 1: two cards run replica, each card its slots (card 0 the
// odd one), one card its slots (plan 13c); at most kSlotsPerCard slots a card. Qwen3.8 only (b70_serve.cc refuses
// the others by config.json, DFlash and a draft vocabulary beside slots).
inline constexpr uint32_t kSlotsPerCard = runtime::SlotDims::kMaxSeqs;
// Before the device: --parallel tp keeps one slot (TP + slots was ruled out, 2026-11-11) and pp runs one request;
// replica runs a request on each card, so N is 2 to 8 there and 2 (one slot a card) when not given. Returns N.
inline uint32_t check_max_num_seqs(const PipelineArgs& a, uint32_t n, bool given) {
  if (a.replica()) {
    if (!given) return 2;
    if (n < 2)
      throw std::runtime_error("--parallel replica runs a request on each card: --max-num-seqs 2 to " +
                               std::to_string(2 * kSlotsPerCard) +
                               " (2: one full-context slot a card); one request at a time is --parallel auto");
  }
  if (n <= 1) return n;
  if (a.tp())
    throw std::runtime_error("--parallel tp runs one request at a time (tensor parallel with slots was ruled out); "
                             "--max-num-seqs " + std::to_string(n) +
                             " runs on --parallel replica (two cards, each its slots) - or --parallel auto");
  if (a.mode == Parallel::Pp)
    throw std::runtime_error("--parallel pp runs one request at a time; --max-num-seqs " + std::to_string(n) +
                             " runs on --parallel replica (two cards, each its slots) or one card's slots "
                             "(--parallel 1)");
  return n;
}
// After the hardware gate (`cards` visible): --parallel auto with N > 1 is replica on two cards, one card's slots on
// one; then the limit - kSlotsPerCard a card.
inline void resolve_max_num_seqs(PipelineArgs& a, uint32_t n, uint32_t cards) {
  if (n <= 1) return;
  if (a.mode == Parallel::Auto) {
    a.mode = cards >= 2 ? Parallel::Replica : Parallel::One;
    a.devices = 1;
  }
  const uint32_t limit = a.replica() ? 2 * kSlotsPerCard : kSlotsPerCard;
  if (n > limit)
    throw std::runtime_error("--max-num-seqs " + std::to_string(n) + ": " +
                             (a.replica() ? std::string("two cards run") : std::string("one card runs")) + " at most " +
                             std::to_string(limit) + " requests at once (" + std::to_string(kSlotsPerCard) +
                             " slots a card)" +
                             (!a.replica() && cards >= 2 ? "; --parallel replica runs up to " +
                                                               std::to_string(2 * kSlotsPerCard) + " on two"
                                                         : std::string()));
}

// The device count the driver shows (l0::Context::gpu_count), held to two before anything
// is opened (P4: a missing card is a clear error, never a hang or a single-card run).
inline void require_two_devices(uint32_t gpus) {
  if (gpus < 2)
    throw std::runtime_error("--parallel pp needs two GPUs and Level Zero shows " +
                             std::to_string(gpus) +
                             " - is ZE_AFFINITY_MASK hiding the second card (docs/10-the-box.md)?");
}

}  // namespace cli
