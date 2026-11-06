#pragma once
// `--spec auto|off|mtp|lookup|dflash|dflash-lookup` and its one cap, `--spec-max K` - the speculative proposer
// (spec 19 decision 3: one per server), shared by b70-serve and b70-decode. The zero-config cleanup (operator
// 2026-11-06) merged the old `--mtp off|K|auto` into it and dropped the tuning flags (--mtp-max, --mtp-cost,
// --spec-cost, --spec-min-match, --spec-history, --draft-format): the engine uses the measured defaults.
//
//   auto           the model's validated proposer: MTP with an adaptive K (spec 8 §10, AdaptiveK over the int8
//                  head's measured cost table) on the Qwen family (Qwen3.8, Agnes, Ornith - the models with an
//                  MTP head and that table); none elsewhere (K2-Horizon, Kolibri-1, Nemotron 3.5 Bolt have no head;
//                  Qwen3.8-Flash-Next's MTP is not validated). b70-serve's default; b70-decode's is off.
//   off            no proposer.
//   mtp            the checkpoint's MTP head: K <= --spec-max (1..3, default 3) per iteration, chosen per request
//                  from its own acceptance (the Qwen family); Qwen3.8-Flash-Next drafts a fixed K = --spec-max
//                  (its adaptive policy waits for spec 22's full-model cost numbers).
//   lookup         prompt lookup (spec 19e): the continuation of the longest earlier match (>= 3 ids) of the
//                  request's own ids, verified through the MTP verify lists (the head is loaded for them); K <=
//                  --spec-max (1..3, default 3) per iteration.
//   dflash         the DFlash drafter (spec 19c, Qwen3.8 only; --draft-model PATH, default
//                  z-lab/Qwen3.8-27B-DFlash2 from the HF cache): K <= --spec-max (1..7, default 7) drafts a block.
//   dflash-lookup  dflash with prompt lookup inside it (spec 19f; opt-in until queue row 70 measures it).
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cli {

// "auto" | "off" | "mtp" | "lookup" | "dflash" | "dflash-lookup"; anything else throws naming them.
inline std::string parse_spec(const std::string& v) {
  if (v == "auto" || v == "off" || v == "mtp" || v == "lookup" || v == "dflash" || v == "dflash-lookup") return v;
  throw std::runtime_error("--spec expects auto, off, mtp, lookup, dflash or dflash-lookup, got '" + v + "'");
}

// Which proposers a model family has: Qwen (Qwen3.8, Agnes, Ornith: the MTP head and its cost table; the
// drafter for Qwen3.8), Qwen4Exp (Qwen3.8-Flash-Next: the head, fixed K), None (no head, no drafter).
enum class SpecFamily { Qwen, Qwen4Exp, None };
enum class SpecKind { Off, Mtp, Lookup, DFlash, DFlashLookup };

struct Spec {
  SpecKind kind = SpecKind::Off;
  uint32_t max_k = 0;      // the cap: the head's / lookup's / the drafter's drafts per iteration at most
  bool adaptive = false;   // K per iteration from the request's acceptance (AdaptiveK); else K = max_k
};

// `model` names the model in a refusal ("K2-Horizon", "Kolibri-1", ...).
inline Spec resolve_spec(const std::string& arg, SpecFamily family, const std::string& model, bool have_max,
                         uint32_t max, bool have_draft_model) {
  std::string v = arg;
  if (v == "auto") v = family == SpecFamily::Qwen ? "mtp" : "off";
  const bool dflash = v == "dflash" || v == "dflash-lookup";
  if (have_draft_model && !dflash) throw std::runtime_error("--draft-model needs --spec dflash");
  Spec s;
  if (v == "off") {
    if (have_max) throw std::runtime_error("--spec-max caps a proposer's drafts; --spec off has none");
    return s;
  }
  if (family == SpecFamily::None) {
    if (v == "mtp") throw std::runtime_error(model + " has no MTP head: drop --spec mtp (its --spec auto is off)");
    if (v == "lookup")
      throw std::runtime_error("--spec lookup verifies its drafts through the MTP verify lists, which " + model +
                               "'s engine does not capture (no MTP head): drop --spec lookup");
    throw std::runtime_error("--spec " + v + " is built for Qwen3.8 beside its drafter (spec 19c): no DFlash drafter "
                             "exists for " + model + " (spec 19 §1)");
  }
  if (dflash) {
    if (family != SpecFamily::Qwen)
      throw std::runtime_error("--spec " + v + " is built for Qwen3.8 beside its drafter (spec 19c); " + model +
                               " has none");
    if (have_max && (max == 0 || max > 7))
      throw std::runtime_error("--spec-max with --spec " + v + " expects 1..7 (the drafter's block of 8)");
    s.kind = v == "dflash" ? SpecKind::DFlash : SpecKind::DFlashLookup;
    s.max_k = have_max ? max : 7;
    s.adaptive = true;
    return s;
  }
  if (have_max && (max == 0 || max > 3))
    throw std::runtime_error("--spec-max with --spec " + v + " expects 1..3 (the verify lists' M <= 4)");
  s.max_k = have_max ? max : 3;
  if (v == "lookup") {
    s.kind = SpecKind::Lookup;
    s.adaptive = true;
    return s;
  }
  s.kind = SpecKind::Mtp;
  s.adaptive = family == SpecFamily::Qwen;
  return s;
}

}  // namespace cli
