#pragma once
// `--spec auto|off|mtp|lookup|dflash|dflash-lookup` and its one cap, `--spec-max K` - the speculative proposer
// (spec 19 decision 3: one per server), shared by b70-serve and b70-decode. The zero-config cleanup (operator
// 2026-11-06) merged the old `--mtp off|K|auto` into it and dropped the tuning flags (--mtp-max, --mtp-cost,
// --spec-cost, --spec-min-match, --spec-history, --draft-format): the engine uses the measured defaults.
//
//   auto           b70-serve (operator ruling 2026-11-14, resolve_auto below): dflash on one card (and replica, a
//                  drafter per card) when the model has a DFlash drafter and it is in the HF cache; else MTP with
//                  an adaptive K (spec 8 §10) on the Qwen family (Qwen3.8, Ornith - an MTP head and its cost
//                  table); none elsewhere (K2-Horizon, Kolibri-1, Nemotron 3.5 Bolt have no head;
//                  Qwen3.8-Flash-Next's MTP is not validated; Bolt's drafter becomes its auto only on spec 26 §7's
//                  D5 rule, box row 65). b70-serve's default; b70-decode's is off (its auto is the MTP rule).
//   off            no proposer.
//   mtp            the checkpoint's MTP head: K <= --spec-max (1..3, default 3) per iteration, chosen per request
//                  from its own acceptance (the Qwen family); Qwen3.8-Flash-Next drafts a fixed K = --spec-max
//                  (its adaptive policy waits for spec 22's full-model cost numbers).
//   lookup         prompt lookup (spec 19e): the continuation of the longest earlier match (>= 3 ids) of the
//                  request's own ids, verified through the MTP verify lists (the head is loaded for them); K <=
//                  --spec-max (1..3, default 3) per iteration.
//   dflash         the DFlash drafter (spec 19c, Qwen3.8 only; --draft-model PATH, default
//                  urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound from the HF cache): K <= --spec-max (1..7,
//                  default 7) drafts a block.
//                  b70-serve on one card folds prompt lookup into it (spec 19f, operator ruling 2026-11-14):
//                  `dflash` there IS `dflash-lookup`; B70_SPEC_LOOKUP=0 turns the fold off (A/B runs).
//   dflash-lookup  dflash with prompt lookup inside it (spec 19f); a developer value, `dflash`'s one-card default.
// `lookup` and `dflash-lookup` stay accepted for b70-decode, tests and A/B runs; the user-facing values are auto, off,
// mtp and dflash.
//
// Nemotron 3.5 Bolt (spec 26c, SpecFamily::Bolt; b70-serve only - b70-decode decodes it plainly): no MTP head, but
// its own verify lists at M = K + 1 <= 8 rows, so `lookup` (prompt lookup over them) and `dflash` (the LiLiCorr
// drafter on the last card, --draft-model PATH, default nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr from
// the HF cache) both take --spec-max 1..7 (default 7), K per iteration from the request's acceptance; `mtp` and
// `dflash-lookup` (built for Qwen3.8) are refused. The drafter's format (int8) and its draft distribution (vLLM's
// lattice) are the engine's choice (cli/bolt_spec.h).
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cli {

// "auto" | "off" | "mtp" | "lookup" | "dflash" | "dflash-lookup"; anything else throws naming them.
inline std::string parse_spec(const std::string& v) {
  if (v == "auto" || v == "off" || v == "mtp" || v == "lookup" || v == "dflash" || v == "dflash-lookup") return v;
  throw std::runtime_error("--spec expects auto, off, mtp, lookup, dflash or dflash-lookup, got '" + v + "'");
}

// Which proposers a model family has: Qwen (Qwen3.8, Ornith: the MTP head and its cost table; the
// drafter for Qwen3.8), Qwen4Exp (Qwen3.8-Flash-Next: the head, fixed K), Bolt (Nemotron 3.5 Bolt: no head, its
// own verify lists to M = 8 - prompt lookup and the LiLiCorr drafter, spec 26c), None (no head, no drafter).
enum class SpecFamily { Qwen, Qwen4Exp, Bolt, None };
enum class SpecKind { Off, Mtp, Lookup, DFlash, DFlashLookup };

struct Spec {
  SpecKind kind = SpecKind::Off;
  uint32_t max_k = 0;      // the cap: the head's / lookup's / the drafter's drafts per iteration at most
  bool adaptive = false;   // K per iteration from the request's acceptance (AdaptiveK); else K = max_k
};

// --spec auto's pick for b70-serve (operator ruling 2026-11-14): DFlash where it runs and its drafter is at hand, the
// MTP head where the model has one, else off. The engine never downloads: a drafter missing from the HF cache falls
// back to MTP, and `why` names the command that fetches it.
struct AutoSetup {
  bool dflash_target = false;    // the model has a DFlash drafter (Qwen3.8)
  bool drafter_cached = false;   // its drafter (--draft-model, default the packed one) resolves in the HF cache
  std::string drafter;           // that drafter's repo id or path, for the log line
  // Where DFlash cannot run, named for the log line ("" = one card, or replica's drafter per card): tensor parallel,
  // pipeline parallel, several slots a card.
  std::string refused;
};
struct AutoPick {
  std::string value;   // "dflash", "mtp" or "off"
  std::string why;     // the startup log's reason
};
// Item 22 (DFlash under tensor parallel) lands here: when TP serves the drafter, the caller stops naming
// "tensor parallel" in AutoSetup::refused.
inline AutoPick resolve_auto(SpecFamily family, const AutoSetup& s) {
  const bool mtp = family == SpecFamily::Qwen;
  const std::string fallback = mtp ? "mtp" : "off";
  const std::string head = mtp ? "the model's MTP head" : "no proposer (no MTP head)";
  if (!s.dflash_target) return {fallback, head + "; the model has no DFlash drafter"};
  if (!s.refused.empty()) return {fallback, head + "; DFlash is not built under " + s.refused};
  if (!s.drafter_cached)
    return {fallback, head + "; the DFlash drafter " + s.drafter + " is not in the HF cache (the engine never "
                      "downloads it: `hf download " + s.drafter + "`, then restart)"};
  return {"dflash", "the DFlash drafter " + s.drafter + " (in the HF cache)"};
}

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
  if (family == SpecFamily::Bolt) {   // spec 26c: Bolt's verify lists take M = K + 1 <= 8 for both proposers
    if (v == "mtp") throw std::runtime_error(model + " has no MTP head: drop --spec mtp (its --spec auto is off)");
    if (v == "dflash-lookup")
      throw std::runtime_error("--spec dflash-lookup is built for Qwen3.8 beside its drafter (spec 19f); " + model +
                               " serves --spec dflash (the LiLiCorr drafter) or --spec lookup");
    if (have_max && (max == 0 || max > 7))
      throw std::runtime_error("--spec-max with " + model + " expects 1..7 (LiLiCorr's block of 8 - 1, spec 26 "
                               "Decision 5; the verify lists' M <= 8)");
    s.kind = v == "dflash" ? SpecKind::DFlash : SpecKind::Lookup;
    s.max_k = have_max ? max : 7;
    s.adaptive = true;
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
