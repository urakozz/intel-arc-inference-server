#pragma once
// `--spec auto|off|mtp|dflash` and its one cap, `--spec-max K` - the speculative proposer
// (spec 19 decision 3: one per server), shared by b70-serve and b70-decode. The zero-config cleanup (
// 2026-11-06) merged the old `--mtp off|K|auto` into it and dropped the tuning flags (--mtp-max, --mtp-cost,
// --spec-cost, --spec-min-match, --spec-history, --draft-format): the engine uses the measured defaults.
//
//   auto           MTP with an adaptive K (spec 8 §10) on the Qwen family (Qwen3.8, Ornith - an MTP head and its
//                  cost table;  no extra VRAM by default, DFlash only when asked) and on
//                  Qwen3.8-Flash-Next (the head beat plain on real text, its own cost table);
//                  none elsewhere (Kolibri-1 has no head). b70-serve's default; b70-decode's is off (its auto
//                  is the MTP rule; Qwen3.8-Flash-Next decodes plain there).
//   off            no proposer.
//   mtp            the checkpoint's MTP head: K <= --spec-max (1..3, default 3) per iteration, chosen per request
//                  from its own acceptance (the Qwen family and Qwen3.8-Flash-Next).
//   dflash         the DFlash drafter (spec 19c, Qwen3.8 only; --draft-model PATH, required: no default
//                  drafter, cli/dflash_args.h check_flags says which two): K <= --spec-max (1..7,
//                  default 7) drafts a block.
//                  b70-serve on one card folds prompt lookup into it (spec 19f, ruling 2026-11-14), as it does
//                  into the MTP head: prompt lookup is on inside every proposer, with no value or switch of its own.
//
// Kolibri-1 (Kolibri DSpark, SpecFamily::Kolibri; b70-serve only): no MTP head, its own verify lists at M = K + 1 <= 8
// rows; `dflash` with --draft-model <a DSpark drafter> (intuit/Kolibri-1-dspark; no default: the drafter's config picks
// DSpark) takes --spec-max 1..7 (default 7), K per iteration from the request's acceptance (cli/kolibri_spec.h).
// `mtp` is refused; --spec auto stays off.
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cli {

// "auto" | "off" | "mtp" | "dflash"; anything else throws naming them.
inline std::string parse_spec(const std::string& v) {
  if (v == "auto" || v == "off" || v == "mtp" || v == "dflash") return v;
  throw std::runtime_error("--spec expects auto, off, mtp or dflash, got '" + v + "'");
}

// Which proposers a model family has: Qwen (Qwen3.8, Ornith: the MTP head and its cost table; the
// drafter for Qwen3.8), Qwen4Exp (Qwen3.8-Flash-Next: the head and its own cost table), Kolibri (Kolibri-1: no head, its own
// verify lists to M = 8 - the DSpark drafter), None (no head, no drafter).
enum class SpecFamily { Qwen, Qwen4Exp, Kolibri, None };
enum class SpecKind { Off, Mtp, DFlash };

struct Spec {
  SpecKind kind = SpecKind::Off;
  uint32_t max_k = 0;      // the cap: the head's / lookup's / the drafter's drafts per iteration at most
  bool adaptive = false;   // K per iteration from the request's acceptance (AdaptiveK); else K = max_k
};

// `model` names the model in a refusal ("Kolibri-1", ...).
inline Spec resolve_spec(const std::string& arg, SpecFamily family, const std::string& model, bool have_max,
                         uint32_t max, bool have_draft_model) {
  std::string v = arg;
  if (v == "auto") v = family == SpecFamily::Qwen || family == SpecFamily::Qwen4Exp ? "mtp" : "off";
  const bool dflash = v == "dflash";
  if (have_draft_model && !dflash) throw std::runtime_error("--draft-model needs --spec dflash");
  Spec s;
  if (v == "off") {
    if (have_max) throw std::runtime_error("--spec-max caps a proposer's drafts; --spec off has none");
    return s;
  }
  if (family == SpecFamily::Kolibri) {   // Kolibri DSpark: the DSpark drafter over Kolibri's verify lists (M <= 8)
    if (v == "mtp") throw std::runtime_error(model + " has no MTP head: drop --spec mtp (its --spec auto is off)");
    if (have_max && (max == 0 || max > 7))
      throw std::runtime_error("--spec-max with " + model + " expects 1..7 (DSpark's block of 8 - 1; the verify lists' "
                               "M <= 8)");
    s.kind = SpecKind::DFlash;
    s.max_k = have_max ? max : 7;
    s.adaptive = true;
    return s;
  }
  if (family == SpecFamily::None) {
    if (v == "mtp") throw std::runtime_error(model + " has no MTP head: drop --spec mtp (its --spec auto is off)");
    throw std::runtime_error("b70-decode runs " + model + " plainly: drop --spec " + v +
                             " (b70-serve serves --spec dflash for Kolibri-1 with its DSpark drafter)");
  }
  if (dflash) {
    if (family != SpecFamily::Qwen)
      throw std::runtime_error("--spec " + v + " is built for Qwen3.8 beside its drafter; " + model +
                               " has none");
    if (have_max && (max == 0 || max > 7))
      throw std::runtime_error("--spec-max with --spec " + v + " expects 1..7 (the drafter's block of 8)");
    s.kind = SpecKind::DFlash;
    s.max_k = have_max ? max : 7;
    s.adaptive = true;
    return s;
  }
  if (have_max && (max == 0 || max > 3))
    throw std::runtime_error("--spec-max with --spec " + v + " expects 1..3 (the verify lists' M <= 4)");
  s.max_k = have_max ? max : 3;
  s.kind = SpecKind::Mtp;
  s.adaptive = true;   // the Qwen family and Qwen3.8-Flash-Next: the families above returned
  return s;
}

}  // namespace cli
