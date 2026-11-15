#pragma once
// Spec 19f (spec 19 §13.1, `b70-serve --spec dflash-lookup`): prompt lookup inside DFlash - the policy. Pure host
// code over the server's matcher (server::PromptLookup, which holds the prompt and every id consumed so far, the
// pending one included); unit-tested by tests/server/lookup_in_dflash_test.cc.
//
// DFlash stays the proposer; lookup takes over only where it is near-certain and the drafter is blind:
//   - ENTER lookup when the matcher finds a continuation match of >= kEnterMatch (8) ids whose source lies BEYOND the
//     drafter's window (source position < pos - window, pos = the pending id's position; the drafter's ring holds
//     positions pos - window .. pos - 1, so a copy from inside it is DFlash's own business);
//   - STAY while each lookup verify keeps >= kStayAccepted (3) drafts (or every draft, when fewer than 3 were
//     verified: a match cut short by a boundary or the engine's max_verify_k is not a miss) AND a match of
//     >= kStayMatch (3) ids beyond the window still exists; EXIT to DFlash otherwise - and re-entering needs 8 again
//     (the published policy's ENTER 8 / HOT 3 / STAY 3: "any match >= 3" loses on free generation, so a short match
//     only continues a copy that is already paying);
//   - full-depth lookup verify: K = max_k (the server's --spec-max, the drafter's deepest K), capped by the drafts the
//     match has (PromptLookup::propose stops at a boundary) - not AdaptiveK's DFlash-tuned K.
// The constants are the published policy's (sly-radiance; spec 19 §13.1), code constants by the plan (no flags); the
// window is the drafter's own (runtime::TapDrafter::window(), 2048 for Qwen3.8's DFlash2), so a second drafter with
// another window needs no change here.
//
// What the engine does with the answer (EngineIface::step_drafts_or_many): drafts -> they are verified on the tap
// verify list (the drafter's draft pass skipped, its projection of the committed rows kept, so the ring is warm at the
// exit); no drafts -> one DFlash iteration.
#include <cstdint>
#include <vector>

#include "server/prompt_lookup.h"

namespace server {

// Open-work 28 (lookup inside MTP on one card, the engine's wide verify list): long runs go wide.
//   - HOT: the last lookup verify kept every draft it checked (or, wide, >= kWideMin of them): the next one asks the
//     matcher for up to wide_k drafts and, when the source continues for >= kWideMin, checks them all in one wide verify
//     (fewer: the usual max_k);
//   - a short match the drafter confirms (candidate() / confirmed()): outside lookup, the continuation of the context's
//     longest match of any length >= 2, when it holds >= kWideMin drafts; the engine adapter checks it against the MTP
//     head's first two drafts and, when they agree, verifies the candidate wide and calls confirmed() - lookup is
//     entered hot. (Step 1's recordings: after the head's two drafts agree, the source continues 4-9 ids on average;
//     a suffix match alone, below 8 ids, does not pay against MTP.)
// wide_k 0 (the default, every engine without the wide list) is the policy above unchanged.
struct LookupInDFlashOptions {
  uint32_t window = 2048;   // the drafter's window: a source must lie before pos - window
  uint32_t max_k = 7;       // the lookup verify's depth (--spec-max)
  uint32_t wide_k = 0;      // open-work 28: the engine's wide verify depth (runtime::Engine::kWideK), 0 = none
};

class LookupInDFlash {
 public:
  static constexpr uint32_t kEnterMatch = 8;     // ENTER: a match of >= 8 ids beyond the window
  static constexpr uint32_t kStayAccepted = 3;   // STAY: each lookup verify keeps >= 3 drafts ...
  static constexpr uint32_t kStayMatch = 3;      // ... and a match of >= 3 ids beyond the window remains
  static constexpr uint32_t kWideMin = 8;        // open-work 28: a wide verify needs >= 8 drafts (and keeps it hot)
  static constexpr uint32_t kConfirmMatch = 2;   // ... a confirmed candidate's shortest match

  explicit LookupInDFlash(const LookupInDFlashOptions& o = {});

  // A request starts in DFlash (the counters are kept: they are the server's whole run).
  void reset() { lookup_ = hot_ = false; }

  // This iteration's lookup drafts, or none (a DFlash iteration). `m` holds every id consumed so far and the pending
  // one; its size() is the pending id's position + 1. Entering or leaving lookup happens here; `cap_k` is the engine's
  // bound for this iteration (0 = no bound beyond max_k).
  std::vector<uint32_t> propose(const PromptLookup& m, uint32_t cap_k = 0);
  // What the lookup iteration just proposed did: `accepted` of `proposed` drafts kept (proposed > 0). Below the STAY
  // bar the policy exits to DFlash.
  void observe(uint32_t proposed, uint32_t accepted);
  // Open-work 28: outside lookup, the continuation a drafter may confirm (empty below kWideMin drafts or without
  // wide_k); confirmed() enters lookup hot - the iteration's observe() follows as for any lookup verify.
  std::vector<uint32_t> candidate(const PromptLookup& m) const;
  void confirmed();

  bool in_lookup() const { return lookup_; }
  const LookupInDFlashOptions& options() const { return o_; }

  // Counters since construction.
  uint64_t enters = 0, exits = 0, iterations = 0, drafted = 0, accepted = 0;
  uint64_t wide = 0, confirms = 0;   // open-work 28: wide verifies, confirmed entries

 private:
  LookupInDFlashOptions o_;
  bool lookup_ = false;
  bool hot_ = false;   // open-work 28
};

}  // namespace server
