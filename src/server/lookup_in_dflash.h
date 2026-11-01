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

struct LookupInDFlashOptions {
  uint32_t window = 2048;   // the drafter's window: a source must lie before pos - window
  uint32_t max_k = 7;       // the lookup verify's depth (--spec-max)
};

class LookupInDFlash {
 public:
  static constexpr uint32_t kEnterMatch = 8;     // ENTER: a match of >= 8 ids beyond the window
  static constexpr uint32_t kStayAccepted = 3;   // STAY: each lookup verify keeps >= 3 drafts ...
  static constexpr uint32_t kStayMatch = 3;      // ... and a match of >= 3 ids beyond the window remains

  explicit LookupInDFlash(const LookupInDFlashOptions& o = {});

  // A request starts in DFlash (the counters are kept: they are the server's whole run).
  void reset() { lookup_ = false; }

  // This iteration's lookup drafts, or none (a DFlash iteration). `m` holds every id consumed so far and the pending
  // one; its size() is the pending id's position + 1. Entering or leaving lookup happens here; `cap_k` is the engine's
  // bound for this iteration (0 = no bound beyond max_k).
  std::vector<uint32_t> propose(const PromptLookup& m, uint32_t cap_k = 0);
  // What the lookup iteration just proposed did: `accepted` of `proposed` drafts kept (proposed > 0). Below the STAY
  // bar the policy exits to DFlash.
  void observe(uint32_t proposed, uint32_t accepted);

  bool in_lookup() const { return lookup_; }
  const LookupInDFlashOptions& options() const { return o_; }

  // Counters since construction.
  uint64_t enters = 0, exits = 0, iterations = 0, drafted = 0, accepted = 0;

 private:
  LookupInDFlashOptions o_;
  bool lookup_ = false;
};

}  // namespace server
