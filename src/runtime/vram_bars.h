#pragma once
// The startup log's VRAM chart: one block per card the run uses, printed under the memory-plan line, and the host
// memory the model pins. Host text only (tests/runtime/vram_bars_test.cc), fed from each model's own plan:
//
//   VRAM, card 0 of 2                                       32.5 GB
//   ████████████████████████████████▓▓▓▓▓▓▒▒░░░░░▚▚▚▞▞··········|||
//   weights 18.2 GB, KV cache 3.4, MTP head 0.9, prefill scratch 0.9, free 4.8, reserve 1.5
//
// The bar is kVramCells cells scaled to the card's total: every used part and the reserve round(bytes / total x 64)
// cells, at least 1 when above zero; free takes the cells that remain, so every card's bar is 64 cells and they line
// up (when rounding leaves free none, or there is no free, the largest used part gives or takes the difference).
// Neighbouring used parts cycle █ ▓ ▒ ░ ▚ ▞; free is ·, the reserve |. Outside a UTF-8 locale: # and = alternating,
// . and |, and "x" for "×". GB = 1e9 B, one decimal, " GB" on the breakdown's first number only (the host line
// names it on each).
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "runtime/memory_plan.h"

namespace runtime {

inline constexpr int kVramCells = 64;
inline constexpr size_t kVramWrap = 90;   // the breakdown's line width

struct VramPart {
  std::string name;
  size_t bytes = 0;
};

// One card: its used parts in the bar's order (zero parts are left out), the reserve and the device total. Free is
// total - used - reserve, 0 when they pass the total.
struct VramCard {
  std::vector<VramPart> used;
  size_t reserve = 0, total = 0;
};

// The used parts of memory_line()'s components, in the bar's order: weights (the draft vocabulary's compact head too,
// unless a drafter drafts with it), KV cache, `state` (decode_state: the MTP head and the per-slot decode state),
// `drafter` (its weights and buffers, with the draft vocabulary), the l0-int8 prefill state, prefill scratch.
std::vector<VramPart> vram_parts(const MemoryComponents& c, const std::string& state, const std::string& drafter);
// `state` for a one-card or tensor-parallel plan: "MTP × N slots" / "MTP head" with the head, else "decode state"
// (× N slots beside sequence slots).
std::string vram_state_name(const MemoryPlan& p);

// The cells per part: the used parts, then free and the reserve (each omitted when 0), summing to kVramCells.
std::vector<int> vram_cells(const VramCard& c);

// Whether LC_ALL / LC_CTYPE / LANG (the first one set) names UTF-8.
bool vram_utf8();

// "VRAM, card <card> of <cards>", the bar and the breakdown, newline-terminated.
std::string vram_block(const VramCard& c, uint32_t card, uint32_t cards, bool utf8);
// "host memory: experts 16.7 GB, PLE table 51.8 GB (pinned)" over the non-zero parts; "" when none.
std::string vram_host_line(const std::vector<VramPart>& host);

}  // namespace runtime
