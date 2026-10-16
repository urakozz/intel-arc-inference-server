#pragma once
#include <cstddef>
#include <cstdint>

#include "kernels/dflash_dims.h"

// Spec 19b: the DFlash lists' two host-visible blocks (shared memory), beside runtime::Control - which the draft list
// reads for the anchor (pos, cur_token[0]) and writes the drafts into (cur_token[1..K], what the verify list reads) and
// which nothing here changes.
//
//   DFlashCtl        what varies per draft or per projection and is not Control's: K (the drafts this replay makes,
//                    1..7 - one list captured for every K: rows past K are masked, plan 19b Task 3), the walk's
//                    temperature and 64-bit seed (spec 19 §2: Gumbel-argmax at the request's temperature), and the
//                    projection's first position and row count (19c's commit list: the committed rows j + 1).
//                    The host writes it before a replay under the drain rule (docs/04: a host write to a block the
//                    list reads is preceded by a drain).
//   DFlashDraftOut   the draft list's readback: per draft row j < K the chosen id and its index among the row's 16
//                    candidates, the candidates (real ids - Review Focus 3), their head logits (unary) and the
//                    selector's realized scores (the draft distribution q for sampled acceptance is their softmax at
//                    the temperature, 19c). Written by df_select (src/kernels/dflash/df_select.cl's OUT_* words).
namespace runtime::dflash {

struct DFlashCtl {
  uint32_t k = 0;            // DCTL_K: drafts this replay (1 .. kernels::dflash::kR)
  uint32_t proj_pos = 0;     // DCTL_PPOS: the projection's first position
  uint32_t proj_n = 0;       // DCTL_PN: the projection's rows (<= kernels::dflash::kM)
  float temperature = 0.0f;  // DCTL_TEMP: 0 = greedy (argmax)
  uint32_t seed_lo = 0, seed_hi = 0;   // DCTL_SEED_LO / _HI
  uint32_t pad[10] = {};
};
static_assert(sizeof(DFlashCtl) == 64, "the DFlash control block is one cache line");
static_assert(offsetof(DFlashCtl, k) == kernels::dflash::kDctlK * 4 &&
                  offsetof(DFlashCtl, proj_pos) == kernels::dflash::kDctlProjPos * 4 &&
                  offsetof(DFlashCtl, proj_n) == kernels::dflash::kDctlProjN * 4 &&
                  offsetof(DFlashCtl, temperature) == kernels::dflash::kDctlTemp * 4 &&
                  offsetof(DFlashCtl, seed_lo) == kernels::dflash::kDctlSeedLo * 4 &&
                  offsetof(DFlashCtl, seed_hi) == kernels::dflash::kDctlSeedHi * 4,
              "DFlashCtl and the CMake block's DCTL_DEFINES disagree");

// The readback at MAXR rows (df_select.cl's / mg_dflash.cl's OUT_* words at their MAXR): Qwen3.8's drafter at 8
// (DFlashDraftOut), spec 23f's Muse-Glimmer assistant at 16 (MuseDraftOut - mg_df_select writes it).
template <uint32_t MAXR>
struct DraftOutT {
  static constexpr uint32_t kRows = MAXR, kTopK = kernels::dflash::kTopK;
  uint32_t k;                       // OUT_K: the K this replay drafted
  uint32_t anchor;                  // OUT_ANCHOR: ctrl.cur_token[0] when it ran
  uint32_t pos;                     // OUT_POS: ctrl.pos (the anchor's position)
  uint32_t pad0;
  uint32_t ids[kRows];              // OUT_IDS: draft j's id (j < k)
  uint32_t chosen[kRows];           // OUT_CHOSEN: its index in cand[j]
  uint32_t cand[kRows][kTopK];      // OUT_CAND: row j's 16 candidates, best unary first (ties: lower id)
  float unary[kRows][kTopK];        // OUT_UNARY: their head logits (x out_mult, softcapped)
  float scores[kRows][kTopK];       // OUT_SCORES: the selector's realized scores (v1: = unary)
};
using DFlashDraftOut = DraftOutT<kernels::dflash::kM>;
using MuseDraftOut = DraftOutT<kernels::dflash::kMuseM>;
static_assert(offsetof(DFlashDraftOut, ids) == 4 * 4 && offsetof(DFlashDraftOut, chosen) == 12 * 4 &&
                  offsetof(DFlashDraftOut, cand) == 20 * 4 && offsetof(DFlashDraftOut, unary) == (20 + 128) * 4 &&
                  offsetof(DFlashDraftOut, scores) == (20 + 256) * 4 && sizeof(DFlashDraftOut) == 404 * 4,
              "DFlashDraftOut and df_select.cl's OUT_* words disagree");
static_assert(offsetof(MuseDraftOut, ids) == 4 * 4 && offsetof(MuseDraftOut, chosen) == 20 * 4 &&
                  offsetof(MuseDraftOut, cand) == 36 * 4 && offsetof(MuseDraftOut, unary) == (36 + 256) * 4 &&
                  offsetof(MuseDraftOut, scores) == (36 + 512) * 4 && sizeof(MuseDraftOut) == 804 * 4,
              "MuseDraftOut and mg_dflash.cl's OUT_* words (MAXR 16) disagree");

}  // namespace runtime::dflash
