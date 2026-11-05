#pragma once
// Bolt's synthetic gates: the FIRST-FLIP CUT-OFF (operator ruling 2026-11-05; spec 25 §9.1).
//
// 25b's random-weight synthetics route in near-ties at every MoE block (bolt_short: 172 of 640 (row, block) 9th / 10th
// margins below 1e-3), and two correct bf16 implementations disagree on those margins by 1e-4 - 1e-3, so one of them flips
// a near-tie within the first rows; the flipped row's residual drops to ~0.97 cosine and the SSM states carry it into every
// later row and block (the bf16 reference against its own f32 run: 12 / 32 tokens, 273 / 640 expert sets). So on a
// synthetic checkpoint the strict bars bind only on what is computed BEFORE the first flip; after it every number is printed.
//
// Definitions, over (row, MoE block) pairs fed in CAUSAL order (row ascending, blocks ascending within a row):
//   compare a (row, block): the engine's 9 expert ids against the reference's as SETS, with the reference's 9th / 10th
//     selection margin `gap`:
//       Exact         the sets are equal;
//       Undetermined  they differ and gap == 0 (an exact tie in the reference: either id is right);
//       NearTie       they differ by exactly one expert and 0 < gap <= tol (tol: the existing near-tie tolerance, 1e-3);
//       Beyond        any other difference.
//   THE FIRST FLIP (r*, l*): the first pair classified NearTie or Undetermined.
//   UPSTREAM of the flip: (row, block) < (r*, l*) lexicographically - the pairs computed without it (block l at row r
//     depends on blocks < l at row r and on every block at rows < r). With no flip everything is upstream.
//   A Beyond difference upstream of the flip FAILS (a differing set at a margin above the tolerance is a defect, as before);
//   one after the flip is printed (bad_after). Quantities gate when every pair they depend on is upstream:
//     h tap (row r, block l)                 upstream(r, l)
//     a logits row r / a decision row r      row_clean(r): no flip at or before row r
//     SSM state / conv columns of block l    upstream(T - 1, l) (after the prompt's T rows)
//     K / V row of position p, block l       upstream(p, l)
// `enabled` false (a real checkpoint) keeps the old rule exactly: no cut-off, every Beyond counted in bad_before.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace bolt_cutoff {

enum class Diff { Exact, Undetermined, NearTie, Beyond };

inline const char* diff_name(Diff d) {
  switch (d) {
    case Diff::Exact: return "exact";
    case Diff::Undetermined: return "undetermined";
    case Diff::NearTie: return "near-tie";
    default: return "beyond";
  }
}

// The engine's ids `eng` against the reference's `ref` (k each, any order), the reference's margin `gap`.
template <typename A, typename B>
Diff classify(const A* eng, const B* ref, uint32_t k, float gap, float tol) {
  std::vector<int64_t> e(eng, eng + k), r(ref, ref + k);
  std::sort(e.begin(), e.end());
  std::sort(r.begin(), r.end());
  if (e == r) return Diff::Exact;
  if (gap == 0.0f) return Diff::Undetermined;
  uint32_t common = 0;
  for (int64_t x : e) common += std::binary_search(r.begin(), r.end(), x) ? 1 : 0;
  return gap <= tol && common + 1 == k ? Diff::NearTie : Diff::Beyond;
}

struct Cut {
  static constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();
  bool enabled = true;
  bool found = false;
  uint32_t row = kNone, block = 0;   // the first flip
  uint64_t pairs = 0, exact = 0, ties = 0, undetermined = 0, bad_before = 0, bad_after = 0;
  // Feed one (row, block) in causal order; returns its class.
  Diff add(uint32_t r, uint32_t l, Diff d) {
    ++pairs;
    switch (d) {
      case Diff::Exact: ++exact; break;
      case Diff::NearTie: ++ties; break;
      case Diff::Undetermined: ++undetermined; break;
      case Diff::Beyond: (found ? bad_after : bad_before) += 1; break;
    }
    if (enabled && !found && (d == Diff::NearTie || d == Diff::Undetermined)) {
      found = true;
      row = r;
      block = l;
    }
    return d;
  }
  bool upstream(uint32_t r, uint32_t l) const { return !found || r < row || (r == row && l < block); }
  bool row_clean(uint32_t r) const { return !found || r < row; }
};

}  // namespace bolt_cutoff
