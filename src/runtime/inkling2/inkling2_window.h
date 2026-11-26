#pragma once
// Inkling's draft window: how many DSpark drafts a round verifies, by expected ids per plain step of cost (the form
// of basalt's draft policy). For K drafts the expected ids are E(K) = 1 + q1 + q1 q2 + ... + q1 ... qK, q_i the
// request's running acceptance at draft position i given the earlier drafts were accepted; the cost is the draft pass
// plus the verify of K + 1 rows, both in plain steps from a measured table - a table, not the request's own timings,
// so the same request makes the same rounds (verify rows are not the plain step's arithmetic). The K with the most ids
// per cost wins; 0 is a plain step, and while 0 wins a one-draft round now and then keeps the rates current.
// The costs move with the context (the drafter's full layer reads every position, the plain step's full visits too):
// a table by depth, interpolated linearly at the round's position, the ends held.
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace runtime::inkling2 {

struct WindowCost {
  std::array<double, 8> verify;   // verify[m - 1]: an m-row target step over the plain step (verify[0] = 1)
  double draft;                   // the draft pass (all seven drafts at once) over the plain step
};

// One card, depth 4096 (b70-decode --profile, card 1, 2026-11-26, the glue and router a row over several work-groups).
inline constexpr WindowCost kOneCardCost{{1.000, 1.182, 1.380, 1.534, 1.731, 1.870, 2.008, 2.218}, 0.329};
// `--parallel tp`, depth 4096 (b70-decode --profile --parallel tp, both cards, 2026-11-26, the drafter's layers split;
// with its head and fold split too the pass is 0.253 / 0.310 at 4k / 32k, not yet served as the table): a verify row
// costs less against tp's plain step (its crossings carry the rows), the drafter more.
inline constexpr WindowCost kTpCost{{1.000, 1.102, 1.226, 1.328, 1.454, 1.535, 1.640, 1.835}, 0.310};

struct DepthCost {
  uint32_t depth;
  WindowCost cost;
};
// One card from 32k on (bf16 cache, card 1) and pp (both cards, int8 cache): b70-decode --profile, 2026-11-26, the
// drafter's full layer on DPAS at 128 spans. The plain step: one card 16.9 / 22.2 ms at 32k / 128k; pp 19.3 / 21.3 /
// 39.2 / 61.2 ms at 32k / 128k / 512k / 1,000,000.
inline constexpr std::array<DepthCost, 3> kOneCardCosts{{
    {4096, kOneCardCost},
    {32768, {{1.000, 1.106, 1.337, 1.457, 1.598, 1.712, 1.815, 1.945}, 0.300}},
    {131072, {{1.000, 1.110, 1.407, 1.512, 1.617, 1.703, 1.781, 1.899}, 0.409}},
}};
// Plan 41k: one card with the DFlash 2 drafter (int8 linears; b70-decode --profile --spec dflash, card 0,
// 2026-11-26): the verify rows are the target's (as above, re-measured), the draft pass flat in depth - every drafter
// layer slides over its 2048-position ring - at 2.94 / 2.84 / 2.83 ms at 4k / 32k / 128k with the draft head on the
// 2-bit candidates and the exact rerank (card 1; the int8 head: 3.35-3.38 ms, bf16 linears: 3.82).
inline constexpr WindowCost kOneCardDflashCost{{1.000, 1.167, 1.355, 1.503, 1.679, 1.826, 1.963, 2.175}, 0.232};
inline constexpr std::array<DepthCost, 3> kOneCardDflashCosts{{
    {4096, kOneCardDflashCost},
    {32768, {{1.000, 1.154, 1.415, 1.547, 1.699, 1.832, 1.959, 2.139}, 0.197}},
    {131072, {{1.000, 1.121, 1.465, 1.554, 1.662, 1.760, 1.848, 1.968}, 0.143}},
}};
// The DFlash 2 drafter under pp (on card 1, plain step 12.7 / 14.6 / 19.8 / 36.8 ms at 4k / 32k / 128k / 512k, the last
// on the int8 cache) and tp (whole on both cards but its head's rows, plain 10.8 / 12.1 / 14.8 ms): b70-decode
// --profile --spec dflash, both cards, 2026-11-26; the draft pass flat. pp's drafter is the one-card list on card 1:
// its column is the one-card pass (2.94 / 2.84 / 2.83 ms, the 128k pass at 512k) over pp's plain steps.
inline constexpr std::array<DepthCost, 4> kPpDflashCosts{{
    {4096, {{1.000, 1.177, 1.368, 1.518, 1.702, 1.844, 1.987, 2.194}, 0.231}},
    {32768, {{1.000, 1.145, 1.403, 1.536, 1.694, 1.832, 1.949, 2.123}, 0.195}},
    {131072, {{1.000, 1.117, 1.482, 1.584, 1.705, 1.803, 1.890, 2.016}, 0.143}},
    {524288, {{1.000, 1.138, 1.548, 1.615, 1.679, 1.756, 1.810, 1.880}, 0.077}},
}};
// The drafter's head rows split over the cards: its pass 3.16 / 3.10 ms at 32k / 128k (flat in depth), 3.13 ms over
// 4k's 10.80 ms plain step.
inline constexpr WindowCost kTpDflashCost{{1.000, 1.134, 1.275, 1.405, 1.552, 1.663, 1.796, 2.027}, 0.290};
inline constexpr std::array<DepthCost, 3> kTpDflashCosts{{
    {4096, kTpDflashCost},
    {32768, {{1.000, 1.130, 1.297, 1.442, 1.641, 1.741, 1.891, 2.134}, 0.265}},
    {131072, {{1.000, 1.150, 1.389, 1.572, 1.884, 2.007, 2.209, 2.508}, 0.211}},
}};
inline constexpr std::array<DepthCost, 2> kTpCosts{{
    {4096, kTpCost},
    {32768, {{1.000, 1.095, 1.244, 1.368, 1.521, 1.608, 1.750, 1.953}, 0.357}},
}};
inline constexpr std::array<DepthCost, 4> kPpCosts{{
    {32768, {{1.000, 1.107, 1.277, 1.418, 1.492, 1.578, 1.766, 1.805}, 0.345}},
    {131072, {{1.000, 1.120, 1.410, 1.495, 1.599, 1.698, 1.774, 1.894}, 0.425}},
    {524288, {{1.000, 1.121, 1.509, 1.569, 1.629, 1.697, 1.750, 1.817}, 0.592}},
    {1000000, {{1.000, 1.118, 1.555, 1.596, 1.650, 1.680, 1.720, 1.759}, 0.658}},
}};

// The table's costs at `pos`: linear between the two depths around it, the nearest end's outside them.
inline WindowCost cost_at(std::span<const DepthCost> table, uint32_t pos) {
  if (pos <= table.front().depth) return table.front().cost;
  if (pos >= table.back().depth) return table.back().cost;
  size_t i = 1;
  while (table[i].depth < pos) ++i;
  const auto& a = table[i - 1];
  const auto& b = table[i];
  const double t = double(pos - a.depth) / double(b.depth - a.depth);
  WindowCost c{};
  for (size_t m = 0; m < c.verify.size(); ++m) c.verify[m] = a.cost.verify[m] + t * (b.cost.verify[m] - a.cost.verify[m]);
  c.draft = a.cost.draft + t * (b.cost.draft - a.cost.draft);
  return c;
}

class DraftWindow {
 public:
  static constexpr uint32_t kMaxDrafts = 7;
  static constexpr double kPrior = 0.5, kPriorWeight = 2, kDecay = 0.95;
  static constexpr uint32_t kProbeFirst = 4, kProbeMax = 32;

  explicit DraftWindow(std::span<const DepthCost> table) : table_(table), cost_(table.front().cost) { reset(); }

  void reset() {
    ok_.fill(kPrior * kPriorWeight);
    tried_.fill(kPriorWeight);
    plain_run_ = deeper_run_ = 0;
    probe_gap_ = kProbeFirst;
  }

  // Drafts for the coming round at position pos, at most cap.
  uint32_t next(uint32_t cap, uint32_t pos) {
    cost_ = cost_at(table_, pos);
    const uint32_t k = std::min(best(), cap);
    // The positions past K are only tested when a round drafts them: every kProbeMax rounds at K, one round at K + 1,
    // so a rate that two early misses pushed down can recover (the drafts past K cost only their verify rows).
    if (k > 0 && k < cap && ++deeper_run_ >= kProbeMax) {
      deeper_run_ = 0;
      return k + 1;
    }
    if (k > 0 || cap == 0) return k;
    if (++plain_run_ < probe_gap_) return 0;
    plain_run_ = 0;
    probe_gap_ = std::min(2 * probe_gap_, kProbeMax);
    return 1;
  }

  // A round of k drafts kept `accepted` of them: positions 1 .. accepted were accepted, accepted + 1 (if drafted)
  // was the first miss; the positions after it were not tested.
  void observe(uint32_t k, uint32_t accepted) {
    if (k == 0) return;
    for (uint32_t i = 0; i < std::min(accepted + 1, k); ++i) {
      ok_[i] = kDecay * ok_[i] + (i < accepted ? 1. : 0.);
      tried_[i] = kDecay * tried_[i] + 1.;
    }
    if (best() > 0) probe_gap_ = kProbeFirst;
  }

  double rate(uint32_t i) const { return ok_[i] / tried_[i]; }   // q_{i + 1}

  uint32_t best() const {
    uint32_t pick = 0;
    double best_rate = 1., ids = 1., run = 1.;
    for (uint32_t k = 1; k <= kMaxDrafts; ++k) {
      run *= rate(k - 1);
      ids += run;
      const double r = ids / (cost_.verify[k] + cost_.draft);
      if (r > best_rate) {
        best_rate = r;
        pick = k;
      }
    }
    return pick;
  }

 private:
  std::span<const DepthCost> table_;
  WindowCost cost_;   // at the last next()'s position
  std::array<double, kMaxDrafts> ok_{}, tried_{};
  uint32_t plain_run_ = 0, probe_gap_ = kProbeFirst, deeper_run_ = 0;
};

}  // namespace runtime::inkling2
