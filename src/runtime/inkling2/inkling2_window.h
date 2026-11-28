#pragma once
// Inkling's draft window: how many DSpark drafts a round verifies, by expected ids per plain step of cost (the form
// of basalt's draft policy). For K drafts the expected ids are E(K) = 1 + q1 + q1 q2 + ... + q1 ... qK, q_i the
// request's running acceptance at draft position i given the earlier drafts were accepted; the cost is the draft pass
// plus the verify of K + 1 rows, both in plain steps from a measured table - a table, not the request's own timings,
// so the same request makes the same rounds (verify rows are not the plain step's arithmetic). The K with the most ids
// per cost wins; 0 is a plain step, and while 0 wins a one-draft round now and then keeps the rates current.
// The costs move with the context (the drafter's full layer reads every position, the plain step's full visits too):
// a table by depth, interpolated linearly at the round's position, the ends held. b70-serve does not serve on these
// tables: it measures its own round costs at load and keeps them current from every round (server::LiveCost); a
// table here is b70-decode's window and the server's prior for a width or depth no round has timed yet.
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
// Plan 41k: the DFlash 2 drafter (the packed int4 drafter, the 2-bit candidate head; b70-decode --profile --spec
// dflash, main 49f34642, 2026-11-27): one card (card 0) plain 11.71 / 13.42 / 18.75 ms at 4k / 32k / 128k, the draft
// pass flat at 1.70-1.72 ms - every drafter layer slides over its 2048-position ring.
inline constexpr WindowCost kOneCardDflashCost{{1.000, 1.170, 1.360, 1.476, 1.614, 1.761, 1.872, 1.960}, 0.145};
inline constexpr std::array<DepthCost, 3> kOneCardDflashCosts{{
    {4096, kOneCardDflashCost},
    {32768, {{1.000, 1.151, 1.429, 1.524, 1.639, 1.776, 1.851, 1.954}, 0.127}},
    {131072, {{1.000, 1.138, 1.488, 1.553, 1.645, 1.745, 1.821, 1.881}, 0.092}},
}};
// ... under pp (the drafter on card 1; plain 11.68 / 13.54 / 18.77 / 43.46 ms at 4k / 32k / 128k / 512k, the
// pass 1.72-1.82 ms) and tp (whole on both cards but its head's rows; plain 9.99 / 10.82 / 13.90 ms, the pass 2.10-2.13 ms),
// both cards, the same run.
inline constexpr std::array<DepthCost, 4> kPpDflashCosts{{
    {4096, {{1.000, 1.176, 1.370, 1.485, 1.628, 1.770, 1.874, 1.959}, 0.147}},
    {32768, {{1.000, 1.149, 1.426, 1.518, 1.640, 1.772, 1.840, 1.939}, 0.129}},
    {131072, {{1.000, 1.119, 1.489, 1.570, 1.647, 1.734, 1.800, 1.859}, 0.093}},
    {524288, {{1.000, 1.043, 1.507, 1.527, 1.544, 1.589, 1.595, 1.629}, 0.042}},
}};
inline constexpr WindowCost kTpDflashCost{{1.000, 1.097, 1.246, 1.363, 1.477, 1.586, 1.663, 1.778}, 0.213};
inline constexpr std::array<DepthCost, 3> kTpDflashCosts{{
    {4096, kTpDflashCost},
    {32768, {{1.000, 1.105, 1.282, 1.428, 1.584, 1.717, 1.839, 1.985}, 0.194}},
    {131072, {{1.000, 1.135, 1.379, 1.552, 1.845, 1.975, 2.140, 2.368}, 0.152}},
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
