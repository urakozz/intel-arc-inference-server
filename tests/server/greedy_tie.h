#pragma once
// greedy_tie.h - spec 28 N6 for the greedy speculative gates (mtp_gpu_test M3, dflash_gpu_test D2, the lookup
// gate): a speculative run against the plain step() loop of the same engine is identical EXCEPT where the golden
// tie rule (docs/14-golden-gate.md; prefill_gate_test's argv[7] rule) classes the first divergence as a near-tie.
//
// Since spec 28b the verify rows run gemm_sm (DPAS) and are no longer bitwise the M = 1 rows (spec 28 §3), so the
// former IDENTITY bar becomes: at the first divergence d, the candidate's id is the reference row's runner-up and
// the reference row's top-2 margin is <= 1 bf16 ulp of its top logit - a near-tie (counted, recorded beside
// today's census); anything else is a non-tie divergence and fails. After a near-tie the two continuations
// differ legitimately, so the comparison of that run stops there.
#include <cmath>
#include <cstdint>
#include <vector>

namespace greedy_tie {

// The reference row's decision at one position: top-1, runner-up, and the margin in bf16 ulps of the top logit.
struct RefStep {
  uint32_t top1 = 0, second = 0;
  double margin_ulps = 0;
};
inline RefStep ref_step(const float* row, uint32_t used) {
  RefStep r;
  uint32_t t = 0;
  for (uint32_t i = 1; i < used; ++i)
    if (row[i] > row[t]) t = i;
  uint32_t s2 = t == 0 ? 1u : 0u;
  for (uint32_t i = 0; i < used; ++i)
    if (i != t && row[i] > row[s2]) s2 = i;
  int e = 0;
  std::frexp(std::fabs(row[t]), &e);
  r.top1 = t;
  r.second = s2;
  r.margin_ulps = (double(row[t]) - double(row[s2])) / std::ldexp(1.0, e - 8);
  return r;
}

// Verdict of a candidate id stream against the reference ids and the reference rows: `steps[i]` is the row that
// decided ref[i] (absent - empty RefStep vector entry count < i + 1 - for ids the rows were not recorded for, e.g.
// the prefill's first id: a divergence there is never a tie).
enum class Verdict { Same, Tie, Diverged };
struct Result {
  Verdict v = Verdict::Same;
  size_t at = SIZE_MAX;
  double margin_ulps = 0;
};
inline Result classify(const std::vector<uint32_t>& ref, const std::vector<RefStep>& steps,
                       const std::vector<uint32_t>& cand) {
  const size_t n = ref.size() < cand.size() ? ref.size() : cand.size();
  for (size_t i = 0; i < n; ++i) {
    if (ref[i] == cand[i]) continue;
    Result r;
    r.at = i;
    if (i < steps.size() && steps[i].top1 == ref[i] && steps[i].second == cand[i] && steps[i].margin_ulps <= 1.0) {
      r.v = Verdict::Tie;
      r.margin_ulps = steps[i].margin_ulps;
    } else {
      r.v = Verdict::Diverged;
      if (i < steps.size()) r.margin_ulps = steps[i].margin_ulps;
    }
    return r;
  }
  Result r;
  if (ref.size() != cand.size()) {
    r.v = Verdict::Diverged;
    r.at = n;
  }
  return r;
}

}  // namespace greedy_tie
