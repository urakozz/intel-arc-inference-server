#pragma once
// spec_judge.h - spec 28 N4 / N6 under the operator's ruling of 2026-11-03 (option B): since spec 28b the verify rows
// run gemm_sm and are not bitwise the M = 1 rows, so speculative output is graded by the f32 JUDGE - it passes when it
// is no further from f32 than plain decode on the same context. The judges are tests/golden/forced_judge.h's: the
// test writes each context it would judge (B70_DUMP_FORCED=<dir>), the f32 reference is run on exactly that context
// on the box CPU, and the test reads it back (B70_FORCED_JUDGE=<dir>). The bitwise and tie numbers stay printed.
//
//   N6 (streams)  a speculative id stream against the plain one: identical, or - from their first divergence d, where
//                 the two share the context prompt + plain[0, d) - the judge (one reference per stream, forced on its
//                 own suffix) rejects no more of the speculative ids than of the plain ids; or every speculative miss
//                 is a near-miss (<= golden::kNearMissUlp bf16 ulp below the f32 top: forced_judge.h's rule 1 - the
//                 streams' logits rows are not read back, so its cosine half does not apply). Without a judge for a
//                 context the golden tie rule (greedy_tie.h) decides, as before the ruling - and says so.
//   N4 (rows)     a verify row against the plain row of the same context and the judge's row there: the verify row's
//                 cosine distance to f32 at most golden::kCosSlack x the plain row's (forced_judge.h's "tiny
//                 tolerance": two bf16 paths whose rounding points differ), or the old bar (cosine >= 0.99999
//                 against the plain row). Without a judge only the old bar.
//
// A judge must be the one for the test's head form (the tag carries it: the int8 head's references are the
// --lm-head int8 runs, f32_refs.sh's -i8head suffix).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../golden/forced_judge.h"
#include "greedy_tie.h"

namespace spec_judge {

struct StreamVerdict {
  greedy_tie::Result tie;            // the tie-rule numbers (always computed, always printed)
  bool judged = false, pass = false;
  uint32_t miss_spec = 0, miss_plain = 0;
  double worst_spec_ulp = 0;         // the speculative stream's furthest miss below the f32 top
  std::string line() const {
    char b[256];
    if (tie.v == greedy_tie::Verdict::Same) return "identical";
    const char* tv = tie.v == greedy_tie::Verdict::Tie ? "near-tie" : "beyond the 1-ulp tie rule";
    if (!judged) {
      std::snprintf(b, sizeof b, "first difference at %zu (%s, rank %u, %.3f ulp below the plain top); no f32 judge "
                    "yet - %s", tie.at, tv, tie.cand_rank, tie.cand_gap, pass ? "the tie rule passes it" : "FAILS");
    } else {
      std::snprintf(b, sizeof b, "first difference at %zu (%s, rank %u, %.3f ulp); the f32 judge rejects %u speculative "
                    "/ %u plain ids from there (worst speculative miss %.2f ulp) - %s", tie.at, tv, tie.cand_rank,
                    tie.cand_gap, miss_spec, miss_plain, worst_spec_ulp, pass ? "PASS" : "FAIL");
    }
    return b;
  }
};

// The misses of one forced stream against its judge: rows r in [0, gen) where the judge rejects force[r]; the worst
// one's distance below the f32 top in bf16 ulps.
inline uint32_t misses(const golden::Golden& j, const std::vector<uint32_t>& force, uint32_t V, uint32_t VU,
                       double* worst) {
  const float* rows = j.f32("logits", size_t(j.dim("logits", 2, 0)) * V);
  uint32_t n = 0;
  for (size_t r = 0; r < force.size(); ++r) {
    const float* row = rows + r * V;
    if (golden::row_accepts(row, force[r], V, VU)) continue;
    ++n;
    const golden::JudgedRow jr = golden::judge_row(row, V, VU);
    if (worst) *worst = std::max(*worst, (double(row[jr.top]) - double(row[force[r]])) / jr.ulp);
  }
  return n;
}

// N6. `steps[i]` is the plain row that decided plain[i] (greedy_tie.h); `tag` names the test, prompt and head form.
inline StreamVerdict streams(const golden::ForcedJudge& fj, const std::string& tag, const std::vector<uint32_t>& prompt,
                             const std::vector<uint32_t>& plain, const std::vector<greedy_tie::RefStep>& steps,
                             const std::vector<uint32_t>& spec, uint32_t V, uint32_t VU) {
  StreamVerdict v;
  v.tie = greedy_tie::classify(plain, steps, spec);
  if (v.tie.v == greedy_tie::Verdict::Same) {
    v.pass = true;
    return v;
  }
  const size_t d = v.tie.at, n = std::min(plain.size(), spec.size());
  std::vector<uint32_t> ctx = prompt;
  ctx.insert(ctx.end(), plain.begin(), plain.begin() + d);
  const std::vector<uint32_t> fp(plain.begin() + d, plain.begin() + n), fs(spec.begin() + d, spec.begin() + n);
  const std::string kp = golden::ForcedJudge::key(tag, ctx, fp), ks = golden::ForcedJudge::key(tag, ctx, fs);
  fj.dump(kp, ctx, fp);
  fj.dump(ks, ctx, fs);
  const std::unique_ptr<golden::Golden> jp = fj.open(kp, fp.size(), V), js = fj.open(ks, fs.size(), V);
  if (!jp || !js) {
    v.pass = v.tie.v == greedy_tie::Verdict::Tie;
    return v;
  }
  v.judged = true;
  double unused = 0;
  v.miss_plain = misses(*jp, fp, V, VU, &unused);
  v.miss_spec = misses(*js, fs, V, VU, &v.worst_spec_ulp);
  v.pass = v.miss_spec <= v.miss_plain || v.worst_spec_ulp <= golden::kNearMissUlp;
  return v;
}

// N4's judge for one context: forced on `force` after `prompt` - row i decides force[i] (f32_refs.sh --lean).
struct RowJudge {
  std::unique_ptr<golden::Golden> g;
  uint32_t V = 0;
  explicit operator bool() const { return g != nullptr; }
  const float* row(size_t i) const { return g->f32("logits", size_t(g->dim("logits", 2, 0)) * V) + i * V; }
};
inline RowJudge open_rows(const golden::ForcedJudge& fj, const std::string& tag, const std::vector<uint32_t>& prompt,
                          const std::vector<uint32_t>& force, uint32_t V) {
  const std::string k = golden::ForcedJudge::key(tag, prompt, force);
  fj.dump(k, prompt, force);
  RowJudge r;
  r.g = fj.open(k, force.size(), V);
  r.V = V;
  return r;
}
// N4 on one row: the verify row against the plain row of the same context (`cos_plain_vs_verify`, the old bar) and,
// with a judge row, each one's distance to f32.
struct RowVerdict {
  bool judged = false, pass = false;
  double cos_verify_f32 = 1, cos_plain_f32 = 1, ratio = 1;
};
inline RowVerdict row(const float* verify, const float* plain, const float* judge_row, double cos_plain_vs_verify,
                      uint32_t VU) {
  RowVerdict v;
  v.pass = cos_plain_vs_verify >= 0.99999;
  if (!judge_row) return v;
  v.judged = true;
  v.cos_verify_f32 = golden::row_cos(judge_row, verify, VU);
  v.cos_plain_f32 = golden::row_cos(judge_row, plain, VU);
  v.ratio = golden::cos_dist_ratio(v.cos_verify_f32, v.cos_plain_f32);
  v.pass = v.pass || v.ratio <= golden::kCosSlack;
  return v;
}
inline std::string row_line(const RowVerdict& v) {
  if (!v.judged) return v.pass ? "" : " [no f32 judge: the 0.99999 bar FAILS]";
  char b[160];
  std::snprintf(b, sizeof b, " [f32 judge: cos verify %.7f, plain %.7f, distance x%.2f (slack %.1f) %s]", v.cos_verify_f32,
                v.cos_plain_f32, v.ratio, golden::kCosSlack, v.pass ? "ok" : "FAIL");
  return b;
}

}  // namespace spec_judge
