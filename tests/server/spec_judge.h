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
//                 context a near-tie passes by the golden tie rule (greedy_tie.h), and anything beyond it AWAITS its
//                 reference (below).
//   N4 (rows)     a verify row against the plain row of the same context and the judge's row there: the verify row's
//                 cosine distance to f32 at most golden::kCosSlack x the plain row's (forced_judge.h's "tiny
//                 tolerance": two bf16 paths whose rounding points differ), or the old bar (cosine >= 0.99999
//                 against the plain row). Without a judge the old bar, and a row below it AWAITS its reference.
//
// THE bf16-UNSTABLE RULE (operator 2026-11-04): at a row where standard bf16 itself lands far from f32, a speculative
// miss is graded against how far plain decoding already is from f32 there, not against the fixed 8 ulp; everywhere
// else the strict rule stays. Per speculative miss at row r of its judge (the f32 top a, the speculative pick b):
//   g32 = f32[a] - f32[b], in bf16 ulps of the f32 top (golden::bf16_ulp_at) - the miss
//   g16 = the same pair's gap in a standard transformers bf16 reference of the same context
//         (<B70_FORCED_BF16>/<key>.bf16top.safetensors, tools/oracle/qwen_bf16_top.py: top-k ids and logits per row
//         plus the f32 top and the forced id), in the same unit
//   the row is bf16-UNSTABLE when |g16 - g32| > golden::kNearMissUlp; then
//   gp  = the pair's gap in the ENGINE's plain (M = 1) row on the same context (the test's PlainRow: a plain replay
//         of prompt + plain[0, d) + spec[d, d + r))
//   and the miss passes when g32 <= max(kNearMissUlp, |gp - g32|) - no further from f32 than plain decoding itself
//   is on that pair. A miss with no bf16 reference, a stable row, or no plain row stays on the strict bar. Every row
//   judged by this rule is printed (ids, the three gaps, the allowance, the verdict).
//
// A judge must be the one for the test's head form (the tag carries it: the int8 head's references are the
// --lm-head int8 runs, f32_refs.sh's -i8head suffix).
//
// AWAITING (operator 2026-11-04): a divergence the old rule fails (beyond the 1-ulp tie rule; a row below 0.99999)
// whose context has NO f32 reference yet - B70_FORCED_JUDGE unset, or its <key>.golden.safetensors not made yet - is
// not judged, so it is not failed either: it is reported as "awaiting f32 reference" and the test SKIPS (exit_code:
// 77, ctest's SKIP_RETURN_CODE) when nothing else failed. Every divergence a judge grades, and every structural /
// bitwise check of the tests, fails as before. The contexts are dumped as always (B70_DUMP_FORCED) - the dumps are
// what f32_refs.sh turns into the references.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../golden/forced_judge.h"
#include "greedy_tie.h"

namespace spec_judge {

// The process's divergences awaiting an f32 reference, and their contexts' keys.
struct Awaiting {
  uint64_t divergences = 0;
  std::set<std::string> contexts;
};
inline Awaiting& awaiting() {
  static Awaiting a;
  return a;
}
inline void await_reference(const std::string& key) {
  ++awaiting().divergences;
  awaiting().contexts.insert(key);
}

// The engine's plain (M = 1) logits row after the test's prompt + `generated` (the ids decoded after it), [V] f32 -
// the test replays its own plain path (tests/server/spec_plain_row.h). Empty function: no plain rows.
using PlainRow = std::function<std::vector<float>(const std::vector<uint32_t>& generated)>;

// One speculative miss beyond the near-miss bar, judged by the bf16-unstable rule.
struct UnstableRow {
  size_t row = 0;
  uint32_t a = 0, b = 0;            // the f32 top, the speculative pick
  double g32 = 0, g16 = NAN, gp = NAN, allow = golden::kNearMissUlp;
  bool have_bf16 = false, unstable = false, have_plain = false, pass = false;
  std::string line() const {
    char buf[320];
    if (!have_bf16)
      std::snprintf(buf, sizeof buf, "row %zu: ids %u (f32 top) / %u (speculative): f32 gap %.2f ulp; no bf16 reference - "
                    "the strict bar %.0f: FAIL", row, a, b, g32, golden::kNearMissUlp);
    else if (!unstable)
      std::snprintf(buf, sizeof buf, "row %zu: ids %u / %u: f32 gap %.2f, bf16 gap %.2f ulp - bf16 stable (|diff| %.2f <= "
                    "%.0f): the strict bar: FAIL", row, a, b, g32, g16, std::fabs(g16 - g32), golden::kNearMissUlp);
    else if (!have_plain)
      std::snprintf(buf, sizeof buf, "row %zu: ids %u / %u: f32 gap %.2f, bf16 gap %.2f ulp - bf16-UNSTABLE, but no plain "
                    "row: the strict bar: FAIL", row, a, b, g32, g16);
    else
      std::snprintf(buf, sizeof buf, "row %zu: ids %u / %u: f32 gap %.2f, bf16 gap %.2f, plain gap %.2f ulp - "
                    "bf16-UNSTABLE: allowance max(%.0f, |plain - f32| = %.2f) = %.2f: %s", row, a, b, g32, g16, gp,
                    golden::kNearMissUlp, std::fabs(gp - g32), allow, pass ? "PASS" : "FAIL");
    return buf;
  }
};

struct StreamVerdict {
  greedy_tie::Result tie;            // the tie-rule numbers (always computed, always printed)
  bool judged = false, pass = false;
  bool awaiting = false;             // not judged: the old rule fails it and its context has no f32 reference yet
  uint32_t miss_spec = 0, miss_plain = 0;
  double worst_spec_ulp = 0;         // the speculative stream's furthest miss below the f32 top
  std::vector<UnstableRow> beyond;   // the speculative misses beyond the near-miss bar, each by the bf16-unstable rule
  std::string line() const {
    char b[256];
    if (tie.v == greedy_tie::Verdict::Same) return "identical";
    const char* tv = tie.v == greedy_tie::Verdict::Tie ? "near-tie" : "beyond the 1-ulp tie rule";
    if (!judged) {
      std::snprintf(b, sizeof b, "first difference at %zu (%s, rank %u, %.3f ulp below the plain top); no f32 judge "
                    "yet - %s", tie.at, tv, tie.cand_rank, tie.cand_gap,
                    awaiting ? "AWAITING f32 reference (not judged: a SKIP, not a FAIL)"
                    : pass   ? "the tie rule passes it"
                             : "FAILS");
    } else {
      std::snprintf(b, sizeof b, "first difference at %zu (%s, rank %u, %.3f ulp); the f32 judge rejects %u speculative "
                    "/ %u plain ids from there (worst speculative miss %.2f ulp) - %s", tie.at, tv, tie.cand_rank,
                    tie.cand_gap, miss_spec, miss_plain, worst_spec_ulp, pass ? "PASS" : "FAIL");
    }
    std::string out = b;
    for (const UnstableRow& u : beyond) out += "\n      ** the bf16-unstable rule (operator 2026-11-04): " + u.line();
    return out;
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

// A standard-bf16 reference's rows, top-k only (tools/oracle/qwen_bf16_top.py): `ids` int32 [1 + gen][C], `logits`
// f32 [1 + gen][C] - the bf16 row's own top ids, the f32 top and the forced id, each with its bf16 logit.
struct Bf16Top {
  std::unique_ptr<golden::Golden> g;
  size_t rows = 0, cols = 0;
  explicit operator bool() const { return g != nullptr; }
  // The bf16 logit of `id` at `row`, or NaN when the reference did not keep it.
  double at(size_t row, uint32_t id) const {
    const int32_t* ids = static_cast<const int32_t*>(g->raw("ids", "I32", rows * cols));
    const float* lg = g->f32("logits", rows * cols);
    for (size_t c = 0; c < cols; ++c)
      if (uint32_t(ids[row * cols + c]) == id) return lg[row * cols + c];
    return NAN;
  }
};
inline Bf16Top open_bf16top(const golden::ForcedJudge& fj, const std::string& k, size_t gen) {
  Bf16Top t;
  if (fj.twin_dir.empty()) return t;
  const std::string p = fj.twin_dir + "/" + k + ".bf16top.safetensors";
  if (!golden::exists(p)) {
    std::printf("  no bf16 reference for %s (%s; tools/oracle/qwen_bf16_top.py) - the strict near-miss bar applies\n",
                k.c_str(), p.c_str());
    return t;
  }
  t.g = std::make_unique<golden::Golden>(p);
  t.rows = size_t(t.g->dim("ids", 2, 0));
  t.cols = size_t(t.g->dim("ids", 2, 1));
  CHECK_EQ(t.rows, gen + 1);
  CHECK_EQ(uint64_t(t.cols), t.g->dim("logits", 2, 1));
  return t;
}

// The bf16-unstable rule on one speculative miss (row `r` of the judge row `jrow`, the speculative pick `b`).
// `plain_row` yields the engine's plain row on the same context (empty: none).
inline UnstableRow unstable_rule(const float* jrow, uint32_t V, uint32_t VU, size_t r, uint32_t b, const Bf16Top& bt,
                                 const std::function<std::vector<float>()>& plain_row) {
  const golden::JudgedRow jr = golden::judge_row(jrow, V, VU);
  UnstableRow u;
  u.row = r;
  u.a = jr.top;
  u.b = b;
  u.g32 = (double(jrow[u.a]) - double(jrow[b])) / jr.ulp;
  if (bt) {
    const double fa = bt.at(r, u.a), fb = bt.at(r, b);
    u.have_bf16 = !std::isnan(fa) && !std::isnan(fb);
    if (u.have_bf16) u.g16 = (fa - fb) / jr.ulp;
  }
  u.unstable = u.have_bf16 && std::fabs(u.g16 - u.g32) > golden::kNearMissUlp;
  if (u.unstable && plain_row) {
    const std::vector<float> p = plain_row();
    u.have_plain = p.size() >= V;
    if (u.have_plain) {
      u.gp = (double(p[u.a]) - double(p[b])) / jr.ulp;
      u.allow = std::max(golden::kNearMissUlp, std::fabs(u.gp - u.g32));
    }
  }
  u.pass = u.unstable && u.have_plain && u.g32 <= u.allow;
  return u;
}

// N6. `steps[i]` is the plain row that decided plain[i] (greedy_tie.h); `tag` names the test, prompt and head form.
inline StreamVerdict streams(const golden::ForcedJudge& fj, const std::string& tag, const std::vector<uint32_t>& prompt,
                             const std::vector<uint32_t>& plain, const std::vector<greedy_tie::RefStep>& steps,
                             const std::vector<uint32_t>& spec, uint32_t V, uint32_t VU,
                             const PlainRow& plain_row = {}) {
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
    v.pass = true;   // a near-tie passes by the tie rule; beyond it the judge decides - and it is not there yet
    v.awaiting = v.tie.v != greedy_tie::Verdict::Tie;
    if (v.awaiting) await_reference(!jp ? kp : ks);
    return v;
  }
  v.judged = true;
  double unused = 0;
  v.miss_plain = misses(*jp, fp, V, VU, &unused);
  v.miss_spec = misses(*js, fs, V, VU, &v.worst_spec_ulp);
  v.pass = v.miss_spec <= v.miss_plain || v.worst_spec_ulp <= golden::kNearMissUlp;
  if (v.pass) return v;
  // More misses than plain, one beyond the near-miss bar: each such miss by the bf16-unstable rule.
  const Bf16Top bt = open_bf16top(fj, ks, fs.size());
  const float* rows = js->f32("logits", size_t(js->dim("logits", 2, 0)) * V);
  bool all = true;
  for (size_t r = 0; r < fs.size(); ++r) {
    const float* jrow = rows + r * V;
    if (golden::row_accepts(jrow, fs[r], V, VU)) continue;
    const golden::JudgedRow jr = golden::judge_row(jrow, V, VU);
    if ((double(jrow[jr.top]) - double(jrow[fs[r]])) / jr.ulp <= golden::kNearMissUlp) continue;   // a near-miss
    std::function<std::vector<float>()> pr;
    if (plain_row)
      pr = [&] {
        std::vector<uint32_t> gen(plain.begin(), plain.begin() + d);
        gen.insert(gen.end(), fs.begin(), fs.begin() + r);
        return plain_row(gen);
      };
    v.beyond.push_back(unstable_rule(jrow, V, VU, r, fs[r], bt, pr));
    all = all && v.beyond.back().pass;
  }
  v.pass = all;
  return v;
}

// N4's judge for one context: forced on `force` after `prompt` - row i decides force[i] (f32_refs.sh --lean).
struct RowJudge {
  std::unique_ptr<golden::Golden> g;
  uint32_t V = 0;
  std::string key;
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
  r.key = k;
  return r;
}
// N4 on one row: the verify row against the plain row of the same context (`cos_plain_vs_verify`, the old bar) and,
// with the judge `rj`'s row `judge_row`, each one's distance to f32. Below the old bar without a judge: awaiting.
struct RowVerdict {
  bool judged = false, pass = false, awaiting = false;
  double cos_verify_f32 = 1, cos_plain_f32 = 1, ratio = 1;
};
inline RowVerdict row(const float* verify, const float* plain, const RowJudge& rj, size_t judge_row,
                      double cos_plain_vs_verify, uint32_t VU) {
  RowVerdict v;
  v.pass = cos_plain_vs_verify >= 0.99999;
  if (!rj) {
    if (!v.pass) {
      v.pass = true;
      v.awaiting = true;
      await_reference(rj.key);
    }
    return v;
  }
  const float* jrow = rj.row(judge_row);
  v.judged = true;
  v.cos_verify_f32 = golden::row_cos(jrow, verify, VU);
  v.cos_plain_f32 = golden::row_cos(jrow, plain, VU);
  v.ratio = golden::cos_dist_ratio(v.cos_verify_f32, v.cos_plain_f32);
  v.pass = v.pass || v.ratio <= golden::kCosSlack;
  return v;
}
inline std::string row_line(const RowVerdict& v) {
  if (!v.judged)
    return v.awaiting ? " [no f32 judge: below the 0.99999 bar - AWAITING f32 reference (a SKIP, not a FAIL)]" : "";
  char b[160];
  std::snprintf(b, sizeof b, " [f32 judge: cos verify %.7f, plain %.7f, distance x%.2f (slack %.1f) %s]", v.cos_verify_f32,
                v.cos_plain_f32, v.ratio, golden::kCosSlack, v.pass ? "ok" : "FAIL");
  return b;
}

// The test's exit code: 1 when anything failed; else 77 (ctest SKIP: SKIP_RETURN_CODE 77) when divergences await
// their f32 reference; else 0. The awaiting count is printed loudly either way - it is what the next
// tools/box_validate/f32_refs.sh run has to make.
inline int exit_code(bool ok, const char* test) {
  const Awaiting& a = awaiting();
  if (a.divergences) {
    const char* jd = std::getenv("B70_FORCED_JUDGE");
    std::printf("\n*** %s: %llu divergence(s) in %zu context(s) AWAIT AN f32 REFERENCE - not judged, not failed "
                "(B70_FORCED_JUDGE=%s; the contexts are dumped with B70_DUMP_FORCED, tools/box_validate/f32_refs.sh "
                "forced / forced-qwen makes their references) ***\n",
                test, (unsigned long long)a.divergences, a.contexts.size(), jd && *jd ? jd : "(unset)");
    for (const std::string& k : a.contexts) std::printf("    awaiting f32 reference: %s\n", k.c_str());
    if (ok)
      std::printf("SKIP_REASON %s: %zu context(s) await their f32 reference; every other check passed\n", test,
                  a.contexts.size());
  }
  if (!ok) return 1;
  return a.divergences ? 77 : 0;
}

}  // namespace spec_judge
