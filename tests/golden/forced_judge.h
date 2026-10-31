#pragma once
// The f32 judges of the gates f32_oracle.sh's three golden prompts do not cover (operator ruling 2026-10-29): a
// test's engine run is the reference context - a prompt and the ids every compared row is teacher-forced on - and
// the f32 reference is run on exactly that context afterwards, on the box CPU (tools/box_validate/f32_refs.sh
// forced <model> <dir>: ornith_ref.py / k2_ref.py run --lean --force).
//
//   B70_DUMP_FORCED=<dir>    the test writes <dir>/<key>.prompt.ids and <key>.force.ids for every context it
//                            would judge (the engine is deterministic: a second run is the same context)
//   B70_FORCED_JUDGE=<dir>   the test reads <dir>/<key>.golden.safetensors (--lean: `logits` [1 + gen][V], row j
//                            the decision of forced id j) and judges with golden_common.h's judge_row
//   B70_FORCED_BF16=<dir>    the bf16 twin of each judge (f32_refs.sh forced ... with DTYPE=bf16 --gemm f32): the
//                            plain bf16 reference on the same context. Operator ruling 2026-10-31: with it, an engine
//                            path passes when the judge rejects no more of its ids than of the twin's argmax (the
//                            Ornith R2 principle, 2026-10-28) - rank-1/2 misses a few ulp below the f32 top are what
//                            bf16 arithmetic does, and the twin measures how many of them this context has.
//                            Round 2 of the same day: more misses than the twin still pass by the near-miss
//                            rule (NearMiss, below) - each extra one a near-tie, the path's logits as close.
//
// <key> = <name>_<FNV-1a 64 of the prompt and the forced ids, hex>: two configurations that reach the same tokens
// share one reference, and a judge can never be applied to another context than its own.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "golden_common.h"

namespace golden {

struct ForcedJudge {
  std::string dump_dir, judge_dir, twin_dir;
  ForcedJudge() {
    if (const char* d = std::getenv("B70_DUMP_FORCED")) dump_dir = d;
    if (const char* j = std::getenv("B70_FORCED_JUDGE")) judge_dir = j;
    if (const char* t = std::getenv("B70_FORCED_BF16")) twin_dir = t;
  }
  static std::string key(const std::string& name, const std::vector<uint32_t>& prompt,
                         const std::vector<uint32_t>& force) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint32_t v) {
      for (int b = 0; b < 4; ++b) h = (h ^ ((v >> (8 * b)) & 0xFFu)) * 1099511628211ull;
    };
    for (uint32_t v : prompt) mix(v);
    mix(0xFFFFFFFFu);
    for (uint32_t v : force) mix(v);
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long)h);
    std::string n;
    for (char c : name) n += (c == ' ' || c == ':' || c == '/' || c == ',') ? '_' : c;
    return n + "_" + hex;
  }
  void dump(const std::string& k, const std::vector<uint32_t>& prompt, const std::vector<uint32_t>& force) const {
    if (dump_dir.empty()) return;
    auto write = [&](const char* ext, const std::vector<uint32_t>& v) {
      std::ofstream f(dump_dir + "/" + k + ext);
      for (size_t i = 0; i < v.size(); ++i) f << (i ? " " : "") << v[i];
      f << "\n";
    };
    write(".prompt.ids", prompt);
    write(".force.ids", force);
    std::printf("  dumped the forced context %s (%zu prompt ids, %zu forced) to %s\n", k.c_str(), prompt.size(),
                force.size(), dump_dir.c_str());
  }
  // The judge of context `k`, or none (no judge dir, or its file is not there yet - said so).
  std::unique_ptr<Golden> open(const std::string& k, size_t gen, size_t vocab) const {
    if (judge_dir.empty()) return nullptr;
    const std::string p = judge_dir + "/" + k + ".golden.safetensors";
    if (!exists(p)) {
      std::printf("  no f32 judge for %s (%s; tools/box_validate/f32_refs.sh forced) - the old rule applies\n",
                  k.c_str(), p.c_str());
      return nullptr;
    }
    auto g = std::make_unique<Golden>(p);
    CHECK_EQ(g->dim("logits", 2, 0), uint64_t(gen + 1));
    CHECK_EQ(g->dim("logits", 2, 1), uint64_t(vocab));
    return g;
  }
  // The judge's bf16 twin, or none (said so: the gate then compares against the other engine run, as before).
  std::unique_ptr<Golden> open_twin(const std::string& k, size_t gen, size_t vocab) const {
    if (twin_dir.empty()) return nullptr;
    const std::string p = twin_dir + "/" + k + ".golden.safetensors";
    if (!exists(p)) {
      std::printf("  no bf16 twin for %s (%s; f32_refs.sh forced with DTYPE=bf16 --gemm f32) - the engine runs are"
                  " compared with each other\n", k.c_str(), p.c_str());
      return nullptr;
    }
    auto g = std::make_unique<Golden>(p);
    CHECK_EQ(g->dim("logits", 2, 0), uint64_t(gen + 1));
    CHECK_EQ(g->dim("logits", 2, 1), uint64_t(vocab));
    return g;
  }
};

// A judge's decision row against an id: does it accept it (judge_row's one-ulp rule)?
inline bool row_accepts(const float* r, uint32_t id, uint32_t vocab, uint32_t used) {
  const JudgedRow jr = judge_row(r, vocab, used);
  return jr.d.determined() ? id == jr.top : jr.d.contains(id);
}
// A forced-context row judged: does the f32 judge accept `id` there (judge_row's one-ulp rule)?
inline bool judge_accepts(const Golden& j, uint32_t row, uint32_t id, uint32_t vocab, uint32_t used) {
  return row_accepts(j.f32("logits", size_t(j.dim("logits", 2, 0)) * vocab) + size_t(row) * vocab, id, vocab, used);
}

// The judge's margin at a row where it rejects one run's id (`who`): how far below the f32 top that id sits, in
// the judge's own units - the top-2 gap and one bf16 ulp of the top logit (judge_row's near-tie width). Printed
// beside every rejection, so a reader can tell a near-miss (a gap of a few ulp) from a divergence.
inline void judge_explain(const Golden& j, uint32_t row, uint32_t id, uint32_t vocab, uint32_t used,
                          const char* who) {
  const float* r = j.f32("logits", size_t(j.dim("logits", 2, 0)) * vocab) + size_t(row) * vocab;
  const JudgedRow jr = judge_row(r, vocab, used);
  const double below = double(r[jr.top]) - double(r[id]);
  uint32_t rank = 0;   // ids strictly above `id` in the judge's row
  for (uint32_t i = 0; i < (vocab < used ? vocab : used); ++i) rank += r[i] > r[id];
  std::printf("    row %u: the judge rejects the %s id %u - %.4g below its top %u (%.2f bf16 ulp of the top %.4f; "
              "rank %u; the judge's own top-2 gap %.4g = %.2f ulp)\n",
              row, who, id, below, jr.top, below / jr.ulp, double(r[jr.top]), rank, jr.gap, jr.gap / jr.ulp);
}

}  // namespace golden

namespace golden {

// The bf16 twin's own misses over rows [0, rows): rows where the judge rejects every id of the twin's argmax set
// (golden_decision: a bf16 tie is any of its ids), each printed with its margin.
inline uint32_t twin_misses(const Golden& judge, const Golden& twin, uint32_t rows, uint32_t vocab, uint32_t used) {
  const float* t = twin.f32("logits", size_t(twin.dim("logits", 2, 0)) * vocab);
  uint32_t n = 0;
  for (uint32_t r = 0; r < rows; ++r) {
    const GoldenDecision gd = golden_decision(t + size_t(r) * vocab, vocab, used);
    bool any = false;
    for (uint32_t id : gd.set) any = any || judge_accepts(judge, r, id, vocab, used);
    if (!any) {
      ++n;
      judge_explain(judge, r, gd.set[0], vocab, used, "bf16 reference's");
    }
  }
  return n;
}

// A run's logits row against the judge's row `row`, cosine over the first `used` columns; and the min / median of
// a run's rows - how far a whole path sits from f32, beside the per-row decisions.
inline double row_cos(const float* r, const float* v, uint32_t used) {
  double ab = 0, aa = 0, bb = 0;
  for (uint32_t i = 0; i < used; ++i) {
    ab += double(r[i]) * v[i];
    aa += double(r[i]) * r[i];
    bb += double(v[i]) * v[i];
  }
  return ab / std::sqrt(aa * bb);
}
inline double judge_cos(const Golden& j, uint32_t row, const float* v, uint32_t vocab, uint32_t used) {
  return row_cos(j.f32("logits", size_t(j.dim("logits", 2, 0)) * vocab) + size_t(row) * vocab, v, used);
}
struct CosRows {
  std::vector<double> c;
  double min() const { return c.empty() ? 1.0 : *std::min_element(c.begin(), c.end()); }
  double median() const {
    if (c.empty()) return 1.0;
    std::vector<double> s = c;
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
  }
  std::string str() const {
    if (c.empty()) return "-";
    char b[64];
    std::snprintf(b, sizeof b, "%.6f / %.6f", min(), median());
    return b;
  }
};

// ---- the near-miss rule (operator ruling 2026-10-31, round 2) --------------------------------------------------
// An engine path the f32 judge rejects at MORE rows than its bf16 twin (the plain bf16 reference on the same
// context: B70_FORCED_BF16, a gate's bf16 golden, k2_ref's bf16+kv8 twin) still passes when both hold:
//   1. every EXTRA miss - a row where the judge rejects the path's id but accepts the twin's - is a near-tie: the
//      id at most kNearMissUlp bf16 ulp (golden_common.h bf16_ulp_at, of the f32 top) below the judge's top. The
//      judge's own rule calls one ulp a tie; the extra misses measured on 2026-10-30/31 sat 1.8 - 6.2 ulp below it
//      (K2 prefix split c row 8: 4.72, a 3-way f32 near-tie; Ornith prefix l0 a / e: 6.24, 5.36, 3.05), and the
//      bf16 twins' OWN misses on the same contexts 3.1 - 8.6 ulp - what bf16 arithmetic alone does there.
//   2. the path's logits rows are no further from the judge's than the twin's are: the cosine distance (1 - cos)
//      of its min row and of its median row each at most kCosSlack times the twin's. kCosSlack is the "tiny
//      tolerance" (operator): the engine's UNCACHED paths measured up to 1.53x the twin's distance on the same
//      contexts (Ornith prefix l0 e: cold median 0.999249 vs the twin's 0.999545) - two bf16 implementations of
//      one model whose rounding points differ - so 2 is that spread, not room for a defect (a wrong row or a lost
//      state moves the distance by orders of magnitude).
// Every number is printed: each miss with its margin, both counts, both cosines and their distance ratios.
constexpr double kNearMissUlp = 8.0;
constexpr double kCosSlack = 2.0;

// The ratio of two cosine distances, (1 - a) / (1 - b) (1 when both are exact).
inline double cos_dist_ratio(double a, double b) {
  const double da = std::max(0.0, 1.0 - a), db = std::max(0.0, 1.0 - b);
  return db > 0 ? da / db : (da > 0 ? INFINITY : 1.0);
}

struct NearMiss {
  uint32_t engine = 0, twin = 0;                    // rows the judge rejects of the path's id / the twin's argmax
  std::vector<std::pair<uint32_t, double>> extra;   // the path's extra misses: (row, ulp below the f32 top)
  CosRows cos_engine, cos_twin;                     // each one's rows against the judge's
  // One row: the judge's row `j`, the path's id and logits, the twin's logits (its argmax set is its decision).
  // `quiet`: count without printing the misses (a caller that prints its own rows).
  void add(uint32_t row, const float* j, uint32_t id, const float* engine_logits, const float* twin_logits,
           uint32_t vocab, uint32_t used, const char* who, bool quiet = false) {
    const bool e_ok = row_accepts(j, id, vocab, used);
    const GoldenDecision td = golden_decision(twin_logits, vocab, used);
    bool t_ok = false;
    for (uint32_t t : td.set) t_ok = t_ok || row_accepts(j, t, vocab, used);
    const JudgedRow jr = judge_row(j, vocab, used);
    auto explain = [&](uint32_t x, const char* w) {
      if (quiet) return;
      const double below = double(j[jr.top]) - double(j[x]);
      uint32_t rank = 0;
      for (uint32_t i = 0; i < (vocab < used ? vocab : used); ++i) rank += j[i] > j[x];
      std::printf("    row %u: the judge rejects the %s id %u - %.4g below its top %u (%.2f bf16 ulp of the top %.4f; "
                  "rank %u; the judge's own top-2 gap %.4g = %.2f ulp)\n",
                  row, w, x, below, jr.top, below / jr.ulp, double(j[jr.top]), rank, jr.gap, jr.gap / jr.ulp);
    };
    if (!e_ok) {
      ++engine;
      explain(id, who);
      if (t_ok) extra.emplace_back(row, (double(j[jr.top]) - double(j[id])) / jr.ulp);
    }
    if (!t_ok) {
      ++twin;
      explain(td.set[0], "bf16 twin's");
    }
    cos_engine.c.push_back(row_cos(j, engine_logits, used));
    cos_twin.c.push_back(row_cos(j, twin_logits, used));
  }
  void merge(const NearMiss& o) {
    engine += o.engine;
    twin += o.twin;
    extra.insert(extra.end(), o.extra.begin(), o.extra.end());
    cos_engine.c.insert(cos_engine.c.end(), o.cos_engine.c.begin(), o.cos_engine.c.end());
    cos_twin.c.insert(cos_twin.c.end(), o.cos_twin.c.begin(), o.cos_twin.c.end());
  }
  // The verdict, printed on one line (and the extra rows on the next).
  bool verdict(const std::string& name, const char* who) const {
    bool near = true;
    std::string rows;
    for (const auto& [r, u] : extra) {
      near = near && u <= kNearMissUlp;
      char b[64];
      std::snprintf(b, sizeof b, "%s%u (%.2f ulp)", rows.empty() ? "" : ", ", r, u);
      rows += b;
    }
    const double rmin = cos_dist_ratio(cos_engine.min(), cos_twin.min());
    const double rmed = cos_dist_ratio(cos_engine.median(), cos_twin.median());
    const bool cos_ok = rmin <= kCosSlack && rmed <= kCosSlack;
    const bool by_count = engine <= twin;
    const bool pass = by_count || (near && cos_ok);
    std::printf("  %s near-miss rule (operator 2026-10-31, round 2): the judge rejects %u of the %s ids, %u of the "
                "bf16 twin's; cosine min / median %s vs the twin's %s (distance x%.2f / x%.2f, slack %.1f)%s%s\n",
                name.c_str(), engine, who, twin, cos_engine.str().c_str(), cos_twin.str().c_str(), rmin, rmed,
                kCosSlack, by_count ? " - no more misses than the twin" : "",
                pass ? (by_count ? "" : " - PASS by the near-miss rule") : "  ** FAIL **");
    if (!extra.empty())
      std::printf("    extra misses (the judge accepts the twin's id there): rows %s - %s (bar %.0f ulp)%s\n",
                  rows.c_str(), near ? "near-ties" : "NOT all near-ties", kNearMissUlp,
                  cos_ok ? "" : "; the logits further from f32 than the slack allows");
    return pass;
  }
};

}  // namespace golden
