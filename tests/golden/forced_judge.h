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
//
// <key> = <name>_<FNV-1a 64 of the prompt and the forced ids, hex>: two configurations that reach the same tokens
// share one reference, and a judge can never be applied to another context than its own.
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
  std::string dump_dir, judge_dir;
  ForcedJudge() {
    if (const char* d = std::getenv("B70_DUMP_FORCED")) dump_dir = d;
    if (const char* j = std::getenv("B70_FORCED_JUDGE")) judge_dir = j;
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
};

// A forced-context row judged: does the f32 judge accept `id` there (judge_row's one-ulp rule)?
inline bool judge_accepts(const Golden& j, uint32_t row, uint32_t id, uint32_t vocab, uint32_t used) {
  const float* r = j.f32("logits", size_t(j.dim("logits", 2, 0)) * vocab) + size_t(row) * vocab;
  const JudgedRow jr = judge_row(r, vocab, used);
  return jr.d.determined() ? id == jr.top : jr.d.contains(id);
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
