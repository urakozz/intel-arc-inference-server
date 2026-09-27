// Spec 7 C3 (plan 7c Task 4): the tie-aware rule at a replay divergence. For each case,
// reproduce the COLD run (the --prefix-cache-gb 0 server's path: reset, prefill the prompt,
// then one decode replay per generated id) up to the first differing generated token, and
// judge the cached run's id against that row: determined and equal, a bf16 near-tie (the
// cold run's runner-up, top-2 margin <= 1 bf16 ulp of the top logit, prefill_gate_test's
// argv[7] rule), inside an exact-tie set, or a real mismatch.
//   probe_tie_judge <snapshot> <cases file> [--max-len 131072]
// Cases file, three lines per case: "P <prompt ids>", "G <common generated ids>",
// "C <cached id> <cold id> <label>". Exit 0 when every case is accepted.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/engine.h"

namespace {
using Ids = std::vector<uint32_t>;
Ids parse(const std::string& rest) {
  std::istringstream in(rest);
  Ids v;
  uint32_t x = 0;
  while (in >> x) v.push_back(x);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: probe_tie_judge <snapshot> <cases> [--max-len N]\n");
    return 2;
  }
  uint32_t max_len = 131072;
  if (argc > 4 && std::string(argv[3]) == "--max-len") max_len = uint32_t(std::stoul(argv[4]));
  struct Case { Ids p, g; uint32_t a = 0, b = 0; std::string label; };
  std::vector<Case> cases;
  std::ifstream in(argv[2]);
  for (std::string line; std::getline(in, line);) {
    if (line.size() < 2) continue;
    const std::string rest = line.substr(2);
    if (line[0] == 'P') cases.push_back({parse(rest), {}, 0, 0, ""});
    else if (line[0] == 'G') cases.back().g = parse(rest);
    else if (line[0] == 'C') {
      std::istringstream s(rest);
      s >> cases.back().a >> cases.back().b >> cases.back().label;
    }
  }
  if (cases.empty()) {
    std::printf("probe_tie_judge: no cases\n");
    return 0;
  }
  using model::Qwen35;
  l0::Context ctx(0);
  runtime::Engine eng(ctx, loader::load(ctx, loader::resolve_snapshot(argv[1]), max_len), max_len);
  eng.prepare_prefill();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::vector<float> row(Qwen35::kVocab);
  bool all = true;
  for (const Case& c : cases) {
    eng.reset();
    eng.prefill(c.p);
    const l0::Mem* src = &eng.prefill_scratch()->logits;
    for (uint32_t id : c.g) {
      eng.ingest({id});
      src = &eng.buffers().logits;
    }
    imm.copy(row.data(), src->ptr(), size_t(Qwen35::kVocab) * 4);
    const uint32_t used = Qwen35::kVocabUsed;
    uint32_t t1 = 0;
    for (uint32_t i = 1; i < used; ++i)
      if (row[i] > row[t1]) t1 = i;
    uint32_t t2 = t1 == 0 ? 1 : 0;
    for (uint32_t i = 0; i < used; ++i)
      if (i != t1 && row[i] > row[t2]) t2 = i;
    int e = 0;
    std::frexp(std::fabs(row[t1]), &e);
    const double ulp = std::ldexp(1.0, e - 8);
    const double margin = double(row[t1]) - double(row[t2]);
    const char* verdict;
    bool ok = true;
    if (c.a == t1) verdict = "cached id is the cold argmax (the cold run's own id differs: ?)";
    else if (margin == 0 && c.a == t2) verdict = "UNDETERMINED (exact tie), accepted";
    else if (c.a == t2 && margin <= ulp) verdict = "bf16 NEAR-TIE (runner-up, margin <= 1 ulp), accepted";
    else { verdict = "MISMATCH (determined row)"; ok = false; }
    all = all && ok;
    std::printf("%s: prompt %zu + %zu generated | cold top1 %u (%.5f) top2 %u (%.5f), margin %.5f"
                " = %.3f bf16 ulp | cached %u cold %u | %s\n", c.label.c_str(), c.p.size(),
                c.g.size(), t1, row[t1], t2, row[t2], margin, margin / ulp, c.a, c.b, verdict);
    std::fflush(stdout);
  }
  std::printf("probe_tie_judge: %s\n", all ? "every divergence accepted by the tie rule"
                                          : "at least one determined mismatch");
  return all ? 0 : 1;
}
