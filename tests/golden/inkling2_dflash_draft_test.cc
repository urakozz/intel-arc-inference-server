// Plan 41k Task 3 (card): the DFlash 2 drafter's draft list on one card against the CPU reference on the same taps.
//   inkling2_dflash_draft_test <Inkling snapshot> <DFlash 2 snapshot> <drafts.txt>
// drafts.txt (tools/oracle/dflash_accept.py drafts, from a b70-decode --dump-taps run): the run's ids, its prompt
// length, then per anchor its position and the reference's 7 drafts (fp32 drafter over the dump's tap rows, the int8
// head). The engine replays the same run - the prompt pass, then one-row verify steps of the run's own ids (the dump's
// path: its taps are the engine's) - and drafts at each anchor (propose, 7 rows). Printed: drafts equal by position
// and the first difference. The reference is fp32 and the engine's linears bf16 rows, so a near-tie may differ: the
// bar is the first draft equal at >= 90 % of anchors and every draft at >= 75 %. SKIP 77 without the files.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "l0/context.h"
#include "loader/inkling2_loader.h"
#include "runtime/inkling2/inkling2_engine.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: inkling2_dflash_draft_test <Inkling snapshot> <DFlash 2 snapshot> <drafts.txt>\n");
    return 2;
  }
  std::ifstream in(argv[3]);
  if (!in) {
    std::printf("SKIP: no %s\n", argv[3]);
    return 77;
  }
  std::string line;
  std::vector<uint32_t> ids;
  uint32_t n_prompt = 0;
  std::getline(in, line);
  {
    std::istringstream s(line);
    for (uint32_t v; s >> v;) ids.push_back(v);
  }
  in >> n_prompt;
  std::vector<std::pair<uint32_t, std::vector<uint32_t>>> anchors;
  for (uint32_t p; in >> p;) {
    std::vector<uint32_t> d(7);
    for (auto& v : d) in >> v;
    anchors.push_back({p, d});
  }
  CHECK(!anchors.empty() && n_prompt > 0 && anchors.back().first < ids.size());

  const loader::Inkling2Checkpoint checkpoint(argv[1], argv[2]);
  l0::Context ctx(0);
  runtime::inkling2::Engine engine(ctx, checkpoint, uint32_t(ids.size() + 16), true, runtime::inkling2::kPrefillRows);
  CHECK(engine.dflash());
  engine.prefill({ids.data(), n_prompt}, runtime::inkling2::kPrefillRows);
  std::vector<uint32_t> equal(7, 0);
  uint32_t shown = 0;
  size_t a = 0;
  for (uint32_t p = n_prompt; a < anchors.size(); ++p) {
    if (anchors[a].first == p) {
      const auto got = engine.propose(ids[p], 7);
      const auto& want = anchors[a].second;
      for (uint32_t j = 0; j < 7; ++j) equal[j] += got[j] == want[j];
      if (shown < 4 && got[0] != want[0]) {
        std::printf("  anchor %u: engine %u %u %u, reference %u %u %u\n", p, got[0], got[1], got[2], want[0], want[1],
                    want[2]);
        ++shown;
      }
      ++a;
    }
    engine.verify({&ids[p], 1});
    engine.commit(1);
  }
  const double n = double(anchors.size());
  std::printf("%zu anchors from %u: drafts equal the reference's by position:", anchors.size(), anchors[0].first);
  uint32_t all = 0;
  for (const auto e : equal) {
    std::printf(" %.3f", e / n);
    all += e;
  }
  std::printf("\n");
  CHECK(equal[0] >= 0.9 * n);
  CHECK(all >= 0.75 * 7 * n);
  std::printf("inkling2_dflash_draft_test OK\n");
  return 0;
}
