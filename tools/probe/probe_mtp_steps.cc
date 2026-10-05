// probe_mtp_steps - spec 8 plan 8b: the production MTP lists' wall time per call.
//
//   probe_mtp_steps <snapshot> [depth = 4096] [calls = 32] [rounds = 3] [lm_head = bf16]
//                   [draft_vocab = off]
//
// `lm_head` (spec 9 H2): bf16 (the checkpoint's) or int8; the draft list reads it too.
// `draft_vocab` (spec 8 §11): off, 32k, 64k or 128k (needs int8) - the draft list reads
// the compact head instead; V' = tokenizer.json's added tokens (the EOS ids among them on
// Qwen3.8 and Agnes) and the lowest ids, no ranked list (the step time depends on |V'|
// only). The draft arms then price `--mtp-cost`'s draft row for that size (spec 8 §10).
//
// One engine with the head, `depth` ids of tests/golden/prompts/long32k.ids prefilled
// (cwd = the source tree). Arms, timed as the engine API runs them (host writes, submit,
// fence wait - what 8c's loop pays): verify(k) + commit(0) for k = 0..3, i.e. the verify
// list at M = k + 1, and draft(k) for k = 1..3. Each round runs every arm `calls` times,
// arms in a rotated order, after a warm-up pass; the median over rounds of each arm's
// mean ms/call is printed. commit(0) advances pos by one per call, so a round moves the
// depth by 4 x calls positions (0.8% of 4096 at the defaults) - the same drift for every
// arm, as the order rotates.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "l0/context.h"
#include "loader/draft_vocab.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/engine.h"

namespace {
std::vector<uint32_t> read_ids(const std::string& p) {
  std::vector<uint32_t> v;
  FILE* f = std::fopen(p.c_str(), "r");
  if (!f) return v;
  unsigned x;
  while (std::fscanf(f, "%u", &x) == 1) v.push_back(x);
  std::fclose(f);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <snapshot> [depth] [calls] [rounds] [bf16|int8] [off|32k|64k|128k]\n",
                 argv[0]);
    return 2;
  }
  const uint32_t depth = argc > 2 ? std::stoul(argv[2]) : 4096;
  const uint32_t calls = argc > 3 ? std::stoul(argv[3]) : 32;
  const uint32_t rounds = argc > 4 ? std::stoul(argv[4]) : 3;
  std::vector<uint32_t> ids = read_ids("tests/golden/prompts/long32k.ids");
  if (ids.size() < depth) return 2;
  ids.resize(depth);
  l0::Context ctx(0);
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  if (argc > 5 && !loader::parse_lm_head_form(argv[5], lm_head)) return 2;
  loader::DraftVocabSpec dv;
  if (argc > 6) {
    if (!loader::parse_draft_vocab(argv[6], dv.size)) return 2;
    if (dv.size != 0)
      dv.added = loader::added_token_ids_file(loader::resolve_snapshot(argv[1]) + "tokenizer.json");
  }
  std::printf("lm_head: %s, draft vocab %s\n", loader::lm_head_form_name(lm_head),
              loader::draft_vocab_name(dv.size).c_str());
  runtime::Engine e(ctx, loader::load(ctx, argv[1], 16384, /*mtp=*/true, lm_head, dv), 16384);
  e.prefill(ids);
  using Clock = std::chrono::steady_clock;
  struct Arm {
    std::string name;
    uint32_t k;
    bool draft;
    std::vector<double> ms;
  };
  std::vector<Arm> arms;
  for (uint32_t k = 0; k <= 3; ++k) arms.push_back({"verify M=" + std::to_string(k + 1), k, false, {}});
  for (uint32_t k = 1; k <= 3; ++k) arms.push_back({"draft k=" + std::to_string(k), k, true, {}});
  auto run = [&](Arm& a, uint32_t n) {
    const auto t0 = Clock::now();
    for (uint32_t i = 0; i < n; ++i) {
      if (a.draft) {
        e.draft(a.k);
      } else {
        e.verify(a.k);
        e.commit(0, e.verify_ids()[0]);
      }
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / n;
  };
  for (Arm& a : arms) run(a, 8);   // warm-up
  for (uint32_t r = 0; r < rounds; ++r)
    for (size_t i = 0; i < arms.size(); ++i) {
      Arm& a = arms[(i + r) % arms.size()];
      a.ms.push_back(run(a, calls));
    }
  std::printf("probe_mtp_steps: depth %u..%u, %u calls x %u rounds\n", depth, e.pos(), calls,
              rounds);
  const double base = [&] {
    std::vector<double> v = arms[0].ms;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }();
  for (Arm& a : arms) {
    std::vector<double> v = a.ms;
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    std::printf("| %-12s | %8.3f ms | x M=1 verify %.3f | range %.3f-%.3f |\n", a.name.c_str(), med,
                med / base, v.front(), v.back());
  }
  return 0;
}
