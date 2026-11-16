// Kolibri DSpark on the cards (NOT YET RUN - waits for the first card run, after Flash-Next): the real Kolibri-1 on two
// cards (int8 head, as b70-serve runs it) with the DSpark drafter (intuit/Kolibri-1-dspark, bf16) on device 1, through
// b70-serve's own glue (cli/kolibri_spec.h attach_drafter, cli::kolibri::KolibriEngineAdapterT's step_many).
//
//   1. The drafter beside the engine: its bytes, its key, the snapshot's share (the target's state + the 20,971,520 B ring).
//   2. Greedy is lossless (D2): per prompt, 256 ids by step_many at --spec-max 7 against 256 plain ids of a plain engine -
//      token for token. A divergence prints its position and the plain row's top-2 margin (the golden tie rule's input,
//      docs/14-golden-gate.md: a near-tie is judged by hand) and FAILS.
//   3. The drafter works: the accepted drafts per verify, printed per prompt (the model card's MT-Bench-like mean is 2.98 tokens
//      a verify at K 7 on vLLM); a mean below 1.2 tokens a verify FAILS - the taps, the anchor-first rows or the Markov walk
//      are wrong then (TODO(card): set the real bar from this run).
//   4. A sampled request (T 0.7, seeded) runs and is reproducible; every id a defined token.
// Arguments: <Kolibri-1 snapshot> <DSpark snapshot> [prompt .ids files...]. SKIP 77 without either checkpoint or two cards.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "cli/kolibri_spec.h"
#include "kolibri_rig.h"
#include "loader/snapshot.h"

namespace {

namespace rk = runtime::kolibri;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 256, kUsed = 127998;

server::Sampling greedy() {
  server::Sampling s;
  s.greedy = true;
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snapdir = kolibri_rig::kolibri_snapshot(argc > 1 ? argv[1] : "");
  std::string draft;
  try {
    draft = loader::resolve_snapshot(argc > 2 ? argv[2] : "");
  } catch (const std::exception&) {
  }
  if (snapdir.empty() || draft.empty()) {
    std::printf("SKIP: Kolibri-1 '%s' or the DSpark drafter '%s' is not on this machine\n", argc > 1 ? argv[1] : "",
                argc > 2 ? argv[2] : "");
    return 77;
  }
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(snapdir);
  std::vector<Ids> prompts;
  for (int i = 3; i < argc; ++i) {
    Ids p = kolibri_rig::read_ids(argv[i], pre.vocab);
    if (!p.empty()) prompts.push_back(std::move(p));
  }
  if (prompts.empty()) prompts.push_back({27498, 2625, 19919, 32, 49, 56, 55, 49, 346, 7524, 744, 38897, 573, 122340});
  kolibri_rig::Options o;
  o.max_len = kMaxLen;
  o.devices = 2;
  o.int8_head = true;

  // The plain references first (one engine), then the speculative engine.
  std::vector<Ids> ref(prompts.size());
  std::vector<std::vector<float>> margins(prompts.size());
  kolibri_rig::Rig rig;
  {
    kolibri_rig::build(rig, snapdir, o);
    cli::kolibri::KolibriEngineAdapterT<rk::KolibriEngine> a(*rig.eng, kUsed);
    for (size_t p = 0; p < prompts.size(); ++p) {
      a.reset();
      a.prefill(prompts[p]);
      for (uint32_t i = 0; i < kN; ++i) {
        ref[p].push_back(a.step(greedy()));
        std::vector<float> row = rig.eng->read_logits();
        std::partial_sort(row.begin(), row.begin() + 2, row.end(), std::greater<float>());
        margins[p].push_back(row[0] - row[1]);
      }
    }
  }
  kolibri_rig::build(rig, snapdir, o);
  rk::KolibriEngine& e = *rig.eng;
  e.enable_spec();
  cli::kolibri::SpecArgs sa;
  sa.on = true;
  sa.draft_model = draft;
  sa.max_k = 7;
  cli::kolibri::KolibriDrafter dr = cli::kolibri::attach_drafter(e, sa);

  // --- 1. the drafter beside the engine --------------------------------------------------------------------------------
  CHECK(dr.drafter->desc().dspark() && dr.drafter->key() != 0 && e.drafter_key() == dr.drafter->key());
  CHECK_EQ(e.state_bytes(), rk::state_snapshot_bytes(e.model().desc) + size_t(20971520));
  std::printf("1. DSpark beside Kolibri-1: %zu B + the replica %zu B on device 1, key %016llx\n%s\n", dr.drafter->bytes(),
              dr.replica->size(), (unsigned long long)dr.drafter->key(), e.memory_line().c_str());

  // --- 2. / 3. greedy lossless, the acceptance ------------------------------------------------------------------------
  bool ok = true;
  cli::kolibri::KolibriEngineAdapterT<rk::KolibriEngine> a(e, kUsed, 7, cli::kolibri::draft_hooks(e, *dr.drafter));
  for (size_t p = 0; p < prompts.size(); ++p) {
    a.reset();
    a.prefill(prompts[p]);
    const uint64_t it0 = a.spec_iterations(), acc0 = a.spec_accepted();
    Ids out;
    while (out.size() < kN)
      for (uint32_t id : a.step_many(greedy(), 7)) out.push_back(id);
    out.resize(kN);
    const size_t first = std::mismatch(out.begin(), out.end(), ref[p].begin()).first - out.begin();
    const double per = 1.0 + double(a.spec_accepted() - acc0) / double(std::max<uint64_t>(1, a.spec_iterations() - it0));
    if (first < kN) {
      std::printf("  prompt %zu: DIVERGES at id %zu (%u vs plain %u; the plain row's top-2 margin %.6g)\n", p, first,
                  out[first], ref[p][first], first ? margins[p][first - 1] : -1.0);
      ok = false;
    }
    std::printf("2. prompt %zu: %u greedy ids %s; 3. %.3f tokens a verify\n", p, kN, first < kN ? "DIFFER" : "== plain", per);
    if (per < 1.2) ok = false;
  }

  // --- 4. sampled -----------------------------------------------------------------------------------------------------
  {
    auto run = [&](uint64_t seed) {
      server::Sampling s;
      s.greedy = false;
      s.temperature = 0.7f;
      s.top_p = 0.97f;
      s.top_k = 128;
      s.has_seed = true;
      s.seed = seed;
      a.reset();
      a.prefill(prompts[0]);
      Ids out;
      while (out.size() < 128)
        for (uint32_t id : a.step_many(s, 7)) out.push_back(id);
      for (uint32_t id : out) CHECK(id < kUsed);
      return out;
    };
    const Ids x = run(42), y = run(42);
    CHECK(x == y);
    std::printf("4. a sampled request (T 0.7, seed 42): reproducible, defined ids only\n");
  }
  std::printf("kolibri_dspark_test: %s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
