// Spec 26c Task 6 (Review Focus 4 on the cards): Bolt + LiLiCorr snapshots carry the drafter's ring and anchor. Box only
// (labels checkpoint;bolt;lilicorr).
//   bolt_dflash_prefix_gpu_test <bolt checkpoint> <lilicorr snapshot> [--format exact|int8|int4]
//
// A prompt of 5000 ids prefilled with the block hook, each block end's session saved (state + the ring and the anchor after it,
// and the KV of [0, E)); the cold run's ring and 32 greedy speculative ids after the prompt are the reference. Then:
//   1. a restore at 2048 and at 4096 + the rest of the prompt prefilled: the ring and the anchor bitwise the cold run's, the
//      first block's drafts (the readback's ids, candidates and lattice) bitwise, and the 32 ids equal
//   2. vLLM a608b6e6e0: a sibling sharing the prompt to 4097 and then another id restores at the block end 4096 (the largest
//      <= the common prefix) - its ring after the rest equals its own cold prefill's
//   3. a snapshot whose ring part is zeros (what a drafter-less snapshot leaves the drafter): it restores, the drafts change
//      (counted), the output does not (greedy)
//   4. state_bytes = the target's + 5,259,264 B (the ring + the anchor); the store root differs with and without the drafter
// Exit 77 (SKIP) without two GPUs, the Bolt export or the drafter.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "bolt_rig.h"
#include "bolt_spec_rig.h"
#include "check.h"
#include "cli/bolt_serve.h"
#include "cli/bolt_spec.h"
#include "runtime/bolt/bolt_sizes.h"
#include "server/prefix_cache.h"

namespace {
namespace rb = runtime::bolt;
using Adapter = cli::bolt::BoltEngineAdapterT<rb::BoltEngine>;

std::vector<uint8_t> ring_of(rb::BoltEngine& e, runtime::dflash::LiliDrafter& d) {
  std::vector<uint8_t> h(d.ring_bytes());
  l0::CmdList imm = l0::CmdList::immediate(e.context(e.devices() - 1));
  d.save_ring(imm, h.data());
  return h;
}
struct Run {
  std::vector<uint8_t> ring;
  std::vector<uint32_t> ids, first_ids;
  std::vector<float> first_lattice;
};
// After a prefill (or a restore + the rest): the ring, the first block's drafts, 32 greedy speculative ids.
Run continue_run(Adapter& a, rb::BoltEngine& e, runtime::dflash::LiliDrafter& d) {
  Run r;
  r.ring = ring_of(e, d);
  server::Sampling greedy;
  while (r.ids.size() < 32) {
    const std::vector<uint32_t> run = a.step_many(greedy, 7);
    if (r.first_ids.empty()) {
      const runtime::dflash::LiliDraftOut& o = d.readback();
      r.first_ids.assign(o.ids, o.ids + 7);
      r.first_lattice.assign(&o.lattice[0][0][0], &o.lattice[0][0][0] + 7 * 8 * 8);
    }
    r.ids.insert(r.ids.end(), run.begin(), run.end());
  }
  a.flush_commit();
  r.ids.resize(32);
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: bolt_dflash_prefix_gpu_test <bolt checkpoint> <lilicorr snapshot> [--format F]\n");
    return 2;
  }
  const std::string snap = bolt_rig::bolt_snapshot(argv[1]);
  if (snap.empty() || l0::Context::gpu_count() < 2) {
    std::printf("SKIP: needs SergiioB's Bolt export (%s) and two GPUs\n", argv[1]);
    return 77;
  }
  cli::bolt::SpecArgs sa;
  sa.spec = "dflash";
  sa.draft_model = argv[2];
  sa.have_model = true;
  for (int i = 3; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--format") sa.format_arg = argv[++i];
  try {
    cli::bolt::check_spec(sa, 2, runtime::PpHandoff::Copy, true);
    (void)loader::LiliCheckpoint(sa.draft_model);
  } catch (const std::exception& e) {
    std::printf("SKIP: the drafter: %s\n", e.what());
    return 77;
  }
  bolt_rig::Options o;
  o.devices = 2;
  o.max_len = 16384;
  bolt_rig::Rig rig;
  bolt_rig::build(rig, snap, o);
  rb::BoltEngine& e = *rig.eng;
  const model::BoltDesc& d = e.model().desc;
  const size_t target_state = e.state_bytes();
  e.enable_spec(rb::VerifyMoe::Union);
  cli::bolt::BoltDrafter dr = cli::bolt::attach_drafter(e, sa, snap);
  CHECK_EQ(e.state_bytes(), target_state + size_t(5259264));   // the ring + the anchor (spec 26 §6)
  CHECK(server::store_root(cli::bolt::store_key(false), e.drafter_key()) != server::store_root(cli::bolt::store_key(false), 0));
  Adapter a(e, d.vocab, 7, false, cli::bolt::draft_hooks(e, *dr.drafter), sa.q);

  std::vector<uint32_t> P;
  for (uint32_t i = 0; i < 5000; ++i) P.push_back((i * 40503u + 101u) % d.vocab);
  std::map<uint32_t, bolt_spec_rig::Session> at;
  a.reset();
  e.set_block_hook([&](uint32_t end, bool block_end) {
    if (block_end) at[end] = bolt_spec_rig::save(e);
  });
  a.prefill(P);
  e.set_block_hook({});
  CHECK(at.count(2048) && at.count(4096));
  const Run cold = continue_run(a, e, *dr.drafter);
  int bad = 0;
  for (uint32_t E : {2048u, 4096u}) {   // 1. restores at the block ends
    a.reset();
    a.load_state(at[E].state.data(), E);
    e.load_kv(0, E, at[E].kv.data());
    a.prefill(std::vector<uint32_t>(P.begin() + E, P.end()));
    const Run w = continue_run(a, e, *dr.drafter);
    if (w.ring != cold.ring || w.first_ids != cold.first_ids || w.first_lattice != cold.first_lattice || w.ids != cold.ids) {
      std::fprintf(stderr, "restore at %u: the ring / the drafts / the ids differ from the cold run's\n", E);
      ++bad;
    }
    std::printf("restore at %u + the rest: ring, anchor, the first block's drafts and 32 ids %s the cold run's\n", E,
                bad ? "DIFFER from" : "=");
  }
  {   // 2. a608b6e6e0's sibling
    std::vector<uint32_t> sib(P.begin(), P.begin() + 4097);
    sib.push_back((P[4097] + 1) % d.vocab);
    for (uint32_t i = 0; i < 200; ++i) sib.push_back((i * 31u + 5u) % d.vocab);
    a.reset();
    a.prefill(sib);
    const std::vector<uint8_t> cold_sib = ring_of(e, *dr.drafter);
    a.reset();
    a.load_state(at[4096].state.data(), 4096);
    e.load_kv(0, 4096, at[4096].kv.data());
    a.prefill(std::vector<uint32_t>(sib.begin() + 4096, sib.end()));
    if (ring_of(e, *dr.drafter) != cold_sib) {
      std::fprintf(stderr, "the sibling restored at 4096: its ring differs from its own cold prefill's\n");
      ++bad;
    }
  }
  {   // 3. a ring of zeros
    std::vector<uint8_t> zeroed = at[4096].state;
    std::fill(zeroed.begin() + long(target_state), zeroed.end(), uint8_t(0));
    a.reset();
    a.load_state(zeroed.data(), 4096);
    e.load_kv(0, 4096, at[4096].kv.data());
    a.prefill(std::vector<uint32_t>(P.begin() + 4096, P.end()));
    const Run z = continue_run(a, e, *dr.drafter);
    size_t differ = 0;
    for (size_t i = 0; i < 7; ++i) differ += z.first_ids[i] != cold.first_ids[i];
    if (z.ids != cold.ids) {
      std::fprintf(stderr, "a ring of zeros changed the OUTPUT\n");
      ++bad;
    }
    std::printf("a ring of zeros: the output = the cold run's; the first block's lattice %s, %zu of 7 drafts differ\n",
                z.first_lattice != cold.first_lattice ? "moved" : "did not move", differ);
  }
  CHECK_EQ(bad, 0);
  std::printf("bolt_dflash_prefix_gpu_test OK (%s drafter)\n", loader::draft_format_name(sa.format));
  return 0;
}
