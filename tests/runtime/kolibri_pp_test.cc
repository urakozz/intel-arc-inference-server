// Spec 20c Task 6: Kolibri-1 across two cards on spec 16b's pieces (box only; label checkpoint;kolibri;pp).
//   kolibri_pp_test <synthetic checkpoint> <real checkpoint> <ids>
//
//   (a) the synthetic checkpoint: --pp 1 against --pp 2 --pipeline-split 3, under each hand-off (copy,
//       peer): the prompt + 32 greedy tokens - the tokens, logits, route rows, every full layer's KV and
//       every ring BITWISE equal (plan 20c Review Focus 5: the cut is invisible; a stale hand-off buffer -
//       spec 16 §2's silent receive - shows here); both Control blocks equal after every phase
//   (b) the real checkpoint (spec 20b), --pipeline-split 25 against 20, each hand-off: the same, bitwise
//   (c) P4: drop_next_handoff() - the step throws within the bounds (never hangs), every later step
//       throws until reset(), and after reset() the run is again bitwise (a)'s
//   (d) replay determinism on two cards: two runs bitwise
// The peer-access refusal (zeDeviceCanAccessPeer false -> refused naming it, KolibriEngine's
// constructor) cannot be provoked on a P2P-capable pair; it is read, not run.
// Exit 77 (SKIP) with one GPU, or without the synthetic checkpoint ((b) alone SKIPs without the real one).
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "kolibri_rig.h"
#include "runtime/control.h"

namespace {
constexpr uint32_t kMaxLen = 4096, kGen = 32;

struct Snapshot {
  std::vector<uint32_t> tokens, routes;
  std::vector<float> logits;
  std::vector<std::vector<uint16_t>> kv;
  bool operator==(const Snapshot& o) const {
    return tokens == o.tokens && routes == o.routes && logits == o.logits && kv == o.kv;
  }
};

void controls_equal(runtime::kolibri::KolibriEngine& e, const char* phase) {
  if (e.devices() < 2) return;
  const runtime::Control a = e.control(0), b = e.control(1);
  if (std::memcmp(&a, &b, sizeof a) != 0) {
    std::fprintf(stderr, "the two Control blocks differ after %s (pos %u / %u)\n", phase, a.pos, b.pos);
    std::exit(1);
  }
}

Snapshot run(runtime::kolibri::KolibriEngine& e, const std::vector<uint32_t>& ids) {
  const model::Kolibri1Desc& d = e.model().desc;
  e.reset();
  controls_equal(e, "reset");
  e.ingest(ids);
  controls_equal(e, "ingest");
  Snapshot s;
  s.tokens = e.generate(kGen);
  controls_equal(e, "generate");
  s.logits = e.read_logits();
  s.routes = e.read_routes();
  const uint32_t end = e.pos();
  for (uint32_t l = 0; l < d.layers; ++l) {
    const uint32_t first = d.is_sliding(l) && end > d.window ? end - d.window : 0;
    std::vector<uint16_t> k = e.read_kv(l, first, end - first, false), v = e.read_kv(l, first, end - first, true);
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(std::move(k));
  }
  return s;
}

Snapshot with(kolibri_rig::Rig& rig, const std::string& snap, const std::vector<uint32_t>& ids, uint32_t devices,
              uint32_t split, runtime::PpHandoff h) {
  kolibri_rig::Options o;
  o.max_len = kMaxLen;
  o.devices = devices;
  o.split = split;
  o.handoff = h;
  o.timeout_ms = 5000;   // P4's lost hand-off fails within 5 s, not the default 30
  kolibri_rig::build(rig, snap, o);
  const Snapshot s = run(*rig.eng, ids);
  std::printf("  %s: %u device(s)%s, %zu launches, tokens", snap.c_str(), devices,
              devices == 2 ? (std::string(", split ") + std::to_string(rig.eng->split()) + " " +
                              runtime::pp_handoff_name(rig.eng->handoff())).c_str() : "",
              rig.eng->launches());
  for (uint32_t i = 0; i < 6; ++i) std::printf(" %u", s.tokens[i]);
  std::printf(" ...\n");
  return s;
}
}  // namespace

int main(int argc, char** argv) {
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two GPUs needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const std::string synth = kolibri_rig::kolibri_snapshot(argc > 1 ? argv[1] : "");
  const std::string real = kolibri_rig::kolibri_snapshot(argc > 2 ? argv[2] : "");
  if (synth.empty()) {
    std::printf("SKIP: no synthetic Kolibri-1 checkpoint at '%s' (kolibri_oracle.sh synth)\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  const std::vector<uint32_t> ids = kolibri_rig::read_ids(argc > 3 ? argv[3] : "", 128000);
  CHECK(!ids.empty());
  kolibri_rig::Rig rig;
  const runtime::PpHandoff hs[] = {runtime::PpHandoff::Copy, runtime::PpHandoff::Peer};

  // (a) one card against two, each hand-off
  std::printf("(a) synthetic: --pp 1 against --pp 2 --pipeline-split 3\n");
  const Snapshot one = with(rig, synth, ids, 1, 0, runtime::kDefaultPpHandoff);
  for (runtime::PpHandoff h : hs) {
    const Snapshot two = with(rig, synth, ids, 2, 3, h);
    CHECK(two == one);
    // (d) replay determinism on two cards
    CHECK(run(*rig.eng, ids) == two);
    // (c) P4: a hand-off that never arrives
    rig.eng->ingest({ids[0]});
    rig.eng->drop_next_handoff();
    bool threw = false;
    try {
      rig.eng->ingest({ids[1]});
    } catch (const std::runtime_error& e) {
      threw = true;
      std::printf("  P4 (%s): %s\n", runtime::pp_handoff_name(h), e.what());
    }
    CHECK(threw);
    threw = false;
    try {
      rig.eng->ingest({ids[2]});
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);   // until reset()
    CHECK(run(*rig.eng, ids) == one);   // reset() recovers, bitwise
  }
  // (b) the real checkpoint: split 25 against 20
  if (real.empty()) {
    std::printf("(b) SKIP: no real Kolibri-1 checkpoint at '%s' (spec 20b)\n", argc > 2 ? argv[2] : "");
  } else {
    std::printf("(b) real: --pipeline-split 25 against 20\n");
    for (runtime::PpHandoff h : hs) {
      const Snapshot a = with(rig, real, ids, 2, 25, h);
      const Snapshot b = with(rig, real, ids, 2, 20, h);
      CHECK(a == b);
    }
  }
  std::puts("kolibri_pp_test OK");
  return 0;
}
