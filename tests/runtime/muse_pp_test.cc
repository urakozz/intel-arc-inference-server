// Spec 23c Task 5: Muse-Glimmer across two cards on spec 16b's pieces (box only; labels checkpoint;muse;pp).
//   muse_pp_test <synthetic checkpoint> <ids> [<real checkpoint>]
//
//   (a) the 4-layer synthetic checkpoint: --pp 1 against --pp 2 --pipeline-split 2, under each hand-off (copy, peer):
//       the prompt + 32 greedy tokens - the tokens, logits, every full layer's KV and every sliding layer's whole ring
//       BITWISE equal (the cut is invisible: device 0 ends with layer 1's mg_post_add, which wrote resid and the 26
//       sums layer 2's mg_norm reads); both Control blocks equal after every phase
//   (b) P4: drop_next_handoff() - the step throws within the 5 s bound naming device 1 (never hangs), every later
//       step throws until reset(), and after reset() the run is again bitwise (a)'s one-card run
//   (c) replay determinism on two cards: two runs bitwise
//   (d) the real checkpoint (when given and present): split 26 against 20 under each hand-off, bitwise
// The peer-access refusal (zeDeviceCanAccessPeer false -> MuseEngine's constructor refuses naming it) cannot be
// provoked on a P2P-capable pair; it is read, not run.
// Exit 77 (SKIP) with one GPU or without the synthetic checkpoint ((d) alone SKIPs without the real one).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "runtime/control.h"
#include "runtime/muse_rig.h"

namespace {
constexpr uint32_t kGen = 32;
namespace rm = runtime::muse;

struct Snapshot {
  std::vector<uint32_t> tokens;
  std::vector<float> logits;
  std::vector<std::vector<uint16_t>> kv;
  bool operator==(const Snapshot& o) const { return tokens == o.tokens && logits == o.logits && kv == o.kv; }
};

void controls_equal(rm::MuseEngine& e, const char* phase) {
  if (e.devices() < 2) return;
  const runtime::Control a = e.control(0), b = e.control(1);
  if (std::memcmp(&a, &b, sizeof a) != 0) {
    std::fprintf(stderr, "the two Control blocks differ after %s (pos %u / %u)\n", phase, a.pos, b.pos);
    std::exit(1);
  }
}

Snapshot run(rm::MuseEngine& e, const std::vector<uint32_t>& ids) {
  const model::MuseGlimmerDesc& d = e.weights().desc;
  e.reset();
  controls_equal(e, "reset");
  e.ingest(ids);
  controls_equal(e, "ingest");
  Snapshot s;
  s.tokens = e.generate(kGen);
  controls_equal(e, "generate");
  s.logits = e.read_logits();
  const uint32_t end = e.pos();
  for (uint32_t l = 0; l < d.layers; ++l) {
    std::vector<uint16_t> k, v;
    if (d.is_full(l)) {
      k = e.read_kv(l, 0, end, false);
      v = e.read_kv(l, 0, end, true);
    } else {
      k = e.read_ring(l, false);
      v = e.read_ring(l, true);
    }
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(std::move(k));
  }
  return s;
}

Snapshot with(muse_rig::Rig& rig, const std::string& snap, const std::vector<uint32_t>& ids, uint32_t devices,
              uint32_t split, runtime::PpHandoff h) {
  rm::MuseOptions o;
  o.max_len = 4096;
  o.split = split;
  o.handoff = h;
  o.timeout_ms = 5000;   // P4's lost hand-off fails within 5 s, not the default 30
  muse_rig::build(rig, snap, devices, o);
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
  const std::string synth = muse_rig::muse_snapshot(argc > 1 ? argv[1] : "");
  if (synth.empty()) {
    std::printf("SKIP: no synthetic Muse-Glimmer checkpoint at '%s' (muse_oracle.sh synth-ckpt)\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  const std::vector<uint32_t> ids = muse_rig::read_ids(argc > 2 ? argv[2] : "", 202048);
  CHECK(ids.size() >= 3);
  const std::string real = muse_rig::muse_snapshot(argc > 3 ? argv[3] : "");
  muse_rig::Rig rig;
  const runtime::PpHandoff hs[] = {runtime::PpHandoff::Copy, runtime::PpHandoff::Peer};

  std::printf("(a) synthetic: --pp 1 against --pp 2 --pipeline-split 2\n");
  const Snapshot one = with(rig, synth, ids, 1, 0, runtime::kDefaultPpHandoff);
  for (runtime::PpHandoff h : hs) {
    const Snapshot two = with(rig, synth, ids, 2, 2, h);
    CHECK(two == one);
    CHECK_EQ(rig.eng->launches(), rm::decode_launches(rig.eng->weights().desc, rig.eng->weights().placement,
                                                      rig.eng->attention(), h));
    // (c) replay determinism on two cards
    CHECK(run(*rig.eng, ids) == two);
    // (b) P4: a hand-off that never arrives
    rig.eng->ingest({ids[0]});
    rig.eng->drop_next_handoff();
    bool threw = false;
    const auto t0 = std::chrono::steady_clock::now();
    try {
      rig.eng->ingest({ids[1]});
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()).find("device 1") != std::string::npos;
      std::printf("  P4 (%s): %s\n", runtime::pp_handoff_name(h), e.what());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(threw);
    CHECK(s < 5.0 * 3);   // the 5 s bound, and the release's second bounded wait
    threw = false;
    try {
      rig.eng->ingest({ids[2]});
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);   // until reset()
    CHECK(run(*rig.eng, ids) == one);   // reset() recovers, bitwise
  }
  if (real.empty()) {
    std::printf("(d) SKIP: no real Muse-Glimmer checkpoint at '%s'\n", argc > 3 ? argv[3] : "");
  } else {
    std::printf("(d) real: --pipeline-split 26 against 20\n");
    for (runtime::PpHandoff h : hs) {
      const Snapshot a = with(rig, real, ids, 2, 26, h);
      const Snapshot b = with(rig, real, ids, 2, 20, h);
      CHECK(a == b);
    }
  }
  std::puts("muse_pp_test OK");
  return 0;
}
