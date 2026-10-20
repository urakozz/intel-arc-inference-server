// Spec 24c Task 6: LFM2.5-24B-A2B across two cards on spec 16b's pieces (box only; label checkpoint;lfm2moe;pp).
//   lfm2moe_pp_test <synthetic checkpoint> <rtn checkpoint> <ids>
//
//   (a) the synthetic N 4: --pp 1 against --pp 2 --pipeline-split 2 (a DENSE cut: layer 1's down GEMV slices cross
//       with h and device 1's first norm folds them) and --pipeline-split 3 (a MoE cut: h alone), under each hand-off
//       (copy, peer): the prompt + 32 greedy tokens - the tokens, logits, route rows, every attention layer's KV and
//       every conv ring BITWISE equal (plan 24c Review Focus 6: the cut is invisible); both Control blocks equal
//       after every phase; replay determinism on two cards
//   (b) the RTN export (24q), --pipeline-split 20 against 19, each hand-off: bitwise
//   (c) P4: drop_next_handoff() - the step throws within the bounds (never hangs), every later step throws until
//       reset(), and after reset() the run is again bitwise (a)'s
//   (d) the landing layout: pp_landing_layout(4096, 0) at a MoE cut - h only (the receiving res_norm _NF recomputes
//       Σ); h + S x 2048 fp32 slices at a dense cut
// Exit 77 (SKIP) with one GPU, or without the synthetic checkpoint ((b) alone SKIPs without the RTN export).
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "lfm2moe_rig.h"
#include "runtime/control.h"
#include "runtime/lfm2moe/lfm2moe_capture.h"

namespace {
namespace rl = runtime::lfm2moe;
constexpr uint32_t kMaxLen = 4096, kGen = 32;

struct Snapshot {
  std::vector<uint32_t> tokens, routes;
  std::vector<float> logits;
  std::vector<std::vector<uint16_t>> kv, rings;
  bool operator==(const Snapshot& o) const {
    return tokens == o.tokens && routes == o.routes && logits == o.logits && kv == o.kv && rings == o.rings;
  }
};

void controls_equal(rl::Lfm2MoeEngine& e, const char* phase) {
  if (e.devices() < 2) return;
  const runtime::Control a = e.control(0), b = e.control(1);
  if (std::memcmp(&a, &b, sizeof a) != 0) {
    std::fprintf(stderr, "the two Control blocks differ after %s (pos %u / %u)\n", phase, a.pos, b.pos);
    std::exit(1);
  }
}

Snapshot run(rl::Lfm2MoeEngine& e, const std::vector<uint32_t>& ids) {
  const model::Lfm2MoeDesc& d = e.model().desc;
  e.reset();
  controls_equal(e, "reset");
  e.ingest(ids);
  controls_equal(e, "ingest");
  Snapshot s;
  s.tokens = e.generate(kGen);
  controls_equal(e, "generate");
  s.logits = e.read_logits();
  s.routes = e.read_routes();
  for (uint32_t a = 0; a < d.attn_layers(); ++a) {
    std::vector<uint16_t> k = e.read_kv(a, 0, e.pos(), false), v = e.read_kv(a, 0, e.pos(), true);
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(std::move(k));
  }
  for (uint32_t c = 0; c < d.conv_layers(); ++c) s.rings.push_back(e.read_conv_ring(c));
  return s;
}

Snapshot with(lfm2moe_rig::Rig& rig, const std::string& snap, const std::vector<uint32_t>& ids, uint32_t devices,
              uint32_t split, runtime::PpHandoff h) {
  lfm2moe_rig::Options o;
  o.max_len = kMaxLen;
  o.devices = devices;
  o.split = split;
  o.handoff = h;
  o.timeout_ms = 5000;   // P4's lost hand-off fails within 5 s, not the default 30
  lfm2moe_rig::build(rig, snap, o);
  const Snapshot s = run(*rig.eng, ids);
  std::printf("  %s: %u device(s)%s, %zu launches, tokens", snap.c_str(), devices,
              devices == 2 ? (std::string(", split ") + std::to_string(rig.eng->split()) + " " +
                              runtime::pp_handoff_name(rig.eng->handoff())).c_str() : "",
              rig.eng->launches());
  for (uint32_t i = 0; i < 6 && i < s.tokens.size(); ++i) std::printf(" %u", s.tokens[i]);
  std::printf(" ...\n");
  return s;
}
}  // namespace

int main(int argc, char** argv) {
  // (d) the landing layout, host arithmetic (before the GPU check: it needs none)
  {
    const model::Lfm2MoeDesc& d = model::lfm2moe();
    const runtime::PpLandingLayout moe = rl::landing_layout(d, model::LfmPlacement::two(d, 20));
    const runtime::PpLandingLayout ref = runtime::pp_landing_layout(4096, 0);
    CHECK_EQ(moe.resid_bytes, size_t(4096));
    CHECK_EQ(moe.sumsq_bytes, size_t(0));
    CHECK_EQ(moe.total, ref.total);
    CHECK_EQ(rl::cut_pending_s(d, model::LfmPlacement::two(d, 20)), 0u);
    const runtime::PpLandingLayout dense = rl::landing_layout(d, model::LfmPlacement::two(d, 2));
    CHECK_EQ(rl::cut_pending_s(d, model::LfmPlacement::two(d, 2)), 1u);   // 24b's dense down: S 1
    CHECK_EQ(dense.sumsq_bytes, size_t(2048) * 4);
    std::printf("(d) landing: a MoE cut carries h alone (pp_landing_layout(4096, 0), %zu B); a dense cut h + the down "
                "slices (%zu B)\n", moe.total, dense.total);
  }
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two GPUs needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const std::string synth = lfm2moe_rig::lfm2moe_snapshot(argc > 1 ? argv[1] : "");
  const std::string rtn = lfm2moe_rig::lfm2moe_snapshot(argc > 2 ? argv[2] : "");
  if (synth.empty()) {
    std::printf("SKIP: no synthetic LFM2.5-24B-A2B checkpoint at '%s' (lfm2moe_oracle.sh synth-ckpt)\n",
                argc > 1 ? argv[1] : "");
    return 77;
  }
  const std::vector<uint32_t> ids = lfm2moe_rig::read_ids(argc > 3 ? argv[3] : "", 128000);
  CHECK(!ids.empty());
  lfm2moe_rig::Rig rig;
  const runtime::PpHandoff hs[] = {runtime::PpHandoff::Copy, runtime::PpHandoff::Peer};

  // (a) one card against two, each hand-off, a dense and a MoE cut
  std::printf("(a) synthetic N 4: --pp 1 against --pp 2 --pipeline-split 2 (dense cut) and 3 (MoE cut)\n");
  const Snapshot one = with(rig, synth, ids, 1, 0, runtime::kDefaultPpHandoff);
  for (uint32_t split : {2u, 3u})
    for (runtime::PpHandoff h : hs) {
      const Snapshot two = with(rig, synth, ids, 2, split, h);
      CHECK(two == one);
      CHECK(run(*rig.eng, ids) == two);   // replay determinism on two cards
      // (c) P4: a hand-off that never arrives
      rig.eng->ingest({ids[0]});
      rig.eng->drop_next_handoff();
      bool threw = false;
      try {
        rig.eng->ingest({ids[1]});
      } catch (const std::runtime_error& e) {
        threw = true;
        std::printf("  P4 (split %u, %s): %s\n", split, runtime::pp_handoff_name(h), e.what());
      }
      CHECK(threw);
      threw = false;
      try {
        rig.eng->ingest({ids[2]});
      } catch (const std::runtime_error&) {
        threw = true;
      }
      CHECK(threw);                        // until reset()
      CHECK(run(*rig.eng, ids) == one);   // reset() recovers, bitwise
    }
  // (b) the RTN export: split 20 against 19
  if (rtn.empty()) {
    std::printf("(b) SKIP: no RTN export at '%s' (24q)\n", argc > 2 ? argv[2] : "");
  } else {
    std::printf("(b) RTN: --pipeline-split 20 against 19\n");
    for (runtime::PpHandoff h : hs) {
      const Snapshot a = with(rig, rtn, ids, 2, 20, h);
      const Snapshot b = with(rig, rtn, ids, 2, 19, h);
      CHECK(a == b);
    }
  }
  std::puts("lfm2moe_pp_test OK");
  return 0;
}
