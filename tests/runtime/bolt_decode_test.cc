// Spec 25c Task 4 (B4 on decode), Task 5 (the fused arms) and Review Focus 4 / 6 (the norm sites, the two-card cut): BoltEngine
// on the card. Box only (labels checkpoint;golden;bolt).
//   bolt_decode_test <checkpoint> <ids> [bf16] [eager] [b16] [layers:<N>] [--fuse router|route|conv|all,...]
//                    [--pp <split a>,<split b>]
// <checkpoint>: 25b's synthetic (make_synth.py n12-*: 12 blocks, one card) or our real g64 export (two cards).
//
//   1. the plan == the allocation, device by device (runtime::bolt::device_plan's kv + ssm + conv_rings + decode_scratch =
//      the buffers' persistent group + scratch); the list's launches == decode_launches (92 at the synthetic N 12, 653 at
//      88 blocks; eager 645; the arms -40 each)
//   2. B4, replay determinism: two reset + ingest + generate(32) runs give bitwise equal tokens, logits, route rows, every
//      Mamba block's ring and SSM state, every attention block's KV rows, both residual buffers (the ping-pong), and every
//      block's tap after every prompt id (Review Focus 4: the residual after each block, the alternation asserted by the
//      capture)
//   3. --fuse <arms> (Task 5, Review Focus 5): an engine with the arms against the as-built engine - the same tokens, logits,
//      routes, rings, states, KV and taps, bitwise
//   4. --pp a,b (Review Focus 6): the model on two cards at split a (copy) and split b (peer) - bitwise each other and, when
//      the model fits one card, bitwise the one-card run
// Exit 77 (SKIP) when the checkpoint is absent or the placement cannot run (one GPU for --pp, a model too large).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bolt_rig.h"
#include "check.h"
#include "runtime/bolt/bolt_sizes.h"

namespace {
namespace rb = runtime::bolt;
constexpr uint32_t kMaxLen = 4096, kGen = 32;

struct Snapshot {
  std::vector<uint32_t> tokens, routes;
  std::vector<float> logits;
  std::vector<std::vector<uint16_t>> rings, kv;
  std::vector<std::vector<float>> states;
  std::vector<uint16_t> residuals, taps;
};

Snapshot run(rb::BoltEngine& e, const std::vector<uint32_t>& ids) {
  const model::BoltDesc& d = e.model().desc;
  e.reset();
  Snapshot s;
  for (uint32_t id : ids) {   // the prompt one replay at a time, every block's tap kept
    e.ingest({id});
    const std::vector<uint16_t> t = e.read_debug_h();
    s.taps.insert(s.taps.end(), t.begin(), t.end());
  }
  s.tokens = e.generate(kGen);
  s.logits = e.read_logits();
  s.routes = e.read_routes();
  for (uint32_t m = 0; m < d.mamba_layers(); ++m) {
    s.rings.push_back(e.read_conv_ring(m));
    s.states.push_back(e.read_ssm_state(m));
  }
  for (uint32_t a = 0; a < d.attn_layers(); ++a) {
    std::vector<uint16_t> k = e.read_kv(a, 0, e.pos(), false), v = e.read_kv(a, 0, e.pos(), true);
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(std::move(k));
  }
  for (uint32_t dev = 0; dev < e.devices(); ++dev) {
    const std::vector<uint16_t> r = e.read_residuals(dev);
    s.residuals.insert(s.residuals.end(), r.begin(), r.end());
  }
  return s;
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

void same(const Snapshot& a, const Snapshot& b, const char* what) {
  CHECK(a.tokens == b.tokens);
  CHECK(bits_equal(a.logits, b.logits));
  CHECK(a.routes == b.routes);
  CHECK(a.rings == b.rings);
  CHECK(a.states.size() == b.states.size());
  for (size_t i = 0; i < a.states.size(); ++i) CHECK(bits_equal(a.states[i], b.states[i]));
  CHECK(a.kv == b.kv);
  CHECK(a.taps == b.taps);
  std::printf("%s: tokens, logits, %zu route words, %zu rings and SSM states, %zu attention blocks' KV and %zu tap values "
              "bitwise\n", what, a.routes.size(), a.rings.size(), a.kv.size(), a.taps.size());
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = bolt_rig::bolt_snapshot(argc > 1 ? argv[1] : "");
  if (snap.empty()) {
    std::printf("SKIP: no Nemotron 3.5 Bolt checkpoint at '%s'\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  bolt_rig::Options o;
  o.max_len = kMaxLen;
  o.debug_tap = true;
  bool fuse_given = false;
  rb::BoltFuse arms{};
  uint32_t pp_a = 0, pp_b = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "bf16") {
      o.int8_head = false;
    } else if (a == "eager") {
      o.attn = rb::BoltAttn::Eager;
    } else if (a == "b16") {
      o.state = rb::BoltState::Bf16;
    } else if (a.rfind("layers:", 0) == 0) {
      o.layers = uint32_t(std::strtoul(a.c_str() + 7, nullptr, 10));
    } else if (a == "--fuse" && i + 1 < argc) {
      arms = rb::parse_fuse(argv[++i]);
      fuse_given = true;
    } else if (a == "--pp" && i + 1 < argc) {
      const std::string v = argv[++i];
      pp_a = uint32_t(std::strtoul(v.c_str(), nullptr, 10));
      pp_b = uint32_t(std::strtoul(v.c_str() + v.find(',') + 1, nullptr, 10));
    } else {
      std::fprintf(stderr, "bolt_decode_test: unknown argument %s (bf16, eager, b16, layers:N, --fuse <arms>, --pp a,b)\n",
                   a.c_str());
      return 2;
    }
  }
  const model::BoltDesc d0 = loader::bolt_checkpoint_desc(snap, o.layers);
  const bool one_fits = bolt_rig::fits(d0, o);
  if (!one_fits) {   // the real model: two cards (Decision 4)
    if (l0::Context::gpu_count() < 2) {
      std::printf("SKIP: %s (%u blocks) needs two cards, Level Zero shows %u\n", snap.c_str(), d0.layers,
                  l0::Context::gpu_count());
      return 77;
    }
    o.devices = 2;
  }
  const std::vector<uint32_t> ids = bolt_rig::read_ids(argc > 2 ? argv[2] : "", d0.vocab);
  CHECK(!ids.empty());
  bolt_rig::Rig rig;
  bolt_rig::build(rig, snap, o);
  rb::BoltEngine& e = *rig.eng;
  const model::BoltDesc& d = e.model().desc;

  // 1. plan == allocation; the launch count
  for (uint32_t dev = 0; dev < e.devices(); ++dev) {
    rb::BoltReserve r;
    r.decode_scratch = rb::decode_scratch_bytes(d, e.model().place, dev, kMaxLen, e.attention(), e.state());
    r.bf16_state = e.state() == rb::BoltState::Bf16;
    const rb::DevicePlan pl = rb::device_plan(d, e.model().place, dev, kMaxLen, o.int8_head, r, 0);
    const runtime::MemoryComponents got = e.memory_use(dev);
    CHECK_EQ(got.model, pl.weights);
    CHECK_EQ(e.persistent_bytes(dev) + e.scratch_bytes(dev), pl.kv + pl.ssm + pl.conv_rings + pl.decode_scratch);
  }
  CHECK_EQ(e.launches(), rb::decode_launches(d, e.model().place, e.attention(), e.fuse(), e.handoff()));
  if (d.layers == 88) CHECK_EQ(e.launches(), size_t(e.attention() == rb::BoltAttn::Eager ? 645 : 653));
  if (d.layers == 12 && e.devices() == 1 && e.attention() == rb::BoltAttn::Flash) CHECK_EQ(e.launches(), size_t(92));
  std::printf("plan == allocation; %zu launches a token (%s attention, fuse %s, %s state); %s\n", e.launches(),
              rb::bolt_attn_name(e.attention()), rb::bolt_fuse_name(e.fuse()).c_str(), rb::bolt_state_name(e.state()),
              e.memory_line().c_str());

  // 2. B4: replay determinism (the residual buffers too)
  const Snapshot a = run(e, ids), b = run(e, ids);
  same(a, b, "B4: two runs");
  CHECK(a.residuals == b.residuals);
  std::printf("    both residual buffers equal after the runs; tokens:");
  for (uint32_t t : a.tokens) std::printf(" %u", t);
  std::printf("\n");
  // `e` and `d` belong to the engine a rebuild destroys: keep what is needed first.
  const size_t base_launches = e.launches();
  const uint32_t layers = d.layers, devices = e.devices();

  // 3. the fused arms against the as-built list
  if (fuse_given) {
    bolt_rig::Options of = o;
    of.fuse = arms;
    bolt_rig::build(rig, snap, of);
    rb::BoltEngine& f = *rig.eng;
    CHECK_EQ(f.launches(), rb::decode_launches(f.model().desc, f.model().place, f.attention(), arms, f.handoff()));
    if (layers == 88 && devices == 2 && f.attention() == rb::BoltAttn::Flash) {
      const int n = int(arms.router) + int(arms.route) + int(arms.conv);
      CHECK_EQ(f.launches(), size_t(653 - 40 * n));   // spec 25 §6.1: 613 a single arm, 533 all three
    }
    const Snapshot c = run(f, ids);
    same(a, c, ("Review Focus 5: fuse " + rb::bolt_fuse_name(arms) + " against none (" + std::to_string(f.launches()) +
                " launches against " + std::to_string(base_launches) + ")").c_str());
  }

  // 4. the two-card cut at two splits, copy and peer (and against one card when it fits)
  if (pp_a && pp_b) {
    if (l0::Context::gpu_count() < 2) {
      std::printf("SKIP (part 4): --pp needs two GPUs, Level Zero shows %u\n", l0::Context::gpu_count());
    } else {
      std::vector<Snapshot> pp;
      for (int k = 0; k < 2; ++k) {
        bolt_rig::Options op = o;
        op.devices = 2;
        op.split = k ? pp_b : pp_a;
        op.handoff = k ? runtime::PpHandoff::Peer : runtime::PpHandoff::Copy;
        bolt_rig::build(rig, snap, op);
        rb::BoltEngine& p = *rig.eng;
        CHECK_EQ(p.launches(), rb::decode_launches(p.model().desc, p.model().place, p.attention(), p.fuse(), p.handoff()));
        pp.push_back(run(p, ids));
        std::printf("    two cards, split %u (%s), %zu launches\n", op.split, k ? "peer" : "copy", p.launches());
      }
      same(pp[0], pp[1], ("Review Focus 6: split " + std::to_string(pp_a) + " (copy) against split " + std::to_string(pp_b) +
                          " (peer)").c_str());
      if (devices == 1) same(a, pp[0], "Review Focus 6: two cards against one");
    }
  }
  std::puts("bolt_decode_test OK");
  return 0;
}
