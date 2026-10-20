// Spec 24c Task 4 (F4 on decode) and Task 5 (the fused arms): Lfm2MoeEngine on one card. Box only (labels
// checkpoint;golden;lfm2moe).
//   lfm2moe_decode_test <checkpoint> <ids> [bf16] [eager] [--fuse route|router|conv|all]
// <checkpoint>: a synthetic one (24b's make_synth.py --layers 4) or the real export (24q's).
//
//   1. the plan == the allocation, component by component (runtime::lfm2moe::plan vs memory_use); the list's
//      launches == decode_launches (41 at the synthetic N 4, 383 at 40 layers; eager +2 an attention layer)
//   2. F4, replay determinism: two reset + ingest + generate(32) runs give bitwise equal tokens, logits, route rows,
//      every conv layer's ring, every attention layer's KV rows, both residual buffers (the ping-pong), and every
//      layer's tap (Review Focus 3: the residual after each layer)
//   3. --fuse <arms> (Task 5, Review Focus 4): an engine with the arms (in_proj in the conv-tile layout under conv)
//      against the as-built engine: the same tokens, logits, routes, conv rings, KV and taps, bitwise; the launch
//      counts 345 / 345 / 353 / 277 at 40 layers
// Exit 77 (SKIP) when the checkpoint is absent or does not fit one card.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "lfm2moe_rig.h"
#include "loader/lfm2moe_layout.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

namespace {
namespace rl = runtime::lfm2moe;
constexpr uint32_t kMaxLen = 4096, kGen = 32;

struct Snapshot {
  std::vector<uint32_t> tokens, routes;
  std::vector<float> logits;
  std::vector<std::vector<uint16_t>> rings, kv;
  std::vector<uint16_t> residuals, taps;
};

Snapshot run(rl::Lfm2MoeEngine& e, const std::vector<uint32_t>& ids) {
  const model::Lfm2MoeDesc& d = e.model().desc;
  e.reset();
  Snapshot s;
  // The prompt one replay at a time, every layer's tap kept (the residual after each layer, Review Focus 3).
  for (uint32_t id : ids) {
    e.ingest({id});
    const std::vector<uint16_t> t = e.read_debug_h();
    s.taps.insert(s.taps.end(), t.begin(), t.end());
  }
  s.tokens = e.generate(kGen);
  s.logits = e.read_logits();
  s.routes = e.read_routes();
  for (uint32_t c = 0; c < d.conv_layers(); ++c) s.rings.push_back(e.read_conv_ring(c));
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

void same(const Snapshot& a, const Snapshot& b, const char* what) {
  CHECK(a.tokens == b.tokens);
  CHECK(a.logits == b.logits);
  CHECK(a.routes == b.routes);
  CHECK(a.rings == b.rings);
  CHECK(a.kv == b.kv);
  CHECK(a.taps == b.taps);
  std::printf("%s: tokens, logits, %zu route words, %zu conv rings, %zu attention layers' KV and %zu tap values bitwise\n",
              what, a.routes.size(), a.rings.size(), a.kv.size(), a.taps.size());
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = lfm2moe_rig::lfm2moe_snapshot(argc > 1 ? argv[1] : "");
  if (snap.empty()) {
    std::printf("SKIP: no LFM2.5-24B-A2B checkpoint at '%s'\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  lfm2moe_rig::Options o;
  o.max_len = kMaxLen;
  o.debug_tap = true;
  bool fuse_given = false;
  rl::LfmFuse arms{};
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "bf16") {
      o.int8_head = false;
    } else if (a == "eager") {
      o.attn = rl::LfmAttn::Eager;
    } else if (a == "--fuse" && i + 1 < argc) {
      arms = rl::parse_fuse(argv[++i]);
      fuse_given = true;
    } else {
      std::fprintf(stderr, "lfm2moe_decode_test: unknown argument %s (bf16, eager, --fuse <arms>)\n", a.c_str());
      return 2;
    }
  }
  const model::Lfm2MoeDesc d0 = loader::lfm2moe_checkpoint_desc(snap);
  if (!lfm2moe_rig::fits_one_card(d0, o.int8_head, kMaxLen)) {
    std::printf("SKIP: %s (%u layers) does not fit one card\n", snap.c_str(), d0.layers);
    return 77;
  }
  const std::vector<uint32_t> ids = lfm2moe_rig::read_ids(argc > 2 ? argv[2] : "", d0.vocab);
  CHECK(!ids.empty());
  lfm2moe_rig::Rig rig;
  lfm2moe_rig::build(rig, snap, o);
  rl::Lfm2MoeEngine& e = *rig.eng;
  const model::Lfm2MoeDesc& d = e.model().desc;

  // 1. plan == allocation; the launch count
  {
    const rl::Plan pl = rl::plan(d, e.model().place, 0, kMaxLen, o.int8_head, e.attention());
    const runtime::MemoryComponents got = e.memory_use(0);
    CHECK_EQ(got.model, pl.weights + pl.rope);
    CHECK_EQ(got.kv, pl.kv);
    CHECK_EQ(got.decode_state, pl.conv_rings + pl.decode_scratch + rl::tap_bytes(d));   // + the debug tap
    CHECK_EQ(e.launches(), rl::decode_launches(d, e.model().place, e.attention(), e.fuse(), e.handoff()));
    std::printf("plan == allocation; %zu launches a token (%s attention, fuse %s); %s\n", e.launches(),
                rl::lfm_attn_name(e.attention()), rl::lfm_fuse_name(e.fuse()).c_str(), e.memory_line().c_str());
  }
  // 2. F4: replay determinism (the residual buffers too)
  const Snapshot a = run(e, ids), b = run(e, ids);
  same(a, b, "F4: two runs");
  CHECK(a.residuals == b.residuals);
  std::printf("    both residual buffers equal after the runs; tokens:");
  for (uint32_t t : a.tokens) std::printf(" %u", t);
  std::printf("\n");
  // 3. the fused arms against the as-built list
  if (fuse_given) {
    lfm2moe_rig::Options of = o;
    of.fuse = arms;
    lfm2moe_rig::build(rig, snap, of);
    rl::Lfm2MoeEngine& f = *rig.eng;
    CHECK_EQ(f.launches(), rl::decode_launches(f.model().desc, f.model().place, f.attention(), arms, f.handoff()));
    if (d.layers == 40) {
      const size_t want = arms.route && arms.router && arms.conv ? 277
                          : (arms.route || arms.router) && !arms.conv && !(arms.route && arms.router) ? 345
                          : arms.conv && !arms.route && !arms.router ? 353 : f.launches();
      CHECK_EQ(f.launches(), want);
    }
    const Snapshot c = run(f, ids);
    same(a, c, ("Review Focus 4: fuse " + rl::lfm_fuse_name(arms) + " against none (" + std::to_string(f.launches()) +
                " launches against " + std::to_string(e.launches()) + ")").c_str());
  }
  std::puts("lfm2moe_decode_test OK");
  return 0;
}
