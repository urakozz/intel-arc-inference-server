// Spec 25e (plan 25e Task 4 Step 1, Review Focus 4): BoltEngine's prefix-cache snapshot calls on the card with a checkpoint - box
// only (labels checkpoint;bolt). 25b's synthetic checkpoints (tools/quantize/bolt/make_synth.py: n12 = twelve real-width blocks of
// Bolt's pattern) carry it before the 41.9 GB export is loaded; SergiioB's on two cards after. The host side (the layouts, the
// adapter, spec 7's request path) is bolt_snapshot_test's.
//
//   bolt_snapshot_gpu_test <checkpoint> <ids> [pp2[:split]] [cross] [b16]
//
// The prompt: <ids> cycled to kN = 5200 positions (3 chunks: 2048, 2048, 1104; the conv ring wraps 325 times); 32 greedy ids after
// it. The block hook is set in every run (so every run's chunks end at the same block ends).
//   default (one card, or two with pp2):
//     0. state_bytes() / kv_bytes(n) are the plan's (170,229,760 B (86,343,680 with b16) and n x 8,192 B at the real shapes);
//     a. a restore at a BLOCK END (2048 and 4096, from the hook's own saves of the cold run): reset, load_kv(0, at) +
//        load_state(at), prefill the tail, 32 greedy ids - the ids, the last logits, every attention block's KV [0, n), every
//        Mamba block's last three ring columns and SSM state BITWISE the cold run's;
//     b. a restore at a REQUEST END (2049, 5000: a session that prefilled [0, at), saved there, then prefilled the rest): reset,
//        load, the same tail prefill and 32 ids - bitwise the session that never left (the same chunks);
//     c. the same at the tiny positions 0, 1, 2, 3 (the state's columns below position 0 are zeros the conv never reads) and 300.
//   cross (two GPUs; spec 16b's rule): a snapshot taken under --pp 2 (split pp2:<s>, default 3) restores under --pp 1 and
//     continues bitwise as the two-card session does, and the reverse - one host layout (the synthetic only: the real model does
//     not fit one card).
// Exit 77 (SKIP) when the checkpoint is absent, does not fit the asked placement, or two GPUs are needed.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "bolt_rig.h"
#include "check.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/control.h"

namespace {

namespace rb = runtime::bolt;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 5200, kGen = 32;

struct Saved {
  std::vector<uint8_t> state, kv;   // the state at `at` and the KV [0, at)
};

Saved save(rb::BoltEngine& e) {
  Saved s;
  const uint32_t at = e.pos();
  s.state.resize(e.state_bytes());
  s.kv.resize(e.kv_bytes(at));
  e.save_state(s.state.data());
  if (at > 0) e.save_kv(0, at, s.kv.data());
  return s;
}

void restore(rb::BoltEngine& e, const Saved& s, uint32_t at) {
  if (at > 0) e.load_kv(0, at, s.kv.data());
  e.load_state(s.state.data(), at);
  CHECK_EQ(e.pos(), at);
}

// What a continuation leaves: the ids, the last logits, every attention block's K and V rows [0, n), every Mamba block's ring
// columns of the last three positions and its SSM state (the rest of a restored ring is stale, as the host test's slots are).
struct Out {
  Ids ids;
  std::vector<float> logits;
  std::vector<std::vector<uint16_t>> kv, rings;
  std::vector<std::vector<float>> states;
  bool operator==(const Out& o) const {
    return ids == o.ids && logits == o.logits && kv == o.kv && rings == o.rings && states == o.states;
  }
};

Out finish(rb::BoltEngine& e) {
  Out o;
  o.ids = e.generate(kGen);
  o.logits = e.read_logits();
  const model::BoltDesc& d = e.model().desc;
  const uint32_t n = e.pos();
  for (uint32_t a = 0; a < d.attn_layers(); ++a) {
    std::vector<uint16_t> k = e.read_kv(a, 0, n, false), v = e.read_kv(a, 0, n, true);
    k.insert(k.end(), v.begin(), v.end());
    o.kv.push_back(std::move(k));
  }
  for (uint32_t m = 0; m < d.mamba_layers(); ++m) {
    const std::vector<uint16_t> ring = e.read_conv_ring(m);   // bf16 [16][conv_dim]
    std::vector<uint16_t> cols;
    for (uint32_t back : {1u, 2u, 3u}) {
      const uint32_t slot = (n - back) % 16;
      cols.insert(cols.end(), ring.begin() + size_t(slot) * d.m_conv_dim, ring.begin() + size_t(slot + 1) * d.m_conv_dim);
    }
    o.rings.push_back(std::move(cols));
    o.states.push_back(e.read_ssm_state(m));
  }
  return o;
}

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }

bool report(const char* what, const Out& a, const Out& b) {
  const bool ok = a == b;
  std::printf("  %s: ids %s, logits %s, KV %s, conv rings %s, SSM states %s\n", what, a.ids == b.ids ? "bitwise" : "DIFFER",
              a.logits == b.logits ? "bitwise" : "DIFFER", a.kv == b.kv ? "bitwise" : "DIFFER",
              a.rings == b.rings ? "bitwise" : "DIFFER", a.states == b.states ? "bitwise" : "DIFFER");
  return ok;
}

int run_default(rb::BoltEngine& e, const Ids& prompt) {
  const model::BoltDesc& d = e.model().desc;
  bool ok = true;
  // 0. the sizes
  CHECK_EQ(e.state_bytes(), rb::state_snapshot_bytes(d, e.state()));
  CHECK_EQ(e.kv_bytes(7), rb::kv_snapshot_bytes(d, 7));
  std::printf("0. state %zu B, KV %zu B a position (%u Mamba + %u attention blocks, %s SSM state)\n", e.state_bytes(),
              e.kv_bytes(1), d.mamba_layers(), d.attn_layers(), rb::bolt_state_name(e.state()));

  // a. a block end: the hook saves at 2048 and 4096 during the cold prefill.
  std::map<uint32_t, Saved> at_hook;
  e.set_block_hook([&](uint32_t end, bool) { at_hook[end] = save(e); });
  e.reset();
  e.prefill(prompt);
  CHECK(at_hook.count(2048) && at_hook.count(4096) && at_hook.count(kN));
  const Out cold = finish(e);
  for (uint32_t at : {2048u, 4096u}) {
    e.reset();
    restore(e, at_hook.at(at), at);
    e.prefill(slice(prompt, at, kN));
    ok = report(("a. restore at the block end " + std::to_string(at)).c_str(), finish(e), cold) && ok;
  }

  e.set_block_hook({});   // parts b and c do not need the block-end saves (a full KV copy to the host at every chunk end)

  // b / c. request ends: the uncached session (prefill to `at`, save, prefill the rest) against a restore.
  for (uint32_t at : {2049u, 5000u, 300u, 3u, 2u, 1u, 0u}) {
    e.reset();
    if (at > 0) e.prefill(slice(prompt, 0, at));
    const Saved s = save(e);
    e.prefill(slice(prompt, at, kN));
    const Out ref = finish(e);
    e.reset();
    // another session's leftovers in the rings, the states and the KV: the restore must not depend on a clean engine
    e.prefill(slice(prompt, 17, 17 + 600));
    e.reset();
    e.prefill(slice(prompt, 40, 40 + 300));
    restore(e, s, at);
    e.prefill(slice(prompt, at, kN));
    const std::string what = std::string(at >= 2049 ? "b" : "c") + ". restore at the request end " + std::to_string(at);
    ok = report(what.c_str(), finish(e), ref) && ok;
  }
  return ok ? 0 : 1;
}

int run_cross(bolt_rig::Rig& rig, const std::string& snap, bolt_rig::Options o, const Ids& prompt) {
  bool ok = true;
  const uint32_t at = 2049;
  for (int dir = 0; dir < 2; ++dir) {
    bolt_rig::Options from = o, to = o;
    from.devices = dir == 0 ? 2 : 1;
    to.devices = dir == 0 ? 1 : 2;
    if (from.devices == 2 && !from.split) from.split = 3;
    if (to.devices == 2 && !to.split) to.split = 3;
    bolt_rig::build(rig, snap, from);
    rb::BoltEngine& a = *rig.eng;
    a.reset();
    a.prefill(slice(prompt, 0, at));
    const Saved s = save(a);
    const size_t state_bytes = a.state_bytes(), kv_bytes = a.kv_bytes(at);
    a.prefill(slice(prompt, at, kN));
    const Out ref = finish(a);
    bolt_rig::build(rig, snap, to);
    rb::BoltEngine& b = *rig.eng;
    CHECK_EQ(b.state_bytes(), state_bytes);
    CHECK_EQ(b.kv_bytes(at), kv_bytes);
    b.reset();
    restore(b, s, at);
    b.prefill(slice(prompt, at, kN));
    ok = report(dir == 0 ? "--pp 2 snapshot restored under --pp 1" : "--pp 1 snapshot restored under --pp 2", finish(b), ref) &&
         ok;
  }
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snap = bolt_rig::bolt_snapshot(argc > 1 ? argv[1] : "");
  if (snap.empty()) {
    std::printf("SKIP: no Nemotron 3.5 Bolt checkpoint at '%s' (the synthetic sets: tools/box_validate/bolt_synth.sh; the real "
                "one: SergiioB's export)\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  bolt_rig::Options o;
  o.max_len = kMaxLen;
  bool cross = false;
  for (int i = 3; i < argc; ++i) {
    const std::string x = argv[i];
    if (x == "pp2") {
      o.devices = 2;
    } else if (x.rfind("pp2:", 0) == 0) {
      o.devices = 2;
      o.split = uint32_t(std::strtoul(x.c_str() + 4, nullptr, 10));
    } else if (x == "cross") {
      cross = true;
    } else if (x == "b16") {
      o.state = rb::BoltState::Bf16;
    } else {
      std::fprintf(stderr, "bolt_snapshot_gpu_test: unknown flag %s (pp2[:split], cross, b16)\n", x.c_str());
      return 2;
    }
  }
  const model::BoltDesc pre = loader::bolt_checkpoint_desc(snap, 0);
  if ((cross || o.devices == 2) && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  if ((cross || o.devices == 1) && !bolt_rig::fits(pre, o, /*prefill=*/true)) {
    std::printf("SKIP: %s (%u blocks) does not fit one card with the prefill scratch\n", snap.c_str(), pre.layers);
    return 77;
  }
  const Ids base = bolt_rig::read_ids(argc > 2 ? argv[2] : "", pre.vocab);
  CHECK(!base.empty());
  Ids ids(kN);
  for (uint32_t i = 0; i < kN; ++i) ids[i] = base[i % base.size()];
  bolt_rig::Rig rig;
  if (cross) {
    const int rc = run_cross(rig, snap, o, ids);
    std::printf("bolt_snapshot_gpu_test cross %s\n", rc == 0 ? "OK" : "FAILED");
    return rc;
  }
  bolt_rig::build(rig, snap, o);
  rb::BoltEngine& e = *rig.eng;
  std::printf("bolt_snapshot_gpu_test: %u ids, %u blocks, %u device(s)%s\n", kN, e.model().desc.layers, e.devices(),
              e.devices() == 2 ? (std::string(", split ") + std::to_string(e.split())).c_str() : "");
  const int rc = run_default(e, ids);
  std::printf("bolt_snapshot_gpu_test %s\n", rc == 0 ? "OK" : "FAILED");
  return rc;
}
