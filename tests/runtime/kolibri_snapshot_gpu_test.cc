// Spec 20e (plan 20e Task 3 Step 1, Review Focus 4): KolibriEngine's prefix-cache snapshot calls on the card
// with a checkpoint - box only (labels checkpoint;kolibri). The synthetic checkpoints
// (tools/box_validate/kolibri_oracle.sh synth: 5 real-width layers - 4 sliding, 1 full) carry it before spec
// 20b; the real one runs on two cards after it. The host side (the layouts, the adapter, spec 7's request path)
// is kolibri_snapshot_test's.
//
//   kolibri_snapshot_gpu_test <checkpoint> <ids> [pp2[:split]] [cross]
//
// The prompt: <ids> cycled to kN = 5000 positions (3 chunks; the window wraps the 4096-slot ring); 32 greedy
// ids after it. The block hook is set in every run (so every run's chunks end at the same block ends).
//   default (one card, or two with pp2):
//     0. state_bytes() / kv_bytes(n) are the plan's (41,943,040 B and n x 20,480 B at the real shapes);
//     a. a restore at a BLOCK END (4096, from the hook's own saves of the cold run): reset, load_kv(0, 4096)
//        + load_state(at 4096), prefill the tail, 32 greedy ids - the ids, the last logits, every full
//        layer's KV [0, n) and every ring's rows of [4096 - 512, n) BITWISE the cold run's;
//     b. a restore at a REQUEST END (2049: a session that prefilled [0, 2049), saved there, then prefilled
//        the rest): reset, load, the same tail prefill and 32 ids - bitwise the session that never left
//        (the same chunks: a request-end restore continues the uncached session exactly; against ONE cold
//        prefill of the whole prompt it differs only where 20d's split rule allows, a split at 2049 not
//        being a multiple of 64);
//     c. the same at 300 (< 512: the state's rows below position 0 are zeros the window never reads).
//   cross (two GPUs; spec 16b's rule): a snapshot taken under --pp 2 (split pp2:<s>, default 3) restores under
//     --pp 1 and continues bitwise as the two-card session does, and the reverse - one host layout.
// Exit 77 (SKIP) when the checkpoint is absent, does not fit the asked placement, or two GPUs are needed.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "check.h"
#include "kolibri_rig.h"
#include "runtime/control.h"
#include "runtime/kolibri/kolibri_sizes.h"

namespace {

namespace rk = runtime::kolibri;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 5000, kGen = 32;

struct Saved {
  std::vector<uint8_t> state, kv;   // the state at `at` and the KV [0, at)
};

Saved save(rk::KolibriEngine& e) {
  Saved s;
  const uint32_t at = e.pos();
  s.state.resize(e.state_bytes());
  s.kv.resize(e.kv_bytes(at));
  e.save_state(s.state.data());
  e.save_kv(0, at, s.kv.data());
  return s;
}

void restore(rk::KolibriEngine& e, const Saved& s, uint32_t at) {
  e.load_kv(0, at, s.kv.data());
  e.load_state(s.state.data(), at);
  CHECK_EQ(e.pos(), at);
}

// What a continuation leaves: the ids, the last logits, every full layer's KV [0, n) and every ring's rows
// of [lo, n) (lo: the oldest position both runs must hold - a restored ring is stale below at - 512).
struct Out {
  Ids ids;
  std::vector<float> logits;
  std::vector<std::vector<uint16_t>> kv;
  bool operator==(const Out& o) const { return ids == o.ids && logits == o.logits && kv == o.kv; }
};

Out finish(rk::KolibriEngine& e, uint32_t lo) {
  Out o;
  o.ids = e.generate(kGen);
  o.logits = e.read_logits();
  const model::Kolibri1Desc& d = e.model().desc;
  const uint32_t n = e.pos();
  for (uint32_t l = 0; l < d.layers; ++l) {
    uint32_t first = 0;
    if (d.is_sliding(l)) first = std::max<uint32_t>(lo, n > model::Kolibri1Desc::kRing ? n - model::Kolibri1Desc::kRing : 0);
    std::vector<uint16_t> k = e.read_kv(l, first, n - first, false), v = e.read_kv(l, first, n - first, true);
    k.insert(k.end(), v.begin(), v.end());
    o.kv.push_back(std::move(k));
  }
  return o;
}

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }

bool report(const char* what, const Out& a, const Out& b) {
  const bool ok = a == b;
  std::printf("  %s: ids %s, logits %s, KV + rings %s\n", what, a.ids == b.ids ? "bitwise" : "DIFFER",
              a.logits == b.logits ? "bitwise" : "DIFFER", a.kv == b.kv ? "bitwise" : "DIFFER");
  return ok;
}

int run_default(rk::KolibriEngine& e, const Ids& prompt) {
  const model::Kolibri1Desc& d = e.model().desc;
  bool ok = true;
  // 0. the sizes
  CHECK_EQ(e.state_bytes(), rk::state_snapshot_bytes(d));
  CHECK_EQ(e.kv_bytes(7), rk::kv_snapshot_bytes(d, 7));
  std::printf("0. state %zu B, KV %zu B a position (%u layers)\n", e.state_bytes(), e.kv_bytes(1), d.layers);

  // a. a block end: the hook saves at 2048 and 4096 during the cold prefill.
  std::map<uint32_t, Saved> at_hook;
  e.set_block_hook([&](uint32_t end, bool) { at_hook[end] = save(e); });
  e.reset();
  e.prefill(prompt);
  CHECK(at_hook.count(2048) && at_hook.count(4096) && at_hook.count(kN));
  const Out cold = finish(e, 4096 - 512);
  e.reset();
  restore(e, at_hook.at(4096), 4096);
  e.prefill(slice(prompt, 4096, kN));
  ok = report("a. restore at the block end 4096", finish(e, 4096 - 512), cold) && ok;

  // b / c. request ends: the uncached session (prefill to `at`, save, prefill the rest) against a restore.
  for (uint32_t at : {2049u, 300u}) {
    e.reset();
    e.prefill(slice(prompt, 0, at));
    const Saved s = save(e);
    e.prefill(slice(prompt, at, kN));
    const uint32_t lo = at > 512 ? at - 512 : 0;
    const Out ref = finish(e, lo);
    e.reset();
    restore(e, s, at);
    e.prefill(slice(prompt, at, kN));
    const std::string what = std::string(at == 2049 ? "b" : "c") + ". restore at the request end " + std::to_string(at);
    ok = report(what.c_str(), finish(e, lo), ref) && ok;
  }
  e.set_block_hook({});
  return ok ? 0 : 1;
}

int run_cross(kolibri_rig::Rig& rig, const std::string& snapdir, kolibri_rig::Options o, const Ids& prompt) {
  bool ok = true;
  const uint32_t at = 2049;
  for (int dir = 0; dir < 2; ++dir) {
    kolibri_rig::Options from = o, to = o;
    from.devices = dir == 0 ? 2 : 1;
    to.devices = dir == 0 ? 1 : 2;
    if (from.devices == 2 && !from.split) from.split = 3;
    if (to.devices == 2 && !to.split) to.split = 3;
    kolibri_rig::build(rig, snapdir, from);
    rk::KolibriEngine& a = *rig.eng;
    a.reset();
    a.prefill(slice(prompt, 0, at));
    const Saved s = save(a);
    const size_t state_bytes = a.state_bytes(), kv_bytes = a.kv_bytes(at);
    a.prefill(slice(prompt, at, kN));
    const Out ref = finish(a, at - 512);
    kolibri_rig::build(rig, snapdir, to);
    rk::KolibriEngine& b = *rig.eng;
    CHECK_EQ(b.state_bytes(), state_bytes);
    CHECK_EQ(b.kv_bytes(at), kv_bytes);
    b.reset();
    restore(b, s, at);
    b.prefill(slice(prompt, at, kN));
    ok = report(dir == 0 ? "--pp 2 snapshot restored under --pp 1" : "--pp 1 snapshot restored under --pp 2",
                finish(b, at - 512), ref) && ok;
  }
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snapdir = kolibri_rig::kolibri_snapshot(argc > 1 ? argv[1] : "");
  if (snapdir.empty()) {
    std::printf("SKIP: no Kolibri-1 checkpoint at '%s' (the synthetic sets: kolibri_oracle.sh synth; the real one: "
                "spec 20b)\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  kolibri_rig::Options o;
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
    } else {
      std::fprintf(stderr, "kolibri_snapshot_gpu_test: unknown flag %s (pp2[:split], cross)\n", x.c_str());
      return 2;
    }
  }
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(snapdir);
  if ((cross || o.devices == 2) && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  if ((cross || o.devices == 1) && !rk::fits_one_card(pre, false, size_t(32530000000ull), size_t(1.5e9), true)) {
    std::printf("SKIP: %s (%u layers) does not fit one card with the prefill scratch\n", snapdir.c_str(), pre.layers);
    return 77;
  }
  const Ids base = kolibri_rig::read_ids(argc > 2 ? argv[2] : "", pre.vocab);
  CHECK(!base.empty());
  Ids ids(kN);
  for (uint32_t i = 0; i < kN; ++i) ids[i] = base[i % base.size()];
  kolibri_rig::Rig rig;
  if (cross) {
    const int rc = run_cross(rig, snapdir, o, ids);
    std::printf("kolibri_snapshot_gpu_test cross %s\n", rc == 0 ? "OK" : "FAILED");
    return rc;
  }
  kolibri_rig::build(rig, snapdir, o);
  rk::KolibriEngine& e = *rig.eng;
  std::printf("kolibri_snapshot_gpu_test: %u ids, %u layers, %u device(s)%s\n", kN, e.model().desc.layers, e.devices(),
              e.devices() == 2 ? (std::string(", split ") + std::to_string(e.split())).c_str() : "");
  const int rc = run_default(e, ids);
  std::printf("kolibri_snapshot_gpu_test %s\n", rc == 0 ? "OK" : "FAILED");
  return rc;
}
