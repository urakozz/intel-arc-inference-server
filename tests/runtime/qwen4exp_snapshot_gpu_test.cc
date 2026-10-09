// Spec 21e Task 2 (plan 21e Review Focus 1 and 5, spec 21 F4's restores): Qwen4ExpEngine's prefix-cache snapshot
// calls on the card with a checkpoint - box only (labels checkpoint;qwen4exp). The synthetic checkpoints (21b's
// make_synth.py: 4 real-width layers - three GDN, the PLE layer among them, one QSA) carry it; Intel's truncated
// model at --layers 18 when present. The host side (the layouts, round trips across placements) is
// qwen4exp_snapshot_test's.
//
//   qwen4exp_snapshot_gpu_test <checkpoint> <ids> [pp2[:split]] [cross] [layers:N] [int8]
//
// The prompt: <ids> cycled to kN = 5004 positions (3 chunks: the QSA selection is active past 2050); then 32 greedy
// ids. The block hook is set in every run, so every run's chunks end at the same block ends.
//   default (one card, or two with pp2):
//     0. state_bytes() / kv_bytes(n) are runtime::qwen4exp::state_snapshot_bytes / kv_snapshot_bytes;
//     a. restores at the BLOCK ENDS 2048 and 4096 (the hook's own saves during the cold prefill): reset, load_kv(0,
//        at) + load_state(at), prefill the tail, 32 greedy ids - the ids, the last logits, routes, selections, every
//        layer's persistent bytes (KV, compressed keys, tail rings, GDN states and conv rings, the PLE rings) and the
//        PLE ids BITWISE the cold run's (qwen4exp_rig::read_state);
//     b. restores at the REQUEST ENDS 2049, 2050, 2051 (the QSA cut), 2052 and 5000: a session that prefilled [0, at),
//        saved there and prefilled the rest, against a reset + restore + the same tail prefill - bitwise (against ONE
//        cold prefill of the whole prompt a split off the 64-row grid differs where 21d's GDN WY sub-chunks move, plan
//        21d departure 5);
//     c. the history before 0: a restore at 1 (one id prefilled) continues bitwise (the PLE ids before 0 are EOS on
//        the host, the conv rows zeros).
//   cross (two GPUs; spec 16b's rule): a snapshot taken under --pp 2 (pp2:<s>, default 2) restores under --pp 1 and
//     continues bitwise as the two-card session does, and the reverse - one host layout.
// Exit 77 (SKIP) when the checkpoint is absent or two GPUs are needed.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "check.h"
#include "qwen4exp_rig.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace {

namespace rq = runtime::qwen4exp;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 5004, kGen = 32;

struct Saved {
  std::vector<uint8_t> state, kv;   // the state at `at` and the blocks [0, at)
};

Saved save(rq::Qwen4ExpEngine& e) {
  Saved s;
  const uint32_t at = e.pos();
  s.state.resize(e.state_bytes());
  s.kv.resize(e.kv_bytes(at));
  e.save_state(s.state.data());
  e.save_kv(0, at, s.kv.data());
  return s;
}

void restore(rq::Qwen4ExpEngine& e, const Saved& s, uint32_t at) {
  e.load_kv(0, at, s.kv.data());
  e.load_state(s.state.data(), at);
  CHECK_EQ(e.pos(), at);
}

struct Out {
  Ids ids;
  qwen4exp_rig::State state;
};

Out finish(rq::Qwen4ExpEngine& e) {
  Out o;
  o.ids = e.generate(kGen);
  o.state = qwen4exp_rig::read_state(e);
  return o;
}

Ids slice(const Ids& v, size_t a, size_t b) { return Ids(v.begin() + a, v.begin() + b); }

bool report(const std::string& what, const Out& got, const Out& want) {
  const std::string diff = qwen4exp_rig::first_difference(got.state, want.state);
  const bool ok = got.ids == want.ids && diff.empty();
  std::printf("  %s: ids %s, state %s\n", what.c_str(), got.ids == want.ids ? "bitwise" : "DIFFER",
              diff.empty() ? "bitwise" : ("DIFFERS at " + diff).c_str());
  return ok;
}

int run_default(rq::Qwen4ExpEngine& e, const Ids& prompt) {
  const model::Qwen4ExpDesc& d = e.model().desc;
  bool ok = true;
  CHECK_EQ(e.state_bytes(), rq::state_snapshot_bytes(d, false));
  CHECK_EQ(e.kv_bytes(4096), rq::kv_snapshot_bytes(d, 0, 4096, false));
  std::printf("0. state %zu B, blocks %zu B a position (%u layers)\n", e.state_bytes(), e.kv_bytes(4) / 4, d.layers);

  // a. the block ends: the hook saves at 2048 and 4096 during the cold prefill.
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
    ok = report("a. restore at the block end " + std::to_string(at), finish(e), cold) && ok;
  }

  // b / c. request ends: the uncached session (prefill to `at`, save, prefill the rest) against a restore.
  for (uint32_t at : {2049u, 2050u, 2051u, 2052u, 5000u, 1u}) {
    e.reset();
    e.prefill(slice(prompt, 0, at));
    const Saved s = save(e);
    e.prefill(slice(prompt, at, kN));
    const Out ref = finish(e);
    e.reset();
    restore(e, s, at);
    e.prefill(slice(prompt, at, kN));
    ok = report(std::string(at == 1 ? "c" : "b") + ". restore at the request end " + std::to_string(at), finish(e), ref) &&
         ok;
  }
  e.set_block_hook({});
  return ok ? 0 : 1;
}

int run_cross(qwen4exp_rig::Rig& rig, const std::string& snapdir, qwen4exp_rig::Options o, const Ids& prompt) {
  bool ok = true;
  for (uint32_t at : {2051u, 4096u}) {
    for (int dir = 0; dir < 2; ++dir) {
      qwen4exp_rig::Options from = o, to = o;
      from.devices = dir == 0 ? 2 : 1;
      to.devices = dir == 0 ? 1 : 2;
      if (from.devices == 2 && !from.split) from.split = 2;
      if (to.devices == 2 && !to.split) to.split = 2;
      qwen4exp_rig::build(rig, snapdir, from);
      rq::Qwen4ExpEngine& a = *rig.eng;
      a.set_block_hook([](uint32_t, bool) {});   // the same chunk ends in every run
      a.reset();
      a.prefill(slice(prompt, 0, at));
      const Saved s = save(a);
      const size_t state_bytes = a.state_bytes(), kv_bytes = a.kv_bytes(at);
      a.prefill(slice(prompt, at, kN));
      const Out ref = finish(a);
      qwen4exp_rig::build(rig, snapdir, to);
      rq::Qwen4ExpEngine& b = *rig.eng;
      b.set_block_hook([](uint32_t, bool) {});
      CHECK_EQ(b.state_bytes(), state_bytes);
      CHECK_EQ(b.kv_bytes(at), kv_bytes);
      b.reset();
      restore(b, s, at);
      b.prefill(slice(prompt, at, kN));
      ok = report(std::string(dir == 0 ? "--pp 2 snapshot at " : "--pp 1 snapshot at ") + std::to_string(at) +
                      (dir == 0 ? " restored under --pp 1" : " restored under --pp 2"),
                  finish(b), ref) && ok;
    }
  }
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string why;
  const std::string snapdir = qwen4exp_rig::qwen4exp_snapshot(argc > 1 ? argv[1] : "", &why);
  if (snapdir.empty()) {
    std::printf("SKIP: no Qwen3.8-Flash-Next checkpoint at '%s' (%s)\n", argc > 1 ? argv[1] : "", why.c_str());
    return 77;
  }
  qwen4exp_rig::Options o;
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
    } else if (x.rfind("layers:", 0) == 0) {
      o.layers = uint32_t(std::strtoul(x.c_str() + 7, nullptr, 10));
    } else if (x == "int8") {
      o.int8_head = true;
    } else {
      std::fprintf(stderr, "qwen4exp_snapshot_gpu_test: unknown flag %s (pp2[:split], cross, layers:N, int8)\n", x.c_str());
      return 2;
    }
  }
  if ((cross || o.devices == 2) && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const model::Qwen4ExpDesc pre = loader::qwen4exp_checkpoint_desc(snapdir, o.layers);
  const Ids base = qwen4exp_rig::read_ids(argc > 2 ? argv[2] : "", pre.vocab);
  CHECK(!base.empty());
  Ids ids(kN);
  for (uint32_t i = 0; i < kN; ++i) ids[i] = base[i % base.size()];
  qwen4exp_rig::Rig rig;
  if (cross) {
    const int rc = run_cross(rig, snapdir, o, ids);
    std::printf("qwen4exp_snapshot_gpu_test cross %s\n", rc == 0 ? "OK" : "FAILED");
    return rc;
  }
  qwen4exp_rig::build(rig, snapdir, o);
  rq::Qwen4ExpEngine& e = *rig.eng;
  std::printf("qwen4exp_snapshot_gpu_test: %u ids, %u layers, %u device(s)%s\n", kN, e.model().desc.layers, e.devices(),
              e.devices() == 2 ? (std::string(", split ") + std::to_string(e.split())).c_str() : "");
  const int rc = run_default(e, ids);
  std::printf("qwen4exp_snapshot_gpu_test %s\n", rc == 0 ? "OK" : "FAILED");
  return rc;
}
