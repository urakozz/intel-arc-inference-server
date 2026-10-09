// Spec 21c Task 4: Qwen3.8-Flash-Next across two cards on spec 16b's pieces (box only; label checkpoint;qwen4exp;pp).
//   qwen4exp_pp_test <synthetic checkpoint> <Intel's checkpoint> <ids> [prompt length]
//
//   (a) the synthetic checkpoint (4 layers): --pp 1 against --pp 2 --pipeline-split 2, under each hand-off (copy,
//       peer): the prompt + 32 greedy tokens - the tokens and, after every phase, the logits, every route row,
//       every QSA layer's selection and diagnostic, every layer's persistent bytes (KV, the compressed keys and tail
//       rings, the GDN state and conv rings, the PLE rings) and the PLE ids BITWISE equal (plan 21c Review Focus 6:
//       the cut is invisible - the materialised H crosses); both Control blocks equal after every phase; and
//       --pipeline-split 1 (the cut AT the PLE layer: device 1's PLE prologue folds nothing) likewise
//   (b) Intel's checkpoint at --layers 38, split 19 against 20, each hand-off: the same, bitwise (SKIPs alone
//       without it)
//   (c) P4: drop_next_handoff() - the step throws within the bounds (never hangs), every later step throws until
//       reset(), and after reset() the run is again bitwise (a)'s
//   (d) the hand-off layout: pp_landing_layout(20480, 0) - the norm-sum region empty, the stamp right after H's
//       page, the flag on a page of its own
// The peer-access refusal (zeDeviceCanAccessPeer false -> refused naming it, Qwen4ExpEngine's constructor) cannot
// be provoked on a P2P-capable pair; it is read, not run. Exit 77 (SKIP) with one GPU or without the synthetic
// checkpoint.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "runtime/control.h"
#include "runtime/pipeline_plan.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp_rig.h"

namespace {
namespace rq = runtime::qwen4exp;
constexpr uint32_t kGen = 32;

void controls_equal(rq::Qwen4ExpEngine& e, const char* phase) {
  if (e.devices() < 2) return;
  const runtime::Control a = e.control(0), b = e.control(1);
  if (std::memcmp(&a, &b, sizeof a) != 0) {
    std::fprintf(stderr, "the two Control blocks differ after %s (pos %u / %u)\n", phase, a.pos, b.pos);
    std::exit(1);
  }
}

struct Run {
  std::vector<uint32_t> tokens;
  std::vector<qwen4exp_rig::State> marks;   // after the prompt, after the generation
};

Run run(rq::Qwen4ExpEngine& e, const std::vector<uint32_t>& ids) {
  Run r;
  e.reset();
  controls_equal(e, "reset");
  e.ingest(ids);
  controls_equal(e, "ingest");
  r.marks.push_back(qwen4exp_rig::read_state(e));
  r.tokens = e.generate(kGen);
  controls_equal(e, "generate");
  r.marks.push_back(qwen4exp_rig::read_state(e));
  return r;
}

void same(const Run& a, const Run& b, const std::string& what) {
  if (a.tokens != b.tokens) {
    std::fprintf(stderr, "%s: the tokens differ\n", what.c_str());
    std::exit(1);
  }
  for (size_t i = 0; i < a.marks.size(); ++i) {
    const std::string diff = qwen4exp_rig::first_difference(a.marks[i], b.marks[i]);
    if (!diff.empty()) {
      std::fprintf(stderr, "%s: %s differs %s\n", what.c_str(), diff.c_str(), i == 0 ? "after the prompt" : "after 32 tokens");
      std::exit(1);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  // (d) the layout, host arithmetic
  {
    const runtime::PpLandingLayout l = runtime::pp_landing_layout(20480, 0);
    CHECK_EQ(l.resid_bytes, size_t(20480));
    CHECK_EQ(l.sumsq_bytes, size_t(0));
    CHECK_EQ(l.sumsq_off % runtime::kPpPage, size_t(0));
    CHECK(l.sumsq_off >= l.resid_bytes);
    CHECK_EQ(l.stamp_off, l.sumsq_off);
    CHECK(l.flag_off % runtime::kPpPage == 0 && l.flag_off >= l.stamp_off + 4 && l.total == l.flag_off + runtime::kPpPage);
    const runtime::PpLandingLayout q = rq::landing_layout(model::qwen4exp());
    CHECK(q.resid_bytes == l.resid_bytes && q.sumsq_bytes == 0 && q.total == l.total);
    std::printf("(d) the landing layout: H %zu B, no norm sums, the stamp at %zu, the flag's page at %zu (%zu B)\n",
                l.resid_bytes, l.stamp_off, l.flag_off, l.total);
  }
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <synthetic checkpoint> <Intel's checkpoint> <ids> [prompt length]\n", argv[0]);
    return 2;
  }
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two GPUs needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  std::string why;
  const std::string synth = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (synth.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  std::vector<uint32_t> ids = qwen4exp_rig::read_ids(argv[3], 248320);
  if (argc > 4) ids.resize(std::min<size_t>(ids.size(), std::strtoul(argv[4], nullptr, 10)));
  CHECK(!ids.empty());
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::Options o;
  o.max_len = 16384;
  o.timeout_ms = 10000;   // a 4-layer step is milliseconds; (c)'s lost hand-off waits this long

  // (a) the synthetic checkpoint
  qwen4exp_rig::build(rig, synth, o);
  const Run one = run(*rig.eng, ids);
  std::printf("(a) --pp 1: %zu launches, tokens", rig.eng->launches());
  for (uint32_t j = 0; j < 8; ++j) std::printf(" %u", one.tokens[j]);
  std::printf(" ...\n");
  for (uint32_t split : {2u, 1u})
    for (runtime::PpHandoff h : {runtime::PpHandoff::Copy, runtime::PpHandoff::Peer}) {
      o.devices = 2;
      o.split = split;
      o.handoff = h;
      qwen4exp_rig::build(rig, synth, o);
      CHECK_EQ(rig.eng->launches(), rq::decode_launches(rig.eng->model().desc, rig.eng->model().placement,
                                                         rig.eng->attention(), h));
      const Run two = run(*rig.eng, ids);
      same(one, two, "--pp 2 split " + std::to_string(split) + " " + runtime::pp_handoff_name(h));
      std::printf("(a) --pp 2 --pipeline-split %u --pipeline-handoff %s: %zu launches, bitwise --pp 1\n", split,
                  runtime::pp_handoff_name(h), rig.eng->launches());
      if (split == 2 && h == runtime::PpHandoff::Copy) {
        // (c) P4: a lost hand-off throws within the bounds; every later step throws until reset()
        rq::Qwen4ExpEngine& e = *rig.eng;
        e.reset();
        e.ingest({ids[0]});
        e.drop_next_handoff();
        bool threw = false;
        try {
          e.ingest({ids[1]});
        } catch (const std::exception& ex) {
          threw = true;
          std::printf("(c) the lost hand-off: %s\n", ex.what());
        }
        CHECK(threw);
        threw = false;
        try {
          e.ingest({ids[1]});
        } catch (const std::exception&) {
          threw = true;
        }
        CHECK(threw);
        same(one, run(e, ids), "--pp 2 after a lost hand-off and reset()");
        std::printf("(c) later steps refused until reset(); after it the run is bitwise again\n");
      }
    }

  // (b) Intel's checkpoint at --layers 38, split 19 against 20
  const std::string intel = qwen4exp_rig::qwen4exp_snapshot(argv[2], &why);
  if (intel.empty()) {
    std::printf("(b) SKIP: %s\n", why.c_str());
  } else {
    o.layers = 38;
    o.int8_head = true;
    std::vector<Run> runs;
    for (uint32_t split : {19u, 20u})
      for (runtime::PpHandoff h : {runtime::PpHandoff::Copy, runtime::PpHandoff::Peer}) {
        o.split = split;
        o.handoff = h;
        qwen4exp_rig::build(rig, intel, o);
        runs.push_back(run(*rig.eng, ids));
        std::printf("(b) Intel's --layers 38 split %u %s: %zu launches\n", split, runtime::pp_handoff_name(h),
                    rig.eng->launches());
      }
    for (size_t i = 1; i < runs.size(); ++i) {
      // the persistent bytes live on different cards per split but read back in layer order: bitwise
      same(runs[0], runs[i], "Intel's --layers 38 run " + std::to_string(i));
    }
    std::printf("(b) splits 19 / 20 under copy and peer bitwise\n");
  }
  std::puts("qwen4exp_pp_test OK");
  return 0;
}
