// qwen4exp_decode_test - spec 21c Task 3, F4 on decode: Qwen3.8-Flash-Next's engine (runtime::qwen4exp) on a
// checkpoint (the synthetic real-width ones: tools/box_validate/qwen4exp_oracle.sh synth-ckpt; Intel's at
// --layers N), no golden set needed.
//
//   1. plan == allocation: every device's memory_use() components equal runtime::qwen4exp::plan's (model, KV +
//      indexer keys + tails, decode state = control + GDN / PLE state + scratch + link), and the list's launches
//      equal decode_launches (one card: 75 at --layers 4, 779 at 48)
//   2. replay determinism: two reset + ingest(prompt) + generate(32) runs bitwise - the logits, every layer's route
//      row, every QSA layer's selection and its 512th / 513th diagnostic, every layer's persistent bytes (KV, the
//      compressed indexer keys, the 8-slot tail rings, the GDN state and conv rings, the PLE id and conv rings) and
//      the PLE ids, compared after the prompt and after the generation
//   3. the injected capture (spec 21 F3's debug input): its launch count printed (2 fewer a QSA layer); fed the
//      free run's own selections step by step, it reproduces the free run bitwise (logits and every state) - the
//      injection path changes where the lists come from, nothing else
//
// argv: <checkpoint> <ids file> [prompt length] [pp2 | pp2:<split>] [copy | peer] [layers:<N>] [int8]
// The prompt is the file's first <prompt length> ids (default all of them; > 2051 exercises the selection). Exit
// 77 (SKIP) when the checkpoint, its PLE file or the ids are absent.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp_rig.h"

namespace {
namespace rq = runtime::qwen4exp;
constexpr uint32_t kGen = 32;
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <checkpoint> <ids> [prompt length] [pp2[:split]] [copy|peer] [layers:N] [int8]\n", argv[0]);
    return 2;
  }
  qwen4exp_rig::Options o;
  o.max_len = 16384;
  uint32_t plen = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "pp2") o.devices = 2;
    else if (a.rfind("pp2:", 0) == 0) { o.devices = 2; o.split = uint32_t(std::strtoul(a.c_str() + 4, nullptr, 10)); }
    else if (a == "copy" || a == "peer") o.handoff = a == "peer" ? runtime::PpHandoff::Peer : runtime::PpHandoff::Copy;
    else if (a.rfind("layers:", 0) == 0) o.layers = uint32_t(std::strtoul(a.c_str() + 7, nullptr, 10));
    else if (a == "int8") o.int8_head = true;
    else if (!a.empty() && a.find_first_not_of("0123456789") == std::string::npos) plen = uint32_t(std::stoul(a));
    else {
      std::fprintf(stderr, "qwen4exp_decode_test: unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap.empty()) {
    std::printf("SKIP: %s (the synthetic checkpoints: qwen4exp_oracle.sh synth-ckpt; Intel's: r31.ple_convert)\n", why.c_str());
    return 77;
  }
  std::vector<uint32_t> ids = qwen4exp_rig::read_ids(argv[2], 248320);
  if (ids.empty()) {
    std::printf("SKIP: no ids in %s\n", argv[2]);
    return 77;
  }
  if (plen && plen < ids.size()) ids.resize(plen);
  if (o.devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: pp2 needs two GPUs, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::build(rig, snap, o);
  rq::Qwen4ExpEngine& eng = *rig.eng;
  const model::Qwen4ExpDesc& d = eng.model().desc;
  const model::Q4Placement& p = eng.model().placement;

  // --- 1. plan == allocation; the launches -----------------------------------------------------------------------
  const std::vector<rq::DevicePlan> plan = rq::plan(d, p, eng.max_len(), o.int8_head, false);
  for (uint32_t dev = 0; dev < eng.devices(); ++dev) {
    const runtime::MemoryComponents m = eng.memory_use(dev);
    std::printf("device %u: model %zu / plan %zu, kv %zu / %zu, decode state %zu / %zu\n", dev, m.model, plan[dev].model,
                m.kv, plan[dev].kv, m.decode_state, plan[dev].decode_state);
    CHECK_EQ(m.model, plan[dev].model);
    CHECK_EQ(m.kv, plan[dev].kv);
    CHECK_EQ(m.decode_state, plan[dev].decode_state);
  }
  CHECK_EQ(eng.launches(), rq::decode_launches(d, p, eng.attention(), eng.handoff()));
  std::printf("%s: %u layers on %u device(s), %zu launches a token (%s attention), %zu with injected selections; the "
              "PLE table's %zu host pages read back by the device; prompt %zu ids\n", snap.c_str(), d.layers,
              eng.devices(), eng.launches(), rq::q4_attn_name(eng.attention()), eng.injected_launches(),
              eng.ple_pages_checked(), ids.size());

  // --- 2. two runs bitwise -----------------------------------------------------------------------------------------
  const auto run = [&](std::vector<qwen4exp_rig::State>* per_step, std::vector<std::vector<std::vector<uint32_t>>>* sels,
                       bool inject) {
    eng.reset();
    eng.set_injected_selection(inject);
    std::vector<qwen4exp_rig::State> marks;
    std::vector<uint32_t> toks;
    uint32_t step = 0;
    const uint32_t nq = d.qsa_before(d.layers);
    const auto feed_inj = [&]() {   // the free run's lists for this step
      if (!inject) return;
      for (uint32_t q = 0; q < nq; ++q) {
        const std::vector<uint32_t>& sel = (*sels)[step][q];
        uint32_t* row = eng.injected_list(q);
        const uint32_t c = sel.back();
        for (uint32_t i = 0; i < c; ++i) row[i] = sel[i];
        row[rq::kCountWord] = c;
      }
    };
    const auto after = [&]() {
      if (!inject && sels) {
        std::vector<std::vector<uint32_t>> s;
        for (uint32_t l = 0; l < d.layers; ++l)
          if (d.is_qsa(l)) s.push_back(eng.read_selection(l));
        sels->push_back(s);
      }
      ++step;
    };
    for (uint32_t id : ids) {
      feed_inj();
      eng.ingest({id});
      after();
    }
    marks.push_back(qwen4exp_rig::read_state(eng));
    for (uint32_t j = 0; j < kGen; ++j) {
      feed_inj();
      toks.push_back(eng.generate(1)[0]);
      after();
    }
    marks.push_back(qwen4exp_rig::read_state(eng));
    if (per_step) *per_step = marks;
    return toks;
  };
  std::vector<qwen4exp_rig::State> a, b, c;
  std::vector<std::vector<std::vector<uint32_t>>> sels;
  const std::vector<uint32_t> ta = run(&a, &sels, false);
  const std::vector<uint32_t> tb = run(&b, nullptr, false);
  CHECK(ta == tb);
  for (size_t i = 0; i < a.size(); ++i) {
    const std::string diff = qwen4exp_rig::first_difference(a[i], b[i]);
    if (!diff.empty()) std::fprintf(stderr, "replay %s: %s differs\n", i == 0 ? "after the prompt" : "after 32 tokens", diff.c_str());
    CHECK(diff.empty());
  }
  std::printf("F4: two runs bitwise (logits, routes, selections and diagnostics, KV, indexer keys, tail rings, GDN state, "
              "conv rings, PLE rings and ids) after the prompt and after %u tokens\n", kGen);
  // --- 3. the injected capture, fed the free run's own lists ------------------------------------------------------
  const std::vector<uint32_t> tc = run(&c, &sels, true);
  eng.set_injected_selection(false);
  CHECK(tc == ta);
  for (size_t i = 0; i < a.size(); ++i) {
    c[i].diag = a[i].diag;   // the injected list runs no q4_qsa_select: its diagnostics are not produced
    const std::string diff = qwen4exp_rig::first_difference(a[i], c[i]);
    if (!diff.empty()) std::fprintf(stderr, "injected run %s: %s differs\n", i == 0 ? "after the prompt" : "after 32 tokens", diff.c_str());
    CHECK(diff.empty());
  }
  std::printf("injected: %zu launches a token; fed the free run's selections, the run is bitwise the free run\n",
              eng.injected_launches());
  std::printf("first generated ids:");
  for (uint32_t j = 0; j < 8 && j < ta.size(); ++j) std::printf(" %u", ta[j]);
  std::printf("\nqwen4exp_decode_test OK\n");
  return 0;
}
