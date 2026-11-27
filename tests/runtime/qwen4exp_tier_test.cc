// Plan 21-QFN-v2a Task 8: the expert tier on the card (GPU only; labels checkpoint;qwen4exp, + pp for two cards), on the
// synthetic real-width checkpoint (ours: 4 layers, 512 experts, the MTP head).
//
//   qwen4exp_tier_test <checkpoint> <ids> 1|pp2   O0 / O1 - residency never changes arithmetic. Placements: the
//       direct binaries (B70_Q4_EXPERTS=direct), then the table with every expert in VRAM ("all"), none (every expert
//       read from pinned host memory), random:0.5:7 and an adversarial one (each layer's most-routed half of the
//       experts - by the direct run's prefill routes - in host memory, the rest in VRAM, so an arena slot is never
//       the expert id). Each placement runs plain and with the MTP head, each compared bitwise with the direct run of
//       the same kind, the first difference named:
//         plain (no head): a 2200-id prompt by prefill in chunks (every chunk's routes and selections), 8 decode
//           steps (logits, routes, selections and diagnostics, every layer's state, the PLE ids after each),
//           8 generated ids;
//         with the MTP head: the same prefill, then verify at M = 2, 3, 4 from one position (every row's logits,
//           routes, selections and id, every layer's state), commit(1), 4 generated ids and the state.
//       pp2: --parallel pp, split 2 - device 1's layers [2, 4) read their host experts through device 1's table (the
//       regression test of the table launches' host indirect-access flag). These runs set B70_Q4_EXPERT_CHECK=off:
//       the load check's own flagged kernel could make device 1's mirrors resident before the lists run and hide a
//       launch without the flag (the check mode covers the check).
//   qwen4exp_tier_test <checkpoint> <ids> check    O3 - the load check (q4_expert_check at engine construction)
//       refuses a wrong upload: under random:0.5:7, after the load, one word of a host mirror, one word of a VRAM
//       arena, or two table entries swapped - the construction throws naming that layer and expert; the untouched
//       load constructs and reports its words checked.
//       Plan 21-QFN-v3a Task 0: after each plain decode step the engine's miss count (read_step_misses: the route rows
//       against the device's expert table) equals the count from the placement's own table (ExpertTier::resident).
//   ... 1|pp2 cpu   PERF-124: the same with every host-memory expert computed on the CPU (CpuExperts::Mode::All: the
//       decode and verify lists' hand-off at M = 1..4), still bitwise the direct run.
//   qwen4exp_tier_test <checkpoint> <ids> tp2 [cpu]   --parallel tp (Qwen4ExpTpEngine; each card's half experts in its
//       own tier, the same forced placement on both cards), plain and with the MTP head (whole on card 1). The prompt
//       is fed by decode (tp's prefill is held to a cosine bar, not bitwise), its first kTpPrompt ids, then as above: 8
//       forced decode steps (each step's logits, both cards' H and route rows), 8 generated ids; with the head, verify
//       at M = 2, 3, 4 (every row's logits and id), commit(1), 4 generated ids. Each placement (all, none,
//       random:0.5:7) bitwise tp's "all" run of the same kind, the first difference named; and every run's ids the pp2
//       direct run's (the same flow on Qwen4ExpEngine, split 2) - tp's rule against pp. With `cpu` every host-memory
//       half expert is computed on the CPU (each card's partial, the all-reduce unchanged).
// Exit 77 (SKIP) without the checkpoint or, for pp2 / tp2, with one GPU.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "check.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp/qwen4exp_tp_engine.h"
#include "runtime/qwen4exp_rig.h"

namespace {
namespace rq = runtime::qwen4exp;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kPrompt = 2200, kMaxLen = 8192, kCheckLayer = 2, kTpPrompt = 120;
rq::CpuExperts::Mode g_cpu = rq::CpuExperts::Mode::Off;   // argv[4] "cpu": Mode::All
int g_miss_bad = 0;                                        // plain steps whose miss count is not the placement's
uint32_t g_misses = 0;                                     // the misses counted over a run's plain steps

// Named byte strings in run order; two runs compare part by part.
struct Rec {
  std::vector<std::pair<std::string, std::vector<uint8_t>>> parts;
  template <class T>
  void add(const std::string& name, const std::vector<T>& v) {
    std::vector<uint8_t> b(v.size() * sizeof(T));
    if (!b.empty()) std::memcpy(b.data(), v.data(), b.size());
    parts.emplace_back(name, std::move(b));
  }
};
std::string first_difference(const Rec& a, const Rec& b) {
  for (size_t i = 0; i < a.parts.size() && i < b.parts.size(); ++i)
    if (a.parts[i] != b.parts[i]) return a.parts[i].first;
  return a.parts.size() == b.parts.size() ? "" : "the number of parts";
}

// `step`: the last replay was a decode step, so its route rows, selections, diagnostics and PLE ids are written (a prefill
// writes its own buffers - the chunk's, recorded per chunk - and leaves decode's as allocated).
void add_state(Rec& r, rq::Qwen4ExpEngine& e, const std::string& at, bool step = true) {
  const qwen4exp_rig::State s = qwen4exp_rig::read_state(e);
  r.add(at + ": logits", s.logits);
  if (step) {
    r.add(at + ": route rows", s.routes);
    for (size_t i = 0; i < s.selections.size(); ++i) r.add(at + ": QSA selection #" + std::to_string(i), s.selections[i]);
    r.add(at + ": selection diagnostics", s.diag);
    r.add(at + ": PLE ids", s.ple_ids);
  }
  for (size_t l = 0; l < s.layers.size(); ++l) r.add(at + ": layer " + std::to_string(l) + "'s state", s.layers[l]);
}

// The run (the header); `counts` (the direct run's): every layer's prefill routes counted per expert.
Rec run(rq::Qwen4ExpEngine& e, const Ids& prompt, const Ids& force, std::vector<uint32_t>* counts) {
  const model::Qwen4ExpDesc& d = e.model().desc;
  Rec r;
  e.reset();
  for (uint32_t p = 0; p < prompt.size(); p += rq::kPfC) {
    const uint32_t c = std::min<uint32_t>(rq::kPfC, uint32_t(prompt.size()) - p);
    e.prefill(Ids(prompt.begin() + p, prompt.begin() + p + c));
    const std::vector<uint32_t> rt = e.read_prefill_routes();
    const std::string at = "prefill chunk at " + std::to_string(p);
    r.add(at + ": route rows", rt);
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (d.is_qsa(l)) r.add(at + ": layer " + std::to_string(l) + "'s selections", e.read_prefill_selection(l));
      if (counts)
        for (uint32_t row = 0; row < c; ++row)
          for (uint32_t k = 0; k < d.top_k; ++k)
            ++(*counts)[size_t(l) * d.experts + rt[rq::pf_route_at(l) / 4 + size_t(row) * rq::kRouteWords + k]];
    }
  }
  add_state(r, e, "after the prefill", false);
  if (!e.mtp()) {
    for (uint32_t j = 0; j < force.size(); ++j) {
      e.ingest({force[j]});
      add_state(r, e, "decode step " + std::to_string(j));
      const std::vector<uint32_t> rt = e.read_routes();
      uint32_t want = 0;
      for (uint32_t l = 0; l < d.layers; ++l)
        for (uint32_t k = 0; k < d.top_k; ++k)
          want += !e.model().tier.resident(l, rt[size_t(l) * rq::kRouteWords + k]);
      const uint32_t got = e.read_step_misses();
      g_misses += got;
      if (got != want) {
        std::printf("FAIL decode step %u: %u misses read through the table, the placement's %u\n", j, got, want);
        ++g_miss_bad;
      }
    }
    r.add("8 generated ids", e.generate(8));
    return r;
  }
  const uint32_t x = e.pending();
  for (uint32_t k = 1; k <= rq::Qwen4ExpEngine::kMaxDraft; ++k) {
    const std::string at = "verify at M = " + std::to_string(k + 1);
    e.set_token(x);
    for (uint32_t i = 0; i < k; ++i) e.set_draft_input(i, force[i]);
    e.verify(k);
    std::vector<float> rows(size_t(k + 1) * d.vocab);
    e.read_logits_into(rows.data(), k + 1);
    r.add(at + ": logits", rows);
    r.add(at + ": route rows", e.read_verify_routes());
    r.add(at + ": ids", Ids(e.verify_ids(), e.verify_ids() + k + 1));
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (d.is_qsa(l))
        for (uint32_t row = 0; row <= k; ++row)
          r.add(at + ": layer " + std::to_string(l) + " row " + std::to_string(row) + "'s selection",
                e.read_verify_selection(l, row));
      r.add(at + ": layer " + std::to_string(l) + "'s state", e.read_layer_state(l));
    }
  }
  e.commit(1, e.verify_ids()[1]);
  r.add("4 generated ids after commit(1)", e.generate(4));
  add_state(r, e, "after commit(1)");
  return r;
}

// rig.eng over a load at `tier` (null: the loader's own - every expert resident, or B70_Q4_EXPERTS=direct's form).
void build(qwen4exp_rig::Rig& rig, const std::string& snap, const qwen4exp_rig::Options& o, const rq::ExpertTier* tier) {
  rig.eng.reset();
  qwen4exp_rig::open(rig, o.devices);
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, 0);
  loader::Q4LoadedModel m = loader::load_qwen4exp(rig.ctx, snap, o.max_len, qwen4exp_rig::placement(d, o),
                                                  loader::LmHeadForm::Int8, 0, o.mtp, tier);
  runtime::PipelineOptions po;
  po.timeout_ms = o.timeout_ms;
  po.prefill_timeout_ms = o.prefill_timeout_ms;
  rig.eng = std::make_unique<rq::Qwen4ExpEngine>(rig.ctx, std::move(m), o.max_len, false, po,
                                                 runtime::CaptureOptions{}, g_cpu);
}

rq::ExpertTier forced(const model::Qwen4ExpDesc& d, const char* spec) {
  return rq::fill_tier(d, rq::ExpertBudget{}, nullptr, spec);
}

int residency(const std::string& snap, const Ids& prompt, const Ids& force, uint32_t devices) {
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::Options o;
  o.max_len = kMaxLen;
  o.int8_head = true;
  o.devices = devices;
  o.split = devices == 2 ? 2 : 0;
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, 0);
  std::vector<uint32_t> counts(size_t(d.layers) * d.experts, 0);
  ::setenv("B70_Q4_EXPERT_CHECK", "off", 1);   // the header: nothing flagged runs before the lists
  Rec ref[2];
  for (const bool mtp : {false, true}) {
    o.mtp = mtp;
    ::setenv("B70_Q4_EXPERTS", "direct", 1);
    build(rig, snap, o, nullptr);
    ::unsetenv("B70_Q4_EXPERTS");
    ref[mtp] = run(*rig.eng, prompt, force, mtp ? nullptr : &counts);
    std::printf("O0 %s, %u card(s): the direct form, %zu parts recorded\n", mtp ? "with the MTP head" : "plain", devices,
                ref[mtp].parts.size());
  }
  // adversarial: each layer's most-routed half in host memory (ties to the lower id, as fill_tier's ranking)
  rq::ExpertTier adv = rq::all_resident(d);
  adv.source = "adversarial (the direct run's most-routed half of each layer in host memory)";
  for (uint32_t l = 0; l < d.layers; ++l) {
    std::vector<uint32_t> o2(d.experts);
    for (uint32_t e = 0; e < d.experts; ++e) o2[e] = e;
    const uint32_t* c = counts.data() + size_t(l) * d.experts;
    std::stable_sort(o2.begin(), o2.end(), [&](uint32_t a, uint32_t b) { return c[a] > c[b]; });
    for (uint32_t i = 0; i < d.experts / 2; ++i) adv.vram[size_t(l) * d.experts + o2[i]] = 0;
  }
  const std::vector<std::pair<std::string, rq::ExpertTier>> placements = {
      {"all", forced(d, "all")}, {"none", forced(d, "none")}, {"random:0.5:7", forced(d, "random:0.5:7")},
      {"adversarial", adv}};
  int bad = 0;
  for (const auto& [name, tier] : placements)
    for (const bool mtp : {false, true}) {
      o.mtp = mtp;
      build(rig, snap, o, &tier);
      g_misses = 0;
      const Rec got = run(*rig.eng, prompt, force, nullptr);
      const std::string diff = first_difference(got, ref[mtp]);
      if (!mtp) std::printf("  %u misses over %zu plain steps, each the placement's count\n", g_misses, force.size());
      std::printf("O1 %s %s, %u card(s): %zu experts in host memory (%zu B), %zu words checked at load (B70_Q4_EXPERT_CHECK=off) - %s%s\n",
                  name.c_str(), mtp ? "with the MTP head" : "plain", devices,
                  size_t(d.layers) * d.experts - std::count(tier.vram.begin(), tier.vram.end(), uint8_t(1)),
                  rig.eng->model().mirror_bytes, rig.eng->expert_words_checked(),
                  diff.empty() ? "BITWISE the direct form's" : "DIFFERS first at ", diff.c_str());
      bad += !diff.empty();
    }
  return bad + g_miss_bad;
}

// tp2's flow on either engine (the header): the prompt fed by decode, the steps' records into `r` (tp: also both
// cards' H and route rows) and every id it produced into `ids` (the pp comparison).
template <class E>
void run_fed(E& e, const Ids& prompt, const Ids& force, Rec& r, Ids& ids) {
  const uint32_t V = e.model().desc.vocab;
  e.reset();
  e.ingest(prompt);
  const auto logits = [&](const std::string& at, uint32_t rows) {
    std::vector<float> l(size_t(rows) * V);
    e.read_logits_into(l.data(), rows);
    r.add(at + ": logits", l);
    if constexpr (std::is_same_v<E, rq::Qwen4ExpTpEngine>)
      for (uint32_t c = 0; c < 2; ++c) {
        r.add(at + ": card " + std::to_string(c) + "'s H", e.read_H(c));
        r.add(at + ": card " + std::to_string(c) + "'s route rows", e.read_routes(c));
      }
  };
  const auto gen = [&](const std::string& at, uint32_t n) {
    const Ids g = e.generate(n);
    r.add(at, g);
    ids.insert(ids.end(), g.begin(), g.end());
  };
  if (!e.mtp()) {
    for (uint32_t j = 0; j < force.size(); ++j) {
      e.ingest({force[j]});
      logits("decode step " + std::to_string(j), 1);
      ids.push_back(e.pending());
    }
    gen("8 generated ids", 8);
    return;
  }
  const uint32_t x = e.pending();
  for (uint32_t k = 1; k <= rq::kMaxDraft; ++k) {
    const std::string at = "verify at M = " + std::to_string(k + 1);
    e.set_token(x);
    for (uint32_t i = 0; i < k; ++i) e.set_draft_input(i, force[i]);
    e.verify(k);
    logits(at, k + 1);
    const Ids v(e.verify_ids(), e.verify_ids() + k + 1);
    r.add(at + ": ids", v);
    ids.insert(ids.end(), v.begin(), v.end());
  }
  e.commit(1, e.verify_ids()[1]);
  gen("4 generated ids after commit(1)", 4);
}

int tensor_parallel(const std::string& snap, const Ids& prompt, const Ids& force) {
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::Options o;
  o.max_len = kMaxLen;
  o.int8_head = true;
  o.devices = 2;
  o.split = 2;
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, 0);
  ::setenv("B70_Q4_EXPERT_CHECK", "off", 1);   // as residency(): nothing flagged runs before the lists
  const Ids fed(prompt.begin(), prompt.begin() + kTpPrompt);
  Ids pp_ids[2];
  for (const bool mtp : {false, true}) {
    o.mtp = mtp;
    ::setenv("B70_Q4_EXPERTS", "direct", 1);
    build(rig, snap, o, nullptr);
    ::unsetenv("B70_Q4_EXPERTS");
    Rec unused;
    run_fed(*rig.eng, fed, force, unused, pp_ids[mtp]);
    std::printf("tp2: pp2's direct form %s, %zu ids recorded\n", mtp ? "with the MTP head" : "plain", pp_ids[mtp].size());
  }
  rig.eng.reset();
  const std::vector<std::pair<std::string, rq::ExpertTier>> placements = {
      {"all", forced(d, "all")}, {"none", forced(d, "none")}, {"random:0.5:7", forced(d, "random:0.5:7")}};
  Rec ref[2];
  int bad = 0;
  for (const auto& [name, tier] : placements)
    for (const bool mtp : {false, true}) {
      std::optional<rq::Qwen4ExpTpEngine> e;
      e.emplace(*rig.c0, *rig.c1,
                loader::load_qwen4exp(rig.ctx, snap, kMaxLen, model::Q4Placement::tp2(d, model::Q4TpEnds::Split),
                                      loader::LmHeadForm::Int8, 0, mtp, &tier, &tier),
                kMaxLen, rq::Q4TpOptions{}, g_cpu);
      Rec got;
      Ids ids;
      run_fed(*e, fed, force, got, ids);
      const char* kind = mtp ? "with the MTP head" : "plain";
      const size_t host = size_t(d.layers) * d.experts - std::count(tier.vram.begin(), tier.vram.end(), uint8_t(1));
      const bool same_ids = ids == pp_ids[mtp];
      bad += !same_ids;
      if (name == "all") {
        ref[mtp] = std::move(got);
        std::printf("tp2 %s %s: the reference; ids %s\n", name.c_str(), kind,
                    same_ids ? "the pp2 direct run's" : "DIFFER from the pp2 direct run's");
        continue;
      }
      const std::string diff = first_difference(got, ref[mtp]);
      std::printf("tp2 %s %s: %zu experts' halves a card in host memory, the CPU %s - %s%s; ids %s\n", name.c_str(),
                  kind, host, e->cpu_experts() ? "computes them" : "off (read over PCIe)",
                  diff.empty() ? "BITWISE tp's all-resident run" : "DIFFERS first at ", diff.c_str(),
                  same_ids ? "the pp2 direct run's" : "DIFFER from the pp2 direct run's");
      bad += !diff.empty();
      bad += g_cpu == rq::CpuExperts::Mode::All && host > 0 && !e->cpu_experts();   // asked for and none built
    }
  return bad;
}

int check(const std::string& snap) {
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::open(rig, 1);
  ::unsetenv("B70_Q4_EXPERT_CHECK");
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, 0);
  const rq::ExpertTier tier = forced(d, "random:0.5:7");
  const uint32_t L = kCheckLayer;
  uint32_t host = d.experts, vram[2] = {d.experts, d.experts};
  for (uint32_t e = 0, nv = 0; e < d.experts; ++e)
    if (!tier.resident(L, e)) host = std::min(host, e);
    else if (nv < 2) vram[nv++] = e;
  CHECK(host < d.experts && vram[1] < d.experts);
  int bad = 0;
  for (const std::string what : {"nothing", "a host mirror word", "a VRAM arena word", "two table entries"}) {
    loader::Q4LoadedModel m = loader::load_qwen4exp(rig.ctx, snap, kMaxLen, model::Q4Placement::one(d),
                                                    loader::LmHeadForm::Int8, 0, false, &tier);
    loader::Q4DevicePart& P = m.parts[0];
    const loader::Q4Layer& Lr = P.layers[L - P.first];
    l0::CmdList imm = l0::CmdList::immediate(*rig.c0);
    uint32_t named = host;
    if (what == "a host mirror word") {
      Lr.mirror->as<uint64_t>()[0] ^= 1;   // mirror slot 0: the layer's first expert in host memory, word 0
    } else if (what == "a VRAM arena word") {
      uint64_t w = 0;   // arena slot 0: the layer's first resident expert, its gate||up block's word 0
      imm.copy(&w, Lr.gate_up->ptr(), 8);
      w ^= 1;
      imm.copy(Lr.gate_up->ptr(), &w, 8);
      named = vram[0];
    } else if (what == "two table entries") {
      uint64_t* row = reinterpret_cast<uint64_t*>(P.expert_table->as<uint8_t>() + size_t(L - P.first) * 2 * d.experts * 8);
      uint64_t a = 0, b = 0;
      imm.copy(&a, row + vram[0], 8);
      imm.copy(&b, row + vram[1], 8);
      imm.copy(row + vram[0], &b, 8);
      imm.copy(row + vram[1], &a, 8);
      named = vram[0];
    }
    try {
      rq::Qwen4ExpEngine e(rig.ctx, std::move(m), kMaxLen);
      const bool ok = what == "nothing" && e.expert_words_checked() != 0;
      std::printf("O3 %s corrupted: the engine constructed (%zu words checked, %.1f ms with the sampling) - %s\n",
                  what.c_str(), e.expert_words_checked(), e.model().expert_check_ms, ok ? "as it should" : "FAIL");
      bad += !ok;
    } catch (const std::runtime_error& x) {
      const std::string msg = x.what();
      const std::string want = "layer " + std::to_string(L) + "'s expert " + std::to_string(named) + " ";
      const bool ok = what != "nothing" && msg.find(want) != std::string::npos;
      std::printf("O3 %s corrupted: refused - %s\n  %s\n", what.c_str(), ok ? "naming it" : "FAIL", msg.c_str());
      bad += !ok;
    }
  }
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <checkpoint> <ids> 1|pp2|tp2|check [cpu]\n", argv[0]);
    return 2;
  }
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  const std::string mode = argv[3];
  if (argc > 4 && std::string(argv[4]) == "cpu") g_cpu = rq::CpuExperts::Mode::All;
  if ((mode == "pp2" || mode == "tp2") && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two GPUs needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, 0);
  CHECK(d.layers == 4 && d.layers > kCheckLayer);
  const Ids src = qwen4exp_rig::read_ids(argv[2], d.vocab);
  CHECK(src.size() >= 8);
  Ids prompt(kPrompt);
  for (uint32_t p = 0; p < kPrompt; ++p) prompt[p] = src[p % src.size()];
  const Ids force(src.begin(), src.begin() + 8);
  const int bad = mode == "check" ? check(snap)
                  : mode == "tp2" ? tensor_parallel(snap, prompt, force)
                                  : residency(snap, prompt, force, mode == "pp2" ? 2 : 1);
  if (bad) {
    std::fprintf(stderr, "qwen4exp_tier_test %s: %d FAIL(s)\n", mode.c_str(), bad);
    return 1;
  }
  std::printf("qwen4exp_tier_test %s OK\n", mode.c_str());
  return 0;
}
