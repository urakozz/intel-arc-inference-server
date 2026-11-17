// Plan 21-QFN-v2a Task 8: the expert tier on the card (box only; labels checkpoint;qwen4exp, + pp for two cards), on the
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
// Exit 77 (SKIP) without the checkpoint or, for pp2, with one GPU.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp_rig.h"

namespace {
namespace rq = runtime::qwen4exp;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kPrompt = 2200, kMaxLen = 8192, kCheckLayer = 2;

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
  rig.eng = std::make_unique<rq::Qwen4ExpEngine>(rig.ctx, std::move(m), o.max_len, false, po);
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
      const Rec got = run(*rig.eng, prompt, force, nullptr);
      const std::string diff = first_difference(got, ref[mtp]);
      std::printf("O1 %s %s, %u card(s): %zu experts in host memory (%zu B), %zu words checked at load (B70_Q4_EXPERT_CHECK=off) - %s%s\n",
                  name.c_str(), mtp ? "with the MTP head" : "plain", devices,
                  size_t(d.layers) * d.experts - std::count(tier.vram.begin(), tier.vram.end(), uint8_t(1)),
                  rig.eng->model().mirror_bytes, rig.eng->expert_words_checked(),
                  diff.empty() ? "BITWISE the direct form's" : "DIFFERS first at ", diff.c_str());
      bad += !diff.empty();
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
    std::fprintf(stderr, "usage: %s <checkpoint> <ids> 1|pp2|check\n", argv[0]);
    return 2;
  }
  std::string why;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap.empty()) {
    std::printf("SKIP: %s\n", why.c_str());
    return 77;
  }
  const std::string mode = argv[3];
  if (mode == "pp2" && l0::Context::gpu_count() < 2) {
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
  const int bad = mode == "check" ? check(snap) : residency(snap, prompt, force, mode == "pp2" ? 2 : 1);
  if (bad) {
    std::fprintf(stderr, "qwen4exp_tier_test %s: %d FAIL(s)\n", mode.c_str(), bad);
    return 1;
  }
  std::printf("qwen4exp_tier_test %s OK\n", mode.c_str());
  return 0;
}
