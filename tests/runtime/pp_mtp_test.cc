// Spec 16d (plan 16d Review Focus 1-3): MTP across the split is BITWISE the single-card MTP
// on device 0, and spec 8's M2 holds across the split. Every comparison is memcmp: PP changes
// where the layers and the head run, not what they compute (the same kernels in the same order
// on the same data, plus the hand-off's copies).
//
//   reference  one card (runtime::Engine on device 0, the head loaded): the prose prompt
//              through Engine::prefill (which fills the head's KV); then a greedy speculative
//              run of 64 iterations with K cycling 3, 1, 2, 0 - every draft's logits row,
//              every verify's k + 1 rows, the ids; the session at the end (save_state: the
//              live verify slot, the conv rings, the head's hidden row; save_kv over [0, pos)
//              with the head's layer; the Control block). M2: from a snapshot, 4 plain steps'
//              logits rows against ONE verify at M = 4 over the same ids (teacher-forced).
//              Spec 7 with MTP: a 4103-id hooked prefill (block hooks at 2048 and 4096, the
//              state and KV saved inside each hook, then at the end).
//   each configuration  --pp 2 copy and peer at the auto split, and the cuts at 5 and 62: the
//              same three runs on runtime::PipelineEngine, everything above bitwise one card's;
//              both Control blocks equal after every call (Review Focus 3); M2 on the pipeline
//              itself; the one card's snapshot after the prefill restored into the pipeline and
//              the speculative run continued from it, bitwise.
//
// argv: [1] the checkpoint, [2] `int8` for the int8 lm_head. B70_KV_CACHE selects the KV form
// (pp_mtp_kv8_test). Exits 77 (SKIP) with fewer than two GPUs or without peer access.
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_plan.h"

namespace {
using model::Qwen35;
using Bytes = std::vector<uint8_t>;
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kIters = 64;
constexpr uint32_t kHooked = 4103;   // 2048 + 2048 + 7: two mid-prompt block ends

// What a run leaves, compared bitwise across the engines.
struct Record {
  std::vector<uint32_t> ids;
  std::vector<float> rows;   // every draft row, then every verify's rows, in order
  Bytes state, kv;
  runtime::Control ctl{};
  std::vector<float> m2_plain, m2_verify;   // M2: 4 plain rows / one M = 4 verify's rows
  std::vector<Bytes> hooked;                // state + KV at 2048, 4096, 4103
};

// The engines' calls, one shape (runtime::Engine on device 0, or the pipeline).
struct OneCard {
  runtime::Engine& e;
  l0::CmdList imm;
  runtime::Control* c() { return e.buffers().control.as<runtime::Control>(); }
  uint32_t pending() { return c()->cur_token[0]; }
  void set_pending(uint32_t id) { c()->cur_token[0] = id; }
  void set_draft(uint32_t i, uint32_t id) { c()->cur_token[1 + i] = id; }
  void rows(std::vector<float>& out, uint32_t n) {
    const size_t at = out.size();
    out.resize(at + size_t(n) * Qwen35::kVocab);
    imm.copy(out.data() + at, e.verify_logits_device(), size_t(n) * Qwen35::kVocab * 4);
  }
  void qrow(std::vector<float>& out, uint32_t i) {
    const size_t at = out.size();
    out.resize(at + Qwen35::kVocab);
    imm.copy(out.data() + at, e.mtp_logits_device() + size_t(i) * Qwen35::kVocab, Qwen35::kVocab * 4);
  }
  runtime::Control control() { return *c(); }
  void check_in_step(const char*) {}
};
struct TwoCards {
  runtime::PipelineEngine& e;
  uint32_t pending() { return e.pending(); }
  void set_pending(uint32_t id) { e.set_token(id); }
  void set_draft(uint32_t i, uint32_t id) { e.set_draft_input(i, id); }
  void rows(std::vector<float>& out, uint32_t n) {
    const size_t at = out.size();
    out.resize(at + size_t(n) * Qwen35::kVocab);
    e.read_logits_into(out.data() + at, n);
  }
  void qrow(std::vector<float>& out, uint32_t i) {
    const size_t at = out.size();
    out.resize(at + Qwen35::kVocab);
    e.read_draft_logits_into(out.data() + at, i);
  }
  runtime::Control control() { return e.control(1); }
  void check_in_step(const char* where) {
    if (std::memcmp(&e.control(0), &e.control(1), sizeof(runtime::Control)) != 0) {
      std::fprintf(stderr, "%s: the two Control blocks differ (pos %u / %u, live %u / %u)\n", where,
                   e.control(0).pos, e.control(1).pos, e.control(0).gdn_live, e.control(1).gdn_live);
      CHECK(false);
    }
  }
};

const loader::LoadedModel& model_of(runtime::Engine& e) { return e.model(); }
const loader::LoadedModel& model_of(runtime::PipelineEngine& e) { return e.model(0); }

template <class E>
Bytes state_of(E& e) {
  Bytes s(e.state_bytes());
  e.save_state(s.data());
  return s;
}
template <class E>
Bytes kv_of(E& e, uint32_t end) {
  Bytes s(e.kv_bytes(end));
  e.save_kv(0, end, s.data());
  return s;
}

// The speculative run: K cycles 3, 1, 2, 0; greedy acceptance (b70-decode --mtp's loop).
template <class E, class W>
void spec_run(E& e, W& w, Record& r) {
  static constexpr uint32_t kCycle[4] = {3, 1, 2, 0};
  for (uint32_t it = 0; it < kIters; ++it) {
    const uint32_t k = std::min(kCycle[it % 4], e.max_verify_k());
    const uint32_t x = w.pending();
    r.ids.push_back(x);
    if (k == 0) {
      e.verify(0);
      w.rows(r.rows, 1);
      e.commit(0, e.verify_ids()[0]);
    } else {
      e.draft(k);
      for (uint32_t i = 0; i < k; ++i) w.qrow(r.rows, i);
      e.verify(k);
      w.rows(r.rows, k + 1);
      uint32_t j = 0;
      while (j < k && e.draft_ids()[j] == e.verify_ids()[j]) ++j;
      for (uint32_t i = 0; i < j; ++i) r.ids.push_back(e.draft_ids()[i]);
      e.commit(j, e.verify_ids()[j]);
    }
    w.check_in_step("spec run");
  }
  r.state = state_of(e);
  r.kv = kv_of(e, e.pos());
  r.ctl = w.control();
}

// M2 at the session's pos: 4 plain steps, then (restored) one verify at M = 4 of those ids.
template <class E, class W>
void m2(E& e, W& w, Record& r) {
  const uint32_t at = e.pos();
  const Bytes st = state_of(e), kv = kv_of(e, at);
  const uint32_t x0 = w.pending();
  std::vector<uint32_t> ids{x0};
  for (uint32_t i = 0; i < 4; ++i) {
    e.verify(0);
    w.rows(r.m2_plain, 1);
    ids.push_back(e.verify_ids()[0]);
    e.commit(0, e.verify_ids()[0]);
  }
  e.load_state(st.data(), at);
  e.load_kv(0, at, kv.data());
  w.set_pending(x0);
  for (uint32_t i = 0; i < 3; ++i) w.set_draft(i, ids[1 + i]);
  e.verify(3);
  w.rows(r.m2_verify, 4);
  e.commit(3, e.verify_ids()[3]);
  w.check_in_step("M2");
}

// The head's KV row at `end - 1` in a mid-prompt snapshot: one card has not written it yet (the
// next chunk's step_mtp_kv writes it, from the pair (h_{end-1}, x_end)); under --pp 2 that
// chunk may already be running on device 1 when the hook reads the KV. Nothing reads the row
// before rewriting it after a restore (the next prefill chunk or verify fills it first), so the
// comparison blanks it in both records. Host layout (Engine's kv_runs): per tensor the FA
// layers' rows, the head's rows, [the layers' scales, the head's scales].
void blank_head_row(Bytes& kv, const runtime::KvLayout& L, uint32_t fa, uint32_t end) {
  const size_t rows = size_t(end) * L.row_bytes(), scales = size_t(end) * L.scale_row_bytes();
  const size_t tensor = (fa + 1) * (rows + scales);
  for (size_t t = 0; t < 2; ++t) {
    uint8_t* base = kv.data() + t * tensor;
    std::memset(base + fa * rows + size_t(end - 1) * L.row_bytes(), 0, L.row_bytes());
    if (scales != 0)
      std::memset(base + (fa + 1) * rows + fa * scales + size_t(end - 1) * L.scale_row_bytes(), 0,
                  L.scale_row_bytes());
  }
}

// Spec 7 with MTP: a hooked prefill; the hook saves the state and the KV (the head's layer too).
template <class E>
void hooked(E& e, const std::vector<uint32_t>& ids, Record& r) {
  const model::ModelDesc& d = *model_of(e).desc;
  const runtime::KvLayout L = runtime::kv_layout(kMaxLen, d, 1, runtime::default_kv_cache());
  e.set_block_hook([&](uint32_t end, bool) {
    Bytes b = state_of(e);
    Bytes kv = kv_of(e, end);
    if (end < ids.size()) blank_head_row(kv, L, d.fa_layers, end);   // a mid-prompt block end
    b.insert(b.end(), kv.begin(), kv.end());
    r.hooked.push_back(std::move(b));
  });
  e.prefill(ids);
  e.set_block_hook({});
}

void same(const std::vector<float>& a, const std::vector<float>& b, const std::string& what) {
  CHECK_EQ(a.size(), b.size());
  if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
    size_t i = 0;
    while (i < a.size() && std::memcmp(&a[i], &b[i], 4) == 0) ++i;
    std::fprintf(stderr, "%s: differs at float %zu (row %zu)\n", what.c_str(), i, i / Qwen35::kVocab);
    CHECK(false);
  }
}
void same(const Bytes& a, const Bytes& b, const std::string& what) {
  CHECK_EQ(a.size(), b.size());
  if (a != b) {
    size_t i = 0;
    while (a[i] == b[i]) ++i;
    std::fprintf(stderr, "%s: differs at byte %zu of %zu\n", what.c_str(), i, a.size());
    CHECK(false);
  }
}
void compare(const Record& ref, const Record& got, const std::string& what) {
  if (ref.ids != got.ids) {
    size_t i = 0;
    while (i < ref.ids.size() && i < got.ids.size() && ref.ids[i] == got.ids[i]) ++i;
    std::fprintf(stderr, "%s: ids differ at %zu\n", what.c_str(), i);
    CHECK(false);
  }
  same(ref.rows, got.rows, what + ": draft / verify logits");
  same(ref.state, got.state, what + ": state (live slot, conv rings, head hidden)");
  same(ref.kv, got.kv, what + ": KV with the head's layer");
  CHECK_EQ(ref.ctl.pos, got.ctl.pos);
  CHECK_EQ(ref.ctl.cur_token[0], got.ctl.cur_token[0]);
  CHECK_EQ(ref.ctl.gdn_live, got.ctl.gdn_live);
  same(ref.m2_plain, got.m2_plain, what + ": M2 plain rows");
  same(ref.m2_verify, got.m2_verify, what + ": M2 verify rows");
  CHECK_EQ(ref.hooked.size(), got.hooked.size());
  for (size_t i = 0; i < ref.hooked.size(); ++i)
    same(ref.hooked[i], got.hooked[i], what + ": hooked snapshot " + std::to_string(i));
  std::printf("%s: %zu ids, %zu logits rows, state, KV + head layer, M2 and %zu hooked snapshots "
              "bitwise one card; pos %u, live slot %u\n",
              what.c_str(), got.ids.size(), got.rows.size() / Qwen35::kVocab, got.hooked.size(),
              got.ctl.pos, got.ctl.gdn_live);
  std::fflush(stdout);
}

std::vector<uint32_t> repeat_to(const std::vector<uint32_t>& src, size_t n) {
  std::vector<uint32_t> out;
  while (out.size() < n) out.insert(out.end(), src.begin(), src.end());
  out.resize(n);
  return out;
}

struct Config {
  runtime::PpHandoff handoff;
  uint32_t split;   // 0 = auto
};
}  // namespace

int main(int argc, char** argv) {
  const std::string checkpoint = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const loader::LmHeadForm head = argc > 2 && std::string(argv[2]) == "int8" ? loader::LmHeadForm::Int8
                                                                             : loader::LmHeadForm::Checkpoint;
  const runtime::KvCache kv = runtime::default_kv_cache();
  const uint32_t gpus = l0::Context::gpu_count();
  if (gpus < 2) {
    std::printf("SKIP: --pp 2 needs two GPUs, Level Zero shows %u\n", gpus);
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1)) {
    std::printf("SKIP: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer)\n");
    return 77;
  }
  const std::vector<uint32_t> prompt = golden::read_ids("tests/golden/prompts/prose.ids");
  const std::vector<uint32_t> longp = repeat_to(golden::read_ids("tests/golden/prompts/long32k.ids"), kHooked);

  // --- the single-card reference on device 0 -------------------------------------------------
  Record ref;
  Bytes snap_state, snap_kv;
  uint32_t snap_at = 0, snap_pending = 0;
  {
    runtime::Engine e(d0, loader::load(d0, checkpoint, kMaxLen, /*mtp=*/true, head), kMaxLen, false, kv);
    OneCard w{e, l0::CmdList::immediate(d0)};
    e.prepare_prefill();
    hooked(e, longp, ref);
    e.reset();
    e.prefill(prompt);
    snap_at = e.pos();
    snap_pending = w.pending();
    snap_state = state_of(e);
    snap_kv = kv_of(e, snap_at);
    spec_run(e, w, ref);
    m2(e, w, ref);
    // One card's own M2: the verify rows ARE the plain rows (spec 8, plan 8b).
    same(ref.m2_plain, ref.m2_verify, "one card: M2");
    std::printf("reference (one card, device 0): %zu ids in %u iterations, M2 bitwise\n", ref.ids.size(), kIters);
  }

  bool first = true;
  for (const Config c : {Config{runtime::PpHandoff::Copy, 0}, Config{runtime::PpHandoff::Peer, 0},
                         Config{runtime::PpHandoff::Copy, 5}, Config{runtime::PpHandoff::Peer, 62}}) {
    loader::LoadedModel full = loader::load(d0, checkpoint, kMaxLen, /*mtp=*/true, head);
    const runtime::PpWeights wts = runtime::pp_weights(full);
    runtime::PpExtras x;
    x.mtp = true;
    const uint32_t split = c.split ? c.split
                                   : runtime::pp_auto_split(*full.desc, wts, kMaxLen, kv,
                                                            runtime::PrefillPath{}, x);
    runtime::PipelineOptions opt;
    opt.handoff = c.handoff;
    runtime::PipelineEngine e(d0, d1, runtime::place_stages(d0, d1, std::move(full), split), kMaxLen, opt, kv);
    CHECK(e.mtp());
    TwoCards w{e};
    const std::string what = std::string("--pp 2 --mtp, ") + runtime::pp_handoff_name(c.handoff) +
                             ", split " + std::to_string(split);
    // The lists: the stages' verify launches add up to one card's at every M (capture asserts
    // it too), the draft lists are one card's.
    for (uint32_t M = 1; M <= 4; ++M)
      CHECK_EQ(e.verify_step(0, M).kernel_count + e.verify_step(1, M).kernel_count,
               runtime::pp_stage_verify_launches(*e.model(0).desc, e.stage(0)) +
                   runtime::pp_stage_verify_launches(*e.model(0).desc, e.stage(1)) +
                   (c.handoff == runtime::PpHandoff::Peer ? 2u : 0u));
    for (uint32_t d = 0; d < 2; ++d) std::printf("  %s\n", e.memory_line(d).c_str());
    Record got;
    e.prepare_prefill();
    hooked(e, longp, got);
    w.check_in_step("hooked prefill");
    e.reset();
    e.prefill(prompt);
    w.check_in_step("prefill");
    CHECK_EQ(e.pos(), snap_at);
    same(state_of(e), snap_state, what + ": the state after the prefill");
    same(kv_of(e, snap_at), snap_kv, what + ": the KV after the prefill (the head's prefill fill)");
    spec_run(e, w, got);
    m2(e, w, got);
    same(got.m2_plain, got.m2_verify, what + ": M2 across the split");
    compare(ref, got, what);
    // A one-card snapshot continued on two cards (Review Focus 2: the layouts are one).
    if (first) {
      e.reset();
      e.load_state(snap_state.data(), snap_at);
      e.load_kv(0, snap_at, snap_kv.data());
      w.set_pending(snap_pending);
      Record cont;
      spec_run(e, w, cont);
      CHECK(cont.ids == ref.ids);
      same(cont.rows, ref.rows, what + ": continued from one card's snapshot");
      std::printf("%s: one card's snapshot at %u continued on two cards, bitwise\n", what.c_str(), snap_at);
      first = false;
    }
  }
  std::printf("pp_mtp_test: OK\n");
  return 0;
}
