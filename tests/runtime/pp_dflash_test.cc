// Spec 19d on two cards: a DFlash drafter across the split (b70-serve --spec dflash --pp 2) is BITWISE the
// single-card drafter on device 0 (spec 19c), exactly as spec 16d holds MTP across the split to one card's: the same
// kernels in the same order on the same data, plus the hand-off's and the taps' copies.
//
//   reference  one card (runtime::Engine on device 0 with the int8 drafter, the int8 head, bf16 KV): a 3000-id
//              prompt through Engine::prefill (the ring from the prompt's last 2048 positions); the drafter's ring;
//              then the server's greedy loop for 160 ids at K = 7 (draft -> verify(7) -> the longest agreeing
//              prefix -> commit): every draft's ids and every verify's k + 1 logits rows, the ids; the session
//              (save_state: the live slot, the conv rings, the ring) at the end. A 4103-id hooked prefill: the state
//              saved inside the hooks at 2048 and 4096 (ring included) and at the end.
//   pipeline   runtime::PipelineEngine (copy hand-off, the auto split and the cut at 20 - the 19 / 33 taps on either
//              side) with the drafter on device 1 (place_stages(..., drafter = true), attach_target): the same runs,
//              everything above bitwise one card's; then the one card's snapshot after the prefill restored into
//              the pipeline and the loop continued from it, bitwise (a snapshot moves between --pp 1 and 2 with the
//              drafter's ring). (The peer hand-off's refusal is the CLI's host test: cli_reject_serve_dflash_pp.)
//
// usage: pp_dflash_test <target snapshot> <drafter>   (cwd = the source tree). SKIP 77 with fewer than two GPUs,
// without peer access, or without either checkpoint.
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/engine.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_plan.h"

namespace {
using model::Qwen35;
using Bytes = std::vector<uint8_t>;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 16384, kK = 7, kGen = 160, kHooked = 4103;

struct Record {
  Ids ids, drafts;
  std::vector<float> rows;   // every verify's k + 1 logits rows, in order
  Bytes ring, state;
  std::vector<Bytes> hooked;   // the state at 2048, 4096 and 4103
};

Ids repeat_to(const Ids& src, size_t n) {
  Ids out;
  while (out.size() < n) out.insert(out.end(), src.begin(), src.end());
  out.resize(n);
  return out;
}

// The two engines behind one shape.
struct OneCard {
  runtime::Engine& e;
  runtime::dflash::DFlashDrafter& d;
  l0::CmdList imm;
  const runtime::dflash::DFlashDraftOut& draft(uint32_t k) { return d.draft(e, k, 0.0f, 0); }
  uint32_t max_verify_k() { return e.max_verify_k(); }
  void verify(uint32_t k) { e.verify(k); }
  const uint32_t* verify_ids() { return e.verify_ids(); }
  void commit(uint32_t j, uint32_t t) { e.commit(j, t); }
  std::vector<float> rows(uint32_t n) {
    std::vector<float> r(size_t(n) * Qwen35::kVocab);
    imm.copy(r.data(), e.verify_logits_device(), r.size() * 4);
    return r;
  }
  uint32_t pending() { return e.buffers().control.as<runtime::Control>()->cur_token[0]; }
  void set_pending(uint32_t id) {
    auto* c = e.buffers().control.as<runtime::Control>();
    c->cur_token[0] = id;
    c->n_active = 1;
  }
  Bytes state() {
    Bytes b(e.state_bytes());
    e.save_state(b.data());
    return b;
  }
  Bytes kv(uint32_t n) {
    Bytes b(e.kv_bytes(n));
    e.save_kv(0, n, b.data());
    return b;
  }
  void restore(const Bytes& st, const Bytes& kv, uint32_t at, uint32_t pend) {
    e.load_state(st.data(), at);
    e.load_kv(0, at, kv.data());
    set_pending(pend);
  }
  void prefill(const Ids& ids) { e.prefill(ids); }
  void reset() { e.reset(); }
  void hook(runtime::Engine::BlockHook h) { e.set_block_hook(std::move(h)); }
};
struct TwoCards {
  runtime::PipelineEngine& e;
  runtime::dflash::DFlashDrafter& d;
  l0::CmdList imm;
  float* host = nullptr;
  const runtime::dflash::DFlashDraftOut& draft(uint32_t k) { return e.draft_tap(k, 0.0f, 0); }
  uint32_t max_verify_k() { return e.max_verify_k(); }
  void verify(uint32_t k) { e.verify(k); }
  const uint32_t* verify_ids() { return e.verify_ids(); }
  void commit(uint32_t j, uint32_t t) { e.commit(j, t); }
  std::vector<float> rows(uint32_t n) {
    if (!host) host = e.host_rows(size_t(kK + 1) * Qwen35::kVocab);
    e.read_logits_into(host, n);
    return std::vector<float>(host, host + size_t(n) * Qwen35::kVocab);
  }
  uint32_t pending() { return e.pending(); }
  void set_pending(uint32_t id) { e.set_token(id); }
  Bytes state() {
    Bytes b(e.state_bytes());
    e.save_state(b.data());
    return b;
  }
  Bytes kv(uint32_t n) {
    Bytes b(e.kv_bytes(n));
    e.save_kv(0, n, b.data());
    return b;
  }
  void restore(const Bytes& st, const Bytes& kv, uint32_t at, uint32_t pend) {
    e.load_state(st.data(), at);
    e.load_kv(0, at, kv.data());
    set_pending(pend);
  }
  void prefill(const Ids& ids) { e.prefill(ids); }
  void reset() { e.reset(); }
  void hook(runtime::PipelineEngine::BlockHook h) { e.set_block_hook(std::move(h)); }
};

template <class W>
void loop(W& w, Record& r) {
  while (r.ids.size() < kGen) {
    const uint32_t k = std::min(kK, w.max_verify_k());
    r.ids.push_back(w.pending());
    if (k == 0) {
      w.verify(0);
      w.commit(0, w.verify_ids()[0]);
      continue;
    }
    const runtime::dflash::DFlashDraftOut& o = w.draft(k);
    const Ids d(o.ids, o.ids + k);
    r.drafts.insert(r.drafts.end(), d.begin(), d.end());
    w.verify(k);
    const std::vector<float> rows = w.rows(k + 1);
    r.rows.insert(r.rows.end(), rows.begin(), rows.end());
    uint32_t j = 0;
    while (j < k && d[j] == w.verify_ids()[j]) ++j;
    for (uint32_t i = 0; i < j; ++i) r.ids.push_back(d[i]);
    w.commit(j, w.verify_ids()[j]);
  }
  r.ids.resize(kGen);
}

template <class W>
void run_all(W& w, const Ids& prompt, const Ids& longp, Record& r, Bytes* snap_state, Bytes* snap_kv,
             uint32_t* snap_pend) {
  w.reset();
  w.hook([&](uint32_t, bool) { r.hooked.push_back(w.state()); });
  w.prefill(longp);
  w.hook({});
  w.reset();
  w.prefill(prompt);
  if (snap_state) {
    *snap_state = w.state();
    *snap_kv = w.kv(uint32_t(prompt.size()));
    *snap_pend = w.pending();
  }
  r.ring = Bytes(w.d.ring_bytes());
  w.d.save_ring(w.imm, r.ring.data());
  loop(w, r);
  r.state = w.state();
}

bool same_bytes(const void* a, const void* b, size_t n, const std::string& what) {
  const bool ok = std::memcmp(a, b, n) == 0;
  std::printf("  %s: %s\n", what.c_str(), ok ? "bitwise" : "DIFFERS");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  std::string snap, dsnap;
  try {
    snap = loader::resolve_snapshot(argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ");
    dsnap = loader::resolve_snapshot(argc > 2 ? argv[2] : "z-lab/Qwen3.8-27B-DFlash2");
  } catch (const std::exception& ex) {
    std::printf("SKIP: %s\n", ex.what());
    return 77;
  }
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: --pp 2 needs two GPUs\n");
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1)) {
    std::printf("SKIP: device 0 cannot access device 1's memory\n");
    return 77;
  }
  const Ids prompt = golden::read_ids("tests/golden/prompts/long32k.ids");
  const Ids p3000(prompt.begin(), prompt.begin() + 3000);
  const Ids longp = repeat_to(prompt, kHooked);
  bool ok = true;

  Record ref;
  Bytes snap_state, snap_kv;
  uint32_t snap_pend = 0;
  {
    loader::LoadedModel m = loader::load(d0, snap, kMaxLen, /*mtp=*/false, loader::LmHeadForm::Int8);
    loader::DFlashWeights w = loader::load_drafter(d0, dsnap, m, loader::DraftFormat::Int8);
    const uint32_t rows = m.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
    runtime::dflash::DFlashDrafter d(d0, std::move(w), rows, kMaxLen, kK);
    runtime::Engine e(d0, std::move(m), kMaxLen, false, runtime::KvCache::Bf16, &d);
    d.attach(e, false);
    e.prepare_prefill();
    OneCard oc{e, d, l0::CmdList::immediate(d0)};
    run_all(oc, p3000, longp, ref, &snap_state, &snap_kv, &snap_pend);
    std::printf("reference (one card): %zu ids, %zu drafts, %zu hooked snapshots\n", ref.ids.size(), ref.drafts.size(),
                ref.hooked.size());
  }

  for (uint32_t cut : {0u, 20u}) {
    loader::LoadedModel full = loader::load(d0, snap, kMaxLen, /*mtp=*/false, loader::LmHeadForm::Int8);
    const runtime::PpWeights wts = runtime::pp_weights(full);
    const uint32_t split = cut ? cut : runtime::pp_auto_split(*full.desc, wts, kMaxLen, runtime::KvCache::Bf16,
                                                             runtime::PrefillPath{}, runtime::PpExtras{});
    std::vector<loader::LoadedModel> stages = runtime::place_stages(d0, d1, std::move(full), split, /*drafter=*/true);
    loader::DFlashWeights w = loader::load_drafter(d1, dsnap, stages[1], loader::DraftFormat::Int8);
    const uint32_t rows = stages[1].linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
    runtime::dflash::DFlashDrafter d(d1, std::move(w), rows, kMaxLen, kK);
    runtime::PipelineOptions opt;
    runtime::PipelineEngine e(d0, d1, std::move(stages), kMaxLen, opt, runtime::KvCache::Bf16, &d);
    d.attach_target(d1, e.model(1), e.buffers(1).control, kMaxLen, false);
    e.prepare_prefill();
    TwoCards tc{e, d, l0::CmdList::immediate(d1)};
    Record got;
    run_all(tc, p3000, longp, got, nullptr, nullptr, nullptr);
    std::printf("--pp 2 --spec dflash, copy, split %u:\n", split);
    bool c_ok = got.ids == ref.ids && got.drafts == ref.drafts;
    std::printf("  ids %s, drafts %s\n", got.ids == ref.ids ? "equal" : "DIFFER", got.drafts == ref.drafts ? "equal" : "DIFFER");
    c_ok &= got.rows.size() == ref.rows.size() &&
            same_bytes(got.rows.data(), ref.rows.data(), ref.rows.size() * 4, "every verify's logits rows");
    c_ok &= same_bytes(got.ring.data(), ref.ring.data(), ref.ring.size(), "the ring after the prefill");
    c_ok &= got.state.size() == ref.state.size() &&
            same_bytes(got.state.data(), ref.state.data(), ref.state.size(), "the session (state + ring) at the end");
    c_ok &= got.hooked.size() == ref.hooked.size();
    for (size_t i = 0; i < got.hooked.size() && i < ref.hooked.size(); ++i)
      c_ok &= same_bytes(got.hooked[i].data(), ref.hooked[i].data(), ref.hooked[i].size(),
                         "hooked snapshot " + std::to_string(i));
    // A one-card snapshot continued on two cards.
    Record cont;
    tc.restore(snap_state, snap_kv, 3000, snap_pend);
    loop(tc, cont);
    std::printf("  the one card's snapshot continued on two cards: ids %s\n", cont.ids == ref.ids ? "equal" : "DIFFER");
    c_ok &= cont.ids == ref.ids && cont.drafts == ref.drafts;
    std::printf("  %s\n", c_ok ? "PASS" : "FAIL");
    ok &= c_ok;
  }
  std::printf("pp_dflash_test: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
