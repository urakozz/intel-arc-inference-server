// Plan 19v3b Task 3: runtime::Engine's sequence slots beside the DFlash drafter - a drafter session (ring, taps,
// control block, readback, lists) and GDN states per slot (the gate checkpoint, int8 head, KV form B70_KV_CACHE's).
//
// The session: prompt long32k[0, 1998) prefilled, then 24 greedy DFlash iterations at K = 3 (draft, verify(3), the
// agreeing prefix, commit). Its record: the ids, and after the last iteration the drafter session's ring and taps, and
// the slot's live GDN state - each hashed.
//
//   A   the session in a one-slot engine (max_len 32768, the slots' sum: the same attention builds): the reference.
//   B1  a slot alone is the one-slot engine: in an engine of slots 16384 + 8192 + 8192, the session in slot 0, in
//       slot 1 and in slot 2 gives A's record bitwise (slot 0 binds today's 8-state lists over its pages, a further
//       slot its 4-state lists - the same kernels and arithmetic over its own state, session and KV).
//   B2  isolation: slots 1 and 2 run the session in turn (one iteration each, alternating) - both give A's record,
//       and slot 0 (not stepped) keeps every byte (Control, GDN states, conv ring, drafter session).
//   B3  a further slot drafts at most 3: max_verify_k() <= 3 there and verify(4) is refused.
//   B4  reset, snapshot and restore of slot 1 leave slot 2's bytes alone, and the restore brings slot 1's ring back
//       bitwise.
//
// usage: dflash_slots_test <target snapshot> <drafter>   (cwd = the source tree)
// SKIP (77) when either checkpoint is missing.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/engine.h"

namespace {
constexpr uint32_t kK = 3, kPrompt = 1998, kIters = 24;
const std::vector<uint32_t> kSlots = {16384, 8192, 8192};

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

struct Rig {
  l0::Context& ctx;
  std::unique_ptr<runtime::dflash::DFlashDrafter> d;
  std::unique_ptr<runtime::Engine> e;
  l0::CmdList imm;
  Rig(l0::Context& c, const std::string& snap, const std::string& dsnap, std::vector<uint32_t> lens)
      : ctx(c), imm(l0::CmdList::immediate(c)) {
    uint32_t max_len = 0;
    for (uint32_t l : lens) max_len += l;
    if (lens.size() == 1) lens.clear();
    loader::LoadedModel model = loader::load(ctx, snap, max_len, /*mtp=*/false, loader::LmHeadForm::Int8);
    loader::DFlashWeights w = loader::load_drafter(ctx, dsnap, model, loader::DraftFormat::Int8);
    const uint32_t rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
    d = std::make_unique<runtime::dflash::DFlashDrafter>(ctx, std::move(w), rows, max_len, 7);
    e = std::make_unique<runtime::Engine>(ctx, std::move(model), max_len, false, runtime::default_kv_cache(), d.get(),
                                          lens);
    d->attach(*e, false);
  }
  uint64_t dev(const void* p, size_t n, uint64_t h) {
    std::vector<uint8_t> v(n);
    imm.copy(v.data(), p, n);
    return fnv(v.data(), n, h);
  }
  // Slot q's drafter session (ring and taps) and its GDN states (all, or only the live one) and conv ring.
  uint64_t session_hash(uint32_t q) {
    const runtime::dflash::DFlashBuffers::Session s = d->buffers().session(q);
    return dev(s.taps->ptr(), s.taps->size(),
               dev(s.ring_v->ptr(), s.ring_v->size(), dev(s.ring_k->ptr(), s.ring_k->size(), 1469598103934665603ull)));
  }
  uint64_t live_state_hash(uint32_t q) {
    const runtime::Engine::SlotView v = e->slot_view(q);
    const size_t st = e->buffers().gdn_state.size();
    const uint32_t live = v.control->gdn_live;
    return dev(live == 0 ? v.state0 : v.spec + size_t(live - 1) * st, st, 1469598103934665603ull);
  }
  uint64_t slot_hash(uint32_t q) {
    const runtime::Engine::SlotView v = e->slot_view(q);
    const size_t st = e->buffers().gdn_state.size();
    uint64_t h = fnv(v.control, sizeof(runtime::Control));
    h = dev(v.state0, st, h);
    h = dev(v.spec, st * (v.states - 1), h);
    h = dev(v.ring, e->buffers().conv_ring.size(), h);
    return h ^ session_hash(q);
  }
};

struct Record {
  std::vector<uint32_t> ids;
  uint64_t session = 0, state = 0;
  bool operator==(const Record& o) const { return ids == o.ids && session == o.session && state == o.state; }
};

// One greedy DFlash iteration at K = kK on the selected slot.
void iterate(Rig& r, std::vector<uint32_t>& out) {
  runtime::Engine& e = *r.e;
  out.push_back(e.control()->cur_token[0]);
  const uint32_t k = std::min(kK, e.max_verify_k());
  const runtime::dflash::DFlashDraftOut& o = r.d->draft(e, k, 0.0f, 0);
  const std::vector<uint32_t> drafts(o.ids, o.ids + k);
  e.verify(k);
  uint32_t j = 0;
  while (j < k && drafts[j] == e.verify_ids()[j]) ++j;
  for (uint32_t i = 0; i < j; ++i) out.push_back(drafts[i]);
  e.commit(j, e.verify_ids()[j]);
}

void start(Rig& r, uint32_t q, const std::vector<uint32_t>& prompt) {
  r.e->select(q);
  r.e->reset();
  r.e->prefill(prompt);
}
Record finish(Rig& r, uint32_t q, std::vector<uint32_t> ids) {
  r.e->select(q);
  r.e->flush_projection();
  return {std::move(ids), r.session_hash(q), r.live_state_hash(q)};
}
Record session(Rig& r, uint32_t q, const std::vector<uint32_t>& prompt) {
  start(r, q, prompt);
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < kIters; ++i) iterate(r, ids);
  return finish(r, q, ids);
}
void print(const char* what, const Record& x) {
  std::printf("%s: %zu ids, session %016llx, live GDN state %016llx\n", what, x.ids.size(),
              (unsigned long long)x.session, (unsigned long long)x.state);
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
  const std::vector<uint32_t> all = golden::read_ids("tests/golden/prompts/long32k.ids");
  CHECK(all.size() >= kPrompt);
  const std::vector<uint32_t> prompt(all.begin(), all.begin() + kPrompt);
  l0::Context ctx(0);
  bool ok = true;

  Record ref;
  {   // A
    Rig r(ctx, snap, dsnap, {32768});
    ref = session(r, 0, prompt);
    print("A  one slot", ref);
  }

  Rig r(ctx, snap, dsnap, kSlots);
  std::printf("%s\n", r.e->memory_line().c_str());
  // B1
  for (uint32_t q = 0; q < kSlots.size(); ++q) {
    const Record x = session(r, q, prompt);
    const bool same = x == ref;
    std::printf("B1 slot %u: %s\n", q, same ? "bitwise the one-slot engine" : "DIFFERS");
    print("   ", x);
    ok &= same;
  }

  {   // B2: slots 1 and 2 in turn, slot 0 untouched
    start(r, 0, prompt);
    r.e->flush_projection();
    const uint64_t z0 = r.slot_hash(0);
    start(r, 1, prompt);
    start(r, 2, prompt);
    std::vector<uint32_t> a, b;
    for (uint32_t i = 0; i < kIters; ++i) {
      r.e->select(1);
      iterate(r, a);
      r.e->select(2);
      iterate(r, b);
    }
    const Record x1 = finish(r, 1, a), x2 = finish(r, 2, b);
    const bool same = x1 == ref && x2 == ref, still = r.slot_hash(0) == z0;
    std::printf("B2 slots 1 and 2 in turn: %s; slot 0's bytes %s\n", same ? "both bitwise the one-slot engine" : "DIFFER",
                still ? "unchanged" : "MOVED");
    ok &= same && still;
  }

  {   // B3
    r.e->select(1);
    const uint32_t mk = r.e->max_verify_k();
    bool refused = false;
    try {
      r.e->verify(4);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    const bool b3 = mk == 3 && r.e->max_draft() == 3 && refused;
    std::printf("B3 slot 1: max_verify_k %u, max_draft %u, verify(4) %s: %s\n", mk, r.e->max_draft(),
                refused ? "refused" : "RAN", b3 ? "ok" : "WRONG");
    ok &= b3;
  }

  {   // B4: slot 1's snapshot, reset and restore; slot 2 untouched
    start(r, 1, prompt);
    start(r, 2, prompt);
    std::vector<uint32_t> scratch;
    iterate(r, scratch);   // slot 2 one iteration ahead, so its bytes differ from slot 1's
    r.e->select(1);
    r.e->flush_projection();
    const auto ring_hash = [&](uint32_t q) {
      const runtime::dflash::DFlashBuffers::Session s = r.d->buffers().session(q);
      return r.dev(s.ring_v->ptr(), s.ring_v->size(), r.dev(s.ring_k->ptr(), s.ring_k->size(), 1469598103934665603ull));
    };
    const uint64_t s2 = r.slot_hash(2), ring1 = ring_hash(1);
    l0::Mem st(ctx, l0::MemKind::Host, r.e->state_bytes());
    l0::Mem kv(ctx, l0::MemKind::Host, r.e->kv_bytes(r.e->pos()));
    const uint32_t pos = r.e->pos();
    r.e->save_state(st.ptr());
    r.e->save_kv(0, pos, kv.ptr());
    r.e->reset();
    const bool zeroed = ring_hash(1) != ring1;
    r.e->load_state(st.ptr(), pos);
    r.e->load_kv(0, pos, kv.ptr());
    const bool back = ring_hash(1) == ring1;
    const bool b4 = zeroed && back && r.slot_hash(2) == s2;
    std::printf("B4 slot 1 reset %s, restore %s; slot 2's bytes %s: %s\n", zeroed ? "zeroed its ring" : "LEFT it",
                back ? "brought the ring back bitwise" : "DIFFERS", r.slot_hash(2) == s2 ? "unchanged" : "MOVED",
                b4 ? "ok" : "WRONG");
    ok &= b4;
  }

  std::printf("dflash_slots_test: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
