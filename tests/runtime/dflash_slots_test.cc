// Plan 19v3b Task 3: runtime::Engine's sequence slots beside the DFlash drafter - a drafter session (ring, taps,
// control block, readback, lists) and GDN states per slot (the gate checkpoint, int8 head, KV form B70_KV_CACHE's).
//
// The session: prompt long32k[0, 1998) prefilled, then 24 greedy DFlash iterations at K = 3 (draft, verify(3), the
// agreeing prefix, commit). Its record: the ids, and after the last iteration the drafter session's ring and taps, and
// the slot's live GDN state - each hashed.
//
//   A   the session in a one-slot engine (max_len 40960, the pool's positions: the same attention builds): the
//       reference.
//   B1  a slot alone is the one-slot engine: in an engine of 4 sequences up to 16384 sharing a KV page pool of 20
//       pages (spec 31), the session in every slot gives A's record bitwise (slot 0 binds today's 8-state lists over
//       its pages, a further slot its 4-state lists - the same kernels and arithmetic over its own state, session
//       and KV).
//   B2  isolation: slots 1 and 2 run the session in turn (one iteration each, alternating) - both give A's record,
//       and slot 0 (not stepped) keeps every byte (Control, GDN states, conv ring, drafter session).
//   B3  a further slot drafts the drafter's whole block (plan 40a Task 2: its rows pending, no speculative states):
//       max_verify_k() 7 there and verify(8) is refused.
//   B4  reset, snapshot and restore of slot 1 leave slot 2's bytes alone, and the restore brings slot 1's ring back
//       bitwise.
//   B5  the batched draft (DFlashDrafter::draft_rows): slots 1 and 2 at K = 3 in one pass against each slot's own draft
//       from the same state - the drafts printed, equal unless a near-tie (the batched attention sums at KS 1, the lone
//       one at KS 4: within the drafter's f32-judge form), the candidates' scores within 0.12 (they only steer drafting;
//       verify decides every token - owner ruling 2026-11-24).
//   B6  a row step beside the drafter: verify_rows of B5's runs, the greedy commits, ONE rows projection - the ring
//       rows it stored for each slot against the same verify done alone (verify(3) + commit) from the same state:
//       worst row cosine >= 0.999 (the row kernels round as the small-batch GEMMs do).
//   B7  the batched session: slots 1 and 2, 24 iterations of batched draft + row step + commit each, against A's ids -
//       the first difference printed (the row kernels' rounding; judged at the server, Task 7).
//   B8  (plan 40a Task 4, a drafter with the batched pass at four blocks: the int4 drafter with the candidate head)
//       four slots at different positions, K = 7 each: (a) every block of the 32-row pass bitwise the same slot's run
//       through the 8-row batched list alone (ids, candidates, head logits, scores); (b) against each slot's own draft
//       (KS 4): equal drafts, the candidates' score differences printed (the batched part's KS 1 sums against the lone
//       KS 4: up to 0.11 measured, the 8-row list's own - (a) holds the 32-row pass to it); (c) the 32-row step
//       (verify_rows 4 x K 7, the greedy commits, ONE rows projection at 32 rows) - each slot's ring rows against the
//       same verify done alone, worst cosine >= 0.999 (B6's); (d) the batched session of the four slots, kIters
//       iterations, against A's ids - the first difference printed (B7's).
//   T   (timing, informational, busy-box grade) the draft list of one slot at K = 7 against the batched draft of 2
//       slots x K 3 and 4 slots x K 1 (and, with B8's pass, 4 slots x K 7), interleaved, medians of 15 rounds; the
//       batched one in the lone draft's plain steps (0.217 at 4k, BENCHMARKS "the final form").
//
// usage: dflash_slots_test <target snapshot> <drafter> [--pool]   (cwd = the source tree; --pool: pool_main below)
// SKIP (77) when either checkpoint is missing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "cli/slots_engine.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_loader.h"
#include "loader/dflash_repack.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/engine.h"

namespace {
constexpr uint32_t kK = 3, kPrompt = 1998, kIters = 24;
// The pool: 4 sequences up to 16384 sharing 20 pages (40960 positions).
constexpr uint32_t kSeqs = 4, kCap = 16384, kPoolPages = 20;

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
  // One sequence of lens[0], or (spec 31) several sharing `pool_pages` KV pages, every one up to lens[0].
  Rig(l0::Context& c, const std::string& snap, const std::string& dsnap, std::vector<uint32_t> lens,
      uint32_t pool_pages = 0)
      : ctx(c), imm(l0::CmdList::immediate(c)) {
    const uint32_t max_len = pool_pages != 0 ? pool_pages * runtime::kKvPage : lens[0];
    if (lens.size() == 1) lens.clear();
    loader::LoadedModel model = loader::load(ctx, snap, max_len, /*mtp=*/false, loader::LmHeadForm::Int8);
    // As b70-serve loads it: a packed checkpoint as int4 with the candidate head, the original as int8.
    const bool packed = loader::DFlashCheckpoint(dsnap).packed();
    loader::DFlashWeights w = loader::load_drafter(ctx, dsnap, model,
                                                   packed ? loader::DraftFormat::Int4G64 : loader::DraftFormat::Int8, 0,
                                                   packed);
    const uint32_t rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
    d = std::make_unique<runtime::dflash::DFlashDrafter>(ctx, std::move(w), rows, max_len, 7,
                                                         std::max<uint32_t>(1, uint32_t(lens.size())));
    e = std::make_unique<runtime::Engine>(ctx, std::move(model), max_len, false, runtime::default_kv_cache(), d.get(),
                                          lens, pool_pages);
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
  // The selected slot's committed GDN state (a one-slot engine's deferred rows applied first: verify-cost-m8).
  uint64_t live_state_hash(uint32_t q) {
    e->select(q);
    return dev(e->gdn_state_now(), e->buffers().gdn_state.size(), 1469598103934665603ull);
  }
  uint64_t slot_hash(uint32_t q) {
    const runtime::Engine::SlotView v = e->slot_view(q);
    const size_t st = e->buffers().gdn_state.size();
    uint64_t h = fnv(v.control, sizeof(runtime::Control));
    h = dev(v.state0, st, h);
    if (v.states > 1) h = dev(v.spec, st * (v.states - 1), h);
    h = dev(v.ring, e->buffers().conv_ring.size(), h);
    return h ^ session_hash(q);
  }
};

struct Record {
  std::vector<uint32_t> ids;
  uint64_t session = 0, state = 0;
  bool operator==(const Record& o) const { return ids == o.ids && session == o.session && state == o.state; }
};

// One greedy DFlash iteration at K = `kk` (kK unless given) on the selected slot.
void iterate(Rig& r, std::vector<uint32_t>& out, uint32_t kk = kK) {
  runtime::Engine& e = *r.e;
  out.push_back(e.control()->cur_token[0]);
  const uint32_t k = std::min(kk, e.max_verify_k());
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
Record session(Rig& r, uint32_t q, const std::vector<uint32_t>& prompt, uint32_t call = 0, uint32_t kk = kK) {
  if (call == 0) {
    start(r, q, prompt);
  } else {   // the prompt in calls of `call` ids, as the scheduler hands a several-slot engine its chunks
    r.e->select(q);
    r.e->reset();
    for (size_t i = 0; i < prompt.size(); i += call)
      r.e->prefill(std::vector<uint32_t>(prompt.begin() + i, prompt.begin() + std::min(prompt.size(), i + call)));
  }
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < kIters; ++i) iterate(r, ids, kk);
  return finish(r, q, ids);
}
void print(const char* what, const Record& x) {
  std::printf("%s: %zu ids, session %016llx, live GDN state %016llx\n", what, x.ids.size(),
              (unsigned long long)x.session, (unsigned long long)x.state);
}

// Every byte of physical KV page `page` (every layer's K and V rows, and at int8 their scales) is 0.
bool page_zero(Rig& r, uint32_t page) {
  const runtime::KvLayout& L = r.e->buffers().kv_lay;
  std::vector<uint8_t> host;
  bool zero = true;
  for (uint32_t l = 0; l < L.layers; ++l) {
    const runtime::KvLayer y = L.at_base(L.layer(r.e->buffers().kv_k.ptr(), r.e->buffers().kv_v.ptr(), l),
                                         page * runtime::kKvPage);
    for (int t = 0; t < 4; ++t) {
      void* ptr = t == 0 ? y.k : t == 1 ? y.v : t == 2 ? y.ks : y.vs;
      if (!ptr) continue;
      host.resize(size_t(runtime::kKvPage) * (t < 2 ? L.row_bytes() : L.scale_row_bytes()));
      r.imm.copy(host.data(), ptr, host.size());
      zero = zero && std::all_of(host.begin(), host.end(), [](uint8_t x) { return x == 0; });
    }
  }
  return zero;
}
std::string pages_str(const std::vector<uint32_t>& v) {
  std::string out;
  for (uint32_t p : v) out += (out.empty() ? "" : ",") + std::to_string(p);
  return out;
}

// Plan 40a Task 1, `--pool`: the drafter beside the KV page pool (spec 31) - B1's engine.
//   P1  each sequence alone gives A's record bitwise, its pages handed out in order, reversed, and interleaved with
//       another sequence that iterates beside it;
//   P2  a sequence's reset gives its pages back, every byte zero, and the next session on them gives A's record;
//   P4  a prompt over three pages on sequence 1 (two prompt chunks, the drafter's window projected at its end) gives
//       the one-slot engine's record bitwise; P5 the same prompt in calls of 2048 ids (the scheduler's chunks).
int pool_main(l0::Context& ctx, const std::string& snap, const std::string& dsnap, const std::vector<uint32_t>& prompt,
              const std::vector<uint32_t>& longer) {
  constexpr uint32_t P = runtime::kKvPage;
  Record ref, ref_long;
  {
    Rig r(ctx, snap, dsnap, {kPoolPages * P});
    ref = session(r, 0, prompt);
    print("A  one slot", ref);
    ref_long = session(r, 0, longer);
    print("A  one slot, the longer prompt", ref_long);
  }
  Rig r(ctx, snap, dsnap, std::vector<uint32_t>(kSeqs, kCap), kPoolPages);
  runtime::Engine& e = *r.e;
  std::printf("dflash_slots_test --pool: 4 sequences up to %u sharing %u pages; %s\n", e.max_len(), e.kv_pool_pages(),
              e.memory_line().c_str());
  CHECK_EQ(e.pages_free(), kPoolPages);
  const auto reset_all = [&] {
    for (uint32_t q = e.slots(); q-- > 0;) {
      e.select(q);
      e.reset();
    }
  };
  bool ok = true;
  for (uint32_t q = 0; q < e.slots(); ++q) {   // P1 in order
    reset_all();
    const Record x = session(r, q, prompt);
    const bool same = x == ref;
    std::printf("P1 sequence %u in order (pages %s): %s\n", q, pages_str(e.pool_pages_of(q)).c_str(),
                same ? "bitwise the one-slot engine" : "DIFFERS");
    ok &= same;
  }
  {   // P1 reversed: sequences 1..3 take a page each and give them back - the next three come out reversed
    reset_all();
    for (uint32_t q = 1; q < 4; ++q) {
      e.select(q);
      CHECK(e.take_pages(P));
    }
    for (uint32_t q = 1; q < 4; ++q) {
      e.select(q);
      e.reset();
    }
    const Record x = session(r, 0, prompt);
    const std::vector<uint32_t> pb = e.pool_pages_of(0);
    const bool same = x == ref && pb.size() >= 2 && pb[0] > pb[1];
    std::printf("P1 reversed (pages %s): %s\n", pages_str(pb).c_str(), same ? "PASS" : "FAIL");
    ok &= same;
  }
  {   // P1 interleaved: sequences 0 and 1 take pages in turn and iterate in turn
    reset_all();
    for (uint32_t i = 1; i <= 2; ++i)
      for (uint32_t q = 0; q < 2; ++q) {
        e.select(q);
        CHECK(e.take_pages(i * P));
      }
    start(r, 0, prompt);
    start(r, 1, prompt);
    std::vector<uint32_t> a, b;
    for (uint32_t i = 0; i < kIters; ++i) {
      e.select(0);
      iterate(r, a);
      e.select(1);
      iterate(r, b);
    }
    const Record x0 = finish(r, 0, a), x1 = finish(r, 1, b);
    const bool same = x0 == ref && x1 == ref;
    std::printf("P1 interleaved (pages %s and %s): %s\n", pages_str(e.pool_pages_of(0)).c_str(),
                pages_str(e.pool_pages_of(1)).c_str(), same ? "PASS (both bitwise the one-slot engine)" : "FAIL");
    ok &= same;
  }
  {   // P2
    e.select(0);
    const std::vector<uint32_t> held = e.pool_pages_of(0);
    const uint32_t free_before = e.pages_free();
    e.reset();
    bool zero = e.pages_held(0) == 0 && e.pages_free() == free_before + held.size();
    for (uint32_t p : held) zero = zero && page_zero(r, p);
    const Record x = session(r, 2, prompt);
    bool reused = true;
    for (uint32_t p : e.pool_pages_of(2)) reused = reused && std::find(held.begin(), held.end(), p) != held.end();
    const bool same = x == ref && reused;
    std::printf("P2 reset: %zu pages back, all zero %s; the next session on them (reused %s): %s\n", held.size(),
                zero ? "yes" : "NO - FAIL", reused ? "yes" : "no", same ? "PASS" : "FAIL");
    ok &= zero && same;
  }
  {   // P4
    reset_all();
    const Record x = session(r, 1, longer);
    const bool same = x == ref_long;
    std::printf("P4 %zu ids on sequence 1 (pages %s): %s\n", longer.size(), pages_str(e.pool_pages_of(1)).c_str(),
                same ? "bitwise the one-slot engine" : "DIFFERS");
    ok &= same;
    reset_all();
    print("P5 the same in calls of 2048 ids", session(r, 1, longer, runtime::kKvPage));
    // P6, the served bench's order: a 3-id request on sequence 0 (prefill, 8 lone iterations), then 3,425 ids on 1.
    reset_all();
    print("P6 sequence 0, 3 ids", session(r, 0, std::vector<uint32_t>(longer.begin(), longer.begin() + 3)));
    print("P6 then sequence 1, 3425 ids", session(r, 1, std::vector<uint32_t>(longer.begin(), longer.begin() + 3425),
                                                  runtime::kKvPage));
    // P7: K 7 alone on a further sequence (its 8-row verify list over its pages), as a lone served request drafts.
    reset_all();
    print("P7 sequence 1 at K 7", session(r, 1, prompt, 0, 7));
    // P8: every K from 1 to 7 in turn on sequences 0 and 1 (every verify list M = 2..8, as adaptive K walks them).
    for (uint32_t q : {0u, 1u}) {
      reset_all();
      start(r, q, prompt);
      std::vector<uint32_t> ids;
      for (uint32_t i = 0; i < 21; ++i) iterate(r, ids, 1 + i % 7);
      print(q == 0 ? "P8 sequence 0, K 1..7" : "P8 sequence 1, K 1..7", finish(r, q, ids));
    }
    // P10: K 7 from a 3-id prompt (8-row verifies over the first keys of a page) on sequences 0 and 1.
    for (uint32_t q : {0u, 1u}) {
      reset_all();
      print(q == 0 ? "P10 sequence 0, 3 ids, K 7" : "P10 sequence 1, 3 ids, K 7",
            session(r, q, std::vector<uint32_t>(longer.begin(), longer.begin() + 3), 0, 7));
    }
    // P9: the served bench's requests through cli::SlotsEngine, as the scheduler drives it - a 3-id request on
    // sequence 0 (its chunk, the first-id peek, lone K 7 runs, the run's tail handed back with truncate_to at 8 ids),
    // then 3,425 ids on sequence 1 in the engine's own chunks and its first runs.
    reset_all();
    {
      cli::SlotsEngine se(e, e.model().desc->vocab_used, 7, false, false, runtime::Engine::kMaxRows, true);
      const server::Sampling greedy;
      std::mt19937_64 rng;
      const auto request = [&](uint32_t q, const std::vector<uint32_t>& ids, uint32_t want) {
        se.reset(q);
        for (uint32_t done = 0; done < ids.size();) {
          const uint32_t n = se.prefill_cut(q, done, uint32_t(ids.size()) - done);
          se.prefill({{q, ids.data() + done, n, done + n == ids.size(), false, &greedy}});
          done += n;
        }
        std::vector<uint32_t> out;
        if (const auto first = se.pending_id(q, greedy)) out.push_back(*first);
        const uint32_t start = se.pos(q);
        while (out.size() < want) {
          const std::vector<std::vector<uint32_t>> run = se.step_spec({{q, &greedy, &rng, 7}});
          for (uint32_t id : run.at(0)) out.push_back(id);
        }
        if (out.size() > want) se.truncate_to(q, start + want - 1);
        std::printf("P9 sequence %u: %zu prompt ids, %u ids kept, pos %u\n", q, ids.size(), want, se.pos(q));
      };
      request(0, std::vector<uint32_t>(longer.begin(), longer.begin() + 3), 8);
      request(1, std::vector<uint32_t>(longer.begin(), longer.begin() + 3425), 32);
    }
  }
  std::printf("dflash_slots_test --pool: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

// B8 (header): the batched pass at four blocks of K 7.
bool b8(Rig& r, const std::vector<uint32_t>& prompt, const Record& ref) {
  constexpr uint32_t K = 7, Q = 4;
  l0::Context& ctx = r.ctx;
  for (uint32_t q = 0; q < Q; ++q) {
    start(r, q, prompt);
    std::vector<uint32_t> scratch;
    for (uint32_t i = 0; i < q; ++i) iterate(r, scratch);   // the slots at different positions
  }
  std::vector<l0::Mem> st, kv;
  uint32_t pos[Q], pend[Q];
  for (uint32_t q = 0; q < Q; ++q) {
    r.e->select(q);
    r.e->flush_projection();
    pos[q] = r.e->pos();
    pend[q] = r.e->control()->cur_token[0];
    st.emplace_back(ctx, l0::MemKind::Host, r.e->state_bytes());
    kv.emplace_back(ctx, l0::MemKind::Host, r.e->kv_bytes(4096));
    r.e->save_state(st[q].ptr());
    r.e->save_kv(0, pos[q], kv[q].ptr());
  }
  const auto back = [&] {
    for (uint32_t q = 0; q < Q; ++q) {
      r.e->select(q);
      r.e->load_state(st[q].ptr(), pos[q]);
      r.e->load_kv(0, pos[q], kv[q].ptr());
      r.e->control()->cur_token[0] = pend[q];
      r.e->control()->n_active = 1;
    }
  };
  const auto ring_rows = [&](uint32_t q, uint32_t p, uint32_t n) {   // B5's: every layer's K and V rows
    const runtime::dflash::DFlashBuffers::Session s = r.d->buffers().session(q);
    const model::DFlashDesc& dd = r.d->desc();
    const size_t kvn = dd.kv_n(), slots = dd.window[0];
    std::vector<uint16_t> k(s.ring_k->size() / 2), v(s.ring_v->size() / 2), out;
    r.imm.copy(k.data(), s.ring_k->ptr(), s.ring_k->size());
    r.imm.copy(v.data(), s.ring_v->ptr(), s.ring_v->size());
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t l = 0; l < dd.layers; ++l)
        for (const auto* x : {&k, &v}) {
          const size_t b = (size_t(l) * slots + (p + i) % slots) * kvn;
          out.insert(out.end(), x->begin() + b, x->begin() + b + kvn);
        }
    return out;
  };
  using Out = runtime::dflash::DFlashDraftOut;
  const auto same_rows = [&](const Out& a, const Out& b) {
    if (a.k != b.k || a.anchor != b.anchor || a.pos != b.pos) return false;
    for (uint32_t j = 0; j < K; ++j)
      if (a.ids[j] != b.ids[j] || std::memcmp(a.cand[j], b.cand[j], sizeof a.cand[j]) ||
          std::memcmp(a.unary[j], b.unary[j], sizeof a.unary[j]) || std::memcmp(a.scores[j], b.scores[j], sizeof a.scores[j]))
        return false;
    return true;
  };
  // alone: each slot's own draft (KS 4), verify(7), greedy commit
  Out lone[Q], one[Q];
  uint32_t lone_j[Q];
  std::vector<uint16_t> lone_ring[Q];
  for (uint32_t q = 0; q < Q; ++q) {
    r.e->select(q);
    lone[q] = r.d->draft(*r.e, K, 0.0f, 0);
    r.e->verify(K);
    uint32_t j = 0;
    while (j < K && lone[q].ids[j] == r.e->verify_ids()[j]) ++j;
    const uint32_t p = r.e->pos();
    r.e->commit(j, r.e->verify_ids()[j]);
    lone_j[q] = j;
    lone_ring[q] = ring_rows(q, p, j + 1);
  }
  back();
  for (uint32_t q = 0; q < Q; ++q) {   // each through the 8-row batched list alone
    r.d->draft_rows(*r.e, {{q, K, 0.0f, 0}});
    one[q] = r.d->readback(q);
  }
  r.d->draft_rows(*r.e, {{0, K, 0.0f, 0}, {1, K, 0.0f, 0}, {2, K, 0.0f, 0}, {3, K, 0.0f, 0}});
  bool a = true, b = true;
  for (uint32_t q = 0; q < Q; ++q) {
    const Out& o = r.d->readback(q);
    const bool bit = same_rows(o, one[q]);
    float worst = 0;
    uint32_t unmatched = 0;
    for (uint32_t j = 0; j < K; ++j)
      for (uint32_t c = 0; c < 16; ++c) {
        bool found = false;
        for (uint32_t c2 = 0; c2 < 16 && !found; ++c2)
          if (lone[q].cand[j][c2] == o.cand[j][c]) {
            found = true;
            worst = std::max(worst, std::fabs(o.scores[j][c] - lone[q].scores[j][c2]));
          }
        unmatched += !found;
      }
    const bool drafts = std::equal(o.ids, o.ids + K, lone[q].ids);
    std::printf("B8 slot %u at %u: the 32-row pass %s the 8-row list alone; drafts %s the lone draft's, max |score "
                "diff| %.2e, %u candidates not among the lone ones\n",
                q, pos[q], bit ? "bitwise" : "DIFFERS from", drafts ? "equal" : "differ from", worst, unmatched);
    a &= bit;
    b &= drafts;
  }
  // (c) the 32-row step and its rows projection
  r.e->verify_rows({{0, K}, {1, K}, {2, K}, {3, K}});
  std::vector<uint32_t> ids(r.e->rows_ids(), r.e->rows_ids() + Q * (K + 1));
  bool c = true;
  Out bat[Q];
  for (uint32_t q = 0; q < Q; ++q) bat[q] = r.d->readback(q);
  uint32_t got_j[Q], got_pos[Q];
  for (uint32_t q = 0; q < Q; ++q) {
    const uint32_t* row = ids.data() + q * (K + 1);
    uint32_t j = 0;
    while (j < K && bat[q].ids[j] == row[j]) ++j;
    r.e->select(q);
    got_pos[q] = r.e->pos();
    got_j[q] = j;
    r.e->commit(j, row[j], /*defer_projection=*/true);
  }
  r.e->flush_projection();
  for (uint32_t q = 0; q < Q; ++q) {
    if (got_j[q] != lone_j[q] || !std::equal(bat[q].ids, bat[q].ids + K, lone[q].ids)) {
      std::printf("B8 slot %u: a different acceptance (%u against %u alone) - ring rows not compared\n", q, got_j[q],
                  lone_j[q]);
      continue;
    }
    const std::vector<uint16_t> got = ring_rows(q, got_pos[q], got_j[q] + 1);
    const size_t row_n = got.size() / (got_j[q] + 1);
    double worst = 1;
    for (uint32_t i = 0; i <= got_j[q]; ++i) {
      double ab = 0, aa = 0, bb = 0;
      for (size_t x = 0; x < row_n; ++x) {
        uint32_t ua = uint32_t(got[i * row_n + x]) << 16, ub = uint32_t(lone_ring[q][i * row_n + x]) << 16;
        float fa, fb;
        std::memcpy(&fa, &ua, 4);
        std::memcpy(&fb, &ub, 4);
        ab += double(fa) * fb, aa += double(fa) * fa, bb += double(fb) * fb;
      }
      worst = std::min(worst, ab / std::sqrt(aa * bb));
    }
    std::printf("B8 slot %u: %u rows committed by the 32-row step, the rows projection's ring rows against alone: "
                "worst cos %.6f\n", q, got_j[q] + 1, worst);
    c &= worst >= 0.999;
  }
  // (d) the batched session of the four slots at K `kk` each, timed (the steps only, not the prefill): ids a second
  std::vector<uint32_t> seq[Q];
  const auto run_session = [&](uint32_t kk, uint32_t iters) {
  for (uint32_t q = 0; q < Q; ++q) {
    seq[q].clear();
    start(r, q, prompt);
  }
  r.e->flush_projection();
  const auto t0 = std::chrono::steady_clock::now();
  for (uint32_t it = 0; it < iters; ++it) {
    std::vector<runtime::TapDrafter::DraftRun> runs;
    std::vector<runtime::Engine::Run> vr;
    for (uint32_t q = 0; q < Q; ++q) {
      r.e->select(q);
      const uint32_t k = std::min(kk, r.e->max_verify_k());
      runs.push_back({q, k, 0.0f, 0});
      vr.push_back({q, k});
      seq[q].push_back(r.e->control()->cur_token[0]);
    }
    r.d->draft_rows(*r.e, runs);
    std::vector<std::vector<uint32_t>> drafts;
    for (uint32_t q = 0; q < Q; ++q) drafts.emplace_back(r.d->readback(q).ids, r.d->readback(q).ids + vr[q].k);
    r.e->verify_rows(vr);
    const uint32_t* rid = r.e->rows_ids();
    uint32_t r0 = 0;
    for (uint32_t q = 0; q < Q; ++q) {
      const uint32_t k = vr[q].k;
      uint32_t j = 0;
      while (j < k && drafts[q][j] == rid[r0 + j]) ++j;
      for (uint32_t t = 0; t < j; ++t) seq[q].push_back(drafts[q][t]);
      r.e->select(q);
      r.e->commit(j, rid[r0 + j], /*defer_projection=*/true);
      r0 += k + 1;
    }
  }
  r.e->flush_projection();
  size_t n = 0;
  for (uint32_t q = 0; q < Q; ++q) n += seq[q].size();
  return n / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  run_session(K, kIters);
  for (uint32_t q = 0; q < Q; ++q) {
    size_t n = std::min(seq[q].size(), ref.ids.size()), same = 0;
    while (same < n && seq[q][same] == ref.ids[same]) ++same;
    std::printf("B8 session slot %u: %zu ids in %u steps, the first %zu equal A's%s\n", q, seq[q].size(), kIters,
                same, same == n ? " (all)" : "");
  }
  // T8 (timing, informational, busy box): the four slots' greedy session, 96 steps, at K 1 each (the 8-row step the
  // old deal gives four requests) against K 7 each (32 rows) - every id the engine keeps, a second, A B A B.
  double t1[2] = {0, 0}, t7[2] = {0, 0};
  for (int round = 0; round < 2; ++round) {
    t1[round] = run_session(1, 96);
    t7[round] = run_session(K, 96);
  }
  std::printf("T8: four slots, 96 greedy steps: K 1 each (8 rows) %.1f / %.1f ids/s, K 7 each (32 rows) %.1f / %.1f "
              "ids/s: %.3fx\n", t1[0], t1[1], t7[0], t7[1], (t7[0] + t7[1]) / (t1[0] + t1[1]));
  std::printf("B8: (a) %s, (b) %s, (c) %s\n", a ? "ok" : "WRONG", b ? "ok" : "WRONG", c ? "ok" : "WRONG");
  return a && b && c;
}

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
  if (argc > 3 && std::string(argv[3]) == "--pool")
    return pool_main(ctx, snap, dsnap, prompt, std::vector<uint32_t>(all.begin(), all.begin() + 5000));
  bool ok = true;

  Record ref;
  {   // A
    Rig r(ctx, snap, dsnap, {kPoolPages * runtime::kKvPage});
    ref = session(r, 0, prompt);
    print("A  one slot", ref);
  }

  Rig r(ctx, snap, dsnap, std::vector<uint32_t>(kSeqs, kCap), kPoolPages);
  std::printf("%s\n", r.e->memory_line().c_str());
  // B1
  for (uint32_t q = 0; q < kSeqs; ++q) {
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

  {   // B3 (plan 40a Task 2: a further slot keeps its rows pending, so it drafts the drafter's whole block)
    r.e->select(1);
    const uint32_t mk = r.e->max_verify_k();
    bool refused = false;
    try {
      r.e->verify(8);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    const bool b3 = mk == 7 && r.e->max_draft() == 7 && refused;
    std::printf("B3 slot 1: max_verify_k %u, max_draft %u, verify(8) %s: %s\n", mk, r.e->max_draft(),
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

  // --- B5 / B6: the batched draft and a row step, against each slot alone from the same state ------------------
  {
    start(r, 1, prompt);
    start(r, 2, prompt);
    std::vector<uint32_t> scratch;
    r.e->select(1);
    iterate(r, scratch);   // the two slots at different positions
    const auto snap = [&](uint32_t q, l0::Mem& st, l0::Mem& kv, uint32_t& pos, uint32_t& pend) {
      r.e->select(q);
      r.e->flush_projection();
      pos = r.e->pos();
      pend = r.e->control()->cur_token[0];
      r.e->save_state(st.ptr());
      r.e->save_kv(0, pos, kv.ptr());
    };
    const auto back = [&](uint32_t q, l0::Mem& st, l0::Mem& kv, uint32_t pos, uint32_t pend) {
      r.e->select(q);
      r.e->load_state(st.ptr(), pos);
      r.e->load_kv(0, pos, kv.ptr());
      r.e->control()->cur_token[0] = pend;
      r.e->control()->n_active = 1;
    };
    l0::Mem st1(ctx, l0::MemKind::Host, r.e->state_bytes()), st2(ctx, l0::MemKind::Host, r.e->state_bytes());
    l0::Mem kv1(ctx, l0::MemKind::Host, r.e->kv_bytes(4096)), kv2(ctx, l0::MemKind::Host, r.e->kv_bytes(4096));
    uint32_t p1, p2, x1, x2;
    snap(1, st1, kv1, p1, x1);
    snap(2, st2, kv2, p2, x2);
    // the ring rows a commit of n rows at pos stored: every layer's K and V rows
    const auto ring_rows = [&](uint32_t q, uint32_t pos, uint32_t n) {
      const runtime::dflash::DFlashBuffers::Session s = r.d->buffers().session(q);
      const model::DFlashDesc& dd = r.d->desc();
      const size_t kvn = dd.kv_n(), slots = dd.window[0];
      std::vector<uint16_t> k(s.ring_k->size() / 2), v(s.ring_v->size() / 2), out;
      r.imm.copy(k.data(), s.ring_k->ptr(), s.ring_k->size());
      r.imm.copy(v.data(), s.ring_v->ptr(), s.ring_v->size());
      for (uint32_t i = 0; i < n; ++i)
        for (uint32_t l = 0; l < dd.layers; ++l)
          for (const auto* x : {&k, &v}) {
            const size_t b = (size_t(l) * slots + (pos + i) % slots) * kvn;
            out.insert(out.end(), x->begin() + b, x->begin() + b + kvn);
          }
      return out;
    };
    // alone: each slot's own draft, verify(3), greedy commit
    std::vector<uint32_t> lone_d[3], lone_j(3);
    std::vector<float> lone_s[3];
    std::vector<uint32_t> lone_c[3];
    std::vector<uint16_t> lone_ring[3];
    for (uint32_t q : {1u, 2u}) {
      r.e->select(q);
      const runtime::dflash::DFlashDraftOut& o = r.d->draft(*r.e, kK, 0.0f, 0);
      lone_d[q].assign(o.ids, o.ids + kK);
      lone_s[q].assign(reinterpret_cast<const float*>(o.scores), reinterpret_cast<const float*>(o.scores) + kK * 16);
      lone_c[q].assign(reinterpret_cast<const uint32_t*>(o.cand), reinterpret_cast<const uint32_t*>(o.cand) + kK * 16);
      r.e->verify(kK);
      uint32_t j = 0;
      while (j < kK && lone_d[q][j] == r.e->verify_ids()[j]) ++j;
      const uint32_t pos = r.e->pos();
      r.e->commit(j, r.e->verify_ids()[j]);
      lone_j[q] = j;
      lone_ring[q] = ring_rows(q, pos, j + 1);
    }
    back(1, st1, kv1, p1, x1);
    back(2, st2, kv2, p2, x2);
    r.d->draft_rows(*r.e, {{1, kK, 0.0f, 0}, {2, kK, 0.0f, 0}});
    bool b5 = true;
    std::vector<uint32_t> bat_d[3];
    for (uint32_t q : {1u, 2u}) {
      const runtime::dflash::DFlashDraftOut& o = r.d->readback(q);
      bat_d[q].assign(o.ids, o.ids + kK);
      // each row's candidates by id: a near-tie may order them differently (KS 1 against KS 4 sums)
      const float* sc = reinterpret_cast<const float*>(o.scores);
      const uint32_t* cd = reinterpret_cast<const uint32_t*>(o.cand);
      float worst = 0;
      uint32_t unmatched = 0;
      for (uint32_t j = 0; j < kK; ++j)
        for (uint32_t c = 0; c < 16; ++c) {
          bool found = false;
          for (uint32_t c2 = 0; c2 < 16 && !found; ++c2)
            if (lone_c[q][j * 16 + c2] == cd[j * 16 + c]) {
              found = true;
              worst = std::max(worst, std::fabs(sc[j * 16 + c] - lone_s[q][j * 16 + c2]));
            }
          unmatched += !found;
        }
      std::printf("B5 slot %u: batched drafts %u %u %u, alone %u %u %u; candidates matched by id: max |score diff| "
                  "%.2e, %u not among the lone ones\n",
                  q, bat_d[q][0], bat_d[q][1], bat_d[q][2], lone_d[q][0], lone_d[q][1], lone_d[q][2], worst, unmatched);
      b5 &= bat_d[q] == lone_d[q] && worst < 0.12f && unmatched <= 2;   // KS 1 against KS 4: the judged form
    }
    std::printf("B5: %s\n", b5 ? "ok" : "WRONG");
    ok &= b5;
    r.e->verify_rows({{1, kK}, {2, kK}});
    const uint32_t* ids = r.e->rows_ids();
    bool b6 = true;
    for (uint32_t q : {1u, 2u}) {
      const uint32_t* row = ids + (q == 1 ? 0 : kK + 1);
      uint32_t j = 0;
      while (j < kK && bat_d[q][j] == row[j]) ++j;
      r.e->select(q);
      const uint32_t pos = r.e->pos();
      r.e->commit(j, row[j], /*defer_projection=*/true);
      r.e->flush_projection();
      if (j != lone_j[q] || bat_d[q] != lone_d[q]) {
        std::printf("B6 slot %u: a different acceptance (%u against %u alone) - ring rows not compared\n", q, j,
                    lone_j[q]);
        continue;
      }
      const std::vector<uint16_t> got = ring_rows(q, pos, j + 1);
      const size_t row_n = got.size() / (j + 1);
      double worst = 1;
      for (uint32_t i = 0; i <= j; ++i) {
        double ab = 0, aa = 0, bb = 0;
        for (size_t x = 0; x < row_n; ++x) {
          uint32_t ua = uint32_t(got[i * row_n + x]) << 16, ub = uint32_t(lone_ring[q][i * row_n + x]) << 16;
          float fa, fb;
          std::memcpy(&fa, &ua, 4);
          std::memcpy(&fb, &ub, 4);
          ab += double(fa) * fb, aa += double(fa) * fa, bb += double(fb) * fb;
        }
        worst = std::min(worst, ab / std::sqrt(aa * bb));
      }
      std::printf("B6 slot %u: %u rows committed, the rows projection's ring rows against alone: worst cos %.6f\n", q,
                  j + 1, worst);
      b6 &= worst >= 0.999;
    }
    std::printf("B6: %s\n", b6 ? "ok" : "WRONG");
    ok &= b6;
  }

  {   // B7: the batched session of slots 1 and 2 against A
    start(r, 1, prompt);
    start(r, 2, prompt);
    std::vector<uint32_t> a, b;
    for (uint32_t it = 0; it < kIters; ++it) {
      std::vector<runtime::TapDrafter::DraftRun> runs;
      std::vector<runtime::Engine::Run> vr;
      for (uint32_t q : {1u, 2u}) {
        r.e->select(q);
        const uint32_t k = std::min(kK, r.e->max_verify_k());
        runs.push_back({q, k, 0.0f, 0});
        vr.push_back({q, k});
        (q == 1 ? a : b).push_back(r.e->control()->cur_token[0]);
      }
      r.d->draft_rows(*r.e, runs);
      std::vector<std::vector<uint32_t>> drafts;
      for (uint32_t q : {1u, 2u}) {
        const runtime::dflash::DFlashDraftOut& o = r.d->readback(q);
        drafts.emplace_back(o.ids, o.ids + o.k);
      }
      r.e->verify_rows(vr);
      const uint32_t* ids = r.e->rows_ids();
      uint32_t r0 = 0;
      for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t q = i + 1, k = vr[i].k;
        uint32_t j = 0;
        while (j < k && drafts[i][j] == ids[r0 + j]) ++j;
        for (uint32_t t = 0; t < j; ++t) (q == 1 ? a : b).push_back(drafts[i][t]);
        r.e->select(q);
        r.e->commit(j, ids[r0 + j], /*defer_projection=*/true);
        r0 += k + 1;
      }
    }
    r.e->flush_projection();
    for (const auto* x : {&a, &b}) {
      size_t n = std::min(x->size(), ref.ids.size()), same = 0;
      while (same < n && (*x)[same] == ref.ids[same]) ++same;
      std::printf("B7 slot %u: %zu ids, the first %zu equal A's%s\n", x == &a ? 1u : 2u, x->size(), same,
                  same == n ? " (all)" : "");
    }
  }

  const bool wide = r.d->draft_rows_max() >= 32;
  if (wide) ok &= b8(r, prompt, ref);

  {   // T
    for (uint32_t q = 0; q < kSeqs; ++q) start(r, q, prompt);
    std::vector<double> lone, two, four, four7;
    const auto ms = [](auto f) {
      const auto t0 = std::chrono::steady_clock::now();
      f();
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    for (int round = 0; round < 16; ++round) {
      const double a = ms([&] {
        r.e->select(0);
        r.d->draft(*r.e, 7, 0.0f, 0);
      });
      const double b = ms([&] { r.d->draft_rows(*r.e, {{0, 3, 0.0f, 0}, {1, 3, 0.0f, 0}}); });
      const double c = ms([&] { r.d->draft_rows(*r.e, {{0, 1, 0.0f, 0}, {1, 1, 0.0f, 0}, {2, 1, 0.0f, 0}, {3, 1, 0.0f, 0}}); });
      const double w = !wide ? 0.0 : ms([&] {
        r.d->draft_rows(*r.e, {{0, 7, 0.0f, 0}, {1, 7, 0.0f, 0}, {2, 7, 0.0f, 0}, {3, 7, 0.0f, 0}});
      });
      if (round == 0) continue;   // warm-up
      lone.push_back(a), two.push_back(b), four.push_back(c), four7.push_back(w);
    }
    const auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    std::printf("T: the draft pass, one slot K 7 %.3f ms; batched 2 x K 3 %.3f ms (%.3f plain), 4 x K 1 %.3f ms (%.3f "
                "plain) - busy box\n",
                med(lone), med(two), 0.217 * med(two) / med(lone), med(four), 0.217 * med(four) / med(lone));
    if (wide)
      std::printf("T: batched 4 x K 7 (32 rows) %.3f ms (%.3f plain, %.2fx the lone K 7 draft)\n", med(four7),
                  0.217 * med(four7) / med(lone), med(four7) / med(lone));
  }

  std::printf("dflash_slots_test: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
