// Spec 13c (plan 13c Tasks 2-3): runtime::Engine's sequence slots on the card - several requests on one B70.
//
// One engine: Qwen3.8 with its MTP head and the int8 lm_head (the served form), slots 16384 + 3 x 8192 (40960
// positions), the KV form B70_KV_CACHE's (slots_kv8_test: int8). Prompts: long32k / code / prose / long, cut per slot.
//
//   S1  a slot alone is today's engine: the same session (prefill, 32 plain steps, 8 MTP iterations at K = 3) in slot 0
//       and in slot 2 gives the same ids and every step's logits bitwise (each slot's lists are today's kernels over
//       its own state and KV range).
//   S2  isolation (Review Focus 3-4): a slot's prefill, decode, reset, and a snapshot restored into it leave every
//       other slot's bytes - its GDN states (the speculative ones too), conv ring, hh rows, Control and the KV rows it
//       holds (main and head) - byte-identical; save_state / save_kv of slot 1 loaded into slot 3 continues exactly as
//       slot 1 does.
//   S3  B1 for the row kernels, bitwise, at mixed K and mixed depth (spec 13a's check through the engine): slots 0 (deep,
//       K = 3), 1 (K = 1) and 2 (K = 1) in ONE verify_rows step (8 rows) against each run alone (verify_rows of its own
//       rows): every row's logits and id, the GDN state each row wrote, the KV rows it wrote (main and head); and B3 -
//       nothing else of any slot moved (slot 3 whole, the other slots' sources and KV windows), and commit after the
//       batched step leaves each slot as after its run alone.
//   S4  B1 by the f32 judge (spec 13 §11.3): four requests decoded together (32 batched plain steps, then 8 batched
//       greedy MTP iterations at K = 1 each - 8 rows) against each alone (the slot's own lists): identical, or
//       spec_judge::streams' rule (tag slots_<kv>_<n>: the int8 head's references) - a divergence whose context has no
//       f32 reference yet AWAITS it (exit 77).
//   S5  determinism: the S4 batched run twice from the same state gives the same ids.
//   `--mtp off` (slots_nomtp_test): the engine without the head - one GDN state a slot (SlotBuffers' states): S1 and
//   S2 with plain steps, S3's isolation over a plain four-row step, S4 / S5 plain rows only; no S6.
//   S6  the batched drafts (Engine::draft_rows: one head pass per draft index for several slots): slots 0..2 at K 3 /
//       2 / 1 against each slot's own draft lists from the same state - the drafts printed (equal unless a near-tie:
//       the lm_head's small-batch GEMM rounds differently from the M = 1 GEMV), and B3 for drafts: slot 3 (not
//       drafting) and every slot's state and committed KV byte-identical (a draft writes only its slot's head KV
//       rows past its last committed position).
//   S7  plan 19v3b Task 0, the slots' KV in pages: with the identity page table and with a scattered one (slot 1's
//       pages interleaved with slot 0's, slot 3's reversed, slot 2's after them) - S1's session alone in every slot
//       (a prompt over three pages; ids and every step's logits), the four slots batched (verify_rows, with MTP K = 1
//       each: ids), every slot's KV rows (hashed in position order) and slot 1's snapshot image (save_kv): all
//       bitwise the same under both tables, and every slot's lone session the same as slot 0's. S1 prints its
//       session's digest - the same line on main (contiguous slots) is the "bitwise to main" check.
//   `--pool` (slots_pool_test, slots_pool_kv8_test; spec 31 / plan 31 Task 5): the fixed-slot engine's lone session
//   (slot 0: a 5000-id prompt over three pages, 16 plain steps, 4 MTP iterations at K = 3) is the reference; then an
//   engine of 4 sequences sharing a KV page pool (20 pages - the fixed engine's 40960 positions - a request up to
//   36864):
//   P1  the same session alone gives the reference's ids and every step's logits bitwise with its pages handed out
//       in order, reversed, and interleaved with a second request that grows beside it (a step of it after each).
//   P2  the session's further growth (a prefill over another page, rejected drafts) and its reset leave the second
//       request's state and KV bytes unchanged; the reset gives every page back, each reads all zero (main and head,
//       rows and scales), and the next request on those pages gives the reference again.
//   P3  a request grows past 32768 while the others are empty, the same ids in sequence 3 as in sequence 0.
//   P4  with no page free, max_verify_k() stops at the page's end and a verify that needs the next page is refused
//       with nothing of the sequence written; it then continues as an uninterrupted request does.
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "contract_death.h"
#include "golden_common.h"
#include "greedy_tie.h"
#include "l0/cmdlist.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "spec_judge.h"

namespace {
using runtime::Engine;
constexpr size_t V = model::Qwen35::kVocab;
const std::vector<uint32_t> kSlots = {16384, 8192, 8192, 8192};

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

struct Ctx {
  l0::Context& ctx;
  Engine& e;
  l0::CmdList imm;
  l0::Mem host;   // pinned scratch for readbacks
  uint32_t VU;
  Ctx(l0::Context& c, Engine& en)
      : ctx(c), e(en), imm(l0::CmdList::immediate(c)), host(c, l0::MemKind::Host, size_t(160) << 20),
        VU(en.model().desc->vocab_used) {}

  uint64_t dev_hash(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const size_t chunk = host.size();
    for (size_t off = 0; off < n; off += chunk) {
      const size_t m = std::min(chunk, n - off);
      imm.copy(host.ptr(), static_cast<const uint8_t*>(p) + off, m);
      h = fnv(host.ptr(), m, h);
    }
    return h;
  }
  size_t state_bytes() const { return e.buffers().gdn_state.size(); }
  // Every byte of slot q's persistent state, its KV rows [0, npos) of every layer (main and head) included.
  uint64_t slot_hash(uint32_t q, uint32_t npos, bool with_control = true) {
    const Engine::SlotView v = e.slot_view(q);
    uint64_t h = with_control ? fnv(v.control, sizeof(runtime::Control)) : 1469598103934665603ull;
    h = dev_hash(v.state0, state_bytes(), h);
    if (v.spec) h = dev_hash(v.spec, state_bytes() * (v.states - 1), h);
    h = dev_hash(v.ring, e.buffers().conv_ring.size(), h);
    if (v.hh) h = dev_hash(v.hh, size_t(runtime::DecodeScratchDims::kM + 1) * e.model().desc->hidden * 2, h);
    return kv_hash(q, 0, npos, h);
  }
  // KV rows [p, p + n) of slot q, every layer, main and head, rows and scales - in position order through the page
  // table (plan 19v3b Task 0: Engine::kv_row), so the hash is the same whatever the table.
  uint64_t kv_hash(uint32_t q, uint32_t p, uint32_t n, uint64_t h = 1469598103934665603ull) {
    std::vector<std::pair<uint32_t, uint32_t>> runs;   // (cache row, rows), one page at most each
    for (uint32_t i = p; i < p + n;) {
      const uint32_t m = std::min(p + n, (i / runtime::kKvPage + 1) * runtime::kKvPage) - i;
      runs.push_back({e.kv_row(q, i), m});
      i += m;
    }
    const auto one = [&](const runtime::KvLayout& L, void* k, void* vv) {
      for (uint32_t l = 0; l < L.layers; ++l) {
        const runtime::KvLayer x = L.layer(k, vv, l);
        for (int t = 0; t < 4; ++t)
          for (const auto& [row, m] : runs) {
            const runtime::KvLayer y = L.at_base(x, row);
            void* ptr = t == 0 ? y.k : t == 1 ? y.v : t == 2 ? y.ks : y.vs;
            if (ptr) h = dev_hash(ptr, size_t(m) * (t < 2 ? L.row_bytes() : L.scale_row_bytes()), h);
          }
      }
    };
    one(e.buffers().kv_lay, e.buffers().kv_k.ptr(), e.buffers().kv_v.ptr());
    if (e.mtp_buffers()) one(e.mtp_buffers()->kv_lay, e.mtp_buffers()->kv_k.ptr(), e.mtp_buffers()->kv_v.ptr());
    return h;
  }
  std::vector<float> rows(const float* dev, uint32_t first, uint32_t n) {
    std::vector<float> v(size_t(n) * V);
    imm.copy(v.data(), dev + size_t(first) * V, v.size() * 4);
    return v;
  }
};

std::vector<uint32_t> cut(const std::vector<uint32_t>& src, size_t from, size_t n) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < n; ++i) out.push_back(src[(from + i) % src.size()]);
  return out;
}

// One greedy step of the selected slot through its own lists: k = 0 plain (verify(0) + commit; without the head the
// plain list, generate(1)), else draft + verify + the longest agreeing prefix. Returns the ids emitted; `logits` gets
// every verify row's logits hash.
std::vector<uint32_t> lone_step(Ctx& c, uint32_t k, std::vector<uint64_t>* logits = nullptr) {
  Engine& e = c.e;
  std::vector<uint32_t> out{e.control()->cur_token[0]};
  if (!e.mtp()) {
    e.generate(1);
    if (logits) {
      const std::vector<float> r = c.rows(e.buffers().logits.as<float>(), 0, 1);
      logits->push_back(fnv(r.data(), r.size() * 4));
    }
    return out;
  }
  if (k > 0) e.draft(k);
  e.verify(k);
  if (logits) {
    const std::vector<float> r = c.rows(e.verify_logits_device(), 0, k + 1);
    logits->push_back(fnv(r.data(), r.size() * 4));
  }
  uint32_t j = 0;
  while (j < k && e.draft_ids()[j] == e.verify_ids()[j]) ++j;
  for (uint32_t i = 0; i < j; ++i) out.push_back(e.draft_ids()[i]);
  e.commit(j, e.verify_ids()[j]);
  return out;
}

// S1.
bool s1(Ctx& c, const std::vector<uint32_t>& prompt) {
  Engine& e = c.e;
  auto run = [&](uint32_t q, std::vector<uint64_t>& lg) {
    e.select(q);
    e.reset();
    e.prefill(prompt);
    std::vector<uint32_t> ids;
    for (int i = 0; i < 32; ++i)
      for (uint32_t id : lone_step(c, 0, &lg)) ids.push_back(id);
    for (int i = 0; i < 8; ++i)
      for (uint32_t id : lone_step(c, e.mtp() ? 3 : 0, &lg)) ids.push_back(id);
    return ids;
  };
  std::vector<uint64_t> l0, l2;
  const std::vector<uint32_t> a = run(0, l0), b = run(2, l2);
  const bool ok = a == b && l0 == l2;
  std::printf("S1 slot 0 vs slot 2, the same session alone (%zu ids, %zu verify steps): %s\n", a.size(), l0.size(),
              ok ? "PASS (ids and every step's logits bitwise)" : "FAIL");
  std::printf("S1 digest: ids %016llx logits %016llx\n", (unsigned long long)fnv(a.data(), a.size() * 4),
              (unsigned long long)fnv(l0.data(), l0.size() * 8));
  return ok;
}

// S2.
bool s2(Ctx& c, const std::vector<uint32_t>& p1, const std::vector<uint32_t>& p2, const std::vector<uint32_t>& p3) {
  Engine& e = c.e;
  bool ok = true;
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    e.reset();
  }
  e.select(1);
  e.prefill(p1);
  for (int i = 0; i < 6; ++i) lone_step(c, i % 2 && e.mtp() ? 3 : 0);   // a rejected draft or two: speculative states
  const uint32_t n1 = e.pos() + 16;
  auto others = [&](uint32_t busy) {
    std::vector<uint64_t> h;
    for (uint32_t q = 0; q < e.slots(); ++q) h.push_back(q == busy ? 0 : c.slot_hash(q, q == 1 ? n1 : 64));
    return h;
  };
  // A prefill and a decode of slot 2.
  std::vector<uint64_t> before = others(2);
  e.select(2);
  e.prefill(p2);
  for (int i = 0; i < 6; ++i) lone_step(c, i % 2 && e.mtp() ? 3 : 0);
  bool r = others(2) == before;
  std::printf("S2 slot 2's prefill + decode: the other slots %s\n", r ? "byte-identical" : "MOVED - FAIL");
  ok &= r;
  // A reset of slot 3 (after a prefill).
  e.select(3);
  e.prefill(p3);
  before = others(3);
  e.select(3);
  e.reset();
  r = others(3) == before && e.pos() == 0;
  std::printf("S2 reset(slot 3): the other slots %s, its pos %u\n", r ? "byte-identical" : "MOVED - FAIL", e.pos());
  ok &= r;
  // Slot 1's snapshot into slot 3.
  e.select(1);
  const uint32_t p = e.pos();
  l0::Mem st(c.ctx, l0::MemKind::Host, e.state_bytes()), kv(c.ctx, l0::MemKind::Host, e.kv_bytes(p));
  e.save_state(st.ptr());
  e.save_kv(0, p, kv.ptr());
  const uint32_t pending = e.control()->cur_token[0];
  before = others(3);
  e.select(3);
  e.load_state(st.ptr(), p);
  e.load_kv(0, p, kv.ptr());
  e.control()->cur_token[0] = pending;   // cur_token is not snapshotted (Engine::load_state)
  r = others(3) == before;
  std::vector<uint32_t> a, b;
  e.select(1);
  for (int i = 0; i < 16; ++i)
    for (uint32_t id : lone_step(c, i % 2 && e.mtp() ? 2 : 0)) a.push_back(id);
  e.select(3);
  for (int i = 0; i < 16; ++i)
    for (uint32_t id : lone_step(c, i % 2 && e.mtp() ? 2 : 0)) b.push_back(id);
  std::printf("S2 slot 1's snapshot loaded into slot 3: the other slots %s; slot 3 continues as slot 1: %s\n",
              r ? "byte-identical" : "MOVED - FAIL", a == b ? "yes (ids bitwise)" : "NO - FAIL");
  return ok && r && a == b;
}

// S3.
struct RowOut {
  std::vector<float> logits;
  uint32_t id = 0;
  uint64_t dst = 0, kv = 0;
};
bool s3_plain(Ctx& c, const std::vector<std::vector<uint32_t>>& prompts) {
  // Without the head: one plain verify_rows step of slots 0..2 leaves slot 3 and every slot's committed KV untouched,
  // and its rows' ids are what the slots' own commits take.
  Engine& e = c.e;
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    e.reset();
    e.prefill(prompts[q]);
  }
  std::vector<uint32_t> pos;
  std::vector<uint64_t> kv;
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    pos.push_back(e.pos());
    kv.push_back(c.kv_hash(q, 0, e.pos()));
  }
  const uint64_t slot3 = c.slot_hash(3, pos[3] + 16);
  e.verify_rows({{0, 0}, {1, 0}, {2, 0}});
  bool ok = c.slot_hash(3, pos[3] + 16) == slot3;
  for (uint32_t q = 0; q < 3; ++q) {
    ok = ok && c.kv_hash(q, 0, pos[q]) == kv[q];
    e.select(q);
    e.commit(0, e.rows_ids()[q]);
    ok = ok && e.pos() == pos[q] + 1 && e.control()->cur_token[0] == e.rows_ids()[q];
  }
  std::printf("S3 B3 (isolation, a plain three-row step without the head): %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool s3(Ctx& c, const std::vector<uint32_t>& deep, const std::vector<uint32_t>& p1, const std::vector<uint32_t>& p2,
        const std::vector<uint32_t>& p3) {
  Engine& e = c.e;
  if (!e.mtp()) return s3_plain(c, {deep, p1, p2, p3});
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    e.reset();
  }
  const std::vector<std::vector<uint32_t>> prompts = {deep, p1, p2, p3};
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    e.prefill(prompts[q]);
  }
  const std::vector<uint32_t> ks = {3, 1, 1};
  // Every slot's state before the step (live state, ring, hh row 0), its pending id and its drafts.
  std::vector<l0::Mem> st;
  std::vector<std::vector<uint32_t>> drafts(3);
  std::vector<uint32_t> pend(3), pos(3);
  for (uint32_t q = 0; q < 3; ++q) {
    e.select(q);
    e.draft(ks[q]);
    drafts[q] = e.draft_ids();
    pend[q] = e.control()->cur_token[0];
    pos[q] = e.pos();
    st.emplace_back(c.ctx, l0::MemKind::Host, e.state_bytes());
    e.save_state(st.back().ptr());
  }
  const uint32_t n3 = uint32_t(p3.size()) + 16;
  const uint64_t slot3 = c.slot_hash(3, n3);
  auto restore = [&](uint32_t q) {
    e.select(q);
    e.load_state(st[q].ptr(), pos[q]);
    e.control()->cur_token[0] = pend[q];
    for (uint32_t i = 0; i < ks[q]; ++i) e.control()->cur_token[1 + i] = drafts[q][i];
  };
  // The rows each run writes: its GDN states (indices (live + j) % 4 of its group; live is 0 after load_state) and
  // its KV rows at pos .. pos + k (the head's at pos - 1 ..).
  auto outputs = [&](const std::vector<uint32_t>& slots, uint32_t first_row) {
    std::vector<RowOut> o;
    uint32_t m = first_row;
    for (uint32_t q : slots) {
      const Engine::SlotView v = e.slot_view(q);
      for (uint32_t j = 0; j <= ks[q]; ++j, ++m) {
        RowOut r;
        r.logits = c.rows(e.rows_logits_device(), m, 1);
        r.id = e.rows_ids()[m];
        r.dst = c.dev_hash(j == 0 ? v.state0 : v.spec + size_t(j - 1) * c.state_bytes(), c.state_bytes());
        r.kv = c.kv_hash(q, pos[q] - 1 + j, 2);
        o.push_back(r);
      }
    }
    return o;
  };
  // KV rows the step must not write: [pos - 9, pos - 1) and [pos + 8, pos + 16) (it writes the main rows pos .. pos + k
  // and the head's pos - 1 .. pos + k - 1).
  auto window = [&](uint32_t q) { return c.kv_hash(q, pos[q] - 9, 8) ^ (c.kv_hash(q, pos[q] + 8, 8) * 31); };
  for (uint32_t q = 0; q < 3; ++q) restore(q);
  std::vector<uint64_t> win0;
  for (uint32_t q = 0; q < 3; ++q) win0.push_back(window(q));
  e.verify_rows({{0, 3}, {1, 1}, {2, 1}});
  const std::vector<RowOut> batch = outputs({0, 1, 2}, 0);
  bool b3 = c.slot_hash(3, n3) == slot3;
  for (uint32_t q = 0; q < 3; ++q) b3 = b3 && window(q) == win0[q];
  // Commit after the batched step: each slot's state after it (as the run alone leaves it, below).
  std::vector<uint64_t> after_batch;
  for (uint32_t q = 0; q < 3; ++q) {
    e.select(q);
    e.commit(1, e.rows_ids()[q == 0 ? 1 : q == 1 ? 5 : 7]);
    after_batch.push_back(c.slot_hash(q, pos[q] + 8));
  }
  // Each run alone.
  bool b1 = true;
  std::string why;
  uint32_t m0 = 0;
  for (uint32_t q = 0; q < 3; ++q) {
    for (uint32_t r = 0; r < 3; ++r) restore(r);
    e.verify_rows({{q, ks[q]}});
    const std::vector<RowOut> alone = outputs({q}, 0);
    for (uint32_t j = 0; j <= ks[q]; ++j) {
      const RowOut &x = batch[m0 + j], &y = alone[j];
      const std::string w = " slot" + std::to_string(q) + ".row" + std::to_string(j);
      if (std::memcmp(x.logits.data(), y.logits.data(), V * 4) != 0) why += w + ":logits";
      if (x.id != y.id) why += w + ":id";
      if (x.dst != y.dst) why += w + ":gdn-state";
      if (x.kv != y.kv) why += w + ":kv";
    }
    e.select(q);
    e.commit(1, e.rows_ids()[1]);
    if (c.slot_hash(q, pos[q] + 8) != after_batch[q]) why += " slot" + std::to_string(q) + ":commit";
    m0 += ks[q] + 1;
  }
  b1 = why.empty();
  std::printf("S3 B1 (bitwise, the row kernels) 1x4+2x2 at %u / %u / %u: %s%s\n", pos[0], pos[1], pos[2],
              b1 ? "PASS (logits, ids, GDN states, KV rows and the commit equal each run alone)" : "FAIL", why.c_str());
  std::printf("S3 B3 (isolation): %s\n", b3 ? "PASS (slot 3 and every slot's KV window byte-identical)" : "FAIL");
  return b1 && b3;
}

// S6.
bool s6(Ctx& c, const std::vector<std::vector<uint32_t>>& prompts) {
  Engine& e = c.e;
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    e.reset();
    e.prefill(prompts[q]);
  }
  const std::vector<uint32_t> ks = {3, 2, 1};
  std::vector<uint64_t> before, committed;
  std::vector<uint32_t> pos;
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    pos.push_back(e.pos());
    before.push_back(c.slot_hash(q, e.pos()));                 // KV rows [0, pos), every state, the Control
    committed.push_back(c.slot_hash(q, e.pos() - 1, false));   // the same without the Control and row pos - 1
  }
  // The head KV rows a draft writes (pos - 1 ...) are not committed state, and the slot's own drafts land in its
  // Control's cur_token[1..]: compare those slots' hashes over [0, pos - 1) and without the Control after the drafts.
  e.draft_rows({{0, ks[0]}, {1, ks[1]}, {2, ks[2]}});
  std::vector<std::vector<uint32_t>> rows(3);
  for (uint32_t q = 0; q < 3; ++q) {
    e.select(q);
    rows[q] = e.draft_ids();
  }
  bool iso = c.slot_hash(3, pos[3]) == before[3];
  for (uint32_t q = 0; q < 3; ++q) iso = iso && c.slot_hash(q, pos[q] - 1, false) == committed[q];
  std::vector<std::vector<uint32_t>> own(3);
  size_t same = 0, total = 0;
  for (uint32_t q = 0; q < 3; ++q) {
    e.select(q);
    e.draft(ks[q]);
    own[q] = e.draft_ids();
    iso = iso && rows[q].size() == ks[q];
    for (uint32_t i = 0; i < ks[q] && i < rows[q].size(); ++i) {
      same += rows[q][i] == own[q][i];
      ++total;
    }
  }
  std::printf("S6 draft_rows (K 3 / 2 / 1) against each slot's own draft lists: %zu / %zu drafts equal; B3 for drafts: "
              "%s\n", same, total,
              iso ? "PASS (slot 3 byte-identical; slots 0..2's states and committed KV too; every run its K drafts)"
                  : "FAIL");
  return iso;
}

// S7 (plan 19v3b Task 0).
bool s7(Ctx& c, const std::vector<uint32_t>& prompt, const std::vector<std::vector<uint32_t>>& prompts) {
  Engine& e = c.e;
  const uint32_t S = e.slots();
  const std::vector<uint32_t> identity = e.kv_page_table();
  // The scattered table: the physical pages handed out in the order slot 0 / slot 1 alternating while both have pages
  // left, then slot 2's, then slot 3's last page first.
  std::vector<std::vector<uint32_t>> virt(S);   // each slot's virtual pages
  for (uint32_t q = 0; q < S; ++q)
    for (uint32_t i = 0; i < runtime::kv_pages(e.slot_len(q)); ++i) virt[q].push_back(e.slot_base(q) / runtime::kKvPage + i);
  std::vector<uint32_t> order;   // virtual pages in physical order
  for (size_t i = 0; i < std::max(virt[0].size(), virt[1].size()); ++i) {
    if (i < virt[0].size()) order.push_back(virt[0][i]);
    if (i < virt[1].size()) order.push_back(virt[1][i]);
  }
  for (uint32_t q = 2; q < S; ++q)
    for (size_t i = 0; i < virt[q].size(); ++i) order.push_back(q == 3 ? virt[q][virt[q].size() - 1 - i] : virt[q][i]);
  CHECK_EQ(order.size(), identity.size());
  std::vector<uint32_t> scattered(identity.size());
  for (uint32_t phys = 0; phys < order.size(); ++phys) scattered[order[phys]] = phys;
  struct Out {
    std::vector<std::vector<uint32_t>> ids;   // [slot] the lone session's, then [S + slot] the batched run's
    std::vector<std::vector<uint64_t>> logits;
    std::vector<uint64_t> kv;                 // [slot] after its lone session, then [S + slot] after the batched run
    uint64_t snapshot = 0;                    // slot 1's save_kv image after its lone session
  };
  auto reset_all = [&]() {
    for (uint32_t q = S; q-- > 0;) {
      e.select(q);
      e.reset();
    }
  };
  auto all = [&]() {
    Out o;
    o.ids.resize(2 * S);
    o.logits.resize(S);
    o.kv.resize(2 * S);
    for (uint32_t q = 0; q < S; ++q) {   // the lone sessions
      e.select(q);
      e.reset();
      e.prefill(prompt);
      for (int i = 0; i < 16; ++i)
        for (uint32_t id : lone_step(c, 0, &o.logits[q])) o.ids[q].push_back(id);
      for (int i = 0; i < 4; ++i)
        for (uint32_t id : lone_step(c, e.mtp() ? 3 : 0, &o.logits[q])) o.ids[q].push_back(id);
      o.kv[q] = c.kv_hash(q, 0, e.pos());
      if (q == 1) {
        l0::Mem kv(c.ctx, l0::MemKind::Host, e.kv_bytes(e.pos()));
        e.save_kv(0, e.pos(), kv.ptr());
        o.snapshot = fnv(kv.ptr(), kv.size());
      }
    }
    for (uint32_t q = 0; q < S; ++q) {   // the batched run
      e.select(q);
      e.reset();
      e.prefill(prompts[q]);
    }
    for (int i = 0; i < 8; ++i) {
      std::vector<Engine::Run> runs;
      std::vector<std::vector<uint32_t>> d(S);
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        o.ids[S + q].push_back(e.control()->cur_token[0]);
        if (e.mtp()) {
          e.draft(1);
          d[q] = e.draft_ids();
        }
        runs.push_back({q, e.mtp() ? 1u : 0u});
      }
      e.verify_rows(runs);
      for (uint32_t q = 0; q < S; ++q) {
        const uint32_t* ids = e.rows_ids() + (e.mtp() ? 2 * q : q);
        const uint32_t j = e.mtp() && d[q][0] == ids[0] ? 1 : 0;
        if (j) o.ids[S + q].push_back(d[q][0]);
        e.select(q);
        e.commit(j, ids[j]);
      }
    }
    for (uint32_t q = 0; q < S; ++q) {
      e.select(q);
      o.kv[S + q] = c.kv_hash(q, 0, e.pos());
    }
    return o;
  };
  reset_all();
  const Out a = all();
  e.set_kv_page_table(scattered);
  reset_all();
  const Out b = all();
  e.set_kv_page_table(identity);
  reset_all();
  bool same_slots = true;
  for (uint32_t q = 1; q < S; ++q) same_slots = same_slots && a.ids[q] == a.ids[0] && a.logits[q] == a.logits[0];
  const bool lone = a.ids == b.ids && a.logits == b.logits;
  const bool kv = a.kv == b.kv && a.snapshot == b.snapshot;
  std::printf("S7 pages: identity vs scattered (%zu pages; slot 0 %u..%u at physical %u, %u, ...): lone sessions in "
              "every slot %s, batched ids %s, every slot's KV rows and slot 1's snapshot %s; every slot's lone session "
              "as slot 0's: %s\n",
              identity.size(), virt[0].front(), virt[0].back(), scattered[virt[0][0]], scattered[virt[0][1]],
              lone ? "bitwise (ids, logits)" : "DIFFER - FAIL",
              std::equal(a.ids.begin() + S, a.ids.end(), b.ids.begin() + S) ? "bitwise" : "DIFFER - FAIL",
              kv ? "bitwise" : "DIFFER - FAIL", same_slots ? "yes" : "NO - FAIL");
  return same_slots && lone && kv;
}

// ---- spec 31: the KV page pool ----------------------------------------------------------------------------------

struct Session {
  std::vector<uint32_t> ids;
  std::vector<uint64_t> logits;
  bool operator==(const Session&) const = default;
};
// Sequence q's lone session, without a reset (the caller decides the pages it starts with) - s7's: the prompt, 16
// plain steps, 4 MTP iterations at K = 3; `beside` after every step (another sequence's work).
Session session(Ctx& c, uint32_t q, const std::vector<uint32_t>& prompt, const std::function<void()>& beside = {}) {
  Engine& e = c.e;
  Session o;
  e.select(q);
  e.prefill(prompt);
  for (int i = 0; i < 20; ++i) {
    for (uint32_t id : lone_step(c, i < 16 || !e.mtp() ? 0 : 3, &o.logits)) o.ids.push_back(id);
    if (beside) {
      beside();
      e.select(q);
    }
  }
  return o;
}
void reset_all(Engine& e) {
  for (uint32_t q = e.slots(); q-- > 0;) {
    e.select(q);
    e.reset();
  }
}
// Every byte of physical page `page` - every layer's K and V rows (and at int8 their scales), main and head - is 0.
bool page_zero(Ctx& c, uint32_t page) {
  Engine& e = c.e;
  bool zero = true;
  const auto one = [&](const runtime::KvLayout& L, void* k, void* vv) {
    for (uint32_t l = 0; l < L.layers; ++l) {
      const runtime::KvLayer y = L.at_base(L.layer(k, vv, l), page * runtime::kKvPage);
      for (int t = 0; t < 4; ++t) {
        void* ptr = t == 0 ? y.k : t == 1 ? y.v : t == 2 ? y.ks : y.vs;
        if (!ptr) continue;
        const size_t n = size_t(runtime::kKvPage) * (t < 2 ? L.row_bytes() : L.scale_row_bytes());
        for (size_t off = 0; off < n; off += c.host.size()) {
          const size_t m = std::min(c.host.size(), n - off);
          c.imm.copy(c.host.ptr(), static_cast<const uint8_t*>(ptr) + off, m);
          const uint8_t* b = static_cast<const uint8_t*>(c.host.ptr());
          zero = zero && std::all_of(b, b + m, [](uint8_t x) { return x == 0; });
        }
      }
    }
  };
  one(e.buffers().kv_lay, e.buffers().kv_k.ptr(), e.buffers().kv_v.ptr());
  if (e.mtp_buffers()) one(e.mtp_buffers()->kv_lay, e.mtp_buffers()->kv_k.ptr(), e.mtp_buffers()->kv_v.ptr());
  return zero;
}
std::string pages_str(const std::vector<uint32_t>& v) {
  std::string out;
  for (uint32_t p : v) out += (out.empty() ? "" : ",") + std::to_string(p);
  return out;
}

bool pool_cases(Ctx& c, const Session& ref, const std::vector<uint32_t>& prompt, const std::vector<uint32_t>& p1,
                const std::vector<uint32_t>& p2, const std::vector<uint32_t>& grow, const std::vector<uint32_t>& edge) {
  Engine& e = c.e;
  constexpr uint32_t P = runtime::kKvPage;
  bool ok = true;
  // P1, in order.
  reset_all(e);
  const Session a = session(c, 0, prompt);
  const std::vector<uint32_t> pa = e.pool_pages_of(0);
  bool r = a == ref;
  std::printf("P1 in order (pages %s): %s\n", pages_str(pa).c_str(), r ? "PASS (ids and logits = the fixed slots')" : "FAIL");
  ok &= r;
  // P1, reversed: sequences 1..3 take a page each and give them back in that order - the next three come out reversed.
  reset_all(e);
  for (uint32_t q = 1; q < 4; ++q) {
    e.select(q);
    CHECK(e.take_pages(P));
  }
  for (uint32_t q = 1; q < 4; ++q) {
    e.select(q);
    e.reset();
  }
  const Session b = session(c, 0, prompt);
  const std::vector<uint32_t> pb = e.pool_pages_of(0);
  r = b == ref && pb.size() >= 3 && pb[0] > pb[1] && pb[1] > pb[2];
  std::printf("P1 reversed (pages %s): %s\n", pages_str(pb).c_str(), r ? "PASS" : "FAIL");
  ok &= r;
  // P1, interleaved with sequence 1's pages, sequence 1 stepping beside it.
  reset_all(e);
  for (uint32_t i = 1; i <= 3; ++i)
    for (uint32_t q = 0; q < 2; ++q) {
      e.select(q);
      CHECK(e.take_pages(i * P));
    }
  e.select(1);
  e.prefill(p1);
  const Session d = session(c, 0, prompt, [&] {
    e.select(1);
    lone_step(c, 0);
  });
  const std::vector<uint32_t> pd = e.pool_pages_of(0), p1d = e.pool_pages_of(1);
  r = d == ref;
  std::printf("P1 interleaved (pages %s; beside it %s): %s\n", pages_str(pd).c_str(), pages_str(p1d).c_str(),
              r ? "PASS" : "FAIL");
  ok &= r;
  // P2: sequence 0 grows (a prefill over another page, rejected drafts), then resets; sequence 1 does not move.
  e.select(1);
  const uint32_t n1 = e.pos();
  const uint64_t h1 = c.slot_hash(1, n1);
  e.select(0);
  e.prefill(p2);
  for (int i = 0; i < 6; ++i) lone_step(c, i % 2 && e.mtp() ? 3 : 0);
  const std::vector<uint32_t> held = e.pool_pages_of(0);
  const uint32_t free_before = e.pages_free();
  e.select(0);
  e.reset();
  bool zero = e.pages_held(0) == 0 && e.pages_free() == free_before + held.size();
  for (uint32_t p : held) zero = zero && page_zero(c, p);
  r = c.slot_hash(1, n1) == h1;
  std::printf("P2 sequence 0's growth to %zu pages, drafts and reset: sequence 1 %s; its pages back and all zero: %s\n",
              held.size(), r ? "byte-identical" : "MOVED - FAIL", zero ? "yes" : "NO - FAIL");
  ok &= r && zero;
  const Session g = session(c, 2, prompt);   // the pages sequence 0 gave back are the next ones taken
  const std::vector<uint32_t> pg = e.pool_pages_of(2);
  bool reused = true;
  for (uint32_t p : pg) reused = reused && std::find(held.begin(), held.end(), p) != held.end();
  r = g == ref && reused;
  std::printf("P2 the next request on those pages (%s, reused %s): %s\n", pages_str(pg).c_str(), reused ? "yes" : "no",
              r ? "PASS (the reference's ids and logits)" : "FAIL");
  ok &= r;
  // P3: past 32768 with the others empty.
  std::vector<uint32_t> long_ids[2];
  for (const uint32_t q : {3u, 0u}) {
    reset_all(e);
    e.select(q);
    e.prefill(grow);
    std::vector<uint32_t>& ids = long_ids[q == 0];
    for (int i = 0; i < 4; ++i)
      for (uint32_t id : lone_step(c, 0)) ids.push_back(id);
    CHECK(e.pos() > 32768u && e.pages_held(q) == runtime::kv_pages(e.pos() + 1));
  }
  r = long_ids[0] == long_ids[1];
  std::printf("P3 a request of %zu ids past 32768 (sequence 3, then 0): %s\n", grow.size(), r ? "PASS" : "FAIL");
  ok &= r;
  // P4: a verify that needs a page the pool does not have.
  reset_all(e);
  e.select(2);
  e.prefill(edge);
  std::vector<uint32_t> want;
  for (int i = 0; i < 9; ++i)
    for (uint32_t id : lone_step(c, 0)) want.push_back(id);
  e.reset();
  e.select(0);
  e.prefill(edge);
  for (uint32_t q = 1; q < 4; ++q) {   // the others take every free page
    e.select(q);
    CHECK(e.take_pages(std::min(e.pages_free(), runtime::kv_pages(e.max_len())) * P));
  }
  e.select(0);
  const uint32_t mvk = e.max_verify_k(), pos = e.pos();
  const uint64_t before = c.slot_hash(0, pos);
  bool refused = false;
  try {
    e.verify(3);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  const bool same = c.slot_hash(0, pos) == before && e.pos() == pos;
  std::vector<uint32_t> got;
  for (uint32_t id : lone_step(c, 0)) got.push_back(id);
  for (uint32_t q = 1; q < 4; ++q) {
    e.select(q);
    e.reset();
  }
  e.select(0);
  for (int i = 0; i < 8; ++i)
    for (uint32_t id : lone_step(c, 0)) got.push_back(id);
  r = mvk == P - pos - 1 && refused && same && got == want;
  std::printf("P4 no free page at pos %u: max_verify_k %u, verify(3) %s, the sequence %s, then %s\n", pos, mvk,
              refused ? "refused" : "NOT refused - FAIL", same ? "unchanged" : "MOVED - FAIL",
              got == want ? "as an uninterrupted request (9 ids)" : "DIFFERENT - FAIL");
  ok &= r;
  return ok;
}

// S4 / S5: four requests alone and together.
struct Streams {
  std::vector<std::vector<uint32_t>> ids;
};
bool s4(Ctx& c, const std::vector<std::vector<uint32_t>>& prompts, const std::string& tag) {
  Engine& e = c.e;
  const uint32_t S = e.slots();
  const golden::ForcedJudge fj;
  // Alone: each slot's own lists, 32 plain steps then 8 greedy MTP iterations at K = 1; the plain steps' rows for the
  // tie rule.
  std::vector<std::vector<uint32_t>> plain(S);
  std::vector<std::vector<greedy_tie::RefStep>> steps(S);
  std::vector<std::vector<uint32_t>> lone(S);
  for (uint32_t q = 0; q < S; ++q) {
    e.select(q);
    e.reset();
    e.prefill(prompts[q]);
    plain[q].push_back(e.control()->cur_token[0]);
    for (int i = 0; i < 48; ++i) {
      if (e.mtp()) {
        e.verify(0);
        const std::vector<float> row = c.rows(e.verify_logits_device(), 0, 1);
        steps[q].push_back(greedy_tie::ref_step(row.data(), c.VU));
        e.commit(0, e.verify_ids()[0]);
      } else {
        e.generate(1);
        const std::vector<float> row = c.rows(e.buffers().logits.as<float>(), 0, 1);
        steps[q].push_back(greedy_tie::ref_step(row.data(), c.VU));
      }
      plain[q].push_back(e.control()->cur_token[0]);
    }
  }
  auto together = [&]() {
    std::vector<std::vector<uint32_t>> out(S);
    for (uint32_t q = 0; q < S; ++q) {
      e.select(q);
      e.reset();
      e.prefill(prompts[q]);
    }
    for (int i = 0; i < (e.mtp() ? 32 : 48); ++i) {
      std::vector<Engine::Run> runs;
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        out[q].push_back(e.control()->cur_token[0]);
        runs.push_back({q, 0});
      }
      e.verify_rows(runs);
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        e.commit(0, e.rows_ids()[q]);
      }
    }
    for (int i = 0; e.mtp() && i < 8; ++i) {   // greedy MTP, K = 1 each: 8 rows
      std::vector<Engine::Run> runs;
      std::vector<std::vector<uint32_t>> d(S);
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        out[q].push_back(e.control()->cur_token[0]);
        e.draft(1);
        d[q] = e.draft_ids();
        runs.push_back({q, 1});
      }
      e.verify_rows(runs);
      for (uint32_t q = 0; q < S; ++q) {
        const uint32_t* ids = e.rows_ids() + 2 * q;
        const uint32_t j = d[q][0] == ids[0] ? 1 : 0;
        if (j) out[q].push_back(d[q][0]);
        e.select(q);
        e.commit(j, ids[j]);
      }
    }
    for (auto& o : out) o.resize(std::min<size_t>(o.size(), 48));
    return out;
  };
  const std::vector<std::vector<uint32_t>> a = together(), b = together();
  bool ok = a == b;
  std::printf("S5 determinism: the batched run twice from the same state: %s\n", ok ? "PASS (identical ids)" : "FAIL");
  for (uint32_t q = 0; q < S; ++q) {
    // Both streams start at the prefill's pending id (the context's last id); steps[i] decided plain[i + 1].
    const size_t n = a[q].size();
    ok &= n >= 2 && a[q][0] == plain[q][0];
    std::vector<uint32_t> ctx = prompts[q];
    ctx.push_back(plain[q][0]);
    const std::vector<uint32_t> pl(plain[q].begin() + 1, plain[q].begin() + n), sp(a[q].begin() + 1, a[q].end());
    const std::vector<greedy_tie::RefStep> ps(steps[q].begin(), steps[q].begin() + (n - 1));
    const spec_judge::StreamVerdict v =
        spec_judge::streams(fj, tag + "_" + std::to_string(q), ctx, pl, ps, sp, uint32_t(V), c.VU);
    std::printf("S4 B1 slot %u (%zu ids, batched vs alone): %s\n", q, n, v.line().c_str());
    ok &= v.pass;
  }
  return ok;
}
}  // namespace

// Spec 31: the fixed slots' reference, then the pool engine on the same card (the first engine and its model freed).
int pool_main(const std::string& checkpoint) {
  const uint32_t total = runtime::slots_kv_len(kSlots);
  constexpr uint32_t kPoolPages = 20, kCap = 36864;
  CHECK_EQ(kPoolPages * runtime::kKvPage, total);
  l0::Context ctx(0);
  const auto long32k = golden::read_ids("tests/golden/prompts/long32k.ids");
  const auto code = golden::read_ids("tests/golden/prompts/code.ids");
  const auto prose = golden::read_ids("tests/golden/prompts/prose.ids");
  CHECK(long32k.size() > 9000 && !code.empty() && !prose.empty());
  const std::vector<uint32_t> prompt = cut(long32k, 0, 5000);
  Session ref;
  {
    Engine f(ctx, loader::load(ctx, checkpoint, total, true, loader::LmHeadForm::Int8), total, false,
             runtime::default_kv_cache(), nullptr, kSlots);
    f.prepare_prefill();
    Ctx c(ctx, f);
    reset_all(f);
    ref = session(c, 0, prompt);
    std::printf("slots_test --pool: the fixed slots' session (slot 0 of %s): ids %016llx logits %016llx\n",
                runtime::slot_layout_name(kSlots).c_str(), (unsigned long long)fnv(ref.ids.data(), ref.ids.size() * 4),
                (unsigned long long)fnv(ref.logits.data(), ref.logits.size() * 8));
  }
  Engine e(ctx, loader::load(ctx, checkpoint, total, true, loader::LmHeadForm::Int8), total, false,
           runtime::default_kv_cache(), nullptr, std::vector<uint32_t>(4, kCap), kPoolPages);
  e.prepare_prefill();
  std::printf("slots_test --pool: 4 sequences up to %u sharing %u pages of %u, %s KV; %s\n", e.max_len(),
              e.kv_pool_pages(), runtime::kKvPage, runtime::kv_cache_name(e.kv_cache()), e.memory_line().c_str());
  CHECK_EQ(e.slots(), 4u);
  CHECK_EQ(e.slot_base(1), runtime::kv_pages(kCap) * runtime::kKvPage);
  CHECK_EQ(e.pages_free(), kPoolPages);
  // The engine's bounds are contracts: a violation ends the process before any device work - a cache row past the
  // positions the sequence's pages hold (none yet), a slot the engine does not have, a snapshot past its pages.
  CHECK(dies([&] { e.kv_row(0, e.pages_held(0) * runtime::kKvPage); }));
  CHECK(dies([&] { e.select(e.slots()); }));
  CHECK(dies([&] { e.save_kv(0, 1, nullptr); }));
  std::printf("slots_test --pool: kv_row, select and save_kv past their bounds end the process (contracts)\n");
  Ctx c(ctx, e);
  const bool ok = pool_cases(c, ref, prompt, cut(code, 0, 1200), cut(prose, 0, 2500), cut(long32k, 0, 33500),
                             cut(long32k, 0, runtime::kKvPage - 2));
  std::printf("slots_test --pool: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  const std::string checkpoint = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  golden::load_end_of_turn(checkpoint);
  if (argc > 2 && std::string(argv[2]) == "--pool") return pool_main(checkpoint);
  const uint32_t total = runtime::slots_kv_len(kSlots);
  l0::Context ctx(0);
  const bool mtp = !(argc > 3 && std::string(argv[2]) == "--mtp" && std::string(argv[3]) == "off");
  Engine e(ctx, loader::load(ctx, checkpoint, total, mtp, loader::LmHeadForm::Int8), total, false,
           runtime::default_kv_cache(), nullptr, kSlots);
  e.prepare_prefill();
  std::printf("slots_test: %u slots (%s), %s KV; %s\n", e.slots(), runtime::slot_layout_name(kSlots).c_str(),
              runtime::kv_cache_name(e.kv_cache()), e.memory_line().c_str());
  CHECK_EQ(e.slots(), 4u);
  CHECK_EQ(e.slot_base(2), 16384u + 8192u);
  CHECK_EQ(e.max_len(), 16384u);
  const auto long32k = golden::read_ids("tests/golden/prompts/long32k.ids");
  const auto code = golden::read_ids("tests/golden/prompts/code.ids");
  const auto prose = golden::read_ids("tests/golden/prompts/prose.ids");
  const auto lng = golden::read_ids("tests/golden/prompts/long.ids");
  CHECK(long32k.size() > 9000 && !code.empty() && !prose.empty() && !lng.empty());
  const std::vector<uint32_t> p_deep = cut(long32k, 0, 6000), p1 = cut(code, 0, 1200), p2 = cut(prose, 0, 900),
                              p3 = cut(lng, 0, 1000);
  Ctx c(ctx, e);
  bool ok = true;
  ok &= s1(c, p1);
  ok &= s2(c, p1, p2, p3);
  ok &= s3(c, p_deep, p1, p2, p3);
  const std::string tag = std::string("slots_") + runtime::kv_cache_name(e.kv_cache()) + (e.mtp() ? "" : "_nomtp");
  ok &= s4(c, {p_deep, p1, p2, p3}, tag);
  if (e.mtp()) ok &= s6(c, {p_deep, p1, p2, p3});
  ok &= s7(c, cut(long32k, 0, 5000), {p_deep, p1, p2, p3});
  std::printf("slots_test: %s\n", ok ? "PASS" : "FAIL");
  return spec_judge::exit_code(ok, "slots_test");
}
