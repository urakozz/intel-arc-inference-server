// probe_batch_step - spec 13a (plan 13a Task 2): a decode step whose rows belong to DIFFERENT sequences, on one card.
// Built only with -DB70_BATCH_PROBE=ON (tools/probe/CMakeLists.txt); record: docs/probe-batching-2026-10-23.md.
//
//   probe_batch_step <snapshot> [--kv bf16|int8] [--mtp on|off] [--slots L0,L1,..] [--deep 32768] [--shallow 4096]
//                    [--calls 32] [--rounds 3] [--arms a,b,..] [--no-check] [--no-time] [--no-chunks]
//                    [--chunks 512,1024,2048] [--chunk-depths 4096,32768] [--chunk-calls 3]
//
// **The allocation.** ONE engine (the int8 lm_head, the MTP head unless --mtp off) at max_len = the slots' summed
// lengths, so a slot is a position range of the engine's KV cache (its KV base; the MTP head's KV layer alike), and:
//   GDN states   index 0 = the engine's gdn_state = slot 0's; a pool of (slots - 1) sources (slot s's is index s) and
//                8 destinations (index slots + m for row m - the probe's scratch: 13b gives each slot its own);
//   conv rings   slot 0's = the engine's; a pool of slots - 1;
//   hprev        bf16 [slots][hidden], each slot's h_{pos-1} (the MTP head's input at a run's first row).
// Default slots (spec 13 §9.1 / §9.2's layouts at the probe's depths): Qwen3.8 36864 + 3 x 8192; Agnes 36864 + 8192
// with MTP, 36864 + 3 x 8192 with --mtp off. Each slot's sequence - tests/golden/prompts/long32k.ids from a per-slot
// offset, repeated past its end - is prefilled into slot 0 by Engine::prefill (the served path, MTP head included) and
// copied out (state, ring, KV rows and scales, the head's KV rows, h_{pos-1}); slot 0's is prefilled last and stays.
// Slot 0 holds `deep` ids, the others `shallow`.
//
// **The arms.** An arm is a table: per slot its rows (1 = a plain row, K + 1 = an MTP verify run) at the slot's depth
// - "@<shallow>" every slot at the shallow depth (slot 0 rewound there: the attention reads `shallow` keys; its GDN
// state is the deep one - a step's cost does not depend on the values), "@<deep>+<shallow>" slot 0 deep. Each arm's
// list is runtime::build_rows at M = its rows (plus the head's KV fill with MTP).
//
// **The check** (unless --no-check; plan 13a Review Focus 2-3), every arm: the B-row step, then every slot's run alone
// (the same rows, the same src / dst, at M = its rows): logits rows BITWISE, the argmax ids, every row's dst GDN state,
// the main KV rows written (every FA layer, K / V and the int8 scales) and the head's KV rows written - all equal; and
// isolation: no src state and no KV row of a window [pos - 8, pos + 16) of any slot moves in the B-row step but the
// rows' own. Then the PRODUCTION tie for slot 0 at the deep depth: Engine::verify(0) and verify(3) with MTP (the lists
// that ship at M = 1 and 4), Engine::generate(1) without - logits bitwise the row list's for slot 0 alone (state 0 is
// restored around them). "CHECK <arm>: PASS|FAIL ..." per arm, "check: PASS|FAIL" at the end.
//
// **The timing** (unless --no-time): every arm against the baseline `1x1@<shallow>` (one row of slot 1) as INTERLEAVED
// PAIRS - baseline and arm back to back, the order alternating by round, `calls` replays each - after a warm-up of
// every arm (8 replays), median of `rounds` of the ratio; a replay is what an engine pays: host writes of the tables
// and ids, submit, fence wait. With MTP, the production M = 1 step (Engine::verify(0), pos rewound) is paired against
// the row list's 1x1 of slot 0 at the same depth (a lone request must not pay for the row build, spec 13 §8).
//
// **The chunk arms** (unless --no-chunks): Engine::prefill of ONE chunk of C ids at depth d into slot 0 (C in
// --chunks, d in --chunk-depths), rotated rounds, median of `rounds` of the mean of --chunk-calls chunks; printed
// beside the measured 1-row steps, and the worst inter-token gap of a decoding slot while a prompt prefills at that
// chunk (one chunk + one step, DERIVED from the two measured numbers).
//
// runtime::build_rows exists only when b70_runtime is built with B70_BATCH_PROBE (src/runtime/CMakeLists.txt), and
// tools/probe/CMakeLists.txt builds this file only then. A compile without the define (the Mac's section 2 syntax
// check walks every probe source) sees the stub at the end; the real body is checked with -DB70_BATCH_PROBE=1.
#if B70_BATCH_PROBE
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"
#include "runtime/batch_rows.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
using Clock = std::chrono::steady_clock;

std::vector<uint32_t> read_ids(const std::string& p) {
  std::vector<uint32_t> v;
  FILE* f = std::fopen(p.c_str(), "r");
  if (!f) return v;
  unsigned x;
  while (std::fscanf(f, "%u", &x) == 1) v.push_back(x);
  std::fclose(f);
  return v;
}
std::vector<uint32_t> parse_list(const std::string& s) {
  std::vector<uint32_t> v;
  size_t i = 0;
  while (i < s.size()) {
    const size_t j = s.find(',', i);
    v.push_back(uint32_t(std::stoul(s.substr(i, j == std::string::npos ? std::string::npos : j - i))));
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return v;
}
double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
std::string kname(uint32_t n) { return n % 1024 == 0 ? std::to_string(n / 1024) + "k" : std::to_string(n); }
uint8_t* u8(void* p) { return static_cast<uint8_t*>(p); }
uint64_t hash_bytes(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w;
    std::memcpy(&w, b + i, 8);
    h = (h ^ w) * 1099511628211ull;
  }
  for (; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

// One arm: per chosen slot its rows, every slot at the shallow depth or slot 0 at the deep one.
struct Arm {
  std::string name;
  std::vector<uint32_t> slots, rows;   // parallel: the run order
  bool deep = false;
  uint32_t m() const {
    uint32_t n = 0;
    for (uint32_t r : rows) n += r;
    return n;
  }
  bool spec() const {
    for (uint32_t r : rows)
      if (r > 1) return true;
    return false;
  }
};
Arm make_arm(uint32_t B, const std::vector<uint32_t>& rows, bool deep, uint32_t deep_len, uint32_t shallow_len) {
  Arm a;
  a.rows = rows;
  a.deep = deep;
  // Slots: slot 0 first when deep; otherwise slots 1.. first and slot 0 (rewound) last, so the narrow arms never
  // rewind it.
  std::vector<uint32_t> order;
  if (deep) order.push_back(0);
  for (uint32_t s = 1; s < B; ++s) order.push_back(s);
  if (!deep) order.push_back(0);
  if (rows.size() > B) throw std::invalid_argument("an arm with more slots than the layout");
  a.slots.assign(order.begin(), order.begin() + rows.size());
  for (size_t i = 0; i < rows.size();) {   // "1x4+3x1": runs of equal counts
    size_t j = i;
    while (j < rows.size() && rows[j] == rows[i]) ++j;
    a.name += (a.name.empty() ? "" : "+") + std::to_string(j - i) + "x" + std::to_string(rows[i]);
    i = j;
  }
  a.name += "@" + (deep ? kname(deep_len) + "+" + kname(shallow_len) : kname(shallow_len));
  if (deep && rows.size() == 1) a.name = "1x" + std::to_string(rows[0]) + "@" + kname(deep_len);
  return a;
}

// The acceptance spec 13 §9 derives with (measured, spec 8: ids a verify of K drafts yields on average).
double ids_per_run(uint32_t rows) { return rows == 1 ? 1.0 : rows == 2 ? 1.92 : rows == 4 ? 3.455 : 0.0; }

struct Probe {
  l0::Context& ctx;
  runtime::Engine& e;
  const model::ModelDesc& d;
  runtime::BatchLayout lay;
  uint32_t B;
  bool mtp;
  uint32_t deep, shallow;
  std::vector<std::vector<uint32_t>> seq;   // each slot's ids (prompt + continuation)
  std::vector<uint32_t> pend;               // each slot's pending id after its prefill
  size_t state_bytes, ring_bytes, hid_bytes;
  std::unique_ptr<l0::Mem> state_pool, ring_pool, hprev, rows, hrows, host, backup0;
  l0::Queue q;
  l0::Fence fence;
  l0::CmdList imm;
  std::map<std::string, runtime::CapturedStep> lists;

  Probe(l0::Context& c, runtime::Engine& en, runtime::BatchLayout l, bool m, uint32_t dp, uint32_t sh)
      : ctx(c), e(en), d(*en.model().desc), lay(std::move(l)), B(uint32_t(lay.slots.size())), mtp(m), deep(dp),
        shallow(sh), q(c), fence(q), imm(l0::CmdList::immediate(c)) {
    runtime::DecodeBuffers& b = e.buffers();
    state_bytes = b.gdn_state.size();
    ring_bytes = b.conv_ring.size();
    hid_bytes = size_t(d.hidden) * 2;
    state_pool = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, state_bytes * (B - 1 + runtime::DecodeScratchDims::kM));
    if (B > 1) ring_pool = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, ring_bytes * (B - 1));
    hprev = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, hid_bytes * B);
    rows = std::make_unique<l0::Mem>(ctx, l0::MemKind::Shared, sizeof(runtime::BatchRowTable));
    hrows = std::make_unique<l0::Mem>(ctx, l0::MemKind::Shared, sizeof(runtime::BatchRowTable));
    host = std::make_unique<l0::Mem>(ctx, l0::MemKind::Host, std::max<size_t>(state_bytes, size_t(8) * 248320 * 4));
    backup0 = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, state_bytes);
    imm.fill(state_pool->ptr(), 0u, state_pool->size());
    if (ring_pool) imm.fill(ring_pool->ptr(), 0u, ring_pool->size());
    imm.fill(hprev->ptr(), 0u, hprev->size());
  }
  runtime::Control* ctl() { return e.buffers().control.as<runtime::Control>(); }
  runtime::Control* hctl() { return e.mtp_buffers()->hctl.as<runtime::Control>(); }
  void* state_at(uint32_t i) {
    return i == 0 ? e.buffers().gdn_state.ptr() : u8(state_pool->ptr()) + size_t(i - 1) * state_bytes;
  }
  void* ring_at(uint32_t s) {
    return s == 0 ? e.buffers().conv_ring.ptr() : u8(ring_pool->ptr()) + size_t(s - 1) * ring_bytes;
  }

  // KV rows [from, from + n) -> [to, to + n) in every main layer (and the head's), K, V and the int8 scales.
  void copy_kv(uint32_t from, uint32_t to, uint32_t n) {
    const auto one = [&](const runtime::KvLayout& kl, void* kb, void* vb) {
      for (uint32_t f = 0; f < kl.layers; ++f) {
        const runtime::KvLayer L = kl.layer(kb, vb, f);
        const size_t rb = kl.row_bytes(), sb = kl.scale_row_bytes();
        imm.copy(u8(L.k) + to * rb, u8(L.k) + from * rb, n * rb);
        imm.copy(u8(L.v) + to * rb, u8(L.v) + from * rb, n * rb);
        if (L.int8()) {
          imm.copy(u8(L.ks) + to * sb, u8(L.ks) + from * sb, n * sb);
          imm.copy(u8(L.vs) + to * sb, u8(L.vs) + from * sb, n * sb);
        }
      }
    };
    one(e.buffers().kv_lay, e.buffers().kv_k.ptr(), e.buffers().kv_v.ptr());
    if (mtp) one(e.mtp_buffers()->kv_lay, e.mtp_buffers()->kv_k.ptr(), e.mtp_buffers()->kv_v.ptr());
  }
  // A hash of KV rows [p, p + n) of every main layer (or of the head's layer).
  uint64_t kv_hash(uint32_t p, uint32_t n, bool head) {
    const runtime::KvLayout& kl = head ? e.mtp_buffers()->kv_lay : e.buffers().kv_lay;
    void* kb = head ? e.mtp_buffers()->kv_k.ptr() : e.buffers().kv_k.ptr();
    void* vb = head ? e.mtp_buffers()->kv_v.ptr() : e.buffers().kv_v.ptr();
    uint64_t h = 1469598103934665603ull;
    for (uint32_t f = 0; f < kl.layers; ++f) {
      const runtime::KvLayer L = kl.layer(kb, vb, f);
      const size_t rb = kl.row_bytes(), sb = kl.scale_row_bytes();
      for (void* base : {L.k, L.v}) {
        imm.copy(host->ptr(), u8(base) + size_t(p) * rb, n * rb);
        h = hash_bytes(host->ptr(), n * rb, h);
      }
      if (L.int8())
        for (void* base : {L.ks, L.vs}) {
          imm.copy(host->ptr(), u8(base) + size_t(p) * sb, n * sb);
          h = hash_bytes(host->ptr(), n * sb, h);
        }
    }
    return h;
  }
  uint64_t state_hash(uint32_t i) {
    imm.copy(host->ptr(), state_at(i), state_bytes);
    return hash_bytes(host->ptr(), state_bytes);
  }

  // Prefill every slot (module comment).
  void prefill_slots(const std::vector<uint32_t>& file) {
    e.reset();
    pend.assign(B, 0);
    seq.assign(B, {});
    for (uint32_t s = 0; s < B; ++s) {
      const uint32_t n = s == 0 ? deep : shallow;
      for (uint32_t i = 0; i < n + 64; ++i) seq[s].push_back(file[(size_t(s) * 7919 + i) % file.size()]);
    }
    for (uint32_t s = B; s-- > 0;) {
      const uint32_t n = s == 0 ? deep : shallow;
      if (s != B - 1) {   // a fresh session in slot 0's place: zero state and ring, pos 0
        imm.fill(e.buffers().gdn_state.ptr(), 0u, state_bytes);
        imm.fill(e.buffers().conv_ring.ptr(), 0u, ring_bytes);
        ctl()->pos = 0;
        ctl()->n_active = 0;
        ctl()->gdn_live = 0;
      }
      const Clock::time_point t0 = Clock::now();
      e.prefill(std::vector<uint32_t>(seq[s].begin(), seq[s].begin() + n));
      pend[s] = ctl()->cur_token[0];
      if (mtp) imm.copy(u8(hprev->ptr()) + s * hid_bytes, e.mtp_buffers()->hh.ptr(), hid_bytes);
      if (s != 0) {
        imm.copy(state_at(s), state_at(0), state_bytes);
        imm.copy(ring_at(s), ring_at(0), ring_bytes);
        copy_kv(0, lay.kv_base(s), n);
      }
      std::printf("prefill: slot %u, %u ids (KV base %u) in %.1f s; pending id %u\n", s, n, lay.kv_base(s),
                  std::chrono::duration<double>(Clock::now() - t0).count(), pend[s]);
    }
    imm.copy(backup0->ptr(), state_at(0), state_bytes);   // slot 0's source, for the production tie
  }

  // The arm's table (and the head's), every slot's rows from its depth.
  std::vector<runtime::BatchRow> table(const Arm& a, int only = -1) {
    std::vector<runtime::RowRun> runs;
    uint32_t m = 0;
    for (size_t i = 0; i < a.slots.size(); ++i) {
      const uint32_t s = a.slots[i];
      runtime::RowRun r;
      r.slot = s;
      r.pos = (a.deep && s == 0) ? deep : shallow;
      r.rows = a.rows[i];
      r.src = s;
      for (uint32_t j = 0; j < r.rows; ++j) r.dst.push_back(B + m + j);   // the row's place in the WHOLE arm
      m += r.rows;
      if (only < 0 || int(s) == only) runs.push_back(r);
    }
    return runtime::make_rows(lay, runs);
  }
  // The row ids: the slot's pending id, then the continuation of its sequence (arbitrary drafts).
  // A run's first row sits at its slot's depth (deep or shallow); later rows at depth + j (j < 8 < deep - shallow).
  uint32_t row_id(const runtime::BatchRow& r) {
    return (r.pos == deep || r.pos == shallow) ? pend[r.slot] : seq[r.slot][r.pos % seq[r.slot].size()];
  }
  runtime::CapturedStep& list(const std::vector<runtime::BatchRow>& t) {
    const std::vector<uint32_t> shape = runtime::run_lengths(t);
    std::string key = std::to_string(t.size());
    std::vector<uint32_t> run_slots;
    for (size_t m = 0; m < t.size(); ++m)
      if (m == 0 || t[m - 1].slot != t[m].slot) run_slots.push_back(t[m].slot);
    for (size_t i = 0; i < shape.size(); ++i) key += ":" + std::to_string(run_slots[i]) + "x" + std::to_string(shape[i]);
    auto it = lists.find(key);
    if (it != lists.end()) return it->second;
    runtime::BatchRowsBind bind;
    bind.rows = rows.get();
    bind.state_pool = state_pool.get();
    bind.ring_pool = ring_pool.get();
    if (mtp) {
      bind.mtp = e.mtp_buffers();
      bind.head_rows = hrows.get();
      bind.hprev = hprev.get();
      bind.run_slots = run_slots;
      bind.run_lengths = shape;
    }
    return lists.emplace(key, runtime::build_rows(ctx, e.model(), e.buffers(), bind, uint32_t(t.size()))).first->second;
  }
  // One replay of a table: the host writes, submit, fence.
  void replay(runtime::CapturedStep& l, const std::vector<runtime::BatchRow>& t) {
    runtime::BatchRowTable* rt = rows->as<runtime::BatchRowTable>();
    runtime::BatchRowTable* ht = hrows->as<runtime::BatchRowTable>();
    runtime::Control* c = ctl();
    c->n_active = uint32_t(t.size());
    for (size_t m = 0; m < t.size(); ++m) {
      rt->row[m] = t[m];
      c->cur_token[m] = row_id(t[m]);
    }
    if (mtp) {
      runtime::Control* h = hctl();
      h->n_active = uint32_t(t.size());
      for (size_t m = 0; m < t.size(); ++m) {
        ht->row[m] = t[m];
        ht->row[m].pos = t[m].pos - 1;   // the head's pair (h_{p-1}, x_p) sits at p - 1
        h->cur_token[m] = c->cur_token[m];
      }
    }
    q.execute(l.list, &fence);
    fence.wait();
  }

  struct RowOut {
    std::vector<float> logits;
    uint32_t id = 0;
    uint64_t dst = 0, kv = 0, head_kv = 0;
  };
  std::vector<RowOut> outputs(const std::vector<runtime::BatchRow>& t) {
    std::vector<RowOut> o(t.size());
    const size_t V = 248320;
    imm.copy(host->ptr(), e.buffers().logits.ptr(), t.size() * V * 4);
    for (size_t m = 0; m < t.size(); ++m) {
      o[m].logits.assign(host->as<float>() + m * V, host->as<float>() + (m + 1) * V);
      o[m].id = ctl()->out_token[m];
    }
    for (size_t m = 0; m < t.size(); ++m) {
      o[m].dst = state_hash(t[m].dst);
      o[m].kv = kv_hash(t[m].kv_base + t[m].pos, 1, false);
      if (mtp) o[m].head_kv = kv_hash(t[m].kv_base + t[m].pos - 1, 1, true);
    }
    return o;
  }

  // Review Focus 2-3 on one arm. Returns the failures.
  int check(const Arm& a) {
    const std::vector<runtime::BatchRow> t = table(a);
    runtime::CapturedStep& l = list(t);
    // Isolation: every slot's source state, and the KV rows of a window [depth - 8, depth + 16) of every slot but the
    // rows the step writes (its rows' positions), before and after.
    const auto window_hash = [&](uint32_t s) {
      const uint32_t dpt = (s == 0 && a.deep) ? deep : shallow, base = lay.kv_base(s);
      uint32_t n = 0;
      for (const runtime::BatchRow& r : t) n += r.slot == s;
      return kv_hash(base + dpt - 8, 8, false) ^ (kv_hash(base + dpt + n, 16 - n, false) * 31);
    };
    std::vector<uint64_t> src0(B), win0(B);
    for (uint32_t s = 0; s < B; ++s) {
      src0[s] = state_hash(s);
      win0[s] = window_hash(s);
    }
    replay(l, t);
    const std::vector<RowOut> batch = outputs(t);
    int bad = 0;
    std::string why;
    for (uint32_t s = 0; s < B; ++s) {
      if (state_hash(s) != src0[s]) {
        ++bad;
        why += " src" + std::to_string(s) + "-moved";
      }
      if (window_hash(s) != win0[s]) {
        ++bad;
        why += " kv-window" + std::to_string(s) + "-moved";
      }
    }
    // Every run alone.
    uint32_t m0 = 0;
    for (size_t i = 0; i < a.slots.size(); ++i) {
      const std::vector<runtime::BatchRow> ta = table(a, int(a.slots[i]));
      replay(list(ta), ta);
      const std::vector<RowOut> alone = outputs(ta);
      for (size_t j = 0; j < ta.size(); ++j) {
        const RowOut &x = batch[m0 + j], &y = alone[j];
        const std::string w = " row" + std::to_string(m0 + j);
        if (std::memcmp(x.logits.data(), y.logits.data(), x.logits.size() * 4) != 0) { ++bad; why += w + ":logits"; }
        if (x.id != y.id) { ++bad; why += w + ":id"; }
        if (x.dst != y.dst) { ++bad; why += w + ":gdn-state"; }
        if (x.kv != y.kv) { ++bad; why += w + ":kv"; }
        if (x.head_kv != y.head_kv) { ++bad; why += w + ":head-kv"; }
      }
      m0 += uint32_t(ta.size());
    }
    std::printf("CHECK %-20s M=%u: %s%s\n", a.name.c_str(), a.m(), bad ? "FAIL" : "PASS",
                bad ? why.c_str() : " (each slot's rows bitwise its run alone: logits, ids, GDN states, KV and head KV "
                                    "rows; no other slot's source or KV moved)");
    return bad;
  }

  // The production tie for slot 0 at the deep depth (module comment).
  int tie() {
    int bad = 0;
    const auto restore = [&] {
      imm.copy(state_at(0), backup0->ptr(), state_bytes);
      ctl()->gdn_live = 0;
      if (mtp) imm.copy(e.mtp_buffers()->hh.ptr(), hprev->ptr(), hid_bytes);
    };
    for (uint32_t k : mtp ? std::vector<uint32_t>{0, 3} : std::vector<uint32_t>{0}) {
      const Arm a = make_arm(B, {k + 1}, true, deep, shallow);
      restore();
      const std::vector<runtime::BatchRow> t = table(a);
      replay(list(t), t);
      const std::vector<RowOut> rowsout = outputs(t);
      restore();
      runtime::Control* c = ctl();
      c->pos = deep;
      for (uint32_t r = 0; r <= k; ++r) c->cur_token[r] = row_id(t[r]);
      std::vector<RowOut> prod(k + 1);
      if (mtp) {
        e.verify(k);
        const size_t V = 248320;
        imm.copy(host->ptr(), e.verify_logits_device(), (k + 1) * V * 4);
        for (uint32_t r = 0; r <= k; ++r) prod[r].logits.assign(host->as<float>() + r * V, host->as<float>() + (r + 1) * V);
      } else {
        c->n_active = 1;
        e.generate(1);
        imm.copy(host->ptr(), e.buffers().logits.ptr(), 248320 * 4);
        prod[0].logits.assign(host->as<float>(), host->as<float>() + 248320);
      }
      int b = 0;
      for (uint32_t r = 0; r <= k; ++r)
        b += std::memcmp(prod[r].logits.data(), rowsout[r].logits.data(), prod[r].logits.size() * 4) != 0;
      std::printf("CHECK production M=%u (%s) vs rows 1x%u@%s: %s\n", k + 1,
                  mtp ? "Engine::verify" : "Engine::generate, the plain list", k + 1, kname(deep).c_str(),
                  b ? "FAIL (logits differ)" : "PASS (logits bitwise)");
      bad += b;
    }
    restore();
    return bad;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <snapshot> [--kv bf16|int8] [--mtp on|off] [--slots L0,L1,..] [--deep N] [--shallow N]\n"
                 "       [--calls N] [--rounds N] [--arms a,b,..] [--no-check] [--no-time] [--no-chunks]\n"
                 "       [--chunks 512,1024,2048] [--chunk-depths 4096,32768] [--chunk-calls N]\n",
                 argv[0]);
    return 2;
  }
  std::string kv_s = "bf16", arms_s;
  bool mtp = true, do_check = true, do_time = true, do_chunks = true;
  uint32_t deep = 32768, shallow = 4096, calls = 32, rounds = 3, chunk_calls = 3;
  std::vector<uint32_t> slots_arg, chunks = {512, 1024, 2048}, chunk_depths = {4096, 32768};
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    if (a == "--kv") kv_s = next();
    else if (a == "--mtp") mtp = next() != "off";
    else if (a == "--slots") slots_arg = parse_list(next());
    else if (a == "--deep") deep = uint32_t(std::stoul(next()));
    else if (a == "--shallow") shallow = uint32_t(std::stoul(next()));
    else if (a == "--calls") calls = uint32_t(std::stoul(next()));
    else if (a == "--rounds") rounds = uint32_t(std::stoul(next()));
    else if (a == "--arms") arms_s = next();
    else if (a == "--no-check") do_check = false;
    else if (a == "--no-time") do_time = false;
    else if (a == "--no-chunks") do_chunks = false;
    else if (a == "--chunks") chunks = parse_list(next());
    else if (a == "--chunk-depths") chunk_depths = parse_list(next());
    else if (a == "--chunk-calls") chunk_calls = uint32_t(std::stoul(next()));
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  const runtime::KvCache kv = runtime::parse_kv_cache(kv_s.c_str());
  std::vector<uint32_t> file = read_ids("tests/golden/prompts/long32k.ids");   // cwd = the source tree
  if (file.empty()) {
    std::fprintf(stderr, "tests/golden/prompts/long32k.ids not found (run from the source tree)\n");
    return 2;
  }
  // Load at the four-slot length first; the model decides the default layout, then the RoPE table is re-tabled to it.
  std::vector<uint32_t> four = {36864, 8192, 8192, 8192};
  uint32_t load_len = 0;
  for (uint32_t l : (slots_arg.empty() ? four : slots_arg)) load_len += l;
  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, argv[1], load_len, mtp, loader::LmHeadForm::Int8);
  const model::ModelDesc& d = *model.desc;
  const bool agnes = d.name.find("agnes") != std::string::npos;
  if (slots_arg.empty()) slots_arg = (agnes && mtp) ? std::vector<uint32_t>{36864, 8192} : four;
  runtime::BatchLayout lay;
  for (uint32_t l : slots_arg) lay.slots.push_back({l, mtp});
  const uint32_t total = lay.total();
  if (total != load_len) loader::set_max_len(ctx, model, total);
  const uint32_t B = uint32_t(lay.slots.size());
  bool ok = B >= 2 && shallow >= 16 && deep >= shallow + 16 && slots_arg[0] >= deep + 16;
  for (uint32_t s = 1; s < B; ++s) ok = ok && slots_arg[s] >= shallow + 16;
  if (!ok) {
    std::fprintf(stderr, "the layout needs >= 2 slots, 16 <= shallow, shallow + 16 <= deep, slot 0 >= deep + 16 and "
                         "every other slot >= shallow + 16 (a chunk arm past slot 0's length is skipped)\n");
    return 2;
  }
  std::printf("probe_batch_step: %s, %s KV, MTP %s, int8 lm_head; slots", d.name.c_str(), runtime::kv_cache_name(kv),
              mtp ? "on" : "off");
  for (uint32_t s = 0; s < B; ++s) std::printf(" %u", slots_arg[s]);
  std::printf(" (%u positions); depths %u (slot 0) / %u\n", total, deep, shallow);
  runtime::Engine e(ctx, std::move(model), total, false, kv);
  Probe p(ctx, e, lay, mtp, deep, shallow);
  std::printf("memory: %s; probe pools %.3f GB (%u GDN states, %u rings)\n", e.memory_line().c_str(),
              (p.state_pool->size() + (p.ring_pool ? p.ring_pool->size() : 0) + p.backup0->size()) / 1e9,
              B - 1 + runtime::DecodeScratchDims::kM + 1, B - 1);
  p.prefill_slots(file);

  // The arms: plain rows at equal and mixed depths, and (MTP) verify-shaped runs up to the 8-row budget.
  std::vector<Arm> arms;
  const auto add = [&](std::vector<uint32_t> r, bool dp) {
    if (r.size() <= B) arms.push_back(make_arm(B, r, dp, deep, shallow));
  };
  add({1}, false);   // the baseline: one row of slot 1
  add({1}, true);
  for (uint32_t n = 2; n <= B; ++n) add(std::vector<uint32_t>(n, 1), false);
  add({1, 1}, true);
  if (B > 2) add(std::vector<uint32_t>(B, 1), true);
  if (mtp) {
    add({2}, false);
    add({4}, false);
    add({2, 2}, false);
    add({4, 4}, false);
    add({2, 2}, true);
    add({4, 4}, true);
    add({4, 1}, true);
    if (B >= 4) {
      add({2, 2, 2, 2}, false);
      add({2, 2, 2, 2}, true);
      add({4, 1, 1, 1}, true);
      add({4, 1, 1, 1}, false);
    }
  }
  if (!arms_s.empty()) {
    std::vector<Arm> keep;
    for (const Arm& a : arms)
      if (keep.empty() || a.name == arms[0].name || ("," + arms_s + ",").find("," + a.name + ",") != std::string::npos)
        keep.push_back(a);
    arms = keep;
  }

  int bad = 0;
  if (do_check) {
    for (const Arm& a : arms) bad += p.check(a);
    bad += p.tie();
    std::printf("check: %s (%zu arms + the production tie)\n", bad ? "FAIL" : "PASS", arms.size());
  }

  double step_shallow = 0, step_deep = 0;
  if (do_time) {
    const auto run = [&](const Arm& a, uint32_t n) {
      const std::vector<runtime::BatchRow> t = p.table(a);
      runtime::CapturedStep& l = p.list(t);
      const Clock::time_point t0 = Clock::now();
      for (uint32_t i = 0; i < n; ++i) p.replay(l, t);
      return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / n;
    };
    const Arm& base = arms[0];
    for (const Arm& a : arms) run(a, 8);   // warm-up: every list captured and replayed
    std::printf("timing: interleaved pairs against %s (order alternating by round), %u replays each, median of %u\n",
                base.name.c_str(), calls, rounds);
    std::printf("| arm | slots | rows | ms (median) | x %s (median) | range | aggregate ids / step-cost (DERIVED) |\n",
                base.name.c_str());
    for (size_t i = 0; i < arms.size(); ++i) {
      const Arm& a = arms[i];
      std::vector<double> ratio, ms, msb;
      for (uint32_t r = 0; r < rounds; ++r) {
        double tb, ta;
        if ((r + i) % 2 == 0) {
          tb = run(base, calls);
          ta = run(a, calls);
        } else {
          ta = run(a, calls);
          tb = run(base, calls);
        }
        ratio.push_back(ta / tb);
        ms.push_back(ta);
        msb.push_back(tb);
      }
      std::vector<double> rs = ratio;
      std::sort(rs.begin(), rs.end());
      double ids = 0;
      for (uint32_t rr : a.rows) ids += ids_per_run(rr);
      std::printf("| %s | %zu | %u | %.3f | %.3f | %.3f-%.3f | %.2f |\n", a.name.c_str(), a.rows.size(), a.m(),
                  median_of(ms), median_of(ratio), rs.front(), rs.back(), ids / median_of(ratio));
      if (a.name == base.name) step_shallow = median_of(ms);
      if (a.rows.size() == 1 && a.rows[0] == 1 && a.deep) step_deep = median_of(ms);
    }
    if (mtp) {   // the production M = 1 step (verify(0), pos rewound) against the row list's 1x1 of slot 0, deep
      const Arm one = make_arm(B, {1}, true, deep, shallow);
      const auto prod = [&](uint32_t n) {
        runtime::Control* c = p.ctl();
        const Clock::time_point t0 = Clock::now();
        for (uint32_t i = 0; i < n; ++i) {
          c->pos = deep;
          c->gdn_live = 0;
          c->cur_token[0] = p.pend[0];
          e.verify(0);
        }
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / n;
      };
      prod(8);
      std::vector<double> ratio;
      for (uint32_t r = 0; r < rounds; ++r) {
        double tp, tr;
        if (r % 2 == 0) {
          tp = prod(calls);
          tr = run(one, calls);
        } else {
          tr = run(one, calls);
          tp = prod(calls);
        }
        ratio.push_back(tr / tp);
      }
      std::printf("| rows 1x1@%s vs production verify(0) | 1 | 1 | - | %.4f | - | - |\n", kname(deep).c_str(),
                  median_of(ratio));
      p.imm.copy(p.state_at(0), p.backup0->ptr(), p.state_bytes);
    }
  }

  if (do_chunks) {
    struct Ch {
      uint32_t c, depth;
      std::vector<double> ms;
    };
    std::vector<Ch> ch;
    for (uint32_t dpt : chunk_depths)
      for (uint32_t c : chunks)
        if (size_t(dpt) + c <= slots_arg[0]) ch.push_back({c, dpt, {}});
    const auto run = [&](Ch& x, uint32_t n) {
      const std::vector<uint32_t> ids(p.seq[0].begin(), p.seq[0].begin() + x.c);
      double sum = 0;
      for (uint32_t i = 0; i < n; ++i) {
        p.ctl()->pos = x.depth;
        p.ctl()->gdn_live = 0;
        const Clock::time_point t0 = Clock::now();
        e.prefill(ids, x.c);
        sum += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      }
      return sum / n;
    };
    for (Ch& x : ch) run(x, 1);   // warm-up
    for (uint32_t r = 0; r < rounds; ++r)
      for (size_t i = 0; i < ch.size(); ++i) {
        Ch& x = ch[(i + r) % ch.size()];
        x.ms.push_back(run(x, chunk_calls));
      }
    std::printf("chunks: one Engine::prefill chunk into slot 0 at depth, rotated rounds, median of %u of %u chunks\n",
                rounds, chunk_calls);
    std::printf("| chunk | depth | ms / chunk | ms / id | x the 1-row step | worst gap of a decoding slot, ms (DERIVED) |\n");
    for (Ch& x : ch) {
      const double m = median_of(x.ms);
      const double step = x.depth >= deep ? step_deep : step_shallow;
      std::printf("| %u | %u | %.2f | %.4f | %s | %s |\n", x.c, x.depth, m, m / x.c,
                  step > 0 ? std::to_string(m / step).c_str() : "-",
                  step > 0 ? std::to_string(m + step).c_str() : "-");
    }
  }
  std::printf("probe_batch_step: done%s\n", do_check ? (bad ? " - CHECK FAILED" : " - check PASS") : "");
  return bad ? 1 : 0;
}
#else
#include <cstdio>
int main() {
  std::fprintf(stderr, "probe_batch_step: built without B70_BATCH_PROBE (cmake -DB70_BATCH_PROBE=ON)\n");
  return 2;
}
#endif
