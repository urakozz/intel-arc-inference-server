// Spec 17c: `TpEngine::prefill` - tensor-parallel PREFILL across two B70s - and its accessors, compiled into
// b70_prefill_host for Engine::prefill's reason (engine_prefill.cc): a decode-only binary links b70_runtime and must
// not acquire the prefill library. TpEngine reaches this half through TpPrefill, an incomplete type to it.
//
// **The wavefront (operator decision D8, plan 17c's probe record).** A TP card's prefill chunk is one card's chunk at
// the card's shapes, except that after each row-parallel linear (out_proj, o_proj, down: 127 a chunk that matter,
// the last layer's down only for the head's row) the two cards add their halves. At 2048 rows that is 40 MiB of fp32
// each way a crossing, 4.35 ms with both directions at once (probe_tp_pf: ~9.6 GB/s a direction on the copy engines,
// the compute lists and kernel stores alike; pulls 1.8 GB/s) - ~550 ms a 2048-row chunk, about what a card's half of
// the chunk's compute costs, so the crossings must run BESIDE the compute. They do on the copy-only queue group (the
// probe: neither slows the other). The prompt's chunks therefore go in pairs (A, B), and each card appends
//
//   for unit u = 0 .. 2 x layers - 1 (layer u / 2's mixer, or its MLP - prefill::step_tp_unit):
//     for chunk h in (A, B):
//       compute  [u > 0: wait sent(peer, h, u - 1) and sent(self, h, u - 1)]
//                step_tp_unit(h, u): the fold of unit u - 1's halves, the norm, the unit's body -> own[h]
//                signal ready(self, h, u)
//       copy     wait ready(self, h, u); own[h] -> the peer's land[h][u & 1]; signal sent(self, h, u)
//
// so chunk A's crossing after unit u runs while the compute engine runs chunk B's unit u, and the reverse. Unit u of
// chunk B follows unit u of chunk A on the same in-order list - the GDN state, conv ring and KV of layer u / 2 are
// A's when B reads them, exactly as two consecutive chunks on one card. The last unit's half crosses only for the
// prompt's last chunk and only its last row (the head's). Two landing slots per chunk: the peer's send of unit u + 2
// into slot u & 1 waits (through its own fold of u + 1) for this card's send of u + 1, which this card issues after
// its fold of u has read the slot.
//
// Events: one pool of [card][chunk][unit][ready | sent], reset by the host between pairs (both cards' lists drained).
// Every host wait is bounded (TpOptions::timeout_ms); on a miss the host signals every event of the pair so the lists
// drain on whatever they hold, throws by name, and the engine stays failed until reset().
//
// Without a copy-only queue group the crossings go on the compute lists in order (correct, no overlap) - not this
// box (every B70 shows ordinal 1, COPY-only, one queue).
//
// **Spec 17d: the MTP head and the block hook.** With the head (whole on each card) every row's hidden is needed - the
// head's KV of a chunk is built over (h_{pos-1+r}, x[pos+r]) - so the last unit's halves cross for EVERY row of every
// chunk, and after the pair's units each card runs, per chunk in order, its tail on the compute list: the fold of the
// last unit's halves and the final norm of every row (prefill::step_tp_final) into the hidden rows (row 0 = h_{pos-1},
// copied from MtpBuffers::hh row 0), the head's KV (prefill::step_tp_mtp_kv, in passes the card's scratch holds), and
// the chunk's last hidden back into hh row 0 - Engine::prefill's per-chunk MTP steps. The prompt's head then reads the
// last row's normed hidden. With a block hook every pair ends on a multiple of the hook's stride (tp_hook_stride: a
// pair's rows rounded up to whole blocks) or at the prompt's end; the hook runs after such a pair has drained on both
// cards (Control pos = the end, n_active 0) and after the prompt's head, as Engine::prefill's.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <level_zero/ze_api.h>

#include "l0/error.h"
#include "l0/memory.h"
#include "l0/sync_event.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill/int8.h"
#include "runtime/prefill/profile.h"
#include "kernels/prefill/pf_kernels.h"
#include "runtime/prefill/step.h"
#include "runtime/tp_engine.h"
#include "runtime/tp_engine_card.h"
#include "runtime/tp_plan.h"

namespace runtime {
namespace {
constexpr uint32_t kPair = kTpPrefillChunks;   // chunks in flight
constexpr uint32_t kLandSlots = 2;             // landing slots per chunk (unit parity)
uint64_t ns(uint32_t ms) { return uint64_t(ms) * 1000000ull; }

// The copy-only queue group's ordinal (COPY without COMPUTE), or UINT32_MAX.
uint32_t copy_ordinal(ze_device_handle_t dev) {
  uint32_t n = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &n, nullptr));
  std::vector<ze_command_queue_group_properties_t> p(n);
  for (auto& q : p) q.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &n, p.data()));
  for (uint32_t i = 0; i < n; ++i)
    if ((p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
        !(p[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
      return i;
  return UINT32_MAX;
}
}  // namespace

// One card's prefill half.
struct TpPfCard {
  TpPfCard(l0::Context& c, const model::ModelDesc& d, uint32_t max_len, uint32_t cap)
      : ctx(c),
        cx(c),
        kc(c),
        s(c, max_len, d, cap),
        resid_b(c, l0::MemKind::Device, size_t(cap) * d.hidden * 2, 4096),
        own(c, l0::MemKind::Device, size_t(kPair) * cap * d.hidden * 4, 4096),
        land(c, l0::MemKind::Device, size_t(kPair) * kLandSlots * cap * d.hidden * 4, 4096),
        ctl(c, l0::MemKind::Host, kPair * sizeof(Control)),
        ids(c, l0::MemKind::Host, size_t(kPair) * cap * 4) {
    std::memset(ctl.ptr(), 0, ctl.size());
    const uint32_t ord = copy_ordinal(c.device());
    if (ord != UINT32_MAX) {
      ze_command_queue_desc_t qd{};
      qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
      qd.ordinal = ord;
      qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
      qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
      qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
      ZE_CHECK(zeCommandListCreateImmediate(c.handle(), c.device(), &qd, &bcs));
    }
  }
  ~TpPfCard() {
    if (bcs) {
      zeCommandListHostSynchronize(bcs, UINT64_MAX);
      zeCommandListDestroy(bcs);
    }
  }
  TpPfCard(const TpPfCard&) = delete;
  TpPfCard& operator=(const TpPfCard&) = delete;

  // Spec 17d: the MTP head's prefill buffers (allocated by TpEngine::prepare_prefill with the head): the hidden rows
  // bf16 [cap + 1][hidden] (row 0 = h_{pos-1}), the head's bf16 slab [2 x hidden][1024] (fc's K: the card's own slab
  // is [intermediate 8704][1024]) and its Controls, one per (chunk, pass) of a pair (host USM: written while idle).
  void make_mtp(uint32_t cap) {
    hid = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, (size_t(cap) + 1) * hidden() * 2);
    hslab = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(2) * hidden() * kernels::kPfSlabWidth * 2);
    hctl = std::make_unique<l0::Mem>(ctx, l0::MemKind::Host, size_t(kPair) * kHeadPasses * sizeof(Control));
    std::memset(hctl->ptr(), 0, hctl->size());
  }
  static constexpr uint32_t kHeadPasses = 2;   // prefill::tp_mtp_kv_passes at most: chunk rows / (cap / 2)
  Control* head_ctl(uint32_t h, uint32_t p) { return hctl->as<Control>() + h * kHeadPasses + p; }
  size_t hidden() const { return s.desc().hidden; }
  void* resid(uint32_t h) { return h == 0 ? s.resid.ptr() : resid_b.ptr(); }
  float* own_of(uint32_t h) { return own.as<float>() + size_t(h) * s.chunk_cap() * hidden(); }
  float* land_of(uint32_t h, uint32_t slot) {
    return land.as<float>() + (size_t(h) * kLandSlots + slot) * s.chunk_cap() * hidden();
  }
  Control* control(uint32_t h) { return ctl.as<Control>() + h; }
  uint32_t* ids_of(uint32_t h) { return ids.as<uint32_t>() + size_t(h) * s.chunk_cap(); }
  size_t device_bytes() const {
    return s.bytes() + s.lazy_bytes() + resid_b.size() + own.size() + land.size() + (hid ? hid->size() : 0) +
           (hslab ? hslab->size() : 0) + (int8 ? int8->bytes() : 0);
  }

  // The crossing's copy, on the copy engine after `ready`, or in order on the compute list.
  void send(void* dst, const void* src, size_t bytes, ze_event_handle_t ready, ze_event_handle_t sent) {
    if (bcs) {
      cx.signal(ready);
      ZE_CHECK(zeCommandListAppendMemoryCopy(bcs, dst, src, bytes, sent, 1, &ready));
      return;
    }
    cx.copy(dst, src, bytes);
    cx.signal(sent);
  }
  bool drain(uint64_t timeout_ns) {
    if (!cx.wait_for(timeout_ns)) return false;
    return !bcs || zeCommandListHostSynchronize(bcs, timeout_ns) == ZE_RESULT_SUCCESS;
  }

  l0::Context& ctx;
  prefill::Context cx;
  prefill::KernelCache kc;
  PrefillScratch s;
  std::unique_ptr<prefill::Int8State> int8;
  l0::Mem resid_b;   // chunk B's residual rows (chunk A's are s.resid): bf16 [cap][hidden]
  l0::Mem own;       // fp32 [kPair][cap][hidden]: this card's half of the last row-parallel output, per chunk
  l0::Mem land;      // fp32 [kPair][kLandSlots][cap][hidden]: the peer's halves, written only by the peer
  // Host USM (pipeline_prefill.cc's PfDevice::ctl reason): the host writes the pair's Controls and ids while nothing
  // runs, and the walk's attention reads pos / n_active per chunk.
  l0::Mem ctl;       // Control [kPair]
  l0::Mem ids;       // u32 [kPair][cap]
  ze_command_list_handle_t bcs = nullptr;   // the copy engine's list (null: crossings on the compute list)
  std::unique_ptr<l0::Mem> hid, hslab, hctl;   // spec 17d: the MTP head's (make_mtp)
};

struct TpPrefill {
  TpPrefill(l0::Context& d0, l0::Context& d1, const model::ModelDesc& d, uint32_t max_len, uint32_t cap)
      : units(2 * d.layers),
        ev(d0, {&d0, &d1}, model::kTpSize * kPair * 2 * d.layers * 2) {
    c[0] = std::make_unique<TpPfCard>(d0, d, max_len, cap);
    c[1] = std::make_unique<TpPfCard>(d1, d, max_len, cap);
  }
  ~TpPrefill() {
    // Never destroy a list that waits on an event nobody will signal (prefill::Context's destructor waits without a
    // bound): release every event first if a pair may be in flight.
    if (in_flight) release();
    for (auto& k : c)
      if (k) k->drain(ns(30000));
  }
  uint32_t ready(uint32_t r, uint32_t h, uint32_t u) const { return ((r * kPair + h) * units + u) * 2; }
  uint32_t sent(uint32_t r, uint32_t h, uint32_t u) const { return ready(r, h, u) + 1; }
  void reset_events() {
    for (uint32_t i = 0; i < ev.size(); ++i) ev.host_reset(i);
  }
  void release() {
    for (uint32_t i = 0; i < ev.size(); ++i) {
      try {
        ev.host_signal(i);
      } catch (...) {
      }
    }
  }
  bool overlaps() const { return c[0]->bcs && c[1]->bcs; }

  uint32_t units;
  l0::SyncEventPool ev;
  bool in_flight = false;
  // Declared last, destroyed first: the lists go before the events they name.
  std::array<std::unique_ptr<TpPfCard>, model::kTpSize> c;
};

namespace {
void destroy_tp_prefill(TpPrefill* p) { delete p; }

// One pair of the prompt: chunk h's absolute position and rows (rows 0: no chunk B).
struct PairPlan {
  uint32_t pos[kPair] = {0, 0};
  uint32_t rows[kPair] = {0, 0};
};
}  // namespace

void TpEngine::set_prefill_backend(PrefillBackend backend) {
  if (pf_ && backend != pf_backend_)
    throw std::logic_error("runtime::TpEngine::set_prefill_backend: set the backend before prepare_prefill()");
  pf_backend_ = backend;
}

void TpEngine::set_prefill_chunk_cap(uint32_t cap) {
  check_prefill_chunk_cap(*cd(0).model.desc, cap);
  if (pf_ && pf_->c[0]->s.chunk_cap() != cap)
    throw std::logic_error("runtime::TpEngine::set_prefill_chunk_cap: the prefill scratch already holds " +
                           std::to_string(pf_->c[0]->s.chunk_cap()) + "-row chunks; set the cap (" +
                           std::to_string(cap) + ") before prepare_prefill()");
  pf_cap_ = cap;
}

size_t TpEngine::prefill_launches() const {
  return pf_ ? pf_->c[0]->cx.launches() + pf_->c[1]->cx.launches() : 0;
}

bool TpEngine::prefill_overlaps() const { return pf_ && pf_->overlaps(); }

void TpEngine::prepare_prefill() {
  if (pf_) return;
  if (prefill::profile_enabled())
    throw std::runtime_error("--parallel tp --prefill: B70_PREFILL_PROFILE's per-phase waits would stall one card on "
                             "a crossing the other card's list has not been given yet; profile one card instead");
  if (prefill::attn_mode() != prefill::AttnMode::Flash)
    throw std::runtime_error("runtime::TpEngine::prefill: a TP card runs the flash attention only");
  const model::ModelDesc& d = *cd(0).model.desc;
  check_prefill_chunk_cap(d, pf_cap_);
  pf_ = std::unique_ptr<TpPrefill, void (*)(TpPrefill*)>(
      new TpPrefill(cd(0).ctx, cd(1).ctx, d, max_len_, pf_cap_), &destroy_tp_prefill);
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    TpPfCard& k = *pf_->c[r];
    if (kv_ == KvCache::Int8) k.s.kv8_raw_buffer();   // spec 12b's widening scratch, at the card's 2 kv heads
    if (pf_backend_ == PrefillBackend::L0Int8) {
      k.int8 = std::make_unique<prefill::Int8State>(k.ctx, prefill_int8_max_k(d), pf_cap_);
      for (const auto& [key, w] : cd(r).model.linears) {
        if (key.second == model::LinearId::LmHead || key.second == model::LinearId::AB) continue;
        if (l0_int8_bf16_linear(d, key.second)) continue;   // open-work 17: the mixer linears are the l0 walk's
        if (w.shape.N % kernels::kPfSlabWidth != 0)
          throw std::runtime_error("runtime::TpEngine::prepare_prefill: int8 N is not whole 1024-column slabs");
        k.int8->scales(k.cx, k.kc, w);
      }
    }
    k.s.slab_buffer();   // the l0 linears' slab (l0-int8's mixer linears too): outside the first timed chunk
    if (mtp_) k.make_mtp(pf_cap_);   // spec 17d
    pf_int8_bytes_[r] = k.int8 ? k.int8->bytes() : 0;
    pf_bytes_[r] = k.device_bytes() - pf_int8_bytes_[r];
  }
}

void TpEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  if (broken_) throw std::runtime_error("tensor parallel: a crossing failed earlier in this session; reset() first");
  if (ids.empty()) throw std::runtime_error("runtime::TpEngine::prefill: no ids");
  if (chunk == 0) chunk = pf_cap_;
  if (chunk > pf_cap_)
    throw std::runtime_error("runtime::TpEngine::prefill: chunk " + std::to_string(chunk) +
                             " exceeds the session's prefill chunk cap " + std::to_string(pf_cap_));
  if (chunk % 64 != 0)
    throw std::runtime_error("runtime::TpEngine::prefill: chunk " + std::to_string(chunk) +
                             " is not a multiple of 64 (a pair splits on the 64-grid)");
  for (uint32_t id : ids)
    if (id >= model::Qwen35::kVocab)
      throw std::runtime_error("runtime::TpEngine::prefill: id " + std::to_string(id) + " is outside the vocabulary (" +
                               std::to_string(model::Qwen35::kVocab) + " rows)");
  const uint32_t base = cd(0).ctl->pos;
  if (size_t(base) + ids.size() > size_t(max_len_))
    throw std::runtime_error("runtime::TpEngine::prefill: pos " + std::to_string(base) + " + " +
                             std::to_string(ids.size()) + " ids exceeds max_len " + std::to_string(max_len_) +
                             " - the KV cache and the RoPE table stop there. Start a new session with reset().");
  prepare_prefill();
  // A decode step left nothing running (step_once settles both fences), so the prefill lists start on its state.
  for (uint32_t r = 0; r < model::kTpSize; ++r)
    if (!settle(r)) fail("card " + std::to_string(r) + "'s last decode step is still running");

  TpPrefill& p = *pf_;
  const model::ModelDesc& d = *cd(0).model.desc;
  const uint32_t U = p.units;
  const size_t row = size_t(d.hidden) * 4;
  // Spec 17d: the prefill GDN path reads slot 0, so a live slot left by a commit goes there first; with a hook every
  // pair ends on the hook's stride or the prompt's end.
  if (mtp_) {
    mtp_normalise_live();
    verify_k_ = kNoVerify;
  }
  const bool hooked = static_cast<bool>(block_hook_);
  const std::vector<std::array<uint32_t, 2>> chunks =
      tp_prefill_chunks(base, ids.size(), chunk, hooked ? tp_hook_stride(chunk, kBlock) : 0);
  for (size_t j = 0; j < chunks.size(); j += kPair) {
    PairPlan pp;
    const uint32_t n_ch = uint32_t(std::min<size_t>(kPair, chunks.size() - j));
    for (uint32_t h = 0; h < n_ch; ++h) {
      pp.pos[h] = chunks[j + h][0];
      pp.rows[h] = chunks[j + h][1];
    }
    const bool last_pair = j + n_ch == chunks.size();
    const uint32_t last_h = n_ch - 1;
    // Both cards' lists are idle (the previous pair was waited for): the pair's events, Controls and ids.
    p.reset_events();
    for (uint32_t r = 0; r < model::kTpSize; ++r)
      for (uint32_t h = 0; h < n_ch; ++h) {
        TpPfCard& k = *p.c[r];
        k.control(h)->pos = pp.pos[h];
        k.control(h)->n_active = pp.rows[h];
        std::memcpy(k.ids_of(h), ids.data() + (pp.pos[h] - base), size_t(pp.rows[h]) * 4);
      }
    p.in_flight = true;
    for (uint32_t u = 0; u < U; ++u)
      for (uint32_t h = 0; h < n_ch; ++h)
        for (uint32_t r = 0; r < model::kTpSize; ++r) {
          TpPfCard& k = *p.c[r];
          TpPfCard& peer = *p.c[1 - r];
          Card& card = cd(r);
          const uint32_t C = pp.rows[h];
          prefill::TpUnitIo io;
          io.ids = k.ids_of(h);
          io.resid = k.resid(h);
          io.own = k.own_of(h);
          const bool cross = !opt_.prefill_timing_no_crossing;
          if (u > 0) {
            // The previous unit's halves: this card's copy has read own[h], the peer's has filled land[h].
            if (cross) {
              k.cx.wait_event(p.ev.handle(p.sent(1 - r, h, u - 1)));
              k.cx.wait_event(p.ev.handle(p.sent(r, h, u - 1)));
            }
            const float* mine = k.own_of(h);
            const float* theirs = k.land_of(h, (u - 1) & 1u);
            io.p0 = r == 0 ? mine : theirs;
            io.p1 = r == 0 ? theirs : mine;
          }
          prefill::step_tp_unit(k.cx, k.kc, k.s, card.model, max_len_, k.control(h), pp.pos[h], C, u,
                                card.persist.gdn_state, card.persist.conv_ring, card.persist.kv_k, card.persist.kv_v,
                                card.persist.kv_lay, io, pf_backend_, k.int8.get());
          if (!cross) continue;
          if (u + 1 < U || mtp_) {   // spec 17d: with the head the last unit's every row crosses too
            k.send(peer.land_of(h, u & 1u), k.own_of(h), size_t(C) * row, p.ev.handle(p.ready(r, h, u)),
                   p.ev.handle(p.sent(r, h, u)));
          } else if (last_pair && h == last_h) {
            // The last unit: only the head's row crosses.
            const size_t at = size_t(C - 1) * d.hidden;
            k.send(peer.land_of(h, u & 1u) + at, k.own_of(h) + at, row, p.ev.handle(p.ready(r, h, u)),
                   p.ev.handle(p.sent(r, h, u)));
          }
        }
    if (mtp_) {
      // Spec 17d: each card's MTP tail of every chunk, in order (the header's **Spec 17d**).
      const size_t hrow = size_t(d.hidden) * 2;
      for (uint32_t h = 0; h < n_ch; ++h)
        for (uint32_t r = 0; r < model::kTpSize; ++r) {
          TpPfCard& k = *p.c[r];
          Card& card = cd(r);
          if (!opt_.prefill_timing_no_crossing) {
            k.cx.wait_event(p.ev.handle(p.sent(1 - r, h, U - 1)));
            k.cx.wait_event(p.ev.handle(p.sent(r, h, U - 1)));
          }
          const uint32_t C = pp.rows[h], pos = pp.pos[h];
          const float* mine = k.own_of(h);
          const float* theirs = k.land_of(h, (U - 1) & 1u);
          uint16_t* hid = k.hid->as<uint16_t>();
          k.cx.copy(hid, card.mtp->hh.ptr(), hrow);   // h_{pos-1} into row 0
          prefill::step_tp_final(k.cx, k.kc, k.s, card.model, r == 0 ? mine : theirs, r == 0 ? theirs : mine,
                                 k.resid(h), hid, C);
          void* hctl[TpPfCard::kHeadPasses] = {nullptr, nullptr};
          const uint32_t passes = prefill::tp_mtp_kv_passes(pos, C, k.s);
          if (passes > TpPfCard::kHeadPasses)
            fail("the MTP head's prefill needs " + std::to_string(passes) + " passes a chunk (built for " +
                 std::to_string(TpPfCard::kHeadPasses) + ")");
          for (uint32_t q = 0; q < passes; ++q) {
            Control* c = k.head_ctl(h, q);
            c->pos = (pos == 0 ? 0 : pos - 1) + q * prefill::tp_mtp_kv_pass(k.s);
            c->n_active = std::min(prefill::tp_mtp_kv_pass(k.s), prefill::mtp_kv_rows(pos, C) - q * prefill::tp_mtp_kv_pass(k.s));
            hctl[q] = c;
          }
          const MtpBuffers& m = *card.mtp;
          prefill::step_tp_mtp_kv(k.cx, k.kc, k.s, card.model, model::qwen38(), hctl, *k.hslab, k.ids_of(h), pos, C,
                                  hid, m.kv_lay.layer(m.kv_k.ptr(), m.kv_v.ptr(), 0));
          k.cx.copy(card.mtp->hh.ptr(), hid + size_t(C) * d.hidden, hrow);   // h_{end-1}: the next chunk's row 0
        }
    }
    if (last_pair) {
      // The head: the last row's fold, the final norm, the card's vocabulary half, its argmax into the card's
      // Control (pos = the last position, n_active 1: tp_argmax_stage2 leaves pos = base + n). Spec 17d: with the
      // MTP head the fold and the norm ran in the last chunk's tail - lm_head reads its last hidden row.
      const uint32_t C = pp.rows[last_h];
      const size_t at = size_t(C - 1) * d.hidden;
      const uint32_t rows = cd(0).model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
      for (uint32_t r = 0; r < model::kTpSize; ++r) {
        cd(r).ctl->pos = base + uint32_t(ids.size()) - 1;
        cd(r).ctl->n_active = 1;
      }
      for (uint32_t r = 0; r < model::kTpSize; ++r) {
        TpPfCard& k = *p.c[r];
        if (!opt_.prefill_timing_no_crossing) {
          k.cx.wait_event(p.ev.handle(p.sent(1 - r, last_h, U - 1)));
          k.cx.wait_event(p.ev.handle(p.sent(r, last_h, U - 1)));
        }
        const float* mine = k.own_of(last_h) + at;
        const float* theirs = k.land_of(last_h, (U - 1) & 1u) + at;
        const uint32_t first = r * rows;
        const uint32_t used = d.vocab_used > first ? std::min(rows, d.vocab_used - first) : 0;
        prefill::step_tp_head(k.cx, k.kc, k.s, cd(r).model, cd(r).ctl, r == 0 ? mine : theirs,
                              r == 0 ? theirs : mine, static_cast<uint8_t*>(k.resid(last_h)) + at * 2, used, first,
                              mtp_ ? k.hid->as<uint16_t>() + size_t(C) * d.hidden : nullptr);
      }
    }
    for (uint32_t r = 0; r < model::kTpSize; ++r)
      if (!p.c[r]->drain(ns(opt_.timeout_ms))) {
        p.release();
        p.c[1 - r]->drain(ns(opt_.timeout_ms));
        p.c[r]->drain(ns(opt_.timeout_ms));
        fail("card " + std::to_string(r) + "'s prefill of the pair at position " + std::to_string(pp.pos[0]) +
             " did not finish within " + std::to_string(opt_.timeout_ms) + " ms");
      }
    p.in_flight = false;
    // Spec 17d: a mid-prompt pair that ends on a block end - the state is exactly at its end on both cards.
    const uint32_t end = pp.pos[last_h] + pp.rows[last_h];
    if (hooked && !last_pair && end % kBlock == 0) {
      for (uint32_t r = 0; r < model::kTpSize; ++r) {
        cd(r).ctl->pos = end;
        cd(r).ctl->n_active = 0;
      }
      block_hook_(end, true);
    }
  }
  merge_heads();
  if (hooked) {
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
}

}  // namespace runtime
