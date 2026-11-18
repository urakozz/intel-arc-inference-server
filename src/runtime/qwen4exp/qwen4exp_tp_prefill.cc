// Qwen4ExpTpEngine::prefill and its accessors - plan 21v2b Task 4, compiled into b70_qwen4exp_prefill (the one-card
// engine's arrangement: qwen4exp_prefill_engine.cc), so a decode-only target links what it always linked.
//
// Each pair of chunks (plan 21-QFN-v2b Task 4 Step 2; the prompt's last may be one): the host (both cards' prefill
// lists idle) resets the pair's crossing events, writes each chunk's ids and Control (the card's, at pos / C) on each
// card, appends both cards' walks a unit at a time (prefill_pair_tp: Qwen3.8's wavefront - a chunk's crossing after
// unit u on the card's copy-engine list beside the other chunk's unit u; each wait appended after the signal it waits
// for), and waits on both with a bound. A pair's events and fold slots are free once both lists are idle. Without a
// copy-only queue group the crossings go on the compute lists (correct, no overlap). A card that does not finish in time:
// every event is host-signalled so both lists drain on whatever they hold, the engine throws naming the card, the chunk
// and each card's crossings sent, and stays failed until reset(). After the last chunk both cards run prefill_head_tp
// on its last row and the host merges the heads (the larger value, ties to card 0) as a decode step does.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "l0/error.h"
#include "l0/sync_event.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_chunks.h"
#include "runtime/qwen4exp/qwen4exp_prefill.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp/qwen4exp_tp_card.h"
#include "runtime/qwen4exp/qwen4exp_tp_engine.h"

namespace runtime::qwen4exp {

struct Q4TpPrefillState {
  struct Dev {
    Dev(l0::Context& c, const model::Qwen4ExpDesc& card, const model::Qwen4ExpDesc& one, uint32_t max_len)
        : cx(c),
          kc(c),
          s(c, card, max_len),
          fold(c, l0::MemKind::Device, size_t(kTpPfChunks) * 2 * 2 * kPfC * one.hidden * 4, 4096),
          yf(c, l0::MemKind::Device, size_t(pf_tiles(one, kPfC)) * kPfTm * one.hidden * 4, 4096),
          H1(c, l0::MemKind::Device, size_t(kPfC) * one.hc_n() * 2),
          inj1(c, l0::MemKind::Device, size_t(kPfC) * one.hc * 4),
          lists1(c, l0::MemKind::Device, size_t(kPfC) * kListRow * 4),
          ctl(c, l0::MemKind::Host, kTpPfChunks * sizeof(Control)),
          rctl(c, l0::MemKind::Host, kTpPfChunks * sizeof(Control)) {
      std::memset(ctl.ptr(), 0, ctl.size());
      std::memset(rctl.ptr(), 0, rctl.size());
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
    ~Dev() {
      if (bcs) {
        zeCommandListHostSynchronize(bcs, UINT64_MAX);
        zeCommandListDestroy(bcs);
      }
    }
    // chunk h's slot k: fp32 [2 slices][kPfC][hidden]
    size_t slot_off(uint32_t h, uint32_t k) const { return (size_t(h) * 2 + k) * 2 * kPfC * hidden; }
    Control* ctl_at(uint32_t h) { return ctl.as<Control>() + h; }
    Control* rctl_at(uint32_t h) { return rctl.as<Control>() + h; }
    uint32_t* ids_at(uint32_t h) { return s.ids.as<uint32_t>() + size_t(h) * kPfC; }   // s.ids holds kPfDepth chunks
    prefill::Context cx;
    prefill::KernelCache kc;
    Qwen4ExpPrefillScratch s;
    l0::Mem fold;       // fp32 [chunk][2 slots][2 slices][kPfC][hidden]
    l0::Mem yf;         // fp32 [tmax x 32][hidden]
    l0::Mem H1, inj1;   // the pair's second chunk's residual rows and inject weights (the first's are s.H / s.inj)
    l0::Mem lists1;     // its selection rows of one QSA layer (the indexer split by rows; the first's: s.lists)
    l0::Mem ctl;        // host: each chunk's Control
    l0::Mem rctl;       // host: each chunk's row Control (the row split: this card's first row and rows)
    size_t hidden = 0;
    std::array<Q4TpPrefill, kTpPfChunks> tp;
    ze_command_list_handle_t bcs = nullptr;   // the copy engine's list (null: crossings on the compute list)
    // Task 5, the MTP head's card: the head pass's R rows, its Control for the chunk (host), its own rows at one
    // card's widths (Qwen4ExpMtpPrefill::partials / q16)
    std::unique_ptr<l0::Mem> mtp_R, hctl, mtp_part, mtp_q16;
    std::unique_ptr<Q4PfRingState> ring;   // the host-memory experts read ahead (null: no expert table)
    std::unique_ptr<Q4LendState> lend;     // the scratch lent to the tier between prefills (with the ring)
  };
  // [chunk][sent card 0 | sent card 1 | ready card 0 | ready card 1 | the same four of the row split's lists][unit]
  Q4TpPrefillState(l0::Context& d0, l0::Context& d1, uint32_t units)
      : ev(d0, {&d0, &d1}, kTpPfChunks * 8 * units), units(units) {}
  ~Q4TpPrefillState() {   // never free a scratch under a running list
    release();
    for (auto& d : dev)
      if (d) d->cx.wait_for(timeout_ns);
  }
  void release() {
    for (uint32_t i = 0; i < ev.size(); ++i) {
      try {
        ev.host_signal(i);
      } catch (...) {
      }
    }
  }
  void reset_events() {
    for (uint32_t i = 0; i < ev.size(); ++i) ev.host_reset(i);
    for (auto& d : dev)
      if (d && d->ring) d->ring->reset();
  }
  // reset()'s hook: release a list waiting on a crossing that will not come; true when both lists are idle.
  bool settle() {
    release();
    bool ok = true;
    for (auto& d : dev) ok = d->cx.wait_for(timeout_ns) && ok;
    reset_events();
    return ok;
  }
  l0::SyncEventPool ev;
  uint32_t units;
  uint64_t timeout_ns = 0;
  std::unique_ptr<Dev> dev[2];   // declared after the events: the lists go before the events they name
};

namespace {
void destroy_tp_prefill(Q4TpPrefillState* p) { delete p; }
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::qwen4exp::Qwen4ExpTpEngine::prefill: " + what);
}
}  // namespace

void Qwen4ExpTpEngine::prepare_prefill() {
  if (pf_) return;
  const model::Qwen4ExpDesc& d = model_.desc;
  for (const std::string& v : kernels::qwen4exp::tp_prefill_variants(d, model_.ple.scale == loader::Q4PleScale::Bf16))
    require(std::ifstream(kernels::path(v)).good(),
            v + " is not compiled (" + kernels::path(v) + "): build with B70_Q4EXP=ON and B70_TP=ON");
  require(d.forms.shared == model::Q4Form::Int4,
          "tensor parallel prefill runs the int4 shared expert (our export's); this checkpoint's is bf16");
  auto* ps = new Q4TpPrefillState(cards_[0]->ctx, cards_[1]->ctx, 2 * d.layers);
  pf_ = std::unique_ptr<Q4TpPrefillState, void (*)(Q4TpPrefillState*)>(ps, &destroy_tp_prefill);
  ps->timeout_ns = uint64_t(opt_.timeout_ms) * 1000000ull;
  for (uint32_t r = 0; r < 2; ++r) {
    ps->dev[r] = std::make_unique<Q4TpPrefillState::Dev>(cards_[r]->ctx, cards_[r]->desc, d, max_len_);
    ps->dev[r]->hidden = d.hidden;
    ps->dev[r]->ring = make_pf_ring(cards_[r]->ctx, model_.parts[r], cards_[r]->desc);
    require(ps->dev[r]->s.bytes() + ps->dev[r]->fold.size() + ps->dev[r]->yf.size() + ps->dev[r]->H1.size() +
                    ps->dev[r]->inj1.size() + ps->dev[r]->lists1.size() + (ps->dev[r]->ring ? pf_ring_bytes(d) : 0) ==
                prefill_sizes(cards_[r]->desc, max_len_).total() + tp_prefill_bytes(d),
            "card " + std::to_string(r) + "'s prefill allocation is not the plan's (prefill_sizes + tp_prefill_bytes)");
  }
  if (mtp_) {   // Task 5: the head's pass on card 1
    for (const std::string& v : kernels::qwen4exp::mtp_prefill_variants(mtp_bind_.single))
      require(std::ifstream(kernels::path(v)).good(), v + " is not compiled (" + kernels::path(v) + "): build with B70_MTP=ON");
    Q4TpPrefillState::Dev& D = *ps->dev[1];
    l0::Context& c1 = cards_[1]->ctx;
    D.mtp_R = std::make_unique<l0::Mem>(c1, l0::MemKind::Device, mtp_prefill_R_bytes(d));
    D.hctl = std::make_unique<l0::Mem>(c1, l0::MemKind::Host, kTpPfChunks * sizeof(Control));
    D.mtp_part = std::make_unique<l0::Mem>(c1, l0::MemKind::Device, size_t(kPfC) * pf_ld(d.qkvg_n()) * 4);
    D.mtp_q16 = std::make_unique<l0::Mem>(c1, l0::MemKind::Device, size_t(kPfC) * d.q_n() * 2);
  }
  const bool overlap = ps->dev[0]->bcs && ps->dev[1]->bcs;   // both cards' copy engines, or neither crosses there
  for (uint32_t r = 0; r < 2; ++r) {
    Q4TpPrefillState::Dev& D = *ps->dev[r];
    Q4TpPrefillState::Dev& P = *ps->dev[1 - r];
    for (uint32_t h = 0; h < kTpPfChunks; ++h) {
      Q4TpPrefill& t = D.tp[h];
      t.rank = r;
      t.card = &cards_[r]->desc;
      t.embed = cards_[r]->binding.embed;
      t.ple_ptrs = r == 1 ? model_.ple_ptrs1->ptr() : model_.ple.ptrs->ptr();
      for (uint32_t k = 0; k < 2; ++k) {
        t.fold[k] = D.fold.as<float>() + D.slot_off(h, k);
        t.peer_fold[k] = P.fold.as<float>() + P.slot_off(h, k);
      }
      t.yf = D.yf.as<float>();
      t.ev = &ps->ev;
      t.units = ps->units;
      t.ev_off = h * 8 * ps->units;
      t.H = h ? D.H1.ptr() : nullptr;
      t.inj = h ? D.inj1.ptr() : nullptr;
      t.bcs = overlap ? D.bcs : nullptr;
      t.rctl = D.rctl_at(h);
      t.lists = h ? D.lists1.ptr() : nullptr;
      t.peer_lists = h ? P.lists1.ptr() : P.s.lists.ptr();
    }
  }
  for (uint32_t r = 0; r < 2; ++r) {   // the lent experts: the scratch, the pair's own rows and the ring's buffers
    Q4TpPrefillState::Dev& D = *ps->dev[r];
    if (!D.ring) continue;
    std::vector<Q4LendState::Region> rg = lend_regions(D.s, D.ring.get());
    for (const l0::Mem* m : {&D.fold, &D.yf}) rg.push_back({m->ptr(), m->size()});
    D.lend = std::make_unique<Q4LendState>(cards_[r]->ctx, model_.parts[r], cards_[r]->desc, r ? model_.tier1 : model_.tier,
                                           rg, D.ring->list);
    if (D.lend->experts() > 0) {
      const ExpertTier& t = r ? model_.tier1 : model_.tier;
      size_t res = 0;
      for (uint32_t l = 0; l < d.layers; ++l) res += t.n_vram(l);
      const double all = double(d.layers) * d.experts;
      std::fprintf(stderr, "experts: card %u lends its idle prefill scratch to %u more experts' halves between prefills "
                   "(resident f %.3f -> %.3f while decoding)\n", r, D.lend->experts(), res / all,
                   (res + D.lend->experts()) / all);
    }
  }
  pf_lend_ = [ps] {
    for (auto& D : ps->dev)
      if (D->lend) D->lend->lend();
  };
  pf_settle_ = [ps] { return ps->settle(); };
}

double Qwen4ExpTpEngine::read_host_experts() {
  prepare_prefill();
  const auto t0 = std::chrono::steady_clock::now();
  for (auto& D : pf_->dev)
    if (D->ring) D->ring->read_all();
  for (auto& D : pf_->dev)
    if (D->ring) ZE_CHECK(zeCommandListHostSynchronize(D->ring->list, UINT64_MAX));
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

size_t Qwen4ExpTpEngine::prefill_launches() const {
  return pf_ ? pf_->dev[0]->cx.launches() + pf_->dev[1]->cx.launches() : 0;
}

std::vector<uint32_t> Qwen4ExpTpEngine::read_prefill_routes() {
  require(bool(pf_), "read_prefill_routes before the first prefill");
  Q4TpPrefillState::Dev& D = *pf_->dev[0];
  require(D.cx.wait_for(pf_->timeout_ns), "card 0's prefill list is still running");
  std::vector<uint32_t> v(size_t(model_.desc.layers) * kPfC * kRouteWords);
  cards_[0]->imm.copy(v.data(), D.s.routes.ptr(), v.size() * 4);
  return v;
}

void Qwen4ExpTpEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (broken_) throw std::runtime_error("tensor parallel: a crossing failed earlier in this session; reset() first");
  require(!ids.empty(), "no ids");
  if (chunk == 0) chunk = kPfC;
  require(chunk <= kPfC, "chunk " + std::to_string(chunk) + " exceeds kPfC " + std::to_string(kPfC));
  for (uint32_t id : ids) require(id < d.vocab, "id " + std::to_string(id) + " is outside the vocabulary");
  const uint32_t base = cards_[0]->ctl->pos;
  require(size_t(base) + ids.size() <= max_len_,
          "pos " + std::to_string(base) + " + " + std::to_string(ids.size()) + " ids exceeds max_len " +
              std::to_string(max_len_) + " - the KV, the indexer keys and the RoPE table stop there");
  for (uint32_t r = 0; r < 2; ++r) {   // no decode step in flight writes the state or the Control
    Card& c = *cards_[r];
    if (c.pending && !c.fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull))
      fail("card " + std::to_string(r) + "'s last decode step is still running");
    c.pending = false;
  }
  // Task 5: the prefill's GDN path reads slot 0 directly, so a live slot a commit left is copied there first; a verify
  // waiting for its commit is dropped (the prompt continues from pos).
  if (mtp_) {
    mtp_normalise_live();
    verify_k_ = kNoVerify;
  }
  prepare_prefill();
  Q4TpPrefillState& P = *pf_;
  for (auto& D : P.dev)   // the lent experts' rows back: this prefill overwrites the scratch
    if (D->lend) D->lend->reclaim();
  std::array<Qwen4ExpMtpPrefill, kTpPfChunks> mp;   // the head's pass on card 1, a chunk each (qwen4exp_prefill.h)
  if (mtp_)
    for (uint32_t h = 0; h < kTpPfChunks; ++h) {
      Q4TpPrefillState::Dev& D = *P.dev[1];
      mp[h].bufs = mtp_.get();
      mp[h].R = D.mtp_R.get();   // one: the pair's passes run in order on card 1's list
      mp[h].embed = mtp_bind_.embed;
      mp[h].single = mtp_bind_.single;
      mp[h].hctl = D.hctl->as<Control>() + h;
      mp[h].partials = D.mtp_part.get();
      mp[h].q16 = D.mtp_q16.get();
    }
  const uint64_t tmo = P.timeout_ns;
  // A wait past 5 s says, every 5 s, where each card is: each chunk's crossings sent, the read ahead's layers staged
  // and released, and which of the card's lists are still running.
  const auto wait_logged = [&](uint32_t r, uint32_t pos, const char* what) {
    constexpr uint64_t kSlice = 5000000000ull;
    for (uint64_t waited = 0;;) {
      const uint64_t s = std::min(kSlice, tmo - waited);
      if (P.dev[r]->cx.wait_for(s)) {
        if (waited) std::fprintf(stderr, "prefill: card %u's %s at position %u finished after %.1f s\n", r, what, pos,
                                 double(waited + s) / 1e9);
        return true;
      }
      waited += s;
      std::string at;
      for (uint32_t c = 0; c < 2; ++c) {
        Q4TpPrefillState::Dev& D = *P.dev[c];
        uint32_t sent[kTpPfChunks] = {};
        for (uint32_t h = 0; h < kTpPfChunks; ++h)
          while (sent[h] < P.units && P.ev.signalled(h * 8 * P.units + c * P.units + sent[h])) ++sent[h];
        uint32_t staged = 0, released = 0;
        if (D.ring)
          for (uint32_t i = 0; 2 * i + 1 < D.ring->ev->size(); ++i) {
            staged += D.ring->ev->signalled(2 * i);
            released += D.ring->ev->signalled(2 * i + 1);
          }
        const auto idle = [](ze_command_list_handle_t l) {
          return l == nullptr ? "-" : zeCommandListHostSynchronize(l, 0) == ZE_RESULT_SUCCESS ? "idle" : "running";
        };
        at += std::format("; card {}: sent {} / {} of {}, layers staged {} released {}, compute {}, crossing copies {}, "
                          "read ahead {}", c, sent[0], sent[1], P.units, staged, released,
                          D.cx.wait_for(0) ? "idle" : "running", idle(D.bcs), idle(D.ring ? D.ring->list : nullptr));
      }
      std::fprintf(stderr, "prefill: card %u's %s at position %u still running after %.0f s%s\n", r, what, pos,
                   double(waited) / 1e9, at.c_str());
      if (waited >= tmo) return false;
    }
  };
  const auto drain = [&](uint32_t pos, const char* what) {
    for (uint32_t r = 0; r < 2; ++r)
      if (!wait_logged(r, pos, what)) {
        std::string got;   // each card's crossings signalled so far
        for (uint32_t c = 0; c < 2; ++c) {
          uint32_t n = 0;
          while (n < P.units && P.ev.signalled(c * P.units + n)) ++n;
          got += (c ? ", card 1 " : " (crossings sent: card 0 ") + std::to_string(n) + " of " + std::to_string(P.units);
        }
        P.release();
        P.dev[1 - r]->cx.wait_for(tmo);
        P.dev[r]->cx.wait_for(tmo);
        fail("card " + std::to_string(r) + "'s prefill " + what + " at position " + std::to_string(pos) +
             " did not finish within " + std::to_string(opt_.timeout_ms) + " ms" + got + ")");
      }
  };
  // Spec 7's prefix cache: with a block hook every chunk ends at a block end or at the prompt end (the one-card
  // engine's rule, one shared function); the hook runs after each pair that ends mid-prompt on a block end, with both
  // cards idle (Qwen3.8's tp rule: the block ends inside a pair are not hooked), and at the prompt's end.
  const bool hooked = static_cast<bool>(block_hook_);
  uint32_t last_C = 0, last_h = 0;
  for (size_t off = 0; off < ids.size();) {
    uint32_t pos[kTpPfChunks], C[kTpPfChunks], n = 0;
    for (size_t o = off; n < kTpPfChunks && o < ids.size(); ++n) {
      pos[n] = base + uint32_t(o);
      C[n] = prefill_chunk_rows(pos[n], ids.size() - o, chunk, hooked, kBlock);
      o += C[n];
    }
    P.reset_events();
    std::array<std::array<Q4TpChunk, 2>, 2> k{};
    std::array<std::array<Q4PfRing, 2>, 2> rv{};   // the read ahead: chunk A copies the first layers, the last releases
    const bool ring = n == 2;   // a chunk alone reads its experts in place (kPfRingMinRows)
    for (uint32_t h = 0; h < n; ++h)
      for (uint32_t r = 0; r < 2; ++r)
        if (P.dev[r]->ring && ring) rv[h][r] = P.dev[r]->ring->view(h == 0, h + 1 == n);
    for (uint32_t h = 0; h < n; ++h) {
      for (uint32_t r = 0; r < 2; ++r) {
        Q4TpPrefillState::Dev& D = *P.dev[r];
        std::memcpy(D.ids_at(h), ids.data() + (pos[h] - base), size_t(C[h]) * 4);
        Control* c = D.ctl_at(h);
        *c = *cards_[r]->ctl;
        c->pos = pos[h];
        c->n_active = C[h];
        Control* rc = D.rctl_at(h);   // the row split's: this card's rows of the chunk
        *rc = *c;
        rc->pos = pos[h] + (r == 1 ? tp_rows_cut(C[h]) : 0);
        rc->n_active = r == 1 ? C[h] - tp_rows_cut(C[h]) : tp_rows_cut(C[h]);
        k[h][r] = {&D.cx, &D.kc, &D.s, &model_.parts[r], &cards_[r]->buffers, c, D.ids_at(h), &D.tp[h],
                   mtp_ && r == 1 ? &mp[h] : nullptr, D.ring && ring ? &rv[h][r] : nullptr};
      }
      if (mtp_) {   // the head's pass covers positions pos - 1 .. pos + C - 2 (from 0 at pos 0)
        Control* hc = P.dev[1]->hctl->as<Control>() + h;
        *hc = *mtp_->hctl.as<Control>();
        hc->pos = pos[h] == 0 ? 0 : pos[h] - 1;
        hc->n_active = mtp_prefill_rows(pos[h], C[h]);
      }
    }
    try {
      prefill_pair_tp(model_, k, pos, C, n);
    } catch (...) {   // a walk's own error (a missing binary, a shape check): release what waits, then throw
      P.release();
      for (auto& d : P.dev) d->cx.wait_for(tmo);
      broken_ = true;
      throw;
    }
    drain(pos[0], n == 2 ? "pair" : "chunk");
    const uint32_t end = pos[n - 1] + C[n - 1];
    off = end - base;
    last_C = C[n - 1];
    last_h = n - 1;
    if (hooked && off < ids.size() && end % kBlock == 0) {
      for (auto& c : cards_) {
        c->ctl->pos = end;
        c->ctl->n_active = 0;
      }
      block_hook_(end, true);
    }
  }
  if (mtp_) {   // the head's Control as the last chunk left it
    const Control& h = *(P.dev[1]->hctl->as<Control>() + last_h);
    mtp_->hctl.as<Control>()->pos = h.pos;
    mtp_->hctl.as<Control>()->n_active = h.n_active;
  }
  // The head on both cards: the last unit's fold slot holds the last row's two partials.
  for (uint32_t r = 0; r < 2; ++r) {
    cards_[r]->ctl->pos = base + uint32_t(ids.size()) - 1;
    cards_[r]->ctl->n_active = 1;
  }
  const uint32_t last_slot = (P.units - 1) & 1u;
  for (uint32_t r = 0; r < 2; ++r) {
    Q4TpPrefillState::Dev& D = *P.dev[r];
    const Q4TpPrefill& t = D.tp[last_h];
    const void* R_last = mtp_ && r == 1 ? static_cast<const uint8_t*>(D.mtp_R->ptr()) + size_t(last_C) * d.hc_n() * 2
                                        : nullptr;   // the head's pass left the last row in R row C
    prefill_head_tp(D.cx, D.kc, D.s, model_, model_.parts[r], cards_[r]->buffers, last_C - 1, t.fold[last_slot], r,
                    cards_[r]->desc, t.H, t.inj, R_last);
  }
  drain(base + uint32_t(ids.size()) - 1, "head");
  merge_heads();
  if (hooked) {   // the prompt end, the first generated id pending
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
  for (auto& D : P.dev)   // the scratch is idle until the next prefill: lend it to the tier
    if (D->lend) D->lend->fill();
}

}  // namespace runtime::qwen4exp
