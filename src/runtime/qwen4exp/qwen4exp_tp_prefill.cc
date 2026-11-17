// Qwen4ExpTpEngine::prefill and its accessors - plan 21v2b Task 4, compiled into b70_qwen4exp_prefill (the one-card
// engine's arrangement: qwen4exp_prefill_engine.cc), so a decode-only target links what it always linked.
//
// Each chunk: the host (both cards' prefill lists idle) resets the chunk's crossing events, writes the ids and a chunk
// Control (the card's, at pos / C) on each card, appends card 0's walk and then card 1's (qwen4exp_prefill.h,
// Q4TpPrefill: each list waits only on the peer's copies, never on the host), and waits on both with a bound. One chunk
// is in flight at a time: the cards compute each unit side by side and cross after it (20 MiB of fp32 each way at 2048
// rows), so a chunk's events and fold slots are free once both lists are idle. A card that does not finish in time:
// every event is host-signalled so both lists drain on whatever they hold, the engine throws naming the card and the
// chunk, and stays failed until reset(). After the last chunk both cards run prefill_head_tp on its last row and the
// host merges the heads (the larger value, ties to card 0) as a decode step does.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/qwen4exp_kernels.h"
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
          fold(c, l0::MemKind::Device, size_t(2) * 2 * kPfC * one.hidden * 4, 4096),
          yf(c, l0::MemKind::Device, size_t(pf_tiles(one, kPfC)) * kPfTm * one.hidden * 4, 4096),
          ctl(c, l0::MemKind::Host, sizeof(Control)) {
      std::memset(ctl.ptr(), 0, ctl.size());
    }
    size_t slot_off(uint32_t k) const { return size_t(k) * 2 * kPfC * hidden; }
    prefill::Context cx;
    prefill::KernelCache kc;
    Qwen4ExpPrefillScratch s;
    l0::Mem fold;   // fp32 [2 slots][2 slices][kPfC][hidden]
    l0::Mem yf;     // fp32 [tmax x 32][hidden]
    l0::Mem ctl;    // host: the chunk's Control
    size_t hidden = 0;
    Q4TpPrefill tp;
  };
  Q4TpPrefillState(l0::Context& d0, l0::Context& d1, uint32_t units) : ev(d0, {&d0, &d1}, 2 * units), units(units) {}
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
    require(ps->dev[r]->s.bytes() + ps->dev[r]->fold.size() + ps->dev[r]->yf.size() ==
                prefill_sizes(cards_[r]->desc, max_len_).total() + tp_prefill_bytes(d),
            "card " + std::to_string(r) + "'s prefill allocation is not the plan's (prefill_sizes + tp_prefill_bytes)");
  }
  for (uint32_t r = 0; r < 2; ++r) {
    Q4TpPrefillState::Dev& D = *ps->dev[r];
    Q4TpPrefillState::Dev& P = *ps->dev[1 - r];
    Q4TpPrefill& t = D.tp;
    t.rank = r;
    t.card = &cards_[r]->desc;
    t.embed = cards_[r]->binding.embed;
    t.ple_ptrs = r == 1 ? model_.ple_ptrs1->ptr() : model_.ple.ptrs->ptr();
    for (uint32_t k = 0; k < 2; ++k) {
      t.fold[k] = D.fold.as<float>() + D.slot_off(k);
      t.peer_fold[k] = P.fold.as<float>() + P.slot_off(k);
    }
    t.yf = D.yf.as<float>();
    t.ev = &ps->ev;
    t.units = ps->units;
  }
  pf_settle_ = [ps] { return ps->settle(); };
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
  prepare_prefill();
  Q4TpPrefillState& P = *pf_;
  const uint64_t tmo = P.timeout_ns;
  const auto drain = [&](uint32_t pos, const char* what) {
    for (uint32_t r = 0; r < 2; ++r)
      if (!P.dev[r]->cx.wait_for(tmo)) {
        P.release();
        P.dev[1 - r]->cx.wait_for(tmo);
        P.dev[r]->cx.wait_for(tmo);
        fail("card " + std::to_string(r) + "'s prefill " + what + " at position " + std::to_string(pos) +
             " did not finish within " + std::to_string(opt_.timeout_ms) + " ms");
      }
  };
  // Spec 7's prefix cache: with a block hook every chunk ends at a block end or at the prompt end (the one-card
  // engine's rule, one shared function), and the hook runs after each mid-prompt block end with both cards idle.
  const bool hooked = static_cast<bool>(block_hook_);
  uint32_t last_C = 0;
  for (size_t off = 0; off < ids.size();) {
    const uint32_t pos = base + uint32_t(off);
    const uint32_t C = prefill_chunk_rows(pos, ids.size() - off, chunk, hooked, kBlock);
    P.reset_events();
    for (uint32_t r = 0; r < 2; ++r) {
      Q4TpPrefillState::Dev& D = *P.dev[r];
      std::memcpy(D.s.ids.ptr(), ids.data() + off, size_t(C) * 4);
      Control* k = D.ctl.as<Control>();
      *k = *cards_[r]->ctl;
      k->pos = pos;
      k->n_active = C;
    }
    for (uint32_t r = 0; r < 2; ++r) {
      Q4TpPrefillState::Dev& D = *P.dev[r];
      prefill_chunk(D.cx, D.kc, D.s, model_, model_.parts[r], cards_[r]->buffers, pos, C, false, nullptr, false, nullptr,
                    D.ctl.as<Control>(), D.s.ids.as<uint32_t>(), &D.tp);
    }
    drain(pos, "chunk");
    off += C;
    last_C = C;
    if (hooked && off < ids.size() && (pos + C) % kBlock == 0) {
      for (auto& c : cards_) {
        c->ctl->pos = pos + C;
        c->ctl->n_active = 0;
      }
      block_hook_(pos + C, true);
    }
  }
  // The head on both cards: the last unit's fold slot holds the last row's two partials.
  for (uint32_t r = 0; r < 2; ++r) {
    cards_[r]->ctl->pos = base + uint32_t(ids.size()) - 1;
    cards_[r]->ctl->n_active = 1;
  }
  const uint32_t last_slot = (P.units - 1) & 1u;
  for (uint32_t r = 0; r < 2; ++r) {
    Q4TpPrefillState::Dev& D = *P.dev[r];
    prefill_head_tp(D.cx, D.kc, D.s, model_, model_.parts[r], cards_[r]->buffers, last_C - 1, D.tp.fold[last_slot], r,
                    cards_[r]->desc);
  }
  drain(base + uint32_t(ids.size()) - 1, "head");
  merge_heads();
  if (hooked) {   // the prompt end, the first generated id pending
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
}

}  // namespace runtime::qwen4exp
