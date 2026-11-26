// Qwen4ExpEngine::prefill and its accessors - the only Qwen4ExpEngine members not compiled into b70_qwen4exp_runtime
// (qwen4exp_engine.h says why: KolibriEngine's arrangement, kolibri_prefill_engine.cc).
//
// **One card:** each chunk is device 0's walk (qwen4exp_prefill.cc) on its immediate list, then a wait.
// **Two cards** (spec 16b's pieces, as 21-QFN-v1c Task 4 did for decode): device 0's list runs the embedding and layers
// [0, s) and ends with combine _Y_NN (the chunk's materialised H), then a device-to-device copy of H [C][10240] into
// the prefill link's landing buffer on device 1 and a barrier signalling the link's cross-device event; device 1's
// list waits on the event, copies H into its own scratch and runs layers [s, L) (nothing pending: _X). The host resets
// the event per chunk, appends both lists and waits on device 1 with a bound (PipelineOptions::prefill_timeout_ms); a
// lost hand-off host-signals the event (device 1's list drains on garbage instead of holding the device), throws
// naming it and marks the engine until reset() (PipelineEngine's rule). The last chunk's head runs on device 1, then
// the Control mirror (device 1's block into device 0's).
//
// **The overlap (plan 21-QFN-v2b Task 0, spec 16c's pipeline in this engine's terms).** Device 0 runs chunk j + 1 while
// device 1 runs chunk j. Every per-chunk input the host writes is doubled and chunk j's sits at slot j % kPfDepth: the
// ids (s.ids), each device's chunk Control (the walk's kernels read pos / n_active there, never the device's own
// block), the head's chunk Control, the injected rows, and the link (its landing buffer and its event). The host's
// order, per chunk j: [j >= 1: wait device 0 to finish j - 1 (its list's wait is for everything appended)], write
// chunk j's slot, append device 0's chunk j; [j >= 1: wait device 1 to finish j - 1, then chunk j - 1's block hook],
// append device 1's chunk j. So device 0 runs chunk j while device 1 runs j - 1, nothing of a slot is rewritten, nor
// its event reset, while chunk j - 2 still reads or signals it, and device 1's chunk j starts only after chunk j - 1's
// hook: the hook reads device 1's state live. Device 0 may then be inside chunk j, so a hooked chunk
// copies device 0's state runs (state_runs at its end) into a host shadow at its slot right after its walk, on its
// list, and save_state reads device 0's runs there (pf_shadow_); the KV rows and complete blocks save_kv reads are
// final (chunk j writes later positions). A hook that throws drains device 0 and copies the shadow back: the state is
// the block end's, as on one card. Each walk runs the same launches in the same order on its own device as before,
// so the result is the sequential form's, bitwise.
//
// **Pairs (PERF-30).** On two cards each device walks the chunks in pairs (2p, 2p + 1) a layer at a time
// (prefill_pair_pp), so each layer's copy of its host-memory experts into the read ahead serves both chunks - tp's
// arrangement. The overlap above then runs per pair: device 0 runs pair p while device 1 runs pair p - 1. Chunk j's
// slot is still j % kPfDepth (its place in its pair), but a device's slots are written right before that device's
// half of the pair is appended (its lists idle): device 0's for pair p after it finished p - 1, device 1's after it
// finished p - 1 - device 1 may still read its slots of pair p - 1 while device 0 runs pair p. Device 0's hand-off
// (H into the two landing buffers, their events) is appended after its walk only once device 1 has finished pair
// p - 1, whose copies out of those buffers it would overwrite. The block hook runs after each pair that ends
// mid-prompt on a block end (tp's rule: a block end inside a pair is not hooked - no moment holds the state at it).
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_chunks.h"
#include "runtime/qwen4exp/qwen4exp_engine.h"
#include "runtime/qwen4exp/qwen4exp_prefill.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace runtime::qwen4exp {

struct Qwen4ExpPrefillState {
  struct Dev {
    prefill::Context cx;
    prefill::KernelCache kc;
    Qwen4ExpPrefillScratch s;
    std::unique_ptr<l0::Mem> inj[kPfDepth];   // host USM [qsa layers][rows][kListRow] a slot: the injected rows (lazy)
    std::unique_ptr<l0::Mem> mtp_R;   // spec 21-QFN-v1e, the last device with the MTP head: the head pass's R rows
    l0::Mem ctl;                      // host [kPfDepth] Control: the chunks' Controls
    std::unique_ptr<l0::Mem> hctl;    // the last device with the MTP head: host [kPfDepth] Control, the head's
    std::unique_ptr<Q4PfRingState> ring;   // the host-memory experts read ahead (null: no expert table)
    std::unique_ptr<Q4LendState> lend;     // the scratch lent to the tier between prefills (with the ring)
    // PERF-30, two devices: the pair's first chunk's own residual rows, inject weights and MoE output (pf_pair_bytes)
    std::unique_ptr<l0::Mem> H1, inj1, y1;
    Q4PfOwnRows own;
    Dev(l0::Context& c, const model::Qwen4ExpDesc& d, uint32_t max_len, uint32_t rows)
        : cx(c), kc(c), s(c, d, max_len, rows), ctl(c, l0::MemKind::Host, kPfDepth * sizeof(Control)) {}
    Control* ctl_at(uint32_t k) { return ctl.as<Control>() + k; }
    uint32_t* ids_at(uint32_t k) { return s.ids.as<uint32_t>() + size_t(k) * s.rows; }
  };
  static_assert(gs::buffers_problem<l0::Mem, Dev, PfDevDeviceSizes, PfDevHostSizes>().empty(),
                gs::buffers_problem<l0::Mem, Dev, PfDevDeviceSizes, PfDevHostSizes>());
  std::vector<std::unique_ptr<Dev>> dev;
  std::unique_ptr<PipelineLink> link[kPfDepth];   // two devices: at chunk size (pf_landing_layout), one a slot
  std::unique_ptr<l0::Mem> shadow[kPfDepth];      // two devices with a block hook: host, device 0's state runs at
                                                  // pair p's end at p % kPfDepth (lazy)
  uint32_t last = 0;                              // the last chunk's slot (read_prefill_selection's injected rows)
  uint64_t timeout_ns = 0;
  void signal_links() {
    for (auto& l : link)
      if (l) l->event->host_signal();
  }
  ~Qwen4ExpPrefillState() {   // never free a scratch under a running list
    signal_links();
    for (auto& d : dev) {
      try {
        d->cx.wait_for(timeout_ns);
      } catch (...) {
      }
    }
  }
  // reset()'s hook: release a list waiting on a hand-off that will not come; true when every list is idle.
  bool settle() {
    signal_links();
    bool ok = true;
    for (auto& d : dev) ok = d->cx.wait_for(timeout_ns) && ok;
    for (auto& l : link)
      if (l) l->event->host_reset();
    return ok;
  }
};
// Spec 39: the state's own allocation - device 0's state runs at a pair's end, host memory (outside the plan), written
// at the block end before save_state reads it.
struct PfStateSizes {
  [[=gs::scratch]] size_t shadow = 0;
};
static_assert(gs::buffers_problem<l0::Mem, Qwen4ExpPrefillState, PfStateSizes>().empty(),
              gs::buffers_problem<l0::Mem, Qwen4ExpPrefillState, PfStateSizes>());

namespace {
void destroy_prefill(Qwen4ExpPrefillState* p) { delete p; }
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::prefill: " + what);
}
}  // namespace

void Qwen4ExpEngine::prepare_prefill() {
  if (pf_) return;
  const model::Qwen4ExpDesc& d = model_.desc;
  const bool eager = attn_ == Q4Attn::Eager;
  const uint32_t rows = model_.pf_rows;   // the planner's chunk rows (kPfC or kPfCWide)
  // A missing binary is named here, before anything is allocated or appended (the decode capture's rule); the head's
  // binaries are decode's, which the capture already checked.
  for (const std::string& v :
       kernels::qwen4exp::prefill_variants(d, eager, model_.ple.scale == loader::Q4PleScale::Bf16, rows))
    require(std::ifstream(kernels::path(v)).good(),
            v + " is not compiled (" + kernels::path(v) + "): build with B70_Q4EXP=ON");
  if (mtp_)   // spec 21-QFN-v1e: the head's pass per chunk
    for (const std::string& v : kernels::qwen4exp::mtp_prefill_variants(mtp_norm_ == MtpNorm::Single, rows, d.hc_int8))
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_Q4EXP=ON and B70_MTP=ON");
  auto* ps = new Qwen4ExpPrefillState();
  pf_ = std::unique_ptr<Qwen4ExpPrefillState, void (*)(Qwen4ExpPrefillState*)>(ps, &destroy_prefill);
  ps->timeout_ns = uint64_t(opt_.prefill_timeout_ms) * 1000000ull;
  for (uint32_t i = 0; i < devices(); ++i)
    ps->dev.push_back(std::make_unique<Qwen4ExpPrefillState::Dev>(stage_ctx(i), d, max_len_, rows));
  if (devices() == 2)
    for (auto& l : ps->link) l = std::make_unique<PipelineLink>(stage_ctx(0), stage_ctx(1), pf_landing_layout(d, rows));
  if (mtp_) {
    ps->dev[devices() - 1]->mtp_R = std::make_unique<l0::Mem>(stage_ctx(devices() - 1), l0::MemKind::Device,
                                                              pf_dev_sizes(d, rows, true, false).mtp_R);
    ps->dev[devices() - 1]->hctl =
        std::make_unique<l0::Mem>(stage_ctx(devices() - 1), l0::MemKind::Host, kPfDepth * sizeof(Control));
  }
  for (uint32_t i = 0; i < devices(); ++i) {
    Qwen4ExpPrefillState::Dev& D = *ps->dev[i];
    if (devices() == 2) {
      const PfDevDeviceSizes ds = pf_dev_sizes(d, rows, false, true);
      D.H1 = std::make_unique<l0::Mem>(stage_ctx(i), l0::MemKind::Device, ds.H1);
      D.inj1 = std::make_unique<l0::Mem>(stage_ctx(i), l0::MemKind::Device, ds.inj1);
      D.y1 = std::make_unique<l0::Mem>(stage_ctx(i), l0::MemKind::Device, ds.y1);
      D.own = {D.H1->ptr(), D.inj1->ptr(), D.y1->ptr()};
    }
    D.ring = make_pf_ring(stage_ctx(i), model_.parts[i], d);
    if (D.ring)
      D.lend = std::make_unique<Q4LendState>(stage_ctx(i), model_.parts[i], d, model_.tier, lend_regions(D.s, D.ring.get()),
                                             D.ring->list);
    if (D.lend && D.lend->experts() > 0) {
      const loader::Q4DevicePart& p = model_.parts[i];
      size_t res = 0;
      for (uint32_t l = p.first; l < p.end; ++l) res += model_.tier.n_vram(l);
      const double all = double(p.end - p.first) * d.experts;
      std::fprintf(stderr, "experts: device %u lends its idle prefill scratch to %u more experts between prefills "
                   "(resident f %.3f -> %.3f while decoding)\n", i, D.lend->experts(), res / all,
                   (res + D.lend->experts()) / all);
    }
  }
  pf_lend_ = [this, ps] {
    for (uint32_t i = 0; i < ps->dev.size(); ++i)
      if (Q4LendState* L = ps->dev[i]->lend.get()) {
        const bool was = L->state == Q4LendState::State::Lent;
        L->lend();
        if (cpu_ && !was && L->state == Q4LendState::State::Lent) cpu_->set_rows(i, L->lent);   // PERF-124 step 4
      }
  };
  pf_dev_bytes_.assign(devices(), 0);
  pf_bytes_ = 0;
  for (uint32_t i = 0; i < devices(); ++i) {
    size_t link = 0;
    for (auto& l : ps->link) link += l ? l->bytes(i) : 0;
    const Qwen4ExpPrefillState::Dev& D = *ps->dev[i];
    // Spec 39: the device's own allocations beside the scratch (PfDevDeviceSizes: the pair's rows, the head's R rows).
    const size_t pair = D.H1 ? D.H1->size() + D.inj1->size() + D.y1->size() : 0;
    pf_dev_bytes_[i] = D.s.bytes() + link + gs::buffer_bytes<l0::Mem, PfDevDeviceSizes>(D) +
                       (D.ring ? pf_ring_bytes(d) : 0);
    pf_bytes_ += pf_dev_bytes_[i];
    contract_assert(ps->dev[i]->s.bytes() == prefill_sizes(d, max_len_, rows).total());   // prefill_sizes()'s scratch
    contract_assert(!ps->link[0] || link == pf_link_bytes(d, i, rows));                  // pf_link_bytes()'s link
    contract_assert(!D.H1 || pair == pf_pair_bytes(d, rows));                            // pf_pair_bytes()'s rows
  }
  pf_settle_ = [ps] { return ps->settle(); };
}

size_t Qwen4ExpEngine::prefill_launches(uint32_t dev) const {
  return pf_ && dev < pf_->dev.size() ? pf_->dev[dev]->cx.launches() : 0;
}
size_t Qwen4ExpEngine::prefill_launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < devices(); ++i) n += prefill_launches(i);
  return n;
}

std::vector<uint32_t> Qwen4ExpEngine::read_prefill_routes() {
  require(bool(pf_), "read_prefill_routes before the first prefill");
  const model::Qwen4ExpDesc& d = model_.desc;
  const uint32_t rows = model_.pf_rows;
  std::vector<uint32_t> v(size_t(d.layers) * rows * kRouteWords, 0);
  for (uint32_t i = 0; i < devices(); ++i) {
    Qwen4ExpPrefillState::Dev& D = *pf_->dev[i];
    require(D.cx.wait_for(pf_->timeout_ns), "device " + std::to_string(i) + "'s prefill list is still running");
    const uint32_t first = model_.placement.first(i), n = model_.placement.count(i);
    stage_imm(i).copy(v.data() + pf_route_at(first, rows) / 4,
                      static_cast<const uint8_t*>(D.s.routes.ptr()) + pf_route_at(first, rows),
                      size_t(n) * rows * kRouteWords * 4);
  }
  return v;
}

std::vector<uint32_t> Qwen4ExpEngine::read_prefill_selection(uint32_t layer) {
  require(bool(pf_), "read_prefill_selection before the first prefill");
  const model::Qwen4ExpDesc& d = model_.desc;
  if (layer >= d.layers || !d.is_qsa(layer))
    throw std::out_of_range("read_prefill_selection: layer " + std::to_string(layer) + " is not a QSA layer");
  const uint32_t qi = d.qsa_before(layer), dev = model_.placement.device_of(layer);
  Qwen4ExpPrefillState::Dev& D = *pf_->dev[dev];
  require(D.cx.wait_for(pf_->timeout_ns), "device " + std::to_string(dev) + "'s prefill list is still running");
  const uint32_t rows = model_.pf_rows;
  std::vector<uint32_t> v(size_t(rows) * kListRow);
  if (injected_ && D.inj[pf_->last])
    std::memcpy(v.data(), static_cast<const uint8_t*>(D.inj[pf_->last]->ptr()) + pf_list_at(qi, rows), v.size() * 4);
  else
    stage_imm(dev).copy(v.data(), static_cast<const uint8_t*>(D.s.lists.ptr()) + pf_list_at(qi, rows), v.size() * 4);
  return v;
}

std::vector<float> Qwen4ExpEngine::read_prefill_last_qsa_scores() {
  require(bool(pf_), "read_prefill_last_qsa_scores before the first prefill");
  const model::Qwen4ExpDesc& d = model_.desc;
  uint32_t last = d.layers;
  for (uint32_t l = 0; l < d.layers; ++l)
    if (d.is_qsa(l)) last = l;
  if (last == d.layers) throw std::out_of_range("read_prefill_last_qsa_scores: no QSA layer is loaded");
  const uint32_t dev = model_.placement.device_of(last);
  Qwen4ExpPrefillState::Dev& D = *pf_->dev[dev];
  require(D.cx.wait_for(pf_->timeout_ns), "device " + std::to_string(dev) + "'s prefill list is still running");
  std::vector<float> v(size_t(model_.pf_rows) * (max_len_ / d.idx_compress));
  stage_imm(dev).copy(v.data(), D.s.scores.ptr(), v.size() * 4);
  return v;
}

void Qwen4ExpEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  require(!ids.empty(), "no ids");
  if (chunk == 0) chunk = model_.pf_rows;
  require(chunk <= model_.pf_rows, "chunk " + std::to_string(chunk) + " exceeds the prefill scratch's " +
                                       std::to_string(model_.pf_rows) + " rows");
  for (uint32_t id : ids)
    require(id < d.vocab, "id " + std::to_string(id) + " is outside the vocabulary (" + std::to_string(d.vocab) +
                              " rows) - pf_embed_gather has no debug_flag channel, so the host is the only bound");
  const uint32_t base = stage_control(devices() - 1)->pos;
  require(size_t(base) + ids.size() <= max_len_,
          "pos " + std::to_string(base) + " + " + std::to_string(ids.size()) + " ids exceeds max_len " +
              std::to_string(max_len_) + " - the KV, the indexer keys and the RoPE table stop there");
  for (uint32_t i = 0; i < devices(); ++i)   // no decode step in flight writes the state or the Control
    require(settle(i), "device " + std::to_string(i) + "'s last decode step is still running");
  if (injected_) require(bool(pf_inject_), "the injected run on prefill needs set_prefill_injector's lists");
  // Spec 21-QFN-v1e: the prefill's GDN path reads gdn_state directly, so the rows a commit left are replayed into it
  // first (PERF-51); a verify waiting for its commit is dropped (the prompt continues from pos).
  if (mtp_) {
    flush_gdn();
    verify_k_ = kNoVerify;
  }

  prepare_prefill();
  Qwen4ExpPrefillState& P = *pf_;
  for (uint32_t i = 0; i < P.dev.size(); ++i)   // the lent experts' rows back: this prefill overwrites the scratch
    if (Q4LendState* L = P.dev[i]->lend.get()) {
      const bool was = L->state == Q4LendState::State::Lent;
      L->reclaim();
      if (cpu_ && was) cpu_->set_rows(i, L->orig);   // PERF-124 step 4: the CPU's copy of the rows
    }
  const bool eager = attn_ == Q4Attn::Eager;
  const uint32_t ndev = devices();
  Qwen4ExpMtpPrefill mp[kPfDepth];   // the head's pass, a slot each (its Control: the slot's)
  if (mtp_)
    for (uint32_t k = 0; k < kPfDepth; ++k) {
      mp[k].bufs = mtp_.get();
      mp[k].R = P.dev[ndev - 1]->mtp_R.get();   // one: the passes run in order on the last device's list
      mp[k].embed = mtp_bind_.embed;
      mp[k].single = mtp_norm_ == MtpNorm::Single;
      mp[k].hctl = P.dev[ndev - 1]->hctl->as<Control>() + k;
    }
  const uint64_t tmo = P.timeout_ns;
  const bool inj = injected_;
  if (inj)
    for (uint32_t i = 0; i < ndev; ++i)
      for (auto& m : P.dev[i]->inj)
        if (!m) {
          m = std::make_unique<l0::Mem>(stage_ctx(i), l0::MemKind::Host, pf_injected_bytes(d, model_.pf_rows));
          std::memset(m->ptr(), 0, m->size());
        }
  const auto set_ctl = [&](uint32_t pos, uint32_t n) {
    for (uint32_t i = 0; i < ndev; ++i) {
      stage_control(i)->pos = pos;
      stage_control(i)->n_active = n;
    }
  };

  // Spec 21-QFN-v1e's prefix cache: with a block hook every chunk ends at a block end or at the prompt end - Engine::prefill's
  // rule, one shared function - so each completed block is a storable one. PERF-36, Qwen3.8's rule (open-work 18): a
  // 4096-row chunk hooks every 4096 positions (the block between is stored as KV by the hook's store call), and over the
  // last kPrefillTail positions (6144-8191 after the round-down) the chunks are 2048 rows with a hook at every block
  // end, so a follow-up turn that leaves the prompt within them restarts at most 2047 positions back.
  const bool hooked = static_cast<bool>(block_hook_);
  struct Chunk {
    uint32_t pos, C;
    bool hook;   // a mid-prompt block end: block_hook_(end, true) after it
  };
  std::vector<Chunk> chunks;
  for (size_t off = 0; off < ids.size();) {
    const uint32_t pos = base + uint32_t(off);
    const uint32_t C = prefill_chunk_rows(pos, ids.size() - off, chunk, hooked, kBlock, /*wide_stride=*/true, kPrefillTail);
    off += C;
    chunks.push_back({pos, C, hooked && off < ids.size() && (pos + C) % kBlock == 0});
  }
  const uint32_t n = uint32_t(chunks.size());
  const bool two = ndev == 2;
  if (two && hooked)
    for (auto& m : P.shadow)
      if (!m) m = std::make_unique<l0::Mem>(stage_ctx(0), l0::MemKind::Host, state_bytes());
  bool sent[kPfDepth] = {true, true};   // P4's test hook: whether device 0's half of the slot's pair was appended

  // Chunk j's slot on device i (written while that device's lists are idle): the ids, the device's chunk Control (a
  // copy of its block at pos / C), on the last device the head's, the injected rows of the device's QSA layers.
  const auto fill = [&](uint32_t i, uint32_t j) {
    const Chunk& c = chunks[j];
    const uint32_t k = j % kPfDepth;
    Qwen4ExpPrefillState::Dev& D = *P.dev[i];
    std::memcpy(D.ids_at(k), ids.data() + (c.pos - base), size_t(c.C) * 4);
    *D.ctl_at(k) = *stage_control(i);
    D.ctl_at(k)->pos = c.pos;
    D.ctl_at(k)->n_active = c.C;
    if (mtp_ && i + 1 == ndev) {   // spec 21-QFN-v1e: the head's pass covers positions pos - 1 .. pos + C - 2 (from 0 at pos 0)
      Control* h = D.hctl->as<Control>() + k;
      *h = *hctl();
      h->pos = c.pos == 0 ? 0 : c.pos - 1;
      h->n_active = mtp_prefill_rows(c.pos, c.C);
    }
    if (inj) {   // the caller's lists for this chunk, of each QSA layer this device holds
      for (uint32_t l = 0; l < d.layers; ++l) {
        if (!d.is_qsa(l) || model_.placement.device_of(l) != i) continue;
        const uint32_t qi = d.qsa_before(l);
        pf_inject_(qi, c.pos, c.C,
                   reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(D.inj[k]->ptr()) + pf_list_at(qi, model_.pf_rows)));
      }
    }
  };
  // One card: the walk for chunk j (appended immediately), on its slot.
  const auto run_dev = [&](uint32_t i, uint32_t j) {
    Qwen4ExpPrefillState::Dev& D = *P.dev[i];
    const uint32_t k = j % kPfDepth;
    const bool head = mtp_ && i + 1 == ndev;
    if (D.ring) D.ring->reset();   // the device's lists are idle: its previous chunk has finished
    const Q4PfRing ring = D.ring ? D.ring->view(true, true) : Q4PfRing{};
    const bool staged = D.ring && pf_ring_staged(chunks[j].C, 0);   // a short chunk reads its experts in place
    prefill_chunk(D.cx, D.kc, D.s, model_, model_.parts[i], stage_buffers(i), chunks[j].pos, chunks[j].C, eager,
                  inj ? D.inj[k].get() : nullptr, two, head ? &mp[k] : nullptr, D.ctl_at(k), D.ids_at(k),
                  staged ? &ring : nullptr);
  };
  // Two cards (PERF-30): pair p is chunks 2p and 2p + 1 (the prompt's last pair may be one chunk), at slots 0 and 1.
  const uint32_t npair = (n + 1) / 2;
  const auto pair_n = [&](uint32_t p) { return std::min(2u, n - 2 * p); };
  // Chunk h of pair p's residual rows on device D: the pair's first chunk's own, else the scratch's.
  const auto rows_H = [&](Qwen4ExpPrefillState::Dev& D, uint32_t p, uint32_t h) {
    return pair_n(p) == 2 && h == 0 ? D.H1->ptr() : D.s.H.ptr();
  };
  // Device i's half of pair p (its lists idle): its slots, then both chunks a layer at a time.
  const auto walk = [&](uint32_t i, uint32_t p) {
    Qwen4ExpPrefillState::Dev& D = *P.dev[i];
    const uint32_t m = pair_n(p), j0 = 2 * p;
    for (uint32_t h = 0; h < m; ++h) fill(i, j0 + h);
    if (D.ring) D.ring->reset();
    const bool staged = D.ring && pf_ring_staged(chunks[j0].C, m == 2 ? chunks[j0 + 1].C : 0);
    const bool head = mtp_ && i + 1 == ndev;
    std::array<Q4PfRing, 2> rv{};
    std::array<Q4PfChunk, 2> c{};
    for (uint32_t h = 0; h < m; ++h) {   // chunk j0 + h's slot is h (j0 is even)
      const Chunk& ch = chunks[j0 + h];
      if (staged) rv[h] = D.ring->view(h == 0, h + 1 == m);
      c[h] = {ch.pos, ch.C, inj ? D.inj[h].get() : nullptr, head ? &mp[h] : nullptr, D.ctl_at(h), D.ids_at(h),
              staged ? &rv[h] : nullptr, m == 2 && h == 0 ? &D.own : nullptr};
    }
    prefill_pair_pp(D.cx, D.kc, D.s, model_, model_.parts[i], stage_buffers(i), eager, two, c, m);
  };
  const size_t rows_max = size_t(model_.pf_rows) * d.hc_n() * 2;
  // The prefill landing buffer holds a chunk's H rows (and no norm sums).
  contract_assert(!two || (rows_max <= P.link[0]->layout.resid_bytes && P.link[0]->layout.sumsq_bytes == 0));
  // Device 0's walk of pair p, [hooked at the pair's end: its state runs into the shadow].
  const auto run0 = [&](uint32_t p) {
    const uint32_t k = p % kPfDepth;
    // P4's test hook (drop_next_handoff, decode's): device 0's half of this pair is not appended - a hand-off that
    // never arrives.
    sent[k] = !drop_next_;
    drop_next_ = false;
    if (!sent[k]) return;
    Qwen4ExpPrefillState::Dev& D0 = *P.dev[0];
    walk(0, p);
    const Chunk& last = chunks[2 * p + pair_n(p) - 1];
    if (last.hook) {
      uint8_t* sh = P.shadow[k]->as<uint8_t>();
      for (const SnapRun& r : state_runs(d, model_.placement, last.pos + last.C, mtp()))
        if (r.device == 0 && !r.pad && !r.zero) {
          contract_assert(sh + r.bytes <= P.shadow[k]->as<uint8_t>() + P.shadow[k]->size());   // within the shadow
          D0.cx.copy(sh, snap_ptr(r), r.bytes);
          sh += r.bytes;
        }
    }
  };
  // Device 0's hand-off of pair p after its walk - appended once device 1 has finished pair p - 1 (whose copies out
  // of the landing buffers it overwrites): each chunk's H into landing slot h, its event.
  const auto send0 = [&](uint32_t p) {
    for (uint32_t h = 0; h < pair_n(p); ++h) P.link[h]->event->host_reset();
    if (!sent[p % kPfDepth]) return;
    Qwen4ExpPrefillState::Dev& D0 = *P.dev[0];
    for (uint32_t h = 0; h < pair_n(p); ++h) {
      PipelineLink& L = *P.link[h];
      D0.cx.copy(L.landing.as<uint8_t>(), rows_H(D0, p, h), size_t(chunks[2 * p + h].C) * d.hc_n() * 2);
      D0.cx.signal(L.event->handle());
    }
  };
  const auto run1 = [&](uint32_t p) {
    Qwen4ExpPrefillState::Dev& D1 = *P.dev[1];
    for (uint32_t h = 0; h < pair_n(p); ++h) {
      PipelineLink& L = *P.link[h];
      D1.cx.wait_event(L.event->handle());
      D1.cx.copy(rows_H(D1, p, h), L.landing.as<uint8_t>(), size_t(chunks[2 * p + h].C) * d.hc_n() * 2);
    }
    walk(1, p);
  };
  // One card: chunk j; two cards: pair j.
  const auto wait0 = [&](uint32_t j) {
    if (!P.dev[0]->cx.wait_for(tmo))
      fail("device 0's prefill chunk at " + std::to_string(chunks[two ? 2 * j : j].pos) + " did not finish within " +
           std::to_string(opt_.prefill_timeout_ms) + " ms");
  };
  const auto wait1 = [&](uint32_t p) {
    Qwen4ExpPrefillState::Dev& D0 = *P.dev[0];
    Qwen4ExpPrefillState::Dev& D1 = *P.dev[1];
    if (D1.cx.wait_for(tmo)) return;
    // Device 1 still waits (on the event device 0 never signalled, or on a hung walk): release it so its list drains
    // on garbage instead of holding the device, then report.
    P.signal_links();
    const bool drained = D1.cx.wait_for(tmo);
    const bool d0 = D0.cx.wait_for(tmo);
    const Chunk& c = chunks[2 * p];
    uint32_t rows = 0;
    for (uint32_t h = 0; h < pair_n(p); ++h) rows += chunks[2 * p + h].C;
    fail("device 1's prefill chunks at " + std::to_string(c.pos) + " (" + std::to_string(rows) +
         " rows) did not finish within " + std::to_string(opt_.prefill_timeout_ms) + " ms - " +
         (d0 ? "device 0 finished but its rows never released device 1" : "device 0 did not finish either") +
         " (the prefill link's cross-device event; device 1 " + (drained ? "released" : "did not drain") + ")" +
         (sent[p % kPfDepth] ? "" : " [device 0's half was not appended: drop_next_handoff()]"));
  };
  // A mid-prompt block end: the state holds exactly [0, end); say so in every Control before the hook runs, so a hook
  // that throws leaves pos matching the chunks written. On two cards (chunk j ends pair `p`) device 0 may be inside the
  // next pair: its state runs come from the shadow, and a throwing hook gets them back.
  const auto hook = [&](uint32_t j, uint32_t p) {
    const uint32_t end = chunks[j].pos + chunks[j].C;
    set_ctl(end, 0);
    if (!two) {
      block_hook_(end, true);
      return;
    }
    pf_shadow_ = P.shadow[p % kPfDepth]->as<uint8_t>();
    try {
      block_hook_(end, true);
    } catch (...) {
      P.dev[0]->cx.wait_for(tmo);
      const uint8_t* sh = pf_shadow_;
      for (const SnapRun& r : state_runs(d, model_.placement, end, mtp()))
        if (r.device == 0 && !r.pad && !r.zero) {
          stage_imm(0).copy(snap_ptr(r), sh, r.bytes);
          sh += r.bytes;
        }
      pf_shadow_ = nullptr;
      throw;
    }
    pf_shadow_ = nullptr;
  };

  if (!two) {
    for (uint32_t j = 0; j < n; ++j) {
      fill(0, j);
      run_dev(0, j);
      wait0(j);
      if (chunks[j].hook) hook(j, 0);
    }
  } else {
    for (uint32_t p = 0; p < npair; ++p) {
      if (p >= 1) wait0(p - 1);
      run0(p);
      if (p >= 1) {
        wait1(p - 1);
        if (chunks[2 * p - 1].hook) hook(2 * p - 1, p - 1);   // pair p - 1 is whole: it is not the last
      }
      send0(p);
      run1(p);
    }
    wait1(npair - 1);
    wait0(npair - 1);
  }
  P.last = (n - 1) % kPfDepth;
  const uint32_t C = chunks[n - 1].C;
  if (mtp_) {   // the head's Control as the sequential form left it: the last chunk's
    const Control& h = *(P.dev[ndev - 1]->hctl->as<Control>() + P.last);
    hctl()->pos = h.pos;
    hctl()->n_active = h.n_active;
  }

  // The tail on the last device: pos = base + L - 1 and n_active = 1 make argmax_stage2 leave pos = base + L and the
  // first generated id in cur_token[0]; `C - 1` is the last chunk's last row.
  set_ctl(base + uint32_t(ids.size()) - 1, 1);
  {
    Qwen4ExpPrefillState::Dev& D = *P.dev[ndev - 1];
    const void* R_last = mtp_ ? static_cast<const uint8_t*>(mp[0].R->ptr()) + size_t(C) * d.hc_n() * 2 : nullptr;
    prefill_head(D.cx, D.kc, D.s, model_, model_.parts[ndev - 1], stage_buffers(ndev - 1), C - 1, R_last);
    if (!D.cx.wait_for(tmo))
      fail("device " + std::to_string(ndev - 1) + "'s prefill head did not finish within " +
           std::to_string(opt_.prefill_timeout_ms) + " ms");
  }
  // The token's way back (16b's mirror): device 0 reads pos and cur_token from its own block next step.
  if (ndev == 2) std::memcpy(stage_control(0), stage_control(1), sizeof(Control));
  if (hooked) {   // the prompt end, the first generated id in cur_token
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
  for (auto& D : P.dev)   // the scratch is idle until the next prefill: lend it to the tier
    if (D->lend) D->lend->fill();
}

}  // namespace runtime::qwen4exp
