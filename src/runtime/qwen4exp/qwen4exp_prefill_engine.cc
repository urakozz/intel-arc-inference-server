// Qwen4ExpEngine::prefill and its accessors - the only Qwen4ExpEngine members not compiled into b70_qwen4exp_runtime
// (qwen4exp_engine.h says why: KolibriEngine's arrangement, kolibri_prefill_engine.cc).
//
// **One card:** each chunk is device 0's walk (qwen4exp_prefill.cc) on its immediate list, then a wait.
// **Two cards** (spec 16b's pieces, as 21c Task 4 did for decode): device 0's list runs the embedding and layers
// [0, s) and ends with combine _Y_NN (the chunk's materialised H), then a device-to-device copy of H [C][10240] into
// the prefill link's landing buffer on device 1 and a barrier signalling the link's cross-device event; device 1's
// list waits on the event, copies H into its own scratch and runs layers [s, L) (nothing pending: _X). The host resets
// the event per chunk, appends both lists and waits on device 1 with a bound (PipelineOptions::prefill_timeout_ms); a
// lost hand-off host-signals the event (device 1's list drains on garbage instead of holding the device), throws
// naming it and marks the engine until reset() (PipelineEngine's rule). The last chunk's head runs on device 1, then
// the Control mirror (device 1's block into device 0's). The chunks cross SEQUENTIALLY: device 0 idles while device
// 1 runs a chunk - spec 16c's overlapped chunk pipeline is plan 21d Task 5's recorded lever.
#include <algorithm>
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
  struct Chunk {
    uint32_t pos, rows;
    bool injected;
    std::unique_ptr<prefill::Context::Recording> recording;
  };
  struct Dev {
    prefill::Context cx;
    prefill::KernelCache kc;
    Qwen4ExpPrefillScratch s;
    std::unique_ptr<l0::Mem> inj;   // host USM [qsa layers][kPfC][kListRow]: the injected rows (lazy)
    std::unique_ptr<l0::Mem> mtp_R;   // spec 21e, the last device with the MTP head: the head pass's R rows
    // Declared after cx / kc / s: recordings are destroyed first.
    std::vector<Chunk> chunks;
    Dev(l0::Context& c, const model::Qwen4ExpDesc& d, uint32_t max_len) : cx(c), kc(c), s(c, d, max_len) {}
  };
  std::vector<std::unique_ptr<Dev>> dev;
  std::unique_ptr<PipelineLink> link;   // two devices: `copy` mode at chunk size (pf_landing_layout)
  uint64_t timeout_ns = 0;
  ~Qwen4ExpPrefillState() {   // never free a scratch under a running list
    if (link && link->event) link->event->host_signal();
    for (auto& d : dev) {
      try {
        d->cx.wait_for(timeout_ns);
      } catch (...) {
      }
    }
  }
  // reset()'s hook: release a list waiting on a hand-off that will not come; true when every list is idle.
  bool settle() {
    if (link && link->event) link->event->host_signal();
    bool ok = true;
    for (auto& d : dev) ok = d->cx.wait_for(timeout_ns) && ok;
    if (link && link->event) link->event->host_reset();
    return ok;
  }
};

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
  // A missing binary is named here, before anything is allocated or appended (the decode capture's rule); the head's
  // binaries are decode's, which the capture already checked.
  for (const std::string& v : kernels::qwen4exp::prefill_variants(d, eager, model_.ple.scale == loader::Q4PleScale::Bf16))
    require(std::ifstream(kernels::path(v)).good(),
            v + " is not compiled (" + kernels::path(v) + "): build with B70_Q4EXP=ON (spec 21d)");
  if (mtp_)   // spec 21e: the head's pass per chunk
    for (const std::string& v : kernels::qwen4exp::mtp_prefill_variants(mtp_norm_ == MtpNorm::Single))
      require(std::ifstream(kernels::path(v)).good(),
              v + " is not compiled (" + kernels::path(v) + "): build with B70_Q4EXP=ON and B70_MTP=ON (spec 21e)");
  auto* ps = new Qwen4ExpPrefillState();
  pf_ = std::unique_ptr<Qwen4ExpPrefillState, void (*)(Qwen4ExpPrefillState*)>(ps, &destroy_prefill);
  ps->timeout_ns = uint64_t(opt_.prefill_timeout_ms) * 1000000ull;
  for (uint32_t i = 0; i < devices(); ++i)
    ps->dev.push_back(std::make_unique<Qwen4ExpPrefillState::Dev>(stage_ctx(i), d, max_len_));
  if (devices() == 2)
    ps->link = std::make_unique<PipelineLink>(stage_ctx(0), stage_ctx(1), pf_landing_layout(d), PpHandoff::Copy);
  if (mtp_)
    ps->dev[devices() - 1]->mtp_R =
        std::make_unique<l0::Mem>(stage_ctx(devices() - 1), l0::MemKind::Device, mtp_prefill_R_bytes(d));
  pf_dev_bytes_.assign(devices(), 0);
  pf_bytes_ = 0;
  for (uint32_t i = 0; i < devices(); ++i) {
    pf_dev_bytes_[i] = ps->dev[i]->s.bytes() + (ps->link ? ps->link->bytes(i) : 0) +
                       (ps->dev[i]->mtp_R ? ps->dev[i]->mtp_R->size() : 0);
    pf_bytes_ += pf_dev_bytes_[i];
    require(ps->dev[i]->s.bytes() == prefill_sizes(d, max_len_).total(), "the prefill scratch is not prefill_sizes()'s");
    require(!ps->link || ps->link->bytes(i) == pf_link_bytes(d, i), "the prefill link is not pf_link_bytes()'s");
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
  std::vector<uint32_t> v(size_t(d.layers) * kPfC * kRouteWords, 0);
  for (uint32_t i = 0; i < devices(); ++i) {
    Qwen4ExpPrefillState::Dev& D = *pf_->dev[i];
    require(D.cx.wait_for(pf_->timeout_ns), "device " + std::to_string(i) + "'s prefill list is still running");
    const uint32_t first = model_.placement.first(i), n = model_.placement.count(i);
    stage_imm(i).copy(v.data() + pf_route_at(first) / 4, static_cast<const uint8_t*>(D.s.routes.ptr()) + pf_route_at(first),
                      size_t(n) * kPfC * kRouteWords * 4);
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
  std::vector<uint32_t> v(size_t(kPfC) * kListRow);
  if (injected_ && D.inj)
    std::memcpy(v.data(), static_cast<const uint8_t*>(D.inj->ptr()) + pf_list_at(qi), v.size() * 4);
  else
    stage_imm(dev).copy(v.data(), static_cast<const uint8_t*>(D.s.lists.ptr()) + pf_list_at(qi), v.size() * 4);
  return v;
}

void Qwen4ExpEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  require(!ids.empty(), "no ids");
  if (chunk == 0) chunk = kPfC;
  require(chunk <= kPfC, "chunk " + std::to_string(chunk) + " exceeds kPfC " + std::to_string(kPfC));
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
  // Spec 21e: the prefill's GDN path reads slot 0 directly, so a live slot a commit left is copied there first; a
  // verify waiting for its commit is dropped (the prompt continues from pos).
  if (mtp_) {
    mtp_normalise_live();
    verify_k_ = kNoVerify;
  }

  prepare_prefill();
  Qwen4ExpPrefillState& P = *pf_;
  Qwen4ExpMtpPrefill mp;
  if (mtp_) {
    mp.bufs = mtp_.get();
    mp.R = P.dev[devices() - 1]->mtp_R.get();
    mp.embed = mtp_bind_.embed;
    mp.single = mtp_norm_ == MtpNorm::Single;
  }
  const bool eager = attn_ == Q4Attn::Eager;
  const char* env = std::getenv("B70_PREFILL_REPLAY");
  const bool replay = pf_replay_ >= 0 ? pf_replay_ == 1 : (env && std::strcmp(env, "1") == 0);
  const uint32_t ndev = devices();
  const uint64_t tmo = P.timeout_ns;
  const bool inj = injected_;
  if (inj)
    for (uint32_t i = 0; i < ndev; ++i)
      if (!P.dev[i]->inj) {
        P.dev[i]->inj = std::make_unique<l0::Mem>(stage_ctx(i), l0::MemKind::Host, pf_injected_bytes(d));
        std::memset(P.dev[i]->inj->ptr(), 0, P.dev[i]->inj->size());
      }
  const auto set_ctl = [&](uint32_t pos, uint32_t n) {
    for (uint32_t i = 0; i < ndev; ++i) {
      stage_control(i)->pos = pos;
      stage_control(i)->n_active = n;
    }
  };
  const auto wait_all = [&](const char* what) {
    for (uint32_t i = 0; i < ndev; ++i)
      if (!P.dev[i]->cx.wait_for(tmo))
        fail(std::string("device ") + std::to_string(i) + "'s prefill list did not finish within " +
             std::to_string(opt_.prefill_timeout_ms) + " ms (" + what + ")");
  };
  // One device's walk for a chunk: immediately, or through its recording (B70_PREFILL_REPLAY).
  const auto run_dev = [&](uint32_t i, uint32_t pos, uint32_t C) {
    Qwen4ExpPrefillState::Dev& D = *P.dev[i];
    auto encode = [&] {
      prefill_chunk(D.cx, D.kc, D.s, model_, model_.parts[i], stage_buffers(i), pos, C, eager, inj ? D.inj.get() : nullptr,
                    ndev == 2, mtp_ && i + 1 == ndev ? &mp : nullptr);
    };
    if (!replay) return encode();
    auto it = std::find_if(D.chunks.begin(), D.chunks.end(), [&](const Qwen4ExpPrefillState::Chunk& c) {
      return c.pos == pos && c.rows == C && c.injected == inj;
    });
    if (it == D.chunks.end()) {
      if (D.chunks.size() == 8) D.chunks.erase(D.chunks.begin());   // a FIFO bound on retained lists
      D.chunks.push_back({pos, C, inj, D.cx.capture(encode)});
      it = D.chunks.end() - 1;
    }
    D.cx.replay(*it->recording);
  };

  // Spec 21e's prefix cache: with a block hook every chunk ends at a block end or at the prompt end - Engine::prefill's
  // rule, one shared function - so each completed block is a storable one.
  const bool hooked = static_cast<bool>(block_hook_);
  uint32_t C = 0;
  for (size_t off = 0; off < ids.size(); off += C) {
    C = prefill_chunk_rows(base + uint32_t(off), ids.size() - off, chunk, hooked, kBlock);
    const uint32_t pos = base + uint32_t(off);
    wait_all("before a chunk");   // before touching the ids buffers, the injected rows or a Control block
    for (uint32_t i = 0; i < ndev; ++i) std::memcpy(P.dev[i]->s.ids.ptr(), ids.data() + off, size_t(C) * 4);
    set_ctl(pos, C);
    if (mtp_) {   // spec 21e: the head's pass covers positions pos - 1 .. pos + C - 2 (from 0 at pos 0)
      hctl()->pos = pos == 0 ? 0 : pos - 1;
      hctl()->n_active = mtp_prefill_rows(pos, C);
    }
    if (inj) {   // the caller's lists for this chunk, into the device holding each QSA layer
      for (uint32_t l = 0; l < d.layers; ++l) {
        if (!d.is_qsa(l)) continue;
        const uint32_t qi = d.qsa_before(l), dv = model_.placement.device_of(l);
        pf_inject_(qi, pos, C, reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(P.dev[dv]->inj->ptr()) + pf_list_at(qi)));
      }
    }
    if (ndev == 1) {
      run_dev(0, pos, C);
      wait_all("one card");
    } else {
      PipelineLink& L = *P.link;
      Qwen4ExpPrefillState::Dev& D0 = *P.dev[0];
      Qwen4ExpPrefillState::Dev& D1 = *P.dev[1];
      uint8_t* land = L.landing.as<uint8_t>();
      const size_t rows = size_t(C) * d.hc_n() * 2;
      require(rows <= L.layout.resid_bytes && L.layout.sumsq_bytes == 0,
              "the prefill landing buffer does not hold the chunk's H rows (and no norm sums)");
      L.event->host_reset();
      // P4's test hook (drop_next_handoff, decode's): device 0's half of this chunk is not appended - a hand-off that
      // never arrives. Not under replay (device 1's replay waits without a bound).
      const bool send = !drop_next_ || replay;
      drop_next_ = false;
      if (send) {
        run_dev(0, pos, C);
        D0.cx.copy(land, D0.s.H.ptr(), rows);
        D0.cx.signal(L.event->handle());
      }
      D1.cx.wait_event(L.event->handle());
      D1.cx.copy(D1.s.H.ptr(), land, rows);
      run_dev(1, pos, C);
      if (!D1.cx.wait_for(tmo)) {
        // Device 1 still waits (on the event device 0 never signalled, or on a hung walk): release it so its list
        // drains on garbage instead of holding the device, then report.
        L.event->host_signal();
        const bool drained = D1.cx.wait_for(tmo);
        const bool d0 = D0.cx.wait_for(tmo);
        fail("device 1's prefill chunk at " + std::to_string(pos) + " (" + std::to_string(C) +
             " rows) did not finish within " + std::to_string(opt_.prefill_timeout_ms) + " ms - " +
             (d0 ? "device 0 finished but its rows never released device 1" : "device 0 did not finish either") +
             " (the prefill link's cross-device event; device 1 " + (drained ? "released" : "did not drain") + ")" +
             (send ? "" : " [device 0's half was not appended: drop_next_handoff()]"));
      }
      if (!D0.cx.wait_for(tmo))
        fail("device 0's prefill chunk at " + std::to_string(pos) + " did not finish within " +
             std::to_string(opt_.prefill_timeout_ms) + " ms");
    }
    const uint32_t end = pos + C;
    if (hooked && off + C < ids.size() && end % kBlock == 0) {
      // A mid-prompt block end: the state holds exactly [0, end); say so in every Control before the hook runs, so a
      // hook that throws leaves pos matching the chunks written.
      set_ctl(end, 0);
      block_hook_(end, true);
    }
  }

  // The tail on the last device: pos = base + L - 1 and n_active = 1 make argmax_stage2 leave pos = base + L and the
  // first generated id in cur_token[0]; `C - 1` is the last chunk's last row.
  set_ctl(base + uint32_t(ids.size()) - 1, 1);
  {
    Qwen4ExpPrefillState::Dev& D = *P.dev[ndev - 1];
    const void* R_last = mtp_ ? static_cast<const uint8_t*>(mp.R->ptr()) + size_t(C) * d.hc_n() * 2 : nullptr;
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
}

}  // namespace runtime::qwen4exp
