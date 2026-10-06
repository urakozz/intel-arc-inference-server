// KolibriEngine::prefill and its accessors - the only KolibriEngine members not compiled into
// b70_kolibri_runtime (kolibri_engine.h says why: K2Engine's arrangement, k2_prefill_engine.cc).
//
// **One card:** each chunk is device 0's walk (kolibri_prefill.cc) on its immediate list, then a wait.
// **Two cards** (spec 16b's pieces, as 20c Task 6 did for decode): device 0's list runs the embedding
// and layers [0, s), then two device-to-device copies - the chunk's resid rows [C][2560] and their
// [20][kPfC] norm sums - into the prefill link's landing buffer on device 1 and a barrier signalling the
// link's cross-device event; device 1's list waits on the event, copies both into its own scratch and
// runs layers [s, L). The host resets the event per chunk, appends both lists and waits on device 1 with
// a bound (PipelineOptions::prefill_timeout_ms); a lost hand-off host-signals the event (device 1's list
// drains on garbage instead of holding the device), throws naming it and marks the engine until reset()
// (PipelineEngine's rule). The last chunk's head runs on device 1, then the Control mirror (device 1's
// block into device 0's). The chunks cross SEQUENTIALLY: device 0 idles while device 1 runs a chunk -
// spec 16c's overlapped chunk pipeline (two landing slots, per-slot events) is plan 20d Task 5's lever.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/kolibri_kernels.h"
#include "runtime/control.h"
#include "runtime/kolibri/kolibri_engine.h"
#include "runtime/kolibri/kolibri_prefill.h"
#include "runtime/kolibri/kolibri_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_chunks.h"

namespace runtime::kolibri {

struct KolibriPrefillState {
  struct Chunk {
    uint32_t pos, rows;
    std::unique_ptr<prefill::Context::Recording> recording;
  };
  struct Dev {
    prefill::Context cx;
    prefill::KernelCache kc;
    KolibriPrefillScratch s;
    // Declared after cx / kc / s: recordings are destroyed first.
    std::vector<Chunk> chunks;
    Dev(l0::Context& c, const model::Kolibri1Desc& d) : cx(c), kc(c), s(c, d) {}
  };
  std::vector<std::unique_ptr<Dev>> dev;
  std::unique_ptr<PipelineLink> link;   // two devices: `copy` mode at chunk size (pf_landing_layout)
  uint64_t timeout_ns = 0;
  ~KolibriPrefillState() {   // never free a scratch under a running list
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
void destroy_prefill(KolibriPrefillState* p) { delete p; }
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::kolibri::KolibriEngine::prefill: " + what);
}
}  // namespace

void KolibriEngine::prepare_prefill() {
  if (pf_) return;
  const model::Kolibri1Desc& d = model_.desc;
  const bool eager = attn_ == KolAttn::Eager;
  // A missing binary is named here, before anything is allocated or appended (the decode capture's
  // rule); the head's binaries are decode's, which the capture already checked.
  for (const std::string& v : kernels::kolibri::prefill_variants(d.attn, eager))
    require(std::ifstream(kernels::path(v)).good(),
            v + " is not compiled (" + kernels::path(v) + "): build with B70_KOLIBRI=ON (spec 20d)");
  auto* ps = new KolibriPrefillState();
  pf_ = std::unique_ptr<KolibriPrefillState, void (*)(KolibriPrefillState*)>(ps, &destroy_prefill);
  ps->timeout_ns = uint64_t(opt_.prefill_timeout_ms) * 1000000ull;
  for (uint32_t i = 0; i < devices(); ++i) ps->dev.push_back(std::make_unique<KolibriPrefillState::Dev>(stage_ctx(i), d));
  if (devices() == 2)
    ps->link = std::make_unique<PipelineLink>(stage_ctx(0), stage_ctx(1), pf_landing_layout(d), PpHandoff::Copy);
  pf_dev_bytes_.assign(devices(), 0);
  pf_bytes_ = 0;
  for (uint32_t i = 0; i < devices(); ++i) {
    pf_dev_bytes_[i] = ps->dev[i]->s.bytes() + (ps->link ? ps->link->bytes(i) : 0);
    pf_bytes_ += pf_dev_bytes_[i];
    require(ps->dev[i]->s.bytes() == prefill_sizes(d).total(), "the prefill scratch is not prefill_sizes()'s");
    require(!ps->link || ps->link->bytes(i) == pf_link_bytes(d, i), "the prefill link is not pf_link_bytes()'s");
  }
  pf_settle_ = [ps] { return ps->settle(); };
}

size_t KolibriEngine::prefill_launches(uint32_t dev) const {
  return pf_ && dev < pf_->dev.size() ? pf_->dev[dev]->cx.launches() : 0;
}
size_t KolibriEngine::prefill_launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < devices(); ++i) n += prefill_launches(i);
  return n;
}

std::vector<uint32_t> KolibriEngine::read_prefill_routes() {
  require(bool(pf_), "read_prefill_routes before the first prefill");
  const model::Kolibri1Desc& d = model_.desc;
  std::vector<uint32_t> v(prefill_sizes(d).routes / 4);
  for (uint32_t i = 0; i < devices(); ++i) {
    KolibriPrefillState::Dev& D = *pf_->dev[i];
    require(D.cx.wait_for(pf_->timeout_ns), "device " + std::to_string(i) + "'s prefill list is still running");
    const uint32_t first = model_.placement.first(i), n = model_.placement.count(i);
    stage_imm(i).copy(v.data() + pf_route_at(first) / 4, static_cast<const uint8_t*>(D.s.routes.ptr()) + pf_route_at(first),
                   size_t(n) * kPfC * kRouteWords * 4);
  }
  return v;
}

void KolibriEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  const model::Kolibri1Desc& d = model_.desc;
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
              std::to_string(max_len_) + " - the full layers' KV and the RoPE table stop there");
  for (uint32_t i = 0; i < devices(); ++i)   // no decode step in flight writes the KV or the Control
    require(settle(i), "device " + std::to_string(i) + "'s last decode step is still running");

  prepare_prefill();
  KolibriPrefillState& P = *pf_;
  const bool eager = attn_ == KolAttn::Eager;
  const char* env = std::getenv("B70_PREFILL_REPLAY");
  const bool replay = pf_replay_ >= 0 ? pf_replay_ == 1 : (env && std::strcmp(env, "1") == 0);
  const uint32_t ndev = devices();
  const uint64_t tmo = P.timeout_ns;
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
    KolibriPrefillState::Dev& D = *P.dev[i];
    auto encode = [&] { prefill_chunk(D.cx, D.kc, D.s, model_.parts[i], d, stage_buffers(i), pos, C, eager); };
    if (!replay) return encode();
    auto it = std::find_if(D.chunks.begin(), D.chunks.end(),
                           [&](const KolibriPrefillState::Chunk& c) { return c.pos == pos && c.rows == C; });
    if (it == D.chunks.end()) {
      if (D.chunks.size() == 8) D.chunks.erase(D.chunks.begin());   // a FIFO bound on retained lists
      D.chunks.push_back({pos, C, D.cx.capture(encode)});
      it = D.chunks.end() - 1;
    }
    D.cx.replay(*it->recording);
  };

  // Spec 20e's prefix cache: with a block hook every chunk ends at a block end or at the prompt end -
  // Engine::prefill's rule, one shared function - so each completed block is a storable one.
  const bool hooked = static_cast<bool>(block_hook_);
  uint32_t C = 0;
  for (size_t off = 0; off < ids.size(); off += C) {
    C = prefill_chunk_rows(base + uint32_t(off), ids.size() - off, chunk, hooked, kBlock);
    const uint32_t pos = base + uint32_t(off);
    wait_all("before a chunk");   // before touching the ids buffer or a Control block
    std::memcpy(P.dev[0]->s.ids.ptr(), ids.data() + off, size_t(C) * 4);
    set_ctl(pos, C);
    if (ndev == 1) {
      run_dev(0, pos, C);
      wait_all("one card");
    } else {
      PipelineLink& L = *P.link;
      KolibriPrefillState::Dev& D0 = *P.dev[0];
      KolibriPrefillState::Dev& D1 = *P.dev[1];
      uint8_t* land = L.landing.as<uint8_t>();
      const size_t rows = size_t(C) * d.hidden * 2, sums = D0.s.sumsq_r.size();
      require(rows <= L.layout.resid_bytes && sums == L.layout.sumsq_bytes,
              "the prefill landing buffer does not hold the chunk's rows and exactly its norm sums");
      L.event->host_reset();
      run_dev(0, pos, C);
      D0.cx.copy(land, D0.s.resid.ptr(), rows);
      D0.cx.copy(land + L.layout.sumsq_off, D0.s.sumsq_r.ptr(), sums);
      D0.cx.signal(L.event->handle());
      D1.cx.wait_event(L.event->handle());
      D1.cx.copy(D1.s.resid.ptr(), land, rows);
      D1.cx.copy(D1.s.sumsq_r.ptr(), land + L.layout.sumsq_off, sums);
      run_dev(1, pos, C);
      if (!D1.cx.wait_for(tmo)) {
        // Device 1 still waits (on the event device 0 never signalled, or on a hung walk): release it so
        // its list drains on garbage instead of holding the device, then report.
        L.event->host_signal();
        const bool drained = D1.cx.wait_for(tmo);
        const bool d0 = D0.cx.wait_for(tmo);
        fail("device 1's prefill chunk at " + std::to_string(pos) + " (" + std::to_string(C) +
             " rows) did not finish within " + std::to_string(opt_.prefill_timeout_ms) + " ms - " +
             (d0 ? "device 0 finished but its rows never released device 1" : "device 0 did not finish either") +
             " (the prefill link's cross-device event; device 1 " + (drained ? "released" : "did not drain") + ")");
      }
      if (!D0.cx.wait_for(tmo))
        fail("device 0's prefill chunk at " + std::to_string(pos) + " did not finish within " +
             std::to_string(opt_.prefill_timeout_ms) + " ms");
    }
    const uint32_t end = pos + C;
    if (hooked && off + C < ids.size() && end % kBlock == 0) {
      // A mid-prompt block end: the KV holds exactly [0, end); say so in every Control before the hook
      // runs, so a hook that throws leaves pos matching the chunks written.
      set_ctl(end, 0);
      block_hook_(end, true);
    }
  }

  // The tail on the last device: pos = base + L - 1 and n_active = 1 make argmax_stage2 leave pos = base + L
  // and the first generated id in cur_token[0]; `C - 1` is the last chunk's last row.
  set_ctl(base + uint32_t(ids.size()) - 1, 1);
  {
    KolibriPrefillState::Dev& D = *P.dev[ndev - 1];
    prefill_head(D.cx, D.kc, D.s, model_.parts[ndev - 1], d, stage_buffers(ndev - 1), C - 1);
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

}  // namespace runtime::kolibri
