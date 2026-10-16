// MuseEngine::prefill and its accessors - the only MuseEngine members not compiled into b70_muse_runtime
// (muse_engine.h says why: Kolibri's arrangement, runtime/kolibri/kolibri_prefill_engine.cc).
//
// **One card:** each chunk is device 0's walk (muse_prefill.cc) on its immediate list, then a wait.
// **Two cards** (spec 16b's pieces, spec 20 §12's sequential hand-off): device 0's list runs the embedding and layers
// [0, s), then two device-to-device copies - the chunk's resid rows [C][6656] and their [26][kPfC] norm sums - into the
// prefill link's landing buffer on device 1 and a barrier signalling the link's cross-device event; device 1's list
// waits on the event, copies both into its own scratch and runs layers [s, L). The host resets the event per chunk,
// appends both lists and waits on device 1 with a bound (MuseOptions::prefill_timeout_ms); a lost hand-off
// host-signals the event (device 1's list drains on garbage instead of holding the device), throws naming it and marks
// the engine until reset(). The last chunk's head runs on device 1, then the Control mirror (device 1's block into
// device 0's). The chunks cross SEQUENTIALLY: device 0 idles while device 1 runs a chunk.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/muse_kernels.h"
#include "runtime/control.h"
#include "runtime/muse/muse_engine.h"
#include "runtime/muse/muse_prefill.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_chunks.h"

namespace runtime::muse {

struct MusePrefillState {
  struct Chunk {
    uint32_t pos, rows;
    std::unique_ptr<prefill::Context::Recording> recording;
  };
  struct Dev {
    prefill::Context cx;
    prefill::KernelCache kc;
    MusePrefillScratch s;
    // Declared after cx / kc / s: recordings are destroyed first.
    std::vector<Chunk> chunks;
    Dev(l0::Context& c, const model::MuseGlimmerDesc& d) : cx(c), kc(c), s(c, d) {}
  };
  std::vector<std::unique_ptr<Dev>> dev;
  std::unique_ptr<PipelineLink> link;   // two devices: `copy` mode at chunk size (pf_landing_layout)
  // Spec 23f: a drafter's tap staging [kPfC][taps][hidden] bf16 (runtime::muse::spec_prefill_taps_bytes), one card.
  std::unique_ptr<l0::Mem> taps_stage;
  uint64_t timeout_ns = 0;
  ~MusePrefillState() {   // never free a scratch under a running list
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
void destroy_prefill(MusePrefillState* p) { delete p; }
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::muse::MuseEngine::prefill: " + what);
}
}  // namespace

void MuseEngine::prepare_prefill() {
  if (pf_) return;
  const model::MuseGlimmerDesc& d = w_.desc;
  const bool eager = attn_ == MuseAttn::Eager;
  // A missing binary is named here, before anything is allocated or appended (the decode capture's rule); the head's
  // lm_head and argmax binaries are decode's, which the capture already checked.
  for (const std::string& v : kernels::muse::prefill_variants(eager))
    require(std::ifstream(kernels::path(v)).good(),
            v + " is not compiled (" + kernels::path(v) + "): build with B70_MUSE=ON (spec 23d)");
  auto* ps = new MusePrefillState();
  pf_ = std::unique_ptr<MusePrefillState, void (*)(MusePrefillState*)>(ps, &destroy_prefill);
  ps->timeout_ns = uint64_t(opt_.prefill_timeout_ms) * 1000000ull;
  for (uint32_t i = 0; i < devices(); ++i) ps->dev.push_back(std::make_unique<MusePrefillState::Dev>(stage_ctx(i), d));
  if (devices() == 2)
    ps->link = std::make_unique<PipelineLink>(stage_ctx(0), stage_ctx(1), pf_landing_layout(d), PpHandoff::Copy);
  if (opt_.spec.drafter.on())   // spec 23f: the planned tap staging (one card)
    ps->taps_stage = std::make_unique<l0::Mem>(stage_ctx(0), l0::MemKind::Device,
                                               spec_prefill_taps_bytes(d, opt_.spec.drafter));
  pf_dev_bytes_.assign(devices(), 0);
  pf_bytes_ = 0;
  for (uint32_t i = 0; i < devices(); ++i) {
    pf_dev_bytes_[i] = ps->dev[i]->s.bytes() + (ps->link ? ps->link->bytes(i) : 0) +
                       (i == 0 && ps->taps_stage ? ps->taps_stage->size() : 0);
    pf_bytes_ += pf_dev_bytes_[i];
    require(ps->dev[i]->s.bytes() == prefill_sizes(d).total(), "the prefill scratch is not prefill_sizes()'s");
    require(!ps->link || ps->link->bytes(i) == pf_link_bytes(d, i), "the prefill link is not pf_link_bytes()'s");
  }
  pf_settle_ = [ps] { return ps->settle(); };
}

size_t MuseEngine::prefill_launches(uint32_t dev) const {
  return pf_ && dev < pf_->dev.size() ? pf_->dev[dev]->cx.launches() : 0;
}
size_t MuseEngine::prefill_launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < devices(); ++i) n += prefill_launches(i);
  return n;
}

void MuseEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  const model::MuseGlimmerDesc& d = w_.desc;
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  require(!ids.empty(), "no ids");
  if (chunk == 0) chunk = kPfC;
  require(chunk <= kPfC, "chunk " + std::to_string(chunk) + " exceeds kPfC " + std::to_string(kPfC) +
                             " (the 4096-slot ring holds a chunk of 2048 and the 2047 keys before it)");
  for (uint32_t id : ids)
    require(id < d.vocab, "id " + std::to_string(id) + " is outside the vocabulary (" + std::to_string(d.vocab) +
                              " rows) - mg_pf_embed has no debug_flag channel, so the host is the only bound");
  const uint32_t base = stage_control(devices() - 1)->pos;
  require(size_t(base) + ids.size() <= max_len_,
          "pos " + std::to_string(base) + " + " + std::to_string(ids.size()) + " ids exceeds max_len " +
              std::to_string(max_len_) + " - the full layers' KV and the RoPE table stop there");
  for (uint32_t i = 0; i < devices(); ++i)   // no decode step in flight writes the KV or the Control
    require(settle(i), "device " + std::to_string(i) + "'s last decode step is still running");

  prepare_prefill();
  MusePrefillState& P = *pf_;
  const bool eager = attn_ == MuseAttn::Eager;
  const char* env = std::getenv("B70_PREFILL_REPLAY");
  const bool replay = pf_replay_ >= 0 ? pf_replay_ == 1 : (env && std::strcmp(env, "1") == 0);
  const uint32_t ndev = devices();
  // The bound a chunk's lists are waited for: prefill_wait_ms (muse_sizes.h) - the configured bound, or under eager one
  // bound per 2048 keys of the chunk's deepest row (the portable eager build's cost grows with depth).
  uint64_t wait_ms = opt_.prefill_timeout_ms, tmo = P.timeout_ns;
  const auto set_bound = [&](uint32_t depth) {
    wait_ms = prefill_wait_ms(opt_.prefill_timeout_ms, attn_, depth);
    tmo = wait_ms * 1000000ull;
  };
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
             std::to_string(wait_ms) + " ms (" + what + ")");
  };
  // Spec 23f: with a drafter, every chunk taps its rows inside the drafter's window of this call's end (the positions
  // its ring reads back: [end - window(), end)) into the staging, projected after the chunk (16 rows a replay).
  // Positions before this call keep the ring's own rows (the session's, or a restored snapshot's). Copies cannot sit in
  // a recorded chunk: B70_PREFILL_REPLAY is refused with a drafter.
  uint32_t df_from = 0;
  if (drafter_) {
    require(!replay, "a drafter's prefill taps are copies, which a recorded chunk cannot hold - unset "
                     "B70_PREFILL_REPLAY with --spec dflash");
    require(ndev == 1 && P.taps_stage, "a drafter's prefill taps run on one card, into the planned staging");
    const uint32_t end_all = base + uint32_t(ids.size());
    df_from = end_all > drafter_->window() ? end_all - drafter_->window() : 0;
  }
  MusePrefillTaps df_taps;
  // One device's walk for a chunk: immediately, or through its recording (B70_PREFILL_REPLAY).
  const auto run_dev = [&](uint32_t i, uint32_t pos, uint32_t C) {
    MusePrefillState::Dev& D = *P.dev[i];
    const MusePrefillTaps* tp = drafter_ && df_taps.r0 < C ? &df_taps : nullptr;
    auto encode = [&] { prefill_chunk(D.cx, D.kc, D.s, w_.parts[i], d, stage_buffers(i), pos, C, eager, tp); };
    if (!replay) return encode();
    auto it = std::find_if(D.chunks.begin(), D.chunks.end(),
                           [&](const MusePrefillState::Chunk& c) { return c.pos == pos && c.rows == C; });
    if (it == D.chunks.end()) {
      if (D.chunks.size() == 8) D.chunks.erase(D.chunks.begin());   // a FIFO bound on retained lists
      D.chunks.push_back({pos, C, D.cx.capture(encode)});
      it = D.chunks.end() - 1;
    }
    D.cx.replay(*it->recording);
  };

  // With a block hook every chunk ends at a block end or at the prompt end - Engine::prefill's rule, one shared
  // function - so each completed block is a storable one (spec 23e).
  const bool hooked = static_cast<bool>(block_hook_);
  uint32_t C = 0;
  for (size_t off = 0; off < ids.size(); off += C) {
    C = prefill_chunk_rows(base + uint32_t(off), ids.size() - off, chunk, hooked, kBlock);
    const uint32_t pos = base + uint32_t(off);
    wait_all("before a chunk");   // before touching the ids buffer or a Control block (the previous chunk's bound)
    set_bound(pos + C);
    std::memcpy(P.dev[0]->s.ids.ptr(), ids.data() + off, size_t(C) * 4);
    set_ctl(pos, C);
    if (drafter_) {
      df_taps.dst = P.taps_stage->ptr();
      df_taps.layers = &drafter_->tap_layers();
      df_taps.r0 = df_from > pos ? std::min<uint32_t>(C, df_from - pos) : 0;
    }
    if (ndev == 1) {
      run_dev(0, pos, C);
      wait_all("one card");
      if (drafter_ && df_taps.r0 < C)
        drafter_project_staged(P.taps_stage->as<uint8_t>(), pos + df_taps.r0, C - df_taps.r0);
    } else {
      PipelineLink& L = *P.link;
      MusePrefillState::Dev& D0 = *P.dev[0];
      MusePrefillState::Dev& D1 = *P.dev[1];
      uint8_t* land = L.landing.as<uint8_t>();
      const size_t rows = size_t(C) * d.hidden * 2, sums = D0.s.sumsq_r.size();
      require(rows <= L.layout.resid_bytes && sums == L.layout.sumsq_bytes,
              "the prefill landing buffer does not hold the chunk's rows and exactly its norm sums");
      L.event->host_reset();
      // P4's test hook (drop_next_handoff, decode's): device 0's half of this chunk is not appended - a hand-off that
      // never arrives. Not under replay (device 1's replay waits without a bound).
      const bool send = !drop_next_ || replay;
      drop_next_ = false;
      if (send) {
        run_dev(0, pos, C);
        D0.cx.copy(land, D0.s.resid.ptr(), rows);
        D0.cx.copy(land + L.layout.sumsq_off, D0.s.sumsq_r.ptr(), sums);
        D0.cx.signal(L.event->handle());
      }
      D1.cx.wait_event(L.event->handle());
      D1.cx.copy(D1.s.resid.ptr(), land, rows);
      D1.cx.copy(D1.s.sumsq_r.ptr(), land + L.layout.sumsq_off, sums);
      run_dev(1, pos, C);
      if (!D1.cx.wait_for(tmo)) {
        // Device 1 still waits (on the event device 0 never signalled, or on a hung walk): release it so its list
        // drains on garbage instead of holding the device, then report.
        L.event->host_signal();
        const bool drained = D1.cx.wait_for(tmo);
        const bool d0 = D0.cx.wait_for(tmo);
        fail("device 1's prefill chunk at " + std::to_string(pos) + " (" + std::to_string(C) +
             " rows) did not finish within " + std::to_string(wait_ms) + " ms - " +
             (d0 ? "device 0 finished but its rows never released device 1" : "device 0 did not finish either") +
             " (the prefill link's cross-device event; device 1 " + (drained ? "released" : "did not drain") + ")" +
             (send ? "" : " [device 0's half was not appended: drop_next_handoff()]"));
      }
      if (!D0.cx.wait_for(tmo))
        fail("device 0's prefill chunk at " + std::to_string(pos) + " did not finish within " +
             std::to_string(wait_ms) + " ms");
    }
    const uint32_t end = pos + C;
    if (hooked && off + C < ids.size() && end % kBlock == 0) {
      // A mid-prompt block end: the KV holds exactly [0, end); say so in every Control before the hook runs, so a
      // hook that throws leaves pos matching the chunks written.
      set_ctl(end, 0);
      block_hook_(end, true);
    }
  }

  // The tail on the last device: pos = base + L - 1 and n_active = 1 make argmax_stage2 leave pos = base + L and the
  // first generated id in cur_token[0]; `C - 1` is the last chunk's last row.
  set_ctl(base + uint32_t(ids.size()) - 1, 1);
  {
    MusePrefillState::Dev& D = *P.dev[ndev - 1];
    prefill_head(D.cx, D.kc, D.s, w_.parts[ndev - 1], d, stage_buffers(ndev - 1), C - 1);
    if (!D.cx.wait_for(tmo))
      fail("device " + std::to_string(ndev - 1) + "'s prefill head did not finish within " +
           std::to_string(wait_ms) + " ms");
  }
  // The token's way back (16b's mirror): device 0 reads pos and cur_token from its own block next step.
  if (ndev == 2) std::memcpy(stage_control(0), stage_control(1), sizeof(Control));
  if (hooked) {   // the prompt end, the first generated id in cur_token
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
}

// Spec 23f: the staged rows [n][taps][hidden] at positions pos .. pos + n - 1 through the drafter's projection, 16 a
// replay on device 0's queue (the prefill list has completed).
void MuseEngine::drafter_project_staged(const uint8_t* staged, uint32_t pos, uint32_t n) {
  const size_t row = drafter_->tap_layers().size() * size_t(w_.desc.hidden) * 2;
  for (uint32_t r = 0; r < n; r += kVerifyMaxRows) {
    const uint32_t g = std::min<uint32_t>(kVerifyMaxRows, n - r);
    stage_imm(0).copy(drafter_->taps().ptr(), staged + size_t(r) * row, size_t(g) * row);
    replay_aux(drafter_->projection(pos + r, g));
  }
}

}  // namespace runtime::muse
