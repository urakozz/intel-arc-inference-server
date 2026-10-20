// Lfm2MoeEngine::prefill and its accessors - the only Lfm2MoeEngine members not compiled into b70_lfm2moe_runtime
// (lfm2moe_engine.h says why: Kolibri's arrangement, runtime/kolibri/kolibri_prefill_engine.cc).
//
// **One card:** each chunk is device 0's walk (lfm2moe_prefill.cc) on its immediate list, then a wait.
// **Two cards** (spec 16b's pieces, decision 10, as 24c did for decode): device 0's list runs the gather and layers
// [0, s), then device-to-device copies - the chunk's h rows [C][2048] (and, at a DENSE cut, the down's pending fp32
// rows [C][2048]: 24c's departure, spec 24 §14) - into the prefill link's landing buffer on device 1 and a barrier
// signalling the link's cross-device event; device 1's list waits on the event, copies them into its own h0 (and
// partials) and runs layers [s, L). The host resets the event per chunk, appends both lists and waits on device 1 with
// a bound (PipelineOptions::prefill_timeout_ms); a lost hand-off host-signals the event (device 1's list drains on
// garbage instead of holding the device), throws naming it and marks the engine until reset() (PipelineEngine's
// rule). The last chunk's head runs on device 1, then the Control mirror (device 1's block into device 0's). The chunks
// cross SEQUENTIALLY: device 0 idles while device 1 runs a chunk (spec 16c's overlap is a later lever).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/lfm2moe_kernels.h"
#include "runtime/control.h"
#include "runtime/lfm2moe/lfm2moe_engine.h"
#include "runtime/lfm2moe/lfm2moe_prefill.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_chunks.h"

namespace runtime::lfm2moe {

struct Lfm2MoePrefillState {
  struct Chunk {
    uint32_t pos, rows;
    std::unique_ptr<prefill::Context::Recording> recording;
  };
  struct Dev {
    prefill::Context cx;
    prefill::KernelCache kc;
    Lfm2MoePrefillScratch s;
    // Declared after cx / kc / s: recordings are destroyed first.
    std::vector<Chunk> chunks;
    Dev(l0::Context& c, const model::Lfm2MoeDesc& d) : cx(c), kc(c), s(c, d) {}
  };
  std::vector<std::unique_ptr<Dev>> dev;
  std::unique_ptr<PipelineLink> link;   // two devices: `copy` mode at chunk size (pf_landing_layout)
  uint64_t timeout_ns = 0;
  ~Lfm2MoePrefillState() {   // never free a scratch under a running list
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
void destroy_prefill(Lfm2MoePrefillState* p) { delete p; }
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine::prefill: " + what);
}
// The __kernel entry points of one prefill binary (kernels::lfm2moe::prefill_variants' names), for the pre-load.
std::vector<const char*> prefill_entries(const std::string& v) {
  auto starts = [&](const char* p) { return v.rfind(p, 0) == 0; };
  if (starts("lfm_pf_embed_gather")) return {"lfm_pf_embed_gather"};
  if (starts("lfm_pf_conv")) return {"lfm_pf_conv_seed", "lfm_pf_conv"};
  if (starts("lfm_pf_moe")) return {"lfm_pf_sort", "lfm_pf_gather", "lfm_pf_dequant_gu", "lfm_pf_dequant_dn", "lfm_pf_combine"};
  if (starts("lfm_pf_attn")) return {"lfm_pf_attn"};
  if (starts("lfm_res_norm")) return {"lfm_res_norm"};
  if (starts("lfm_router")) return {"lfm_router"};
  if (starts("lfm_route")) return {"lfm_route"};
  if (starts("lfm_attn_prep")) return {"lfm_attn_prep"};
  if (starts("pf_dequant_slab")) return {"pf_dequant_slab"};
  if (starts("pf_moe_gemm")) return {"pf_moe_gemm"};
  if (starts("pf_gemm")) return {"pf_gemm"};
  throw std::runtime_error("runtime::lfm2moe::prepare_prefill: no entry list for " + v);
}
}  // namespace

void Lfm2MoeEngine::prepare_prefill() {
  if (pf_) return;
  const model::Lfm2MoeDesc& d = model_.desc;
  const bool eager = attn_ == LfmAttn::Eager;
  // A missing binary is named here, before anything is allocated or appended (the decode capture's rule); the head's
  // binaries are decode's, which the capture already checked.
  for (const std::string& v : kernels::lfm2moe::prefill_variants(d, eager, model_.conv_tiles))
    require(std::ifstream(kernels::path(v)).good(),
            v + " is not compiled (" + kernels::path(v) + "): build with B70_LFM2MOE=ON (spec 24d)");
  auto* ps = new Lfm2MoePrefillState();
  pf_ = std::unique_ptr<Lfm2MoePrefillState, void (*)(Lfm2MoePrefillState*)>(ps, &destroy_prefill);
  ps->timeout_ns = uint64_t(opt_.prefill_timeout_ms) * 1000000ull;
  for (uint32_t i = 0; i < devices(); ++i)
    ps->dev.push_back(std::make_unique<Lfm2MoePrefillState::Dev>(stage_ctx(i), d));
  if (devices() == 2)
    ps->link = std::make_unique<PipelineLink>(stage_ctx(0), stage_ctx(1), pf_landing_layout(d, model_.place),
                                              PpHandoff::Copy);
  // The route rows are read back whole (read_prefill_routes): zero them once, so rows no chunk wrote (a dense layer's,
  // rows past the last chunk's C) read the same on every engine.
  for (uint32_t i = 0; i < devices(); ++i) stage_imm(i).fill(ps->dev[i]->s.routes.ptr(), 0u, ps->dev[i]->s.routes.size());
  // Every module the walk and the head bind, created here - outside a timed prefill (KernelCache creates lazily).
  for (uint32_t i = 0; i < devices(); ++i) {
    prefill::KernelCache& kc = ps->dev[i]->kc;
    for (const std::string& v : kernels::lfm2moe::prefill_variants(d, eager, model_.conv_tiles))
      for (const char* e : prefill_entries(v)) (void)kc(v, e);
  }
  {   // the head's (decode's binaries, on the last device)
    namespace kl = kernels::lfm2moe;
    prefill::KernelCache& kc = ps->dev[devices() - 1]->kc;
    const bool int8 = bool(model_.parts[devices() - 1].head_int8);
    (void)kc(kl::res_norm_variant(1, prefill_resid_out(d, model_.place, devices() - 1).pending, false), "lfm_res_norm");
    (void)kc(kl::head_variant(int8), int8 ? "gemv_i8w" : "gemv_bf16");
    (void)kc(kl::argmax1_variant(1), "argmax_stage1");
    (void)kc(kl::argmax2_variant(), "argmax_stage2");
  }
  pf_dev_bytes_.assign(devices(), 0);
  pf_bytes_ = 0;
  for (uint32_t i = 0; i < devices(); ++i) {
    pf_dev_bytes_[i] = ps->dev[i]->s.bytes() + (ps->link ? ps->link->bytes(i) : 0);
    pf_bytes_ += pf_dev_bytes_[i];
    require(ps->dev[i]->s.bytes() == prefill_sizes(d).total(), "the prefill scratch is not prefill_sizes()'s");
    require(!ps->link || ps->link->bytes(i) == pf_link_bytes(d, model_.place, i),
            "the prefill link is not pf_link_bytes()'s");
  }
  pf_settle_ = [ps] { return ps->settle(); };
}

size_t Lfm2MoeEngine::prefill_launches(uint32_t dev) const {
  return pf_ && dev < pf_->dev.size() ? pf_->dev[dev]->cx.launches() : 0;
}
size_t Lfm2MoeEngine::prefill_launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < devices(); ++i) n += prefill_launches(i);
  return n;
}

std::vector<uint32_t> Lfm2MoeEngine::read_prefill_routes() {
  require(bool(pf_), "read_prefill_routes before the first prefill");
  const model::Lfm2MoeDesc& d = model_.desc;
  std::vector<uint32_t> v(prefill_sizes(d).routes / 4);
  for (uint32_t i = 0; i < devices(); ++i) {
    Lfm2MoePrefillState::Dev& D = *pf_->dev[i];
    require(D.cx.wait_for(pf_->timeout_ns), "device " + std::to_string(i) + "'s prefill list is still running");
    const uint32_t first = model_.place.first(i), n = model_.place.count(i);
    stage_imm(i).copy(v.data() + pf_route_at(first) / 4, static_cast<const uint8_t*>(D.s.routes.ptr()) + pf_route_at(first),
                      size_t(n) * kPfC * kRouteWords * 4);
  }
  return v;
}

void Lfm2MoeEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  const model::Lfm2MoeDesc& d = model_.desc;
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  require(!ids.empty(), "no ids");
  if (chunk == 0) chunk = kPfC;
  require(chunk <= kPfC, "chunk " + std::to_string(chunk) + " exceeds kPfC " + std::to_string(kPfC));
  for (uint32_t id : ids)
    require(id < d.vocab, "id " + std::to_string(id) + " is outside the vocabulary (" + std::to_string(d.vocab) +
                              " rows) - lfm_pf_embed_gather has no debug_flag channel, so the host is the only bound");
  const uint32_t base = stage_control(devices() - 1)->pos;
  require(size_t(base) + ids.size() <= max_len_,
          "pos " + std::to_string(base) + " + " + std::to_string(ids.size()) + " ids exceeds max_len " +
              std::to_string(max_len_) + " - the KV and the RoPE table stop there");
  for (uint32_t i = 0; i < devices(); ++i)   // no decode step in flight writes the KV, the rings or the Control
    require(settle(i), "device " + std::to_string(i) + "'s last decode step is still running");

  prepare_prefill();
  Lfm2MoePrefillState& P = *pf_;
  const bool eager = attn_ == LfmAttn::Eager;
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
  // Where each device's walk takes and leaves h (its PrefillResid): a function of the placement alone (the walk
  // asserts it), so a replayed recording needs no walk to know it.
  const PrefillResid in1 = prefill_resid_in(d, model_.place, 1);
  const PrefillResid last = prefill_resid_out(d, model_.place, ndev - 1);
  // One device's walk for a chunk: immediately, or through its recording (B70_PREFILL_REPLAY). The tap (debug) is
  // written on the last chunk only and never under a recording (a copy is not recordable).
  const auto run_dev = [&](uint32_t i, uint32_t pos, uint32_t C, bool last_chunk) {
    Lfm2MoePrefillState::Dev& D = *P.dev[i];
    l0::Mem* tap = !replay && last_chunk ? taps_[i].get() : nullptr;
    auto encode = [&] {
      prefill_chunk(D.cx, D.kc, D.s, model_.parts[i], d, stage_buffers(i), pos, C, eager, model_.conv_tiles,
                    prefill_resid_in(d, model_.place, i), tap);
    };
    if (!replay) return encode();
    auto it = std::find_if(D.chunks.begin(), D.chunks.end(),
                           [&](const Lfm2MoePrefillState::Chunk& c) { return c.pos == pos && c.rows == C; });
    if (it == D.chunks.end()) {
      if (D.chunks.size() == 8) D.chunks.erase(D.chunks.begin());   // a FIFO bound on retained lists
      D.chunks.push_back({pos, C, D.cx.capture(encode)});
      it = D.chunks.end() - 1;
    }
    D.cx.replay(*it->recording);
  };

  uint32_t C = 0;
  for (size_t off = 0; off < ids.size(); off += C) {
    C = prefill_chunk_rows(base + uint32_t(off), ids.size() - off, chunk, false, kPfC);
    const uint32_t pos = base + uint32_t(off);
    const bool last_chunk = off + C == ids.size();
    wait_all("before a chunk");   // before touching the ids buffer or a Control block
    std::memcpy(P.dev[0]->s.ids.ptr(), ids.data() + off, size_t(C) * 4);
    set_ctl(pos, C);
    if (ndev == 1) {
      run_dev(0, pos, C, last_chunk);
      wait_all("one card");
    } else {
      PipelineLink& L = *P.link;
      Lfm2MoePrefillState::Dev& D0 = *P.dev[0];
      Lfm2MoePrefillState::Dev& D1 = *P.dev[1];
      uint8_t* land = L.landing.as<uint8_t>();
      const size_t rows = size_t(C) * d.hidden * 2, pend = in1.pending ? size_t(C) * d.hidden * 4 : 0;
      require(rows <= L.layout.resid_bytes && pend <= L.layout.sumsq_bytes,
              "the prefill landing buffer does not hold the chunk's rows and its pending down rows");
      L.event->host_reset();
      // P4's test hook (drop_next_handoff, decode's): device 0's half of this chunk is not appended - a hand-off that
      // never arrives. Not under replay (device 1's replay waits without a bound).
      const bool send = !drop_next_ || replay;
      drop_next_ = false;
      if (send) {
        run_dev(0, pos, C, last_chunk);
        const PrefillResid r0 = prefill_resid_out(d, model_.place, 0);
        require(r0.pending == in1.pending, "device 0's walk ends with other rows pending than the cut's");
        D0.cx.copy(land, D0.s.h(r0.cur).ptr(), rows);
        if (pend) D0.cx.copy(land + L.layout.sumsq_off, D0.s.partials.ptr(), pend);
        D0.cx.signal(L.event->handle());
      }
      D1.cx.wait_event(L.event->handle());
      D1.cx.copy(D1.s.h0.ptr(), land, rows);
      if (pend) D1.cx.copy(D1.s.partials.ptr(), land + L.layout.sumsq_off, pend);
      run_dev(1, pos, C, last_chunk);
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
  }

  // The tail on the last device: pos = base + L - 1 and n_active = 1 make argmax_stage2 leave pos = base + L and the
  // first generated id in cur_token[0]; `C - 1` is the last chunk's last row.
  set_ctl(base + uint32_t(ids.size()) - 1, 1);
  {
    Lfm2MoePrefillState::Dev& D = *P.dev[ndev - 1];
    prefill_head(D.cx, D.kc, D.s, model_.parts[ndev - 1], d, stage_buffers(ndev - 1), C - 1, last,
                 replay ? nullptr : taps_[ndev - 1].get());
    if (!D.cx.wait_for(tmo))
      fail("device " + std::to_string(ndev - 1) + "'s prefill head did not finish within " +
           std::to_string(opt_.prefill_timeout_ms) + " ms");
  }
  // The token's way back (16b's mirror): device 0 reads pos and cur_token from its own block next step.
  if (ndev == 2) std::memcpy(stage_control(0), stage_control(1), sizeof(Control));
}

}  // namespace runtime::lfm2moe
