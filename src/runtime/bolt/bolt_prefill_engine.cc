// BoltEngine::prefill and its accessors - the only BoltEngine members not compiled into b70_bolt_runtime (bolt_engine.h says
// why: Kolibri's arrangement, runtime/kolibri/kolibri_prefill_engine.cc).
//
// **One card:** each chunk is device 0's walk (bolt_prefill.cc) on its immediate list, then a wait.
// **Two cards** (spec 16b's pieces, Decision 4; Kolibri's sequential hand-off per chunk, spec 20 §12): device 0's list runs
// the gather and blocks [0, s), then device-to-device copies - the chunk's h rows [C][4096] and block s - 1's pending GEMM
// rows ([C][4096] fp32: o_proj's / out_proj's, or fc2's and shared_down's) - into the prefill link's landing buffer on
// device 1 and a barrier signalling the link's cross-device event; device 1's list waits on the event, copies them into its
// own h0 and pa (and pb) and runs blocks [s, L). The host resets the event per chunk, appends both lists and waits on device
// 1 with a bound (PipelineOptions::prefill_timeout_ms); a lost hand-off host-signals the event (device 1's list drains on
// garbage instead of holding the device), throws naming it and marks the engine until reset(). The last chunk's head runs
// on device 1, then the Control mirror (device 1's block into device 0's). The chunks cross SEQUENTIALLY: device 0 idles
// while device 1 runs a chunk (spec 16c's overlap is a later lever).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/bolt_kernels.h"
#include "kernels/kernels.h"
#include "runtime/bolt/bolt_engine.h"
#include "runtime/bolt/bolt_prefill.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace runtime::bolt {

struct BoltPrefillState {
  struct Chunk {
    uint32_t pos, rows;
    std::unique_ptr<prefill::Context::Recording> recording;
  };
  struct Dev {
    prefill::Context cx;
    prefill::KernelCache kc;
    BoltPrefillScratch s;
    // Declared after cx / kc / s: recordings are destroyed first.
    std::vector<Chunk> chunks;
    Dev(l0::Context& c, const model::BoltDesc& d) : cx(c), kc(c), s(c, d) {}
  };
  std::vector<std::unique_ptr<Dev>> dev;
  std::unique_ptr<PipelineLink> link;   // two devices: `copy` mode at chunk size (pf_landing_layout)
  uint64_t timeout_ns = 0;
  ~BoltPrefillState() {   // never free a scratch under a running list
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
void destroy_prefill(BoltPrefillState* p) { delete p; }
void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::bolt::BoltEngine::prefill: " + what);
}
// The __kernel entry points of one prefill / head binary (kernels::bolt's names), for the pre-load.
std::vector<const char*> prefill_entries(const std::string& v) {
  auto starts = [&](const char* p) { return v.rfind(p, 0) == 0; };
  if (starts("nb_pf_embed_gather")) return {"nb_pf_embed_gather"};
  if (starts("nb_pf_mamba")) return {"nb_pf_seed", "nb_pf_conv", "nb_pf_dt", "nb_pf_ssd", "nb_pf_gnorm"};
  if (starts("nb_pf_cb")) return {"nb_pf_cb"};
  if (starts("nb_pf_moe_gemm")) return {"nb_pf_moe_gemm"};
  if (starts("nb_pf_moe")) return {"nb_pf_sort", "nb_pf_gather", "nb_pf_dequant_up", "nb_pf_dequant_dn", "nb_pf_combine"};
  if (starts("nb_res_norm")) return {"nb_res_norm"};
  if (starts("nb_router")) return {"nb_router"};
  if (starts("nb_route")) return {"nb_route"};
  if (starts("nb_sh_act")) return {"nb_sh_act"};
  if (starts("nb_attn_prep")) return {"nb_attn_prep"};
  if (starts("mg_pf_flash_attn")) return {"mg_pf_flash_attn"};
  if (starts("k2_pf_dequant_slab")) return {"k2_pf_dequant_slab"};
  if (starts("pf_dequant_slab")) return {"pf_dequant_slab"};
  if (starts("pf_gemm")) return {"pf_gemm"};
  if (starts("gemv_i8w")) return {"gemv_i8w"};
  if (starts("gemv_bf16")) return {"gemv_bf16"};
  if (starts("nb_argmax_stage1")) return {"argmax_stage1"};
  if (starts("nb_argmax_stage2")) return {"argmax_stage2"};
  throw std::runtime_error("runtime::bolt::prepare_prefill: no entry list for " + v);
}
}  // namespace

void BoltEngine::prepare_prefill() {
  if (pf_) return;
  const model::BoltDesc& d = model_.desc;
  const bool eager = attn_ == BoltAttn::Eager, b16 = state_ == BoltState::Bf16;
  std::vector<std::string> need = kernels::bolt::prefill_variants(d, eager, b16);
  for (const std::string& v : kernels::bolt::prefill_head_variants(d, model_.int8_head)) need.push_back(v);
  // A missing binary is named here, before anything is allocated or appended (the decode capture's rule).
  for (const std::string& v : need)
    require(std::ifstream(kernels::path(v)).good(),
            v + " is not compiled (" + kernels::path(v) + "): build with B70_BOLT=ON (spec 25d)");
  auto* ps = new BoltPrefillState();
  pf_ = std::unique_ptr<BoltPrefillState, void (*)(BoltPrefillState*)>(ps, &destroy_prefill);
  ps->timeout_ns = uint64_t(opt_.prefill_timeout_ms) * 1000000ull;
  for (uint32_t i = 0; i < devices(); ++i) ps->dev.push_back(std::make_unique<BoltPrefillState::Dev>(stage_ctx(i), d));
  if (devices() == 2)
    ps->link = std::make_unique<PipelineLink>(stage_ctx(0), stage_ctx(1), pf_landing_layout(d, model_.place),
                                              PpHandoff::Copy);
  // The route rows are read back whole (read_prefill_routes): zero them once, so rows no chunk wrote (a Mamba / attention
  // block's, rows past the last chunk's C) read the same on every engine.
  for (uint32_t i = 0; i < devices(); ++i) stage_imm(i).fill(ps->dev[i]->s.routes.ptr(), 0u, ps->dev[i]->s.routes.size());
  // Every module the walk and the head bind, created here - outside a timed prefill (KernelCache creates lazily).
  for (uint32_t i = 0; i < devices(); ++i)
    for (const std::string& v : need)
      for (const char* e : prefill_entries(v)) (void)ps->dev[i]->kc(v, e);
  pf_dev_bytes_.assign(devices(), 0);
  pf_bytes_ = 0;
  for (uint32_t i = 0; i < devices(); ++i) {
    pf_dev_bytes_[i] = ps->dev[i]->s.bytes() + (ps->link ? ps->link->bytes(i) : 0);
    pf_bytes_ += pf_dev_bytes_[i];
    require(ps->dev[i]->s.bytes() == prefill_sizes(d).total(), "the prefill scratch is not prefill_sizes()'s");
    require(!ps->link || ps->link->bytes(i) == pf_link_bytes(d, model_.place, i), "the prefill link is not pf_link_bytes()'s");
  }
  pf_settle_ = [ps] { return ps->settle(); };
}

size_t BoltEngine::prefill_launches(uint32_t dev) const {
  return pf_ && dev < pf_->dev.size() ? pf_->dev[dev]->cx.launches() : 0;
}
size_t BoltEngine::prefill_launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < devices(); ++i) n += prefill_launches(i);
  return n;
}

std::vector<uint32_t> BoltEngine::read_prefill_routes() {
  require(bool(pf_), "read_prefill_routes before the first prefill");
  const model::BoltDesc& d = model_.desc;
  std::vector<uint32_t> v(prefill_sizes(d).routes / 4, 0);
  for (uint32_t i = 0; i < devices(); ++i) {
    BoltPrefillState::Dev& D = *pf_->dev[i];
    require(D.cx.wait_for(pf_->timeout_ns), "device " + std::to_string(i) + "'s prefill list is still running");
    const uint32_t first = model_.place.first(i), n = model_.place.count(i);
    stage_imm(i).copy(v.data() + pf_route_at(first) / 4, static_cast<const uint8_t*>(D.s.routes.ptr()) + pf_route_at(first),
                      size_t(n) * kPfC * kRouteWords * 4);
  }
  return v;
}

void BoltEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  const model::BoltDesc& d = model_.desc;
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  require(!ids.empty(), "no ids");
  if (chunk == 0) chunk = kPfC;
  check_prefill_chunk(chunk);
  for (uint32_t id : ids)
    require(id < d.vocab, "id " + std::to_string(id) + " is outside the vocabulary (" + std::to_string(d.vocab) +
                              " rows) - nb_pf_embed_gather has no debug_flag channel, so the host is the only bound");
  const uint32_t ndev = devices();
  const uint32_t base = stage_control(ndev - 1)->pos;
  require(size_t(base) + ids.size() <= max_len_, "pos " + std::to_string(base) + " + " + std::to_string(ids.size()) +
                                                     " ids exceeds max_len " + std::to_string(max_len_) + " - the KV stops there");
  for (uint32_t i = 0; i < ndev; ++i)   // no decode step in flight writes the KV, the rings, the states or the Control
    require(settle(i), "device " + std::to_string(i) + "'s last decode step is still running");

  prepare_prefill();
  BoltPrefillState& P = *pf_;
  const bool eager = attn_ == BoltAttn::Eager;
  const char* env = std::getenv("B70_PREFILL_REPLAY");
  const bool replay = pf_replay_ >= 0 ? pf_replay_ == 1 : (env && std::strcmp(env, "1") == 0);
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
  // Where each device's walk takes and leaves h (its PrefillResid): a function of the placement alone (the walk asserts it).
  const PrefillResid in1 = prefill_resid_in(d, model_.place, 1);
  const PrefillResid last = prefill_resid_out(d, model_.place, ndev - 1);
  // One device's walk for a chunk: immediately, or through its recording (B70_PREFILL_REPLAY). The tap (debug) is written
  // on the last chunk only and never under a recording (a copy is not recordable).
  const auto run_dev = [&](uint32_t i, uint32_t pos, uint32_t C, bool last_chunk) {
    BoltPrefillState::Dev& D = *P.dev[i];
    l0::Mem* tap = !replay && last_chunk ? taps_[i].get() : nullptr;
    auto encode = [&] {
      prefill_chunk(D.cx, D.kc, D.s, model_.parts[i], d, stage_buffers(i), pos, C, eager,
                    prefill_resid_in(d, model_.place, i), tap);
    };
    if (!replay) return encode();
    auto it = std::find_if(D.chunks.begin(), D.chunks.end(),
                           [&](const BoltPrefillState::Chunk& c) { return c.pos == pos && c.rows == C; });
    if (it == D.chunks.end()) {
      if (D.chunks.size() == 8) D.chunks.erase(D.chunks.begin());   // a FIFO bound on retained lists
      D.chunks.push_back({pos, C, D.cx.capture(encode)});
      it = D.chunks.end() - 1;
    }
    D.cx.replay(*it->recording);
  };

  const bool hooked = static_cast<bool>(block_hook_);
  uint32_t C = 0;
  for (size_t off = 0; off < ids.size(); off += C) {
    const uint32_t pos = base + uint32_t(off);
    C = prefill_chunk_rows(pos, ids.size() - off, chunk, hooked, kBlock);
    const bool last_chunk = off + C == ids.size();
    wait_all("before a chunk");   // before touching the ids buffer or a Control block
    std::memcpy(P.dev[0]->s.ids.ptr(), ids.data() + off, size_t(C) * 4);
    set_ctl(pos, C);
    if (ndev == 1) {
      run_dev(0, pos, C, last_chunk);
      wait_all("one card");
    } else {
      PipelineLink& L = *P.link;
      BoltPrefillState::Dev& D0 = *P.dev[0];
      BoltPrefillState::Dev& D1 = *P.dev[1];
      uint8_t* land = L.landing.as<uint8_t>();
      const size_t rows = size_t(C) * d.hidden * 2, part = size_t(C) * d.hidden * 4;
      const size_t pend = size_t(in1.pending) * part;
      require(rows <= L.layout.resid_bytes && pend <= L.layout.sumsq_bytes,
              "the prefill landing buffer does not hold the chunk's rows and its pending GEMM rows");
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
        if (in1.pending >= 1) D0.cx.copy(land + L.layout.sumsq_off, D0.s.pa.ptr(), part);
        if (in1.pending == 2) D0.cx.copy(land + L.layout.sumsq_off + part, D0.s.pb.ptr(), part);
        D0.cx.signal(L.event->handle());
      }
      D1.cx.wait_event(L.event->handle());
      D1.cx.copy(D1.s.h0.ptr(), land, rows);
      if (in1.pending >= 1) D1.cx.copy(D1.s.pa.ptr(), land + L.layout.sumsq_off, part);
      if (in1.pending == 2) D1.cx.copy(D1.s.pb.ptr(), land + L.layout.sumsq_off + part, part);
      run_dev(1, pos, C, last_chunk);
      if (!D1.cx.wait_for(tmo)) {
        // Device 1 still waits (on the event device 0 never signalled, or on a hung walk): release it so its list drains
        // on garbage instead of holding the device, then report.
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
    if (hooked && !last_chunk && end % kBlock == 0) {
      // A mid-prompt block end: the KV holds exactly [0, end), the rings the raw columns of end - 3 .. end - 1 and the
      // states the state after end - 1; say so in every Control before the hook runs, so a hook that throws leaves pos
      // matching the chunks written.
      set_ctl(end, 0);
      block_hook_(end, true);
    }
  }

  // The tail on the last device: pos = base + L - 1 and n_active = 1 make argmax_stage2 leave pos = base + L and the first
  // generated id in cur_token[0]; `C - 1` is the last chunk's last row.
  set_ctl(base + uint32_t(ids.size()) - 1, 1);
  {
    BoltPrefillState::Dev& D = *P.dev[ndev - 1];
    prefill_head(D.cx, D.kc, D.s, model_.parts[ndev - 1], d, stage_buffers(ndev - 1), C - 1, last, model_.int8_head,
                 replay ? nullptr : taps_[ndev - 1].get());
    if (!D.cx.wait_for(tmo))
      fail("device " + std::to_string(ndev - 1) + "'s prefill head did not finish within " +
           std::to_string(opt_.prefill_timeout_ms) + " ms");
  }
  // The token's way back (16b's mirror): device 0 reads pos and cur_token from its own block next step.
  if (ndev == 2) std::memcpy(stage_control(0), stage_control(1), sizeof(Control));
  if (hooked) {   // the prompt end, the first generated id in cur_token (25e)
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
}

}  // namespace runtime::bolt
