// Spec 16c: `PipelineEngine::prefill` - pipeline-parallel PREFILL across two B70s - and its
// accessors, compiled into b70_prefill_host for Engine::prefill's reason
// (engine_prefill.cc): a decode-only binary links b70_runtime and must not acquire the
// prefill library. PipelineEngine reaches this half through PipelinePrefillBase's virtual
// calls only.
//
// The order is runtime/pipeline_prefill_plan.h's, run by its one executor
// (pp_prefill_run) over PrefillDriver below - the same executor tests/runtime/
// pp_prefill_protocol_test.cc runs over two host threads. Per chunk j (resources by j % 2):
//
//   device 0  [Run0]  ids and Control j % 2, timestamp, step_stage(layers [0, s) + layer
//             s's fold), timestamp, [hooked: GDN state + conv ring -> shadow j % 2], the
//             hand-off out into landing slot j % 2 (copy: two device-to-device copies; peer:
//             pp_send), signal ready j % 2, signal done0 j % 2.
//   device 1  [Run1]  Control j % 2, [test: wait hold j % 2], wait ready j % 2, the
//             hand-off in (copy: two copies from the slot; peer: pp_recv, which finds the flag
//             already published - the event ordered it - and checks the stamp), timestamp,
//             step_stage(layer s from its norm-finish, layers (s, L)), timestamp, [hooked:
//             shadow j % 2], signal done1 j % 2.
//   host      [WaitN] done N j % 2 with a bound; device 1 also reports pp_recv's status.
//             [Hook] spec 7's hook from the shadows. [Head] step_head on device 1's list.
//             [Drain] both lists idle, bounded.
//
// **MTP (spec 16d).** With the head on device 1, device 1's chunk also fills the head's KV
// after its layers, as Engine::prefill does after step_chunk: the chunk's ids into its own
// scratch (step_mtp_kv's embed_gather reads them there), h_{pos-1} (MtpBuffers::hh row 0)
// into the hidden rows' row 0, step_mtp_kv over a per-chunk head Control (hctl j % 2: the
// host writes chunk j + 1's while chunk j may run), then the chunk's last hidden row back
// into hh row 0 - all in order on device 1's list, so chunk j + 1 reads chunk j's. With a
// hook, that row is shadowed too (save_state's h_{end-1} at a mid-prompt block end). The head
// reads the last row step_mtp_kv normalised, as on one card.
//
// **Why peer waits on the event too.** In 16b's decode the flag alone orders device 1 after
// device 0: a step is ~18 ms, well inside pp_recv's bound of 2^24 flag loads. A prefill
// stage is ~0.5 s at 4k and seconds near 256k, and pp_recv's bound counts loads, not time
// (the per-load time is unmeasured - row 22 measures it). So under `peer` the rows still
// move by pp_send's stores into device 1's memory (spec 16 §2 option (b)), and pp_recv
// still checks the flag and the stamp with system-scope loads (the silent-hand-off check),
// but device 1's list reaches pp_recv only after the ready event: the spin is then a check,
// not a wait.
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/memory.h"
#include "l0/sync_event.h"
#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_prefill_plan.h"
#include "runtime/pipeline_stage.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/int8.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill/moe.h"
#include "runtime/prefill/profile.h"
#include "runtime/prefill/step.h"

namespace runtime {
static_assert(PipelineEngine::kBlock == kPpPfBlock, "spec 7's block is one number");

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kDepth = kPpPfDepth;
uint64_t ns(uint32_t ms) { return uint64_t(ms) * 1000000ull; }
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
// pp_handoff.cl's state words (pp_recv), as pipeline_engine.cc reads them.
constexpr uint32_t kStStatus = 1, kStFlag = 2, kStStamp = 3, kStWant = 4;
constexpr uint32_t kStatusTimeout = 1, kStatusStale = 2;

// A failure the engine reports by name: a bound passed, or pp_recv saw a bad hand-off.
struct PrefillFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The back-pressure test's hook (PipelineOptions::prefill_hold_ms): signals events at their
// deadlines from a host thread. Its destructor signals what is left, so no list is left
// waiting on a hold.
class Releaser {
 public:
  Releaser() : t_([this] { loop(); }) {}
  ~Releaser() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
      cv_.notify_all();
    }
    t_.join();
  }
  void at(Clock::time_point when, l0::SyncEvent* e) {
    std::lock_guard<std::mutex> l(m_);
    q_.push_back({when, e});
    cv_.notify_all();
  }

 private:
  struct Item {
    Clock::time_point when;
    l0::SyncEvent* e;
  };
  void loop() {
    std::unique_lock<std::mutex> l(m_);
    for (;;) {
      if (q_.empty()) {
        if (stop_) return;
        cv_.wait(l);
        continue;
      }
      // Deadlines arrive in order (one per chunk, appended in order).
      const Item it = q_.front();
      if (stop_ || Clock::now() >= it.when) {
        q_.pop_front();
        try {
          it.e->host_signal();
        } catch (...) {
        }
        continue;
      }
      cv_.wait_until(l, it.when);
    }
  }
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Item> q_;
  bool stop_ = false;
  std::thread t_;
};

// One device's prefill half.
struct PfDevice {
  PfDevice(l0::Context& c, const model::ModelDesc& d, uint32_t max_len)
      : ctx(c),
        cx(c),
        kc(c),
        scratch(c, max_len, d),
        ctl(c, l0::MemKind::Shared, kDepth * sizeof(Control)),
        ts(c, l0::MemKind::Shared, kDepth * 2 * sizeof(uint64_t)) {
    for (auto& e : done) e = std::make_unique<l0::SyncEvent>(c, std::vector<const l0::Context*>{&c});
    std::memset(ctl.ptr(), 0, ctl.size());
  }
  l0::Context& ctx;
  prefill::Context cx;
  prefill::KernelCache kc;
  PrefillScratch scratch;
  // Spec 5's h8 state, this device's linears' scales only. Declared after the context and
  // before nothing that names it.
  std::unique_ptr<prefill::Int8State> int8;
  l0::Mem ctl;   // Control [kDepth]: chunk j's pos / n_active in j % 2
  l0::Mem ts;    // uint64 [kDepth][2]: chunk j's walk start / end, device timestamps
  std::array<std::unique_ptr<l0::SyncEvent>, kDepth> done;
  // Spec 7's block-end shadows (only with a hook): this device's GDN state / conv ring.
  std::array<std::unique_ptr<l0::Mem>, kDepth> shadow_gdn, shadow_conv;
  Control* control(uint32_t j) { return ctl.as<Control>() + j % kDepth; }
  uint64_t* stamps(uint32_t j) { return ts.as<uint64_t>() + size_t(j % kDepth) * 2; }
  size_t shadow_bytes() const {
    size_t b = 0;
    for (uint32_t s = 0; s < kDepth; ++s)
      b += (shadow_gdn[s] ? shadow_gdn[s]->size() : 0) + (shadow_conv[s] ? shadow_conv[s]->size() : 0);
    return b;
  }
};

struct PipelinePrefill final : PipelinePrefillBase {
  PipelinePrefill(l0::Context& d0, l0::Context& d1, const model::ModelDesc& d, uint32_t max_len,
                  PpHandoff m)
      : mode(m),
        lay(pp_prefill_link_layout(d)),
        ids(d0, l0::MemKind::Host, size_t(kDepth) * PrefillScratch::kC * 4),
        landing(d1, l0::MemKind::Device, lay.total, kPpLandingAlign),
        send_seq(d0, l0::MemKind::Device, size_t(kDepth) * kPpStateWords * 4),
        recv_state(d1, l0::MemKind::Shared, size_t(kDepth) * kPpStateWords * 4),
        imm0(l0::CmdList::immediate(d0)),
        imm1(l0::CmdList::immediate(d1)) {
    dev[0] = std::make_unique<PfDevice>(d0, d, max_len);
    dev[1] = std::make_unique<PfDevice>(d1, d, max_len);
    for (auto& e : ready)
      e = std::make_unique<l0::SyncEvent>(d0, std::vector<const l0::Context*>{&d0, &d1});
    for (auto& e : hold) e = std::make_unique<l0::SyncEvent>(d1, std::vector<const l0::Context*>{&d1});
    zero();
  }
  ~PipelinePrefill() override {
    // Never destroy a list that waits on an event nobody will signal (prefill::Context's
    // destructor synchronises without a bound).
    try {
      settle(30000);
    } catch (...) {
    }
  }

  // Host-signals every event a device-1 list may be waiting on, so it drains (on whatever
  // the slot holds - the host has already thrown, and reset() must follow).
  void release() {
    for (auto& e : ready) e->host_signal();
    for (auto& e : hold) e->host_signal();
  }
  // Both lists idle within the bound, nothing released: everything appended completes.
  bool drain(uint32_t timeout_ms) {
    const bool ok = dev[0]->cx.wait_for(ns(timeout_ms)) && dev[1]->cx.wait_for(ns(timeout_ms));
    if (ok) in_flight = false;
    return ok;
  }
  // After a failure (or at teardown): release device 1 first if a prefill may have left it
  // waiting on a hand-off that will not come, then drain.
  bool settle(uint32_t timeout_ms) override {
    if (in_flight) release();
    return drain(timeout_ms);
  }
  void zero() override {
    imm1.fill(landing.ptr(), 0u, landing.size());
    imm0.fill(send_seq.ptr(), 0u, send_seq.size());
    imm1.fill(recv_state.ptr(), 0u, recv_state.size());
    for (auto& e : ready) e->host_reset();
    for (auto& e : hold) e->host_reset();
    for (auto& d : dev)
      for (auto& e : d->done) e->host_reset();
  }
  MemoryComponents memory(uint32_t d) const override {
    const PfDevice& p = *dev.at(d);
    MemoryComponents c;
    c.prefill_scratch = p.scratch.bytes() + p.scratch.lazy_bytes() + p.shadow_bytes() +
                        (d == kPpDevices - 1 ? shadow_hh_bytes() : 0);
    c.int8 = p.int8 ? p.int8->bytes() : 0;
    c.decode_state = p.ctl.size() + p.ts.size() +
                     (d == 0 ? send_seq.size() : landing.size() + recv_state.size()) + mtp_bytes(d);
    return c;
  }

  // Spec 7: each device's shadows, the size of its state (on the first hooked prefill).
  // Spec 16d: `hh_bytes` > 0 (MTP on) - and device 1's shadows of the head's hidden row.
  void ensure_shadows(const PersistentBuffers& p0, const PersistentBuffers& p1, size_t hh_bytes = 0) {
    const std::array<const PersistentBuffers*, kPpDevices> p{&p0, &p1};
    for (uint32_t d = 0; d < kPpDevices; ++d)
      for (uint32_t s = 0; s < kDepth; ++s) {
        if (dev[d]->shadow_gdn[s]) continue;
        dev[d]->shadow_gdn[s] =
            std::make_unique<l0::Mem>(dev[d]->ctx, l0::MemKind::Device, p[d]->gdn_state.size());
        dev[d]->shadow_conv[s] =
            std::make_unique<l0::Mem>(dev[d]->ctx, l0::MemKind::Device, p[d]->conv_ring.size());
      }
    if (hh_bytes != 0)
      for (auto& m : shadow_hh)
        if (!m) m = std::make_unique<l0::Mem>(dev[kPpDevices - 1]->ctx, l0::MemKind::Device, hh_bytes);
  }

  uint8_t* slot(uint32_t j) const { return landing.as<uint8_t>() + lay.slot(j); }
  uint32_t* seq(uint32_t j) const { return send_seq.as<uint32_t>() + size_t(j % kDepth) * kPpStateWords; }
  uint32_t* state(uint32_t j) const { return recv_state.as<uint32_t>() + size_t(j % kDepth) * kPpStateWords; }
  uint32_t* ids_of(uint32_t j) const { return ids.as<uint32_t>() + size_t(j % kDepth) * PrefillScratch::kC; }

  PpHandoff mode;
  PpPfLinkLayout lay;
  l0::Mem ids;          // Host [kDepth][kC] u32: device 0's embed input, by chunk parity
  l0::Mem landing;      // device 1, written only by device 0
  l0::Mem send_seq;     // device 0: pp_send's counter per slot
  l0::Mem recv_state;   // device 1, shared: pp_recv's state words per slot
  l0::CmdList imm0, imm1;
  std::array<std::unique_ptr<l0::SyncEvent>, kDepth> ready;   // device 0 -> 1, per slot
  std::array<std::unique_ptr<l0::SyncEvent>, kDepth> hold;    // the back-pressure test's hook
  bool in_flight = false;   // a prefill's lists may still hold work / waits
  // Spec 16d, MTP on: device 1's head prefill state - the hidden rows step_mtp_kv writes
  // ([kC + 1][hidden] bf16, row 0 = h_{pos-1}), two head Control blocks (chunk parity), and
  // with a hook two shadows of hh row 0. Allocated by ensure_mtp / ensure_shadows.
  std::unique_ptr<l0::Mem> mtp_hid, mtp_ctl;
  std::array<std::unique_ptr<l0::Mem>, kDepth> shadow_hh;
  void ensure_mtp(l0::Context& d1, const model::ModelDesc& d) {
    if (mtp_hid) return;
    mtp_hid = std::make_unique<l0::Mem>(d1, l0::MemKind::Device, mtp_prefill_hidden_bytes(d));
    mtp_ctl = std::make_unique<l0::Mem>(d1, l0::MemKind::Shared, kDepth * sizeof(Control));
    std::memset(mtp_ctl->ptr(), 0, mtp_ctl->size());
  }
  Control* head_control(uint32_t j) const { return mtp_ctl->as<Control>() + j % kDepth; }
  size_t mtp_bytes(uint32_t d) const {
    if (d != kPpDevices - 1 || !mtp_hid) return 0;
    return mtp_hid->size() + mtp_ctl->size();
  }
  size_t shadow_hh_bytes() const {
    size_t b = 0;
    for (const auto& m : shadow_hh) b += m ? m->size() : 0;
    return b;
  }
  // Declared last, destroyed first: the lists go before the events and buffers they name.
  std::array<std::unique_ptr<PfDevice>, kPpDevices> dev;
};

PipelinePrefill& as_pf(PipelinePrefillBase& b) { return static_cast<PipelinePrefill&>(b); }
}  // namespace

// --- the driver pp_prefill_run runs ------------------------------------------------------

struct PipelineEngine::PrefillDriver {
  PipelineEngine& e;
  PipelinePrefill& p;
  const std::vector<uint32_t>& ids;
  uint32_t base;
  PrefillBackend backend;
  int64_t drop_chunk;        // drop_next_handoff(): device 0 hands this chunk over to nobody
  std::unique_ptr<Releaser> releaser;
  bool hook_threw = false;

  prefill::Int8State* q(uint32_t d) const {
    return backend == PrefillBackend::L0Int8 ? p.dev[d]->int8.get() : nullptr;
  }

  void run(uint32_t d, const PpChunk& c, uint32_t j) {
    PfDevice& D = *p.dev[d];
    Stage& S = e.st(d);
    const uint32_t s = j % kDepth;
    Control* ctl = D.control(j);   // the walk's pos / n_active: chunk j - 2 is done with it
    ctl->pos = c.pos;
    ctl->n_active = c.rows;
    D.done[s]->host_reset();
    const PpPfLinkLayout& L = p.lay;
    const size_t rows_bytes = size_t(c.rows) * S.model.desc->hidden * 2;
    if (d == 0) {
      uint32_t* in = p.ids_of(j);
      std::memcpy(in, ids.data() + (c.pos - base), size_t(c.rows) * 4);
      p.ready[s]->host_reset();   // device 1's chunk j - 2 waited on it, and is done
      D.cx.timestamp(D.stamps(j));
      prefill::step_stage(D.cx, D.kc, D.scratch, S.model, e.max_len_, ctl, c.pos, c.rows, in,
                          S.persist.gdn_state, S.persist.conv_ring, S.persist.kv_k, S.persist.kv_v,
                          S.persist.kv_lay, backend, q(0), S.range);
      D.cx.timestamp(D.stamps(j) + 1);
      shadow(D, S, c, s);
      // drop_next_handoff() (P4's test): device 1 never hears of this chunk - no ready
      // signal; under copy the rows are not copied either. Under peer pp_send still runs, so
      // once the host releases device 1 its pp_recv finds the flag at once instead of spinning
      // out its bound (16b's pp_fail_test owns that bound).
      const bool dropped = int64_t(j) == drop_chunk;
      if (!(dropped && p.mode == PpHandoff::Copy)) {
        if (p.mode == PpHandoff::Copy) {
          D.cx.copy(p.slot(j), D.scratch.resid.ptr(), rows_bytes);
          D.cx.copy(p.slot(j) + L.sumsq_off, D.scratch.norm_sumsq.ptr(), L.sumsq_bytes);
        } else {
          // pp_send(resid, sumsq, land_resid, land_sumsq, flag, seq, resid_words, sumsq_words)
          const uint32_t rw = uint32_t(rows_bytes / 4), sw = uint32_t(L.sumsq_bytes / 4);
          D.cx.launch(D.kc(kernels::pp_handoff_variant(), "pp_send"), 1, 1, 1,
                      {prefill::PtrArg(D.scratch.resid.ptr()), prefill::PtrArg(D.scratch.norm_sumsq.ptr()),
                       prefill::PtrArg(p.slot(j)), prefill::PtrArg(p.slot(j) + L.sumsq_off),
                       prefill::PtrArg(p.slot(j) + L.flag_off), prefill::PtrArg(p.seq(j)),
                       prefill::arg_val(rw), prefill::arg_val(sw)});
        }
      }
      if (!dropped) D.cx.signal(p.ready[s]->handle());
      D.cx.signal(D.done[s]->handle());
      return;
    }
    if (releaser) {   // the test hook: device 1 held back before it takes the rows
      p.hold[s]->host_reset();
      D.cx.wait_event(p.hold[s]->handle());
      releaser->at(Clock::now() + std::chrono::milliseconds(e.opt_.prefill_hold_ms), p.hold[s].get());
    }
    D.cx.wait_event(p.ready[s]->handle());
    if (p.mode == PpHandoff::Copy) {
      D.cx.copy(D.scratch.resid.ptr(), p.slot(j), rows_bytes);
      D.cx.copy(D.scratch.norm_sumsq.ptr(), p.slot(j) + L.sumsq_off, L.sumsq_bytes);
    } else {
      // pp_recv(land_resid, land_sumsq, flag, state, resid, sumsq, resid_words, sumsq_words,
      // spin_limit): the flag is already published (the event), so this is the stamp check.
      const uint32_t rw = uint32_t(rows_bytes / 4), sw = uint32_t(L.sumsq_bytes / 4);
      const uint32_t spin = e.opt_.spin_limit;
      D.cx.launch(D.kc(kernels::pp_handoff_variant(), "pp_recv"), 1, 1, 1,
                  {prefill::PtrArg(p.slot(j)), prefill::PtrArg(p.slot(j) + L.sumsq_off),
                   prefill::PtrArg(p.slot(j) + L.flag_off), prefill::PtrArg(p.state(j)),
                   prefill::PtrArg(D.scratch.resid.ptr()), prefill::PtrArg(D.scratch.norm_sumsq.ptr()),
                   prefill::arg_val(rw), prefill::arg_val(sw), prefill::arg_val(spin)});
    }
    D.cx.timestamp(D.stamps(j));
    prefill::step_stage(D.cx, D.kc, D.scratch, S.model, e.max_len_, ctl, c.pos, c.rows, nullptr,
                        S.persist.gdn_state, S.persist.conv_ring, S.persist.kv_k, S.persist.kv_v,
                        S.persist.kv_lay, backend, q(1), S.range);
    if (S.mtp) mtp_kv(D, S, c, j);   // spec 16d: the head's KV, as Engine::prefill
    D.cx.timestamp(D.stamps(j) + 1);
    shadow(D, S, c, s);
    D.cx.signal(D.done[s]->handle());
  }

  // Spec 16d: Engine::prefill's MTP steps for chunk j, on device 1's list in order (the file
  // header has why each is where it is).
  void mtp_kv(PfDevice& D, Stage& S, const PpChunk& c, uint32_t j) {
    const size_t row = size_t(S.model.desc->hidden) * 2;
    uint8_t* hid = p.mtp_hid->as<uint8_t>();
    Control* hc = p.head_control(j);   // chunk j - 2 is done with it (back-pressure)
    hc->pos = c.pos == 0 ? 0 : c.pos - 1;
    hc->n_active = prefill::mtp_kv_rows(c.pos, c.rows);
    D.cx.copy(D.scratch.ids.ptr(), p.ids_of(j), size_t(c.rows) * 4);   // step_mtp_kv's embed input
    D.cx.copy(hid, S.mtp->hh.ptr(), row);                               // h_{pos-1} into row 0
    prefill::step_mtp_kv(D.cx, D.kc, D.scratch, S.model, hc, c.pos, c.rows,
                         reinterpret_cast<uint16_t*>(hid),
                         S.mtp->kv_lay.layer(S.mtp->kv_k.ptr(), S.mtp->kv_v.ptr(), 0));
    // h_{end-1}: the next chunk's row 0, a snapshot's hidden, the first draft's h.
    D.cx.copy(S.mtp->hh.ptr(), hid + size_t(c.rows) * row, row);
    if (c.hook) D.cx.copy(p.shadow_hh[j % kDepth]->ptr(), hid + size_t(c.rows) * row, row);
  }

  // Spec 7: after a hooked chunk, this device's GDN state and conv ring as they are at its
  // end - in order on its list, before anything of the next chunk.
  void shadow(PfDevice& D, Stage& S, const PpChunk& c, uint32_t s) {
    if (!c.hook) return;
    D.cx.copy(D.shadow_gdn[s]->ptr(), S.persist.gdn_state.ptr(), S.persist.gdn_state.size());
    D.cx.copy(D.shadow_conv[s]->ptr(), S.persist.conv_ring.ptr(), S.persist.conv_ring.size());
  }

  void wait(uint32_t d, uint32_t j) {
    PfDevice& D = *p.dev[d];
    const uint32_t s = j % kDepth;
    if (!D.done[s]->host_wait(ns(e.opt_.prefill_timeout_ms))) {
      std::string why = "device " + std::to_string(d) + "'s chunk " + std::to_string(j) +
                        " did not finish within " + std::to_string(e.opt_.prefill_timeout_ms) + " ms";
      if (d == 1 && !p.ready[s]->signalled())
        why += " - device 0's rows never arrived (the ready event was not signalled)";
      if (int64_t(j) == drop_chunk) why += " [device 0 did not hand it over: drop_next_handoff()]";
      throw PrefillFailure(why);
    }
    const uint64_t* t = D.stamps(j);
    const double busy = D.cx.timestamp_ms(t[0], t[1]);
    e.pf_stats_.busy_ms[d] += busy;
    e.pf_stats_.busy_max_ms[d] = std::max(e.pf_stats_.busy_max_ms[d], busy);
    if (d == 1 && p.mode == PpHandoff::Peer) {
      const uint32_t* w = p.state(j);
      if (w[kStStatus] == kStatusTimeout)
        throw PrefillFailure("chunk " + std::to_string(j) + ": pp_recv found slot " +
                             std::to_string(s) + "'s flag at " + std::to_string(w[kStFlag]) +
                             ", expected " + std::to_string(w[kStWant]) +
                             " - device 0's rows never arrived although the event said so");
      if (w[kStStatus] == kStatusStale)
        throw PrefillFailure("chunk " + std::to_string(j) + ": slot " + std::to_string(s) +
                             "'s flag reached " + std::to_string(w[kStWant]) +
                             " but the stamp after the norm sums reads " +
                             std::to_string(w[kStStamp]) +
                             " - the rows did not arrive with the flag (a silent hand-off, spec 16 §2)");
      if (w[kStStatus] != 0)
        throw PrefillFailure("pp_recv reported status " + std::to_string(w[kStStatus]));
    }
  }

  // Spec 7 at a mid-prompt block end: both devices have finished chunk j; chunk j + 1 may be
  // running. The hook sees pos at the block end and save_state() reads the shadows.
  void hook(const PpChunk& c, uint32_t j) {
    const uint32_t s = j % kDepth;
    for (uint32_t d = 0; d < kPpDevices; ++d) {
      e.st(d).ctl->pos = c.end();
      e.st(d).ctl->n_active = 0;
      e.snap_gdn_[d] = p.dev[d]->shadow_gdn[s].get();
      e.snap_conv_[d] = p.dev[d]->shadow_conv[s].get();
    }
    if (e.mtp()) e.snap_hh_ = p.shadow_hh[s].get();   // spec 16d
    try {
      e.block_hook_(c.end(), true);
    } catch (...) {
      e.snap_gdn_ = {};
      e.snap_conv_ = {};
      e.snap_hh_ = nullptr;
      // As on one card, the state is exactly at the block end: let what was appended after it
      // finish - on its own, nothing is released early (every ready event it waits on is
      // signalled by device 0's half, already appended) - then put the shadows back.
      if (!p.drain(e.opt_.prefill_timeout_ms))
        throw PrefillFailure("the block hook at " + std::to_string(c.end()) +
                             " threw, and the lists did not drain within " +
                             std::to_string(e.opt_.prefill_timeout_ms) + " ms");
      for (uint32_t d = 0; d < kPpDevices; ++d) {
        Stage& S = e.st(d);
        S.imm.copy(S.persist.gdn_state.ptr(), p.dev[d]->shadow_gdn[s]->ptr(), S.persist.gdn_state.size());
        S.imm.copy(S.persist.conv_ring.ptr(), p.dev[d]->shadow_conv[s]->ptr(), S.persist.conv_ring.size());
        S.ctl->pos = c.end();
        S.ctl->n_active = 0;
      }
      if (e.mtp()) {   // spec 16d: and h_{end-1}
        Stage& S1 = e.st(kPpDevices - 1);
        S1.imm.copy(S1.mtp->hh.ptr(), p.shadow_hh[s]->ptr(), size_t(S1.model.desc->hidden) * 2);
      }
      hook_threw = true;
      throw;
    }
    e.snap_gdn_ = {};
    e.snap_conv_ = {};
    e.snap_hh_ = nullptr;
  }

  // The head on device 1, after its last chunk: one card's step_head, into device 1's
  // (decode) Control - pos = the last position, n_active 1, so argmax_stage2 leaves pos at
  // the prompt's end and the first generated id in cur_token[0].
  void head(const PpChunk& c, uint32_t) {
    PfDevice& D = *p.dev[1];
    Stage& S = e.st(1);
    S.ctl->pos = c.end() - 1;
    S.ctl->n_active = 1;
    // Spec 16d: with the head, step_mtp_kv already normalised every row of the last chunk
    // into the hidden rows; the head reads row C (Engine::prefill's step_head call).
    const void* normed = S.mtp ? p.mtp_hid->as<uint8_t>() + size_t(c.rows) * S.model.desc->hidden * 2
                               : nullptr;
    prefill::step_head(D.cx, D.kc, D.scratch, S.model, S.ctl, c.rows - 1, normed);
  }

  void drain() {
    const uint64_t t = ns(e.opt_.prefill_timeout_ms);
    for (uint32_t d = 0; d < kPpDevices; ++d)
      if (!p.dev[d]->cx.wait_for(t))
        throw PrefillFailure("device " + std::to_string(d) + "'s prefill list did not finish within " +
                             std::to_string(e.opt_.prefill_timeout_ms) + " ms");
  }
};

// --- the public half ---------------------------------------------------------------------

PrefillBackend PipelineEngine::prefill_backend() const {
  return pf_backend_set_ ? pf_backend_ : prefill::default_prefill_backend();
}

size_t PipelineEngine::prefill_launches(uint32_t dev) const {
  return pf_ ? as_pf(*pf_).dev.at(dev)->cx.launches() : 0;
}

std::vector<float> PipelineEngine::read_prefill_logits() const {
  if (!pf_) throw std::runtime_error("runtime::PipelineEngine::read_prefill_logits: no prefill yet");
  std::vector<float> out(model::Qwen35::kVocab);
  st(1).imm.copy(out.data(), as_pf(*pf_).dev[1]->scratch.logits.ptr(), out.size() * sizeof(float));
  return out;
}

void PipelineEngine::prepare_prefill() {
  const model::ModelDesc& d = *st(0).model.desc;
  model::require_prefill(d);
  const PrefillBackend b = prefill_backend();
  if (!is_l0(b))
    throw std::runtime_error(std::string("--pp 2 prefills on the L0 backends (l0, l0-int8), not ") +
                             prefill_backend_name(b) +
                             ": sycl-tla's walk waits on the host between runtimes and has no "
                             "two-card pipeline (spec 16c); --prefill-backend l0 or l0-int8");
  if (prefill::attn_mode() != prefill::AttnMode::Flash)
    throw std::runtime_error("--pp 2 prefills with the flash attention only: "
                             "B70_PREFILL_ATTN=composed has no two-card walk (spec 16c)");
  const char* replay = std::getenv("B70_PREFILL_REPLAY");
  if (replay && std::strcmp(replay, "1") == 0)
    throw std::runtime_error("--pp 2 has no prefill replay (B70_PREFILL_REPLAY=1): the "
                             "recordings are one card's walk (spec 16c)");
  if (prefill::profile_enabled())
    throw std::runtime_error("--pp 2 prefill is not profiled (B70_PREFILL_PROFILE=1): the "
                             "phase waits would serialise the pipeline; each device's busy time "
                             "is reported instead (spec 16c)");
  if (kv_ == KvCache::Int8) prefill::require_kv8_path(b);
  // Spec 16d: the head's prefill runs on the L0 backends only (pf_gemm) - already required.
  if (!pf_) {
    // The two stages' walks add up to one card's, launch for launch (step.h).
    for (uint32_t C : {1u, 256u, PrefillScratch::kC})
      if (prefill::step_stage_launches(d, b, C, stage(0)) + prefill::step_stage_launches(d, b, C, stage(1)) !=
          prefill::step_chunk_launches(d, b, C))
        throw std::logic_error("runtime::PipelineEngine::prepare_prefill: the two stages' launches "
                               "at C = " + std::to_string(C) + " do not add up to one card's");
    pf_ = std::make_unique<PipelinePrefill>(st(0).ctx, st(1).ctx, d, max_len_, opt_.handoff);
  }
  if (mtp()) as_pf(*pf_).ensure_mtp(st(1).ctx, d);   // spec 16d
  if (b != PrefillBackend::L0Int8) return;
  PipelinePrefill& p = as_pf(*pf_);
  // Engine::prepare_prefill's l0-int8 pass, per device over ITS linears: every int4
  // linear's rotated column scales (not lm_head, not a||b), and a MoE layer's expert arrays.
  for (uint32_t i = 0; i < kPpDevices; ++i) {
    PfDevice& D = *p.dev[i];
    const loader::LoadedModel& m = st(i).model;
    if (!D.int8) D.int8 = std::make_unique<prefill::Int8State>(D.ctx, prefill_int8_max_k(d));
    for (const auto& [key, w] : m.linears)
      if (key.second != model::LinearId::LmHead && key.second != model::LinearId::AB &&
          w.kind == model::WeightKind::Int4)
        D.int8->scales(D.cx, D.kc, w);
    prefill::moe_prepare_int8(D.cx, D.kc, *D.int8, m);
  }
}

void PipelineEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  if (broken_)
    throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  if (ids.empty()) throw std::runtime_error("runtime::PipelineEngine::prefill: no ids");
  if (chunk == 0) chunk = PrefillScratch::kC;
  if (chunk > PrefillScratch::kC)
    throw std::runtime_error("runtime::PipelineEngine::prefill: chunk " + std::to_string(chunk) +
                             " exceeds PrefillScratch::kC " + std::to_string(PrefillScratch::kC));
  for (uint32_t id : ids)
    if (id >= model::Qwen35::kVocab)
      throw std::runtime_error("runtime::PipelineEngine::prefill: id " + std::to_string(id) +
                               " is outside the vocabulary (" + std::to_string(model::Qwen35::kVocab) +
                               " rows)");
  const uint32_t base = st(1).ctl->pos;
  if (size_t(base) + ids.size() > size_t(max_len_))
    throw std::runtime_error("runtime::PipelineEngine::prefill: pos " + std::to_string(base) + " + " +
                             std::to_string(ids.size()) + " ids exceeds max_len " +
                             std::to_string(max_len_) +
                             " - the KV cache and the RoPE table stop there. Start a new session "
                             "with reset(), or load with a larger max_len.");
  prepare_prefill();
  PipelinePrefill& p = as_pf(*pf_);
  const bool hooked = static_cast<bool>(block_hook_);
  if (hooked) p.ensure_shadows(st(0).persist, st(1).persist, mtp() ? size_t(st(1).model.desc->hidden) * 2 : 0);
  mtp_before_prefill();   // spec 16d: the live verify slot into slot 0 on both devices
  const std::vector<PpChunk> chunks = pp_prefill_chunks(base, ids.size(), chunk, hooked);
  const std::vector<PpPfStep> steps = pp_prefill_schedule(chunks);
  const std::string bad = pp_prefill_check(steps, chunks);
  if (!bad.empty()) throw std::logic_error("runtime::PipelineEngine::prefill: the order breaks a rule: " + bad);

  const PrefillBackend backend = prefill_backend();
  PrefillDriver drv{*this, p, ids, base, backend, drop_next_ ? 0 : -1, nullptr, false};
  drop_next_ = false;
  if (opt_.prefill_hold_ms != 0) drv.releaser = std::make_unique<Releaser>();
  pf_stats_ = PrefillStats{};
  pf_stats_.chunks = uint32_t(chunks.size());
  const std::array<size_t, kPpDevices> launches0{p.dev[0]->cx.launches(), p.dev[1]->cx.launches()};
  const size_t peer = opt_.handoff == PpHandoff::Peer ? 1 : 0;   // pp_send / pp_recv per chunk
  for (const PpChunk& c : chunks)
    for (uint32_t i = 0; i < kPpDevices; ++i)
      pf_stats_.expected_launches[i] += prefill::step_stage_launches(*st(i).model.desc, backend, c.rows, stage(i)) + peer;
  pf_stats_.expected_launches[1] += prefill::kStepHeadLaunches;
  if (mtp()) {   // spec 16d: step_mtp_kv per chunk; the head's two norm launches are skipped
    for (const PpChunk& c : chunks) pf_stats_.expected_launches[1] += prefill::step_mtp_kv_launches(c.pos, c.rows);
    pf_stats_.expected_launches[1] -= 2;
  }

  const Clock::time_point t0 = Clock::now();
  p.in_flight = true;
  try {
    pp_prefill_run(chunks, steps, drv);
  } catch (...) {
    drv.releaser.reset();   // signals any hold still pending
    if (drv.hook_threw) throw;   // the hook's own exception; the state is at its block end
    // A bound passed or a hand-off was bad: release device 1, let both lists drain (bounded),
    // and mark the session as 16b's failed step does.
    snap_gdn_ = {};
    snap_conv_ = {};
    const bool drained = p.settle(opt_.prefill_timeout_ms);
    broken_ = true;
    std::string what = "?";
    try {
      throw;
    } catch (const std::exception& x) {
      what = x.what();
    } catch (...) {
    }
    throw std::runtime_error("pipeline prefill: " + what +
                             (drained ? "" : " (and the lists did not drain after the release)") +
                             ". The session's state is not valid; reset() before the next step");
  }
  p.in_flight = false;
  drv.releaser.reset();
  pf_stats_.wall_ms = ms_since(t0);
  for (uint32_t i = 0; i < kPpDevices; ++i) pf_stats_.launches[i] = p.dev[i]->cx.launches() - launches0[i];
  // The token's way back, as after a decode step: device 1's argmax advanced pos and wrote
  // the first generated id; device 0's block mirrors it.
  std::memcpy(st(0).ctl, st(1).ctl, sizeof(Control));
  if (hooked) {
    const uint32_t end = base + uint32_t(ids.size());
    block_hook_(end, end % kBlock == 0);
  }
}

}  // namespace runtime
