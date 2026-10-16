#include "runtime/muse/muse_engine.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "l0/cmdlist.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/muse_layout.h"
#include "runtime/muse/muse_capture.h"

namespace runtime::muse {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// pp_handoff.cl's state words (pp_recv): [1] status, [2] the flag read at the failure, [3] the stamp read, [4] the
// sequence expected (runtime/pipeline_engine.cc's).
constexpr uint32_t kStStatus = 1, kStFlag = 2, kStStamp = 3, kStWant = 4;
constexpr uint32_t kStatusTimeout = 1, kStatusStale = 2;
}  // namespace

struct MuseEngine::Stage {
  Stage(l0::Context& c, const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t dev, uint32_t max_len,
        MuseAttn a, uint32_t rows)
      : ctx(c), buffers(c, d, p, dev, max_len, a, rows), queue(c), fence(queue), imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()) {}
  l0::Context& ctx;
  MuseBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  std::optional<CapturedStep> step;
};

l0::Context& MuseEngine::stage_ctx(uint32_t dev) const { return st(dev).ctx; }
MuseBuffers& MuseEngine::stage_buffers(uint32_t dev) const { return st(dev).buffers; }
Control* MuseEngine::stage_control(uint32_t dev) const { return st(dev).ctl; }
l0::CmdList& MuseEngine::stage_imm(uint32_t dev) const { return st(dev).imm; }

MuseEngine::Stage& MuseEngine::st(uint32_t dev) const {
  if (dev >= st_.size())
    throw std::out_of_range("runtime::muse::MuseEngine: device " + std::to_string(dev) + " of " + std::to_string(st_.size()));
  return *st_[dev];
}

MuseEngine::MuseEngine(std::vector<l0::Context*> ctx, const std::string& snapshot, const MuseOptions& o)
    : opt_(o), attn_(muse_attn()) {
  const uint32_t devs = uint32_t(ctx.size());
  if (devs != 1 && devs != 2)
    throw std::invalid_argument("runtime::muse::MuseEngine: " + std::to_string(devs) + " contexts (one card, or --pp 2)");
  if (devs == 2) {
    if (ctx[1]->handle() != ctx[0]->handle())
      throw std::invalid_argument("runtime::muse::MuseEngine: device 1 is not a view of device 0's Level Zero context "
                                  "(spec 16 decision 1)");
    if (!ctx[0]->can_access_peer(*ctx[1]))
      throw std::runtime_error("--pp 2: device 0 (" + ctx[0]->name() + ") cannot access device 1's memory "
                               "(zeDeviceCanAccessPeer is false), and both hand-offs write it. Peer access needs the "
                               "P2P-capable kernel and both cards under one root complex (docs/10-the-box.md)");
    if (o.timeout_ms == 0 || (o.handoff == PpHandoff::Peer && o.spin_limit == 0))
      throw std::invalid_argument("runtime::muse::MuseEngine: a hand-off needs a non-zero bound");
  }
  // The plan before the first allocation: the descriptor from config.json alone, the decode scratch reserved
  // (at 131072 while the length is still auto, then at the length chosen).
  const model::MuseGlimmerDesc d = loader::muse_checkpoint_desc(snapshot, o.layers);
  std::vector<size_t> caps;
  for (l0::Context* c : ctx) caps.push_back(c->memory_bytes());
  // Spec 23d: a run that prefills (MuseOptions::prefill) plans the prefill scratch and its link too.
  // Spec 23f: the verify lists' scratch rows and the drafter's term (one card only).
  if (o.spec.rows == 0 || o.spec.rows > kVerifyMaxRows)
    throw std::invalid_argument("runtime::muse::MuseEngine: spec.rows " + std::to_string(o.spec.rows) +
                                " outside 1..16 (the anchor + up to 15 drafts)");
  if (o.spec.on() && devs != 1)
    throw std::invalid_argument("runtime::muse::MuseEngine: speculative decoding (--spec dflash / lookup) runs on one "
                                "card (spec 23f); --pp 2 serves Muse-Glimmer plainly");
  const MusePlan first = plan_muse(d, devs, o.max_len, o.int8_head,
                                   run_reserve(d, o.max_len, devs, attn_, o.prefill, o.spec), caps,
                                   devs == 2 ? o.split : 0);
  require_fits(first);
  max_len_ = first.max_len;
  const MuseReserve reserve = run_reserve(d, max_len_, devs, attn_, o.prefill, o.spec);
  plan_ = plan_at(d, first.placement, max_len_, o.int8_head, reserve, caps);
  require_fits(plan_);
  w_ = loader::load_muse_glimmer(ctx, snapshot, max_len_, plan_.placement, o.int8_head, o.layers, reserve);

  const model::MuseGlimmerDesc& D = w_.desc;
  const model::MusePlacement& P = w_.placement;
  model::validate(P, D);
  if (w_.parts.size() != P.devices)
    throw std::logic_error("runtime::muse::MuseEngine: " + std::to_string(w_.parts.size()) + " loaded parts for a " +
                           std::to_string(P.devices) + "-device placement");
  for (uint32_t i = 0; i < P.devices; ++i) {
    st_.push_back(std::make_unique<Stage>(*ctx[i], D, P, i, max_len_, attn_, o.spec.rows));
    taps_.push_back(o.debug_tap ? std::make_unique<l0::Mem>(*ctx[i], l0::MemKind::Device, tap_bytes(D)) : nullptr);
  }
  pending_.assign(P.devices, false);
  StageLink binding;
  if (P.devices == 2) {
    link_ = std::make_unique<PipelineLink>(*ctx[0], *ctx[1], landing_layout(D), o.handoff);
    binding = link_->binding(o.spin_limit);
  }
  for (uint32_t i = 0; i < P.devices; ++i)
    st(i).step.emplace(build(st(i).ctx, w_.parts[i], D, st(i).buffers, link_ ? &binding : nullptr, taps_[i].get()));
  if (o.spec.on()) {   // spec 23f: the verify's ids / output (the lists themselves on first use)
    vin_ = std::make_unique<l0::Mem>(*ctx[0], l0::MemKind::Shared, kVerifyMaxRows * 4);
    vout_ = std::make_unique<l0::Mem>(*ctx[0], l0::MemKind::Shared, kVerifyMaxRows * 4);
    std::memset(vin_->ptr(), 0, vin_->size());
    std::memset(vout_->ptr(), 0, vout_->size());
    verify_steps_.resize(o.spec.rows);
  }
  reset();
}

MuseEngine::~MuseEngine() {
  for (uint32_t i = 0; i < st_.size(); ++i) {   // never free buffers under a running list
    try {
      settle(i);
    } catch (...) {
    }
  }
}

bool MuseEngine::settle(uint32_t dev) {
  if (!pending_[dev]) return true;
  if (st(dev).fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull)) pending_[dev] = false;
  return !pending_[dev];
}

void MuseEngine::reset() {
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error("runtime::muse::MuseEngine::reset: device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
  if (pf_settle_ && !pf_settle_())   // spec 23d: no prefill list may still run into the buffers
    throw std::runtime_error("runtime::muse::MuseEngine::reset: a prefill list is still running after " +
                             std::to_string(opt_.prefill_timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i) st(i).buffers.zero(st(i).imm);
  if (link_) link_->zero(st(0).imm, st(1).imm);
  if (drafter_) drafter_->reset(st(0).imm);   // spec 23f: a cold drafter
  verify_k_ = kNoVerify;
  broken_ = false;
  drop_next_ = false;
}

// --- spec 23f: speculative decoding ---------------------------------------------------------------------------
void MuseEngine::attach_drafter(TapSink& d) {
  if (!opt_.spec.on())
    throw std::logic_error("runtime::muse::MuseEngine::attach_drafter: the engine was built without verify lists "
                           "(MuseOptions::spec.rows is 1)");
  if (drafter_) throw std::logic_error("runtime::muse::MuseEngine::attach_drafter: a drafter is already attached");
  if (d.max_k() + 1 > opt_.spec.rows)
    throw std::invalid_argument("runtime::muse::MuseEngine::attach_drafter: the drafter drafts up to " +
                                std::to_string(d.max_k()) + ", the verify lists hold " +
                                std::to_string(opt_.spec.rows) + " rows");
  for (uint32_t t : d.tap_layers())
    if (t >= w_.desc.layers)
      throw std::invalid_argument("runtime::muse::MuseEngine::attach_drafter: the drafter taps layer " +
                                  std::to_string(t) + " of a " + std::to_string(w_.desc.layers) + "-layer target");
  for (auto& s : verify_steps_) s.reset();   // lookup's lists bound no taps: recaptured with them
  drafter_ = &d;
  drafter_->reset(st(0).imm);
}

uint32_t MuseEngine::max_verify_k() const {
  if (!opt_.spec.on()) return 0;
  const uint32_t k = drafter_ ? std::min(drafter_->max_k(), opt_.spec.rows - 1) : opt_.spec.rows - 1;
  const uint32_t p = pos();
  if (size_t(p) + 1 >= size_t(max_len_)) return 0;
  return std::min<uint32_t>(k, max_len_ - p - 1);
}

void MuseEngine::run_verify(uint32_t M) {
  std::optional<CapturedStep>& s = verify_steps_.at(M - 1);
  if (!s) {
    VerifyBind v;
    v.ids = vin_.get();
    v.out = vout_.get();
    if (drafter_) {
      v.taps = &drafter_->taps();
      v.tap_layers = &drafter_->tap_layers();
    }
    s.emplace(build_verify(st(0).ctx, w_.parts[0], w_.desc, st(0).buffers, M, v));
  }
  st(0).queue.execute(s->list, &st(0).fence);
  st(0).fence.wait();
}

void MuseEngine::verify(const uint32_t* drafts, uint32_t k) {
  if (!opt_.spec.on()) throw std::logic_error("runtime::muse::MuseEngine::verify: no verify lists (spec.rows 1)");
  if (broken_) throw std::runtime_error("runtime::muse::MuseEngine::verify: reset() first");
  if (k > max_verify_k())
    throw std::invalid_argument("runtime::muse::MuseEngine::verify: k " + std::to_string(k) + " exceeds " +
                                std::to_string(max_verify_k()) + " at pos " + std::to_string(pos()));
  Control& c = *st(0).ctl;
  if (size_t(c.pos) + k + 1 > size_t(max_len_))
    throw std::runtime_error("runtime::muse::MuseEngine::verify: pos " + std::to_string(c.pos) + " + " +
                             std::to_string(k + 1) + " rows exceeds max_len " + std::to_string(max_len_));
  uint32_t* in = vin_->as<uint32_t>();
  in[0] = c.cur_token[0];
  for (uint32_t i = 0; i < k; ++i) {
    if (drafts[i] >= w_.desc.vocab)
      throw std::invalid_argument("runtime::muse::MuseEngine::verify: draft " + std::to_string(drafts[i]) +
                                  " is outside the vocabulary");
    in[1 + i] = drafts[i];
  }
  c.n_active = k + 1;
  run_verify(k + 1);
  verify_k_ = k;
  verify_pos_ = c.pos;
}

const uint32_t* MuseEngine::verify_ids() const {
  if (!vout_) throw std::logic_error("runtime::muse::MuseEngine::verify_ids: no verify lists");
  return vout_->as<uint32_t>();
}

void MuseEngine::read_verify_logits(float* host, uint32_t rows) {
  if (verify_k_ == kNoVerify || rows > verify_k_ + 1)
    throw std::logic_error("runtime::muse::MuseEngine::read_verify_logits: " + std::to_string(rows) +
                           " rows after no verify / a verify of fewer");
  st(0).imm.copy(host, st(0).buffers.logits.ptr(), size_t(rows) * w_.desc.vocab * 4);
}

void MuseEngine::commit(uint32_t j, uint32_t next_token) {
  Control& c = *st(0).ctl;
  if (verify_k_ == kNoVerify || c.pos != verify_pos_)
    throw std::logic_error("runtime::muse::MuseEngine::commit: no verify at pos " + std::to_string(c.pos));
  if (j > verify_k_)
    throw std::invalid_argument("runtime::muse::MuseEngine::commit: j " + std::to_string(j) + " > k " +
                                std::to_string(verify_k_));
  if (next_token >= w_.desc.vocab)
    throw std::invalid_argument("runtime::muse::MuseEngine::commit: id " + std::to_string(next_token) +
                                " is outside the vocabulary");
  const uint32_t n = c.pos;
  verify_k_ = kNoVerify;
  if (drafter_) replay_aux(drafter_->projection(n, j + 1));   // rows 0..j of the taps: positions n .. n + j
  c.pos = n + j + 1;
  c.cur_token[0] = next_token;
  c.n_active = 1;
}

void MuseEngine::replay_aux(l0::CmdList& list) {
  st(0).queue.execute(list, &st(0).fence);
  st(0).fence.wait();
}

const l0::Mem& MuseEngine::control_mem() const { return st(0).buffers.control; }
const l0::Mem& MuseEngine::embed() const {
  if (!w_.parts.at(0).embed) throw std::logic_error("runtime::muse::MuseEngine: device 0 holds no embedding");
  return *w_.parts.at(0).embed;
}
const loader::DeviceWeight& MuseEngine::lm_head() const {
  const loader::MuseDevicePart& p = w_.parts.back();
  if (!p.lm_head) throw std::logic_error("runtime::muse::MuseEngine: the last device holds no lm_head");
  return *p.lm_head;
}
size_t MuseEngine::verify_launches(uint32_t M) const {
  return M >= 1 && M <= verify_steps_.size() && verify_steps_[M - 1] ? verify_steps_[M - 1]->kernel_count : 0;
}

void MuseEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("pipeline hand-off: " + what + ". The session's state is not valid; reset() before the next step");
}

void MuseEngine::step_once() {
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  Stage& last = st(uint32_t(st_.size()) - 1);
  const size_t pos = last.ctl->pos, n = last.ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error("runtime::muse::MuseEngine: pos " + std::to_string(pos) + " + n_active " + std::to_string(n) +
                             " exceeds max_len " + std::to_string(max_len_) +
                             " - the full layers' KV and the RoPE table stop there");
  if (st_.size() == 1) {
    st(0).queue.execute(st(0).step->list, &st(0).fence);
    st(0).fence.wait();
    return;
  }
  if (link_->event) link_->event->host_reset();
  const bool send = !drop_next_;
  drop_next_ = false;
  if (send) {
    st(0).queue.execute(st(0).step->list, &st(0).fence);
    pending_[0] = true;
  }
  st(1).queue.execute(st(1).step->list, &st(1).fence);
  pending_[1] = true;
  if (!settle(1)) {
    // Device 1 still waits (copy: on the event device 0 never signalled): release it so its queue drains on garbage
    // instead of holding the device, then report.
    if (link_->event) link_->event->host_signal();
    const bool drained = settle(1);
    settle(0);
    fail(std::string("device 1's step did not finish within ") + std::to_string(opt_.timeout_ms) +
         " ms - device 0's residual never arrived" +
         (link_->event ? std::string(" (the cross-device event was not signalled; device 1 ") +
                             (drained ? "released" : "did not drain either") + ")"
                       : std::string(drained ? "" : " (device 1 did not drain)")) +
         (send ? "" : " [device 0's list was not submitted: drop_next_handoff()]"));
  }
  if (!settle(0)) fail("device 0's step did not finish within " + std::to_string(opt_.timeout_ms) + " ms");
  if (link_->mode == PpHandoff::Peer) {
    const uint32_t* w = link_->recv_state.as<uint32_t>();
    const uint32_t status = w[kStStatus];
    if (status == kStatusTimeout)
      fail("device 1's pp_recv gave up after " + std::to_string(opt_.spin_limit) + " loads of the flag: it read " +
           std::to_string(w[kStFlag]) + ", expected " + std::to_string(w[kStWant]) +
           " (device 0 never published the residual)" + (send ? "" : " [device 0's list was not submitted: drop_next_handoff()]"));
    if (status == kStatusStale)
      fail("the flag reached " + std::to_string(w[kStWant]) + " but the stamp after the norm sums reads " +
           std::to_string(w[kStStamp]) + " - the data did not arrive with the flag (a silent hand-off, spec 16 §2)");
    if (status != 0) fail("pp_recv reported status " + std::to_string(status));
  }
  // The token's way back: device 1's argmax advanced pos and wrote the id; device 0 reads both from its own block on
  // the next step.
  std::memcpy(st(0).ctl, st(1).ctl, sizeof(Control));
}

void MuseEngine::set_token(uint32_t id) {
  if (id >= w_.desc.vocab)
    throw std::runtime_error("runtime::muse::MuseEngine: id " + std::to_string(id) + " is outside the vocabulary (" +
                             std::to_string(w_.desc.vocab) + ")");
  for (auto& s : st_) s->ctl->cur_token[0] = id;
}

void MuseEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (auto& s : st_) s->ctl->n_active = 1;
    set_token(id);
    if (drafter_) {   // spec 23f: one plain token through the M = 1 verify list, its taps projected
      verify(nullptr, 0);
      commit(0, verify_ids()[0]);
      continue;
    }
    step_once();
  }
}

std::vector<uint32_t> MuseEngine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  for (auto& s : st_) s->ctl->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = st(uint32_t(st_.size()) - 1).ctl->cur_token[0];   // == device 0's after the mirror
    const Clock::time_point f0 = Clock::now();
    if (drafter_) {   // spec 23f: the plain token through the M = 1 verify list (the drafter's ring stays filled)
      verify(nullptr, 0);
      commit(0, verify_ids()[0]);
    } else {
      step_once();
    }
    fence_ms += ms_since(f0);
    out.push_back(id);
    if (on_token) on_token(id);
  }
  last_gen_ms_ = ms_since(t0);
  last_fence_ms_ = fence_ms;
  last_tok_per_s_ = (n != 0 && last_gen_ms_ > 0.0) ? double(n) * 1000.0 / last_gen_ms_ : 0.0;
  return out;
}

uint32_t MuseEngine::pending() const { return st(uint32_t(st_.size()) - 1).ctl->cur_token[0]; }
uint32_t MuseEngine::pos() const { return st(uint32_t(st_.size()) - 1).ctl->pos; }
Control MuseEngine::control(uint32_t dev) const { return *st(dev).ctl; }
size_t MuseEngine::launches(uint32_t dev) const { return st(dev).step->kernel_count; }
size_t MuseEngine::launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < st_.size(); ++i) n += launches(i);
  return n;
}
const CapturedStep& MuseEngine::step(uint32_t dev) const { return *st(dev).step; }
const MuseBuffers& MuseEngine::buffers(uint32_t dev) const { return st(dev).buffers; }

std::vector<float> MuseEngine::read_logits() {
  Stage& s = st(uint32_t(st_.size()) - 1);
  std::vector<float> v(w_.desc.vocab);
  s.imm.copy(v.data(), s.buffers.logits.ptr(), v.size() * 4);
  return v;
}

void MuseEngine::read_logits_into(float* host) {
  Stage& s = st(uint32_t(st_.size()) - 1);
  s.imm.copy(host, s.buffers.logits.ptr(), size_t(w_.desc.vocab) * 4);
}

// --- spec 23e: prefix-cache snapshots ---------------------------------------------------------------------------
l0::Mem& MuseEngine::snap_mem(uint32_t dev, SnapTensor t) const {
  MuseBuffers& b = st(dev).buffers;
  const std::unique_ptr<l0::Mem>& m = t == SnapTensor::FullK   ? b.full_k
                                      : t == SnapTensor::FullV ? b.full_v
                                      : t == SnapTensor::RingK ? b.ring_k
                                                               : b.ring_v;
  if (!m)
    throw std::logic_error("runtime::muse::MuseEngine: device " + std::to_string(dev) +
                           " holds no allocation for a snapshot run (state_runs / kv_runs disagree with MuseBuffers)");
  return *m;
}

void MuseEngine::settle_all(const char* what) {
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error(std::string("runtime::muse::MuseEngine::") + what + ": device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
}

// Spec 23f: with a drafter, its ring after the target's state (spec 19d decision 5 as ruled, the operator's option a).
size_t MuseEngine::state_bytes() const {
  return state_snapshot_bytes(w_.desc) + (drafter_ ? drafter_->ring_bytes() : 0);
}
size_t MuseEngine::kv_bytes(uint32_t n_pos) const { return kv_snapshot_bytes(w_.desc, n_pos); }

void MuseEngine::save_state(void* host) {
  settle_all("save_state");
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : state_runs(w_.desc, w_.placement, pos())) {
    if (r.zero)
      std::memset(h, 0, r.bytes);   // positions before 0: never read, zero on the host
    else
      st(r.device).imm.copy(h, static_cast<const uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
  if (drafter_) drafter_->save_ring(st(0).imm, h);
}

void MuseEngine::load_state(const void* host, uint32_t p) {
  if (p > max_len_)
    throw std::runtime_error("runtime::muse::MuseEngine::load_state: pos " + std::to_string(p) + " exceeds max_len " +
                             std::to_string(max_len_));
  if (broken_) reset();   // a failed hand-off: start from a clean session, then the restore rewrites what it reads
  settle_all("load_state");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : state_runs(w_.desc, w_.placement, p)) {
    st(r.device).imm.copy(static_cast<uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
  if (drafter_) drafter_->load_ring(st(0).imm, h);
  verify_k_ = kNoVerify;
  for (auto& s : st_) {
    s->ctl->pos = p;
    s->ctl->n_active = 0;
  }
}

void MuseEngine::save_kv(uint32_t begin, uint32_t end, void* host) {
  settle_all("save_kv");
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : kv_runs(w_.desc, w_.placement, max_len_, begin, end)) {
    st(r.device).imm.copy(h, static_cast<const uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
}

void MuseEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  settle_all("load_kv");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : kv_runs(w_.desc, w_.placement, max_len_, begin, end)) {
    st(r.device).imm.copy(static_cast<uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
}

std::vector<uint16_t> MuseEngine::read_kv(uint32_t layer, uint32_t first, uint32_t count, bool v) {
  const model::MuseGlimmerDesc& d = w_.desc;
  if (layer >= d.layers) throw std::out_of_range("read_kv: layer " + std::to_string(layer));
  Stage& s = st(w_.placement.device_of(layer));
  const uint8_t* base = static_cast<const uint8_t*>(v ? s.buffers.v_layer(layer) : s.buffers.k_layer(layer));
  const bool sliding = !d.is_full(layer);
  const size_t row = size_t(d.kv_n()) * 2;
  if (!sliding && size_t(first) + count > max_len_) throw std::out_of_range("read_kv: past max_len");
  if (sliding && count > model::MuseGlimmerDesc::kRing) throw std::out_of_range("read_kv: more positions than the ring holds");
  std::vector<uint16_t> out(size_t(count) * d.kv_n());
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t p = first + i;
    const size_t r = sliding ? (p & (model::MuseGlimmerDesc::kRing - 1)) : p;
    s.imm.copy(out.data() + size_t(i) * d.kv_n(), base + r * row, row);
  }
  return out;
}

std::vector<uint16_t> MuseEngine::read_ring(uint32_t layer, bool v) {
  const model::MuseGlimmerDesc& d = w_.desc;
  if (layer >= d.layers || d.is_full(layer)) throw std::out_of_range("read_ring: layer " + std::to_string(layer) + " has no ring");
  Stage& s = st(w_.placement.device_of(layer));
  std::vector<uint16_t> out(size_t(model::MuseGlimmerDesc::kRing) * d.kv_n());
  s.imm.copy(out.data(), v ? s.buffers.v_layer(layer) : s.buffers.k_layer(layer), out.size() * 2);
  return out;
}

std::vector<uint16_t> MuseEngine::read_debug_resid() {
  const model::MuseGlimmerDesc& d = w_.desc;
  if (!taps_[0]) throw std::runtime_error("runtime::muse::MuseEngine: constructed without debug_tap");
  std::vector<uint16_t> v(size_t(d.layers) * d.hidden);
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const uint32_t first = w_.placement.first(i), n = w_.placement.count(i);
    st(i).imm.copy(v.data() + size_t(first) * d.hidden,
                   static_cast<const uint8_t*>(taps_[i]->ptr()) + size_t(first) * d.hidden * 2, size_t(n) * d.hidden * 2);
  }
  return v;
}

MemoryComponents MuseEngine::memory_use(uint32_t dev) const {
  const loader::MuseDevicePart& P = w_.parts.at(dev);
  const MuseBuffers& b = st(dev).buffers;
  MemoryComponents c;
  c.model = P.bytes;
  c.kv = b.kv_bytes() + b.ring_bytes();
  c.decode_state = b.scratch_bytes() + (taps_[dev] ? taps_[dev]->size() : 0) + (link_ ? link_->bytes(dev) : 0);
  c.prefill_scratch = dev < pf_dev_bytes_.size() ? pf_dev_bytes_[dev] : 0;   // spec 23d: 0 until prepared
  if (dev == 0 && vin_) c.decode_state += vin_->size() + vout_->size();      // spec 23f
  if (dev + 1 == st_.size() && drafter_) c.drafter = drafter_->bytes();    // spec 23f: weights + buffers
  return c;
}

std::string MuseEngine::memory_line() const {
  std::string s;
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const std::string label = st_.size() > 1 ? "memory, device " + std::to_string(i) : std::string("memory");
    s += (i ? "\n" : "") + format_memory(label.c_str(), memory_use(i), st(i).ctx.memory_bytes());
  }
  return s;
}

}  // namespace runtime::muse
