#include "runtime/kolibri/kolibri_engine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "runtime/kolibri/kolibri_capture.h"
#include "runtime/kolibri/kolibri_sizes.h"

namespace runtime::kolibri {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
}  // namespace

struct KolibriEngine::Stage {
  Stage(l0::Context& c, const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t dev, uint32_t max_len,
        KolAttn a)
      : ctx(c), buffers(c, d, p, dev, max_len, a), queue(c), fence(queue), imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()) {}
  l0::Context& ctx;
  KolibriBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  std::optional<CapturedStep> step;
};

KolibriEngine::Stage& KolibriEngine::st(uint32_t dev) const {
  if (dev >= st_.size())
    throw std::out_of_range("runtime::kolibri::KolibriEngine: device " + std::to_string(dev) + " of " +
                            std::to_string(st_.size()));
  return *st_[dev];
}

l0::Context& KolibriEngine::stage_ctx(uint32_t dev) const { return st(dev).ctx; }
KolibriBuffers& KolibriEngine::stage_buffers(uint32_t dev) const { return st(dev).buffers; }
Control* KolibriEngine::stage_control(uint32_t dev) const { return st(dev).ctl; }
l0::CmdList& KolibriEngine::stage_imm(uint32_t dev) const { return st(dev).imm; }

KolibriEngine::KolibriEngine(std::vector<l0::Context*> devices, loader::KolLoadedModel model, uint32_t max_len,
                             bool debug_tap, const PipelineOptions& opt)
    : KolibriEngine(std::move(devices), std::move(model), max_len, DebugTap{debug_tap, TapAt::LayerOut}, opt) {}

KolibriEngine::KolibriEngine(std::vector<l0::Context*> devices, loader::KolLoadedModel model, uint32_t max_len,
                             DebugTap tap, const PipelineOptions& opt)
    : model_(std::move(model)), max_len_(max_len), opt_(opt), attn_(kolibri_attn()), tap_at_(tap.at) {
  const bool debug_tap = tap.on;
  const model::Kolibri1Desc& d = model_.desc;
  const model::KolPlacement& p = model_.placement;
  model::validate(p, d);
  if (model_.max_len != max_len)
    throw std::runtime_error("runtime::kolibri::KolibriEngine: the model was loaded with max_len " +
                             std::to_string(model_.max_len) + " but the engine was asked for " + std::to_string(max_len) +
                             " - the RoPE table and the KV are one number");
  if (devices.size() != p.devices || model_.parts.size() != p.devices)
    throw std::invalid_argument("runtime::kolibri::KolibriEngine: " + std::to_string(devices.size()) + " context(s), " +
                                std::to_string(model_.parts.size()) + " loaded part(s), a " + std::to_string(p.devices) +
                                "-device placement");
  if (p.devices == 2) {
    if (devices[1]->handle() != devices[0]->handle())
      throw std::invalid_argument("runtime::kolibri::KolibriEngine: device 1 is not a view of device 0's Level Zero "
                                  "context (spec 16 decision 1)");
    if (!devices[0]->can_access_peer(*devices[1]))
      throw std::runtime_error("--parallel pp: device 0 (" + devices[0]->name() + ") cannot access device 1's memory "
                               "(zeDeviceCanAccessPeer is false), and the hand-off writes it. Peer access needs the "
                               "P2P-capable kernel and both cards under one root complex (docs/10-the-box.md)");
    if (opt.timeout_ms == 0)
      throw std::invalid_argument("runtime::kolibri::KolibriEngine: a hand-off needs a non-zero bound");
  }
  for (uint32_t i = 0; i < p.devices; ++i) {
    st_.push_back(std::make_unique<Stage>(*devices[i], d, p, i, max_len, attn_));
    taps_.push_back(debug_tap ? std::make_unique<l0::Mem>(*devices[i], l0::MemKind::Device, tap_bytes(d)) : nullptr);
  }
  pending_.assign(p.devices, false);
  StageLink binding;
  if (p.devices == 2) {
    link_ = std::make_unique<PipelineLink>(*devices[0], *devices[1], landing_layout(d));
    binding = link_->binding();
  }
  for (uint32_t i = 0; i < p.devices; ++i)
    st(i).step.emplace(build(st(i).ctx, model_.parts[i], d, st(i).buffers, link_ ? &binding : nullptr, taps_[i].get(), tap_at_));
  reset();
}

KolibriEngine::~KolibriEngine() {
  for (uint32_t i = 0; i < st_.size(); ++i) {   // never free buffers under a running list
    try {
      settle(i);
    } catch (...) {
    }
  }
}

bool KolibriEngine::settle(uint32_t dev) {
  if (!pending_[dev]) return true;
  if (st(dev).fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull)) pending_[dev] = false;
  return !pending_[dev];
}

void KolibriEngine::reset() {
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error("runtime::kolibri::KolibriEngine::reset: device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
  if (pf_settle_ && !pf_settle_())   // spec 20d: no prefill list may still run into the buffers
    throw std::runtime_error("runtime::kolibri::KolibriEngine::reset: a prefill list is still running after " +
                             std::to_string(opt_.prefill_timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i) st(i).buffers.zero(st(i).imm);
  if (link_) link_->zero(st(1).imm);
  if (vlink_) vlink_->zero(st(1).imm);                                // Kolibri DSpark
  if (drafter_) drafter_->reset(st(uint32_t(st_.size()) - 1).imm);    // Kolibri DSpark: a cold drafter
  verify_k_ = kNoVerify;
  logits_row_ = -1;
  broken_ = false;
  drop_next_ = false;
}

void KolibriEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("pipeline hand-off: " + what + ". The session's state is not valid; reset() before the next step");
}

void KolibriEngine::step_once() {
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  logits_row_ = -1;   // Kolibri DSpark: this step's row is the decode scratch's
  Stage& last = st(uint32_t(st_.size()) - 1);
  const size_t pos = last.ctl->pos, n = last.ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error("runtime::kolibri::KolibriEngine: pos " + std::to_string(pos) + " + n_active " +
                             std::to_string(n) + " exceeds max_len " + std::to_string(max_len_) +
                             " - the full layers' KV and the RoPE table stop there");
  if (st_.size() == 1) {
    st(0).queue.execute(st(0).step->list, &st(0).fence);
    st(0).fence.wait();
    return;
  }
  link_->event->host_reset();
  const bool send = !drop_next_;
  drop_next_ = false;
  if (send) {
    st(0).queue.execute(st(0).step->list, &st(0).fence);
    pending_[0] = true;
  }
  st(1).queue.execute(st(1).step->list, &st(1).fence);
  pending_[1] = true;
  if (!settle(1)) {
    // Device 1 still waits (on the event device 0 never signalled): release it so its queue
    // drains on garbage instead of holding the device, then report.
    link_->event->host_signal();
    const bool drained = settle(1);
    settle(0);
    fail(std::string("device 1's step did not finish within ") + std::to_string(opt_.timeout_ms) +
         " ms - device 0's residual never arrived" +
         " (the cross-device event was not signalled; device 1 " +
         (drained ? "released" : "did not drain either") + ")" +
         (send ? "" : " [device 0's list was not submitted: drop_next_handoff()]"));
  }
  if (!settle(0)) fail("device 0's step did not finish within " + std::to_string(opt_.timeout_ms) + " ms");
  // The token's way back: device 1's argmax advanced pos and wrote the id; device 0 reads both from
  // its own block on the next step.
  std::memcpy(st(0).ctl, st(1).ctl, sizeof(Control));
}

void KolibriEngine::set_token(uint32_t id) {
  if (id >= model_.desc.vocab)
    throw std::runtime_error("runtime::kolibri::KolibriEngine: id " + std::to_string(id) +
                             " is outside the vocabulary (" + std::to_string(model_.desc.vocab) + ")");
  for (auto& s : st_) s->ctl->cur_token[0] = id;
}

void KolibriEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (auto& s : st_) s->ctl->n_active = 1;
    set_token(id);
    if (spec_on()) {   // Kolibri DSpark: one plain token through the M = 1 verify list (a drafter's ring stays filled)
      verify(nullptr, 0);
      commit(0, verify_ids()[0]);
      continue;
    }
    step_once();
  }
}

std::vector<uint32_t> KolibriEngine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  for (auto& s : st_) s->ctl->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = st(uint32_t(st_.size()) - 1).ctl->cur_token[0];   // == device 0's after the mirror
    const Clock::time_point f0 = Clock::now();
    if (spec_on()) {   // Kolibri DSpark: the plain token through the M = 1 verify list
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

uint32_t KolibriEngine::pos() const { return st(uint32_t(st_.size()) - 1).ctl->pos; }

Control KolibriEngine::control(uint32_t dev) const { return *st(dev).ctl; }

size_t KolibriEngine::launches(uint32_t dev) const { return st(dev).step->kernel_count; }
size_t KolibriEngine::launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < st_.size(); ++i) n += launches(i);
  return n;
}
const CapturedStep& KolibriEngine::step(uint32_t dev) const { return *st(dev).step; }

std::vector<uint16_t> KolibriEngine::read_debug_resid() {
  const model::Kolibri1Desc& d = model_.desc;
  if (!taps_[0]) throw std::runtime_error("runtime::kolibri::KolibriEngine: constructed without debug_tap");
  std::vector<uint16_t> v(size_t(d.layers) * d.hidden);
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const uint32_t first = model_.placement.first(i), n = model_.placement.count(i);
    st(i).imm.copy(v.data() + size_t(first) * d.hidden, static_cast<const uint8_t*>(taps_[i]->ptr()) + size_t(first) * d.hidden * 2,
                   size_t(n) * d.hidden * 2);
  }
  return v;
}

std::vector<uint32_t> KolibriEngine::read_routes() {
  const model::Kolibri1Desc& d = model_.desc;
  std::vector<uint32_t> v(size_t(d.layers) * kRouteWords);
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const uint32_t first = model_.placement.first(i), n = model_.placement.count(i);
    st(i).imm.copy(v.data() + size_t(first) * kRouteWords,
                   static_cast<const uint8_t*>(st(i).buffers.routes.ptr()) + route_at(first), size_t(n) * kRouteWords * 4);
  }
  return v;
}

std::vector<float> KolibriEngine::read_logits() {
  std::vector<float> v(model_.desc.vocab);
  read_logits_into(v.data());
  return v;
}

void KolibriEngine::read_logits_into(float* host) {
  const uint32_t last = uint32_t(st_.size()) - 1;
  Stage& s = st(last);
  if (logits_row_ >= 0) {   // Kolibri DSpark: the last commit's row j of the verify scratch (the row its next id came from)
    s.imm.copy(host, static_cast<const uint8_t*>(vb_[last]->logits.ptr()) + size_t(logits_row_) * model_.desc.vocab * 4,
               size_t(model_.desc.vocab) * 4);
    return;
  }
  s.imm.copy(host, s.buffers.logits.ptr(), size_t(model_.desc.vocab) * 4);
}

uint32_t KolibriEngine::pending() const { return st(uint32_t(st_.size()) - 1).ctl->cur_token[0]; }

// --- spec 20e: prefix-cache snapshots ----------------------------------------------------------------
l0::Mem& KolibriEngine::snap_mem(uint32_t dev, SnapTensor t) const {
  KolibriBuffers& b = st(dev).buffers;
  switch (t) {
    case SnapTensor::FullK: return b.full_k;
    case SnapTensor::FullV: return b.full_v;
    case SnapTensor::RingK: return b.ring_k;
    default: return b.ring_v;
  }
}

void KolibriEngine::settle_all(const char* what) {
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error(std::string("runtime::kolibri::KolibriEngine::") + what + ": device " +
                               std::to_string(i) + "'s last step is still running after " +
                               std::to_string(opt_.timeout_ms) + " ms");
}

// Kolibri DSpark: with a drafter, its ring after the target's state (19d's rule).
size_t KolibriEngine::state_bytes() const {
  return state_snapshot_bytes(model_.desc) + (drafter_ ? drafter_->ring_bytes() : 0);
}
size_t KolibriEngine::kv_bytes(uint32_t n_pos) const { return kv_snapshot_bytes(model_.desc, n_pos); }

void KolibriEngine::save_state(void* host) {
  settle_all("save_state");
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : state_runs(model_.desc, model_.placement, pos())) {
    if (r.zero)
      std::memset(h, 0, r.bytes);   // positions before 0: never read, zero on the host
    else
      st(r.device).imm.copy(h, static_cast<const uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
  if (drafter_) drafter_->save_ring(st(uint32_t(st_.size()) - 1).imm, h);   // Kolibri DSpark: the ring
}

void KolibriEngine::load_state(const void* host, uint32_t p) {
  if (p > max_len_)
    throw std::runtime_error("runtime::kolibri::KolibriEngine::load_state: pos " + std::to_string(p) +
                             " exceeds max_len " + std::to_string(max_len_));
  if (broken_) reset();   // a failed hand-off: the restore below rewrites what the session reads
  settle_all("load_state");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : state_runs(model_.desc, model_.placement, p)) {
    st(r.device).imm.copy(static_cast<uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
  if (drafter_) drafter_->load_ring(st(uint32_t(st_.size()) - 1).imm, h);   // Kolibri DSpark
  verify_k_ = kNoVerify;
  logits_row_ = -1;
  for (auto& s : st_) {
    s->ctl->pos = p;
    s->ctl->n_active = 0;
  }
}

void KolibriEngine::save_kv(uint32_t begin, uint32_t end, void* host) {
  settle_all("save_kv");
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : kv_runs(model_.desc, model_.placement, max_len_, begin, end)) {
    st(r.device).imm.copy(h, static_cast<const uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
}

void KolibriEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  settle_all("load_kv");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : kv_runs(model_.desc, model_.placement, max_len_, begin, end)) {
    st(r.device).imm.copy(static_cast<uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
}

std::vector<uint16_t> KolibriEngine::read_kv(uint32_t layer, uint32_t first, uint32_t count, bool v) {
  const model::Kolibri1Desc& d = model_.desc;
  if (layer >= d.layers) throw std::out_of_range("read_kv: layer " + std::to_string(layer));
  Stage& s = st(model_.placement.device_of(layer));
  const uint8_t* base = static_cast<const uint8_t*>(v ? s.buffers.v_layer(layer) : s.buffers.k_layer(layer));
  const bool sliding = d.is_sliding(layer);
  const size_t row = size_t(d.kv_n()) * 2;
  if (!sliding && size_t(first) + count > max_len_) throw std::out_of_range("read_kv: past max_len");
  if (sliding && count > model::Kolibri1Desc::kRing) throw std::out_of_range("read_kv: more positions than the ring holds");
  std::vector<uint16_t> out(size_t(count) * d.kv_n());
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t p = first + i;
    const size_t r = sliding ? (p & (model::Kolibri1Desc::kRing - 1)) : p;
    s.imm.copy(out.data() + size_t(i) * d.kv_n(), base + r * row, row);
  }
  return out;
}

MemoryComponents KolibriEngine::memory_use(uint32_t dev) const {
  const loader::KolDevicePart& P = model_.parts.at(dev);
  const KolibriBuffers& b = st(dev).buffers;
  MemoryComponents c;
  c.model = P.bytes + P.rope_bytes;
  c.kv = b.full_k.size() + b.full_v.size() + b.ring_k.size() + b.ring_v.size();
  c.decode_state = b.control.size() + b.scratch_bytes() + (taps_[dev] ? taps_[dev]->size() : 0) +
                   (link_ ? link_->bytes(dev) : 0);
  c.prefill_scratch = dev < pf_dev_bytes_.size() ? pf_dev_bytes_[dev] : 0;   // spec 20d: 0 until prepared
  if (dev < vb_.size()) c.decode_state += vb_[dev]->bytes() + (vlink_ ? vlink_->bytes(dev) : 0);   // Kolibri DSpark
  if (dev + 1 == st_.size()) {
    if (drafter_) c.drafter = drafter_->bytes();   // its weights + buffers (the embedding replica is the CLI's)
    if (pf_taps_) c.prefill_scratch += pf_taps_->size();
  }
  return c;
}

// --- Kolibri DSpark: speculative decoding ---------------------------------------------------------------------------------
void KolibriEngine::enable_spec() {
  if (spec_on()) throw std::logic_error("runtime::kolibri::KolibriEngine::enable_spec: the verify lists are already on");
  if (attn_ != KolAttn::Flash)
    throw std::runtime_error("runtime::kolibri::KolibriEngine::enable_spec: the verify lists run the flash attention only - "
                             "unset B70_KOLIBRI_ATTN=eager for --spec");
  settle_all("enable_spec");
  const model::Kolibri1Desc& d = model_.desc;
  for (uint32_t i = 0; i < st_.size(); ++i) {
    vb_.push_back(std::make_unique<KolibriVerifyBuffers>(st(i).ctx, d));
    if (vb_.back()->bytes() != verify_sizes(d).total())
      throw std::logic_error("runtime::kolibri::KolibriEngine::enable_spec: device " + std::to_string(i) +
                             "'s verify buffers are not verify_sizes()'");
  }
  if (st_.size() == 2) {
    vlink_ = std::make_unique<PipelineLink>(st(0).ctx, st(1).ctx, verify_landing_layout(d));
    for (uint32_t i = 0; i < 2; ++i)
      if (vlink_->bytes(i) != verify_link_bytes(d, i))
        throw std::logic_error("runtime::kolibri::KolibriEngine::enable_spec: the verify link is not verify_link_bytes()'");
    vlink_->zero(st(1).imm);
  }
  vsteps_.resize(st_.size());
  for (auto& dev : vsteps_) dev.resize(kVerifyRows);
  verify_k_ = kNoVerify;
}

void KolibriEngine::attach_drafter(TapSink& dr) {
  const model::Kolibri1Desc& d = model_.desc;
  if (!spec_on())
    throw std::logic_error("runtime::kolibri::KolibriEngine::attach_drafter: enable_spec first (the verify lists write its taps)");
  if (drafter_) throw std::logic_error("runtime::kolibri::KolibriEngine::attach_drafter: a drafter is already attached");
  if (dr.max_k() == 0 || dr.max_k() + 1 > kVerifyRows)
    throw std::invalid_argument("runtime::kolibri::KolibriEngine::attach_drafter: the drafter drafts up to " +
                                std::to_string(dr.max_k()) + ", the verify lists hold 8 rows (K 1..7)");
  for (uint32_t t : dr.tap_layers())
    if (t >= d.layers)
      throw std::invalid_argument("runtime::kolibri::KolibriEngine::attach_drafter: the drafter taps layer " +
                                  std::to_string(t) + " of a " + std::to_string(d.layers) + "-layer target");
  if (dr.taps().size() < size_t(kVerifyRows) * dr.tap_layers().size() * d.hidden * 2)
    throw std::invalid_argument("runtime::kolibri::KolibriEngine::attach_drafter: the drafter's tap buffer is smaller than "
                                "[8][taps][hidden] bf16");
  settle_all("attach_drafter");
  for (auto& dev : vsteps_)
    for (auto& s : dev) s.reset();   // lists captured before bound no taps: recaptured with them
  drafter_ = &dr;
  drafter_->reset(st(uint32_t(st_.size()) - 1).imm);
  // Every verify list (M = 1..8, every device) captured now, with the taps: a missing binary fails at startup, never inside
  // a request (prepare_prefill's rule).
  for (uint32_t M = 1; M <= kVerifyRows; ++M) capture_verify(M);
}

void KolibriEngine::capture_verify(uint32_t M) {
  const StageLink binding = vlink_ ? vlink_->binding() : StageLink{};
  for (uint32_t i = 0; i < st_.size(); ++i) {
    std::optional<CapturedStep>& s = vsteps_[i].at(M - 1);
    if (s) continue;
    VerifyBind v;
    v.vb = vb_[i].get();
    v.M = M;
    if (drafter_) {
      v.taps = &drafter_->taps();
      v.tap_layers = &drafter_->tap_layers();
    }
    s.emplace(build_verify(st(i).ctx, model_.parts[i], model_.desc, st(i).buffers, vlink_ ? &binding : nullptr, v));
  }
}

uint32_t KolibriEngine::max_verify_k() const {
  if (!spec_on()) return 0;
  const uint32_t k = drafter_ ? std::min(drafter_->max_k(), kVerifyRows - 1) : kVerifyRows - 1;
  const uint32_t p = pos();
  if (size_t(p) + 1 >= size_t(max_len_)) return 0;
  return std::min<uint32_t>(k, max_len_ - p - 1);
}

void KolibriEngine::run_verify(uint32_t M) {
  const uint32_t nd = uint32_t(st_.size());
  capture_verify(M);   // captured by attach_drafter; only a drafter-less engine (the tests') captures here
  if (nd == 1) {
    st(0).queue.execute(vsteps_[0][M - 1]->list, &st(0).fence);
    st(0).fence.wait();
    return;
  }
  vlink_->event->host_reset();
  st(0).queue.execute(vsteps_[0][M - 1]->list, &st(0).fence);
  pending_[0] = true;
  st(1).queue.execute(vsteps_[1][M - 1]->list, &st(1).fence);
  pending_[1] = true;
  if (!settle(1)) {
    vlink_->event->host_signal();   // release device 1 so its queue drains on garbage, then report
    const bool drained = settle(1);
    settle(0);
    fail("device 1's M = " + std::to_string(M) + " verify did not finish within " + std::to_string(opt_.timeout_ms) +
         " ms - device 0's rows never arrived (the verify link's event was not signalled; device 1 " +
         (drained ? "released" : "did not drain either") + ")");
  }
  if (!settle(0)) fail("device 0's M = " + std::to_string(M) + " verify did not finish within " +
                       std::to_string(opt_.timeout_ms) + " ms");
}

void KolibriEngine::verify(const uint32_t* drafts, uint32_t k) {
  if (!spec_on()) throw std::logic_error("runtime::kolibri::KolibriEngine::verify: no verify lists (enable_spec)");
  if (broken_) throw std::runtime_error("runtime::kolibri::KolibriEngine::verify: a hand-off failed earlier; reset() first");
  if (k > max_verify_k())
    throw std::invalid_argument("runtime::kolibri::KolibriEngine::verify: k " + std::to_string(k) + " exceeds " +
                                std::to_string(max_verify_k()) + " at pos " + std::to_string(pos()));
  Control& c0 = *st(0).ctl;
  Control& cl = *st(uint32_t(st_.size()) - 1).ctl;
  const uint32_t n = cl.pos, x = cl.cur_token[0], M = k + 1;
  if (size_t(n) + M > size_t(max_len_))
    throw std::runtime_error("runtime::kolibri::KolibriEngine::verify: pos " + std::to_string(n) + " + " +
                             std::to_string(M) + " rows exceeds max_len " + std::to_string(max_len_));
  for (uint32_t i = 0; i < k; ++i) {
    if (drafts[i] >= model_.desc.vocab)
      throw std::invalid_argument("runtime::kolibri::KolibriEngine::verify: draft " + std::to_string(drafts[i]) +
                                  " is outside the vocabulary");
    c0.cur_token[1 + i] = drafts[i];   // the gather reads cur_token[0 .. k] on device 0
  }
  for (uint32_t i = 0; i < st_.size(); ++i) {
    st(i).ctl->n_active = M;
    Control& hi = *vb_[i]->control_hi.as<Control>();   // the attention's rows 4..M-1 (a list at M <= 4 never reads it)
    hi.pos = n + kAttnRowsMax;
    hi.n_active = M > kAttnRowsMax ? M - kAttnRowsMax : 0;
  }
  run_verify(M);
  for (uint32_t r = 0; r < M; ++r) verify_ids_[r] = cl.out_token[r];
  // argmax_stage2 advanced the last device's pos and wrote its last id into cur_token[0]: the host commits, so put both back.
  cl.pos = n;
  cl.cur_token[0] = x;
  verify_k_ = k;
  verify_pos_ = n;
}

void KolibriEngine::read_verify_logits(float* host, uint32_t rows) {
  if (verify_k_ == kNoVerify || rows == 0 || rows > verify_k_ + 1)
    throw std::logic_error("runtime::kolibri::KolibriEngine::read_verify_logits: " + std::to_string(rows) +
                           " rows after no verify / a verify of fewer");
  const uint32_t last = uint32_t(st_.size()) - 1;
  st(last).imm.copy(host, vb_[last]->logits.ptr(), size_t(rows) * model_.desc.vocab * 4);
}

void KolibriEngine::commit(uint32_t j, uint32_t next_token) {
  Control& cl = *st(uint32_t(st_.size()) - 1).ctl;
  if (verify_k_ == kNoVerify || cl.pos != verify_pos_)
    throw std::logic_error("runtime::kolibri::KolibriEngine::commit: no verify at pos " + std::to_string(cl.pos));
  if (j > verify_k_)
    throw std::invalid_argument("runtime::kolibri::KolibriEngine::commit: j " + std::to_string(j) + " > k " +
                                std::to_string(verify_k_));
  if (next_token >= model_.desc.vocab)
    throw std::invalid_argument("runtime::kolibri::KolibriEngine::commit: id " + std::to_string(next_token) +
                                " is outside the vocabulary");
  const uint32_t n = cl.pos;
  verify_k_ = kNoVerify;
  if (drafter_) replay_aux(drafter_->projection(n, j + 1));   // rows 0..j of the taps: positions n .. n + j
  for (auto& s : st_) {
    s->ctl->pos = n + j + 1;
    s->ctl->cur_token[0] = next_token;
    s->ctl->n_active = 1;
  }
  logits_row_ = int(j);   // read_logits: the row next_token came from
}

void KolibriEngine::replay_aux(l0::CmdList& list) {
  Stage& s = st(uint32_t(st_.size()) - 1);
  s.queue.execute(list, &s.fence);
  s.fence.wait();
}

size_t KolibriEngine::verify_launches(uint32_t M) const {
  size_t n = 0;
  if (M == 0 || M > kVerifyRows) return 0;
  for (const auto& dev : vsteps_)
    if (dev[M - 1]) n += dev[M - 1]->kernel_count;
  return n;
}

const l0::Mem& KolibriEngine::control_mem(uint32_t dev) const { return st(dev).buffers.control; }
const loader::DeviceWeight& KolibriEngine::lm_head() const {
  const loader::KolDevicePart& p = model_.parts.back();
  if (!p.lm_head) throw std::logic_error("runtime::kolibri::KolibriEngine: the last device holds no lm_head");
  return *p.lm_head;
}
const l0::Mem& KolibriEngine::embedding() const {
  if (!model_.parts.front().embed) throw std::logic_error("runtime::kolibri::KolibriEngine: device 0 holds no embedding");
  return *model_.parts.front().embed;
}

std::string KolibriEngine::memory_line() const {
  std::string s;
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const std::string label = st_.size() > 1 ? "memory, device " + std::to_string(i) : std::string("memory");
    s += (i ? "\n" : "") + format_memory(label.c_str(), memory_use(i), st(i).ctx.memory_bytes());
  }
  return s;
}

}  // namespace runtime::kolibri
