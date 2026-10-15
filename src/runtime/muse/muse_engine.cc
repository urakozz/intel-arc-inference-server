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
        MuseAttn a)
      : ctx(c), buffers(c, d, p, dev, max_len, a), queue(c), fence(queue), imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()) {}
  l0::Context& ctx;
  MuseBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  std::optional<CapturedStep> step;
};

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
  const MusePlan first = plan_muse(d, devs, o.max_len, o.int8_head, decode_reserve(d, o.max_len, devs, attn_), caps,
                                   devs == 2 ? o.split : 0);
  require_fits(first);
  max_len_ = first.max_len;
  const MuseReserve reserve = decode_reserve(d, max_len_, devs, attn_);
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
    st_.push_back(std::make_unique<Stage>(*ctx[i], D, P, i, max_len_, attn_));
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
  for (uint32_t i = 0; i < st_.size(); ++i) st(i).buffers.zero(st(i).imm);
  if (link_) link_->zero(st(0).imm, st(1).imm);
  broken_ = false;
  drop_next_ = false;
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
    step_once();
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
