#include "runtime/bolt/bolt_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "common/bf16.h"
#include "runtime/bolt/bolt_capture.h"
#include "runtime/bolt/bolt_sizes.h"

namespace runtime::bolt {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// pp_handoff.cl's state words (pp_recv): [1] status, [2] the flag read at the failure, [3] the stamp read, [4] the
// sequence expected (runtime/pipeline_engine.cc's).
constexpr uint32_t kStStatus = 1, kStFlag = 2, kStStamp = 3, kStWant = 4;
constexpr uint32_t kStatusTimeout = 1, kStatusStale = 2;
}  // namespace

struct BoltEngine::Stage {
  Stage(l0::Context& c, const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len, BoltAttn a,
        BoltState s, bool profile)
      : ctx(c), buffers(c, d, p, dev, max_len, a, s), queue(c), fence(queue), imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()), prof(profile ? std::make_unique<ProfileEvents>(c) : nullptr) {}
  l0::Context& ctx;
  BoltBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  std::unique_ptr<ProfileEvents> prof;   // before `step`: the list signals its events
  std::optional<CapturedStep> step;
};

BoltEngine::Stage& BoltEngine::st(uint32_t dev) const {
  if (dev >= st_.size())
    throw std::out_of_range("runtime::bolt::BoltEngine: device " + std::to_string(dev) + " of " + std::to_string(st_.size()));
  return *st_[dev];
}

BoltEngine::BoltEngine(std::vector<l0::Context*> devices, loader::BoltLoaded model, uint32_t max_len, bool debug_tap,
                       const PipelineOptions& opt, bool profile)
    : BoltEngine(std::move(devices), std::move(model), max_len, debug_tap, opt, profile, bolt_attn(), bolt_fuse(),
                 bolt_state()) {}

BoltEngine::BoltEngine(std::vector<l0::Context*> devices, loader::BoltLoaded model, uint32_t max_len, bool debug_tap,
                       const PipelineOptions& opt, bool profile, BoltAttn attn, BoltFuse fuse, BoltState state)
    : model_(std::move(model)), max_len_(max_len), opt_(opt), attn_(attn), fuse_(fuse), state_(state), profile_(profile) {
  const model::BoltDesc& d = model_.desc;
  const model::BoltPlacement& p = model_.place;
  model::validate(p, d);
  if (model_.max_len != max_len)
    throw std::runtime_error("runtime::bolt::BoltEngine: the model was loaded with max_len " + std::to_string(model_.max_len) +
                             " but the engine was asked for " + std::to_string(max_len) + " - the plan and the KV are one number");
  if (devices.size() != p.devices || model_.parts.size() != p.devices)
    throw std::invalid_argument("runtime::bolt::BoltEngine: " + std::to_string(devices.size()) + " context(s), " +
                                std::to_string(model_.parts.size()) + " loaded part(s), a " + std::to_string(p.devices) +
                                "-device placement");
  if (p.devices == 2) {
    if (devices[1]->handle() != devices[0]->handle())
      throw std::invalid_argument("runtime::bolt::BoltEngine: device 1 is not a view of device 0's Level Zero context "
                                  "(spec 16 decision 1)");
    if (!devices[0]->can_access_peer(*devices[1]))
      throw std::runtime_error("--pp 2: device 0 (" + devices[0]->name() + ") cannot access device 1's memory "
                               "(zeDeviceCanAccessPeer is false), and both hand-offs write it (docs/10-the-box.md)");
    if (opt.timeout_ms == 0 || (opt.handoff == PpHandoff::Peer && opt.spin_limit == 0))
      throw std::invalid_argument("runtime::bolt::BoltEngine: a hand-off needs a non-zero bound");
  }
  for (uint32_t i = 0; i < p.devices; ++i) {
    st_.push_back(std::make_unique<Stage>(*devices[i], d, p, i, max_len, attn_, state_, profile_));
    taps_.push_back(debug_tap ? std::make_unique<l0::Mem>(*devices[i], l0::MemKind::Device, tap_bytes(d)) : nullptr);
  }
  pending_.assign(p.devices, false);
  StageLink binding;
  if (p.devices == 2) {
    link_ = std::make_unique<PipelineLink>(*devices[0], *devices[1], landing_layout(d, p), opt.handoff);
    binding = link_->binding(opt.spin_limit);
  }
  for (uint32_t i = 0; i < p.devices; ++i)
    st(i).step.emplace(build(st(i).ctx, model_.parts[i], d, st(i).buffers, fuse_, model_.int8_head,
                             link_ ? &binding : nullptr, taps_[i].get(), st(i).prof.get()));
  reset();
}

BoltEngine::~BoltEngine() {
  for (uint32_t i = 0; i < st_.size(); ++i) {   // never free buffers under a running list
    try {
      settle(i);
    } catch (...) {
    }
  }
}

bool BoltEngine::settle(uint32_t dev) {
  if (!pending_[dev]) return true;
  if (st(dev).fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull)) pending_[dev] = false;
  return !pending_[dev];
}

void BoltEngine::reset() {
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error("runtime::bolt::BoltEngine::reset: device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i) st(i).buffers.zero(st(i).imm);
  if (link_) link_->zero(st(0).imm, st(1).imm);
  broken_ = false;
  drop_next_ = false;
}

void BoltEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("pipeline hand-off: " + what + ". The session's state is not valid; reset() before the next step");
}

void BoltEngine::step_once() {
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  Stage& last = st(uint32_t(st_.size()) - 1);
  const size_t pos = last.ctl->pos, n = last.ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error("runtime::bolt::BoltEngine: pos " + std::to_string(pos) + " + n_active " + std::to_string(n) +
                             " exceeds max_len " + std::to_string(max_len_) + " - the KV stops there");
  if (profile_)
    for (auto& s : st_)
      for (l0::Event& e : s->prof->events) e.reset();
  const Clock::time_point t0 = Clock::now();
  if (st_.size() == 1) {
    st(0).queue.execute(st(0).step->list, &st(0).fence);
    st(0).fence.wait();
    if (profile_) accumulate_profile(ms_since(t0) * 1e3);
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
    if (link_->event) link_->event->host_signal();   // release device 1 so its queue drains on garbage, then report
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
      fail("the flag reached " + std::to_string(w[kStWant]) + " but the stamp after the slices reads " +
           std::to_string(w[kStStamp]) + " - the data did not arrive with the flag (a silent hand-off, spec 16 §2)");
    if (status != 0) fail("pp_recv reported status " + std::to_string(status));
  }
  // The token's way back: device 1's argmax advanced pos and wrote the id; device 0 reads both from its own block.
  std::memcpy(st(0).ctl, st(1).ctl, sizeof(Control));
  if (profile_) accumulate_profile(ms_since(t0) * 1e3);
}

void BoltEngine::accumulate_profile(double fence_us) {
  double sum = 0.0;
  for (auto& s : st_) {
    const CapturedStep& cs = *s->step;
    for (size_t i = 0; i < cs.labels.size(); ++i) {
      const double us = s->prof->events[i].duration_us();
      const std::string& lab = cs.labels[i];
      const std::string fam = lab.substr(lab.find(' ') + 1);   // "<kernel> <binary>", the block stripped
      auto& f = prof_us_[fam];
      f.first += us;
      if (prof_replays_ == 0) ++f.second;
      sum += us;
    }
  }
  prof_wall_us_.push_back(fence_us);
  prof_sum_us_.push_back(sum);
  ++prof_replays_;
}

void BoltEngine::clear_profile() {
  prof_us_.clear();
  prof_wall_us_.clear();
  prof_sum_us_.clear();
  prof_replays_ = 0;
}

std::string BoltEngine::profile_table() const {
  if (!profile_) throw std::runtime_error("runtime::bolt::BoltEngine: constructed without profile");
  if (prof_replays_ == 0) return "no profiled replay yet\n";
  const double n = double(prof_replays_);
  std::vector<std::pair<std::string, std::pair<double, size_t>>> rows(prof_us_.begin(), prof_us_.end());
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
  double sum = 0.0, wall = 0.0;
  for (double v : prof_sum_us_) sum += v;
  for (double v : prof_wall_us_) wall += v;
  sum /= n;
  wall /= n;
  const size_t L = launches();
  std::string out = "| launch family (kernel binary) | launches / step | in-situ us / step | us / launch | share of Σ |\n"
                    "|---|---:|---:|---:|---:|\n";
  char buf[512];
  for (const auto& [fam, v] : rows) {
    const double us = v.first / n;
    std::snprintf(buf, sizeof buf, "| %s | %zu | %.2f | %.3f | %.1f %% |\n", fam.c_str(), v.second, us,
                  v.second ? us / double(v.second) : 0.0, sum > 0 ? 100.0 * us / sum : 0.0);
    out += buf;
  }
  std::snprintf(buf, sizeof buf,
                "\nMEASURED over %zu profiled replays (%zu launches a step, %s attention, fuse %s, %s SSM state): Σ in-situ "
                "%.1f us, fence wall %.1f us (submit + wait, profiled: a per-launch flush), gap per launch %.3f us\n",
                prof_replays_, L, bolt_attn_name(attn_), bolt_fuse_name(fuse_).c_str(), bolt_state_name(state_), sum, wall,
                L ? (wall - sum) / double(L) : 0.0);
  out += buf;
  return out;
}

void BoltEngine::set_token(uint32_t id) {
  if (id >= model_.desc.vocab)
    throw std::runtime_error("runtime::bolt::BoltEngine: id " + std::to_string(id) + " is outside the vocabulary (" +
                             std::to_string(model_.desc.vocab) + ")");
  for (auto& s : st_) s->ctl->cur_token[0] = id;
}

void BoltEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (auto& s : st_) s->ctl->n_active = 1;
    set_token(id);
    step_once();
  }
}

std::vector<uint32_t> BoltEngine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  for (auto& s : st_) s->ctl->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = st(uint32_t(st_.size()) - 1).ctl->cur_token[0];
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

uint32_t BoltEngine::pos() const { return st(uint32_t(st_.size()) - 1).ctl->pos; }
uint32_t BoltEngine::pending() const { return st(uint32_t(st_.size()) - 1).ctl->cur_token[0]; }
Control BoltEngine::control(uint32_t dev) const { return *st(dev).ctl; }
size_t BoltEngine::launches(uint32_t dev) const { return st(dev).step->kernel_count; }
size_t BoltEngine::launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < st_.size(); ++i) n += launches(i);
  return n;
}
const CapturedStep& BoltEngine::step(uint32_t dev) const { return *st(dev).step; }
size_t BoltEngine::persistent_bytes(uint32_t dev) const { return st(dev).buffers.persistent_bytes(); }
size_t BoltEngine::scratch_bytes(uint32_t dev) const { return st(dev).buffers.scratch_bytes(); }

std::vector<uint16_t> BoltEngine::read_debug_h() {
  const model::BoltDesc& d = model_.desc;
  if (!taps_[0]) throw std::runtime_error("runtime::bolt::BoltEngine: constructed without debug_tap");
  std::vector<uint16_t> v(size_t(d.layers) * d.hidden);
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const uint32_t first = model_.place.first(i), n = model_.place.count(i);
    st(i).imm.copy(v.data() + size_t(first) * d.hidden, static_cast<const uint8_t*>(taps_[i]->ptr()) + size_t(first) * d.hidden * 2,
                   size_t(n) * d.hidden * 2);
  }
  // Device 0's last block is folded (and tapped) by device 1's first norm (bolt_capture.cc handoff_in).
  if (st_.size() == 2) {
    const uint32_t l = model_.place.split - 1;
    st(1).imm.copy(v.data() + size_t(l) * d.hidden, static_cast<const uint8_t*>(taps_[1]->ptr()) + size_t(l) * d.hidden * 2,
                   size_t(d.hidden) * 2);
  }
  return v;
}

std::vector<uint32_t> BoltEngine::read_routes() {
  const model::BoltDesc& d = model_.desc;
  std::vector<uint32_t> v;
  v.reserve(size_t(d.moe_layers()) * kRouteWords);
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_moe(l)) continue;
    Stage& s = st(model_.place.device_of(l));
    std::vector<uint32_t> row(kRouteWords);
    s.imm.copy(row.data(), static_cast<const uint8_t*>(s.buffers.routes.ptr()) + route_at(l), kRouteWords * 4);
    v.insert(v.end(), row.begin(), row.end());
  }
  return v;
}

std::vector<uint16_t> BoltEngine::read_conv_ring(uint32_t mamba_index) {
  const model::BoltDesc& d = model_.desc;
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_mamba(l) || d.mamba_index(l) != mamba_index) continue;
    Stage& s = st(model_.place.device_of(l));
    std::vector<uint16_t> v(ring_layer_bytes(d) / 2);
    s.imm.copy(v.data(), s.buffers.ring_layer(l), v.size() * 2);
    return v;
  }
  throw std::out_of_range("read_conv_ring: Mamba block " + std::to_string(mamba_index) + " of " +
                          std::to_string(d.mamba_layers()));
}

std::vector<float> BoltEngine::read_ssm_state(uint32_t mamba_index) {
  const model::BoltDesc& d = model_.desc;
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_mamba(l) || d.mamba_index(l) != mamba_index) continue;
    Stage& s = st(model_.place.device_of(l));
    const size_t n = size_t(d.m_heads) * d.m_head_dim * d.m_state;
    std::vector<float> v(n);
    if (state_ == BoltState::Bf16) {
      std::vector<uint16_t> h(n);
      s.imm.copy(h.data(), s.buffers.ssm_layer(l), n * 2);
      for (size_t i = 0; i < n; ++i) v[i] = common::bf16_to_f32(h[i]);
    } else {
      s.imm.copy(v.data(), s.buffers.ssm_layer(l), n * 4);
    }
    return v;
  }
  throw std::out_of_range("read_ssm_state: Mamba block " + std::to_string(mamba_index) + " of " +
                          std::to_string(d.mamba_layers()));
}

std::vector<uint16_t> BoltEngine::read_kv(uint32_t attn_index, uint32_t first, uint32_t count, bool v) {
  const model::BoltDesc& d = model_.desc;
  if (size_t(first) + count > max_len_) throw std::out_of_range("read_kv: past max_len");
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_attention(l) || d.attn_index(l) != attn_index) continue;
    Stage& s = st(model_.place.device_of(l));
    const uint8_t* base = static_cast<const uint8_t*>(v ? s.buffers.v_layer(l) : s.buffers.k_layer(l));
    std::vector<uint16_t> out(size_t(count) * d.kv_n());
    s.imm.copy(out.data(), base + size_t(first) * d.kv_n() * 2, out.size() * 2);
    return out;
  }
  throw std::out_of_range("read_kv: attention block " + std::to_string(attn_index) + " of " + std::to_string(d.attn_layers()));
}

std::vector<uint16_t> BoltEngine::read_residuals(uint32_t dev) {
  Stage& s = st(dev);
  const size_t n = size_t(kM) * model_.desc.hidden;
  std::vector<uint16_t> v(2 * n);
  s.imm.copy(v.data(), s.buffers.h0.ptr(), n * 2);
  s.imm.copy(v.data() + n, s.buffers.h1.ptr(), n * 2);
  return v;
}

std::vector<float> BoltEngine::read_logits() {
  std::vector<float> v(model_.desc.vocab);
  read_logits_into(v.data());
  return v;
}
void BoltEngine::read_logits_into(float* host) {
  Stage& s = st(uint32_t(st_.size()) - 1);
  s.imm.copy(host, s.buffers.logits.ptr(), size_t(model_.desc.vocab) * 4);
}

MemoryComponents BoltEngine::memory_use(uint32_t dev) const {
  const loader::BoltDevicePart& P = model_.parts.at(dev);
  const BoltBuffers& b = st(dev).buffers;
  MemoryComponents c;
  c.model = P.bytes;
  c.kv = b.kv_k.size() + b.kv_v.size();
  c.decode_state = b.control.size() + b.conv_ring.size() + b.ssm.size() + b.scratch_bytes() +
                   (taps_[dev] ? taps_[dev]->size() : 0) + (link_ ? link_->bytes(dev) : 0);
  return c;
}

std::string BoltEngine::memory_line() const {
  std::string s;
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const std::string label = st_.size() > 1 ? "memory, device " + std::to_string(i) : std::string("memory");
    s += (i ? "\n" : "") + format_memory(label.c_str(), memory_use(i), st(i).ctx.memory_bytes());
  }
  return s;
}

}  // namespace runtime::bolt
