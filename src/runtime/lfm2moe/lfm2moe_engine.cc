#include "runtime/lfm2moe/lfm2moe_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "runtime/lfm2moe/lfm2moe_capture.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

namespace runtime::lfm2moe {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// pp_handoff.cl's state words (pp_recv): [1] status, [2] the flag read at the failure, [3] the stamp read, [4] the
// sequence expected (runtime/pipeline_engine.cc's).
constexpr uint32_t kStStatus = 1, kStFlag = 2, kStStamp = 3, kStWant = 4;
constexpr uint32_t kStatusTimeout = 1, kStatusStale = 2;
}  // namespace

struct Lfm2MoeEngine::Stage {
  Stage(l0::Context& c, const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, uint32_t max_len,
        LfmAttn a, bool profile)
      : ctx(c), buffers(c, d, p, dev, max_len, a), queue(c), fence(queue), imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()), prof(profile ? std::make_unique<ProfileEvents>(c) : nullptr) {}
  l0::Context& ctx;
  Lfm2MoeBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  std::unique_ptr<ProfileEvents> prof;   // before `step`: the list signals its events
  std::optional<CapturedStep> step;
};

// Spec 24d: the prefill half's view of a device (b70_lfm2moe_prefill cannot see Stage).
l0::Context& Lfm2MoeEngine::stage_ctx(uint32_t dev) const { return st(dev).ctx; }
Lfm2MoeBuffers& Lfm2MoeEngine::stage_buffers(uint32_t dev) const { return st(dev).buffers; }
Control* Lfm2MoeEngine::stage_control(uint32_t dev) const { return st(dev).ctl; }
l0::CmdList& Lfm2MoeEngine::stage_imm(uint32_t dev) const { return st(dev).imm; }

Lfm2MoeEngine::Stage& Lfm2MoeEngine::st(uint32_t dev) const {
  if (dev >= st_.size())
    throw std::out_of_range("runtime::lfm2moe::Lfm2MoeEngine: device " + std::to_string(dev) + " of " +
                            std::to_string(st_.size()));
  return *st_[dev];
}

Lfm2MoeEngine::Lfm2MoeEngine(std::vector<l0::Context*> devices, loader::LfmLoaded model, uint32_t max_len,
                             bool debug_tap, const PipelineOptions& opt, bool profile)
    : Lfm2MoeEngine(std::move(devices), std::move(model), max_len, debug_tap, opt, profile, lfm_attn(), lfm_fuse()) {}

Lfm2MoeEngine::Lfm2MoeEngine(std::vector<l0::Context*> devices, loader::LfmLoaded model, uint32_t max_len,
                             bool debug_tap, const PipelineOptions& opt, bool profile, LfmAttn attn, LfmFuse fuse)
    : model_(std::move(model)), max_len_(max_len), opt_(opt), attn_(attn), fuse_(fuse), profile_(profile) {
  const model::Lfm2MoeDesc& d = model_.desc;
  const model::LfmPlacement& p = model_.place;
  model::validate(p, d);
  if (model_.max_len != max_len)
    throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine: the model was loaded with max_len " +
                             std::to_string(model_.max_len) + " but the engine was asked for " + std::to_string(max_len) +
                             " - the RoPE table and the KV are one number");
  if (model_.conv_tiles != fuse_.conv)
    throw std::runtime_error(std::string("runtime::lfm2moe::Lfm2MoeEngine: in_proj was loaded ") +
                             (model_.conv_tiles ? "in the conv-tile layout" : "in the plain layout") +
                             " but B70_LFM_FUSE " + (fuse_.conv ? "has" : "has not") +
                             " the arm conv - load with LfmLoadOptions::conv_tiles = the arm");
  if (devices.size() != p.devices || model_.parts.size() != p.devices)
    throw std::invalid_argument("runtime::lfm2moe::Lfm2MoeEngine: " + std::to_string(devices.size()) + " context(s), " +
                                std::to_string(model_.parts.size()) + " loaded part(s), a " +
                                std::to_string(p.devices) + "-device placement");
  if (p.devices == 2) {
    if (devices[1]->handle() != devices[0]->handle())
      throw std::invalid_argument("runtime::lfm2moe::Lfm2MoeEngine: device 1 is not a view of device 0's Level Zero "
                                  "context (spec 16 decision 1)");
    if (!devices[0]->can_access_peer(*devices[1]))
      throw std::runtime_error("--pp 2: device 0 (" + devices[0]->name() + ") cannot access device 1's memory "
                               "(zeDeviceCanAccessPeer is false), and both hand-offs write it (docs/10-the-box.md)");
    if (opt.timeout_ms == 0 || (opt.handoff == PpHandoff::Peer && opt.spin_limit == 0))
      throw std::invalid_argument("runtime::lfm2moe::Lfm2MoeEngine: a hand-off needs a non-zero bound");
  }
  for (uint32_t i = 0; i < p.devices; ++i) {
    st_.push_back(std::make_unique<Stage>(*devices[i], d, p, i, max_len, attn_, profile_));
    taps_.push_back(debug_tap ? std::make_unique<l0::Mem>(*devices[i], l0::MemKind::Device, tap_bytes(d)) : nullptr);
  }
  pending_.assign(p.devices, false);
  StageLink binding;
  if (p.devices == 2) {
    link_ = std::make_unique<PipelineLink>(*devices[0], *devices[1], landing_layout(d, p), opt.handoff);
    binding = link_->binding(opt.spin_limit);
  }
  for (uint32_t i = 0; i < p.devices; ++i)
    st(i).step.emplace(build(st(i).ctx, model_.parts[i], d, st(i).buffers, fuse_, model_.conv_tiles,
                             link_ ? &binding : nullptr, taps_[i].get(), st(i).prof.get()));
  reset();
}

Lfm2MoeEngine::~Lfm2MoeEngine() {
  for (uint32_t i = 0; i < st_.size(); ++i) {   // never free buffers under a running list
    try {
      settle(i);
    } catch (...) {
    }
  }
}

bool Lfm2MoeEngine::settle(uint32_t dev) {
  if (!pending_[dev]) return true;
  if (st(dev).fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull)) pending_[dev] = false;
  return !pending_[dev];
}

void Lfm2MoeEngine::reset() {
  // Spec 24d: a prefill list may still wait on a hand-off that will not come (P4): release it and drain first.
  if (pf_settle_ && !pf_settle_())
    throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine::reset: a prefill list is still running after the "
                             "prefill timeout");
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine::reset: device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i) st(i).buffers.zero(st(i).imm);
  if (link_) link_->zero(st(0).imm, st(1).imm);
  broken_ = false;
  drop_next_ = false;
}

void Lfm2MoeEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("pipeline hand-off: " + what + ". The session's state is not valid; reset() before the next step");
}

void Lfm2MoeEngine::step_once() {
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  Stage& last = st(uint32_t(st_.size()) - 1);
  const size_t pos = last.ctl->pos, n = last.ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine: pos " + std::to_string(pos) + " + n_active " +
                             std::to_string(n) + " exceeds max_len " + std::to_string(max_len_) +
                             " - the KV and the RoPE table stop there");
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
           " (device 0 never published the residual)" +
           (send ? "" : " [device 0's list was not submitted: drop_next_handoff()]"));
    if (status == kStatusStale)
      fail("the flag reached " + std::to_string(w[kStWant]) + " but the stamp after the slices reads " +
           std::to_string(w[kStStamp]) + " - the data did not arrive with the flag (a silent hand-off, spec 16 §2)");
    if (status != 0) fail("pp_recv reported status " + std::to_string(status));
  }
  // The token's way back: device 1's argmax advanced pos and wrote the id; device 0 reads both from its own block.
  std::memcpy(st(0).ctl, st(1).ctl, sizeof(Control));
  if (profile_) accumulate_profile(ms_since(t0) * 1e3);
}

void Lfm2MoeEngine::accumulate_profile(double fence_us) {
  double sum = 0.0;
  for (auto& s : st_) {
    const CapturedStep& cs = *s->step;
    for (size_t i = 0; i < cs.labels.size(); ++i) {
      const double us = s->prof->events[i].duration_us();
      const std::string& lab = cs.labels[i];
      const std::string fam = lab.substr(lab.find(' ') + 1);   // "<kernel> <binary>", the layer stripped
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

void Lfm2MoeEngine::clear_profile() {
  prof_us_.clear();
  prof_wall_us_.clear();
  prof_sum_us_.clear();
  prof_replays_ = 0;
}

std::string Lfm2MoeEngine::profile_table() const {
  if (!profile_) throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine: constructed without profile");
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
  std::string out;
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "| launch family (kernel binary) | launches / step | in-situ us / step | us / launch | share of Σ |\n"
                "|---|---:|---:|---:|---:|\n");
  out += buf;
  for (const auto& [fam, v] : rows) {
    const double us = v.first / n;
    std::snprintf(buf, sizeof buf, "| %s | %zu | %.2f | %.3f | %.1f %% |\n", fam.c_str(), v.second, us,
                  v.second ? us / double(v.second) : 0.0, sum > 0 ? 100.0 * us / sum : 0.0);
    out += buf;
  }
  std::snprintf(buf, sizeof buf,
                "\nMEASURED over %zu profiled replays (%zu launches a step, %s attention, fuse %s): Σ in-situ %.1f us, "
                "fence wall %.1f us (submit + wait, profiled: a per-launch flush), gap per launch %.3f us\n",
                prof_replays_, L, lfm_attn_name(attn_), lfm_fuse_name(fuse_).c_str(), sum, wall,
                L ? (wall - sum) / double(L) : 0.0);
  out += buf;
  return out;
}

void Lfm2MoeEngine::set_token(uint32_t id) {
  if (id >= model_.desc.vocab)
    throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine: id " + std::to_string(id) +
                             " is outside the vocabulary (" + std::to_string(model_.desc.vocab) + ")");
  for (auto& s : st_) s->ctl->cur_token[0] = id;
}

void Lfm2MoeEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (auto& s : st_) s->ctl->n_active = 1;
    set_token(id);
    step_once();
  }
}

std::vector<uint32_t> Lfm2MoeEngine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
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

uint32_t Lfm2MoeEngine::pos() const { return st(uint32_t(st_.size()) - 1).ctl->pos; }
uint32_t Lfm2MoeEngine::pending() const { return st(uint32_t(st_.size()) - 1).ctl->cur_token[0]; }
Control Lfm2MoeEngine::control(uint32_t dev) const { return *st(dev).ctl; }
size_t Lfm2MoeEngine::launches(uint32_t dev) const { return st(dev).step->kernel_count; }
size_t Lfm2MoeEngine::launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < st_.size(); ++i) n += launches(i);
  return n;
}
const CapturedStep& Lfm2MoeEngine::step(uint32_t dev) const { return *st(dev).step; }

// --- spec 24e: prefix-cache snapshots (the run lists are runtime/lfm2moe/lfm2moe_sizes.h's) ---------------------------------
l0::Mem& Lfm2MoeEngine::snap_mem(uint32_t dev, SnapTensor t) const {
  Lfm2MoeBuffers& b = st(dev).buffers;
  switch (t) {
    case SnapTensor::KvK: return b.kv_k;
    case SnapTensor::KvV: return b.kv_v;
    default: return b.conv_ring;
  }
}

void Lfm2MoeEngine::settle_all(const char* what) {
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error(std::string("runtime::lfm2moe::Lfm2MoeEngine::") + what + ": device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
}

size_t Lfm2MoeEngine::state_bytes() const { return state_snapshot_bytes(model_.desc); }
size_t Lfm2MoeEngine::kv_bytes(uint32_t n_pos) const { return kv_snapshot_bytes(model_.desc, n_pos); }

void Lfm2MoeEngine::save_state(void* host) {
  settle_all("save_state");
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : state_runs(model_.desc, model_.place, pos())) {
    if (r.zero)
      std::memset(h, 0, r.bytes);   // a position before 0: never read, zero on the host
    else
      st(r.device).imm.copy(h, static_cast<const uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
}

void Lfm2MoeEngine::load_state(const void* host, uint32_t p) {
  if (p > max_len_)
    throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine::load_state: pos " + std::to_string(p) + " exceeds max_len " +
                             std::to_string(max_len_));
  if (broken_) reset();   // a failed hand-off: the restore below rewrites what the session reads
  settle_all("load_state");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : state_runs(model_.desc, model_.place, p)) {
    st(r.device).imm.copy(static_cast<uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
  for (auto& s : st_) {
    s->ctl->pos = p;
    s->ctl->n_active = 0;
  }
}

void Lfm2MoeEngine::save_kv(uint32_t begin, uint32_t end, void* host) {
  settle_all("save_kv");
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : kv_runs(model_.desc, model_.place, max_len_, begin, end)) {
    st(r.device).imm.copy(h, static_cast<const uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
}

void Lfm2MoeEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  settle_all("load_kv");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : kv_runs(model_.desc, model_.place, max_len_, begin, end)) {
    st(r.device).imm.copy(static_cast<uint8_t*>(snap_mem(r.device, r.tensor).ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
}

std::vector<uint16_t> Lfm2MoeEngine::read_debug_h() {
  const model::Lfm2MoeDesc& d = model_.desc;
  if (!taps_[0]) throw std::runtime_error("runtime::lfm2moe::Lfm2MoeEngine: constructed without debug_tap");
  std::vector<uint16_t> v(size_t(d.layers) * d.hidden);
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const uint32_t first = model_.place.first(i), n = model_.place.count(i);
    st(i).imm.copy(v.data() + size_t(first) * d.hidden,
                   static_cast<const uint8_t*>(taps_[i]->ptr()) + size_t(first) * d.hidden * 2, size_t(n) * d.hidden * 2);
  }
  // A dense layer at the cut is folded (and tapped) on device 1 (lfm2moe_capture.cc handoff_in).
  if (st_.size() == 2 && cut_pending_s(d, model_.place)) {
    const uint32_t l = model_.place.split - 1;
    st(1).imm.copy(v.data() + size_t(l) * d.hidden, static_cast<const uint8_t*>(taps_[1]->ptr()) + size_t(l) * d.hidden * 2,
                   size_t(d.hidden) * 2);
  }
  return v;
}

std::vector<uint32_t> Lfm2MoeEngine::read_routes() {
  const model::Lfm2MoeDesc& d = model_.desc;
  std::vector<uint32_t> v;
  v.reserve(size_t(d.moe_layers()) * kRouteWords);
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (d.is_dense(l)) continue;
    Stage& s = st(model_.place.device_of(l));
    std::vector<uint32_t> row(kRouteWords);
    s.imm.copy(row.data(), static_cast<const uint8_t*>(s.buffers.routes.ptr()) + route_at(l), kRouteWords * 4);
    v.insert(v.end(), row.begin(), row.end());
  }
  return v;
}

std::vector<uint16_t> Lfm2MoeEngine::read_conv_ring(uint32_t conv_index) {
  const model::Lfm2MoeDesc& d = model_.desc;
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (d.is_attention(l) || d.conv_index(l) != conv_index) continue;
    Stage& s = st(model_.place.device_of(l));
    std::vector<uint16_t> v(size_t(loader::lfm::kConvRing) * d.hidden);
    s.imm.copy(v.data(), s.buffers.ring_layer(l), v.size() * 2);
    return v;
  }
  throw std::out_of_range("read_conv_ring: conv layer " + std::to_string(conv_index) + " of " +
                          std::to_string(d.conv_layers()));
}

std::vector<uint16_t> Lfm2MoeEngine::read_kv(uint32_t attn_index, uint32_t first, uint32_t count, bool v) {
  const model::Lfm2MoeDesc& d = model_.desc;
  if (size_t(first) + count > max_len_) throw std::out_of_range("read_kv: past max_len");
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_attention(l) || d.attn_index(l) != attn_index) continue;
    Stage& s = st(model_.place.device_of(l));
    const uint8_t* base = static_cast<const uint8_t*>(v ? s.buffers.v_layer(l) : s.buffers.k_layer(l));
    std::vector<uint16_t> out(size_t(count) * d.kv_n());
    s.imm.copy(out.data(), base + size_t(first) * d.kv_n() * 2, out.size() * 2);
    return out;
  }
  throw std::out_of_range("read_kv: attention layer " + std::to_string(attn_index) + " of " +
                          std::to_string(d.attn_layers()));
}

std::vector<uint16_t> Lfm2MoeEngine::read_residuals(uint32_t dev) {
  Stage& s = st(dev);
  const size_t n = size_t(kM) * model_.desc.hidden;
  std::vector<uint16_t> v(2 * n);
  s.imm.copy(v.data(), s.buffers.h0.ptr(), n * 2);
  s.imm.copy(v.data() + n, s.buffers.h1.ptr(), n * 2);
  return v;
}

std::vector<float> Lfm2MoeEngine::read_logits() {
  std::vector<float> v(model_.desc.vocab);
  read_logits_into(v.data());
  return v;
}
void Lfm2MoeEngine::read_logits_into(float* host) {
  Stage& s = st(uint32_t(st_.size()) - 1);
  s.imm.copy(host, s.buffers.logits.ptr(), size_t(model_.desc.vocab) * 4);
}

MemoryComponents Lfm2MoeEngine::memory_use(uint32_t dev) const {
  const loader::LfmDevicePart& P = model_.parts.at(dev);
  const Lfm2MoeBuffers& b = st(dev).buffers;
  MemoryComponents c;
  c.model = P.bytes + P.rope_bytes;
  c.kv = b.kv_k.size() + b.kv_v.size();
  c.decode_state = b.control.size() + b.conv_ring.size() + b.scratch_bytes() + (taps_[dev] ? taps_[dev]->size() : 0) +
                   (link_ ? link_->bytes(dev) : 0);
  c.prefill_scratch = dev < pf_dev_bytes_.size() ? pf_dev_bytes_[dev] : 0;   // spec 24d: after the first prefill
  return c;
}

std::string Lfm2MoeEngine::memory_line() const {
  std::string s;
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const std::string label = st_.size() > 1 ? "memory, device " + std::to_string(i) : std::string("memory");
    s += (i ? "\n" : "") + format_memory(label.c_str(), memory_use(i), st(i).ctx.memory_bytes());
  }
  return s;
}

}  // namespace runtime::lfm2moe
