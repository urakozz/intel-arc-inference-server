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

l0::Context& BoltEngine::stage_ctx(uint32_t dev) const { return st(dev).ctx; }
BoltBuffers& BoltEngine::stage_buffers(uint32_t dev) const { return st(dev).buffers; }
Control* BoltEngine::stage_control(uint32_t dev) const { return st(dev).ctl; }
l0::CmdList& BoltEngine::stage_imm(uint32_t dev) const { return st(dev).imm; }

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
  // Spec 25d: a prefill list may still wait on a hand-off that will not come (P4): release it and drain first.
  if (pf_settle_ && !pf_settle_())
    throw std::runtime_error("runtime::bolt::BoltEngine::reset: a prefill list is still running after the prefill timeout");
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error("runtime::bolt::BoltEngine::reset: device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i) st(i).buffers.zero(st(i).imm);   // Control::gdn_live 0: slot 0 live
  if (link_) link_->zero(st(0).imm, st(1).imm);
  if (vlink_) vlink_->zero(st(0).imm, st(1).imm);                      // spec 26c
  if (drafter_) drafter_->reset(st(uint32_t(st_.size()) - 1).imm);    // spec 26c: a cold drafter
  verify_k_ = kNoVerify;
  broken_ = false;
  drop_next_ = false;
}

void BoltEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("pipeline hand-off: " + what + ". The session's state is not valid; reset() before the next step");
}

void BoltEngine::step_once() {
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  if (st(0).ctl->gdn_live != 0) canonical_ssm();   // spec 26c: the plain list steps slot 0 in place
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
    if (spec_on()) {   // spec 26c: one plain token through the M = 1 verify list (the slots, the drafter's ring)
      verify(nullptr, 0);
      commit(0, verify_ids()[0]);
      continue;
    }
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
    if (spec_on()) {   // spec 26c: the plain token through the M = 1 verify list
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

// --- spec 25e: prefix-cache snapshots (the run lists are runtime/bolt/bolt_sizes.h's) ---------------------------------------
l0::Mem& BoltEngine::snap_mem(uint32_t dev, int tensor) const {
  BoltBuffers& b = st(dev).buffers;
  switch (static_cast<SnapTensor>(tensor)) {
    case SnapTensor::KvK: return b.kv_k;
    case SnapTensor::KvV: return b.kv_v;
    case SnapTensor::Ssm: return b.ssm;
    default: return b.conv_ring;
  }
}

void BoltEngine::settle_all(const char* what) {
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error(std::string("runtime::bolt::BoltEngine::") + what + ": device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
}

// Spec 26c: with a drafter, its ring and anchor after the target's state (spec 26 §6: +5,242,880 + 16,384 B, 19d's rule).
size_t BoltEngine::state_bytes() const {
  return state_snapshot_bytes(model_.desc, state_) + (drafter_ ? drafter_->ring_bytes() : 0);
}
size_t BoltEngine::kv_bytes(uint32_t n_pos) const { return kv_snapshot_bytes(model_.desc, n_pos); }

void BoltEngine::save_state(void* host) {
  settle_all("save_state");
  canonical_ssm();   // spec 26c: the state runs read slot 0
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : state_runs(model_.desc, model_.place, pos(), state_)) {
    if (r.zero)
      std::memset(h, 0, r.bytes);   // a position before 0: never read, zero on the host
    else
      st(r.device).imm.copy(h, static_cast<const uint8_t*>(snap_mem(r.device, int(r.tensor)).ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
  if (drafter_) drafter_->save_ring(st(uint32_t(st_.size()) - 1).imm, h);   // spec 26c: the ring, then the anchor
}

void BoltEngine::load_state(const void* host, uint32_t p) {
  if (p > max_len_)
    throw std::runtime_error("runtime::bolt::BoltEngine::load_state: pos " + std::to_string(p) + " exceeds max_len " +
                             std::to_string(max_len_));
  if (broken_) reset();   // a failed hand-off: the restore below rewrites what the session reads
  settle_all("load_state");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : state_runs(model_.desc, model_.place, p, state_)) {
    st(r.device).imm.copy(static_cast<uint8_t*>(snap_mem(r.device, int(r.tensor)).ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
  if (drafter_) drafter_->load_ring(st(uint32_t(st_.size()) - 1).imm, h);   // spec 26c
  verify_k_ = kNoVerify;
  for (auto& s : st_) {
    s->ctl->pos = p;
    s->ctl->n_active = 0;
    s->ctl->gdn_live = 0;   // spec 26c: the restored state is slot 0's
  }
}

void BoltEngine::save_kv(uint32_t begin, uint32_t end, void* host) {
  settle_all("save_kv");
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : kv_runs(model_.desc, model_.place, max_len_, begin, end)) {
    st(r.device).imm.copy(h, static_cast<const uint8_t*>(snap_mem(r.device, int(r.tensor)).ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
}

void BoltEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  settle_all("load_kv");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : kv_runs(model_.desc, model_.place, max_len_, begin, end)) {
    st(r.device).imm.copy(static_cast<uint8_t*>(snap_mem(r.device, int(r.tensor)).ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
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
  return read_ssm_slot(mamba_index, st(0).ctl->gdn_live);   // spec 26c: the live slot (0 without the verify lists)
}

std::vector<float> BoltEngine::read_ssm_slot(uint32_t mamba_index, uint32_t slot) {
  const model::BoltDesc& d = model_.desc;
  if (slot >= kSpecSlots || (slot != 0 && !spec_on()))
    throw std::out_of_range("read_ssm_slot: slot " + std::to_string(slot) + (spec_on() ? " of 8" : " without the verify lists"));
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_mamba(l) || d.mamba_index(l) != mamba_index) continue;
    const uint32_t dev = model_.place.device_of(l);
    Stage& s = st(dev);
    const void* src = slot == 0 ? s.buffers.ssm_layer(l)
                                : static_cast<const uint8_t*>(vb_[dev]->slots_layer(l)) +
                                      size_t(slot - 1) * ssm_layer_bytes(d, state_);
    const size_t n = size_t(d.m_heads) * d.m_head_dim * d.m_state;
    std::vector<float> v(n);
    if (state_ == BoltState::Bf16) {
      std::vector<uint16_t> h(n);
      s.imm.copy(h.data(), src, n * 2);
      for (size_t i = 0; i < n; ++i) v[i] = common::bf16_to_f32(h[i]);
    } else {
      s.imm.copy(v.data(), src, n * 4);
    }
    return v;
  }
  throw std::out_of_range("read_ssm_slot: Mamba block " + std::to_string(mamba_index) + " of " +
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
  c.prefill_scratch = dev < pf_dev_bytes_.size() ? pf_dev_bytes_[dev] : 0;   // spec 25d: after the first prefill
  if (dev < vb_.size()) c.decode_state += vb_[dev]->bytes() + (vlink_ ? vlink_->bytes(dev) : 0);   // spec 26c
  if (dev + 1 == st_.size()) {
    if (drafter_) c.drafter = drafter_->bytes();   // spec 26c: its weights + buffers (the replica is the CLI's)
    if (pf_taps_) c.prefill_scratch += pf_taps_->size();
  }
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

// --- spec 26c: speculative decoding -----------------------------------------------------------------------------------------
void BoltEngine::enable_spec(VerifyMoe moe) {
  if (spec_on()) throw std::logic_error("runtime::bolt::BoltEngine::enable_spec: the verify lists are already on");
  if (st_.size() == 2 && opt_.handoff != PpHandoff::Copy)
    throw std::invalid_argument("--spec dflash / lookup on Nemotron 3.5 Bolt crosses the cards by the copy hand-off only "
                                "(19d: the drafter's taps ride on its barrier; peer has no place for them) - drop "
                                "--pipeline-handoff peer");
  settle_all("enable_spec");
  const model::BoltDesc& d = model_.desc;
  for (uint32_t i = 0; i < st_.size(); ++i) {
    vb_.push_back(std::make_unique<BoltVerifyBuffers>(st(i).ctx, d, model_.place, i, max_len_, attn_, state_));
    if (vb_.back()->bytes() != verify_sizes(d, model_.place, i, max_len_, attn_, state_).total())
      throw std::logic_error("runtime::bolt::BoltEngine::enable_spec: device " + std::to_string(i) +
                             "'s verify buffers are not verify_sizes()'");
  }
  if (st_.size() == 2) {
    vlink_ = std::make_unique<PipelineLink>(st(0).ctx, st(1).ctx, verify_landing_layout(d, model_.place), PpHandoff::Copy);
    for (uint32_t i = 0; i < 2; ++i)
      if (vlink_->bytes(i) != verify_link_bytes(d, model_.place, i))
        throw std::logic_error("runtime::bolt::BoltEngine::enable_spec: the verify link is not verify_link_bytes()'");
    vlink_->zero(st(0).imm, st(1).imm);
  }
  vsteps_.resize(st_.size());
  for (auto& dev : vsteps_) dev.resize(kVerifyRows);
  vmoe_ = moe;
  verify_k_ = kNoVerify;
}

void BoltEngine::attach_drafter(TapSink& dr) {
  const model::BoltDesc& d = model_.desc;
  if (!spec_on())
    throw std::logic_error("runtime::bolt::BoltEngine::attach_drafter: enable_spec first (the verify lists write its taps)");
  if (drafter_) throw std::logic_error("runtime::bolt::BoltEngine::attach_drafter: a drafter is already attached");
  if (dr.max_k() == 0 || dr.max_k() + 1 > kVerifyRows)
    throw std::invalid_argument("runtime::bolt::BoltEngine::attach_drafter: the drafter drafts up to " +
                                std::to_string(dr.max_k()) + ", the verify lists hold 8 rows (K 1..7)");
  for (uint32_t t : dr.tap_layers())
    if (t + 1 >= d.layers)
      throw std::invalid_argument("runtime::bolt::BoltEngine::attach_drafter: the drafter taps block " + std::to_string(t) +
                                  " of a " + std::to_string(d.layers) + "-block target (a tap block's output is folded by "
                                  "a later block's norm)");
  if (dr.taps().size() < size_t(kVerifyRows) * dr.tap_layers().size() * d.hidden * 2)
    throw std::invalid_argument("runtime::bolt::BoltEngine::attach_drafter: the drafter's tap buffer is smaller than "
                                "[8][taps][hidden] bf16");
  settle_all("attach_drafter");
  for (auto& dev : vsteps_)
    for (auto& s : dev) s.reset();   // lookup's lists bound no taps: recaptured with them
  drafter_ = &dr;
  drafter_->reset(st(uint32_t(st_.size()) - 1).imm);
}

uint32_t BoltEngine::max_verify_k() const {
  if (!spec_on()) return 0;
  const uint32_t k = drafter_ ? std::min(drafter_->max_k(), kVerifyRows - 1) : kVerifyRows - 1;
  const uint32_t p = pos();
  if (size_t(p) + 1 >= size_t(max_len_)) return 0;
  return std::min<uint32_t>(k, max_len_ - p - 1);
}

void BoltEngine::run_verify(uint32_t M) {
  const model::BoltDesc& d = model_.desc;
  const uint32_t nd = uint32_t(st_.size());
  StageLink binding;
  if (vlink_) binding = vlink_->binding(opt_.spin_limit);
  for (uint32_t i = 0; i < nd; ++i) {
    std::optional<CapturedStep>& s = vsteps_[i].at(M - 1);
    if (s) continue;
    VerifyBind v;
    v.vb = vb_[i].get();
    v.M = M;
    v.union_moe = vmoe_ == VerifyMoe::Union;
    if (drafter_) {
      v.taps = &drafter_->taps();
      v.tap_layers = &drafter_->tap_layers();
    }
    s.emplace(build_verify(st(i).ctx, model_.parts[i], d, st(i).buffers, model_.int8_head, vlink_ ? &binding : nullptr, v));
  }
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

void BoltEngine::verify(const uint32_t* drafts, uint32_t k) {
  if (!spec_on()) throw std::logic_error("runtime::bolt::BoltEngine::verify: no verify lists (enable_spec)");
  if (broken_) throw std::runtime_error("runtime::bolt::BoltEngine::verify: a hand-off failed earlier; reset() first");
  if (k > max_verify_k())
    throw std::invalid_argument("runtime::bolt::BoltEngine::verify: k " + std::to_string(k) + " exceeds " +
                                std::to_string(max_verify_k()) + " at pos " + std::to_string(pos()));
  Control& c0 = *st(0).ctl;
  Control& cl = *st(uint32_t(st_.size()) - 1).ctl;
  const uint32_t n = cl.pos, x = cl.cur_token[0];
  if (size_t(n) + k + 1 > size_t(max_len_))
    throw std::runtime_error("runtime::bolt::BoltEngine::verify: pos " + std::to_string(n) + " + " + std::to_string(k + 1) +
                             " rows exceeds max_len " + std::to_string(max_len_));
  for (uint32_t i = 0; i < k; ++i) {
    if (drafts[i] >= model_.desc.vocab)
      throw std::invalid_argument("runtime::bolt::BoltEngine::verify: draft " + std::to_string(drafts[i]) +
                                  " is outside the vocabulary");
    c0.cur_token[1 + i] = drafts[i];   // the gather reads cur_token[0 .. k] on device 0
  }
  for (auto& s : st_) s->ctl->n_active = k + 1;
  run_verify(k + 1);
  for (uint32_t r = 0; r <= k; ++r) verify_ids_[r] = cl.out_token[r];
  // argmax_stage2 advanced the last device's pos and wrote its last id into cur_token[0]: the host commits, so put both back.
  cl.pos = n;
  cl.cur_token[0] = x;
  verify_k_ = k;
  verify_pos_ = n;
}

void BoltEngine::read_verify_logits(float* host, uint32_t rows) {
  if (verify_k_ == kNoVerify || rows == 0 || rows > verify_k_ + 1)
    throw std::logic_error("runtime::bolt::BoltEngine::read_verify_logits: " + std::to_string(rows) +
                           " rows after no verify / a verify of fewer");
  const uint32_t last = uint32_t(st_.size()) - 1;
  st(last).imm.copy(host, vb_[last]->logits.ptr(), size_t(rows) * model_.desc.vocab * 4);
}

void BoltEngine::commit(uint32_t j, uint32_t next_token) {
  Control& cl = *st(uint32_t(st_.size()) - 1).ctl;
  if (verify_k_ == kNoVerify || cl.pos != verify_pos_)
    throw std::logic_error("runtime::bolt::BoltEngine::commit: no verify at pos " + std::to_string(cl.pos));
  if (j > verify_k_)
    throw std::invalid_argument("runtime::bolt::BoltEngine::commit: j " + std::to_string(j) + " > k " +
                                std::to_string(verify_k_));
  if (next_token >= model_.desc.vocab)
    throw std::invalid_argument("runtime::bolt::BoltEngine::commit: id " + std::to_string(next_token) +
                                " is outside the vocabulary");
  const uint32_t n = cl.pos;
  verify_k_ = kNoVerify;
  if (drafter_) replay_aux(drafter_->projection(n, j + 1));   // rows 0..j of the taps: positions n .. n + j
  const uint32_t live = (cl.gdn_live + j) % kSpecSlots;       // the slot holding the state after row j
  for (auto& s : st_) {
    s->ctl->pos = n + j + 1;
    s->ctl->cur_token[0] = next_token;
    s->ctl->n_active = 1;
    s->ctl->gdn_live = live;
  }
}

void BoltEngine::replay_aux(l0::CmdList& list) {
  Stage& s = st(uint32_t(st_.size()) - 1);
  s.queue.execute(list, &s.fence);
  s.fence.wait();
}

void BoltEngine::canonical_ssm() {
  const uint32_t live = st(0).ctl->gdn_live;
  if (live == 0) return;
  if (!spec_on())
    throw std::logic_error("runtime::bolt::BoltEngine: live SSM slot " + std::to_string(live) + " without the verify lists");
  settle_all("canonical_ssm");
  const model::BoltDesc& d = model_.desc;
  const size_t bytes = ssm_layer_bytes(d, state_);
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_mamba(l)) continue;
    const uint32_t dev = model_.place.device_of(l);
    st(dev).imm.copy(st(dev).buffers.ssm_layer(l),
                     static_cast<const uint8_t*>(vb_[dev]->slots_layer(l)) + size_t(live - 1) * bytes, bytes);
  }
  for (auto& s : st_) s->ctl->gdn_live = 0;
}

size_t BoltEngine::verify_launches(uint32_t M) const {
  size_t n = 0;
  if (M == 0 || M > kVerifyRows) return 0;
  for (const auto& dev : vsteps_)
    if (dev[M - 1]) n += dev[M - 1]->kernel_count;
  return n;
}

uint32_t BoltEngine::ssm_live() const { return st(0).ctl->gdn_live; }
const l0::Mem& BoltEngine::control_mem(uint32_t dev) const { return st(dev).buffers.control; }
const loader::DeviceWeight& BoltEngine::lm_head() const {
  const loader::BoltDevicePart& p = model_.parts.back();
  if (!p.head) throw std::logic_error("runtime::bolt::BoltEngine: the last device holds no lm_head");
  return *p.head;
}

}  // namespace runtime::bolt
