#include "runtime/pipeline_engine.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "model/qwen35.h"
#include "runtime/pipeline_place.h"
#include "runtime/pipeline_stage.h"

namespace runtime {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// pp_handoff.cl's state words (pp_recv): [0] last sequence consumed, [1] status, [2] the flag
// read at the failure, [3] the stamp read, [4] the sequence expected.
constexpr uint32_t kStStatus = 1, kStFlag = 2, kStStamp = 3, kStWant = 4;
constexpr uint32_t kStatusTimeout = 1, kStatusStale = 2;
}  // namespace

// --- PipelineLink ------------------------------------------------------------------------

PipelineLink::PipelineLink(l0::Context& d0, l0::Context& d1, const model::ModelDesc& d,
                           PpHandoff m)
    : PipelineLink(d0, d1, pp_landing_layout(d), m) {}

PipelineLink::PipelineLink(l0::Context& d0, l0::Context& d1, const PpLandingLayout& l, PpHandoff m)
    : mode(m),
      layout(l),
      landing(d1, l0::MemKind::Device, layout.total, kPpLandingAlign),
      send_seq(d0, l0::MemKind::Device, kPpStateWords * 4),
      recv_state(d1, l0::MemKind::Shared, kPpStateWords * 4),
      event(m == PpHandoff::Copy
                ? std::make_unique<l0::SyncEvent>(d0, std::vector<const l0::Context*>{&d0, &d1})
                : nullptr) {}

StageLink PipelineLink::binding(uint32_t spin_limit) const {
  StageLink s;
  s.mode = mode;
  uint8_t* base = landing.as<uint8_t>();
  s.land_resid = base;
  s.land_sumsq = base + layout.sumsq_off;
  s.flag = base + layout.flag_off;
  s.resid_cap = layout.resid_bytes;
  s.sumsq_cap = layout.sumsq_bytes;
  s.send_seq = send_seq.ptr();
  s.recv_state = recv_state.ptr();
  s.spin_limit = spin_limit;
  s.event = event ? event->handle() : nullptr;
  return s;
}

size_t PipelineLink::bytes(uint32_t device) const {
  return device == 0 ? send_seq.size() : landing.size() + recv_state.size();
}

void PipelineLink::zero(l0::CmdList& imm0, l0::CmdList& imm1) {
  imm1.fill(landing.ptr(), 0u, landing.size());
  imm0.fill(send_seq.ptr(), 0u, send_seq.size());
  imm1.fill(recv_state.ptr(), 0u, recv_state.size());
  if (event) event->host_reset();
}

// --- one device's half: runtime/pipeline_stage.h -------------------------------------

PipelineEngine::Stage& PipelineEngine::st(uint32_t dev) const {
  if (dev >= kPpDevices)
    throw std::out_of_range("runtime::PipelineEngine: device " + std::to_string(dev) +
                            " (two are built)");
  return *stages_[dev];
}

PipelineEngine::PipelineEngine(l0::Context& d0, l0::Context& d1,
                               std::vector<loader::LoadedModel> stages, uint32_t max_len,
                               const PipelineOptions& opt, KvCache kv)
    : max_len_(max_len), opt_(opt), kv_(kv) {
  if (stages.size() != kPpDevices)
    throw std::invalid_argument("runtime::PipelineEngine: " + std::to_string(stages.size()) +
                                " stage models (spec 16b builds two)");
  if (d1.handle() != d0.handle())
    throw std::invalid_argument("runtime::PipelineEngine: device 1 is not a view of device 0's "
                                "Level Zero context (spec 16 decision 1)");
  // P4: both hand-offs write device 1's memory from device 0 (the copy path's peer copy,
  // pp_send's stores). Refused here, by name, rather than as a fault on the first token.
  if (!d0.can_access_peer(d1))
    throw std::runtime_error(
        "--pp 2: device 0 (" + d0.name() + ") cannot access device 1's memory "
        "(zeDeviceCanAccessPeer is false), and both hand-offs write it. Peer access needs "
        "the P2P-capable kernel and both cards under one root complex (docs/10-the-box.md)");
  const model::ModelDesc* desc = stages[0].desc;
  if (!desc || stages[1].desc != desc)
    throw std::invalid_argument("runtime::PipelineEngine: the two stage models are not one model");
  for (const loader::LoadedModel& m : stages) {
    if (m.max_len != max_len)
      throw std::runtime_error("runtime::PipelineEngine: a stage model was loaded with max_len " +
                               std::to_string(m.max_len) + " but the engine was asked for " +
                               std::to_string(max_len));
    if (m.mtp || m.draft_vocab)
      throw std::runtime_error("runtime::PipelineEngine: MTP under pipeline parallel is spec 16d");
  }
  if (opt.timeout_ms == 0 || (opt.handoff == PpHandoff::Peer && opt.spin_limit == 0))
    throw std::invalid_argument("runtime::PipelineEngine: a hand-off needs a non-zero bound");
  const uint32_t split = static_cast<uint32_t>(stages[0].layer_small.size());
  const std::array<PpStage, kPpDevices> ranges = pp_stages(*desc, split);
  if (stages[1].layer_small.size() != ranges[1].layers())
    throw std::invalid_argument("runtime::PipelineEngine: stage 1 holds " +
                                std::to_string(stages[1].layer_small.size()) +
                                " layers, not the " + std::to_string(ranges[1].layers()) +
                                " of layers [" + std::to_string(split) + ", " +
                                std::to_string(desc->layers) + ")");
  stages_[0] = std::make_unique<Stage>(d0, std::move(stages[0]), max_len, ranges[0], kv);
  stages_[1] = std::make_unique<Stage>(d1, std::move(stages[1]), max_len, ranges[1], kv);
  link_ = std::make_unique<PipelineLink>(d0, d1, *desc, opt.handoff);
  const StageLink binding = link_->binding(opt.spin_limit);
  for (uint32_t i = 0; i < kPpDevices; ++i) {
    Stage& s = st(i);
    s.step.emplace(build_stage(s.ctx, s.model, s.buffers, s.range, binding));
  }
  reset();
}

PipelineEngine::~PipelineEngine() {
  // Never free buffers under a running list (a failed step's survivor): bounded, and a
  // destructor does not throw. The prefill half first (spec 16c): its lists use the stages'
  // state.
  if (pf_) {
    try {
      pf_->settle(opt_.prefill_timeout_ms);
    } catch (...) {
    }
    pf_.reset();
  }
  for (uint32_t d = 0; d < kPpDevices; ++d) {
    try {
      if (stages_[d]) settle(d);
    } catch (...) {
    }
  }
}

void PipelineEngine::reset() {
  // A failed step may have left a list running (a peer spin past the fence bound): it must
  // finish before its buffers are rewritten under it.
  for (uint32_t d = 0; d < kPpDevices; ++d)
    if (!settle(d))
      throw std::runtime_error("runtime::PipelineEngine::reset: device " + std::to_string(d) +
                               "'s last step is still running after " +
                               std::to_string(opt_.timeout_ms) + " ms");
  // Spec 16c: and a failed prefill's lists (released from any hand-off wait first).
  if (pf_) {
    if (!pf_->settle(opt_.prefill_timeout_ms))
      throw std::runtime_error("runtime::PipelineEngine::reset: a prefill list is still running after " +
                               std::to_string(opt_.prefill_timeout_ms) + " ms");
    pf_->zero();
  }
  snap_gdn_ = {};
  snap_conv_ = {};
  st(0).persist.zero(st(0).imm);
  st(1).persist.zero(st(1).imm);
  link_->zero(st(0).imm, st(1).imm);
  broken_ = false;
  drop_next_ = false;
}

const PpStage& PipelineEngine::stage(uint32_t dev) const { return st(dev).range; }
const CapturedStep& PipelineEngine::step(uint32_t dev) const { return *st(dev).step; }
DecodeBuffers& PipelineEngine::buffers(uint32_t dev) { return st(dev).buffers; }
const Control& PipelineEngine::control(uint32_t dev) const { return *st(dev).ctl; }
const loader::LoadedModel& PipelineEngine::model(uint32_t dev) const { return st(dev).model; }
l0::Context& PipelineEngine::context(uint32_t dev) const { return st(dev).ctx; }
uint32_t PipelineEngine::pos() const { return st(1).ctl->pos; }

bool PipelineEngine::settle(uint32_t dev) {
  if (!pending_[dev]) return true;
  if (st(dev).fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull)) pending_[dev] = false;
  return !pending_[dev];
}

void PipelineEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("pipeline hand-off: " + what +
                           ". The session's state is not valid; reset() before the next step");
}

void PipelineEngine::step_once() {
  if (broken_)
    throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  Stage& s0 = st(0);
  Stage& s1 = st(1);
  const size_t pos = s1.ctl->pos, n = s1.ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error(
        "runtime::PipelineEngine: pos " + std::to_string(pos) + " + n_active " +
        std::to_string(n) + " exceeds max_len " + std::to_string(max_len_) +
        " - the KV cache and the RoPE table stop there. Start a new session with reset(), or"
        " load with a larger max_len.");
  if (link_->event) link_->event->host_reset();
  const bool send = !drop_next_;
  drop_next_ = false;
  if (send) {
    s0.queue.execute(s0.step->list, &s0.fence);
    pending_[0] = true;
  }
  s1.queue.execute(s1.step->list, &s1.fence);
  pending_[1] = true;
  if (!settle(1)) {
    // Device 1 is still waiting (copy: on the event device 0 never signalled). Release it so
    // its queue drains on garbage instead of holding the device, then report.
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
  if (!settle(0))
    fail("device 0's step did not finish within " + std::to_string(opt_.timeout_ms) + " ms");
  if (link_->mode == PpHandoff::Peer) {
    const uint32_t* w = link_->recv_state.as<uint32_t>();
    const uint32_t status = w[kStStatus];
    if (status == kStatusTimeout)
      fail("device 1's pp_recv gave up after " + std::to_string(opt_.spin_limit) +
           " loads of the flag: it read " + std::to_string(w[kStFlag]) + ", expected " +
           std::to_string(w[kStWant]) + " (device 0 never published the residual)" +
           (send ? "" : " [device 0's list was not submitted: drop_next_handoff()]"));
    if (status == kStatusStale)
      fail("the flag reached " + std::to_string(w[kStWant]) + " but the stamp after the norm sums"
           " reads " + std::to_string(w[kStStamp]) +
           " - the data did not arrive with the flag (a silent hand-off, spec 16 §2)");
    if (status != 0) fail("pp_recv reported status " + std::to_string(status));
  }
  // The token's way back (Review Focus 2): device 1's argmax advanced pos and wrote the id;
  // device 0 reads both from its own block on the next step.
  std::memcpy(s0.ctl, s1.ctl, sizeof(Control));
}

void PipelineEngine::set_token(uint32_t id) {
  if (id >= model::Qwen35::kVocab)
    throw std::runtime_error("runtime::PipelineEngine::set_token: id " + std::to_string(id) +
                             " is outside the vocabulary");
  st(0).ctl->cur_token[0] = id;
  st(1).ctl->cur_token[0] = id;
}

void PipelineEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (uint32_t d = 0; d < kPpDevices; ++d) st(d).ctl->n_active = 1;
    set_token(id);
    step_once();
  }
}

std::vector<uint32_t> PipelineEngine::generate(uint32_t n,
                                               const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  for (uint32_t d = 0; d < kPpDevices; ++d) st(d).ctl->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = st(1).ctl->cur_token[0];   // == device 0's after the mirror
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

// --- spec 7 snapshots, in the single-card layout --------------------------------------------

size_t PipelineEngine::state_bytes() const {
  size_t t = 0;
  for (uint32_t d = 0; d < kPpDevices; ++d)
    t += st(d).persist.gdn_state.size() + st(d).persist.conv_ring.size();
  return t;
}

size_t PipelineEngine::kv_bytes(uint32_t n_pos) const {
  const model::ModelDesc& d = *st(0).model.desc;
  return 2 * st(0).persist.kv_lay.pos_bytes() * size_t(d.fa_layers) * n_pos;
}

void PipelineEngine::save_state(void* host) const {
  // Inside a mid-prompt block hook (spec 16c) the block end's shadows, else the live state;
  // the same bytes in the same order either way.
  const auto gdn = [&](uint32_t d) -> const l0::Mem& {
    return snap_gdn_[d] ? *snap_gdn_[d] : st(d).persist.gdn_state;
  };
  const auto conv = [&](uint32_t d) -> const l0::Mem& {
    return snap_conv_[d] ? *snap_conv_[d] : st(d).persist.conv_ring;
  };
  auto* h = static_cast<uint8_t*>(host);
  for (uint32_t d = 0; d < kPpDevices; ++d) {   // every GDN layer's state, device 0's first
    st(d).imm.copy(h, gdn(d).ptr(), st(d).persist.gdn_state.size());
    h += st(d).persist.gdn_state.size();
  }
  for (uint32_t d = 0; d < kPpDevices; ++d) {   // then every conv ring
    st(d).imm.copy(h, conv(d).ptr(), st(d).persist.conv_ring.size());
    h += st(d).persist.conv_ring.size();
  }
}

void PipelineEngine::load_state(const void* host, uint32_t pos) {
  if (pos > max_len_)
    throw std::runtime_error("runtime::PipelineEngine::load_state: pos " + std::to_string(pos) +
                             " exceeds max_len " + std::to_string(max_len_));
  const auto* h = static_cast<const uint8_t*>(host);
  for (uint32_t d = 0; d < kPpDevices; ++d) {
    st(d).imm.copy(st(d).persist.gdn_state.ptr(), h, st(d).persist.gdn_state.size());
    h += st(d).persist.gdn_state.size();
  }
  for (uint32_t d = 0; d < kPpDevices; ++d) {
    st(d).imm.copy(st(d).persist.conv_ring.ptr(), h, st(d).persist.conv_ring.size());
    h += st(d).persist.conv_ring.size();
  }
  // Review Focus 3: pos moves on both devices, together. cur_token is not restored (spec 7
  // §3.3 step 4: ingest at least one id after a restore), as on one card.
  for (uint32_t d = 0; d < kPpDevices; ++d) {
    st(d).ctl->pos = pos;
    st(d).ctl->n_active = 0;
  }
}

namespace {
struct KvRun {
  uint32_t dev;
  uint8_t* ptr;
  size_t bytes;
};

// Engine::kv_runs' order over two devices: per tensor (K, V) every layer's rows - device 0's
// layers, then device 1's, which is the model's FA order - then at int8 every layer's scales
// the same way.
std::vector<KvRun> pp_kv_runs(const std::array<const PersistentBuffers*, kPpDevices>& p,
                              uint32_t begin, uint32_t end) {
  std::vector<KvRun> runs;
  const size_t n = end - begin;
  for (int kv = 0; kv < 2; ++kv) {
    for (uint32_t d = 0; d < kPpDevices; ++d) {
      const KvLayout& L = p[d]->kv_lay;
      uint8_t* base = (kv == 0 ? p[d]->kv_k : p[d]->kv_v).as<uint8_t>();
      for (uint32_t l = 0; l < L.layers; ++l)
        runs.push_back({d, base + L.rows_offset(l) + size_t(begin) * L.row_bytes(), n * L.row_bytes()});
    }
    for (uint32_t d = 0; d < kPpDevices; ++d) {
      const KvLayout& L = p[d]->kv_lay;
      if (L.scale_row_bytes() == 0) continue;
      uint8_t* base = (kv == 0 ? p[d]->kv_k : p[d]->kv_v).as<uint8_t>();
      for (uint32_t l = 0; l < L.layers; ++l)
        runs.push_back({d, base + L.scales_offset(l) + size_t(begin) * L.scale_row_bytes(),
                        n * L.scale_row_bytes()});
    }
  }
  return runs;
}
}  // namespace

void PipelineEngine::save_kv(uint32_t begin, uint32_t end, void* host) const {
  if (begin > end || end > max_len_)
    throw std::runtime_error("runtime::PipelineEngine::save_kv: range [" + std::to_string(begin) +
                             ", " + std::to_string(end) + ") is not within [0, max_len " +
                             std::to_string(max_len_) + ")");
  if (begin == end) return;
  std::vector<KvRun> runs;
  runs = pp_kv_runs({&st(0).persist, &st(1).persist}, begin, end);
  auto* h = static_cast<uint8_t*>(host);
  for (const KvRun& r : runs) {
    st(r.dev).imm.copy(h, r.ptr, r.bytes);
    h += r.bytes;
  }
}

void PipelineEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  if (begin > end || end > max_len_)
    throw std::runtime_error("runtime::PipelineEngine::load_kv: range [" + std::to_string(begin) +
                             ", " + std::to_string(end) + ") is not within [0, max_len " +
                             std::to_string(max_len_) + ")");
  if (begin == end) return;
  std::vector<KvRun> runs;
  runs = pp_kv_runs({&st(0).persist, &st(1).persist}, begin, end);
  const auto* h = static_cast<const uint8_t*>(host);
  for (const KvRun& r : runs) {
    st(r.dev).imm.copy(r.ptr, h, r.bytes);
    h += r.bytes;
  }
}

std::vector<float> PipelineEngine::read_logits() const {
  std::vector<float> out(model::Qwen35::kVocab);
  st(1).imm.copy(out.data(), st(1).buffers.logits.ptr(), out.size() * sizeof(float));
  return out;
}

MemoryComponents PipelineEngine::memory_use(uint32_t dev) const {
  const Stage& s = st(dev);
  MemoryComponents c;
  c.model = model_device_bytes(s.model);
  c.kv = s.persist.kv_k.size() + s.persist.kv_v.size();
  c.decode_state = s.persist.bytes() - c.kv + s.scratch.bytes() + link_->bytes(dev);
  if (pf_) {   // spec 16c: the prefill half, once prepared
    const MemoryComponents p = pf_->memory(dev);
    c.prefill_scratch += p.prefill_scratch;
    c.int8 += p.int8;
    c.decode_state += p.decode_state;
  }
  return c;
}

std::string PipelineEngine::memory_line(uint32_t dev) const {
  const std::string label = "memory, device " + std::to_string(dev);
  return format_memory(label.c_str(), memory_use(dev), st(dev).ctx.memory_bytes());
}

}  // namespace runtime
