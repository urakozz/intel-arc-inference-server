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
  for (const loader::LoadedModel& m : stages)
    if (m.max_len != max_len)
      throw std::runtime_error("runtime::PipelineEngine: a stage model was loaded with max_len " +
                               std::to_string(m.max_len) + " but the engine was asked for " +
                               std::to_string(max_len));
  // Spec 16d: the MTP head (and a draft vocabulary) is device 1's, with lm_head
  // (runtime::place_stages); device 0's model holds neither.
  if (stages[0].mtp || stages[0].draft_vocab || (stages[1].draft_vocab && !stages[1].mtp))
    throw std::invalid_argument("runtime::PipelineEngine: the MTP head and a draft vocabulary "
                                "belong to device 1's stage (runtime::place_stages)");
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
  // Spec 16d: the head's buffers on device 1, device 0's verify slots, both devices' verify
  // lists at M = 1..kSlots and device 1's draft lists. Every list pair shares the one link:
  // only one pair runs at a time, the copy event is host-reset before each, and the peer
  // counters advance in submission order whichever pair it is.
  if (st(1).model.mtp) {
    Stage& s0 = st(0);
    Stage& s1 = st(1);
    s1.mtp = std::make_unique<MtpBuffers>(
        d1, max_len, *desc, s1.model.draft_vocab ? s1.model.draft_vocab->size() : 0u, kv);
    s0.gdn_spec = std::make_unique<l0::Mem>(d0, l0::MemKind::Device, pp_gdn_spec_bytes(*desc, s0.range.gdn));
    s0.imm.fill(s0.gdn_spec->ptr(), 0u, s0.gdn_spec->size());
    for (uint32_t M = 1; M <= MtpBuffers::kSlots; ++M)
      for (uint32_t i = 0; i < kPpDevices; ++i) {
        Stage& s = st(i);
        s.verify.push_back(build_stage_verify(s.ctx, s.model, s.buffers, s.range, binding, M,
                                              s.mtp.get(), *s.spec()));
      }
    for (uint32_t i = 0; i < kMaxDraft; ++i)
      s1.draft.push_back(build_stage_draft(s1.ctx, s1.model, s1.buffers, s1.range, *s1.mtp, i));
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
  snap_hh_ = nullptr;
  st(0).persist.zero(st(0).imm);
  st(1).persist.zero(st(1).imm);
  // Spec 16d: and the MTP state (Engine::reset's mtp_->zero), on both devices.
  if (st(1).mtp) st(1).mtp->zero(st(1).imm);
  if (st(0).gdn_spec) st(0).imm.fill(st(0).gdn_spec->ptr(), 0u, st(0).gdn_spec->size());
  verify_k_ = kNoVerify;
  draft_ids_.clear();
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

void PipelineEngine::step_once() { run_pair(st(0).step->list, st(1).step->list, "step"); }

void PipelineEngine::run_pair(l0::CmdList& list0, l0::CmdList& list1, const char* what) {
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
    s0.queue.execute(list0, &s0.fence);
    pending_[0] = true;
  }
  s1.queue.execute(list1, &s1.fence);
  pending_[1] = true;
  if (!settle(1)) {
    // Device 1 is still waiting (copy: on the event device 0 never signalled). Release it so
    // its queue drains on garbage instead of holding the device, then report.
    if (link_->event) link_->event->host_signal();
    const bool drained = settle(1);
    settle(0);
    fail(std::string("device 1's ") + what + " did not finish within " + std::to_string(opt_.timeout_ms) +
         " ms - device 0's residual never arrived" +
         (link_->event ? std::string(" (the cross-device event was not signalled; device 1 ") +
                             (drained ? "released" : "did not drain either") + ")"
                       : std::string(drained ? "" : " (device 1 did not drain)")) +
         (send ? "" : " [device 0's list was not submitted: drop_next_handoff()]"));
  }
  if (!settle(0))
    fail(std::string("device 0's ") + what + " did not finish within " + std::to_string(opt_.timeout_ms) + " ms");
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
    if (mtp())
      mtp_step1();   // spec 16d: Engine's rule - the head's KV stays filled
    else
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
    if (mtp())
      mtp_step1();
    else
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
  if (mtp()) t += size_t(st(0).model.desc->hidden) * 2;   // spec 16d: hh row 0, last
  return t;
}

size_t PipelineEngine::kv_bytes(uint32_t n_pos) const {
  const model::ModelDesc& d = *st(0).model.desc;
  return 2 * st(0).persist.kv_lay.pos_bytes() * (size_t(d.fa_layers) + (mtp() ? 1 : 0)) * n_pos;
}

size_t PipelineEngine::slot_bytes() const {
  const model::ModelDesc& d = *st(0).model.desc;
  return size_t(d.gdn_layers) * d.gdn_v_heads * model::Qwen35::kGdnHeadDim * model::Qwen35::kGdnHeadDim * 4;
}

// The GDN state save_state reads on device `dev`: inside a mid-prompt hook the block end's
// shadow, else the live slot - slot 0 is the stage's gdn_state, slot s > 0 the stage's
// slice at (s - 1) whole-model slots into its verify slots (spec 16d; gdn_live is 0 unless
// a commit moved it).
const l0::Mem& PipelineEngine::live_gdn(uint32_t dev) const {
  if (snap_gdn_[dev]) return *snap_gdn_[dev];
  return st(dev).persist.gdn_state;
}

void PipelineEngine::save_state(void* host) const {
  // Inside a mid-prompt block hook (spec 16c) the block end's shadows, else the live state;
  // the same bytes in the same order either way.
  const auto gdn = [&](uint32_t d) -> const void* {
    const uint32_t live = mtp() && !snap_gdn_[d] ? st(d).ctl->gdn_live % MtpBuffers::kSlots : 0;
    if (live == 0) return live_gdn(d).ptr();
    return st(d).spec()->as<uint8_t>() + size_t(live - 1) * slot_bytes();
  };
  const auto conv = [&](uint32_t d) -> const l0::Mem& {
    return snap_conv_[d] ? *snap_conv_[d] : st(d).persist.conv_ring;
  };
  auto* h = static_cast<uint8_t*>(host);
  for (uint32_t d = 0; d < kPpDevices; ++d) {   // every GDN layer's state, device 0's first
    st(d).imm.copy(h, gdn(d), st(d).persist.gdn_state.size());
    h += st(d).persist.gdn_state.size();
  }
  for (uint32_t d = 0; d < kPpDevices; ++d) {   // then every conv ring
    st(d).imm.copy(h, conv(d).ptr(), st(d).persist.conv_ring.size());
    h += st(d).persist.conv_ring.size();
  }
  if (mtp())   // spec 16d: then the head's input hidden h_{pos-1} (Engine's MTP layout)
    st(1).imm.copy(h, snap_hh_ ? snap_hh_->ptr() : st(1).mtp->hh.ptr(),
                   size_t(st(1).model.desc->hidden) * 2);
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
  if (mtp()) {   // spec 16d: the head's hidden row; the state went into slot 0 on both
    st(1).imm.copy(st(1).mtp->hh.ptr(), h, size_t(st(1).model.desc->hidden) * 2);
    for (uint32_t d = 0; d < kPpDevices; ++d) st(d).ctl->gdn_live = 0;
    verify_k_ = kNoVerify;
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
}  // namespace

// Engine::kv_runs' order over two devices (runtime::pp_kv_runs, device-free and host-tested):
// per tensor (K, V) every layer's rows - device 0's layers, then device 1's, which is the
// model's FA order - then (spec 16d, MTP on) the head's layer, then at int8 every layer's
// scales the same way. Resolved here to device pointers.
static std::vector<KvRun> resolve_kv_runs(const std::array<const PersistentBuffers*, kPpDevices>& p,
                                          const MtpBuffers* head, uint32_t begin, uint32_t end) {
  std::vector<KvRun> runs;
  for (const PpKvRun& r :
       pp_kv_runs({p[0]->kv_lay, p[1]->kv_lay}, head ? &head->kv_lay : nullptr, begin, end)) {
    const l0::Mem& m = r.head ? (r.tensor == 0 ? head->kv_k : head->kv_v)
                              : (r.tensor == 0 ? p[r.device]->kv_k : p[r.device]->kv_v);
    runs.push_back({r.device, m.as<uint8_t>() + r.offset, r.bytes});
  }
  return runs;
}

void PipelineEngine::save_kv(uint32_t begin, uint32_t end, void* host) const {
  if (begin > end || end > max_len_)
    throw std::runtime_error("runtime::PipelineEngine::save_kv: range [" + std::to_string(begin) +
                             ", " + std::to_string(end) + ") is not within [0, max_len " +
                             std::to_string(max_len_) + ")");
  if (begin == end) return;
  const std::vector<KvRun> runs =
      resolve_kv_runs({&st(0).persist, &st(1).persist}, st(1).mtp.get(), begin, end);
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
  const std::vector<KvRun> runs =
      resolve_kv_runs({&st(0).persist, &st(1).persist}, st(1).mtp.get(), begin, end);
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

// --- spec 16d: MTP across the split ---------------------------------------------------------

bool PipelineEngine::mtp() const { return stages_[1] && stages_[1]->mtp != nullptr; }

void PipelineEngine::require_mtp(const char* what) const {
  if (!mtp())
    throw std::runtime_error(std::string("runtime::PipelineEngine::") + what +
                             ": MTP is off - load the model with its head (loader::load(..., mtp"
                             " = true)) before place_stages");
}

uint32_t PipelineEngine::draft_vocab() const {
  return mtp() && st(1).model.draft_vocab ? st(1).model.draft_vocab->size() : 0;
}

MtpBuffers* PipelineEngine::mtp_buffers() { return mtp() ? st(1).mtp.get() : nullptr; }

const CapturedStep& PipelineEngine::verify_step(uint32_t dev, uint32_t M) const {
  require_mtp("verify_step");
  return st(dev).verify.at(M - 1);
}

const CapturedStep& PipelineEngine::draft_step(uint32_t i) const {
  require_mtp("draft_step");
  return st(1).draft.at(i);
}

uint32_t PipelineEngine::max_verify_k() const {
  const uint32_t pos = st(1).ctl->pos;
  if (!mtp() || pos == 0 || pos + 1 >= max_len_) return 0;
  return std::min<uint32_t>(kMaxDraft, max_len_ - pos - 1);
}

const uint32_t* PipelineEngine::verify_ids() const { return st(1).ctl->out_token; }

uint32_t PipelineEngine::pending() const { return st(1).ctl->cur_token[0]; }

void PipelineEngine::set_draft_input(uint32_t i, uint32_t id) {
  if (i >= kMaxDraft || id >= model::Qwen35::kVocab)
    throw std::runtime_error("runtime::PipelineEngine::set_draft_input: draft " + std::to_string(i) +
                             " id " + std::to_string(id) + " is outside [0, " +
                             std::to_string(kMaxDraft) + ") x the vocabulary");
  for (uint32_t d = 0; d < kPpDevices; ++d) st(d).ctl->cur_token[1 + i] = id;
}

void PipelineEngine::read_logits_into(float* host, uint32_t rows) const {
  if (rows == 0 || rows > MtpBuffers::kSlots)
    throw std::runtime_error("runtime::PipelineEngine::read_logits_into: rows " + std::to_string(rows));
  st(1).imm.copy(host, st(1).buffers.logits.ptr(), size_t(rows) * model::Qwen35::kVocab * sizeof(float));
}

void PipelineEngine::read_draft_logits_into(float* host, uint32_t i) const {
  require_mtp("read_draft_logits_into");
  if (i >= kMaxDraft) throw std::runtime_error("runtime::PipelineEngine::read_draft_logits_into: row " + std::to_string(i));
  st(1).imm.copy(host, st(1).mtp->logits.as<float>() + size_t(i) * model::Qwen35::kVocab,
                 model::Qwen35::kVocab * sizeof(float));
}

float* PipelineEngine::host_rows(size_t floats) {
  if (!host_rows_ || host_rows_->size() < floats * sizeof(float))
    host_rows_ = std::make_unique<l0::Mem>(st(1).ctx, l0::MemKind::Host, floats * sizeof(float));
  return host_rows_->as<float>();
}

void PipelineEngine::run_one(uint32_t dev, l0::CmdList& list, const char* what) {
  if (broken_)
    throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  Stage& s = st(dev);
  s.queue.execute(list, &s.fence);
  pending_[dev] = true;
  if (!settle(dev))
    fail("device " + std::to_string(dev) + "'s " + what + " did not finish within " +
         std::to_string(opt_.timeout_ms) + " ms");
}

void PipelineEngine::draft(uint32_t k, const std::function<uint32_t(uint32_t)>& pick) {
  require_mtp("draft");
  const uint32_t pos = st(1).ctl->pos;
  if (k == 0 || k > kMaxDraft || k > max_verify_k())
    throw std::runtime_error("runtime::PipelineEngine::draft: k " + std::to_string(k) +
                             " is outside [1, max_verify_k() = " + std::to_string(max_verify_k()) +
                             "] at pos " + std::to_string(pos));
  Stage& s1 = st(1);
  Control* hctl = s1.mtp->hctl.as<Control>();
  const size_t row = size_t(s1.model.desc->hidden) * 2;
  // Engine::draft: the chain starts at h_{pos-1} (hh row 0), draft i at position pos - 1 + i.
  s1.imm.copy(s1.mtp->dh.ptr(), s1.mtp->hh.ptr(), row);
  hctl->pos = pos - 1;
  hctl->n_active = 1;
  hctl->cur_token[0] = s1.ctl->cur_token[0];
  draft_ids_.clear();
  for (uint32_t i = 0; i < k; ++i) {
    run_one(1, s1.draft[i].list, "draft list");
    uint32_t d = hctl->out_token[0];
    if (pick) {
      d = pick(i);
      if (d >= model::Qwen35::kVocab)
        throw std::runtime_error("runtime::PipelineEngine::draft: picked id " + std::to_string(d) +
                                 " is outside the vocabulary");
      hctl->cur_token[0] = d;   // the next draft list's input
    }
    draft_ids_.push_back(d);
    for (uint32_t dev = 0; dev < kPpDevices; ++dev) st(dev).ctl->cur_token[1 + i] = d;
  }
}

void PipelineEngine::verify(uint32_t k) {
  require_mtp("verify");
  const uint32_t pos = st(1).ctl->pos;
  if (k > kMaxDraft || pos == 0 || size_t(pos) + k + 1 > max_len_)
    throw std::runtime_error("runtime::PipelineEngine::verify: k " + std::to_string(k) + " at pos " +
                             std::to_string(pos) + " (max_verify_k() = " +
                             std::to_string(max_verify_k()) + "; pos must be >= 1 and pos + k"
                             " + 1 <= max_len " + std::to_string(max_len_) + ")");
  for (uint32_t r = 0; r <= k; ++r)
    if (st(1).ctl->cur_token[r] >= model::Qwen35::kVocab || st(0).ctl->cur_token[r] != st(1).ctl->cur_token[r])
      throw std::runtime_error("runtime::PipelineEngine::verify: row " + std::to_string(r) + "'s id " +
                               std::to_string(st(1).ctl->cur_token[r]) +
                               " is outside the vocabulary or not on both devices");
  for (uint32_t d = 0; d < kPpDevices; ++d) st(d).ctl->n_active = k + 1;
  // The head's KV fill of the same rows at pos - 1 .. pos + k - 1 (build_verify's tail).
  Control* hctl = st(1).mtp->hctl.as<Control>();
  hctl->pos = pos - 1;
  hctl->n_active = k + 1;
  for (uint32_t r = 0; r <= k; ++r) hctl->cur_token[r] = st(1).ctl->cur_token[r];
  verify_pos_ = pos;
  verify_k_ = k;
  run_pair(st(0).verify[k].list, st(1).verify[k].list, "verify");
  // Device 1's argmax advanced pos by k + 1 and wrote the rows' ids: device 0 mirrors it, as
  // after a step; commit() sets the real pos on both.
  std::memcpy(st(0).ctl, st(1).ctl, sizeof(Control));
}

void PipelineEngine::commit(uint32_t j, uint32_t next_token) {
  require_mtp("commit");
  if (verify_k_ == kNoVerify || j > verify_k_)
    throw std::runtime_error("runtime::PipelineEngine::commit: j " + std::to_string(j) +
                             (verify_k_ == kNoVerify ? " with no verify() pending"
                                                     : " exceeds the last verify's k " +
                                                           std::to_string(verify_k_)));
  if (next_token >= model::Qwen35::kVocab)
    throw std::runtime_error("runtime::PipelineEngine::commit: next_token " + std::to_string(next_token) +
                             " is outside the vocabulary");
  Stage& s1 = st(1);
  const size_t row = size_t(s1.model.desc->hidden) * 2;
  // Row j's GDN state is slot (live + j) % kSlots on BOTH devices (each holds its own layers'
  // slots); row j's hidden (hh row 1 + j, device 1) is the next iteration's h_{pos-1}.
  s1.imm.copy(s1.mtp->hh.ptr(), s1.mtp->hh.as<uint8_t>() + (1 + size_t(j)) * row, row);
  for (uint32_t d = 0; d < kPpDevices; ++d) {
    Control* c = st(d).ctl;
    c->gdn_live = (c->gdn_live + j) % MtpBuffers::kSlots;
    c->pos = verify_pos_ + j + 1;
    c->n_active = 1;
    c->cur_token[0] = next_token;
  }
  verify_k_ = kNoVerify;
}

void PipelineEngine::mtp_step1() {
  if (st(1).ctl->pos == 0) {
    // No h_{-1}: the plain lists (slot 0, live 0 at pos 0), then the step's final-normed
    // hidden into hh row 0 - Engine::mtp_step1.
    if (st(1).ctl->gdn_live != 0) mtp_normalise_live();
    for (uint32_t d = 0; d < kPpDevices; ++d) st(d).ctl->n_active = 1;
    step_once();
    st(1).imm.copy(st(1).mtp->hh.ptr(), st(1).buffers.x.ptr(), size_t(st(1).model.desc->hidden) * 2);
    return;
  }
  verify(0);
  commit(0, st(1).ctl->out_token[0]);
}

void PipelineEngine::mtp_normalise_live() {
  if (!mtp()) return;
  const uint32_t live = st(1).ctl->gdn_live % MtpBuffers::kSlots;
  if (live != 0)
    for (uint32_t d = 0; d < kPpDevices; ++d) {
      Stage& s = st(d);
      s.imm.copy(s.persist.gdn_state.ptr(), s.spec()->as<uint8_t>() + size_t(live - 1) * slot_bytes(),
                 s.persist.gdn_state.size());
    }
  for (uint32_t d = 0; d < kPpDevices; ++d) st(d).ctl->gdn_live = 0;
}

void PipelineEngine::mtp_before_prefill() {
  if (!mtp()) return;
  mtp_normalise_live();
  verify_k_ = kNoVerify;
}

MemoryComponents PipelineEngine::memory_use(uint32_t dev) const {
  const Stage& s = st(dev);
  MemoryComponents c;
  c.model = model_device_bytes(s.model);
  c.kv = s.persist.kv_k.size() + s.persist.kv_v.size();
  c.decode_state = s.persist.bytes() - c.kv + s.scratch.bytes() + link_->bytes(dev);
  // Spec 16d: the MTP state (device 1's MtpBuffers, device 0's verify slots) is decode
  // state, the draft vocabulary its own term - Engine::memory_use's split, per device.
  c.decode_state += (s.mtp ? s.mtp->bytes() : 0) + (s.gdn_spec ? s.gdn_spec->size() : 0);
  if (s.model.draft_vocab) {
    c.draft_vocab = s.model.draft_vocab->bytes();
    c.model -= c.draft_vocab;
  }
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
