#include "runtime/qwen4exp/qwen4exp_engine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "kernels/kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "l0/kernel.h"
#include "l0/module.h"
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_ple_hash.h"
#include "runtime/qwen4exp/qwen4exp_capture.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace runtime::qwen4exp {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// pp_handoff.cl's state words (pp_recv): [1] status, [2] the flag read at the failure, [3] the stamp read, [4]
// the sequence expected (runtime/pipeline_engine.cc's).
constexpr uint32_t kStStatus = 1, kStFlag = 2, kStStamp = 3, kStWant = 4;
constexpr uint32_t kStatusTimeout = 1, kStatusStale = 2;
}  // namespace

struct Qwen4ExpEngine::Stage {
  Stage(l0::Context& c, const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, uint32_t max_len,
        Q4Attn a, bool mtp)
      : ctx(c), buffers(c, d, p, dev, max_len, a, mtp), queue(c), fence(queue), imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()) {}
  l0::Context& ctx;
  Qwen4ExpBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  std::optional<CapturedStep> step, inj_step;
  // Spec 21e, with the MTP head: the verify lists at M = 1..4 and (the last device) the draft steps.
  std::vector<CapturedStep> verify, drafts;
};

Qwen4ExpEngine::Stage& Qwen4ExpEngine::st(uint32_t dev) const {
  if (dev >= st_.size())
    throw std::out_of_range("runtime::qwen4exp::Qwen4ExpEngine: device " + std::to_string(dev) + " of " +
                            std::to_string(st_.size()));
  return *st_[dev];
}

Qwen4ExpEngine::Qwen4ExpEngine(std::vector<l0::Context*> devices, loader::Q4LoadedModel model, uint32_t max_len,
                               bool debug_tap, const PipelineOptions& opt)
    : model_(std::move(model)), max_len_(max_len), opt_(opt), attn_(q4_attn()) {
  const model::Qwen4ExpDesc& d = model_.desc;
  const model::Q4Placement& p = model_.placement;
  model::validate(p, d);
  if (model_.max_len != max_len)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: the model was loaded with max_len " +
                             std::to_string(model_.max_len) + " but the engine was asked for " + std::to_string(max_len) +
                             " - the RoPE table and the KV are one number");
  if (devices.size() != p.devices || model_.parts.size() != p.devices)
    throw std::invalid_argument("runtime::qwen4exp::Qwen4ExpEngine: " + std::to_string(devices.size()) + " context(s), " +
                                std::to_string(model_.parts.size()) + " loaded part(s), a " + std::to_string(p.devices) +
                                "-device placement");
  if (p.devices == 2) {
    if (devices[1]->handle() != devices[0]->handle())
      throw std::invalid_argument("runtime::qwen4exp::Qwen4ExpEngine: device 1 is not a view of device 0's Level Zero "
                                  "context (spec 16 decision 1)");
    if (!devices[0]->can_access_peer(*devices[1]))
      throw std::runtime_error("--pp 2: device 0 (" + devices[0]->name() + ") cannot access device 1's memory "
                               "(zeDeviceCanAccessPeer is false), and both hand-offs write it. Peer access needs the "
                               "P2P-capable kernel and both cards under one root complex (docs/10-the-box.md)");
    if (opt.timeout_ms == 0 || (opt.handoff == PpHandoff::Peer && opt.spin_limit == 0))
      throw std::invalid_argument("runtime::qwen4exp::Qwen4ExpEngine: a hand-off needs a non-zero bound");
  }
  // Spec 21e: the MTP head (the loader's, on the last device) - read the switches before anything is allocated.
  const bool mtp = model_.mtp;
  if (mtp) {
    if (!model_.parts.back().mtp || !model_.parts.back().mtp_fc || !model_.parts.back().mtp_mixer)
      throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: the model says mtp but the last device holds no head");
    mtp_norm_ = runtime::qwen4exp::mtp_norm();
    mtp_select_ = runtime::qwen4exp::mtp_select();
  }
  for (uint32_t i = 0; i < p.devices; ++i) {
    st_.push_back(std::make_unique<Stage>(*devices[i], d, p, i, max_len, attn_, mtp));
    taps_.push_back(debug_tap ? std::make_unique<l0::Mem>(*devices[i], l0::MemKind::Device, tap_bytes(d)) : nullptr);
    inj_rows_.push_back(nullptr);
  }
  pending_.assign(p.devices, false);
  if (p.devices == 2) {
    link_ = std::make_unique<PipelineLink>(*devices[0], *devices[1], landing_layout_rows(d, mtp ? kVerifyRows : kM),
                                           opt.handoff);
    binding_ = link_->binding(opt.spin_limit);
  }
  ple_init();
  for (uint32_t i = 0; i < p.devices; ++i)
    st(i).step.emplace(build(st(i).ctx, model_, model_.parts[i], st(i).buffers, link_ ? &binding_ : nullptr,
                             taps_[i].get(), nullptr));
  if (mtp) {
    const uint32_t last = p.devices - 1;
    mtp_ = std::make_unique<Qwen4ExpMtpBuffers>(st(last).ctx, d, max_len);
    mtp_bind_.bufs = mtp_.get();
    mtp_bind_.embed = model_.parts[0].embed->ptr();   // device 0's table: the head reads it over peer access
    mtp_bind_.single = mtp_norm_ == MtpNorm::Single;
    for (uint32_t i = 0; i < p.devices; ++i)
      for (uint32_t M = 1; M <= kVerifyRows; ++M) {
        ListSpec ls;
        ls.kind = ListKind::Verify;
        ls.M = M;
        st(i).verify.push_back(build(st(i).ctx, model_, model_.parts[i], st(i).buffers, link_ ? &binding_ : nullptr,
                                     nullptr, nullptr, ls, &mtp_bind_));
      }
    for (uint32_t s = 0; s < kMaxDraft; ++s) {
      ListSpec ls;
      ls.kind = ListKind::Draft;
      ls.draft = s;
      ls.select = s == 0 || mtp_select_ == MtpSelect::Fresh;
      st(last).drafts.push_back(build(st(last).ctx, model_, model_.parts[last], st(last).buffers, nullptr, nullptr,
                                      nullptr, ls, &mtp_bind_));
    }
  }
  reset();
}

Qwen4ExpEngine::~Qwen4ExpEngine() {
  for (uint32_t i = 0; i < st_.size(); ++i) {   // never free buffers under a running list
    try {
      settle(i);
    } catch (...) {
    }
  }
}

// The PLE device: the hash constants into ple_consts (written once; scratch is never zeroed), then the table's
// host pages read back by the device through the pointer table and held to the loader's page words.
void Qwen4ExpEngine::ple_init() {
  const model::Qwen4ExpDesc& d = model_.desc;
  const uint32_t dev = model_.placement.device_of(d.ple_layer);
  if (dev != model_.ple_device)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: the PLE table was pinned for device " +
                             std::to_string(model_.ple_device) + " but the PLE layer lives on device " + std::to_string(dev));
  Stage& s = st(dev);
  const std::array<uint64_t, 3> mult = loader::q4_ple_multipliers(d.vocab, d.ngram, 0, d.ple_seed);
  const std::vector<uint64_t> sizes = loader::q4_ple_primes(d.ple_base, d.ple_heads, 0);
  const std::vector<uint64_t> offs = loader::q4_ple_offsets(sizes);
  std::vector<uint64_t> consts(kPleConsts, 0);
  for (uint32_t i = 0; i < 3; ++i) consts[i] = mult[i];
  for (uint32_t h = 0; h < d.ple_heads; ++h) {
    consts[3 + h] = sizes[h];
    consts[3 + d.ple_heads + h] = offs[h];
  }
  s.imm.copy(s.buffers.ple_consts.ptr(), consts.data(), consts.size() * 8);
  // q4_ple_check: one lane per 2 MiB page of every range (q[0..15], s[0..15]), the loader's order
  const loader::Q4PleTable& t = model_.ple;
  const uint32_t ranges = uint32_t(t.q.size() + t.s.size());
  std::vector<uint32_t> first(ranges + 1, 0);
  for (uint32_t r = 0; r < ranges; ++r) {
    const l0::Mem& m = r < t.q.size() ? *t.q[r] : *t.s[r - t.q.size()];
    uint64_t pages = 0;
    while (pages * loader::kQ4PlePage + 8 <= m.size()) ++pages;
    first[r + 1] = first[r] + uint32_t(pages);
  }
  const uint32_t pages = first[ranges];
  if (pages != t.page_words.size())
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: the PLE table records " +
                             std::to_string(t.page_words.size()) + " page words for " + std::to_string(pages) + " pages");
  l0::Mem fp(s.ctx, l0::MemKind::Device, first.size() * 4), out(s.ctx, l0::MemKind::Device, size_t(pages) * 8 + 8);
  s.imm.copy(fp.ptr(), first.data(), first.size() * 4);
  {
    const bool bf16s = t.scale == loader::Q4PleScale::Bf16;
    l0::Module mod(s.ctx, kernels::path(kernels::qwen4exp::ple_check_variant(bf16s)));
    l0::Kernel k = mod.kernel("q4_ple_check");
    k.group_size(kernels::qwen4exp::kPleCheckWg);
    k.arg_ptr(0, t.ptrs->ptr());
    k.arg_ptr(1, fp.ptr());
    k.arg_ptr(2, out.ptr());
    k.arg<uint32_t>(3, pages);
    l0::CmdList list = l0::CmdList::regular(s.ctx);
    list.launch(k, (pages + kernels::qwen4exp::kPleCheckWg - 1) / kernels::qwen4exp::kPleCheckWg);
    list.close();
    s.queue.execute(list, &s.fence);
    s.fence.wait();
  }
  std::vector<uint64_t> got(pages);
  s.imm.copy(got.data(), out.ptr(), size_t(pages) * 8);
  for (uint32_t g = 0; g < pages; ++g)
    if (got[g] != t.page_words[g]) {
      uint32_t r = 0;
      while (r + 1 < ranges && first[r + 1] <= g) ++r;
      throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: the device reads PLE host range " + std::to_string(r) +
                               " (" + (r < t.q.size() ? "q" : "s") + "[" + std::to_string(r % t.q.size()) + "]) page " +
                               std::to_string(g - first[r]) + " as a different word than the loader wrote - an aliased "
                               "or unmapped pinned page (the xe hazard, spec 22 §1)");
    }
  ple_pages_checked_ = pages;
}

bool Qwen4ExpEngine::settle(uint32_t dev) {
  if (!pending_[dev]) return true;
  if (st(dev).fence.wait_for(uint64_t(opt_.timeout_ms) * 1000000ull)) pending_[dev] = false;
  return !pending_[dev];
}

l0::Context& Qwen4ExpEngine::stage_ctx(uint32_t dev) const { return st(dev).ctx; }
Qwen4ExpBuffers& Qwen4ExpEngine::stage_buffers(uint32_t dev) const { return st(dev).buffers; }
Control* Qwen4ExpEngine::stage_control(uint32_t dev) const { return st(dev).ctl; }
l0::CmdList& Qwen4ExpEngine::stage_imm(uint32_t dev) const { return st(dev).imm; }

void Qwen4ExpEngine::reset() {
  // Spec 21d: a prefill list waiting on a hand-off that will not come is released first (pf_settle_ is set by
  // prepare_prefill; b70_qwen4exp_prefill's) - the buffers below are not touched under a running list.
  if (pf_settle_ && !pf_settle_())
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::reset: a prefill list is still running after " +
                             std::to_string(opt_.prefill_timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::reset: device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i) st(i).buffers.zero(st(i).imm);
  if (link_) link_->zero(st(0).imm, st(1).imm);
  if (mtp_) mtp_->zero(st(uint32_t(st_.size()) - 1).imm);   // spec 21e: the head's Control, KV, keys, tail, R rows
  verify_k_ = kNoVerify;
  draft_ids_.clear();
  broken_ = false;
  drop_next_ = false;
}

void Qwen4ExpEngine::fail(const std::string& what) {
  broken_ = true;
  throw std::runtime_error("pipeline hand-off: " + what + ". The session's state is not valid; reset() before the next step");
}

CapturedStep& Qwen4ExpEngine::active(uint32_t dev) const {
  return injected_ ? *st(dev).inj_step : *st(dev).step;
}

void Qwen4ExpEngine::step_once() {
  std::vector<CapturedStep*> lists;
  for (uint32_t i = 0; i < st_.size(); ++i) lists.push_back(&active(i));
  submit(lists);
}

void Qwen4ExpEngine::submit(const std::vector<CapturedStep*>& lists) {
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  Stage& last = st(uint32_t(st_.size()) - 1);
  const size_t pos = last.ctl->pos, n = last.ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: pos " + std::to_string(pos) + " + n_active " +
                             std::to_string(n) + " exceeds max_len " + std::to_string(max_len_) +
                             " - the KV, the indexer keys and the RoPE table stop there");
  if (st_.size() == 1) {
    st(0).queue.execute(lists[0]->list, &st(0).fence);
    st(0).fence.wait();
    return;
  }
  if (link_->event) link_->event->host_reset();
  const bool send = !drop_next_;
  drop_next_ = false;
  if (send) {
    st(0).queue.execute(lists[0]->list, &st(0).fence);
    pending_[0] = true;
  }
  st(1).queue.execute(lists[1]->list, &st(1).fence);
  pending_[1] = true;
  if (!settle(1)) {
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
           " (device 0 never published H)" + (send ? "" : " [device 0's list was not submitted: drop_next_handoff()]"));
    if (status == kStatusStale)
      fail("the flag reached " + std::to_string(w[kStWant]) + " but the stamp reads " + std::to_string(w[kStStamp]) +
           " - the data did not arrive with the flag (a silent hand-off, spec 16 §2)");
    if (status != 0) fail("pp_recv reported status " + std::to_string(status));
  }
  // The token's way back: device 1's argmax advanced pos and wrote the id; device 0 reads both from its own block.
  std::memcpy(st(0).ctl, st(1).ctl, sizeof(Control));
}

void Qwen4ExpEngine::set_token(uint32_t id) {
  if (id >= model_.desc.vocab)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: id " + std::to_string(id) +
                             " is outside the vocabulary (" + std::to_string(model_.desc.vocab) + ")");
  for (auto& s : st_) s->ctl->cur_token[0] = id;
}

void Qwen4ExpEngine::ingest(const std::vector<uint32_t>& ids) {
  for (uint32_t id : ids) {
    for (auto& s : st_) s->ctl->n_active = 1;
    set_token(id);
    if (mtp_)
      mtp_step1();
    else
      step_once();
  }
}

std::vector<uint32_t> Qwen4ExpEngine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  for (auto& s : st_) s->ctl->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = st(uint32_t(st_.size()) - 1).ctl->cur_token[0];
    const Clock::time_point f0 = Clock::now();
    if (mtp_)
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

uint32_t Qwen4ExpEngine::pending() const { return st(uint32_t(st_.size()) - 1).ctl->cur_token[0]; }
uint32_t Qwen4ExpEngine::pos() const { return st(uint32_t(st_.size()) - 1).ctl->pos; }
Control Qwen4ExpEngine::control(uint32_t dev) const { return *st(dev).ctl; }

// --- the injected run ------------------------------------------------------------------------------------------
void Qwen4ExpEngine::build_injected() {
  for (uint32_t i = 0; i < st_.size(); ++i) {
    if (st(i).inj_step) continue;
    if (!inj_rows_[i]) {
      inj_rows_[i] = std::make_unique<l0::Mem>(st(i).ctx, l0::MemKind::Host, injected_bytes(model_.desc));
      std::memset(inj_rows_[i]->ptr(), 0, inj_rows_[i]->size());
    }
    st(i).inj_step.emplace(build(st(i).ctx, model_, model_.parts[i], st(i).buffers, link_ ? &binding_ : nullptr,
                                 taps_[i].get(), inj_rows_[i].get()));
  }
}
void Qwen4ExpEngine::set_injected_selection(bool on) {
  if (on && mtp_)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: the injected selection is a decode-list debug run; with "
                             "the MTP head every step runs the verify lists - load without the head");
  if (on) build_injected();
  injected_ = on;
}
uint32_t* Qwen4ExpEngine::injected_list(uint32_t qsa_index) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (qsa_index >= d.qsa_before(d.layers))
    throw std::out_of_range("injected_list: QSA layer " + std::to_string(qsa_index) + " of " +
                            std::to_string(d.qsa_before(d.layers)));
  build_injected();
  uint32_t layer = 0, seen = 0;
  for (; layer < d.layers; ++layer)
    if (d.is_qsa(layer) && seen++ == qsa_index) break;
  const uint32_t dev = model_.placement.device_of(layer);
  return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(inj_rows_[dev]->ptr()) + list_at(qsa_index));
}
size_t Qwen4ExpEngine::injected_launches() const {
  return decode_launches(model_.desc, model_.placement, attn_, opt_.handoff, true);
}

size_t Qwen4ExpEngine::launches(uint32_t dev) const { return active(dev).kernel_count; }
size_t Qwen4ExpEngine::launches() const {
  size_t n = 0;
  for (uint32_t i = 0; i < st_.size(); ++i) n += launches(i);
  return n;
}
const CapturedStep& Qwen4ExpEngine::step(uint32_t dev) const { return *st(dev).step; }

// --- read-backs ----------------------------------------------------------------------------------------------------
std::vector<uint16_t> Qwen4ExpEngine::read_debug_H() {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (!taps_[0]) throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: constructed without debug_tap");
  std::vector<uint16_t> v(size_t(d.layers) * d.hc_n());
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const uint32_t first = model_.placement.first(i), n = model_.placement.count(i);
    st(i).imm.copy(v.data() + size_t(first) * d.hc_n(),
                   static_cast<const uint8_t*>(taps_[i]->ptr()) + size_t(first) * d.hc_n() * 2, size_t(n) * d.hc_n() * 2);
  }
  return v;
}

std::vector<uint32_t> Qwen4ExpEngine::read_routes() {
  const model::Qwen4ExpDesc& d = model_.desc;
  std::vector<uint32_t> v(size_t(d.layers) * kRouteWords);
  for (uint32_t l = 0; l < d.layers; ++l) {   // row 0 of each layer's rows (the scratch's stride: 1, or 4 with the head)
    Stage& s = st(model_.placement.device_of(l));
    s.imm.copy(v.data() + size_t(l) * kRouteWords,
               static_cast<const uint8_t*>(s.buffers.routes.ptr()) + s.buffers.route_off(l), size_t(kRouteWords) * 4);
  }
  return v;
}

std::vector<uint32_t> Qwen4ExpEngine::read_selection(uint32_t layer) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (layer >= d.layers || !d.is_qsa(layer)) throw std::out_of_range("read_selection: layer " + std::to_string(layer) + " is not a QSA layer");
  const uint32_t qi = d.qsa_before(layer), dev = model_.placement.device_of(layer);
  std::vector<uint32_t> row(kListRow);
  if (injected_)
    std::memcpy(row.data(), static_cast<const uint8_t*>(inj_rows_[dev]->ptr()) + list_at(qi), size_t(kListRow) * 4);
  else
    st(dev).imm.copy(row.data(), static_cast<const uint8_t*>(st(dev).buffers.list.ptr()) + st(dev).buffers.list_off(qi),
                     size_t(kListRow) * 4);
  const uint32_t c = std::min(row[kCountWord], kListMax);
  std::vector<uint32_t> out(row.begin(), row.begin() + c);
  out.push_back(row[kCountWord]);
  return out;
}

std::vector<float> Qwen4ExpEngine::read_selection_diag() {
  const model::Qwen4ExpDesc& d = model_.desc;
  const uint32_t nq = d.qsa_before(d.layers);
  std::vector<float> v(size_t(nq) * 2, 0.0f);
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_qsa(l)) continue;
    const uint32_t qi = d.qsa_before(l), dev = model_.placement.device_of(l);
    st(dev).imm.copy(v.data() + size_t(qi) * 2,
                     static_cast<const uint8_t*>(st(dev).buffers.diag.ptr()) + st(dev).buffers.diag_off(qi), 8);
  }
  return v;
}

std::vector<uint64_t> Qwen4ExpEngine::read_ple_ids() {
  Stage& s = st(model_.placement.device_of(model_.desc.ple_layer));
  std::vector<uint64_t> v(model_.desc.ple_heads);
  s.imm.copy(v.data(), s.buffers.ple_ids.ptr(), v.size() * 8);
  return v;
}

std::vector<float> Qwen4ExpEngine::read_logits() {
  Stage& s = st(uint32_t(st_.size()) - 1);
  std::vector<float> v(model_.desc.vocab);
  s.imm.copy(v.data(), s.buffers.logits.ptr(), v.size() * 4);
  return v;
}
void Qwen4ExpEngine::read_logits_into(float* host) {
  Stage& s = st(uint32_t(st_.size()) - 1);
  s.imm.copy(host, s.buffers.logits.ptr(), size_t(model_.desc.vocab) * 4);
}

std::vector<uint16_t> Qwen4ExpEngine::read_kv(uint32_t layer, uint32_t first, uint32_t count, bool v) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (layer >= d.layers || !d.is_qsa(layer)) throw std::out_of_range("read_kv: layer " + std::to_string(layer));
  if (size_t(first) + count > max_len_) throw std::out_of_range("read_kv: past max_len");
  Stage& s = st(model_.placement.device_of(layer));
  const uint8_t* base = static_cast<const uint8_t*>(v ? s.buffers.v_layer(layer) : s.buffers.k_layer(layer));
  std::vector<uint16_t> out(size_t(count) * d.kv_n());
  s.imm.copy(out.data(), base + size_t(first) * d.kv_n() * 2, out.size() * 2);
  return out;
}

std::vector<uint16_t> Qwen4ExpEngine::read_idx_keys(uint32_t layer, uint32_t first_block, uint32_t count) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (layer >= d.layers || !d.is_qsa(layer)) throw std::out_of_range("read_idx_keys: layer " + std::to_string(layer));
  if (size_t(first_block) + count > max_len_ / d.idx_compress) throw std::out_of_range("read_idx_keys: past max_len");
  Stage& s = st(model_.placement.device_of(layer));
  std::vector<uint16_t> out(size_t(count) * d.idx_dim);
  s.imm.copy(out.data(), static_cast<const uint8_t*>(s.buffers.idx_keys_layer(layer)) + size_t(first_block) * d.idx_dim * 2,
             out.size() * 2);
  return out;
}

std::vector<uint8_t> Qwen4ExpEngine::read_layer_state(uint32_t layer) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (layer >= d.layers) throw std::out_of_range("read_layer_state: layer " + std::to_string(layer));
  Stage& s = st(model_.placement.device_of(layer));
  const Qwen4ExpBuffers& b = s.buffers;
  std::vector<uint8_t> out;
  const auto append = [&](const void* p, size_t n) {
    const size_t at0 = out.size();
    out.resize(at0 + n);
    s.imm.copy(out.data() + at0, p, n);
  };
  if (d.is_qsa(layer)) {
    append(b.k_layer(layer), size_t(max_len_) * d.kv_n() * 2 * 2);   // K then V
    append(b.idx_keys_layer(layer), size_t(max_len_ / d.idx_compress) * d.idx_dim * 2);
    append(b.tail_layer(layer), size_t(kIdxTail) * d.idx_dim * 2);
  } else {
    append(b.gdn_state_layer(layer), gdn_state_bytes_per_layer(d));
    append(b.conv_ring_layer(layer), conv_ring_bytes_per_layer(d));
  }
  if (layer == d.ple_layer) append(b.ple.ptr(), ple_state_bytes(d));
  return out;
}

MemoryComponents Qwen4ExpEngine::memory_use(uint32_t dev) const {
  const loader::Q4DevicePart& P = model_.parts.at(dev);
  const Qwen4ExpBuffers& b = st(dev).buffers;
  MemoryComponents c;
  c.model = P.bytes + P.rope_bytes;
  c.kv = b.kv.size() + b.idx_keys.size() + b.idx_tail.size();
  c.decode_state = b.control.size() + b.gdn_state.size() + b.conv_ring.size() + b.ple.size() + b.scratch_bytes() +
                   (taps_[dev] ? taps_[dev]->size() : 0) + (link_ ? link_->bytes(dev) : 0) + b.mtp_bytes();
  if (mtp_ && dev + 1 == st_.size()) {   // spec 21e: the head's KV and keys in kv, the rest in decode state (plan())
    const size_t kv = mtp_->kv.size() + mtp_->idx_keys.size() + mtp_->idx_tail.size();
    c.kv += kv;
    c.decode_state += mtp_->bytes() - kv;
  }
  c.prefill_scratch = dev < pf_dev_bytes_.size() ? pf_dev_bytes_[dev] : 0;   // spec 21d: after prepare_prefill
  return c;
}

// --- spec 21e: prefix-cache snapshots ------------------------------------------------------------------------------
void Qwen4ExpEngine::settle_all(const char* what) {
  if (pf_settle_ && !pf_settle_())
    throw std::runtime_error(std::string("runtime::qwen4exp::Qwen4ExpEngine::") + what + ": a prefill list is still "
                             "running after " + std::to_string(opt_.prefill_timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error(std::string("runtime::qwen4exp::Qwen4ExpEngine::") + what + ": device " +
                               std::to_string(i) + "'s last step is still running after " +
                               std::to_string(opt_.timeout_ms) + " ms");
}

uint8_t* Qwen4ExpEngine::snap_ptr(const SnapRun& r) const {
  const Qwen4ExpBuffers& b = st(r.device).buffers;
  const l0::Mem* m = nullptr;
  const bool head = r.tensor == SnapTensor::MtpKv || r.tensor == SnapTensor::MtpIdxKeys ||
                    r.tensor == SnapTensor::MtpIdxTail || r.tensor == SnapTensor::MtpHidden;
  if (head && (!mtp_ || r.device + 1 != st_.size()))
    throw std::logic_error(std::string("runtime::qwen4exp::Qwen4ExpEngine: a ") + snap_tensor_name(r.tensor) +
                           " snapshot run without the MTP head (or off the last device)");
  switch (r.tensor) {
    case SnapTensor::Kv: m = &b.kv; break;
    case SnapTensor::IdxKeys: m = &b.idx_keys; break;
    case SnapTensor::IdxTail: m = &b.idx_tail; break;
    case SnapTensor::GdnState: {
      // The LIVE slot (spec 8: after a commit the state may be one of gdn_spec's): the layer's slice of gdn_state at
      // slot 0, else gdn_spec's slot (layer-major: [its GDN layers][3][state]).
      const uint32_t live = mtp_ ? st(r.device).ctl->gdn_live : 0;
      if (live == 0) {
        m = &b.gdn_state;
        break;
      }
      const size_t S = gdn_state_bytes_per_layer(model_.desc), i = r.offset / S;
      if (r.offset % S != 0 || r.bytes != S || !b.gdn_spec)
        throw std::logic_error("runtime::qwen4exp::Qwen4ExpEngine: a GDN state run that is not one layer's state");
      return static_cast<uint8_t*>(b.gdn_spec->ptr()) + (i * (kGdnSlots - 1) + (live - 1)) * S;
    }
    case SnapTensor::ConvRing: m = &b.conv_ring; break;
    case SnapTensor::PleIds:
    case SnapTensor::PleRing: m = &b.ple; break;
    case SnapTensor::MtpKv: m = &mtp_->kv; break;
    case SnapTensor::MtpIdxKeys: m = &mtp_->idx_keys; break;
    case SnapTensor::MtpIdxTail: m = &mtp_->idx_tail; break;
    case SnapTensor::MtpHidden: m = &mtp_->hh; break;   // row 0: R_{pos-1}
  }
  if (r.offset + r.bytes > m->size())
    throw std::logic_error(std::string("runtime::qwen4exp::Qwen4ExpEngine: a ") + snap_tensor_name(r.tensor) +
                           " snapshot run past its allocation (" + std::to_string(r.offset + r.bytes) + " > " +
                           std::to_string(m->size()) + ")");
  return static_cast<uint8_t*>(m->ptr()) + r.offset;
}

size_t Qwen4ExpEngine::state_bytes() const { return state_snapshot_bytes(model_.desc, mtp()); }
size_t Qwen4ExpEngine::kv_bytes(uint32_t n_pos) const { return kv_snapshot_bytes(model_.desc, 0, n_pos, mtp()); }

void Qwen4ExpEngine::save_state(void* host) {
  settle_all("save_state");
  if (verify_k_ != kNoVerify)
    throw std::logic_error("runtime::qwen4exp::Qwen4ExpEngine::save_state: a verify is pending its commit");
  auto* h = static_cast<uint8_t*>(host);
  const uint32_t eos = model_.desc.ple_eos;
  for (const SnapRun& r : state_runs(model_.desc, model_.placement, pos(), mtp())) {
    if (r.pad || r.zero) {
      std::memset(h, 0, r.bytes);   // positions before 0 / filler: a cold run's history
      if (r.zero && r.tensor == SnapTensor::PleIds) std::memcpy(h, &eos, 4);   // q4_ple_gather reads EOS there
    } else {
      st(r.device).imm.copy(h, snap_ptr(r), r.bytes);
    }
    h += r.bytes;
  }
}

void Qwen4ExpEngine::load_state(const void* host, uint32_t p) {
  if (p > max_len_)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::load_state: pos " + std::to_string(p) +
                             " exceeds max_len " + std::to_string(max_len_));
  if (broken_) reset();   // a failed hand-off: the restore below rewrites what the session reads
  settle_all("load_state");
  // The GDN state goes into slot 0, made live (spec 8's rule) - before snap_ptr resolves the runs.
  for (auto& s : st_) s->ctl->gdn_live = 0;
  verify_k_ = kNoVerify;
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : state_runs(model_.desc, model_.placement, p, mtp())) {
    if (!r.pad) st(r.device).imm.copy(snap_ptr(r), h, r.bytes);
    h += r.bytes;
  }
  for (auto& s : st_) {
    s->ctl->pos = p;
    s->ctl->n_active = 0;
  }
}

void Qwen4ExpEngine::save_kv(uint32_t begin, uint32_t end, void* host) {
  settle_all("save_kv");
  auto* h = static_cast<uint8_t*>(host);
  for (const SnapRun& r : kv_runs(model_.desc, model_.placement, max_len_, begin, end, mtp())) {
    st(r.device).imm.copy(h, snap_ptr(r), r.bytes);
    h += r.bytes;
  }
}

void Qwen4ExpEngine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  settle_all("load_kv");
  const auto* h = static_cast<const uint8_t*>(host);
  for (const SnapRun& r : kv_runs(model_.desc, model_.placement, max_len_, begin, end, mtp())) {
    st(r.device).imm.copy(snap_ptr(r), h, r.bytes);
    h += r.bytes;
  }
}

std::string Qwen4ExpEngine::memory_line() const {
  std::string s;
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const std::string label = st_.size() > 1 ? "memory, device " + std::to_string(i) : std::string("memory");
    s += (i ? "\n" : "") + format_memory(label.c_str(), memory_use(i), st(i).ctx.memory_bytes());
  }
  return s;
}

// --- spec 21e: the MTP head - spec 8's draft / verify / commit -----------------------------------------------------
void Qwen4ExpEngine::require_mtp(const char* what) const {
  if (!mtp_)
    throw std::runtime_error(std::string("runtime::qwen4exp::Qwen4ExpEngine::") + what +
                             ": MTP is off - load the model with its head (loader::load_qwen4exp(..., mtp = true), "
                             "b70-serve --mtp K)");
}
Control* Qwen4ExpEngine::hctl() const { return mtp_->hctl.as<Control>(); }

uint32_t Qwen4ExpEngine::max_verify_k() const {
  const uint32_t p = pos();
  if (!mtp_ || p == 0 || p + 1 >= max_len_) return 0;
  return std::min<uint32_t>(kMaxDraft, max_len_ - p - 1);
}

void Qwen4ExpEngine::draft(uint32_t k, const std::function<uint32_t(uint32_t)>& pick) {
  require_mtp("draft");
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  const uint32_t p = pos();
  if (k == 0 || k > kMaxDraft || k > max_verify_k())
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::draft: k " + std::to_string(k) +
                             " is outside [1, max_verify_k() = " + std::to_string(max_verify_k()) + "] at pos " +
                             std::to_string(p));
  const uint32_t last = uint32_t(st_.size()) - 1;
  Stage& L = st(last);
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i)) throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::draft: a step is still running");
  // The chain: step i at position p - 1 + i (the head runs one position behind); step 0's R is hh row 0 (R_{p-1})
  // and its token the pending x_p; each step's argmax advances the head Control and leaves its id in cur_token[0].
  Control* h = hctl();
  h->pos = p - 1;
  h->n_active = 1;
  h->cur_token[0] = L.ctl->cur_token[0];
  draft_ids_.clear();
  for (uint32_t i = 0; i < k; ++i) {
    L.queue.execute(L.drafts[i].list, &L.fence);
    L.fence.wait();
    uint32_t d = h->out_token[0];
    if (pick) {
      d = pick(i);
      if (d >= model_.desc.vocab)
        throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::draft: picked id " + std::to_string(d) +
                                 " is outside the vocabulary");
      h->cur_token[0] = d;   // the next draft step's input
    }
    draft_ids_.push_back(d);
    for (auto& s : st_) s->ctl->cur_token[1 + i] = d;
  }
}

const uint32_t* Qwen4ExpEngine::verify_ids() const {
  require_mtp("verify_ids");
  return st(uint32_t(st_.size()) - 1).ctl->out_token;
}

void Qwen4ExpEngine::set_draft_input(uint32_t i, uint32_t id) {
  require_mtp("set_draft_input");
  if (i >= kMaxDraft || id >= model_.desc.vocab)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::set_draft_input: draft " + std::to_string(i) + " id " +
                             std::to_string(id));
  for (auto& s : st_) s->ctl->cur_token[1 + i] = id;
}

void Qwen4ExpEngine::verify(uint32_t k) {
  require_mtp("verify");
  const uint32_t p = pos();
  if (k > kMaxDraft || p == 0 || size_t(p) + k + 1 > max_len_)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::verify: k " + std::to_string(k) + " at pos " +
                             std::to_string(p) + " (max_verify_k() = " + std::to_string(max_verify_k()) +
                             "; pos must be >= 1 and pos + k + 1 <= max_len " + std::to_string(max_len_) + ")");
  const uint32_t last = uint32_t(st_.size()) - 1;
  Control* c = st(last).ctl;
  for (uint32_t r = 0; r <= k; ++r)
    if (c->cur_token[r] >= model_.desc.vocab)
      throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::verify: id " + std::to_string(c->cur_token[r]) +
                               " in row " + std::to_string(r) + " is outside the vocabulary");
  for (auto& s : st_) {
    s->ctl->n_active = k + 1;
    for (uint32_t r = 0; r <= k; ++r) s->ctl->cur_token[r] = c->cur_token[r];
  }
  // The head's KV pass over the same rows at p - 1 .. p + k - 1: row r is (hh[r], x_{p+r}).
  Control* h = hctl();
  h->pos = p - 1;
  h->n_active = k + 1;
  for (uint32_t r = 0; r <= k; ++r) h->cur_token[r] = c->cur_token[r];
  verify_pos_ = p;
  verify_k_ = k;
  last_verify_m_ = k + 1;
  std::vector<CapturedStep*> lists;
  for (uint32_t i = 0; i < st_.size(); ++i) lists.push_back(&st(i).verify[k]);
  submit(lists);
}

void Qwen4ExpEngine::commit(uint32_t j, uint32_t next_token) {
  require_mtp("commit");
  if (verify_k_ == kNoVerify || j > verify_k_)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::commit: j " + std::to_string(j) +
                             (verify_k_ == kNoVerify ? " with no verify() pending"
                                                     : " exceeds the last verify's k " + std::to_string(verify_k_)));
  if (next_token >= model_.desc.vocab)
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::commit: next_token " + std::to_string(next_token) +
                             " is outside the vocabulary");
  // Row j's GDN state is slot (live + j) % 4 on every device: make it live. Row j's pre-mixer H (hh row 1 + j) is
  // the next iteration's R_{pos-1}.
  for (auto& s : st_) {
    s->ctl->gdn_live = (s->ctl->gdn_live + j) % kGdnSlots;
    s->ctl->pos = verify_pos_ + j + 1;
    s->ctl->n_active = 1;
    s->ctl->cur_token[0] = next_token;
  }
  const size_t row = size_t(model_.desc.hc_n()) * 2;
  st(uint32_t(st_.size()) - 1).imm.copy(mtp_->hh_row(0), mtp_->hh_row(1 + j), row);
  verify_k_ = kNoVerify;
}

void Qwen4ExpEngine::mtp_step1() {
  if (pos() == 0) {
    // No R_{-1}: the plain list, which reads and writes GDN slot 0 (live is 0 at pos 0 - reset() and load_state()
    // both leave it so), then its pre-mixer H (the final mixer's combine materialised it in b.H) into hh row 0.
    if (st(0).ctl->gdn_live != 0) mtp_normalise_live();
    step_once();
    Stage& L = st(uint32_t(st_.size()) - 1);
    L.imm.copy(mtp_->hh_row(0), L.buffers.H.ptr(), size_t(model_.desc.hc_n()) * 2);
    return;
  }
  verify(0);
  commit(0, verify_ids()[0]);
}

void Qwen4ExpEngine::mtp_normalise_live() {
  if (!mtp_) return;
  const uint32_t live = st(0).ctl->gdn_live;
  if (live == 0) return;
  const model::Qwen4ExpDesc& d = model_.desc;
  const size_t S = gdn_state_bytes_per_layer(d);
  for (uint32_t i = 0; i < st_.size(); ++i) {
    Stage& s = st(i);
    for (uint32_t l = model_.placement.first(i); l < model_.placement.end(i); ++l)
      if (!d.is_qsa(l)) s.imm.copy(s.buffers.gdn_state_layer(l), s.buffers.gdn_slot(l, live), S);
    s.ctl->gdn_live = 0;
  }
}

void Qwen4ExpEngine::read_logits_into(float* host, uint32_t rows) {
  if (rows == 0 || rows > st(uint32_t(st_.size()) - 1).buffers.rows)
    throw std::out_of_range("read_logits_into: " + std::to_string(rows) + " rows");
  Stage& s = st(uint32_t(st_.size()) - 1);
  s.imm.copy(host, s.buffers.logits.ptr(), size_t(rows) * model_.desc.vocab * 4);
}

void Qwen4ExpEngine::read_draft_logits_into(float* host, uint32_t i) {
  require_mtp("read_draft_logits_into");
  if (i >= kMaxDraft) throw std::out_of_range("read_draft_logits_into: draft " + std::to_string(i));
  st(uint32_t(st_.size()) - 1).imm.copy(host, static_cast<const uint8_t*>(mtp_->logits.ptr()) +
                                                    size_t(i) * model_.desc.vocab * 4,
                                        size_t(model_.desc.vocab) * 4);
}

float* Qwen4ExpEngine::host_rows(size_t floats) {
  if (!host_rows_ || host_rows_->size() < floats * 4)
    host_rows_ = std::make_unique<l0::Mem>(st(uint32_t(st_.size()) - 1).ctx, l0::MemKind::Host, floats * 4);
  return host_rows_->as<float>();
}

std::vector<uint32_t> Qwen4ExpEngine::read_verify_routes() {
  require_mtp("read_verify_routes");
  const model::Qwen4ExpDesc& d = model_.desc;
  const uint32_t M = last_verify_m_;
  if (M == 0) throw std::runtime_error("read_verify_routes before the first verify");
  std::vector<uint32_t> v(size_t(d.layers) * M * kRouteWords);
  for (uint32_t l = 0; l < d.layers; ++l) {
    Stage& s = st(model_.placement.device_of(l));
    s.imm.copy(v.data() + size_t(l) * M * kRouteWords,
               static_cast<const uint8_t*>(s.buffers.routes.ptr()) + s.buffers.route_off(l), size_t(M) * kRouteWords * 4);
  }
  return v;
}

std::vector<uint32_t> Qwen4ExpEngine::read_verify_selection(uint32_t layer, uint32_t r) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (layer >= d.layers || !d.is_qsa(layer))
    throw std::out_of_range("read_verify_selection: layer " + std::to_string(layer) + " is not a QSA layer");
  Stage& s = st(model_.placement.device_of(layer));
  if (r >= s.buffers.rows) throw std::out_of_range("read_verify_selection: row " + std::to_string(r));
  std::vector<uint32_t> row(kListRow);
  s.imm.copy(row.data(),
             static_cast<const uint8_t*>(s.buffers.list.ptr()) + s.buffers.list_off(d.qsa_before(layer)) +
                 size_t(r) * kListRow * 4,
             size_t(kListRow) * 4);
  const uint32_t c = std::min(row[kCountWord], kListMax);
  std::vector<uint32_t> out(row.begin(), row.begin() + c);
  out.push_back(row[kCountWord]);
  return out;
}

std::vector<float> Qwen4ExpEngine::read_gdn_slot(uint32_t layer, uint32_t slot) {
  const model::Qwen4ExpDesc& d = model_.desc;
  if (layer >= d.layers || d.is_qsa(layer)) throw std::out_of_range("read_gdn_slot: layer " + std::to_string(layer));
  if (slot != 0) require_mtp("read_gdn_slot");
  Stage& s = st(model_.placement.device_of(layer));
  std::vector<float> v(gdn_state_bytes_per_layer(d) / 4);
  s.imm.copy(v.data(), s.buffers.gdn_slot(layer, slot), v.size() * 4);
  return v;
}

uint32_t Qwen4ExpEngine::gdn_live() const { return st(0).ctl->gdn_live; }

std::vector<uint32_t> Qwen4ExpEngine::read_draft_selection() {
  require_mtp("read_draft_selection");
  std::vector<uint32_t> row(kListRow);
  st(uint32_t(st_.size()) - 1).imm.copy(row.data(), mtp_->list.ptr(), size_t(kListRow) * 4);
  const uint32_t c = std::min(row[kCountWord], kListMax);
  std::vector<uint32_t> out(row.begin(), row.begin() + c);
  out.push_back(row[kCountWord]);
  return out;
}

std::vector<uint32_t> Qwen4ExpEngine::read_draft_routes(uint32_t i) {
  require_mtp("read_draft_routes");
  if (i >= kMaxDraft) throw std::out_of_range("read_draft_routes: draft " + std::to_string(i));
  std::vector<uint32_t> v(kRouteWords);
  st(uint32_t(st_.size()) - 1).imm.copy(v.data(), static_cast<const uint8_t*>(mtp_->routes.ptr()) + size_t(i) * kRouteWords * 4,
                                        size_t(kRouteWords) * 4);
  return v;
}

std::vector<uint16_t> Qwen4ExpEngine::read_mtp_R() {
  require_mtp("read_mtp_R");
  std::vector<uint16_t> v(model_.desc.hc_n());
  st(uint32_t(st_.size()) - 1).imm.copy(v.data(), mtp_->hh_row(0), v.size() * 2);
  return v;
}

void Qwen4ExpEngine::write_mtp_R(const std::vector<uint16_t>& R) {
  require_mtp("write_mtp_R");
  if (R.size() != model_.desc.hc_n()) throw std::invalid_argument("write_mtp_R: R is not [10240]");
  st(uint32_t(st_.size()) - 1).imm.copy(mtp_->hh_row(0), R.data(), R.size() * 2);
}

std::vector<uint16_t> Qwen4ExpEngine::read_draft_H() {
  require_mtp("read_draft_H");
  Stage& s = st(uint32_t(st_.size()) - 1);
  std::vector<uint16_t> v(model_.desc.hc_n());
  s.imm.copy(v.data(), s.buffers.H.ptr(), v.size() * 2);
  return v;
}

size_t Qwen4ExpEngine::verify_list_launches(uint32_t M) const {
  require_mtp("verify_list_launches");
  if (M == 0 || M > kVerifyRows) throw std::out_of_range("verify_list_launches: M " + std::to_string(M));
  size_t n = 0;
  for (uint32_t i = 0; i < st_.size(); ++i) n += st(i).verify[M - 1].kernel_count;
  return n;
}
size_t Qwen4ExpEngine::draft_list_launches(uint32_t i) const {
  require_mtp("draft_list_launches");
  return st(uint32_t(st_.size()) - 1).drafts.at(i).kernel_count;
}

std::vector<uint16_t> Qwen4ExpEngine::read_mtp_kv(uint32_t first, uint32_t count, bool v) {
  require_mtp("read_mtp_kv");
  if (size_t(first) + count > max_len_) throw std::out_of_range("read_mtp_kv: past max_len");
  const size_t row = size_t(model_.desc.kv_n()) * 2;
  std::vector<uint16_t> out(size_t(count) * model_.desc.kv_n());
  st(uint32_t(st_.size()) - 1).imm.copy(out.data(), static_cast<const uint8_t*>(v ? mtp_->v() : mtp_->k()) + first * row,
                                        count * row);
  return out;
}

std::vector<uint16_t> Qwen4ExpEngine::read_mtp_idx_keys(uint32_t first_block, uint32_t count) {
  require_mtp("read_mtp_idx_keys");
  const model::Qwen4ExpDesc& d = model_.desc;
  if (size_t(first_block) + count > max_len_ / d.idx_compress) throw std::out_of_range("read_mtp_idx_keys: past max_len");
  std::vector<uint16_t> out(size_t(count) * d.idx_dim);
  st(uint32_t(st_.size()) - 1).imm.copy(out.data(), static_cast<const uint8_t*>(mtp_->idx_keys.ptr()) +
                                                    size_t(first_block) * d.idx_dim * 2,
                                        out.size() * 2);
  return out;
}

}  // namespace runtime::qwen4exp
