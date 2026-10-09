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
        Q4Attn a)
      : ctx(c), buffers(c, d, p, dev, max_len, a), queue(c), fence(queue), imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()) {}
  l0::Context& ctx;
  Qwen4ExpBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  std::optional<CapturedStep> step, inj_step;
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
  for (uint32_t i = 0; i < p.devices; ++i) {
    st_.push_back(std::make_unique<Stage>(*devices[i], d, p, i, max_len, attn_));
    taps_.push_back(debug_tap ? std::make_unique<l0::Mem>(*devices[i], l0::MemKind::Device, tap_bytes(d)) : nullptr);
    inj_rows_.push_back(nullptr);
  }
  pending_.assign(p.devices, false);
  if (p.devices == 2) {
    link_ = std::make_unique<PipelineLink>(*devices[0], *devices[1], landing_layout(d), opt.handoff);
    binding_ = link_->binding(opt.spin_limit);
  }
  ple_init();
  for (uint32_t i = 0; i < p.devices; ++i)
    st(i).step.emplace(build(st(i).ctx, model_, model_.parts[i], st(i).buffers, link_ ? &binding_ : nullptr,
                             taps_[i].get(), nullptr));
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

void Qwen4ExpEngine::reset() {
  for (uint32_t i = 0; i < st_.size(); ++i)
    if (!settle(i))
      throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine::reset: device " + std::to_string(i) +
                               "'s last step is still running after " + std::to_string(opt_.timeout_ms) + " ms");
  for (uint32_t i = 0; i < st_.size(); ++i) st(i).buffers.zero(st(i).imm);
  if (link_) link_->zero(st(0).imm, st(1).imm);
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
  if (broken_) throw std::runtime_error("pipeline: a hand-off failed earlier in this session; reset() first");
  Stage& last = st(uint32_t(st_.size()) - 1);
  const size_t pos = last.ctl->pos, n = last.ctl->n_active;
  if (pos + n > size_t(max_len_))
    throw std::runtime_error("runtime::qwen4exp::Qwen4ExpEngine: pos " + std::to_string(pos) + " + n_active " +
                             std::to_string(n) + " exceeds max_len " + std::to_string(max_len_) +
                             " - the KV, the indexer keys and the RoPE table stop there");
  if (st_.size() == 1) {
    st(0).queue.execute(active(0).list, &st(0).fence);
    st(0).fence.wait();
    return;
  }
  if (link_->event) link_->event->host_reset();
  const bool send = !drop_next_;
  drop_next_ = false;
  if (send) {
    st(0).queue.execute(active(0).list, &st(0).fence);
    pending_[0] = true;
  }
  st(1).queue.execute(active(1).list, &st(1).fence);
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
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const uint32_t first = model_.placement.first(i), n = model_.placement.count(i);
    st(i).imm.copy(v.data() + size_t(first) * kRouteWords,
                   static_cast<const uint8_t*>(st(i).buffers.routes.ptr()) + route_at(first), size_t(n) * kRouteWords * 4);
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
    st(dev).imm.copy(row.data(), static_cast<const uint8_t*>(st(dev).buffers.list.ptr()) + list_at(qi), size_t(kListRow) * 4);
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
    st(dev).imm.copy(v.data() + size_t(qi) * 2, static_cast<const uint8_t*>(st(dev).buffers.diag.ptr()) + diag_at(qi), 8);
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
                   (taps_[dev] ? taps_[dev]->size() : 0) + (link_ ? link_->bytes(dev) : 0);
  return c;
}

std::string Qwen4ExpEngine::memory_line() const {
  std::string s;
  for (uint32_t i = 0; i < st_.size(); ++i) {
    const std::string label = st_.size() > 1 ? "memory, device " + std::to_string(i) : std::string("memory");
    s += (i ? "\n" : "") + format_memory(label.c_str(), memory_use(i), st(i).ctx.memory_bytes());
  }
  return s;
}

}  // namespace runtime::qwen4exp
