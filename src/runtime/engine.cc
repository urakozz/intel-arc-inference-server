#include "runtime/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "model/qwen35.h"

namespace runtime {
namespace {
using model::Qwen35;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// The captured list's M. It is runtime/capture.cc's `kCapM` - 1 in this plan -
// and it sizes the tap the same way capture.cc's check_sizes() expects it, so
// the two constants have to move together. (capture.cc keeps it file-local; it
// is the M the kernel *variants* were compiled for, not DecodeBuffers::kM,
// which is the allocation capacity.)
constexpr uint32_t kCapM = 1;
// The tap is bf16 [layers][kCapM][5120] - the descriptor's layer count (spec 14).
size_t tap_elems(const model::ModelDesc& d) { return size_t(d.layers) * kCapM * Qwen35::kHidden; }

// Checked in the member-initialiser list so it throws before DecodeBuffers
// allocates ~1.2 GB. runtime::build() checks the same identity, but only after
// the buffers exist and the weights are already resident.
uint32_t checked_max_len(const loader::LoadedModel& m, uint32_t max_len) {
  if (m.max_len != max_len)
    throw std::runtime_error("runtime::Engine: model was loaded with max_len " +
                             std::to_string(m.max_len) + " but the engine was asked for " +
                             std::to_string(max_len) +
                             " - the RoPE table, the KV allocation and the attention variants'"
                             " baked MAXLEN are one number");
  return max_len;
}
}  // namespace

Engine::Engine(l0::Context& ctx, loader::LoadedModel model, uint32_t max_len, bool debug_resid)
    : ctx_(ctx),
      model_(std::move(model)),
      persist_(ctx, checked_max_len(model_, max_len), *model_.desc),
      decode_scratch_(ctx, persist_.max_len, *model_.desc),
      buffers_(persist_, decode_scratch_),
      tap_(debug_resid ? std::unique_ptr<l0::Mem>(
                             new l0::Mem(ctx, l0::MemKind::Device, tap_elems(*model_.desc) * 2))
                       : nullptr),
      mtp_(model_.mtp ? std::make_unique<MtpBuffers>(ctx, persist_.max_len, *model_.desc)
                      : nullptr),
      step_(build(ctx, model_, buffers_, tap_.get())),
      queue_(ctx),
      fence_(queue_),
      imm_(l0::CmdList::immediate(ctx)),
      control_(buffers_.control.as<Control>()) {
  if (mtp_) {
    // Spec 8: the draft lists (one per draft index, each writing its own logits row)
    // and the verify lists at M = 1..kSlots. Only the head's model carries these.
    hctl_ = mtp_->hctl.as<Control>();
    for (uint32_t i = 0; i < kMaxDraft; ++i)
      draft_steps_.push_back(build_draft(ctx, model_, buffers_, *mtp_, i));
    for (uint32_t M = 1; M <= MtpBuffers::kSlots; ++M)
      verify_steps_.push_back(build_verify(ctx, model_, buffers_, *mtp_, M));
  }
  reset();
}

void Engine::reset() {
  // Exactly the persistent group, in exactly the order PersistentBuffers'
  // constructor filled it - one function, so "reset writes what construction
  // wrote" is a property of the code and not of two lists agreeing. Scratch is
  // left alone on purpose - see the header.
  persist_.zero(imm_);
  if (mtp_) mtp_->zero(imm_);
  verify_k_ = kNoVerify;
  draft_ids_.clear();
}

void Engine::replay() {
  const size_t pos = control_->pos, n = control_->n_active;
  if (pos + n > size_t(buffers_.max_len))
    throw std::runtime_error(
        "runtime::Engine: pos " + std::to_string(pos) + " + n_active " + std::to_string(n) +
        " exceeds max_len " + std::to_string(buffers_.max_len) +
        " - the KV cache and the RoPE table stop there. Start a new session with reset(), or"
        " load the model and the engine with a larger max_len.");
  queue_.execute(step_.list, &fence_);
  fence_.wait();
}

void Engine::ingest(const std::vector<uint32_t>& ids) {
  control_->n_active = 1;
  for (uint32_t id : ids) {
    control_->cur_token[0] = id;
    if (mtp_)
      mtp_step1();
    else
      replay();
  }
}

std::vector<uint32_t> Engine::generate(uint32_t n,
                                       const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  control_->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    // Produced by the previous fence; replay i is what turns it into i + 1.
    const uint32_t id = control_->cur_token[0];
    const Clock::time_point f0 = Clock::now();
    if (mtp_)
      mtp_step1();
    else
      replay();
    fence_ms += ms_since(f0);
    out.push_back(id);
    if (on_token) on_token(id);
  }
  last_gen_ms_ = ms_since(t0);
  last_fence_ms_ = fence_ms;
  last_tok_per_s_ = (n != 0 && last_gen_ms_ > 0.0) ? double(n) * 1000.0 / last_gen_ms_ : 0.0;
  return out;
}

// --- spec 8: MTP -------------------------------------------------------------------

namespace {
constexpr size_t kHidBytes = size_t(Qwen35::kHidden) * 2;   // one bf16 hidden row
}  // namespace

void Engine::require_mtp(const char* what) const {
  if (!mtp_)
    throw std::runtime_error(std::string("runtime::Engine::") + what +
                             ": MTP is off - load the model with its head (loader::load(..., mtp"
                             " = true), b70 --mtp)");
}

uint32_t Engine::max_verify_k() const {
  const uint32_t pos = control_->pos;
  if (!mtp_ || pos == 0 || pos + 1 >= buffers_.max_len) return 0;
  return std::min<uint32_t>(kMaxDraft, buffers_.max_len - pos - 1);
}

const float* Engine::mtp_logits_device() const {
  require_mtp("mtp_logits_device");
  return mtp_->logits.as<float>();
}
const float* Engine::verify_logits_device() const {
  require_mtp("verify_logits_device");
  return buffers_.logits.as<float>();
}
const CapturedStep& Engine::verify_step(uint32_t M) const {
  require_mtp("verify_step");
  return verify_steps_.at(M - 1);
}
const CapturedStep& Engine::draft_step(uint32_t i) const {
  require_mtp("draft_step");
  return draft_steps_.at(i);
}

void Engine::draft(uint32_t k, const std::function<uint32_t(uint32_t)>& pick) {
  require_mtp("draft");
  const uint32_t pos = control_->pos;
  if (k == 0 || k > kMaxDraft || k > max_verify_k())
    throw std::runtime_error("runtime::Engine::draft: k " + std::to_string(k) +
                             " is outside [1, max_verify_k() = " + std::to_string(max_verify_k()) +
                             "] at pos " + std::to_string(pos));
  // The chain's first hidden is h_{pos-1} (hh row 0); each draft list writes its own
  // output hidden back into dh. Draft i runs at position pos - 1 + i (P0 §1).
  imm_.copy(mtp_->dh.ptr(), mtp_->hh.ptr(), kHidBytes);
  hctl_->pos = pos - 1;
  hctl_->n_active = 1;
  hctl_->cur_token[0] = control_->cur_token[0];
  draft_ids_.clear();
  for (uint32_t i = 0; i < k; ++i) {
    queue_.execute(draft_steps_[i].list, &fence_);
    fence_.wait();
    uint32_t d = hctl_->out_token[0];
    if (pick) {
      d = pick(i);
      if (d >= Qwen35::kVocab)
        throw std::runtime_error("runtime::Engine::draft: picked id " + std::to_string(d) +
                                 " is outside the vocabulary");
      hctl_->cur_token[0] = d;   // the next draft list's input
    }
    draft_ids_.push_back(d);
    control_->cur_token[1 + i] = d;
  }
}

void Engine::verify(uint32_t k) {
  require_mtp("verify");
  const uint32_t pos = control_->pos;
  if (k > kMaxDraft || pos == 0 || size_t(pos) + k + 1 > buffers_.max_len)
    throw std::runtime_error("runtime::Engine::verify: k " + std::to_string(k) + " at pos " +
                             std::to_string(pos) + " (max_verify_k() = " +
                             std::to_string(max_verify_k()) + "; pos must be >= 1 and pos + k"
                             " + 1 <= max_len " + std::to_string(buffers_.max_len) + ")");
  for (uint32_t r = 0; r <= k; ++r)
    if (control_->cur_token[r] >= Qwen35::kVocab)
      throw std::runtime_error("runtime::Engine::verify: id " +
                               std::to_string(control_->cur_token[r]) + " in row " +
                               std::to_string(r) + " is outside the vocabulary");
  control_->n_active = k + 1;
  // The head's KV fill of the same rows at pos - 1 .. pos + k - 1: row r is the pair
  // (hh[r], cur_token[r]) = (h_{pos-1+r}, x[pos+r]) - capture.h, build_verify.
  hctl_->pos = pos - 1;
  hctl_->n_active = k + 1;
  for (uint32_t r = 0; r <= k; ++r) hctl_->cur_token[r] = control_->cur_token[r];
  verify_pos_ = pos;
  verify_k_ = k;
  queue_.execute(verify_steps_[k].list, &fence_);
  fence_.wait();
}

void Engine::commit(uint32_t j, uint32_t next_token) {
  require_mtp("commit");
  if (verify_k_ == kNoVerify || j > verify_k_)
    throw std::runtime_error("runtime::Engine::commit: j " + std::to_string(j) +
                             (verify_k_ == kNoVerify ? " with no verify() pending"
                                                     : " exceeds the last verify's k " +
                                                           std::to_string(verify_k_)));
  if (next_token >= Qwen35::kVocab)
    throw std::runtime_error("runtime::Engine::commit: next_token " + std::to_string(next_token) +
                             " is outside the vocabulary");
  // Row j's GDN state is slot (live + j) % kSlots (gdn_step.cl SPEC_SLOTS): make it
  // live. Row j's hidden (hh row 1 + j) is the next iteration's h_{pos-1}.
  control_->gdn_live = (control_->gdn_live + j) % MtpBuffers::kSlots;
  imm_.copy(mtp_->hh.ptr(), mtp_->hh.as<uint8_t>() + (1 + size_t(j)) * kHidBytes, kHidBytes);
  control_->pos = verify_pos_ + j + 1;
  control_->n_active = 1;
  control_->cur_token[0] = next_token;
  verify_k_ = kNoVerify;
}

void Engine::mtp_step1() {
  if (control_->pos == 0) {
    // No h_{-1}: the plain list, which reads and writes slot 0 (live is 0 at pos 0 -
    // reset() and load_state() both leave it so), then its hidden into hh row 0.
    if (control_->gdn_live != 0) mtp_normalise_live();
    replay();
    imm_.copy(mtp_->hh.ptr(), buffers_.x.ptr(), kHidBytes);
    return;
  }
  verify(0);
  commit(0, control_->out_token[0]);
}

void Engine::mtp_normalise_live() {
  const uint32_t live = control_->gdn_live;
  if (!mtp_ || live == 0) return;
  imm_.copy(persist_.gdn_state.ptr(),
            mtp_->gdn_spec.as<uint8_t>() + size_t(live - 1) * persist_.gdn_state.size(),
            persist_.gdn_state.size());
  control_->gdn_live = 0;
}

// --- spec 7 snapshots -----------------------------------------------------------

namespace {
void check_range(uint32_t begin, uint32_t end, uint32_t max_len, const char* what) {
  if (begin > end || end > max_len)
    throw std::runtime_error(std::string("runtime::Engine::") + what + ": range [" +
                             std::to_string(begin) + ", " + std::to_string(end) +
                             ") is not within [0, max_len " + std::to_string(max_len) + ")");
}
}  // namespace

size_t Engine::state_bytes() const {
  return persist_.gdn_state.size() + persist_.conv_ring.size() + (mtp_ ? kHidBytes : 0);
}

size_t Engine::kv_bytes(uint32_t n_pos) const {
  const size_t fa = model_.desc->fa_layers;   // 16 on Qwen3.8, 18 on Agnes (spec 14)
  const size_t per_layer_pos = persist_.kv_k.size() / fa / buffers_.max_len;
  return 2 * per_layer_pos * (fa + (mtp_ ? 1 : 0)) * n_pos;   // K and V
}

void Engine::save_state(void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  // The LIVE GDN slot (spec 8: after a commit it may be one of gdn_spec's).
  const uint32_t live = mtp_ ? control_->gdn_live : 0;
  const void* gdn = live == 0 ? persist_.gdn_state.ptr()
                              : mtp_->gdn_spec.as<uint8_t>() +
                                    size_t(live - 1) * persist_.gdn_state.size();
  imm_.copy(h, gdn, persist_.gdn_state.size());
  imm_.copy(h + persist_.gdn_state.size(), persist_.conv_ring.ptr(), persist_.conv_ring.size());
  if (mtp_)
    imm_.copy(h + persist_.gdn_state.size() + persist_.conv_ring.size(), mtp_->hh.ptr(),
              kHidBytes);
}

void Engine::load_state(const void* host, uint32_t pos) {
  if (pos > buffers_.max_len)
    throw std::runtime_error("runtime::Engine::load_state: pos " + std::to_string(pos) +
                             " exceeds max_len " + std::to_string(buffers_.max_len));
  const auto* h = static_cast<const uint8_t*>(host);
  imm_.copy(persist_.gdn_state.ptr(), h, persist_.gdn_state.size());
  imm_.copy(persist_.conv_ring.ptr(), h + persist_.gdn_state.size(), persist_.conv_ring.size());
  if (mtp_) {
    imm_.copy(mtp_->hh.ptr(), h + persist_.gdn_state.size() + persist_.conv_ring.size(),
              kHidBytes);
    control_->gdn_live = 0;   // the state went into slot 0
    verify_k_ = kNoVerify;
  }
  control_->pos = pos;
  control_->n_active = 0;
}

// kv_k / kv_v are [fa_layers][max_len][4][256]: a position range is one contiguous run
// per layer, so fa_layers (16 on Qwen3.8, 18 on Agnes) copies each for K and V.
void Engine::save_kv(uint32_t begin, uint32_t end, void* host) const {
  check_range(begin, end, buffers_.max_len, "save_kv");
  if (begin == end) return;
  const size_t kKvLayers = model_.desc->fa_layers;
  const size_t per_pos = persist_.kv_k.size() / kKvLayers / buffers_.max_len;
  const size_t run = per_pos * (end - begin);
  auto* h = static_cast<uint8_t*>(host);
  for (int kv = 0; kv < 2; ++kv) {
    const l0::Mem* m = kv == 0 ? &persist_.kv_k : &persist_.kv_v;
    for (size_t l = 0; l < kKvLayers; ++l, h += run)
      imm_.copy(h, m->as<uint8_t>() + (l * buffers_.max_len + begin) * per_pos, run);
    if (mtp_) {   // the head's layer, after the last FA layer
      const l0::Mem& hm = kv == 0 ? mtp_->kv_k : mtp_->kv_v;
      imm_.copy(h, hm.as<uint8_t>() + size_t(begin) * per_pos, run);
      h += run;
    }
  }
}

void Engine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  check_range(begin, end, buffers_.max_len, "load_kv");
  if (begin == end) return;
  const size_t kKvLayers = model_.desc->fa_layers;
  const size_t per_pos = persist_.kv_k.size() / kKvLayers / buffers_.max_len;
  const size_t run = per_pos * (end - begin);
  const auto* h = static_cast<const uint8_t*>(host);
  for (int kv = 0; kv < 2; ++kv) {
    l0::Mem* m = kv == 0 ? &persist_.kv_k : &persist_.kv_v;
    for (size_t l = 0; l < kKvLayers; ++l, h += run)
      imm_.copy(m->as<uint8_t>() + (l * buffers_.max_len + begin) * per_pos, h, run);
    if (mtp_) {
      l0::Mem& hm = kv == 0 ? mtp_->kv_k : mtp_->kv_v;
      imm_.copy(hm.as<uint8_t>() + size_t(begin) * per_pos, h, run);
      h += run;
    }
  }
}

std::vector<uint16_t> Engine::read_debug_resid() {
  if (!tap_)
    throw std::runtime_error(
        "runtime::Engine::read_debug_resid: this engine was constructed with debug_resid=false,"
        " so no per-layer tap was captured");
  std::vector<uint16_t> out(tap_elems(*model_.desc));
  imm_.copy(out.data(), tap_->ptr(), out.size() * 2);
  return out;
}

}  // namespace runtime
