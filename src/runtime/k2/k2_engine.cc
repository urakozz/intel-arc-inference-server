#include "runtime/k2/k2_engine.h"

#include <chrono>
#include <stdexcept>
#include <utility>

#include "runtime/k2/k2_capture.h"
#include "runtime/k2/k2_sizes.h"

namespace runtime::k2 {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
uint32_t checked_max_len(const loader::K2LoadedModel& m, uint32_t max_len) {
  if (m.max_len != max_len)
    throw std::runtime_error("runtime::k2::K2Engine: the model was loaded with max_len " +
                             std::to_string(m.max_len) + " but the engine was asked for " +
                             std::to_string(max_len) + " - the RoPE table and the KV cache are one number");
  return max_len;
}
}  // namespace

K2Engine::K2Engine(l0::Context& ctx, loader::K2LoadedModel model, uint32_t max_len, bool debug_tap)
    : ctx_(ctx),
      model_(std::move(model)),
      buffers_(ctx, *model_.desc, checked_max_len(model_, max_len)),
      tap_(debug_tap ? std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, tap_bytes(*model_.desc))
                     : nullptr),
      step_(build(ctx, model_, buffers_, tap_.get())),
      queue_(ctx),
      fence_(queue_),
      imm_(l0::CmdList::immediate(ctx)),
      control_(buffers_.control.as<Control>()) {
  reset();
}

void K2Engine::reset() { buffers_.zero(imm_); }

void K2Engine::replay() {
  const size_t pos = control_->pos, n = control_->n_active;
  if (pos + n > size_t(buffers_.max_len))
    throw std::runtime_error("runtime::k2::K2Engine: pos " + std::to_string(pos) + " + n_active " +
                             std::to_string(n) + " exceeds max_len " + std::to_string(buffers_.max_len) +
                             " - the KV cache and the RoPE table stop there");
  queue_.execute(step_.list, &fence_);
  fence_.wait();
}

void K2Engine::ingest(const std::vector<uint32_t>& ids) {
  control_->n_active = 1;
  for (uint32_t id : ids) {
    // embed_gather would refuse it on the device (Control::debug_flag); the host says why.
    if (id >= model_.desc->vocab)
      throw std::runtime_error("runtime::k2::K2Engine::ingest: id " + std::to_string(id) +
                               " is outside the vocabulary (" + std::to_string(model_.desc->vocab) + ")");
    control_->cur_token[0] = id;
    replay();
  }
}

std::vector<uint32_t> K2Engine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  control_->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = control_->cur_token[0];
    const Clock::time_point f0 = Clock::now();
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

std::vector<uint16_t> K2Engine::read_debug_resid() {
  if (!tap_) throw std::runtime_error("runtime::k2::K2Engine: constructed without debug_tap");
  std::vector<uint16_t> v(tap_->size() / 2);
  imm_.copy(v.data(), tap_->ptr(), tap_->size());
  return v;
}

std::vector<uint32_t> K2Engine::read_routes() {
  std::vector<uint32_t> v(buffers_.routes.size() / 4);
  imm_.copy(v.data(), buffers_.routes.ptr(), buffers_.routes.size());
  return v;
}

std::vector<float> K2Engine::read_logits() {
  std::vector<float> v(model_.desc->vocab);
  imm_.copy(v.data(), buffers_.logits.ptr(), v.size() * 4);
  return v;
}

MemoryComponents K2Engine::memory_use() const {
  MemoryComponents c;
  c.model = model_.report.total();
  c.kv = buffers_.kv_k.size() + buffers_.kv_v.size();
  c.decode_state = buffers_.control.size() + buffers_.scratch_bytes() + (tap_ ? tap_->size() : 0);
  c.prefill_scratch = pf_bytes_;   // spec 18c: 0 until the first prefill / prepare_prefill
  return c;
}

std::string K2Engine::memory_line() const {
  return format_memory("memory", memory_use(), ctx_.memory_bytes());
}

}  // namespace runtime::k2
