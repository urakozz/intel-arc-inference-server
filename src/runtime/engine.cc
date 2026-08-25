#include "runtime/engine.h"

#include <chrono>
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
constexpr size_t kTapElems = size_t(Qwen35::kLayers) * kCapM * Qwen35::kHidden;
constexpr size_t kTapBytes = kTapElems * 2;  // bf16

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
      buffers_(ctx, checked_max_len(model_, max_len)),
      tap_(debug_resid ? std::unique_ptr<l0::Mem>(
                             new l0::Mem(ctx, l0::MemKind::Device, kTapBytes))
                       : nullptr),
      step_(build(ctx, model_, buffers_, tap_.get())),
      queue_(ctx),
      fence_(queue_),
      imm_(l0::CmdList::immediate(ctx)),
      control_(buffers_.control.as<Control>()) {
  reset();
}

void Engine::reset() {
  // Exactly DecodeBuffers' persistent group. Scratch is left alone on purpose
  // - see the header.
  for (l0::Mem* m : {&buffers_.control, &buffers_.gdn_state, &buffers_.conv_ring, &buffers_.kv_k,
                     &buffers_.kv_v})
    imm_.fill(m->ptr(), 0u, m->size());
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

std::vector<uint16_t> Engine::read_debug_resid() {
  if (!tap_)
    throw std::runtime_error(
        "runtime::Engine::read_debug_resid: this engine was constructed with debug_resid=false,"
        " so no per-layer tap was captured");
  std::vector<uint16_t> out(kTapElems);
  imm_.copy(out.data(), tap_->ptr(), kTapBytes);
  return out;
}

}  // namespace runtime
