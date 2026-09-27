#include "runtime/engine.h"

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
      persist_(ctx, checked_max_len(model_, max_len)),
      decode_scratch_(ctx, persist_.max_len),
      buffers_(persist_, decode_scratch_),
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
  // Exactly the persistent group, in exactly the order PersistentBuffers'
  // constructor filled it - one function, so "reset writes what construction
  // wrote" is a property of the code and not of two lists agreeing. Scratch is
  // left alone on purpose - see the header.
  persist_.zero(imm_);
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

// --- spec 7 snapshots -----------------------------------------------------------

namespace {
constexpr size_t kKvLayers = 16;
void check_range(uint32_t begin, uint32_t end, uint32_t max_len, const char* what) {
  if (begin > end || end > max_len)
    throw std::runtime_error(std::string("runtime::Engine::") + what + ": range [" +
                             std::to_string(begin) + ", " + std::to_string(end) +
                             ") is not within [0, max_len " + std::to_string(max_len) + ")");
}
}  // namespace

size_t Engine::state_bytes() const { return persist_.gdn_state.size() + persist_.conv_ring.size(); }

size_t Engine::kv_bytes(uint32_t n_pos) const {
  return 2 * persist_.kv_k.size() / buffers_.max_len * n_pos;   // K and V
}

void Engine::save_state(void* host) const {
  auto* h = static_cast<uint8_t*>(host);
  imm_.copy(h, persist_.gdn_state.ptr(), persist_.gdn_state.size());
  imm_.copy(h + persist_.gdn_state.size(), persist_.conv_ring.ptr(), persist_.conv_ring.size());
}

void Engine::load_state(const void* host, uint32_t pos) {
  if (pos > buffers_.max_len)
    throw std::runtime_error("runtime::Engine::load_state: pos " + std::to_string(pos) +
                             " exceeds max_len " + std::to_string(buffers_.max_len));
  const auto* h = static_cast<const uint8_t*>(host);
  imm_.copy(persist_.gdn_state.ptr(), h, persist_.gdn_state.size());
  imm_.copy(persist_.conv_ring.ptr(), h + persist_.gdn_state.size(), persist_.conv_ring.size());
  control_->pos = pos;
  control_->n_active = 0;
}

// kv_k / kv_v are [16 layers][max_len][4][256]: a position range is one contiguous run
// per layer, so 16 copies each for K and V.
void Engine::save_kv(uint32_t begin, uint32_t end, void* host) const {
  check_range(begin, end, buffers_.max_len, "save_kv");
  if (begin == end) return;
  const size_t per_pos = persist_.kv_k.size() / kKvLayers / buffers_.max_len;
  const size_t run = per_pos * (end - begin);
  auto* h = static_cast<uint8_t*>(host);
  for (const l0::Mem* m : {&persist_.kv_k, &persist_.kv_v})
    for (size_t l = 0; l < kKvLayers; ++l, h += run)
      imm_.copy(h, m->as<uint8_t>() + (l * buffers_.max_len + begin) * per_pos, run);
}

void Engine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  check_range(begin, end, buffers_.max_len, "load_kv");
  if (begin == end) return;
  const size_t per_pos = persist_.kv_k.size() / kKvLayers / buffers_.max_len;
  const size_t run = per_pos * (end - begin);
  const auto* h = static_cast<const uint8_t*>(host);
  for (l0::Mem* m : {&persist_.kv_k, &persist_.kv_v})
    for (size_t l = 0; l < kKvLayers; ++l, h += run)
      imm_.copy(m->as<uint8_t>() + (l * buffers_.max_len + begin) * per_pos, h, run);
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
