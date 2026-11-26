// Plan 41f Task 4: the prefix cache's snapshots of runtime::inkling2::Engine (inkling2_engine.h says what is stored).
#include <algorithm>
#include <cstring>
#include <iterator>
#include <optional>
#include <stdexcept>

#include "l0/cmdlist.h"
#include "runtime/inkling2/inkling2_engine.h"

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;

// The window's reach back from the next position: its keys are [pos - (window - 1), pos].
constexpr uint32_t kStateRows = D::window - 1;

// Where each tensor's rows live. No default: a new SnapTensor must say how it is stored.
enum class Kind { WindowRing, Whole, Positioned };
Kind kind_of(SnapTensor t) {
  switch (t) {
    case SnapTensor::RingK:
    case SnapTensor::RingV: return Kind::WindowRing;
    case SnapTensor::ConvK:
    case SnapTensor::ConvV:
    case SnapTensor::ConvAttn:
    case SnapTensor::ConvMlp:
    case SnapTensor::DraftRingK:
    case SnapTensor::DraftRingV:
    case SnapTensor::DraftTap: return Kind::Whole;
    case SnapTensor::FullK:
    case SnapTensor::FullV:
    case SnapTensor::FullKScale:
    case SnapTensor::FullVScale:
    case SnapTensor::DraftFullK:
    case SnapTensor::DraftFullV: return Kind::Positioned;
  }
  throw std::logic_error("Inkling snapshot: an unknown tensor");
}

constexpr SnapTensor kTensors[] = {SnapTensor::FullK,      SnapTensor::FullV,      SnapTensor::FullKScale,
                                   SnapTensor::FullVScale, SnapTensor::RingK,      SnapTensor::RingV,
                                   SnapTensor::ConvK,      SnapTensor::ConvV,      SnapTensor::ConvAttn,
                                   SnapTensor::ConvMlp,    SnapTensor::DraftFullK, SnapTensor::DraftFullV,
                                   SnapTensor::DraftRingK, SnapTensor::DraftRingV, SnapTensor::DraftTap};
static_assert(std::size(kTensors) == size_t(SnapTensor::DraftTap) + 1, "every SnapTensor once, in order");

bool drafter_tensor(SnapTensor t) { return t >= SnapTensor::DraftFullK; }

}  // namespace

l0::Mem* Engine::snapshot_buffer(SnapTensor t) const {
  auto& b = const_cast<Buffers&>(buffers_);
  l0::Mem* m = drafter_tensor(t) ? gs::snapshot_buffer<l0::Mem, SnapTensor, DraftSizes>(b.draft, t)
                                 : gs::snapshot_buffer<l0::Mem, SnapTensor, PersistentSizes>(b.persistent, t);
  return m;   // null: not allocated here (the int8 scales under bf16, the drafter's without one)
}

std::vector<Engine::SnapRun> Engine::state_runs(uint32_t pos) const {
  std::vector<SnapRun> runs;
  for (const SnapTensor t : kTensors) {
    l0::Mem* m = snapshot_buffer(t);
    if (m == nullptr) continue;
    const Kind k = kind_of(t);
    if (k == Kind::Whole) {
      runs.push_back({m, 0, m->size(), false});
    } else if (k == Kind::WindowRing) {
      const size_t row = tp_.kv_n() * sizeof(uint16_t), slot = size_t(kRing) * row;
      for (uint32_t s = 0; s < share_.window; ++s) {
        // Positions pos - kStateRows .. pos - 1, ascending, at ring slot q & (kRing - 1) (a position before 0 too:
        // its row is zero on the host, written back as zeros).
        for (int64_t q = int64_t(pos) - kStateRows; q < int64_t(pos);) {
          const uint32_t at = uint32_t(q) & (kRing - 1);
          int64_t stop = std::min<int64_t>(pos, q + (kRing - at));
          if (q < 0) stop = std::min<int64_t>(stop, 0);
          runs.push_back({m, s * slot + at * row, size_t(stop - q) * row, q < 0});
          q = stop;
        }
      }
    }
  }
  return runs;
}

std::vector<Engine::SnapRun> Engine::kv_runs(uint32_t begin, uint32_t end) const {
  std::vector<SnapRun> runs;
  if (begin == end) return runs;
  for (const SnapTensor t : kTensors) {
    l0::Mem* m = snapshot_buffer(t);
    if (m == nullptr || kind_of(t) != Kind::Positioned) continue;
    const uint32_t slots = drafter_tensor(t) ? 1 : share_.full;
    const size_t slot = m->size() / slots, row = slot / max_len_;
    for (uint32_t s = 0; s < slots; ++s) runs.push_back({m, s * slot + begin * row, (end - begin) * row, false});
  }
  return runs;
}

size_t Engine::shadow_image(SnapTensor t) const {
  size_t at = 0;
  for (const SnapTensor u : kTensors) {
    if (u == t) return at;
    const l0::Mem* m = snapshot_buffer(u);
    if (m && kind_of(u) != Kind::Positioned) at += m->size();
  }
  throw std::logic_error("Inkling snapshot: an unknown tensor");
}

void Engine::record_visit_shadow(l0::CmdList& list, uint32_t visit, uint32_t shadow) {
  const auto& mem = *buffers_.handoff.snapshot_shadow;
  auto* base = static_cast<std::byte*>(mem.ptr()) + shadow * (mem.size() / 2);
  const auto copy = [&](SnapTensor t, size_t offset, size_t bytes) {
    const l0::Mem* m = snapshot_buffer(t);
    if (m) list.copy(base + shadow_image(t) + offset, static_cast<const std::byte*>(m->ptr()) + offset, bytes);
  };
  if (visit == D::visit_count) {   // the drafter's rings and taps, written last in the chunk
    for (const SnapTensor t : {SnapTensor::DraftRingK, SnapTensor::DraftRingV, SnapTensor::DraftTap})
      if (const l0::Mem* m = snapshot_buffer(t)) copy(t, 0, m->size());
    return;
  }
  const auto& v = desc_.visits[visit];
  const size_t here = visit - stage_.first, kv = 3 * size_t(tp_.kv_n()) * sizeof(float);
  const size_t hidden = 3 * size_t(D::hidden) * sizeof(float);
  if (v.window) {
    const size_t ring = size_t(kRing) * tp_.kv_n() * sizeof(uint16_t);
    copy(SnapTensor::RingK, slot(v) * ring, ring);
    copy(SnapTensor::RingV, slot(v) * ring, ring);
  }
  copy(SnapTensor::ConvK, here * kv, kv);
  copy(SnapTensor::ConvV, here * kv, kv);
  copy(SnapTensor::ConvAttn, here * hidden, hidden);
  copy(SnapTensor::ConvMlp, here * hidden, hidden);
}

size_t Engine::state_bytes() const {
  size_t n = 0;
  for (const SnapRun& r : state_runs(0)) n += r.bytes;
  return n;
}

size_t Engine::kv_bytes(uint32_t positions) const {
  size_t n = 0;
  for (const SnapRun& r : kv_runs(0, positions)) n += r.bytes;
  return n;
}

void Engine::save_state(void* host) {
  queue_.synchronize();
  // The copy engine: the window rings' device-to-host runs at ~2/3 of the compute engine's time (request-end store
  // 36 -> 23 ms at 32k).
  auto list = l0::CmdList::immediate_copy(ctx_);
  auto* h = static_cast<std::byte*>(host);
  for (const SnapRun& r : state_runs(committed_)) {
    if (r.zero) std::memset(h, 0, r.bytes);   // positions before 0: never read
    else list.copy(h, static_cast<const std::byte*>(r.mem->ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
  list.host_synchronize();
}

void Engine::load_state(const void* host, uint32_t pos) {
  queue_.synchronize();
  auto list = l0::CmdList::immediate_async(ctx_);
  const auto* h = static_cast<const std::byte*>(host);
  for (const SnapRun& r : state_runs(pos)) {
    list.copy(static_cast<std::byte*>(r.mem->ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
  list.host_synchronize();
  committed_ = ctrl_->pos = pos;
  ctrl_->n_active = 0;
  pending_ = 0;
  prefill_logits_ = false;
  written_ = std::max(written_, pos);
}

void Engine::save_kv(uint32_t begin, uint32_t end, void* host) {
  queue_.synchronize();
  if (overlap_kv_ && !kv_store_) kv_store_ = std::make_unique<l0::CmdList>(l0::CmdList::immediate_copy(ctx_));
  auto own = overlap_kv_ ? std::optional<l0::CmdList>{} : std::optional<l0::CmdList>{l0::CmdList::immediate_async(ctx_)};
  l0::CmdList& list = overlap_kv_ ? *kv_store_ : *own;
  auto* h = static_cast<std::byte*>(host);
  for (const SnapRun& r : kv_runs(begin, end)) {
    list.copy(h, static_cast<const std::byte*>(r.mem->ptr()) + r.offset, r.bytes);
    h += r.bytes;
  }
  if (!overlap_kv_) list.host_synchronize();
}

void Engine::load_kv(uint32_t begin, uint32_t end, const void* host) {
  queue_.synchronize();
  auto list = l0::CmdList::immediate_async(ctx_);
  const auto* h = static_cast<const std::byte*>(host);
  for (const SnapRun& r : kv_runs(begin, end)) {
    list.copy(static_cast<std::byte*>(r.mem->ptr()) + r.offset, h, r.bytes);
    h += r.bytes;
  }
  list.host_synchronize();
  written_ = std::max(written_, end);
}

}  // namespace runtime::inkling2
