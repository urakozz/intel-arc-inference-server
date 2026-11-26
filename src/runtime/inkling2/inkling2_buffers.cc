#include "runtime/inkling2/inkling2_buffers.h"

#include <stdexcept>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "runtime/dflash/dflash_sizes.h"

namespace runtime::inkling2 {
namespace {
using l0::Mem, l0::MemKind;
using L = loader::Inkling2WeightLayout;

const PersistentSizes& checked(l0::Context& ctx, const Sizes& s, const L& layout) {
  if (s.weights.model != layout.bytes[L::kModel] || s.weights.layers != layout.layer_bytes() ||
      (s.weights.draft && s.weights.draft != layout.bytes[L::kDraft]))
    throw std::logic_error("Inkling weight layout does not match its allocation plan");
  constexpr size_t reserve = size_t(kDefaultReserveGb * 1e9);
  const size_t capacity = ctx.memory_bytes();
  if (capacity < reserve || s.memory().total() > capacity - reserve)
    throw std::runtime_error("Inkling exceeds the card budget. Use a shorter --max-len.");
  return s.persistent;
}

template <class S, class B>
void allocate(l0::Context& ctx, B& b, const S& s) {
  template for (constexpr auto member : gs::fields<S>()) {
    constexpr auto dst = gs::field_named<B>(std::meta::identifier_of(member));
    constexpr auto name = std::meta::identifier_of(member);
    if (s.[:member:])
      b.[:dst:] = std::make_unique<Mem>(ctx,
          (name == "draft_control" || name == "ids") ? MemKind::Shared : MemKind::Device, s.[:member:]);
  }
}

}  // namespace

const Mem& WeightBuffers::arena(uint32_t index) const {
  if (index == L::kModel) return *model;
  if (index == L::kDraft) return *draft;
  return *layers.at(index - L::kFirstLayer);
}

PersistentBuffers::PersistentBuffers(l0::Context& ctx, const PersistentSizes& s)
    : control(ctx, MemKind::Shared, s.control), full_k(ctx, MemKind::Device, s.full_k),
      full_v(ctx, MemKind::Device, s.full_v), ring_k(ctx, MemKind::Device, s.ring_k),
      ring_v(ctx, MemKind::Device, s.ring_v), conv_k(ctx, MemKind::Device, s.conv_k),
      conv_v(ctx, MemKind::Device, s.conv_v), conv_attn(ctx, MemKind::Device, s.conv_attn),
      conv_mlp(ctx, MemKind::Device, s.conv_mlp) {
  if (s.full_k_scale) {
    full_k_scale = std::make_unique<Mem>(ctx, MemKind::Device, s.full_k_scale);
    full_v_scale = std::make_unique<Mem>(ctx, MemKind::Device, s.full_v_scale);
  }
}

ScratchBuffers::ScratchBuffers(l0::Context& ctx, const ScratchSizes& s) { allocate(ctx, *this, s); }
DraftBuffers::DraftBuffers(l0::Context& ctx, const DraftSizes& s) { allocate(ctx, *this, s); }
PrefillBuffers::PrefillBuffers(l0::Context& ctx, const PrefillSizes& s) { allocate(ctx, *this, s); }

Buffers::Buffers(l0::Context& ctx, const Sizes& s, const L& layout)
    : persistent(ctx, checked(ctx, s, layout)), scratch(ctx, s.scratch), draft(ctx, s.draft), prefill(ctx, s.prefill) {
  weights.model = std::make_unique<Mem>(ctx, MemKind::Device, s.weights.model);
  for (uint32_t l = 0; l < weights.layers.size(); ++l)
    weights.layers[l] = std::make_unique<Mem>(ctx, MemKind::Device, layout.bytes[L::kFirstLayer + l]);
  if (s.weights.draft) weights.draft = std::make_unique<Mem>(ctx, MemKind::Device, s.weights.draft);
}

void Buffers::upload(l0::Context& ctx, const loader::Inkling2Checkpoint& checkpoint, unsigned threads) {
  auto list = l0::CmdList::immediate(ctx);
  checkpoint.stream([&](uint32_t a, size_t offset, std::span<const std::byte> bytes) {
    auto* ptr = static_cast<std::byte*>(weights.arena(a).ptr());
    list.copy(ptr + offset, bytes.data(), bytes.size());
  }, bool(weights.draft), threads);
  if (draft.rope) {
    const auto& d = checkpoint.desc().draft;
    const auto rows = uint32_t(draft.rope->size() / (d.head_dim * sizeof(float)));
    const auto rope = dflash::rope_table(d.theta, d.head_dim, rows);
    list.copy(draft.rope->ptr(), rope.data(), draft.rope->size());
  }
}

void Buffers::zero(l0::Context& ctx) {
  auto list = l0::CmdList::immediate(ctx);
  const auto clear = [&](Mem& mem) { list.fill(mem.ptr(), 0, mem.size()); };
  gs::zero<Mem, PersistentSizes>(persistent, clear);
  gs::zero<Mem, DraftSizes>(draft, clear);
}

}  // namespace runtime::inkling2
