#include "runtime/inkling2/inkling2_buffers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "runtime/dflash/dflash_sizes.h"

namespace runtime::inkling2 {
namespace {
using l0::Mem, l0::MemKind;
using L = loader::Inkling2WeightLayout;

const PersistentSizes& checked(l0::Context& ctx, const Sizes& s, const L& layout, const StageShare& share) {
  if (s.weights.model != layout.bytes[L::kModel] || s.weights.layers != share.layer_bytes ||
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
          (name == "draft_control" || name == "ids" || name == "tp_state") ? MemKind::Shared
          : name == "step_io"                         ? MemKind::Host
                                                      : MemKind::Device,
          s.[:member:]);
  }
}

}  // namespace

const Mem& WeightBuffers::arena(uint32_t index) const {
  if (index == L::kModel) return *model;
  if (index == L::kDraft) return *draft;
  const auto& layer = layers.at(index - L::kFirstLayer);
  contract_assert(layer != nullptr);   // a layer this card's visits read
  return *layer;
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
HandoffBuffers::HandoffBuffers(l0::Context& ctx, const HandoffSizes& s) { allocate(ctx, *this, s); }
TpBuffers::TpBuffers(l0::Context& ctx, const TpSizes& s) { allocate(ctx, *this, s); }

Buffers::Buffers(l0::Context& ctx, const Sizes& s, const L& layout, const StageShare& share)
    : persistent(ctx, checked(ctx, s, layout, share)), scratch(ctx, s.scratch), draft(ctx, s.draft),
      prefill(ctx, s.prefill), prefill_b(ctx, s.prefill_b), handoff(ctx, s.handoff), tp(ctx, s.tp) {
  weights.model = std::make_unique<Mem>(ctx, MemKind::Device, s.weights.model);
  for (uint32_t l = 0; l < weights.layers.size(); ++l)
    if (share.layers[l]) weights.layers[l] = std::make_unique<Mem>(ctx, MemKind::Device, layout.bytes[L::kFirstLayer + l]);
  if (s.weights.draft) weights.draft = std::make_unique<Mem>(ctx, MemKind::Device, s.weights.draft);
  if (s.weights.prompt_int8) weights.prompt_int8 = std::make_unique<Mem>(ctx, MemKind::Device, s.weights.prompt_int8);
  if (s.weights.candidate_head)
    weights.candidate_head = std::make_unique<Mem>(ctx, MemKind::Device, s.weights.candidate_head);
}

void Buffers::upload(l0::Context& ctx, const loader::Inkling2Checkpoint& checkpoint, unsigned threads) {
  const std::pair<l0::Context*, Buffers*> card{&ctx, this};
  upload({&card, 1}, checkpoint, threads);
}

void Buffers::upload(std::span<const std::pair<l0::Context*, Buffers*>> cards,
                     const loader::Inkling2Checkpoint& checkpoint, unsigned threads) {
  std::vector<l0::CmdList> lists;
  bool draft = false;
  for (const auto& [ctx, b] : cards) {
    lists.push_back(l0::CmdList::immediate(*ctx));
    draft = draft || b->weights.draft;
  }
  checkpoint.stream([&](uint32_t a, size_t offset, std::span<const std::byte> bytes) {
    for (size_t i = 0; i < cards.size(); ++i) {
      const auto& w = cards[i].second->weights;
      if (a == L::kDraft ? !w.draft : a != L::kModel && !w.layers[a - L::kFirstLayer]) continue;
      lists[i].copy(static_cast<std::byte*>(w.arena(a).ptr()) + offset, bytes.data(), bytes.size());
    }
  }, draft, threads);
  upload_rope(cards, lists, checkpoint);
}

void Buffers::upload(std::span<const std::pair<l0::Context*, Buffers*>> cards, std::span<const TpWeights* const> split,
                     const loader::Inkling2Checkpoint& checkpoint, unsigned threads) {
  std::vector<l0::CmdList> lists;
  bool draft = false;
  for (const auto& [ctx, b] : cards) {
    lists.push_back(l0::CmdList::immediate(*ctx));
    draft = draft || b->weights.draft;
  }
  checkpoint.stream([&](uint32_t a, size_t offset, std::span<const std::byte> bytes) {
    for (size_t i = 0; i < cards.size(); ++i) {
      const auto& w = cards[i].second->weights;
      if (a == L::kDraft && !w.draft) continue;
      auto* base = static_cast<std::byte*>(w.arena(a).ptr());
      tp_write(*split[i], a, offset, bytes, [&](size_t at, std::span<const std::byte> piece) {
        lists[i].copy(base + at, piece.data(), piece.size());
      });
    }
  }, draft, threads);
  upload_rope(cards, lists, checkpoint);
}

void Buffers::upload_rope(std::span<const std::pair<l0::Context*, Buffers*>> cards, std::vector<l0::CmdList>& lists,
                          const loader::Inkling2Checkpoint& checkpoint) {
  for (size_t i = 0; i < cards.size(); ++i) {
    const auto& rope = cards[i].second->draft.rope;
    if (!rope) continue;
    const auto& d = checkpoint.desc().draft;
    if (checkpoint.dflash()) {   // the DFlash 2 drafter's kernels take inv_freq and compute the angles (ROPE_INV)
      std::vector<float> inv(d.head_dim / 2);
      for (uint32_t j = 0; j < inv.size(); ++j) inv[j] = float(1.0 / std::pow(d.theta, double(2 * j) / d.head_dim));
      lists[i].copy(rope->ptr(), inv.data(), rope->size());
      continue;
    }
    const auto rows = uint32_t(rope->size() / (d.head_dim * sizeof(float)));
    const auto table = dflash::rope_table(d.theta, d.head_dim, rows);
    lists[i].copy(rope->ptr(), table.data(), rope->size());
  }
}

void Buffers::zero(l0::Context& ctx, uint32_t written, uint32_t max_len, uint32_t full_slots) {
  auto list = l0::CmdList::immediate(ctx);
  // The position-indexed caches, [slot][max_len][row], and their slot counts.
  const std::array<std::pair<const Mem*, uint32_t>, 6> positioned{
      {{&persistent.full_k, full_slots}, {&persistent.full_v, full_slots}, {persistent.full_k_scale.get(), full_slots},
       {persistent.full_v_scale.get(), full_slots}, {draft.draft_full_k.get(), 1}, {draft.draft_full_v.get(), 1}}};
  const auto clear = [&](Mem& mem) {
    const auto it = std::find_if(positioned.begin(), positioned.end(), [&](const auto& p) { return p.first == &mem; });
    if (it == positioned.end() || written >= max_len) return list.fill(mem.ptr(), 0, mem.size());
    const size_t slot = mem.size() / it->second, row = slot / max_len;
    for (uint32_t i = 0; i < it->second && written; ++i)
      list.fill(static_cast<std::byte*>(mem.ptr()) + i * slot, 0, size_t(written) * row);
  };
  gs::zero<Mem, PersistentSizes>(persistent, clear);
  gs::zero<Mem, DraftSizes>(draft, clear);
  gs::zero<Mem, TpSizes>(tp, [&](Mem& mem) {
    if (mem.kind() == MemKind::Shared) std::memset(mem.ptr(), 0, mem.size());
    else list.fill(mem.ptr(), 0, mem.size());
  });
}

}  // namespace runtime::inkling2
