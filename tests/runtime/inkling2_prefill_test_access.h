#pragma once
#include "runtime/inkling2/inkling2_engine.h"

namespace runtime::inkling2 {
struct PrefillTestAccess {
  static auto state(Engine& engine) {
    const auto& p = engine.buffers_.persistent;
    const auto& d = engine.buffers_.draft;
    return std::array{
        &p.full_k, &p.full_v, &p.ring_k, &p.ring_v, &p.conv_k, &p.conv_v, &p.conv_attn, &p.conv_mlp,
        static_cast<const l0::Mem*>(d.draft_full_k.get()), static_cast<const l0::Mem*>(d.draft_full_v.get()),
        static_cast<const l0::Mem*>(d.draft_ring_k.get()), static_cast<const l0::Mem*>(d.draft_ring_v.get())};
  }
  // `--kv-cache int8`: the full visits' fp16 scales (null with the bf16 cache).
  static std::array<const l0::Mem*, 2> scales(Engine& engine) {
    const auto& p = engine.buffers_.persistent;
    return {p.full_k_scale.get(), p.full_v_scale.get()};
  }
  static const l0::Mem& trace(Engine& engine) {
    engine.trace_prelude_ = true;
    return *engine.buffers_.prefill.prelude_trace;
  }

};
}  // namespace runtime::inkling2
