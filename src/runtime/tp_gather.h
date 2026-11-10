#pragma once
// Spec 17b: the two TP cards' persistent state in runtime::Engine's ONE-CARD host layouts (spec 17 §4.7) - host
// arithmetic over host copies, so tests/runtime/tp_gather_test.cc checks it on any host and runtime::TpEngine's
// save_state / save_kv are a download per card plus this. 17d adds the scatter (the restore) as its inverse.
//
//   state  gdn_state [gdn layers][v-heads][128][128] fp32: a layer's v-heads are card 0's then card 1's (the card's
//          heads are a contiguous range, loader::load_tp); conv_ring [gdn layers][16][conv channels] bf16: a ring
//          row's channels are q | k | v, each card 0's part then card 1's.
//   kv     per tensor (K, then V): rows [fa layers][n][kv-heads][256] (bf16, or int8), then at int8 the scales
//          [fa layers][n][kv-heads] fp16: a position's kv-heads are card 0's then card 1's. A card's own bytes come
//          in that layout at its own head count (TpEngine::save_kv downloads them so; Engine::save_kv's form).
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/buffer_sizes.h"

namespace runtime {
namespace tp {

// One-card state bytes from the two cards' gdn_state and conv_ring (each a whole allocation's host copy).
inline void gather_state(const model::ModelDesc& card, const uint8_t* const gdn[2], const uint8_t* const conv[2],
                         uint8_t* out) {
  constexpr size_t kD = model::Qwen35::kGdnHeadDim;
  const size_t head = kD * kD * 4, card_layer = size_t(card.gdn_v_heads) * head;
  for (uint32_t l = 0; l < card.gdn_layers; ++l)
    for (uint32_t r = 0; r < model::kTpSize; ++r) {
      std::memcpy(out, gdn[r] + l * card_layer, card_layer);
      out += card_layer;
    }
  const size_t kq = size_t(card.gdn_k_heads) * kD * 2, v = size_t(card.gdn_v_heads) * kD * 2;   // bytes, bf16
  const size_t card_row = 2 * kq + v;
  const size_t rows = size_t(card.gdn_layers) * PersistentDims::kConvRing;
  for (size_t i = 0; i < rows; ++i)
    for (const auto& [off, bytes] : {std::pair<size_t, size_t>{0, kq}, {kq, kq}, {2 * kq, v}})
      for (uint32_t r = 0; r < model::kTpSize; ++r) {
        std::memcpy(out, conv[r] + i * card_row + off, bytes);
        out += bytes;
      }
}

// One-card KV bytes of n positions from the two cards' save_kv-layout bytes (`card[r]`, kv_bytes at the card's heads).
inline void gather_kv(const model::ModelDesc& card, KvCache form, uint32_t n, const uint8_t* const cards[2],
                      uint8_t* out) {
  const size_t row = size_t(card.fa_kv_heads) * model::Qwen35::kFaHeadDim * (form == KvCache::Int8 ? 1 : 2);
  const size_t scale = form == KvCache::Int8 ? size_t(card.fa_kv_heads) * 2 : 0;
  const size_t L = card.fa_layers;
  const size_t tensor = L * n * (row + scale);   // one tensor's bytes on one card
  for (uint32_t t = 0; t < 2; ++t) {             // K, then V
    for (const auto& [base, unit] : {std::pair<size_t, size_t>{0, row}, {L * n * row, scale}}) {
      if (unit == 0) continue;
      for (size_t i = 0; i < L * n; ++i)
        for (uint32_t r = 0; r < model::kTpSize; ++r) {
          std::memcpy(out, cards[r] + t * tensor + base + i * unit, unit);
          out += unit;
        }
    }
  }
}

}  // namespace tp
}  // namespace runtime
