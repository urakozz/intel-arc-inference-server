#pragma once
// Spec 17b: the two TP cards' persistent state in runtime::Engine's ONE-CARD host layouts (spec 17 §4.7) - host
// arithmetic over host copies, so tests/runtime/tp_gather_test.cc checks it on any host and runtime::TpEngine's
// save_state / save_kv are a download per card plus this. Spec 17d: the scatter (the restore) is its exact inverse,
// and with the MTP head the one-card KV layout carries the head's layer after the last FA layer (splice_head_kv /
// split_head_kv) - the head is whole on each card, so its bytes are one card's own.
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

// Spec 17d: gather_state's inverse - the one-card state bytes `in` into the two cards' gdn_state and conv_ring.
inline void scatter_state(const model::ModelDesc& card, const uint8_t* in, uint8_t* const gdn[2], uint8_t* const conv[2]) {
  constexpr size_t kD = model::Qwen35::kGdnHeadDim;
  const size_t head = kD * kD * 4, card_layer = size_t(card.gdn_v_heads) * head;
  for (uint32_t l = 0; l < card.gdn_layers; ++l)
    for (uint32_t r = 0; r < model::kTpSize; ++r) {
      std::memcpy(gdn[r] + l * card_layer, in, card_layer);
      in += card_layer;
    }
  const size_t kq = size_t(card.gdn_k_heads) * kD * 2, v = size_t(card.gdn_v_heads) * kD * 2;
  const size_t card_row = 2 * kq + v;
  const size_t rows = size_t(card.gdn_layers) * PersistentDims::kConvRing;
  for (size_t i = 0; i < rows; ++i)
    for (const auto& [off, bytes] : {std::pair<size_t, size_t>{0, kq}, {kq, kq}, {2 * kq, v}})
      for (uint32_t r = 0; r < model::kTpSize; ++r) {
        std::memcpy(conv[r] + i * card_row + off, in, bytes);
        in += bytes;
      }
}

// Spec 17d: gather_kv's inverse - n positions of one-card KV bytes `in` into the two cards' save_kv-layout bytes.
inline void scatter_kv(const model::ModelDesc& card, KvCache form, uint32_t n, const uint8_t* in, uint8_t* const cards[2]) {
  const size_t row = size_t(card.fa_kv_heads) * model::Qwen35::kFaHeadDim * (form == KvCache::Int8 ? 1 : 2);
  const size_t scale = form == KvCache::Int8 ? size_t(card.fa_kv_heads) * 2 : 0;
  const size_t L = card.fa_layers;
  const size_t tensor = L * n * (row + scale);
  for (uint32_t t = 0; t < 2; ++t) {
    for (const auto& [base, unit] : {std::pair<size_t, size_t>{0, row}, {L * n * row, scale}}) {
      if (unit == 0) continue;
      for (size_t i = 0; i < L * n; ++i)
        for (uint32_t r = 0; r < model::kTpSize; ++r) {
          std::memcpy(cards[r] + t * tensor + base + i * unit, in, unit);
          in += unit;
        }
    }
  }
}

// Spec 17d: the one-card KV layout with the MTP head's layer (Engine::save_kv with MTP on): per tensor (K, V) the FA
// layers' rows then the head's, then at int8 the FA layers' scales then the head's. `main` is gather_kv's output (the
// FA layers alone, one-card heads `kv_heads`), `head` the head's own bytes in Engine::save_kv's per-layer form - per
// tensor its n rows then (int8) its n scales.
struct KvSplit {
  size_t rows_main, rows_head, scales_main, scales_head;   // one tensor's four runs, bytes
  size_t main_tensor() const { return rows_main + scales_main; }
  size_t head_tensor() const { return rows_head + scales_head; }
};
inline KvSplit kv_split(uint32_t fa_layers, uint32_t kv_heads, uint32_t head_kv_heads, KvCache form, uint32_t n) {
  const size_t rb = form == KvCache::Int8 ? 1 : 2, sb = form == KvCache::Int8 ? 2 : 0;
  const size_t D = model::Qwen35::kFaHeadDim;
  return {size_t(fa_layers) * n * kv_heads * D * rb, size_t(n) * head_kv_heads * D * rb,
          size_t(fa_layers) * n * kv_heads * sb, size_t(n) * head_kv_heads * sb};
}
inline void splice_head_kv(const KvSplit& k, const uint8_t* main, const uint8_t* head, uint8_t* out) {
  for (uint32_t t = 0; t < 2; ++t) {
    const uint8_t* m = main + t * k.main_tensor();
    const uint8_t* h = head + t * k.head_tensor();
    for (const auto& [src, bytes] : {std::pair<const uint8_t*, size_t>{m, k.rows_main}, {h, k.rows_head},
                                     {m + k.rows_main, k.scales_main}, {h + k.rows_head, k.scales_head}}) {
      std::memcpy(out, src, bytes);
      out += bytes;
    }
  }
}
inline void split_head_kv(const KvSplit& k, const uint8_t* in, uint8_t* main, uint8_t* head) {
  for (uint32_t t = 0; t < 2; ++t) {
    uint8_t* m = main + t * k.main_tensor();
    uint8_t* h = head + t * k.head_tensor();
    for (const auto& [dst, bytes] : {std::pair<uint8_t*, size_t>{m, k.rows_main}, {h, k.rows_head},
                                     {m + k.rows_main, k.scales_main}, {h + k.rows_head, k.scales_head}}) {
      std::memcpy(dst, in, bytes);
      in += bytes;
    }
  }
}

}  // namespace tp
}  // namespace runtime
