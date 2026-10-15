#pragma once
#include <cstddef>
#include <cstdint>

#include "loader/int4_layout.h"
#include "loader/lm_head_int8.h"   // lm_head_int8_bytes (header-inline)
#include "model/muse_glimmer.h"

// Spec 23b: Muse-Glimmer-30B on the card as device-free arithmetic - the ONE home of every weight allocation's
// size, header-only so the planner (runtime/muse/muse_sizes.h, no loader library), the loader (loader/
// muse_loader.cc allocates exactly these) and the kernels' host half share it (kolibri1_layout.h's arrangement).
//
// Per layer (model::MuseGlimmerDesc's shapes; derived, the loader asserts them), int4 g64 in the GEMV layout of
// each row (desc.linear(id).shape.layout; 0 today - words [K/8][N] + f16 scales [K/64][N]; layout 1 has the same
// count at these shapes):
//
//   qgkv     q || gate || k || v  6656 x 8704    30,777,344 B (28,966,912 words + 1,810,432 scales)
//   oproj    o_proj               4096 x 6656    14,483,456 B
//   gate_up  gate || up           6656 x 39936  141,213,696 B
//   down     down_proj           19968 x 6656    70,606,848 B
//            int4 a layer                       257,081,344 B (x 52 = 13,368,229,888 B, spec 23 §2's 13.368 GB)
//   norms    fp32 [4][hidden] (input | post_attention | pre_ff | post_ff, the bf16 w widened)  106,496 B
//   total    257,187,840 B a layer
//
// Top level: embed bf16 [vocab][hidden] 2,689,662,976 B (device 0, gathered: one 13,312 B row a token); the
// final norm fp32 [hidden] 26,624 B and lm_head - int8 + fp32 row scales 1,345,639,680 B (spec 9, the default)
// or bf16 tiled 2,689,662,976 B - on the last device. One card, int8 head: 17,409,096,960 B (spec 23 §2's
// 17.41 GB). Read per decoded token: every layer + the final norm + the head = 14,719,433,984 B (14.72 GB).
//
// The attention state (runtime/muse/muse_sizes.h plans it; the formulas live here so 23c's buffers share them):
// the 13 full layers' KV, bf16 [max_len][kv_heads][head_dim] K and V - 13,312 B a position; the 39 sliding
// layers' rings, bf16 [kRing 4096][kv_heads][head_dim] K and V - 4,194,304 B a layer (163,577,856 B for 39).
namespace loader {

struct MuseLayerBytes {
  size_t qgkv = 0, oproj = 0, gate_up = 0, down = 0, norms = 0;
  size_t int4() const { return qgkv + oproj + gate_up + down; }
  size_t total() const { return int4() + norms; }
};

// One fused int4 linear's device bytes in its row's layout.
inline size_t muse_linear_bytes(const model::MuseGlimmerDesc& d, model::MuseLinearId id) {
  const model::GemvShape s = d.linear(id).shape;
  return s.layout == 0 ? int4_layout0_bytes(s.K, s.N) : int4_layout1_bytes(s.K, s.N);
}

inline MuseLayerBytes muse_layer_bytes(const model::MuseGlimmerDesc& d) {
  MuseLayerBytes b;
  b.qgkv = muse_linear_bytes(d, model::MuseLinearId::QGKV);
  b.oproj = muse_linear_bytes(d, model::MuseLinearId::OProj);
  b.gate_up = muse_linear_bytes(d, model::MuseLinearId::GateUp);
  b.down = muse_linear_bytes(d, model::MuseLinearId::Down);
  b.norms = size_t(d.norm_floats()) * 4;
  return b;
}
inline size_t muse_embed_bytes(const model::MuseGlimmerDesc& d) { return size_t(d.vocab) * d.hidden * 2; }
inline size_t muse_lm_head_bytes(const model::MuseGlimmerDesc& d, bool int8) {
  return int8 ? lm_head_int8_bytes(d.hidden, d.vocab) : size_t(d.vocab) * d.hidden * 2;
}
inline size_t muse_final_norm_bytes(const model::MuseGlimmerDesc& d) { return size_t(d.hidden) * 4; }

inline size_t muse_ring_slots() { return model::MuseGlimmerDesc::kRing; }
// K and V of one sliding layer's ring.
inline size_t muse_ring_bytes_per_layer(const model::MuseGlimmerDesc& d) {
  return muse_ring_slots() * d.kv_n() * 2 * 2;
}
// K and V of every full layer, one position.
inline size_t muse_kv_bytes_per_position(const model::MuseGlimmerDesc& d) {
  return size_t(d.full_layers()) * d.kv_n() * 2 * 2;
}
// K and V of ONE full layer at max_len (the per-layer term of the split by bytes): max_len x 1024 B.
inline size_t muse_full_layer_kv_bytes(const model::MuseGlimmerDesc& d, uint32_t max_len) {
  return size_t(max_len) * d.kv_n() * 2 * 2;
}

// A device's weights under a placement: its layers, the embedding on device 0, the final norm and lm_head on
// the last device.
inline size_t muse_device_weight_bytes(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t dev,
                                       bool int8_head) {
  size_t b = size_t(p.count(dev)) * muse_layer_bytes(d).total();
  if (dev == 0) b += muse_embed_bytes(d);
  if (dev + 1 == p.devices) b += muse_final_norm_bytes(d) + muse_lm_head_bytes(d, int8_head);
  return b;
}

// What ONE decode token reads of a layer (derived): every weight byte of it (dense).
inline size_t muse_layer_read_per_token(const model::MuseGlimmerDesc& d) { return muse_layer_bytes(d).total(); }
// ... and of the whole model's weights (the KV and the windows apart): the layers, the final norm, the head.
inline size_t muse_read_per_token(const model::MuseGlimmerDesc& d, bool int8_head) {
  return size_t(d.layers) * muse_layer_read_per_token(d) + muse_final_norm_bytes(d) + muse_lm_head_bytes(d, int8_head);
}

}  // namespace loader
