#pragma once
// Spec 17a: the tensor-parallel view of a dense GDN-hybrid model (Qwen3.8) - spec 17 §R2.5 - from its
// `ModelDesc` through tp_rank / tp_size, for probe_tp (the all-reduce's count and row size; the per-card linears the
// step arm times). Megatron split: column-parallel qkv‖z, a‖b, q‖k‖v, gate‖up and the head's vocabulary;
// row-parallel out_proj, o_proj and down (each followed by one all-reduce of an fp32 row of the hidden width a
// row); norms and the embedding replicated. Head counts halve per rank; every K slice is a whole number of
// 64-groups; the fused layouts are re-fused per card from that card's heads (the part widths below), never sliced
// from the one-card buffer (§4.2's correction). Nothing here is a constant of the probe: every number comes from
// the descriptor. Header-only; tests/kernels/tp_shapes_test.cc holds it to §R2.5's table.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/model_desc.h"

namespace tp {

struct Gemv {
  uint32_t K = 0, N = 0, S = 1, layout = 0;
};

struct Linear {
  const char* key;      // probe name: qkvz out qkv o gateup down head ab
  const char* name;
  model::LinearId id;
  model::WeightKind kind;
  Gemv one;             // the one-card shape (the descriptor's row)
  Gemv card;            // this rank's shape
  uint32_t layers;      // launches a token (1 for the head)
  bool row_parallel;    // an all-reduce follows it
  uint32_t real_n_card; // columns before a zero-pad (a‖b: 2 x v-heads per card); = card.N otherwise
};

struct Parts3 { uint32_t q = 0, k = 0, v = 0; };
struct Parts4 { uint32_t q = 0, k = 0, v = 0, z = 0; };

struct Shapes {
  uint32_t tp_rank = 0, tp_size = 1;
  uint32_t hidden = 0;
  uint32_t calls = 0;           // all-reduces a token: 2 a layer (0 at tp 1)
  uint32_t head_exchanges = 0;  // the head's argmax exchange (spec 17 §4.4): 1 a token (0 at tp 1)
  uint32_t row_bytes = 0;       // one all-reduce row: hidden x fp32
  uint32_t vocab_offset = 0;    // this rank's first lm_head row
  Parts4 qkvz_parts;            // the card's re-fused qkv‖z: q, k (k-heads), v, z (v-heads) x 128
  Parts3 qkv_parts;             // the card's re-fused q‖k‖v: q‖gate (q-heads x 2 x 256), k, v (kv-heads x 256)
  uint32_t conv_channels_card = 0, gdn_k_heads_card = 0, gdn_v_heads_card = 0;
  uint32_t fa_q_heads_card = 0, fa_kv_heads_card = 0;
  uint32_t ffn_card = 0;        // intermediate / tp: gate (and up) columns per card
  std::vector<Linear> linears;
  uint64_t token_bytes(uint32_t M) const { return uint64_t(M) * calls * row_bytes; }
};

inline const model::ModelDesc& model_by_name(const std::string& s) {
  if (s == "qwen38") return model::qwen38();
  throw std::runtime_error("probe_tp: --model '" + s + "' (qwen38: the dense GDN hybrid of spec 17 §R2; Agnes 3.0 Flash "
                           "was dropped 2026-11-07)");
}

namespace detail {
inline void need(bool ok, const model::ModelDesc& d, const std::string& what) {
  if (!ok) throw std::runtime_error("tp_shapes(" + d.name + "): " + what);
}
inline Gemv of(const model::GemvShape& s) { return {s.K, s.N, s.S, s.layout}; }
}  // namespace detail

inline Shapes tp_shapes(const model::ModelDesc& d, uint32_t tp_rank, uint32_t tp_size) {
  using detail::need;
  using model::LinearId;
  need(tp_size >= 1, d, "tp_size must be >= 1");
  need(tp_rank < tp_size, d, "tp_rank " + std::to_string(tp_rank) + " >= tp_size " + std::to_string(tp_size));
  need(!d.is_moe(), d, "a MoE model is not split this way (spec 17 §4.3); dense GDN hybrids only");
  const uint32_t t = tp_size;
  need(d.gdn_k_heads % t == 0 && d.gdn_v_heads % t == 0, d, "GDN heads do not split evenly");
  need(d.fa_q_heads % t == 0 && d.fa_kv_heads % t == 0, d, "FA heads do not split evenly");
  need(d.intermediate % t == 0 && (d.intermediate / t) % 16 == 0, d,
       "the FFN width does not split into whole 16-column interleave blocks");
  const model::FusedLinear& head = d.lm_head(model::WeightKind::Int8);
  need(head.shape.N % (t * 64) == 0, d, "the head's rows do not split into whole 64-column tiles");

  Shapes s;
  s.tp_rank = tp_rank;
  s.tp_size = t;
  s.hidden = d.hidden;
  s.calls = t > 1 ? 2 * d.layers : 0;
  s.head_exchanges = t > 1 ? 1 : 0;
  s.row_bytes = d.hidden * 4;
  s.gdn_k_heads_card = d.gdn_k_heads / t;
  s.gdn_v_heads_card = d.gdn_v_heads / t;
  s.fa_q_heads_card = d.fa_q_heads / t;
  s.fa_kv_heads_card = d.fa_kv_heads / t;
  constexpr uint32_t kGdnDim = model::Qwen35::kGdnHeadDim, kFaDim = model::Qwen35::kFaHeadDim;
  s.qkvz_parts = {s.gdn_k_heads_card * kGdnDim, s.gdn_k_heads_card * kGdnDim, s.gdn_v_heads_card * kGdnDim,
                  s.gdn_v_heads_card * kGdnDim};
  s.conv_channels_card = s.qkvz_parts.q + s.qkvz_parts.k + s.qkvz_parts.v;
  s.qkv_parts = {2 * s.fa_q_heads_card * kFaDim, s.fa_kv_heads_card * kFaDim, s.fa_kv_heads_card * kFaDim};
  s.ffn_card = d.intermediate / t;
  const uint32_t head_n = head.shape.N / t;
  s.vocab_offset = tp_rank * head_n;

  auto add = [&](const char* key, const char* name, LinearId id, model::WeightKind kind, const model::GemvShape& one,
                 Gemv card, uint32_t layers, bool rowp, uint32_t real_n) {
    need(card.K % 64 == 0 && (card.K / 64) % card.S == 0, d,
         std::string(key) + ": K " + std::to_string(card.K) + " is not S " + std::to_string(card.S) +
             " slices of whole 64-groups");
    s.linears.push_back({key, name, id, kind, detail::of(one), card, layers, rowp, real_n});
  };
  const auto& qkvz = d.shape(LinearId::QkvZ);
  const auto& ab = d.shape(LinearId::AB);
  const auto& out = d.shape(LinearId::OutProj);
  const auto& qkv = d.shape(LinearId::Qkv);
  const auto& o = d.shape(LinearId::OProj);
  const auto& gu = d.shape(LinearId::GateUp);
  const auto& dn = d.shape(LinearId::Down);
  const uint32_t qkvz_n = s.qkvz_parts.q + s.qkvz_parts.k + s.qkvz_parts.v + s.qkvz_parts.z;
  const uint32_t qkv_n = s.qkv_parts.q + s.qkv_parts.k + s.qkv_parts.v;
  // column-parallel: N per card from the card's heads; row-parallel: K per card, S kept (whole groups checked)
  add("qkvz", "qkv‖z", LinearId::QkvZ, model::WeightKind::Int4, qkvz, {qkvz.K, qkvz_n, qkvz.S, qkvz.layout},
      d.gdn_layers, false, qkvz_n);
  add("out", "out_proj", LinearId::OutProj, model::WeightKind::Int4, out, {out.K / t, out.N, out.S, out.layout},
      d.gdn_layers, t > 1, out.N);
  add("qkv", "q‖k‖v", LinearId::Qkv, model::WeightKind::Int4, qkv, {qkv.K, qkv_n, qkv.S, qkv.layout}, d.fa_layers,
      false, qkv_n);
  add("o", "o_proj", LinearId::OProj, model::WeightKind::Int4, o, {o.K / t, o.N, o.S, o.layout}, d.fa_layers, t > 1,
      o.N);
  add("gateup", "gate‖up", LinearId::GateUp, model::WeightKind::Int4, gu, {gu.K, 2 * s.ffn_card, gu.S, gu.layout},
      d.layers, false, 2 * s.ffn_card);
  add("down", "down", LinearId::Down, model::WeightKind::Int4, dn, {s.ffn_card, dn.N, dn.S, dn.layout}, d.layers,
      t > 1, dn.N);
  add("head", "lm_head int8", LinearId::LmHead, model::WeightKind::Int8, head.shape,
      {head.shape.K, head_n, 1, head.shape.layout}, 1, false, head_n);
  // a‖b: 2 x v-heads per card, zero-padded to the bf16 GEMV's tile (the one-card padded width)
  add("ab", "a‖b bf16", LinearId::AB, model::WeightKind::Bf16, ab, {ab.K, ab.N, ab.S, ab.layout}, d.gdn_layers,
      false, 2 * s.gdn_v_heads_card);
  if (t == 1)
    for (Linear& l : s.linears) l.row_parallel = false;
  return s;
}

}  // namespace tp
