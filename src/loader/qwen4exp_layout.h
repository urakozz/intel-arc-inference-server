#pragma once
#include <cstddef>
#include <cstdint>

#include "loader/small_layout.h"
#include "model/qwen4exp.h"

// Spec 21b: Qwen3.8-Flash-Next on the card as device-free arithmetic - the ONE home of every weight
// allocation's size and of every offset inside a packed block, header-only so the planner
// (runtime/qwen4exp/qwen4exp_sizes.h, no loader library), the loader (loader/qwen4exp_loader.cc allocates
// exactly these) and 21c's kernels' host half share it.
//
// Per layer (model::Qwen4ExpDesc's shapes; the forms are the checkpoint's - ours / Intel's):
//
//   hc_attn, hc_mlp   one allocation per gated residual (Q4HcOffsets):
//                       down||inject  bf16 gemv_bf16 tiles {K 10240, N 336}: down's 320 rows, block_inject's 4,
//                                     zero rows 324..335                                   6,881,280 B
//                       up            bf16 tiles {K 320, N 10240}                          6,553,600 B
//                       norm          fp32 (1 + w) [10240]                                    40,960 B
//                     13,475,840 B each, 26,951,680 B a layer
//   GDN layers        qkv||z {2560, 16384} (S 1), a||b bf16 tiles {2560, 128} (96 used), out_proj
//                     {6144, 2560}: int4 GPTQ layout 0 (words [K/8][N] + f16 scales [K/64][N]) - ours - or
//                     bf16 tiles - Intel's; the small block make_small_layout(2560, 10240, 48) (conv taps
//                     fp32, -exp(A_log), dt_bias fp32, the gated norm plain w bf16): 164,480 B
//   QSA layers        q||gate||k||v {2560, 13312} (k at column 12288, v at 12800), the indexer bf16 tiles
//                     {2560, 640}, o_proj {6144, 2560} (int4 layout 0 or bf16 tiles); the small block
//                     (Q4QsaSmall): q_norm, k_norm [256], the indexer's q / k norms [128], fp32 (1 + w)
//   router            bf16 tiles {2560, 528}: rows 0..511 mlp.gate, row 512 shared_expert_gate, 513..527 zero
//   gate_up, down     int4 layout-1 blocks, block e at q4_gate_up_offset(d, e) = e x 1,740,800 B
//                     (cols_interleave16(gate, up) {2560, 1280}) and q4_down_offset(d, e) = e x 870,400 B
//                     ({640, 2560}): 512 x 2,611,200 = 1,336,934,400 B a layer. Each expert's two blocks are
//                     ONE contiguous range each (spec 22: its host mirror copies or reads one range per
//                     expert; its indirection table replaces exactly these two offsets)
//   shared            ONE allocation: ours int4 - a gate||up layout-1 block then a down block, an expert's
//                     shapes (2,611,200 B); Intel's bf16 - tiles {2560, 1280} (gate||up interleave16) then
//                     {640, 2560} (9,830,400 B)
//   ple (layer 1)     key||value bf16 tiles {2560, 12800}, then norm_key, norm_query, norm_conv fp32 (1 + w)
//                     [10240] each, then the conv taps fp32 [10240][4] (Q4PleOffsets): 65,822,720 B
//
// Top level: embed bf16 [vocab][hidden] 1,271,398,400 B (device 0, gathered); the final mixer (down {10240,
// 320} + up + norm: 13,148,160 B) and lm_head (bf16 tiles 1,271,398,400 B, or spec 9's int8 + fp32 row
// scales 636,692,480 B) on the last device; the MTP head (when loaded, the last device): its QSA layer at
// bf16 dense / shared arms with int4 g64 experts, its fc block (Q4MtpFcOffsets) and its own final mixer:
// 1,518,728,192 B (derived; qwen4exp_repack_test pins the formula). The RoPE table
// fp32 [max_len][2][32] on every device holding a QSA layer (the descriptor's rope_table_bytes).
// The PLE table is not device memory: 16 host-USM ranges (loader/qwen4exp_ple.h).
namespace loader {

inline constexpr uint32_t kQ4TileU32 = 136;   // layout 1: 128 nibble + 8 scale u32 per tile

inline constexpr size_t q4_layout1_bytes(uint32_t K, uint32_t N) { return size_t(N / 16) * (K / 64) * kQ4TileU32 * 4; }
inline constexpr size_t q4_layout0_bytes(uint32_t K, uint32_t N) {
  return size_t(K / 8) * N * 4 + size_t(K / 64) * N * 2;
}
inline constexpr size_t q4_bf16_bytes(uint32_t K, uint32_t N) { return size_t(K) * N * 2; }

// The descriptor of the MTP head's layer: one QSA layer whose dense projections and shared expert are bf16
// in both checkpoints (21a's expected_names: `mtp.*` is outside AutoRound's block), its routed experts int4
// g64 (shipped, or RTN-quantised at load from bf16 - spec 15e's rule, loader/rtn.h).
inline model::Qwen4ExpDesc q4_mtp_desc(const model::Qwen4ExpDesc& d) {
  model::Qwen4ExpDesc m = d;
  m.forms.dense = model::Q4Form::Bf16;
  m.forms.shared = model::Q4Form::Bf16;
  return m;
}

// A linear's device bytes in the descriptor's form (int4: layout 0 words + scales; bf16: tiles).
inline size_t q4_linear_bytes(const model::Qwen4ExpDesc& d, model::Q4LinearId id) {
  const model::Q4Linear l = d.linear(id);
  return l.kind == model::WeightKind::Int4 ? q4_layout0_bytes(l.shape.K, l.shape.N) : q4_bf16_bytes(l.shape.K, l.shape.N);
}

// --- the routed experts: the ONE address formula ------------------------------------------------------------
inline size_t q4_gate_up_block_bytes(const model::Qwen4ExpDesc& d) { return q4_layout1_bytes(d.hidden, 2 * d.moe_inter); }
inline size_t q4_down_block_bytes(const model::Qwen4ExpDesc& d) { return q4_layout1_bytes(d.moe_inter, d.hidden); }
inline size_t q4_gate_up_offset(const model::Qwen4ExpDesc& d, uint32_t e) { return size_t(e) * q4_gate_up_block_bytes(d); }
inline size_t q4_down_offset(const model::Qwen4ExpDesc& d, uint32_t e) { return size_t(e) * q4_down_block_bytes(d); }

// --- a hyper-connection block (one gated residual, or the final mixer: no inject rows) ----------------------
struct Q4HcOffsets {
  uint32_t down_n = 0;                        // 336 (inject) or 320 (the final mixer)
  size_t down = 0, up = 0, norm = 0, total = 0;   // byte offsets; total = the allocation
};
inline Q4HcOffsets q4_hc_offsets(const model::Qwen4ExpDesc& d, bool inject) {
  Q4HcOffsets o;
  o.down_n = inject ? d.hc_down_rows() : d.hc_low;
  o.down = 0;
  o.up = o.down + q4_bf16_bytes(d.hc_n(), o.down_n);
  o.norm = o.up + q4_bf16_bytes(d.hc_low, d.hc_n());
  o.total = o.norm + size_t(d.hc_n()) * 4;
  return o;
}

// --- the QSA small block: fp32 (1 + w) -----------------------------------------------------------------------
struct Q4QsaSmall {
  size_t q_norm = 0, k_norm = 0, idx_q_norm = 0, idx_k_norm = 0, total = 0;   // byte offsets
};
inline Q4QsaSmall q4_qsa_small(const model::Qwen4ExpDesc& d) {
  Q4QsaSmall s;
  s.q_norm = 0;
  s.k_norm = s.q_norm + size_t(d.head_dim) * 4;
  s.idx_q_norm = s.k_norm + size_t(d.head_dim) * 4;
  s.idx_k_norm = s.idx_q_norm + size_t(d.idx_dim) * 4;
  s.total = s.idx_k_norm + size_t(d.idx_dim) * 4;
  return s;
}

// --- the GDN small block: Qwen3.8's (loader/small_layout.h), at this family's widths ------------------------
inline SmallLayout q4_gdn_small(const model::Qwen4ExpDesc& d) {
  return make_small_layout(d.hidden, d.conv_rows(), d.gdn_v_heads);
}

// --- the PLE layer's block -----------------------------------------------------------------------------------
struct Q4PleOffsets {
  size_t kv = 0, norm_key = 0, norm_query = 0, norm_conv = 0, conv = 0, total = 0;   // byte offsets
};
inline Q4PleOffsets q4_ple_offsets(const model::Qwen4ExpDesc& d) {
  Q4PleOffsets o;
  o.kv = 0;
  o.norm_key = o.kv + q4_bf16_bytes(d.ple_e(), d.ple_kv_n());
  o.norm_query = o.norm_key + size_t(d.hc_n()) * 4;
  o.norm_conv = o.norm_query + size_t(d.hc_n()) * 4;
  o.conv = o.norm_conv + size_t(d.hc_n()) * 4;
  o.total = o.conv + size_t(d.hc_n()) * d.ple_conv_taps * 4;
  return o;
}

// --- the shared expert: one allocation, gate||up then down ---------------------------------------------------
inline size_t q4_shared_gate_up_bytes(const model::Qwen4ExpDesc& d) {
  return d.forms.shared == model::Q4Form::Int4 ? q4_layout1_bytes(d.hidden, 2 * d.shared_inter)
                                               : q4_bf16_bytes(d.hidden, 2 * d.shared_inter);
}
inline size_t q4_shared_down_bytes(const model::Qwen4ExpDesc& d) {
  return d.forms.shared == model::Q4Form::Int4 ? q4_layout1_bytes(d.shared_inter, d.hidden)
                                               : q4_bf16_bytes(d.shared_inter, d.hidden);
}

// --- the MTP head's fc block: fc_embedding | fc_hidden tiles, pre_fc_norm_embedding | pre_fc_norm_hidden ----
struct Q4MtpFcOffsets {
  size_t fc_embedding = 0, fc_hidden = 0, norm_embedding = 0, norm_hidden = 0, total = 0;   // byte offsets
};
inline Q4MtpFcOffsets q4_mtp_fc_offsets(const model::Qwen4ExpDesc& d) {
  Q4MtpFcOffsets o;
  o.fc_embedding = 0;
  o.fc_hidden = q4_bf16_bytes(d.hidden, d.hidden);
  o.norm_embedding = o.fc_hidden + q4_bf16_bytes(d.hidden, d.hidden);
  o.norm_hidden = o.norm_embedding + size_t(d.hidden) * 4;
  o.total = o.norm_hidden + size_t(d.hc_n()) * 4;
  return o;
}

// --- one layer's device bytes ----------------------------------------------------------------------------------
struct Q4LayerBytes {
  size_t hc_attn = 0, hc_mlp = 0;
  size_t gdn_qkvz = 0, gdn_ab = 0, gdn_out = 0, gdn_small = 0;   // GDN layers
  size_t qsa_qkvg = 0, qsa_idx = 0, qsa_o = 0, qsa_small = 0;    // QSA layers
  size_t router = 0, gate_up = 0, down = 0, shared_gate_up = 0, shared_down = 0;
  size_t ple = 0;                                                // the PLE layer
  size_t total() const {
    return hc_attn + hc_mlp + gdn_qkvz + gdn_ab + gdn_out + gdn_small + qsa_qkvg + qsa_idx + qsa_o + qsa_small + router +
           gate_up + down + shared_gate_up + shared_down + ple;
  }
  size_t experts() const { return gate_up + down; }
  size_t shared() const { return shared_gate_up + shared_down; }
};

namespace detail {
inline Q4LayerBytes q4_layer_bytes_of(const model::Qwen4ExpDesc& d, bool qsa, bool ple) {
  using model::Q4LinearId;
  Q4LayerBytes b;
  b.hc_attn = b.hc_mlp = q4_hc_offsets(d, true).total;
  if (qsa) {
    b.qsa_qkvg = q4_linear_bytes(d, Q4LinearId::QsaQkvg);
    b.qsa_idx = q4_linear_bytes(d, Q4LinearId::QsaIdx);
    b.qsa_o = q4_linear_bytes(d, Q4LinearId::QsaO);
    b.qsa_small = q4_qsa_small(d).total;
  } else {
    b.gdn_qkvz = q4_linear_bytes(d, Q4LinearId::GdnQkvz);
    b.gdn_ab = q4_linear_bytes(d, Q4LinearId::GdnAb);
    b.gdn_out = q4_linear_bytes(d, Q4LinearId::GdnOut);
    b.gdn_small = q4_gdn_small(d).gdn_block_bytes;
  }
  b.router = q4_linear_bytes(d, Q4LinearId::Router);
  b.gate_up = q4_gate_up_block_bytes(d) * d.experts;
  b.down = q4_down_block_bytes(d) * d.experts;
  b.shared_gate_up = q4_shared_gate_up_bytes(d);
  b.shared_down = q4_shared_down_bytes(d);
  if (ple) b.ple = q4_ple_offsets(d).total;
  return b;
}
}  // namespace detail

inline Q4LayerBytes q4_layer_bytes(const model::Qwen4ExpDesc& d, uint32_t layer) {
  return detail::q4_layer_bytes_of(d, d.is_qsa(layer), layer == d.ple_layer);
}
// The MTP head's layer (QSA, bf16 dense and shared, int4 experts, no PLE).
inline Q4LayerBytes q4_mtp_layer_bytes(const model::Qwen4ExpDesc& d) {
  return detail::q4_layer_bytes_of(q4_mtp_desc(d), true, false);
}

inline size_t q4_embed_bytes(const model::Qwen4ExpDesc& d) { return size_t(d.vocab) * d.hidden * 2; }
inline size_t q4_lm_head_bytes(const model::Qwen4ExpDesc& d, bool int8) {
  return int8 ? size_t(d.vocab) * d.hidden + size_t(d.vocab) * 4 : size_t(d.vocab) * d.hidden * 2;
}
inline size_t q4_final_mixer_bytes(const model::Qwen4ExpDesc& d) { return q4_hc_offsets(d, false).total; }
inline size_t q4_mtp_bytes(const model::Qwen4ExpDesc& d) {
  return q4_mtp_layer_bytes(d).total() + q4_mtp_fc_offsets(d).total + q4_final_mixer_bytes(d);
}

// A device's weights under a placement (the RoPE table excluded): its layers, the embedding on device 0,
// the final mixer, lm_head and (mtp) the MTP head on the last device.
inline size_t q4_device_weight_bytes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev,
                                     bool int8_head, bool mtp) {
  size_t b = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) b += q4_layer_bytes(d, l).total();
  if (dev == 0) b += q4_embed_bytes(d);
  if (dev + 1 == p.devices) {
    b += q4_final_mixer_bytes(d) + q4_lm_head_bytes(d, int8_head);
    if (mtp) b += q4_mtp_bytes(d);
  }
  return b;
}
// Whether device `dev` holds a QSA layer (and so a RoPE table): one of its layers, or the MTP head's.
inline bool q4_device_has_qsa(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, bool mtp) {
  if (mtp && dev + 1 == p.devices) return true;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l)
    if (d.is_qsa(l)) return true;
  return false;
}

// What ONE decode token reads of layer `layer` (derived; spec 21 §3's rows): both HCs, the mixer's
// projections and small block, the router, top_k routed experts' blocks, the shared expert, and on the PLE
// layer its key||value projection and norms (the 16 table rows are host reads, not counted here).
inline size_t q4_layer_read_per_token(const model::Qwen4ExpDesc& d, uint32_t layer) {
  const Q4LayerBytes b = q4_layer_bytes(d, layer);
  return b.hc_attn + b.hc_mlp + b.gdn_qkvz + b.gdn_ab + b.gdn_out + b.gdn_small + b.qsa_qkvg + b.qsa_idx + b.qsa_o +
         b.qsa_small + b.router + size_t(d.top_k) * (q4_gate_up_block_bytes(d) + q4_down_block_bytes(d)) + b.shared() +
         b.ple;
}

}  // namespace loader
