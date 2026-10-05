#include "runtime/buffer_sizes.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include "kernels/prefill/pf_kernels.h"
#include "runtime/control.h"

// Every size here is derived from `model::Qwen35` (the shared head dims and
// vocab), the model descriptor (spec 14: layer counts, intermediate, the GEMV
// table; spec 15b: hidden, the head counts and their derived widths) and `max_len`; the only
// literals are element sizes and the two shapes the kernels fix rather than
// the model (the attention partial's `+2` and argmax's chunk), both named.
// Sizes that disagree with the kernels are silent corruption on token 2 -
// tests/runtime/buffers_test.cc pins the totals with the arithmetic spelled
// out, so a change has to be argued for there. (Moved from buffers.cc unchanged by
// the max_len auto amendment: the constructors there allocate what these return.)
namespace runtime {
namespace {
using Q = model::Qwen35;

constexpr size_t kBf16 = 2, kFp32 = 4;

// Spec 15b: the per-model widths are the descriptor's - desc.gdn_conv_dim() (GDN
// conv1d channels, 10240 on Qwen3.8), desc.gdn_value_dim() and desc.fa_value_dim()
// (6144 each on Qwen3.8), desc.hidden (5120), the head counts.

// Spec 15b: `x` holds three things - prep's normed hidden row (K = hidden), the
// MTP head's two concatenated pre-fc norms ([M][2 x hidden], capture.cc's X stride)
// and SiLU(gate) x up ([M][intermediate]) - so it is the widest of them. On
// Qwen3.8 and Agnes the MLP is (17408 / 19456 > 10240), which is the size spec 14
// gave it; Ornith's 512-wide shared expert is not, and 2 x 2048 is.
uint32_t x_width(const model::ModelDesc& d) { return std::max(d.intermediate, 2 * d.hidden); }
// Prefill's `mixer_out` is the out_proj / o_proj input (GDN value dim, FA value
// dim) AND, on the fused-SiLU path, the normed [C][hidden] rows (step.cc): 6144
// on Qwen3.8 (both value dims), 4096 on Ornith.
uint32_t mixer_width(const model::ModelDesc& d) {
  return std::max({d.gdn_value_dim(), d.fa_value_dim(), d.hidden});
}

// Spec 15d: on a MoE model the table's GateUp / Down rows are the shared expert, which the
// loader puts into the expert blocks (loader/moe_layout.h), not into a linear of its own:
// nothing on the prefill walk runs them as a dense linear.
bool moe_ffn_row(const model::ModelDesc& d, model::LinearId id) {
  return d.is_moe() && (id == model::LinearId::GateUp || id == model::LinearId::Down);
}
// The widest K of a dense int4 linear the prefill walk runs - the L0 slab's rows: the MLP
// intermediate (down's K) on a dense model, as before spec 15d; the widest of the four
// mixer linears on a MoE model (Ornith: out_proj / o_proj's 4096).
uint32_t slab_k(const model::ModelDesc& d) {
  if (!d.is_moe()) return d.intermediate;
  uint32_t k = 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(model::LinearId::kCount); ++i) {
    const model::FusedLinear& fl = d.linear(static_cast<model::LinearId>(i));
    if (fl.kind == model::WeightKind::Int4 && !moe_ffn_row(d, fl.id)) k = std::max(k, fl.shape.K);
  }
  return k;
}

// attn_decode writes a running (m, l) pair plus the head accumulator for each
// (q head, KV block, token); attn_reduce combines them.
constexpr uint32_t kAttnPartStride = Q::kFaHeadDim + 2;  // 258

// argmax_stage1 gives one work-group 1024 logits and writes one (max, index)
// pair per group (plan 3 task 3).
constexpr uint32_t kArgmaxChunk = 1024;
constexpr uint32_t kArgmaxGroups = (Q::kVocab + kArgmaxChunk - 1) / kArgmaxChunk;  // 243

// `partials` is the S-split accumulator every int4 GEMV writes before its prep
// kernel sums it. It is the conservative max-S x max-N rectangle over the int4
// rows of the model table (Qwen3.8, Task 4: [S_max=8][8][34816]), not the max of
// S*N: one buffer has to fit whichever row is running. The bf16 rows are
// excluded because they do not use it - a||b writes ab_out, lm_head writes
// logits directly.
//
// `rows` is DecodeScratch::kM for decode. The prefill path passes its own
// chunk width and **S = 1** (ruling R1): split-K exists to buy hardware
// threads at M = 1, and at M = C the M tile axis already saturates the grid,
// so an [S][C][N] rectangle at S = 8 would be 1.14 TB at C = 2048 for nothing.
size_t max_int4_n(const model::ModelDesc& desc) {
  uint32_t max_n = 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(model::LinearId::kCount); ++i) {
    const model::FusedLinear& fl = desc.linear(static_cast<model::LinearId>(i));
    if (fl.kind != model::WeightKind::Int4) continue;
    if (fl.shape.N > max_n) max_n = fl.shape.N;
  }
  return max_n;
}

size_t partials_bytes(const model::ModelDesc& desc) {
  uint32_t max_s = 1;
  for (uint32_t i = 0; i < static_cast<uint32_t>(model::LinearId::kCount); ++i) {
    const model::FusedLinear& fl = desc.linear(static_cast<model::LinearId>(i));
    if (fl.kind != model::WeightKind::Int4) continue;
    if (fl.shape.S > max_s) max_s = fl.shape.S;
  }
  return size_t{max_s} * DecodeScratchDims::kM * max_int4_n(desc) * kFp32;
}

// KV blocks an attn_decode grid covers. Rounded up: a max_len that is not a
// multiple of kAttnBlock still needs a slot for its tail block.
uint32_t attn_blocks(uint32_t max_len) {
  return (max_len + DecodeScratchDims::kAttnBlock - 1) / DecodeScratchDims::kAttnBlock;
}

// One GDN state slot: fp32 [gdn_layers][v-heads][128 k][128 v] (150.99 MB on Qwen3.8).
size_t gdn_state_bytes(const model::ModelDesc& desc) {
  return size_t{desc.gdn_layers} * desc.gdn_v_heads * Q::kGdnHeadDim * Q::kGdnHeadDim * kFp32;
}
}  // namespace

KvCache default_kv_cache() {
  const char* v = std::getenv("B70_KV_CACHE");
  if (v == nullptr || *v == '\0') return kDefaultKvCache;
  return parse_kv_cache(v);
}

KvCache parse_kv_cache(const char* v) {
  if (std::strcmp(v, "bf16") == 0) return KvCache::Bf16;
  if (std::strcmp(v, "int8") == 0) return KvCache::Int8;
  throw std::runtime_error(std::string("--kv-cache / B70_KV_CACHE ") + v +
                           ": expected bf16 or int8");
}

const char* kv_cache_name(KvCache kv) { return kv == KvCache::Int8 ? "int8" : "bf16"; }

KvLayer KvLayout::layer(void* k_base, void* v_base, uint32_t l) const {
  auto at = [](void* base, size_t off) {
    return static_cast<void*>(static_cast<uint8_t*>(base) + off);
  };
  KvLayer r;
  r.k = at(k_base, rows_offset(l));
  r.v = at(v_base, rows_offset(l));
  if (form == KvCache::Int8) {
    r.ks = at(k_base, scales_offset(l));
    r.vs = at(v_base, scales_offset(l));
  }
  return r;
}

// One K (or V) allocation of `layers` FA layers in the given form (KvLayout).
KvLayout kv_layout(uint32_t max_len, const model::ModelDesc& desc, uint32_t layers, KvCache kv) {
  KvLayout l;
  l.form = kv;
  l.max_len = max_len;
  l.kv_heads = desc.fa_kv_heads;
  l.head_dim = Q::kFaHeadDim;
  l.layers = layers;
  return l;
}

PersistentSizes PersistentDims::sizes(uint32_t max_len, const model::ModelDesc& desc,
                                      KvCache kv) {
  PersistentSizes s{};
  s.control = sizeof(Control);
  s.gdn_state = gdn_state_bytes(desc);
  s.conv_ring = size_t{desc.gdn_layers} * kConvRing * desc.gdn_conv_dim() * kBf16;
  // bf16 [fa_layers][max_len][kv-heads][256], or (spec 12b) the int8 rows + fp16 scales.
  s.kv_k = kv_layout(max_len, desc, desc.fa_layers, kv).bytes();
  s.kv_v = s.kv_k;
  return s;
}

DecodeAttn decode_attn() {
  const char* v = std::getenv("B70_DECODE_ATTN");
  if (v == nullptr || *v == '\0') return kDefaultDecodeAttn;
  if (std::strcmp(v, "v1") == 0) return DecodeAttn::V1;
  if (std::strcmp(v, "v2") == 0) return DecodeAttn::V2;
  throw std::runtime_error(std::string("B70_DECODE_ATTN=") + v + ": expected v1 or v2");
}

const char* decode_attn_name(DecodeAttn a) { return a == DecodeAttn::V2 ? "v2" : "v1"; }

DecodeScratchSizes DecodeScratchDims::sizes(uint32_t max_len, const model::ModelDesc& desc,
                                            DecodeAttn attn) {
  DecodeScratchSizes s{};
  s.resid = size_t{kM} * desc.hidden * kBf16;
  s.x = size_t{kM} * x_width(desc) * kBf16;
  s.partials = partials_bytes(desc);
  s.ab_out = size_t{kM} * desc.shape(model::LinearId::AB).N * kFp32;
  // prep_res_fold's chunk sums-of-squares, one fp32 per (work-group,
  // token): 20 x 8 x 4 = 640 B, one allocation reused by all 129 sites
  // because the list is in-order and each site's stage B consumes what its
  // own stage A wrote before the next site's stage A overwrites it.
  s.norm_sumsq = size_t{kNormGroups} * kM * kFp32;
  s.gdn_o = size_t{kM} * desc.gdn_value_dim() * kFp32;
  s.attn_q = size_t{kM} * desc.fa_value_dim() * kFp32;
  s.attn_gate = size_t{kM} * desc.fa_value_dim() * kFp32;
  // v1 strides its partials by every block of the cache, [q-heads][max_len / 64][M][258];
  // v2 never has more than kAttnV2Blocks per row, [q-heads][32][M][258], whatever max_len
  // is - 6.34 MB on Qwen3.8 against v1's 405.8 MB at 131072 (0.595 GB at 192k), which
  // `--max-len auto` turns into context.
  const size_t blocks = attn == DecodeAttn::V1 ? attn_blocks(max_len) : kAttnV2Blocks;
  s.attn_part = size_t{desc.fa_q_heads} * blocks * kM * kAttnPartStride * kFp32;
  s.attn_out = size_t{kM} * desc.fa_value_dim() * kBf16;
  s.logits = size_t{kM} * Q::kVocab * kFp32;
  s.argmax_part = size_t{kM} * kArgmaxGroups * 2 * kFp32;
  s.moe = moe_scratch_layout(desc).total;
  return s;
}

MoeScratchLayout moe_scratch_layout(const model::ModelDesc& desc) {
  MoeScratchLayout l;
  if (!desc.is_moe()) return l;
  const model::MoeDesc& m = desc.moe;
  const size_t kM = DecodeScratchDims::kM;
  l.layer_slots = desc.layers + 1;
  l.logits_layer = kM * m.router_n() * kFp32;
  l.route_layer = kM * kMoeRouteWords * sizeof(uint32_t);
  l.logits_off = 0;
  l.route_off = l.logits_off + l.layer_slots * l.logits_layer;
  l.h_off = l.route_off + l.layer_slots * l.route_layer;
  l.h = kM * m.slots() * m.expert_intermediate * kBf16;
  l.total = l.h_off + l.h;
  return l;
}

PrefillScratchSizes PrefillScratchDims::sizes(uint32_t max_len, const model::ModelDesc& desc) {
  static_assert(kC % kGdnChunk == 0, "kC must be a whole number of 64-position GDN chunks");
  PrefillScratchSizes s{};
  s.ids = size_t{kC} * sizeof(uint32_t);
  s.resid = size_t{kC} * desc.hidden * kBf16;
  s.x = size_t{kC} * x_width(desc) * kBf16;
  // S = 1 (ruling R1), so the rectangle is [C][max int4 N] and not
  // [S][C][N] - the sole reason a 2048-row chunk fits at all.
  s.partials = size_t{kC} * max_int4_n(desc) * kFp32;
  s.ab_out = size_t{kC} * desc.shape(model::LinearId::AB).N * kFp32;
  s.norm_sumsq = size_t{kNormGroups} * kC * kFp32;
  s.gdn_o = size_t{kC} * desc.gdn_value_dim() * kFp32;
  s.mixer_out = size_t{kC} * mixer_width(desc) * kBf16;
  // lm_head runs on the chunk's LAST position only (spec §3.5), so one row.
  s.logits = size_t{Q::kVocab} * kFp32;
  s.argmax_part = size_t{kArgmaxGroups} * 2 * kFp32;
  s.gdn_xb = size_t{kC} * desc.gdn_conv_dim() * kBf16;
  s.gdn_seed = size_t{3} * desc.gdn_conv_dim() * kBf16;
  s.gdn_g = size_t{kC} * desc.gdn_v_heads * kFp32;
  s.gdn_beta = size_t{kC} * desc.gdn_v_heads * kFp32;
  s.gdn_A = size_t{kC / kGdnChunk} * desc.gdn_v_heads * kGdnChunk * kGdnChunk * kFp32;
  s.gdn_A2 = s.gdn_A;
  s.gdn_w = size_t{kC} * desc.gdn_value_dim() * kBf16;
  s.gdn_u = size_t{kC} * desc.gdn_value_dim() * kBf16;
  s.pf_q = size_t{kC} * desc.fa_value_dim() * kBf16;
  s.pf_attn = size_t{kC} * desc.fa_value_dim() * kBf16;
  s.pf_o = size_t{desc.fa_q_heads} * kC * Q::kFaHeadDim * kFp32;
  s.pf_rowsum = size_t{desc.fa_q_heads} * kC * kFp32;
  // The lazy four (spec 2.1 §3.3, spec 6): sycl-tla's bf16 dequant [hidden][max int4 N];
  // the L0 slab [intermediate][1024] (one pf_gemm slab of the widest K); the composed
  // attention's S and P, [one GQA group][kC][max_len] fp32 and bf16 - the only prefill
  // scratch that scales with max_len.
  s.dequant = size_t{desc.hidden} * max_int4_n(desc) * kBf16;
  s.slab = size_t{slab_k(desc)} * kernels::kPfSlabWidth * kBf16;
  s.pf_s = size_t{desc.fa_gqa()} * kC * max_len * kFp32;
  s.pf_p = size_t{desc.fa_gqa()} * kC * max_len * kBf16;
  s.moe = moe_prefill_layout(desc).total;   // spec 15d: 0 on a dense model
  return s;
}

MtpSizes MtpDims::sizes(uint32_t max_len, const model::ModelDesc& desc, uint32_t draft_vocab,
                        KvCache kv) {
  MtpSizes s{};
  s.hctl = sizeof(Control);
  s.gdn_spec = size_t{kSlots - 1} * gdn_state_bytes(desc);
  s.kv_k = kv_layout(max_len, desc, 1, kv).bytes();   // the head's own (17th) KV layer
  s.kv_v = s.kv_k;
  s.hh = size_t{DecodeScratchDims::kM + 1} * desc.hidden * kBf16;
  s.dh = size_t{desc.hidden} * kBf16;
  s.logits = size_t{kMaxK} * Q::kVocab * kFp32;
  s.dv_logits = size_t{draft_vocab} * kFp32;   // spec 8 §11: the compact head's logits
  return s;
}

size_t mtp_prefill_hidden_bytes(const model::ModelDesc& desc) {
  return size_t{PrefillScratchDims::kC + 1} * desc.hidden * kBf16;
}

Int8ScratchSizes int8_scratch_sizes(uint32_t max_k) {
  Int8ScratchSizes s{};
  s.xq = size_t{PrefillScratchDims::kC} * max_k;                // int8 [kC][max_k]
  s.xs = size_t{PrefillScratchDims::kC} * kFp32;                // fp32 [kC]
  s.w8 = size_t{max_k / 4} * kernels::kPfSlabWidth * 4;         // u32 [max_k / 4][1024]
  return s;
}

size_t int8_scale_bytes(const model::ModelDesc& desc) {
  size_t b = 0;
  for (const model::LayerDesc& ld : desc.layer_descs())
    for (const model::FusedLinear& fl : ld.linears)
      if (fl.kind == model::WeightKind::Int4 && !moe_ffn_row(desc, fl.id))
        b += 2 * size_t{fl.shape.N} * kFp32;   // ws, 1/ws
  // Spec 15d: a MoE layer's expert gate||up array (every block, the shared expert's too)
  // is one h8 weight of blocks x 2 I columns (Int8State::scales_raw); its down is bf16.
  if (desc.is_moe())
    b += size_t{desc.layers} * 2 * size_t{desc.moe.blocks()} * 2 * desc.moe.expert_intermediate * kFp32;
  return b;
}

uint32_t prefill_int8_max_k(const model::ModelDesc& desc) {
  return desc.is_moe() ? slab_k(desc) : desc.intermediate;
}

// --- spec 15d: the MoE prefill scratch ------------------------------------------------

uint32_t moe_prefill_tiles(const model::ModelDesc& desc, uint32_t C) {
  if (!desc.is_moe()) return 0;
  const uint32_t tm = kernels::pf_moe::kTileM;
  return (C * desc.moe.top_k + desc.moe.experts * (tm - 1)) / tm + (C + tm - 1) / tm;
}

namespace {
// One expert weight block in each grouped-GEMM B form (pf_moe_gemm.cl).
size_t moe_block_bytes(const model::ModelDesc& d, MoeWeightForm f) {
  const size_t gu = size_t{d.hidden} * 2 * d.moe.expert_intermediate;   // elements
  const size_t dn = size_t{d.moe.expert_intermediate} * d.hidden;
  switch (f) {
    case MoeWeightForm::GateUpInt8: return gu;            // int8 (VNNI-4 u32 words)
    case MoeWeightForm::GateUpBf16: return gu * kBf16;
    case MoeWeightForm::DownBf16: return dn * kBf16;
  }
  return 0;
}
size_t align256(size_t b) { return (b + 255) / 256 * 256; }
}  // namespace

MoePrefillLayout moe_prefill_layout(const model::ModelDesc& desc) {
  MoePrefillLayout l;
  if (!desc.is_moe()) return l;
  const model::MoeDesc& m = desc.moe;
  const size_t kC = PrefillScratchDims::kC;
  l.tmax = moe_prefill_tiles(desc, PrefillScratchDims::kC);
  l.rows = l.tmax * kernels::pf_moe::kTileM;
  // The weight batch: every block in the int8 gate||up and bf16 down forms, half the
  // blocks (rounded up) in the bf16 gate||up form - whichever is largest.
  const uint32_t half = (m.blocks() + 1) / 2;
  l.w = std::max({moe_block_bytes(desc, MoeWeightForm::GateUpInt8) * m.blocks(),
                  moe_block_bytes(desc, MoeWeightForm::DownBf16) * m.blocks(),
                  moe_block_bytes(desc, MoeWeightForm::GateUpBf16) * half});
  size_t at = 0;
  auto region = [&](size_t bytes) {
    const size_t off = at;
    at = align256(at + bytes);
    return off;
  };
  l.logits_off = region(kC * m.router_n() * kFp32);
  l.route_layer = kC * kMoeRouteWords * sizeof(uint32_t);
  l.route_off = region(size_t{desc.layers} * l.route_layer);
  l.hdr_off = region(size_t{kernels::pf_moe::hdr_words(m.experts)} * sizeof(uint32_t));
  l.tiles_off = region(size_t{l.tmax} * 2 * sizeof(uint32_t));
  l.row_tok_off = region(size_t{l.rows} * sizeof(uint32_t));
  l.pair_row_off = region(kC * m.top_k * sizeof(uint32_t));
  l.xs_off = region(size_t{l.rows} * kFp32);
  l.h_off = region(size_t{l.rows} * m.expert_intermediate * kBf16);
  l.xg_off = region(size_t{l.rows} * desc.hidden * kBf16);
  l.w_off = region(l.w);
  l.total = at;
  return l;
}

uint32_t moe_prefill_batch_blocks(const model::ModelDesc& desc, MoeWeightForm f) {
  if (!desc.is_moe()) return 0;
  const size_t per = std::max<size_t>(moe_prefill_layout(desc).w / moe_block_bytes(desc, f), 1);
  return uint32_t(std::min<size_t>(per, desc.moe.blocks()));
}

}  // namespace runtime
