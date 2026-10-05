#include "runtime/k2/k2_sizes.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "runtime/control.h"

namespace runtime::k2 {
namespace {
// The kernels' constants the sizes depend on (src/kernels/k2_kernels.h is their home on the
// host; this file stays free of the kernel-dir define, so it is spelled here and checked
// against it in k2_capture.cc).
constexpr uint32_t kNormG = 20, kAttnTgt = 32, kAttnPart = 130, kArgmaxChunk = 1024;
}  // namespace

K2Attn k2_attn() {
  const char* v = std::getenv("B70_K2_ATTN");
  if (v == nullptr || *v == '\0') return kDefaultK2Attn;
  if (std::strcmp(v, "flash") == 0) return K2Attn::Flash;
  if (std::strcmp(v, "eager") == 0) return K2Attn::Eager;
  throw std::runtime_error(std::string("B70_K2_ATTN=") + v + ": expected flash or eager");
}

const char* k2_attn_name(K2Attn a) { return a == K2Attn::Eager ? "eager" : "flash"; }

PersistentSizes persistent_sizes(const model::K2Desc& d, uint32_t max_len) {
  PersistentSizes s;
  s.control = sizeof(Control);
  s.kv_k = s.kv_v = size_t(d.layers) * kv_layer_bytes(d, max_len);
  return s;
}

size_t partials_floats(const model::K2Desc& d) {
  size_t m = 0;
  for (model::K2LinearId id : {model::K2LinearId::AttnDense, model::K2LinearId::AttnSparse,
                               model::K2LinearId::OProj, model::K2LinearId::DenseGateUp,
                               model::K2LinearId::DenseDown}) {
    const model::GemvShape s = d.linear(id).shape;
    m = std::max(m, size_t(s.S) * s.N);
  }
  return m;
}

ScratchSizes scratch_sizes(const model::K2Desc& d) {
  ScratchSizes s;
  s.resid = size_t(kM) * d.hidden * 2;
  s.x = size_t(kM) * std::max(d.hidden, d.dense_inter) * 2;
  s.partials = partials_floats(d) * kM * 4;
  s.norm_sumsq = size_t(kNormG) * kM * 4;
  s.attn_q = s.attn_gate = size_t(kM) * d.q_n() * 4;
  s.attn_part = size_t(d.q_heads) * kAttnTgt * kM * kAttnPart * 4;
  s.attn_out = size_t(kM) * d.q_n() * 2;
  s.router = size_t(kM) * d.router_n() * 4;
  s.routes = size_t(d.layers) * 2 * kM * kRouteWords * 4;
  s.moe_h = size_t(kM) * (d.top_k + 1) * d.moe_inter * 2;
  s.logits = size_t(kM) * d.vocab * 4;
  s.argmax_part = size_t(kM) * ((d.vocab + kArgmaxChunk - 1) / kArgmaxChunk) * 2 * 4;
  return s;
}

size_t attn_scores_bytes(const model::K2Desc& d, uint32_t max_len, K2Attn a) {
  return a == K2Attn::Eager ? size_t(kM) * d.q_heads * max_len * 4 : 0;
}

size_t decode_launches(const model::K2Desc& d, K2Attn a) {
  const size_t extra = a == K2Attn::Eager ? 2 : 0;   // score + softmax + P·V + reduce
  return 1 + size_t(d.dense_layers) * (12 + extra) + size_t(d.sparse_layers()) * (15 + extra) + 5;
}

// --- Spec 18c: the prefill chunk --------------------------------------------------------

uint32_t pf_tiles(uint32_t experts, uint32_t top_k, bool shared, uint32_t C) {
  return (C * top_k + experts * (kPfTm - 1)) / kPfTm + (shared ? (C + kPfTm - 1) / kPfTm : 0);
}
uint32_t pf_moe_tiles(const model::K2Desc& d, uint32_t C) { return pf_tiles(d.experts, d.top_k, true, C); }
uint32_t pf_mova_tiles(const model::K2Desc& d, uint32_t C) {
  return pf_tiles(d.value_experts, d.value_top_k, false, C);
}

size_t pf_block_bf16(const model::K2Desc& d, PfGroup g) {
  switch (g) {
    case PfGroup::Value: return size_t(d.hidden) * d.kv_n() * 2;
    case PfGroup::GateUp: return size_t(d.hidden) * 2 * d.moe_inter * 2;
    case PfGroup::Down: return size_t(d.moe_inter) * d.hidden * 2;
  }
  return 0;
}
namespace {
uint32_t group_blocks(const model::K2Desc& d, PfGroup g) {
  return g == PfGroup::Value ? d.value_experts : d.moe_blocks();
}
}  // namespace
size_t pf_weight_batch_bytes(const model::K2Desc& d) {
  return std::max({pf_block_bf16(d, PfGroup::Value) * d.value_experts,
                   pf_block_bf16(d, PfGroup::Down) * d.moe_blocks(),
                   pf_block_bf16(d, PfGroup::GateUp) * ((d.moe_blocks() + 1) / 2)});
}
uint32_t pf_batch_blocks(const model::K2Desc& d, PfGroup g) {
  const size_t per = std::max<size_t>(pf_weight_batch_bytes(d) / pf_block_bf16(d, g), 1);
  return uint32_t(std::min<size_t>(per, group_blocks(d, g)));
}
uint32_t pf_batches(const model::K2Desc& d, PfGroup g) {
  const uint32_t per = pf_batch_blocks(d, g);
  return (group_blocks(d, g) + per - 1) / per;
}

uint32_t pf_ld_max(const model::K2Desc& d) {
  uint32_t m = 0;
  for (model::K2LinearId id : {model::K2LinearId::AttnDense, model::K2LinearId::AttnSparse,
                               model::K2LinearId::OProj, model::K2LinearId::DenseDown})
    m = std::max(m, pf_ld(d.linear(id).shape.N));
  return m;   // gate||up writes the SiLU'd x (xi), never partials
}

PrefillSizes prefill_sizes(const model::K2Desc& d) {
  PrefillSizes s;
  const size_t C = kPfC;
  s.ids = C * 4;
  s.resid = s.xn = C * d.hidden * 2;
  s.xi = C * d.dense_inter * 2;
  s.partials = C * pf_ld_max(d) * 4;
  size_t slab = 0;
  for (model::K2LinearId id : {model::K2LinearId::AttnDense, model::K2LinearId::AttnSparse,
                               model::K2LinearId::OProj, model::K2LinearId::DenseGateUp,
                               model::K2LinearId::DenseDown}) {
    const model::GemvShape sh = d.linear(id).shape;
    slab = std::max(slab, size_t(sh.K) * pf_slab_width(sh.N, 0) * 2);
  }
  s.slab = slab;
  s.sumsq = size_t(kNormG) * C * 4;
  s.attn_q = s.attn_gate = C * d.q_n() * 4;
  s.attn_out = C * d.q_n() * 2;
  s.logits = C * d.router_n() * 4;
  s.routes = size_t(d.layers) * 2 * C * kRouteWords * 4;
  const uint32_t maxe = std::max(d.experts, d.value_experts);
  s.hdr = size_t((4 + maxe + 1 + 15) / 16 * 16) * 4;
  const uint32_t moe_t = pf_moe_tiles(d, kPfC), mova_t = pf_mova_tiles(d, kPfC);
  const size_t rows = size_t(std::max(moe_t, mova_t)) * kPfTm;
  s.tiles = size_t(std::max(moe_t, mova_t)) * 2 * 4;
  s.row_tok = rows * 4;
  s.pair_row = C * std::max(d.top_k, d.value_top_k) * 4;
  s.xg = rows * d.hidden * 2;
  s.h = std::max(size_t(moe_t) * kPfTm * d.moe_inter * 2, size_t(mova_t) * kPfTm * d.kv_n() * 2);
  s.w = pf_weight_batch_bytes(d);
  return s;
}

size_t prefill_chunk_launches(const model::K2Desc& d) {
  const auto lin = [&](model::K2LinearId id) { return 2 * size_t(pf_slabs(d.linear(id).shape.N)); };
  const size_t attn_tail = 1 + 1;   // k2_attn_prep, the flash attention (gate fused)
  const size_t dense = 2 + lin(model::K2LinearId::AttnDense) + attn_tail + lin(model::K2LinearId::OProj) +
                       2 + lin(model::K2LinearId::DenseGateUp) + lin(model::K2LinearId::DenseDown);
  const size_t mova = 4 + 2 * size_t(pf_batches(d, PfGroup::Value));
  const size_t moe = 5 + 2 * size_t(pf_batches(d, PfGroup::GateUp)) + 2 * size_t(pf_batches(d, PfGroup::Down));
  const size_t sparse = 2 + lin(model::K2LinearId::AttnSparse) + mova + attn_tail +
                        lin(model::K2LinearId::OProj) + 2 + moe;
  return 1 + size_t(d.dense_layers) * dense + size_t(d.sparse_layers()) * sparse;
}

Plan plan(const model::K2Desc& d, uint32_t max_len, size_t model_bytes, bool debug_tap,
          bool prefill, K2Attn a) {
  Plan p;
  p.max_len = max_len;
  p.rope = d.rope_table_bytes(max_len);
  p.model = model_bytes + p.rope;
  const PersistentSizes ps = persistent_sizes(d, max_len);
  p.kv = ps.kv();
  p.decode_state = ps.control + scratch_sizes(d).total() + attn_scores_bytes(d, max_len, a) +
                   (debug_tap ? tap_bytes(d) : 0);
  p.prefill_scratch = prefill ? prefill_sizes(d).total() : 0;
  return p;
}

uint32_t max_len_that_fits(const model::K2Desc& d, size_t model_bytes, size_t device_bytes,
                           size_t reserve_bytes, uint32_t cap, bool prefill) {
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("k2::max_len_that_fits: the cap " + std::to_string(cap) +
                                " is below one " + std::to_string(kMaxLenQuantum) +
                                "-position quantum");
  const auto fits = [&](uint32_t len) {
    return plan(d, len, model_bytes, false, prefill).total() + reserve_bytes <= device_bytes;
  };
  // Every term is non-decreasing in max_len (the KV, the RoPE table): the quanta that fit
  // are a prefix, and a bisection finds its end (memory_plan.cc's argument).
  uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum;
  uint32_t hi = cap / kMaxLenQuantum;
  if (!fits(lo * kMaxLenQuantum)) return 0;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (fits(mid * kMaxLenQuantum))
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo * kMaxLenQuantum;
}

std::string describe(const Plan& p, size_t device_bytes, size_t reserve_bytes) {
  const std::string label = "plan at max_len " + std::to_string(p.max_len);
  std::string s = format_memory(label.c_str(), p, device_bytes);
  char buf[192];
  std::snprintf(buf, sizeof buf, "; + reserve %.3f GB = %.3f GB (RoPE %.3f GB in model; %s)",
                reserve_bytes / 1e9, (p.total() + reserve_bytes) / 1e9, p.rope / 1e9,
                p.prefill_scratch ? "the K2 prefill scratch planned, spec 18c"
                                  : "decode only: no prefill scratch");
  return s + buf;
}

}  // namespace runtime::k2
