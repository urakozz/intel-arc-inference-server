#include "runtime/k2/k2_sizes.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "runtime/control.h"

namespace runtime::k2 {
namespace {
// The kernels' constants the sizes depend on (src/kernels/k2_kernels.h is their home on the
// host; this file stays free of the kernel-dir define, so it is spelled here and checked
// against it in k2_capture.cc).
constexpr uint32_t kNormG = 20, kAttnTgt = 32, kAttnPart = 130, kArgmaxChunk = 1024;
}  // namespace

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

size_t decode_launches(const model::K2Desc& d) {
  return 1 + size_t(d.dense_layers) * 12 + size_t(d.sparse_layers()) * 15 + 5;
}

Plan plan(const model::K2Desc& d, uint32_t max_len, size_t model_bytes, bool debug_tap) {
  Plan p;
  p.max_len = max_len;
  p.rope = d.rope_table_bytes(max_len);
  p.model = model_bytes + p.rope;
  const PersistentSizes ps = persistent_sizes(d, max_len);
  p.kv = ps.kv();
  p.decode_state = ps.control + scratch_sizes(d).total() + (debug_tap ? tap_bytes(d) : 0);
  return p;
}

uint32_t max_len_that_fits(const model::K2Desc& d, size_t model_bytes, size_t device_bytes,
                           size_t reserve_bytes, uint32_t cap) {
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("k2::max_len_that_fits: the cap " + std::to_string(cap) +
                                " is below one " + std::to_string(kMaxLenQuantum) +
                                "-position quantum");
  const auto fits = [&](uint32_t len) {
    return plan(d, len, model_bytes).total() + reserve_bytes <= device_bytes;
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
  std::snprintf(buf, sizeof buf, "; + reserve %.3f GB = %.3f GB (RoPE %.3f GB in model; K2: no "
                "prefill scratch, spec 18c)", reserve_bytes / 1e9, (p.total() + reserve_bytes) / 1e9,
                p.rope / 1e9);
  return s + buf;
}

}  // namespace runtime::k2
