#include "runtime/lfm2moe/lfm2moe_sizes.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "runtime/control.h"

namespace runtime::lfm2moe {
namespace {
// The kernels' constants the sizes depend on (src/kernels/lfm2moe_kernels.h is their home on the host; this file
// stays free of the kernel-dir define - lfm2moe_capture.cc checks them against it).
constexpr uint32_t kAttnTgt = 32, kAttnPart = 66, kArgmaxChunk = 1024;
constexpr size_t kLine = 64;   // an empty group's allocation
size_t line(size_t b) { return b ? b : kLine; }
}  // namespace

LfmAttn parse_attn(const std::string& v) {
  if (v.empty() || v == "flash") return LfmAttn::Flash;
  if (v == "eager") return LfmAttn::Eager;
  throw std::runtime_error("B70_LFM_ATTN=" + v + ": expected flash or eager");
}
LfmAttn lfm_attn() {
  const char* v = std::getenv("B70_LFM_ATTN");
  return parse_attn(v ? v : "");
}
const char* lfm_attn_name(LfmAttn a) { return a == LfmAttn::Eager ? "eager" : "flash"; }

LfmFuse parse_fuse(const std::string& v) {
  LfmFuse f;
  if (v.empty() || v == "none") return f;
  if (v == "all") return LfmFuse{true, true, true};
  size_t at = 0;
  while (at <= v.size()) {
    const size_t comma = v.find(',', at);
    const std::string w = v.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
    if (w == "route")
      f.route = true;
    else if (w == "router")
      f.router = true;
    else if (w == "conv")
      f.conv = true;
    else
      throw std::runtime_error("B70_LFM_FUSE=" + v + ": '" + w +
                               "' is not an arm - expected none, all or a comma list of route, router, conv");
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  return f;
}
LfmFuse lfm_fuse() {
  const char* v = std::getenv("B70_LFM_FUSE");
  return parse_fuse(v ? v : "");
}
std::string lfm_fuse_name(LfmFuse f) {
  if (!f.any()) return "none";
  if (f.route && f.router && f.conv) return "all";
  std::string s;
  if (f.route) s += "route";
  if (f.router) s += std::string(s.empty() ? "" : ",") + "router";
  if (f.conv) s += std::string(s.empty() ? "" : ",") + "conv";
  return s;
}

ScratchSizes scratch_sizes(const model::Lfm2MoeDesc& d, uint32_t max_len, LfmAttn a) {
  ScratchSizes s;
  size_t widest = 0;
  for (uint32_t l = 0; l < d.layers; ++l)
    for (const model::LfmLinear& ln : d.linears(l))
      if (ln.kind == model::WeightKind::Int4) widest = std::max(widest, size_t(ln.shape.S) * ln.shape.N);
  s.h0 = s.h1 = s.x = s.conv_y = size_t(kM) * d.hidden * 2;
  s.slices = widest * kM * 4;
  s.logits_r = size_t(kM) * d.experts * 4;
  s.routes = size_t(d.layers) * kM * kRouteWords * 4;
  s.moe_h = size_t(kM) * d.top_k * d.moe_inter * 2;
  s.attn_q = size_t(kM) * d.q_n() * 4;
  s.attn_part = size_t(d.q_heads) * kAttnTgt * kM * kAttnPart * 4;
  s.attn_out = size_t(kM) * d.q_n() * 2;
  s.s_mlp = size_t(kM) * d.dense_inter * 2;
  s.logits = size_t(kM) * d.vocab * 4;
  s.argmax_part = size_t(kM) * ((d.vocab + kArgmaxChunk - 1) / kArgmaxChunk) * 2 * 4;
  s.attn_scores = a == LfmAttn::Eager ? size_t(kM) * d.q_heads * max_len * 4 : 0;
  return s;
}

PersistentSizes persistent_sizes(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev,
                                 uint32_t max_len) {
  PersistentSizes s;
  s.control = sizeof(Control);
  s.kv_k = s.kv_v = line(size_t(device_attn_layers(d, p, dev)) * kv_layer_bytes(d, max_len));
  s.conv_ring = line(size_t(device_conv_layers(d, p, dev)) * ring_layer_bytes(d));
  return s;
}

size_t device_launches(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, LfmAttn a, LfmFuse f,
                       PpHandoff h) {
  size_t n = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) {
    const bool attn = d.is_attention(l), dense = d.is_dense(l);
    size_t k = 2;                                           // operator_norm, ffn_norm
    k += attn ? 5 + (a == LfmAttn::Eager ? 2 : 0) : 3;      // q||k||v, prep, attn x 2 (eager x 4), o | in, conv, out
    k += dense ? 3 : 4;                                     // gate||up, silu, down | router, route, gate||up, down
    if (!attn && f.conv) k -= 1;
    if (!dense && f.route) k -= 1;
    if (!dense && f.router) k -= 1;
    n += k;
  }
  if (dev == 0) n += 1;                                     // lfm_embed_gather
  if (dev + 1 == p.devices) n += 4;                         // the final norm, the head, the two argmax stages
  if (p.devices > 1 && h == PpHandoff::Peer) n += 1;        // pp_send (device 0) / pp_recv (device 1)
  return n;
}

size_t decode_launches(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, LfmAttn a, LfmFuse f, PpHandoff h) {
  size_t n = 0;
  for (uint32_t dev = 0; dev < p.devices; ++dev) n += device_launches(d, p, dev, a, f, h);
  return n;
}

Plan plan(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, uint32_t max_len,
         bool int8_head, LfmAttn a) {
  model::validate(p, d);
  if (max_len > d.trained_max_len)
    throw std::runtime_error("runtime::lfm2moe::plan: max_len " + std::to_string(max_len) +
                             " exceeds the trained context " + std::to_string(d.trained_max_len) +
                             " (max_position_embeddings)");
  Plan pl;
  pl.max_len = max_len;
  pl.weights = loader::lfm::device_weight_bytes(d, p, dev, int8_head);
  // 24c: what Lfm2MoeBuffers allocates (persistent_sizes - an empty group is one 64-byte line), so the plan IS the
  // allocation (lfm2moe_decode_test checks it on the card).
  const PersistentSizes ps = persistent_sizes(d, p, dev, max_len);
  pl.kv = ps.kv_k + ps.kv_v;
  pl.conv_rings = ps.conv_ring;
  pl.rope = loader::lfm::device_has_attention(d, p, dev) ? loader::lfm::rope_table_bytes(d, max_len) : 0;
  pl.decode_scratch = ps.control + scratch_sizes(d, max_len, a).total();
  pl.reserve = size_t(kDefaultReserveGb * 1e9);
  return pl;
}

void require_fits(const Plan& p, size_t device_bytes) {
  if (p.total() <= device_bytes) return;
  char buf[512];
  std::snprintf(buf, sizeof buf,
               "runtime::lfm2moe: the plan at max_len %u does not fit %zu B (weights %zu, kv %zu, "
               "conv_rings %zu, rope %zu, decode_scratch %zu, prefill_scratch %zu, reserve %zu; total %zu)",
               p.max_len, device_bytes, p.weights, p.kv, p.conv_rings, p.rope, p.decode_scratch,
               p.prefill_scratch, p.reserve, p.total());
  throw std::runtime_error(buf);
}

uint32_t max_len_auto(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, size_t device_bytes,
                      double reserve_gb, bool int8_head, LfmAttn a) {
  model::validate(p, d);
  const uint32_t cap = d.trained_max_len;
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("runtime::lfm2moe::max_len_auto: the trained context " + std::to_string(cap) +
                                " is below one " + std::to_string(kMaxLenQuantum) + "-position quantum");
  const size_t reserve_bytes = size_t(reserve_gb * 1e9);
  const auto fits = [&](uint32_t len) {
    for (uint32_t dev = 0; dev < p.devices; ++dev) {
      Plan pl = plan(d, p, dev, len, int8_head, a);
      pl.reserve = reserve_bytes;
      if (pl.total() > device_bytes) return false;
    }
    return true;
  };
  uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum, hi = cap / kMaxLenQuantum;
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

uint32_t pp_split(const model::Lfm2MoeDesc& d, bool int8_head) {
  if (d.layers < 2)
    throw std::invalid_argument(d.name + ": " + std::to_string(d.layers) + " layer(s), nothing to split");
  std::vector<size_t> layer_bytes(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l) layer_bytes[l] = loader::lfm::layer_weight_bytes(d, l);
  const size_t dev0_fixed = loader::lfm::embed_tiled_bytes(d);
  const size_t dev1_fixed = size_t(d.hidden) * 4 +
                            (int8_head ? loader::lfm::head_int8_bytes(d) : loader::lfm::embed_tiled_bytes(d));
  return runtime::pp_balance(layer_bytes, dev0_fixed, dev1_fixed).split;
}

std::string describe(const Plan& p, size_t device_bytes) {
  char buf[400];
  std::snprintf(buf, sizeof buf,
               "max_len %u: weights %.3f GB, kv %.3f GB, conv rings %zu B, rope %zu B, reserve %.3f GB, "
               "total %.3f GB",
               p.max_len, p.weights / 1e9, p.kv / 1e9, p.conv_rings, p.rope, p.reserve / 1e9, p.total() / 1e9);
  std::string out = buf;
  if (device_bytes) {
    char tail[64];
    std::snprintf(tail, sizeof tail, " of %.3f GB", device_bytes / 1e9);
    out += tail;
  }
  return out;
}

}  // namespace runtime::lfm2moe
