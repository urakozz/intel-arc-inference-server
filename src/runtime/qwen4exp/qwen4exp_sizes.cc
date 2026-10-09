#include "runtime/qwen4exp/qwen4exp_sizes.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "loader/qwen4exp_layout.h"
#include "runtime/control.h"

namespace runtime::qwen4exp {
namespace {

constexpr size_t kLine = 64;   // an empty group's allocation
size_t line(size_t b) { return b ? b : kLine; }

uint32_t count_qsa(const model::Qwen4ExpDesc& d, uint32_t first, uint32_t end) {
  uint32_t n = 0;
  for (uint32_t l = first; l < end; ++l) n += d.is_qsa(l) ? 1 : 0;
  return n;
}

size_t qsa_state_per_layer(const model::Qwen4ExpDesc& d, uint32_t max_len) {
  return size_t(max_len) * 2 * d.kv_n() * 2 + size_t(max_len / d.idx_compress) * d.idx_dim * 2 +
         size_t(kIdxTail) * d.idx_dim * 2;
}

std::string gb(size_t b) {
  char s[32];
  std::snprintf(s, sizeof s, "%.3f GB", double(b) / 1e9);
  return s;
}

}  // namespace

size_t gdn_state_bytes_per_layer(const model::Qwen4ExpDesc& d) {
  return size_t(d.gdn_v_heads) * d.gdn_head * d.gdn_head * 4;
}
size_t conv_ring_bytes_per_layer(const model::Qwen4ExpDesc& d) { return size_t(kConvRing) * d.conv_rows() * 2; }
size_t ple_state_bytes(const model::Qwen4ExpDesc& d) { return kPleConvOff + size_t(kPleRing) * d.hc_n() * 2; }

PersistentSizes persistent_sizes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev,
                                 uint32_t max_len) {
  const uint32_t qsa = count_qsa(d, p.first(dev), p.end(dev)), gdn = p.count(dev) - qsa;
  PersistentSizes s;
  s.control = sizeof(Control);
  s.kv = line(size_t(qsa) * max_len * 2 * d.kv_n() * 2);
  s.idx_keys = line(size_t(qsa) * (max_len / d.idx_compress) * d.idx_dim * 2);
  s.idx_tail = line(size_t(qsa) * kIdxTail * d.idx_dim * 2);
  s.gdn_state = line(size_t(gdn) * gdn_state_bytes_per_layer(d));
  s.conv_ring = line(size_t(gdn) * conv_ring_bytes_per_layer(d));
  const bool has_ple = d.ple_layer >= p.first(dev) && d.ple_layer < p.end(dev);
  s.ple = line(has_ple ? ple_state_bytes(d) : 0);
  return s;
}

size_t kv_bytes_per_pos(const model::Qwen4ExpDesc& d) {
  return size_t(d.qsa_before(d.layers)) * (size_t(2) * d.kv_n() * 2 + size_t(d.idx_dim) * 2 / d.idx_compress);
}

size_t host_ple_bytes(const model::Qwen4ExpDesc& d, loader::Q4PleScale s) {
  return loader::q4_ple_table_bytes(loader::q4_ple_primes(d.ple_base, d.ple_heads, 0), d.ple_dim, s);
}

// --- spec 21c: the decode list ----------------------------------------------------------------------------------
Q4Attn q4_attn() {
  const char* e = std::getenv("B70_Q4_ATTN");
  if (e == nullptr || *e == '\0') return kDefaultQ4Attn;
  if (std::strcmp(e, "flash") == 0) return Q4Attn::Flash;
  if (std::strcmp(e, "eager") == 0) return Q4Attn::Eager;
  throw std::runtime_error(std::string("B70_Q4_ATTN=") + e + ": expected flash or eager (unset: " +
                           q4_attn_name(kDefaultQ4Attn) + ")");
}
const char* q4_attn_name(Q4Attn a) { return a == Q4Attn::Eager ? "eager" : "flash"; }

size_t partials_floats(const model::Qwen4ExpDesc& d) {
  size_t m = 0;
  for (model::Q4LinearId id : {model::Q4LinearId::GdnQkvz, model::Q4LinearId::GdnOut, model::Q4LinearId::QsaQkvg,
                               model::Q4LinearId::QsaO}) {
    const model::Q4Linear l = d.linear(id);
    m = std::max(m, size_t(l.shape.S) * l.shape.N);
  }
  return m * kM;
}

ScratchSizes scratch_sizes(const model::Qwen4ExpDesc& d, uint32_t max_len, Q4Attn) {
  const size_t M = kM, qsa = std::max<uint32_t>(1u, d.qsa_before(d.layers));
  ScratchSizes s;
  s.H = M * d.hc_n() * 2;
  s.xn = M * d.hc_n() * 2;
  s.x = M * d.hidden * 2;
  s.partials = partials_floats(d) * 4;
  s.down_f32 = M * d.hc_down_rows() * 4;
  s.inj = line(0);
  s.ab = M * model::Qwen4ExpDesc::kAbPaddedN * 4;
  s.gdn_o = M * d.gdn_v_heads * d.gdn_head * 4;
  s.idx_f32 = M * d.idx_n() * 4;
  s.idx_q = M * d.idx_heads * d.idx_dim * 4;
  s.scores = M * (max_len / d.idx_compress) * 4;
  s.list = qsa * M * kListRow * 4;
  s.diag = line(qsa * M * 2 * 4);
  s.attn_q = M * d.q_n() * 4;
  s.attn_gate = M * d.q_n() * 4;
  s.attn_part = size_t(d.q_heads) * kAttnTgt * M * kAttnPart * 4;
  s.attn_out = M * d.q_n() * 2;
  s.logits_r = M * d.router_n() * 4;
  s.routes = size_t(d.layers) * M * kRouteWords * 4;
  s.moe_h = M * (d.top_k + 1) * d.moe_inter * 2;
  s.y = M * d.hidden * 2;
  s.ple_e = M * d.ple_e() * 2;
  s.ple_kv = M * d.ple_kv_n() * 4;
  s.ple_ids = line(M * d.ple_heads * 8);
  s.ple_consts = line(size_t(kPleConsts) * 8);
  s.logits = M * d.vocab * 4;
  s.argmax_part = M * ((d.vocab + 1023) / 1024) * 2 * 4;
  return s;
}

PpLandingLayout landing_layout(const model::Qwen4ExpDesc& d) { return pp_landing_layout(size_t(kM) * d.hc_n() * 2, 0); }
size_t link_bytes(const model::Qwen4ExpDesc& d, uint32_t dev) {
  return dev == 0 ? kPpStateWords * 4 : landing_layout(d).total + kPpStateWords * 4;
}

size_t device_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, Q4Attn a,
                       PpHandoff h, bool injected) {
  model::validate(p, d);
  if (dev >= p.devices) throw std::out_of_range("qwen4exp::device_launches: device " + std::to_string(dev));
  const size_t gdn = 15, qsa = (a == Q4Attn::Eager ? 18 : 19) - (injected ? 2 : 0);
  const bool two = p.devices == 2;
  size_t n = 0;
  if (dev == 0) n += 1;                                  // embed_gather
  if (two && dev == 1 && h == PpHandoff::Peer) n += 1;   // pp_recv
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) {
    n += d.is_qsa(l) ? qsa : gdn;
    if (l == d.ple_layer) n += (two && dev == 1 && l == p.first(1)) ? 3 : 4;
  }
  if (two && dev == 0) n += 1 + (h == PpHandoff::Peer ? 1 : 0);   // combine_norm _Y_NN (+ pp_send)
  if (dev + 1 == p.devices) n += 6;                      // the final mixer's 3, lm_head, argmax x 2
  return n;
}
size_t decode_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, Q4Attn a, PpHandoff h, bool injected) {
  size_t n = 0;
  for (uint32_t dev = 0; dev < p.devices; ++dev) n += device_launches(d, p, dev, a, h, injected);
  return n;
}

std::vector<size_t> pp_layer_bytes(const model::Qwen4ExpDesc& d, uint32_t max_len) {
  std::vector<size_t> v(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l) {
    v[l] = loader::q4_layer_bytes(d, l).total() +
           (d.is_qsa(l) ? qsa_state_per_layer(d, max_len) : gdn_state_bytes_per_layer(d) + conv_ring_bytes_per_layer(d));
    if (l == d.ple_layer) v[l] += ple_state_bytes(d);
  }
  return v;
}

PpBalance pp_split(const model::Qwen4ExpDesc& d, uint32_t max_len, bool int8_head, bool mtp) {
  if (d.layers < 2) throw std::invalid_argument(d.name + ": " + std::to_string(d.layers) + " layer(s), nothing to split");
  const size_t rope = d.rope_table_bytes(max_len), ctl = sizeof(Control);
  const size_t dev0 = loader::q4_embed_bytes(d) + rope + ctl;
  const size_t dev1 = loader::q4_final_mixer_bytes(d) + loader::q4_lm_head_bytes(d, int8_head) +
                      (mtp ? loader::q4_mtp_bytes(d) : 0) + rope + ctl;
  return pp_balance(pp_layer_bytes(d, max_len), dev0, dev1);
}

std::vector<DevicePlan> plan(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t max_len,
                             bool int8_head, bool mtp, bool debug_tap) {
  model::validate(p, d);
  std::vector<DevicePlan> out(p.devices);
  for (uint32_t dev = 0; dev < p.devices; ++dev) {
    DevicePlan& dp = out[dev];
    dp.device = dev;
    dp.first = p.first(dev);
    dp.end = p.end(dev);
    dp.mtp = mtp && dev + 1 == p.devices;
    dp.whole = d.layers == model::qwen4exp().layers;
    dp.weights = loader::q4_device_weight_bytes(d, p, dev, int8_head, mtp);
    dp.rope = loader::q4_device_has_qsa(d, p, dev, mtp) ? d.rope_table_bytes(max_len) : 0;
    dp.model = dp.weights + dp.rope;
    const PersistentSizes ps = persistent_sizes(d, p, dev, max_len);
    dp.kv = ps.kv_total();
    dp.state = ps.state();
    dp.scratch = scratch_sizes(d, max_len, kDefaultQ4Attn).total() + (debug_tap ? tap_bytes(d) : 0);
    dp.link = p.devices == 2 ? link_bytes(d, dev) : 0;
    dp.decode_state = dp.state + dp.scratch + dp.link;
  }
  return out;
}

bool fits(const std::vector<DevicePlan>& p, const std::array<size_t, kPpDevices>& device_bytes, size_t reserve) {
  for (const DevicePlan& dp : p)
    if (dp.total() + reserve > device_bytes[dp.device]) return false;
  return true;
}

model::Qwen4ExpDesc truncated(const model::Qwen4ExpDesc& d, uint32_t layers) {
  if (layers < d.ple_layer + 1 || layers > d.layers)
    throw std::invalid_argument("--layers " + std::to_string(layers) + ": " + d.name + " runs " +
                                std::to_string(d.ple_layer + 1) + ".." + std::to_string(d.layers) +
                                " layers (the PLE layer " + std::to_string(d.ple_layer) + " must be loaded)");
  model::Qwen4ExpDesc t = d;
  t.layers = layers;
  return t;
}

model::Q4Placement placement_for(const model::Qwen4ExpDesc& t, uint32_t devices, uint32_t max_len, bool int8_head,
                                 bool mtp, size_t device_bytes, size_t reserve) {
  if (devices == 1) return model::Q4Placement::one(t);
  if (devices != 2) throw std::invalid_argument("qwen4exp: " + std::to_string(devices) + " devices (1 or 2)");
  if (t.layers < 2) throw std::invalid_argument("qwen4exp: one layer cannot split over two devices");
  const std::array<size_t, kPpDevices> caps = {device_bytes, device_bytes};
  const model::Q4Placement balanced = model::Q4Placement::two(t, pp_split(t, max_len, int8_head, mtp).split);
  if (fits(plan(t, balanced, max_len, int8_head, mtp), caps, reserve)) return balanced;
  for (uint32_t s = 1; s < t.layers; ++s) {
    const model::Q4Placement pl = model::Q4Placement::two(t, s);
    if (fits(plan(t, pl, max_len, int8_head, mtp), caps, reserve)) return pl;
  }
  return balanced;   // nothing fits: the balanced split (the caller's require_fits names the bytes)
}

uint32_t layers_that_fit(const model::Qwen4ExpDesc& d, uint32_t devices, uint32_t max_len, bool int8_head, bool mtp,
                         size_t device_bytes, size_t reserve) {
  const std::array<size_t, kPpDevices> caps = {device_bytes, device_bytes};
  uint32_t best = 0;
  for (uint32_t n = d.ple_layer + 1; n <= d.layers; ++n) {
    if (devices == 2 && n < 2) continue;
    const model::Qwen4ExpDesc t = truncated(d, n);
    const model::Q4Placement pl = placement_for(t, devices, max_len, int8_head, mtp, device_bytes, reserve);
    if (fits(plan(t, pl, max_len, int8_head, mtp), caps, reserve)) best = n;
  }
  return best;
}

uint32_t max_len_that_fits(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, bool int8_head, bool mtp,
                           const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap) {
  if (cap == 0 || cap > d.trained_max_len) cap = d.trained_max_len;
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("qwen4exp::max_len_that_fits: the cap " + std::to_string(cap) + " is below one " +
                                std::to_string(kMaxLenQuantum) + "-position quantum");
  const auto ok = [&](uint32_t len) { return fits(plan(d, p, len, int8_head, mtp), device_bytes, reserve); };
  // Every term is non-decreasing in max_len on every device: the quanta that fit are a prefix.
  uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum, hi = cap / kMaxLenQuantum;
  if (!ok(lo * kMaxLenQuantum)) return 0;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (ok(mid * kMaxLenQuantum))
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo * kMaxLenQuantum;
}

void require_fits(const std::vector<DevicePlan>& p, const std::array<size_t, kPpDevices>& device_bytes, size_t reserve) {
  if (fits(p, device_bytes, reserve)) return;
  size_t weights = 0, total = 0, caps = 0;
  std::string per;
  for (const DevicePlan& dp : p) {
    weights += dp.weights;
    total += dp.total();
    caps += device_bytes[dp.device];
    per += "; device " + std::to_string(dp.device) + " layers [" + std::to_string(dp.first) + ", " +
           std::to_string(dp.end) + "): " + std::to_string(dp.total()) + " B (" + gb(dp.total()) + ": weights " +
           gb(dp.weights) + ", KV " + gb(dp.kv) + ", state " + gb(dp.state) + ") + reserve " + gb(reserve) +
           " against " + std::to_string(device_bytes[dp.device]) + " B (" + gb(device_bytes[dp.device]) + ")";
  }
  const std::string head =
      p.empty() || !p[0].whole
          ? "Qwen3.8-Flash-Next at layers [0, " + std::to_string(p.empty() ? 0 : p.back().end) + ") does not fit " +
                std::to_string(p.size()) + " card(s): " + std::to_string(total) + " B planned"
          : "Qwen3.8-Flash-Next holds " + gb(weights) + " of weights (" + std::to_string(weights) +
                " B) and " + std::to_string(p.size()) + " B70(s) " + gb(caps) +
                ": it runs whole only with spec 22's expert-offload tier";
  throw std::runtime_error(head + per + " - use --layers N (the planner's N: runtime::qwen4exp::layers_that_fit); the "
                           "whole model needs spec 22");
}

std::string describe(const std::vector<DevicePlan>& p, const model::Q4Placement& pl, uint32_t max_len,
                     const std::array<size_t, kPpDevices>& device_bytes, size_t reserve) {
  char head[256];
  if (pl.devices > 1)
    std::snprintf(head, sizeof head, "pipeline plan at max_len %u, split %u (device 0 layers [0, %u), device 1 "
                  "layers [%u, %u)):", max_len, pl.split, pl.split, pl.split, pl.layers);
  else
    std::snprintf(head, sizeof head, "plan at max_len %u (one card, layers [0, %u)):", max_len, pl.layers);
  std::string out = head;
  for (const DevicePlan& dp : p) {
    const std::string label = "\n  device " + std::to_string(dp.device);
    out += format_memory(label.c_str(), dp, device_bytes[dp.device]);
    char tail[300];
    std::snprintf(tail, sizeof tail, "; + reserve %.3f GB = %.3f GB (weights %.3f GB%s, RoPE %.3f GB, KV + indexer "
                  "keys %.3f GB, GDN / PLE state %.3f GB, decode scratch %.3f GB%s)", reserve / 1e9,
                  (dp.total() + reserve) / 1e9, dp.weights / 1e9, dp.mtp ? " with the MTP head" : "", dp.rope / 1e9,
                  dp.kv / 1e9, dp.state / 1e9, dp.scratch / 1e9, dp.link ? ", hand-off" : "");
    out += tail;
  }
  return out;
}

}  // namespace runtime::qwen4exp
