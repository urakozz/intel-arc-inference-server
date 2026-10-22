#include "runtime/bolt/bolt_sizes.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace runtime::bolt {
namespace {

uint32_t device_blocks(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, model::BoltBlock k) {
  uint32_t n = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) n += d.block.at(l) == k ? 1 : 0;
  return n;
}

[[noreturn]] void no_fit(const model::BoltDesc& d, const DevicePlan& p, uint32_t dev, uint32_t max_len, size_t device_bytes,
                         bool one_card) {
  char buf[900];
  std::snprintf(buf, sizeof buf,
                "runtime::bolt: the plan at max_len %u does not fit device %u's %zu B (weights %zu, kv %zu, ssm %zu, "
                "conv_rings %zu, decode_scratch %zu, prefill_scratch %zu, drafter %zu, drafter_embed_replica %zu, reserve "
                "%zu; total %zu)%s",
                max_len, dev, device_bytes, p.weights, p.kv, p.ssm, p.conv_rings, p.decode_scratch, p.prefill_scratch,
                p.drafter, p.drafter_embed_replica, p.reserve, p.total(),
                one_card ? " - " : "");
  std::string msg = buf;
  if (one_card)
    msg += d.name + " does not fit one 32 GB card (Decision 4: two cards by default, --pp 2; one card only through "
                    "spec 22's expert-offload tier, not built - spec 25 §4.2)";
  throw std::runtime_error(msg);
}

}  // namespace

DevicePlan device_plan(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len, bool int8_head,
                       const BoltReserve& extra, size_t reserve_bytes) {
  model::validate(p, d);
  if (max_len > d.max_pos)
    throw std::runtime_error("runtime::bolt::plan: max_len " + std::to_string(max_len) +
                             " exceeds the trained context " + std::to_string(d.max_pos) + " (max_position_embeddings)");
  DevicePlan pl;
  pl.weights = loader::bolt::device_weight_bytes(d, p, dev, int8_head);
  const uint32_t attn = device_blocks(d, p, dev, model::BoltBlock::Attention);
  const uint32_t mamba = device_blocks(d, p, dev, model::BoltBlock::Mamba);
  pl.kv = size_t(attn) * max_len * d.kv_heads * d.head_dim * 2 * 2;                 // K and V, bf16
  pl.ssm = loader::bolt::ssm_state_bytes(d, mamba) * (1 + size_t(extra.verify_slots)) / (extra.bf16_state ? 2 : 1);
  pl.conv_rings = loader::bolt::conv_ring_bytes(d, mamba);
  pl.decode_scratch = extra.decode_scratch;
  pl.prefill_scratch = extra.prefill_scratch;
  if (dev + 1 == p.devices) {
    pl.drafter = extra.drafter;
    pl.drafter_embed_replica = extra.drafter_embed_replica;
  }
  pl.reserve = reserve_bytes;
  return pl;
}

void require_fits(const DevicePlan& p, uint32_t dev, uint32_t max_len, size_t device_bytes) {
  if (p.total() <= device_bytes) return;
  char buf[700];
  std::snprintf(buf, sizeof buf,
                "runtime::bolt: the plan at max_len %u does not fit device %u's %zu B (weights %zu, kv %zu, ssm %zu, "
                "conv_rings %zu, decode_scratch %zu, prefill_scratch %zu, drafter %zu, drafter_embed_replica %zu, reserve "
                "%zu; total %zu)",
                max_len, dev, device_bytes, p.weights, p.kv, p.ssm, p.conv_rings, p.decode_scratch, p.prefill_scratch,
                p.drafter, p.drafter_embed_replica, p.reserve, p.total());
  throw std::runtime_error(buf);
}

uint32_t pp_split(const model::BoltDesc& d, bool int8_head) {
  if (d.layers < 2)
    throw std::invalid_argument(d.name + ": " + std::to_string(d.layers) + " block(s), nothing to split");
  std::vector<size_t> layer_bytes(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l) layer_bytes[l] = loader::bolt::layer_weight_bytes(d, l);
  const size_t dev0_fixed = loader::bolt::embed_tiled_bytes(d);
  const size_t dev1_fixed =
      loader::bolt::final_norm_bytes(d) + (int8_head ? loader::bolt::head_int8_bytes(d) : loader::bolt::head_bf16_bytes(d));
  return runtime::pp_balance(layer_bytes, dev0_fixed, dev1_fixed).split;
}

BoltPlan plan(const model::BoltDesc& d, uint32_t max_len, const std::vector<size_t>& device_free, const BoltReserve& extra,
              std::optional<uint32_t> split, size_t reserve_bytes, bool int8_head) {
  if (device_free.size() != 1 && device_free.size() != 2)
    throw std::invalid_argument("runtime::bolt::plan: " + std::to_string(device_free.size()) +
                                " devices (spec 25 §4.1 builds and gates two; one for synthetic / --layers checkpoints)");
  BoltPlan out;
  if (device_free.size() == 1) {
    if (split) throw std::invalid_argument("runtime::bolt::plan: a split on one device");
    out.place = model::BoltPlacement::one(d);
  } else {
    out.place = model::BoltPlacement::two(d, split ? *split : pp_split(d, int8_head));
  }
  model::validate(out.place, d);
  const bool one_card = device_free.size() == 1;
  // The largest max_len in [lo, hi] (multiples of 256) whose plan fits every device, or 0.
  const auto fits = [&](uint32_t len) {
    for (uint32_t dev = 0; dev < out.place.devices; ++dev)
      if (device_plan(d, out.place, dev, len, int8_head, extra, reserve_bytes).total() > device_free[dev]) return false;
    return true;
  };
  uint32_t len = max_len;
  if (len == 0) {
    const uint32_t cap = d.max_pos;
    if (cap < kMaxLenQuantum)
      throw std::invalid_argument("runtime::bolt::plan: the trained context " + std::to_string(cap) + " is below one " +
                                  std::to_string(kMaxLenQuantum) + "-position quantum");
    uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum, hi = cap / kMaxLenQuantum;
    if (fits(lo * kMaxLenQuantum)) {
      while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1) / 2;
        if (fits(mid * kMaxLenQuantum))
          lo = mid;
        else
          hi = mid - 1;
      }
      len = lo * kMaxLenQuantum;
    } else {
      len = std::min(kMinAutoMaxLen, cap);   // not even this fits: fall through to the naming refusal below
    }
  } else {
    if (len > d.max_pos)
      throw std::runtime_error("runtime::bolt::plan: --max-len " + std::to_string(len) + " exceeds the trained context " +
                               std::to_string(d.max_pos) + " (max_position_embeddings)");
    if (len % kMaxLenQuantum != 0)
      throw std::invalid_argument("runtime::bolt::plan: --max-len " + std::to_string(len) + " is not a multiple of " +
                                  std::to_string(kMaxLenQuantum) + " (the prefill attention's grid)");
  }
  out.max_len = len;
  for (uint32_t dev = 0; dev < out.place.devices; ++dev) {
    DevicePlan pl = device_plan(d, out.place, dev, len, int8_head, extra, reserve_bytes);
    if (pl.total() > device_free[dev]) no_fit(d, pl, dev, len, device_free[dev], one_card && d.layers == model::bolt().layers);
    out.device.push_back(pl);
  }
  return out;
}

std::string describe(const DevicePlan& p, size_t device_bytes) {
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "weights %.3f GB, kv %.3f GB, ssm %.3f GB, conv rings %zu B, scratch %.3f GB, drafter %.3f GB, reserve %.3f "
                "GB, total %.3f GB",
                double(p.weights) / 1e9, double(p.kv) / 1e9, double(p.ssm) / 1e9, p.conv_rings,
                double(p.decode_scratch + p.prefill_scratch) / 1e9, double(p.drafter + p.drafter_embed_replica) / 1e9,
                double(p.reserve) / 1e9, double(p.total()) / 1e9);
  std::string s = buf;
  if (device_bytes) s += " of " + std::to_string(double(device_bytes) / 1e9) + " GB";
  return s;
}

// ==== spec 25c: the decode list ==========================================================================================
namespace {
// The kernels' constants the sizes depend on (src/kernels/bolt_kernels.h is their home on the host; this file stays free
// of the kernel-dir define - bolt_capture.cc checks them against it).
constexpr uint32_t kAttnTgt = 64, kAttnPart = 130, kArgmaxChunk = 1024, kTopK = 9;
constexpr size_t kLine = 64;   // an empty group's allocation
size_t line(size_t b) { return b ? b : kLine; }
model::GemvShape shape_of(const model::BoltDesc& d, uint32_t l, model::BoltLinearId id) {
  for (const model::BoltLinear& ln : d.linears(l))
    if (ln.id == id) return ln.shape;
  return {0, 0, 0, 0};
}
}  // namespace

BoltAttn parse_attn(const std::string& v) {
  if (v.empty() || v == "flash") return BoltAttn::Flash;
  if (v == "eager") return BoltAttn::Eager;
  throw std::runtime_error("B70_BOLT_ATTN=" + v + ": expected flash or eager");
}
BoltAttn bolt_attn() {
  const char* v = std::getenv("B70_BOLT_ATTN");
  return parse_attn(v ? v : "");
}
const char* bolt_attn_name(BoltAttn a) { return a == BoltAttn::Eager ? "eager" : "flash"; }

BoltFuse parse_fuse(const std::string& v) {
  BoltFuse f;
  if (v.empty() || v == "none") return f;
  if (v == "all") return BoltFuse{true, true, true};
  size_t at = 0;
  while (at <= v.size()) {
    const size_t comma = v.find(',', at);
    const std::string w = v.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
    if (w == "router")
      f.router = true;
    else if (w == "route")
      f.route = true;
    else if (w == "conv")
      f.conv = true;
    else
      throw std::runtime_error("B70_BOLT_FUSE=" + v + ": '" + w +
                               "' is not an arm - expected none, all or a comma list of router, route, conv");
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  return f;
}
BoltFuse bolt_fuse() {
  const char* v = std::getenv("B70_BOLT_FUSE");
  return parse_fuse(v ? v : "");
}
std::string bolt_fuse_name(BoltFuse f) {
  if (!f.any()) return "none";
  if (f.router && f.route && f.conv) return "all";
  std::string s;
  if (f.router) s += "router";
  if (f.route) s += std::string(s.empty() ? "" : ",") + "route";
  if (f.conv) s += std::string(s.empty() ? "" : ",") + "conv";
  return s;
}

BoltState parse_state(const std::string& v) {
  if (v.empty() || v == "fp32") return BoltState::Fp32;
  if (v == "bf16") return BoltState::Bf16;
  throw std::runtime_error("B70_BOLT_SSM=" + v + ": expected fp32 or bf16 (Decision 2)");
}
BoltState bolt_state() {
  const char* v = std::getenv("B70_BOLT_SSM");
  return parse_state(v ? v : "");
}
const char* bolt_state_name(BoltState s) { return s == BoltState::Bf16 ? "bf16" : "fp32"; }

BoltPending pending_of(const model::BoltDesc& d, uint32_t l) {
  if (d.is_mamba(l)) return {shape_of(d, l, model::BoltLinearId::OutProj).S, 0};
  if (d.is_attention(l)) return {shape_of(d, l, model::BoltLinearId::OProj).S, 0};
  return {shape_of(d, l, model::BoltLinearId::Fc2).S, shape_of(d, l, model::BoltLinearId::SharedDown).S};
}
BoltPending cut_pending(const model::BoltDesc& d, const model::BoltPlacement& p) {
  if (p.devices < 2) return {};
  return pending_of(d, p.split - 1);
}
PpLandingLayout landing_layout(const model::BoltDesc& d, const model::BoltPlacement& p) {
  return pp_landing_layout(size_t(kM) * d.hidden * 2, size_t(cut_pending(d, p).total()) * kM * d.hidden * 4);
}

ScratchSizes scratch_sizes(const model::BoltDesc& d, uint32_t max_len, BoltAttn a) {
  ScratchSizes s;
  size_t widest = 0;
  uint32_t max_i = 0;
  bool mamba = false, attn = false, moe = false;
  for (uint32_t l = 0; l < d.layers; ++l) {
    for (const model::BoltLinear& ln : d.linears(l))
      if (ln.kind == model::WeightKind::Int4 && ln.id != model::BoltLinearId::ExpertUp &&
          ln.id != model::BoltLinearId::ExpertDown)
        widest = std::max(widest, size_t(ln.shape.S) * ln.shape.N);
    if (d.is_moe(l)) widest = std::max(widest, size_t(pending_of(d, l).total()) * d.hidden);
    mamba = mamba || d.is_mamba(l);
    attn = attn || d.is_attention(l);
    moe = moe || d.is_moe(l);
    max_i = std::max(max_i, d.inter.at(l));
  }
  const size_t M = kM;
  s.h0 = s.h1 = s.x = M * d.hidden * 2;
  s.slices = line(widest * M * 4);
  s.xc = line(mamba ? M * d.m_conv_dim * 2 : 0);
  s.y = s.g = line(mamba ? M * d.m_inner * 2 : 0);
  s.logits_r = line(moe ? M * d.max_experts() * 4 : 0);
  s.routes = line(size_t(d.layers) * M * kRouteWords * 4);
  s.lat = line(moe ? M * d.latent * 2 : 0);
  s.sh = line(moe ? M * d.shared_inter * 2 : 0);
  s.moe_a = line(moe ? M * kTopK * max_i * 2 : 0);
  s.moe_r = line(moe ? M * d.latent * 2 : 0);
  s.attn_q = line(attn ? M * d.q_n() * 4 : 0);
  s.attn_part = line(attn ? size_t(d.q_heads) * kAttnTgt * M * kAttnPart * 4 : 0);
  s.attn_out = line(attn ? M * d.q_n() * 2 : 0);
  s.ones = line(attn ? M * d.q_n() * 2 : 0);
  s.attn_scores = attn && a == BoltAttn::Eager ? M * d.q_heads * size_t(max_len) * 4 : 0;
  s.logits = M * d.vocab * 4;
  s.argmax_part = M * ((d.vocab + kArgmaxChunk - 1) / kArgmaxChunk) * 2 * 4;
  return s;
}

PersistentSizes persistent_sizes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len,
                                 BoltState st) {
  PersistentSizes s;
  s.control = 128;   // sizeof(runtime::Control), checked in bolt_buffers.cc
  const uint32_t attn = device_blocks(d, p, dev, model::BoltBlock::Attention);
  const uint32_t mamba = device_blocks(d, p, dev, model::BoltBlock::Mamba);
  s.kv_k = s.kv_v = line(size_t(attn) * kv_layer_bytes(d, max_len));
  s.conv_ring = line(size_t(mamba) * ring_layer_bytes(d));
  s.ssm = line(size_t(mamba) * ssm_layer_bytes(d, st));
  return s;
}

size_t decode_scratch_bytes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len,
                            BoltAttn a, BoltState st) {
  const PersistentSizes ps = persistent_sizes(d, p, dev, max_len, st);
  const uint32_t attn = device_blocks(d, p, dev, model::BoltBlock::Attention);
  const uint32_t mamba = device_blocks(d, p, dev, model::BoltBlock::Mamba);
  const size_t lines = (attn ? 0 : 2 * kLine) + (mamba ? 0 : 2 * kLine);   // what plan()'s exact terms leave out
  return ps.control + scratch_sizes(d, max_len, a).total() + lines;
}

size_t device_launches(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, BoltAttn a, BoltFuse f,
                       PpHandoff h) {
  model::validate(p, d);
  size_t n = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) {
    if (d.is_mamba(l))
      n += 6 - (f.conv ? 1 : 0);
    else if (d.is_attention(l))
      n += a == BoltAttn::Eager ? 5 : 6;
    else
      n += 9 - (f.router ? 1 : 0) - (f.route ? 1 : 0);
  }
  if (dev == 0) n += 1;                                     // nb_embed_gather
  if (dev + 1 == p.devices) n += 4;                         // the final norm, the head, the two argmax stages
  if (p.devices > 1 && h == PpHandoff::Peer) n += 1;        // pp_send (device 0) / pp_recv (device 1)
  return n;
}
size_t decode_launches(const model::BoltDesc& d, const model::BoltPlacement& p, BoltAttn a, BoltFuse f, PpHandoff h) {
  size_t n = 0;
  for (uint32_t dev = 0; dev < p.devices; ++dev) n += device_launches(d, p, dev, a, f, h);
  return n;
}

// ==== spec 25d ===========================================================================================================
uint32_t prefill_chunk_rows(uint32_t pos, size_t remaining, uint32_t chunk, bool hooked, uint32_t block) {
  uint32_t c = uint32_t(std::min<size_t>(chunk, remaining));
  if (c < remaining && (pos + c) % kPfSub != 0) c -= (pos + c) % kPfSub;
  if (hooked) c = std::min(c, block - pos % block);
  return c;
}
void check_prefill_chunk(uint32_t chunk) {
  if (chunk == 0 || chunk > kPfC || chunk % kPfSub != 0)
    throw std::runtime_error("runtime::bolt: prefill chunk " + std::to_string(chunk) + " is not a multiple of " +
                             std::to_string(kPfSub) + " in (0, " + std::to_string(kPfC) +
                             "] - the SSD's sub-chunks are absolute 64-row windows, and a chunk ends on that grid");
}

PrefillSizes prefill_sizes(const model::BoltDesc& d) {
  bool mamba = false, attn = false, moe = false;
  uint32_t max_i = 0, max_k = 0, max_ld = d.hidden;
  for (uint32_t l = 0; l < d.layers; ++l) {
    mamba = mamba || d.is_mamba(l);
    attn = attn || d.is_attention(l);
    moe = moe || d.is_moe(l);
    max_i = std::max(max_i, d.inter.at(l));
    for (const model::BoltLinear& ln : d.linears(l))
      if (ln.kind == model::WeightKind::Int4 && ln.id != model::BoltLinearId::ExpertUp &&
          ln.id != model::BoltLinearId::ExpertDown)
      {
        max_k = std::max(max_k, ln.shape.K);
        max_ld = std::max(max_ld, pf_ld(ln.shape.N));
      }
  }
  const size_t C = kPfC, H = d.hidden, E = d.max_experts(), T = pf_tiles(kPfC, d.max_experts());
  PrefillSizes s;
  s.ids = C * 4;
  s.h0 = s.h1 = s.x = C * H * 2;
  s.pa = C * max_ld * 4;   // the widest linear's pitch (in_proj's 18688 on the real model)
  s.pb = line(moe ? C * H * 4 : 0);
  s.slab = size_t(max_k) * kPfSlab * 2;
  s.seed = line(mamba ? size_t(3) * d.m_conv_dim * 2 : 0);
  s.xc = line(mamba ? C * d.m_conv_dim * 2 : 0);
  s.dt = s.cum = line(mamba ? C * d.m_heads * 4 : 0);
  s.cb = line(mamba ? size_t(kPfMaxSub) * d.m_groups * kPfSub * kPfSub * 4 : 0);
  s.y = line(mamba ? C * d.m_inner * 2 : attn ? C * d.q_n() * 2 : 0);
  s.g = line(mamba ? C * d.m_inner * 2 : 0);
  s.attn_q = line(attn ? C * d.q_n() * 4 : 0);
  s.logits = line(moe ? C * E * 4 : 0);
  s.routes = line(size_t(d.layers) * C * kRouteWords * 4);
  s.lat = line(moe ? C * d.latent * 2 : 0);
  s.sh = line(moe ? C * d.shared_inter * 2 : 0);
  s.r = line(moe ? C * d.latent * 2 : 0);
  s.hdr = line(moe ? size_t(kPfHdrWords) * 4 : 0);
  s.tiles = line(moe ? 2 * T * 4 : 0);
  s.row_tok = line(moe ? T * kPfTm * 4 : 0);
  s.pair_row = line(moe ? C * kTopK * 4 : 0);
  s.xg = line(moe ? T * kPfTm * d.latent * 2 : 0);
  s.moe_h = line(moe ? T * kPfTm * max_i * 2 : 0);
  s.w = line(moe ? kPfBatchBytes : 0);
  return s;
}

PpLandingLayout pf_landing_layout(const model::BoltDesc& d, const model::BoltPlacement& p) {
  const uint32_t pend = p.devices > 1 ? (d.is_moe(p.split - 1) ? 2u : 1u) : 0u;   // prefill folds S 1
  return pp_landing_layout(size_t(kPfC) * d.hidden * 2, size_t(pend) * kPfC * d.hidden * 4);
}
size_t pf_link_bytes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev) {
  if (p.devices < 2) return 0;
  return dev == 0 ? kPpStateWords * 4 : pf_landing_layout(d, p).total + kPpStateWords * 4;
}
size_t prefill_scratch_bytes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev) {
  return prefill_sizes(d).total() + pf_link_bytes(d, p, dev);
}

size_t prefill_device_launches(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev) {
  model::validate(p, d);
  size_t n = dev == 0 ? 1 : 0;   // nb_pf_embed_gather
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) {
    n += 1;   // the norm site
    for (const model::BoltLinear& ln : d.linears(l))
      if (ln.kind == model::WeightKind::Int4 && ln.id != model::BoltLinearId::ExpertUp &&
          ln.id != model::BoltLinearId::ExpertDown)
        n += 2 * pf_slabs(ln.shape.N);   // dequant + GEMM a slab
    if (d.is_mamba(l))
      n += 6;   // seed, conv, dt, C B^T, scan, gnorm
    else if (d.is_attention(l))
      n += 2;   // prep, flash (eager: one launch too)
    else
      n += 6 + 2 * 2 * pf_batches(d, l);   // router, route, sh_act, sort, gather, combine; up and down batches
  }
  return n;
}
size_t prefill_chunk_launches(const model::BoltDesc& d, const model::BoltPlacement& p) {
  size_t n = 0;
  for (uint32_t dev = 0; dev < p.devices; ++dev) n += prefill_device_launches(d, p, dev);
  return n;
}


// --- spec 25e: prefix-cache snapshots (the run lists; the engine copies them) ------------------------------------------------
namespace {
// Block l's index among the blocks of its kind that its device holds (the allocation's block order).
uint32_t snap_local_index(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t l) {
  uint32_t n = 0;
  for (uint32_t k = p.first(p.device_of(l)); k < l; ++k) n += d.block.at(k) == d.block.at(l) ? 1 : 0;
  return n;
}
}  // namespace

std::vector<SnapRun> state_runs(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t pos, BoltState s) {
  model::validate(p, d);
  constexpr uint32_t R = loader::bolt::kConvRing;
  if (kStateColumns + 1 > R)
    throw std::invalid_argument("runtime::bolt::state_runs: the ring's " + std::to_string(R) + " slots cannot hold the " +
                                std::to_string(kStateColumns) + " columns the next position reads plus the one it writes");
  const size_t row = size_t(d.m_conv_dim) * 2;
  std::vector<SnapRun> runs;
  for (uint32_t l = 0; l < d.layers; ++l) {
    if (!d.is_mamba(l)) continue;
    const uint32_t dev = p.device_of(l), idx = snap_local_index(d, p, l);
    runs.push_back({dev, SnapTensor::Ssm, size_t(idx) * ssm_layer_bytes(d, s), ssm_layer_bytes(d, s), false});
    for (uint32_t j = 0; j < kStateColumns; ++j) {
      // Position pos - 1 - j; below 0 a zero row, kept in the slot a decode at that position would not read.
      const int64_t q = int64_t(pos) - 1 - int64_t(j);
      const uint32_t slot = uint32_t((q % int64_t(R) + int64_t(R)) % int64_t(R));
      runs.push_back({dev, SnapTensor::ConvRing, size_t(idx) * ring_layer_bytes(d) + size_t(slot) * row, row, q < 0});
    }
  }
  return runs;
}

std::vector<SnapRun> kv_runs(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t max_len, uint32_t begin,
                             uint32_t end) {
  model::validate(p, d);
  if (begin > end || end > max_len)
    throw std::invalid_argument("runtime::bolt::kv_runs: range [" + std::to_string(begin) + ", " + std::to_string(end) +
                                ") is not within [0, max_len " + std::to_string(max_len) + ")");
  std::vector<SnapRun> runs;
  if (begin == end) return runs;
  const size_t row = size_t(d.kv_n()) * 2;
  for (SnapTensor t : {SnapTensor::KvK, SnapTensor::KvV}) {
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (!d.is_attention(l)) continue;
      const size_t base = size_t(snap_local_index(d, p, l)) * kv_layer_bytes(d, max_len);
      runs.push_back({p.device_of(l), t, base + size_t(begin) * row, size_t(end - begin) * row, false});
    }
  }
  return runs;
}

}  // namespace runtime::bolt
