#include "runtime/qwen4exp/qwen4exp_buffers.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "l0/cmdlist.h"

namespace runtime::qwen4exp {
namespace {
// Layer `layer`'s index among this device's QSA (or GDN) layers.
uint32_t kind_index(const model::Qwen4ExpDesc& d, uint32_t first, uint32_t layer) {
  const bool qsa = d.is_qsa(layer);
  uint32_t n = 0;
  for (uint32_t l = first; l < layer; ++l) n += d.is_qsa(l) == qsa ? 1 : 0;
  return n;
}
size_t kv_rows_bytes(const model::Qwen4ExpDesc& d, uint32_t max_len) { return size_t(max_len) * d.kv_n() * 2; }
}  // namespace

#define Q4_PS persistent_sizes(d, p, dev, len)
#define Q4_SS scratch_sizes(d, len, at, mtp ? kVerifyRows : kM)
Qwen4ExpBuffers::Qwen4ExpBuffers(l0::Context& ctx, const model::Qwen4ExpDesc& d, const model::Q4Placement& p,
                                 uint32_t dev, uint32_t len, Q4Attn at, bool mtp)
    : control(ctx, l0::MemKind::Shared, Q4_PS.control),
      kv(ctx, l0::MemKind::Device, Q4_PS.kv),
      idx_keys(ctx, l0::MemKind::Device, Q4_PS.idx_keys),
      idx_tail(ctx, l0::MemKind::Device, Q4_PS.idx_tail),
      gdn_state(ctx, l0::MemKind::Device, Q4_PS.gdn_state),
      conv_ring(ctx, l0::MemKind::Device, Q4_PS.conv_ring),
      ple(ctx, l0::MemKind::Device, Q4_PS.ple),
      H(ctx, l0::MemKind::Device, Q4_SS.H),
      xn(ctx, l0::MemKind::Device, Q4_SS.xn),
      x(ctx, l0::MemKind::Device, Q4_SS.x),
      partials(ctx, l0::MemKind::Device, Q4_SS.partials),
      down_f32(ctx, l0::MemKind::Device, Q4_SS.down_f32),
      inj(ctx, l0::MemKind::Device, Q4_SS.inj),
      ab(ctx, l0::MemKind::Device, Q4_SS.ab),
      gdn_o(ctx, l0::MemKind::Device, Q4_SS.gdn_o),
      idx_f32(ctx, l0::MemKind::Device, Q4_SS.idx_f32),
      idx_q(ctx, l0::MemKind::Device, Q4_SS.idx_q),
      scores(ctx, l0::MemKind::Device, Q4_SS.scores),
      list(ctx, l0::MemKind::Device, Q4_SS.list),
      diag(ctx, l0::MemKind::Device, Q4_SS.diag),
      attn_q(ctx, l0::MemKind::Device, Q4_SS.attn_q),
      attn_gate(ctx, l0::MemKind::Device, Q4_SS.attn_gate),
      attn_part(ctx, l0::MemKind::Device, Q4_SS.attn_part),
      attn_out(ctx, l0::MemKind::Device, Q4_SS.attn_out),
      logits_r(ctx, l0::MemKind::Device, Q4_SS.logits_r),
      routes(ctx, l0::MemKind::Device, Q4_SS.routes),
      moe_h(ctx, l0::MemKind::Device, Q4_SS.moe_h),
      y(ctx, l0::MemKind::Device, Q4_SS.y),
      ple_e(ctx, l0::MemKind::Device, Q4_SS.ple_e),
      ple_kv(ctx, l0::MemKind::Device, Q4_SS.ple_kv),
      ple_ids(ctx, l0::MemKind::Device, Q4_SS.ple_ids),
      ple_consts(ctx, l0::MemKind::Device, Q4_SS.ple_consts),
      logits(ctx, l0::MemKind::Device, Q4_SS.logits),
      argmax_part(ctx, l0::MemKind::Device, Q4_SS.argmax_part),
      desc(d),
      placement(p),
      device(dev),
      max_len(len),
      attn(at),
      rows(mtp ? kVerifyRows : kM) {
  model::validate(p, d);
  if (len % d.idx_compress != 0)
    throw std::invalid_argument("Qwen4ExpBuffers: max_len " + std::to_string(len) + " is not whole indexer blocks");
  if (mtp) {   // spec 21e: the verify's GDN slots and (the PLE layer's device) its PLE rows
    gdn_spec = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, gdn_spec_bytes(d, p, dev));
    if (holds_ple()) {
      const size_t half = ple_verify_bytes(d, p, dev) / 2;
      ple_gated = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, half);
      ple_gn = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, half);
    }
  }
}
#undef Q4_PS
#undef Q4_SS

static void require_kind(const Qwen4ExpBuffers& b, uint32_t layer, bool qsa, const char* what) {
  if (!b.holds(layer))
    throw std::logic_error(std::string("Qwen4ExpBuffers::") + what + ": layer " + std::to_string(layer) +
                           " is not on device " + std::to_string(b.device));
  if (b.desc.is_qsa(layer) != qsa)
    throw std::logic_error(std::string("Qwen4ExpBuffers::") + what + ": layer " + std::to_string(layer) + " is a " +
                           (qsa ? "GDN" : "QSA") + " layer");
}

void* Qwen4ExpBuffers::k_layer(uint32_t layer) const {
  require_kind(*this, layer, true, "k_layer");
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return static_cast<uint8_t*>(kv.ptr()) + size_t(i) * 2 * kv_rows_bytes(desc, max_len);
}
void* Qwen4ExpBuffers::v_layer(uint32_t layer) const {
  return static_cast<uint8_t*>(k_layer(layer)) + kv_rows_bytes(desc, max_len);
}
void* Qwen4ExpBuffers::idx_keys_layer(uint32_t layer) const {
  require_kind(*this, layer, true, "idx_keys_layer");
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return static_cast<uint8_t*>(idx_keys.ptr()) + size_t(i) * (max_len / desc.idx_compress) * desc.idx_dim * 2;
}
void* Qwen4ExpBuffers::tail_layer(uint32_t layer) const {
  require_kind(*this, layer, true, "tail_layer");
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return static_cast<uint8_t*>(idx_tail.ptr()) + size_t(i) * kIdxTail * desc.idx_dim * 2;
}
void* Qwen4ExpBuffers::gdn_state_layer(uint32_t layer) const {
  require_kind(*this, layer, false, "gdn_state_layer");
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return static_cast<uint8_t*>(gdn_state.ptr()) + size_t(i) * gdn_state_bytes_per_layer(desc);
}
void* Qwen4ExpBuffers::conv_ring_layer(uint32_t layer) const {
  require_kind(*this, layer, false, "conv_ring_layer");
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return static_cast<uint8_t*>(conv_ring.ptr()) + size_t(i) * conv_ring_bytes_per_layer(desc);
}

void* Qwen4ExpBuffers::gdn_spec_layer(uint32_t layer) const {
  require_kind(*this, layer, false, "gdn_spec_layer");
  if (!gdn_spec) throw std::logic_error("Qwen4ExpBuffers::gdn_spec_layer: no verify slots (the MTP head is not loaded)");
  const uint32_t i = kind_index(desc, placement.first(device), layer);
  return static_cast<uint8_t*>(gdn_spec->ptr()) + size_t(i) * (kGdnSlots - 1) * gdn_state_bytes_per_layer(desc);
}
void* Qwen4ExpBuffers::gdn_slot(uint32_t layer, uint32_t s) const {
  if (s == 0) return gdn_state_layer(layer);
  if (s >= kGdnSlots) throw std::logic_error("Qwen4ExpBuffers::gdn_slot: slot " + std::to_string(s));
  return static_cast<uint8_t*>(gdn_spec_layer(layer)) + size_t(s - 1) * gdn_state_bytes_per_layer(desc);
}

size_t Qwen4ExpBuffers::mtp_bytes() const {
  return (gdn_spec ? gdn_spec->size() : 0) + (ple_gated ? ple_gated->size() + ple_gn->size() : 0);
}

size_t Qwen4ExpBuffers::persistent_bytes() const {
  return control.size() + kv.size() + idx_keys.size() + idx_tail.size() + gdn_state.size() + conv_ring.size() + ple.size();
}
size_t Qwen4ExpBuffers::scratch_bytes() const {
  size_t n = 0;
  for (const l0::Mem* m : {&H, &xn, &x, &partials, &down_f32, &inj, &ab, &gdn_o, &idx_f32, &idx_q, &scores, &list, &diag,
                           &attn_q, &attn_gate, &attn_part, &attn_out, &logits_r, &routes, &moe_h, &y, &ple_e, &ple_kv,
                           &ple_ids, &ple_consts, &logits, &argmax_part})
    n += m->size();
  return n;
}

void Qwen4ExpBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&control, &kv, &idx_keys, &idx_tail, &gdn_state, &conv_ring, &ple}) imm.fill(m->ptr(), 0u, m->size());
}

// --- spec 21e: the MTP head's buffers ------------------------------------------------------------------------------
#define Q4_MS mtp_sizes(d, len)
Qwen4ExpMtpBuffers::Qwen4ExpMtpBuffers(l0::Context& ctx, const model::Qwen4ExpDesc& d, uint32_t len)
    : hctl(ctx, l0::MemKind::Shared, Q4_MS.ctl),
      kv(ctx, l0::MemKind::Device, Q4_MS.kv),
      idx_keys(ctx, l0::MemKind::Device, Q4_MS.idx_keys),
      idx_tail(ctx, l0::MemKind::Device, Q4_MS.idx_tail),
      hh(ctx, l0::MemKind::Device, Q4_MS.hh),
      list(ctx, l0::MemKind::Device, Q4_MS.list),
      diag(ctx, l0::MemKind::Device, Q4_MS.diag),
      xe(ctx, l0::MemKind::Device, Q4_MS.xe),
      xh(ctx, l0::MemKind::Device, Q4_MS.xh),
      fe(ctx, l0::MemKind::Device, Q4_MS.fe),
      fh(ctx, l0::MemKind::Device, Q4_MS.fh),
      logits(ctx, l0::MemKind::Device, Q4_MS.logits),
      routes(ctx, l0::MemKind::Device, Q4_MS.routes),
      desc(d),
      max_len(len) {
  if (len % d.idx_compress != 0)
    throw std::invalid_argument("Qwen4ExpMtpBuffers: max_len " + std::to_string(len) + " is not whole indexer blocks");
}
#undef Q4_MS

size_t Qwen4ExpMtpBuffers::bytes() const {
  size_t n = 0;
  for (const l0::Mem* m : {&hctl, &kv, &idx_keys, &idx_tail, &hh, &list, &diag, &xe, &xh, &fe, &fh, &logits, &routes})
    n += m->size();
  return n;
}

void Qwen4ExpMtpBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&hctl, &kv, &idx_keys, &idx_tail, &hh}) imm.fill(m->ptr(), 0u, m->size());
}

}  // namespace runtime::qwen4exp
