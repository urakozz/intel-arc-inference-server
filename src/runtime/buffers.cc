#include "runtime/buffers.h"

#include <cstddef>
#include "l0/cmdlist.h"
#include "runtime/control.h"

// Every size here is derived from `model::Qwen35` and `max_len`; the only
// literals are element sizes and the two shapes the kernels fix rather than
// the model (the attention partial's `+2` and argmax's chunk), both named.
// Sizes that disagree with the kernels are silent corruption on token 2 -
// tests/runtime/buffers_test.cc pins the totals with the arithmetic spelled
// out, so a change has to be argued for there.
namespace runtime {
namespace {
using Q = model::Qwen35;

constexpr size_t kBf16 = 2, kFp32 = 4;

// is_fa(l) == (l % 4 == 3): one FA layer in four, the rest GDN.
constexpr uint32_t kFaLayers = Q::kLayers / 4;           // 16
constexpr uint32_t kGdnLayers = Q::kLayers - kFaLayers;  // 48

// GDN conv1d channels: q and k (kGdnKHeads each) plus v (kGdnVHeads), all at
// kGdnHeadDim. The z half of the qkv||z GEMV does not go through the conv.
constexpr uint32_t kConvDim = (2 * Q::kGdnKHeads + Q::kGdnVHeads) * Q::kGdnHeadDim;  // 10240
constexpr uint32_t kGdnValueDim = Q::kGdnVHeads * Q::kGdnHeadDim;                    // 6144
constexpr uint32_t kFaValueDim = Q::kFaQHeads * Q::kFaHeadDim;                       // 6144

// attn_decode writes a running (m, l) pair plus the head accumulator for each
// (q head, KV block, token); attn_reduce combines them.
constexpr uint32_t kAttnPartStride = Q::kFaHeadDim + 2;  // 258

// argmax_stage1 gives one work-group 1024 logits and writes one (max, index)
// pair per group (plan 3 task 3).
constexpr uint32_t kArgmaxChunk = 1024;
constexpr uint32_t kArgmaxGroups = (Q::kVocab + kArgmaxChunk - 1) / kArgmaxChunk;  // 243

// `partials` is the S-split accumulator every int4 GEMV writes before its prep
// kernel sums it. It is the conservative max-S x max-N rectangle over the int4
// rows of the model table (Task 4: [S_max=8][8][34816]), not the max of
// S*N: one buffer has to fit whichever row is running. The bf16 rows are
// excluded because they do not use it - a||b writes ab_out, lm_head writes
// logits directly.
//
// `rows` is DecodeScratch::kM for decode. The prefill path passes its own
// chunk width and **S = 1** (ruling R1): split-K exists to buy hardware
// threads at M = 1, and at M = C the M tile axis already saturates the grid,
// so an [S][C][N] rectangle at S = 8 would be 1.14 TB at C = 2048 for nothing.
size_t max_int4_n() {
  uint32_t max_n = 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(model::LinearId::kCount); ++i) {
    const model::FusedLinear& fl = Q::linear(static_cast<model::LinearId>(i));
    if (fl.kind != model::WeightKind::Int4) continue;
    if (fl.shape.N > max_n) max_n = fl.shape.N;
  }
  return max_n;
}

size_t partials_bytes() {
  uint32_t max_s = 1;
  for (uint32_t i = 0; i < static_cast<uint32_t>(model::LinearId::kCount); ++i) {
    const model::FusedLinear& fl = Q::linear(static_cast<model::LinearId>(i));
    if (fl.kind != model::WeightKind::Int4) continue;
    if (fl.shape.S > max_s) max_s = fl.shape.S;
  }
  return size_t{max_s} * DecodeScratch::kM * max_int4_n() * kFp32;
}

// KV blocks an attn_decode grid covers. Rounded up: a max_len that is not a
// multiple of kAttnBlock still needs a slot for its tail block.
uint32_t attn_blocks(uint32_t max_len) {
  return (max_len + DecodeScratch::kAttnBlock - 1) / DecodeScratch::kAttnBlock;
}
}  // namespace

// --- PersistentBuffers -------------------------------------------------------

PersistentBuffers::PersistentBuffers(l0::Context& ctx, uint32_t max_len)
    : control(ctx, l0::MemKind::Shared, sizeof(Control)),
      gdn_state(ctx, l0::MemKind::Device,
                size_t{kGdnLayers} * Q::kGdnVHeads * Q::kGdnHeadDim * Q::kGdnHeadDim * kFp32),
      conv_ring(ctx, l0::MemKind::Device, size_t{kGdnLayers} * kConvRing * kConvDim * kBf16),
      kv_k(ctx, l0::MemKind::Device,
           size_t{kFaLayers} * max_len * Q::kFaKvHeads * Q::kFaHeadDim * kBf16),
      kv_v(ctx, l0::MemKind::Device,
           size_t{kFaLayers} * max_len * Q::kFaKvHeads * Q::kFaHeadDim * kBf16),
      max_len(max_len) {
  // A run starts from an empty GDN state, an empty conv ring, an empty KV
  // cache and pos = 0. Scratch is written before it is read every token, so
  // it is left alone. The immediate list is synchronous: the fills have
  // landed when the constructor returns.
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  zero(imm);
}

void PersistentBuffers::zero(l0::CmdList& imm) {
  for (l0::Mem* m : {&control, &gdn_state, &conv_ring, &kv_k, &kv_v})
    imm.fill(m->ptr(), 0u, m->size());
}

size_t PersistentBuffers::bytes() const {
  return control.size() + gdn_state.size() + conv_ring.size() + kv_k.size() + kv_v.size();
}

// --- DecodeScratch -----------------------------------------------------------

DecodeScratch::DecodeScratch(l0::Context& ctx, uint32_t max_len)
    : resid(ctx, l0::MemKind::Device, size_t{kM} * Q::kHidden * kBf16),
      x(ctx, l0::MemKind::Device, size_t{kM} * Q::kIntermediate * kBf16),
      partials(ctx, l0::MemKind::Device, partials_bytes()),
      ab_out(ctx, l0::MemKind::Device, size_t{kM} * Q::shape(model::LinearId::AB).N * kFp32),
      // prep_res_fold's chunk sums-of-squares, one fp32 per (work-group,
      // token): 20 x 8 x 4 = 640 B, one allocation reused by all 129 sites
      // because the list is in-order and each site's stage B consumes what its
      // own stage A wrote before the next site's stage A overwrites it.
      norm_sumsq(ctx, l0::MemKind::Device, size_t{kNormGroups} * kM * kFp32),
      gdn_o(ctx, l0::MemKind::Device, size_t{kM} * kGdnValueDim * kFp32),
      attn_q(ctx, l0::MemKind::Device, size_t{kM} * kFaValueDim * kFp32),
      attn_gate(ctx, l0::MemKind::Device, size_t{kM} * kFaValueDim * kFp32),
      attn_part(ctx, l0::MemKind::Device,
                size_t{Q::kFaQHeads} * attn_blocks(max_len) * kM * kAttnPartStride * kFp32),
      attn_out(ctx, l0::MemKind::Device, size_t{kM} * kFaValueDim * kBf16),
      logits(ctx, l0::MemKind::Device, size_t{kM} * Q::kVocab * kFp32),
      argmax_part(ctx, l0::MemKind::Device, size_t{kM} * kArgmaxGroups * 2 * kFp32) {}

size_t DecodeScratch::bytes() const {
  return resid.size() + x.size() + partials.size() + ab_out.size() + norm_sumsq.size() +
         gdn_o.size() + attn_q.size() + attn_gate.size() + attn_part.size() + attn_out.size() +
         logits.size() + argmax_part.size();
}

// --- PrefillScratch ----------------------------------------------------------

PrefillScratch::PrefillScratch(l0::Context& ctx, uint32_t max_len)
    // `ids` is the one Host allocation: the host writes C ids per chunk and
    // the device gathers embeddings from them.
    : ids(ctx, l0::MemKind::Host, size_t{kC} * sizeof(uint32_t)),
      resid(ctx, l0::MemKind::Device, size_t{kC} * Q::kHidden * kBf16),
      x(ctx, l0::MemKind::Device, size_t{kC} * Q::kIntermediate * kBf16),
      // S = 1 (ruling R1), so the rectangle is [C][max int4 N] and not
      // [S][C][N] - the sole reason a 2048-row chunk fits at all.
      partials(ctx, l0::MemKind::Device, size_t{kC} * max_int4_n() * kFp32),
      ab_out(ctx, l0::MemKind::Device, size_t{kC} * Q::shape(model::LinearId::AB).N * kFp32),
      norm_sumsq(ctx, l0::MemKind::Device, size_t{kNormGroups} * kC * kFp32),
      gdn_o(ctx, l0::MemKind::Device, size_t{kC} * kGdnValueDim * kFp32),
      mixer_out(ctx, l0::MemKind::Device, size_t{kC} * kGdnValueDim * kBf16),
      // lm_head runs on the chunk's LAST position only (spec §3.5), so one row.
      logits(ctx, l0::MemKind::Device, size_t{Q::kVocab} * kFp32),
      argmax_part(ctx, l0::MemKind::Device, size_t{kArgmaxGroups} * 2 * kFp32),
      gdn_xb(ctx, l0::MemKind::Device, size_t{kC} * kConvDim * kBf16),
      gdn_seed(ctx, l0::MemKind::Device, size_t{3} * kConvDim * kBf16),
      gdn_g(ctx, l0::MemKind::Device, size_t{kC} * Q::kGdnVHeads * kFp32),
      gdn_beta(ctx, l0::MemKind::Device, size_t{kC} * Q::kGdnVHeads * kFp32),
      gdn_A(ctx, l0::MemKind::Device,
            size_t{kC / kGdnChunk} * Q::kGdnVHeads * kGdnChunk * kGdnChunk * kFp32),
      gdn_A2(ctx, l0::MemKind::Device,
             size_t{kC / kGdnChunk} * Q::kGdnVHeads * kGdnChunk * kGdnChunk * kFp32),
      gdn_w(ctx, l0::MemKind::Device, size_t{kC} * kGdnValueDim * kBf16),
      gdn_u(ctx, l0::MemKind::Device, size_t{kC} * kGdnValueDim * kBf16),
      pf_q(ctx, l0::MemKind::Device, size_t{kC} * kFaValueDim * kBf16),
      pf_attn(ctx, l0::MemKind::Device, size_t{kC} * kFaValueDim * kBf16),
      pf_s(ctx, l0::MemKind::Device, size_t{kSHeads} * kC * max_len * kFp32),
      pf_p(ctx, l0::MemKind::Device, size_t{kSHeads} * kC * max_len * kBf16),
      pf_o(ctx, l0::MemKind::Device, size_t{Q::kFaQHeads} * kC * Q::kFaHeadDim * kFp32),
      pf_rowsum(ctx, l0::MemKind::Device, size_t{Q::kFaQHeads} * kC * kFp32),
      max_len(max_len),
      ctx_(&ctx) {
  // Nothing is zero-filled: no prefill kernel reads scratch it has not first
  // written, and the prefill determinism gate is the standing proof of that -
  // the same rule, and the same reason, Engine::reset() gives for decode.
  static_assert(kC % kGdnChunk == 0, "kC must be a whole number of 64-position GDN chunks");
}

size_t PrefillScratch::bytes() const {
  return ids.size() + resid.size() + x.size() + partials.size() + ab_out.size() +
         norm_sumsq.size() + gdn_o.size() + mixer_out.size() + logits.size() +
         argmax_part.size() + gdn_xb.size() + gdn_seed.size() + gdn_g.size() +
         gdn_beta.size() + gdn_A.size() + gdn_A2.size() + gdn_w.size() + gdn_u.size() +
         pf_q.size() + pf_attn.size() + pf_s.size() + pf_p.size() + pf_o.size() +
         pf_rowsum.size();
}

l0::Mem& PrefillScratch::dequant_buffer() {
  if (!dequant_)
    dequant_ = std::make_unique<l0::Mem>(*ctx_, l0::MemKind::Device,
                                         size_t{Q::kHidden} * max_int4_n() * kBf16);
  return *dequant_;
}
l0::Mem& PrefillScratch::slab_buffer() {
  if (!slab_)
    slab_ = std::make_unique<l0::Mem>(*ctx_, l0::MemKind::Device,
                                      size_t{Q::kIntermediate} * 1024 * kBf16);
  return *slab_;
}
size_t PrefillScratch::lazy_bytes() const {
  return (dequant_ ? dequant_->size() : 0) + (slab_ ? slab_->size() : 0);
}

// --- DecodeBuffers, the view -------------------------------------------------

DecodeBuffers::DecodeBuffers(l0::Context& ctx, uint32_t max_len)
    : own_p_(new PersistentBuffers(ctx, max_len)),
      own_s_(new DecodeScratch(ctx, max_len)),
      control(own_p_->control),
      gdn_state(own_p_->gdn_state),
      conv_ring(own_p_->conv_ring),
      kv_k(own_p_->kv_k),
      kv_v(own_p_->kv_v),
      resid(own_s_->resid),
      x(own_s_->x),
      partials(own_s_->partials),
      ab_out(own_s_->ab_out),
      norm_sumsq(own_s_->norm_sumsq),
      gdn_o(own_s_->gdn_o),
      attn_q(own_s_->attn_q),
      attn_gate(own_s_->attn_gate),
      attn_part(own_s_->attn_part),
      attn_out(own_s_->attn_out),
      logits(own_s_->logits),
      argmax_part(own_s_->argmax_part),
      max_len(max_len) {}

DecodeBuffers::DecodeBuffers(PersistentBuffers& p, DecodeScratch& s)
    : control(p.control),
      gdn_state(p.gdn_state),
      conv_ring(p.conv_ring),
      kv_k(p.kv_k),
      kv_v(p.kv_v),
      resid(s.resid),
      x(s.x),
      partials(s.partials),
      ab_out(s.ab_out),
      norm_sumsq(s.norm_sumsq),
      gdn_o(s.gdn_o),
      attn_q(s.attn_q),
      attn_gate(s.attn_gate),
      attn_part(s.attn_part),
      attn_out(s.attn_out),
      logits(s.logits),
      argmax_part(s.argmax_part),
      max_len(p.max_len) {}

size_t DecodeBuffers::persistent_bytes() const {
  return control.size() + gdn_state.size() + conv_ring.size() + kv_k.size() + kv_v.size();
}

size_t DecodeBuffers::scratch_bytes() const {
  return resid.size() + x.size() + partials.size() + ab_out.size() + norm_sumsq.size() +
         gdn_o.size() + attn_q.size() + attn_gate.size() + attn_part.size() + attn_out.size() +
         logits.size() + argmax_part.size();
}
}  // namespace runtime
