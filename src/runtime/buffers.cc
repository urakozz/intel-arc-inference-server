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
// rows of the model table (spec §7's [S_max=16][8][34816]), not the max of
// S*N: one buffer has to fit whichever row is running. The bf16 rows are
// excluded because they do not use it - a||b writes ab_out, lm_head writes
// logits directly.
size_t partials_bytes() {
  uint32_t max_s = 1, max_n = 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(model::LinearId::kCount); ++i) {
    const model::FusedLinear& fl = Q::linear(static_cast<model::LinearId>(i));
    if (fl.kind != model::WeightKind::Int4) continue;
    if (fl.shape.S > max_s) max_s = fl.shape.S;
    if (fl.shape.N > max_n) max_n = fl.shape.N;
  }
  return size_t{max_s} * DecodeBuffers::kM * max_n * kFp32;
}

// KV blocks an attn_decode grid covers. Rounded up: a max_len that is not a
// multiple of kAttnBlock still needs a slot for its tail block.
uint32_t attn_blocks(uint32_t max_len) {
  return (max_len + DecodeBuffers::kAttnBlock - 1) / DecodeBuffers::kAttnBlock;
}
}  // namespace

DecodeBuffers::DecodeBuffers(l0::Context& ctx, uint32_t max_len)
    : control(ctx, l0::MemKind::Shared, sizeof(Control)),
      gdn_state(ctx, l0::MemKind::Device,
                size_t{kGdnLayers} * Q::kGdnVHeads * Q::kGdnHeadDim * Q::kGdnHeadDim * kFp32),
      conv_ring(ctx, l0::MemKind::Device, size_t{kGdnLayers} * kConvRing * kConvDim * kBf16),
      kv_k(ctx, l0::MemKind::Device,
           size_t{kFaLayers} * max_len * Q::kFaKvHeads * Q::kFaHeadDim * kBf16),
      kv_v(ctx, l0::MemKind::Device,
           size_t{kFaLayers} * max_len * Q::kFaKvHeads * Q::kFaHeadDim * kBf16),
      resid(ctx, l0::MemKind::Device, size_t{kM} * Q::kHidden * kBf16),
      x(ctx, l0::MemKind::Device, size_t{kM} * Q::kIntermediate * kBf16),
      partials(ctx, l0::MemKind::Device, partials_bytes()),
      ab_out(ctx, l0::MemKind::Device,
             size_t{kM} * Q::shape(model::LinearId::AB).N * kFp32),
      gdn_o(ctx, l0::MemKind::Device, size_t{kM} * kGdnValueDim * kFp32),
      attn_q(ctx, l0::MemKind::Device, size_t{kM} * kFaValueDim * kFp32),
      attn_gate(ctx, l0::MemKind::Device, size_t{kM} * kFaValueDim * kFp32),
      attn_part(ctx, l0::MemKind::Device,
                size_t{Q::kFaQHeads} * attn_blocks(max_len) * kM * kAttnPartStride * kFp32),
      attn_out(ctx, l0::MemKind::Device, size_t{kM} * kFaValueDim * kBf16),
      logits(ctx, l0::MemKind::Device, size_t{kM} * Q::kVocab * kFp32),
      argmax_part(ctx, l0::MemKind::Device, size_t{kM} * kArgmaxGroups * 2 * kFp32),
      max_len(max_len) {
  // A run starts from an empty GDN state, an empty conv ring, an empty KV
  // cache and pos = 0. Scratch is written before it is read every token, so
  // it is left alone. The immediate list is synchronous: the fills have
  // landed when the constructor returns.
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  for (l0::Mem* m : {&control, &gdn_state, &conv_ring, &kv_k, &kv_v})
    imm.fill(m->ptr(), 0u, m->size());
}

size_t DecodeBuffers::persistent_bytes() const {
  return control.size() + gdn_state.size() + conv_ring.size() + kv_k.size() + kv_v.size();
}

size_t DecodeBuffers::scratch_bytes() const {
  return resid.size() + x.size() + partials.size() + ab_out.size() + gdn_o.size() +
         attn_q.size() + attn_gate.size() + attn_part.size() + attn_out.size() + logits.size() +
         argmax_part.size();
}
}  // namespace runtime
