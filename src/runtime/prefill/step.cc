#include "runtime/prefill/step.h"

#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "loader/small_layout.h"
#include "model/qwen35.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill/dequant.h"
#include "runtime/prefill/gdn.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/linear_l0.h"
#include "runtime/prefill/profile.h"

// The per-chunk walk: `capture.cc`'s decode order (`:472-591`), at runtime `M`,
// at S = 1, with the int4 GEMVs replaced by the two-pass
// `dequant_to_bf16` -> `gemm_bf16` (rulings A23/A24) and the attention replaced
// by the composed path (ruling A14).
//
// **The one structural difference from `capture.cc`, and it is not cosmetic.**
// A captured decode list is 774 launches on ONE queue with a single submit; this
// walk interleaves two runtimes on one hardware compute queue. Every
// `dequant -> gemm` and every `gemm -> consumer` boundary is a host
// `Context::wait()` because there is nothing else on this device to order them
// with (A24, Probe B: `zeDeviceGetCommandQueueGroupProperties` reports exactly
// one COMPUTE group with `numQueues = 1`). The waits are counted and reported
// (`step_chunk_waits()`) rather than hidden, because at a measured 22.35 us
// each they are a real term in the chunk's cost.
namespace runtime::prefill {
namespace {

using loader::DeviceWeight;
using loader::LoadedModel;
using model::LinearId;
using model::Qwen35;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::step_chunk: " + what);
}

uint8_t* at(const l0::Mem& m, size_t off) {
  return static_cast<uint8_t*>(m.ptr()) + off;
}
const void* at_const(const l0::Mem& m, size_t off) {
  return static_cast<const uint8_t*>(m.ptr()) + off;
}

// One int4 linear on the prefill path -- THE seam of spec 2.1 §3.2. The sycl-tla side is
// spec 2's two-pass path unchanged (backend_sycl.cc); the L0 side is S2's slab walk.
void pf_linear(Context& cx, KernelCache& kc, PrefillScratch& s, const DeviceWeight& w,
               const uint16_t* x, uint32_t M, PrefillBackend backend) {
  if (backend == PrefillBackend::SyclTla) {
    linear_sycl(cx, kc, s, w, x, M);
    return;
  }
  linear_l0(cx, kc, s, w, x, M);
}

// The res_norm pair at `M` rows. `s_prev` is 0 for layer 0's leading norm --
// nothing has been written into `partials` yet -- and 1 everywhere else, because
// every producer on this path is S = 1 (ruling R1).
void pf_res_norm(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t s_prev,
                 const void* norm_w, const void* partials, void* resid, void* x, uint32_t M) {
  const uint32_t g = PrefillScratch::kNormGroups;
  cx.launch(kc(kernels::pf_res_fold_variant(Qwen35::kHidden, s_prev, g), "pf_res_fold"), g, M, 1,
            {PtrArg(partials), PtrArg(resid), PtrArg(s.norm_sumsq.ptr()), arg_val(M)});
  cx.launch(kc(kernels::pf_norm_finish_variant(Qwen35::kHidden, g, g), "pf_norm_finish"), g, M, 1,
            {PtrArg(s.norm_sumsq.ptr()), PtrArg(resid), PtrArg(norm_w), PtrArg(x), arg_val(M)});
  profile_wait(cx, Phase::kNorm);
}

constexpr uint32_t kSiluChunk = 4096;

}  // namespace

void step_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, const LoadedModel& m,
                uint32_t max_len, void* ctrl, uint32_t pos, uint32_t C, l0::Mem& gdn_state_mem,
                l0::Mem& conv_ring_mem, l0::Mem& kv_k_mem, l0::Mem& kv_v_mem,
                PrefillBackend backend) {
  require(C > 0 && C <= PrefillScratch::kC,
          "C = " + std::to_string(C) + " is outside (0, PrefillScratch::kC]");
  require(size_t(pos) + C <= size_t(max_len), "pos + C exceeds max_len");

  // The per-layer slice strides, re-derived from `model::Qwen35` here rather
  // than shared with buffers.cc -- capture.cc:105-116's arrangement, and for
  // its reason: a divergence is then a throw from the three `require`s below,
  // not a wrong KV slot at position 3000.
  const size_t kv_stride = size_t(max_len) * Qwen35::kFaKvHeads * Qwen35::kFaHeadDim * 2;
  const size_t gdn_state_stride =
      size_t(Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim * 4;
  const size_t conv_ring_stride = size_t(PersistentBuffers::kConvRing) * 10240 * 2;
  require(kv_stride * 16 == kv_k_mem.size(), "kv_k stride x 16 FA layers != its allocation");
  require(kv_stride * 16 == kv_v_mem.size(), "kv_v stride x 16 FA layers != its allocation");
  require(gdn_state_stride * 48 == gdn_state_mem.size(),
          "gdn_state stride x 48 GDN layers != its allocation");
  require(conv_ring_stride * 48 == conv_ring_mem.size(),
          "conv_ring stride x 48 GDN layers != its allocation");

  // embed: ids -> resid, bf16 [C][5120].
  cx.launch(kc(kernels::pf_embed_gather_variant(), "pf_embed_gather"), 1, C, 1,
            {PtrArg(s.ids.ptr()), PtrArg(m.embed.ptr()), PtrArg(s.resid.ptr()), arg_val(C)});

  uint32_t gdn = 0, fa = 0;
  for (const model::LayerDesc& L : Qwen35::layers()) {
    const uint32_t l = L.index;
    pf_res_norm(cx, kc, s, l == 0 ? 0u : 1u,
                at_const(m.layer_small[l].norms, loader::kNormsOffInput), s.partials.ptr(),
                s.resid.ptr(), s.x.ptr(), C);                     // x stride 5120

    if (L.kind == model::LayerKind::GDN) {
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::QkvZ}), s.x.as<uint16_t>(), C, backend);
      {   // a||b: a bf16 GEMV, not a linear -- 128 columns is no DPAS shape.
        const DeviceWeight& w = m.linears.at({l, LinearId::AB});
        cx.launch(kc(kernels::pf_ab_proj_variant(), "pf_ab_proj"), w.shape.N / 16, (C + 7) / 8, 1,
                  {PtrArg(w.mem.ptr()), PtrArg(s.x.ptr()), PtrArg(s.ab_out.ptr()), arg_val(C)});
        profile_wait(cx, Phase::kAbGdn);
      }
      gdn_chunk(cx, kc, s, pos, C, s.partials.as<float>(), s.ab_out.as<float>(),
                reinterpret_cast<float*>(at(gdn_state_mem, size_t(gdn) * gdn_state_stride)),
                reinterpret_cast<uint16_t*>(at(conv_ring_mem, size_t(gdn) * conv_ring_stride)),
                m.layer_small[l].gdn.ptr(), s.mixer_out.as<uint16_t>());   // y stride 6144
      ++gdn;
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::OutProj}), s.mixer_out.as<uint16_t>(), C,
               backend);
    } else {
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::Qkv}), s.x.as<uint16_t>(), C, backend);
      uint16_t* kk = reinterpret_cast<uint16_t*>(at(kv_k_mem, size_t(fa) * kv_stride));
      uint16_t* vv = reinterpret_cast<uint16_t*>(at(kv_v_mem, size_t(fa) * kv_stride));
      // `Control::{pos, n_active}` were set by the caller before this chunk and
      // are what `pf_attn_prep` reads; the whole chunk is one attention pass, so
      // nothing here re-writes them (plan 6b's `kAttnC` sub-chunk loop was
      // ruling R5's, and ruling A14 retired it with the M = 64 route).
      attn_prep_chunk(cx, kc, s, C, ctrl, s.partials.as<float>(),
                      m.layer_small[l].gdn.as<float>(), m.rope.as<float>(), kk, vv);
      profile_wait(cx, Phase::kAttnPrep);
      attn_chunk(cx, kc, s, pos, C, s.pf_q.as<uint16_t>(), kk, vv, backend);
      attn_gate_chunk(cx, kc, s, C, attn_rows(C, backend), s.partials.as<float>(),
                      s.mixer_out.as<uint16_t>());
      profile_wait(cx, Phase::kAttnGate);
      ++fa;
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::OProj}), s.mixer_out.as<uint16_t>(), C,
               backend);
    }

    // The MLP half, identical in both layer kinds.
    pf_res_norm(cx, kc, s, 1u, at_const(m.layer_small[l].norms, loader::kNormsOffPost),
                s.partials.ptr(), s.resid.ptr(), s.x.ptr(), C);   // x stride 5120
    pf_linear(cx, kc, s, m.linears.at({l, LinearId::GateUp}), s.x.as<uint16_t>(), C, backend);
    cx.launch(kc(kernels::pf_silu_mul_variant(), "pf_silu_mul"),
              (Qwen35::kIntermediate + kSiluChunk - 1) / kSiluChunk, C, 1,
              {PtrArg(s.partials.ptr()), PtrArg(s.x.ptr()), arg_val(C)});   // x stride 17408
    profile_wait(cx, Phase::kSilu);
    pf_linear(cx, kc, s, m.linears.at({l, LinearId::Down}), s.x.as<uint16_t>(), C, backend);
  }
  require(gdn == 48 && fa == 16, "the layer table did not give 48 GDN and 16 FA layers");
}

void step_head(Context& cx, KernelCache& kc, PrefillScratch& s, const LoadedModel& m, void* ctrl,
               uint32_t last_row) {
  // The final norm, ONE row. `sumsq[g * 1 + 0] == sumsq[g]`, which is exactly
  // what the M = 1 pair reads.
  pf_res_norm(cx, kc, s, 1u, m.final_norm.ptr(),
              at(s.partials, size_t(last_row) * Qwen35::kHidden * 4),
              at(s.resid, size_t(last_row) * Qwen35::kHidden * 2),
              at(s.x, size_t(last_row) * Qwen35::kHidden * 2), 1u);

  // lm_head at M = 1 through the EXISTING decode binary, chosen off the loaded
  // weight's kind exactly as capture.cc:625-631 chooses it. It writes straight
  // into `logits`: `gemv.cl`'s `out[(s*M + m)*N + n]` at S = 1, M = 1 IS the
  // [1][N] row `argmax_stage1` reads. **Not `dequant_to_bf16` + `gemm_bf16`**:
  // the lm_head's [5120][248320] bf16 expansion is 2.54 GB, seven times the
  // dequant scratch, for one row of output.
  const DeviceWeight& lm = m.linears.at({loader::kTopLevel, LinearId::LmHead});
  const void* xrow = at(s.x, size_t(last_row) * Qwen35::kHidden * 2);
  if (lm.kind == model::WeightKind::Int4) {
    cx.launch(kc(kernels::gemv_variant(1, lm.shape.K, lm.shape.N, lm.shape.S, lm.shape.layout),
                 "gemv"),
              lm.shape.N / 64, lm.shape.S, 1,
              {PtrArg(lm.mem.ptr()),
               PtrArg(lm.shape.layout == 0 ? lm.scales->ptr() : lm.mem.ptr()), PtrArg(xrow),
               PtrArg(s.logits.ptr())});
  } else {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(lm.shape.N);
    cx.launch(kc(kernels::gemv_bf16_variant(1, lm.shape.K, lm.shape.N, t), "gemv_bf16"),
              lm.shape.N / t.cols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(xrow), PtrArg(s.logits.ptr())});
  }
  // The sampler, byte for byte decode's. `argmax_stage2` is the ONLY writer of
  // `pos` and `cur_token` and is therefore the last launch of the step: the
  // caller has set pos = base + L - 1 and n_active = 1, so stage 2 leaves
  // pos = base + L and the first generated id in cur_token[0] -- exactly what
  // the last ingest-by-decode replay would have left.
  cx.launch(kc(kernels::argmax_stage1_variant(1), "argmax_stage1"), (Qwen35::kVocab + 1023) / 1024,
            1, 1, {PtrArg(s.logits.ptr()), PtrArg(s.argmax_part.ptr())});
  cx.launch(kc(kernels::argmax_stage2_variant(), "argmax_stage2"), 1, 1, 1,
            {PtrArg(ctrl), PtrArg(s.argmax_part.ptr())});
  profile_wait(cx, Phase::kHead);
}

// --- the launch arithmetic, derived from the walk above ---------------------
//
// Per GDN layer, L0 launches: 2 (res_norm) + 1 (dequant qkv||z) + 1 (a||b)
//   + 10 (gdn_chunk) + 1 (dequant out_proj) + 2 (res_norm) + 1 (dequant
//   gate||up) + 1 (silu_mul) + 1 (dequant down) = 20.
// Per FA layer, L0 launches: 2 + 1 (dequant qkv) + 1 (attn_prep)
//   + 4 (one softmax per kv group) + 1 (attn_gate) + 1 (dequant o_proj) + 2
//   + 1 (dequant gate||up) + 1 (silu_mul) + 1 (dequant down) = 15.
// Boundary: 1 (embed).  Chunk total: 1 + 48.20 + 16.15 = 1201.
// `step_head` adds 5 once per prefill: 2 (final norm) + 1 (lm_head) + 2 (argmax).
//
// **None of these depends on C**, which is the whole point of the runtime-M
// rule: one binary set serves every `--pp-chunk`.
namespace {
// sycl-tla, spec 2's arithmetic (unchanged): per GDN layer 20 L0 launches, 4 SYCL GEMMs,
// 8 waits; per FA layer 15 launches, 4 + 2 x 4 GEMMs, 17 waits; 1 embed.
constexpr size_t kGdnLayerLaunches = 2 + 1 + 1 + kGdnChunkLaunches + 1 + 2 + 1 + 1 + 1;
constexpr size_t kFaLayerLaunches =
    2 + 1 + kAttnPrepLaunches + kAttnChunkLaunches + kAttnGateLaunches + 1 + 2 + 1 + 1 + 1;
constexpr size_t kGdnLayerGemms = 4;
constexpr size_t kFaLayerGemms = 4 + 2 * attn::kKvHeads;
constexpr size_t kGdnLayerWaits = 2 * 4;
constexpr size_t kFaLayerWaits = 2 * 4 + 2 * attn::kKvHeads + 1;
// L0 (spec 2.1 S2): the four dequant launches of a layer become 2 x (N / 1024) launches per
// linear -- GDN: qkv||z 32 + out_proj 10 + gate||up 68 + down 10 = 120; FA: q||k||v 28 +
// o_proj 10 + 68 + 10 = 116.
// L0 (spec 2.1 S3): attention's two GEMMs per kv group are L0 launches now, no SYCL GEMM and
// no host wait remains -- 1 + 48 x 136 + 16 x 135 = 8689 (spec §3.3, derived).
constexpr size_t kL0GdnLayerLaunches = kGdnLayerLaunches - 4 + 120;                    // 136
constexpr size_t kL0FaLayerLaunches = kFaLayerLaunches - 4 + 116 + 2 * attn::kKvHeads; // 135
}  // namespace
size_t step_chunk_launches(PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? 1 + 48 * kGdnLayerLaunches + 16 * kFaLayerLaunches
                                      : 1 + 48 * kL0GdnLayerLaunches + 16 * kL0FaLayerLaunches;
}
size_t step_chunk_gemms(PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? 48 * kGdnLayerGemms + 16 * kFaLayerGemms : 0;
}
size_t step_chunk_waits(PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? 48 * kGdnLayerWaits + 16 * kFaLayerWaits : 0;
}

}  // namespace runtime::prefill
