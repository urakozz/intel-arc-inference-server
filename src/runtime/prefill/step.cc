#include "runtime/prefill/step.h"

#include <cstdlib>
#include <cstring>
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
#include "runtime/prefill/int8.h"
#include "runtime/prefill/linear_l0.h"
#include "runtime/prefill/moe.h"
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

// **The pre-S2a fallback, opt-in through the environment.**
// `B70_PREFILL_SILU_FUSED=0` makes the L0 backend run the old pair -- gate||up
// into the fp32 [C][34816] `partials`, then `pf_silu_mul` -- instead of the
// fused epilogue. It exists so that "before and after" is ONE binary and one
// session rather than two builds, which is the role `B70_PREFILL_GDN_SCAN`
// played for the scan experiment, and so the ABBA that prices this stage can be
// taken without rebuilding between arms.
//
// The fused path is the DEFAULT and is what every gate runs. The two are
// bitwise identical -- `pf_gemm_test`'s silu case compares them directly, on
// device, at M = 2048 and M = 772 -- so this selector chooses a cost, never a
// value. `step_chunk_launches` reads it too, so the launch arithmetic always
// describes the walk that will actually run; `prefill_smoke_test`'s pinned
// constants are the default path's and will fail loudly under `=0`, which is
// the correct behaviour for a knob that changes the list.
bool silu_fused() {
  static const bool on = [] {
    const char* v = std::getenv("B70_PREFILL_SILU_FUSED");
    return v == nullptr || std::strcmp(v, "0") != 0;
  }();
  return on;
}

uint8_t* at(const l0::Mem& m, size_t off) {
  return static_cast<uint8_t*>(m.ptr()) + off;
}
const void* at_const(const l0::Mem& m, size_t off) {
  return static_cast<const uint8_t*>(m.ptr()) + off;
}

// One int4 linear on the prefill path -- THE seam of spec 2.1 §3.2. The sycl-tla side is
// spec 2's two-pass path unchanged (backend_sycl.cc); the L0 side is S2's slab walk; the
// l0-int8 side is spec 5's h8 walk (quantiser, then requant + i8 GEMM per slab).
void pf_linear(Context& cx, KernelCache& kc, PrefillScratch& s, const DeviceWeight& w,
               const uint16_t* x, uint32_t M, PrefillBackend backend, Int8State* q) {
  if (backend == PrefillBackend::SyclTla) {
    linear_sycl(cx, kc, s, w, x, M);
    return;
  }
  if (backend == PrefillBackend::L0Int8) {
    linear_i8(cx, kc, s, *q, w, x, M);
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
  cx.launch(kc(kernels::pf_res_fold_variant(s.desc().hidden, s_prev, g), "pf_res_fold"), g, M, 1,
            {PtrArg(partials), PtrArg(resid), PtrArg(s.norm_sumsq.ptr()), arg_val(M)});
  cx.launch(kc(kernels::pf_norm_finish_variant(s.desc().hidden, g, g), "pf_norm_finish"), g, M, 1,
            {PtrArg(s.norm_sumsq.ptr()), PtrArg(resid), PtrArg(norm_w), PtrArg(x), arg_val(M)});
  profile_wait(cx, Phase::kNorm);
}

constexpr uint32_t kSiluChunk = 4096;

}  // namespace

void step_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, const LoadedModel& m,
                uint32_t max_len, void* ctrl, uint32_t pos, uint32_t C, l0::Mem& gdn_state_mem,
                l0::Mem& conv_ring_mem, l0::Mem& kv_k_mem, l0::Mem& kv_v_mem,
                const KvLayout& kv, PrefillBackend backend, Int8State* q) {
  require((q != nullptr) == (backend == PrefillBackend::L0Int8),
          "the int8 state must be given iff the backend is l0-int8");
  require(C > 0 && C <= PrefillScratch::kC,
          "C = " + std::to_string(C) + " is outside (0, PrefillScratch::kC]");
  require(size_t(pos) + C <= size_t(max_len), "pos + C exceeds max_len");

  // The per-layer slice strides, re-derived from the descriptor here rather
  // than shared with buffers.cc -- capture.cc's arrangement, and for its
  // reason: a divergence is then a throw from the four `require`s below, not a
  // wrong KV slot at position 3000. The layer COUNTS and (spec 15b) the head
  // counts and widths are the descriptor's.
  const model::ModelDesc& d = *m.desc;
  require(&s.desc() == &d, "the prefill scratch was sized for another model");
  // One layer's rows: bf16, or int8 with the scales after every layer's rows (spec 12b).
  const bool kv8 = kv.form == KvCache::Int8;
  const size_t kv_stride = size_t(max_len) * d.fa_kv_heads * Qwen35::kFaHeadDim * (kv8 ? 1 : 2);
  const size_t gdn_state_stride =
      size_t(d.gdn_v_heads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim * 4;
  // The conv channels (Qwen3.8: 10240 = (2 x 16 k-heads + 48 v-heads) x 128).
  const size_t conv_ring_stride = size_t(PersistentBuffers::kConvRing) * d.gdn_conv_dim() * 2;
  const std::string fa_n = std::to_string(d.fa_layers), gdn_n = std::to_string(d.gdn_layers);
  require(kv.max_len == max_len && kv.layers == d.fa_layers && kv.layer_rows() == kv_stride,
          "the KV layout is not " + fa_n + " FA layers at this max_len");
  require(kv.bytes() == kv_k_mem.size(), "kv_k stride x " + fa_n + " FA layers != its allocation");
  require(kv.bytes() == kv_v_mem.size(), "kv_v stride x " + fa_n + " FA layers != its allocation");
  if (kv8) require_kv8_path(backend);
  // Spec 15d: a MoE model's FFN is the grouped-expert block (moe.cc), L0 backends only.
  if (d.is_moe())
    require(is_l0(backend) && m.moe.size() == d.layers,
            "a mixture-of-experts model prefills on the L0 backends (l0, l0-int8) with every "
            "layer's expert blocks loaded");
  require(gdn_state_stride * d.gdn_layers == gdn_state_mem.size(),
          "gdn_state stride x " + gdn_n + " GDN layers != its allocation");
  require(conv_ring_stride * d.gdn_layers == conv_ring_mem.size(),
          "conv_ring stride x " + gdn_n + " GDN layers != its allocation");

  // embed: ids -> resid, bf16 [C][hidden].
  cx.launch(kc(kernels::pf_embed_gather_variant(s.desc().hidden), "pf_embed_gather"), 1, C, 1,
            {PtrArg(s.ids.ptr()), PtrArg(m.embed.ptr()), PtrArg(s.resid.ptr()), arg_val(C)});

  const bool l0 = is_l0(backend);
  uint32_t gdn = 0, fa = 0;
  for (const model::LayerDesc& L : d.layer_descs()) {
    const uint32_t l = L.index;
    // S_PREV: nothing to fold before layer 0, and nothing after a MoE layer either - its
    // combine folded the block's output into resid itself (ModelDesc::ffn_fold_s() == 0).
    pf_res_norm(cx, kc, s, l == 0 || d.ffn_fold_s() == 0 ? 0u : 1u,
                at_const(m.layer_small[l].norms, s.desc().small_layout().norms_off_input), s.partials.ptr(),
                s.resid.ptr(), s.x.ptr(), C);                     // x stride hidden

    if (L.kind == model::LayerKind::GDN) {
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::QkvZ}), s.x.as<uint16_t>(), C, backend, q);
      {   // a||b: a bf16 GEMV, not a linear -- 128 columns is no DPAS shape. The bf16
          // weight: the checkpoint's own, or (spec 15 §13) an int4 a||b's copy the loader
          // dequantised with the prefill dequant's arithmetic (loader::prefill_ab).
        const DeviceWeight& w = loader::prefill_ab(m, l);
        cx.launch(kc(kernels::pf_ab_proj_variant(d.hidden), "pf_ab_proj"), w.shape.N / 16, (C + 7) / 8, 1,
                  {PtrArg(w.mem.ptr()), PtrArg(s.x.ptr()), PtrArg(s.ab_out.ptr()), arg_val(C)});
        profile_wait(cx, Phase::kAbGdn);
      }
      gdn_chunk(cx, kc, s, pos, C, s.partials.as<float>(), s.ab_out.as<float>(),
                reinterpret_cast<float*>(at(gdn_state_mem, size_t(gdn) * gdn_state_stride)),
                reinterpret_cast<uint16_t*>(at(conv_ring_mem, size_t(gdn) * conv_ring_stride)),
                m.layer_small[l].gdn.ptr(), s.mixer_out.as<uint16_t>());   // y stride v-heads x 128
      ++gdn;
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::OutProj}), s.mixer_out.as<uint16_t>(), C,
               backend, q);
    } else {
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::Qkv}), s.x.as<uint16_t>(), C, backend, q);
      const KvLayer lay = kv.layer(kv_k_mem.ptr(), kv_v_mem.ptr(), fa);
      uint16_t* kk = static_cast<uint16_t*>(lay.k);
      uint16_t* vv = static_cast<uint16_t*>(lay.v);
      // `Control::{pos, n_active}` were set by the caller before this chunk and
      // are what `pf_attn_prep` reads; the whole chunk is one attention pass, so
      // nothing here re-writes them (plan 6b's `kAttnC` sub-chunk loop was
      // ruling R5's, and ruling A14 retired it with the M = 64 route).
      if (kv8) {   // spec 12b: the same three launches over the int8 cache
        attn_prep_chunk_kv8(cx, kc, s, C, ctrl, s.partials.as<float>(),
                            m.layer_small[l].gdn.as<float>(), m.rope.as<float>(), lay);
        profile_wait(cx, Phase::kAttnPrep);
        attn_chunk_kv8(cx, kc, s, pos, C, s.pf_q.as<uint16_t>(), lay, backend);
        attn_gate_chunk_kv8(cx, kc, s, C, attn_rows(C, backend), s.partials.as<float>(),
                            s.mixer_out.as<uint16_t>());
      } else {
        attn_prep_chunk(cx, kc, s, C, ctrl, s.partials.as<float>(),
                        m.layer_small[l].gdn.as<float>(), m.rope.as<float>(), kk, vv);
        profile_wait(cx, Phase::kAttnPrep);
        attn_chunk(cx, kc, s, pos, C, s.pf_q.as<uint16_t>(), kk, vv, backend);
        attn_gate_chunk(cx, kc, s, C, attn_rows(C, backend), s.partials.as<float>(),
                        s.mixer_out.as<uint16_t>());
      }
      profile_wait(cx, Phase::kAttnGate);
      ++fa;
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::OProj}), s.mixer_out.as<uint16_t>(), C,
               backend, q);
    }

    // Spec 15d: a MoE model's FFN - the post-attention norm into x (pitch hidden), then
    // the routed experts and the shared expert through the grouped GEMMs, folded into
    // resid by the combine (runtime/prefill/moe.cc).
    if (d.is_moe()) {
      pf_res_norm(cx, kc, s, 1u, at_const(m.layer_small[l].norms, s.desc().small_layout().norms_off_post),
                  s.partials.ptr(), s.resid.ptr(), s.x.ptr(), C);
      moe_chunk(cx, kc, s, m.moe[l], l, C, backend, q);
      continue;
    }

    // The MLP half, identical in both layer kinds.
    //
    // **Parity program S2(a), L0 only: the normed activations go to
    // `mixer_out`, not `x`.** The fused gate||up GEMM reads its A operand row m
    // at pitch 5120 and writes x row m at pitch I (17408 / 19456) in the SAME kernel, so the
    // two cannot share a buffer: work-group 0's x rows would overwrite A rows
    // the later work-groups have not read yet. (The unfused pair could share it,
    // because the GEMM had fully retired before `pf_silu_mul` ran.) `mixer_out`
    // is bf16 [kC][6144] and its last reader -- out_proj / o_proj -- is three
    // launches back, so it is free and large enough for [pad256(C)][5120].
    // spec 5: the int8 gate||up exists only in the fused form, whatever silu_fused() says.
    const bool fuse = l0 && (backend == PrefillBackend::L0Int8 || silu_fused());
    void* const mlp_x = fuse ? s.mixer_out.ptr() : s.x.ptr();
    pf_res_norm(cx, kc, s, 1u, at_const(m.layer_small[l].norms, s.desc().small_layout().norms_off_post),
                s.partials.ptr(), s.resid.ptr(), mlp_x, C);   // stride hidden
    if (fuse) {
      // One launch pair per slab and NO `pf_silu_mul`: the epilogue writes x
      // directly, so the fp32 [C][2 x I] `partials` rectangle -- the largest on
      // the walk -- is neither written nor read back (spec §3, S2).
      if (backend == PrefillBackend::L0Int8)
        linear_i8_silu(cx, kc, s, *q, m.linears.at({l, LinearId::GateUp}),
                       static_cast<const uint16_t*>(mlp_x), C, s.x.as<uint16_t>(),
                       d.intermediate);
      else
        linear_l0_silu(cx, kc, s, m.linears.at({l, LinearId::GateUp}),
                       static_cast<const uint16_t*>(mlp_x), C, s.x.as<uint16_t>(),
                       d.intermediate);
    } else {
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::GateUp}), s.x.as<uint16_t>(), C, backend, q);
      cx.launch(kc(kernels::pf_silu_mul_variant(d.intermediate), "pf_silu_mul"),
                (d.intermediate + kSiluChunk - 1) / kSiluChunk, C, 1,
                {PtrArg(s.partials.ptr()), PtrArg(s.x.ptr()), arg_val(C)});   // x stride I
      profile_wait(cx, Phase::kSilu);
    }
    pf_linear(cx, kc, s, m.linears.at({l, LinearId::Down}), s.x.as<uint16_t>(), C, backend, q);
  }
  require(gdn == d.gdn_layers && fa == d.fa_layers,
          "the layer table did not give " + gdn_n + " GDN and " + fa_n + " FA layers");
}

void step_head(Context& cx, KernelCache& kc, PrefillScratch& s, const LoadedModel& m, void* ctrl,
               uint32_t last_row, const void* normed_row) {
  // The final norm, ONE row. `sumsq[g * 1 + 0] == sumsq[g]`, which is exactly
  // what the M = 1 pair reads. (Spec 8: skipped when step_mtp_kv already wrote it.)
  if (!normed_row)   // S_PREV 0 after a MoE model's last layer (its combine folded)
    pf_res_norm(cx, kc, s, m.desc->ffn_fold_s() == 0 ? 0u : 1u, m.final_norm.ptr(),
                at(s.partials, size_t(last_row) * s.desc().hidden * 4),
                at(s.resid, size_t(last_row) * s.desc().hidden * 2),
                at(s.x, size_t(last_row) * s.desc().hidden * 2), 1u);

  // lm_head at M = 1 through the EXISTING decode binary, chosen off the loaded
  // weight's kind exactly as capture.cc:625-631 chooses it. It writes straight
  // into `logits`: `gemv.cl`'s `out[(s*M + m)*N + n]` at S = 1, M = 1 IS the
  // [1][N] row `argmax_stage1` reads. **Not `dequant_to_bf16` + `gemm_bf16`**:
  // the lm_head's [5120][248320] bf16 expansion is 2.54 GB, seven times the
  // dequant scratch, for one row of output.
  const DeviceWeight& lm = m.linears.at({loader::kTopLevel, LinearId::LmHead});
  const void* xrow = normed_row ? normed_row : at(s.x, size_t(last_row) * s.desc().hidden * 2);
  if (lm.kind == model::WeightKind::Int4) {
    cx.launch(kc(kernels::gemv_variant(1, lm.shape.K, lm.shape.N, lm.shape.S, lm.shape.layout),
                 "gemv"),
              lm.shape.N / 64, lm.shape.S, 1,
              {PtrArg(lm.mem.ptr()),
               PtrArg(lm.shape.layout == 0 ? lm.scales->ptr() : lm.mem.ptr()), PtrArg(xrow),
               PtrArg(s.logits.ptr())});
  } else if (lm.kind == model::WeightKind::Int8) {
    // Spec 9 §3: the int8 head, so the first generated token uses the same form as
    // every later one.
    cx.launch(kc(kernels::gemv_i8w_variant(1, lm.shape.K, lm.shape.N), "gemv_i8w"),
              lm.shape.N / kernels::kGemvI8wCols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(lm.scales->ptr()), PtrArg(xrow),
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
  cx.launch(kc(kernels::argmax_stage1_variant(1, m.desc->vocab_used), "argmax_stage1"), (Qwen35::kVocab + 1023) / 1024,
            1, 1, {PtrArg(s.logits.ptr()), PtrArg(s.argmax_part.ptr())});
  cx.launch(kc(kernels::argmax_stage2_variant(), "argmax_stage2"), 1, 1, 1,
            {PtrArg(ctrl), PtrArg(s.argmax_part.ptr())});
  profile_wait(cx, Phase::kHead);
}

// --- spec 8: the MTP head's K/V during prefill (plan 8b Task 3) --------------

namespace {
// One bf16 linear (the MTP head's; gemv_bf16's tiled layout) on pf_gemm: for each
// 1024-column slab in [n_begin, n_end), pf_bf16_slab untiles it into the slab buffer and
// pf_gemm multiplies x [M][K] (pitch K) by it into out + n0 (pitch ldc). linear_l0's
// walk with the dequant replaced by a copy.
void linear_bf16(Context& cx, KernelCache& kc, PrefillScratch& s, const DeviceWeight& w,
                 const uint16_t* x, uint32_t M, uint32_t n_begin, uint32_t n_end, float* out,
                 uint32_t ldc) {
  const model::GemvShape& sh = w.shape;
  constexpr uint32_t kNs = kernels::kPfSlabWidth;
  require(w.kind == model::WeightKind::Bf16, "linear_bf16 on an int4 weight");
  require(n_begin % kNs == 0 && n_end % kNs == 0 && n_end <= sh.N, "slab range is not whole slabs");
  require(M > 0 && pad256(M) <= PrefillScratch::kC, "M padded to 256 exceeds kC");
  l0::Mem& slab = s.slab_buffer();
  require(slab.size() >= size_t(sh.K) * kNs * 2, "the slab buffer is smaller than [K][1024]");
  l0::Kernel& un = kc(kernels::pf_bf16_slab_variant(sh.K), "pf_bf16_slab");
  for (uint32_t n0 = n_begin; n0 < n_end; n0 += kNs) {
    cx.launch(un, kNs / 16, sh.K / 8, 1, {PtrArg(w.mem.ptr()), PtrArg(slab.ptr()), arg_val(n0)});
    const GemmBatch b{M, sh.K, kNs, 1, sh.K, kNs, ldc, 0, 0, 0};
    gemm_l0(cx, kc, b, x, slab.as<uint16_t>(), out + n0, /*transB=*/false);
  }
}
}  // namespace

void step_mtp_kv(Context& cx, KernelCache& kc, PrefillScratch& s, const LoadedModel& m,
                 void* hctl, uint32_t pos, uint32_t C, uint16_t* hid, const KvLayer& kv) {
  require(m.mtp != nullptr, "step_mtp_kv without the MTP head");
  // Spec 15e: a MoE head (Ornith's) fills its KV through this same front - fc, the input
  // norm, q||k||v, attn_prep; its MoE FFN never runs in prefill. The only MoE term is the
  // final norm's fold below (nothing to fold after a MoE block, as step_head's).
  require(C > 0 && C <= PrefillScratch::kC, "C is outside (0, kC]");
  const loader::MtpHead& h = *m.mtp;
  const uint32_t H = m.desc->hidden, G = PrefillScratch::kNormGroups;
  // q||gate is [0, q_proj) and k||v [q_proj, qkv) (Qwen3.8 12288 / 14336); the
  // slab walk below starts at q_proj, a whole number of 1024-column slabs.
  const uint32_t kv_n0 = m.desc->fa_q_proj_n();
  require(kv_n0 % kernels::kPfSlabWidth == 0, "q_proj is not a whole number of slabs");
  // 1. The main model's final norm on every row (what decode's b.x holds after a
  //    step), into hid rows 1..C. step_head's own single-row norm is then skipped.
  pf_res_norm(cx, kc, s, m.desc->ffn_fold_s() == 0 ? 0u : 1u, m.final_norm.ptr(),
              s.partials.ptr(), s.resid.ptr(), hid + H, C);
  const uint32_t rows = mtp_kv_rows(pos, C);
  if (rows == 0) return;
  const uint32_t r0 = C - rows;   // 1 at pos 0 (no h_{-1}), else 0
  // 2. embed(ids[r0 + r]) -> resid rows (the main residual is dead after step 1).
  cx.launch(kc(kernels::pf_embed_gather_variant(s.desc().hidden), "pf_embed_gather"), 1, rows, 1,
            {PtrArg(s.ids.as<uint32_t>() + r0), PtrArg(m.embed.ptr()), PtrArg(s.resid.ptr()),
             arg_val(rows)});
  // 3. The two pre-fc norms into one [rows][2 x hidden] row of x: embed first, hidden second.
  const std::string fold0 = kernels::pf_res_fold_variant(H, 0, G);
  const std::string cat = kernels::pf_norm_finish_strided_variant(H, G, G, 2 * H);
  auto fold_norm = [&](const std::string& fv, const std::string& nv, const void* partials,
                       void* resid, const void* w, void* out) {
    cx.launch(kc(fv, "pf_res_fold"), G, rows, 1,
              {PtrArg(partials), PtrArg(resid), PtrArg(s.norm_sumsq.ptr()), arg_val(rows)});
    cx.launch(kc(nv, "pf_norm_finish"), G, rows, 1,
              {PtrArg(s.norm_sumsq.ptr()), PtrArg(resid), PtrArg(w), PtrArg(out), arg_val(rows)});
  };
  fold_norm(fold0, cat, s.partials.ptr(), s.resid.ptr(), at_const(h.norms, loader::mtp_norm_off(*m.desc, loader::kMtpNormPreE)),
            s.x.ptr());
  fold_norm(fold0, cat, s.partials.ptr(), hid + size_t(r0) * H,
            at_const(h.norms, loader::mtp_norm_off(*m.desc, loader::kMtpNormPreH)), s.x.as<uint16_t>() + H);
  // 4. fc -> partials [rows][hidden]; the head's residual starts at fc's output
  //    (ZERO_RESID), input_layernorm -> x (pitch hidden).
  linear_bf16(cx, kc, s, h.fc, s.x.as<uint16_t>(), rows, 0, H, s.partials.as<float>(), H);
  fold_norm(kernels::pf_res_fold_zero_variant(H, G), kernels::pf_norm_finish_variant(H, G, G),
            s.partials.ptr(), s.resid.ptr(), at_const(h.norms, loader::mtp_norm_off(*m.desc, loader::kMtpNormInput)), s.x.ptr());
  // 5. Only the k||v columns of q||k||v, then attn_prep writes K/V at hctl.pos + r. Its q
  //    rows read unwritten partials columns into pf_q, which nothing reads.
  linear_bf16(cx, kc, s, h.qkv, s.x.as<uint16_t>(), rows, kv_n0, h.qkv.shape.N,
              s.partials.as<float>(), h.qkv.shape.N);
  if (kv.int8())   // spec 12b: the head's KV in the main cache's form
    attn_prep_chunk_kv8(cx, kc, s, rows, hctl, s.partials.as<float>(), h.fa.as<float>(),
                        m.rope.as<float>(), kv);
  else
    attn_prep_chunk(cx, kc, s, rows, hctl, s.partials.as<float>(), h.fa.as<float>(),
                    m.rope.as<float>(), static_cast<uint16_t*>(kv.k), static_cast<uint16_t*>(kv.v));
  profile_wait(cx, Phase::kAttnPrep);
}

size_t step_mtp_kv_launches(uint32_t pos, uint32_t C) {
  if (mtp_kv_rows(pos, C) == 0) return 2;
  // final norm 2 + embed 1 + 2 x 2 pre-norms + fc (2 x 5) + fold/norm 2 + k||v (2 x 2)
  // + attn_prep 1
  return 2 + 1 + 4 + 10 + 2 + 4 + kAttnPrepLaunches;
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
// **Only one term depends on C**, and it is new in spec S3: the L0 backend's
// attention issues its QK^T GEMM per row block of 256 rows rather than once per
// chunk, so an FA layer costs `attn_chunk_launches(C, L0)` there instead of a
// constant. Everything else is still C-free -- the runtime-M rule, one binary
// set for every `--prefill-chunk` -- and sycl-tla's arithmetic is untouched.
namespace {
// sycl-tla, spec 2's arithmetic (unchanged): per GDN layer 20 L0 launches, 4 SYCL GEMMs,
// 8 waits; per FA layer 15 launches, 4 + 2 x 4 GEMMs, 17 waits; 1 embed.
constexpr size_t kGdnLayerLaunches = 2 + 1 + 1 + kGdnChunkLaunches + 1 + 2 + 1 + 1 + 1;
// Spec 15b: the FA terms are per kv-head (4 on Qwen3.8), so per descriptor.
size_t fa_layer_launches(const model::ModelDesc& d) {
  return 2 + 1 + kAttnPrepLaunches + attn_chunk_launches_sycl(d) + kAttnGateLaunches + 1 + 2 +
         1 + 1 + 1;
}
constexpr size_t kGdnLayerGemms = 4;
size_t fa_layer_gemms(const model::ModelDesc& d) { return 4 + 2 * size_t(d.fa_kv_heads); }
constexpr size_t kGdnLayerWaits = 2 * 4;
size_t fa_layer_waits(const model::ModelDesc& d) { return 2 * 4 + 2 * size_t(d.fa_kv_heads) + 1; }
// L0 (spec 2.1 S2): the four dequant launches of a layer become 2 x (N / 1024) launches per
// linear -- GDN: qkv||z 32 + out_proj 10 + gate||up 68 + down 10 = 120; FA: q||k||v 28 +
// o_proj 10 + 68 + 10 = 116 (Qwen3.8; spec 14: gate||up is 2 x 38 = 76 on Agnes, so 128 /
// 124 - `slab_launches` reads the descriptor's rows).
// L0 (spec 2.1 S3): attention's two GEMMs per kv group are L0 launches now, no SYCL GEMM and
// no host wait remains -- 1 + 48 x 136 + 16 x 135 = 8689 at one row block (spec §3.3).
// L0 (parity program S3): QK^T is issued per row block of 256 rows (P·V is not -- see
// attn.cc), so an FA layer is 131 + 4 x blocks launches -- 135 at C <= 256, 163 at
// C = 2048. A chunk is then 1 + 48 x 136 + 16 x (131 + 4 x blocks): 8689 at C <= 256 and
// 9137 at C = 2048 (derived).
// L0 (parity program S2a): `pf_silu_mul` is fused into the gate||up slab GEMM's epilogue,
// so EVERY layer loses exactly one launch -- 135 per GDN layer and 130 + 4 x blocks per FA
// layer. A chunk is 8625 at C <= 256 and 9073 at C = 2048 (derived; -64 = -1 per layer).
// sycl-tla keeps its own `pf_silu_mul` launch and its arithmetic is untouched.
// L0 (spec 6, plan 6b): in Flash mode -- the default, `attn_mode()` -- attention is ONE
// `pf_flash_attn` launch per FA layer in place of 4 x (2 + blocks), so an FA layer is
// 123 launches whatever C is, and a chunk is 1 + 48 x 135 + 16 x 123 = 8449 at every C
// (derived; l0-int8 8705). `B70_PREFILL_ATTN=composed` restores the counts above.
// `silu_fused()` is the one term that is not a property of the shape: it is a
// diagnostic selector, and the arithmetic follows the walk rather than the
// default so that a `=0` session's counter still matches its own prediction.
// The 2 x N / 1024 slab launches of one layer kind's four int4 linears.
size_t slab_launches(const model::ModelDesc& d, model::LayerKind kind) {
  size_t n = 0;
  for (LinearId id : Qwen35::linear_order(kind))
    if (d.linear(id).kind == model::WeightKind::Int4) n += 2 * (d.shape(id).N / 1024);
  return n;
}
size_t l0_gdn_layer_launches(const model::ModelDesc& d) {
  return kGdnLayerLaunches - 4 + slab_launches(d, model::LayerKind::GDN) -
         (silu_fused() ? 1 : 0);
}
size_t l0_fa_layer_launches(const model::ModelDesc& d, uint32_t C) {
  return fa_layer_launches(d) - 4 + slab_launches(d, model::LayerKind::FA) -
         attn_chunk_launches_sycl(d) + attn_chunk_launches(d, C, PrefillBackend::L0) -
         (silu_fused() ? 1 : 0);
}
}  // namespace
// Spec 15d, a MoE model on an L0 backend. Per layer: the input norm 2, the mixer's two
// int4 linears (2 x N / 1024 slab launches each, plus the quantiser per linear on
// l0-int8), the mixer itself (GDN: a||b + gdn_chunk; FA: attn_prep + attention + gate),
// the post-attention norm 2 and moe_chunk_launches. Ornith at C <= 2048, flash:
// GDN 2 + 24 + 1 + 10 + 4 + 2 + 11 = 54, FA 2 + 18 + 1 + 1 + 1 + 4 + 2 + 11 = 40 on l0,
// so 1 + 30 x 54 + 10 x 40 = 2021; l0-int8 2 more per layer (the quantisers) and 1 fewer
// (the MoE block's 10): 1 + 30 x 55 + 10 x 41 = 2061 (derived).
namespace {
size_t moe_step_chunk_launches(const model::ModelDesc& d, PrefillBackend b, uint32_t C) {
  auto lin = [&](LinearId id) {
    return 2 * size_t(d.shape(id).N / 1024) + (b == PrefillBackend::L0Int8 ? 1 : 0);
  };
  const size_t ffn = 2 + moe_chunk_launches(d, b);
  const size_t gdn = 2 + lin(LinearId::QkvZ) + 1 + kGdnChunkLaunches + lin(LinearId::OutProj) + ffn;
  const size_t fa = 2 + lin(LinearId::Qkv) + kAttnPrepLaunches + attn_chunk_launches(d, C, b) +
                    kAttnGateLaunches + lin(LinearId::OProj) + ffn;
  return 1 + d.gdn_layers * gdn + d.fa_layers * fa;
}
}  // namespace

size_t step_chunk_launches(const model::ModelDesc& d, PrefillBackend b, uint32_t C) {
  if (d.is_moe()) return is_l0(b) ? moe_step_chunk_launches(d, b, C) : 0;
  if (b == PrefillBackend::SyclTla)
    return 1 + d.gdn_layers * kGdnLayerLaunches + d.fa_layers * fa_layer_launches(d);
  const size_t base =
      1 + d.gdn_layers * l0_gdn_layer_launches(d) + d.fa_layers * l0_fa_layer_launches(d, C);
  if (b != PrefillBackend::L0Int8) return base;
  // spec 5: +1 launch (the activation quantiser, pf_quant_had) per int4 linear, 4 per
  // layer, the slab walk being the same 2 x N/1024 launches. The int8 gate||up is ALWAYS
  // the fused SiLU form, while `base` follows silu_fused(): an unfused session's base
  // counts one pf_silu_mul per layer the int8 walk never issues.
  return base + size_t(d.layers) * 4 - (silu_fused() ? 0 : d.layers);
}
size_t step_chunk_gemms(const model::ModelDesc& d, PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? d.gdn_layers * kGdnLayerGemms + d.fa_layers * fa_layer_gemms(d)
                                      : 0;
}
size_t step_chunk_waits(const model::ModelDesc& d, PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? d.gdn_layers * kGdnLayerWaits + d.fa_layers * fa_layer_waits(d)
                                      : 0;
}

}  // namespace runtime::prefill
