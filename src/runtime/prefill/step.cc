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
                PrefillBackend backend, Int8State* q) {
  require((q != nullptr) == (backend == PrefillBackend::L0Int8),
          "the int8 state must be given iff the backend is l0-int8");
  require(C > 0 && C <= PrefillScratch::kC,
          "C = " + std::to_string(C) + " is outside (0, PrefillScratch::kC]");
  require(size_t(pos) + C <= size_t(max_len), "pos + C exceeds max_len");

  // The per-layer slice strides, re-derived from `model::Qwen35` here rather
  // than shared with buffers.cc -- capture.cc:105-116's arrangement, and for
  // its reason: a divergence is then a throw from the four `require`s below,
  // not a wrong KV slot at position 3000. The layer COUNTS are the descriptor's.
  const model::ModelDesc& d = *m.desc;
  const size_t kv_stride = size_t(max_len) * Qwen35::kFaKvHeads * Qwen35::kFaHeadDim * 2;
  const size_t gdn_state_stride =
      size_t(Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim * 4;
  // 10240 = the conv channels (2 x 16 k-heads + 48 v-heads) x 128.
  const size_t conv_ring_stride = size_t(PersistentBuffers::kConvRing) * 10240 * 2;
  const std::string fa_n = std::to_string(d.fa_layers), gdn_n = std::to_string(d.gdn_layers);
  require(kv_stride * d.fa_layers == kv_k_mem.size(),
          "kv_k stride x " + fa_n + " FA layers != its allocation");
  require(kv_stride * d.fa_layers == kv_v_mem.size(),
          "kv_v stride x " + fa_n + " FA layers != its allocation");
  require(gdn_state_stride * d.gdn_layers == gdn_state_mem.size(),
          "gdn_state stride x " + gdn_n + " GDN layers != its allocation");
  require(conv_ring_stride * d.gdn_layers == conv_ring_mem.size(),
          "conv_ring stride x " + gdn_n + " GDN layers != its allocation");

  // embed: ids -> resid, bf16 [C][5120].
  cx.launch(kc(kernels::pf_embed_gather_variant(), "pf_embed_gather"), 1, C, 1,
            {PtrArg(s.ids.ptr()), PtrArg(m.embed.ptr()), PtrArg(s.resid.ptr()), arg_val(C)});

  const bool l0 = is_l0(backend);
  uint32_t gdn = 0, fa = 0;
  for (const model::LayerDesc& L : d.layer_descs()) {
    const uint32_t l = L.index;
    pf_res_norm(cx, kc, s, l == 0 ? 0u : 1u,
                at_const(m.layer_small[l].norms, loader::kNormsOffInput), s.partials.ptr(),
                s.resid.ptr(), s.x.ptr(), C);                     // x stride 5120

    if (L.kind == model::LayerKind::GDN) {
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::QkvZ}), s.x.as<uint16_t>(), C, backend, q);
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
               backend, q);
    } else {
      pf_linear(cx, kc, s, m.linears.at({l, LinearId::Qkv}), s.x.as<uint16_t>(), C, backend, q);
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
               backend, q);
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
    pf_res_norm(cx, kc, s, 1u, at_const(m.layer_small[l].norms, loader::kNormsOffPost),
                s.partials.ptr(), s.resid.ptr(), mlp_x, C);   // stride 5120
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
  if (!normed_row)
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
  const void* xrow = normed_row ? normed_row : at(s.x, size_t(last_row) * Qwen35::kHidden * 2);
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
  cx.launch(kc(kernels::argmax_stage1_variant(1), "argmax_stage1"), (Qwen35::kVocab + 1023) / 1024,
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
constexpr uint32_t kMtpKvN0 = 12288;   // q||gate is [0, 12288); k||v is [12288, 14336)
}  // namespace

void step_mtp_kv(Context& cx, KernelCache& kc, PrefillScratch& s, const LoadedModel& m,
                 void* hctl, uint32_t pos, uint32_t C, uint16_t* hid, uint16_t* kv_k,
                 uint16_t* kv_v) {
  require(m.mtp != nullptr, "step_mtp_kv without the MTP head");
  require(C > 0 && C <= PrefillScratch::kC, "C is outside (0, kC]");
  const loader::MtpHead& h = *m.mtp;
  const uint32_t H = Qwen35::kHidden, G = PrefillScratch::kNormGroups;
  // 1. The main model's final norm on every row (what decode's b.x holds after a
  //    step), into hid rows 1..C. step_head's own single-row norm is then skipped.
  pf_res_norm(cx, kc, s, 1u, m.final_norm.ptr(), s.partials.ptr(), s.resid.ptr(),
              hid + H, C);
  const uint32_t rows = mtp_kv_rows(pos, C);
  if (rows == 0) return;
  const uint32_t r0 = C - rows;   // 1 at pos 0 (no h_{-1}), else 0
  // 2. embed(ids[r0 + r]) -> resid rows (the main residual is dead after step 1).
  cx.launch(kc(kernels::pf_embed_gather_variant(), "pf_embed_gather"), 1, rows, 1,
            {PtrArg(s.ids.as<uint32_t>() + r0), PtrArg(m.embed.ptr()), PtrArg(s.resid.ptr()),
             arg_val(rows)});
  // 3. The two pre-fc norms into one [rows][10240] row of x: embed first, hidden second.
  const std::string fold0 = kernels::pf_res_fold_variant(H, 0, G);
  const std::string cat = kernels::pf_norm_finish_strided_variant(H, G, G, 2 * H);
  auto fold_norm = [&](const std::string& fv, const std::string& nv, const void* partials,
                       void* resid, const void* w, void* out) {
    cx.launch(kc(fv, "pf_res_fold"), G, rows, 1,
              {PtrArg(partials), PtrArg(resid), PtrArg(s.norm_sumsq.ptr()), arg_val(rows)});
    cx.launch(kc(nv, "pf_norm_finish"), G, rows, 1,
              {PtrArg(s.norm_sumsq.ptr()), PtrArg(resid), PtrArg(w), PtrArg(out), arg_val(rows)});
  };
  fold_norm(fold0, cat, s.partials.ptr(), s.resid.ptr(), at_const(h.norms, loader::kMtpNormPreE),
            s.x.ptr());
  fold_norm(fold0, cat, s.partials.ptr(), hid + size_t(r0) * H,
            at_const(h.norms, loader::kMtpNormPreH), s.x.as<uint16_t>() + H);
  // 4. fc -> partials [rows][5120]; the head's residual starts at fc's output (ZERO_RESID),
  //    input_layernorm -> x (pitch 5120).
  linear_bf16(cx, kc, s, h.fc, s.x.as<uint16_t>(), rows, 0, H, s.partials.as<float>(), H);
  fold_norm(kernels::pf_res_fold_zero_variant(H, G), kernels::pf_norm_finish_variant(H, G, G),
            s.partials.ptr(), s.resid.ptr(), at_const(h.norms, loader::kMtpNormInput), s.x.ptr());
  // 5. Only the k||v columns of q||k||v, then attn_prep writes K/V at hctl.pos + r. Its q
  //    rows read unwritten partials columns into pf_q, which nothing reads.
  linear_bf16(cx, kc, s, h.qkv, s.x.as<uint16_t>(), rows, kMtpKvN0, h.qkv.shape.N,
              s.partials.as<float>(), h.qkv.shape.N);
  attn_prep_chunk(cx, kc, s, rows, hctl, s.partials.as<float>(), h.fa.as<float>(),
                  m.rope.as<float>(), kv_k, kv_v);
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
// set for every `--pp-chunk` -- and sycl-tla's arithmetic is untouched.
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
  return kFaLayerLaunches - 4 + slab_launches(d, model::LayerKind::FA) - kAttnChunkLaunches +
         attn_chunk_launches(C, PrefillBackend::L0) - (silu_fused() ? 1 : 0);
}
}  // namespace
size_t step_chunk_launches(const model::ModelDesc& d, PrefillBackend b, uint32_t C) {
  if (b == PrefillBackend::SyclTla)
    return 1 + d.gdn_layers * kGdnLayerLaunches + d.fa_layers * kFaLayerLaunches;
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
  return b == PrefillBackend::SyclTla ? d.gdn_layers * kGdnLayerGemms + d.fa_layers * kFaLayerGemms
                                      : 0;
}
size_t step_chunk_waits(const model::ModelDesc& d, PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? d.gdn_layers * kGdnLayerWaits + d.fa_layers * kFaLayerWaits
                                      : 0;
}

}  // namespace runtime::prefill
